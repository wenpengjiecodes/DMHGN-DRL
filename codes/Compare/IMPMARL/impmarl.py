
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
import json
import random
import os
import glob
import csv
import time
import gc
from bisect import bisect_left, insort

# ========================================================================
# 1. 环境基础类 (完全复用自 V1_01.py)
# ========================================================================
class Config:
    def __init__(self, json_path=None):
        self.json_path = json_path
        if json_path:
            self.load_from_json(json_path)

    def load_from_json(self, json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        reference = data.get("reference_solution", {})
        self.ref_cmax = reference.get("Cmax", None)
        self.ref_tec = reference.get("TEC", None)
        self.ref_favg = reference.get("Favg", None)
        global_config = data.get("global_config", {})
        self.d_operation = 32
        self.d_batch = 16
        self.d_worker = 16
        self.d_global = 5 + self.d_operation + self.d_batch + self.d_worker
        self.fatigue_threshold = global_config.get("fatigue_threshold", 0.8)
        self.num_jobs = global_config.get("num_jobs", 10)
        self.num_jobs_static = global_config.get("num_jobs_static", 8)
        self.num_jobs_dynamic = global_config.get("num_jobs_dynamic", 2)
        self.num_ops_per_job = global_config.get("num_ops_per_job", 5)
        self.num_machines = global_config.get("num_machines", 10)
        self.num_workers = global_config.get("num_workers", 5)
        self.total_pieces_per_job = global_config.get("total_pieces_per_job", 10)
        self.machine_data = {}
        for mk, mi in data.get("machine_static_data", {}).items():
            self.machine_data[int(mk.split("_")[1])] = mi
        self.worker_data = {}
        for wk, wi in data.get("worker_static_data", {}).items():
            self.worker_data[int(wk.split("_")[1])] = {
                "lambda": wi.get("lambda", 0.05),
                "mu": wi.get("mu", 0.1),
            }
        self.job_data = {}
        self.job_ready_triggers = {}
        for jk, ji in data.get("job_operation_static_data", {}).items():
            jid = int(jk.split("_")[1])
            self.job_data[jid] = {}
            if "ready_trigger" in ji.get("op_0", {}):
                self.job_ready_triggers[jid] = ji["op_0"]["ready_trigger"]
            for ok, oi in ji.items():
                oid = int(ok.split("_")[1])
                if "base_process_time" in oi:
                    oi["base_process_time"] = {int(k): v for k, v in oi["base_process_time"].items()}
                if "machine_candidates" in oi:
                    oi["machine_candidates"] = [int(x) for x in oi["machine_candidates"]]
                self.job_data[jid][oid] = oi


class Operation:
    def __init__(self, job_id, op_id, machine_candidates, base_pt, total_pieces=10, is_preemptive=False):
        self.job_id = job_id
        self.op_id = op_id
        self.machine_candidates = machine_candidates
        self.base_pt = base_pt
        self.total_pieces = total_pieces
        self.is_preemptive = is_preemptive
        self.is_scheduled = False
        self.prev_op = None
        self.next_op = None
        self.complete_time = 0
        self.completed_pieces = 0
        self.predecessors = []
        self.successors = []
        self.ready_time = 0
        self.all_predecessors_completed = False

    def get_prev_op_completion_time(self):
        return self.prev_op.complete_time if self.prev_op else 0

    def are_all_predecessors_completed(self):
        for p in self.predecessors:
            if not p.is_scheduled or p.completed_pieces < p.total_pieces:
                return False
        return True

    def update_predecessor_status(self):
        self.all_predecessors_completed = self.are_all_predecessors_completed()
        if self.all_predecessors_completed and self.ready_time == 0:
            self.ready_time = max([p.complete_time for p in self.predecessors] + [0])

    def is_ready(self):
        return (
            self.all_predecessors_completed
            and self.completed_pieces < self.total_pieces
            and not self.is_scheduled
        )

    def update_completion(self, new_completed_pieces, completion_time=None):
        self.completed_pieces += new_completed_pieces
        if self.completed_pieces >= self.total_pieces:
            self.is_scheduled = True
            if completion_time:
                self.complete_time = max(self.complete_time, completion_time)
            for s in self.successors:
                s.update_predecessor_status()


class SubBatch:
    def __init__(self, op_id, batch_id, size, machine_id, worker_id):
        self.op_id = op_id
        self.batch_id = batch_id
        self.size = size
        self.machine_id = machine_id
        self.worker_id = worker_id
        self.start_time = 0
        self.complete_time = 0
        self.status = "pending"


class Worker:
    def __init__(self, worker_id, lambda_, mu):
        self.worker_id = worker_id
        self.lambda_ = lambda_
        self.mu = mu
        self.current_fatigue = 0.0
        self.total_work_time = 0.0
        self.is_busy = False
        self.is_resting = False
        self.current_machine = None
        self.current_batch = None
        self.assigned_batches = []
        self.next_available_time = 0.0
        self.config = None

    def update_fatigue_after_work(self, actual_time, start_time):
        fatigue_increase = (1 - self.current_fatigue) * (1 - np.exp(-self.lambda_ * actual_time))
        self.current_fatigue = min(1.0, self.current_fatigue + fatigue_increase)
        if self.current_fatigue >= self.config.fatigue_threshold:
            self.is_resting = False
            recovery_time = -np.log(0.5 / self.current_fatigue) / self.mu
            self.next_available_time = recovery_time + start_time + actual_time
            self.current_fatigue = 0.5
        else:
            self.next_available_time = start_time + actual_time
            self.is_resting = False

    def assign_batch(self, batch):
        self.assigned_batches.append(batch)
        self.current_batch = batch
        self.is_busy = True
        self.current_machine = batch.machine_id

    def complete_batch(self, complete_time, start_time, actual_time):
        if self.current_batch:
            self.update_fatigue_after_work(actual_time, start_time)
            self.current_batch.status = "completed"
            self.current_batch.complete_time = complete_time
            self.total_work_time += actual_time
            self.current_batch = None
            self.is_busy = False
            self.current_machine = None

    def is_available(self, check_time):
        if self.is_busy:
            return False
        if check_time < self.next_available_time:
            return False
        if check_time >= self.next_available_time:
            self.is_resting = False
        return not self.is_busy and not self.is_resting and self.current_fatigue < self.config.fatigue_threshold


class MachineState:
    def __init__(self, machine_id, config=None):
        self.machine_id = machine_id
        self.config = config
        self.is_busy = False
        self.next_available_time = 0.0
        self.current_batch = None
        self.total_processing_time = 0.0
        self.total_idle_energy = 0.0
        self.total_switch_energy = 0.0
        self.total_process_energy = 0.0
        self.is_first_use = True

    def assign_batch(self, batch, start_time, process_time):
        self.is_busy = True
        self.current_batch = batch
        self.next_available_time = start_time + process_time
        self.total_processing_time += process_time
        if self.config and self.machine_id in self.config.machine_data:
            self.total_process_energy += process_time * self.config.machine_data[self.machine_id].get("energy_process", 0)

    def complete_batch(self):
        self.is_busy = False
        self.current_batch = None

    def add_idle_energy(self, idle_time):
        if idle_time > 0 and self.config and self.machine_id in self.config.machine_data:
            self.total_idle_energy += idle_time * self.config.machine_data[self.machine_id].get("energy_idle", 0)

    def add_switch_energy(self, switch_time):
        if switch_time > 0 and self.config and self.machine_id in self.config.machine_data:
            self.total_switch_energy += switch_time * self.config.machine_data[self.machine_id].get("energy_switch", 0)

    @property
    def total_energy(self):
        return self.total_idle_energy + self.total_switch_energy + self.total_process_energy


class HeterogeneousDisjunctiveGraph:
    def __init__(self, config):
        self.config = config
        self.operation_nodes = {}
        self.worker_nodes = {}
        self.current_time = 0
        # ★ 新增：缓存字典加速查找
        self._job_ops_map = {}  # job_id -> [op_key, ...]

    def add_operation(self, op):
        op_key = "J{}_O{}".format(op.job_id, op.op_id)
        self.operation_nodes[op_key] = op
        if op.job_id not in self._job_ops_map:
            self._job_ops_map[op.job_id] = []
        self._job_ops_map[op.job_id].append(op_key)

    def add_worker(self, worker):
        self.worker_nodes[worker.worker_id] = worker

    def get_ready_operations(self):
        return [k for k, op in self.operation_nodes.items() if op.is_ready()]

    def update_operation_status(self):
        for op in self.operation_nodes.values():
            op.update_predecessor_status()

    def check_machine_availability(self, op_key, machine_states):
        op = self.operation_nodes[op_key]
        for m_id in op.machine_candidates:
            if m_id in machine_states and machine_states[m_id].is_busy:
                return False
        return True

    def get_spt_machine_order(self, op_key, machine_states):
        op = self.operation_nodes[op_key]
        times = []
        for m_id in op.machine_candidates:
            if m_id in machine_states:
                times.append((m_id, op.base_pt.get(m_id, float("inf"))))
        return [m_id for m_id, _ in sorted(times, key=lambda x: x[1])]

    def get_remaining_ops_in_job(self, job_id):
        """★ 新增：O(k) 获取某 job 未完成的工序数，k 为该 job 的工序数"""
        keys = self._job_ops_map.get(job_id, [])
        return sum(1 for k in keys if not self.operation_nodes[k].is_scheduled)

    def get_total_ops(self):
        return len(self.operation_nodes)

    def get_completed_ops(self):
        return sum(1 for op in self.operation_nodes.values() if op.is_scheduled)


# ========================================================================
# 2. IMPMARL 核心网络与算法
# ========================================================================
class LightweightStateEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.op_proj = nn.Linear(6, config.d_operation)
        self.worker_proj = nn.Linear(5, config.d_worker)

    def forward(self, hdg, machine_states, norm_params, dev):
        cmax_norm, energy_norm = norm_params
        workers = list(hdg.worker_nodes.values())

        # ★ 优化：只对 ready_ops 编码，而非全部 1200 个工序
        ops = list(hdg.operation_nodes.values())
        ready_ops = [op for op in ops if op.is_ready()]

        if ready_ops:
            op_feats = []
            for op in ready_ops:
                pt_vals = list(op.base_pt.values()) if op.base_pt else [0.0]
                rem_ops = hdg.get_remaining_ops_in_job(op.job_id)
                op_feats.append(
                    [
                        1.0 if op.is_scheduled else 0.0,
                        1.0 if op.is_ready() else 0.0,
                        min(pt_vals) / 5.0,
                        max(pt_vals) / 5.0,
                        (max(pt_vals) - min(pt_vals)) / 5.0,
                        rem_ops / max(1, self.config.num_ops_per_job),
                    ]
                )
            op_embeds = self.op_proj(
                torch.tensor(op_feats, dtype=torch.float32, device=dev)
            ).mean(dim=0)
        else:
            op_embeds = torch.zeros(self.config.d_operation, device=dev)

        if workers:
            w_feats = [
                [
                    w.lambda_,
                    w.mu,
                    w.current_fatigue,
                    w.next_available_time / max(1e-6, cmax_norm),
                    w.total_work_time / max(1e-6, cmax_norm),
                ]
                for w in workers
            ]
            w_embeds = self.worker_proj(
                torch.tensor(w_feats, dtype=torch.float32, device=dev)
            ).mean(dim=0)
        else:
            w_embeds = torch.zeros(self.config.d_worker, device=dev)

        cmax = max([o.complete_time for o in ops] + [0])
        energy = sum(m.total_energy for m in machine_states.values())
        progress = hdg.get_completed_ops() / max(1, hdg.get_total_ops())
        fatigue_ratio = (
                sum(1 for w in workers if w.current_fatigue < self.config.fatigue_threshold)
                / max(1, len(workers))
        )
        avg_fatigue = float(np.mean([w.current_fatigue for w in workers])) if workers else 0.0
        stats = torch.tensor(
            [progress, fatigue_ratio,
             cmax / max(1e-6, cmax_norm),
             energy / max(1e-6, energy_norm),
             avg_fatigue],
            dtype=torch.float32, device=dev,
        )
        batch_pool = torch.zeros(self.config.d_batch, device=dev)
        return torch.cat([stats, op_embeds, batch_pool, w_embeds], dim=-1)


class OpAgent(nn.Module):
    """工序级调度智能体"""

    def __init__(self, d_global, d_op=9):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(d_global + d_op, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, global_feat, op_feats):
        B, N, _ = op_feats.shape
        g = global_feat.unsqueeze(1).expand(-1, N, -1)
        x = torch.cat([g, op_feats], dim=-1)
        return self.fusion(x).squeeze(-1)


class MchAgent(nn.Module):


    def __init__(self, d_global, max_machines, d_w=5):
        super().__init__()
        self.context_mlp = nn.Sequential(
            nn.Linear(d_global + 9, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
        )
        self.batch_head = nn.Linear(64, max_machines)
        self.w_query = nn.Linear(64, 32)
        self.w_key = nn.Linear(d_w, 32)

    def forward(self, global_feat, sel_op_feat, worker_feats, valid_machines):
        ctx = self.context_mlp(torch.cat([global_feat, sel_op_feat], dim=-1))
        batch_logits = self.batch_head(ctx)
        batch_logits = batch_logits.masked_fill(~valid_machines, -1e9)
        q = self.w_query(ctx).unsqueeze(1)
        k = self.w_key(worker_feats)
        worker_scores = torch.matmul(q, k.transpose(1, 2)).squeeze(1) / (32 ** 0.5)
        worker_probs = F.softmax(worker_scores, dim=-1)
        return batch_logits, worker_probs


# ========================================================================
# ★ 新增：高效向量化 Pareto 过滤工具
# ========================================================================
def fast_pareto_filter_3d(points_np):
    """
    向量化 3 目标 Pareto 非支配过滤 (全部最小化)
    输入: np.ndarray (N, 3)
    输出: np.ndarray (M, 3), M <= N
    """
    if len(points_np) == 0:
        return np.empty((0, 3))
    # 按第 1 目标升序排序
    order = np.argsort(points_np[:, 0], kind="mergesort")
    sorted_pts = points_np[order]
    # 沿第 2 目标维护前缀最小值
    min_2 = np.minimum.accumulate(sorted_pts[:, 1])
    # 沿第 3 目标维护前缀最小值
    min_3 = np.minimum.accumulate(sorted_pts[:, 2])
    # 非支配条件: 第 2/3 目标值 == 当前行之前的累积最小值
    mask = (sorted_pts[:, 1] == min_2) & (sorted_pts[:, 2] == min_3)
    return sorted_pts[mask]


def approx_hv_3d_fast(pareto_points, ref_point, n_samples=200):
    """Monte Carlo HV 近似，配合向量化 Pareto 过滤"""
    if len(pareto_points) == 0:
        return 0.0
    samples = np.random.uniform(0, 1, (n_samples, 3)) * ref_point
    # 向量化判断每个采样点是否被 Pareto 集支配
    dominated = np.all(
        pareto_points[:, None, :] <= samples[None, :, :], axis=2
    )
    return float(np.prod(ref_point) * np.mean(np.any(dominated, axis=0)))


# ========================================================================
# 3. IMPMARL Agent (大规模优化版)
# ========================================================================
class IMPMARLAgent:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("IMPMARL 使用设备:", self.device)

        # ★ 固定最大维度，防止跨实例崩溃
        self.MAX_MACHINES = 30
        self.MAX_WORKERS = 30

        self.hdg = HeterogeneousDisjunctiveGraph(config)
        self.machine_states = {}
        self.pending_dynamic_jobs = []
        self.trained_episodes = 0
        self._compute_dynamic_parameters()

        self.encoder = LightweightStateEncoder(config).to(self.device)
        self.op_agent = OpAgent(config.d_global, d_op=9).to(self.device)
        self.mch_agent = MchAgent(config.d_global, self.MAX_MACHINES, d_w=5).to(self.device)

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.encoder.parameters(), "lr": 5e-5},
                {"params": self.op_agent.parameters(), "lr": 5e-5},
                {"params": self.mch_agent.parameters(), "lr": 5e-5},
            ]
        )

        # ★ Pareto archive 上限，控制 HV 计算开销
        self.PARETO_CAP = 300
        self.pareto_archive_np = np.empty((0, 3))


        self.fixed_reward_weights = np.array([0.4, 0.4, 0.2], dtype=float)
        self.gamma = 0.95
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.1
        self.entropy_coef = 0.001
        self.update_epochs = 3
        self.batch_size = 64


        self.reward_hv_interval = 5

        self.csv_log_path = "training_log_impmarl.csv"
        self.current_instance_name = ""
        self._init_csv_logger()
        self.best_val_score = float("inf")
        self.best_val_metrics = None

    def _compute_dynamic_parameters(self):
        self.cmax_norm_factor = max(100.0, getattr(self.config, "ref_cmax", None) or 500)
        self.energy_norm_factor = max(1000.0, getattr(self.config, "ref_tec", None) or 5000)
        # ★ 合理的 max_steps：总工序数 × 3（留余量），但不超过 4000
        total_ops = getattr(self.config, "num_jobs", 10) * getattr(self.config, "num_ops_per_job", 5)
        self.max_steps_per_episode = min(total_ops * 3, 4000)

    def _norm_params(self):
        return self.cmax_norm_factor, self.energy_norm_factor

    def _initialize_from_json(self):
        self.hdg = HeterogeneousDisjunctiveGraph(self.config)
        self.machine_states = {
            m: MachineState(m, self.config) for m in range(self.config.num_machines)
        }
        for w_id, w_info in self.config.worker_data.items():
            w = Worker(w_id, float(w_info.get("lambda", 0.05)), float(w_info.get("mu", 0.1)))
            w.config = self.config
            self.hdg.add_worker(w)
        self.pending_dynamic_jobs = []
        operations_by_job = {}
        for j_id in range(self.config.num_jobs_static):
            if j_id not in self.config.job_data:
                continue
            job_ops = []
            for o_id in range(self.config.num_ops_per_job):
                if o_id not in self.config.job_data[j_id]:
                    continue
                o_info = self.config.job_data[j_id][o_id]
                op = Operation(
                    j_id, o_id,
                    o_info.get("machine_candidates", []),
                    o_info.get("base_process_time", {}),
                    self.config.total_pieces_per_job,
                )
                job_ops.append(op)
            for i, op in enumerate(job_ops):
                if i > 0:
                    op.prev_op = job_ops[i - 1]
                    op.predecessors.append(job_ops[i - 1])
                if i < len(job_ops) - 1:
                    op.next_op = job_ops[i + 1]
                    op.successors.append(job_ops[i + 1])
                self.hdg.add_operation(op)
            operations_by_job[j_id] = job_ops
        for j_id in range(self.config.num_jobs_static, self.config.num_jobs):
            if j_id in self.config.job_data:
                self.pending_dynamic_jobs.append(
                    (j_id, self.config.job_ready_triggers.get(j_id, 0.5))
                )
        self.hdg.update_operation_status()
        return len(operations_by_job) > 0

    def reset_instance(self, json_path):
        self.config = Config(json_path=json_path)
        self.current_instance_name = os.path.basename(json_path)
        self._compute_dynamic_parameters()
        self._initialize_from_json()

    def _get_static_progress(self):
        total = completed = 0
        for op in self.hdg.operation_nodes.values():
            if op.job_id < self.config.num_jobs_static:
                total += 1
                if op.is_scheduled:
                    completed += 1
        return completed / total if total > 0 else 0.0

    def _check_dynamic_jobs_arrival(self):
        if not self.pending_dynamic_jobs:
            return False
        prog = self._get_static_progress()
        arrived = [
            (jid, trig) for jid, trig in self.pending_dynamic_jobs if prog >= trig
        ]
        for jid, _ in arrived:
            job_data = self.config.job_data[jid]
            job_ops = []
            for o_id in range(self.config.num_ops_per_job):
                if o_id not in job_data:
                    continue
                o_info = job_data[o_id]
                op = Operation(
                    jid, o_id,
                    o_info.get("machine_candidates", []),
                    o_info.get("base_process_time", {}),
                    self.config.total_pieces_per_job,
                )
                job_ops.append(op)
            for i, op in enumerate(job_ops):
                if i > 0:
                    op.prev_op = job_ops[i - 1]
                    op.predecessors.append(job_ops[i - 1])
                if i < len(job_ops) - 1:
                    op.next_op = job_ops[i + 1]
                    op.successors.append(job_ops[i + 1])
                self.hdg.add_operation(op)
            self.pending_dynamic_jobs.remove((jid, _))
        self.hdg.update_operation_status()
        return len(arrived) > 0

    def _execute_action(self, op_key, B_k, worker_ids):
        op = self.hdg.operation_nodes[op_key]
        prev_ct = op.get_prev_op_completion_time()
        machine_ids = self.hdg.get_spt_machine_order(op_key, self.machine_states)[:B_k]
        remaining = op.total_pieces - op.completed_pieces
        if remaining <= 0 or not machine_ids:
            return
        sizes = [remaining // B_k] * B_k
        for i in range(remaining % B_k):
            sizes[i] += 1
        sorted_workers = sorted(
            worker_ids, key=lambda w: self.hdg.worker_nodes[w].current_fatigue
        )
        size_worker_pairs = sorted(
            zip(sizes, sorted_workers), key=lambda x: x[0], reverse=True
        )
        max_ct = 0
        for b_idx, (b_size, w_id) in enumerate(size_worker_pairs):
            if b_size <= 0 or b_idx >= len(machine_ids):
                continue
            m_id = machine_ids[b_idx]
            m_state = self.machine_states.get(m_id)
            w = self.hdg.worker_nodes.get(w_id)
            if not m_state or not w:
                continue
            switch_time = self.config.machine_data.get(m_id, {}).get("switch_time", 0.0)
            m_avail = m_state.next_available_time
            w_avail = w.next_available_time
            if m_state.is_first_use or m_avail == 0:
                m_ready = m_avail
                actual_switch = 0.0
            else:
                m_ready = m_avail + switch_time
                actual_switch = switch_time
            start = max(prev_ct, m_ready, w_avail)
            idle_time = start - m_avail - actual_switch
            if idle_time > 0:
                m_state.add_idle_energy(idle_time)
            if actual_switch > 0:
                m_state.add_switch_energy(actual_switch)
            m_state.is_first_use = False
            base_t = op.base_pt.get(m_id, 1.0)
            actual_t = base_t * b_size * (1 + w.current_fatigue)
            ct = start + actual_t
            batch = SubBatch(op_key, b_idx, b_size, m_id, w_id)
            w.assign_batch(batch)
            batch.start_time = start
            batch.complete_time = ct
            batch.status = "completed"
            m_state.assign_batch(batch, start, actual_t)
            w.complete_batch(ct, start, actual_t)
            m_state.complete_batch()
            if ct > max_ct:
                max_ct = ct
        if max_ct > 0:
            op.update_completion(sum(s for s, _ in size_worker_pairs), max_ct)
        self.hdg.current_time = max_ct
        self.hdg.update_operation_status()

    def _get_current_metrics(self):
        cmax = max([o.complete_time for o in self.hdg.operation_nodes.values()] + [0])
        tec = sum(m.total_energy for m in self.machine_states.values())
        favg = (
            float(np.mean([w.current_fatigue for w in self.hdg.worker_nodes.values()]))
            if self.hdg.worker_nodes else 0.0
        )
        return cmax, tec, favg

    def _normalize_objs(self, objs):
        return np.array(objs) / np.array(
            [self.cmax_norm_factor, self.energy_norm_factor, 1.0]
        )

    # ====================================================================
    # ★ 优化：高效 Pareto archive + HV 计算
    # ====================================================================
    def _update_pareto_archive(self, new_point):
        """向 archive 追加一个点，然后用向量化过滤裁剪"""
        if len(self.pareto_archive_np) == 0:
            self.pareto_archive_np = new_point.reshape(1, 3)
            return
        self.pareto_archive_np = np.vstack([self.pareto_archive_np, new_point.reshape(1, 3)])
        # 向量化 Pareto 过滤
        self.pareto_archive_np = fast_pareto_filter_3d(self.pareto_archive_np)

        if len(self.pareto_archive_np) > self.PARETO_CAP:
            self.pareto_archive_np = self.pareto_archive_np[:self.PARETO_CAP]

    def _compute_hv(self):
        if len(self.pareto_archive_np) == 0:
            return 0.0
        return approx_hv_3d_fast(
            self.pareto_archive_np, self.hv_ref_point, n_samples=200
        )

    def _extract_op_features(self, op_keys, dev):
        """★ 优化：O(N) 而非 O(N × total_ops)，利用 hdg._job_ops_map"""
        if not op_keys:
            return torch.zeros(0, 9, dtype=torch.float32, device=dev)
        feats = []
        for op_k in op_keys:
            op = self.hdg.operation_nodes[op_k]
            pt_vals = list(op.base_pt.values()) if op.base_pt else [1.0]
            rem_ops = self.hdg.get_remaining_ops_in_job(op.job_id)
            wait_t = max(0, self.hdg.current_time - op.get_prev_op_completion_time())
            feats.append(
                [
                    op.completed_pieces / max(1, op.total_pieces),
                    1.0 if op.is_ready() else 0.0,
                    np.mean(pt_vals) / 5.0,
                    np.min(pt_vals) / 5.0,
                    np.max(pt_vals) / 5.0,
                    len(op.machine_candidates) / max(1, self.config.num_machines),
                    rem_ops / max(1, self.config.num_ops_per_job),
                    wait_t / max(1, self.cmax_norm_factor * 0.1),
                    (op.total_pieces - op.completed_pieces) / max(1, op.total_pieces),
                ]
            )
        return torch.tensor(feats, dtype=torch.float32, device=dev)

    def _extract_worker_features(self, dev):
        workers = list(self.hdg.worker_nodes.values())
        if not workers:
            return torch.zeros(1, 1, 5, dtype=torch.float32, device=dev)
        w_feats = [
            [
                w.lambda_, w.mu, w.current_fatigue,
                w.next_available_time / max(1e-6, self.cmax_norm_factor),
                w.total_work_time / max(1e-6, self.cmax_norm_factor),
            ]
            for w in workers
        ]
        return torch.tensor(w_feats, dtype=torch.float32, device=dev).unsqueeze(0)

    # ====================================================================
    # 核心 Episode 循环
    # ====================================================================
    def run_one_episode(self, episode=1, deterministic=False, collect_experience=True):
        # 重置 archive
        self.pareto_archive_np = np.empty((0, 3))
        self.hv_ref_point = np.array(
            [self.cmax_norm_factor * 1.2, self.energy_norm_factor * 1.2, 1.2]
        )
        self.prev_norm_objs = None
        self.prev_hv_val = 0.0

        episode_transitions = []
        done = False
        step = 0
        total_reward = 0.0
        episode_weights = []
        dev = self.device

        with torch.no_grad():
            while not done:
                step += 1

                # 1) 动态工件到达
                if self.pending_dynamic_jobs:
                    self._check_dynamic_jobs_arrival()

                # 2) 获取可调度工序
                ready_ops = self.hdg.get_ready_operations()
                if not ready_ops:
                    if (self.hdg.get_completed_ops() >= self.hdg.get_total_ops()
                            and not self.pending_dynamic_jobs) or step >= self.max_steps_per_episode:
                        break
                    continue

                # 3) 筛选有可用机器的工序
                valid_ops = [
                    op for op in ready_ops
                    if self.hdg.check_machine_availability(op, self.machine_states)
                ]
                if not valid_ops:
                    if (self.hdg.get_completed_ops() >= self.hdg.get_total_ops()
                            and not self.pending_dynamic_jobs) or step >= self.max_steps_per_episode:
                        break
                    continue

                # 4) OpAgent: 选择工序
                global_feat = self.encoder(
                    self.hdg, self.machine_states, self._norm_params(), dev
                ).unsqueeze(0)  # (1, 69)

                op_feats = self._extract_op_features(valid_ops, dev).unsqueeze(0)  # (1, N, 9)
                op_logits = self.op_agent(global_feat, op_feats).squeeze(0)  # (N,)

                if deterministic:
                    op_idx = torch.argmax(op_logits).item()
                else:
                    op_idx = Categorical(logits=op_logits).sample().item()

                selected_op_key = valid_ops[op_idx]
                sel_op = self.hdg.operation_nodes[selected_op_key]
                K = len(sel_op.machine_candidates)
                if K == 0:
                    continue

                # 5) MchAgent: 机器分批 + 工人选择（不再输出自适应偏好权重）
                K_act = min(K, self.MAX_MACHINES)
                sel_op_feat = self._extract_op_features([selected_op_key], dev)  # (1, 9)
                valid_machines_mask = torch.zeros(
                    1, self.MAX_MACHINES, dtype=torch.bool, device=dev
                )
                valid_machines_mask[0, :K_act] = True
                w_feats = self._extract_worker_features(dev)  # (1, W, 5)

                batch_logits, worker_probs = self.mch_agent(
                    global_feat, sel_op_feat, w_feats, valid_machines_mask
                )

                valid_batch_logits = batch_logits.squeeze(0)[:K_act]

                if deterministic:
                    b_k_idx = torch.argmax(valid_batch_logits).item()
                else:
                    b_k_idx = Categorical(logits=valid_batch_logits).sample().item()
                B_k = b_k_idx + 1
                B_k_actual = min(B_k, w_feats.shape[1], K_act)

                # 6) 工人采样
                w_probs = worker_probs.squeeze(0)
                if deterministic:
                    _, top_w_indices = torch.topk(w_probs, B_k_actual)
                else:
                    sampled_indices = []
                    w_probs_temp = w_probs.clone()
                    for _ in range(B_k_actual):
                        dist_w = Categorical(probs=w_probs_temp)
                        idx = dist_w.sample()
                        sampled_indices.append(idx.item())
                        w_probs_temp[idx] = 0.0
                        w_probs_temp = w_probs_temp / (w_probs_temp.sum() + 1e-8)
                    top_w_indices = torch.tensor(sampled_indices, device=dev)

                workers_list = list(self.hdg.worker_nodes.values())
                selected_worker_ids = [
                    workers_list[i].worker_id
                    for i in top_w_indices.tolist() if i < len(workers_list)
                ]

                w_cmax, w_tec, w_favg = self.fixed_reward_weights.tolist()
                episode_weights.append([w_cmax, w_tec, w_favg])

                # 7) 执行动作
                self._execute_action(selected_op_key, B_k_actual, selected_worker_ids)

                # 8) 固定权重奖励计算
                # 说明：这里不再使用 EMA 自适应权重，也不使用 HV 增量作为训练奖励。
                # 对比算法仅采用固定权重标量化，突出主算法的自适应多目标优化优势。
                curr_objs = np.array(self._get_current_metrics(), dtype=float)
                norm_objs = self._normalize_objs(curr_objs)
                self._update_pareto_archive(norm_objs)

                reward = 0.0
                if self.prev_norm_objs is not None and step > 1:
                    prev_n = self.prev_norm_objs
                    r_cmax = (prev_n[0] - norm_objs[0]) / (prev_n[0] + 1e-5)
                    r_tec = (prev_n[1] - norm_objs[1]) / (prev_n[1] + 1e-5)
                    r_favg = (prev_n[2] - norm_objs[2]) / (prev_n[2] + 1e-5)

                    fixed_w = self.fixed_reward_weights
                    reward = (
                        fixed_w[0] * r_cmax
                        + fixed_w[1] * r_tec
                        + fixed_w[2] * r_favg
                    )
                    reward = float(np.clip(reward, -2.0, 2.0))

                self.prev_norm_objs = norm_objs
                total_reward += reward

                # 9) 收集经验
                if collect_experience and step > 1:
                    dist_op = Categorical(logits=op_logits)
                    dist_b = Categorical(logits=valid_batch_logits)
                    episode_transitions.append({
                        "global_feat": global_feat.squeeze(0).cpu(),
                        "op_feats": op_feats.squeeze(0).cpu(),
                        "sel_op_feat": sel_op_feat.cpu(),
                        "w_feats": w_feats.squeeze(0).cpu(),
                        "valid_machines_mask": valid_machines_mask.squeeze(0).cpu(),
                        "op_idx": op_idx,
                        "b_k_idx": b_k_idx,
                        "worker_indices": top_w_indices.cpu(),
                        "op_log_prob": dist_op.log_prob(torch.tensor(op_idx, device=dev)).item(),
                        "b_log_prob": dist_b.log_prob(torch.tensor(b_k_idx, device=dev)).item(),
                        "reward": reward,
                    })

                # 10) 终止判定
                if (self.hdg.get_completed_ops() >= self.hdg.get_total_ops()
                        and not self.pending_dynamic_jobs):
                    break
                if step >= self.max_steps_per_episode:
                    break

        cmax, tec, favg = self._get_current_metrics()
        avg_w = (
            np.mean(episode_weights, axis=0).tolist()
            if episode_weights else [1 / 3, 1 / 3, 1 / 3]
        )
        return episode_transitions, {
            "makespan": cmax,
            "energy": tec,
            "fatigue_mean": favg,
            "reward": total_reward,
            "weights": avg_w,
        }

    # ====================================================================
    # ★ 优化：批量化 PPO 更新
    # ====================================================================
    def update_policy(self, episode_transitions):
        if len(episode_transitions) < 16:
            return
        dev = self.device
        for _epoch in range(self.update_epochs):
            indices = random.sample(
                range(len(episode_transitions)),
                min(self.batch_size, len(episode_transitions)),
            )
            self.optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=dev, requires_grad=True)
            for idx in indices:
                tr = episode_transitions[idx]
                gf = tr["global_feat"].to(dev)
                op_f = tr["op_feats"].to(dev)
                sel_f = tr["sel_op_feat"].to(dev)
                w_f = tr["w_feats"].to(dev).unsqueeze(0)
                vmask = tr["valid_machines_mask"].to(dev).unsqueeze(0)

                op_logits = self.op_agent(
                    gf.unsqueeze(0), op_f.unsqueeze(0)
                ).squeeze(0)
                batch_logits, _ = self.mch_agent(gf.unsqueeze(0), sel_f, w_f, vmask)
                K = int(tr["valid_machines_mask"].sum().item())
                new_op_lp = Categorical(logits=op_logits).log_prob(
                    torch.tensor(tr["op_idx"], device=dev)
                )
                new_b_lp = Categorical(
                    logits=batch_logits.squeeze(0)[:K]
                ).log_prob(torch.tensor(tr["b_k_idx"], device=dev))
                ratio_op = torch.exp(new_op_lp - tr["op_log_prob"])
                ratio_b = torch.exp(new_b_lp - tr["b_log_prob"])
                r = tr["reward"]
                surr_op = torch.min(
                    ratio_op * r,
                    torch.clamp(ratio_op, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * r
                )
                surr_b = torch.min(
                    ratio_b * r,
                    torch.clamp(ratio_b, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * r
                )
                total_loss = total_loss - (surr_op + surr_b)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters())
                + list(self.op_agent.parameters())
                + list(self.mch_agent.parameters()),
                0.5,
            )
            self.optimizer.step()

    # ====================================================================
    # 验证与训练
    # ====================================================================
    def evaluate_validation(self, val_files, eval_runs=1):
        cmax_l, tec_l, favg_l = [], [], []
        for j_path in val_files:
            for _ in range(eval_runs):
                self.reset_instance(j_path)
                _, m = self.run_one_episode(0, True, False)
                cmax_l.append(m["makespan"])
                tec_l.append(m["energy"])
                favg_l.append(m["fatigue_mean"])
        return {
            "mean_cmax": np.mean(cmax_l),
            "mean_tec": np.mean(tec_l),
            "mean_favg": np.mean(favg_l),
            "mean_cmax_ratio": np.mean(cmax_l) / max(1e-6, self.cmax_norm_factor),
            "mean_tec_ratio": np.mean(tec_l) / max(1e-6, self.energy_norm_factor),
            "mean_favg_ratio": np.mean(favg_l),
        }

    def compute_val_score(self, vm):
        return (
            0.50 * vm.get("mean_cmax_ratio", 1)
            + 0.30 * vm.get("mean_tec_ratio", 1)
            + 0.20 * vm.get("mean_favg_ratio", 0.5)
        )

    def is_better_validation(self, vs, vm, min_improve=0.003, tol=0.03):
        if not self.best_val_metrics or not np.isfinite(self.best_val_score):
            return True
        if vs < self.best_val_score * (1 - min_improve):
            o = self.best_val_metrics
            return (
                vm["mean_cmax"] <= o["mean_cmax"] * (1 + tol)
                and vm["mean_tec"] <= o["mean_tec"] * (1 + tol)
                and vm["mean_favg"] <= o["mean_favg"] * (1 + tol)
            )
        return False

    def train_multi_instance(
        self,
        train_files,
        val_files=None,
        total_episodes=5000,
        update_interval=8,
        eval_interval=500,
        save_interval=500,
        save_dir="saved_models_impmarl",
        debug_interval=10,
    ):
        os.makedirs(save_dir, exist_ok=True)
        old_ep = self.trained_episodes
        final_ep = old_ep + total_episodes
        print(
            "\n🚀 IMPMARL 三目标对比训练开始 | 已训练={}, 本次={}, 目标={}".format(
                old_ep, total_episodes, final_ep
            )
        )
        print("  max_steps/episode={}, MAX_MACHINES={}, PARETO_CAP={}".format(
            self.max_steps_per_episode, self.MAX_MACHINES, self.PARETO_CAP
        ))
        print("  固定奖励权重: Cmax={:.2f}, TEC={:.2f}, Favg={:.2f}".format(
            self.fixed_reward_weights[0], self.fixed_reward_weights[1], self.fixed_reward_weights[2]
        ))

        for ep in range(old_ep + 1, final_ep + 1):
            t0 = time.time()
            self.reset_instance(random.choice(train_files))
            trans, metrics = self.run_one_episode(ep, False, True)

            if trans and ep % update_interval == 0:
                self.update_policy(trans)
            self.trained_episodes = ep
            self._log_episode_to_csv(
                ep, time.time() - t0,
                metrics["makespan"], metrics["energy"],
                metrics["fatigue_mean"], metrics["reward"], metrics["weights"],
            )
            if ep % debug_interval == 0:
                print(
                    "Ep {:5d} | {} | Cmax={:.1f} | TEC={:.1f} | Favg={:.3f} | R={:.3f} | {:.1f}s".format(
                        ep, self.current_instance_name,
                        metrics["makespan"], metrics["energy"],
                        metrics["fatigue_mean"], metrics["reward"],
                        time.time() - t0,
                    )
                )
            if val_files and ep % eval_interval == 0:
                vm = self.evaluate_validation(val_files)
                vs = self.compute_val_score(vm)
                print(
                    "[验证] Ep {} | score={:.4f} | Cmax={:.1f} TEC={:.1f} Favg={:.3f}".format(
                        ep, vs, vm["mean_cmax"], vm["mean_tec"], vm["mean_favg"]
                    )
                )
                if self.is_better_validation(vs, vm):
                    self.best_val_score = vs
                    self.best_val_metrics = vm
                    self.save_models(save_dir, "best_val")
            if ep % save_interval == 0:
                self.save_models(save_dir, "latest")
            del trans
            gc.collect()
        self.save_models(save_dir, "final")

    # ====================================================================
    # 模型保存/加载/日志
    # ====================================================================
    def save_models(self, path, name):
        os.makedirs(path, exist_ok=True)
        p = os.path.join(path, "impmarl_{}.pth".format(name))
        torch.save({
            "encoder": self.encoder.state_dict(),
            "op_agent": self.op_agent.state_dict(),
            "mch_agent": self.mch_agent.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "trained_episodes": self.trained_episodes,
            "best_val_score": self.best_val_score,
            "best_val_metrics": self.best_val_metrics,
            "MAX_MACHINES": self.MAX_MACHINES,
            "fixed_reward_weights": self.fixed_reward_weights.tolist(),
        }, p)
        print("模型已保存:", p)

    def load_models(self, path):
        ckp = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckp["encoder"])
        self.op_agent.load_state_dict(ckp["op_agent"])
        self.mch_agent.load_state_dict(ckp["mch_agent"])
        self.optimizer.load_state_dict(ckp["optimizer"])
        self.trained_episodes = ckp.get("trained_episodes", 0)
        self.best_val_score = ckp.get("best_val_score", float("inf"))
        self.best_val_metrics = ckp.get("best_val_metrics", None)
        if "fixed_reward_weights" in ckp:
            self.fixed_reward_weights = np.array(ckp["fixed_reward_weights"], dtype=float)
        print("模型已从 {} 加载 (已训练 {} episodes)".format(path, self.trained_episodes))

    def _init_csv_logger(self):
        if not os.path.exists(self.csv_log_path):
            with open(self.csv_log_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    "instance_name", "episode", "duration_seconds",
                    "cmax", "total_energy", "avg_fatigue", "total_reward",
                    "weight_cmax", "weight_energy", "weight_fatigue",
                ])

    def _log_episode_to_csv(self, ep, dur, cmax, energy, favg, reward, weights):
        try:
            with open(self.csv_log_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    self.current_instance_name, ep, "{:.2f}".format(dur),
                    "{:.2f}".format(cmax), "{:.2f}".format(energy),
                    "{:.4f}".format(favg), "{:.4f}".format(reward),
                    "{:.4f}".format(weights[0]), "{:.4f}".format(weights[1]),
                    "{:.4f}".format(weights[2]),
                ])
        except:
            pass


# ========================================================================
# 入口
# ========================================================================
if __name__ == "__main__":
    train_dir = "../../TestData/train2"
    val_dir = "../../TestData/yanzheng1"
    save_dir = "saved_models_impmarl"
    train_files = sorted(glob.glob(os.path.join(train_dir, "*.json")))[:100]
    val_files = (
        sorted(glob.glob(os.path.join(val_dir, "*.json")))
        if os.path.exists(val_dir) else None
    )
    print("训练实例数:", len(train_files))
    print("验证实例数:", len(val_files) if val_files else 0)
    agent = IMPMARLAgent(Config(json_path=train_files[0]))
    latest_path = os.path.join(save_dir, "impmarl_best_val.pth")
    if os.path.exists(latest_path):
        print("[载入已有模型]", latest_path)
        agent.load_models(latest_path)
    agent.train_multi_instance(
        train_files=train_files,
        val_files=val_files,
        total_episodes=5000,
        update_interval=8,
        eval_interval=500,
        save_interval=500,
        save_dir=save_dir,
        debug_interval=10,
    )
