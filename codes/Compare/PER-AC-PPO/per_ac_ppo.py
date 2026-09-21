

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import defaultdict, deque
from itertools import combinations
import copy
import json
import random
import math
import matplotlib.pyplot as plt
import os
import glob
import csv
import time
import gc
from datetime import datetime




class Config:
    def __init__(self, json_path=None):
        self.json_path = json_path
        if json_path:
            self.load_from_json(json_path)

    def load_from_json(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        reference = data.get('reference_solution', {})
        self.ref_cmax = reference.get('Cmax', None)
        self.ref_tec = reference.get('TEC', None)
        self.ref_favg = reference.get('Favg', None)
        global_config = data.get('global_config', {})
        self.d_operation = 32
        self.d_batch = 16
        self.d_worker = 16
        self.d_global = 5 + self.d_operation + self.d_batch + self.d_worker
        self.fatigue_threshold = global_config.get('fatigue_threshold', 0.8)
        self.num_jobs = global_config.get('num_jobs', 10)
        self.num_jobs_static = global_config.get('num_jobs_static', 8)
        self.num_jobs_dynamic = global_config.get('num_jobs_dynamic', 2)
        self.num_ops_per_job = global_config.get('num_ops_per_job', 5)
        self.num_machines = global_config.get('num_machines', 10)
        self.num_workers = global_config.get('num_workers', 5)
        self.total_pieces_per_job = global_config.get('total_pieces_per_job', 10)
        self.fusion_dims = [128, 64, 32]
        self.score_dims = [32, 16]
        self.value_hidden_dims = [128, 64, 32]
        self.actor_lr = 2e-4
        self.critic_lr = 2e-4
        self.hdgat_lr = 2e-4
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.entropy_coef = 0.01
        self.batch_size = 8
        self.update_epochs = 1
        self.target_kl = 0.02
        self.value_coef = 0.05
        self.use_action_attention = False
        machine_static = data.get('machine_static_data', {})
        self.machine_data = {}
        for machine_key, machine_info in machine_static.items():
            machine_id = int(machine_key.split('_')[1])
            self.machine_data[machine_id] = machine_info
        worker_static = data.get('worker_static_data', {})
        self.worker_data = {}
        for worker_key, worker_info in worker_static.items():
            worker_id = int(worker_key.split('_')[1])
            self.worker_data[worker_id] = {
                'lambda': worker_info.get('lambda', 0.05),
                'mu': worker_info.get('mu', 0.1)
            }
        job_data = data.get('job_operation_static_data', {})
        self.job_data = {}
        self.job_ready_triggers = {}
        for job_key, job_info in job_data.items():
            job_id = int(job_key.split('_')[1])
            self.job_data[job_id] = {}
            first_op = job_info.get('op_0', {})
            if 'ready_trigger' in first_op:
                self.job_ready_triggers[job_id] = first_op['ready_trigger']
            for op_key, op_info in job_info.items():
                op_id = int(op_key.split('_')[1])
                if 'base_process_time' in op_info:
                    base_pt = op_info['base_process_time']
                    base_pt_int = {int(k): v for k, v in base_pt.items()}
                    op_info['base_process_time'] = base_pt_int
                if 'machine_candidates' in op_info:
                    machine_candidates = [int(x) for x in op_info['machine_candidates']]
                    op_info['machine_candidates'] = machine_candidates
                self.job_data[job_id][op_id] = op_info


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
        self.waiting_time = 0

    def get_prev_op_completion_time(self):
        if self.prev_op:
            return self.prev_op.complete_time
        return 0

    def are_all_predecessors_completed(self):
        for prev_op in self.predecessors:
            if not prev_op.is_scheduled or prev_op.completed_pieces < prev_op.total_pieces:
                return False
        return True

    def update_predecessor_status(self):
        self.all_predecessors_completed = self.are_all_predecessors_completed()
        if self.all_predecessors_completed and self.ready_time == 0:
            self.ready_time = max([prev_op.complete_time for prev_op in self.predecessors] + [0])

    def is_ready(self):
        return (self.all_predecessors_completed and
                self.completed_pieces < self.total_pieces and
                not self.is_scheduled)

    def mark_completed(self, completion_time):
        self.is_scheduled = True
        self.complete_time = completion_time
        self.completed_pieces = self.total_pieces
        for succ in self.successors:
            succ.update_predecessor_status()

    def update_completion(self, new_completed_pieces, completion_time=None):
        self.completed_pieces += new_completed_pieces
        if self.completed_pieces >= self.total_pieces:
            self.is_scheduled = True
            if completion_time:
                self.complete_time = max(self.complete_time, completion_time)
                for succ in self.successors:
                    succ.update_predecessor_status()


class ProcessingRecord:
    def __init__(self, op_id, batch_id, size, machine_id, worker_id):
        self.op_id = op_id
        self.batch_id = batch_id
        self.size = size
        self.machine_id = machine_id
        self.worker_id = worker_id
        self.start_time = 0
        self.complete_time = 0
        self.status = "pending"
        self.is_minimal_unit = True


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
        self.last_fatigue_update_time = 0.0
        self.next_available_time = 0.0
        self.config = None

    def update_fatigue_after_work(self, actual_time, start_time):
        fatigue_increase = (1 - self.current_fatigue) * (1 - np.exp(-self.lambda_ * actual_time))
        self.current_fatigue = min(1.0, self.current_fatigue + fatigue_increase)
        if self.current_fatigue >= self.config.fatigue_threshold:
            self.is_resting = False
            recovery_time = self._calculate_recovery_time_to_target(0.5)
            self.next_available_time = recovery_time + start_time + actual_time
            self.current_fatigue = 0.5
        else:
            self.next_available_time = start_time + actual_time
            self.is_resting = False

    def _calculate_recovery_time_to_target(self, target_fatigue):
        if self.current_fatigue <= target_fatigue:
            return 0.0
        recovery_time = -np.log(target_fatigue / self.current_fatigue) / self.mu
        return round(max(0.0, recovery_time), 2)

    def assign_batch(self, batch):
        self.assigned_batches.append(batch)
        self.current_batch = batch
        self.is_busy = True
        self.current_machine = batch.machine_id

    def complete_batch(self, complete_time, start_time, actual_time):
        """完成当前加工记录，修复 current_record/current_batch 变量名不一致问题。"""
        batch = getattr(self, "current_batch", None)

        # 兼容旧版本中可能残留的 current_record 字段。
        if batch is None:
            batch = getattr(self, "current_record", None)

        if batch is not None:
            self.update_fatigue_after_work(actual_time, start_time)
            batch.status = "completed"
            batch.complete_time = complete_time
            self.total_work_time += actual_time

        self.current_batch = None
        self.current_record = None
        self.is_busy = False
        self.current_machine = None

    def is_available(self, check_time):
        if self.is_busy:
            return False
        if check_time < self.next_available_time:
            return False
        if check_time >= self.next_available_time:
            self.is_resting = False
        return (not self.is_busy and
                not self.is_resting and
                self.current_fatigue < self.config.fatigue_threshold)


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
            energy_process = self.config.machine_data[self.machine_id].get('energy_process', 0)
            self.total_process_energy += process_time * energy_process

    def complete_batch(self):
        self.is_busy = False
        self.current_batch = None

    def add_idle_energy(self, idle_time):
        if idle_time > 0 and self.config and self.machine_id in self.config.machine_data:
            energy_idle = self.config.machine_data[self.machine_id].get('energy_idle', 0)
            self.total_idle_energy += idle_time * energy_idle

    def add_switch_energy(self, switch_time):
        if switch_time > 0 and self.config and self.machine_id in self.config.machine_data:
            energy_switch = self.config.machine_data[self.machine_id].get('energy_switch', 0)
            self.total_switch_energy += switch_time * energy_switch

    @property
    def total_energy(self):
        return self.total_idle_energy + self.total_switch_energy + self.total_process_energy


class HeterogeneousDisjunctiveGraph:
    def __init__(self, config: Config):
        self.config = config
        self.start_node = "S"
        self.end_node = "E"
        self.operation_nodes = {}
        # PER-AC-PPO 对比算法按原论文仅使用 O-M-W 图结构，
        # 即工序节点、机器节点、工人节点及 O-M/O-W 关联。
        # 不建立子批节点，也不把分批数作为动作，避免引入本文主算法的可变子批创新。
        self.worker_nodes = {}
        self.temporal_arcs = []
        self.current_time = 0

    def add_operation(self, op: Operation):
        op_key = f"J{op.job_id}_O{op.op_id}"
        self.operation_nodes[op_key] = op
        if op.prev_op is None:
            self.temporal_arcs.append((self.start_node, op_key))
        if op.prev_op:
            prev_op_key = f"J{op.prev_op.job_id}_O{op.prev_op.op_id}"
            self.temporal_arcs.append((prev_op_key, op_key))
        if op.next_op:
            next_op_key = f"J{op.next_op.job_id}_O{op.next_op.op_id}"
            self.temporal_arcs.append((op_key, next_op_key))
        else:
            self.temporal_arcs.append((op_key, self.end_node))

    def add_worker(self, worker: Worker):
        self.worker_nodes[worker.worker_id] = worker

    def get_ready_operations(self):
        ready_ops = []
        for op_key, op in self.operation_nodes.items():
            if op.is_ready():
                ready_ops.append(op_key)
        return ready_ops

    def update_operation_status(self):
        for op in self.operation_nodes.values():
            op.update_predecessor_status()

    def check_machine_availability(self, op_key, machine_states):
        op = self.operation_nodes[op_key]
        for machine_id in op.machine_candidates:
            if machine_id in machine_states:
                if machine_states[machine_id].is_busy:
                    return False
            else:
                return False
        return True

    def get_spt_machine_order(self, op_key, machine_states):
        op = self.operation_nodes[op_key]
        machine_times = []
        for machine_id in op.machine_candidates:
            if machine_id in machine_states:
                base_time = op.base_pt.get(machine_id, float('inf'))
                machine_times.append((machine_id, base_time))
        sorted_machines = sorted(machine_times, key=lambda x: x[1])
        return [machine_id for machine_id, _ in sorted_machines]


# ============================================================================
#  第二部分：NHGNN 三阶段异构图神经网络编码器（论文核心创新）
# ============================================================================

class MachineAttentionLayer(nn.Module):
    """
    NHGNN Stage 1: 机器节点嵌入
    Step 1: 将工人特征嵌入到工序节点（O←W 路径）
    Step 2: 将增强的工序节点嵌入到机器节点（M←O^W 路径）
    """

    def __init__(self, d, d_OW, d_OM):
        super().__init__()
        self.d = d
        # Step 1: Worker → Operation
        self.W_O = nn.Linear(d, d)
        self.WOW = nn.Linear(d + d_OW, d)
        # Step 2: Enhanced Operation → Machine
        self.W_M = nn.Linear(d, d)
        self.WOM = nn.Linear(d + d_OM, d)
        self.a = nn.Parameter(torch.randn(2 * d) * 0.01)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x_op, z_worker, v_ow, y_machine, u_om, mask_ow, mask_om):
        """
        x_op:      (N_op, d)       工序嵌入
        z_worker:  (N_worker, d)   工人嵌入
        v_ow:      (N_op, N_worker, d_OW)  O-W 边特征
        y_machine: (N_machine, d)  机器嵌入
        u_om:      (N_op, N_machine, d_OM) O-M 边特征
        mask_ow:   (N_op, N_worker)         O-W 邻接掩码
        mask_om:   (N_machine, N_op)        M-O 邻接掩码（转置视图）
        """
        N_op = x_op.size(0)
        N_worker = z_worker.size(0)
        N_machine = y_machine.size(0)

        # ---- Step 1: Worker → Operation ----
        WO_x = self.W_O(x_op)                                                    # (N_op, d)
        z_exp = z_worker.unsqueeze(0).expand(N_op, -1, -1)                       # (N_op, N_worker, d)
        WOW_zv = self.WOW(torch.cat([z_exp, v_ow], dim=-1))                      # (N_op, N_worker, d)

        # 自注意力: e_ij
        concat_self = torch.cat([WO_x, WO_x], dim=-1)                            # (N_op, 2d)
        e_self = self.leaky_relu(torch.einsum('ij,j->i', concat_self, self.a))    # (N_op,)

        # 交叉注意力: e_ijr
        WO_x_exp = WO_x.unsqueeze(1).expand(-1, N_worker, -1)                   # (N_op, N_worker, d)
        concat_cross = torch.cat([WO_x_exp, WOW_zv], dim=-1)                     # (N_op, N_worker, 2d)
        e_cross = self.leaky_relu(torch.einsum('ijk,k->ij', concat_cross, self.a))# (N_op, N_worker)

        # 合并自注意力与交叉注意力，Softmax 归一化
        e_all = torch.cat([e_self.unsqueeze(1), e_cross], dim=1)                 # (N_op, 1+N_worker)
        mask_all = torch.cat([
            torch.ones(N_op, 1, device=x_op.device, dtype=x_op.dtype), mask_ow
        ], dim=1)
        e_all = e_all.masked_fill(mask_all == 0, -1e9)
        alpha = F.softmax(e_all, dim=1)
        alpha_self = alpha[:, 0:1]                                                # (N_op, 1)
        alpha_cross = alpha[:, 1:]                                               # (N_op, N_worker)

        x_op_w = torch.sigmoid(
            alpha_self * WO_x + torch.einsum('ij,ijk->ik', alpha_cross, WOW_zv)
        )                                                                         # (N_op, d)

        # ---- Step 2: Enhanced Operation → Machine ----
        WM_y = self.W_M(y_machine)                                                # (N_machine, d)
        x_op_w_exp = x_op_w.unsqueeze(0).expand(N_machine, -1, -1)              # (N_machine, N_op, d)
        u_om_T = u_om.transpose(0, 1)                                            # (N_machine, N_op, d_OM)
        WOM_xu = self.WOM(torch.cat([x_op_w_exp, u_om_T], dim=-1))              # (N_machine, N_op, d)

        concat_self_m = torch.cat([WM_y, WM_y], dim=-1)                          # (N_machine, 2d)
        e_self_m = self.leaky_relu(torch.einsum('ij,j->i', concat_self_m, self.a))

        WM_y_exp = WM_y.unsqueeze(1).expand(-1, N_op, -1)                        # (N_machine, N_op, d)
        concat_cross_m = torch.cat([WM_y_exp, WOM_xu], dim=-1)
        e_cross_m = self.leaky_relu(torch.einsum('ijk,k->ij', concat_cross_m, self.a))

        e_all_m = torch.cat([e_self_m.unsqueeze(1), e_cross_m], dim=1)
        mask_all_m = torch.cat([
            torch.ones(N_machine, 1, device=y_machine.device, dtype=y_machine.dtype),
            mask_om
        ], dim=1)
        e_all_m = e_all_m.masked_fill(mask_all_m == 0, -1e9)
        alpha_m = F.softmax(e_all_m, dim=1)
        alpha_self_m = alpha_m[:, 0:1]
        alpha_cross_m = alpha_m[:, 1:]

        y_new = torch.sigmoid(
            alpha_self_m * WM_y + torch.einsum('ij,ijk->ik', alpha_cross_m, WOM_xu)
        )                                                                         # (N_machine, d)
        return y_new


class WorkerAttentionLayer(nn.Module):
    """
    NHGNN Stage 2: 工人节点嵌入
    Step 1: 将机器特征嵌入到工序节点（O←M 路径）
    Step 2: 将增强的工序节点嵌入到工人节点（W←O^M 路径）
    """

    def __init__(self, d, d_OM, d_OW):
        super().__init__()
        self.d = d
        self.W_O = nn.Linear(d, d)
        self.WOM = nn.Linear(d + d_OM, d)
        self.W_W = nn.Linear(d, d)
        self.WOW = nn.Linear(d + d_OW, d)
        self.a = nn.Parameter(torch.randn(2 * d) * 0.01)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x_op, y_machine, u_om, z_worker, v_ow, mask_om_T, mask_ow_T):
        """
        x_op:       (N_op, d)
        y_machine:  (N_machine, d)
        u_om:       (N_op, N_machine, d_OM)
        z_worker:   (N_worker, d)
        v_ow:       (N_op, N_worker, d_OW)
        mask_om_T:  (N_op, N_machine)       O-M 邻接掩码
        mask_ow_T:  (N_worker, N_op)        W-O 邻接掩码（转置视图）
        """
        N_op = x_op.size(0)
        N_machine = y_machine.size(0)
        N_worker = z_worker.size(0)

        # ---- Step 1: Machine → Operation ----
        WO_x = self.W_O(x_op)
        y_exp = y_machine.unsqueeze(0).expand(N_op, -1, -1)
        WOM_yu = self.WOM(torch.cat([y_exp, u_om], dim=-1))

        concat_self = torch.cat([WO_x, WO_x], dim=-1)
        e_self = self.leaky_relu(torch.einsum('ij,j->i', concat_self, self.a))

        WO_x_exp = WO_x.unsqueeze(1).expand(-1, N_machine, -1)
        concat_cross = torch.cat([WO_x_exp, WOM_yu], dim=-1)
        e_cross = self.leaky_relu(torch.einsum('ijk,k->ij', concat_cross, self.a))

        e_all = torch.cat([e_self.unsqueeze(1), e_cross], dim=1)
        mask_all = torch.cat([
            torch.ones(N_op, 1, device=x_op.device, dtype=x_op.dtype), mask_om_T
        ], dim=1)
        e_all = e_all.masked_fill(mask_all == 0, -1e9)
        alpha = F.softmax(e_all, dim=1)
        alpha_self = alpha[:, 0:1]
        alpha_cross = alpha[:, 1:]

        x_op_m = torch.sigmoid(
            alpha_self * WO_x + torch.einsum('ij,ijk->ik', alpha_cross, WOM_yu)
        )

        # ---- Step 2: Enhanced Operation → Worker ----
        WW_z = self.W_W(z_worker)
        x_op_m_exp = x_op_m.unsqueeze(0).expand(N_worker, -1, -1)
        v_ow_T = v_ow.transpose(0, 1)                                            # (N_worker, N_op, d_OW)
        WOW_xv = self.WOW(torch.cat([x_op_m_exp, v_ow_T], dim=-1))

        concat_self_w = torch.cat([WW_z, WW_z], dim=-1)
        e_self_w = self.leaky_relu(torch.einsum('ij,j->i', concat_self_w, self.a))

        WW_z_exp = WW_z.unsqueeze(1).expand(-1, N_op, -1)
        concat_cross_w = torch.cat([WW_z_exp, WOW_xv], dim=-1)
        e_cross_w = self.leaky_relu(torch.einsum('ijk,k->ij', concat_cross_w, self.a))

        e_all_w = torch.cat([e_self_w.unsqueeze(1), e_cross_w], dim=1)
        mask_all_w = torch.cat([
            torch.ones(N_worker, 1, device=z_worker.device, dtype=z_worker.dtype),
            mask_ow_T
        ], dim=1)
        e_all_w = e_all_w.masked_fill(mask_all_w == 0, -1e9)
        alpha_w = F.softmax(e_all_w, dim=1)
        alpha_self_w = alpha_w[:, 0:1]
        alpha_cross_w = alpha_w[:, 1:]

        z_new = torch.sigmoid(
            alpha_self_w * WW_z + torch.einsum('ij,ijk->ik', alpha_cross_w, WOW_xv)
        )
        return z_new


class OperationMLPLayer(nn.Module):
    """
    NHGNN Stage 3: 工序节点嵌入
    使用 MLP 聚合前驱/后继工序原始特征 + 邻居机器/工人嵌入
    """

    def __init__(self, d, d_O):
        super().__init__()
        self.d = d
        self.mlp_prev = nn.Sequential(nn.Linear(d_O, d), nn.ELU(), nn.Linear(d, d))
        self.mlp_next = nn.Sequential(nn.Linear(d_O, d), nn.ELU(), nn.Linear(d, d))
        self.mlp_machine = nn.Sequential(nn.Linear(d, d), nn.ELU(), nn.Linear(d, d))
        self.mlp_worker = nn.Sequential(nn.Linear(d, d), nn.ELU(), nn.Linear(d, d))
        self.mlp_cur = nn.Sequential(nn.Linear(d_O, d), nn.ELU(), nn.Linear(d, d))
        self.mlp_final = nn.Sequential(nn.Linear(5 * d, d), nn.ELU(), nn.Linear(d, d))

    def forward(self, op_raw, op_prev_raw, op_next_raw, y_machine_avg, z_worker_avg):
        h1 = self.mlp_prev(op_prev_raw)
        h2 = self.mlp_next(op_next_raw)
        h3 = self.mlp_machine(y_machine_avg)
        h4 = self.mlp_worker(z_worker_avg)
        h5 = self.mlp_cur(op_raw)
        concat = torch.cat([h1, h2, h3, h4, h5], dim=-1)
        return self.mlp_final(F.elu(concat))


class NHGNNEncoder(nn.Module):
    """
    三阶段新颖异构图神经网络 (NHGNN)
    - Stage 1: 机器节点嵌入 (Worker→Op→Machine)
    - Stage 2: 工人节点嵌入 (Machine→Op→Worker)
    - Stage 3: 工序节点嵌入 (MLP 聚合)
    - Graph-level: 均值池化拼接 → 3d 维全局状态向量

    特征维度（对齐论文 Table 3）:
      Operation: d_O=7,  Machine: d_M=5,  Worker: d_W=5
      O-M edge: d_OM=1,  O-W edge: d_OW=2
    """

    def __init__(self, config, d=64, num_layers=2):
        super().__init__()
        self.config = config
        self.d = d
        self.num_layers = num_layers
        self.d_O = 7
        self.d_M = 5
        self.d_W = 5
        self.d_OM = 1
        self.d_OW = 2

        # 初始投影层（仅 Layer 0 使用，将原始特征映射到 d 维）
        self.op_proj = nn.Linear(self.d_O, d)
        self.machine_proj = nn.Linear(self.d_M, d)
        self.worker_proj = nn.Linear(self.d_W, d)

        # 堆叠 N 层独立参数的三阶段编码
        self.machine_layers = nn.ModuleList([
            MachineAttentionLayer(d, self.d_OW, self.d_OM) for _ in range(num_layers)
        ])
        self.worker_layers = nn.ModuleList([
            WorkerAttentionLayer(d, self.d_OM, self.d_OW) for _ in range(num_layers)
        ])
        self.op_layers = nn.ModuleList([
            OperationMLPLayer(d, self.d_O) for _ in range(num_layers)
        ])

        self.agent_ref = None

    def set_agent(self, agent):
        self.agent_ref = agent

    def _extract_raw_features(self, hdg, machine_states, device):
        """提取 NHGNN 所需的原始节点特征和边特征"""
        current_cmax = max([o.complete_time for o in hdg.operation_nodes.values()] + [0])
        cmax_norm = self.agent_ref.cmax_norm_factor if self.agent_ref else 1.0
        total_ops = len(hdg.operation_nodes)

        # ---- 工序原始特征 (d_O=7) ----
        op_raw_list, op_keys_list = [], []
        op_prev_raw_list, op_next_raw_list = [], []
        for op_key, op in hdg.operation_nodes.items():
            pt_values = list(op.base_pt.values()) if op.base_pt else [1.0]
            avg_pt = float(np.mean(pt_values))
            remaining_ops = sum(
                1 for o in hdg.operation_nodes.values()
                if o.job_id == op.job_id and not o.is_scheduled
            )
            # 预估工件完成时间
            job_ops = [o for o in hdg.operation_nodes.values() if o.job_id == op.job_id]
            job_est_complete = max(
                (o.complete_time if o.is_scheduled else current_cmax + avg_pt) for o in job_ops
            ) if job_ops else 0.0

            op_raw_list.append([
                1.0 if op.is_scheduled else 0.0,
                remaining_ops / max(1, self.config.num_ops_per_job),
                avg_pt / 5.0,
                len(op.machine_candidates) / max(1, self.config.num_machines),
                self.config.num_workers / max(1, self.config.num_workers),
                (op.ready_time if op.ready_time > 0 else 0.0) / max(1e-6, cmax_norm),
                job_est_complete / max(1e-6, cmax_norm),
            ])

            # 前驱工序原始特征
            if op.prev_op and op.prev_op.job_id == op.job_id:
                prev = op.prev_op
                prev_pt = list(prev.base_pt.values()) if prev.base_pt else [1.0]
                op_prev_raw_list.append([
                    1.0 if prev.is_scheduled else 0.0, float(np.mean(prev_pt)) / 5.0,
                    len(prev.machine_candidates) / max(1, self.config.num_machines),
                    0.0, 0.0, 0.0, 0.0
                ])
            else:
                op_prev_raw_list.append([0.0] * 7)

            # 后继工序原始特征
            if op.next_op and op.next_op.job_id == op.job_id:
                nxt = op.next_op
                nxt_pt = list(nxt.base_pt.values()) if nxt.base_pt else [1.0]
                op_next_raw_list.append([
                    1.0 if nxt.is_scheduled else 0.0, float(np.mean(nxt_pt)) / 5.0,
                    len(nxt.machine_candidates) / max(1, self.config.num_machines),
                    0.0, 0.0, 0.0, 0.0
                ])
            else:
                op_next_raw_list.append([0.0] * 7)

            op_keys_list.append(op_key)

        op_raw = torch.tensor(op_raw_list, dtype=torch.float32, device=device) if op_raw_list \
            else torch.zeros((0, self.d_O), dtype=torch.float32, device=device)
        op_prev_raw = torch.tensor(op_prev_raw_list, dtype=torch.float32, device=device) if op_prev_raw_list \
            else torch.zeros((0, self.d_O), dtype=torch.float32, device=device)
        op_next_raw = torch.tensor(op_next_raw_list, dtype=torch.float32, device=device) if op_next_raw_list \
            else torch.zeros((0, self.d_O), dtype=torch.float32, device=device)

        # ---- 机器原始特征 (d_M=5) ----
        machine_raw_list = []
        for m_id in range(self.config.num_machines):
            ms = machine_states.get(m_id)
            if ms:
                ep = self.config.machine_data.get(m_id, {}).get('energy_process', 0)
                # 统计该机器可加工的工序数
                n_proc = sum(1 for o in hdg.operation_nodes.values()
                             if m_id in o.machine_candidates)
                util = ms.total_processing_time / max(1e-6, cmax_norm)
                machine_raw_list.append([
                    ep / 5.0,
                    0.0,  # mek 等价（用户数据无此项，置零）
                    n_proc / max(1, total_ops),
                    ms.next_available_time / max(1e-6, cmax_norm),
                    min(1.0, util),
                ])
            else:
                machine_raw_list.append([0.0] * 5)
        machine_raw = torch.tensor(machine_raw_list, dtype=torch.float32, device=device)

        # ---- 工人原始特征 (d_W=5) ----
        worker_raw_list = []
        for w_id, w in hdg.worker_nodes.items():
            util = w.total_work_time / max(1e-6, cmax_norm)
            worker_raw_list.append([
                w.lambda_,
                w.current_fatigue,
                total_ops / max(1, total_ops),  # 所有工人可加工所有工序
                w.next_available_time / max(1e-6, cmax_norm),
                min(1.0, util),
            ])
        worker_raw = torch.tensor(worker_raw_list, dtype=torch.float32, device=device) \
            if worker_raw_list else torch.zeros((0, self.d_W), dtype=torch.float32, device=device)

        # ---- O-M 边特征 (d_OM=1): 预设加工时间 ----
        N_op = len(op_keys_list)
        N_machine = self.config.num_machines
        N_worker = len(hdg.worker_nodes)
        om_edges = torch.zeros(N_op, N_machine, self.d_OM, dtype=torch.float32, device=device)
        mask_om = torch.zeros(N_op, N_machine, dtype=torch.float32, device=device)
        for i, op_key in enumerate(op_keys_list):
            op = hdg.operation_nodes[op_key]
            for m_id in op.machine_candidates:
                if 0 <= m_id < N_machine:
                    om_edges[i, m_id, 0] = op.base_pt.get(m_id, 1.0) / 5.0
                    mask_om[i, m_id] = 1.0

        # ---- O-W 边特征 (d_OW=2): 疲劳因子 + lambda ----
        ow_edges = torch.zeros(N_op, N_worker, self.d_OW, dtype=torch.float32, device=device)
        mask_ow = torch.ones(N_op, N_worker, dtype=torch.float32, device=device)
        worker_ids = sorted(hdg.worker_nodes.keys())
        for j, w_id in enumerate(worker_ids):
            w = hdg.worker_nodes[w_id]
            ow_edges[:, j, 0] = 1.0 + w.current_fatigue
            ow_edges[:, j, 1] = w.lambda_

        # ---- 统计向量 ----
        workers = list(hdg.worker_nodes.values())
        completed_ops = sum(1 for o in hdg.operation_nodes.values() if o.is_scheduled)
        progress = completed_ops / max(1, total_ops)
        non_fatigue_ratio = sum(1 for w in workers if w.current_fatigue < self.config.fatigue_threshold) / max(1, len(workers))
        total_energy = sum(m.total_energy for m in machine_states.values()) if machine_states else 0.0
        avg_fatigue = float(np.mean([w.current_fatigue for w in workers])) if workers else 0.0
        energy_norm = self.agent_ref.energy_norm_factor if self.agent_ref else 1.0
        stats_vector = torch.tensor([
            progress, non_fatigue_ratio,
            current_cmax / max(1e-6, cmax_norm),
            total_energy / max(1e-6, energy_norm),
            avg_fatigue
        ], dtype=torch.float32, device=device)

        return {
            'op_raw': op_raw, 'op_prev_raw': op_prev_raw, 'op_next_raw': op_next_raw,
            'machine_raw': machine_raw, 'worker_raw': worker_raw,
            'om_edges': om_edges, 'ow_edges': ow_edges,
            'mask_om': mask_om, 'mask_ow': mask_ow,
            'mask_om_T': mask_om.transpose(0, 1),   # (N_machine, N_op)
            'mask_ow_T': mask_ow.transpose(0, 1),   # (N_worker, N_op)
            'stats_vector': stats_vector,
            'op_keys': op_keys_list,
        }

    def _compute_neighbor_avg(self, node_embeds, mask):
        """计算每个工序的邻居节点嵌入均值"""
        if node_embeds.size(0) == 0 or mask.size(0) == 0:
            return torch.zeros(mask.size(0), node_embeds.size(1), device=node_embeds.device)
        expanded = node_embeds.unsqueeze(0).expand(mask.size(0), -1, -1)
        mask_exp = mask.unsqueeze(-1)
        summed = (expanded * mask_exp).sum(dim=1)
        count = mask.sum(dim=1, keepdim=True).clamp(min=1)
        return summed / count

    def extract_raw_features(self, hdg, machine_states, device):
        """提取并返回分离梯度的原始特征（用于经验回放存储）"""
        raw = self._extract_raw_features(hdg, machine_states, device)
        return {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in raw.items()}

    def encode_raw_features(self, raw, device):
        """从存储的原始特征重新编码（带梯度），用于 PPO 更新"""
        op_raw = raw['op_raw'].to(device)
        op_prev_raw = raw['op_prev_raw'].to(device)
        op_next_raw = raw['op_next_raw'].to(device)
        machine_raw = raw['machine_raw'].to(device)
        worker_raw = raw['worker_raw'].to(device)
        om_edges = raw['om_edges'].to(device)
        ow_edges = raw['ow_edges'].to(device)
        mask_om = raw['mask_om'].to(device)
        mask_ow = raw['mask_ow'].to(device)
        mask_om_T = raw['mask_om_T'].to(device)
        mask_ow_T = raw['mask_ow_T'].to(device)

        if op_raw.size(0) == 0 or worker_raw.size(0) == 0:
            d = self.d
            zero_g = torch.zeros(3 * d + 5, device=device)
            return zero_g, zero_g, zero_g, zero_g

        # 初始投影
        x = self.op_proj(op_raw)              # (N_op, d)
        y = self.machine_proj(machine_raw)    # (N_machine, d)
        z = self.worker_proj(worker_raw)      # (N_worker, d)

        for i in range(self.num_layers):
            y = self.machine_layers[i](x, z, ow_edges, y, om_edges, mask_ow, mask_om_T)
            z = self.worker_layers[i](x, y, om_edges, z, ow_edges, mask_om, mask_ow_T)
            y_avg = self._compute_neighbor_avg(y, mask_om)
            z_avg = self._compute_neighbor_avg(z, mask_ow)
            x = self.op_layers[i](op_raw, op_prev_raw, op_next_raw, y_avg, z_avg)

        stats = raw['stats_vector'].to(device)
        g = torch.cat([x.mean(0), y.mean(0), z.mean(0), stats], dim=-1)
        return x, y, z, g

    def forward(self, hdg, machine_states, device):
        raw = self.extract_raw_features(hdg, machine_states, device)
        return self.encode_raw_features(raw, device)


# ============================================================================
#  第三部分：策略网络与价值网络（论文基于 MLP，非 HDGAT 的融合编码器）
# ============================================================================

class NHGNNActor(nn.Module):
    """
    论文策略网络：MLPθ[x'_ij || y'_k || z'_r || g_t]
    输入为每个候选动作的节点嵌入拼接 + 全局状态，输出动作概率分布
    """

    def __init__(self, config, d=64):
        super().__init__()
        self.config = config
        self.d = d
        self.action_input_dim = d * 3 + 3 * d + 5  # = 6d + 5
        self.d_hidden = 128
        self.mlp = nn.Sequential(
            nn.Linear(self.action_input_dim, self.d_hidden),
            nn.Tanh(),
            nn.Linear(self.d_hidden, self.d_hidden),
            nn.Tanh(),
            nn.Linear(self.d_hidden, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, action_embeddings, progress=1.0, mask=None):
        """
        action_embeddings: (B, K, action_input_dim)
        progress: 训练进度（用于温度退火）
        mask: (B, K) 有效动作掩码
        """
        B, K, _ = action_embeddings.shape
        flat = action_embeddings.view(B * K, -1)
        scores = self.mlp(flat).view(B, K)  # (B, K)

        # 温度退火（与论文一致：训练初期高温度鼓励探索，后期降低）
        temp = 0.2 + 0.8 * ((1.0 - float(progress)) ** 2)
        logits = scores / max(temp, 0.05)

        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e9)
        logits = logits - logits.max(dim=-1, keepdim=True)[0]
        probs = F.softmax(logits, dim=-1)
        return scores, probs

    def get_action(self, action_embeddings, progress=1.0, deterministic=False, mask=None):
        """单步推理接口"""
        if action_embeddings.dim() == 2:
            action_embeddings = action_embeddings.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)

        scores, probs = self(action_embeddings, progress=progress, mask=mask)
        probs = probs.squeeze(0)

        if deterministic:
            selected_idx = torch.argmax(probs).item()
            entropy = None
        else:
            probs_safe = (probs + 1e-9) / (probs + 1e-9).sum()
            dist = Categorical(probs=probs_safe)
            selected_idx = dist.sample().item()
            entropy = dist.entropy().mean().item()

        return selected_idx, probs[selected_idx].item(), entropy


class NHGNNCritic(nn.Module):
    """
    论文价值网络：MLPϕ[g_t] → 标量价值估计
    """

    def __init__(self, config, d=64):
        super().__init__()
        self.d_hidden = 128
        input_dim = 3 * d + 5  # global embedding dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, self.d_hidden),
            nn.Tanh(),
            nn.Linear(self.d_hidden, self.d_hidden),
            nn.Tanh(),
            nn.Linear(self.d_hidden, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                if m is self.mlp[-1]:
                    nn.init.orthogonal_(m.weight, gain=0.01)
                else:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, global_features):
        if global_features.dim() == 1:
            global_features = global_features.unsqueeze(0)
        return self.mlp(global_features).squeeze(-1)


# ============================================================================
#  第四部分：优先经验回放缓冲区（论文 PER 机制）
# ============================================================================

class PrioritizedReplayBuffer:
    """
    论文的优先经验回放：基于累计奖励的优先级采样
    优先级 pw = (1/cw) / Σ(1/cw)，cw 为样本累计奖励
    """

    def __init__(self, capacity=1024):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []

    def add(self, sample, cumulative_reward):
        """添加样本，优先级基于累计奖励的倒数"""
        if len(self.buffer) >= self.capacity:
            # 移除优先级最低的样本
            min_idx = int(np.argmin(self.priorities))
            self.buffer.pop(min_idx)
            self.priorities.pop(min_idx)
        self.buffer.append(sample)
        priority = 1.0 / (abs(cumulative_reward) + 1e-6)
        self.priorities.append(priority)

    def sample(self, batch_size, device):
        """按优先级采样"""
        if len(self.buffer) == 0:
            return {}
        sample_size = min(batch_size, len(self.buffer))
        total_priority = sum(self.priorities)
        probs = [p / total_priority for p in self.priorities]
        indices = np.random.choice(len(self.buffer), sample_size, replace=False, p=probs)

        batch = defaultdict(list)
        for idx in indices:
            sample = self.buffer[idx]
            for key, value in sample.items():
                batch[key].append(value)

        # 转换张量
        tensor_keys = ['global_return', 'global_advantage', 'prob', 'selected_idx']
        for key in tensor_keys:
            if key in batch:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32, device=device)
        if 'selected_idx' in batch:
            batch['selected_idx'] = batch['selected_idx'].long()

        # 显存优化：update_policy 会根据 raw + valid_actions 重新构造动作嵌入，
        # 因此这里不再把历史 action_embeddings padding 后搬到 GPU。
        # 该张量在大规模实例中会占用大量显存，且当前更新逻辑并不使用它。
        if 'action_embeddings' in batch:
            del batch['action_embeddings']

        # action_mask 保持为 CPU list，在 update_policy 中按单样本搬到 GPU。
        # 不做 batch 级 padding，避免额外显存占用。
        if 'action_mask' in batch:
            batch['action_mask'] = [m.detach().cpu() if isinstance(m, torch.Tensor) else torch.tensor(m, dtype=torch.float32)
                                    for m in batch['action_mask']]

        return dict(batch)

    def size(self):
        return len(self.buffer)

    def clear(self):
        self.buffer = []
        self.priorities = []


# ============================================================================
#  第五部分：PER-AC-PPO 智能体（核心对比算法）
# ============================================================================

class PERACPPOAgent:
    def __init__(self, config: Config, obj_weights=None):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[PER-AC-PPO] 使用设备: {self.device}")

        # 三目标权重 [Cmax, TEC, Favg]
        self.obj_weights = obj_weights or [0.5, 0.3, 0.2]

        self.hdg = HeterogeneousDisjunctiveGraph(config)
        self.machine_states = {}
        self.pending_dynamic_jobs = []
        self.total_episodes = 500
        self.trained_episodes = 0

        # ---- 论文核心组件（16G显存轻量版）----
        # 原论文核心仍保留：NHGNN + PER + adaptive clipping + PPO。
        # 为避免大规模实例在16G显存上OOM，将隐藏维度和层数设为轻量版本。
        self.d = 32
        self.nhgnn_layers = 1
        self.nhgnn = NHGNNEncoder(config, d=self.d, num_layers=self.nhgnn_layers).to(self.device)
        self.nhgnn.set_agent(self)
        self.actor = NHGNNActor(config, d=self.d).to(self.device)
        self.critic = NHGNNCritic(config, d=self.d).to(self.device)

        self.optimizer = torch.optim.Adam([
            {'params': self.nhgnn.parameters(), 'lr': config.hdgat_lr},
            {'params': self.actor.parameters(), 'lr': config.actor_lr},
            {'params': self.critic.parameters(), 'lr': config.critic_lr},
        ])

        self.buffer = PrioritizedReplayBuffer(capacity=256)

        self.fixed_rule_split_cap = 2


        self.config.batch_size = min(getattr(self.config, 'batch_size', 8), 8)
        self.config.update_epochs = min(getattr(self.config, 'update_epochs', 1), 1)

        # ---- 自适应裁剪参数（论文 Eq.21）----
        self.clip_coef_start = 0.2
        self.clip_coef_end = 0.05

        # ---- 训练记录 ----
        self.best_makespan = float('inf')
        self.best_energy = float('inf')
        self.best_fatigue_mean = float('inf')
        self.best_val_score = float('inf')
        self.best_val_metrics = None

        self.training_history = {
            'episode': [], 'reward': [], 'makespan': [], 'energy': [],
            'fatigue_mean': [], 'completion_rate': [], 'is_complete': [],
            'unscheduled_ops': [], 'actor_loss': [], 'critic_loss': [], 'policy_entropy': []
        }
        self.csv_log_path = "training_log_per_ac_ppo_paper_faithful.csv"
        self.current_instance_name = ""
        self._init_csv_logger()

        self._compute_dynamic_parameters()

    # ---------- 动态参数计算（与 V1_01.py 相同）----------

    def _compute_dynamic_parameters(self):
        self.total_ops = self.config.num_jobs * self.config.num_ops_per_job
        self._compute_normalization_params()
        self.max_steps_per_episode = self._compute_max_steps()

        self.max_action_candidates = 100
        self.action_cap_mode = 'stratified_random'

    def _compute_normalization_params(self):
        self.total_ops = self.config.num_jobs * self.config.num_ops_per_job
        total_candidates = 0
        total_ops_count = 0
        for job_id in range(self.config.num_jobs):
            if job_id in self.config.job_data:
                for op_id in range(self.config.num_ops_per_job):
                    if op_id in self.config.job_data[job_id]:
                        op_info = self.config.job_data[job_id][op_id]
                        total_candidates += len(op_info.get('machine_candidates', []))
                        total_ops_count += 1
        avg_candidates = total_candidates / max(1, total_ops_count)
        avg_op_time = 2.0
        per_op_total_time = avg_op_time * (10 / avg_candidates)
        worst_case_cmax = self.total_ops * per_op_total_time * 1.2
        avg_energy_per_time = 4.0
        worst_case_energy = worst_case_cmax * self.config.num_machines * avg_energy_per_time * 0.8

        self.time_norm_factor = max(10, worst_case_cmax * 0.1)
        self.cmax_norm_factor = max(100.0, worst_case_cmax)
        self.energy_norm_factor = max(2000.0, worst_case_energy)
        self.wait_norm_factor = max(10, worst_case_cmax * 0.1)
        self.fatigue_norm_factor = 1.0

        if getattr(self.config, 'ref_cmax', None) is not None:
            self.cmax_norm_factor = max(1.0, float(self.config.ref_cmax))
            self.time_norm_factor = max(10.0, self.cmax_norm_factor * 0.5)
            self.wait_norm_factor = max(10.0, self.cmax_norm_factor * 0.2)
        if getattr(self.config, 'ref_tec', None) is not None:
            self.energy_norm_factor = max(1.0, float(self.config.ref_tec))
        if getattr(self.config, 'ref_favg', None) is not None:
            self.fatigue_norm_factor = max(1e-6, float(self.config.ref_favg))

    def _compute_max_steps(self):
        base = self.config.num_jobs * self.config.num_machines * 2
        batch_factor = min(1.5, self.config.num_machines / self.config.num_ops_per_job)
        dynamic_factor = 1.3 if self.config.num_jobs_dynamic > 0 else 1.0
        return min(3000, max(100, int(base * batch_factor * dynamic_factor)))

    # ---------- 自适应裁剪系数（论文 Eq.21）----------

    def get_adaptive_clip(self, current_step, total_steps):
        """
        cp = exp(-t/T * ln(cp0/cp1)) * cp0
        从 cp0=0.2 指数衰减到 cp1=0.05
        """
        cp0, cp1 = self.clip_coef_start, self.clip_coef_end
        if total_steps <= 0:
            return cp0
        ratio = math.log(cp0 / cp1)
        cp = math.exp(-current_step / total_steps * ratio) * cp0
        return max(cp1, cp)

    # ---------- 环境初始化（与 V1_01.py 相同）----------

    def _initialize_from_json(self):
        self.hdg = HeterogeneousDisjunctiveGraph(self.config)
        self.machine_states = {}
        for m_id in range(self.config.num_machines):
            self.machine_states[m_id] = MachineState(m_id, self.config)
        for w_id, w_info in self.config.worker_data.items():
            worker = Worker(w_id, float(w_info.get('lambda', 0.05)), float(w_info.get('mu', 0.1)))
            worker.config = self.config
            self.hdg.add_worker(worker)
        self.pending_dynamic_jobs = []
        operations_by_job = {}
        all_operations = []
        for job_id in range(self.config.num_jobs_static):
            if job_id not in self.config.job_data:
                continue
            job_ops = []
            for op_id in range(self.config.num_ops_per_job):
                if op_id not in self.config.job_data[job_id]:
                    continue
                op_info = self.config.job_data[job_id][op_id]
                base_pt_int = {int(k): v for k, v in op_info.get('base_process_time', {}).items()}
                op = Operation(job_id, op_id, op_info.get('machine_candidates', []),
                               base_pt_int, self.config.total_pieces_per_job,
                               op_info.get('is_preemptive', False))
                job_ops.append(op)
                all_operations.append(op)
            operations_by_job[job_id] = job_ops
        for job_id, job_ops in operations_by_job.items():
            for i in range(len(job_ops)):
                if i > 0:
                    job_ops[i].prev_op = job_ops[i - 1]
                    job_ops[i].predecessors.append(job_ops[i - 1])
                if i < len(job_ops) - 1:
                    job_ops[i].next_op = job_ops[i + 1]
                    job_ops[i].successors.append(job_ops[i + 1])
        for op in all_operations:
            self.hdg.add_operation(op)
        for job_id in range(self.config.num_jobs_static, self.config.num_jobs):
            if job_id in self.config.job_data:
                trigger = self.config.job_ready_triggers.get(job_id, 0.5)
                self.pending_dynamic_jobs.append((job_id, trigger))
        self.hdg.update_operation_status()
        return len(operations_by_job) > 0

    def _check_dynamic_jobs_arrival(self):
        if not self.pending_dynamic_jobs:
            return False
        static_progress = self._get_static_progress()
        arrived = []
        for job_id, trigger in self.pending_dynamic_jobs:
            if static_progress >= trigger:
                arrived.append((job_id, trigger))
        arrived.sort(key=lambda x: x[1])
        for job_id, _ in arrived:
            self._add_job_to_graph(job_id)
            self.pending_dynamic_jobs.remove((job_id, _))
        return len(arrived) > 0

    def _get_static_progress(self):
        total = completed = 0
        for op in self.hdg.operation_nodes.values():
            if op.job_id < self.config.num_jobs_static:
                total += 1
                if op.is_scheduled and op.completed_pieces >= op.total_pieces:
                    completed += 1
        return completed / max(1, total)

    def _add_job_to_graph(self, job_id):
        job_data = self.config.job_data[job_id]
        job_ops = []
        for op_id in range(self.config.num_ops_per_job):
            if op_id not in job_data:
                continue
            op_info = job_data[op_id]
            base_pt_int = {int(k): v for k, v in op_info.get('base_process_time', {}).items()}
            op = Operation(job_id, op_id, op_info.get('machine_candidates', []),
                           base_pt_int, self.config.total_pieces_per_job,
                           op_info.get('is_preemptive', False))
            job_ops.append(op)
        for i in range(len(job_ops)):
            if i > 0:
                job_ops[i].prev_op = job_ops[i - 1]
                job_ops[i].predecessors.append(job_ops[i - 1])
            if i < len(job_ops) - 1:
                job_ops[i].next_op = job_ops[i + 1]
                job_ops[i].successors.append(job_ops[i + 1])
        for op in job_ops:
            self.hdg.add_operation(op)
        self.hdg.update_operation_status()

    # ---------- 动作生成（与 V1_01.py 相同的启发式逻辑）----------

    def _generate_valid_actions(self, hdg):

        hdg.update_operation_status()
        ready_ops = hdg.get_ready_operations()
        all_workers = list(self.hdg.worker_nodes.values())

        grouped_actions = []

        for op_key in ready_ops:
            op = hdg.operation_nodes[op_key]
            if op.completed_pieces >= op.total_pieces:
                continue
            if not op.machine_candidates:
                continue

            local_actions = []
            for machine_id in op.machine_candidates:
                if machine_id not in self.machine_states:
                    continue
                if self.machine_states[machine_id].is_busy:
                    continue

                for worker in all_workers:
                    # 仅排除超过疲劳阈值的工人；如果当前暂不可用，执行时通过 next_available_time 等待。
                    if worker.current_fatigue >= self.config.fatigue_threshold:
                        continue
                    local_actions.append((op_key, int(machine_id), int(worker.worker_id)))

            if local_actions:
                # 固定局部顺序，便于复现；采样时不使用任何目标值或价值评分。
                local_actions.sort(key=lambda x: (x[0], x[1], x[2]))
                grouped_actions.append(local_actions)

        if not grouped_actions:
            return [], None, None

        valid_actions = [a for group in grouped_actions for a in group]

        max_candidates = getattr(self, 'max_action_candidates', None)
        if isinstance(max_candidates, int) and max_candidates > 0 and len(valid_actions) > max_candidates:
            # 分层随机采样：每个就绪工序尽量保留若干个机器-工人组合，避免某些工序被完全丢弃。
            # 注意：不按 estimated_finish、energy、fatigue 或 value 排序。
            per_op_cap = max(1, max_candidates // max(1, len(grouped_actions)))
            selected_actions = []
            remaining_pool = []

            for local_actions in grouped_actions:
                if len(local_actions) <= per_op_cap:
                    selected_actions.extend(local_actions)
                else:
                    chosen = random.sample(local_actions, per_op_cap)
                    selected_actions.extend(chosen)
                    chosen_set = set(chosen)
                    remaining_pool.extend([a for a in local_actions if a not in chosen_set])

            # 若还有容量，从剩余动作中继续随机补足。
            remain_slots = max_candidates - len(selected_actions)
            if remain_slots > 0 and remaining_pool:
                selected_actions.extend(
                    random.sample(remaining_pool, min(remain_slots, len(remaining_pool)))
                )

            # 极端情况：就绪工序数量超过 max_candidates 时，最终再随机截断。
            if len(selected_actions) > max_candidates:
                selected_actions = random.sample(selected_actions, max_candidates)

            valid_actions = selected_actions

        # 不按调度目标排序；这里只做字典序排列，保证同一候选子集下动作索引稳定。
        valid_actions.sort(key=lambda x: (x[0], x[1], x[2]))
        return valid_actions, None, None

    # ---------- NHGNN 动作嵌入构建 ----------

    def _build_action_embeddings(self, valid_actions, action_info, op_embeds, machine_embeds,
                                  worker_embeds, global_emb, device, op_keys_override=None):

        if not valid_actions or op_embeds is None:
            return None

        op_keys = op_keys_override if op_keys_override is not None else [k for k, _ in self._iter_op_nodes()]
        op_key_to_idx = {k: i for i, k in enumerate(op_keys)}
        worker_ids = sorted(self.hdg.worker_nodes.keys())
        worker_id_to_idx = {w_id: i for i, w_id in enumerate(worker_ids)}

        embeddings = []
        for action in valid_actions:
            op_key, machine_id, worker_id = action

            op_idx = op_key_to_idx.get(op_key, None)
            if op_idx is None or op_idx >= op_embeds.size(0):
                op_e = torch.zeros(self.d, device=device)
            else:
                op_e = op_embeds[op_idx]

            if 0 <= int(machine_id) < machine_embeds.size(0):
                machine_e = machine_embeds[int(machine_id)]
            else:
                machine_e = torch.zeros(self.d, device=device)

            w_idx = worker_id_to_idx.get(worker_id, None)
            if w_idx is not None and w_idx < worker_embeds.size(0):
                worker_e = worker_embeds[w_idx]
            else:
                worker_e = torch.zeros(self.d, device=device)

            action_e = torch.cat([op_e, machine_e, worker_e, global_emb], dim=-1)
            embeddings.append(action_e)

        if not embeddings:
            return None
        action_tensor = torch.stack(embeddings, dim=0)  # (K, action_input_dim)
        return action_tensor

    # ---------- 三目标奖励函数（适配 Cmax、TEC、Favg）----------

    def _compute_reward(self, prev_state, current_state):

        r_cmax = prev_state['makespan'] - current_state['makespan']
        r_tec = prev_state['energy'] - current_state['energy']
        r_favg = prev_state['fatigue_mean'] - current_state['fatigue_mean']

        r_cmax = r_cmax / max(1e-6, self.cmax_norm_factor) * 100.0
        r_tec = r_tec / max(1e-6, self.energy_norm_factor) * 100.0
        r_favg = r_favg * 10.0

        r_cmax = float(np.clip(r_cmax, -2.0, 2.0))
        r_tec = float(np.clip(r_tec, -2.0, 2.0))
        r_favg = float(np.clip(r_favg, -2.0, 2.0))
        return [r_cmax, r_tec, r_favg]

    # ---------- 动作执行：a_t=(op_key, machine_id, worker_id) ----------

    def _execute_action(self, action):

        op_key, machine_id, worker_id = action
        if op_key not in self.hdg.operation_nodes:
            return
        op = self.hdg.operation_nodes[op_key]
        if machine_id not in op.machine_candidates:
            return
        if worker_id not in self.hdg.worker_nodes:
            return
        if machine_id not in self.machine_states:
            return

        remaining_pieces = op.total_pieces - op.completed_pieces
        if remaining_pieces <= 0:
            return

        prev_op_completion_time = op.get_prev_op_completion_time()

        # ---------- 1) 固定规则确定分批数 ----------
        split_cap = int(getattr(self, 'fixed_rule_split_cap', 1))
        split_cap = max(1, split_cap)

        # 主机器由智能体动作给定，额外机器按 SPT 从候选机器中补足。
        candidate_machines = [m for m in op.machine_candidates if m in self.machine_states]
        extra_machines = [m for m in candidate_machines if m != machine_id]
        extra_machines.sort(key=lambda m: op.base_pt.get(m, float('inf')))
        selected_machines = [machine_id] + extra_machines

        # 主工人由智能体动作给定，额外工人按低疲劳、早可用补足。
        eligible_workers = [
            w for w in self.hdg.worker_nodes.values()
            if w.current_fatigue < self.config.fatigue_threshold
        ]
        extra_workers = [w for w in eligible_workers if w.worker_id != worker_id]
        extra_workers.sort(key=lambda w: (w.current_fatigue, w.next_available_time, w.worker_id))
        selected_workers = [self.hdg.worker_nodes[worker_id]] + extra_workers

        B = min(split_cap, remaining_pieces, len(selected_machines), len(selected_workers))
        B = max(1, int(B))
        selected_machines = selected_machines[:B]
        selected_workers = selected_workers[:B]

        # ---------- 2) 按机器能力固定分配子批大小 ----------
        # 加工时间越短的机器承担越多件数；该规则是固定执行策略，不是可学习 B_k。
        capabilities = []
        for m_id in selected_machines:
            base_time = max(1e-6, float(op.base_pt.get(m_id, 1.0)))
            capabilities.append(1.0 / base_time)
        total_cap = sum(capabilities)

        if B == 1:
            batch_sizes = [remaining_pieces]
        elif total_cap <= 0:
            base_size = remaining_pieces // B
            batch_sizes = [base_size] * B
            for i in range(remaining_pieces % B):
                batch_sizes[i] += 1
        else:
            batch_sizes = []
            remain = remaining_pieces
            for i in range(B - 1):
                size = int(round(remaining_pieces * capabilities[i] / total_cap))
                size = max(1, min(size, remain - (B - i - 1)))
                batch_sizes.append(size)
                remain -= size
            batch_sizes.append(remain)

        # 修正极端 rounding 情况，保证总件数完全一致且无非正子批。
        batch_sizes = [max(1, int(s)) for s in batch_sizes]
        diff = remaining_pieces - sum(batch_sizes)
        batch_sizes[-1] += diff
        if batch_sizes[-1] <= 0:
            # 若修正后最后一批异常，则退化为均匀分配。
            base_size = remaining_pieces // B
            batch_sizes = [base_size] * B
            for i in range(remaining_pieces % B):
                batch_sizes[i] += 1

        machine_size_pairs = list(zip(selected_machines, batch_sizes))
        machine_size_pairs.sort(key=lambda x: x[1], reverse=True)
        selected_workers.sort(key=lambda w: (w.current_fatigue, w.next_available_time, w.worker_id))


        sub_complete_times = []
        for batch_id, ((m_id, batch_size), worker) in enumerate(zip(machine_size_pairs, selected_workers)):
            if batch_size <= 0:
                continue

            machine_state = self.machine_states[m_id]
            switch_time = 0.0
            if m_id in self.config.machine_data:
                switch_time = self.config.machine_data[m_id].get('switch_time', 0.0)

            machine_available = machine_state.next_available_time
            worker_available = worker.next_available_time
            if machine_state.is_first_use or machine_available == 0:
                machine_ready_time = machine_available
                actual_switch_time = 0.0
            else:
                machine_ready_time = machine_available + switch_time
                actual_switch_time = switch_time

            start_time = max(prev_op_completion_time, machine_ready_time, worker_available)
            idle_time = start_time - machine_available - actual_switch_time
            if idle_time > 0:
                machine_state.add_idle_energy(idle_time)
            if actual_switch_time > 0:
                machine_state.add_switch_energy(actual_switch_time)
            machine_state.is_first_use = False

            base_time = op.base_pt.get(m_id, 1.0)
            fatigue_factor = 1.0 + worker.current_fatigue
            actual_time = base_time * batch_size * fatigue_factor
            complete_time = start_time + actual_time

            record = ProcessingRecord(op_key, batch_id, batch_size, m_id, worker.worker_id)
            record.start_time = start_time
            record.complete_time = complete_time
            record.status = "processing"

            worker.assign_batch(record)
            machine_state.assign_batch(record, start_time, actual_time)
            worker.complete_batch(complete_time, start_time, actual_time)
            machine_state.complete_batch()

            sub_complete_times.append(complete_time)

        if not sub_complete_times:
            return

        op_complete_time = max(sub_complete_times)
        op.update_completion(remaining_pieces, op_complete_time)
        self.hdg.current_time = max(self.hdg.current_time, op_complete_time)
        self.hdg.update_operation_status()

    # ---------- 状态指标获取 ----------

    def _get_current_state_metrics(self):
        current_makespan = max(
            [o.complete_time for o in self.hdg.operation_nodes.values()] + [0]
        )
        current_energy = sum(m.total_energy for m in self.machine_states.values())
        all_fatigues = [w.current_fatigue for w in self.hdg.worker_nodes.values()]
        current_fatigue_mean = float(np.mean(all_fatigues)) if all_fatigues else 0.0
        return {
            'makespan': current_makespan,
            'energy': current_energy,
            'fatigue_mean': current_fatigue_mean,
        }

    def _get_completion_info(self):

        total_ops = len(self.hdg.operation_nodes)
        scheduled_ops = sum(
            1 for op in self.hdg.operation_nodes.values()
            if op.is_scheduled and op.completed_pieces >= op.total_pieces
        )
        unscheduled_ops = total_ops - scheduled_ops
        pending_dynamic_jobs = len(self.pending_dynamic_jobs)
        completion_rate = scheduled_ops / max(1, total_ops)
        is_complete = (unscheduled_ops == 0 and pending_dynamic_jobs == 0)
        return {
            'total_ops': total_ops,
            'scheduled_ops': scheduled_ops,
            'unscheduled_ops': unscheduled_ops,
            'pending_dynamic_jobs': pending_dynamic_jobs,
            'completion_rate': completion_rate,
            'is_complete': is_complete,
        }

    def _iter_op_nodes(self):
        return [(k, v) for k, v in self.hdg.operation_nodes.items()]

    # ---------- 单 episode 运行 ----------

    def run_one_episode(self, episode=1, deterministic=False, collect_experience=True):

        episode_experience = []
        current_episode_entropies = []
        total_episode_reward = 0.0
        prev_state = self._get_current_state_metrics()
        done = False
        step_count = 0
        current_progress = min(1.0, episode / max(1, self.total_episodes))

        with torch.no_grad():
            while not done:
                step_count += 1
                if self.pending_dynamic_jobs:
                    self._check_dynamic_jobs_arrival()

                raw = self.nhgnn.extract_raw_features(
                    self.hdg, self.machine_states, self.device
                )
                op_embeds, machine_embeds, worker_embeds, global_emb = \
                    self.nhgnn.encode_raw_features(raw, self.device)

                valid_actions, action_info, _ = self._generate_valid_actions(self.hdg)
                if not valid_actions:
                    all_finished = all(
                        o.is_scheduled for o in self.hdg.operation_nodes.values()
                    )
                    if (not self.pending_dynamic_jobs and all_finished) or \
                            step_count >= self.max_steps_per_episode:
                        break
                    continue

                action_tensor = self._build_action_embeddings(
                    valid_actions, action_info,
                    op_embeds, machine_embeds, worker_embeds,
                    global_emb, self.device,
                    op_keys_override=raw.get('op_keys', None)
                )
                if action_tensor is None:
                    break

                action_mask = torch.ones(
                    action_tensor.size(0), dtype=torch.float32, device=self.device
                )

                selected_idx, action_prob, entropy = self.actor.get_action(
                    action_tensor.unsqueeze(0),
                    progress=current_progress,
                    deterministic=deterministic,
                    mask=action_mask.unsqueeze(0),
                )
                if selected_idx is None:
                    break
                if entropy is not None:
                    current_episode_entropies.append(entropy)

                selected_action = valid_actions[selected_idx]
                self._execute_action(selected_action)

                current_state = self._get_current_state_metrics()
                step_reward_vec = self._compute_reward(prev_state, current_state)
                scalar_reward = sum(
                    w * r for w, r in zip(self.obj_weights, step_reward_vec)
                )
                total_episode_reward += scalar_reward

                value = self.critic(global_emb.unsqueeze(0)).item()

                if collect_experience:
                    episode_experience.append({
                        'raw': raw,
                        'valid_actions': list(valid_actions),
                        'action_embeddings': action_tensor.detach().cpu(),
                        'action_mask': action_mask.detach().cpu(),
                        'selected_idx': selected_idx,
                        'prob': action_prob,
                        'reward_vec': step_reward_vec,
                        'scalar_reward': scalar_reward,
                        'value': value,
                        'global_emb': global_emb.detach().cpu(),
                        'done': False,
                    })

                prev_state = current_state
                all_finished = all(
                    o.is_scheduled for o in self.hdg.operation_nodes.values()
                )
                done = all_finished and (not self.pending_dynamic_jobs)
                if step_count >= self.max_steps_per_episode:
                    break

        if episode_experience:
            episode_experience[-1]['done'] = True

        metrics = self._get_current_state_metrics()
        metrics.update(self._get_completion_info())
        metrics['reward'] = total_episode_reward
        metrics['entropy'] = float(np.mean(current_episode_entropies)) \
            if current_episode_entropies else 0.0
        metrics['step_count'] = step_count
        metrics['max_steps'] = self.max_steps_per_episode
        metrics['termination_reason'] = (
            'complete' if metrics['is_complete']
            else 'max_steps' if step_count >= self.max_steps_per_episode
            else 'no_valid_action_or_break'
        )

        if not metrics['is_complete']:
            print(
                f"[警告] Episode 未完整调度 | "
                f"scheduled={metrics['scheduled_ops']}/{metrics['total_ops']} | "
                f"unscheduled={metrics['unscheduled_ops']} | "
                f"pending_dynamic={metrics['pending_dynamic_jobs']} | "
                f"rate={metrics['completion_rate']:.2%} | "
                f"steps={step_count}/{self.max_steps_per_episode} | "
                f"reason={metrics['termination_reason']}"
            )

        return episode_experience, metrics

    # ---------- PPO 更新（自适应裁剪 + PER）----------

    def update_policy(self, episode=1):
        """显存友好版 PPO 更新：逐样本反向传播，避免一次性堆叠 NHGNN 计算图。"""
        update_info = {'actor_loss': None, 'critic_loss': None, 'entropy': None}
        current_progress = min(1.0, episode / max(1, self.total_episodes))
        ce = self.config.entropy_coef * max(0.1, (1.0 - current_progress))
        adaptive_clip = self.get_adaptive_clip(episode, max(1, self.total_episodes))

        for _ in range(self.config.update_epochs):
            batch = self.buffer.sample(self.config.batch_size, self.device)
            if not batch or 'raw' not in batch:
                break

            B_size = len(batch['raw'])
            if B_size == 0:
                break

            # 先构造标量 advantage，并做轻量归一化。
            advantages_all = batch['global_advantage']
            if advantages_all.dim() > 1:
                w = torch.tensor(self.obj_weights, dtype=torch.float32, device=self.device)
                advantages_all = (advantages_all * w).sum(dim=-1)
            if advantages_all.numel() > 1:
                advantages_all = (advantages_all - advantages_all.mean()) / (advantages_all.std(unbiased=False) + 1e-8)

            self.optimizer.zero_grad(set_to_none=True)
            loss_count = 0
            actor_losses, critic_losses, entropies = [], [], []

            for i in range(B_size):
                raw_i = batch['raw'][i]
                valid_actions_i = batch.get('valid_actions', [None] * B_size)[i]
                if not valid_actions_i:
                    continue

                # 单样本编码，避免 global_features_list 堆叠导致多个大图同时驻留显存。
                op_e, mach_e, work_e, g_i = self.nhgnn.encode_raw_features(raw_i, self.device)

                act_tensor_i = self._build_action_embeddings(
                    valid_actions_i, None, op_e, mach_e, work_e, g_i, self.device,
                    op_keys_override=raw_i.get('op_keys', None)
                )
                if act_tensor_i is None or act_tensor_i.size(0) == 0:
                    del op_e, mach_e, work_e, g_i
                    continue

                if 'action_mask' in batch and i < len(batch['action_mask']):
                    mask_i = batch['action_mask'][i].to(self.device)
                    if mask_i.size(0) != act_tensor_i.size(0):
                        mask_i = torch.ones(act_tensor_i.size(0), dtype=torch.float32, device=self.device)
                else:
                    mask_i = torch.ones(act_tensor_i.size(0), dtype=torch.float32, device=self.device)

                _, probs_i = self.actor(
                    act_tensor_i.unsqueeze(0),
                    progress=current_progress,
                    mask=mask_i.unsqueeze(0),
                )
                probs_i = probs_i.squeeze(0)

                sel = int(batch['selected_idx'][i].item())
                if sel >= probs_i.size(0):
                    del op_e, mach_e, work_e, g_i, act_tensor_i, mask_i, probs_i
                    continue

                new_prob = probs_i[sel]
                old_prob = batch['prob'][i]
                ratio = new_prob / (old_prob + 1e-8)

                adv = advantages_all[i].detach()
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - adaptive_clip, 1 + adaptive_clip) * adv
                policy_loss = -torch.min(surr1, surr2)

                value_pred = self.critic(g_i)
                value_target = batch['global_return'][i]
                critic_loss = F.mse_loss(value_pred.view_as(value_target), value_target)

                log_probs = torch.log(probs_i + 1e-9)
                entropy = -torch.sum(probs_i * log_probs * mask_i, dim=-1)

                loss = policy_loss + self.config.value_coef * critic_loss - ce * entropy
                # 按样本数缩放，等价于 mini-batch 平均，但不会保留所有样本计算图。
                (loss / max(1, B_size)).backward()

                actor_losses.append(float(policy_loss.detach().cpu().item()))
                critic_losses.append(float(critic_loss.detach().cpu().item()))
                entropies.append(float(entropy.detach().cpu().item()))
                loss_count += 1

                del op_e, mach_e, work_e, g_i, act_tensor_i, mask_i, probs_i
                del new_prob, old_prob, ratio, policy_loss, critic_loss, entropy, loss

            if loss_count == 0:
                self.optimizer.zero_grad(set_to_none=True)
                continue

            torch.nn.utils.clip_grad_norm_(
                list(self.nhgnn.parameters()) +
                list(self.actor.parameters()) +
                list(self.critic.parameters()),
                0.5,
            )
            self.optimizer.step()

            update_info = {
                'actor_loss': float(np.mean(actor_losses)) if actor_losses else None,
                'critic_loss': float(np.mean(critic_losses)) if critic_losses else None,
                'entropy': float(np.mean(entropies)) if entropies else None,
            }
            if update_info['actor_loss'] is not None:
                self.training_history['actor_loss'].append(update_info['actor_loss'])
            if update_info['critic_loss'] is not None:
                self.training_history['critic_loss'].append(update_info['critic_loss'])

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        return update_info

    # ---------- 实例切换 ----------

    def reset_instance(self, json_path):
        new_config = Config(json_path=json_path)
        self.config = new_config
        self.nhgnn.config = new_config
        self.current_instance_name = os.path.basename(json_path)
        self._compute_dynamic_parameters()
        self._initialize_from_json()

    # ---------- 经验回放存储 ----------

    def _store_episode_to_buffer(self, episode_experience):
        if not episode_experience:
            return

        T = len(episode_experience)
        mc_returns = []
        running_return = 0.0
        for t in reversed(range(T)):
            exp = episode_experience[t]
            running_return = exp['scalar_reward'] + self.config.gamma * running_return
            mc_returns.insert(0, running_return)

        advantages = []
        for t in range(T):
            adv = mc_returns[t] - episode_experience[t]['value']
            advantages.append(adv)

        for t in range(T):
            exp = episode_experience[t]
            sample = {
                'raw': exp['raw'],
                'valid_actions': exp.get('valid_actions', []),
                'action_embeddings': exp['action_embeddings'],
                'action_mask': exp['action_mask'],
                'selected_idx': exp['selected_idx'],
                'prob': exp['prob'],
                'global_return': mc_returns[t],
                'global_advantage': advantages[t],
            }
            self.buffer.add(sample, cumulative_reward=mc_returns[t])

    # ---------- 验证评估（与 V1_01.py 完全相同接口）----------

    def evaluate_validation(self, val_files, eval_runs=1):
        cmax_list, tec_list, favg_list = [], [], []
        complete_list, completion_rate_list, unscheduled_list = [], [], []
        ref_cmax_list, ref_tec_list, ref_favg_list = [], [], []
        cmax_ratio_list, tec_ratio_list, favg_ratio_list = [], [], []

        for json_path in val_files:
            run_cmax, run_tec, run_favg = [], [], []
            for _ in range(eval_runs):
                self.reset_instance(json_path)
                _, metrics = self.run_one_episode(
                    episode=0, deterministic=True, collect_experience=False
                )
                run_cmax.append(metrics['makespan'])
                run_tec.append(metrics['energy'])
                run_favg.append(metrics['fatigue_mean'])
                complete_list.append(1.0 if metrics.get('is_complete', False) else 0.0)
                completion_rate_list.append(float(metrics.get('completion_rate', 0.0)))
                unscheduled_list.append(float(metrics.get('unscheduled_ops', 0)))

            mean_cmax_i = float(np.mean(run_cmax)) if run_cmax else float('inf')
            mean_tec_i = float(np.mean(run_tec)) if run_tec else float('inf')
            mean_favg_i = float(np.mean(run_favg)) if run_favg else float('inf')
            cmax_list.append(mean_cmax_i)
            tec_list.append(mean_tec_i)
            favg_list.append(mean_favg_i)

            ref_cmax = getattr(self.config, 'ref_cmax', None)
            ref_tec = getattr(self.config, 'ref_tec', None)
            ref_favg = getattr(self.config, 'ref_favg', None)
            if ref_cmax is not None:
                ref_cmax = float(ref_cmax)
                ref_cmax_list.append(ref_cmax)
                cmax_ratio_list.append(mean_cmax_i / max(1e-6, ref_cmax))
            if ref_tec is not None:
                ref_tec = float(ref_tec)
                ref_tec_list.append(ref_tec)
                tec_ratio_list.append(mean_tec_i / max(1e-6, ref_tec))
            if ref_favg is not None:
                ref_favg = float(ref_favg)
                ref_favg_list.append(ref_favg)
                favg_ratio_list.append(mean_favg_i / max(1e-6, ref_favg))

        return {
            'mean_cmax': float(np.mean(cmax_list)) if cmax_list else float('inf'),
            'mean_tec': float(np.mean(tec_list)) if tec_list else float('inf'),
            'mean_favg': float(np.mean(favg_list)) if favg_list else float('inf'),
            'mean_ref_cmax': float(np.mean(ref_cmax_list)) if ref_cmax_list else None,
            'mean_ref_tec': float(np.mean(ref_tec_list)) if ref_tec_list else None,
            'mean_ref_favg': float(np.mean(ref_favg_list)) if ref_favg_list else None,
            'mean_cmax_ratio': float(np.mean(cmax_ratio_list)) if cmax_ratio_list else None,
            'mean_tec_ratio': float(np.mean(tec_ratio_list)) if tec_ratio_list else None,
            'mean_favg_ratio': float(np.mean(favg_ratio_list)) if favg_ratio_list else None,
            'std_cmax_ratio': float(np.std(cmax_ratio_list)) if cmax_ratio_list else 0.0,
            'std_tec_ratio': float(np.std(tec_ratio_list)) if tec_ratio_list else 0.0,
            'std_favg_ratio': float(np.std(favg_ratio_list)) if favg_ratio_list else 0.0,
            'complete_rate': float(np.mean(complete_list)) if complete_list else 0.0,
            'mean_completion_rate': float(np.mean(completion_rate_list)) if completion_rate_list else 0.0,
            'mean_unscheduled_ops': float(np.mean(unscheduled_list)) if unscheduled_list else 0.0,
        }

    def compute_val_score(self, val_metrics):
        cmax_ratio = val_metrics.get('mean_cmax_ratio')
        tec_ratio = val_metrics.get('mean_tec_ratio')
        favg_ratio = val_metrics.get('mean_favg_ratio')
        if cmax_ratio is None:
            cmax_ratio = val_metrics['mean_cmax'] / max(1e-6, self.cmax_norm_factor)
        if tec_ratio is None:
            tec_ratio = val_metrics['mean_tec'] / max(1e-6, self.energy_norm_factor)
        if favg_ratio is None:
            favg_ratio = val_metrics['mean_favg'] / 1.0

        stability_penalty = (
            val_metrics.get('std_cmax_ratio', 0.0)
            + val_metrics.get('std_tec_ratio', 0.0)
            + val_metrics.get('std_favg_ratio', 0.0)
        ) / 3.0
        cmax_bad = max(0.0, cmax_ratio - 1.10)
        tec_bad = max(0.0, tec_ratio - 1.10)
        favg_bad = max(0.0, favg_ratio - 1.10)
        bad_penalty = 0.60 * cmax_bad + 0.25 * tec_bad + 0.15 * favg_bad

        incomplete_penalty = max(0.0, 1.0 - val_metrics.get('complete_rate', 1.0))
        mean_unscheduled_penalty = 0.01 * val_metrics.get('mean_unscheduled_ops', 0.0)

        score = (
            0.50 * cmax_ratio + 0.30 * tec_ratio + 0.20 * favg_ratio
            + 0.05 * stability_penalty + bad_penalty
            + incomplete_penalty + mean_unscheduled_penalty
        )
        return float(score)

    def is_better_validation(self, val_score, val_metrics, min_improve=0.003, tolerance=0.00):
        if self.best_val_metrics is None or not np.isfinite(self.best_val_score):
            return True
        score_improved = val_score < self.best_val_score * (1.0 - min_improve)
        if not score_improved:
            return False
        old = self.best_val_metrics
        return (val_metrics['mean_cmax'] <= old['mean_cmax'] * (1.0 + tolerance)
                and val_metrics['mean_tec'] <= old['mean_tec'] * (1.0 + tolerance)
                and val_metrics['mean_favg'] <= old['mean_favg'] * (1.0 + tolerance))

    # ---------- CSV 日志 ----------

    def _init_csv_logger(self):
        if not os.path.exists(self.csv_log_path):
            with open(self.csv_log_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'instance_name', 'episode', 'duration_seconds',
                    'cmax', 'total_energy', 'avg_fatigue', 'total_reward',
                    'total_ops', 'scheduled_ops', 'unscheduled_ops',
                    'pending_dynamic_jobs', 'completion_rate', 'is_complete',
                    'step_count', 'max_steps', 'termination_reason',
                ])

    def _log_episode_to_csv(self, episode, duration, metrics):
        try:
            with open(self.csv_log_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.current_instance_name, episode,
                    f"{duration:.2f}",
                    f"{metrics['makespan']:.2f}",
                    f"{metrics['energy']:.2f}",
                    f"{metrics['fatigue_mean']:.4f}",
                    f"{metrics.get('reward', 0.0):.4f}",
                    metrics.get('total_ops', 0),
                    metrics.get('scheduled_ops', 0),
                    metrics.get('unscheduled_ops', 0),
                    metrics.get('pending_dynamic_jobs', 0),
                    f"{metrics.get('completion_rate', 0.0):.6f}",
                    int(bool(metrics.get('is_complete', False))),
                    metrics.get('step_count', 0),
                    metrics.get('max_steps', self.max_steps_per_episode),
                    metrics.get('termination_reason', ''),
                ])
        except Exception as e:
            print(f"CSV写入失败: {e}")

    # ---------- 多实例训练 ----------

    def train_multi_instance(self, train_files, val_files=None, total_episodes=50000,
                             update_interval=16, eval_interval=1000, save_interval=1000,
                             save_dir='saved_models_per_ac_ppo', debug_interval=10):
        os.makedirs(save_dir, exist_ok=True)

        old_episode_count = int(getattr(self, 'trained_episodes', 0))
        if old_episode_count <= 0 and self.training_history.get('episode'):
            old_episode_count = int(max(self.training_history['episode']))
        final_episode = old_episode_count + total_episodes
        self.total_episodes = final_episode

        print(f"\n🚀 [PER-AC-PPO] 多实例训练开始 | 训练实例数={len(train_files)} | "
              f"已训练={old_episode_count} | 本次新增={total_episodes} | "
              f"目标={final_episode}")
        print(f"   三目标权重: Cmax={self.obj_weights[0]}, "
              f"TEC={self.obj_weights[1]}, Favg={self.obj_weights[2]}")
        print(f"   NHGNN: d={self.d}, layers={self.nhgnn_layers}")
        print(f"   自适应裁剪: {self.clip_coef_start} -> {self.clip_coef_end}")
        print(f"   PER 缓冲区容量: {self.buffer.capacity}")
        print("   动作空间: (O, M, W)，不使用 B_k/子批节点/价值导向Top-K剪枝")
        print(f"   工程候选动作上限: {self.max_action_candidates} | 采样方式: 分层随机采样，不按目标值或价值排序")

        for episode in range(old_episode_count + 1, final_episode + 1):
            ep_start = time.time()
            json_path = random.choice(train_files)
            self.reset_instance(json_path)

            episode_experience, metrics = self.run_one_episode(
                episode=episode, deterministic=False, collect_experience=True
            )

            if episode_experience:
                self._store_episode_to_buffer(episode_experience)

            if self.buffer.size() >= self.config.batch_size and \
                    episode % update_interval == 0:
                self.update_policy(episode=episode)
                if episode % (update_interval * 10) == 0:
                    self.buffer.clear()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            self.training_history['episode'].append(episode)
            self.training_history['reward'].append(metrics.get('reward', 0.0))
            self.training_history['makespan'].append(metrics['makespan'])
            self.training_history['energy'].append(metrics['energy'])
            self.training_history['fatigue_mean'].append(metrics['fatigue_mean'])
            self.training_history['completion_rate'].append(metrics.get('completion_rate', 0.0))
            self.training_history['is_complete'].append(bool(metrics.get('is_complete', False)))
            self.training_history['unscheduled_ops'].append(metrics.get('unscheduled_ops', 0))
            self.training_history['policy_entropy'].append(metrics.get('entropy', 0.0))
            self.trained_episodes = episode

            self._log_episode_to_csv(
                episode, time.time() - ep_start, metrics
            )

            if episode % debug_interval == 0:
                clip_now = self.get_adaptive_clip(
                    self.trained_episodes, max(1, self.total_episodes)
                )
                print(f"[PER-AC-PPO] Ep {episode:5d} | {self.current_instance_name} | "
                      f"Cmax={metrics['makespan']:.1f} | "
                      f"TEC={metrics['energy']:.1f} | "
                      f"Favg={metrics['fatigue_mean']:.3f} | "
                      f"Complete={metrics.get('is_complete', False)} | "
                      f"Rate={metrics.get('completion_rate', 0.0):.2%} | "
                      f"Unsch={metrics.get('unscheduled_ops', 0)} | "
                      f"PendingDyn={metrics.get('pending_dynamic_jobs', 0)} | "
                      f"Steps={metrics.get('step_count', 0)}/{metrics.get('max_steps', self.max_steps_per_episode)} | "
                      f"R={metrics.get('reward', 0.0):.3f} | "
                      f"clip={clip_now:.3f} | "
                      f"PER_size={self.buffer.size()}")

            if episode % save_interval == 0:
                self.save_models(save_dir, 'latest')

            if val_files is not None and episode % eval_interval == 0:
                val_metrics = self.evaluate_validation(val_files, eval_runs=1)
                val_score = self.compute_val_score(val_metrics)
                print(f"[验证] Ep {episode} | score={val_score:.6f} | {val_metrics}")
                if self.is_better_validation(val_score, val_metrics):
                    self.best_val_score = val_score
                    self.best_val_metrics = val_metrics
                    self.save_models(save_dir, 'best_val')
                else:
                    print(f"[验证] Ep {episode} 未更新 best_val | "
                          f"当前best={self.best_val_score:.6f}")

            gc.collect()

        self.save_models(save_dir, 'final')
        gc.collect()

    # ---------- 模型保存/加载 ----------

    def save_models(self, path="saved_models_per_ac_ppo/", episode_num=None):
        os.makedirs(path, exist_ok=True)
        if episode_num is None:
            episode_num = "latest"
        save_path = os.path.join(path, f"per_ac_ppo_episode_{episode_num}.pth")
        torch.save({
            'nhgnn_state_dict': self.nhgnn.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'obj_weights': self.obj_weights,
            'trained_episodes': self.trained_episodes,
            'best_val_score': self.best_val_score,
            'best_val_metrics': self.best_val_metrics,
        }, save_path)
        print(f"[PER-AC-PPO] 模型已保存: {save_path} (已训练 {self.trained_episodes} 轮)")

    def load_models(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.nhgnn.load_state_dict(checkpoint['nhgnn_state_dict'])
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'obj_weights' in checkpoint:
            self.obj_weights = checkpoint['obj_weights']
        if 'trained_episodes' in checkpoint:
            self.trained_episodes = int(checkpoint['trained_episodes'])
        if 'best_val_score' in checkpoint:
            self.best_val_score = checkpoint['best_val_score']
        if 'best_val_metrics' in checkpoint:
            self.best_val_metrics = checkpoint['best_val_metrics']
        print(f"[PER-AC-PPO] 模型已加载: {path} (已训练 {self.trained_episodes} 轮)")

# ============================================================================
#  第六部分：主入口
# ============================================================================

if __name__ == "__main__":
    train_dir = "../../TestData/train1"
    val_dir = "../../TestData/yanzheng1"
    save_dir = "saved_models_per_ac_ppo"

    train_files = sorted(glob.glob(os.path.join(train_dir, "*.json")))[:100]
    if not train_files:
        raise FileNotFoundError(f"未找到训练实例: {train_dir}")
    val_files = sorted(glob.glob(os.path.join(val_dir, "*.json"))) \
        if val_dir and os.path.exists(val_dir) else None

    print(f"训练实例数: {len(train_files)}")
    print(f"验证实例数: {len(val_files) if val_files else 0}")

    init_config = Config(json_path=train_files[0])
    agent = PERACPPOAgent(init_config, obj_weights=[0.5, 0.3, 0.2])

    # 尝试加载已有模型
    latest_path = os.path.join(save_dir, "per_ac_ppo_episode_best_val.pth")
    if os.path.exists(latest_path):
        print(f"[载入已有模型] {latest_path}")
        agent.load_models(latest_path)

    agent.train_multi_instance(
        train_files=train_files,
        val_files=val_files,
        total_episodes=5000,
        update_interval=16,
        eval_interval=500,
        save_interval=500,
        save_dir=save_dir,
        debug_interval=10,
    )
