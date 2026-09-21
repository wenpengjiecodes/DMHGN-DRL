#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapted MHDQN for Dual-Resource Constrained Flexible Job Shop with Lot-Streaming
Target: 3 Objectives (Makespan, Energy, Fatigue), Dynamic Job Arrival based on Progress
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, defaultdict
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from copy import deepcopy
import time


# ==========================================
# 1. 数据结构与配置类
# ==========================================

class Config:
    """从JSON加载配置"""

    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)

        # 全局参数
        g_conf = data.get('global_config', {})
        self.num_machines = g_conf.get('num_machines', 10)
        self.num_workers = g_conf.get('num_workers', 5)
        self.num_jobs = g_conf.get('num_jobs', 10)
        self.num_jobs_static = g_conf.get('num_jobs_static', 8)
        self.num_ops_per_job = g_conf.get('num_ops_per_job', 5)
        self.total_pieces_per_job = g_conf.get('total_pieces_per_job', 10)
        self.fatigue_threshold = g_conf.get('fatigue_threshold', 0.8)

        # 参考解指标：用于验证集 best model 选择时进行归一化。
        # 如果 JSON 中没有 reference_solution，则验证函数会使用规模估计值作为兜底归一化尺度。
        reference = data.get('reference_solution', {})
        self.ref_cmax = reference.get('Cmax', None)
        self.ref_tec = reference.get('TEC', reference.get('Total_Energy', None))
        self.ref_favg = reference.get('Favg', reference.get('Avg_Fatigue', None))

        # 机器数据
        self.machine_data = {}
        for k, v in data.get('machine_static_data', {}).items():
            mid = int(k.split('_')[1])
            self.machine_data[mid] = v

        # 工人数据
        self.worker_data = {}
        for k, v in data.get('worker_static_data', {}).items():
            wid = int(k.split('_')[1])
            self.worker_data[wid] = v

        # 工件数据
        self.job_data = {}
        self.job_ready_triggers = {}
        for jk, jv in data.get('job_operation_static_data', {}).items():
            jid = int(jk.split('_')[1])
            self.job_data[jid] = {}
            for ok, ov in jv.items():
                oid = int(ok.split('_')[1])
                # 处理 base_process_time 键值对
                if 'base_process_time' in ov:
                    bpt = {int(m): t for m, t in ov['base_process_time'].items()}
                    ov['base_process_time'] = bpt
                self.job_data[jid][oid] = ov

            # 检查动态触发
            if 'op_0' in jv and 'ready_trigger' in jv['op_0']:
                self.job_ready_triggers[jid] = jv['op_0']['ready_trigger']


@dataclass
class Operation:
    job_id: int
    op_id: int
    machine_candidates: List[int]
    base_pt: Dict[int, float]
    total_pieces: int
    completed_pieces: int = 0
    is_scheduled: bool = False
    complete_time: float = 0.0

    def is_completed(self):
        return self.completed_pieces >= self.total_pieces


@dataclass
class Worker:
    worker_id: int
    lambda_: float
    mu: float
    current_fatigue: float = 0.0
    next_available_time: float = 0.0
    is_busy: bool = False

    def update_fatigue(self, duration, start_time):
        # 加工过程中疲劳度增加
        increase = (1 - self.current_fatigue) * (1 - np.exp(-self.lambda_ * duration))
        self.current_fatigue = min(1.0, self.current_fatigue + increase)
        self.next_available_time = start_time + duration

        # 检查是否需要强制休息 (简化处理：达到阈值则休息)
        if self.current_fatigue >= 0.8:  # 阈值硬编码或从config读取
            recovery_time = self._calc_recovery(0.5)  # 恢复到0.5
            self.next_available_time += recovery_time
            self.current_fatigue = 0.5

    def _calc_recovery(self, target):
        if self.current_fatigue <= target: return 0.0
        return -np.log(target / self.current_fatigue) / self.mu


@dataclass
class Machine:
    machine_id: int
    next_available_time: float = 0.0
    total_process_time: float = 0.0
    total_idle_energy: float = 0.0
    total_switch_energy: float = 0.0
    total_process_energy: float = 0.0
    is_first_use: bool = True

    def add_energy(self, idle_t, switch_t, process_t, config):
        d = config.machine_data.get(self.machine_id, {})
        self.total_idle_energy += idle_t * d.get('energy_idle', 0)
        self.total_switch_energy += switch_t * d.get('energy_switch', 0)
        self.total_process_energy += process_t * d.get('energy_process', 0)

    @property
    def total_energy(self):
        return self.total_idle_energy + self.total_switch_energy + self.total_process_energy


# ==========================================
# 2. 环境与状态管理
# ==========================================

class SchedulingState:
    def __init__(self, config: Config):
        self.config = config
        self.operations = {}  # (job_id, op_id) -> Operation
        self.machines = {}  # machine_id -> Machine
        self.workers = {}  # worker_id -> Worker
        self.current_time = 0.0
        self.completed_jobs = set()

        # 动态工件队列: (job_id, trigger_ratio)
        self.pending_dynamic_jobs = []

        # 初始化机器和工人
        for m_id in range(config.num_machines):
            self.machines[m_id] = Machine(m_id)
        for w_id, w_info in config.worker_data.items():
            self.workers[w_id] = Worker(w_id, w_info['lambda'], w_info['mu'])

    def get_ready_operations(self):
        """获取就绪且未完成的工序"""
        ready = []
        for key, op in self.operations.items():
            if op.is_scheduled: continue
            # 检查前序约束
            prev_op_key = (op.job_id, op.op_id - 1)
            if prev_op_key in self.operations:
                if not self.operations[prev_op_key].is_scheduled:
                    continue
            ready.append(key)
        return ready

    def get_static_progress(self):
        """计算静态工件完成进度"""
        total_ops = self.config.num_jobs_static * self.config.num_ops_per_job
        completed = sum(1 for jid in range(self.config.num_jobs_static)
                        for oid in range(self.config.num_ops_per_job)
                        if self.operations.get((jid, oid)) and self.operations[(jid, oid)].is_scheduled)
        return completed / total_ops if total_ops > 0 else 0.0

    def check_dynamic_arrival(self):
        """检查动态工件到达"""
        progress = self.get_static_progress()
        arrived = []
        for item in self.pending_dynamic_jobs[:]:
            jid, trigger = item
            if progress >= trigger:
                arrived.append(jid)
                self.pending_dynamic_jobs.remove(item)
        return arrived


class MODFJSPEnv:
    def __init__(self, config: Config):
        self.config = config

    def reset(self) -> SchedulingState:
        state = SchedulingState(self.config)

        # 加载静态工件
        for jid in range(self.config.num_jobs_static):
            if jid not in self.config.job_data: continue
            for oid in range(self.config.num_ops_per_job):
                if oid not in self.config.job_data[jid]: continue
                info = self.config.job_data[jid][oid]
                op = Operation(
                    job_id=jid, op_id=oid,
                    machine_candidates=info['machine_candidates'],
                    base_pt=info['base_process_time'],
                    total_pieces=self.config.total_pieces_per_job
                )
                state.operations[(jid, oid)] = op

        # 加载动态工件队列
        for jid in range(self.config.num_jobs_static, self.config.num_jobs):
            if jid in self.config.job_ready_triggers:
                state.pending_dynamic_jobs.append((jid, self.config.job_ready_triggers[jid]))
            elif jid in self.config.job_data:
                # 如果没有触发比例，默认0.5
                state.pending_dynamic_jobs.append((jid, 0.5))

        return state

    def add_job_to_state(self, state: SchedulingState, job_id: int):
        """将动态工件加入状态"""
        if job_id not in self.config.job_data: return
        for oid in range(self.config.num_ops_per_job):
            if oid not in self.config.job_data[job_id]: continue
            info = self.config.job_data[job_id][oid]
            op = Operation(
                job_id=job_id, op_id=oid,
                machine_candidates=info['machine_candidates'],
                base_pt=info['base_process_time'],
                total_pieces=self.config.total_pieces_per_job
            )
            state.operations[(job_id, oid)] = op
            # print(f"Job {job_id} arrived at time {state.current_time:.2f}")

    def step(self, state: SchedulingState, action_dict: Dict):
        """
        执行一步调度
        action_dict: {
            'op_key': (jid, oid),
            'batch_size': int,
            'machines': [mid1, ...],
            'workers': [wid1, ...]
        }
        """
        op_key = action_dict['op_key']
        B_k = action_dict['batch_size']
        machine_ids = action_dict['machines']
        worker_ids = action_dict['workers']

        op = state.operations[op_key]
        remaining = op.total_pieces - op.completed_pieces

        # 分批计算
        # 简单策略：平均分配 (实际可按机器能力加权)
        sizes = [remaining // B_k] * B_k
        for i in range(remaining % B_k):
            sizes[i] += 1

        max_end_time = 0.0

        for i in range(B_k):
            mid = machine_ids[i]
            wid = worker_ids[i]
            sz = sizes[i]
            if sz == 0: continue

            machine = state.machines[mid]
            worker = state.workers[wid]

            # 计算开始时间
            prev_op_end = 0.0
            if op.op_id > 0:
                prev_op = state.operations.get((op.job_id, op.op_id - 1))
                if prev_op: prev_op_end = prev_op.complete_time

            # 机器切换时间
            switch_t = 0.0
            if not machine.is_first_use:
                switch_t = self.config.machine_data[mid].get('switch_time', 0.0)

            machine_ready = machine.next_available_time + switch_t if not machine.is_first_use else machine.next_available_time
            start_t = max(prev_op_end, machine_ready, worker.next_available_time)

            # 加工时间受疲劳影响
            base_t = op.base_pt.get(mid, 1.0)
            fatigue_factor = 1 + worker.current_fatigue
            duration = base_t * sz * fatigue_factor
            end_t = start_t + duration

            # 更新资源状态
            machine.is_first_use = False
            idle_t = max(0, start_t - switch_t - machine.next_available_time)
            machine.add_energy(idle_t, switch_t, duration, self.config)
            machine.next_available_time = end_t

            worker.update_fatigue(duration, start_t)

            if end_t > max_end_time:
                max_end_time = end_t

        # 更新工序状态
        op.completed_pieces += remaining
        if op.completed_pieces >= op.total_pieces:
            op.is_scheduled = True
            op.complete_time = max_end_time

        state.current_time = max(state.current_time, max_end_time)
        return state


# ==========================================
# 3. 特征提取 (12维 -> 包含能耗和疲劳)
# ==========================================

class FeatureExtractor:


    @staticmethod
    def _mean_base_pt(config: Config) -> float:
        values = []
        for job_info in config.job_data.values():
            for op_info in job_info.values():
                values.extend(list(op_info.get('base_process_time', {}).values()))
        return float(np.mean(values)) if values else 1.0

    @staticmethod
    def _mean_process_energy(config: Config) -> float:
        values = [v.get('energy_process', 1.0) for v in config.machine_data.values()]
        return float(np.mean(values)) if values else 1.0

    @staticmethod
    def _scales(config: Config):
        avg_pt = FeatureExtractor._mean_base_pt(config)
        avg_energy = FeatureExtractor._mean_process_energy(config)
        total_ops = max(1, config.num_jobs * config.num_ops_per_job)
        total_pieces = max(1, config.total_pieces_per_job)

        # 兜底尺度仅用于状态归一化，不作为实验指标。
        fallback_cmax = total_ops * total_pieces * avg_pt / max(1, config.num_machines) * 1.5
        cmax_scale = float(config.ref_cmax) if getattr(config, 'ref_cmax', None) is not None else max(100.0, fallback_cmax)
        energy_scale = float(config.ref_tec) if getattr(config, 'ref_tec', None) is not None else max(2000.0, cmax_scale * config.num_machines * avg_energy)
        pt_scale = max(1.0, avg_pt * total_pieces)
        return max(1e-8, cmax_scale), max(1e-8, energy_scale), max(1e-8, pt_scale)

    def extract(self, state: SchedulingState) -> np.ndarray:
        config = state.config
        total_ops = max(1, len(state.operations))
        done_ops = sum(1 for o in state.operations.values() if o.is_scheduled)
        progress = done_ops / total_ops

        cmax_scale, energy_scale, pt_scale = self._scales(config)

        machine_times = [m.next_available_time for m in state.machines.values()]
        worker_fatigues = [w.current_fatigue for w in state.workers.values()]
        worker_available_times = [w.next_available_time for w in state.workers.values()]
        ready_ops = state.get_ready_operations()

        current_cmax = max(machine_times, default=0.0)
        total_energy = sum(m.total_energy for m in state.machines.values())

        avg_fatigue = float(np.mean(worker_fatigues)) if worker_fatigues else 0.0
        max_fatigue = max(worker_fatigues, default=0.0)
        std_fatigue = float(np.std(worker_fatigues)) if worker_fatigues else 0.0

        pending_cnt = len(state.pending_dynamic_jobs)
        dynamic_jobs = max(1, config.num_jobs - config.num_jobs_static)
        pending_ratio = pending_cnt / dynamic_jobs

        ready_ratio = len(ready_ops) / total_ops
        avail_workers = sum(1 for w in state.workers.values() if w.next_available_time <= state.current_time)
        avail_worker_ratio = avail_workers / max(1, len(state.workers))

        avg_machine_time = float(np.mean(machine_times)) if machine_times else 0.0
        std_machine_time = float(np.std(machine_times)) if machine_times else 0.0
        max_machine_time = max(machine_times, default=0.0)

        avg_worker_available = float(np.mean(worker_available_times)) if worker_available_times else 0.0
        std_worker_available = float(np.std(worker_available_times)) if worker_available_times else 0.0

        if ready_ops:
            min_pts = []
            candidate_counts = []
            for op_key in ready_ops:
                op = state.operations[op_key]
                if op.machine_candidates:
                    min_pts.append(min(op.base_pt.get(m, 1.0) for m in op.machine_candidates))
                    candidate_counts.append(len(op.machine_candidates))
            avg_ready_min_pt = float(np.mean(min_pts)) if min_pts else 0.0
            avg_candidate_ratio = float(np.mean(candidate_counts)) / max(1, config.num_machines) if candidate_counts else 0.0
        else:
            avg_ready_min_pt = 0.0
            avg_candidate_ratio = 0.0

        features = np.array([
            progress,                                           # 1 调度进度
            current_cmax / cmax_scale,                         # 2 当前 Cmax 尺度化
            total_energy / energy_scale,                       # 3 当前能耗尺度化
            avg_fatigue,                                       # 4 平均疲劳
            max_fatigue,                                       # 5 最大疲劳
            std_fatigue,                                       # 6 疲劳离散度
            pending_ratio,                                     # 7 动态工件待到达比例
            ready_ratio,                                       # 8 就绪工序比例
            avg_machine_time / cmax_scale,                     # 9 平均机器可用时间
            std_machine_time / cmax_scale,                     # 10 机器负载离散度
            max_machine_time / cmax_scale,                     # 11 最大机器可用时间
            avail_worker_ratio,                                # 12 当前可用工人比例
            avg_worker_available / cmax_scale,                 # 13 平均工人可用时间
            std_worker_available / cmax_scale,                 # 14 工人可用时间离散度
            avg_ready_min_pt / pt_scale,                       # 15 就绪工序平均最短加工时间
            avg_candidate_ratio,                               # 16 就绪工序平均候选机器比例
        ], dtype=np.float32)

        return np.clip(features, -10.0, 10.0)


# ==========================================
# 4. 复合调度规则 (Agent的动作空间)
# ==========================================

class SchedulingRules:


    READY_OP_LIMIT = 80

    @staticmethod
    def _ready_ops(state: SchedulingState):
        ready_ops = state.get_ready_operations()
        ready_ops = sorted(ready_ops, key=lambda k: (k[0], k[1]))
        if len(ready_ops) > SchedulingRules.READY_OP_LIMIT:
            # 轻量筛选：优先保留最短加工时间较小的就绪工序，避免规则层过强或过慢。
            ready_ops = sorted(
                ready_ops,
                key=lambda k: (SchedulingRules._op_min_pt(state.operations[k]), k[0], k[1])
            )[:SchedulingRules.READY_OP_LIMIT]
        return ready_ops

    @staticmethod
    def _op_min_pt(op: Operation):
        if not op.machine_candidates:
            return float('inf')
        return min(op.base_pt.get(m, float('inf')) for m in op.machine_candidates)

    @staticmethod
    def _op_max_pt(op: Operation):
        if not op.machine_candidates:
            return float('-inf')
        return max(op.base_pt.get(m, float('-inf')) for m in op.machine_candidates)

    @staticmethod
    def _prev_completion_time(op: Operation, state: SchedulingState):
        if op.op_id <= 0:
            return 0.0
        prev_op = state.operations.get((op.job_id, op.op_id - 1))
        return prev_op.complete_time if prev_op is not None else 0.0

    @staticmethod
    def _select_workers(num_needed: int, state: SchedulingState, criterion: str):
        candidates = [w for w in state.workers.values() if not w.is_busy]
        if len(candidates) < num_needed:
            return None

        if criterion == 'earliest':
            candidates.sort(key=lambda w: (w.next_available_time, w.current_fatigue, w.worker_id))
        elif criterion == 'freshest':
            candidates.sort(key=lambda w: (w.current_fatigue, w.next_available_time, w.worker_id))
        else:
            candidates.sort(key=lambda w: w.worker_id)
        return [w.worker_id for w in candidates[:num_needed]]

    @staticmethod
    def _select_machines(op: Operation, num_needed: int, state: SchedulingState, criterion: str):
        cands = list(op.machine_candidates)
        if len(cands) < num_needed:
            return None

        if criterion == 'spt':
            cands.sort(key=lambda m: (op.base_pt.get(m, float('inf')), m))
        elif criterion == 'lpt':
            cands.sort(key=lambda m: (-op.base_pt.get(m, float('-inf')), m))
        elif criterion == 'energy':
            cands.sort(key=lambda m: (
                state.config.machine_data.get(m, {}).get('energy_process', 1.0),
                op.base_pt.get(m, float('inf')),
                m
            ))
        else:
            cands.sort()
        return cands[:num_needed]

    @staticmethod
    def _create_action(op_key, state: SchedulingState, mach_criterion, worker_criterion):
        op = state.operations[op_key]
        B_k = min(len(op.machine_candidates), 3)
        if B_k <= 0:
            return None

        machs = SchedulingRules._select_machines(op, B_k, state, mach_criterion)
        workers = SchedulingRules._select_workers(B_k, state, worker_criterion)
        if not machs or not workers:
            return None

        return {
            'op_key': op_key,
            'batch_size': B_k,
            'machines': machs,
            'workers': workers
        }

    @staticmethod
    def _estimate_action(op_key, state: SchedulingState, mach_criterion: str, worker_criterion: str):
        """对一个 ready operation 的动作效果做轻量估计。"""
        action = SchedulingRules._create_action(op_key, state, mach_criterion, worker_criterion)
        if action is None:
            return None

        op = state.operations[op_key]
        B_k = action['batch_size']
        machine_ids = action['machines']
        worker_ids = action['workers']

        remaining = max(1, op.total_pieces - op.completed_pieces)
        avg_size = remaining / max(1, B_k)
        prev_op_end = SchedulingRules._prev_completion_time(op, state)
        current_cmax = max((m.next_available_time for m in state.machines.values()), default=0.0)

        max_end = 0.0
        total_duration = 0.0
        delta_energy = 0.0
        delta_fatigue_sum = 0.0

        for mid, wid in zip(machine_ids, worker_ids):
            machine = state.machines[mid]
            worker = state.workers[wid]
            switch_t = 0.0 if machine.is_first_use else state.config.machine_data.get(mid, {}).get('switch_time', 0.0)
            machine_ready = machine.next_available_time + switch_t if not machine.is_first_use else machine.next_available_time
            start_t = max(prev_op_end, machine_ready, worker.next_available_time)

            base_t = op.base_pt.get(mid, 1.0)
            duration = base_t * avg_size * (1.0 + worker.current_fatigue)
            end_t = start_t + duration
            idle_t = max(0.0, start_t - switch_t - machine.next_available_time)

            m_conf = state.config.machine_data.get(mid, {})
            delta_energy += idle_t * m_conf.get('energy_idle', 0.0)
            delta_energy += switch_t * m_conf.get('energy_switch', 0.0)
            delta_energy += duration * m_conf.get('energy_process', 1.0)

            delta_fatigue_sum += (1.0 - worker.current_fatigue) * (1.0 - np.exp(-worker.lambda_ * duration))
            total_duration += duration
            max_end = max(max_end, end_t)

        delta_cmax = max(0.0, max(current_cmax, max_end) - current_cmax)
        avg_duration = total_duration / max(1, B_k)
        avg_delta_fatigue = delta_fatigue_sum / max(1, len(state.workers))

        return {
            'action': action,
            'op_key': op_key,
            'est_finish': max_end,
            'delta_cmax': delta_cmax,
            'delta_energy': delta_energy,
            'delta_fatigue': avg_delta_fatigue,
            'avg_duration': avg_duration,
            'min_pt': SchedulingRules._op_min_pt(op),
            'max_pt': SchedulingRules._op_max_pt(op),
            'op_ready': prev_op_end,
        }

    @staticmethod
    def _choose_by_score(state: SchedulingState, mach_criterion: str, worker_criterion: str, score_fn):
        best = None
        best_score = None
        for op_key in SchedulingRules._ready_ops(state):
            est = SchedulingRules._estimate_action(op_key, state, mach_criterion, worker_criterion)
            if est is None:
                continue
            score = score_fn(est)
            # 加入 op_key 作为稳定 tie-break，避免同分时随机波动。
            score = tuple(score) if isinstance(score, (list, tuple)) else (score,)
            score = score + (op_key[0], op_key[1])
            if best is None or score < best_score:
                best = est
                best_score = score
        return best['action'] if best is not None else None

    @staticmethod
    def rule1_spt_earliest_worker(state: SchedulingState):
        """SPT + earliest worker：比较全部就绪工序，选择预计短加工/早完成的动作。"""
        return SchedulingRules._choose_by_score(
            state, 'spt', 'earliest',
            lambda e: (e['avg_duration'], e['est_finish'])
        )

    @staticmethod
    def rule2_spt_lowest_fatigue_worker(state: SchedulingState):
        """SPT + lowest-fatigue worker：兼顾短加工和较小疲劳增量。"""
        return SchedulingRules._choose_by_score(
            state, 'spt', 'freshest',
            lambda e: (e['avg_duration'] * (1.0 + 0.5 * e['delta_fatigue']), e['delta_fatigue'], e['est_finish'])
        )

    @staticmethod
    def rule3_fifo_earliest_worker(state: SchedulingState):
        """FIFO + earliest worker：按工序可开工时间优先，并用预计完成时间打破同分。"""
        return SchedulingRules._choose_by_score(
            state, 'spt', 'earliest',
            lambda e: (e['op_ready'], e['est_finish'])
        )

    @staticmethod
    def rule4_lpt_earliest_worker(state: SchedulingState):
        """LPT + earliest worker：比较全部就绪工序，优先处理估计加工时间较长的工序。"""
        return SchedulingRules._choose_by_score(
            state, 'spt', 'earliest',
            lambda e: (-e['avg_duration'], e['est_finish'])
        )

    @staticmethod
    def rule5_energy_saving_lowest_fatigue_worker(state: SchedulingState):
        """Energy-saving machine + lowest-fatigue worker：比较全部就绪工序的预计能耗。"""
        return SchedulingRules._choose_by_score(
            state, 'energy', 'freshest',
            lambda e: (e['delta_energy'], e['delta_fatigue'], e['delta_cmax'])
        )


# ==========================================
# 5. MHDQN 网络 (3 Heads)
# ==========================================

class MHDQN_Net(nn.Module):
    def __init__(self, state_dim, num_actions):
        super().__init__()
        # 共享层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU()
        )
        # 三个目标头
        self.head_cmax = self._dueling_head(num_actions)
        self.head_energy = self._dueling_head(num_actions)
        self.head_fatigue = self._dueling_head(num_actions)

    def _dueling_head(self, num_actions):
        return nn.ModuleDict({
            'value': nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)),
            'adv': nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, num_actions))
        })

    def forward(self, x):
        feat = self.shared(x)

        def calc_q(head, f):
            v = head['value'](f)
            a = head['adv'](f)
            return v + (a - a.mean(dim=1, keepdim=True))

        return {
            'q_cmax': calc_q(self.head_cmax, feat),
            'q_energy': calc_q(self.head_energy, feat),
            'q_fatigue': calc_q(self.head_fatigue, feat)
        }


# ==========================================
# 6. Agent 与 训练逻辑
# ==========================================

class MHDQN_Agent:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.env = MODFJSPEnv(config)
        self.fe = FeatureExtractor()

        # 动作集：5个规则。
        # 已显式加入 SPT + earliest worker 和 SPT + lowest-fatigue worker，
        # 并替换原先表现不稳定的 energy_focused 与 max_batch 规则。
        self.rules = [
            SchedulingRules.rule1_spt_earliest_worker,
            SchedulingRules.rule2_spt_lowest_fatigue_worker,
            SchedulingRules.rule3_fifo_earliest_worker,
            SchedulingRules.rule4_lpt_earliest_worker,
            SchedulingRules.rule5_energy_saving_lowest_fatigue_worker
        ]
        self.num_actions = len(self.rules)

        # 网络
        state_dim = 16  # FeatureExtractor输出维度：增强后的轻量全局特征
        self.net = MHDQN_Net(state_dim, self.num_actions).to(self.device)
        self.target_net = MHDQN_Net(state_dim, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.net.state_dict())
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-4)

        self.buffer = deque(maxlen=10000)
        self.gamma = 0.99
        self.batch_size = 64
        self.update_steps = 500
        self.step_count = 0

        # 权重循环 (3个目标的权重组合)
        self.weights_pool = [
            [0.5, 0.25, 0.25],
            [0.34, 0.33, 0.33],
            [0.6, 0.2, 0.2],
            [0.2, 0.6, 0.2],
            [0.2, 0.2, 0.6]
        ]
        self.current_weights = self.weights_pool[0]
        # 多实例统一训练时，用全局 episode 数控制 ε 衰减，避免每个实例重新从高 ε 开始。
        self.epsilon_decay_episodes = 5000

        # 动态奖励归一化尺度。切换实例时会自动更新。
        self.reward_cmax_scale, self.reward_energy_scale, self.reward_fatigue_scale = self._compute_reward_scales(config)

    def set_instance(self, config: Config):
        """切换当前训练实例，但保留同一个网络、优化器、目标网络和经验池。

        多实例统一训练时，每个 episode 随机抽取一个 JSON 实例，
        只更新环境配置，不重新初始化模型参数。
        """
        self.config = config
        self.env = MODFJSPEnv(config)
        self.reward_cmax_scale, self.reward_energy_scale, self.reward_fatigue_scale = self._compute_reward_scales(config)

    def _compute_reward_scales(self, config: Config):
        """根据 reference_solution 或实例规模估计训练奖励归一化尺度。"""
        avg_pt = FeatureExtractor._mean_base_pt(config)
        avg_energy = FeatureExtractor._mean_process_energy(config)
        total_ops = max(1, config.num_jobs * config.num_ops_per_job)
        total_pieces = max(1, config.total_pieces_per_job)
        fallback_cmax = total_ops * total_pieces * avg_pt / max(1, config.num_machines) * 1.5
        cmax_scale = float(config.ref_cmax) if getattr(config, 'ref_cmax', None) is not None else max(100.0, fallback_cmax)
        energy_scale = float(config.ref_tec) if getattr(config, 'ref_tec', None) is not None else max(2000.0, cmax_scale * config.num_machines * avg_energy)
        fatigue_scale = float(config.ref_favg) if getattr(config, 'ref_favg', None) is not None else 1.0
        return max(1e-8, cmax_scale), max(1e-8, energy_scale), max(1e-8, fatigue_scale)

    @staticmethod
    def _scaled_delta_reward(delta_value: float, scale_value: float):
        # 使用 clip 防止大实例的单步增量使 Q 值震荡；保留负增量对应的正奖励。
        return float(-np.clip(delta_value / max(1e-8, scale_value), -1.0, 1.0))

    def select_action(self, state_vec, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)

        state_t = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
        with torch.no_grad():
            qs = self.net(state_t)

        # Q值归一化
        q_c = qs['q_cmax'][0].cpu().numpy()
        q_e = qs['q_energy'][0].cpu().numpy()
        q_f = qs['q_fatigue'][0].cpu().numpy()

        q_c_norm = q_c / (np.abs(q_c).max() + 1e-8)
        q_e_norm = q_e / (np.abs(q_e).max() + 1e-8)
        q_f_norm = q_f / (np.abs(q_f).max() + 1e-8)

        # 加权求和
        combined = (self.current_weights[0] * q_c_norm +
                    self.current_weights[1] * q_e_norm +
                    self.current_weights[2] * q_f_norm)

        return int(np.argmax(combined))

    def train_step(self):
        if len(self.buffer) < self.batch_size: return 0.0

        batch = random.sample(self.buffer, self.batch_size)

        # ==========================================
        # 优化：先转换为 np.array，再转为 Tensor
        # ==========================================

        # 1. 提取数据并转换为 numpy 矩阵
        states_np = np.array([b[0] for b in batch])
        actions_np = np.array([b[1] for b in batch])

        # 奖励是列表 [r_c, r_e, r_f]，需要分别提取
        rewards_c_np = np.array([b[2][0] for b in batch])
        rewards_e_np = np.array([b[2][1] for b in batch])
        rewards_f_np = np.array([b[2][2] for b in batch])

        next_states_np = np.array([b[3] for b in batch])
        dones_np = np.array([b[4] for b in batch])

        # 2. 转换为 Tensor
        S = torch.from_numpy(states_np).float().to(self.device)
        A = torch.from_numpy(actions_np).long().to(self.device)
        R_C = torch.from_numpy(rewards_c_np).float().to(self.device)
        R_E = torch.from_numpy(rewards_e_np).float().to(self.device)
        R_F = torch.from_numpy(rewards_f_np).float().to(self.device)
        S_ = torch.from_numpy(next_states_np).float().to(self.device)
        Done = torch.from_numpy(dones_np).float().to(self.device)

        # ==========================================
        # 以下逻辑保持不变
        # ==========================================

        # Current Q
        curr_qs = self.net(S)
        q_c_pred = curr_qs['q_cmax'].gather(1, A.unsqueeze(1)).squeeze()
        q_e_pred = curr_qs['q_energy'].gather(1, A.unsqueeze(1)).squeeze()
        q_f_pred = curr_qs['q_fatigue'].gather(1, A.unsqueeze(1)).squeeze()

        # Target Q (Double DQN)
        with torch.no_grad():
            next_qs_main = self.net(S_)
            next_qs_target = self.target_net(S_)

            # 分别选最优动作
            act_c = next_qs_main['q_cmax'].argmax(1)
            act_e = next_qs_main['q_energy'].argmax(1)
            act_f = next_qs_main['q_fatigue'].argmax(1)

            q_c_next = next_qs_target['q_cmax'].gather(1, act_c.unsqueeze(1)).squeeze()
            q_e_next = next_qs_target['q_energy'].gather(1, act_e.unsqueeze(1)).squeeze()
            q_f_next = next_qs_target['q_fatigue'].gather(1, act_f.unsqueeze(1)).squeeze()

            target_c = R_C + self.gamma * q_c_next * (1 - Done)
            target_e = R_E + self.gamma * q_e_next * (1 - Done)
            target_f = R_F + self.gamma * q_f_next * (1 - Done)

        loss_c = nn.SmoothL1Loss()(q_c_pred, target_c)
        loss_e = nn.SmoothL1Loss()(q_e_pred, target_e)
        loss_f = nn.SmoothL1Loss()(q_f_pred, target_f)

        # 梯度聚合更新
        total_loss = loss_c + loss_e + loss_f

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.update_steps == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        return total_loss.item()

    def run_episode(self, episode_idx, max_steps=1000, train=True, deterministic=False, weights_override=None):
        """运行一个 episode。

        train=True  时：写入 replay buffer，并执行 DQN 更新；
        train=False 时：仅前向决策，用于验证/测试，不更新网络和经验池。
        deterministic=True 时：epsilon=0，固定采用当前 Q 值最大的规则动作。
        weights_override 可用于验证阶段固定多目标权重，避免验证分数受权重轮换影响。
        """
        state = self.env.reset()
        if weights_override is not None:
            self.current_weights = list(weights_override)
        else:
            self.current_weights = self.weights_pool[episode_idx % len(self.weights_pool)]

        prev_metrics = {'cmax': 0, 'energy': 0, 'fatigue': 0}
        total_reward = [0, 0, 0]
        steps = 0

        while steps < max_steps:
            # 1. 检查动态工件
            arrived = state.check_dynamic_arrival()
            for jid in arrived:
                self.env.add_job_to_state(state, jid)

            # 2. 获取就绪工序
            ready_ops = state.get_ready_operations()
            if not ready_ops:
                # 如果没有就绪工序，时间推进到下一个机器释放时刻
                next_time = min(m.next_available_time for m in state.machines.values())
                state.current_time = max(state.current_time, next_time + 0.1)
                # 检查是否全部完成
                if all(op.is_scheduled for op in state.operations.values()):
                    break
                continue

            # 3. 提取特征 & 选规则
            feat = self.fe.extract(state)
            if deterministic:
                epsilon = 0.0
            elif train:
                epsilon = max(0.01, 1.0 - episode_idx / max(1.0, float(self.epsilon_decay_episodes)))
            else:
                # 非确定性验证/测试时保留极小探索；默认验证使用 deterministic=True。
                epsilon = 0.01
            action_idx = self.select_action(feat, epsilon)

            # 4. 执行规则
            action_dict = self.rules[action_idx](state)
            if action_dict is None:
                # 规则不可行时按顺序尝试其他规则，避免在同一状态无限循环。
                for fallback_rule in self.rules:
                    action_dict = fallback_rule(state)
                    if action_dict is not None:
                        break
                if action_dict is None:
                    break

            next_state = self.env.step(state, action_dict)

            # 5. 计算奖励 (增量式)
            curr_cmax = max(m.next_available_time for m in next_state.machines.values())
            curr_energy = sum(m.total_energy for m in next_state.machines.values())
            curr_fatigue = np.mean([w.current_fatigue for w in next_state.workers.values()])

            # 动态归一化增量奖励：避免不同规模实例之间奖励尺度差异过大。
            r_c = self._scaled_delta_reward(curr_cmax - prev_metrics['cmax'], self.reward_cmax_scale)
            r_e = self._scaled_delta_reward(curr_energy - prev_metrics['energy'], self.reward_energy_scale)
            r_f = self._scaled_delta_reward(curr_fatigue - prev_metrics['fatigue'], self.reward_fatigue_scale)

            total_reward = [total_reward[i] + r for i, r in enumerate([r_c, r_e, r_f])]

            done = all(op.is_scheduled for op in next_state.operations.values())

            if train:
                # 训练阶段才写入 Buffer 并更新网络；验证阶段不能污染经验池或模型参数。
                next_feat = self.fe.extract(next_state)
                self.buffer.append((feat, action_idx, [r_c, r_e, r_f], next_feat, float(done)))

            state = next_state
            prev_metrics = {'cmax': curr_cmax, 'energy': curr_energy, 'fatigue': curr_fatigue}
            steps += 1

            if train:
                self.train_step()

        return {
            'cmax': prev_metrics['cmax'],
            'energy': prev_metrics['energy'],
            'fatigue': prev_metrics['fatigue'],
            'reward': total_reward
        }


# ==========================================
# 7. 主程序入口：多实例统一训练一个模型
# ==========================================

def collect_train_files(base_dir, target_levels=None):
    """收集训练 JSON。

    支持两种数据结构：
    1) base_dir/level_1/*.json ... base_dir/level_8/*.json
    2) base_dir/*.json 或 base_dir 下任意子目录递归包含 JSON
    """
    import glob

    files = []
    if target_levels:
        for level in target_levels:
            level_dir = os.path.join(base_dir, level)
            if os.path.isdir(level_dir):
                files.extend(glob.glob(os.path.join(level_dir, "*.json")))

    if not files:
        files.extend(glob.glob(os.path.join(base_dir, "*.json")))
        files.extend(glob.glob(os.path.join(base_dir, "**", "*.json"), recursive=True))

    # 去重并排序，保证可复现
    files = sorted(set(files))
    return files


def save_mhdqn_checkpoint(agent, model_save_dir, episode, train_files, tag, extra_info=None):

    os.makedirs(model_save_dir, exist_ok=True)

    checkpoint_path = os.path.join(model_save_dir, f"mhdqn_unified_checkpoint_{tag}.pth")
    state_dict_path = os.path.join(model_save_dir, f"mhdqn_unified_{tag}.pth")

    checkpoint = {
        'episode': int(episode),
        'net_state_dict': agent.net.state_dict(),
        'target_net_state_dict': agent.target_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'step_count': int(agent.step_count),
        'train_files': list(train_files),
        'epsilon_decay_episodes': int(agent.epsilon_decay_episodes),
    }
    if extra_info:
        checkpoint.update(extra_info)

    torch.save(checkpoint, checkpoint_path)

    torch.save(agent.net.state_dict(), state_dict_path)
    return checkpoint_path, state_dict_path


def _mean_base_process_time(config: Config):
    values = []
    for job_info in config.job_data.values():
        for op_info in job_info.values():
            values.extend(list(op_info.get('base_process_time', {}).values()))
    return float(np.mean(values)) if values else 1.0


def _mean_machine_process_energy(config: Config):
    values = [v.get('energy_process', 1.0) for v in config.machine_data.values()]
    return float(np.mean(values)) if values else 1.0


def _validation_scales(config: Config):
    """返回 Cmax / TEC / Favg 的验证归一化尺度。优先使用 JSON 中 reference_solution。"""
    avg_pt = _mean_base_process_time(config)
    avg_process_energy = _mean_machine_process_energy(config)
    total_ops = max(1, config.num_jobs * config.num_ops_per_job)
    total_pieces = max(1, config.total_pieces_per_job)

    # 兜底尺度只用于不同 episode 间比较 best_val，不作为最终实验指标。
    fallback_cmax = total_ops * total_pieces * avg_pt / max(1, config.num_machines) * 1.5
    cmax_scale = float(config.ref_cmax) if config.ref_cmax is not None else max(1.0, fallback_cmax)
    tec_scale = float(config.ref_tec) if config.ref_tec is not None else max(1.0, cmax_scale * config.num_machines * avg_process_energy)
    favg_scale = float(config.ref_favg) if config.ref_favg is not None else 1.0
    return max(1e-8, cmax_scale), max(1e-8, tec_scale), max(1e-8, favg_scale)


def evaluate_agent_on_files(agent, val_files, val_runs=1, deterministic=True, max_steps=1000):
    """在验证集上评估当前模型，并返回用于选择 best model 的加权归一化分数。"""
    if not val_files:
        return None, []

    was_training = agent.net.training
    agent.net.eval()
    agent.target_net.eval()

    detail_rows = []
    balanced_weights = [0.34, 0.33, 0.33]

    with torch.no_grad():
        for json_path in val_files:
            config = Config(json_path)
            instance_name = os.path.splitext(os.path.basename(json_path))[0]
            cmax_scale, tec_scale, favg_scale = _validation_scales(config)

            for run_id in range(1, val_runs + 1):
                agent.set_instance(config)
                metrics = agent.run_episode(
                    episode_idx=0,
                    max_steps=max_steps,
                    train=False,
                    deterministic=deterministic,
                    weights_override=balanced_weights,
                )
                c_ratio = metrics['cmax'] / cmax_scale
                e_ratio = metrics['energy'] / tec_scale
                f_ratio = metrics['fatigue'] / favg_scale
                score = (c_ratio + e_ratio + f_ratio) / 3.0

                detail_rows.append({
                    'instance': instance_name,
                    'run': run_id,
                    'cmax': metrics['cmax'],
                    'energy': metrics['energy'],
                    'fatigue_mean': metrics['fatigue'],
                    'cmax_ratio': c_ratio,
                    'energy_ratio': e_ratio,
                    'fatigue_ratio': f_ratio,
                    'score': score,
                })

    if was_training:
        agent.net.train()
        agent.target_net.train()

    summary = {
        'val_score': float(np.mean([r['score'] for r in detail_rows])),
        'val_mean_cmax': float(np.mean([r['cmax'] for r in detail_rows])),
        'val_mean_energy': float(np.mean([r['energy'] for r in detail_rows])),
        'val_mean_fatigue': float(np.mean([r['fatigue_mean'] for r in detail_rows])),
        'val_mean_cmax_ratio': float(np.mean([r['cmax_ratio'] for r in detail_rows])),
        'val_mean_energy_ratio': float(np.mean([r['energy_ratio'] for r in detail_rows])),
        'val_mean_fatigue_ratio': float(np.mean([r['fatigue_ratio'] for r in detail_rows])),
    }
    return summary, detail_rows


if __name__ == "__main__":
    import pandas as pd

    # ================= 参数配置 =================
    BASE_DATA_PATH = "../../TestData/train1"
    # 验证集路径：用于训练过程中保存 best_val 模型。请按你的实际路径修改。
    VAL_DATA_PATH = "../../TestData/yanzheng1"
    RESULTS_SAVE_DIR = "comparison_results_mhdqn_new"
    MODEL_SAVE_DIR = "saved_models_mhdqn_new"

    # 如果你的训练集没有 level 子文件夹，也可以保持不变，代码会自动递归搜索 JSON。
    TARGET_LEVELS = ['level_1', 'level_2', 'level_3', 'level_4', 'level_5', 'level_6', 'level_7', 'level_8']

    # 多实例统一训练：总 episode 数，而不是每个实例单独训练的 episode 数。
    TOTAL_EPISODES = 5000
    DEBUG_INTERVAL = 500
    SAVE_INTERVAL = 500
    # 每隔多少 episode 在验证集上评估一次，并保存验证集最优模型。
    VAL_INTERVAL = 500
    VAL_RUNS = 1
    VAL_DETERMINISTIC = True
    SEED = 2026

    # 断点继续训练；不需要继续训练时设为 None。
    RESUME_CHECKPOINT = None
    # 示例：RESUME_CHECKPOINT = "saved_models_mhdqn/mhdqn_unified_checkpoint_latest.pth"

    os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    train_files = collect_train_files(BASE_DATA_PATH, TARGET_LEVELS)
    if not train_files:
        raise FileNotFoundError(f"未找到训练实例 JSON，请检查 BASE_DATA_PATH: {BASE_DATA_PATH}")

    val_files = []
    if os.path.exists(VAL_DATA_PATH):
        val_files = collect_train_files(VAL_DATA_PATH, TARGET_LEVELS)

    if not val_files:
        # 兜底：没有单独验证集时，使用训练集前若干个实例做监控。
        # 正式论文实验建议指定独立 VAL_DATA_PATH。
        fallback_n = max(1, min(4, len(train_files) // 5 if len(train_files) >= 5 else 1))
        val_files = train_files[:fallback_n]
        print(f"[Warning] 未找到独立验证集，将使用 {fallback_n} 个训练实例作为验证监控。正式实验建议设置 VAL_DATA_PATH。")

    print(f"{'=' * 20} MHDQN Unified Multi-instance Training Start {'=' * 20}")
    print(f"训练实例数: {len(train_files)}")
    print(f"验证实例数: {len(val_files)}")
    print(f"总训练轮数: {TOTAL_EPISODES}")

    # 只创建一个 agent，之后所有实例共享同一个 net / target_net / optimizer / replay buffer。
    first_config = Config(train_files[0])
    agent = MHDQN_Agent(first_config)
    agent.epsilon_decay_episodes = TOTAL_EPISODES

    start_episode = 1
    loaded_checkpoint = None
    if RESUME_CHECKPOINT and os.path.exists(RESUME_CHECKPOINT):
        checkpoint = torch.load(RESUME_CHECKPOINT, map_location=agent.device)
        loaded_checkpoint = checkpoint
        agent.net.load_state_dict(checkpoint['net_state_dict'])
        agent.target_net.load_state_dict(checkpoint.get('target_net_state_dict', checkpoint['net_state_dict']))
        if 'optimizer_state_dict' in checkpoint:
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.step_count = int(checkpoint.get('step_count', 0))
        start_episode = int(checkpoint.get('episode', 0)) + 1
        # 如果断点中保存过 best_val 信息，则继续沿用。
        # 这里变量稍后在主循环前初始化；若你从 best_val 断点继续训练，也不会影响继续训练。
        print(f"已加载断点: {RESUME_CHECKPOINT}，从 episode {start_episode} 继续训练")

    history = []
    val_history = []
    val_detail_history = []
    best_val_score = float('inf')
    best_val_summary = None
    if loaded_checkpoint is not None:
        best_val_score = float(loaded_checkpoint.get('best_val_score', best_val_score))
        best_val_summary = loaded_checkpoint.get('best_val_summary', best_val_summary)
    instance_counter = defaultdict(int)
    training_start_time = time.time()

    for ep in range(start_episode, TOTAL_EPISODES + 1):
        # 记录当前 episode 的实际耗时
        episode_start_time = time.time()

        # 每个 episode 随机抽取一个训练实例，统一训练同一个模型。
        json_path = random.choice(train_files)
        config = Config(json_path)
        agent.set_instance(config)

        instance_name = os.path.splitext(os.path.basename(json_path))[0]
        instance_counter[instance_name] += 1

        try:
            metrics = agent.run_episode(ep - 1)
        except Exception as e:
            print(f"[Error] Ep {ep} | {instance_name} 运行失败: {e}")
            import traceback
            traceback.print_exc()
            continue

        # 当前 episode 耗时与累计训练耗时
        episode_time_sec = time.time() - episode_start_time
        total_elapsed_sec = time.time() - training_start_time

        row = {
            'episode': ep,
            'instance': instance_name,
            'cmax': metrics['cmax'],
            'energy': metrics['energy'],
            'fatigue_mean': metrics['fatigue'],
            'reward_cmax': metrics['reward'][0],
            'reward_energy': metrics['reward'][1],
            'reward_fatigue': metrics['reward'][2],
            'buffer_size': len(agent.buffer),
            'epsilon': max(0.01, 1.0 - (ep - 1) / max(1.0, float(TOTAL_EPISODES))),
            'episode_time_sec': episode_time_sec,
            'total_elapsed_sec': total_elapsed_sec,
        }
        history.append(row)

        if ep % DEBUG_INTERVAL == 0 or ep == start_episode:
            recent = history[-min(DEBUG_INTERVAL, len(history)):]
            mean_cmax = np.mean([x['cmax'] for x in recent])
            mean_energy = np.mean([x['energy'] for x in recent])
            mean_fatigue = np.mean([x['fatigue_mean'] for x in recent])
            mean_time = np.mean([x['episode_time_sec'] for x in recent])
            print(
                f"Ep {ep:5d}/{TOTAL_EPISODES} | {instance_name} | "
                f"Cmax={metrics['cmax']:.1f} | TEC={metrics['energy']:.1f} | Favg={metrics['fatigue']:.3f} | "
                f"Time={episode_time_sec:.2f}s | recent_time={mean_time:.2f}s | "
                f"recent_mean=({mean_cmax:.1f}, {mean_energy:.1f}, {mean_fatigue:.3f}) | "
                f"buffer={len(agent.buffer)}"
            )

        if ep % VAL_INTERVAL == 0 or ep == TOTAL_EPISODES:
            val_start_time = time.time()
            val_summary, val_detail_rows = evaluate_agent_on_files(
                agent,
                val_files,
                val_runs=VAL_RUNS,
                deterministic=VAL_DETERMINISTIC,
            )
            val_time_sec = time.time() - val_start_time

            if val_summary is not None:
                is_best = val_summary['val_score'] < best_val_score
                if is_best:
                    best_val_score = val_summary['val_score']
                    best_val_summary = dict(val_summary)

                val_row = {
                    'episode': ep,
                    **val_summary,
                    'best_val_score': best_val_score,
                    'is_best_val': int(is_best),
                    'val_time_sec': val_time_sec,
                }
                val_history.append(val_row)

                for d in val_detail_rows:
                    val_detail_history.append({'episode': ep, **d})

                # 同步到训练日志最后一行，便于一个 CSV 中查看验证变化。
                history[-1].update(val_row)

                val_log_path = os.path.join(RESULTS_SAVE_DIR, "mhdqn_unified_validation_log.csv")
                val_detail_path = os.path.join(RESULTS_SAVE_DIR, "mhdqn_unified_validation_detail.csv")
                pd.DataFrame(val_history).to_csv(val_log_path, index=False)
                pd.DataFrame(val_detail_history).to_csv(val_detail_path, index=False)

                print(
                    f"[Validation] Ep {ep:5d} | score={val_summary['val_score']:.6f} | "
                    f"Cmax={val_summary['val_mean_cmax']:.2f} | "
                    f"TEC={val_summary['val_mean_energy']:.2f} | "
                    f"Favg={val_summary['val_mean_fatigue']:.4f} | "
                    f"time={val_time_sec:.2f}s"
                )

                if is_best:
                    extra = {'best_val_score': best_val_score, 'best_val_summary': best_val_summary}
                    best_ckpt, best_sd = save_mhdqn_checkpoint(
                        agent, MODEL_SAVE_DIR, ep, train_files, tag="best_val", extra_info=extra
                    )
                    print(f"[Best Val Updated] score={best_val_score:.6f}")
                    print(f"[Saved] best checkpoint: {best_ckpt}")
                    print(f"[Saved] best state_dict: {best_sd}")
                else:
                    print(f"[Validation] 未更新 best_val | 当前 best_val_score={best_val_score:.6f}")

        if ep % SAVE_INTERVAL == 0:
            log_path = os.path.join(RESULTS_SAVE_DIR, "mhdqn_unified_training_log.csv")
            pd.DataFrame(history).to_csv(log_path, index=False)
            extra = {'best_val_score': best_val_score, 'best_val_summary': best_val_summary}
            ckpt_path, sd_path = save_mhdqn_checkpoint(agent, MODEL_SAVE_DIR, ep, train_files, tag=f"ep{ep}", extra_info=extra)
            save_mhdqn_checkpoint(agent, MODEL_SAVE_DIR, ep, train_files, tag="latest", extra_info=extra)
            print(f"[Saved] checkpoint: {ckpt_path}")
            print(f"[Saved] state_dict: {sd_path}")

    # 保存最终训练日志和模型
    final_log_path = os.path.join(RESULTS_SAVE_DIR, "mhdqn_unified_training_log.csv")
    pd.DataFrame(history).to_csv(final_log_path, index=False)

    val_log_path = os.path.join(RESULTS_SAVE_DIR, "mhdqn_unified_validation_log.csv")
    val_detail_path = os.path.join(RESULTS_SAVE_DIR, "mhdqn_unified_validation_detail.csv")
    if val_history:
        pd.DataFrame(val_history).to_csv(val_log_path, index=False)
    if val_detail_history:
        pd.DataFrame(val_detail_history).to_csv(val_detail_path, index=False)

    sample_count_path = os.path.join(RESULTS_SAVE_DIR, "mhdqn_unified_instance_sample_count.csv")
    pd.DataFrame([
        {'instance': k, 'sampled_episodes': v}
        for k, v in sorted(instance_counter.items())
    ]).to_csv(sample_count_path, index=False)

    extra = {'best_val_score': best_val_score, 'best_val_summary': best_val_summary}
    final_ckpt, final_sd = save_mhdqn_checkpoint(agent, MODEL_SAVE_DIR, TOTAL_EPISODES, train_files, tag="final", extra_info=extra)
    save_mhdqn_checkpoint(agent, MODEL_SAVE_DIR, TOTAL_EPISODES, train_files, tag="latest", extra_info=extra)

    total_training_time_sec = time.time() - training_start_time

    print(f"\n训练总耗时: {total_training_time_sec:.2f}s ({total_training_time_sec / 60:.2f} min)")
    print(f"训练日志已保存: {final_log_path}")
    if val_history:
        print(f"验证日志已保存: {val_log_path}")
        print(f"验证明细已保存: {val_detail_path}")
        print(f"验证集最优 score: {best_val_score:.6f}")
        print(f"验证集最优模型: {os.path.join(MODEL_SAVE_DIR, 'mhdqn_unified_best_val.pth')}")
    print(f"实例采样次数已保存: {sample_count_path}")
    print(f"最终 checkpoint: {final_ckpt}")
    print(f"最终网络权重: {final_sd}")
    print(f"{'=' * 20} MHDQN Unified Multi-instance Training Finished {'=' * 20}")
