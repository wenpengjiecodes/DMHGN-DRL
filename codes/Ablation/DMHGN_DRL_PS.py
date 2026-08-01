# 训练模型文件 — HGSAP-PPO-w/o PS (消融实验：去掉剪枝策略)
# 基于 V1_01.py，去掉两层剪枝：(1) 启发式工人组合筛选 → 全组合枚举；(2) Top-K 截断 → 保留全部候选
# 改进：动态归一化参数、动态最大步数
#改进核心特征的提取
# 引入日志

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
import matplotlib.pyplot as plt
from collections import deque
import os
import glob  # <--- 在这里添加这一行
import csv
import time
import gc
from datetime import datetime
import re    # 如果你之后还要用到正则匹配文件名，也建议加上
from mpl_toolkits.mplot3d import Axes3D



# 全局配置类
class Config:
    def __init__(self, json_path=None):
        self.json_path = json_path
        if json_path:
            # 从JSON文件加载配置
            self.load_from_json(json_path)

    def load_from_json(self, json_path):
        """从JSON文件加载配置"""
        with open(json_path, 'r') as f:
            data = json.load(f)

        # 加载参考解指标（次优解/启发式解），用于归一化、验证集模型选择
        reference = data.get('reference_solution', {})
        self.ref_cmax = reference.get('Cmax', None)
        self.ref_tec = reference.get('TEC', None)
        self.ref_favg = reference.get('Favg', None)

        # 加载全局配置
        global_config = data.get('global_config', {})
        # 网络隐藏维度采用代码侧强制覆盖，避免不同 JSON 中旧维度配置
        # （如 d_operation=64, d_batch=32, d_worker=32）影响当前轻量化模型结构。
        self.d_operation = 32
        self.d_batch = 16
        self.d_worker = 16
        self.d_global = 5 + self.d_operation + self.d_batch + self.d_worker  # 69
        self.fatigue_threshold = global_config.get('fatigue_threshold', 0.8)
        self.num_jobs = global_config.get('num_jobs', 10)
        self.num_jobs_static = global_config.get('num_jobs_static', 8)
        self.num_jobs_dynamic = global_config.get('num_jobs_dynamic', 2)
        self.num_ops_per_job = global_config.get('num_ops_per_job', 5)
        self.num_machines = global_config.get('num_machines', 10)
        self.num_workers = global_config.get('num_workers', 5)
        self.total_pieces_per_job = global_config.get('total_pieces_per_job', 10)

        # 网络参数（JSON中没有，使用默认值）
        self.fusion_dims = [128, 64, 32]
        self.score_dims = [32, 16]
        self.value_hidden_dims = [128, 64, 32]

        # 训练参数（JSON中没有，使用默认值）
        self.actor_lr = 2e-4
        self.critic_lr = 2e-4
        self.hdgat_lr = 2e-4
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.entropy_coef = 0.01
        self.temperature_start = 0.9
        self.temperature_end = 0.15
        self.logit_scale_start = 10.0
        self.logit_scale_end = 30.0
        self.makespan_ema_beta = 0.995
        self.clip_value = 0.2
        self.value_coef = 0.05
        self.batch_size = 128
        self.update_epochs = 4
        self.target_kl = 0.02
        self.energy_penalty_coef = 0.01

        # 显存优化：关闭候选动作多头自注意力，避免 Top-K 较大时产生 K×K 注意力开销
        self.use_action_attention = False

        # 消融实验：去掉剪枝策略（启发式工人筛选 + Top-K 截断）
        self.use_pruning_strategy = False
        # 随机采样上限：候选动作超过此值时随机截断，避免组合爆炸导致 OOM
        self.max_ablation_candidates = 200

        # 加载机器数据（包含能耗参数）
        machine_static = data.get('machine_static_data', {})
        self.machine_data = {}
        for machine_key, machine_info in machine_static.items():
            machine_id = int(machine_key.split('_')[1])
            self.machine_data[machine_id] = machine_info

        # 加载工人数据
        worker_static = data.get('worker_static_data', {})
        self.worker_data = {}
        for worker_key, worker_info in worker_static.items():
            worker_id = int(worker_key.split('_')[1])
            self.worker_data[worker_id] = {
                'lambda': worker_info.get('lambda', 0.05),
                'mu': worker_info.get('mu', 0.1)
            }

        # 加载工件工序数据
        job_data = data.get('job_operation_static_data', {})
        self.job_data = {}
        self.job_ready_triggers = {}  # 存储每个动态工件的到达触发比例

        for job_key, job_info in job_data.items():
            job_id = int(job_key.split('_')[1])
            self.job_data[job_id] = {}

            # 检查是否为动态工件（第一道工序是否有ready_trigger）
            first_op = job_info.get('op_0', {})
            if 'ready_trigger' in first_op:
                self.job_ready_triggers[job_id] = first_op['ready_trigger']

            for op_key, op_info in job_info.items():
                op_id = int(op_key.split('_')[1])

                # 转换字符串键为整数
                if 'base_process_time' in op_info:
                    base_pt = op_info['base_process_time']
                    base_pt_int = {int(k): v for k, v in base_pt.items()}
                    op_info['base_process_time'] = base_pt_int

                if 'machine_candidates' in op_info:
                    machine_candidates = [int(x) for x in op_info['machine_candidates']]
                    op_info['machine_candidates'] = machine_candidates

                self.job_data[job_id][op_id] = op_info

    def _validate_dims(self):
        """校验维度参数合法性"""
        dims = [self.d_global, self.d_operation, self.d_batch, self.d_worker]
        for dim in dims:
            if not isinstance(dim, int) or dim <= 0:
                raise ValueError(f"维度参数必须为正整数，当前值：{dim}")


# 工序类
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

        # 工序级完整转移相关
        self.predecessors = []
        self.successors = []
        self.ready_time = 0

        # 工序级状态
        self.all_predecessors_completed = False
        self.waiting_time = 0

    def get_prev_op_completion_time(self):
        """获取上一个工序的完成时间"""
        if self.prev_op:
            return self.prev_op.complete_time
        return 0

    def are_all_predecessors_completed(self):
        """检查所有前序工序是否全部完成"""
        for prev_op in self.predecessors:
            if not prev_op.is_scheduled or prev_op.completed_pieces < prev_op.total_pieces:
                return False
        return True

    def update_predecessor_status(self):
        """更新前序工序状态"""
        self.all_predecessors_completed = self.are_all_predecessors_completed()
        if self.all_predecessors_completed and self.ready_time == 0:
            self.ready_time = max([prev_op.complete_time for prev_op in self.predecessors] + [0])

    def is_ready(self):
        """检查工序是否就绪"""
        return (self.all_predecessors_completed and
                self.completed_pieces < self.total_pieces and
                not self.is_scheduled)

    def mark_completed(self, completion_time):
        """标记工序完成"""
        self.is_scheduled = True
        self.complete_time = completion_time
        self.completed_pieces = self.total_pieces

        for succ in self.successors:
            succ.update_predecessor_status()

    def update_completion(self, new_completed_pieces, completion_time=None):
        """更新工序完成情况"""
        self.completed_pieces += new_completed_pieces

        if self.completed_pieces >= self.total_pieces:
            self.is_scheduled = True
            if completion_time:
                self.complete_time = max(self.complete_time, completion_time)
                for succ in self.successors:
                    succ.update_predecessor_status()


# 子批数目节点类：图中的子批节点，B_k 表示该工序分为 k 批
class BatchCountNode:
    def __init__(self, split_count):
        self.split_count = int(split_count)
        self.node_type = "batch_count"


# 实际加工子批类：仅用于动作执行中的加工时间、能耗和疲劳计算，不再加入异构图 batch_nodes
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
        self.is_minimal_unit = True


# 工人类
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

        # 时间跟踪
        self.last_fatigue_update_time = 0.0
        self.next_available_time = 0.0
        self.config = None

    def update_fatigue_after_work(self, actual_time, start_time):
        """加工完成后更新疲劳度"""
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
        """计算从当前疲劳度恢复到目标值所需时间"""
        if self.current_fatigue <= target_fatigue:
            return 0.0
        recovery_time = -np.log(target_fatigue / self.current_fatigue) / self.mu
        return round(max(0.0, recovery_time), 2)

    def assign_batch(self, batch):
        """分配子批给工人"""
        self.assigned_batches.append(batch)
        self.current_batch = batch
        self.is_busy = True
        self.current_machine = batch.machine_id

    def complete_batch(self, complete_time, start_time, actual_time):
        """完成当前子批"""
        if self.current_batch:
            self.update_fatigue_after_work(actual_time, start_time)
            self.current_batch.status = "completed"
            self.current_batch.complete_time = complete_time
            self.total_work_time += actual_time

        self.current_batch = None
        self.is_busy = False
        self.current_machine = None

    def is_available(self, check_time):
        """检查工人在指定时刻是否可用"""
        if self.is_busy:
            return False
        if check_time < self.next_available_time:
            return False
        if check_time >= self.next_available_time:
            self.is_resting = False
        return (not self.is_busy and
                not self.is_resting and
                self.current_fatigue < self.config.fatigue_threshold)

    def get_current_status(self, current_time):
        """获取工人当前状态信息"""
        is_avail = self.is_available(current_time)

        if self.is_busy:
            status = "忙碌中"
        elif self.is_resting:
            remaining_rest = max(0.0, self.next_available_time - current_time)
            status = f"休息中（疲劳度: {self.current_fatigue:.3f}, 剩余休息: {remaining_rest:.2f}）"
        else:
            status = f"可用（可用时间: {self.next_available_time:.2f}）"

        return {
            'worker_id': self.worker_id,
            'status': status,
            'fatigue': self.current_fatigue,
            'next_available': self.next_available_time,
            'is_busy': self.is_busy,
            'is_resting': self.is_resting,
            'total_work_time': self.total_work_time,
            'is_available': is_avail
        }


# MachineState类
class MachineState:
    def __init__(self, machine_id, config=None):
        self.machine_id = machine_id
        self.config = config
        self.is_busy = False
        self.next_available_time = 0.0
        self.current_batch = None
        self.total_processing_time = 0.0

        # 能耗累加器
        self.total_idle_energy = 0.0
        self.total_switch_energy = 0.0
        self.total_process_energy = 0.0

        # 标记是否为该机器的第一次加工
        self.is_first_use = True

    def assign_batch(self, batch, start_time, process_time):
        """分配子批给机器"""
        self.is_busy = True
        self.current_batch = batch
        self.next_available_time = start_time + process_time
        self.total_processing_time += process_time

        if self.config and self.machine_id in self.config.machine_data:
            energy_process = self.config.machine_data[self.machine_id].get('energy_process', 0)
            self.total_process_energy += process_time * energy_process

    def complete_batch(self):
        """完成当前子批"""
        self.is_busy = False
        self.current_batch = None

    def add_idle_energy(self, idle_time):
        """累加待机能耗"""
        if idle_time > 0 and self.config and self.machine_id in self.config.machine_data:
            energy_idle = self.config.machine_data[self.machine_id].get('energy_idle', 0)
            self.total_idle_energy += idle_time * energy_idle

    def add_switch_energy(self, switch_time):
        """累加切换能耗"""
        if switch_time > 0 and self.config and self.machine_id in self.config.machine_data:
            energy_switch = self.config.machine_data[self.machine_id].get('energy_switch', 0)
            self.total_switch_energy += switch_time * energy_switch

    @property
    def total_energy(self):
        """总能耗"""
        return self.total_idle_energy + self.total_switch_energy + self.total_process_energy

    def get_energy_state_dict(self):
        """获取能耗状态字典"""
        return {
            'machine_id': self.machine_id,
            'total_idle_energy': self.total_idle_energy,
            'total_switch_energy': self.total_switch_energy,
            'total_process_energy': self.total_process_energy,
            'is_first_use': self.is_first_use,
            'next_available_time': self.next_available_time,
            'total_processing_time': self.total_processing_time
        }

    def load_energy_state_dict(self, state_dict):
        """加载能耗状态"""
        self.total_idle_energy = state_dict['total_idle_energy']
        self.total_switch_energy = state_dict['total_switch_energy']
        self.total_process_energy = state_dict['total_process_energy']
        self.is_first_use = state_dict['is_first_use']
        self.next_available_time = state_dict['next_available_time']
        self.total_processing_time = state_dict['total_processing_time']


# 异构析取图实现
class HeterogeneousDisjunctiveGraph:
    def __init__(self, config: Config):
        self.config = config
        self.start_node = "S"
        self.end_node = "E"
        self.operation_nodes = {}
        # 固定建立 M 个子批数目节点，M 为当前实例机器数。
        # B1 表示分为 1 批，B2 表示分为 2 批，...，BM 表示分为 M 批。
        # 注意：真实加工子批不再放入 batch_nodes，而是在执行动作时临时生成。
        self.batch_nodes = {
            f"B{k}": BatchCountNode(split_count=k)
            for k in range(1, self.config.num_machines + 1)
        }
        self.worker_nodes = {}

        self.temporal_arcs = []
        self.batch_assoc_arcs = []
        self.worker_assign_arcs = []

        self.current_time = 0

    def add_operation(self, op: Operation):
        """添加工序节点并建立时序弧"""
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
        """添加工人节点"""
        self.worker_nodes[worker.worker_id] = worker

    def create_batch_nodes(self, op_key: str, B_k: int, W_set: set, machine_ids: list):
        """
        根据动作选择分批数目节点 B_k，并生成实际加工子批。

        图结构中：
            op_key -> B{k}
        表示该工序选择分为 k 批。

        注意：真实加工子批只作为临时对象返回，不再加入 self.batch_nodes，
        以避免子批节点随调度步数不断增长导致显存占用增加。
        """
        op = self.operation_nodes[op_key]
        remaining_pieces = op.total_pieces - op.completed_pieces

        B_k = int(B_k)
        W_set = set(W_set)

        if remaining_pieces <= 0:
            return []

        if B_k < 1 or B_k > self.config.num_machines:
            raise ValueError(f"分批数 B_k={B_k} 不合法，应在 1 到机器数 {self.config.num_machines} 之间")

        if len(W_set) != B_k:
            raise ValueError(f"工人集合大小({len(W_set)})必须等于分批数目({B_k})")

        if len(machine_ids) < B_k:
            raise ValueError(f"机器列表大小({len(machine_ids)})小于分批数目({B_k})")

        machine_ids = list(machine_ids)[:B_k]
        batch_node_key = f"B{B_k}"
        if batch_node_key not in self.batch_nodes:
            raise ValueError(f"分批数目节点 {batch_node_key} 不存在")

        # 一个工序只保留一条分批数选择边，防止重复执行/重置时累积脏边。
        self.batch_assoc_arcs = [arc for arc in self.batch_assoc_arcs if arc[0] != op_key]
        self.batch_assoc_arcs.append((op_key, batch_node_key, True))

        # 动态确定各真实加工子批尺寸：加工时间越短的机器承担更多件数。
        capabilities = []
        for machine_id in machine_ids:
            base_time = op.base_pt.get(machine_id, 1.0)
            capability = 1.0 / (base_time + 1e-6)
            capabilities.append(capability)

        total_cap = sum(capabilities)

        if total_cap <= 0:
            base_size = remaining_pieces // B_k
            sizes = [base_size] * B_k
            remainder = remaining_pieces % B_k
            for i in range(remainder):
                sizes[i] += 1
        else:
            sizes = []
            for i in range(B_k - 1):
                size = int(round(remaining_pieces * capabilities[i] / total_cap))
                sizes.append(max(1, size))

            last_size = remaining_pieces - sum(sizes)
            sizes.append(max(1, last_size))

            # 极端情况下 round 可能导致最后一批为负，做保底修正。
            if sizes[-1] < 0:
                while sizes[-1] < 1:
                    changed = False
                    for i in range(B_k - 1):
                        if sizes[i] > 1:
                            sizes[i] -= 1
                            sizes[-1] += 1
                            changed = True
                            if sizes[-1] >= 1:
                                break
                    if not changed:
                        break

        # 修正总件数，保证所有真实子批件数之和等于 remaining_pieces。
        if sum(sizes) != remaining_pieces:
            sizes[-1] = remaining_pieces - sum(sizes[:-1])

        # 低疲劳工人优先处理大子批。
        sorted_workers = sorted(list(W_set), key=lambda w_id: self.worker_nodes[w_id].current_fatigue)
        machine_size_pairs = list(zip(machine_ids, sizes))
        sorted_pairs = sorted(machine_size_pairs, key=lambda x: x[1], reverse=True)

        real_batches = []
        for i, (machine_id, batch_size) in enumerate(sorted_pairs):
            if batch_size <= 0 or i >= len(sorted_workers):
                continue

            worker_id = sorted_workers[i]
            batch = SubBatch(op_key, i, batch_size, machine_id, worker_id)
            batch.split_count = B_k
            real_batches.append(batch)

            # 记录 B_k 与工人的适配关系；当前 WorkerBatchAttention 未直接使用该边，保留用于日志/后续扩展。
            fitness = self._calc_worker_fitness(worker_id, batch)
            self.worker_assign_arcs.append((op_key, batch_node_key, worker_id, fitness))

        return real_batches

    def get_ready_operations(self):
        """获取所有就绪的工序"""
        ready_ops = []
        for op_key, op in self.operation_nodes.items():
            if op.is_ready():
                ready_ops.append(op_key)
        return ready_ops

    def update_operation_status(self):
        """更新所有工序的状态"""
        for op in self.operation_nodes.values():
            op.update_predecessor_status()

    def _calc_worker_fitness(self, worker_id: int, batch: SubBatch):
        """计算工人-子批适配度"""
        worker = self.worker_nodes[worker_id]
        fatigue_factor = 1 - worker.current_fatigue / self.config.fatigue_threshold
        return fatigue_factor

    def check_machine_availability(self, op_key: str, machine_states: dict):
        """检查工序的所有候选机器是否都可用"""
        op = self.operation_nodes[op_key]

        for machine_id in op.machine_candidates:
            if machine_id in machine_states:
                machine_state = machine_states[machine_id]
                if machine_state.is_busy:
                    return False
            else:
                return False

        return True

    def get_spt_machine_order(self, op_key: str, machine_states: dict):
        """获取按SPT规则排序的可用机器列表"""
        op = self.operation_nodes[op_key]

        machine_times = []
        for machine_id in op.machine_candidates:
            if machine_id in machine_states:
                base_time = op.base_pt.get(machine_id, float('inf'))
                machine_times.append((machine_id, base_time))

        sorted_machines = sorted(machine_times, key=lambda x: x[1])
        return [machine_id for machine_id, _ in sorted_machines]


# HDGAT编码器
class OperationBatchAttention(nn.Module):
    """模块A：工序-子批局部注意力"""

    def __init__(self, d_op: int, d_batch: int):
        super().__init__()
        self.d_op = d_op
        self.d_batch = d_batch
        self.unified_dim = max(d_op, d_batch)

        self.op_proj = nn.Linear(d_op, self.unified_dim)
        self.batch_proj = nn.Linear(d_batch, self.unified_dim)

        self.W_O = nn.Linear(self.unified_dim, self.unified_dim)
        self.a_O = nn.Parameter(torch.randn(self.unified_dim * 2))
        self.W_B = nn.Linear(self.unified_dim, self.unified_dim)
        self.a_OB = nn.Parameter(torch.randn(self.unified_dim * 2))
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.op_reduce = nn.Linear(self.unified_dim, self.d_op)
        self.batch_reduce = nn.Linear(self.unified_dim, self.d_batch)

    def forward(self, op_embeds, batch_embeds, op_batch_mask):
        num_ops, num_batches = op_embeds.shape[0], batch_embeds.shape[0]

        op_unified = self.op_proj(op_embeds)
        batch_unified = self.batch_proj(batch_embeds)

        op_batch_pair = torch.cat([
            op_unified.unsqueeze(1).repeat(1, num_batches, 1),
            self.W_B(batch_unified).unsqueeze(0).repeat(num_ops, 1, 1)
        ], dim=-1)
        op_batch_attn = self.leaky_relu(torch.matmul(op_batch_pair, self.a_OB))
        valid_rows = (op_batch_mask.sum(dim=-1, keepdim=True) > 0).float()
        op_batch_attn = op_batch_attn.masked_fill(op_batch_mask == 0, -1e9)
        op_batch_weights = F.softmax(op_batch_attn, dim=-1) * valid_rows

        op_updated_unified = op_unified + torch.matmul(op_batch_weights, batch_unified)
        batch_op_weights = op_batch_weights.T
        batch_op_weights = batch_op_weights / (batch_op_weights.sum(dim=-1, keepdim=True) + 1e-8)
        batch_updated_unified = batch_unified + torch.matmul(batch_op_weights, op_unified)

        op_updated = self.op_reduce(op_updated_unified)
        batch_updated = self.batch_reduce(batch_updated_unified)

        return op_updated, batch_updated


class WorkerBatchAttention(nn.Module):
    """模块B：工人-子批全局注意力（去掉额外鲁棒全局特征，仅做节点交互）"""

    def __init__(self, d_worker: int, d_batch: int, d_global: int = 0):
        super().__init__()
        self.d_worker = d_worker
        self.d_batch = d_batch
        self.attn_weights = nn.Parameter(torch.randn(d_worker + d_batch, 1))
        nn.init.xavier_uniform_(self.attn_weights)
        self.W_w = nn.Linear(d_worker, d_worker)
        self.W_b = nn.Linear(d_batch, d_batch)
        self.b = nn.Parameter(torch.zeros(1))
        self.W_w_prime = nn.Linear(d_worker, d_worker)
        self.W_b_prime = nn.Linear(d_batch, d_batch)
        self.b_prime = nn.Parameter(torch.zeros(1))

    def forward(self, worker_embeds, batch_embeds):
        num_workers, num_batches = worker_embeds.shape[0], batch_embeds.shape[0]
        if num_workers == 0 or num_batches == 0:
            return worker_embeds, batch_embeds
        w = self.W_w(worker_embeds)
        b = self.W_b(batch_embeds)
        pair = torch.cat([w.unsqueeze(1).repeat(1, num_batches, 1), b.unsqueeze(0).repeat(num_workers, 1, 1)], dim=-1) + self.b
        worker_to_batch = torch.matmul(torch.tanh(pair), self.attn_weights).squeeze(-1)
        w2 = self.W_w_prime(worker_embeds)
        b2 = self.W_b_prime(batch_embeds)
        pair2 = torch.cat([b2.unsqueeze(1).repeat(1, num_workers, 1), w2.unsqueeze(0).repeat(num_batches, 1, 1)], dim=-1) + self.b_prime
        batch_to_worker = torch.matmul(torch.tanh(pair2), self.attn_weights).squeeze(-1)
        affinity = (worker_to_batch + batch_to_worker.T) / 2
        worker_weights = F.softmax(affinity, dim=-1)
        batch_weights = F.softmax(affinity.T, dim=-1)
        return worker_embeds + torch.matmul(worker_weights, batch_embeds), batch_embeds + torch.matmul(batch_weights, worker_embeds)

class HDGAT(nn.Module):
    """轻量化 HDGAT：工序6维、子批3维、工人5维，全局状态5维。"""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.op_embed = nn.Linear(6, self.config.d_operation)
        self.batch_embed = nn.Linear(3, self.config.d_batch)
        self.worker_embed = nn.Linear(5, self.config.d_worker)
        self.op_batch_attn = OperationBatchAttention(self.config.d_operation, self.config.d_batch)
        self.worker_batch_attn = WorkerBatchAttention(self.config.d_worker, self.config.d_batch)
        self.agent_ref = None

    def set_agent(self, agent):
        self.agent_ref = agent

    def _extract_features(self, hdg: HeterogeneousDisjunctiveGraph, device):
        current_cmax = max([o.complete_time for o in hdg.operation_nodes.values()] + [0])
        total_ops = len(hdg.operation_nodes)
        op_features_list, op_keys_list = [], []
        for op_key, op in hdg.operation_nodes.items():
            pt_values = list(op.base_pt.values()) if op.base_pt else [0.0]
            min_pt, max_pt = min(pt_values), max(pt_values)
            remaining_ops_in_job = sum(1 for o in hdg.operation_nodes.values() if o.job_id == op.job_id and not o.is_scheduled)
            op_features_list.append([
                1.0 if op.is_scheduled else 0.0,
                1.0 if op.is_ready() else 0.0,
                min_pt / 5.0,
                max_pt / 5.0,
                (max_pt - min_pt) / 5.0,
                remaining_ops_in_job / max(1, self.config.num_ops_per_job)
            ])
            op_keys_list.append(op_key)
        op_features = torch.tensor(op_features_list, dtype=torch.float32, device=device) if op_features_list else torch.zeros((0, 6), dtype=torch.float32, device=device)

        batch_features_list, batch_keys_list = [], []
        max_k = max(1, self.config.num_machines)
        select_counts = getattr(self.agent_ref, 'batch_select_count', defaultdict(int)) if self.agent_ref else defaultdict(int)
        total_select_count = sum(select_counts.values())
        for batch_key, batch in hdg.batch_nodes.items():
            k = int(getattr(batch, 'split_count', max(1, getattr(batch, 'batch_id', 0) + 1)))
            k = max(1, k)
            batch_features_list.append([
                min(1.0, k / max_k),
                1.0 / k,
                select_counts.get(k, 0) / max(1, total_select_count)
            ])
            batch_keys_list.append(batch_key)
        batch_features = torch.tensor(batch_features_list, dtype=torch.float32, device=device) if batch_features_list else torch.zeros((0, 3), dtype=torch.float32, device=device)

        worker_features_list = []
        cmax_norm = self.agent_ref.cmax_norm_factor if self.agent_ref else 1.0
        for _, worker in hdg.worker_nodes.items():
            worker_features_list.append([
                worker.lambda_, worker.mu, worker.current_fatigue,
                worker.next_available_time / max(1e-6, cmax_norm),
                worker.total_work_time / max(1e-6, cmax_norm)
            ])
        worker_features = torch.tensor(worker_features_list, dtype=torch.float32, device=device) if worker_features_list else torch.zeros((0, 5), dtype=torch.float32, device=device)

        workers = list(hdg.worker_nodes.values())
        completed_ops = sum(1 for o in hdg.operation_nodes.values() if o.is_scheduled)
        progress = completed_ops / max(1, total_ops)
        non_fatigue_ratio = sum(1 for w in workers if w.current_fatigue < self.config.fatigue_threshold) / max(1, len(workers))
        total_energy = sum(m.total_energy for m in self.agent_ref.machine_states.values()) if self.agent_ref else 0.0
        avg_fatigue = float(np.mean([w.current_fatigue for w in workers])) if workers else 0.0
        self.stats_vector = torch.tensor([
            progress, non_fatigue_ratio,
            current_cmax / max(1e-6, self.agent_ref.cmax_norm_factor if self.agent_ref else 1.0),
            total_energy / max(1e-6, self.agent_ref.energy_norm_factor if self.agent_ref else 1.0),
            avg_fatigue
        ], dtype=torch.float32, device=device)
        return op_features, batch_features, worker_features, op_keys_list, batch_keys_list

    def _build_op_batch_mask(self, hdg, op_keys, batch_keys, device):
        """
        根据工序-分批数目节点边构造 mask。
        若存在 op_key -> Bk，则表示该工序采用 k 子批划分方案。
        """
        mask = torch.zeros((len(op_keys), len(batch_keys)), dtype=torch.float32, device=device)
        edge_set = set()
        for arc in hdg.batch_assoc_arcs:
            if len(arc) >= 2:
                edge_set.add((arc[0], arc[1]))

        for i, o_key in enumerate(op_keys):
            for j, b_key in enumerate(batch_keys):
                if (o_key, b_key) in edge_set:
                    mask[i, j] = 1.0
        return mask

    def extract_raw_features(self, hdg: HeterogeneousDisjunctiveGraph, device):
        op_feats, batch_feats, worker_feats, op_keys, batch_keys = self._extract_features(hdg, device)
        return {
            'op_feats': op_feats.detach().cpu(),
            'batch_feats': batch_feats.detach().cpu(),
            'worker_feats': worker_feats.detach().cpu(),
            'op_batch_mask': self._build_op_batch_mask(hdg, op_keys, batch_keys, device).detach().cpu(),
            'stats_vector': self.stats_vector.detach().cpu()
        }

    def encode_raw_features(self, raw, device):
        op_feats = raw['op_feats'].to(device)
        batch_feats = raw['batch_feats'].to(device)
        worker_feats = raw['worker_feats'].to(device)
        op_batch_mask = raw['op_batch_mask'].to(device)
        stats_vector = raw['stats_vector'].to(device)
        op_embeds = self.op_embed(op_feats) if op_feats.size(0) > 0 else torch.zeros((0, self.config.d_operation), device=device)
        batch_embeds = self.batch_embed(batch_feats) if batch_feats.size(0) > 0 else torch.zeros((0, self.config.d_batch), device=device)
        worker_embeds = self.worker_embed(worker_feats) if worker_feats.size(0) > 0 else torch.zeros((0, self.config.d_worker), device=device)
        if op_embeds.size(0) > 0 and batch_embeds.size(0) > 0:
            op_embeds, batch_embeds = self.op_batch_attn(op_embeds, batch_embeds, op_batch_mask)
        if worker_embeds.size(0) > 0 and batch_embeds.size(0) > 0:
            worker_embeds, batch_embeds = self.worker_batch_attn(worker_embeds, batch_embeds)
        op_pool = op_embeds.mean(dim=0) if op_embeds.size(0) > 0 else torch.zeros(self.config.d_operation, device=device)
        batch_pool = batch_embeds.mean(dim=0) if batch_embeds.size(0) > 0 else torch.zeros(self.config.d_batch, device=device)
        worker_pool = worker_embeds.mean(dim=0) if worker_embeds.size(0) > 0 else torch.zeros(self.config.d_worker, device=device)
        return torch.cat([stats_vector, op_pool, batch_pool, worker_pool], dim=-1)

    def forward(self, hdg: HeterogeneousDisjunctiveGraph, device):
        raw = self.extract_raw_features(hdg, device)
        return None, self.encode_raw_features(raw, device)

# Actor网络组件
class FeatureFusionEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, x):
        return self.network(x)


class ActionSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, dropout=0.1
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # 输入 x: (Batch, Seq_Len, Embed_Dim)
        # MultiheadAttention 原生支持 Batch 维度
        attn_output, _ = self.multihead_attn(x, x, x)
        return self.norm(x + attn_output)


class ScoringHead(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.GELU()])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


# Actor网络
class ActorNetwork(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.d_action = 15

        # 图级状态 = 5维全局状态 + 工序池化 + 子批池化 + 工人池化
        self.d_global = 5 + config.d_operation + config.d_batch + config.d_worker

        self.d_input = self.d_global + self.d_action

        self.fusion_encoder = FeatureFusionEncoder(self.d_input, config.fusion_dims)

        # 显存优化：默认关闭候选动作多头自注意力。
        # 关闭后，Actor 直接基于“图级状态 + 动作特征”的融合表示进行前馈评分，
        # 可显著降低 Top-K 较大时的显存占用。
        self.use_attention = getattr(config, 'use_action_attention', False)
        self.action_attn = ActionSelfAttention(config.fusion_dims[-1]) if self.use_attention else None

        self.scoring_head = ScoringHead(config.fusion_dims[-1], config.score_dims)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, global_feature, action_features_batch, progress=1.0, mask=None):
        """
        优化版前向传播：支持批处理输入
        Args:
            global_feature: (B, D_global) 或 (D_global)
            action_features_batch: (B, K, D_action) 或 list of tensors
            progress: 训练进度
        """
        # 兼容性处理：如果是单样本，扩展维度
        if global_feature.dim() == 1:
            global_feature = global_feature.unsqueeze(0)

        # 统一转换为 Tensor (B, K, D_action)
        if isinstance(action_features_batch, list):
            # List 输入通常用于单步推理 (Batch=1)
            N = len(action_features_batch)
            if N == 0: return torch.tensor([]), torch.tensor([])

            # 堆叠并增加 Batch 维度: List -> (N, D) -> (1, N, D)
            action_feats = torch.stack(action_features_batch).unsqueeze(0)

            # 扩展 global_feature: (1, D_g) -> (1, N, D_g)
            global_feats = global_feature.unsqueeze(1).expand(-1, N, -1)
        else:
            # 批处理模式: Tensor (B, K, D_action)
            action_feats = action_features_batch
            B, K, _ = action_feats.shape

            # 扩展 global_feature: (B, D_g) -> (B, K, D_g)
            global_feats = global_feature.unsqueeze(1).expand(-1, K, -1)

        # 此时 global_feats 和 action_feats 必然是 3维 (B, K, ...)
        B, K, D_a = action_feats.shape

        # 展平用于 Fusion Encoder
        flat_action_feats = action_feats.view(B * K, -1)

        # fusion_encoder 输入: (B*K, D_in)
        fused_input = torch.cat([global_feats.reshape(B * K, -1), flat_action_feats], dim=-1)
        action_embeddings = self.fusion_encoder(fused_input)  # (B*K, D_hid)

        # Reshape 回以进行组内注意力
        action_embeddings = action_embeddings.view(B, K, -1)

        if self.use_attention and self.action_attn is not None and K > 1:
            action_embeddings = self.action_attn(action_embeddings)

        # Scoring Head
        scores = self.scoring_head(action_embeddings.view(B * K, -1)).view(B, K)  # (B, K)

        # 温度参数与 Softmax
        temp_start = 1.0
        temp_end = 0.2
        temp = temp_end + (temp_start - temp_end) * ((1.0 - float(progress)) ** 2)
        temp = max(temp_end, float(temp))

        logits = (scores * 10) / temp

        # Mask 处理
        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e9)

        logits = logits - logits.max(dim=-1, keepdim=True)[0]  # 数值稳定
        action_probs = F.softmax(logits, dim=-1)

        return scores, action_probs

    def get_action(self, global_feature, action_features_list, progress=1.0, deterministic=False, mask=None):
        # 兼容空输入
        if isinstance(action_features_list, list) and len(action_features_list) == 0:
            return None, 0.0, 0.0
        if isinstance(action_features_list, torch.Tensor) and action_features_list.shape[1] == 0:
            return None, 0.0, 0.0

        scores, probs = self(global_feature, action_features_list, progress=progress, mask=mask)

        # ===== 修改点：移除 Batch 维度 (B=1) =====
        # forward 返回的 probs 形状为 (B, K)，单步推理时 B=1
        if probs.dim() == 2 and probs.shape[0] == 1:
            probs = probs.squeeze(0)  # (1, K) -> (K,)

        if deterministic:
            selected_idx = torch.argmax(probs).item()
            entropy = None
        else:
            probs = (probs + 1e-9) / (probs + 1e-9).sum()
            dist = Categorical(probs=probs)
            selected_idx = dist.sample().item()
            entropy = dist.entropy().mean().item()

        return selected_idx, probs[selected_idx].item(), entropy


class CriticNetwork(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        layers = []

        # 图级状态 = 5维全局状态 + 工序池化 + 子批池化 + 工人池化
        prev_dim = 5 + config.d_operation + config.d_batch + config.d_worker

        for hidden_dim in config.value_hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            ])
            if prev_dim != config.value_hidden_dims[-1]:
                layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        self.feature_extractor = nn.Sequential(*layers)
        self.value_head = nn.Linear(prev_dim, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if name == 'value_head':
                    nn.init.orthogonal_(module.weight, gain=0.01)
                else:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, global_features):
        if global_features.dim() == 1:
            global_features = global_features.unsqueeze(0)
        features = self.feature_extractor(global_features)
        value = self.value_head(features)
        if global_features.size(0) == 1:
            value = value.squeeze(0)
        return value


class AdaptiveWeightCritic(CriticNetwork):
    def __init__(self, config: Config):
        super().__init__(config)
        del self.value_head
        self.value_heads = nn.ModuleList([
            nn.Linear(config.value_hidden_dims[-1], 1) for _ in range(3)
        ])

    def forward(self, global_features):
        features = self.feature_extractor(global_features)  # (B, H)
        values = torch.cat([head(features) for head in self.value_heads], dim=-1)  # (B, 3)

        # ⚠️ 新增：使用基于状态的动态硬逻辑生成权重
        with torch.no_grad():  # 确保权重不参与也不干扰反向传播
            # 根据你的 HDGAT 代码，global_features 的前 8 维是 stats_vector
            # stats_vector = [progress, availability_ratio, energy_norm, cmax_norm, avg_fatigue, ready_ratio, 0.0, 0.0]
            # 提取当前三大目标的归一化状态表现 (值越大，说明当前该项表现越差)
            cmax_state = global_features[:, 2]  # stats_vector[2] 是 Cmax归一化值
            energy_state = global_features[:, 3]  # stats_vector[3] 是 TEC归一化值
            fatigue_state = global_features[:, 4]  # stats_vector[4] 是 Favg

            # 我们放大系数（如乘以2.0或3.0）以增加权重在不同状态下的区分度 (Temperature Scaling)
            temperature = 2.0
            raw_weights = torch.stack([
                cmax_state * temperature,
                energy_state * temperature,
                fatigue_state * temperature
            ], dim=-1)

            # 使用 Softmax 输出动态概率作为权重，总和永远为 1.0
            weights = F.softmax(raw_weights, dim=-1)

        return values, weights


# 探索策略
class AdaptiveExploration:
    def __init__(self, initial_temp=0.5, final_temp=0.05, decay_steps=10000):
        self.temperature = initial_temp
        self.final_temp = final_temp
        self.decay_rate = (initial_temp - final_temp) / decay_steps

    def apply_temperature(self, logits):
        return logits / self.temperature

    def step(self):
        self.temperature = max(self.final_temp, self.temperature - self.decay_rate)

    def get_entropy_coefficient(self):
        return 0.01 * self.temperature


# 回报计算器
class ReturnCalculator:
    def __init__(self, gamma=0.99, gae_lambda=0.95):
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def compute_returns(self, rewards, values, next_values, dones):
        """
        参数:
            rewards: (T, 3) 三维奖励
            values:   (T, 3) 三维价值
            next_values: (T, 3) 下一状态价值
            dones:    (T, 1) 终止标志
        返回:
            returns: (T, 3) 多维折扣回报
            advantages: (T, 3) 多维优势
        """
        T, dim = rewards.shape
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        last_gae_lam = torch.zeros(dim, device=rewards.device)

        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - dones[t]
                next_v = next_values[t]
            else:
                next_non_terminal = 1.0 - dones[t]
                next_v = values[t + 1]

            delta = rewards[t] + self.gamma * next_v * next_non_terminal - values[t]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam

        returns = advantages + values
        return returns, advantages


# 经验缓冲区
class ReplayBuffer:
    def __init__(self, batch_size=32, device='cpu'):
        self.batch_size = batch_size
        self.buffer = []
        self.device = device

    def add(self, episode_experience, returns, advantages):
        for i, exp in enumerate(episode_experience):
            self.buffer.append({
                'state_raw': exp['state_raw'],
                'action_features': exp['action_features'].cpu().detach(),
                'mask': exp['mask'].cpu().detach(),
                'selected_idx': exp['selected_idx'],
                'prob': exp['prob'],
                'return': returns[i].cpu().detach(),
                'advantage': advantages[i].cpu().detach(),
            })

    def sample(self, device):
        if len(self.buffer) == 0:
            return {}
        sample_size = min(self.batch_size, len(self.buffer))
        indices = np.random.choice(len(self.buffer), sample_size, replace=False)
        batch = defaultdict(list)
        for idx in indices:
            for key, value in self.buffer[idx].items():
                batch[key].append(value)
        batch['prob'] = torch.tensor(batch['prob'], dtype=torch.float32, device=device)
        batch['return'] = torch.stack(batch['return']).to(device)
        batch['advantage'] = torch.stack(batch['advantage']).to(device)
        batch['selected_idx'] = torch.tensor(batch['selected_idx'], dtype=torch.long, device=device)

        # 多实例训练时，不同状态的候选动作数量 K 可能不同。
        # 对 action_features 和 mask 做 mini-batch 内 padding，避免 torch.stack 形状不一致。
        action_features_list = batch['action_features']
        mask_list = batch['mask']
        max_k = max(feat.shape[0] for feat in action_features_list)
        action_dim = action_features_list[0].shape[-1]

        padded_action_features = []
        padded_masks = []
        for feat, mask in zip(action_features_list, mask_list):
            k = feat.shape[0]
            if k < max_k:
                pad_feat = torch.zeros((max_k - k, action_dim), dtype=feat.dtype)
                pad_mask = torch.zeros(max_k - k, dtype=mask.dtype)
                feat = torch.cat([feat, pad_feat], dim=0)
                mask = torch.cat([mask, pad_mask], dim=0)
            padded_action_features.append(feat)
            padded_masks.append(mask)

        batch['action_features'] = torch.stack(padded_action_features).to(device)
        batch['mask'] = torch.stack(padded_masks).to(device)
        return dict(batch)

    def size(self):
        return len(self.buffer)

    def clear(self):
        self.buffer = []

# PPO智能体（修改动态工件到达逻辑，添加动态参数）
class PPOAgent:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        self.hdg = HeterogeneousDisjunctiveGraph(config)
        self.machine_states = {}

        # 动态工件相关属性
        self.pending_dynamic_jobs = []  # 存储 (job_id, ready_trigger) 元组
        self.total_episodes = 500
        # 累计已训练的 episode 数；用于继续训练时接着编号，不再依赖 training_history 长度。
        self.trained_episodes = 0

        # 创建模型
        self.hdgat = HDGAT(config).to(self.device)
        self.hdgat.set_agent(self)
        self.actor = ActorNetwork(config).to(self.device)
        self.critic = AdaptiveWeightCritic(config).to(self.device)

        self.optimizer = torch.optim.Adam([
            {'params': self.hdgat.parameters(), 'lr': config.hdgat_lr},
            {'params': self.actor.parameters(), 'lr': config.actor_lr},
            {'params': self.critic.parameters(), 'lr': config.critic_lr},
        ])
        self.exploration = AdaptiveExploration(initial_temp=0.2, final_temp=0.01)
        self.return_calc = ReturnCalculator(config.gamma, config.gae_lambda)
        self.buffer = ReplayBuffer(config.batch_size, self.device)
        self.batch_select_count = defaultdict(int)

        self.entropy_history = []

        self.best_makespan = float('inf')
        self.best_energy = float('inf')
        self.best_fatigue_mean = float('inf')

        # 新增：EMA 平滑值（初始为 None）
        self.ema_makespan = None
        self.ema_energy = None
        self.ema_fatigue_mean  = None

        # 从配置中读取 beta，默认为 0.995
        self.ema_beta = getattr(config, 'makespan_ema_beta', 0.995)
        # CSV日志记录相关
        self.csv_log_path = "training_log_wo_ps.csv"
        self.current_instance_name = ""
        self._init_csv_logger()

        self.training_history = {
            'episode': [],
            'reward': [],
            'makespan': [],
            'energy': [],
            'fatigue_mean': [],
            'completion_rate': [],
            'actor_loss': [],
            'critic_loss': [],
            'policy_entropy': [],
            # 新增：记录三个目标的自适应权重
            'weight_cmax': [],
            'weight_energy': [],
            'weight_fatigue': []
        }

        self.best_val_score = float('inf')
        self.best_val_metrics = None

        self.smooth_window = 10

        self.best_metrics = {'makespan': float('inf'), 'energy': float('inf'), 'fatigue_mean': float('inf')}

        # 计算动态参数
        self._compute_dynamic_parameters()

        self._validate_dimensions()

    def _cleanup_memory(self, aggressive=False):
        """清理 Python 垃圾和 CUDA 缓存，缓解长时间训练中的显存碎片与临时张量残留。"""
        if aggressive:
            gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _compute_dynamic_parameters(self):
        """计算所有动态参数：归一化参数、最大步数、Top-K动作数"""
        # 基础统计
        self.total_ops = self.config.num_jobs * self.config.num_ops_per_job

        # 1. 归一化参数
        self._compute_normalization_params()

        # 2. 最大步数
        self.max_steps_per_episode = self._compute_max_steps()

        # 3. Top-K动作数
        self.top_k_actions = self._compute_top_k_actions()

        # print(f"\n动态参数汇总:")
        # print(f"  总工序数: {self.total_ops}")
        # print(f"  最大步数: {self.max_steps_per_episode}")
        # print(f"  Top-K动作数: {self.top_k_actions}")
        # print(f"  time_norm: {self.time_norm_factor:.1f}")
        # print(f"  cmax_norm: {self.cmax_norm_factor:.1f}")
        # print(f"  energy_norm: {self.energy_norm_factor:.1f}")

    def _compute_normalization_params(self):
        """
        计算归一化参数 - 修正版：考虑工序并行度和实际加工时间
        """
        # 基础统计
        self.total_ops = self.config.num_jobs * self.config.num_ops_per_job
        pieces_per_job = self.config.total_pieces_per_job  # 通常是10

        # 估计平均每个工序的候选机器数
        # 从配置中统计所有工序的候选机器数
        total_candidates = 0
        total_ops_count = 0
        for job_id in range(self.config.num_jobs):
            if job_id in self.config.job_data:
                for op_id in range(self.config.num_ops_per_job):
                    if op_id in self.config.job_data[job_id]:
                        op_info = self.config.job_data[job_id][op_id]
                        candidates = op_info.get('machine_candidates', [])
                        total_candidates += len(candidates)
                        total_ops_count += 1

        avg_candidates_per_op = total_candidates / max(1, total_ops_count)
        # print('候选', avg_candidates_per_op)


        # 估计每个工序的实际加工时间
        avg_op_time = 2.0  # 平均单件加工时间

        per_op_total_time = avg_op_time * (10 / avg_candidates_per_op)

        # 最坏情况Cmax：所有工序串行加工
        worst_case_cmax = self.total_ops * per_op_total_time

        # 考虑分批开销
        batch_overhead = 1.2
        worst_case_cmax = worst_case_cmax * batch_overhead

        # 估计理论最大能耗
        avg_energy_per_time = 4.0  # 平均单位时间能耗

        # 最坏情况能耗：所有机器同时运行，且满负荷
        # 但实际中机器不会全部同时满负荷，所以用0.8系数调整
        machine_utilization = 0.8  # 机器利用率
        worst_case_energy = worst_case_cmax * self.config.num_machines * avg_energy_per_time * machine_utilization

        # 设置归一化参数；如果实例提供 reference_solution，则优先使用参考解指标
        self.time_norm_factor = max(10, worst_case_cmax * 0.1)
        self.cmax_norm_factor = max(100.0, worst_case_cmax)
        self.energy_norm_factor = max(2000.0, worst_case_energy)
        self.wait_norm_factor = max(10, worst_case_cmax * 0.1)
        if getattr(self.config, 'ref_cmax', None) is not None:
            self.cmax_norm_factor = max(1.0, float(self.config.ref_cmax))
            self.time_norm_factor = max(10.0, self.cmax_norm_factor * 0.5)
            self.wait_norm_factor = max(10.0, self.cmax_norm_factor * 0.2)
        if getattr(self.config, 'ref_tec', None) is not None:
            self.energy_norm_factor = max(1.0, float(self.config.ref_tec))
        self.fatigue_norm_factor = max(1e-6, float(self.config.ref_favg)) if getattr(self.config, 'ref_favg', None) is not None else 1.0
        self.batch_norm_factor = self.time_norm_factor

        # print(f"\n归一化参数计算:")
        # print(
        #     f"  问题规模: {self.config.num_jobs}个工件 × {self.config.num_ops_per_job}道工序 = {self.total_ops}道工序")
        # print(f"  每工序件数: {pieces_per_job}")
        # print(f"  平均候选机器数: {avg_candidates_per_op:.2f}")
        # print(f"  每工序总加工时间估计: {per_op_total_time:.1f}")
        # print(f"  机器数量: {self.config.num_machines}")
        # print(f"  最坏情况Cmax估计: {worst_case_cmax:.1f}")
        # print(f"  最坏情况能耗估计: {worst_case_energy:.1f}")
        # print(f"  time_norm_factor: {self.time_norm_factor:.1f}")
        # print(f"  cmax_norm_factor: {self.cmax_norm_factor:.1f}")
        # print(f"  energy_norm_factor: {self.energy_norm_factor:.1f}")

    def _compute_max_steps(self):
        """
        计算最大步数：工件总数 * 机器总数 * 2
        规则：工件总数 * 单个工件的工序数 * (机器总数 / 单个工件的工序数) * 2
        简化后：工件总数 * 机器总数 * 2
        """
        # 基础计算
        base_steps = self.config.num_jobs * self.config.num_machines * 2

        # 考虑分批因素
        batch_factor = min(1.5, self.config.num_machines / self.config.num_ops_per_job)

        # 考虑动态工件
        dynamic_factor = 1.3 if self.config.num_jobs_dynamic > 0 else 1.0

        # 最终最大步数
        max_steps = int(base_steps * batch_factor * dynamic_factor)

        # 设置合理范围
        min_steps = max(100, self.total_ops)
        max_steps = max(max_steps, min_steps)
        max_steps = min(max_steps, 3000)  # 绝对上限

        # base_steps = self.config.num_jobs * self.config.num_ops_per_job
        # max_steps = base_steps

        return max_steps

    def _compute_top_k_actions(self):
        """
        计算Top-K动作数：总工序数 × 20%
        规则：工件总数 * 单个工件的工序数 * 20%
        """
        base_top_k = int(self.total_ops * 0.1)

        # # 设置合理范围
        min_top_k = 5
        max_top_k = 100
        #
        top_k = max(min_top_k, base_top_k)
        top_k = min(max_top_k, top_k)

        return top_k

    def _validate_dimensions(self):
        """验证网络维度"""
        try:
            test_hdg = HeterogeneousDisjunctiveGraph(self.config)
            node_embeds, global_feature = self.hdgat(test_hdg, self.device)
            print(f"HDGAT测试通过: global_feature维度={global_feature.shape}")

            # 构造测试输入: (Batch=1, K=3, D_action)
            dummy_action_feats = torch.randn(1, 3, self.actor.d_action).to(self.device)
            dummy_mask = torch.ones(1, 3).to(self.device)

            # 调用 Actor (注意: 传入 Tensor 而非 List)
            scores, probs = self.actor(global_feature.unsqueeze(0), dummy_action_feats, mask=dummy_mask)
            print(f"Actor测试通过: scores维度={scores.shape}")
        except Exception as e:
            print(f"维度验证失败: {e}")
            raise

    def _select_workers_by_earliest(self, available_workers, B_k, prev_op_completion_time):
        """选择最早可用时间的工人"""
        current_time = prev_op_completion_time

        worker_available_times = []
        for w_id in available_workers:
            worker = self.hdg.worker_nodes[w_id]
            available_time = worker.next_available_time
            worker_available_times.append((w_id, available_time))

        worker_available_times.sort(key=lambda x: x[1])
        earliest_time = worker_available_times[0][1]
        earliest_workers = [w_id for w_id, time in worker_available_times if time == earliest_time]

        if len(earliest_workers) >= B_k:
            from itertools import combinations
            worker_combinations = []
            for combo in combinations(earliest_workers, B_k):
                worker_combinations.append(set(combo))
            return worker_combinations
        else:
            selected_workers = set(earliest_workers)
            remaining_needed = B_k - len(selected_workers)

            if remaining_needed > 0:
                remaining_workers = [(w_id, time) for w_id, time in worker_available_times
                                     if w_id not in selected_workers]
                remaining_workers.sort(key=lambda x: x[1])
                for w_id, _ in remaining_workers[:remaining_needed]:
                    selected_workers.add(w_id)

            return [selected_workers]

    def _generate_actions_for_op(self, op_key, prev_op_completion_time):
        """为单个工序生成候选动作"""
        op = self.hdg.operation_nodes[op_key]
        B_k = len(op.machine_candidates)

        all_workers = list(self.hdg.worker_nodes.keys())

        if len(all_workers) < B_k:
            print(f"警告: 工人数量({len(all_workers)})少于需求({B_k})")
            return []

        worker_combinations = self._select_workers_by_earliest(
            all_workers, B_k, prev_op_completion_time
        )

        action_candidates = []
        for worker_set in worker_combinations:
            action_candidates.append((op_key, B_k, worker_set))

        return action_candidates

    def _generate_valid_actions(self, hdg, episode=None):
        """生成候选动作，并进行 Padding 以支持批处理
        消融版：去掉剪枝策略 — 随机采样工人组合（无启发式偏好）
        """
        # 1. 生成候选动作列表
        all_candidates = []  # 存储
        hdg.update_operation_status()
        ready_ops = hdg.get_ready_operations()

        max_candidates = getattr(self.config, 'max_ablation_candidates', 200)

        for op_key in ready_ops:
            op = hdg.operation_nodes[op_key]
            if op.completed_pieces >= op.total_pieces:
                continue

            B_k = len(op.machine_candidates)
            if B_k == 0 or not hdg.check_machine_availability(op_key, self.machine_states):
                continue

            all_workers = list(self.hdg.worker_nodes.values())
            if len(all_workers) < B_k:
                continue

            machine_ids = self.hdg.get_spt_machine_order(op_key, self.machine_states)
            if len(machine_ids) < B_k:
                continue

            # === 消融修改：随机采样工人组合（替代启发式2组合） ===
            all_worker_ids = [w.worker_id for w in all_workers]
            total_combos = 1
            for i in range(B_k):
                total_combos = total_combos * (len(all_worker_ids) - i) // (i + 1)

            # 当组合数较小时直接枚举，组合数大时随机采样避免构建巨大列表
            sample_budget = max(10, max_candidates // max(1, len(ready_ops)))
            if total_combos <= sample_budget:
                sampled_combos = list(combinations(all_worker_ids, B_k))
            else:
                seen = set()
                sampled_combos = []
                attempts = 0
                max_attempts = sample_budget * 5
                while len(sampled_combos) < sample_budget and attempts < max_attempts:
                    combo = tuple(sorted(random.sample(all_worker_ids, B_k)))
                    if combo not in seen:
                        seen.add(combo)
                        sampled_combos.append(combo)
                    attempts += 1

            for worker_tuple in sampled_combos:
                worker_ready = max([self.hdg.worker_nodes[w].next_available_time for w in worker_tuple])
                op_ready = op.get_prev_op_completion_time()
                avg_base_pt = np.mean([op.base_pt.get(m, 1.0) for m in machine_ids[:B_k]])
                finish_time = max(worker_ready, op_ready) + (avg_base_pt * (op.total_pieces / B_k))
                max_fatigue = max([self.hdg.worker_nodes[w].current_fatigue for w in worker_tuple])

                action = (op_key, B_k, worker_tuple)
                all_candidates.append((action, finish_time, max_fatigue))

        if not all_candidates:
            return [], torch.zeros((0, self.actor.d_action), device=self.device), torch.zeros(0, device=self.device)

        # 2. 计算统计量
        finish_times = [ft for _, ft, _ in all_candidates]
        max_fatigues = [mf for _, _, mf in all_candidates]

        group_stats = {
            'min_finish': min(finish_times),
            'range_finish': max(finish_times) - min(finish_times) + 1e-8,
            'min_fatigue': min(max_fatigues),
            'range_fatigue': max(max_fatigues) - min(max_fatigues) + 1e-8,
        }

        # 3. 如果候选超过上限，随机截取
        if len(all_candidates) > max_candidates:
            random.shuffle(all_candidates)
            all_candidates = all_candidates[:max_candidates]

        valid_len = len(all_candidates)

        # 4. 批量提取特征
        action_features = self._extract_action_features_batch(all_candidates, hdg, group_stats)

        # 5. 生成 Mask（动态长度，全部为有效动作）
        mask = torch.ones(valid_len, dtype=torch.float32, device=self.device)

        # valid_actions 包含全部动作
        valid_actions = [item[0] for item in all_candidates]

        return valid_actions, action_features, mask

    def _extract_action_features_batch(self, actions_with_stats, hdg, group_stats):
        """候选动作特征：15维，不再复用 HDGAT 节点嵌入层。"""
        features = []
        current_cmax = max([o.complete_time for o in hdg.operation_nodes.values()] + [0])
        current_energy = sum(m.total_energy for m in self.machine_states.values())
        cur_fatigues_all = [w.current_fatigue for w in hdg.worker_nodes.values()]
        current_favg = float(np.mean(cur_fatigues_all)) if cur_fatigues_all else 0.0
        for action, finish_time, max_fatigue in actions_with_stats:
            op_key, B_k, W_set = action
            if op_key is None or op_key not in hdg.operation_nodes or B_k <= 0:
                features.append([0.0] * 15)
                continue
            op = hdg.operation_nodes[op_key]
            workers = [hdg.worker_nodes[w_id] for w_id in W_set if w_id in hdg.worker_nodes]
            if not workers:
                features.append([0.0] * 15)
                continue
            worker_fatigues = [w.current_fatigue for w in workers]
            worker_avails = [w.next_available_time for w in workers]
            ready_time = op.ready_time if op.ready_time > 0 else op.get_prev_op_completion_time()
            op_wait_time = max(0.0, current_cmax - ready_time)
            machine_ids = self.hdg.get_spt_machine_order(op_key, self.machine_states)[:B_k]
            if len(machine_ids) < B_k:
                machine_ids = op.machine_candidates[:B_k]
            remaining_pieces = max(1, op.total_pieces - op.completed_pieces)
            avg_size = remaining_pieces / max(1, B_k)
            batch_times = [op.base_pt.get(m_id, 1.0) * avg_size for m_id in machine_ids] or [1.0]
            min_batch_time, max_batch_time = float(np.min(batch_times)), float(np.max(batch_times))
            avg_batch_time = float(np.mean(batch_times))
            range_batch_time = max_batch_time - min_batch_time
            machine_ready = max([self.machine_states[m].next_available_time for m in machine_ids if m in self.machine_states] + [0.0])
            worker_ready = max(worker_avails)
            est_start = max(ready_time, machine_ready, worker_ready)
            estimated_finish = est_start + max_batch_time
            delta_cmax = max(0.0, max(current_cmax, estimated_finish) - current_cmax)
            simulated_fatigues = list(cur_fatigues_all)
            worker_ids = list(hdg.worker_nodes.keys())
            worker_index = {wid: i for i, wid in enumerate(worker_ids)}
            for w in workers:
                idx = worker_index.get(w.worker_id)
                if idx is not None:
                    inc = (1 - simulated_fatigues[idx]) * (1 - np.exp(-w.lambda_ * avg_batch_time))
                    simulated_fatigues[idx] = min(1.0, simulated_fatigues[idx] + inc)
            after_favg = float(np.mean(simulated_fatigues)) if simulated_fatigues else current_favg
            delta_favg = after_favg - current_favg
            delta_tec = 0.0
            for m_id, b_time in zip(machine_ids, batch_times):
                delta_tec += b_time * self.config.machine_data.get(m_id, {}).get('energy_process', 0.0)
            features.append([
                op_wait_time / max(1e-6, self.wait_norm_factor),
                estimated_finish / max(1e-6, self.cmax_norm_factor),
                float(np.min(worker_fatigues)),
                float(np.max(worker_fatigues)),
                float(np.mean(worker_fatigues)),
                float(np.min(worker_avails)) / max(1e-6, self.cmax_norm_factor),
                float(np.mean(worker_avails)) / max(1e-6, self.cmax_norm_factor),
                float(np.max(worker_avails)) / max(1e-6, self.cmax_norm_factor),
                min_batch_time / max(1e-6, self.batch_norm_factor),
                max_batch_time / max(1e-6, self.batch_norm_factor),
                avg_batch_time / max(1e-6, self.batch_norm_factor),
                range_batch_time / max(1e-6, self.batch_norm_factor),
                delta_cmax / max(1e-6, self.cmax_norm_factor),
                delta_favg,
                delta_tec / max(1e-6, self.energy_norm_factor),
            ])
        return torch.tensor(features, dtype=torch.float32, device=self.device)

    def _get_static_progress(self):
        """计算静态工件的完成进度"""
        total_static_ops = 0
        completed_static_ops = 0

        for op_key, op in self.hdg.operation_nodes.items():
            if op.job_id < self.config.num_jobs_static:  # 静态工件
                total_static_ops += 1
                if op.is_scheduled and op.completed_pieces >= op.total_pieces:
                    completed_static_ops += 1

        if total_static_ops == 0:
            return 0.0

        return completed_static_ops / total_static_ops

    def _check_dynamic_jobs_arrival(self):
        """
        检查动态工件是否应该到达
        基于静态工件的完成进度触发
        """
        if not self.pending_dynamic_jobs:
            return False

        static_progress = self._get_static_progress()
        arrived_jobs = []

        # 检查每个待到达的动态工件
        for job_id, trigger_ratio in self.pending_dynamic_jobs:
            if static_progress >= trigger_ratio:
                arrived_jobs.append((job_id, trigger_ratio))

        # 按触发比例排序，优先触发比例小的
        arrived_jobs.sort(key=lambda x: x[1])

        # 添加到达的工件
        for job_id, trigger_ratio in arrived_jobs:
            # print(f"  [动态到达] 工件 {job_id} 到达 (触发比例: {trigger_ratio:.2f}, 当前进度: {static_progress:.2f})")
            self._add_job_to_graph(job_id)
            self.pending_dynamic_jobs.remove((job_id, trigger_ratio))

        return len(arrived_jobs) > 0

    def _add_job_to_graph(self, job_id):
        """将单个动态工件及其所有工序加入图结构"""
        # print(f"  [动态到达]工件 {job_id} 到达")

        job_data = self.config.job_data[job_id]
        job_ops = []

        for op_id in range(self.config.num_ops_per_job):
            if op_id not in job_data:
                continue

            op_info = job_data[op_id]
            base_pt = op_info.get('base_process_time', {})
            base_pt_int = {int(k): v for k, v in base_pt.items()}

            op = Operation(
                job_id=job_id,
                op_id=op_id,
                machine_candidates=op_info.get('machine_candidates', []),
                base_pt=base_pt_int,
                total_pieces=self.config.total_pieces_per_job,
                is_preemptive=op_info.get('is_preemptive', False)
            )
            job_ops.append(op)

        # 建立工序间的时序关系
        for i in range(len(job_ops)):
            if i > 0:
                job_ops[i].prev_op = job_ops[i - 1]
                job_ops[i].predecessors.append(job_ops[i - 1])
            if i < len(job_ops) - 1:
                job_ops[i].next_op = job_ops[i + 1]
                job_ops[i].successors.append(job_ops[i + 1])

        for op in job_ops:
            self.hdg.add_operation(op)

        # add_operation 已经负责 S/E 与工序时序边，避免重复添加。

        self.hdg.update_operation_status()

    def _initialize_from_json(self):
        """
        从JSON配置初始化图结构和机器状态
        只加载静态工件，动态工件存入pending_dynamic_jobs队列（带触发比例）
        """
        self.hdg = HeterogeneousDisjunctiveGraph(self.config)

        # 1. 初始化机器状态
        self.machine_states = {}
        for machine_id in range(self.config.num_machines):
            self.machine_states[machine_id] = MachineState(machine_id, self.config)

        # 2. 创建工人
        for worker_id, worker_info in self.config.worker_data.items():
            worker = Worker(
                worker_id=worker_id,
                lambda_=float(worker_info.get('lambda', 0.05)),
                mu=float(worker_info.get('mu', 0.1))
            )
            worker.config = self.config
            self.hdg.add_worker(worker)

        # 3. 清空并重新初始化动态工件队列（存储元组 (job_id, ready_trigger)）
        self.pending_dynamic_jobs = []

        # 4. 创建工序，只加载静态工件
        operations_by_job = {}
        all_operations = []

        # 只加载静态工件（job_0 ~ job_{num_jobs_static-1}）
        for job_id in range(self.config.num_jobs_static):
            if job_id not in self.config.job_data:
                continue

            job_ops = []
            for op_id in range(self.config.num_ops_per_job):
                if op_id not in self.config.job_data[job_id]:
                    continue

                op_info = self.config.job_data[job_id][op_id]
                base_pt = op_info.get('base_process_time', {})
                base_pt_int = {int(k): v for k, v in base_pt.items()}

                op = Operation(
                    job_id=job_id,
                    op_id=op_id,
                    machine_candidates=op_info.get('machine_candidates', []),
                    base_pt=base_pt_int,
                    total_pieces=self.config.total_pieces_per_job,
                    is_preemptive=op_info.get('is_preemptive', False)
                )
                job_ops.append(op)
                all_operations.append(op)

            operations_by_job[job_id] = job_ops

        # 5. 建立同一工件内的工序时序关系
        for job_id, job_ops in operations_by_job.items():
            for i in range(len(job_ops)):
                if i > 0:
                    job_ops[i].prev_op = job_ops[i - 1]
                    job_ops[i].predecessors.append(job_ops[i - 1])
                if i < len(job_ops) - 1:
                    job_ops[i].next_op = job_ops[i + 1]
                    job_ops[i].successors.append(job_ops[i + 1])

        # 6. 将静态工序添加到图中
        for op in all_operations:
            self.hdg.add_operation(op)

        # 7. 初始化动态工件队列，存储 (job_id, ready_trigger)
        for job_id in range(self.config.num_jobs_static, self.config.num_jobs):
            if job_id in self.config.job_data:
                # 从config中获取ready_trigger
                trigger = self.config.job_ready_triggers.get(job_id, 0.5)  # 默认0.5
                self.pending_dynamic_jobs.append((job_id, trigger))

        # 8. 初始化工序状态
        self.hdg.update_operation_status()

        return len(operations_by_job) > 0

    def _extract_action_feature(self, action, hdg, group_stats, finish_time, max_fatigue):
        op_key, B_k, W_set = action
        op = hdg.operation_nodes.get(op_key)
        if not op:
            return torch.zeros(self.actor.d_action).to(self.device)

        # 获取工人实际状态
        workers = [hdg.worker_nodes[w_id] for w_id in W_set]
        fatigues = [w.current_fatigue for w in workers]
        avail_times = [w.next_available_time for w in workers]

        # 差距特征（使用传入的组内统计量）
        finish_gap = (finish_time - group_stats['min_finish']) / (group_stats['range_finish'] + 1e-8)
        fatigue_gap = (max_fatigue - group_stats['min_fatigue']) / (group_stats['range_fatigue'] + 1e-8)

        # 工序特征 (8维)
        op_feat = torch.tensor([
            finish_gap,
            fatigue_gap,
            op.completed_pieces / max(1, op.total_pieces),
            finish_time / self.cmax_norm_factor,
            (max(avail_times) - op.get_prev_op_completion_time()) / self.wait_norm_factor,
            1.0 if op.all_predecessors_completed else 0.0,
            B_k / max(1, self.config.num_machines),
            max_fatigue,
        ], dtype=torch.float32).to(self.device)
        op_embed = self.hdgat.op_embed(op_feat)

        # 工人特征 (6维，使用实际状态)
        worker_feat = torch.tensor([
            np.mean(fatigues),
            np.max(fatigues),
            np.std(fatigues) if len(fatigues) > 1 else 0.0,
            np.mean(avail_times) / self.time_norm_factor,
            np.min(avail_times) / self.time_norm_factor,
            len(workers) / self.config.num_workers,
        ], dtype=torch.float32).to(self.device)
        worker_embed = self.hdgat.worker_embed(worker_feat)

        # 子批特征 (6维，使用预估信息)
        machine_ids = self.hdg.get_spt_machine_order(op_key, self.machine_states)
        avg_machine_id = np.mean(machine_ids) / self.config.num_machines if machine_ids else 0.5
        avg_batch_size = op.total_pieces / B_k
        batch_feat = torch.tensor([
            avg_batch_size / op.total_pieces,
            avg_machine_id,
            np.mean(fatigues),
            0.0, 0.0, 0.0  # 预留
        ], dtype=torch.float32).to(self.device)
        batch_embed = self.hdgat.batch_embed(batch_feat)

        return torch.cat([op_embed, batch_embed, worker_embed])

    def _compute_reward(self, hdg, prev_state, current_state, action):
        """计算三维即时奖励：[-Δmakespan, -Δenergy, -Δfatigue_mean]"""

        makespan_inc = current_state['makespan'] - prev_state['makespan']
        energy_inc = current_state['energy'] - prev_state['energy']
        fatigue_mean_inc = current_state['fatigue_mean'] - prev_state['fatigue_mean']

        # 1. Cmax 奖励 (结合全局增量 和 动作消耗的微小局部惩罚)
        # 单纯用 makespan_inc 会稀疏，我们让它主要反映 Cmax 的变化，
        # 并用 cmax_norm_factor 归一化，使其处于合理的范围。
        # 放大一点系数（比如 10.0），避免数值过小被忽略。
        r_cmax = - (makespan_inc * 10.0) / (self.cmax_norm_factor + 1e-6)

        # 2. Energy 奖励
        # 能耗增量通常是连续的（每步都有），同理将其映射到相近的数量级
        r_energy = - (energy_inc * 10.0) / (self.energy_norm_factor + 1e-6)

        # 3. Fatigue mean 奖励
        # 疲劳标准差本身在 0~0.5 之间。其变化量通常在 0.01~0.1 级别。
        # 如果不处理，会比上述两项大很多。为了统一，我们对它进行适当缩放。
        r_fatigue = - (fatigue_mean_inc * 1.0)  # 可调整系数

        # 4. 边界截断 (Clip)，防止单步奖励异常导致梯度爆炸
        r_cmax = float(np.clip(r_cmax, -2.0, 2.0))
        r_energy = float(np.clip(r_energy, -2.0, 2.0))
        r_fatigue = float(np.clip(r_fatigue, -2.0, 2.0))

        return [r_cmax, r_energy, r_fatigue], current_state

    def _execute_action(self, action):
        """执行动作"""
        op_key, B_k, W_set = action
        self.batch_select_count[int(B_k)] += 1
        op = self.hdg.operation_nodes[op_key]

        prev_op_completion_time = op.get_prev_op_completion_time()
        machine_ids = self.hdg.get_spt_machine_order(op_key, self.machine_states)

        if len(machine_ids) < B_k:
            machine_ids = op.machine_candidates[:B_k]
        else:
            machine_ids = machine_ids[:B_k]

        created_batches = self.hdg.create_batch_nodes(op_key, B_k, W_set, machine_ids)

        if not created_batches:
            print(f"错误: 未能为工序{op_key}创建任何实际加工子批")
            return

        max_completion_time = 0

        for batch in created_batches:
            if not batch or batch.status != "pending":
                continue

            machine_id = batch.machine_id
            worker_id = batch.worker_id

            machine_state = self.machine_states.get(machine_id)
            worker = self.hdg.worker_nodes.get(worker_id)

            if not machine_state or not worker:
                continue

            switch_time = 0.0
            if machine_id in self.config.machine_data:
                switch_time = self.config.machine_data[machine_id].get('switch_time', 0.0)

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

            base_time = op.base_pt.get(machine_id, 1.0)
            fatigue_factor = 1 + worker.current_fatigue
            actual_time = base_time * batch.size * fatigue_factor

            complete_time = start_time + actual_time

            worker.assign_batch(batch)

            batch.start_time = start_time
            batch.complete_time = complete_time
            batch.status = "completed"

            machine_state.assign_batch(batch, start_time, actual_time)
            worker.complete_batch(complete_time, start_time, actual_time)
            machine_state.complete_batch()

            if complete_time > max_completion_time:
                max_completion_time = complete_time

        if max_completion_time > 0:
            completed_sizes = sum(
                batch.size for batch in created_batches
                if batch.status == "completed"
            )

            if completed_sizes > 0:
                op.update_completion(completed_sizes, max_completion_time)
                self.hdg.current_time = max_completion_time

        self.hdg.update_operation_status()

    def plot_training_curves(self, save_path=None, show_plot=True):
        """绘制训练曲线"""
        if len(self.training_history['episode']) == 0:
            print("没有训练数据，无法绘图")
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Comprehensive Training Analysis', fontsize=16)

        episodes = self.training_history['episode']

        def safe_plot(ax, key, title, color):
            data = self.training_history.get(key, [])
            if len(data) == len(episodes):
                ax.plot(episodes, data, color=color)
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
            else:
                ax.set_title(f"{title} (No Data)")
                print(f"警告: {key} 数据长度 ({len(data)}) 与 episode ({len(episodes)}) 不匹配")

        safe_plot(axes[0, 0], 'reward', 'Episode Rewards', 'tab:blue')
        safe_plot(axes[0, 1], 'makespan', 'Makespan (Cmax)', 'tab:red')
        # ---> 新增：在右上角 (0, 2) 绘制三目标动态权重图 <---
        ax_weights = axes[0, 2]
        w_cmax = self.training_history.get('weight_cmax', [])
        w_energy = self.training_history.get('weight_energy', [])
        w_fatigue = self.training_history.get('weight_fatigue', [])

        if len(w_cmax) == len(episodes):
            ax_weights.plot(episodes, w_cmax, label='Cmax Weight', color='tab:red', alpha=0.8)
            ax_weights.plot(episodes, w_energy, label='Energy Weight', color='tab:green', alpha=0.8)
            ax_weights.plot(episodes, w_fatigue, label='Fatigue Weight', color='tab:pink', alpha=0.8)
            ax_weights.set_title('Adaptive Objective Weights (Critic)')
            ax_weights.legend(loc='best')
            ax_weights.grid(True, alpha=0.3)
            ax_weights.set_ylim(0, 1.0)  # 权重是由Softmax输出的，总和为1
        else:
            ax_weights.set_title("Adaptive Weights (No Data)")
        # -----------------------------------------------------

        v_losses = self.training_history.get('critic_loss', [])
        if v_losses:
            axes[1, 0].plot(v_losses, color='tab:purple', alpha=0.5)
            axes[1, 0].set_title('Critic Value Loss (MSE)')
            axes[1, 0].set_yscale('log')

        safe_plot(axes[1, 1], 'energy', 'Total Energy Consumption', 'tab:green')
        safe_plot(axes[1, 2], 'fatigue_mean', 'Worker Average Fatigue', 'tab:pink')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save_path:
            plt.savefig(save_path)
        if show_plot:
            plt.show()

        # ====== 新增：关闭画板，防止批量训练时内存溢出 ======
        plt.close(fig)
        # =================================================

    def plot_critic_diagnostic(self, save_path='critic_diagnostic.png'):
        """绘制Critic诊断图"""
        if not self.training_history['critic_loss']:
            print("没有 Critic Loss 数据可供绘制")
            return

        losses = np.array(self.training_history['critic_loss'])
        window_size = max(1, len(losses) // 20)
        moving_avg = np.convolve(losses, np.ones(window_size) / window_size, mode='valid')

        plt.figure(figsize=(12, 6))
        plt.plot(losses, color='cyan', alpha=0.3, label='Raw Value Loss')
        plt.plot(range(window_size - 1, len(losses)), moving_avg, color='blue', linewidth=2,
                 label=f'Moving Average (window={window_size})')
        plt.title('Critic Network Diagnostic: Value Loss Trend', fontsize=14)
        plt.xlabel('Update Steps', fontsize=12)
        plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.legend()

        if len(moving_avg) > 10:
            initial_loss = np.mean(moving_avg[:5])
            final_loss = np.mean(moving_avg[-5:])
            improvement = (initial_loss - final_loss) / (initial_loss + 1e-8) * 100
            status_text = f"Initial Loss: {initial_loss:.4f}\nFinal Loss: {final_loss:.4f}\nImprovement: {improvement:.1f}%"
            plt.text(0.02, 0.95, status_text, transform=plt.gca().transAxes,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Critic 诊断图已保存到: {save_path}")

        plt.show()

    def plot_comparison_chart(self, save_path=None):
        """绘制各项指标的对比图"""
        if len(self.training_history['episode']) < 10:
            print("数据不足，无法绘制对比图")
            return

        n = len(self.training_history['episode'])
        early = slice(0, n // 3)
        middle = slice(n // 3, 2 * n // 3)
        late = slice(2 * n // 3, n)

        stages = ['Early', 'Middle', 'Late']

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Performance Comparison Across Training Stages', fontsize=14)

        ax = axes[0]
        data = [
            self.training_history['makespan'][early],
            self.training_history['makespan'][middle],
            self.training_history['makespan'][late]
        ]
        ax.boxplot(data, labels=stages)
        ax.set_ylabel('Makespan')
        ax.set_title('Makespan Distribution')
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        data = [
            self.training_history['energy'][early],
            self.training_history['energy'][middle],
            self.training_history['energy'][late]
        ]
        ax.boxplot(data, labels=stages)
        ax.set_ylabel('Energy')
        ax.set_title('Energy Distribution')
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        data = [
            self.training_history['fatigue_mean'][early],
            self.training_history['fatigue_mean'][middle],
            self.training_history['fatigue_mean'][late]
        ]
        ax.boxplot(data, labels=stages)
        ax.set_ylabel('Fatigue Mean')
        ax.set_title('Fatigue Mean Distribution')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"对比图已保存到: {save_path}")

        plt.show()

    def _save_elite_weights(self, episode_experience):
        """精英权重保存已移除。"""
        return

    def _rollback_to_elite(self, mutation_sigma=0.005):
        """精英回滚已移除。"""
        return

    def _get_current_state_metrics(self):
        current_makespan = max([o.complete_time for o in self.hdg.operation_nodes.values()] + [0])
        current_energy = sum(m.total_energy for m in self.machine_states.values())
        all_fatigues = [w.current_fatigue for w in self.hdg.worker_nodes.values()]
        current_fatigue_mean = float(np.mean(all_fatigues)) if all_fatigues else 0.0
        return {'makespan': current_makespan, 'energy': current_energy, 'fatigue_mean': current_fatigue_mean}

    def run_one_episode(self, episode=1, deterministic=False, collect_experience=True):
        """运行一个完整 episode。采样阶段保存 state_raw，更新阶段重新编码以训练 HDGAT。"""
        episode_experience = []
        current_episode_entropies = []
        current_episode_weights = []
        total_episode_reward = 0.0
        prev_state = {'makespan': 0.0, 'energy': 0.0, 'fatigue_mean': 0.0}
        done = False
        step_count = 0
        current_progress = min(1.0, episode / max(1, self.total_episodes))

        with torch.no_grad():
            while not done:
                step_count += 1
                if self.pending_dynamic_jobs:
                    self._check_dynamic_jobs_arrival()

                state_raw = self.hdgat.extract_raw_features(self.hdg, self.device)
                global_feature = self.hdgat.encode_raw_features(state_raw, self.device)
                valid_actions, action_features, mask = self._generate_valid_actions(self.hdg, episode=episode)

                if not valid_actions:
                    all_ops_finished = all(o.is_scheduled for o in self.hdg.operation_nodes.values())
                    if (not self.pending_dynamic_jobs and all_ops_finished) or step_count >= self.max_steps_per_episode:
                        break
                    continue

                selected_idx, action_prob, entropy = self.actor.get_action(
                    global_feature.unsqueeze(0),
                    action_features.unsqueeze(0),
                    progress=current_progress,
                    deterministic=deterministic,
                    mask=mask.unsqueeze(0)
                )
                if selected_idx is None:
                    break
                if entropy is not None:
                    current_episode_entropies.append(entropy)

                selected_action = valid_actions[selected_idx]
                self._execute_action(selected_action)

                current_state = self._get_current_state_metrics()
                step_reward_vec, _ = self._compute_reward(self.hdg, prev_state, current_state, selected_action)

                values, weights = self.critic(global_feature.unsqueeze(0))
                value_vec = values.squeeze(0).cpu()
                weight_vec = weights.squeeze(0).cpu()
                current_episode_weights.append(weight_vec.tolist())
                scalar_reward = sum(w * r for w, r in zip(weight_vec.tolist(), step_reward_vec))
                total_episode_reward += scalar_reward

                if collect_experience:
                    episode_experience.append({
                        'state_raw': state_raw,
                        'action_features': action_features,
                        'mask': mask,
                        'selected_idx': selected_idx,
                        'prob': action_prob,
                        'reward': step_reward_vec,
                        'value': value_vec,
                        'done': False
                    })

                prev_state = current_state
                all_ops_finished = all(o.is_scheduled for o in self.hdg.operation_nodes.values())
                done = all_ops_finished and (not self.pending_dynamic_jobs)
                if step_count >= self.max_steps_per_episode:
                    break

        if episode_experience:
            episode_experience[-1]['done'] = True

        metrics = self._get_current_state_metrics()
        metrics['reward'] = total_episode_reward
        metrics['entropy'] = float(np.mean(current_episode_entropies)) if current_episode_entropies else 0.0
        metrics['weights'] = np.mean(current_episode_weights, axis=0).tolist() if current_episode_weights else [0.0, 0.0, 0.0]
        return episode_experience, metrics

    def update_policy(self, episode=1):
        """PPO 更新：用 state_raw 重新编码，联合更新 HDGAT、Actor、Critic。"""
        update_info = {'actor_loss': None, 'critic_loss': None, 'entropy': None}
        current_progress = min(1.0, episode / max(1, self.total_episodes))
        curr_entropy_coef = self.config.entropy_coef * max(0.1, (1.0 - current_progress))

        for _ in range(self.config.update_epochs):
            batch = self.buffer.sample(self.device)
            if not batch:
                break

            global_features = torch.stack([
                self.hdgat.encode_raw_features(raw, self.device)
                for raw in batch['state_raw']
            ], dim=0)

            current_v, weights = self.critic(global_features)
            value_loss = F.mse_loss(current_v, batch['return'])

            _, all_probs = self.actor(
                global_features,
                batch['action_features'],
                progress=current_progress,
                mask=batch['mask']
            )
            batch_size = global_features.size(0)
            idx = torch.arange(batch_size, device=self.device)
            new_probs = all_probs[idx, batch['selected_idx']]
            ratios = new_probs / (batch['prob'] + 1e-8)

            log_probs = torch.log(all_probs + 1e-9)
            entropy = -torch.sum(all_probs * log_probs * batch['mask'], dim=-1)

            scalar_adv = (batch['advantage'] * weights.detach()).sum(dim=-1)
            scalar_adv = (scalar_adv - scalar_adv.mean()) / (scalar_adv.std() + 1e-8)

            surr1 = ratios * scalar_adv
            surr2 = torch.clamp(ratios, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * scalar_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            total_loss = policy_loss + self.config.value_coef * value_loss - curr_entropy_coef * entropy.mean()

            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.hdgat.parameters()) + list(self.actor.parameters()) + list(self.critic.parameters()),
                0.5
            )
            self.optimizer.step()

            update_info = {
                'actor_loss': float(policy_loss.item()),
                'critic_loss': float(value_loss.item()),
                'entropy': float(entropy.mean().item())
            }
            self.training_history['actor_loss'].append(update_info['actor_loss'])
            self.training_history['critic_loss'].append(update_info['critic_loss'])

            # 释放本轮 PPO 更新中的临时计算图和中间张量
            del global_features, current_v, weights, value_loss
            del all_probs, new_probs, ratios, log_probs, entropy, scalar_adv
            del surr1, surr2, policy_loss, total_loss
            self._cleanup_memory(aggressive=False)

        self._cleanup_memory(aggressive=True)
        return update_info

    def reset_instance(self, json_path):
        """多实例随机训练时切换实例，只重置环境，不重建网络。"""
        new_config = Config(json_path=json_path)
        self.config = new_config
        self.hdgat.config = new_config
        self.actor.config = new_config
        self.critic.config = new_config
        self.current_instance_name = os.path.basename(json_path)
        self._compute_dynamic_parameters()
        self._initialize_from_json()

    def _compute_episode_returns(self, episode_experience):
        rewards = torch.tensor([e['reward'] for e in episode_experience], dtype=torch.float32, device=self.device)
        values = torch.stack([e['value'] for e in episode_experience]).to(self.device)
        dones = torch.tensor([float(e['done']) for e in episode_experience], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_values = torch.cat([values[1:], torch.zeros(1, 3, device=self.device)], dim=0)
        return self.return_calc.compute_returns(rewards, values, next_values, dones)

    def train_multi_instance(self, train_files, val_files=None, total_episodes=50000,
                             update_interval=8, eval_interval=1000, save_interval=1000,
                             save_dir='saved_models', debug_interval=10):
        """多实例随机采样训练，支持从已有模型继续累计训练轮数。"""
        os.makedirs(save_dir, exist_ok=True)

        # 继续训练时，不再依赖 training_history 的长度判断已训练轮数。
        # 优先使用模型文件中的 trained_episodes；兼容旧模型时，可从历史 episode 最大值兜底恢复。
        old_episode_count = int(getattr(self, 'trained_episodes', 0))
        if old_episode_count <= 0 and self.training_history.get('episode'):
            old_episode_count = int(max(self.training_history['episode']))

        final_episode = old_episode_count + total_episodes
        self.total_episodes = final_episode

        print(f"\n🚀 多实例随机训练开始 | 训练实例数={len(train_files)} | "
              f"已训练={old_episode_count} | 本次新增={total_episodes} | 目标总episode={final_episode}")

        for episode in range(old_episode_count + 1, final_episode + 1):
            episode_start = time.time()
            json_path = random.choice(train_files)
            self.reset_instance(json_path)

            episode_experience, metrics = self.run_one_episode(episode=episode, deterministic=False, collect_experience=True)
            if episode_experience:
                returns, advantages = self._compute_episode_returns(episode_experience)
                self.buffer.add(episode_experience, returns, advantages)

            if self.buffer.size() >= self.config.batch_size and episode % update_interval == 0:
                self.update_policy(episode=episode)
                self.buffer.clear()
                self._cleanup_memory(aggressive=True)

            self.training_history['episode'].append(episode)
            self.training_history['reward'].append(metrics.get('reward', 0.0))
            self.training_history['makespan'].append(metrics['makespan'])
            self.training_history['energy'].append(metrics['energy'])
            self.training_history['fatigue_mean'].append(metrics['fatigue_mean'])
            self.training_history['policy_entropy'].append(metrics.get('entropy', 0.0))
            weights = metrics.get('weights', [0.0, 0.0, 0.0])
            self.training_history['weight_cmax'].append(weights[0])
            self.training_history['weight_energy'].append(weights[1])
            self.training_history['weight_fatigue'].append(weights[2])

            # 更新累计训练轮数；保存模型时只保存这个数字，不再保存完整 training_history。
            self.trained_episodes = episode

            self._log_episode_to_csv(
                episode=episode,
                duration=time.time() - episode_start,
                cmax=metrics['makespan'],
                energy=metrics['energy'],
                fatigue_mean=metrics['fatigue_mean'],
                reward=metrics.get('reward', 0.0),
                weights=weights
            )

            if episode % debug_interval == 0:
                print(f"Ep {episode:5d} | {self.current_instance_name} | Cmax={metrics['makespan']:.1f} | "
                      f"TEC={metrics['energy']:.1f} | Favg={metrics['fatigue_mean']:.3f} | "
                      f"R={metrics.get('reward', 0.0):.3f}")

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
                    print(f"[验证] Ep {episode} 未更新 best_val | 当前best_score={self.best_val_score:.6f}")

            # 每个 episode 末尾清理临时对象，避免长时间训练时显存/内存逐步上涨
            try:
                del episode_experience
            except UnboundLocalError:
                pass
            try:
                del returns, advantages
            except UnboundLocalError:
                pass
            self._cleanup_memory(aggressive=False)

        self.save_models(save_dir, 'final')
        self._cleanup_memory(aggressive=True)

    def train(self, total_episodes=500, debug_interval=10):
        """兼容旧调用：仅在当前实例上训练，不进行预演校准。"""
        if not getattr(self.config, 'json_path', None):
            raise ValueError('当前 Config 不包含 json_path，无法使用兼容 train()。请调用 train_multi_instance。')
        return self.train_multi_instance(
            train_files=[self.config.json_path],
            val_files=None,
            total_episodes=total_episodes,
            update_interval=8,
            save_interval=max(100, total_episodes),
            debug_interval=debug_interval
        )

    def evaluate_validation(self, val_files, eval_runs=1):
        """
        验证集评估函数。

        重要说明：
        reference_solution 只作为归一化基准，不再默认代表最优解。
        因此这里按“每个实例先求相对参考比例，再对实例求平均”的方式统计，
        避免某些大规模实例直接主导整体均值。
        """
        cmax_list, tec_list, favg_list = [], [], []
        ref_cmax_list, ref_tec_list, ref_favg_list = [], [], []
        cmax_ratio_list, tec_ratio_list, favg_ratio_list = [], [], []

        for json_path in val_files:
            run_cmax, run_tec, run_favg = [], [], []

            for _ in range(eval_runs):
                self.reset_instance(json_path)
                _, metrics = self.run_one_episode(
                    episode=0,
                    deterministic=True,
                    collect_experience=False
                )
                run_cmax.append(metrics['makespan'])
                run_tec.append(metrics['energy'])
                run_favg.append(metrics['fatigue_mean'])

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

            # 保留参考解均值，方便日志观察。它只作为参考，不代表最优。
            'mean_ref_cmax': float(np.mean(ref_cmax_list)) if ref_cmax_list else None,
            'mean_ref_tec': float(np.mean(ref_tec_list)) if ref_tec_list else None,
            'mean_ref_favg': float(np.mean(ref_favg_list)) if ref_favg_list else None,

            # 新增：逐实例相对参考比例，score 主要基于这些比例计算。
            'mean_cmax_ratio': float(np.mean(cmax_ratio_list)) if cmax_ratio_list else None,
            'mean_tec_ratio': float(np.mean(tec_ratio_list)) if tec_ratio_list else None,
            'mean_favg_ratio': float(np.mean(favg_ratio_list)) if favg_ratio_list else None,

            # 新增：稳定性统计，越小说明不同验证实例上的表现越稳定。
            'std_cmax_ratio': float(np.std(cmax_ratio_list)) if cmax_ratio_list else 0.0,
            'std_tec_ratio': float(np.std(tec_ratio_list)) if tec_ratio_list else 0.0,
            'std_favg_ratio': float(np.std(favg_ratio_list)) if favg_ratio_list else 0.0,
        }

    def compute_val_score(self, val_metrics):
        """
        验证集综合评分，分数越小越好。

        修改点：
        1. reference_solution 仅作为归一化基准，不视为最优解；
        2. 优先使用逐实例 ratio 的平均值，避免大规模实例主导；
        3. 提高 Cmax 权重，防止为了降低疲劳而明显牺牲完工时间；
        4. 加入稳定性惩罚和明显恶化惩罚。
        """
        cmax_ratio = val_metrics.get('mean_cmax_ratio')
        tec_ratio = val_metrics.get('mean_tec_ratio')
        favg_ratio = val_metrics.get('mean_favg_ratio')

        # 如果验证实例没有 reference_solution，则退回到当前实例动态归一化参数。
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

        # 如果某个目标明显差于参考基准，额外加罚。
        # 这里不是认为参考解最优，而是防止模型出现严重偏科。
        cmax_bad_penalty = max(0.0, cmax_ratio - 1.10)
        tec_bad_penalty = max(0.0, tec_ratio - 1.10)
        favg_bad_penalty = max(0.0, favg_ratio - 1.10)
        bad_penalty = (
            0.60 * cmax_bad_penalty
            + 0.25 * tec_bad_penalty
            + 0.15 * favg_bad_penalty
        )

        score = (
            0.50 * cmax_ratio
            + 0.30 * tec_ratio
            + 0.20 * favg_ratio
            + 0.05 * stability_penalty
            + bad_penalty
        )
        return float(score)

    def is_better_validation(self, val_score, val_metrics, min_improve=0.003, tolerance=0.00):
        """
        判断当前验证结果是否值得保存为 best_val。

        min_improve=0.003 表示综合分数至少提升 0.3%；
        tolerance=0.03 表示即使综合分数变好，单目标也不能比当前 best_val 恶化超过 3%。
        这样可以避免保存“Cmax 严重变差、仅疲劳略好”的偏科模型。
        """
        if self.best_val_metrics is None or not np.isfinite(self.best_val_score):
            return True

        # 综合分数需要有足够幅度的提升，防止验证噪声导致频繁覆盖 best_val。
        score_improved = val_score < self.best_val_score * (1.0 - min_improve)
        if not score_improved:
            return False

        old = self.best_val_metrics
        cmax_ok = val_metrics['mean_cmax'] <= old['mean_cmax'] * (1.0 + tolerance)
        tec_ok = val_metrics['mean_tec'] <= old['mean_tec'] * (1.0 + tolerance)
        favg_ok = val_metrics['mean_favg'] <= old['mean_favg'] * (1.0 + tolerance)

        return cmax_ok and tec_ok and favg_ok

    def _warmup_and_calibrate(self, num_episodes=5):
        """
        预演校准：使用未经训练的初始策略运行几个回合，
        收集真实的 Makespan 和 Energy 作为归一化基准。
        """
        print(f"\n⚙️ 正在进行训练前预演校准 (运行 {num_episodes} 代)...")
        warmup_makespans = []
        warmup_energies = []

        # 禁用梯度，纯粹为了收集数据
        with torch.no_grad():
            for ep in range(num_episodes):
                if not self._initialize_from_json():
                    continue

                done = False
                step_count = 0
                while not done:
                    step_count += 1
                    # 处理动态工件
                    if self.pending_dynamic_jobs:
                        self._check_dynamic_jobs_arrival()

                    # 获取状态特征
                    _, global_feature = self.hdgat(self.hdg, self.device)

                    # ===== 修改点：解包 3 个返回值 =====
                    valid_actions, action_features, mask = self._generate_valid_actions(self.hdg, episode=ep)

                    if not valid_actions:
                        if not self.pending_dynamic_jobs and all(
                                o.is_scheduled for o in self.hdg.operation_nodes.values()):
                            break
                        continue

                    # 使用初始未训练的网络进行动作采样 (随机探索)
                    # ===== 修改点：增加 Batch 维度 (unsqueeze) =====
                    selected_idx, _, _ = self.actor.get_action(
                        global_feature.unsqueeze(0),
                        action_features.unsqueeze(0),  # (K, D) -> (1, K, D)
                        progress=0.0,
                        mask=mask.unsqueeze(0)  # (K,) -> (1, K)
                    )

                    if selected_idx is None:
                        break

                    # 执行动作
                    self._execute_action(valid_actions[selected_idx])

                    # 终止条件
                    all_ops_finished = all(o.is_scheduled for o in self.hdg.operation_nodes.values())
                    done = all_ops_finished and (not self.pending_dynamic_jobs)
                    if step_count >= self.max_steps_per_episode:
                        break

                # 记录这代结束时的最终结果
                final_makespan = max([o.complete_time for o in self.hdg.operation_nodes.values()] + [0])
                final_energy = sum(m.total_energy for m in self.machine_states.values())
                warmup_makespans.append(final_makespan)
                warmup_energies.append(final_energy)
                print(f" 预演 {ep + 1}/{num_episodes} -> Makespan: {final_makespan:.1f}, Energy: {final_energy:.1f}")

        # 计算校准后的归一化参数 (取平均值，并加上 10% 的余量防止越界)
        if warmup_makespans and warmup_energies:
            avg_makespan = np.mean(warmup_makespans)
            avg_energy = np.mean(warmup_energies)
            self.cmax_norm_factor = max(100.0, avg_makespan * 1.3)
            self.energy_norm_factor = max(1000.0, avg_energy * 1.1)

            # 其他时间相关的归一化参数也基于真实的 cmax 衍生
            self.time_norm_factor = max(10.0, self.cmax_norm_factor * 0.5)
            self.wait_norm_factor = max(10.0, self.cmax_norm_factor * 0.2)

            print(f"✅ 校准完成！")
            print(f" --> Cmax 归一化基准设定为: {self.cmax_norm_factor:.1f}")
            print(f" --> Energy 归一化基准设定为: {self.energy_norm_factor:.1f}")
            print("-" * 50)
        else:
            print("⚠️ 预演失败，回退到默认粗略估计值。")

    def _is_better(self, current, best):
        """
        多目标加权比较，使用 EMA 平滑值（如果已初始化）减少噪声影响。
        权重固定为 [0.4, 0.3, 0.3]（可根据需要调整）。

        参数:
            current: dict, 包含当前 episode 的原始指标 ('makespan', 'energy', 'fatigue_mean')
            best: dict, 包含历史最优原始指标

        返回:
            bool: 如果当前（平滑后）综合得分优于历史最优，则返回 True
        """
        w = [0.4, 0.3, 0.3]  # Cmax, 能耗, 平均疲劳度的权重

        # 如果 EMA 已初始化，使用 EMA 值代替当前原始值
        if self.ema_makespan is not None:
            makespan_val = self.ema_makespan
            energy_val = self.ema_energy
            fatigue_val = self.ema_fatigue_mean
        else:
            makespan_val = current['makespan']
            energy_val = current['energy']
            fatigue_val = current['fatigue_mean']

        current_score = (w[0] * makespan_val / self.cmax_norm_factor +
                         w[1] * energy_val / self.energy_norm_factor +
                         w[2] * fatigue_val)

        best_score = (w[0] * best['makespan'] / self.cmax_norm_factor +
                      w[1] * best['energy'] / self.energy_norm_factor +
                      w[2] * best['fatigue_mean'])

        return current_score < best_score

    def _init_csv_logger(self):
        """初始化CSV日志记录器"""
        # 检查文件是否存在，不存在则创建并写入表头
        if not os.path.exists(self.csv_log_path):
            with open(self.csv_log_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'instance_name',  # 实例名字
                    'episode',  # 代数
                    'duration_seconds',  # 训练时长（秒）
                    'cmax',  # 最大完工时间
                    'total_energy',  # 总能耗
                    'avg_fatigue',  # 工人平均疲劳度
                    'total_reward',  # 奖励值
                    'weight_cmax',  # Cmax权重
                    'weight_energy',  # Energy权重
                    'weight_fatigue'  # Fatigue权重
                ])
            print(f"✅ CSV日志文件已创建: {self.csv_log_path}")

    def set_instance_name(self, name):
        """设置当前训练实例名称"""
        self.current_instance_name = name

    def _log_episode_to_csv(self, episode, duration, cmax, energy,
                            fatigue_mean, reward, weights):
        """
        将一代的训练数据记录到CSV文件

        参数:
            episode: 代数
            duration: 训练时长（秒）
            cmax: 最大完工时间
            energy: 总能耗
            fatigue_mean: 工人平均疲劳度
            reward: 奖励值
            weights: 三目标权重列表 [w_cmax, w_energy, w_fatigue]
        """
        try:
            # 确保权重是列表格式
            if isinstance(weights, torch.Tensor):
                weights = weights.cpu().detach().tolist()
            elif weights is None or len(weights) < 3:
                weights = [0.0, 0.0, 0.0]

            with open(self.csv_log_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.current_instance_name,  # 实例名字
                    episode,  # 代数
                    f"{duration:.2f}",  # 训练时长
                    f"{cmax:.2f}",  # Cmax
                    f"{energy:.2f}",  # 总能耗
                    f"{fatigue_mean:.4f}",  # 工人平均疲劳度
                    f"{reward:.4f}",  # 奖励值
                    f"{weights[0]:.4f}",  # Cmax权重
                    f"{weights[1]:.4f}",  # Energy权重
                    f"{weights[2]:.4f}"  # Fatigue权重
                ])
        except Exception as e:
            print(f"⚠️ CSV写入失败: {e}")

    def _perform_elite_reinforcement(self, progress):
        """精英学习已移除。"""
        return

    def save_models(self, path="saved_models/", episode_num=None):
        """保存训练好的模型。

        说明：
        1. 模型继续训练只依赖网络参数、优化器状态和累计训练轮数。
        2. 不再把完整 training_history 写入 .pth，避免 9000+ episode 后模型文件过大。
        3. 完整训练过程仍由 training_log.csv 保存，便于后续画图和论文分析。
        """
        os.makedirs(path, exist_ok=True)

        if episode_num is None:
            episode_num = "latest"

        history_episode_max = max(self.training_history.get('episode', [0]) or [0])
        trained_episodes = max(int(getattr(self, 'trained_episodes', 0)), int(history_episode_max))

        save_path = os.path.join(path, f"ppo_agent_episode_{episode_num}.pth")

        torch.save({
            'hdgat_state_dict': self.hdgat.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_makespan': self.best_makespan,
            'best_energy': self.best_energy,
            'best_fatigue_mean': self.best_fatigue_mean,

            # 只保存累计训练轮数，不保存完整训练历史。
            'trained_episodes': trained_episodes,

            # 保存验证集最优状态，便于断点继续时沿用更稳的 best_val 判断规则。
            'best_val_score': self.best_val_score,
            'best_val_metrics': self.best_val_metrics,
        }, save_path)

        print(f"模型已保存到 {save_path}")
        print(f"  累计训练轮数: {trained_episodes}")
        print("  已跳过保存完整 training_history，完整日志请查看 training_log.csv")

    def load_models(self, path):
        """加载训练好的模型。

        兼容旧版 checkpoint：
        - 新版模型直接读取 trained_episodes；
        - 旧版模型如果只有 training_history，则仅用它恢复累计轮数，不再把完整历史加载到当前对象中。
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.hdgat.load_state_dict(checkpoint['hdgat_state_dict'])
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'best_makespan' in checkpoint:
            self.best_makespan = checkpoint['best_makespan']
            self.best_energy = checkpoint['best_energy']
            self.best_fatigue_mean = checkpoint['best_fatigue_mean']

        if 'best_val_score' in checkpoint:
            self.best_val_score = checkpoint['best_val_score']
        if 'best_val_metrics' in checkpoint:
            self.best_val_metrics = checkpoint['best_val_metrics']

        if 'trained_episodes' in checkpoint:
            self.trained_episodes = int(checkpoint['trained_episodes'])
            print(f"  已恢复累计训练轮数: {self.trained_episodes}")
        elif 'training_history' in checkpoint:
            old_history = checkpoint.get('training_history', {})
            old_episodes = old_history.get('episode', [])
            self.trained_episodes = int(max(old_episodes)) if old_episodes else 0
            print(f"  检测到旧版模型包含训练历史，共 {len(old_episodes)} 个episode")
            print(f"  已仅恢复累计训练轮数: {self.trained_episodes}，不再载入完整 training_history")
        else:
            self.trained_episodes = 0
            print("  未检测到累计训练轮数，将从 0 开始计数")

        # 主动释放旧版 checkpoint 中可能包含的大体积训练历史。
        if 'training_history' in checkpoint:
            del checkpoint['training_history']

        print(f"模型已从 {path} 加载")


# 运行训练
if __name__ == "__main__":
    train_dir = "../TestData/train"
    val_dir = "../TestData/yanzheng"
    save_dir = "saved_models_wo_ps"

    train_files = sorted(glob.glob(os.path.join(train_dir, "*.json")))[:100]
    if not train_files:
        raise FileNotFoundError(f"未找到训练实例: {train_dir}")
    val_files = sorted(glob.glob(os.path.join(val_dir, "*.json"))) if val_dir and os.path.exists(val_dir) else None

    print(f"训练实例数: {len(train_files)}")
    print(f"验证实例数: {len(val_files) if val_files else 0}")

    init_config = Config(json_path=train_files[0])
    agent = PPOAgent(init_config)

    latest_path = os.path.join(save_dir, "ppo_agent_episode_best_val.pth")
    if os.path.exists(latest_path):
        print(f"[载入已有模型] {latest_path}")
        agent.load_models(latest_path)

    agent.train_multi_instance(
        train_files=train_files,
        val_files=val_files,
        total_episodes=5000,
        update_interval=8,
        eval_interval=500,
        save_interval=500,
        save_dir=save_dir,
        debug_interval=10
    )
