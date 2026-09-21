# IMPRL 对比算法 Agent：论文式"双注意力GAT + PPO + 操作级MILP"
#
# 与自研 PPOAgent 的关键差异：
#   1) 编码器：论文双注意力 GAT（imprl_gat.IMPRL_GAT），而非 HDGAT
#   2) 动作空间：(工序, 分批系数 ξ)，ξ 离散 11 级；而非 (工序, B_k, 工人集)
#   3) 分批尺寸 + 机器选择：由操作级 MILP 求解（imprl_milp），而非启发式加权
#   4) 工人分配：MILP 定机台后，贪心分配"疲劳最低的空闲工人"
#   5) 三目标加权（与自研公平对齐）：MILP 目标 = w1·C + w2·E + w3·F，
#      权重取自适应 critic 的当前动态权重
#
# 复用 PPOAgent：run_one_episode / update_policy / train_multi_instance /
# evaluate_validation / save_models / load_models 等全部原样继承。

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, deque

from V1_01 import (Config, Operation, Worker, MachineState, SubBatch,
                   BatchCountNode, HeterogeneousDisjunctiveGraph,
                   ActorNetwork, FeatureFusionEncoder, CriticNetwork,
                   AdaptiveExploration, ReturnCalculator, ReplayBuffer, PPOAgent)

from imprl_milp import solve_operation_milp, _fallback_split
from imprl_gat import IMPRL_GAT


class FixedWeightCritic(CriticNetwork):
    """固定权重多目标 Critic（线性标量化，非自适应）。

    论文 IMPRL 是单目标，无多目标机制。为在用户的多目标问题上做公平对比，
    这里采用"多价值头 + 固定权重标量化"——三目标各自一个价值头 (B,3)，
    但**权重为固定常数**（不随状态变化），区别于自研算法的 AdaptiveWeightCritic
    （状态驱动 softmax 动态权重）。审稿上可透明地把它视为"IMPRL 的固定权重多目标扩展"。
    """

    def __init__(self, config, weights=(1.0 / 3, 1.0 / 3, 1.0 / 3)):
        super().__init__(config)
        del self.value_head
        self.value_heads = nn.ModuleList([
            nn.Linear(config.value_hidden_dims[-1], 1) for _ in range(3)
        ])
        self.fixed_weights = tuple(weights)
        # 手动初始化价值头（继承的 _initialize_weights 只覆盖原有 value_head）
        for head in self.value_heads:
            nn.init.xavier_uniform_(head.weight)
            if head.bias is not None:
                nn.init.zeros_(head.bias)

    def forward(self, global_features):
        if global_features.dim() == 1:
            global_features = global_features.unsqueeze(0)
        features = self.feature_extractor(global_features)          # (B, H)
        values = torch.cat([head(features) for head in self.value_heads], dim=-1)  # (B, 3)
        B = features.shape[0]
        w = torch.tensor(self.fixed_weights, dtype=torch.float32, device=values.device)
        w = w.view(1, -1).expand(B, -1)                             # (B, 3) 恒定权重
        return values, w


class IMPRLAgent(PPOAgent):
    """论文式 IMPRL 智能体。继承自研 PPOAgent 复用训练/评估框架。"""

    # 分批系数离散等级（论文最优 l=11）
    XI_LEVELS = 11

    def __init__(self, config: Config):
        # 不复用 PPOAgent.__init__（其会构建 HDGAT 并触发自研维度测试）。
        # 复制共享初始化，仅替换决策层组件。
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[IMPRL] 使用设备: {self.device}")

        self.hdg = HeterogeneousDisjunctiveGraph(config)
        self.machine_states = {}

        # 动态工件相关
        self.pending_dynamic_jobs = []
        self.total_episodes = 500
        self.trained_episodes = 0

        # ===== 决策层（与自研不同）=====
        self.hdgat = IMPRL_GAT(config).to(self.device)   # 论文式 GAT，仍命名为 hdgat 以复用框架
        self.hdgat.set_agent(self)
        self.actor = ActorNetwork(config).to(self.device)
        # 固定权重多目标 critic（三目标固定等权），不使用自研的动态权重机制
        self.obj_weights = (1.0 / 3, 1.0 / 3, 1.0 / 3)  # (w_cmax, w_energy, w_fatigue)，可改
        self.critic = FixedWeightCritic(config, weights=self.obj_weights).to(self.device)

        # 动作维度覆盖：IMPRL 动作特征维度（见 _extract_action_features_batch）
        # d_hidden(32) + d_hidden(32) + D_global(69) + pair(6) + sublot(6) = 145
        self.actor.d_action = 32 + 32 + config.d_global + 6 + 6
        self._rebuild_actor_input()

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
        self.ema_makespan = None
        self.ema_energy = None
        self.ema_fatigue_mean = None
        self.ema_beta = getattr(config, 'makespan_ema_beta', 0.995)

        self.csv_log_path = "training_log_imprl.csv"
        self.current_instance_name = ""
        self._init_csv_logger()

        self.training_history = {
            'episode': [], 'reward': [], 'makespan': [], 'energy': [],
            'fatigue_mean': [], 'completion_rate': [], 'actor_loss': [],
            'critic_loss': [], 'policy_entropy': [],
            'weight_cmax': [], 'weight_energy': [], 'weight_fatigue': [],
        }
        self.best_val_score = float('inf')
        self.best_val_metrics = None
        self.smooth_window = 10
        self.best_metrics = {'makespan': float('inf'), 'energy': float('inf'),
                             'fatigue_mean': float('inf')}

        # 动态参数
        self._compute_dynamic_parameters()

        # 固定目标权重（供 MILP 目标与奖励标量化使用，不再随 critic 状态动态变化）
        self.last_weights = self.obj_weights

        self._validate_dimensions()

    def _rebuild_actor_input(self):
        """Actor 的 fusion_encoder 输入维度由 d_global+d_action 决定。
        覆盖 d_action 后需重建第一层 Linear 权重并重新初始化。"""
        cfg = self.config
        d_input = self.actor.d_global + self.actor.d_action
        self.actor.d_input = d_input
        self.actor.fusion_encoder = FeatureFusionEncoder(d_input, cfg.fusion_dims).to(self.device)
        self.actor._initialize_weights()

    # ---------------- 覆盖：动作生成 ----------------
    def _generate_valid_actions(self, hdg, episode=None):
        """候选工序门控与自研一致（就绪 + 机器空闲 + 工人足够），
        再按论文"剩余最大负载(FS)"预选部分工序，对每工序 × XI_LEVELS 个 ξ 生成动作。
        返回 (valid_actions, action_features(K,D), mask(K,))。
        """
        hdg.update_operation_status()
        ready_ops = hdg.get_ready_operations()

        feasible_ops = []
        for op_key in ready_ops:
            op = hdg.operation_nodes[op_key]
            if op.completed_pieces >= op.total_pieces:
                continue
            if not hdg.check_machine_availability(op_key, self.machine_states):
                continue
            feasible_ops.append(op_key)

        if not feasible_ops:
            return [], torch.zeros((0, self.actor.d_action), device=self.device), torch.zeros(0, device=self.device)

        # 确保 _step_raw/_step_embeds 就绪（run_one_episode 每步已算；eval 独立调用时兜底）
        # 目标权重固定（self.obj_weights），无需在此经 critic 计算
        raw_now = getattr(self, '_step_raw', None)
        if raw_now is None or getattr(self, '_step_embeds_key', None) != raw_now.get('_ver'):
            raw_now = self.hdgat.extract_raw_features(hdg, self.device)
            self._step_raw = raw_now
            self._step_embeds = self._encode_step(raw_now)
            self._step_embeds_key = raw_now.get('_ver')

        # 论文 FS：按剩余最大负载预选一部分工序（这里用全部可行工序 × ξ 也不会超 top_k）
        # 剩余负载 = 该工件尚未完成的工序数（用工序总数近似）
        max_ops = len(feasible_ops)
        selected_ops = feasible_ops[:max_ops]  # 规模小，全选；FS 主要用于动作空间压缩

        # 生成 (工序, ξ) 动作
        xi_levels = [v / (self.XI_LEVELS - 1) for v in range(self.XI_LEVELS)]
        all_candidates = []
        for op_key in selected_ops:
            for xi in xi_levels:
                all_candidates.append((op_key, xi))

        # 特征筛选：按剩余负载排序取 Top-K
        K = self.top_k_actions
        if len(all_candidates) > K:
            # 剩余负载高的工序优先（预计算每个 job 的未完成工序数，避免 O(N²)）
            job_unscheduled = defaultdict(int)
            for o in hdg.operation_nodes.values():
                if not o.is_scheduled:
                    job_unscheduled[o.job_id] += 1

            def load(op_key):
                return job_unscheduled[hdg.operation_nodes[op_key].job_id]

            all_candidates.sort(key=lambda a: -load(a[0]))
            all_candidates = all_candidates[:K]

        valid_len = len(all_candidates)
        valid_actions = all_candidates

        # Padding 到 K
        if valid_len < K:
            dummy = (None, 0.0)
            all_candidates.extend([dummy] * (K - valid_len))

        action_features = self._extract_action_features_batch(all_candidates, hdg)
        mask = torch.zeros(K, dtype=torch.float32, device=self.device)
        mask[:valid_len] = 1.0

        return valid_actions, action_features, mask

    def _encode_step(self, raw):
        """对本步 raw 做一次完整 GAT 前向，返回 (op_embeds, mach_embeds, g_global)。"""
        with torch.no_grad():
            op_embeds = self.hdgat.op_embed(raw['op_feats'].to(self.device))
            mach_embeds = self.hdgat.mach_embed(raw['mach_feats'].to(self.device))
            if op_embeds.size(0) > 1:
                self.hdgat.op_attn._op_offsets = raw.get('job_seq_map', {})
                self.hdgat.op_attn._adj_ids = tuple(raw.get('op_job_ids', []))
                op_embeds = self.hdgat.op_attn(op_embeds)
            if mach_embeds.size(0) > 1 and op_embeds.size(0) > 0:
                mmask = self.hdgat._build_mach_op_mask(raw, self.device)
                if mmask is not None:
                    mach_embeds = self.hdgat.mach_attn(mach_embeds, op_embeds, mmask)
            g_global = self.hdgat.encode_raw_features(raw, self.device)  # (D_global,)
        return op_embeds, mach_embeds, g_global

    def run_one_episode(self, episode=1, deterministic=False, collect_experience=True):
        """运行一个完整 episode（IMPRL 优化版）：
        每步只做一次 GAT extract + encode，嵌入缓存给动作特征/权重复用，避免重复前向。
        其余逻辑（奖励、经验、done 判断）与自研 PPOAgent 一致。
        """
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

                # 单次 extract + encode，本步所有方法复用
                state_raw = self.hdgat.extract_raw_features(self.hdg, self.device)
                self._step_raw = state_raw
                self._step_embeds = self._encode_step(state_raw)
                self._step_embeds_key = state_raw.get('_ver')
                global_feature = self._step_embeds[2]

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

    def _extract_action_features_batch(self, actions, hdg):
        """(工序, ξ) 动作特征：论文式 concat 结构（见论文 4.2.5 Eq.16）。
        特征 = [ h_L_op ‖ top机器池化 h_L_M ‖ h_G ‖ avg_pair(op,topM) ‖ sublot_feat(ξ) ]
        复用 run_one_episode 已算好的 raw 与嵌入（同一状态），避免重复 GAT 前向。
        """
        raw = getattr(self, '_step_raw', None)
        if raw is None:
            raw = self.hdgat.extract_raw_features(hdg, self.device)
            self._step_raw = raw

        # 复用本步已算的 op/machine 深层嵌入（key = raw 版本号）
        if getattr(self, '_step_embeds_key', None) != raw.get('_ver'):
            self._step_embeds = self._encode_step(raw)
            self._step_embeds_key = raw.get('_ver')

        op_embeds, mach_embeds, g_global = self._step_embeds

        op_keys = raw['op_keys']
        idx_map = {k: i for i, k in enumerate(op_keys)}

        cmax_norm = self.cmax_norm_factor
        energy_norm = self.energy_norm_factor
        d_hidden = self.hdgat.d_hidden

        features = []
        for action in actions:
            op_key, xi = action
            if op_key is None or op_key not in hdg.operation_nodes:
                features.append(torch.zeros(self.actor.d_action, device=self.device))
                continue

            op = hdg.operation_nodes[op_key]
            oi = idx_map.get(op_key)
            if oi is None:
                features.append(torch.zeros(self.actor.d_action, device=self.device))
                continue

            h_op = op_embeds[oi]  # (d_hidden,)

            # 候选机器 = 该工序所有候选机器（MILP 会用其中一部分）
            cand_machines = sorted(op.machine_candidates)
            if not cand_machines:
                features.append(torch.zeros(self.actor.d_action, device=self.device))
                continue

            # top 机器（取候选机器中前 2 台，够用）
            cand_idx = [m for m in cand_machines if m < mach_embeds.shape[0]]
            if not cand_idx:
                features.append(torch.zeros(self.actor.d_action, device=self.device))
                continue
            top_idx = cand_idx[:min(2, len(cand_idx))]
            top_mach_pool = mach_embeds[top_idx].mean(dim=0)  # (d_hidden,)

            # pair 特征（工序-机器对）：加工时间、切换、能耗、机器负载 等 6 维
            pair_feats = []
            for m in top_idx:
                base_pt = op.base_pt.get(m, 1.0)
                ms = self.machine_states.get(m)
                switch = self.config.machine_data.get(m, {}).get('switch_time', 0.0)
                energy_proc = self.config.machine_data.get(m, {}).get('energy_process', 0.0)
                next_avail = ms.next_available_time if ms else 0.0
                load = ms.total_processing_time if ms else 0.0
                pair_feats.append([
                    base_pt / 5.0,
                    switch / max(1e-6, cmax_norm),
                    energy_proc / max(1e-6, energy_norm) if energy_norm > 0 else 0.0,
                    next_avail / max(1e-6, cmax_norm),
                    min(1.0, load / max(1e-6, cmax_norm)),
                    1.0 if (ms and ms.is_busy) else 0.0,
                ])
            while len(pair_feats) < 2:
                pair_feats.append([0.0] * 6)
            pair_feats = torch.tensor(pair_feats[:2], dtype=torch.float32, device=self.device)
            avg_pair = pair_feats.mean(dim=0)  # (6,)

            # sublot(ξ) 特征：ξ、估计批次数、估计平均尺寸、估计总时长、切换估计
            D_rem = max(1, op.total_pieces - op.completed_pieces)
            est_k = max(1, int(round((1 - xi) * len(cand_machines) + 0.5)))
            est_k = min(est_k, len(cand_machines))
            avg_base = np.mean([op.base_pt.get(m, 1.0) for m in cand_machines])
            avg_size = D_rem / max(1, est_k)
            est_total_time = avg_base * D_rem
            est_setup = sum(self.config.machine_data.get(m, {}).get('switch_time', 0.0)
                            for m in cand_machines[:est_k])
            sublot_feat = torch.tensor([
                xi,
                est_k / max(1, len(cand_machines)),
                avg_size / max(1, D_rem),
                est_total_time / max(1e-6, cmax_norm),
                est_setup / max(1e-6, cmax_norm),
                0.0,
            ], dtype=torch.float32, device=self.device)

            feat = torch.cat([h_op, top_mach_pool, g_global, avg_pair, sublot_feat])
            # 维度检查：d_hidden + d_hidden + D_global + 6 + 6
            features.append(feat)

        if features:
            feat_tensor = torch.stack(features)  # (K, D_action)
        else:
            feat_tensor = torch.zeros((0, self.actor.d_action), device=self.device)
        return feat_tensor

    # ---------------- 覆盖：执行动作（MILP 定尺寸 + 贪心工人）----------------
    def _execute_action(self, action):
        """执行 (工序, ξ) 动作：
        1) MILP 求分批尺寸 + 机器选择（三目标加权，权重取当前 critic 权重）
        2) 按尺寸降序贪心分配"疲劳最低空闲工人"
        3) 逐子批执行（切换/待机能耗、疲劳更新、完成时间与自研完全一致）
        """
        op_key, xi = action
        if op_key is None:
            return
        op = self.hdg.operation_nodes.get(op_key)
        if op is None:
            return

        D_rem = op.total_pieces - op.completed_pieces
        if D_rem <= 0:
            return

        cand_machines = sorted(op.machine_candidates)
        if not cand_machines:
            return

        # 机器可用时间
        machine_start = {m: self.machine_states[m].next_available_time
                         for m in cand_machines if m in self.machine_states}
        # 每台机器拟分配工人：当前疲劳最低的空闲工人
        idle_workers = [w for w in self.hdg.worker_nodes.values()
                        if w.is_available(self.hdg.current_time)]
        worker_lambda, worker_fatigue = {}, {}
        for m in cand_machines:
            # 优先选疲劳最低的空闲工人；不够则取所有工人里疲劳最低的
            if idle_workers:
                w = min(idle_workers, key=lambda w: w.current_fatigue)
            else:
                w = min(self.hdg.worker_nodes.values(), key=lambda w: w.current_fatigue)
            worker_lambda[m] = w.lambda_
            worker_fatigue[m] = w.current_fatigue

        energy_process = {m: self.config.machine_data.get(m, {}).get('energy_process', 0.0)
                          for m in cand_machines}
        switch_time = {m: self.config.machine_data.get(m, {}).get('switch_time', 0.0)
                       for m in cand_machines}
        base_pt = {m: op.base_pt.get(m, 1.0) for m in cand_machines}

        # 最大子批数 ≤ min(候选机器数, 可用工人数)
        max_sublots = min(len(cand_machines), max(1, len(self.hdg.worker_nodes)))

        w1, w2, w3 = self.last_weights
        sizes = solve_operation_milp(
            cand_machines, D_rem, base_pt,
            machine_start=machine_start,
            energy_process=energy_process,
            switch_time=switch_time,
            worker_lambda=worker_lambda,
            worker_fatigue=worker_fatigue,
            xi=float(xi),
            weights=(w1, w2, w3),
            w_cmax=self.cmax_norm_factor,
            w_energy=self.energy_norm_factor,
            fatigue_scale=1.0,
            max_sublots=max_sublots,
        )
        if not sizes:
            sizes = _fallback_split(cand_machines, D_rem, base_pt, max_sublots)
        if not sizes:
            return

        self.batch_select_count[len(sizes)] += 1

        # 按尺寸降序分配工人（低疲劳优先）
        ordered = sorted(sizes.items(), key=lambda kv: -kv[1])
        assigned = []  # (machine, size, worker)
        used_workers = set()
        for machine_id, size in ordered:
            avail = [w for w in self.hdg.worker_nodes.values()
                     if w.worker_id not in used_workers and w.is_available(self.hdg.current_time)]
            if avail:
                worker = min(avail, key=lambda w: w.current_fatigue)
            else:
                worker = min((w for w in self.hdg.worker_nodes.values()
                              if w.worker_id not in used_workers),
                             key=lambda w: w.current_fatigue)
            used_workers.add(worker.worker_id)
            assigned.append((machine_id, size, worker))

        # 逐子批执行（与自研 _execute_action 逻辑一致）
        prev_op_completion_time = op.get_prev_op_completion_time()
        max_completion_time = 0
        completed_sizes = 0
        for machine_id, size, worker in assigned:
            machine_state = self.machine_states.get(machine_id)
            if machine_state is None:
                continue

            switch_t = self.config.machine_data.get(machine_id, {}).get('switch_time', 0.0)
            machine_available = machine_state.next_available_time
            worker_available = worker.next_available_time

            if machine_state.is_first_use or machine_available == 0:
                machine_ready_time = machine_available
                actual_switch = 0.0
            else:
                machine_ready_time = machine_available + switch_t
                actual_switch = switch_t

            start_time = max(prev_op_completion_time, machine_ready_time, worker_available)
            idle_time = start_time - machine_available - actual_switch
            if idle_time > 0:
                machine_state.add_idle_energy(idle_time)
            if actual_switch > 0:
                machine_state.add_switch_energy(actual_switch)
            machine_state.is_first_use = False

            base_time = op.base_pt.get(machine_id, 1.0)
            fatigue_factor = 1 + worker.current_fatigue
            actual_time = base_time * size * fatigue_factor
            complete_time = start_time + actual_time

            batch = SubBatch(op_key, 0, size, machine_id, worker.worker_id)
            batch.start_time = start_time
            batch.complete_time = complete_time
            batch.status = "completed"
            worker.assign_batch(batch)
            machine_state.assign_batch(batch, start_time, actual_time)
            worker.complete_batch(complete_time, start_time, actual_time)
            machine_state.complete_batch()

            completed_sizes += size
            if complete_time > max_completion_time:
                max_completion_time = complete_time

        if max_completion_time > 0 and completed_sizes > 0:
            op.update_completion(completed_sizes, max_completion_time)
            self.hdg.current_time = max_completion_time

        self.hdg.update_operation_status()

    # ---------------- 覆盖：奖励（三目标，与自研一致）----------------
    def _compute_reward(self, hdg, prev_state, current_state, action):
        makespan_inc = current_state['makespan'] - prev_state['makespan']
        energy_inc = current_state['energy'] - prev_state['energy']
        fatigue_inc = current_state['fatigue_mean'] - prev_state['fatigue_mean']

        r_cmax = - (makespan_inc * 10.0) / (self.cmax_norm_factor + 1e-6)
        r_energy = - (energy_inc * 10.0) / (self.energy_norm_factor + 1e-6)
        r_fatigue = - fatigue_inc * 1.0

        r_cmax = float(np.clip(r_cmax, -2.0, 2.0))
        r_energy = float(np.clip(r_energy, -2.0, 2.0))
        r_fatigue = float(np.clip(r_fatigue, -2.0, 2.0))

        return [r_cmax, r_energy, r_fatigue], current_state

    # ---------------- 覆盖：维度校验 ----------------
    def _validate_dimensions(self):
        try:
            test_hdg = HeterogeneousDisjunctiveGraph(self.config)
            _, global_feature = self.hdgat(test_hdg, self.device)
            print(f"[IMPRL] GAT测试通过: global_feature维度={global_feature.shape}")
            dummy_action_feats = torch.randn(1, 3, self.actor.d_action).to(self.device)
            dummy_mask = torch.ones(1, 3).to(self.device)
            scores, probs = self.actor(global_feature.unsqueeze(0), dummy_action_feats, mask=dummy_mask)
            print(f"[IMPRL] Actor测试通过: scores维度={scores.shape}")
        except Exception as e:
            print(f"[IMPRL] 维度验证失败: {e}")
            raise
