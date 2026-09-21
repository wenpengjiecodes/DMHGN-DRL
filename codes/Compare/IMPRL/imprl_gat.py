# 论文式双注意力 GAT 编码器（IMPRL 第 4.1 节）
#
# 结构（忠实论文公式）：
#   1) OperationMessageAttention（Eq.5-7）：工序节点沿同一工件内的前驱/后继做注意力，
#      刻画工序优先级。h'_O = LeakyReLU(Σ α_neigh · W h_neigh)
#   2) MachineFitnessAttention（Eq.8-10）：两机器共享的可排待排工序集合构成兼容性向量
#      f_kq = Σ_{O ∈ F_kq ∩ J_unscheduled} h_O，再算机器间注意力
#      u_kq = LeakyReLU(b^T[Z1 h_Mk ‖ Z1 h_Mq ‖ Z2 f_kq])，softmax 后
#      h'_Mk = LeakyReLU(Σ β_kq · Z1 h_Mq)
#   3) 全局特征 = stats_vector(5) + 工序池化 + 机器池化 + 子批(ξ节点)池化
#
# 与 my/train/V1_01.py 的 HDGAT 接口完全一致：
#   set_agent(agent) / extract_raw_features(hdg, device) / encode_raw_features(raw, device)
#   / forward(hdg, device) -> (None, global_feature)
# 以便复用 PPOAgent 的 run_one_episode / update_policy / save_models 等。
#
# 节点特征维度：
#   工序 op_feats  : 9 维
#   机器 mach_feats: 6 维
#   子批(ξ) b_feats: 9 维（B1..BM 节点，M=机器数，每个对应一个分批数 k）
#   worker_feats  : 5 维（仅作全局池化，不参与 GAT 注意力）
#   全局 stats    : 5 维

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


class OperationMessageAttention(nn.Module):
    """工序消息注意力块（论文 Eq.5-7）：
    对每个工序，沿同一工件内前驱/后继做 GAT，更新工序嵌入。

    性能优化：邻接结构（前驱/后继索引）在 job 拓扑不变时是固定的，
    仅在 _op_offsets 内容或工序数变化时重建一次，避免每步重复 Python 循环。
    """

    def __init__(self, d_op, d_out=None):
        super().__init__()
        d_out = d_out or d_op
        self.d_out = d_out
        self.W = nn.Linear(d_op, d_out, bias=False)
        self.a = nn.Parameter(torch.randn(2 * d_out))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self._adj_ids_key = None
        self._adj_N = None
        self._adj = None

    def forward(self, op_embeds):
        """op_embeds: (N, d_op)。同一工件内相邻工序（前驱/后继）互为邻居。"""
        N = op_embeds.shape[0]
        if N <= 1:
            return op_embeds
        h = self.W(op_embeds)  # (N, d_out)

        adj = self._get_adjacency(N, h.device)
        if adj is None:
            return op_embeds
        nbr, valid = adj

        # 批量注意力（Eq.5-6）
        hi = h.unsqueeze(1).expand(N, 2, -1)                    # (N,2,d_out)
        hn = h[nbr]                                             # (N,2,d_out)
        pair = torch.cat([hi, hn], dim=-1)                      # (N,2,2*d_out)
        e = self.leaky_relu(pair @ self.a)                      # (N,2)
        e = e.masked_fill(~valid, -1e9)
        alpha = F.softmax(e, dim=-1)                            # (N,2)
        alpha = alpha * valid.float()                           # 边界邻居权重=0

        # Eq.7: h'_i = LeakyReLU(Σ α_j W h_j)
        agg = (alpha.unsqueeze(-1) * hn).sum(dim=1)             # (N,d_out)
        has_neighbor = valid.any(dim=1, keepdim=True).float()
        out = self.leaky_relu(agg * has_neighbor + h * (1 - has_neighbor))
        return out

    def _get_adjacency(self, N, device):
        """缓存的前驱/后继邻接矩阵。
        key = (op_job_ids 内容, N)：拓扑不变（同一 episode 内大部分步）时跨步命中；
        动态工件到达（op_job_ids 变化）时重建。内容 key 无对象 id 复用风险。
        由 encode_raw_features 在调用前设置 self._adj_ids = tuple(op_job_ids)。"""
        ids = getattr(self, '_adj_ids', None)
        if ids is None or not getattr(self, '_op_offsets', None):
            return None
        if (getattr(self, '_adj_ids_key', None) != ids) or (getattr(self, '_adj_N', None) != N):
            offsets = self._op_offsets
            # 反向：index -> (job_id, 在 job 工序序列中的位置)
            pos_in_job = {}
            for jid, seq in offsets.items():
                for p, idx in enumerate(seq):
                    pos_in_job[idx] = (jid, p)
            nbr = torch.full((N, 2), -1, dtype=torch.long, device=device)
            for i in range(N):
                if i not in pos_in_job:
                    continue
                jid, p = pos_in_job[i]
                seq = offsets[jid]
                if p > 0:
                    nbr[i, 0] = seq[p - 1]
                if p < len(seq) - 1:
                    nbr[i, 1] = seq[p + 1]
            self._adj_ids_key = ids
            self._adj_N = N
            self._adj = (nbr.clamp(min=0), nbr >= 0)
        return self._adj

    def _get_neighbors(self, i, index_to_job):
        """保留：单工序邻居查询（测试/调试用）。"""
        if hasattr(self, '_op_offsets') and self._op_offsets and index_to_job is not None:
            jid = index_to_job[i]
            seq = self._op_offsets[jid]  # 该工件的 op 索引序列（已按 op_id 排序）
            pos = seq.index(i)
            nbrs = []
            if pos > 0:
                nbrs.append(seq[pos - 1])
            if pos < len(seq) - 1:
                nbrs.append(seq[pos + 1])
            return nbrs
        return []


class MachineFitnessAttention(nn.Module):
    """机器适配注意力块（论文 Eq.8-10）：
    两机器共享的可排待排工序构成兼容性向量 f_kq，机器间注意力更新机器嵌入。
    """

    def __init__(self, d_mach, d_op, d_out=None):
        super().__init__()
        d_out = d_out or d_mach
        self.d_out = d_out
        self.Z1 = nn.Linear(d_mach, d_out, bias=False)
        self.Z2 = nn.Linear(d_op, d_out, bias=False)
        self.b = nn.Parameter(torch.randn(3 * d_out))
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, mach_embeds, op_embeds, mach_op_mask):
        """mach_embeds: (K, d_mach)；op_embeds: (N, d_op)；
        mach_op_mask: (K, N) 0/1，表示机器 m 能否加工工序 o。
        兼容性向量 f_kq = Σ_{o∈F_kq∩未排} h_o（k,q 都能加工的未排工序）。
        """
        K, N = mach_embeds.shape[0], op_embeds.shape[0]
        if K <= 1 or N == 0:
            return mach_embeds
        h = self.Z1(mach_embeds)      # (K, d_out)
        # f_kq: (K, K, d_out)：机器对共享可排工序的嵌入和
        # 对每个机器 k，其可排工序集合 = mask[k]；两机器交集 = mask[k] & mask[q]
        # f_kq = Σ_o (mask[k,o]·mask[q,o]) · Z2(h_o)
        op_proj = self.Z2(op_embeds)  # (N, d_out)
        m = mach_op_mask.float()       # (K, N)
        # (K,1,N) * (1,K,N) -> (K,K,N)，再乘 op_proj -> (K,K,d_out)
        inter = m.unsqueeze(1) * m.unsqueeze(0)          # (K,K,N)
        f = torch.einsum('kqo,od->kqd', inter, op_proj)  # (K,K,d_out)

        # Eq.9: u_kq = LeakyReLU(b^T[Z1 h_Mk ‖ Z1 h_Mq ‖ Z2 f_kq])
        hk = h.unsqueeze(1).expand(K, K, -1)  # (K,K,d_out)
        hq = h.unsqueeze(0).expand(K, K, -1)  # (K,K,d_out)
        pair = torch.cat([hk, hq, f], dim=-1)  # (K,K,3*d_out)
        u = self.leaky_relu(pair @ self.b)     # (K,K)
        # 屏蔽自环与空交集
        diag_mask = torch.eye(K, device=u.device).bool()
        empty = inter.sum(dim=-1) == 0
        u = u.masked_fill(diag_mask | empty, -1e9)
        beta = F.softmax(u, dim=-1)            # (K,K)
        # Eq.10: h'_Mk = LeakyReLU(Σ_q β_kq · Z1 h_Mq)
        out = self.leaky_relu(torch.einsum('kq,qd->kd', beta, h))
        return out


class IMPRL_GAT(nn.Module):
    """论文式双注意力 GAT。接口与 HDGAT 一致。

    节点：
      - 工序节点 op_feats (N, 9)
      - 机器节点 mach_feats (K, 6)，K = num_machines
      - 子批(ξ)节点 b_feats (M, 9)，M = num_machines（B1..BM 对应分批数 1..M）
      - 工人节点 worker_feats (W, 5)，仅池化
    全局特征 = stats_vector(5) + op_pool + mach_pool + b_pool + worker_pool。
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_op = 9
        self.d_mach = 6
        self.d_b = 9
        self.d_worker = 5
        self.d_hidden = 32  # 注意力隐层维度

        self.op_embed = nn.Linear(self.d_op, self.d_hidden)
        self.mach_embed = nn.Linear(self.d_mach, self.d_hidden)
        self.b_embed = nn.Linear(self.d_b, self.d_hidden)
        self.worker_embed = nn.Linear(self.d_worker, self.d_hidden)

        self.op_attn = OperationMessageAttention(self.d_hidden, self.d_hidden)
        self.mach_attn = MachineFitnessAttention(self.d_hidden, self.d_hidden, self.d_hidden)

        self.agent_ref = None

        # 与 config 维度对齐：d_operation/d_batch/d_worker 用于统计拼接
        self.d_operation = config.d_operation
        self.d_batch = config.d_batch
        self.d_worker = config.d_worker
        self.d_global = 5 + self.d_operation + self.d_batch + self.d_worker

        # 池化投影回原维度，使全局特征维度与自研一致（供 Actor/Critic 复用）
        self.op_pool_proj = nn.Linear(self.d_hidden, self.d_operation)
        self.mach_b_pool_proj = nn.Linear(self.d_hidden * 2, self.d_batch)
        self.worker_pool_proj = nn.Linear(self.d_hidden, self.d_worker)

    def set_agent(self, agent):
        self.agent_ref = agent

    # ---------------- 特征提取 ----------------
    def _extract_features(self, hdg, device):
        cfg = self.config
        cmax_norm = self.agent_ref.cmax_norm_factor if self.agent_ref else 1.0
        energy_norm = self.agent_ref.energy_norm_factor if self.agent_ref else 1.0
        current_cmax = max([o.complete_time for o in hdg.operation_nodes.values()] + [0])

        # 工序节点特征 (9 维)
        op_features_list, op_keys_list = [], []
        op_candidates_list = []   # 每 op 候选机器（用于机器-工序 mask，须随 raw 保存）
        op_job_ids_list = []      # 每 op 所属 job_id（用于前驱/后继注意力，须随 raw 保存）
        job_seq_map = defaultdict(list)  # job_id -> [index in list]
        total_ops = len(hdg.operation_nodes)

        # 预处理：各 job 未完成工序数（避免每 op O(N) 全图扫描）
        job_unscheduled = defaultdict(int)
        for _, op in hdg.operation_nodes.items():
            if not op.is_scheduled:
                job_unscheduled[op.job_id] += 1

        for idx, (op_key, op) in enumerate(hdg.operation_nodes.items()):
            pt_values = list(op.base_pt.values()) if op.base_pt else [0.0]
            min_pt, max_pt = min(pt_values), max(pt_values)
            avg_pt = sum(pt_values) / max(1, len(pt_values))
            remaining_pieces = op.total_pieces - op.completed_pieces
            remaining_ops_in_job = job_unscheduled[op.job_id]
            wait_time = max(0.0, current_cmax - max(0.0, op.ready_time))
            op_features_list.append([
                1.0 if op.is_scheduled else 0.0,
                1.0 if op.is_ready() else 0.0,
                min_pt / 5.0,
                max_pt / 5.0,
                avg_pt / 5.0,
                remaining_pieces / max(1, op.total_pieces),
                remaining_ops_in_job / max(1, cfg.num_ops_per_job),
                wait_time / max(1e-6, cmax_norm),
                (max_pt - min_pt) / 5.0,
            ])
            op_keys_list.append(op_key)
            op_candidates_list.append(list(op.machine_candidates))
            op_job_ids_list.append(op.job_id)
            job_seq_map[op.job_id].append(idx)

        # 同一工件内按 op_id 排序，保证前驱/后继注意力顺序正确
        for jid, seq in job_seq_map.items():
            job_seq_map[jid] = sorted(seq, key=lambda i: hdg.operation_nodes[op_keys_list[i]].op_id)

        op_features = torch.tensor(op_features_list, dtype=torch.float32, device=device) if op_features_list \
            else torch.zeros((0, 9), dtype=torch.float32, device=device)

        # 机器节点特征 (6 维)
        mach_features_list = []
        machine_states = self.agent_ref.machine_states if self.agent_ref else {}
        total_energy = sum(m.total_energy for m in machine_states.values())
        for m_id in range(cfg.num_machines):
            ms = machine_states.get(m_id)
            if ms is None:
                mach_features_list.append([0.0] * 6)
                continue
            util = ms.total_processing_time / max(1e-6, cmax_norm)
            mach_features_list.append([
                1.0 if ms.is_busy else 0.0,
                ms.next_available_time / max(1e-6, cmax_norm),
                ms.total_processing_time / max(1e-6, cmax_norm),
                ms.total_energy / max(1e-6, energy_norm),
                min(1.0, util),
                1.0 if ms.is_first_use else 0.0,
            ])
        mach_features = torch.tensor(mach_features_list, dtype=torch.float32, device=device) if mach_features_list \
            else torch.zeros((0, 6), dtype=torch.float32, device=device)

        # 子批(ξ)节点特征 (9 维)：B1..BM
        b_features_list, b_keys_list = [], []
        max_k = max(1, cfg.num_machines)
        select_counts = getattr(self.agent_ref, 'batch_select_count', defaultdict(int)) if self.agent_ref else defaultdict(int)
        total_select = sum(select_counts.values())
        for k in range(1, cfg.num_machines + 1):
            avg_base_pt = 2.0  # 默认估计
            est_batch_time = avg_base_pt * (cfg.total_pieces_per_job / max(1, k))
            b_features_list.append([
                min(1.0, k / max_k),
                1.0 / k,
                est_batch_time / max(1e-6, cmax_norm),
                select_counts.get(k, 0) / max(1, total_select),
                float(k) / max_k,
                min(1.0, cfg.total_pieces_per_job / max(1, k * 2)),
                0.0, 0.0, 0.0,
            ])
            b_keys_list.append(f"B{k}")
        b_features = torch.tensor(b_features_list, dtype=torch.float32, device=device) if b_features_list \
            else torch.zeros((0, 9), dtype=torch.float32, device=device)

        # 工人节点特征 (5 维)
        worker_features_list = []
        for _, worker in hdg.worker_nodes.items():
            worker_features_list.append([
                worker.lambda_, worker.mu, worker.current_fatigue,
                worker.next_available_time / max(1e-6, cmax_norm),
                worker.total_work_time / max(1e-6, cmax_norm),
            ])
        worker_features = torch.tensor(worker_features_list, dtype=torch.float32, device=device) if worker_features_list \
            else torch.zeros((0, 5), dtype=torch.float32, device=device)

        # 全局统计 (5 维)
        workers = list(hdg.worker_nodes.values())
        completed_ops = sum(1 for o in hdg.operation_nodes.values() if o.is_scheduled)
        progress = completed_ops / max(1, total_ops)
        non_fatigue_ratio = sum(1 for w in workers if w.current_fatigue < cfg.fatigue_threshold) / max(1, len(workers))
        avg_fatigue = float(np.mean([w.current_fatigue for w in workers])) if workers else 0.0
        self.stats_vector = torch.tensor([
            progress, non_fatigue_ratio,
            current_cmax / max(1e-6, cmax_norm),
            total_energy / max(1e-6, energy_norm),
            avg_fatigue,
        ], dtype=torch.float32, device=device)

        return (op_features, mach_features, b_features, worker_features,
                op_keys_list, b_keys_list, job_seq_map, op_candidates_list,
                op_job_ids_list)

    def _build_index_to_job(self, raw):
        return raw.get('op_job_ids', [])

    def extract_raw_features(self, hdg, device):
        # 记录本次 extract 时的环境时间，供上层判断状态是否已变化（避免重复前向）
        self._last_extract_time = getattr(hdg, 'current_time', 0.0)
        # 单调递增版本号：作为本步缓存 key，同一步内复用 raw 时命中，跨步必然重建
        self._raw_counter = getattr(self, '_raw_counter', 0) + 1
        (op_feats, mach_feats, b_feats, worker_feats,
         op_keys, b_keys, job_seq_map, op_candidates, op_job_ids) = self._extract_features(hdg, device)
        ver = self._raw_counter
        self._last_ver = ver
        return {
            'op_feats': op_feats.detach().cpu(),
            'mach_feats': mach_feats.detach().cpu(),
            'b_feats': b_feats.detach().cpu(),
            'worker_feats': worker_feats.detach().cpu(),
            'stats_vector': self.stats_vector.detach().cpu(),
            'op_keys': op_keys,
            'b_keys': b_keys,
            'job_seq_map': job_seq_map,
            'op_candidates': op_candidates,
            'op_job_ids': op_job_ids,
            '_ver': ver,
        }

    # ---------------- 编码 ----------------
    def encode_raw_features(self, raw, device):
        op_feats = raw['op_feats'].to(device)
        mach_feats = raw['mach_feats'].to(device)
        b_feats = raw['b_feats'].to(device)
        worker_feats = raw['worker_feats'].to(device)
        stats_vector = raw['stats_vector'].to(device)

        op_embeds = self.op_embed(op_feats) if op_feats.size(0) > 0 else torch.zeros((0, self.d_hidden), device=device)
        mach_embeds = self.mach_embed(mach_feats) if mach_feats.size(0) > 0 else torch.zeros((0, self.d_hidden), device=device)
        b_embeds = self.b_embed(b_feats) if b_feats.size(0) > 0 else torch.zeros((0, self.d_hidden), device=device)
        worker_embeds = self.worker_embed(worker_feats) if worker_feats.size(0) > 0 else torch.zeros((0, self.d_hidden), device=device)

        # 工序消息注意力（Eq.5-7）：沿同工件前驱/后继
        if op_embeds.size(0) > 1:
            self.op_attn._op_offsets = raw.get('job_seq_map', {})
            self.op_attn._adj_ids = tuple(raw.get('op_job_ids', []))
            op_embeds = self.op_attn(op_embeds)

        # 机器适配注意力（Eq.8-10）
        if mach_embeds.size(0) > 1 and op_embeds.size(0) > 0:
            mach_op_mask = self._build_mach_op_mask(raw, device)
            if mach_op_mask is not None:
                mach_embeds = self.mach_attn(mach_embeds, op_embeds, mach_op_mask)

        # 池化 + 投影回自研维度，保证全局特征结构与 HDGAT 一致：
        #   cat([stats(5), op_pool(d_operation), batch_pool(d_batch), worker_pool(d_worker)])
        op_pool = op_embeds.mean(dim=0) if op_embeds.size(0) > 0 else torch.zeros(self.d_hidden, device=device)
        mach_pool = mach_embeds.mean(dim=0) if mach_embeds.size(0) > 0 else torch.zeros(self.d_hidden, device=device)
        worker_pool = worker_embeds.mean(dim=0) if worker_embeds.size(0) > 0 else torch.zeros(self.d_hidden, device=device)

        # 子批(ξ)信息并入"batch_pool"位置：机器池化 + 子批池化拼接后投影到 d_batch
        b_pool = b_embeds.mean(dim=0) if b_embeds.size(0) > 0 else torch.zeros(self.d_hidden, device=device)

        op_pool = self.op_pool_proj(op_pool)                      # (d_operation,)
        batch_pool = self.mach_b_pool_proj(torch.cat([mach_pool, b_pool], dim=-1))  # (d_batch,)
        worker_pool = self.worker_pool_proj(worker_pool)          # (d_worker,)

        return torch.cat([stats_vector, op_pool, batch_pool, worker_pool], dim=-1)

    def _build_mach_op_mask(self, raw, device):
        """机器-工序兼容性 mask (K, N)：机器 m 能否加工工序 o。
        用 raw 自带的 op_candidates 构建，K = 该 raw 的机器节点数，
        从而支持不同实例样本混合进 buffer 后仍能独立编码。

        性能优化：op 的候选机器集合在 episode 内不变（除非动态工件到达），
        掩码结果仅依赖候选集合与机器数，缓存避免每步重建。"""
        op_candidates = raw.get('op_candidates', [])
        n_mach = raw['mach_feats'].shape[0]
        if not op_candidates or n_mach == 0:
            return None
        # key = (候选机器内容, 机器数)：拓扑不变时跨步命中，动态工件到达时重建
        cand_ids = tuple(tuple(c) for c in op_candidates)
        if (getattr(self, '_mask_ids', None) != cand_ids) or (getattr(self, '_mask_N', None) != n_mach):
            N = len(op_candidates)
            mask = torch.zeros((n_mach, N), dtype=torch.float32, device=device)
            for j, cand in enumerate(op_candidates):
                for m in cand:
                    if 0 <= m < n_mach:
                        mask[m, j] = 1.0
            self._mask_ids = cand_ids
            self._mask_N = n_mach
            self._mask = mask
        return self._mask

    def forward(self, hdg, device):
        raw = self.extract_raw_features(hdg, device)
        return None, self.encode_raw_features(raw, device)


# 便捷：把 op 特征传给 OperationMessageAttention 时按 job 分组需要 job_ids
# 通过 extract_raw_features 里的 job_seq_map 反向得到每个 op 的 job_id 与 op_id。
def _build_job_ids(hdg, op_keys):
    return [hdg.operation_nodes[k].job_id for k in op_keys]
