# 操作级 MILP：分批尺寸 + 机器选择协同优化（论文 4.3 节）
# 目标（三目标加权版，把论文单目标 C_ij 扩展为三目标加权）：
#   min (1-ξ)·[ w1·C_ij/W_cmax + w2·E/W_energy + w3·F_est ] + ξ·Σ S_ε·x_ε
#   其中 E = Σ σ_ε·P_ε·e_proc_ε，F_est 为疲劳一阶近似（见下）
# 约束（论文 Eq.1-4 + 附加）：
#   Σ σ_ε = D                          (1) 各机器子批件数之和等于总需求
#   c_ε ≥ Start_ε + S_ε + σ_ε·P_ε       (2) 子批完成时间（x_ε=1 时生效）
#   σ_ε ≤ x_ε·D                         (3) 选中才可分配件数
#   C_ij ≥ c_ε                          (4) 工序完成时间取最大
#   Σ x_ε ≤ max_sublots                      子批数上限（≤ 工人数）
#   σ_ε ∈ Z+                             件数为整数
#
# 疲劳项线性近似：F_est = Σ (1 - f_ε)·λ_ε·P_ε·σ_ε，
# 其中 f_ε = 拟分配给机器 m 的工人当前疲劳度，λ_ε = 该工人疲劳增长系数。
# 真实疲劳由环境精确计算（指数增长），此处用一阶近似作为优化导向。

import pulp


def solve_operation_milp(candidate_machines, D, base_pt,
                         machine_start=None, energy_process=None, switch_time=None,
                         worker_lambda=None, worker_fatigue=None,
                         xi=0.0, weights=(1.0, 0.0, 0.0),
                         w_cmax=1.0, w_energy=1.0, fatigue_scale=1.0,
                         max_sublots=None, time_limit=2.0):
    """
    对单个工序求解分批 + 机器选择 MILP。

    Parameters
    ----------
    candidate_machines : list[int]
        候选机器 id 列表。
    D : int
        该工序待加工总件数。
    base_pt : dict[int, float]
        machine_id -> 单件加工时间。
    machine_start : dict[int, float] | None
        machine_id -> 机器最早可用时间。None 则取 0。
    energy_process : dict[int, float] | None
        machine_id -> 单位时间加工能耗。None 则取 0。
    switch_time : dict[int, float] | None
        machine_id -> 切换时间。None 则取 0。
    worker_lambda : dict[int, float] | None
        machine_id -> 拟分配工人的疲劳增长系数。None 则取 0.05。
    worker_fatigue : dict[int, float] | None
        machine_id -> 拟分配工人的当前疲劳度。None 则取 0。
    xi : float
        分批系数 ξ ∈ [0,1]。ξ 越大越倾向减少切换/机器数。
    weights : tuple[float,float,float]
        三目标权重 (w_cmax, w_energy, w_fatigue)，调用方（critic）动态给定。
    w_cmax, w_energy : float
        归一化基准。
    fatigue_scale : float
        疲劳项整体缩放（默认 1.0）。
    max_sublots : int | None
        最大子批数上限。None 则取候选机器数。
    time_limit : float
        CBC 求解时间上限（秒）。候选机器数小，通常 <1ms。

    Returns
    -------
    dict[int, int] | None
        machine_id -> 子批件数（仅 x_ε=1 的机器）。不可行/异常返回 None。
    """
    if D <= 0 or not candidate_machines:
        return None

    machine_start = machine_start or {}
    energy_process = energy_process or {}
    switch_time = switch_time or {}
    worker_lambda = worker_lambda or {}
    worker_fatigue = worker_fatigue or {}

    M = len(candidate_machines)
    if max_sublots is None:
        max_sublots = M
    max_sublots = max(1, min(max_sublots, M))

    prob = pulp.LpProblem("OperationMILP", pulp.LpMinimize)

    x = {m: pulp.LpVariable(f"x_{m}", 0, 1, cat=pulp.LpBinary) for m in candidate_machines}
    sigma = {m: pulp.LpVariable(f"s_{m}", 0, D, cat=pulp.LpInteger) for m in candidate_machines}
    c = {m: pulp.LpVariable(f"c_{m}", 0, None, cat=pulp.LpContinuous) for m in candidate_machines}
    C_ij = pulp.LpVariable("C", 0, None, cat=pulp.LpContinuous)

    w1, w2, w3 = weights
    w1 = max(0.0, w1); w2 = max(0.0, w2); w3 = max(0.0, w3)

    # ---- 目标 ----
    obj = []
    for m in candidate_machines:
        P = base_pt.get(m, 1.0)
        S = switch_time.get(m, 0.0)
        e = energy_process.get(m, 0.0)
        lam = worker_lambda.get(m, 0.05)
        f = worker_fatigue.get(m, 0.0)

        # (1-ξ)·[ w1·C/W + w2·E/W + w3·F ]
        obj.append((1 - xi) * w1 * C_ij / max(1e-9, w_cmax))
        obj.append((1 - xi) * w2 * sigma[m] * P * e / max(1e-9, w_energy))
        obj.append((1 - xi) * w3 * fatigue_scale * (1 - f) * lam * P * sigma[m])
        # ξ·S_ε·x_ε
        obj.append(xi * S * x[m])
    prob += pulp.lpSum(obj)

    # ---- 约束 ----
    # (1) 总件数
    prob += pulp.lpSum(sigma[m] for m in candidate_machines) == D

    # (2) 完成时间（big-M 使未选中机器约束失效）
    BIGM = 1e6
    for m in candidate_machines:
        P = base_pt.get(m, 1.0)
        S = switch_time.get(m, 0.0)
        start = machine_start.get(m, 0.0)
        prob += c[m] >= start + S + sigma[m] * P - BIGM * (1 - x[m])
        prob += c[m] >= 0

    # (3) 选中才可分配
    for m in candidate_machines:
        prob += sigma[m] <= x[m] * D

    # (4) C_ij ≥ c_ε
    for m in candidate_machines:
        prob += C_ij >= c[m]

    # 子批数上限
    prob += pulp.lpSum(x[m] for m in candidate_machines) <= max_sublots

    # ---- 求解 ----
    try:
        prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit))
    except Exception:
        return _fallback_split(candidate_machines, D, base_pt, max_sublots)

    status = pulp.LpStatus[prob.status]
    if status not in ("Optimal", "Not Solved"):
        return _fallback_split(candidate_machines, D, base_pt, max_sublots)

    try:
        sizes = {}
        total = 0
        for m in candidate_machines:
            v = int(round(sigma[m].value() or 0))
            if v > 0:
                sizes[m] = v
            total += v
        # 校验：件数和必须等于 D，且使用了不超过 max_sublots 台机器
        used = len(sizes)
        if total != D or used == 0 or used > max_sublots:
            return _fallback_split(candidate_machines, D, base_pt, max_sublots)
        return sizes
    except Exception:
        return _fallback_split(candidate_machines, D, base_pt, max_sublots)


def _fallback_split(candidate_machines, D, base_pt, max_sublots):
    """回退：按加工能力（1/P）加权分配，最多 max_sublots 台机器，保证件数和 = D。"""
    if D <= 0 or not candidate_machines:
        return None
    n_use = max(1, min(len(candidate_machines), max_sublots))
    used = sorted(candidate_machines, key=lambda m: base_pt.get(m, 1e9))[:n_use]

    caps = [1.0 / max(1e-6, base_pt.get(m, 1.0)) for m in used]
    total_cap = sum(caps)
    sizes = {}
    remaining = D
    for i, m in enumerate(used[:-1]):
        s = max(1, int(round(D * caps[i] / total_cap)))
        sizes[m] = s
        remaining -= s
    sizes[used[-1]] = max(1, remaining)
    # 极端情况修复：保证总和 = D
    if sum(sizes.values()) != D:
        sizes[used[-1]] = D - sum(sizes[m] for m in used[:-1])
        sizes[used[-1]] = max(1, sizes[used[-1]])
    return sizes
