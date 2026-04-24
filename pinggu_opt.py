# -*- coding: utf-8 -*-
"""
IEEE33 韧性优化（增强版）
- 严格树拓扑 MILP（SCF）
- 时序 DER CSV 读取
- 开关动作次数约束（相对初始状态）
- 分线路故障率/修复时间
- 多场景真实重求解 + 柱状图
"""

import os
import json
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import networkx as nx

# =========================
# 0) 参数区
# =========================
DATA_DIR = "data"
OUT_DIR = "outputs"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

ROOT_NODE = 0
TS_HOUR = 1.0
MAX_SWITCH_ACTIONS = 6   # 开关动作上限（可调）
BIGM_FLOW = None         # 若None则自动用(num_nodes-1)
GUROBI_OUTPUT = 1

# =========================
# 1) IEEE33 数据
# =========================
num_nodes = 33
load_data_raw = [
    [0, 0.0], [1, 100.0], [2, 90.0], [3, 120.0], [4, 60.0], [5, 60.0], [6, 200.0], [7, 200.0],
    [8, 60.0], [9, 60.0], [10, 45.0], [11, 60.0], [12, 60.0], [13, 120.0], [14, 60.0], [15, 60.0],
    [16, 60.0], [17, 90.0], [18, 90.0], [19, 90.0], [20, 90.0], [21, 90.0], [22, 90.0], [23, 420.0],
    [24, 420.0], [25, 60.0], [26, 60.0], [27, 60.0], [28, 120.0], [29, 200.0], [30, 150.0], [31, 210.0],
    [32, 60.0]
]
load_power = np.array([x[1] for x in load_data_raw], dtype=float)
load_weights = np.ones(num_nodes)
load_weights[[6, 23, 24, 29]] = 10.0

# [branch_id, from, to, sw_type(0分段,1联络)]
branch_data_raw = np.array([
    [0, 0, 1, 0], [1, 1, 2, 0], [2, 2, 3, 0], [3, 3, 4, 0], [4, 4, 5, 0], [5, 5, 6, 0],
    [6, 6, 7, 0], [7, 7, 8, 0], [8, 8, 9, 0], [9, 9, 10, 0], [10, 11, 12, 0], [11, 12, 13, 0],
    [12, 13, 14, 0], [13, 14, 15, 0], [14, 15, 16, 0], [15, 16, 17, 0], [16, 17, 18, 0], [17, 1, 19, 0],
    [18, 19, 20, 0], [19, 20, 21, 0], [20, 2, 22, 0], [21, 22, 23, 0], [22, 23, 24, 0], [23, 5, 25, 0],
    [24, 26, 27, 0], [25, 27, 28, 0], [26, 28, 29, 0], [27, 29, 30, 0], [28, 30, 31, 0], [29, 31, 32, 0],
    [30, 2, 22, 0], [31, 24, 25, 0],  # 2-22重复，后续去重
    [32, 7, 20, 1], [33, 8, 14, 1], [34, 11, 21, 1], [35, 17, 32, 1], [36, 24, 28, 1]
], dtype=int)

# =========================
# 2) 工具函数
# =========================
def deduplicate_branches(branch_data: np.ndarray) -> pd.DataFrame:
    seen = set()
    rows = []
    for b, u, v, t in branch_data:
        key = (min(int(u), int(v)), max(int(u), int(v)))
        if key in seen:
            continue
        seen.add(key)
        rows.append([int(b), int(u), int(v), int(t)])
    return pd.DataFrame(rows, columns=["branch_id", "from_node", "to_node", "sw_type"]).sort_values("branch_id").reset_index(drop=True)

def ensure_sample_csvs():
    # 若用户未提供，则生成示例24h曲线
    pv_path = os.path.join(DATA_DIR, "pv_profile.csv")
    wt_path = os.path.join(DATA_DIR, "wt_profile.csv")
    ld_path = os.path.join(DATA_DIR, "load_profile.csv")

    if not os.path.exists(pv_path):
        h = np.arange(24)
        pv = np.zeros(24)
        for i in range(24):
            if 6 <= i <= 18:
                pv[i] = np.sin((i - 6) / 12 * np.pi)
        pd.DataFrame({"hour": h, "cf": np.clip(pv, 0, 1)}).to_csv(pv_path, index=False)

    if not os.path.exists(wt_path):
        h = np.arange(24)
        wt = 0.35 + 0.15*np.cos((h-2)/24*2*np.pi) + 0.05*np.sin(h/24*4*np.pi)
        pd.DataFrame({"hour": h, "cf": np.clip(wt, 0.05, 0.85)}).to_csv(wt_path, index=False)

    if not os.path.exists(ld_path):
        h = np.arange(24)
        lf = 0.75 + 0.15*np.sin((h-7)/24*2*np.pi) + 0.12*np.sin((h-19)/24*2*np.pi)
        pd.DataFrame({"hour": h, "lf": np.clip(lf, 0.55, 1.05)}).to_csv(ld_path, index=False)

def load_profiles() -> pd.DataFrame:
    pv = pd.read_csv(os.path.join(DATA_DIR, "pv_profile.csv"))
    wt = pd.read_csv(os.path.join(DATA_DIR, "wt_profile.csv"))
    ld = pd.read_csv(os.path.join(DATA_DIR, "load_profile.csv"))
    df = pv.rename(columns={"cf":"cf_pv"}).merge(
        wt.rename(columns={"cf":"cf_wt"}), on="hour", how="inner"
    ).merge(ld[["hour","lf"]], on="hour", how="inner").sort_values("hour").reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("时序数据为空，请检查hour对齐")
    return df

def approximate_branch_lengths(branch_df: pd.DataFrame) -> np.ndarray:
    # 无地理坐标时的近似长度（可替换为真实线路长度）
    # 分段线稍长、联络线偏长
    lengths = np.zeros(len(branch_df))
    for k, r in branch_df.iterrows():
        u, v, t = int(r["from_node"]), int(r["to_node"]), int(r["sw_type"])
        base = 0.30 + 0.02 * abs(u - v)   # km 近似
        if t == 1:
            base *= 1.35
        lengths[k] = base
    return lengths

def build_failure_and_repair_params(branch_df: pd.DataFrame, lengths_km: np.ndarray):
    """
    分线路参数：
    lambda_k = lambda_per_km * length * type_factor
    tr_k = tr_base + tr_per_km * length + type_extra
    """
    # 可根据实际地区统计替换
    lambda_per_km_year = 0.08
    tr_base = 2.0
    tr_per_km = 1.5
    type_factor = np.where(branch_df["sw_type"].to_numpy()==1, 1.20, 1.00)
    type_extra = np.where(branch_df["sw_type"].to_numpy()==1, 0.8, 0.3)

    lambda_k = lambda_per_km_year * lengths_km * type_factor
    tr_k = tr_base + tr_per_km * lengths_km + type_extra
    return lambda_k, tr_k

def generate_m_grid(num_nodes: int, branch_df: pd.DataFrame, root=0):
    """
    用常闭分段开关构造基础图，得到路径边矩阵M_grid[k,i]
    """
    M = np.zeros((len(branch_df), num_nodes))
    G = nx.Graph()
    edge_to_k = {}
    for k, r in branch_df.iterrows():
        u, v, t = int(r["from_node"]), int(r["to_node"]), int(r["sw_type"])
        key = (min(u,v), max(u,v))
        edge_to_k[key] = k
        if t == 0:
            G.add_edge(u, v)

    for i in range(1, num_nodes):
        try:
            p = nx.shortest_path(G, source=root, target=i)
            for j in range(len(p)-1):
                a, b = p[j], p[j+1]
                k = edge_to_k[(min(a,b), max(a,b))]
                M[k, i] = 1.0
        except nx.NetworkXNoPath:
            pass
    return M

def build_p_comp_from_timeseries(load_power, load_weights, ts_df, pv_cap_kw, wt_cap_kw, eps=1e-6):
    cf_pv = ts_df["cf_pv"].to_numpy()
    cf_wt = ts_df["cf_wt"].to_numpy()
    lf = ts_df["lf"].to_numpy()

    N, T = len(load_power), len(ts_df)
    supply_ratio = np.zeros((N, T))
    der_ts = np.zeros((N, T))
    load_ts = np.zeros((N, T))

    for t in range(T):
        der_t = pv_cap_kw * cf_pv[t] + wt_cap_kw * cf_wt[t]
        load_t = load_power * lf[t]
        ratio = np.minimum(1.0, der_t / (load_t + eps))
        der_ts[:, t] = der_t
        load_ts[:, t] = load_t
        supply_ratio[:, t] = ratio

    p_comp = 1.0 - supply_ratio.mean(axis=1)
    p_comp = np.clip(p_comp, 0.0, 1.0)
    alpha = np.ones(N) * TS_HOUR
    beta = (p_comp / load_weights) * 1.0  # 这里先算“比例项”，最终会乘每条线路tr_k
    return p_comp, alpha, beta, supply_ratio, der_ts, load_ts

def solve_one_scenario(
    scenario_name: str,
    branch_df: pd.DataFrame,
    load_power: np.ndarray,
    load_weights: np.ndarray,
    M_grid: np.ndarray,
    lambda_k: np.ndarray,
    tr_k: np.ndarray,
    alpha_i: np.ndarray,
    beta_i_base: np.ndarray,
    x0_init: np.ndarray,
    max_switch_actions: int = MAX_SWITCH_ACTIONS,
    root_node: int = ROOT_NODE,
    gurobi_output: int = GUROBI_OUTPUT
):
    """
    每个场景单独重求解：
    min Σ_i Σ_k M_logic[k,i] * lambda_k * (alpha_i + beta_i_base*tr_k) * load_i * weight_i
    s.t. SCF树约束 + 开关动作次数约束
    """
    n_nodes = len(load_power)
    n_edges = len(branch_df)
    U = n_nodes - 1 if BIGM_FLOW is None else BIGM_FLOW

    m = gp.Model(f"SCF_{scenario_name}")
    m.setParam("OutputFlag", gurobi_output)

    x = m.addVars(n_edges, vtype=GRB.BINARY, name="x")
    y = m.addVars(n_edges, vtype=GRB.BINARY, name="y_action")  # 与初始状态差异
    M_logic = m.addVars(n_edges, n_nodes, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="M_logic")

    # arcs
    arcs = []
    for k, r in branch_df.iterrows():
        u, v = int(r["from_node"]), int(r["to_node"])
        arcs.append((u, v, k))
        arcs.append((v, u, k))
    f = m.addVars(arcs, vtype=GRB.CONTINUOUS, lb=0.0, name="f")

    # --- 树约束 ---
    m.addConstr(gp.quicksum(x[k] for k in range(n_edges)) == n_nodes - 1, name="tree_edge_count")

    # --- SCF流平衡 ---
    for i in range(n_nodes):
        inflow = gp.quicksum(f[u,v,k] for (u,v,k) in arcs if v == i)
        outflow = gp.quicksum(f[u,v,k] for (u,v,k) in arcs if u == i)
        if i == root_node:
            m.addConstr(inflow - outflow == -(n_nodes - 1), name=f"flow_root_{i}")
        else:
            m.addConstr(inflow - outflow == 1, name=f"flow_{i}")

    # --- 流量-开关耦合 ---
    for k, r in branch_df.iterrows():
        u, v = int(r["from_node"]), int(r["to_node"])
        m.addConstr(f[u,v,k] <= U*x[k], name=f"cap_uv_{k}")
        m.addConstr(f[v,u,k] <= U*x[k], name=f"cap_vu_{k}")

    # --- M逻辑 ---
    for i in range(n_nodes):
        for k in range(n_edges):
            if M_grid[k, i] == 0:
                m.addConstr(M_logic[k, i] == 0, name=f"M0_{k}_{i}")
            else:
                m.addConstr(M_logic[k, i] == x[k], name=f"Mx_{k}_{i}")

    # --- 开关动作次数约束 ---
    # y_k >= |x_k - x0_k|
    for k in range(n_edges):
        m.addConstr(y[k] >= x[k] - x0_init[k], name=f"act_pos_{k}")
        m.addConstr(y[k] >= x0_init[k] - x[k], name=f"act_neg_{k}")
    m.addConstr(gp.quicksum(y[k] for k in range(n_edges)) <= max_switch_actions, name="max_switch_actions")

    # --- 目标 ---
    # 分线路tr_k进入节点恢复项：alpha_i + beta_i_base * tr_k
    obj = gp.quicksum(
        M_logic[k, i] * lambda_k[k] * (alpha_i[i] + beta_i_base[i] * tr_k[k]) * load_power[i] * load_weights[i]
        for i in range(1, n_nodes) for k in range(n_edges)
    )
    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()

    result = {
        "scenario": scenario_name,
        "status": int(m.status),
        "objective": None,
        "x": None,
        "y_actions": None
    }
    if m.status == GRB.OPTIMAL:
        x_sol = np.array([x[k].X for k in range(n_edges)])
        y_sol = np.array([y[k].X for k in range(n_edges)])
        result["objective"] = float(m.objVal)
        result["x"] = x_sol
        result["y_actions"] = y_sol
    return result

def save_switch_table(branch_df, x_sol, y_sol, x0, out_csv):
    rows = []
    for k, r in branch_df.iterrows():
        rows.append({
            "k_index": int(k),
            "branch_id": int(r["branch_id"]),
            "from_node": int(r["from_node"]),
            "to_node": int(r["to_node"]),
            "sw_type": int(r["sw_type"]),
            "x0_init": int(round(x0[k])),
            "x_opt": int(round(x_sol[k])),
            "action": int(round(y_sol[k])),
            "status": "closed" if x_sol[k] >= 0.5 else "open"
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False)

def plot_heatmap_node_resilience(p_comp, load_weights, out_png):
    score = (1.0 - p_comp) * load_weights
    plt.figure(figsize=(11, 3.5))
    img = plt.imshow(score.reshape(1, -1), cmap="YlGn", aspect="auto")
    plt.yticks([0], ["Resilience"])
    plt.xticks(np.arange(len(score)), [str(i) for i in range(len(score))], fontsize=8)
    plt.title("Node Resilience Heatmap")
    cbar = plt.colorbar(img)
    cbar.set_label("Score")
    for i, v in enumerate(score):
        plt.text(i, 0, f"{v:.1f}", ha="center", va="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

def plot_scenario_results_bar(df_res, out_png):
    plt.figure(figsize=(8,5))
    plt.bar(df_res["scenario"], df_res["objective"], color=["#e74c3c","#f39c12","#2ecc71","#3498db"][:len(df_res)])
    plt.ylabel("Weighted EENS (kWh/Year)")
    plt.title("Multi-Scenario Optimization Results (Re-solved MILP)")
    for i, v in enumerate(df_res["objective"]):
        plt.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

# =========================
# 3) 准备数据
# =========================
print("=== 数据准备 ===")
branch_df = deduplicate_branches(branch_data_raw)
num_edges = len(branch_df)
print(f"支路去重后: {num_edges}")

ensure_sample_csvs()
ts_df = load_profiles()
print(f"时序点数: {len(ts_df)}")

# 线路参数
length_km = approximate_branch_lengths(branch_df)
lambda_k, tr_k = build_failure_and_repair_params(branch_df, length_km)

# 基础M_grid
M_grid = generate_m_grid(num_nodes, branch_df, root=ROOT_NODE)

# 初始开关状态 x0：分段闭合1，联络断开0（典型初始）
x0_init = np.where(branch_df["sw_type"].to_numpy() == 0, 1, 0).astype(float)

# 场景定义：真实重求解
# 注意：每个场景都对应不同DER装机，进而得到不同P_comp/beta，最后重新建模求解
scenarios = {
    "Base_No_DER": {
        "pv_cap_kw": np.zeros(num_nodes),
        "wt_cap_kw": np.zeros(num_nodes),
    },
    "PV_Only": {
        "pv_cap_kw": np.zeros(num_nodes),
        "wt_cap_kw": np.zeros(num_nodes),
    },
    "WT_PV_Synergy": {
        "pv_cap_kw": np.zeros(num_nodes),
        "wt_cap_kw": np.zeros(num_nodes),
    }
}

# 你可按工程规划调整
scenarios["PV_Only"]["pv_cap_kw"][[23,24,25,29]] = [350,320,180,120]
scenarios["WT_PV_Synergy"]["pv_cap_kw"][[23,24,25,29]] = [350,320,180,120]
scenarios["WT_PV_Synergy"]["wt_cap_kw"][[23,24,25,29]] = [220,200,140,100]

# =========================
# 4) 多场景逐个求解
# =========================
print("=== 多场景优化求解 ===")
all_results = []
detail_dir = os.path.join(OUT_DIR, "scenarios")
os.makedirs(detail_dir, exist_ok=True)

for s_name, s_cfg in scenarios.items():
    print(f"\n--- 场景: {s_name} ---")
    pv_cap = s_cfg["pv_cap_kw"]
    wt_cap = s_cfg["wt_cap_kw"]

    p_comp, alpha_i, beta_i_base, supply_ratio, der_ts, load_ts = build_p_comp_from_timeseries(
        load_power=load_power,
        load_weights=load_weights,
        ts_df=ts_df,
        pv_cap_kw=pv_cap,
        wt_cap_kw=wt_cap
    )

    res = solve_one_scenario(
        scenario_name=s_name,
        branch_df=branch_df,
        load_power=load_power,
        load_weights=load_weights,
        M_grid=M_grid,
        lambda_k=lambda_k,
        tr_k=tr_k,
        alpha_i=alpha_i,
        beta_i_base=beta_i_base,
        x0_init=x0_init,
        max_switch_actions=MAX_SWITCH_ACTIONS,
        root_node=ROOT_NODE,
        gurobi_output=GUROBI_OUTPUT
    )

    if res["status"] == GRB.OPTIMAL:
        print(f"✅ {s_name} 最优目标值: {res['objective']:.4f}")
        n_actions = int(np.round(res["y_actions"]).sum())
        print(f"✅ 开关动作次数: {n_actions} / 限制 {MAX_SWITCH_ACTIONS}")

        # 导出该场景开关结果
        sw_csv = os.path.join(detail_dir, f"{s_name}_optimal_switches.csv")
        save_switch_table(branch_df, res["x"], res["y_actions"], x0_init, sw_csv)

        # 导出节点韧性热图（每场景）
        hm_png = os.path.join(detail_dir, f"{s_name}_node_resilience_heatmap.png")
        plot_heatmap_node_resilience(p_comp, load_weights, hm_png)

        # 导出时序供电比例
        sr_csv = os.path.join(detail_dir, f"{s_name}_node_supply_ratio_timeseries.csv")
        sr_df = pd.DataFrame(supply_ratio.T, columns=[f"node_{i}" for i in range(num_nodes)])
        sr_df.insert(0, "hour", ts_df["hour"].values)
        sr_df.to_csv(sr_csv, index=False)

        all_results.append({
            "scenario": s_name,
            "status": "OPTIMAL",
            "objective": res["objective"],
            "switch_actions": n_actions,
            "switch_csv": sw_csv,
            "heatmap_png": hm_png,
            "supply_ratio_csv": sr_csv
        })
    else:
        print(f"❌ {s_name} 未求得最优解, status={res['status']}")
        all_results.append({
            "scenario": s_name,
            "status": f"STATUS_{res['status']}",
            "objective": np.nan,
            "switch_actions": np.nan,
            "switch_csv": "",
            "heatmap_png": "",
            "supply_ratio_csv": ""
        })

# =========================
# 5) 总结果导出
# =========================
res_df = pd.DataFrame(all_results)
res_csv = os.path.join(OUT_DIR, "scenario_objectives.csv")
res_df.to_csv(res_csv, index=False)

# 仅对OPTIMAL画图
plot_df = res_df[res_df["status"] == "OPTIMAL"].copy()
if len(plot_df) > 0:
    bar_png = os.path.join(OUT_DIR, "scenario_comparison_real_resolve.png")
    plot_scenario_results_bar(plot_df[["scenario","objective"]], bar_png)
    print(f"\n📈 真实多场景重求解柱状图: {bar_png}")
else:
    bar_png = ""

# 汇总JSON
summary_json = os.path.join(OUT_DIR, "summary.json")
summary = {
    "max_switch_actions": MAX_SWITCH_ACTIONS,
    "num_nodes": num_nodes,
    "num_edges_after_dedup": int(num_edges),
    "result_csv": res_csv,
    "bar_chart_png": bar_png,
    "scenarios": all_results
}
with open(summary_json, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"🧾 场景结果CSV: {res_csv}")
print(f"🧾 总结JSON: {summary_json}")
print("=== 完成 ===")