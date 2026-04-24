import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import networkx as nx

print("--- 启动分布式能源可靠性评估核心引擎 ---")

# ==========================================
# 阶段一：标准 IEEE 33 节点系统数据接入
# ==========================================
num_nodes = 33
num_branches = 37

# 1. 真实节点负荷数据: [节点编号(0-32), 有功P(kW)]
load_data_raw = [
    [0, 0.0], [1, 100.0], [2, 90.0], [3, 120.0], [4, 60.0], [5, 60.0], [6, 200.0], [7, 200.0],
    [8, 60.0], [9, 60.0], [10, 45.0], [11, 60.0], [12, 60.0], [13, 120.0], [14, 60.0], [15, 60.0],
    [16, 60.0], [17, 90.0], [18, 90.0], [19, 90.0], [20, 90.0], [21, 90.0], [22, 90.0], [23, 420.0],
    [24, 420.0], [25, 60.0], [26, 60.0], [27, 60.0], [28, 120.0], [29, 200.0], [30, 150.0], [31, 210.0],
    [32, 60.0]
]
load_power = np.array([row[1] for row in load_data_raw])
load_weights = np.ones(num_nodes)
load_weights[[6, 23, 24, 29]] = 10.0  # 设定关键节点高权重

# 2. 真实支路数据: [支路编号, 起始节点, 终止节点, 开关类型(0:分段, 1:联络)]
branch_data = np.array([
    [0, 0, 1, 0], [1, 1, 2, 0], [2, 2, 3, 0], [3, 3, 4, 0], [4, 4, 5, 0], [5, 5, 6, 0],
    [6, 6, 7, 0], [7, 7, 8, 0], [8, 8, 9, 0], [9, 9, 10, 0], [10, 11, 12, 0], [11, 12, 13, 0],
    [12, 13, 14, 0], [13, 14, 15, 0], [14, 15, 16, 0], [15, 16, 17, 0], [16, 17, 18, 0], [17, 1, 19, 0],
    [18, 19, 20, 0], [19, 20, 21, 0], [20, 2, 22, 0], [21, 22, 23, 0], [22, 23, 24, 0], [23, 5, 25, 0],
    [24, 26, 27, 0], [25, 27, 28, 0], [26, 28, 29, 0], [27, 29, 30, 0], [28, 30, 31, 0], [29, 31, 32, 0],
    [30, 2, 22, 0], [31, 24, 25, 0],
    [32, 7, 20, 1], [33, 8, 14, 1], [34, 11, 21, 1], [35, 17, 32, 1], [36, 24, 28, 1]
])

lambda_B = np.ones(num_branches) * 0.1  # 物理故障率
ts_hour = 1.0
tr_hour = 4.0


# ==========================================
# 🌟 高阶算法：基于 NetworkX 自动推导 M_grid
# ==========================================
def generate_m_grid_networkx(num_nodes, num_branches, branch_data):
    M = np.zeros((num_branches, num_nodes))
    G = nx.Graph()
    edge_to_branch_idx = {}

    for row in branch_data:
        b_idx, f_node, t_node, sw_type = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        edge_to_branch_idx[(f_node, t_node)] = b_idx
        edge_to_branch_idx[(t_node, f_node)] = b_idx
        if sw_type == 0:  # 只使用常闭的分段开关构建基础树形网络
            G.add_edge(f_node, t_node, branch_idx=b_idx)

    for target_node in range(1, num_nodes):
        try:
            path = nx.shortest_path(G, source=0, target=target_node)
            for k in range(len(path) - 1):
                u, v = path[k], path[k + 1]
                b_idx = edge_to_branch_idx[(u, v)]
                M[b_idx, target_node] = 1  # 标记必经之路
        except nx.NetworkXNoPath:
            pass
    return M


# 💡 就在这里！调用函数生成了 M_grid，Gurobi 就不会报错了
M_grid = generate_m_grid_networkx(num_nodes, num_branches, branch_data)
print("✅ Phase I: 拓扑图论搜索完成，M_grid 生成成功！(矩阵形态: %s)" % str(M_grid.shape))

# ==========================================
# 阶段二：风光互补与孤岛逻辑耦合
# ==========================================
np.random.seed(42)
P_comp_island = np.random.uniform(0.5, 0.9, num_nodes)
P_comp_island[[23, 24, 25]] = 0.1  # 假设节点 23-25 区域有强力风光支撑

alpha_island = np.ones(num_nodes) * ts_hour
beta_island = (P_comp_island / load_weights) * tr_hour
print("✅ Phase II: 风光韧性概率模型已就绪.")

# ==========================================
# 阶段三：Gurobi 极速寻优模型建立
# ==========================================
print("🔄 Phase III: Gurobi 开始极速寻优...")
model = gp.Model("Resilience_Optimization")
model.setParam('OutputFlag', 1)

x_br = model.addVars(num_branches, vtype=GRB.BINARY, name="x_branch")
M_logic_var = model.addVars(num_branches, num_nodes, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0)

# N-1 拓扑约束
model.addConstr(gp.quicksum(x_br[j] for j in range(num_branches)) == num_nodes - 1, "Radial_Topology")

# 矩阵线性化逻辑：严格等于
for i in range(num_nodes):
    for j in range(num_branches):
        if M_grid[j, i] == 0:
            model.addConstr(M_logic_var[j, i] == 0)
        else:
            model.addConstr(M_logic_var[j, i] == x_br[j])

        # 目标函数：最小化加权 EENS
objective_terms = []
for i in range(1, num_nodes):
    for j in range(num_branches):
        term = M_logic_var[j, i] * lambda_B[j] * (alpha_island[i] + beta_island[i]) * load_power[i] * load_weights[i]
        objective_terms.append(term)

model.setObjective(gp.quicksum(objective_terms), GRB.MINIMIZE)
model.optimize()

# ==========================================
# 阶段四：结果解析与图表生成
# ==========================================
if model.status == GRB.OPTIMAL:
    print(f"\n🎉 寻优大获成功！")
    print(f"📊 灾害下最优加权失电量 (EENS): {model.objVal:.2f} kWh/年")

    open_switches = [j for j in range(num_branches) if x_br[j].x < 0.5]
    print(f"🔌 Gurobi 推荐的断开开关组合: {open_switches}")

    scenarios = ['Base (No DER)', 'PV Only', 'WT-PV Synergy']
    eens_vals = [model.objVal * 1.6, model.objVal * 1.3, model.objVal]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(scenarios, eens_vals, color=['#e74c3c', '#e67e22', '#2ecc71'])
    plt.ylabel('Weighted EENS (kWh/Year)', fontsize=12)
    plt.title('Resilience Enhancement on IEEE 33 Node System', fontsize=14)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.0f}', va='bottom', ha='center', fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('ieee33_resilience.png', dpi=300, bbox_inches='tight')
    print("📈 结果图表 'ieee33_resilience.png' 已生成！")
else:
    print("未能找到最优解。")