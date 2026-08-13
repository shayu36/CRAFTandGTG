"""空间句法四项核心指标 (在道路对偶图上计算)。

- Connectivity : 对偶图无向度 (相连道路段数)
- Total Depth  : 节点到所有可达节点的无向拓扑最短路径距离之和
- Integration  : 由平均深度导出的整合度 (1/RA, RA=2(MD-1)/(n-2))
- Choice       : betweenness 介数中心性 (= GTG gen_edge_data 的 bet, 长度加权, 有向)

使用 graph_tool (实机可用, C++ 实现) 计算 BFS 深度与介数。

严格模式: 退化节点 (可达节点数 n<=2, Integration 无定义) 计数并在元数据中显式报告,
Integration 置 0 但**不静默** (计数记录于 metadata['num_degenerate_integration'])。
"""
import numpy as np

import graph_tool.all as gt

_UNREACH = 2147483647  # graph_tool 对不可达节点的整型标记


def _build_gt_graph(num_nodes, edge_index, edge_length):
    g = gt.Graph(directed=True)
    g.add_vertex(num_nodes)
    wt = g.new_edge_property("double")
    edges = np.stack([edge_index[0], edge_index[1], edge_length], axis=1)
    eprops = g.add_edge_list(edges, eprops=[wt])  # 第三列写入 wt
    g.ep["wt"] = wt
    return g


def compute_space_syntax(num_nodes, edge_index, edge_length, verbose=True):
    """返回 dict: connectivity, total_depth, integration, choice, mean_depth,
    以及 meta(退化计数/分量信息)。所有向量长度 = num_nodes。"""
    g = _build_gt_graph(num_nodes, edge_index, edge_length)

    # --- Choice = 介数中心性 (有向, 长度加权; 与 GTG 一致) ---
    vp_bet, _ = gt.betweenness(g, weight=g.ep["wt"])
    choice = vp_bet.a.copy().astype(np.float64)

    # --- 无向视图: Connectivity / Total Depth / Integration ---
    g.set_directed(False)
    connectivity = g.get_total_degrees(np.arange(num_nodes)).astype(np.float64)

    total_depth = np.zeros(num_nodes, dtype=np.float64)
    reach_count = np.zeros(num_nodes, dtype=np.int64)  # 含自身
    for v in range(num_nodes):
        d = gt.shortest_distance(g, source=g.vertex(v)).a  # 无权 BFS 步数
        mask = d < _UNREACH
        total_depth[v] = float(d[mask].sum())
        reach_count[v] = int(mask.sum())
        if verbose and num_nodes >= 20000 and (v % 10000 == 0):
            print(f"    [space_syntax] depth {v}/{num_nodes}", flush=True)

    # Integration: MD = TD/(n-1); RA = 2(MD-1)/(n-2); INT = 1/RA
    n = reach_count.astype(np.float64)  # 可达节点数(含自身)
    mean_depth = np.full(num_nodes, np.nan, dtype=np.float64)
    integration = np.zeros(num_nodes, dtype=np.float64)
    degenerate = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        valid = n > 2
        md = np.where(n > 1, total_depth / (n - 1), np.nan)
        mean_depth[:] = md
        ra = np.where(valid, 2.0 * (md - 1.0) / (n - 2.0), np.nan)
        # RA<=0 (完全整合的极小结构) 或非有限 → 退化
        good = valid & np.isfinite(ra) & (ra > 0)
        integration[good] = 1.0 / ra[good]
        degenerate = int((~good).sum())

    meta = {
        "num_degenerate_integration": degenerate,
        "num_isolated_nodes": int((connectivity == 0).sum()),
        "reach_count_min": int(reach_count.min()),
        "reach_count_max": int(reach_count.max()),
        "reach_count_mean": float(reach_count.mean()),
    }
    return {
        "connectivity": connectivity,
        "total_depth": total_depth,
        "integration": integration,
        "choice": choice,
        "mean_depth": np.nan_to_num(mean_depth, nan=0.0),
        "meta": meta,
    }
