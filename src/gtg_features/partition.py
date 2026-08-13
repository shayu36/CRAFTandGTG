"""Metis 图分区。

忠实移植 GTG-main/dataloader.py: metis_cluster:
  - 无向化邻接 (两个方向都加, 否则 metis core dump)
  - num_clusters = num_nodes // local_size
  - pymetis.part_graph
分区标签用于产出 partition 级池化统计, 作为额外的 region 拓扑上下文特征。
"""
import numpy as np
import pymetis


def metis_partition(num_nodes, edge_index, local_size=50):
    """返回 (labels[num_nodes], num_clusters)。"""
    num_clusters = max(1, num_nodes // local_size)
    if num_clusters <= 1:
        return np.zeros(num_nodes, dtype=np.int64), 1

    adj_list = [[] for _ in range(num_nodes)]
    src = edge_index[0]
    trg = edge_index[1]
    for u, v in zip(src.tolist(), trg.tolist()):
        if v not in adj_list[u]:
            adj_list[u].append(v)
        if u not in adj_list[v]:
            adj_list[v].append(u)

    _, labels = pymetis.part_graph(num_clusters, adjacency=adj_list)
    return np.asarray(labels, dtype=np.int64), num_clusters


def add_partition_features(road_feat: dict, labels: np.ndarray):
    """按分区对 road 级空间句法做均值池化, 再广播回每条道路,
    得到 '所属分区的平均拓扑水平' 上下文特征 (前缀 part_)。"""
    out = {}
    keys = ["connectivity", "total_depth", "integration", "choice"]
    num_clusters = int(labels.max()) + 1
    for k in keys:
        vals = road_feat[k]
        part_mean = np.zeros(num_clusters, dtype=np.float64)
        counts = np.bincount(labels, minlength=num_clusters).astype(np.float64)
        sums = np.bincount(labels, weights=vals, minlength=num_clusters)
        nz = counts > 0
        part_mean[nz] = sums[nz] / counts[nz]
        out[f"part_{k}"] = part_mean[labels]
    return out
