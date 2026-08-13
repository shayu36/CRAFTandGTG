"""对偶图与空间句法单元测试 (合成小路网, 结果可手工核对)。"""
import numpy as np
import pandas as pd

from gtg_features.dual_graph import build_dual_graph, calc_angle
from gtg_features.space_syntax import compute_space_syntax
from gtg_features.partition import metis_partition
from shapely.geometry import LineString


def _toy_roads():
    # 路径 A->B->C->D, 四个节点三条边, 经度递增 (芝加哥附近, utm 32616)
    rows = [
        {"road_id": 0, "from_node_id": "A", "to_node_id": "B", "length": 100.0,
         "geometry": "LINESTRING (-87.60 41.90, -87.59 41.90)"},
        {"road_id": 1, "from_node_id": "B", "to_node_id": "C", "length": 100.0,
         "geometry": "LINESTRING (-87.59 41.90, -87.58 41.90)"},
        {"road_id": 2, "from_node_id": "C", "to_node_id": "D", "length": 100.0,
         "geometry": "LINESTRING (-87.58 41.90, -87.57 41.90)"},
    ]
    return pd.DataFrame(rows)


def test_calc_angle_horizontal():
    line = LineString([(0, 0), (1, 0)])
    assert abs(calc_angle(line) - 0.0) < 1e-9


def test_dual_graph_edges():
    dg = build_dual_graph(_toy_roads(), utm_epsg=32616)
    assert dg["num_nodes"] == 3
    edges = set(map(tuple, dg["edge_index"].T.tolist()))
    # road0.to==B==road1.from -> 0->1 ; road1.to==C==road2.from -> 1->2
    assert edges == {(0, 1), (1, 2)}
    assert dg["edge_length"].shape == (2,)
    assert np.all(dg["edge_dist"] > 0)


def test_space_syntax_connectivity():
    dg = build_dual_graph(_toy_roads(), utm_epsg=32616)
    ss = compute_space_syntax(dg["num_nodes"], dg["edge_index"], dg["edge_length"], verbose=False)
    # 无向度: 节点0->1 度1, 节点1<->0,2 度2, 节点2->1 度1
    assert list(ss["connectivity"]) == [1.0, 2.0, 1.0]
    # 全部有限
    for k in ["total_depth", "integration", "choice", "mean_depth"]:
        assert np.all(np.isfinite(ss[k]))
    # 中间节点介数最高 (Choice)
    assert ss["choice"][1] >= ss["choice"][0]


def test_metis_partition_small():
    dg = build_dual_graph(_toy_roads(), utm_epsg=32616)
    labels, k = metis_partition(dg["num_nodes"], dg["edge_index"], local_size=50)
    # 3 节点 / local_size=50 -> num_clusters=0 -> 回退单簇
    assert k == 1
    assert labels.shape == (3,)
