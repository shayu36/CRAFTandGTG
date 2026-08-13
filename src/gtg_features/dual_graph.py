"""道路对偶图构造。

忠实移植 GTG-main/prepare.py: gen_edge_data / calc_angle:
  - 节点 = 道路段 (link)
  - 有向边: 若 road_B.from_node_id == road_A.to_node_id, 则 A -> B
  - 边属性: length = (len_A+len_B)/2, dist = 形心距(投影CRS), angle = 端点方位角差
与 GTG 的差异: GTG 原实现对每行做 DataFrame 过滤 (O(N^2)), 这里用 groupby 建邻接字典向量化,
结果等价但可在 3~5 万条道路上高效运行。
"""
import math

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import LineString


def calc_angle(line: LineString) -> float:
    """端点方位角 (移植自 GTG prepare.calc_angle)。"""
    start, end = line.coords[0], line.coords[-1]
    return math.atan2(end[1] - start[1], end[0] - start[0])


def build_dual_graph(road_df: pd.DataFrame, utm_epsg: int):
    """构造对偶图。

    road_df 需含: from_node_id, to_node_id, length, geometry(WKT LINESTRING, WGS84)。
    道路按行顺序重编号为 link_id = 0..N-1 (GTG 同款做法)。

    返回 dict:
      num_nodes            道路数 N
      edge_index           (2, E) int64, 有向 A->B
      edge_length          (E,)  (len_A+len_B)/2   [米]
      edge_dist            (E,)  形心距            [米]
      edge_angle           (E,)  angle_B - angle_A [弧度]
      road_id              (N,)  原始 road_id
      angles               (N,)  每条道路方位角
      length_m             (N,)  投影后道路长度 [米]
      geom_utm             GeoSeries (投影后, 供 road->region 映射复用)
    """
    required = {"from_node_id", "to_node_id", "length", "geometry"}
    missing = required - set(road_df.columns)
    if missing:
        raise ValueError(f"严格模式: road.csv 缺少列 {missing}")

    road_df = road_df.reset_index(drop=True).copy()
    n = len(road_df)
    road_df["link_id"] = np.arange(n)

    # 几何: WGS84 -> UTM 投影 (度量计算)
    geom_wgs = gpd.GeoSeries(road_df["geometry"].apply(wkt.loads), crs=4326)
    geom_utm = geom_wgs.to_crs(utm_epsg)
    if geom_utm.is_empty.any() or (~geom_utm.is_valid).any():
        bad = int(geom_utm.is_empty.sum() + (~geom_utm.is_valid).sum())
        raise ValueError(f"严格模式: {bad} 条道路几何为空/无效")

    angles = np.array([calc_angle(g) for g in geom_utm.values], dtype=np.float64)
    length_m = geom_utm.length.to_numpy()
    centroids = geom_utm.centroid
    cx = centroids.x.to_numpy()
    cy = centroids.y.to_numpy()
    len_arr = road_df["length"].to_numpy(dtype=np.float64)

    # 邻接: from_node_id -> [link_id...]
    from_group = {}
    for lid, fnode in zip(road_df["link_id"].to_numpy(), road_df["from_node_id"].to_numpy()):
        from_group.setdefault(fnode, []).append(int(lid))

    src_list, trg_list = [], []
    for lid, tnode in zip(road_df["link_id"].to_numpy(), road_df["to_node_id"].to_numpy()):
        for nb in from_group.get(tnode, ()):
            if nb == lid:
                continue  # 去自环
            src_list.append(int(lid))
            trg_list.append(nb)

    if len(src_list) == 0:
        raise ValueError("严格模式: 对偶图无边, 道路拓扑异常")

    src = np.asarray(src_list, dtype=np.int64)
    trg = np.asarray(trg_list, dtype=np.int64)
    edge_length = (len_arr[src] + len_arr[trg]) / 2.0
    edge_dist = np.sqrt((cx[src] - cx[trg]) ** 2 + (cy[src] - cy[trg]) ** 2)
    edge_angle = angles[trg] - angles[src]

    return {
        "num_nodes": n,
        "edge_index": np.stack([src, trg], axis=0),
        "edge_length": edge_length,
        "edge_dist": edge_dist,
        "edge_angle": edge_angle,
        "road_id": road_df["road_id"].to_numpy() if "road_id" in road_df.columns else np.arange(n),
        "angles": angles,
        "length_m": length_m,
        "geom_utm": geom_utm,
    }
