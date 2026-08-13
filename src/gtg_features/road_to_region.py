"""road 级拓扑/空间句法特征 → CRAFT region 级映射 (长度加权)。

- 道路 (LINESTRING, UTM) 与区域 (POLYGON, UTM) 做空间相交;
- 一条道路可能跨多个区域, 以“道路 ∩ 区域”的长度作为权重;
- 区域特征 = 该区域内所有道路段的长度加权均值 (空间句法指标为强度量, 用加权均值)。

严格模式: 记录未映射道路比例、空区域比例、空区域列表; 空区域比例超阈值报错。
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt


def map_roads_to_regions(
    geom_utm: gpd.GeoSeries,
    road_features: dict,
    region_df: pd.DataFrame,
    utm_epsg: int,
    num_regions: int,
    empty_region_error_ratio: float = 1.01,
):
    """
    geom_utm        : 投影后的道路 GeoSeries (与 road_features 行对齐)
    road_features   : {name: np.ndarray(num_roads,)}  road 级特征
    region_df       : grid_region.csv (region_id, geometry WKT POLYGON WGS84)
    num_regions     : 区域总数 (来自 grid_region_feature, 严格校验)
    返回 (region_feat[num_regions, K], feature_names, coverage_report)
    """
    if "region_id" not in region_df.columns or "geometry" not in region_df.columns:
        raise ValueError("严格模式: grid_region.csv 需含 region_id, geometry 列")
    region_ids = region_df["region_id"].to_numpy()
    if not np.array_equal(region_ids, np.arange(len(region_df))):
        raise ValueError("严格模式: grid_region.csv region_id 非 0..N-1 连续")
    if len(region_df) != num_regions:
        raise ValueError(
            f"严格模式: 区域数量不一致 grid_region={len(region_df)} vs feature={num_regions}"
        )

    region_geom = gpd.GeoSeries(region_df["geometry"].apply(wkt.loads), crs=4326).to_crs(utm_epsg)
    regions_gdf = gpd.GeoDataFrame({"region_id": region_ids}, geometry=region_geom, crs=utm_epsg)

    feat_names = list(road_features.keys())
    K = len(feat_names)
    feat_mat = np.stack([road_features[k] for k in feat_names], axis=1)  # (num_roads, K)

    roads_gdf = gpd.GeoDataFrame(
        {"road_idx": np.arange(len(geom_utm))}, geometry=geom_utm.values, crs=utm_epsg
    )

    # 空间连接: 道路 x 区域 (相交)
    joined = gpd.sjoin(roads_gdf, regions_gdf, predicate="intersects", how="inner")
    if len(joined) == 0:
        raise ValueError("严格模式: 无任何道路与区域相交, 坐标系或数据异常")

    # 逐对计算 道路∩区域 的长度作为权重
    left_geom = roads_gdf.geometry.values[joined["road_idx"].to_numpy()]
    right_geom = region_geom.values[joined["region_id"].to_numpy()]
    inter = gpd.GeoSeries(left_geom, crs=utm_epsg).intersection(
        gpd.GeoSeries(right_geom, crs=utm_epsg)
    )
    weight = inter.length.to_numpy()
    joined = joined.assign(weight=weight)
    joined = joined[joined["weight"] > 0.0]

    region_feat = np.zeros((num_regions, K), dtype=np.float64)
    weight_sum = np.zeros(num_regions, dtype=np.float64)
    road_idx = joined["road_idx"].to_numpy()
    reg_idx = joined["region_id"].to_numpy()
    w = joined["weight"].to_numpy()

    for r in range(num_regions):
        sel = reg_idx == r
        if not np.any(sel):
            continue
        ws = w[sel]
        fr = feat_mat[road_idx[sel]]  # (m, K)
        region_feat[r] = (fr * ws[:, None]).sum(axis=0) / ws.sum()
        weight_sum[r] = ws.sum()

    empty_regions = np.where(weight_sum == 0.0)[0].tolist()
    mapped_roads = np.unique(road_idx)
    num_roads = len(geom_utm)
    unmapped_ratio = 1.0 - len(mapped_roads) / num_roads
    empty_ratio = len(empty_regions) / num_regions

    coverage = {
        "num_roads": int(num_roads),
        "num_regions": int(num_regions),
        "num_mapped_roads": int(len(mapped_roads)),
        "unmapped_road_ratio": float(unmapped_ratio),
        "num_empty_regions": int(len(empty_regions)),
        "empty_region_ratio": float(empty_ratio),
        "empty_regions": empty_regions,
    }

    if empty_ratio > empty_region_error_ratio:
        raise ValueError(
            f"严格模式: 空区域比例 {empty_ratio:.3f} 超阈值 {empty_region_error_ratio}"
        )

    return region_feat, feat_names, coverage
