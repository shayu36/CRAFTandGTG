"""GTG 拓扑特征预处理编排 + 缓存。

对每个城市:
  road.csv --对偶图--> 空间句法(road级) + Metis分区池化 --长度加权--> region 级特征矩阵
产出 (缓存原始/未归一化, 归一化在装配区域图时按训练城市统计执行, 防泄漏):
  cache/gtg/{city}_gtg_region.npz   : region_feat(N,K), feat_names
  cache/gtg/{city}_gtg_meta.json    : 规模/覆盖/退化统计/参数
"""
import json
import os
import sys
from os.path import join

import numpy as np
import pandas as pd

from .dual_graph import build_dual_graph
from .space_syntax import compute_space_syntax
from .partition import metis_partition, add_partition_features
from .road_to_region import map_roads_to_regions

FEATURE_ORDER = [
    "connectivity", "total_depth", "integration", "choice", "mean_depth",
    "part_connectivity", "part_total_depth", "part_integration", "part_choice",
]


def _read_utm_epsg(city_dir):
    with open(join(city_dir, "data_feature.json")) as f:
        meta = json.load(f)
    epsg = meta.get("utm_epsg")
    if epsg is None:
        raise ValueError(f"严格模式: {city_dir}/data_feature.json 缺少 utm_epsg")
    return int(epsg)


def build_city(city, craft_root, cache_dir, local_size=50, verbose=True):
    city_dir = join(craft_root, city)
    road_pth = join(city_dir, "road.csv")
    region_pth = join(city_dir, "grid_region.csv")
    region_feat_pth = join(city_dir, "grid_region_feature.csv")
    for p in (road_pth, region_pth, region_feat_pth):
        if not os.path.exists(p):
            raise FileNotFoundError(f"严格模式: 缺失 {p}")

    utm_epsg = _read_utm_epsg(city_dir)
    road_df = pd.read_csv(road_pth)
    region_df = pd.read_csv(region_pth)
    num_regions = len(pd.read_csv(region_feat_pth))

    if verbose:
        print(f"[gtg] {city}: roads={len(road_df)} regions={num_regions} utm={utm_epsg}", flush=True)

    # 1) 对偶图
    dg = build_dual_graph(road_df, utm_epsg)

    # 2) 空间句法 (road 级)
    ss = compute_space_syntax(dg["num_nodes"], dg["edge_index"], dg["edge_length"], verbose=verbose)

    # 3) Metis 分区 + 分区级池化特征
    labels, num_clusters = metis_partition(dg["num_nodes"], dg["edge_index"], local_size=local_size)
    part_feat = add_partition_features(ss, labels)

    road_features = {
        "connectivity": ss["connectivity"],
        "total_depth": ss["total_depth"],
        "integration": ss["integration"],
        "choice": ss["choice"],
        "mean_depth": ss["mean_depth"],
        **part_feat,
    }
    # 校验特征顺序完整
    assert set(road_features.keys()) == set(FEATURE_ORDER), "特征集不一致"

    # 4) road -> region 长度加权映射
    region_feat, feat_names_raw, coverage = map_roads_to_regions(
        geom_utm=dg["geom_utm"],
        road_features={k: road_features[k] for k in FEATURE_ORDER},
        region_df=region_df,
        utm_epsg=utm_epsg,
        num_regions=num_regions,
    )
    feat_names = list(feat_names_raw)

    # NaN/Inf 严格检查
    if not np.all(np.isfinite(region_feat)):
        bad = int(np.sum(~np.isfinite(region_feat)))
        raise ValueError(f"严格模式: {city} region 特征含 {bad} 个 NaN/Inf")

    # 5) 缓存
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(
        join(cache_dir, f"{city}_gtg_region.npz"),
        region_feat=region_feat.astype(np.float32),
        feat_names=np.array(feat_names),
    )
    meta = {
        "city": city,
        "utm_epsg": utm_epsg,
        "num_roads": int(dg["num_nodes"]),
        "num_dual_edges": int(dg["edge_index"].shape[1]),
        "num_regions": int(num_regions),
        "num_clusters": int(num_clusters),
        "local_size": local_size,
        "feature_names": feat_names,
        "feature_dim": len(feat_names),
        "space_syntax_meta": ss["meta"],
        "coverage": coverage,
        "region_feat_mean": {n: float(region_feat[:, i].mean()) for i, n in enumerate(feat_names)},
        "region_feat_std": {n: float(region_feat[:, i].std()) for i, n in enumerate(feat_names)},
    }
    with open(join(cache_dir, f"{city}_gtg_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    if verbose:
        cov = coverage
        print(f"  -> unmapped_road={cov['unmapped_road_ratio']:.3f} "
              f"empty_region={cov['empty_region_ratio']:.3f} "
              f"degenerate_INT={ss['meta']['num_degenerate_integration']}", flush=True)
    return meta


def load_city_gtg(city, cache_dir):
    """加载缓存的原始 region 级 GTG 特征。返回 (region_feat[N,K], feat_names)。"""
    pth = join(cache_dir, f"{city}_gtg_region.npz")
    if not os.path.exists(pth):
        raise FileNotFoundError(
            f"严格模式: 缺失 GTG 缓存 {pth}, 请先运行 scripts/build_gtg_features.py"
        )
    data = np.load(pth, allow_pickle=True)
    return data["region_feat"].astype(np.float32), list(data["feat_names"])
