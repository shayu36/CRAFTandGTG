"""Road→Spatial Syntax→CRAFT Region 三层静态图离线构建。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from gtg_features.dual_graph import build_dual_graph
from gtg_features.partition import metis_partition

from .contracts import CityStaticHierarchy, validate_city_static_hierarchy
from .operators import coalesce_edges, weighted_region_projection


CRAFT_FEATURE_ORDER = (
    ["population", "population_density", "dist_to_center", "road_num", "road_length"]
    + [f"poi_num_{k}" for k in range(12)]
    + [f"poi_score_{k}" for k in range(12)]
    + [f"road_num_{k}" for k in range(8)]
    + [f"road_length_{k}" for k in range(8)]
)
SYNTAX_FEATURE_ORDER = ["connectivity", "total_depth", "integration", "choice", "mean_depth"]
ROAD_TOPO_FEATURE_ORDER = ["bias", "in_degree", "out_degree", "total_degree"]


def _read_epsg(city_dir: Path) -> int:
    path = city_dir / "data_feature.json"
    if not path.exists():
        raise FileNotFoundError(f"严格模式: 缺失 {path}")
    meta = json.loads(path.read_text())
    if meta.get("utm_epsg") is None:
        raise ValueError(f"严格模式: {path} 缺少 utm_epsg")
    try:
        epsg = int(meta["utm_epsg"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"严格模式: {path} 的 utm_epsg 必须为整数") from exc
    if epsg <= 0:
        raise ValueError(f"严格模式: {path} 的 utm_epsg 必须为正")
    return epsg


def _load_region_inputs(city_dir: Path) -> tuple[np.ndarray, torch.Tensor, pd.DataFrame, int]:
    feature_path, region_path, relation_path = (
        city_dir / "grid_region_feature.csv", city_dir / "grid_region.csv", city_dir / "grid_region_rel.csv"
    )
    for path in (feature_path, region_path, relation_path):
        if not path.exists():
            raise FileNotFoundError(f"严格模式: 缺失 {path}")
    features = pd.read_csv(feature_path)
    regions = pd.read_csv(region_path)
    relations = pd.read_csv(relation_path)
    if features.empty or regions.empty:
        raise ValueError("严格模式: Region 文件不能为空")
    if "region_id" not in features or "region_id" not in regions:
        raise ValueError("严格模式: Region 文件缺少 region_id")
    def _strict_ids(frame: pd.DataFrame, name: str) -> np.ndarray:
        numeric = pd.to_numeric(frame[name], errors="coerce")
        values = numeric.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"严格模式: {name} 必须为有限整数 ID")
        result = values.astype(np.int64)
        if len(np.unique(result)) != len(result):
            raise ValueError(f"严格模式: {name} 存在重复 ID")
        return result

    feature_ids = _strict_ids(features, "region_id")
    region_ids = _strict_ids(regions, "region_id")
    if not np.array_equal(feature_ids, np.arange(len(feature_ids))) or not np.array_equal(region_ids, feature_ids):
        raise ValueError("严格模式: Region ID/顺序必须在两个文件中均为 0..N-1")
    missing = sorted(set(CRAFT_FEATURE_ORDER) - set(features.columns))
    if missing:
        raise ValueError(f"严格模式: CRAFT 45 维缺列 {missing}")
    region_x = features[CRAFT_FEATURE_ORDER].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    if not np.isfinite(region_x).all():
        raise ValueError("严格模式: region_x 含 NaN/Inf 或非数值")
    required_rel = {"ori", "des", "is_adj"}
    if required_rel - set(relations.columns):
        raise ValueError(f"严格模式: grid_region_rel.csv 缺列 {sorted(required_rel-set(relations.columns))}")
    is_adj = pd.to_numeric(relations["is_adj"], errors="coerce")
    if is_adj.isna().any():
        raise ValueError("严格模式: grid_region_rel.csv 的 is_adj 含非数值")
    rel_df = relations.loc[is_adj == 1, ["ori", "des"]]
    if rel_df.empty:
        rel = np.empty((2, 0), dtype=np.int64)
    else:
        rel_numeric = rel_df.apply(pd.to_numeric, errors="coerce")
        rel_values = rel_numeric.to_numpy(dtype=np.float64)
        if not np.isfinite(rel_values).all() or not np.equal(rel_values, np.floor(rel_values)).all():
            raise ValueError("严格模式: Region edge 的 ori/des 必须为有限整数")
        rel = rel_values.astype(np.int64).T
    if rel.size and (rel.min() < 0 or rel.max() >= len(feature_ids)):
        raise ValueError("严格模式: Region edge_index 越界")
    return region_x, torch.from_numpy(rel).long(), regions, len(feature_ids)


def _load_or_compute_syntax(
    city: str,
    road_path: Path,
    road_df: pd.DataFrame,
    dual: dict[str, Any],
    syntax_cache_dir: Path | None,
) -> tuple[np.ndarray, list[str]]:
    """优先复用已有 GTG Road cache；缺失时在离线进程中重算空间句法。"""

    cache = syntax_cache_dir / f"{city}_gtg_road.npz" if syntax_cache_dir is not None else None
    expected_ids = np.asarray([str(value) for value in road_df["road_id"]], dtype=str)
    if cache is not None and cache.exists():
        data = np.load(cache, allow_pickle=True)
        for key in ("road_feat", "road_id", "feat_names"):
            if key not in data:
                raise KeyError(f"严格模式: GTG Road cache 缺少 {key}: {cache}")
        cached_ids = np.asarray([str(value) for value in data["road_id"]], dtype=str)
        if not np.array_equal(cached_ids, expected_ids):
            raise ValueError(f"严格模式: GTG Road cache road_id/顺序与 {road_path} 不一致")
        names = [str(value) for value in data["feat_names"].tolist()]
        if names[:5] != SYNTAX_FEATURE_ORDER:
            raise ValueError("严格模式: GTG Road cache 前五个空间句法特征顺序错误")
        feat = np.asarray(data["road_feat"], dtype=np.float32)
        if feat.shape != (len(road_df), 9) or not np.isfinite(feat).all():
            raise ValueError("严格模式: GTG Road cache shape/finite 错误")
        return feat[:, :5], SYNTAX_FEATURE_ORDER.copy()
    # 只有离线 cache 缺失时才加载 graph_tool；Torch 训练/测试进程不会触发该导入。
    from gtg_features.space_syntax import compute_space_syntax
    ss = compute_space_syntax(dual["num_nodes"], dual["edge_index"], dual["edge_length"], verbose=False)
    syntax = np.stack([ss[name] for name in SYNTAX_FEATURE_ORDER], axis=1).astype(np.float32)
    if syntax.shape != (len(road_df), 5) or not np.isfinite(syntax).all():
        raise ValueError("严格模式: 重算空间句法特征 shape/finite 错误")
    return syntax, SYNTAX_FEATURE_ORDER.copy()


def _road_topology_features(edge_index: np.ndarray, num_roads: int) -> np.ndarray:
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("严格模式: Road edge_index 必须为 [2,E]")
    if num_roads <= 0:
        raise ValueError("严格模式: Road 节点数必须为正")
    src, dst = edge_index
    if edge_index.size and (edge_index.min() < 0 or edge_index.max() >= num_roads):
        raise ValueError("严格模式: Road edge_index 越界")
    in_degree = np.bincount(dst, minlength=num_roads).astype(np.float32)
    out_degree = np.bincount(src, minlength=num_roads).astype(np.float32)
    return np.stack([np.ones(num_roads, dtype=np.float32), in_degree, out_degree, in_degree + out_degree], axis=1)


def _road_intersection_matrix(geom_utm, regions: pd.DataFrame, epsg: int) -> np.ndarray:
    import geopandas as gpd
    from shapely import wkt

    region_geom = gpd.GeoSeries(regions["geometry"].map(wkt.loads), crs=4326).to_crs(epsg)
    if region_geom.is_empty.any() or (~region_geom.is_valid).any():
        raise ValueError("严格模式: Region 几何为空或无效")
    regions_gdf = gpd.GeoDataFrame({"region_id": regions["region_id"].to_numpy()}, geometry=region_geom, crs=epsg)
    roads_gdf = gpd.GeoDataFrame({"road_idx": np.arange(len(geom_utm))}, geometry=geom_utm.values, crs=epsg)
    joined = gpd.sjoin(roads_gdf, regions_gdf, how="inner", predicate="intersects")
    if joined.empty:
        # 保留全空映射，由调用方按 empty_region_error_ratio 判断是否允许。
        return np.zeros((len(geom_utm), len(regions)), dtype=np.float64)
    left = roads_gdf.geometry.iloc[joined["road_idx"].to_numpy()].reset_index(drop=True)
    right = region_geom.iloc[joined["region_id"].to_numpy()].reset_index(drop=True)
    lengths = left.intersection(right).length.to_numpy(dtype=np.float64)
    matrix = np.zeros((len(geom_utm), len(regions)), dtype=np.float64)
    for road, region, length in zip(joined["road_idx"].to_numpy(), joined["region_id"].to_numpy(), lengths):
        if length > 1e-8:
            matrix[int(road), int(region)] += float(length)
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("严格模式: Road×Region 相交长度含非法值")
    return matrix


def build_city_static_hierarchy(
    city: str,
    craft_root: str | Path,
    *,
    syntax_cache_dir: str | Path | None = None,
    local_size: int = 50,
    empty_region_error_ratio: float = 0.2,
) -> CityStaticHierarchy:
    """构建一个城市的三层静态图，所有排序均沿 CSV/Metis 输出固定。"""

    if not city:
        raise ValueError("严格模式: city 不能为空")
    if local_size <= 0:
        raise ValueError("严格模式: local_size 必须 > 0")
    if not 0 <= empty_region_error_ratio <= 1:
        raise ValueError("严格模式: empty_region_error_ratio 必须在 [0,1]")
    city_dir = Path(craft_root) / city
    if not city_dir.exists():
        raise FileNotFoundError(f"严格模式: 缺失城市目录 {city_dir}")
    road_path = city_dir / "gtg_road.csv"
    if not road_path.exists():
        road_path = city_dir / "road.csv"
    if not road_path.exists():
        raise FileNotFoundError(f"严格模式: 缺失 gtg_road.csv/road.csv: {city_dir}")
    road_df = pd.read_csv(road_path)
    required_road = {"road_id", "from_node_id", "to_node_id", "length", "geometry"}
    if required_road - set(road_df.columns):
        raise ValueError(f"严格模式: {road_path} 缺列 {sorted(required_road-set(road_df.columns))}")
    if road_df.empty or road_df["road_id"].isna().any():
        raise ValueError("严格模式: Road 文件为空或含缺失 road_id")
    for column in ("from_node_id", "to_node_id", "length"):
        numeric = pd.to_numeric(road_df[column], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise ValueError(f"严格模式: {road_path} 的 {column} 含非数值/NaN/Inf")
    if (pd.to_numeric(road_df["length"], errors="coerce") < 0).any():
        raise ValueError(f"严格模式: {road_path} 的 length 不能为负")
    road_ids = tuple(str(value) for value in road_df["road_id"].tolist())
    if any(not road_id.strip() for road_id in road_ids):
        raise ValueError("严格模式: Road ID 不能为空字符串")
    if len(set(road_ids)) != len(road_ids):
        raise ValueError("严格模式: Road ID 重复")
    region_x, region_edge_index, region_df, num_regions = _load_region_inputs(city_dir)
    epsg = _read_epsg(city_dir)
    dual = build_dual_graph(road_df, epsg)
    if not np.array_equal(np.asarray([str(v) for v in dual["road_id"]]), np.asarray(road_ids)):
        raise ValueError("严格模式: dual graph Road 顺序未保持 CSV 顺序")
    road_edge_index = torch.from_numpy(dual["edge_index"]).long()
    road_topo_x = torch.from_numpy(_road_topology_features(dual["edge_index"], len(road_df)))
    assignment_np, num_syntax = metis_partition(len(road_df), dual["edge_index"], local_size=local_size)
    assignment_np = np.asarray(assignment_np, dtype=np.int64)
    if assignment_np.shape != (len(road_df),) or assignment_np.min() < 0 or assignment_np.max() >= num_syntax:
        raise ValueError("严格模式: Metis Road→Syntax assignment 越界或 shape 错误")
    counts = np.bincount(assignment_np, minlength=num_syntax)
    if (counts == 0).any():
        raise ValueError("严格模式: Metis 产生空 Syntax 分区")
    road_syntax, _ = _load_or_compute_syntax(city, road_path, road_df, dual, Path(syntax_cache_dir) if syntax_cache_dir else None)
    syntax_x = np.stack([
        np.bincount(assignment_np, weights=road_syntax[:, col], minlength=num_syntax) / counts
        for col in range(road_syntax.shape[1])
    ], axis=1).astype(np.float32)
    if not np.isfinite(syntax_x).all():
        raise ValueError("严格模式: syntax_x 含 NaN/Inf")
    crossing_src, crossing_dst, crossing_weight = [], [], []
    for src, dst in zip(dual["edge_index"][0].tolist(), dual["edge_index"][1].tolist()):
        left, right = int(assignment_np[src]), int(assignment_np[dst])
        if left != right:
            crossing_src.append(left); crossing_dst.append(right); crossing_weight.append(1.0)
    syntax_edge_np, syntax_weight_np, _ = coalesce_edges(crossing_src, crossing_dst, crossing_weight)
    syntax_edge_index = torch.from_numpy(syntax_edge_np).long()
    # Syntax edge weights are metadata only; the model uses directed edge_index without a new loss.
    intersection = _road_intersection_matrix(dual["geom_utm"], region_df, epsg)
    sr_edge, sr_weight, sr_shape, region_has_syntax = weighted_region_projection(
        assignment_np, intersection, num_syntax, num_regions
    )
    empty_ratio = float((~region_has_syntax).sum().item()) / num_regions
    if empty_ratio > empty_region_error_ratio:
        raise ValueError(f"严格模式: 空 Region 映射比例 {empty_ratio:.3f} 超阈值 {empty_region_error_ratio}")
    rs_edge = torch.from_numpy(np.asarray([assignment_np, np.arange(len(road_df))], dtype=np.int64))
    rs_weight = torch.from_numpy((1.0 / counts[assignment_np]).astype(np.float32))
    metadata = {
        "feature_version": "three-layer-static-v1",
        "city": city,
        "num_regions": num_regions,
        "num_roads": len(road_df),
        "num_road_edges": int(road_edge_index.shape[1]),
        "num_syntax_nodes": num_syntax,
        "num_syntax_edges": int(syntax_edge_index.shape[1]),
        "num_road_to_syntax_links": len(road_df),
        "num_syntax_to_region_links": int(sr_edge.shape[1]),
        "road_topo_feature_names": ROAD_TOPO_FEATURE_ORDER,
        "syntax_feature_names": SYNTAX_FEATURE_ORDER,
        "region_feature_order": list(CRAFT_FEATURE_ORDER),
        "empty_region_ids": torch.where(~region_has_syntax)[0].tolist(),
        "empty_region_ratio": empty_ratio,
        "utm_epsg": epsg,
        "source_road_file": road_path.name,
        "local_size": int(local_size),
        "syntax_edge_weight": syntax_weight_np.astype(np.float32).tolist(),
    }
    result = CityStaticHierarchy(
        city_id=city,
        region_x=torch.from_numpy(region_x), region_edge_index=region_edge_index,
        road_topo_x=road_topo_x, road_edge_index=road_edge_index, road_ids=road_ids,
        syntax_x=torch.from_numpy(syntax_x), syntax_edge_index=syntax_edge_index,
        road_to_syntax_assignment=torch.from_numpy(assignment_np),
        road_to_syntax_edge_index=rs_edge, road_to_syntax_weight=rs_weight,
        road_to_syntax_shape=(num_syntax, len(road_df)),
        syntax_to_region_edge_index=sr_edge, syntax_to_region_weight=sr_weight,
        syntax_to_region_shape=sr_shape, region_has_syntax=region_has_syntax,
        metadata=metadata,
    )
    validate_city_static_hierarchy(result)
    return result
