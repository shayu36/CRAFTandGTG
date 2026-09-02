"""Road→Spatial Syntax→CRAFT Region 三层静态图离线构建。"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from gtg_features.dual_graph import build_dual_graph
from gtg_features.partition import metis_partition
from gtg_preprocessing.contracts import ROAD_TYPES, ROAD_TYPE_TO_ID

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
START_ROAD_FEATURE_ORDER = [
    *[f"road_type_{name}" for name in ROAD_TYPES],
    "length_log_minmax",
    "lanes_unknown", "lanes_1", "lanes_2", "lanes_3", "lanes_4", "lanes_5_plus",
    "maxspeed_unknown", "maxspeed_le_30", "maxspeed_31_50", "maxspeed_51_70",
    "maxspeed_71_90", "maxspeed_gt_90",
    "indegree_0", "indegree_1", "indegree_2", "indegree_3", "indegree_4", "indegree_5_plus",
    "outdegree_0", "outdegree_1", "outdegree_2", "outdegree_3", "outdegree_4", "outdegree_5_plus",
]
LANES_BUCKET_ORDER = ["unknown", "1", "2", "3", "4", "5_plus"]
MAXSPEED_BUCKET_ORDER = ["unknown", "le_30", "31_50", "51_70", "71_90", "gt_90"]
DEGREE_BUCKET_ORDER = ["0", "1", "2", "3", "4", "5_plus"]
_NUMERIC_PREFIX = re.compile(r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))")


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


def _numeric_prefix(value: object) -> float | None:
    """解析道路属性的严格数值前缀；缺失/无法解析返回 None。"""

    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "na", "n/a"}:
        return None
    match = _NUMERIC_PREFIX.match(text)
    if match is None:
        return None
    number = float(match.group(1))
    return number if np.isfinite(number) else None


def _one_hot_bucket(value: int | None, buckets: list[str], prefix: str) -> tuple[np.ndarray, str]:
    result = np.zeros(len(buckets), dtype=np.float32)
    if value is None:
        result[0] = 1.0
        return result, "unknown"
    if value >= 5:
        bucket = "5_plus"
    elif value in {0, 1, 2, 3, 4}:
        bucket = str(value)
    else:
        bucket = "unknown"
    result[buckets.index(bucket)] = 1.0
    return result, bucket


def _parse_lanes_bucket(value: object) -> str:
    number = _numeric_prefix(value)
    if number is None or number < 1 or not float(number).is_integer():
        return "unknown"
    integer = int(number)
    return "5_plus" if integer >= 5 else str(integer)


def _parse_maxspeed_kmh(value: object, unit: str) -> float | None:
    """将数值/带单位限速统一为 km/h；无法解析返回 None。"""

    if unit not in {"km/h", "m/s"}:
        raise ValueError("严格模式: maxspeed_unit 只能为 'km/h' 或 'm/s'")
    number = _numeric_prefix(value)
    if number is None or number <= 0:
        return None
    text = str(value).strip().lower()
    suffix = text[_NUMERIC_PREFIX.match(text).end():] if _NUMERIC_PREFIX.match(text) else ""
    if "mph" in suffix:
        return number * 1.609344
    if "m/s" in suffix or "mps" in suffix:
        return number * 3.6
    if "km/h" in suffix or "kmh" in suffix or "kph" in suffix:
        return number
    # 纯数字遵循数据契约中的单位；没有明确单位的其他文本不猜测。
    if not suffix.strip():
        return number if unit == "km/h" else number * 3.6
    return None


def build_start_static_road_features(
    road_df: pd.DataFrame,
    edge_index: np.ndarray,
    length_m: np.ndarray,
    *,
    maxspeed_unit: str = "km/h",
) -> tuple[np.ndarray, dict[str, object]]:
    """构造跨城市固定 33 维 START 风格静态 Road 特征。

    Road 节点仍是稳定排序的有向 Road segment；``edge_index`` 只用于计算
    原始有向对偶图的入度/出度，不能包含人为添加的 self-loop。
    """

    if maxspeed_unit not in {"km/h", "m/s"}:
        raise ValueError("严格模式: maxspeed_unit 只能为 'km/h' 或 'm/s'")
    n = len(road_df)
    if n <= 0:
        raise ValueError("严格模式: Road 文件不能为空")
    lengths = np.asarray(length_m, dtype=np.float64)
    if lengths.shape != (n,) or not np.isfinite(lengths).all() or (lengths <= 0).any():
        raise ValueError("严格模式: START Road length_m 必须为有限正值 [M]")
    edge_index = np.asarray(edge_index, dtype=np.int64)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("严格模式: START Road edge_index 必须为 [2,E]")
    if edge_index.size and (edge_index.min() < 0 or edge_index.max() >= n):
        raise ValueError("严格模式: START Road edge_index 越界")
    required = {"road_type_id", "lanes", "maxspeed"}
    missing = sorted(required - set(road_df.columns))
    if missing:
        raise ValueError(f"严格模式: START Road 文件缺少列 {missing}")

    type_values = pd.to_numeric(road_df["road_type_id"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(type_values).all() or not np.equal(type_values, np.floor(type_values)).all():
        raise ValueError("严格模式: road_type_id 必须为有限整数")
    type_ids = type_values.astype(np.int64)
    if len(ROAD_TYPES) != 8 or len(ROAD_TYPE_TO_ID) != 8:
        raise ValueError("严格模式: 项目 ROAD_TYPES 必须保持固定 8 类 schema")
    if (type_ids < 0).any() or (type_ids >= len(ROAD_TYPES)).any():
        raise ValueError("严格模式: road_type_id 越界，必须属于 0..7")
    type_one_hot = np.eye(len(ROAD_TYPES), dtype=np.float32)[type_ids]

    length_log = np.log1p(lengths)
    lo, hi = float(length_log.min()), float(length_log.max())
    if hi == lo:
        length_scaled = np.zeros(n, dtype=np.float32)
        warnings.warn("START Road 所有道路长度相同，length_log_minmax 全部为 0", RuntimeWarning)
        length_constant = True
    else:
        length_scaled = ((length_log - lo) / (hi - lo)).astype(np.float32)
        length_constant = False

    lanes_buckets = [_parse_lanes_bucket(value) for value in road_df["lanes"].tolist()]
    lanes_one_hot = np.zeros((n, 6), dtype=np.float32)
    for idx, bucket in enumerate(lanes_buckets):
        lanes_one_hot[idx, LANES_BUCKET_ORDER.index(bucket)] = 1.0

    speed_values = [_parse_maxspeed_kmh(value, maxspeed_unit) for value in road_df["maxspeed"].tolist()]
    speed_buckets = []
    for value in speed_values:
        if value is None:
            speed_buckets.append("unknown")
        elif value <= 30:
            speed_buckets.append("le_30")
        elif value <= 50:
            speed_buckets.append("31_50")
        elif value <= 70:
            speed_buckets.append("51_70")
        elif value <= 90:
            speed_buckets.append("71_90")
        else:
            speed_buckets.append("gt_90")
    speed_one_hot = np.zeros((n, 6), dtype=np.float32)
    for idx, bucket in enumerate(speed_buckets):
        speed_one_hot[idx, MAXSPEED_BUCKET_ORDER.index(bucket)] = 1.0

    out_degree = np.bincount(edge_index[0], minlength=n)
    in_degree = np.bincount(edge_index[1], minlength=n)
    degree_one_hot = np.zeros((n, 6), dtype=np.float32)
    out_degree_one_hot = np.zeros((n, 6), dtype=np.float32)
    for idx, degree in enumerate(in_degree.tolist()):
        degree_one_hot[idx, DEGREE_BUCKET_ORDER.index("5_plus" if degree >= 5 else str(degree))] = 1.0
    for idx, degree in enumerate(out_degree.tolist()):
        out_degree_one_hot[idx, DEGREE_BUCKET_ORDER.index("5_plus" if degree >= 5 else str(degree))] = 1.0

    road_x = np.concatenate(
        [type_one_hot, length_scaled[:, None], lanes_one_hot, speed_one_hot,
         degree_one_hot, out_degree_one_hot], axis=1
    ).astype(np.float32)
    if road_x.shape != (n, 33) or not np.isfinite(road_x).all():
        raise ValueError("严格模式: START Road 特征必须为有限 [M,33]")
    metadata = {
        "road_feature_names": list(START_ROAD_FEATURE_ORDER),
        "road_feature_dim": 33,
        "road_feature_mode": "start_static",
        "road_feature_rules": {
            "road_type": "ROAD_TYPES/ROAD_TYPE_TO_ID fixed 8-class one-hot",
            "length": "log1p(projected_length_m), city-local min-max [0,1]",
            "lanes": "numeric prefix; unknown/1/2/3/4/5_plus",
            "maxspeed": "km/h buckets; explicit mph/m/s suffix conversion; unitless values use maxspeed_unit",
            "degree": "directed dual-graph in/out degree before self-loops, 0/1/2/3/4/5_plus",
        },
        "road_type_order": list(ROAD_TYPES),
        "lanes_bucket_order": list(LANES_BUCKET_ORDER),
        "maxspeed_bucket_order": list(MAXSPEED_BUCKET_ORDER),
        "indegree_bucket_order": list(DEGREE_BUCKET_ORDER),
        "outdegree_bucket_order": list(DEGREE_BUCKET_ORDER),
        "maxspeed_unit": maxspeed_unit,
        "missing_lanes_count": int(sum(bucket == "unknown" for bucket in lanes_buckets)),
        "missing_lanes_ratio": float(sum(bucket == "unknown" for bucket in lanes_buckets) / n),
        "missing_maxspeed_count": int(sum(bucket == "unknown" for bucket in speed_buckets)),
        "missing_maxspeed_ratio": float(sum(bucket == "unknown" for bucket in speed_buckets) / n),
        "length_log_min": lo,
        "length_log_max": hi,
        "length_constant": length_constant,
    }
    return road_x, metadata


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
    road_feature_mode: str = "topology_only",
    maxspeed_unit: str = "km/h",
) -> CityStaticHierarchy:
    """构建一个城市的三层静态图，所有排序均沿 CSV/Metis 输出固定。"""

    if not city:
        raise ValueError("严格模式: city 不能为空")
    if local_size <= 0:
        raise ValueError("严格模式: local_size 必须 > 0")
    if not 0 <= empty_region_error_ratio <= 1:
        raise ValueError("严格模式: empty_region_error_ratio 必须在 [0,1]")
    if road_feature_mode == "cospec":
        raise NotImplementedError("CoSpec road features are not implemented in Stage 1")
    if road_feature_mode not in {"topology_only", "start_static"}:
        raise ValueError(f"未知 road_feature_mode={road_feature_mode!r}")
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
    if road_feature_mode == "topology_only":
        road_x_np = _road_topology_features(dual["edge_index"], len(road_df))
        road_feature_metadata = {
            "feature_version": "three-layer-static-v1",
            "road_topo_feature_names": ROAD_TOPO_FEATURE_ORDER,
            "road_feature_mode": "topology_only",
            "road_feature_dim": 4,
        }
    else:
        road_x_np, road_feature_metadata = build_start_static_road_features(
            road_df, dual["edge_index"], dual["length_m"], maxspeed_unit=maxspeed_unit
        )
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
        "feature_version": "three-layer-static-v1" if road_feature_mode == "topology_only" else "three-layer-start-road-v2",
        "city": city,
        "num_regions": num_regions,
        "num_roads": len(road_df),
        "num_road_edges": int(road_edge_index.shape[1]),
        "num_syntax_nodes": num_syntax,
        "num_syntax_edges": int(syntax_edge_index.shape[1]),
        "num_road_to_syntax_links": len(road_df),
        "num_syntax_to_region_links": int(sr_edge.shape[1]),
        "syntax_feature_names": SYNTAX_FEATURE_ORDER,
        "region_feature_order": list(CRAFT_FEATURE_ORDER),
        "empty_region_ids": torch.where(~region_has_syntax)[0].tolist(),
        "empty_region_ratio": empty_ratio,
        "utm_epsg": epsg,
        "source_road_file": road_path.name,
        "local_size": int(local_size),
        "syntax_edge_weight": syntax_weight_np.astype(np.float32).tolist(),
    }
    metadata.update(road_feature_metadata)
    result = CityStaticHierarchy(
        city_id=city,
        region_x=torch.from_numpy(region_x), region_edge_index=region_edge_index,
        road_x=torch.from_numpy(road_x_np), road_edge_index=road_edge_index, road_ids=road_ids,
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
