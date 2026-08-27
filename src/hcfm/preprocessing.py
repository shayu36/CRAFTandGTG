"""Stage A：从真实 CRAFT 静态数据构建 HCFM 层次图缓存。"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import pandas as pd
import torch

from .hierarchy import build_road_edge_index, sparse_coo


CRAFT_FEATURE_ORDER = (
    ["population", "population_density", "dist_to_center", "road_num", "road_length"]
    + [f"poi_num_{k}" for k in range(12)]
    + [f"poi_score_{k}" for k in range(12)]
    + [f"road_num_{k}" for k in range(8)]
    + [f"road_length_{k}" for k in range(8)]
)


def _numeric_prefix(value: Any, missing: float = -1.0) -> float:
    if pd.isna(value):
        return missing
    if isinstance(value, (int, float, np.number)):
        return float(value)
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value))
    return float(match.group()) if match else missing


def _as_bool(series: pd.Series) -> np.ndarray:
    values = series.astype(str).str.strip().str.lower()
    valid = {"true", "false", "1", "0", "yes", "no"}
    bad = ~values.isin(valid)
    if bad.any():
        raise ValueError(f"严格模式: oneway 含无法解释的值 {sorted(values[bad].unique())}")
    return values.isin({"true", "1", "yes"}).to_numpy()


def load_road_gtg_cache(path: str | Path, source_road_ids: np.ndarray) -> tuple[np.ndarray, list[str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"严格模式: 缺少 Road 级 GTG 空间句法缓存 {path}；请重新运行 "
            "scripts/build_gtg_features.py。禁止用 Region 级 9 维缓存冒充 Road 节点特征。"
        )
    data = np.load(path, allow_pickle=True)
    road_feat = np.asarray(data["road_feat"], dtype=np.float32)
    cached_ids = np.asarray(data["road_id"])
    if not np.array_equal(cached_ids.astype(str), source_road_ids.astype(str)):
        raise ValueError("严格模式: GTG Road 缓存 road_id/顺序与 CRAFT road.csv 不一致")
    if road_feat.shape[0] != len(source_road_ids) or not np.isfinite(road_feat).all():
        raise ValueError("严格模式: GTG Road 缓存行数错误或含 NaN/Inf")
    return road_feat, [str(x) for x in data["feat_names"].tolist()]


def _serialize_sparse(matrix: torch.Tensor, prefix: str, output: Dict[str, np.ndarray]) -> None:
    matrix = matrix.coalesce().cpu()
    output[f"{prefix}_indices"] = matrix.indices().numpy()
    output[f"{prefix}_values"] = matrix.values().numpy()
    output[f"{prefix}_shape"] = np.asarray(matrix.shape, dtype=np.int64)


def _deserialize_sparse(data: Mapping[str, np.ndarray], prefix: str) -> torch.Tensor:
    shape = tuple(int(x) for x in data[f"{prefix}_shape"])
    return torch.sparse_coo_tensor(
        torch.from_numpy(data[f"{prefix}_indices"]).long(),
        torch.from_numpy(data[f"{prefix}_values"]).float(),
        shape,
    ).coalesce()


def build_city_static_graph(
    city: str,
    craft_root: str | Path,
    gtg_cache_dir: str | Path,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """构建一个城市的 Region 图、逻辑有向 Road 图、P/B 稀疏算子。

    返回的 tensor dict 可写入 Paper cache；所有源目录只读。道路没有真实
    ``osm_way_id`` 时保留 ``parent_source_road_id`` 并将 OSM id 记为 null。
    """

    # 重依赖仅在离线预处理进程导入，避免与 torch 训练进程中的系统库冲突。
    import geopandas as gpd
    from shapely import wkt
    from shapely.geometry import Point

    city_dir = Path(craft_root) / city
    required = ["grid_region.csv", "grid_region_feature.csv", "grid_region_rel.csv", "road.csv", "data_feature.json"]
    for name in required:
        if not (city_dir / name).exists():
            raise FileNotFoundError(f"严格模式: 缺少 {city_dir / name}")
    region_df = pd.read_csv(city_dir / "grid_region.csv")
    region_feat_df = pd.read_csv(city_dir / "grid_region_feature.csv")
    rel_df = pd.read_csv(city_dir / "grid_region_rel.csv")
    road_df = pd.read_csv(city_dir / "road.csv")
    with open(city_dir / "data_feature.json") as handle:
        city_meta = json.load(handle)
    utm_epsg = int(city_meta["utm_epsg"])

    region_ids = region_feat_df["region_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(region_ids, np.arange(len(region_ids))):
        raise ValueError("严格模式: Region 顺序不是 0..N-1")
    if not np.array_equal(region_df["region_id"].to_numpy(dtype=np.int64), region_ids):
        raise ValueError("严格模式: grid_region 与 grid_region_feature Region 顺序不一致")
    missing_features = set(CRAFT_FEATURE_ORDER) - set(region_feat_df.columns)
    if missing_features:
        raise ValueError(f"严格模式: CRAFT 45 维缺列 {sorted(missing_features)}")
    region_x = region_feat_df[CRAFT_FEATURE_ORDER].to_numpy(dtype=np.float32)
    if not np.isfinite(region_x).all():
        raise ValueError("严格模式: CRAFT Region 45 维含 NaN/Inf")
    region_edge_index = rel_df.loc[rel_df["is_adj"] == 1, ["ori", "des"]].to_numpy(dtype=np.int64).T

    required_road = {"road_id", "from_node_id", "to_node_id", "road_type_id", "length", "geometry", "oneway"}
    if required_road - set(road_df.columns):
        raise ValueError(f"严格模式: road.csv 缺列 {sorted(required_road - set(road_df.columns))}")
    source_road_ids = road_df["road_id"].to_numpy()
    gtg_base, gtg_names = load_road_gtg_cache(Path(gtg_cache_dir) / f"{city}_gtg_road.npz", source_road_ids)
    oneway = _as_bool(road_df["oneway"])

    geom_wgs = gpd.GeoSeries(road_df["geometry"].map(wkt.loads), crs=4326)
    geom_utm = geom_wgs.to_crs(utm_epsg)
    if geom_utm.is_empty.any() or (~geom_utm.is_valid).any():
        raise ValueError("严格模式: Road 几何为空或无效")
    region_geom = gpd.GeoSeries(region_df["geometry"].map(wkt.loads), crs=4326).to_crs(utm_epsg)
    regions = gpd.GeoDataFrame({"region_id": region_ids}, geometry=region_geom, crs=utm_epsg)
    roads = gpd.GeoDataFrame({"base_idx": np.arange(len(road_df))}, geometry=geom_utm, crs=utm_epsg)

    joined = gpd.sjoin(roads, regions, how="inner", predicate="intersects")
    if joined.empty:
        raise ValueError("严格模式: Road 与 Region 无相交，检查 CRS")
    left = roads.geometry.iloc[joined["base_idx"].to_numpy()].reset_index(drop=True)
    right = region_geom.iloc[joined["region_id"].to_numpy()].reset_index(drop=True)
    inter_len = left.intersection(right).length.to_numpy(dtype=np.float64)
    joined = joined.reset_index(drop=True).assign(intersection_length=inter_len)
    joined = joined[joined["intersection_length"] > 1e-8]

    # 每个 source road 展开为一个 forward，以及非单行道的 reverse。
    base_for_directed, direction, source_nodes, target_nodes, directed_ids = [], [], [], [], []
    for idx, row in road_df.iterrows():
        rid = str(row["road_id"])
        base_for_directed.append(idx); direction.append(1.0)
        source_nodes.append(row["from_node_id"]); target_nodes.append(row["to_node_id"])
        directed_ids.append(f"{rid}:f")
        if not oneway[idx]:
            base_for_directed.append(idx); direction.append(-1.0)
            source_nodes.append(row["to_node_id"]); target_nodes.append(row["from_node_id"])
            directed_ids.append(f"{rid}:r")
    base_for_directed = np.asarray(base_for_directed, dtype=np.int64)
    direction = np.asarray(direction, dtype=np.float32)
    m, n = len(base_for_directed), len(region_ids)

    base_static = np.stack([
        road_df["road_type_id"].to_numpy(dtype=np.float32),
        road_df["length"].to_numpy(dtype=np.float32),
        oneway.astype(np.float32),
        road_df.get("lanes", pd.Series([-1] * len(road_df))).map(_numeric_prefix).to_numpy(dtype=np.float32),
        road_df.get("maxspeed", pd.Series([-1] * len(road_df))).map(_numeric_prefix).to_numpy(dtype=np.float32),
    ], axis=1)
    road_x = np.concatenate([base_static[base_for_directed], direction[:, None], gtg_base[base_for_directed]], axis=1)
    if not np.isfinite(road_x).all():
        raise ValueError("严格模式: Road 静态特征含 NaN/Inf")
    road_edge_index = build_road_edge_index(source_nodes, target_nodes)

    # P_struct: source road 的每个相交长度复制给其方向节点，再按 Region 行归一化。
    directed_by_base: dict[int, list[int]] = {}
    for didx, bidx in enumerate(base_for_directed.tolist()):
        directed_by_base.setdefault(bidx, []).append(didx)
    p_rows, p_cols, p_raw = [], [], []
    membership: dict[int, dict[int, float]] = {}
    for row in joined.itertuples():
        bidx, rid, length = int(row.base_idx), int(row.region_id), float(row.intersection_length)
        membership.setdefault(bidx, {})[rid] = membership.setdefault(bidx, {}).get(rid, 0.0) + length
        for didx in directed_by_base[bidx]:
            p_rows.append(rid); p_cols.append(didx); p_raw.append(length)
    denom = np.bincount(p_rows, weights=p_raw, minlength=n)
    p_values = [value / denom[row] for row, value in zip(p_rows, p_raw)]
    p_struct = sparse_coo(p_rows, p_cols, p_values, (n, m))

    # 端点归属：点在共享边界时，从该 road 的实际相交 Region 中选择相交长度最大的候选。
    region_index = regions.sindex

    def locate(base_idx: int, point: Point) -> int:
        # 空间索引先筛到局部 Region；predicate=intersects 包含边界点。共享边界仍按
        # 该 road 在各候选 Region 中的真实相交长度确定，保持原严格规则。
        candidates = [int(rid) for rid in region_index.query(point, predicate="intersects")]
        if not candidates:
            return -1
        lengths = membership.get(base_idx, {})
        return max(candidates, key=lambda rid: (lengths.get(rid, 0.0), -rid))

    base_start, base_end = [], []
    for idx, geom in enumerate(geom_utm):
        base_start.append(locate(idx, Point(geom.coords[0])))
        base_end.append(locate(idx, Point(geom.coords[-1])))
    base_sequences = []
    for bidx, geom in enumerate(geom_utm):
        ordered = []
        for rid in membership.get(bidx, {}):
            intersection = geom.intersection(region_geom.iloc[rid])
            parts = list(intersection.geoms) if hasattr(intersection, "geoms") else [intersection]
            for part in parts:
                if part.is_empty or part.length <= 1e-8:
                    continue
                midpoint = part.interpolate(0.5, normalized=True)
                ordered.append((float(geom.project(midpoint)), int(rid)))
        sequence = []
        for _, rid in sorted(ordered):
            if not sequence or sequence[-1] != rid:
                sequence.append(rid)
        if not sequence:
            sequence = [base_start[bidx], base_end[bidx]]
        if sequence[0] != base_start[bidx]:
            sequence.insert(0, base_start[bidx])
        if sequence[-1] != base_end[bidx]:
            sequence.append(base_end[bidx])
        base_sequences.append(sequence)

    start_region, end_region, directed_sequences = [], [], []
    for bidx, sign in zip(base_for_directed, direction):
        if sign > 0:
            start_region.append(base_start[bidx]); end_region.append(base_end[bidx])
            directed_sequences.append(base_sequences[bidx])
        else:
            start_region.append(base_end[bidx]); end_region.append(base_start[bidx])
            directed_sequences.append(list(reversed(base_sequences[bidx])))

    from .hierarchy import build_boundary_operators_from_sequences
    b_in, b_out = build_boundary_operators_from_sequences(directed_sequences, n)
    p_row_sum = torch.sparse.sum(p_struct, dim=1).to_dense().numpy()
    empty_regions = np.flatnonzero(p_row_sum == 0).tolist()
    crossing = int(sum(len(set(sequence) - {-1}) > 1 for sequence in directed_sequences))
    transitions = int(sum(
        source != target
        for sequence in directed_sequences
        for source, target in zip(sequence[:-1], sequence[1:])
    ))
    tensors = {
        "region_x": torch.from_numpy(region_x),
        "region_edge_index": torch.from_numpy(region_edge_index).long(),
        "road_x": torch.from_numpy(road_x.astype(np.float32)),
        "road_edge_index": torch.from_numpy(road_edge_index).long(),
        "p_struct": p_struct,
        "b_in": b_in,
        "b_out": b_out,
    }
    manifest = {
        "data_version": "hcfm-static-v1",
        "city_id": city,
        "region_crs": "EPSG:4326",
        "road_crs": "EPSG:4326",
        "metric_crs": f"EPSG:{utm_epsg}",
        "num_regions": n,
        "num_directed_roads": m,
        "num_region_edges": int(region_edge_index.shape[1]),
        "num_road_edges": int(road_edge_index.shape[1]),
        "region_feature_order": list(CRAFT_FEATURE_ORDER),
        "road_feature_order": ["road_type_id", "length_m", "oneway", "lanes", "maxspeed", "direction"] + gtg_names,
        "directed_road_ids": directed_ids,
        "parent_source_road_ids": [str(source_road_ids[i]) for i in base_for_directed],
        "parent_osm_way_ids": None,
        "empty_regions": empty_regions,
        "empty_region_ratio": len(empty_regions) / n,
        "cross_region_directed_roads": crossing,
        "cross_region_ratio": crossing / m,
        "boundary_transition_count": transitions,
        "outside_start_count": int(sum(r < 0 for r in start_region)),
        "outside_end_count": int(sum(r < 0 for r in end_region)),
        "p_struct_nnz": p_struct._nnz(),
        "b_in_nnz": b_in._nnz(),
        "b_out_nnz": b_out._nnz(),
    }
    return tensors, manifest


def save_static_cache(
    tensors: Mapping[str, torch.Tensor], manifest: Mapping[str, Any], output_dir: str | Path
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    city = str(manifest["city_id"])
    arrays: Dict[str, np.ndarray] = {}
    for key in ("region_x", "region_edge_index", "road_x", "road_edge_index"):
        arrays[key] = tensors[key].cpu().numpy()
    for key in ("p_struct", "b_in", "b_out"):
        _serialize_sparse(tensors[key], key, arrays)
    p = tensors["p_struct"].coalesce().cpu()
    region_to_road = p.indices().numpy()
    arrays["region_to_road_edge_index"] = region_to_road
    arrays["region_to_road_weight"] = p.values().numpy()
    arrays["road_to_region_edge_index"] = region_to_road[[1, 0]]
    arrays["road_to_region_weight"] = p.values().numpy()
    npz_path = output_dir / f"{city}_hierarchy.npz"
    json_path = output_dir / f"{city}_hierarchy.json"
    np.savez_compressed(npz_path, **arrays)
    with open(json_path, "w") as handle:
        json.dump(dict(manifest), handle, indent=2)
    return npz_path, json_path


def load_static_cache(cache_dir: str | Path, city: str) -> tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    cache_dir = Path(cache_dir)
    data = np.load(cache_dir / f"{city}_hierarchy.npz")
    with open(cache_dir / f"{city}_hierarchy.json") as handle:
        manifest = json.load(handle)
    if manifest["city_id"] != city:
        raise ValueError("严格模式: hierarchy manifest city_id 不一致")
    tensors = {
        "region_x": torch.from_numpy(data["region_x"]).float(),
        "region_edge_index": torch.from_numpy(data["region_edge_index"]).long(),
        "road_x": torch.from_numpy(data["road_x"]).float(),
        "road_edge_index": torch.from_numpy(data["road_edge_index"]).long(),
        "p_struct": _deserialize_sparse(data, "p_struct"),
        "b_in": _deserialize_sparse(data, "b_in"),
        "b_out": _deserialize_sparse(data, "b_out"),
    }
    for key in (
        "region_to_road_edge_index", "region_to_road_weight",
        "road_to_region_edge_index", "road_to_region_weight",
    ):
        if key not in data:
            raise KeyError(f"严格模式: hierarchy cache 缺少双向层次边 {key}")
        tensors[key] = torch.from_numpy(data[key]).long() if key.endswith("edge_index") else torch.from_numpy(data[key]).float()
    return tensors, manifest


def load_sequence_column(value: Any, seq_length: int, name: str) -> np.ndarray:
    sequence = ast.literal_eval(value) if isinstance(value, str) else value
    result = np.asarray(sequence, dtype=np.float32)
    if result.shape != (seq_length,) or not np.isfinite(result).all():
        raise ValueError(f"严格模式: {name} 必须是有限长度 {seq_length} 序列")
    return result


def validate_micro_sequence_table(
    frame: pd.DataFrame, manifest: Mapping[str, Any], seq_length: int, split: str
) -> pd.DataFrame:
    """验证预聚合道路 passage count 宽表，不做 map matching 或缺行补零。"""

    required = {"city_id", "directed_road_id", "date", "start_hour", "split", "road_passage_count"}
    if required - set(frame.columns):
        raise ValueError(f"严格模式: micro 表缺列 {sorted(required - set(frame.columns))}")
    city = manifest["city_id"]
    if set(frame["city_id"]) != {city} or set(frame["split"]) != {split}:
        raise ValueError("严格模式: micro city_id/split 与实验不一致")
    expected = set(manifest["directed_road_ids"])
    for _, group in frame.groupby(["date", "start_hour", "split"], sort=False):
        actual = set(group["directed_road_id"].astype(str))
        if actual != expected:
            raise ValueError(
                f"严格模式: micro 快照 Road id 覆盖不完整 missing={len(expected-actual)} extra={len(actual-expected)}"
            )
    frame = frame.copy()
    frame["road_passage_count"] = frame["road_passage_count"].map(
        lambda value: load_sequence_column(value, seq_length, "road_passage_count")
    )
    if any((value < 0).any() for value in frame["road_passage_count"]):
        raise ValueError("严格模式: road_passage_count 含负值")
    return frame
