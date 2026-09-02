"""三层静态图缓存的序列化与加载。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .contracts import CityStaticHierarchy, validate_city_static_hierarchy


def _array(hierarchy: CityStaticHierarchy) -> dict[str, np.ndarray]:
    arrays = {
        "region_x": hierarchy.region_x.cpu().numpy(),
        "region_edge_index": hierarchy.region_edge_index.cpu().numpy(),
        "road_edge_index": hierarchy.road_edge_index.cpu().numpy(),
        "road_ids": np.asarray(hierarchy.road_ids, dtype=str),
        "syntax_x": hierarchy.syntax_x.cpu().numpy(),
        "syntax_edge_index": hierarchy.syntax_edge_index.cpu().numpy(),
        "road_to_syntax_assignment": hierarchy.road_to_syntax_assignment.cpu().numpy(),
        "road_to_syntax_edge_index": hierarchy.road_to_syntax_edge_index.cpu().numpy(),
        "road_to_syntax_weight": hierarchy.road_to_syntax_weight.cpu().numpy(),
        "road_to_syntax_shape": np.asarray(hierarchy.road_to_syntax_shape, dtype=np.int64),
        "syntax_to_region_edge_index": hierarchy.syntax_to_region_edge_index.cpu().numpy(),
        "syntax_to_region_weight": hierarchy.syntax_to_region_weight.cpu().numpy(),
        "syntax_to_region_shape": np.asarray(hierarchy.syntax_to_region_shape, dtype=np.int64),
        "region_has_syntax": hierarchy.region_has_syntax.cpu().numpy(),
    }
    if hierarchy.metadata.get("feature_version") == "three-layer-start-road-v2":
        arrays["road_x"] = hierarchy.road_x.cpu().numpy()
    else:
        arrays["road_topo_x"] = hierarchy.road_x.cpu().numpy()
    return arrays


def save_city_static_hierarchy(
    hierarchy: CityStaticHierarchy, output_dir: str | Path
) -> tuple[Path, Path]:
    """保存为独立的 ``static_hierarchy`` cache，不覆盖旧 GTG cache。"""

    validate_city_static_hierarchy(hierarchy)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    city = hierarchy.city_id
    npz_path = output_dir / f"{city}_static_hierarchy.npz"
    meta_path = output_dir / f"{city}_static_hierarchy_meta.json"
    np.savez_compressed(npz_path, **_array(hierarchy))
    meta = dict(hierarchy.metadata)
    meta["city"] = city
    meta["road_ids"] = list(hierarchy.road_ids)
    meta["road_to_syntax_shape"] = list(hierarchy.road_to_syntax_shape)
    meta["syntax_to_region_shape"] = list(hierarchy.syntax_to_region_shape)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return npz_path, meta_path


def _required(data: Any, keys: tuple[str, ...], path: Path) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise KeyError(f"严格模式: 三层静态 cache 缺少字段 {missing}: {path}")


def load_city_static_hierarchy(
    cache_dir: str | Path,
    city: str,
    *,
    expected_feature_version: str | None = None,
) -> CityStaticHierarchy:
    cache_dir = Path(cache_dir)
    npz_path = cache_dir / f"{city}_static_hierarchy.npz"
    meta_path = cache_dir / f"{city}_static_hierarchy_meta.json"
    if not npz_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"严格模式: 缺少三层静态 cache: {npz_path} / {meta_path}")
    data = np.load(npz_path, allow_pickle=False)
    keys = (
        "region_x", "region_edge_index", "road_edge_index", "road_ids",
        "syntax_x", "syntax_edge_index", "road_to_syntax_assignment",
        "road_to_syntax_edge_index", "road_to_syntax_weight", "road_to_syntax_shape",
        "syntax_to_region_edge_index", "syntax_to_region_weight", "syntax_to_region_shape",
        "region_has_syntax",
    )
    _required(data, keys, npz_path)
    meta = json.loads(meta_path.read_text())
    feature_version = meta.get("feature_version")
    if expected_feature_version is not None and feature_version != expected_feature_version:
        raise ValueError(
            f"严格模式: cache feature_version={feature_version!r}，"
            f"期望 {expected_feature_version!r}: {meta_path}"
        )
    road_key = "road_x" if feature_version == "three-layer-start-road-v2" else "road_topo_x"
    _required(data, (road_key,), npz_path)
    required_meta = (
        "city", "num_regions", "num_roads", "num_road_edges", "num_syntax_nodes",
        "num_syntax_edges", "num_road_to_syntax_links", "num_syntax_to_region_links",
        "road_ids", "syntax_feature_names", "region_feature_order", "empty_region_ids",
        "empty_region_ratio", "utm_epsg", "source_road_file", "local_size",
        "feature_version", "road_to_syntax_shape", "syntax_to_region_shape",
    )
    missing_meta = [key for key in required_meta if key not in meta]
    if missing_meta:
        raise KeyError(f"严格模式: 三层静态 cache metadata 缺少字段 {missing_meta}: {meta_path}")
    if meta.get("city") != city or feature_version not in {
        "three-layer-static-v1", "three-layer-start-road-v2"
    }:
        raise ValueError("严格模式: 三层静态 cache metadata city/feature_version 不一致")
    road_meta_key = "road_feature_names" if feature_version == "three-layer-start-road-v2" else "road_topo_feature_names"
    if road_meta_key not in meta:
        raise KeyError(f"严格模式: 三层静态 cache metadata 缺少字段 [{road_meta_key}]: {meta_path}")
    hierarchy = CityStaticHierarchy(
        city_id=city,
        region_x=torch.from_numpy(data["region_x"]).float(),
        region_edge_index=torch.from_numpy(data["region_edge_index"]).long(),
        road_x=torch.from_numpy(data[road_key]).float(),
        road_edge_index=torch.from_numpy(data["road_edge_index"]).long(),
        road_ids=tuple(str(value) for value in data["road_ids"].tolist()),
        syntax_x=torch.from_numpy(data["syntax_x"]).float(),
        syntax_edge_index=torch.from_numpy(data["syntax_edge_index"]).long(),
        road_to_syntax_assignment=torch.from_numpy(data["road_to_syntax_assignment"]).long(),
        road_to_syntax_edge_index=torch.from_numpy(data["road_to_syntax_edge_index"]).long(),
        road_to_syntax_weight=torch.from_numpy(data["road_to_syntax_weight"]).float(),
        road_to_syntax_shape=tuple(int(value) for value in data["road_to_syntax_shape"].tolist()),
        syntax_to_region_edge_index=torch.from_numpy(data["syntax_to_region_edge_index"]).long(),
        syntax_to_region_weight=torch.from_numpy(data["syntax_to_region_weight"]).float(),
        syntax_to_region_shape=tuple(int(value) for value in data["syntax_to_region_shape"].tolist()),
        region_has_syntax=torch.from_numpy(data["region_has_syntax"]).bool(),
        metadata=meta,
    )
    validate_city_static_hierarchy(hierarchy)
    return hierarchy
