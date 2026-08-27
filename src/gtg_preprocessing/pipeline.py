"""GTG -> CRAFT 兼容城市目录的端到端严格编排。"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from shapely import wkt

from .contracts import GTG_ROAD_TYPE_MAP, POI_TYPES, ROAD_TYPES
from .flow import (
    aggregate_boundary_crossings,
    build_road_region_paths,
    build_sliding_windows,
    collect_used_road_ids,
    interpolate_internal_zeros,
    normalize_train_validation,
    serialize_flow_lists,
)
from .static import (
    aggregate_road_features,
    build_grid_region_feature,
    build_grid_region_tables,
    build_trajectory_grid,
    convert_gtg_roads,
    load_gtg_road_geometries,
    merge_static_roads,
    normalize_osm_supplement_roads,
    prepare_poi_features,
    prepare_population_features,
    read_vector,
)


def _resolve_path(value, config_dir: Path, label: str) -> Path:
    if value is None or str(value).strip() == "":
        raise ValueError(f"严格模式: 配置缺少 {label}")
    path = Path(value).expanduser()
    return path if path.is_absolute() else (config_dir / path).resolve()


def load_config(config_path: str | Path) -> tuple[dict, Path]:
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"严格模式: 缺失配置 {path}")
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"严格模式: {path} 顶层必须是 mapping")
    return config, path.parent


def validate_city_config(city: str, city_cfg: dict, config: dict, config_dir: Path) -> dict:
    if not isinstance(city_cfg, dict):
        raise ValueError(f"严格模式: cities.{city} 必须是 mapping")
    gtg_dir = _resolve_path(city_cfg.get("gtg_dir"), config_dir, f"cities.{city}.gtg_dir")
    paths = {
        "gtg_dir": gtg_dir,
        "road": gtg_dir / "map" / "road.csv",
        "train": gtg_dir / "traj" / "train.csv",
        "test": gtg_dir / "traj" / "test.csv",
        "osm_poi": _resolve_path(city_cfg.get("osm_poi"), config_dir, f"cities.{city}.osm_poi"),
        "osm_roads": _resolve_path(city_cfg.get("osm_roads"), config_dir, f"cities.{city}.osm_roads"),
        "population": _resolve_path(
            city_cfg.get("population"), config_dir, f"cities.{city}.population"
        ),
    }
    missing = [str(path) for key, path in paths.items() if key != "gtg_dir" and not path.exists()]
    if missing:
        raise FileNotFoundError("严格模式: 缺失输入文件:\n  " + "\n  ".join(missing))
    use_source_validation = bool(config.get("use_source_validation", True))
    validation_start = city_cfg.get("validation_start")
    if use_source_validation:
        if validation_start is None:
            raise ValueError(f"严格模式: cities.{city}.validation_start 未配置")
        validation_start = pd.Timestamp(validation_start)
        if validation_start.tzinfo is not None:
            raise ValueError(f"严格模式: cities.{city}.validation_start 应写 Asia/Shanghai 本地朴素时间")
    else:
        validation_start = None

    snapshot = city_cfg.get("osm_snapshot_date")
    if snapshot is None or pd.Timestamp(snapshot).year != 2026:
        raise ValueError(f"严格模式: cities.{city}.osm_snapshot_date 必须记录实际 2026 OSM 提取日期")
    if int(city_cfg.get("population_year", -1)) != 2026:
        raise ValueError(f"严格模式: cities.{city}.population_year 必须为 2026")
    return {
        **paths,
        "validation_start": validation_start,
        "use_source_validation": use_source_validation,
        "osm_snapshot_date": str(snapshot),
        "population_year": 2026,
        "osm_poi_layer": city_cfg.get("osm_poi_layer"),
        "osm_road_layer": city_cfg.get("osm_road_layer"),
    }


def _write_json(path: Path, value: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def _ensure_new_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"严格模式: 输出目录已存在且非空 {path}；为避免覆盖，请改用新目录或人工确认清理"
        )


def build_city(city: str, config: dict, config_dir: Path) -> dict:
    city_cfg = validate_city_config(city, config["cities"][city], config, config_dir)
    output_root = _resolve_path(config.get("output_root"), config_dir, "output_root")
    norm_root = _resolve_path(config.get("norm_output_root"), config_dir, "norm_output_root")
    city_output = output_root / city
    city_norm_output = norm_root / city
    _ensure_new_output_dir(city_output)
    _ensure_new_output_dir(city_norm_output)

    raw_road, roads_wgs84, utm_epsg = load_gtg_road_geometries(city_cfg["road"])
    trajectories = [city_cfg["train"], city_cfg["test"]]
    used_road_ids, usage_meta = collect_used_road_ids(
        trajectories,
        num_roads=len(raw_road),
        chunksize=int(config.get("trajectory_chunksize", 100_000)),
    )
    full_grid, selected_grid, grid_meta = build_trajectory_grid(
        roads_wgs84,
        used_road_ids,
        utm_epsg,
        grid_size_m=float(config.get("grid_size_m", 2000.0)),
    )
    grid_region, grid_rel, data_feature = build_grid_region_tables(
        full_grid, selected_grid, utm_epsg
    )

    topology_road, gtg_static_road, gtg_road_meta = convert_gtg_roads(
        raw_road, roads_wgs84, utm_epsg
    )
    osm_roads = read_vector(city_cfg["osm_roads"], layer=city_cfg["osm_road_layer"])
    osm_supplement, osm_road_meta = normalize_osm_supplement_roads(osm_roads, utm_epsg)
    osm_supplement_gdf = gpd.GeoDataFrame(
        osm_supplement.copy(),
        geometry=osm_supplement["geometry"].map(wkt.loads),
        crs=4326,
    ).to_crs(utm_epsg)
    osm_overlap = gpd.sjoin(
        osm_supplement_gdf,
        selected_grid[["region_id", "geometry"]],
        predicate="intersects",
        how="inner",
    )
    if osm_overlap.empty:
        raise ValueError("严格模式: OSM 补充道路与 GTG 保留格网无相交，疑似坐标或裁剪范围错误")
    osm_road_meta["road_grid_intersection_rows"] = len(osm_overlap)
    road_csv = merge_static_roads(gtg_static_road, osm_supplement)
    road_counts, road_lengths, road_aggregation_meta = aggregate_road_features(
        road_csv, selected_grid, utm_epsg
    )

    osm_poi = read_vector(city_cfg["osm_poi"], layer=city_cfg["osm_poi_layer"])
    poi_csv, poi_counts, poi_scores, poi_meta = prepare_poi_features(
        osm_poi, full_grid, selected_grid, utm_epsg
    )
    population_csv, population_values, population_meta = prepare_population_features(
        city_cfg["population"], full_grid, selected_grid, utm_epsg
    )
    grid_feature = build_grid_region_feature(
        full_grid,
        selected_grid,
        poi_counts,
        poi_scores,
        population_values,
        road_counts,
        road_lengths,
    )

    road_geometries_utm = list(roads_wgs84.to_crs(utm_epsg).geometry)
    road_paths, crossing_geometry_meta = build_road_region_paths(
        road_geometries_utm,
        selected_grid[["region_id", "geometry"]],
        ambiguous_interval_error_ratio=float(
            config.get("ambiguous_interval_error_ratio", 1e-3)
        ),
    )
    raw_hourly, crossing_meta = aggregate_boundary_crossings(
        trajectories,
        road_paths,
        num_regions=len(selected_grid),
        timezone_name=str(config.get("timezone", "Asia/Shanghai")),
        chunksize=int(config.get("trajectory_chunksize", 100_000)),
    )
    split_boundaries = (
        [city_cfg["validation_start"]] if city_cfg["use_source_validation"] else []
    )
    interpolated_hourly, interpolation_meta = interpolate_internal_zeros(
        raw_hourly, split_boundaries=split_boundaries
    )
    full_windows, train_windows, validation_windows, window_meta = build_sliding_windows(
        interpolated_hourly,
        seq_length=int(config.get("seq_length", 24)),
        validation_start=city_cfg["validation_start"],
    )
    norm_train, norm_validation, norm_meta = normalize_train_validation(
        train_windows, validation_windows
    )

    # 到这里所有严格检查均已通过，才开始创建输出目录并写文件。
    city_output.mkdir(parents=True, exist_ok=True)
    city_norm_output.mkdir(parents=True, exist_ok=True)
    grid_region.to_csv(city_output / "grid_region.csv", index=False)
    grid_feature.to_csv(city_output / "grid_region_feature.csv", index=False)
    grid_rel.to_csv(city_output / "grid_region_rel.csv", index=False)
    poi_csv.to_csv(city_output / "poi.csv", index=False)
    population_csv.to_csv(city_output / "population.csv", index=False)
    road_csv.to_csv(city_output / "road.csv", index=False)
    topology_road.to_csv(city_output / "gtg_road.csv", index=False)
    serialize_flow_lists(full_windows).to_csv(city_output / "slide_bike_flow.csv", index=False)
    serialize_flow_lists(train_windows).to_csv(
        city_output / "slide_bike_flow_train.csv", index=False
    )
    if city_cfg["use_source_validation"]:
        serialize_flow_lists(validation_windows).to_csv(
            city_output / "slide_bike_flow_test.csv", index=False
        )
    raw_hourly.to_csv(city_output / "hourly_boundary_flow_raw.csv", index=False)
    interpolated_hourly.to_csv(
        city_output / "hourly_boundary_flow_interpolated.csv", index=False
    )
    serialize_flow_lists(norm_train).to_csv(
        city_norm_output / "norm_train_len_24.csv", index=False
    )
    if city_cfg["use_source_validation"]:
        serialize_flow_lists(norm_validation).to_csv(
            city_norm_output / "norm_test_len_24.csv", index=False
        )
    _write_json(city_output / "data_feature.json", data_feature)
    _write_json(city_norm_output / "normalization_meta.json", norm_meta)
    _write_json(
        city_output / "road_type_mapping.json",
        {
            "craft_road_types": {str(idx): name for idx, name in enumerate(ROAD_TYPES)},
            "gtg_mapping": GTG_ROAD_TYPE_MAP,
            "osm_supplement_types": ["residential", "living_street"],
        },
    )
    meta = {
        "city": city,
        "source_semantics": "GTG 车辆轨迹的栅格边界穿越流量",
        "craft_compatibility_name_note": "slide_bike_flow 文件名仅为兼容代码，数据不是共享单车",
        "osm_snapshot_date": city_cfg["osm_snapshot_date"],
        "population_year": city_cfg["population_year"],
        "utm_epsg": utm_epsg,
        "trajectory_usage": usage_meta,
        "grid": grid_meta,
        "gtg_road_conversion": gtg_road_meta,
        "osm_road_supplement": osm_road_meta,
        "road_aggregation": road_aggregation_meta,
        "poi": poi_meta,
        "population": population_meta,
        "crossing_geometry": crossing_geometry_meta,
        "crossing_flow": crossing_meta,
        "interpolation": interpolation_meta,
        "windows": window_meta,
        "normalization": norm_meta,
        "confirmed_vs_inferred": {
            "confirmed": [
                "合并 GTG 原随机 train/test 后聚合",
                "start_time 按 UTC Unix 秒解析并转换 Asia/Shanghai",
                "统计车辆穿越栅格边界",
                "约 2 km 栅格按轨迹覆盖确定",
                "48 个值全正筛选",
                "道路相交即累加整条长度",
                "POI/人口使用真实 2026 外部数据，不以零占位",
            ],
            "inferred": [
                "线性插值采用内部零段、无首尾外推，并在训练/验证边界分段",
                "归一化采用本仓库重建的逐城市训练集全局 Min-Max",
                "OSM 多标签 POI 若未预分类，按 CRAFT 类别顺序选择单一类别",
            ],
        },
    }
    _write_json(city_output / "preprocess_meta.json", meta)
    return meta


def run(config_path: str | Path, cities: list[str] | None = None) -> list[dict]:
    config, config_dir = load_config(config_path)
    city_configs = config.get("cities")
    if not isinstance(city_configs, dict) or not city_configs:
        raise ValueError("严格模式: 配置 cities 为空")
    requested = list(city_configs) if not cities else cities
    unknown = sorted(set(requested) - set(city_configs))
    if unknown:
        raise ValueError(f"严格模式: 配置中不存在城市 {unknown}")
    if str(config.get("timezone", "Asia/Shanghai")) != "Asia/Shanghai":
        raise ValueError("严格模式: GTG 三城统一要求 timezone=Asia/Shanghai")
    if int(config.get("seq_length", 24)) != 24:
        raise ValueError("严格模式: CRAFT 兼容要求 seq_length=24")
    reports = []
    for city in requested:
        reports.append(build_city(city, config, config_dir))
    return reports
