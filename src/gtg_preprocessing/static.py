"""构建 CRAFT 兼容的栅格、POI、人口和 8 类道路静态文件。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS
from shapely import wkt
from shapely.geometry import LineString, MultiLineString, Point, box
from sklearn.feature_extraction.text import TfidfTransformer

from .contracts import (
    CRAFT_ROAD_COLUMNS,
    GTG_ROAD_REQUIRED_COLUMNS,
    GTG_ROAD_TYPE_MAP,
    POI_TYPES,
    ROAD_TYPES,
    ROAD_TYPE_TO_ID,
    craft_feature_columns,
)


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"严格模式: {label} 缺少列 {sorted(missing)}")


def load_gtg_road_geometries(path: str | Path) -> tuple[pd.DataFrame, gpd.GeoDataFrame, int]:
    """读取原 GTG road.csv，严格保持 link_id 与行号对齐。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"严格模式: 缺失 GTG road.csv: {path}")
    road = pd.read_csv(path, low_memory=False)
    _require_columns(road, GTG_ROAD_REQUIRED_COLUMNS, str(path))
    link_ids = road["link_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(link_ids, np.arange(len(road))):
        raise ValueError("严格模式: GTG link_id 必须为 0..N-1 且与 road.csv 行顺序一致")
    try:
        geometries = road["geometry"].map(wkt.loads)
    except Exception as exc:
        raise ValueError(f"严格模式: {path} 含无法解析的 WKT geometry") from exc
    if not geometries.map(lambda geom: isinstance(geom, LineString) and geom.is_valid and not geom.is_empty).all():
        raise ValueError("严格模式: GTG road.csv 必须全部为有效非空 LineString")
    roads_wgs84 = gpd.GeoDataFrame(road.copy(), geometry=geometries, crs=4326)
    utm = roads_wgs84.estimate_utm_crs()
    if utm is None or utm.to_epsg() is None:
        raise ValueError("严格模式: 无法从 GTG 道路估计 UTM CRS")
    return road, roads_wgs84, int(utm.to_epsg())


def convert_gtg_roads(
    raw_road: pd.DataFrame, roads_wgs84: gpd.GeoDataFrame, utm_epsg: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """转换 GTG 道路。

    返回：
      topology_road: 全部 GTG 有向路段，供 GTG 拓扑提取；cycleway 仅在此归入
                     unclassified 以保留原拓扑。
      static_road:   机动车静态道路，cycleway 明确排除。
    """
    names = raw_road["link_type_name"].astype(str).str.strip().str.lower()
    known = set(GTG_ROAD_TYPE_MAP) | {"cycleway"}
    unknown = sorted(set(names.unique()) - known)
    if unknown:
        raise ValueError(f"严格模式: GTG 出现未定义道路类型 {unknown}")
    geometry_length = roads_wgs84.to_crs(utm_epsg).length.to_numpy(dtype=float)
    if np.any(~np.isfinite(geometry_length)) or np.any(geometry_length <= 0):
        raise ValueError("严格模式: GTG 投影道路长度含非有限值或非正值")
    declared_length = pd.to_numeric(raw_road["length"], errors="coerce").to_numpy(dtype=float)
    relative_error = np.abs(geometry_length - declared_length) / np.maximum(geometry_length, 1.0)

    type_ids = names.map(lambda value: GTG_ROAD_TYPE_MAP.get(value, ROAD_TYPE_TO_ID["unclassified"]))
    converted = pd.DataFrame(
        {
            "road_id": np.arange(len(raw_road), dtype=int),
            "from_node_id": raw_road["from_node_id"].to_numpy(),
            "to_node_id": raw_road["to_node_id"].to_numpy(),
            "road_type": [ROAD_TYPES[int(idx)] for idx in type_ids],
            "road_type_id": type_ids.to_numpy(dtype=int),
            # 度量特征统一使用投影几何重算长度，不信任来源单位。
            "length": geometry_length,
            "geometry": raw_road["geometry"].astype(str).to_numpy(),
            "oneway": True,
            "lanes": raw_road["lanes"].to_numpy() if "lanes" in raw_road else np.nan,
            "maxspeed": raw_road["free_speed"].to_numpy() if "free_speed" in raw_road else np.nan,
        }
    )
    topology_road = converted[list(CRAFT_ROAD_COLUMNS)].copy()
    static_mask = names != "cycleway"
    static_road = converted.loc[static_mask, list(CRAFT_ROAD_COLUMNS)].reset_index(drop=True)
    static_road["road_id"] = np.arange(len(static_road), dtype=int)
    return topology_road, static_road, {
        "source_road_count": len(raw_road),
        "static_gtg_road_count": len(static_road),
        "cycleway_excluded_from_static_count": int((~static_mask).sum()),
        "source_type_counts": names.value_counts().sort_index().to_dict(),
        "length_relative_error_median": float(np.nanmedian(relative_error)),
        "length_relative_error_p95": float(np.nanpercentile(relative_error, 95)),
        "service_track_mapping": "unclassified",
        "cycleway_policy": "静态机动车道路排除；GTG 拓扑文件中保留为 unclassified",
    }


def build_trajectory_grid(
    roads_wgs84: gpd.GeoDataFrame,
    used_road_ids: set[int],
    utm_epsg: int,
    grid_size_m: float = 2000.0,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, dict]:
    """构造 UTM 正方形完整格网，并仅保留与轨迹使用道路有正长度相交的格子。"""
    if grid_size_m <= 0:
        raise ValueError("严格模式: grid_size_m 必须为正")
    if not used_road_ids:
        raise ValueError("严格模式: used_road_ids 为空")
    used_idx = np.asarray(sorted(used_road_ids), dtype=int)
    if used_idx.min() < 0 or used_idx.max() >= len(roads_wgs84):
        raise ValueError("严格模式: used_road_ids 越界")
    used = roads_wgs84.iloc[used_idx].to_crs(utm_epsg)
    minx, miny, maxx, maxy = used.total_bounds
    x0 = math.floor(minx / grid_size_m) * grid_size_m
    y0 = math.floor(miny / grid_size_m) * grid_size_m
    x1 = math.ceil(maxx / grid_size_m) * grid_size_m
    y1 = math.ceil(maxy / grid_size_m) * grid_size_m
    xs = np.arange(x0, x1, grid_size_m)
    ys = np.arange(y0, y1, grid_size_m)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("严格模式: 无法生成轨迹覆盖格网")
    polygons = [box(x, y, x + grid_size_m, y + grid_size_m) for y in ys for x in xs]
    full = gpd.GeoDataFrame(
        {"full_grid_id": np.arange(len(polygons), dtype=int)}, geometry=polygons, crs=utm_epsg
    )
    used_sindex = used.sindex
    keep = []
    for cell in full.geometry:
        candidates = np.asarray(used_sindex.query(cell, predicate="intersects"), dtype=int)
        has_line_overlap = any(cell.intersection(used.geometry.iloc[idx]).length > 1e-6 for idx in candidates)
        keep.append(has_line_overlap)
    selected = full.loc[np.asarray(keep)].copy().reset_index(drop=True)
    if selected.empty:
        raise ValueError("严格模式: 没有栅格与轨迹道路形成正长度相交")
    selected.insert(0, "region_id", np.arange(len(selected), dtype=int))
    return full, selected, {
        "grid_crs": f"EPSG:{utm_epsg}",
        "grid_size_m": float(grid_size_m),
        "full_grid_count": len(full),
        "selected_grid_count": len(selected),
        "selection_rule": "与合并 GTG train/test 中实际使用道路形成正长度相交",
        "projected_bounds": [float(x0), float(y0), float(x1), float(y1)],
    }


def build_grid_region_tables(
    full_grid_utm: gpd.GeoDataFrame, selected_grid_utm: gpd.GeoDataFrame, utm_epsg: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """输出 grid_region.csv、全有向非自环 grid_region_rel.csv 与 data_feature.json。"""
    selected_wgs = selected_grid_utm.to_crs(4326)
    grid_region = pd.DataFrame(
        {
            "region_id": selected_grid_utm["region_id"].to_numpy(dtype=int),
            "geometry": selected_wgs.geometry.map(lambda geom: geom.wkt),
        }
    )
    centroids = selected_grid_utm.geometry.centroid
    rel_records = []
    for ori in range(len(selected_grid_utm)):
        for des in range(len(selected_grid_utm)):
            if ori == des:
                continue
            rel_records.append(
                {
                    "ori": ori,
                    "des": des,
                    "is_adj": int(selected_grid_utm.geometry.iloc[ori].touches(selected_grid_utm.geometry.iloc[des])),
                    "dist": float(centroids.iloc[ori].distance(centroids.iloc[des]) / 1000.0),
                }
            )
    grid_rel = pd.DataFrame.from_records(rel_records, columns=["ori", "des", "is_adj", "dist"])
    if grid_rel.empty or int(grid_rel["is_adj"].sum()) == 0:
        raise ValueError("严格模式: grid_region_rel 没有邻接边")

    full_wgs = full_grid_utm.to_crs(4326)
    min_lon, min_lat, max_lon, max_lat = full_wgs.total_bounds
    data_feature = {
        "mean_lon": float((min_lon + max_lon) / 2.0),
        "mean_lat": float((min_lat + max_lat) / 2.0),
        "min_lon": float(min_lon),
        "min_lat": float(min_lat),
        "max_lon": float(max_lon),
        "max_lat": float(max_lat),
        "type": "Polygon",
        "coordinates": [[
            [float(min_lon), float(min_lat)],
            [float(min_lon), float(max_lat)],
            [float(max_lon), float(max_lat)],
            [float(max_lon), float(min_lat)],
            [float(min_lon), float(min_lat)],
        ]],
        "utm_epsg": int(utm_epsg),
    }
    return grid_region, grid_rel, data_feature


def read_vector(path: str | Path, layer: str | None = None) -> gpd.GeoDataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"严格模式: 缺失外部矢量文件 {path}")
    kwargs = {"layer": layer} if layer else {}
    frame = gpd.read_file(path, **kwargs)
    if frame.crs is None:
        raise ValueError(f"严格模式: {path} 没有 CRS")
    if frame.empty:
        raise ValueError(f"严格模式: {path} 为空")
    frame = frame.to_crs(4326)
    if frame.geometry.is_empty.any() or frame.geometry.isna().any():
        raise ValueError(f"严格模式: {path} 含空 geometry")
    if (~frame.geometry.is_valid).any():
        raise ValueError(f"严格模式: {path} 含无效 geometry")
    return frame


def normalize_osm_supplement_roads(
    osm: gpd.GeoDataFrame, utm_epsg: int
) -> tuple[pd.DataFrame, dict]:
    """只提取 OSM residential/living_street，作为 GTG 缺失类别补充。"""
    if "highway" not in osm.columns:
        raise ValueError("严格模式: OSM 补充道路缺少 highway 标签列")
    osm = osm.copy()
    osm["highway"] = osm["highway"].astype(str).str.lower().str.strip()
    osm = osm[osm["highway"].isin({"residential", "living_street"})].copy()
    if osm.empty:
        raise ValueError("严格模式: OSM 输入中没有 residential/living_street 真实道路")
    osm = osm.explode(index_parts=False, ignore_index=True)
    osm = osm[osm.geometry.map(lambda geom: isinstance(geom, LineString))].copy().reset_index(drop=True)
    if osm.empty:
        raise ValueError("严格模式: OSM residential/living_street 没有 LineString geometry")
    # 同一 OSM 导出可能含双向重复边；规范化几何方向后严格去重并记录。
    normalized_wkb = osm.geometry.map(lambda geom: geom.normalize().wkb_hex)
    duplicate = normalized_wkb.duplicated(keep="first")
    duplicate_count = int(duplicate.sum())
    osm = osm.loc[~duplicate].reset_index(drop=True)
    lengths = osm.to_crs(utm_epsg).length.to_numpy(dtype=float)
    if np.any(lengths <= 0) or np.any(~np.isfinite(lengths)):
        raise ValueError("严格模式: OSM 补充道路长度异常")
    n = len(osm)
    out = pd.DataFrame(
        {
            "road_id": np.arange(n, dtype=int),
            # 补充道路只用于 CRAFT 静态计数；使用独立负节点避免伪造与 GTG 拓扑的连接。
            "from_node_id": -(2 * np.arange(n, dtype=int) + 1),
            "to_node_id": -(2 * np.arange(n, dtype=int) + 2),
            "road_type": osm["highway"].to_numpy(),
            "road_type_id": osm["highway"].map(ROAD_TYPE_TO_ID).to_numpy(dtype=int),
            "length": lengths,
            "geometry": osm.geometry.map(lambda geom: geom.wkt).to_numpy(),
            # OSM 中存在但为 None 的 oneway 标签按未声明单向处理为 False；
            # 这是道路静态统计字段，不用于伪造 GTG 拓扑连接。
            "oneway": osm["oneway"].fillna(False).to_numpy() if "oneway" in osm else False,
            "lanes": osm["lanes"].to_numpy() if "lanes" in osm else np.nan,
            "maxspeed": osm["maxspeed"].to_numpy() if "maxspeed" in osm else np.nan,
        }
    )
    return out[list(CRAFT_ROAD_COLUMNS)], {
        "osm_supplement_road_count": len(out),
        "osm_duplicate_geometry_rows_removed": duplicate_count,
        "osm_supplement_type_counts": out["road_type"].value_counts().sort_index().to_dict(),
    }


def merge_static_roads(gtg_static: pd.DataFrame, osm_supplement: pd.DataFrame) -> pd.DataFrame:
    merged = pd.concat([gtg_static, osm_supplement], ignore_index=True)
    merged["road_id"] = np.arange(len(merged), dtype=int)
    if not set(merged["road_type_id"].unique()).issubset(set(range(len(ROAD_TYPES)))):
        raise ValueError("严格模式: 合并道路含越界 road_type_id")
    mandatory = [
        "road_id",
        "from_node_id",
        "to_node_id",
        "road_type",
        "road_type_id",
        "length",
        "geometry",
        "oneway",
    ]
    if merged[mandatory].isna().any().any():
        raise ValueError("严格模式: 合并 road.csv 的必需字段含缺失")
    return merged[list(CRAFT_ROAD_COLUMNS)]


def _poi_candidates(row: pd.Series) -> list[int]:
    candidates = []
    amenity = str(row.get("amenity", "")).strip().lower()
    amenity_map = {
        "bicycle_rental": 0,
        "fast_food": 1,
        "restaurant": 2,
        "bicycle_parking": 3,
        "cafe": 4,
    }
    if amenity in amenity_map:
        candidates.append(amenity_map[amenity])
    for column, type_id in (
        ("public_transport", 5),
        ("shop", 6),
        ("tourism", 7),
        ("leisure", 8),
        ("office", 9),
        ("historic", 10),
        ("sport", 11),
    ):
        value = row.get(column)
        if pd.notna(value) and str(value).strip().lower() not in {"", "no", "none", "nan"}:
            candidates.append(type_id)
    return sorted(set(candidates))


def prepare_poi_features(
    osm_poi: gpd.GeoDataFrame,
    full_grid_utm: gpd.GeoDataFrame,
    selected_grid_utm: gpd.GeoDataFrame,
    utm_epsg: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    """分类 OSM POI，并在完整矩形格网上拟合 CRAFT 同款 TF-IDF。"""
    poi = osm_poi.copy()
    source_id_column = next(
        (column for column in ("@id", "osmid", "osm_id") if column in poi.columns),
        None,
    )
    if source_id_column is not None:
        element_type_column = next(
            (column for column in ("@type", "element_type", "osm_type") if column in poi.columns),
            None,
        )
        dedupe_key = poi[source_id_column].astype(str)
        if element_type_column is not None:
            dedupe_key = poi[element_type_column].astype(str) + ":" + dedupe_key
        dedupe_rule = source_id_column if element_type_column is None else f"{element_type_column}+{source_id_column}"
    else:
        tag_columns = [
            column
            for column in (
                "amenity",
                "public_transport",
                "shop",
                "tourism",
                "leisure",
                "office",
                "historic",
                "sport",
            )
            if column in poi.columns
        ]
        tag_key = poi[tag_columns].fillna("").astype(str).agg("|".join, axis=1)
        dedupe_key = poi.geometry.map(lambda geom: geom.normalize().wkb_hex) + "|" + tag_key
        dedupe_rule = "normalized_geometry+CRAFT_relevant_tags"
    duplicate_mask = dedupe_key.duplicated(keep="first")
    duplicate_count = int(duplicate_mask.sum())
    poi = poi.loc[~duplicate_mask].copy()
    if "poi_type_id" in poi.columns:
        numeric = pd.to_numeric(poi["poi_type_id"], errors="coerce")
        if numeric.isna().any() or not numeric.astype(int).between(0, len(POI_TYPES) - 1).all():
            raise ValueError("严格模式: OSM POI 的 poi_type_id 必须完整且位于 0..11")
        poi["poi_type_id"] = numeric.astype(int)
        multi_category_count = 0
        unclassified_count = 0
        classification = "外部文件预分类 poi_type_id"
    else:
        candidates = poi.apply(_poi_candidates, axis=1)
        multi_category_count = int(candidates.map(lambda value: len(value) > 1).sum())
        unclassified_count = int(candidates.map(len).eq(0).sum())
        poi = poi.loc[candidates.map(len).gt(0)].copy()
        candidates = candidates.loc[candidates.map(len).gt(0)]
        # CRAFT poi.csv 每条记录只有一个类别；无原脚本时按固定类别优先级确定。
        poi["poi_type_id"] = candidates.map(lambda value: value[0]).astype(int)
        classification = "由 OSM 标签按 CRAFT 类别顺序确定单一类别（推断规则）"
    if poi.empty:
        raise ValueError("严格模式: 12 类 OSM POI 分类后为空")
    poi["geometry"] = poi.geometry.map(
        lambda geom: geom if isinstance(geom, Point) else geom.representative_point()
    )
    poi = gpd.GeoDataFrame(poi, geometry="geometry", crs=4326).to_crs(utm_epsg)
    full = full_grid_utm[["full_grid_id", "geometry"]]
    joined = gpd.sjoin(poi, full, predicate="within", how="inner")
    if joined.empty:
        raise ValueError("严格模式: OSM POI 与 GTG 完整格网无任何重合，疑似坐标不一致")
    # 位于边界的点可能未被 within 捕获；不做静默最近格回填。
    counts = np.zeros((len(full_grid_utm), len(POI_TYPES)), dtype=np.int64)
    for row in joined.itertuples():
        counts[int(row.full_grid_id), int(row.poi_type_id)] += 1
    scores = TfidfTransformer().fit_transform(counts).toarray()
    selected_ids = selected_grid_utm["full_grid_id"].to_numpy(dtype=int)

    joined_wgs = joined.to_crs(4326).reset_index(drop=True)
    poi_csv = pd.DataFrame(
        {
            "poi_id": np.arange(len(joined_wgs), dtype=int),
            "name": joined_wgs["name"].fillna("").astype(str).to_numpy()
            if "name" in joined_wgs
            else "",
            "poi_type": [POI_TYPES[int(idx)] for idx in joined_wgs["poi_type_id"]],
            "poi_type_id": joined_wgs["poi_type_id"].to_numpy(dtype=int),
            "geometry": joined_wgs.geometry.map(lambda geom: geom.wkt).to_numpy(),
        }
    )
    return poi_csv, counts[selected_ids], scores[selected_ids], {
        "input_poi_count": len(osm_poi),
        "duplicate_poi_rows_removed": duplicate_count,
        "poi_deduplication_rule": dedupe_rule,
        "classified_poi_count": len(poi),
        "poi_inside_full_grid_count": len(joined),
        "unclassified_poi_count": unclassified_count,
        "multi_category_poi_count": multi_category_count,
        "classification_rule": classification,
        "poi_type_counts": poi_csv["poi_type"].value_counts().sort_index().to_dict(),
        "tfidf_scope": "完整未筛选矩形格网，然后选择轨迹覆盖格子",
        "tfidf_transformer": "sklearn.feature_extraction.text.TfidfTransformer defaults",
    }


def prepare_population_features(
    population_path: str | Path,
    full_grid_utm: gpd.GeoDataFrame,
    selected_grid_utm: gpd.GeoDataFrame,
    utm_epsg: int,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """读取已从 2026 WorldPop 人口数栅格转换/裁剪的点 CSV。"""
    population_path = Path(population_path)
    if not population_path.exists():
        raise FileNotFoundError(f"严格模式: 缺失 2026 人口点文件 {population_path}")
    population = pd.read_csv(population_path)
    _require_columns(population, {"lon", "lat", "population"}, str(population_path))
    values = population[["lon", "lat", "population"]].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError("严格模式: population.csv 含非数值/NaN/Inf")
    if (values["population"] < 0).any():
        raise ValueError("严格模式: population.csv 含负人口")
    points = gpd.GeoDataFrame(
        values.copy(),
        geometry=gpd.points_from_xy(values.lon, values.lat),
        crs=4326,
    ).to_crs(utm_epsg)
    joined = gpd.sjoin(
        points,
        selected_grid_utm[["region_id", "geometry"]],
        predicate="within",
        how="inner",
    )
    if joined.empty:
        raise ValueError("严格模式: 2026 人口点与保留格网无有效像元重合，疑似坐标/产品错误")
    sums = np.zeros(len(selected_grid_utm), dtype=float)
    coverage = np.zeros(len(selected_grid_utm), dtype=np.int64)
    grouped = joined.groupby("region_id")["population"].sum()
    counts = joined.groupby("region_id").size()
    for region_id, value in grouped.items():
        sums[int(region_id)] = float(value)
    for region_id, value in counts.items():
        coverage[int(region_id)] = int(value)
    missing_regions = np.flatnonzero(coverage == 0).astype(int).tolist()
    if missing_regions:
        raise ValueError(
            "严格模式: selected region 完全没有有效人口像元覆盖，禁止补 0; "
            f"missing_region_ids={missing_regions}"
        )
    if not np.all(np.isfinite(sums)):
        raise ValueError("严格模式: 区域人口聚合出现 NaN/Inf")
    full_joined = gpd.sjoin(
        points,
        full_grid_utm[["full_grid_id", "geometry"]],
        predicate="within",
        how="inner",
    ).to_crs(4326)
    population_csv = full_joined[["lon", "lat", "population"]].copy().reset_index(drop=True)
    return population_csv, sums, {
        "input_population_points": len(population),
        "population_points_inside_full_grid": len(full_joined),
        "population_points_inside_selected_grid": len(joined),
        "selected_grid_total_population": float(sums.sum()),
        "selected_grid_valid_pixel_counts": coverage.astype(int).tolist(),
        "zero_population_region_ids": np.flatnonzero((coverage > 0) & (sums == 0)).astype(int).tolist(),
        "population_semantics": "WorldPop 2026 每像元人口数求和，不是人口密度栅格求和",
    }


def aggregate_road_features(
    road_csv: pd.DataFrame, selected_grid_utm: gpd.GeoDataFrame, utm_epsg: int
) -> tuple[np.ndarray, np.ndarray, dict]:
    """复现 CRAFT 样例：相交即计数，并把整条原始道路长度计入每个相交格。"""
    _require_columns(road_csv, set(CRAFT_ROAD_COLUMNS), "合并 road.csv")
    geometries = road_csv["geometry"].map(wkt.loads)
    roads = gpd.GeoDataFrame(road_csv.copy(), geometry=geometries, crs=4326).to_crs(utm_epsg)
    joined = gpd.sjoin(
        roads[["road_id", "road_type_id", "length", "geometry"]],
        selected_grid_utm[["region_id", "geometry"]],
        predicate="intersects",
        how="inner",
    )
    if joined.empty:
        raise ValueError("严格模式: 合并道路与保留格网无相交")
    counts = np.zeros((len(selected_grid_utm), len(ROAD_TYPES)), dtype=np.int64)
    lengths = np.zeros((len(selected_grid_utm), len(ROAD_TYPES)), dtype=float)
    for row in joined.itertuples():
        region_id = int(row.region_id)
        type_id = int(row.road_type_id)
        counts[region_id, type_id] += 1
        lengths[region_id, type_id] += float(row.length)
    return counts, lengths, {
        "road_grid_intersection_rows": len(joined),
        "aggregation_rule": "道路与格相交即计 1，并累加整条道路 length；跨格道路重复计入",
        "regions_without_roads": np.flatnonzero(counts.sum(axis=1) == 0).astype(int).tolist(),
    }


def build_grid_region_feature(
    full_grid_utm: gpd.GeoDataFrame,
    selected_grid_utm: gpd.GeoDataFrame,
    poi_counts: np.ndarray,
    poi_scores: np.ndarray,
    population: np.ndarray,
    road_counts: np.ndarray,
    road_lengths: np.ndarray,
) -> pd.DataFrame:
    """组装 CRAFT 读取器实际使用的 45 维区域特征及几何元数据。"""
    n = len(selected_grid_utm)
    expected_shapes = {
        "poi_counts": (n, len(POI_TYPES)),
        "poi_scores": (n, len(POI_TYPES)),
        "population": (n,),
        "road_counts": (n, len(ROAD_TYPES)),
        "road_lengths": (n, len(ROAD_TYPES)),
    }
    actual = {
        "poi_counts": poi_counts.shape,
        "poi_scores": poi_scores.shape,
        "population": population.shape,
        "road_counts": road_counts.shape,
        "road_lengths": road_lengths.shape,
    }
    if actual != expected_shapes:
        raise ValueError(f"严格模式: 静态特征形状不一致 actual={actual}, expected={expected_shapes}")
    selected_wgs = selected_grid_utm.to_crs(4326)
    centers_utm = selected_grid_utm.geometry.centroid
    centers_wgs = gpd.GeoSeries(centers_utm, crs=selected_grid_utm.crs).to_crs(4326)
    minx, miny, maxx, maxy = full_grid_utm.total_bounds
    study_center = Point((minx + maxx) / 2.0, (miny + maxy) / 2.0)
    areas = selected_grid_utm.geometry.area.to_numpy(dtype=float)
    frame = pd.DataFrame(
        {
            "region_id": np.arange(n, dtype=int),
            "geometry": selected_wgs.geometry.map(lambda geom: geom.wkt),
            "area": areas,
            "lon": centers_wgs.x.to_numpy(dtype=float),
            "lat": centers_wgs.y.to_numpy(dtype=float),
            "dist_to_center": centers_utm.distance(study_center).to_numpy(dtype=float),
            "population": population.astype(float),
            "population_density": population.astype(float) / areas,
        }
    )
    for type_id in range(len(POI_TYPES)):
        frame[f"poi_num_{type_id}"] = poi_counts[:, type_id].astype(int)
        frame[f"poi_score_{type_id}"] = poi_scores[:, type_id].astype(float)
    for type_id in range(len(ROAD_TYPES)):
        frame[f"road_num_{type_id}"] = road_counts[:, type_id].astype(int)
        frame[f"road_length_{type_id}"] = road_lengths[:, type_id].astype(float)
    frame["road_num"] = road_counts.sum(axis=1).astype(int)
    frame["road_length"] = road_lengths.sum(axis=1).astype(float)
    required = craft_feature_columns()
    if frame[required].isna().any().any() or not np.isfinite(frame[required].to_numpy(dtype=float)).all():
        raise ValueError("严格模式: grid_region_feature 45 维含 NaN/Inf")
    return frame
