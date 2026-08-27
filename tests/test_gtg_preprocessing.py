"""GTG -> CRAFT 车辆边界流量预处理的合成契约测试。"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, box

from gtg_preprocessing.flow import (
    aggregate_boundary_crossings,
    build_road_region_paths,
    build_sliding_windows,
    interpolate_internal_zeros,
    normalize_train_validation,
)
from gtg_preprocessing.contracts import craft_feature_columns
from gtg_preprocessing.static import (
    aggregate_road_features,
    build_grid_region_feature,
    prepare_poi_features,
    prepare_population_features,
)


def _write_traj(path: Path, start_epoch: int):
    pd.DataFrame(
        {
            "traj_id": [0],
            "start_time": [start_epoch],
            "rid_list": ["0"],
            "dur_list": ["3600"],
        }
    ).to_csv(path, sep=";", index=False)


def test_boundary_crossing_merges_original_splits_and_uses_shanghai_hour(tmp_path):
    regions = gpd.GeoDataFrame(
        {"region_id": [0, 1]},
        geometry=[box(0, 0, 1000, 1000), box(1000, 0, 2000, 1000)],
        crs=3857,
    )
    # 有向道路从 region 0 进入 region 1，边界位于道路长度一半处。
    roads = [LineString([(100, 500), (1900, 500)])]
    paths, path_meta = build_road_region_paths(roads, regions)
    assert paths[0].first_region == 0
    assert paths[0].last_region == 1
    assert len(paths[0].transitions) == 1
    assert abs(paths[0].transitions[0].fraction - 0.5) < 1e-9
    assert path_meta["ambiguous_path_intervals"] == 0

    # 北京时间 2020-01-01 00:30 出发，01:00 穿界；两个原 split 各一条，应合计为 2。
    start_epoch = int(pd.Timestamp("2020-01-01 00:30", tz="Asia/Shanghai").timestamp())
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    _write_traj(train_path, start_epoch)
    _write_traj(test_path, start_epoch)
    hourly, meta = aggregate_boundary_crossings(
        [train_path, test_path], paths, num_regions=2, timezone_name="Asia/Shanghai"
    )
    at_one = hourly[hourly.timestamp == pd.Timestamp("2020-01-01 01:00")]
    assert int(at_one.loc[at_one.region_id == 0, "out_flow"].iloc[0]) == 2
    assert int(at_one.loc[at_one.region_id == 1, "in_flow"].iloc[0]) == 2
    assert int(at_one.loc[at_one.region_id == 0, "in_flow"].iloc[0]) == 0
    assert int(at_one.loc[at_one.region_id == 1, "out_flow"].iloc[0]) == 0
    assert meta["merged_trajectory_rows"] == 2
    assert meta["timezone"] == "Asia/Shanghai"


def test_interpolation_windows_positive_filter_and_train_only_normalization():
    timestamps = pd.date_range("2020-01-01 00:00", periods=48, freq="h")
    in_flow = np.ones(48, dtype=float)
    out_flow = np.full(48, 2.0)
    in_flow[[5, 6, 30]] = 0
    out_flow[[8, 31]] = 0
    # 保证训练区间存在非退化 Min-Max。
    in_flow[20] = 3
    hourly = pd.DataFrame(
        {
            "region_id": 0,
            "timestamp": timestamps,
            "in_flow": in_flow,
            "out_flow": out_flow,
        }
    )
    validation_start = pd.Timestamp("2020-01-02 00:00")
    interpolated, interpolation_meta = interpolate_internal_zeros(
        hourly, split_boundaries=[validation_start]
    )
    assert (interpolated[["in_flow", "out_flow"]].to_numpy() > 0).all()
    assert interpolation_meta["imputed_values"] == {"in_flow": 3, "out_flow": 2}

    full, train, validation, window_meta = build_sliding_windows(
        interpolated, validation_start=validation_start
    )
    assert len(full) == 25
    assert len(train) == 1
    assert len(validation) == 1
    assert window_meta["cross_boundary_windows_dropped"] == 23
    assert train.iloc[0].date == "2020-01-01"
    assert validation.iloc[0].date == "2020-01-02"

    # 验证集加入训练范围外大值，归一化参数仍必须保持训练 max=3，并显式计裁剪。
    validation = validation.copy()
    validation.at[validation.index[0], "in_flow"] = [10.0] * 24
    norm_train, norm_validation, norm_meta = normalize_train_validation(train, validation)
    assert norm_meta["train_min"] == 1.0
    assert norm_meta["train_max"] == 3.0
    assert norm_meta["validation_clipped_values"] == 24
    assert max(norm_validation.iloc[0].in_flow) == 1.0
    assert min(norm_train.iloc[0].in_flow) == 0.0


def test_no_source_validation_uses_all_positive_windows_for_training():
    timestamps = pd.date_range("2020-01-01 00:00", periods=48, freq="h")
    hourly = pd.DataFrame(
        {
            "region_id": 0,
            "timestamp": timestamps,
            "in_flow": np.linspace(1.0, 2.0, 48),
            "out_flow": np.linspace(2.0, 3.0, 48),
        }
    )
    full, train, validation, meta = build_sliding_windows(
        hourly, validation_start=None
    )
    assert len(full) == 25
    assert len(train) == 25
    assert validation.empty
    assert meta["source_validation_enabled"] is False
    assert meta["cross_boundary_windows_dropped"] == 0
    assert meta["regions_without_positive_validation_window"] == []

    norm_train, norm_validation, norm_meta = normalize_train_validation(train, validation)
    assert len(norm_train) == 25
    assert norm_validation.empty
    assert norm_meta["validation_total_values"] == 0


def test_craft_road_aggregation_counts_full_length_in_every_intersected_grid():
    # EPSG:3857 下约 1.8 km 的道路；先转 WGS84 WKT，以匹配 road.csv 契约。
    line_utm = gpd.GeoSeries([LineString([(100, 500), (1900, 500)])], crs=3857)
    line_wgs = line_utm.to_crs(4326).iloc[0]
    regions = gpd.GeoDataFrame(
        {"region_id": [0, 1]},
        geometry=[box(0, 0, 1000, 1000), box(1000, 0, 2000, 1000)],
        crs=3857,
    )
    roads = pd.DataFrame(
        {
            "road_id": [0],
            "from_node_id": [1],
            "to_node_id": [2],
            "road_type": ["trunk"],
            "road_type_id": [1],
            "length": [1800.0],
            "geometry": [line_wgs.wkt],
            "oneway": [True],
            "lanes": [2],
            "maxspeed": [60],
        }
    )
    counts, lengths, meta = aggregate_road_features(roads, regions, utm_epsg=3857)
    assert counts[:, 1].tolist() == [1, 1]
    assert lengths[:, 1].tolist() == [1800.0, 1800.0]
    assert "整条道路" in meta["aggregation_rule"]


def test_real_static_inputs_build_complete_45_dim_features(tmp_path):
    full = gpd.GeoDataFrame(
        {"full_grid_id": [0, 1]},
        geometry=[box(0, 0, 1000, 1000), box(1000, 0, 2000, 1000)],
        crs=3857,
    )
    selected = full.copy()
    selected.insert(0, "region_id", [0, 1])
    poi_points = gpd.GeoSeries([Point(500, 500), Point(500, 500)], crs=3857).to_crs(4326)
    pois = gpd.GeoDataFrame(
        {"@id": ["node/1", "node/1"], "name": ["cafe", "cafe"], "amenity": ["cafe", "cafe"]},
        geometry=poi_points,
        crs=4326,
    )
    _, poi_counts, poi_scores, poi_meta = prepare_poi_features(pois, full, selected, 3857)
    assert poi_counts[:, 4].tolist() == [1, 0]
    assert poi_scores[0, 4] > 0
    assert poi_meta["duplicate_poi_rows_removed"] == 1

    population_points = gpd.GeoSeries(
        [Point(500, 500), Point(1500, 500)],
        crs=3857,
    ).to_crs(4326)
    population_path = tmp_path / "population_2026.csv"
    pd.DataFrame(
        {
            "lon": population_points.x,
            "lat": population_points.y,
            "population": [10.0, 20.0],
        }
    ).to_csv(population_path, index=False)
    _, population, _ = prepare_population_features(population_path, full, selected, 3857)
    assert population.tolist() == [10.0, 20.0]

    road_counts = np.zeros((2, 8), dtype=int)
    road_lengths = np.zeros((2, 8), dtype=float)
    road_counts[:, 1] = 1
    road_lengths[:, 1] = 1800.0
    feature = build_grid_region_feature(
        full,
        selected,
        poi_counts,
        poi_scores,
        population,
        road_counts,
        road_lengths,
    )
    assert len(craft_feature_columns()) == 45
    assert feature[craft_feature_columns()].shape == (2, 45)
    assert np.isfinite(feature[craft_feature_columns()].to_numpy()).all()
