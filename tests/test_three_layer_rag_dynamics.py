from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from three_layer_rag.dynamics import (
    build_hourly_three_layer_dynamics,
    normalize_hourly_dynamics,
    split_snapshot_starts,
    temporal_window,
)


def _hierarchy():
    return SimpleNamespace(
        city_id="toy",
        num_roads=2,
        num_syntax=1,
        num_regions=1,
        road_ids=("0", "1"),
        road_to_syntax_edge_index=torch.tensor([[0, 0], [0, 1]], dtype=torch.long),
        road_to_syntax_weight=torch.tensor([0.25, 0.75]),
    )


def _epoch(local_time: str) -> int:
    return int(pd.Timestamp(local_time, tz="Asia/Shanghai").timestamp())


def _write_inputs(tmp_path):
    road_path = tmp_path / "road.csv"
    pd.DataFrame({"link_id": [0, 1], "length": [100.0, 200.0]}).to_csv(road_path, index=False)
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    pd.DataFrame({
        "traj_id": [0],
        "start_time": [_epoch("2020-01-01 00:00:00")],
        "rid_list": ["0,1"],
        "dur_list": ["10,20"],
    }).to_csv(train_path, sep=";", index=False)
    pd.DataFrame({
        "traj_id": [1],
        "start_time": [_epoch("2020-01-01 01:00:00")],
        "rid_list": ["1"],
        "dur_list": ["0"],
    }).to_csv(test_path, sep=";", index=False)
    timestamps = pd.date_range("2020-01-01", periods=48, freq="h")
    region_path = tmp_path / "region.csv"
    pd.DataFrame({
        "region_id": np.zeros(len(timestamps), dtype=int),
        "timestamp": timestamps,
        "in_flow": np.arange(len(timestamps), dtype=float),
        "out_flow": np.arange(len(timestamps), dtype=float) + 1.0,
    }).to_csv(region_path, index=False)
    return road_path, train_path, test_path, region_path


def test_real_dynamic_builder_keeps_road_order_and_weighted_syntax_pool(tmp_path):
    road_path, train_path, test_path, region_path = _write_inputs(tmp_path)
    dynamics = build_hourly_three_layer_dynamics(
        hierarchy=_hierarchy(),
        road_csv=road_path,
        trajectory_paths=[train_path, test_path],
        region_hourly_flow=region_path,
        chunksize=1,
    )
    road = dynamics.values["road"]
    syntax = dynamics.values["syntax"]
    region = dynamics.values["region"]
    # 00:00 receives road 0 then road 1.  The latter starts 10 seconds later,
    # so both use the fixed, zero-based hierarchy Road order.
    assert road.shape == (48, 2, 3)
    assert road[0, :, 0].tolist() == [1.0, 1.0]
    assert np.allclose(road[0, :, 1], [36.0, 36.0])
    assert np.allclose(road[0, :, 2], [10.0, 20.0])
    # Syntax must retain the cached 0.25/0.75 mean-pooling weights.
    assert np.allclose(syntax[0, 0], 0.25 * road[0, 0] + 0.75 * road[0, 1])
    # A zero-duration observation still contributes one passage but no speed/TT.
    assert road[1, 1].tolist() == [1.0, 0.0, 0.0]
    assert region.shape == (48, 1, 2)
    assert region[3, 0].tolist() == [3.0, 4.0]


def test_snapshots_do_not_cross_split_boundary_and_keep_history_value_disjoint(tmp_path):
    road_path, train_path, test_path, region_path = _write_inputs(tmp_path)
    raw = build_hourly_three_layer_dynamics(
        hierarchy=_hierarchy(),
        road_csv=road_path,
        trajectory_paths=[train_path, test_path],
        region_hourly_flow=region_path,
        chunksize=2,
    )
    starts, boundaries = split_snapshot_starts(
        48, history_length=4, value_length=4, stride_hours=4, split_ratios=(0.5, 0.25, 0.25)
    )
    assert all(start + 4 <= boundaries["train_end_hour_index"] for start in starts["train"])
    assert all(start >= boundaries["train_end_hour_index"] for start in starts["val"])
    assert all(start >= boundaries["valid_end_hour_index"] for start in starts["test"])
    normalized, normalizer = normalize_hourly_dynamics(
        {"toy": raw},
        train_end_indices={"toy": boundaries["train_end_hour_index"]},
        mode="log1p_zscore",
    )
    assert normalizer["mode"] == "log1p_zscore"
    start = starts["train"][0]
    source = normalized["toy"].values["road"]
    history, value = temporal_window(source, start=start, history_length=4, value_length=4)
    assert history.shape == value.shape == (2, 3, 4)
    # The first Value hour directly follows the final history hour.  Sparse
    # traffic can make their *values* equal, so validate their source indices
    # and independent backing tensors instead of demanding unequal values.
    assert torch.equal(history[:, :, -1], torch.from_numpy(source[start - 1].copy()))
    assert torch.equal(value[:, :, 0], torch.from_numpy(source[start].copy()))
    assert history.data_ptr() != value.data_ptr()
