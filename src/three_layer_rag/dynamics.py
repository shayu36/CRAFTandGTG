"""Build leakage-aware three-layer temporal tensors from local GTG artifacts.

This module deliberately keeps the temporal route separate from static
GraphGPS.  It uses the timestamped GTG trajectories for road passage counts
and per-traversal speed/travel-time; Syntax tensors are *only* the existing
weighted Road -> Syntax pooling operator; Region tensors are the already
audited hourly boundary in/out flow.  No node ID is ever used as a dynamic
feature.

``train_label.csv`` / ``valid_label.csv`` are hour-of-day summaries (their
``time_index`` is 0..23), rather than dated hourly observations.  They can be
used as an explicitly opt-in sparse prior, but are not silently broadcast into
historical query sequences because that would leak an aggregate computed from
the full observation period.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from gtg_preprocessing.flow import SHANGHAI_UTC_OFFSET_SECONDS, iter_trajectory_chunks
from static_hierarchy.contracts import CityStaticHierarchy


DYNAMIC_FEATURE_VERSION = "three-layer-rag-dynamics-v1"
LAYER_NAMES = ("road", "syntax", "region")


@dataclass(frozen=True)
class HourlyThreeLayerDynamics:
    """Chronologically aligned temporal observations.

    Values have shape ``[H, N_layer, C_layer]``.  Road and Syntax use the
    channel order ``passage_count, speed_kmh, travel_time_seconds``; Region
    uses ``in_flow, out_flow``.  ``timestamps`` are Asia/Shanghai local civil
    timestamps stored as timezone-naive values, consistent with the existing
    GTG boundary-flow preprocessing output.
    """

    timestamps: pd.DatetimeIndex
    values: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any]

    def validate(self, hierarchy: CityStaticHierarchy) -> "HourlyThreeLayerDynamics":
        if not isinstance(self.timestamps, pd.DatetimeIndex) or len(self.timestamps) == 0:
            raise ValueError("严格模式: 动态序列 timestamps 必须为非空 DatetimeIndex")
        if self.timestamps.tz is not None:
            raise ValueError("严格模式: 动态 timestamps 必须是 Asia/Shanghai 本地朴素时间")
        expected = pd.date_range(self.timestamps[0], periods=len(self.timestamps), freq="h")
        if not self.timestamps.equals(expected):
            raise ValueError("严格模式: 动态 timestamps 必须连续且逐小时")
        if set(self.values) != set(LAYER_NAMES):
            raise ValueError("严格模式: 动态序列必须包含 road/syntax/region")
        expected_nodes = {
            "road": hierarchy.num_roads,
            "syntax": hierarchy.num_syntax,
            "region": hierarchy.num_regions,
        }
        expected_channels = {"road": 3, "syntax": 3, "region": 2}
        for layer in LAYER_NAMES:
            value = np.asarray(self.values[layer])
            if value.shape != (len(self.timestamps), expected_nodes[layer], expected_channels[layer]):
                raise ValueError(
                    f"严格模式: {layer} 动态序列应为 "
                    f"[{len(self.timestamps)},{expected_nodes[layer]},{expected_channels[layer]}]，"
                    f"实得 {tuple(value.shape)}"
                )
            if not np.isfinite(value).all() or (value < 0).any():
                raise ValueError(f"严格模式: {layer} 动态序列必须有限且非负")
        return self


def _parse_int_list(value: Any, field_name: str) -> np.ndarray:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        raise ValueError(f"严格模式: {field_name} 为空")
    text = str(value).strip()
    if not text:
        raise ValueError(f"严格模式: {field_name} 为空字符串")
    try:
        return np.fromiter((int(item) for item in text.split(",")), dtype=np.int64)
    except ValueError as exc:
        raise ValueError(f"严格模式: 无法解析 {field_name}={text[:120]!r}") from exc


def _local_hour_number(epoch_seconds: np.ndarray | float) -> np.ndarray | int:
    """Return the existing GTG pipeline's Asia/Shanghai local hour number."""

    return np.floor_divide(np.asarray(epoch_seconds, dtype=np.int64) + SHANGHAI_UTC_OFFSET_SECONDS, 3600)


def _timestamps_to_hour_numbers(timestamps: pd.DatetimeIndex) -> np.ndarray:
    return (timestamps.asi8 // 3_600_000_000_000).astype(np.int64, copy=False)


def _as_hourly_region_values(
    path: str | Path,
    *,
    num_regions: int,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Load the pre-existing Region boundary flow without re-aggregating it."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"严格模式: 缺失 Region hourly flow {path}")
    frame = pd.read_csv(path)
    required = {"region_id", "timestamp", "in_flow", "out_flow"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"严格模式: {path} 缺少列 {missing}")
    if frame.empty:
        raise ValueError(f"严格模式: {path} 为空")
    frame = frame.loc[:, ["region_id", "timestamp", "in_flow", "out_flow"]].copy()
    frame["region_id"] = pd.to_numeric(frame["region_id"], errors="coerce")
    if frame["region_id"].isna().any() or not np.array_equal(
        np.sort(frame["region_id"].dropna().unique()), np.arange(num_regions)
    ):
        raise ValueError("严格模式: Region flow 的 region_id 必须恰好是 0..N-1")
    frame["region_id"] = frame["region_id"].astype(np.int64)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    if frame["timestamp"].isna().any() or getattr(frame["timestamp"].dt, "tz", None) is not None:
        raise ValueError("严格模式: Region flow timestamp 必须为可解析的本地朴素时间")
    if frame.duplicated(["timestamp", "region_id"]).any():
        raise ValueError("严格模式: Region flow 含重复 timestamp/region_id")
    for name in ("in_flow", "out_flow"):
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
        if frame[name].isna().any() or not np.isfinite(frame[name].to_numpy()).all() or (frame[name] < 0).any():
            raise ValueError(f"严格模式: Region flow.{name} 必须有限且非负")
    timestamps = pd.DatetimeIndex(np.sort(frame["timestamp"].unique()))
    expected = pd.date_range(timestamps[0], periods=len(timestamps), freq="h")
    if not timestamps.equals(expected):
        raise ValueError("严格模式: Region flow timestamps 不连续，不能安全构造历史/未来窗口")
    expected_rows = len(timestamps) * num_regions
    if len(frame) != expected_rows:
        raise ValueError("严格模式: Region flow 没有覆盖每个 timestamp 的全部 Region")
    ordered = frame.set_index(["timestamp", "region_id"]).sort_index()
    full_index = pd.MultiIndex.from_product([timestamps, np.arange(num_regions)], names=["timestamp", "region_id"])
    ordered = ordered.reindex(full_index)
    if ordered[["in_flow", "out_flow"]].isna().any().any():
        raise ValueError("严格模式: Region flow 重排后存在缺失节点")
    values = ordered[["in_flow", "out_flow"]].to_numpy(dtype=np.float32).reshape(
        len(timestamps), num_regions, 2
    )
    return timestamps, values


def load_stable_road_lengths(
    road_csv: str | Path,
    hierarchy: CityStaticHierarchy,
) -> np.ndarray:
    """Resolve GTG ``link_id`` lengths into the hierarchy's fixed Road order."""

    road_csv = Path(road_csv)
    if not road_csv.exists():
        raise FileNotFoundError(f"严格模式: 缺失 GTG road.csv {road_csv}")
    frame = pd.read_csv(road_csv, usecols=["link_id", "length"])
    if len(frame) != hierarchy.num_roads:
        raise ValueError(
            f"严格模式: {road_csv} 道路行数={len(frame)}，但 hierarchy 有 {hierarchy.num_roads} 条 Road"
        )
    ids = pd.to_numeric(frame["link_id"], errors="coerce")
    lengths = pd.to_numeric(frame["length"], errors="coerce")
    if ids.isna().any() or lengths.isna().any() or not np.isfinite(lengths.to_numpy()).all() or (lengths <= 0).any():
        raise ValueError("严格模式: GTG road.csv 的 link_id/length 必须为有限正数")
    ids_i64 = ids.to_numpy(dtype=np.int64)
    expected_ids = np.arange(hierarchy.num_roads, dtype=np.int64)
    if not np.array_equal(np.sort(ids_i64), expected_ids):
        raise ValueError("严格模式: GTG link_id 必须稳定覆盖 0..M-1，不能重排 Road 节点")
    stable_ids = tuple(str(value) for value in expected_ids.tolist())
    if hierarchy.road_ids != stable_ids:
        raise ValueError("严格模式: static hierarchy road_ids 不等于 GTG 稳定 link_id 顺序")
    output = np.empty(hierarchy.num_roads, dtype=np.float32)
    output[ids_i64] = lengths.to_numpy(dtype=np.float32)
    return output


def _load_hour_of_day_label_prior(
    paths: Sequence[str | Path], *, num_roads: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Load optional hour-of-day ``speed_mean``/``dur_mean`` labels.

    The returned arrays are [24, M].  They are a descriptive aggregate, not a
    dated observation, hence callers must explicitly opt in before using them
    to fill sparse chronological values.
    """

    if len(paths) != 2:
        raise ValueError("严格模式: Road label 必须同时提供 train_label 与 valid_label")
    speed_sum = np.zeros((24, num_roads), dtype=np.float64)
    duration_sum = np.zeros((24, num_roads), dtype=np.float64)
    count = np.zeros((24, num_roads), dtype=np.int32)
    rows = 0
    for path_like in paths:
        path = Path(path_like)
        if not path.exists():
            raise FileNotFoundError(f"严格模式: 缺失 Road label {path}")
        frame = pd.read_csv(path)
        required = {"time_index", "rid", "dur_mean", "speed_mean"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise KeyError(f"严格模式: {path} 缺少列 {missing}")
        for name in required:
            frame[name] = pd.to_numeric(frame[name], errors="coerce")
        if frame[list(required)].isna().any().any():
            raise ValueError(f"严格模式: {path} Road label 含非数值")
        hours = frame["time_index"].to_numpy(dtype=np.int64)
        roads = frame["rid"].to_numpy(dtype=np.int64)
        durations = frame["dur_mean"].to_numpy(dtype=np.float64)
        speeds = frame["speed_mean"].to_numpy(dtype=np.float64)
        if (
            (hours < 0).any() or (hours >= 24).any() or (roads < 0).any() or (roads >= num_roads).any()
            or not np.isfinite(durations).all() or not np.isfinite(speeds).all()
            or (durations <= 0).any() or (speeds <= 0).any()
        ):
            raise ValueError(f"严格模式: {path} Road label time_index/rid/dur_mean/speed_mean 非法")
        np.add.at(speed_sum, (hours, roads), speeds)
        np.add.at(duration_sum, (hours, roads), durations)
        np.add.at(count, (hours, roads), 1)
        rows += len(frame)
    speed = np.divide(speed_sum, count, out=np.zeros_like(speed_sum), where=count > 0).astype(np.float32)
    duration = np.divide(duration_sum, count, out=np.zeros_like(duration_sum), where=count > 0).astype(np.float32)
    return speed, duration, {"label_rows": rows, "label_populated_hour_road_pairs": int((count > 0).sum())}


def aggregate_road_hourly_features(
    trajectory_paths: Sequence[str | Path],
    *,
    timestamps: pd.DatetimeIndex,
    road_lengths_m: np.ndarray,
    chunksize: int = 100_000,
    label_paths: Sequence[str | Path] | None = None,
    label_fill_mode: str = "none",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Aggregate passage/speed/travel-time into ``[H,M,3]`` without IDs.

    Passage count is assigned to the local hour at Road entry.  Traversal
    speed is ``length_m / dur_seconds * 3.6`` (km/h) and travel time is the
    observed ``dur_list`` duration.  Zero-duration traversals count as a
    passage but are excluded from the two mean channels.
    """

    if label_fill_mode not in {"none", "hour_of_day_prior"}:
        raise ValueError("严格模式: label_fill_mode 仅支持 none 或 hour_of_day_prior")
    lengths = np.asarray(road_lengths_m, dtype=np.float32).reshape(-1)
    if lengths.size == 0 or not np.isfinite(lengths).all() or (lengths <= 0).any():
        raise ValueError("严格模式: road_lengths_m 必须为有限正数 [M]")
    if len(timestamps) == 0:
        raise ValueError("严格模式: timestamps 不能为空")
    hour_numbers = _timestamps_to_hour_numbers(timestamps)
    first_hour, last_hour = int(hour_numbers[0]), int(hour_numbers[-1])
    if not np.array_equal(hour_numbers, np.arange(first_hour, last_hour + 1, dtype=np.int64)):
        raise ValueError("严格模式: Road 聚合要求连续逐小时 timestamps")
    hours, roads = len(timestamps), len(lengths)
    passage = np.zeros((hours, roads), dtype=np.float32)
    speed_sum = np.zeros((hours, roads), dtype=np.float32)
    travel_sum = np.zeros((hours, roads), dtype=np.float32)
    observed = np.zeros((hours, roads), dtype=np.float32)
    rows = 0
    entries = 0
    zero_duration_entries = 0
    outside_timeline_entries = 0

    for source_name, chunk in iter_trajectory_chunks(trajectory_paths, chunksize=chunksize):
        for row in chunk.itertuples(index=False):
            try:
                start_time = int(row.start_time)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"严格模式: {source_name} traj_id={row.traj_id} start_time 非 Unix 秒"
                ) from exc
            rids = _parse_int_list(row.rid_list, "rid_list")
            durations = _parse_int_list(row.dur_list, "dur_list")
            if len(rids) != len(durations) or len(rids) == 0:
                raise ValueError(
                    f"严格模式: {source_name} traj_id={row.traj_id} rid_list/dur_list 长度非法"
                )
            if (rids < 0).any() or (rids >= roads).any() or (durations < 0).any():
                raise ValueError(f"严格模式: {source_name} traj_id={row.traj_id} rid/dur 越界或为负")
            entry_seconds = start_time + np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(durations[:-1], dtype=np.int64)))
            entry_hours = _local_hour_number(entry_seconds).astype(np.int64, copy=False)
            slots = entry_hours - first_hour
            in_range = (slots >= 0) & (slots < hours)
            outside_timeline_entries += int((~in_range).sum())
            if in_range.any():
                slots_valid = slots[in_range]
                roads_valid = rids[in_range]
                durations_valid = durations[in_range]
                np.add.at(passage, (slots_valid, roads_valid), 1.0)
                positive = durations_valid > 0
                if positive.any():
                    ps = slots_valid[positive]
                    pr = roads_valid[positive]
                    pdur = durations_valid[positive].astype(np.float32)
                    np.add.at(travel_sum, (ps, pr), pdur)
                    np.add.at(speed_sum, (ps, pr), lengths[pr] * 3.6 / pdur)
                    np.add.at(observed, (ps, pr), 1.0)
                zero_duration_entries += int((durations_valid == 0).sum())
            entries += len(rids)
            rows += 1

    speed = np.divide(speed_sum, observed, out=np.zeros_like(speed_sum), where=observed > 0)
    travel_time = np.divide(travel_sum, observed, out=np.zeros_like(travel_sum), where=observed > 0)
    label_meta: dict[str, Any] = {
        "label_fill_mode": label_fill_mode,
        "label_prior_semantics": "not_used" if label_fill_mode == "none" else "hour_of_day_aggregate_prior",
    }
    if label_fill_mode == "hour_of_day_prior":
        if label_paths is None:
            raise ValueError("严格模式: hour_of_day_prior 必须提供 train_label/valid_label")
        prior_speed, prior_duration, prior_meta = _load_hour_of_day_label_prior(label_paths, num_roads=roads)
        local_hours = timestamps.hour.to_numpy(dtype=np.int64)
        missing = observed == 0
        prior_speed_full = prior_speed[local_hours]
        prior_duration_full = prior_duration[local_hours]
        usable = missing & (prior_speed_full > 0) & (prior_duration_full > 0)
        speed[usable] = prior_speed_full[usable]
        travel_time[usable] = prior_duration_full[usable]
        label_meta.update(prior_meta)
        label_meta["label_filled_hour_road_pairs"] = int(usable.sum())
        label_meta["leakage_notice"] = (
            "hour-of-day labels are full-period aggregates; use only when this prior is allowed "
            "by the experimental protocol, never as a causal historical observation"
        )
    values = np.stack((passage, speed, travel_time), axis=-1).astype(np.float32, copy=False)
    if not np.isfinite(values).all() or (values < 0).any():
        raise RuntimeError("严格模式: Road 动态聚合产生 NaN/Inf/负数")
    metadata = {
        "trajectory_rows": rows,
        "road_entries": entries,
        "zero_duration_road_entries": zero_duration_entries,
        "outside_region_timeline_entries": outside_timeline_entries,
        "passage_count_rule": "one count at the Asia/Shanghai local hour of each road entry",
        "speed_rule": "per-traversal length_m / dur_seconds * 3.6; arithmetic mean per hour/road",
        "travel_time_rule": "per-traversal dur_list seconds; arithmetic mean per hour/road",
        "road_channels": ["passage_count", "speed_kmh", "travel_time_seconds"],
        **label_meta,
    }
    return values, metadata


def weighted_parent_pool(
    child_values: np.ndarray,
    *,
    edge_index: torch.Tensor | np.ndarray,
    weight: torch.Tensor | np.ndarray,
    num_parents: int,
    name: str,
) -> np.ndarray:
    """Apply a static COO parent operator (row=parent, col=child) per hour."""

    child = np.asarray(child_values, dtype=np.float32)
    if child.ndim != 3 or child.shape[0] == 0 or child.shape[1] == 0 or child.shape[2] == 0:
        raise ValueError(f"严格模式: {name} child_values 必须为 [H,N,C]")
    edges = edge_index.detach().cpu().numpy() if isinstance(edge_index, torch.Tensor) else np.asarray(edge_index)
    weights = weight.detach().cpu().numpy() if isinstance(weight, torch.Tensor) else np.asarray(weight)
    if edges.shape[0] != 2 or edges.ndim != 2 or weights.shape != (edges.shape[1],):
        raise ValueError(f"严格模式: {name} edge_index/weight shape 非法")
    rows = edges[0].astype(np.int64, copy=False)
    cols = edges[1].astype(np.int64, copy=False)
    weights = weights.astype(np.float32, copy=False)
    if (
        rows.size == 0 or (rows < 0).any() or (rows >= num_parents).any()
        or (cols < 0).any() or (cols >= child.shape[1]).any()
        or not np.isfinite(weights).all() or (weights <= 0).any()
    ):
        raise ValueError(f"严格模式: {name} parent operator 非法")
    operator = sp.coo_matrix((weights, (rows, cols)), shape=(num_parents, child.shape[1])).tocsr()
    output = np.empty((child.shape[0], num_parents, child.shape[2]), dtype=np.float32)
    for channel in range(child.shape[2]):
        output[:, :, channel] = operator.dot(child[:, :, channel].T).T
    if not np.isfinite(output).all() or (output < 0).any():
        raise RuntimeError(f"严格模式: {name} 加权聚合产生 NaN/Inf/负数")
    return output


def build_hourly_three_layer_dynamics(
    *,
    hierarchy: CityStaticHierarchy,
    road_csv: str | Path,
    trajectory_paths: Sequence[str | Path],
    region_hourly_flow: str | Path,
    chunksize: int = 100_000,
    label_paths: Sequence[str | Path] | None = None,
    label_fill_mode: str = "none",
) -> HourlyThreeLayerDynamics:
    """Build the aligned real-data three-layer dynamic representation."""

    timestamps, region = _as_hourly_region_values(region_hourly_flow, num_regions=hierarchy.num_regions)
    lengths = load_stable_road_lengths(road_csv, hierarchy)
    road, road_meta = aggregate_road_hourly_features(
        trajectory_paths,
        timestamps=timestamps,
        road_lengths_m=lengths,
        chunksize=chunksize,
        label_paths=label_paths,
        label_fill_mode=label_fill_mode,
    )
    syntax = weighted_parent_pool(
        road,
        edge_index=hierarchy.road_to_syntax_edge_index,
        weight=hierarchy.road_to_syntax_weight,
        num_parents=hierarchy.num_syntax,
        name="road_to_syntax",
    )
    result = HourlyThreeLayerDynamics(
        timestamps=timestamps,
        values={"road": road, "syntax": syntax, "region": region},
        metadata={
            "format_version": DYNAMIC_FEATURE_VERSION,
            "city_id": hierarchy.city_id,
            "timezone": "Asia/Shanghai",
            "road_to_syntax_rule": "static weighted mean using hierarchy.road_to_syntax_weight",
            "region_rule": f"local {Path(region_hourly_flow).name} in_flow/out_flow",
            "road": road_meta,
        },
    )
    return result.validate(hierarchy)


def split_snapshot_starts(
    num_hours: int,
    *,
    history_length: int,
    value_length: int,
    stride_hours: int,
    split_ratios: Sequence[float] = (0.8, 0.1, 0.1),
) -> tuple[dict[str, list[int]], dict[str, int]]:
    """Return non-leaking target starts for history ``[t-H,t)`` and Value ``[t,t+V)``.

    The split is based on the *whole* future Value interval.  Windows crossing
    a train/valid boundary are dropped rather than assigned ambiguously.
    """

    if min(num_hours, history_length, value_length, stride_hours) <= 0:
        raise ValueError("严格模式: num_hours/history_length/value_length/stride_hours 必须为正")
    ratios = np.asarray(split_ratios, dtype=np.float64)
    if ratios.shape != (3,) or not np.isfinite(ratios).all() or (ratios <= 0).any() or not np.isclose(ratios.sum(), 1.0):
        raise ValueError("严格模式: split_ratios 必须是和为 1 的三个有限正数")
    train_end = max(1, int(math.floor(num_hours * ratios[0])))
    valid_end = min(num_hours - 1, train_end + max(1, int(math.floor(num_hours * ratios[1]))))
    if train_end >= valid_end:
        raise ValueError("严格模式: 时序长度不足以建立 train/valid 边界")
    result = {"train": [], "val": [], "test": []}
    for start in range(history_length, num_hours - value_length + 1, stride_hours):
        end = start + value_length
        if end <= train_end:
            result["train"].append(start)
        elif start >= train_end and end <= valid_end:
            result["val"].append(start)
        elif start >= valid_end:
            result["test"].append(start)
    if not any(result.values()):
        raise ValueError("严格模式: 当前时序长度无法产生完整 history/value snapshot")
    return result, {"train_end_hour_index": train_end, "valid_end_hour_index": valid_end}


def calendar_for_timestamp(timestamp: pd.Timestamp, *, holiday_dates: Iterable[str] = ()) -> dict[str, int]:
    """Return the model calendar condition for a future Value start timestamp."""

    stamp = pd.Timestamp(timestamp)
    if stamp.tzinfo is not None:
        raise ValueError("严格模式: calendar timestamp 必须是本地朴素时间")
    holidays = {str(pd.Timestamp(value).date()) for value in holiday_dates}
    return {
        "month": int(stamp.month),
        "weekday": int(stamp.weekday()),
        "start_hour": int(stamp.hour),
        "holiday": int(str(stamp.date()) in holidays),
        "time_of_day": float(stamp.hour),
    }


def temporal_window(values: np.ndarray, *, start: int, history_length: int, value_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert one chronological history/future pair to contract order [N,C,T]."""

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 3 or start < history_length or start + value_length > array.shape[0]:
        raise ValueError("严格模式: temporal window 超出动态序列范围")
    history = np.ascontiguousarray(array[start - history_length:start].transpose(1, 2, 0))
    value = np.ascontiguousarray(array[start:start + value_length].transpose(1, 2, 0))
    return torch.from_numpy(history), torch.from_numpy(value)


def normalize_hourly_dynamics(
    cities: Mapping[str, HourlyThreeLayerDynamics],
    *,
    train_end_indices: Mapping[str, int],
    mode: str = "log1p_zscore",
    normalizer: Mapping[str, Any] | None = None,
) -> tuple[dict[str, HourlyThreeLayerDynamics], dict[str, Any]]:
    """Fit/apply a source-train-only, per-layer/channel normalizer.

    ``log1p_zscore`` is deliberately fitted only on hours before each city's
    train boundary.  A supplied normalizer is validated and applied as-is,
    which lets a held-out target city use source-only statistics.
    """

    if mode not in {"none", "log1p_zscore"}:
        raise ValueError("严格模式: normalization mode 仅支持 none 或 log1p_zscore")
    if not cities:
        raise ValueError("严格模式: 没有可归一化的城市动态序列")
    channels = {"road": 3, "syntax": 3, "region": 2}
    if normalizer is None:
        if mode == "none":
            identity_layers = {
                layer: {"mean": [0.0] * count, "std": [1.0] * count}
                for layer, count in channels.items()
            }
            fitted: dict[str, Any] = {
                "format_version": DYNAMIC_FEATURE_VERSION,
                "mode": "none",
                "fit_cities": sorted(cities),
                "layers": identity_layers,
            }
        else:
            layer_stats: dict[str, dict[str, list[float]]] = {}
            for layer, count in channels.items():
                total = np.zeros(count, dtype=np.float64)
                squared = np.zeros(count, dtype=np.float64)
                elements = 0
                for city_id, dynamics in cities.items():
                    end = int(train_end_indices[city_id])
                    values = np.log1p(np.asarray(dynamics.values[layer][:end], dtype=np.float64))
                    total += values.sum(axis=(0, 1))
                    squared += np.square(values).sum(axis=(0, 1))
                    elements += values.shape[0] * values.shape[1]
                if elements <= 0:
                    raise ValueError(f"严格模式: {layer} 没有 source-train 归一化样本")
                mean = total / elements
                variance = np.maximum(squared / elements - np.square(mean), 0.0)
                std = np.sqrt(variance)
                std[std < 1e-6] = 1.0
                layer_stats[layer] = {"mean": mean.tolist(), "std": std.tolist()}
            fitted = {
                "format_version": DYNAMIC_FEATURE_VERSION,
                "mode": "log1p_zscore",
                "fit_cities": sorted(cities),
                "layers": layer_stats,
            }
    else:
        fitted = dict(normalizer)
        if fitted.get("format_version") != DYNAMIC_FEATURE_VERSION or fitted.get("mode") != mode:
            raise ValueError("严格模式: dynamic normalizer 版本或 mode 不匹配")
    normalized: dict[str, HourlyThreeLayerDynamics] = {}
    for city_id, dynamics in cities.items():
        values: dict[str, np.ndarray] = {}
        for layer, count in channels.items():
            source = np.asarray(dynamics.values[layer], dtype=np.float32)
            if mode == "none":
                target = source.copy()
            else:
                layer_stat = dict(fitted.get("layers", {}).get(layer, {}))
                mean = np.asarray(layer_stat.get("mean"), dtype=np.float32)
                std = np.asarray(layer_stat.get("std"), dtype=np.float32)
                if mean.shape != (count,) or std.shape != (count,) or (std <= 0).any() or not np.isfinite(mean).all() or not np.isfinite(std).all():
                    raise ValueError(f"严格模式: dynamic normalizer.{layer} 不合法")
                target = (np.log1p(source) - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
                target = target.astype(np.float32, copy=False)
            if not np.isfinite(target).all():
                raise RuntimeError(f"严格模式: {city_id}.{layer} 归一化后含 NaN/Inf")
            values[layer] = target
        metadata = dict(dynamics.metadata)
        metadata["normalization"] = {
            "mode": mode,
            "fit_cities": list(fitted.get("fit_cities", [])),
            "source_train_only": normalizer is None,
            "layers": dict(fitted.get("layers", {})),
        }
        normalized[city_id] = HourlyThreeLayerDynamics(dynamics.timestamps, values, metadata)
    return normalized, fitted
