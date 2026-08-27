"""GTG 轨迹到逐小时栅格边界穿越流量。

已确认的语义：车辆从区域 A 穿越到区域 B 时，A 的 out_flow +1，B 的
in_flow +1。道路内部的穿越时刻按该道路上累计几何长度占比乘以 dur_list
中的道路耗时估算；这是只有路段级耗时而没有逐点 GPS 时可复现的严格定义。
"""

from __future__ import annotations

import ast
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPoint, Point

from .contracts import FLOW_COLUMNS, TRAJECTORY_REQUIRED_COLUMNS

SHANGHAI_UTC_OFFSET_SECONDS = 8 * 3600


@dataclass(frozen=True)
class BoundaryTransition:
    fraction: float
    from_region: int | None
    to_region: int | None


@dataclass(frozen=True)
class RoadRegionPath:
    first_region: int | None
    last_region: int | None
    transitions: tuple[BoundaryTransition, ...]


def _parse_int_list(value, field_name: str) -> list[int]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        raise ValueError(f"严格模式: {field_name} 为空")
    text = str(value).strip()
    if not text:
        raise ValueError(f"严格模式: {field_name} 为空字符串")
    try:
        return [int(item) for item in text.split(",")]
    except ValueError as exc:
        raise ValueError(f"严格模式: 无法解析 {field_name}={text[:120]!r}") from exc


def iter_trajectory_chunks(
    paths: Sequence[str | Path], chunksize: int = 100_000
) -> Iterator[tuple[str, pd.DataFrame]]:
    """按给定顺序读取 GTG train/test；traj_id 仅在各文件内有意义。"""
    if len(paths) != 2:
        raise ValueError("严格模式: 必须同时提供 GTG 原始 train.csv 与 test.csv")
    for path_like in paths:
        path = Path(path_like)
        if not path.exists():
            raise FileNotFoundError(f"严格模式: 缺失轨迹文件 {path}")
        for chunk in pd.read_csv(path, sep=";", chunksize=chunksize):
            missing = TRAJECTORY_REQUIRED_COLUMNS - set(chunk.columns)
            if missing:
                raise ValueError(f"严格模式: {path} 缺少轨迹列 {sorted(missing)}")
            yield path.name, chunk


def collect_used_road_ids(
    paths: Sequence[str | Path], num_roads: int, chunksize: int = 100_000
) -> tuple[set[int], dict]:
    """扫描合并语料，收集真实被轨迹使用的 link_id。"""
    used: set[int] = set()
    rows = 0
    bad_list_lengths = 0
    for source_name, chunk in iter_trajectory_chunks(paths, chunksize=chunksize):
        for row in chunk.itertuples(index=False):
            rids = _parse_int_list(row.rid_list, "rid_list")
            durs = _parse_int_list(row.dur_list, "dur_list")
            if len(rids) != len(durs):
                bad_list_lengths += 1
                raise ValueError(
                    f"严格模式: {source_name} traj_id={row.traj_id} "
                    f"rid_list({len(rids)}) != dur_list({len(durs)})"
                )
            if any(rid < 0 or rid >= num_roads for rid in rids):
                bad = next(rid for rid in rids if rid < 0 or rid >= num_roads)
                raise ValueError(
                    f"严格模式: {source_name} traj_id={row.traj_id} 含越界 rid={bad}, "
                    f"道路数={num_roads}"
                )
            if any(dur < 0 for dur in durs):
                raise ValueError(f"严格模式: {source_name} traj_id={row.traj_id} 含负 dur")
            used.update(rids)
            rows += 1
    if not used:
        raise ValueError("严格模式: 合并 GTG train/test 后没有有效道路")
    return used, {
        "merged_trajectory_rows": rows,
        "used_road_count": len(used),
        "used_road_ratio": len(used) / num_roads,
        "bad_list_lengths": bad_list_lengths,
        "merge_reason": "GTG 原 train/test 是同一时段随机轨迹划分，流量聚合前合并",
    }


def _intersection_break_distances(line: LineString, geometry) -> list[float]:
    """把线与栅格边界交点转换为沿线距离。重叠线段取两端点。"""
    if geometry.is_empty:
        return []
    if isinstance(geometry, Point):
        return [float(line.project(geometry))]
    if isinstance(geometry, MultiPoint):
        return [float(line.project(point)) for point in geometry.geoms]
    if isinstance(geometry, LineString):
        return [
            float(line.project(Point(geometry.coords[0]))),
            float(line.project(Point(geometry.coords[-1]))),
        ]
    if isinstance(geometry, (MultiLineString, GeometryCollection)):
        out: list[float] = []
        for part in geometry.geoms:
            out.extend(_intersection_break_distances(line, part))
        return out
    return []


def build_road_region_paths(
    road_geometries_utm: Sequence[LineString],
    regions_utm: gpd.GeoDataFrame,
    ambiguous_interval_error_ratio: float = 1e-3,
) -> tuple[list[RoadRegionPath], dict]:
    """预计算每条有向道路沿几何方向依次穿过的区域及穿越比例。"""
    if "region_id" not in regions_utm.columns:
        raise ValueError("严格模式: regions_utm 缺少 region_id")
    expected = np.arange(len(regions_utm))
    if not np.array_equal(regions_utm["region_id"].to_numpy(), expected):
        raise ValueError("严格模式: region_id 必须为 0..N-1 且与行顺序一致")
    if regions_utm.crs is None or not regions_utm.crs.is_projected:
        raise ValueError("严格模式: 道路穿越计算必须使用投影坐标系")

    region_geoms = list(regions_utm.geometry)
    sindex = regions_utm.sindex
    paths: list[RoadRegionPath] = []
    ambiguous_intervals = 0
    total_intervals = 0
    zero_length_roads: list[int] = []

    for rid, line in enumerate(road_geometries_utm):
        if not isinstance(line, LineString) or line.is_empty or not line.is_valid:
            raise ValueError(f"严格模式: rid={rid} 不是有效 LineString")
        length = float(line.length)
        if length <= 0:
            zero_length_roads.append(rid)
            paths.append(RoadRegionPath(None, None, ()))
            continue

        candidate_idx = np.asarray(sindex.query(line, predicate="intersects"), dtype=int)
        breaks = [0.0, length]
        for idx in candidate_idx:
            breaks.extend(_intersection_break_distances(line, line.intersection(region_geoms[idx].boundary)))
        # 浮点交点可能重复；按道路长度比例设置稳定容差。
        tolerance = max(1e-7, length * 1e-10)
        breaks = sorted(min(length, max(0.0, value)) for value in breaks)
        unique_breaks = [breaks[0]]
        for value in breaks[1:]:
            if value - unique_breaks[-1] > tolerance:
                unique_breaks.append(value)
        if length - unique_breaks[-1] > tolerance:
            unique_breaks.append(length)

        states: list[tuple[float, float, int | None]] = []
        for left, right in zip(unique_breaks[:-1], unique_breaks[1:]):
            if right - left <= tolerance:
                continue
            midpoint = line.interpolate((left + right) / 2.0)
            candidates = np.asarray(sindex.query(midpoint, predicate="intersects"), dtype=int)
            covered = [idx for idx in candidates if region_geoms[idx].covers(midpoint)]
            total_intervals += 1
            if len(covered) > 1:
                ambiguous_intervals += 1
                # 仅用于形成可审计结果；超过阈值会在函数末尾严格报错。
                covered.sort()
            region = int(covered[0]) if covered else None
            if states and states[-1][2] == region:
                states[-1] = (states[-1][0], right, region)
            else:
                states.append((left, right, region))

        if not states:
            states = [(0.0, length, None)]
        transitions = tuple(
            BoundaryTransition(
                fraction=float(states[idx][1] / length),
                from_region=states[idx][2],
                to_region=states[idx + 1][2],
            )
            for idx in range(len(states) - 1)
            if states[idx][2] != states[idx + 1][2]
        )
        paths.append(RoadRegionPath(states[0][2], states[-1][2], transitions))

    if zero_length_roads:
        raise ValueError(
            f"严格模式: {len(zero_length_roads)} 条零长度道路，示例 {zero_length_roads[:10]}"
        )
    ambiguity_ratio = ambiguous_intervals / max(total_intervals, 1)
    if ambiguity_ratio > ambiguous_interval_error_ratio:
        raise ValueError(
            f"严格模式: 道路沿栅格边界导致的多区域歧义比例 {ambiguity_ratio:.6f} "
            f"> {ambiguous_interval_error_ratio:.6f}"
        )
    return paths, {
        "road_path_count": len(paths),
        "total_path_intervals": total_intervals,
        "ambiguous_path_intervals": ambiguous_intervals,
        "ambiguous_path_interval_ratio": ambiguity_ratio,
        "crossing_time_assumption": "每条道路内匀速，按投影几何累计长度占比分配 dur_list",
    }


def _local_hour_number(epoch_seconds: float, timezone_name: str) -> int:
    if timezone_name != "Asia/Shanghai":
        raise ValueError("严格模式: 当前 GTG 车辆流量只允许 timezone=Asia/Shanghai")
    return math.floor((epoch_seconds + SHANGHAI_UTC_OFFSET_SECONDS) / 3600.0)


def _record_transition(
    source: int | None,
    target: int | None,
    epoch_seconds: float,
    timezone_name: str,
    in_counter: Counter,
    out_counter: Counter,
) -> None:
    if source == target:
        return
    hour_number = _local_hour_number(epoch_seconds, timezone_name)
    if source is not None:
        out_counter[(int(source), hour_number)] += 1
    if target is not None:
        in_counter[(int(target), hour_number)] += 1


def aggregate_boundary_crossings(
    trajectory_paths: Sequence[str | Path],
    road_paths: Sequence[RoadRegionPath],
    num_regions: int,
    timezone_name: str = "Asia/Shanghai",
    chunksize: int = 100_000,
) -> tuple[pd.DataFrame, dict]:
    """合并两个原随机 split，统计逐小时区域边界流入/流出。"""
    in_counter: Counter = Counter()
    out_counter: Counter = Counter()
    row_count = 0
    zero_duration_count = 0
    transition_count = 0
    min_start = math.inf
    max_end = -math.inf

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
            if len(rids) != len(durations):
                raise ValueError(
                    f"严格模式: {source_name} traj_id={row.traj_id} "
                    f"rid_list({len(rids)}) != dur_list({len(durations)})"
                )
            if not rids:
                raise ValueError(f"严格模式: {source_name} traj_id={row.traj_id} 无道路")
            if any(rid < 0 or rid >= len(road_paths) for rid in rids):
                raise ValueError(f"严格模式: {source_name} traj_id={row.traj_id} rid 越界")
            if any(duration < 0 for duration in durations):
                raise ValueError(f"严格模式: {source_name} traj_id={row.traj_id} dur 为负")

            current_time = float(start_time)
            min_start = min(min_start, current_time)
            previous_region: int | None = None
            has_previous_road = False
            for rid, duration in zip(rids, durations):
                duration = int(duration)
                zero_duration_count += int(duration == 0)
                road_path = road_paths[rid]
                if has_previous_road and previous_region != road_path.first_region:
                    _record_transition(
                        previous_region,
                        road_path.first_region,
                        current_time,
                        timezone_name,
                        in_counter,
                        out_counter,
                    )
                    transition_count += 1
                for crossing in road_path.transitions:
                    crossing_time = current_time + duration * crossing.fraction
                    _record_transition(
                        crossing.from_region,
                        crossing.to_region,
                        crossing_time,
                        timezone_name,
                        in_counter,
                        out_counter,
                    )
                    transition_count += 1
                previous_region = road_path.last_region
                has_previous_road = True
                current_time += duration
            max_end = max(max_end, current_time)
            row_count += 1

    if row_count == 0:
        raise ValueError("严格模式: 合并轨迹为空")
    first_local_hour = _local_hour_number(min_start, timezone_name)
    last_local_hour = _local_hour_number(max_end, timezone_name)
    # 覆盖首末轨迹所在的完整本地自然日，未发生穿越的小时显式为真实 0。
    first_ts = pd.to_datetime(first_local_hour * 3600, unit="s").normalize()
    last_ts = pd.to_datetime(last_local_hour * 3600, unit="s").normalize() + pd.Timedelta(hours=23)
    timestamps = pd.date_range(first_ts, last_ts, freq="h")
    hour_numbers = (timestamps.astype("int64") // 3_600_000_000_000).to_numpy()

    records = []
    for region_id in range(num_regions):
        for timestamp, hour_number in zip(timestamps, hour_numbers):
            records.append(
                {
                    "region_id": region_id,
                    "timestamp": timestamp,
                    "in_flow": int(in_counter[(region_id, int(hour_number))]),
                    "out_flow": int(out_counter[(region_id, int(hour_number))]),
                }
            )
    hourly = pd.DataFrame.from_records(records)
    return hourly, {
        "timezone": timezone_name,
        "start_time_interpretation": "Unix seconds (UTC instant) converted to Asia/Shanghai local civil time",
        "merged_trajectory_rows": row_count,
        "zero_duration_road_entries": zero_duration_count,
        "region_transition_events": transition_count,
        "counted_in_events": int(sum(in_counter.values())),
        "counted_out_events": int(sum(out_counter.values())),
        "first_local_hour": first_ts.isoformat(),
        "last_local_hour": last_ts.isoformat(),
    }


def interpolate_internal_zeros(
    hourly: pd.DataFrame, split_boundaries: Iterable[pd.Timestamp] = ()
) -> tuple[pd.DataFrame, dict]:
    """把内部 0 段视为稀疏缺测，对 in/out 分别做线性插值。

    CRAFT 论文确认使用线性插值，但未公开脚本。本函数采用保守、显式的推断：
    只填充左右均有正值约束的内部 0，不外推首尾，并在数据划分边界处分段以免验证
    数据参与训练段插值。
    """
    required = {"region_id", "timestamp", "in_flow", "out_flow"}
    missing = required - set(hourly.columns)
    if missing:
        raise ValueError(f"严格模式: hourly 缺少列 {sorted(missing)}")
    out = hourly.sort_values(["region_id", "timestamp"]).copy()
    boundaries = sorted(pd.Timestamp(value) for value in split_boundaries)
    segment_id = np.zeros(len(out), dtype=np.int64)
    for boundary in boundaries:
        segment_id += (out["timestamp"].to_numpy() >= np.datetime64(boundary)).astype(np.int64)
    out["_segment_id"] = segment_id
    imputed = {"in_flow": 0, "out_flow": 0}
    for column in ("in_flow", "out_flow"):
        original = out[column].astype(float)

        def interpolate_group(series: pd.Series) -> pd.Series:
            return series.mask(series == 0).interpolate(method="linear", limit_area="inside").fillna(0.0)

        interpolated = out.assign(_value=original).groupby(
            ["region_id", "_segment_id"], sort=False
        )["_value"].transform(interpolate_group)
        imputed[column] = int(((original == 0) & (interpolated > 0)).sum())
        out[column] = interpolated.astype(float)
    out = out.drop(columns="_segment_id")
    return out, {
        "mode": "linear_internal_zero_runs",
        "evidence_status": "推断：CRAFT 论文/CSV 确认线性插值，但原始预处理脚本未公开",
        "zero_semantics": "内部 0 被视为稀疏缺测；首尾 0 不外推",
        "split_boundaries": [value.isoformat() for value in boundaries],
        "imputed_values": imputed,
    }


def build_sliding_windows(
    hourly: pd.DataFrame,
    seq_length: int = 24,
    validation_start: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """生成全部窗口，并按配置执行 CRAFT 全正筛选；验证集可关闭。"""
    if seq_length != 24:
        raise ValueError("严格模式: CRAFT 兼容输出要求 seq_length=24")
    if validation_start is not None:
        validation_start = pd.Timestamp(validation_start)
        if validation_start.tzinfo is not None:
            validation_start = validation_start.tz_localize(None)

    records = []
    for region_id, group in hourly.groupby("region_id", sort=True):
        group = group.sort_values("timestamp").reset_index(drop=True)
        expected = pd.date_range(group.timestamp.iloc[0], group.timestamp.iloc[-1], freq="h")
        if not np.array_equal(group.timestamp.to_numpy(), expected.to_numpy()):
            raise ValueError(f"严格模式: region_id={region_id} 小时序列不连续")
        in_values = group.in_flow.to_numpy(dtype=float)
        out_values = group.out_flow.to_numpy(dtype=float)
        for start_idx in range(0, len(group) - seq_length + 1):
            start = pd.Timestamp(group.timestamp.iloc[start_idx])
            end_exclusive = start + pd.Timedelta(hours=seq_length)
            records.append(
                {
                    "region_id": int(region_id),
                    "date": start.strftime("%Y-%m-%d"),
                    "weekday": int(start.weekday()),
                    "start_hour": int(start.hour),
                    "in_flow": in_values[start_idx : start_idx + seq_length].tolist(),
                    "out_flow": out_values[start_idx : start_idx + seq_length].tolist(),
                    "month": int(start.month),
                    "_start": start,
                    "_end_exclusive": end_exclusive,
                }
            )
    full = pd.DataFrame.from_records(records)
    if full.empty:
        raise ValueError("严格模式: 不足 24 个连续小时，无法生成滑窗")
    if validation_start is None:
        crosses = np.zeros(len(full), dtype=bool)
        train = full.copy()
        validation = full.iloc[0:0].copy()
    else:
        crosses = (full["_start"] < validation_start) & (full["_end_exclusive"] > validation_start)
        train = full[full["_end_exclusive"] <= validation_start].copy()
        validation = full[full["_start"] >= validation_start].copy()

    def positive_mask(frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(
            [
                bool(np.all(np.asarray(row.in_flow) > 0) and np.all(np.asarray(row.out_flow) > 0))
                for row in frame.itertuples(index=False)
            ],
            dtype=bool,
        )

    train_before = len(train)
    validation_before = len(validation)
    train = train[positive_mask(train)].copy()
    if not validation.empty:
        validation = validation[positive_mask(validation)].copy()
    if train.empty:
        raise ValueError("严格模式: 48 值全正筛选后 GTG 训练窗口为空")
    if validation_start is not None and validation.empty:
        raise ValueError("严格模式: 48 值全正筛选后 GTG 验证窗口为空")
    train_regions = sorted(train.region_id.unique().astype(int).tolist())
    validation_regions = sorted(validation.region_id.unique().astype(int).tolist())
    all_regions = sorted(full.region_id.unique().astype(int).tolist())
    report = {
        "seq_length": seq_length,
        "source_validation_enabled": validation_start is not None,
        "validation_start_local": validation_start.isoformat() if validation_start is not None else None,
        "full_windows": len(full),
        "cross_boundary_windows_dropped": int(crosses.sum()),
        "train_windows_before_positive_filter": train_before,
        "train_windows_after_positive_filter": len(train),
        "validation_windows_before_positive_filter": validation_before,
        "validation_windows_after_positive_filter": len(validation),
        "train_positive_filter_drop_ratio": 1.0 - len(train) / max(train_before, 1),
        "validation_positive_filter_drop_ratio": (
            1.0 - len(validation) / validation_before if validation_before else 0.0
        ),
        "regions_without_positive_train_window": sorted(set(all_regions) - set(train_regions)),
        "regions_without_positive_validation_window": (
            sorted(set(all_regions) - set(validation_regions))
            if validation_start is not None else []
        ),
        "split_semantics": (
            "GTG 全部合并窗口用于训练；不设源域验证集；最终测试城市来自 CRAFT"
            if validation_start is None
            else "GTG test 文件供 CRAFT 训练早停验证；最终测试城市来自 CRAFT"
        ),
    }
    return (
        full[list(FLOW_COLUMNS)].copy(),
        train[list(FLOW_COLUMNS)].copy(),
        validation[list(FLOW_COLUMNS)].copy(),
        report,
    )


def normalize_train_validation(
    train: pd.DataFrame, validation: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """按当前仓库对 CRAFT 缺失 norm 文件的重建方式做逐城市全局 Min-Max。"""
    train_values = np.concatenate(
        [
            np.concatenate(train["in_flow"].map(np.asarray).tolist()),
            np.concatenate(train["out_flow"].map(np.asarray).tolist()),
        ]
    ).astype(float)
    if not np.all(np.isfinite(train_values)):
        raise ValueError("严格模式: GTG 训练流量含 NaN/Inf")
    low = float(train_values.min())
    high = float(train_values.max())
    if high <= low:
        raise ValueError(f"严格模式: GTG 训练归一化区间退化 low={low}, high={high}")

    def apply(frame: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
        result = frame.copy()
        clipped = 0
        total = 0
        for column in ("in_flow", "out_flow"):
            arrays = []
            for value in result[column]:
                normalized = (np.asarray(value, dtype=float) - low) / (high - low)
                clipped += int(np.sum((normalized < 0) | (normalized > 1)))
                total += normalized.size
                arrays.append(np.clip(normalized, 0.0, 1.0).tolist())
            result[column] = arrays
        return result, clipped, total

    norm_train, train_clipped, train_total = apply(train)
    norm_validation, validation_clipped, validation_total = apply(validation)
    return norm_train, norm_validation, {
        "mode": "per_city_global_minmax",
        "fit_scope": "GTG source city positive training windows, in_flow/out_flow jointly",
        "evidence_status": (
            "CRAFT 读取代码只确认输入为 [0,1]；此 Min-Max 是本仓库对缺失原预处理脚本的无泄漏重建"
        ),
        "train_min": low,
        "train_max": high,
        "train_clipped_values": train_clipped,
        "train_total_values": train_total,
        "validation_clipped_values": validation_clipped,
        "validation_total_values": validation_total,
    }


def serialize_flow_lists(frame: pd.DataFrame) -> pd.DataFrame:
    """转成 CRAFT 的 CSV 字符串列表表示。"""
    out = frame[list(FLOW_COLUMNS)].copy()
    for column in ("in_flow", "out_flow"):
        out[column] = out[column].map(lambda value: str([float(item) for item in value]))
    return out
