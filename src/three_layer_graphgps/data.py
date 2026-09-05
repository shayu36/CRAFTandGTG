"""第二阶段 GraphGPS 的三层 cache 与 source Region 标签加载。"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch

from static_hierarchy.contracts import CityStaticHierarchy
from static_hierarchy.data import load_city_static_hierarchy

from .spectral_lap_pe import HierarchyLaplacianPE, prepare_hierarchy_lappe


@dataclass(frozen=True)
class RegionFlowTargets:
    city_id: str
    split: str
    region_ids: torch.Tensor
    values: torch.Tensor
    observation_count: torch.Tensor

    def to(self, device: torch.device | str) -> "RegionFlowTargets":
        return RegionFlowTargets(
            self.city_id,
            self.split,
            self.region_ids.to(device),
            self.values.to(device),
            self.observation_count.to(device),
        )


@dataclass(frozen=True)
class GraphGPSCityData:
    hierarchy: CityStaticHierarchy
    posenc: HierarchyLaplacianPE
    targets: Mapping[str, RegionFlowTargets] | None = None


def load_stage2_hierarchy(cache_dir: str | Path, city: str) -> CityStaticHierarchy:
    """只接受第一阶段 START v2 三层 cache，并给缺失 road_x 明确错误。"""

    try:
        hierarchy = load_city_static_hierarchy(
            cache_dir,
            city,
            expected_feature_version="three-layer-start-road-v2",
        )
    except KeyError as exc:
        if "road_x" in str(exc):
            raise KeyError(f"Missing `road_x` in three-layer graph cache for city={city}.") from exc
        raise
    required = {
        "road_x": hierarchy.road_x,
        "road_edge_index": hierarchy.road_edge_index,
        "syntax_x": hierarchy.syntax_x,
        "syntax_edge_index": hierarchy.syntax_edge_index,
        "region_x": hierarchy.region_x,
        "region_edge_index": hierarchy.region_edge_index,
        "road_to_syntax": hierarchy.road_to_syntax_edge_index,
        "syntax_to_region": hierarchy.syntax_to_region_edge_index,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise KeyError(f"严格模式: three-layer graph cache 缺少字段 {missing}")
    return hierarchy


def _parse_flow_vector(value: object, name: str, row_index: int) -> np.ndarray:
    try:
        parsed = ast.literal_eval(value) if isinstance(value, str) else value
        result = np.asarray(parsed, dtype=np.float32)
    except (SyntaxError, ValueError, TypeError) as exc:
        raise ValueError(f"严格模式: 第 {row_index} 行 {name} 无法解析") from exc
    if result.shape != (24,) or not np.isfinite(result).all():
        raise ValueError(f"严格模式: 第 {row_index} 行 {name} 应为有限 [24]")
    return result


def _aggregate_targets(
    frame: pd.DataFrame,
    *,
    city: str,
    split: str,
    num_regions: int,
) -> RegionFlowTargets:
    region_ids, values, counts = [], [], []
    for region_id, group in frame.groupby("region_id", sort=True):
        numeric = float(region_id)
        if not np.isfinite(numeric) or numeric != int(numeric):
            raise ValueError(f"严格模式: {city} {split} region_id={region_id!r} 不是整数")
        region_id = int(numeric)
        if region_id < 0 or region_id >= num_regions:
            raise ValueError(f"严格模式: {city} {split} region_id={region_id} 越界")
        stacked = np.stack(group["_flow"].tolist(), axis=0)
        mean_value = stacked.mean(axis=0)
        if mean_value.shape != (48,) or not np.isfinite(mean_value).all():
            raise ValueError(f"严格模式: {city} {split} Region 标签 shape/finite 错误")
        region_ids.append(region_id)
        values.append(mean_value)
        counts.append(len(group))
    if not region_ids:
        raise ValueError(f"严格模式: {city} {split} 没有 Region 流量标签")
    return RegionFlowTargets(
        city_id=city,
        split=split,
        region_ids=torch.tensor(region_ids, dtype=torch.long),
        values=torch.tensor(np.stack(values), dtype=torch.float32),
        observation_count=torch.tensor(counts, dtype=torch.long),
    )


def load_source_region_flow_splits(
    norm_flow_root: str | Path,
    city: str,
    *,
    num_regions: int,
    seq_length: int = 24,
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, RegionFlowTargets]:
    """按唯一 ``date + start_hour`` 时间键切分 source train 窗口。

    仓库当前三个 source 只有 ``norm_train_len_24.csv``，因此这里不伪造
    valid/test 文件，也不把 target 动态数据引入训练。
    """

    if seq_length != 24:
        raise ValueError("严格模式: 当前 Region 预测头仅支持 seq_length=24")
    ratios = np.asarray(split_ratios, dtype=np.float64)
    if ratios.shape != (3,) or not np.isfinite(ratios).all() or (ratios <= 0).any():
        raise ValueError("严格模式: split_ratios 必须是三个有限正数")
    if not np.isclose(ratios.sum(), 1.0, atol=1e-8):
        raise ValueError("严格模式: split_ratios 总和必须为 1")
    path = Path(norm_flow_root) / city / f"norm_train_len_{seq_length}.csv"
    if not path.exists():
        raise FileNotFoundError(f"严格模式: 缺少 source flow 文件 {path}")
    frame = pd.read_csv(path)
    required = {"region_id", "date", "start_hour", "in_flow", "out_flow"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"严格模式: {path} 缺少列 {missing}")
    if frame.empty:
        raise ValueError(f"严格模式: source flow 文件为空 {path}")
    parsed = []
    for row_index, row in frame.iterrows():
        in_flow = _parse_flow_vector(row["in_flow"], "in_flow", int(row_index))
        out_flow = _parse_flow_vector(row["out_flow"], "out_flow", int(row_index))
        parsed.append(np.concatenate([in_flow, out_flow], axis=0))
    frame["_flow"] = parsed
    dates = pd.to_datetime(frame["date"], errors="coerce")
    hours = pd.to_numeric(frame["start_hour"], errors="coerce")
    if dates.isna().any() or hours.isna().any() or (hours < 0).any() or (hours > 23).any():
        raise ValueError(f"严格模式: {path} date/start_hour 非法")
    frame["_time_key"] = dates + pd.to_timedelta(hours, unit="h")
    unique_times = np.sort(frame["_time_key"].unique())
    if len(unique_times) < 3:
        raise ValueError(f"严格模式: {city} 唯一时间窗口不足 3 个，无法 train/valid/test")
    train_end = max(1, int(np.floor(len(unique_times) * ratios[0])))
    valid_size = max(1, int(np.floor(len(unique_times) * ratios[1])))
    valid_end = min(len(unique_times) - 1, train_end + valid_size)
    if train_end >= valid_end:
        raise ValueError(f"严格模式: {city} 时间窗口不足以形成 validation")
    time_sets = {
        "train": set(unique_times[:train_end]),
        "valid": set(unique_times[train_end:valid_end]),
        "test": set(unique_times[valid_end:]),
    }
    result = {}
    for split, keys in time_sets.items():
        subset = frame[frame["_time_key"].isin(keys)]
        result[split] = _aggregate_targets(
            subset,
            city=city,
            split=split,
            num_regions=num_regions,
        )
    return result


def prepare_city_data(
    *,
    city: str,
    hierarchy_cache_dir: str | Path,
    lappe_cache_dir: str | Path | None,
    road_k: int,
    syntax_k: int,
    region_k: int,
    normalization: str = "sym",
    require_targets: bool,
    norm_flow_root: str | Path | None = None,
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> GraphGPSCityData:
    hierarchy = load_stage2_hierarchy(hierarchy_cache_dir, city)
    posenc = prepare_hierarchy_lappe(
        hierarchy,
        road_k=road_k,
        syntax_k=syntax_k,
        region_k=region_k,
        normalization=normalization,
        cache_dir=lappe_cache_dir,
    )
    targets = None
    if require_targets:
        if norm_flow_root is None:
            raise ValueError("严格模式: source labels 需要 norm_flow_root")
        targets = load_source_region_flow_splits(
            norm_flow_root,
            city,
            num_regions=hierarchy.num_regions,
            split_ratios=split_ratios,
        )
    return GraphGPSCityData(hierarchy=hierarchy, posenc=posenc, targets=targets)
