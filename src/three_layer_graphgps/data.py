"""第二阶段 GraphGPS 的三层 cache 与 source Region 标签加载。"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
from pathlib import Path
import string
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from static_hierarchy.contracts import CityStaticHierarchy
from static_hierarchy.data import load_city_static_hierarchy

from .spectral_lap_pe import (
    HierarchyLaplacianPE,
    pe_graph_hash,
    prepare_hierarchy_lappe,
)


SPECTRAL_FEATURE_VERSION = "three-layer-spectral-features-v1"
STATIC_FEATURE_VERSION = "three-layer-start-road-v2"


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
    seq_length: int = 24,
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
            seq_length=seq_length,
            split_ratios=split_ratios,
        )
    return GraphGPSCityData(hierarchy=hierarchy, posenc=posenc, targets=targets)


def checkpoint_fingerprint(path: str | Path) -> str:
    """Return a content SHA-256 fingerprint for a trained checkpoint."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"严格模式: checkpoint 不存在 {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cpu_tensor(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"严格模式: {name} 必须为 Tensor")
    if not torch.isfinite(value).all():
        raise ValueError(f"严格模式: {name} 含 NaN/Inf")
    return value.detach().cpu().contiguous()


def export_spectral_features(
    path: str | Path,
    *,
    data: GraphGPSCityData,
    output: Mapping[str, torch.Tensor],
    checkpoint_sha256: str,
) -> Path:
    """Write stable three-layer low/high features for Stage 3 consumers."""

    hierarchy, posenc = data.hierarchy, data.posenc
    if not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64:
        raise ValueError("严格模式: checkpoint_sha256 必须为 64 位 SHA-256")
    if any(character not in string.hexdigits for character in checkpoint_sha256):
        raise ValueError("严格模式: checkpoint_sha256 含非十六进制字符")
    if hierarchy.metadata.get("feature_version") != STATIC_FEATURE_VERSION:
        raise ValueError("严格模式: 频率特征只能从 START v2 static cache 导出")
    required = (
        "H_road",
        "H_road_low",
        "H_road_high",
        "H_syntax",
        "H_syntax_low",
        "H_syntax_high",
        "H_region",
        "H_region_low",
        "H_region_high",
        "road_low_coefficients",
        "syntax_low_coefficients",
        "region_low_coefficients",
    )
    missing = sorted(set(required) - set(output))
    if missing:
        raise KeyError(f"严格模式: 模型输出缺少频率字段 {missing}")

    layer_nodes = {
        "road": hierarchy.num_roads,
        "syntax": hierarchy.num_syntax,
        "region": hierarchy.num_regions,
    }
    for layer, num_nodes in layer_nodes.items():
        mixed = output[f"H_{layer}"]
        low = output[f"H_{layer}_low"]
        high = output[f"H_{layer}_high"]
        if mixed.ndim != 2 or mixed.shape[0] != num_nodes:
            raise ValueError(f"严格模式: H_{layer} 节点数与静态图不一致")
        if low.shape != mixed.shape or high.shape != mixed.shape:
            raise ValueError(f"严格模式: {layer} low/high shape 与 mixed 不一致")
        if not torch.allclose(mixed, low + high, atol=1e-5, rtol=1e-5):
            raise ValueError(f"严格模式: {layer} low/high 无法重构 mixed")

    road_ids = tuple(hierarchy.road_ids)
    if len(road_ids) != hierarchy.num_roads:
        raise ValueError("严格模式: road_ids 与 Road tensor 行数不一致")
    payload: dict[str, Any] = {
        "format_version": SPECTRAL_FEATURE_VERSION,
        "city_id": hierarchy.city_id,
        "checkpoint_fingerprint": checkpoint_sha256,
        "static_feature_version": STATIC_FEATURE_VERSION,
        "road_ids": road_ids,
        "syntax_ids": torch.arange(hierarchy.num_syntax, dtype=torch.long),
        "region_ids": torch.arange(hierarchy.num_regions, dtype=torch.long),
        "road_graph_hash": pe_graph_hash(hierarchy.road_edge_index, hierarchy.num_roads),
        "syntax_graph_hash": pe_graph_hash(hierarchy.syntax_edge_index, hierarchy.num_syntax),
        "region_graph_hash": pe_graph_hash(hierarchy.region_edge_index, hierarchy.num_regions),
    }
    for layer in layer_nodes:
        expected_graph_hash = payload[f"{layer}_graph_hash"]
        if getattr(posenc, layer).metadata.get("graph_hash") != expected_graph_hash:
            raise ValueError(f"严格模式: {layer} LapPE graph hash 与静态图不一致")
        payload[f"H_{layer}"] = _cpu_tensor(output[f"H_{layer}"], f"H_{layer}")
        payload[f"H_{layer}_low"] = _cpu_tensor(output[f"H_{layer}_low"], f"H_{layer}_low")
        payload[f"H_{layer}_high"] = _cpu_tensor(output[f"H_{layer}_high"], f"H_{layer}_high")
        payload[f"{layer}_eigvals"] = _cpu_tensor(
            getattr(posenc, layer).eigvals, f"{layer}_eigvals"
        )
        coefficients = output[f"{layer}_low_coefficients"]
        payload[f"{layer}_num_low_modes"] = int(coefficients.shape[0])

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def load_spectral_features(
    path: str | Path,
    *,
    hierarchy: CityStaticHierarchy,
    expected_checkpoint_fingerprint: str,
) -> dict[str, Any]:
    """Load an export and reject graph, node-order, or checkpoint mismatches."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format_version") != SPECTRAL_FEATURE_VERSION:
        raise ValueError("严格模式: 不是 three-layer-spectral-features-v1 文件")
    if payload.get("static_feature_version") != STATIC_FEATURE_VERSION:
        raise ValueError("严格模式: spectral/static feature version 不匹配")
    if payload.get("checkpoint_fingerprint") != expected_checkpoint_fingerprint:
        raise ValueError("严格模式: spectral feature checkpoint fingerprint 不匹配")
    if payload.get("city_id") != hierarchy.city_id:
        raise ValueError("严格模式: spectral feature city_id 不匹配")
    if tuple(payload.get("road_ids", ())) != tuple(hierarchy.road_ids):
        raise ValueError("严格模式: road_ids 与静态图节点顺序不匹配")
    expected_ids = {
        "syntax_ids": torch.arange(hierarchy.num_syntax, dtype=torch.long),
        "region_ids": torch.arange(hierarchy.num_regions, dtype=torch.long),
    }
    for name, expected in expected_ids.items():
        actual = payload.get(name)
        if not isinstance(actual, torch.Tensor) or actual.dtype != torch.long or not torch.equal(actual, expected):
            raise ValueError(f"严格模式: {name} 与静态图节点顺序不匹配")
    expected_hashes = {
        "road_graph_hash": pe_graph_hash(hierarchy.road_edge_index, hierarchy.num_roads),
        "syntax_graph_hash": pe_graph_hash(hierarchy.syntax_edge_index, hierarchy.num_syntax),
        "region_graph_hash": pe_graph_hash(hierarchy.region_edge_index, hierarchy.num_regions),
    }
    for name, expected in expected_hashes.items():
        if payload.get(name) != expected:
            raise ValueError(f"严格模式: {name} 与当前静态图不匹配")
    for layer, num_nodes in (
        ("road", hierarchy.num_roads),
        ("syntax", hierarchy.num_syntax),
        ("region", hierarchy.num_regions),
    ):
        mixed = payload.get(f"H_{layer}")
        low = payload.get(f"H_{layer}_low")
        high = payload.get(f"H_{layer}_high")
        if not all(isinstance(value, torch.Tensor) for value in (mixed, low, high)):
            raise KeyError(f"严格模式: 导出缺少 {layer} mixed/low/high")
        if mixed.ndim != 2 or mixed.shape[0] != num_nodes or low.shape != mixed.shape or high.shape != mixed.shape:
            raise ValueError(f"严格模式: 导出 {layer} tensor shape 非法")
        if not all(torch.isfinite(value).all() for value in (mixed, low, high)):
            raise ValueError(f"严格模式: 导出 {layer} tensor 含 NaN/Inf")
        if not torch.allclose(mixed, low + high, atol=1e-5, rtol=1e-5):
            raise ValueError(f"严格模式: 导出 {layer} low/high 无法重构 mixed")
        eigvals = payload.get(f"{layer}_eigvals")
        num_low_modes = payload.get(f"{layer}_num_low_modes")
        if not isinstance(eigvals, torch.Tensor) or eigvals.ndim != 3:
            raise ValueError(f"严格模式: 导出 {layer}_eigvals shape 非法")
        if eigvals.shape[0] != num_nodes or eigvals.shape[2] != 1 or not torch.isfinite(eigvals).all():
            raise ValueError(f"严格模式: 导出 {layer}_eigvals 节点数/finite 非法")
        if (
            not isinstance(num_low_modes, int)
            or num_low_modes <= 0
            or num_low_modes > eigvals.shape[1]
        ):
            raise ValueError(f"严格模式: 导出 {layer}_num_low_modes 非法")
    return payload
