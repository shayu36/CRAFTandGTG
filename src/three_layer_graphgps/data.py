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
    LAPPE_VERSION,
    pe_graph_hash,
    prepare_hierarchy_lappe,
    to_undirected_edge_index_with_weight,
    weighted_pe_graph_hash,
)


SPECTRAL_FEATURE_VERSION = "three-layer-joint-graphgps-weighted-spectral-features-v3"
STATIC_FEATURE_VERSION = "three-layer-start-road-v2"


RELATION_NAMES = (
    "road_intra",
    "syntax_intra",
    "region_intra",
    "road_to_syntax",
    "syntax_to_region",
)
RELATION_TO_ID = {name: index for index, name in enumerate(RELATION_NAMES)}


@dataclass(frozen=True)
class JointThreeLayerGraph:
    """Stable Road/Syntax/Region graph consumed by the single GraphGPS stack.

    The first three node ranges are part of the serialized contract.  The
    message graph is directed and relation/weight aware; ``edge_index_pe`` is
    created separately by the LapPE module and must never replace it.
    """

    city_id: str
    x_joint: torch.Tensor
    edge_index_joint: torch.Tensor
    edge_type: torch.Tensor
    edge_weight: torch.Tensor
    node_type: torch.Tensor
    road_node_range: tuple[int, int]
    syntax_node_range: tuple[int, int]
    region_node_range: tuple[int, int]
    road_ids: tuple[str, ...]
    syntax_ids: tuple[int, ...]
    region_ids: tuple[int, ...]
    joint_graph_hash: str
    metadata: dict[str, Any]
    road_edge_index: torch.Tensor
    syntax_edge_index: torch.Tensor
    region_edge_index: torch.Tensor
    road_to_syntax_edge_index: torch.Tensor
    syntax_to_region_edge_index: torch.Tensor

    @property
    def num_roads(self) -> int:
        return self.road_node_range[1] - self.road_node_range[0]

    @property
    def num_syntax(self) -> int:
        return self.syntax_node_range[1] - self.syntax_node_range[0]

    @property
    def num_regions(self) -> int:
        return self.region_node_range[1] - self.region_node_range[0]

    @property
    def num_nodes(self) -> int:
        return int(self.x_joint.shape[0])

    @property
    def edge_index_joint_msg(self) -> torch.Tensor:
        return self.edge_index_joint

    def to(self, device: torch.device | str) -> "JointThreeLayerGraph":
        values = {
            name: value.to(device) if isinstance(value, torch.Tensor) else value
            for name, value in self.__dict__.items()
        }
        return JointThreeLayerGraph(**values)


def _joint_hash(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    edge_weight: torch.Tensor,
    num_nodes: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray([num_nodes], dtype="<i8").tobytes())
    digest.update(edge_index.detach().cpu().contiguous().numpy().astype("<i8").tobytes())
    digest.update(edge_type.detach().cpu().contiguous().numpy().astype("<i8").tobytes())
    digest.update(edge_weight.detach().cpu().contiguous().numpy().astype("<f4").tobytes())
    return digest.hexdigest()


def joint_graph_hash(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    edge_weight: torch.Tensor,
    num_nodes: int,
) -> str:
    """Stable hash of the directed typed/weighted unified message graph."""

    return _joint_hash(edge_index, edge_type, edge_weight, num_nodes)


def validate_joint_three_layer_graph(graph: JointThreeLayerGraph) -> None:
    """Strictly validate ranges, offsets, relation types, directions and weights."""

    if not isinstance(graph, JointThreeLayerGraph):
        raise TypeError("严格模式: graph 必须为 JointThreeLayerGraph")
    if graph.x_joint.ndim != 2 or graph.x_joint.shape[0] <= 0:
        raise ValueError("严格模式: x_joint 必须为非空 [V,D]")
    if not torch.isfinite(graph.x_joint).all():
        raise ValueError("严格模式: x_joint 含 NaN/Inf")
    v = int(graph.x_joint.shape[0])
    ranges = (graph.road_node_range, graph.syntax_node_range, graph.region_node_range)
    if ranges[0] != (0, graph.num_roads):
        raise ValueError("严格模式: Road 节点范围必须从 0 开始")
    if ranges[1] != (graph.num_roads, graph.num_roads + graph.num_syntax):
        raise ValueError("严格模式: Syntax 节点偏移错误")
    if ranges[2] != (graph.num_roads + graph.num_syntax, v):
        raise ValueError("严格模式: Region 节点偏移错误")
    if any(start < 0 or end <= start or end > v for start, end in ranges):
        raise ValueError("严格模式: 节点范围越界或为空")
    if graph.node_type.dtype != torch.long or graph.node_type.shape != (v,):
        raise ValueError("严格模式: node_type 必须为 LongTensor[V]")
    expected_types = torch.cat([
        torch.zeros(graph.num_roads, dtype=torch.long),
        torch.ones(graph.num_syntax, dtype=torch.long),
        torch.full((graph.num_regions,), 2, dtype=torch.long),
    ])
    if not torch.equal(graph.node_type.detach().cpu(), expected_types):
        raise ValueError("严格模式: node_type 或节点顺序不符合 Road/Syntax/Region 固定编号")
    if len(graph.road_ids) != graph.num_roads:
        raise ValueError("严格模式: road_ids 长度错误")
    if tuple(graph.syntax_ids) != tuple(range(graph.num_syntax)):
        raise ValueError("严格模式: syntax_ids 必须按稳定连续顺序")
    if tuple(graph.region_ids) != tuple(range(graph.num_regions)):
        raise ValueError("严格模式: region_ids 必须按稳定连续顺序")
    if graph.edge_index_joint.dtype != torch.long or graph.edge_index_joint.ndim != 2 or graph.edge_index_joint.shape[0] != 2:
        raise ValueError("严格模式: edge_index_joint 必须为 LongTensor[2,E]")
    e = int(graph.edge_index_joint.shape[1])
    if e and (int(graph.edge_index_joint.min()) < 0 or int(graph.edge_index_joint.max()) >= v):
        raise ValueError("严格模式: edge_index_joint 节点越界")
    if graph.edge_type.dtype != torch.long or graph.edge_type.shape != (e,):
        raise ValueError("严格模式: edge_type 长度必须等于边数")
    if e and (int(graph.edge_type.min()) < 0 or int(graph.edge_type.max()) >= len(RELATION_NAMES)):
        raise ValueError("严格模式: edge_type 含未知关系")
    if graph.edge_weight.ndim != 1 or graph.edge_weight.shape != (e,) or not graph.edge_weight.is_floating_point():
        raise ValueError("严格模式: edge_weight 必须为浮点 [E]")
    if not torch.isfinite(graph.edge_weight).all() or (graph.edge_weight <= 0).any():
        raise ValueError("严格模式: edge_weight 必须为有限正数")

    src, dst = graph.edge_index_joint
    rel = graph.edge_type
    # Intra-layer edges stay within their own range.
    ranges_by_rel = {
        RELATION_TO_ID["road_intra"]: graph.road_node_range,
        RELATION_TO_ID["syntax_intra"]: graph.syntax_node_range,
        RELATION_TO_ID["region_intra"]: graph.region_node_range,
    }
    for relation, (start, end) in ranges_by_rel.items():
        mask = rel == relation
        if mask.any() and not (((src[mask] >= start) & (src[mask] < end) & (dst[mask] >= start) & (dst[mask] < end)).all()):
            raise ValueError("严格模式: 层内边落在错误节点范围")
    cross_masks = {
        "road_to_syntax": rel == RELATION_TO_ID["road_to_syntax"],
        "syntax_to_region": rel == RELATION_TO_ID["syntax_to_region"],
    }
    mask = cross_masks["road_to_syntax"]
    if mask.any() and not (((src[mask] >= 0) & (src[mask] < graph.num_roads) & (dst[mask] >= graph.num_roads) & (dst[mask] < graph.num_roads + graph.num_syntax)).all()):
        raise ValueError("严格模式: Road→Syntax 边方向或偏移错误")
    mask = cross_masks["syntax_to_region"]
    if mask.any() and not (((src[mask] >= graph.num_roads) & (src[mask] < graph.num_roads + graph.num_syntax) & (dst[mask] >= graph.num_roads + graph.num_syntax)).all()):
        raise ValueError("严格模式: Syntax→Region 边方向或偏移错误")
    if ((src < graph.num_roads) & (dst >= graph.num_roads + graph.num_syntax)).any():
        raise ValueError("严格模式: Unified graph 禁止 Road→Region 直连")
    expected_hash = joint_graph_hash(graph.edge_index_joint, graph.edge_type, graph.edge_weight, v)
    if graph.joint_graph_hash != expected_hash:
        raise ValueError("严格模式: joint_graph_hash 与联合消息图不一致")


def build_joint_three_layer_graph(hierarchy: CityStaticHierarchy) -> JointThreeLayerGraph:
    """Build the stable directed heterogeneous graph without changing Stage 1 data."""

    from static_hierarchy.contracts import validate_city_static_hierarchy

    validate_city_static_hierarchy(hierarchy)
    m, k, n = hierarchy.num_roads, hierarchy.num_syntax, hierarchy.num_regions
    device = hierarchy.road_x.device
    # The graph builder does not concatenate raw 33/5/45-D features (that is
    # impossible without padding).  The model replaces this placeholder with
    # projected hidden features; zeros keep standalone graph validation finite.
    x_placeholder = torch.zeros(
        (m + k + n, 1), dtype=hierarchy.road_x.dtype, device=device
    )
    node_type = torch.cat([
        torch.zeros(m, dtype=torch.long, device=device),
        torch.ones(k, dtype=torch.long, device=device),
        torch.full((n,), 2, dtype=torch.long, device=device),
    ])
    edges, relations, weights = [], [], []

    def append(edge_index: torch.Tensor, relation: str, weight: torch.Tensor) -> None:
        edges.append(edge_index.long())
        relations.append(torch.full(
            (edge_index.shape[1],),
            RELATION_TO_ID[relation],
            dtype=torch.long,
            device=edge_index.device,
        ))
        weight = weight.to(dtype=torch.float32).reshape(-1)
        if weight.shape != (edge_index.shape[1],):
            raise ValueError(f"严格模式: {relation} edge_weight 长度错误")
        weights.append(weight)

    append(
        hierarchy.road_edge_index,
        "road_intra",
        torch.ones(hierarchy.road_edge_index.shape[1], device=device),
    )
    append(hierarchy.syntax_edge_index + m, "syntax_intra", torch.as_tensor(
        hierarchy.metadata.get("syntax_edge_weight", [1.0] * hierarchy.syntax_edge_index.shape[1]),
        dtype=torch.float32,
        device=device,
    ))
    append(
        hierarchy.region_edge_index + m + k,
        "region_intra",
        torch.ones(hierarchy.region_edge_index.shape[1], device=device),
    )
    # Stage 1 COO is [syntax, road] and [region, syntax]; convert to message
    # direction source→destination while retaining the exact operator weights.
    rs = hierarchy.road_to_syntax_edge_index
    append(torch.stack([rs[1], rs[0] + m]), "road_to_syntax", hierarchy.road_to_syntax_weight)
    sr = hierarchy.syntax_to_region_edge_index
    append(torch.stack([sr[1] + m, sr[0] + m + k]), "syntax_to_region", hierarchy.syntax_to_region_weight)
    edge_index = torch.cat(edges, dim=1)
    edge_type = torch.cat(relations, dim=0)
    edge_weight = torch.cat(weights, dim=0)
    graph = JointThreeLayerGraph(
        city_id=hierarchy.city_id,
        x_joint=x_placeholder,
        edge_index_joint=edge_index,
        edge_type=edge_type,
        edge_weight=edge_weight,
        node_type=node_type,
        road_node_range=(0, m),
        syntax_node_range=(m, m + k),
        region_node_range=(m + k, m + k + n),
        road_ids=tuple(hierarchy.road_ids),
        syntax_ids=tuple(range(k)),
        region_ids=tuple(range(n)),
        joint_graph_hash=joint_graph_hash(edge_index, edge_type, edge_weight, m + k + n),
        metadata={
            "relation_names": RELATION_NAMES,
            "joint_graph_hash": joint_graph_hash(edge_index, edge_type, edge_weight, m + k + n),
            "num_joint_nodes": m + k + n,
            "road_node_range": (0, m),
            "syntax_node_range": (m, m + k),
            "region_node_range": (m + k, m + k + n),
            "hierarchy_version": hierarchy.metadata.get("feature_version"),
        },
        road_edge_index=hierarchy.road_edge_index,
        syntax_edge_index=hierarchy.syntax_edge_index,
        region_edge_index=hierarchy.region_edge_index,
        road_to_syntax_edge_index=hierarchy.road_to_syntax_edge_index,
        syntax_to_region_edge_index=hierarchy.syntax_to_region_edge_index,
    )
    validate_joint_three_layer_graph(graph)
    return graph


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
    joint_graph: JointThreeLayerGraph | None = None


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
    road_k: int = 16,
    syntax_k: int = 16,
    region_k: int = 16,
    joint_k: int | None = None,
    normalization: str = "sym",
    require_targets: bool,
    norm_flow_root: str | Path | None = None,
    seq_length: int = 24,
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> GraphGPSCityData:
    hierarchy = load_stage2_hierarchy(hierarchy_cache_dir, city)
    if joint_k is None:
        joint_k = max(int(road_k), int(syntax_k), int(region_k))
    posenc = prepare_hierarchy_lappe(
        hierarchy,
        joint_k=joint_k,
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
    return GraphGPSCityData(
        hierarchy=hierarchy,
        posenc=posenc,
        targets=targets,
        joint_graph=build_joint_three_layer_graph(hierarchy),
    )


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
    global_attention_scope: str,
) -> Path:
    """Write weighted-v3 joint-GraphGPS features for later Stage 3 consumers."""

    hierarchy, posenc = data.hierarchy, data.posenc
    if not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64:
        raise ValueError("严格模式: checkpoint_sha256 必须为 64 位 SHA-256")
    if any(character not in string.hexdigits for character in checkpoint_sha256):
        raise ValueError("严格模式: checkpoint_sha256 含非十六进制字符")
    if hierarchy.metadata.get("feature_version") != STATIC_FEATURE_VERSION:
        raise ValueError("严格模式: 频率特征只能从 START v2 static cache 导出")
    required = (
        "H_joint", "H_joint_low", "H_joint_high", "joint_low_coefficients",
        "H_road",
        "H_road_low",
        "H_road_high",
        "H_syntax",
        "H_syntax_low",
        "H_syntax_high",
        "H_region",
        "H_region_low",
        "H_region_high",
    )
    missing = sorted(set(required) - set(output))
    if missing:
        raise KeyError(f"严格模式: 模型输出缺少频率字段 {missing}")

    joint = data.joint_graph or build_joint_three_layer_graph(hierarchy)
    validate_joint_three_layer_graph(joint)
    if posenc.metadata.get("joint_graph_hash") != joint.joint_graph_hash:
        raise ValueError("严格模式: joint LapPE graph hash 与联合图不一致")
    expected_spectrum_hash = weighted_pe_graph_hash(
        joint.edge_index_joint, joint.num_nodes, joint.edge_weight, joint.edge_type
    )
    if posenc.metadata.get("pe_version") != LAPPE_VERSION:
        raise ValueError("严格模式: LapPE version 不是当前 weighted v3")
    if posenc.metadata.get("weighted_pe") is not True:
        raise ValueError("严格模式: spectral export 必须使用 weighted LapPE")
    if posenc.metadata.get("weighted_spectrum_hash") != expected_spectrum_hash:
        raise ValueError("严格模式: weighted spectrum hash 与联合图不一致")
    if posenc.joint.edge_weight_pe is None or posenc.joint.edge_type_pe is None:
        raise ValueError("严格模式: weighted LapPE 缺少 PE edge weight/type audit tensors")
    if global_attention_scope not in {"joint", "same_layer"}:
        raise ValueError("严格模式: global_attention_scope 非法")
    if output["H_joint"].shape[0] != joint.num_nodes:
        raise ValueError("严格模式: H_joint 节点数与联合图不一致")
    if not torch.allclose(output["H_joint"], output["H_joint_low"] + output["H_joint_high"], atol=1e-5, rtol=1e-5):
        raise ValueError("严格模式: joint low/high 无法重构 H_joint")
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
        "lappe_version": LAPPE_VERSION,
        "weighted_pe": True,
        "weighted_spectrum_hash": expected_spectrum_hash,
        "global_attention_scope": global_attention_scope,
        "road_ids": road_ids,
        "syntax_ids": torch.arange(hierarchy.num_syntax, dtype=torch.long),
        "region_ids": torch.arange(hierarchy.num_regions, dtype=torch.long),
        "joint_graph_hash": joint.joint_graph_hash,
        "num_joint_nodes": joint.num_nodes,
        "road_node_range": joint.road_node_range,
        "syntax_node_range": joint.syntax_node_range,
        "region_node_range": joint.region_node_range,
        "joint_edge_index": joint.edge_index_joint,
        "joint_edge_type": joint.edge_type,
        "joint_edge_weight": joint.edge_weight,
        "joint_edge_index_pe": _cpu_tensor(posenc.joint.edge_index_pe, "joint_edge_index_pe"),
        "joint_edge_weight_pe": _cpu_tensor(posenc.joint.edge_weight_pe, "joint_edge_weight_pe"),
        "joint_edge_type_pe": posenc.joint.edge_type_pe.detach().cpu().contiguous(),
        # The spectrum is shared by every node; export one canonical row.
        "joint_eigvals": _cpu_tensor(posenc.joint.eigvals[:1], "joint_eigvals"),
        "joint_eigvecs": _cpu_tensor(posenc.joint.eigvecs, "joint_eigvecs"),
        "joint_eigenpair_mask": posenc.joint.mask.detach().cpu(),
        "H_joint": _cpu_tensor(output["H_joint"], "H_joint"),
        "H_joint_low": _cpu_tensor(output["H_joint_low"], "H_joint_low"),
        "H_joint_high": _cpu_tensor(output["H_joint_high"], "H_joint_high"),
        "joint_num_low_modes": int(output["joint_low_coefficients"].shape[0]),
    }
    for layer in layer_nodes:
        payload[f"H_{layer}"] = _cpu_tensor(output[f"H_{layer}"], f"H_{layer}")
        payload[f"H_{layer}_low"] = _cpu_tensor(output[f"H_{layer}_low"], f"H_{layer}_low")
        payload[f"H_{layer}_high"] = _cpu_tensor(output[f"H_{layer}_high"], f"H_{layer}_high")

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
        raise ValueError(f"严格模式: 不是 {SPECTRAL_FEATURE_VERSION} 文件；旧无权谱特征不得混合")
    if payload.get("static_feature_version") != STATIC_FEATURE_VERSION:
        raise ValueError("严格模式: spectral/static feature version 不匹配")
    if payload.get("lappe_version") != LAPPE_VERSION or payload.get("weighted_pe") is not True:
        raise ValueError("严格模式: spectral feature 不是当前 weighted LapPE 语义")
    if payload.get("global_attention_scope") not in {"joint", "same_layer"}:
        raise ValueError("严格模式: spectral feature 缺少 global_attention_scope")
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
    joint = build_joint_three_layer_graph(hierarchy)
    validate_joint_three_layer_graph(joint)
    if payload.get("joint_graph_hash") != joint.joint_graph_hash:
        raise ValueError("严格模式: joint_graph_hash 与当前联合图不匹配")
    expected_spectrum_hash = weighted_pe_graph_hash(
        joint.edge_index_joint, joint.num_nodes, joint.edge_weight, joint.edge_type
    )
    if payload.get("weighted_spectrum_hash") != expected_spectrum_hash:
        raise ValueError("严格模式: weighted_spectrum_hash 与当前联合图不匹配")
    expected_pe_edges, expected_pe_weights, expected_pe_types = (
        to_undirected_edge_index_with_weight(
            joint.edge_index_joint,
            joint.num_nodes,
            edge_weight=joint.edge_weight,
            edge_type=joint.edge_type,
        )
    )
    for key, expected in (
        ("joint_edge_index_pe", expected_pe_edges),
        ("joint_edge_weight_pe", expected_pe_weights),
        ("joint_edge_type_pe", expected_pe_types),
    ):
        actual = payload.get(key)
        if (
            not isinstance(actual, torch.Tensor)
            or expected is None
            or not torch.equal(actual, expected)
        ):
            raise ValueError(f"严格模式: {key} 与当前加权无向 LapPE 图不匹配")
    if int(payload.get("num_joint_nodes", -1)) != joint.num_nodes:
        raise ValueError("严格模式: num_joint_nodes 与当前联合图不匹配")
    for key, expected in (("road_node_range", joint.road_node_range), ("syntax_node_range", joint.syntax_node_range), ("region_node_range", joint.region_node_range)):
        if tuple(payload.get(key, ())) != tuple(expected):
            raise ValueError(f"严格模式: {key} 与当前节点编号不匹配")
    for key, expected in (("joint_edge_index", joint.edge_index_joint), ("joint_edge_type", joint.edge_type), ("joint_edge_weight", joint.edge_weight)):
        actual = payload.get(key)
        if not isinstance(actual, torch.Tensor) or not torch.equal(actual, expected.cpu() if isinstance(expected, torch.Tensor) else expected):
            raise ValueError(f"严格模式: {key} 与当前联合消息图不匹配")
    for key in ("joint_eigvals", "joint_eigvecs", "joint_eigenpair_mask", "H_joint", "H_joint_low", "H_joint_high"):
        if not isinstance(payload.get(key), torch.Tensor):
            raise KeyError(f"严格模式: 导出缺少 {key}")
    if not torch.allclose(payload["H_joint"], payload["H_joint_low"] + payload["H_joint_high"], atol=1e-5, rtol=1e-5):
        raise ValueError("严格模式: 导出 joint low/high 无法重构 H_joint")
    if payload["H_joint"].shape[0] != joint.num_nodes:
        raise ValueError("严格模式: 导出 H_joint 节点数非法")
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
    eigvals = payload["joint_eigvals"]
    if eigvals.ndim != 3 or eigvals.shape[0] != 1 or eigvals.shape[2] != 1 or not torch.isfinite(eigvals).all():
        raise ValueError("严格模式: 导出 joint_eigvals shape/finite 非法")
    eigvecs, mask = payload["joint_eigvecs"], payload["joint_eigenpair_mask"]
    if eigvecs.ndim != 2 or eigvecs.shape != (joint.num_nodes, eigvals.shape[1]) or not torch.isfinite(eigvecs).all():
        raise ValueError("严格模式: 导出 joint_eigvecs shape/finite 非法")
    if mask.dtype != torch.bool or mask.shape != (eigvals.shape[1],):
        raise ValueError("严格模式: 导出 joint_eigenpair_mask shape/dtype 非法")
    if not isinstance(payload.get("joint_num_low_modes"), int) or not (0 < payload["joint_num_low_modes"] <= eigvals.shape[1]):
        raise ValueError("严格模式: 导出 joint_num_low_modes 非法")
    return payload
