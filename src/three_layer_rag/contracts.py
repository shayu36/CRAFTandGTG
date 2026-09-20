"""Three-layer RAG data contracts.

The contract deliberately keeps node identities out of retrieval features.  A
snapshot contains tensors in the stable Road/Syntax/Region order produced by
Stage 2, while city/node identifiers are used only for leakage protection and
never enter a query/key/value projection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


LAYER_NAMES = ("road", "syntax", "region")
PARENT_NAMES = ("road_to_syntax", "syntax_to_region")
RAG_CONTRACT_VERSION = "three-layer-hierarchical-rag-v3-weighted-stage2"
SNAPSHOT_BUNDLE_VERSION = "three-layer-rag-snapshot-bundle-v2-weighted-stage2"
GRAPH_IDENTITY_KEYS = (
    "joint_graph_hash",
    "checkpoint_fingerprint",
    "static_feature_version",
    "spectral_feature_version",
    "lappe_version",
    "weighted_pe",
    "weighted_spectrum_hash",
    "global_attention_scope",
    "road_node_range",
    "syntax_node_range",
    "region_node_range",
)


def _as_float_tensor(value: Any, name: str) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    tensor = tensor.detach().clone().to(dtype=torch.float32)
    if not torch.isfinite(tensor).all():
        raise ValueError(f"严格模式: {name} 含 NaN/Inf")
    return tensor.contiguous()


def _as_parent_index(value: Any, name: str, size: int, parent_size: int) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    tensor = tensor.detach().clone().to(dtype=torch.long).reshape(-1)
    if tensor.shape != (size,):
        raise ValueError(f"严格模式: {name} 必须为 [{size}]，实得 {tuple(tensor.shape)}")
    if tensor.numel() and (int(tensor.min()) < 0 or int(tensor.max()) >= parent_size):
        raise ValueError(f"严格模式: {name} 含越界父节点索引")
    return tensor.contiguous()


def _as_parent_operator(
    value: Mapping[str, Any], name: str, child_size: int, parent_size: int
) -> dict[str, torch.Tensor]:
    """Validate a sparse parent operator with ``row=parent, col=child``."""

    if not isinstance(value, Mapping) or set(value) != {"edge_index", "weight"}:
        raise ValueError(f"严格模式: {name} 必须包含 edge_index/weight")
    edge_index = value["edge_index"] if isinstance(value["edge_index"], torch.Tensor) else torch.as_tensor(value["edge_index"])
    edge_index = edge_index.detach().clone().to(dtype=torch.long)
    weight = value["weight"] if isinstance(value["weight"], torch.Tensor) else torch.as_tensor(value["weight"])
    weight = weight.detach().clone().to(dtype=torch.float32)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"严格模式: {name}.edge_index 必须为 LongTensor[2,E]")
    if weight.ndim != 1 or weight.shape[0] != edge_index.shape[1]:
        raise ValueError(f"严格模式: {name}.weight 长度必须等于边数")
    if edge_index.numel() and (
        int(edge_index[0].min()) < 0 or int(edge_index[0].max()) >= parent_size
        or int(edge_index[1].min()) < 0 or int(edge_index[1].max()) >= child_size
    ):
        raise ValueError(f"严格模式: {name} parent/child 索引越界")
    if not torch.isfinite(weight).all() or (weight <= 0).any():
        raise ValueError(f"严格模式: {name}.weight 必须为有限正数")
    return {"edge_index": edge_index.contiguous(), "weight": weight.contiguous()}


def _calendar_scalar(calendar: Mapping[str, Any], name: str, default: Any = None) -> Any:
    value = calendar.get(name, default)
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"严格模式: calendar.{name} 必须是单个快照条件")
        value = value.detach().cpu().reshape(-1)[0].item()
    return value


@dataclass(frozen=True)
class ThreeLayerRAGInputs:
    """One city snapshot supplied to the hierarchical RAG.

    ``low_features`` are Stage-2 ``H_*_low`` tensors ``[num_nodes,D]``;
    ``temporal_features`` are raw/normalized dynamic tensors ``[num_nodes,C,T]``.
    ``parent_index`` is optional because a caller may not have materialized a
    one-parent assignment.  The model then uses a documented global parent
    mean fallback, never a fabricated node ID embedding.
    """

    city_id: str
    split: str
    low_features: Mapping[str, torch.Tensor]
    temporal_features: Mapping[str, torch.Tensor]
    calendar: Mapping[str, Any]
    parent_index: Mapping[str, torch.Tensor] | None = None
    parent_operator: Mapping[str, Mapping[str, torch.Tensor]] | None = None
    value_temporal_features: Mapping[str, torch.Tensor] | None = None
    graph_metadata: Mapping[str, Any] | None = None

    @property
    def history_temporal_features(self) -> Mapping[str, torch.Tensor]:
        """Historical observable sequences used by Query/Key."""

        return self.temporal_features

    @property
    def value_features(self) -> Mapping[str, torch.Tensor]:
        """Future/value sequences; legacy inputs are intentionally explicit."""

        if self.value_temporal_features is None:
            raise ValueError(
                "严格模式: source RAG Value 必须提供独立 value_temporal_features；"
                "不能复用 history_temporal_features"
            )
        return self.value_temporal_features

    def validate(self, *, temporal_channels: Mapping[str, int] | None = None) -> "ThreeLayerRAGInputs":
        if not isinstance(self.city_id, str) or not self.city_id:
            raise ValueError("严格模式: RAG city_id 必须是非空字符串")
        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"严格模式: RAG split 非法: {self.split!r}")
        if set(self.low_features) != set(LAYER_NAMES):
            raise ValueError("严格模式: low_features 必须包含 road/syntax/region 三层")
        if not isinstance(self.temporal_features, Mapping) or set(self.temporal_features) != set(LAYER_NAMES):
            raise ValueError("严格模式: history temporal_features 必须包含 road/syntax/region 三层")
        if self.value_temporal_features is not None and set(self.value_temporal_features) != set(LAYER_NAMES):
            raise ValueError("严格模式: value_temporal_features 必须包含 road/syntax/region 三层")
        node_counts: dict[str, int] = {}
        low_dim: int | None = None
        for layer in LAYER_NAMES:
            low = _as_float_tensor(self.low_features[layer], f"low_features.{layer}")
            temporal = _as_float_tensor(self.temporal_features[layer], f"temporal_features.{layer}")
            if low.ndim != 2 or low.shape[0] <= 0 or low.shape[1] <= 0:
                raise ValueError(f"严格模式: low_features.{layer} 必须为非空 [N,D]")
            if temporal.ndim != 3 or temporal.shape[0] != low.shape[0] or temporal.shape[1] <= 0 or temporal.shape[2] <= 0:
                raise ValueError(
                    f"严格模式: temporal_features.{layer} 必须为 [N,C,T] 且 N 与低频特征一致"
                )
            if low_dim is None:
                low_dim = int(low.shape[1])
            elif int(low.shape[1]) != low_dim:
                raise ValueError("严格模式: 三层低频特征的 hidden dimension 必须一致")
            if temporal_channels is not None and int(temporal.shape[1]) != int(temporal_channels[layer]):
                raise ValueError(
                    f"严格模式: {layer} 动态通道数 {temporal.shape[1]} != 配置 {temporal_channels[layer]}"
                )
            node_counts[layer] = int(low.shape[0])
            if self.value_temporal_features is not None:
                value = _as_float_tensor(self.value_temporal_features[layer], f"value_temporal_features.{layer}")
                if value.shape != temporal.shape:
                    raise ValueError(
                        f"严格模式: value_temporal_features.{layer} 必须与 history temporal shape 一致"
                    )

        required_calendar = ("month", "weekday", "start_hour")
        for name in required_calendar:
            value = _calendar_scalar(self.calendar, name)
            if value is None:
                raise ValueError(f"严格模式: 缺少 calendar.{name}")
        month = int(_calendar_scalar(self.calendar, "month"))
        weekday = int(_calendar_scalar(self.calendar, "weekday"))
        hour = int(_calendar_scalar(self.calendar, "start_hour"))
        if not 1 <= month <= 12 or not 0 <= weekday < 7 or not 0 <= hour < 24:
            raise ValueError("严格模式: calendar month/weekday/start_hour 超出范围")
        holiday = _calendar_scalar(self.calendar, "holiday", 0)
        if int(holiday) not in (0, 1):
            raise ValueError("严格模式: calendar.holiday 必须为 0/1")

        if self.parent_index is not None:
            unknown = set(self.parent_index) - set(PARENT_NAMES)
            if unknown:
                raise ValueError(f"严格模式: 未知 parent_index 字段 {sorted(unknown)}")
            if "road_to_syntax" in self.parent_index:
                _as_parent_index(
                    self.parent_index["road_to_syntax"], "parent_index.road_to_syntax",
                    node_counts["road"], node_counts["syntax"],
                )
            if "syntax_to_region" in self.parent_index:
                _as_parent_index(
                    self.parent_index["syntax_to_region"], "parent_index.syntax_to_region",
                    node_counts["syntax"], node_counts["region"],
                )
        if self.parent_operator is not None:
            unknown = set(self.parent_operator) - set(PARENT_NAMES)
            if unknown:
                raise ValueError(f"严格模式: 未知 parent_operator 字段 {sorted(unknown)}")
            if "road_to_syntax" in self.parent_operator:
                _as_parent_operator(
                    self.parent_operator["road_to_syntax"], "parent_operator.road_to_syntax",
                    node_counts["road"], node_counts["syntax"],
                )
            if "syntax_to_region" in self.parent_operator:
                _as_parent_operator(
                    self.parent_operator["syntax_to_region"], "parent_operator.syntax_to_region",
                    node_counts["syntax"], node_counts["region"],
                )
        if self.graph_metadata is not None:
            if not isinstance(self.graph_metadata, Mapping):
                raise ValueError("严格模式: graph_metadata 必须为 mapping")
            for name in GRAPH_IDENTITY_KEYS:
                if name in self.graph_metadata and self.graph_metadata[name] is None:
                    raise ValueError(f"严格模式: graph_metadata.{name} 不能为 None")
        return self

    def detached_cpu(self) -> "ThreeLayerRAGInputs":
        self.validate()
        parents = None
        if self.parent_index is not None:
            parents = {name: value.detach().cpu().clone() for name, value in self.parent_index.items()}
        return ThreeLayerRAGInputs(
            city_id=self.city_id,
            split=self.split,
            low_features={name: _as_float_tensor(value, f"low_features.{name}").cpu() for name, value in self.low_features.items()},
            temporal_features={name: _as_float_tensor(value, f"temporal_features.{name}").cpu() for name, value in self.temporal_features.items()},
            calendar=dict(self.calendar),
            parent_index=parents,
            parent_operator={
                name: {
                    "edge_index": value["edge_index"].detach().cpu().clone(),
                    "weight": value["weight"].detach().cpu().clone(),
                }
                for name, value in (self.parent_operator or {}).items()
            } or None,
            value_temporal_features=(
                {
                    name: _as_float_tensor(value, f"value_temporal_features.{name}").cpu()
                    for name, value in self.value_temporal_features.items()
                }
                if self.value_temporal_features is not None else None
            ),
            graph_metadata=dict(self.graph_metadata or {}) or None,
        )

    @classmethod
    def from_graphgps_output(
        cls,
        output: Mapping[str, torch.Tensor],
        temporal_features: Mapping[str, torch.Tensor],
        calendar: Mapping[str, Any],
        *,
        city_id: str,
        split: str,
        parent_index: Mapping[str, torch.Tensor] | None = None,
        parent_operator: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
        value_temporal_features: Mapping[str, torch.Tensor] | None = None,
        graph_metadata: Mapping[str, Any] | None = None,
    ) -> "ThreeLayerRAGInputs":
        """Build the RAG input from Stage-2 low-band output only."""

        missing = [f"H_{layer}_low" for layer in LAYER_NAMES if f"H_{layer}_low" not in output]
        if missing:
            raise KeyError(f"严格模式: Stage-2 输出缺少低频字段 {missing}")
        low = {layer: output[f"H_{layer}_low"] for layer in LAYER_NAMES}
        return cls(
            city_id, split, low, temporal_features, calendar, parent_index,
            parent_operator, value_temporal_features, graph_metadata,
        ).validate()


@dataclass(frozen=True)
class ThreeLayerRAGMemory:
    """Source-only train memory; raw tensors remain local and are never IDs."""

    snapshots: tuple[ThreeLayerRAGInputs, ...]
    source_cities: tuple[str, ...]
    source_split: str = "train"
    version: str = RAG_CONTRACT_VERSION
    require_value_separation: bool = True
    require_graph_identity: bool = False
    graph_identity: Mapping[str, Any] | None = None

    @classmethod
    def from_snapshots(
        cls,
        snapshots: Sequence[ThreeLayerRAGInputs],
        *,
        source_cities: Sequence[str],
        split: str = "train",
        require_value_separation: bool = True,
        require_graph_identity: bool = False,
    ) -> "ThreeLayerRAGMemory":
        sources = tuple(sorted({str(city) for city in source_cities}))
        if split != "train":
            raise ValueError("RAG 泄漏防护: memory 只能从 train split 构建")
        if not sources:
            raise ValueError("严格模式: RAG source_cities 不能为空")
        if not snapshots:
            raise ValueError("严格模式: RAG memory 不能为空")
        checked = []
        for snapshot in snapshots:
            snapshot.validate()
            if snapshot.split != split or snapshot.city_id not in sources:
                raise ValueError(
                    f"RAG 泄漏防护: snapshot city={snapshot.city_id!r}, split={snapshot.split!r} "
                    f"不属于 source train {sources}"
                )
            checked.append(snapshot.detached_cpu())
        graph_identity: dict[str, dict[str, Any]] = {}
        for item in checked:
            identity = dict(item.graph_metadata or {})
            if identity:
                previous = graph_identity.get(item.city_id)
                if previous is not None and previous != identity:
                    raise ValueError(f"严格模式: city={item.city_id} 的 graph identity 不一致")
                graph_identity[item.city_id] = identity
        if require_value_separation and any(item.value_temporal_features is None for item in checked):
            raise ValueError(
                "严格模式: source RAG memory 必须把 history_temporal_features 与 "
                "value_temporal_features 分开保存"
            )
        if require_graph_identity:
            missing = [
                f"{city}.{name}"
                for city in sources
                for name in GRAPH_IDENTITY_KEYS
                if city not in graph_identity or name not in graph_identity[city]
            ]
            if missing:
                raise ValueError(f"严格模式: graph identity 缺少字段 {missing}")
        return cls(tuple(checked), sources, split, RAG_CONTRACT_VERSION,
                   require_value_separation, require_graph_identity, graph_identity or None)

    def validate(self, expected_graph_identity: Mapping[str, Any] | None = None) -> None:
        if self.source_split != "train" or not self.source_cities or not self.snapshots:
            raise ValueError("严格模式: 非法或空的 source train RAG memory")
        for snapshot in self.snapshots:
            snapshot.validate()
            if snapshot.city_id not in self.source_cities or snapshot.split != "train":
                raise ValueError("RAG memory 含非 source-train snapshot")
            if self.require_value_separation and snapshot.value_temporal_features is None:
                raise ValueError("严格模式: RAG memory 缺少独立 value_temporal_features")
        if self.require_graph_identity:
            identities = dict(self.graph_identity or {})
            missing = [
                f"{city}.{name}"
                for city in self.source_cities
                for name in GRAPH_IDENTITY_KEYS
                if city not in identities or name not in identities[city]
            ]
            if missing:
                raise ValueError(f"严格模式: RAG memory graph identity 缺少字段 {missing}")
        if expected_graph_identity is not None:
            expected = dict(expected_graph_identity)
            city = expected.pop("city_id", None)
            identities = dict(self.graph_identity or {})
            candidates = [identities[str(city)]] if city is not None and str(city) in identities else list(identities.values())
            if not candidates or not any(
                all(identity.get(name) == value for name, value in expected.items())
                for identity in candidates
            ):
                raise ValueError("严格模式: RAG memory graph identity 不匹配")

    def save(self, path: str | Path) -> None:
        self.validate()
        payload = {
            "version": self.version,
            "source_cities": self.source_cities,
            "source_split": self.source_split,
            "require_value_separation": self.require_value_separation,
            "require_graph_identity": self.require_graph_identity,
            "graph_identity": dict(self.graph_identity or {}),
            "snapshots": [
                {
                    "city_id": item.city_id,
                    "split": item.split,
                    "low_features": dict(item.low_features),
                    "temporal_features": dict(item.temporal_features),
                    "calendar": dict(item.calendar),
                    "parent_index": dict(item.parent_index or {}),
                    "parent_operator": dict(item.parent_operator or {}),
                    "value_temporal_features": dict(item.value_temporal_features or {}),
                    "graph_metadata": dict(item.graph_metadata or {}),
                }
                for item in self.snapshots
            ],
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, destination)

    @classmethod
    def load(cls, path: str | Path) -> "ThreeLayerRAGMemory":
        payload = torch.load(Path(path), map_location="cpu")
        if not isinstance(payload, Mapping) or payload.get("version") != RAG_CONTRACT_VERSION:
            raise ValueError("严格模式: RAG memory 版本不匹配")
        snapshots = []
        for item in payload.get("snapshots", []):
            snapshots.append(ThreeLayerRAGInputs(
                city_id=str(item["city_id"]), split=str(item["split"]),
                low_features=item["low_features"], temporal_features=item["temporal_features"],
                calendar=item["calendar"], parent_index=item.get("parent_index") or None,
                parent_operator=item.get("parent_operator") or None,
                value_temporal_features=item.get("value_temporal_features") or None,
                graph_metadata=item.get("graph_metadata") or None,
            ))
        memory = cls.from_snapshots(
            snapshots, source_cities=tuple(payload["source_cities"]),
            split=str(payload.get("source_split", "train")),
            require_value_separation=bool(payload.get("require_value_separation", True)),
            require_graph_identity=bool(payload.get("require_graph_identity", False)),
        )
        if memory.version != payload.get("version"):
            raise ValueError("严格模式: RAG memory version 元数据不一致")
        return memory


def assert_no_high_frequency_input(features: Mapping[str, Any]) -> None:
    """Reject accidental ``H_*_high`` injection at the RAG boundary."""

    forbidden = sorted(key for key in features if key.endswith("_high") or "high" in key.lower())
    if forbidden:
        raise ValueError(
            "严格模式: 高频特征不能进入 RAG；请仅传入 H_road_low/H_syntax_low/H_region_low，"
            f"发现 {forbidden}"
        )
