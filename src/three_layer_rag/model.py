"""Hierarchical three-layer retrieval augmented generation.

This module is intentionally independent from Flow Matching.  It produces
``R_region``, ``R_syntax`` and ``R_road`` references from source-city train
memory.  Stage-2 high-frequency features are rejected at the public boundary;
they belong to the later Flow Matching condition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .contracts import (
    LAYER_NAMES,
    ThreeLayerRAGInputs,
    ThreeLayerRAGMemory,
    assert_no_high_frequency_input,
)


class TemporalSequenceEncoder(nn.Module):
    """Encode ``[B,N,C,T]`` dynamic sequences into ``[B,N,D_time]``."""

    def __init__(self, channels: int, hidden_dim: int, layers: int = 1, dropout: float = 0.0):
        super().__init__()
        if channels <= 0 or hidden_dim <= 0 or layers <= 0:
            raise ValueError("TemporalSequenceEncoder channels/hidden_dim/layers 必须为正")
        self.channels = int(channels)
        self.hidden_dim = int(hidden_dim)
        self.input_proj = nn.Linear(self.channels, self.hidden_dim)
        self.encoder = nn.GRU(
            self.hidden_dim,
            self.hidden_dim,
            num_layers=int(layers),
            batch_first=True,
            dropout=float(dropout) if layers > 1 else 0.0,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[2] != self.channels:
            raise ValueError(
                f"TemporalSequenceEncoder 期望 [B,N,{self.channels},T]，实得 {tuple(value.shape)}"
            )
        if not torch.isfinite(value).all():
            raise ValueError("严格模式: 动态序列含 NaN/Inf")
        batch, nodes, _, length = value.shape
        sequence = value.permute(0, 1, 3, 2).reshape(batch * nodes, length, self.channels)
        encoded = self.input_proj(sequence)
        encoded, _ = self.encoder(encoded)
        encoded = encoded.mean(dim=1)
        return encoded.reshape(batch, nodes, self.hidden_dim)


class CalendarEncoder(nn.Module):
    """Encode month/weekday/hour/holiday and a continuous time-of-day signal."""

    def __init__(
        self,
        month_dim: int = 8,
        weekday_dim: int = 8,
        hour_dim: int = 8,
        holiday_dim: int = 4,
    ):
        super().__init__()
        if min(month_dim, weekday_dim, hour_dim, holiday_dim) <= 0:
            raise ValueError("CalendarEncoder embedding 维度必须为正")
        # Index 0 is reserved so conventional month values 1..12 are accepted.
        self.month_embedding = nn.Embedding(13, int(month_dim))
        self.weekday_embedding = nn.Embedding(7, int(weekday_dim))
        self.hour_embedding = nn.Embedding(24, int(hour_dim))
        self.holiday_embedding = nn.Embedding(2, int(holiday_dim))

    @property
    def output_dim(self) -> int:
        return (
            self.month_embedding.embedding_dim
            + self.weekday_embedding.embedding_dim
            + self.hour_embedding.embedding_dim
            + self.holiday_embedding.embedding_dim
            + 2
        )

    @staticmethod
    def _value(
        calendar: Mapping[str, Any], name: str, batch_size: int, device: torch.device,
        *, default: Any = None, dtype: torch.dtype = torch.long,
    ) -> torch.Tensor:
        value = calendar.get(name, default)
        if value is None:
            raise ValueError(f"严格模式: 缺少 calendar.{name}")
        value = torch.as_tensor(value, device=device, dtype=dtype).reshape(-1)
        if value.numel() == 1 and batch_size > 1:
            value = value.expand(batch_size)
        if value.numel() != batch_size:
            raise ValueError(f"严格模式: calendar.{name} 与 batch 不一致")
        return value

    def forward(
        self, calendar: Mapping[str, Any], batch_size: int, device: torch.device
    ) -> torch.Tensor:
        month = self._value(calendar, "month", batch_size, device)
        weekday = self._value(calendar, "weekday", batch_size, device)
        hour = self._value(calendar, "start_hour", batch_size, device)
        holiday = self._value(calendar, "holiday", batch_size, device, default=0)
        if (month < 1).any() or (month > 12).any():
            raise ValueError("严格模式: calendar.month 必须在 1..12")
        if (weekday < 0).any() or (weekday >= 7).any():
            raise ValueError("严格模式: calendar.weekday 必须在 0..6")
        if (hour < 0).any() or (hour >= 24).any():
            raise ValueError("严格模式: calendar.start_hour 必须在 0..23")
        if (holiday < 0).any() or (holiday >= 2).any():
            raise ValueError("严格模式: calendar.holiday 必须为 0/1")
        time_of_day = self._value(
            calendar, "time_of_day", batch_size, device,
            default=hour.to(dtype=torch.float32), dtype=torch.float32,
        )
        phase = 2.0 * torch.pi * time_of_day / 24.0
        return torch.cat([
            self.month_embedding(month),
            self.weekday_embedding(weekday),
            self.hour_embedding(hour),
            self.holiday_embedding(holiday),
            torch.sin(phase).unsqueeze(-1),
            torch.cos(phase).unsqueeze(-1),
        ], dim=-1)


@dataclass
class _SourceEntry:
    city_id: str
    calendar: dict[str, int]
    key: torch.Tensor
    value: torch.Tensor
    temporal_embedding: torch.Tensor


class HierarchicalThreeLayerRAG(nn.Module):
    """Three independent retrieval branches with coarse-to-fine conditioning.

    Query/key inputs for each level contain its own low-frequency graph
    feature, encoded dynamic sequence and calendar embedding.  Syntax also
    receives Region context; Road receives Syntax and Region context.  The
    retrieved parent temporal embeddings are fed into the next query branch,
    making the retrieval itself coarse-to-fine rather than merely concatenating
    three independent references.
    """

    _PARENTS = {"region": (), "syntax": ("region",), "road": ("syntax", "region")}

    def __init__(
        self,
        low_dims: int | Mapping[str, int] | None = None,
        temporal_channels: Mapping[str, int] | None = None,
        *,
        low_dim: int | None = None,
        temporal_dim: int = 64,
        retrieval_dim: int = 128,
        seq_length: int | None = None,
        temporal_layers: int = 1,
        temporal_dropout: float = 0.0,
        calendar_dims: Mapping[str, int] | None = None,
        top_k: int = 5,
        metric: str = "cosine",
        temperature: float = 1.0,
        match_month: bool = True,
        match_holiday: bool = False,
    ):
        super().__init__()
        if low_dims is None:
            low_dims = low_dim
        elif low_dim is not None:
            raise ValueError("RAG 同时提供 low_dims 和 low_dim")
        if low_dims is None or temporal_channels is None:
            raise ValueError("RAG 必须提供 low_dims/low_dim 与 temporal_channels")
        if isinstance(low_dims, Mapping):
            self.low_dims = {layer: int(low_dims[layer]) for layer in LAYER_NAMES}
        else:
            self.low_dims = {layer: int(low_dims) for layer in LAYER_NAMES}
        self.temporal_channels = {layer: int(temporal_channels[layer]) for layer in LAYER_NAMES}
        if any(value <= 0 for value in self.low_dims.values()):
            raise ValueError("RAG low_dims 必须为正")
        if any(value <= 0 for value in self.temporal_channels.values()):
            raise ValueError("RAG temporal_channels 必须为正")
        self.temporal_dim = int(temporal_dim)
        self.retrieval_dim = int(retrieval_dim)
        if seq_length is not None and seq_length <= 0:
            raise ValueError("RAG seq_length 必须为正")
        self.seq_length = None if seq_length is None else int(seq_length)
        self.temporal_encoders = nn.ModuleDict({
            layer: TemporalSequenceEncoder(
                self.temporal_channels[layer], self.temporal_dim,
                layers=temporal_layers, dropout=temporal_dropout,
            )
            for layer in LAYER_NAMES
        })
        calendar_dims = dict(calendar_dims or {})
        self.calendar_encoder = CalendarEncoder(
            int(calendar_dims.get("month", 8)),
            int(calendar_dims.get("weekday", 8)),
            int(calendar_dims.get("start_hour", calendar_dims.get("hour", 8))),
            int(calendar_dims.get("holiday", 4)),
        )
        self.calendar_dim = self.calendar_encoder.output_dim
        self.query_projections = nn.ModuleDict()
        self.key_projections = nn.ModuleDict()
        self.branch_input_dims: dict[str, int] = {}
        for layer in LAYER_NAMES:
            input_dim = self.low_dims[layer] + self.temporal_dim + self.calendar_dim
            for parent in self._PARENTS[layer]:
                # Parent low/time context plus the coarse-to-fine retrieved time.
                input_dim += self.low_dims[parent] + 2 * self.temporal_dim
            self.branch_input_dims[layer] = input_dim
            self.query_projections[layer] = nn.Sequential(
                nn.LayerNorm(input_dim), nn.Linear(input_dim, self.retrieval_dim)
            )
            self.key_projections[layer] = nn.Sequential(
                nn.LayerNorm(input_dim), nn.Linear(input_dim, self.retrieval_dim)
            )
        if top_k <= 0:
            raise ValueError("RAG top_k 必须为正")
        if metric not in {"cosine", "euclidean"}:
            raise ValueError(f"未知 RAG metric={metric!r}")
        if temperature <= 0:
            raise ValueError("RAG temperature 必须为正")
        self.top_k = int(top_k)
        self.metric = metric
        self.temperature = float(temperature)
        self.match_month = bool(match_month)
        self.match_holiday = bool(match_holiday)
        self.memory: ThreeLayerRAGMemory | None = None

    @property
    def num_branches(self) -> int:
        return len(LAYER_NAMES)

    @property
    def uses_high_frequency(self) -> bool:
        return False

    def set_memory(self, memory: ThreeLayerRAGMemory) -> "HierarchicalThreeLayerRAG":
        memory.validate()
        self.memory = memory
        return self

    @staticmethod
    def build_memory(
        snapshots: Sequence[ThreeLayerRAGInputs], *, source_cities: Iterable[str], split: str = "train"
    ) -> ThreeLayerRAGMemory:
        return ThreeLayerRAGMemory.from_snapshots(
            snapshots, source_cities=tuple(source_cities), split=split
        )

    def _validate_batch(
        self,
        low_features: Mapping[str, torch.Tensor],
        temporal_features: Mapping[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], int]:
        if set(low_features) != set(LAYER_NAMES) or set(temporal_features) != set(LAYER_NAMES):
            raise ValueError("严格模式: RAG 输入必须同时包含 road/syntax/region 三层")
        lows: dict[str, torch.Tensor] = {}
        temporals: dict[str, torch.Tensor] = {}
        batch_size: int | None = None
        sequence_lengths: set[int] = set()
        for layer in LAYER_NAMES:
            low = low_features[layer]
            temporal = temporal_features[layer]
            if not isinstance(low, torch.Tensor):
                low = torch.as_tensor(low, dtype=torch.float32)
            if not isinstance(temporal, torch.Tensor):
                temporal = torch.as_tensor(temporal, dtype=torch.float32)
            if low.ndim == 2:
                low = low.unsqueeze(0)
            if temporal.ndim == 3:
                temporal = temporal.unsqueeze(0)
            if low.ndim != 3 or low.shape[2] != self.low_dims[layer]:
                raise ValueError(
                    f"严格模式: low_features.{layer} 应为 [B,N,{self.low_dims[layer]}]"
                )
            if temporal.ndim != 4 or temporal.shape[2] != self.temporal_channels[layer]:
                raise ValueError(
                    f"严格模式: temporal_features.{layer} 应为 [B,N,{self.temporal_channels[layer]},T]"
                )
            if low.shape[:2] != temporal.shape[:2] or temporal.shape[3] <= 0:
                raise ValueError(f"严格模式: {layer} 低频/动态节点维度不一致")
            sequence_lengths.add(int(temporal.shape[3]))
            if batch_size is None:
                batch_size = int(low.shape[0])
            elif int(low.shape[0]) != batch_size:
                raise ValueError("严格模式: RAG 三层 batch 维度不一致")
            if not torch.isfinite(low).all() or not torch.isfinite(temporal).all():
                raise ValueError(f"严格模式: {layer} 输入含 NaN/Inf")
            lows[layer] = low
            temporals[layer] = temporal
        assert batch_size is not None
        if len(sequence_lengths) != 1:
            raise ValueError("严格模式: 三层动态序列 T 必须一致")
        if self.seq_length is not None and sequence_lengths != {self.seq_length}:
            raise ValueError(
                f"严格模式: 动态序列 T={next(iter(sequence_lengths))} != 配置 {self.seq_length}"
            )
        return lows, temporals, batch_size

    @staticmethod
    def _batch_parents(
        parent_index: Mapping[str, torch.Tensor] | None, batch_size: int, device: torch.device,
        expected_sizes: Mapping[str, tuple[int, int]] | None = None,
    ) -> dict[str, torch.Tensor]:
        if parent_index is None:
            return {}
        unknown = set(parent_index) - {"road_to_syntax", "syntax_to_region"}
        if unknown:
            raise ValueError(f"严格模式: 未知 parent_index 字段 {sorted(unknown)}")
        output = {}
        for name, value in parent_index.items():
            tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
            tensor = tensor.to(device=device, dtype=torch.long)
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0).expand(batch_size, -1)
            if tensor.ndim != 2 or tensor.shape[0] not in {1, batch_size}:
                raise ValueError(f"严格模式: parent_index.{name} batch 维度错误")
            if tensor.shape[0] == 1 and batch_size > 1:
                tensor = tensor.expand(batch_size, -1)
            if expected_sizes is not None and name in expected_sizes:
                target_size, parent_size = expected_sizes[name]
                if tensor.shape[1] != target_size:
                    raise ValueError(f"严格模式: parent_index.{name} 节点数不匹配")
                if tensor.numel() and (int(tensor.min()) < 0 or int(tensor.max()) >= parent_size):
                    raise ValueError(f"严格模式: parent_index.{name} 含越界父节点索引")
            output[name] = tensor
        return output

    @staticmethod
    def _aggregate_parent(
        parent: torch.Tensor, target_size: int, indices: torch.Tensor | None,
    ) -> torch.Tensor:
        # parent [B,P,F], indices [B,target].  Missing assignment uses a
        # deterministic mean context, never a synthetic ID.
        batch, _, width = parent.shape
        if indices is None:
            return parent.mean(dim=1, keepdim=True).expand(batch, target_size, width)
        if indices.shape != (batch, target_size):
            raise ValueError("严格模式: parent assignment 与目标节点数不一致")
        return torch.gather(parent, 1, indices.unsqueeze(-1).expand(-1, -1, width))

    @staticmethod
    def _compose_parent_indices(
        first: torch.Tensor | None, second: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if first is None or second is None:
            return None
        return torch.gather(second, 1, first)

    def _hierarchy_contexts(
        self,
        lows: Mapping[str, torch.Tensor],
        temporal_embeddings: Mapping[str, torch.Tensor],
        parents: Mapping[str, torch.Tensor],
    ) -> dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
        road_to_syntax = parents.get("road_to_syntax")
        syntax_to_region = parents.get("syntax_to_region")
        syntax_region_low = self._aggregate_parent(
            lows["region"], lows["syntax"].shape[1], syntax_to_region
        )
        syntax_region_time = self._aggregate_parent(
            temporal_embeddings["region"], lows["syntax"].shape[1], syntax_to_region
        )
        road_syntax_low = self._aggregate_parent(
            lows["syntax"], lows["road"].shape[1], road_to_syntax
        )
        road_syntax_time = self._aggregate_parent(
            temporal_embeddings["syntax"], lows["road"].shape[1], road_to_syntax
        )
        road_region_index = self._compose_parent_indices(road_to_syntax, syntax_to_region)
        road_region_low = self._aggregate_parent(
            lows["region"], lows["road"].shape[1], road_region_index
        )
        road_region_time = self._aggregate_parent(
            temporal_embeddings["region"], lows["road"].shape[1], road_region_index
        )
        return {
            "region": {},
            "syntax": {"region": (syntax_region_low, syntax_region_time)},
            "road": {
                "syntax": (road_syntax_low, road_syntax_time),
                "region": (road_region_low, road_region_time),
            },
        }

    def _parent_reference_times(
        self,
        layer: str,
        retrieved_times: Mapping[str, torch.Tensor],
        lows: Mapping[str, torch.Tensor],
        parents: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if layer == "region":
            return {}
        output: dict[str, torch.Tensor] = {}
        if layer == "syntax":
            output["region"] = self._aggregate_parent(
                retrieved_times["region"], lows["syntax"].shape[1], parents.get("syntax_to_region")
            )
            return output
        output["syntax"] = self._aggregate_parent(
            retrieved_times["syntax"], lows["road"].shape[1], parents.get("road_to_syntax")
        )
        region_index = self._compose_parent_indices(
            parents.get("road_to_syntax"), parents.get("syntax_to_region")
        )
        output["region"] = self._aggregate_parent(
            retrieved_times["region"], lows["road"].shape[1], region_index
        )
        return output

    def _branch_input(
        self,
        layer: str,
        lows: Mapping[str, torch.Tensor],
        temporal_embeddings: Mapping[str, torch.Tensor],
        calendar_node: torch.Tensor,
        contexts: Mapping[str, Mapping[str, tuple[torch.Tensor, torch.Tensor]]],
        parent_reference_times: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        parts = [lows[layer], temporal_embeddings[layer], calendar_node]
        for parent in self._PARENTS[layer]:
            parent_low, parent_time = contexts[layer][parent]
            reference_time = (
                parent_reference_times[parent]
                if parent_reference_times is not None and parent in parent_reference_times
                else parent_time
            )
            parts.extend([parent_low, parent_time, reference_time])
        result = torch.cat(parts, dim=-1)
        if result.shape[-1] != self.branch_input_dims[layer]:
            raise RuntimeError(f"RAG {layer} query/key 输入维度内部不一致")
        return result

    @staticmethod
    def _calendar_at(calendar: Mapping[str, Any], index: int) -> dict[str, int]:
        result = {}
        for name, default in (("month", None), ("weekday", None), ("start_hour", None), ("holiday", 0)):
            value = calendar.get(name, default)
            if value is None:
                raise ValueError(f"严格模式: 缺少 calendar.{name}")
            tensor = torch.as_tensor(value).reshape(-1)
            if tensor.numel() == 1:
                result[name] = int(tensor[0].item())
            elif index < tensor.numel():
                result[name] = int(tensor[index].item())
            else:
                raise ValueError(f"严格模式: calendar.{name} batch 不足")
        return result

    def _source_entries(
        self, memory: ThreeLayerRAGMemory, expected_length: int | None = None
    ) -> dict[str, list[_SourceEntry]]:
        entries = {layer: [] for layer in LAYER_NAMES}
        device = next(self.parameters()).device
        source_lengths: set[int] = set()
        for snapshot in memory.snapshots:
            snapshot.validate(temporal_channels=self.temporal_channels)
            for layer in LAYER_NAMES:
                if snapshot.low_features[layer].shape[1] != self.low_dims[layer]:
                    raise ValueError(
                        f"严格模式: source {layer} low dimension "
                        f"{snapshot.low_features[layer].shape[1]} != {self.low_dims[layer]}"
                    )
            lows = {
                layer: snapshot.low_features[layer].unsqueeze(0).to(device)
                for layer in LAYER_NAMES
            }
            raw_temporals = {
                layer: snapshot.temporal_features[layer].unsqueeze(0).to(device)
                for layer in LAYER_NAMES
            }
            snapshot_lengths = {int(raw_temporals[layer].shape[3]) for layer in LAYER_NAMES}
            if len(snapshot_lengths) != 1:
                raise ValueError("严格模式: source snapshot 三层动态序列 T 必须一致")
            source_lengths.update(snapshot_lengths)
            temporal_embeddings = {
                layer: self.temporal_encoders[layer](raw_temporals[layer]) for layer in LAYER_NAMES
            }
            parents = self._batch_parents(
                snapshot.parent_index, 1, lows["road"].device,
                expected_sizes={
                    "road_to_syntax": (lows["road"].shape[1], lows["syntax"].shape[1]),
                    "syntax_to_region": (lows["syntax"].shape[1], lows["region"].shape[1]),
                },
            )
            contexts = self._hierarchy_contexts(lows, temporal_embeddings, parents)
            calendar = {name: snapshot.calendar.get(name, 0) for name in ("month", "weekday", "start_hour", "holiday")}
            calendar_tensor = self.calendar_encoder(calendar, 1, lows["road"].device)
            for layer in LAYER_NAMES:
                node_count = lows[layer].shape[1]
                calendar_node = calendar_tensor[:, None].expand(1, node_count, -1)
                # Source keys use the source parent temporal context.  Query
                # branches replace this with the coarser retrieved reference.
                branch_input = self._branch_input(
                    layer, lows, temporal_embeddings, calendar_node, contexts
                )
                key = F.normalize(self.key_projections[layer](branch_input[0]), dim=-1)
                for node in range(node_count):
                    entries[layer].append(_SourceEntry(
                        city_id=snapshot.city_id,
                        calendar=dict(calendar),
                        # Keep source keys attached to the current encoder
                        # graph so key projections/temporal encoders train.
                        key=key[node],
                        value=raw_temporals[layer][0, node].detach(),
                        temporal_embedding=temporal_embeddings[layer][0, node].detach(),
                    ))
        if len(source_lengths) != 1:
            raise ValueError("严格模式: source memory 的三层动态序列长度不一致")
        if expected_length is not None and source_lengths != {expected_length}:
            raise ValueError(
                f"严格模式: source memory T={next(iter(source_lengths))} != query T={expected_length}"
            )
        return entries

    def _matches_calendar(self, source: Mapping[str, int], target: Mapping[str, int]) -> bool:
        if int(source["weekday"]) != int(target["weekday"]):
            return False
        if int(source["start_hour"]) != int(target["start_hour"]):
            return False
        if self.match_month and int(source["month"]) != int(target["month"]):
            return False
        if self.match_holiday and int(source.get("holiday", 0)) != int(target.get("holiday", 0)):
            return False
        return True

    def _retrieve_vectorized(
        self,
        layer: str,
        query: torch.Tensor,
        entries: Sequence[_SourceEntry],
        calendar: Mapping[str, Any],
        batch_size: int,
        target_city: str | Sequence[str] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        outputs, time_outputs, all_weights, all_indices = [], [], [], []
        for batch_index in range(batch_size):
            target_calendar = self._calendar_at(calendar, batch_index)
            excluded_city = None
            if isinstance(target_city, str):
                excluded_city = target_city
            elif target_city is not None:
                excluded_city = str(target_city[batch_index])
            candidates = [
                entry for entry in entries
                if self._matches_calendar(entry.calendar, target_calendar)
                and (excluded_city is None or entry.city_id != excluded_city)
            ]
            if not candidates:
                raise LookupError(
                    f"严格模式: {layer} RAG 无 source-train 候选 "
                    f"month={target_calendar['month']} weekday={target_calendar['weekday']} "
                    f"hour={target_calendar['start_hour']}"
                )
            keys = torch.stack([entry.key for entry in candidates]).to(query.device, query.dtype)
            query_row = F.normalize(query[batch_index], dim=-1)
            keys = F.normalize(keys, dim=-1)
            if self.metric == "cosine":
                scores = query_row @ keys.transpose(0, 1)
            else:
                scores = -torch.cdist(query_row, keys)
            count = min(self.top_k, len(candidates))
            scores, selected = torch.topk(scores, k=count, dim=-1)
            weights = torch.softmax(scores / self.temperature, dim=-1)
            # scores [N,K], values [K,C,T] -> [N,C,T]
            # Candidate indices are shared by all query nodes in this batch;
            # gather values with the selected matrix for exact per-node top-k.
            value_bank = torch.stack([candidate.value for candidate in candidates]).to(query.device, query.dtype)
            latent_bank = torch.stack([candidate.temporal_embedding for candidate in candidates]).to(query.device, query.dtype)
            selected_values = value_bank[selected]
            selected_latent = latent_bank[selected]
            outputs.append(torch.einsum("nk,nkct->nct", weights, selected_values))
            time_outputs.append(torch.einsum("nk,nkd->nd", weights, selected_latent))
            all_weights.append(weights)
            all_indices.append(selected)
        return (
            torch.stack(outputs),
            torch.stack(time_outputs),
            {"weights": torch.stack(all_weights), "indices": torch.stack(all_indices)},
        )

    def forward(
        self,
        low_features: Mapping[str, torch.Tensor] | ThreeLayerRAGInputs | None = None,
        temporal_features: Mapping[str, torch.Tensor] | None = None,
        calendar: Mapping[str, Any] | None = None,
        *,
        memory: ThreeLayerRAGMemory | None = None,
        target_city: str | Sequence[str] | None = None,
        parent_index: Mapping[str, torch.Tensor] | None = None,
        high_features: Mapping[str, torch.Tensor] | None = None,
        inputs: ThreeLayerRAGInputs | None = None,
    ) -> dict[str, Any]:
        if high_features is not None:
            raise ValueError("严格模式: 高频 H_*_high 不能进入 RAG")
        if inputs is not None:
            if low_features is not None:
                raise ValueError("RAG 同时收到 inputs 和 low_features")
            low_features = inputs
        city_from_input = None
        if isinstance(low_features, ThreeLayerRAGInputs):
            city_from_input = low_features.city_id
            if temporal_features is not None or calendar is not None or parent_index is not None:
                raise ValueError("RAG ThreeLayerRAGInputs 不应再重复提供 temporal/calendar/parent")
            temporal_features = low_features.temporal_features
            calendar = low_features.calendar
            parent_index = low_features.parent_index
            low_features = low_features.low_features
        if low_features is None or temporal_features is None or calendar is None:
            raise ValueError("RAG 需要三层 low_features、temporal_features 和 calendar")
        if city_from_input is not None and target_city is None:
            target_city = city_from_input
        assert_no_high_frequency_input(low_features)
        lows, raw_temporals, batch_size = self._validate_batch(low_features, temporal_features)
        device = next(self.parameters()).device
        lows = {layer: value.to(device) for layer, value in lows.items()}
        raw_temporals = {layer: value.to(device) for layer, value in raw_temporals.items()}
        parents = self._batch_parents(
            parent_index, batch_size, device,
            expected_sizes={
                "road_to_syntax": (lows["road"].shape[1], lows["syntax"].shape[1]),
                "syntax_to_region": (lows["syntax"].shape[1], lows["region"].shape[1]),
            },
        )
        temporal_embeddings = {
            layer: self.temporal_encoders[layer](raw_temporals[layer]) for layer in LAYER_NAMES
        }
        calendar_embedding = self.calendar_encoder(calendar, batch_size, device)
        calendar_nodes = {
            layer: calendar_embedding[:, None].expand(-1, lows[layer].shape[1], -1)
            for layer in LAYER_NAMES
        }
        contexts = self._hierarchy_contexts(lows, temporal_embeddings, parents)
        active_memory = memory or self.memory
        if active_memory is None:
            raise RuntimeError("严格模式: RAG 尚未设置 source-train memory")
        active_memory.validate()
        query_length = int(raw_temporals["region"].shape[3])
        source_entries = self._source_entries(active_memory, expected_length=query_length)

        retrieved_values: dict[str, torch.Tensor] = {}
        retrieved_times: dict[str, torch.Tensor] = {}
        diagnostics: dict[str, dict[str, torch.Tensor]] = {}
        query_features: dict[str, torch.Tensor] = {}
        query_keys: dict[str, torch.Tensor] = {}
        # Region -> Syntax -> Road is deliberate coarse-to-fine ordering.
        for layer in ("region", "syntax", "road"):
            parent_refs = self._parent_reference_times(layer, retrieved_times, lows, parents)
            branch_input = self._branch_input(
                layer, lows, temporal_embeddings, calendar_nodes[layer], contexts, parent_refs
            )
            query_features[layer] = branch_input
            query = F.normalize(self.query_projections[layer](branch_input), dim=-1)
            query_keys[layer] = query
            values, times, info = self._retrieve_vectorized(
                layer, query, source_entries[layer], calendar, batch_size, target_city
            )
            retrieved_values[layer] = values
            retrieved_times[layer] = times
            diagnostics[layer] = info

        return {
            "R_road": retrieved_values["road"],
            "R_syntax": retrieved_values["syntax"],
            "R_region": retrieved_values["region"],
            "R_road_time": retrieved_times["road"],
            "R_syntax_time": retrieved_times["syntax"],
            "R_region_time": retrieved_times["region"],
            "query_road": query_keys["road"],
            "query_syntax": query_keys["syntax"],
            "query_region": query_keys["region"],
            "retrieval": diagnostics,
            "calendar_embedding": calendar_embedding,
            "source_cities": active_memory.source_cities,
            "city_id": city_from_input,
        }


# Concise alias used by downstream stage-3 code and configuration docs.
ThreeLayerRAG = HierarchicalThreeLayerRAG
HierarchicalRAG = HierarchicalThreeLayerRAG
