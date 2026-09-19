"""Condition encoders and sparse coarse-to-fine broadcasting."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from three_layer_rag.model import CalendarEncoder, TemporalSequenceEncoder


def broadcast_parent_to_children(
    parent: torch.Tensor,
    operator: Mapping[str, torch.Tensor],
    *,
    child_count: int,
) -> torch.Tensor:
    """Apply the transpose of a ``row=parent,col=child`` sparse operator.

    Each child's incoming weights are normalized independently.  The function
    accepts arbitrary trailing dimensions and never materializes a dense
    parent-by-child matrix.
    """

    if parent.ndim < 3:
        raise ValueError("parent 必须是 [B,P,...]")
    if child_count <= 0:
        raise ValueError("child_count 必须为正")
    if not isinstance(operator, Mapping) or set(operator) != {"edge_index", "weight"}:
        raise ValueError("parent operator 必须只包含 edge_index/weight")
    edge = torch.as_tensor(operator["edge_index"], device=parent.device, dtype=torch.long)
    weight = torch.as_tensor(operator["weight"], device=parent.device, dtype=parent.dtype)
    if edge.ndim != 2 or edge.shape[0] != 2 or weight.shape != (edge.shape[1],):
        raise ValueError("parent operator shape 非法")
    if edge.shape[1] == 0:
        raise ValueError("parent operator 不能为空")
    parent_index, child_index = edge[0], edge[1]
    if (
        int(parent_index.min()) < 0
        or int(parent_index.max()) >= parent.shape[1]
        or int(child_index.min()) < 0
        or int(child_index.max()) >= child_count
    ):
        raise ValueError("parent operator 索引越界")
    if not torch.isfinite(weight).all() or (weight <= 0).any():
        raise ValueError("parent operator 权重必须为有限正数")
    denominator = parent.new_zeros(child_count)
    denominator.index_add_(0, child_index, weight)
    if (denominator <= 0).any():
        missing = torch.nonzero(denominator <= 0, as_tuple=False).reshape(-1).tolist()
        raise ValueError(f"严格模式: parent operator 存在无父边 child: {missing[:10]}")
    normalized = weight / denominator[child_index]
    flat_width = int(parent[0, 0].numel())
    parent_flat = parent.reshape(parent.shape[0], parent.shape[1], flat_width)
    result = parent.new_zeros((parent.shape[0], child_count, flat_width))
    for batch_index in range(parent.shape[0]):
        messages = parent_flat[batch_index, parent_index] * normalized.unsqueeze(-1)
        result[batch_index].index_add_(0, child_index, messages)
    return result.reshape(parent.shape[0], child_count, *parent.shape[2:])


class LayerDiffusionConditioner(nn.Module):
    """Build one node condition without accepting low-frequency features."""

    def __init__(
        self,
        *,
        high_dim: int,
        channels: int,
        cond_dim: int,
        temporal_dim: int,
        calendar_dims: Mapping[str, int],
        parent_channels: int | None = None,
        parent_temporal_dim: int | None = None,
        temporal_layers: int = 1,
        temporal_dropout: float = 0.0,
    ):
        super().__init__()
        self.high_dim = int(high_dim)
        self.channels = int(channels)
        self.cond_dim = int(cond_dim)
        self.reference_encoder = TemporalSequenceEncoder(
            channels, temporal_dim, layers=temporal_layers, dropout=temporal_dropout
        )
        self.parent_encoder = None
        parent_width = 0
        if parent_channels is not None:
            parent_width = int(parent_temporal_dim or temporal_dim)
            self.parent_encoder = TemporalSequenceEncoder(
                parent_channels,
                parent_width,
                layers=temporal_layers,
                dropout=temporal_dropout,
            )
        self.calendar_encoder = CalendarEncoder(
            month_dim=int(calendar_dims["month"]),
            weekday_dim=int(calendar_dims["weekday"]),
            hour_dim=int(calendar_dims["start_hour"]),
            holiday_dim=int(calendar_dims["holiday"]),
        )
        input_dim = high_dim + temporal_dim + self.calendar_encoder.output_dim + parent_width
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    @staticmethod
    def _batched(value: torch.Tensor, rank_without_batch: int, name: str) -> torch.Tensor:
        if value.ndim == rank_without_batch:
            value = value.unsqueeze(0)
        if value.ndim != rank_without_batch + 1:
            raise ValueError(f"{name} rank 非法")
        return value

    def forward(
        self,
        high_feature: torch.Tensor,
        rag_reference: torch.Tensor,
        calendar: Mapping[str, Any],
        *,
        parent_dynamic: torch.Tensor | None = None,
    ) -> torch.Tensor:
        high_feature = self._batched(high_feature, 2, "high_feature")
        rag_reference = self._batched(rag_reference, 3, "rag_reference")
        if high_feature.shape[:2] != rag_reference.shape[:2]:
            raise ValueError("high feature 与 RAG reference 的 B/N 不一致")
        if high_feature.shape[-1] != self.high_dim:
            raise ValueError("high feature hidden dimension 不匹配")
        if rag_reference.shape[2] != self.channels:
            raise ValueError("RAG reference channel 不匹配")
        if not torch.isfinite(high_feature).all() or not torch.isfinite(rag_reference).all():
            raise ValueError("严格模式: Diffusion condition 输入含 NaN/Inf")
        batch, nodes = high_feature.shape[:2]
        reference = self.reference_encoder(rag_reference)
        calendar_embedding = self.calendar_encoder(calendar, batch, high_feature.device)
        calendar_nodes = calendar_embedding[:, None].expand(-1, nodes, -1)
        parts = [high_feature, reference, calendar_nodes]
        if self.parent_encoder is None:
            if parent_dynamic is not None:
                raise ValueError("Region condition 不接受父层动态")
        else:
            if parent_dynamic is None:
                raise ValueError("Syntax/Road condition 缺少父层动态")
            parent_dynamic = self._batched(parent_dynamic, 3, "parent_dynamic")
            if parent_dynamic.shape[:2] != (batch, nodes):
                raise ValueError("父层广播动态与子层 B/N 不一致")
            parts.append(self.parent_encoder(parent_dynamic))
        return self.projection(torch.cat(parts, dim=-1))


def flatten_nodes(value: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    """Convert ``[B,N,C,T]`` to ``[B*N,C,T]`` with an audit shape."""

    if value.ndim != 4:
        raise ValueError("flatten_nodes 期望 [B,N,C,T]")
    batch, nodes, channels, length = value.shape
    return value.reshape(batch * nodes, channels, length), (batch, nodes)


def flatten_node_conditions(value: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    if value.ndim != 3:
        raise ValueError("flatten_node_conditions 期望 [B,N,D]")
    batch, nodes, width = value.shape
    return value.reshape(batch * nodes, width), (batch, nodes)


def unflatten_nodes(value: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    if value.ndim != 3 or value.shape[0] != shape[0] * shape[1]:
        raise ValueError("unflatten_nodes shape 不匹配")
    return value.reshape(shape[0], shape[1], value.shape[1], value.shape[2])
