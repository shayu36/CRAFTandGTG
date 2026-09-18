"""Adapters between local Stage-2 exports/static caches and RAG contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from .contracts import ThreeLayerRAGInputs


STAGE2_SPECTRAL_VERSION = "three-layer-joint-graphgps-spectral-features-v2"


def load_stage2_low_features(
    path: str | Path,
    *,
    expected_city_id: str | None = None,
    expected_checkpoint_fingerprint: str | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load only ``H_*_low`` and preserve Stage-2 identity metadata."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("严格模式: Stage-2 spectral feature 必须是 mapping")
    if payload.get("format_version") != STAGE2_SPECTRAL_VERSION:
        raise ValueError(
            "严格模式: RAG 只接受 three-layer-joint-graphgps-spectral-features-v2，旧 v1 不得混用"
        )
    if expected_city_id is not None and payload.get("city_id") != expected_city_id:
        raise ValueError("严格模式: Stage-2 feature city_id 不匹配")
    if (
        expected_checkpoint_fingerprint is not None
        and payload.get("checkpoint_fingerprint") != expected_checkpoint_fingerprint
    ):
        raise ValueError("严格模式: Stage-2 checkpoint fingerprint 不匹配")
    low = {}
    for layer in ("road", "syntax", "region"):
        key = f"H_{layer}_low"
        value = payload.get(key)
        if not isinstance(value, torch.Tensor) or value.ndim != 2 or not torch.isfinite(value).all():
            raise ValueError(f"严格模式: Stage-2 缺少有限二维低频字段 {key}")
        low[layer] = value.detach().cpu().float().contiguous()
    expected_ranges = {
        "road": payload.get("road_node_range"),
        "syntax": payload.get("syntax_node_range"),
        "region": payload.get("region_node_range"),
    }
    for layer, node_range in expected_ranges.items():
        if node_range is None or len(node_range) != 2 or int(node_range[1]) - int(node_range[0]) != low[layer].shape[0]:
            raise ValueError(f"严格模式: Stage-2 {layer} low 节点数与 node range 不一致")
    identity = {
        name: payload[name]
        for name in (
            "joint_graph_hash", "checkpoint_fingerprint", "static_feature_version",
            "format_version", "road_node_range", "syntax_node_range", "region_node_range",
        )
        if name in payload
    }
    identity["spectral_feature_version"] = payload.get("format_version")
    identity["city_id"] = payload.get("city_id")
    return low, identity


def parent_operators_from_hierarchy(hierarchy: Any) -> dict[str, dict[str, torch.Tensor]]:
    """Use static hierarchy sparse weighted operators without collapsing them."""

    return {
        "road_to_syntax": {
            "edge_index": hierarchy.road_to_syntax_edge_index.detach().cpu().long(),
            "weight": hierarchy.road_to_syntax_weight.detach().cpu().float(),
        },
        "syntax_to_region": {
            "edge_index": hierarchy.syntax_to_region_edge_index.detach().cpu().long(),
            "weight": hierarchy.syntax_to_region_weight.detach().cpu().float(),
        },
    }


def build_rag_inputs_from_local_artifacts(
    *,
    spectral_path: str | Path,
    history_temporal_features: Mapping[str, torch.Tensor],
    value_temporal_features: Mapping[str, torch.Tensor] | None,
    calendar: Mapping[str, Any],
    city_id: str,
    split: str,
    hierarchy: Any,
) -> ThreeLayerRAGInputs:
    """Connect an exported Stage-2 city with locally-built dynamic tensors."""

    if getattr(hierarchy, "city_id", city_id) != city_id:
        raise ValueError("严格模式: hierarchy city_id 与 Stage-2 feature city_id 不一致")
    low, identity = load_stage2_low_features(spectral_path, expected_city_id=city_id)
    graph_metadata = dict(identity)
    graph_metadata["region_node_range"] = tuple(graph_metadata.get("region_node_range", ()))
    graph_metadata["syntax_node_range"] = tuple(graph_metadata.get("syntax_node_range", ()))
    graph_metadata["road_node_range"] = tuple(graph_metadata.get("road_node_range", ()))
    return ThreeLayerRAGInputs(
        city_id=city_id,
        split=split,
        low_features=low,
        temporal_features=history_temporal_features,
        calendar=calendar,
        parent_operator=parent_operators_from_hierarchy(hierarchy),
        value_temporal_features=value_temporal_features,
        graph_metadata=graph_metadata,
    ).validate()
