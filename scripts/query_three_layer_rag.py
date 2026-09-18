#!/usr/bin/env python
"""Run one local three-layer RAG query.

The query bundle is a ``ThreeLayerRAGInputs`` object or a mapping with the same
fields.  It must contain history temporal features; Value tensors are required
only for source memory construction, never for a target query.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from three_layer_rag import (  # noqa: E402
    HierarchicalThreeLayerRAG,
    ThreeLayerRAGInputs,
    ThreeLayerRAGMemory,
)


def _snapshot(payload: object) -> ThreeLayerRAGInputs:
    if isinstance(payload, ThreeLayerRAGInputs):
        return payload
    if not isinstance(payload, dict):
        raise ValueError("query bundle 必须是 ThreeLayerRAGInputs 或 mapping")
    return ThreeLayerRAGInputs(
        city_id=str(payload["city_id"]),
        split=str(payload.get("split", "test")),
        low_features=payload["low_features"],
        temporal_features=payload["temporal_features"],
        calendar=payload["calendar"],
        parent_index=payload.get("parent_index"),
        parent_operator=payload.get("parent_operator"),
        value_temporal_features=payload.get("value_temporal_features"),
        graph_metadata=payload.get("graph_metadata"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one local hierarchical three-layer RAG query")
    parser.add_argument("--config", required=True)
    parser.add_argument("--memory", required=True)
    parser.add_argument("--query", required=True, help="local torch.save query bundle")
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-city")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    rag_cfg = config["rag"]
    memory = ThreeLayerRAGMemory.load(args.memory)
    if bool(rag_cfg.get("require_value_separation", True)) and not memory.require_value_separation:
        raise ValueError("严格模式: 当前配置要求 history/Value 分离，但 memory 未启用该契约")
    if bool(rag_cfg.get("require_graph_identity", False)) and not memory.require_graph_identity:
        raise ValueError("严格模式: 当前配置要求 graph identity，但 memory 未启用该契约")
    query_payload = torch.load(Path(args.query), map_location="cpu", weights_only=False)
    query = _snapshot(query_payload)
    model = HierarchicalThreeLayerRAG(
        low_dim=int(rag_cfg["low_dim"]),
        temporal_channels=rag_cfg["temporal_channels"],
        temporal_dim=int(rag_cfg["temporal_dim"]),
        retrieval_dim=int(rag_cfg["retrieval_dim"]),
        seq_length=int(rag_cfg["seq_length"]),
        temporal_layers=int(rag_cfg["temporal_layers"]),
        temporal_dropout=float(rag_cfg["temporal_dropout"]),
        calendar_dims=rag_cfg["calendar_dims"],
        top_k=int(rag_cfg["top_k"]),
        metric=str(rag_cfg["metric"]),
        temperature=float(rag_cfg["temperature"]),
        match_month=bool(rag_cfg["match_month"]),
        match_holiday=bool(rag_cfg["match_holiday"]),
        require_value_separation=bool(rag_cfg.get("require_value_separation", True)),
        # Source memories may contain different city graph hashes.  Their own
        # strict metadata is validated by ThreeLayerRAGMemory.load(); a target
        # city's hash is not required to equal a source city's hash.
        expected_graph_identity=None,
    )
    output = model(query, memory=memory, target_city=args.target_city or query.city_id)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, destination)
    print({
        "output": str(destination),
        "city_id": query.city_id,
        "source_cities": memory.source_cities,
        "shapes": {name: list(output[name].shape) for name in ("R_road", "R_syntax", "R_region")},
    })


if __name__ == "__main__":
    main()
