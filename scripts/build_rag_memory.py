#!/usr/bin/env python
"""Build a source-train three-layer RAG memory from a local tensor manifest.

The input is deliberately a local ``torch.save`` bundle and is not committed
to Git.  It may be either ``[mapping, ...]`` or ``{"snapshots": [...]}``; each
mapping follows :class:`three_layer_rag.ThreeLayerRAGInputs`:

``city_id, split, low_features, temporal_features, value_temporal_features,
calendar, parent_index/parent_operator, graph_metadata``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from three_layer_rag import ThreeLayerRAGInputs, ThreeLayerRAGMemory


def _load_snapshots(path: Path) -> list[ThreeLayerRAGInputs]:
    # Snapshot bundles deliberately contain ThreeLayerRAGInputs dataclasses,
    # not only tensor weights.  Be explicit for PyTorch versions whose
    # weights_only default changed.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "snapshots" in payload:
        payload = payload["snapshots"]
    if not isinstance(payload, (list, tuple)):
        raise ValueError("输入 bundle 必须是 snapshot list 或 {'snapshots': list}")
    snapshots = []
    for index, item in enumerate(payload):
        if isinstance(item, ThreeLayerRAGInputs):
            snapshots.append(item)
            continue
        if not isinstance(item, dict):
            raise ValueError(f"snapshot[{index}] 不是 mapping")
        required = {
            "city_id", "split", "low_features", "temporal_features",
            "value_temporal_features", "calendar",
        }
        missing = required - set(item)
        if missing:
            raise KeyError(f"snapshot[{index}] 缺少字段 {sorted(missing)}")
        snapshots.append(ThreeLayerRAGInputs(
            city_id=str(item["city_id"]),
            split=str(item["split"]),
            low_features=item["low_features"],
            temporal_features=item["temporal_features"],
            calendar=item["calendar"],
            parent_index=item.get("parent_index"),
            parent_operator=item.get("parent_operator"),
            value_temporal_features=item["value_temporal_features"],
            graph_metadata=item.get("graph_metadata"),
        ))
    return snapshots


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local source-train three-layer RAG memory")
    parser.add_argument("--input", required=True, help="local torch.save snapshot bundle")
    parser.add_argument("--output", required=True, help="local output .pt memory path")
    parser.add_argument("--source-cities", nargs="+", required=True)
    parser.add_argument("--require-graph-identity", action="store_true")
    args = parser.parse_args()
    snapshots = _load_snapshots(Path(args.input))
    memory = ThreeLayerRAGMemory.from_snapshots(
        snapshots, source_cities=args.source_cities, split="train",
        require_value_separation=True,
        require_graph_identity=args.require_graph_identity,
    )
    memory.save(args.output)
    print({
        "output": str(Path(args.output)),
        "version": memory.version,
        "source_cities": memory.source_cities,
        "num_snapshots": len(memory.snapshots),
    })


if __name__ == "__main__":
    main()
