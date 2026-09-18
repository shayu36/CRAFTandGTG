#!/usr/bin/env python
"""Assemble one RAG snapshot from local Stage-2/static/dynamic artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from static_hierarchy.data import load_city_static_hierarchy  # noqa: E402
from three_layer_rag import build_rag_inputs_from_local_artifacts  # noqa: E402


def _mapping(path: Path, name: str):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and name in payload:
        payload = payload[name]
    if not isinstance(payload, dict):
        raise ValueError(f"{path} 必须保存 mapping 或包含 {name} 的 mapping")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one local three-layer RAG input snapshot")
    parser.add_argument("--spectral-feature", required=True)
    parser.add_argument("--hierarchy-cache-dir", required=True)
    parser.add_argument("--city", required=True)
    parser.add_argument("--history", required=True, help="torch.save history_temporal_features mapping")
    parser.add_argument("--value", required=True, help="torch.save future/value temporal mapping")
    parser.add_argument("--calendar", required=True, help="JSON object with month/weekday/start_hour")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    hierarchy = load_city_static_hierarchy(args.hierarchy_cache_dir, args.city)
    history = _mapping(Path(args.history), "history_temporal_features")
    value = _mapping(Path(args.value), "value_temporal_features")
    calendar = json.loads(Path(args.calendar).read_text(encoding="utf-8"))
    snapshot = build_rag_inputs_from_local_artifacts(
        spectral_path=args.spectral_feature,
        history_temporal_features=history,
        value_temporal_features=value,
        calendar=calendar,
        city_id=args.city,
        split=args.split,
        hierarchy=hierarchy,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(snapshot, destination)
    print({"output": str(destination), "city_id": snapshot.city_id, "split": snapshot.split})


if __name__ == "__main__":
    main()

