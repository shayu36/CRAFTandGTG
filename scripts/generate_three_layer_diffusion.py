#!/usr/bin/env python3
"""Generate Region -> Syntax -> Road futures without conditioning on truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from three_layer_diffusion import (  # noqa: E402
    DynamicNormalizer,
    ModelEMA,
    build_system,
    file_sha256,
    load_diffusion_checkpoint,
    load_snapshot_bundle,
    load_stage2_high_features,
    three_layer_physical_metrics,
    validate_runtime_identities,
)
from three_layer_rag import ThreeLayerRAGMemory  # noqa: E402


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate hierarchical conditional Diffusion outputs")
    parser.add_argument("--config", default="configs/stage4_three_layer_diffusion.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-snapshots", type=int)
    args = parser.parse_args()

    config = yaml.safe_load(_path(args.config).read_text(encoding="utf-8"))
    seed = int(config.get("seed", 2026))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"请求了 {args.device}，但当前 PyTorch CUDA 不可用")
    data_cfg = config["data"]
    snapshots, bundle_metadata = load_snapshot_bundle(
        _path(args.input), splits=data_cfg["generation_splits"]
    )
    memory_path = _path(data_cfg["rag_memory"])
    memory = ThreeLayerRAGMemory.load(memory_path)
    configured_sources = tuple(sorted(str(city) for city in data_cfg["source_cities"]))
    if configured_sources != tuple(sorted(memory.source_cities)):
        raise ValueError("严格模式: config source_cities 与 RAG memory 不一致")
    high, high_identities = {}, {}
    for snapshot in snapshots:
        if snapshot.city_id in high:
            continue
        feature, identity = load_stage2_high_features(
            _path(data_cfg["spectral_feature_dir"]) / f"{snapshot.city_id}_spectral_features.pt",
            expected_city_id=snapshot.city_id,
            expected_graph_metadata=snapshot.graph_metadata,
            expected_low_features=snapshot.low_features,
        )
        high[snapshot.city_id], high_identities[snapshot.city_id] = feature, identity
    identity = validate_runtime_identities(snapshots, high_identities, memory)
    identity["rag_memory_sha256"] = file_sha256(memory_path)
    normalizer = DynamicNormalizer.load(_path(data_cfg["dynamic_normalizer"]))
    normalizer.validate_bundle_metadata(bundle_metadata)
    identity["dynamic_normalizer_fingerprint"] = normalizer.fingerprint
    # A trained checkpoint is bound to the complete source-memory graph set.
    # A target-only input bundle is validated above, but must not replace that
    # source identity with a one-city subset during checkpoint loading.
    checkpoint_identity = dict(identity)
    checkpoint_identity["joint_graph_hashes"] = {
        city: str(metadata["joint_graph_hash"])
        for city, metadata in dict(memory.graph_identity or {}).items()
    }
    limit = args.max_snapshots
    if limit is None:
        limit = config.get("generation", {}).get("max_snapshots")
    if limit is not None:
        snapshots = snapshots[:int(limit)]

    system = build_system(config).to(device)
    ema = ModelEMA(
        system,
        decay=float(config["training"].get("ema_decay", 0.995)),
        update_every=int(config["training"].get("ema_update_every", 10)),
    ).to(device)
    load_diffusion_checkpoint(
        _path(args.checkpoint),
        system,
        ema=ema,
        expected_identity=checkpoint_identity,
        use_ema_weights=bool(config.get("generation", {}).get("use_ema", True)),
    )
    system.eval()
    destination = _path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, snapshot in enumerate(snapshots):
        # ThreeLayerRAGDiffusionSystem.generate reconstructs a query with
        # value_temporal_features=None before retrieval and denoising.
        output = system.generate(
            snapshot,
            high_features=high[snapshot.city_id],
            memory=memory,
        )
        normalized = {
            "region": output["generated_region"].cpu(),
            "syntax": output["generated_syntax"].cpu(),
            "road": output["generated_road"].cpu(),
        }
        physical = {
            layer: normalizer.inverse(layer, value) for layer, value in normalized.items()
        }
        payload = {
            "format_version": "three-layer-hierarchical-diffusion-generation-v1",
            "city_id": snapshot.city_id,
            "split": snapshot.split,
            "calendar": dict(snapshot.calendar),
            "graph_metadata": dict(snapshot.graph_metadata or {}),
            "normalizer_fingerprint": normalizer.fingerprint,
            "used_future_value": output["used_future_value"],
            "generation_order": output["generation_order"],
            "sampling": output["sampling"],
        }
        if bool(config.get("generation", {}).get("save_normalized", True)):
            payload["normalized"] = normalized
        if bool(config.get("generation", {}).get("save_physical", True)):
            payload["physical"] = physical
        if snapshot.value_temporal_features is not None:
            target = {
                layer: snapshot.value_temporal_features[layer].unsqueeze(0)
                for layer in ("region", "syntax", "road")
            }
            # Truth is read only after generation, solely for offline metrics.
            payload["offline_metrics"] = three_layer_physical_metrics(
                normalized,
                target,
                normalizer,
                layer_weights=config["evaluation"]["layer_weights"],
            )
        filename = f"{index:05d}_{snapshot.city_id}_{snapshot.split}.pt"
        torch.save(payload, destination / filename)
        manifest.append({
            "file": filename,
            "city_id": snapshot.city_id,
            "split": snapshot.split,
            "used_future_value": False,
            "sampling": output["sampling"],
            "offline_metrics": payload.get("offline_metrics"),
        })
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(destination),
        "num_snapshots": len(manifest),
        "normalizer_fingerprint": normalizer.fingerprint,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
