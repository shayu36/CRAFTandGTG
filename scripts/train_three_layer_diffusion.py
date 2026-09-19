#!/usr/bin/env python3
"""Jointly train hierarchical RAG and three-layer conditional Diffusion."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from three_layer_diffusion import (  # noqa: E402
    CityGraphBucketBatchSampler,
    DynamicNormalizer,
    ModelEMA,
    SnapshotDataset,
    ThreeLayerDiffusionTrainer,
    build_system,
    collate_city_graph_bucket,
    load_diffusion_checkpoint,
    load_snapshot_bundle,
    load_stage2_high_features,
    save_diffusion_checkpoint,
    three_layer_physical_metrics,
    validate_runtime_identities,
)
from three_layer_rag import ThreeLayerRAGMemory  # noqa: E402


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"请求了 {name}，但当前 PyTorch CUDA 不可用")
    return device


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_high(
    snapshots: list[Any], directory: Path
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, dict[str, Any]]]:
    features, identities = {}, {}
    by_city = {}
    for snapshot in snapshots:
        by_city.setdefault(snapshot.city_id, snapshot)
    for city, snapshot in by_city.items():
        feature, identity = load_stage2_high_features(
            directory / f"{city}_spectral_features.pt",
            expected_city_id=city,
            expected_graph_metadata=snapshot.graph_metadata,
            expected_low_features=snapshot.low_features,
        )
        features[city], identities[city] = feature, identity
    return features, identities


@torch.no_grad()
def _evaluate_noise(
    system: torch.nn.Module,
    snapshots: list[Any],
    high: Mapping[str, Mapping[str, torch.Tensor]],
    memory: Any,
    validation_bank: list[Mapping[str, Mapping[str, torch.Tensor]]],
) -> dict[str, float]:
    system.eval()
    totals, layers = [], {name: [] for name in ("region", "syntax", "road")}
    topk, top_weight = [], []
    for index, snapshot in enumerate(snapshots):
        bank = validation_bank[index]
        result = system.training_loss(
            snapshot,
            high_features=high[snapshot.city_id],
            memory=memory,
            noise={layer: bank[layer]["noise"] for layer in ("region", "syntax", "road")},
            timesteps={layer: bank[layer]["timesteps"] for layer in ("region", "syntax", "road")},
            force_self_condition=False,
        )
        totals.append(result["loss"].detach())
        for layer in layers:
            layers[layer].append(result["layer_losses"][layer].detach())
            weights = result["rag"]["retrieval"][layer]["weights"]
            topk.append(float(weights.shape[-1]))
            top_weight.append(float(weights.max(dim=-1).values.mean()))
    output = {"normalized_noise_loss": float(torch.stack(totals).mean())}
    output.update({
        f"{layer}_normalized_noise_loss": float(torch.stack(values).mean())
        for layer, values in layers.items()
    })
    output["rag_mean_actual_top_k"] = sum(topk) / len(topk)
    output["rag_mean_top1_weight"] = sum(top_weight) / len(top_weight)
    return output


def _build_validation_bank(
    snapshots: list[Any], *, time_steps: int, seed: int
) -> list[dict[str, dict[str, torch.Tensor]]]:
    """Create epoch-independent timestep/noise draws for stable model selection."""

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    channels = {"region": 2, "syntax": 3, "road": 3}
    bank = []
    for snapshot in snapshots:
        layers = {}
        for layer, count in channels.items():
            target = snapshot.value_temporal_features[layer]
            if target.ndim != 3 or target.shape[1] != count:
                raise ValueError(f"validation {layer} shape 不符合固定 noise bank 契约")
            layers[layer] = {
                "timesteps": torch.randint(0, time_steps, (target.shape[0],), generator=generator),
                "noise": torch.randn(target.shape, generator=generator),
            }
        bank.append(layers)
    return bank


@torch.no_grad()
def _evaluate_generation(
    system: torch.nn.Module,
    snapshots: list[Any],
    high: Mapping[str, Mapping[str, torch.Tensor]],
    memory: Any,
    normalizer: DynamicNormalizer,
    layer_weights: Mapping[str, float],
) -> dict[str, Any]:
    collected = []
    sampling = None
    for snapshot in snapshots:
        output = system.generate(snapshot, high_features=high[snapshot.city_id], memory=memory)
        prediction = {
            "region": output["generated_region"].cpu(),
            "syntax": output["generated_syntax"].cpu(),
            "road": output["generated_road"].cpu(),
        }
        target = {
            layer: snapshot.value_temporal_features[layer].unsqueeze(0)
            for layer in ("region", "syntax", "road")
        }
        collected.append(three_layer_physical_metrics(
            prediction, target, normalizer, layer_weights=layer_weights
        ))
        sampling = output["sampling"]
    layers = {}
    for layer in ("region", "syntax", "road"):
        keys = collected[0]["layers"][layer]
        layers[layer] = {
            key: sum(item["layers"][layer][key] for item in collected) / len(collected)
            for key in keys
        }
    return {
        "num_generated_snapshots": len(collected),
        "layers": layers,
        "weighted_mae": sum(item["weighted_mae"] for item in collected) / len(collected),
        "weighted_rmse": sum(item["weighted_rmse"] for item in collected) / len(collected),
        "sampling": sampling,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train hierarchical RAG + conditional Diffusion")
    parser.add_argument("--config", default="configs/stage4_three_layer_diffusion.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--max-train-snapshots", type=int)
    parser.add_argument("--max-validation-snapshots", type=int)
    args = parser.parse_args()

    config = yaml.safe_load(_path(args.config).read_text(encoding="utf-8"))
    seed = int(config.get("seed", 2026))
    _seed_all(seed)
    device = _device(args.device)
    data_cfg, training_cfg = config["data"], config["training"]
    train, train_metadata = load_snapshot_bundle(
        _path(data_cfg["train_snapshots"]), splits=data_cfg["train_splits"]
    )
    validation, validation_metadata = load_snapshot_bundle(
        _path(data_cfg["eval_snapshots"]), splits=data_cfg["validation_splits"]
    )
    if args.max_train_snapshots is not None:
        train = train[:args.max_train_snapshots]
    if args.max_validation_snapshots is not None:
        validation = validation[:args.max_validation_snapshots]
    memory = ThreeLayerRAGMemory.load(_path(data_cfg["rag_memory"]))
    configured_sources = tuple(sorted(str(city) for city in data_cfg["source_cities"]))
    if configured_sources != tuple(sorted(memory.source_cities)):
        raise ValueError("严格模式: config source_cities 与 RAG memory 不一致")
    all_snapshots = train + validation
    high, high_identities = _load_high(
        all_snapshots, _path(data_cfg["spectral_feature_dir"])
    )
    identity = validate_runtime_identities(all_snapshots, high_identities, memory)
    normalizer = DynamicNormalizer.load(_path(data_cfg["dynamic_normalizer"]))
    normalizer.validate_bundle_metadata(train_metadata)
    normalizer.validate_bundle_metadata(validation_metadata)
    identity["dynamic_normalizer_fingerprint"] = normalizer.fingerprint

    system = build_system(config).to(device)
    optimizer = torch.optim.AdamW(
        system.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(training_cfg.get("scheduler_factor", 0.9)),
        patience=int(training_cfg.get("scheduler_patience", 10)),
    )
    ema = ModelEMA(
        system,
        decay=float(training_cfg.get("ema_decay", 0.995)),
        update_every=int(training_cfg.get("ema_update_every", 10)),
    ).to(device)
    trainer = ThreeLayerDiffusionTrainer(
        system,
        optimizer,
        memory=memory,
        high_features=high,
        ema=ema,
        scheduler=scheduler,
        gradient_clip_norm=float(training_cfg.get("gradient_clip_norm", 1.0)),
    )
    start_epoch = 0
    best = float("inf")
    history = []
    if args.resume:
        payload = load_diffusion_checkpoint(
            _path(args.resume),
            system,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_identity=identity,
        )
        start_epoch = int(payload["epoch"]) + 1
        trainer.global_step = int(payload["global_step"])
        if payload.get("best_metric") is not None:
            best = float(payload["best_metric"])
        history = list(payload.get("history", []))

    dataset = SnapshotDataset(train)
    batch_sampler = CityGraphBucketBatchSampler(
        train,
        batch_size=int(training_cfg.get("city_bucket_size", 1)),
        shuffle=True,
        seed=seed,
    )
    loader = DataLoader(
        dataset, batch_sampler=batch_sampler, collate_fn=collate_city_graph_bucket
    )
    epochs = int(args.max_epochs or training_cfg["epochs"])
    output_dir = _path(config["outputs"]["directory"])
    best_path = _path(config["outputs"]["best_checkpoint"])
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_bank = _build_validation_bank(
        validation,
        time_steps=int(config["diffusion"]["time_steps"]),
        seed=seed + 17_071,
    )
    generation_cfg = training_cfg.get("validation_generation", {})
    for epoch in range(start_epoch, epochs):
        batch_sampler.set_epoch(epoch)
        train_metrics = []
        for batch in loader:
            train_metrics.append(trainer.train_batch(batch))
        validation_metrics = _evaluate_noise(
            ema.ema_model, validation, high, memory, validation_bank
        )
        scheduler.step(validation_metrics["normalized_noise_loss"])
        record: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": sum(item["loss"] for item in train_metrics) / len(train_metrics),
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        if (
            bool(generation_cfg.get("enabled", True))
            and (epoch + 1) % int(generation_cfg.get("every_epochs", 10)) == 0
        ):
            count = min(len(validation), int(generation_cfg.get("max_snapshots", 1)))
            record["physical_generation"] = _evaluate_generation(
                ema.ema_model,
                validation[:count],
                high,
                memory,
                normalizer,
                config["evaluation"]["layer_weights"],
            )
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        improved = validation_metrics["normalized_noise_loss"] < best
        if improved:
            best = validation_metrics["normalized_noise_loss"]
        save_diffusion_checkpoint(
            output_dir / "last.pt",
            system,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=trainer.global_step,
            config=config,
            identity=identity,
            seed=seed,
            best_metric=best,
            history=history,
        )
        if improved:
            save_diffusion_checkpoint(
                best_path,
                system,
                ema=ema,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=trainer.global_step,
                config=config,
                identity=identity,
                seed=seed,
                best_metric=best,
                history=history,
            )
        (output_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
