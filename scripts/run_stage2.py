#!/usr/bin/env python3
"""HCFM 第二阶段统一入口。

示例：
  python scripts/run_stage2.py --config configs/stage2_hierarchical_fm.yaml --action validate
  python scripts/run_stage2.py --config configs/stage2_hierarchical_fm.yaml --action preprocess --cities chi

``smoke`` 只接受用户提供/预处理得到的真实 sample bundle，不生成占位数据。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcfm.adversarial import assert_optimizer_covers  # noqa: E402
from hcfm.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from hcfm.config import load_config  # noqa: E402
from hcfm.data import HierarchicalCityDataset, SourceOnlyNormalizer, collate_city_snapshots  # noqa: E402
from hcfm.engine import HCFMTrainer  # noqa: E402
from hcfm.preprocessing import build_city_static_graph, save_static_cache  # noqa: E402
from hcfm.runtime import build_model  # noqa: E402
from hcfm.metrics import conservation_metrics, flow_metrics, topology_difference_error  # noqa: E402


def _fit_normalizers(samples, source_cities, data_version, require_micro=True):
    macro_rows, micro_rows, static_rows, cost_rows, seen_static_cities = [], [], [], [], set()
    for sample in samples:
        if sample["split"] != "train" or sample["city_id"] not in source_cities:
            continue
        macro = sample["macro_flow"].permute(0, 2, 1)[sample["region_mask"]]
        macro_rows.append(macro)
        if sample["city_id"] not in seen_static_cities:
            static_rows.append(sample["road_x"])
            seen_static_cities.add(sample["city_id"])
        if require_micro:
            micro = sample["micro_flow"].permute(0, 2, 1)[sample["road_mask"]]
            micro_rows.append(micro)
        if "road_cost_target" in sample and "road_cost_mask" in sample:
            cost_rows.append(sample["road_cost_target"][sample["road_cost_mask"]])
    if not macro_rows or (require_micro and not micro_rows):
        raise ValueError("严格模式: bundle 没有源城市 train 动态数据，无法拟合 normalizer")
    fitted_cities = sorted(seen_static_cities)
    macro_normalizer = SourceOnlyNormalizer().fit(
        torch.cat(macro_rows), cities=fitted_cities, source_cities=source_cities,
        split="train", feature_order=["in_flow", "out_flow"], data_version=data_version,
    )
    static_dim = static_rows[0].shape[-1]
    result = {
        "macro_normalizer": macro_normalizer,
        "static_feature_normalizer": SourceOnlyNormalizer().fit(
            torch.cat(static_rows), cities=fitted_cities, source_cities=source_cities,
            split="train", feature_order=[f"road_feature_{i}" for i in range(static_dim)],
            data_version=data_version,
        ),
    }
    if require_micro:
        result["micro_count_normalizer"] = SourceOnlyNormalizer().fit(
            torch.cat(micro_rows), cities=fitted_cities, source_cities=source_cities,
            split="train", feature_order=["road_passage_count"], data_version=data_version,
        )
    if cost_rows:
        costs = torch.cat(cost_rows)
        result["micro_time_normalizer"] = SourceOnlyNormalizer().fit(
            costs[:, 0:1], cities=fitted_cities, source_cities=source_cities,
            split="train", feature_order=["travel_time_cost"], data_version=data_version,
        )
        result["micro_speed_normalizer"] = SourceOnlyNormalizer().fit(
            costs[:, 1:2], cities=fitted_cities, source_cities=source_cities,
            split="train", feature_order=["speed_cost"], data_version=data_version,
        )
    return result


def _load_bundle(path: Path):
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    required = {"source_train_samples", "target_static", "source_cities", "city_labels"}
    if required - set(bundle):
        raise KeyError(f"真实 sample bundle 缺字段 {sorted(required-set(bundle))}")
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--action", choices=["validate", "preprocess", "smoke", "train", "generate", "evaluate"], required=True)
    parser.add_argument("--cities", nargs="+", default=["chi", "dc", "toronto", "ny"])
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs" / "hcfm_smoke.pth")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "hcfm_generation.pt")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.action == "validate":
        print(json.dumps(config, indent=2, ensure_ascii=False))
        return
    if args.action == "preprocess":
        paths = config.get("paths", {})
        output = ROOT / paths.get("hierarchy_cache_dir", "cache/hcfm")
        for city in args.cities:
            tensors, manifest = build_city_static_graph(
                city, ROOT / paths.get("craft_data_root", "../CRAFT/cleared_data"),
                ROOT / paths.get("gtg_cache_dir", "cache/gtg"),
            )
            cache_paths = save_static_cache(tensors, manifest, output)
            print(f"{city}: {cache_paths[0]} {cache_paths[1]}")
        return
    if config["model_mode"] != "hierarchical_flow_matching":
        raise ValueError("当前 smoke/generate bundle 入口用于 hierarchical_flow_matching")
    if args.bundle is None:
        raise ValueError("smoke/generate 必须提供 --bundle；不会自动生成占位数据")
    bundle = _load_bundle(args.bundle)
    dataset = HierarchicalCityDataset(
        bundle["source_train_samples"], seq_length=int(config["model"]["seq_length"]),
        require_micro=bool(config.get("generate_micro", True)),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_city_snapshots)
    model = build_model(config).to(args.device)
    normalizers = _fit_normalizers(
        dataset.samples, set(bundle["source_cities"]), config["data"]["data_version"],
        require_micro=bool(config.get("generate_micro", True)),
    )
    if args.action in {"smoke", "train"}:
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["lr"]))
        assert_optimizer_covers(model, optimizer)
        trainer = HCFMTrainer(model, optimizer, normalizers, config)
        target_static = bundle["target_static"]
        for key, value in list(target_static.items()):
            if isinstance(value, torch.Tensor):
                target_static[key] = value.to(args.device)
        labels = bundle["city_labels"]
        epochs = 1 if args.action == "smoke" else int(config["training"].get("epochs", 300))
        step, log = 0, None
        for _epoch in range(epochs):
            for batch in loader:
                for key, value in list(batch.items()):
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(args.device)
                cost_target, cost_mask = batch.get("road_cost_target"), batch.get("road_cost_mask")
                needs_cost = float(config["loss"].get("cost", 0)) > 0 or float(config["loss"].get("rank", 0)) > 0
                if needs_cost and (cost_target is None or cost_mask is None):
                    raise ValueError("loss.cost/rank > 0 但真实 bundle 缺少源城市 road_cost_target/road_cost_mask")
                log = trainer.train_step(
                    batch, target_static,
                    source_city_label=int(labels[batch["city_id"]]),
                    target_city_label=int(labels[target_static["city_id"]]),
                    source_cost_target=cost_target[0] if cost_target is not None else None,
                    source_cost_mask=cost_mask[0] if cost_mask is not None else None,
                )
                step += 1
                if args.action == "smoke":
                    break
            if args.action == "smoke":
                break
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            args.checkpoint, model=model, optimizer=optimizer, config=config,
            normalizers=normalizers, data_version=config["data"]["data_version"], step=step,
        )
        load_checkpoint(
            args.checkpoint, model=model, optimizer=optimizer, normalizers=normalizers,
            expected_data_version=config["data"]["data_version"],
        )
        print(json.dumps({"steps": step, **log}, indent=2))
        return
    load_checkpoint(
        args.checkpoint, model=model, normalizers=normalizers,
        expected_data_version=config["data"]["data_version"],
    )
    batch = next(iter(loader))
    for key, value in list(batch.items()):
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(args.device)
    batch["reference"] = normalizers["macro_normalizer"].transform(
        batch["reference"].permute(0, 1, 3, 2)
    ).permute(0, 1, 3, 2)
    batch["road_x"] = normalizers["static_feature_normalizer"].transform(batch["road_x"])
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
    started = time.perf_counter()
    with torch.no_grad():
        macro, micro, stats = model.generate(
            batch, steps=int(config["flow_matching"]["steps"]),
            solver=str(config["flow_matching"]["solver"]),
        )
    latency = time.perf_counter() - started
    peak_memory = (
        int(torch.cuda.max_memory_allocated(args.device))
        if torch.cuda.is_available() and str(args.device).startswith("cuda") else 0
    )
    macro_physical = normalizers["macro_normalizer"].inverse(
        macro.permute(0, 1, 3, 2)
    ).permute(0, 1, 3, 2)
    micro_physical = (
        normalizers["micro_count_normalizer"].inverse(micro.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        if micro.numel() and "micro_count_normalizer" in normalizers else micro
    )
    efficiency = {
        **stats.__dict__, "generation_latency_seconds": latency,
        "samples_per_second": macro.shape[0] / max(latency, 1e-12),
        "peak_gpu_memory_bytes": peak_memory,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"macro": macro_physical.cpu(), "micro": micro_physical.cpu(), "efficiency": efficiency}, args.output)
    result = {"macro_shape": list(macro.shape), "micro_shape": list(micro.shape), **efficiency}
    if args.action == "evaluate":
        result.update(flow_metrics(macro_physical[:, :, 0:1], batch["macro_flow"][:, :, 0:1], batch["region_mask"], "macro_in"))
        result.update(flow_metrics(macro_physical[:, :, 1:2], batch["macro_flow"][:, :, 1:2], batch["region_mask"], "macro_out"))
        if config.get("generate_micro", True):
            result.update(flow_metrics(micro_physical, batch["micro_flow"], batch["road_mask"], "road"))
            result["topology_difference_error"] = topology_difference_error(
                micro_physical, batch["micro_flow"], batch["road_edge_index"], batch["road_mask"]
            )
            result.update(conservation_metrics(
                macro_physical, micro_physical, batch["b_in"], batch["b_out"], batch["region_mask"]
            ))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
