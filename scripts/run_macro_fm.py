#!/usr/bin/env python3
"""A2 Macro Flow Matching 独立训练/生成/评价入口；只读取真实 macro bundle。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcfm.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from hcfm.config import load_config  # noqa: E402
from hcfm.data import SourceOnlyNormalizer  # noqa: E402
from hcfm.metrics import flow_metrics  # noqa: E402
from hcfm.runtime import build_model  # noqa: E402


def add_batch(sample, device):
    required = {
        "city_id", "split", "aligned_region_rep", "reference", "macro_flow",
        "region_mask", "region_edge_index", "start_hour", "weekday", "month",
    }
    if required - set(sample):
        raise KeyError(f"macro sample 缺字段 {sorted(required-set(sample))}")
    result = dict(sample)
    for key in ("aligned_region_rep", "reference", "macro_flow", "region_mask"):
        result[key] = sample[key].unsqueeze(0).to(device)
    result["region_edge_index"] = sample["region_edge_index"].to(device)
    return result


def fit_normalizer(samples, source_cities, data_version):
    values = []
    for sample in samples:
        if sample["split"] != "train" or sample["city_id"] not in source_cities:
            raise ValueError("Macro FM normalizer bundle 含非源城市 train 数据")
        values.append(sample["macro_flow"].permute(0, 2, 1)[sample["region_mask"]])
    return SourceOnlyNormalizer().fit(
        torch.cat(values), cities=source_cities, source_cities=source_cities, split="train",
        feature_order=["in_flow", "out_flow"], data_version=data_version,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage2_macro_fm.yaml")
    parser.add_argument("--action", choices=["smoke", "train", "generate", "evaluate"], required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/macro_fm.pth")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/macro_fm_generation.pt")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = load_config(args.config)
    if config["model_mode"] != "macro_flow_matching":
        raise ValueError("run_macro_fm.py 只接受 macro_flow_matching 配置")
    bundle = torch.load(args.bundle, map_location="cpu", weights_only=False)
    required = {"source_train_samples", "test_samples", "source_cities"}
    if required - set(bundle):
        raise KeyError(f"macro bundle 缺字段 {sorted(required-set(bundle))}")
    version = bundle.get("data_version", "macro-fm-v1")
    normalizer = fit_normalizer(bundle["source_train_samples"], set(bundle["source_cities"]), version)
    model = build_model(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["lr"]))
    normalizers = {"macro_normalizer": normalizer}
    if args.action in {"smoke", "train"}:
        epochs = 1 if args.action == "smoke" else int(config["training"].get("epochs", 300))
        step, last = 0, None
        for _ in range(epochs):
            for sample in bundle["source_train_samples"]:
                batch = add_batch(sample, args.device)
                batch["reference"] = normalizer.transform(
                    batch["reference"].permute(0, 1, 3, 2)
                ).permute(0, 1, 3, 2)
                optimizer.zero_grad(set_to_none=True)
                loss = model.loss(batch, normalizer)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Macro FM loss 为 NaN/Inf")
                loss.backward(); optimizer.step()
                last, step = float(loss.detach()), step + 1
                if args.action == "smoke":
                    break
            if args.action == "smoke":
                break
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            args.checkpoint, model=model, optimizer=optimizer, config=config,
            normalizers=normalizers, data_version=version, step=step,
        )
        print(json.dumps({"steps": step, "fm_macro": last}, indent=2)); return
    load_checkpoint(
        args.checkpoint, model=model, normalizers=normalizers, expected_data_version=version
    )
    sample = bundle["test_samples"][0]
    batch = add_batch(sample, args.device)
    # Reference 是宏观流量条件，进入训练时同一归一化空间。
    batch["reference"] = normalizer.transform(batch["reference"].permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
    started = time.perf_counter()
    with torch.no_grad():
        generated, stats = model.generate(
            batch, steps=int(config["flow_matching"]["steps"]),
            solver=str(config["flow_matching"]["solver"]),
        )
    latency = time.perf_counter() - started
    physical = normalizer.inverse(generated.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
    result = {**stats.__dict__, "generation_latency_seconds": latency}
    if args.action == "evaluate":
        result.update(flow_metrics(physical[:, :, 0:1], batch["macro_flow"][:, :, 0:1], batch["region_mask"], "macro_in"))
        result.update(flow_metrics(physical[:, :, 1:2], batch["macro_flow"][:, :, 1:2], batch["region_mask"], "macro_out"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"macro": physical.cpu(), "stats": result}, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
