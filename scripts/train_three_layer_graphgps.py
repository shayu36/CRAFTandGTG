#!/usr/bin/env python3
"""第二阶段三层 GraphGPS + LapPE 独立入口。

不会调用 HCFM、Flow Matching、Diffusion、RAG 或生成代码。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from three_layer_graphgps.data import GraphGPSCityData, prepare_city_data  # noqa: E402
from three_layer_graphgps.engine import (  # noqa: E402
    evaluate_split,
    load_checkpoint,
    shape_summary,
    train_and_evaluate,
)
from three_layer_graphgps.model import ThreeLayerGraphGPSLapPE  # noqa: E402


def _load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("严格模式: Stage 2 GraphGPS config 必须为 mapping")
    for section in ("model", "posenc", "attention", "hierarchy", "data", "training"):
        if not isinstance(config.get(section), dict):
            raise KeyError(f"严格模式: Stage 2 GraphGPS config 缺少 mapping {section}")
    if config["model"].get("name") != "three_layer_graphgps_lappe":
        raise ValueError("严格模式: model.name 必须为 three_layer_graphgps_lappe")
    if config["data"].get("hierarchy_feature_version") != "three-layer-start-road-v2":
        raise ValueError("严格模式: 第二阶段只接受 three-layer-start-road-v2 cache")
    return config


def _absolute(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _prepare(
    config: dict,
    cities: list[str],
    *,
    require_targets: bool,
) -> list[GraphGPSCityData]:
    data_cfg, pos_cfg = config["data"], config["posenc"]
    ratios = tuple(float(value) for value in data_cfg.get("split_ratios", [0.8, 0.1, 0.1]))
    return [
        prepare_city_data(
            city=city,
            hierarchy_cache_dir=_absolute(data_cfg["hierarchy_cache_dir"]),
            lappe_cache_dir=(
                _absolute(pos_cfg["cache_dir"]) if bool(pos_cfg.get("cache", True)) else None
            ),
            road_k=int(pos_cfg.get("road_num_eig", 16)),
            syntax_k=int(pos_cfg.get("syntax_num_eig", 16)),
            region_k=int(pos_cfg.get("region_num_eig", 16)),
            normalization=str(pos_cfg.get("laplacian_norm", "sym")),
            require_targets=require_targets,
            norm_flow_root=_absolute(data_cfg["norm_flow_root"]) if require_targets else None,
            split_ratios=ratios,
        )
        for city in cities
    ]


def _static_summary(data: GraphGPSCityData) -> dict:
    hierarchy, pe = data.hierarchy, data.posenc
    return {
        "city": hierarchy.city_id,
        "road_x": list(hierarchy.road_x.shape),
        "road_edge_index_msg": list(hierarchy.road_edge_index.shape),
        "road_edge_index_pe": list(pe.road.edge_index_pe.shape),
        "road_eigvals": list(pe.road.eigvals.shape),
        "road_eigvecs": list(pe.road.eigvecs.shape),
        "syntax_x": list(hierarchy.syntax_x.shape),
        "syntax_eigvals": list(pe.syntax.eigvals.shape),
        "syntax_eigvecs": list(pe.syntax.eigvecs.shape),
        "region_x": list(hierarchy.region_x.shape),
        "region_eigvals": list(pe.region.eigvals.shape),
        "region_eigvecs": list(pe.region.eigvecs.shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "stage2_three_layer_graphgps_lappe.yaml",
    )
    parser.add_argument(
        "--action", choices=["precompute", "validate", "smoke", "train", "evaluate"], required=True
    )
    parser.add_argument("--source_cities", nargs="+")
    parser.add_argument("--target_city")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device")
    args = parser.parse_args()

    config = _load_config(args.config)
    data_cfg, train_cfg = config["data"], config["training"]
    source_cities = args.source_cities or list(data_cfg.get("source_cities", []))
    if not source_cities or any(not city for city in source_cities):
        raise ValueError("严格模式: source_cities 不能为空")
    target_city = args.target_city or data_cfg.get("target_city")
    device = args.device or train_cfg.get("device", "cpu")
    config["posenc"]["cache_dir"] = str(_absolute(config["posenc"]["cache_dir"]))

    if args.action in {"precompute", "validate"}:
        all_cities = source_cities + ([target_city] if target_city else [])
        prepared = _prepare(config, all_cities, require_targets=False)
        summaries = [_static_summary(data) for data in prepared]
        if args.action == "validate":
            model = ThreeLayerGraphGPSLapPE(config).to(device).eval()
            with torch.no_grad():
                for data, summary in zip(prepared, summaries):
                    output = model(data.hierarchy, data.posenc, return_edge_audit=True)
                    summary.update(shape_summary(data, output))
                    if not torch.equal(
                        output["road_edge_index_msg"].cpu(), data.hierarchy.road_edge_index.cpu()
                    ):
                        raise RuntimeError("严格模式: Road message edge 被 LapPE 覆盖")
        print(json.dumps({"action": args.action, "cities": summaries}, indent=2, ensure_ascii=False))
        return

    source_data = _prepare(config, source_cities, require_targets=True)
    output_dir = _absolute(train_cfg.get("output_dir", "outputs/stage2_three_layer_graphgps_lappe"))
    if args.action in {"smoke", "train"}:
        result = train_and_evaluate(
            config,
            source_data,
            output_dir=output_dir,
            device=device,
            epochs_override=1 if args.action == "smoke" else None,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    checkpoint = args.checkpoint or output_dir / "best.pt"
    model = ThreeLayerGraphGPSLapPE(config).to(device)
    load_checkpoint(checkpoint, model=model)
    result = {
        "checkpoint": str(checkpoint),
        "valid": evaluate_split(model, source_data, "valid"),
        "test": evaluate_split(model, source_data, "test"),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
