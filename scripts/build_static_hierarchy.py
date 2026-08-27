#!/usr/bin/env python3
"""构建/校验第一阶段三层城市静态图 cache。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "craft_integrated"))

from static_hierarchy.data import save_city_static_hierarchy  # noqa: E402
from static_hierarchy.preprocessing import build_city_static_hierarchy  # noqa: E402


def _path(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _city_root(config: dict, city: str, default_root: Path) -> Path:
    """支持 source GTG 输出目录与 target CRAFT 只读目录并存。"""
    override = (config.get("city_data_dirs") or {}).get(city)
    if override:
        city_dir = _path(str(override))
        if city_dir is None or not city_dir.exists():
            raise FileNotFoundError(f"严格模式: 缺失 city_data_dirs[{city}]={city_dir}")
        if city_dir.name != city:
            raise ValueError(f"严格模式: city_data_dirs[{city}] 末级目录必须为 {city}")
        return city_dir.parent
    return default_root


def _load_config(path: Path) -> dict:
    with path.open() as handle:
        config = yaml.safe_load(handle) or {}
    if config.get("static_structure_mode") != "three_layer":
        raise ValueError("静态阶段配置必须 static_structure_mode: three_layer")
    if config.get("road_feature_mode", "topology_only") == "cospec":
        raise NotImplementedError("CoSpec road features are not implemented in Stage 1")
    if config.get("road_feature_mode", "topology_only") != "topology_only":
        raise ValueError("当前仅支持 road_feature_mode: topology_only")
    return config


def _cities(config: dict, source_override: list[str] | None, target_override: str | None) -> tuple[list[str], str]:
    sources = source_override or config.get("source_cities")
    target = target_override if target_override is not None else config.get("target_city")
    if not sources or len(set(sources)) != len(sources):
        raise ValueError("必须显式提供非空且不重复的 source_cities")
    if not target:
        raise ValueError("target_city 必须由配置文件或 --target_city 显式提供")
    if target in sources:
        raise ValueError("target_city 不能同时出现在 source_cities")
    return list(sources), str(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/stage1_three_layer_static.yaml")
    parser.add_argument("--action", choices=["validate", "preprocess", "smoke", "pretrain"], required=True)
    parser.add_argument("--source_cities", nargs="+")
    parser.add_argument("--target_city")
    parser.add_argument("--device")
    args = parser.parse_args()
    config = _load_config(args.config.resolve())
    if args.device:
        config["device"] = args.device
    sources, target = _cities(config, args.source_cities, args.target_city)
    craft_root = _path(config.get("craft_data_root"))
    syntax_cache = _path(config.get("gtg_syntax_cache_dir"))
    hierarchy_cache = _path(config.get("static_hierarchy_cache_dir"))
    if craft_root is None or hierarchy_cache is None:
        raise ValueError("craft_data_root 和 static_hierarchy_cache_dir 必须配置")
    # 统一将相对配置解析到 Paper 根目录，保证从任意当前工作目录调用入口时
    # 仍使用同一份静态数据和 cache，不写入 CRAFT/GTG 原始目录。
    config["craft_data_root"] = str(craft_root)
    config["static_hierarchy_cache_dir"] = str(hierarchy_cache)
    if config.get("norm_flow_root") is not None:
        config["norm_flow_root"] = str(_path(config["norm_flow_root"]))
    if config.get("gtg_syntax_cache_dir") is not None:
        config["gtg_syntax_cache_dir"] = str(syntax_cache)
    if config.get("city_data_dirs"):
        config["city_data_dirs"] = {
            city: str(_path(path)) for city, path in config["city_data_dirs"].items()
        }
    cities = sources + [target]
    built = {}
    for city in cities:
        hierarchy = build_city_static_hierarchy(
            city, _city_root(config, city, craft_root), syntax_cache_dir=syntax_cache,
            local_size=int(config.get("metis_local_size", 50)),
            empty_region_error_ratio=float(config.get("empty_region_error_ratio", 0.2)),
        )
        built[city] = hierarchy
        if args.action == "preprocess":
            npz, meta = save_city_static_hierarchy(hierarchy, hierarchy_cache)
            print(f"{city}: {npz} {meta}")
    if args.action == "validate":
        print(json.dumps({
            "config": config,
            "source_cities": sources,
            "target_city": target,
            "cities": cities,
            "validated": {
                city: {
                    "num_regions": hierarchy.num_regions,
                    "num_roads": hierarchy.num_roads,
                    "num_syntax_nodes": hierarchy.num_syntax,
                }
                for city, hierarchy in built.items()
            },
        }, indent=2, ensure_ascii=False))
        return
    if args.action == "smoke":
        from static_hierarchy.model import ThreeLayerStaticEncoder
        import torch

        model = ThreeLayerStaticEncoder(config).to(config.get("device", "cpu"))
        model.train()
        outputs = {}
        for city, hierarchy in built.items():
            output = model(hierarchy)
            if output.shape != (hierarchy.num_regions, int(config.get("rep_dim", 128))):
                raise RuntimeError(f"{city}: region_rep shape 错误 {tuple(output.shape)}")
            if not torch.isfinite(output).all():
                raise FloatingPointError(f"{city}: region_rep 含 NaN/Inf")
            outputs[city] = list(output.shape)
        # 仅检查计算图连通性，不作为训练损失。
        built[sources[0]].to(config.get("device", "cpu"))
        check = model(built[sources[0]]).sum()
        check.backward()
        if not any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
            raise RuntimeError("三层编码器没有可用梯度")
        print(json.dumps({"outputs": outputs, "grad_check": True}, indent=2))
    elif args.action == "pretrain":
        # 预训练仅调用 CRAFT 原有 TFA/CCA，静态表征保存后立即结束。
        import torch
        import numpy as np
        import data_loaders
        from rep_model import GTAggregator

        data_loaders.configure(config)
        graph_dict = data_loaders.get_graph_datasets(sources, [target])
        model = GTAggregator(config).to(config.get("device", "cpu"))
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("lr", 5e-6)))
        epochs = int(config.get("pretrain_epoch", 1))
        if epochs <= 0:
            raise ValueError("pretrain_epoch 必须 > 0")
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = model.calc_loss(graph_dict)
            if not torch.isfinite(loss):
                raise FloatingPointError("CRAFT 原有 TFA/CCA loss 为 NaN/Inf")
            loss.backward()
            optimizer.step()
        output_dir = _path(config.get("static_rep_output_dir")) or ROOT / "outputs/stage1_static"
        output_dir.mkdir(parents=True, exist_ok=True)
        for graph in graph_dict["src_graphs"] + graph_dict["trg_graphs"]:
            rep = model.encode_graph(graph).detach().cpu().numpy()
            np.save(output_dir / f"{graph.city}_region_rep.npy", rep)
        torch.save({"model": model.state_dict(), "config": config, "source_cities": sources, "target_city": target}, output_dir / "static_encoder.pth")
        print(json.dumps({"epochs": epochs, "output_dir": str(output_dir), "cities": cities}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
