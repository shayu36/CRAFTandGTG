"""HCFM checkpoint 与第一阶段部分权重兼容加载。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import torch


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    config: Mapping[str, Any],
    normalizers: Mapping[str, Any],
    data_version: str,
    step: int,
) -> None:
    state = {
        "format_version": "hcfm-checkpoint-v1",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "config": dict(config),
        "normalizers": {name: normalizer.state_dict() for name, normalizer in normalizers.items()},
        "data_version": str(data_version),
        "step": int(step),
    }
    torch.save(state, Path(path))


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    normalizers: Mapping[str, Any] | None = None,
    expected_data_version: str | None = None,
) -> Dict[str, Any]:
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    if state.get("format_version") != "hcfm-checkpoint-v1":
        raise ValueError("不是 HCFM v1 checkpoint")
    if expected_data_version is not None and state["data_version"] != expected_data_version:
        raise ValueError(
            f"checkpoint data_version={state['data_version']!r} != {expected_data_version!r}"
        )
    missing, unexpected = model.load_state_dict(state["model"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"完整 checkpoint 加载不严格: missing={missing}, unexpected={unexpected}")
    if optimizer is not None and state["optimizer"] is not None:
        optimizer.load_state_dict(state["optimizer"])
    if normalizers is not None:
        if set(normalizers) != set(state["normalizers"]):
            raise ValueError("checkpoint normalizer 名称集合不一致")
        for name, normalizer in normalizers.items():
            normalizer.load_state_dict(
                state["normalizers"][name], expected={"data_version": state["data_version"]}
            )
    return {"config": state["config"], "step": state["step"], "data_version": state["data_version"]}


def load_stage1_gfa(
    model: torch.nn.Module, checkpoint_path: str | Path
) -> Dict[str, list[str]]:
    """只加载第一阶段地理编码器/GFA；GTG Region 分支和 Diffusion 参数均不误载。"""

    raw = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    if "ori_model" in raw or "ema_model" in raw:
        raise ValueError("GFA 应从 rep_model.pth 加载，而不是 craft.pth Diffusion checkpoint")
    target = model.state_dict()
    mapped, skipped = {}, []
    for key, value in raw.items():
        if key.startswith("init_proj.") or key.startswith("gnn."):
            candidate = f"macro_encoder.{key}"
            if candidate in target and target[candidate].shape == value.shape:
                mapped[candidate] = value
            else:
                skipped.append(key)
        else:
            skipped.append(key)
    result = model.load_state_dict(mapped, strict=False)
    return {
        "loaded": sorted(mapped), "skipped": sorted(skipped),
        "missing": sorted(result.missing_keys), "unexpected": sorted(result.unexpected_keys),
    }


def load_stage1_craft_conditions(
    model: torch.nn.Module, checkpoint_path: str | Path, use_ema: bool = True
) -> Dict[str, list[str]]:
    """为 Macro FM 加载第一阶段时间/Reference 条件，明确跳过 generator_model(Diffusion)。"""

    raw = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    source = raw["ema_model" if use_ema else "ori_model"]
    target = model.state_dict()
    prefixes = ("hour_embedding.", "weekday_embedding.", "month_embedding.", "temporal_encoder.")
    mapped, skipped = {}, []
    for key, value in source.items():
        if key.startswith("generator_model"):
            skipped.append(key)
            continue
        candidate = key
        if key.startswith("hour_embedding."):
            candidate = "time_condition." + key
        elif key.startswith("weekday_embedding."):
            candidate = "time_condition." + key
        elif key.startswith("month_embedding."):
            candidate = "time_condition." + key
        if key.startswith("temporal_encoder.") or key.startswith(prefixes[:3]):
            if candidate in target and target[candidate].shape == value.shape:
                mapped[candidate] = value
            else:
                skipped.append(key)
        else:
            skipped.append(key)
    result = model.load_state_dict(mapped, strict=False)
    return {
        "loaded": sorted(mapped), "skipped": sorted(skipped),
        "missing": sorted(result.missing_keys), "unexpected": sorted(result.unexpected_keys),
    }

