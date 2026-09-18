"""三层 GraphGPS 的 source-only 监督训练、验证与测试。"""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from .data import GraphGPSCityData, RegionFlowTargets
from .model import ThreeLayerGraphGPSLapPE


CHECKPOINT_VERSION = "three-layer-joint-graphgps-spectral-checkpoint-v3"


def _checkpoint_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Fields which alter feature semantics even when parameter shapes match."""

    model_cfg = config.get("model", {})
    pos_cfg = config.get("posenc", {})
    frequency_cfg = config.get("frequency", {})
    data_cfg = config.get("data", {})
    attention_cfg = config.get("attention", {})
    return {
        "model_name": model_cfg.get("name"),
        "hidden_dim": int(model_cfg.get("hidden_dim", 128)),
        "output_dim": int(model_cfg.get("output_dim", 48)),
        "joint_num_eig": int(pos_cfg.get("joint_num_eig", 16)),
        "frequency_method": frequency_cfg.get("method"),
        "decomposition_position": frequency_cfg.get("decomposition_position"),
        "joint_low_modes": int(frequency_cfg.get("joint_low_modes", 16)),
        "frequency_output_version": frequency_cfg.get("output_version"),
        "global_attn": attention_cfg.get("global_attn", "linear"),
        "full_attention_max_nodes": int(attention_cfg.get("full_attention_max_nodes", 4096)),
        "static_feature_version": data_cfg.get("hierarchy_feature_version"),
        "seq_length": int(data_cfg.get("seq_length", 24)),
    }


def region_prediction_loss(prediction: torch.Tensor, target: RegionFlowTargets) -> torch.Tensor:
    if prediction.ndim != 2 or prediction.shape[1] != 48:
        raise ValueError("严格模式: prediction 必须为 [N,48]")
    target = target.to(prediction.device)
    if target.region_ids.numel() == 0:
        raise ValueError("严格模式: Region target 不能为空")
    if int(target.region_ids.min()) < 0 or int(target.region_ids.max()) >= prediction.shape[0]:
        raise ValueError("严格模式: Region target ID 越界")
    selected = prediction[target.region_ids]
    if selected.shape != target.values.shape:
        raise ValueError("严格模式: prediction/label shape 不一致")
    if not torch.isfinite(selected).all() or not torch.isfinite(target.values).all():
        raise ValueError("严格模式: prediction/label 含 NaN/Inf")
    return F.mse_loss(selected, target.values)


def _city_metrics(prediction: torch.Tensor, target: RegionFlowTargets) -> dict[str, float | int]:
    target = target.to(prediction.device)
    selected = prediction[target.region_ids]
    difference = selected - target.values
    return {
        "mse": float((difference.square()).mean().detach().cpu()),
        "mae": float(difference.abs().mean().detach().cpu()),
        "rmse": float(difference.square().mean().sqrt().detach().cpu()),
        "num_regions": int(target.region_ids.numel()),
        "num_observations": int(target.observation_count.sum().detach().cpu()),
    }


def shape_summary(data: GraphGPSCityData, output: Mapping[str, torch.Tensor]) -> dict[str, list[int]]:
    hierarchy, pe = data.hierarchy, data.posenc
    return {
        "num_joint_nodes": [int(hierarchy.num_roads + hierarchy.num_syntax + hierarchy.num_regions)],
        "joint_eigvals": list(pe.joint.eigvals.shape),
        "joint_eigvecs": list(pe.joint.eigvecs.shape),
        "H_joint": list(output["H_joint"].shape),
        "H_joint_low": list(output["H_joint_low"].shape),
        "H_joint_high": list(output["H_joint_high"].shape),
        "joint_low_coefficients": list(output["joint_low_coefficients"].shape),
        "road_x": list(hierarchy.road_x.shape),
        "road_node_range": list(pe.road_node_range),
        "H_road": list(output["H_road"].shape),
        "H_road_low": list(output["H_road_low"].shape),
        "H_road_high": list(output["H_road_high"].shape),
        "road_to_syntax_pool": list(output["pooled_road_to_syntax"].shape),
        "syntax_x": list(hierarchy.syntax_x.shape),
        "syntax_node_range": list(pe.syntax_node_range),
        "H_syntax": list(output["H_syntax"].shape),
        "H_syntax_low": list(output["H_syntax_low"].shape),
        "H_syntax_high": list(output["H_syntax_high"].shape),
        "syntax_to_region_pool": list(output["pooled_syntax_to_region"].shape),
        "region_x": list(hierarchy.region_x.shape),
        "region_node_range": list(pe.region_node_range),
        "H_region": list(output["H_region"].shape),
        "H_region_low": list(output["H_region_low"].shape),
        "H_region_high": list(output["H_region_high"].shape),
        "pred": list(output["pred"].shape),
    }


def frequency_diagnostics(
    data: GraphGPSCityData,
    output: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, float | int]]:
    """Return reconstruction and orthogonal-residual diagnostics per layer."""

    diagnostics: dict[str, dict[str, float | int]] = {}
    eigenpairs = data.posenc.joint
    mixed = output["H_joint"]
    low = output["H_joint_low"]
    high = output["H_joint_high"]
    coefficients = output["joint_low_coefficients"]
    eigenpairs = eigenpairs.to(mixed.device)
    valid_indices = torch.nonzero(eigenpairs.mask, as_tuple=False).flatten()
    values = eigenpairs.eigvals[0, valid_indices, 0]
    order = torch.argsort(values)
    count = int(coefficients.shape[0])
    selected = valid_indices[order[:count]]
    basis = eigenpairs.eigvecs[:, selected]
    denominator = torch.linalg.vector_norm(mixed).clamp_min(torch.finfo(mixed.dtype).eps)
    reconstruction_error = torch.linalg.vector_norm(mixed - low - high) / denominator
    residual_error = torch.linalg.vector_norm(basis.transpose(0, 1) @ high) / denominator
    diagnostics = {"joint": {
        "num_low_modes": count,
        "cutoff_eigenvalue": float(eigenpairs.eigvals[0, selected[-1], 0].detach().cpu()),
        "reconstruction_relative_error": float(reconstruction_error.detach().cpu()),
        "orthogonal_residual_relative_error": float(residual_error.detach().cpu()),
    }}
    for layer in ("road", "syntax", "region"):
        mixed = output[f"H_{layer}"]
        low = output[f"H_{layer}_low"]
        high = output[f"H_{layer}_high"]
        denominator = torch.linalg.vector_norm(mixed).clamp_min(torch.finfo(mixed.dtype).eps)
        reconstruction_error = torch.linalg.vector_norm(mixed - low - high) / denominator
        diagnostics[layer] = {
            "num_low_modes": count,
            "cutoff_eigenvalue": diagnostics["joint"]["cutoff_eigenvalue"],
            "reconstruction_relative_error": float(reconstruction_error.detach().cpu()),
            "orthogonal_residual_relative_error": diagnostics["joint"]["orthogonal_residual_relative_error"],
        }
    return diagnostics


@torch.no_grad()
def evaluate_split(
    model: ThreeLayerGraphGPSLapPE,
    city_data: Iterable[GraphGPSCityData],
    split: str,
) -> dict[str, Any]:
    model.eval()
    per_city = {}
    for data in city_data:
        if data.targets is None or split not in data.targets:
            raise ValueError(f"严格模式: {data.hierarchy.city_id} 缺少 {split} target")
        output = model(data.hierarchy, data.posenc)
        per_city[data.hierarchy.city_id] = _city_metrics(output["pred"], data.targets[split])
    if not per_city:
        raise ValueError("严格模式: evaluate_split 没有城市数据")
    return {
        "split": split,
        "city_macro_mse": float(np.mean([row["mse"] for row in per_city.values()])),
        "city_macro_mae": float(np.mean([row["mae"] for row in per_city.values()])),
        "city_macro_rmse": float(np.mean([row["rmse"] for row in per_city.values()])),
        "per_city": per_city,
    }


def save_checkpoint(
    path: str | Path,
    *,
    model: ThreeLayerGraphGPSLapPE,
    optimizer: torch.optim.Optimizer | None,
    config: Mapping[str, Any],
    epoch: int,
    best_valid_rmse: float,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": CHECKPOINT_VERSION,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "config": dict(config),
            "contract": _checkpoint_contract(config),
            "epoch": int(epoch),
            "best_valid_rmse": float(best_valid_rmse),
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    *,
    model: ThreeLayerGraphGPSLapPE,
    optimizer: torch.optim.Optimizer | None = None,
    expected_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    if state.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError("严格模式: 不是 three-layer-joint-graphgps-spectral-checkpoint-v3")
    if state.get("contract") != _checkpoint_contract(state.get("config", {})):
        raise ValueError("严格模式: checkpoint 内部 config/contract 不一致")
    if expected_config is not None and state.get("contract") != _checkpoint_contract(expected_config):
        raise ValueError("严格模式: checkpoint 与当前 Stage 2 配置不匹配")
    model.load_state_dict(state["model"], strict=True)
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    return state


def train_and_evaluate(
    config: Mapping[str, Any],
    city_data: list[GraphGPSCityData],
    *,
    output_dir: str | Path,
    device: str | torch.device,
    epochs_override: int | None = None,
) -> dict[str, Any]:
    if not city_data:
        raise ValueError("严格模式: 训练至少需要一个 source 城市")
    training_cfg = config.get("training", {})
    if not isinstance(training_cfg, Mapping):
        raise TypeError("严格模式: training 配置必须为 mapping")
    if int(training_cfg.get("batch_size", 1)) != 1:
        raise ValueError("严格模式: 第一版变长城市图训练仅支持 batch_size=1")
    seed = int(training_cfg.get("seed", 20260905))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    epochs = int(epochs_override if epochs_override is not None else training_cfg.get("epochs", 100))
    if epochs <= 0:
        raise ValueError("严格模式: epochs 必须为正")
    model = ThreeLayerGraphGPSLapPE(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("lr", 1e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-5)),
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path, last_path = output_dir / "best.pt", output_dir / "last.pt"
    history, best_valid = [], float("inf")
    first_shapes = None

    for epoch in range(epochs):
        model.train()
        order = list(range(len(city_data)))
        random.shuffle(order)
        train_losses = []
        for index in order:
            data = city_data[index]
            if data.targets is None:
                raise ValueError(f"严格模式: source {data.hierarchy.city_id} 缺少 targets")
            optimizer.zero_grad(set_to_none=True)
            output = model(data.hierarchy, data.posenc)
            loss = region_prediction_loss(output["pred"], data.targets["train"])
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f"严格模式: 参数 {name} 梯度含 NaN/Inf")
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
            if first_shapes is None:
                first_shapes = shape_summary(data, output)
                first_shapes["label"] = list(data.targets["train"].values.shape)
                print(json.dumps({"shape_log": first_shapes}, ensure_ascii=False))
        valid = evaluate_split(model, city_data, "valid")
        row = {
            "epoch": epoch,
            "train_city_macro_mse": float(np.mean(train_losses)),
            "valid_city_macro_rmse": valid["city_macro_rmse"],
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if valid["city_macro_rmse"] < best_valid:
            best_valid = float(valid["city_macro_rmse"])
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                config=config,
                epoch=epoch,
                best_valid_rmse=best_valid,
            )
    save_checkpoint(
        last_path,
        model=model,
        optimizer=optimizer,
        config=config,
        epoch=epochs - 1,
        best_valid_rmse=best_valid,
    )
    load_checkpoint(best_path, model=model, expected_config=config)
    metrics = {
        "checkpoint": str(best_path),
        "best_valid_rmse": best_valid,
        "shapes": first_shapes,
        "train": evaluate_split(model, city_data, "train"),
        "valid": evaluate_split(model, city_data, "valid"),
        "test": evaluate_split(model, city_data, "test"),
        "history": history,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metrics
