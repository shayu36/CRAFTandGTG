"""三层 GraphGPS 的 source-only 监督训练、验证与测试。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from .data import (
    GraphGPSCityData,
    RegionFlowTargets,
    build_joint_three_layer_graph,
    validate_joint_three_layer_graph,
)
from .model import ThreeLayerGraphGPSLapPE
from .spectral_lap_pe import LAPPE_VERSION, weighted_pe_graph_hash


CHECKPOINT_VERSION = "three-layer-joint-graphgps-weighted-spectral-checkpoint-v4"


def stage2_graph_identities(
    city_data: Iterable[GraphGPSCityData],
) -> dict[str, dict[str, Any]]:
    """Return the exact per-city graph/spectrum identity used by Stage 2."""

    identities: dict[str, dict[str, Any]] = {}
    for data in city_data:
        city_id = data.hierarchy.city_id
        if not isinstance(city_id, str) or not city_id or city_id in identities:
            raise ValueError("严格模式: Stage 2 graph identity 的 city_id 为空或重复")
        joint = data.joint_graph or build_joint_three_layer_graph(data.hierarchy)
        validate_joint_three_layer_graph(joint)
        spectrum_hash = weighted_pe_graph_hash(
            joint.edge_index_joint,
            joint.num_nodes,
            joint.edge_weight,
            joint.edge_type,
        )
        metadata = data.posenc.metadata
        if metadata.get("pe_version") != LAPPE_VERSION:
            raise ValueError(f"严格模式: {city_id} 不是当前 weighted LapPE version")
        if metadata.get("weighted_pe") is not True:
            raise ValueError(f"严格模式: {city_id} 未使用 weighted LapPE")
        if metadata.get("joint_graph_hash") != joint.joint_graph_hash:
            raise ValueError(f"严格模式: {city_id} LapPE/message graph identity 不一致")
        if metadata.get("weighted_spectrum_hash") != spectrum_hash:
            raise ValueError(f"严格模式: {city_id} weighted spectrum identity 不一致")
        identities[city_id] = {
            "city_id": city_id,
            "joint_graph_hash": joint.joint_graph_hash,
            "weighted_spectrum_hash": spectrum_hash,
            "num_joint_nodes": joint.num_nodes,
            "road_node_range": list(joint.road_node_range),
            "syntax_node_range": list(joint.syntax_node_range),
            "region_node_range": list(joint.region_node_range),
            "lappe_version": LAPPE_VERSION,
            "weighted_pe": True,
        }
    if not identities:
        raise ValueError("严格模式: Stage 2 checkpoint 至少需要一个训练图 identity")
    return {city: identities[city] for city in sorted(identities)}


def _graph_identities_fingerprint(identities: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        identities,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_serialized_graph_identities(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("严格模式: checkpoint 缺少 training_graph_identities")
    normalized: dict[str, dict[str, Any]] = {}
    for city_id, row in value.items():
        if not isinstance(city_id, str) or not city_id or not isinstance(row, Mapping):
            raise ValueError("严格模式: checkpoint training_graph_identities 格式非法")
        if row.get("city_id") != city_id:
            raise ValueError("严格模式: checkpoint graph identity city_id 不一致")
        for key in ("joint_graph_hash", "weighted_spectrum_hash"):
            digest = row.get(key)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest.lower())
            ):
                raise ValueError(f"严格模式: checkpoint {city_id}.{key} 不是 SHA-256")
        if row.get("lappe_version") != LAPPE_VERSION or row.get("weighted_pe") is not True:
            raise ValueError("严格模式: checkpoint graph identity 不是当前 weighted LapPE")
        num_nodes = row.get("num_joint_nodes")
        ranges = [
            row.get(name)
            for name in ("road_node_range", "syntax_node_range", "region_node_range")
        ]
        if not isinstance(num_nodes, int) or num_nodes <= 0:
            raise ValueError("严格模式: checkpoint num_joint_nodes 非法")
        if not all(isinstance(item, (list, tuple)) and len(item) == 2 for item in ranges):
            raise ValueError("严格模式: checkpoint 节点范围非法")
        canonical_ranges = [[int(item[0]), int(item[1])] for item in ranges]
        if (
            canonical_ranges[0][0] != 0
            or canonical_ranges[0][1] != canonical_ranges[1][0]
            or canonical_ranges[1][1] != canonical_ranges[2][0]
            or canonical_ranges[2][1] != num_nodes
            or any(start >= end for start, end in canonical_ranges)
        ):
            raise ValueError("严格模式: checkpoint 节点范围不连续")
        normalized[city_id] = {
            "city_id": city_id,
            "joint_graph_hash": row["joint_graph_hash"],
            "weighted_spectrum_hash": row["weighted_spectrum_hash"],
            "num_joint_nodes": num_nodes,
            "road_node_range": canonical_ranges[0],
            "syntax_node_range": canonical_ranges[1],
            "region_node_range": canonical_ranges[2],
            "lappe_version": LAPPE_VERSION,
            "weighted_pe": True,
        }
    return {city: normalized[city] for city in sorted(normalized)}


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
        "laplacian_norm": pos_cfg.get("laplacian_norm", "sym"),
        "lappe_version": LAPPE_VERSION,
        "weighted_pe": True,
        "frequency_method": frequency_cfg.get("method"),
        "decomposition_position": frequency_cfg.get("decomposition_position"),
        "joint_low_modes": int(frequency_cfg.get("joint_low_modes", 16)),
        "frequency_output_version": frequency_cfg.get("output_version"),
        "global_attn": attention_cfg.get("global_attn", "linear"),
        "global_attention_scope": attention_cfg.get("global_attention_scope", "joint"),
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
    training_graph_identities: Mapping[str, Any],
    epoch: int,
    best_valid_rmse: float,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    graph_identities = _validate_serialized_graph_identities(training_graph_identities)
    torch.save(
        {
            "format_version": CHECKPOINT_VERSION,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "config": dict(config),
            "contract": _checkpoint_contract(config),
            "training_graph_identities": graph_identities,
            "training_graph_identities_sha256": _graph_identities_fingerprint(graph_identities),
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
    expected_graph_identities: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    if state.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError(f"严格模式: 不是 {CHECKPOINT_VERSION}；旧无权 LapPE checkpoint 不得加载")
    if state.get("contract") != _checkpoint_contract(state.get("config", {})):
        raise ValueError("严格模式: checkpoint 内部 config/contract 不一致")
    if expected_config is not None and state.get("contract") != _checkpoint_contract(expected_config):
        raise ValueError("严格模式: checkpoint 与当前 Stage 2 配置不匹配")
    graph_identities = _validate_serialized_graph_identities(
        state.get("training_graph_identities")
    )
    if state.get("training_graph_identities_sha256") != _graph_identities_fingerprint(
        graph_identities
    ):
        raise ValueError("严格模式: checkpoint training graph identity 摘要不一致")
    if expected_graph_identities is not None:
        expected = _validate_serialized_graph_identities(expected_graph_identities)
        if graph_identities != expected:
            raise ValueError("严格模式: checkpoint 与当前训练城市 graph/spectrum identity 不匹配")
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
    graph_identities = stage2_graph_identities(city_data)
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1

    def sync_gradients() -> None:
        if not distributed:
            return
        # DDP cannot wrap this model directly because the training call uses
        # custom graph objects instead of Module.forward(). Average gradients
        # explicitly while keeping all ranks on the same optimizer step.
        for parameter in model.parameters():
            gradient = parameter.grad
            if gradient is None:
                gradient = torch.zeros_like(parameter)
            dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
            gradient.div_(world_size)
            parameter.grad = gradient

    for epoch in range(epochs):
        model.train()
        order = list(range(len(city_data)))
        random.shuffle(order)
        train_losses = []
        if distributed:
            # Every rank performs the same number of optimizer steps. The
            # cyclic schedule keeps all five GPUs active even though the
            # source-only Stage 2 set contains three cities.
            local_order = [
                order[(step * world_size + rank) % len(order)]
                for step in range(len(order))
            ]
        else:
            local_order = order
        for index in local_order:
            data = city_data[index]
            if data.targets is None:
                raise ValueError(f"严格模式: source {data.hierarchy.city_id} 缺少 targets")
            optimizer.zero_grad(set_to_none=True)
            output = model(data.hierarchy, data.posenc)
            loss = region_prediction_loss(output["pred"], data.targets["train"])
            loss.backward()
            sync_gradients()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f"严格模式: 参数 {name} 梯度含 NaN/Inf")
            optimizer.step()
            metric_loss = loss.detach()
            if distributed:
                dist.all_reduce(metric_loss, op=dist.ReduceOp.SUM)
                metric_loss.div_(world_size)
            train_losses.append(float(metric_loss.cpu()))
            if rank == 0 and first_shapes is None:
                first_shapes = shape_summary(data, output)
                first_shapes["label"] = list(data.targets["train"].values.shape)
                print(json.dumps({"shape_log": first_shapes}, ensure_ascii=False))
        if distributed:
            dist.barrier()
        valid = evaluate_split(model, city_data, "valid") if rank == 0 else None
        if distributed:
            dist.barrier()
        if rank != 0:
            continue
        assert valid is not None
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
                training_graph_identities=graph_identities,
                epoch=epoch,
                best_valid_rmse=best_valid,
            )
    save_checkpoint(
        last_path,
        model=model,
        optimizer=optimizer,
        config=config,
        training_graph_identities=graph_identities,
        epoch=epochs - 1,
        best_valid_rmse=best_valid,
    ) if rank == 0 else None
    if distributed:
        dist.barrier()
    if rank != 0:
        return {"rank": rank, "world_size": world_size}
    load_checkpoint(
        best_path,
        model=model,
        expected_config=config,
        expected_graph_identities=graph_identities,
    )
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
