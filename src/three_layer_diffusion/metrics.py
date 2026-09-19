"""Normalized and physical-space metrics for three dynamic layers."""

from __future__ import annotations

from typing import Mapping

import torch

from .data import CHANNEL_NAMES, DynamicNormalizer


def physical_layer_metrics(
    layer: str,
    prediction_normalized: torch.Tensor,
    target_normalized: torch.Tensor,
    normalizer: DynamicNormalizer,
    *,
    mask: torch.Tensor | None = None,
) -> dict[str, float]:
    if prediction_normalized.shape != target_normalized.shape:
        raise ValueError("physical metric prediction/target shape 不一致")
    prediction = normalizer.inverse(layer, prediction_normalized)
    target = normalizer.inverse(layer, target_normalized)
    if mask is None:
        mask = torch.ones_like(target)
    else:
        mask = torch.broadcast_to(mask.to(target.device, target.dtype), target.shape)
    output: dict[str, float] = {}
    error = prediction - target
    denominator = mask.sum().clamp_min(1.0)
    output["mae"] = float((error.abs() * mask).sum() / denominator)
    output["rmse"] = float(((error.square() * mask).sum() / denominator).sqrt())
    channel_axis = target.ndim - 2
    for index, name in enumerate(CHANNEL_NAMES[layer]):
        channel_error = error.select(channel_axis, index)
        channel_mask = mask.select(channel_axis, index)
        count = channel_mask.sum().clamp_min(1.0)
        output[f"{name}_mae"] = float((channel_error.abs() * channel_mask).sum() / count)
        output[f"{name}_rmse"] = float(
            ((channel_error.square() * channel_mask).sum() / count).sqrt()
        )
    return output


def three_layer_physical_metrics(
    generated: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    normalizer: DynamicNormalizer,
    *,
    masks: Mapping[str, torch.Tensor] | None = None,
    layer_weights: Mapping[str, float] | None = None,
) -> dict[str, object]:
    weights = dict(layer_weights or {"region": 1.0, "syntax": 1.0, "road": 1.0})
    metrics = {
        layer: physical_layer_metrics(
            layer,
            generated[layer],
            target[layer],
            normalizer,
            mask=None if masks is None else masks.get(layer),
        )
        for layer in ("region", "syntax", "road")
    }
    denominator = sum(float(weights[layer]) for layer in metrics)
    if denominator <= 0:
        raise ValueError("metric layer weights 总和必须为正")
    return {
        "layers": metrics,
        "weighted_mae": sum(weights[layer] * metrics[layer]["mae"] for layer in metrics) / denominator,
        "weighted_rmse": sum(weights[layer] * metrics[layer]["rmse"] for layer in metrics) / denominator,
    }
