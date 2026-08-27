"""HCFM 宏观、微观、守恒、表征与效率指标。"""

from __future__ import annotations

from typing import Dict

import torch

from .hierarchy import aggregate_micro_to_macro, graph_difference


def _masked_values(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    expanded = mask.view(*mask.shape, *([1] * (prediction.ndim - 2))).expand_as(prediction)
    return prediction[expanded], target[expanded]


def cpc(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    prediction, target = prediction.clamp_min(0), target.clamp_min(0)
    return 2 * torch.minimum(prediction, target).sum() / (prediction.sum() + target.sum()).clamp_min(eps)


def flow_metrics(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, prefix: str
) -> Dict[str, float]:
    pred, true = _masked_values(prediction, target, mask)
    span = (true.max() - true.min()).clamp_min(1e-8)
    return {
        f"{prefix}_cpc": float(cpc(pred, true)),
        f"{prefix}_mae": float((pred - true).abs().mean()),
        f"{prefix}_rmse": float(((pred - true) ** 2).mean().sqrt()),
        f"{prefix}_min_max_mae": float((pred - true).abs().mean() / span),
        f"{prefix}_min_max_rmse": float(((pred - true) ** 2).mean().sqrt() / span),
    }


def topology_difference_error(
    prediction: torch.Tensor, target: torch.Tensor, edge_index: torch.Tensor, road_mask: torch.Tensor
) -> float:
    pred_diff, true_diff = graph_difference(prediction, edge_index), graph_difference(target, edge_index)
    src, dst = edge_index.to(road_mask.device)
    if road_mask.ndim == 1:
        road_mask = road_mask.unsqueeze(0)
    edge_mask = road_mask[:, src] & road_mask[:, dst]
    pred, true = _masked_values(pred_diff, true_diff, edge_mask)
    return float((pred - true).abs().mean())


def conservation_metrics(
    macro: torch.Tensor,
    micro: torch.Tensor,
    b_in: torch.Tensor,
    b_out: torch.Tensor,
    region_mask: torch.Tensor,
) -> Dict[str, object]:
    aggregated = aggregate_micro_to_macro(micro, b_in, b_out)
    if region_mask.ndim == 1:
        region_mask = region_mask.unsqueeze(0)
    difference = (macro - aggregated).abs()
    valid = region_mask[:, :, None, None].expand_as(difference)
    relative = difference / macro.abs().clamp_min(1e-6)
    per_region = difference.mean(dim=(0, 2, 3))
    return {
        "cross_scale_MAE": float(difference[valid].mean()),
        "cross_scale_relative_error": float(relative[valid].mean()),
        "per_region_conservation_gap": per_region.detach().cpu().tolist(),
        "inflow_consistency": float(difference[:, :, 0][region_mask].mean()),
        "outflow_consistency": float(difference[:, :, 1][region_mask].mean()),
    }


def representation_metrics(
    *,
    semantic: torch.Tensor,
    domain: torch.Tensor,
    semantic_domain_logits: torch.Tensor,
    domain_logits: torch.Tensor,
    city_label: int,
    cost_prediction: torch.Tensor | None = None,
    cost_target: torch.Tensor | None = None,
    cost_mask: torch.Tensor | None = None,
) -> Dict[str, float]:
    """GTG 表征的领域、cost、rank 与 semantic-domain 相似度指标。"""

    target = torch.full(
        (semantic.shape[0],), int(city_label), dtype=torch.long, device=semantic.device
    )
    result = {
        "semantic_domain_classification_accuracy": float(
            (semantic_domain_logits.argmax(dim=-1) == target).float().mean()
        ),
        "domain_classification_accuracy": float(
            (domain_logits.argmax(dim=-1) == target).float().mean()
        ),
        "semantic_domain_cosine_similarity": float(
            torch.nn.functional.cosine_similarity(semantic, domain, dim=-1).abs().mean()
        ),
    }
    if cost_prediction is not None or cost_target is not None:
        if cost_prediction is None or cost_target is None or cost_prediction.shape != cost_target.shape:
            raise ValueError("cost prediction/target 必须同时提供且同形")
        mask = torch.ones(cost_prediction.shape[0], dtype=torch.bool, device=cost_prediction.device)
        if cost_mask is not None:
            mask = cost_mask.to(cost_prediction.device).bool()
        prediction, truth = cost_prediction[mask], cost_target.to(cost_prediction)[mask]
        if not len(prediction):
            raise ValueError("cost metric mask 无有效标签")
        result["cost_prediction_mse"] = float(((prediction - truth) ** 2).mean())
        correct, count = prediction.new_zeros(()), 0
        for channel in range(prediction.shape[1]):
            true_diff = truth[1:, channel] - truth[:-1, channel]
            pred_diff = prediction[1:, channel] - prediction[:-1, channel]
            informative = true_diff != 0
            if informative.any():
                correct = correct + ((pred_diff[informative] * true_diff[informative]) > 0).float().sum()
                count += int(informative.sum())
        result["rank_accuracy"] = float(correct / count) if count else 1.0
    return result
