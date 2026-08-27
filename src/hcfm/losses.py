"""HCFM 宏观、微观、跨尺度损失。"""

from __future__ import annotations

from typing import Callable, Dict

import torch
import torch.nn.functional as F

from .flow_matching import estimate_endpoint, masked_mean, masked_mse, validate_velocity_consistency
from .hierarchy import aggregate_micro_to_macro, graph_difference


def _inverse_channels(
    values: torch.Tensor, inverse: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    """把通道移到末维调用 normalizer.inverse，再移回 ``[B,V,C,T]``。"""

    reordered = values.permute(0, 1, 3, 2)
    restored = inverse(reordered)
    if restored.shape != reordered.shape:
        raise ValueError("normalizer.inverse 改变了 tensor shape")
    return restored.permute(0, 1, 3, 2)


def cross_state_loss(
    macro_endpoint: torch.Tensor,
    micro_endpoint: torch.Tensor,
    macro_true: torch.Tensor,
    b_in: torch.Tensor,
    b_out: torch.Tensor,
    region_mask: torch.Tensor,
    road_mask: torch.Tensor,
    macro_inverse: Callable[[torch.Tensor], torch.Tensor],
    micro_inverse: Callable[[torch.Tensor], torch.Tensor],
    micro_to_macro_calibration: Callable[[torch.Tensor], torch.Tensor] | None = None,
    *,
    huber_delta: float = 1.0,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """在反归一化物理 passage-count 单位中计算鲁棒状态一致性。

    返回 ``(loss, macro_pred_physical, macro_from_micro_physical)``。
    """

    macro_physical = _inverse_channels(macro_endpoint, macro_inverse)
    macro_true_physical = _inverse_channels(macro_true, macro_inverse)
    micro_physical = _inverse_channels(micro_endpoint, micro_inverse)
    # 无效 Road 不得通过 B 泄漏到一致性损失。
    road_mask4 = road_mask if road_mask.ndim == 2 else road_mask.unsqueeze(0)
    micro_physical = micro_physical * road_mask4[:, :, None, None]
    macro_from_micro = aggregate_micro_to_macro(micro_physical, b_in, b_out)
    if micro_to_macro_calibration is not None:
        macro_from_micro = micro_to_macro_calibration(macro_from_micro)
    scale = macro_true_physical.abs().mean(dim=-1, keepdim=True).clamp_min(eps)
    normalized_difference = (macro_physical - macro_from_micro) / scale
    element = F.huber_loss(
        normalized_difference, torch.zeros_like(normalized_difference),
        delta=huber_delta, reduction="none",
    )
    return masked_mean(element, region_mask), macro_physical, macro_from_micro


def cross_velocity_loss(
    macro_velocity: torch.Tensor,
    micro_velocity: torch.Tensor,
    b_in: torch.Tensor,
    b_out: torch.Tensor,
    region_mask: torch.Tensor,
    *,
    prior_mode: str,
    weight: float,
) -> torch.Tensor:
    validate_velocity_consistency(prior_mode, weight)
    if weight == 0:
        return macro_velocity.sum() * 0.0
    aggregated = aggregate_micro_to_macro(micro_velocity, b_in, b_out)
    return masked_mse(macro_velocity, aggregated, region_mask)


def topology_difference_loss(
    micro_endpoint: torch.Tensor,
    micro_true: torch.Tensor,
    road_edge_index: torch.Tensor,
    road_mask: torch.Tensor,
) -> torch.Tensor:
    """保持真实道路图局部差分，不强迫相邻道路流量相等。"""

    prediction = graph_difference(micro_endpoint, road_edge_index)
    target = graph_difference(micro_true, road_edge_index)
    src, dst = road_edge_index.to(road_mask.device)
    if road_mask.ndim == 1:
        road_mask = road_mask.unsqueeze(0)
    edge_mask = road_mask[:, src] & road_mask[:, dst]
    return masked_mean((prediction - target).abs(), edge_mask)


def generation_losses(
    *,
    macro_velocity: torch.Tensor,
    macro_target_velocity: torch.Tensor,
    micro_velocity: torch.Tensor,
    micro_target_velocity: torch.Tensor,
    macro_state: torch.Tensor,
    micro_state: torch.Tensor,
    macro_true: torch.Tensor,
    micro_true: torch.Tensor,
    time: torch.Tensor,
    region_mask: torch.Tensor,
    road_mask: torch.Tensor,
    b_in: torch.Tensor,
    b_out: torch.Tensor,
    road_edge_index: torch.Tensor,
    macro_inverse: Callable[[torch.Tensor], torch.Tensor],
    micro_inverse: Callable[[torch.Tensor], torch.Tensor],
    prior_mode: str,
    cross_velocity_weight: float,
    micro_to_macro_calibration: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> Dict[str, torch.Tensor]:
    macro_endpoint = estimate_endpoint(macro_state, macro_velocity, time)
    micro_endpoint = estimate_endpoint(micro_state, micro_velocity, time)
    state, _, _ = cross_state_loss(
        macro_endpoint, micro_endpoint, macro_true, b_in, b_out,
        region_mask, road_mask, macro_inverse, micro_inverse,
        micro_to_macro_calibration,
    )
    velocity = cross_velocity_loss(
        macro_velocity, micro_velocity, b_in, b_out, region_mask,
        prior_mode=prior_mode, weight=cross_velocity_weight,
    )
    return {
        "fm_macro": masked_mse(macro_velocity, macro_target_velocity, region_mask),
        "fm_micro": masked_mse(micro_velocity, micro_target_velocity, road_mask),
        "cross_state": state,
        "cross_velocity": velocity,
        "topology": topology_difference_loss(
            micro_endpoint, micro_true, road_edge_index, road_mask
        ),
    }
