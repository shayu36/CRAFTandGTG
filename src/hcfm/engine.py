"""HCFM warm-up 阶段约束与单步训练引擎。"""

from __future__ import annotations

from typing import Any, Dict, Mapping

import torch

from .adversarial import assert_optimizer_covers
from .model import compose_total_loss


PHASE_ALLOWED = {
    "encoder_pretrain": {"cost", "rank", "semantic_domain", "domain", "orthogonal", "gfa"},
    "macro_fm": {"fm_macro", "gfa"},
    "macro_adversarial": {"fm_macro", "cost", "rank", "semantic_domain", "domain", "orthogonal", "gfa"},
    "joint_fm": {"fm_macro", "fm_micro", "topology", "cost", "rank", "semantic_domain", "domain", "orthogonal", "gfa"},
    "cross_state": {"fm_macro", "fm_micro", "cross_state", "topology", "cost", "rank", "semantic_domain", "domain", "orthogonal", "gfa"},
    "coupled_velocity": {"fm_macro", "fm_micro", "cross_state", "cross_velocity", "topology", "cost", "rank", "semantic_domain", "domain", "orthogonal", "gfa"},
}


def validate_training_phase(phase: str, weights: Mapping[str, float], prior_mode: str) -> None:
    if phase not in PHASE_ALLOWED:
        raise ValueError(f"未知 training phase={phase!r}")
    forbidden = [name for name, value in weights.items() if value > 0 and name not in PHASE_ALLOWED[phase]]
    if forbidden:
        raise ValueError(f"训练阶段 {phase} 提前启用了损失 {forbidden}")
    if weights.get("cross_velocity", 0) > 0 and (phase != "coupled_velocity" or prior_mode != "coupled"):
        raise ValueError("cross_velocity 只允许 coupled_velocity 阶段 + coupled prior")


class HCFMTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        normalizers: Mapping[str, Any],
        config: Mapping[str, Any],
    ):
        self.model, self.optimizer = model, optimizer
        self.macro_normalizer = normalizers["macro_normalizer"]
        self.micro_normalizer = normalizers.get("micro_count_normalizer")
        self.conservation_calibrator = normalizers.get("conservation_calibrator")
        self.static_normalizer = normalizers.get("static_feature_normalizer")
        self.time_normalizer = normalizers.get("micro_time_normalizer")
        self.speed_normalizer = normalizers.get("micro_speed_normalizer")
        self.weights = config["loss"]
        self.phase = config["training"]["phase"]
        validate_training_phase(self.phase, self.weights, model.prior_mode)
        assert_optimizer_covers(model, optimizer)

    def train_step(
        self,
        source_batch: Mapping[str, Any],
        target_static: Mapping[str, Any],
        *,
        source_city_label: int,
        target_city_label: int,
        source_cost_target: torch.Tensor | None = None,
        source_cost_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        if self.static_normalizer is not None:
            source_batch = dict(source_batch)
            target_static = dict(target_static)
            source_batch["road_x"] = self.static_normalizer.transform(source_batch["road_x"])
            target_static["road_x"] = self.static_normalizer.transform(target_static["road_x"])
        flow_losses, auxiliary = self.model.flow_losses(
            source_batch, self.macro_normalizer, self.micro_normalizer,
            cross_velocity_weight=float(self.weights.get("cross_velocity", 0.0)),
            conservation_calibrator=self.conservation_calibrator,
            generator=generator,
        )
        auxiliary_losses = self.model.auxiliary_losses(
            source_batch, auxiliary, target_static,
            source_city_label=source_city_label, target_city_label=target_city_label,
            source_cost_target=source_cost_target, source_cost_mask=source_cost_mask,
        )
        losses = {**flow_losses, **auxiliary_losses}
        total, groups = compose_total_loss(losses, self.weights)
        if not torch.isfinite(total):
            raise FloatingPointError("HCFM total loss 为 NaN/Inf")
        total.backward()
        self.optimizer.step()
        output = {name: float(value.detach()) for name, value in losses.items()}
        output.update({name: float(value.detach()) for name, value in groups.items()})
        output["total_loss"] = float(total.detach())
        return output
