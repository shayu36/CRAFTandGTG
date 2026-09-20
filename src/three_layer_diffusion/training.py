"""EMA, joint optimization, validation and strict Stage-4 checkpoints."""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from .model import DIFFUSION_CONTRACT_VERSION, ThreeLayerRAGDiffusionSystem


DIFFUSION_CHECKPOINT_VERSION = "three-layer-rag-diffusion-checkpoint-v3-weighted-stage2"


class ModelEMA(nn.Module):
    def __init__(self, model: nn.Module, decay: float = 0.995, update_every: int = 1):
        super().__init__()
        if not 0.0 <= decay < 1.0 or update_every <= 0:
            raise ValueError("EMA decay/update_every 非法")
        self.decay = float(decay)
        self.update_every = int(update_every)
        self.register_buffer("num_updates", torch.tensor(0, dtype=torch.long))
        self.ema_model = copy.deepcopy(model).eval()
        for parameter in self.ema_model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.num_updates.add_(1)
        if int(self.num_updates.item()) % self.update_every:
            return
        model_parameters = dict(model.named_parameters())
        for name, ema_parameter in self.ema_model.named_parameters():
            source = model_parameters[name].detach()
            ema_parameter.lerp_(source, 1.0 - self.decay)
        model_buffers = dict(model.named_buffers())
        for name, ema_buffer in self.ema_model.named_buffers():
            if name in model_buffers:
                ema_buffer.copy_(model_buffers[name].detach())
        rag = getattr(self.ema_model, "rag", None)
        clear_cache = getattr(rag, "clear_source_cache", None)
        if callable(clear_cache):
            clear_cache()


class ThreeLayerDiffusionTrainer:
    """Jointly optimize RAG query/key encoders and three Diffusion experts."""

    def __init__(
        self,
        system: ThreeLayerRAGDiffusionSystem,
        optimizer: torch.optim.Optimizer,
        *,
        memory: Any,
        high_features: Mapping[str, Mapping[str, torch.Tensor]],
        ema: ModelEMA | None = None,
        scheduler: Any = None,
        gradient_clip_norm: float | None = None,
    ):
        self.system = system
        self.optimizer = optimizer
        self.memory = memory
        self.high_features = high_features
        self.ema = ema
        self.scheduler = scheduler
        self.gradient_clip_norm = gradient_clip_norm
        self.global_step = 0

    def train_batch(self, snapshots: list[Any]) -> dict[str, float]:
        """Accumulate a homogeneous city bucket without cross-city padding."""

        self.system.train()
        self.optimizer.zero_grad(set_to_none=True)
        totals = []
        layer_values = {layer: [] for layer in ("region", "syntax", "road")}
        for snapshot in snapshots:
            result = self.system.training_loss(
                snapshot,
                high_features=self.high_features[snapshot.city_id],
                memory=self.memory,
            )
            totals.append(result["loss"])
            for layer in layer_values:
                layer_values[layer].append(result["layer_losses"][layer])
        loss = torch.stack(totals).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("严格模式: joint RAG+Diffusion loss 含 NaN/Inf")
        loss.backward()
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            for parameter in self.system.parameters():
                gradient = parameter.grad
                if gradient is None:
                    gradient = torch.zeros_like(parameter)
                dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
                gradient.div_(world_size)
                parameter.grad = gradient
        gradients = [
            parameter.grad for parameter in self.system.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise FloatingPointError("严格模式: joint RAG+Diffusion gradient 缺失或含 NaN/Inf")
        if self.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.system.parameters(), self.gradient_clip_norm)
        self.optimizer.step()
        if self.ema is not None:
            self.ema.update(self.system)
        self.global_step += 1
        metrics = {"loss": float(loss.detach())}
        metrics.update({
            f"{layer}_noise_loss": float(torch.stack(values).mean().detach())
            for layer, values in layer_values.items()
        })
        return metrics


def _require_checkpoint_identity(identity: Mapping[str, Any]) -> None:
    required = {
        "graphgps_checkpoint_fingerprint",
        "joint_graph_hashes",
        "static_feature_version",
        "spectral_feature_version",
        "lappe_version",
        "weighted_spectrum_hashes",
        "global_attention_scope",
        "rag_memory_version",
        "rag_memory_sha256",
        "dynamic_normalizer_fingerprint",
    }
    missing = sorted(required - set(identity))
    if missing:
        raise ValueError(f"严格模式: checkpoint identity 缺少 {missing}")


def save_diffusion_checkpoint(
    path: str | Path,
    system: ThreeLayerRAGDiffusionSystem,
    *,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    global_step: int,
    config: Mapping[str, Any],
    identity: Mapping[str, Any],
    seed: int,
    best_metric: float | None = None,
    history: list[Mapping[str, Any]] | None = None,
) -> Path:
    _require_checkpoint_identity(identity)
    payload = {
        "checkpoint_version": DIFFUSION_CHECKPOINT_VERSION,
        "diffusion_contract_version": DIFFUSION_CONTRACT_VERSION,
        "rag_model_state": system.rag.state_dict(),
        "diffusion_state": system.diffusion.state_dict(),
        "ema_state": ema.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "config": copy.deepcopy(dict(config)),
        "identity": copy.deepcopy(dict(identity)),
        "random_seed": int(seed),
        "best_metric": None if best_metric is None else float(best_metric),
        "history": copy.deepcopy(list(history or [])),
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    return destination


def load_diffusion_checkpoint(
    path: str | Path,
    system: ThreeLayerRAGDiffusionSystem,
    *,
    ema: ModelEMA | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    expected_identity: Mapping[str, Any] | None = None,
    use_ema_weights: bool = False,
    restore_rng: bool = True,
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("严格模式: Stage-4 checkpoint 必须是 mapping")
    if payload.get("checkpoint_version") != DIFFUSION_CHECKPOINT_VERSION:
        raise ValueError("严格模式: Stage-4 checkpoint version 不匹配")
    if payload.get("diffusion_contract_version") != DIFFUSION_CONTRACT_VERSION:
        raise ValueError("严格模式: Diffusion contract version 不匹配")
    actual_identity = dict(payload.get("identity", {}))
    _require_checkpoint_identity(actual_identity)
    if expected_identity is not None:
        for key, expected in expected_identity.items():
            if actual_identity.get(key) != expected:
                raise ValueError(f"严格模式: checkpoint identity 不匹配: {key}")
    system.rag.load_state_dict(payload["rag_model_state"])
    system.diffusion.load_state_dict(payload["diffusion_state"])
    if ema is not None:
        ema.load_state_dict(payload["ema_state"])
    if use_ema_weights:
        if ema is None:
            raise ValueError("use_ema_weights=true 需要提供 EMA 实例")
        system.load_state_dict(ema.ema_model.state_dict())
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    if restore_rng and isinstance(payload.get("rng_state"), Mapping):
        rng_state = payload["rng_state"]
        if isinstance(rng_state.get("torch"), torch.Tensor):
            torch.set_rng_state(rng_state["torch"])
        if torch.cuda.is_available() and isinstance(rng_state.get("cuda"), (list, tuple)):
            torch.cuda.set_rng_state_all(rng_state["cuda"])
        if rng_state.get("numpy") is not None:
            np.random.set_state(rng_state["numpy"])
        if rng_state.get("python") is not None:
            random.setstate(rng_state["python"])
    return dict(payload)
