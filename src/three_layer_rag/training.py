"""Joint-training and checkpoint utilities for the three-layer RAG.

RAG has no independent future-flow label objective here.  The downstream
Flow-Matching objective is supplied as ``objective(output)`` so retrieval
parameters are trained without inventing a Region-only surrogate target.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from .contracts import RAG_CONTRACT_VERSION


RAG_CHECKPOINT_VERSION = "three-layer-rag-checkpoint-v1"


class RAGTrainer:
    """A small optimizer wrapper intended for joint FM/RAG training."""

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
        self.model = model
        self.optimizer = optimizer

    def train_step(
        self,
        inputs: Any,
        *,
        objective: Callable[[Mapping[str, Any]], torch.Tensor],
        memory: Any = None,
        target_city: str | None = None,
    ) -> tuple[Mapping[str, Any], dict[str, float]]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        output = self.model(inputs, memory=memory, target_city=target_city)
        loss = objective(output)
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
            raise ValueError("严格模式: RAG joint objective 必须返回标量 Tensor")
        if not torch.isfinite(loss):
            raise FloatingPointError("严格模式: RAG joint loss 为 NaN/Inf")
        loss.backward()
        gradients = [parameter.grad for parameter in self.model.parameters() if parameter.grad is not None]
        if gradients and not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise FloatingPointError("严格模式: RAG gradient 含 NaN/Inf")
        self.optimizer.step()
        return output, {"rag_loss": float(loss.detach())}


def save_rag_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    step: int = 0,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save model/optimizer and source-memory identity without serializing data IDs."""

    memory = getattr(model, "memory", None)
    if memory is not None:
        memory.validate()
    payload = {
        "checkpoint_version": RAG_CHECKPOINT_VERSION,
        "rag_contract_version": RAG_CONTRACT_VERSION,
        "step": int(step),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "memory_version": getattr(memory, "version", None),
        "memory_graph_identity": dict(getattr(memory, "graph_identity", {}) or {}),
        "metadata": dict(metadata or {}),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    return destination


def load_rag_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    expected_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("严格模式: RAG checkpoint 必须为 mapping")
    if payload.get("checkpoint_version") != RAG_CHECKPOINT_VERSION:
        raise ValueError("严格模式: RAG checkpoint 版本不匹配")
    if payload.get("rag_contract_version") != RAG_CONTRACT_VERSION:
        raise ValueError("严格模式: RAG contract version 不匹配")
    if expected_metadata:
        actual = payload.get("metadata", {})
        for key, value in expected_metadata.items():
            if actual.get(key) != value:
                raise ValueError(f"严格模式: RAG checkpoint metadata 不匹配: {key}")
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return dict(payload)

