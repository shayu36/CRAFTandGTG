"""Region -> Syntax -> Road hierarchical conditional diffusion."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from three_layer_rag import ThreeLayerRAGInputs, ThreeLayerRAGMemory

from .conditioning import (
    LayerDiffusionConditioner,
    broadcast_parent_to_children,
    flatten_node_conditions,
    flatten_nodes,
    unflatten_nodes,
)
from .gaussian_diffusion import ConditionalGaussianDiffusion1D
from .unet import ConditionalUnet1D


LAYER_ORDER = ("region", "syntax", "road")
LAYER_CHANNELS = {"region": 2, "syntax": 3, "road": 3}
DIFFUSION_CONTRACT_VERSION = "three-layer-hierarchical-conditional-diffusion-v1"


def _ensure_batched(value: torch.Tensor, rank_without_batch: int, name: str) -> torch.Tensor:
    if value.ndim == rank_without_batch:
        value = value.unsqueeze(0)
    if value.ndim != rank_without_batch + 1:
        raise ValueError(f"{name} rank 非法")
    if not torch.isfinite(value).all():
        raise ValueError(f"严格模式: {name} 含 NaN/Inf")
    return value


class HierarchicalThreeLayerDiffusion(nn.Module):
    """One hierarchical generator containing three channel-specific experts."""

    def __init__(
        self,
        *,
        high_dim: int = 128,
        seq_length: int = 24,
        cond_dim: int = 256,
        temporal_dim: int = 64,
        parent_temporal_dim: int = 64,
        calendar_dims: Mapping[str, int] | None = None,
        time_steps: int = 500,
        sampling_time_steps: int = 500,
        beta_schedule: str = "linear",
        ddim_sampling_eta: float = 0.0,
        use_self_cond: bool = True,
        clip_x0: bool = False,
        sampling_x0_clip: float | None = None,
        self_condition_probability: float = 0.5,
        init_dim: int = 64,
        base_dim: int = 32,
        dim_mults: tuple[int, ...] = (1, 2, 4, 8),
        sinusoidal_theta: float = 10_000.0,
        dropout: float = 0.1,
        attention_dim_head: int = 64,
        attention_heads: int = 4,
        temporal_layers: int = 1,
        temporal_dropout: float = 0.0,
        layer_loss_weights: Mapping[str, float] | None = None,
        parent_condition_dropout: float = 0.0,
        node_chunk_size: int | None = None,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        calendar_dims = calendar_dims or {
            "month": 8, "weekday": 8, "start_hour": 8, "holiday": 4
        }
        weights = dict(layer_loss_weights or {layer: 1.0 for layer in LAYER_ORDER})
        if set(weights) != set(LAYER_ORDER) or any(float(value) < 0 for value in weights.values()):
            raise ValueError("layer_loss_weights 必须包含三层有限非负权重")
        if not 0.0 <= parent_condition_dropout <= 1.0:
            raise ValueError("parent_condition_dropout 必须在 [0,1]")
        self.high_dim = int(high_dim)
        self.seq_length = int(seq_length)
        self.cond_dim = int(cond_dim)
        self.layer_loss_weights = {key: float(value) for key, value in weights.items()}
        self.parent_condition_dropout = float(parent_condition_dropout)
        if node_chunk_size is not None and int(node_chunk_size) <= 0:
            raise ValueError("node_chunk_size 必须为正或为 null")
        self.node_chunk_size = None if node_chunk_size is None else int(node_chunk_size)

        parent_channels = {"region": None, "syntax": 2, "road": 3}
        self.conditioners = nn.ModuleDict({
            layer: LayerDiffusionConditioner(
                high_dim=high_dim,
                channels=LAYER_CHANNELS[layer],
                cond_dim=cond_dim,
                temporal_dim=temporal_dim,
                calendar_dims=calendar_dims,
                parent_channels=parent_channels[layer],
                parent_temporal_dim=parent_temporal_dim,
                temporal_layers=temporal_layers,
                temporal_dropout=temporal_dropout,
            )
            for layer in LAYER_ORDER
        })
        experts = {}
        for layer in LAYER_ORDER:
            estimator = ConditionalUnet1D(
                data_channels=LAYER_CHANNELS[layer],
                cond_dim=cond_dim,
                init_dim=init_dim,
                base_dim=base_dim,
                dim_mults=tuple(dim_mults),
                use_self_cond=use_self_cond,
                sinusoidal_theta=sinusoidal_theta,
                dropout=dropout,
                attention_dim_head=attention_dim_head,
                attention_heads=attention_heads,
            )
            experts[layer] = ConditionalGaussianDiffusion1D(
                estimator,
                data_channels=LAYER_CHANNELS[layer],
                seq_length=seq_length,
                time_steps=time_steps,
                sampling_time_steps=sampling_time_steps,
                beta_schedule=beta_schedule,
                ddim_sampling_eta=ddim_sampling_eta,
                use_self_cond=use_self_cond,
                clip_x0=clip_x0,
                sampling_x0_clip=sampling_x0_clip,
                self_condition_probability=self_condition_probability,
                gradient_checkpointing=gradient_checkpointing,
            )
        self.diffusions = nn.ModuleDict(experts)

    @property
    def region_diffusion(self) -> ConditionalGaussianDiffusion1D:
        return self.diffusions["region"]

    @property
    def syntax_diffusion(self) -> ConditionalGaussianDiffusion1D:
        return self.diffusions["syntax"]

    @property
    def road_diffusion(self) -> ConditionalGaussianDiffusion1D:
        return self.diffusions["road"]

    @staticmethod
    def _reject_low(features: Mapping[str, torch.Tensor]) -> None:
        forbidden = [key for key in features if "low" in key.lower()]
        if forbidden:
            raise ValueError(f"严格模式: H_*_low 不能绕过 RAG 进入 Diffusion: {forbidden}")

    def _validate_inputs(
        self,
        high_features: Mapping[str, torch.Tensor],
        rag_outputs: Mapping[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        self._reject_low(high_features)
        highs, references = {}, {}
        device = next(self.parameters()).device
        for layer in LAYER_ORDER:
            high_key = f"H_{layer}_high"
            reference_key = f"R_{layer}"
            if high_key not in high_features or reference_key not in rag_outputs:
                raise KeyError(f"Diffusion 缺少 {high_key} 或 {reference_key}")
            highs[layer] = _ensure_batched(high_features[high_key], 2, high_key).to(device)
            references[layer] = _ensure_batched(
                rag_outputs[reference_key], 3, reference_key
            ).to(device)
            if highs[layer].shape[:2] != references[layer].shape[:2]:
                raise ValueError(f"{layer} high/reference B/N 不一致")
            if highs[layer].shape[-1] != self.high_dim:
                raise ValueError(f"{layer} high dimension 不匹配")
            if references[layer].shape[2:] != (LAYER_CHANNELS[layer], self.seq_length):
                raise ValueError(f"{layer} RAG reference C/T 不匹配")
        batches = {value.shape[0] for value in highs.values()}
        if len(batches) != 1:
            raise ValueError("三层 batch size 不一致")
        return highs, references

    @staticmethod
    def _operator(
        parent_operator: Mapping[str, Mapping[str, torch.Tensor]], name: str
    ) -> Mapping[str, torch.Tensor]:
        if name not in parent_operator:
            raise KeyError(f"严格模式: 缺少稀疏父子算子 {name}")
        return parent_operator[name]

    def _condition(
        self,
        layer: str,
        high: torch.Tensor,
        reference: torch.Tensor,
        calendar: Mapping[str, Any],
        parent_dynamic: torch.Tensor | None = None,
    ) -> torch.Tensor:
        condition = self.conditioners[layer](
            high, reference, calendar, parent_dynamic=parent_dynamic
        )
        flat, shape = flatten_node_conditions(condition)
        if shape != high.shape[:2]:
            raise RuntimeError("内部 condition node reshape 不一致")
        return flat

    def _maybe_dropout_parent(self, value: torch.Tensor) -> torch.Tensor:
        if not self.training or self.parent_condition_dropout <= 0:
            return value
        keep = torch.rand(
            (value.shape[0], 1, 1, 1), device=value.device
        ) >= self.parent_condition_dropout
        return value * keep.to(value.dtype)

    @staticmethod
    def _slice_batch(value: torch.Tensor | None, start: int, end: int) -> torch.Tensor | None:
        if value is None or value.ndim == 0:
            return value
        return value[start:end]

    def _training_loss_chunked(
        self,
        diffusion: ConditionalGaussianDiffusion1D,
        value: torch.Tensor,
        condition: torch.Tensor,
        *,
        mask: torch.Tensor | None,
        noise: torch.Tensor | None,
        timesteps: torch.Tensor | None,
        force_self_condition: bool | None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Run node-wise diffusion chunks while preserving the full mean loss."""

        chunk_size = self.node_chunk_size
        total = int(value.shape[0])
        if chunk_size is None or chunk_size >= total:
            return diffusion.training_loss(
                value,
                condition,
                mask=mask,
                noise=noise,
                timesteps=timesteps,
                force_self_condition=force_self_condition,
            )

        slices = [(start, min(start + chunk_size, total)) for start in range(0, total, chunk_size)]
        if mask is None:
            weights = [float(end - start) / float(total) for start, end in slices]
        else:
            mask_total = torch.broadcast_to(mask, value.shape).sum().detach().item()
            if mask_total > 0:
                weights = [
                    float(torch.broadcast_to(mask[start:end], value[start:end].shape).sum().detach())
                    / mask_total
                    for start, end in slices
                ]
            else:
                weights = [0.0] * len(slices)

        losses = []
        for (start, end), weight in zip(slices, weights):
            loss, _ = diffusion.training_loss(
                value[start:end],
                condition[start:end],
                mask=self._slice_batch(mask, start, end),
                noise=self._slice_batch(noise, start, end),
                timesteps=self._slice_batch(timesteps, start, end),
                force_self_condition=force_self_condition,
            )
            losses.append(loss * weight)
        return torch.stack(losses).sum(), {
            "chunk_count": len(slices),
            "chunk_size": self.node_chunk_size,
            "num_nodes": total,
        }

    def _sample_chunked(
        self,
        diffusion: ConditionalGaussianDiffusion1D,
        condition: torch.Tensor,
        *,
        initial_noise: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Sample node chunks independently and concatenate them on the node axis."""

        chunk_size = self.node_chunk_size
        total = int(condition.shape[0])
        if chunk_size is None or chunk_size >= total:
            return diffusion.sample(condition, initial_noise=initial_noise)
        outputs, info = [], None
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            chunk, info = diffusion.sample(
                condition[start:end],
                initial_noise=None if initial_noise is None else initial_noise[start:end],
            )
            outputs.append(chunk)
        assert info is not None
        info = dict(info)
        info["chunk_count"] = len(outputs)
        info["chunk_size"] = self.node_chunk_size
        return torch.cat(outputs, dim=0), info

    def training_loss(
        self,
        *,
        future: Mapping[str, torch.Tensor],
        high_features: Mapping[str, torch.Tensor],
        rag_outputs: Mapping[str, torch.Tensor],
        calendar: Mapping[str, Any],
        parent_operator: Mapping[str, Mapping[str, torch.Tensor]],
        masks: Mapping[str, torch.Tensor] | None = None,
        noise: Mapping[str, torch.Tensor] | None = None,
        timesteps: Mapping[str, torch.Tensor] | None = None,
        force_self_condition: bool | None = None,
    ) -> dict[str, Any]:
        """Train with real normalized parents (teacher forcing)."""

        highs, references = self._validate_inputs(high_features, rag_outputs)
        device = next(self.parameters()).device
        targets = {
            layer: _ensure_batched(future[layer], 3, f"future.{layer}").to(device)
            for layer in LAYER_ORDER
        }
        for layer in LAYER_ORDER:
            if targets[layer].shape != references[layer].shape:
                raise ValueError(f"future.{layer} 与 RAG reference shape 不一致")
        region_condition = self._condition(
            "region", highs["region"], references["region"], calendar
        )
        region_flat, region_shape = flatten_nodes(targets["region"])
        region_loss, region_details = self._training_loss_chunked(
            self.region_diffusion,
            region_flat,
            region_condition,
            mask=None if masks is None else flatten_nodes(_ensure_batched(masks["region"], 3, "mask.region").to(device))[0],
            noise=None if noise is None else flatten_nodes(_ensure_batched(noise["region"], 3, "noise.region").to(device))[0],
            timesteps=None if timesteps is None else timesteps.get("region"),
            force_self_condition=force_self_condition,
        )

        syntax_parent = broadcast_parent_to_children(
            targets["region"],
            self._operator(parent_operator, "syntax_to_region"),
            child_count=targets["syntax"].shape[1],
        )
        syntax_parent = self._maybe_dropout_parent(syntax_parent)
        syntax_condition = self._condition(
            "syntax", highs["syntax"], references["syntax"], calendar, syntax_parent
        )
        syntax_flat, syntax_shape = flatten_nodes(targets["syntax"])
        syntax_loss, syntax_details = self._training_loss_chunked(
            self.syntax_diffusion,
            syntax_flat,
            syntax_condition,
            mask=None if masks is None else flatten_nodes(_ensure_batched(masks["syntax"], 3, "mask.syntax").to(device))[0],
            noise=None if noise is None else flatten_nodes(_ensure_batched(noise["syntax"], 3, "noise.syntax").to(device))[0],
            timesteps=None if timesteps is None else timesteps.get("syntax"),
            force_self_condition=force_self_condition,
        )

        road_parent = broadcast_parent_to_children(
            targets["syntax"],
            self._operator(parent_operator, "road_to_syntax"),
            child_count=targets["road"].shape[1],
        )
        road_parent = self._maybe_dropout_parent(road_parent)
        road_condition = self._condition(
            "road", highs["road"], references["road"], calendar, road_parent
        )
        road_flat, road_shape = flatten_nodes(targets["road"])
        road_loss, road_details = self._training_loss_chunked(
            self.road_diffusion,
            road_flat,
            road_condition,
            mask=None if masks is None else flatten_nodes(_ensure_batched(masks["road"], 3, "mask.road").to(device))[0],
            noise=None if noise is None else flatten_nodes(_ensure_batched(noise["road"], 3, "noise.road").to(device))[0],
            timesteps=None if timesteps is None else timesteps.get("road"),
            force_self_condition=force_self_condition,
        )
        losses = {"region": region_loss, "syntax": syntax_loss, "road": road_loss}
        total = sum(self.layer_loss_weights[layer] * losses[layer] for layer in LAYER_ORDER)
        return {
            "loss": total,
            "layer_losses": losses,
            "details": {
                "region": region_details,
                "syntax": syntax_details,
                "road": road_details,
            },
            "node_batch_shapes": {
                "region": region_shape, "syntax": syntax_shape, "road": road_shape
            },
            "teacher_forcing": True,
        }

    @torch.no_grad()
    def generate(
        self,
        *,
        high_features: Mapping[str, torch.Tensor],
        rag_outputs: Mapping[str, torch.Tensor],
        calendar: Mapping[str, Any],
        parent_operator: Mapping[str, Mapping[str, torch.Tensor]],
        initial_noise: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        """Generate coarse-to-fine without accepting any real future tensor."""

        highs, references = self._validate_inputs(high_features, rag_outputs)
        generated: dict[str, torch.Tensor] = {}
        sampling: dict[str, Any] = {}

        region_condition = self._condition(
            "region", highs["region"], references["region"], calendar
        )
        region_shape = highs["region"].shape[:2]
        region_noise = None
        if initial_noise is not None and "region" in initial_noise:
            region_noise = flatten_nodes(_ensure_batched(initial_noise["region"], 3, "initial_noise.region").to(region_condition.device))[0]
        region_flat, sampling["region"] = self._sample_chunked(
            self.region_diffusion,
            region_condition,
            initial_noise=region_noise,
        )
        generated["region"] = unflatten_nodes(region_flat, region_shape)

        syntax_parent = broadcast_parent_to_children(
            generated["region"],
            self._operator(parent_operator, "syntax_to_region"),
            child_count=highs["syntax"].shape[1],
        )
        syntax_condition = self._condition(
            "syntax", highs["syntax"], references["syntax"], calendar, syntax_parent
        )
        syntax_shape = highs["syntax"].shape[:2]
        syntax_noise = None
        if initial_noise is not None and "syntax" in initial_noise:
            syntax_noise = flatten_nodes(_ensure_batched(initial_noise["syntax"], 3, "initial_noise.syntax").to(syntax_condition.device))[0]
        syntax_flat, sampling["syntax"] = self._sample_chunked(
            self.syntax_diffusion,
            syntax_condition,
            initial_noise=syntax_noise,
        )
        generated["syntax"] = unflatten_nodes(syntax_flat, syntax_shape)

        road_parent = broadcast_parent_to_children(
            generated["syntax"],
            self._operator(parent_operator, "road_to_syntax"),
            child_count=highs["road"].shape[1],
        )
        road_condition = self._condition(
            "road", highs["road"], references["road"], calendar, road_parent
        )
        road_shape = highs["road"].shape[:2]
        road_noise = None
        if initial_noise is not None and "road" in initial_noise:
            road_noise = flatten_nodes(_ensure_batched(initial_noise["road"], 3, "initial_noise.road").to(road_condition.device))[0]
        road_flat, sampling["road"] = self._sample_chunked(
            self.road_diffusion,
            road_condition,
            initial_noise=road_noise,
        )
        generated["road"] = unflatten_nodes(road_flat, road_shape)
        return {
            "generated_region": generated["region"],
            "generated_syntax": generated["syntax"],
            "generated_road": generated["road"],
            "sampling": sampling,
            "generation_order": LAYER_ORDER,
            "teacher_forcing": False,
        }


class ThreeLayerRAGDiffusionSystem(nn.Module):
    """Wire trainable RAG references into the hierarchical Diffusion loss."""

    def __init__(self, rag: nn.Module, diffusion: HierarchicalThreeLayerDiffusion):
        super().__init__()
        self.rag = rag
        self.diffusion = diffusion

    @staticmethod
    def _high_mapping(high_features: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            f"H_{layer}_high": high_features[f"H_{layer}_high"]
            for layer in LAYER_ORDER
        }

    def training_loss(
        self,
        inputs: ThreeLayerRAGInputs,
        *,
        high_features: Mapping[str, torch.Tensor],
        memory: ThreeLayerRAGMemory,
        masks: Mapping[str, torch.Tensor] | None = None,
        noise: Mapping[str, torch.Tensor] | None = None,
        timesteps: Mapping[str, torch.Tensor] | None = None,
        force_self_condition: bool | None = None,
    ) -> dict[str, Any]:
        inputs.validate(temporal_channels=LAYER_CHANNELS)
        if inputs.value_temporal_features is None:
            raise ValueError("Diffusion 训练必须提供独立 value_temporal_features")
        rag_outputs = self.rag(inputs, memory=memory, target_city=inputs.city_id)
        result = self.diffusion.training_loss(
            future=inputs.value_temporal_features,
            high_features=self._high_mapping(high_features),
            rag_outputs=rag_outputs,
            calendar=inputs.calendar,
            parent_operator=inputs.parent_operator or {},
            masks=masks,
            noise=noise,
            timesteps=timesteps,
            force_self_condition=force_self_condition,
        )
        result["rag"] = rag_outputs
        return result

    @torch.no_grad()
    def generate(
        self,
        inputs: ThreeLayerRAGInputs,
        *,
        high_features: Mapping[str, torch.Tensor],
        memory: ThreeLayerRAGMemory,
        initial_noise: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        """Use history only; drop any attached value before the RAG query."""

        query = ThreeLayerRAGInputs(
            city_id=inputs.city_id,
            split=inputs.split,
            low_features=inputs.low_features,
            temporal_features=inputs.history_temporal_features,
            calendar=inputs.calendar,
            parent_index=inputs.parent_index,
            parent_operator=inputs.parent_operator,
            value_temporal_features=None,
            graph_metadata=inputs.graph_metadata,
        ).validate(temporal_channels=LAYER_CHANNELS)
        rag_outputs = self.rag(query, memory=memory, target_city=query.city_id)
        result = self.diffusion.generate(
            high_features=self._high_mapping(high_features),
            rag_outputs=rag_outputs,
            calendar=query.calendar,
            parent_operator=query.parent_operator or {},
            initial_noise=initial_noise,
        )
        result["rag"] = rag_outputs
        result["used_future_value"] = False
        return result
