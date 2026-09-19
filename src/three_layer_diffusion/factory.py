"""Configuration factories shared by Stage-4 train/generate CLIs."""

from __future__ import annotations

from typing import Any, Mapping

from three_layer_rag import HierarchicalThreeLayerRAG

from .model import HierarchicalThreeLayerDiffusion, ThreeLayerRAGDiffusionSystem


def build_rag_model(config: Mapping[str, Any]) -> HierarchicalThreeLayerRAG:
    rag = config["rag"]
    return HierarchicalThreeLayerRAG(
        low_dim=int(rag["low_dim"]),
        temporal_channels=rag["temporal_channels"],
        temporal_dim=int(rag["temporal_dim"]),
        retrieval_dim=int(rag["retrieval_dim"]),
        seq_length=int(rag["seq_length"]),
        temporal_layers=int(rag.get("temporal_layers", 1)),
        temporal_dropout=float(rag.get("temporal_dropout", 0.0)),
        calendar_dims=rag["calendar_dims"],
        top_k=int(rag["top_k"]),
        metric=str(rag["metric"]),
        temperature=float(rag["temperature"]),
        match_month=bool(rag.get("match_month", True)),
        match_holiday=bool(rag.get("match_holiday", False)),
        require_value_separation=bool(rag.get("require_value_separation", True)),
        expected_graph_identity=None,
    )


def build_diffusion_model(config: Mapping[str, Any]) -> HierarchicalThreeLayerDiffusion:
    diffusion = config["diffusion"]
    unet = diffusion["unet"]
    condition = diffusion["condition"]
    return HierarchicalThreeLayerDiffusion(
        high_dim=int(condition["high_dim"]),
        seq_length=int(diffusion["seq_length"]),
        cond_dim=int(condition["cond_dim"]),
        temporal_dim=int(condition["rag_temporal_dim"]),
        parent_temporal_dim=int(condition["parent_temporal_dim"]),
        calendar_dims=condition["calendar_dims"],
        time_steps=int(diffusion["time_steps"]),
        sampling_time_steps=int(diffusion["sampling_time_steps"]),
        beta_schedule=str(diffusion["beta_schedule"]),
        ddim_sampling_eta=float(diffusion["ddim_sampling_eta"]),
        use_self_cond=bool(diffusion["use_self_cond"]),
        clip_x0=bool(diffusion.get("clip_x0", False)),
        self_condition_probability=float(diffusion.get("self_condition_probability", 0.5)),
        init_dim=int(unet["init_dim"]),
        base_dim=int(unet["base_dim"]),
        dim_mults=tuple(int(value) for value in unet["dim_mults"]),
        sinusoidal_theta=float(unet.get("sinusoidal_theta", 10_000.0)),
        dropout=float(unet.get("dropout", 0.1)),
        attention_dim_head=int(unet.get("attention_dim_head", 64)),
        attention_heads=int(unet.get("attention_heads", 4)),
        temporal_layers=int(condition.get("temporal_layers", 1)),
        temporal_dropout=float(condition.get("temporal_dropout", 0.0)),
        layer_loss_weights=diffusion["layer_loss_weights"],
        parent_condition_dropout=float(condition.get("parent_condition_dropout", 0.0)),
    )


def build_system(config: Mapping[str, Any]) -> ThreeLayerRAGDiffusionSystem:
    return ThreeLayerRAGDiffusionSystem(build_rag_model(config), build_diffusion_model(config))
