"""Configuration factories shared by Stage-4 train/generate CLIs."""

from __future__ import annotations

from typing import Any, Mapping

from three_layer_rag import HierarchicalThreeLayerRAG, RAG_CONTRACT_VERSION

from .model import (
    DIFFUSION_CONTRACT_VERSION,
    HierarchicalThreeLayerDiffusion,
    ThreeLayerRAGDiffusionSystem,
)


def validate_stage4_config(config: Mapping[str, Any]) -> None:
    """Fail early on cross-module contract mismatches instead of partial builds."""

    if not isinstance(config.get("rag"), Mapping) or not isinstance(config.get("diffusion"), Mapping):
        raise ValueError("Stage-4 配置必须包含 rag/diffusion mapping")
    rag, diffusion = config["rag"], config["diffusion"]
    if rag.get("contract_version") != RAG_CONTRACT_VERSION:
        raise ValueError("RAG contract_version 与实现不匹配")
    if diffusion.get("contract_version") != DIFFUSION_CONTRACT_VERSION:
        raise ValueError("Diffusion contract_version 与实现不匹配")
    if int(rag.get("seq_length", -1)) != int(diffusion.get("seq_length", -2)):
        raise ValueError("RAG/Diffusion seq_length 必须一致")
    expected_channels = {"road": 3, "syntax": 3, "region": 2}
    if {str(k): int(v) for k, v in rag.get("temporal_channels", {}).items()} != expected_channels:
        raise ValueError("RAG temporal_channels 必须为 road=3/syntax=3/region=2")
    condition = diffusion.get("condition", {})
    if int(condition.get("high_dim", 0)) <= 0 or int(condition.get("cond_dim", 0)) <= 0:
        raise ValueError("Diffusion condition high_dim/cond_dim 必须为正")
    if int(diffusion.get("sampling_time_steps", 0)) > int(diffusion.get("time_steps", 0)):
        raise ValueError("sampling_time_steps 不能大于 time_steps")
    if set(diffusion.get("layer_loss_weights", {})) != {"road", "syntax", "region"}:
        raise ValueError("layer_loss_weights 必须覆盖三层")


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
        candidate_chunk_size=int(rag.get("candidate_chunk_size", 4096)),
        city_top_k=None if rag.get("city_top_k") is None else int(rag["city_top_k"]),
        train_source_keys=bool(rag.get("train_source_keys", True)),
        source_cache_device=str(rag.get("source_cache_device", "model")),
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
        node_chunk_size=(
            None if diffusion.get("node_chunk_size") is None
            else int(diffusion["node_chunk_size"])
        ),
        gradient_checkpointing=bool(diffusion.get("gradient_checkpointing", False)),
    )


def build_system(config: Mapping[str, Any]) -> ThreeLayerRAGDiffusionSystem:
    validate_stage4_config(config)
    return ThreeLayerRAGDiffusionSystem(build_rag_model(config), build_diffusion_model(config))
