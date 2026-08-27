"""HCFM 宏观/微观编码、CRAFT GFA/RAG 条件与耦合 Flow Matching。"""

from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np
import ot
import torch
import torch.nn as nn
import torch.nn.functional as F

from craft_integrated.graph_transformer_pytorch import GraphTransformer

from .adversarial import RoadAdversarialModule, RoadEncoder, adversarial_losses
from .flow_matching import (
    GraphTemporalVectorField,
    integrate_coupled_ode,
    sample_priors,
    straight_path,
    validate_velocity_consistency,
)
from .hierarchy import aggregate_micro_to_macro, sparse_transpose_apply
from .interaction import HierarchicalInteraction, pool_roads_to_regions
from .losses import generation_losses


class CraftFeatureInitLayer(nn.Module):
    """与第一阶段 ``FeatureInitLayer`` 参数结构和逐城市标准化语义一致。"""

    def __init__(self, raw_feature_dim: int, rep_dim: int):
        super().__init__()
        self.eps = 1e-5
        self.init_proj = nn.Sequential(
            nn.Dropout(p=0.05), nn.Linear(raw_feature_dim, rep_dim), nn.ReLU(),
            nn.Linear(rep_dim, rep_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        mean = value.mean(dim=0, keepdim=True)
        variance = value.var(dim=0, unbiased=False, keepdim=True)
        return self.init_proj((value - mean) / torch.sqrt(variance + self.eps))


class CraftMacroEncoderGFA(nn.Module):
    """CRAFT 45 维 MacroEncoder 与原 GraphTransformer/GFA。"""

    def __init__(self, raw_dim: int = 45, rep_dim: int = 128):
        super().__init__()
        self.init_proj = CraftFeatureInitLayer(raw_dim, rep_dim)
        self.gnn = GraphTransformer(
            dim=rep_dim, depth=3, heads=4, dim_head=64,
            with_feedforwards=True, rel_pos_emb=False,
            accept_adjacency_matrix=True,
        )

    def encode(self, region_x: torch.Tensor) -> torch.Tensor:
        """``region_x [B,N,45] -> H_macro [B,N,D]``。"""

        if region_x.ndim != 3 or region_x.shape[-1] != 45:
            raise ValueError("MacroEncoder 期望 region_x[B,N,45]")
        return torch.stack([self.init_proj(graph) for graph in region_x], dim=0)

    def align(self, macro: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """层次融合后进入原 CRAFT GraphTransformer/GFA。"""

        nodes = macro.shape[1]
        adjacency = torch.zeros(
            (nodes, nodes), dtype=macro.dtype, device=macro.device
        )
        src, dst = edge_index.to(macro.device)
        adjacency[src, dst] = 1.0
        aligned, _ = self.gnn(nodes=macro, adj_mat=adjacency.unsqueeze(0).expand(macro.shape[0], -1, -1))
        return aligned


def tfa_self_similarity_loss(embedding: torch.Tensor, values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    emb_distance = torch.cdist(embedding, embedding)
    value_distance = torch.cdist(values, values)
    emb_range = (emb_distance.max() - emb_distance.min()).clamp_min(eps)
    value_range = (value_distance.max() - value_distance.min()).clamp_min(eps)
    return F.mse_loss(
        (emb_distance - emb_distance.min()) / emb_range,
        (value_distance - value_distance.min()) / value_range,
    )


def cca_wasserstein_loss(source: torch.Tensor, target: torch.Tensor, metric: str = "euclidean") -> torch.Tensor:
    """保留第一阶段 POT/EMD 固定 transport plan 的 CCA 语义。"""

    if metric == "euclidean":
        source_norm = source / source.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        target_norm = target / target.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        distance = torch.cdist(source_norm, target_norm)
    elif metric == "cosine":
        distance = 1.0 - F.cosine_similarity(source[:, None], target[None], dim=-1)
    else:
        raise ValueError(f"未知 CCA metric={metric!r}")
    plan = ot.emd(
        np.ones(len(source)) / len(source), np.ones(len(target)) / len(target),
        distance.detach().cpu().numpy(),
    )
    return (torch.as_tensor(plan, dtype=distance.dtype, device=distance.device) * distance).sum()


def positional_encoding(max_length: int, dim: int) -> torch.Tensor:
    result = torch.zeros(max_length, dim)
    position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
    divisor = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
    result[:, 0::2] = torch.sin(position * divisor)
    result[:, 1::2] = torch.cos(position * divisor)
    return result


class ReferenceEncoder(nn.Module):
    """与 CRAFT ReferTransformer 等价的 Region 级 Reference 编码。"""

    def __init__(self, channels: int, seq_length: int, dim: int, heads: int, layers: int):
        super().__init__()
        self.position = nn.Embedding.from_pretrained(positional_encoding(seq_length, dim), freeze=True)
        self.input_proj = nn.Linear(channels, dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=dim, batch_first=True
            ),
            layers,
        )

    def forward(self, reference: torch.Tensor) -> torch.Tensor:
        """``[B,N,2,T] -> [B,N,D]``，每个 Region 独立编码。"""

        if reference.ndim != 4 or reference.shape[2] != 2:
            raise ValueError("ReferenceEncoder 期望 [B,N,2,T]")
        b, n, _, length = reference.shape
        value = reference.reshape(b * n, 2, length).permute(0, 2, 1)
        value = self.input_proj(value)
        # 第一阶段实现未显式相加 position embedding；为 checkpoint/语义兼容保持不加。
        value = self.transformer(value).mean(dim=1)
        return value.reshape(b, n, -1)


class TimeCondition(nn.Module):
    def __init__(self, hour_dim: int, weekday_dim: int, month_dim: int):
        super().__init__()
        self.hour_embedding = nn.Embedding(24, hour_dim)
        self.weekday_embedding = nn.Embedding(7, weekday_dim)
        self.month_embedding = nn.Embedding(12, month_dim)

    @property
    def output_dim(self) -> int:
        return (
            self.hour_embedding.embedding_dim + self.weekday_embedding.embedding_dim
            + self.month_embedding.embedding_dim
        )

    def forward(self, batch: Mapping[str, Any], device: torch.device, batch_size: int) -> torch.Tensor:
        def tensor_value(name: str, upper: int) -> torch.Tensor:
            value = batch.get(name, batch.get("time_features", {}).get(name))
            value = torch.as_tensor(value, dtype=torch.long, device=device).reshape(-1)
            if value.numel() == 1 and batch_size > 1:
                value = value.expand(batch_size)
            if value.numel() != batch_size or (value < 0).any() or (value >= upper).any():
                raise ValueError(f"时间字段 {name} 非法或 batch 不一致")
            return value
        return torch.cat([
            self.hour_embedding(tensor_value("start_hour", 24)),
            self.weekday_embedding(tensor_value("weekday", 7)),
            self.month_embedding(tensor_value("month", 12)),
        ], dim=-1)


def _broadcast_macro_sequence(p_struct: torch.Tensor, macro: torch.Tensor) -> torch.Tensor:
    """``P.T @ macro [B,N,C,T] -> [B,M,C,T]``。"""

    b, n, channels, length = macro.shape
    if n != p_struct.shape[0]:
        raise ValueError("macro 与 P_struct Region 维不一致")
    flat = macro.permute(1, 0, 2, 3).reshape(n, -1)
    road = torch.sparse.mm(p_struct.coalesce().transpose(0, 1).to(macro.device), flat)
    return road.reshape(p_struct.shape[1], b, channels, length).permute(1, 0, 2, 3)


def _normalizer_transform(normalizer, value: torch.Tensor) -> torch.Tensor:
    return normalizer.transform(value.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)


class MacroFlowMatchingModel(nn.Module):
    """A2 公平对比：第一阶段条件不变，只把 Diffusion 替换为 Macro FM。"""

    def __init__(self, cfg: Mapping[str, Any]):
        super().__init__()
        model, flow = cfg["model"], cfg["flow_matching"]
        rep_dim, refer_dim = int(model["rep_dim"]), int(model["refer_dim"])
        self.temporal_encoder = ReferenceEncoder(
            2, int(model["seq_length"]), refer_dim,
            int(model["refer_heads"]), int(model["refer_layers"]),
        )
        self.time_condition = TimeCondition(
            int(model["hour_dim"]), int(model["weekday_dim"]), int(model["month_dim"])
        )
        condition_dim = rep_dim + refer_dim + self.time_condition.output_dim
        self.vector_field = GraphTemporalVectorField(
            2, condition_dim, 0, int(flow["hidden_dim"]), int(flow["num_blocks"]),
            int(flow["time_dim"]), float(flow["dropout"]),
        )

    def condition(self, batch: Mapping[str, Any]) -> torch.Tensor:
        aligned, reference = batch["aligned_region_rep"], batch["reference"]
        if aligned.ndim != 3 or reference.ndim != 4 or aligned.shape[:2] != reference.shape[:2]:
            raise ValueError("Macro FM 条件应为 aligned[B,N,D], reference[B,N,2,T]")
        time = self.time_condition(batch, aligned.device, aligned.shape[0])
        return torch.cat([
            aligned, self.temporal_encoder(reference),
            time[:, None].expand(-1, aligned.shape[1], -1),
        ], dim=-1)

    def loss(
        self, batch: Mapping[str, Any], macro_normalizer, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        target = _normalizer_transform(macro_normalizer, batch["macro_flow"])
        initial = torch.randn(target.shape, dtype=target.dtype, device=target.device, generator=generator)
        time = torch.rand(target.shape[0], dtype=target.dtype, device=target.device, generator=generator)
        state, target_velocity = straight_path(initial, target, time)
        velocity = self.vector_field(
            state, time, self.condition(batch), batch["region_edge_index"]
        )
        from .flow_matching import masked_mse
        return masked_mse(velocity, target_velocity, batch["region_mask"])

    def generate(
        self, batch: Mapping[str, Any], *, steps: int = 16, solver: str = "euler",
        generator: torch.Generator | None = None,
    ):
        shape = batch["macro_flow"].shape
        initial = torch.randn(shape, dtype=batch["macro_flow"].dtype, device=batch["macro_flow"].device, generator=generator)
        condition = self.condition(batch)
        dummy = torch.empty((shape[0], 0, 1, shape[-1]), device=initial.device)
        def field(macro, _micro, time):
            return self.vector_field(macro, time, condition, batch["region_edge_index"]), _micro
        macro, _, stats = integrate_coupled_ode(
            field, initial, dummy, steps=steps, solver=solver
        )
        return macro, stats


class HCFMModel(nn.Module):
    """完整 Hierarchical Cross-City Flow Matching。"""

    def __init__(self, cfg: Mapping[str, Any]):
        super().__init__()
        model, hierarchy, adversarial, flow = (
            cfg["model"], cfg["hierarchy"], cfg["micro_adversarial"], cfg["flow_matching"]
        )
        rep_dim, road_dim = int(model["rep_dim"]), int(model["road_hidden_dim"])
        self.prior_mode = str(flow["prior_mode"])
        self.generate_micro = bool(cfg.get("generate_micro", True))
        self.use_micro_adversarial = bool(cfg.get("use_micro_adversarial", True))
        self.macro_encoder = CraftMacroEncoderGFA(45, rep_dim)
        road_cfg = dict(adversarial)
        road_cfg.update({"road_dim": int(model["road_dim"]), "hidden_dim": road_dim})
        if self.use_micro_adversarial:
            self.road_adversarial = RoadAdversarialModule(road_cfg)
            self.road_encoder = None
        else:
            self.road_adversarial = None
            self.road_encoder = RoadEncoder(
                road_dim=int(model["road_dim"]), hidden_dim=road_dim,
                num_layers=int(adversarial.get("num_layers", 4)),
                heads=int(adversarial.get("heads", 4)),
                dropout=float(adversarial.get("dropout", 0.1)),
                edge_dim=int(adversarial["edge_dim"]) if adversarial.get("edge_dim") is not None else None,
            )
        self.hierarchy = HierarchicalInteraction(
            rep_dim, road_dim, int(hierarchy["num_layers"]), str(hierarchy["fusion"]),
            bool(hierarchy["bidirectional"]),
        )
        refer_dim = int(model["refer_dim"])
        self.reference_encoder = ReferenceEncoder(
            2, int(model["seq_length"]), refer_dim,
            int(model["refer_heads"]), int(model["refer_layers"]),
        )
        self.time_condition = TimeCondition(
            int(model["hour_dim"]), int(model["weekday_dim"]), int(model["month_dim"])
        )
        time_dim = self.time_condition.output_dim
        macro_condition_dim = rep_dim + refer_dim + time_dim + road_dim
        micro_condition_dim = road_dim + rep_dim + time_dim
        self.macro_vector_field = GraphTemporalVectorField(
            2, macro_condition_dim, 2 if self.generate_micro else 0,
            int(flow["hidden_dim"]), int(flow["num_blocks"]),
            int(flow["time_dim"]), float(flow["dropout"]),
        )
        self.micro_vector_field = (
            GraphTemporalVectorField(
                1, micro_condition_dim, 2, int(flow["hidden_dim"]), int(flow["num_blocks"]),
                int(flow["time_dim"]), float(flow["dropout"]),
            ) if self.generate_micro else None
        )

    def encode_conditions(self, batch: Mapping[str, Any]) -> Dict[str, Any]:
        region_x, road_x = batch["region_x"], batch["road_x"]
        if region_x.shape[0] != 1 or road_x.shape[0] != 1:
            raise ValueError("HCFM 当前严格采用 batch_size=1 的整城快照")
        macro = self.macro_encoder.encode(region_x)
        if self.use_micro_adversarial:
            road_output = self.road_adversarial(
                road_x[0], batch["road_edge_index"], batch.get("road_edge_attr")
            )
            road_semantic = road_output["semantic"].unsqueeze(0)
        else:
            road_output = None
            road_semantic = self.road_encoder(
                road_x[0], batch["road_edge_index"], batch.get("road_edge_attr")
            ).unsqueeze(0)
        macro_fused, road_fused = self.hierarchy(macro, road_semantic, batch["p_struct"])
        aligned = self.macro_encoder.align(macro_fused, batch["region_edge_index"])
        reference = batch.get("reference")
        if reference is None:
            raise KeyError("严格模式: HCFM 缺少源城市 train RAG 生成的 Region reference")
        encoded_reference = self.reference_encoder(reference)
        time = self.time_condition(batch, aligned.device, aligned.shape[0])
        time_region = time[:, None].expand(-1, aligned.shape[1], -1)
        time_road = time[:, None].expand(-1, road_fused.shape[1], -1)
        pooled_road = pool_roads_to_regions(batch["p_struct"], road_fused)
        macro_context = sparse_transpose_apply(batch["p_struct"], aligned)
        return {
            "macro_condition": torch.cat(
                [aligned, encoded_reference, time_region, pooled_road], dim=-1
            ),
            "micro_condition": torch.cat([road_fused, macro_context, time_road], dim=-1),
            "aligned": aligned,
            "road_fused": road_fused,
            "road_adversarial": road_output,
        }

    def velocities(
        self,
        macro_state: torch.Tensor,
        micro_state: torch.Tensor,
        time: torch.Tensor,
        conditions: Mapping[str, torch.Tensor],
        batch: Mapping[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.generate_micro or self.micro_vector_field is None:
            raise RuntimeError("generate_micro=false 时应调用 macro_velocity，不应调用耦合 velocities")
        aggregated_micro = aggregate_micro_to_macro(
            micro_state, batch["b_in"], batch["b_out"]
        )
        broadcast_macro = _broadcast_macro_sequence(batch["p_struct"], macro_state)
        macro_velocity = self.macro_vector_field(
            macro_state, time, conditions["macro_condition"],
            batch["region_edge_index"], aggregated_micro,
        )
        micro_velocity = self.micro_vector_field(
            micro_state, time, conditions["micro_condition"],
            batch["road_edge_index"], broadcast_macro,
        )
        return macro_velocity, micro_velocity

    def macro_velocity(
        self, macro_state: torch.Tensor, time: torch.Tensor,
        conditions: Mapping[str, torch.Tensor], batch: Mapping[str, Any],
    ) -> torch.Tensor:
        return self.macro_vector_field(
            macro_state, time, conditions["macro_condition"], batch["region_edge_index"]
        )

    def flow_losses(
        self,
        batch: Mapping[str, Any],
        macro_normalizer,
        micro_normalizer,
        *,
        cross_velocity_weight: float,
        conservation_calibrator=None,
        generator: torch.Generator | None = None,
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        validate_velocity_consistency(self.prior_mode, cross_velocity_weight)
        macro_true = _normalizer_transform(macro_normalizer, batch["macro_flow"])
        micro_true = (
            _normalizer_transform(micro_normalizer, batch["micro_flow"])
            if self.generate_micro else None
        )
        if not self.generate_micro:
            if self.prior_mode != "independent" or cross_velocity_weight > 0:
                raise ValueError("generate_micro=false 只支持 independent prior 且 cross_velocity=0")
            macro_initial = torch.randn(
                macro_true.shape, dtype=macro_true.dtype, device=macro_true.device, generator=generator
            )
            time = torch.rand(
                macro_true.shape[0], dtype=macro_true.dtype, device=macro_true.device, generator=generator
            )
            macro_state, macro_target_velocity = straight_path(macro_initial, macro_true, time)
            model_batch = dict(batch)
            model_batch["reference"] = _normalizer_transform(macro_normalizer, batch["reference"])
            conditions = self.encode_conditions(model_batch)
            macro_velocity = self.macro_velocity(macro_state, time, conditions, model_batch)
            zero = macro_velocity.sum() * 0.0
            from .flow_matching import masked_mse
            return {
                "fm_macro": masked_mse(macro_velocity, macro_target_velocity, batch["region_mask"]),
                "fm_micro": zero, "cross_state": zero, "cross_velocity": zero, "topology": zero,
            }, {"conditions": conditions, "macro_initial": macro_initial, "time": time}
        macro_initial, micro_initial = sample_priors(
            micro_true, macro_true, batch["b_in"], batch["b_out"], self.prior_mode, generator
        )
        time = torch.rand(
            macro_true.shape[0], dtype=macro_true.dtype, device=macro_true.device, generator=generator
        )
        macro_state, macro_target_velocity = straight_path(macro_initial, macro_true, time)
        micro_state, micro_target_velocity = straight_path(micro_initial, micro_true, time)
        # reference 也进入与宏观 flow 相同的归一化空间。
        model_batch = dict(batch)
        model_batch["reference"] = _normalizer_transform(macro_normalizer, batch["reference"])
        conditions = self.encode_conditions(model_batch)
        macro_velocity, micro_velocity = self.velocities(
            macro_state, micro_state, time, conditions, model_batch
        )
        losses = generation_losses(
            macro_velocity=macro_velocity, macro_target_velocity=macro_target_velocity,
            micro_velocity=micro_velocity, micro_target_velocity=micro_target_velocity,
            macro_state=macro_state, micro_state=micro_state,
            macro_true=macro_true, micro_true=micro_true, time=time,
            region_mask=batch["region_mask"], road_mask=batch["road_mask"],
            b_in=batch["b_in"], b_out=batch["b_out"],
            road_edge_index=batch["road_edge_index"],
            macro_inverse=macro_normalizer.inverse, micro_inverse=micro_normalizer.inverse,
            prior_mode=self.prior_mode, cross_velocity_weight=cross_velocity_weight,
            micro_to_macro_calibration=(
                conservation_calibrator.apply if conservation_calibrator is not None else None
            ),
        )
        auxiliary = {
            "conditions": conditions,
            "macro_initial": macro_initial,
            "micro_initial": micro_initial,
            "time": time,
        }
        return losses, auxiliary

    def auxiliary_losses(
        self,
        source_batch: Mapping[str, Any],
        source_auxiliary: Mapping[str, Any],
        target_static: Mapping[str, Any],
        *,
        source_city_label: int,
        target_city_label: int,
        source_cost_target: torch.Tensor | None = None,
        source_cost_mask: torch.Tensor | None = None,
        gfa_metric: str = "euclidean",
    ) -> Dict[str, torch.Tensor]:
        """道路对抗使用目标静态图但不接收目标动态标签；GFA 保持 TFA+CCA。"""

        target_road = target_static["road_x"]
        if target_road.ndim == 3:
            target_road = target_road[0]
        if self.use_micro_adversarial:
            target_output = self.road_adversarial(
                target_road, target_static["road_edge_index"], target_static.get("road_edge_attr")
            )
            target_road_semantic = target_output["semantic"].unsqueeze(0)
            adv = adversarial_losses(
                [source_auxiliary["conditions"]["road_adversarial"], target_output],
                [source_city_label, target_city_label],
                source_cost_target=source_cost_target, source_cost_mask=source_cost_mask,
            )
        else:
            zero = source_auxiliary["conditions"]["aligned"].sum() * 0.0
            adv = {name: zero for name in ("cost", "rank", "semantic_domain", "domain", "orthogonal")}
            target_road_semantic = self.road_encoder(
                target_road, target_static["road_edge_index"], target_static.get("road_edge_attr")
            ).unsqueeze(0)
        target_region = target_static["region_x"]
        if target_region.ndim == 2:
            target_region = target_region.unsqueeze(0)
        target_macro_base = self.macro_encoder.encode(target_region)
        target_macro_fused, _ = self.hierarchy(
            target_macro_base, target_road_semantic, target_static["p_struct"]
        )
        target_macro = self.macro_encoder.align(
            target_macro_fused, target_static["region_edge_index"]
        )[0]
        source_aligned = source_auxiliary["conditions"]["aligned"][0]
        source_value = source_batch["macro_flow"][0].reshape(source_aligned.shape[0], -1)
        region_mask = source_batch["region_mask"][0]
        tfa = tfa_self_similarity_loss(source_aligned[region_mask], source_value[region_mask])
        cca = cca_wasserstein_loss(source_aligned, target_macro, gfa_metric)
        return {**adv, "gfa_tfa": tfa, "gfa_cca": cca, "gfa": tfa + cca}

    def generate(
        self,
        batch: Mapping[str, Any],
        *,
        steps: int = 16,
        solver: str = "euler",
        generator: torch.Generator | None = None,
    ):
        macro_shape = batch["macro_flow"].shape
        if not self.generate_micro:
            macro_initial = torch.randn(
                macro_shape, dtype=batch["macro_flow"].dtype,
                device=batch["macro_flow"].device, generator=generator,
            )
            conditions = self.encode_conditions(batch)
            dummy = torch.empty((macro_shape[0], 0, 1, macro_shape[-1]), device=macro_initial.device)
            def macro_field(macro, empty, time):
                return self.macro_velocity(macro, time, conditions, batch), empty
            return integrate_coupled_ode(
                macro_field, macro_initial, dummy, steps=steps, solver=solver
            )
        micro_shape = batch["micro_flow"].shape
        dummy_macro = torch.zeros(macro_shape, dtype=batch["macro_flow"].dtype, device=batch["macro_flow"].device)
        dummy_micro = torch.zeros(micro_shape, dtype=batch["micro_flow"].dtype, device=batch["micro_flow"].device)
        macro_initial, micro_initial = sample_priors(
            dummy_micro, dummy_macro, batch["b_in"], batch["b_out"], self.prior_mode, generator
        )
        conditions = self.encode_conditions(batch)
        def field(macro, micro, time):
            return self.velocities(macro, micro, time, conditions, batch)
        return (*integrate_coupled_ode(
            field, macro_initial, micro_initial, steps=steps, solver=solver
        ),)


def compose_total_loss(
    losses: Mapping[str, torch.Tensor], weights: Mapping[str, float]
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """按宏观/微观/跨尺度三组汇总，且保留所有子损失供日志记录。"""

    required = {
        "fm_macro", "fm_micro", "cross_state", "cross_velocity", "topology",
        "cost", "rank", "semantic_domain", "domain", "orthogonal", "gfa",
    }
    missing = required - set(losses)
    if missing:
        raise KeyError(f"总损失缺少子项 {sorted(missing)}")
    def weighted(name: str) -> torch.Tensor:
        return float(weights.get(name, 0.0)) * losses[name]
    macro = weighted("fm_macro") + weighted("gfa")
    micro = sum((weighted(name) for name in (
        "fm_micro", "topology", "cost", "rank", "semantic_domain", "domain", "orthogonal"
    )), start=losses["fm_micro"].new_zeros(()))
    cross = weighted("cross_state") + weighted("cross_velocity")
    return macro + micro + cross, {"L_macro": macro, "L_micro": micro, "L_cross_scale": cross}
