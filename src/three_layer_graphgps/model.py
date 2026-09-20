"""One GraphGPS encoder over the complete Road/Syntax/Region hierarchy."""

from __future__ import annotations

from typing import Any, Mapping
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from static_hierarchy.contracts import CityStaticHierarchy, validate_city_static_hierarchy

from .data import (
    RELATION_NAMES,
    SPECTRAL_FEATURE_VERSION,
    build_joint_three_layer_graph,
    validate_joint_three_layer_graph,
)
from .frequency import SpectralFeatureDecoupler
from .pooling import pool_road_to_syntax, pool_syntax_to_region
from .posenc import LapPEEncoder
from .spectral_lap_pe import LAPPE_VERSION, HierarchyLaplacianPE, prepare_hierarchy_lappe


class LinearGlobalAttention(nn.Module):
    """Positive-kernel linear attention without a node-by-node matrix."""

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("严格模式: hidden_dim 必须能被 attention heads 整除")
        self.heads, self.head_dim = int(heads), dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.output = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nodes = x.shape[0]
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = F.elu(q.view(nodes, self.heads, self.head_dim)) + 1.0
        k = F.elu(k.view(nodes, self.heads, self.head_dim)) + 1.0
        v = v.view(nodes, self.heads, self.head_dim)
        kv = torch.einsum("nhd,nhe->hde", k, v)
        denominator = torch.einsum("nhd,hd->nh", q, k.sum(dim=0)).clamp_min(1e-6)
        attended = torch.einsum("nhd,hde->nhe", q, kv) / denominator.unsqueeze(-1)
        return self.output(self.dropout(attended.reshape(nodes, -1)))


class GlobalAttentionBranch(nn.Module):
    """Global attention over the complete joint node set."""

    ALLOWED = {"none", "local", "linear", "full"}

    def __init__(self, dim: int, heads: int, dropout: float, mode: str, *, full_attention_max_nodes: int, layer_name: str = "joint", scope: str = "joint") -> None:
        super().__init__()
        if mode not in self.ALLOWED:
            raise ValueError(f"严格模式: global_attn={mode!r} 非法")
        if full_attention_max_nodes <= 0:
            raise ValueError("严格模式: full_attention_max_nodes 必须为正")
        self.mode = mode
        if scope not in {"joint", "same_layer"}:
            raise ValueError("严格模式: global_attention_scope 必须为 joint 或 same_layer")
        self.scope = scope
        self.layer_name = layer_name
        self.full_attention_max_nodes = int(full_attention_max_nodes)
        self.full_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True) if mode == "full" else None
        self.linear_attention = LinearGlobalAttention(dim, heads, dropout) if mode in {"linear", "full"} else None

    def _forward_scope(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode in {"none", "local"}:
            return torch.zeros_like(x)
        if self.mode == "linear":
            return self.linear_attention(x)
        if x.shape[0] > self.full_attention_max_nodes:
            warnings.warn(
                f"{self.layer_name} full attention 收到联合图 {x.shape[0]} 个节点，超过阈值 {self.full_attention_max_nodes}；自动 fallback 到 linear attention",
                RuntimeWarning,
                stacklevel=2,
            )
            return self.linear_attention(x)
        result, _ = self.full_attention(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0), need_weights=False)
        return result.squeeze(0)

    def forward(self, x: torch.Tensor, node_type: torch.Tensor | None = None) -> torch.Tensor:
        if self.scope == "joint":
            return self._forward_scope(x)
        if node_type is None or node_type.shape != (x.shape[0],):
            raise ValueError("same_layer global attention 需要 node_type[V]")
        output = torch.zeros_like(x)
        for kind in torch.unique(node_type, sorted=True).tolist():
            mask = node_type == int(kind)
            output[mask] = self._forward_scope(x[mask])
        return output


class RelationAwareWeightedMessagePassing(nn.Module):
    """Directed relation-aware local aggregation using exact edge weights."""

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.transforms = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in RELATION_NAMES])
        self.relation_bias = nn.Parameter(torch.zeros(len(RELATION_NAMES), dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        if edge_index.dtype != torch.long or edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("严格模式: local edge_index 必须为 LongTensor[2,E]")
        e = edge_index.shape[1]
        if edge_type.dtype != torch.long or edge_type.shape != (e,):
            raise ValueError("严格模式: local edge_type 长度错误")
        if edge_weight.shape != (e,) or not edge_weight.is_floating_point():
            raise ValueError("严格模式: local edge_weight 长度/dtype 错误")
        if e and (int(edge_index.min()) < 0 or int(edge_index.max()) >= x.shape[0]):
            raise ValueError("严格模式: local edge_index 越界")
        if e and (int(edge_type.min()) < 0 or int(edge_type.max()) >= len(RELATION_NAMES)):
            raise ValueError("严格模式: local edge_type 含未知关系")
        if not torch.isfinite(edge_weight).all() or (edge_weight <= 0).any():
            raise ValueError("严格模式: local edge_weight 必须为有限正数")
        src, dst = edge_index
        aggregate = torch.zeros_like(x)
        denominator = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        for relation, transform in enumerate(self.transforms):
            mask = edge_type == relation
            if not mask.any():
                continue
            weight = edge_weight[mask].to(x.device, dtype=x.dtype)
            message = transform(x[src[mask]]) + self.relation_bias[relation]
            aggregate.index_add_(0, dst[mask], message * weight.unsqueeze(-1))
            denominator.index_add_(0, dst[mask], weight)
        return self.dropout(aggregate / denominator.clamp_min(torch.finfo(x.dtype).eps).unsqueeze(-1))


class GraphGPSLayer(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float, global_attention: str, *, full_attention_max_nodes: int, global_attention_scope: str = "joint") -> None:
        super().__init__()
        self.local_norm = nn.LayerNorm(dim)
        self.global_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.local_mpnn = RelationAwareWeightedMessagePassing(dim, dropout)
        self.global_attention = GlobalAttentionBranch(dim, heads, dropout, global_attention, full_attention_max_nodes=full_attention_max_nodes, scope=global_attention_scope)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor, edge_weight: torch.Tensor, node_type: torch.Tensor | None = None) -> torch.Tensor:
        local_h = self.local_mpnn(self.local_norm(x), edge_index, edge_type, edge_weight)
        global_h = self.global_attention(self.global_norm(x), node_type=node_type)
        x = x + self.dropout(local_h) + self.dropout(global_h)
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        if not torch.isfinite(x).all():
            raise FloatingPointError("严格模式: GraphGPS layer 输出含 NaN/Inf")
        return x


class GraphGPSStack(nn.Module):
    """The only stack in the model; it always receives all joint nodes."""

    def __init__(self, dim: int, depth: int, heads: int, dropout: float, global_attention: str, *, full_attention_max_nodes: int, global_attention_scope: str = "joint") -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("严格模式: GraphGPS depth 必须为正")
        self.layers = nn.ModuleList([
            GraphGPSLayer(dim, heads, dropout, global_attention, full_attention_max_nodes=full_attention_max_nodes, global_attention_scope=global_attention_scope)
            for _ in range(depth)
        ])
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor, edge_weight: torch.Tensor, node_type: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, edge_index, edge_type, edge_weight, node_type=node_type)
        return self.output_norm(x)


def _nested(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"严格模式: 配置 {name} 必须为 mapping")
    return value


def validate_stage2_config(config: Mapping[str, Any]) -> None:
    if not isinstance(config.get("frequency"), Mapping) or not isinstance(config.get("data"), Mapping):
        raise KeyError("严格模式: Stage 2 config 缺少 frequency/data mapping")
    model_cfg, pos_cfg = _nested(config, "model"), _nested(config, "posenc")
    frequency_cfg, data_cfg, attention_cfg = _nested(config, "frequency"), _nested(config, "data"), _nested(config, "attention")
    if model_cfg.get("name", "three_layer_graphgps_lappe") != "three_layer_graphgps_lappe":
        raise ValueError("严格模式: model.name 必须为 three_layer_graphgps_lappe")
    if pos_cfg.get("type", "LapPE") != "LapPE":
        raise ValueError("严格模式: 第一版仅支持 posenc.type=LapPE")
    if frequency_cfg.get("enabled") is not True:
        raise ValueError("严格模式: frequency.enabled 必须为 true")
    if frequency_cfg.get("method", "low_rank_spectral_projection") != "low_rank_spectral_projection":
        raise ValueError("严格模式: frequency.method 仅支持 low_rank_spectral_projection")
    if frequency_cfg.get("decomposition_position", "after_graphgps") != "after_graphgps":
        raise ValueError("严格模式: frequency.decomposition_position 仅支持 after_graphgps")
    if frequency_cfg.get("output_version", SPECTRAL_FEATURE_VERSION) != SPECTRAL_FEATURE_VERSION:
        raise ValueError(f"严格模式: frequency.output_version 必须为 {SPECTRAL_FEATURE_VERSION}")
    joint_k, joint_low = int(pos_cfg.get("joint_num_eig", 16)), int(frequency_cfg.get("joint_low_modes", 16))
    if joint_k <= 0 or joint_low <= 0 or joint_low > joint_k:
        raise ValueError("严格模式: joint LapPE/low modes 配置非法")
    sequence_length = int(data_cfg.get("seq_length", 0))
    if sequence_length != 24:
        raise ValueError("严格模式: 第一版仅支持 data.seq_length=24")
    if int(model_cfg.get("output_dim", 48)) != 2 * sequence_length:
        raise ValueError("严格模式: model.output_dim 必须等于 2 * data.seq_length")
    attention = str(attention_cfg.get("global_attn", "linear"))
    if attention not in GlobalAttentionBranch.ALLOWED:
        raise ValueError("严格模式: attention.global_attn 非法")
    if int(attention_cfg.get("full_attention_max_nodes", 4096)) <= 0:
        raise ValueError("严格模式: attention.full_attention_max_nodes 必须为正")
    if str(attention_cfg.get("global_attention_scope", "joint")) not in {"joint", "same_layer"}:
        raise ValueError("严格模式: attention.global_attention_scope 必须为 joint 或 same_layer")


class ThreeLayerGraphGPSLapPE(nn.Module):
    """One shared encoder, followed by joint spectral decomposition and slicing."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        validate_stage2_config(config)
        model_cfg, pos_cfg = _nested(config, "model"), _nested(config, "posenc")
        attention_cfg, frequency_cfg = _nested(config, "attention"), _nested(config, "frequency")
        self.hidden_dim = int(model_cfg.get("hidden_dim", 128))
        self.output_dim = int(model_cfg.get("output_dim", 48))
        self.dropout = float(model_cfg.get("dropout", 0.1))
        self.joint_k = int(pos_cfg.get("joint_num_eig", 16))
        self.joint_low_modes = int(frequency_cfg.get("joint_low_modes", 16))
        self.frequency_output_version = str(frequency_cfg.get("output_version"))
        self.laplacian_norm = str(pos_cfg.get("laplacian_norm", "sym"))
        self.pe_cache_dir = pos_cfg.get("cache_dir") if bool(pos_cfg.get("cache", True)) else None
        pe_dim = int(pos_cfg.get("pe_dim", 16))
        heads, global_attn, full_max = int(attention_cfg.get("num_heads", 4)), str(attention_cfg.get("global_attn", "linear")), int(attention_cfg.get("full_attention_max_nodes", 4096))
        global_scope = str(attention_cfg.get("global_attention_scope", "joint"))
        self.global_attention_scope = global_scope
        self.road_input, self.syntax_input, self.region_input = nn.Linear(33, self.hidden_dim), nn.Linear(5, self.hidden_dim), nn.Linear(45, self.hidden_dim)
        # Explicit per-layer type embeddings make the otherwise identical
        # projected feature spaces distinguishable inside the joint graph.
        self.road_type_embedding = nn.Embedding(1, self.hidden_dim)
        self.syntax_type_embedding = nn.Embedding(1, self.hidden_dim)
        self.region_type_embedding = nn.Embedding(1, self.hidden_dim)
        self.joint_lappe = LapPEEncoder(self.joint_k, pe_dim, encoder=str(pos_cfg.get("encoder", "DeepSet")), pooling="mean", dropout=self.dropout)
        self.pe_projection = nn.Linear(pe_dim, self.hidden_dim)
        depth = int(model_cfg.get(
            "num_layers",
            max(
                int(model_cfg.get("num_layers_road", 2)),
                int(model_cfg.get("num_layers_syntax", 2)),
                int(model_cfg.get("num_layers_region", 2)),
            ),
        ))
        self.graphgps = GraphGPSStack(self.hidden_dim, depth, heads, self.dropout, global_attn, full_attention_max_nodes=full_max, global_attention_scope=global_scope)
        self.num_graphgps_stacks = 1
        self.joint_decoupler = SpectralFeatureDecoupler(orthogonality_tolerance=float(frequency_cfg.get("orthogonality_tolerance", 1e-3)), reconstruction_tolerance=float(frequency_cfg.get("reconstruction_tolerance", 1e-5)))
        self.frequency_fusion = nn.Sequential(nn.Linear(2 * self.hidden_dim, self.hidden_dim), nn.LayerNorm(self.hidden_dim), nn.GELU())
        self.prediction_head = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.GELU(), nn.Dropout(self.dropout), nn.Linear(self.hidden_dim, self.output_dim))

    def prepare_posenc(self, hierarchy: CityStaticHierarchy) -> HierarchyLaplacianPE:
        return prepare_hierarchy_lappe(hierarchy, joint_k=self.joint_k, normalization=self.laplacian_norm, cache_dir=self.pe_cache_dir)

    @staticmethod
    def _validate_pe_nodes(hierarchy: CityStaticHierarchy, pe: HierarchyLaplacianPE) -> None:
        graph = build_joint_three_layer_graph(hierarchy)
        if pe.joint.eigvecs.shape[0] != graph.num_nodes or pe.metadata.get("num_joint_nodes") != graph.num_nodes:
            raise ValueError("严格模式: joint LapPE 节点数与联合图不一致")
        if pe.metadata.get("joint_graph_hash") != graph.joint_graph_hash or pe.joint.metadata.get("joint_graph_hash") != graph.joint_graph_hash:
            raise ValueError("严格模式: joint LapPE graph hash 与联合图不一致")
        for name, expected in (("road_node_range", graph.road_node_range), ("syntax_node_range", graph.syntax_node_range), ("region_node_range", graph.region_node_range)):
            if tuple(getattr(pe, name)) != tuple(expected):
                raise ValueError(f"严格模式: LapPE {name} 与联合图节点编号不一致")

    def forward(self, hierarchy: CityStaticHierarchy, posenc: HierarchyLaplacianPE | None = None, *, return_edge_audit: bool = False) -> dict[str, torch.Tensor]:
        validate_city_static_hierarchy(hierarchy)
        if hierarchy.metadata.get("feature_version") != "three-layer-start-road-v2":
            raise ValueError("严格模式: GraphGPS 仅接受 START v2 three-layer cache")
        if hierarchy.road_x.shape[1] != 33:
            raise ValueError("Missing `road_x` in three-layer graph cache or road_x is not [M,33].")
        if posenc is None:
            posenc = self.prepare_posenc(hierarchy)
        self._validate_pe_nodes(hierarchy, posenc)
        if posenc.joint.eigvecs.shape[1] != self.joint_k:
            raise ValueError("严格模式: joint LapPE k 与模型配置不一致")
        if posenc.metadata.get("pe_version") != LAPPE_VERSION:
            raise ValueError("严格模式: 模型拒绝旧 LapPE version")
        if posenc.metadata.get("weighted_pe") is not True:
            raise ValueError("严格模式: 模型只接受 weighted joint LapPE")
        device = next(self.parameters()).device
        hierarchy, posenc = hierarchy.to(device), posenc.to(device)
        graph = build_joint_three_layer_graph(hierarchy).to(device)
        validate_joint_three_layer_graph(graph)
        m, k, n = hierarchy.num_roads, hierarchy.num_syntax, hierarchy.num_regions
        road_h0, syntax_h0, region_h0 = self.road_input(hierarchy.road_x), self.syntax_input(hierarchy.syntax_x), self.region_input(hierarchy.region_x)
        raw_h = torch.cat([road_h0, syntax_h0, region_h0], dim=0)
        type_h = torch.cat([
            self.road_type_embedding(torch.zeros(m, dtype=torch.long, device=device)),
            self.syntax_type_embedding(torch.zeros(k, dtype=torch.long, device=device)),
            self.region_type_embedding(torch.zeros(n, dtype=torch.long, device=device)),
        ], dim=0)
        x_joint = raw_h + type_h + self.pe_projection(self.joint_lappe(posenc.joint))
        # Exactly one invocation on the complete V=M+K+N node set.
        if self.global_attention_scope == "joint":
            # Preserve the public four-argument GraphGPS call used by older
            # hooks/tests; joint attention does not need node_type.
            h_joint = self.graphgps(x_joint, graph.edge_index_joint, graph.edge_type, graph.edge_weight)
        else:
            h_joint = self.graphgps(x_joint, graph.edge_index_joint, graph.edge_type, graph.edge_weight, node_type=graph.node_type)
        frequency = self.joint_decoupler(h_joint, posenc.joint, self.joint_low_modes)
        low_joint, high_joint = frequency.low, frequency.high
        road_sl, syntax_sl, region_sl = slice(0, m), slice(m, m + k), slice(m + k, m + k + n)
        road_h, syntax_h, region_h = h_joint[road_sl], h_joint[syntax_sl], h_joint[region_sl]
        road_low, syntax_low, region_low = low_joint[road_sl], low_joint[syntax_sl], low_joint[region_sl]
        road_high, syntax_high, region_high = high_joint[road_sl], high_joint[syntax_sl], high_joint[region_sl]
        pooled_road = pool_road_to_syntax(road_h, k, assignment=hierarchy.road_to_syntax_assignment, edge_index=hierarchy.road_to_syntax_edge_index, weight=hierarchy.road_to_syntax_weight, shape=hierarchy.road_to_syntax_shape)
        pooled_syntax = pool_syntax_to_region(syntax_h, edge_index=hierarchy.syntax_to_region_edge_index, weight=hierarchy.syntax_to_region_weight, shape=hierarchy.syntax_to_region_shape)
        prediction = self.prediction_head(self.frequency_fusion(torch.cat([region_low, region_high], dim=-1)))
        result = {"x_joint": x_joint, "H_joint": h_joint, "H_joint_low": low_joint, "H_joint_high": high_joint, "joint_low_coefficients": frequency.coefficients, "H_road": road_h, "H_road_low": road_low, "H_road_high": road_high, "pooled_road_to_syntax": pooled_road, "H_syntax": syntax_h, "H_syntax_low": syntax_low, "H_syntax_high": syntax_high, "pooled_syntax_to_region": pooled_syntax, "H_region": region_h, "H_region_low": region_low, "H_region_high": region_high, "pred": prediction}
        for name, tensor in result.items():
            if isinstance(tensor, torch.Tensor) and not torch.isfinite(tensor).all():
                raise FloatingPointError(f"严格模式: {name} 含 NaN/Inf")
        if return_edge_audit:
            result.update({
                "edge_index_joint_msg": graph.edge_index_joint,
                "edge_index_joint_pe": posenc.joint.edge_index_pe,
                "edge_weight_joint_pe": posenc.joint.edge_weight_pe,
                "edge_type_joint_pe": posenc.joint.edge_type_pe,
                "edge_type_joint": graph.edge_type,
                "edge_weight_joint": graph.edge_weight,
            })
        return result
