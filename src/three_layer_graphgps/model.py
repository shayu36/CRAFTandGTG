"""Road→Syntax→Region 三层 GraphGPS + LapPE 编码器。"""

from __future__ import annotations

from typing import Any, Mapping
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from craft_integrated.pyg_compat import GATv2Conv
from static_hierarchy.contracts import CityStaticHierarchy, validate_city_static_hierarchy

from .pooling import pool_road_to_syntax, pool_syntax_to_region
from .posenc import FeatureLapPEInit
from .spectral_lap_pe import HierarchyLaplacianPE, pe_graph_hash, prepare_hierarchy_lappe


class LinearGlobalAttention(nn.Module):
    """正核线性 self-attention，Road 大图默认使用 O(ND²/H) 路径。"""

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("严格模式: hidden_dim 必须能被 attention heads 整除")
        self.heads = int(heads)
        self.head_dim = dim // heads
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
    """GraphGPS global branch；``local`` 表示只保留同层 local MPNN。"""

    ALLOWED = {"none", "local", "linear", "full"}

    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float,
        mode: str,
        *,
        full_attention_max_nodes: int,
        layer_name: str,
    ) -> None:
        super().__init__()
        if mode not in self.ALLOWED:
            raise ValueError(f"严格模式: {layer_name}_global_attn={mode!r} 非法")
        if full_attention_max_nodes <= 0:
            raise ValueError("严格模式: full_attention_max_nodes 必须为正")
        if dim % heads != 0:
            raise ValueError("严格模式: hidden_dim 必须能被 attention heads 整除")
        self.mode = mode
        self.layer_name = layer_name
        self.full_attention_max_nodes = int(full_attention_max_nodes)
        self.full_attention = (
            nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
            if mode == "full"
            else None
        )
        self.linear_attention = (
            LinearGlobalAttention(dim, heads, dropout)
            if mode in {"linear", "full"}
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode in {"none", "local"}:
            return torch.zeros_like(x)
        if self.mode == "linear":
            return self.linear_attention(x)
        if x.shape[0] > self.full_attention_max_nodes:
            warnings.warn(
                f"{self.layer_name} full attention 收到 {x.shape[0]} 个节点，超过阈值 "
                f"{self.full_attention_max_nodes}；自动 fallback 到 linear attention",
                RuntimeWarning,
            )
            return self.linear_attention(x)
        result, _ = self.full_attention(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0), need_weights=False)
        return result.squeeze(0)


class GraphGPSLayer(nn.Module):
    """并行 local GATv2 + global attention，再接 residual FFN。"""

    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float,
        global_attention: str,
        *,
        full_attention_max_nodes: int,
        layer_name: str,
    ) -> None:
        super().__init__()
        self.local_norm = nn.LayerNorm(dim)
        self.global_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)
        # 不在 edge_index 上添加自环；显式 residual 是节点自身信息路径。
        self.local_mpnn = GATv2Conv(
            dim,
            dim,
            heads=heads,
            concat=False,
            dropout=dropout,
            add_self_loops=False,
        )
        self.global_attention = GlobalAttentionBranch(
            dim,
            heads,
            dropout,
            global_attention,
            full_attention_max_nodes=full_attention_max_nodes,
            layer_name=layer_name,
        )
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.dtype != torch.long or edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("严格模式: GraphGPS edge_index 必须为 LongTensor[2,E]")
        if edge_index.numel() and (
            int(edge_index.min()) < 0 or int(edge_index.max()) >= x.shape[0]
        ):
            raise ValueError("严格模式: GraphGPS edge_index 越界")
        local_h = self.local_mpnn(self.local_norm(x), edge_index)
        global_h = self.global_attention(self.global_norm(x))
        x = x + self.dropout(local_h) + self.dropout(global_h)
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        if not torch.isfinite(x).all():
            raise FloatingPointError("严格模式: GraphGPS layer 输出含 NaN/Inf")
        return x


class GraphGPSStack(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dropout: float,
        global_attention: str,
        *,
        full_attention_max_nodes: int,
        layer_name: str,
    ) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("严格模式: GraphGPS depth 必须为正")
        self.layers = nn.ModuleList(
            [
                GraphGPSLayer(
                    dim,
                    heads,
                    dropout,
                    global_attention,
                    full_attention_max_nodes=full_attention_max_nodes,
                    layer_name=layer_name,
                )
                for _ in range(depth)
            ]
        )
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, edge_index)
        return self.output_norm(x)


def _nested(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"严格模式: 配置 {name} 必须为 mapping")
    return value


class ThreeLayerGraphGPSLapPE(nn.Module):
    """共享参数的三层 GraphGPS。

    唯一跨层消息路径是 ``Road → Syntax → Region``。类中不存在 Road→Region
    池化模块或参数。
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        model_cfg = _nested(config, "model")
        pos_cfg = _nested(config, "posenc")
        attention_cfg = _nested(config, "attention")
        hierarchy_cfg = _nested(config, "hierarchy")
        if model_cfg.get("name", "three_layer_graphgps_lappe") != "three_layer_graphgps_lappe":
            raise ValueError("严格模式: model.name 必须为 three_layer_graphgps_lappe")
        if pos_cfg.get("type", "LapPE") != "LapPE":
            raise ValueError("严格模式: 第一版仅支持 posenc.type=LapPE")
        self.hidden_dim = int(model_cfg.get("hidden_dim", 128))
        self.output_dim = int(model_cfg.get("output_dim", 48))
        self.dropout = float(model_cfg.get("dropout", 0.1))
        self.road_k = int(pos_cfg.get("road_num_eig", 16))
        self.syntax_k = int(pos_cfg.get("syntax_num_eig", 16))
        self.region_k = int(pos_cfg.get("region_num_eig", 16))
        pe_dim = int(pos_cfg.get("pe_dim", 16))
        pe_encoder = str(pos_cfg.get("encoder", "DeepSet"))
        self.laplacian_norm = str(pos_cfg.get("laplacian_norm", "sym"))
        self.pe_cache_dir = pos_cfg.get("cache_dir") if bool(pos_cfg.get("cache", True)) else None
        heads = int(attention_cfg.get("num_heads", 4))
        road_attention = str(attention_cfg.get("road_global_attn", "linear"))
        syntax_attention = str(attention_cfg.get("syntax_global_attn", "full"))
        region_attention = str(attention_cfg.get("region_global_attn", "full"))
        road_full_max = int(attention_cfg.get("road_full_attn_max_nodes", 4096))
        if self.hidden_dim <= 0 or self.output_dim <= 0:
            raise ValueError("严格模式: hidden_dim/output_dim 必须为正")
        if not 0 <= self.dropout <= 1:
            raise ValueError("严格模式: dropout 必须在 [0,1]")
        self.road_to_syntax_pool = str(hierarchy_cfg.get("road_to_syntax_pool", "mean"))
        self.syntax_to_region_pool = str(
            hierarchy_cfg.get("syntax_to_region_pool", "weighted_mean")
        )

        init_kwargs = {
            "hidden_dim": self.hidden_dim,
            "pe_dim": pe_dim,
            "encoder": pe_encoder,
            "dropout": self.dropout,
        }
        self.road_input = FeatureLapPEInit(33, num_eigenvectors=self.road_k, **init_kwargs)
        self.syntax_input = FeatureLapPEInit(5, num_eigenvectors=self.syntax_k, **init_kwargs)
        self.region_input = FeatureLapPEInit(45, num_eigenvectors=self.region_k, **init_kwargs)
        self.syntax_hierarchy_fusion = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        self.region_hierarchy_fusion = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        stack_common = {"dim": self.hidden_dim, "heads": heads, "dropout": self.dropout}
        self.road_graphgps = GraphGPSStack(
            depth=int(model_cfg.get("num_layers_road", 2)),
            global_attention=road_attention,
            full_attention_max_nodes=road_full_max,
            layer_name="road",
            **stack_common,
        )
        self.syntax_graphgps = GraphGPSStack(
            depth=int(model_cfg.get("num_layers_syntax", 2)),
            global_attention=syntax_attention,
            full_attention_max_nodes=1_000_000,
            layer_name="syntax",
            **stack_common,
        )
        self.region_graphgps = GraphGPSStack(
            depth=int(model_cfg.get("num_layers_region", 2)),
            global_attention=region_attention,
            full_attention_max_nodes=1_000_000,
            layer_name="region",
            **stack_common,
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    def prepare_posenc(self, hierarchy: CityStaticHierarchy) -> HierarchyLaplacianPE:
        return prepare_hierarchy_lappe(
            hierarchy,
            road_k=self.road_k,
            syntax_k=self.syntax_k,
            region_k=self.region_k,
            normalization=self.laplacian_norm,
            cache_dir=self.pe_cache_dir,
        )

    @staticmethod
    def _validate_pe_nodes(hierarchy: CityStaticHierarchy, pe: HierarchyLaplacianPE) -> None:
        expected = {
            "road": (hierarchy.num_roads, hierarchy.road_edge_index),
            "syntax": (hierarchy.num_syntax, hierarchy.syntax_edge_index),
            "region": (hierarchy.num_regions, hierarchy.region_edge_index),
        }
        for name, (num_nodes, edge_index) in expected.items():
            layer = getattr(pe, name)
            if layer.eigvecs.shape[0] != num_nodes or layer.eigvals.shape[0] != num_nodes:
                raise ValueError(f"严格模式: {name} LapPE 节点数与三层图不一致")
            if layer.metadata.get("graph_hash") != pe_graph_hash(edge_index, num_nodes):
                raise ValueError(f"严格模式: {name} LapPE graph hash 与三层图不一致")

    def forward(
        self,
        hierarchy: CityStaticHierarchy,
        posenc: HierarchyLaplacianPE | None = None,
        *,
        return_edge_audit: bool = False,
    ) -> dict[str, torch.Tensor]:
        validate_city_static_hierarchy(hierarchy)
        if hierarchy.metadata.get("feature_version") != "three-layer-start-road-v2":
            raise ValueError("严格模式: GraphGPS 仅接受 START v2 three-layer cache")
        if hierarchy.road_x.shape[1] != 33:
            raise ValueError("Missing `road_x` in three-layer graph cache or road_x is not [M,33].")
        if posenc is None:
            posenc = self.prepare_posenc(hierarchy)
        self._validate_pe_nodes(hierarchy, posenc)
        device = next(self.parameters()).device
        hierarchy = hierarchy.to(device)
        posenc = posenc.to(device)

        road_h0 = self.road_input(hierarchy.road_x, posenc.road)
        road_h = self.road_graphgps(road_h0, hierarchy.road_edge_index)
        pooled_road = pool_road_to_syntax(
            road_h,
            hierarchy.num_syntax,
            assignment=hierarchy.road_to_syntax_assignment,
            edge_index=hierarchy.road_to_syntax_edge_index,
            weight=hierarchy.road_to_syntax_weight,
            shape=hierarchy.road_to_syntax_shape,
            mode=self.road_to_syntax_pool,
        )

        syntax_semantic = self.syntax_input(hierarchy.syntax_x, posenc.syntax)
        syntax_h0 = self.syntax_hierarchy_fusion(torch.cat([syntax_semantic, pooled_road], dim=-1))
        syntax_h = self.syntax_graphgps(syntax_h0, hierarchy.syntax_edge_index)
        pooled_syntax = pool_syntax_to_region(
            syntax_h,
            edge_index=hierarchy.syntax_to_region_edge_index,
            weight=hierarchy.syntax_to_region_weight,
            shape=hierarchy.syntax_to_region_shape,
            mode=self.syntax_to_region_pool,
        )

        region_semantic = self.region_input(hierarchy.region_x, posenc.region)
        region_h0 = self.region_hierarchy_fusion(torch.cat([region_semantic, pooled_syntax], dim=-1))
        region_h = self.region_graphgps(region_h0, hierarchy.region_edge_index)
        prediction = self.prediction_head(region_h)
        for name, tensor in (
            ("H_road", road_h),
            ("pooled_road_to_syntax", pooled_road),
            ("H_syntax", syntax_h),
            ("pooled_syntax_to_region", pooled_syntax),
            ("H_region", region_h),
            ("pred", prediction),
        ):
            if not torch.isfinite(tensor).all():
                raise FloatingPointError(f"严格模式: {name} 含 NaN/Inf")
        result = {
            "H_road": road_h,
            "pooled_road_to_syntax": pooled_road,
            "H_syntax": syntax_h,
            "pooled_syntax_to_region": pooled_syntax,
            "H_region": region_h,
            "pred": prediction,
        }
        if return_edge_audit:
            result["road_edge_index_msg"] = hierarchy.road_edge_index
            result["road_edge_index_pe"] = posenc.road.edge_index_pe
        return result
