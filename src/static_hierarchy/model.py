"""Road→Syntax→Region 三层静态编码器。"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn

from craft_integrated.graph_transformer_pytorch import GraphTransformer
from craft_integrated.pyg_compat import GATv2Conv

from .contracts import CityStaticHierarchy, validate_city_static_hierarchy
from .operators import sparse_operator, sparse_pool


class _FeatureInit(nn.Module):
    """与 CRAFT FeatureInitLayer 同语义的逐图标准化投影。"""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.eps = 1e-5
        self.proj = nn.Sequential(
            nn.Dropout(p=0.05), nn.Linear(self.input_dim, self.output_dim), nn.ReLU(),
            nn.Linear(self.output_dim, self.output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if not value.is_floating_point() or value.ndim != 2 or value.shape[1] != self.input_dim:
            raise ValueError(f"静态编码器输入维度错误: 期望 [V,{self.input_dim}]，实得 {tuple(value.shape)}")
        if not torch.isfinite(value).all():
            raise ValueError("严格模式: 静态编码器输入含 NaN/Inf")
        mean = value.mean(dim=0, keepdim=True)
        variance = value.var(dim=0, unbiased=False, keepdim=True)
        return self.proj((value - mean) / torch.sqrt(variance + self.eps))


def _standardize(value: torch.Tensor) -> torch.Tensor:
    """按城市节点统计做标准化，不跨城市汇聚统计量。"""

    if not value.is_floating_point() or value.ndim != 2 or not torch.isfinite(value).all():
        raise ValueError("严格模式: 静态编码器输入必须是有限二维张量")
    mean = value.mean(dim=0, keepdim=True)
    variance = value.var(dim=0, unbiased=False, keepdim=True)
    return (value - mean) / torch.sqrt(variance + 1e-5)


class _GATStack(nn.Module):
    def __init__(self, dim: int, layers: int, heads: int, dropout: float, *, add_self_loops: bool = True):
        super().__init__()
        if dim < 1 or layers < 1 or heads < 1:
            raise ValueError("GAT 维度、层数和 heads 必须为正")
        if not 0 <= dropout <= 1:
            raise ValueError("GAT dropout 必须在 [0,1]")
        self.layers = nn.ModuleList([
            GATv2Conv(
                dim, dim, heads=heads, concat=False, dropout=dropout,
                add_self_loops=add_self_loops,
            )
            for _ in range(layers)
        ])

    def forward(self, value: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.dtype != torch.long or edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("严格模式: GAT edge_index 必须为 LongTensor[2,E]")
        if edge_index.numel() and (int(edge_index.min()) < 0 or int(edge_index.max()) >= value.shape[0]):
            raise ValueError("严格模式: GAT edge_index 节点索引越界")
        for layer in self.layers:
            value = torch.relu(layer(value, edge_index) + value)
        return value


class RoadTopologyEncoder(nn.Module):
    """Road 纯拓扑编码器：标准化 → Linear(4, rep_dim) → GATv2 残差。"""

    def __init__(self, rep_dim: int, layers: int, heads: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Linear(4, rep_dim)
        self.gnn_layers = _GATStack(rep_dim, layers, heads, dropout)

    def forward(self, road_topo_x: torch.Tensor, road_edge_index: torch.Tensor) -> torch.Tensor:
        if road_topo_x.ndim != 2 or road_topo_x.shape[1] != 4:
            raise ValueError(f"严格模式: road_topo_x 应为 [M,4]，实得 {tuple(road_topo_x.shape)}")
        h = self.input_proj(_standardize(road_topo_x))
        return self.gnn_layers(h, road_edge_index)


class RoadStaticEncoder(nn.Module):
    """START 风格静态 Road 编码器，输入固定 33 维、保持有向边。"""

    input_dim = 33

    def __init__(self, rep_dim: int, layers: int, heads: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Linear(self.input_dim, rep_dim)
        self.input_norm = nn.LayerNorm(rep_dim)
        self.gnn_layers = _GATStack(
            rep_dim, layers, heads, dropout, add_self_loops=True
        )

    def forward(self, road_x: torch.Tensor, road_edge_index: torch.Tensor) -> torch.Tensor:
        if road_x.ndim != 2 or road_x.shape[1] != self.input_dim:
            raise ValueError(f"严格模式: road_x 应为 [M,33]，实得 {tuple(road_x.shape)}")
        if not road_x.is_floating_point() or not torch.isfinite(road_x).all():
            raise ValueError("严格模式: START Road 输入必须为有限浮点 Tensor")
        h = torch.relu(self.input_norm(self.input_proj(road_x)))
        return self.gnn_layers(h, road_edge_index)


class SyntaxEncoder(nn.Module):
    """Syntax 节点语义与 Road 池化表征融合后的 GATv2 编码器。"""

    def __init__(self, rep_dim: int, layers: int, heads: int, dropout: float):
        super().__init__()
        self.syntax_init = _FeatureInit(5, rep_dim)
        self.fusion = nn.Sequential(
            nn.Linear(2 * rep_dim, rep_dim), nn.ReLU(),
            nn.Linear(rep_dim, rep_dim),
        )
        self.gnn_layers = _GATStack(rep_dim, layers, heads, dropout)

    def forward(
        self,
        syntax_x: torch.Tensor,
        road_to_syntax_h: torch.Tensor,
        syntax_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        syntax_h = self.syntax_init(syntax_x)
        if road_to_syntax_h.shape != syntax_h.shape:
            raise ValueError("严格模式: Road→Syntax 聚合结果与 Syntax 表征 shape 不一致")
        syntax_h = self.fusion(torch.cat([syntax_h, road_to_syntax_h], dim=-1))
        return self.gnn_layers(syntax_h, syntax_edge_index)


class ThreeLayerStaticEncoder(nn.Module):
    """共享参数的三层城市静态图编码器。

    唯一的模型消息路径是 ``Road → Syntax → Region``；Road 与 Region
    不直接做可学习聚合。几何只在离线构造 ``Syntax→Region`` 算子时使用。
    """

    def __init__(self, cfg: Mapping[str, Any]):
        super().__init__()
        road_feature_mode = cfg.get("road_feature_mode", "topology_only")
        if road_feature_mode == "cospec":
            raise NotImplementedError("CoSpec road features are not implemented in Stage 1")
        if road_feature_mode not in {"topology_only", "start_static"}:
            raise ValueError(f"未知 road_feature_mode={road_feature_mode!r}")
        self.rep_dim = int(cfg.get("rep_dim", 128))
        if self.rep_dim <= 0:
            raise ValueError("严格模式: rep_dim 必须为正")
        if road_feature_mode == "topology_only" and int(cfg.get("road_topo_feature_dim", 4)) != 4:
            raise ValueError("严格模式: road_topo_feature_dim 必须为 4")
        if road_feature_mode == "start_static" and int(cfg.get("road_feature_dim", 33)) != 33:
            raise ValueError("严格模式: start_static road_feature_dim 必须为 33")
        if int(cfg.get("syntax_feature_dim", 5)) != 5:
            raise ValueError("严格模式: syntax_feature_dim 必须为 5")
        road_encoder_cls = RoadTopologyEncoder if road_feature_mode == "topology_only" else RoadStaticEncoder
        self.road_encoder = road_encoder_cls(
            self.rep_dim, int(cfg.get("road_gat_layers", 4)),
            int(cfg.get("road_gat_heads", 8)), float(cfg.get("road_dropout", 0.1)),
        )
        self.road_feature_mode = road_feature_mode
        self.syntax_encoder = SyntaxEncoder(
            self.rep_dim, int(cfg.get("syntax_gat_layers", 2)),
            int(cfg.get("syntax_gat_heads", 4)), float(cfg.get("syntax_dropout", 0.1)),
        )
        # Region 层必须复用 CRAFT 原 FeatureInitLayer；延迟导入避免 rep_model
        # 在导入 ThreeLayerStaticEncoder 时产生循环依赖。
        try:
            from craft_integrated.rep_model import FeatureInitLayer
        except ModuleNotFoundError:
            from rep_model import FeatureInitLayer
        self.region_init = FeatureInitLayer(raw_feature_dim=45, rep_dim=self.rep_dim)
        self.region_fusion = nn.Sequential(
            nn.Linear(2 * self.rep_dim, self.rep_dim), nn.ReLU(),
            nn.Linear(self.rep_dim, self.rep_dim),
        )
        self.region_gnn = GraphTransformer(
            dim=self.rep_dim, depth=3, heads=4, dim_head=64,
            with_feedforwards=True, rel_pos_emb=False,
            accept_adjacency_matrix=True,
        )

    @property
    def road_init(self) -> nn.Module:
        """兼容早期测试/调用方的 Road 输入投影访问器。"""

        return self.road_encoder.input_proj

    @property
    def syntax_init(self) -> nn.Module:
        return self.syntax_encoder.syntax_init

    @property
    def syntax_fusion(self) -> nn.Module:
        return self.syntax_encoder.fusion

    @staticmethod
    def _operator(hierarchy: CityStaticHierarchy, name: str, device: torch.device) -> torch.Tensor:
        if name == "road_to_syntax":
            return sparse_operator(
                hierarchy.road_to_syntax_edge_index,
                hierarchy.road_to_syntax_weight,
                hierarchy.road_to_syntax_shape,
            ).to(device)
        return sparse_operator(
            hierarchy.syntax_to_region_edge_index,
            hierarchy.syntax_to_region_weight,
            hierarchy.syntax_to_region_shape,
        ).to(device)

    def forward(
        self, hierarchy: CityStaticHierarchy, *, return_intermediates: bool = False
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        validate_city_static_hierarchy(hierarchy)
        device = next(self.parameters()).device
        hierarchy = hierarchy.to(device)
        expected_version = (
            "three-layer-static-v1" if self.road_feature_mode == "topology_only"
            else "three-layer-start-road-v2"
        )
        if hierarchy.metadata.get("feature_version") != expected_version:
            raise ValueError(
                f"严格模式: 模型 road_feature_mode={self.road_feature_mode!r} "
                f"不能加载 {hierarchy.metadata.get('feature_version')!r} cache"
            )
        # Bottom layer: topology-only [M,4] or START static [M,33].
        road_h = self.road_encoder(hierarchy.road_x, hierarchy.road_edge_index)
        road_to_syntax = sparse_pool(self._operator(hierarchy, "road_to_syntax", device), road_h)
        # Middle layer: five static spatial-syntax statistics plus pooled Road representation.
        syntax_h = self.syntax_encoder(
            hierarchy.syntax_x, road_to_syntax, hierarchy.syntax_edge_index
        )
        syntax_to_region = sparse_pool(self._operator(hierarchy, "syntax_to_region", device), syntax_h)
        # Top layer: CRAFT 45-D Region semantic input plus Syntax representation.
        region_semantic_h = self.region_init(hierarchy.region_x)
        region_h = self.region_fusion(torch.cat([region_semantic_h, syntax_to_region], dim=-1))
        nodes = hierarchy.num_regions
        adjacency = torch.zeros((nodes, nodes), dtype=region_h.dtype, device=device)
        if hierarchy.region_edge_index.numel():
            src, dst = hierarchy.region_edge_index
            adjacency[src, dst] = 1.0
        region_rep, _ = self.region_gnn(
            nodes=region_h.unsqueeze(0), adj_mat=adjacency.unsqueeze(0)
        )
        region_rep = region_rep.squeeze(0)
        if not torch.isfinite(region_rep).all():
            raise FloatingPointError("严格模式: 三层静态编码器输出含 NaN/Inf")
        if return_intermediates:
            return {
                "road_h": road_h,
                "road_to_syntax_h": road_to_syntax,
                "syntax_h": syntax_h,
                "syntax_to_region_h": syntax_to_region,
                "region_semantic_h": region_semantic_h,
                "region_rep": region_rep,
            }
        return region_rep
