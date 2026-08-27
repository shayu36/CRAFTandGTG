"""Region--Road 双向门控残差层次交互。"""

from __future__ import annotations

import torch
import torch.nn as nn

from .hierarchy import sparse_transpose_apply


def pool_roads_to_regions(p_struct: torch.Tensor, road_state: torch.Tensor) -> torch.Tensor:
    """``P_struct [N,M] @ road_state [B,M,D] -> [B,N,D]``。"""

    if road_state.ndim != 3 or road_state.shape[1] != p_struct.shape[1]:
        raise ValueError("road_state 与 P_struct Road 维不一致")
    b, _, dim = road_state.shape
    flat = road_state.permute(1, 0, 2).reshape(road_state.shape[1], -1)
    pooled = torch.sparse.mm(p_struct.coalesce().to(road_state.device), flat)
    return pooled.reshape(p_struct.shape[0], b, dim).permute(1, 0, 2)


class BidirectionalHierarchyLayer(nn.Module):
    """门控 micro->macro，并可配置 macro->micro 反向条件。"""

    def __init__(self, macro_dim: int, road_dim: int, bidirectional: bool = True):
        super().__init__()
        self.bidirectional = bool(bidirectional)
        self.road_to_macro = nn.Linear(road_dim, macro_dim)
        self.gate = nn.Sequential(nn.Linear(2 * macro_dim, macro_dim), nn.Sigmoid())
        self.macro_norm = nn.LayerNorm(macro_dim)
        self.macro_to_road = nn.Linear(macro_dim, road_dim)
        self.road_norm = nn.LayerNorm(road_dim)

    def forward(
        self, macro: torch.Tensor, road: torch.Tensor, p_struct: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """输入/输出为 ``macro [B,N,Dm]``、``road [B,M,Dr]``。"""

        pooled = pool_roads_to_regions(p_struct, road)
        projected = self.road_to_macro(pooled)
        gate = self.gate(torch.cat([macro, projected], dim=-1))
        macro_fused = self.macro_norm(macro + gate * projected)
        if not self.bidirectional:
            return macro_fused, road
        broadcast = sparse_transpose_apply(p_struct, self.macro_to_road(macro_fused))
        road_fused = self.road_norm(road + broadcast)
        return macro_fused, road_fused


class HierarchicalInteraction(nn.Module):
    def __init__(
        self, macro_dim: int, road_dim: int, num_layers: int = 1,
        fusion: str = "gated_residual", bidirectional: bool = True,
    ):
        super().__init__()
        if fusion != "gated_residual":
            raise ValueError(f"不支持 hierarchy.fusion={fusion!r}")
        if num_layers < 1:
            raise ValueError("hierarchy.num_layers 必须 >=1")
        self.layers = nn.ModuleList([
            BidirectionalHierarchyLayer(macro_dim, road_dim, bidirectional)
            for _ in range(num_layers)
        ])

    def forward(
        self, macro: torch.Tensor, road: torch.Tensor, p_struct: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            macro, road = layer(macro, road, p_struct)
        return macro, road

