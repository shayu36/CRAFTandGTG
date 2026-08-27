"""连续时间 HCFM 概率路径、耦合速度场与 ODE 求解器。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hierarchy import aggregate_micro_to_macro


def continuous_time_embedding(time: torch.Tensor, dim: int, theta: float = 10000.0) -> torch.Tensor:
    """``time [B]`` 的连续 sinusoidal embedding，支持精确 t=0/1。"""

    if time.ndim != 1:
        raise ValueError("continuous time 必须为 [B]")
    half = dim // 2
    if half == 0:
        return time[:, None]
    frequencies = torch.exp(
        -math.log(theta) * torch.arange(half, device=time.device, dtype=time.dtype)
        / max(half - 1, 1)
    )
    angles = time[:, None] * frequencies[None]
    embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
    if embedding.shape[-1] < dim:
        embedding = F.pad(embedding, (0, dim - embedding.shape[-1]))
    return embedding


def expand_time(time: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    """把 ``t [B]`` 展开到 ``[B,1,1,1]``，验证 batch 一致。"""

    if time.ndim != 1 or time.shape[0] != state.shape[0]:
        raise ValueError("t 必须为与 state batch 对齐的 [B]")
    return time.view(state.shape[0], *([1] * (state.ndim - 1)))


def straight_path(
    initial: torch.Tensor, target: torch.Tensor, time: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """条件直线路径 ``Xt=(1-t)X0+tX1`` 与常速度 ``Ut=X1-X0``。"""

    if initial.shape != target.shape:
        raise ValueError("X0/X1 shape 不一致")
    t = expand_time(time.to(initial), initial)
    return (1.0 - t) * initial + t * target, target - initial


def estimate_endpoint(state: torch.Tensor, velocity: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    """随机时间点的一阶终点估计 ``X1_hat=Xt+(1-t)V``。"""

    if state.shape != velocity.shape:
        raise ValueError("state/velocity shape 不一致")
    return state + (1.0 - expand_time(time.to(state), state)) * velocity


def _broadcast_node_mask(mask: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if mask.ndim != 2 or tuple(mask.shape) != tuple(value.shape[:2]):
        raise ValueError(f"mask {tuple(mask.shape)} 与 value {tuple(value.shape)} 节点维不一致")
    return mask.view(mask.shape[0], mask.shape[1], *([1] * (value.ndim - 2)))


def masked_mean(value: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    mask = _broadcast_node_mask(node_mask, value).expand_as(value)
    count = mask.sum()
    if count == 0:
        raise ValueError("严格模式: mask 没有有效元素")
    return value.masked_select(mask).mean()


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("masked_mse prediction/target shape 不一致")
    return masked_mean((prediction - target) ** 2, node_mask)


class TemporalResidualBlock(nn.Module):
    """CRAFT 1D U-Net 同类的时间卷积残差块，条件/连续时间采用 FiLM。"""

    def __init__(self, hidden_dim: int, cond_dim: int, time_dim: int, dropout: float):
        super().__init__()
        groups = min(8, hidden_dim)
        while hidden_dim % groups:
            groups -= 1
        self.norm1 = nn.GroupNorm(groups, hidden_dim)
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)
        self.condition = nn.Linear(cond_dim + time_dim, 2 * hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, condition: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.condition(torch.cat([condition, time_emb], dim=-1)).chunk(2, dim=-1)
        hidden = self.conv1(F.silu(self.norm1(x)))
        hidden = hidden * (1.0 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        hidden = self.conv2(self.dropout(F.silu(self.norm2(hidden))))
        return x + hidden


class GraphTemporalMessage(nn.Module):
    """在时间网络中间做有向 Region/Road 图消息传递。"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.self_projection = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.neighbor_projection = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """``hidden [B,V,H,T]``，沿有向边 src->dst 聚合均值。"""

        if hidden.ndim != 4 or edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("GraphTemporalMessage shape 错误")
        src, dst = edge_index.to(hidden.device)
        if src.numel() == 0:
            return hidden
        neighbor = torch.zeros_like(hidden)
        neighbor.index_add_(1, dst, hidden[:, src])
        degree = torch.zeros(hidden.shape[1], dtype=hidden.dtype, device=hidden.device)
        degree.index_add_(0, dst, torch.ones_like(dst, dtype=hidden.dtype))
        neighbor = neighbor / degree.clamp_min(1.0).view(1, -1, 1, 1)
        b, v, h, t = hidden.shape
        own = self.self_projection(hidden.reshape(b * v, h, t))
        msg = self.neighbor_projection(neighbor.reshape(b * v, h, t))
        mixed = (own + msg).reshape(b, v, h, t)
        mixed = self.norm(mixed.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        return hidden + F.silu(mixed)


class GraphTemporalVectorField(nn.Module):
    """整城图时序速度场。

    Args at forward:
        state: ``[B,V,C,T]``；condition: ``[B,V,D]``；coupled_state:
        ``[B,V,Cc,T]``。输出与 state 同形。宏/微分支分别实例化。
    """

    def __init__(
        self,
        state_channels: int,
        condition_dim: int,
        coupling_channels: int,
        hidden_dim: int = 64,
        num_blocks: int = 4,
        time_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_blocks < 2:
            raise ValueError("速度场至少需要两个 temporal residual blocks")
        self.state_channels = state_channels
        self.coupling_channels = coupling_channels
        self.time_dim = time_dim
        self.input = nn.Conv1d(state_channels + coupling_channels, hidden_dim, 5, padding=2)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, 4 * time_dim), nn.SiLU(), nn.Linear(4 * time_dim, time_dim)
        )
        self.blocks = nn.ModuleList([
            TemporalResidualBlock(hidden_dim, condition_dim, time_dim, dropout)
            for _ in range(num_blocks)
        ])
        self.graph_message = GraphTemporalMessage(hidden_dim)
        self.output = nn.Sequential(
            nn.GroupNorm(min(8, hidden_dim), hidden_dim), nn.SiLU(),
            nn.Conv1d(hidden_dim, state_channels, 3, padding=1),
        )

    def forward(
        self,
        state: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor,
        edge_index: torch.Tensor,
        coupled_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if state.ndim != 4 or condition.shape[:2] != state.shape[:2]:
            raise ValueError("速度场期望 state[B,V,C,T], condition[B,V,D]")
        if state.shape[2] != self.state_channels:
            raise ValueError("state channel 与速度场配置不一致")
        if self.coupling_channels:
            if coupled_state is None or coupled_state.shape[:2] != state.shape[:2]:
                raise ValueError("耦合速度场缺少节点对齐的 coupled_state")
            if coupled_state.shape[2] != self.coupling_channels or coupled_state.shape[-1] != state.shape[-1]:
                raise ValueError("coupled_state channel/T 不一致")
            network_input = torch.cat([state, coupled_state], dim=2)
        else:
            network_input = state
        b, nodes, _, length = network_input.shape
        hidden = self.input(network_input.reshape(b * nodes, -1, length))
        cond = condition.reshape(b * nodes, -1)
        time_emb = self.time_mlp(continuous_time_embedding(time.to(state), self.time_dim))
        time_emb = time_emb[:, None].expand(-1, nodes, -1).reshape(b * nodes, -1)
        middle = len(self.blocks) // 2
        for index, block in enumerate(self.blocks):
            hidden = block(hidden, cond, time_emb)
            if index == middle - 1:
                hidden = self.graph_message(hidden.reshape(b, nodes, -1, length), edge_index)
                hidden = hidden.reshape(b * nodes, -1, length)
        return self.output(hidden).reshape(b, nodes, self.state_channels, length)


def sample_priors(
    micro_target: torch.Tensor,
    macro_target: torch.Tensor,
    b_in: torch.Tensor,
    b_out: torch.Tensor,
    prior_mode: Literal["independent", "coupled"],
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """采样 X0；coupled 模式严格令 ``X0_macro=S(X0_micro)``。"""

    micro = torch.randn(
        micro_target.shape, dtype=micro_target.dtype, device=micro_target.device, generator=generator
    )
    if prior_mode == "independent":
        macro = torch.randn(
            macro_target.shape, dtype=macro_target.dtype, device=macro_target.device, generator=generator
        )
    elif prior_mode == "coupled":
        macro = aggregate_micro_to_macro(micro, b_in, b_out)
        if macro.shape != macro_target.shape:
            raise ValueError("coupled prior 聚合 shape 与 macro target 不一致")
    else:
        raise ValueError(f"未知 prior_mode={prior_mode!r}")
    return macro, micro


def validate_velocity_consistency(prior_mode: str, weight: float) -> None:
    if prior_mode == "independent" and weight > 0:
        raise ValueError(
            "flow_matching.prior_mode=independent 时禁止 velocity_consistency_weight > 0；"
            "独立 X0 不满足可聚合速度路径"
        )


@dataclass
class ODESolverStats:
    solver: str
    steps: int
    nfe: int
    output_min: float
    output_max: float
    has_nonfinite: bool


def integrate_coupled_ode(
    vector_field: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    macro_initial: torch.Tensor,
    micro_initial: torch.Tensor,
    *,
    steps: int = 16,
    solver: Literal["euler", "heun"] = "euler",
) -> tuple[torch.Tensor, torch.Tensor, ODESolverStats]:
    """从 t=0 联合积分到 1；宏/微每次 NFE 使用同一个 t。"""

    if steps <= 0 or solver not in {"euler", "heun"}:
        raise ValueError("solver 必须为 euler/heun 且 steps>0")
    macro, micro = macro_initial, micro_initial
    dt, nfe = 1.0 / steps, 0
    batch = macro.shape[0]
    for step in range(steps):
        time = torch.full((batch,), step * dt, dtype=macro.dtype, device=macro.device)
        vm, vr = vector_field(macro, micro, time)
        nfe += 1
        if solver == "euler":
            macro, micro = macro + dt * vm, micro + dt * vr
        else:
            macro_e, micro_e = macro + dt * vm, micro + dt * vr
            next_time = torch.full((batch,), (step + 1) * dt, dtype=macro.dtype, device=macro.device)
            vm2, vr2 = vector_field(macro_e, micro_e, next_time)
            nfe += 1
            macro = macro + 0.5 * dt * (vm + vm2)
            micro = micro + 0.5 * dt * (vr + vr2)
        if not torch.isfinite(macro).all() or not torch.isfinite(micro).all():
            raise FloatingPointError(f"{solver} ODE 在 step={step} 产生 NaN/Inf")
    nonempty = [value for value in (macro, micro) if value.numel()]
    combined_min = min(float(value.min()) for value in nonempty)
    combined_max = max(float(value.max()) for value in nonempty)
    stats = ODESolverStats(
        solver=solver, steps=steps, nfe=nfe, output_min=combined_min,
        output_max=combined_max, has_nonfinite=False,
    )
    return macro, micro, stats
