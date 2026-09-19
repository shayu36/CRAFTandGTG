"""CRAFT-style conditional one-dimensional U-Net.

This is an engineering adaptation of ``/root/autodl-tmp/projects/CRAFT/unet.py``.
The original temporal ResNet/attention layout and self-conditioning path are
kept, while configuration is explicit and nodes are always folded into the
batch dimension.  Attention therefore operates over the short temporal axis,
never over Road/Syntax/Region nodes.
"""

from __future__ import annotations

import math
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int, theta: float = 10_000.0):
        super().__init__()
        if dim < 4 or dim % 2:
            raise ValueError("SinusoidalPosEmb dim 必须是大于等于 4 的偶数")
        self.dim = int(dim)
        self.theta = float(theta)

    def forward(self, step: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(self.theta) / (half - 1)
        frequencies = torch.exp(
            torch.arange(half, device=step.device, dtype=torch.float32) * -scale
        )
        angles = step.float().unsqueeze(1) * frequencies.unsqueeze(0)
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.normalize(value, dim=1) * self.scale * (value.shape[1] ** 0.5)


class Residual(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.module(value)


class PreNorm(nn.Module):
    def __init__(self, dim: int, module: nn.Module):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.module = module

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.module(self.norm(value))


class LinearAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.heads = int(heads)
        self.dim_head = int(dim_head)
        self.scale = dim_head ** -0.5
        hidden = heads * dim_head
        self.to_qkv = nn.Conv1d(dim, hidden * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv1d(hidden, dim, 1), RMSNorm(dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, _, length = value.shape
        q, k, v = self.to_qkv(value).chunk(3, dim=1)
        reshape = lambda x: x.reshape(batch, self.heads, self.dim_head, length)
        q, k, v = reshape(q), reshape(k), reshape(v)
        q = q.softmax(dim=-2) * self.scale
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        output = torch.einsum("bhde,bhdn->bhen", context, q)
        return self.to_out(output.reshape(batch, self.heads * self.dim_head, length))


class TemporalAttention(nn.Module):
    """Full attention over T only; node count is part of the batch axis."""

    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.heads = int(heads)
        self.dim_head = int(dim_head)
        self.scale = dim_head ** -0.5
        hidden = heads * dim_head
        self.to_qkv = nn.Conv1d(dim, hidden * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden, dim, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, _, length = value.shape
        q, k, v = self.to_qkv(value).chunk(3, dim=1)
        reshape = lambda x: x.reshape(batch, self.heads, self.dim_head, length)
        q, k, v = reshape(q), reshape(k), reshape(v)
        similarity = torch.einsum("bhdi,bhdj->bhij", q * self.scale, k)
        attention = similarity.softmax(dim=-1)
        output = torch.einsum("bhij,bhdj->bhdi", attention, v)
        return self.to_out(output.reshape(batch, self.heads * self.dim_head, length))


class Block(nn.Module):
    def __init__(self, dim: int, dim_out: int, dropout: float = 0.0):
        super().__init__()
        self.projection = nn.Conv1d(dim, dim_out, 3, padding=1)
        self.norm = RMSNorm(dim_out)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, value: torch.Tensor, scale_shift: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> torch.Tensor:
        value = self.norm(self.projection(value))
        if scale_shift is not None:
            scale, shift = scale_shift
            value = value * (scale + 1.0) + shift
        return self.dropout(self.activation(value))


class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        *,
        time_dim: int,
        cond_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(time_dim + cond_dim, dim_out * 2)
        )
        self.block1 = Block(dim, dim_out, dropout)
        self.block2 = Block(dim_out, dim_out)
        self.residual = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(
        self, value: torch.Tensor, time_embedding: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        embedding = torch.cat((time_embedding, condition), dim=-1)
        scale_shift = self.modulation(embedding).unsqueeze(-1).chunk(2, dim=1)
        hidden = self.block1(value, scale_shift)
        hidden = self.block2(hidden)
        return hidden + self.residual(value)


def _downsample(dim: int, dim_out: int) -> nn.Module:
    return nn.Conv1d(dim, dim_out, 4, 2, 1)


def _upsample(dim: int, dim_out: int) -> nn.Module:
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv1d(dim, dim_out, 3, padding=1),
    )


class ConditionalUnet1D(nn.Module):
    """Conditional epsilon estimator for tensors ``[B*N,C,T]``."""

    def __init__(
        self,
        *,
        data_channels: int,
        cond_dim: int,
        init_dim: int = 64,
        base_dim: int = 32,
        dim_mults: tuple[int, ...] = (1, 2, 4, 8),
        use_self_cond: bool = True,
        sinusoidal_theta: float = 10_000.0,
        dropout: float = 0.1,
        attention_dim_head: int = 64,
        attention_heads: int = 4,
    ):
        super().__init__()
        if min(data_channels, cond_dim, init_dim, base_dim) <= 0 or not dim_mults:
            raise ValueError("ConditionalUnet1D 维度必须为正")
        self.data_channels = int(data_channels)
        self.cond_dim = int(cond_dim)
        self.use_self_cond = bool(use_self_cond)
        input_channels = data_channels * (2 if use_self_cond else 1)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *(base_dim * int(multiplier) for multiplier in dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        time_dim = base_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_dim, theta=sinusoidal_theta),
            nn.Linear(base_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim), nn.GELU(), nn.Linear(cond_dim, cond_dim)
        )
        block = partial(
            ResnetBlock, time_dim=time_dim, cond_dim=cond_dim, dropout=dropout
        )
        self.downs = nn.ModuleList()
        for index, (dim_in, dim_out) in enumerate(in_out):
            last = index == len(in_out) - 1
            self.downs.append(nn.ModuleList([
                block(dim_in, dim_in),
                block(dim_in, dim_in),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                nn.Conv1d(dim_in, dim_out, 3, padding=1)
                if last else _downsample(dim_in, dim_out),
            ]))

        middle = dims[-1]
        self.mid_block1 = block(middle, middle)
        self.mid_attention = Residual(PreNorm(
            middle,
            TemporalAttention(middle, heads=attention_heads, dim_head=attention_dim_head),
        ))
        self.mid_block2 = block(middle, middle)

        self.ups = nn.ModuleList()
        for index, (dim_in, dim_out) in enumerate(reversed(in_out)):
            last = index == len(in_out) - 1
            self.ups.append(nn.ModuleList([
                block(dim_out + dim_in, dim_out),
                block(dim_out + dim_in, dim_out),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                nn.Conv1d(dim_out, dim_in, 3, padding=1)
                if last else _upsample(dim_out, dim_in),
            ]))

        self.final_block = block(init_dim * 2, init_dim)
        self.final_conv = nn.Conv1d(init_dim, data_channels, 1)

    @property
    def temporal_downsample_factor(self) -> int:
        return 2 ** max(len(self.downs) - 1, 0)

    def forward(
        self,
        value: torch.Tensor,
        step: torch.Tensor,
        condition: torch.Tensor,
        x_self_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if value.ndim != 3 or value.shape[1] != self.data_channels:
            raise ValueError(
                f"ConditionalUnet1D 期望 [B,{self.data_channels},T]，实得 {tuple(value.shape)}"
            )
        if value.shape[-1] % self.temporal_downsample_factor:
            raise ValueError(
                f"时间长度 {value.shape[-1]} 必须能被 {self.temporal_downsample_factor} 整除"
            )
        if condition.shape != (value.shape[0], self.cond_dim):
            raise ValueError("ConditionalUnet1D condition shape 不匹配")
        if step.shape != (value.shape[0],):
            raise ValueError("ConditionalUnet1D timestep shape 不匹配")
        if self.use_self_cond:
            if x_self_cond is None:
                x_self_cond = torch.zeros_like(value)
            if x_self_cond.shape != value.shape:
                raise ValueError("ConditionalUnet1D self condition shape 不匹配")
            value = torch.cat((x_self_cond, value), dim=1)
        elif x_self_cond is not None:
            raise ValueError("use_self_cond=false 时不接受 x_self_cond")

        condition = self.cond_mlp(condition)
        value = self.init_conv(value)
        residual = value.clone()
        time_embedding = self.time_mlp(step)
        skips: list[torch.Tensor] = []
        for block1, block2, attention, downsample in self.downs:
            value = block1(value, time_embedding, condition)
            skips.append(value)
            value = attention(block2(value, time_embedding, condition))
            skips.append(value)
            value = downsample(value)
        value = self.mid_block1(value, time_embedding, condition)
        value = self.mid_attention(value)
        value = self.mid_block2(value, time_embedding, condition)
        for block1, block2, attention, upsample in self.ups:
            value = block1(torch.cat((value, skips.pop()), dim=1), time_embedding, condition)
            value = block2(torch.cat((value, skips.pop()), dim=1), time_embedding, condition)
            value = attention(value)
            value = upsample(value)
        value = self.final_block(
            torch.cat((value, residual), dim=1), time_embedding, condition
        )
        return self.final_conv(value)


# CRAFT-compatible public name used in configuration discussions.
Unet1D = ConditionalUnet1D
