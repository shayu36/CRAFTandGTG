"""PyG ABI 兼容入口。

正常环境直接导出 PyG；仅当可选 CUDA 扩展因 torch ABI 不匹配而 ImportError/OSError
时，提供本项目实际用到的最小纯 PyTorch Data/GATv2Conv。不会修改系统环境。
"""

from __future__ import annotations

import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F


USING_FALLBACK = False

try:
    from torch_geometric.data import Data as Data  # type: ignore
    from torch_geometric.nn import GATv2Conv as GATv2Conv  # type: ignore
except (ImportError, OSError):
    USING_FALLBACK = True

    class Data:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        @property
        def num_nodes(self):
            return len(self.x)

        def to(self, device):
            for key, value in self.__dict__.items():
                if isinstance(value, torch.Tensor):
                    setattr(self, key, value.to(device))
            return self

    class GATv2Conv(nn.Module):
        """GATv2 公式兼容后备；支持本项目用到的 concat=False/edge_attr。"""

        def __init__(
            self, in_channels, out_channels, heads=1, concat=True, dropout=0.0,
            edge_dim=None, **_kwargs,
        ):
            super().__init__()
            self.out_channels, self.heads = int(out_channels), int(heads)
            self.concat, self.dropout = bool(concat), float(dropout)
            self.lin_l = nn.Linear(in_channels, heads * out_channels, bias=True)
            self.lin_r = nn.Linear(in_channels, heads * out_channels, bias=True)
            self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False) if edge_dim else None
            self.att = nn.Parameter(torch.empty(1, heads, out_channels))
            output_dim = heads * out_channels if concat else out_channels
            self.bias = nn.Parameter(torch.zeros(output_dim))
            nn.init.xavier_uniform_(self.att)

        def forward(self, x, edge_index, edge_attr=None):
            src, dst = edge_index.to(x.device)
            left = self.lin_l(x).view(len(x), self.heads, self.out_channels)
            right = self.lin_r(x).view(len(x), self.heads, self.out_channels)
            pair = left[src] + right[dst]
            if self.lin_edge is not None:
                if edge_attr is None:
                    raise ValueError("GATv2Conv edge_dim 已配置但缺 edge_attr")
                pair = pair + self.lin_edge(edge_attr.to(x)).view(-1, self.heads, self.out_channels)
            logits = (F.leaky_relu(pair, 0.2) * self.att).sum(-1)
            expanded = dst[:, None].expand_as(logits)
            maxima = torch.full(
                (len(x), self.heads), -torch.inf, dtype=x.dtype, device=x.device
            )
            maxima.scatter_reduce_(0, expanded, logits, reduce="amax", include_self=True)
            weights = torch.exp(logits - maxima[dst])
            denominator = torch.zeros_like(maxima)
            denominator.scatter_add_(0, expanded, weights)
            weights = weights / denominator[dst].clamp_min(torch.finfo(x.dtype).eps)
            weights = F.dropout(weights, self.dropout, self.training)
            messages = left[src] * weights.unsqueeze(-1)
            output = torch.zeros(
                (len(x), self.heads, self.out_channels), dtype=x.dtype, device=x.device
            )
            output.index_add_(0, dst, messages)
            output = output.reshape(len(x), -1) if self.concat else output.mean(1)
            return output + self.bias

    # 第一阶段回归测试随后动态读取只读原始 CRAFT，它仍执行
    # ``from torch_geometric.nn import GATv2Conv``。仅在真实 PyG 已失败的进程中安装
    # 最小模块，使该 baseline 对比能够继续；正常 PyG 环境不触发。
    geometric = types.ModuleType("torch_geometric")
    data_module = types.ModuleType("torch_geometric.data")
    nn_module = types.ModuleType("torch_geometric.nn")
    data_module.Data = Data
    nn_module.GATv2Conv = GATv2Conv
    geometric.data, geometric.nn = data_module, nn_module
    sys.modules["torch_geometric"] = geometric
    sys.modules["torch_geometric.data"] = data_module
    sys.modules["torch_geometric.nn"] = nn_module

