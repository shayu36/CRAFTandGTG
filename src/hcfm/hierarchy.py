"""稀疏 Region--Road 结构与动态守恒算子。"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np
import torch


def sparse_coo(
    rows: Sequence[int], cols: Sequence[int], values: Sequence[float], shape: tuple[int, int]
) -> torch.Tensor:
    """构造 coalesced float32 COO；允许合法空矩阵。"""

    if len(rows) != len(cols) or len(rows) != len(values):
        raise ValueError("COO rows/cols/values 长度不一致")
    indices = torch.tensor([rows, cols], dtype=torch.long)
    vals = torch.tensor(values, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, vals, shape).coalesce()


def sparse_apply(operator: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """应用 ``operator [N,M]`` 到 ``values [B,M,C,T]``，返回 ``[B,N,C,T]``。"""

    if operator.layout != torch.sparse_coo:
        raise TypeError("operator 必须是 sparse COO")
    if values.ndim != 4 or values.shape[1] != operator.shape[1]:
        raise ValueError(
            f"values 应为 [B,{operator.shape[1]},C,T], 实得 {tuple(values.shape)}"
        )
    b, _, c, t = values.shape
    flat = values.permute(1, 0, 2, 3).reshape(values.shape[1], -1)
    result = torch.sparse.mm(operator.coalesce().to(values.device), flat)
    return result.reshape(operator.shape[0], b, c, t).permute(1, 0, 2, 3)


def sparse_transpose_apply(operator: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """应用 ``operator.T [M,N]`` 到 ``values [B,N,C]``，返回 ``[B,M,C]``。"""

    if values.ndim != 3 or values.shape[1] != operator.shape[0]:
        raise ValueError("values 与 operator Region 维不匹配")
    b, _, c = values.shape
    flat = values.permute(1, 0, 2).reshape(values.shape[1], -1)
    result = torch.sparse.mm(operator.coalesce().transpose(0, 1).to(values.device), flat)
    return result.reshape(operator.shape[1], b, c).permute(1, 0, 2)


def aggregate_micro_to_macro(
    micro_flow: torch.Tensor, b_in: torch.Tensor, b_out: torch.Tensor
) -> torch.Tensor:
    """道路 passage count 聚合为 Region in/out。

    Args:
        micro_flow: ``[B,M,1,T]``，物理计数或与之线性对应的速度。
        b_in/b_out: sparse ``[N,M]``。
    Returns:
        ``[B,N,2,T]``，通道顺序严格为 in/out。
    """

    if tuple(b_in.shape) != tuple(b_out.shape):
        raise ValueError("B_in/B_out shape 不一致")
    in_flow = sparse_apply(b_in, micro_flow)
    out_flow = sparse_apply(b_out, micro_flow)
    return torch.cat([in_flow, out_flow], dim=2)


def build_road_edge_index(source_nodes: Sequence[object], target_nodes: Sequence[object]) -> np.ndarray:
    """按有向 Road 对偶关系构建 ``[2,E]``，不添加隐式反向边。"""

    if len(source_nodes) != len(target_nodes):
        raise ValueError("source_nodes/target_nodes 长度不一致")
    starts: dict[object, list[int]] = defaultdict(list)
    for idx, node in enumerate(source_nodes):
        starts[node].append(idx)
    src, dst = [], []
    for idx, node in enumerate(target_nodes):
        for nxt in starts.get(node, ()):
            if nxt != idx:
                src.append(idx)
                dst.append(nxt)
    return np.asarray([src, dst], dtype=np.int64)


def build_structural_operator(
    intersection_lengths: np.ndarray, *, zero_tolerance: float = 1e-10
) -> torch.Tensor:
    """由真实相交长度 ``[N,M]`` 构造 Region 行归一化 ``P_struct``。

    空 Region 保持空行；负值、NaN/Inf 直接报错。
    """

    lengths = np.asarray(intersection_lengths, dtype=np.float64)
    if lengths.ndim != 2 or not np.isfinite(lengths).all() or (lengths < 0).any():
        raise ValueError("严格模式: intersection_lengths 必须是有限非负二维矩阵")
    row_sum = lengths.sum(axis=1)
    rows, cols = np.nonzero(lengths > zero_tolerance)
    vals = lengths[rows, cols] / row_sum[rows]
    return sparse_coo(rows.tolist(), cols.tolist(), vals.tolist(), lengths.shape)


def build_boundary_operators(
    start_region: Sequence[int], end_region: Sequence[int], num_regions: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """从每条有向道路的起终 Region 构造 ``B_in/B_out``。

    Region id ``-1`` 表示城市外。内部道路 ``start==end`` 不计边界流；方向反转会交换
    in/out。该函数不根据静态道路属性猜测动态 passage count。
    """

    if len(start_region) != len(end_region):
        raise ValueError("start_region/end_region 长度不一致")
    in_rows, in_cols, out_rows, out_cols = [], [], [], []
    for road, (src, dst) in enumerate(zip(start_region, end_region)):
        src, dst = int(src), int(dst)
        if src < -1 or src >= num_regions or dst < -1 or dst >= num_regions:
            raise ValueError(f"严格模式: road {road} Region id 越界: {src}->{dst}")
        if src == dst:
            continue
        if src >= 0:
            out_rows.append(src)
            out_cols.append(road)
        if dst >= 0:
            in_rows.append(dst)
            in_cols.append(road)
    shape = (num_regions, len(start_region))
    b_in = sparse_coo(in_rows, in_cols, [1.0] * len(in_rows), shape)
    b_out = sparse_coo(out_rows, out_cols, [1.0] * len(out_rows), shape)
    return b_in, b_out


def build_boundary_operators_from_sequences(
    region_sequences: Sequence[Sequence[int]], num_regions: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """按每条 Road 沿方向经过的 Region 序列记录全部边界转移。

    例如 ``[-1,0,1,2,-1]`` 在同一 Road 列产生 outside->0、0->1、1->2、
    2->outside 四次边界事件。连续重复 Region 会去重；非连续重入会保留。
    """

    in_rows, in_cols, out_rows, out_cols = [], [], [], []
    for road, raw_sequence in enumerate(region_sequences):
        sequence = []
        for region in raw_sequence:
            region = int(region)
            if region < -1 or region >= num_regions:
                raise ValueError(f"严格模式: road {road} Region id 越界: {region}")
            if not sequence or sequence[-1] != region:
                sequence.append(region)
        for source, target in zip(sequence[:-1], sequence[1:]):
            if source == target:
                continue
            if source >= 0:
                out_rows.append(source); out_cols.append(road)
            if target >= 0:
                in_rows.append(target); in_cols.append(road)
    shape = (num_regions, len(region_sequences))
    return (
        sparse_coo(in_rows, in_cols, [1.0] * len(in_rows), shape),
        sparse_coo(out_rows, out_cols, [1.0] * len(out_rows), shape),
    )


def graph_difference(values: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """有向 Road 图局部差分，``values [B,M,C,T] -> [B,E,C,T]``。"""

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index 必须为 [2,E]")
    src, dst = edge_index.to(values.device)
    return values[:, dst] - values[:, src]
