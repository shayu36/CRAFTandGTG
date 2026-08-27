"""三层静态图的稀疏跨层算子。"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch


def coalesce_edges(
    src: Iterable[int], dst: Iterable[int], weight: Iterable[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按 ``(src,dst)`` 聚合重复有向边，保持首次出现的稳定排序。"""

    src, dst, weight = list(src), list(dst), list(weight)
    if not (len(src) == len(dst) == len(weight)):
        raise ValueError("严格模式: 图边 src/dst/weight 长度不一致")
    result: dict[tuple[int, int], float] = {}
    order: list[tuple[int, int]] = []
    for left, right, value in zip(src, dst, weight):
        value = float(value)
        if not np.isfinite(value) or value < 0:
            raise ValueError("严格模式: 图边权重必须为有限非负值")
        key = (int(left), int(right))
        if key not in result:
            order.append(key)
            result[key] = 0.0
        result[key] += value
    if not order:
        return (
            np.empty((2, 0), dtype=np.int64),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    edge = np.asarray(order, dtype=np.int64).T
    values = np.asarray([result[key] for key in order], dtype=np.float32)
    return edge, values, np.asarray(order, dtype=np.int64)


def sparse_operator(
    edge_index: torch.Tensor, weight: torch.Tensor, shape: tuple[int, int]
) -> torch.Tensor:
    """构造 row=上层、column=下层的稀疏 COO 算子。"""

    if edge_index.dtype != torch.long or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("严格模式: 稀疏算子 edge_index 必须为 LongTensor[2,E]")
    if weight.ndim != 1 or weight.shape[0] != edge_index.shape[1]:
        raise ValueError("严格模式: 稀疏算子 weight shape 错误")
    if not torch.isfinite(weight).all() or (weight < 0).any():
        raise ValueError("严格模式: 稀疏算子权重必须为有限非负值")
    if edge_index.numel() and (
        int(edge_index[0].min()) < 0 or int(edge_index[0].max()) >= shape[0]
        or int(edge_index[1].min()) < 0 or int(edge_index[1].max()) >= shape[1]
    ):
        raise ValueError("严格模式: 稀疏算子索引越界")
    return torch.sparse_coo_tensor(edge_index, weight.float(), shape).coalesce()


def sparse_pool(operator: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """将 ``values [M,D]`` 按 ``operator [K,M]`` 池化为 ``[K,D]``。"""

    if operator.layout != torch.sparse_coo:
        raise TypeError("严格模式: operator 必须是 sparse COO")
    if values.ndim != 2 or values.shape[0] != operator.shape[1]:
        raise ValueError("严格模式: sparse_pool 输入与 operator 下层维度不一致")
    return torch.sparse.mm(operator.coalesce().to(values.device), values)


def weighted_region_projection(
    road_assignment: np.ndarray,
    road_intersection: np.ndarray,
    num_syntax: int,
    num_regions: int,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int], torch.Tensor]:
    """从 Road 所属 Syntax 和 Road×Region 相交长度生成 Syntax→Region。

    ``road_intersection`` 为 ``[M,N]``，输出算子为 ``[N,K]``，每个非空
    Region 行按相交长度归一化。
    """

    assignment = np.asarray(road_assignment, dtype=np.int64)
    lengths = np.asarray(road_intersection, dtype=np.float64)
    if num_syntax <= 0 or num_regions <= 0:
        raise ValueError("严格模式: Syntax/Region 节点数必须为正")
    if assignment.shape != (lengths.shape[0],):
        raise ValueError("严格模式: Road assignment 与相交矩阵行数不一致")
    if lengths.ndim != 2 or lengths.shape[1] != num_regions or not np.isfinite(lengths).all() or (lengths < 0).any():
        raise ValueError("严格模式: Road×Region 相交长度必须为有限非负 [M,N]")
    if assignment.size and (assignment.min() < 0 or assignment.max() >= num_syntax):
        raise ValueError("严格模式: Road→Syntax assignment 越界")
    raw = np.zeros((num_regions, num_syntax), dtype=np.float64)
    for road, syntax in enumerate(assignment.tolist()):
        raw[:, syntax] += lengths[road]
    row_sum = raw.sum(axis=1)
    # 仅零长度才表示没有几何覆盖；不以任意阈值静默丢弃合法的小相交长度。
    rows, cols = np.nonzero(raw > 0)
    values = raw[rows, cols] / row_sum[rows]
    edge = torch.from_numpy(np.asarray([rows, cols], dtype=np.int64))
    weight = torch.from_numpy(values.astype(np.float32))
    mask = torch.from_numpy(row_sum > 0)
    return edge, weight, (num_regions, num_syntax), mask
