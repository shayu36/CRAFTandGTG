"""Road→Syntax→Region 的严格稀疏层级池化。"""

from __future__ import annotations

import torch

from static_hierarchy.operators import sparse_operator, sparse_pool


def pool_road_to_syntax(
    road_h: torch.Tensor,
    num_syntax: int,
    *,
    assignment: torch.Tensor | None = None,
    edge_index: torch.Tensor | None = None,
    weight: torch.Tensor | None = None,
    shape: tuple[int, int] | None = None,
    mode: str = "mean",
) -> torch.Tensor:
    """将 ``road_h[M,D]`` 均值池化为 ``[K,D]``。

    优先使用第一阶段已缓存的稀疏算子；也支持只有 assignment 的数据对象。
    """

    if road_h.ndim != 2 or road_h.shape[0] <= 0 or not torch.isfinite(road_h).all():
        raise ValueError("严格模式: road_h 必须为有限非空 [M,D]")
    if mode != "mean":
        raise ValueError("严格模式: 第一版 Road→Syntax 仅支持 mean pooling")
    num_syntax = int(num_syntax)
    if num_syntax <= 0:
        raise ValueError("严格模式: num_syntax 必须为正")
    if edge_index is not None or weight is not None or shape is not None:
        if edge_index is None or weight is None or shape is None:
            raise ValueError("严格模式: Road→Syntax sparse edge/weight/shape 必须同时提供")
        if tuple(shape) != (num_syntax, road_h.shape[0]):
            raise ValueError("严格模式: Road→Syntax sparse shape 错误")
        if assignment is not None:
            if assignment.shape != (road_h.shape[0],) or assignment.dtype != torch.long:
                raise ValueError("严格模式: Road→Syntax assignment shape/dtype 错误")
            expected_columns = torch.arange(road_h.shape[0], device=edge_index.device)
            if not torch.equal(edge_index[1], expected_columns) or not torch.equal(
                edge_index[0], assignment.to(edge_index.device)
            ):
                raise ValueError("严格模式: Road→Syntax sparse mapping 与 assignment 不一致")
        operator = sparse_operator(edge_index, weight, tuple(shape))
        return sparse_pool(operator, road_h)
    if assignment is None:
        raise ValueError("严格模式: Road→Syntax 必须提供 sparse mapping 或 assignment")
    assignment = assignment.to(road_h.device)
    if assignment.dtype != torch.long or assignment.shape != (road_h.shape[0],):
        raise ValueError("严格模式: Road→Syntax assignment 应为 LongTensor[M]")
    if assignment.numel() and (
        int(assignment.min()) < 0 or int(assignment.max()) >= num_syntax
    ):
        raise ValueError("严格模式: Road→Syntax assignment 越界")
    counts = torch.bincount(assignment, minlength=num_syntax)
    if (counts == 0).any():
        raise ValueError("严格模式: Road→Syntax assignment 产生空 Syntax 节点")
    pooled = road_h.new_zeros((num_syntax, road_h.shape[1]))
    pooled.index_add_(0, assignment, road_h)
    return pooled / counts.to(road_h.dtype).unsqueeze(-1)


def pool_syntax_to_region(
    syntax_h: torch.Tensor,
    *,
    edge_index: torch.Tensor,
    weight: torch.Tensor,
    shape: tuple[int, int],
    mode: str = "weighted_mean",
) -> torch.Tensor:
    """用第一阶段 UTM 相交长度归一化权重池化 ``[K,D] -> [N,D]``。"""

    if syntax_h.ndim != 2 or syntax_h.shape[0] <= 0 or not torch.isfinite(syntax_h).all():
        raise ValueError("严格模式: syntax_h 必须为有限非空 [K,D]")
    if mode != "weighted_mean":
        raise ValueError("严格模式: 第一版 Syntax→Region 仅支持 weighted_mean")
    if tuple(shape)[1] != syntax_h.shape[0] or tuple(shape)[0] <= 0:
        raise ValueError("严格模式: Syntax→Region sparse shape 错误")
    operator = sparse_operator(edge_index, weight, tuple(shape))
    return sparse_pool(operator, syntax_h)
