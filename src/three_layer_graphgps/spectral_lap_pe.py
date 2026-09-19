"""Unified three-layer sparse Laplacian positional encoding.

The directed heterogeneous message graph is never replaced.  LapPE receives a
separate undirected copy containing every relation and uses sparse ``eigsh``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any
import warnings

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import ArpackNoConvergence, eigsh

from static_hierarchy.contracts import CityStaticHierarchy, validate_city_static_hierarchy


LAPPE_VERSION = "three-layer-joint-lappe-v3-weighted"


@dataclass(frozen=True)
class LaplacianEigenpairs:
    """单层图的定长 Laplacian 特征对。

    ``mask[j]`` 表示第 ``j`` 个频率是真实求得的特征对；其余位置为零 padding。
    ``edge_index_pe`` 是用于构造 Laplacian 的无向、去重边，仅用于审计。
    """

    eigvals: torch.Tensor
    eigvecs: torch.Tensor
    mask: torch.Tensor
    edge_index_pe: torch.Tensor
    metadata: dict[str, Any]
    edge_weight_pe: torch.Tensor | None = None
    edge_type_pe: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> "LaplacianEigenpairs":
        return LaplacianEigenpairs(
            eigvals=self.eigvals.to(device),
            eigvecs=self.eigvecs.to(device),
            mask=self.mask.to(device),
            edge_index_pe=self.edge_index_pe.to(device),
            metadata=self.metadata,
            edge_weight_pe=None if self.edge_weight_pe is None else self.edge_weight_pe.to(device),
            edge_type_pe=None if self.edge_type_pe is None else self.edge_type_pe.to(device),
        )


@dataclass(frozen=True)
class HierarchyLaplacianPE:
    joint: LaplacianEigenpairs
    road_node_range: tuple[int, int]
    syntax_node_range: tuple[int, int]
    region_node_range: tuple[int, int]
    metadata: dict[str, Any]

    def to(self, device: torch.device | str) -> "HierarchyLaplacianPE":
        return HierarchyLaplacianPE(
            joint=self.joint.to(device),
            road_node_range=self.road_node_range,
            syntax_node_range=self.syntax_node_range,
            region_node_range=self.region_node_range,
            metadata=self.metadata,
        )

    @property
    def joint_eigvals(self) -> torch.Tensor:
        """Compatibility view of the shared spectrum as ``[1,k,1]``."""

        return self.joint.eigvals[:1]

    @property
    def joint_eigvecs(self) -> torch.Tensor:
        return self.joint.eigvecs

    @property
    def joint_eigenpair_mask(self) -> torch.Tensor:
        return self.joint.mask

    @property
    def edge_index_joint_pe(self) -> torch.Tensor:
        return self.joint.edge_index_pe

    @property
    def edge_weight_joint_pe(self) -> torch.Tensor | None:
        return self.joint.edge_weight_pe

    @property
    def edge_type_joint_pe(self) -> torch.Tensor | None:
        return self.joint.edge_type_pe


def _validate_edge_index(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if num_nodes <= 0:
        raise ValueError("严格模式: LapPE num_nodes 必须为正")
    if not isinstance(edge_index, torch.Tensor):
        edge_index = torch.as_tensor(edge_index)
    if edge_index.dtype != torch.long or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("严格模式: LapPE edge_index 必须为 LongTensor[2,E]")
    edge_index = edge_index.detach().cpu()
    if edge_index.numel() and (
        int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes
    ):
        raise ValueError("严格模式: LapPE edge_index 节点索引越界")
    return edge_index


def to_undirected_edge_index_with_weight(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: torch.Tensor | None = None,
    edge_type: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """无向化并聚合平行边，同时保留消息边权。

    每个无向 pair 以两个方向保存；同一 pair 的所有有向关系权重求和，
    这是 normalized Laplacian 对有向消息图的可审计无向副本。关系类型
    不参与数值求和，但会作为 ``edge_type_pe`` 和 cache identity 保存，
    防止不同关系图错误复用同一组 eigenpairs。
    """
    """返回稳定排序、去重且不含自环的无向 COO 边。

    无向边以两个方向显式保存。输入不会被原地修改。
    """

    edge_index = _validate_edge_index(edge_index, num_nodes)
    if edge_weight is None:
        weights = torch.ones(edge_index.shape[1], dtype=torch.float64)
    else:
        weights = torch.as_tensor(edge_weight).detach().cpu().reshape(-1)
        if weights.shape != (edge_index.shape[1],) or not weights.is_floating_point():
            raise ValueError("严格模式: LapPE edge_weight 长度/dtype 错误")
        if not torch.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("严格模式: LapPE edge_weight 必须为有限正数")
        weights = weights.to(torch.float64)
    types = None
    if edge_type is not None:
        types = torch.as_tensor(edge_type).detach().cpu().reshape(-1)
        if types.shape != (edge_index.shape[1],) or types.dtype != torch.long:
            raise ValueError("严格模式: LapPE edge_type 长度/dtype 错误")
    if edge_index.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.float64), None if types is None else torch.empty(0, dtype=torch.long)
    src = edge_index[0].numpy().astype(np.int64, copy=False)
    dst = edge_index[1].numpy().astype(np.int64, copy=False)
    keep = src != dst
    kept_weights = weights.numpy()[keep]
    kept_types = None if types is None else types.numpy()[keep]
    src, dst = src[keep], dst[keep]
    if src.size == 0:
        return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.float64), None if types is None else torch.empty(0, dtype=torch.long)
    left = np.minimum(src, dst)
    right = np.maximum(src, dst)
    pairs = np.stack([left, right], axis=1)
    unique_pairs, inverse = np.unique(pairs, axis=0, return_inverse=True)
    pair_weights = np.zeros(unique_pairs.shape[0], dtype=np.float64)
    np.add.at(pair_weights, inverse, kept_weights)
    both = np.concatenate([unique_pairs, unique_pairs[:, ::-1]], axis=0)
    both_weights = np.concatenate([pair_weights, pair_weights], axis=0)
    both_types = None
    if kept_types is not None:
        # Preserve a deterministic audit label for each undirected pair.  The
        # numeric Laplacian uses the aggregated weight above; relation labels
        # are retained for cache identity and inspection.
        pair_types = np.zeros(unique_pairs.shape[0], dtype=np.int64)
        for pair_index in range(unique_pairs.shape[0]):
            pair_types[pair_index] = int(np.min(kept_types[inverse == pair_index]))
        both_types = np.concatenate([pair_types, pair_types], axis=0)
    order = np.lexsort((both[:, 1], both[:, 0]))
    return (
        torch.from_numpy(both[order].T.copy()).long(),
        torch.from_numpy(both_weights[order].copy()),
        None if both_types is None else torch.from_numpy(both_types[order].copy()).long(),
    )


def to_undirected_edge_index(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """返回稳定排序、去重且不含自环的无向 COO 边（兼容旧 API）。"""

    return to_undirected_edge_index_with_weight(edge_index, num_nodes)[0]


def _graph_hash(
    edge_index_pe: torch.Tensor,
    num_nodes: int,
    edge_weight_pe: torch.Tensor | None = None,
    edge_type_pe: torch.Tensor | None = None,
) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray([num_nodes], dtype="<i8").tobytes())
    digest.update(edge_index_pe.contiguous().numpy().astype("<i8", copy=False).tobytes())
    if edge_weight_pe is not None:
        digest.update(np.asarray(edge_weight_pe.cpu(), dtype="<f8").tobytes())
    if edge_type_pe is not None:
        digest.update(edge_type_pe.contiguous().numpy().astype("<i8", copy=False).tobytes())
    return digest.hexdigest()


def pe_graph_hash(edge_index: torch.Tensor, num_nodes: int) -> str:
    """计算 LapPE 无向图的稳定 hash，用于外部 PE 与静态图的严格对齐。"""

    return _graph_hash(to_undirected_edge_index(edge_index, num_nodes), num_nodes)


def _laplacian(
    edge_index_pe: torch.Tensor,
    num_nodes: int,
    normalization: str,
    edge_weight_pe: torch.Tensor | None = None,
) -> sparse.csr_matrix:
    if normalization != "sym":
        raise ValueError("严格模式: 第一版 LapPE 仅支持 normalization='sym'")
    if edge_index_pe.numel():
        indices = edge_index_pe.numpy()
        values = (
            np.ones(indices.shape[1], dtype=np.float64)
            if edge_weight_pe is None
            else np.asarray(edge_weight_pe.detach().cpu(), dtype=np.float64)
        )
        adjacency = sparse.coo_matrix(
            (values, (indices[0], indices[1])),
            shape=(num_nodes, num_nodes),
        ).tocsr()
        adjacency.sum_duplicates()
    else:
        adjacency = sparse.csr_matrix((num_nodes, num_nodes), dtype=np.float64)
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    inv_sqrt_degree = np.zeros_like(degree)
    positive = degree > 0
    inv_sqrt_degree[positive] = 1.0 / np.sqrt(degree[positive])
    scale = sparse.diags(inv_sqrt_degree, format="csr")
    return (sparse.eye(num_nodes, dtype=np.float64, format="csr") - scale @ adjacency @ scale).tocsr()


def _solve_smallest(laplacian: sparse.csr_matrix, requested: int) -> tuple[np.ndarray, np.ndarray]:
    num_nodes = laplacian.shape[0]
    if num_nodes == 1:
        return np.zeros(1, dtype=np.float64), np.ones((1, 1), dtype=np.float64)
    k_eff = min(requested, num_nodes - 1)
    deterministic_v0 = np.linspace(1.0, 2.0, num_nodes, dtype=np.float64)
    try:
        values, vectors = eigsh(
            laplacian,
            k=k_eff,
            which="SM",
            tol=1e-6,
            maxiter=max(1000, 10 * num_nodes),
            v0=deterministic_v0,
        )
    except ArpackNoConvergence as exc:
        values = np.asarray(exc.eigenvalues if exc.eigenvalues is not None else [], dtype=np.float64)
        vectors = np.asarray(
            exc.eigenvectors if exc.eigenvectors is not None else np.empty((num_nodes, 0)),
            dtype=np.float64,
        )
        if values.size == 0:
            retry_k = max(1, k_eff // 2)
            warnings.warn(
                f"eigsh 未收敛到 {k_eff} 个 eigenpairs，重试 {retry_k} 个并对其余频率 padding",
                RuntimeWarning,
            )
            try:
                values, vectors = eigsh(
                    laplacian,
                    k=retry_k,
                    which="SM",
                    tol=1e-5,
                    maxiter=max(2000, 20 * num_nodes),
                    v0=deterministic_v0,
                )
            except ArpackNoConvergence as retry_exc:
                values = np.asarray(
                    retry_exc.eigenvalues if retry_exc.eigenvalues is not None else [],
                    dtype=np.float64,
                )
                vectors = np.asarray(
                    retry_exc.eigenvectors
                    if retry_exc.eigenvectors is not None
                    else np.empty((num_nodes, 0)),
                    dtype=np.float64,
                )
        else:
            warnings.warn(
                f"eigsh 仅收敛到 {values.size}/{k_eff} 个 eigenpairs，其余频率使用 mask padding",
                RuntimeWarning,
            )
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    vectors = np.asarray(vectors, dtype=np.float64).reshape(num_nodes, -1)
    if values.shape[0] != vectors.shape[1]:
        raise RuntimeError("严格模式: eigsh eigenvalue/eigenvector 数量不一致")
    if values.size:
        order = np.argsort(values, kind="stable")
        values, vectors = values[order], vectors[:, order]
        values[np.abs(values) < 1e-7] = 0.0
        # eigsh eigenvectors are sign-indeterminate.  Canonicalize each
        # column using its largest-magnitude entry so cached/recomputed PE
        # has a stable sign (up to genuinely repeated-eigenvalue rotations).
        for column in range(vectors.shape[1]):
            pivot = int(np.argmax(np.abs(vectors[:, column])))
            if vectors[pivot, column] < 0:
                vectors[:, column] *= -1.0
    return values, vectors


def _cache_path(cache_dir: Path, cache_key: str, identity: dict[str, Any]) -> Path:
    safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", cache_key).strip("_") or "graph"
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return cache_dir / f"{safe_key}_{fingerprint}.npz"


def _from_cache(path: Path, identity: dict[str, Any]) -> LaplacianEigenpairs:
    data = np.load(path, allow_pickle=False)
    required = {"eigvals", "eigvecs", "mask", "edge_index_pe", "metadata_json"}
    missing = sorted(required - set(data.files))
    if missing:
        raise KeyError(f"严格模式: LapPE cache 缺少字段 {missing}: {path}")
    metadata = json.loads(str(data["metadata_json"].item()))
    if any(metadata.get(key) != value for key, value in identity.items()):
        raise ValueError(f"严格模式: LapPE cache identity 不匹配: {path}")
    if not isinstance(metadata.get("num_computed"), int):
        raise ValueError(f"严格模式: LapPE cache 缺少 num_computed: {path}")
    num_nodes, k = int(identity["num_nodes"]), int(identity["k"])
    eigvals = torch.from_numpy(np.asarray(data["eigvals"], dtype=np.float32))
    eigvecs = torch.from_numpy(np.asarray(data["eigvecs"], dtype=np.float32))
    mask = torch.from_numpy(np.asarray(data["mask"], dtype=np.bool_))
    edge_index_pe = torch.from_numpy(np.asarray(data["edge_index_pe"], dtype=np.int64)).long()
    edge_weight_pe = None
    if "edge_weight_pe" in data.files:
        edge_weight_pe = torch.from_numpy(np.asarray(data["edge_weight_pe"], dtype=np.float64))
    edge_type_pe = None
    if "edge_type_pe" in data.files:
        edge_type_pe = torch.from_numpy(np.asarray(data["edge_type_pe"], dtype=np.int64)).long()
    result = LaplacianEigenpairs(eigvals, eigvecs, mask, edge_index_pe, metadata, edge_weight_pe, edge_type_pe)
    _validate_result(result, num_nodes, k)
    return result


def _validate_result(result: LaplacianEigenpairs, num_nodes: int, k: int) -> None:
    if result.eigvals.shape != (num_nodes, k, 1):
        raise ValueError(f"严格模式: eigvals 应为 [{num_nodes},{k},1]")
    if result.eigvecs.shape != (num_nodes, k):
        raise ValueError(f"严格模式: eigvecs 应为 [{num_nodes},{k}]")
    if result.mask.dtype != torch.bool or result.mask.shape != (k,):
        raise ValueError(f"严格模式: LapPE mask 应为 BoolTensor[{k}]")
    if not torch.isfinite(result.eigvals).all() or not torch.isfinite(result.eigvecs).all():
        raise ValueError("严格模式: LapPE eigenpairs 含 NaN/Inf")
    if result.edge_weight_pe is not None:
        if result.edge_weight_pe.shape != (result.edge_index_pe.shape[1],):
            raise ValueError("严格模式: LapPE edge_weight_pe 长度错误")
        if not torch.isfinite(result.edge_weight_pe).all() or (result.edge_weight_pe <= 0).any():
            raise ValueError("严格模式: LapPE edge_weight_pe 必须为有限正数")
    if result.edge_type_pe is not None and result.edge_type_pe.shape != (result.edge_index_pe.shape[1],):
        raise ValueError("严格模式: LapPE edge_type_pe 长度错误")
    valid_values = result.eigvals[0, result.mask, 0]
    if valid_values.numel() > 1 and torch.any(valid_values[1:] < valid_values[:-1] - 1e-6):
        raise ValueError("严格模式: LapPE eigenvalues 未按升序排列")


def compute_sparse_laplacian_eigenpairs(
    edge_index: torch.Tensor,
    num_nodes: int,
    k: int,
    normalization: str = "sym",
    is_directed: bool = False,
    cache_key: str | None = None,
    cache_dir: str | Path | None = None,
    *,
    pe_version: str = LAPPE_VERSION,
    metadata_extra: dict[str, Any] | None = None,
    edge_weight: torch.Tensor | None = None,
    edge_type: torch.Tensor | None = None,
) -> LaplacianEigenpairs:
    """以固定形状返回最小 Laplacian eigenpairs，并可按图指纹缓存。

    ``is_directed`` 记录消息图语义；无论其取值，PE 图都会显式无向化。
    原始 ``edge_index`` 不会被修改。
    """

    num_nodes, k = int(num_nodes), int(k)
    if k <= 0:
        raise ValueError("严格模式: LapPE k 必须为正")
    original = _validate_edge_index(edge_index, num_nodes)
    edge_index_pe, edge_weight_pe, edge_type_pe = to_undirected_edge_index_with_weight(
        original, num_nodes, edge_weight=edge_weight, edge_type=edge_type
    )
    graph_hash = _graph_hash(edge_index_pe, num_nodes, edge_weight_pe, edge_type_pe)
    edge_type_hash = None
    if edge_type_pe is not None:
        edge_type_hash = hashlib.sha256(edge_type_pe.numpy().astype("<i8", copy=False).tobytes()).hexdigest()
    identity = {
        "cache_key": cache_key,
        "num_nodes": num_nodes,
        "num_undirected_edges": int(edge_index_pe.shape[1]),
        "graph_hash": graph_hash,
        "k": k,
        "normalization": normalization,
        "message_graph_is_directed": bool(is_directed),
        "pe_graph_is_undirected": True,
        "weighted_pe": edge_weight is not None,
        "edge_type_pe_hash": edge_type_hash,
        "pe_version": str(pe_version),
    }
    if metadata_extra:
        identity.update(metadata_extra)
    path = None
    if cache_dir is not None:
        if not cache_key:
            raise ValueError("严格模式: 启用 LapPE cache 时必须提供 cache_key")
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = _cache_path(cache_dir, cache_key, identity)
        if path.exists():
            return _from_cache(path, identity)

    values, vectors = _solve_smallest(_laplacian(edge_index_pe, num_nodes, normalization, edge_weight_pe), k)
    valid_count = min(k, values.shape[0])
    padded_values = np.zeros(k, dtype=np.float32)
    padded_vectors = np.zeros((num_nodes, k), dtype=np.float32)
    mask = np.zeros(k, dtype=np.bool_)
    if valid_count:
        padded_values[:valid_count] = values[:valid_count].astype(np.float32)
        padded_vectors[:, :valid_count] = vectors[:, :valid_count].astype(np.float32)
        mask[:valid_count] = True
    identity["num_computed"] = int(valid_count)
    eigvals = np.broadcast_to(padded_values[None, :, None], (num_nodes, k, 1)).copy()
    result = LaplacianEigenpairs(
        eigvals=torch.from_numpy(eigvals),
        eigvecs=torch.from_numpy(padded_vectors),
        mask=torch.from_numpy(mask),
        edge_index_pe=edge_index_pe,
        metadata=identity,
        edge_weight_pe=edge_weight_pe,
        edge_type_pe=edge_type_pe,
    )
    _validate_result(result, num_nodes, k)
    if path is not None:
        # identity 中的 num_computed 在求解后才能确定，因此用最终 identity 重算文件名。
        cache_identity = dict(identity)
        path = _cache_path(Path(cache_dir), str(cache_key), {
            key: value for key, value in cache_identity.items() if key != "num_computed"
        })
        cache_payload = {
            "eigvals": result.eigvals.numpy(),
            "eigvecs": result.eigvecs.numpy(),
            "mask": result.mask.numpy(),
            "edge_index_pe": result.edge_index_pe.numpy(),
            "metadata_json": np.asarray(json.dumps(cache_identity, sort_keys=True)),
        }
        if result.edge_weight_pe is not None:
            cache_payload["edge_weight_pe"] = result.edge_weight_pe.numpy()
        if result.edge_type_pe is not None:
            cache_payload["edge_type_pe"] = result.edge_type_pe.numpy()
        np.savez_compressed(path, **cache_payload)
    return result


def prepare_hierarchy_lappe(
    hierarchy: CityStaticHierarchy,
    *,
    road_k: int | None = None,
    syntax_k: int | None = None,
    region_k: int | None = None,
    joint_k: int | None = None,
    normalization: str = "sym",
    cache_dir: str | Path | None = None,
    pe_version: str = LAPPE_VERSION,
) -> HierarchyLaplacianPE:
    """Prepare one shared LapPE spectrum for the complete directed hierarchy."""

    validate_city_static_hierarchy(hierarchy)
    # Import lazily to avoid a module cycle (data.py also exposes the public
    # graph builder and imports this module for the PE dataclass).
    from .data import build_joint_three_layer_graph

    graph = build_joint_three_layer_graph(hierarchy)
    requested = [value for value in (road_k, syntax_k, region_k) if value is not None]
    if joint_k is None:
        joint_k = max(requested) if requested else 16
    joint_k = int(joint_k)
    if joint_k <= 0:
        raise ValueError("严格模式: joint_k 必须为正")
    graph_version = hierarchy.metadata.get("feature_version")
    base = f"{hierarchy.city_id}_{graph_version}_joint_{graph.joint_graph_hash[:16]}"
    metadata_extra = {
        "hierarchy_version": str(graph_version),
        "joint_graph_hash": graph.joint_graph_hash,
        "num_joint_nodes": graph.num_nodes,
        "road_node_range": list(graph.road_node_range),
        "syntax_node_range": list(graph.syntax_node_range),
        "region_node_range": list(graph.region_node_range),
    }
    joint = compute_sparse_laplacian_eigenpairs(
        graph.edge_index_joint,
        graph.num_nodes,
        joint_k,
        normalization,
        is_directed=True,
        cache_key=base,
        cache_dir=cache_dir,
        pe_version=pe_version,
        metadata_extra=metadata_extra,
        edge_weight=graph.edge_weight,
        edge_type=graph.edge_type,
    )
    if joint.metadata.get("joint_graph_hash") != graph.joint_graph_hash:
        raise ValueError("严格模式: LapPE joint_graph_hash 与联合消息图不一致")
    return HierarchyLaplacianPE(
        joint=joint,
        road_node_range=graph.road_node_range,
        syntax_node_range=graph.syntax_node_range,
        region_node_range=graph.region_node_range,
        metadata={
            "joint_graph_hash": graph.joint_graph_hash,
            "num_joint_nodes": graph.num_nodes,
            "hierarchy_version": graph_version,
            "pe_version": pe_version,
        },
    )
