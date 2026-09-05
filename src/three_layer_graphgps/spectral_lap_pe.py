"""三层 GraphGPS 使用的稀疏 Laplacian positional encoding。

Road/Syntax/Region 的消息图保持原契约；本模块只为 LapPE 构造无向图副本，
并使用 SciPy sparse ``eigsh`` 求解最小特征对，禁止在大图上稠密分解。
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


LAPPE_VERSION = "three-layer-lappe-v1"


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

    def to(self, device: torch.device | str) -> "LaplacianEigenpairs":
        return LaplacianEigenpairs(
            eigvals=self.eigvals.to(device),
            eigvecs=self.eigvecs.to(device),
            mask=self.mask.to(device),
            edge_index_pe=self.edge_index_pe.to(device),
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class HierarchyLaplacianPE:
    road: LaplacianEigenpairs
    syntax: LaplacianEigenpairs
    region: LaplacianEigenpairs

    def to(self, device: torch.device | str) -> "HierarchyLaplacianPE":
        return HierarchyLaplacianPE(
            road=self.road.to(device),
            syntax=self.syntax.to(device),
            region=self.region.to(device),
        )


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


def to_undirected_edge_index(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """返回稳定排序、去重且不含自环的无向 COO 边。

    无向边以两个方向显式保存。输入不会被原地修改。
    """

    edge_index = _validate_edge_index(edge_index, num_nodes)
    if edge_index.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)
    src = edge_index[0].numpy().astype(np.int64, copy=False)
    dst = edge_index[1].numpy().astype(np.int64, copy=False)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    if src.size == 0:
        return torch.empty((2, 0), dtype=torch.long)
    left = np.minimum(src, dst)
    right = np.maximum(src, dst)
    undirected_pairs = np.unique(np.stack([left, right], axis=1), axis=0)
    both = np.concatenate([undirected_pairs, undirected_pairs[:, ::-1]], axis=0)
    order = np.lexsort((both[:, 1], both[:, 0]))
    return torch.from_numpy(both[order].T.copy()).long()


def _graph_hash(edge_index_pe: torch.Tensor, num_nodes: int) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray([num_nodes], dtype="<i8").tobytes())
    digest.update(edge_index_pe.contiguous().numpy().astype("<i8", copy=False).tobytes())
    return digest.hexdigest()


def pe_graph_hash(edge_index: torch.Tensor, num_nodes: int) -> str:
    """计算 LapPE 无向图的稳定 hash，用于外部 PE 与静态图的严格对齐。"""

    return _graph_hash(to_undirected_edge_index(edge_index, num_nodes), num_nodes)


def _laplacian(edge_index_pe: torch.Tensor, num_nodes: int, normalization: str) -> sparse.csr_matrix:
    if normalization != "sym":
        raise ValueError("严格模式: 第一版 LapPE 仅支持 normalization='sym'")
    if edge_index_pe.numel():
        indices = edge_index_pe.numpy()
        adjacency = sparse.coo_matrix(
            (np.ones(indices.shape[1], dtype=np.float64), (indices[0], indices[1])),
            shape=(num_nodes, num_nodes),
        ).tocsr()
        adjacency.sum_duplicates()
        adjacency.data.fill(1.0)
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
    result = LaplacianEigenpairs(eigvals, eigvecs, mask, edge_index_pe, metadata)
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
) -> LaplacianEigenpairs:
    """以固定形状返回最小 Laplacian eigenpairs，并可按图指纹缓存。

    ``is_directed`` 记录消息图语义；无论其取值，PE 图都会显式无向化。
    原始 ``edge_index`` 不会被修改。
    """

    num_nodes, k = int(num_nodes), int(k)
    if k <= 0:
        raise ValueError("严格模式: LapPE k 必须为正")
    original = _validate_edge_index(edge_index, num_nodes)
    edge_index_pe = to_undirected_edge_index(original, num_nodes)
    graph_hash = _graph_hash(edge_index_pe, num_nodes)
    identity = {
        "cache_key": cache_key,
        "num_nodes": num_nodes,
        "num_undirected_edges": int(edge_index_pe.shape[1]),
        "graph_hash": graph_hash,
        "k": k,
        "normalization": normalization,
        "message_graph_is_directed": bool(is_directed),
        "pe_graph_is_undirected": True,
        "pe_version": str(pe_version),
    }
    path = None
    if cache_dir is not None:
        if not cache_key:
            raise ValueError("严格模式: 启用 LapPE cache 时必须提供 cache_key")
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = _cache_path(cache_dir, cache_key, identity)
        if path.exists():
            return _from_cache(path, identity)

    values, vectors = _solve_smallest(_laplacian(edge_index_pe, num_nodes, normalization), k)
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
    )
    _validate_result(result, num_nodes, k)
    if path is not None:
        # identity 中的 num_computed 在求解后才能确定，因此用最终 identity 重算文件名。
        cache_identity = dict(identity)
        path = _cache_path(Path(cache_dir), str(cache_key), {
            key: value for key, value in cache_identity.items() if key != "num_computed"
        })
        np.savez_compressed(
            path,
            eigvals=result.eigvals.numpy(),
            eigvecs=result.eigvecs.numpy(),
            mask=result.mask.numpy(),
            edge_index_pe=result.edge_index_pe.numpy(),
            metadata_json=np.asarray(json.dumps(cache_identity, sort_keys=True)),
        )
    return result


def prepare_hierarchy_lappe(
    hierarchy: CityStaticHierarchy,
    *,
    road_k: int,
    syntax_k: int,
    region_k: int,
    normalization: str = "sym",
    cache_dir: str | Path | None = None,
    pe_version: str = LAPPE_VERSION,
) -> HierarchyLaplacianPE:
    """为三层静态图准备 LapPE，不读取任何动态流量。"""

    validate_city_static_hierarchy(hierarchy)
    graph_version = hierarchy.metadata.get("feature_version")
    base = f"{hierarchy.city_id}_{graph_version}"
    return HierarchyLaplacianPE(
        road=compute_sparse_laplacian_eigenpairs(
            hierarchy.road_edge_index,
            hierarchy.num_roads,
            road_k,
            normalization,
            is_directed=True,
            cache_key=f"{base}_road",
            cache_dir=cache_dir,
            pe_version=pe_version,
        ),
        syntax=compute_sparse_laplacian_eigenpairs(
            hierarchy.syntax_edge_index,
            hierarchy.num_syntax,
            syntax_k,
            normalization,
            is_directed=True,
            cache_key=f"{base}_syntax",
            cache_dir=cache_dir,
            pe_version=pe_version,
        ),
        region=compute_sparse_laplacian_eigenpairs(
            hierarchy.region_edge_index,
            hierarchy.num_regions,
            region_k,
            normalization,
            is_directed=True,
            cache_key=f"{base}_region",
            cache_dir=cache_dir,
            pe_version=pe_version,
        ),
    )
