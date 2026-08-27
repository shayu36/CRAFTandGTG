"""HCFM 整城数据契约、严格校验与无泄漏归一化。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence

import torch
from torch.utils.data import Dataset


REQUIRED_SAMPLE_KEYS = {
    "city_id", "date", "start_hour", "region_x", "region_edge_index",
    "macro_flow", "road_x", "road_edge_index", "micro_flow", "p_struct",
    "b_in", "b_out", "region_mask", "road_mask", "time_features", "split",
}
REQUIRED_MACRO_HIERARCHY_KEYS = REQUIRED_SAMPLE_KEYS - {"micro_flow", "road_mask"}


def _require_finite(name: str, value: torch.Tensor) -> None:
    if value.is_floating_point() and not torch.isfinite(value).all():
        bad = int((~torch.isfinite(value)).sum().item())
        raise ValueError(f"严格模式: {name} 含 {bad} 个 NaN/Inf")


def _check_sparse(name: str, matrix: torch.Tensor, shape: tuple[int, int]) -> None:
    if matrix.layout != torch.sparse_coo:
        raise TypeError(f"严格模式: {name} 必须是 torch.sparse_coo_tensor")
    if tuple(matrix.shape) != shape:
        raise ValueError(f"严格模式: {name} shape={tuple(matrix.shape)} != {shape}")
    matrix = matrix.coalesce()
    _require_finite(f"{name}.values", matrix.values())
    if (matrix.values() < 0).any():
        raise ValueError(f"严格模式: {name} 含负权重")


def validate_joint_sample(
    sample: Mapping[str, Any], seq_length: int | None = None, *, require_micro: bool = True
) -> None:
    """验证一个未加 batch 维的整城联合样本。

    期望 ``macro_flow [N,2,T]``、``micro_flow [M,1,T]``。缺失值必须由
    mask 显式排除；张量本身仍必须有限，禁止用 NaN 充当 mask。
    """

    required = REQUIRED_SAMPLE_KEYS if require_micro else REQUIRED_MACRO_HIERARCHY_KEYS
    missing = required - set(sample)
    if missing:
        raise KeyError(f"严格模式: 层次样本缺少字段 {sorted(missing)}")
    if not isinstance(sample["city_id"], str) or not sample["city_id"]:
        raise ValueError("严格模式: city_id 必须是非空字符串")
    if sample["split"] not in {"train", "val", "test"}:
        raise ValueError(f"严格模式: 非法 split={sample['split']!r}")

    region_x, road_x = sample["region_x"], sample["road_x"]
    macro, micro = sample["macro_flow"], sample.get("micro_flow")
    if region_x.ndim != 2 or region_x.shape[1] != 45:
        raise ValueError(f"严格模式: region_x 应为 [N,45], 实得 {tuple(region_x.shape)}")
    if road_x.ndim != 2 or road_x.shape[1] == 0:
        raise ValueError(f"严格模式: road_x 应为 [M,d_road], 实得 {tuple(road_x.shape)}")
    n, m = region_x.shape[0], road_x.shape[0]
    if macro.ndim != 3 or tuple(macro.shape[:2]) != (n, 2):
        raise ValueError(f"严格模式: macro_flow 应为 [N,2,T], 实得 {tuple(macro.shape)}")
    if require_micro:
        if micro.ndim != 3 or tuple(micro.shape[:2]) != (m, 1):
            raise ValueError(f"严格模式: micro_flow 应为 [M,1,T], 实得 {tuple(micro.shape)}")
        if macro.shape[-1] != micro.shape[-1]:
            raise ValueError("严格模式: macro/micro 时间长度不一致")
    if seq_length is not None and macro.shape[-1] != seq_length:
        raise ValueError(f"严格模式: T={macro.shape[-1]} != 配置 {seq_length}")

    for name, edge, nodes in (
        ("region_edge_index", sample["region_edge_index"], n),
        ("road_edge_index", sample["road_edge_index"], m),
    ):
        if edge.dtype != torch.long or edge.ndim != 2 or edge.shape[0] != 2:
            raise ValueError(f"严格模式: {name} 必须为 LongTensor[2,E]")
        if edge.numel() and (int(edge.min()) < 0 or int(edge.max()) >= nodes):
            raise ValueError(f"严格模式: {name} 节点索引越界")

    for name, mask, size in (
        ("region_mask", sample["region_mask"], n),
        *(([("road_mask", sample["road_mask"], m)] if require_micro else [])),
    ):
        if mask.dtype != torch.bool or tuple(mask.shape) != (size,):
            raise ValueError(f"严格模式: {name} 必须为 BoolTensor[{size}]")
    if not sample["region_mask"].any():
        raise ValueError("严格模式: region_mask 没有任何有效监督")
    if require_micro and not sample["road_mask"].any():
        raise ValueError("严格模式: road_mask 没有任何有效监督")

    for name in ("region_x", "road_x", "macro_flow") + (("micro_flow",) if require_micro else ()):
        _require_finite(name, sample[name])
    for name in ("p_struct", "b_in", "b_out"):
        _check_sparse(name, sample[name], (n, m))

    p = sample["p_struct"].coalesce()
    row_sum = torch.sparse.sum(p, dim=1).to_dense()
    nonempty = row_sum > 0
    if nonempty.any() and not torch.allclose(
        row_sum[nonempty], torch.ones_like(row_sum[nonempty]), atol=1e-5, rtol=1e-5
    ):
        raise ValueError("严格模式: P_struct 非空行未归一化为 1")

    tf = sample["time_features"]
    if int(tf["start_hour"]) != int(sample["start_hour"]):
        raise ValueError("严格模式: time_features.start_hour 与样本主键不一致")
    meta = sample.get("dynamic_metadata", {})
    for branch in ("macro", "micro"):
        branch_meta = meta.get(branch)
        if branch_meta is None:
            continue
        for key in ("city_id", "date", "start_hour", "split"):
            expected = sample[key]
            if branch_meta.get(key) != expected:
                raise ValueError(
                    f"严格模式: {branch}.{key}={branch_meta.get(key)!r} != {expected!r}"
                )


class HierarchicalCityDataset(Dataset):
    """整城快照 Dataset；每个 item 保持城市自身 N/M，不做节点 padding。"""

    def __init__(
        self, samples: Sequence[Dict[str, Any]], seq_length: int = 24, *, require_micro: bool = True
    ):
        if not samples:
            raise ValueError("严格模式: HierarchicalCityDataset 不能为空")
        self.samples = list(samples)
        self.seq_length = int(seq_length)
        self.require_micro = bool(require_micro)
        for sample in self.samples:
            validate_joint_sample(sample, self.seq_length, require_micro=self.require_micro)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.samples[index]


def collate_city_snapshots(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """一批一个城市快照。

    动态张量添加 batch 维得到 ``[1,N,2,T]``/``[1,M,1,T]``；静态图和
    稀疏算子不复制。多城市 batch 必须显式使用块对角实现，不能误用本函数。
    """

    if len(items) != 1:
        raise ValueError("HCFM 当前采用一批一个城市快照，batch_size 必须为 1")
    item = items[0]
    out = dict(item)
    for key in ("region_x", "macro_flow", "road_x", "micro_flow", "region_mask", "road_mask"):
        if key in item:
            out[key] = item[key].unsqueeze(0)
    if "reference" in item:
        out["reference"] = item["reference"].unsqueeze(0)
    for key in ("road_cost_target", "road_cost_mask"):
        if key in item:
            out[key] = item[key].unsqueeze(0)
    return out


@dataclass(frozen=True)
class NormalizerMetadata:
    method: str
    fitted_cities: tuple[str, ...]
    fitted_split: str
    feature_order: tuple[str, ...]
    data_version: str
    eps: float


class SourceOnlyNormalizer:
    """仅允许源城市 train 拟合的逐通道标准化器。

    ``fit`` 的输入通道必须在最后一维，例如先将 macro 重排为 ``[...,2]``。
    center/scale 会进入 checkpoint；跨尺度损失用 :meth:`inverse` 回到物理单位。
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = float(eps)
        self.center: torch.Tensor | None = None
        self.scale: torch.Tensor | None = None
        self.metadata: NormalizerMetadata | None = None

    def fit(
        self,
        values: torch.Tensor,
        *,
        cities: Iterable[str],
        split: str,
        source_cities: Iterable[str],
        feature_order: Sequence[str],
        data_version: str,
        mask: torch.Tensor | None = None,
    ) -> "SourceOnlyNormalizer":
        cities, sources = tuple(sorted(set(cities))), tuple(sorted(set(source_cities)))
        if split != "train":
            raise ValueError("泄漏防护: 归一化器只能在 train split 拟合")
        if not cities or not set(cities).issubset(sources):
            raise ValueError(f"泄漏防护: fitted cities {cities} 不是 source cities {sources} 的子集")
        if values.shape[-1] != len(feature_order):
            raise ValueError("feature_order 与 values 最后一维不一致")
        _require_finite("normalizer.fit.values", values)
        flat = values.reshape(-1, values.shape[-1])
        if mask is not None:
            mask = mask.reshape(-1).bool()
            if len(mask) != len(flat):
                raise ValueError("normalizer mask 与 values 非通道元素数不一致")
            flat = flat[mask]
        if len(flat) == 0:
            raise ValueError("严格模式: 归一化拟合没有有效样本")
        self.center = flat.mean(dim=0)
        self.scale = flat.std(dim=0, unbiased=False).clamp_min(self.eps)
        self.metadata = NormalizerMetadata(
            method="standard", fitted_cities=cities, fitted_split=split,
            feature_order=tuple(feature_order), data_version=str(data_version), eps=self.eps,
        )
        return self

    def _check_fitted(self) -> None:
        if self.center is None or self.scale is None or self.metadata is None:
            raise RuntimeError("归一化器尚未拟合")

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        self._check_fitted()
        return (values - self.center.to(values)) / self.scale.to(values)

    def inverse(self, values: torch.Tensor) -> torch.Tensor:
        self._check_fitted()
        return values * self.scale.to(values) + self.center.to(values)

    def state_dict(self) -> Dict[str, Any]:
        self._check_fitted()
        return {"center": self.center, "scale": self.scale, "metadata": asdict(self.metadata)}

    def load_state_dict(self, state: Mapping[str, Any], expected: Mapping[str, Any] | None = None) -> None:
        meta = dict(state["metadata"])
        if expected:
            for key, value in expected.items():
                actual = meta.get(key)
                if isinstance(actual, list):
                    actual = tuple(actual)
                if isinstance(value, list):
                    value = tuple(value)
                if actual != value:
                    raise ValueError(f"归一化元数据不一致: {key}={actual!r} != {value!r}")
        self.center = state["center"].detach().clone()
        self.scale = state["scale"].detach().clone()
        self.metadata = NormalizerMetadata(
            method=meta["method"], fitted_cities=tuple(meta["fitted_cities"]),
            fitted_split=meta["fitted_split"], feature_order=tuple(meta["feature_order"]),
            data_version=meta["data_version"], eps=float(meta["eps"]),
        )
