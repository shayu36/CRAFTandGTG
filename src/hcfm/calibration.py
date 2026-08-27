"""宏观 OD 流与道路边界 passage count 的源训练集口径校准。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable

import torch


@dataclass(frozen=True)
class CalibrationMetadata:
    fitted_cities: tuple[str, ...]
    fitted_split: str
    method: str
    data_version: str


class SourceOnlyConservationCalibrator:
    """每个 in/out 通道的非负最小二乘比例 ``macro ~= scale * S(micro)``。

    不使用截距，保持零 passage 对应零流量的可解释性；仅源城市 train 可拟合。
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = float(eps)
        self.scale: torch.Tensor | None = None
        self.metadata: CalibrationMetadata | None = None

    def fit(
        self,
        aggregated_micro: torch.Tensor,
        macro: torch.Tensor,
        region_mask: torch.Tensor,
        *,
        cities: Iterable[str],
        source_cities: Iterable[str],
        split: str,
        data_version: str,
    ) -> "SourceOnlyConservationCalibrator":
        cities, sources = tuple(sorted(set(cities))), set(source_cities)
        if split != "train" or not cities or not set(cities).issubset(sources):
            raise ValueError("泄漏防护: conservation calibration 只能拟合源城市 train")
        if aggregated_micro.shape != macro.shape or aggregated_micro.ndim != 4 or macro.shape[2] != 2:
            raise ValueError("calibration 期望 macro/aggregated_micro 同形 [B,N,2,T]")
        if not torch.isfinite(aggregated_micro).all() or not torch.isfinite(macro).all():
            raise ValueError("calibration 输入含 NaN/Inf")
        if region_mask.ndim == 1:
            region_mask = region_mask.unsqueeze(0)
        mask = region_mask[:, :, None, None].expand_as(macro)
        scales = []
        for channel in range(2):
            channel_mask = mask[:, :, channel]
            x = aggregated_micro[:, :, channel][channel_mask]
            y = macro[:, :, channel][channel_mask]
            denominator = (x * x).sum()
            if denominator <= self.eps:
                raise ValueError(f"calibration channel={channel} 没有非零 micro passage")
            scales.append(((x * y).sum() / denominator).clamp_min(0.0))
        self.scale = torch.stack(scales)
        self.metadata = CalibrationMetadata(
            fitted_cities=cities, fitted_split=split,
            method="nonnegative_scale_no_intercept", data_version=str(data_version),
        )
        return self

    def apply(self, aggregated_micro: torch.Tensor) -> torch.Tensor:
        if self.scale is None:
            raise RuntimeError("conservation calibrator 尚未拟合")
        return aggregated_micro * self.scale.to(aggregated_micro).view(1, 1, 2, 1)

    def state_dict(self) -> Dict[str, Any]:
        if self.scale is None or self.metadata is None:
            raise RuntimeError("conservation calibrator 尚未拟合")
        return {"scale": self.scale, "metadata": asdict(self.metadata)}

    def load_state_dict(self, state: Dict[str, Any], expected: Dict[str, Any] | None = None) -> None:
        meta = dict(state["metadata"])
        if expected:
            for key, value in expected.items():
                if meta.get(key) != value:
                    raise ValueError(f"calibration 元数据不一致: {key}")
        self.scale = state["scale"].detach().clone()
        self.metadata = CalibrationMetadata(
            fitted_cities=tuple(meta["fitted_cities"]), fitted_split=meta["fitted_split"],
            method=meta["method"], data_version=meta["data_version"],
        )


def conservation_gap_report(
    macro: torch.Tensor, aggregated_micro: torch.Tensor, region_mask: torch.Tensor,
    eps: float = 1e-6,
) -> Dict[str, Any]:
    if macro.shape != aggregated_micro.shape:
        raise ValueError("conservation report shape 不一致")
    if region_mask.ndim == 1:
        region_mask = region_mask.unsqueeze(0)
    difference = (macro - aggregated_micro).abs()
    valid = region_mask[:, :, None, None].expand_as(difference)
    per_region = difference.mean(dim=(0, 2, 3))
    return {
        "absolute_error": float(difference[valid].mean()),
        "relative_error": float((difference / macro.abs().clamp_min(eps))[valid].mean()),
        "per_region_absolute_error": per_region.detach().cpu().tolist(),
    }
