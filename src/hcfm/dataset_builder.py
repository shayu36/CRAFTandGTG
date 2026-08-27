"""真实宏观/道路序列表到整城 HCFM 快照的严格适配。"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .data import validate_joint_sample
from .preprocessing import load_sequence_column, validate_micro_sequence_table
from .geo_time import localize_city_timestamps


def validate_macro_sequence_table(
    frame: pd.DataFrame, *, city: str, num_regions: int, seq_length: int, split: str
) -> pd.DataFrame:
    required = {"region_id", "date", "start_hour", "weekday", "month", "in_flow", "out_flow"}
    if required - set(frame.columns):
        raise ValueError(f"严格模式: macro 表缺列 {sorted(required-set(frame.columns))}")
    result = frame.copy()
    if "city_id" in result and set(result["city_id"]) != {city}:
        raise ValueError("严格模式: macro city_id 不一致")
    if "split" in result and set(result["split"]) != {split}:
        raise ValueError("严格模式: macro split 不一致")
    result["city_id"], result["split"] = city, split
    if result.duplicated(["region_id", "date", "start_hour"]).any():
        raise ValueError("严格模式: macro 同一 Region/窗口有重复行")
    if (result["region_id"] < 0).any() or (result["region_id"] >= num_regions).any():
        raise ValueError("严格模式: macro region_id 越界")
    for column in ("in_flow", "out_flow"):
        result[column] = result[column].map(
            lambda value: load_sequence_column(value, seq_length, column)
        )
        if any((value < 0).any() for value in result[column]):
            raise ValueError(f"严格模式: {column} 含负值")
    return result


def assemble_joint_samples(
    static: Mapping[str, torch.Tensor],
    manifest: Mapping[str, Any],
    macro_frame: pd.DataFrame,
    micro_frame: pd.DataFrame,
    references: Mapping[tuple[str, int], np.ndarray],
    *,
    split: str,
    seq_length: int,
    timezone: str,
    dst_policy: str,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    """只取同城同窗口 inner join，并报告两侧未联合窗口，绝不跨城/跨日期拼接。"""

    city, n, m = manifest["city_id"], manifest["num_regions"], manifest["num_directed_roads"]
    macro = validate_macro_sequence_table(
        macro_frame, city=city, num_regions=n, seq_length=seq_length, split=split
    )
    micro = validate_micro_sequence_table(micro_frame, manifest, seq_length, split)
    key_cols = ["date", "start_hour"]
    macro_keys = {tuple(row) for row in macro[key_cols].drop_duplicates().itertuples(index=False, name=None)}
    micro_keys = {tuple(row) for row in micro[key_cols].drop_duplicates().itertuples(index=False, name=None)}
    joint_keys = sorted(macro_keys & micro_keys)
    if not joint_keys:
        raise ValueError(f"严格模式: {city} macro/micro 无同时间联合窗口")
    if dst_policy not in {"raise", "first", "second"}:
        raise ValueError("dst_policy 必须为 raise/first/second")
    ambiguous = {"raise": "raise", "first": True, "second": False}[dst_policy]
    road_order = {str(value): index for index, value in enumerate(manifest["directed_road_ids"])}
    samples = []
    for date, start_hour in joint_keys:
        localized = localize_city_timestamps(
            [f"{date} {int(start_hour):02d}:00:00"], timezone,
            ambiguous=ambiguous, nonexistent="raise",
        )[0]
        macro_group = macro[(macro["date"] == date) & (macro["start_hour"] == start_hour)]
        micro_group = micro[(micro["date"] == date) & (micro["start_hour"] == start_hour)]
        macro_flow = torch.zeros((n, 2, seq_length), dtype=torch.float32)
        region_mask = torch.zeros(n, dtype=torch.bool)
        for row in macro_group.itertuples():
            rid = int(row.region_id)
            macro_flow[rid, 0] = torch.from_numpy(row.in_flow)
            macro_flow[rid, 1] = torch.from_numpy(row.out_flow)
            region_mask[rid] = True
        micro_flow = torch.empty((m, 1, seq_length), dtype=torch.float32)
        seen = torch.zeros(m, dtype=torch.bool)
        for row in micro_group.itertuples():
            rid = road_order[str(row.directed_road_id)]
            if seen[rid]:
                raise ValueError(f"严格模式: micro road {row.directed_road_id} 重复")
            micro_flow[rid, 0] = torch.from_numpy(row.road_passage_count)
            seen[rid] = True
        if not seen.all():
            raise ValueError("严格模式: micro 快照 Road 覆盖不完整")
        weekday_values, month_values = set(macro_group["weekday"]), set(macro_group["month"])
        if len(weekday_values) != 1 or len(month_values) != 1:
            raise ValueError("严格模式: 同一宏观窗口 weekday/month 不唯一")
        ref_key = (str(date), int(start_hour))
        if ref_key not in references:
            raise KeyError(f"严格模式: 缺少 Region RAG reference {ref_key}")
        reference = torch.as_tensor(references[ref_key], dtype=torch.float32)
        if reference.shape != macro_flow.shape or not torch.isfinite(reference).all():
            raise ValueError("严格模式: reference 应为有限 [N,2,T]")
        sample = {
            "city_id": city, "date": str(date), "start_hour": int(start_hour),
            "region_x": static["region_x"], "region_edge_index": static["region_edge_index"],
            "macro_flow": macro_flow,
            "road_x": static["road_x"], "road_edge_index": static["road_edge_index"],
            "micro_flow": micro_flow,
            "p_struct": static["p_struct"], "b_in": static["b_in"], "b_out": static["b_out"],
            "region_mask": region_mask, "road_mask": seen,
            "reference": reference,
            "time_features": {
                "month": int(next(iter(month_values))), "weekday": int(next(iter(weekday_values))),
                "start_hour": int(start_hour),
            },
            "month": int(next(iter(month_values))), "weekday": int(next(iter(weekday_values))),
            "split": split,
            "timezone": timezone, "dst_policy": dst_policy,
            "localized_start_time": localized.isoformat(),
            "dynamic_metadata": {
                branch: {
                    "city_id": city, "date": str(date), "start_hour": int(start_hour),
                    "split": split, "timezone": timezone, "dst_policy": dst_policy,
                }
                for branch in ("macro", "micro")
            },
        }
        validate_joint_sample(sample, seq_length)
        samples.append(sample)
    report = {
        "city_id": city, "split": split,
        "macro_windows": len(macro_keys), "micro_windows": len(micro_keys),
        "joint_windows": len(joint_keys),
        "macro_only_windows": len(macro_keys - micro_keys),
        "micro_only_windows": len(micro_keys - macro_keys),
    }
    return samples, report
