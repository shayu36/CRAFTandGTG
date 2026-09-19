#!/usr/bin/env python3
"""Build real local three-layer RAG snapshots from GTG, Stage-2 and Region flow.

Output is a local ``torch.save({"snapshots": [...]})`` bundle directly
consumable by ``scripts/build_rag_memory.py``.  It is intentionally not a
Git-tracked artifact: it includes derived city temporal tensors and repeated
Stage-2 low-band features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from static_hierarchy.data import load_city_static_hierarchy  # noqa: E402
from three_layer_rag import (  # noqa: E402
    ThreeLayerRAGInputs,
    build_hourly_three_layer_dynamics,
    load_stage2_low_features,
    normalize_hourly_dynamics,
    parent_operators_from_hierarchy,
    split_snapshot_starts,
    temporal_window,
    calendar_for_timestamp,
)


def _absolute(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def _load_holidays(path_like: str | None) -> tuple[set[str], str]:
    if path_like is None:
        return set(), "not_provided_all_zero"
    path = _absolute(path_like)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("holiday_dates", payload.get("holidays"))
    if not isinstance(payload, list):
        raise ValueError("严格模式: holiday JSON 必须为日期数组或含 holiday_dates 数组的 mapping")
    import pandas as pd

    return {str(pd.Timestamp(item).date()) for item in payload}, str(path)


def _normalizer_from_path(path_like: str | None) -> dict | None:
    if path_like is None:
        return None
    path = _absolute(path_like)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build real GTG Road/Syntax/Region RAG snapshot bundle")
    parser.add_argument("--cities", nargs="+", required=True)
    parser.add_argument("--hierarchy-cache-dir", default="cache/static_hierarchy_start_v2")
    parser.add_argument("--spectral-feature-dir", default="outputs/stage2_three_layer_graphgps_lappe/spectral_features")
    parser.add_argument("--gtg-data-root", default="data")
    parser.add_argument("--region-flow-root", default="data/gtg_craft")
    parser.add_argument(
        "--region-flow-file-name",
        choices=["hourly_boundary_flow_raw.csv", "hourly_boundary_flow_interpolated.csv"],
        default="hourly_boundary_flow_raw.csv",
        help="RAG history 默认使用原始已观测流；插值版须显式选择",
    )
    parser.add_argument("--output", required=True, help="new local .pt snapshot bundle")
    parser.add_argument("--normalizer-out", help="new JSON; defaults beside --output")
    parser.add_argument("--normalizer-in", help="source-only dynamic normalizer JSON for a held-out target")
    parser.add_argument("--normalization", choices=["none", "log1p_zscore"], default="log1p_zscore")
    parser.add_argument("--history-length", type=int, default=24)
    parser.add_argument("--value-length", type=int, default=24)
    parser.add_argument("--snapshot-stride-hours", type=int, default=24)
    parser.add_argument("--split-ratios", type=float, nargs=3, default=[0.8, 0.1, 0.1])
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "valid", "test"],
        default=["train"],
        help="RAG 契约使用 val；valid 作为兼容别名接受",
    )
    parser.add_argument("--max-snapshots-per-city", type=int, default=None)
    parser.add_argument("--trajectory-chunksize", type=int, default=100_000)
    parser.add_argument("--label-fill-mode", choices=["none", "hour_of_day_prior"], default="none")
    parser.add_argument("--holiday-dates", help="optional JSON list of Asia/Shanghai holiday dates")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    requested_splits = ["val" if split == "valid" else split for split in args.splits]
    requested_splits = list(dict.fromkeys(requested_splits))

    if len(set(args.cities)) != len(args.cities):
        raise ValueError("严格模式: --cities 不能重复")
    if args.max_snapshots_per_city is not None and args.max_snapshots_per_city <= 0:
        raise ValueError("严格模式: --max-snapshots-per-city 必须为正")
    destination = _absolute(args.output)
    if destination.exists() and not args.overwrite:
        raise FileExistsError(f"严格模式: 输出已存在 {destination}; 如确认覆盖请显式使用 --overwrite")
    hierarchy_dir = _absolute(args.hierarchy_cache_dir)
    spectral_dir = _absolute(args.spectral_feature_dir)
    gtg_root = _absolute(args.gtg_data_root)
    region_root = _absolute(args.region_flow_root)
    holidays, holiday_source = _load_holidays(args.holiday_dates)

    hierarchies = {}
    low_features = {}
    graph_metadata = {}
    hourly = {}
    starts = {}
    split_meta = {}
    for city in args.cities:
        hierarchy = load_city_static_hierarchy(hierarchy_dir, city)
        spectral_path = spectral_dir / f"{city}_spectral_features.pt"
        low, identity = load_stage2_low_features(spectral_path, expected_city_id=city)
        metadata = dict(identity)
        metadata.update({
            "road_node_range": tuple(metadata["road_node_range"]),
            "syntax_node_range": tuple(metadata["syntax_node_range"]),
            "region_node_range": tuple(metadata["region_node_range"]),
        })
        trajectories = [gtg_root / city / "traj" / "train.csv", gtg_root / city / "traj" / "test.csv"]
        labels = [gtg_root / city / "traj" / "train_label.csv", gtg_root / city / "traj" / "valid_label.csv"]
        hourly_city = build_hourly_three_layer_dynamics(
            hierarchy=hierarchy,
            road_csv=gtg_root / city / "map" / "road.csv",
            trajectory_paths=trajectories,
            region_hourly_flow=region_root / city / args.region_flow_file_name,
            chunksize=args.trajectory_chunksize,
            label_paths=labels if args.label_fill_mode != "none" else None,
            label_fill_mode=args.label_fill_mode,
        )
        city_starts, city_split_meta = split_snapshot_starts(
            len(hourly_city.timestamps),
            history_length=args.history_length,
            value_length=args.value_length,
            stride_hours=args.snapshot_stride_hours,
            split_ratios=args.split_ratios,
        )
        hierarchies[city] = hierarchy
        low_features[city] = low
        graph_metadata[city] = metadata
        hourly[city] = hourly_city
        starts[city] = city_starts
        split_meta[city] = city_split_meta

    train_end_indices = {city: int(split_meta[city]["train_end_hour_index"]) for city in args.cities}
    normalized, normalizer = normalize_hourly_dynamics(
        hourly,
        train_end_indices=train_end_indices,
        mode=args.normalization,
        normalizer=_normalizer_from_path(args.normalizer_in),
    )
    normalizer_path = _absolute(args.normalizer_out) if args.normalizer_out else destination.with_suffix(".normalizer.json")
    if normalizer_path.exists() and not args.overwrite:
        raise FileExistsError(f"严格模式: normalizer 已存在 {normalizer_path}; 如确认覆盖请显式使用 --overwrite")

    snapshots: list[ThreeLayerRAGInputs] = []
    per_city_counts = {}
    for city in args.cities:
        selected = []
        for split in requested_splits:
            selected.extend((split, start) for start in starts[city][split])
        selected.sort(key=lambda item: item[1])
        if args.max_snapshots_per_city is not None:
            selected = selected[:args.max_snapshots_per_city]
        count_by_split = {split: 0 for split in requested_splits}
        for split, start in selected:
            history = {}
            value = {}
            for layer, values in normalized[city].values.items():
                history[layer], value[layer] = temporal_window(
                    values,
                    start=start,
                    history_length=args.history_length,
                    value_length=args.value_length,
                )
            snapshot = ThreeLayerRAGInputs(
                city_id=city,
                split=split,
                low_features=low_features[city],
                temporal_features=history,
                value_temporal_features=value,
                calendar=calendar_for_timestamp(normalized[city].timestamps[start], holiday_dates=holidays),
                parent_operator=parent_operators_from_hierarchy(hierarchies[city]),
                graph_metadata=graph_metadata[city],
            ).validate()
            snapshots.append(snapshot)
            count_by_split[split] += 1
        per_city_counts[city] = count_by_split

    if not snapshots:
        raise ValueError("严格模式: 按当前 splits/窗口设置未产生任何 snapshot")
    destination.parent.mkdir(parents=True, exist_ok=True)
    normalizer_path.parent.mkdir(parents=True, exist_ok=True)
    with normalizer_path.open("w", encoding="utf-8") as handle:
        json.dump(normalizer, handle, ensure_ascii=False, indent=2)
    torch.save({
        "format_version": "three-layer-rag-snapshot-bundle-v1",
        "snapshots": snapshots,
        "metadata": {
            "dynamic_feature_version": "three-layer-rag-dynamics-v1",
            "cities": list(args.cities),
            "per_city_snapshot_counts": per_city_counts,
            "history_length": args.history_length,
            "value_length": args.value_length,
            "snapshot_stride_hours": args.snapshot_stride_hours,
            "split_ratios": list(args.split_ratios),
            "split_boundaries": split_meta,
            "normalization": normalizer,
            "holiday_source": holiday_source,
            "label_fill_mode": args.label_fill_mode,
            "region_flow_file_name": args.region_flow_file_name,
        },
    }, destination)
    print(json.dumps({
        "output": str(destination),
        "normalizer": str(normalizer_path),
        "num_snapshots": len(snapshots),
        "per_city_snapshot_counts": per_city_counts,
        "splits": requested_splits,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
