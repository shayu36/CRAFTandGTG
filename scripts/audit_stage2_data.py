#!/usr/bin/env python3
"""只读审计 CRAFT 四城与 GTG-main 动态资产，输出 JSON 到 Paper cache。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def audit_city(craft_root: Path, cache_root: Path, city: str):
    city_dir = craft_root / city
    region = pd.read_csv(city_dir / "grid_region_feature.csv")
    relation = pd.read_csv(city_dir / "grid_region_rel.csv")
    roads = pd.read_csv(city_dir / "road.csv")
    oneway = roads["oneway"].astype(str).str.lower().isin(["true", "1", "yes"])
    source = np.r_[roads["from_node_id"].values, roads.loc[~oneway, "to_node_id"].values]
    target = np.r_[roads["to_node_id"].values, roads.loc[~oneway, "from_node_id"].values]
    starts, ends = pd.Series(source).value_counts(), pd.Series(target).value_counts()
    road_edges = sum(int(ends.get(node, 0)) * int(count) for node, count in starts.items())
    macro = {}
    for split in ("train", "test"):
        frame = pd.read_csv(
            city_dir / f"slide_bike_flow_{split}.csv",
            usecols=["region_id", "date", "start_hour"],
        )
        macro[split] = {
            "rows": len(frame), "observed_regions": int(frame["region_id"].nunique()),
            "date_min": str(frame["date"].min()), "date_max": str(frame["date"].max()),
            "start_hour_min": int(frame["start_hour"].min()),
            "start_hour_max": int(frame["start_hour"].max()),
            "unobserved_region_ratio": 1.0 - frame["region_id"].nunique() / len(region),
        }
    with open(city_dir / "data_feature.json") as handle:
        geographic = json.load(handle)
    with open(cache_root / f"{city}_gtg_meta.json") as handle:
        gtg = json.load(handle)
    trip_columns = pd.read_csv(city_dir / "bike_trip.csv", nrows=0).columns.tolist()
    has_path = any(column in trip_columns for column in ("rid_list", "directed_road_id", "trajectory_id"))
    return {
        "city_id": city, "regions": len(region),
        "region_edges": int((relation["is_adj"] == 1).sum()),
        "source_roads": len(roads), "directed_roads": len(source),
        "directed_road_edges": int(road_edges),
        "road_source_crs": "EPSG:4326", "metric_crs": f"EPSG:{geographic['utm_epsg']}",
        "road_null_cells": int(roads.isna().sum().sum()), "macro": macro,
        "stage1_coverage": gtg["coverage"],
        "trajectory_columns": trip_columns, "has_map_matched_path": has_path,
        "micro_flow_available": False, "micro_time_range": None,
        "joint_windows": 0, "road_flow_missing_ratio": 1.0,
        "trajectory_match_rate": None, "timezone": None, "dst_policy": None,
    }


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--craft-root", type=Path, default=root.parent / "CRAFT" / "cleared_data")
    parser.add_argument("--gtg-cache", type=Path, default=root / "cache" / "gtg")
    parser.add_argument("--cities", nargs="+", default=["chi", "dc", "toronto", "ny"])
    parser.add_argument("--output", type=Path, default=root / "cache" / "hcfm" / "stage2_data_audit.json")
    args = parser.parse_args()
    report = {
        "data_version": "stage2-audit-v1", "cities": [
            audit_city(args.craft_root, args.gtg_cache, city) for city in args.cities
        ],
        "gtg_main_cities": ["beijing", "chengdushi", "xianshi"],
        "gtg_can_join_craft": False,
        "blocker": "CRAFT has OD endpoints only; same-city directed-road passage counts are absent",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=2)
    print(args.output)


if __name__ == "__main__":
    main()

