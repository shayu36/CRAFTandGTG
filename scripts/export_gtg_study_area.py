"""只用 GTG 合并轨迹导出 2 km 研究格网，供 OSM/WorldPop 裁剪。"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gtg_preprocessing.flow import collect_used_road_ids  # noqa: E402
from gtg_preprocessing.static import build_trajectory_grid, load_gtg_road_geometries  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True, choices=["beijing", "chengdushi", "xianshi"])
    parser.add_argument("--data_root", default="/root/autodl-tmp/projects/Paper/data")
    parser.add_argument("--output_root", default="/root/autodl-tmp/projects/Paper/data/external_2026")
    parser.add_argument("--grid_size_m", type=float, default=2000.0)
    parser.add_argument("--trajectory_chunksize", type=int, default=100000)
    args = parser.parse_args()

    gtg_dir = Path(args.data_root) / args.city
    city_output = Path(args.output_root) / args.city / "study_area"
    if city_output.exists() and any(city_output.iterdir()):
        raise FileExistsError(
            f"严格模式: 输出目录已存在且非空 {city_output}，为避免覆盖请改用新目录"
        )
    road_path = gtg_dir / "map" / "road.csv"
    trajectories = [gtg_dir / "traj" / "train.csv", gtg_dir / "traj" / "test.csv"]
    raw_road, roads_wgs84, utm_epsg = load_gtg_road_geometries(road_path)
    used, usage_meta = collect_used_road_ids(
        trajectories, len(raw_road), chunksize=args.trajectory_chunksize
    )
    full, selected, grid_meta = build_trajectory_grid(
        roads_wgs84, used, utm_epsg, grid_size_m=args.grid_size_m
    )
    city_output.mkdir(parents=True, exist_ok=True)
    full.to_crs(4326).to_file(city_output / "full_grid.geojson", driver="GeoJSON")
    selected.to_crs(4326).to_file(city_output / "selected_grid.geojson", driver="GeoJSON")
    min_lon, min_lat, max_lon, max_lat = full.to_crs(4326).total_bounds
    meta = {
        "city": args.city,
        "overpass_bbox_order": {
            "south": float(min_lat),
            "west": float(min_lon),
            "north": float(max_lat),
            "east": float(max_lon),
        },
        "usage": usage_meta,
        "grid": grid_meta,
    }
    with (city_output / "study_area_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
