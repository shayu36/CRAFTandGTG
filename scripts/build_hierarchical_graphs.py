#!/usr/bin/env python3
"""构建 HCFM 静态层次图缓存；原始 CRAFT/GTG 目录只读。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcfm.preprocessing import build_city_static_graph, save_static_cache  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities", nargs="+", default=["chi", "dc", "toronto", "ny"])
    parser.add_argument("--craft-root", default=str(ROOT.parent / "CRAFT" / "cleared_data"))
    parser.add_argument("--gtg-cache-dir", default=str(ROOT / "cache" / "gtg"))
    parser.add_argument("--output-dir", default=str(ROOT / "cache" / "hcfm"))
    args = parser.parse_args()
    reports = []
    for city in args.cities:
        tensors, manifest = build_city_static_graph(city, args.craft_root, args.gtg_cache_dir)
        npz_path, json_path = save_static_cache(tensors, manifest, args.output_dir)
        reports.append(manifest)
        print(f"[hcfm] {city}: {npz_path} {json_path}", file=sys.stderr)
    summary = Path(args.output_dir) / "hierarchy_build_summary.json"
    with open(summary, "w") as handle:
        json.dump(reports, handle, indent=2)
    print(f"[hcfm] summary: {summary}", file=sys.stderr)


if __name__ == "__main__":
    main()
