"""预处理入口: 为各城市构建 region 级 GTG 拓扑特征并缓存。

用法:
  python scripts/build_gtg_features.py --cities chi dc toronto ny
只读 CRAFT/cleared_data; 写 Paper/cache/gtg。
"""
import argparse
import json
import os
import sys
from os.path import join

# 保证可 import Paper/src/gtg_features
sys.path.insert(0, join(os.path.dirname(__file__), "..", "src"))

from gtg_features.pipeline import build_city  # noqa: E402

CITIES = ["chi", "dc", "toronto", "ny"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--craft_root", default="/root/autodl-tmp/projects/CRAFT/cleared_data")
    ap.add_argument("--cache_dir", default="/root/autodl-tmp/projects/Paper/cache/gtg")
    ap.add_argument("--local_size", type=int, default=50)
    ap.add_argument("--cities", nargs="*", default=CITIES)
    args = ap.parse_args()

    metas = []
    for city in args.cities:
        meta = build_city(city, args.craft_root, args.cache_dir, local_size=args.local_size)
        metas.append(meta)

    summary_pth = join(args.cache_dir, "gtg_build_summary.json")
    with open(summary_pth, "w") as f:
        json.dump(metas, f, indent=2)
    print(f"[build_gtg_features] done. summary -> {summary_pth}", file=sys.stderr)


if __name__ == "__main__":
    main()
