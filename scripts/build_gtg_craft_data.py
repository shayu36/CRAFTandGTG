"""构建 GTG 三城的 CRAFT 兼容车辆边界穿越流量与静态数据。"""

import argparse
import json
import os
import sys
from os.path import join

sys.path.insert(0, join(os.path.dirname(__file__), "..", "src"))

from gtg_preprocessing.pipeline import run  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=join(os.path.dirname(__file__), "..", "configs", "gtg_craft_preprocess.yaml"),
    )
    parser.add_argument("--cities", nargs="*", default=None)
    args = parser.parse_args()
    reports = run(args.config, cities=args.cities)
    print(
        json.dumps(
            [
                {
                    "city": report["city"],
                    "regions": report["grid"]["selected_grid_count"],
                    "train_windows": report["windows"]["train_windows_after_positive_filter"],
                    "validation_windows": report["windows"]["validation_windows_after_positive_filter"],
                }
                for report in reports
            ],
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
