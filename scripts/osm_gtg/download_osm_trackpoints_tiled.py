import argparse
import csv
import json
import math
import re
import time
from pathlib import Path
from xml.etree import ElementTree as ET

import requests


CRAFT_ROOT = Path("/root/autodl-tmp/projects/CRAFT/cleared_data")
OUT_ROOT = Path("/root/autodl-tmp/projects/Paper/data/osm_gtg")

API_URL = "https://api.openstreetmap.org/api/0.6/trackpoints"

HEADERS = {
    "User-Agent": (
        "CRAFTandGTG-research/1.0 "
        "(https://github.com/shayu36/CRAFTandGTG)"
    )
}

# 初始块大小。
INITIAL_STEP = 0.02

# 503 后最小允许切到这个尺度。
MIN_STEP = 0.01

# 每次成功请求后主动休息，避免请求过密。
REQUEST_SLEEP = 1.2

# 单请求超时。
REQUEST_TIMEOUT = 90


def local_name(tag):
    return tag.split("}")[-1]


def count_points(xml_bytes):
    try:
        root = ET.fromstring(xml_bytes)
    except Exception:
        return -1

    n = 0

    for elem in root.iter():
        if local_name(elem.tag) == "trkpt":
            n += 1

    return n


def extract_tracks(xml_bytes):
    """
    提取页面中的轨迹元信息。
    同一个 trace 可能在多个 bbox / page 中重复出现，
    最后按 trace_id 去重。
    """
    root = ET.fromstring(xml_bytes)

    records = []

    for trk in root.iter():
        if local_name(trk.tag) != "trk":
            continue

        name = ""
        desc = ""
        url = ""

        point_count = 0
        timestamp_count = 0

        for elem in trk.iter():
            tag = local_name(elem.tag)

            if elem is not trk:
                if tag == "name" and not name:
                    name = elem.text or ""

                elif tag == "desc" and not desc:
                    desc = elem.text or ""

                elif tag == "url" and not url:
                    url = elem.text or ""

                elif tag == "trkpt":
                    point_count += 1

                    for child in elem:
                        if (
                            local_name(child.tag) == "time"
                            and child.text
                        ):
                            timestamp_count += 1
                            break

        trace_id = None

        # 常见形式：
        # /user/xxx/traces/123456
        # 或其他包含 /traces/123456 的 URL
        m = re.search(r"/traces/(\d+)", url)

        if m:
            trace_id = int(m.group(1))

        records.append({
            "trace_id": trace_id,
            "name": name,
            "description": desc,
            "url": url,
            "points_in_page": point_count,
            "timestamp_points_in_page": timestamp_count,
        })

    return records


def bbox_key(bbox):
    left, bottom, right, top = bbox

    return (
        f"{left:.6f}_{bottom:.6f}_"
        f"{right:.6f}_{top:.6f}"
    )


def make_initial_tiles(bounds, step):
    left, bottom, right, top = bounds

    tiles = []

    x = left

    while x < right - 1e-12:
        x2 = min(x + step, right)

        y = bottom

        while y < top - 1e-12:
            y2 = min(y + step, top)

            tiles.append(
                (
                    round(x, 9),
                    round(y, 9),
                    round(x2, 9),
                    round(y2, 9),
                )
            )

            y = y2

        x = x2

    return tiles


def split_bbox(bbox):
    left, bottom, right, top = bbox

    mid_x = (left + right) / 2
    mid_y = (bottom + top) / 2

    return [
        (left, bottom, mid_x, mid_y),
        (mid_x, bottom, right, mid_y),
        (left, mid_y, mid_x, top),
        (mid_x, mid_y, right, top),
    ]


def request_page(bbox, page):
    left, bottom, right, top = bbox

    params = {
        "bbox": f"{left},{bottom},{right},{top}",
        "page": page,
    }

    try:
        r = requests.get(
            API_URL,
            params=params,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as e:
        return {
            "status": "network_error",
            "error": str(e),
            "http": None,
            "content": b"",
        }

    return {
        "status": "ok",
        "error": None,
        "http": r.status_code,
        "content": r.content,
    }


def save_manifest(trace_map, manifest_path):
    fields = [
        "trace_id",
        "name",
        "description",
        "url",
        "points_seen",
        "timestamp_points_seen",
    ]

    with open(
        manifest_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        def sort_key(x):
            return (
                x is None,
                x if x is not None else -1,
            )

        for trace_id in sorted(
            trace_map,
            key=sort_key,
        ):
            writer.writerow(
                trace_map[trace_id]
            )


def load_bounds(city):
    meta_path = (
        CRAFT_ROOT
        / city
        / "data_feature.json"
    )

    with open(meta_path) as f:
        meta = json.load(f)

    return (
        float(meta["min_lon"]),
        float(meta["min_lat"]),
        float(meta["max_lon"]),
        float(meta["max_lat"]),
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--city",
        default="chi",
        choices=[
            "chi",
            "dc",
            "ny",
            "toronto",
        ],
    )

    parser.add_argument(
        "--initial-step",
        type=float,
        default=INITIAL_STEP,
    )

    parser.add_argument(
        "--min-step",
        type=float,
        default=MIN_STEP,
    )

    args = parser.parse_args()

    city = args.city

    root = OUT_ROOT / city / "traj"

    raw_dir = root / "raw_pages"
    audit_dir = root / "audit"

    raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    bounds = load_bounds(city)

    initial_tiles = make_initial_tiles(
        bounds,
        args.initial_step,
    )

    queue = list(initial_tiles)

    trace_map = {}

    # 用特殊 key 保存没有 trace_id 的轨迹，
    # 防止全部被 None 覆盖。
    anonymous_counter = -1

    request_count = 0
    success_pages = 0
    skipped_existing = 0
    failed_tiles = 0
    split_count = 0
    total_points = 0

    completed_tiles = []

    print("=" * 90)
    print(f"CITY         : {city}")
    print(f"CRAFT BBOX   : {bounds}")
    print(f"INITIAL STEP : {args.initial_step}")
    print(f"MIN STEP     : {args.min_step}")
    print(f"INITIAL TILES: {len(initial_tiles)}")
    print("=" * 90)

    tile_serial = 0

    while queue:
        bbox = queue.pop(0)

        left, bottom, right, top = bbox

        width = right - left
        height = top - bottom

        tile_serial += 1

        key = bbox_key(bbox)

        print()
        print(
            f"[TILE {tile_serial}] "
            f"queue={len(queue)} "
            f"bbox={bbox} "
            f"size={width:.5f}x{height:.5f}"
        )

        page = 0
        tile_success = True

        while True:
            outfile = (
                raw_dir
                / f"{key}_page_{page:04d}.gpx"
            )

            # -------------------------------------------------
            # 断点续跑：
            # 已有且非空文件直接解析，不重新请求。
            # -------------------------------------------------
            if (
                outfile.exists()
                and outfile.stat().st_size > 100
            ):
                content = outfile.read_bytes()
                points = count_points(content)

                if points >= 0:
                    print(
                        f"  [CACHE] page={page} "
                        f"points={points}"
                    )

                    skipped_existing += 1

                    tracks = extract_tracks(content)

                    for tr in tracks:
                        trace_id = tr["trace_id"]

                        if trace_id is None:
                            trace_key = anonymous_counter
                            anonymous_counter -= 1
                        else:
                            trace_key = trace_id

                        if trace_key not in trace_map:
                            trace_map[trace_key] = {
                                "trace_id": trace_id,
                                "name": tr["name"],
                                "description": tr["description"],
                                "url": tr["url"],
                                "points_seen": 0,
                                "timestamp_points_seen": 0,
                            }

                        trace_map[trace_key]["points_seen"] += (
                            tr["points_in_page"]
                        )

                        trace_map[trace_key][
                            "timestamp_points_seen"
                        ] += tr["timestamp_points_in_page"]

                    total_points += points
                    success_pages += 1

                    if points < 5000:
                        break

                    page += 1
                    continue

            # -------------------------------------------------
            # 请求 OSM
            # -------------------------------------------------
            result = request_page(
                bbox,
                page,
            )

            request_count += 1

            http = result["http"]

            print(
                f"  [HTTP] page={page} "
                f"status={http}"
            )

            # -------------------------------------------------
            # 成功
            # -------------------------------------------------
            if http == 200:
                content = result["content"]
                points = count_points(content)

                if points < 0:
                    print(
                        "  [ERROR] 返回内容不是有效 GPX"
                    )
                    tile_success = False
                    break

                outfile.write_bytes(
                    content
                )

                print(
                    f"  [SAVE] points={points} "
                    f"bytes={len(content):,}"
                )

                tracks = extract_tracks(
                    content
                )

                for tr in tracks:
                    trace_id = tr["trace_id"]

                    if trace_id is None:
                        trace_key = anonymous_counter
                        anonymous_counter -= 1
                    else:
                        trace_key = trace_id

                    if trace_key not in trace_map:
                        trace_map[trace_key] = {
                            "trace_id": trace_id,
                            "name": tr["name"],
                            "description": tr["description"],
                            "url": tr["url"],
                            "points_seen": 0,
                            "timestamp_points_seen": 0,
                        }

                    trace_map[trace_key]["points_seen"] += (
                        tr["points_in_page"]
                    )

                    trace_map[trace_key][
                        "timestamp_points_seen"
                    ] += tr["timestamp_points_in_page"]

                total_points += points
                success_pages += 1

                # 一页少于 5000 点：
                # 当前 bbox 已无更多分页。
                if points < 5000:
                    break

                page += 1

                time.sleep(
                    REQUEST_SLEEP
                )

                continue

            # -------------------------------------------------
            # 503：
            # 当前 bbox 查询太重。
            #
            # 注意：
            # 如果 page>0 已经成功过，不能简单拆分然后保留
            # 当前 bbox 的前几页，否则会重复。
            #
            # 因此该 bbox 的已保存页面会删除，
            # 然后从四个子 bbox 重新开始。
            # -------------------------------------------------
            if http == 503:
                can_split = (
                    width / 2 >= args.min_step - 1e-9
                    and height / 2 >= args.min_step - 1e-9
                )

                if can_split:
                    print(
                        "  [503] bbox 太重，自动四等分"
                    )

                    for old in raw_dir.glob(
                        f"{key}_page_*.gpx"
                    ):
                        old.unlink()

                    children = split_bbox(
                        bbox
                    )

                    # 插到队头，尽快处理当前区域
                    queue = children + queue

                    split_count += 1
                    tile_success = False

                    break

                print(
                    "  [503] 已达到最小 bbox，"
                    "仍然无法获取"
                )

                failed_tiles += 1
                tile_success = False

                break

            # -------------------------------------------------
            # 429：限流
            # -------------------------------------------------
            if http == 429:
                print(
                    "  [429] 限流，等待 60 秒后重试"
                )

                time.sleep(60)
                continue

            # -------------------------------------------------
            # 其他 5xx
            # -------------------------------------------------
            if (
                http is not None
                and http >= 500
            ):
                print(
                    f"  [{http}] 服务端异常，"
                    "等待 20 秒重试一次"
                )

                time.sleep(20)

                result2 = request_page(
                    bbox,
                    page,
                )

                request_count += 1

                if result2["http"] == 200:
                    content = result2["content"]
                    points = count_points(
                        content
                    )

                    if points >= 0:
                        outfile.write_bytes(
                            content
                        )

                        print(
                            f"  [RECOVERED] "
                            f"points={points}"
                        )

                        tracks = extract_tracks(
                            content
                        )

                        for tr in tracks:
                            trace_id = tr["trace_id"]

                            if trace_id is None:
                                trace_key = anonymous_counter
                                anonymous_counter -= 1
                            else:
                                trace_key = trace_id

                            if trace_key not in trace_map:
                                trace_map[trace_key] = {
                                    "trace_id": trace_id,
                                    "name": tr["name"],
                                    "description": tr["description"],
                                    "url": tr["url"],
                                    "points_seen": 0,
                                    "timestamp_points_seen": 0,
                                }

                            trace_map[trace_key][
                                "points_seen"
                            ] += tr["points_in_page"]

                            trace_map[trace_key][
                                "timestamp_points_seen"
                            ] += tr[
                                "timestamp_points_in_page"
                            ]

                        total_points += points
                        success_pages += 1

                        if points < 5000:
                            break

                        page += 1
                        continue

                tile_success = False
                failed_tiles += 1
                break

            # -------------------------------------------------
            # 网络错误 / 其他 HTTP
            # -------------------------------------------------
            print(
                "  [ERROR]",
                result["error"]
                or f"HTTP {http}"
            )

            tile_success = False
            failed_tiles += 1
            break

        if tile_success:
            completed_tiles.append(
                bbox
            )

        # 每处理完一个 bbox 都写一次 manifest，
        # 防止中断后信息全丢。
        manifest_path = (
            audit_dir
            / "trace_manifest.csv"
        )

        save_manifest(
            trace_map,
            manifest_path,
        )

        time.sleep(
            REQUEST_SLEEP
        )

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------
    valid_trace_ids = {
        v["trace_id"]
        for v in trace_map.values()
        if v["trace_id"] is not None
    }

    summary = {
        "city": city,
        "craft_bbox": list(bounds),
        "initial_step": args.initial_step,
        "min_step": args.min_step,
        "initial_tiles": len(initial_tiles),
        "split_count": split_count,
        "completed_tiles": len(completed_tiles),
        "failed_tiles": failed_tiles,
        "http_requests": request_count,
        "successful_pages": success_pages,
        "cached_pages": skipped_existing,
        "total_points_seen": total_points,
        "unique_trace_ids": len(valid_trace_ids),
        "manifest": str(
            audit_dir
            / "trace_manifest.csv"
        ),
    }

    summary_path = (
        audit_dir
        / "download_summary.json"
    )

    with open(
        summary_path,
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("=" * 90)
    print("FINAL SUMMARY")
    print("=" * 90)

    for k, v in summary.items():
        print(
            f"{k:22s}: {v}"
        )

    print()
    print("[OUTPUT]")
    print(
        audit_dir
        / "trace_manifest.csv"
    )
    print(summary_path)


if __name__ == "__main__":
    main()
