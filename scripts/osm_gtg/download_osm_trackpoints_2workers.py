import argparse
import json
import math
import os
import queue
import threading
import time
from pathlib import Path
from xml.etree import ElementTree as ET

import requests


CRAFT_ROOT = Path("/root/autodl-tmp/projects/CRAFT/cleared_data")
OUT_ROOT = Path("/root/autodl-tmp/projects/Paper/data/osm_gtg")

API_URL = "https://api.openstreetmap.org/api/0.6/trackpoints"

USER_AGENT = (
    "CRAFTandGTG-research/1.0 "
    "(https://github.com/shayu36/CRAFTandGTG)"
)

# OSM 官方要求最多 2 download threads。
MAX_WORKERS = 2

INITIAL_STEP = 0.02
MIN_STEP = 0.005

# 每个 worker 每次请求后主动停顿。
REQUEST_DELAY = 1.5

CONNECT_TIMEOUT = 15
READ_TIMEOUT = 120

# page > 0 时遇到 503，不切 bbox，只对当前 page 重试。
MAX_PAGE_RETRIES = 10

# page == 0 时连续 503 后允许拆 bbox。
MAX_PAGE0_RETRIES = 4


print_lock = threading.Lock()
stats_lock = threading.Lock()

stats = {
    "requests": 0,
    "http_200": 0,
    "http_429": 0,
    "http_503": 0,
    "other_errors": 0,
    "cached_pages": 0,
    "downloaded_pages": 0,
    "downloaded_points": 0,
    "completed_tiles": 0,
    "split_tiles": 0,
    "failed_tiles": 0,
}


def log(msg):
    with print_lock:
        print(msg, flush=True)


def inc(key, n=1):
    with stats_lock:
        stats[key] += n


def local_name(tag):
    return tag.split("}")[-1]


def count_points(content):
    try:
        root = ET.fromstring(content)
    except Exception:
        return -1

    return sum(
        1
        for elem in root.iter()
        if local_name(elem.tag) == "trkpt"
    )


def load_bounds(city):
    p = CRAFT_ROOT / city / "data_feature.json"

    with open(p) as f:
        meta = json.load(f)

    return (
        float(meta["min_lon"]),
        float(meta["min_lat"]),
        float(meta["max_lon"]),
        float(meta["max_lat"]),
    )


def make_tiles(bounds, step):
    left, bottom, right, top = bounds

    tiles = []

    x = left

    while x < right - 1e-12:
        x2 = min(x + step, right)

        y = bottom

        while y < top - 1e-12:
            y2 = min(y + step, top)

            tiles.append((
                round(x, 9),
                round(y, 9),
                round(x2, 9),
                round(y2, 9),
            ))

            y = y2

        x = x2

    return tiles


def split_bbox(bbox):
    left, bottom, right, top = bbox

    mx = (left + right) / 2
    my = (bottom + top) / 2

    return [
        (left, bottom, mx, my),
        (mx, bottom, right, my),
        (left, my, mx, top),
        (mx, my, right, top),
    ]


def bbox_key(bbox):
    l, b, r, t = bbox

    return (
        f"{l:.6f}_{b:.6f}_"
        f"{r:.6f}_{t:.6f}"
    )


def request_page(session, bbox, page):
    l, b, r, t = bbox

    params = {
        "bbox": f"{l},{b},{r},{t}",
        "page": page,
    }

    try:
        resp = session.get(
            API_URL,
            params=params,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
        )

        inc("requests")

        return resp

    except requests.RequestException as e:
        inc("requests")
        inc("other_errors")
        return e


def process_tile(worker_id, city, bbox, task_queue, raw_dir):
    key = bbox_key(bbox)

    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT
    })

    log(
        f"[W{worker_id}] START "
        f"bbox={bbox} "
        f"size={width:.5f}x{height:.5f}"
    )

    page = 0

    while True:
        outfile = (
            raw_dir
            / f"{key}_page_{page:04d}.gpx"
        )

        # ------------------------------------------------------
        # CACHE：已有页面直接使用
        # ------------------------------------------------------
        if (
            outfile.exists()
            and outfile.stat().st_size > 100
        ):
            content = outfile.read_bytes()

            n = count_points(content)

            if n >= 0:
                inc("cached_pages")

                log(
                    f"[W{worker_id}] CACHE "
                    f"page={page} points={n}"
                )

                if n < 5000:
                    inc("completed_tiles")

                    log(
                        f"[W{worker_id}] DONE "
                        f"bbox={bbox} "
                        f"last_page={page}"
                    )

                    return

                page += 1
                continue

            # 损坏缓存删除后重下
            outfile.unlink(missing_ok=True)

        # ------------------------------------------------------
        # 当前 page 下载
        # ------------------------------------------------------
        retry = 0

        while True:
            resp = request_page(
                session,
                bbox,
                page,
            )

            if isinstance(
                resp,
                requests.RequestException
            ):
                retry += 1

                wait = min(
                    15 * (2 ** min(retry, 4)),
                    180
                )

                log(
                    f"[W{worker_id}] NETWORK ERROR "
                    f"page={page} retry={retry} "
                    f"sleep={wait}s: {resp}"
                )

                if retry >= MAX_PAGE_RETRIES:
                    inc("failed_tiles")
                    return

                time.sleep(wait)
                continue

            status = resp.status_code

            # --------------------------------------------------
            # 200
            # --------------------------------------------------
            if status == 200:
                inc("http_200")

                content = resp.content
                n = count_points(content)

                if n < 0:
                    retry += 1

                    log(
                        f"[W{worker_id}] INVALID GPX "
                        f"page={page} retry={retry}"
                    )

                    if retry >= MAX_PAGE_RETRIES:
                        inc("failed_tiles")
                        return

                    time.sleep(15)
                    continue

                tmp = outfile.with_suffix(
                    outfile.suffix + ".tmp"
                )

                tmp.write_bytes(content)
                os.replace(tmp, outfile)

                inc("downloaded_pages")
                inc("downloaded_points", n)

                log(
                    f"[W{worker_id}] SAVE "
                    f"page={page} "
                    f"points={n} "
                    f"bytes={len(content):,}"
                )

                time.sleep(REQUEST_DELAY)

                # 最后一页
                if n < 5000:
                    inc("completed_tiles")

                    log(
                        f"[W{worker_id}] DONE "
                        f"bbox={bbox} "
                        f"last_page={page}"
                    )

                    return

                page += 1
                break

            # --------------------------------------------------
            # 429
            # --------------------------------------------------
            elif status == 429:
                inc("http_429")

                retry_after = resp.headers.get(
                    "Retry-After"
                )

                try:
                    wait = max(
                        120,
                        int(retry_after)
                    )
                except Exception:
                    wait = 120

                log(
                    f"[W{worker_id}] HTTP 429 "
                    f"page={page} "
                    f"sleep={wait}s"
                )

                time.sleep(wait)
                continue

            # --------------------------------------------------
            # 503
            # --------------------------------------------------
            elif status == 503:
                inc("http_503")
                retry += 1

                # page > 0:
                # 说明这个 bbox 前面一直是可查询的。
                # 不切块、不删除历史页面。
                if page > 0:
                    wait = min(
                        10 * (2 ** min(retry - 1, 5)),
                        300
                    )

                    log(
                        f"[W{worker_id}] HTTP 503 "
                        f"page={page} "
                        f"retry={retry}/{MAX_PAGE_RETRIES} "
                        f"sleep={wait}s"
                    )

                    if retry >= MAX_PAGE_RETRIES:
                        log(
                            f"[W{worker_id}] PAUSE TILE "
                            f"page={page}; "
                            f"已有页面全部保留，"
                            f"下次断点续跑"
                        )

                        inc("failed_tiles")
                        return

                    time.sleep(wait)
                    continue

                # page == 0:
                # 连续失败后才允许细分 bbox。
                wait = min(
                    10 * (2 ** (retry - 1)),
                    120
                )

                log(
                    f"[W{worker_id}] HTTP 503 "
                    f"page=0 retry={retry}/"
                    f"{MAX_PAGE0_RETRIES}"
                )

                if retry < MAX_PAGE0_RETRIES:
                    time.sleep(wait)
                    continue

                half_w = width / 2
                half_h = height / 2

                if (
                    half_w >= MIN_STEP - 1e-9
                    and
                    half_h >= MIN_STEP - 1e-9
                ):
                    children = split_bbox(
                        bbox
                    )

                    log(
                        f"[W{worker_id}] SPLIT "
                        f"{bbox} -> 4 children"
                    )

                    inc("split_tiles")

                    for child in children:
                        task_queue.put(child)

                    return

                log(
                    f"[W{worker_id}] FAIL "
                    f"minimum bbox still 503: "
                    f"{bbox}"
                )

                inc("failed_tiles")
                return

            # --------------------------------------------------
            # 其他服务器错误
            # --------------------------------------------------
            elif status >= 500:
                retry += 1

                wait = min(
                    15 * (2 ** min(retry - 1, 4)),
                    180
                )

                log(
                    f"[W{worker_id}] HTTP {status} "
                    f"page={page} "
                    f"retry={retry} "
                    f"sleep={wait}s"
                )

                if retry >= MAX_PAGE_RETRIES:
                    inc("failed_tiles")
                    return

                time.sleep(wait)
                continue

            else:
                inc("other_errors")

                log(
                    f"[W{worker_id}] HTTP {status} "
                    f"page={page}; stop tile"
                )

                inc("failed_tiles")
                return


def worker_loop(
    worker_id,
    city,
    task_queue,
    raw_dir,
):
    while True:
        bbox = task_queue.get()

        try:
            if bbox is None:
                return

            process_tile(
                worker_id,
                city,
                bbox,
                task_queue,
                raw_dir,
            )

        finally:
            task_queue.task_done()


def monitor(task_queue, audit_dir, stop_event):
    start = time.time()

    while not stop_event.wait(60):
        with stats_lock:
            snap = dict(stats)

        elapsed_h = (
            time.time() - start
        ) / 3600

        summary = {
            **snap,
            "queue_unfinished":
                task_queue.unfinished_tasks,
            "elapsed_hours":
                elapsed_h,
        }

        p = (
            audit_dir
            / "parallel_live_summary.json"
        )

        with open(p, "w") as f:
            json.dump(
                summary,
                f,
                indent=2,
            )

        log(
            "\n"
            + "=" * 80
            + "\n[LIVE] "
            + f"elapsed={elapsed_h:.2f}h "
            + f"queue={task_queue.unfinished_tasks} "
            + f"downloaded_pages="
            + f"{snap['downloaded_pages']} "
            + f"cached_pages="
            + f"{snap['cached_pages']} "
            + f"completed_tiles="
            + f"{snap['completed_tiles']} "
            + f"429={snap['http_429']} "
            + f"503={snap['http_503']}"
            + "\n"
            + "=" * 80
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

    args = parser.parse_args()

    city = args.city

    raw_dir = (
        OUT_ROOT
        / city
        / "traj"
        / "raw_pages"
    )

    audit_dir = (
        OUT_ROOT
        / city
        / "traj"
        / "audit"
    )

    raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    bounds = load_bounds(city)

    tiles = make_tiles(
        bounds,
        args.initial_step,
    )

    q = queue.Queue()

    for tile in tiles:
        q.put(tile)

    log("=" * 90)
    log(f"CITY            : {city}")
    log(f"CRAFT BBOX      : {bounds}")
    log(f"INITIAL STEP    : {args.initial_step}")
    log(f"INITIAL TILES   : {len(tiles)}")
    log(f"DOWNLOAD WORKERS: {MAX_WORKERS}")
    log(
        "PAGE LIMIT      : NONE "
        "(完整分页)"
    )
    log(
        "CACHE           : ENABLED "
        "(复用之前下载)"
    )
    log("=" * 90)

    stop_event = threading.Event()

    monitor_thread = threading.Thread(
        target=monitor,
        args=(
            q,
            audit_dir,
            stop_event,
        ),
        daemon=True,
    )

    monitor_thread.start()

    workers = []

    for i in range(MAX_WORKERS):
        t = threading.Thread(
            target=worker_loop,
            args=(
                i,
                city,
                q,
                raw_dir,
            ),
            daemon=True,
        )

        t.start()
        workers.append(t)

    try:
        q.join()

    except KeyboardInterrupt:
        log(
            "\n[INTERRUPTED] "
            "已下载页面全部保留。"
        )

        stop_event.set()
        raise

    stop_event.set()

    # worker 退出
    for _ in workers:
        q.put(None)

    for t in workers:
        t.join()

    with stats_lock:
        final_stats = dict(stats)

    raw_files = list(
        raw_dir.glob("*.gpx")
    )

    total_bytes = sum(
        p.stat().st_size
        for p in raw_files
    )

    final = {
        "city": city,
        "workers": MAX_WORKERS,
        "initial_step": args.initial_step,
        "initial_tiles": len(tiles),
        **final_stats,
        "raw_page_files":
            len(raw_files),
        "raw_page_bytes":
            total_bytes,
        "raw_page_gib":
            total_bytes / (1024 ** 3),
    }

    final_path = (
        audit_dir
        / "parallel_download_summary.json"
    )

    with open(final_path, "w") as f:
        json.dump(
            final,
            f,
            indent=2,
        )

    log("")
    log("=" * 90)
    log("FINAL SUMMARY")
    log("=" * 90)

    for k, v in final.items():
        log(f"{k:24s}: {v}")

    log("")
    log(f"[OUTPUT] {final_path}")


if __name__ == "__main__":
    main()
