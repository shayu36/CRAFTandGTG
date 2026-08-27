import argparse
import json
import math
import re
import time
from pathlib import Path
from datetime import datetime
from xml.etree import ElementTree as ET

import pandas as pd
import requests


CRAFT_ROOT = Path("/root/autodl-tmp/projects/CRAFT/cleared_data")
OUT_ROOT = Path("/root/autodl-tmp/projects/Paper/data/osm_gtg")

API_URL = "https://api.openstreetmap.org/api/0.6/trackpoints"

HEADERS = {
    "User-Agent":
        "CRAFTandGTG-research/1.0 "
        "(https://github.com/shayu36/CRAFTandGTG)"
}


def local_name(tag):
    return tag.split("}")[-1]


def parse_time(text):
    if not text:
        return None
    try:
        return datetime.fromisoformat(
            text.replace("Z", "+00:00")
        )
    except Exception:
        return None


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1)
        * math.cos(phi2)
        * math.sin(dlambda / 2) ** 2
    )

    return 2 * R * math.asin(math.sqrt(a))


def request_page(city, page):
    meta_path = CRAFT_ROOT / city / "data_feature.json"

    with open(meta_path, "r") as f:
        meta = json.load(f)

    left = float(meta["min_lon"])
    bottom = float(meta["min_lat"])
    right = float(meta["max_lon"])
    top = float(meta["max_lat"])

    width = right - left
    height = top - bottom

    print("=" * 80)
    print(f"CITY   : {city}")
    print(f"BBOX   : {left},{bottom},{right},{top}")
    print(f"WIDTH  : {width:.6f} deg")
    print(f"HEIGHT : {height:.6f} deg")
    print(f"PAGE   : {page}")

    if width > 0.25 or height > 0.25:
        raise RuntimeError(
            "CRAFT bbox 超过 OSM trackpoints API 0.25° 限制"
        )

    params = {
        "bbox": f"{left},{bottom},{right},{top}",
        "page": page,
    }

    delay = 2

    for attempt in range(6):
        try:
            r = requests.get(
                API_URL,
                params=params,
                headers=HEADERS,
                timeout=120,
            )

            print(
                f"[HTTP] status={r.status_code}, "
                f"bytes={len(r.content):,}"
            )

            if r.status_code == 200:
                return r.content, {
                    "left": left,
                    "bottom": bottom,
                    "right": right,
                    "top": top,
                    "width": width,
                    "height": height,
                }

            if r.status_code == 429 or r.status_code >= 500:
                print(
                    f"[RETRY] sleep {delay}s "
                    f"(attempt {attempt + 1}/6)"
                )
                time.sleep(delay)
                delay *= 2
                continue

            r.raise_for_status()

        except requests.RequestException as e:
            if attempt == 5:
                raise

            print(
                f"[REQUEST ERROR] {e}; "
                f"sleep {delay}s"
            )
            time.sleep(delay)
            delay *= 2

    raise RuntimeError("OSM 请求失败")


def parse_gpx(xml_bytes):
    root = ET.fromstring(xml_bytes)

    records = []

    total_points = 0
    total_segments = 0

    tracks = [
        x for x in root
        if local_name(x.tag) == "trk"
    ]

    for trk_idx, trk in enumerate(tracks):
        name = ""
        desc = ""
        url = ""

        segments = []

        for child in trk:
            tag = local_name(child.tag)

            if tag == "name":
                name = child.text or ""

            elif tag == "desc":
                desc = child.text or ""

            elif tag == "url":
                url = child.text or ""

            elif tag == "trkseg":
                segments.append(child)

        trace_id = None

        m = re.search(r"/traces/(\d+)", url)

        if m:
            trace_id = int(m.group(1))

        point_count = 0
        timestamp_count = 0

        all_points = []

        for seg in segments:
            total_segments += 1

            for pt in seg:
                if local_name(pt.tag) != "trkpt":
                    continue

                try:
                    lat = float(pt.attrib["lat"])
                    lon = float(pt.attrib["lon"])
                except Exception:
                    continue

                timestamp = None

                for c in pt:
                    if local_name(c.tag) == "time":
                        timestamp = parse_time(c.text)
                        break

                point_count += 1
                total_points += 1

                if timestamp is not None:
                    timestamp_count += 1

                all_points.append(
                    (lat, lon, timestamp)
                )

        # ----------------------------------------------------
        # 根据当前返回顺序计算轨迹基本统计
        # ----------------------------------------------------
        total_distance_m = 0.0
        speeds_kmh = []

        for i in range(1, len(all_points)):
            lat1, lon1, t1 = all_points[i - 1]
            lat2, lon2, t2 = all_points[i]

            d = haversine_m(
                lat1, lon1,
                lat2, lon2
            )

            total_distance_m += d

            if t1 is not None and t2 is not None:
                dt = (t2 - t1).total_seconds()

                if dt > 0:
                    speed = d / dt * 3.6

                    if math.isfinite(speed):
                        speeds_kmh.append(speed)

        valid_times = [
            p[2]
            for p in all_points
            if p[2] is not None
        ]

        duration_sec = None

        if len(valid_times) >= 2:
            duration_sec = (
                max(valid_times)
                - min(valid_times)
            ).total_seconds()

        timestamp_ratio = (
            timestamp_count / point_count
            if point_count > 0
            else 0
        )

        median_speed = None
        max_speed = None

        if speeds_kmh:
            s = sorted(speeds_kmh)

            median_speed = s[len(s) // 2]
            max_speed = max(speeds_kmh)

        # ----------------------------------------------------
        # 这里仅判断能否作为“候选顺序轨迹”
        # 不在这里判断是不是汽车
        # ----------------------------------------------------
        sequence_candidate = (
            trace_id is not None
            and point_count >= 10
            and timestamp_ratio >= 0.8
            and duration_sec is not None
            and duration_sec > 0
        )

        records.append({
            "trk_index": trk_idx,
            "trace_id": trace_id,
            "name": name,
            "description": desc,
            "url": url,
            "num_segments": len(segments),
            "num_points": point_count,
            "timestamp_points": timestamp_count,
            "timestamp_ratio": timestamp_ratio,
            "duration_sec": duration_sec,
            "distance_km_raw": (
                total_distance_m / 1000.0
            ),
            "median_speed_kmh_raw": median_speed,
            "max_speed_kmh_raw": max_speed,
            "sequence_candidate": sequence_candidate,
        })

    return records, {
        "num_tracks": len(tracks),
        "num_segments": total_segments,
        "num_points": total_points,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--city",
        default="chi",
        choices=["chi", "dc", "ny", "toronto"],
    )

    parser.add_argument(
        "--page",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    city = args.city
    page = args.page

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

    xml_bytes, bbox = request_page(
        city,
        page,
    )

    raw_path = (
        raw_dir
        / f"trackpoints_page_{page:04d}.gpx"
    )

    raw_path.write_bytes(xml_bytes)

    records, stats = parse_gpx(
        xml_bytes
    )

    df = pd.DataFrame(records)

    csv_path = (
        audit_dir
        / f"trackpoints_page_{page:04d}_audit.csv"
    )

    df.to_csv(
        csv_path,
        index=False,
    )

    candidate_num = (
        int(df["sequence_candidate"].sum())
        if len(df)
        and "sequence_candidate" in df.columns
        else 0
    )

    trace_id_num = (
        int(df["trace_id"].notna().sum())
        if len(df)
        and "trace_id" in df.columns
        else 0
    )

    points_with_time = (
        int(df["timestamp_points"].sum())
        if len(df)
        else 0
    )

    summary = {
        "city": city,
        "page": page,
        "bbox": bbox,
        **stats,
        "tracks_with_trace_id": trace_id_num,
        "sequence_candidates": candidate_num,
        "points_with_timestamp": points_with_time,
        "timestamp_point_ratio": (
            points_with_time / stats["num_points"]
            if stats["num_points"]
            else 0
        ),
        "raw_gpx": str(raw_path),
        "audit_csv": str(csv_path),
    }

    json_path = (
        audit_dir
        / f"trackpoints_page_{page:04d}_summary.json"
    )

    with open(
        json_path,
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("=" * 80)
    print("OSM GPS PAGE AUDIT")
    print("=" * 80)

    for k, v in summary.items():
        print(f"{k:24s}: {v}")

    print()
    print("=" * 80)
    print("TRACK SAMPLE")
    print("=" * 80)

    if len(df):
        cols = [
            "trk_index",
            "trace_id",
            "num_segments",
            "num_points",
            "timestamp_ratio",
            "duration_sec",
            "distance_km_raw",
            "median_speed_kmh_raw",
            "sequence_candidate",
            "name",
        ]

        cols = [
            c for c in cols
            if c in df.columns
        ]

        print(
            df[cols]
            .head(30)
            .to_string(index=False)
        )
    else:
        print("No tracks returned.")

    print()
    print("[OUTPUT]")
    print(raw_path)
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
