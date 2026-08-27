from pathlib import Path
import json

import pandas as pd
import geopandas as gpd
from shapely import wkt


CRAFT_ROOT = Path("/root/autodl-tmp/projects/CRAFT/cleared_data")
OUT_ROOT = Path("/root/autodl-tmp/projects/Paper/data/osm_gtg/audit")

CITIES = ["chi", "dc", "ny", "toronto"]

# GTG prepare.py / dataloader.py 真正需要或后续需要构造的基础字段
GTG_DIRECT_REQUIRED = [
    "from_node_id",
    "to_node_id",
    "length",
    "geometry",
]

CRAFT_EXPECTED = [
    "road_id",
    "from_node_id",
    "to_node_id",
    "road_type",
    "road_type_id",
    "length",
    "geometry",
    "oneway",
    "lanes",
    "maxspeed",
]


def safe_bool_series(s):
    return (
        s.astype(str)
         .str.strip()
         .str.lower()
         .isin(["true", "1", "yes", "t"])
    )


def audit_city(city):
    print()
    print("=" * 90)
    print(f"CITY: {city}")
    print("=" * 90)

    base = CRAFT_ROOT / city
    road_path = base / "road.csv"
    region_path = base / "grid_region.csv"
    feature_path = base / "data_feature.json"

    road = pd.read_csv(road_path)

    print(f"[FILE] {road_path}")
    print(f"[ROWS] {len(road):,}")
    print(f"[COLS] {len(road.columns)}")
    print("[COLUMNS]")
    print(", ".join(road.columns))

    # ---------------------------------------------------------
    # 1. 基础字段
    # ---------------------------------------------------------
    missing_gtg = [c for c in GTG_DIRECT_REQUIRED if c not in road.columns]
    missing_craft = [c for c in CRAFT_EXPECTED if c not in road.columns]

    print()
    print("[1] GTG 基础字段")
    if missing_gtg:
        print("  FAIL missing:", missing_gtg)
    else:
        print("  PASS")

    print()
    print("[2] 当前 CRAFT 预期字段")
    if missing_craft:
        print("  WARN missing:", missing_craft)
    else:
        print("  PASS")

    # ---------------------------------------------------------
    # 2. road_id
    # ---------------------------------------------------------
    print()
    print("[3] road_id 连续性")

    if "road_id" in road.columns:
        ids = road["road_id"]

        unique = ids.nunique() == len(ids)
        integer_like = pd.api.types.is_integer_dtype(ids)

        expected = list(range(len(road)))
        actual = ids.tolist()

        continuous = actual == expected

        print("  unique     :", unique)
        print("  integer    :", integer_like)
        print("  continuous :", continuous)
        print("  min/max    :", ids.min(), ids.max())

        if not continuous:
            print("  NOTE: GTG 转换时重新生成 link_id=0..N-1 即可")
    else:
        unique = False
        continuous = False
        print("  FAIL: road_id missing")

    # ---------------------------------------------------------
    # 3. node IDs
    # ---------------------------------------------------------
    print()
    print("[4] topology IDs")

    node_ok = True

    for c in ["from_node_id", "to_node_id"]:
        if c not in road.columns:
            print(f"  {c}: MISSING")
            node_ok = False
        else:
            null_num = int(road[c].isna().sum())
            print(
                f"  {c}: "
                f"null={null_num:,}, "
                f"unique={road[c].nunique():,}"
            )
            if null_num:
                node_ok = False

    # ---------------------------------------------------------
    # 4. length
    # ---------------------------------------------------------
    print()
    print("[5] road length")

    if "length" in road.columns:
        length = pd.to_numeric(road["length"], errors="coerce")

        null_len = int(length.isna().sum())
        nonpositive = int((length <= 0).sum())

        print("  null       :", f"{null_len:,}")
        print("  <= 0       :", f"{nonpositive:,}")
        print("  mean (m)   :", round(float(length.mean()), 3))
        print("  median (m) :", round(float(length.median()), 3))
        print("  max (m)    :", round(float(length.max()), 3))

        length_ok = null_len == 0 and nonpositive == 0
    else:
        length_ok = False
        print("  FAIL: length missing")

    # ---------------------------------------------------------
    # 5. geometry
    # ---------------------------------------------------------
    print()
    print("[6] geometry")

    geom_ok = False
    geometry_valid = 0
    geometry_invalid = 0

    if "geometry" in road.columns:
        geoms = []
        for x in road["geometry"]:
            try:
                geoms.append(wkt.loads(x))
            except Exception:
                geoms.append(None)

        geometry_valid = sum(
            g is not None and not g.is_empty and g.geom_type == "LineString"
            for g in geoms
        )

        geometry_invalid = len(geoms) - geometry_valid

        print("  valid LineString :", f"{geometry_valid:,}")
        print("  invalid          :", f"{geometry_invalid:,}")

        geom_ok = geometry_invalid == 0
    else:
        print("  FAIL: geometry missing")

    # ---------------------------------------------------------
    # 6. road types
    # ---------------------------------------------------------
    print()
    print("[7] road types")

    if "road_type" in road.columns:
        type_counts = road["road_type"].fillna("NULL").value_counts()

        print("  unique types:", len(type_counts))

        for name, n in type_counts.items():
            print(f"    {str(name):25s} {n:8,d}")

        link_like = road["road_type"].astype(str).str.endswith("_link").sum()
        print("  *_link roads:", f"{int(link_like):,}")
    else:
        type_counts = pd.Series(dtype=int)
        print("  road_type missing")

    # ---------------------------------------------------------
    # 7. road_type_id
    # ---------------------------------------------------------
    print()
    print("[8] road_type_id")

    if "road_type_id" in road.columns:
        print(
            "  unique:",
            sorted(road["road_type_id"].dropna().unique().tolist())
        )
    else:
        print("  missing")

    # ---------------------------------------------------------
    # 8. oneway -> GTG from_biway candidate
    # ---------------------------------------------------------
    print()
    print("[9] oneway")

    if "oneway" in road.columns:
        oneway = safe_bool_series(road["oneway"])

        print("  one-way :", f"{int(oneway.sum()):,}")
        print("  two-way :", f"{int((~oneway).sum()):,}")
        print(
            "  NOTE: 后续可以结合反向 edge 实际存在性构造 GTG from_biway，"
            "不是简单直接复制 oneway"
        )
    else:
        print("  missing")

    # ---------------------------------------------------------
    # 9. graph connectivity
    # ---------------------------------------------------------
    print()
    print("[10] directed road adjacency")

    if node_ok:
        from_counts = road.groupby("from_node_id").size()
        to_counts = road.groupby("to_node_id").size()

        next_count = 0

        from_map = {}
        for idx, row in road[["from_node_id", "to_node_id"]].iterrows():
            from_map.setdefault(row["from_node_id"], 0)
            from_map[row["from_node_id"]] += 1

        for _, row in road[["to_node_id"]].iterrows():
            next_count += from_map.get(row["to_node_id"], 0)

        print("  unique from nodes :", f"{len(from_counts):,}")
        print("  unique to nodes   :", f"{len(to_counts):,}")
        print("  road->road edges  :", f"{next_count:,}")

        adjacency_ok = next_count > 0
    else:
        adjacency_ok = False
        next_count = 0

    # ---------------------------------------------------------
    # 10. CRAFT region
    # ---------------------------------------------------------
    print()
    print("[11] CRAFT region alignment")

    region = pd.read_csv(region_path)

    region_geoms = gpd.GeoSeries(
        region["geometry"].map(wkt.loads),
        crs="EPSG:4326",
    )

    try:
        region_union = region_geoms.union_all()
    except AttributeError:
        region_union = region_geoms.unary_union

    road_geoms = gpd.GeoSeries(
        [wkt.loads(x) for x in road["geometry"]],
        crs="EPSG:4326",
    )

    intersects = road_geoms.intersects(region_union)

    inside_num = int(intersects.sum())
    inside_ratio = inside_num / len(road) if len(road) else 0

    print("  regions            :", f"{len(region):,}")
    print("  roads intersect    :", f"{inside_num:,}/{len(road):,}")
    print("  intersect ratio    :", f"{inside_ratio:.4%}")

    # ---------------------------------------------------------
    # 11. metadata
    # ---------------------------------------------------------
    with open(feature_path) as f:
        meta = json.load(f)

    print()
    print("[12] spatial metadata")
    print(
        "  bounds:",
        meta.get("min_lon"),
        meta.get("min_lat"),
        meta.get("max_lon"),
        meta.get("max_lat"),
    )
    print("  utm_epsg:", meta.get("utm_epsg", "NOT_FOUND"))

    # ---------------------------------------------------------
    # Final
    # ---------------------------------------------------------
    basic_ok = (
        not missing_gtg
        and node_ok
        and length_ok
        and geom_ok
        and adjacency_ok
    )

    print()
    print("-" * 90)

    if basic_ok:
        print(
            "[RESULT] PASS: 当前 CRAFT road.csv "
            "具备转换成 GTG 基础 road/map 数据的结构条件"
        )
    else:
        print(
            "[RESULT] FAIL: 当前 road.csv 仍有基础结构问题，"
            "转换前需要处理"
        )

    print(
        "[IMPORTANT] Space Syntax 字段当前不属于本次基础审计；"
        "GTG 完整 Backbone 后续还需单独生成/对齐。"
    )

    result = {
        "city": city,
        "num_roads": int(len(road)),
        "columns": road.columns.tolist(),
        "missing_gtg_base_fields": missing_gtg,
        "road_id_unique": bool(unique),
        "road_id_continuous": bool(continuous),
        "node_ids_ok": bool(node_ok),
        "length_ok": bool(length_ok),
        "geometry_ok": bool(geom_ok),
        "geometry_invalid": int(geometry_invalid),
        "road_to_road_edges": int(next_count),
        "adjacency_ok": bool(adjacency_ok),
        "region_count": int(len(region)),
        "roads_intersect_region": int(inside_num),
        "road_region_intersection_ratio": float(inside_ratio),
        "road_types": {
            str(k): int(v)
            for k, v in type_counts.items()
        },
        "basic_gtg_convertible": bool(basic_ok),
    }

    return result


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    results = []

    for city in CITIES:
        results.append(audit_city(city))

    output = OUT_ROOT / "craft_road_gtg_compatibility.json"

    with open(output, "w") as f:
        json.dump(results, f, indent=2)

    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)

    for r in results:
        print(
            f"{r['city']:8s} | "
            f"roads={r['num_roads']:7,d} | "
            f"road_edges={r['road_to_road_edges']:8,d} | "
            f"region_hit={r['road_region_intersection_ratio']:8.2%} | "
            f"GTG_BASE={'PASS' if r['basic_gtg_convertible'] else 'FAIL'}"
        )

    print()
    print("[OUTPUT]", output)


if __name__ == "__main__":
    main()
