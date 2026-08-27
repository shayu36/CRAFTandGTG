"""CRAFT 兼容 CSV 的固定类别与字段契约。"""

POI_TYPES = (
    "bicycle_rental_amenity",
    "fast_food_amenity",
    "restaurant_amenity",
    "bicycle_parking_amenity",
    "cafe_amenity",
    "public_transport",
    "shop",
    "tourism",
    "leisure",
    "office",
    "historic",
    "sport",
)

ROAD_TYPES = (
    "residential",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "motorway",
    "living_street",
    "unclassified",
)

ROAD_TYPE_TO_ID = {name: idx for idx, name in enumerate(ROAD_TYPES)}

# GTG 的 link_type_name -> CRAFT road_type_id。
# service/track 没有 CRAFT 独立类别，按已确认方案归入 unclassified；
# cycleway 不作为机动车静态道路，显式丢弃并在审计中计数。
GTG_ROAD_TYPE_MAP = {
    "motorway": ROAD_TYPE_TO_ID["motorway"],
    "trunk": ROAD_TYPE_TO_ID["trunk"],
    "primary": ROAD_TYPE_TO_ID["primary"],
    "secondary": ROAD_TYPE_TO_ID["secondary"],
    "tertiary": ROAD_TYPE_TO_ID["tertiary"],
    "unclassified": ROAD_TYPE_TO_ID["unclassified"],
    "service": ROAD_TYPE_TO_ID["unclassified"],
    "track": ROAD_TYPE_TO_ID["unclassified"],
}

GTG_ROAD_REQUIRED_COLUMNS = {
    "link_id",
    "from_node_id",
    "to_node_id",
    "link_type_name",
    "length",
    "geometry",
}

TRAJECTORY_REQUIRED_COLUMNS = {"traj_id", "start_time", "rid_list", "dur_list"}

CRAFT_ROAD_COLUMNS = (
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
)

FLOW_COLUMNS = (
    "region_id",
    "date",
    "weekday",
    "start_hour",
    "in_flow",
    "out_flow",
    "month",
)


def craft_feature_columns():
    """返回 CRAFT 读取器实际使用的 45 维静态特征顺序。"""
    return (
        ["population", "population_density", "dist_to_center", "road_num", "road_length"]
        + [f"poi_num_{idx}" for idx in range(len(POI_TYPES))]
        + [f"poi_score_{idx}" for idx in range(len(POI_TYPES))]
        + [f"road_num_{idx}" for idx in range(len(ROAD_TYPES))]
        + [f"road_length_{idx}" for idx in range(len(ROAD_TYPES))]
    )
