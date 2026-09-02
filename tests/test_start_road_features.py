"""START 静态 Road 33 维特征与 v1/v2 cache 隔离测试。"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from static_hierarchy.data import load_city_static_hierarchy  # noqa: E402
from static_hierarchy.preprocessing import (  # noqa: E402
    START_ROAD_FEATURE_ORDER,
    build_start_static_road_features,
)


def _roads():
    return pd.DataFrame(
        {
            "road_type_id": [0, 1, 7, 2],
            "lanes": ["1", "5+", None, "3 lanes"],
            "maxspeed": ["30", "30 mph", "60 km/h", None],
        }
    )


def test_start_static_feature_schema_and_buckets():
    roads = _roads()
    edge_index = np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    features, meta = build_start_static_road_features(
        roads, edge_index, np.asarray([10.0, 20.0, 40.0, 80.0]), maxspeed_unit="km/h"
    )
    assert features.shape == (4, 33)
    assert meta["road_feature_names"] == START_ROAD_FEATURE_ORDER
    assert np.isfinite(features).all()
    # 30 mph is converted to 48.28032 km/h and therefore falls in 31_50.
    assert features[1, 17] == 1.0
    # Missing lanes/maxspeed are explicit unknown buckets.
    assert features[2, 9] == 1.0
    assert features[3, 15] == 1.0


def test_start_degree_is_directed_before_self_loops():
    roads = pd.DataFrame({"road_type_id": [0, 0], "lanes": [1, 1], "maxspeed": [40, 40]})
    edge_index = np.asarray([[0], [1]], dtype=np.int64)
    features, _ = build_start_static_road_features(roads, edge_index, [1.0, 2.0])
    # indegree columns begin at 21; outdegree columns begin at 27.
    assert features[:, 21:27].argmax(axis=1).tolist() == [0, 1]
    assert features[:, 27:33].argmax(axis=1).tolist() == [1, 0]


def test_start_maxspeed_supports_mps_contract():
    roads = pd.DataFrame({"road_type_id": [0], "lanes": [1], "maxspeed": [10.0]})
    features, meta = build_start_static_road_features(
        roads, np.empty((2, 0), dtype=np.int64), [1.0], maxspeed_unit="m/s"
    )
    assert meta["maxspeed_unit"] == "m/s"
    # 10 m/s = 36 km/h -> 31_50 bucket (index 17).
    assert features[0, 17] == 1.0


def test_start_invalid_road_type_is_rejected():
    roads = pd.DataFrame({"road_type_id": [8], "lanes": [1], "maxspeed": [40]})
    with pytest.raises(ValueError, match="road_type_id 越界"):
        build_start_static_road_features(roads, np.empty((2, 0), dtype=np.int64), [1.0])


@pytest.mark.parametrize("city", ["beijing", "chengdushi", "xianshi"])
def test_real_start_v2_cache_contract(city):
    hierarchy = load_city_static_hierarchy(
        ROOT / "cache" / "static_hierarchy_start_v2",
        city,
        expected_feature_version="three-layer-start-road-v2",
    )
    assert hierarchy.road_x.shape[1] == 33
    assert hierarchy.metadata["maxspeed_unit"] == "km/h"
    assert hierarchy.metadata["road_feature_dim"] == 33


def test_v1_cache_is_not_silently_read_as_v2():
    with pytest.raises(ValueError, match="feature_version"):
        load_city_static_hierarchy(
            ROOT / "cache" / "static_hierarchy",
            "beijing",
            expected_feature_version="three-layer-start-road-v2",
        )
