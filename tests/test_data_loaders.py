"""data_loaders 集成测试: baseline 返回 45 维, fusion 返回 45+9 维, 区域图可构建。"""
import os

import numpy as np
import pytest

import data_loaders
from conftest import CACHE_DIR, NORM_DIR, CRAFT_ROOT, CITIES

RAW_DIM = 45
GTG_DIM = 9


def _baseline_cfg():
    return {
        "craft_data_root": CRAFT_ROOT,
        "norm_flow_root": NORM_DIR,
        "use_gtg_topology": False,
    }


def _fusion_cfg():
    return {
        "craft_data_root": CRAFT_ROOT,
        "norm_flow_root": NORM_DIR,
        "use_gtg_topology": True,
        "gtg_cache_dir": CACHE_DIR,
        "gtg_feature_dim": GTG_DIM,
    }


@pytest.mark.parametrize("city", CITIES)
def test_baseline_feature_dim(city):
    data_loaders.configure(_baseline_cfg())
    feat, edge_index = data_loaders.load_region_feature(city)
    assert feat.shape[1] == RAW_DIM
    assert edge_index.shape[0] == 2


@pytest.mark.parametrize("city", CITIES)
def test_fusion_feature_dim(city):
    data_loaders.configure(_fusion_cfg())
    feat, edge_index = data_loaders.load_region_feature(city)
    assert feat.shape[1] == RAW_DIM + GTG_DIM, f"{city} 融合维度 {feat.shape[1]}"
    assert np.all(np.isfinite(feat))
    # 前 45 维应与基线一致
    data_loaders.configure(_baseline_cfg())
    base_feat, _ = data_loaders.load_region_feature(city)
    assert np.allclose(feat[:, :RAW_DIM], base_feat)


def test_fusion_missing_cache_dir_raises():
    with pytest.raises(ValueError):
        data_loaders.configure({
            "craft_data_root": CRAFT_ROOT,
            "use_gtg_topology": True,
            "gtg_cache_dir": None,
        })


def test_region_graph_masks_regions_without_train_windows():
    data_loaders.configure(_fusion_cfg())
    graph = data_loaders.load_region_graph("chi")
    assert graph.x.shape[0] == 73
    assert graph.value.shape == (41, 48)
    assert graph.value_region_ids.shape == (41,)
    assert graph.value_mask.shape == (73,)
    assert int(graph.value_mask.sum()) == 41


def test_city_data_dirs_override_only_selected_city(tmp_path):
    cfg = _baseline_cfg()
    custom_city = tmp_path / "beijing"
    cfg["city_data_dirs"] = {"beijing": str(custom_city)}
    data_loaders.configure(cfg)
    assert data_loaders._get_city_data_dir("beijing") == str(custom_city)
    assert data_loaders._get_city_data_dir("chi") == os.path.join(CRAFT_ROOT, "chi")


def teardown_function(_):
    # 复位为默认, 避免污染其他测试
    data_loaders.configure({"craft_data_root": CRAFT_ROOT, "norm_flow_root": NORM_DIR})
