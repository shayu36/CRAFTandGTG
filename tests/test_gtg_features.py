"""GTG 区域特征质量测试: 缓存存在、形状、无 NaN/Inf、覆盖率、维度一致。"""
import json
import os

import numpy as np
import pytest

from conftest import CACHE_DIR, CRAFT_ROOT, CITIES

EXPECT_DIM = 9
FEATURE_ORDER = [
    "connectivity", "total_depth", "integration", "choice", "mean_depth",
    "part_connectivity", "part_total_depth", "part_integration", "part_choice",
]


def _num_regions(city):
    import pandas as pd
    return len(pd.read_csv(os.path.join(CRAFT_ROOT, city, "grid_region_feature.csv")))


@pytest.mark.parametrize("city", CITIES)
def test_cache_exists_and_shape(city):
    pth = os.path.join(CACHE_DIR, f"{city}_gtg_region.npz")
    assert os.path.exists(pth), f"缺失缓存 {pth}"
    d = np.load(pth, allow_pickle=True)
    rf = d["region_feat"]
    names = list(d["feat_names"])
    assert names == FEATURE_ORDER, f"{city} 特征顺序不符: {names}"
    assert rf.shape == (_num_regions(city), EXPECT_DIM), f"{city} 形状 {rf.shape}"


@pytest.mark.parametrize("city", CITIES)
def test_no_nan_inf(city):
    d = np.load(os.path.join(CACHE_DIR, f"{city}_gtg_region.npz"), allow_pickle=True)
    assert np.all(np.isfinite(d["region_feat"])), f"{city} 含 NaN/Inf"


@pytest.mark.parametrize("city", CITIES)
def test_coverage_reasonable(city):
    meta = json.load(open(os.path.join(CACHE_DIR, f"{city}_gtg_meta.json")))
    cov = meta["coverage"]
    # 全部道路应被映射 (对偶图基于 road.csv, 区域覆盖完整城市范围)
    assert cov["unmapped_road_ratio"] <= 0.05, f"{city} 未映射道路比例过高 {cov['unmapped_road_ratio']}"
    # 空区域比例应在合理范围 (无流量的边缘网格允许为空)
    assert cov["empty_region_ratio"] <= 0.2, f"{city} 空区域比例过高 {cov['empty_region_ratio']}"


@pytest.mark.parametrize("city", CITIES)
def test_metrics_have_variation(city):
    # 空间句法指标应有跨区域差异 (非常数), 否则说明计算退化
    d = np.load(os.path.join(CACHE_DIR, f"{city}_gtg_region.npz"), allow_pickle=True)
    rf = d["region_feat"]
    for i, name in enumerate(FEATURE_ORDER):
        col = rf[:, i]
        nz = col[col != 0]
        assert nz.std() > 0, f"{city} {name} 无差异 (可能计算退化)"
