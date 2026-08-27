import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from gtg_preprocessing.static import prepare_population_features


def _grids():
    full = gpd.GeoDataFrame({"full_grid_id": [0, 1]}, geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)], crs=4326)
    selected = gpd.GeoDataFrame({"region_id": [0, 1]}, geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)], crs=4326)
    return full, selected


def test_population_coverage_and_real_zero(tmp_path):
    full, selected = _grids()
    path = tmp_path / "population.csv"
    pd.DataFrame({"lon": [0.5, 1.5], "lat": [0.5, 0.5], "population": [0.0, 4.0]}).to_csv(path, index=False)
    _, sums, meta = prepare_population_features(path, full, selected, 4326)
    assert np.allclose(sums, [0.0, 4.0])
    assert meta["selected_grid_valid_pixel_counts"] == [1, 1]
    assert meta["zero_population_region_ids"] == [0]


def test_population_missing_coverage_raises(tmp_path):
    full, selected = _grids()
    path = tmp_path / "population.csv"
    pd.DataFrame({"lon": [0.5], "lat": [0.5], "population": [3.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="没有有效人口像元覆盖"):
        prepare_population_features(path, full, selected, 4326)
