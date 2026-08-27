import torch
import pytest

from hcfm.preprocessing import load_static_cache


EXPECTED = {
    "chi": (73, 52681), "dc": (81, 70499), "toronto": (60, 37943), "ny": (95, 68995),
}


@pytest.mark.parametrize("city", EXPECTED)
def test_real_hierarchy_cache_shapes_finite_and_normalized(city):
    tensors, manifest = load_static_cache("cache/hcfm", city)
    n, m = EXPECTED[city]
    assert tensors["region_x"].shape == (n, 45)
    assert tensors["road_x"].shape == (m, 15)
    assert tensors["p_struct"].shape == (n, m)
    assert tensors["b_in"].shape == (n, m) and tensors["b_out"].shape == (n, m)
    assert torch.isfinite(tensors["region_x"]).all() and torch.isfinite(tensors["road_x"]).all()
    row_sum = torch.sparse.sum(tensors["p_struct"], dim=1).to_dense()
    nonempty = row_sum > 0
    assert torch.allclose(row_sum[nonempty], torch.ones_like(row_sum[nonempty]), atol=1e-5)
    assert tensors["b_in"]._nnz() == manifest["b_in_nnz"]
    assert tensors["b_out"]._nnz() == manifest["b_out_nnz"]
    p = tensors["p_struct"].coalesce()
    assert torch.equal(tensors["region_to_road_edge_index"], p.indices())
    assert torch.equal(tensors["road_to_region_edge_index"], p.indices()[[1, 0]])
    assert torch.equal(tensors["road_to_region_weight"], p.values())
    assert manifest["parent_osm_way_ids"] is None
