"""第一阶段 Road→Syntax→Region 三层静态图契约、前向和 target 边界测试。"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "craft_integrated"))

from static_hierarchy.contracts import CityStaticHierarchy, validate_city_static_hierarchy  # noqa: E402
from static_hierarchy.data import load_city_static_hierarchy  # noqa: E402
from static_hierarchy.model import RoadTopologyEncoder, SyntaxEncoder, ThreeLayerStaticEncoder  # noqa: E402
from static_hierarchy.operators import coalesce_edges, weighted_region_projection  # noqa: E402


def _tiny() -> CityStaticHierarchy:
    return CityStaticHierarchy(
        city_id="toy",
        region_x=torch.randn(2, 45),
        region_edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        road_topo_x=torch.tensor([[1, 1, 0, 1], [1, 1, 1, 2], [1, 0, 1, 1]], dtype=torch.float32),
        road_edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        road_ids=("r0", "r1", "r2"),
        syntax_x=torch.randn(2, 5),
        syntax_edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        road_to_syntax_assignment=torch.tensor([0, 0, 1], dtype=torch.long),
        road_to_syntax_edge_index=torch.tensor([[0, 0, 1], [0, 1, 2]], dtype=torch.long),
        road_to_syntax_weight=torch.tensor([0.5, 0.5, 1.0]),
        road_to_syntax_shape=(2, 3),
        syntax_to_region_edge_index=torch.tensor([[0, 1, 1], [0, 0, 1]], dtype=torch.long),
        syntax_to_region_weight=torch.tensor([1.0, 0.25, 0.75]),
        syntax_to_region_shape=(2, 2),
        region_has_syntax=torch.tensor([True, True]),
        metadata={"feature_version": "three-layer-static-v1"},
    )


def test_road_topology_is_fixed_four_dimensional():
    hierarchy = _tiny()
    assert hierarchy.road_topo_x.shape == (3, 4)
    assert hierarchy.metadata["feature_version"] == "three-layer-static-v1"
    # [bias, in_degree, out_degree, total_degree]，不包含 Road 语义/几何字段。
    assert torch.equal(hierarchy.road_topo_x[:, 1:], torch.tensor([[1, 0, 1], [1, 1, 2], [0, 1, 1]], dtype=torch.float32))


def test_road_to_syntax_assignment_and_mean_weights():
    hierarchy = _tiny()
    validate_city_static_hierarchy(hierarchy)
    sums = torch.zeros(2).index_add(0, hierarchy.road_to_syntax_assignment, hierarchy.road_to_syntax_weight)
    assert torch.allclose(sums, torch.ones(2))
    assert torch.equal(hierarchy.road_to_syntax_edge_index[1], torch.arange(3))


def test_syntax_edge_coalesce_preserves_direction():
    edge, weight, _ = coalesce_edges([0, 0, 1], [1, 1, 0], [1, 2, 4])
    assert edge.tolist() == [[0, 1], [1, 0]]
    assert np.allclose(weight, [3, 4])


def test_syntax_to_region_geometry_projection():
    assignment = np.asarray([0, 0, 1])
    lengths = np.asarray([[2.0, 0.0], [0.0, 3.0], [0.0, 1.0]])
    edge, weight, shape, mask = weighted_region_projection(assignment, lengths, 2, 2)
    assert shape == (2, 2)
    assert edge.tolist() == [[0, 1, 1], [0, 0, 1]]
    assert np.allclose(weight.numpy(), [1.0, 0.75, 0.25])
    assert mask.tolist() == [True, True]


def test_syntax_to_region_keeps_tiny_positive_intersection_and_empty_rows():
    assignment = np.asarray([0])
    lengths = np.asarray([[1e-12, 0.0, 0.0]])
    edge, weight, shape, mask = weighted_region_projection(assignment, lengths, 1, 3)
    assert shape == (3, 1)
    assert edge.tolist() == [[0], [0]]
    assert np.allclose(weight.numpy(), [1.0])
    assert mask.tolist() == [True, False, False]


def test_invalid_syntax_assignment_is_rejected():
    with pytest.raises(ValueError, match="assignment 越界"):
        weighted_region_projection(np.asarray([1]), np.asarray([[1.0]]), 1, 1)


def test_duplicate_road_ids_are_rejected():
    hierarchy = _tiny()
    hierarchy.road_ids = ("r0", "r0", "r2")
    with pytest.raises(ValueError, match="road_ids 存在重复"):
        validate_city_static_hierarchy(hierarchy)


def test_three_layer_forward_and_gradient():
    hierarchy = _tiny()
    config = {
        "rep_dim": 8, "road_topo_feature_dim": 4, "syntax_feature_dim": 5,
        "road_gat_layers": 1, "road_gat_heads": 2, "road_dropout": 0.0,
        "syntax_gat_layers": 1, "syntax_gat_heads": 2, "syntax_dropout": 0.0,
    }
    model = ThreeLayerStaticEncoder(config)
    result = model(hierarchy, return_intermediates=True)
    assert result["road_h"].shape == (3, 8)
    assert result["syntax_h"].shape == (2, 8)
    assert result["region_rep"].shape == (2, 8)
    assert all(torch.isfinite(value).all() for value in result.values())
    result["region_rep"].sum().backward()
    for name in ("road_init", "road_encoder", "syntax_init", "syntax_encoder", "region_init", "region_gnn"):
        assert any(parameter.grad is not None for parameter in getattr(model, name).parameters()), name


def test_public_layer_encoders_and_no_direct_road_region_path():
    config = {"rep_dim": 8, "road_gat_layers": 1, "road_gat_heads": 2,
              "syntax_gat_layers": 1, "syntax_gat_heads": 2}
    assert isinstance(RoadTopologyEncoder(8, 1, 2, 0.0), RoadTopologyEncoder)
    assert isinstance(SyntaxEncoder(8, 1, 2, 0.0), SyntaxEncoder)
    result = ThreeLayerStaticEncoder(config)(_tiny(), return_intermediates=True)
    assert "syntax_to_region_h" in result
    assert "road_to_region_h" not in result


@pytest.mark.parametrize("city", ["beijing", "chengdushi", "xianshi", "chi"])
def test_real_static_cache_contract(city):
    hierarchy = load_city_static_hierarchy(ROOT / "cache" / "static_hierarchy", city)
    assert hierarchy.region_x.shape[1] == 45
    assert hierarchy.road_topo_x.shape[1] == 4
    assert hierarchy.syntax_x.shape[1] == 5
    assert hierarchy.road_to_syntax_edge_index.shape[1] == hierarchy.num_roads
    assert torch.isfinite(hierarchy.region_x).all()


def test_target_static_loader_does_not_read_dynamic_flow(monkeypatch):
    import data_loaders

    data_loaders.configure({
        "static_structure_mode": "three_layer",
        "road_feature_mode": "topology_only",
        "static_hierarchy_cache_dir": str(ROOT / "cache" / "static_hierarchy"),
    })
    def fail(*_args, **_kwargs):
        raise AssertionError("target static loader must not read norm_train")
    monkeypatch.setattr(data_loaders, "load_norm_flow", fail)
    target = data_loaders.load_region_graph("chi", require_flow_labels=False)
    assert target.value is None
    assert target.region_x.shape[1] == 45


def test_cospec_is_explicitly_rejected():
    import data_loaders
    with pytest.raises(NotImplementedError, match="CoSpec"):
        data_loaders.configure({"road_feature_mode": "cospec"})


def teardown_function(_):
    import data_loaders
    data_loaders.configure({
        "craft_data_root": str(ROOT.parent / "CRAFT" / "cleared_data"),
        "norm_flow_root": str(ROOT / "data" / "norm_flow"),
    })
