"""第二阶段三层 GraphGPS + sparse LapPE 最小正确性测试。"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
import torch

from static_hierarchy.contracts import CityStaticHierarchy
from static_hierarchy.preprocessing import START_ROAD_FEATURE_ORDER
from three_layer_graphgps.data import (
    GraphGPSCityData,
    RegionFlowTargets,
    load_source_region_flow_splits,
)
from three_layer_graphgps.engine import train_and_evaluate
from three_layer_graphgps.model import GlobalAttentionBranch, ThreeLayerGraphGPSLapPE
from three_layer_graphgps.pooling import pool_road_to_syntax, pool_syntax_to_region
from three_layer_graphgps.spectral_lap_pe import (
    compute_sparse_laplacian_eigenpairs,
    prepare_hierarchy_lappe,
)


def _toy_hierarchy() -> CityStaticHierarchy:
    torch.manual_seed(7)
    road_x = torch.randn(8, 33)
    # 明确的有向链；PE 会生成反向边，但消息边不应改变。
    road_edge = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7]], dtype=torch.long
    )
    assignment = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2], dtype=torch.long)
    counts = torch.bincount(assignment, minlength=3)
    road_to_syntax_edge = torch.stack([assignment, torch.arange(8)])
    road_to_syntax_weight = 1.0 / counts[assignment].float()
    syntax_to_region_edge = torch.tensor([[0, 0, 1], [0, 1, 2]], dtype=torch.long)
    syntax_to_region_weight = torch.tensor([0.6, 0.4, 1.0])
    return CityStaticHierarchy(
        city_id="toy",
        region_x=torch.randn(2, 45),
        region_edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        road_x=road_x,
        road_edge_index=road_edge,
        road_ids=tuple(str(index) for index in range(8)),
        syntax_x=torch.randn(3, 5),
        syntax_edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        road_to_syntax_assignment=assignment,
        road_to_syntax_edge_index=road_to_syntax_edge,
        road_to_syntax_weight=road_to_syntax_weight,
        road_to_syntax_shape=(3, 8),
        syntax_to_region_edge_index=syntax_to_region_edge,
        syntax_to_region_weight=syntax_to_region_weight,
        syntax_to_region_shape=(2, 3),
        region_has_syntax=torch.tensor([True, True]),
        metadata={
            "feature_version": "three-layer-start-road-v2",
            "road_feature_mode": "start_static",
            "road_feature_dim": 33,
            "road_feature_names": list(START_ROAD_FEATURE_ORDER),
            "maxspeed_unit": "km/h",
        },
    )


def _config(road_attention: str = "linear", full_max: int = 4096) -> dict:
    return {
        "model": {
            "name": "three_layer_graphgps_lappe",
            "hidden_dim": 16,
            "output_dim": 48,
            "num_layers_road": 1,
            "num_layers_syntax": 1,
            "num_layers_region": 1,
            "dropout": 0.0,
        },
        "posenc": {
            "type": "LapPE",
            "laplacian_norm": "sym",
            "road_num_eig": 4,
            "syntax_num_eig": 4,
            "region_num_eig": 4,
            "pe_dim": 4,
            "encoder": "DeepSet",
            "cache": False,
        },
        "attention": {
            "road_global_attn": road_attention,
            "road_full_attn_max_nodes": full_max,
            "syntax_global_attn": "full",
            "region_global_attn": "full",
            "num_heads": 4,
        },
        "hierarchy": {
            "road_to_syntax_pool": "mean",
            "syntax_to_region_pool": "weighted_mean",
        },
    }


def test_sparse_lappe_shape_order_padding_and_cache(tmp_path):
    edge = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    first = compute_sparse_laplacian_eigenpairs(
        edge, 4, 6, is_directed=True, cache_key="toy_road", cache_dir=tmp_path
    )
    assert first.eigvals.shape == (4, 6, 1)
    assert first.eigvecs.shape == (4, 6)
    assert first.mask.shape == (6,)
    assert int(first.mask.sum()) == 3
    assert torch.isfinite(first.eigvals).all()
    assert torch.isfinite(first.eigvecs).all()
    valid = first.eigvals[0, first.mask, 0]
    assert torch.all(valid[1:] >= valid[:-1] - 1e-6)
    second = compute_sparse_laplacian_eigenpairs(
        edge, 4, 6, is_directed=True, cache_key="toy_road", cache_dir=tmp_path
    )
    assert torch.equal(first.mask, second.mask)
    assert torch.allclose(first.eigvals, second.eigvals)
    assert torch.allclose(first.eigvecs, second.eigvecs)
    source = inspect.getsource(compute_sparse_laplacian_eigenpairs)
    assert "toarray" not in source
    assert "np.linalg.eigh" not in source


def test_road_pe_uses_undirected_copy_without_overwriting_message_edges():
    hierarchy = _toy_hierarchy()
    original = hierarchy.road_edge_index.clone()
    pe = prepare_hierarchy_lappe(hierarchy, road_k=4, syntax_k=4, region_k=4)
    pairs = set(map(tuple, pe.road.edge_index_pe.t().tolist()))
    assert (0, 1) in pairs and (1, 0) in pairs
    assert torch.equal(hierarchy.road_edge_index, original)
    model = ThreeLayerGraphGPSLapPE(_config()).eval()
    output = model(hierarchy, pe, return_edge_audit=True)
    assert torch.equal(output["road_edge_index_msg"], original)
    assert not torch.equal(output["road_edge_index_msg"], output["road_edge_index_pe"])


def test_road_to_syntax_pool_supports_assignment_and_sparse_operator():
    hierarchy = _toy_hierarchy()
    road_h = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    by_assignment = pool_road_to_syntax(
        road_h, 3, assignment=hierarchy.road_to_syntax_assignment
    )
    by_sparse = pool_road_to_syntax(
        road_h,
        3,
        assignment=hierarchy.road_to_syntax_assignment,
        edge_index=hierarchy.road_to_syntax_edge_index,
        weight=hierarchy.road_to_syntax_weight,
        shape=hierarchy.road_to_syntax_shape,
    )
    expected = torch.stack([road_h[:3].mean(0), road_h[3:6].mean(0), road_h[6:].mean(0)])
    assert torch.allclose(by_assignment, expected)
    assert torch.allclose(by_sparse, expected)


def test_syntax_to_region_weighted_pool():
    hierarchy = _toy_hierarchy()
    syntax_h = torch.tensor([[1.0, 2.0], [3.0, 4.0], [7.0, 8.0]])
    pooled = pool_syntax_to_region(
        syntax_h,
        edge_index=hierarchy.syntax_to_region_edge_index,
        weight=hierarchy.syntax_to_region_weight,
        shape=hierarchy.syntax_to_region_shape,
    )
    assert pooled.shape == (2, 2)
    assert torch.allclose(pooled[0], 0.6 * syntax_h[0] + 0.4 * syntax_h[1])
    assert torch.allclose(pooled[1], syntax_h[2])


def test_three_layer_graphgps_forward_backward_and_no_road_region_shortcut():
    hierarchy = _toy_hierarchy()
    pe = prepare_hierarchy_lappe(hierarchy, road_k=4, syntax_k=4, region_k=4)
    model = ThreeLayerGraphGPSLapPE(_config())
    output = model(hierarchy, pe)
    assert output["H_road"].shape == (8, 16)
    assert output["pooled_road_to_syntax"].shape == (3, 16)
    assert output["H_syntax"].shape == (3, 16)
    assert output["pooled_syntax_to_region"].shape == (2, 16)
    assert output["H_region"].shape == (2, 16)
    assert output["pred"].shape == (2, 48)
    assert all(torch.isfinite(value).all() for value in output.values())
    output["pred"].square().mean().backward()
    modules = (
        model.road_input,
        model.road_graphgps,
        model.syntax_input,
        model.syntax_graphgps,
        model.region_input,
        model.region_graphgps,
        model.prediction_head,
    )
    for module in modules:
        gradients = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
        assert any(gradient is not None for gradient in gradients)
        assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)
    assert not any("road_to_region" in name for name, _ in model.named_modules())
    assert "road_to_region_h" not in output


def test_road_features_reach_region_only_through_syntax():
    hierarchy = _toy_hierarchy()
    pe = prepare_hierarchy_lappe(hierarchy, road_k=4, syntax_k=4, region_k=4)
    model = ThreeLayerGraphGPSLapPE(_config()).eval()
    with torch.no_grad():
        original = model(hierarchy, pe)["H_region"]
        changed_hierarchy = _toy_hierarchy()
        changed_hierarchy.road_x[0, 0] += 10.0
        changed = model(changed_hierarchy, pe)["H_region"]
    assert not torch.allclose(original, changed)


def test_road_full_attention_has_node_count_fallback():
    branch = GlobalAttentionBranch(
        8, 2, 0.0, "full", full_attention_max_nodes=4, layer_name="road"
    )
    with pytest.warns(RuntimeWarning, match="fallback"):
        output = branch(torch.randn(8, 8))
    assert output.shape == (8, 8)
    assert torch.isfinite(output).all()


def test_external_lappe_graph_hash_mismatch_is_rejected():
    hierarchy = _toy_hierarchy()
    pe = prepare_hierarchy_lappe(hierarchy, road_k=4, syntax_k=4, region_k=4)
    changed = _toy_hierarchy()
    changed.road_edge_index = changed.road_edge_index.clone()
    changed.road_edge_index[:, 0] = torch.tensor([0, 2])
    with pytest.raises(ValueError, match="graph hash"):
        ThreeLayerGraphGPSLapPE(_config())(changed, pe)


def test_source_flow_is_split_by_time_and_training_runs_train_valid_test(tmp_path):
    rows = []
    for hour in range(10):
        for region_id in range(2):
            base = float(hour + region_id) / 20.0
            rows.append(
                {
                    "region_id": region_id,
                    "date": "2020-01-01",
                    "start_hour": hour,
                    "in_flow": str([base] * 24),
                    "out_flow": str([base + 0.1] * 24),
                }
            )
    city_dir = tmp_path / "flow" / "toy"
    city_dir.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(city_dir / "norm_train_len_24.csv", index=False)
    targets = load_source_region_flow_splits(
        tmp_path / "flow", "toy", num_regions=2, split_ratios=(0.6, 0.2, 0.2)
    )
    assert set(targets) == {"train", "valid", "test"}
    assert all(target.values.shape == (2, 48) for target in targets.values())

    hierarchy = _toy_hierarchy()
    pe = prepare_hierarchy_lappe(hierarchy, road_k=4, syntax_k=4, region_k=4)
    data = GraphGPSCityData(hierarchy=hierarchy, posenc=pe, targets=targets)
    config = _config()
    config["training"] = {
        "batch_size": 1,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "epochs": 1,
        "seed": 4,
    }
    metrics = train_and_evaluate(
        config, [data], output_dir=tmp_path / "output", device="cpu"
    )
    assert set(("train", "valid", "test")) <= set(metrics)
    assert (tmp_path / "output" / "best.pt").exists()
    assert (tmp_path / "output" / "last.pt").exists()
    assert (tmp_path / "output" / "metrics.json").exists()
    assert np.isfinite(metrics["test"]["city_macro_rmse"])
