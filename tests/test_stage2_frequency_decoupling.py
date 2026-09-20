"""Stage 2 explicit low/high spectral feature contract tests."""

from __future__ import annotations

import inspect
import copy

import pytest
import torch

import three_layer_graphgps.data as stage2_data_module
from three_layer_graphgps.data import (
    GraphGPSCityData,
    export_spectral_features,
    load_spectral_features,
)
from three_layer_graphgps.frequency import (
    SpectralFeatureDecoupler,
    build_low_frequency_transfer_inputs,
)
from three_layer_graphgps.model import ThreeLayerGraphGPSLapPE
from three_layer_graphgps.engine import (
    load_checkpoint,
    save_checkpoint,
    stage2_graph_identities,
)
from three_layer_graphgps.spectral_lap_pe import (
    LaplacianEigenpairs,
    compute_sparse_laplacian_eigenpairs,
    prepare_hierarchy_lappe,
)

from test_stage2_graphgps_lappe import _config, _toy_hierarchy


def _manual_eigenpairs() -> LaplacianEigenpairs:
    # The final two padded columns are deliberately non-zero and must be ignored.
    eigvecs = torch.tensor(
        [
            [0.5, 0.5, 5.0, -8.0],
            [0.5, -0.5, 6.0, -7.0],
            [0.5, 0.5, 7.0, -6.0],
            [0.5, -0.5, 8.0, -5.0],
        ],
        dtype=torch.float32,
    )
    eigenvalues = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)
    return LaplacianEigenpairs(
        eigvals=eigenvalues.view(1, 4, 1).expand(4, -1, -1).clone(),
        eigvecs=eigvecs,
        mask=torch.tensor([True, True, False, False]),
        edge_index_pe=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        metadata={"graph_hash": "manual"},
    )


def _assert_module_has_finite_gradient(module: torch.nn.Module) -> None:
    gradients = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
    assert any(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)


def test_decoupler_shapes_reconstruction_residual_and_padding_mask():
    hidden = torch.randn(4, 6, requires_grad=True)
    eigenpairs = _manual_eigenpairs()
    decoupler = SpectralFeatureDecoupler(
        orthogonality_tolerance=1e-5,
        reconstruction_tolerance=1e-6,
    )
    with pytest.warns(RuntimeWarning, match="only 2 valid"):
        result = decoupler(hidden, eigenpairs, num_low_modes=4)
    assert result.low.shape == result.high.shape == result.mixed.shape == (4, 6)
    assert result.coefficients.shape == (2, 6)
    assert result.num_low_modes == 2
    assert result.cutoff_eigenvalue == pytest.approx(1.0)
    assert torch.allclose(hidden, result.low + result.high, atol=1e-6, rtol=1e-6)
    valid_basis = eigenpairs.eigvecs[:, :2]
    assert torch.allclose(
        valid_basis.transpose(0, 1) @ result.high,
        torch.zeros(2, 6),
        atol=1e-5,
        rtol=1e-5,
    )
    # Invalid padded columns contain large values; selecting only the first mode
    # must still equal the analytic constant-mode projection.
    first = decoupler(hidden, eigenpairs, num_low_modes=1)
    expected = valid_basis[:, :1] @ (valid_basis[:, :1].transpose(0, 1) @ hidden)
    assert torch.allclose(first.low, expected)
    assert torch.equal(result.high, hidden - result.low)


def test_frequency_losses_keep_all_graphgps_gradients_connected():
    hierarchy = _toy_hierarchy()
    posenc = prepare_hierarchy_lappe(hierarchy, road_k=4, syntax_k=4, region_k=4)
    model = ThreeLayerGraphGPSLapPE(_config())

    output = model(hierarchy, posenc)
    low_loss = sum(output[f"H_{layer}_low"].square().mean() for layer in ("road", "syntax", "region"))
    low_loss.backward()
    for module in (model.road_input, model.syntax_input, model.region_input, model.joint_lappe, model.graphgps):
        _assert_module_has_finite_gradient(module)

    model.zero_grad(set_to_none=True)
    output = model(hierarchy, posenc)
    high_loss = sum(output[f"H_{layer}_high"].square().mean() for layer in ("road", "syntax", "region"))
    high_loss.backward()
    for module in (model.road_input, model.syntax_input, model.region_input, model.joint_lappe, model.graphgps):
        _assert_module_has_finite_gradient(module)

    model.zero_grad(set_to_none=True)
    output = model(hierarchy, posenc)
    output["pred"].square().mean().backward()
    for module in (model.road_input, model.syntax_input, model.region_input, model.joint_lappe, model.graphgps, model.frequency_fusion, model.prediction_head):
        _assert_module_has_finite_gradient(module)


def test_model_frequency_outputs_reconstruct_and_no_dense_projector_source():
    hierarchy = _toy_hierarchy()
    posenc = prepare_hierarchy_lappe(hierarchy, road_k=4, syntax_k=4, region_k=4)
    output = ThreeLayerGraphGPSLapPE(_config()).eval()(hierarchy, posenc)
    expected_shapes = {"joint": (13, 16), "road": (8, 16), "syntax": (3, 16), "region": (2, 16)}
    for layer, shape in expected_shapes.items():
        assert output[f"H_{layer}_low"].shape == shape
        assert output[f"H_{layer}_high"].shape == shape
        assert torch.allclose(
            output[f"H_{layer}"],
            output[f"H_{layer}_low"] + output[f"H_{layer}_high"],
            atol=1e-5,
            rtol=1e-5,
        )
        assert torch.equal(
            output[f"H_{layer}_high"],
            output[f"H_{layer}"] - output[f"H_{layer}_low"],
        )
    source = inspect.getsource(SpectralFeatureDecoupler.forward)
    for forbidden in ("toarray", "np.linalg.eigh", "torch.linalg.eigh"):
        assert forbidden not in source
    assert not any("road_to_region" in name for name, _ in ThreeLayerGraphGPSLapPE(_config()).named_modules())


def test_default_attention_contract_is_preserved():
    model = ThreeLayerGraphGPSLapPE(_config())
    assert all(layer.global_attention.mode == "linear" for layer in model.graphgps.layers)


def test_frequency_config_contract_rejects_invalid_values():
    config = _config()
    config["frequency"]["joint_low_modes"] = 5
    with pytest.raises(ValueError, match="joint LapPE/low modes"):
        ThreeLayerGraphGPSLapPE(config)

    config = _config()
    config["frequency"]["method"] = "unsupported"
    with pytest.raises(ValueError, match="frequency.method"):
        ThreeLayerGraphGPSLapPE(config)

    config = _config()
    config["model"]["output_dim"] = 47
    with pytest.raises(ValueError, match=r"2 \* data.seq_length"):
        ThreeLayerGraphGPSLapPE(config)

    config = _config()
    config["data"]["seq_length"] = 12
    config["model"]["output_dim"] = 24
    with pytest.raises(ValueError, match="仅支持 data.seq_length=24"):
        ThreeLayerGraphGPSLapPE(config)


def test_low_frequency_transfer_interface_has_equal_source_city_mass():
    sources = [torch.randn(2, 4), torch.randn(5, 4), torch.randn(3, 4)]
    target = torch.randn(7, 4)
    transfer = build_low_frequency_transfer_inputs(sources, target)
    assert transfer.cost_metric == "cosine"
    assert transfer.source_city_sizes == (2, 5, 3)
    offset = 0
    for size in transfer.source_city_sizes:
        assert transfer.source_marginals[offset : offset + size].sum() == pytest.approx(1 / 3)
        offset += size
    assert transfer.source_marginals.sum() == pytest.approx(1.0)
    assert transfer.target_marginals.sum() == pytest.approx(1.0)


def test_spectral_export_roundtrip_rejects_graph_and_checkpoint_mismatch(tmp_path):
    hierarchy = _toy_hierarchy()
    posenc = prepare_hierarchy_lappe(hierarchy, road_k=4, syntax_k=4, region_k=4)
    data = GraphGPSCityData(hierarchy=hierarchy, posenc=posenc, targets=None)
    output = ThreeLayerGraphGPSLapPE(_config()).eval()(hierarchy, posenc)
    fingerprint = "a" * 64
    path = export_spectral_features(
        tmp_path / "toy.pt",
        data=data,
        output=output,
        checkpoint_sha256=fingerprint,
        global_attention_scope="joint",
    )
    loaded = load_spectral_features(
        path,
        hierarchy=hierarchy,
        expected_checkpoint_fingerprint=fingerprint,
    )
    assert loaded["road_ids"] == hierarchy.road_ids
    assert loaded["H_road"].shape[0] == len(loaded["road_ids"])
    assert loaded["H_syntax"].shape[0] == len(loaded["syntax_ids"])
    assert loaded["H_region"].shape[0] == len(loaded["region_ids"])
    assert loaded["lappe_version"] == "three-layer-joint-lappe-v3-weighted"
    assert loaded["weighted_pe"] is True
    assert len(loaded["weighted_spectrum_hash"]) == 64
    assert loaded["global_attention_scope"] == "joint"

    changed_pe = dict(loaded)
    changed_pe["joint_edge_weight_pe"] = loaded["joint_edge_weight_pe"].clone()
    changed_pe["joint_edge_weight_pe"][0] += 1.0
    changed_pe_path = tmp_path / "changed-pe.pt"
    torch.save(changed_pe, changed_pe_path)
    with pytest.raises(ValueError, match="joint_edge_weight_pe"):
        load_spectral_features(
            changed_pe_path,
            hierarchy=hierarchy,
            expected_checkpoint_fingerprint=fingerprint,
        )

    with pytest.raises(ValueError, match="checkpoint fingerprint"):
        load_spectral_features(
            path,
            hierarchy=hierarchy,
            expected_checkpoint_fingerprint="b" * 64,
        )
    changed = _toy_hierarchy()
    changed.road_edge_index = changed.road_edge_index.clone()
    changed.road_edge_index[:, 0] = torch.tensor([0, 2])
    with pytest.raises(ValueError, match="joint_graph_hash"):
        load_spectral_features(
            path,
            hierarchy=changed,
            expected_checkpoint_fingerprint=fingerprint,
        )
    old = dict(loaded)
    old["format_version"] = "three-layer-joint-graphgps-spectral-features-v2"
    old_path = tmp_path / "old-v2.pt"
    torch.save(old, old_path)
    with pytest.raises(ValueError, match="旧无权谱特征"):
        load_spectral_features(
            old_path,
            hierarchy=hierarchy,
            expected_checkpoint_fingerprint=fingerprint,
        )


def test_stage2_checkpoint_binds_training_weighted_spectrum_identity(tmp_path):
    hierarchy = _toy_hierarchy()
    posenc = prepare_hierarchy_lappe(hierarchy, road_k=4, syntax_k=4, region_k=4)
    data = GraphGPSCityData(hierarchy=hierarchy, posenc=posenc, targets=None)
    identities = stage2_graph_identities([data])
    model = ThreeLayerGraphGPSLapPE(_config())
    path = tmp_path / "stage2.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=None,
        config=_config(),
        training_graph_identities=identities,
        epoch=0,
        best_valid_rmse=1.0,
    )
    loaded = load_checkpoint(
        path,
        model=ThreeLayerGraphGPSLapPE(_config()),
        expected_config=_config(),
        expected_graph_identities=identities,
    )
    assert loaded["training_graph_identities"] == identities
    assert len(loaded["training_graph_identities_sha256"]) == 64

    tampered = copy.deepcopy(loaded)
    tampered["training_graph_identities"]["toy"]["weighted_spectrum_hash"] = "0" * 64
    tampered_path = tmp_path / "tampered.pt"
    torch.save(tampered, tampered_path)
    with pytest.raises(ValueError, match="identity 摘要"):
        load_checkpoint(
            tampered_path,
            model=ThreeLayerGraphGPSLapPE(_config()),
            expected_config=_config(),
            expected_graph_identities=identities,
        )


def test_target_static_prepare_does_not_load_flow(monkeypatch):
    hierarchy = _toy_hierarchy()
    posenc = prepare_hierarchy_lappe(hierarchy, road_k=4, syntax_k=4, region_k=4)
    monkeypatch.setattr(stage2_data_module, "load_stage2_hierarchy", lambda *_args: hierarchy)
    monkeypatch.setattr(stage2_data_module, "prepare_hierarchy_lappe", lambda *_args, **_kwargs: posenc)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("target static path must not load dynamic flow")

    monkeypatch.setattr(stage2_data_module, "load_source_region_flow_splits", fail_if_called)
    prepared = stage2_data_module.prepare_city_data(
        city="target_without_flow",
        hierarchy_cache_dir="unused",
        lappe_cache_dir=None,
        road_k=4,
        syntax_k=4,
        region_k=4,
        require_targets=False,
        norm_flow_root=None,
        seq_length=24,
    )
    assert prepared.targets is None


def test_lappe_preserves_weighted_undirected_spectrum_inputs():
    edge_index = torch.tensor([[0, 1, 1], [1, 0, 2]], dtype=torch.long)
    edge_weight = torch.tensor([2.0, 3.0, 4.0])
    edge_type = torch.tensor([0, 1, 2], dtype=torch.long)
    result = compute_sparse_laplacian_eigenpairs(
        edge_index, 3, 2, edge_weight=edge_weight, edge_type=edge_type
    )
    assert result.edge_weight_pe is not None
    assert result.edge_type_pe is not None
    assert torch.allclose(result.edge_weight_pe, torch.tensor([5.0, 5.0, 4.0, 4.0], dtype=result.edge_weight_pe.dtype))
    assert result.metadata["weighted_pe"] is True
    assert result.metadata["edge_type_pe_hash"]
