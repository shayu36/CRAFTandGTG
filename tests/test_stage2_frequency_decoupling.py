"""Stage 2 explicit low/high spectral feature contract tests."""

from __future__ import annotations

import inspect

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
from three_layer_graphgps.spectral_lap_pe import (
    LaplacianEigenpairs,
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
    for module in (
        model.road_input,
        model.road_input.lappe,
        model.road_graphgps,
        model.syntax_input,
        model.syntax_input.lappe,
        model.syntax_graphgps,
        model.region_input,
        model.region_input.lappe,
        model.region_graphgps,
    ):
        _assert_module_has_finite_gradient(module)

    model.zero_grad(set_to_none=True)
    output = model(hierarchy, posenc)
    high_loss = sum(output[f"H_{layer}_high"].square().mean() for layer in ("road", "syntax", "region"))
    high_loss.backward()
    for module in (
        model.road_input,
        model.road_input.lappe,
        model.road_graphgps,
        model.syntax_input,
        model.syntax_input.lappe,
        model.syntax_graphgps,
        model.region_input,
        model.region_input.lappe,
        model.region_graphgps,
    ):
        _assert_module_has_finite_gradient(module)

    model.zero_grad(set_to_none=True)
    output = model(hierarchy, posenc)
    output["pred"].square().mean().backward()
    for module in (
        model.road_input,
        model.road_input.lappe,
        model.road_graphgps,
        model.syntax_input,
        model.syntax_input.lappe,
        model.syntax_graphgps,
        model.region_input,
        model.region_input.lappe,
        model.region_graphgps,
        model.frequency_fusion,
        model.prediction_head,
    ):
        _assert_module_has_finite_gradient(module)


def test_model_frequency_outputs_reconstruct_and_no_dense_projector_source():
    hierarchy = _toy_hierarchy()
    posenc = prepare_hierarchy_lappe(hierarchy, road_k=4, syntax_k=4, region_k=4)
    output = ThreeLayerGraphGPSLapPE(_config()).eval()(hierarchy, posenc)
    expected_shapes = {"road": (8, 16), "syntax": (3, 16), "region": (2, 16)}
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
    assert all(layer.global_attention.mode == "linear" for layer in model.road_graphgps.layers)
    assert all(layer.global_attention.mode == "full" for layer in model.syntax_graphgps.layers)
    assert all(layer.global_attention.mode == "full" for layer in model.region_graphgps.layers)


def test_frequency_config_contract_rejects_invalid_values():
    config = _config()
    config["frequency"]["road_low_modes"] = 5
    with pytest.raises(ValueError, match="不能超过"):
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

    with pytest.raises(ValueError, match="checkpoint fingerprint"):
        load_spectral_features(
            path,
            hierarchy=hierarchy,
            expected_checkpoint_fingerprint="b" * 64,
        )
    changed = _toy_hierarchy()
    changed.road_edge_index = changed.road_edge_index.clone()
    changed.road_edge_index[:, 0] = torch.tensor([0, 2])
    with pytest.raises(ValueError, match="road_graph_hash"):
        load_spectral_features(
            path,
            hierarchy=changed,
            expected_checkpoint_fingerprint=fingerprint,
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
