import json
import math

import pytest
import torch
from torch import nn

from three_layer_diffusion import (
    CityGraphBucketBatchSampler,
    DIFFUSION_CONTRACT_VERSION,
    ConditionalGaussianDiffusion1D,
    DynamicNormalizer,
    HierarchicalThreeLayerDiffusion,
    ModelEMA,
    ThreeLayerRAGDiffusionSystem,
    broadcast_parent_to_children,
    cosine_beta_schedule,
    flatten_node_conditions,
    flatten_nodes,
    linear_beta_schedule,
    load_stage2_high_features,
    load_diffusion_checkpoint,
    save_diffusion_checkpoint,
    unflatten_nodes,
)
from three_layer_rag import HierarchicalThreeLayerRAG, ThreeLayerRAGInputs


class TinyEstimator(nn.Module):
    def __init__(self, channels: int, use_self_cond: bool = False):
        super().__init__()
        self.channels = channels
        self.use_self_cond = use_self_cond
        self.scale = nn.Parameter(torch.tensor(0.0))
        self.self_condition_seen = False

    def forward(self, value, step, condition, x_self_cond=None):
        self.self_condition_seen = self.self_condition_seen or x_self_cond is not None
        cond = condition.mean(dim=-1, keepdim=True).unsqueeze(-1)
        output = self.scale * value + 0.01 * cond
        if x_self_cond is not None:
            output = output + 0.001 * x_self_cond
        return output.expand_as(value)


def _gaussian(channels=2, *, steps=4, sampling=2, self_cond=False):
    estimator = TinyEstimator(channels, self_cond)
    return ConditionalGaussianDiffusion1D(
        estimator,
        data_channels=channels,
        seq_length=8,
        time_steps=steps,
        sampling_time_steps=sampling,
        beta_schedule="cosine",
        ddim_sampling_eta=0.0,
        use_self_cond=self_cond,
        clip_x0=False,
    )


def _operator():
    return {
        "road_to_syntax": {
            "edge_index": torch.tensor([[0, 0, 1, 1], [0, 1, 2, 3]]),
            "weight": torch.tensor([1.0, 1.0, 1.0, 1.0]),
        },
        "syntax_to_region": {
            "edge_index": torch.tensor([[0, 0], [0, 1]]),
            "weight": torch.tensor([0.25, 0.75]),
        },
    }


def _snapshot(city="target", split="test", offset=0.0):
    low = {
        "road": torch.randn(4, 4) + offset,
        "syntax": torch.randn(2, 4) + offset,
        "region": torch.randn(1, 4) + offset,
    }
    history = {
        "road": torch.randn(4, 3, 8) + offset,
        "syntax": torch.randn(2, 3, 8) + offset,
        "region": torch.randn(1, 2, 8) + offset,
    }
    value = {layer: tensor + 0.5 for layer, tensor in history.items()}
    return ThreeLayerRAGInputs(
        city_id=city,
        split=split,
        low_features=low,
        temporal_features=history,
        value_temporal_features=value,
        calendar={"month": 1, "weekday": 2, "start_hour": 3, "holiday": 0},
        parent_operator=_operator(),
    ).validate()


def _rag():
    return HierarchicalThreeLayerRAG(
        low_dim=4,
        temporal_channels={"road": 3, "syntax": 3, "region": 2},
        temporal_dim=4,
        retrieval_dim=8,
        seq_length=8,
        top_k=2,
        metric="cosine",
    )


def _hierarchical(*, road_nodes=4, self_cond=False, sampling=2):
    return HierarchicalThreeLayerDiffusion(
        high_dim=4,
        seq_length=8,
        cond_dim=8,
        temporal_dim=4,
        parent_temporal_dim=4,
        time_steps=4,
        sampling_time_steps=sampling,
        beta_schedule="cosine",
        use_self_cond=self_cond,
        init_dim=8,
        base_dim=8,
        dim_mults=(1, 2),
        dropout=0.0,
        attention_dim_head=4,
        attention_heads=1,
    )


def _direct_inputs(road_nodes=4):
    counts = {"region": 1, "syntax": 2, "road": road_nodes}
    channels = {"region": 2, "syntax": 3, "road": 3}
    high = {f"H_{layer}_high": torch.randn(count, 4) for layer, count in counts.items()}
    rag = {
        f"R_{layer}": torch.randn(1, count, channels[layer], 8)
        for layer, count in counts.items()
    }
    future = {
        layer: torch.randn(count, channels[layer], 8) for layer, count in counts.items()
    }
    road_parent = torch.arange(road_nodes) % 2
    operator = {
        "syntax_to_region": _operator()["syntax_to_region"],
        "road_to_syntax": {
            "edge_index": torch.stack([road_parent, torch.arange(road_nodes)]),
            "weight": torch.ones(road_nodes),
        },
    }
    return high, rag, future, operator


def _identity():
    return {
        "graphgps_checkpoint_fingerprint": "a" * 64,
        "joint_graph_hashes": {"source": "hash-a", "target": "hash-b"},
        "static_feature_version": "three-layer-start-road-v2",
        "spectral_feature_version": "three-layer-joint-graphgps-spectral-features-v2",
        "rag_memory_version": "three-layer-hierarchical-rag-v2",
        "dynamic_normalizer_fingerprint": "b" * 64,
    }


def test_craft_beta_schedules_match_reference_formulas():
    steps = 500
    expected_linear = torch.linspace(
        1000 / steps * 0.0001, 1000 / steps * 0.02, steps, dtype=torch.float64
    )
    assert torch.equal(linear_beta_schedule(steps), expected_linear)
    s = 0.008
    x = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
    cumulative = torch.cos(((x / steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    cumulative = cumulative / cumulative[0]
    expected_cosine = (1 - cumulative[1:] / cumulative[:-1]).clamp(0, 0.999)
    assert torch.allclose(cosine_beta_schedule(steps), expected_cosine)


def test_diffusion_coefficients_are_buffers_and_q_sample_is_finite():
    diffusion = _gaussian()
    buffers = dict(diffusion.named_buffers())
    assert "betas" in buffers and "posterior_mean_coef1" in buffers
    x0 = torch.randn(3, 2, 8)
    timestep = torch.tensor([0, 1, 3], dtype=torch.long)
    sampled = diffusion.q_sample(x0, timestep, torch.ones_like(x0))
    assert sampled.shape == x0.shape
    assert sampled.device == x0.device
    assert torch.isfinite(sampled).all()


def test_three_layer_channels_and_node_reshape_round_trip():
    model = _hierarchical()
    assert model.region_diffusion.data_channels == 2
    assert model.syntax_diffusion.data_channels == 3
    assert model.road_diffusion.data_channels == 3
    value = torch.randn(2, 5, 3, 8)
    flat, shape = flatten_nodes(value)
    assert flat.shape == (10, 3, 8)
    assert torch.equal(unflatten_nodes(flat, shape), value)
    condition, cond_shape = flatten_node_conditions(torch.randn(2, 5, 7))
    assert condition.shape == (10, 7) and cond_shape == shape


def test_rag_reference_and_high_feature_both_change_diffusion_condition():
    model = _hierarchical()
    conditioner = model.conditioners["region"]
    high = torch.randn(1, 2, 4)
    reference = torch.randn(1, 2, 2, 8, requires_grad=True)
    calendar = {"month": 1, "weekday": 2, "start_hour": 3, "holiday": 0}
    first = conditioner(high, reference, calendar)
    second = conditioner(high + 1.0, reference + 1.0, calendar)
    assert not torch.allclose(first, second)
    first.square().mean().backward()
    assert reference.grad is not None and torch.isfinite(reference.grad).all()


def test_low_features_are_rejected_by_diffusion_and_high_by_rag():
    model = _hierarchical()
    high, rag_output, future, operator = _direct_inputs()
    high["H_region_low"] = torch.zeros(1, 4)
    with pytest.raises(ValueError, match="low"):
        model.training_loss(
            future=future,
            high_features=high,
            rag_outputs=rag_output,
            calendar={"month": 1, "weekday": 2, "start_hour": 3},
            parent_operator=operator,
        )
    rag = _rag()
    memory = rag.build_memory([_snapshot("source", "train")], source_cities=["source"])
    with pytest.raises(ValueError, match="高频"):
        rag(_snapshot(), memory=memory, high_features={"H_road_high": torch.zeros(1)})


def test_sparse_parent_broadcast_uses_transpose_direction_and_weights():
    parent = torch.tensor([[[10.0], [20.0]]])
    operator = {
        "edge_index": torch.tensor([[0, 1, 1], [0, 0, 1]]),
        "weight": torch.tensor([0.25, 0.75, 2.0]),
    }
    children = broadcast_parent_to_children(parent, operator, child_count=2)
    assert children.shape == (1, 2, 1)
    assert torch.allclose(children[0, :, 0], torch.tensor([17.5, 20.0]))


def test_masked_noise_loss_is_normalized_by_valid_elements():
    prediction = torch.zeros(2, 2, 2)
    target = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    mask = torch.tensor([[[1.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]])
    loss = ConditionalGaussianDiffusion1D.masked_epsilon_loss(prediction, target, mask)
    assert torch.allclose(loss, torch.tensor((1.0 + 9.0) / 2.0))


def test_more_road_nodes_do_not_rescale_region_or_syntax_losses():
    torch.manual_seed(4)
    small = _hierarchical(road_nodes=4)
    large = _hierarchical(road_nodes=8)
    large.load_state_dict(small.state_dict())
    small_inputs = _direct_inputs(4)
    large_inputs = _direct_inputs(8)
    for index in (0, 1, 2):
        for layer in ("region", "syntax"):
            key = f"H_{layer}_high" if index == 0 else (f"R_{layer}" if index == 1 else layer)
            large_inputs[index][key] = small_inputs[index][key].clone()
    fixed_noise_small = {
        layer: torch.ones_like(small_inputs[2][layer]) for layer in ("region", "syntax", "road")
    }
    fixed_noise_large = {
        "region": fixed_noise_small["region"],
        "syntax": fixed_noise_small["syntax"],
        "road": torch.ones_like(large_inputs[2]["road"]),
    }
    steps_small = {"region": torch.zeros(1, dtype=torch.long), "syntax": torch.zeros(2, dtype=torch.long), "road": torch.zeros(4, dtype=torch.long)}
    steps_large = {**steps_small, "road": torch.zeros(8, dtype=torch.long)}
    calendar = {"month": 1, "weekday": 2, "start_hour": 3}
    small.eval()
    large.eval()
    first = small.training_loss(
        future=small_inputs[2], high_features=small_inputs[0], rag_outputs=small_inputs[1],
        calendar=calendar, parent_operator=small_inputs[3], noise=fixed_noise_small,
        timesteps=steps_small,
    )
    second = large.training_loss(
        future=large_inputs[2], high_features=large_inputs[0], rag_outputs=large_inputs[1],
        calendar=calendar, parent_operator=large_inputs[3], noise=fixed_noise_large,
        timesteps=steps_large,
    )
    assert torch.allclose(first["layer_losses"]["region"], second["layer_losses"]["region"])
    assert torch.allclose(first["layer_losses"]["syntax"], second["layer_losses"]["syntax"])


def test_target_city_cannot_retrieve_from_its_own_only_memory():
    rag = _rag()
    memory = rag.build_memory([_snapshot("same", "train")], source_cities=["same"])
    with pytest.raises(LookupError, match="无 source-train 候选"):
        rag(_snapshot("same", "test"), memory=memory, target_city="same")


def test_ddpm_and_ddim_tiny_sampling_and_deterministic_eta_zero():
    condition = torch.randn(2, 5)
    initial = torch.randn(2, 2, 8)
    ddpm = _gaussian(sampling=4)
    ddpm_value, ddpm_info = ddpm.sample(condition, initial_noise=initial.clone())
    assert ddpm_value.shape == initial.shape and torch.isfinite(ddpm_value).all()
    assert ddpm_info["method"] == "ddpm" and ddpm_info["sampling_steps"] == 4
    ddim = _gaussian(sampling=2)
    first, info = ddim.sample(condition, initial_noise=initial.clone())
    torch.manual_seed(999)
    second, _ = ddim.sample(condition, initial_noise=initial.clone())
    assert info["method"] == "ddim" and info["sampling_steps"] == 2
    assert torch.equal(first, second)


@pytest.mark.parametrize("enabled", [False, True])
def test_self_conditioning_switch_runs(enabled):
    diffusion = _gaussian(self_cond=enabled)
    x0 = torch.randn(2, 2, 8)
    condition = torch.randn(2, 5)
    loss, _ = diffusion.training_loss(
        x0, condition, force_self_condition=enabled
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert diffusion.estimator.self_condition_seen is enabled


def test_joint_diffusion_loss_backpropagates_to_rag_parameters():
    torch.manual_seed(12)
    rag = _rag()
    memory = rag.build_memory(
        [_snapshot("source-a", "train"), _snapshot("source-b", "train", 1.0)],
        source_cities=["source-a", "source-b"],
    )
    diffusion = _hierarchical(self_cond=False)
    system = ThreeLayerRAGDiffusionSystem(rag, diffusion)
    query = _snapshot("target", "test", 0.2)
    high = {f"H_{layer}_high": torch.randn(query.low_features[layer].shape[0], 4) for layer in ("region", "syntax", "road")}
    output = system.training_loss(query, high_features=high, memory=memory)
    output["loss"].backward()
    gradients = [
        parameter.grad for name, parameter in rag.named_parameters()
        if ("query_projections" in name or "key_projections" in name or "temporal_encoders" in name)
        and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0


def test_ema_and_strict_checkpoint_round_trip(tmp_path):
    system = ThreeLayerRAGDiffusionSystem(_rag(), _hierarchical())
    optimizer = torch.optim.Adam(system.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    ema = ModelEMA(system, decay=0.9)
    ema.update(system)
    path = tmp_path / "stage4.pt"
    save_diffusion_checkpoint(
        path,
        system,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=2,
        global_step=7,
        config={"diffusion": {"contract_version": DIFFUSION_CONTRACT_VERSION}},
        identity=_identity(),
        seed=4,
    )
    restored = ThreeLayerRAGDiffusionSystem(_rag(), _hierarchical())
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-3)
    restored_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(restored_optimizer)
    restored_ema = ModelEMA(restored, decay=0.9)
    payload = load_diffusion_checkpoint(
        path,
        restored,
        ema=restored_ema,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        expected_identity=_identity(),
    )
    assert payload["epoch"] == 2 and payload["global_step"] == 7
    with pytest.raises(ValueError, match="identity"):
        load_diffusion_checkpoint(path, restored, expected_identity={"rag_memory_version": "wrong"})


def test_log1p_zscore_inverse_is_not_craft_minus_one_to_one_mapping(tmp_path):
    path = tmp_path / "normalizer.json"
    payload = {
        "format_version": "three-layer-rag-dynamics-v1",
        "mode": "log1p_zscore",
        "layers": {
            "road": {"mean": [1.0, 2.0, 3.0], "std": [2.0, 2.0, 2.0]},
            "syntax": {"mean": [1.0, 2.0, 3.0], "std": [2.0, 2.0, 2.0]},
            "region": {"mean": [1.0, 2.0], "std": [2.0, 2.0]},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    normalizer = DynamicNormalizer.load(path)
    normalized = torch.zeros(1, 2, 3)
    physical = normalizer.inverse("region", normalized)
    expected = torch.expm1(torch.tensor([1.0, 2.0])).reshape(1, 2, 1).expand_as(physical)
    assert torch.allclose(physical, expected)
    assert not torch.allclose(physical, torch.full_like(physical, 0.5))
    metadata = {"normalization": payload}
    normalizer.validate_bundle_metadata(metadata)
    broken = json.loads(json.dumps(payload))
    broken["layers"]["region"]["mean"][0] += 1.0
    with pytest.raises(ValueError, match="normalizer"):
        normalizer.validate_bundle_metadata({"normalization": broken})


def test_stage2_high_loader_binds_exact_low_node_order(tmp_path):
    low = {"road": torch.randn(4, 4), "syntax": torch.randn(2, 4), "region": torch.randn(1, 4)}
    payload = {
        "format_version": "three-layer-joint-graphgps-spectral-features-v2",
        "city_id": "toy",
        "joint_graph_hash": "hash",
        "checkpoint_fingerprint": "a" * 64,
        "static_feature_version": "three-layer-start-road-v2",
        "road_node_range": (0, 4),
        "syntax_node_range": (4, 6),
        "region_node_range": (6, 7),
    }
    for layer in ("road", "syntax", "region"):
        payload[f"H_{layer}_low"] = low[layer]
        payload[f"H_{layer}_high"] = torch.randn_like(low[layer])
    path = tmp_path / "spectral.pt"
    torch.save(payload, path)
    high, identity = load_stage2_high_features(
        path, expected_city_id="toy", expected_low_features=low
    )
    assert high["H_road_high"].shape == (4, 4)
    assert identity["joint_graph_hash"] == "hash"
    reordered = dict(low)
    reordered["road"] = low["road"].flip(0)
    with pytest.raises(ValueError, match="stable node order"):
        load_stage2_high_features(path, expected_low_features=reordered)


def test_city_graph_bucket_sampler_never_mixes_variable_city_graphs():
    snapshots = [
        _snapshot("a", "train"),
        _snapshot("b", "train"),
        _snapshot("a", "train", 1.0),
        _snapshot("b", "train", 1.0),
    ]
    sampler = CityGraphBucketBatchSampler(snapshots, batch_size=2, shuffle=False)
    batches = list(sampler)
    assert len(batches) == 2
    for batch in batches:
        assert len({snapshots[index].city_id for index in batch}) == 1


def test_road_conditioning_has_no_dense_node_attention():
    model = _hierarchical()
    conditioner = model.conditioners["road"]
    nodes = 1024
    condition = conditioner(
        torch.randn(1, nodes, 4),
        torch.randn(1, nodes, 3, 8),
        {"month": 1, "weekday": 2, "start_hour": 3},
        parent_dynamic=torch.randn(1, nodes, 3, 8),
    )
    assert condition.shape == (1, nodes, 8)
    assert not any(
        tensor.ndim >= 2 and tensor.shape[-2:] == (nodes, nodes)
        for tensor in list(model.parameters()) + list(model.buffers())
    )


def test_tiny_end_to_end_rag_region_syntax_road_generation_ignores_future():
    rag = _rag()
    memory = rag.build_memory([_snapshot("source", "train")], source_cities=["source"])
    system = ThreeLayerRAGDiffusionSystem(rag, _hierarchical())
    query = _snapshot("target", "test")
    changed_value = {
        layer: value.clone() for layer, value in query.value_temporal_features.items()
    }
    changed_value["region"].fill_(1e6)
    query_with_changed_future = ThreeLayerRAGInputs(
        city_id=query.city_id,
        split=query.split,
        low_features=query.low_features,
        temporal_features=query.temporal_features,
        value_temporal_features=changed_value,
        calendar=query.calendar,
        parent_operator=query.parent_operator,
    ).validate()
    high = {
        f"H_{layer}_high": torch.randn(query.low_features[layer].shape[0], 4)
        for layer in ("region", "syntax", "road")
    }
    initial_noise = {
        "region": torch.randn(1, 1, 2, 8),
        "syntax": torch.randn(1, 2, 3, 8),
        "road": torch.randn(1, 4, 3, 8),
    }
    result = system.generate(
        query, high_features=high, memory=memory, initial_noise=initial_noise
    )
    changed = system.generate(
        query_with_changed_future,
        high_features=high,
        memory=memory,
        initial_noise=initial_noise,
    )
    assert result["generation_order"] == ("region", "syntax", "road")
    assert result["used_future_value"] is False
    assert result["generated_region"].shape == (1, 1, 2, 8)
    assert result["generated_syntax"].shape == (1, 2, 3, 8)
    assert result["generated_road"].shape == (1, 4, 3, 8)
    assert all(torch.isfinite(result[f"generated_{layer}"]).all() for layer in ("region", "syntax", "road"))
    for layer in ("region", "syntax", "road"):
        assert torch.equal(result[f"generated_{layer}"], changed[f"generated_{layer}"])
