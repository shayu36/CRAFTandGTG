import torch
import pytest

from three_layer_rag import (
    HierarchicalThreeLayerRAG,
    STAGE2_LAPPE_VERSION,
    STAGE2_SPECTRAL_VERSION,
    ThreeLayerRAGInputs,
    ThreeLayerRAGMemory,
    load_stage2_low_features,
    RAGTrainer,
    load_rag_checkpoint,
    save_rag_checkpoint,
)


def _snapshot(city: str, split: str = "train", offset: float = 0.0):
    nodes = {"road": 4, "syntax": 2, "region": 1}
    low = {layer: torch.randn(count, 8) + offset for layer, count in nodes.items()}
    temporal = {
        "road": torch.randn(4, 3, 6) + offset,
        "syntax": torch.randn(2, 2, 6) + offset,
        "region": torch.randn(1, 2, 6) + offset,
    }
    value_temporal = {layer: value + 3.0 for layer, value in temporal.items()}
    return ThreeLayerRAGInputs(
        city_id=city,
        split=split,
        low_features=low,
        temporal_features=temporal,
        calendar={"month": 11, "weekday": 1, "start_hour": 8, "holiday": 0},
        parent_index={
            "road_to_syntax": torch.tensor([0, 0, 1, 1]),
            "syntax_to_region": torch.tensor([0, 0]),
        },
        value_temporal_features=value_temporal,
    )


def _model():
    return HierarchicalThreeLayerRAG(
        low_dim=8,
        temporal_channels={"road": 3, "syntax": 2, "region": 2},
        temporal_dim=6,
        retrieval_dim=12,
        top_k=2,
        metric="cosine",
    )


def test_three_layer_rag_returns_all_branches_and_finite_gradients():
    torch.manual_seed(11)
    model = _model()
    memory = model.build_memory(
        [_snapshot("beijing"), _snapshot("chengdushi", offset=1.0)],
        source_cities=["beijing", "chengdushi"],
    )
    output = model(_snapshot("xianshi", "test", offset=0.2), memory=memory)
    assert output["R_road"].shape == (1, 4, 3, 6)
    assert output["R_syntax"].shape == (1, 2, 2, 6)
    assert output["R_region"].shape == (1, 1, 2, 6)
    assert output["R_road_time"].shape == (1, 4, 6)
    assert all(torch.isfinite(output[name]).all() for name in (
        "R_road", "R_syntax", "R_region", "query_road", "query_syntax", "query_region"
    ))
    loss = sum(output[f"query_{layer}"].square().mean() for layer in ("road", "syntax", "region"))
    loss = loss + sum(output[f"R_{layer}"].square().mean() for layer in ("road", "syntax", "region"))
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients and all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_rag_memory_is_source_train_only_and_round_trips(tmp_path):
    model = _model()
    memory = model.build_memory([_snapshot("beijing")], source_cities=["beijing"])
    path = tmp_path / "memory.pt"
    memory.save(path)
    loaded = ThreeLayerRAGMemory.load(path)
    assert loaded.source_cities == ("beijing",)
    assert loaded.snapshots[0].city_id == "beijing"
    assert loaded.snapshots[0].value_temporal_features is not None
    with pytest.raises(ValueError, match="RAG 泄漏防护"):
        ThreeLayerRAGMemory.from_snapshots(
            [_snapshot("xianshi", "test")], source_cities=["xianshi"]
        )


def test_high_frequency_features_are_rejected():
    model = _model()
    memory = model.build_memory([_snapshot("beijing")], source_cities=["beijing"])
    with pytest.raises(ValueError, match="高频"):
        model(
            _snapshot("xianshi", "test"), memory=memory,
            high_features={"H_road_high": torch.zeros(1)},
        )


def test_graphgps_adapter_reads_low_bands_without_high_band_dependency():
    snapshot = _snapshot("xianshi", "test")
    output = {
        "H_road_low": snapshot.low_features["road"],
        "H_syntax_low": snapshot.low_features["syntax"],
        "H_region_low": snapshot.low_features["region"],
    }
    converted = ThreeLayerRAGInputs.from_graphgps_output(
        output,
        snapshot.temporal_features,
        snapshot.calendar,
        city_id="xianshi",
        split="test",
        parent_index=snapshot.parent_index,
        value_temporal_features=snapshot.value_temporal_features,
    )
    assert set(converted.low_features) == {"road", "syntax", "region"}


def test_calendar_mismatch_has_no_silent_candidate_fallback():
    model = _model()
    memory = model.build_memory([_snapshot("beijing")], source_cities=["beijing"])
    query = _snapshot("xianshi", "test")
    query = ThreeLayerRAGInputs(
        city_id=query.city_id, split=query.split,
        low_features=query.low_features, temporal_features=query.temporal_features,
        calendar={"month": 12, "weekday": 1, "start_hour": 8},
        parent_index=query.parent_index,
    )
    with pytest.raises(LookupError, match="无 source-train 候选"):
        model(query, memory=memory)


def test_sparse_weighted_parent_operator_is_used_for_multi_parent_context():
    model = _model()
    source = _snapshot("beijing")
    target = _snapshot("xianshi", "test")
    operator = {
        "road_to_syntax": {
            "edge_index": torch.tensor([[0, 0, 1, 1], [0, 1, 2, 3]]),
            "weight": torch.tensor([1.0, 1.0, 1.0, 1.0]),
        },
        "syntax_to_region": {
            "edge_index": torch.tensor([[0, 0, 0], [0, 1, 1]]),
            "weight": torch.tensor([0.2, 0.3, 0.7]),
        },
    }
    source = ThreeLayerRAGInputs(
        city_id=source.city_id, split=source.split,
        low_features=source.low_features, temporal_features=source.temporal_features,
        calendar=source.calendar, parent_operator=operator,
        value_temporal_features=source.value_temporal_features,
    )
    target = ThreeLayerRAGInputs(
        city_id=target.city_id, split=target.split,
        low_features=target.low_features, temporal_features=target.temporal_features,
        calendar=target.calendar, parent_operator=operator,
        value_temporal_features=target.value_temporal_features,
    )
    memory = model.build_memory([source], source_cities=["beijing"])
    output = model(target, memory=memory)
    assert output["R_syntax"].shape == (1, 2, 2, 6)
    assert torch.isfinite(output["R_road"]).all()


def test_history_and_value_sequences_cannot_be_silently_reused():
    snapshot = _snapshot("beijing")
    legacy = ThreeLayerRAGInputs(
        city_id=snapshot.city_id, split=snapshot.split,
        low_features=snapshot.low_features, temporal_features=snapshot.temporal_features,
        calendar=snapshot.calendar, parent_index=snapshot.parent_index,
    )
    with pytest.raises(ValueError, match="value_temporal_features"):
        HierarchicalThreeLayerRAG.build_memory([legacy], source_cities=["beijing"])


def test_memory_graph_identity_is_per_city_and_strict(tmp_path):
    first = _snapshot("beijing")
    second = _snapshot("chengdushi", offset=1.0)
    required = {
        "joint_graph_hash": "hash",
        "checkpoint_fingerprint": "checkpoint",
        "static_feature_version": "three-layer-start-road-v2",
        "spectral_feature_version": STAGE2_SPECTRAL_VERSION,
        "lappe_version": STAGE2_LAPPE_VERSION,
        "weighted_pe": True,
        "weighted_spectrum_hash": "a" * 64,
        "global_attention_scope": "joint",
        "road_node_range": (0, 4),
        "syntax_node_range": (4, 6),
        "region_node_range": (6, 7),
    }
    first = ThreeLayerRAGInputs(**{**first.__dict__, "graph_metadata": {**required, "city_id": "beijing"}})
    second_meta = {**required, "city_id": "chengdushi", "joint_graph_hash": "other-hash"}
    second = ThreeLayerRAGInputs(**{**second.__dict__, "graph_metadata": second_meta})
    memory = ThreeLayerRAGMemory.from_snapshots(
        [first, second], source_cities=["beijing", "chengdushi"], require_graph_identity=True
    )
    path = tmp_path / "strict-memory.pt"
    memory.save(path)
    loaded = ThreeLayerRAGMemory.load(path)
    assert loaded.graph_identity["beijing"]["joint_graph_hash"] == "hash"
    assert loaded.graph_identity["chengdushi"]["joint_graph_hash"] == "other-hash"


def test_stage2_low_feature_loader_rejects_old_version_and_reads_weighted_v3(tmp_path):
    payload = {
        "format_version": STAGE2_SPECTRAL_VERSION,
        "city_id": "beijing",
        "checkpoint_fingerprint": "a" * 64,
        "joint_graph_hash": "hash",
        "static_feature_version": "three-layer-start-road-v2",
        "lappe_version": STAGE2_LAPPE_VERSION,
        "weighted_pe": True,
        "weighted_spectrum_hash": "a" * 64,
        "global_attention_scope": "joint",
        "road_node_range": (0, 4), "syntax_node_range": (4, 6), "region_node_range": (6, 7),
        "H_road_low": torch.zeros(4, 8),
        "H_syntax_low": torch.zeros(2, 8),
        "H_region_low": torch.zeros(1, 8),
    }
    path = tmp_path / "features.pt"
    torch.save(payload, path)
    low, identity = load_stage2_low_features(path, expected_city_id="beijing")
    assert low["road"].shape == (4, 8)
    assert identity["spectral_feature_version"] == STAGE2_SPECTRAL_VERSION
    old_path = tmp_path / "old-v2.pt"
    torch.save({**payload, "format_version": "three-layer-joint-graphgps-spectral-features-v2"}, old_path)
    with pytest.raises(ValueError, match="旧无权谱特征"):
        load_stage2_low_features(old_path)


def test_rag_filters_calendar_and_loco_before_source_encoding():
    model = _model()
    eligible = _snapshot("beijing")
    excluded = _snapshot("xianshi", offset=1.0)
    wrong_calendar = _snapshot("chengdushi", offset=2.0)
    wrong_calendar = ThreeLayerRAGInputs(**{
        **wrong_calendar.__dict__,
        "calendar": {"month": 12, "weekday": 4, "start_hour": 9, "holiday": 0},
    })
    memory = model.build_memory(
        [eligible, excluded, wrong_calendar],
        source_cities=["beijing", "xianshi", "chengdushi"],
    )
    output = model(_snapshot("xianshi", "test", offset=0.3), memory=memory)
    assert output["num_encoded_source_snapshots"] == 1
    assert output["retrieval"]["road"]["city_names"] == [["beijing"]]


def test_chunked_retrieval_matches_large_chunk_result():
    torch.manual_seed(41)
    small = _model().eval()
    large = _model().eval()
    large.load_state_dict(small.state_dict())
    small.candidate_chunk_size = 1
    large.candidate_chunk_size = 10_000
    snapshots = [_snapshot("beijing"), _snapshot("chengdushi", offset=1.0)]
    memory = small.build_memory(
        snapshots, source_cities=["beijing", "chengdushi"]
    )
    query = _snapshot("xianshi", "test", offset=0.2)
    left = small(query, memory=memory)
    right = large(query, memory=memory)
    for layer in ("road", "syntax", "region"):
        assert torch.allclose(left[f"R_{layer}"], right[f"R_{layer}"], atol=1e-6)
        assert torch.allclose(
            left["retrieval"][layer]["weights"],
            right["retrieval"][layer]["weights"],
            atol=1e-6,
        )


def test_candidate_city_diagnostics_are_kept_per_batch_row():
    model = _model()
    memory = model.build_memory(
        [
            _snapshot("beijing"),
            _snapshot("chengdushi", offset=1.0),
            _snapshot("xianshi", offset=2.0),
        ],
        source_cities=["beijing", "chengdushi", "xianshi"],
    )
    first = _snapshot("xianshi", "test", offset=0.2)
    second = _snapshot("chengdushi", "test", offset=0.4)
    low = {
        layer: torch.stack([first.low_features[layer], second.low_features[layer]])
        for layer in ("road", "syntax", "region")
    }
    temporal = {
        layer: torch.stack([
            first.temporal_features[layer], second.temporal_features[layer]
        ])
        for layer in ("road", "syntax", "region")
    }
    calendar = {
        "month": torch.tensor([11, 11]),
        "weekday": torch.tensor([1, 1]),
        "start_hour": torch.tensor([8, 8]),
        "holiday": torch.tensor([0, 0]),
    }
    output = model(
        low,
        temporal,
        calendar,
        memory=memory,
        target_city=["xianshi", "chengdushi"],
        parent_index=first.parent_index,
    )
    road_diagnostics = output["retrieval"]["road"]
    assert road_diagnostics["city_names"] == [
        ["beijing", "chengdushi"],
        ["beijing", "xianshi"],
    ]
    assert road_diagnostics["candidate_city_counts"] == [[4, 4], [4, 4]]


def test_joint_trainer_and_checkpoint_round_trip(tmp_path):
    model = _model()
    source = _snapshot("beijing")
    query = _snapshot("xianshi", "test")
    memory = model.build_memory([source], source_cities=["beijing"])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = RAGTrainer(model, optimizer)
    _, metrics = trainer.train_step(
        query,
        memory=memory,
        objective=lambda output: sum(output[f"R_{layer}"].square().mean() for layer in ("road", "syntax", "region")),
    )
    assert metrics["rag_loss"] >= 0
    path = tmp_path / "rag.pt"
    save_rag_checkpoint(path, model, optimizer=optimizer, step=3)
    loaded = load_rag_checkpoint(path, model, optimizer=optimizer)
    assert loaded["step"] == 3
