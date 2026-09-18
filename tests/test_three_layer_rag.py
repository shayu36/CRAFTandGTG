import torch
import pytest

from three_layer_rag import (
    HierarchicalThreeLayerRAG,
    ThreeLayerRAGInputs,
    ThreeLayerRAGMemory,
)


def _snapshot(city: str, split: str = "train", offset: float = 0.0):
    nodes = {"road": 4, "syntax": 2, "region": 1}
    low = {layer: torch.randn(count, 8) + offset for layer, count in nodes.items()}
    temporal = {
        "road": torch.randn(4, 3, 6) + offset,
        "syntax": torch.randn(2, 2, 6) + offset,
        "region": torch.randn(1, 2, 6) + offset,
    }
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
