import copy

import pytest
import torch

from hcfm.data import (
    HierarchicalCityDataset, SourceOnlyNormalizer, collate_city_snapshots,
    validate_joint_sample,
)
from hcfm.hierarchy import build_structural_operator
from hcfm.rag import SourceTrainRegionRetriever, assert_retriever_no_target_leakage
from hcfm_helpers import tiny_sample


def test_joint_sample_contract_and_batch_shapes():
    sample = tiny_sample()
    validate_joint_sample(sample, 4)
    dataset = HierarchicalCityDataset([sample], 4)
    batch = collate_city_snapshots([dataset[0]])
    assert batch["macro_flow"].shape == (1, 3, 2, 4)
    assert batch["micro_flow"].shape == (1, 4, 1, 4)
    assert batch["region_x"].shape == (1, 3, 45)


def test_city_time_alignment_rejected():
    sample = tiny_sample()
    sample["dynamic_metadata"]["micro"]["city_id"] = "wrong"
    with pytest.raises(ValueError, match="micro.city_id"):
        validate_joint_sample(sample, 4)


def test_region_and_road_order_bounds_rejected():
    sample = tiny_sample()
    sample["road_edge_index"] = torch.tensor([[0], [4]], dtype=torch.long)
    with pytest.raises(ValueError, match="索引越界"):
        validate_joint_sample(sample, 4)


def test_nan_rejected_not_silently_filled():
    sample = tiny_sample()
    sample["road_x"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN/Inf"):
        validate_joint_sample(sample, 4)


def test_p_struct_row_normalization_and_empty_row():
    p = build_structural_operator(torch.tensor([[2.0, 2.0], [0.0, 0.0]]).numpy())
    sums = torch.sparse.sum(p, dim=1).to_dense()
    assert torch.allclose(sums, torch.tensor([1.0, 0.0]))


def test_source_only_normalizer_and_roundtrip():
    values = torch.tensor([[1.0, 10.0], [3.0, 14.0]])
    norm = SourceOnlyNormalizer().fit(
        values, cities=["source"], source_cities=["source", "other"], split="train",
        feature_order=["in", "out"], data_version="v1",
    )
    assert torch.allclose(norm.inverse(norm.transform(values)), values)
    with pytest.raises(ValueError, match="train split"):
        SourceOnlyNormalizer().fit(
            values, cities=["source"], source_cities=["source"], split="test",
            feature_order=["in", "out"], data_version="v1",
        )
    with pytest.raises(ValueError, match="不是 source cities"):
        SourceOnlyNormalizer().fit(
            values, cities=["target"], source_cities=["source"], split="train",
            feature_order=["in", "out"], data_version="v1",
        )


def test_rag_source_train_only_and_target_leakage():
    rows = [{
        "city_id": "source", "split": "train", "region_id": 0,
        "month": 0, "weekday": 0, "start_hour": 0,
        "feature": [1.0, 0.0], "flow": [[1.0] * 4, [2.0] * 4],
    }]
    retriever = SourceTrainRegionRetriever.fit(
        rows, source_cities=["source"], split="train", seq_length=4
    )
    ref = retriever.query_city(
        torch.tensor([[1.0, 0.0]]).numpy(), month=0, weekday=0, start_hour=0,
        top_k=1, query_city="target",
    )
    assert ref.shape == (1, 2, 4)
    assert_retriever_no_target_leakage(retriever, ["target"])
    with pytest.raises(ValueError, match="非 train"):
        SourceTrainRegionRetriever.fit(rows, source_cities=["source"], split="test", seq_length=4)
    with pytest.raises(ValueError, match="target cities"):
        assert_retriever_no_target_leakage(retriever, ["source"])


def test_batch_size_greater_than_one_rejected():
    with pytest.raises(ValueError, match="batch_size"):
        collate_city_snapshots([tiny_sample(), copy.deepcopy(tiny_sample())])

