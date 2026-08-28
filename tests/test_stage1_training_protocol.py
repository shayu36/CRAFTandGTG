"""第一阶段多 Source TFA/CCA 训练协议测试。"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "craft_integrated"))

import rep_model  # noqa: E402


def _model():
    return rep_model.GTAggregator({
        "device": "cpu",
        "rep_dim": 4,
        "raw_feature_dim": 45,
        "static_structure_mode": "three_layer",
        "road_feature_mode": "topology_only",
        "cca_metric": "cosine",
        "road_gat_layers": 1,
        "road_gat_heads": 1,
        "syntax_gat_layers": 1,
        "syntax_gat_heads": 1,
        "use_sim_loss": True,
        "use_w_loss": True,
    })


def test_three_layer_tfa_is_citywise_and_cca_uses_full_source(monkeypatch):
    model = _model()
    source_a = SimpleNamespace(
        city="a",
        reps=torch.arange(12, dtype=torch.float32).reshape(3, 4),
        value=torch.ones(2, 48),
        value_region_ids=torch.tensor([0, 2], dtype=torch.long),
    )
    source_b = SimpleNamespace(
        city="b",
        reps=torch.arange(8, dtype=torch.float32).reshape(2, 4) + 20,
        value=torch.ones(1, 48),
        value_region_ids=torch.tensor([1], dtype=torch.long),
    )
    target = SimpleNamespace(
        city="target",
        reps=torch.arange(16, dtype=torch.float32).reshape(4, 4) + 40,
    )
    model.encode_graph = lambda graph: graph.reps

    tfa_calls = []

    def fake_tfa(embeddings, values):
        tfa_calls.append((embeddings.detach().clone(), values.detach().clone()))
        return torch.tensor(float(len(tfa_calls)), dtype=embeddings.dtype)

    cca_call = {}

    def fake_cca(src_emb, trg_emb, metric, src_marginals=None, trg_marginals=None):
        cca_call.update({
            "src_emb": src_emb.detach().clone(),
            "trg_emb": trg_emb.detach().clone(),
            "metric": metric,
            "src_marginals": src_marginals.detach().clone(),
            "trg_marginals": trg_marginals,
        })
        return torch.tensor(3.0)

    monkeypatch.setattr(rep_model, "self_sim_loss", fake_tfa)
    monkeypatch.setattr(rep_model, "wasserstein_loss", fake_cca)
    loss, items = model.calc_loss({
        "src_graphs": [source_a, source_b],
        "trg_graphs": [target],
    })

    assert items["sim_loss"] == pytest.approx(1.5)
    assert loss.item() == pytest.approx(4.5)
    assert [call[0].shape[0] for call in tfa_calls] == [2, 1]
    assert torch.equal(tfa_calls[0][0], source_a.reps[[0, 2]])
    assert torch.equal(tfa_calls[1][0], source_b.reps[[1]])

    # CCA 必须看到两个 Source 城市的全部 3+2 个静态 Region。
    assert cca_call["src_emb"].shape == (5, 4)
    assert torch.equal(cca_call["src_emb"], torch.cat([source_a.reps, source_b.reps]))
    assert cca_call["trg_emb"].shape == (4, 4)
    assert cca_call["metric"] == "cosine"
    expected = torch.tensor([1 / 6, 1 / 6, 1 / 6, 1 / 4, 1 / 4])
    assert torch.allclose(cca_call["src_marginals"], expected)
    assert torch.allclose(cca_call["src_marginals"][:3].sum(), torch.tensor(0.5))
    assert torch.allclose(cca_call["src_marginals"][3:].sum(), torch.tensor(0.5))
    assert cca_call["trg_marginals"] is None


def test_three_layer_cca_metric_cannot_fall_back_to_euclidean():
    with pytest.raises(ValueError, match="必须为 cosine"):
        rep_model.GTAggregator({
            "device": "cpu",
            "rep_dim": 4,
            "static_structure_mode": "three_layer",
            "road_feature_mode": "topology_only",
            "cca_metric": "euclidean",
        })


def test_wasserstein_cosine_metric_builds_one_minus_cosine_cost(monkeypatch):
    captured = {}

    def fake_emd(a, b, M):
        captured["a"] = a.copy()
        captured["b"] = b.copy()
        captured["M"] = M.copy()
        # A valid transport plan for two equally weighted samples.
        return torch.tensor([[0.5, 0.0], [0.0, 0.5]], dtype=torch.float64).numpy()

    monkeypatch.setattr(rep_model.ot, "emd", fake_emd)
    source = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    rep_model.wasserstein_loss(source, target, metric="cosine")

    expected = 1.0 - torch.tensor([
        [1.0, 2 ** -0.5],
        [0.0, 2 ** -0.5],
    ]).numpy()
    assert captured["M"].shape == (2, 2)
    assert captured["M"] == pytest.approx(expected)
