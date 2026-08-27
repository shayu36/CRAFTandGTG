from pathlib import Path

import torch

from hcfm.adversarial import assert_optimizer_covers
from hcfm.checkpoint import load_checkpoint, save_checkpoint
from hcfm.data import SourceOnlyNormalizer, collate_city_snapshots
from hcfm.data import HierarchicalCityDataset
from hcfm.engine import HCFMTrainer
from hcfm.model import HCFMModel
from hcfm_helpers import tiny_config, tiny_sample, tiny_target_static


def fit_normalizers(sample):
    macro = sample["macro_flow"].permute(0, 2, 1)
    micro = sample["micro_flow"].permute(0, 2, 1)
    return {
        "macro_normalizer": SourceOnlyNormalizer().fit(
            macro, cities=["source"], source_cities=["source"], split="train",
            feature_order=["in_flow", "out_flow"], data_version="tiny-v1",
        ),
        "micro_count_normalizer": SourceOnlyNormalizer().fit(
            micro, cities=["source"], source_cities=["source"], split="train",
            feature_order=["road_passage_count"], data_version="tiny-v1",
        ),
    }


def test_full_hcfm_forward_backward_step_checkpoint_and_euler(tmp_path: Path):
    torch.manual_seed(11)
    config = tiny_config("independent")
    sample = tiny_sample()
    batch = collate_city_snapshots([sample])
    target = tiny_target_static()
    normalizers = fit_normalizers(sample)
    model = HCFMModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    assert_optimizer_covers(model, optimizer)
    trainer = HCFMTrainer(model, optimizer, normalizers, config)
    before = model.micro_vector_field.output[-1].weight.detach().clone()
    log = trainer.train_step(
        batch, target, source_city_label=0, target_city_label=1,
        source_cost_target=torch.ones(4, 2),
        source_cost_mask=torch.ones(4, dtype=torch.bool),
        generator=torch.Generator().manual_seed(4),
    )
    assert set(["L_macro", "L_micro", "L_cross_scale", "total_loss"]).issubset(log)
    assert before.ne(model.micro_vector_field.output[-1].weight.detach()).any()
    path = tmp_path / "hcfm.pth"
    save_checkpoint(
        path, model=model, optimizer=optimizer, config=config, normalizers=normalizers,
        data_version="tiny-v1", step=1,
    )
    reloaded = HCFMModel(config)
    reloaded_optimizer = torch.optim.AdamW(reloaded.parameters(), lr=1e-3)
    meta = load_checkpoint(
        path, model=reloaded, optimizer=reloaded_optimizer, normalizers=normalizers,
        expected_data_version="tiny-v1",
    )
    assert meta["step"] == 1
    # 生成条件需要处于训练时相同的 macro 归一化空间。
    batch["reference"] = normalizers["macro_normalizer"].transform(
        batch["reference"].permute(0, 1, 3, 2)
    ).permute(0, 1, 3, 2)
    reloaded.eval()
    with torch.no_grad():
        macro, micro, stats = reloaded.generate(
            batch, steps=2, solver="euler", generator=torch.Generator().manual_seed(5)
        )
    assert macro.shape == (1, 3, 2, 4) and micro.shape == (1, 4, 1, 4)
    assert stats.nfe == 2 and torch.isfinite(macro).all() and torch.isfinite(micro).all()


def test_hierarchy_macro_ablation_requires_no_micro_dynamic():
    config = tiny_config("independent")
    config["generate_micro"] = False
    config["use_micro_adversarial"] = False
    for name in ("fm_micro", "cross_state", "cross_velocity", "topology", "cost", "rank", "semantic_domain", "domain", "orthogonal"):
        config["loss"][name] = 0.0
    config["training"]["phase"] = "macro_fm"
    sample = tiny_sample()
    sample.pop("micro_flow")
    sample.pop("road_mask")
    HierarchicalCityDataset([sample], 4, require_micro=False)
    batch = collate_city_snapshots([sample])
    macro_normalizer = SourceOnlyNormalizer().fit(
        sample["macro_flow"].permute(0, 2, 1),
        cities=["source"], source_cities=["source"], split="train",
        feature_order=["in_flow", "out_flow"], data_version="tiny-v1",
    )
    model = HCFMModel(config)
    assert model.micro_vector_field is None
    assert model.road_adversarial is None
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = HCFMTrainer(
        model, optimizer, {"macro_normalizer": macro_normalizer}, config
    )
    log = trainer.train_step(
        batch, tiny_target_static(), source_city_label=0, target_city_label=1,
        generator=torch.Generator().manual_seed(9),
    )
    assert torch.isfinite(torch.tensor(log["total_loss"]))
    assert log["fm_micro"] == 0.0 and log["cross_state"] == 0.0


def test_full_hcfm_coupled_prior_velocity_step():
    config = tiny_config("coupled")
    sample = tiny_sample()
    batch = collate_city_snapshots([sample])
    normalizers = fit_normalizers(sample)
    model = HCFMModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = HCFMTrainer(model, optimizer, normalizers, config)
    log = trainer.train_step(
        batch, tiny_target_static(), source_city_label=0, target_city_label=1,
        source_cost_target=torch.ones(4, 2),
        source_cost_mask=torch.ones(4, dtype=torch.bool),
        generator=torch.Generator().manual_seed(12),
    )
    assert torch.isfinite(torch.tensor(log["cross_velocity"]))
    assert log["L_cross_scale"] >= 0.0
