from pathlib import Path

import pandas as pd
import pytest
import torch

from hcfm.config import load_config
from hcfm.geo_time import localize_city_timestamps, transform_wkt_geometries
from hcfm.model import MacroFlowMatchingModel


def test_crs_transform_and_roundtrip():
    projected = transform_wkt_geometries(["POINT (-73.98 40.75)"], "EPSG:4326", "EPSG:32618")
    assert projected.crs.to_epsg() == 32618
    back = projected.to_crs(4326).iloc[0]
    assert abs(back.x + 73.98) < 1e-6 and abs(back.y - 40.75) < 1e-6


def test_timezone_required_and_dst_ambiguous_rejected():
    with pytest.raises(ValueError, match="timezone"):
        localize_city_timestamps(["2023-01-01 00:00"], "")
    with pytest.raises(Exception):
        localize_city_timestamps(["2023-11-05 01:30"], "America/New_York")
    localized = localize_city_timestamps(
        ["2023-11-05 01:30"], "America/New_York", ambiguous=True
    )
    assert localized.tz is not None


def test_all_stage2_and_ablation_configs_parse():
    root = Path(__file__).resolve().parents[1]
    paths = list((root / "configs").glob("stage*.yaml")) + list((root / "configs/ablations").glob("*.yaml"))
    assert len(paths) == 13
    for path in paths:
        assert load_config(path)["model_mode"]


def test_macro_flow_matching_forward_backward_and_heun():
    cfg = {
        "model": {
            "seq_length": 4, "rep_dim": 8, "refer_dim": 8, "refer_heads": 2,
            "refer_layers": 1, "hour_dim": 2, "weekday_dim": 2, "month_dim": 2,
        },
        "flow_matching": {"hidden_dim": 8, "num_blocks": 2, "time_dim": 8, "dropout": 0.0},
    }
    model = MacroFlowMatchingModel(cfg)
    batch = {
        "aligned_region_rep": torch.ones(1, 3, 8),
        "reference": torch.ones(1, 3, 2, 4),
        "macro_flow": torch.arange(24, dtype=torch.float32).reshape(1, 3, 2, 4),
        "region_mask": torch.ones(1, 3, dtype=torch.bool),
        "region_edge_index": torch.tensor([[0, 1], [1, 2]]),
        "start_hour": 0, "weekday": 0, "month": 0,
    }
    class Identity:
        transform = staticmethod(lambda value: value)
    loss = model.loss(batch, Identity(), torch.Generator().manual_seed(1))
    loss.backward()
    assert torch.isfinite(loss)
    macro, stats = model.generate(
        batch, steps=2, solver="heun", generator=torch.Generator().manual_seed(2)
    )
    assert macro.shape == batch["macro_flow"].shape and stats.nfe == 4

