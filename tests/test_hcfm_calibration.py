import pytest
import torch

from hcfm.calibration import SourceOnlyConservationCalibrator, conservation_gap_report


def test_source_train_calibration_recovers_scale():
    micro = torch.tensor([1.0, 2.0, 3.0, 4.0]).reshape(1, 1, 2, 2)
    macro = micro * torch.tensor([2.0, 3.0]).view(1, 1, 2, 1)
    calibrator = SourceOnlyConservationCalibrator().fit(
        micro, macro, torch.ones(1, 1, dtype=torch.bool),
        cities=["source"], source_cities=["source"], split="train", data_version="v1",
    )
    calibrated = calibrator.apply(micro)
    assert torch.allclose(calibrator.scale, torch.tensor([2.0, 3.0]))
    report = conservation_gap_report(macro, calibrated, torch.ones(1, 1, dtype=torch.bool))
    assert report["absolute_error"] == 0.0 and report["relative_error"] == 0.0


def test_target_or_test_calibration_rejected():
    value = torch.ones(1, 1, 2, 1)
    with pytest.raises(ValueError, match="源城市 train"):
        SourceOnlyConservationCalibrator().fit(
            value, value, torch.ones(1, 1, dtype=torch.bool),
            cities=["target"], source_cities=["source"], split="test", data_version="v1",
        )

