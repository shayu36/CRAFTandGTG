import pytest
import torch

from hcfm.hierarchy import aggregate_micro_to_macro, build_boundary_operators
from hcfm.losses import cross_state_loss, cross_velocity_loss, topology_difference_loss


def identity(value):
    return value


def test_exact_physical_state_consistency_near_zero():
    b_in, b_out = build_boundary_operators([-1, 0, 1], [0, 1, -1], 2)
    micro = torch.tensor([1.0, 2.0, 3.0]).view(1, 3, 1, 1)
    macro = aggregate_micro_to_macro(micro, b_in, b_out)
    loss, _, aggregated = cross_state_loss(
        macro, micro, macro, b_in, b_out,
        torch.ones(1, 2, dtype=torch.bool), torch.ones(1, 3, dtype=torch.bool),
        identity, identity,
    )
    assert float(loss) < 1e-8 and torch.equal(aggregated, macro)


def test_wrong_road_direction_nonzero_state_loss():
    good_in, good_out = build_boundary_operators([-1, 0, 1], [0, 1, -1], 2)
    bad_in, bad_out = build_boundary_operators([0, 1, -1], [-1, 0, 1], 2)
    micro = torch.tensor([1.0, 2.0, 4.0]).view(1, 3, 1, 1)
    macro = aggregate_micro_to_macro(micro, good_in, good_out)
    loss, _, _ = cross_state_loss(
        macro, micro, macro, bad_in, bad_out,
        torch.ones(1, 2, dtype=torch.bool), torch.ones(1, 3, dtype=torch.bool),
        identity, identity,
    )
    assert float(loss) > 0.1


def test_padding_road_does_not_contribute_state_loss():
    b_in, b_out = build_boundary_operators([-1, 0], [0, -1], 1)
    micro = torch.tensor([2.0, 999.0]).view(1, 2, 1, 1)
    macro = torch.tensor([2.0, 0.0]).view(1, 1, 2, 1)
    loss, _, _ = cross_state_loss(
        macro, micro, macro, b_in, b_out,
        torch.ones(1, 1, dtype=torch.bool), torch.tensor([[True, False]]), identity, identity,
    )
    # 第二条 road mask=False，出流差仍会反映宏观给定 0，与其 999 无关。
    micro2 = micro.clone(); micro2[:, 1] = 12345
    loss2, _, _ = cross_state_loss(
        macro, micro2, macro, b_in, b_out,
        torch.ones(1, 1, dtype=torch.bool), torch.tensor([[True, False]]), identity, identity,
    )
    assert torch.equal(loss, loss2)


def test_topology_difference_matches_true_not_equal_neighbor_constraint():
    edge = torch.tensor([[0, 1], [1, 2]])
    true = torch.tensor([1.0, 3.0, 8.0]).view(1, 3, 1, 1)
    mask = torch.ones(1, 3, dtype=torch.bool)
    assert float(topology_difference_loss(true, true, edge, mask)) == 0.0
    flat = torch.ones_like(true) * 3
    assert float(topology_difference_loss(flat, true, edge, mask)) > 0


def test_independent_cross_velocity_raises():
    b_in, b_out = build_boundary_operators([-1], [0], 1)
    with pytest.raises(ValueError, match="independent"):
        cross_velocity_loss(
            torch.zeros(1, 1, 2, 1), torch.zeros(1, 1, 1, 1), b_in, b_out,
            torch.ones(1, 1, dtype=torch.bool), prior_mode="independent", weight=0.1,
        )
