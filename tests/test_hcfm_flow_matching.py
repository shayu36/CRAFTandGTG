import pytest
import torch

from hcfm.flow_matching import (
    GraphTemporalVectorField, estimate_endpoint, integrate_coupled_ode,
    masked_mse, sample_priors, straight_path, validate_velocity_consistency,
)
from hcfm.hierarchy import aggregate_micro_to_macro, build_boundary_operators


@pytest.mark.parametrize("time,expected", [(0.0, 0.0), (0.5, 1.0), (1.0, 2.0)])
def test_straight_path_endpoints_and_middle(time, expected):
    x0, x1 = torch.zeros(1, 2, 1, 3), torch.full((1, 2, 1, 3), 2.0)
    state, velocity = straight_path(x0, x1, torch.tensor([time]))
    assert torch.allclose(state, torch.full_like(state, expected))
    assert torch.equal(velocity, torch.full_like(velocity, 2.0))
    assert torch.allclose(estimate_endpoint(state, velocity, torch.tensor([time])), x1)


def test_mask_excludes_padding_and_gradient():
    prediction = torch.tensor([[[[1.0]], [[100.0]]]], requires_grad=True)
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[True, False]])
    loss = masked_mse(prediction, target, mask)
    loss.backward()
    assert float(loss) == 1.0
    assert float(prediction.grad[0, 0]) == 2.0
    assert float(prediction.grad[0, 1]) == 0.0


def test_vector_field_shape_gradient_and_finite():
    field = GraphTemporalVectorField(2, 5, 0, hidden_dim=8, num_blocks=2, time_dim=8, dropout=0)
    state = torch.ones(1, 3, 2, 4, requires_grad=True)
    condition = torch.ones(1, 3, 5)
    edge = torch.tensor([[0, 1], [1, 2]])
    velocity = field(state, torch.tensor([0.25]), condition, edge)
    assert velocity.shape == state.shape and torch.isfinite(velocity).all()
    velocity.sum().backward()
    assert state.grad is not None and torch.isfinite(state.grad).all()


@pytest.mark.parametrize("solver,nfe", [("euler", 4), ("heun", 8)])
def test_solver_constant_velocity_analytic(solver, nfe):
    x0 = torch.zeros(1, 2, 1, 3)
    q0 = torch.zeros(1, 3, 1, 3)
    def field(x, q, _time):
        return torch.ones_like(x) * 2, torch.ones_like(q) * 3
    x1, q1, stats = integrate_coupled_ode(field, x0, q0, steps=4, solver=solver)
    assert torch.allclose(x1, torch.full_like(x1, 2.0))
    assert torch.allclose(q1, torch.full_like(q1, 3.0))
    assert stats.nfe == nfe and not stats.has_nonfinite


def test_solver_reproducible_fixed_initial():
    x0, q0 = torch.zeros(1, 1, 1, 2), torch.ones(1, 1, 1, 2)
    def field(x, q, time):
        return x + time.view(-1, 1, 1, 1), -q
    first = integrate_coupled_ode(field, x0, q0, steps=3, solver="heun")[:2]
    second = integrate_coupled_ode(field, x0, q0, steps=3, solver="heun")[:2]
    assert all(torch.equal(a, b) for a, b in zip(first, second))


def test_independent_velocity_consistency_forbidden():
    with pytest.raises(ValueError, match="independent"):
        validate_velocity_consistency("independent", 0.1)


def test_coupled_prior_and_target_velocity_aggregate():
    b_in, b_out = build_boundary_operators([-1, 0, 1], [0, 1, -1], 2)
    micro_target = torch.arange(6, dtype=torch.float32).reshape(1, 3, 1, 2)
    macro_target = aggregate_micro_to_macro(micro_target, b_in, b_out)
    generator = torch.Generator().manual_seed(7)
    macro_initial, micro_initial = sample_priors(
        micro_target, macro_target, b_in, b_out, "coupled", generator
    )
    assert torch.equal(macro_initial, aggregate_micro_to_macro(micro_initial, b_in, b_out))
    assert torch.allclose(
        macro_target - macro_initial,
        aggregate_micro_to_macro(micro_target - micro_initial, b_in, b_out),
    )
