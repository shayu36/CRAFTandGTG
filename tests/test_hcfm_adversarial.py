import pytest
import torch

from hcfm.adversarial import (
    GradientReversalLayer, RoadAdversarialModule, adversarial_losses,
    assert_optimizer_covers, orthogonal_loss,
)
from hcfm.metrics import representation_metrics


def config():
    return {
        "road_dim": 6, "hidden_dim": 8, "num_layers": 2, "heads": 2,
        "dropout": 0.0, "edge_dim": None, "cost_dim": 2,
        "num_domains": 2, "grl_coefficient": 0.5,
    }


def test_grl_forward_identity_and_backward_sign():
    x = torch.tensor([1.0, -2.0], requires_grad=True)
    y = GradientReversalLayer(0.5)(x)
    assert torch.equal(x, y)
    y.sum().backward()
    assert torch.equal(x.grad, torch.tensor([-0.5, -0.5]))


def test_adversarial_modules_optimizer_and_losses():
    model = RoadAdversarialModule(config())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    assert_optimizer_covers(model, optimizer)
    edge = torch.tensor([[0, 1, 2], [1, 2, 3]])
    source = model(torch.arange(24, dtype=torch.float32).reshape(4, 6), edge)
    target = model(torch.arange(24, dtype=torch.float32).reshape(4, 6) + 0.5, edge)
    cost_target = torch.ones(4, 2)
    losses = adversarial_losses(
        [source, target], [0, 1], source_cost_target=cost_target,
        source_cost_mask=torch.ones(4, dtype=torch.bool),
    )
    total = sum(losses.values())
    total.backward()
    for name in ["semantic_encoder", "domain_encoder"]:
        module = getattr(model.disentangled_encoder, name)
        assert any(p.grad is not None for p in module.parameters())
    assert any(p.grad is not None for p in model.cost_predictor.parameters())
    assert any(p.grad is not None for p in model.semantic_discriminator.parameters())


def test_target_dynamic_labels_not_accepted_by_api():
    # adversarial_losses 只有 source_cost_target 参数，目标输出只参与 domain/orthogonal。
    assert "target_cost_target" not in adversarial_losses.__annotations__


def test_orthogonal_loss_zero_safe():
    loss = orthogonal_loss(torch.zeros(3, 4), torch.zeros(3, 4))
    assert torch.isfinite(loss) and float(loss) == 0.0


def test_optimizer_missing_parameter_detected():
    model = RoadAdversarialModule(config())
    optimizer = torch.optim.Adam(model.road_encoder.parameters(), lr=1e-3)
    with pytest.raises(RuntimeError, match="optimizer 缺少"):
        assert_optimizer_covers(model, optimizer)


def test_representation_metrics_are_reported():
    model = RoadAdversarialModule(config())
    edge = torch.tensor([[0, 1, 2], [1, 2, 3]])
    output = model(torch.arange(24, dtype=torch.float32).reshape(4, 6), edge)
    metrics = representation_metrics(
        semantic=output["semantic"], domain=output["domain"],
        semantic_domain_logits=output["semantic_domain_logits"],
        domain_logits=output["domain_logits"], city_label=0,
        cost_prediction=output["cost"], cost_target=torch.arange(8).reshape(4, 2),
        cost_mask=torch.ones(4, dtype=torch.bool),
    )
    assert {"domain_classification_accuracy", "cost_prediction_mse", "rank_accuracy", "semantic_domain_cosine_similarity"}.issubset(metrics)
