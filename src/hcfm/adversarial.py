"""GTG 道路侧拓扑编码与解耦领域对抗学习。"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, coefficient: float) -> torch.Tensor:
        ctx.coefficient = float(coefficient)
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return -ctx.coefficient * gradient, None


class GradientReversalLayer(nn.Module):
    """前向恒等、反向乘 ``-coefficient``。"""

    def __init__(self, coefficient: float = 1.0):
        super().__init__()
        if coefficient < 0:
            raise ValueError("GRL coefficient 必须非负")
        self.coefficient = float(coefficient)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(value, self.coefficient)


class GATv2MessageLayer(nn.Module):
    """无 PyG 二进制依赖的 GATv2 消息层。

    注意力严格采用 ``a^T LeakyReLU(W_s x_s + W_t x_t + W_e e)``，按目标
    Road 的入边做 softmax。输入 ``x [M,D]``、``edge_index [2,E]``，输出
    ``[M,out_dim]``。这保留 GTG ``TopoAggregator`` 的 GATv2/多头/边属性语义。
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 4,
        edge_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if out_dim <= 0 or heads <= 0:
            raise ValueError("GATv2 out_dim/heads 必须为正")
        self.out_dim, self.heads, self.dropout = out_dim, heads, float(dropout)
        self.lin_src = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.lin_dst = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.lin_edge = nn.Linear(edge_dim, heads * out_dim, bias=False) if edge_dim else None
        self.attention = nn.Parameter(torch.empty(heads, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.attention)

    @staticmethod
    def _segment_softmax(logits: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
        # logits [E,H]。scatter_reduce_ 在 torch>=1.12 可用，且不依赖 torch_scatter ABI。
        expanded = index[:, None].expand_as(logits)
        maxima = torch.full(
            (size, logits.shape[1]), -torch.inf, dtype=logits.dtype, device=logits.device
        )
        maxima.scatter_reduce_(0, expanded, logits, reduce="amax", include_self=True)
        exp = torch.exp(logits - maxima[index])
        denom = torch.zeros_like(maxima)
        denom.scatter_add_(0, expanded, exp)
        return exp / denom[index].clamp_min(torch.finfo(exp.dtype).eps)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None = None
    ) -> torch.Tensor:
        if x.ndim != 2 or edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("GATv2 期望 x[M,D], edge_index[2,E]")
        src, dst = edge_index.to(x.device)
        if src.numel() == 0:
            return self.bias.expand(x.shape[0], -1)
        source = self.lin_src(x).view(x.shape[0], self.heads, self.out_dim)
        target = self.lin_dst(x).view(x.shape[0], self.heads, self.out_dim)
        pair = source[src] + target[dst]
        if self.lin_edge is not None:
            if edge_attr is None or edge_attr.shape[0] != src.shape[0]:
                raise ValueError("配置 edge_dim 后必须提供与 E 对齐的 edge_attr")
            pair = pair + self.lin_edge(edge_attr.to(x)).view(-1, self.heads, self.out_dim)
        logits = (F.leaky_relu(pair, 0.2) * self.attention).sum(dim=-1)
        weights = self._segment_softmax(logits, dst, x.shape[0])
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        messages = source[src] * weights.unsqueeze(-1)
        output = torch.zeros(
            (x.shape[0], self.heads, self.out_dim), dtype=x.dtype, device=x.device
        )
        output.index_add_(0, dst, messages)
        return output.mean(dim=1) + self.bias


class RoadEncoder(nn.Module):
    """GTG TopoAggregator 风格的 Road 编码器，``[M,d_road] -> [M,D]``。"""

    def __init__(
        self,
        road_dim: int,
        hidden_dim: int,
        num_layers: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
        edge_dim: int | None = None,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(road_dim)
        self.projection = nn.Linear(road_dim, hidden_dim)
        self.layers = nn.ModuleList([
            GATv2MessageLayer(hidden_dim, hidden_dim, heads, edge_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = float(dropout)

    def forward(
        self, road_x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.projection(self.input_norm(road_x))
        for layer, norm in zip(self.layers, self.norms):
            update = layer(x, edge_index, edge_attr)
            x = norm(x + F.dropout(update, self.dropout, self.training))
            x = F.gelu(x)
        return x


class DisentangledEncoder(nn.Module):
    """把 Road 拓扑表示解耦为 city-invariant semantic 与 city-specific domain。"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        def block() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            )
        self.semantic_encoder = block()
        self.domain_encoder = block()

    def forward(self, road_rep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.semantic_encoder(road_rep), self.domain_encoder(road_rep)


class CostPredictor(nn.Module):
    """预测 duration/speed cost；只在源城市真实标签处监督。"""

    def __init__(self, hidden_dim: int, output_dim: int = 2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, output_dim), nn.Softplus(),
        )

    def forward(self, semantic: torch.Tensor) -> torch.Tensor:
        return self.network(semantic)


class DomainDiscriminator(nn.Module):
    def __init__(self, hidden_dim: int, num_domains: int):
        super().__init__()
        if num_domains < 2:
            raise ValueError("领域判别至少需要两个城市")
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, num_domains),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


def orthogonal_loss(semantic: torch.Tensor, domain: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """逐 Road cosine^2，零向量通过 eps 安全处理。"""

    semantic = semantic / semantic.norm(dim=-1, keepdim=True).clamp_min(eps)
    domain = domain / domain.norm(dim=-1, keepdim=True).clamp_min(eps)
    return ((semantic * domain).sum(dim=-1) ** 2).mean()


def rank_loss(prediction: torch.Tensor, target: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    """相邻有效 Road 对的可微排序损失；不构造 O(M^2) 全对。"""

    if prediction.shape != target.shape:
        raise ValueError("rank prediction/target shape 不一致")
    if prediction.numel() < 2:
        return prediction.sum() * 0.0
    pred_diff = prediction[1:] - prediction[:-1]
    true_diff = target[1:] - target[:-1]
    informative = true_diff != 0
    if not informative.any():
        return prediction.sum() * 0.0
    sign = true_diff[informative].sign()
    return F.relu(float(margin) - sign * pred_diff[informative]).mean()


class RoadAdversarialModule(nn.Module):
    """RoadEncoder + GTG 解耦/成本/领域对抗的完整可训练模块。"""

    def __init__(self, cfg: Mapping[str, object]):
        super().__init__()
        hidden = int(cfg["hidden_dim"])
        self.road_encoder = RoadEncoder(
            road_dim=int(cfg["road_dim"]), hidden_dim=hidden,
            num_layers=int(cfg.get("num_layers", 4)), heads=int(cfg.get("heads", 4)),
            dropout=float(cfg.get("dropout", 0.1)),
            edge_dim=int(cfg["edge_dim"]) if cfg.get("edge_dim") is not None else None,
        )
        self.disentangled_encoder = DisentangledEncoder(hidden)
        self.cost_predictor = CostPredictor(hidden, int(cfg.get("cost_dim", 2)))
        self.semantic_grl = GradientReversalLayer(float(cfg.get("grl_coefficient", 1.0)))
        domains = int(cfg["num_domains"])
        self.semantic_discriminator = DomainDiscriminator(hidden, domains)
        self.domain_discriminator = DomainDiscriminator(hidden, domains)

    def forward(
        self, road_x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None = None
    ) -> Dict[str, torch.Tensor]:
        road_rep = self.road_encoder(road_x, edge_index, edge_attr)
        semantic, domain = self.disentangled_encoder(road_rep)
        return {
            "road_rep": road_rep,
            "semantic": semantic,
            "domain": domain,
            "cost": self.cost_predictor(semantic),
            "semantic_domain_logits": self.semantic_discriminator(self.semantic_grl(semantic)),
            "domain_logits": self.domain_discriminator(domain),
        }


def adversarial_losses(
    outputs: Iterable[Mapping[str, torch.Tensor]],
    city_labels: Iterable[int],
    *,
    source_cost_target: torch.Tensor | None = None,
    source_cost_mask: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    """计算 GTG 损失；动态 cost 标签只允许传第一个（源城市）输出。"""

    outputs, labels = list(outputs), list(city_labels)
    if len(outputs) != len(labels) or not outputs:
        raise ValueError("outputs/city_labels 必须非空且等长")
    sem_domain, dom_domain, orth = [], [], []
    for output, label in zip(outputs, labels):
        count = output["semantic"].shape[0]
        target = torch.full((count,), int(label), dtype=torch.long, device=output["semantic"].device)
        sem_domain.append(F.cross_entropy(output["semantic_domain_logits"], target))
        dom_domain.append(F.cross_entropy(output["domain_logits"], target))
        orth.append(orthogonal_loss(output["semantic"], output["domain"]))
    zero = outputs[0]["semantic"].sum() * 0.0
    cost, ranking = zero, zero
    if source_cost_target is not None:
        prediction = outputs[0]["cost"]
        target = source_cost_target.to(prediction)
        if prediction.shape != target.shape:
            raise ValueError("源城市 cost 标签 shape 与预测不一致")
        mask = torch.ones(prediction.shape[0], dtype=torch.bool, device=prediction.device)
        if source_cost_mask is not None:
            mask = source_cost_mask.to(prediction.device).bool()
        if not mask.any():
            raise ValueError("源城市 cost mask 无有效标签")
        cost = F.mse_loss(prediction[mask], target[mask])
        ranking = sum(rank_loss(prediction[mask, k], target[mask, k]) for k in range(prediction.shape[1]))
        ranking = ranking / prediction.shape[1]
    return {
        "cost": cost,
        "rank": ranking,
        "semantic_domain": torch.stack(sem_domain).mean(),
        "domain": torch.stack(dom_domain).mean(),
        "orthogonal": torch.stack(orth).mean(),
    }


def assert_optimizer_covers(module: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    """确保所有 requires_grad 参数进入 optimizer，防止新增分支漏注册。"""

    expected = {id(parameter): name for name, parameter in module.named_parameters() if parameter.requires_grad}
    actual = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    missing = [name for identity, name in expected.items() if identity not in actual]
    if missing:
        raise RuntimeError(f"optimizer 缺少可训练参数: {missing}")

