"""HCFM 算法单测的手工小图，不作为真实数据验收或实验结果。"""

from __future__ import annotations

import torch

from hcfm.hierarchy import build_boundary_operators, sparse_coo


def tiny_config(prior_mode: str = "independent"):
    return {
        "model_mode": "hierarchical_flow_matching",
        "generator_type": "flow_matching",
        "use_hierarchy": True,
        "generate_micro": True,
        "model": {
            "seq_length": 4, "rep_dim": 16, "road_dim": 6, "road_hidden_dim": 16,
            "refer_dim": 16, "refer_heads": 2, "refer_layers": 1,
            "hour_dim": 4, "weekday_dim": 4, "month_dim": 4,
        },
        "hierarchy": {"num_layers": 1, "fusion": "gated_residual", "bidirectional": True},
        "micro_adversarial": {
            "num_layers": 2, "heads": 2, "dropout": 0.0, "edge_dim": None,
            "cost_dim": 2, "num_domains": 2, "grl_coefficient": 1.0,
        },
        "flow_matching": {
            "prior_mode": prior_mode, "solver": "euler", "steps": 4,
            "hidden_dim": 8, "num_blocks": 2, "time_dim": 8, "dropout": 0.0,
        },
        "loss": {
            "fm_macro": 1.0, "fm_micro": 1.0, "cross_state": 0.1,
            "cross_velocity": 0.0 if prior_mode == "independent" else 0.05,
            "topology": 0.1, "cost": 0.1, "rank": 0.1,
            "semantic_domain": 0.01, "domain": 0.01, "orthogonal": 0.01, "gfa": 0.1,
        },
        "training": {"phase": "cross_state" if prior_mode == "independent" else "coupled_velocity", "lr": 1e-3},
        "data": {"data_version": "tiny-v1"},
    }


def tiny_sample(city_id: str = "source", split: str = "train"):
    n, m, length = 3, 4, 4
    p = sparse_coo(
        [0, 1, 1, 2], [0, 1, 2, 3], [1.0, 0.5, 0.5, 1.0], (n, m)
    )
    # roads: outside->R0, R0->R1, R1->R2, R2->outside
    b_in, b_out = build_boundary_operators([-1, 0, 1, 2], [0, 1, 2, -1], n)
    micro = (torch.arange(m * length, dtype=torch.float32).reshape(m, 1, length) % 5) + 1
    from hcfm.hierarchy import aggregate_micro_to_macro
    macro = aggregate_micro_to_macro(micro.unsqueeze(0), b_in, b_out)[0]
    region_x = torch.arange(n * 45, dtype=torch.float32).reshape(n, 45) / 100.0
    road_x = torch.arange(m * 6, dtype=torch.float32).reshape(m, 6) / 10.0
    sample = {
        "city_id": city_id, "date": "2023-01-01", "start_hour": 0,
        "region_x": region_x,
        "region_edge_index": torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
        "macro_flow": macro,
        "road_x": road_x,
        "road_edge_index": torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        "micro_flow": micro,
        "p_struct": p, "b_in": b_in, "b_out": b_out,
        "region_mask": torch.ones(n, dtype=torch.bool),
        "road_mask": torch.ones(m, dtype=torch.bool),
        "reference": macro.clone(),
        "time_features": {"month": 0, "weekday": 6, "start_hour": 0},
        "month": 0, "weekday": 6, "split": split,
        "dynamic_metadata": {
            branch: {"city_id": city_id, "date": "2023-01-01", "start_hour": 0, "split": split}
            for branch in ("macro", "micro")
        },
    }
    return sample


def tiny_target_static():
    sample = tiny_sample("target", "test")
    return {
        "city_id": "target",
        "region_x": sample["region_x"].unsqueeze(0) + 0.01,
        "region_edge_index": sample["region_edge_index"],
        "road_x": sample["road_x"].unsqueeze(0) + 0.01,
        "road_edge_index": sample["road_edge_index"],
        "p_struct": sample["p_struct"],
    }

