import torch

from hcfm.hierarchy import (
    aggregate_micro_to_macro, build_boundary_operators, build_road_edge_index,
    build_boundary_operators_from_sequences,
)


def test_manual_directional_dynamic_aggregation():
    b_in, b_out = build_boundary_operators(
        [-1, 0, 1, 2, 1], [0, 1, 2, -1, 1], 3
    )
    # outside->0, 0->1, 1->2, 2->outside, internal 1->1
    q = torch.tensor([1.0, 2.0, 3.0, 4.0, 99.0]).view(1, 5, 1, 1)
    result = aggregate_micro_to_macro(q, b_in, b_out)[0, :, :, 0]
    assert torch.equal(result, torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]))


def test_bidirectional_roads_are_separate():
    b_in, b_out = build_boundary_operators([0, 1], [1, 0], 2)
    q = torch.tensor([5.0, 7.0]).view(1, 2, 1, 1)
    result = aggregate_micro_to_macro(q, b_in, b_out)[0, :, :, 0]
    assert torch.equal(result, torch.tensor([[7.0, 5.0], [5.0, 7.0]]))


def test_road_graph_respects_direction():
    edge = build_road_edge_index([10, 20, 20], [20, 30, 10])
    pairs = set(map(tuple, edge.T.tolist()))
    assert (0, 1) in pairs and (0, 2) in pairs
    assert (1, 0) not in pairs


def test_one_road_crossing_three_regions_counts_every_boundary():
    b_in, b_out = build_boundary_operators_from_sequences([[-1, 0, 1, 2, -1]], 3)
    q = torch.tensor([5.0]).view(1, 1, 1, 1)
    result = aggregate_micro_to_macro(q, b_in, b_out)[0, :, :, 0]
    assert torch.equal(result, torch.tensor([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]]))
