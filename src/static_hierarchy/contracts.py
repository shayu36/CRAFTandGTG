"""三层城市静态图的数据契约与严格校验。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(init=False)
class CityStaticHierarchy:
    """稳定排序的 Road、Syntax、Region 三层城市静态图。

    所有张量均不含 batch 维。跨层算子使用 ``row=上层节点``、
    ``column=下层节点`` 的 COO 边表示。
    """

    city_id: str
    region_x: torch.Tensor
    region_edge_index: torch.Tensor
    # ``road_x`` is the canonical storage.  ``road_topo_x`` remains a
    # read-only compatibility property for the v1 topology-only cache and
    # older callers.
    road_x: torch.Tensor
    road_edge_index: torch.Tensor
    road_ids: tuple[str, ...]
    syntax_x: torch.Tensor
    syntax_edge_index: torch.Tensor
    road_to_syntax_assignment: torch.Tensor
    road_to_syntax_edge_index: torch.Tensor
    road_to_syntax_weight: torch.Tensor
    road_to_syntax_shape: tuple[int, int]
    syntax_to_region_edge_index: torch.Tensor
    syntax_to_region_weight: torch.Tensor
    syntax_to_region_shape: tuple[int, int]
    region_has_syntax: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)
    # 仅在 source 静态预训练适配器中填充；target 静态图保持 None。
    value: torch.Tensor | None = None
    value_region_ids: torch.Tensor | None = None
    value_mask: torch.Tensor | None = None

    def __init__(
        self,
        city_id: str,
        region_x: torch.Tensor,
        region_edge_index: torch.Tensor,
        road_topo_x: torch.Tensor | None = None,
        road_edge_index: torch.Tensor | None = None,
        road_ids: tuple[str, ...] = (),
        syntax_x: torch.Tensor | None = None,
        syntax_edge_index: torch.Tensor | None = None,
        road_to_syntax_assignment: torch.Tensor | None = None,
        road_to_syntax_edge_index: torch.Tensor | None = None,
        road_to_syntax_weight: torch.Tensor | None = None,
        road_to_syntax_shape: tuple[int, int] = (0, 0),
        syntax_to_region_edge_index: torch.Tensor | None = None,
        syntax_to_region_weight: torch.Tensor | None = None,
        syntax_to_region_shape: tuple[int, int] = (0, 0),
        region_has_syntax: torch.Tensor | None = None,
        metadata: dict[str, Any] | None = None,
        value: torch.Tensor | None = None,
        value_region_ids: torch.Tensor | None = None,
        value_mask: torch.Tensor | None = None,
        *,
        road_x: torch.Tensor | None = None,
    ) -> None:
        if road_x is None:
            road_x = road_topo_x
        elif road_topo_x is not None and road_x is not road_topo_x:
            if not torch.equal(road_x, road_topo_x):
                raise ValueError("严格模式: road_x 与兼容参数 road_topo_x 不一致")
        if road_x is None:
            raise TypeError("严格模式: 必须提供 road_x（旧调用方可使用 road_topo_x）")
        self.city_id = city_id
        self.region_x = region_x
        self.region_edge_index = region_edge_index
        self.road_x = road_x
        self.road_edge_index = road_edge_index
        self.road_ids = tuple(str(value) for value in road_ids)
        self.syntax_x = syntax_x
        self.syntax_edge_index = syntax_edge_index
        self.road_to_syntax_assignment = road_to_syntax_assignment
        self.road_to_syntax_edge_index = road_to_syntax_edge_index
        self.road_to_syntax_weight = road_to_syntax_weight
        self.road_to_syntax_shape = tuple(int(value) for value in road_to_syntax_shape)
        self.syntax_to_region_edge_index = syntax_to_region_edge_index
        self.syntax_to_region_weight = syntax_to_region_weight
        self.syntax_to_region_shape = tuple(int(value) for value in syntax_to_region_shape)
        self.region_has_syntax = region_has_syntax
        self.metadata = {} if metadata is None else metadata
        self.value = value
        self.value_region_ids = value_region_ids
        self.value_mask = value_mask

    @property
    def road_topo_x(self) -> torch.Tensor:
        """旧 v1 调用方兼容别名；对象内部只保存一份 ``road_x``。"""

        return self.road_x

    @property
    def city(self) -> str:
        return self.city_id

    @property
    def num_regions(self) -> int:
        return int(self.region_x.shape[0])

    @property
    def num_roads(self) -> int:
        return int(self.road_x.shape[0])

    @property
    def num_syntax(self) -> int:
        return int(self.syntax_x.shape[0])

    def to(self, device: torch.device | str) -> "CityStaticHierarchy":
        values = {
            name: value.to(device) if isinstance(value, torch.Tensor) else value
            for name, value in self.__dict__.items()
        }
        return CityStaticHierarchy(**values)


def _finite(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"严格模式: {name} 必须为 torch.Tensor")
    if value.is_floating_point() and not torch.isfinite(value).all():
        raise ValueError(f"严格模式: {name} 含 NaN/Inf")


def _edge_index(name: str, value: torch.Tensor, nodes: int) -> None:
    if not isinstance(value, torch.Tensor) or value.dtype != torch.long or value.ndim != 2 or value.shape[0] != 2:
        raise ValueError(f"严格模式: {name} 必须为 LongTensor[2,E]")
    if value.numel() and (int(value.min()) < 0 or int(value.max()) >= nodes):
        raise ValueError(f"严格模式: {name} 节点索引越界")


def _operator_edge(name: str, value: torch.Tensor, shape: tuple[int, int]) -> None:
    if not isinstance(value, torch.Tensor) or value.dtype != torch.long or value.ndim != 2 or value.shape[0] != 2:
        raise ValueError(f"严格模式: {name} 必须为 LongTensor[2,E]")
    if value.numel() and (
        int(value[0].min()) < 0 or int(value[0].max()) >= shape[0]
        or int(value[1].min()) < 0 or int(value[1].max()) >= shape[1]
    ):
        raise ValueError(f"严格模式: {name} 跨层索引越界")


def validate_city_static_hierarchy(hierarchy: CityStaticHierarchy) -> None:
    """验证形状、排序、跨层算子和有限性，不做静默修复。"""

    if not isinstance(hierarchy, CityStaticHierarchy):
        raise TypeError("严格模式: hierarchy 必须为 CityStaticHierarchy")
    if not hierarchy.city_id:
        raise ValueError("严格模式: city_id 不能为空")
    if not isinstance(hierarchy.metadata, dict):
        raise TypeError("严格模式: metadata 必须为 dict")
    region_x, road_x, syntax_x = hierarchy.region_x, hierarchy.road_x, hierarchy.syntax_x
    for name, value in (("region_x", region_x), ("road_x", road_x), ("syntax_x", syntax_x)):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"严格模式: {name} 必须为 torch.Tensor")
    if region_x.ndim != 2 or region_x.shape[1] != 45:
        raise ValueError(f"严格模式: region_x 应为 [N,45]，实得 {tuple(region_x.shape)}")
    feature_version = hierarchy.metadata.get("feature_version")
    road_feature_mode = hierarchy.metadata.get("road_feature_mode", "topology_only")
    if feature_version == "three-layer-static-v1":
        expected_road_dim = 4
        if road_feature_mode != "topology_only":
            raise ValueError("严格模式: v1 cache 的 road_feature_mode 必须为 topology_only")
    elif feature_version == "three-layer-start-road-v2":
        expected_road_dim = 33
        if road_feature_mode != "start_static":
            raise ValueError("严格模式: v2 cache 的 road_feature_mode 必须为 start_static")
    else:
        raise ValueError("严格模式: 缺少或错误的三层静态图 feature_version")
    if road_x.ndim != 2 or road_x.shape[1] != expected_road_dim:
        raise ValueError(
            f"严格模式: road_x 应为 [M,{expected_road_dim}]，实得 {tuple(road_x.shape)}"
        )
    if syntax_x.ndim != 2 or syntax_x.shape[1] != 5:
        raise ValueError(f"严格模式: syntax_x 应为 [K,5]，实得 {tuple(syntax_x.shape)}")
    if region_x.shape[0] <= 0:
        raise ValueError("严格模式: Region 图不能为空")
    if road_x.shape[0] <= 0:
        raise ValueError("严格模式: Road 图不能为空")
    if syntax_x.shape[0] <= 0:
        raise ValueError("严格模式: Syntax 图不能为空")
    for name, value in (("region_x", region_x), ("road_x", road_x), ("syntax_x", syntax_x)):
        if not value.is_floating_point():
            raise ValueError(f"严格模式: {name} 必须为浮点 Tensor")
    if len(hierarchy.road_ids) != hierarchy.num_roads:
        raise ValueError("严格模式: road_ids 与 Road 节点数量不一致")
    if len(set(hierarchy.road_ids)) != len(hierarchy.road_ids):
        raise ValueError("严格模式: road_ids 存在重复 ID")
    for name, value in (
        ("region_x", region_x), ("road_x", road_x), ("syntax_x", syntax_x),
        ("road_to_syntax_weight", hierarchy.road_to_syntax_weight),
        ("syntax_to_region_weight", hierarchy.syntax_to_region_weight),
    ):
        _finite(name, value)
    _edge_index("region_edge_index", hierarchy.region_edge_index, hierarchy.num_regions)
    _edge_index("road_edge_index", hierarchy.road_edge_index, hierarchy.num_roads)
    _edge_index("syntax_edge_index", hierarchy.syntax_edge_index, hierarchy.num_syntax)
    if hierarchy.syntax_edge_index.numel() and torch.any(
        hierarchy.syntax_edge_index[0] == hierarchy.syntax_edge_index[1]
    ):
        raise ValueError("严格模式: Syntax 图不允许同分区自环")
    if hierarchy.syntax_edge_index.shape[1] > 1:
        syntax_pairs = hierarchy.syntax_edge_index.t()
        if torch.unique(syntax_pairs, dim=0).shape[0] != syntax_pairs.shape[0]:
            raise ValueError("严格模式: Syntax 图存在未 coalesce 的重复边")
    _operator_edge("road_to_syntax_edge_index", hierarchy.road_to_syntax_edge_index, hierarchy.road_to_syntax_shape)
    if hierarchy.road_to_syntax_edge_index.shape[1] != hierarchy.num_roads:
        raise ValueError("严格模式: Road→Syntax 必须恰好有 M 条映射边")
    if hierarchy.road_to_syntax_weight.shape != (hierarchy.num_roads,):
        raise ValueError("严格模式: road_to_syntax_weight shape 错误")
    if hierarchy.road_to_syntax_shape != (hierarchy.num_syntax, hierarchy.num_roads):
        raise ValueError("严格模式: road_to_syntax_shape 错误")
    if (hierarchy.road_to_syntax_weight <= 0).any():
        raise ValueError("严格模式: Road→Syntax 均值权重必须为正")
    assignment = hierarchy.road_to_syntax_assignment
    if assignment.dtype != torch.long or assignment.shape != (hierarchy.num_roads,):
        raise ValueError("严格模式: road_to_syntax_assignment 应为 LongTensor[M]")
    if assignment.numel() and (int(assignment.min()) < 0 or int(assignment.max()) >= hierarchy.num_syntax):
        raise ValueError("严格模式: road_to_syntax_assignment 越界")
    expected = torch.arange(hierarchy.num_roads, device=assignment.device)
    if not torch.equal(hierarchy.road_to_syntax_edge_index[1], expected):
        raise ValueError("严格模式: Road→Syntax column 必须按稳定 Road 顺序 0..M-1")
    if not torch.equal(hierarchy.road_to_syntax_edge_index[0], assignment):
        raise ValueError("严格模式: Road→Syntax assignment 与 COO 行不一致")
    syntax_sums = torch.zeros(hierarchy.num_syntax, dtype=hierarchy.road_to_syntax_weight.dtype, device=assignment.device)
    syntax_sums.index_add_(0, assignment, hierarchy.road_to_syntax_weight)
    if not torch.allclose(syntax_sums, torch.ones_like(syntax_sums), atol=1e-5, rtol=1e-5):
        raise ValueError("严格模式: 每个 Syntax 节点的 Road→Syntax 均值权重和必须为 1")
    _operator_edge("syntax_to_region_edge_index", hierarchy.syntax_to_region_edge_index, hierarchy.syntax_to_region_shape)
    if hierarchy.syntax_to_region_shape != (hierarchy.num_regions, hierarchy.num_syntax):
        raise ValueError("严格模式: syntax_to_region_shape 错误")
    if hierarchy.syntax_to_region_weight.ndim != 1 or hierarchy.syntax_to_region_weight.shape[0] != hierarchy.syntax_to_region_edge_index.shape[1]:
        raise ValueError("严格模式: syntax_to_region_weight shape 错误")
    if (hierarchy.syntax_to_region_weight < 0).any():
        raise ValueError("严格模式: Syntax→Region 权重不能为负")
    mask = hierarchy.region_has_syntax
    if mask.dtype != torch.bool or mask.shape != (hierarchy.num_regions,):
        raise ValueError("严格模式: region_has_syntax 应为 BoolTensor[N]")
    rows = hierarchy.syntax_to_region_edge_index[0]
    sums = torch.zeros(hierarchy.num_regions, dtype=hierarchy.syntax_to_region_weight.dtype, device=rows.device)
    if rows.numel():
        sums.index_add_(0, rows, hierarchy.syntax_to_region_weight)
    nonempty = mask.to(sums.device)
    if nonempty.any() and not torch.allclose(sums[nonempty], torch.ones_like(sums[nonempty]), atol=1e-5, rtol=1e-5):
        raise ValueError("严格模式: 非空 Region 的 Syntax→Region 权重和必须为 1")
    if (~nonempty).any() and (sums[~nonempty] != 0).any():
        raise ValueError("严格模式: region_has_syntax=False 的 Region 不应有映射边")
    # Cache metadata is part of the contract as well.  Do not silently accept
    # arrays that disagree with the recorded city sizes or feature ordering.
    expected_counts = {
        "num_regions": hierarchy.num_regions,
        "num_roads": hierarchy.num_roads,
        "num_road_edges": int(hierarchy.road_edge_index.shape[1]),
        "num_syntax_nodes": hierarchy.num_syntax,
        "num_syntax_edges": int(hierarchy.syntax_edge_index.shape[1]),
        "num_road_to_syntax_links": hierarchy.num_roads,
        "num_syntax_to_region_links": int(hierarchy.syntax_to_region_edge_index.shape[1]),
    }
    for key, expected_value in expected_counts.items():
        if key in hierarchy.metadata and int(hierarchy.metadata[key]) != expected_value:
            raise ValueError(
                f"严格模式: metadata.{key}={hierarchy.metadata[key]!r} "
                f"与实际值 {expected_value} 不一致"
            )
    if "city" in hierarchy.metadata and hierarchy.metadata["city"] != hierarchy.city_id:
        raise ValueError("严格模式: metadata.city 与 city_id 不一致")
    if "road_ids" in hierarchy.metadata:
        meta_road_ids = tuple(str(value) for value in hierarchy.metadata["road_ids"])
        if meta_road_ids != hierarchy.road_ids:
            raise ValueError("严格模式: metadata.road_ids 与 Road 节点顺序不一致")
    for key, expected_shape in (
        ("road_to_syntax_shape", hierarchy.road_to_syntax_shape),
        ("syntax_to_region_shape", hierarchy.syntax_to_region_shape),
    ):
        if key in hierarchy.metadata and tuple(int(value) for value in hierarchy.metadata[key]) != expected_shape:
            raise ValueError(f"严格模式: metadata.{key} 与实际稀疏算子 shape 不一致")
    if feature_version == "three-layer-static-v1":
        if "road_topo_feature_names" in hierarchy.metadata and list(hierarchy.metadata["road_topo_feature_names"]) != [
            "bias", "in_degree", "out_degree", "total_degree"
        ]:
            raise ValueError("严格模式: Road 拓扑特征顺序错误")
    else:
        expected_start_names = [
            "road_type_residential", "road_type_trunk", "road_type_primary", "road_type_secondary",
            "road_type_tertiary", "road_type_motorway", "road_type_living_street", "road_type_unclassified",
            "length_log_minmax",
            "lanes_unknown", "lanes_1", "lanes_2", "lanes_3", "lanes_4", "lanes_5_plus",
            "maxspeed_unknown", "maxspeed_le_30", "maxspeed_31_50", "maxspeed_51_70",
            "maxspeed_71_90", "maxspeed_gt_90",
            "indegree_0", "indegree_1", "indegree_2", "indegree_3", "indegree_4", "indegree_5_plus",
            "outdegree_0", "outdegree_1", "outdegree_2", "outdegree_3", "outdegree_4", "outdegree_5_plus",
        ]
        if list(hierarchy.metadata.get("road_feature_names", [])) != expected_start_names:
            raise ValueError("严格模式: START Road 特征顺序错误")
        if int(hierarchy.metadata.get("road_feature_dim", -1)) != 33:
            raise ValueError("严格模式: START Road feature_dim 必须为 33")
        if hierarchy.metadata.get("maxspeed_unit") != "km/h":
            raise ValueError("严格模式: START Road maxspeed_unit 必须明确为 km/h")
    if "syntax_feature_names" in hierarchy.metadata and list(hierarchy.metadata["syntax_feature_names"]) != [
        "connectivity", "total_depth", "integration", "choice", "mean_depth"
    ]:
        raise ValueError("严格模式: Syntax 特征顺序错误")
    if "region_feature_order" in hierarchy.metadata:
        expected_region_features = (
            ["population", "population_density", "dist_to_center", "road_num", "road_length"]
            + [f"poi_num_{k}" for k in range(12)]
            + [f"poi_score_{k}" for k in range(12)]
            + [f"road_num_{k}" for k in range(8)]
            + [f"road_length_{k}" for k in range(8)]
        )
        if list(hierarchy.metadata["region_feature_order"]) != expected_region_features:
            raise ValueError("严格模式: CRAFT Region 45 维特征顺序错误")
    expected_empty = torch.where(~mask)[0].cpu().tolist()
    if "empty_region_ids" in hierarchy.metadata and list(hierarchy.metadata["empty_region_ids"]) != expected_empty:
        raise ValueError("严格模式: metadata.empty_region_ids 与 region_has_syntax 不一致")
    if "empty_region_ratio" in hierarchy.metadata:
        actual_ratio = float((~mask).sum().item()) / hierarchy.num_regions
        if abs(float(hierarchy.metadata["empty_region_ratio"]) - actual_ratio) > 1e-6:
            raise ValueError("严格模式: metadata.empty_region_ratio 与 region_has_syntax 不一致")
    if "syntax_edge_weight" in hierarchy.metadata:
        try:
            syntax_edge_weight = torch.as_tensor(hierarchy.metadata["syntax_edge_weight"], dtype=torch.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("严格模式: metadata.syntax_edge_weight 非法") from exc
        if syntax_edge_weight.shape != (hierarchy.syntax_edge_index.shape[1],):
            raise ValueError("严格模式: metadata.syntax_edge_weight shape 错误")
        if not torch.isfinite(syntax_edge_weight).all() or (syntax_edge_weight < 0).any():
            raise ValueError("严格模式: metadata.syntax_edge_weight 必须为有限非负值")
    if hierarchy.value is not None:
        _finite("value", hierarchy.value)
        if hierarchy.value.ndim != 2 or hierarchy.value.shape[1] != 48:
            raise ValueError("严格模式: source value 应为 [num_active_regions,48]")
        if hierarchy.value_region_ids is None or hierarchy.value_mask is None:
            raise ValueError("严格模式: source value 必须同时提供 value_region_ids/value_mask")
        ids = hierarchy.value_region_ids
        if ids.dtype != torch.long or ids.ndim != 1 or ids.shape[0] != hierarchy.value.shape[0]:
            raise ValueError("严格模式: value_region_ids shape/dtype 错误")
        if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= hierarchy.num_regions):
            raise ValueError("严格模式: value_region_ids 越界")
        if len(torch.unique(ids)) != len(ids):
            raise ValueError("严格模式: value_region_ids 存在重复 ID")
        if hierarchy.value_mask.dtype != torch.bool or hierarchy.value_mask.shape != (hierarchy.num_regions,):
            raise ValueError("严格模式: value_mask shape/dtype 错误")
        expected_mask = torch.zeros_like(hierarchy.value_mask)
        expected_mask[ids.to(expected_mask.device)] = True
        if not torch.equal(expected_mask, hierarchy.value_mask):
            raise ValueError("严格模式: value_mask 与 value_region_ids 不一致")
