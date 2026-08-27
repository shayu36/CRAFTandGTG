"""Hierarchical Cross-City Flow Matching (HCFM).

第二阶段实现与 ``craft_integrated`` 第一阶段代码并列，避免改变原 Diffusion
入口和 checkpoint 语义。
"""

from .data import (  # noqa: F401
    HierarchicalCityDataset,
    SourceOnlyNormalizer,
    collate_city_snapshots,
    validate_joint_sample,
)
from .hierarchy import aggregate_micro_to_macro  # noqa: F401

