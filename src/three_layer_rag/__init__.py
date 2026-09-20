"""Hierarchical Road/Syntax/Region RAG for the Stage-2 representation route."""

from .contracts import (
    LAYER_NAMES,
    PARENT_NAMES,
    RAG_CONTRACT_VERSION,
    SNAPSHOT_BUNDLE_VERSION,
    ThreeLayerRAGInputs,
    ThreeLayerRAGMemory,
    assert_no_high_frequency_input,
)
from .model import (
    CalendarEncoder,
    HierarchicalThreeLayerRAG,
    HierarchicalRAG,
    TemporalSequenceEncoder,
    ThreeLayerRAG,
)
from .io import (
    STAGE2_SPECTRAL_VERSION,
    STAGE2_LAPPE_VERSION,
    build_rag_inputs_from_local_artifacts,
    load_stage2_low_features,
    parent_operators_from_hierarchy,
)
from .training import (
    RAG_CHECKPOINT_VERSION,
    RAGTrainer,
    load_rag_checkpoint,
    save_rag_checkpoint,
)
from .dynamics import (
    DYNAMIC_FEATURE_VERSION,
    HourlyThreeLayerDynamics,
    aggregate_road_hourly_features,
    build_hourly_three_layer_dynamics,
    calendar_for_timestamp,
    normalize_hourly_dynamics,
    split_snapshot_starts,
    temporal_window,
    weighted_parent_pool,
)

__all__ = [
    "CalendarEncoder",
    "HierarchicalThreeLayerRAG",
    "HierarchicalRAG",
    "LAYER_NAMES",
    "PARENT_NAMES",
    "RAG_CONTRACT_VERSION",
    "SNAPSHOT_BUNDLE_VERSION",
    "TemporalSequenceEncoder",
    "ThreeLayerRAG",
    "ThreeLayerRAGInputs",
    "ThreeLayerRAGMemory",
    "assert_no_high_frequency_input",
    "STAGE2_SPECTRAL_VERSION",
    "STAGE2_LAPPE_VERSION",
    "build_rag_inputs_from_local_artifacts",
    "load_stage2_low_features",
    "parent_operators_from_hierarchy",
    "RAG_CHECKPOINT_VERSION",
    "RAGTrainer",
    "load_rag_checkpoint",
    "save_rag_checkpoint",
    "DYNAMIC_FEATURE_VERSION",
    "HourlyThreeLayerDynamics",
    "aggregate_road_hourly_features",
    "build_hourly_three_layer_dynamics",
    "calendar_for_timestamp",
    "normalize_hourly_dynamics",
    "split_snapshot_starts",
    "temporal_window",
    "weighted_parent_pool",
]
