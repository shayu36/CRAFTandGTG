"""Hierarchical Road/Syntax/Region RAG for the Stage-2 representation route."""

from .contracts import (
    LAYER_NAMES,
    PARENT_NAMES,
    RAG_CONTRACT_VERSION,
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

__all__ = [
    "CalendarEncoder",
    "HierarchicalThreeLayerRAG",
    "HierarchicalRAG",
    "LAYER_NAMES",
    "PARENT_NAMES",
    "RAG_CONTRACT_VERSION",
    "TemporalSequenceEncoder",
    "ThreeLayerRAG",
    "ThreeLayerRAGInputs",
    "ThreeLayerRAGMemory",
    "assert_no_high_frequency_input",
    "STAGE2_SPECTRAL_VERSION",
    "build_rag_inputs_from_local_artifacts",
    "load_stage2_low_features",
    "parent_operators_from_hierarchy",
    "RAG_CHECKPOINT_VERSION",
    "RAGTrainer",
    "load_rag_checkpoint",
    "save_rag_checkpoint",
]
