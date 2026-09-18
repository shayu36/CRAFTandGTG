"""第二阶段 Road→Syntax→Region GraphGPS + LapPE。"""

from .data import (
    RELATION_NAMES,
    RELATION_TO_ID,
    SPECTRAL_FEATURE_VERSION,
    checkpoint_fingerprint,
    export_spectral_features,
    GraphGPSCityData,
    JointThreeLayerGraph,
    RegionFlowTargets,
    load_source_region_flow_splits,
    load_spectral_features,
    load_stage2_hierarchy,
    prepare_city_data,
    build_joint_three_layer_graph,
    joint_graph_hash,
    validate_joint_three_layer_graph,
)
from .frequency import (
    FrequencyComponents,
    LowFrequencyTransferInputs,
    SpectralFeatureDecoupler,
    build_low_frequency_transfer_inputs,
)
from .model import ThreeLayerGraphGPSLapPE, validate_stage2_config
from .pooling import pool_road_to_syntax, pool_syntax_to_region
from .posenc import FeatureLapPEInit, LapPEEncoder
from .spectral_lap_pe import (
    HierarchyLaplacianPE,
    LAPPE_VERSION,
    LaplacianEigenpairs,
    compute_sparse_laplacian_eigenpairs,
    pe_graph_hash,
    prepare_hierarchy_lappe,
    to_undirected_edge_index,
)

__all__ = [
    "FeatureLapPEInit",
    "JointThreeLayerGraph",
    "FrequencyComponents",
    "GraphGPSCityData",
    "HierarchyLaplacianPE",
    "LAPPE_VERSION",
    "LapPEEncoder",
    "LaplacianEigenpairs",
    "LowFrequencyTransferInputs",
    "RegionFlowTargets",
    "SPECTRAL_FEATURE_VERSION",
    "RELATION_NAMES",
    "RELATION_TO_ID",
    "SpectralFeatureDecoupler",
    "ThreeLayerGraphGPSLapPE",
    "compute_sparse_laplacian_eigenpairs",
    "checkpoint_fingerprint",
    "export_spectral_features",
    "build_low_frequency_transfer_inputs",
    "load_source_region_flow_splits",
    "load_spectral_features",
    "load_stage2_hierarchy",
    "build_joint_three_layer_graph",
    "joint_graph_hash",
    "validate_joint_three_layer_graph",
    "pool_road_to_syntax",
    "pool_syntax_to_region",
    "pe_graph_hash",
    "prepare_city_data",
    "prepare_hierarchy_lappe",
    "to_undirected_edge_index",
    "validate_stage2_config",
]
