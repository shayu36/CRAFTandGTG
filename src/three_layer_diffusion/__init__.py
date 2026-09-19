"""Current Stage-4 hierarchical conditional Diffusion route."""

from .conditioning import (
    LayerDiffusionConditioner,
    broadcast_parent_to_children,
    flatten_node_conditions,
    flatten_nodes,
    unflatten_nodes,
)
from .data import (
    CHANNEL_NAMES,
    DYNAMIC_FEATURE_VERSION,
    SNAPSHOT_BUNDLE_VERSION,
    CityGraphBucketBatchSampler,
    DynamicNormalizer,
    SnapshotDataset,
    collate_city_graph_bucket,
    file_sha256,
    load_snapshot_bundle,
    load_stage2_high_features,
    validate_runtime_identities,
)
from .factory import build_diffusion_model, build_rag_model, build_system, validate_stage4_config
from .gaussian_diffusion import (
    ConditionalGaussianDiffusion1D,
    DiffusionPrediction,
    GaussianDiffusion1D,
    cosine_beta_schedule,
    extract,
    linear_beta_schedule,
)
from .metrics import physical_layer_metrics, three_layer_physical_metrics
from .model import (
    DIFFUSION_CONTRACT_VERSION,
    LAYER_CHANNELS,
    LAYER_ORDER,
    HierarchicalThreeLayerDiffusion,
    ThreeLayerRAGDiffusionSystem,
)
from .training import (
    DIFFUSION_CHECKPOINT_VERSION,
    ModelEMA,
    ThreeLayerDiffusionTrainer,
    load_diffusion_checkpoint,
    save_diffusion_checkpoint,
)
from .unet import ConditionalUnet1D, Unet1D

__all__ = [
    "CHANNEL_NAMES",
    "ConditionalGaussianDiffusion1D",
    "ConditionalUnet1D",
    "DIFFUSION_CHECKPOINT_VERSION",
    "DIFFUSION_CONTRACT_VERSION",
    "DYNAMIC_FEATURE_VERSION",
    "DiffusionPrediction",
    "DynamicNormalizer",
    "GaussianDiffusion1D",
    "HierarchicalThreeLayerDiffusion",
    "LAYER_CHANNELS",
    "LAYER_ORDER",
    "LayerDiffusionConditioner",
    "ModelEMA",
    "SNAPSHOT_BUNDLE_VERSION",
    "SnapshotDataset",
    "CityGraphBucketBatchSampler",
    "ThreeLayerDiffusionTrainer",
    "ThreeLayerRAGDiffusionSystem",
    "Unet1D",
    "broadcast_parent_to_children",
    "build_diffusion_model",
    "build_rag_model",
    "build_system",
    "validate_stage4_config",
    "collate_city_graph_bucket",
    "cosine_beta_schedule",
    "extract",
    "file_sha256",
    "flatten_node_conditions",
    "flatten_nodes",
    "linear_beta_schedule",
    "load_diffusion_checkpoint",
    "load_snapshot_bundle",
    "load_stage2_high_features",
    "physical_layer_metrics",
    "save_diffusion_checkpoint",
    "three_layer_physical_metrics",
    "unflatten_nodes",
    "validate_runtime_identities",
]
