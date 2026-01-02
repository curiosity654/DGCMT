"""
Argoverse2 Evaluation Module

This module contains all utilities for AV2 detection evaluation:
- av2_utils.py: Core utilities (matching, error computation, filtering)
- av2_eval_util.py: Evaluation pipeline and remote data loading
- summarize_metrics_av2.py: Metrics summarization (AP, Recall, CDS)

Usage:
    from projects.mmdet3d_plugin.datasets.av2_evaluation import DetectionCfg, evaluate, summarize_metrics
"""

from .av2_utils import (
    DetectionCfg,
    accumulate,
    assign,
    distance,
    compute_affinity_matrix,
    compute_evaluated_dts_mask,
    compute_evaluated_gts_mask,
    compute_objects_in_roi_mask,
    xyz_to_quat,
    yaw_to_quat,
)

from .av2_eval_util import (
    evaluate,
    load_mapped_avm_and_egoposes,
    read_feather_remote,
    ArgoverseStaticMapRemote,
    GroundHeightLayerRemote,
)

from .summarize_metrics_av2 import (
    summarize_metrics,
    compute_average_precision,
    interpolate_precision,
    TruePositiveErrorNames,
    MetricNames,
)

__all__ = [
    # Config
    'DetectionCfg',
    
    # Main evaluation functions
    'evaluate',
    'summarize_metrics',
    
    # Core utilities
    'accumulate',
    'assign',
    'distance',
    'compute_affinity_matrix',
    'compute_evaluated_dts_mask',
    'compute_evaluated_gts_mask',
    'compute_objects_in_roi_mask',
    
    # Coordinate conversion
    'xyz_to_quat',
    'yaw_to_quat',
    
    # Data loading
    'load_mapped_avm_and_egoposes',
    'read_feather_remote',
    'ArgoverseStaticMapRemote',
    'GroundHeightLayerRemote',
    
    # Metrics computation
    'compute_average_precision',
    'interpolate_precision',
    
    # Enums
    'TruePositiveErrorNames',
    'MetricNames',
]
