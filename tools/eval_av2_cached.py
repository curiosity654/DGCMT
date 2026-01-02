#!/usr/bin/env python
"""
Standalone AV2 evaluation script using cached feather files.

Uses official AV2 evaluation protocol via av2 library.

Usage:
    python tools/eval_av2_cached.py --dts <dts.feather> --gts <gts.feather>
    
Example:
    python tools/eval_av2_cached.py \
        --dts data/argo2/val_dts.feather \
        --gts data/argo2/val_anno.feather
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description='Evaluate AV2 cached results using official AV2 protocol')
    parser.add_argument('--dts', type=str, required=True, help='Path to detections feather file')
    parser.add_argument('--gts', type=str, default='data/argo2/val_anno.feather', 
                        help='Path to ground truth feather file')
    parser.add_argument('--eval-range', type=float, nargs=2, default=[0.0, 150.0], 
                        help='Evaluation range in meters (min max)')
    parser.add_argument('--dataset-dir', type=str, default=None,
                        help='Path to AV2 dataset directory for ROI filtering (optional)')
    parser.add_argument('--no-roi', action='store_true', 
                        help='Disable ROI filtering (use when dataset_dir not available)')
    args = parser.parse_args()
    
    print(f"Loading detections from {args.dts}...")
    dts = pd.read_feather(args.dts)
    print(f"  Loaded {len(dts)} detections")
    
    print(f"Loading ground truth from {args.gts}...")
    gts = pd.read_feather(args.gts)
    print(f"  Loaded {len(gts)} ground truth boxes")
    
    # Import AV2 evaluation modules
    try:
        from av2.evaluation import SensorCompetitionCategories
        from projects.mmdet3d_plugin.datasets.av2_evaluation import DetectionCfg, evaluate as av2_evaluate
    except ImportError as e:
        print(f"Error: Could not import AV2 evaluation modules: {e}")
        print("Make sure 'av2' is installed: pip install av2")
        sys.exit(1)
    
    # Determine evaluation categories
    available_categories = set(gts['category'].unique().tolist())
    pred_categories = set(dts['category'].unique().tolist())
    competition_categories = set(x.value for x in SensorCompetitionCategories)
    
    # Use intersection of available and competition categories
    eval_categories = (available_categories | pred_categories) & competition_categories
    print(f"\nEvaluating on {len(eval_categories)} categories: {sorted(eval_categories)}")
    
    # Setup evaluation config
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else None
    eval_only_roi = not args.no_roi and dataset_dir is not None and dataset_dir.exists()
    
    if eval_only_roi:
        print(f"ROI filtering enabled, using dataset: {dataset_dir}")
    else:
        print("ROI filtering disabled")
    
    cfg = DetectionCfg(
        dataset_dir=dataset_dir,
        categories=tuple(sorted(eval_categories)),
        eval_range_m=tuple(args.eval_range),
        eval_only_roi_instances=eval_only_roi,
    )
    
    print(f"Evaluation range: {cfg.eval_range_m[0]}m - {cfg.eval_range_m[1]}m")
    print(f"Affinity thresholds: {cfg.affinity_thresholds_m}")
    
    # Run evaluation
    print("\nRunning AV2 evaluation...")
    try:
        eval_dts, eval_gts, metrics, recall3d = av2_evaluate(dts, gts, cfg)
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Print results
    valid_categories = sorted(eval_categories) + ["AVERAGE_METRICS"]
    print("\n" + "=" * 100)
    print("Argoverse2 Evaluation Results:")
    print("=" * 100)
    print(metrics.loc[valid_categories].to_string())
    print("=" * 100)
    
    # Print summary
    if 'AVERAGE_METRICS' in metrics.index:
        print("\nSummary:")
        for col in metrics.columns:
            val = metrics.loc['AVERAGE_METRICS', col]
            print(f"  {col}: {val:.4f}")
    
    return metrics


if __name__ == '__main__':
    main()
