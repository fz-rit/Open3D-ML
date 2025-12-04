#!/usr/bin/env python
"""Inference script for RandLANet on Semantic3DUnified dataset.

Usage Examples:
    # Run inference on all test samples and save results as LAS files:
    python inference_randlanet_semantic3dunified.py --config /home/fzhcis/mylab/Open3D-ML/ml3d/configs/randlanet_semantic3dunified_xyz.yml --all --save-las ./results/semantic3dunified --metrics
    
    # Run inference on specific test samples:
    python inference_randlanet_semantic3dunified.py --config /home/fzhcis/mylab/Open3D-ML/ml3d/configs/randlanet_semantic3dunified_xyz.yml --indices 0 1 2 --metrics
    
    # Run inference and compute metrics only (no LAS output):
    python inference_randlanet_semantic3dunified.py --config /home/fzhcis/mylab/Open3D-ML/ml3d/configs/randlanet_semantic3dunified_xyz.yml --all --metrics
    
    # Run on all test samples with LAS output:
    python inference_randlanet_semantic3dunified.py \
        --config /home/fzhcis/mylab/Open3D-ML/ml3d/configs/randlanet_semantic3dunified_xyz.yml\
        --all  --save-las /home/fzhcis/data/open3d_outputs/randlanet/semantic3dunified \
        --metrics > inference_120125.log 2>&1

Arguments:
    --config: Path to YAML configuration file
    --all: Run inference on all test samples
    --indices: List of test sample indices (space-separated integers)
    --metrics: Compute and display quantitative metrics (accuracy, IoU, confusion matrix)
    --save-las: Directory to save results as .las files (requires laspy)
"""

import logging
import sys
# import os
import argparse
from pathlib import Path
# import matplotlib.pyplot as plt
# import pandas as pd
# Add the Open3D-ML repository root to Python path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

# import ml3d.datasets as datasets
# import ml3d.torch.models as models
# import ml3d.torch.pipelines as pipelines
import ml3d.utils as utils
# from ml3d.torch.modules.metrics import SemSegMetric
# import torch
from scripts.run_tests_fei.inference_utils import (require_paths,
                                                      build_model_dataset_pipeline,
                                                      select_indices,
                                                      check_pred_labels,
                                                      save_las_file,
                                                      compare_histograms,
                                                      compute_and_display_metrics)
import numpy as np
# import laspy

log = logging.getLogger(__name__)

###########################
# Helpers                 #
###########################

def load_cfg(config_path: str):
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return utils.Config.load_from_file(config_path)



def main():
    parser = argparse.ArgumentParser(
        description='Run inference with RandLANet on Semantic3DUnified',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--config', type=str, 
                        default='ml3d/configs/randlanet_semantic3dunified_xyz.yml',
                        help='Path to config YAML file')
    parser.add_argument('--all', action='store_true',
                        help='Run inference on all test samples')
    parser.add_argument('--indices', type=int, nargs='+',
                        help='List of test sample indices to run (space-separated)')
    parser.add_argument('--metrics', action='store_true',
                        help='Compute and display quantitative metrics (accuracy, IoU, confusion matrix)')
    parser.add_argument('--save-las', type=str, default=None,
                        help='Directory to save results as .las files (requires laspy)')
    args = parser.parse_args()
    
    # Load configuration and required paths
    cfg = load_cfg(args.config)
    dataset_path, ckpt_path = require_paths(cfg, args.config)

    # Build components and load checkpoint
    model, dataset, pipeline = build_model_dataset_pipeline(cfg, dataset_path)
    pipeline.load_ckpt(ckpt_path=ckpt_path)
    
    # Get label mapping
    label_to_names = dataset.label_to_names
    num_classes_cfg = cfg.model.get('num_classes', None)

    # Setup output directory for LAS files if requested
    las_output_dir = None
    if args.save_las:
        las_output_dir = Path(args.save_las)
        las_output_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"LAS files will be saved to: {las_output_dir}")
    
    # Run inference
    test_split = dataset.get_split("test")
    assert test_split is not None, "Test split not found in dataset."
    total = len(test_split)

    # Determine which indices to run
    indices = select_indices(args, total)

    log.info(f"Running inference on {len(indices)} sample(s) from test set (total: {total})")

    all_gt_labels = []
    all_pred_labels = []
    
    for k, idx in enumerate(indices, start=1):
        data = test_split.get_data(idx)
        attr = test_split.get_attr(idx)
        result = pipeline.run_inference(data)

        # Model outputs 0-4 (for classes 1-5)
        pred_labels_raw = result['predict_labels'].astype(np.int32)
        check_pred_labels(pred_labels_raw)
        gt_labels = data['label'].astype(np.int32)
        
        # Save as LAS file if requested
        if las_output_dir is not None:
            las_filename = f"{attr['name']}_predictions.las"
            las_path = las_output_dir / las_filename
            save_las_file(
                points=data['point'],
                gt_labels=gt_labels,
                pred_labels=pred_labels_raw,
                intensity=data.get('intensity'),
                rgb=data.get('feat'),
                output_path=str(las_path)
            )
            
            # Save class distribution histogram
            compare_histograms(
                gt_labels, 
                pred_labels_raw + 1,  # Shift to 1-5 for display
                save_path=las_output_dir / f"{attr['name']}_class_distribution.png",
                label_to_names=label_to_names,
                save_csv=True
            )
        
        # Collect for metrics (use raw 0-4 predictions)
        if args.metrics:
            all_gt_labels.append(gt_labels)
            all_pred_labels.append(pred_labels_raw)
    
        log.info(f"[{k}/{len(indices)}] Sample: {attr['name']} | Points: {data['point'].shape[0]}")
        log.info(f"Predicted labels unique: {np.unique(pred_labels_raw+1)}")
    
    # Compute and display metrics
    if args.metrics:
        compute_and_display_metrics(all_gt_labels, all_pred_labels, label_to_names, num_classes_cfg)
    
    log.info("Inference complete!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(asctime)s - %(module)s - %(message)s",
    )
    main()
