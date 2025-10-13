#!/usr/bin/env python
"""Visualization script for Mangrove3D predictions using trained RandLANet model."""

import logging
import sys
import os
from pathlib import Path
import numpy as np

# Add the Open3D-ML repository root to Python path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

import ml3d.datasets as datasets
import ml3d.torch.models as models
import ml3d.torch.pipelines as pipelines
import ml3d.vis as vis
import ml3d.utils as utils
from ml3d.torch.modules.metrics import SemSegMetric
import torch

log = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
CHECKPOINT_PATH = "/home/fzhcis/mylab/Open3D-ML/scripts/run_tests_fei/logs/RandLANet_Mangrove3D_torch/checkpoint/ckpt_00100.pth"
CONFIG_PATH = repo_root / "ml3d/configs/randlanet_mangrove3d.yml"
DATA_PATH = "/home/fzhcis/data/mangrove3d_pcd/"
# ============================================================================


def main():
    # Load dataset configuration and initialize dataset
    cfg = utils.Config.load_from_file(str(CONFIG_PATH))
    dataset = datasets.Mangrove3D(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
    
    # Get label mapping
    mangrove_labels = dataset.label_to_names
    
    # Custom colormap for Mangrove3D
    # COLOR_TO_INDEX: "128,0,128": 0, "165,42,42": 1, "0,128,0": 2, "255,165,0": 3, "255,255,0": 4
    color_map = {
        0: [128, 0, 128],    # Ground & Water - Purple
        1: [165, 42, 42],     # Stem - Brown
        2: [0, 128, 0],       # Canopy - Green
        3: [255, 165, 0],     # Roots - Orange
        4: [255, 255, 0],     # Objects - Yellow
    }
    
    # Setup visualizer with label lookup table
    v = vis.Visualizer()
    lut = vis.LabelLUT()
    for val in sorted(mangrove_labels.keys()):
        lut.add_label(mangrove_labels[val], val, color_map.get(val, [128, 128, 128]))
    v.set_lut("labels", lut)
    v.set_lut("pred", lut)
    
    # Load model
    model = models.RandLANet(**cfg.model)
    pipeline = pipelines.SemanticSegmentation(model, dataset=dataset)
    
    # Load checkpoint
    if not Path(CHECKPOINT_PATH).exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    pipeline.load_ckpt(CHECKPOINT_PATH)
    log.info(f"Loaded checkpoint: {CHECKPOINT_PATH}")
    
    # Get validation split data
    val_split = dataset.get_split("validation")
    log.info(f"Processing {len(val_split)} validation samples...")
    
    vis_points = []
    all_gt_labels = []
    all_pred_labels = []
    
    # Process each validation sample
    for idx in range(len(val_split)):
        data = val_split.get_data(idx)
        attr = val_split.get_attr(idx)
        
        # Run inference
        results = pipeline.run_inference(data)
        pred_labels = results['predict_labels'].astype(np.int32)
        gt_labels = data['label'].astype(np.int32)
        
        # Collect for metrics
        all_gt_labels.append(gt_labels)
        all_pred_labels.append(pred_labels)
        
        # Prepare visualization data
        vis_d = {
            "name": f"{attr['name']}_pred",
            "points": data['point'],
            "labels": gt_labels,
            "pred": pred_labels,
        }
        vis_points.append(vis_d)
        
        log.info(f"Processed {idx+1}/{len(val_split)}: {attr['name']}")
    
    # ========================================================================
    # Quantitative Analysis: Accuracy, IoU/mIoU, and Confusion Matrix
    # ========================================================================
    all_gt = np.concatenate(all_gt_labels)
    all_pred = np.concatenate(all_pred_labels)
    
    metric = SemSegMetric()
    num_classes = len(mangrove_labels)
    
    # Convert predictions to one-hot format for metric computation
    scores = torch.nn.functional.one_hot(
        torch.tensor(all_pred, dtype=torch.long), 
        num_classes=num_classes
    ).float()
    labels = torch.tensor(all_gt, dtype=torch.long)
    
    # Update metric
    metric.update(scores, labels)
    
    # Get metrics
    accuracies = metric.acc()
    ious = metric.iou()
    confusion_mat = metric.confusion_matrix
    
    # Display results
    print("\n" + "="*70)
    print("QUANTITATIVE ANALYSIS RESULTS - VALIDATION SET")
    print("="*70)
    print(f"Overall Accuracy: {accuracies[-1]*100:.2f}%")
    print(f"Mean IoU (mIoU):  {ious[-1]*100:.2f}%")
    print("\nPer-Class Metrics:")
    print(f"{'Class Name':<30} {'Accuracy':>12} {'IoU':>12}")
    print("-"*70)
    for i in sorted(mangrove_labels.keys()):
        label_name = mangrove_labels[i]
        acc_val = accuracies[i] * 100 if not np.isnan(accuracies[i]) else 0.0
        iou_val = ious[i] * 100 if not np.isnan(ious[i]) else 0.0
        print(f"{label_name:<30} {acc_val:>11.2f}% {iou_val:>11.2f}%")
    print("="*70)
    print(f"\nConfusion Matrix:\n{confusion_mat}")
    print("="*70 + "\n")
    
    # Visualize results
    log.info("Launching visualizer...")
    v.visualize(vis_points)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(asctime)s - %(module)s - %(message)s",
    )
    main()
