#!/usr/bin/env python
"""Inference script for RandLANet on Semantic3D dataset."""

import logging
import sys
import os
import argparse
from pathlib import Path

# Add the Open3D-ML repository root to Python path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

import ml3d.datasets as datasets
import ml3d.torch.models as models
import ml3d.torch.pipelines as pipelines
import ml3d.utils as utils
import ml3d.vis as vis
from ml3d.torch.modules.metrics import SemSegMetric
import torch
import numpy as np

log = logging.getLogger(__name__)

###########################
# Helpers (kept minimal)  #
###########################

def load_cfg(config_path: str):
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return utils.Config.load_from_file(config_path)


def require_paths(cfg, cfg_path: str):
    # dataset_path
    dataset_path = cfg.dataset.get('dataset_path') if isinstance(cfg.dataset, dict) else getattr(cfg.dataset, 'dataset_path', None)
    if not dataset_path:
        raise ValueError(
            "Missing 'dataset.dataset_path' in config.\n"
            f"Config file: {cfg_path}\n"
            "Please set it under the 'dataset' section, e.g.:\n\n"
            "dataset:\n  dataset_path: /abs/path/to/Semantic3D\n"
        )
    if not Path(dataset_path).exists():
        raise FileNotFoundError(
            f"Configured dataset_path does not exist: {dataset_path}\n"
            f"Config file: {cfg_path}\n"
            "Please update 'dataset.dataset_path' to a valid directory."
        )

    # checkpoint
    ckpt_path = cfg.model.get('ckpt_path') if isinstance(cfg.model, dict) else getattr(cfg.model, 'ckpt_path', None)
    if not ckpt_path:
        raise ValueError(
            "Missing 'model.ckpt_path' in config.\n"
            f"Config file: {cfg_path}\n"
            "Please set it under the 'model' section, e.g.:\n\n"
            "model:\n  ckpt_path: /abs/path/to/checkpoint.pth\n"
        )
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(
            f"Configured checkpoint file not found: {ckpt_path}\n"
            f"Config file: {cfg_path}\n"
            "Please update 'model.ckpt_path' to point to an existing .pth file."
        )
    return dataset_path, ckpt_path


def build_components(cfg, dataset_path: str):
    # dataset
    cfg.dataset['dataset_path'] = dataset_path
    dataset = datasets.Semantic3D(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
    # model + pipeline
    model = models.RandLANet(**cfg.model)
    pipeline = pipelines.SemanticSegmentation(model, dataset=dataset, device="gpu", **cfg.pipeline)
    return model, dataset, pipeline


def select_indices(args, total: int):
    if total == 0:
        return []
    if args.all:
        return list(range(total))
    if args.indices is not None:
        invalid = [i for i in args.indices if i < 0 or i >= total]
        if invalid:
            raise IndexError(f"Indices out of range (size={total}): {invalid}")
        return list(dict.fromkeys(args.indices))
    raise ValueError("Please provide either --all or --indices <i j ...>")


def setup_visualizer(labels):
    v = vis.Visualizer()
    lut = vis.LabelLUT()
    for val in sorted(labels.keys()):
        lut.add_label(labels[val], val)
    v.set_lut("labels", lut)
    v.set_lut("pred", lut)
    return v


def compute_and_display_metrics(all_gt_labels, all_pred_labels, semantic3d_label2names, num_classes):
    """Compute and display quantitative metrics (accuracy, IoU, confusion matrix).
    
    Args:
        all_gt_labels: List of ground truth label arrays (Semantic3D IDs: 0-8)
        all_pred_labels: List of prediction label arrays (model output: 0-7)
        semantic3d_label2names: Label ID to name mapping
        num_classes: Number of valid classes (excluding unlabeled)
    """
    all_gt = np.concatenate(all_gt_labels)
    all_pred = np.concatenate(all_pred_labels)

    # Mask out ignored/unlabeled ground-truth (label 0)
    valid_mask = all_gt != 0
    valid_gt = all_gt[valid_mask]
    valid_pred = all_pred[valid_mask]

    if valid_gt.size == 0:
        log.info("\n" + "="*70)
        log.info("QUANTITATIVE ANALYSIS RESULTS - TEST SET")
        log.info("="*70)
        log.info("No valid labeled points after excluding unlabeled (0). Skipping metrics.\n")
        return

    # Remap GT from Semantic3D IDs {1..8} to 0-based indices {0..7}
    # Predictions are already 0-7 from model output
    gt_idx = valid_gt - 1  # {1..8} -> {0..7}

    metric = SemSegMetric()
    # Convert predictions to one-hot format for metric computation
    scores = torch.nn.functional.one_hot(
        torch.tensor(valid_pred, dtype=torch.long),
        num_classes=num_classes
    ).float()
    labels = torch.tensor(gt_idx, dtype=torch.long)

    # Update metric
    metric.update(scores, labels)

    # Get metrics
    accuracies = metric.acc()
    ious = metric.iou()
    confusion_mat = metric.confusion_matrix

    # Display results
    log.info("\n" + "="*70)
    log.info("QUANTITATIVE ANALYSIS RESULTS - TEST SET")
    log.info("="*70)
    log.info(f"Overall Accuracy: {accuracies[-1]*100:.2f}%")
    log.info(f"Mean IoU (mIoU):  {ious[-1]*100:.2f}%")
    log.info("\nPer-Class Metrics:")
    log.info(f"{'Class Name':<30} {'Accuracy':>12} {'IoU':>12}")
    log.info("-"*70)
    # Only log.info valid classes (exclude unlabeled 0)
    class_ids = [i for i in sorted(semantic3d_label2names.keys()) if i != 0]
    for idx, label_id in enumerate(class_ids):
        label_name = semantic3d_label2names[label_id]
        acc_val = accuracies[idx] * 100 if not np.isnan(accuracies[idx]) else 0.0
        iou_val = ious[idx] * 100 if not np.isnan(ious[idx]) else 0.0
        log.info(f"{label_name:<30} {acc_val:>11.2f}% {iou_val:>11.2f}%")
    log.info("="*70)
    log.info(f"\nConfusion Matrix:\n{confusion_mat}")
    log.info("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Run inference with RandLANet on Semantic3D')
    parser.add_argument('--config', type=str, 
                        default='ml3d/configs/randlanet_semantic3d.yml',
                        help='Path to config YAML file')
    parser.add_argument('--all', action='store_true',
                        help='Run inference on all test samples')
    parser.add_argument('--indices', type=int, nargs='+',
                        help='List of test sample indices to run (space-separated)')
    parser.add_argument('--visualize', action='store_true',
                        help='Launch visualizer after inference')
    parser.add_argument('--metrics', action='store_true',
                        help='Compute and display quantitative metrics (accuracy, IoU, confusion matrix)')
    args = parser.parse_args()
    
    # Load configuration and required paths
    cfg = load_cfg(args.config)
    dataset_path, ckpt_path = require_paths(cfg, args.config)

    # Build components and load checkpoint
    model, dataset, pipeline = build_components(cfg, dataset_path)
    pipeline.load_ckpt(ckpt_path=ckpt_path)
    
    # Get label mapping
    semantic3d_label2names = dataset.label_to_names

    # Fetch num_classes from config (required)
    if isinstance(cfg.model, dict):
        num_classes_cfg = cfg.model.get('num_classes', None)
    else:
        num_classes_cfg = getattr(cfg.model, 'num_classes', None)
    if not num_classes_cfg:
        raise ValueError(
            "Missing 'model.num_classes' in config.\n"
            f"Config file: {args.config}\n"
            "Please set it under the 'model' section, e.g.:\n\n"
            "model:\n  num_classes: 8\n"
        )
    
    # Setup visualizer if needed
    v = setup_visualizer(semantic3d_label2names) if args.visualize else None
    
    # Run inference
    test_split = dataset.get_split("test")
    total = len(test_split)

    if total == 0:
        log.warning("No test samples found. Exiting.")
        return

    # Determine which indices to run
    indices = select_indices(args, total)

    log.info(f"Running inference on {len(indices)} sample(s) from test set (total: {total})")

    vis_points = []
    all_gt_labels = []
    all_pred_labels = []
    
    for k, idx in enumerate(indices, start=1):
        data = test_split.get_data(idx)
        attr = test_split.get_attr(idx)
        result = pipeline.run_inference(data)

        # Model outputs 0-7; keep as-is for metrics, add 1 only for visualization (Semantic3D IDs are 1-8)
        pred_labels_raw = result['predict_labels'].astype(np.int32)
        gt_labels = data['label'].astype(np.int32)
        
        # Collect for metrics (use raw 0-7 predictions)
        if args.metrics:
            all_gt_labels.append(gt_labels)
            all_pred_labels.append(pred_labels_raw)
        
        # Prepare visualization data (convert predictions to 1-8 for display)
        if v is not None:
            vis_points.append({
                "name": f"{attr['name']}_pred",
                "points": data['point'],
                "labels": gt_labels,
                "pred": pred_labels_raw + 1,  # Display as Semantic3D IDs (1-8)
            })
    
    # Compute and display metrics
    if args.metrics and len(all_gt_labels) > 0:
        compute_and_display_metrics(all_gt_labels, all_pred_labels, semantic3d_label2names, num_classes_cfg)
    
    # Visualize results
    if v is not None and len(vis_points) > 0:
        log.info("Launching visualizer...")
        v.visualize(vis_points)
    
    # Uncomment to run full test evaluation
    # log.info("Running full test evaluation...")
    # pipeline.run_test()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(asctime)s - %(module)s - %(message)s",
    )
    main()