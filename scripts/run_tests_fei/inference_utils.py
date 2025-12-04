import logging
import sys
# import os
# import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
# Add the Open3D-ML repository root to Python path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

import ml3d.datasets as datasets
import ml3d.torch.models as models
import ml3d.torch.pipelines as pipelines
import ml3d.utils as utils
from ml3d.torch.modules.metrics import SemSegMetric
import torch
import numpy as np
import laspy

log = logging.getLogger(__name__)



def require_paths(cfg, cfg_path: str):
    # dataset_path
    dataset_path = cfg.dataset.get('dataset_path') if isinstance(cfg.dataset, dict) else getattr(cfg.dataset, 'dataset_path', None)
    if not dataset_path:
        raise ValueError(
            "Missing 'dataset.dataset_path' in config.\n"
            f"Config file: {cfg_path}\n"
            "Please set it under the 'dataset' section, e.g.:\n\n"
            "dataset:\n  dataset_path: /abs/path/to/Semantic3DUnified\n"
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


def build_model_dataset_pipeline(cfg, dataset_path: str):
    # Import dataset dynamically
    import importlib
    dataset_name = cfg.dataset['name']
    dataset_class = getattr(datasets, dataset_name, None)
    if dataset_class is None:
        try:
            importlib.import_module(f"ml3d.datasets.{dataset_name.lower()}")
        except Exception:
            pass
        try:
            dataset_class = utils.get_module('dataset', dataset_name)
        except Exception:
            dataset_class = getattr(datasets, dataset_name, None)
    
    if dataset_class is None:
        raise ValueError(
            f"Dataset '{dataset_name}' not found. Make sure its module is imported "
            f"(e.g., added to ml3d/datasets/__init__.py) or it registers with DATASET."
        )
    
    # Initialize dataset
    cfg.dataset['dataset_path'] = dataset_path
    dataset = dataset_class(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
    
    # Initialize model + pipeline
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


def compute_and_display_metrics(gt_labels, pred_labels_raw, label_to_names, num_classes):
    """Compute and display quantitative metrics (accuracy, IoU, confusion matrix).
    
    Args:
        gt_labels: List of ground truth label arrays (0-5)
        pred_labels_raw: List of prediction label arrays (0-4)
        label_to_names: Label ID to name mapping
        num_classes: Number of valid classes (excluding unlabeled)
    """
    all_gt = np.concatenate(gt_labels)
    all_pred = np.concatenate(pred_labels_raw)
    log.info(f"Total label-0 points (unlabeled): {(all_gt == 0).sum()} / {all_gt.shape[0]}")
    log.info(f"Predicted label 0 points: {(all_pred == 0).sum()} / {all_pred.shape[0]}")

    # Mask out ignored/unlabeled ground-truth (label 0)
    valid_mask = all_gt != 0
    valid_gt = all_gt[valid_mask]
    valid_pred = all_pred[valid_mask]

    if valid_gt.size == 0:
        raise ValueError("No valid ground truth labels found for metric computation after masking out label 0.")

    # Remap GT from {1..5} to {0..4} for metric computation
    # Predictions are already 0-4 from model output
    valid_gt_shifted = valid_gt - 1

    metric = SemSegMetric()
    # Convert predictions to one-hot format for metric computation
    scores = torch.nn.functional.one_hot(
        torch.tensor(valid_pred, dtype=torch.long),
        num_classes=num_classes
    ).float()
    labels = torch.tensor(valid_gt_shifted, dtype=torch.long)

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
    # Only log valid classes (exclude unlabeled 0)
    class_ids = [i for i in sorted(label_to_names.keys()) if i != 0]
    for idx, label_id in enumerate(class_ids):
        label_name = label_to_names[label_id]
        acc_val = accuracies[idx] * 100 if not np.isnan(accuracies[idx]) else 0.0
        iou_val = ious[idx] * 100 if not np.isnan(ious[idx]) else 0.0
        log.info(f"{label_name:<30} {acc_val:>11.2f}% {iou_val:>11.2f}%")
    log.info("="*70)
    log.info(f"\nConfusion Matrix:\n{confusion_mat}")
    log.info("="*70 + "\n")


def compare_histograms(gt_labels, pred_labels, save_path=None, label_to_names=None, save_csv=True):
    """Generate and save class distribution histogram comparing ground truth and predictions.
    
    Creates a dual-panel histogram with:
    - Left: Ground truth distribution with counts and ratios
    - Right: Prediction distribution with counts and ratios
    - Annotations showing exact counts on bars and ratio percentages on line plots
    
    Args:
        gt_labels: Ground truth labels array (0-5)
        pred_labels: Predicted labels array (1-5, shifted from model output 0-4)
        save_path: Path to save the histogram plot (PNG)
        label_to_names: Dict mapping label ID to class name (optional)
        save_csv: Whether to save distribution data to CSV file
    
    Returns:
        tuple: (gt_counts, gt_ratios, pred_counts, pred_ratios) as numpy arrays
    """
    

    FONTSIZE = 14
    
    # Get unique class IDs and compute counts/ratios
    gt_class_ids = np.unique(gt_labels)
    pred_class_ids = np.unique(pred_labels)
    all_class_ids = np.unique(np.concatenate([gt_class_ids, pred_class_ids]))
    
    gt_counts = np.array([(gt_labels == cid).sum() for cid in all_class_ids])
    pred_counts = np.array([(pred_labels == cid).sum() for cid in all_class_ids])
    
    gt_ratios = gt_counts / gt_counts.sum() if gt_counts.sum() > 0 else np.zeros_like(gt_counts)
    pred_ratios = pred_counts / pred_counts.sum() if pred_counts.sum() > 0 else np.zeros_like(pred_counts)
    
    # Create evenly spaced positions for discrete class IDs
    x_positions = np.arange(len(all_class_ids))
    
    # Create figure with two subplots stacked vertically
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(16, len(all_class_ids) * 2), 10))
    
    # --- Left panel: Ground Truth ---
    bars_gt = ax1.bar(x_positions, gt_counts, width=0.6, edgecolor='black', 
                      align='center', color='skyblue', label='Ground Truth')
    ax1.set_ylabel('Number of Points', color='blue', fontsize=FONTSIZE)
    ax1.tick_params(axis='y', labelcolor='blue', labelsize=FONTSIZE)
    ax1.set_title('Ground Truth Distribution', fontsize=FONTSIZE + 2, fontweight='bold')
    
    # Set x-axis labels
    if label_to_names:
        x_labels = [f"{label_to_names.get(cid, f'Class {cid}')}\n(ID: {cid})" for cid in all_class_ids]
    else:
        x_labels = [f"Class {cid}\n(ID: {cid})" for cid in all_class_ids]
    
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=FONTSIZE)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_xlim(-0.5, len(all_class_ids) - 0.5)
    
    # Add count annotations on bars
    for bar in bars_gt:
        height = bar.get_height()
        if height > 0:
            ax1.annotate(f'{int(height):,}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=FONTSIZE - 2)
    
    # Add ratio line (right y-axis)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(x_positions, gt_ratios, 'o-', color='darkred', 
                  label='Ratio', markersize=8, linewidth=2)
    ax1_twin.set_ylabel('Ratio (%)', color='darkred', fontsize=FONTSIZE)
    ax1_twin.tick_params(axis='y', labelcolor='darkred', labelsize=FONTSIZE)
    max_ratio_gt = max(gt_ratios) if len(gt_ratios) > 0 and max(gt_ratios) > 0 else 1.0
    ax1_twin.set_ylim(0, max_ratio_gt * 1.2)
    ax1_twin.set_yticks(np.linspace(0, max_ratio_gt * 1.2, 5))
    ax1_twin.set_yticklabels([f"{r*100:.1f}%" for r in np.linspace(0, max_ratio_gt * 1.2, 5)], 
                             fontsize=FONTSIZE - 2)
    
    # Add ratio annotations
    for x_pos, y in zip(x_positions, gt_ratios):
        if y > 0:
            ax1_twin.annotate(f'{y*100:.1f}%',
                            xy=(x_pos, y),
                            xytext=(5, 0),
                            textcoords='offset points',
                            ha='left', va='center', fontsize=FONTSIZE - 2, color='darkred')
    
    # --- Right panel: Predictions ---
    bars_pred = ax2.bar(x_positions, pred_counts, width=0.6, edgecolor='black',
                        align='center', color='orange', label='Predictions')
    ax2.set_ylabel('Number of Points', color='blue', fontsize=FONTSIZE)
    ax2.tick_params(axis='y', labelcolor='blue', labelsize=FONTSIZE)
    ax2.set_title('Prediction Distribution', fontsize=FONTSIZE + 2, fontweight='bold')
    
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=FONTSIZE)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_xlim(-0.5, len(all_class_ids) - 0.5)
    
    # Add count annotations on bars
    for bar in bars_pred:
        height = bar.get_height()
        if height > 0:
            ax2.annotate(f'{int(height):,}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=FONTSIZE - 2)
    
    # Add ratio line (right y-axis)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(x_positions, pred_ratios, 'o-', color='darkred',
                  label='Ratio', markersize=8, linewidth=2)
    ax2_twin.set_ylabel('Ratio (%)', color='darkred', fontsize=FONTSIZE)
    ax2_twin.tick_params(axis='y', labelcolor='darkred', labelsize=FONTSIZE)
    max_ratio_pred = max(pred_ratios) if len(pred_ratios) > 0 and max(pred_ratios) > 0 else 1.0
    ax2_twin.set_ylim(0, max_ratio_pred * 1.2)
    ax2_twin.set_yticks(np.linspace(0, max_ratio_pred * 1.2, 5))
    ax2_twin.set_yticklabels([f"{r*100:.1f}%" for r in np.linspace(0, max_ratio_pred * 1.2, 5)],
                             fontsize=FONTSIZE - 2)
    
    # Add ratio annotations
    for x_pos, y in zip(x_positions, pred_ratios):
        if y > 0:
            ax2_twin.annotate(f'{y*100:.1f}%',
                            xy=(x_pos, y),
                            xytext=(5, 0),
                            textcoords='offset points',
                            ha='left', va='center', fontsize=FONTSIZE - 2, color='darkred')
    
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        log.info(f"Saved class distribution histogram: {save_path}")
        
        # Save CSV if requested
        if save_csv:
            csv_path = Path(save_path).with_suffix('.csv')
            df_data = {
                'Class_ID': all_class_ids,
                'GT_Count': gt_counts,
                'GT_Ratio_%': gt_ratios * 100,
                'Pred_Count': pred_counts,
                'Pred_Ratio_%': pred_ratios * 100
            }
            
            if label_to_names:
                df_data['Class_Name'] = [label_to_names.get(cid, f'Class_{cid}') for cid in all_class_ids]
                # Reorder columns to put Class_Name after Class_ID
                df = pd.DataFrame(df_data)[['Class_ID', 'Class_Name', 'GT_Count', 'GT_Ratio_%', 
                                            'Pred_Count', 'Pred_Ratio_%']]
            else:
                df = pd.DataFrame(df_data)
            
            df.to_csv(csv_path, index=False, float_format='%.2f')
            log.info(f"Saved class distribution CSV: {csv_path}")
    
    plt.close()
    
    return gt_counts, gt_ratios, pred_counts, pred_ratios


def save_las_file(points, gt_labels, pred_labels, intensity, rgb, output_path):
    """Save point cloud with GT and predicted labels to a .las file.
    
    Args:
        points: Nx3 array of xyz coordinates
        gt_labels: N array of ground truth labels (0-5)
        pred_labels: N array of predicted labels (model output 0-4, will be shifted to 1-5)
        intensity: N array of intensity values
        rgb: Nx3 array of RGB values
        output_path: Path to save the .las file
    """
    try:
        # Create a new LAS file with point format 3 (includes RGB and classification)
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.offsets = np.min(points, axis=0)
        header.scales = np.array([0.001, 0.001, 0.001])
        
        las = laspy.LasData(header)
        
        # Set coordinates
        las.x = points[:, 0]
        las.y = points[:, 1]
        las.z = points[:, 2]
        
        # Set intensity (scale to 0-65535 range if needed)
        if intensity is not None:
            intensity_scaled = np.clip(intensity, 0, 65535).astype(np.uint16)
            las.intensity = intensity_scaled
        
        # Set RGB (laspy expects 0-65535 range)
        if rgb is not None:
            rgb_scaled = np.clip(rgb * 257, 0, 65535).astype(np.uint16)  # 0-255 -> 0-65535
            las.red = rgb_scaled[:, 0]
            las.green = rgb_scaled[:, 1]
            las.blue = rgb_scaled[:, 2]
        
        
        # Store predictions in user_data field
        # Shift predictions from 0-4 to 1-5 to match label IDs
        pred_labels_shifted = (pred_labels + 1).astype(np.uint8)
        las.classification = pred_labels_shifted
        
        las.add_extra_dim(laspy.ExtraBytesParams(name="gtlabel", type=np.uint8))
        las.gtlabel = gt_labels.astype(np.uint8)
        
        # Write to file
        las.write(output_path)
        log.info(f"Saved LAS file including predictions and ground truth labels: {output_path}")
        log.info(f"  - Points: {len(las.x)}")
    except Exception as e:
        log.error(f"Failed to save LAS file {output_path}: {e}")


def check_pred_labels(pred_labels):
    unique_labels = np.unique(pred_labels)
    
    if np.any((unique_labels < 0) | (unique_labels > 4)):
        raise ValueError(
            f"Predicted labels contain invalid values: {unique_labels}\n"
            "Expected range is 0-4 corresponding to classes 1-5."
        )
    
    log.info(f"✅ Predicted labels check passed. Unique labels: {unique_labels}")
