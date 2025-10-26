# import os
# import open3d.ml as _ml3d
# import open3d.ml.torch as ml3d

# cfg_file = "ml3d/configs/randlanet_semantic3d.yml"
# cfg = _ml3d.utils.Config.load_from_file(cfg_file)

# model = ml3d.models.RandLANet(**cfg.model)

# # The dataset path should contain .txt files and .labels files, where the .labels files are the ground
# # truth labels for the corresponding .txt files. The .txt files that does not have a corresponding .labels, they will
# # be considered as test files.

# # cfg.dataset['dataset_path'] = "/shared/rc/mangrove/data/Semantic3D/"
# cfg.dataset['dataset_path'] = "/home/fzhcis/mylab/data/semantic3d/open3d_randlanet/test_pcd"

# dataset = ml3d.datasets.Semantic3D(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
# pipeline = ml3d.pipelines.SemanticSegmentation(model, dataset=dataset, device="gpu", **cfg.pipeline)

# # download the weights.
# ckpt_folder = "./logs/"
# os.makedirs(ckpt_folder, exist_ok=True)
# ckpt_path = ckpt_folder + "randlanet_semantic3d_202201071330utc.pth"
# randlanet_url = "https://storage.googleapis.com/open3d-releases/model-zoo/randlanet_semantic3d_202201071330utc.pth"
# if not os.path.exists(ckpt_path):
#     cmd = "wget {} -O {}".format(randlanet_url, ckpt_path)
#     os.system(cmd)

# # load the parameters.
# pipeline.load_ckpt(ckpt_path=ckpt_path)

# test_split = dataset.get_split("test")
# data = test_split.get_data(0)

# # run inference on a single example.
# # returns dict with 'predict_labels' and 'predict_scores'.
# result = pipeline.run_inference(data)
# print({k: (v.shape if hasattr(v, "shape") else type(v)) for k, v in result.items()})

# # # evaluate performance on the test set; this will write logs to './logs'.
# # pipeline.run_test()


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


def main():
    parser = argparse.ArgumentParser(description='Run inference with RandLANet on Semantic3D')
    parser.add_argument('--config', type=str, 
                        default='ml3d/configs/randlanet_semantic3d.yml',
                        help='Path to config YAML file')
    # Enforce dataset_path and checkpoint provided via config file only
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
    log.info(f"Dataset path (from config): {dataset_path}")
    log.info(f"Checkpoint (from config): {ckpt_path}")

    # Build components and load checkpoint
    model, dataset, pipeline = build_components(cfg, dataset_path)
    pipeline.load_ckpt(ckpt_path=ckpt_path)
    
    # Get label mapping
    semantic3d_labels = dataset.label_to_names
    
    # Setup visualizer if needed
    v = setup_visualizer(semantic3d_labels) if args.visualize else None
    
    # Run inference
    test_split = dataset.get_split("test")
    total = len(test_split)
    log.info(f"Test split size: {total} samples")

    if total == 0:
        log.warning("No test samples found. Exiting.")
        return

    # Determine which indices to run
    indices = select_indices(args, total)

    log.info(f"Running inference on {len(indices)} sample(s): {indices[:5]}{' ...' if len(indices) > 5 else ''}")

    vis_points = []
    all_gt_labels = []
    all_pred_labels = []
    
    for k, idx in enumerate(indices, start=1):
        log.info(f"[{k}/{len(indices)}] Inference on test index {idx}")
        
        data = test_split.get_data(idx)
        attr = test_split.get_attr(idx)
        result = pipeline.run_inference(data)

        pred_labels = result['predict_labels'].astype(np.int32)
        gt_labels = data['label'].astype(np.int32)
        
        # Collect for metrics
        if args.metrics:
            all_gt_labels.append(gt_labels)
            all_pred_labels.append(pred_labels)
        
        # Prepare visualization data
        if v is not None:
            vis_points.append({
                "name": f"{attr['name']}_pred",
                "points": data['point'],
                "labels": gt_labels,
                "pred": pred_labels,
            })
        
        # Display concise results for each sample
        shapes = {k: (v.shape if hasattr(v, 'shape') else type(v)) for k, v in result.items()}
        log.info(f"Results: {shapes}")
    
    # ========================================================================
    # Quantitative Analysis: Accuracy, IoU/mIoU, and Confusion Matrix
    # ========================================================================
    if args.metrics and len(all_gt_labels) > 0:
        all_gt = np.concatenate(all_gt_labels)
        all_pred = np.concatenate(all_pred_labels)
        
        metric = SemSegMetric()
        num_classes = len(semantic3d_labels)
        
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
        print("QUANTITATIVE ANALYSIS RESULTS - TEST SET")
        print("="*70)
        print(f"Overall Accuracy: {accuracies[-1]*100:.2f}%")
        print(f"Mean IoU (mIoU):  {ious[-1]*100:.2f}%")
        print("\nPer-Class Metrics:")
        print(f"{'Class Name':<30} {'Accuracy':>12} {'IoU':>12}")
        print("-"*70)
        for i in sorted(semantic3d_labels.keys()):
            label_name = semantic3d_labels[i]
            acc_val = accuracies[i] * 100 if not np.isnan(accuracies[i]) else 0.0
            iou_val = ious[i] * 100 if not np.isnan(ious[i]) else 0.0
            print(f"{label_name:<30} {acc_val:>11.2f}% {iou_val:>11.2f}%")
        print("="*70)
        print(f"\nConfusion Matrix:\n{confusion_mat}")
        print("="*70 + "\n")
    
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