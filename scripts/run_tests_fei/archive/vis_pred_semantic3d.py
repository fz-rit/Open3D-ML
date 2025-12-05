#!/usr/bin/env python
import logging
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
# Add the Open3D-ML repository root to Python path
repo_root = Path(__file__).parent.parent.parent  # Go up 3 levels to reach Open3D-ML root
sys.path.insert(0, str(repo_root))

# Now import ml3d from the local repository
import ml3d
import ml3d.datasets as datasets
import ml3d.torch.models as models
import ml3d.torch.pipelines as pipelines
import ml3d.vis as vis
import ml3d.utils as utils
from ml3d.torch.modules.metrics import SemSegMetric
import torch
from scripts.run_tests_fei.archive.compare_gt_pred_labels import compare_gt_pred_histogram
from os.path import exists, join, dirname

example_dir = os.path.dirname(os.path.realpath(__file__))

log = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION - Modify these paths as needed
# ============================================================================
# parent_dir = Path(r"D:\mylab\data\Semantic3D-dataset")
parent_dir = Path("/home/fzhcis/data/semantic3d_full/selected/")
CHECKPOINT_PATH = parent_dir / "checkpoints/ckpt_00100.pth"
DATA_PATH = parent_dir
PC_NAMES = ["untermaederbrunnen_station3_xyz_intensity_rgb"]  # List of point cloud names to process
# ============================================================================

if not parent_dir.exists():
    raise ValueError("Parent directory does not exist.") 

def get_custom_data(pc_names, path):

    pc_data = []
    for i, name in enumerate(pc_names):
        # Find first files starting with name
        points_dir = join(path, 'points')
        labels_dir = join(path, 'labels')
        
        pc_path = join(points_dir, next(f for f in os.listdir(points_dir) if f.startswith(name)))
        label_path = join(labels_dir, next(f for f in os.listdir(labels_dir) if f.startswith(name)))
        assert exists(pc_path), f"Point cloud file {pc_path} does not exist."
        assert exists(label_path), f"Label file {label_path} does not exist."

        pc = pd.read_csv(pc_path,
                         header=None,
                         sep=r'\s+',
                         dtype=np.float32).values

        points = pc[:, 0:3]
        feat = pc[:, [4, 5, 6]]
        intensity = pc[:, 3]

        points = np.array(points, dtype=np.float32)
        feat = np.array(feat, dtype=np.float32)
        intensity = np.array(intensity, dtype=np.float32)

        labels = pd.read_csv(label_path,
                            header=None,
                            sep=r'\s+',
                            dtype=np.int32).values
        labels = np.array(labels, dtype=np.int32).reshape((-1,))

        data = {
            'point': points,
            'feat': feat,
            'intensity': intensity,
            'label': labels
        }
        pc_data.append(data)

    return pc_data


def pred_custom_data(pc_names, pcs, pipeline_r):
    vis_points = []
    for i, data in enumerate(pcs):
        name = pc_names[i]

        results_r = pipeline_r.run_inference(data)
        pred_label_r = (results_r['predict_labels'] + 1).astype(np.int32)
        # pred_label_r = results_r['predict_labels'].astype(np.int32)
        # Fill "unlabeled" value because predictions have no 0 values.
        # pred_label_r[0] = 0


        label = data['label']
        pts = data['point']


        vis_d = {
            "name": name + "_randlanet",
            "points": pts,
            "labels": label,
            "pred": pred_label_r,
        }
        vis_points.append(vis_d)


    return vis_points


def get_torch_ckpts():

    # ckpt_folder = "./logs/"
    # os.makedirs(ckpt_folder, exist_ok=True)
    ckpt_path = CHECKPOINT_PATH
    randlanet_url = "https://storage.googleapis.com/open3d-releases/model-zoo/randlanet_semantic3d_202201071330utc.pth"
    if not os.path.exists(ckpt_path):
        # raise FileNotFoundError(f"Checkpoint file {ckpt_path} not found. Please download it manually.")
        cmd = "wget {} -O {}".format(randlanet_url, ckpt_path)
        os.system(cmd)

    return ckpt_path



def main():
    semantic3d_labels = datasets.Semantic3D.get_label_to_names()
    v = vis.Visualizer()
    lut = vis.LabelLUT()
    for val in sorted(semantic3d_labels.keys()):
        lut.add_label(semantic3d_labels[val], val)
    v.set_lut("labels", lut)
    v.set_lut("pred", lut)


    # Get the model configuration from the ml3d repository.
    cfg_path = repo_root / 'ml3d/configs/randlanet_semantic3d.yml'
    model_cfg = utils.Config.load_from_file(str(cfg_path)).model
    model = models.RandLANet(**model_cfg)

    pipeline_r = pipelines.SemanticSegmentation(model)
    pipeline_r.load_ckpt(get_torch_ckpts())


    data_path = DATA_PATH
    pc_names = PC_NAMES
    pcs = get_custom_data(pc_names, data_path)
    pcs_with_pred = pred_custom_data(pc_names, pcs, pipeline_r)

    gt_labels = pcs_with_pred[0]['labels']
    pred_labels_r = pcs_with_pred[0]['pred']

    # ========================================================================
    # Quantitative Analysis: Accuracy, IoU/mIoU, and Confusion Matrix
    # ========================================================================
    metric = SemSegMetric()
    num_classes = len(semantic3d_labels)
    
    # Convert predictions to one-hot format for metric computation
    scores = torch.nn.functional.one_hot(
        torch.tensor(pred_labels_r, dtype=torch.long), 
        num_classes=num_classes
    ).float()
    labels = torch.tensor(gt_labels, dtype=torch.long)
    
    # Update metric with predictions and ground truth
    metric.update(scores, labels)
    
    # Get metrics
    accuracies = metric.acc()
    ious = metric.iou()
    confusion_mat = metric.confusion_matrix
    
    # Display results
    print("\n" + "="*70)
    print("QUANTITATIVE ANALYSIS RESULTS")
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
    print(f"\nConfusion Matrix shape: {confusion_mat.shape}")
    print(f"Confusion Matrix:\n{confusion_mat}")
    print("="*70 + "\n")

    # Compare the gt and pred labels in terms of histogram
    # compare_gt_pred_histogram(gt_labels, pred_labels_r, semantic3d_labels)
    v.visualize(pcs_with_pred)


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(asctime)s - %(module)s - %(message)s",
    )

    main()
