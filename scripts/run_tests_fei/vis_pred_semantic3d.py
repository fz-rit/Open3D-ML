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
from compare_gt_pred_labels import compare_gt_pred_histogram
from os.path import exists, join, dirname

example_dir = os.path.dirname(os.path.realpath(__file__))

log = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION - Modify these paths as needed
# ============================================================================
CHECKPOINT_PATH = Path("/home/fzhcis/mylab/Open3D-ML/logs/RandLANet_Semantic3D_torch/checkpoint/ckpt_00100.pth")
DATA_PATH = Path("/home/fzhcis/mylab/data/semantic3d/preprocessed/test")
PC_NAMES = ["bildstein_station3"]  # List of point cloud names to process
# ============================================================================


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
                         sep='\s+',
                         dtype=np.float32).values

        points = pc[:, 0:3]
        feat = pc[:, [4, 5, 6]]
        intensity = pc[:, 3]

        points = np.array(points, dtype=np.float32)
        feat = np.array(feat, dtype=np.float32)
        intensity = np.array(intensity, dtype=np.float32)

        labels = pd.read_csv(label_path,
                            header=None,
                            sep='\s+',
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
        # pred_label_r = (results_r['predict_labels'] + 1).astype(np.int32)
        pred_label_r = results_r['predict_labels'].astype(np.int32)
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

    ckpt_folder = "./logs/"
    os.makedirs(ckpt_folder, exist_ok=True)
    ckpt_path = CHECKPOINT_PATH
    # randlanet_url = "https://storage.googleapis.com/open3d-releases/model-zoo/randlanet_semantic3d_202201071330utc.pth"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file {ckpt_path} not found. Please download it manually.")
        # cmd = "wget {} -O {}".format(randlanet_url, ckpt_path)
        # os.system(cmd)

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


    # data_path = ensure_demo_data()
    # data_path = Path("/home/fzhcis/mylab/data/semantic3d/open3d_randlanet/vis_dir")
    data_path = DATA_PATH
    # pc_names = ["bildstein_station1", "untermaederbrunnen_station3"]
    pc_names = PC_NAMES
    pcs = get_custom_data(pc_names, data_path)
    pcs_with_pred = pred_custom_data(pc_names, pcs, pipeline_r)

    gt_labels = pcs_with_pred[0]['labels']
    pred_labels_r = pcs_with_pred[0]['pred']

    # Compare the gt and pred labels in terms of histogram
    # compare_gt_pred_histogram(gt_labels, pred_labels_r, semantic3d_labels)
    v.visualize(pcs_with_pred)


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(asctime)s - %(module)s - %(message)s",
    )

    main()
