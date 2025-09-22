import logging
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
# Add the Open3D-ML repository root to Python path
repo_root = Path(__file__).parent.parent.parent  # Go up 3 levels to reach Open3D-ML root
sys.path.insert(0, str(repo_root))

import ml3d.utils as utils
import ml3d.datasets as datasets
import ml3d.torch.models as models
import ml3d.torch.pipelines as pipelines
log = logging.getLogger(__name__)

cfg_file = "ml3d/configs/randlanet_semantic3d.yml"
cfg = utils.Config.load_from_file(cfg_file)

model = models.RandLANet(**cfg.model)

# The dataset path should contain .txt files and .labels files, where the .labels files are the ground
# truth labels for the corresponding .txt files. The .txt files that does not have a corresponding .labels, they will
# be considered as test files.

# cfg.dataset['dataset_path'] = "/shared/rc/mangrove/data/Semantic3D/"
cfg.dataset['dataset_path'] = "/home/fzhcis/mylab/data/semantic3d/preprocessed/test"

dataset = datasets.Semantic3D(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
pipeline = pipelines.SemanticSegmentation(model, dataset=dataset, device="gpu", **cfg.pipeline)
pipeline.run_train()
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

# # evaluate performance on the test set; this will write logs to './logs'.
# pipeline.run_test()