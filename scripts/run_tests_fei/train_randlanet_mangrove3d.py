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


def main():
	cfg_file = "/home/fzhcis/mylab/Open3D-ML/ml3d/configs/randlanet_mangrove3d.yml"
	cfg = utils.Config.load_from_file(cfg_file)
	model = models.RandLANet(**cfg.model)

	dataset = datasets.Mangrove3D(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
	pipeline = pipelines.SemanticSegmentation(model, dataset=dataset, device="gpu", **cfg.pipeline)
	pipeline.run_train()


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


if __name__ == "__main__":
	main()