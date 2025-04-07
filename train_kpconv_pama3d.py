import os
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d
from ml3d.datasets.pama3d import PaMa3D

cfg_file = "ml3d/configs/kpconv_pama3d.yml"
cfg = _ml3d.utils.Config.load_from_file(cfg_file)

model = ml3d.models.KPFCNN(**cfg.model)

dataset = PaMa3D(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
pipeline = ml3d.pipelines.SemanticSegmentation(model, dataset=dataset, device="cuda", **cfg.pipeline)
pipeline.run_train()
