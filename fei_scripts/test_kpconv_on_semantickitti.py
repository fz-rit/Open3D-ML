import os
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d
from pathlib import Path
import numpy as np

cfg_file = "ml3d/configs/kpconv_semantickitti.yml"
cfg = _ml3d.utils.Config.load_from_file(cfg_file)

model = ml3d.models.KPFCNN(**cfg.model)
dataset = ml3d.datasets.SemanticKITTI(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
pipeline = ml3d.pipelines.SemanticSegmentation(model, dataset=dataset, device="gpu", **cfg.pipeline)

# download the weights.
ckpt_folder = Path("./logs/KPFCNN__torch/checkpoint")
ckpt_folder.mkdir(parents=True, exist_ok=True)
ckpt_path = ckpt_folder / "kpconv_semantickitti_202009090354utc.pth"
kpconv_url = "https://storage.googleapis.com/open3d-releases/model-zoo/kpconv_semantickitti_202009090354utc.pth"
if not ckpt_path.exists():
    cmd = "wget {} -O {}".format(kpconv_url, str(ckpt_path))
    os.system(cmd)

if not ckpt_path.exists():
    raise ValueError("-------------Download model failed!!-------------")

# load the parameters.
pipeline.load_ckpt(ckpt_path=ckpt_path)

test_split = dataset.get_split("test")
data = test_split.get_data(0)

# print information of data, shape, number of elements, etc.
# print(len(data))
# print(data)
# print(data.keys())
# print(data["point"].shape)
# print(type(data["feat"]))
# print(data["label"].shape)

# run inference on a single example.
# returns dict with 'predict_labels' and 'predict_scores'.
result = pipeline.run_inference(data)
print(result['predict_labels'].shape)
print(result['predict_scores'].shape)
# print histogram of predicted labels.
hist = np.bincount(result['predict_labels'].flatten())
print(hist)
# # evaluate performance on the test set; this will write logs to './logs'.
# pipeline.run_test()
