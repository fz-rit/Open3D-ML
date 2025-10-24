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

import open3d.ml as _ml3d
import open3d.ml.torch as ml3d

log = logging.getLogger(__name__)


def download_checkpoint(ckpt_path, url):
    """Download model checkpoint if it doesn't exist."""
    if not os.path.exists(ckpt_path):
        log.info(f"Downloading checkpoint from {url}")
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        cmd = f"wget {url} -O {ckpt_path}"
        os.system(cmd)
        log.info("Download complete")
    else:
        log.info(f"Using existing checkpoint: {ckpt_path}")


def main():
    parser = argparse.ArgumentParser(description='Run inference with RandLANet on Semantic3D')
    parser.add_argument('--config', type=str, 
                        default='ml3d/configs/randlanet_semantic3d.yml',
                        help='Path to config YAML file')
    parser.add_argument('--dataset_path', type=str,
                        default='/home/fzhcis/data/semantic3d_full/Semantic3D',
                        help='Path to dataset directory')
    parser.add_argument('--checkpoint', type=str,
                        default='./logs/RandLANet_Semantic3D_torch/checkpoint/ckpt_00200.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--all', action='store_true',
                        help='Run inference on all test samples')
    parser.add_argument('--indices', type=int, nargs='+',
                        help='List of test sample indices to run (space-separated)')
    args = parser.parse_args()
    
    # Load configuration
    if not Path(args.config).exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    cfg = _ml3d.utils.Config.load_from_file(args.config)
    
    # Override dataset path
    cfg.dataset['dataset_path'] = args.dataset_path
    log.info(f"Dataset path: {cfg.dataset['dataset_path']}")
    
    # Initialize model and dataset
    model = ml3d.models.RandLANet(**cfg.model)
    dataset = ml3d.datasets.Semantic3D(
        cfg.dataset.pop('dataset_path', None), 
        **cfg.dataset
    )
    
    # Initialize pipeline
    pipeline = ml3d.pipelines.SemanticSegmentation(
        model, 
        dataset=dataset, 
        device="gpu", 
        **cfg.pipeline
    )
    
    pipeline.load_ckpt(ckpt_path=args.checkpoint)
    
    # Run inference
    test_split = dataset.get_split("test")
    total = len(test_split)
    log.info(f"Test split size: {total} samples")

    if total == 0:
        log.warning("No test samples found. Exiting.")
        return

    # Determine which indices to run
    if args.all:
        indices = list(range(total))
    elif args.indices is not None:
        invalid = [i for i in args.indices if i < 0 or i >= total]
        if invalid:
            raise IndexError(f"Indices out of range (size={total}): {invalid}")
        indices = list(dict.fromkeys(args.indices))  # dedupe, keep order
    else:
        raise ValueError("Please provide either --all or --indices <i j ...>")

    log.info(f"Running inference on {len(indices)} sample(s): {indices[:5]}{' ...' if len(indices) > 5 else ''}")

    for k, idx in enumerate(indices, start=1):
        log.info(f"[{k}/{len(indices)}] Inference on test index {idx}")
        data = test_split.get_data(idx)
        result = pipeline.run_inference(data)

        # Display concise results for each sample
        shapes = {k: (v.shape if hasattr(v, 'shape') else type(v)) for k, v in result.items()}
        log.info(f"Results: {shapes}")
    
    # Uncomment to run full test evaluation
    # log.info("Running full test evaluation...")
    # pipeline.run_test()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(asctime)s - %(module)s - %(message)s",
    )
    main()