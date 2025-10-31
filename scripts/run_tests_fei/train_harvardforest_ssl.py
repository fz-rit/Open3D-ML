#!/usr/bin/env python
"""
Training script for HarvardForest3D Self-Supervised Learning.

Usage:
    python scripts/train_harvardforest_ssl.py --cfg ml3d/configs/harvardforest3d_ssl.yml
    python scripts/train_harvardforest_ssl.py --cfg ml3d/configs/harvardforest3d_ssl.yml --test_only
"""

import argparse
import sys
from pathlib import Path

# Add Open3D-ML to path (use local repo, not installed package)
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import torch
import ml3d.utils as utils
import ml3d.datasets as datasets
import ml3d.torch.models as models
import ml3d.torch.pipelines as pipelines


def parse_args():
    parser = argparse.ArgumentParser(description='Train RandLANetSSL on HarvardForest3D')
    parser.add_argument('--cfg', type=str, required=True, help='Path to config file')
    parser.add_argument('--test_only', action='store_true', help='Run testing only')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    cfg_path = Path(args.cfg)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    
    cfg = utils.Config.load_from_file(str(cfg_path))
    
    # Validate required config sections
    for section in ['dataset', 'model', 'pipeline']:
        if not hasattr(cfg, section):
            raise ValueError(f"Config missing required section: '{section}'")
    
    # Validate required dataset config
    if not hasattr(cfg.dataset, 'name'):
        raise ValueError("Config missing 'dataset.name'")
    if not hasattr(cfg.dataset, 'dataset_path'):
        raise ValueError("Config missing 'dataset.dataset_path'")
    
    dataset_path = Path(cfg.dataset.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    
    # Validate required model config
    if not hasattr(cfg.model, 'name'):
        raise ValueError("Config missing 'model.name'")
    
    # Validate required pipeline config
    if not hasattr(cfg.pipeline, 'name'):
        raise ValueError("Config missing 'pipeline.name'")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires a GPU to run.")
    
    device = 'cuda'
    
    print("=" * 60)
    print("HarvardForest3D SSL Training")
    print("=" * 60)
    print(f"Config: {args.cfg}")
    print(f"Device: {device}")
    print(f"Dataset: {cfg.dataset.name}")
    print(f"Model: {cfg.model.name}")
    print(f"Pipeline: {cfg.pipeline.name}")
    print("=" * 60)
    
    # Initialize dataset
    dataset_class = getattr(datasets, cfg.dataset.name)
    # Extract dataset_path and pass remaining config as kwargs
    dataset_kwargs = {k: v for k, v in cfg.dataset.items() if k != 'dataset_path'}
    dataset = dataset_class(cfg.dataset.dataset_path, **dataset_kwargs)
    print(f"Train files: {len(dataset.train_files)}, Val files: {len(dataset.val_files)}")
    
    # Initialize model
    model_class = getattr(models, cfg.model.name)
    model = model_class(**cfg.model)
    
    # Initialize pipeline
    pipeline_class = getattr(pipelines, cfg.pipeline.name)
    pipeline = pipeline_class(model=model, dataset=dataset, device=device, **cfg.pipeline)
    
    # Run training or testing
    if args.test_only:
        print("\nRunning testing...")
        pipeline.run_test()
    else:
        print("\nStarting training...")
        pipeline.run_train()
        print(f"\nTraining complete! Checkpoints: {pipeline.cfg.logs_dir}/checkpoint/")
        print(f"TensorBoard: tensorboard --logdir {pipeline.tensorboard_dir}")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
