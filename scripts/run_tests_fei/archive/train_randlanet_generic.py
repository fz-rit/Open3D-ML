#!/usr/bin/env python
"""Generic training script for RandLANet."""

import logging
import sys
import argparse
from pathlib import Path

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

import ml3d.utils as utils
import ml3d.datasets as datasets
import ml3d.torch.models as models
import ml3d.torch.pipelines as pipelines

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Train RandLANet')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    args = parser.parse_args()
    
    # Load and validate config
    if not Path(args.config).exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    cfg = utils.Config.load_from_file(args.config)
    
    for section in ['dataset', 'model', 'pipeline']:
        if not hasattr(cfg, section) or not getattr(cfg, section):
            raise ValueError(f"Config missing '{section}' section")
    
    for param in ['dataset_path', 'name']:
        if param not in cfg.dataset or not cfg.dataset[param]:
            raise ValueError(f"Config missing 'dataset.{param}' parameter")
    
    if not Path(cfg.dataset['dataset_path']).exists():
        raise FileNotFoundError(f"Dataset path not found: {cfg.dataset['dataset_path']}")
    
    log.info(f"Config: {args.config} | Dataset: {cfg.dataset['name']}")
    
    # Initialize and train
    model = models.RandLANet(**cfg.model)
    dataset_class = getattr(datasets, cfg.dataset['name'])
    dataset = dataset_class(cfg.dataset.pop('dataset_path'), **cfg.dataset)
    pipeline = pipelines.SemanticSegmentation(model, dataset=dataset, device="gpu", **cfg.pipeline)
    pipeline.run_train()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(asctime)s - %(message)s")
    main()
