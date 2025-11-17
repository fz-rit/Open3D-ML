#!/usr/bin/env python
"""Generic training script for semantic segmentation models (KPFCNN, RandLANet, etc.).

Usage Examples:
    # Train RandLANet on Semantic3D:
    python train_generic.py --model RandLANet --config /home/fzhcis/mylab/Open3D-ML/ml3d/configs/randlanet_semantic3dunified_xyz.yml
    
    # Train KPFCNN on S3DIS:
    python train_generic.py --model KPFCNN --config /home/fzhcis/mylab/Open3D-ML/ml3d/configs/kpconv_s3dis.yml
    
    # Train RandLANet on SemanticKITTI:
    python train_generic.py --model RandLANet --config /home/fzhcis/mylab/Open3D-ML/ml3d/configs/randlanet_semantickitti.yml
    
    # Train KPFCNN on Toronto3D:
    python train_generic.py --model KPFCNN --config /home/fzhcis/mylab/Open3D-ML/ml3d/configs/kpconv_toronto3d.yml
    
    # Dry-run to validate configuration without training:
    python train_generic.py --model RandLANet --config /path/to/config.yml --dry-run

Arguments:
    --model: Model architecture to use (RandLANet, KPFCNN, PointTransformer, etc.)
    --config: Path to YAML configuration file
    --dry-run: Validate configuration and setup without running training
"""

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
from ml3d.torch.dataloaders import get_sampler, TorchDataloader
from torch.utils.data import DataLoader
import torch

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Generic training script for semantic segmentation models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--model', type=str, required=True, 
                       help='Model architecture (e.g., RandLANet, KPFCNN, PointTransformer)')
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to config YAML file')
    parser.add_argument('--dry-run', action='store_true',
                       help='Validate configuration without running training')
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
    
    log.info(f"Model: {args.model} | Config: {args.config} | Dataset: {cfg.dataset['name']}")
    
    # Initialize model dynamically
    try:
        model_class = getattr(models, args.model)
    except AttributeError:
        raise ValueError(f"Model '{args.model}' not found in ml3d.torch.models. "
                        f"Available models: {[m for m in dir(models) if not m.startswith('_')]}")
    
    model = model_class(**cfg.model)
    
    # Initialize dataset (simple and robust)
    import importlib
    dataset_name = cfg.dataset['name']
    dataset_class = getattr(datasets, dataset_name, None)
    if dataset_class is None:
        # Try to import module by lowercase name to trigger registry
        try:
            importlib.import_module(f"ml3d.datasets.{dataset_name.lower()}")
        except Exception:
            pass
        # Try registry first, then getattr again
        try:
            dataset_class = utils.get_module('dataset', dataset_name)
        except Exception:
            dataset_class = getattr(datasets, dataset_name, None)

    if dataset_class is None:
        raise ValueError(
            f"Dataset '{dataset_name}' not found. Make sure its module is imported "
            f"(e.g., added to ml3d/datasets/__init__.py) or it registers with DATASET."
        )

    dataset = dataset_class(cfg.dataset.pop('dataset_path'), **cfg.dataset)
    
    # Initialize pipeline and train
    pipeline = pipelines.SemanticSegmentation(model, dataset=dataset, device="gpu", **cfg.pipeline)
    
    if args.dry_run:
        log.info("=" * 80)
        log.info("DRY RUN - Checking config and dataloader...")
        log.info("=" * 80)
        log.info(f"Model: {args.model}")
        log.info(f"Dataset: {cfg.dataset['name']} at {dataset.cfg.dataset_path}")
        log.info(f"Pipeline device: {pipeline.device}")
        log.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Split sizes
        train_split = dataset.get_split('train')
        val_split = dataset.get_split('validation')
        log.info(f"Training split size: {len(train_split)} | Validation split size: {len(val_split)}")

        # Minimal dataloader check
        log.info("-" * 80)
        log.info("Testing dataloader...")
        batcher = pipeline.get_batcher(pipeline.device)
        train_sampler = train_split.sampler
        train_ds = TorchDataloader(
            dataset=train_split,
            preprocess=model.preprocess,
            transform=model.transform,
            sampler=train_sampler,
            use_cache=dataset.cfg.use_cache,
            steps_per_epoch=dataset.cfg.get('steps_per_epoch_train', None)
        )
        loader = DataLoader(
            train_ds,
            batch_size=pipeline.cfg.batch_size,
            sampler=get_sampler(train_sampler),
            num_workers=0,
            pin_memory=pipeline.cfg.get('pin_memory', True),
            collate_fn=batcher.collate_fn
        )
        next(iter(loader))
        log.info("Dataloader OK (created and fetched one batch)")
        log.info("=" * 80)
        log.info("Dry-run complete. Remove --dry-run to start training.")
        log.info("=" * 80)
    else:
        pipeline.run_train()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(asctime)s - %(message)s")
    main()
