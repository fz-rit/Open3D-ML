#!/usr/bin/env python
"""
Domain Adaptation Training Script for Point Cloud Semantic Segmentation

This script demonstrates how to train a model with CORAL-based domain adaptation
to transfer knowledge from Semantic3DUnified (source) to ForestSemantic/DigiForest (target).

Usage:
    python train_domain_adaptation.py --config ml3d/configs/randlanet_da_semantic3d_to_forest.yml
    python train_domain_adaptation.py --config ml3d/configs/kpconv_da_semantic3d_to_digiforest.yml
"""

import argparse
import yaml
import logging
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

import torch
from ml3d.torch import pipelines, models
import ml3d.datasets as datasets

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available! This script requires a CUDA-enabled GPU.\n"
            "Please ensure you have:\n"
            "  1. A CUDA-compatible GPU\n"
            "  2. CUDA toolkit installed\n"
            "  3. PyTorch with CUDA support installed"
        )
    
    parser = argparse.ArgumentParser(
        description='Domain Adaptation Training for Point Cloud Segmentation'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to domain adaptation config file'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Load datasets and model without training'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    log.info(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Extract configurations
    source_cfg = config['source_dataset']
    target_cfg = config['target_dataset']
    model_cfg = config['model']
    pipeline_cfg = config['pipeline']
    
    # Initialize SOURCE dataset (labeled)
    log.info(f"Initializing SOURCE dataset: {source_cfg['name']}")
    SourceDataset = getattr(datasets, source_cfg['name'])
    source_dataset = SourceDataset(**source_cfg)
    log.info(f"Source dataset loaded: {len(source_dataset.get_split_list('train'))} train files")
    
    # Initialize TARGET dataset (unlabeled for training)
    log.info(f"Initializing TARGET dataset: {target_cfg['name']}")
    TargetDataset = getattr(datasets, target_cfg['name'])
    target_dataset = TargetDataset(**target_cfg)
    log.info(f"Target dataset loaded: {len(target_dataset.get_split_list('test'))} test files")
    
    # Initialize model
    log.info(f"Initializing model: {model_cfg['name']}")
    Model = getattr(models, model_cfg['name'])
    model = Model(**model_cfg)
    
    # Verify model supports domain adaptation
    if not hasattr(model, 'forward') or 'return_features' not in str(model.forward.__code__.co_varnames):
        raise RuntimeError(
            f"Model {model_cfg['name']} does not fully support domain adaptation. "
            "Consider using RandLANetDA or KPFCNNDA."
        )
    
    # Load pretrained checkpoint if provided
    if model_cfg.get('ckpt_path') and Path(model_cfg['ckpt_path']).exists():
        log.info(f"Loading pretrained checkpoint: {model_cfg['ckpt_path']}")
        # Checkpoint will be loaded by pipeline
    
    # Initialize domain adaptation pipeline
    log.info(f"Initializing pipeline: {pipeline_cfg['name']}")
    Pipeline = getattr(pipelines, pipeline_cfg['name'])
    
    pipeline = Pipeline(
        model=model,
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        device='cuda',
        **pipeline_cfg
    )
    
    if args.dry_run:
        log.info("Dry-run mode: Configuration loaded successfully!")
        log.info(f"Source dataset: {source_dataset.name}")
        log.info(f"Target dataset: {target_dataset.name}")
        log.info(f"Model: {model.__class__.__name__}")
        log.info(f"CORAL weight: {pipeline_cfg.get('coral_weight', 0.1)}")
        log.info(f"Progressive steps: {pipeline_cfg.get('progressive_steps', 5000)}")
        log.info(f"Alignment layers: {pipeline_cfg.get('alignment_layers', 'model default')}")
        return
    
    # Start training
    log.info("=" * 60)
    log.info("Starting Domain Adaptation Training")
    log.info("=" * 60)
    log.info(f"Source: {source_dataset.name}")
    log.info(f"Target: {target_dataset.name}")
    log.info(f"Max epochs: {pipeline_cfg.get('max_epoch', 100)}")
    log.info(f"Batch size: {pipeline_cfg.get('batch_size', 4)}")
    log.info("=" * 60)
    
    try:
        pipeline.run_train()
        log.info("Training completed successfully!")
    except KeyboardInterrupt:
        log.info("Training interrupted by user")
    except Exception as e:
        log.error(f"Training failed with error: {e}")
        raise


if __name__ == '__main__':
    main()
