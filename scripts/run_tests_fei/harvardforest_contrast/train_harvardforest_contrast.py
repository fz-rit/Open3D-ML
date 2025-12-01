#!/usr/bin/env python3
"""
Training script for PointContrast-style contrastive learning on HarvardForest3D.

This script trains RandLANet or KPConv encoders with contrastive learning
on unlabeled HarvardForest point clouds, starting from Semantic3D pretrained weights.

Usage:
    # Train RandLANet with contrastive learning (using conda env)
    conda run -n pcd_seg_open3d_env python scripts/run_tests_fei/train_harvardforest_contrast.py \
        --cfg ml3d/configs/randlanet_harvardforest_contrast.yml \
        --device cuda
    
    # Or activate environment first
    conda activate pcd_seg_open3d_env
    python scripts/run_tests_fei/train_harvardforest_contrast.py \
        --cfg ml3d/configs/randlanet_harvardforest_contrast.yml \
        --device cuda
    
    # Train KPConv with contrastive learning
    conda run -n pcd_seg_open3d_env python scripts/run_tests_fei/train_harvardforest_contrast.py \
        --cfg ml3d/configs/kpconv_harvardforest_contrast.yml \
        --device cuda \
        --batch_size 4
    
    # Resume training from checkpoint
    conda run -n pcd_seg_open3d_env python scripts/run_tests_fei/train_harvardforest_contrast.py \
        --cfg ml3d/configs/randlanet_harvardforest_contrast.yml \
        --resume logs/checkpoint/ckpt_epoch_050.pth \
        --device cuda
"""

import argparse
import logging
import sys
from pathlib import Path

# Add the Open3D-ML repository root to Python path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

import torch
import numpy as np
import ml3d as _ml3d

from ml3d.utils import Config
from ml3d.datasets import HarvardForest3DContrastive
from ml3d.torch.models import RandLANetContrast, KPConvContrast
from ml3d.torch.pipelines import ContrastiveLearning

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('contrastive_training.log')
    ]
)
log = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train contrastive learning on HarvardForest3D'
    )
    
    parser.add_argument(
        '--cfg',
        type=str,
        required=True,
        help='Path to config file (e.g., ml3d/configs/randlanet_harvardforest_contrast.yml)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for training'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Override batch size from config'
    )
    
    parser.add_argument(
        '--max_epoch',
        type=int,
        default=None,
        help='Override max epochs from config'
    )
    
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=None,
        help='Override learning rate from config'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    return args


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    log.info(f"Random seed set to {seed}")


def load_config(cfg_path):
    """Load configuration from YAML file."""
    if not Path(cfg_path).exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    
    cfg = Config.load_from_file(cfg_path)
    log.info(f"Loaded config from {cfg_path}")
    return cfg


def initialize_dataset(cfg):
    """Initialize HarvardForest3DContrastive dataset."""
    log.info("Initializing HarvardForest3DContrastive dataset...")
    
    dataset_cfg = cfg.dataset
    dataset_path = Path(dataset_cfg.dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset path does not exist: {dataset_path}\n"
            f"Please update 'dataset_path' in the config file."
        )
    
    # Check for LAS files
    las_files = list(dataset_path.glob('*.las'))
    if len(las_files) == 0:
        raise FileNotFoundError(
            f"No .las files found in {dataset_path}\n"
            f"Please check the dataset path."
        )
    
    log.info(f"Found {len(las_files)} .las files in {dataset_path}")
    
    # Initialize dataset
    dataset = HarvardForest3DContrastive(
        dataset_path=str(dataset_path),
        cache_dir=dataset_cfg.get('cache_dir', './logs/cache_harvardforest_contrast'),
        num_points=dataset_cfg.get('num_points', 8192),
        val_split_ratio=dataset_cfg.get('val_split_ratio', 0.1),
        val_split_seed=dataset_cfg.get('val_split_seed', 42),
        test_split_ratio=dataset_cfg.get('test_split_ratio', 0.0),
        use_intensity=dataset_cfg.get('use_intensity', False),
        use_rgb=dataset_cfg.get('use_rgb', False),
        # Augmentation parameters
        augment_rotation_range=dataset_cfg.get('augment_rotation_range', 360),
        augment_scale_min=dataset_cfg.get('augment_scale_min', 0.8),
        augment_scale_max=dataset_cfg.get('augment_scale_max', 1.2),
        augment_jitter_std=dataset_cfg.get('augment_jitter_std', 0.01),
        augment_dropout_ratio=dataset_cfg.get('augment_dropout_ratio', 0.2),
        augment_translate_range=dataset_cfg.get('augment_translate_range', 0.5),
        # Correspondence parameters
        correspondence_threshold=dataset_cfg.get('correspondence_threshold', 0.05),
        min_correspondences=dataset_cfg.get('min_correspondences', 512),
    )
    
    log.info("Dataset initialized successfully")
    return dataset


def initialize_model(cfg, device):
    """Initialize contrastive model (RandLANetContrast or KPConvContrast)."""
    log.info(f"Initializing model: {cfg.model.name}")
    
    model_cfg = cfg.model
    model_name = model_cfg.name
    
    if model_name == 'RandLANetContrast':
        model = RandLANetContrast(
            name=model_name,
            # Encoder parameters
            num_neighbors=model_cfg.get('num_neighbors', 16),
            num_layers=model_cfg.get('num_layers', 5),
            num_points=model_cfg.get('num_points', 8192),
            sub_sampling_ratio=model_cfg.get('sub_sampling_ratio', [4, 4, 4, 4, 2]),
            in_channels=model_cfg.get('in_channels', 3),
            dim_features=model_cfg.get('dim_features', 8),
            dim_output=model_cfg.get('dim_output', [16, 64, 128, 256, 512]),
            grid_size=model_cfg.get('grid_size', 0.06),
            # Projection head
            projection_hidden_dims=model_cfg.get('projection_hidden_dims', [512, 256]),
            projection_output_dim=model_cfg.get('projection_output_dim', 128),
            projection_use_bn=model_cfg.get('projection_use_bn', True),
            projection_dropout=model_cfg.get('projection_dropout', 0.1),
            # Pretrained weights
            pretrained_encoder_path=model_cfg.get('pretrained_encoder_path', None),
            load_encoder_strict=model_cfg.get('load_encoder_strict', False),
            freeze_encoder_epochs=model_cfg.get('freeze_encoder_epochs', 5),
            encoder_lr_scale=model_cfg.get('encoder_lr_scale', 0.1),
            batcher=model_cfg.get('batcher', 'DefaultBatcher'),
        )
    elif model_name == 'KPConvContrast':
        model = KPConvContrast(
            name=model_name,
            # Encoder parameters
            architecture=model_cfg.get('architecture', ['simple', 'resnetb']),
            in_radius=model_cfg.get('in_radius', 4.0),
            first_subsampling_dl=model_cfg.get('first_subsampling_dl', 0.06),
            first_features_dim=model_cfg.get('first_features_dim', 128),
            in_features_dim=model_cfg.get('in_features_dim', 1),
            # Projection head
            projection_hidden_dims=model_cfg.get('projection_hidden_dims', [512, 256]),
            projection_output_dim=model_cfg.get('projection_output_dim', 128),
            projection_use_bn=model_cfg.get('projection_use_bn', True),
            projection_dropout=model_cfg.get('projection_dropout', 0.1),
            # Pretrained weights
            pretrained_encoder_path=model_cfg.get('pretrained_encoder_path', None),
            load_encoder_strict=model_cfg.get('load_encoder_strict', False),
            freeze_encoder_epochs=model_cfg.get('freeze_encoder_epochs', 10),
            encoder_lr_scale=model_cfg.get('encoder_lr_scale', 0.1),
            batcher=model_cfg.get('batcher', 'ConcatBatcher'),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = model.to(device)
    log.info(f"Model initialized and moved to {device}")
    
    return model


def initialize_pipeline(model, dataset, cfg, device):
    """Initialize contrastive learning pipeline."""
    log.info("Initializing ContrastiveLearning pipeline...")
    
    pipeline_cfg = cfg.pipeline
    
    pipeline = ContrastiveLearning(
        model=model,
        dataset=dataset,
        name=pipeline_cfg.get('name', 'ContrastiveLearning'),
        device=device,
        # Training parameters
        learning_rate=pipeline_cfg.get('learning_rate', 1e-3),
        weight_decay=pipeline_cfg.get('weight_decay', 1e-4),
        scheduler_gamma=pipeline_cfg.get('scheduler_gamma', 0.99),
        batch_size=pipeline_cfg.get('batch_size', 8),
        val_batch_size=pipeline_cfg.get('val_batch_size', 8),
        max_epoch=pipeline_cfg.get('max_epoch', 100),
        save_ckpt_freq=pipeline_cfg.get('save_ckpt_freq', 10),
        # Contrastive parameters
        temperature=pipeline_cfg.get('temperature', 0.07),
        freeze_encoder_epochs=pipeline_cfg.get('freeze_encoder_epochs', 5),
        # Logging
        main_log_dir=pipeline_cfg.get('main_log_dir', './logs'),
        train_sum_dir=pipeline_cfg.get('train_sum_dir', 'train_log'),
    )
    
    log.info("Pipeline initialized successfully")
    return pipeline


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load config
    cfg = load_config(args.cfg)
    
    # Override config with command line arguments
    if args.batch_size is not None:
        cfg.pipeline.batch_size = args.batch_size
        cfg.pipeline.val_batch_size = args.batch_size
        log.info(f"Batch size overridden to {args.batch_size}")
    
    if args.max_epoch is not None:
        cfg.pipeline.max_epoch = args.max_epoch
        log.info(f"Max epochs overridden to {args.max_epoch}")
    
    if args.learning_rate is not None:
        cfg.pipeline.learning_rate = args.learning_rate
        log.info(f"Learning rate overridden to {args.learning_rate}")
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        log.warning("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    log.info("=" * 80)
    log.info("POINTCONTRAST-STYLE CONTRASTIVE LEARNING ON HARVARDFOREST3D")
    log.info("=" * 80)
    log.info(f"Config: {args.cfg}")
    log.info(f"Device: {args.device}")
    log.info(f"Model: {cfg.model.name}")
    log.info("=" * 80)
    
    # Initialize components
    dataset = initialize_dataset(cfg)
    model = initialize_model(cfg, args.device)
    pipeline = initialize_pipeline(model, dataset, cfg, args.device)
    
    # Resume from checkpoint if specified
    if args.resume is not None:
        log.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        pipeline.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        pipeline.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        pipeline.current_epoch = checkpoint['epoch'] + 1
        log.info(f"Resumed from epoch {checkpoint['epoch'] + 1}")
    
    # Start training
    log.info("\n" + "=" * 80)
    log.info("STARTING TRAINING")
    log.info("=" * 80 + "\n")
    
    try:
        pipeline.run_train()
    except KeyboardInterrupt:
        log.info("\n" + "=" * 80)
        log.info("Training interrupted by user")
        log.info("=" * 80)
    except Exception as e:
        log.error(f"Training failed with error: {e}", exc_info=True)
        raise
    
    log.info("\n" + "=" * 80)
    log.info("TRAINING COMPLETED")
    log.info("=" * 80)
    log.info(f"Checkpoints saved in: {cfg.pipeline.main_log_dir}/checkpoint/")
    log.info(f"Best model: {cfg.pipeline.main_log_dir}/checkpoint/ckpt_best.pth")
    log.info("=" * 80)


if __name__ == '__main__':
    main()
