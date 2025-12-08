#!/usr/bin/env python
"""
Validate pretrained checkpoint for domain adaptation.

This script tests if a pretrained checkpoint:
1. Produces valid (non-NaN) outputs on source domain
2. Produces valid outputs on target domain  
3. Has reasonable feature distributions
4. Is compatible with the DA model architecture

Usage:
    python validate_checkpoint_da.py \
        --checkpoint /path/to/checkpoint.pth \
        --config /path/to/da_config.yml \
        --num_batches 10
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import numpy as np
import yaml
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from ml3d.datasets import Semantic3DUnified, ForestSemantic
from ml3d.torch.models import RandLANetDA
from ml3d.torch.dataloaders import TorchDataloader, DefaultBatcher, get_sampler

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def load_config(config_path):
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def test_checkpoint_on_dataset(model, dataset, cfg, num_batches=10, domain_name="Source"):
    """Test checkpoint on a dataset and check for NaN/Inf."""
    log.info(f"\n{'='*60}")
    log.info(f"Testing on {domain_name} domain: {dataset.name}")
    log.info(f"{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.device = device  # Set device attribute for the model
    model.eval()
    
    # Get test split
    test_dataset = dataset.get_split('test')
    test_sampler = test_dataset.sampler
    test_split = TorchDataloader(
        dataset=test_dataset,
        preprocess=model.preprocess,
        transform=model.transform,
        sampler=test_sampler,
        use_cache=dataset.cfg.get('use_cache', True)
    )
    
    batcher = DefaultBatcher()
    test_loader = DataLoader(
        test_split,
        batch_size=cfg['pipeline']['batch_size'],
        sampler=get_sampler(test_sampler),
        collate_fn=batcher.collate_fn
    )
    
    model.trans_point_sampler = test_sampler.get_point_sampler()
    
    # Statistics trackers
    batch_stats = []
    feature_stats = []
    has_nan = False
    has_inf = False
    
    log.info(f"Running {num_batches} batches...")
    
    with torch.no_grad():
        for batch_idx, inputs in enumerate(test_loader):
            if batch_idx >= num_batches:
                break
            
            try:
                if hasattr(inputs['data'], 'to'):
                    inputs['data'].to(device)
                
                # Forward pass with features
                results = model(inputs['data'], return_intermediate_features=True)
                
                if isinstance(results, tuple):
                    logits, features = results
                else:
                    logits = results
                    features = []
                
                # Check logits
                logits_has_nan = torch.isnan(logits).any().item()
                logits_has_inf = torch.isinf(logits).any().item()
                logits_min = logits.min().item()
                logits_max = logits.max().item()
                logits_mean = logits.mean().item()
                logits_std = logits.std().item()
                
                batch_stat = {
                    'batch': batch_idx,
                    'logits_shape': logits.shape,
                    'has_nan': logits_has_nan,
                    'has_inf': logits_has_inf,
                    'min': logits_min,
                    'max': logits_max,
                    'mean': logits_mean,
                    'std': logits_std
                }
                batch_stats.append(batch_stat)
                
                # Check features
                if len(features) > 0:
                    layer_stats = []
                    for layer_idx, feat in enumerate(features):
                        feat_flat = feat.reshape(feat.size(0), -1) if feat.dim() > 2 else feat
                        layer_stat = {
                            'layer': layer_idx,
                            'shape': feat.shape,
                            'has_nan': torch.isnan(feat).any().item(),
                            'has_inf': torch.isinf(feat).any().item(),
                            'mean': feat.mean().item(),
                            'std': feat.std().item()
                        }
                        layer_stats.append(layer_stat)
                    feature_stats.append(layer_stats)
                
                has_nan = has_nan or logits_has_nan
                has_inf = has_inf or logits_has_inf
                
                # Log every batch
                status = "✅ OK" if not (logits_has_nan or logits_has_inf) else "❌ FAIL"
                log.info(f"  Batch {batch_idx}: {status} | "
                        f"logits [{logits_min:.4f}, {logits_max:.4f}] "
                        f"mean={logits_mean:.4f} std={logits_std:.4f}")
                
                if logits_has_nan or logits_has_inf:
                    log.error(f"    ⚠️  NaN={logits_has_nan}, Inf={logits_has_inf}")
                    
                    # Debug info
                    log.error(f"    Input data stats:")
                    if 'labels' in inputs['data']:
                        labels = inputs['data']['labels']
                        log.error(f"      Labels shape: {labels.shape}")
                        log.error(f"      Labels unique: {torch.unique(labels).tolist()}")
                    
                    if len(features) > 0:
                        log.error(f"    Feature stats:")
                        for layer_idx, feat in enumerate(features):
                            log.error(f"      Layer {layer_idx}: shape={feat.shape}, "
                                    f"NaN={torch.isnan(feat).any().item()}, "
                                    f"mean={feat.mean().item():.4f}")
            
            except Exception as e:
                log.error(f"  Batch {batch_idx}: ❌ ERROR - {e}")
                import traceback
                log.error(traceback.format_exc())
                has_nan = True
                break
    
    # Summary
    log.info(f"\n{'='*60}")
    log.info(f"Summary for {domain_name} domain:")
    log.info(f"{'='*60}")
    log.info(f"Total batches tested: {len(batch_stats)}")
    
    if has_nan or has_inf:
        log.error(f"❌ VALIDATION FAILED - Found NaN={has_nan}, Inf={has_inf}")
        failed_batches = [s['batch'] for s in batch_stats if s['has_nan'] or s['has_inf']]
        log.error(f"Failed batches: {failed_batches}")
    else:
        log.info(f"✅ VALIDATION PASSED - No NaN/Inf detected")
    
    # Statistics
    if batch_stats:
        logits_mins = [s['min'] for s in batch_stats if not s['has_nan']]
        logits_maxs = [s['max'] for s in batch_stats if not s['has_nan']]
        logits_means = [s['mean'] for s in batch_stats if not s['has_nan']]
        logits_stds = [s['std'] for s in batch_stats if not s['has_nan']]
        
        if logits_mins:
            log.info(f"\nLogits statistics (valid batches):")
            log.info(f"  Min range: [{min(logits_mins):.4f}, {max(logits_mins):.4f}]")
            log.info(f"  Max range: [{min(logits_maxs):.4f}, {max(logits_maxs):.4f}]")
            log.info(f"  Mean: {np.mean(logits_means):.4f} ± {np.std(logits_means):.4f}")
            log.info(f"  Std: {np.mean(logits_stds):.4f} ± {np.std(logits_stds):.4f}")
    
    # Feature statistics
    if feature_stats:
        log.info(f"\nFeature statistics:")
        num_layers = len(feature_stats[0])
        for layer_idx in range(num_layers):
            layer_means = [batch[layer_idx]['mean'] for batch in feature_stats 
                          if not batch[layer_idx]['has_nan']]
            layer_stds = [batch[layer_idx]['std'] for batch in feature_stats
                         if not batch[layer_idx]['has_nan']]
            if layer_means:
                log.info(f"  Layer {layer_idx}: "
                        f"mean={np.mean(layer_means):.4f}±{np.std(layer_means):.4f}, "
                        f"std={np.mean(layer_stds):.4f}±{np.std(layer_stds):.4f}")
    
    return not (has_nan or has_inf), batch_stats, feature_stats


def main():
    parser = argparse.ArgumentParser(description='Validate checkpoint for domain adaptation')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to pretrained checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to DA configuration YAML')
    parser.add_argument('--num_batches', type=int, default=300,
                       help='Number of batches to test (default: 300)')
    parser.add_argument('--test_source', action='store_true',
                       help='Test on source domain')
    parser.add_argument('--test_target', action='store_true',
                       help='Test on target domain')
    
    args = parser.parse_args()
    
    # Default: test both if neither specified
    if not args.test_source and not args.test_target:
        args.test_source = True
        args.test_target = True
    
    log.info("="*60)
    log.info("Checkpoint Validation for Domain Adaptation")
    log.info("="*60)
    log.info(f"Checkpoint: {args.checkpoint}")
    log.info(f"Config: {args.config}")
    log.info(f"Batches per domain: {args.num_batches}")
    
    # Load config
    cfg = load_config(args.config)
    
    # Initialize datasets
    log.info("\nInitializing datasets...")
    source_dataset = Semantic3DUnified(**cfg['source_dataset'])
    target_dataset = ForestSemantic(**cfg['target_dataset'])
    
    # Initialize model
    log.info("\nInitializing model...")
    model_cfg = cfg['model'].copy()
    model_cfg['ckpt_path'] = args.checkpoint
    model = RandLANetDA(**model_cfg)
    
    # Load checkpoint
    log.info(f"\nLoading checkpoint: {args.checkpoint}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    log.info(f"✅ Checkpoint loaded (epoch {ckpt.get('epoch', 'unknown')})")
    
    # Check model config matches
    log.info("\nModel configuration:")
    log.info(f"  grid_size: {model.cfg.grid_size}")
    log.info(f"  num_points: {model.cfg.num_points}")
    log.info(f"  num_neighbors: {model.cfg.num_neighbors}")
    log.info(f"  num_classes: {model.cfg.num_classes}")
    
    # Run tests
    results = {}
    
    if args.test_source:
        source_ok, source_stats, source_features = test_checkpoint_on_dataset(
            model, source_dataset, cfg, args.num_batches, "Source"
        )
        results['source'] = {
            'passed': source_ok,
            'batch_stats': source_stats,
            'feature_stats': source_features
        }
    
    if args.test_target:
        target_ok, target_stats, target_features = test_checkpoint_on_dataset(
            model, target_dataset, cfg, args.num_batches, "Target"
        )
        results['target'] = {
            'passed': target_ok,
            'batch_stats': target_stats,
            'feature_stats': target_features
        }
    
    # Final summary
    log.info("\n" + "="*60)
    log.info("FINAL VALIDATION RESULTS")
    log.info("="*60)
    
    all_passed = True
    if 'source' in results:
        status = "✅ PASSED" if results['source']['passed'] else "❌ FAILED"
        log.info(f"Source domain: {status}")
        all_passed = all_passed and results['source']['passed']
    
    if 'target' in results:
        status = "✅ PASSED" if results['target']['passed'] else "❌ FAILED"
        log.info(f"Target domain: {status}")
        all_passed = all_passed and results['target']['passed']
    
    log.info("="*60)
    
    if all_passed:
        log.info("✅ Checkpoint is VALID for domain adaptation!")
        log.info("You can proceed with DA training.")
        return 0
    else:
        log.error("❌ Checkpoint validation FAILED!")
        log.error("Do NOT use this checkpoint for DA training.")
        log.error("\nPossible issues:")
        log.error("  1. Checkpoint was trained with different grid_size")
        log.error("  2. Checkpoint is corrupted or has NaN weights")
        log.error("  3. Model architecture mismatch")
        log.error("  4. Numerical instability in pretrained model")
        return 1


if __name__ == '__main__':
    sys.exit(main())
