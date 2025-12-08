"""
Standalone Domain Adaptation Monitoring Script.

Run this script to analyze a trained DA model and generate comprehensive monitoring reports.

Usage:
    python scripts/monitor_domain_adaptation.py --config path/to/config.yml --checkpoint path/to/ckpt
    
    # Or monitor all checkpoints in a directory
    python scripts/monitor_domain_adaptation.py --config path/to/config.yml --ckpt_dir path/to/ckpt_dir
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import torch
import numpy as np
from tqdm import tqdm

# Add Open3D-ML to path
sys.path.append(str(Path(__file__).parent.parent))

from ml3d.torch.dataloaders import TorchDataloader, get_sampler
from ml3d.torch.modules.metrics.domain_metrics import (
    compute_mmd, compute_covariance_distance, compute_layer_alignment_quality,
    compute_a_distance, DomainMetricsTracker
)
from ml3d.torch.modules.metrics.domain_visualizations import (
    plot_tsne, plot_umap, plot_feature_distributions, plot_covariance_matrices,
    plot_layer_alignment_progress, plot_training_metrics
)
from ml3d.torch.modules.metrics.domain_report import generate_html_report, save_metrics_json
from ml3d.utils import Config
from ml3d.torch.models import RandLANetDA, KPFCNNDA
from ml3d.datasets import Semantic3DUnified, ForestSemantic, DigiForestUnified
from torch.utils.data import DataLoader


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_dataset(dataset_cfg):
    """Initialize dataset from config."""
    dataset_name = dataset_cfg['name']
    dataset_path = dataset_cfg['dataset_path']
    
    if dataset_name == 'Semantic3DUnified':
        return Semantic3DUnified(dataset_path)
    elif dataset_name == 'ForestSemantic':
        return ForestSemantic(dataset_path)
    elif dataset_name == 'DigiForestUnified':
        labels_path = dataset_cfg.get('labels_path')
        return DigiForestUnified(dataset_path, labels_path=labels_path)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_model(model_cfg, device):
    """Initialize model from config."""
    model_name = model_cfg['name']
    
    if model_name == 'RandLANetDA':
        model = RandLANetDA(**model_cfg)
    elif model_name == 'KPFCNNDA':
        model = KPFCNNDA(**model_cfg)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = model.to(device)
    model.eval()
    return model


def load_checkpoint(model, checkpoint_path):
    """Load model weights from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 0)
    else:
        model.load_state_dict(checkpoint)
        epoch = 0
    
    return epoch


def extract_features(dataloader, model, device, max_batches=20):
    """Extract features from dataloader."""
    features_by_layer = None
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, inputs in enumerate(tqdm(dataloader, desc='Extracting features', total=max_batches)):
            if batch_idx >= max_batches:
                break
            
            try:
                if hasattr(inputs['data'], 'to'):
                    inputs['data'].to(device)
                
                results = model(inputs['data'], return_intermediate_features=True)
                if isinstance(results, tuple) and len(results) >= 2:
                    _, features = results[0], results[1]
                else:
                    continue
                
                if features_by_layer is None:
                    features_by_layer = [[] for _ in range(len(features))]
                
                for layer_idx, feat in enumerate(features):
                    if feat.dim() > 2:
                        feat = feat.reshape(feat.size(0), -1)
                    features_by_layer[layer_idx].append(feat.cpu())
                
                if 'labels' in inputs:
                    labels = inputs['labels']
                    if isinstance(labels, torch.Tensor):
                        all_labels.append(labels.cpu().flatten())
            
            except Exception as e:
                print(f"Warning: Failed to extract features from batch {batch_idx}: {e}")
                continue
    
    if features_by_layer is None:
        return None, None
    
    features_list = []
    for layer_feats in features_by_layer:
        if layer_feats:
            features_list.append(torch.cat(layer_feats, dim=0))
    
    labels_tensor = torch.cat(all_labels, dim=0) if all_labels else None
    
    return features_list, labels_tensor


def run_monitoring(config_path, checkpoint_path, output_dir, device='cuda'):
    """Run domain adaptation monitoring."""
    
    # Load config
    print("Loading configuration...")
    cfg = load_config(config_path)
    
    # Initialize datasets
    print("Initializing datasets...")
    source_dataset = get_dataset(cfg['source_dataset'])
    target_dataset = get_dataset(cfg['target_dataset'])
    
    # Initialize model
    print("Initializing model...")
    model = get_model(cfg['model'], device)
    
    # Load checkpoint
    epoch = load_checkpoint(model, checkpoint_path)
    
    # Create dataloaders
    print("Creating dataloaders...")
    source_valid = source_dataset.get_split('validation')
    source_loader = DataLoader(
        TorchDataloader(
            dataset=source_valid,
            preprocess=model.preprocess,
            transform=model.transform,
            sampler=source_valid.sampler,
            use_cache=False,
            steps_per_epoch=20
        ),
        batch_size=cfg['pipeline'].get('batch_size', 8),
        collate_fn=model.batcher.collate_fn
    )
    
    target_test = target_dataset.get_split('test')
    target_loader = DataLoader(
        TorchDataloader(
            dataset=target_test,
            preprocess=model.preprocess,
            transform=model.transform,
            sampler=target_test.sampler,
            use_cache=False,
            steps_per_epoch=20
        ),
        batch_size=cfg['pipeline'].get('batch_size', 8),
        collate_fn=model.batcher.collate_fn
    )
    
    # Extract features
    print("\n=== Extracting Source Features ===")
    source_features_list, source_labels = extract_features(source_loader, model, device)
    
    print("\n=== Extracting Target Features ===")
    target_features_list, target_labels = extract_features(target_loader, model, device)
    
    if not source_features_list or not target_features_list:
        print("ERROR: Failed to extract features")
        return
    
    # Compute metrics
    print("\n=== Computing Alignment Metrics ===")
    layer_metrics = compute_layer_alignment_quality(
        source_features_list, target_features_list, 
        use_geodesic=cfg['pipeline'].get('use_geodesic', True)
    )
    
    source_feat_deep = source_features_list[-1]
    target_feat_deep = target_features_list[-1]
    
    mmd = compute_mmd(source_feat_deep, target_feat_deep, kernel='rbf')
    cov_dist = compute_covariance_distance(source_feat_deep, target_feat_deep, True)
    
    try:
        a_dist = compute_a_distance(source_feat_deep, target_feat_deep)
    except:
        a_dist = 0.0
    
    metrics = {
        'mmd': mmd,
        'cov_distance': cov_dist,
        'a_distance': a_dist,
        'source_val_iou': 0.0,  # Would need to run inference
        'target_val_iou': 0.0,
    }
    
    print(f"MMD (RBF): {mmd:.4f}")
    print(f"Covariance Distance: {cov_dist:.4f}")
    print(f"A-Distance: {a_dist:.4f}")
    
    # Generate visualizations
    print("\n=== Generating Visualizations ===")
    os.makedirs(output_dir, exist_ok=True)
    
    plots = {}
    
    # t-SNE
    try:
        tsne_path = os.path.join(output_dir, 'tsne.png')
        plot_tsne(source_feat_deep, target_feat_deep, source_labels, target_labels, save_path=tsne_path)
        plots['tsne'] = tsne_path
        print(f"✓ t-SNE: {tsne_path}")
    except Exception as e:
        print(f"✗ t-SNE failed: {e}")
    
    # UMAP
    try:
        umap_path = os.path.join(output_dir, 'umap.png')
        umap_result = plot_umap(source_feat_deep, target_feat_deep, source_labels, target_labels, save_path=umap_path)
        if umap_result:
            plots['umap'] = umap_path
            print(f"✓ UMAP: {umap_path}")
    except Exception as e:
        print(f"✗ UMAP failed: {e}")
    
    # Feature distributions
    try:
        dist_path = os.path.join(output_dir, 'feature_distributions.png')
        plot_feature_distributions(source_feat_deep, target_feat_deep, save_path=dist_path)
        plots['feature_distributions'] = dist_path
        print(f"✓ Feature Distributions: {dist_path}")
    except Exception as e:
        print(f"✗ Feature distributions failed: {e}")
    
    # Covariance matrices
    try:
        cov_path = os.path.join(output_dir, 'covariance_matrices.png')
        plot_covariance_matrices(source_feat_deep, target_feat_deep, save_path=cov_path)
        plots['covariance_matrices'] = cov_path
        print(f"✓ Covariance Matrices: {cov_path}")
    except Exception as e:
        print(f"✗ Covariance matrices failed: {e}")
    
    # Generate HTML report
    print("\n=== Generating HTML Report ===")
    try:
        report_data = {
            'metrics': metrics,
            'layer_metrics': layer_metrics,
            'plots': plots,
            'config': {
                'coral_weight': cfg['pipeline'].get('coral_weight', 0.1),
                'progressive_steps': cfg['pipeline'].get('progressive_steps', 5000),
                'alignment_layers': cfg['pipeline'].get('alignment_layers', [0, 2, 4]),
                'layer_weights': cfg['pipeline'].get('layer_weights', [0.33, 0.33, 0.34]),
                'batch_size': cfg['pipeline'].get('batch_size', 8),
                'learning_rate': cfg['pipeline']['optimizer'].get('lr', 0.001),
                'max_epoch': cfg['pipeline'].get('max_epoch', 150),
            }
        }
        
        report_path = os.path.join(output_dir, 'report.html')
        generate_html_report(report_data, report_path, epoch)
        print(f"✓ HTML Report: {report_path}")
        
        json_path = os.path.join(output_dir, 'metrics.json')
        save_metrics_json(metrics, json_path)
        print(f"✓ JSON Metrics: {json_path}")
        
    except Exception as e:
        print(f"✗ Report generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n=== Monitoring Complete ===")
    print(f"All outputs saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Domain Adaptation Monitoring')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--ckpt_dir', type=str, help='Directory with checkpoints (monitor all)')
    parser.add_argument('--output_dir', type=str, default='./da_monitoring', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    if not args.checkpoint and not args.ckpt_dir:
        parser.error("Must provide either --checkpoint or --ckpt_dir")
    
    if args.checkpoint:
        # Monitor single checkpoint
        output_dir = args.output_dir
        run_monitoring(args.config, args.checkpoint, output_dir, args.device)
    
    elif args.ckpt_dir:
        # Monitor all checkpoints in directory
        ckpt_files = sorted(Path(args.ckpt_dir).glob('*.pth'))
        print(f"Found {len(ckpt_files)} checkpoints")
        
        for ckpt_file in ckpt_files:
            ckpt_name = ckpt_file.stem
            output_dir = os.path.join(args.output_dir, ckpt_name)
            print(f"\n{'='*60}")
            print(f"Processing: {ckpt_name}")
            print(f"{'='*60}")
            
            try:
                run_monitoring(args.config, str(ckpt_file), output_dir, args.device)
            except Exception as e:
                print(f"ERROR processing {ckpt_name}: {e}")
                import traceback
                traceback.print_exc()
                continue


if __name__ == '__main__':
    main()
