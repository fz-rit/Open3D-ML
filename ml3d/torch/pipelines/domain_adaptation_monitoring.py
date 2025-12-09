"""
Domain Adaptation Monitoring Module.

Handles feature extraction, metric computation, and visualization generation
for domain adaptation training monitoring.
"""

import logging
import os
from os.path import join

import numpy as np
import torch

from ..modules.metrics.domain_metrics import (
    compute_mmd, compute_covariance_distance, compute_layer_alignment_quality,
    compute_a_distance, DomainMetricsTracker
)
from ..modules.metrics.domain_visualizations import (
    plot_tsne, plot_umap, plot_feature_distributions, plot_covariance_matrices,
    plot_layer_alignment_progress, plot_training_metrics
)
from ..modules.metrics.domain_report import generate_html_report, save_metrics_json
from ...utils import make_dir

log = logging.getLogger(__name__)


class DomainAdaptationMonitor:
    """Handles monitoring and visualization for domain adaptation training."""
    
    def __init__(self, monitoring_dir, use_geodesic=True, monitor_freq=5, 
                 enable_tsne=True, enable_umap=True):
        """
        Initialize domain adaptation monitor.
        
        Args:
            monitoring_dir: Directory to save monitoring outputs
            use_geodesic: Use geodesic distance for CORAL metrics
            monitor_freq: Frequency of monitoring (every N epochs)
            enable_tsne: Enable t-SNE visualization
            enable_umap: Enable UMAP visualization
        """
        self.monitoring_dir = monitoring_dir
        self.use_geodesic = use_geodesic
        self.monitor_freq = monitor_freq
        self.enable_tsne = enable_tsne
        self.enable_umap = enable_umap
        self.metrics_tracker = DomainMetricsTracker()
        
        make_dir(monitoring_dir)
        log.info(f"Domain adaptation monitoring initialized: {monitoring_dir}")
    
    def should_monitor(self, epoch, max_epoch):
        """Check if monitoring should run for this epoch."""
        return epoch % self.monitor_freq == 0 or epoch == max_epoch - 1
    
    def run_monitoring(self, epoch, source_loader, target_loader, model, device,
                      source_val_iou=0.0, target_val_iou=0.0):
        """
        Run complete monitoring for current epoch.
        
        Args:
            epoch: Current epoch number
            source_loader: Source domain validation dataloader
            target_loader: Target domain dataloader
            model: Model instance
            device: Device to use
            source_val_iou: Current source validation IoU
            target_val_iou: Current target validation IoU
        """
        log.info(f"Running domain adaptation monitoring for epoch {epoch}...")
        model.eval()
        
        # Extract features from both domains
        log.info("Extracting source domain features...")
        source_features_list = self._extract_features(
            source_loader, model, device, max_batches=20
        )
        
        log.info("Extracting target domain features...")
        target_features_list = self._extract_features(
            target_loader, model, device, max_batches=20
        )
        
        if not source_features_list or not target_features_list:
            log.warning("Failed to extract features for monitoring")
            return
        
        # Compute metrics
        metrics, layer_metrics = self._compute_metrics(
            source_features_list, target_features_list,
            source_val_iou, target_val_iou
        )

        # If metrics are None (e.g., zero variance), skip tracking/plots gracefully
        if metrics is None:
            log.warning("Domain monitoring metrics unavailable for this epoch (skipping tracker and visualizations)")
            return
        
        # Update tracker
        self.metrics_tracker.update(epoch, metrics)
        self.metrics_tracker.update_layer_metrics(epoch, layer_metrics)
        
        # Generate visualizations (no labels - domain separation only)
        self._generate_visualizations(
            epoch, source_features_list, target_features_list,
            metrics, layer_metrics
        )
        
        log.info(f"Domain monitoring complete for epoch {epoch}")
    
    def _extract_features(self, dataloader, model, device, max_batches=20):
        """
        Extract features from a dataloader.
        
        Args:
            dataloader: DataLoader instance
            model: Model with feature extraction capability
            device: Device to use
            max_batches: Maximum batches to process
        
        Returns:
            features_list: List of tensors for each alignment layer
        """
        features_by_layer = None
        
        with torch.no_grad():
            for batch_idx, inputs in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break
                
                try:
                    if hasattr(inputs['data'], 'to'):
                        inputs['data'].to(device)
                    
                    # Extract features
                    results = model(inputs['data'], return_intermediate_features=True)
                    if isinstance(results, tuple) and len(results) >= 2:
                        _, features = results[0], results[1]
                    else:
                        if batch_idx == 0:
                            raise RuntimeError(f"Model did not return features! results type: {type(results)}, "
                                             f"len: {len(results) if isinstance(results, tuple) else 'N/A'}")
                        continue
                    
                    # Initialize storage on first batch
                    if features_by_layer is None:
                        features_by_layer = [[] for _ in range(len(features))]
                    
                    for layer_idx, feat in enumerate(features):
                        # Flatten spatial dimensions if needed
                        if feat.dim() > 2:
                            feat = feat.reshape(feat.size(0), -1)
                        # Filter out NaN values
                        if torch.isnan(feat).any():
                            feat = torch.nan_to_num(feat, nan=0.0)
                        features_by_layer[layer_idx].append(feat.cpu())
                
                except Exception as e:
                    log.error(f"Failed to extract features from batch {batch_idx}: {e}")
                    if batch_idx == 0:
                        # Re-raise on first batch to fail fast with full context
                        raise
                    continue
        
        if features_by_layer is None:
            return None, None
        
        # Concatenate all batches
        features_list = []
        for layer_feats in features_by_layer:
            if layer_feats:
                concatenated = torch.cat(layer_feats, dim=0)
                features_list.append(concatenated)
        
        log.info(f"Extracted features from {len(features_list)} layers")
        return features_list
    
    def _compute_metrics(self, source_features_list, target_features_list,
                        source_val_iou, target_val_iou):
        """Compute domain adaptation metrics."""
        # Layer-wise metrics
        layer_metrics = compute_layer_alignment_quality(
            source_features_list, target_features_list, self.use_geodesic
        )
        
        # Overall metrics (use deepest layer)
        source_feat_deep = source_features_list[-1]
        target_feat_deep = target_features_list[-1]
        
        # Filter NaN values
        if torch.isnan(source_feat_deep).any():
            log.warning(f"Source features contain {torch.isnan(source_feat_deep).sum().item()} NaN values, replacing with 0")
            source_feat_deep = torch.nan_to_num(source_feat_deep, nan=0.0)
        if torch.isnan(target_feat_deep).any():
            log.warning(f"Target features contain {torch.isnan(target_feat_deep).sum().item()} NaN values, replacing with 0")
            target_feat_deep = torch.nan_to_num(target_feat_deep, nan=0.0)
        
        # Check for valid variance
        source_valid = source_feat_deep.std() > 1e-6
        target_valid = target_feat_deep.std() > 1e-6
        if not source_valid or not target_valid:
            log.warning(f"Features have zero variance (source_std={source_feat_deep.std():.6f}, target_std={target_feat_deep.std():.6f})")
            return None, layer_metrics
        
        # Compute metrics
        mmd = compute_mmd(source_feat_deep, target_feat_deep, kernel='rbf')
        cov_dist = compute_covariance_distance(source_feat_deep, target_feat_deep, self.use_geodesic)
        
        try:
            a_dist = compute_a_distance(source_feat_deep, target_feat_deep)
        except Exception as e:
            log.warning(f"Failed to compute A-distance: {e}")
            a_dist = 0.0
        
        metrics = {
            'mmd': mmd,
            'cov_distance': cov_dist,
            'a_distance': a_dist,
            'source_val_iou': source_val_iou,
            'target_val_iou': target_val_iou,
        }
        
        return metrics, layer_metrics
    
    def _generate_visualizations(self, epoch, source_features_list, target_features_list,
                                metrics, layer_metrics):
        """Generate all visualization plots and reports (domain separation only, no per-class labels).
        
        Focus on bottleneck layer (deepest encoder) - most abstract domain-invariant features.
        Layer-wise metrics already track alignment across all specified layers.
        """
        epoch_dir = join(self.monitoring_dir, f'epoch_{epoch:04d}')
        make_dir(epoch_dir)
        
        # Use bottleneck (deepest) layer for visualizations - most important for domain alignment
        source_feat_deep = source_features_list[-1]
        target_feat_deep = target_features_list[-1]
        
        plots = {}
        
        # t-SNE plot (domain separation only)
        if self.enable_tsne:
            try:
                tsne_path = join(epoch_dir, 'tsne.png')
                plot_tsne(source_feat_deep, target_feat_deep, 
                         None, None, save_path=tsne_path)
                plots['tsne'] = tsne_path
                log.info(f"Generated t-SNE plot: {tsne_path}")
            except Exception as e:
                log.warning(f"Failed to generate t-SNE plot: {e}")
        
        # UMAP plot (domain separation only)
        if self.enable_umap:
            try:
                umap_path = join(epoch_dir, 'umap.png')
                umap_result = plot_umap(source_feat_deep, target_feat_deep,
                                       None, None, save_path=umap_path)
                if umap_result:
                    plots['umap'] = umap_path
                    log.info(f"Generated UMAP plot: {umap_path}")
            except Exception as e:
                log.warning(f"UMAP visualization failed: {e}")
        
        # Feature distributions
        try:
            dist_path = join(epoch_dir, 'feature_distributions.png')
            plot_feature_distributions(source_feat_deep, target_feat_deep, save_path=dist_path)
            plots['feature_distributions'] = dist_path
            log.info(f"Generated feature distributions: {dist_path}")
        except Exception as e:
            log.warning(f"Failed to generate feature distributions: {e}")
        
        # Covariance matrices
        try:
            cov_path = join(epoch_dir, 'covariance_matrices.png')
            plot_covariance_matrices(source_feat_deep, target_feat_deep, save_path=cov_path)
            plots['covariance_matrices'] = cov_path
            log.info(f"Generated covariance matrices: {cov_path}")
        except Exception as e:
            log.warning(f"Failed to generate covariance matrices: {e}")
        
        # Layer alignment progress
        try:
            layer_path = join(epoch_dir, 'layer_alignment.png')
            plot_layer_alignment_progress(self.metrics_tracker.layer_history, save_path=layer_path)
            plots['layer_alignment'] = layer_path
            log.info(f"Generated layer alignment plot: {layer_path}")
        except Exception as e:
            log.warning(f"Failed to generate layer alignment plot: {e}")
        
        # Training metrics over time
        try:
            metrics_path = join(epoch_dir, 'training_metrics.png')
            plot_training_metrics(self.metrics_tracker.history, save_path=metrics_path)
            plots['training_metrics'] = metrics_path
            log.info(f"Generated training metrics plot: {metrics_path}")
        except Exception as e:
            log.warning(f"Failed to generate training metrics plot: {e}")
        
        # Check for decoder plots in same epoch directory and add to plots dict
        decoder_plots = {
            'decoder_tsne_classes': join(epoch_dir, 'decoder_tsne_classes.png'),
            'decoder_umap_classes': join(epoch_dir, 'decoder_umap_classes.png'),
            'decoder_feature_distributions': join(epoch_dir, 'decoder_feature_distributions.png'),
        }
        
        for key, path in decoder_plots.items():
            if os.path.exists(path):
                plots[key] = path
                log.info(f"Found decoder plot for report: {key}")
        
        # Generate HTML report
        if metrics:
            try:
                report_path = join(epoch_dir, 'report.html')
                json_path = join(epoch_dir, 'metrics.json')
                
                # Note: config will need to be passed from main pipeline
                report_data = {
                    'metrics': metrics,
                    'layer_metrics': layer_metrics,
                    'plots': plots,
                }
                
                generate_html_report(report_data, report_path, epoch)
                save_metrics_json(metrics, json_path)
                
                log.info(f"Generated HTML report: {report_path}")
            except Exception as e:
                log.warning(f"Failed to generate HTML report: {e}")
