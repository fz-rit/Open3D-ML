"""
Decoder Class Distribution Monitoring Module.

Monitors decoder features with per-class visualization to understand
how well the model separates different semantic classes in feature space.
Useful for diagnosing class confusion and adaptation quality.
"""

import logging
from os.path import join

import numpy as np
import torch

from ..modules.metrics.domain_visualizations import (
    plot_tsne, plot_umap, plot_feature_distributions
)
from ...utils import make_dir

log = logging.getLogger(__name__)


class DecoderClassMonitor:
    """Handles monitoring and visualization for decoder features with class labels."""
    
    def __init__(self, monitoring_dir, monitor_freq=5, enable_tsne=True, enable_umap=True):
        """
        Initialize decoder class monitor.
        
        Args:
            monitoring_dir: Directory to save monitoring outputs
            monitor_freq: Frequency of monitoring (every N epochs)
            enable_tsne: Enable t-SNE visualization
            enable_umap: Enable UMAP visualization
        """
        self.monitoring_dir = monitoring_dir
        self.monitor_freq = monitor_freq
        self.enable_tsne = enable_tsne
        self.enable_umap = enable_umap
        
        make_dir(monitoring_dir)
        log.info(f"Decoder class monitoring initialized: {monitoring_dir}")
    
    def should_monitor(self, epoch, max_epoch):
        """Check if monitoring should run for this epoch."""
        return epoch % self.monitor_freq == 0 or epoch == max_epoch - 1
    
    def run_monitoring(self, epoch, source_loader, target_loader, model, device):
        """
        Run complete monitoring for current epoch.
        
        Args:
            epoch: Current epoch number
            source_loader: Source domain validation dataloader
            target_loader: Target domain validation dataloader
            model: Model instance
            device: Device to use
        """
        log.info(f"Running decoder class monitoring for epoch {epoch}...")
        model.eval()
        
        # Extract decoder features with labels from both domains
        log.info("Extracting source decoder features...")
        source_features_list, source_labels = self._extract_decoder_features(
            source_loader, model, device, max_batches=10
        )
        
        log.info("Extracting target decoder features...")
        target_features_list, target_labels = self._extract_decoder_features(
            target_loader, model, device, max_batches=10
        )
        
        if not source_features_list or not target_features_list:
            log.warning("Failed to extract decoder features for monitoring")
            return
        
        # Generate visualizations for each decoder layer
        self._generate_visualizations(
            epoch, source_features_list, target_features_list,
            source_labels, target_labels
        )
        
        log.info(f"Decoder class monitoring complete for epoch {epoch}")
    
    def _extract_decoder_features(self, dataloader, model, device, max_batches=10):
        """
        Extract decoder features and labels from a dataloader.
        
        Args:
            dataloader: DataLoader instance
            model: Model with decoder feature extraction capability
            device: Device to use
            max_batches: Maximum batches to process
        
        Returns:
            features_list: List of tensors for each decoder layer
            labels: Tensor of point-level labels
        """
        features_by_layer = None
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, inputs in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break
                
                try:
                    if hasattr(inputs['data'], 'to'):
                        inputs['data'].to(device)
                    
                    # Extract decoder features
                    results = model(inputs['data'], return_decoder_features=True)
                    if isinstance(results, tuple) and len(results) >= 2:
                        _, features = results[0], results[1]
                    else:
                        if batch_idx == 0:
                            log.error(f"Model did not return decoder features! results type: {type(results)}")
                        continue
                    
                    # Only keep last decoder layer (closest to final classification)
                    features = [features[-1]]
                    
                    # Initialize storage on first batch
                    if features_by_layer is None:
                        features_by_layer = [[] for _ in range(len(features))]
                        log.info(f"Extracting last decoder layer only (layer {len(features)-1})")
                    
                    # Extract labels first - handle both dict and attribute access
                    labels = None
                    if 'data' in inputs and isinstance(inputs['data'], dict):
                        if 'labels' in inputs['data']:
                            labels = inputs['data']['labels']
                        elif 'label' in inputs['data']:
                            labels = inputs['data']['label']
                    elif 'data' in inputs and hasattr(inputs['data'], 'label'):
                        labels = inputs['data'].label
                    
                    if labels is not None:
                        if not isinstance(labels, torch.Tensor):
                            labels = torch.from_numpy(labels)
                        # Flatten to match feature points: (B*N,)
                        labels_flat = labels.cpu().flatten()
                        
                        # Get ignored label indices
                        ignored_label_inds = model.cfg.get('ignored_label_inds', [0])
                        
                        # Process the last decoder layer
                        feat = features[0]  # Only one layer now
                        
                        # Features shape: (N_points, C)
                        if torch.isnan(feat).any():
                            feat = torch.nan_to_num(feat, nan=0.0)
                        
                        n_feat_points = feat.size(0)
                        
                        # Match labels to feature points
                        # Decoder features may be subsampled, so take first n_feat_points labels
                        if n_feat_points <= labels_flat.size(0):
                            labels_for_layer = labels_flat[:n_feat_points]
                        else:
                            # Feature points exceed label points - pad with ignored label
                            pad_size = n_feat_points - labels_flat.size(0)
                            pad_label = ignored_label_inds[0] if ignored_label_inds else 0
                            labels_for_layer = torch.cat([
                                labels_flat,
                                torch.full((pad_size,), pad_label, dtype=labels_flat.dtype)
                            ])
                        
                        # Create mask for valid (non-ignored) labels
                        valid_mask = torch.ones(n_feat_points, dtype=torch.bool)
                        for ign_label in ignored_label_inds:
                            valid_mask = valid_mask & (labels_for_layer != ign_label)
                        
                        # Filter features by valid mask
                        feat_filtered = feat.cpu()[valid_mask]
                        features_by_layer[0].append(feat_filtered)
                        
                        # Filter and remap labels
                        labels_filtered = labels_for_layer[valid_mask]
                        
                        # Remap: shift down by number of ignored labels below each label
                        # E.g., if ignored=[0], then {1,2,3,4} -> {0,1,2,3}
                        for ign_label in sorted(ignored_label_inds):
                            labels_filtered = torch.where(labels_filtered > ign_label, 
                                                         labels_filtered - 1, 
                                                         labels_filtered)
                        
                        all_labels.append(labels_filtered)
                
                except Exception as e:
                    log.error(f"Failed to extract decoder features from batch {batch_idx}: {e}")
                    if batch_idx == 0:
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
        
        labels_tensor = torch.cat(all_labels, dim=0) if all_labels else None
        
        if labels_tensor is not None:
            unique, counts = torch.unique(labels_tensor, return_counts=True)
            label_hist = {int(l): int(c) for l, c in zip(unique, counts)}
            log.info(f"Extracted {labels_tensor.shape[0]} labels, histogram: {label_hist}")
        else:
            log.warning("No labels extracted from decoder features")
        
        return features_list, labels_tensor
    
    def _generate_visualizations(self, epoch, source_features_list, target_features_list,
                                source_labels, target_labels):
        """Generate visualization plots for decoder layers.
        
        Visualizes the last decoder layer (penultimate before classification) - most semantic features.
        """
        epoch_dir = join(self.monitoring_dir, f'epoch_{epoch:04d}')
        make_dir(epoch_dir)
        
        # We only extract and visualize the last decoder layer
        source_feat = source_features_list[0]
        target_feat = target_features_list[0]
        
        log.info(f"Generating decoder class visualizations (penultimate layer) "
                f"(source: {source_feat.shape}, target: {target_feat.shape})")
        
        # Stratified sampling: 8000 samples per class for balanced visualization
        samples_per_class = 8000
        
        def stratified_sample(features, labels, n_per_class):
            """Sample n_per_class points from each class."""
            if labels is None:
                return features, labels
            
            unique_labels = torch.unique(labels)
            sampled_indices = []
            
            for label in unique_labels:
                label_mask = (labels == label)
                label_indices = torch.where(label_mask)[0]
                
                # Sample up to n_per_class points from this class
                n_available = label_indices.size(0)
                n_sample = min(n_available, n_per_class)
                
                if n_sample < n_available:
                    # Random sample
                    perm = torch.randperm(n_available)[:n_sample]
                    sampled = label_indices[perm]
                else:
                    # Use all available
                    sampled = label_indices
                
                sampled_indices.append(sampled)
            
            # Concatenate all sampled indices
            all_indices = torch.cat(sampled_indices)
            return features[all_indices], labels[all_indices]
        
        source_feat_vis, source_labels_vis = stratified_sample(
            source_feat, source_labels, samples_per_class)
        target_feat_vis, target_labels_vis = stratified_sample(
            target_feat, target_labels, samples_per_class)
        
        # t-SNE plot with class labels (use decoder_ prefix to distinguish from encoder plots)
        if self.enable_tsne:
            try:
                tsne_path = join(epoch_dir, 'decoder_tsne_classes.png')
                # Don't pass n_samples - already stratified sampled above
                plot_tsne(source_feat_vis, target_feat_vis, 
                         source_labels_vis, target_labels_vis, 
                         save_path=tsne_path, n_samples=100000)  # High limit since already sampled
                log.info(f"Generated decoder t-SNE plot: {tsne_path}")
            except Exception as e:
                log.warning(f"Failed to generate decoder t-SNE plot: {e}")
        
        # UMAP plot with class labels
        if self.enable_umap:
            try:
                umap_path = join(epoch_dir, 'decoder_umap_classes.png')
                umap_result = plot_umap(source_feat_vis, target_feat_vis,
                                       source_labels_vis, target_labels_vis, 
                                       save_path=umap_path, n_samples=100000)  # High limit since already sampled
                if umap_result:
                    log.info(f"Generated decoder UMAP plot: {umap_path}")
            except Exception as e:
                log.warning(f"UMAP visualization failed: {e}")
        
        # Feature distributions
        try:
            dist_path = join(epoch_dir, 'decoder_feature_distributions.png')
            plot_feature_distributions(source_feat, target_feat, save_path=dist_path)
            log.info(f"Generated decoder feature distributions: {dist_path}")
        except Exception as e:
            log.warning(f"Failed to generate decoder feature distributions: {e}")
