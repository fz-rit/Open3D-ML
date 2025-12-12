"""
Visualization utilities for domain adaptation monitoring.
Generates t-SNE, UMAP plots, and distribution comparisons.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import os


def plot_tsne(source_features: torch.Tensor,
              target_features: torch.Tensor,
              source_labels: Optional[torch.Tensor] = None,
              target_labels: Optional[torch.Tensor] = None,
              save_path: str = 'tsne_plot.png',
              perplexity: int = 30,
              n_samples: int = 30000) -> str:
    """
    Create t-SNE visualization of source and target features.
    
    Args:
        source_features: [N_s, D] source features
        target_features: [N_t, D] target features
        source_labels: [N_s] optional class labels
        target_labels: [N_t] optional class labels
        save_path: Path to save plot
        perplexity: t-SNE perplexity parameter
        n_samples: Maximum samples to plot (for speed)
    
    Returns:
        Path to saved plot
    """
    from sklearn.manifold import TSNE
    
    # Subsample if needed
    source_orig_size = source_features.size(0)
    if source_orig_size > n_samples:
        idx = torch.randperm(source_orig_size)[:n_samples]
        source_features = source_features[idx]
        if source_labels is not None and source_labels.size(0) == source_orig_size:
            source_labels = source_labels[idx]
        elif source_labels is not None:
            # Label count mismatch - disable labels
            source_labels = None
    
    target_orig_size = target_features.size(0)
    if target_orig_size > n_samples:
        idx = torch.randperm(target_orig_size)[:n_samples]
        target_features = target_features[idx]
        if target_labels is not None and target_labels.size(0) == target_orig_size:
            target_labels = target_labels[idx]
        elif target_labels is not None:
            # Label count mismatch - disable labels
            target_labels = None
    
    # Combine features
    all_features = torch.cat([source_features, target_features], dim=0).cpu().numpy()
    n_source = source_features.size(0)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    embedded = tsne.fit_transform(all_features)
    
    # Create plot - either domain separation or class distribution
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    if source_labels is not None and target_labels is not None:
        # Class distribution visualization
        source_labels_np = source_labels.cpu().numpy()
        target_labels_np = target_labels.cpu().numpy()
        
        # Class names mapping
        class_names = {0: 'Ground', 1: 'Trunk', 2: 'Canopy', 3: 'Understory'}
        
        unique_labels = np.unique(np.concatenate([source_labels_np, target_labels_np]))
        # Use Set1 colormap for bright, vibrant colors (better for 4 classes)
        colors = plt.cm.Set1(np.linspace(0, 0.8, max(4, len(unique_labels))))
        
        for i, label in enumerate(unique_labels):
            class_name = class_names.get(int(label), f'Class{label}')
            
            # Source points for this class
            mask_s = source_labels_np == label
            if mask_s.sum() > 0:
                ax.scatter(embedded[:n_source][mask_s, 0], embedded[:n_source][mask_s, 1],
                          c=[colors[i]], alpha=0.8, s=30, label=f'S-{class_name}', marker='o', 
                          edgecolors='white', linewidths=0.8)
            
            # Target points for this class
            mask_t = target_labels_np == label
            if mask_t.sum() > 0:
                ax.scatter(embedded[n_source:][mask_t, 0], embedded[n_source:][mask_t, 1],
                          c=[colors[i]], alpha=0.4, s=40, label=f'T-{class_name}', marker='^',
                          edgecolors='white', linewidths=0.8)
        
        ax.set_title('t-SNE: Class Distribution\n(Decoder Features)', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    else:
        # Domain separation visualization (no labels)
        ax.scatter(embedded[:n_source, 0], embedded[:n_source, 1], 
                  c='blue', alpha=0.6, s=30, label='Source', marker='o', edgecolors='darkblue', linewidths=0.5)
        ax.scatter(embedded[n_source:, 0], embedded[n_source:, 1],
                  c='red', alpha=0.6, s=30, label='Target', marker='^', edgecolors='darkred', linewidths=0.5)
        ax.set_title('t-SNE: Domain Alignment Visualization\n(Encoder Features)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=11)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path


def plot_umap(source_features: torch.Tensor,
              target_features: torch.Tensor,
              source_labels: Optional[torch.Tensor] = None,
              target_labels: Optional[torch.Tensor] = None,
              save_path: str = 'umap_plot.png',
              n_neighbors: int = 15,
              n_samples: int = 30000) -> str:
    """
    Create UMAP visualization of source and target features.
    
    Args:
        source_features: [N_s, D] source features
        target_features: [N_t, D] target features
        source_labels: [N_s] optional class labels
        target_labels: [N_t] optional class labels
        save_path: Path to save plot
        n_neighbors: UMAP n_neighbors parameter
        n_samples: Maximum samples to plot
    
    Returns:
        Path to saved plot
    """
    try:
        import umap
    except ImportError:
        print("UMAP not installed. Install with: pip install umap-learn")
        return None
    
    # Subsample if needed
    source_orig_size = source_features.size(0)
    if source_orig_size > n_samples:
        idx = torch.randperm(source_orig_size)[:n_samples]
        source_features = source_features[idx]
        if source_labels is not None and source_labels.size(0) == source_orig_size:
            source_labels = source_labels[idx]
        elif source_labels is not None:
            # Label count mismatch - disable labels
            source_labels = None
    
    target_orig_size = target_features.size(0)
    if target_orig_size > n_samples:
        idx = torch.randperm(target_orig_size)[:n_samples]
        target_features = target_features[idx]
        if target_labels is not None and target_labels.size(0) == target_orig_size:
            target_labels = target_labels[idx]
        elif target_labels is not None:
            # Label count mismatch - disable labels
            target_labels = None
    
    # Combine features
    all_features = torch.cat([source_features, target_features], dim=0).cpu().numpy()
    n_source = source_features.size(0)
    
    # Apply UMAP
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
    embedded = reducer.fit_transform(all_features)
    
    # Create plot - either domain separation or class distribution
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    if source_labels is not None and target_labels is not None:
        # Class distribution visualization
        source_labels_np = source_labels.cpu().numpy()
        target_labels_np = target_labels.cpu().numpy()
        
        # Class names mapping
        class_names = {0: 'Ground', 1: 'Trunk', 2: 'Canopy', 3: 'Understory'}
        
        unique_labels = np.unique(np.concatenate([source_labels_np, target_labels_np]))
        # Use Set1 colormap for bright, vibrant colors (better for 4 classes)
        colors = plt.cm.Set1(np.linspace(0, 0.8, max(4, len(unique_labels))))
        
        for i, label in enumerate(unique_labels):
            class_name = class_names.get(int(label), f'Class{label}')
            
            # Source points for this class
            mask_s = source_labels_np == label
            if mask_s.sum() > 0:
                ax.scatter(embedded[:n_source][mask_s, 0], embedded[:n_source][mask_s, 1],
                          c=[colors[i]], alpha=0.8, s=30, label=f'S-{class_name}', marker='o',
                          edgecolors='white', linewidths=0.8)
            
            # Target points for this class
            mask_t = target_labels_np == label
            if mask_t.sum() > 0:
                ax.scatter(embedded[n_source:][mask_t, 0], embedded[n_source:][mask_t, 1],
                          c=[colors[i]], alpha=0.4, s=40, label=f'T-{class_name}', marker='^',
                          edgecolors='white', linewidths=0.8)
        
        ax.set_title('UMAP: Class Distribution\n(Decoder Features)', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    else:
        # Domain separation visualization (no labels)
        ax.scatter(embedded[:n_source, 0], embedded[:n_source, 1], 
                  c='blue', alpha=0.6, s=30, label='Source', marker='o', edgecolors='darkblue', linewidths=0.5)
        ax.scatter(embedded[n_source:, 0], embedded[n_source:, 1],
                  c='red', alpha=0.6, s=30, label='Target', marker='^', edgecolors='darkred', linewidths=0.5)
        ax.set_title('UMAP: Domain Alignment Visualization\n(Encoder Features)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
    
    ax.set_xlabel('UMAP Dimension 1', fontsize=11)
    ax.set_ylabel('UMAP Dimension 2', fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path


def plot_feature_distributions(source_features: torch.Tensor,
                               target_features: torch.Tensor,
                               save_path: str = 'feature_distributions.png',
                               n_features: int = 48) -> str:
    """
    Plot histograms comparing source and target feature distributions.
    
    Args:
        source_features: [N_s, D] source features
        target_features: [N_t, D] target features
        save_path: Path to save plot
        n_features: Number of feature dimensions to plot
    
    Returns:
        Path to saved plot
    """
    d = min(source_features.size(1), n_features)
    
    fig, axes = plt.subplots(6, 8, figsize=(24, 18))
    axes = axes.flatten()
    
    for i in range(d):
        ax = axes[i]
        
        source_vals = source_features[:, i].cpu().numpy()
        target_vals = target_features[:, i].cpu().numpy()
        
        ax.hist(source_vals, bins=50, alpha=0.5, label='Source', color='blue', density=True)
        ax.hist(target_vals, bins=50, alpha=0.5, label='Target', color='red', density=True)
        ax.set_title(f'Feature Dim {i}', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    plt.suptitle('Feature Distribution Comparison (First 48 Dims)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path


def plot_covariance_matrices(source_features: torch.Tensor,
                             target_features: torch.Tensor,
                             save_path: str = 'covariance_matrices.png') -> str:
    """
    Visualize covariance matrices of source and target features.
    
    Args:
        source_features: [N_s, D] source features
        target_features: [N_t, D] target features
        save_path: Path to save plot
    
    Returns:
        Path to saved plot
    """
    # Center features
    source_centered = source_features - source_features.mean(dim=0, keepdim=True)
    target_centered = target_features - target_features.mean(dim=0, keepdim=True)
    
    # Compute covariances
    cov_s = torch.mm(source_centered.t(), source_centered) / (source_features.size(0) - 1)
    cov_t = torch.mm(target_centered.t(), target_centered) / (target_features.size(0) - 1)
    
    cov_s_np = cov_s.cpu().numpy()
    cov_t_np = cov_t.cpu().numpy()
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Source covariance
    im1 = axes[0].imshow(cov_s_np, cmap='viridis', aspect='auto')
    axes[0].set_title('Source Covariance', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=axes[0])
    
    # Target covariance
    im2 = axes[1].imshow(cov_t_np, cmap='viridis', aspect='auto')
    axes[1].set_title('Target Covariance', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=axes[1])
    
    # Difference
    diff = np.abs(cov_s_np - cov_t_np)
    im3 = axes[2].imshow(diff, cmap='hot', aspect='auto')
    axes[2].set_title('Absolute Difference', fontsize=12, fontweight='bold')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path


def plot_layer_alignment_progress(layer_history: Dict[int, Dict[str, List]],
                                  save_path: str = 'layer_alignment.png') -> str:
    """
    Plot alignment metrics over training for each layer.
    
    Args:
        layer_history: Dictionary mapping layer_idx -> {metric: [values]}
        save_path: Path to save plot
    
    Returns:
        Path to saved plot
    """
    n_layers = len(layer_history)
    if n_layers == 0:
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_layers))
    
    # Plot MMD
    ax = axes[0]
    for i, (layer_idx, history) in enumerate(sorted(layer_history.items())):
        if 'mmd_rbf' in history and len(history['mmd_rbf']) > 0:
            ax.plot(history['epoch'], history['mmd_rbf'], 
                   marker='o', label=f'Layer {layer_idx}', color=colors[i])
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('MMD (RBF)', fontsize=11)
    ax.set_title('MMD per Layer', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot Covariance Distance
    ax = axes[1]
    for i, (layer_idx, history) in enumerate(sorted(layer_history.items())):
        if 'cov_distance' in history and len(history['cov_distance']) > 0:
            ax.plot(history['epoch'], history['cov_distance'],
                   marker='s', label=f'Layer {layer_idx}', color=colors[i])
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Covariance Distance', fontsize=11)
    ax.set_title('Covariance Distance per Layer', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path


def plot_training_metrics(metrics_history: Dict[str, List],
                          save_path: str = 'training_metrics.png') -> str:
    """
    Plot overall training metrics (losses, IoU, etc.).
    
    Args:
        metrics_history: Dictionary with epoch, mmd, source_val_iou, target_val_iou
        save_path: Path to save plot
    
    Returns:
        Path to saved plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = metrics_history.get('epoch', [])
    
    # Plot 1: MMD over time
    ax = axes[0, 0]
    if 'mmd' in metrics_history and len(metrics_history['mmd']) > 0:
        ax.plot(epochs, metrics_history['mmd'], marker='o', color='purple', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MMD')
        ax.set_title('Domain Discrepancy (MMD)', fontweight='bold')
        ax.grid(alpha=0.3)
    
    # Plot 2: Validation IoU
    ax = axes[0, 1]
    if 'source_val_iou' in metrics_history and len(metrics_history['source_val_iou']) > 0:
        ax.plot(epochs, metrics_history['source_val_iou'], 
               marker='o', label='Source', color='blue', linewidth=2)
    if 'target_val_iou' in metrics_history and len(metrics_history['target_val_iou']) > 0:
        ax.plot(epochs, metrics_history['target_val_iou'],
               marker='s', label='Target', color='red', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('IoU')
    ax.set_title('Validation IoU', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 3: Covariance distance
    ax = axes[1, 0]
    if 'cov_distance' in metrics_history and len(metrics_history['cov_distance']) > 0:
        ax.plot(epochs, metrics_history['cov_distance'], marker='o', color='green', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Covariance Distance')
        ax.set_title('Covariance Alignment', fontweight='bold')
        ax.grid(alpha=0.3)
    
    # Plot 4: A-distance
    ax = axes[1, 1]
    if 'a_distance' in metrics_history and len(metrics_history['a_distance']) > 0:
        ax.plot(epochs, metrics_history['a_distance'], marker='o', color='orange', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('A-distance')
        ax.set_title('Domain Separability (A-distance)', fontweight='bold')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path
