"""
Visualize learned contrastive embeddings using t-SNE, UMAP, and PCA.

This script extracts embeddings from a trained contrastive model and visualizes
them in 2D to show the learned feature space structure.
"""

import sys
import os
import argparse
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Add Open3D-ML to path
# sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from ml3d.torch.dataloaders import ConcatBatcher
from ml3d.utils import Config
from ml3d import datasets, models
from ml3d.torch.pipelines import ContrastiveLearning

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def extract_embeddings(model, dataset, device='cuda', max_samples=None):
    """Extract embeddings for all samples in dataset.
    
    Args:
        model: Trained contrastive model
        dataset: Dataset to extract embeddings from
        device: Device to run on
        max_samples: Maximum number of samples (None = all)
        
    Returns:
        embeddings: (N, D) array of embeddings
        labels: (N,) array of sample names/indices
    """
    model.eval()
    model = model.to(device)
    
    embeddings_list = []
    labels_list = []
    
    num_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    
    log.info(f"Extracting embeddings for {num_samples} samples...")
    
    with torch.no_grad():
        for i in tqdm(range(num_samples)):
            sample = dataset[i]
            
            # Get point cloud
            points = torch.from_numpy(sample['point_view1']).unsqueeze(0).to(device)
            
            # Forward pass
            outputs = model({'point': points})
            embedding = outputs['embeddings'].cpu().numpy()[0]  # (D,)
            
            embeddings_list.append(embedding)
            labels_list.append(sample.get('name', f'sample_{i}'))
    
    embeddings = np.array(embeddings_list)  # (N, D)
    labels = np.array(labels_list)  # (N,)
    
    log.info(f"Extracted embeddings shape: {embeddings.shape}")
    
    return embeddings, labels


def visualize_embeddings(embeddings, labels, output_dir, method='tsne'):
    """Visualize embeddings using dimensionality reduction.
    
    Args:
        embeddings: (N, D) array of embeddings
        labels: (N,) array of labels
        output_dir: Directory to save plots
        method: 'tsne', 'umap', or 'pca'
    """
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log.info(f"Computing {method.upper()} projection...")
    
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        coords_2d = reducer.fit_transform(embeddings)
        title = 't-SNE Projection of Learned Embeddings'
    elif method == 'umap':
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
            coords_2d = reducer.fit_transform(embeddings)
            title = 'UMAP Projection of Learned Embeddings'
        except ImportError:
            log.error("UMAP not installed. Install with: pip install umap-learn")
            return
    elif method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        coords_2d = reducer.fit_transform(embeddings)
        variance = reducer.explained_variance_ratio_
        title = f'PCA Projection (Var: {variance[0]:.1%} + {variance[1]:.1%} = {variance.sum():.1%})'
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot points
    scatter = plt.scatter(
        coords_2d[:, 0], 
        coords_2d[:, 1],
        c=np.arange(len(embeddings)),  # Color by index
        cmap='viridis',
        alpha=0.7,
        s=100,
        edgecolors='black',
        linewidths=0.5
    )
    
    # Add labels for some points (to avoid clutter)
    step = max(1, len(labels) // 20)  # Show ~20 labels
    for i in range(0, len(labels), step):
        plt.annotate(
            labels[i],
            (coords_2d[i, 0], coords_2d[i, 1]),
            fontsize=8,
            alpha=0.7,
            xytext=(5, 5),
            textcoords='offset points'
        )
    
    plt.colorbar(scatter, label='Sample Index')
    plt.xlabel(f'{method.upper()} Component 1', fontsize=12)
    plt.ylabel(f'{method.upper()} Component 2', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    save_path = output_dir / f'embeddings_{method}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    log.info(f"Saved {method.upper()} plot to {save_path}")
    
    # Also save as PDF
    save_path_pdf = output_dir / f'embeddings_{method}.pdf'
    plt.savefig(save_path_pdf, bbox_inches='tight')
    
    plt.close()


def plot_embedding_distribution(embeddings, output_dir):
    """Plot distribution of embedding values.
    
    Args:
        embeddings: (N, D) array of embeddings
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Histogram of all embedding values
    axes[0, 0].hist(embeddings.flatten(), bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Embedding Value', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('Distribution of Embedding Values', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Mean and std per dimension
    mean_per_dim = embeddings.mean(axis=0)
    std_per_dim = embeddings.std(axis=0)
    dims = np.arange(len(mean_per_dim))
    
    axes[0, 1].errorbar(dims, mean_per_dim, yerr=std_per_dim, fmt='o', alpha=0.6)
    axes[0, 1].set_xlabel('Embedding Dimension', fontsize=12)
    axes[0, 1].set_ylabel('Mean ± Std', fontsize=12)
    axes[0, 1].set_title('Mean and Std per Dimension', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. L2 norm distribution
    norms = np.linalg.norm(embeddings, axis=1)
    axes[1, 0].hist(norms, bins=30, alpha=0.7, edgecolor='black', color='orange')
    axes[1, 0].set_xlabel('L2 Norm', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title(f'L2 Norm Distribution (Mean: {norms.mean():.3f})', 
                         fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axvline(norms.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    axes[1, 0].legend()
    
    # 4. Pairwise cosine similarity heatmap (sample)
    from sklearn.metrics.pairwise import cosine_similarity
    n_samples = min(50, len(embeddings))
    sample_indices = np.linspace(0, len(embeddings)-1, n_samples, dtype=int)
    sample_embeddings = embeddings[sample_indices]
    
    similarity = cosine_similarity(sample_embeddings)
    im = axes[1, 1].imshow(similarity, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    axes[1, 1].set_xlabel('Sample Index', fontsize=12)
    axes[1, 1].set_ylabel('Sample Index', fontsize=12)
    axes[1, 1].set_title(f'Cosine Similarity Matrix ({n_samples} samples)', 
                         fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=axes[1, 1], label='Cosine Similarity')
    
    plt.tight_layout()
    
    save_path = output_dir / 'embedding_statistics.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    log.info(f"Saved statistics plot to {save_path}")
    
    plt.close()


def compare_before_after(embeddings_before, embeddings_after, labels, output_dir):
    """Compare embeddings before and after training using t-SNE.
    
    Args:
        embeddings_before: (N, D) embeddings before training (random init)
        embeddings_after: (N, D) embeddings after training
        labels: (N,) sample labels
        output_dir: Directory to save plots
    """
    from sklearn.manifold import TSNE
    
    output_dir = Path(output_dir)
    
    log.info("Computing t-SNE for before/after comparison...")
    
    # Compute t-SNE for both
    perplexity = min(30, len(embeddings_before) - 1)
    coords_before = TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(embeddings_before)
    coords_after = TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(embeddings_after)
    
    # Create side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Before training
    scatter1 = axes[0].scatter(
        coords_before[:, 0], coords_before[:, 1],
        c=np.arange(len(embeddings_before)),
        cmap='viridis', alpha=0.7, s=100,
        edgecolors='black', linewidths=0.5
    )
    axes[0].set_xlabel('t-SNE Component 1', fontsize=14)
    axes[0].set_ylabel('t-SNE Component 2', fontsize=14)
    axes[0].set_title('Before Training (Random Initialization)', fontsize=16, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # After training
    scatter2 = axes[1].scatter(
        coords_after[:, 0], coords_after[:, 1],
        c=np.arange(len(embeddings_after)),
        cmap='viridis', alpha=0.7, s=100,
        edgecolors='black', linewidths=0.5
    )
    axes[1].set_xlabel('t-SNE Component 1', fontsize=14)
    axes[1].set_ylabel('t-SNE Component 2', fontsize=14)
    axes[1].set_title('After Training (Contrastive Learning)', fontsize=16, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = output_dir / 'embeddings_before_after_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    log.info(f"Saved before/after comparison to {save_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize contrastive learning embeddings')
    parser.add_argument('--cfg', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to visualize')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default='./visualizations',
                        help='Output directory for plots')
    parser.add_argument('--methods', type=str, nargs='+',
                        default=['tsne', 'pca', 'umap'],
                        help='Visualization methods to use')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load config
    log.info(f"Loading config from {args.cfg}")
    cfg = Config.load_from_file(args.cfg)
    
    # Load dataset
    log.info(f"Loading dataset: {cfg.dataset.name}")
    dataset_class = getattr(datasets, cfg.dataset.name)
    dataset = dataset_class(**cfg.dataset)
    split_dataset = dataset.get_split(args.split)
    
    log.info(f"Dataset split '{args.split}': {len(split_dataset)} samples")
    
    # Load model
    log.info(f"Loading model: {cfg.model.name}")
    model_class = getattr(models, cfg.model.name)
    model = model_class(**cfg.model)
    
    # Load checkpoint
    log.info(f"Loading checkpoint from {args.ckpt}")
    checkpoint = torch.load(args.ckpt, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Extract embeddings
    embeddings, labels = extract_embeddings(
        model, split_dataset, 
        device=args.device,
        max_samples=args.max_samples
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save embeddings
    np.save(output_dir / 'embeddings.npy', embeddings)
    np.save(output_dir / 'labels.npy', labels)
    log.info(f"Saved embeddings to {output_dir}")
    
    # Visualize using different methods
    for method in args.methods:
        if method in ['tsne', 'pca', 'umap']:
            visualize_embeddings(embeddings, labels, output_dir, method=method)
        else:
            log.warning(f"Unknown method: {method}, skipping")
    
    # Plot statistics
    plot_embedding_distribution(embeddings, output_dir)
    
    log.info("=" * 50)
    log.info("Visualization complete!")
    log.info(f"Results saved to: {output_dir}")
    log.info("=" * 50)


if __name__ == '__main__':
    main()
