"""
Dataset Statistics Utility for Point Cloud Datasets

This module provides utilities to calculate and visualize dataset statistics
to help determine optimal training hyperparameters:
    - num_neighbors: Based on point density distribution
    - batch_size: Based on point count distribution and GPU memory
    - dataloader_iterations_per_epoch: Based on total dataset size and desired coverage

Usage:
    from ml3d.utils.dataset_statistics import DatasetStatistics
    
    # Calculate statistics when loading dataset
    stats = DatasetStatistics(dataset, split='train')
    stats.calculate()
    stats.print_summary()
    stats.plot_histograms(save_path='./logs/dataset_stats.png')
    
    # Get recommended hyperparameters
    recommendations = stats.get_hyperparameter_recommendations(
        target_points_per_window=10240,
        batch_size=16,
        gpu_memory_gb=40,
        coverage_ratio=3.0
    )
"""

import numpy as np
import logging
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)


class DatasetStatistics:
    """Calculate and visualize point cloud dataset statistics."""
    
    def __init__(self, dataset, split='train', sample_limit=None):
        """
        Initialize dataset statistics calculator.
        
        Args:
            dataset: Open3D-ML dataset object
            split: Dataset split to analyze ('train', 'validation', 'test')
            sample_limit: Maximum number of files to sample (None = all files)
        """
        self.dataset = dataset
        self.split = split
        self.sample_limit = sample_limit
        
        # Statistics storage
        self.point_counts = []
        self.class_distributions = defaultdict(int)
        self.file_names = []
        
        # Computed statistics
        self.total_points = 0
        self.mean_points = 0
        self.median_points = 0
        self.std_points = 0
        self.min_points = 0
        self.max_points = 0
        
        log.info(f"Initialized DatasetStatistics for {dataset.name} - {split} split")
    
    def calculate(self, verbose=True, preprocess=None, use_cache=False, cache_dir=None):
        """
        Calculate dataset statistics by iterating through all files.
        Uses cached preprocessed data if available, otherwise applies preprocessing.
        
        Args:
            verbose: Whether to print progress information
            preprocess: Preprocessing function (e.g., model.preprocess for grid subsampling)
            use_cache: Whether to use cached preprocessed data
            cache_dir: Cache directory path
        """
        from ..utils import Cache, get_hash
        
        dataset_split = self.dataset.get_split(self.split)
        n_files = len(dataset_split)
        
        if self.sample_limit and self.sample_limit < n_files:
            n_files = self.sample_limit
            log.info(f"Sampling {n_files} files out of {len(dataset_split)} total")
        
        # Setup cache if requested and preprocessing is provided
        cache_convert = None
        if preprocess is not None and use_cache and cache_dir:
            cache_convert = Cache(preprocess, cache_dir=cache_dir, cache_key=get_hash(repr(preprocess)))
            log.info(f"Using cache from: {cache_dir}")
        elif preprocess is not None:
            log.info("Applying preprocessing (grid subsampling) without cache")
        else:
            log.info("Calculating statistics on raw data (no preprocessing)")
        
        log.info(f"Calculating statistics for {n_files} files...")
        
        for idx in range(n_files):
            try:
                attr = dataset_split.get_attr(idx)
                
                # Get data: try cache first, then preprocess, then raw
                if cache_convert:
                    name = attr['name']
                    if name in cache_convert.cached_ids:
                        data = cache_convert(name)
                    else:
                        data = dataset_split.get_data(idx)
                        data = preprocess(data, attr)
                elif preprocess is not None:
                    data = dataset_split.get_data(idx)
                    data = preprocess(data, attr)
                else:
                    data = dataset_split.get_data(idx)
                
                points = data.get('point', data.get('points', None))
                labels = data.get('label', data.get('labels', None))
                
                if points is None:
                    log.warning(f"No points found in file {idx}")
                    continue
                
                n_points = len(points)
                self.point_counts.append(n_points)
                self.file_names.append(attr.get('name', f'file_{idx}'))
                
                # Calculate class distribution if labels available
                if labels is not None:
                    unique, counts = np.unique(labels, return_counts=True)
                    for label, count in zip(unique, counts):
                        self.class_distributions[int(label)] += int(count)
                
                if verbose and (idx + 1) % max(1, n_files // 10) == 0:
                    log.info(f"Processed {idx + 1}/{n_files} files...")
                    
            except Exception as e:
                log.error(f"Error processing file {idx}: {e}")
                continue
        
        # Compute summary statistics
        if len(self.point_counts) > 0:
            self.point_counts = np.array(self.point_counts)
            self.total_points = np.sum(self.point_counts)
            self.mean_points = np.mean(self.point_counts)
            self.median_points = np.median(self.point_counts)
            self.std_points = np.std(self.point_counts)
            self.min_points = np.min(self.point_counts)
            self.max_points = np.max(self.point_counts)
            
            log.info(f"Statistics calculation complete: {len(self.point_counts)} files, "
                    f"{self.total_points:,} total points")
        else:
            log.warning("No valid point clouds found!")
    
    def print_summary(self):
        """Print comprehensive dataset statistics summary."""
        print("\n" + "=" * 80)
        print(f"DATASET STATISTICS: {self.dataset.name} - {self.split.upper()} SPLIT")
        print("=" * 80)
        
        print(f"\n📊 Point Cloud Statistics:")
        print(f"  Total files:        {len(self.point_counts):,}")
        print(f"  Total points:       {self.total_points:,}")
        print(f"  Mean points/file:   {self.mean_points:,.0f} (± {self.std_points:,.0f})")
        print(f"  Median points/file: {self.median_points:,.0f}")
        print(f"  Min points/file:    {self.min_points:,}")
        print(f"  Max points/file:    {self.max_points:,}")
        
        if len(self.class_distributions) > 0:
            print(f"\n🏷️  Class Distribution:")
            label_to_names = self.dataset.get_label_to_names()
            total_labeled = sum(self.class_distributions.values())
            
            for label, count in sorted(self.class_distributions.items()):
                name = label_to_names.get(label, f"Class_{label}")
                percentage = 100 * count / total_labeled if total_labeled > 0 else 0
                print(f"  {label}: {name:20s} - {count:,} points ({percentage:.2f}%)")
        
        print("\n" + "=" * 80 + "\n")
    
    def get_hyperparameter_recommendations(self, 
                                          target_points_per_window=10240,
                                          batch_size=16,
                                          gpu_memory_gb=40,
                                          coverage_ratio=3.0,
                                          model_type='randlanet'):
        """
        Generate hyperparameter recommendations based on dataset statistics.
        
        Args:
            target_points_per_window: Desired points per local window (e.g., 10240 for RandLANet)
            batch_size: Intended batch size
            gpu_memory_gb: Available GPU memory in GB
            coverage_ratio: Desired dataset coverage per epoch (e.g., 3.0 = 3x dataset)
            model_type: 'randlanet' or 'kpconv' (affects batch calculation)
        
        Returns:
            dict: Recommended hyperparameters
        """
        print("\n" + "=" * 80)
        print("HYPERPARAMETER RECOMMENDATIONS")
        print("=" * 80)
        
        recommendations = {}
        
        # 1. Calculate dataloader_iterations_per_epoch
        total_windows_per_epoch = int(coverage_ratio * self.total_points / target_points_per_window)
        dataloader_iterations = int(np.ceil(total_windows_per_epoch / batch_size))
        
        recommendations['dataloader_iterations_per_epoch_train'] = dataloader_iterations
        recommendations['dataloader_iterations_per_epoch_valid'] = max(1, dataloader_iterations // 8)
        
        print(f"\n📐 Iterations Per Epoch:")
        print(f"  Target coverage:       {coverage_ratio}x dataset = {coverage_ratio * self.total_points:,.0f} points")
        print(f"  Points per window:     {target_points_per_window:,}")
        print(f"  Windows needed:        {total_windows_per_epoch:,}")
        print(f"  Batch size:            {batch_size}")
        print(f"  ➜ Training iterations: {dataloader_iterations:,} iterations/epoch")
        print(f"  ➜ Valid iterations:    {recommendations['dataloader_iterations_per_epoch_valid']:,} iterations/epoch")
        
        # 2. Recommend num_neighbors based on model type
        if model_type.lower() == 'randlanet':
            recommendations['num_neighbors'] = 32
            print(f"\n🔍 Neighborhood Size (RandLANet):")
            print(f"  ➜ num_neighbors:       32 (recommended for efficiency)")
            print(f"     - Original paper uses 16")
            print(f"     - 32 provides good balance")
            print(f"     - 128+ is excessive and slow")
        elif model_type.lower() == 'kpconv':
            # Estimate based on grid size and radius
            grid_size = 0.06  # typical value
            in_radius = 6.0   # typical value
            estimated_points = int((2 * in_radius / grid_size) ** 3)
            recommendations['max_in_points'] = min(estimated_points, 80000)
            print(f"\n🔍 Neighborhood Size (KPConv):")
            print(f"  Grid size:             {grid_size}m")
            print(f"  Input radius:          {in_radius}m")
            print(f"  ➜ max_in_points:       {recommendations['max_in_points']:,}")
        
        # 3. Recommend batch_limit for KPConv
        if model_type.lower() == 'kpconv':
            # Estimate based on GPU memory (rough heuristic)
            if gpu_memory_gb >= 40:
                batch_limit = 1_000_000
            elif gpu_memory_gb >= 24:
                batch_limit = 600_000
            elif gpu_memory_gb >= 16:
                batch_limit = 400_000
            else:
                batch_limit = 200_000
            
            recommendations['batch_limit'] = batch_limit
            print(f"\n💾 Batch Configuration (KPConv):")
            print(f"  GPU memory:            {gpu_memory_gb} GB")
            print(f"  ➜ batch_limit:         {batch_limit:,} points")
            print(f"     (total points across all samples in batch)")
        
        # 4. Estimate training time
        if model_type.lower() == 'randlanet':
            time_per_iteration = 0.5  # seconds (rough estimate)
        else:
            time_per_iteration = 1.0  # KPConv is slower
        
        time_per_epoch = dataloader_iterations * time_per_iteration / 60  # minutes
        print(f"\n⏱️  Estimated Training Time:")
        print(f"  ~{time_per_iteration:.1f}s per iteration")
        print(f"  ~{time_per_epoch:.1f} min per epoch")
        print(f"  ~{time_per_epoch * 100 / 60:.1f} hours for 100 epochs")
        
        print("\n" + "=" * 80 + "\n")
        
        return recommendations
    
    def plot_histograms(self, save_path=None, show=False):
        """
        Plot histograms of dataset statistics.
        
        Args:
            save_path: Path to save the figure (None = don't save)
            show: Whether to display the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{self.dataset.name} - {self.split.upper()} Split Statistics', 
                    fontsize=16, fontweight='bold')
        
        # 1. Point count distribution
        ax = axes[0, 0]
        ax.hist(self.point_counts, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(self.mean_points, color='red', linestyle='--', linewidth=2, label=f'Mean: {self.mean_points:,.0f}')
        ax.axvline(self.median_points, color='orange', linestyle='--', linewidth=2, label=f'Median: {self.median_points:,.0f}')
        ax.set_xlabel('Points per file', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Point Count Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Point count box plot
        ax = axes[0, 1]
        ax.boxplot(self.point_counts, vert=True, patch_artist=True,
                  boxprops=dict(facecolor='lightblue', alpha=0.7),
                  medianprops=dict(color='red', linewidth=2))
        ax.set_ylabel('Points per file', fontsize=12)
        ax.set_title('Point Count Box Plot', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Class distribution (if available)
        ax = axes[1, 0]
        if len(self.class_distributions) > 0:
            labels = sorted(self.class_distributions.keys())
            counts = [self.class_distributions[l] for l in labels]
            label_names = self.dataset.get_label_to_names()
            names = [label_names.get(l, f'Class {l}') for l in labels]
            
            bars = ax.bar(range(len(labels)), counts, color='forestgreen', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Class Label', fontsize=12)
            ax.set_ylabel('Total Points', fontsize=12)
            ax.set_title('Class Distribution', fontsize=14, fontweight='bold')
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(names, rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add count labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{count:,.0f}',
                       ha='center', va='bottom', fontsize=9)
        else:
            ax.text(0.5, 0.5, 'No label information available', 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
        
        # 4. Cumulative point count
        ax = axes[1, 1]
        sorted_counts = np.sort(self.point_counts)
        cumsum = np.cumsum(sorted_counts)
        ax.plot(range(len(sorted_counts)), cumsum / 1e6, linewidth=2, color='purple')
        ax.fill_between(range(len(sorted_counts)), cumsum / 1e6, alpha=0.3, color='purple')
        ax.set_xlabel('File index (sorted by size)', fontsize=12)
        ax.set_ylabel('Cumulative points (millions)', fontsize=12)
        ax.set_title('Cumulative Point Count', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add total annotation
        ax.text(0.95, 0.05, f'Total: {self.total_points / 1e6:.1f}M points',
               transform=ax.transAxes, ha='right', va='bottom',
               fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            log.info(f"Saved histogram to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig


def calculate_dataset_statistics(dataset, split='train', save_dir=None, 
                                 sample_limit=None, model_type='randlanet',
                                 model=None, use_cache=False,
                                 **hyperparameter_kwargs):
    """
    Convenience function to calculate and report dataset statistics.
    Uses cached preprocessed data if available, otherwise applies preprocessing.
    
    Args:
        dataset: Open3D-ML dataset object
        split: Dataset split to analyze
        save_dir: Directory to save outputs (None = don't save)
        sample_limit: Maximum files to sample (None = all)
        model_type: 'randlanet' or 'kpconv'
        model: Model instance with preprocess function (optional, for accurate statistics)
        use_cache: Whether to use cached preprocessed data
        **hyperparameter_kwargs: Arguments for get_hyperparameter_recommendations()
    
    Returns:
        tuple: (stats object, recommendations dict)
    """
    stats = DatasetStatistics(dataset, split=split, sample_limit=sample_limit)
    
    # Setup preprocessing and cache
    preprocess = model.preprocess if model is not None else None
    cache_dir = dataset.cfg.get('cache_dir') if use_cache else None
    
    stats.calculate(
        preprocess=preprocess,
        use_cache=use_cache,
        cache_dir=cache_dir
    )
    stats.print_summary()
    
    recommendations = stats.get_hyperparameter_recommendations(
        model_type=model_type,
        **hyperparameter_kwargs
    )
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save histogram
        hist_path = save_dir / f'{dataset.name}_{split}_statistics.png'
        stats.plot_histograms(save_path=hist_path)
        
        # Save text summary
        import json
        summary_path = save_dir / f'{dataset.name}_{split}_summary.json'
        summary = {
            'dataset': dataset.name,
            'split': split,
            'n_files': len(stats.point_counts),
            'total_points': int(stats.total_points),
            'mean_points': float(stats.mean_points),
            'median_points': float(stats.median_points),
            'std_points': float(stats.std_points),
            'min_points': int(stats.min_points),
            'max_points': int(stats.max_points),
            'class_distribution': dict(stats.class_distributions),
            'recommendations': recommendations
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        log.info(f"Saved summary to {summary_path}")
    
    return stats, recommendations
