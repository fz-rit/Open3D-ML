#!/usr/bin/env python
"""
Preprocess DigiForest dataset with grid subsampling.

This script:
1. Reads .ply files from the dataset directory
2. Reads corresponding .labels files from the labels directory
3. Applies grid subsampling (default 0.06m grid size)
4. Saves subsampled point clouds as .ply files in subsamplegrid06 subfolder
5. Saves subsampled labels as .labels files in subsamplegrid06 subfolder

Usage:
    python scripts/preprocess_digiforest.py \
        --dataset_path /path/to/ply/files \
        --labels_path /path/to/label/files \
        --grid_size 0.06
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import logging
import csv
import matplotlib.pyplot as plt
from collections import defaultdict

# Add parent directory to path to import ml3d modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import open3d as o3d
from ml3d.datasets.utils import DataProcessing

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def subsample_digiforest(dataset_path, labels_path, grid_size=0.06, output_suffix='grid06'):
    """
    Subsample DigiForest dataset with grid subsampling.
    
    Args:
        dataset_path: Path to directory containing .ply files
        labels_path: Path to directory containing .labels files
        grid_size: Grid size for subsampling (default 0.06m)
        output_suffix: Suffix for output subdirectory name (default 'grid06')
    """
    dataset_root = Path(dataset_path)
    labels_root = Path(labels_path)
    
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_root}")
    if not labels_root.exists():
        raise FileNotFoundError(f"Labels path not found: {labels_root}")
    
    # Create output directories
    output_dataset_dir = dataset_root / f'subsample{output_suffix}'
    output_labels_dir = labels_root / f'subsample{output_suffix}'
    output_dataset_dir.mkdir(exist_ok=True)
    output_labels_dir.mkdir(exist_ok=True)
    
    log.info(f"Dataset input:  {dataset_root}")
    log.info(f"Labels input:   {labels_root}")
    log.info(f"Dataset output: {output_dataset_dir}")
    log.info(f"Labels output:  {output_labels_dir}")
    log.info(f"Grid size:      {grid_size}m")
    
    # Find all .ply files
    ply_files = sorted(dataset_root.glob('*.ply'))
    
    if len(ply_files) == 0:
        raise FileNotFoundError(f"No .ply files found in {dataset_root}")
    
    log.info(f"Found {len(ply_files)} .ply files to process")
    
    # Process each file
    total_original_points = 0
    total_subsampled_points = 0
    
    # Statistics tracking
    file_stats = []  # List of dicts for each file
    original_class_counts = defaultdict(int)  # Total counts across all files
    subsampled_class_counts = defaultdict(int)
    
    for ply_path in tqdm(ply_files, desc="Processing files"):
        stem = ply_path.stem
        
        # Check if label file exists
        label_path = labels_root / f"{stem}.labels"
        if not label_path.exists():
            log.warning(f"Label file not found for {stem}, skipping")
            continue
        
        # Read PLY file
        pcd = o3d.io.read_point_cloud(str(ply_path))
        points = np.asarray(pcd.points, dtype=np.float32)
        
        # Read colors if available
        if pcd.has_colors():
            colors = np.asarray(pcd.colors, dtype=np.float32) * 255.0
        else:
            colors = None
        
        # Read labels
        labels = np.loadtxt(label_path, dtype=np.int32)
        
        # Validate shapes
        if points.shape[0] != labels.shape[0]:
            log.error(f"Shape mismatch for {stem}: points={points.shape[0]}, labels={labels.shape[0]}")
            continue
        
        original_count = points.shape[0]
        total_original_points += original_count
        
        # Count original labels by class
        unique_orig, counts_orig = np.unique(labels, return_counts=True)
        orig_label_dist = dict(zip(unique_orig.tolist(), counts_orig.tolist()))
        
        # Update global counts
        for label, count in orig_label_dist.items():
            original_class_counts[label] += count
        
        # Apply grid subsampling
        if colors is not None:
            sub_points, sub_colors, sub_labels = DataProcessing.grid_subsampling(
                points, features=colors, labels=labels, grid_size=grid_size
            )
        else:
            sub_points, sub_labels = DataProcessing.grid_subsampling(
                points, labels=labels, grid_size=grid_size
            )
            sub_colors = None
        
        subsampled_count = sub_points.shape[0]
        total_subsampled_points += subsampled_count
        
        # Count subsampled labels by class
        unique_sub, counts_sub = np.unique(sub_labels, return_counts=True)
        sub_label_dist = dict(zip(unique_sub.tolist(), counts_sub.tolist()))
        
        # Update global counts
        for label, count in sub_label_dist.items():
            subsampled_class_counts[label] += count
        
        # Store per-file statistics
        file_stat = {
            'filename': stem,
            'original_total': original_count,
            'subsampled_total': subsampled_count,
            'reduction_pct': (1 - subsampled_count / original_count) * 100,
        }
        # Add per-class counts
        all_labels = sorted(set(list(orig_label_dist.keys()) + list(sub_label_dist.keys())))
        for label in all_labels:
            file_stat[f'original_class_{label}'] = orig_label_dist.get(label, 0)
            file_stat[f'subsampled_class_{label}'] = sub_label_dist.get(label, 0)
        
        file_stats.append(file_stat)
        
        # Create output PLY
        output_pcd = o3d.geometry.PointCloud()
        output_pcd.points = o3d.utility.Vector3dVector(sub_points)
        if sub_colors is not None:
            # Convert back to [0, 1] range for PLY format
            output_pcd.colors = o3d.utility.Vector3dVector(sub_colors / 255.0)
        
        # Save subsampled PLY
        output_ply_path = output_dataset_dir / f"{stem}_{output_suffix}.ply"
        o3d.io.write_point_cloud(str(output_ply_path), output_pcd)
        
        # Save subsampled labels
        output_label_path = output_labels_dir / f"{stem}_{output_suffix}.labels"
        np.savetxt(output_label_path, sub_labels.astype(np.int32), fmt='%d')
        
        log.debug(f"{stem}: {original_count:,} → {subsampled_count:,} points "
                 f"({subsampled_count/original_count*100:.1f}%)")
    
    # Export statistics to CSV
    report_path = output_dataset_dir / f'subsampling_report_{output_suffix}.csv'
    all_labels = sorted(set(list(original_class_counts.keys()) + list(subsampled_class_counts.keys())))
    
    with open(report_path, 'w', newline='') as csvfile:
        # Determine all columns
        if file_stats:
            fieldnames = list(file_stats[0].keys())
        else:
            fieldnames = ['filename', 'original_total', 'subsampled_total', 'reduction_pct']
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(file_stats)
        
        # Add summary row
        summary_row = {
            'filename': 'TOTAL',
            'original_total': total_original_points,
            'subsampled_total': total_subsampled_points,
            'reduction_pct': (1 - total_subsampled_points / total_original_points) * 100,
        }
        for label in all_labels:
            summary_row[f'original_class_{label}'] = original_class_counts.get(label, 0)
            summary_row[f'subsampled_class_{label}'] = subsampled_class_counts.get(label, 0)
        writer.writerow(summary_row)
    
    log.info(f"Saved statistics report: {report_path}")
    
    # Generate histogram plots
    label_names = {
        0: 'Unlabeled',
        1: 'Ground',
        2: 'Trunk',
        3: 'Canopy',
        4: 'Understory',
        5: 'Misc'
    }
    
    # Prepare data for plotting (exclude label 0 if present)
    plot_labels = [l for l in all_labels if l != 0]
    orig_counts = [original_class_counts[l] for l in plot_labels]
    sub_counts = [subsampled_class_counts[l] for l in plot_labels]
    label_strings = [f"{l}: {label_names.get(l, 'Unknown')}" for l in plot_labels]
    
    if len(plot_labels) > 0:
        # Create figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Subplot 1: Absolute counts
        x = np.arange(len(plot_labels))
        width = 0.35
        
        axes[0].bar(x - width/2, orig_counts, width, label='Original', alpha=0.8)
        axes[0].bar(x + width/2, sub_counts, width, label='Subsampled', alpha=0.8)
        axes[0].set_xlabel('Class', fontsize=12)
        axes[0].set_ylabel('Number of Points', fontsize=12)
        axes[0].set_title(f'Class Distribution (Grid Size: {grid_size}m)', fontsize=14)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(label_strings, rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        # Format y-axis with comma separators
        axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
        
        # Subplot 2: Percentage distribution
        orig_pcts = np.array(orig_counts) / sum(orig_counts) * 100
        sub_pcts = np.array(sub_counts) / sum(sub_counts) * 100
        
        axes[1].bar(x - width/2, orig_pcts, width, label='Original', alpha=0.8)
        axes[1].bar(x + width/2, sub_pcts, width, label='Subsampled', alpha=0.8)
        axes[1].set_xlabel('Class', fontsize=12)
        axes[1].set_ylabel('Percentage (%)', fontsize=12)
        axes[1].set_title('Class Distribution (Percentage)', fontsize=14)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(label_strings, rotation=45, ha='right')
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dataset_dir / f'subsampling_histogram_{output_suffix}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        log.info(f"Saved histogram plot: {plot_path}")
    
    # Summary with class breakdown
    log.info(f"\n{'='*60}")
    log.info(f"Processing complete!")
    log.info(f"{'='*60}")
    log.info(f"Files processed:      {len(ply_files)}")
    log.info(f"Total original:       {total_original_points:,} points")
    log.info(f"Total subsampled:     {total_subsampled_points:,} points")
    log.info(f"Reduction:            {(1 - total_subsampled_points/total_original_points)*100:.1f}%")
    log.info(f"\nClass Distribution:")
    log.info(f"{'Class':<20} {'Original':>15} {'Subsampled':>15} {'Reduction':>12}")
    log.info(f"{'-'*65}")
    for label in all_labels:
        label_name = label_names.get(label, f'Class {label}')
        orig_count = original_class_counts[label]
        sub_count = subsampled_class_counts[label]
        reduction = (1 - sub_count / orig_count) * 100 if orig_count > 0 else 0
        log.info(f"{label_name:<20} {orig_count:>15,} {sub_count:>15,} {reduction:>11.1f}%")
    log.info(f"\nOutput dataset dir:   {output_dataset_dir}")
    log.info(f"Output labels dir:    {output_labels_dir}")
    log.info(f"Report saved to:      {report_path}")
    if len(plot_labels) > 0:
        log.info(f"Histogram saved to:   {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Subsample DigiForest dataset with grid subsampling'
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        required=True,
        help='Path to directory containing .ply files'
    )
    parser.add_argument(
        '--labels_path',
        type=str,
        required=True,
        help='Path to directory containing .labels files'
    )
    parser.add_argument(
        '--grid_size',
        type=float,
        default=0.06,
        help='Grid size for subsampling in meters (default: 0.06)'
    )
    parser.add_argument(
        '--output_suffix',
        type=str,
        default='grid06',
        help='Suffix for output subdirectory name (default: grid06)'
    )
    
    args = parser.parse_args()
    
    subsample_digiforest(
        dataset_path=args.dataset_path,
        labels_path=args.labels_path,
        grid_size=args.grid_size,
        output_suffix=args.output_suffix
    )


if __name__ == '__main__':
    main()
