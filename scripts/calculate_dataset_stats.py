#!/usr/bin/env python
"""
Calculate Dataset Statistics and Hyperparameter Recommendations

This script analyzes a dataset and provides:
1. Point cloud statistics (count, distribution, etc.)
2. Visualizations (histograms, box plots, class distributions)
3. Hyperparameter recommendations (dataloader_iterations, num_neighbors, batch_size)

Usage:
    # Analyze RandLANet dataset
    python scripts/calculate_dataset_stats.py \
        --config ml3d/configs/randlanet_semantic3dunified_xyz_4class.yml \
        --model-type randlanet \
        --batch-size 16 \
        --coverage-ratio 3.0 \
        --gpu-memory 40
    
    # Analyze KPConv dataset
    python scripts/calculate_dataset_stats.py \\
        --config ml3d/configs/kpconv_semantic3dunified_xyz_4class.yml \\
        --model-type kpconv \\
        --batch-size 24 \\
        --coverage-ratio 5.0 \\
        --gpu-memory 40
    
    # Quick analysis (sample first 10 files only)
    python scripts/calculate_dataset_stats.py \\
        --config ml3d/configs/randlanet_semantic3dunified_xyz_4class.yml \\
        --sample-limit 10 \\
        --splits train validation

Output:
    - Terminal: Comprehensive statistics and recommendations
    - {output_dir}/{dataset}_{split}_statistics.png: Visualization histograms
    - {output_dir}/{dataset}_{split}_summary.json: JSON summary with recommendations
"""

import argparse
import sys
import logging
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import ml3d.utils as utils
import ml3d.datasets as datasets
import ml3d.torch.models as models
from ml3d.utils import calculate_dataset_statistics

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Calculate dataset statistics and hyperparameter recommendations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument('--config', type=str, required=True,
                       help='Path to dataset configuration YAML file')
    
    # Model configuration
    parser.add_argument('--model-type', type=str, default='randlanet',
                       choices=['randlanet', 'kpconv'],
                       help='Model type for hyperparameter recommendations')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Intended batch size for training')
    parser.add_argument('--num-points', type=int, default=None,
                       help='Points per window (default: 10240 for RandLANet, computed for KPConv)')
    parser.add_argument('--coverage-ratio', type=float, default=3.0,
                       help='Target dataset coverage per epoch (e.g., 3.0 = 3x dataset)')
    parser.add_argument('--gpu-memory', type=int, default=40,
                       help='Available GPU memory in GB')
    parser.add_argument('--use-cache', action='store_true',
                       help='Use cached preprocessed data if available')
    
    # Analysis options
    parser.add_argument('--splits', nargs='+', default=['train'],
                       help='Dataset splits to analyze (default: train)')
    parser.add_argument('--sample-limit', type=int, default=None,
                       help='Maximum files to sample per split (None = all files)')
    parser.add_argument('--output-dir', type=str, default='./logs/dataset_statistics',
                       help='Directory to save output visualizations and summaries')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save visualizations and JSON summary')
    
    args = parser.parse_args()
    
    # Validate config file
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    # Load configuration
    log.info(f"Loading configuration from: {args.config}")
    cfg = utils.Config.load_from_file(args.config)
    
    # Validate required config sections
    if not hasattr(cfg, 'dataset') or not cfg.dataset:
        raise ValueError("Config must contain 'dataset' section")
    
    dataset_cfg = cfg.dataset
    if 'name' not in dataset_cfg or 'dataset_path' not in dataset_cfg:
        raise ValueError("Config dataset section must contain 'name' and 'dataset_path'")
    
    # Check dataset path exists
    dataset_path = Path(dataset_cfg['dataset_path'])
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    
    # Initialize dataset
    log.info(f"Initializing dataset: {dataset_cfg['name']}")
    dataset_class = getattr(datasets, dataset_cfg['name'])
    # Pass dataset_path as positional arg, rest as kwargs (excluding 'name')
    dataset_kwargs = {k: v for k, v in dataset_cfg.items() if k not in ['name', 'dataset_path']}
    dataset = dataset_class(dataset_cfg['dataset_path'], **dataset_kwargs)
    
    # Initialize model for preprocessing
    log.info(f"Initializing model for preprocessing: {cfg.model.get('name', args.model_type)}")
    try:
        if hasattr(cfg, 'model') and 'name' in cfg.model:
            model_name = cfg.model['name']
        else:
            model_name = 'RandLANet' if args.model_type == 'randlanet' else 'KPFCNN'
        
        model_class = getattr(models, model_name)
        model = model_class(**cfg.model)
        log.info(f"Model initialized: {model_name}")
    except Exception as e:
        log.warning(f"Could not initialize model: {e}")
        log.warning("Statistics will be calculated on raw data without preprocessing")
        model = None
    
    log.info(f"Dataset: {dataset.name}")
    log.info(f"Dataset path: {dataset_cfg['dataset_path']}")
    log.info(f"Use cache: {args.use_cache}")
    if args.use_cache:
        log.info(f"Cache dir: {dataset_cfg.get('cache_dir', 'Not specified')}")
    log.info(f"Splits to analyze: {args.splits}")
    if args.sample_limit:
        log.info(f"Sampling limit: {args.sample_limit} files per split")
    
    # Determine num_points
    if args.num_points is None:
        if args.model_type == 'randlanet':
            num_points = 10240  # Default for RandLANet
        else:  # kpconv
            # Get from model config if available
            num_points = cfg.model.get('max_in_points', 40960)
    else:
        num_points = args.num_points
    
    log.info(f"\nHyperparameter calculation settings:")
    log.info(f"  Model type: {args.model_type}")
    log.info(f"  Batch size: {args.batch_size}")
    log.info(f"  Points per window: {num_points:,}")
    log.info(f"  Coverage ratio: {args.coverage_ratio}x")
    log.info(f"  GPU memory: {args.gpu_memory} GB")
    
    # Calculate statistics for each split
    all_recommendations = {}
    
    for split in args.splits:
        log.info(f"\n{'=' * 80}")
        log.info(f"ANALYZING SPLIT: {split.upper()}")
        log.info(f"{'=' * 80}\n")
        
        try:
            save_dir = None if args.no_save else args.output_dir
            
            stats, recommendations = calculate_dataset_statistics(
                dataset=dataset,
                split=split,
                save_dir=save_dir,
                sample_limit=args.sample_limit,
                model_type=args.model_type,
                model=model,
                use_cache=args.use_cache,
                target_points_per_window=num_points,
                batch_size=args.batch_size,
                gpu_memory_gb=args.gpu_memory,
                coverage_ratio=args.coverage_ratio
            )
            
            all_recommendations[split] = recommendations
            
        except Exception as e:
            log.error(f"Failed to analyze {split} split: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary of recommendations
    if len(all_recommendations) > 0:
        print("\n" + "=" * 80)
        print("CONFIGURATION SUMMARY")
        print("=" * 80)
        print("\nAdd these parameters to your config file:\n")
        print("dataset:")
        
        for split, recs in all_recommendations.items():
            if split == 'train' or split == 'training':
                print(f"  dataloader_iterations_per_epoch_train: {recs.get('dataloader_iterations_per_epoch_train', 'N/A')}")
            elif split in ['validation', 'val']:
                print(f"  dataloader_iterations_per_epoch_valid: {recs.get('dataloader_iterations_per_epoch_valid', 'N/A')}")
        
        print("\nmodel:")
        if args.model_type == 'randlanet':
            print(f"  num_neighbors: {all_recommendations[args.splits[0]].get('num_neighbors', 32)}")
            print(f"  num_points: {num_points}")
        else:  # kpconv
            print(f"  max_in_points: {all_recommendations[args.splits[0]].get('max_in_points', 40960)}")
            print(f"  batch_limit: {all_recommendations[args.splits[0]].get('batch_limit', 1000000)}")
        
        print("\npipeline:")
        print(f"  batch_size: {args.batch_size}")
        
        print("\n" + "=" * 80)
        
        if not args.no_save:
            print(f"\n✓ Statistics saved to: {args.output_dir}")
            print(f"  - PNG visualizations: {dataset.name}_{{split}}_statistics.png")
            print(f"  - JSON summaries: {dataset.name}_{{split}}_summary.json")
    
    log.info("\n✓ Analysis complete!")


if __name__ == '__main__':
    main()
