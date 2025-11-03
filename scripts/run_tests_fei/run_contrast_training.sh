#!/bin/bash
# Convenience script to run contrastive learning training with conda environment
# Usage: bash run_contrast_training.sh [OPTIONS]

set -e

# Default values
CONFIG="ml3d/configs/randlanet_harvardforest_contrast.yml"
DEVICE="cuda"
CONDA_ENV="pcd_seg_open3d_env"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cfg)
            CONFIG="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --env)
            CONDA_ENV="$2"
            shift 2
            ;;
        *)
            # Pass through other arguments
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

echo "=========================================="
echo "PointContrast Training Script"
echo "=========================================="
echo "Conda Environment: $CONDA_ENV"
echo "Config: $CONFIG"
echo "Device: $DEVICE"
echo "Extra Args: $EXTRA_ARGS"
echo "=========================================="

# Run with conda
conda run -n "$CONDA_ENV" python scripts/run_tests_fei/train_harvardforest_contrast.py \
    --cfg "$CONFIG" \
    --device "$DEVICE" \
    $EXTRA_ARGS
