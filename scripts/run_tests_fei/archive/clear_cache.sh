#!/bin/bash
# Clear the cache directory to force reloading data with new label format

echo "Clearing cache directory..."
rm -rf ./logs/cache_mangrove3d/*
echo "Cache cleared!"
echo ""
echo "Now run your training script:"
echo "CUDA_LAUNCH_BLOCKING=1 python train_randlanet_mangrove3d_debug.py"
