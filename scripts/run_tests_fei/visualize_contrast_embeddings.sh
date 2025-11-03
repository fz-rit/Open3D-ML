#!/bin/bash
# Visualize learned contrastive embeddings

# Configuration
CONFIG="ml3d/configs/randlanet_harvardforest_contrast.yml"
CHECKPOINT="logs/checkpoint/ckpt_best.pth"
OUTPUT_DIR="./visualizations/randlanet_contrast"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT"
    echo "Please train the model first or specify a different checkpoint."
    exit 1
fi

echo "=============================================="
echo "Visualizing Contrastive Learning Embeddings"
echo "=============================================="
echo "Config: $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "Output: $OUTPUT_DIR"
echo ""

# Note: Run this script from within the pcd_seg_open3d_env conda environment

# Install visualization dependencies if needed
echo "Checking visualization dependencies..."
pip install -q scikit-learn matplotlib 2>/dev/null || true

# Optional: Install UMAP for better visualizations
# Uncomment the line below if you want UMAP visualizations
# pip install -q umap-learn

# Run visualization
python scripts/visualize_embeddings.py \
    --cfg "$CONFIG" \
    --ckpt "$CHECKPOINT" \
    --split train \
    --max_samples 100 \
    --output_dir "$OUTPUT_DIR" \
    --methods tsne pca \
    --device cuda

echo ""
echo "=============================================="
echo "Visualization complete!"
echo "Check results in: $OUTPUT_DIR"
echo "=============================================="
echo ""
echo "Generated plots:"
echo "  - embeddings_tsne.png : t-SNE projection"
echo "  - embeddings_pca.png  : PCA projection"
echo "  - embedding_statistics.png : Distribution analysis"
echo ""
