# Decoder Class Distribution Monitoring

## Overview

Added decoder feature monitoring to visualize **class distribution quality** in feature space, complementing the existing encoder domain alignment monitoring.

## Conceptual Separation

### Encoder Features → Domain Alignment
- **Purpose**: Visualize domain discrepancy (source vs target)
- **Features**: Spatially pooled/downsampled (81920 → 5120 → 320)
- **Visualization**: Domain separation only (no class labels)
- **Output**: `domain_monitoring/`

### Decoder Features → Class Distribution
- **Purpose**: Visualize class separability and confusion
- **Features**: Upsampled back to point-level with semantic information
- **Visualization**: Per-class coloring with t-SNE/UMAP
- **Output**: `decoder_class_monitoring/`

## Implementation Details

### 1. Model Modifications (`randlanet_da.py`)
- Added `return_decoder_features` parameter to `forward()`
- Decoder now collects intermediate features from all 4 decoder layers
- Features flattened to (B*N, C) for visualization

### 2. Decoder Class Monitor (`decoder_class_monitoring.py`)
- Extracts decoder features with ground-truth labels
- Generates t-SNE/UMAP plots with per-class coloring
- Visualizes all decoder layers separately
- Subsamples to 5000 points per domain for speed

### 3. Pipeline Integration (`domain_adaptation_semseg.py`)
- Target validation enabled by default (`num_target_validate_batch=10`)
- Decoder monitoring runs every 10 epochs (configurable via `decoder_monitor_freq`)
- Uses validation data for both source and target (ground truth available)

## Configuration

Add to your config YAML:

```yaml
pipeline:
  num_target_validate_batch: 10      # Number of target batches for validation
  decoder_monitor_freq: 10           # Decoder monitoring frequency (epochs)
  monitor_freq: 5                    # Encoder monitoring frequency (epochs)
  enable_tsne: True
  enable_umap: True
```

## Output Structure

```
train_log/00XXX_RandLANet_DA_torch/
├── domain_monitoring/              # Encoder: domain alignment
│   ├── epoch_0000/
│   │   ├── tsne.png               # Domain separation (source vs target)
│   │   ├── umap.png
│   │   ├── layer_alignment.png
│   │   └── ...
│   └── ...
├── decoder_class_monitoring/       # Decoder: class distribution
│   ├── epoch_0000/
│   │   ├── decoder_layer_0/
│   │   │   ├── tsne_classes.png   # Per-class coloring
│   │   │   ├── umap_classes.png
│   │   │   └── feature_distributions.png
│   │   ├── decoder_layer_1/
│   │   ├── decoder_layer_2/
│   │   └── decoder_layer_3/
│   └── ...
└── ...
```

## Interpretation Guide

### Encoder Monitoring (Domain Alignment)
- **Good**: Source and target points overlap in t-SNE/UMAP
- **Bad**: Clear separation between blue (source) and red (target)
- **Metrics**: MMD, covariance distance, A-distance should decrease

### Decoder Monitoring (Class Distribution)
- **Good**: Same class from different domains cluster together
- **Bad**: Class confusion (overlapping different class colors)
- **Patterns**:
  - Layer 0 (earliest): More spatial/geometric features
  - Layer 3 (latest): More semantic/class features
  - Target classes should align with source classes by color

## What to Look For

1. **Class Separability**: Different colors (classes) should form distinct clusters
2. **Domain Consistency**: Same class in source (circles) and target (triangles) should overlap
3. **Class Confusion**: Overlapping colors indicate classes that are hard to distinguish
4. **Adaptation Quality**: Target domain class distributions approaching source domain

## Benefits

- **Diagnose class-specific issues**: Identify which classes adapt poorly
- **Validate semantic alignment**: Ensure adaptation works at class level, not just domain level
- **Layer-wise analysis**: Understand how semantic information flows through decoder
- **Ground truth comparison**: Use actual labels to validate feature quality

## Next Steps

After training completes:
1. Check encoder monitoring → Is domain alignment improving?
2. Check decoder monitoring → Are classes well-separated?
3. Compare early vs late epochs → Is adaptation progressing?
4. Identify problematic classes → Focus on confused classes in next iteration
