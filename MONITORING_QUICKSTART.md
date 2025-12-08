# Domain Adaptation Monitoring System - Quick Start

## What Was Created

### Core Monitoring Modules

1. **`ml3d/torch/modules/metrics/domain_metrics.py`**
   - `compute_mmd()` - Maximum Mean Discrepancy with RBF/linear kernels
   - `compute_covariance_distance()` - Geodesic distance on SPD manifold
   - `compute_a_distance()` - Domain separability metric
   - `compute_layer_alignment_quality()` - Per-layer analysis
   - `DomainMetricsTracker` - Track metrics over training

2. **`ml3d/torch/modules/metrics/domain_visualizations.py`**
   - `plot_tsne()` - t-SNE 2D projections
   - `plot_umap()` - UMAP 2D projections  
   - `plot_feature_distributions()` - Histogram comparisons
   - `plot_covariance_matrices()` - Covariance heatmaps
   - `plot_layer_alignment_progress()` - Per-layer metrics over time
   - `plot_training_metrics()` - Overall training progress

3. **`ml3d/torch/modules/metrics/domain_report.py`**
   - `generate_html_report()` - Interactive HTML report with embedded images
   - `save_metrics_json()` - Raw metrics in JSON format
   - Includes recommendations and color-coded status

### Integration

4. **`ml3d/torch/pipelines/domain_adaptation_semseg.py`** (Modified)
   - Added `run_domain_monitoring()` method
   - Added `_extract_features()` helper
   - Automatic monitoring every `monitor_freq` epochs
   - Saves to `train_log/.../domain_monitoring/`

5. **`scripts/monitor_domain_adaptation.py`** (New)
   - Standalone script for post-training analysis
   - Can process single checkpoint or entire directory
   - Same visualizations without retraining

### Configuration

6. **All 6 config files updated:**
   - `randlanet_da_semantic3d_to_digiforest_{debug,pilot,full}.yml`
   - `randlanet_da_semantic3d_to_forest_{debug,pilot,full}.yml`
   - Added `monitor_freq` parameter (1 for debug, 5 for pilot, 10 for full)

### Documentation

7. **`DOMAIN_ADAPTATION_MONITORING.md`** - Complete user guide
8. **`DOMAIN_ADAPTATION_QUICKREF.md`** - Quick reference for config decisions

## Quick Commands

### Training with Monitoring

```bash
# Debug (monitor every epoch)
python scripts/train_domain_adaptation.py \
    --config ml3d/configs/domain-adaptation-configs/randlanet_da_semantic3d_to_digiforest_debug.yml

# Pilot (monitor every 5 epochs)
python scripts/train_domain_adaptation.py \
    --config ml3d/configs/domain-adaptation-configs/randlanet_da_semantic3d_to_digiforest_pilot.yml

# Full (monitor every 10 epochs)
python scripts/train_domain_adaptation.py \
    --config ml3d/configs/domain-adaptation-configs/randlanet_da_semantic3d_to_digiforest_full.yml
```

### Standalone Monitoring

```bash
# Analyze single checkpoint
python scripts/monitor_domain_adaptation.py \
    --config ml3d/configs/domain-adaptation-configs/randlanet_da_semantic3d_to_digiforest_pilot.yml \
    --checkpoint train_log/00001_RandLANetDA_.../checkpoints/ckpt_00020.pth \
    --output_dir ./analysis_epoch20

# Analyze all checkpoints
python scripts/monitor_domain_adaptation.py \
    --config ml3d/configs/domain-adaptation-configs/randlanet_da_semantic3d_to_digiforest_pilot.yml \
    --ckpt_dir train_log/00001_RandLANetDA_.../checkpoints/ \
    --output_dir ./analysis_all
```

## Output Structure

```
train_log/
└── 00001_RandLANetDA_Semantic3DUnified_to_DigiForestUnified_torch/
    ├── checkpoints/           # Model checkpoints (existing)
    │   ├── ckpt_00005.pth
    │   └── ckpt_00010.pth
    └── domain_monitoring/     # NEW: Monitoring outputs (parallel)
        ├── epoch_0000/
        │   ├── report.html    # ← Open this in browser
        │   ├── metrics.json
        │   ├── tsne.png
        │   ├── umap.png
        │   ├── feature_distributions.png
        │   ├── covariance_matrices.png
        │   ├── layer_alignment.png
        │   └── training_metrics.png
        ├── epoch_0005/
        └── epoch_0010/
```

## What to Look For

### In HTML Report

1. **Executive Summary** - Overall status and recommendations
2. **MMD & Covariance Distance** - Should decrease over training
3. **t-SNE/UMAP Plots** - Domains should mix (not separate clusters)
4. **Layer-wise Table** - Which layers align well vs struggle
5. **Training Progress** - Convergence trends

### Good vs Bad Alignment

| Visualization | ✅ Good | ❌ Poor |
|---------------|---------|---------|
| **t-SNE/UMAP** | Source & target well-mixed | Clear separation |
| **MMD** | < 0.05 (decreasing) | > 0.3 (not improving) |
| **Cov Distance** | < 0.1 | > 0.5 |
| **Feature Dists** | Overlapping histograms | Completely different |

## Metrics Interpretation

- **MMD (Maximum Mean Discrepancy)**: Distribution distance. Lower = better alignment
- **Covariance Distance**: Second-order alignment (CORAL objective). Lower = better
- **A-Distance**: Domain separability. Lower = more similar domains
- **Source Val IoU**: Should maintain (>80% of baseline)
- **Target Val IoU**: Should improve over training

## Troubleshooting

### Monitoring is slow
- Reduce `monitor_freq` (e.g., 20 instead of 5)
- Reduce `max_batches` in feature extraction (default: 20)

### Out of memory
- Reduce `max_batches` to 10 or 5
- Monitor less frequently

### UMAP not available
```bash
pip install umap-learn
```

### Plots look strange
- Wait for more epochs (features not learned yet)
- Check if datasets load correctly
- Increase `max_batches` for more samples

## Workflow Recommendation

1. **Debug phase (3 epochs)**: Verify monitoring works, check for errors
2. **Pilot phase (20 epochs)**: Analyze every 5th epoch, tune hyperparameters
3. **Full training (150 epochs)**: Monitor every 10th epoch, find best checkpoint
4. **Post-training**: Run standalone script on all checkpoints for comparison

## Dependencies

Install if missing:
```bash
pip install umap-learn seaborn
```

All other dependencies (torch, numpy, matplotlib, scikit-learn) are already in requirements.

## Key Features

✅ **Automatic** - Runs during training without manual intervention
✅ **Comprehensive** - Multiple metrics and visualizations
✅ **Interpretable** - HTML reports with recommendations
✅ **Flexible** - Standalone script for post-hoc analysis
✅ **Efficient** - Configurable frequency and sampling
✅ **Parallel** - Saved alongside checkpoints for easy access

## Next Steps

1. Run debug config to test monitoring system
2. Review first HTML report to understand output format
3. Run pilot experiment, check reports every 5 epochs
4. Tune based on monitoring insights
5. Full training with final config
6. Use standalone script to compare all checkpoints and select best

Refer to `DOMAIN_ADAPTATION_MONITORING.md` for detailed documentation!
