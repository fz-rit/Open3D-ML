# Domain Adaptation Monitoring System

Comprehensive monitoring tools for analyzing feature alignment quality during domain adaptation training.

## Features

✅ **Automatic Monitoring During Training**
- Runs periodically every N epochs (configurable via `monitor_freq`)
- Extracts features from both domains
- Computes alignment metrics (MMD, covariance distance, A-distance)
- Generates visualizations (t-SNE, UMAP, distributions)
- Creates HTML reports with recommendations

✅ **Standalone Monitoring Script**
- Analyze trained checkpoints post-hoc
- Compare multiple checkpoints
- Generate reports without retraining

✅ **Comprehensive Metrics**
- **MMD (Maximum Mean Discrepancy)**: Measures distribution difference
- **Covariance Distance**: CORAL alignment quality (geodesic distance on SPD manifold)
- **A-Distance**: Domain separability via linear classifier
- **Per-layer Alignment**: Track which layers align well vs struggle

✅ **Rich Visualizations**
- **t-SNE & UMAP**: 2D projections showing domain mixing
- **Feature Distributions**: Histogram comparisons across dimensions
- **Covariance Matrices**: Visual comparison of second-order statistics
- **Training Progress**: Metrics over epochs
- **Layer-wise Analysis**: Alignment quality per encoder layer

## Directory Structure

After training with monitoring enabled, you'll find:

```
train_log/
└── 00001_RandLANetDA_Semantic3DUnified_to_DigiForestUnified_torch/
    ├── checkpoints/
    │   ├── ckpt_00005.pth
    │   ├── ckpt_00010.pth
    │   └── ...
    └── domain_monitoring/          # ← Monitoring outputs parallel to checkpoints
        ├── epoch_0000/
        │   ├── report.html         # Interactive HTML report
        │   ├── metrics.json        # Raw metrics data
        │   ├── tsne.png
        │   ├── umap.png
        │   ├── feature_distributions.png
        │   ├── covariance_matrices.png
        │   ├── layer_alignment.png
        │   └── training_metrics.png
        ├── epoch_0005/
        │   └── ...
        └── epoch_0010/
            └── ...
```

## Usage

### 1. Automatic Monitoring (During Training)

Add `monitor_freq` to your config file:

```yaml
pipeline:
  # ... other parameters ...
  monitor_freq: 5  # Run monitoring every 5 epochs
```

Then train normally:

```bash
python scripts/train_domain_adaptation.py --config your_config.yml
```

**Recommended frequencies:**
- **Debug**: `monitor_freq: 1` (every epoch, ~2-3 min overhead)
- **Pilot**: `monitor_freq: 5` (every 5 epochs, ~3-5 min overhead)
- **Full**: `monitor_freq: 10` (every 10 epochs, ~3-5 min overhead)

### 2. Standalone Monitoring (Post-training)

Analyze a single checkpoint:

```bash
python scripts/monitor_domain_adaptation.py \
    --config ml3d/configs/domain-adaptation-configs/randlanet_da_semantic3d_to_digiforest_pilot.yml \
    --checkpoint train_log/.../checkpoints/ckpt_00020.pth \
    --output_dir ./analysis_epoch20
```

Analyze all checkpoints in a directory:

```bash
python scripts/monitor_domain_adaptation.py \
    --config ml3d/configs/domain-adaptation-configs/randlanet_da_semantic3d_to_digiforest_pilot.yml \
    --ckpt_dir train_log/.../checkpoints/ \
    --output_dir ./analysis_all
```

## Interpreting Results

### HTML Report

Open `epoch_XXXX/report.html` in a browser to see:

1. **Executive Summary**
   - Overall alignment status (Excellent/Good/Fair/Needs Improvement)
   - Key metrics at a glance
   - Actionable recommendations

2. **Key Metrics Cards**
   - MMD, Covariance Distance, A-Distance
   - Source and Target validation IoU
   - Gap reduction percentage

3. **Feature Space Visualizations**
   - t-SNE and UMAP plots showing domain mixing
   - Feature distribution histograms
   - Covariance matrix comparisons

4. **Layer-wise Analysis Table**
   - Alignment quality per encoder layer
   - Identifies which layers need attention

5. **Training Progress Plots**
   - Metrics over time
   - Convergence trends

### Metrics Interpretation

| Metric | Good Range | Interpretation |
|--------|------------|----------------|
| **MMD (RBF)** | < 0.05 | Lower = better alignment. >0.3 = poor |
| **Cov Distance** | < 0.1 | Geodesic distance between covariances |
| **A-Distance** | < 0.5 | Domain separability. Lower = more similar |
| **Source Val IoU** | Maintain >80% baseline | Should not drop significantly |
| **Target Val IoU** | Improve from baseline | Should increase over training |

### Visual Cues

**t-SNE/UMAP Plots:**
- ✅ **Good**: Source (blue ○) and target (red △) are well-mixed
- ⚠️ **Fair**: Some mixing but visible clusters
- ❌ **Poor**: Clear separation between domains

**Feature Distributions:**
- ✅ **Good**: Source and target histograms overlap
- ⚠️ **Fair**: Similar shapes but shifted
- ❌ **Poor**: Completely different distributions

**Covariance Matrices:**
- ✅ **Good**: Source and target look similar, difference is dim
- ⚠️ **Fair**: Some structure differences
- ❌ **Poor**: Bright differences, very different patterns

## Troubleshooting

### Issue: Monitoring is too slow

**Solution:**
```yaml
pipeline:
  monitor_freq: 20  # Increase interval
```

Or in standalone script, reduce samples:
```python
# In monitor_domain_adaptation.py, line ~285
max_batches=10  # Default is 20
```

### Issue: Out of memory during monitoring

**Solution:**
- Reduce `max_batches` in feature extraction (default: 20)
- Monitor less frequently
- Use smaller batch_size during training

### Issue: UMAP not available

**Solution:**
```bash
pip install umap-learn
```

t-SNE will still work (uses scikit-learn).

### Issue: Plots look strange

**Possible causes:**
1. **Too few samples**: Increase `max_batches` in extraction
2. **Early training**: Features not learned yet, wait for more epochs
3. **Data issues**: Check if datasets load correctly

## Monitoring Workflow Example

```bash
# 1. Start with DEBUG config to verify everything works
python scripts/train_domain_adaptation.py \
    --config ml3d/configs/domain-adaptation-configs/randlanet_da_semantic3d_to_digiforest_debug.yml

# Check: train_log/.../domain_monitoring/epoch_0000/report.html
# - Verify plots are generated
# - Check if features are extracting correctly

# 2. Run PILOT experiment with monitoring every 5 epochs
python scripts/train_domain_adaptation.py \
    --config ml3d/configs/domain-adaptation-configs/randlanet_da_semantic3d_to_digiforest_pilot.yml

# After 20 epochs, review:
# - Is MMD decreasing? (Good sign)
# - Are t-SNE plots showing mixing? (Good sign)
# - Is source IoU maintained? (Critical)
# - Is target IoU improving? (Goal)

# 3. Based on pilot, tune hyperparameters and run FULL training
# Edit config: adjust coral_weight, alignment_layers based on pilot results
python scripts/train_domain_adaptation.py \
    --config ml3d/configs/domain-adaptation-configs/randlanet_da_semantic3d_to_digiforest_full.yml

# 4. Post-training analysis of all checkpoints
python scripts/monitor_domain_adaptation.py \
    --config ml3d/configs/domain-adaptation-configs/randlanet_da_semantic3d_to_digiforest_full.yml \
    --ckpt_dir train_log/.../checkpoints/ \
    --output_dir ./full_analysis

# Compare reports across epochs to find best checkpoint
```

## Technical Details

### Feature Extraction

Features are extracted from model's intermediate layers specified in `alignment_layers`:

```python
# In model forward pass
logits, features = model(inputs, return_intermediate_features=True)
# features = [layer_0_features, layer_2_features, layer_4_features]
```

For RandLANet (5 layers total):
- Layer 0: Early geometric features (16 dims)
- Layer 2: Mid-level structures (128 dims)
- Layer 4: High-level semantics (512 dims)

### Metric Computation

**MMD (RBF Kernel):**
```python
MMD² = E[K(s,s)] + E[K(t,t)] - 2E[K(s,t)]
K(x,y) = exp(-||x-y||²/(2σ²))
```
Bandwidth σ computed via median heuristic.

**Covariance Distance (Geodesic):**
```python
d(Cs, Ct) = ||log(Cs^{-1/2} Ct Cs^{-1/2})||_F / (4d²)
```
This is the geodesic distance on the SPD manifold (same as CORAL loss).

**A-Distance:**
```python
Train binary classifier to distinguish source vs target
A-distance ≈ 2(1 - 2*error)
```
Proxy for domain discrepancy (0 = identical, 2 = completely different).

### Visualization Sampling

To keep monitoring fast, features are subsampled:
- Max 2000 points per domain for t-SNE/UMAP
- Max 20 batches for feature extraction
- Adjust in code if needed

## Dependencies

Required:
- `torch`
- `numpy`
- `matplotlib`
- `scikit-learn` (for t-SNE, A-distance)

Optional:
- `umap-learn` (for UMAP plots)
- `seaborn` (for prettier plots, falls back to matplotlib)

Install all:
```bash
pip install torch numpy matplotlib scikit-learn umap-learn seaborn
```

## FAQ

**Q: How much overhead does monitoring add?**
A: ~2-5 minutes per monitoring run, depending on `max_batches` and visualization complexity.

**Q: Can I disable specific visualizations?**
A: Yes, edit `run_domain_monitoring()` in `domain_adaptation_semseg.py` and comment out unwanted plots.

**Q: Can I monitor during inference/testing?**
A: Yes, use the standalone script with test checkpoints.

**Q: What if alignment looks good but target IoU doesn't improve?**
A: Possible causes:
1. Source and target have different label distributions
2. Alignment layers are wrong (try decoder layers if encoder fails)
3. Need more training epochs
4. Increase `coral_weight`

**Q: Can I visualize more than 2000 points?**
A: Yes, edit `n_samples` parameter in `plot_tsne()` and `plot_umap()`, but expect longer runtime.

## Citation

If you use this monitoring system, please cite the underlying techniques:

**CORAL (Correlation Alignment):**
```bibtex
@inproceedings{sun2016coral,
  title={Deep CORAL: Correlation alignment for deep domain adaptation},
  author={Sun, Baochen and Saenko, Kate},
  booktitle={ECCV Workshops},
  year={2016}
}
```

**SqueezeSegV2 (Geodesic Distance):**
```bibtex
@inproceedings{wu2019squeezesegv2,
  title={SqueezeSegV2: Improved model structure and unsupervised domain adaptation for road-object segmentation from a lidar point cloud},
  author={Wu, Bichen and Zhou, Xuanyu and Zhao, Sicheng and Yue, Xiangyu and Keutzer, Kurt},
  booktitle={ICRA},
  year={2019}
}
```
