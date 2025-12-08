# Domain Adaptation for Point Cloud Semantic Segmentation

This directory contains the implementation of unsupervised domain adaptation using **Correlation Alignment (CORAL)** for point cloud semantic segmentation, inspired by SqueezeSegV2.

## Overview

Domain adaptation helps transfer knowledge from a **labeled source domain** (e.g., Semantic3DUnified) to an **unlabeled target domain** (e.g., ForestSemantic, DigiForest) to improve model performance on the target domain without requiring target labels during training.

## Key Components

### 1. CORAL Loss (`ml3d/torch/modules/losses/coral_loss.py`)

Three variants of CORAL loss for domain alignment:

- **`CORALLoss`**: Basic correlation alignment with geodesic distance on SPD manifold
- **`MultiLayerCORALLoss`**: Aligns features at multiple network depths
- **`AdaptiveCORALLoss`**: Progressive domain calibration (gradually increases weight)

### 2. Domain Adaptation Models

- **`RandLANetDA`** (`ml3d/torch/models/randlanet_da.py`): RandLANet with feature extraction
- **`KPFCNNDA`** (`ml3d/torch/models/kpconv_da.py`): KPConv with feature extraction

Key features:
- `forward()` returns `(logits, features_list)` when `return_features=True`
- `get_encoder_features()` extracts features without full forward pass
- Configurable `alignment_layers` to specify which encoder layers to align

### 3. Domain Adaptation Pipeline

**`DomainAdaptationSemanticSegmentation`** (`ml3d/torch/pipelines/domain_adaptation_semseg.py`)

Handles dual dataloaders for source and target domains:
- Source domain: supervised segmentation loss
- Target domain: unsupervised CORAL loss
- Progressive weight scheduling
- Optional target domain validation

## Configuration Files

### RandLANet Configs

1. **Semantic3D → ForestSemantic**: `ml3d/configs/randlanet_da_semantic3d_to_forest.yml`
2. **Semantic3D → DigiForest**: `ml3d/configs/randlanet_da_semantic3d_to_digiforest.yml`

### KPConv Configs

1. **Semantic3D → ForestSemantic**: `ml3d/configs/kpconv_da_semantic3d_to_forest.yml`
2. **Semantic3D → DigiForest**: `ml3d/configs/kpconv_da_semantic3d_to_digiforest.yml`

## Usage

### 1. Update Configuration

Edit the config file to set your dataset paths:

```yaml
source_dataset:
  dataset_path: /path/to/semantic3dunified/

target_dataset:
  dataset_path: /path/to/forestsemantic/
```

### 2. Train with Domain Adaptation

```bash
# RandLANet: Semantic3D → ForestSemantic
python scripts/train_domain_adaptation.py \
    --config ml3d/configs/randlanet_da_semantic3d_to_forest.yml \
    --device cuda

# KPConv: Semantic3D → DigiForest
python scripts/train_domain_adaptation.py \
    --config ml3d/configs/kpconv_da_semantic3d_to_digiforest.yml \
    --device cuda

# Dry run (test configuration)
python scripts/train_domain_adaptation.py \
    --config ml3d/configs/randlanet_da_semantic3d_to_forest.yml \
    --dry-run
```

### 3. Monitor Training

View tensorboard logs:

```bash
tensorboard --logdir=./train_log
```

Key metrics to monitor:
- **Segmentation loss**: Source domain supervised loss
- **CORAL loss**: Domain alignment loss
- **CORAL weight**: Progressive weight schedule
- **Source validation IoU**: Source domain performance
- **Target validation IoU**: Target domain performance (if enabled)

### 4. Test on Target Domain

After training, test on the target domain:

```python
from ml3d.torch import pipelines, models, datasets

# Load trained model
model = models.RandLANetDA(ckpt_path='path/to/best_checkpoint.pth')

# Load target dataset
target_dataset = datasets.ForestSemantic(dataset_path='...')

# Initialize pipeline
pipeline = pipelines.DomainAdaptationSemanticSegmentation(
    model=model,
    source_dataset=source_dataset,
    target_dataset=target_dataset
)

# Run test
pipeline.run_test()
```

## Hyperparameter Tuning

### Critical Hyperparameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `coral_weight` | 0.01-1.0 | 0.1 | CORAL loss weight |
| `progressive_steps` | 1000-10000 | 5000 | Steps to ramp up weight |
| `alignment_layers` | Subset of layers | [0,2,4] | Which layers to align |
| `layer_weights` | Sum to 1.0 | Equal | Per-layer weights |
| `learning_rate` | 1e-5 to 1e-3 | 5e-4 | Lower than standard |

### Tuning Strategy

1. **Start with default settings**
2. **Adjust `coral_weight`**:
   - Too high: Model focuses on alignment, poor segmentation
   - Too low: Insufficient domain adaptation
   - Try: 0.01, 0.05, 0.1, 0.5, 1.0
3. **Tune `progressive_steps`**:
   - Shorter: Faster adaptation, less stable
   - Longer: More stable, slower adaptation
4. **Select `alignment_layers`**:
   - More layers: Better alignment, more computation
   - Fewer layers: Faster, may miss important features
   - Try: early layers [0,1,2], middle [2,3], late [3,4], or all

## Expected Results

Based on SqueezeSegV2 findings:

| Scenario | Source Only | With CORAL DA | Improvement |
|----------|-------------|---------------|-------------|
| Typical | 40-50% | 60-75% | +15-25% |
| Best case | 50-60% | 70-85% | +20-25% |

Your results may vary depending on:
- Domain shift magnitude
- Source dataset size and quality
- Target dataset characteristics
- Hyperparameter tuning

## Implementation Details

### CORAL Loss Computation

```python
# Compute covariance matrices
C_s = (X_s^T X_s) / (n_s - 1)  # Source covariance
C_t = (X_t^T X_t) / (n_t - 1)  # Target covariance

# Geodesic distance (more accurate)
M = C_s^{-1/2} C_t C_s^{-1/2}
L_coral = ||log(M)||_F / (4*d*d)

# Or Frobenius norm (faster)
L_coral = ||C_s - C_t||_F^2 / (4*d*d)
```

### Progressive Weight Schedule

```python
progress = min(1.0, global_step / progressive_steps)
adaptive_weight = coral_weight * progress
total_loss = seg_loss + adaptive_weight * coral_loss
```

### Multi-Layer Alignment

```python
features = [encoder1_out, encoder2_out, encoder3_out, ...]
coral_loss = sum(w_i * CORAL(s_feat_i, t_feat_i) 
                 for w_i, s_feat_i, t_feat_i 
                 in zip(layer_weights, source_features, target_features))
```

## Troubleshooting

### Issue: CORAL loss is NaN

**Solution**: Check for:
- Ill-conditioned covariance matrices → Increase epsilon
- Too few samples → Increase batch size
- Extreme feature values → Check normalization

### Issue: No improvement on target domain

**Solution**: Try:
- Increase `coral_weight` (0.1 → 0.5)
- Add more `alignment_layers`
- Longer training (more epochs)
- Verify target data preprocessing

### Issue: Source performance drops

**Solution**:
- Decrease `coral_weight` (0.1 → 0.05)
- Longer `progressive_steps` for gradual adaptation
- Use `coral_loss_type: adaptive`

### Issue: Training is unstable

**Solution**:
- Lower learning rate
- Longer `progressive_steps`
- Add gradient clipping
- Use `use_geodesic: false` for simpler distance

## Advanced Techniques

### 1. Self-Training (Post-CORAL)

After CORAL training, use pseudo-labels on target domain:

```python
# Generate pseudo-labels on target
predictions = model.predict(target_unlabeled)
confident_mask = predictions.confidence > 0.9

# Fine-tune with pseudo-labels
model.train(target_data[confident_mask], predictions[confident_mask])
```

### 2. Multi-Source Domain Adaptation

Adapt from multiple source domains:

```python
coral_loss = sum(CORAL(source_i_features, target_features) 
                 for source_i_features in all_source_features)
```

### 3. Class-Conditional Alignment

Apply different alignment per semantic class (requires modifications).

## References

1. **SqueezeSegV2**: Wu et al., "SqueezeSegV2: Improved Model Structure and Unsupervised Domain Adaptation for Road-Object Segmentation from a LiDAR Point Cloud", arXiv:1809.08495
2. **Deep CORAL**: Sun et al., "Deep CORAL: Correlation Alignment for Deep Domain Adaptation", ECCV 2016
3. **RandLA-Net**: Hu et al., "RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds", CVPR 2020
4. **KPConv**: Thomas et al., "KPConv: Flexible and Deformable Convolution for Point Clouds", ICCV 2019

## Citation

If you use this implementation, please cite:

```bibtex
@article{wu2018squeezesegv2,
  title={SqueezeSegV2: Improved Model Structure and Unsupervised Domain Adaptation for Road-Object Segmentation from a LiDAR Point Cloud},
  author={Wu, Bichen and Zhou, Xuanyu and Zhao, Sicheng and Yue, Xiangyu and Keutzer, Kurt},
  journal={arXiv preprint arXiv:1809.08495},
  year={2018}
}
```
