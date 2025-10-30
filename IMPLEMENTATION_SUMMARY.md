# HarvardForest3D SSL Implementation Summary

## Overview

Complete implementation of self-supervised learning (SSL) for HarvardForest3D dataset using rotation prediction as the pretext task. The encoder can be initialized from a pretrained RandLANet checkpoint trained on Semantic3D.

## Files Created

### 1. Dataset Loader
- **File**: `ml3d/datasets/harvardforest3d.py`
- **Purpose**: Load LAS point cloud files, create deterministic train/val splits, extract XYZ + features
- **Key Features**:
  - Reads LAS format using `laspy`
  - Deterministic split (90/10 train/val by default)
  - Persistent split manifest in cache
  - Supports intensity and RGB features
  - No ground truth labels (SSL dataset)

### 2. SSL Model Wrapper
- **File**: `ml3d/torch/models/randlanet_ssl.py`
- **Purpose**: Wrap RandLANet encoder with SSL head for rotation prediction
- **Key Features**:
  - Reuses RandLANet encoder (fc0, bn0, encoder blocks, mlp)
  - Global average pooling + MLP classifier head
  - Load pretrained encoder from Semantic3D checkpoint
  - Conditional encoder freezing (warmup)
  - Separate learning rates for encoder and head

### 3. SSL Training Pipeline
- **File**: `ml3d/torch/pipelines/ssl_rotation.py`
- **Purpose**: Training loop for rotation prediction SSL
- **Key Features**:
  - Apply discrete Z-axis rotations (0°, 90°, 180°, 270°)
  - Cross-entropy loss on rotation class
  - Training and validation loops
  - TensorBoard logging
  - Checkpoint saving (best + periodic)

### 4. Configuration File
- **File**: `ml3d/configs/harvardforest3d_ssl.yml`
- **Purpose**: All hyperparameters and paths
- **Key Settings**:
  - Dataset path (UPDATE THIS)
  - Pretrained encoder path (UPDATE THIS)
  - Batch size: 4
  - Epochs: 50
  - LR head: 0.001, LR encoder: 0.0001
  - 4 rotation classes
  - Freeze encoder: 5 epochs

### 5. Training Script
- **File**: `scripts/train_harvardforest_ssl.py`
- **Purpose**: Easy command-line training launcher
- **Features**:
  - Parse config file
  - Override settings via CLI args
  - Validate dataset path
  - Initialize dataset, model, pipeline
  - Run training or testing
  - Comprehensive error handling

### 6. Documentation
- **File**: `docs/harvardforest3d.md`
- **Purpose**: Complete dataset and SSL task documentation
- **Sections**:
  - Dataset structure and splits
  - Prerequisites and installation
  - Usage examples (config and code)
  - SSL rotation task explanation
  - Hyperparameter guide
  - Troubleshooting
  - Advanced usage

- **File**: `docs/QUICKSTART_HARVARDFOREST.md`
- **Purpose**: Quick start guide with commands
- **Sections**:
  - Setup steps
  - Training commands
  - Monitoring (TensorBoard)
  - Testing
  - Troubleshooting

### 7. Module Registration
- **Updated**: `ml3d/datasets/__init__.py` - Added `HarvardForest3D`
- **Updated**: `ml3d/torch/models/__init__.py` - Added `RandLANetSSL`
- **Updated**: `ml3d/torch/pipelines/__init__.py` - Added `SSLRotation`

## How to Use

### Step 1: Install Dependencies

```bash
pip install laspy
```

### Step 2: Update Paths

Edit `ml3d/configs/harvardforest3d_ssl.yml`:

```yaml
dataset:
  dataset_path: /home/fzhcis/mylab/data/HarvardForest  # YOUR PATH

model:
  pretrained_encoder_path: logs/randlanet_semantic3d_202201071330utc.pth  # YOUR CHECKPOINT
```

### Step 3: Train

```bash
python scripts/train_harvardforest_ssl.py \
  --cfg ml3d/configs/harvardforest3d_ssl.yml \
  --device cuda
```

### Step 4: Monitor

```bash
tensorboard --logdir train_log/RandLANetSSL_HarvardForest3D_torch/
```

### Step 5: Test

```bash
python scripts/train_harvardforest_ssl.py \
  --cfg ml3d/configs/harvardforest3d_ssl.yml \
  --test_only \
  --device cuda
```

## Architecture

```
Input: LAS point cloud (N x 3) + features
  ↓
RandLANet Encoder (pretrained from Semantic3D)
  - fc0: Input projection
  - encoder: 5 LocalFeatureAggregation layers
  - mlp: Bottleneck features
  ↓
Global Average Pooling (per-point → global)
  ↓
SSL Head (MLP classifier)
  - FC(512 → 256) + BN + ReLU + Dropout
  - FC(256 → 128) + BN + ReLU + Dropout
  - FC(128 → 4)  # 4 rotation classes
  ↓
Output: Rotation class logits
```

## Training Strategy

1. **Warmup (Epochs 0-4)**:
   - Freeze encoder (only train SSL head)
   - LR head: 0.001

2. **Fine-tuning (Epochs 5-49)**:
   - Unfreeze encoder
   - LR encoder: 0.0001
   - LR head: 0.001

3. **Scheduler**: Exponential decay (γ=0.995)

4. **Loss**: Cross-entropy on rotation class

5. **Metric**: Top-1 accuracy

## Expected Performance

- **Random baseline**: 25% (4 classes)
- **Target accuracy**: 85-95% on validation
- **Training time**: ~2-4 hours (50 epochs, batch size 4, 1 GPU)

## Outputs

### Checkpoints
- Location: `logs/checkpoint/`
- Files:
  - `ckpt_epoch_005.pth`, `ckpt_epoch_010.pth`, etc. (every 5 epochs)
  - `ckpt_best.pth` (best validation accuracy)

### Logs
- TensorBoard: `train_log/RandLANetSSL_HarvardForest3D_torch/<runid>/`
- Text logs: `logs/log_train_<timestamp>.txt`

### Split Manifest
- Location: `logs/cache_harvardforest/HarvardForest3D_split.json`
- Content: Deterministic train/val file lists

## Common Commands

### Train with custom batch size
```bash
python scripts/train_harvardforest_ssl.py \
  --cfg ml3d/configs/harvardforest3d_ssl.yml \
  --batch_size 2 \
  --device cuda
```

### Train for more epochs
```bash
python scripts/train_harvardforest_ssl.py \
  --cfg ml3d/configs/harvardforest3d_ssl.yml \
  --max_epoch 100 \
  --device cuda
```

### Resume training
```bash
python scripts/train_harvardforest_ssl.py \
  --cfg ml3d/configs/harvardforest3d_ssl.yml \
  --resume \
  --device cuda
```

### Train from scratch (no pretrained encoder)
Edit config: Remove or comment out `pretrained_encoder_path`

## Next Steps After Training

### 1. Feature Extraction
Use the trained encoder to extract embeddings:

```python
model.load_state_dict(torch.load('logs/checkpoint/ckpt_best.pth')['model_state_dict'])
model.eval()

# Forward through encoder only (before SSL head)
with torch.no_grad():
    feat = model.mlp(encoder_output)  # (B, 512, N', 1)
    embeddings = feat.mean(dim=2)     # (B, 512)
```

### 2. Downstream Fine-tuning
If you get labeled data later:
- Replace SSL head with segmentation head
- Load pretrained encoder weights
- Fine-tune on labeled subset

### 3. Visualization
- t-SNE/UMAP of learned embeddings
- Nearest-neighbor retrieval between tiles
- Clustering forest structures

## Troubleshooting

### Issue: CUDA out of memory
**Solution**: Reduce `batch_size` to 2 or 1

### Issue: Low validation accuracy (<50%)
**Possible causes**:
- Encoder frozen too long → reduce `freeze_encoder_epochs`
- Learning rate too high/low → adjust in config
- Model too small → increase `ssl_head_dims`

### Issue: laspy ImportError
**Solution**: `pip install laspy`

### Issue: Pretrained checkpoint not loading
**Check**:
- Path exists: `ls logs/.../*.pth`
- Config has correct path
- Use `load_encoder_strict: false` for partial loading

## Files Summary Table

| File | Type | Purpose | Status |
|------|------|---------|--------|
| `ml3d/datasets/harvardforest3d.py` | Dataset | LAS reader, splits | ✅ Created |
| `ml3d/torch/models/randlanet_ssl.py` | Model | SSL wrapper | ✅ Created |
| `ml3d/torch/pipelines/ssl_rotation.py` | Pipeline | Training loop | ✅ Created |
| `ml3d/configs/harvardforest3d_ssl.yml` | Config | Hyperparams | ✅ Created |
| `scripts/train_harvardforest_ssl.py` | Script | CLI launcher | ✅ Created |
| `docs/harvardforest3d.md` | Docs | Full guide | ✅ Created |
| `docs/QUICKSTART_HARVARDFOREST.md` | Docs | Quick start | ✅ Created |
| `ml3d/datasets/__init__.py` | Init | Registration | ✅ Updated |
| `ml3d/torch/models/__init__.py` | Init | Registration | ✅ Updated |
| `ml3d/torch/pipelines/__init__.py` | Init | Registration | ✅ Updated |

## Dependencies

- `laspy>=2.0.0` - LAS file reading
- `torch` - Deep learning framework
- `numpy` - Numerical operations
- `pandas` (optional) - Data handling
- `tensorboard` - Logging and visualization
- `tqdm` - Progress bars
- `sklearn` - KDTree for nearest neighbors

## Contact and Support

For issues:
1. Check `docs/harvardforest3d.md` for detailed troubleshooting
2. Review training logs in `logs/log_train_*.txt`
3. Check TensorBoard for loss curves
4. Verify dataset path and LAS files exist

---

**Implementation Date**: October 2025
**Author**: GitHub Copilot
**Framework**: Open3D-ML (PyTorch)
