# Quick Start Guide - PointContrast on HarvardForest3D

## Environment Setup

```bash
# Activate conda environment
conda activate pcd_seg_open3d_env

# Or use conda run (no activation needed)
conda run -n pcd_seg_open3d_env [command]
```

## Training Commands

### Option 1: Direct Python (after activating env)
```bash
conda activate pcd_seg_open3d_env
cd /home/fzhcis/mylab/Open3D-ML

python scripts/run_tests_fei/train_harvardforest_contrast.py \
  --cfg ml3d/configs/randlanet_harvardforest_contrast.yml \
  --device cuda
```

### Option 2: Using conda run (no activation needed)
```bash
cd /home/fzhcis/mylab/Open3D-ML

conda run -n pcd_seg_open3d_env python scripts/run_tests_fei/train_harvardforest_contrast.py \
  --cfg ml3d/configs/randlanet_harvardforest_contrast.yml \
  --device cuda
```

### Option 3: Using convenience script (simplest)
```bash
cd /home/fzhcis/mylab/Open3D-ML

# Default (RandLANet)
bash scripts/run_tests_fei/run_contrast_training.sh

# Custom config
bash scripts/run_tests_fei/run_contrast_training.sh \
  --cfg ml3d/configs/kpconv_harvardforest_contrast.yml \
  --device cuda

# With extra arguments
bash scripts/run_tests_fei/run_contrast_training.sh \
  --batch_size 4 \
  --max_epoch 150
```

## Before Training

### 1. Update Paths in Config
Edit `ml3d/configs/randlanet_harvardforest_contrast.yml`:

```yaml
dataset:
  dataset_path: /home/fzhcis/mylab/data/HarvardForest  # YOUR PATH

model:
  pretrained_encoder_path: /home/fzhcis/mylab/data/semantic3d/open3d_randlanet/test_1027/randlanet_semantic3d_xyz_ckpt_2025-10-30_22-15-32_00400.pth
```

### 2. Verify Data
```bash
ls /home/fzhcis/mylab/data/HarvardForest/*.las
# Should show S01_000.las, S01_001.las, etc.
```

## Monitoring Training

```bash
# Follow training log
tail -f contrastive_training.log

# Check GPU usage
watch -n 1 nvidia-smi

# View saved checkpoints
ls -lh logs/checkpoint/
```

## Common Commands

### Train RandLANet (Default)
```bash
conda run -n pcd_seg_open3d_env python scripts/run_tests_fei/train_harvardforest_contrast.py \
  --cfg ml3d/configs/randlanet_harvardforest_contrast.yml \
  --device cuda
```

### Train KPConv
```bash
conda run -n pcd_seg_open3d_env python scripts/run_tests_fei/train_harvardforest_contrast.py \
  --cfg ml3d/configs/kpconv_harvardforest_contrast.yml \
  --device cuda \
  --batch_size 4
```

### Resume from Checkpoint
```bash
conda run -n pcd_seg_open3d_env python scripts/run_tests_fei/train_harvardforest_contrast.py \
  --cfg ml3d/configs/randlanet_harvardforest_contrast.yml \
  --device cuda \
  --resume logs/checkpoint/ckpt_epoch_050.pth
```

### Train with Custom Settings
```bash
conda run -n pcd_seg_open3d_env python scripts/run_tests_fei/train_harvardforest_contrast.py \
  --cfg ml3d/configs/randlanet_harvardforest_contrast.yml \
  --device cuda \
  --batch_size 4 \
  --max_epoch 150 \
  --learning_rate 0.0005
```

### Background Training (with logging)
```bash
conda run -n pcd_seg_open3d_env python scripts/run_tests_fei/train_harvardforest_contrast.py \
  --cfg ml3d/configs/randlanet_harvardforest_contrast.yml \
  --device cuda \
  2>&1 | tee training_$(date +%Y%m%d_%H%M%S).log &
```

## Expected Output

```
==========================================
POINTCONTRAST-STYLE CONTRASTIVE LEARNING ON HARVARDFOREST3D
==========================================
Config: ml3d/configs/randlanet_harvardforest_contrast.yml
Device: cuda
Model: RandLANetContrast
==========================================
INFO - Found 70 .las files in /path/to/HarvardForest
INFO - HarvardForest3D: train=63 val=7 test=0
INFO - RandLANetContrast initialized:
INFO -   Encoder output dim: 1024
INFO -   Projection output dim: 128
INFO -   Freeze encoder epochs: 5
INFO - Loading pretrained encoder from ...
INFO - Successfully loaded 123 encoder parameters
==========================================
STARTING TRAINING
==========================================

Train Epoch 1: 100%|██████| 8/8 [02:34<00:00, 19.32s/it]
Val Epoch 1: 100%|██████| 1/1 [00:15<00:00, 15.23s/it]
Epoch 1/100 | Train Loss: 1.8234 | Train Acc: 0.3421 | Val Loss: 1.7891 | Val Acc: 0.3589 | LR: 1.00e-03

Train Epoch 2: 100%|██████| 8/8 [02:31<00:00, 18.94s/it]
...
```

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
--batch_size 4  # or 2, or 1
```

### Module Import Error
```bash
# Make sure you're in the right environment
conda activate pcd_seg_open3d_env
python -c "import ml3d; print('OK')"

# Or check environment
conda env list
conda list | grep torch
```

### Dataset Not Found
```bash
# Check path exists
ls /home/fzhcis/mylab/data/HarvardForest/

# Check .las files
ls /home/fzhcis/mylab/data/HarvardForest/*.las | wc -l
```

## Files and Directories

```
Open3D-ML/
├── scripts/run_tests_fei/
│   ├── train_harvardforest_contrast.py      ← Main training script
│   └── run_contrast_training.sh             ← Convenience wrapper
├── ml3d/configs/
│   ├── randlanet_harvardforest_contrast.yml ← RandLANet config
│   └── kpconv_harvardforest_contrast.yml    ← KPConv config
├── logs/
│   ├── checkpoint/                          ← Saved checkpoints
│   │   ├── ckpt_best.pth                   ← Best model
│   │   └── ckpt_epoch_XXX.pth              ← Periodic saves
│   └── cache_harvardforest_contrast/        ← Dataset cache
├── contrastive_training.log                 ← Training log
└── docs/
    └── POINTCONTRAST_HARVARDFOREST.md      ← Full documentation
```

## Next Steps After Training

1. **Check Best Checkpoint**
   ```bash
   ls -lh logs/checkpoint/ckpt_best.pth
   ```

2. **Extract Encoder for Downstream Tasks**
   ```python
   checkpoint = torch.load('logs/checkpoint/ckpt_best.pth')
   encoder_weights = {k: v for k, v in checkpoint['model_state_dict'].items()
                      if 'projection_head' not in k}
   ```

3. **Fine-tune on Labeled Data**
   - Use encoder_weights to initialize segmentation model
   - Train on labeled subset of HarvardForest or other dataset

---

**Quick Help**: For detailed documentation, see `docs/POINTCONTRAST_HARVARDFOREST.md`
