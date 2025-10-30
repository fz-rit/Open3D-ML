# HarvardForest3D SSL Quick Start Guide

## ✅ Setup Complete!

All files have been created and verified. You're ready to start SSL training.

## 📁 Files Created

### Core Implementation
- ✅ `ml3d/datasets/harvardforest3d.py` - Dataset loader for LAS files
- ✅ `ml3d/torch/models/randlanet_ssl.py` - RandLANet SSL wrapper
- ✅ `ml3d/torch/pipelines/ssl_rotation.py` - SSL rotation training pipeline
- ✅ `ml3d/configs/harvardforest3d_ssl.yml` - Configuration file
- ✅ `scripts/train_harvardforest_ssl.py` - Training launcher script
- ✅ `scripts/verify_harvardforest_setup.py` - Setup verification script
- ✅ `docs/harvardforest3d.md` - Full documentation

### Module Registration
- ✅ Updated `ml3d/datasets/__init__.py`
- ✅ Updated `ml3d/torch/models/__init__.py`
- ✅ Updated `ml3d/torch/pipelines/__init__.py`

## 🚀 Quick Commands

### 1. Verify Setup (Optional)
```bash
conda activate pcd_seg_open3d_env
cd /home/fzhcis/mylab/Open3D-ML
python scripts/verify_harvardforest_setup.py --cfg ml3d/configs/harvardforest3d_ssl.yml
```

### 2. Start Training
```bash
conda activate pcd_seg_open3d_env
cd /home/fzhcis/mylab/Open3D-ML

# Full training with defaults
python scripts/train_harvardforest_ssl.py \
  --cfg ml3d/configs/harvardforest3d_ssl.yml \
  --device cuda

# Or with custom settings
python scripts/train_harvardforest_ssl.py \
  --cfg ml3d/configs/harvardforest3d_ssl.yml \
  --batch_size 2 \
  --max_epoch 100 \
  --device cuda \
  --gpu_id 0
```

### 3. Monitor Training
```bash
# In a separate terminal
conda activate pcd_seg_open3d_env
tensorboard --logdir train_log/RandLANetSSL_HarvardForest3D_torch/
```

Then open browser to: http://localhost:6006

### 4. Test Trained Model
```bash
python scripts/train_harvardforest_ssl.py \
  --cfg ml3d/configs/harvardforest3d_ssl.yml \
  --test_only \
  --device cuda
```

## 📊 Current Configuration

### Dataset
- **Path**: `/home/fzhcis/mylab/data/HarvardForest`
- **Files**: 62 LAS files
- **Split**: 90% train / 10% val (deterministic)
- **Points per sample**: 65,536

### Model
- **Architecture**: RandLANet encoder + SSL head
- **Pretrained**: `/home/fzhcis/mylab/data/semantic3d/open3d_randlanet/test_1027/ckpt_00400.pth`
- **Rotation classes**: 4 (0°, 90°, 180°, 270°)

### Training
- **Epochs**: 50
- **Batch size**: 4
- **Learning rate (head)**: 0.001
- **Learning rate (encoder)**: 0.0001
- **Freeze encoder**: First 5 epochs
- **GPU**: NVIDIA RTX A2000 12GB

## 📈 Expected Results

- **Training time**: ~2-4 hours (50 epochs)
- **Random baseline**: 25% accuracy
- **Target accuracy**: 85-95% on validation
- **Memory usage**: ~8-10 GB GPU

## 🔧 Common Adjustments

### If GPU memory is limited:
```bash
python scripts/train_harvardforest_ssl.py \
  --cfg ml3d/configs/harvardforest3d_ssl.yml \
  --batch_size 2 \
  --device cuda
```

### Train longer:
```bash
python scripts/train_harvardforest_ssl.py \
  --cfg ml3d/configs/harvardforest3d_ssl.yml \
  --max_epoch 100 \
  --device cuda
```

### Different learning rates:
```bash
python scripts/train_harvardforest_ssl.py \
  --cfg ml3d/configs/harvardforest3d_ssl.yml \
  --learning_rate 0.0005 \
  --encoder_lr 0.00005 \
  --device cuda
```

## 📂 Output Locations

After training starts, check these directories:

- **Checkpoints**: `logs/checkpoint/`
  - `ckpt_epoch_005.pth`, `ckpt_epoch_010.pth`, etc.
  - `ckpt_best.pth` (best validation accuracy)

- **TensorBoard logs**: `train_log/RandLANetSSL_HarvardForest3D_torch/<runid>/`

- **Text logs**: `logs/log_train_<timestamp>.txt`

- **Split manifest**: `logs/cache_harvardforest/HarvardForest3D_split.json`

## 🐛 Troubleshooting

### Issue: Module not found errors
**Solution**: Make sure to activate the conda environment:
```bash
conda activate pcd_seg_open3d_env
```

### Issue: CUDA out of memory
**Solution**: Reduce batch size or num_points in config

### Issue: Low accuracy (<50%)
**Check**:
- Loss is decreasing in tensorboard
- Encoder unfreezes after epoch 5
- Learning rates are reasonable

### Issue: Import errors
**Solution**: Run from repo root and ensure sys.path includes local repo:
```bash
cd /home/fzhcis/mylab/Open3D-ML
python scripts/train_harvardforest_ssl.py ...
```

## 📖 Full Documentation

For detailed information, see:
- `docs/harvardforest3d.md` - Complete dataset and SSL documentation
- `IMPLEMENTATION_SUMMARY.md` - Full implementation details

## 🎯 Next Steps After Training

1. **Check validation accuracy** in tensorboard or final logs
2. **Load best checkpoint** for feature extraction
3. **Extract embeddings** for downstream tasks:
   - Clustering forest structures
   - Nearest-neighbor search
   - Transfer learning to labeled data

## 📝 Training Command Template

Save this for repeated use:

```bash
#!/bin/bash
# train_harvardforest.sh

conda activate pcd_seg_open3d_env
cd /home/fzhcis/mylab/Open3D-ML

python scripts/train_harvardforest_ssl.py \
  --cfg ml3d/configs/harvardforest3d_ssl.yml \
  --device cuda \
  --gpu_id 0 \
  2>&1 | tee training_$(date +%Y%m%d_%H%M%S).log
```

Make it executable:
```bash
chmod +x train_harvardforest.sh
./train_harvardforest.sh
```

---

**Ready to train!** 🚀

Run the training command and monitor progress via tensorboard or text logs.
