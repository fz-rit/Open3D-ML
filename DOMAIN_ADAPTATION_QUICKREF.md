# Quick Reference: Domain Adaptation Configuration Guide

## Alignment Layers Decision Matrix

### Where to Align: ENCODER ONLY ✓

```
Input → [Encoder Layers 0→1→2→3→4] → Decoder → Output
         ↑     ↑     ↑     ↑     ↑
         │     │     │     │     └── Deepest (high-level semantics)
         │     │     │     └──────── High-level features
         │     │     └────────────── Mid-level structures  
         │     └──────────────────── Early features
         └────────────────────────── Geometric primitives

❌ Don't align decoder - it's task-specific
```

### Number of Layers

| Configuration | Layers | Use Case | Memory | Accuracy |
|--------------|--------|----------|---------|----------|
| **Minimal** | `[4]` | Debugging, fast experiments | Low | Baseline |
| **Balanced** | `[2, 4]` | Good middle ground | Medium | Good |
| **Recommended** | `[0, 2, 4]` | Best for most cases | Medium | Better |
| **Aggressive** | `[0, 1, 2, 3, 4]` | Maximum alignment | High | Best |

### Layer Weights

**Progressive weighting** (recommended):
```yaml
alignment_layers: [0, 2, 4]
layer_weights: [0.2, 0.3, 0.5]  # Emphasize deeper features
```

**Equal weighting** (simpler):
```yaml
alignment_layers: [0, 2, 4]
layer_weights: [0.33, 0.33, 0.34]
```

**Custom weighting** (when you know which matters):
```yaml
alignment_layers: [0, 2, 4]
layer_weights: [0.1, 0.3, 0.6]  # Heavy on semantics
```

## Steps Per Epoch Guide

### Training Phases

| Phase | Purpose | steps_per_epoch_train | steps_per_epoch_valid | max_epoch | Time/Epoch |
|-------|---------|----------------------|----------------------|-----------|------------|
| **Debug** | Verify pipeline works | 5 | 2 | 2-3 | 1-2 min |
| **Pilot** | Test DA effectiveness | 20-32 | 10-16 | 10-20 | 5-10 min |
| **Full** | Best performance | 64-128 | 24-48 | 100-200 | 30-60 min |

### Inference

No `steps_per_epoch` needed - automatically processes entire test set:
```python
pipeline.run_test()  # Iterates through all test files
```

## Complete Configuration Examples

### Example 1: DEBUG Configuration (Fast Iteration)

```yaml
source_dataset:
  steps_per_epoch_train: 5
  steps_per_epoch_valid: 2

model:
  alignment_layers: [4]  # Single layer

pipeline:
  coral_weight: 0.1
  progressive_steps: 50  # Fast ramp
  alignment_layers: [4]
  layer_weights: [1.0]
  max_epoch: 3
  batch_size: 2
```

**Use for**: Catching bugs, verifying pipeline setup
**Time**: ~5 minutes total

### Example 2: PILOT Configuration (Quick Validation)

```yaml
source_dataset:
  steps_per_epoch_train: 32
  steps_per_epoch_valid: 16

model:
  alignment_layers: [2, 4]  # Two layers

pipeline:
  coral_weight: 0.1
  progressive_steps: 500
  alignment_layers: [2, 4]
  layer_weights: [0.4, 0.6]
  max_epoch: 20
  batch_size: 8
```

**Use for**: Testing if CORAL helps, initial hyperparameter search
**Time**: 2-4 hours

### Example 3: FULL Configuration (Production Training)

```yaml
source_dataset:
  steps_per_epoch_train: 64
  steps_per_epoch_valid: 24

model:
  alignment_layers: [0, 2, 4]  # Multi-level

pipeline:
  coral_weight: 0.1  # Tune: 0.05, 0.1, 0.2, 0.5
  progressive_steps: 5000
  alignment_layers: [0, 2, 4]
  layer_weights: [0.2, 0.3, 0.5]
  max_epoch: 150
  batch_size: 8-16
```

**Use for**: Best model, publication results
**Time**: 1-2 days

## Hyperparameter Tuning Workflow

### Step 1: Debug (Required)
```bash
# Set in config:
# - alignment_layers: [4]
# - steps_per_epoch_train: 5
# - max_epoch: 2

python scripts/train_domain_adaptation.py --config your_config.yml
```
**Goal**: Verify no crashes, losses are computed

### Step 2: Baseline (Recommended)
```bash
# Set in config:
# - alignment_layers: [2, 4]
# - steps_per_epoch_train: 32
# - max_epoch: 20
# - coral_weight: 0.1

python scripts/train_domain_adaptation.py --config your_config.yml
```
**Goal**: Establish baseline CORAL effectiveness

### Step 3: Tune CORAL Weight
```bash
# Run 3-5 experiments with different coral_weight:
# - 0.05, 0.1, 0.2, 0.5

# Keep other settings from Step 2
```
**Goal**: Find optimal domain adaptation strength

### Step 4: Tune Layers
```bash
# Try different alignment configurations:
# - [4] - single
# - [2, 4] - dual
# - [0, 2, 4] - triple
# - [0, 1, 2, 3, 4] - all

# Use best coral_weight from Step 3
```
**Goal**: Find optimal feature alignment

### Step 5: Full Training
```bash
# Set in config:
# - Best alignment_layers from Step 4
# - Best coral_weight from Step 3
# - steps_per_epoch_train: 64
# - max_epoch: 150+

python scripts/train_domain_adaptation.py --config your_config.yml
```
**Goal**: Train final model

## Troubleshooting Decision Tree

```
Problem: Source IoU drops significantly (>5%)
├─ Solution 1: Reduce coral_weight (0.1 → 0.05)
├─ Solution 2: Use fewer layers ([0,2,4] → [4])
├─ Solution 3: Increase progressive_steps (5000 → 10000)
└─ Solution 4: Lower learning_rate (0.0005 → 0.0003)

Problem: Target improvement is minimal (<5%)
├─ Solution 1: Increase coral_weight (0.1 → 0.2 or 0.5)
├─ Solution 2: Add more layers ([2,4] → [0,2,4])
├─ Solution 3: Train longer (max_epoch: 200)
└─ Solution 4: Check target data quality/preprocessing

Problem: Training is unstable (loss spikes)
├─ Solution 1: Longer progressive_steps (5000 → 10000)
├─ Solution 2: Lower learning_rate
├─ Solution 3: Smaller batch_size
└─ Solution 4: Use use_geodesic: false (simpler distance)

Problem: CORAL loss is NaN
├─ Solution 1: Check batch_size (too small? increase to 4+)
├─ Solution 2: Use use_geodesic: false
├─ Solution 3: Check data normalization
└─ Solution 4: Reduce coral_weight

Problem: Training is too slow
├─ Solution 1: Reduce alignment_layers ([0,2,4] → [4])
├─ Solution 2: Reduce steps_per_epoch (64 → 32)
├─ Solution 3: Use use_geodesic: false
└─ Solution 4: Smaller batch_size with gradient accumulation
```

## Memory Optimization

If you encounter OOM (Out of Memory):

```yaml
# Reduce in this order:
1. batch_size: 8 → 4 → 2
2. alignment_layers: [0,2,4] → [2,4] → [4]
3. num_points: 10240 → 8192 → 4096
4. use_geodesic: true → false
```

## Quick Commands

**Debug run (2 minutes):**
```bash
# Edit config: steps_per_epoch_train=5, max_epoch=2
python scripts/train_domain_adaptation.py --config config.yml
```

**Dry run (no training):**
```bash
python scripts/train_domain_adaptation.py --config config.yml --dry-run
```

**Monitor training:**
```bash
tensorboard --logdir=./train_log
```

**Test after training:**
```bash
python scripts/run_pipeline.py --config config.yml --split=test
```

## Key Metrics to Monitor

During training, watch these in tensorboard:

1. **Segmentation loss** (should decrease steadily)
2. **CORAL loss** (should decrease and stabilize)
3. **CORAL weight** (should ramp from 0 to target)
4. **Source validation IoU** (should stay high, not drop >5%)
5. **Target validation IoU** (should improve if DA is working)

## Expected Timeline

**Conservative estimate:**
- Debug phase: 1 hour
- Pilot experiments: 4 hours
- Hyperparameter tuning: 1 day
- Final training: 1-2 days
- **Total**: 2-3 days for complete DA training

**Quick estimate (if everything works):**
- Debug: 30 min
- One pilot: 2 hours
- Final training: 1 day
- **Total**: 1.5 days
