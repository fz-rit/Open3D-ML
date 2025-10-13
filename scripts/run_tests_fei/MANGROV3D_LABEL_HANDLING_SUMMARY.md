# Label Handling in Mangrove3D Dataset

## Overview
The Mangrove3D dataset uses a simple label conversion system to handle the mismatch between file format (1-based) and PyTorch requirements (0-based).

## Label Conversion Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        Label Workflow                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input .label files:     1, 2, 3, 4, 5  (1-based)             │
│           ↓                                                     │
│  _read_labels():         subtract 1                            │
│           ↓                                                     │
│  Training/Eval:          0, 1, 2, 3, 4  (0-based)             │
│           ↓                                                     │
│  Predictions:            0, 1, 2, 3, 4  (0-based)             │
│           ↓                                                     │
│  save_test_result():     add 1                                 │
│           ↓                                                     │
│  Output .label files:    1, 2, 3, 4, 5  (1-based)             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Class Mapping

| File Label (1-based) | Internal Label (0-based) | Class Name       |
|---------------------|--------------------------|------------------|
| 1                   | 0                        | Ground_and_water |
| 2                   | 1                        | Stem             |
| 3                   | 2                        | Canopy           |
| 4                   | 3                        | Roots            |
| 5                   | 4                        | Object           |

## Implementation Details

### 1. Reading Labels (`_read_labels()` in `mangrove3d.py`)
```python
def _read_labels(self, label_path: Path) -> np.ndarray:
    """Convert 1-based file labels to 0-based internal labels."""
    labels = pd.read_csv(label_path, header=None, sep=r'\s+', dtype=np.int32).values
    labels = labels.squeeze().astype(np.int32)
    # Convert from 1-based (file format) to 0-based (internal format)
    labels = labels - 1
    return labels
```

### 2. Saving Predictions (`save_test_result()` in `mangrove3d.py`)
```python
def save_test_result(self, results, attr):
    """Convert 0-based predictions back to 1-based for output files."""
    # Convert predictions back to 1-based format for output files
    pred = results['predict_labels'] + 1
    # Save to file...
```

### 3. Configuration (`randlanet_mangrove3d.yml`)
```yaml
dataset:
  label_to_names:
    0: Ground_and_water  # 0-based for internal use
    1: Stem
    2: Canopy
    3: Roots
    4: Object

model:
  num_classes: 5  # Must match number of classes
```

## Important Notes

1. **Cache**: When enabling cache (`use_cache: true`), ensure to clear the cache directory after changing label handling code to rebuild with correct labels.

2. **Validation**: The dataset includes automatic validation to ensure labels are in the valid range [0, num_classes-1] after conversion.

3. **num_workers**: Set to 0 due to unpicklable local functions in the sampler. This has minimal performance impact when caching is enabled.

## Troubleshooting

### CUDA Assertion Error: "index out of bounds"
**Cause**: Labels outside the valid range [0, num_classes-1]

**Solutions**:
- Clear cache directory: `rm -rf ./logs/cache_mangrove3d/*`
- Verify file labels are 1-based (1,2,3,4,5)
- Check `num_classes` matches the number of classes

### Segmentation Fault with num_workers > 0s
**Cause**: Unpicklable local functions in `SemSegRandomSampler`

**Solution**: Use `num_workers: 0` (minimal performance impact with caching)

