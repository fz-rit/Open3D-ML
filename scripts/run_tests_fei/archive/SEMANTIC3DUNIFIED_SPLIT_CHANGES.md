# Semantic3DUnified Dataset Split Changes

## Summary
Modified the `Semantic3DUnified` dataset class to support dynamic train/validation/test splitting based on configurable ratios instead of requiring predefined train/test subfolders.

## Key Changes

### 1. New Dataset Structure Support
- **Before**: Required `train/` and `test/` subfolders with point cloud files
- **After**: All `.txt` files can be in a single root folder (e.g., `/dataset_path/*.txt`)

### 2. Dynamic Split Ratios
Added new parameters to control dataset splitting:
- `train_ratio`: Proportion of data for training (default: 0.7)
- `val_ratio`: Proportion of data for validation (default: 0.15)
- `test_ratio`: Proportion of data for testing (default: 0.15)
- `split_seed`: Random seed for reproducible splits (default: 42)

### 3. Automatic Split Generation
The dataset now:
1. Collects all `.txt` files from the dataset root directory
2. Validates that ratios sum to 1.0 (normalizes if they don't)
3. Randomly shuffles files using the specified seed
4. Splits them according to the ratios
5. Logs the split assignment for transparency

### 4. Flexible Label File Locations
Updated `get_data()` to search for label files in multiple locations:
1. Same directory as `.txt` file (e.g., `/dataset_path/file.labels`)
2. `semantic3d_remapped_labels/` subdirectory (legacy structure)
3. `labels/` subdirectory

## Configuration Example

In `randlanet_semantic3dunified_xyz.yml`:

```yaml
dataset:
  name: Semantic3DUnified
  dataset_path: /shared/rc/mangrove/data/Semantic3D/all_remapped/
  # Dynamic split ratios (must sum to 1.0)
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  split_seed: 42
```

## Usage

### Directory Structure
```
/dataset_path/
  ├── scan1.txt
  ├── scan1.labels
  ├── scan2.txt
  ├── scan2.labels
  ├── scan3.txt
  ├── scan3.labels
  └── ...
```

### Customizing Splits
To change the split ratios, modify the config file:

```yaml
# 80% train, 10% val, 10% test
train_ratio: 0.8
val_ratio: 0.1
test_ratio: 0.1

# Use a different random seed
split_seed: 123
```

### Reproducibility
The `split_seed` parameter ensures that the same split is generated across runs. Change the seed to get a different random split.

## Backward Compatibility
- The `val_files` parameter is still accepted but deprecated
- Legacy folder structures with `train/` and `test/` subfolders are no longer supported
- All files should be in the same root directory

## Logging
The dataset now logs:
- Total number of files found
- Number of files in each split
- File names assigned to each split

This helps verify the split is working as expected.

## Benefits
1. **Flexibility**: Easy to adjust train/val/test proportions
2. **Simplicity**: No need to manually organize files into subfolders
3. **Reproducibility**: Seed control ensures consistent splits
4. **Transparency**: Detailed logging shows split assignments
5. **Robustness**: Multiple search paths for label files
