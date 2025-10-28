# LAS File Export for Inference Results

## Overview

The inference script now supports exporting prediction results as `.las` files, which include:
- Original point cloud coordinates (X, Y, Z)
- RGB colors (if available)
- Intensity values (if available)
- Ground truth labels (stored in `classification` field)
- Predicted labels (stored in `user_data` field)

## Installation

To use the LAS export feature, install the `laspy` library:

```bash
pip install laspy
```

## Usage

Add the `--save-las` argument to specify the output directory:

```bash
python scripts/run_tests_fei/inference_randlanet_semantic3d.py \
    --config ml3d/configs/randlanet_semantic3d_p3.yml \
    --indices 0 1 2 \
    --save-las ./output/las_files
```

### Combined with other options

```bash
# Save LAS files + compute metrics + visualize
python scripts/run_tests_fei/inference_randlanet_semantic3d.py \
    --config ml3d/configs/randlanet_semantic3d_p3.yml \
    --indices 0 1 2 \
    --save-las /home/fzhcis/mylab/data/semantic3d/open3d_randlanet/test_1027 \
    --metrics \
    --visualize
```

## LAS File Structure

Each output `.las` file contains:

| Field | Description | Value Range |
|-------|-------------|-------------|
| `x`, `y`, `z` | Point coordinates | Original point cloud coordinates |
| `red`, `green`, `blue` | RGB colors | 0-65535 (scaled from 0-255) |
| `intensity` | Intensity values | 0-65535 |
| `classification` | Ground truth labels | 0-8 (Semantic3D label IDs) |
| `user_data` | Predicted labels | 1-8 (Semantic3D label IDs) |

### Label Mapping (Semantic3D)

- **0**: unlabeled
- **1**: man-made terrain
- **2**: natural terrain
- **3**: high vegetation
- **4**: low vegetation
- **5**: buildings
- **6**: hard scape
- **7**: scanning artefacts
- **8**: cars

### Important Notes

1. **Ground Truth (`classification` field)**: Contains original Semantic3D IDs (0-8), where 0 means unlabeled
2. **Predictions (`user_data` field)**: Contains predicted Semantic3D IDs (1-8). The model outputs 0-7 internally, which are shifted to 1-8 to match Semantic3D convention

## Reading LAS Files

### Using Python (laspy)

```python
import laspy
import numpy as np

# Read the LAS file
las = laspy.read("output/las_files/sample_predictions.las")

# Extract data
points = np.vstack([las.x, las.y, las.z]).T
rgb = np.vstack([las.red, las.green, las.blue]).T / 257  # Scale back to 0-255
intensity = las.intensity
gt_labels = las.classification
pred_labels = las.user_data

print(f"Points: {points.shape}")
print(f"Unique GT labels: {np.unique(gt_labels)}")
print(f"Unique predicted labels: {np.unique(pred_labels)}")
```

### Using CloudCompare

1. Open CloudCompare
2. File → Open → Select the `.las` file
3. The point cloud will display with its original RGB colors
4. View labels:
   - Edit → Scalar Fields → Select `Classification` (for ground truth)
   - Edit → Scalar Fields → Select `UserData` (for predictions)

### Using PDAL

```bash
# View LAS file info
pdal info sample_predictions.las

# Convert to other formats
pdal translate sample_predictions.las sample_predictions.ply
```

## Output Example

When running with `--save-las ./output`:

```
output/
  └── las_files/
      ├── sample1_predictions.las
      ├── sample2_predictions.las
      └── sample3_predictions.las
```

Each file will be named using the format: `{sample_name}_predictions.las`
