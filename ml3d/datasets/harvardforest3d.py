import numpy as np
from pathlib import Path
import glob
import json
import logging

try:
    import laspy
    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False

from .base_dataset import BaseDataset, BaseDatasetSplit
from ..utils import make_dir, DATASET

log = logging.getLogger(__name__)


class HarvardForest3D(BaseDataset):
    """Dataset loader for the HarvardForest3D dataset.

    This dataset is designed for self-supervised learning tasks (e.g., rotation prediction)
    as it does not contain ground truth semantic labels.

    Expected folder structure under ``dataset_path``:

    dataset_path/
      ├── S01_000.las
      ├── S01_001.las
      ├── ...
      ├── S07_009.las
      ├── colormap.json  # optional metadata
      └── rename_map.json  # optional metadata

    LAS files should contain at minimum:
      - X, Y, Z coordinates (scaled)
      - Optionally: intensity, RGB, classification, etc.

    Since this dataset is for SSL, no label files are required.
    """

    def __init__(
        self,
        dataset_path,
        name='HarvardForest3D',
        cache_dir='./logs/cache',
        use_cache=False,
        num_points=65536,
        class_weights=None,
        ignored_label_inds=[],
        val_split_ratio=0.1,
        val_split_seed=42,
        test_split_ratio=0.0,  # No separate test set by default
        test_result_folder='./test',
        use_intensity=True,
        use_rgb=False,
        **kwargs,
    ):
        """Initialize HarvardForest3D dataset configuration.

        Args:
            dataset_path: Root path to HarvardForest3D dataset containing .las files.
            val_split_ratio: Fraction of files to use for validation (deterministic).
            val_split_seed: Random seed for reproducible train/val split.
            test_split_ratio: Fraction for test set (default 0, no test split).
            use_intensity: If True and available, include intensity as a feature.
            use_rgb: If True and available, include RGB colors as features.
        """
        if not HAS_LASPY:
            raise ImportError(
                "laspy is required for HarvardForest3D dataset. "
                "Install it with: pip install laspy"
            )

        super().__init__(
            dataset_path=dataset_path,
            name=name,
            cache_dir=cache_dir,
            use_cache=use_cache,
            class_weights=class_weights,
            num_points=num_points,
            ignored_label_inds=ignored_label_inds,
            test_result_folder=test_result_folder,
            **kwargs,
        )

        cfg = self.cfg

        self.val_split_ratio = val_split_ratio
        self.val_split_seed = val_split_seed
        self.test_split_ratio = test_split_ratio
        self.use_intensity = use_intensity
        self.use_rgb = use_rgb

        # No semantic classes for SSL tasks
        self.label_to_names = {0: 'unlabeled'}
        self.label_values = np.array([0], dtype=np.int32)
        self.label_to_idx = {0: 0}
        self.ignored_labels = np.array(ignored_label_inds, dtype=np.int64)
        self.num_classes = 1  # Dummy for SSL

        root = Path(self.cfg.dataset_path)

        # Collect all .las files
        all_las_files = sorted(glob.glob(str(root / '*.las')))
        if len(all_las_files) == 0:
            raise FileNotFoundError(
                f"No .las files found in {root}. Please check dataset_path."
            )

        log.info(f"Found {len(all_las_files)} .las files in {root}")

        # Deterministic split with persistence in cache_dir
        cache_dir_path = Path(getattr(self.cfg, 'cache_dir', './logs/cache'))
        cache_dir_path.mkdir(parents=True, exist_ok=True)
        manifest_path = cache_dir_path / f"{self.name}_split.json"

        # Build basenames for stability across absolute path changes
        basenames = [Path(p).name for p in all_las_files]
        basename_to_full = {Path(p).name: p for p in all_las_files}

        if manifest_path.exists():
            try:
                with open(manifest_path, 'r') as f:
                    split_info = json.load(f)
                saved_train = split_info.get('train', [])
                saved_val = split_info.get('val', [])
                saved_test = split_info.get('test', [])
                saved_all = set(saved_train) | set(saved_val) | set(saved_test)
                
                if set(basenames) == saved_all:
                    # Reconstruct paths
                    self.train_files = np.array([basename_to_full[b] for b in saved_train])
                    self.val_files = [basename_to_full[b] for b in saved_val]
                    self.test_files = np.array([basename_to_full[b] for b in saved_test])
                    log.info("Loaded existing train/val/test split from manifest.")
                else:
                    log.warning("Split manifest does not match current file list. Recomputing split.")
                    raise ValueError('mismatch')
            except Exception as e:
                log.info(f"Recomputing split due to: {e}")
                self._create_split(basenames, basename_to_full, manifest_path)
        else:
            # Create new split
            self._create_split(basenames, basename_to_full, manifest_path)

        log.info(
            f"HarvardForest3D: train={len(self.train_files)} "
            f"val={len(self.val_files)} test={len(self.test_files)}"
        )

    def _create_split(self, basenames, basename_to_full, manifest_path):
        """Create and persist a deterministic train/val/test split."""
        rng = np.random.default_rng(self.val_split_seed)
        indices = np.arange(len(basenames))
        rng.shuffle(indices)

        n_test = max(0, int(len(indices) * float(self.test_split_ratio)))
        n_val = max(1, int(len(indices) * float(self.val_split_ratio)))

        test_idx = set(indices[:n_test].tolist()) if n_test > 0 else set()
        val_idx = set(indices[n_test:n_test + n_val].tolist())
        train_idx = set(range(len(basenames))) - test_idx - val_idx

        train_b = sorted([basenames[i] for i in train_idx])
        val_b = sorted([basenames[i] for i in val_idx])
        test_b = sorted([basenames[i] for i in test_idx])

        split_info = {
            'train': train_b,
            'val': val_b,
            'test': test_b,
            'seed': self.val_split_seed,
            'val_ratio': float(self.val_split_ratio),
            'test_ratio': float(self.test_split_ratio),
        }

        with open(manifest_path, 'w') as f:
            json.dump(split_info, f, indent=2)

        self.train_files = np.array([basename_to_full[b] for b in train_b])
        self.val_files = [basename_to_full[b] for b in val_b]
        self.test_files = np.array([basename_to_full[b] for b in test_b])

        log.info(f"Created new split and saved to {manifest_path}")

    @staticmethod
    def get_label_to_names():
        """Default label mapping (dummy for SSL)."""
        return {0: 'unlabeled'}

    def get_split(self, split):
        return HarvardForest3DSplit(self, split=split)

    def get_split_list(self, split):
        if split in ['test', 'testing']:
            files = list(self.test_files)
        elif split in ['train', 'training']:
            files = list(self.train_files)
        elif split in ['val', 'validation']:
            files = list(self.val_files)
        elif split in ['all']:
            files = list(self.train_files) + list(self.val_files) + list(self.test_files)
        else:
            raise ValueError(f"Invalid split {split}")
        return files

    def is_tested(self, attr):
        # For SSL tasks, we don't save test predictions in the same way
        return False

    def save_test_result(self, results, attr):
        # Placeholder for SSL tasks
        cfg = self.cfg
        name = attr['name']
        path = cfg.test_result_folder
        make_dir(path)
        log.info(f"SSL task completed for {name}. No semantic predictions to save.")


class HarvardForest3DSplit(BaseDatasetSplit):
    """Split wrapper for HarvardForest3D."""

    def __init__(self, dataset, split='training'):
        super().__init__(dataset, split=split)
        log.info(f"Found {len(self.path_list)} LAS files for {split}")

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        las_path = Path(self.path_list[idx])
        log.debug(f"get_data called for {las_path}")

        # Read LAS file
        try:
            las = laspy.read(str(las_path))
        except Exception as e:
            raise RuntimeError(f"Failed to read LAS file {las_path}: {e}")

        # Extract coordinates (laspy v2+ returns scaled coords by default)
        points = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)

        # Build feature vector
        feat_list = []

        # Intensity
        if self.dataset.use_intensity:
            try:
                intensity = np.array(las.intensity, dtype=np.float32)
                # Normalize intensity to [0, 1] range (typical LAS intensity is 0-65535)
                intensity = intensity / 65535.0
                feat_list.append(intensity.reshape(-1, 1))
            except AttributeError:
                log.warning(f"Intensity not found in {las_path}, using zeros.")
                feat_list.append(np.zeros((points.shape[0], 1), dtype=np.float32))
        
        # RGB colors
        if self.dataset.use_rgb:
            try:
                # LAS RGB is typically 0-65535
                r = np.array(las.red, dtype=np.float32) / 65535.0
                g = np.array(las.green, dtype=np.float32) / 65535.0
                b = np.array(las.blue, dtype=np.float32) / 65535.0
                rgb = np.column_stack([r, g, b])
                feat_list.append(rgb)
            except AttributeError:
                log.warning(f"RGB not found in {las_path}, using zeros.")
                feat_list.append(np.zeros((points.shape[0], 3), dtype=np.float32))

        # If no features selected, use ones
        if len(feat_list) == 0:
            feat = np.ones((points.shape[0], 1), dtype=np.float32)
        else:
            feat = np.concatenate(feat_list, axis=1)

        # Dummy labels (0-based, all zeros for SSL)
        labels = np.zeros((points.shape[0],), dtype=np.int32)

        # For intensity compatibility with models expecting separate intensity
        if self.dataset.use_intensity and len(feat_list) > 0:
            intensity = feat[:, 0]
        else:
            intensity = np.ones((points.shape[0],), dtype=np.float32)

        data = {
            'point': points,
            'feat': feat,
            'intensity': intensity,
            'label': labels,
        }
        return data

    def get_attr(self, idx):
        las_path = Path(self.path_list[idx])
        name = las_path.stem
        split = self.split
        attr = {'idx': idx, 'name': name, 'path': str(las_path), 'split': split}
        return attr


DATASET._register_module(HarvardForest3D)
