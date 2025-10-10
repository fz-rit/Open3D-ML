import numpy as np
import pandas as pd
from pathlib import Path
from os.path import join, exists
import glob
import logging

from .base_dataset import BaseDataset, BaseDatasetSplit
from ..utils import make_dir, DATASET

log = logging.getLogger(__name__)


class Mangrove3D(BaseDataset):
    """Dataset loader for the Mangrove3D dataset.

    Expected folder structure under ``dataset_path``:

    dataset_path/
      ├── train_val/
      │    ├── pcd/*.csv
      │    └── label/*.labels  # one label per line, aligned with points
      └── test/
           ├── pcd/*.csv
           └── label/          # optional, not required for inference

    CSV columns: ['X', 'Y', 'Z', 'zenith', 'azimuth', 'rangemeter', 'Intensity',
    'elevation', 'curvature', 'anisotropy', 'planarity', 'nx', 'ny', 'nz',
    'intensity_adjusted', 'z_adjusted', 'range_adjusted', 'Pseudo-Rn', 'Pseudo-Gn',
    'Pseudo-Bn', 'PCA1', 'PCA2', 'PCA3']

    Feature columns (optional): from 'rangemeter' to 'PCA3' inclusive.
    """

    def __init__(
        self,
        dataset_path,
        name='Mangrove3D',
        cache_dir='./logs/cache',
        use_cache=False,
        num_points=65536,
        class_weights=None,
        ignored_label_inds=[0],
        val_files=None,
        test_result_folder='./test',
        use_features=True,
        feature_first_col='rangemeter',
        feature_last_col='PCA3',
        label_dir_name='label',
        pcd_dir_name='pcd',
    label_ext='.label',
        save_label_offset=0,
        label_to_names=None,
        **kwargs,
    ):
        """Initialize dataset configuration.

        Args:
            dataset_path: Root path to Mangrove3D dataset containing 'train_val' and 'test' folders.
            use_features: If True, include feature columns [feature_first_col..feature_last_col].
            feature_first_col: Name of the first feature column in the CSV (inclusive).
            feature_last_col: Name of the last feature column in the CSV (inclusive).
            label_dir_name: Subfolder name holding label files under each split folder.
            pcd_dir_name: Subfolder name holding CSV point clouds under each split folder.
            label_ext: Extension for label files.
            save_label_offset: Offset added to predictions when saving (default 0; set to 1 if your ground truth labels are 1-based).
            label_to_names: Optional dict mapping label ids to class names. If None, a generic mapping is used {0:'unlabeled', 1:'class1', ...} determined at runtime when possible.
        """
        super().__init__(
            dataset_path=dataset_path,
            name=name,
            cache_dir=cache_dir,
            use_cache=use_cache,
            class_weights=class_weights,
            num_points=num_points,
            ignored_label_inds=ignored_label_inds,
            val_files=val_files or [],
            test_result_folder=test_result_folder,
            **kwargs,
        )

        cfg = self.cfg

        # Feature configuration
        self.use_features = use_features
        self.feature_first_col = feature_first_col
        self.feature_last_col = feature_last_col
        self.pcd_dir_name = pcd_dir_name
        self.label_dir_name = label_dir_name
        self.label_ext = label_ext
        self.save_label_offset = save_label_offset

        # Label mapping
        if label_to_names is not None:
            self.label_to_names = label_to_names
        elif hasattr(cfg, 'label_to_names') and cfg.label_to_names:
            self.label_to_names = cfg.label_to_names
        else:
            # Fallback generic mapping; refined later in split if we can detect max label
            self.label_to_names = {0: 'unlabeled'}
        self.label_values = np.sort([k for k in self.label_to_names.keys()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array(ignored_label_inds, dtype=np.int64)

        root = Path(self.cfg.dataset_path)

        # Collect files
        self.trainval_pcd_files = np.sort(
            glob.glob(str(root / 'train_val' / self.pcd_dir_name / '*.csv'))
        )
        self.test_pcd_files = np.sort(
            glob.glob(str(root / 'test' / self.pcd_dir_name / '*.csv'))
        )

        # No implicit validation set unless specified via val_files (list of base names or substrings)
        self.val_files = []
        if cfg.val_files:
            val_markers = cfg.val_files
            for f in self.trainval_pcd_files:
                if any(marker in f for marker in val_markers):
                    self.val_files.append(f)

        self.train_files = np.array([f for f in self.trainval_pcd_files if f not in self.val_files])
        self.test_files = np.array(self.test_pcd_files)

        log.info(
            f"Mangrove3D: train={len(self.train_files)} val={len(self.val_files)} test={len(self.test_files)}"
        )

    @staticmethod
    def get_label_to_names():
        """Default label mapping.

        Override via constructor arg `label_to_names` or cfg.label_to_names.
        """
        return {0: 'unlabeled'}

    def get_split(self, split):
        return Mangrove3DSplit(self, split=split)

    def get_split_list(self, split):
        if split in ['test', 'testing']:
            files = list(self.test_files)
        elif split in ['train', 'training']:
            files = list(self.train_files)
        elif split in ['val', 'validation']:
            files = list(self.val_files)
        elif split in ['all']:
            files = list(self.val_files) + list(self.train_files) + list(self.test_files)
        else:
            raise ValueError(f"Invalid split {split}")
        return files

    def is_tested(self, attr):
        cfg = self.cfg
        name = attr['name']
        path = cfg.test_result_folder
        store_path = join(path, self.name, name + self.label_ext)
        if exists(store_path):
            log.info(f"{store_path} already exists.")
            return True
        else:
            return False

    def save_test_result(self, results, attr):
        cfg = self.cfg
        name = attr['name']
        path = cfg.test_result_folder
        make_dir(path)

        pred = results['predict_labels'] + self.save_label_offset
        store_path = join(path, self.name, name + self.label_ext)
        make_dir(Path(store_path).parent)
        np.savetxt(store_path, pred.astype(np.int32), fmt='%d')
        log.info(f"Saved {name} in {store_path}.")


class Mangrove3DSplit(BaseDatasetSplit):
    """Split wrapper for Mangrove3D."""

    def __init__(self, dataset, split='training'):
        super().__init__(dataset, split=split)
        log.info("Found {} pointclouds for {}".format(len(self.path_list), split))

    def __len__(self):
        return len(self.path_list)

    def _get_label_path(self, csv_path: Path) -> Path:
        # Map CSV file path to corresponding label path
        csv_path = Path(csv_path)
        # Replace pcd directory with label directory under the same split folder
        if 'train_val' in csv_path.parts:
            split_root = csv_path.parents[1]  # .../train_val
        elif 'test' in csv_path.parts:
            split_root = csv_path.parents[1]  # .../test
        else:
            # Fallback to parent of pcd folder
            split_root = csv_path.parents[1]
        label_dir = split_root / self.dataset.label_dir_name
        label_path = label_dir / (csv_path.stem + self.dataset.label_ext)
        return label_path

    def get_data(self, idx):
        csv_path = Path(self.path_list[idx])
        log.debug("get_data called {}".format(str(csv_path)))

        # Read CSV with headers
        df = pd.read_csv(csv_path)

        # Normalize column names by stripping spaces
        df.columns = [str(c).strip() for c in df.columns]

        # Required coordinate columns
        for col in ['X', 'Y', 'Z']:
            if col not in df.columns:
                raise KeyError(f"Required column '{col}' not found in {csv_path}")
        points = df[['X', 'Y', 'Z']].values.astype(np.float32)

        # Intensity column is required
        if 'Intensity' not in df.columns:
            raise KeyError(f"Required column 'Intensity' not found in {csv_path}")
        intensity = df['Intensity'].values.astype(np.float32)

        # Features selection
        feat = None
        if self.dataset.use_features:
            cols = list(df.columns)
            try:
                start_idx = cols.index(self.dataset.feature_first_col)
                end_idx = cols.index(self.dataset.feature_last_col)
                if end_idx < start_idx:
                    raise ValueError
                feat_cols = cols[start_idx:end_idx + 1]
                feat = df[feat_cols].values.astype(np.float32)
            except (ValueError, IndexError):
                raise KeyError(
                    f"Feature columns from '{self.dataset.feature_first_col}' to "
                    f"'{self.dataset.feature_last_col}' not found or invalid in {csv_path}."
                )
        else:
            # Minimal feature: use Intensity only
            feat = intensity.reshape(-1, 1).astype(np.float32)

        # Labels
        if self.split != 'test':
            label_path = self._get_label_path(csv_path)
            if not label_path.exists():
                raise FileNotFoundError(f"Label file not found for {csv_path}: {label_path}")
            labels = pd.read_csv(label_path, header=None, sep='\s+', dtype=np.int32).values
            labels = labels.squeeze().astype(np.int32)
            if labels.ndim != 1 or labels.shape[0] != points.shape[0]:
                raise ValueError(
                    f"Label shape mismatch for {csv_path}: got {labels.shape}, expected ({points.shape[0]},)"
                )
        else:
            labels = np.zeros((points.shape[0],), dtype=np.int32)

        data = {
            'point': points,
            'feat': feat,
            'intensity': intensity,
            'label': labels,
        }
        return data

    def get_attr(self, idx):
        csv_path = Path(self.path_list[idx])
        name = csv_path.stem
        split = self.split
        attr = {'idx': idx, 'name': name, 'path': str(csv_path), 'split': split}
        return attr


DATASET._register_module(Mangrove3D)
