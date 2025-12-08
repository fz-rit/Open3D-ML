import numpy as np
import os
from pathlib import Path
from os.path import join, exists
import logging

from .base_dataset import BaseDataset, BaseDatasetSplit
from ..utils import make_dir, DATASET
from open3d import io

log = logging.getLogger(__name__)


class DigiForestUnified(BaseDataset):
    """
    DigiForestUnified dataset for inference only.
    Contains .ply point cloud files with corresponding .labels files in a separate folder.
    
    Label mapping (same as Semantic3DUnified):
        0: Unlabeled
        1: Ground
        2: Trunk
        3: Canopy
        4: Understory
        5: Misc
    """

    def __init__(self,
                 dataset_path,
                 labels_path=None,
                 name='DigiForestUnified',
                 cache_dir='./logs/cache',
                 use_cache=False,
                 num_points=65536,
                 class_weights=[
                     250789, 2377612, 20488, 179050, 196894, 1173042
                 ],
                 ignored_label_inds=[0],
                 test_result_folder='./test',
                 **kwargs):
        """Initialize DigiForestUnified dataset.

        Args:
            dataset_path: Path to the directory containing .ply files.
            labels_path: Path to the directory containing .labels files. 
                        If None, assumes labels are in the same directory as .ply files.
            name: Dataset name.
            cache_dir: Directory for cache storage.
            use_cache: Whether to use caching.
            num_points: Maximum number of points per sample.
            class_weights: Class weights for training (not used for inference).
            ignored_label_inds: Labels to ignore during evaluation.
            test_result_folder: Folder to save test results.
        """
        super().__init__(dataset_path=dataset_path,
                         name=name,
                         cache_dir=cache_dir,
                         use_cache=use_cache,
                         class_weights=class_weights,
                         num_points=num_points,
                         ignored_label_inds=ignored_label_inds,
                         test_result_folder=test_result_folder,
                         **kwargs)

        cfg = self.cfg
        self.labels_path = labels_path if labels_path is not None else dataset_path

        self.label_to_names = self.get_label_to_names()
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([0])

        dataset_root = Path(self.cfg.dataset_path)
        labels_root = Path(self.labels_path)
        log.info(f"Dataset root path: {dataset_root}, exists: {dataset_root.exists()}")
        log.info(f"Labels root path: {labels_root}, exists: {labels_root.exists()}")
        
        # Collect all .ply files from the dataset root
        ply_files = sorted(dataset_root.glob('*.ply'))
        
        if len(ply_files) == 0:
            raise FileNotFoundError(
                f"No .ply files found in {dataset_root}\n"
                "Please ensure the dataset directory contains .ply point cloud files."
            )
        
        log.info(f"Found {len(ply_files)} PLY files.")
        
        # DigiForestUnified is test-only, so all files go to test split
        self.train_files = []
        self.val_files = []
        self.test_files = [str(f) for f in ply_files]
        
        log.info(f"Test files: {[Path(f).stem for f in self.test_files]}")

    @staticmethod
    def get_label_to_names():
        """Returns label to names mapping.
        
        Same as Semantic3DUnified to ensure compatibility with trained models.
        """
        label_to_names = {
            0: 'Unlabeled',
            1: 'Ground',
            2: 'Trunk',
            3: 'Canopy',
            4: 'Understory',
            5: 'Misc'
        }
        return label_to_names

    def get_split(self, split):
        """Returns a dataset split.

        Args:
            split: Split name ('test', 'testing', or 'all').

        Returns:
            A dataset split object.
        """
        return DigiForestUnifiedSplit(self, split=split)

    def get_split_list(self, split):
        """Returns the list of files for the requested split.

        Args:
            split: Split name ('test', 'testing', or 'all').

        Returns:
            List of file paths.

        Raises:
            ValueError: If split name is invalid.
        """
        if split in ['test', 'testing', 'all']:
            files = self.test_files
        elif split in ['train', 'training']:
            files = self.train_files
        elif split in ['val', 'validation']:
            files = self.val_files
        else:
            raise ValueError(f"Invalid split '{split}'. DigiForestUnified only supports 'test' split.")
        return files

    def is_tested(self, attr):
        """Checks if a datum has been tested.

        Args:
            attr: Attribute dictionary.

        Returns:
            True if result exists, False otherwise.
        """
        cfg = self.cfg
        name = attr['name']
        path = cfg.test_result_folder
        store_path = join(path, self.name, name + '.labels')
        if exists(store_path):
            print(f"{store_path} already exists.")
            return True
        else:
            return False

    def save_test_result(self, results, attr):
        """Saves inference results.

        Args:
            results: Model output with 'predict_labels'.
            attr: Sample attributes.
        """
        cfg = self.cfg
        name = attr['name'].split('.')[0]
        path = cfg.test_result_folder
        make_dir(path)

        # Shift predictions from 0-4 to 1-5 to match label IDs
        pred = results['predict_labels'] + 1
        store_path = join(path, self.name, name + '.labels')
        make_dir(Path(store_path).parent)
        np.savetxt(store_path, pred.astype(np.int32), fmt='%d')

        log.info(f"Saved {name} in {store_path}")


class DigiForestUnifiedSplit(BaseDatasetSplit):
    """Split class for DigiForestUnified dataset."""

    def __init__(self, dataset, split='test'):
        super().__init__(dataset, split=split)
        log.info(f"Found {len(self.path_list)} pointclouds for {split}")

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        """Load point cloud data from .ply file.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with 'point', 'feat', 'intensity', and 'label'.
        """
        ply_path = Path(self.path_list[idx])
        log.debug(f"get_data called {ply_path}")

        # Read PLY file using Open3D
        pcd = io.read_point_cloud(str(ply_path))
        
        # Extract coordinates
        points = np.asarray(pcd.points, dtype=np.float32)
        
        # Extract colors if available
        if pcd.has_colors():
            # Colors are in [0, 1] range, convert to [0, 255]
            colors = np.asarray(pcd.colors, dtype=np.float32) * 255.0
            feat = colors
        else:
            n_points = points.shape[0]
            feat = np.zeros((n_points, 3), dtype=np.float32)
            # log.warning(
            #     f"No color channels found in {ply_path.name}; using zeros for 'feat'."
            # )
        
        # Extract intensity if available (stored as a custom property)
        # If not available, use zeros
        if hasattr(pcd, 'point') and hasattr(pcd.point, 'intensity'):
            intensity = np.asarray(pcd.point.intensity, dtype=np.float32)
        else:
            n_points = points.shape[0]
            intensity = np.zeros(n_points, dtype=np.float32)
            log.debug(f"No intensity data found in {ply_path.name}; using zeros.")
        
        # Load labels from corresponding .labels file in labels directory
        labels_root = Path(self.dataset.labels_path)
        label_path = labels_root / (ply_path.stem + '.labels')
        
        if not exists(label_path):
            raise FileNotFoundError(
                f"Label file not found for: {ply_path.stem}\n"
                f"PLY file: {ply_path}\n"
                f"Expected label file: {label_path}\n"
                "Each .ply file must have a corresponding .labels file in the labels directory."
            )
        
        # Read labels (one integer per line)
        labels = np.loadtxt(label_path, dtype=np.int32)
        
        # Remap label 5 (Misc) to 0 (Unlabeled) for unified ignoring
        labels[labels == 5] = 0
        
        # Validate data shapes
        n_points = points.shape[0]
        if labels.shape[0] != n_points:
            raise ValueError(
                f"Shape mismatch for {ply_path.stem}: "
                f"points={n_points}, labels={labels.shape[0]}"
            )

        data = {
            'point': points,
            'feat': feat,
            'intensity': intensity,
            'label': labels
        }

        return data

    def get_attr(self, idx):
        """Get sample attributes.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with sample metadata.
        """
        ply_path = Path(self.path_list[idx])
        name = ply_path.stem
        split = self.split
        attr = {'idx': idx, 'name': name, 'path': str(ply_path), 'split': split}
        return attr


DATASET._register_module(DigiForestUnified)
