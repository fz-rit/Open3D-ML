import numpy as np
import os
from pathlib import Path
from os.path import join, exists
import logging

try:
    import laspy
except ImportError:
    raise ImportError("laspy is required for ForestSemantic dataset. Install with: pip install laspy")

from .base_dataset import BaseDataset, BaseDatasetSplit
from ..utils import make_dir, DATASET

log = logging.getLogger(__name__)


class ForestSemantic(BaseDataset):
    """
    ForestSemantic dataset for inference only.
    Contains .las point cloud files with corresponding .labels files.
    
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
                 name='ForestSemantic',
                 cache_dir='./logs/cache',
                 use_cache=False,
                 num_points=65536,
                 class_weights=[
                     250789, 2377612, 20488, 179050, 196894, 1173042
                 ],
                 ignored_label_inds=[0],
                 test_result_folder='./test',
                 **kwargs):
        """Initialize ForestSemantic dataset.

        Args:
            dataset_path: Path to the dataset directory containing .las and .labels files.
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

        self.label_to_names = self.get_label_to_names()
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([0])

        dataset_root = Path(self.cfg.dataset_path)
        log.info(f"Dataset root path: {dataset_root}, exists: {dataset_root.exists()}")
        
        # Collect all .las files from the dataset root
        las_files = sorted(dataset_root.glob('*.las'))
        
        if len(las_files) == 0:
            raise FileNotFoundError(
                f"No .las files found in {dataset_root}\n"
                "Please ensure the dataset directory contains .las point cloud files."
            )
        
        log.info(f"Found {len(las_files)} LAS files.")
        
        # ForestSemantic is test-only, so all files go to test split
        self.train_files = []
        self.val_files = []
        self.test_files = [str(f) for f in las_files]
        
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
        return ForestSemanticSplit(self, split=split)

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
            raise ValueError(f"Invalid split '{split}'. ForestSemantic only supports 'test' split.")
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


class ForestSemanticSplit(BaseDatasetSplit):
    """Split class for ForestSemantic dataset."""

    def __init__(self, dataset, split='test'):
        super().__init__(dataset, split=split)
        log.info(f"Found {len(self.path_list)} pointclouds for {split}")

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        """Load point cloud data from .las file.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with 'point', 'feat', 'intensity', and 'label'.
        """
        las_path = Path(self.path_list[idx])
        log.debug(f"get_data called {las_path}")

        # Read LAS file
        las = laspy.read(las_path)
        
        # Extract coordinates
        points = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)
        
        n_points = points.shape[0]
        feat = np.zeros((n_points, 3), dtype=np.float32)
        log.warning(
            f"No RGB channels found in {las_path.name}; using zeros for 'feat'."
        )
        
        # Extract intensity
        intensity = las.intensity.astype(np.float32)
        
        # Load labels from corresponding .labels file
        label_path = las_path.with_suffix('.labels')
        
        if not exists(label_path):
            raise FileNotFoundError(
                f"Label file not found for: {las_path.stem}\n"
                f"LAS file: {las_path}\n"
                f"Expected label file: {label_path}\n"
                "Each .las file must have a corresponding .labels file."
            )
        
        # Read labels (one integer per line)
        labels = np.loadtxt(label_path, dtype=np.int32)
        
        # Validate data shapes
        n_points = points.shape[0]
        if labels.shape[0] != n_points:
            raise ValueError(
                f"Shape mismatch for {las_path.stem}: "
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
        las_path = Path(self.path_list[idx])
        name = las_path.stem
        split = self.split
        attr = {'idx': idx, 'name': name, 'path': str(las_path), 'split': split}
        return attr


DATASET._register_module(ForestSemantic)
