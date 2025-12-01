import numpy as np
import pandas as pd
import os, sys, glob, pickle
from pathlib import Path
from os.path import join, exists, dirname, abspath
from sklearn.neighbors import KDTree
import logging

from .utils import DataProcessing as DP
from .base_dataset import BaseDataset, BaseDatasetSplit
from ..utils import make_dir, DATASET

log = logging.getLogger(__name__)


class Semantic3DUnified(BaseDataset):
    """
    This dataset is a unified version of the Semantic3D dataset.
    Label mapping is as follows:
        0: Unlabeled
        1: Ground
        2: Trunk
        3: Canopy
        4: Understory
        5: Misc
    
    """

    def __init__(self,
                 dataset_path,
                 name='Semantic3DUnified',
                 cache_dir='./logs/cache',
                 use_cache=False,
                 num_points=65536,
                 class_counts4weight=[
                     2377612, 0, 199539, 196894, 1173042
                 ],
                 ignored_label_inds=[0],
                 test_result_folder='./test',
                 train_ratio=0.7,
                 val_ratio=0.15,
                 test_ratio=0.15,
                 split_seed=42,
                 **kwargs):
        """Initialize the function by passing the dataset and other details.

        Args:
            dataset_path: The path to the dataset to use.
            name: The name of the dataset (Semantic3DUnified in this case).
            cache_dir: The directory where the cache is stored.
            use_cache: Indicates if the dataset should be cached.
            num_points: The maximum number of points to use when splitting the dataset.
            class_counts4weight: Per-class sample counts used to derive loss weights.
            ignored_label_inds: A list of labels that should be ignored in the dataset.
            test_result_folder: The folder where the test results should be stored.
            train_ratio: Ratio of data to use for training (default: 0.7).
            val_ratio: Ratio of data to use for validation (default: 0.15).
            test_ratio: Ratio of data to use for testing (default: 0.15).
            split_seed: Random seed for splitting the dataset (default: 42).

        Returns:
            class: The corresponding class.
        """
        super().__init__(dataset_path=dataset_path,
                         name=name,
                         cache_dir=cache_dir,
                         use_cache=use_cache,
                         class_counts4weight=class_counts4weight,
                         num_points=num_points,
                         ignored_label_inds=ignored_label_inds,
                         test_result_folder=test_result_folder,
                         train_ratio=train_ratio,
                         val_ratio=val_ratio,
                         test_ratio=test_ratio,
                         split_seed=split_seed,
                         **kwargs)

        cfg = self.cfg

        self.label_to_names = self.get_label_to_names()
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([0])

        dataset_root = Path(self.cfg.dataset_path)
        log.info(f"Dataset root path: {dataset_root}, exists: {dataset_root.exists()}")
        
        # Collect all .txt files from the dataset root (no subfolders)
        all_paths = sorted(dataset_root.glob('*.txt'))
        
        log.info(f"Found {len(all_paths)} total point cloud files.")
        
        # Validate ratios
        total_ratio = cfg.train_ratio + cfg.val_ratio + cfg.test_ratio
        if not np.isclose(total_ratio, 1.0):
            log.warning(f"Train/val/test ratios sum to {total_ratio}, normalizing to 1.0")
            cfg.train_ratio /= total_ratio
            cfg.val_ratio /= total_ratio
            cfg.test_ratio /= total_ratio
        
        # Split files based on ratios
        np.random.seed(cfg.split_seed)
        indices = np.random.permutation(len(all_paths))
        
        n_train = int(len(all_paths) * cfg.train_ratio)
        n_val = int(len(all_paths) * cfg.val_ratio)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        self.train_files = sorted([str(all_paths[i]) for i in train_indices])
        self.val_files = sorted([str(all_paths[i]) for i in val_indices])
        self.test_files = sorted([str(all_paths[i]) for i in test_indices])
        
        log.info(f"Split: {len(self.train_files)} train, {len(self.val_files)} val, {len(self.test_files)} test files")
        log.info(f"Train files: {[Path(f).stem for f in self.train_files]}")
        log.info(f"Val files: {[Path(f).stem for f in self.val_files]}")
        log.info(f"Test files: {[Path(f).stem for f in self.test_files]}")

    @staticmethod
    def get_label_to_names():
        """Returns a label to names dictionary object.

        Returns:
            A dict where keys are label numbers and
            values are the corresponding names.
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
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.
        """
        return Semantic3DUnifiedSplit(self, split=split)

    def get_split_list(self, split):
        """Returns the list of data splits available.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.

        Raises:
            ValueError: Indicates that the split name passed is incorrect. The split name should be one of
            'training', 'test', 'validation', or 'all'.
        """
        if split in ['test', 'testing']:
            files = self.test_files
        elif split in ['train', 'training']:
            files = self.train_files
        elif split in ['val', 'validation']:
            files = self.val_files
        elif split in ['all']:
            files = self.val_files + self.train_files + self.test_files
        else:
            raise ValueError("Invalid split {}".format(split))
        return files

    def is_tested(self, attr):
        """Checks if a datum in the dataset has been tested.

        Args:
            attr: The attribute that needs to be checked.

        Returns:
            If the datum attribute is tested, then return the path where the
                attribute is stored; else, returns false.
        """
        cfg = self.cfg
        name = attr['name']
        path = cfg.test_result_folder
        store_path = join(path, self.name, name + '.labels')
        if exists(store_path):
            print("{} already exists.".format(store_path))
            return True
        else:
            return False

    def save_test_result(self, results, attr):
        """Saves the output of a model.

        Args:
            results: The output of a model for the datum associated with the attribute passed.
            attr: The attributes that correspond to the outputs passed in results.
        """
        cfg = self.cfg
        name = attr['name'].split('.')[0]
        path = cfg.test_result_folder
        make_dir(path)

        pred = results['predict_labels'] + 1
        store_path = join(path, self.name, name + '.labels')
        make_dir(Path(store_path).parent)
        np.savetxt(store_path, pred.astype(np.int32), fmt='%d')

        log.info("Saved {} in {}.".format(name, store_path))


class Semantic3DUnifiedSplit(BaseDatasetSplit):
    """This class is used to create a split for Semantic3DUnified dataset.

    Initialize the class.

    Args:
        dataset: The dataset to split.
        split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.
        **kwargs: The configuration of the model as keyword arguments.

    Returns:
        A dataset split object providing the requested subset of the data.
    """

    def __init__(self, dataset, split='training'):
        super().__init__(dataset, split=split)
        log.info("Found {} pointclouds for {}".format(len(self.path_list),
                                                      split))

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        pc_path = Path(self.path_list[idx])
        log.debug("get_data called {}".format(pc_path))

        pc = pd.read_csv(pc_path,
                         header=None,
                         sep=r'\s+',
                         dtype=np.float32).values

        points = pc[:, 0:3]
        feat = pc[:, [4, 5, 6]] # RGB features
        intensity = pc[:, 3]

        points = np.array(points, dtype=np.float32)
        feat = np.array(feat, dtype=np.float32)
        intensity = np.array(intensity, dtype=np.float32)

        label_path = pc_path.with_suffix('.labels')
        
        if not exists(label_path):
            raise FileNotFoundError(
                f"Label file not found for: {pc_path.stem}\n"
                f"Point cloud file: {pc_path}\n"
                f"Tried locations:\n"
                f"  - {pc_path.parent / (pc_path.stem + '.labels')}\n"
                "Each .txt file must have a corresponding .labels file."
            )
        
        labels = pd.read_csv(label_path,
                             header=None,
                             sep=r'\s+',
                             dtype=np.int32).values
        labels = np.array(labels, dtype=np.int32).reshape((-1,))

        data = {
            'point': points,
            'feat': feat,
            'intensity': intensity,
            'label': labels
        }

        return data

    def get_attr(self, idx):
        pc_path = Path(self.path_list[idx])
        name = pc_path.stem

        pc_path = str(pc_path)
        split = self.split
        attr = {'idx': idx, 'name': name, 'path': pc_path, 'split': split}
        return attr


DATASET._register_module(Semantic3DUnified)
