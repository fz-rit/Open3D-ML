import numpy as np
import os, argparse, pickle, sys
from os.path import exists, join, isfile, dirname, abspath, split
import logging
import glob
from sklearn.neighbors import KDTree
import yaml
from pathlib import Path
import sys
sys.path.append('/home/fzhcis/mylab/Open3D-ML/ml3d/')  # Adjust the path accordingly
from datasets.base_dataset import BaseDataset, BaseDatasetSplit
from datasets.utils import DataProcessing
from utils import make_dir, DATASET
import json
import pandas as pd
log = logging.getLogger(__name__)


class FrancForSeg(BaseDataset):
    """This class is used to create a dataset based on the Francisco
     Forest Segmentation Synthetic dataset, and used in visualizer, training, or testing.

    DOI https://doi.org/10.1007/978-3-031-78128-5_5
    github https://github.com/lrse/synthetic-forest-datasets
    """

    def __init__(self,
                 dataset_path,
                 name='FrancForSeg',
                 cache_dir='./logs/cache',
                 use_cache=False,
                 class_weights=[ # Actually frequency of each class
                     2_050_652, # Class 0: Terrain
                     12_852_250,  # Class 1: Trunk
                     43_355_279, # Class 2: Canopy
                     62_201_819 # Class 3: Understorey, including bushes, grass and other vegetation
                 ],
                 ignored_label_inds=[0],
                 test_result_folder='./test', # backup: /home/fzhcis/mylab/data/synthetic-lidar-point-clouds-tree-simulator/LiDAR-like-Dataset/test_result
                 **kwargs):
        """Initialize the function by passing the dataset and other details.

        Args:
            dataset_path: The path to the dataset to use.
            name: The name of the dataset.
            cache_dir: The directory where the cache is stored.
            use_cache: Indicates if the dataset should be cached.
            class_weights: The class weights to use in the dataset.
            ignored_label_inds: A list of labels that should be ignored in the dataset.
            test_result_folder: The folder where the test results should be stored.

        Returns:
            class: The corresponding class.
        """
        super().__init__(dataset_path=dataset_path,
                         name=name,
                         cache_dir=cache_dir,
                         use_cache=use_cache,
                         class_weights=class_weights,
                         ignored_label_inds=ignored_label_inds,
                         test_result_folder=test_result_folder,
                         **kwargs)

        cfg = self.cfg

        self.label_to_names = self.get_label_to_names()
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([0])
        self.train_test_split_json_dir = Path(cfg.train_test_split_json_dir)
        self.train_files = json.load(open(self.train_test_split_json_dir / 'shuffled_train_file_list.json'))
        self.test_files = json.load(open(self.train_test_split_json_dir / 'shuffled_test_file_list.json'))
        self.val_files = json.load(open(self.train_test_split_json_dir / 'shuffled_val_file_list.json'))

        self.train_files = sorted([Path(cfg.dataset_path) / f"{f}.txt" for f in self.train_files])
        self.test_files = sorted([Path(cfg.dataset_path) / f"{f}.txt" for f in self.test_files])
        self.val_files = sorted([Path(cfg.dataset_path) / f"{f}.txt" for f in self.val_files])

    @staticmethod
    def get_label_to_names():
        """Returns a label to names dictionary object.

        Returns:
            A dict where keys are label numbers and
            values are the corresponding names.
        """
        label_to_names = {
            0 : 'Terrain',
            1 : 'Trunk',
            2 : 'Canopy',
            3 : 'Understory', # including bushes, grass and other vegetation
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
        return FrancForSegSplit(self, split=split)
    
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
            raise ValueError(f"Invalid split {split}")
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
        pass

    def save_test_result(self, results, attr):
        """Saves the output of a model.

        Args:
            results: The output of a model for the datum associated with the attribute passed.
            attr: The attributes that correspond to the outputs passed in results.
        """
        cfg = self.cfg
        name = attr['name']
        path = cfg.test_result_folder
        make_dir(path)

        pred = results['predict_labels'] + 1
        store_path = join(path, self.name, name + '.labels')
        make_dir(Path(store_path).parent)
        np.savetxt(store_path, pred.astype(np.int32), fmt='%d')

        log.info("Saved {} in {}.".format(name, store_path))



class FrancForSegSplit(BaseDatasetSplit):

    def __init__(self, dataset, split='training'):
        super().__init__(dataset, split=split)
        self.cfg = dataset.cfg
        path_list = dataset.get_split_list(split)

        self.path_list = path_list
        log.info("Found {} pointclouds for {}".format(len(self.path_list),
                                                      split))

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        pc_path = self.path_list[idx]
        log.debug("get_data called {}".format(pc_path))

        pc = pd.read_csv(pc_path, header=None, delim_whitespace=True, dtype=np.float32).values
        points, labels = pc[:, :3], pc[:, 3].astype(np.int32)

        data = {
            'point': points,
            'feat': None,
            'label': labels
        }
        return data

    def get_attr(self, idx):
        name = self.path_list[idx][1].split('/')[-1].split('.')[0]
        return {
            'name': name,
            'path': str(Path(self.path_list[idx][1])),
            'split': self.split
        }


DATASET._register_module(FrancForSeg)
