import pandas as pd
import numpy as np
from open3d._ml3d.datasets.base_dataset import BaseDataset, BaseDatasetSplit
from open3d._ml3d.utils import make_dir, DATASET
import numpy as np
import os, glob
from os.path import join, exists
from pathlib import Path
import logging
log = logging.getLogger(__name__)

class PaMa3DSplit(BaseDatasetSplit):
    def __init__(self, dataset, split='training'):
        super().__init__(dataset, split=split)
        self.cfg = dataset.cfg
        path_list = dataset.get_split_list(split)
        log.info("Found {} pointclouds for {}".format(len(path_list), split))

        self.path_list = path_list
        self.split = split
        self.dataset = dataset

    def __len__(self):
        return len(self.path_list)


    def get_data(self, idx):
        csv_file = self.path_list[idx]
        df = pd.read_csv(csv_file)

        # # Normalize intensity.
        # df['Intensity'] = (df['Intensity'] - df['Intensity'].mean()) / df['Intensity'].std()

        # Drop bottom and top 1 percent of points based on intensity.
        df = df[(df['Intensity'] > df['Intensity'].quantile(0.01)) & (df['Intensity'] < df['Intensity'].quantile(0.99))]

        # # Remove points with class_id 0, and reassign class_ids 1-5 to 0-4.
        # df = df[df['class_id'] != 0]
        # df['class_id'] = df['class_id'] - 1

        # Rearrange columns to expected order:
        # For training/val: [x, y, z, class, intensity]
        points = df[['X', 'Y', 'Z']].values.astype(np.float32)
        if self.split != 'test':
            labels = df['class_id'].values.astype(np.int32)
            feat = df[['Intensity']].values.astype(np.float32)
        else:
            # For test: no label column
            labels = np.zeros((points.shape[0],), dtype=np.int32)
            feat = df[['intensity']].values.astype(np.float32)
        return {'point': points, 'feat': feat, 'label': labels}

    def get_attr(self, idx):
        name = os.path.basename(self.path_list[idx]).replace('.csv', '')
        return {'name': name, 'path': self.path_list[idx], 'split': self.split}

class PaMa3D(BaseDataset):
    def __init__(self,
                 dataset_path,
                 name='PaMa3D',
                 cache_dir='./logs/cache',
                 use_cache=False,
                 num_points=65536,
                 ignored_label_inds=[],
                 test_result_folder='./test',
                 **kwargs):

        super().__init__(dataset_path=dataset_path,
                         name=name,
                         cache_dir=cache_dir,
                         use_cache=use_cache,
                         num_points=num_points,
                         ignored_label_inds=ignored_label_inds,
                         test_result_folder=test_result_folder,
                         **kwargs)

        cfg = self.cfg

        self.dataset_path = cfg.dataset_path
        self.label_to_names = self.get_label_to_names()

        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array(cfg.ignored_label_inds)

        self.train_dir = str(Path(cfg.dataset_path) / 'train')
        self.val_dir = str(Path(cfg.dataset_path) / 'val')
        self.test_dir = str(Path(cfg.dataset_path) / 'test')

        self.train_files = [f for f in glob.glob(self.train_dir + "/*.csv")]
        self.val_files = [f for f in glob.glob(self.val_dir + "/*.csv")]
        self.test_files = [f for f in glob.glob(self.test_dir + "/*.csv")]

    @staticmethod
    def get_label_to_names():
        # Customize your label dictionary as needed.
        return {
            0: 'Void',
            1: 'Ground & Water',
            2: 'Stem',
            3: 'Canopy',
            4: 'Roots',
            5: 'Object',
        }

    def get_split(self, split):
        return PaMa3DSplit(self, split=split)

    def get_split_list(self, split):
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.

        Raises:
             ValueError: Indicates that the split name passed is incorrect. The
             split name should be one of 'training', 'test', 'validation', or
             'all'.
        """
        if split in ['test', 'testing']:
            self.rng.shuffle(self.test_files)
            return self.test_files
        elif split in ['val', 'validation']:
            self.rng.shuffle(self.val_files)
            return self.val_files
        elif split in ['train', 'training']:
            self.rng.shuffle(self.train_files)
            return self.train_files
        elif split in ['all']:
            files = self.val_files + self.train_files + self.test_files
            return files
        else:
            raise ValueError("Invalid split {}".format(split))
    
    def is_tested(self, attr):
        """Check if a test result already exists for the given sample."""
        cfg = self.cfg
        name = attr['name']
        path = cfg.test_result_folder if hasattr(cfg, 'test_result_folder') else './test_results'
        store_path = join(path, self.name, name + '.npy')
        
        if exists(store_path):
            print(f"[INFO] Test result already exists: {store_path}")
            return True
        else:
            return False
        

    def save_test_result(self, results, attr):
        """Save the predicted labels and points as a CSV file."""
        cfg = self.cfg
        name = attr['name']
        path = cfg.test_result_folder if hasattr(cfg, 'test_result_folder') else './test_results'
        save_folder = join(path, self.name)
        make_dir(save_folder)

        # Extract predictions and points
        pred = results['predict_labels']  # shape (N,)
        points = results['point']         # shape (N, 3)

        # Construct dataframe
        df = pd.DataFrame(points, columns=['x', 'y', 'z'])
        df['pred_class_id'] = pred.astype(np.int32)

        # Save as CSV
        store_path = join(save_folder, name + '.csv')
        df.to_csv(store_path, index=False)
        print(f"[INFO] Saved prediction CSV: {store_path}")

# Register the dataset (if needed)
# from open3d._ml3d.utils import DATASET
DATASET._register_module(PaMa3D)
