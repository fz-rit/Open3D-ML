import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml3d.datasets.pama3d import PaMa3D
from open3d._ml3d.utils import DATASET


def main():
    # Replace with the absolute path to your dataset root folder.
    dataset_path = "/home/fzhcis/data/palau_2024_for_rc"
    
    # Create an instance of your custom dataset.
    dataset = PaMa3D(dataset_path)
    
    # Get the training split.
    train_split = dataset.get_split('train')
    
    print(f"Total training samples: {len(train_split)}")
    
    # Test: load the first sample and print its details.
    if len(train_split) > 0:
        sample = train_split.get_data(0)
        attr = train_split.get_attr(0)
        
        print("Sample attribute:", attr)
        print("Points shape:", sample['point'].shape)
        print("Features shape:", sample['feat'].shape)
        print("Labels shape:", sample['label'].shape)
    else:
        print("No CSV files found in the training csv folder.")


    registered_datasets = DATASET.module_dict
    if 'PaMa3D' in registered_datasets:
        print("PaMa3D dataset is successfully registered.")
    else:
        print("PaMa3D dataset is NOT registered.")

if __name__ == "__main__":
    main()
