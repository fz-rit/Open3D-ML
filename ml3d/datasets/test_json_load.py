import json
from pathlib import Path
train_files = json.load(open(Path("/home/fzhcis/mylab/data/synthetic-lidar-point-clouds-tree-simulator/LiDAR-like-Dataset/train_test_split") / 'shuffled_train_file_list.json'))
# print(train_files)
train_files = sorted([Path("/home/fzhcis/mylab/data/synthetic-lidar-point-clouds-tree-simulator/LiDAR-like-Dataset/")  / f"{f}.txt" for f in train_files])
print(type(train_files))
print(train_files[0])
print(type(train_files[0]))
print(train_files[0].exists())
print(len(train_files))