
# pcd_seg_open3d

Point Cloud Segmentation with Open3D-ML

## Installation

Open3D-ML is included in Open3D's Python package (version 0.11+). To set up your environment:

```bash
# 1. Create and activate environment
conda create -n pcd_seg_open3d_env python=3.11
conda activate pcd_seg_open3d_env

# 2. Upgrade pip and install dependencies
pip install --upgrade pip
pip install open3d pandas scikit-learn matplotlib pyyaml addict tqdm
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install "numpy<2.0.0"

# 3. Test the installation
python -c "import sys; sys.path.insert(0, 'd:/mylab/Open3D-ML'); import ml3d; print('Open3D-ML loaded successfully')"
python -c "import torch; print(torch.cuda.is_available())"
```

*Note: Open3D v0.19.0 only supports torch 2.2.*  
*Do not use the requirements.txt files from the official repository!*


## Train on Semantic3D
1. Download the Semantic3D dataset. Go to the root dir of Open3d_ML, 
	```bash
	cd ./scripts/download_datasets/
	chmod +x download_semantic3d.sh
	conda install -c conda-forge p7zip -y
	./download_semantic3d.sh /home/fzhcis/data/semantic3d_full/
	```

<!-- 2. Preprocess the Semantic3D dataset, clone the the Open3D-ML repository, and then cd to the directory, run (this step may take a while):
	```bash
	conda activate pcd_seg_open3d_env
	python scripts/preprocess_semantic3d.py --dataset_path /home/fzhcis/data/semantic3d_full/Semantic3D --out_path /home/fzhcis/data/semantic3d_full/processed/Semantic3D
	``` -->
