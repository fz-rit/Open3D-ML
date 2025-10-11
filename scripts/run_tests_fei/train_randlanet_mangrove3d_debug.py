import logging
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch.multiprocessing as mp

# Fix for Python 3.12 multiprocessing issues with num_workers > 0
# Set the start method to 'spawn' before importing other modules
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

# Set up debug logging BEFORE importing ml3d modules
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('train_debug.log', mode='w')
    ]
)

# Add the Open3D-ML repository root to Python path
repo_root = Path(__file__).parent.parent.parent  # Go up 3 levels to reach Open3D-ML root
sys.path.insert(0, str(repo_root))

import ml3d.utils as utils
import ml3d.datasets as datasets
import ml3d.torch.models as models
import ml3d.torch.pipelines as pipelines

log = logging.getLogger(__name__)


def main():
    log.info("="*80)
    log.info("Starting DEBUG training run for RandLANet on Mangrove3D")
    log.info("="*80)
    
    cfg_file = "/home/fzhcis/mylab/Open3D-ML/ml3d/configs/randlanet_mangrove3d.yml"
    cfg = utils.Config.load_from_file(cfg_file)
    
    log.info(f"Config loaded: num_classes = {cfg.model.get('num_classes', 'NOT SET')}")
    log.info(f"Config loaded: ignored_label_inds = {cfg.dataset.get('ignored_label_inds', 'NOT SET')}")
    
    model = models.RandLANet(**cfg.model)
    
    dataset = datasets.Mangrove3D(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
    log.info(f"Dataset created: num_classes = {dataset.num_classes}")
    log.info(f"Dataset label_to_names: {dataset.label_to_names}")
    
    pipeline = pipelines.SemanticSegmentation(model, dataset=dataset, device="gpu", **cfg.pipeline)
    
    log.info("Starting training...")
    pipeline.run_train()


if __name__ == "__main__":
    main()
