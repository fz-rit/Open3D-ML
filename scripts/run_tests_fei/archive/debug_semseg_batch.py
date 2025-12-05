#!/usr/bin/env python
"""Debug a single SemanticSegmentation batch for Torch models.

Handles RandLANet (fixed-size dict) and KPFCNN (variable-size tensors).
Prints shapes, keys, label histograms before/after filtering, and checks loss
and metrics for NaNs. Use with any YAML in `ml3d/configs/*semantic3d*`.

Usage:
    python scripts/run_tests_fei/debug_semseg_batch.py \
        --model RandLANet \
        --config ml3d/configs/randlanet_semantic3dunified_xyz_4class.yml 

    python scripts/run_tests_fei/debug_semseg_batch.py \
        --model KPFCNN \
        --config ml3d/configs/kpconv_semantic3dunified_xyz_4class.yml 
"""

import argparse
import logging
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

import ml3d.utils as utils
import ml3d.datasets as datasets
import ml3d.torch.models as models
import ml3d.torch.pipelines as pipelines
from ml3d.torch.modules.losses.semseg_loss import SemSegLoss, filter_valid_label
from ml3d.torch.modules.metrics.semseg_metric import SemSegMetric
from ml3d.torch.dataloaders import TorchDataloader, get_sampler


log = logging.getLogger("debug_semseg_batch")


def label_hist(labels):
    arr = labels.reshape(-1)
    vals, counts = np.unique(arr, return_counts=True)
    hist = {int(v): int(c) for v, c in zip(vals, counts)}
    return hist


def main():
    parser = argparse.ArgumentParser(description="Debug one batch for SemSeg")
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    if not Path(args.config).exists():
        raise FileNotFoundError(args.config)

    cfg = utils.Config.load_from_file(args.config)

    # Build model
    model_cls = getattr(models, args.model)
    model = model_cls(**cfg.model)

    # Build dataset
    ds_name = cfg.dataset["name"]
    ds_cls = getattr(datasets, ds_name)
    dataset = ds_cls(cfg.dataset.pop("dataset_path"), **cfg.dataset)

    pipe = pipelines.SemanticSegmentation(model, dataset=dataset, device="gpu", **cfg.pipeline)
    device = pipe.device
    batcher = pipe.get_batcher(device)

    split = dataset.get_split(args.split if args.split != "train" else "train")
    sampler = split.sampler
    ds = TorchDataloader(
        dataset=split,
        preprocess=model.preprocess,
        transform=model.transform,
        sampler=sampler,
        use_cache=dataset.cfg.use_cache,
        steps_per_epoch=dataset.cfg.get("steps_per_epoch_train" if args.split == "train" else "steps_per_epoch_valid", None),
    )
    loader = DataLoader(
        ds,
        batch_size=pipe.cfg.batch_size,
        sampler=get_sampler(sampler),
        num_workers=0,
        pin_memory=pipe.cfg.get("pin_memory", True),
        collate_fn=batcher.collate_fn,
    )

    model.to(device)
    model.device = device
    model.eval()

    metric = SemSegMetric()
    Loss = SemSegLoss(pipe, model, dataset, device)

    batch = next(iter(loader))
    data = batch["data"]
    keys = list(data.keys()) if isinstance(data, dict) else []
    log.info(f"Input keys: {keys}")

    # Extract labels for different models
    labels = None
    try:
        if args.model == "RandLANet":
            # RandLANet dict has 'labels' and feature/coord tensors
            labels = data.get("labels", None)
            coords = data.get("coords", None)
            feats = data.get("features", None)
            log.info(f"RandLANet coords shape: {None if coords is None else tuple(coords.shape)}")
            log.info(f"RandLANet features shape: {None if feats is None else tuple(feats.shape)}")
        elif args.model in ("KPFCNN", "KPConv"):
            # KPFCNN packs tensors in batch object with .points/.labels
            labels = getattr(batch["data"], "labels", None)
            # Shape logging can be brittle across collate variations; skip to avoid noise
        else:
            labels = data.get("label", data.get("labels", None))
    except Exception as e:
        log.error(f"Label extraction error: {e}")

    if labels is None:
        raise RuntimeError("Could not find labels in batch. Keys: " + ",".join(keys))

    # Compute model outputs
    with torch.no_grad():
        results = model(batch["data"])  # shape depends on model

    # Filter ignored labels
    num_classes = model.cfg.num_classes
    ignored = model.cfg.ignored_label_inds
    scores, lab_valid = filter_valid_label(results, labels, num_classes, ignored, device)

    # Move to CPU for printing
    raw_labels_np = labels.detach().cpu().numpy().reshape(-1)
    valid_labels_np = lab_valid.detach().cpu().numpy().reshape(-1)

    log.info(f"Label histogram (raw): {label_hist(raw_labels_np)}")
    log.info(f"Label histogram (filtered): {label_hist(valid_labels_np)}")

    if scores.size()[-1] == 0 or scores.shape[0] == 0:
        log.info("Filtered batch has 0 valid points → skip loss/metrics.")
        return

    # Loss and metrics
    loss = Loss.weighted_CrossEntropyLoss(scores, lab_valid)
    loss_item = float(loss.detach().cpu().item())
    log.info(f"Loss: {loss_item}")
    if not np.isfinite(loss_item):
        log.error("Loss is not finite (inf or NaN).")

    metric.update(scores, lab_valid)
    accs = metric.acc()
    ious = metric.iou()
    log.info(f"Acc (per-class + overall): {accs}")
    log.info(f"IoU (per-class + mIoU): {ious}")


if __name__ == "__main__":
    main()
