import torch
import torch.nn as nn
import logging

from ....datasets.utils import DataProcessing

log = logging.getLogger(__name__)


def filter_valid_label(scores, labels, num_classes, ignored_label_inds, device):
    """Loss functions for semantic segmentation."""
    valid_scores = scores.reshape(-1, num_classes).to(device)
    valid_labels = labels.reshape(-1).to(device)

    # DEBUG: Log label statistics before processing
    log.debug(f"[DEBUG filter_valid_label] num_classes: {num_classes}")
    log.debug(f"[DEBUG filter_valid_label] ignored_label_inds: {ignored_label_inds}")
    log.debug(f"[DEBUG filter_valid_label] labels shape: {valid_labels.shape}")
    log.debug(f"[DEBUG filter_valid_label] labels min/max: [{valid_labels.min().item()}, {valid_labels.max().item()}]")
    log.debug(f"[DEBUG filter_valid_label] unique labels: {torch.unique(valid_labels).cpu().numpy()}")

    ignored_bool = torch.zeros_like(valid_labels, dtype=torch.bool)
    for ign_label in ignored_label_inds:
        ignored_bool = torch.logical_or(ignored_bool,
                                        torch.eq(valid_labels, ign_label))

    valid_idx = torch.where(torch.logical_not(ignored_bool))[0].to(device)

    valid_scores = torch.gather(valid_scores, 0,
                                valid_idx.unsqueeze(-1).expand(-1, num_classes))
    valid_labels = torch.gather(valid_labels, 0, valid_idx)

    # DEBUG: Log after filtering ignored labels
    log.debug(f"[DEBUG filter_valid_label] After filtering - labels min/max: [{valid_labels.min().item()}, {valid_labels.max().item()}]")
    log.debug(f"[DEBUG filter_valid_label] After filtering - unique labels: {torch.unique(valid_labels).cpu().numpy()}")

    # Reduce label values in the range of logit shape
    reducing_list = torch.arange(0, num_classes, dtype=torch.int64)
    inserted_value = torch.zeros([1], dtype=torch.int64)

    for ign_label in ignored_label_inds:
        if ign_label >= 0:

            reducing_list = torch.cat([
                reducing_list[:ign_label], inserted_value,
                reducing_list[ign_label:]
            ], 0)
    
    # DEBUG: Log reducing_list info
    log.debug(f"[DEBUG filter_valid_label] reducing_list length: {len(reducing_list)}")
    log.debug(f"[DEBUG filter_valid_label] reducing_list: {reducing_list.cpu().numpy()}")
    
    # Check if labels are in valid range for gather operation
    if valid_labels.max().item() >= len(reducing_list):
        log.error(f"[ERROR] Label index {valid_labels.max().item()} exceeds reducing_list length {len(reducing_list)}")
        log.error(f"[ERROR] This will cause the CUDA assertion error!")
        raise ValueError(f"Label index {valid_labels.max().item()} out of bounds for reducing_list of length {len(reducing_list)}")
    
    valid_labels = torch.gather(reducing_list.to(device), 0,
                                valid_labels.long())

    return valid_scores, valid_labels


class SemSegLoss(object):
    """Loss functions for semantic segmentation."""

    def __init__(self, pipeline, model, dataset, device):
        super(SemSegLoss, self).__init__()
        # weighted_CrossEntropyLoss
        if 'class_weights' in dataset.cfg.keys() and len(
                dataset.cfg.class_weights) != 0:
            class_wt = DataProcessing.get_class_weights(
                dataset.cfg.class_weights)
            weights = torch.tensor(class_wt, dtype=torch.float, device=device)

            self.weighted_CrossEntropyLoss = nn.CrossEntropyLoss(weight=weights)
        else:
            self.weighted_CrossEntropyLoss = nn.CrossEntropyLoss()
