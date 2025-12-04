import torch
import torch.nn as nn

from ....datasets.utils import DataProcessing


def filter_valid_label(scores, labels, num_classes, ignored_label_inds, device):
    """Filter out ignored labels and remap remaining labels to 0..num_classes-1.

    - Drops samples whose label is in `ignored_label_inds`.
    - Remaps remaining labels by shifting down for each ignored label below them
      so that labels fall in [0, num_classes-1], matching score columns.
    - Keeps the score columns unchanged (num_classes columns).
    """
    valid_scores = scores.reshape(-1, num_classes).to(device)
    valid_labels = labels.reshape(-1).to(device)

    if ignored_label_inds is None:
        ignored_label_inds = []

    # Drop ignored labels
    if len(ignored_label_inds) > 0:
        ignored_bool = torch.zeros_like(valid_labels, dtype=torch.bool)
        for ign_label in ignored_label_inds:
            ignored_bool = torch.logical_or(ignored_bool, torch.eq(valid_labels, ign_label))
        valid_idx = torch.where(torch.logical_not(ignored_bool))[0].to(device)
        if valid_idx.numel() == 0:
            return valid_scores.new_zeros((0, num_classes)), valid_labels.new_zeros((0,), dtype=torch.long)
        valid_scores = torch.gather(
            valid_scores, 0, valid_idx.unsqueeze(-1).expand(-1, num_classes)
        )
        valid_labels = torch.gather(valid_labels, 0, valid_idx)

    # Remap labels to contiguous [0..num_classes-1]
    # Build reducing_list of length max_label+1 mapping original -> reduced index
    max_label = int(valid_labels.max().item()) if valid_labels.numel() > 0 else -1
    reducing_list = torch.arange(0, max_label + 1, dtype=torch.int64)
    inserted_value = torch.zeros([1], dtype=torch.int64)
    for ign_label in sorted(ignored_label_inds):
        if ign_label >= 0 and ign_label <= max_label:
            reducing_list = torch.cat([
                reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]
            ], 0)
    # Gather reduced labels
    valid_labels = torch.gather(reducing_list.to(device), 0, valid_labels.long())

    # Safety clamp: ensure labels within [0, num_classes-1]
    if valid_labels.numel() > 0:
        valid_labels = torch.clamp(valid_labels, min=0, max=num_classes - 1)

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
