import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Compute the Dice loss as loss function during training
"""

class DiceLoss(nn.Module):
    def __init__(self, num_classes=19, ignore_label=255, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.smooth = smooth

    def forward(self, logits, labels):
        probabilities = F.softmax(logits, dim=1)
        valid_mask = (labels != self.ignore_label)

        cleaned_labels = torch.where(valid_mask, labels, torch.zeros_like(labels))
        one_hot_labels = F.one_hot(cleaned_labels, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        one_hot_labels *= valid_mask.unsqueeze(1).float()

        probs_flat = probabilities.view(probabilities.size(0), self.num_classes, -1)
        labels_flat = one_hot_labels.view(one_hot_labels.size(0), self.num_classes, -1)

        intersect = (probs_flat * labels_flat).sum(dim=-1)
        total = probs_flat.sum(dim=-1) + labels_flat.sum(dim=-1)
        dice = (2 * intersect + self.smooth) / (total + self.smooth)

        # Final loss = 1 - mean dice score
        return 1.0 - dice.mean()