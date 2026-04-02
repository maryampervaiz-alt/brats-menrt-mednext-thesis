from __future__ import annotations

import torch.nn as nn
from monai.losses import DiceLoss


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight: float = 0.7, bce_weight: float = 0.3) -> None:
        super().__init__()
        self.dice = DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, logits, target):
        d = self.dice(logits, target)
        b = self.bce(logits, target)
        return self.dice_weight * d + self.bce_weight * b
