from __future__ import annotations

from typing import Any

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


class BinaryFocalLossWithLogits(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, target):
        bce = self.bce(logits, target)
        pt = (-bce).exp()
        focal = (1 - pt) ** self.gamma * bce
        if self.alpha is not None:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            focal = alpha_t * focal
        return focal.mean()


class DiceFocalLoss(nn.Module):
    def __init__(
        self,
        dice_weight: float = 0.7,
        focal_weight: float = 0.3,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
    ) -> None:
        super().__init__()
        self.dice = DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
        self.focal = BinaryFocalLossWithLogits(gamma=focal_gamma, alpha=focal_alpha)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, logits, target):
        d = self.dice(logits, target)
        f = self.focal(logits, target)
        return self.dice_weight * d + self.focal_weight * f


def build_loss(cfg: dict[str, Any]) -> nn.Module:
    tcfg = cfg["training"]
    loss_name = str(tcfg.get("loss_name", "dice_bce")).lower()
    dice_weight = float(tcfg.get("dice_weight", 0.7))

    if loss_name == "dice":
        return DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    if loss_name == "dice_focal":
        return DiceFocalLoss(
            dice_weight=dice_weight,
            focal_weight=float(tcfg.get("focal_weight", 0.3)),
            focal_gamma=float(tcfg.get("focal_gamma", 2.0)),
            focal_alpha=float(tcfg.get("focal_alpha", 0.25)),
        )
    if loss_name == "dice_bce":
        return DiceBCELoss(
            dice_weight=dice_weight,
            bce_weight=float(tcfg.get("bce_weight", 0.3)),
        )
    raise ValueError(f"Unsupported loss_name: {loss_name}. Use one of [dice, dice_bce, dice_focal].")
