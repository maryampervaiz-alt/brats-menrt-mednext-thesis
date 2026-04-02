from __future__ import annotations

from monai.metrics import DiceMetric, HausdorffDistanceMetric, MeanIoU


def build_metrics():
    dice = DiceMetric(include_background=False, reduction="mean")
    iou = MeanIoU(include_background=False, reduction="mean")
    hd95 = HausdorffDistanceMetric(
        include_background=False,
        percentile=95,
        reduction="mean",
    )
    return {"dice": dice, "iou": iou, "hd95": hd95}
