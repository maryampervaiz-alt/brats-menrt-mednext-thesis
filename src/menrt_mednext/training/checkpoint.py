from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    global_step: int,
    best_val_dice: float,
    config: dict[str, Any],
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "global_step": global_step,
        "best_val_dice": best_val_dice,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "config": config,
    }
    torch.save(ckpt, p)


def load_checkpoint(path: str | Path, model, optimizer=None, scheduler=None, scaler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"]:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"]:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in ckpt and ckpt["scaler_state_dict"]:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt
