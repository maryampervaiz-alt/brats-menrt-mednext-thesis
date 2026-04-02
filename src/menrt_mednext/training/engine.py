from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import Compose
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from menrt_mednext.training.checkpoint import save_checkpoint
from menrt_mednext.training.postprocess import remove_small_components_3d


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn,
        metrics: dict[str, Any],
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        post_metric_transform: Compose,
        device: torch.device,
        run_dir: str | Path,
        cfg: dict[str, Any],
        logger,
        amp: bool = True,
        resume_state: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.post_metric_transform = post_metric_transform
        self.device = device
        self.run_dir = Path(run_dir)
        self.cfg = cfg
        self.logger = logger
        self.amp = amp and torch.cuda.is_available()
        self.scaler = GradScaler(enabled=self.amp)

        self.epochs = int(cfg["training"]["epochs"])
        self.val_interval = int(cfg["training"].get("val_interval", 1))
        self.sw_batch_size = int(cfg["inference"]["sw_batch_size"])
        self.roi_size = tuple(cfg["transforms"]["patch_size"])
        self.overlap = float(cfg["inference"].get("overlap", 0.5))
        self.early_stop_patience = int(cfg["training"].get("early_stop_patience", 1000000))

        self.start_epoch = 1
        self.global_step = 0
        self.best_val_dice = -1.0
        self.no_improve_epochs = 0

        if resume_state:
            self.start_epoch = int(resume_state["epoch"]) + 1
            self.global_step = int(resume_state.get("global_step", 0))
            self.best_val_dice = float(resume_state.get("best_val_dice", -1.0))

        self.metrics_csv = self.run_dir / "metrics" / "history.csv"
        self.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
        if not self.metrics_csv.exists():
            with self.metrics_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "epoch",
                        "train_loss",
                        "val_loss",
                        "val_dice",
                        "val_hd95",
                        "lr",
                        "epoch_seconds",
                        "max_gpu_mem_gb",
                    ]
                )

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        if hasattr(self.optimizer, "train"):
            self.optimizer.train()
        losses = []
        pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch}", leave=False)
        for batch in pbar:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device).float()

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=self.amp):
                logits = self.model(images)
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                loss = self.loss_fn(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_item = float(loss.detach().cpu().item())
            losses.append(loss_item)
            self.global_step += 1
            pbar.set_postfix({"loss": f"{loss_item:.4f}"})
        return float(np.mean(losses)) if losses else float("nan")

    @torch.no_grad()
    def _validate(self, epoch: int) -> dict[str, float]:
        self.model.eval()
        if hasattr(self.optimizer, "eval"):
            self.optimizer.eval()
        val_losses = []

        for met in self.metrics.values():
            met.reset()

        pbar = tqdm(self.val_loader, desc=f"Val Epoch {epoch}", leave=False)
        for batch in pbar:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device).float()

            with autocast(enabled=self.amp):
                logits = sliding_window_inference(
                    inputs=images,
                    roi_size=self.roi_size,
                    sw_batch_size=self.sw_batch_size,
                    predictor=self.model,
                    overlap=self.overlap,
                    mode="gaussian",
                )
                loss = self.loss_fn(logits, labels)

            val_losses.append(float(loss.detach().cpu().item()))
            processed = self.post_metric_transform({"pred": logits, "label": labels})
            pred_bin = processed["pred"]
            label_bin = processed["label"]

            if bool(self.cfg.get("postprocess", {}).get("enabled", False)):
                min_size = int(self.cfg["postprocess"].get("min_size_voxels", 0))
                pred_np = pred_bin.detach().cpu().numpy()
                for b in range(pred_np.shape[0]):
                    pred_np[b, 0] = remove_small_components_3d(pred_np[b, 0], min_size)
                pred_bin = torch.from_numpy(pred_np).to(label_bin.device).float()

            self.metrics["dice"](y_pred=pred_bin, y=label_bin)
            self.metrics["hd95"](y_pred=pred_bin, y=label_bin)

        dice = float(self.metrics["dice"].aggregate().item())
        hd95 = float(self.metrics["hd95"].aggregate().item())
        return {
            "val_loss": float(np.mean(val_losses)) if val_losses else float("nan"),
            "val_dice": dice,
            "val_hd95": hd95,
        }

    def _write_history_row(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_dice: float,
        val_hd95: float,
        lr: float,
        epoch_seconds: float,
        max_gpu_mem_gb: float,
    ) -> None:
        with self.metrics_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, val_dice, val_hd95, lr, epoch_seconds, max_gpu_mem_gb])

    def fit(self) -> None:
        ckpt_dir = self.run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.start_epoch, self.epochs + 1):
            epoch_t0 = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            train_loss = self._train_one_epoch(epoch)
            lr = float(self.optimizer.param_groups[0]["lr"])

            val_loss, val_dice, val_hd95 = float("nan"), float("nan"), float("nan")
            if epoch % self.val_interval == 0:
                val_stats = self._validate(epoch)
                val_loss = val_stats["val_loss"]
                val_dice = val_stats["val_dice"]
                val_hd95 = val_stats["val_hd95"]

                improved = val_dice > self.best_val_dice
                if improved:
                    self.best_val_dice = val_dice
                    self.no_improve_epochs = 0
                    save_checkpoint(
                        path=ckpt_dir / "best_model.pt",
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        scaler=self.scaler,
                        epoch=epoch,
                        global_step=self.global_step,
                        best_val_dice=self.best_val_dice,
                        config=self.cfg,
                    )
                else:
                    self.no_improve_epochs += 1

                self.logger.info(
                    "Epoch %d | train_loss=%.4f | val_loss=%.4f | val_dice=%.4f | val_hd95=%.4f | lr=%.6f",
                    epoch,
                    train_loss,
                    val_loss,
                    val_dice,
                    val_hd95,
                    lr,
                )
            else:
                self.logger.info(
                    "Epoch %d | train_loss=%.4f | val skipped | lr=%.6f",
                    epoch,
                    train_loss,
                    lr,
                )

            save_checkpoint(
                path=ckpt_dir / "latest_checkpoint.pt",
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                epoch=epoch,
                global_step=self.global_step,
                best_val_dice=self.best_val_dice,
                config=self.cfg,
            )
            if epoch % int(self.cfg["training"].get("save_every", 10)) == 0:
                save_checkpoint(
                    path=ckpt_dir / f"epoch_{epoch:04d}.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    epoch=epoch,
                    global_step=self.global_step,
                    best_val_dice=self.best_val_dice,
                    config=self.cfg,
                )

            epoch_seconds = float(time.perf_counter() - epoch_t0)
            max_gpu_mem_gb = (
                float(torch.cuda.max_memory_allocated() / (1024**3)) if torch.cuda.is_available() else float("nan")
            )

            self._write_history_row(
                epoch,
                train_loss,
                val_loss,
                val_dice,
                val_hd95,
                lr,
                epoch_seconds,
                max_gpu_mem_gb,
            )

            if self.scheduler is not None:
                if self.cfg["training"].get("scheduler_name", "").lower() == "reducelronplateau":
                    self.scheduler.step(val_loss if not np.isnan(val_loss) else train_loss)
                else:
                    self.scheduler.step()

            if self.no_improve_epochs >= self.early_stop_patience:
                self.logger.info(
                    "Early stopping triggered at epoch %d (patience=%d).",
                    epoch,
                    self.early_stop_patience,
                )
                break
