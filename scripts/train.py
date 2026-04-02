from __future__ import annotations

import argparse
import copy
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_LOGGING_MIN_LEVEL", "3")

warnings.filterwarnings("ignore", message=".*cuda.cudart module is deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Orientationd.__init__:labels.*", category=FutureWarning)
warnings.filterwarnings(
    "ignore",
    message=".*Using a non-tuple sequence for multidimensional indexing is deprecated.*",
    category=UserWarning,
)
warnings.filterwarnings("ignore", message=".*get_mask_edges:always_return_as_numpy.*", category=FutureWarning)

import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.inferers import sliding_window_inference

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from menrt_mednext.config import Config
from menrt_mednext.data.dataset import (
    build_train_transforms,
    build_val_transforms,
    make_dataset,
    make_loader,
    post_process_for_metric,
    records_to_items,
)
from menrt_mednext.data.discovery import discover_cases
from menrt_mednext.data.splits import (
    load_split_json,
    make_holdout_split,
    make_kfold_split,
    save_split_json,
)
from menrt_mednext.models.mednext_factory import build_mednext
from menrt_mednext.training.checkpoint import load_checkpoint
from menrt_mednext.training.engine import Trainer
from menrt_mednext.training.losses import build_loss
from menrt_mednext.training.metrics import build_metrics
from menrt_mednext.utils.io import ensure_dir, timestamp_now, write_json
from menrt_mednext.utils.logging_utils import build_logger
from menrt_mednext.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MedNeXt on BraTS MEN-RT")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--kaggle", action="store_true")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--split-json", type=str, default="")
    p.add_argument("--use-kfold", action="store_true")
    p.add_argument("--n-folds", type=int, default=-1)
    p.add_argument("--fold-index", type=int, default=-1)
    p.add_argument("--lr", type=float, default=-1.0)
    p.add_argument("--patch-size", type=int, nargs=3, default=None, metavar=("X", "Y", "Z"))
    p.add_argument(
        "--strict-split",
        action="store_true",
        help="Fail if no explicit split JSON is provided (recommended for thesis reproducibility).",
    )
    p.add_argument("--save-split-json", type=str, default="")
    return p.parse_args()


def _resolve_data_root(cfg: dict, use_kaggle: bool) -> Path:
    if not use_kaggle:
        return Path(cfg["data"]["root_dir"])

    root = Path(cfg["data"]["kaggle_root_dir"])
    explicit_subdir = str(cfg["data"].get("kaggle_train_subdir", "")).strip()
    if explicit_subdir:
        explicit_path = Path(explicit_subdir)
        if not explicit_path.is_absolute():
            explicit_path = root / explicit_subdir
        if explicit_path.exists():
            return explicit_path
        raise FileNotFoundError(
            f"Configured data.kaggle_train_subdir not found: {explicit_path}"
        )

    if not cfg["data"].get("kaggle_auto_path", True):
        return root

    if (root / "BraTS-MEN-RT-Train-v2").exists() and (root / "BraTS-MEN-RT-Val-v1").exists():
        raise ValueError(
            "Both train and val folders detected under kaggle_root_dir. "
            "Set data.kaggle_train_subdir explicitly to avoid accidental mixing."
        )

    candidates = [
        root / "BraTS-MEN-RT-Train-v2",
        root / "BraTS2024-MEN-RT-TrainingData",
        root / "BraTS-MEN_RT",
        root,
    ]
    for c in candidates:
        if c.exists():
            return c
    return root


def _build_or_load_split(
    cfg: dict,
    all_records: list,
    save_path: Path | None = None,
    split_json_override: str = "",
    use_kfold: bool = False,
    n_folds_override: int = -1,
    fold_index_override: int = -1,
) -> tuple[list, list]:
    split_json = split_json_override or cfg["data"].get("split_json", "")
    if split_json:
        train_ids, val_ids = load_split_json(split_json)
        train_records = [r for r in all_records if r.case_id in train_ids]
        val_records = [r for r in all_records if r.case_id in val_ids]
        if not train_records or not val_records:
            raise RuntimeError("Configured split_json produced empty train/val split.")
        return train_records, val_records

    cv_cfg = cfg.get("cv", {})
    use_kfold = use_kfold or bool(cv_cfg.get("enabled", False))
    if use_kfold:
        n_folds = int(n_folds_override if n_folds_override > 1 else cv_cfg.get("n_folds", 5))
        fold_index = int(fold_index_override if fold_index_override >= 0 else cv_cfg.get("fold_index", 0))
        train_records, val_records = make_kfold_split(
            records=all_records,
            n_splits=n_folds,
            fold_index=fold_index,
            seed=int(cfg["seed"]),
        )
    else:
        train_records, val_records = make_holdout_split(
            records=all_records,
            val_fraction=float(cfg["data"]["val_fraction"]),
            seed=int(cfg["seed"]),
        )

    if save_path is not None:
        save_split_json(
            save_path,
            train_ids=[x.case_id for x in train_records],
            val_ids=[x.case_id for x in val_records],
        )
    return train_records, val_records


def _plot_history(history_csv: Path, fig_dir: Path) -> None:
    import pandas as pd

    if not history_csv.exists():
        return
    df = pd.read_csv(history_csv)
    if df.empty:
        return

    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], label="train_loss", linewidth=2)
    plt.plot(df["epoch"], df["val_loss"], label="val_loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "loss_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["val_dice"], label="val_dice", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.title("Validation Dice")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "dice_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["val_hd95"], label="val_hd95", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("HD95")
    plt.title("Validation HD95")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "hd95_curve.png", dpi=180)
    plt.close()


def _save_overlay(model, val_loader, cfg: dict, device: torch.device, out_path: Path) -> None:
    batch = next(iter(val_loader), None)
    if batch is None:
        return

    with torch.no_grad():
        image = batch["image"].to(device)
        label = batch["label"].to(device)
        logits = sliding_window_inference(
            inputs=image,
            roi_size=tuple(cfg["transforms"]["patch_size"]),
            sw_batch_size=int(cfg["inference"]["sw_batch_size"]),
            predictor=model,
            overlap=float(cfg["inference"]["overlap"]),
            mode="gaussian",
        )

    image_np = image[0, 0].detach().cpu().numpy()
    label_np = (label[0, 0] > 0.5).detach().cpu().numpy().astype(np.uint8)
    pred_np = (torch.sigmoid(logits)[0, 0] > 0.5).detach().cpu().numpy().astype(np.uint8)

    z = int(np.argmax(label_np.sum(axis=(0, 1)))) if label_np.sum() > 0 else image_np.shape[-1] // 2

    plt.figure(figsize=(12, 4))
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(image_np[:, :, z], cmap="gray")
    ax1.set_title("T1c")
    ax1.axis("off")

    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(image_np[:, :, z], cmap="gray")
    ax2.imshow(np.ma.masked_where(label_np[:, :, z] == 0, label_np[:, :, z]), cmap="Reds", alpha=0.55)
    ax2.set_title("GTV Ground Truth")
    ax2.axis("off")

    ax3 = plt.subplot(1, 3, 3)
    ax3.imshow(image_np[:, :, z], cmap="gray")
    ax3.imshow(np.ma.masked_where(pred_np[:, :, z] == 0, pred_np[:, :, z]), cmap="Blues", alpha=0.55)
    ax3.set_title("MedNeXt Prediction")
    ax3.axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config).raw
    cfg = copy.deepcopy(cfg)

    if args.debug:
        cfg["DEBUG"] = True
    if cfg.get("DEBUG", False):
        cfg["training"]["epochs"] = int(cfg["debug"]["epochs"])
        cfg["system"]["num_workers"] = int(cfg["debug"].get("num_workers", 0))
    if args.lr > 0:
        cfg["training"]["lr"] = float(args.lr)
    if args.patch_size is not None:
        cfg["transforms"]["patch_size"] = [int(v) for v in args.patch_size]

    # Preflight check: schedulefree dependency availability for active optimizer mode.
    optimizer_name = str(cfg["training"].get("optimizer_name", "adamw")).lower()
    if optimizer_name == "schedulefree_adamw":
        try:
            import schedulefree  # noqa: F401
        except Exception as e:
            raise ImportError(
                "optimizer_name is schedulefree_adamw but package `schedulefree` is unavailable in this interpreter."
            ) from e

    split_json_cfg = args.split_json or cfg["data"].get("split_json", "")
    if args.strict_split:
        has_explicit_split_json = bool(split_json_cfg)
        has_reproducible_kfold_split = bool(args.use_kfold and args.save_split_json)
        if not (has_explicit_split_json or has_reproducible_kfold_split):
            raise ValueError(
                "Strict split mode requires either: "
                "(1) explicit split JSON via --split-json/data.split_json, or "
                "(2) k-fold mode with --save-split-json."
            )

    data_root = _resolve_data_root(cfg, use_kaggle=args.kaggle)
    run_name = args.run_name or cfg.get("run_name", "mednext_run")
    if not args.run_name:
        run_name = f"{run_name}_{timestamp_now()}"

    run_dir = ensure_dir(Path(cfg["output"]["root_dir"]) / run_name)
    checkpoints_dir = ensure_dir(run_dir / "checkpoints")
    _ = ensure_dir(run_dir / "logs")
    _ = ensure_dir(run_dir / "metrics")
    figures_dir = ensure_dir(run_dir / "figures")

    logger = build_logger(run_dir / "logs" / "train.log")
    write_json(run_dir / "config_effective.json", cfg)

    set_seed(int(cfg["seed"]), deterministic=bool(cfg.get("deterministic", True)))

    logger.info("Data root: %s", data_root)
    all_records = discover_cases(
        root_dir=data_root,
        image_keywords=tuple(cfg["data"]["image_keywords"]),
        label_keywords=tuple(cfg["data"]["label_keywords"]),
        require_labels=True,
    )

    split_out = Path(args.save_split_json) if args.save_split_json else run_dir / "split.json"
    train_records, val_records = _build_or_load_split(
        cfg,
        all_records,
        save_path=split_out,
        split_json_override=args.split_json,
        use_kfold=args.use_kfold,
        n_folds_override=args.n_folds,
        fold_index_override=args.fold_index,
    )

    if cfg.get("DEBUG", False):
        train_records = train_records[: int(cfg["debug"]["max_train_cases"])]
        val_records = val_records[: int(cfg["debug"]["max_val_cases"])]

    logger.info("Train cases: %d | Val cases: %d", len(train_records), len(val_records))

    train_items = records_to_items(train_records, require_label=True)
    val_items = records_to_items(val_records, require_label=True)

    train_ds = make_dataset(
        train_items,
        transform=build_train_transforms(cfg),
        cache_rate=float(cfg["data"]["cache_rate"]),
        num_workers=int(cfg["system"]["num_workers"]),
    )
    val_ds = make_dataset(
        val_items,
        transform=build_val_transforms(cfg),
        cache_rate=float(cfg["data"].get("val_cache_rate", cfg["data"]["cache_rate"])),
        num_workers=int(cfg["system"]["num_workers"]),
    )

    train_loader = make_loader(
        train_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["system"]["num_workers"]),
        pin_memory=bool(cfg["system"]["pin_memory"]),
    )
    val_loader = make_loader(
        val_ds,
        batch_size=int(cfg["training"]["val_batch_size"]),
        shuffle=False,
        num_workers=int(cfg["system"]["num_workers"]),
        pin_memory=bool(cfg["system"]["pin_memory"]),
    )

    model = build_mednext(
        in_channels=int(cfg["model"]["in_channels"]),
        num_classes=int(cfg["model"]["num_classes"]),
        model_id=str(cfg["model"]["model_id"]),
        kernel_size=int(cfg["model"]["kernel_size"]),
        deep_supervision=bool(cfg["model"]["deep_supervision"]),
        checkpoint_style=cfg["model"].get("checkpoint_style", None),
    )

    loss_fn = build_loss(cfg)
    metrics = build_metrics()

    optimizer_name = str(cfg["training"].get("optimizer_name", "adamw")).lower()
    if optimizer_name == "schedulefree_adamw":
        try:
            from schedulefree import AdamWScheduleFree
        except Exception as e:
            raise ImportError(
                "schedulefree optimizer requested but package is missing. Install `schedulefree`."
            ) from e
        optimizer = AdamWScheduleFree(
            model.parameters(),
            lr=float(cfg["training"]["lr"]),
            weight_decay=float(cfg["training"]["weight_decay"]),
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg["training"]["lr"]),
            weight_decay=float(cfg["training"]["weight_decay"]),
        )

    scheduler_name = str(cfg["training"].get("scheduler_name", "cosine")).lower()
    if optimizer_name == "schedulefree_adamw":
        scheduler = None
    elif scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(cfg["training"]["epochs"]), eta_min=1e-6
        )
    elif scheduler_name == "reducelronplateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=10, factor=0.5)
    else:
        scheduler = None

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    resume_state = None
    resume_path = args.resume or cfg["training"].get("resume_path", "")
    if not resume_path and cfg["training"].get("auto_resume", True):
        latest_ckpt = checkpoints_dir / "latest_checkpoint.pt"
        if latest_ckpt.exists():
            resume_path = str(latest_ckpt)

    if resume_path:
        logger.info("Resuming checkpoint: %s", resume_path)
        resume_state = load_checkpoint(resume_path, model=model, optimizer=optimizer, scheduler=scheduler)

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        metrics=metrics,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        post_metric_transform=post_process_for_metric(),
        device=device,
        run_dir=run_dir,
        cfg=cfg,
        logger=logger,
        amp=bool(cfg["training"]["amp"]),
        resume_state=resume_state,
    )

    trainer.fit()

    _plot_history(run_dir / "metrics" / "history.csv", figures_dir)
    _save_overlay(model, val_loader, cfg, device, figures_dir / "qualitative_overlay.png")

    logger.info("Training finished.")
    logger.info("Best model: %s", checkpoints_dir / "best_model.pt")
    logger.info("Latest model: %s", checkpoints_dir / "latest_checkpoint.pt")


if __name__ == "__main__":
    main()
