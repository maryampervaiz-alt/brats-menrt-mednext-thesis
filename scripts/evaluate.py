from __future__ import annotations

import argparse
import json
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

import pandas as pd
import torch
from monai.inferers import sliding_window_inference
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from menrt_mednext.config import Config
from menrt_mednext.data.dataset import build_val_transforms, make_dataset, make_loader, post_process_for_metric, records_to_items
from menrt_mednext.data.discovery import discover_cases
from menrt_mednext.data.splits import load_split_json, make_holdout_split
from menrt_mednext.models.mednext_factory import build_mednext
from menrt_mednext.training.checkpoint import load_checkpoint
from menrt_mednext.training.losses import build_loss
from menrt_mednext.training.metrics import build_metrics
from menrt_mednext.training.postprocess import remove_small_components_3d
from menrt_mednext.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate MedNeXt checkpoint on validation split")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--kaggle", action="store_true")
    p.add_argument("--split-json", type=str, default="", help="Override split JSON for deterministic fold evaluation.")
    p.add_argument("--out-csv", type=str, default="")
    p.add_argument("--summary-json", type=str, default="")
    p.add_argument("--worst-k", type=int, default=10)
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

    if (root / "BraTS-MEN-RT-Train-v2").exists() and (root / "BraTS-MEN-RT-Val-v1").exists():
        raise ValueError(
            "Both train and val folders detected under kaggle_root_dir. "
            "Set data.kaggle_train_subdir explicitly to avoid accidental mixing."
        )

    for c in [root / "BraTS-MEN-RT-Train-v2", root / "BraTS2024-MEN-RT-TrainingData", root / "BraTS-MEN_RT", root]:
        if c.exists():
            return c
    return root


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config).raw

    if args.debug or cfg.get("DEBUG", False):
        cfg["training"]["epochs"] = int(cfg["debug"]["epochs"])

    set_seed(int(cfg["seed"]), deterministic=bool(cfg.get("deterministic", True)))

    data_root = _resolve_data_root(cfg, use_kaggle=args.kaggle)
    records = discover_cases(
        root_dir=data_root,
        image_keywords=tuple(cfg["data"]["image_keywords"]),
        label_keywords=tuple(cfg["data"]["label_keywords"]),
        require_labels=True,
    )

    split_json = args.split_json or cfg["data"].get("split_json", "")
    if split_json:
        train_ids, val_ids = load_split_json(split_json)
        val_records = [r for r in records if r.case_id in val_ids]
    else:
        _, val_records = make_holdout_split(
            records,
            val_fraction=float(cfg["data"]["val_fraction"]),
            seed=int(cfg["seed"]),
            group_pattern=str(cfg["data"].get("group_pattern", "")).strip(),
        )

    if not val_records:
        raise RuntimeError("No validation records found.")

    if args.debug:
        val_records = val_records[: int(cfg["debug"]["max_val_cases"])]

    val_items = records_to_items(val_records, require_label=True)
    val_ds = make_dataset(
        val_items,
        transform=build_val_transforms(cfg),
        cache_rate=float(cfg["data"].get("val_cache_rate", cfg["data"]["cache_rate"])),
        num_workers=int(cfg["system"]["num_workers"]),
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

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    load_checkpoint(args.checkpoint, model=model)
    model.eval()

    loss_fn = build_loss(cfg)
    metrics = build_metrics()
    post = post_process_for_metric()

    rows = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluate", leave=False):
            image = batch["image"].to(device)
            label = batch["label"].to(device).float()

            logits = sliding_window_inference(
                inputs=image,
                roi_size=tuple(cfg["transforms"]["patch_size"]),
                sw_batch_size=int(cfg["inference"]["sw_batch_size"]),
                predictor=model,
                overlap=float(cfg["inference"]["overlap"]),
                mode="gaussian",
            )
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            loss = float(loss_fn(logits, label).detach().cpu().item())

            processed = post({"pred": logits, "label": label})
            pred_bin = processed["pred"]
            label_bin = processed["label"]

            if bool(cfg.get("postprocess", {}).get("enabled", False)):
                min_size = int(cfg["postprocess"].get("min_size_voxels", 0))
                pred_np = pred_bin.detach().cpu().numpy()
                for b in range(pred_np.shape[0]):
                    pred_np[b, 0] = remove_small_components_3d(pred_np[b, 0], min_size)
                pred_bin = torch.from_numpy(pred_np).to(label_bin.device).float()

            for met in metrics.values():
                met.reset()
            metrics["dice"](y_pred=pred_bin, y=label_bin)
            if "iou" in metrics:
                metrics["iou"](y_pred=pred_bin, y=label_bin)
            # Compute HD95 on CPU to avoid Kaggle CuPy/cuCIM NVRTC toolchain failures.
            metrics["hd95"](y_pred=pred_bin.detach().cpu(), y=label_bin.detach().cpu())

            rows.append(
                {
                    "case_id": batch["case_id"][0],
                    "loss": loss,
                    "dice": float(metrics["dice"].aggregate().item()),
                    "iou": float(metrics["iou"].aggregate().item()) if "iou" in metrics else float("nan"),
                    "hd95": float(metrics["hd95"].aggregate().item()),
                }
            )

    df = pd.DataFrame(rows)
    summary = {
        "num_cases": int(len(df)),
        "dice_mean": float(df["dice"].mean()),
        "dice_std": float(df["dice"].std(ddof=1)) if len(df) > 1 else 0.0,
        "iou_mean": float(df["iou"].mean()),
        "iou_std": float(df["iou"].std(ddof=1)) if len(df) > 1 else 0.0,
        "hd95_mean": float(df["hd95"].mean()),
        "hd95_std": float(df["hd95"].std(ddof=1)) if len(df) > 1 else 0.0,
        "loss_mean": float(df["loss"].mean()),
        "loss_std": float(df["loss"].std(ddof=1)) if len(df) > 1 else 0.0,
    }

    out_csv = Path(args.out_csv) if args.out_csv else Path(args.checkpoint).resolve().parent / "eval_per_case.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values("dice", ascending=True).to_csv(out_csv, index=False)

    out_summary = (
        Path(args.summary_json) if args.summary_json else Path(args.checkpoint).resolve().parent / "eval_summary.json"
    )
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Summary:")
    print(json.dumps(summary, indent=2))
    print(f"Saved per-case CSV: {out_csv}")
    print(f"Saved summary JSON: {out_summary}")
    if len(df) > 0:
        k = max(1, min(int(args.worst_k), len(df)))
        print(f"Worst {k} cases by Dice:")
        print(df.sort_values("dice", ascending=True).head(k).to_string(index=False))


if __name__ == "__main__":
    main()
