from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, EnsureTyped, Invertd, SaveImaged
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from menrt_mednext.config import Config
from menrt_mednext.data.dataset import build_infer_transforms, make_dataset, make_loader
from menrt_mednext.data.discovery import discover_cases
from menrt_mednext.models.mednext_factory import build_mednext
from menrt_mednext.training.checkpoint import load_checkpoint
from menrt_mednext.training.postprocess import remove_small_components_3d
from menrt_mednext.utils.io import ensure_dir
from menrt_mednext.utils.logging_utils import build_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference with CV checkpoint ensemble")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--run-prefix", type=str, default="mednext_cv")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--checkpoint-name", type=str, default="best_model.pt")
    p.add_argument("--data-root", type=str, default="")
    p.add_argument("--out-dir", type=str, default="outputs/ensemble_predictions")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--min-size-voxels", type=int, default=-1)
    return p.parse_args()


def _collect_checkpoint_paths(run_prefix: str, n_folds: int, checkpoint_name: str) -> list[Path]:
    paths = []
    for fold in range(n_folds):
        ckpt = Path("outputs") / f"{run_prefix}_fold{fold}" / "checkpoints" / checkpoint_name
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint for fold {fold}: {ckpt}")
        paths.append(ckpt)
    return paths


def _build_model(cfg: dict, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model = build_mednext(
        in_channels=int(cfg["model"]["in_channels"]),
        num_classes=int(cfg["model"]["num_classes"]),
        model_id=str(cfg["model"]["model_id"]),
        kernel_size=int(cfg["model"]["kernel_size"]),
        deep_supervision=bool(cfg["model"]["deep_supervision"]),
        checkpoint_style=cfg["model"].get("checkpoint_style", None),
    ).to(device)
    load_checkpoint(str(checkpoint), model=model)
    model.eval()
    return model


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config).raw
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    checkpoints = _collect_checkpoint_paths(args.run_prefix, args.n_folds, args.checkpoint_name)

    data_root = args.data_root or cfg["data"]["root_dir"]
    out_dir = ensure_dir(args.out_dir)
    logger = build_logger(out_dir / "ensemble_infer.log")
    logger.info("Using %d checkpoints", len(checkpoints))

    models = [_build_model(cfg, ckpt, device) for ckpt in checkpoints]

    records = discover_cases(
        root_dir=data_root,
        image_keywords=tuple(cfg["data"]["image_keywords"]),
        label_keywords=tuple(cfg["data"]["label_keywords"]),
        require_labels=False,
    )
    infer_items = [{"case_id": r.case_id, "image": r.image} for r in records]

    infer_transforms = build_infer_transforms(cfg)
    infer_ds = make_dataset(
        infer_items,
        infer_transforms,
        cache_rate=0.0,
        num_workers=int(cfg["system"]["num_workers"]),
    )
    infer_loader = make_loader(
        infer_ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg["system"]["num_workers"]),
        pin_memory=bool(cfg["system"]["pin_memory"]),
    )

    save_tf = Compose(
        [
            Invertd(
                keys="pred",
                transform=infer_transforms,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                nearest_interp=True,
                to_tensor=True,
            ),
            EnsureTyped(keys="pred"),
            SaveImaged(
                keys="pred",
                meta_keys="pred_meta_dict",
                output_dir=str(out_dir),
                output_postfix="gtv_pred_ens",
                separate_folder=False,
                output_ext=".nii.gz",
                resample=False,
            ),
        ]
    )

    roi_size = tuple(cfg["transforms"]["patch_size"])
    sw_batch_size = int(cfg["inference"]["sw_batch_size"])
    overlap = float(cfg["inference"]["overlap"])

    cfg_min_size = int(cfg.get("postprocess", {}).get("min_size_voxels", 0))
    min_size = args.min_size_voxels if args.min_size_voxels >= 0 else cfg_min_size

    for batch in tqdm(infer_loader, desc="Ensemble Inference", leave=False):
        image = batch["image"].to(device)

        probs_acc = None
        for model in models:
            logits = sliding_window_inference(
                inputs=image,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=overlap,
                mode="gaussian",
            )
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            probs = torch.sigmoid(logits)
            probs_acc = probs if probs_acc is None else probs_acc + probs

        probs_mean = probs_acc / len(models)
        pred = (probs_mean > float(args.threshold)).float()

        pred_np = pred[0, 0].detach().cpu().numpy().astype(np.uint8)
        if min_size > 0:
            pred_np = remove_small_components_3d(pred_np, min_size)
        pred[0, 0] = torch.from_numpy(pred_np).to(pred.device)

        batch["pred"] = pred
        _ = [save_tf(item) for item in decollate_batch(batch)]

    logger.info("Saved ensemble predictions to: %s", out_dir)


if __name__ == "__main__":
    main()
