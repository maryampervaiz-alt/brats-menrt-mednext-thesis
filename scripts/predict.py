from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import Activationsd, AsDiscreted, Compose, EnsureTyped, SaveImaged
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
from menrt_mednext.utils.io import ensure_dir
from menrt_mednext.utils.logging_utils import build_logger


def parse_args():
    p = argparse.ArgumentParser(description="Generate predictions with trained MedNeXt checkpoint")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data-root", type=str, default="")
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    cfg = Config.from_yaml(args.config).raw
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    data_root = args.data_root or cfg["data"]["root_dir"]
    out_dir = ensure_dir(args.out_dir or (Path(cfg["output"]["root_dir"]) / "inference"))
    logger = build_logger(out_dir / "predict.log")

    records = discover_cases(
        root_dir=data_root,
        image_keywords=tuple(cfg["data"]["image_keywords"]),
        label_keywords=tuple(cfg["data"]["label_keywords"]),
        require_labels=False,
    )
    infer_items = [{"case_id": r.case_id, "image": r.image} for r in records]

    infer_ds = make_dataset(
        infer_items,
        build_infer_transforms(cfg),
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

    model = build_mednext(
        in_channels=int(cfg["model"]["in_channels"]),
        num_classes=int(cfg["model"]["num_classes"]),
        model_id=str(cfg["model"]["model_id"]),
        kernel_size=int(cfg["model"]["kernel_size"]),
        deep_supervision=bool(cfg["model"]["deep_supervision"]),
        checkpoint_style=cfg["model"].get("checkpoint_style", None),
    ).to(device)
    load_checkpoint(args.checkpoint, model=model)
    model.eval()

    post = Compose(
        [
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold=0.5),
            EnsureTyped(keys="pred"),
            SaveImaged(
                keys="pred",
                meta_keys="image_meta_dict",
                output_dir=str(out_dir),
                output_postfix="gtv_pred",
                separate_folder=False,
                output_ext=".nii.gz",
                resample=False,
            ),
        ]
    )

    roi_size = tuple(cfg["transforms"]["patch_size"])
    sw_batch_size = int(cfg["inference"]["sw_batch_size"])
    overlap = float(cfg["inference"]["overlap"])

    logger.info("Starting inference for %d cases", len(infer_items))
    for batch in tqdm(infer_loader, desc="Predict", leave=False):
        image = batch["image"].to(device)
        logits = sliding_window_inference(
            inputs=image,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=overlap,
            mode="gaussian",
        )
        batch["pred"] = logits
        _ = [post(i) for i in decollate_batch(batch)]

    logger.info("Prediction done. Outputs at: %s", out_dir)


if __name__ == "__main__":
    main()