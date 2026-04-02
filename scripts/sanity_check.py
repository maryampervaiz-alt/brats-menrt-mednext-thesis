from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from menrt_mednext.config import Config
from menrt_mednext.models.mednext_factory import build_mednext


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sanity-check MedNeXt forward pass")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config).raw

    model = build_mednext(
        in_channels=int(cfg["model"]["in_channels"]),
        num_classes=int(cfg["model"]["num_classes"]),
        model_id=str(cfg["model"]["model_id"]),
        kernel_size=int(cfg["model"]["kernel_size"]),
        deep_supervision=bool(cfg["model"]["deep_supervision"]),
        checkpoint_style=cfg["model"].get("checkpoint_style", None),
    )
    x = torch.randn(1, int(cfg["model"]["in_channels"]), *cfg["transforms"]["patch_size"])
    with torch.no_grad():
        y = model(x)
    if isinstance(y, (tuple, list)):
        y = y[0]
    print(f"Input shape: {tuple(x.shape)}")
    print(f"Output shape: {tuple(y.shape)}")


if __name__ == "__main__":
    main()