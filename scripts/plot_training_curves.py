from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot training curves from history.csv")
    p.add_argument("--history", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    history_path = Path(args.history)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(history_path)

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], label="train_loss", linewidth=2)
    plt.plot(df["epoch"], df["val_loss"], label="val_loss", linewidth=2)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["val_dice"], label="val_dice", linewidth=2)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "dice_curve.png", dpi=180)
    plt.close()

    if "val_iou" in df.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(df["epoch"], df["val_iou"], label="val_iou", linewidth=2)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "iou_curve.png", dpi=180)
        plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["val_hd95"], label="val_hd95", linewidth=2)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "hd95_curve.png", dpi=180)
    plt.close()

    print(f"Saved plots in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
