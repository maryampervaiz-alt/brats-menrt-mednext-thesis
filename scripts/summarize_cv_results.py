from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize k-fold training results")
    p.add_argument("--run-prefix", type=str, required=True)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--metric", type=str, default="val_dice")
    p.add_argument("--out-dir", type=str, default="outputs/reports")
    return p.parse_args()


def _format_mean_std(series: pd.Series, precision: int = 4) -> str:
    return f"{series.mean():.{precision}f} ± {series.std(ddof=1):.{precision}f}"


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_rows = []
    for fold in range(args.n_folds):
        history_path = Path("outputs") / f"{args.run_prefix}_fold{fold}" / "metrics" / "history.csv"
        if not history_path.exists():
            raise FileNotFoundError(f"Missing history file: {history_path}")

        df = pd.read_csv(history_path)
        if df.empty:
            raise RuntimeError(f"Empty history file: {history_path}")
        if args.metric not in df.columns:
            raise KeyError(f"Metric `{args.metric}` not found in {history_path}.")

        best_idx = df[args.metric].idxmax()
        best = df.loc[best_idx]
        fold_rows.append(
            {
                "fold": fold,
                "best_epoch": int(best["epoch"]),
                "val_dice": float(best["val_dice"]),
                "val_hd95": float(best["val_hd95"]),
                "val_loss": float(best["val_loss"]),
                "train_loss": float(best["train_loss"]),
                "lr": float(best["lr"]),
            }
        )

    fold_df = pd.DataFrame(fold_rows).sort_values("fold").reset_index(drop=True)
    fold_csv = out_dir / f"{args.run_prefix}_fold_metrics.csv"
    fold_df.to_csv(fold_csv, index=False)

    summary = {
        "run_prefix": args.run_prefix,
        "n_folds": args.n_folds,
        "val_dice_mean": fold_df["val_dice"].mean(),
        "val_dice_std": fold_df["val_dice"].std(ddof=1),
        "val_hd95_mean": fold_df["val_hd95"].mean(),
        "val_hd95_std": fold_df["val_hd95"].std(ddof=1),
        "val_loss_mean": fold_df["val_loss"].mean(),
        "val_loss_std": fold_df["val_loss"].std(ddof=1),
        "dice_mean_pm_std": _format_mean_std(fold_df["val_dice"]),
        "hd95_mean_pm_std": _format_mean_std(fold_df["val_hd95"]),
    }
    summary_df = pd.DataFrame([summary])
    summary_csv = out_dir / f"{args.run_prefix}_summary_mean_std.csv"
    summary_df.to_csv(summary_csv, index=False)

    print(f"Saved fold metrics: {fold_csv}")
    print(f"Saved summary: {summary_csv}")


if __name__ == "__main__":
    main()

