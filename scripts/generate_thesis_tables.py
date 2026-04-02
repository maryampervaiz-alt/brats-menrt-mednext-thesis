from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate thesis-ready tables from CV summaries")
    p.add_argument("--summary-csv", type=str, required=True)
    p.add_argument("--fold-csv", type=str, required=True)
    p.add_argument("--out-dir", type=str, default="outputs/reports")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(args.summary_csv)
    fold_df = pd.read_csv(args.fold_csv)

    md_path = out_dir / "thesis_results_table.md"
    tex_path = out_dir / "thesis_results_table.tex"

    row = summary_df.iloc[0]
    run_prefix = row["run_prefix"]
    n_folds = int(row["n_folds"])
    dice_pm = row["dice_mean_pm_std"]
    iou_pm = row["iou_mean_pm_std"] if "iou_mean_pm_std" in summary_df.columns else "N/A"
    hd95_pm = row["hd95_mean_pm_std"]

    md_lines = [
        "| Experiment | Folds | Val Dice (mean +/- std) | Val IoU (mean +/- std) | Val HD95 (mean +/- std) |",
        "|---|---:|---:|---:|---:|",
        f"| {run_prefix} | {n_folds} | {dice_pm} | {iou_pm} | {hd95_pm} |",
        "",
        "### Fold-wise Best Metrics",
        "",
    ]
    header = "| fold | best_epoch | val_dice | val_iou | val_hd95 | val_loss | train_loss | lr |"
    sep = "|---:|---:|---:|---:|---:|---:|---:|---:|"
    md_lines.extend([header, sep])
    for row in fold_df.itertuples(index=False):
        iou_value = float(row.val_iou) if hasattr(row, "val_iou") else float("nan")
        md_lines.append(
            f"| {int(row.fold)} | {int(row.best_epoch)} | {float(row.val_dice):.4f} | "
            f"{iou_value:.4f} | {float(row.val_hd95):.4f} | {float(row.val_loss):.4f} | {float(row.train_loss):.4f} | "
            f"{float(row.lr):.6f} |"
        )
    md_lines.append("")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    tex_lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\begin{tabular}{lcccc}",
        "\\hline",
        "Experiment & Folds & Val Dice ($\\mu\\pm\\sigma$) & Val IoU ($\\mu\\pm\\sigma$) & Val HD95 ($\\mu\\pm\\sigma$)\\\\",
        "\\hline",
        f"{run_prefix} & {n_folds} & {dice_pm} & {iou_pm} & {hd95_pm}\\\\",
        "\\hline",
        "\\end{tabular}",
        "\\caption{Cross-validation performance summary for MedNeXt on MEN-RT.}",
        "\\label{tab:menrt_cv_summary}",
        "\\end{table}",
        "",
    ]
    tex_path.write_text("\n".join(tex_lines), encoding="utf-8")

    print(f"Saved Markdown table: {md_path}")
    print(f"Saved LaTeX table: {tex_path}")


if __name__ == "__main__":
    main()

