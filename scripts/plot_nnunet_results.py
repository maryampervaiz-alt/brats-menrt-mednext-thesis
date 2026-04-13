"""Generate publication-ready figures for the nnUNet baseline experiment.

Reads from:
  menrt_artifacts/reports/       (CSV files from evaluate scripts)
  nnUNet_results/ (fold summary.json files from nnUNet training)

Outputs to:
  menrt_artifacts/figures/
    cv_fold_metrics_bar.png          — mean Dice per CV fold
    cv_dice_boxplot.png              — per-case Dice distribution across folds
    cv_hd95_boxplot.png              — per-case HD95 distribution
    cv_dice_vs_hd95_scatter.png      — Dice vs HD95 scatter
    testset_pre_vs_post_bar.png      — raw vs postprocessed comparison
    testset_dice_boxplot.png         — per-case test Dice (raw vs postproc)
    cv_metrics_table.csv             — mean ± SD table for thesis
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import yaml


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot nnUNet baseline results.")
    p.add_argument("--config", type=str, default="configs/nnunet_baseline.yaml")
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args()


def _load_cfg(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))["nnunet_baseline"]


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_cv_metrics(cfg: dict) -> tuple[list[dict], list[dict]]:
    """Load per-fold and per-case metrics from nnUNet summary.json files."""
    trainer_root = (
        Path(cfg["results_folder"])
        / "nnUNet" / str(cfg["network"]) / str(cfg["task_name"])
        / f'{cfg["trainer_name"]}__{cfg["plans_identifier"]}'
    )
    fold_rows, case_rows = [], []
    for fold in cfg.get("folds", [0, 1, 2, 3, 4]):
        summary_path = trainer_root / f"fold_{fold}" / "validation_raw" / "summary.json"
        if not summary_path.exists():
            continue
        with summary_path.open(encoding="utf-8") as f:
            s = json.load(f)
        mean = s["results"]["mean"]["1"]
        fold_rows.append({
            "fold":        fold,
            "dice":        float(mean.get("Dice", float("nan"))),
            "hd95":        float(mean.get("Hausdorff Distance 95", float("nan"))),
            "sensitivity": float(mean.get("Recall", float("nan"))),
            "precision":   float(mean.get("Precision", float("nan"))),
        })
        for c in s["results"]["all"]:
            case_rows.append({
                "fold":        fold,
                "case_id":     Path(c.get("reference", "")).name.replace(".nii.gz", ""),
                "dice":        float(c["1"].get("Dice", float("nan"))),
                "hd95":        float(c["1"].get("Hausdorff Distance 95", float("nan"))),
                "sensitivity": float(c["1"].get("Recall", float("nan"))),
                "precision":   float(c["1"].get("Precision", float("nan"))),
            })
    return fold_rows, case_rows


def _load_testset_metrics(reports_dir: Path) -> tuple[list[dict], list[dict]]:
    raw_path = reports_dir / "testset_metrics_raw.csv"
    pp_path  = reports_dir / "testset_metrics_postproc.csv"

    def _read(path: Path) -> list[dict]:
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as f:
            return list(csv.DictReader(f))

    return _read(raw_path), _read(pp_path)


# ── Plot helpers ──────────────────────────────────────────────────────────────

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

def _style_ax(ax: "plt.Axes", xlabel: str = "", ylabel: str = "",
              title: str = "") -> None:
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _annotate_bars(ax: "plt.Axes", bars, values: list[float]) -> None:
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold",
        )


# ── CV plots ──────────────────────────────────────────────────────────────────

def _plot_cv_bar(fold_rows: list[dict], figs_dir: Path, dpi: int) -> None:
    if not fold_rows:
        return
    folds  = [r["fold"] for r in fold_rows]
    dices  = [r["dice"] for r in fold_rows]
    labels = [f"Fold {f}" for f in folds]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, dices, color=[COLORS[f % 5] for f in folds],
                  edgecolor="black", linewidth=0.7, width=0.55)
    ax.axhline(np.nanmean(dices), color="red", linestyle="--", linewidth=1.5,
               label=f"Mean = {np.nanmean(dices):.3f}")
    _annotate_bars(ax, bars, dices)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10)
    _style_ax(ax, ylabel="Mean Dice (DSC)", title="CV Fold Dice Scores — nnUNet Baseline")
    plt.tight_layout()
    path = figs_dir / "cv_fold_metrics_bar.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_cv_boxplot(case_rows: list[dict], metric: str, ylabel: str,
                     title: str, figs_dir: Path, dpi: int, fname: str) -> None:
    if not case_rows:
        return
    folds = sorted({r["fold"] for r in case_rows})
    data  = [[r[metric] for r in case_rows if r["fold"] == f
              and not np.isnan(r[metric])] for f in folds]
    labels = [f"Fold {f}" for f in folds]

    fig, ax = plt.subplots(figsize=(7, 4))
    bp = ax.boxplot(data, patch_artist=True, labels=labels, widths=0.5)
    for patch, f in zip(bp["boxes"], folds):
        patch.set_facecolor(COLORS[f % 5])
        patch.set_alpha(0.75)
    _style_ax(ax, ylabel=ylabel, title=title)
    plt.tight_layout()
    path = figs_dir / fname
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_cv_scatter(case_rows: list[dict], figs_dir: Path, dpi: int) -> None:
    if not case_rows:
        return
    folds = sorted({r["fold"] for r in case_rows})
    fig, ax = plt.subplots(figsize=(6, 5))
    for f in folds:
        xs = [r["dice"] for r in case_rows if r["fold"] == f]
        ys = [r["hd95"] for r in case_rows if r["fold"] == f
              and not np.isnan(r["hd95"])]
        xs = xs[:len(ys)]
        ax.scatter(xs, ys, c=COLORS[f % 5], label=f"Fold {f}",
                   alpha=0.7, s=35, edgecolors="black", linewidths=0.4)
    ax.legend(fontsize=9)
    _style_ax(ax, xlabel="Dice (DSC)", ylabel="HD95 (mm)",
              title="Dice vs HD95 — CV Validation Cases")
    plt.tight_layout()
    path = figs_dir / "cv_dice_vs_hd95_scatter.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Test-set plots ────────────────────────────────────────────────────────────

def _plot_testset_comparison(raw: list[dict], pp: list[dict],
                             figs_dir: Path, dpi: int) -> None:
    if not raw and not pp:
        return
    metrics = ["dice", "hd95", "sensitivity", "precision"]
    labels  = ["Dice", "HD95 (mm)", "Sensitivity", "Precision"]

    def _means(rows: list[dict]) -> list[float]:
        return [
            float(np.nanmean([float(r[m]) for r in rows])) if rows else float("nan")
            for m in metrics
        ]

    raw_means = _means(raw)
    pp_means  = _means(pp)
    x = np.arange(len(metrics))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 4))
    b1 = ax.bar(x - w / 2, raw_means, w, label="Raw (no postprocessing)",
                color="#4C72B0", edgecolor="black", linewidth=0.7)
    b2 = ax.bar(x + w / 2, pp_means,  w, label="Postprocessed",
                color="#DD8452", edgecolor="black", linewidth=0.7)
    _annotate_bars(ax, b1, raw_means)
    _annotate_bars(ax, b2, pp_means)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(fontsize=10)
    _style_ax(ax, title="Test Set: Raw vs Postprocessed Predictions — nnUNet Baseline")
    plt.tight_layout()
    path = figs_dir / "testset_pre_vs_post_bar.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_testset_dice_boxplot(raw: list[dict], pp: list[dict],
                               figs_dir: Path, dpi: int) -> None:
    if not raw and not pp:
        return
    data   = []
    labels = []
    colors = []
    if raw:
        data.append([float(r["dice"]) for r in raw])
        labels.append("Raw")
        colors.append("#4C72B0")
    if pp:
        data.append([float(r["dice"]) for r in pp])
        labels.append("Postprocessed")
        colors.append("#DD8452")

    fig, ax = plt.subplots(figsize=(5, 4))
    bp = ax.boxplot(data, patch_artist=True, labels=labels, widths=0.45)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    ax.set_ylim(0, 1.05)
    _style_ax(ax, ylabel="Dice (DSC)",
              title="Test Set Dice Distribution")
    plt.tight_layout()
    path = figs_dir / "testset_dice_boxplot.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Summary table CSV ─────────────────────────────────────────────────────────

def _save_cv_table(fold_rows: list[dict], case_rows: list[dict],
                   reports_dir: Path) -> None:
    if not fold_rows:
        return
    rows = []
    for fr in fold_rows:
        fold = fr["fold"]
        case_dice = [r["dice"] for r in case_rows if r["fold"] == fold]
        case_hd95 = [r["hd95"] for r in case_rows if r["fold"] == fold
                     and not np.isnan(r["hd95"])]
        rows.append({
            "fold":           fold,
            "mean_dice":      round(fr["dice"], 4),
            "std_dice":       round(float(np.nanstd(case_dice)), 4) if case_dice else "",
            "mean_hd95":      round(fr["hd95"], 2),
            "std_hd95":       round(float(np.nanstd(case_hd95)), 2) if case_hd95 else "",
            "mean_sensitivity": round(fr["sensitivity"], 4),
            "mean_precision": round(fr["precision"], 4),
            "n_val_cases":    len(case_dice),
        })
    # Add overall summary row
    all_dice = [r["dice"] for r in case_rows]
    all_hd95 = [r["hd95"] for r in case_rows if not np.isnan(r["hd95"])]
    rows.append({
        "fold":           "OVERALL",
        "mean_dice":      round(float(np.nanmean([r["mean_dice"] for r in rows])), 4),
        "std_dice":       round(float(np.nanstd(all_dice)), 4),
        "mean_hd95":      round(float(np.nanmean([r["mean_hd95"] for r in rows])), 2),
        "std_hd95":       round(float(np.nanstd(all_hd95)), 2),
        "mean_sensitivity": round(float(np.nanmean([r["mean_sensitivity"] for r in rows])), 4),
        "mean_precision": round(float(np.nanmean([r["mean_precision"] for r in rows])), 4),
        "n_val_cases":    len(all_dice),
    })
    path = reports_dir / "cv_metrics_table.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg  = _load_cfg(args.config)

    reports_dir = Path(cfg["results_folder"]) / "menrt_artifacts" / "reports"
    figs_dir    = Path(cfg["results_folder"]) / "menrt_artifacts" / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    print("Loading CV metrics …", flush=True)
    fold_rows, case_rows = _load_cv_metrics(cfg)
    if fold_rows:
        print(f"  Loaded {len(fold_rows)} completed folds, {len(case_rows)} case records.")
    else:
        print("  No completed CV folds found yet — CV plots will be skipped.")

    print("Loading test-set metrics …", flush=True)
    raw_test, pp_test = _load_testset_metrics(reports_dir)
    if raw_test or pp_test:
        print(f"  raw={len(raw_test)} cases, postprocessed={len(pp_test)} cases.")
    else:
        print("  No test-set metrics found yet — test plots will be skipped.")

    print("\nGenerating plots …", flush=True)

    # CV plots
    _plot_cv_bar(fold_rows, figs_dir, args.dpi)
    _plot_cv_boxplot(case_rows, "dice", "Dice (DSC)", "CV Dice Distribution per Fold",
                     figs_dir, args.dpi, "cv_dice_boxplot.png")
    _plot_cv_boxplot(case_rows, "hd95", "HD95 (mm)", "CV HD95 Distribution per Fold",
                     figs_dir, args.dpi, "cv_hd95_boxplot.png")
    _plot_cv_scatter(case_rows, figs_dir, args.dpi)
    _save_cv_table(fold_rows, case_rows, reports_dir)

    # Test-set plots
    _plot_testset_comparison(raw_test, pp_test, figs_dir, args.dpi)
    _plot_testset_dice_boxplot(raw_test, pp_test, figs_dir, args.dpi)

    print(f"\nAll figures saved to: {figs_dir}", flush=True)


if __name__ == "__main__":
    main()
