"""Evaluate nnUNet predictions on the held-out test set.

Computes per-case and mean metrics for:
  - Raw predictions      (nnUNet_predict --disable_postprocessing)
  - Postprocessed preds  (nnUNet_predict default)

Metrics
-------
  Dice (DSC)   — voxel overlap, primary metric for GTV segmentation
  HD95         — 95th-percentile Hausdorff Distance in mm
  Sensitivity  — TP / (TP + FN)  fraction of GT tumour captured
  Precision    — TP / (TP + FP)  fraction of predicted tumour that is correct

Output
------
  menrt_artifacts/reports/testset_metrics_raw.csv
  menrt_artifacts/reports/testset_metrics_postproc.csv
  menrt_artifacts/reports/testset_comparison.json
  menrt_artifacts/reports/testset_comparison_table.txt   (print-ready table)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import yaml


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate nnUNet held-out test set predictions (raw vs postprocessed)."
    )
    p.add_argument("--config", type=str, default="configs/nnunet_baseline.yaml")
    return p.parse_args()


def _load_cfg(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))["nnunet_baseline"]


# ── Metric computation ────────────────────────────────────────────────────────

def _load_binary(path: Path) -> tuple[np.ndarray, "sitk.Image"]:
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.uint8)
    arr = (arr > 0).astype(np.uint8)
    return arr, img


def _dice(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = float((pred & gt).sum())
    denom = float(pred.sum() + gt.sum())
    return 2.0 * inter / denom if denom > 0 else 1.0


def _sensitivity(pred: np.ndarray, gt: np.ndarray) -> float:
    tp = float((pred & gt).sum())
    fn = float(((1 - pred) & gt).sum())
    return tp / (tp + fn) if (tp + fn) > 0 else 1.0


def _precision(pred: np.ndarray, gt: np.ndarray) -> float:
    tp = float((pred & gt).sum())
    fp = float((pred & (1 - gt)).sum())
    return tp / (tp + fp) if (tp + fp) > 0 else 1.0


def _hd95(pred_arr: np.ndarray, gt_arr: np.ndarray,
          pred_img: "sitk.Image") -> float:
    """95th-percentile Hausdorff Distance in mm."""
    if pred_arr.sum() == 0 or gt_arr.sum() == 0:
        return float("nan")
    spacing = pred_img.GetSpacing()   # (x, y, z) in mm
    # Use SimpleITK distance maps for efficiency
    pred_sitk = sitk.GetImageFromArray(pred_arr.astype(np.uint8))
    pred_sitk.CopyInformation(pred_img)
    gt_sitk = sitk.GetImageFromArray(gt_arr.astype(np.uint8))
    gt_sitk.CopyInformation(pred_img)

    dist_pred = sitk.SignedMaurerDistanceMap(
        pred_sitk, squaredDistance=False, useImageSpacing=True
    )
    dist_gt = sitk.SignedMaurerDistanceMap(
        gt_sitk, squaredDistance=False, useImageSpacing=True
    )
    dist_pred_arr = sitk.GetArrayFromImage(dist_pred)
    dist_gt_arr   = sitk.GetArrayFromImage(dist_gt)

    # Surface distances: pred surface → GT, GT surface → pred
    pred_surface = (pred_arr == 1) & (
        sitk.GetArrayFromImage(
            sitk.BinaryContour(pred_sitk, fullyConnected=False)
        ) > 0
    )
    gt_surface = (gt_arr == 1) & (
        sitk.GetArrayFromImage(
            sitk.BinaryContour(gt_sitk, fullyConnected=False)
        ) > 0
    )

    if pred_surface.sum() == 0 or gt_surface.sum() == 0:
        return float("nan")

    d_pred_to_gt = np.abs(dist_gt_arr[pred_surface])
    d_gt_to_pred = np.abs(dist_pred_arr[gt_surface])
    all_dists = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    return float(np.percentile(all_dists, 95))


def _compute_case_metrics(pred_path: Path, gt_path: Path) -> dict:
    pred_arr, pred_img = _load_binary(pred_path)
    gt_arr,   _        = _load_binary(gt_path)

    return {
        "case_id":     pred_path.name.replace(".nii.gz", ""),
        "dice":        round(_dice(pred_arr, gt_arr), 4),
        "hd95":        round(_hd95(pred_arr, gt_arr, pred_img), 2),
        "sensitivity": round(_sensitivity(pred_arr, gt_arr), 4),
        "precision":   round(_precision(pred_arr, gt_arr), 4),
        "pred_volume_vox": int(pred_arr.sum()),
        "gt_volume_vox":   int(gt_arr.sum()),
    }


# ── Batch evaluation ──────────────────────────────────────────────────────────

def _evaluate_folder(pred_dir: Path, gt_dir: Path, label: str) -> list[dict]:
    pred_files = sorted(pred_dir.glob("*.nii.gz"))
    if not pred_files:
        print(f"  No predictions found in {pred_dir}", flush=True)
        return []

    results = []
    for pred_path in pred_files:
        case_id = pred_path.name.replace(".nii.gz", "")
        gt_path = gt_dir / f"{case_id}.nii.gz"
        if not gt_path.exists():
            print(f"  [WARN] GT not found for {case_id} — skipping", flush=True)
            continue
        metrics = _compute_case_metrics(pred_path, gt_path)
        results.append(metrics)
        print(
            f"  [{label}] {case_id}: "
            f"Dice={metrics['dice']:.3f}  "
            f"HD95={metrics['hd95']:.1f}mm  "
            f"Sens={metrics['sensitivity']:.3f}  "
            f"Prec={metrics['precision']:.3f}",
            flush=True,
        )
    return results


# ── Summary stats ─────────────────────────────────────────────────────────────

def _summary(results: list[dict]) -> dict:
    if not results:
        return {}
    for key in ["dice", "hd95", "sensitivity", "precision"]:
        vals = [r[key] for r in results if not np.isnan(r[key])]
        print(
            f"  {key:12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}"
            f"  [min={min(vals):.4f}  max={max(vals):.4f}]",
            flush=True,
        )
    return {
        key: {
            "mean": round(float(np.mean([r[key] for r in results if not np.isnan(r[key])])), 4),
            "std":  round(float(np.std( [r[key] for r in results if not np.isnan(r[key])])), 4),
            "min":  round(float(np.min( [r[key] for r in results if not np.isnan(r[key])])), 4),
            "max":  round(float(np.max( [r[key] for r in results if not np.isnan(r[key])])), 4),
        }
        for key in ["dice", "hd95", "sensitivity", "precision"]
    }


# ── Save helpers ──────────────────────────────────────────────────────────────

def _save_csv(results: list[dict], path: Path) -> None:
    if not results:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(results[0].keys())
    lines = [",".join(headers)]
    for r in results:
        lines.append(",".join(str(r[h]) for h in headers))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Saved CSV: {path}", flush=True)


def _print_table(raw_summary: dict, pp_summary: dict, out_path: Path) -> None:
    lines = [
        "=" * 70,
        "  nnUNet Baseline — Held-Out Test Set Metrics",
        "=" * 70,
        f"{'Metric':<14} {'Raw (no PP)':>18} {'Postprocessed':>18} {'Delta':>10}",
        "-" * 70,
    ]
    for key in ["dice", "hd95", "sensitivity", "precision"]:
        r = raw_summary.get(key, {})
        p = pp_summary.get(key, {})
        rv = r.get("mean", float("nan"))
        pv = p.get("mean", float("nan"))
        rs = r.get("std",  float("nan"))
        ps = p.get("std",  float("nan"))
        delta = pv - rv if not (np.isnan(rv) or np.isnan(pv)) else float("nan")
        sign = "+" if delta > 0 else ""
        lines.append(
            f"  {key:<12} {rv:>8.4f}±{rs:.4f}  {pv:>8.4f}±{ps:.4f}  {sign}{delta:>+8.4f}"
        )
    lines.append("=" * 70)
    lines.append("  Postprocessing: nnUNet default (removes small disconnected components)")
    lines.append("=" * 70)
    table = "\n".join(lines)
    print(f"\n{table}\n", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(table + "\n", encoding="utf-8")
    print(f"  Saved table: {out_path}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg  = _load_cfg(args.config)

    gt_dir      = (Path(cfg["nnunet_raw_data_base"])
                   / "nnUNet_raw_data" / str(cfg["task_name"]) / "labelsTs")
    pred_raw    = Path(cfg.get("predict_output_raw",
                               "/kaggle/working/nnunet_predictions/raw"))
    pred_postproc = Path(cfg.get("predict_output_postproc",
                                 "/kaggle/working/nnunet_predictions/postprocessed"))
    reports_dir = Path(cfg["results_folder"]) / "menrt_artifacts" / "reports"

    if not gt_dir.exists():
        raise FileNotFoundError(
            f"labelsTs not found: {gt_dir}\n"
            "Run the prepare stage first."
        )

    print(f"\nGround truth : {gt_dir}")
    print(f"Raw preds    : {pred_raw}")
    print(f"PP preds     : {pred_postproc}\n")

    raw_results, pp_results = [], []

    if pred_raw.exists():
        print("── Raw predictions ──────────────────────────────────────────────")
        raw_results = _evaluate_folder(pred_raw, gt_dir, "raw")
        print("\nSummary (raw):")
        raw_summary = _summary(raw_results)
        _save_csv(raw_results, reports_dir / "testset_metrics_raw.csv")
    else:
        print(f"Raw prediction folder not found: {pred_raw}")
        raw_summary = {}

    if pred_postproc.exists():
        print("\n── Postprocessed predictions ────────────────────────────────────")
        pp_results = _evaluate_folder(pred_postproc, gt_dir, "postproc")
        print("\nSummary (postprocessed):")
        pp_summary = _summary(pp_results)
        _save_csv(pp_results, reports_dir / "testset_metrics_postproc.csv")
    else:
        print(f"Postprocessed prediction folder not found: {pred_postproc}")
        pp_summary = {}

    # Comparison
    if raw_summary or pp_summary:
        payload = {
            "raw":           {"summary": raw_summary, "per_case": raw_results},
            "postprocessed": {"summary": pp_summary,  "per_case": pp_results},
        }
        cmp_json = reports_dir / "testset_comparison.json"
        cmp_json.parent.mkdir(parents=True, exist_ok=True)
        cmp_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\n  Saved comparison JSON: {cmp_json}")

        _print_table(raw_summary, pp_summary, reports_dir / "testset_comparison_table.txt")


if __name__ == "__main__":
    main()
