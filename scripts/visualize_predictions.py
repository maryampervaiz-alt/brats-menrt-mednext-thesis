"""
scripts/visualize_predictions.py
=================================
Qualitative segmentation overlay visualization for the MedNeXt MEN-RT pipeline.

For each selected case this script produces a multi-panel figure showing:
  Column 1 — T1c MRI only  (grey)
  Column 2 — T1c + ground-truth GTV  (green overlay)
  Column 3 — T1c + ground-truth + MedNeXt prediction  (green = GT, red = pred,
              yellow = overlap between GT and pred)

Three canonical orientations are shown (rows):
  Row 1 — Axial slice at maximum GT foreground
  Row 2 — Coronal slice at GT centroid
  Row 3 — Sagittal slice at GT centroid

Why these slice positions? The slice with the most foreground voxels gives the
most visually informative view of tumour extent; the centroid slices provide a
consistent reference across cases.  This is the standard approach in MICCAI
segmentation papers (e.g. nnU-Net, TransUNet, UNETR).

Modes
-----
  --mode val   : visualise training-fold validation predictions
                 (stored in fold_X/validation_raw/).  GT is available.
  --mode test  : visualise inference output on the held-out test set
                 (stored under predict_output config key).  No GT — shows
                 prediction only.

Usage
-----
  # After training fold 0, visualise best+worst 3 validation cases:
  python scripts/visualize_predictions.py --config configs/mednext_nnunet.yaml \\
      --fold 0 --mode val --num-cases 3

  # Visualise test-set predictions (after running --mode predict):
  python scripts/visualize_predictions.py --config configs/mednext_nnunet.yaml \\
      --mode test --num-cases 5

  # Visualise specific cases listed in best_worst_cases.csv:
  python scripts/visualize_predictions.py --config configs/mednext_nnunet.yaml \\
      --fold 0 --mode val --best-worst-csv /path/to/best_worst_cases.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
# Non-interactive backend — safe on Kaggle / headless GPU servers
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import SimpleITK as sitk
import yaml


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate T1c / GT / prediction overlay figures for the MedNeXt "
            "MEN-RT pipeline.  Run after training is complete."
        )
    )
    p.add_argument("--config", type=str, default="configs/mednext_nnunet.yaml")
    p.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Which training fold to use (validation predictions). Ignored in test mode.",
    )
    p.add_argument(
        "--mode",
        type=str,
        default="val",
        choices=["val", "test"],
        help=(
            "'val' → use fold validation_raw/ predictions (GT available). "
            "'test' → use predict_output predictions (no GT)."
        ),
    )
    p.add_argument(
        "--num-cases",
        type=int,
        default=4,
        help="Number of cases to visualise (default 4). Ignored if --best-worst-csv is set.",
    )
    p.add_argument(
        "--best-worst-csv",
        type=str,
        default="",
        help=(
            "Path to best_worst_cases.csv from plot_results.py. "
            "If given, visualises the best + worst N cases instead of random selection."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output directory. Default: <results_folder>/menrt_repo_artifacts/figures/overlays/",
    )
    p.add_argument("--dpi", type=int, default=200,
                   help="DPI for output PNGs (default 200).")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_cfg(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))["mednext_nnunet"]


def _trainer_root(cfg: dict) -> Path:
    return (
        Path(cfg["results_folder"])
        / "nnUNet"
        / str(cfg["network"])
        / str(cfg["task_name"])
        / f'{cfg["trainer_name"]}__{cfg["plans_identifier"]}'
    )


def _raw_task_dir(cfg: dict) -> Path:
    return (
        Path(cfg["nnunet_raw_data_base"])
        / "nnUNet_raw_data"
        / str(cfg["task_name"])
    )


# ─────────────────────────────────────────────────────────────────────────────
# Case discovery
# ─────────────────────────────────────────────────────────────────────────────

def _find_prediction_files(pred_dir: Path) -> dict[str, Path]:
    """
    Returns {case_id: nii_gz_path} for all .nii.gz files in pred_dir.
    nnU-Net names predictions as  <case_id>.nii.gz  (no channel suffix).
    """
    preds: dict[str, Path] = {}
    for p in sorted(pred_dir.glob("*.nii.gz")):
        case_id = p.stem.replace(".nii", "")   # strip double extension if present
        preds[case_id] = p
    return preds


def _load_best_worst_case_ids(csv_path: str) -> list[str]:
    """Read the case_id column from best_worst_cases.csv (output of plot_results.py)."""
    ids: list[str] = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = row.get("case_id", "").strip()
            if cid and cid not in ids:
                ids.append(cid)
    return ids


# ─────────────────────────────────────────────────────────────────────────────
# NIfTI loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_volume(path: Path) -> np.ndarray:
    """
    Load a NIfTI volume with SimpleITK and return it as a float32 numpy array
    in (Z, Y, X) axis order (standard for axial/coronal/sagittal indexing).
    """
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    return arr  # already (Z, Y, X) from SimpleITK


def _load_mask(path: Path) -> np.ndarray:
    """Load a binary segmentation mask as uint8 (Z, Y, X)."""
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    return (arr > 0).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Slice selection
# ─────────────────────────────────────────────────────────────────────────────

def _best_axial_slice(mask: np.ndarray) -> int:
    """
    Return the axial (Z) index with the maximum number of foreground voxels.
    This is the most informative slice for GTV visualisation.
    Returns the middle slice as fallback if the mask is empty.
    """
    counts = mask.sum(axis=(1, 2))          # shape: (Z,)
    if counts.max() == 0:
        return mask.shape[0] // 2           # empty mask → use middle
    return int(np.argmax(counts))


def _centroid_3d(mask: np.ndarray) -> tuple[int, int, int]:
    """
    Compute the integer centroid (z, y, x) of the mask.
    Falls back to volume centre if mask is empty.
    """
    nz, ny, nx = mask.shape
    if mask.sum() == 0:
        return nz // 2, ny // 2, nx // 2
    z_idx, y_idx, x_idx = np.where(mask > 0)
    return (
        int(np.round(z_idx.mean())),
        int(np.round(y_idx.mean())),
        int(np.round(x_idx.mean())),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Intensity normalisation for display
# ─────────────────────────────────────────────────────────────────────────────

def _window_level(slice_2d: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0) -> np.ndarray:
    """
    Clip to [low_pct, high_pct] percentile and normalise to [0, 1].
    Equivalent to standard MRI window/level adjustment used in clinical viewers.
    """
    lo = float(np.percentile(slice_2d, low_pct))
    hi = float(np.percentile(slice_2d, high_pct))
    if hi <= lo:
        return np.zeros_like(slice_2d, dtype=np.float32)
    clipped = np.clip(slice_2d, lo, hi)
    return ((clipped - lo) / (hi - lo)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Overlay composition
# ─────────────────────────────────────────────────────────────────────────────

def _make_overlay_rgb(
    grey_2d: np.ndarray,
    gt_2d: np.ndarray | None,
    pred_2d: np.ndarray | None,
) -> np.ndarray:
    """
    Compose an RGB overlay from a normalised greyscale slice and optional
    binary masks (GT and/or prediction).

    Colour scheme (chosen for maximum contrast and colourblind accessibility):
        GT only     →  solid green  (R=0,   G=200, B=0)
        Pred only   →  solid red    (R=220, G=0,   B=0)
        Overlap     →  solid yellow (R=220, G=220, B=0)

    The alpha for mask overlays is 0.45 so the underlying MRI texture remains
    visible — important for anatomical context in publications.
    """
    alpha = 0.45                    # mask opacity
    grey_norm = _window_level(grey_2d)

    # Start with greyscale RGB
    rgb = np.stack([grey_norm, grey_norm, grey_norm], axis=-1)

    if gt_2d is None and pred_2d is None:
        return (rgb * 255).astype(np.uint8)

    # Build overlay mask channels
    gt   = (gt_2d   > 0) if gt_2d   is not None else np.zeros_like(grey_2d, dtype=bool)
    pred = (pred_2d > 0) if pred_2d is not None else np.zeros_like(grey_2d, dtype=bool)

    gt_only      = gt   & ~pred       # green
    pred_only    = pred & ~gt          # red
    both         = gt   &  pred        # yellow

    # Apply colours via alpha blending:  out = (1-alpha)*grey + alpha*colour
    for mask_2d, colour in [
        (gt_only,   (0.0,   0.78,  0.0)),
        (pred_only, (0.86,  0.0,   0.0)),
        (both,      (0.86,  0.86,  0.0)),
    ]:
        if mask_2d.any():
            for ch, val in enumerate(colour):
                rgb[mask_2d, ch] = (1.0 - alpha) * rgb[mask_2d, ch] + alpha * val

    return np.clip(rgb, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Single-case figure
# ─────────────────────────────────────────────────────────────────────────────

def _plot_case(
    case_id: str,
    image_vol: np.ndarray,
    gt_vol: np.ndarray | None,
    pred_vol: np.ndarray | None,
    out_path: Path,
    dpi: int,
    mode: str,
    dice: str = "",
    hd95: str = "",
) -> None:
    """
    Generate a 3-row × 3-column figure for one case.
    Rows: axial at max foreground, coronal at centroid, sagittal at centroid.
    Columns: T1c | T1c+GT | T1c+GT+Pred   (val mode)
             T1c | T1c+Pred               (test mode, no GT)
    """
    # Select reference mask for slice positions
    ref_mask = gt_vol if gt_vol is not None else pred_vol
    if ref_mask is None:
        print(f"[visualize] WARNING: No mask found for {case_id}. Skipping.", file=sys.stderr)
        return

    z_ax = _best_axial_slice(ref_mask)
    z_c, y_c, x_c = _centroid_3d(ref_mask)

    # Extract 2-D slices (ensure index stays in bounds)
    nz, ny, nx = image_vol.shape
    z_ax = int(np.clip(z_ax, 0, nz - 1))
    z_c  = int(np.clip(z_c,  0, nz - 1))
    y_c  = int(np.clip(y_c,  0, ny - 1))
    x_c  = int(np.clip(x_c,  0, nx - 1))

    slices_img = {
        "Axial":    image_vol[z_ax, :, :],
        "Coronal":  image_vol[:, y_c, :],
        "Sagittal": image_vol[:, :, x_c],
    }
    slices_gt = {
        "Axial":    gt_vol[z_ax, :, :]  if gt_vol   is not None else None,
        "Coronal":  gt_vol[:,  y_c, :]  if gt_vol   is not None else None,
        "Sagittal": gt_vol[:, :,  x_c]  if gt_vol   is not None else None,
    }
    slices_pred = {
        "Axial":    pred_vol[z_ax, :, :]  if pred_vol is not None else None,
        "Coronal":  pred_vol[:,  y_c, :]  if pred_vol is not None else None,
        "Sagittal": pred_vol[:, :,  x_c]  if pred_vol is not None else None,
    }

    orientations = ["Axial", "Coronal", "Sagittal"]
    n_cols = 3 if mode == "val" else 2
    col_titles = (
        ["T1c only", "T1c + GT (green)", "T1c + GT + Pred\n(green=GT  red=pred  yellow=overlap)"]
        if mode == "val"
        else ["T1c only", "T1c + Prediction (red)"]
    )

    fig, axes = plt.subplots(
        len(orientations), n_cols,
        figsize=(4.0 * n_cols, 4.0 * len(orientations)),
    )
    # Ensure 2-D indexing even with 1 row or 1 column
    if len(orientations) == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for row_idx, orient in enumerate(orientations):
        img_slice  = slices_img[orient]
        gt_slice   = slices_gt[orient]
        pred_slice = slices_pred[orient]

        # Column 0: T1c only
        axes[row_idx, 0].imshow(
            _window_level(img_slice),
            cmap="gray", aspect="equal", origin="lower",
        )
        axes[row_idx, 0].set_ylabel(orient, fontsize=10, fontweight="bold")

        # Column 1: T1c + GT  (val mode)  OR  T1c + Pred  (test mode)
        if mode == "val":
            axes[row_idx, 1].imshow(
                _make_overlay_rgb(img_slice, gt_slice, None),
                aspect="equal", origin="lower",
            )
            if n_cols == 3:
                # Column 2: T1c + GT + Pred
                axes[row_idx, 2].imshow(
                    _make_overlay_rgb(img_slice, gt_slice, pred_slice),
                    aspect="equal", origin="lower",
                )
        else:
            axes[row_idx, 1].imshow(
                _make_overlay_rgb(img_slice, None, pred_slice),
                aspect="equal", origin="lower",
            )

        for ax in axes[row_idx]:
            ax.set_xticks([])
            ax.set_yticks([])

    # Column headers
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=9, pad=4)

    # Legend patches
    legend_handles = []
    if mode == "val":
        legend_handles = [
            mpatches.Patch(color=(0.0, 0.78, 0.0), alpha=0.8, label="GT only"),
            mpatches.Patch(color=(0.86, 0.0, 0.0), alpha=0.8, label="Pred only"),
            mpatches.Patch(color=(0.86, 0.86, 0.0), alpha=0.8, label="Overlap"),
        ]
    else:
        legend_handles = [
            mpatches.Patch(color=(0.86, 0.0, 0.0), alpha=0.8, label="Prediction"),
        ]

    # Overall figure title with metrics
    metric_str = ""
    if dice:
        metric_str = f"  |  Dice = {float(dice):.4f}"
    if hd95:
        metric_str += f"  |  HD95 = {float(hd95):.1f} mm"
    fig.suptitle(
        f"Case: {case_id}{metric_str}",
        fontsize=11, fontweight="bold", y=1.01,
    )

    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=len(legend_handles),
            fontsize=9,
            framealpha=0.9,
            bbox_to_anchor=(0.5, -0.02),
        )

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] Saved → {out_path}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg  = _load_cfg(args.config)

    # ── Resolve directories ────────────────────────────────────────────────
    task_dir    = _raw_task_dir(cfg)
    images_tr   = task_dir / "imagesTr"    # T1c input images
    labels_tr   = task_dir / "labelsTr"    # GT segmentation masks

    if args.mode == "val":
        fold_dir = _trainer_root(cfg) / f"fold_{args.fold}"
        pred_dir = fold_dir / "validation_raw"
        if not pred_dir.exists():
            print(
                f"\n[visualize] ERROR: validation_raw not found at:\n  {pred_dir}\n"
                f"Ensure fold {args.fold} has completed training and nnU-Net validation.\n",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        # test mode: use predict_output from config
        predict_out = str(cfg.get("predict_output", "")).strip()
        if not predict_out:
            print(
                "\n[visualize] ERROR: predict_output is not set in config.\n"
                "Run  python scripts/run_mednext_nnunet.py --mode predict  first.\n",
                file=sys.stderr,
            )
            sys.exit(1)
        pred_dir = Path(predict_out)
        if not pred_dir.exists():
            print(
                f"\n[visualize] ERROR: predict_output directory not found:\n  {pred_dir}\n",
                file=sys.stderr,
            )
            sys.exit(1)

    # ── Output directory ───────────────────────────────────────────────────
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = (
            Path(cfg["results_folder"])
            / "menrt_repo_artifacts"
            / "figures"
            / "overlays"
            / f"{args.mode}_fold{args.fold}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Discover predictions ───────────────────────────────────────────────
    all_preds = _find_prediction_files(pred_dir)
    if not all_preds:
        print(
            f"\n[visualize] ERROR: No .nii.gz prediction files found in:\n  {pred_dir}\n",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[visualize] Found {len(all_preds)} prediction file(s) in: {pred_dir}", flush=True)

    # ── Select cases to visualise ──────────────────────────────────────────
    # Priority 1: best_worst_cases.csv (most informative for thesis)
    # Priority 2: random selection up to --num-cases
    if args.best_worst_csv and Path(args.best_worst_csv).exists():
        target_ids = _load_best_worst_case_ids(args.best_worst_csv)
        # Keep only those that have predictions
        selected = {cid: all_preds[cid] for cid in target_ids if cid in all_preds}
        print(
            f"[visualize] Using best_worst_cases.csv: {len(selected)} case(s) selected.",
            flush=True,
        )
    else:
        n_select = min(args.num_cases, len(all_preds))
        # Deterministic sort so results are reproducible
        selected = dict(list(sorted(all_preds.items()))[:n_select])
        print(f"[visualize] Selected first {n_select} case(s) for visualisation.", flush=True)

    if not selected:
        print("[visualize] WARNING: No cases selected. Nothing to visualise.", file=sys.stderr)
        return

    # ── Optional: load per-case metrics for title annotations ─────────────
    # Read best_worst_cases.csv to get Dice / HD95 for annotation
    case_metrics: dict[str, dict[str, str]] = {}
    if args.best_worst_csv and Path(args.best_worst_csv).exists():
        with open(args.best_worst_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cid = row.get("case_id", "")
                if cid:
                    case_metrics[cid] = {"dice": row.get("dice", ""), "hd95": row.get("hd95", "")}

    # ── Generate figure for each selected case ─────────────────────────────
    for case_id, pred_path in selected.items():
        print(f"[visualize] Processing case: {case_id}", flush=True)

        # Load prediction mask
        try:
            pred_vol = _load_mask(pred_path)
        except Exception as exc:
            print(f"[visualize] WARNING: Could not load prediction {pred_path}: {exc}", file=sys.stderr)
            continue

        # Load T1c image (expected at imagesTr/<case_id>_0000.nii.gz)
        img_path = images_tr / f"{case_id}_0000.nii.gz"
        if not img_path.exists():
            print(f"[visualize] WARNING: T1c image not found: {img_path}. Skipping.", file=sys.stderr)
            continue
        try:
            image_vol = _load_volume(img_path)
        except Exception as exc:
            print(f"[visualize] WARNING: Could not load image {img_path}: {exc}", file=sys.stderr)
            continue

        # Load GT label (only in val mode, where labels are available)
        gt_vol: np.ndarray | None = None
        if args.mode == "val":
            lbl_path = labels_tr / f"{case_id}.nii.gz"
            if lbl_path.exists():
                try:
                    gt_vol = _load_mask(lbl_path)
                except Exception as exc:
                    print(f"[visualize] WARNING: Could not load label {lbl_path}: {exc}", file=sys.stderr)
            else:
                print(f"[visualize] WARNING: GT label not found for {case_id}: {lbl_path}", file=sys.stderr)

        # Retrieve metrics for title annotation (if available)
        met  = case_metrics.get(case_id, {})
        dice = met.get("dice", "")
        hd95 = met.get("hd95", "")

        # Sanitise case_id for use as a filename
        safe_id  = case_id.replace("/", "_").replace("\\", "_")
        out_path = out_dir / f"{safe_id}_overlay.png"

        _plot_case(
            case_id=case_id,
            image_vol=image_vol,
            gt_vol=gt_vol,
            pred_vol=pred_vol,
            out_path=out_path,
            dpi=args.dpi,
            mode=args.mode,
            dice=dice,
            hd95=hd95,
        )

    print(f"\n[visualize] All overlays saved to: {out_dir}", flush=True)
    print("[visualize] Done.", flush=True)


if __name__ == "__main__":
    main()
