"""
scripts/mednext_to_prompts.py
==============================
Convert MedNeXt segmentation predictions to structured prompts for downstream
foundation model refinement (SAM-Med3D, MedSAM, SAM2, etc.).

Pipeline position
-----------------
  MedNeXt prediction (.nii.gz)
          │
          ▼
  This script  →  per-case JSON prompts
          │
          ▼
  Foundation model (SAM-Med3D / MedSAM / SAM2)  →  refined segmentation

Why prompts from MedNeXt?
  MedNeXt provides strong volumetric priors (bounding box, centroid, rough mask)
  that the foundation model can use as spatial guidance to refine boundaries —
  especially important for meningioma GTV where tumour borders are complex and
  depend on subtle MRI contrast.  This is the core novelty of the thesis pipeline.

Prompt formats generated per case
----------------------------------
  bbox_voxel     : [z_min, y_min, x_min, z_max, y_max, x_max]  in voxel coords
  bbox_mm        : same in physical mm coords (using NIfTI spacing)
  centroid_voxel : [z, y, x]  integer voxel index
  centroid_mm    : [z_mm, y_mm, x_mm]  physical position
  positive_points_voxel  : N_pos × [z, y, x]  points sampled inside the mask
  negative_points_voxel  : N_neg × [z, y, x]  points sampled near but outside
  positive_points_mm     : same in mm coords
  negative_points_mm     : same in mm coords
  mask_path      : absolute path to the .nii.gz prediction file
  spacing_zyx_mm : [sz, sy, sx]  voxel spacing in mm

The JSON format is deliberately generic so it can be consumed by adapters for
any foundation model:
  - SAM-Med3D  : uses bbox_voxel and positive_points_voxel
  - MedSAM     : project centroid onto 2D slices, use bbox_voxel Z-slices
  - SAM2       : uses positive_points_voxel + negative_points_voxel per slice

Output
------
  --out-dir/<case_id>_prompts.json    one file per case
  --out-dir/all_prompts_summary.json  manifest of all cases

Usage
-----
  # Generate prompts from fold-0 validation predictions:
  python scripts/mednext_to_prompts.py --config configs/mednext_nnunet.yaml \\
      --source val --fold 0

  # Generate prompts from test-set inference output:
  python scripts/mednext_to_prompts.py --config configs/mednext_nnunet.yaml \\
      --source test

  # Custom prediction directory:
  python scripts/mednext_to_prompts.py --config configs/mednext_nnunet.yaml \\
      --pred-dir /my/predictions/ --out-dir /my/prompts/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import yaml


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Convert MedNeXt predictions to foundation-model prompt JSONs. "
            "Part of the MedNeXt → Foundation Model refinement pipeline."
        )
    )
    p.add_argument("--config", type=str, default="configs/mednext_nnunet.yaml")
    p.add_argument(
        "--source",
        type=str,
        default="val",
        choices=["val", "test", "custom"],
        help=(
            "'val'  → fold validation_raw/ predictions (fold required). "
            "'test' → predict_output from config. "
            "'custom' → use --pred-dir."
        ),
    )
    p.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold index when --source val (default 0).",
    )
    p.add_argument(
        "--pred-dir",
        type=str,
        default="",
        help="Custom prediction directory (required when --source custom).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="",
        help=(
            "Output directory for prompt JSONs. "
            "Default: <results_folder>/menrt_repo_artifacts/prompts/<source>_fold<fold>/"
        ),
    )
    p.add_argument(
        "--n-pos",
        type=int,
        default=5,
        help="Number of positive (foreground) sample points per case (default 5).",
    )
    p.add_argument(
        "--n-neg",
        type=int,
        default=5,
        help="Number of negative (background near boundary) sample points per case (default 5).",
    )
    p.add_argument(
        "--neg-dilation-mm",
        type=float,
        default=5.0,
        help=(
            "Negative points are sampled from a shell of this thickness (mm) "
            "outside the mask boundary.  Default 5.0 mm."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible point sampling (default 42).",
    )
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


def _default_out_dir(cfg: dict, source: str, fold: int) -> Path:
    label = f"{source}_fold{fold}" if source == "val" else source
    return (
        Path(cfg["results_folder"])
        / "menrt_repo_artifacts"
        / "prompts"
        / label
    )


# ─────────────────────────────────────────────────────────────────────────────
# Prediction discovery
# ─────────────────────────────────────────────────────────────────────────────

def _collect_predictions(pred_dir: Path) -> dict[str, Path]:
    """Returns {case_id: path} for all .nii.gz files in pred_dir."""
    preds: dict[str, Path] = {}
    for p in sorted(pred_dir.glob("*.nii.gz")):
        case_id = p.stem
        if case_id.endswith(".nii"):     # handle .nii.gz double-extension
            case_id = Path(case_id).stem
        preds[case_id] = p
    return preds


# ─────────────────────────────────────────────────────────────────────────────
# Mask loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_mask_with_meta(path: Path) -> tuple[np.ndarray, list[float], list[float]]:
    """
    Load a binary NIfTI mask.

    Returns
    -------
    mask     : np.ndarray  uint8  (Z, Y, X)
    spacing  : [sz, sy, sx]  voxel spacing in mm  (Z, Y, X order)
    origin   : [oz, oy, ox]  physical origin in mm
    """
    itk_img = sitk.ReadImage(str(path))

    # SimpleITK stores spacing as (X, Y, Z); we convert to (Z, Y, X) for numpy
    sp_xyz   = itk_img.GetSpacing()            # (sx, sy, sz)
    spacing  = [sp_xyz[2], sp_xyz[1], sp_xyz[0]]   # → (sz, sy, sx)

    or_xyz   = itk_img.GetOrigin()             # (ox, oy, oz)
    origin   = [or_xyz[2], or_xyz[1], or_xyz[0]]   # → (oz, oy, ox)

    arr = sitk.GetArrayFromImage(itk_img)      # already (Z, Y, X)
    mask = (arr > 0).astype(np.uint8)
    return mask, spacing, origin


# ─────────────────────────────────────────────────────────────────────────────
# Geometric computations
# ─────────────────────────────────────────────────────────────────────────────

def _voxel_to_mm(
    point_zyx: list[int],
    spacing: list[float],
    origin: list[float],
) -> list[float]:
    """Convert a (z, y, x) voxel index to physical mm coordinates."""
    return [
        origin[i] + point_zyx[i] * spacing[i]
        for i in range(3)
    ]


def _bounding_box_voxel(mask: np.ndarray) -> list[int] | None:
    """
    Returns [z_min, y_min, x_min, z_max, y_max, x_max] in voxel space,
    or None if the mask is empty.
    """
    if mask.sum() == 0:
        return None
    zz, yy, xx = np.where(mask > 0)
    return [
        int(zz.min()), int(yy.min()), int(xx.min()),
        int(zz.max()), int(yy.max()), int(xx.max()),
    ]


def _centroid_voxel(mask: np.ndarray) -> list[int] | None:
    """Returns [z, y, x] integer centroid of the mask, or None if empty."""
    if mask.sum() == 0:
        return None
    zz, yy, xx = np.where(mask > 0)
    return [int(round(zz.mean())), int(round(yy.mean())), int(round(xx.mean()))]


def _sample_positive_points(
    mask: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> list[list[int]]:
    """
    Uniformly sample N points from inside the foreground mask.
    Returns list of [z, y, x] voxel coordinates.
    Points are spread across the Z axis to cover the full tumour extent
    (important for 3D foundation models like SAM-Med3D).
    """
    zz, yy, xx = np.where(mask > 0)
    if len(zz) == 0:
        return []

    # Stratified by Z to ensure axial coverage
    unique_z = np.unique(zz)
    chosen: list[list[int]] = []

    if len(unique_z) >= n:
        # Sample one point from each of N evenly-spaced Z slices
        z_indices = np.round(np.linspace(0, len(unique_z) - 1, n)).astype(int)
        selected_z = unique_z[z_indices]
        for z_val in selected_z:
            in_slice_mask = zz == z_val
            slice_y = yy[in_slice_mask]
            slice_x = xx[in_slice_mask]
            idx = rng.integers(0, len(slice_y))
            chosen.append([int(z_val), int(slice_y[idx]), int(slice_x[idx])])
    else:
        # Fewer unique Z-slices than requested points — sample with replacement
        idx = rng.choice(len(zz), size=n, replace=True)
        chosen = [[int(zz[i]), int(yy[i]), int(xx[i])] for i in idx]

    return chosen


def _sample_negative_points(
    mask: np.ndarray,
    spacing: list[float],
    neg_dilation_mm: float,
    n: int,
    rng: np.random.Generator,
) -> list[list[int]]:
    """
    Sample N negative (background) points from a thin shell just outside the
    mask boundary.

    Strategy:
      1. Dilate the mask by neg_dilation_mm in each axis (convert to voxels).
      2. Shell = dilated_mask AND NOT original_mask.
      3. Sample uniformly from the shell.

    Points near the boundary are harder negatives than random background,
    which pushes the foundation model to refine the boundary precisely.
    This is the recommended negative-point strategy in the SAM and SAM-Med3D
    papers.
    """
    if mask.sum() == 0:
        return []

    # Convert dilation distance from mm to voxels per axis (Z, Y, X)
    dil_voxels = [max(1, int(round(neg_dilation_mm / sp))) for sp in spacing]

    # Dilate with a box kernel of the computed size using scipy-free approach
    dilated = mask.copy().astype(np.uint8)
    for axis, dv in enumerate(dil_voxels):
        if dv < 1:
            continue
        # Rolling max along each axis — equivalent to morphological dilation
        # with a 1-D flat structuring element of size 2*dv+1
        padded = np.pad(dilated, [(dv, dv) if i == axis else (0, 0) for i in range(3)], mode="constant")
        result = np.zeros_like(dilated)
        for offset in range(-dv, dv + 1):
            slices = [
                slice(dv + offset, dv + offset + dilated.shape[ax]) if ax == axis else slice(None)
                for ax in range(3)
            ]
            result = np.maximum(result, padded[tuple(slices)])
        dilated = result

    shell = (dilated > 0) & (mask == 0)
    sz, sy, sx = np.where(shell)

    if len(sz) == 0:
        # Fallback: sample from any background voxel
        sz, sy, sx = np.where(mask == 0)
        if len(sz) == 0:
            return []

    idx = rng.choice(len(sz), size=min(n, len(sz)), replace=False)
    return [[int(sz[i]), int(sy[i]), int(sx[i])] for i in idx]


# ─────────────────────────────────────────────────────────────────────────────
# Per-case prompt generation
# ─────────────────────────────────────────────────────────────────────────────

def _generate_prompt(
    case_id: str,
    pred_path: Path,
    n_pos: int,
    n_neg: int,
    neg_dilation_mm: float,
    seed: int,
) -> dict:
    """
    Generate the complete prompt dictionary for one case.
    Returns a JSON-serialisable dict.
    """
    mask, spacing, origin = _load_mask_with_meta(pred_path)
    rng = np.random.default_rng(seed=seed)

    # ── Geometric primitives ────────────────────────────────────────────────
    bbox_voxel   = _bounding_box_voxel(mask)
    centroid_vxl = _centroid_voxel(mask)
    pos_pts_vxl  = _sample_positive_points(mask, n_pos, rng)
    neg_pts_vxl  = _sample_negative_points(mask, spacing, neg_dilation_mm, n_neg, rng)

    # ── Convert voxel → mm for all points ──────────────────────────────────
    def _pts_to_mm(pts: list[list[int]]) -> list[list[float]]:
        return [_voxel_to_mm(p, spacing, origin) for p in pts]

    bbox_mm = None
    if bbox_voxel is not None:
        zmin, ymin, xmin, zmax, ymax, xmax = bbox_voxel
        bbox_mm = [
            *_voxel_to_mm([zmin, ymin, xmin], spacing, origin),
            *_voxel_to_mm([zmax, ymax, xmax], spacing, origin),
        ]

    centroid_mm = _voxel_to_mm(centroid_vxl, spacing, origin) if centroid_vxl else None

    # ── Mask statistics (useful for downstream filtering) ──────────────────
    n_foreground = int(mask.sum())
    voxel_vol_mm3 = float(spacing[0] * spacing[1] * spacing[2])
    tumour_vol_cc = float(n_foreground * voxel_vol_mm3 / 1000.0)   # cm³

    return {
        # ── Identifiers ──────────────────────────────────────────────────
        "case_id":               case_id,
        "mask_path":             str(pred_path.resolve()),
        # ── Spatial metadata ─────────────────────────────────────────────
        "spacing_zyx_mm":        spacing,
        "origin_zyx_mm":         origin,
        "volume_shape_zyx":      list(mask.shape),
        # ── Prompt: bounding box ─────────────────────────────────────────
        # Format: [z_min, y_min, x_min, z_max, y_max, x_max]
        "bbox_voxel":            bbox_voxel,
        "bbox_mm":               bbox_mm,
        # ── Prompt: centroid ─────────────────────────────────────────────
        "centroid_voxel":        centroid_vxl,
        "centroid_mm":           centroid_mm,
        # ── Prompt: positive points (inside tumour) ───────────────────────
        # Format: list of [z, y, x]
        "positive_points_voxel": pos_pts_vxl,
        "positive_points_mm":    _pts_to_mm(pos_pts_vxl),
        # ── Prompt: negative points (near boundary, outside tumour) ───────
        "negative_points_voxel": neg_pts_vxl,
        "negative_points_mm":    _pts_to_mm(neg_pts_vxl),
        # ── Mask statistics ───────────────────────────────────────────────
        "foreground_voxels":     n_foreground,
        "tumour_volume_cc":      round(tumour_vol_cc, 4),
        "mask_is_empty":         (n_foreground == 0),
        # ── Sampling config (for reproducibility) ─────────────────────────
        "prompt_config": {
            "n_positive":        n_pos,
            "n_negative":        n_neg,
            "neg_dilation_mm":   neg_dilation_mm,
            "seed":              seed,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg  = _load_cfg(args.config)

    # ── Resolve prediction directory ───────────────────────────────────────
    if args.source == "val":
        pred_dir = _trainer_root(cfg) / f"fold_{args.fold}" / "validation_raw"
        if not pred_dir.exists():
            print(
                f"\n[prompts] ERROR: validation_raw not found:\n  {pred_dir}\n"
                f"Ensure fold {args.fold} training is complete.\n",
                file=sys.stderr,
            )
            sys.exit(1)
    elif args.source == "test":
        predict_out = str(cfg.get("predict_output", "")).strip()
        if not predict_out:
            print(
                "\n[prompts] ERROR: predict_output is not configured.\n"
                "Run  python scripts/run_mednext_nnunet.py --mode predict  first.\n",
                file=sys.stderr,
            )
            sys.exit(1)
        pred_dir = Path(predict_out)
    else:  # custom
        if not args.pred_dir:
            print("\n[prompts] ERROR: --pred-dir is required when --source custom.\n", file=sys.stderr)
            sys.exit(1)
        pred_dir = Path(args.pred_dir)

    if not pred_dir.exists():
        print(f"\n[prompts] ERROR: Prediction directory not found:\n  {pred_dir}\n", file=sys.stderr)
        sys.exit(1)

    # ── Resolve output directory ───────────────────────────────────────────
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else _default_out_dir(cfg, args.source, args.fold)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Discover predictions ───────────────────────────────────────────────
    all_preds = _collect_predictions(pred_dir)
    if not all_preds:
        print(f"\n[prompts] ERROR: No .nii.gz files found in:\n  {pred_dir}\n", file=sys.stderr)
        sys.exit(1)

    print(f"[prompts] Source        : {args.source} (fold {args.fold})", flush=True)
    print(f"[prompts] Prediction dir: {pred_dir}", flush=True)
    print(f"[prompts] Output dir    : {out_dir}", flush=True)
    print(f"[prompts] Cases found   : {len(all_preds)}", flush=True)
    print(f"[prompts] Points per case: {args.n_pos} positive, {args.n_neg} negative", flush=True)

    # ── Process each case ─────────────────────────────────────────────────
    summary_records: list[dict] = []
    n_empty = 0

    for idx, (case_id, pred_path) in enumerate(sorted(all_preds.items()), start=1):
        print(f"[prompts]   [{idx}/{len(all_preds)}] {case_id}", flush=True)

        try:
            prompt = _generate_prompt(
                case_id=case_id,
                pred_path=pred_path,
                n_pos=args.n_pos,
                n_neg=args.n_neg,
                neg_dilation_mm=args.neg_dilation_mm,
                seed=args.seed,
            )
        except Exception as exc:
            print(f"[prompts] WARNING: Failed to process {case_id}: {exc}", file=sys.stderr)
            continue

        if prompt["mask_is_empty"]:
            print(f"[prompts] WARNING: Empty prediction mask for {case_id}. Prompt still saved.", file=sys.stderr)
            n_empty += 1

        # Save per-case JSON
        case_json = out_dir / f"{case_id}_prompts.json"
        case_json.write_text(json.dumps(prompt, indent=2) + "\n", encoding="utf-8")

        summary_records.append(
            {
                "case_id":          case_id,
                "prompt_json":      str(case_json.resolve()),
                "foreground_voxels": prompt["foreground_voxels"],
                "tumour_volume_cc":  prompt["tumour_volume_cc"],
                "mask_is_empty":    prompt["mask_is_empty"],
                "bbox_voxel":       json.dumps(prompt["bbox_voxel"]),
                "centroid_voxel":   json.dumps(prompt["centroid_voxel"]),
                "n_pos_pts":        len(prompt["positive_points_voxel"]),
                "n_neg_pts":        len(prompt["negative_points_voxel"]),
            }
        )

    # ── Write manifest / summary ───────────────────────────────────────────
    summary_path = out_dir / "all_prompts_summary.json"
    summary_payload = {
        "source":           args.source,
        "fold":             args.fold if args.source == "val" else None,
        "pred_dir":         str(pred_dir),
        "out_dir":          str(out_dir),
        "total_cases":      len(summary_records),
        "empty_masks":      n_empty,
        "prompt_config": {
            "n_positive":   args.n_pos,
            "n_negative":   args.n_neg,
            "neg_dilation_mm": args.neg_dilation_mm,
            "seed":         args.seed,
        },
        "cases": summary_records,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")

    print(f"\n[prompts] Processed {len(summary_records)} case(s)  ({n_empty} empty mask(s)).", flush=True)
    print(f"[prompts] Per-case JSONs saved to : {out_dir}", flush=True)
    print(f"[prompts] Summary manifest saved  : {summary_path}", flush=True)
    print("[prompts] Done.", flush=True)


if __name__ == "__main__":
    main()
