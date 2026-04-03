from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export official MedNeXt nnU-Net(v1) training artifacts to real CSV/JSON reports.")
    p.add_argument("--config", type=str, default="configs/mednext_nnunet.yaml")
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--folds", type=int, nargs="*", default=None)
    return p.parse_args()


def _load_cfg(path: str) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return cfg["mednext_nnunet"]


def _trainer_root(cfg: dict) -> Path:
    return (
        Path(cfg["results_folder"])
        / "nnUNet"
        / str(cfg["network"])
        / str(cfg["task_name"])
        / f'{cfg["trainer_name"]}__{cfg["plans_identifier"]}'
    )


def _default_out_dir(cfg: dict) -> Path:
    return Path(cfg["results_folder"]) / "menrt_repo_artifacts" / "reports"


def _metric_block_from_summary(summary: dict) -> dict:
    mean_block = summary.get("results", {}).get("mean", {})
    if "1" in mean_block:
        return mean_block["1"]
    numeric_keys = sorted([k for k in mean_block.keys() if str(k).isdigit() and str(k) != "0"], key=str)
    if numeric_keys:
        return mean_block[numeric_keys[0]]
    return {}


def _extract_case_metrics(summary: dict, fold: int) -> list[dict]:
    rows: list[dict] = []
    all_cases = summary.get("results", {}).get("all", [])
    for item in all_cases:
        metric_block = item.get("1")
        if metric_block is None:
            numeric_keys = sorted([k for k in item.keys() if str(k).isdigit() and str(k) != "0"], key=str)
            metric_block = item.get(numeric_keys[0], {}) if numeric_keys else {}
        case_ref = str(item.get("reference", item.get("test", "")))
        case_id = Path(case_ref).stem if case_ref else ""
        if case_id.endswith(".nii"):
            case_id = Path(case_id).stem
        rows.append(
            {
                "fold": fold,
                "case_id": case_id,
                "reference": str(item.get("reference", "")),
                "prediction": str(item.get("test", "")),
                "dice": metric_block.get("Dice"),
                "jaccard": metric_block.get("Jaccard"),
                "hd95": metric_block.get("Hausdorff Distance 95"),
                "precision": metric_block.get("Precision"),
                "recall": metric_block.get("Recall"),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    cfg = _load_cfg(args.config)
    trainer_root = _trainer_root(cfg)
    folds = args.folds if args.folds is not None and len(args.folds) > 0 else list(cfg.get("folds", [0, 1, 2, 3, 4]))
    out_dir = Path(args.out_dir) if args.out_dir else _default_out_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    per_case_rows: list[dict] = []

    for fold in folds:
        fold_dir = trainer_root / f"fold_{fold}"
        summary_path = fold_dir / "validation_raw" / "summary.json"
        debug_json = fold_dir / "debug.json"
        progress_png = fold_dir / "progress.png"
        best_model = fold_dir / "model_best.model"
        latest_model = fold_dir / "model_latest.model"
        final_model = fold_dir / "model_final_checkpoint.model"

        row = {
            "fold": fold,
            "fold_dir": str(fold_dir),
            "debug_json_exists": debug_json.exists(),
            "progress_png_exists": progress_png.exists(),
            "model_best_exists": best_model.exists(),
            "model_latest_exists": latest_model.exists(),
            "model_final_exists": final_model.exists(),
            "summary_json_exists": summary_path.exists(),
            "dice_mean": None,
            "jaccard_mean": None,
            "hd95_mean": None,
            "precision_mean": None,
            "recall_mean": None,
        }

        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            metrics = _metric_block_from_summary(summary)
            row["dice_mean"] = metrics.get("Dice")
            row["jaccard_mean"] = metrics.get("Jaccard")
            row["hd95_mean"] = metrics.get("Hausdorff Distance 95")
            row["precision_mean"] = metrics.get("Precision")
            row["recall_mean"] = metrics.get("Recall")
            per_case_rows.extend(_extract_case_metrics(summary, fold))

        manifest_rows.append(row)

    manifest_csv = out_dir / "mednext_fold_manifest.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()) if manifest_rows else ["fold"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    per_case_csv = out_dir / "mednext_per_case_metrics.csv"
    with per_case_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "fold",
            "case_id",
            "reference",
            "prediction",
            "dice",
            "jaccard",
            "hd95",
            "precision",
            "recall",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_case_rows)

    summary_json = out_dir / "mednext_results_export.json"
    summary_json.write_text(
        json.dumps(
            {
                "trainer_root": str(trainer_root),
                "folds": folds,
                "manifest_csv": str(manifest_csv),
                "per_case_csv": str(per_case_csv),
                "num_manifest_rows": len(manifest_rows),
                "num_per_case_rows": len(per_case_rows),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Saved manifest CSV: {manifest_csv}")
    print(f"Saved per-case CSV: {per_case_csv}")
    print(f"Saved export JSON: {summary_json}")


if __name__ == "__main__":
    main()
