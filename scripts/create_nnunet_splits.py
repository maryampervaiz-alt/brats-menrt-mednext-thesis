"""Create stratified 5-fold cross-validation splits for nnUNet baseline.

Reads configs/nnunet_baseline.yaml and writes splits_final.pkl to the
nnUNet preprocessed task directory.  Each fold is balanced across GTV
volume strata so that small and large tumours appear in every fold.
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create stratified nnUNet CV splits for MEN-RT baseline.")
    p.add_argument("--config", type=str, default="configs/nnunet_baseline.yaml")
    return p.parse_args()


def _load_cfg(path: str) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return cfg["nnunet_baseline"]


def _raw_task_dir(cfg: dict) -> Path:
    return Path(cfg["nnunet_raw_data_base"]) / "nnUNet_raw_data" / str(cfg["task_name"])


def _preprocessed_task_dir(cfg: dict) -> Path:
    return Path(cfg["nnunet_preprocessed"]) / str(cfg["task_name"])


def _artifacts_dir(cfg: dict) -> Path:
    return Path(cfg["results_folder"]) / "menrt_artifacts"


def _foreground_voxel_count(label_path: Path) -> int:
    img = sitk.ReadImage(str(label_path))
    return int(np.count_nonzero(sitk.GetArrayViewFromImage(img)))


def _assign_volume_strata(
    case_ids: list[str], labels_tr_dir: Path, num_bins: int
) -> dict[str, dict[str, int]]:
    scored = []
    for cid in case_ids:
        lbl = labels_tr_dir / f"{cid}.nii.gz"
        if not lbl.exists():
            raise FileNotFoundError(f"Label not found for splits: {lbl}")
        scored.append((cid, _foreground_voxel_count(lbl)))
    scored.sort(key=lambda x: (x[1], x[0]))
    total = len(scored)
    num_bins = max(1, min(num_bins, total))
    strata: dict[str, dict[str, int]] = {}
    for idx, (cid, voxels) in enumerate(scored):
        strata[cid] = {
            "foreground_voxels": int(voxels),
            "stratum_id": min(num_bins - 1, int(idx * num_bins / total)),
        }
    return strata


def _load_train_case_ids(raw_task_dir: Path) -> list[str]:
    manifest = raw_task_dir / "subset_manifest.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        ids = [str(x) for x in payload.get("selected_train_case_ids", [])]
        if ids:
            return sorted(ids)
    dataset_json = raw_task_dir / "dataset.json"
    if not dataset_json.exists():
        raise FileNotFoundError(f"dataset.json not found: {dataset_json}")
    payload = json.loads(dataset_json.read_text(encoding="utf-8"))
    ids = []
    for item in payload.get("training", []):
        ref = str(item.get("image", ""))
        cid = Path(ref).stem
        if cid.endswith(".nii"):
            cid = Path(cid).stem
        ids.append(cid)
    return sorted(ids)


def _build_stratified_splits(
    case_ids: list[str],
    strata: dict[str, dict[str, int]],
    num_folds: int,
    seed: int,
) -> list[dict]:
    if len(case_ids) < num_folds:
        raise RuntimeError(
            f"Not enough cases ({len(case_ids)}) for {num_folds} folds."
        )
    grouped: dict[int, list[str]] = defaultdict(list)
    for cid in case_ids:
        grouped[strata[cid]["stratum_id"]].append(cid)

    rng = random.Random(seed)
    fold_val: list[list[str]] = [[] for _ in range(num_folds)]
    for sid in sorted(grouped):
        cases = sorted(grouped[sid])
        rng.shuffle(cases)
        for i, cid in enumerate(cases):
            fold_val[i % num_folds].append(cid)

    all_sorted = sorted(case_ids)
    splits = []
    for fold_idx in range(num_folds):
        val = sorted(fold_val[fold_idx])
        val_set = set(val)
        train = [cid for cid in all_sorted if cid not in val_set]
        splits.append({"train": train, "val": val})
    return splits


def main() -> None:
    args = parse_args()
    cfg = _load_cfg(args.config)

    raw_task_dir = _raw_task_dir(cfg)
    labels_tr_dir = raw_task_dir / "labelsTr"
    preprocessed_task_dir = _preprocessed_task_dir(cfg)
    artifacts_dir = _artifacts_dir(cfg)
    preprocessed_task_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    case_ids = _load_train_case_ids(raw_task_dir)
    folds = list(cfg.get("folds", [0, 1, 2, 3, 4]))
    num_folds = len(folds)
    seed = int(cfg.get("split_seed", cfg.get("subset_seed", 42)) or 42)
    num_bins = int(cfg.get("stratify_volume_bins", 5) or 5)

    print(f"Creating {num_folds}-fold stratified splits for {len(case_ids)} cases …", flush=True)
    strata = _assign_volume_strata(case_ids, labels_tr_dir, num_bins)
    splits = _build_stratified_splits(case_ids, strata, num_folds, seed)

    for fold_idx, split in enumerate(splits):
        print(f"  Fold {fold_idx}: {len(split['train'])} train / {len(split['val'])} val")

    splits_path = preprocessed_task_dir / "splits_final.pkl"
    with splits_path.open("wb") as f:
        pickle.dump(splits, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved: {splits_path}", flush=True)

    summary = {
        "task_name": str(cfg["task_name"]),
        "num_cases": len(case_ids),
        "num_folds": num_folds,
        "split_seed": seed,
        "stratify_volume_bins": num_bins,
        "splits_path": str(splits_path),
        "fold_summary": [
            {"fold": i, "train_count": len(s["train"]), "val_count": len(s["val"])}
            for i, s in enumerate(splits)
        ],
        "case_strata": [
            {
                "case_id": cid,
                "foreground_voxels": strata[cid]["foreground_voxels"],
                "stratum_id": strata[cid]["stratum_id"],
            }
            for cid in sorted(case_ids)
        ],
    }
    summary_path = artifacts_dir / "stratified_splits_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
