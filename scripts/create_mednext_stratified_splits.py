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
    p = argparse.ArgumentParser(description="Create stratified nnU-Net(v1) CV splits for MEN-RT.")
    p.add_argument("--config", type=str, default="configs/mednext_nnunet.yaml")
    return p.parse_args()


def _load_cfg(path: str) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return cfg["mednext_nnunet"]


def _raw_task_dir(cfg: dict) -> Path:
    return Path(cfg["nnunet_raw_data_base"]) / "nnUNet_raw_data" / str(cfg["task_name"])


def _preprocessed_task_dir(cfg: dict) -> Path:
    return Path(cfg["nnunet_preprocessed"]) / str(cfg["task_name"])


def _artifacts_dir(cfg: dict) -> Path:
    return Path(cfg["results_folder"]) / "menrt_repo_artifacts"


def _foreground_voxel_count(label_path: Path) -> int:
    img = sitk.ReadImage(str(label_path))
    arr = sitk.GetArrayViewFromImage(img)
    return int(np.count_nonzero(arr))


def _assign_volume_strata(case_ids: list[str], labels_tr_dir: Path, num_bins: int) -> dict[str, dict[str, int]]:
    scored = []
    for case_id in case_ids:
        label_path = labels_tr_dir / f"{case_id}.nii.gz"
        if not label_path.exists():
            raise FileNotFoundError(f"Label not found for split generation: {label_path}")
        scored.append((case_id, _foreground_voxel_count(label_path)))

    scored.sort(key=lambda x: (x[1], x[0]))
    total = len(scored)
    if total == 0:
        raise RuntimeError("No training cases available for split generation.")

    num_bins = max(1, min(int(num_bins), total))
    strata: dict[str, dict[str, int]] = {}
    for idx, (case_id, voxels) in enumerate(scored):
        stratum_id = min(num_bins - 1, int(idx * num_bins / total))
        strata[case_id] = {"foreground_voxels": int(voxels), "stratum_id": int(stratum_id)}
    return strata


def _load_selected_case_ids(raw_task_dir: Path) -> list[str]:
    subset_manifest = raw_task_dir / "subset_manifest.json"
    if subset_manifest.exists():
        payload = json.loads(subset_manifest.read_text(encoding="utf-8"))
        case_ids = [str(x) for x in payload.get("selected_train_case_ids", [])]
        if case_ids:
            return sorted(case_ids)

    dataset_json = raw_task_dir / "dataset.json"
    if not dataset_json.exists():
        raise FileNotFoundError(f"dataset.json not found: {dataset_json}")

    payload = json.loads(dataset_json.read_text(encoding="utf-8"))
    case_ids = []
    for item in payload.get("training", []):
        image_ref = str(item.get("image", ""))
        case_id = Path(image_ref).stem
        if case_id.endswith(".nii"):
            case_id = Path(case_id).stem
        case_ids.append(case_id)
    return sorted(case_ids)


def _build_stratified_splits(case_ids: list[str], strata: dict[str, dict[str, int]], folds: list[int], seed: int) -> list[dict]:
    num_folds = len(folds)
    if num_folds <= 0:
        raise RuntimeError("At least one fold is required.")
    if len(case_ids) < num_folds:
        raise RuntimeError("Number of selected cases is smaller than the number of folds.")

    grouped: dict[int, list[str]] = defaultdict(list)
    for case_id in case_ids:
        grouped[int(strata[case_id]["stratum_id"])].append(case_id)

    rng = random.Random(seed)
    fold_val_sets: list[list[str]] = [[] for _ in range(num_folds)]
    for stratum_id in sorted(grouped):
        cases = sorted(grouped[stratum_id])
        rng.shuffle(cases)
        for idx, case_id in enumerate(cases):
            fold_val_sets[idx % num_folds].append(case_id)

    splits = []
    all_case_ids = sorted(case_ids)
    for fold_index in range(num_folds):
        val_ids = sorted(fold_val_sets[fold_index])
        val_id_set = set(val_ids)
        train_ids = [case_id for case_id in all_case_ids if case_id not in val_id_set]
        splits.append({"train": train_ids, "val": val_ids})
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

    case_ids = _load_selected_case_ids(raw_task_dir)
    folds = list(cfg.get("folds", [0, 1, 2, 3, 4]))
    split_seed = int(cfg.get("split_seed", cfg.get("subset_seed", 42)) or 42)
    num_bins = int(cfg.get("stratify_volume_bins", 5) or 5)

    strata = _assign_volume_strata(case_ids, labels_tr_dir, num_bins)
    splits = _build_stratified_splits(case_ids, strata, folds, split_seed)

    splits_path = preprocessed_task_dir / "splits_final.pkl"
    with splits_path.open("wb") as f:
        pickle.dump(splits, f, protocol=pickle.HIGHEST_PROTOCOL)

    summary = {
        "task_name": str(cfg["task_name"]),
        "folds": folds,
        "num_cases": len(case_ids),
        "split_seed": split_seed,
        "stratify_volume_bins": num_bins,
        "splits_path": str(splits_path),
        "case_strata": [
            {
                "case_id": case_id,
                "foreground_voxels": int(strata[case_id]["foreground_voxels"]),
                "stratum_id": int(strata[case_id]["stratum_id"]),
            }
            for case_id in sorted(case_ids)
        ],
        "fold_summary": [
            {
                "fold": fold,
                "train_count": len(split["train"]),
                "val_count": len(split["val"]),
            }
            for fold, split in zip(folds, splits)
        ],
    }

    summary_path = artifacts_dir / "stratified_splits_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Saved stratified splits: {splits_path}")
    print(f"Saved split summary: {summary_path}")


if __name__ == "__main__":
    main()
