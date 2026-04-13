"""Prepare BraTS MEN-RT data in nnU-Net v1 raw task format.

Differences from the MedNeXt prepare script
--------------------------------------------
* Selects a held-out test set FROM the labeled training pool so that
  inference-time evaluation has ground-truth labels.
* Saves held-out labels to  <task_dir>/labelsTs/  for metric computation.
* Uses nnUNet (not MedNeXt) CLI commands in the generated command sheet.
* No trainer-name / plans-identifier arguments (nnUNet uses defaults).
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import shutil
from pathlib import Path
from typing import TypeVar

import numpy as np
import SimpleITK as sitk

T = TypeVar("T")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare BraTS MEN-RT in nnU-Net(v1) raw task format with a labeled held-out test set."
    )
    p.add_argument("--train-root", type=str, required=True)
    p.add_argument("--nnunet-raw-data-base", type=str, required=True)
    p.add_argument("--task-id", type=int, required=True)
    p.add_argument("--task-name", type=str, required=True)
    p.add_argument("--image-keyword", type=str, default="_t1c.nii.gz")
    p.add_argument("--label-keyword", type=str, default="_gtv.nii.gz")
    p.add_argument("--copy-mode", type=str, default="copy", choices=["copy", "hardlink"])
    p.add_argument("--train-case-limit", type=int, default=50,
                   help="Number of cases for 5-fold CV training (stratified)")
    p.add_argument("--holdout-test-limit", type=int, default=10,
                   help="Number of held-out test cases (selected from labeled pool, never in CV)")
    p.add_argument("--subset-seed", type=int, default=42)
    p.add_argument("--train-subset-strategy", type=str, default="stratified_label_volume",
                   choices=["random", "stratified_label_volume"])
    p.add_argument("--stratify-volume-bins", type=int, default=5)
    p.add_argument("--clean-output", action="store_true")
    return p.parse_args()


# ── File utilities ────────────────────────────────────────────────────────────

def _copy_or_link(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if mode == "hardlink":
        try:
            dst.hardlink_to(src)
            return
        except Exception:
            pass
    if src.suffix.lower() == ".nii" and dst.name.lower().endswith(".nii.gz"):
        with src.open("rb") as fsrc, gzip.open(dst, "wb", compresslevel=6) as fdst:
            shutil.copyfileobj(fsrc, fdst, length=1024 * 1024)
        return
    shutil.copy2(src, dst)


def _clear_dir(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return
    for item in path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def _progress(index: int, total: int, kind: str, case_id: str) -> None:
    if index == 1 or index == total or index % 25 == 0:
        print(f"  [{kind}] {index}/{total}: {case_id}", flush=True)


# ── Case discovery ────────────────────────────────────────────────────────────

def _match_in_dir(case_dir: Path, pattern: str) -> list[Path]:
    pl = pattern.lower()
    files = [p for p in case_dir.iterdir() if p.is_file()]
    exact = [p for p in files if p.name.lower().endswith(pl)]
    if exact:
        return sorted(exact)
    token = pl.replace(".nii.gz", "").replace(".nii", "").strip("_- ")
    return sorted([p for p in files if token in p.name.lower()])


def _collect_labeled_cases(root: Path, image_kw: str, label_kw: str) -> list[tuple[str, Path, Path]]:
    cases = []
    seen: set[Path] = set()
    for case_dir in sorted(root.iterdir()):
        if not case_dir.is_dir() or case_dir in seen:
            continue
        imgs = _match_in_dir(case_dir, image_kw)
        lbls = _match_in_dir(case_dir, label_kw)
        if not imgs or not lbls:
            continue
        if len(lbls) != 1:
            raise RuntimeError(f"Expected 1 label in {case_dir}, found {len(lbls)}")
        seen.add(case_dir)
        cases.append((case_dir.name, imgs[0], lbls[0]))
    return sorted(cases, key=lambda x: x[0])


# ── Volume stratification ─────────────────────────────────────────────────────

def _foreground_voxels(label_path: Path) -> int:
    img = sitk.ReadImage(str(label_path))
    return int(np.count_nonzero(sitk.GetArrayViewFromImage(img)))


def _assign_strata(cases: list[tuple[str, Path, Path]], num_bins: int) -> dict[str, int]:
    scored = []
    total = len(cases)
    for idx, (cid, _, lbl) in enumerate(cases, 1):
        scored.append((cid, _foreground_voxels(lbl)))
        _progress(idx, total, "strata", cid)
    scored.sort(key=lambda x: (x[1], x[0]))
    strata: dict[str, int] = {}
    for idx, (cid, _) in enumerate(scored):
        strata[cid] = min(num_bins - 1, int(idx * num_bins / len(scored)))
    return strata


def _select_stratified(
    cases: list[tuple[str, Path, Path]],
    limit: int,
    strata: dict[str, int],
    seed: int,
    num_bins: int,
) -> list[tuple[str, Path, Path]]:
    if limit <= 0 or limit >= len(cases):
        return list(cases)
    bins: dict[int, list[tuple[str, Path, Path]]] = {}
    for c in cases:
        bins.setdefault(strata[c[0]], []).append(c)
    rng = random.Random(seed)
    for b in bins:
        bins[b] = sorted(bins[b], key=lambda x: x[0])
        rng.shuffle(bins[b])
    total = len(cases)
    allocs: dict[int, int] = {}
    remainders: list[tuple[float, int]] = []
    for b, bc in bins.items():
        exact = limit * len(bc) / total
        allocs[b] = min(len(bc), int(exact))
        remainders.append((exact - allocs[b], b))
    left = limit - sum(allocs.values())
    for _, b in sorted(remainders, key=lambda x: (-x[0], x[1])):
        if left <= 0:
            break
        if allocs[b] < len(bins[b]):
            allocs[b] += 1
            left -= 1
    selected: list[tuple[str, Path, Path]] = []
    for b in sorted(bins):
        selected.extend(bins[b][: allocs[b]])
    return sorted(selected, key=lambda x: x[0])


def _select_random(
    cases: list[tuple[str, Path, Path]], limit: int, seed: int
) -> list[tuple[str, Path, Path]]:
    if limit <= 0 or limit >= len(cases):
        return list(cases)
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(cases)), k=limit))
    return [cases[i] for i in idx]


# ── dataset.json ──────────────────────────────────────────────────────────────

def _write_dataset_json(
    path: Path,
    task_name: str,
    train_cases: list[tuple[str, Path, Path]],
    test_cases: list[tuple[str, Path, Path]],
) -> None:
    payload = {
        "name": task_name,
        "description": "BraTS 2024 MEN-RT meningioma GTV segmentation — nnUNet baseline",
        "tensorImageSize": "3D",
        "reference": "BraTS 2024 MEN-RT",
        "licence": "See dataset terms",
        "release": "1.0",
        "modality": {"0": "T1c"},
        "labels": {"0": "background", "1": "GTV"},
        "numTraining": len(train_cases),
        "numTest": len(test_cases),
        "training": [
            {"image": f"./imagesTr/{cid}.nii.gz", "label": f"./labelsTr/{cid}.nii.gz"}
            for cid, _, _ in train_cases
        ],
        "test": [f"./imagesTs/{cid}.nii.gz" for cid, _, _ in test_cases],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_command_sheet(task_dir: Path, task_id: int, task_name: str) -> None:
    lines = [
        "# ── Set environment variables ───────────────────────────────────────────",
        "export nnUNet_raw_data_base='<ABS_PATH>'",
        "export nnUNet_preprocessed='<ABS_PATH>'",
        "export RESULTS_FOLDER='<ABS_PATH>'",
        "export NNUNET_MAX_EPOCHS=50",
        "",
        "# ── Plan and preprocess ─────────────────────────────────────────────────",
        f"nnUNet_plan_and_preprocess -t {task_id} --verify_dataset_integrity",
        "",
        "# ── Train (one fold) ────────────────────────────────────────────────────",
        f"nnUNet_train 3d_fullres nnUNetTrainerV2_MENRT {task_name} 0",
        "",
        "# ── Predict on held-out test set (ensemble all folds) ───────────────────",
        f"nnUNet_predict -i <imagesTs_dir> -o <output_dir> -t {task_name}",
        "    -m 3d_fullres -tr nnUNetTrainerV2_MENRT -p nnUNetPlansv2.1 -f 0 1 2 3 4",
        "",
        "# ── Evaluate ────────────────────────────────────────────────────────────",
        "nnUNet_evaluate_folder -ref <labelsTs_dir> -pred <output_dir> -l 1",
    ]
    (task_dir / "NNUNET_COMMANDS.txt").write_text("\n".join(lines), encoding="utf-8")


def _write_manifest(
    path: Path,
    all_train: list[tuple[str, Path, Path]],
    sel_train: list[tuple[str, Path, Path]],
    sel_test: list[tuple[str, Path, Path]],
    strata: dict[str, int],
    args: argparse.Namespace,
) -> None:
    payload = {
        "subset_seed": args.subset_seed,
        "train_case_limit": args.train_case_limit,
        "holdout_test_limit": args.holdout_test_limit,
        "train_subset_strategy": args.train_subset_strategy,
        "stratify_volume_bins": args.stratify_volume_bins,
        "full_train_pool_count": len(all_train),
        "selected_train_count": len(sel_train),
        "selected_test_count": len(sel_test),
        "selected_train_case_ids": [cid for cid, _, _ in sel_train],
        "selected_test_case_ids": [cid for cid, _, _ in sel_test],
        "selected_train_cases": [
            {"case_id": cid, "stratum_id": strata.get(cid, -1)}
            for cid, _, _ in sel_train
        ],
        "selected_test_cases": [
            {"case_id": cid, "stratum_id": strata.get(cid, -1)}
            for cid, _, _ in sel_test
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    train_root = Path(args.train_root)
    task_dir = Path(args.nnunet_raw_data_base) / "nnUNet_raw_data" / args.task_name

    images_tr = task_dir / "imagesTr"
    labels_tr = task_dir / "labelsTr"
    images_ts = task_dir / "imagesTs"
    labels_ts = task_dir / "labelsTs"   # non-standard: GT labels for held-out test evaluation

    for d in (images_tr, labels_tr, images_ts, labels_ts):
        d.mkdir(parents=True, exist_ok=True)

    if args.clean_output:
        for d in (images_tr, labels_tr, images_ts, labels_ts):
            _clear_dir(d)

    # ── Collect all labeled cases ─────────────────────────────────────────
    print(f"Scanning {train_root} …", flush=True)
    all_cases = _collect_labeled_cases(train_root, args.image_keyword, args.label_keyword)
    if not all_cases:
        raise RuntimeError("No labeled cases found. Check --train-root and --image/label-keyword.")
    print(f"Found {len(all_cases)} labeled cases total.", flush=True)

    # ── Compute volume strata ─────────────────────────────────────────────
    print("Computing GTV volume strata …", flush=True)
    strata = _assign_strata(all_cases, args.stratify_volume_bins)

    # ── Select training cases ─────────────────────────────────────────────
    if args.train_subset_strategy == "stratified_label_volume":
        train_cases = _select_stratified(
            all_cases, args.train_case_limit, strata, args.subset_seed, args.stratify_volume_bins
        )
    else:
        train_cases = _select_random(all_cases, args.train_case_limit, args.subset_seed)
    print(f"Selected {len(train_cases)} training cases "
          f"(strategy={args.train_subset_strategy}).", flush=True)

    # ── Select held-out test cases from remaining labeled cases ───────────
    train_ids = {cid for cid, _, _ in train_cases}
    remaining = [c for c in all_cases if c[0] not in train_ids]
    # Use a different seed offset so test selection is independent of train selection
    test_cases = _select_stratified(
        remaining, args.holdout_test_limit, strata, args.subset_seed + 99991, args.stratify_volume_bins
    )
    print(f"Selected {len(test_cases)} held-out test cases "
          f"(from remaining {len(remaining)} labeled cases).", flush=True)

    # Sanity: no overlap
    test_ids = {cid for cid, _, _ in test_cases}
    overlap = train_ids & test_ids
    if overlap:
        raise RuntimeError(f"Train/test overlap detected: {sorted(overlap)}")

    # ── Copy training files ───────────────────────────────────────────────
    print("Copying training files …", flush=True)
    for idx, (cid, img, lbl) in enumerate(train_cases, 1):
        _copy_or_link(img, images_tr / f"{cid}_0000.nii.gz", args.copy_mode)
        _copy_or_link(lbl, labels_tr / f"{cid}.nii.gz", args.copy_mode)
        _progress(idx, len(train_cases), "train", cid)

    # ── Copy held-out test files ──────────────────────────────────────────
    print("Copying held-out test files …", flush=True)
    for idx, (cid, img, lbl) in enumerate(test_cases, 1):
        _copy_or_link(img, images_ts / f"{cid}_0000.nii.gz", args.copy_mode)
        _copy_or_link(lbl, labels_ts / f"{cid}.nii.gz", args.copy_mode)   # GT for evaluation
        _progress(idx, len(test_cases), "test", cid)

    # ── Write metadata ────────────────────────────────────────────────────
    _write_dataset_json(task_dir / "dataset.json", args.task_name, train_cases, test_cases)
    _write_command_sheet(task_dir, args.task_id, args.task_name)
    _write_manifest(task_dir / "subset_manifest.json", all_cases, train_cases, test_cases, strata, args)

    print(f"\nTask directory : {task_dir.resolve()}")
    print(f"Training cases : {len(train_cases)}  → imagesTr / labelsTr")
    print(f"Held-out test  : {len(test_cases)}   → imagesTs / labelsTs")
    print(f"dataset.json   : {task_dir / 'dataset.json'}")
    print(f"Manifest       : {task_dir / 'subset_manifest.json'}")


if __name__ == "__main__":
    main()
