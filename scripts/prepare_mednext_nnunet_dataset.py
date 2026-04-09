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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare MEN-RT in official nnU-Net(v1) raw task format for MedNeXt.")
    p.add_argument("--train-root", type=str, required=True, help="Root containing labeled train cases")
    p.add_argument("--val-root", type=str, default="", help="Optional root containing unlabeled validation/test cases")
    p.add_argument("--nnunet-raw-data-base", type=str, required=True)
    p.add_argument("--task-id", type=int, required=True)
    p.add_argument("--task-name", type=str, required=True, help="Example: Task502_BraTSMENRT")
    p.add_argument("--image-keyword", type=str, default="_t1c.nii.gz")
    p.add_argument("--label-keyword", type=str, default="_gtv.nii.gz")
    p.add_argument("--copy-mode", type=str, default="copy", choices=["copy", "hardlink"])
    p.add_argument("--trainer-name", type=str, default="nnUNetTrainerV2_MedNeXt_S_kernel3")
    p.add_argument("--plans-identifier", type=str, default="nnUNetPlansv2.1_trgSp_1x1x1")
    p.add_argument("--train-case-limit", type=int, default=0, help="Use only the first deterministic subset of train cases")
    p.add_argument("--val-case-limit", type=int, default=0, help="Use only the first deterministic subset of val/test cases")
    p.add_argument("--subset-seed", type=int, default=42, help="Deterministic seed for subset selection")
    p.add_argument(
        "--train-subset-strategy",
        type=str,
        default="stratified_label_volume",
        choices=["random", "stratified_label_volume"],
        help="Train subset selection strategy.",
    )
    p.add_argument(
        "--val-subset-strategy",
        type=str,
        default="random",
        choices=["random"],
        help="Val/test subset selection strategy.",
    )
    p.add_argument("--stratify-volume-bins", type=int, default=5, help="Number of quantile bins for label-volume stratification")
    p.add_argument("--clean-output", action="store_true")
    return p.parse_args()


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
    # nnU-Net expects .nii.gz. If the source dataset ships .nii files,
    # convert them to valid gzipped NIfTI on the fly instead of merely renaming
    # the extension, which would be incorrect and also waste more disk space.
    if src.suffix.lower() == ".nii" and dst.name.lower().endswith(".nii.gz"):
        try:
            with src.open("rb") as fsrc, gzip.open(dst, "wb", compresslevel=6) as fdst:
                shutil.copyfileobj(fsrc, fdst, length=1024 * 1024)
        except Exception:
            if dst.exists():
                dst.unlink()
            raise
        return
    shutil.copy2(src, dst)


def _progress_every(index: int, total: int, kind: str, case_id: str) -> None:
    if index == 1 or index == total or index % 25 == 0:
        print(f"[{kind}] processed {index}/{total}: {case_id}", flush=True)


def _clear_dir(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return
    for item in path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def _find_files_case_insensitive(root: Path, pattern: str) -> list[Path]:
    pattern_lower = pattern.lower().strip()
    files = [p for p in root.rglob("*") if p.is_file()]
    if not pattern_lower:
        return sorted(files)

    exact_suffix_matches = sorted([p for p in files if p.name.lower().endswith(pattern_lower)])
    if exact_suffix_matches:
        return exact_suffix_matches

    # Fallback to substring matching so small naming differences are surfaced early
    # instead of silently reporting zero discoverable cases.
    token = pattern_lower.replace(".nii.gz", "").replace(".nii", "").strip("_- ")
    if token:
        substring_matches = sorted([p for p in files if token in p.name.lower()])
        if substring_matches:
            return substring_matches

    return []


def _match_files_in_dir(case_dir: Path, pattern: str) -> list[Path]:
    pattern_lower = pattern.lower().strip()
    files = [p for p in case_dir.iterdir() if p.is_file()]
    if not pattern_lower:
        return sorted(files)

    exact_suffix_matches = sorted([p for p in files if p.name.lower().endswith(pattern_lower)])
    if exact_suffix_matches:
        return exact_suffix_matches

    token = pattern_lower.replace(".nii.gz", "").replace(".nii", "").strip("_- ")
    if token:
        substring_matches = sorted([p for p in files if token in p.name.lower()])
        if substring_matches:
            return substring_matches

    return []


def _collect_labeled_cases(root: Path, image_kw: str, label_kw: str) -> list[tuple[str, Path, Path]]:
    images = _find_files_case_insensitive(root, image_kw)
    if not images:
        return []

    cases = []
    seen_case_dirs = set()
    case_id_to_dirs: dict[str, set[Path]] = {}
    for img in images:
        case_dir = img.parent
        case_id = case_dir.name
        case_id_to_dirs.setdefault(case_id, set()).add(case_dir)
        if case_dir in seen_case_dirs:
            continue
        labels = _match_files_in_dir(case_dir, label_kw)
        if not labels:
            continue
        if len(labels) != 1:
            raise RuntimeError(
                f"Expected exactly one label file matching '*{label_kw}' in {case_dir}, found {len(labels)}: "
                f"{[str(x.name) for x in labels]}"
            )
        seen_case_dirs.add(case_dir)
        cases.append((case_id, img, labels[0]))

    duplicate_ids = {k: sorted(str(x) for x in v) for k, v in case_id_to_dirs.items() if len(v) > 1}
    if duplicate_ids:
        raise RuntimeError(
            "Duplicate case folder names found under train_root. "
            f"Conflicts: {duplicate_ids}"
        )
    return sorted(cases, key=lambda x: x[0])


def _collect_unlabeled_cases(root: Path, image_kw: str) -> list[tuple[str, Path]]:
    images = _find_files_case_insensitive(root, image_kw)
    if not images:
        return []

    cases = []
    seen_case_dirs = set()
    case_id_to_dirs: dict[str, set[Path]] = {}
    for img in images:
        case_dir = img.parent
        case_id = case_dir.name
        case_id_to_dirs.setdefault(case_id, set()).add(case_dir)
        if case_dir in seen_case_dirs:
            continue
        seen_case_dirs.add(case_dir)
        cases.append((case_id, img))

    duplicate_ids = {k: sorted(str(x) for x in v) for k, v in case_id_to_dirs.items() if len(v) > 1}
    if duplicate_ids:
        raise RuntimeError(
            "Duplicate case folder names found under val_root. "
            f"Conflicts: {duplicate_ids}"
        )
    return sorted(cases, key=lambda x: x[0])


def _write_dataset_json(path: Path, task_name: str, training_cases: list[tuple[str, Path, Path]], test_cases: list[tuple[str, Path]]) -> None:
    payload = {
        "name": task_name,
        "description": "BraTS MEN-RT T1c to GTV segmentation",
        "tensorImageSize": "3D",
        "reference": "BraTS 2024 MEN-RT",
        "licence": "See dataset terms",
        "release": "1.0",
        "modality": {"0": "T1c"},
        "labels": {"0": "background", "1": "GTV"},
        "numTraining": int(len(training_cases)),
        "numTest": int(len(test_cases)),
        "training": [
            {
                "image": f"./imagesTr/{case_id}.nii.gz",
                "label": f"./labelsTr/{case_id}.nii.gz",
            }
            for case_id, _, _ in training_cases
        ],
        "test": [f"./imagesTs/{case_id}.nii.gz" for case_id, _ in test_cases],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_command_sheet(task_dir: Path, task_id: int, task_name: str, trainer_name: str, plans_identifier: str) -> None:
    cmd_txt = task_dir / "MEDNEXT_COMMANDS.txt"
    lines = [
        "# Set environment variables",
        "export nnUNet_raw_data_base='<ABS_PATH_TO_nnUNet_raw_data_base>'",
        "export nnUNet_preprocessed='<ABS_PATH_TO_nnUNet_preprocessed>'",
        "export RESULTS_FOLDER='<ABS_PATH_TO_nnUNet_results>'",
        "",
        "# Plan and preprocess",
        f"mednextv1_plan_and_preprocess -t {task_id} -pl3d ExperimentPlanner3D_v21_customTargetSpacing_1x1x1 -pl2d None",
        "",
        "# Train one fold",
        f"mednextv1_train 3d_fullres {trainer_name} {task_name} 0 -p {plans_identifier}",
    ]
    cmd_txt.write_text("\n".join(lines), encoding="utf-8")


def _select_subset(cases: list[T], limit: int, seed: int, seed_offset: int = 0) -> list[T]:
    if limit <= 0 or limit >= len(cases):
        return list(cases)
    rng = random.Random(seed + seed_offset)
    selected_indices = sorted(rng.sample(range(len(cases)), k=limit))
    return [cases[i] for i in selected_indices]


def _foreground_voxel_count(label_path: Path) -> int:
    img = sitk.ReadImage(str(label_path))
    arr = sitk.GetArrayViewFromImage(img)
    return int(np.count_nonzero(arr))


def _assign_volume_strata(
    training_cases: list[tuple[str, Path, Path]],
    num_bins: int,
) -> dict[str, dict[str, int]]:
    if num_bins <= 1 or len(training_cases) <= 1:
        strata = {}
        total = len(training_cases)
        for idx, (case_id, _, lbl) in enumerate(training_cases, start=1):
            strata[case_id] = {"foreground_voxels": _foreground_voxel_count(lbl), "stratum_id": 0}
            _progress_every(idx, total, "strata", case_id)
        return strata

    scored_cases = []
    total = len(training_cases)
    for idx, (case_id, _, lbl) in enumerate(training_cases, start=1):
        scored_cases.append((case_id, _foreground_voxel_count(lbl)))
        _progress_every(idx, total, "strata", case_id)

    scored_cases.sort(key=lambda x: (x[1], x[0]))
    strata: dict[str, dict[str, int]] = {}
    total = len(scored_cases)
    for idx, (case_id, voxels) in enumerate(scored_cases):
        stratum_id = min(num_bins - 1, int(idx * num_bins / total))
        strata[case_id] = {"foreground_voxels": int(voxels), "stratum_id": int(stratum_id)}
    return strata


def _select_subset_stratified(
    training_cases: list[tuple[str, Path, Path]],
    limit: int,
    seed: int,
    strata: dict[str, dict[str, int]],
) -> list[tuple[str, Path, Path]]:
    if limit <= 0 or limit >= len(training_cases):
        return list(training_cases)

    bins: dict[int, list[tuple[str, Path, Path]]] = {}
    for case in training_cases:
        case_id = case[0]
        stratum_id = int(strata[case_id]["stratum_id"])
        bins.setdefault(stratum_id, []).append(case)

    rng = random.Random(seed)
    for stratum_id in bins:
        bins[stratum_id] = sorted(bins[stratum_id], key=lambda x: x[0])
        rng.shuffle(bins[stratum_id])

    total_cases = len(training_cases)
    allocations: dict[int, int] = {}
    remainders: list[tuple[float, int]] = []
    for stratum_id, cases_in_bin in bins.items():
        exact = limit * len(cases_in_bin) / total_cases
        alloc = min(len(cases_in_bin), int(exact))
        allocations[stratum_id] = alloc
        remainders.append((exact - alloc, stratum_id))

    remaining = limit - sum(allocations.values())
    for _, stratum_id in sorted(remainders, key=lambda x: (-x[0], x[1])):
        if remaining <= 0:
            break
        if allocations[stratum_id] < len(bins[stratum_id]):
            allocations[stratum_id] += 1
            remaining -= 1

    selected: list[tuple[str, Path, Path]] = []
    for stratum_id in sorted(bins):
        selected.extend(bins[stratum_id][: allocations[stratum_id]])
    return sorted(selected, key=lambda x: x[0])


def _write_subset_manifest(
    path: Path,
    train_cases_all: list[tuple[str, Path, Path]],
    train_cases_selected: list[tuple[str, Path, Path]],
    test_cases_all: list[tuple[str, Path]],
    test_cases_selected: list[tuple[str, Path]],
    train_case_limit: int,
    val_case_limit: int,
    subset_seed: int,
    train_subset_strategy: str,
    val_subset_strategy: str,
    stratify_volume_bins: int,
    train_case_metadata: dict[str, dict[str, int]],
) -> None:
    payload = {
        "subset_seed": int(subset_seed),
        "train_case_limit": int(train_case_limit),
        "val_case_limit": int(val_case_limit),
        "train_subset_strategy": str(train_subset_strategy),
        "val_subset_strategy": str(val_subset_strategy),
        "stratify_volume_bins": int(stratify_volume_bins),
        "full_train_case_count": int(len(train_cases_all)),
        "selected_train_case_count": int(len(train_cases_selected)),
        "full_val_case_count": int(len(test_cases_all)),
        "selected_val_case_count": int(len(test_cases_selected)),
        "selected_train_case_ids": [case_id for case_id, _, _ in train_cases_selected],
        "selected_val_case_ids": [case_id for case_id, _ in test_cases_selected],
        "selected_train_cases": [
            {
                "case_id": case_id,
                "foreground_voxels": int(train_case_metadata.get(case_id, {}).get("foreground_voxels", -1)),
                "stratum_id": int(train_case_metadata.get(case_id, {}).get("stratum_id", -1)),
            }
            for case_id, _, _ in train_cases_selected
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    train_root = Path(args.train_root)
    val_root = Path(args.val_root) if args.val_root else None
    base = Path(args.nnunet_raw_data_base)
    task_dir = base / "nnUNet_raw_data" / args.task_name

    images_tr = task_dir / "imagesTr"
    labels_tr = task_dir / "labelsTr"
    images_ts = task_dir / "imagesTs"
    for d in (images_tr, labels_tr, images_ts):
        d.mkdir(parents=True, exist_ok=True)
    if args.clean_output:
        _clear_dir(images_tr)
        _clear_dir(labels_tr)
        _clear_dir(images_ts)

    training_cases_all = _collect_labeled_cases(train_root, args.image_keyword, args.label_keyword)
    if not training_cases_all:
        raise RuntimeError("No labeled cases found. Check train root and keywords.")
    print(f"Discovered labeled train cases: {len(training_cases_all)}", flush=True)
    print(
        f"Computing train strata using strategy={args.train_subset_strategy}, "
        f"bins={args.stratify_volume_bins}, seed={args.subset_seed}",
        flush=True,
    )
    train_case_metadata = _assign_volume_strata(training_cases_all, args.stratify_volume_bins)
    if args.train_subset_strategy == "stratified_label_volume":
        training_cases = _select_subset_stratified(
            training_cases_all,
            args.train_case_limit,
            args.subset_seed,
            train_case_metadata,
        )
    else:
        training_cases = _select_subset(training_cases_all, args.train_case_limit, args.subset_seed, seed_offset=0)
    if len(training_cases) != len(training_cases_all):
        print(
            f"Using train subset: {len(training_cases)}/{len(training_cases_all)} cases "
            f"(strategy={args.train_subset_strategy}, seed={args.subset_seed})",
            flush=True,
        )
    else:
        print(
            f"Using full train set: {len(training_cases)} cases "
            f"(strategy={args.train_subset_strategy})",
            flush=True,
        )

    unique_case_ids = {case_id for case_id, _, _ in training_cases}
    total_train = len(training_cases)
    for idx, (case_id, img, lbl) in enumerate(training_cases, start=1):
        _copy_or_link(img, images_tr / f"{case_id}_0000.nii.gz", args.copy_mode)
        _copy_or_link(lbl, labels_tr / f"{case_id}.nii.gz", args.copy_mode)
        _progress_every(idx, total_train, "train", case_id)

    test_cases_all: list[tuple[str, Path]] = []
    test_cases: list[tuple[str, Path]] = []
    if val_root is not None and val_root.exists():
        test_cases_all = _collect_unlabeled_cases(val_root, args.image_keyword)
        print(f"Discovered image-only val/test cases: {len(test_cases_all)}", flush=True)
        test_cases = _select_subset(test_cases_all, args.val_case_limit, args.subset_seed, seed_offset=100003)
        if len(test_cases) != len(test_cases_all):
            print(
                f"Using val/test subset: {len(test_cases)}/{len(test_cases_all)} cases "
                f"(strategy={args.val_subset_strategy}, seed={args.subset_seed})",
                flush=True,
            )
        else:
            print(
                f"Using full val/test set: {len(test_cases)} cases "
                f"(strategy={args.val_subset_strategy})",
                flush=True,
            )
        overlap = unique_case_ids.intersection({case_id for case_id, _ in test_cases})
        if overlap:
            raise RuntimeError(f"Detected overlap between train and val case IDs: {sorted(list(overlap))[:10]}")
        total_val = len(test_cases)
        for idx, (case_id, img) in enumerate(test_cases, start=1):
            _copy_or_link(img, images_ts / f"{case_id}_0000.nii.gz", args.copy_mode)
            _progress_every(idx, total_val, "val", case_id)

    _write_dataset_json(task_dir / "dataset.json", args.task_name, training_cases, test_cases)
    _write_command_sheet(
        task_dir,
        args.task_id,
        args.task_name,
        trainer_name=args.trainer_name,
        plans_identifier=args.plans_identifier,
    )
    _write_subset_manifest(
        task_dir / "subset_manifest.json",
        train_cases_all=training_cases_all,
        train_cases_selected=training_cases,
        test_cases_all=test_cases_all,
        test_cases_selected=test_cases,
        train_case_limit=args.train_case_limit,
        val_case_limit=args.val_case_limit,
        subset_seed=args.subset_seed,
        train_subset_strategy=args.train_subset_strategy,
        val_subset_strategy=args.val_subset_strategy,
        stratify_volume_bins=args.stratify_volume_bins,
        train_case_metadata=train_case_metadata,
    )

    print(f"Prepared official nnU-Net(v1) task at: {task_dir.resolve()}")
    print(f"Training cases: {len(training_cases)}")
    print(f"Test/val image-only cases: {len(test_cases)}")
    print(f"dataset.json: {task_dir / 'dataset.json'}")


if __name__ == "__main__":
    main()
