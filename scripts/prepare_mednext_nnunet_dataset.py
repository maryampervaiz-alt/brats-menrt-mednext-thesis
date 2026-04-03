from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


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


def _collect_labeled_cases(root: Path, image_kw: str, label_kw: str) -> list[tuple[str, Path, Path]]:
    images = sorted(root.rglob(f"*{image_kw}"))
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
        labels = sorted(case_dir.glob(f"*{label_kw}"))
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
    images = sorted(root.rglob(f"*{image_kw}"))
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
                "image": f"./imagesTr/{case_id}_0000.nii.gz",
                "label": f"./labelsTr/{case_id}.nii.gz",
            }
            for case_id, _, _ in training_cases
        ],
        "test": [f"./imagesTs/{case_id}_0000.nii.gz" for case_id, _ in test_cases],
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

    training_cases = _collect_labeled_cases(train_root, args.image_keyword, args.label_keyword)
    if not training_cases:
        raise RuntimeError("No labeled cases found. Check train root and keywords.")

    unique_case_ids = {case_id for case_id, _, _ in training_cases}
    for case_id, img, lbl in training_cases:
        _copy_or_link(img, images_tr / f"{case_id}_0000.nii.gz", args.copy_mode)
        _copy_or_link(lbl, labels_tr / f"{case_id}.nii.gz", args.copy_mode)

    test_cases: list[tuple[str, Path]] = []
    if val_root is not None and val_root.exists():
        test_cases = _collect_unlabeled_cases(val_root, args.image_keyword)
        overlap = unique_case_ids.intersection({case_id for case_id, _ in test_cases})
        if overlap:
            raise RuntimeError(f"Detected overlap between train and val case IDs: {sorted(list(overlap))[:10]}")
        for case_id, img in test_cases:
            _copy_or_link(img, images_ts / f"{case_id}_0000.nii.gz", args.copy_mode)

    _write_dataset_json(task_dir / "dataset.json", args.task_name, training_cases, test_cases)
    _write_command_sheet(
        task_dir,
        args.task_id,
        args.task_name,
        trainer_name=args.trainer_name,
        plans_identifier=args.plans_identifier,
    )

    print(f"Prepared official nnU-Net(v1) task at: {task_dir.resolve()}")
    print(f"Training cases: {len(training_cases)}")
    print(f"Test/val image-only cases: {len(test_cases)}")
    print(f"dataset.json: {task_dir / 'dataset.json'}")


if __name__ == "__main__":
    main()
