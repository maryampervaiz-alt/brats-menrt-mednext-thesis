from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare BraTS MEN-RT in nnUNetv2 raw dataset format")
    p.add_argument("--train-root", type=str, required=True, help="Root containing labeled train cases")
    p.add_argument("--val-root", type=str, default="", help="Optional root containing unlabeled validation cases")
    p.add_argument("--nnunet-raw", type=str, default="./nnUNet_raw")
    p.add_argument("--dataset-id", type=int, default=501)
    p.add_argument("--dataset-name", type=str, default="BraTSMENRT")
    p.add_argument("--image-keyword", type=str, default="_t1c.nii.gz")
    p.add_argument("--label-keyword", type=str, default="_gtv.nii.gz")
    p.add_argument("--copy-mode", type=str, default="copy", choices=["copy", "hardlink"])
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


def _collect_labeled_cases(root: Path, image_kw: str, label_kw: str) -> list[tuple[str, Path, Path]]:
    cases = []
    for case_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        imgs = sorted(case_dir.glob(f"*{image_kw}"))
        lbls = sorted(case_dir.glob(f"*{label_kw}"))
        if not imgs or not lbls:
            continue
        case_id = case_dir.name
        cases.append((case_id, imgs[0], lbls[0]))
    return cases


def _collect_unlabeled_cases(root: Path, image_kw: str) -> list[tuple[str, Path]]:
    cases = []
    for case_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        imgs = sorted(case_dir.glob(f"*{image_kw}"))
        if not imgs:
            continue
        case_id = case_dir.name
        cases.append((case_id, imgs[0]))
    return cases


def _write_dataset_json(path: Path, num_training: int, dataset_name: str) -> None:
    payload = {
        "name": dataset_name,
        "description": "BraTS MEN-RT T1c to GTV segmentation",
        "tensorImageSize": "3D",
        "reference": "BraTS 2024 MEN-RT",
        "licence": "See dataset terms",
        "release": "1.0",
        "channel_names": {"0": "T1c"},
        "labels": {"background": 0, "GTV": 1},
        "numTraining": int(num_training),
        "file_ending": ".nii.gz",
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_command_sheet(dataset_id: int, dataset_folder: Path) -> None:
    cmd_txt = dataset_folder / "NNUNET_COMMANDS.txt"
    lines = [
        "# 1) Set environment variables",
        "export nnUNet_raw='<ABS_PATH_TO_nnUNet_raw>'",
        "export nnUNet_preprocessed='<ABS_PATH_TO_nnUNet_preprocessed>'",
        "export nnUNet_results='<ABS_PATH_TO_nnUNet_results>'",
        "",
        "# 2) Plan and preprocess",
        f"nnUNetv2_plan_and_preprocess -d {dataset_id} --verify_dataset_integrity",
        "",
        "# 3) Train (5 folds)",
        f"nnUNetv2_train {dataset_id} 3d_fullres 0",
        f"nnUNetv2_train {dataset_id} 3d_fullres 1",
        f"nnUNetv2_train {dataset_id} 3d_fullres 2",
        f"nnUNetv2_train {dataset_id} 3d_fullres 3",
        f"nnUNetv2_train {dataset_id} 3d_fullres 4",
        "",
        "# If MedNeXt nnUNet trainer classes are installed, use trainer override (-tr), e.g.:",
        f"nnUNetv2_train {dataset_id} 3d_fullres 0 -tr nnUNetTrainerMedNeXt",
        "",
        "# 4) Find best config",
        f"nnUNetv2_find_best_configuration {dataset_id} -c 3d_fullres",
    ]
    cmd_txt.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    train_root = Path(args.train_root)
    val_root = Path(args.val_root) if args.val_root else None
    nnunet_raw = Path(args.nnunet_raw)

    dataset_dir = nnunet_raw / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_ts = dataset_dir / "imagesTs"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)
    images_ts.mkdir(parents=True, exist_ok=True)

    labeled = _collect_labeled_cases(train_root, args.image_keyword, args.label_keyword)
    if not labeled:
        raise RuntimeError("No labeled cases found. Check train root and keywords.")

    for case_id, img, lbl in labeled:
        _copy_or_link(img, images_tr / f"{case_id}_0000.nii.gz", args.copy_mode)
        _copy_or_link(lbl, labels_tr / f"{case_id}.nii.gz", args.copy_mode)

    unlabeled_count = 0
    if val_root is not None and val_root.exists():
        unlabeled = _collect_unlabeled_cases(val_root, args.image_keyword)
        for case_id, img in unlabeled:
            _copy_or_link(img, images_ts / f"{case_id}_0000.nii.gz", args.copy_mode)
        unlabeled_count = len(unlabeled)

    _write_dataset_json(dataset_dir / "dataset.json", num_training=len(labeled), dataset_name=args.dataset_name)
    _write_command_sheet(args.dataset_id, dataset_dir)

    print(f"Prepared nnUNet dataset at: {dataset_dir.resolve()}")
    print(f"Training cases: {len(labeled)}")
    print(f"Test/val image-only cases: {unlabeled_count}")
    print(f"dataset.json: {dataset_dir / 'dataset.json'}")
    print(f"commands: {dataset_dir / 'NNUNET_COMMANDS.txt'}")


if __name__ == "__main__":
    main()
