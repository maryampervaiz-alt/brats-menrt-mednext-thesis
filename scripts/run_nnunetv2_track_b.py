from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Track-B: strict nnUNetv2-style MedNeXt protocol")
    p.add_argument("--config", type=str, default="configs/track_b_nnunetv2.yaml")
    p.add_argument("--mode", type=str, default="all", choices=["all", "prepare", "preprocess", "train", "findbest"])
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _run(cmd: list[str], env: dict[str, str], dry_run: bool) -> None:
    print("CMD:", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True, env=env)


def _env_from_cfg(tb: dict) -> dict[str, str]:
    env = os.environ.copy()
    env["nnUNet_raw"] = str(tb["nnunet_raw"])
    env["nnUNet_preprocessed"] = str(tb["nnunet_preprocessed"])
    env["nnUNet_results"] = str(tb["nnunet_results"])
    return env


def _require_cmd(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required command not found in PATH: {name}")


def _validate_track_b_config(tb: dict, mode: str) -> None:
    run_prepare = (mode in ("all", "prepare")) and bool(tb.get("run_prepare", True))
    run_preprocess = (mode in ("all", "preprocess")) and bool(tb.get("run_plan_and_preprocess", True))
    run_train = (mode in ("all", "train")) and bool(tb.get("run_train", True))
    run_findbest = (mode in ("all", "findbest")) and bool(tb.get("run_find_best", True))

    if run_prepare:
        train_root = Path(str(tb.get("train_root", "")))
        if not train_root.exists():
            raise FileNotFoundError(f"track_b.train_root not found: {train_root}")
        val_root = str(tb.get("val_root", "")).strip()
        if val_root and not Path(val_root).exists():
            raise FileNotFoundError(f"track_b.val_root not found: {val_root}")
        for k in ("nnunet_raw", "nnunet_preprocessed", "nnunet_results"):
            v = str(tb.get(k, "")).strip()
            if not v:
                raise ValueError(f"Missing track_b.{k} in config.")

    if run_preprocess:
        _require_cmd("nnUNetv2_plan_and_preprocess")
    if run_train:
        _require_cmd("nnUNetv2_train")
    if run_findbest:
        _require_cmd("nnUNetv2_find_best_configuration")


def _prepare_dataset(tb: dict, env: dict[str, str], dry_run: bool) -> None:
    cmd = [
        sys.executable,
        "scripts/prepare_nnunetv2_dataset.py",
        "--train-root",
        str(tb["train_root"]),
        "--val-root",
        str(tb.get("val_root", "")),
        "--nnunet-raw",
        str(tb["nnunet_raw"]),
        "--dataset-id",
        str(tb["dataset_id"]),
        "--dataset-name",
        str(tb["dataset_name"]),
        "--image-keyword",
        str(tb.get("image_keyword", "_t1c.nii.gz")),
        "--label-keyword",
        str(tb.get("label_keyword", "_gtv.nii.gz")),
        "--copy-mode",
        str(tb.get("copy_mode", "copy")),
    ]
    if bool(tb.get("clean_output", True)):
        cmd.append("--clean-output")
    _run(cmd, env=env, dry_run=dry_run)


def _plan_and_preprocess(tb: dict, env: dict[str, str], dry_run: bool) -> None:
    cmd = [
        "nnUNetv2_plan_and_preprocess",
        "-d",
        str(tb["dataset_id"]),
    ]
    if bool(tb.get("verify_dataset_integrity", True)):
        cmd.append("--verify_dataset_integrity")
    _run(cmd, env=env, dry_run=dry_run)


def _train_folds(tb: dict, env: dict[str, str], dry_run: bool) -> None:
    folds = tb.get("folds", [0, 1, 2, 3, 4])
    trainer_name = str(tb.get("trainer_name", "")).strip()
    if bool(tb.get("require_mednext_trainer", True)) and not trainer_name:
        raise ValueError(
            "Track-B requires an explicit MedNeXt trainer class. "
            "Set track_b.trainer_name (for example nnUNetTrainerMedNeXt) "
            "or disable track_b.require_mednext_trainer."
        )
    for fold in folds:
        cmd = [
            "nnUNetv2_train",
            str(tb["dataset_id"]),
            str(tb.get("configuration", "3d_fullres")),
            str(fold),
        ]
        if trainer_name:
            cmd.extend(["-tr", trainer_name])
        _run(cmd, env=env, dry_run=dry_run)


def _find_best(tb: dict, env: dict[str, str], dry_run: bool) -> None:
    cmd = [
        "nnUNetv2_find_best_configuration",
        str(tb["dataset_id"]),
        "-c",
        str(tb.get("configuration", "3d_fullres")),
    ]
    _run(cmd, env=env, dry_run=dry_run)


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    tb = cfg["track_b"]
    _validate_track_b_config(tb, mode=args.mode)
    env = _env_from_cfg(tb)

    mode = args.mode
    if mode in ("all", "prepare") and bool(tb.get("run_prepare", True)):
        _prepare_dataset(tb, env=env, dry_run=args.dry_run)
    if mode in ("all", "preprocess") and bool(tb.get("run_plan_and_preprocess", True)):
        _plan_and_preprocess(tb, env=env, dry_run=args.dry_run)
    if mode in ("all", "train") and bool(tb.get("run_train", True)):
        _train_folds(tb, env=env, dry_run=args.dry_run)
    if mode in ("all", "findbest") and bool(tb.get("run_find_best", True)):
        _find_best(tb, env=env, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
