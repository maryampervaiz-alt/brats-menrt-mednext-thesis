from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run official MedNeXt nnU-Net(v1) workflow for MEN-RT.")
    p.add_argument("--config", type=str, default="configs/mednext_nnunet.yaml")
    p.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "prepare", "preprocess", "make-splits", "install-trainer", "train", "predict"],
    )
    p.add_argument("--fold", type=int, default=-1, help="Override config folds and run a single fold.")
    p.add_argument("--max-epochs", type=int, default=-1, help="Override configured max epochs.")
    p.add_argument("--continue-training", action="store_true")
    p.add_argument("--predict-input", type=str, default="", help="Override prediction input folder for nnUNet_predict.")
    p.add_argument("--predict-output", type=str, default="", help="Override prediction output folder for nnUNet_predict.")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _artifacts_dir(cfg: dict) -> Path:
    return Path(cfg["results_folder"]) / "menrt_repo_artifacts"


def _append_command_log(log_file: Path | None, cmd: list[str], env: dict[str, str]) -> None:
    if log_file is None:
        return
    log_file.parent.mkdir(parents=True, exist_ok=True)
    tracked_env = {
        "nnUNet_raw_data_base": env.get("nnUNet_raw_data_base", ""),
        "nnUNet_preprocessed": env.get("nnUNet_preprocessed", ""),
        "RESULTS_FOLDER": env.get("RESULTS_FOLDER", ""),
    }
    if "MEDNEXT_MAX_EPOCHS" in env:
        tracked_env["MEDNEXT_MAX_EPOCHS"] = env["MEDNEXT_MAX_EPOCHS"]
    if "MEDNEXT_UNPACK_DATA" in env:
        tracked_env["MEDNEXT_UNPACK_DATA"] = env["MEDNEXT_UNPACK_DATA"]
    with log_file.open("a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat(timespec='seconds')}]\n")
        f.write(f"CMD: {' '.join(cmd)}\n")
        for key, value in tracked_env.items():
            f.write(f"{key}={value}\n")
        f.write("\n")


def _write_config_snapshot(cfg: dict, config_path: str, artifacts_dir: Path) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = artifacts_dir / "mednext_nnunet_config_snapshot.yaml"
    raw_cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    snapshot_path.write_text(yaml.safe_dump(raw_cfg, sort_keys=False), encoding="utf-8")


def _write_runtime_snapshot(env: dict[str, str], artifacts_dir: Path) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    runtime_payload = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cwd": str(Path.cwd().resolve()),
        "tracked_env": {
            "nnUNet_raw_data_base": env.get("nnUNet_raw_data_base", ""),
            "nnUNet_preprocessed": env.get("nnUNet_preprocessed", ""),
            "RESULTS_FOLDER": env.get("RESULTS_FOLDER", ""),
            "MEDNEXT_MAX_EPOCHS": env.get("MEDNEXT_MAX_EPOCHS", ""),
            "MEDNEXT_UNPACK_DATA": env.get("MEDNEXT_UNPACK_DATA", ""),
        },
    }
    (artifacts_dir / "runtime_snapshot.json").write_text(
        json.dumps(runtime_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    try:
        freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        (artifacts_dir / "pip_freeze.txt").write_text(freeze.stdout, encoding="utf-8")
    except Exception as exc:
        (artifacts_dir / "pip_freeze_error.txt").write_text(str(exc) + "\n", encoding="utf-8")

    try:
        git_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        (artifacts_dir / "git_head.txt").write_text(git_head.stdout, encoding="utf-8")
    except Exception as exc:
        (artifacts_dir / "git_head_error.txt").write_text(str(exc) + "\n", encoding="utf-8")


def _run(cmd: list[str], env: dict[str, str], dry_run: bool, log_file: Path | None = None) -> None:
    print("CMD:", " ".join(cmd))
    _append_command_log(log_file, cmd, env)
    if dry_run:
        return
    subprocess.run(cmd, check=True, env=env)


def _require_cmd(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required command not found in PATH: {name}")


def _load_cfg(path: str) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return cfg["mednext_nnunet"]


def _task_results_dir(cfg: dict) -> Path:
    return (
        Path(cfg["results_folder"])
        / "nnUNet"
        / str(cfg["network"])
        / str(cfg["task_name"])
        / f'{cfg["trainer_name"]}__{cfg["plans_identifier"]}'
    )


def _task_images_ts_dir(cfg: dict) -> Path:
    return Path(cfg["nnunet_raw_data_base"]) / "nnUNet_raw_data" / str(cfg["task_name"]) / "imagesTs"


def _build_env(cfg: dict, max_epochs_override: int = -1) -> dict[str, str]:
    env = os.environ.copy()
    env["nnUNet_raw_data_base"] = str(cfg["nnunet_raw_data_base"])
    env["nnUNet_preprocessed"] = str(cfg["nnunet_preprocessed"])
    env["RESULTS_FOLDER"] = str(cfg["results_folder"])
    env[str(cfg["trainer_epochs_env"])] = str(max_epochs_override if max_epochs_override > 0 else cfg["max_epochs"])
    env[str(cfg.get("trainer_unpack_env", "MEDNEXT_UNPACK_DATA"))] = "1" if bool(cfg.get("unpack_preprocessed", False)) else "0"
    return env


def _prepare(cfg: dict, env: dict[str, str], dry_run: bool, log_file: Path | None = None) -> None:
    cmd = [
        sys.executable,
        "scripts/prepare_mednext_nnunet_dataset.py",
        "--train-root",
        str(cfg["train_root"]),
        "--val-root",
        str(cfg.get("val_root", "")),
        "--nnunet-raw-data-base",
        str(cfg["nnunet_raw_data_base"]),
        "--task-id",
        str(cfg["task_id"]),
        "--task-name",
        str(cfg["task_name"]),
        "--image-keyword",
        str(cfg.get("image_keyword", "_t1c.nii.gz")),
        "--label-keyword",
        str(cfg.get("label_keyword", "_gtv.nii.gz")),
        "--copy-mode",
        str(cfg.get("copy_mode", "copy")),
        "--trainer-name",
        str(cfg["trainer_name"]),
        "--plans-identifier",
        str(cfg["plans_identifier"]),
    ]
    train_case_limit = int(cfg.get("train_case_limit", 0) or 0)
    val_case_limit = int(cfg.get("val_case_limit", 0) or 0)
    subset_seed = int(cfg.get("subset_seed", 42) or 42)
    train_subset_strategy = str(cfg.get("train_subset_strategy", "stratified_label_volume"))
    val_subset_strategy = str(cfg.get("val_subset_strategy", "random"))
    stratify_volume_bins = int(cfg.get("stratify_volume_bins", 5) or 5)
    if train_case_limit > 0:
        cmd.extend(["--train-case-limit", str(train_case_limit)])
    if val_case_limit > 0:
        cmd.extend(["--val-case-limit", str(val_case_limit)])
    cmd.extend(["--subset-seed", str(subset_seed)])
    cmd.extend(["--train-subset-strategy", train_subset_strategy])
    cmd.extend(["--val-subset-strategy", val_subset_strategy])
    cmd.extend(["--stratify-volume-bins", str(stratify_volume_bins)])
    if bool(cfg.get("clean_task_output", True)):
        cmd.append("--clean-output")
    _run(cmd, env=env, dry_run=dry_run, log_file=log_file)


def _preprocess(cfg: dict, env: dict[str, str], dry_run: bool, log_file: Path | None = None) -> None:
    if not dry_run:
        _require_cmd("mednextv1_plan_and_preprocess")
    cmd = [
        "mednextv1_plan_and_preprocess",
        "-t",
        str(cfg["task_id"]),
        "-pl3d",
        str(cfg["planner_3d"]),
        "-pl2d",
        str(cfg["planner_2d"]),
    ]
    _run(cmd, env=env, dry_run=dry_run, log_file=log_file)


def _install_trainer(cfg: dict, env: dict[str, str], dry_run: bool, log_file: Path | None = None) -> None:
    cmd = [
        sys.executable,
        "scripts/install_mednext_custom_trainer.py",
        "--base-trainer",
        str(cfg["base_trainer"]),
        "--new-trainer",
        str(cfg["trainer_name"]),
        "--epochs-env",
        str(cfg["trainer_epochs_env"]),
        "--default-epochs",
        str(cfg["max_epochs"]),
        "--unpack-env",
        str(cfg.get("trainer_unpack_env", "MEDNEXT_UNPACK_DATA")),
        "--default-unpack",
        "1" if bool(cfg.get("unpack_preprocessed", False)) else "0",
    ]
    _run(cmd, env=env, dry_run=dry_run, log_file=log_file)


def _make_splits(
    cfg: dict,
    config_path: str,
    env: dict[str, str],
    dry_run: bool,
    log_file: Path | None = None,
) -> None:
    cmd = [
        sys.executable,
        "scripts/create_mednext_stratified_splits.py",
        "--config",
        str(config_path),
    ]
    _run(cmd, env=env, dry_run=dry_run, log_file=log_file)


def _train(
    cfg: dict,
    env: dict[str, str],
    dry_run: bool,
    fold_override: int = -1,
    continue_training: bool = False,
    log_file: Path | None = None,
) -> None:
    if not dry_run:
        _require_cmd("mednextv1_train")
    folds = [fold_override] if fold_override >= 0 else list(cfg.get("folds", [0, 1, 2, 3, 4]))
    for fold in folds:
        cmd = [
            "mednextv1_train",
            str(cfg["network"]),
            str(cfg["trainer_name"]),
            str(cfg["task_name"]),
            str(fold),
            "-p",
            str(cfg["plans_identifier"]),
        ]
        if continue_training or bool(cfg.get("continue_training", False)):
            cmd.append("-c")
        _run(cmd, env=env, dry_run=dry_run, log_file=log_file)


def _predict(
    cfg: dict,
    env: dict[str, str],
    dry_run: bool,
    fold_override: int = -1,
    predict_input: str = "",
    predict_output: str = "",
    log_file: Path | None = None,
) -> None:
    if not dry_run:
        _require_cmd("nnUNet_predict")
    folds = [fold_override] if fold_override >= 0 else list(cfg.get("folds", [0]))
    input_dir = str(predict_input or cfg.get("predict_input", "")).strip()
    output_dir = str(predict_output or cfg.get("predict_output", "")).strip()
    if not input_dir:
        input_dir = str(_task_images_ts_dir(cfg))
    if not output_dir:
        raise ValueError("Prediction output folder is required. Set predict_output in config or pass --predict-output.")
    if not predict_output and fold_override >= 0:
        output_dir = str(Path(output_dir) / f"fold_{fold_override}")

    cmd = [
        "nnUNet_predict",
        "-i",
        input_dir,
        "-o",
        output_dir,
        "-t",
        str(cfg["task_name"]),
        "-m",
        str(cfg["network"]),
        "-tr",
        str(cfg["trainer_name"]),
        "-p",
        str(cfg["plans_identifier"]),
        "-f",
    ]
    cmd.extend([str(f) for f in folds])
    _run(cmd, env=env, dry_run=dry_run, log_file=log_file)


def main() -> None:
    args = parse_args()
    cfg = _load_cfg(args.config)
    env = _build_env(cfg, max_epochs_override=args.max_epochs)
    artifacts_dir = Path(".dryrun_artifacts") if args.dry_run else _artifacts_dir(cfg)
    log_file = artifacts_dir / "command_history.log"
    _write_config_snapshot(cfg, args.config, artifacts_dir)
    _write_runtime_snapshot(env, artifacts_dir)

    if args.mode in ("all", "prepare") and bool(cfg.get("run_prepare", True)):
        _prepare(cfg, env=env, dry_run=args.dry_run, log_file=log_file)
    if args.mode in ("all", "preprocess") and bool(cfg.get("run_plan_and_preprocess", True)):
        _preprocess(cfg, env=env, dry_run=args.dry_run, log_file=log_file)
    if args.mode in ("all", "make-splits") and bool(cfg.get("run_make_splits", True)):
        _make_splits(cfg, args.config, env=env, dry_run=args.dry_run, log_file=log_file)
    if args.mode in ("all", "install-trainer") and bool(cfg.get("run_install_trainer", True)):
        _install_trainer(cfg, env=env, dry_run=args.dry_run, log_file=log_file)
    if args.mode in ("all", "train") and bool(cfg.get("run_train", True)):
        _train(
            cfg,
            env=env,
            dry_run=args.dry_run,
            fold_override=args.fold,
            continue_training=args.continue_training,
            log_file=log_file,
        )
    if args.mode == "predict":
        _predict(
            cfg,
            env=env,
            dry_run=args.dry_run,
            fold_override=args.fold,
            predict_input=args.predict_input,
            predict_output=args.predict_output,
            log_file=log_file,
        )

    if args.mode == "train" and not args.dry_run:
        print(f"Results root: {_task_results_dir(cfg)}")


if __name__ == "__main__":
    main()
