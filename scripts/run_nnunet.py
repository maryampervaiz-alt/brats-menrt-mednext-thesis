"""Main orchestrator for the nnUNet baseline pipeline.

Usage
-----
python scripts/run_nnunet.py --config configs/nnunet_baseline.yaml --mode all
python scripts/run_nnunet.py --config configs/nnunet_baseline.yaml --mode train --fold 0 --max-epochs 50
python scripts/run_nnunet.py --config configs/nnunet_baseline.yaml --mode predict-testset
python scripts/run_nnunet.py --config configs/nnunet_baseline.yaml --mode evaluate-testset

Modes
-----
all              : prepare → preprocess → make-splits → install-trainer → train all folds
prepare          : copy subset into nnUNet raw task format
preprocess       : nnUNet_plan_and_preprocess
make-splits      : create stratified 5-fold splits_final.pkl
install-trainer  : install nnUNetTrainerV2_MENRT into the nnunet package
train            : train one or all folds
predict-testset  : ensemble inference on held-out test set (raw + postprocessed)
evaluate-testset : compute metrics on held-out test set predictions
"""
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


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run nnUNet baseline pipeline for BraTS MEN-RT.")
    p.add_argument("--config", type=str, default="configs/nnunet_baseline.yaml")
    p.add_argument(
        "--mode", type=str, default="all",
        choices=["all", "prepare", "preprocess", "make-splits",
                 "install-trainer", "train", "predict-testset", "evaluate-testset"],
    )
    p.add_argument("--fold", type=int, default=-1,
                   help="Single fold override (default: run all folds from config)")
    p.add_argument("--max-epochs", type=int, default=-1,
                   help="Override max_epochs from config")
    p.add_argument("--continue-training", action="store_true",
                   help="Resume training from the latest checkpoint (-c flag for nnUNet_train)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing")
    return p.parse_args()


# ── Config ────────────────────────────────────────────────────────────────────

def _load_cfg(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))["nnunet_baseline"]


def _artifacts_dir(cfg: dict) -> Path:
    return Path(cfg["results_folder"]) / "menrt_artifacts"


def _trainer_root(cfg: dict) -> Path:
    return (
        Path(cfg["results_folder"])
        / "nnUNet"
        / str(cfg["network"])
        / str(cfg["task_name"])
        / f'{cfg["trainer_name"]}__{cfg["plans_identifier"]}'
    )


def _images_ts_dir(cfg: dict) -> Path:
    return (
        Path(cfg["nnunet_raw_data_base"])
        / "nnUNet_raw_data"
        / str(cfg["task_name"])
        / "imagesTs"
    )


def _labels_ts_dir(cfg: dict) -> Path:
    return (
        Path(cfg["nnunet_raw_data_base"])
        / "nnUNet_raw_data"
        / str(cfg["task_name"])
        / "labelsTs"
    )


# ── Environment ───────────────────────────────────────────────────────────────

def _build_env(cfg: dict, max_epochs_override: int = -1) -> dict[str, str]:
    env = os.environ.copy()
    env["nnUNet_raw_data_base"] = str(cfg["nnunet_raw_data_base"])
    env["nnUNet_preprocessed"]  = str(cfg["nnunet_preprocessed"])
    env["RESULTS_FOLDER"]       = str(cfg["results_folder"])
    epochs = max_epochs_override if max_epochs_override > 0 else int(cfg["max_epochs"])
    env[str(cfg["trainer_epochs_env"])] = str(epochs)
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    if bool(cfg.get("torch_force_no_weights_only_load", False)):
        env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    return env


# ── Command runner ────────────────────────────────────────────────────────────

def _require(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required command not found in PATH: {name}")


def _log(log_file: Path | None, cmd: list[str], env: dict[str, str]) -> None:
    if log_file is None:
        return
    log_file.parent.mkdir(parents=True, exist_ok=True)
    tracked = {k: env.get(k, "") for k in
               ["nnUNet_raw_data_base", "nnUNet_preprocessed", "RESULTS_FOLDER",
                "NNUNET_MAX_EPOCHS", "PYTORCH_ALLOC_CONF"]}
    with log_file.open("a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat(timespec='seconds')}]\n")
        f.write(f"CMD: {' '.join(cmd)}\n")
        for k, v in tracked.items():
            f.write(f"{k}={v}\n")
        f.write("\n")


def _run(cmd: list[str], env: dict[str, str], dry_run: bool,
         log_file: Path | None = None) -> None:
    print("CMD:", " ".join(cmd), flush=True)
    _log(log_file, cmd, env)
    if not dry_run:
        subprocess.run(cmd, check=True, env=env)


# ── Snapshots ─────────────────────────────────────────────────────────────────

def _write_snapshots(cfg: dict, config_path: str, artifacts_dir: Path,
                     env: dict[str, str]) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    (artifacts_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(raw, sort_keys=False), encoding="utf-8"
    )

    runtime = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cwd": str(Path.cwd().resolve()),
        "env": {k: env.get(k, "") for k in
                ["nnUNet_raw_data_base", "nnUNet_preprocessed", "RESULTS_FOLDER",
                 "NNUNET_MAX_EPOCHS"]},
    }
    (artifacts_dir / "runtime_snapshot.json").write_text(
        json.dumps(runtime, indent=2) + "\n", encoding="utf-8"
    )

    try:
        r = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                           check=True, capture_output=True, text=True, env=env)
        (artifacts_dir / "pip_freeze.txt").write_text(r.stdout, encoding="utf-8")
    except Exception as exc:
        (artifacts_dir / "pip_freeze_error.txt").write_text(str(exc), encoding="utf-8")

    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"],
                           check=True, capture_output=True, text=True, env=env)
        (artifacts_dir / "git_head.txt").write_text(r.stdout, encoding="utf-8")
    except Exception as exc:
        (artifacts_dir / "git_head_error.txt").write_text(str(exc), encoding="utf-8")


# ── Pipeline stages ───────────────────────────────────────────────────────────

def _prepare(cfg: dict, env: dict, dry_run: bool, log: Path | None) -> None:
    cmd = [
        sys.executable, "scripts/prepare_nnunet_dataset.py",
        "--train-root",          str(cfg["train_root"]),
        "--nnunet-raw-data-base", str(cfg["nnunet_raw_data_base"]),
        "--task-id",             str(cfg["task_id"]),
        "--task-name",           str(cfg["task_name"]),
        "--image-keyword",       str(cfg.get("image_keyword", "_t1c.nii.gz")),
        "--label-keyword",       str(cfg.get("label_keyword", "_gtv.nii.gz")),
        "--copy-mode",           str(cfg.get("copy_mode", "copy")),
        "--train-case-limit",    str(int(cfg.get("train_case_limit", 50) or 50)),
        "--holdout-test-limit",  str(int(cfg.get("holdout_test_limit", 10) or 10)),
        "--subset-seed",         str(int(cfg.get("subset_seed", 42) or 42)),
        "--train-subset-strategy", str(cfg.get("train_subset_strategy", "stratified_label_volume")),
        "--stratify-volume-bins", str(int(cfg.get("stratify_volume_bins", 5) or 5)),
    ]
    if bool(cfg.get("clean_task_output", True)):
        cmd.append("--clean-output")
    _run(cmd, env, dry_run, log)


def _preprocess(cfg: dict, env: dict, dry_run: bool, log: Path | None) -> None:
    if not dry_run:
        _require("nnUNet_plan_and_preprocess")
    cmd = ["nnUNet_plan_and_preprocess", "-t", str(cfg["task_id"]),
           "--verify_dataset_integrity"]
    _run(cmd, env, dry_run, log)


def _make_splits(cfg: dict, config_path: str, env: dict,
                 dry_run: bool, log: Path | None) -> None:
    cmd = [sys.executable, "scripts/create_nnunet_splits.py",
           "--config", str(config_path)]
    _run(cmd, env, dry_run, log)


def _install_trainer(cfg: dict, env: dict, dry_run: bool, log: Path | None) -> None:
    cmd = [
        sys.executable, "scripts/install_nnunet_trainer.py",
        "--base-trainer",    str(cfg["base_trainer"]),
        "--new-trainer",     str(cfg["trainer_name"]),
        "--epochs-env",      str(cfg["trainer_epochs_env"]),
        "--default-epochs",  str(cfg["max_epochs"]),
    ]
    _run(cmd, env, dry_run, log)


def _train(cfg: dict, env: dict, dry_run: bool, fold_override: int = -1,
           continue_training: bool = False, log: Path | None = None) -> None:
    if not dry_run:
        _require("nnUNet_train")
    folds = [fold_override] if fold_override >= 0 else list(cfg.get("folds", [0, 1, 2, 3, 4]))
    for fold in folds:
        cmd = [
            "nnUNet_train",
            str(cfg["network"]),
            str(cfg["trainer_name"]),
            str(cfg["task_name"]),
            str(fold),
        ]
        if continue_training or bool(cfg.get("continue_training", False)):
            cmd.append("--continue")
        _run(cmd, env, dry_run, log)


def _predict_testset(cfg: dict, env: dict, dry_run: bool,
                     fold_override: int = -1, log: Path | None = None) -> None:
    """Ensemble inference on held-out test set — raw and postprocessed."""
    if not dry_run:
        _require("nnUNet_predict")

    images_ts  = str(_images_ts_dir(cfg))
    out_raw    = str(cfg.get("predict_output_raw",     "/kaggle/working/nnunet_predictions/raw"))
    out_postproc = str(cfg.get("predict_output_postproc", "/kaggle/working/nnunet_predictions/postprocessed"))
    folds = [fold_override] if fold_override >= 0 else list(cfg.get("folds", [0, 1, 2, 3, 4]))
    fold_strs = [str(f) for f in folds]

    # ── Postprocessed predictions (nnUNet default) ───────────────────────
    cmd_pp = [
        "nnUNet_predict",
        "-i", images_ts,
        "-o", out_postproc,
        "-t", str(cfg["task_name"]),
        "-m", str(cfg["network"]),
        "-tr", str(cfg["trainer_name"]),
        "-p", str(cfg["plans_identifier"]),
        "-f", *fold_strs,
    ]
    print("\n── Predicting (postprocessed) ──────────────────────────────", flush=True)
    _run(cmd_pp, env, dry_run, log)

    # ── Raw predictions (no postprocessing) ─────────────────────────────
    cmd_raw = cmd_pp[:]
    cmd_raw[4] = out_raw           # replace -o value
    cmd_raw.append("--disable_postprocessing")
    print("\n── Predicting (raw, no postprocessing) ─────────────────────", flush=True)
    _run(cmd_raw, env, dry_run, log)


def _evaluate_testset(cfg: dict, env: dict, dry_run: bool,
                      log: Path | None = None) -> None:
    cmd = [
        sys.executable, "scripts/evaluate_nnunet_testset.py",
        "--config", "configs/nnunet_baseline.yaml",
    ]
    _run(cmd, env, dry_run, log)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg  = _load_cfg(args.config)
    env  = _build_env(cfg, args.max_epochs)

    artifacts_dir = Path(".dryrun_artifacts") if args.dry_run else _artifacts_dir(cfg)
    log_file = artifacts_dir / "command_history.log"

    _write_snapshots(cfg, args.config, artifacts_dir, env)

    if args.mode in ("all", "prepare") and cfg.get("run_prepare", True):
        _prepare(cfg, env, args.dry_run, log_file)

    if args.mode in ("all", "preprocess") and cfg.get("run_plan_and_preprocess", True):
        _preprocess(cfg, env, args.dry_run, log_file)

    if args.mode in ("all", "make-splits") and cfg.get("run_make_splits", True):
        _make_splits(cfg, args.config, env, args.dry_run, log_file)

    if args.mode in ("all", "install-trainer") and cfg.get("run_install_trainer", True):
        _install_trainer(cfg, env, args.dry_run, log_file)

    if args.mode in ("all", "train") and cfg.get("run_train", True):
        _train(cfg, env, args.dry_run, args.fold, args.continue_training, log_file)

    if args.mode == "predict-testset":
        _predict_testset(cfg, env, args.dry_run, args.fold, log_file)

    if args.mode == "evaluate-testset":
        _evaluate_testset(cfg, env, args.dry_run, log_file)

    if args.mode in ("train", "all") and not args.dry_run:
        print(f"\nResults root: {_trainer_root(cfg)}", flush=True)


if __name__ == "__main__":
    main()
