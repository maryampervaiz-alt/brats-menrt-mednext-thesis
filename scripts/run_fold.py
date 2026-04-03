from __future__ import annotations

import argparse
import os
import subprocess
import sys
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_LOGGING_MIN_LEVEL", "3")
warnings.filterwarnings("ignore", message=".*cuda.cudart module is deprecated.*", category=FutureWarning)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a single CV fold with a stable run name for resume-safe training."
    )
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--run-prefix", type=str, default="mednext_cv")
    p.add_argument("--fold-index", type=int, required=True)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--lr", type=float, default=-1.0)
    p.add_argument("--epochs", type=int, default=-1)
    p.add_argument("--patch-size", type=int, nargs=3, default=None, metavar=("X", "Y", "Z"))
    p.add_argument("--kaggle", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--strict-split", action="store_true")
    p.add_argument(
        "--auto-evaluate",
        action="store_true",
        help="Run evaluation after training and export per-case CSV + summary JSON.",
    )
    p.add_argument(
        "--archive-after-run",
        action="store_true",
        help="Archive the completed run directory for Kaggle-safe export/resume.",
    )
    p.add_argument("--archive-path", type=str, default="")
    return p.parse_args()


def _run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    run_name = f"{args.run_prefix}_fold{args.fold_index}"
    split_path = Path("outputs") / "splits" / f"kfold_{args.n_folds}_fold{args.fold_index}.json"

    train_cmd = [
        sys.executable,
        "scripts/train.py",
        "--config",
        args.config,
        "--run-name",
        run_name,
        "--use-kfold",
        "--n-folds",
        str(args.n_folds),
        "--fold-index",
        str(args.fold_index),
        "--save-split-json",
        str(split_path),
    ]
    if args.kaggle:
        train_cmd.append("--kaggle")
    if args.debug:
        train_cmd.append("--debug")
    if args.lr > 0:
        train_cmd.extend(["--lr", str(args.lr)])
    if args.epochs > 0:
        train_cmd.extend(["--epochs", str(args.epochs)])
    if args.patch_size is not None:
        train_cmd.extend(["--patch-size", str(args.patch_size[0]), str(args.patch_size[1]), str(args.patch_size[2])])
    if args.strict_split:
        train_cmd.append("--strict-split")

    _run(train_cmd)

    if not args.auto_evaluate:
        pass
    else:
        ckpt = Path("outputs") / run_name / "checkpoints" / "best_model.pt"
        eval_csv = Path("outputs") / run_name / "checkpoints" / "eval_per_case.csv"
        eval_json = Path("outputs") / run_name / "checkpoints" / "eval_summary.json"
        eval_cmd = [
            sys.executable,
            "scripts/evaluate.py",
            "--config",
            args.config,
            "--checkpoint",
            str(ckpt),
            "--split-json",
            str(split_path),
            "--out-csv",
            str(eval_csv),
            "--summary-json",
            str(eval_json),
        ]
        if args.kaggle:
            eval_cmd.append("--kaggle")
        if args.debug:
            eval_cmd.append("--debug")

        _run(eval_cmd)

    if args.archive_after_run:
        archive_cmd = [
            sys.executable,
            "scripts/archive_run.py",
            "--run-name",
            run_name,
            "--split-json",
            str(split_path),
        ]
        if args.archive_path:
            archive_cmd.extend(["--archive-path", args.archive_path])
        _run(archive_cmd)


if __name__ == "__main__":
    main()
