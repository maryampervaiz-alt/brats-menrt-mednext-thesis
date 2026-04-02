from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run k-fold MedNeXt training")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--run-prefix", type=str, default="mednext_cv")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--start-fold", type=int, default=0)
    p.add_argument("--end-fold", type=int, default=-1)
    p.add_argument("--lr", type=float, default=-1.0)
    p.add_argument("--patch-size", type=int, nargs=3, default=None, metavar=("X", "Y", "Z"))
    p.add_argument("--kaggle", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--strict-split", action="store_true")
    p.add_argument(
        "--auto-evaluate",
        action="store_true",
        help="Run evaluation after each fold and export per-case CSV + summary JSON.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    end_fold = args.end_fold if args.end_fold >= 0 else args.n_folds - 1

    for fold in range(args.start_fold, end_fold + 1):
        run_name = f"{args.run_prefix}_fold{fold}"
        split_path = Path("outputs") / "splits" / f"kfold_{args.n_folds}_fold{fold}.json"

        cmd = [
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
            str(fold),
            "--save-split-json",
            str(split_path),
        ]
        if args.kaggle:
            cmd.append("--kaggle")
        if args.debug:
            cmd.append("--debug")
        if args.lr > 0:
            cmd.extend(["--lr", str(args.lr)])
        if args.patch_size is not None:
            cmd.extend(["--patch-size", str(args.patch_size[0]), str(args.patch_size[1]), str(args.patch_size[2])])
        if args.strict_split:
            cmd.append("--strict-split")

        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

        if args.auto_evaluate:
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
            print("Running:", " ".join(eval_cmd))
            subprocess.run(eval_cmd, check=True)


if __name__ == "__main__":
    main()
