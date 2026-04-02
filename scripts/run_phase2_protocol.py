from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Phase-2 protocol: LR sweep + CV summary + thesis tables")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--run-prefix", type=str, default="menrt_phase2")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--kaggle", action="store_true")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def _load_lr_values(config_path: str) -> list[float]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    values = cfg.get("sweep", {}).get("lr_values", [cfg["training"]["lr"]])
    return [float(v) for v in values]


def _run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    lr_values = _load_lr_values(args.config)

    for lr in lr_values:
        lr_tag = str(lr).replace(".", "p")
        prefix = f"{args.run_prefix}_lr{lr_tag}"

        train_cmd = [
            sys.executable,
            "scripts/train_cv.py",
            "--config",
            args.config,
            "--run-prefix",
            prefix,
            "--n-folds",
            str(args.n_folds),
            "--lr",
            str(lr),
            "--patch-size",
            "128",
            "160",
            "112",
            "--strict-split",
            "--auto-evaluate",
        ]
        if args.kaggle:
            train_cmd.append("--kaggle")
        if args.debug:
            train_cmd.append("--debug")
        _run(train_cmd)

        summary_cmd = [
            sys.executable,
            "scripts/summarize_cv_results.py",
            "--run-prefix",
            prefix,
            "--n-folds",
            str(args.n_folds),
            "--out-dir",
            "outputs/reports",
        ]
        _run(summary_cmd)

        fold_csv = Path("outputs/reports") / f"{prefix}_fold_metrics.csv"
        summary_csv = Path("outputs/reports") / f"{prefix}_summary_mean_std.csv"

        table_cmd = [
            sys.executable,
            "scripts/generate_thesis_tables.py",
            "--summary-csv",
            str(summary_csv),
            "--fold-csv",
            str(fold_csv),
            "--out-dir",
            f"outputs/reports/{prefix}",
        ]
        _run(table_cmd)

        leakage_cmd = [
            sys.executable,
            "scripts/check_split_leakage.py",
            "--split-json",
        ]
        for fold in range(args.n_folds):
            leakage_cmd.append(str(Path("outputs") / "splits" / f"kfold_{args.n_folds}_fold{fold}.json"))
        leakage_cmd.append("--check-cross-fold")
        leakage_cmd.append("--fail-on-overlap")
        _run(leakage_cmd)

    repro_cmd = [
        sys.executable,
        "scripts/generate_repro_report.py",
        "--config",
        args.config,
        "--out-json",
        f"outputs/reports/{args.run_prefix}_repro_report.json",
    ]
    _run(repro_cmd)


if __name__ == "__main__":
    main()
