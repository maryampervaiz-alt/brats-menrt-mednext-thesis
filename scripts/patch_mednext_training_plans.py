from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Patch nnU-Net(v1) training plans for Kaggle-safe MedNeXt training.")
    p.add_argument("--config", type=str, default="configs/mednext_nnunet.yaml")
    p.add_argument("--stage", type=int, default=0)
    return p.parse_args()


def _load_cfg(path: str) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return cfg["mednext_nnunet"]


def _preprocessed_task_dir(cfg: dict) -> Path:
    return Path(cfg["nnunet_preprocessed"]) / str(cfg["task_name"])


def _artifacts_dir(cfg: dict) -> Path:
    return Path(cfg["results_folder"]) / "menrt_repo_artifacts"


def _plans_file(preprocessed_task_dir: Path, plans_identifier: str) -> Path:
    exact = preprocessed_task_dir / f"{plans_identifier}_plans_3D.pkl"
    if exact.exists():
        return exact
    candidates = sorted(preprocessed_task_dir.glob("*_plans_3D.pkl"))
    if not candidates:
        raise FileNotFoundError(f"No 3D plans file found under {preprocessed_task_dir}")
    return candidates[0]


def _get_stage_cfg(plans: dict, stage: int) -> dict:
    plans_per_stage = plans.get("plans_per_stage")
    if isinstance(plans_per_stage, dict):
        if stage in plans_per_stage:
            return plans_per_stage[stage]
        if str(stage) in plans_per_stage:
            return plans_per_stage[str(stage)]
    if isinstance(plans_per_stage, list):
        return plans_per_stage[stage]
    raise KeyError("Could not access plans_per_stage for requested stage.")


def _as_int_list(values) -> list[int]:
    return [int(v) for v in values]


def main() -> None:
    args = parse_args()
    cfg = _load_cfg(args.config)
    patch_size = [int(x) for x in cfg.get("train_patch_size_override", [])]
    batch_size = int(cfg.get("train_batch_size_override", 0) or 0)

    if len(patch_size) != 3:
        raise ValueError("Config must define train_patch_size_override as three integers.")
    if batch_size <= 0:
        raise ValueError("Config must define a positive train_batch_size_override.")

    pre_dir = _preprocessed_task_dir(cfg)
    plans_path = _plans_file(pre_dir, str(cfg["plans_identifier"]))
    backup_path = plans_path.with_suffix(plans_path.suffix + ".bak")
    artifacts_dir = _artifacts_dir(cfg)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    with plans_path.open("rb") as f:
        plans = pickle.load(f)

    stage_cfg = _get_stage_cfg(plans, args.stage)
    old_patch = _as_int_list(np.asarray(stage_cfg["patch_size"]).astype(int))
    old_batch = int(stage_cfg["batch_size"])
    pools = _as_int_list(stage_cfg["num_pool_per_axis"])

    required_divisibility = _as_int_list(2 ** int(n) for n in pools)
    for axis, (size, div) in enumerate(zip(patch_size, required_divisibility)):
        if size % div != 0:
            raise ValueError(
                f"Requested patch size {patch_size} is incompatible with axis {axis} "
                f"pool depth {pools[axis]} (must be divisible by {div})."
            )

    if not backup_path.exists():
        shutil.copy2(plans_path, backup_path)

    stage_cfg["patch_size"] = np.array(patch_size, dtype=np.int64)
    stage_cfg["batch_size"] = int(batch_size)

    with plans_path.open("wb") as f:
        pickle.dump(plans, f, protocol=pickle.HIGHEST_PROTOCOL)

    summary = {
        "plans_path": str(plans_path),
        "backup_path": str(backup_path),
        "stage": int(args.stage),
        "old_patch_size": old_patch,
        "new_patch_size": patch_size,
        "old_batch_size": old_batch,
        "new_batch_size": batch_size,
        "num_pool_per_axis": pools,
        "required_divisibility": required_divisibility,
    }
    summary_path = artifacts_dir / "patched_training_plans_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Patched plans file: {plans_path}")
    print(f"Backup plans file: {backup_path}")
    print(f"Old patch size: {old_patch} -> New patch size: {patch_size}")
    print(f"Old batch size: {old_batch} -> New batch size: {batch_size}")
    print(f"Saved patch summary: {summary_path}")


if __name__ == "__main__":
    main()
