from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Archive official MedNeXt nnU-Net(v1) training state for Kaggle-safe resume.")
    p.add_argument("--config", type=str, default="configs/mednext_nnunet.yaml")
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--archive-path", type=str, default="")
    p.add_argument("--include-preprocessed", action="store_true")
    p.add_argument("--include-raw-task", action="store_true")
    return p.parse_args()


def _load_cfg(path: str) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return cfg["mednext_nnunet"]


def _trainer_root(cfg: dict) -> Path:
    return (
        Path(cfg["results_folder"])
        / "nnUNet"
        / str(cfg["network"])
        / str(cfg["task_name"])
        / f'{cfg["trainer_name"]}__{cfg["plans_identifier"]}'
    )


def _artifacts_root(cfg: dict) -> Path:
    return Path(cfg["results_folder"]) / "menrt_repo_artifacts"


def _safe_arcname(path: Path) -> str:
    return path.as_posix()


def main() -> None:
    args = parse_args()
    cfg = _load_cfg(args.config)

    fold_dir = _trainer_root(cfg) / f"fold_{args.fold}"
    if not fold_dir.exists():
        raise FileNotFoundError(f"Fold directory not found: {fold_dir}")

    archive_path = (
        Path(args.archive_path)
        if args.archive_path
        else Path("archives") / f'{cfg["task_name"]}_fold{args.fold}_state.tar.gz'
    )
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    raw_task_dir = Path(cfg["nnunet_raw_data_base"]) / "nnUNet_raw_data" / str(cfg["task_name"])
    preprocessed_task_dir = Path(cfg["nnunet_preprocessed"]) / str(cfg["task_name"])
    artifacts_root = _artifacts_root(cfg)
    repo_config = Path(args.config)
    repo_requirements = Path("requirements.txt")

    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(fold_dir, arcname=_safe_arcname(fold_dir))
        if args.include_preprocessed and preprocessed_task_dir.exists():
            tar.add(preprocessed_task_dir, arcname=_safe_arcname(preprocessed_task_dir))
        if args.include_raw_task and raw_task_dir.exists():
            tar.add(raw_task_dir, arcname=_safe_arcname(raw_task_dir))
        if artifacts_root.exists():
            tar.add(artifacts_root, arcname=_safe_arcname(artifacts_root))
        if repo_config.exists():
            tar.add(repo_config, arcname=_safe_arcname(repo_config))
        if repo_requirements.exists():
            tar.add(repo_requirements, arcname=_safe_arcname(repo_requirements))

    print(f"Saved archive: {archive_path}")


if __name__ == "__main__":
    main()
