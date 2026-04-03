from __future__ import annotations

import argparse
import importlib
import inspect
import json
import pkgutil
import shutil
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate official MedNeXt nnU-Net(v1) MEN-RT setup before training.")
    p.add_argument("--config", type=str, default="configs/mednext_nnunet.yaml")
    p.add_argument("--out", type=str, default="", help="Optional JSON report output path.")
    p.add_argument("--check-trainer", action="store_true", help="Also verify that the configured base trainer is importable.")
    return p.parse_args()


def _load_cfg(path: str) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return cfg["mednext_nnunet"]


def _find_case_dirs(root: Path, image_keyword: str, label_keyword: str | None) -> tuple[list[Path], dict[str, list[str]], list[str]]:
    image_files = sorted(root.rglob(f"*{image_keyword}"))
    case_dirs = []
    duplicate_case_ids: dict[str, list[str]] = {}
    missing_labels: list[str] = []
    case_id_to_dirs: dict[str, set[Path]] = {}

    seen_case_dirs: set[Path] = set()
    for image_file in image_files:
        case_dir = image_file.parent
        case_id = case_dir.name
        case_id_to_dirs.setdefault(case_id, set()).add(case_dir)
        if case_dir in seen_case_dirs:
            continue
        seen_case_dirs.add(case_dir)
        if label_keyword is not None:
            labels = sorted(case_dir.glob(f"*{label_keyword}"))
            if len(labels) == 0:
                missing_labels.append(case_id)
                continue
            if len(labels) != 1:
                raise RuntimeError(
                    f"Expected exactly one label file matching '*{label_keyword}' in {case_dir}, found {len(labels)}."
                )
        case_dirs.append(case_dir)

    for case_id, dirs in case_id_to_dirs.items():
        if len(dirs) > 1:
            duplicate_case_ids[case_id] = sorted(str(d) for d in dirs)

    return sorted(case_dirs), duplicate_case_ids, sorted(missing_labels)


def _locate_trainer(base_trainer: str) -> dict:
    root_mod = importlib.import_module("nnunet_mednext")
    if hasattr(root_mod, base_trainer):
        return {
            "found": True,
            "module": root_mod.__name__,
            "file": str(Path(inspect.getfile(root_mod)).resolve()),
        }

    mod_path = getattr(root_mod, "__path__", None)
    if mod_path is None:
        raise RuntimeError("nnunet_mednext package has no __path__; cannot scan trainers.")

    for mod_info in pkgutil.walk_packages(mod_path, prefix=f"{root_mod.__name__}."):
        try:
            mod = importlib.import_module(mod_info.name)
        except Exception:
            continue
        if hasattr(mod, base_trainer):
            return {
                "found": True,
                "module": mod.__name__,
                "file": str(Path(inspect.getfile(mod)).resolve()),
            }

    return {"found": False, "module": "", "file": ""}


def main() -> None:
    args = parse_args()
    cfg = _load_cfg(args.config)

    train_root = Path(cfg["train_root"])
    val_root = Path(cfg["val_root"]) if str(cfg.get("val_root", "")).strip() else None

    report: dict[str, object] = {
        "config_path": str(Path(args.config).resolve()),
        "task_name": str(cfg["task_name"]),
        "train_root_exists": train_root.exists(),
        "val_root_exists": bool(val_root.exists()) if val_root is not None else False,
        "required_commands": {
            "mednextv1_plan_and_preprocess": shutil.which("mednextv1_plan_and_preprocess") is not None,
            "mednextv1_train": shutil.which("mednextv1_train") is not None,
            "nnUNet_predict": shutil.which("nnUNet_predict") is not None,
        },
    }

    if train_root.exists():
        train_cases, train_duplicates, missing_labels = _find_case_dirs(
            train_root,
            image_keyword=str(cfg["image_keyword"]),
            label_keyword=str(cfg["label_keyword"]),
        )
        report["train_case_count"] = len(train_cases)
        report["train_duplicate_case_ids"] = train_duplicates
        report["train_missing_labels"] = missing_labels
    else:
        report["train_case_count"] = 0
        report["train_duplicate_case_ids"] = {}
        report["train_missing_labels"] = []

    if val_root is not None and val_root.exists():
        val_cases, val_duplicates, _ = _find_case_dirs(
            val_root,
            image_keyword=str(cfg["image_keyword"]),
            label_keyword=None,
        )
        report["val_case_count"] = len(val_cases)
        report["val_duplicate_case_ids"] = val_duplicates
        train_case_ids = {p.name for p in train_cases} if train_root.exists() else set()
        val_case_ids = {p.name for p in val_cases}
        report["train_val_overlap_case_ids"] = sorted(train_case_ids.intersection(val_case_ids))
    else:
        report["val_case_count"] = 0
        report["val_duplicate_case_ids"] = {}
        report["train_val_overlap_case_ids"] = []

    if args.check_trainer:
        try:
            report["base_trainer_import"] = _locate_trainer(str(cfg["base_trainer"]))
        except Exception as exc:
            report["base_trainer_import"] = {"found": False, "error": str(exc)}

    text = json.dumps(report, indent=2)
    print(text)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
        print(f"Saved setup report: {out_path}")


if __name__ == "__main__":
    main()
