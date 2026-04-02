from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from menrt_mednext.config import Config
from menrt_mednext.data.discovery import discover_cases
from menrt_mednext.data.splits import make_holdout_split, save_split_json
from menrt_mednext.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create train/val splits for BraTS MEN-RT")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--out-dir", type=str, default="outputs/splits")
    p.add_argument("--kaggle", action="store_true")
    return p.parse_args()


def _resolve_data_root(cfg: dict, use_kaggle: bool) -> Path:
    if not use_kaggle:
        return Path(cfg["data"]["root_dir"])

    root = Path(cfg["data"]["kaggle_root_dir"])
    explicit_subdir = str(cfg["data"].get("kaggle_train_subdir", "")).strip()
    if explicit_subdir:
        explicit_path = Path(explicit_subdir)
        if not explicit_path.is_absolute():
            explicit_path = root / explicit_subdir
        if explicit_path.exists():
            return explicit_path
        raise FileNotFoundError(
            f"Configured data.kaggle_train_subdir not found: {explicit_path}"
        )

    if (root / "BraTS-MEN-RT-Train-v2").exists() and (root / "BraTS-MEN-RT-Val-v1").exists():
        raise ValueError(
            "Both train and val folders detected under kaggle_root_dir. "
            "Set data.kaggle_train_subdir explicitly to avoid accidental mixing."
        )

    for c in [root / "BraTS-MEN-RT-Train-v2", root / "BraTS2024-MEN-RT-TrainingData", root / "BraTS-MEN_RT", root]:
        if c.exists():
            return c
    return root


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config).raw

    data_root = _resolve_data_root(cfg, use_kaggle=args.kaggle)
    out_dir = ensure_dir(args.out_dir)

    records = discover_cases(
        root_dir=data_root,
        image_keywords=tuple(cfg["data"]["image_keywords"]),
        label_keywords=tuple(cfg["data"]["label_keywords"]),
        require_labels=True,
    )
    train_records, val_records = make_holdout_split(
        records,
        val_fraction=float(cfg["data"]["val_fraction"]),
        seed=int(cfg["seed"]),
    )

    split_path = out_dir / "holdout_split.json"
    save_split_json(
        split_path,
        train_ids=[r.case_id for r in train_records],
        val_ids=[r.case_id for r in val_records],
    )

    (out_dir / "dataset_index.json").write_text(
        json.dumps(
            [
                {"case_id": r.case_id, "image": r.image, "label": r.label}
                for r in records
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Data root: {data_root}")
    print(f"Saved split: {split_path}")
    print(f"Train={len(train_records)}, Val={len(val_records)}, Total={len(records)}")


if __name__ == "__main__":
    main()
