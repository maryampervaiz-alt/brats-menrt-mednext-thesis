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
from menrt_mednext.data.splits import build_kfold_indices, make_holdout_split
from menrt_mednext.utils.io import ensure_dir


def parse_args():
    p = argparse.ArgumentParser(description="Generate split files for MEN-RT")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--mode", type=str, default="holdout", choices=["holdout", "kfold"])
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--out-dir", type=str, default="splits")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config.from_yaml(args.config).raw
    out_dir = ensure_dir(args.out_dir)

    records = discover_cases(
        root_dir=cfg["data"]["root_dir"],
        image_keywords=tuple(cfg["data"]["image_keywords"]),
        label_keywords=tuple(cfg["data"]["label_keywords"]),
        require_labels=True,
    )
    group_pattern = str(cfg["data"].get("group_pattern", "")).strip()
    if args.mode == "holdout":
        train, val = make_holdout_split(
            records,
            val_fraction=float(cfg["data"]["val_fraction"]),
            seed=int(cfg["seed"]),
            group_pattern=group_pattern,
        )
        payload = {
            "train_case_ids": [x.case_id for x in train],
            "val_case_ids": [x.case_id for x in val],
        }
        out_path = out_dir / "holdout_split.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved: {out_path}")
        return

    group_ids = None
    if group_pattern:
        import re

        def _grp(case_id: str) -> str:
            m = re.match(group_pattern, case_id)
            if m and m.groups():
                return m.group(1)
            return case_id

        group_ids = [_grp(x.case_id) for x in records]
    folds = build_kfold_indices(len(records), n_splits=args.k, seed=int(cfg["seed"]), group_ids=group_ids)
    all_ids = [x.case_id for x in records]
    for i, (tr_idx, va_idx) in enumerate(folds):
        payload = {
            "train_case_ids": [all_ids[j] for j in tr_idx],
            "val_case_ids": [all_ids[j] for j in va_idx],
        }
        out_path = out_dir / f"kfold_{args.k}_fold_{i}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
