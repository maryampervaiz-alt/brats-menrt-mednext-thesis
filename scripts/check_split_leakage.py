from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check train/val leakage in split JSON files")
    p.add_argument("--split-json", type=str, nargs="+", required=True)
    p.add_argument("--fail-on-overlap", action="store_true")
    p.add_argument(
        "--check-cross-fold",
        action="store_true",
        help="Also check overlap between validation sets across provided split files.",
    )
    return p.parse_args()


def _load_split(path: Path) -> tuple[set[str], set[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return set(payload.get("train_case_ids", [])), set(payload.get("val_case_ids", []))


def main() -> None:
    args = parse_args()
    any_overlap = False
    val_sets: list[tuple[Path, set[str]]] = []

    for split_path in args.split_json:
        p = Path(split_path)
        tr, va = _load_split(p)
        val_sets.append((p, va))
        overlap = tr.intersection(va)
        if overlap:
            any_overlap = True
            print(f"[LEAKAGE] {p}: overlap_count={len(overlap)}")
            print(f"Examples: {sorted(list(overlap))[:10]}")
        else:
            print(f"[OK] {p}: no train/val overlap. train={len(tr)}, val={len(va)}")

    if args.check_cross_fold and len(val_sets) > 1:
        for i in range(len(val_sets)):
            p_i, va_i = val_sets[i]
            for j in range(i + 1, len(val_sets)):
                p_j, va_j = val_sets[j]
                overlap = va_i.intersection(va_j)
                if overlap:
                    any_overlap = True
                    print(f"[LEAKAGE] cross-fold val overlap: {p_i.name} vs {p_j.name}, count={len(overlap)}")
                    print(f"Examples: {sorted(list(overlap))[:10]}")

    if args.fail_on_overlap and any_overlap:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
