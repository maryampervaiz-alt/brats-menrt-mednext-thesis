from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


NIFTI_EXTS = (".nii.gz", ".nii")


@dataclass
class CaseRecord:
    case_id: str
    image: str
    label: str | None = None


def _strip_ext(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    if name.endswith(".nii"):
        return name[: -len(".nii")]
    return name


def _build_case_key(path: Path, keywords: Iterable[str]) -> str:
    stem = _strip_ext(path.name).lower()
    for kw in keywords:
        stem = re.sub(rf"[\W_]*{re.escape(kw)}[\w]*", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"[_\-.]+$", "", stem)
    stem = re.sub(r"[_\-.]{2,}", "_", stem)
    return stem


def discover_cases(
    root_dir: str | Path,
    image_keywords: tuple[str, ...] = ("t1c",),
    label_keywords: tuple[str, ...] = ("gtv", "seg"),
    require_labels: bool = True,
) -> list[CaseRecord]:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    nii_files = [
        p
        for p in root.rglob("*")
        if p.is_file() and any(str(p).lower().endswith(ext) for ext in NIFTI_EXTS)
    ]

    image_files = [
        p for p in nii_files if any(kw.lower() in p.name.lower() for kw in image_keywords)
    ]
    label_files = [
        p for p in nii_files if any(kw.lower() in p.name.lower() for kw in label_keywords)
    ]

    image_map: dict[str, Path] = {}
    image_collisions: dict[str, list[Path]] = {}
    for fp in image_files:
        key = _build_case_key(fp, image_keywords)
        if key in image_map:
            image_collisions.setdefault(key, [image_map[key]]).append(fp)
            continue
        image_map[key] = fp

    label_map: dict[str, Path] = {}
    label_collisions: dict[str, list[Path]] = {}
    for fp in label_files:
        key = _build_case_key(fp, label_keywords)
        if key in label_map:
            label_collisions.setdefault(key, [label_map[key]]).append(fp)
            continue
        label_map[key] = fp

    if image_collisions:
        details = "; ".join(
            f"{k}: {[str(p) for p in paths]}"
            for k, paths in sorted(image_collisions.items())
        )
        raise RuntimeError(
            "Duplicate image case keys detected after normalization. "
            f"Resolve naming collisions before training. {details}"
        )
    if label_collisions:
        details = "; ".join(
            f"{k}: {[str(p) for p in paths]}"
            for k, paths in sorted(label_collisions.items())
        )
        raise RuntimeError(
            "Duplicate label case keys detected after normalization. "
            f"Resolve naming collisions before training. {details}"
        )

    records: list[CaseRecord] = []
    missing_labels = 0
    for key, img_path in image_map.items():
        lbl = label_map.get(key)
        if require_labels and lbl is None:
            missing_labels += 1
            continue
        records.append(CaseRecord(case_id=key, image=str(img_path), label=str(lbl) if lbl else None))

    if require_labels and not records:
        raise RuntimeError(
            "No paired image-label cases were found. "
            "Please verify naming patterns and keywords in config."
        )
    if not require_labels and not records:
        raise RuntimeError("No image cases were found.")

    records.sort(key=lambda x: x.case_id)
    return records
