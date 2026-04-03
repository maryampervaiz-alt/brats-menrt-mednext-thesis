from __future__ import annotations

import random
import re
from pathlib import Path

from sklearn.model_selection import KFold

from menrt_mednext.data.discovery import CaseRecord
from menrt_mednext.utils.io import read_json, write_json


def _group_id(case_id: str, group_pattern: str) -> str:
    if not group_pattern:
        return case_id
    match = re.match(group_pattern, case_id)
    if match and match.groups():
        return match.group(1)
    return case_id


def _group_ids(records: list[CaseRecord], group_pattern: str) -> list[str]:
    return [_group_id(r.case_id, group_pattern) for r in records]


def make_holdout_split(
    records: list[CaseRecord],
    val_fraction: float,
    seed: int,
    group_pattern: str = "",
) -> tuple[list[CaseRecord], list[CaseRecord]]:
    if group_pattern:
        groups = _group_ids(records, group_pattern)
        unique_groups = sorted(set(groups))
        if len(unique_groups) < 2:
            raise ValueError("Grouped holdout split requires at least 2 unique groups.")
        rng = random.Random(seed)
        rng.shuffle(unique_groups)
        val_group_count = int(round(len(unique_groups) * val_fraction))
        val_group_count = min(len(unique_groups) - 1, max(1, val_group_count))
        val_groups = set(unique_groups[:val_group_count])
        val = [rec for rec, grp in zip(records, groups) if grp in val_groups]
        train = [rec for rec, grp in zip(records, groups) if grp not in val_groups]
        if not train or not val:
            raise RuntimeError("Grouped holdout split produced an empty train or val partition.")
        return train, val

    rng = random.Random(seed)
    items = records.copy()
    rng.shuffle(items)
    val_count = max(1, int(round(len(items) * val_fraction)))
    val = items[:val_count]
    train = items[val_count:]
    return train, val


def build_kfold_indices(
    num_items: int,
    n_splits: int,
    seed: int,
    group_ids: list[str] | None = None,
) -> list[tuple[list[int], list[int]]]:
    if group_ids is not None:
        if len(group_ids) != num_items:
            raise ValueError("group_ids length must match num_items.")
        unique_groups = sorted(set(group_ids))
        if len(unique_groups) < n_splits:
            raise ValueError("Number of unique groups must be >= n_splits for grouped k-fold.")
        rng = random.Random(seed)
        rng.shuffle(unique_groups)
        group_folds = KFold(n_splits=n_splits, shuffle=False)
        folds = []
        for tr_g_idx, va_g_idx in group_folds.split(unique_groups):
            tr_groups = {unique_groups[i] for i in tr_g_idx.tolist()}
            va_groups = {unique_groups[i] for i in va_g_idx.tolist()}
            tr_idx = [i for i, grp in enumerate(group_ids) if grp in tr_groups]
            va_idx = [i for i, grp in enumerate(group_ids) if grp in va_groups]
            folds.append((tr_idx, va_idx))
        return folds

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_idx = list(range(num_items))
    folds = []
    for tr_idx, va_idx in kf.split(all_idx):
        folds.append((tr_idx.tolist(), va_idx.tolist()))
    return folds


def make_kfold_split(
    records: list[CaseRecord],
    n_splits: int,
    fold_index: int,
    seed: int,
    group_pattern: str = "",
) -> tuple[list[CaseRecord], list[CaseRecord]]:
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2 for k-fold cross-validation.")
    if fold_index < 0 or fold_index >= n_splits:
        raise ValueError(f"fold_index must be in [0, {n_splits - 1}], got {fold_index}.")

    group_ids = _group_ids(records, group_pattern) if group_pattern else None
    folds = build_kfold_indices(len(records), n_splits=n_splits, seed=seed, group_ids=group_ids)
    tr_idx, va_idx = folds[fold_index]
    train = [records[i] for i in tr_idx]
    val = [records[i] for i in va_idx]
    return train, val


def save_split_json(path: str | Path, train_ids: list[str], val_ids: list[str]) -> None:
    write_json(path, {"train_case_ids": train_ids, "val_case_ids": val_ids})


def load_split_json(path: str | Path) -> tuple[set[str], set[str]]:
    payload = read_json(path)
    train_ids = set(payload["train_case_ids"])
    val_ids = set(payload["val_case_ids"])
    return train_ids, val_ids
