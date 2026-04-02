from __future__ import annotations

import random
from pathlib import Path

from sklearn.model_selection import KFold

from menrt_mednext.data.discovery import CaseRecord
from menrt_mednext.utils.io import read_json, write_json


def make_holdout_split(
    records: list[CaseRecord],
    val_fraction: float,
    seed: int,
) -> tuple[list[CaseRecord], list[CaseRecord]]:
    rng = random.Random(seed)
    items = records.copy()
    rng.shuffle(items)
    val_count = max(1, int(round(len(items) * val_fraction)))
    val = items[:val_count]
    train = items[val_count:]
    return train, val


def build_kfold_indices(num_items: int, n_splits: int, seed: int) -> list[tuple[list[int], list[int]]]:
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
) -> tuple[list[CaseRecord], list[CaseRecord]]:
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2 for k-fold cross-validation.")
    if fold_index < 0 or fold_index >= n_splits:
        raise ValueError(f"fold_index must be in [0, {n_splits - 1}], got {fold_index}.")

    folds = build_kfold_indices(len(records), n_splits=n_splits, seed=seed)
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
