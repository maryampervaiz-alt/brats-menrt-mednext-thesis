from __future__ import annotations

from typing import Any

from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandShiftIntensityd,
    SpatialPadd,
    Spacingd,
)

from menrt_mednext.data.discovery import CaseRecord


def build_train_transforms(cfg: dict[str, Any]) -> Compose:
    tcfg = cfg["transforms"]
    patch_size = tcfg["patch_size"]
    spacing = tcfg["spacing"]
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS", labels=None),
            Spacingd(
                keys=["image", "label"],
                pixdim=spacing,
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            # Pad only when a case is smaller than the requested training patch.
            SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=patch_size,
                pos=2,
                neg=1,
                num_samples=1,
                image_key="image",
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandAffined(
                keys=["image", "label"],
                prob=0.2,
                rotate_range=(0.1, 0.1, 0.1),
                scale_range=(0.1, 0.1, 0.1),
                mode=("bilinear", "nearest"),
                padding_mode="border",
            ),
            RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.01),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.2),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.2),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def build_val_transforms(cfg: dict[str, Any]) -> Compose:
    tcfg = cfg["transforms"]
    patch_size = tcfg["patch_size"]
    spacing = tcfg["spacing"]
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS", labels=None),
            Spacingd(
                keys=["image", "label"],
                pixdim=spacing,
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            # Preserve full-volume geometry for validation; do not crop larger cases here.
            SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def build_infer_transforms(cfg: dict[str, Any]) -> Compose:
    tcfg = cfg["transforms"]
    patch_size = tcfg["patch_size"]
    spacing = tcfg["spacing"]
    return Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS", labels=None),
            Spacingd(keys=["image"], pixdim=spacing, mode=("bilinear",)),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            # Preserve full-volume geometry for inference; only pad smaller cases.
            SpatialPadd(keys=["image"], spatial_size=patch_size),
            EnsureTyped(keys=["image"]),
        ]
    )


def records_to_items(records: list[CaseRecord], require_label: bool = True) -> list[dict[str, str]]:
    items = []
    for rec in records:
        row = {"case_id": rec.case_id, "image": rec.image}
        if require_label:
            if rec.label is None:
                continue
            row["label"] = rec.label
        items.append(row)
    return items


def make_dataset(
    items: list[dict[str, str]],
    transform: Compose,
    cache_rate: float,
    num_workers: int,
) -> Dataset:
    if cache_rate > 0:
        return CacheDataset(
            data=items,
            transform=transform,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )
    return Dataset(data=items, transform=transform)


def make_loader(
    ds: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def post_process_for_metric() -> Compose:
    return Compose(
        [
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys=["pred", "label"], threshold=0.5),
        ]
    )
