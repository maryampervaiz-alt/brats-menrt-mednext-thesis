from __future__ import annotations

import numpy as np
from scipy import ndimage


def remove_small_components_3d(mask: np.ndarray, min_size_voxels: int) -> np.ndarray:
    """Remove connected components smaller than min_size_voxels from a binary 3D mask."""
    if min_size_voxels <= 0:
        return (mask > 0).astype(np.uint8)

    binary = (mask > 0).astype(np.uint8)
    labeled, num = ndimage.label(binary)
    if num == 0:
        return binary

    counts = np.bincount(labeled.ravel())
    keep = counts >= int(min_size_voxels)
    keep[0] = False  # background
    filtered = keep[labeled]
    return filtered.astype(np.uint8)

