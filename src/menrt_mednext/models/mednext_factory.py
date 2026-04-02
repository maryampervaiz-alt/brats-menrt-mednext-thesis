from __future__ import annotations

import torch.nn as nn


def build_mednext(
    in_channels: int,
    num_classes: int,
    model_id: str = "B",
    kernel_size: int = 3,
    deep_supervision: bool = False,
    checkpoint_style: bool | None = None,
) -> nn.Module:
    try:
        from nnunet_mednext import create_mednext_v1
    except Exception as e:
        raise ImportError(
            "MedNeXt package not found. Install with: "
            "`pip install git+https://github.com/MIC-DKFZ/MedNeXt.git`"
        ) from e

    model = create_mednext_v1(
        num_channels=in_channels,
        num_classes=num_classes,
        model_id=model_id,
        kernel_size=kernel_size,
        deep_supervision=deep_supervision,
    )

    # Enable checkpointed blocks only if the underlying architecture exposes it.
    if checkpoint_style is not None and hasattr(model, "checkpoint_style"):
        setattr(model, "checkpoint_style", checkpoint_style)
    return model
