from __future__ import annotations

import inspect

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

    # MedNeXt package signatures can differ across commits/releases.
    # Try common keyword variants first, then fall back to positional call.
    attempts = [
        {
            "num_channels": in_channels,
            "num_classes": num_classes,
            "model_id": model_id,
            "kernel_size": kernel_size,
            "deep_supervision": deep_supervision,
        },
        {
            "num_input_channels": in_channels,
            "num_classes": num_classes,
            "model_id": model_id,
            "kernel_size": kernel_size,
            "deep_supervision": deep_supervision,
        },
        {
            "in_channels": in_channels,
            "num_classes": num_classes,
            "model_id": model_id,
            "kernel_size": kernel_size,
            "deep_supervision": deep_supervision,
        },
        {
            "in_channels": in_channels,
            "out_channels": num_classes,
            "model_id": model_id,
            "kernel_size": kernel_size,
            "deep_supervision": deep_supervision,
        },
    ]

    model = None
    last_err: Exception | None = None
    for kwargs in attempts:
        try:
            model = create_mednext_v1(**kwargs)
            break
        except TypeError as e:
            last_err = e

    if model is None:
        # Final fallback: positional for channel/class args.
        try:
            model = create_mednext_v1(
                in_channels,
                num_classes,
                model_id=model_id,
                kernel_size=kernel_size,
                deep_supervision=deep_supervision,
            )
        except TypeError as e:
            sig = None
            try:
                sig = str(inspect.signature(create_mednext_v1))
            except Exception:
                sig = "<signature unavailable>"
            raise TypeError(
                "Could not call create_mednext_v1 with supported argument mappings. "
                f"Detected signature: {sig}. Last error: {last_err or e}"
            ) from e

    # Enable checkpointed blocks only if the underlying architecture exposes it.
    if checkpoint_style is not None and hasattr(model, "checkpoint_style"):
        setattr(model, "checkpoint_style", checkpoint_style)
    return model
