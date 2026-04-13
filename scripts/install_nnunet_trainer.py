"""Install a custom nnUNet v1 trainer into the nnunet package.

Creates  nnUNetTrainerV2_MENRT  — a thin subclass of nnUNetTrainerV2 that
reads max epochs from the  NNUNET_MAX_EPOCHS  environment variable.  This
allows different epoch budgets per fold without modifying installed files.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import pkgutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Install nnUNetTrainerV2_MENRT into the nnunet package."
    )
    p.add_argument("--base-trainer", type=str, default="nnUNetTrainerV2")
    p.add_argument("--new-trainer",  type=str, default="nnUNetTrainerV2_MENRT")
    p.add_argument("--epochs-env",   type=str, default="NNUNET_MAX_EPOCHS")
    p.add_argument("--default-epochs", type=int, default=50)
    return p.parse_args()


def _find_trainer_module(base_trainer: str) -> tuple[str, Path]:
    """Return (module_name, module_file) for the base trainer class."""
    root = importlib.import_module("nnunet")
    root_path = getattr(root, "__path__", None)
    if root_path is None:
        raise RuntimeError("nnunet package has no __path__; cannot scan.")

    for info in pkgutil.walk_packages(root_path, prefix=f"{root.__name__}."):
        try:
            mod = importlib.import_module(info.name)
        except Exception:
            continue
        if hasattr(mod, base_trainer):
            return mod.__name__, Path(inspect.getfile(mod)).resolve()

    raise RuntimeError(
        f"Could not locate class '{base_trainer}' inside the nnunet package.\n"
        "Ensure nnunet==1.7.1 is installed."
    )


def main() -> None:
    args = parse_args()
    mod_name, mod_file = _find_trainer_module(args.base_trainer)
    out_file = mod_file.parent / f"{args.new_trainer}.py"

    code = f'''\
from __future__ import annotations

import os
from {mod_name} import {args.base_trainer}


class {args.new_trainer}({args.base_trainer}):
    """
    Thin wrapper around nnUNetTrainerV2.

    Reads  {args.epochs_env}  at init time so the same trainer name can be
    used across folds with different epoch budgets (e.g. fold 0 = 50 epochs,
    later runs can extend to 1000) without editing installed source files.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.max_num_epochs = int(
            os.environ.get("{args.epochs_env}", "{args.default_epochs}")
        )
        self.save_every = 50   # checkpoint every 50 epochs (nnUNet default)
        print(
            f"[{args.new_trainer}] max_num_epochs = {{self.max_num_epochs}}"
        )
'''

    out_file.write_text(code, encoding="utf-8")
    print(f"Installed trainer : {out_file}")
    print(f"Base trainer module: {mod_name}")

    # Verify the class is importable immediately
    import importlib as _il
    parent_pkg = mod_name.rsplit(".", 1)[0]
    new_mod_name = f"{parent_pkg}.{args.new_trainer}"
    try:
        new_mod = _il.import_module(new_mod_name)
        cls = getattr(new_mod, args.new_trainer)
        print(f"Verified import  : {cls}")
    except Exception as exc:
        raise RuntimeError(
            f"Trainer installed but import verification failed: {exc}\n"
            f"Module path tried: {new_mod_name}"
        ) from exc


if __name__ == "__main__":
    main()
