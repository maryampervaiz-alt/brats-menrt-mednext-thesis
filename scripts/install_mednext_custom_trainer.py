from __future__ import annotations

import argparse
import importlib
import inspect
import pkgutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Install a thin custom MedNeXt nnU-Net(v1) trainer wrapper into the MedNeXt package.")
    p.add_argument("--base-trainer", type=str, required=True)
    p.add_argument("--new-trainer", type=str, required=True)
    p.add_argument("--epochs-env", type=str, default="MEDNEXT_MAX_EPOCHS")
    p.add_argument("--default-epochs", type=int, default=20)
    return p.parse_args()


def _locate_trainer(base_trainer: str) -> tuple[str, Path]:
    root_mod = importlib.import_module("nnunet_mednext")
    if hasattr(root_mod, base_trainer):
        mod = root_mod
        return mod.__name__, Path(inspect.getfile(mod)).resolve()

    mod_path = getattr(root_mod, "__path__", None)
    if mod_path is None:
        raise RuntimeError("nnunet_mednext package has no __path__; cannot scan trainers.")

    for mod_info in pkgutil.walk_packages(mod_path, prefix=f"{root_mod.__name__}."):
        try:
            mod = importlib.import_module(mod_info.name)
        except Exception:
            continue
        if hasattr(mod, base_trainer):
            return mod.__name__, Path(inspect.getfile(mod)).resolve()

    raise RuntimeError(f"Could not find base trainer class: {base_trainer}")


def main() -> None:
    args = parse_args()
    module_name, module_file = _locate_trainer(args.base_trainer)
    out_file = module_file.parent / f"{args.new_trainer}.py"

    code = f'''from __future__ import annotations

import os
from {module_name} import {args.base_trainer}


class {args.new_trainer}({args.base_trainer}):
    """
    Thin wrapper around the official MedNeXt nnU-Net(v1) trainer.
    It preserves the official pipeline while allowing stage-wise epoch control
    through an environment variable so the same trainer path can be resumed later.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        max_epochs = int(os.environ.get("{args.epochs_env}", "{args.default_epochs}"))
        if hasattr(self, "max_num_epochs"):
            self.max_num_epochs = max_epochs
        if hasattr(self, "num_epochs"):
            self.num_epochs = max_epochs
'''
    out_file.write_text(code, encoding="utf-8")
    print(f"Installed trainer wrapper: {out_file}")
    print(f"Base trainer module: {module_name}")


if __name__ == "__main__":
    main()
