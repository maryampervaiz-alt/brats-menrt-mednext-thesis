from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Restore archived MedNeXt nnU-Net(v1) state into current workspace.")
    p.add_argument("--archive", type=str, required=True)
    p.add_argument("--dest", type=str, default=".")
    return p.parse_args()


def _is_within_directory(directory: Path, target: Path) -> bool:
    directory = directory.resolve()
    target = target.resolve()
    return str(target).startswith(str(directory))


def main() -> None:
    args = parse_args()
    archive = Path(args.archive)
    if not archive.exists():
        raise FileNotFoundError(f"Archive not found: {archive}")

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            target = dest / member.name
            if not _is_within_directory(dest, target):
                raise RuntimeError(f"Unsafe archive member path: {member.name}")
        tar.extractall(dest)

    print(f"Restored archive into: {dest.resolve()}")


if __name__ == "__main__":
    main()
