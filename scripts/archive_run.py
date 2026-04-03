from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Archive one run directory for Kaggle-safe resume/export.")
    p.add_argument("--run-name", type=str, required=True)
    p.add_argument("--archive-path", type=str, default="")
    p.add_argument("--split-json", type=str, default="")
    return p.parse_args()


def _safe_arcname(path: Path) -> str:
    return path.as_posix()


def main() -> None:
    args = parse_args()
    run_dir = Path("outputs") / args.run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    archive_path = Path(args.archive_path) if args.archive_path else Path("outputs") / "archives" / f"{args.run_name}.tar.gz"
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(run_dir, arcname=_safe_arcname(run_dir))
        if args.split_json:
            split_path = Path(args.split_json)
            if not split_path.exists():
                raise FileNotFoundError(f"Split JSON not found: {split_path}")
            tar.add(split_path, arcname=_safe_arcname(split_path))

    print(f"Saved archive: {archive_path}")


if __name__ == "__main__":
    main()
