from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate reproducibility report (env + versions + config hash)")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--out-json", type=str, default="outputs/reports/repro_report.json")
    return p.parse_args()


def _safe_run(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
        return out
    except Exception:
        return ""


def _pkg_version(name: str) -> str:
    try:
        mod = __import__(name)
        return str(getattr(mod, "__version__", "unknown"))
    except Exception:
        return "not_installed"


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg_text = cfg_path.read_text(encoding="utf-8")
    cfg = yaml.safe_load(cfg_text)
    cfg_hash = hashlib.sha256(cfg_text.encode("utf-8")).hexdigest()

    report = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "config_path": str(cfg_path.resolve()),
        "config_sha256": cfg_hash,
        "config_training": cfg.get("training", {}),
        "config_transforms": cfg.get("transforms", {}),
        "package_versions": {
            "torch": _pkg_version("torch"),
            "monai": _pkg_version("monai"),
            "numpy": _pkg_version("numpy"),
            "pandas": _pkg_version("pandas"),
            "scipy": _pkg_version("scipy"),
        },
        "git_commit": _safe_run(["git", "rev-parse", "HEAD"]),
        "git_status_short": _safe_run(["git", "status", "--short"]),
        "nvidia_smi": _safe_run(["nvidia-smi"]),
    }

    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved reproducibility report: {out_path.resolve()}")


if __name__ == "__main__":
    main()

