#!/bin/bash
set -euo pipefail

python scripts/prepare_splits.py --config configs/default.yaml --kaggle
python scripts/train.py --config configs/default.yaml --kaggle --run-name kaggle_thesis_run
