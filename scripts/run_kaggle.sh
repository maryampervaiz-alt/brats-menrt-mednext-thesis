#!/bin/bash
set -euo pipefail

python scripts/validate_mednext_nnunet_setup.py --config configs/mednext_nnunet.yaml --check-trainer
python scripts/run_mednext_nnunet.py --config configs/mednext_nnunet.yaml --mode prepare
python scripts/run_mednext_nnunet.py --config configs/mednext_nnunet.yaml --mode preprocess
python scripts/run_mednext_nnunet.py --config configs/mednext_nnunet.yaml --mode make-splits
python scripts/run_mednext_nnunet.py --config configs/mednext_nnunet.yaml --mode install-trainer

for FOLD in 0 1 2 3 4; do
  python scripts/run_mednext_nnunet.py --config configs/mednext_nnunet.yaml --mode train --fold "${FOLD}" --max-epochs 30
done
