#!/bin/bash
set -euo pipefail

python scripts/run_mednext_nnunet.py --config configs/mednext_nnunet.yaml --mode prepare
python scripts/validate_mednext_nnunet_setup.py --config configs/mednext_nnunet.yaml --check-trainer
python scripts/run_mednext_nnunet.py --config configs/mednext_nnunet.yaml --mode preprocess
python scripts/run_mednext_nnunet.py --config configs/mednext_nnunet.yaml --mode install-trainer
python scripts/run_mednext_nnunet.py --config configs/mednext_nnunet.yaml --mode train --fold 0 --max-epochs 20
