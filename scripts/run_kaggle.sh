#!/bin/bash
# =============================================================================
# run_kaggle.sh  —  MedNeXt MEN-RT full pipeline for Kaggle GPU execution
# =============================================================================
#
# Pipeline stages (each guarded so failures are caught early):
#   1. Validate setup       — check commands, dataset paths, case counts
#   2. Prepare dataset      — copy subset (50 train / 20 val) into nnUNet format
#   3. Plan & preprocess    — nnUNet auto-configures patch size / batch size
#   4. Make CV splits       — stratified 5-fold splits written to splits_final.pkl
#   5. Install trainer      — register custom epoch-configurable trainer class
#   6. Train
#        Fold 0 : 30 epochs  (primary fold — used for qualitative analysis)
#        Fold 1 : 20 epochs  ─┐
#        Fold 2 : 20 epochs   │  cross-validation folds (fewer epochs because
#        Fold 3 : 20 epochs   │  smoke test; extend for full training run)
#        Fold 4 : 20 epochs  ─┘
#   7. Export results       — write fold_manifest.csv + per_case_metrics.csv
#   8. Plot results         — publication-ready figures (only after training)
#   9. Visualize fold 0     — segmentation overlays (axial / coronal / sagittal)
#  10. Generate prompts     — MedNeXt → foundation model prompt JSONs (fold 0)
#
# Checkpoints & resuming
#   nnUNet automatically saves model_best.model, model_latest.model, and
#   model_final_checkpoint.model in each fold_X/ directory.
#   To RESUME a fold that was interrupted, add --continue-training to the
#   relevant train command (it passes -c to mednextv1_train).
#   Example: python scripts/run_mednext_nnunet.py --config ... --mode train
#              --fold 0 --max-epochs 30 --continue-training
#
# Usage
#   bash scripts/run_kaggle.sh
#   CONFIG=configs/my_custom.yaml bash scripts/run_kaggle.sh
# =============================================================================

set -euo pipefail   # exit on error, undefined var, or pipe failure

CONFIG="${CONFIG:-configs/mednext_nnunet.yaml}"

echo "========================================================================"
echo " MedNeXt MEN-RT Pipeline — Kaggle GPU"
echo " Config : ${CONFIG}"
echo " $(date)"
echo "========================================================================"

# ── Stage 1: Validate setup ──────────────────────────────────────────────────
echo ""
echo "── Stage 1/10: Validate setup ──────────────────────────────────────────"
python scripts/validate_mednext_nnunet_setup.py \
    --config "${CONFIG}" \
    --check-trainer

# ── Stage 2: Prepare dataset ─────────────────────────────────────────────────
echo ""
echo "── Stage 2/10: Prepare dataset (50 train / 20 val subset) ─────────────"
python scripts/run_mednext_nnunet.py \
    --config "${CONFIG}" \
    --mode prepare

# ── Stage 3: Plan and preprocess ─────────────────────────────────────────────
echo ""
echo "── Stage 3/10: nnUNet plan and preprocess ──────────────────────────────"
echo "   (nnUNet auto-configures patch size and batch size — do not override)"
python scripts/run_mednext_nnunet.py \
    --config "${CONFIG}" \
    --mode preprocess

# ── Stage 4: Create stratified CV splits ─────────────────────────────────────
echo ""
echo "── Stage 4/10: Create stratified 5-fold CV splits ─────────────────────"
python scripts/run_mednext_nnunet.py \
    --config "${CONFIG}" \
    --mode make-splits

# ── Stage 5: Install custom trainer ──────────────────────────────────────────
echo ""
echo "── Stage 5/10: Install custom MedNeXt trainer ──────────────────────────"
python scripts/run_mednext_nnunet.py \
    --config "${CONFIG}" \
    --mode install-trainer

# ── Stage 6: Training ────────────────────────────────────────────────────────
#
# Fold 0 gets 30 epochs because it is the primary evaluation fold used for
# qualitative analysis and prompt generation.  Folds 1–4 use 20 epochs each
# for the smoke-test cross-validation run; increase to match Fold 0 when
# running the full training experiment.
#
# To RESUME any fold if Kaggle disconnects:
#   python scripts/run_mednext_nnunet.py --config ${CONFIG} --mode train \
#       --fold <N> --max-epochs <E> --continue-training

echo ""
echo "── Stage 6/10: Training ─────────────────────────────────────────────────"

echo "   [Fold 0] 30 epochs  (primary fold)"
python scripts/run_mednext_nnunet.py \
    --config "${CONFIG}" \
    --mode train \
    --fold 0 \
    --max-epochs 30

for FOLD in 1 2 3 4; do
    echo ""
    echo "   [Fold ${FOLD}] 20 epochs  (CV fold)"
    python scripts/run_mednext_nnunet.py \
        --config "${CONFIG}" \
        --mode train \
        --fold "${FOLD}" \
        --max-epochs 20
done

# ── Stage 7: Export results to CSV ───────────────────────────────────────────
echo ""
echo "── Stage 7/10: Export training results to CSV ──────────────────────────"
python scripts/export_mednext_results.py \
    --config "${CONFIG}"

# ── Stage 8: Generate plots ───────────────────────────────────────────────────
echo ""
echo "── Stage 8/10: Generate publication-ready figures ───────────────────────"
python scripts/plot_results.py \
    --config "${CONFIG}" \
    --dpi 300

# ── Stage 9: Visualize fold-0 predictions ────────────────────────────────────
echo ""
echo "── Stage 9/10: Generate segmentation overlay figures (fold 0) ──────────"
python scripts/visualize_predictions.py \
    --config "${CONFIG}" \
    --fold 0 \
    --mode val \
    --num-cases 6 \
    --dpi 200

# ── Stage 10: Generate foundation-model prompts from fold-0 predictions ──────
echo ""
echo "── Stage 10/10: Generate foundation-model prompt JSONs (fold 0) ────────"
python scripts/mednext_to_prompts.py \
    --config "${CONFIG}" \
    --source val \
    --fold 0 \
    --n-pos 5 \
    --n-neg 5 \
    --neg-dilation-mm 5.0

echo ""
echo "========================================================================"
echo " Pipeline complete!  $(date)"
echo " Results : \$(python -c \"import yaml; c=yaml.safe_load(open('${CONFIG}'))['mednext_nnunet']; print(c['results_folder'])\")"
echo "========================================================================"
