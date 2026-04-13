# BraTS 2024 MEN-RT — nnUNet Baseline for Meningioma GTV Segmentation

**Thesis project:** Automatic GTV (Gross Tumour Volume) segmentation of meningiomas  
from T1c MRI using nnUNet, with a two-stage pipeline that feeds predictions  
as prompts into a foundation model (SAM/MedSAM) for refinement.

---

## Project Overview

| Item | Detail |
|------|--------|
| Dataset | BraTS 2024 MEN-RT (500 training, 70 validation) |
| Modality | T1c MRI (single channel) |
| Task | Binary GTV segmentation |
| Framework | nnUNet v1 (self-configuring) |
| Smoke-test subset | 50 training + 10 held-out test cases |
| Cross-validation | Stratified 5-fold (by GTV volume) |
| Epochs | 50 per fold |

---

## Repository Structure

```
.
├── configs/
│   ├── nnunet_baseline.yaml          ← Active config (nnUNet baseline)
│   └── mednext_nnunet.yaml           ← Archive (MedNeXt — OOM on Kaggle T4)
│
├── scripts/
│   │
│   │  ── nnUNet baseline pipeline ──────────────────────────────────────
│   ├── run_nnunet.py                 ← Main orchestrator (all modes)
│   ├── prepare_nnunet_dataset.py     ← Stratified subset + nnUNet format
│   ├── create_nnunet_splits.py       ← Stratified 5-fold CV splits
│   ├── install_nnunet_trainer.py     ← Install nnUNetTrainerV2_MENRT
│   ├── evaluate_nnunet_testset.py    ← Held-out test metrics (raw vs postproc)
│   ├── plot_nnunet_results.py        ← Publication-ready figures
│   │
│   │  ── MedNeXt pipeline (archived) ──────────────────────────────────
│   ├── run_mednext_nnunet.py
│   ├── prepare_mednext_nnunet_dataset.py
│   ├── create_mednext_stratified_splits.py
│   ├── install_mednext_custom_trainer.py
│   ├── export_mednext_results.py
│   ├── plot_results.py
│   ├── visualize_predictions.py
│   └── mednext_to_prompts.py
│
├── kaggle_nnunet_baseline.ipynb      ← Kaggle notebook (active)
├── kaggle_smoke_test.ipynb           ← Kaggle notebook (MedNeXt, archived)
├── dataset_summary.json              ← Dataset statistics
└── requirements.txt
```

---

## Quick Start (Kaggle)

1. Open `kaggle_nnunet_baseline.ipynb` in Kaggle
2. Add Data → attach BraTS-MEN-RT dataset
3. Enable Internet and GPU T4
4. Run cells top to bottom

---

## Quick Start (Local / Lab GPU)

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export nnUNet_raw_data_base=/path/to/nnUNet_raw_data_base
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export RESULTS_FOLDER=/path/to/nnUNet_results
export NNUNET_MAX_EPOCHS=50

# Run full pipeline
python scripts/run_nnunet.py --config configs/nnunet_baseline.yaml --mode all

# Or step by step:
python scripts/run_nnunet.py --config configs/nnunet_baseline.yaml --mode prepare
python scripts/run_nnunet.py --config configs/nnunet_baseline.yaml --mode preprocess
python scripts/run_nnunet.py --config configs/nnunet_baseline.yaml --mode make-splits
python scripts/run_nnunet.py --config configs/nnunet_baseline.yaml --mode install-trainer
python scripts/run_nnunet.py --config configs/nnunet_baseline.yaml --mode train --fold 0
python scripts/run_nnunet.py --config configs/nnunet_baseline.yaml --mode predict-testset
python scripts/run_nnunet.py --config configs/nnunet_baseline.yaml --mode evaluate-testset

# Generate plots
python scripts/plot_nnunet_results.py --config configs/nnunet_baseline.yaml --dpi 300
```

---

## Pipeline Stages

### Stage 1 — Dataset Preparation
`prepare_nnunet_dataset.py`

- Scans all 500 training cases
- Computes GTV volume (mm³) per case
- **Stratified sampling:** selects 50 training + 10 held-out test cases,
  each balanced across 5 GTV-volume bins
- Copies to nnUNet raw task format:
  - `imagesTr/` — 50 training images (`case_0000.nii.gz`)
  - `labelsTr/` — 50 training GTV masks
  - `imagesTs/` — 10 held-out test images (for inference)
  - `labelsTs/` — 10 held-out test GTV masks (for evaluation)
- Writes `dataset.json` and `subset_manifest.json`

### Stage 2 — Plan & Preprocess
`nnUNet_plan_and_preprocess`

nnUNet automatically configures:
- Target voxel spacing (from dataset median)
- Patch size and batch size (from GPU budget)
- Intensity normalisation (Z-score for MRI)

**These values are NOT overridden.** nnUNet's auto-configuration is its
key strength and changing them would defeat its purpose.

### Stage 3 — Stratified 5-Fold Splits
`create_nnunet_splits.py`

Creates `splits_final.pkl` with 5 folds where each fold's validation set
contains a balanced distribution of GTV volumes. Seed is fixed (42) for
reproducibility.

### Stage 4 — Custom Trainer
`install_nnunet_trainer.py`

Installs `nnUNetTrainerV2_MENRT` — a thin subclass of `nnUNetTrainerV2`
that reads `max_num_epochs` from the `NNUNET_MAX_EPOCHS` environment variable.
This allows per-fold epoch control without modifying installed source files.

### Stage 5 — Training
`nnUNet_train 3d_fullres nnUNetTrainerV2_MENRT Task503_BraTSMENRT <fold>`

nnUNet saves checkpoints automatically:
- `model_best.model` — whenever validation Dice improves
- `model_latest.model` — every 50 epochs
- `model_final_checkpoint.model` — end of training

### Stage 6 — Inference on Held-Out Test Set
`run_nnunet.py --mode predict-testset`

Runs ensemble inference (all 5 folds) on the 10 held-out test cases:
1. **Postprocessed** — nnUNet default (removes small disconnected components)
2. **Raw** — pure network output (`--disable_postprocessing`)

### Stage 7 — Evaluation
`evaluate_nnunet_testset.py`

Computes per-case and mean metrics for both prediction variants:

| Metric | Description |
|--------|-------------|
| **Dice (DSC)** | Voxel overlap — primary GTV segmentation metric |
| **HD95** | 95th-percentile Hausdorff Distance in mm |
| **Sensitivity** | Fraction of GT tumour voxels captured |
| **Precision** | Fraction of predicted voxels that are correct |

Outputs a comparison table showing whether postprocessing improves results.

### Stage 8 — Figures
`plot_nnunet_results.py`

| Figure | Content |
|--------|---------|
| `cv_fold_metrics_bar.png` | Mean Dice per CV fold |
| `cv_dice_boxplot.png` | Dice distribution across folds |
| `cv_hd95_boxplot.png` | HD95 distribution across folds |
| `cv_dice_vs_hd95_scatter.png` | Dice vs HD95 per validation case |
| `testset_pre_vs_post_bar.png` | Raw vs postprocessed comparison |
| `testset_dice_boxplot.png` | Test-set Dice distribution |
| `cv_metrics_table.csv` | Mean ± SD table for thesis |

---

## Why Held-Out Test Set?

Cross-validation metrics can be optimistic because the model selection
(early stopping on validation Dice) is done on the same cases used for
CV evaluation. The 10 held-out test cases are completely unseen during
training, providing an unbiased estimate of generalisation.

```
500 labeled cases
    ├── 50 cases → 5-fold CV (training + validation within each fold)
    └── 10 cases → held-out test (NEVER used during training)
                   ← unbiased evaluation here
```

---

## Why nnUNet Instead of MedNeXt?

The original plan used MedNeXt-S kernel=3, but:
- MedNeXt-S kernel=3 requires ~15.5 GB VRAM with nnUNet's auto-configured settings
- Kaggle T4 GPU provides 14.56 GB (compatible) — 1 GB short
- Kaggle P100 GPU provides 16 GB but is CUDA sm_60 (incompatible with PyTorch ≥ sm_70)
- Modifying nnUNet's auto-configured patch/batch size was not permitted

Standard nnUNet requires ~8 GB VRAM with the same auto-configuration,
fitting comfortably on the Kaggle T4.

---

## Configuration Reference

See `configs/nnunet_baseline.yaml` for all configurable parameters.
No values are hardcoded in the scripts.

---

## Citation

If using the BraTS 2024 MEN-RT dataset, please cite according to `CITATION.bib`.
