# BraTS MEN-RT MedNeXt Thesis Pipeline

Research-grade training pipeline for BraTS 2024 MEN-RT:
- Modality: 3D T1c MRI
- Target: Binary GTV segmentation
- Backbone: Official MIC-DKFZ MedNeXt (`model_id: B` for base)

Important:
- This is **not** a scratch reimplementation of MedNeXt internals.
- It uses the official MedNeXt repository as a dependency and trains it on MEN-RT data.
- The default `train.py` / `train_cv.py` path is a custom MONAI training pipeline around the official MedNeXt architecture.
- The strict nnUNet-style workflow is provided separately as Track-B.

## What is Fixed

This repository now has one authoritative codepath:
- `src/menrt_mednext/` (canonical package)
- `configs/default.yaml` (canonical config schema)
- `scripts/train.py`, `scripts/evaluate.py`, `scripts/prepare_splits.py`, `scripts/predict.py` (canonical scripts)

The codebase is unified on `menrt_mednext` only (single workflow).

## Install

```bash
pip install -r requirements.txt
```

## Dataset Setup

Set local training root in `configs/default.yaml`:
- `data.root_dir`

For Kaggle:
- Use `--kaggle` flag
- Configure `data.kaggle_root_dir`
- Configure `data.kaggle_train_subdir` explicitly to the labeled training folder

The code discovers files recursively by keywords:
- Image keywords: `t1c`
- Label keywords: `gtv`, `seg`
- Grouped splitting is supported via `data.group_pattern` to reduce patient/session leakage risk.

## 1) Prepare Split

```bash
python scripts/prepare_splits.py --config configs/default.yaml --out-dir outputs/splits
```

This saves:
- `outputs/splits/holdout_split.json`
- `outputs/splits/dataset_index.json`

If you want fixed split for all runs, set in config:
- `data.split_json: "outputs/splits/holdout_split.json"`

## 2) Debug Run (VSCode)

```bash
python scripts/train.py --config configs/default.yaml --debug --run-name debug_mednext
```

## 3) Full Thesis Training

```bash
python scripts/train.py --config configs/default.yaml --run-name thesis_mednext_base
```

## 4) Kaggle Training

```bash
python scripts/train.py --config configs/default.yaml --kaggle --run-name kaggle_mednext_base
```

Note:
- The Kaggle training path should point to the labeled training subset only, for example `BraTS2024-MEN-RT-TrainingData`.

## 5) 5-Fold Cross-Validation (Phase-1)

```bash
python scripts/train_cv.py --config configs/default.yaml --run-prefix menrt_cv --n-folds 5 --strict-split --auto-evaluate
```

This creates fold runs:
- `outputs/menrt_cv_fold0`
- `outputs/menrt_cv_fold1`
- `outputs/menrt_cv_fold2`
- `outputs/menrt_cv_fold3`
- `outputs/menrt_cv_fold4`

Ensemble inference from fold checkpoints:

```bash
python scripts/infer_ensemble_cv.py --config configs/default.yaml --run-prefix menrt_cv --n-folds 5 --checkpoint-name best_model.pt --data-root /path/to/images --out-dir outputs/ensemble_predictions
```

## 6) Phase-2 Protocol (Paper-Inspired, Not Copy)

- Patch size locked to `128x160x112`
- LR sweep across `0.0027` and `0.0005`
- 5-fold CV per LR
- Fold summary CSV and thesis tables generated automatically

```bash
python scripts/run_phase2_protocol.py --config configs/default.yaml --run-prefix menrt_phase2 --n-folds 5
```

Manual summary/table generation:

```bash
python scripts/summarize_cv_results.py --run-prefix menrt_phase2_lr0p0027 --n-folds 5 --out-dir outputs/reports
python scripts/generate_thesis_tables.py --summary-csv outputs/reports/menrt_phase2_lr0p0027_summary_mean_std.csv --fold-csv outputs/reports/menrt_phase2_lr0p0027_fold_metrics.csv --out-dir outputs/reports/menrt_phase2_lr0p0027
```

Quality-control and defense utilities:

```bash
python scripts/check_split_leakage.py --split-json outputs/splits/kfold_5_fold0.json outputs/splits/kfold_5_fold1.json outputs/splits/kfold_5_fold2.json outputs/splits/kfold_5_fold3.json outputs/splits/kfold_5_fold4.json --check-cross-fold --fail-on-overlap
python scripts/generate_repro_report.py --config configs/default.yaml --out-json outputs/reports/repro_report.json
python scripts/compare_experiment_stats.py --a-fold-csv outputs/reports/expA_fold_metrics.csv --b-fold-csv outputs/reports/expB_fold_metrics.csv --metric val_dice --alternative two-sided
```

## 7) Resume Training

Resume priority in `train.py`:
1. `--resume <path>`
2. `training.resume_path` (config)
3. auto-resume from `<run_dir>/checkpoints/latest_checkpoint.pt` when `training.auto_resume=true`

Reproducibility tip:
- use `--strict-split` in thesis runs to force explicit split JSON.

## 8) Evaluate

```bash
python scripts/evaluate.py --config configs/default.yaml --checkpoint outputs/thesis_mednext_base/checkpoints/best_model.pt
```

## 9) Predict / Inference

```bash
python scripts/predict.py --config configs/default.yaml --checkpoint outputs/thesis_mednext_base/checkpoints/best_model.pt --data-root /path/to/images --out-dir outputs/inference
```

Inference notes:
- Default prediction export is inverted back to original image space before saving.
- Postprocessing is disabled by default in config; enable it explicitly only when you intend to report postprocessed metrics.

## 10) Strict nnUNetv2 Pipeline (If Supervisor Demands)

Use:
- [prepare_nnunetv2_dataset.py](scripts/prepare_nnunetv2_dataset.py)
- [run_nnunetv2_track_b.py](scripts/run_nnunetv2_track_b.py)
- [track_b_nnunetv2.yaml](configs/track_b_nnunetv2.yaml)
- [NNUNETV2_PROTOCOL.md](docs/NNUNETV2_PROTOCOL.md)

Example:

```bash
python scripts/prepare_nnunetv2_dataset.py --train-root /path/to/BraTS-MEN-RT-Train-v2 --val-root /path/to/BraTS-MEN-RT-Val-v1 --nnunet-raw /path/to/nnUNet_raw --dataset-id 501 --dataset-name BraTSMENRT
nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity
```

One-command strict run:

```bash
python scripts/run_nnunetv2_track_b.py --config configs/track_b_nnunetv2.yaml --mode all
```

Notes for Track-B:
- `trainer_name` is set to `nnUNetTrainerMedNeXt` and enforced by default.
- `prepare_nnunetv2_dataset.py` is run with output cleanup to avoid stale files.

## Outputs (Per Run)

`outputs/<run_name>/`
- `checkpoints/best_model.pt`
- `checkpoints/latest_checkpoint.pt`
- `checkpoints/epoch_XXXX.pt`
- `metrics/history.csv`
- `logs/train.log`
- `figures/loss_curve.png`
- `figures/dice_curve.png`
- `figures/hd95_curve.png`
- `figures/qualitative_overlay.png`
- `config_effective.json`
- `split.json`

This is the artifact set recommended for thesis defense and MICCAI-style reproducibility.
