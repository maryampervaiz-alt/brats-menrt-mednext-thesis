# Thesis Experiment Checklist

## Baseline Experiment (MedNeXt-Base)

1. Fix run name and seed in config.
2. Save hardware details (GPU, CUDA, RAM).
3. Run training with full data.
4. Report best epoch by validation Dice.
5. Keep both `best_model.pt` and `latest_checkpoint.pt` archived.

## Cross-Validation (Phase-1)

1. Run 5-fold training (`scripts/train_cv.py`).
2. Save each fold split JSON (`outputs/splits/kfold_*_fold*.json`).
3. Save each fold best checkpoint (`outputs/<run_prefix>_foldX/checkpoints/best_model.pt`).
4. Build ensemble predictions with `scripts/infer_ensemble_cv.py`.
5. Report mean and std across folds for Dice and HD95.

## Phase-2 Protocol (Paper-Inspired)

1. Lock patch size to `128x160x112`.
2. Run LR sweep with `0.0027` and `0.0005`.
3. Train full 5-fold CV for each LR.
4. Generate fold metrics CSV and mean+/-std summary.
5. Auto-generate thesis tables (Markdown + LaTeX).

## Track-B (Official nnUNet-Style)

1. Use official wording:
   Architecture: MedNeXt (official package dependency)
   Training framework: nnUNetv2-style protocol with explicit MedNeXt trainer override
2. Prepare nnUNet dataset format (`prepare_nnunetv2_dataset.py`).
3. Run planning/preprocessing (`nnUNetv2_plan_and_preprocess`).
4. Train all 5 folds using `nnUNetv2_train`.
5. Save command logs and final best-configuration summary.

## Ablation Suggestions

1. Patch size ablation: 96^3 vs 128^3.
2. Spacing ablation: native vs isotropic 1mm.
3. Loss ablation: Dice+BCE vs Dice only.
4. Model scale: S vs B.

## Metrics to Report

1. Dice (mean across val cases)
2. HD95 (mean across val cases)
3. Training time per epoch
4. GPU memory footprint

## Reproducibility Artifacts

1. `config_effective.json`
2. `outputs/splits/*.json`
3. `metrics/history.csv`
4. `logs/train.log`
5. checkpoints (`best_model.pt`, `latest_checkpoint.pt`, `epoch_*.pt`)
6. overlay figures (`figures/qualitative_overlay.png`)
7. `outputs/reports/*_repro_report.json`

## Defense-Risk Mitigation

1. Run split leakage check (`scripts/check_split_leakage.py`).
2. Export per-case evaluation CSV and inspect worst Dice cases.
3. Report epoch runtime and max GPU memory from `metrics/history.csv`.
4. Use paired Wilcoxon test between candidate settings (`scripts/compare_experiment_stats.py`).

