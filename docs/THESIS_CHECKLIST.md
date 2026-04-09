# Thesis Experiment Checklist

## Official Pipeline

1. Use official wording:
   Architecture: MedNeXt (official MIC-DKFZ implementation)
   Training framework: official MedNeXt nnU-Net(v1)-based pipeline
2. Prepare MEN-RT in old nnU-Net task format with `scripts/prepare_mednext_nnunet_dataset.py`.
3. Run official preprocessing with `mednextv1_plan_and_preprocess`.
4. Generate stratified `splits_final.pkl` with `scripts/create_mednext_stratified_splits.py`.
5. Train using official MedNeXt trainer family (`nnUNetTrainerV2_MedNeXt_*`).
6. Validate CLI, case discovery, duplicate IDs, and trainer import with `scripts/validate_mednext_nnunet_setup.py`.
7. Run the deterministic subset smoke test first (`train_case_limit: 50`, `val_case_limit: 20`) before any full-dataset run.

## Initial Warm-Start

1. Train one fold at a time.
2. Use `nnUNetTrainerV2_MedNeXt_S_kernel3_MENRT`.
3. For repository validation, first run the subset smoke test across folds.
4. Set `MEDNEXT_MAX_EPOCHS=30` for the Kaggle subset run on each fold.
5. Use the stratified subset and stratified CV splits for the smoke test.
6. Leave the nnU-Net auto-generated patch size and batch size untouched during the subset-50 Kaggle run.
7. Archive fold state after each session with `scripts/archive_mednext_state.py`.
8. Save Kaggle version after each fold/session.

## Full Training Later

1. After subset smoke-test success, set `train_case_limit: 0` and `val_case_limit: 0`.
2. Restore archived state if needed.
3. Use `-c` to continue interrupted runs of the same target training.
4. For final thesis reporting after the subset-50 Kaggle smoke test, prefer a fresh full run at the final target budget on stronger hardware.
5. Keep the same trainer family and plans identifier.

## Output Use

1. Use MedNeXt output as coarse segmentation.
2. Feed coarse segmentation to SAM-Med3D for refinement.
3. Keep fold-wise checkpoints and command logs.
4. Keep runtime metadata artifacts (`runtime_snapshot.json`, `pip_freeze.txt`, `git_head.txt`).
5. Export real fold and per-case CSV reports after training with `scripts/export_mednext_results.py`.

## Defense Points

1. Explain clearly that you selected the official MedNeXt internal training pipeline instead of a custom pipeline.
2. State that this reduces implementation risk and improves reproducibility.
3. State that Stage-1 uses MedNeXt for coarse segmentation and Stage-2 uses SAM-Med3D for refinement.

