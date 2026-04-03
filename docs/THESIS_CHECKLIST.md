# Thesis Experiment Checklist

## Official Pipeline

1. Use official wording:
   Architecture: MedNeXt (official MIC-DKFZ implementation)
   Training framework: official MedNeXt nnU-Net(v1)-based pipeline
2. Prepare MEN-RT in old nnU-Net task format with `scripts/prepare_mednext_nnunet_dataset.py`.
3. Run official preprocessing with `mednextv1_plan_and_preprocess`.
4. Train using official MedNeXt trainer family (`nnUNetTrainerV2_MedNeXt_*`).
5. Validate CLI, case discovery, duplicate IDs, and trainer import with `scripts/validate_mednext_nnunet_setup.py`.

## Initial Warm-Start

1. Train one fold at a time.
2. Use `nnUNetTrainerV2_MedNeXt_S_kernel3_MENRT`.
3. Set `MEDNEXT_MAX_EPOCHS=20` for the first pass on each fold.
4. Archive fold state after each session with `scripts/archive_mednext_state.py`.
5. Save Kaggle version after each fold/session.

## Full Training Later

1. Restore archived state if needed.
2. Use `-c` to continue interrupted runs of the same target training.
3. For final thesis reporting after a 20-epoch preview, prefer a fresh full run at the final target budget, for example `150`.
4. Keep the same trainer family and plans identifier.

## Output Use

1. Use MedNeXt output as coarse segmentation.
2. Feed coarse segmentation to SAM-Med3D for refinement.
3. Keep fold-wise checkpoints and command logs.
4. Export real fold and per-case CSV reports after training with `scripts/export_mednext_results.py`.

## Defense Points

1. Explain clearly that you selected the official MedNeXt internal training pipeline instead of a custom pipeline.
2. State that this reduces implementation risk and improves reproducibility.
3. State that Stage-1 uses MedNeXt for coarse segmentation and Stage-2 uses SAM-Med3D for refinement.

