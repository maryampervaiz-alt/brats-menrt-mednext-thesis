# Official MedNeXt nnU-Net(v1) Protocol

This repository now uses one authoritative workflow only:

- Architecture: **official MIC-DKFZ MedNeXt**
- Training framework: **official MedNeXt internal nnU-Net(v1)-based pipeline**
- Task: **MEN-RT T1c -> binary GTV segmentation**

Important:
- This is **not** the custom MONAI/PyTorch track anymore.
- This is **not** an nnUNetv2 workflow.
- It follows the official MedNeXt repository protocol, which is built on **nnU-Net(v1)**.

## Thesis Wording

Use this wording in thesis/viva:

1. Architecture: **MedNeXt (official MIC-DKFZ implementation)**
2. Training framework: **official MedNeXt nnU-Net(v1)-based pipeline**
3. Preprocessing: **official MedNeXt planner with 1mm isotropic target spacing**
4. Training strategy: **5-fold cross-validation**
5. Engineering strategy: **deterministic subset smoke test before full-dataset execution**

## Core Commands

### 1) Prepare MEN-RT task in old nnU-Net format

```bash
python scripts/prepare_mednext_nnunet_dataset.py \
  --train-root /path/to/BraTS2024-MEN-RT-TrainingData \
  --val-root /path/to/BraTS2024-MEN-RT-ValidationData \
  --nnunet-raw-data-base /path/to/nnUNet_raw_data_base \
  --task-id 502 \
  --task-name Task502_BraTSMENRT \
  --clean-output
```

Current config defaults are intentionally smoke-test oriented:

- `train_case_limit: 50`
- `val_case_limit: 20`
- `subset_seed: 42`
- `train_subset_strategy: stratified_label_volume`
- `stratify_volume_bins: 5`
- `split_seed: 42`

The selected cases are recorded in `subset_manifest.json`.

### 2) Set official nnU-Net(v1) environment variables

```bash
export nnUNet_raw_data_base=/path/to/nnUNet_raw_data_base
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export RESULTS_FOLDER=/path/to/nnUNet_results
```

PowerShell:

```powershell
$env:nnUNet_raw_data_base="C:\path\to\nnUNet_raw_data_base"
$env:nnUNet_preprocessed="C:\path\to\nnUNet_preprocessed"
$env:RESULTS_FOLDER="C:\path\to\nnUNet_results"
```

### 3) Plan and preprocess

```bash
mednextv1_plan_and_preprocess -t 502 -pl3d ExperimentPlanner3D_v21_customTargetSpacing_1x1x1 -pl2d None
```

### 4) Install thin MEN-RT trainer wrapper

```bash
python scripts/install_mednext_custom_trainer.py \
  --base-trainer nnUNetTrainerV2_MedNeXt_S_kernel3 \
  --new-trainer nnUNetTrainerV2_MedNeXt_S_kernel3_MENRT \
  --epochs-env MEDNEXT_MAX_EPOCHS \
  --default-epochs 30
```

### 4.5) Validate setup before training

```bash
python scripts/validate_mednext_nnunet_setup.py --config configs/mednext_nnunet.yaml --check-trainer
```

This validation step should be treated as mandatory before long Kaggle runs. It reports:

- full discovered train/val counts
- effective subset counts after limits are applied
- approximate per-fold train/validation counts for the configured number of folds
- whether the current subset is compatible with 5-fold CV

### 4.75) Generate stratified splits before training

```bash
python scripts/create_mednext_stratified_splits.py --config configs/mednext_nnunet.yaml
```

This writes `splits_final.pkl` for nnU-Net(v1) and a JSON summary of case strata.

### 5) Train one fold for the subset-50 Kaggle target

```bash
MEDNEXT_MAX_EPOCHS=30 \
mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_S_kernel3_MENRT Task502_BraTSMENRT 0 -p nnUNetPlansv2.1_trgSp_1x1x1
```

### 6) Continue an interrupted run later

```bash
MEDNEXT_MAX_EPOCHS=150 \
mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_S_kernel3_MENRT Task502_BraTSMENRT 0 -p nnUNetPlansv2.1_trgSp_1x1x1 -c
```

Because the trainer name stays the same, the output directory stays the same and continuation is clean for interrupted runs.

Important:

- `20` epochs first is useful for quick coarse-mask inspection
- for final thesis-grade reporting, a fresh full run at the intended final epoch budget is the cleaner experimental choice

### 7) Predict coarse masks for SAM-Med3D

```bash
python scripts/run_mednext_nnunet.py --config configs/mednext_nnunet.yaml --mode predict --fold 0
```

If `predict_input` is empty, the runner uses the prepared official task `imagesTs` folder automatically.

## Kaggle Strategy

Recommended:

1. First validate the repository on the deterministic subset smoke test.
2. Prepare + preprocess once.
3. Train one fold at a time.
4. Target `MEDNEXT_MAX_EPOCHS=30` on the `50`-case subset.
5. Leave the nnU-Net auto-generated patch size and batch size untouched.
6. Inspect coarse segmentation quality for SAM-Med3D prompting.
7. Archive fold state after each session.
8. Resume later with a higher `MEDNEXT_MAX_EPOCHS` and `-c`.
9. Only after smoke-test success, switch `train_case_limit: 0` and `val_case_limit: 0` for full-dataset runs.

## Archive / Restore

Archive one fold:

```bash
python scripts/archive_mednext_state.py --config configs/mednext_nnunet.yaml --fold 0 --include-preprocessed
```

Restore later:

```bash
python scripts/restore_mednext_state.py --archive /path/to/Task502_BraTSMENRT_fold0_state.tar.gz
```

## Real Report Export After Training

After official training finishes, export real CSV summaries:

```bash
python scripts/export_mednext_results.py --config configs/mednext_nnunet.yaml
```

## Reproducibility Artifacts

Each runner invocation also writes:

- `command_history.log`
- `mednext_nnunet_config_snapshot.yaml`
- `runtime_snapshot.json`
- `pip_freeze.txt`
- `git_head.txt`
- `stratified_splits_summary.json`
