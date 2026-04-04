# BraTS MEN-RT MedNeXt Thesis Pipeline

This repository now keeps one training workflow only:

- Architecture: **official MIC-DKFZ MedNeXt**
- Training framework: **official MedNeXt internal nnU-Net(v1)-based pipeline**
- Input: **3D T1c MRI**
- Target: **binary GTV segmentation**

The old custom MONAI/PyTorch track has been removed from the main repo surface.

## Why this repo is structured this way

The official MedNeXt repository states that its internal training framework is built on top of **nnU-Net(v1)**. This repository follows that official path instead of maintaining a separate custom training loop.

That means:

- preprocessing is done with `mednextv1_plan_and_preprocess`
- training is done with `mednextv1_train`
- trainer family is the official `nnUNetTrainerV2_MedNeXt_*`

## Install

```bash
pip install -r requirements.txt
```

This installs:

- official MedNeXt package
- explicit `nnunet==1.7.1` so `nnUNet_predict` is available consistently

## Main Config

Use:

```bash
configs/mednext_nnunet.yaml
```

Current thesis-oriented defaults:

- task name: `Task502_BraTSMENRT`
- trainer family: `nnUNetTrainerV2_MedNeXt_S_kernel3`
- custom wrapper trainer: `nnUNetTrainerV2_MedNeXt_S_kernel3_MENRT`
- deterministic smoke-test subset: `150` train cases + `20` image-only val/test cases
- train subset strategy: `stratified_label_volume`
- fold split strategy: stratified 5-fold CV using label-volume bins
- initial warm-start: `20` epochs per fold
- later full continuation: same fold, same trainer, higher epoch target

## Core Scripts

- [prepare_mednext_nnunet_dataset.py](scripts/prepare_mednext_nnunet_dataset.py)
- [create_mednext_stratified_splits.py](scripts/create_mednext_stratified_splits.py)
- [install_mednext_custom_trainer.py](scripts/install_mednext_custom_trainer.py)
- [run_mednext_nnunet.py](scripts/run_mednext_nnunet.py)
- [validate_mednext_nnunet_setup.py](scripts/validate_mednext_nnunet_setup.py)
- [export_mednext_results.py](scripts/export_mednext_results.py)
- [archive_mednext_state.py](scripts/archive_mednext_state.py)
- [restore_mednext_state.py](scripts/restore_mednext_state.py)

## Official Workflow

### 1) Prepare MEN-RT in old nnU-Net(v1) task format

```bash
python scripts/prepare_mednext_nnunet_dataset.py \
  --train-root /path/to/BraTS2024-MEN-RT-TrainingData \
  --val-root /path/to/BraTS2024-MEN-RT-ValidationData \
  --nnunet-raw-data-base /path/to/nnUNet_raw_data_base \
  --task-id 502 \
  --task-name Task502_BraTSMENRT \
  --clean-output
```

This creates:

- `nnUNet_raw_data_base/nnUNet_raw_data/Task502_BraTSMENRT/imagesTr`
- `nnUNet_raw_data_base/nnUNet_raw_data/Task502_BraTSMENRT/labelsTr`
- `nnUNet_raw_data_base/nnUNet_raw_data/Task502_BraTSMENRT/imagesTs`
- `dataset.json`
- `subset_manifest.json`

The current config defaults intentionally use a deterministic subset for smoke testing:

- `train_case_limit: 150`
- `val_case_limit: 20`
- `subset_seed: 42`
- `train_subset_strategy: stratified_label_volume`
- `stratify_volume_bins: 5`
- `split_seed: 42`

This allows you to validate the full pipeline, 5-fold orchestration, reporting, and SAM-Med3D prompt handoff before spending full-dataset compute.

### 2) Set official environment variables

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

### 3) Official MedNeXt preprocessing

```bash
mednextv1_plan_and_preprocess -t 502 -pl3d ExperimentPlanner3D_v21_customTargetSpacing_1x1x1 -pl2d None
```

This follows the official MedNeXt planner strategy:

- `1mm` isotropic spacing
- `128x128x128` planner target conditions used by the official MedNeXt internal pipeline

### 4) Install thin MEN-RT trainer wrapper

```bash
python scripts/install_mednext_custom_trainer.py \
  --base-trainer nnUNetTrainerV2_MedNeXt_S_kernel3 \
  --new-trainer nnUNetTrainerV2_MedNeXt_S_kernel3_MENRT \
  --epochs-env MEDNEXT_MAX_EPOCHS \
  --default-epochs 20
```

Purpose:

- keep the official MedNeXt trainer family
- keep the same trainer/output path across sessions
- allow first-pass `20` epochs and later continuation to `150` or any higher target

### 4.5) Validate setup before launching training

```bash
python scripts/validate_mednext_nnunet_setup.py --config configs/mednext_nnunet.yaml --check-trainer
```

This checks:

- required CLI commands are available
- official nnU-Net environment variables are set from config during validation
- train/val roots exist
- train cases and val cases can be discovered
- duplicate case IDs are detected early
- train/val overlap is reported early
- the configured official base trainer can be imported
- the effective subset size is reported
- approximate per-fold train/validation counts are reported
- subset-vs-fold compatibility is reported early

### 4.75) Generate stratified 5-fold CV splits

```bash
python scripts/create_mednext_stratified_splits.py --config configs/mednext_nnunet.yaml
```

This writes `splits_final.pkl` for nnU-Net(v1) and a JSON summary of the selected strata.

### 5) Initial warm-start for one fold

```bash
MEDNEXT_MAX_EPOCHS=20 \
mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_S_kernel3_MENRT Task502_BraTSMENRT 0 -p nnUNetPlansv2.1_trgSp_1x1x1
```

### 6) Continue an interrupted run later

```bash
MEDNEXT_MAX_EPOCHS=150 \
mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_S_kernel3_MENRT Task502_BraTSMENRT 0 -p nnUNetPlansv2.1_trgSp_1x1x1 -c
```

Because the trainer name stays the same, continuation remains in the same results directory.

Important:

- if a run is **interrupted** and you want to finish the same target training, `-c` is the correct resume path
- if you only ran a **20-epoch preview** and later want thesis-grade final numbers at `150` epochs, the cleaner option is to launch a fresh full run for that fold

## One-command Runner

Wrapper command for this repo:

```bash
python scripts/run_mednext_nnunet.py --config configs/mednext_nnunet.yaml --mode all --fold 0 --max-epochs 20
```

Continue later:

```bash
python scripts/run_mednext_nnunet.py --config configs/mednext_nnunet.yaml --mode train --fold 0 --max-epochs 150 --continue-training
```

Prediction after a trained fold:

```bash
python scripts/run_mednext_nnunet.py --config configs/mednext_nnunet.yaml --mode predict --fold 0
```

If `predict_input` is left empty in config, the runner automatically uses the prepared official task folder:

- `nnUNet_raw_data_base/nnUNet_raw_data/Task502_BraTSMENRT/imagesTs`

## Kaggle Strategy

Recommended workflow:

1. Prepare dataset once.
2. Preprocess once.
3. Install trainer wrapper once.
4. Train **one fold at a time**.
5. Start each fold with `20` epochs.
6. Inspect coarse segmentation behavior for SAM-Med3D prompting.
7. Archive fold state.
8. Resume interrupted runs with `-c`.

Pragmatic recommendation:

- first run the deterministic subset smoke test (`150` train, `20` image-only val/test)
- use `20` epochs first only to inspect coarse masks for SAM-Med3D prompting
- once the pipeline is verified, set `train_case_limit: 0` and `val_case_limit: 0` for full-dataset experiments
- for final thesis reporting, run the selected folds again with the full target budget

This is the safest workflow for limited Kaggle time.

## Archive / Restore

Archive one fold:

```bash
python scripts/archive_mednext_state.py --config configs/mednext_nnunet.yaml --fold 0 --include-preprocessed
```

Restore later:

```bash
python scripts/restore_mednext_state.py --archive /path/to/Task502_BraTSMENRT_fold0_state.tar.gz
```

## Real Result Export After Training

Once official training has actually run, the repo can export real CSV reports from official nnU-Net outputs:

```bash
python scripts/export_mednext_results.py --config configs/mednext_nnunet.yaml
```

This writes real reports only after training artifacts exist:

- fold manifest CSV
- per-case metric CSV from official `validation_raw/summary.json`
- export JSON summary

The repo does not create fake metric files before training.

## Reproducibility Artifacts

Each runner invocation saves real environment metadata under:

- `menrt_repo_artifacts/command_history.log`
- `menrt_repo_artifacts/mednext_nnunet_config_snapshot.yaml`
- `menrt_repo_artifacts/runtime_snapshot.json`
- `menrt_repo_artifacts/pip_freeze.txt`
- `menrt_repo_artifacts/git_head.txt`
- `menrt_repo_artifacts/stratified_splits_summary.json`

## Thesis Framing

Safe wording:

"We used the official MedNeXt implementation within its internal nnU-Net(v1)-based training pipeline to obtain coarse MEN-RT segmentations, which were then used for downstream refinement with SAM-Med3D."

## What to tell your supervisor

Short answer:

- official MedNeXt architecture
- official MedNeXt internal training pipeline
- nnU-Net(v1)-based preprocessing and training
- one fold at a time for stable warm-start and resume-safe execution
- initial `20` epochs to inspect coarse masks before full completion
