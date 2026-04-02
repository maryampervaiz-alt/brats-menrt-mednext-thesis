# nnUNetv2 + MedNeXt Protocol (Strict Supervisor Mode)

This protocol is for the case where your supervisor explicitly asks for nnUNetv2 pipeline usage.

## Official Thesis Wording (Track-B)

Use this exact framing in thesis:

1. Architecture: **MedNeXt (official package dependency)**
2. Training framework: **nnUNetv2-style protocol with explicit MedNeXt trainer override**
3. Data pipeline: nnUNetv2 planning and preprocessing
4. Validation strategy: 5-fold cross-validation on labeled training data

## 1) Prepare nnUNet dataset format

Use the conversion script:

```bash
python scripts/prepare_nnunetv2_dataset.py \
  --train-root /path/to/BraTS-MEN-RT-Train-v2 \
  --val-root /path/to/BraTS-MEN-RT-Val-v1 \
  --nnunet-raw /path/to/nnUNet_raw \
  --dataset-id 501 \
  --dataset-name BraTSMENRT \
  --copy-mode copy \
  --clean-output
```

This creates:
- `nnUNet_raw/Dataset501_BraTSMENRT/imagesTr/*_0000.nii.gz`
- `nnUNet_raw/Dataset501_BraTSMENRT/labelsTr/*.nii.gz`
- `nnUNet_raw/Dataset501_BraTSMENRT/imagesTs/*_0000.nii.gz` (if val root provided)
- `dataset.json`
- `NNUNET_COMMANDS.txt`

## 2) Set nnUNet environment variables

```bash
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results
```

On Windows PowerShell:

```powershell
$env:nnUNet_raw="C:\path\to\nnUNet_raw"
$env:nnUNet_preprocessed="C:\path\to\nnUNet_preprocessed"
$env:nnUNet_results="C:\path\to\nnUNet_results"
```

## 3) Plan and preprocess

```bash
nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity
```

## 4) Train 5 folds

```bash
nnUNetv2_train 501 3d_fullres 0
nnUNetv2_train 501 3d_fullres 1
nnUNetv2_train 501 3d_fullres 2
nnUNetv2_train 501 3d_fullres 3
nnUNetv2_train 501 3d_fullres 4
```

Track-B should be run with MedNeXt trainer override:

```bash
nnUNetv2_train 501 3d_fullres 0 -tr nnUNetTrainerMedNeXt
```

If your environment does not provide this trainer class, install the matching MedNeXt/nnUNet package first.

## 5) Find best configuration

```bash
nnUNetv2_find_best_configuration 501 -c 3d_fullres
```

## One-command Track-B Runner

This repository includes:

1. Config: `configs/track_b_nnunetv2.yaml`
2. Runner: `scripts/run_nnunetv2_track_b.py`

Dry-run to inspect all commands:

```bash
python scripts/run_nnunetv2_track_b.py --config configs/track_b_nnunetv2.yaml --mode all --dry-run
```

Execute full Track-B:

```bash
python scripts/run_nnunetv2_track_b.py --config configs/track_b_nnunetv2.yaml --mode all
```

## Notes

1. This protocol is nnUNet-native. It is separate from the custom MONAI scripts in this repo.
2. `configs/track_b_nnunetv2.yaml` enforces explicit `trainer_name` by default (`require_mednext_trainer: true`).
3. Keep both protocols documented in thesis (custom MedNeXt pipeline + strict nnUNetv2 pipeline).
