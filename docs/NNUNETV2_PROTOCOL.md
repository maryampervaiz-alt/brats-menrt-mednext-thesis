# nnUNetv2 + MedNeXt Protocol (Strict Supervisor Mode)

This protocol is for the case where your supervisor explicitly asks for nnUNetv2 pipeline usage.

## 1) Prepare nnUNet dataset format

Use the conversion script:

```bash
python scripts/prepare_nnunetv2_dataset.py \
  --train-root /path/to/BraTS-MEN-RT-Train-v2 \
  --val-root /path/to/BraTS-MEN-RT-Val-v1 \
  --nnunet-raw /path/to/nnUNet_raw \
  --dataset-id 501 \
  --dataset-name BraTSMENRT \
  --copy-mode copy
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

If MedNeXt-specific nnUNet trainer classes are installed:

```bash
nnUNetv2_train 501 3d_fullres 0 -tr nnUNetTrainerMedNeXt
```

## 5) Find best configuration

```bash
nnUNetv2_find_best_configuration 501 -c 3d_fullres
```

## Notes

1. This protocol is nnUNet-native. It is separate from the custom MONAI scripts in this repo.
2. Use this mode when supervisor explicitly asks for `nnUNetv2_plan_and_preprocess` based workflow.
3. Keep both protocols documented in thesis (custom MedNeXt pipeline + strict nnUNetv2 pipeline).

