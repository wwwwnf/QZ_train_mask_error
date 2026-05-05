# Defect Inpaint LoRA

This project fine-tunes a structure-constrained diffusion inpainting model for local defect generation on electronic components.

## Goal

- Keep component body, camera angle, and background stable
- Generate defect textures only inside the provided mask
- Support different mask positions, sizes, and shapes
- Learn a defect style such as exposed metal, scorch, corrosion, or crack

## Data format

Training uses triplets:

- `source_image`: clean or less-defective base image
- `target_image`: defect image to learn from
- `mask_image`: local defect mask for the target image
- `prompt`: optional text prompt

The recommended directory layout is:

```text
dataset/
  source/
    0001.png
  target/
    0001.png
  mask/
    0001.png
```

Each filename should match across the three folders.

## Workflow

1. Prepare the manifest
2. Optionally augment masks for more shape diversity
3. Train an inpainting LoRA
4. Run inference with new masks in new positions

## Install

```powershell
pip install -r requirements.txt
```

## Directly from `.aqlabel`

If you already have defect images and `.aqlabel` polygon labels from `Mask_generator-main`, you can skip manual mask export.

```powershell
python scripts\prepare_from_aqlabel.py `
  --mask-generator-root "E:\path\to\Mask_Generator-main" `
  --image-dir E:\path\to\images `
  --label-dir E:\path\to\labels `
  --work-dir .\data\metal_exposure `
  --target-label "ALL" `
  --default-prompt "electronic component with local metal exposure defect"
```

This creates:

- `target/`: copied original defect images
- `mask/`: binary masks converted from `.aqlabel`
- `train_manifest.jsonl`: ready for training

## Prepare manifest

```powershell
python scripts\prepare_manifest.py `
  --source-dir .\dataset\source `
  --target-dir .\dataset\target `
  --mask-dir .\dataset\mask `
  --output .\data\train_manifest.jsonl `
  --default-prompt "electronic component with local exposed metal defect"
```

## Optional mask augmentation

```powershell
python scripts\augment_masks.py `
  --input-dir .\dataset\mask `
  --output-dir .\dataset\mask_aug `
  --copies-per-mask 4
```

## Train

```powershell
python scripts\train_lora_inpaint.py `
  --manifest .\data\train_manifest.jsonl `
  --pretrained-model runwayml/stable-diffusion-inpainting `
  --output-dir .\outputs\metal_exposure_lora `
  --resolution 512 `
  --train-batch-size 1 `
  --gradient-accumulation-steps 4 `
  --learning-rate 1e-4 `
  --max-train-steps 3000
```

## One-command train from `.aqlabel`

```powershell
python scripts\train_from_aqlabel.py `
  --mask-generator-root "E:\path\to\Mask_Generator-main" `
  --image-dir E:\path\to\images `
  --label-dir E:\path\to\labels `
  --work-dir .\data\metal_exposure `
  --output-dir .\outputs\metal_exposure_lora `
  --target-label "ALL" `
  --prompt "electronic component with realistic exposed metal defect" `
  --allow-mask-relocation `
  --max-train-steps 3000
```

## Inference

```powershell
python scripts\infer_inpaint.py `
  --pretrained-model runwayml/stable-diffusion-inpainting `
  --lora-dir .\outputs\metal_exposure_lora `
  --input-dir .\dataset\source `
  --mask-dir .\dataset\mask_aug `
  --output-dir .\outputs\samples `
  --prompt "electronic component with realistic exposed metal defect" `
  --num-outputs-per-image 3
```

## What this can and cannot do

- It can learn a defect family and render it into new mask locations
- It can vary defect size and shape if the training masks and inference masks are varied
- It can train directly from defect image plus `.aqlabel`, because the masked input is constructed during training
- It cannot reliably invent unlimited new defect semantics from only a handful of nearly identical examples
- If all masks look the same, the generated defects will also tend to collapse toward similar shapes

## Resource guidance

- Minimum practical GPU: `RTX 3060 12GB`
- Comfortable GPU: `RTX 3090 24GB` or better
- CPU-only training is technically possible in code but not practical
- A small LoRA run at `512x512` usually takes `2-8` hours depending on GPU and steps
