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

## Local files to provide

Large model files and private image data are not tracked in this repository.

Place the pretrained inpainting model here:

```text
models/
  stable-diffusion-inpainting/
    tokenizer/
      merges.txt
      vocab.json
      tokenizer_config.json
      special_tokens_map.json
    text_encoder/
      config.json
      model.fp16.safetensors
    vae/
      config.json
      diffusion_pytorch_model.fp16.safetensors
    unet/
      config.json
      diffusion_pytorch_model.fp16.safetensors
    scheduler/
      scheduler_config.json
```

Place private training images and labels here if you want to reproduce the electrode exposed-copper dataset preparation:

```text
电极露铜数据/
  source/
    image_001.png
    image_002.png
  label/
    image_001.aqlabel
    image_002.aqlabel
```

The image filename stem must match the label filename stem. For example:

```text
source/image_001.png
label/image_001.aqlabel
```

After preparing the private data, generated training files will be written under:

```text
data/metal_exposure/
  target/
  mask/
  train_manifest.jsonl
```

## Directly from `.aqlabel`

If you already have defect images and `.aqlabel` polygon labels from `Mask_generator-main`, you can skip manual mask export.

```powershell
python scripts\prepare_from_aqlabel.py `
  --image-dir .\电极露铜数据\source `
  --label-dir .\电极露铜数据\label `
  --work-dir .\data\metal_exposure `
  --target-label "ALL" `
  --default-prompt "electronic component with realistic exposed copper electrode defect"
```

This creates:

- `target/`: copied original defect images
- `mask/`: binary masks converted from `.aqlabel`
- `train_manifest.jsonl`: ready for training

`--mask-generator-root` is optional. The project includes a built-in `.aqlabel` parser for the current label format.

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
  --manifest .\data\metal_exposure\train_manifest.jsonl `
  --pretrained-model .\models\stable-diffusion-inpainting `
  --output-dir .\outputs\metal_exposure_lora `
  --resolution 256 `
  --train-batch-size 1 `
  --gradient-accumulation-steps 1 `
  --learning-rate 2e-6 `
  --max-grad-norm 1.0 `
  --max-train-steps 500 `
  --checkpointing-steps 500 `
  --rank 4 `
  --mask-perturb-prob 0.2 `
  --allow-mask-relocation `
  --gradient-checkpointing `
  --skip-final-pipeline `
  --num-workers 0
```

## One-command train from `.aqlabel`

```powershell
python scripts\train_from_aqlabel.py `
  --image-dir .\电极露铜数据\source `
  --label-dir .\电极露铜数据\label `
  --work-dir .\data\metal_exposure `
  --output-dir .\outputs\metal_exposure_lora `
  --target-label "ALL" `
  --prompt "electronic component with realistic exposed copper electrode defect" `
  --pretrained-model .\models\stable-diffusion-inpainting `
  --resolution 256 `
  --learning-rate 2e-6 `
  --max-grad-norm 1.0 `
  --checkpointing-steps 500 `
  --rank 4 `
  --mask-perturb-prob 0.2 `
  --allow-mask-relocation `
  --gradient-checkpointing `
  --skip-final-pipeline `
  --max-train-steps 500
```

## Inference

```powershell
python scripts\infer_inpaint.py `
  --pretrained-model .\models\stable-diffusion-inpainting `
  --lora-dir .\outputs\metal_exposure_lora `
  --input-dir .\data\metal_exposure\target `
  --mask-dir .\data\metal_exposure\mask `
  --output-dir .\outputs\samples `
  --prompt "electronic component with realistic exposed copper electrode defect" `
  --num-outputs-per-image 1 `
  --num-inference-steps 30 `
  --resolution 256 `
  --rank 4
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
