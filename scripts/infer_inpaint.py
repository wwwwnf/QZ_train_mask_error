from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from diffusers import StableDiffusionInpaintPipeline, UNet2DConditionModel

from defect_inpaint.mask_ops import perturb_mask
from defect_inpaint.utils import detect_device, list_images, open_mask, open_rgb, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate new local defects with a fine-tuned inpainting model.")
    parser.add_argument("--pretrained-model", type=str, default="runwayml/stable-diffusion-inpainting")
    parser.add_argument("--lora-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative-prompt", type=str, default="different camera angle, changed component geometry, extra parts, text, watermark")
    parser.add_argument("--num-outputs-per-image", type=int, default=1)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--strength", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perturb-mask", action="store_true")
    parser.add_argument("--allow-mask-relocation", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = detect_device()
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    unet = UNet2DConditionModel.from_pretrained(args.lora_dir / "final_unet_lora", torch_dtype=dtype)
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.pretrained_model,
        unet=unet,
        torch_dtype=dtype,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_images = list_images(args.input_dir)

    for image_index, image_path in enumerate(input_images):
        image = open_rgb(image_path)
        mask_path = args.mask_dir / image_path.name
        if not mask_path.exists():
            continue
        mask = open_mask(mask_path)

        for sample_index in range(args.num_outputs_per_image):
            active_mask = mask
            if args.perturb_mask:
                active_mask = perturb_mask(
                    mask,
                    seed=args.seed + image_index * 100 + sample_index,
                    allow_relocation=args.allow_mask_relocation,
                )

            generator = torch.Generator(device=device.type).manual_seed(args.seed + image_index * 100 + sample_index)
            result = pipe(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                image=image,
                mask_image=active_mask,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                strength=args.strength,
                generator=generator,
            ).images[0]

            output_name = f"{image_path.stem}_sample_{sample_index + 1:02d}{image_path.suffix.lower()}"
            result.save(args.output_dir / output_name)

    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
