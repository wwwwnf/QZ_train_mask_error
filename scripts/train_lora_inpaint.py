from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn.functional as functional
from peft import LoraConfig
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionInpaintPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler

from defect_inpaint.dataset import InpaintTrainingDataset
from defect_inpaint.utils import detect_device, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune a LoRA for structure-constrained defect inpainting.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pretrained-model", type=str, default="runwayml/stable-diffusion-inpainting")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler", type=str, default="cosine")
    parser.add_argument("--lr-warmup-steps", type=int, default=100)
    parser.add_argument("--max-train-steps", type=int, default=3000)
    parser.add_argument("--checkpointing-steps", type=int, default=500)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask-perturb-prob", type=float, default=0.6)
    parser.add_argument("--allow-mask-relocation", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


def collate_batch(batch: list[dict], tokenizer: AutoTokenizer) -> dict:
    tokenized = tokenizer(
        [sample["prompt"] for sample in batch],
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    return {
        "source_pixel_values": torch.stack([sample["source_pixel_values"] for sample in batch]),
        "target_pixel_values": torch.stack([sample["target_pixel_values"] for sample in batch]),
        "mask_values": torch.stack([sample["mask_values"] for sample in batch]),
        "masked_source_pixel_values": torch.stack([sample["masked_source_pixel_values"] for sample in batch]),
        "input_ids": tokenized.input_ids,
        "attention_mask": tokenized.attention_mask,
    }


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = detect_device()
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model, subfolder="tokenizer", use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model, subfolder="text_encoder", torch_dtype=dtype)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model, subfolder="vae", torch_dtype=dtype)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model, subfolder="unet", torch_dtype=dtype)
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_config)

    vae.to(device)
    text_encoder.to(device)
    unet.to(device)

    dataset = InpaintTrainingDataset(
        manifest_path=args.manifest,
        resolution=args.resolution,
        mask_perturb_prob=args.mask_perturb_prob,
        allow_mask_relocation=args.allow_mask_relocation,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_batch(batch, tokenizer),
    )

    optimizer = torch.optim.AdamW(
        [parameter for parameter in unet.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0
    progress = tqdm(total=args.max_train_steps, desc="training")

    while global_step < args.max_train_steps:
        for batch in dataloader:
            with torch.no_grad():
                target_latents = vae.encode(batch["target_pixel_values"].to(device=device, dtype=dtype)).latent_dist.sample()
                target_latents = target_latents * vae.config.scaling_factor

                masked_source_latents = vae.encode(
                    batch["masked_source_pixel_values"].to(device=device, dtype=dtype)
                ).latent_dist.sample()
                masked_source_latents = masked_source_latents * vae.config.scaling_factor

                encoder_hidden_states = text_encoder(batch["input_ids"].to(device))[0]

            noise = torch.randn_like(target_latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (target_latents.shape[0],),
                device=device,
                dtype=torch.long,
            )
            noisy_latents = noise_scheduler.add_noise(target_latents, noise, timesteps)

            latent_mask = functional.interpolate(
                batch["mask_values"].to(device=device, dtype=dtype),
                size=(target_latents.shape[-2], target_latents.shape[-1]),
                mode="nearest",
            )

            model_input = torch.cat([noisy_latents, latent_mask, masked_source_latents], dim=1)
            model_pred = unet(model_input, timesteps, encoder_hidden_states).sample

            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(target_latents, noise, timesteps)
            else:
                raise ValueError(f"Unsupported prediction type: {noise_scheduler.config.prediction_type}")

            loss = functional.mse_loss(model_pred.float(), target.float(), reduction="mean")
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            progress.update(1)
            progress.set_postfix(loss=f"{loss.item() * args.gradient_accumulation_steps:.4f}")

            if global_step % args.checkpointing_steps == 0 or global_step == args.max_train_steps:
                checkpoint_dir = args.output_dir / f"checkpoint-{global_step}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                unet.save_pretrained(checkpoint_dir / "unet_lora")
                tokenizer.save_pretrained(checkpoint_dir / "tokenizer")

            if global_step >= args.max_train_steps:
                break

    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        args.pretrained_model,
        unet=unet,
        torch_dtype=dtype,
    )
    pipeline.save_pretrained(args.output_dir / "final_pipeline")
    unet.save_pretrained(args.output_dir / "final_unet_lora")
    tokenizer.save_pretrained(args.output_dir / "tokenizer")
    print(f"Training finished. Outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
