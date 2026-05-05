from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="One-command training from defect images and .aqlabel files.")
    parser.add_argument("--mask-generator-root", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-label", type=str, default="ALL")
    parser.add_argument("--label-match-mode", choices=["exact", "contains"], default="exact")
    parser.add_argument("--prompt", type=str, default="electronic component local defect")
    parser.add_argument("--pretrained-model", type=str, default="runwayml/stable-diffusion-inpainting")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-train-steps", type=int, default=3000)
    parser.add_argument("--checkpointing-steps", type=int, default=500)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask-perturb-prob", type=float, default=0.6)
    parser.add_argument("--allow-mask-relocation", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]

    prepare_script = project_root / "scripts" / "prepare_from_aqlabel.py"
    train_script = project_root / "scripts" / "train_lora_inpaint.py"

    prepare_cmd = [
        sys.executable,
        str(prepare_script),
        "--mask-generator-root",
        str(args.mask_generator_root),
        "--image-dir",
        str(args.image_dir),
        "--label-dir",
        str(args.label_dir),
        "--work-dir",
        str(args.work_dir),
        "--target-label",
        args.target_label,
        "--label-match-mode",
        args.label_match_mode,
        "--default-prompt",
        args.prompt,
    ]
    subprocess.run(prepare_cmd, check=True)

    manifest_path = args.work_dir / "train_manifest.jsonl"
    train_cmd = [
        sys.executable,
        str(train_script),
        "--manifest",
        str(manifest_path),
        "--pretrained-model",
        args.pretrained_model,
        "--output-dir",
        str(args.output_dir),
        "--resolution",
        str(args.resolution),
        "--train-batch-size",
        str(args.train_batch_size),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--learning-rate",
        str(args.learning_rate),
        "--max-train-steps",
        str(args.max_train_steps),
        "--checkpointing-steps",
        str(args.checkpointing_steps),
        "--rank",
        str(args.rank),
        "--seed",
        str(args.seed),
        "--mask-perturb-prob",
        str(args.mask_perturb_prob),
        "--num-workers",
        str(args.num_workers),
    ]
    if args.allow_mask_relocation:
        train_cmd.append("--allow-mask-relocation")

    subprocess.run(train_cmd, check=True)


if __name__ == "__main__":
    main()
