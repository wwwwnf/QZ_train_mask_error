from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defect_inpaint.mask_ops import perturb_mask
from defect_inpaint.utils import list_images, open_mask


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create perturbed mask variants for more location and size diversity.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--copies-per-mask", type=int, default=3)
    parser.add_argument("--allow-relocation", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    image_paths = list_images(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for image_path in image_paths:
        mask = open_mask(image_path)
        base_output = args.output_dir / image_path.name
        mask.save(base_output)
        total += 1
        for index in range(args.copies_per_mask):
            variant = perturb_mask(mask, seed=index, allow_relocation=args.allow_relocation)
            variant_name = f"{image_path.stem}_aug_{index + 1:02d}{image_path.suffix.lower()}"
            variant.save(args.output_dir / variant_name)
            total += 1

    print(f"Saved {total} masks to {args.output_dir}")


if __name__ == "__main__":
    main()
