from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defect_inpaint.utils import list_images, write_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a JSONL manifest from source, target, and mask folders.")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--default-prompt", type=str, default="electronic component local defect")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_images = {path.name: path for path in list_images(args.source_dir)}
    target_images = {path.name: path for path in list_images(args.target_dir)}
    mask_images = {path.name: path for path in list_images(args.mask_dir)}

    shared_names = sorted(set(source_images) & set(target_images) & set(mask_images))
    rows = [
        {
            "id": name,
            "source_image": str(source_images[name].resolve()),
            "target_image": str(target_images[name].resolve()),
            "mask_image": str(mask_images[name].resolve()),
            "prompt": args.default_prompt,
        }
        for name in shared_names
    ]

    if not rows:
        raise ValueError("No matching filenames were found across source, target, and mask directories.")

    write_jsonl(args.output, rows)
    print(f"Saved {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
