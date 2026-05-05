from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defect_inpaint.aqlabel_adapter import AqLabelMaskAdapter
from defect_inpaint.utils import list_images, write_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare training data directly from defect images and .aqlabel files.")
    parser.add_argument("--mask-generator-root", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--target-label", type=str, default="ALL")
    parser.add_argument("--label-match-mode", choices=["exact", "contains"], default="exact")
    parser.add_argument("--default-prompt", type=str, default="electronic component local defect")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    adapter = AqLabelMaskAdapter(args.mask_generator_root)

    target_dir = args.work_dir / "target"
    mask_dir = args.work_dir / "mask"
    target_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    label_map = {path.stem: path for path in args.label_dir.rglob("*.aqlabel")}
    rows: list[dict] = []

    for image_path in list_images(args.image_dir):
        label_path = label_map.get(image_path.stem)
        if label_path is None:
            continue

        output_image = target_dir / image_path.name
        output_mask = mask_dir / f"{image_path.stem}.png"
        if not output_image.exists():
            output_image.write_bytes(image_path.read_bytes())

        mask, labels = adapter.create_mask(
            image_path=image_path,
            label_path=label_path,
            target_label=args.target_label,
            label_match_mode=args.label_match_mode,
        )
        mask.save(output_mask)

        rows.append(
            {
                "id": image_path.stem,
                "source_image": str(output_image.resolve()),
                "target_image": str(output_image.resolve()),
                "mask_image": str(output_mask.resolve()),
                "prompt": args.default_prompt,
                "labels": labels,
                "label_path": str(label_path.resolve()),
            }
        )

    if not rows:
        raise ValueError("No usable image and .aqlabel pairs were found.")

    manifest_path = args.work_dir / "train_manifest.jsonl"
    write_jsonl(manifest_path, rows)
    print(f"Saved {len(rows)} rows to {manifest_path}")


if __name__ == "__main__":
    main()
