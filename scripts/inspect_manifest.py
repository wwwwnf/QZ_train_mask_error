from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defect_inpaint.utils import read_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect a training manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--head", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = read_jsonl(args.manifest)
    print(f"rows={len(rows)}")
    for row in rows[: args.head]:
        print(row)


if __name__ == "__main__":
    main()
