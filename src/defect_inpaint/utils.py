from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def open_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def open_mask(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("L")


def list_images(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
