from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .mask_ops import perturb_mask
from .utils import open_mask, open_rgb, read_jsonl


def _resize_rgb(image: Image.Image, size: int) -> Image.Image:
    return image.resize((size, size), Image.Resampling.BICUBIC)


def _resize_mask(mask: Image.Image, size: int) -> Image.Image:
    return mask.resize((size, size), Image.Resampling.NEAREST)


def _to_image_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1)


def _to_mask_tensor(mask: Image.Image) -> torch.Tensor:
    array = np.asarray(mask, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


class InpaintTrainingDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        resolution: int = 512,
        mask_perturb_prob: float = 0.5,
        allow_mask_relocation: bool = True,
    ) -> None:
        self.records = read_jsonl(Path(manifest_path))
        self.resolution = resolution
        self.mask_perturb_prob = mask_perturb_prob
        self.allow_mask_relocation = allow_mask_relocation

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        source_image = _resize_rgb(open_rgb(record["source_image"]), self.resolution)
        target_image = _resize_rgb(open_rgb(record["target_image"]), self.resolution)
        mask_image = _resize_mask(open_mask(record["mask_image"]), self.resolution)

        if random.random() < self.mask_perturb_prob:
            mask_image = perturb_mask(mask_image, allow_relocation=self.allow_mask_relocation)

        source_tensor = _to_image_tensor(source_image)
        target_tensor = _to_image_tensor(target_image)
        mask_tensor = _to_mask_tensor(mask_image).clamp(0.0, 1.0)
        masked_source_tensor = source_tensor * (mask_tensor < 0.5).float()

        return {
            "source_pixel_values": source_tensor,
            "target_pixel_values": target_tensor,
            "mask_values": mask_tensor,
            "masked_source_pixel_values": masked_source_tensor,
            "prompt": record.get("prompt", "electronic component local defect"),
        }
