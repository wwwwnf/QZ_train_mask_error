from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageFilter


def _shift_binary(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    shifted = np.zeros_like(mask)
    src_y0 = max(0, -dy)
    src_y1 = mask.shape[0] - max(0, dy)
    src_x0 = max(0, -dx)
    src_x1 = mask.shape[1] - max(0, dx)
    dst_y0 = max(0, dy)
    dst_x0 = max(0, dx)
    dst_y1 = dst_y0 + max(0, src_y1 - src_y0)
    dst_x1 = dst_x0 + max(0, src_x1 - src_x0)
    shifted[dst_y0:dst_y1, dst_x0:dst_x1] = mask[src_y0:src_y1, src_x0:src_x1]
    return shifted


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    image = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    size = max(3, radius * 2 + 1)
    return (np.asarray(image.filter(ImageFilter.MaxFilter(size=size))) > 0).astype(np.uint8)


def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
    image = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    size = max(3, radius * 2 + 1)
    return (np.asarray(image.filter(ImageFilter.MinFilter(size=size))) > 0).astype(np.uint8)


def _random_blob(height: int, width: int, rng: random.Random) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    center_x = rng.randint(int(width * 0.15), int(width * 0.85))
    center_y = rng.randint(int(height * 0.15), int(height * 0.85))
    radius_x = rng.randint(max(6, width // 40), max(12, width // 8))
    radius_y = rng.randint(max(6, height // 40), max(12, height // 8))
    blob = (((xx - center_x) / max(radius_x, 1)) ** 2 + ((yy - center_y) / max(radius_y, 1)) ** 2 <= 1.0).astype(np.uint8)
    if rng.random() < 0.5:
        blob = _dilate(blob, rng.randint(1, 6))
    return blob


def perturb_mask(mask: Image.Image, seed: int | None = None, allow_relocation: bool = True) -> Image.Image:
    rng = random.Random(seed)
    binary = (np.asarray(mask.convert("L"), dtype=np.uint8) > 0).astype(np.uint8)
    height, width = binary.shape

    if binary.sum() == 0:
        binary[height // 3 : height // 2, width // 3 : width // 2] = 1

    result = _dilate(binary, rng.randint(2, 8))
    if rng.random() < 0.5:
        result = _erode(result, rng.randint(1, 4))
    if allow_relocation:
        shift_x = rng.randint(-max(4, width // 20), max(4, width // 20))
        shift_y = rng.randint(-max(4, height // 20), max(4, height // 20))
        result = _shift_binary(result, shift_x, shift_y)
    if rng.random() < 0.8:
        result = np.maximum(result, _random_blob(height, width, rng))
    if rng.random() < 0.5:
        result = _dilate(result, rng.randint(1, 8))

    output = Image.fromarray((result * 255).astype(np.uint8), mode="L")
    output = output.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.8, 2.0)))
    output = output.point(lambda value: 255 if value > 64 else 0, mode="L")
    return output
