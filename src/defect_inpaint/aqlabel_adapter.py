from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

from PIL import Image


def _load_module(module_path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class AqLabelMaskAdapter:
    def __init__(self, generator_root: str | Path) -> None:
        self.generator_root = Path(generator_root).resolve()
        src_root = self.generator_root / "src"
        self.make_mask_module = _load_module(src_root / "make_mask.py", "mask_generator_make_mask")
        self.mask_maker_cls = self.make_mask_module.MaskMaker
        self.parse_aqlabel = self.make_mask_module.parse_aqlabel

    def create_mask(
        self,
        image_path: str | Path,
        label_path: str | Path,
        target_label: str = "ALL",
        label_match_mode: str = "exact",
    ) -> tuple[Image.Image, list[str]]:
        image_path = Path(image_path)
        label_path = Path(label_path)
        image = Image.open(image_path).convert("RGB")
        regions = self.parse_aqlabel(label_path, image.size)

        target = target_label.upper()
        selected = []
        for region in regions:
            if target == "ALL":
                selected.append(region)
            elif label_match_mode == "contains" and target_label in region.label:
                selected.append(region)
            elif label_match_mode == "exact" and region.label == target_label:
                selected.append(region)

        mask = self.mask_maker_cls.draw_regions_to_mask(image.size, selected)
        labels = sorted({region.label for region in selected})
        return mask, labels
