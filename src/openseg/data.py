"""Chargement des images et des masques de référence TACO.

Règles :
- TOUTE image est lue par load_image() : correction de l'orientation EXIF.
- Les polygones COCO sont convertis en masques booléens, un masque par annotation.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from pycocotools import mask as mask_utils


def load_image(path) -> Image.Image:
    """Ouvre une image en RGB, orientation EXIF appliquée."""
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")


def ann_to_mask(ann: dict, height: int, width: int) -> np.ndarray:
    """Annotation COCO -> masque booléen (height, width)."""
    seg = ann["segmentation"]
    if isinstance(seg, list):  # polygones
        polys = [p for p in seg if len(p) >= 6]  # un polygone valide a au moins 3 points
        if not polys:
            return np.zeros((height, width), dtype=bool)
        rle = mask_utils.merge(mask_utils.frPyObjects(polys, height, width))
    elif isinstance(seg.get("counts"), list):  # RLE non compressé
        rle = mask_utils.frPyObjects(seg, height, width)
    else:  # RLE compressé
        rle = seg
    return mask_utils.decode(rle).astype(bool)


class TacoSubset:
    """Une partie (dev ou test) du découpage : liste de paires (image, groupe)."""

    def __init__(self, root, split_path, categories_path, subset: str = "dev"):
        if subset not in ("dev", "test"):
            raise ValueError("subset doit valoir 'dev' ou 'test'")
        self.root = Path(root)
        coco = json.loads((self.root / "annotations.json").read_text())
        split = json.loads(Path(split_path).read_text())
        cfg = json.loads(Path(categories_path).read_text())["groups"]

        self.items = split[subset]
        self.prompts = {g: spec["prompt"] for g, spec in cfg.items()}

        name2id = {c["name"]: c["id"] for c in coco["categories"]}
        id2group = {name2id[n]: g for g, spec in cfg.items() for n in spec["taco_categories"]}
        self._images = {im["id"]: im for im in coco["images"]}

        self._anns = defaultdict(list)  # (image_id, groupe) -> annotations
        for ann in coco["annotations"]:
            g = id2group.get(ann["category_id"])
            if g is not None:
                self._anns[(ann["image_id"], g)].append(ann)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        row = self.items[i]
        im = self._images[row["image_id"]]
        image = load_image(self.root / im["file_name"])

        expected = (im["width"], im["height"])
        if image.size != expected:
            raise ValueError(
                f"{im['file_name']} : taille {image.size} != annotation {expected}"
            )

        anns = self._anns[(row["image_id"], row["group"])]
        if anns:
            masks = np.stack([ann_to_mask(a, im["height"], im["width"]) for a in anns])
        else:
            masks = np.zeros((0, im["height"], im["width"]), dtype=bool)

        return {
            "image_id": row["image_id"],
            "file_name": im["file_name"],
            "group": row["group"],
            "prompt": self.prompts[row["group"]],
            "image": image,
            "gt_masks": masks,
            "gt_areas": [a["area"] for a in anns],
            "n_instances": row["n_instances"],
        }
