"""Pipeline open-vocabulary : texte -> boîtes (Grounding DINO) -> masques (SAM 2)."""
from __future__ import annotations

import inspect

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    Sam2Model,
    Sam2Processor,
)


def normalize_prompt(text: str) -> str:
    """Grounding DINO attend du texte en minuscules, avec un point final."""
    text = text.strip().lower()
    if not text.endswith("."):
        text += "."
    return text


class OpenVocabSegmenter:
    def __init__(
        self,
        detector_id: str = "IDEA-Research/grounding-dino-tiny",
        sam_id: str = "facebook/sam2.1-hiera-small",
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.det_processor = AutoProcessor.from_pretrained(detector_id)
        self.detector = (
            AutoModelForZeroShotObjectDetection.from_pretrained(detector_id)
            .to(self.device)
            .eval()
        )
        self.sam_processor = Sam2Processor.from_pretrained(sam_id)
        self.sam = Sam2Model.from_pretrained(sam_id).to(self.device).eval()

        # Le nom du seuil de boîte dépend de la version de transformers
        params = inspect.signature(
            self.det_processor.post_process_grounded_object_detection
        ).parameters
        self._box_key = "threshold" if "threshold" in params else "box_threshold"
        self._has_text_threshold = "text_threshold" in params

    @torch.no_grad()
    def detect(self, image: Image.Image, text: str,
               box_threshold: float = 0.3, text_threshold: float = 0.3):
        """Grounding DINO : renvoie boîtes (N,4), scores (N,), labels (liste)."""
        inputs = self.det_processor(
            images=image, text=normalize_prompt(text), return_tensors="pt"
        ).to(self.device)
        outputs = self.detector(**inputs)

        kwargs = {"target_sizes": [image.size[::-1]], self._box_key: box_threshold}
        if self._has_text_threshold:
            kwargs["text_threshold"] = text_threshold

        res = self.det_processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, **kwargs
        )[0]
        boxes = res["boxes"].cpu().numpy()
        scores = res["scores"].cpu().numpy()
        labels = list(res["text_labels"])
        return boxes, scores, labels

    @torch.no_grad()
    def segment_boxes(self, image: Image.Image, boxes: np.ndarray) -> np.ndarray:
        """SAM 2 : renvoie des masques booléens de forme (N, H, W)."""
        w, h = image.size
        if len(boxes) == 0:
            return np.zeros((0, h, w), dtype=bool)

        inputs = self.sam_processor(
            images=image, input_boxes=[boxes.tolist()], return_tensors="pt"
        ).to(self.device)
        outputs = self.sam(**inputs, multimask_output=False)
        masks = self.sam_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"]
        )[0]
        masks = masks[:, 0] if masks.ndim == 4 else masks
        if masks.dtype != torch.bool:
            masks = masks > 0
        return masks.numpy()

    def __call__(self, image: Image.Image, text: str,
                 box_threshold: float = 0.3, text_threshold: float = 0.3) -> dict:
        image = image.convert("RGB")
        boxes, scores, labels = self.detect(image, text, box_threshold, text_threshold)
        masks = self.segment_boxes(image, boxes)
        return {"boxes": boxes, "scores": scores, "labels": labels, "masks": masks}
