"""Face parsing: which pixels are facial skin and features."""
from __future__ import annotations

import numpy as np

# Label ids of jonathandinu/face-parsing (SegFormer trained on CelebAMask-HQ).
NOSE = 2
EYES = (4, 5)
MOUTH = (10, 11, 12)  # mouth, upper lip, lower lip
FACE_LABELS = (1, NOSE, *EYES, 6, 7, *MOUTH)  # skin, nose, eyes, brows, mouth, lips; not hair/ears/neck


def face_mask(labels: np.ndarray) -> np.ndarray:
    return np.isin(labels, FACE_LABELS)


class FaceParser:
    """SegFormer face parser; returns a label map per view at the view's resolution."""

    def __init__(self, device: str = "cuda", weights: str = "jonathandinu/face-parsing"):
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        self.processor = SegformerImageProcessor.from_pretrained(weights)
        self.model = SegformerForSemanticSegmentation.from_pretrained(weights).to(device).eval()
        self.device = device

    def __call__(self, views: np.ndarray) -> np.ndarray:
        import torch

        images = list(np.round(views * 255).astype(np.uint8))
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            logits = torch.nn.functional.interpolate(logits, size=views.shape[1:3], mode="bilinear",
                                                     align_corners=False)
        return logits.argmax(1).cpu().numpy()
