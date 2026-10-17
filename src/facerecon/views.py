"""Photos -> network-ready views."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps


def load_views(paths, size: int = 518) -> np.ndarray:
    """Photos as VGGT 'pad'-mode inputs: upright (EXIF orientation applied), longest side `size`,
    other side a multiple of 14, centred on a white square.

    Returns (S, size, size, 3) float32 in [0, 1]. Padding (rather than VGGT's default centre crop)
    keeps forehead and chin in portrait photos.
    """
    views = np.ones((len(paths), size, size, 3), np.float32)
    for i, path in enumerate(paths):
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
        w, h = img.size
        if w >= h:
            new_w, new_h = size, round(h * size / w / 14) * 14
        else:
            new_w, new_h = round(w * size / h / 14) * 14, size
        top, left = (size - new_h) // 2, (size - new_w) // 2
        views[i, top:top + new_h, left:left + new_w] = (
            np.asarray(img.resize((new_w, new_h), Image.Resampling.BICUBIC), np.float32) / 255.0)
    return views
