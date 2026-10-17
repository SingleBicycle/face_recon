"""Clinic photo listing and per-image QC: size, camera EXIF, privacy flags, sharpness."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".heic"}
_EXIF_IFD, _GPS_IFD = 0x8769, 0x8825
_DATETIME_TAGS = (0x0132, 0x9003, 0x9004)  # DateTime, DateTimeOriginal, DateTimeDigitized


def list_photos(folder) -> list[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in PHOTO_SUFFIXES)


def photo_info(path) -> dict:
    """Raw (pre-EXIF-rotation) size, camera info and QC values for one photo."""
    with Image.open(path) as img:
        exif = img.getexif()
        sub = exif.get_ifd(_EXIF_IFD)
        gray = img.convert("L")
        gray.thumbnail((1024, 1024))  # comparable sharpness across resolutions
        return {
            "name": Path(path).name,
            "width": img.width,
            "height": img.height,
            "orientation": int(exif.get(0x0112, 1)),
            "make": _text(exif.get(0x010F)),
            "model": _text(exif.get(0x0110)),
            "focal_length_mm": _number(sub.get(0x920A)),
            "focal_length_35mm": _number(sub.get(0xA405)),
            "has_datetime": any(t in exif or t in sub for t in _DATETIME_TAGS),
            "has_gps": bool(exif.get_ifd(_GPS_IFD)),
            "sharpness": float(ndimage.laplace(np.asarray(gray, float)).var()),  # variance of Laplacian
        }


def export_clean_photo(src, dst) -> Path:
    """Write `src` upright (EXIF orientation applied) as a JPEG without any metadata.

    The pixels are copied into a fresh image so that no EXIF / XMP / comment block
    (visit dates, GPS position, camera serial) can travel along.
    """
    with Image.open(src) as img:
        upright = ImageOps.exif_transpose(img).convert("RGB")
    clean = Image.frombytes("RGB", upright.size, upright.tobytes())
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    clean.save(dst, format="JPEG", quality=95)
    return dst


def _text(value) -> str | None:
    text = str(value).strip("\x00 ") if value is not None else ""
    return text or None


def _number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return number if math.isfinite(number) and number > 0 else None
