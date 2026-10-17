"""Load a CT DICOM series as a Hounsfield-unit volume plus its voxel -> patient (LPS, mm) affine.

Works for any slice orientation (axial, coronal reformats, tilted), since the geometry comes
from ImagePositionPatient / ImageOrientationPatient rather than an assumed axial stack.
"""
from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError


@dataclass
class CTVolume:
    hu: np.ndarray  # (K, R, C) float32 indexed by slice, row, column
    affine: np.ndarray  # 4x4 mapping (k, r, c, 1) -> patient LPS mm
    meta: dict = field(default_factory=dict)

    @property
    def spacing(self) -> np.ndarray:
        """Voxel size in mm along the slice, row and column axes."""
        return np.linalg.norm(self.affine[:3, :3], axis=0)

    def index_to_patient(self, kji) -> np.ndarray:
        return np.asarray(kji, float) @ self.affine[:3, :3].T + self.affine[:3, 3]


def find_dicom_series(root) -> dict[str, list[Path]]:
    """Group every DICOM file under `root` by SeriesInstanceUID; non-DICOM files are skipped."""
    series = defaultdict(list)
    for path in sorted(Path(root).rglob("*")):
        if not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=["SeriesInstanceUID"])
        except (InvalidDicomError, OSError):
            continue
        if "SeriesInstanceUID" in ds:
            series[str(ds.SeriesInstanceUID)].append(path)
    return dict(series)


def select_stack(datasets: list) -> tuple[list, dict]:
    """The largest group of distinct, same-orientation images in one series.

    Exports sometimes contain every file twice, or slip a reference image with another
    orientation into a reformatted series. Returns the kept datasets and what was dropped.
    """
    if len({str(ds.SeriesInstanceUID) for ds in datasets}) != 1:
        raise ValueError("files belong to more than one DICOM series")
    seen, unique = set(), []
    for ds in datasets:
        if str(ds.SOPInstanceUID) not in seen:
            seen.add(str(ds.SOPInstanceUID))
            unique.append(ds)
    groups = defaultdict(list)
    for ds in unique:
        if "ImageOrientationPatient" in ds and "ImagePositionPatient" in ds:
            iop = tuple(np.round(np.array(ds.ImageOrientationPatient, float), 3))
            groups[(iop, int(ds.Rows), int(ds.Columns))].append(ds)
    if not groups:
        raise ValueError("no images with slice geometry (screenshot or rendering?)")
    stack = max(groups.values(), key=len)
    return stack, {"duplicates": len(datasets) - len(unique), "other_images": len(unique) - len(stack)}


def load_ct_series(files) -> CTVolume:
    """Read one series, order slices along the slice normal and rescale to HU."""
    slices, dropped = select_stack([pydicom.dcmread(f) for f in files])
    iop = np.array(slices[0].ImageOrientationPatient, float)
    row_dir, col_dir = iop[:3], iop[3:]
    normal = np.cross(row_dir, col_dir)
    slices.sort(key=lambda s: float(np.dot(np.array(s.ImagePositionPatient, float), normal)))
    ipp = np.array([s.ImagePositionPatient for s in slices], float)

    if len(slices) > 1:
        step = (ipp[-1] - ipp[0]) / (len(slices) - 1)  # not necessarily along the normal (gantry tilt)
        gaps = np.diff(ipp @ normal)
        uniform = bool(np.allclose(gaps, gaps.mean(), rtol=1e-2, atol=1e-2))
        if not uniform:
            warnings.warn(f"non-uniform slice spacing ({gaps.min():.3f}-{gaps.max():.3f} mm)")
    else:
        step, uniform = normal * float(slices[0].get("SliceThickness", 1.0)), True

    row_spacing, col_spacing = (float(v) for v in slices[0].PixelSpacing)
    affine = np.eye(4)
    affine[:3, 0] = step
    affine[:3, 1] = row_spacing * col_dir
    affine[:3, 2] = col_spacing * row_dir
    affine[:3, 3] = ipp[0]

    hu = np.stack([s.pixel_array * float(s.get("RescaleSlope", 1.0)) + float(s.get("RescaleIntercept", 0.0))
                   for s in slices]).astype(np.float32)
    meta = {
        "n_slices": len(slices),
        **dropped,
        "uniform_spacing": uniform,
        "slice_thickness": float(slices[0].get("SliceThickness", np.nan) or np.nan),
        "series_description": str(slices[0].get("SeriesDescription", "")),
    }
    return CTVolume(hu=hu, affine=affine, meta=meta)
