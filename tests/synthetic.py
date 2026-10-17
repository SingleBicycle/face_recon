"""Synthetic CT phantoms, DICOM series and photos used by the tests."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, TiffImagePlugin
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

CENTER = np.array([10.0, -20.0, 35.0])  # ellipsoid "head" centre, patient LPS mm
AXES = np.array([30.0, 40.0, 50.0])  # semi-axes along L, P, S


@dataclass
class Grid:
    """CT voxel grid described the way DICOM does: slice k, row r, column c."""

    shape: tuple[int, int, int]
    origin: np.ndarray  # ImagePositionPatient of slice 0
    row_dir: np.ndarray  # IOP[:3], direction of increasing column index
    col_dir: np.ndarray  # IOP[3:], direction of increasing row index
    pixel_spacing: tuple[float, float]  # (between rows, between columns)
    slice_spacing: float

    @property
    def normal(self) -> np.ndarray:
        return np.cross(self.row_dir, self.col_dir)

    @property
    def affine(self) -> np.ndarray:
        a = np.eye(4)
        a[:3, 0] = self.slice_spacing * self.normal
        a[:3, 1] = self.pixel_spacing[0] * self.col_dir
        a[:3, 2] = self.pixel_spacing[1] * self.row_dir
        a[:3, 3] = self.origin
        return a

    def positions(self) -> np.ndarray:
        """Patient coordinates of every voxel centre, shape (K, R, C, 3)."""
        kji = np.stack(np.meshgrid(*map(np.arange, self.shape), indexing="ij"), -1)
        return kji @ self.affine[:3, :3].T + self.affine[:3, 3]


def make_grid(center, shape, pixel_spacing=(1.0, 1.0), slice_spacing=2.0,
              row_dir=(1, 0, 0), col_dir=(0, 1, 0)) -> Grid:
    """Grid whose central voxel sits at `center`."""
    g = Grid(tuple(shape), np.zeros(3), np.asarray(row_dir, float), np.asarray(col_dir, float),
             tuple(pixel_spacing), float(slice_spacing))
    g.origin = np.asarray(center, float) - g.affine[:3, :3] @ ((np.asarray(shape) - 1) / 2.0)
    return g


def rotated_dirs(angle_deg: float) -> dict:
    """Axial slices rotated in-plane by `angle_deg`."""
    a = np.deg2rad(angle_deg)
    return {"row_dir": (np.cos(a), np.sin(a), 0.0), "col_dir": (-np.sin(a), np.cos(a), 0.0)}


def ellipsoid_sdf(p, center=CENTER, axes=AXES):
    """First-order signed distance (mm) to an ellipsoid; negative inside."""
    d = np.asarray(p) - center
    q = np.linalg.norm(d / axes, axis=-1)
    grad = np.linalg.norm(d / axes**2, axis=-1) / np.maximum(q, 1e-9)
    return (q - 1.0) / np.maximum(grad, 1e-9)


def _ramp(sd, width):
    """1 inside, 0 outside, linear over `width` mm centred on the surface (partial volume)."""
    return np.clip(0.5 - sd / width, 0.0, 1.0)


def head_phantom(grid: Grid, cavity_radius=6.0, couch_gap=8.0, tube_radius=None, width=4.0):
    """HU volume: soft-tissue ellipsoid with an internal air cavity and a detached couch slab.

    `tube_radius` adds an airway running from the centre down through the bottom of the head,
    open to the outside like a nostril.
    """
    p = grid.positions()
    tissue = _ramp(ellipsoid_sdf(p), width)
    tissue *= 1.0 - _ramp(np.linalg.norm(p - CENTER, axis=-1) - cavity_radius, width)
    if tube_radius:
        radial = np.linalg.norm(p[..., :2] - CENTER[:2], axis=-1)
        tissue *= 1.0 - (p[..., 2] < CENTER[2]) * _ramp(radial - tube_radius, width)
    hu = -1000.0 + 1040.0 * tissue
    if couch_gap is not None:
        y0 = CENTER[1] + AXES[1] + couch_gap  # couch is posterior (+y) of a supine patient
        hu[(p[..., 1] > y0) & (p[..., 1] < y0 + 6.0)] = 300.0
    return hu


def _ds(v) -> str:
    return f"{float(v):.6f}"


def write_ct_series(out_dir: Path, hu: np.ndarray, grid: Grid, *, patient="Patient 1",
                    shuffle_seed=0, series_uid: str | None = None, prefix="IM", extra: dict | None = None) -> Path:
    """Write `hu` (K, R, C) as a CT series, one file per slice, in shuffled order."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    study, series, frame = generate_uid(), series_uid or generate_uid(), generate_uid()
    order = np.random.default_rng(shuffle_seed).permutation(grid.shape[0])
    for n, k in enumerate(order):
        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.MediaStorageSOPClassUID = CTImageStorage
        ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
        ds.Modality = "CT"
        ds.PatientName = patient
        ds.PatientID = patient
        ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = study, series, frame
        ds.InstanceNumber = int(k) + 1
        ds.ImagePositionPatient = [_ds(v) for v in grid.origin + k * grid.slice_spacing * grid.normal]
        ds.ImageOrientationPatient = [_ds(v) for v in (*grid.row_dir, *grid.col_dir)]
        ds.PixelSpacing = [_ds(v) for v in grid.pixel_spacing]
        ds.SliceThickness = _ds(grid.slice_spacing)
        ds.Rows, ds.Columns = int(grid.shape[1]), int(grid.shape[2])
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
        ds.RescaleIntercept, ds.RescaleSlope = "-1024", "1"
        ds.PixelData = np.round(hu[k] + 1024.0).astype("<i2").tobytes()
        for key, value in (extra or {}).items():
            setattr(ds, key, value)
        ds.save_as(out_dir / f"{prefix}{n:04d}", enforce_file_format=True)
    return out_dir


def checkerboard(h=480, w=640, square=16) -> np.ndarray:
    yy, xx = np.mgrid[:h, :w]
    return ((((yy // square) + (xx // square)) % 2) * 255).astype(np.uint8)


def save_photo(path: Path, array=None, *, datetime=None, focal_mm=None, model="TestCam",
               gps=False, orientation=None) -> Path:
    """Save a JPEG/PNG; JPEGs get EXIF with the camera model and optional date / focal length /
    GPS position / orientation tag."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(checkerboard(48, 64) if array is None else array)
    if path.suffix.lower() == ".png":
        img.save(path)
        return path
    exif = Image.Exif()
    exif[0x0110] = model  # Model
    sub = exif.get_ifd(0x8769)  # Exif sub-IFD
    if datetime:
        sub[0x9003] = datetime  # DateTimeOriginal
    if focal_mm:
        sub[0x920A] = TiffImagePlugin.IFDRational(int(round(focal_mm * 100)), 100)  # FocalLength
    if gps:
        geo = exif.get_ifd(0x8825)
        geo[0x0001] = "N"  # GPSLatitudeRef
        geo[0x0002] = tuple(TiffImagePlugin.IFDRational(v, 1) for v in (30, 17, 0))  # GPSLatitude
    if orientation:
        exif[0x0112] = orientation
    img.save(path, exif=exif, quality=95)
    return path
