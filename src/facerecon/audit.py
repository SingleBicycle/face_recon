"""Dataset audit: per-case CT and photo summaries plus de-identification checks.

Reports name identifying fields that are filled in, but never copy their values.
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom
from pydicom.multival import MultiValue

from .dicom_io import find_dicom_series, select_stack
from .photos import list_photos, photo_info

# Fields blanked by the DICOM basic de-identification profile (PS3.15 Annex E).
PHI_FIELDS = (
    "PatientBirthDate", "PatientAddress", "PatientTelephoneNumbers", "OtherPatientIDs",
    "OtherPatientNames", "PatientMotherBirthName", "InstitutionName", "InstitutionAddress",
    "InstitutionalDepartmentName", "ReferringPhysicianName", "PerformingPhysicianName",
    "OperatorsName", "PhysiciansOfRecord", "NameOfPhysiciansReadingStudy", "RequestingPhysician",
    "AccessionNumber", "StationName", "DeviceSerialNumber",
)
DATE_FIELDS = ("StudyDate", "SeriesDate", "AcquisitionDate", "ContentDate")
_CASE_DIR = re.compile(r"^(?:patient[\s_-]*)?(\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class Case:
    id: int
    dir: Path

    @property
    def slug(self) -> str:
        return f"patient_{self.id:02d}"

    @property
    def photo_dir(self) -> Path:
        return self.dir / "photos"


def list_cases(root) -> list[Case]:
    """Case folders named by number ("12") or "Patient 12", sorted by ID."""
    cases = [Case(int(m.group(1)), d) for d in Path(root).iterdir()
             if d.is_dir() and (m := _CASE_DIR.match(d.name))]
    return sorted(cases, key=lambda c: c.id)


def audit_dataset(root) -> list[dict]:
    return [audit_case(case) for case in list_cases(root)]


def audit_case(case: Case) -> dict:
    row, issues = {"case": case.id, "slug": case.slug}, []
    series = find_dicom_series(case.dir)
    if not series:
        issues.append("no CT series found")
    else:
        if len(series) > 1:
            issues.append(f"{len(series)} DICOM series found; summarising the largest")
        row.update(_ct_summary(case, max(series.values(), key=len), issues))
    row.update(_photo_summary(list_photos(case.photo_dir), issues))
    row["issues"] = issues
    return row


def write_audit(rows: list[dict], out_dir) -> None:
    """audit.json (everything, incl. per-photo details) and audit.csv (one row per case)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audit.json").write_text(json.dumps(rows, indent=2))
    columns = list(dict.fromkeys(k for row in rows for k in row if k != "photos"))
    with open(out_dir / "audit.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "; ".join(map(str, v)) if isinstance(v, list) else v for k, v in row.items()})


def _ct_summary(case: Case, files: list[Path], issues: list[str]) -> dict:
    headers = [pydicom.dcmread(f, stop_before_pixels=True) for f in files]
    try:
        stack, dropped = select_stack(headers)
    except ValueError as e:
        stack, dropped = [], {}
        issues.append(f"CT series unusable: {e}")
    if dropped.get("duplicates"):
        issues.append(f"{dropped['duplicates']} duplicate DICOM files ignored")
    if dropped.get("other_images"):
        issues.append(f"{dropped['other_images']} images with another orientation ignored")
    h = stack[0] if stack else headers[0]
    out = {
        "ct_series": files[0].parent.relative_to(case.dir).as_posix(),
        "ct_slices": len(stack),
        "ct_modality": _text(h.get("Modality")),
        "ct_image_type": _text(h.get("ImageType")),
        "ct_description": _text(h.get("SeriesDescription")),
        "ct_kernel": _text(h.get("ConvolutionKernel")),
        "ct_rows": int(h.get("Rows", 0)),
        "ct_cols": int(h.get("Columns", 0)),
        "ct_pixel_spacing_mm": [float(v) for v in h.get("PixelSpacing", [])],
        "ct_slice_thickness_mm": _float(h.get("SliceThickness")),
        "ct_orientation": None,
        "ct_slice_spacing_mm": None,
        "ct_extent_mm": None,
        "patient_age": _text(h.get("PatientAge")),
        "patient_sex": _text(h.get("PatientSex")),
        "study_year": _text(h.get("StudyDate"))[:4],
        "phi_fields": [k for k in PHI_FIELDS if _text(h.get(k))],
    }
    if stack:
        iop = np.array(h.ImageOrientationPatient, float)
        normal = np.cross(iop[:3], iop[3:])
        pos = np.sort([float(np.dot(np.array(x.ImagePositionPatient, float), normal)) for x in stack])
        axis = int(np.argmax(np.abs(normal)))
        out["ct_orientation"] = ("sagittal", "coronal", "axial")[axis] if abs(normal[axis]) > 0.9 else "oblique"
        out["ct_slice_spacing_mm"] = float(np.median(np.diff(pos))) if len(pos) > 1 else None
        out["ct_extent_mm"] = float(pos[-1] - pos[0])

    if [int(n) for n in re.findall(r"\d+", _text(h.get("PatientName")))] != [case.id]:
        issues.append("PatientName does not match the case folder")
    if out["phi_fields"]:
        issues.append("identifying DICOM fields not blank: " + ", ".join(out["phi_fields"]))
    for key in DATE_FIELDS:
        value = _text(h.get(key))
        if len(value) == 8 and value[4:] != "0101":
            issues.append(f"{key} is a full date, not truncated to the year")
    return out


def _photo_summary(photos: list[Path], issues: list[str]) -> dict:
    infos, unreadable = [], []
    for path in photos:
        try:
            infos.append(photo_info(path))
        except OSError:
            unreadable.append(path.name)
    dated = sum(i["has_datetime"] for i in infos)
    gps = sum(i["has_gps"] for i in infos)
    if not photos:
        issues.append("no photos found")
    if unreadable:
        issues.append("unreadable photos: " + ", ".join(unreadable))
    if dated:
        issues.append(f"{dated} photos carry EXIF dates")
    if gps:
        issues.append(f"{gps} photos carry EXIF GPS")
    return {
        "n_photos": len(photos),
        "photo_min_side": min((min(i["width"], i["height"]) for i in infos), default=None),
        "photo_max_side": max((max(i["width"], i["height"]) for i in infos), default=None),
        "photos_with_focal": sum(i["focal_length_mm"] is not None for i in infos),
        "photos_with_datetime": dated,
        "photos_with_gps": gps,
        "photo_cameras": sorted({" ".join(filter(None, (i["make"], i["model"]))) for i in infos} - {""}),
        "photo_sharpness_median": float(np.median([i["sharpness"] for i in infos])) if infos else None,
        "photos": infos,
    }


def _text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, MultiValue)):
        return "\\".join(str(v) for v in value)
    return str(value).strip()


def _float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
