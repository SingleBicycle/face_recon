import csv
import json
import shutil

import numpy as np

from facerecon.audit import audit_dataset, list_cases, write_audit
from synthetic import make_grid, save_photo, write_ct_series

GRID = make_grid((0, 0, 0), (4, 6, 5))
AIR = np.full(GRID.shape, -1000.0)


def test_list_cases_accepts_numeric_and_patient_prefixed_folders(tmp_path):
    for name in ["10", "2", "Patient 1", "notes"]:
        (tmp_path / name).mkdir()
    (tmp_path / "readme.txt").write_text("x")
    cases = list_cases(tmp_path)
    assert [c.id for c in cases] == [1, 2, 10]
    assert [c.slug for c in cases] == ["patient_01", "patient_02", "patient_10"]


def test_audit_summarises_ct_and_photos_without_copying_identifying_values(tmp_path):
    write_ct_series(tmp_path / "1" / "2014_s6_Coronal", AIR, GRID, patient="Patient 1",
                    extra={"InstitutionName": "Some Hospital", "StudyDate": "20140101"})
    save_photo(tmp_path / "1" / "photos" / "1_photo_01.jpg")
    save_photo(tmp_path / "1" / "photos" / "1_photo_02.jpg", datetime="2014:03:02 10:00:00")

    [row] = audit_dataset(tmp_path)

    assert row["case"] == 1
    assert row["ct_series"] == "2014_s6_Coronal"
    assert row["ct_slices"] == 4
    assert row["ct_orientation"] == "axial"
    assert row["n_photos"] == 2 and row["photos_with_datetime"] == 1
    assert row["phi_fields"] == ["InstitutionName"]
    assert "Some Hospital" not in json.dumps(row)
    assert not any("PatientName" in issue or "StudyDate" in issue for issue in row["issues"])


def test_audit_flags_missing_ct_and_mismatched_patient_name(tmp_path):
    save_photo(tmp_path / "2" / "photos" / "2_photo_01.jpg")
    write_ct_series(tmp_path / "3" / "ct", AIR, GRID, patient="Patient 4")
    rows = {r["case"]: r for r in audit_dataset(tmp_path)}
    assert any("no CT" in issue for issue in rows[2]["issues"])
    assert any("PatientName" in issue for issue in rows[3]["issues"])


def test_audit_flags_dicom_dates_that_are_not_truncated_to_the_year(tmp_path):
    write_ct_series(tmp_path / "1" / "ct", AIR, GRID, patient="Patient 1", extra={"StudyDate": "20140615"})
    [row] = audit_dataset(tmp_path)
    assert any("StudyDate" in issue for issue in row["issues"])


def test_audit_counts_only_the_volume_stack_and_reports_what_was_dropped(tmp_path):
    ct_dir = write_ct_series(tmp_path / "1" / "ct", AIR, GRID, patient="Patient 1")
    for f in list(ct_dir.iterdir()):
        shutil.copy(f, ct_dir / f"copy_{f.name}")

    [row] = audit_dataset(tmp_path)

    assert row["ct_slices"] == GRID.shape[0]
    assert row["ct_slice_spacing_mm"] == GRID.slice_spacing
    assert any("duplicate" in issue for issue in row["issues"])


def test_write_audit_creates_csv_and_json(tmp_path):
    save_photo(tmp_path / "data" / "2" / "photos" / "x.jpg")
    write_audit(audit_dataset(tmp_path / "data"), tmp_path / "out")
    with open(tmp_path / "out" / "audit.csv") as f:
        assert next(csv.DictReader(f))["case"] == "2"
    assert json.loads((tmp_path / "out" / "audit.json").read_text())[0]["case"] == 2
