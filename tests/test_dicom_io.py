import shutil

import numpy as np
import pytest
from pydicom.uid import generate_uid

from facerecon.dicom_io import find_dicom_series, load_ct_series
from synthetic import make_grid, rotated_dirs, write_ct_series


def test_find_dicom_series_groups_by_series_and_skips_other_files(tmp_path):
    grid4, grid3 = make_grid((0, 0, 0), (4, 6, 5)), make_grid((0, 0, 0), (3, 6, 5))
    write_ct_series(tmp_path / "s1", np.full(grid4.shape, -1000.0), grid4)
    write_ct_series(tmp_path / "nested" / "s2", np.full(grid3.shape, -1000.0), grid3)
    (tmp_path / "notes.txt").write_text("not a dicom file")

    series = find_dicom_series(tmp_path)

    assert sorted(len(files) for files in series.values()) == [3, 4]


def test_load_ct_series_restores_slice_order_hu_and_patient_geometry(tmp_path):
    grid = make_grid((10, -20, 35), (7, 9, 11), pixel_spacing=(0.8, 0.6), slice_spacing=2.5,
                     **rotated_dirs(30))
    hu = np.random.default_rng(0).integers(-1000, 2000, grid.shape).astype(float)
    write_ct_series(tmp_path, hu, grid, shuffle_seed=1)

    ct = load_ct_series(sorted(tmp_path.iterdir()))

    np.testing.assert_allclose(ct.hu, hu)
    ijk = np.array([[0, 0, 0], [6, 8, 10], [3, 2, 7]])
    np.testing.assert_allclose(ct.index_to_patient(ijk), grid.positions()[tuple(ijk.T)], atol=1e-3)
    np.testing.assert_allclose(ct.spacing, (2.5, 0.8, 0.6), atol=1e-6)


def test_load_ct_series_ignores_files_exported_twice(tmp_path):
    grid = make_grid((0, 0, 0), (5, 6, 4))
    hu = np.random.default_rng(0).integers(-1000, 1000, grid.shape).astype(float)
    write_ct_series(tmp_path, hu, grid)
    for f in list(tmp_path.iterdir()):
        shutil.copy(f, tmp_path / f"copy_{f.name}")

    ct = load_ct_series(sorted(tmp_path.iterdir()))

    np.testing.assert_allclose(ct.hu, hu)
    np.testing.assert_allclose(ct.spacing[0], grid.slice_spacing)


def test_load_ct_series_ignores_stray_image_with_another_orientation(tmp_path):
    coronal = make_grid((0, 0, 0), (6, 5, 4), row_dir=(1, 0, 0), col_dir=(0, 0, -1))
    hu = np.random.default_rng(0).integers(-1000, 1000, coronal.shape).astype(float)
    uid = generate_uid()
    write_ct_series(tmp_path, hu, coronal, series_uid=uid)
    sagittal = make_grid((0, 0, 0), (1, 5, 4), row_dir=(0, 1, 0), col_dir=(0, 0, -1))
    write_ct_series(tmp_path, np.zeros(sagittal.shape), sagittal, series_uid=uid, prefix="REF")

    ct = load_ct_series(sorted(tmp_path.iterdir()))

    np.testing.assert_allclose(ct.hu, hu)


def test_jpeg_lossless_pixel_data_can_be_decoded():
    # the clinical exports use JPEG Lossless, Process 14 SV1 (1.2.840.10008.1.2.4.70)
    from pydicom.pixels.decoders import JPEGLosslessSV1Decoder
    assert JPEGLosslessSV1Decoder.is_available


def test_load_ct_series_refuses_to_mix_series(tmp_path):
    grid = make_grid((0, 0, 0), (3, 6, 5))
    write_ct_series(tmp_path / "a", np.full(grid.shape, -1000.0), grid)
    write_ct_series(tmp_path / "b", np.full(grid.shape, -1000.0), grid)

    with pytest.raises(ValueError, match="series"):
        load_ct_series(sorted(tmp_path.rglob("IM*")))
