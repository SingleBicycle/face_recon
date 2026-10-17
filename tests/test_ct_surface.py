import numpy as np
import pytest

from facerecon.ct_surface import crop_face_region, extract_skin_mesh
from facerecon.dicom_io import CTVolume
from synthetic import AXES, ellipsoid_sdf, head_phantom, make_grid, rotated_dirs

HEAD_GRID = dict(center=(10, -15, 35), shape=(61, 120, 90))  # 1 x 1 mm pixels, 2 mm slices


def _skin(grid, **phantom_kw):
    return extract_skin_mesh(CTVolume(hu=head_phantom(grid, **phantom_kw), affine=grid.affine))


@pytest.fixture(scope="module")
def head_mesh():
    return _skin(make_grid(**HEAD_GRID))


def test_skin_surface_lies_on_the_true_surface_with_subvoxel_accuracy(head_mesh):
    err = np.abs(ellipsoid_sdf(head_mesh.vertices))
    assert np.median(err) < 0.1
    assert err.max() < 0.5


def test_skin_mesh_is_closed_with_outward_normals(head_mesh):
    assert head_mesh.is_watertight
    assert head_mesh.volume == pytest.approx(4 / 3 * np.pi * AXES.prod(), rel=0.01)  # > 0 => outward


def test_internal_air_and_detached_couch_are_excluded(head_mesh):
    assert len(head_mesh.split(only_watertight=False)) == 1
    assert head_mesh.vertices[:, 1].max() < -20 + AXES[1] + 1.0  # nothing at the couch


def test_rotated_slice_orientation_keeps_patient_geometry():
    mesh = _skin(make_grid(**HEAD_GRID, **rotated_dirs(30)), couch_gap=None)
    assert np.abs(ellipsoid_sdf(mesh.vertices)).max() < 0.5


def test_coronal_series_still_closes_airway_in_axial_planes():
    # coronal reformat: slices step anterior -> posterior, rows run superior -> inferior
    grid = make_grid((10, -20, 35), (45, 120, 90), row_dir=(1, 0, 0), col_dir=(0, 0, -1))
    mesh = _skin(grid, tube_radius=3.0, couch_gap=None)
    assert ellipsoid_sdf(mesh.vertices).min() > -3.0  # no surface deep inside the head


def test_face_region_is_cropped_around_the_most_anterior_point(head_mesh):
    face = crop_face_region(head_mesh, radius=30.0)
    tip = head_mesh.vertices[np.argmin(head_mesh.vertices[:, 1])]
    assert np.linalg.norm(face.vertices - tip, axis=1).max() <= 30.0 + head_mesh.edges_unique_length.max()
    assert 0 < len(face.faces) < len(head_mesh.faces)
    assert np.isclose(face.vertices[:, 1].min(), tip[1])


def test_scan_range_cutting_through_head_leaves_surface_open():
    mesh = _skin(make_grid((10, -15, 35), (31, 120, 90)))  # covers z = 5..65 mm only
    assert not mesh.is_watertight
    z = mesh.vertices[:, 2]
    assert z.min() >= 5 - 1e-6 and z.max() <= 65 + 1e-6
