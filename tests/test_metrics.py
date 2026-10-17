import numpy as np
import pytest
import trimesh

from facerecon.metrics import error_colors, point_to_mesh_distance, summarize


def _points_at_radius(r, n=500, seed=0):
    v = np.random.default_rng(seed).normal(size=(n, 3))
    return r * v / np.linalg.norm(v, axis=1, keepdims=True)


def test_signed_distance_is_positive_outside_and_negative_inside():
    sphere = trimesh.creation.icosphere(subdivisions=5, radius=50.0)
    np.testing.assert_allclose(point_to_mesh_distance(_points_at_radius(52.0), sphere), 2.0, atol=0.02)
    np.testing.assert_allclose(point_to_mesh_distance(_points_at_radius(47.0), sphere), -3.0, atol=0.02)


def test_summarize_reports_absolute_and_signed_statistics():
    s = summarize(np.array([-1.0, 1.0, 2.0, -4.0]))
    assert s["n"] == 4
    assert s["mean_abs"] == pytest.approx(2.0)
    assert s["median_abs"] == pytest.approx(1.5)
    assert s["rmse"] == pytest.approx(np.sqrt(22 / 4))
    assert s["mean_signed"] == pytest.approx(-0.5)
    assert s["within_2mm"] == pytest.approx(0.75)


def test_error_colors_show_outside_red_inside_blue_and_zero_white():
    c = error_colors(np.array([5.0, -5.0, 0.0]), vmax=5.0)
    assert c.shape == (3, 4) and c.dtype == np.uint8
    np.testing.assert_array_equal(c[:, :3], [[255, 0, 0], [0, 0, 255], [255, 255, 255]])
