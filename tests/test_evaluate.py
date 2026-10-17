import numpy as np
import pytest
import trimesh

from facerecon.evaluate import evaluate_reconstruction
from facerecon.registration import apply_transform
from test_registration import _height, similarity


def _face_mesh(half=30.0, step=0.5):
    x, y = np.meshgrid(np.arange(-half, half + step, step), np.arange(-half, half + step, step))
    verts = np.column_stack([x.ravel(), y.ravel(), _height(x.ravel(), y.ravel())])
    n = x.shape[1]
    i = (np.arange(x.shape[0] - 1)[:, None] * n + np.arange(n - 1)).ravel()
    faces = np.vstack([np.column_stack([i, i + 1, i + n]), np.column_stack([i + 1, i + n + 1, i + n])])
    return trimesh.Trimesh(verts, faces, process=False)  # normals point +z ("outward")


def test_evaluation_recovers_alignment_and_reports_noise_level_error():
    rng = np.random.default_rng(0)
    ct = _face_mesh()
    x, y = rng.uniform(-20, 25, (2, 4000))
    truth = np.column_stack([x, y, _height(x, y)]) + rng.normal(scale=0.2, size=(4000, 3))
    pred_to_ct = similarity(80.0, [5, 10, -5], [3, 2, 1])  # e.g. VGGT output in normalised units
    pred = apply_transform(np.linalg.inv(pred_to_ct), truth)
    init = similarity(1.0, [2, -2, 1], [1.5, -1, 1]) @ pred_to_ct  # rough landmark-based guess

    result = evaluate_reconstruction(pred, ct, init=init)

    assert result["scale"] == pytest.approx(80.0, rel=0.01)
    assert result["summary"]["median_abs"] < 0.2  # ~ the 0.2 mm noise, not misalignment
    assert abs(result["summary"]["mean_signed"]) < 0.05
    np.testing.assert_allclose(result["aligned"], apply_transform(result["transform"], pred))
    assert result["distances"].shape == (4000,)
