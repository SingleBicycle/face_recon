import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from facerecon.registration import apply_transform, icp, umeyama


def similarity(scale, rotvec_deg, translation):
    T = np.eye(4)
    T[:3, :3] = scale * Rotation.from_rotvec(np.deg2rad(rotvec_deg)).as_matrix()
    T[:3, 3] = translation
    return T


def _height(x, y):
    """Asymmetric 'face' height field (a nose-like bump and a cheek) so the fit is unique."""
    return (12 * np.exp(-((x - 5) ** 2 + (y + 3) ** 2) / 150)
            + 6 * np.exp(-((x + 15) ** 2 + (y - 10) ** 2) / 60)
            - 0.004 * (x**2 + 1.5 * y**2))


def face_patch(n, seed, half=30.0):
    x, y = np.random.default_rng(seed).uniform(-half, half, (2, n))
    return np.column_stack([x, y, _height(x, y)])


def patch_normals(points, eps=1e-4):
    x, y = points[:, 0], points[:, 1]
    gx = (_height(x + eps, y) - _height(x - eps, y)) / (2 * eps)
    gy = (_height(x, y + eps) - _height(x, y - eps)) / (2 * eps)
    n = np.column_stack([-gx, -gy, np.ones_like(x)])
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def test_umeyama_recovers_similarity_from_correspondences():
    src = np.random.default_rng(0).normal(size=(50, 3)) * 20
    T = similarity(1.3, [10, -20, 5], [5, -3, 40])
    np.testing.assert_allclose(umeyama(src, apply_transform(T, src)), T, atol=1e-9)


def test_umeyama_rigid_mode_keeps_unit_scale():
    src = np.random.default_rng(0).normal(size=(50, 3)) * 20
    T = umeyama(src, apply_transform(similarity(1.3, [10, -20, 5], [5, -3, 40]), src), with_scale=False)
    assert np.linalg.det(T[:3, :3]) == pytest.approx(1.0)


def test_icp_aligns_partial_noisy_scan_with_outliers():
    rng = np.random.default_rng(1)
    ct_surface = face_patch(40_000, seed=2)
    truth = face_patch(3_000, seed=3)
    truth = truth[truth[:, 0] > -15]  # partial view of the face
    truth = truth + rng.normal(scale=0.1, size=truth.shape)  # reconstruction noise
    pred_to_ct = similarity(1.15, [3, -5, 4], [4, -3, 6])
    pred = apply_transform(np.linalg.inv(pred_to_ct), truth)  # reconstruction: own frame and scale
    background = rng.uniform(-40, 40, (150, 3))

    result = icp(np.vstack([pred, background]), ct_surface, patch_normals(ct_surface), with_scale=True, trim=0.85)

    aligned = apply_transform(result.transform, pred)
    assert np.sqrt(np.mean(np.sum((aligned - truth) ** 2, axis=1))) < 0.3  # mm
    assert np.cbrt(np.linalg.det(result.transform[:3, :3])) == pytest.approx(1.15, rel=0.005)
