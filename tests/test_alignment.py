"""Bringing a reconstruction from its own frame and units into CT (patient LPS) coordinates."""
import numpy as np
from scipy.spatial.transform import Rotation

from facerecon.registration import apply_transform, face_frame_init, icp, icp_multistart
from test_registration import _height, similarity

WORLD_FROM_LPS = similarity(1 / 80, [40, -70, 25], [0.3, -0.2, 1.5])  # VGGT-like units and pose


def lps_face(n, seed, half=30.0):
    """Face-like height field looking anterior (-y in LPS), with outward normals."""
    x, z = np.random.default_rng(seed).uniform(-half, half, (2, n))
    pts = np.column_stack([x, -_height(x, z), z])
    eps = 1e-4
    gx = (_height(x + eps, z) - _height(x - eps, z)) / (2 * eps)
    gz = (_height(x, z + eps) - _height(x, z - eps)) / (2 * eps)
    nrm = np.column_stack([-gx, -np.ones_like(x), -gz])  # -grad(y + h(x, z)): anterior
    return pts, nrm / np.linalg.norm(nrm, axis=1, keepdims=True)


def parsed_face(n, seed, half=30.0):
    """lps_face plus face-parsing style masks: two eyes above, nose bump, mouth below."""
    pts, _ = lps_face(n, seed, half)
    x, z = pts[:, 0], pts[:, 2]
    eyes = (np.abs(np.abs(x - 5) - 15) < 5) & (np.abs(z - 14) < 3)
    nose = (x - 5) ** 2 + (z + 3) ** 2 < 36
    mouth = (np.abs(x - 5) < 10) & (np.abs(z + 20) < 3)
    return pts, eyes, mouth, nose


def _yawed(T, degrees, center):
    Y = np.eye(4)
    Y[:3, :3] = Rotation.from_euler("z", degrees, degrees=True).as_matrix()
    Y[:3, 3] = center - Y[:3, :3] @ center
    return Y @ T


def _rms_error(T, pred, truth):
    return np.sqrt(np.mean(np.sum((apply_transform(T, pred) - truth) ** 2, axis=1)))


def test_face_frame_init_orients_reconstruction_from_parsed_eyes_mouth_and_nose():
    truth, eyes, mouth, nose = parsed_face(3000, seed=1)
    pred = apply_transform(WORLD_FROM_LPS, truth)

    T = face_frame_init(pred, eyes, mouth, nose, truth)

    R_est = T[:3, :3] / np.cbrt(np.linalg.det(T[:3, :3]))
    R_true = np.linalg.inv(WORLD_FROM_LPS)[:3, :3] * (1 / 80)
    assert np.degrees(Rotation.from_matrix(R_est @ R_true.T).magnitude()) < 15
    assert np.median(np.linalg.norm(apply_transform(T, pred) - truth, axis=1)) < 5.0  # mm


def test_face_frame_init_then_multistart_icp_aligns_precisely():
    ct, ct_n = lps_face(40_000, seed=2)
    truth, eyes, mouth, nose = parsed_face(3000, seed=3, half=25.0)
    pred = apply_transform(WORLD_FROM_LPS, truth)

    result = icp_multistart(pred, ct, ct_n, face_frame_init(pred, eyes, mouth, nose, ct), trim=0.8)

    assert _rms_error(result.transform, pred, truth) < 0.3


def test_multistart_icp_recovers_from_a_badly_rotated_start():
    ct, ct_n = lps_face(40_000, seed=2)
    truth, _ = lps_face(3000, seed=3, half=25.0)
    pred = apply_transform(WORLD_FROM_LPS, truth)
    wrong = _yawed(np.linalg.inv(WORLD_FROM_LPS), 90, truth.mean(0))

    plain = icp(pred, ct, ct_n, init=wrong, trim=0.8)
    multi = icp_multistart(pred, ct, ct_n, init=wrong, trim=0.8)

    assert _rms_error(plain.transform, pred, truth) > 2.0  # the case is hard for plain ICP
    assert _rms_error(multi.transform, pred, truth) < 0.3


def test_similarity_icp_cannot_collapse_the_scale():
    # point-to-plane residuals shrink to zero if all points collapse onto one surface point
    ct, ct_n = lps_face(40_000, seed=2)
    truth, _ = lps_face(3000, seed=3, half=25.0)
    pred = apply_transform(WORLD_FROM_LPS, truth)
    init = _yawed(np.linalg.inv(WORLD_FROM_LPS), 45, truth.mean(0))

    result = icp(pred, ct, ct_n, init=init, trim=0.8, max_scale_change=1.5)

    scale = np.cbrt(np.linalg.det(result.transform[:3, :3]))
    assert 80 / 1.5 - 1e-6 <= scale <= 80 * 1.5 + 1e-6
