"""Similarity / rigid alignment of a reconstructed face to the CT skin surface."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def apply_transform(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.asarray(points) @ T[:3, :3].T + T[:3, 3]


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> np.ndarray:
    """Least-squares 4x4 transform with dst ~ s R src + t (Umeyama 1991)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    d = np.ones(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:  # avoid a reflection
        d[2] = -1.0
    R = (U * d) @ Vt
    s = (S @ d) / xs.var(0).sum() if with_scale else 1.0
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = mu_d - s * R @ mu_s
    return T


@dataclass
class ICPResult:
    transform: np.ndarray  # 4x4, source -> target
    rmse: float  # point-to-plane residual over the kept correspondences of the last iteration, mm
    iterations: int


def icp(source: np.ndarray, target: np.ndarray, target_normals: np.ndarray, init: np.ndarray | None = None,
        with_scale: bool = True, max_iters: int = 100, trim: float = 0.9, tol: float = 1e-5,
        max_scale_change: float = 1.5) -> ICPResult:
    """Trimmed point-to-plane ICP, optionally estimating scale (similarity transform).

    Point-to-plane avoids the slow sliding and the scale shrinkage of point-to-point ICP on smooth
    surfaces. Each iteration ignores the worst (1 - trim) correspondences so that partial overlap
    (hair, neck, background in the reconstruction; back of the head in CT) does not bias the fit.
    Point-to-plane residuals can always be driven to zero by collapsing the source onto one surface
    point, so the scale is kept within a factor `max_scale_change` of the initial scale.
    Stops when every entry of the update matrix is within `tol` of the identity.
    """
    tree = cKDTree(target)
    T = np.eye(4) if init is None else np.array(init, float)
    s0 = _scale(T)
    for it in range(1, max_iters + 1):
        moved = apply_transform(T, source)
        dist, nn = tree.query(moved)
        keep = dist <= np.quantile(dist, trim)
        p, q, n = moved[keep], target[nn[keep]], target_normals[nn[keep]]
        rmse = float(np.sqrt(np.mean(np.einsum("ij,ij->i", p - q, n) ** 2)))
        s = _scale(T)
        bounds = (s0 / max_scale_change / s, s0 * max_scale_change / s) if with_scale else None
        step = _point_to_plane_step(p, q, n, bounds)
        T = step @ T
        if np.abs(step - np.eye(4)).max() < tol:
            break
    return ICPResult(transform=T, rmse=rmse, iterations=it)


def face_frame_init(points: np.ndarray, eyes: np.ndarray, mouth: np.ndarray, nose: np.ndarray,
                    ct_face_points: np.ndarray) -> np.ndarray:
    """Similarity guess mapping a reconstructed face into CT (LPS) coordinates.

    `eyes`, `mouth`, `nose` are boolean masks over `points` from face parsing. The anterior axis is
    the least-variance direction of the face cap, signed so that the nose protrudes; the superior
    axis points from the mouth to the eyes; left completes the right-handed LPS frame (so eye
    laterality is not needed). The CT head is assumed roughly aligned with the scanner axes.
    Scale comes from the RMS spread of both point sets, translation from the nose tips.
    """
    centre = points.mean(0)
    anterior = np.linalg.svd(points - centre, full_matrices=False)[2][-1]
    if np.dot(points[nose].mean(0) - centre, anterior) < 0:
        anterior = -anterior
    superior = points[eyes].mean(0) - points[mouth].mean(0)
    superior -= np.dot(superior, anterior) * anterior
    superior /= np.linalg.norm(superior)
    left = np.cross(-anterior, superior)
    R = np.stack([left, -anterior, superior])  # rows: the LPS axes in reconstruction coordinates
    rotated = points @ R.T
    s = _rms_radius(ct_face_points) / _rms_radius(rotated)
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = _anterior_tip(ct_face_points) - _anterior_tip(s * rotated)
    return T


def icp_multistart(source: np.ndarray, target: np.ndarray, target_normals: np.ndarray, init: np.ndarray,
                   yaw_deg=(-60, -30, 0, 30, 60), pitch_deg=(-40, -20, 0, 20, 40), trim: float = 0.8,
                   with_scale: bool = True, max_points: int = 5000, seed: int = 0) -> ICPResult:
    """Rigid ICP from rotated copies of `init` (yaw about the superior axis, pitch about the
    left-right axis, around the source centroid), then similarity ICP from the best start.

    Covers a rough initial orientation and a head flexed in the scanner; the starts are rigid so
    the scale cannot collapse.
    """
    sub = source
    if len(source) > max_points:
        sub = source[np.random.default_rng(seed).choice(len(source), max_points, replace=False)]
    c = apply_transform(init, sub).mean(0)
    best = None
    for yaw in yaw_deg:
        for pitch in pitch_deg:
            start = np.eye(4)
            start[:3, :3] = Rotation.from_euler("zx", [yaw, pitch], degrees=True).as_matrix()
            start[:3, 3] = c - start[:3, :3] @ c
            result = icp(sub, target, target_normals, init=start @ init, with_scale=False, trim=trim, max_iters=50)
            if best is None or result.rmse < best.rmse:
                best = result
    return icp(source, target, target_normals, init=best.transform, with_scale=with_scale, trim=trim)


def _scale(T: np.ndarray) -> float:
    return float(abs(np.linalg.det(T[:3, :3])) ** (1 / 3))


def _rms_radius(points: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((points - points.mean(0)) ** 2, axis=1))))


def _anterior_tip(points: np.ndarray, fraction: float = 0.002) -> np.ndarray:
    """Mean of the most anterior (smallest LPS y) points."""
    k = max(10, int(len(points) * fraction))
    return points[np.argpartition(points[:, 1], k)[:k]].mean(0)


def _point_to_plane_step(p, q, n, scale_bounds=None):
    """Gauss-Newton update minimising sum(((s R p + t) - q) . n)^2, linearised about the identity.

    Solved in coordinates centred on p so rotation and scale do not trade off against translation.
    `scale_bounds` (lo, hi) clips this step's scale factor; None keeps the scale fixed.
    """
    c = p.mean(0)
    pc = p - c
    cols = [np.cross(pc, n), n] + ([np.einsum("ij,ij->i", pc, n)] if scale_bounds else [])
    x, *_ = np.linalg.lstsq(np.column_stack(cols), -np.einsum("ij,ij->i", p - q, n), rcond=None)
    factor = float(np.clip(1.0 + x[6], *scale_bounds)) if scale_bounds else 1.0
    sR = factor * Rotation.from_rotvec(x[:3]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = sR
    T[:3, 3] = c + x[3:6] - sR @ c
    return T
