"""Surface error metrics between a reconstruction and the CT reference surface."""
from __future__ import annotations

import numpy as np
import trimesh


def point_to_mesh_distance(points: np.ndarray, mesh: trimesh.Trimesh, signed: bool = True) -> np.ndarray:
    """Distance (mm) from each point to the closest point on `mesh`.

    Signed: positive outside the reference surface (along its outward normals), negative inside.
    """
    closest, dist, tri = trimesh.proximity.closest_point(mesh, points)
    if signed:
        side = np.einsum("ij,ij->i", np.asarray(points) - closest, mesh.face_normals[tri])
        dist = np.where(side < 0, -dist, dist)
    return dist


def summarize(d: np.ndarray) -> dict:
    d = np.asarray(d, float)
    a = np.abs(d)
    return {
        "n": int(d.size),
        "mean_abs": float(a.mean()),
        "median_abs": float(np.median(a)),
        "rmse": float(np.sqrt(np.mean(d**2))),
        "p90_abs": float(np.percentile(a, 90)),
        "p95_abs": float(np.percentile(a, 95)),
        "max_abs": float(a.max()),
        "mean_signed": float(d.mean()),
        "within_1mm": float(np.mean(a <= 1.0)),
        "within_2mm": float(np.mean(a <= 2.0)),
    }


def error_colors(d: np.ndarray, vmax: float = 5.0) -> np.ndarray:
    """RGBA uint8 heatmap: blue (inside, -vmax) -> white (0) -> red (outside, +vmax)."""
    t = np.clip(np.asarray(d, float) / vmax, -1.0, 1.0)
    rgb = np.ones((t.size, 3))
    rgb[t > 0, 1:] -= t[t > 0, None]
    rgb[t < 0, :2] += t[t < 0, None]
    return np.round(np.column_stack([rgb, np.ones(t.size)]) * 255).astype(np.uint8)
