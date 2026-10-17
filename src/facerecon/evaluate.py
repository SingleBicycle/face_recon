"""Evaluate a reconstructed facial surface against the CT skin surface."""
from __future__ import annotations

import numpy as np
import trimesh

from .metrics import point_to_mesh_distance, summarize
from .registration import apply_transform, icp


def evaluate_reconstruction(pred_points: np.ndarray, ct_mesh: trimesh.Trimesh, init: np.ndarray | None = None,
                            with_scale: bool = True, trim: float = 0.9, n_ct_samples: int = 200_000,
                            seed: int = 0) -> dict:
    """Align predicted surface points to the CT skin (similarity ICP starting from `init`) and
    measure the remaining point-to-surface error in mm (positive = reconstruction outside the skin)."""
    ct_points, faces = trimesh.sample.sample_surface(ct_mesh, n_ct_samples, seed=seed)
    reg = icp(pred_points, ct_points, ct_mesh.face_normals[faces], init=init, with_scale=with_scale, trim=trim)
    aligned = apply_transform(reg.transform, pred_points)
    distances = point_to_mesh_distance(aligned, ct_mesh)
    return {
        "transform": reg.transform,
        "scale": float(np.cbrt(np.linalg.det(reg.transform[:3, :3]))),
        "icp_rmse": reg.rmse,
        "icp_iterations": reg.iterations,
        "aligned": aligned,
        "distances": distances,
        "summary": summarize(distances),
    }
