"""Skin (air / soft-tissue boundary) surface extraction from CT."""
from __future__ import annotations

import numpy as np
import trimesh
from scipy import ndimage
from skimage import measure

from .dicom_io import CTVolume

SKIN_HU = -500.0  # half-way between air (-1000) and soft tissue (~0): the 50% partial-volume edge


def body_mask(ct: CTVolume, level: float = SKIN_HU) -> np.ndarray:
    """Largest connected region above `level`, with enclosed air filled in axial planes.

    Filling per axial plane (not in 3D) also closes airways that open to the outside through
    the nostrils, which a 3D hole fill would leave in place.
    """
    labels, n = ndimage.label(ct.hu > level)
    if n == 0:
        raise ValueError(f"no voxels above {level} HU")
    mask = labels == np.argmax(np.bincount(labels.ravel())[1:]) + 1
    return _fill_holes_in_planes(mask, _superior_axis(ct.affine))


def extract_skin_mesh(ct: CTVolume, level: float = SKIN_HU) -> trimesh.Trimesh:
    """Outer skin surface in patient coordinates (LPS, mm), normals pointing outward.

    Marching cubes runs on the HU values so the surface gets sub-voxel accuracy, but voxels are
    forced inside/outside according to `body_mask` so that internal air and detached objects
    (couch, head holder) produce no surface. Where the scan range or field of view cuts through
    the head the surface is left open rather than capped.
    """
    mask = body_mask(ct, level)
    field = ct.hu.astype(np.float32, copy=True)
    field[mask & (field <= level)] = level + 100.0
    field[~mask & (field > level)] = level - 100.0
    field[field == level] -= 0.01  # keep voxels off the iso-level to avoid zero-area triangles
    verts, faces, _, _ = measure.marching_cubes(field, level=level)
    verts = verts @ ct.affine[:3, :3].T + ct.affine[:3, 3]
    mesh = trimesh.Trimesh(verts, faces, process=False)
    _point_normals_outward(mesh, field, ct.affine, level)
    return max(mesh.split(only_watertight=False), key=lambda m: len(m.faces))


def crop_face_region(mesh: trimesh.Trimesh, radius: float = 90.0) -> trimesh.Trimesh:
    """Faces within `radius` mm of the most anterior skin point (the nose tip for a straight head)."""
    tip = mesh.vertices[np.argmin(mesh.vertices[:, 1])]  # LPS: anterior is -y
    keep = np.flatnonzero(np.linalg.norm(mesh.triangles_center - tip, axis=1) <= radius)
    return mesh.submesh([keep], append=True)


def _point_normals_outward(mesh: trimesh.Trimesh, field: np.ndarray, affine: np.ndarray, level: float):
    """Flip the winding if face normals point into the body, judged by probing the field
    half a millimetre along the normals (independent of marching-cubes winding conventions)."""
    idx = np.random.default_rng(0).choice(len(mesh.faces), size=min(2000, len(mesh.faces)), replace=False)
    probe = mesh.triangles_center[idx] + 0.5 * mesh.face_normals[idx]
    kji = np.linalg.solve(affine[:3, :3], (probe - affine[:3, 3]).T)
    if np.mean(ndimage.map_coordinates(field, kji, order=1, mode="nearest") > level) > 0.5:
        mesh.invert()


def _superior_axis(affine: np.ndarray) -> int:
    """Array axis most aligned with the patient superior-inferior (z) axis."""
    return int(np.argmax(np.abs(affine[2, :3]) / np.linalg.norm(affine[:3, :3], axis=0)))


def _fill_holes_in_planes(mask: np.ndarray, axis: int) -> np.ndarray:
    planes = np.moveaxis(mask.copy(), axis, 0)
    for i in range(planes.shape[0]):
        planes[i] = ndimage.binary_fill_holes(planes[i])
    return np.ascontiguousarray(np.moveaxis(planes, 0, axis))
