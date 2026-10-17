#!/usr/bin/env python
"""Stage 2 baseline end to end: clean photos -> VGGT face points -> aligned to CT -> surface error.

Alignment uses no manual landmarks: face-parsing labels on the reconstructed points (eyes, mouth,
nose) give the orientation, the CT face region gives scale and nose-tip position, then multi-start
rigid ICP and a final similarity ICP. Error statistics cover reconstructed points within --radius
mm of the CT nose tip.

Writes <out>/<case>/{face_error.ply, recon.npz, metrics.json} and <out>/summary.csv.
Outputs contain facial geometry: keep them on the protected data volume.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import trimesh

from facerecon.ct_surface import crop_face_region
from facerecon.face_parsing import EYES, MOUTH, NOSE, FaceParser, face_mask
from facerecon.metrics import error_colors, point_to_mesh_distance, summarize
from facerecon.photos import list_photos
from facerecon.recon_vggt import VGGTReconstructor, confident_pixels
from facerecon.registration import apply_transform, face_frame_init, icp_multistart
from facerecon.views import load_views


def run_case(photo_dir, ct_path, out_dir, vggt, parser, radius, keep_quantile, max_eval_points=50_000):
    views = load_views(list_photos(photo_dir))
    recon = vggt(views)
    labels = parser(views)
    keep = confident_pixels(recon["conf"], face_mask(labels), keep_quantile)
    xyz, rgb, lab = recon["points"][keep], views[keep], labels[keep]
    eyes, mouth, nose = np.isin(lab, EYES), np.isin(lab, MOUTH), lab == NOSE
    if not (eyes.any() and mouth.any() and nose.any()):
        raise RuntimeError("face parsing found no eyes, mouth or nose")

    ct = trimesh.load(ct_path, process=False)
    face = crop_face_region(ct, radius)
    ct_pts, idx = trimesh.sample.sample_surface(face, 100_000, seed=0)
    init = face_frame_init(xyz, eyes, mouth, nose, ct_pts)
    reg = icp_multistart(xyz, ct_pts, face.face_normals[idx], init, trim=0.8)
    aligned = apply_transform(reg.transform, xyz)

    tip = face.vertices[np.argmin(face.vertices[:, 1])]
    region = np.flatnonzero(np.linalg.norm(aligned - tip, axis=1) <= radius)
    if len(region) > max_eval_points:
        region = np.random.default_rng(0).choice(region, max_eval_points, replace=False)
    d = point_to_mesh_distance(aligned[region], crop_face_region(ct, radius + 20.0))

    out_dir.mkdir(parents=True, exist_ok=True)
    trimesh.PointCloud(aligned[region], colors=error_colors(d)).export(out_dir / "face_error.ply")
    np.savez_compressed(out_dir / "recon.npz", points=xyz, colors=rgb, labels=lab, transform=reg.transform,
                        extrinsic=recon["extrinsic"], intrinsic=recon["intrinsic"])
    metrics = {
        "n_views": len(views),
        "vggt_focal_px": float(np.median(recon["intrinsic"][:, 0, 0])),  # at 518 px; the 100 mm lens is ~2323
        "n_points": len(xyz),
        "n_eval_points": len(region),
        "scale": float(np.cbrt(np.linalg.det(reg.transform[:3, :3]))),
        "icp_rmse": reg.rmse,
        **summarize(d),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photos_root", type=Path, help="clean photos, one folder per case (clean_photos.py)")
    ap.add_argument("ct_root", type=Path, help="CT skin meshes (ct_to_skin_mesh.py)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cases", nargs="*", help="case slugs, e.g. patient_01 (default: all)")
    ap.add_argument("--radius", type=float, default=90.0, help="face region around the CT nose tip, mm")
    ap.add_argument("--keep-quantile", type=float, default=0.5, help="per-view VGGT confidence cut")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    vggt, parser = VGGTReconstructor(args.device), FaceParser(args.device)
    for photo_dir in sorted(p for p in args.photos_root.iterdir() if p.is_dir()):
        if args.cases and photo_dir.name not in args.cases:
            continue
        ct_path = args.ct_root / f"{photo_dir.name}_skin.ply"
        if not ct_path.exists():
            print(f"{photo_dir.name}: no CT skin mesh, skipped")
            continue
        try:
            m = run_case(photo_dir, ct_path, args.out / photo_dir.name, vggt, parser, args.radius,
                         args.keep_quantile)
        except RuntimeError as e:
            print(f"{photo_dir.name}: FAILED ({e})", flush=True)
            continue
        print(f"{photo_dir.name}: median {m['median_abs']:.2f} mm, mean {m['mean_abs']:.2f}, "
              f"P95 {m['p95_abs']:.2f}, within 2 mm {100 * m['within_2mm']:.0f}%, "
              f"signed mean {m['mean_signed']:+.2f}, ICP rmse {m['icp_rmse']:.2f}, views {m['n_views']}", flush=True)
    write_summary(args.out)


def write_summary(out: Path):
    """summary.csv over every case folder in `out` (so parallel runs into one folder combine)."""
    rows = [{"case": p.parent.name, **json.loads(p.read_text())} for p in sorted(out.glob("*/metrics.json"))]
    if rows:
        with open(out / "summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
