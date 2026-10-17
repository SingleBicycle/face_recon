#!/usr/bin/env python
"""Stages 4-5: align a reconstructed face to the CT skin surface and report the surface error.

The initial alignment comes from --init: a 4x4 matrix (.txt/.npy) mapping the reconstruction
into CT coordinates, or a landmark file (.json) {"pred": {name: [x,y,z]}, "ct": {name: [x,y,z]}}
with at least 3 shared names (e.g. nasion, pronasale, exocanthions, pogonion).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from facerecon.evaluate import evaluate_reconstruction
from facerecon.metrics import error_colors
from facerecon.registration import umeyama


def load_init(path: Path | None, with_scale: bool):
    if path is None:
        return None
    if path.suffix == ".json":
        lm = json.loads(path.read_text())
        names = sorted(set(lm["pred"]) & set(lm["ct"]))
        if len(names) < 3:
            raise SystemExit(f"need >= 3 shared landmarks, got {names}")
        return umeyama(np.array([lm["pred"][n] for n in names], float),
                       np.array([lm["ct"][n] for n in names], float), with_scale)
    return np.load(path) if path.suffix == ".npy" else np.loadtxt(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pred", type=Path, help="reconstruction: mesh or point cloud (.ply/.obj)")
    ap.add_argument("ct", type=Path, help="CT skin mesh from ct_to_skin_mesh.py")
    ap.add_argument("--init", type=Path, help="initial alignment (.txt/.npy 4x4 or landmark .json)")
    ap.add_argument("--rigid", action="store_true", help="do not estimate scale")
    ap.add_argument("--trim", type=float, default=0.9, help="fraction of correspondences kept by ICP")
    ap.add_argument("--vmax", type=float, default=5.0, help="heatmap range in mm")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    pred = trimesh.load(args.pred, process=False)
    ct = trimesh.load(args.ct, process=False)
    result = evaluate_reconstruction(np.asarray(pred.vertices), ct, init=load_init(args.init, not args.rigid),
                                     with_scale=not args.rigid, trim=args.trim)

    args.out.mkdir(parents=True, exist_ok=True)
    trimesh.PointCloud(result["aligned"], colors=error_colors(result["distances"], args.vmax)).export(
        args.out / "error_heatmap.ply")
    np.save(args.out / "distances_mm.npy", result["distances"])
    report = {k: result[k] for k in ("scale", "icp_rmse", "icp_iterations", "summary")}
    report["transform"] = result["transform"].tolist()
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2))
    s = result["summary"]
    print(f"median {s['median_abs']:.2f} mm, mean {s['mean_abs']:.2f} mm, P95 {s['p95_abs']:.2f} mm, "
          f"within 2 mm {100 * s['within_2mm']:.0f}%, scale {result['scale']:.4g} -> {args.out}")


if __name__ == "__main__":
    main()
