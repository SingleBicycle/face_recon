#!/usr/bin/env python
"""Stage 1: CT DICOM -> skin surface mesh (PLY, patient LPS coordinates in mm) per case."""
import argparse
import json
from pathlib import Path

from facerecon.audit import list_cases
from facerecon.ct_surface import SKIN_HU, extract_skin_mesh
from facerecon.dicom_io import find_dicom_series, load_ct_series


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data_root", help="folder with one sub-folder per case")
    ap.add_argument("--out", required=True, help="output folder for <case>_skin.ply / .json")
    ap.add_argument("--level", type=float, default=SKIN_HU, help="air/skin threshold in HU")
    ap.add_argument("--cases", type=int, nargs="*", help="case IDs to process (default: all)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for case in list_cases(args.data_root):
        if args.cases and case.id not in args.cases:
            continue
        series = find_dicom_series(case.dir)
        if not series:
            print(f"{case.slug}: no CT series, skipped")
            continue
        try:
            ct = load_ct_series(max(series.values(), key=len))
            mesh = extract_skin_mesh(ct, args.level)
        except Exception as e:  # keep going; one odd series should not stop the batch
            print(f"{case.slug}: FAILED ({type(e).__name__}: {e})")
            continue
        mesh.export(out / f"{case.slug}_skin.ply")
        info = {
            "case": case.id,
            "level_hu": args.level,
            "volume_shape": list(ct.hu.shape),
            "spacing_mm": [round(float(s), 4) for s in ct.spacing],
            **ct.meta,
            "faces": len(mesh.faces),
            "watertight": bool(mesh.is_watertight),
            "area_cm2": round(float(mesh.area) / 100, 1),
            "extent_mm": [round(float(e), 1) for e in mesh.extents],
        }
        (out / f"{case.slug}_skin.json").write_text(json.dumps(info, indent=2))
        print(f"{case.slug}: {info['faces']} faces, extent {info['extent_mm']} mm, "
              f"spacing {info['spacing_mm']} mm")


if __name__ == "__main__":
    main()
