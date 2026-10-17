#!/usr/bin/env python
"""Stage 0b: upright, metadata-free copies of every clinic photo (raw data stays untouched).

The raw photos carry EXIF visit dates and, for some cases, GPS positions; everything
downstream reads these clean copies instead.
"""
import argparse
from pathlib import Path

from facerecon.audit import list_cases
from facerecon.photos import export_clean_photo, list_photos, photo_info


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data_root", help="folder with one sub-folder per case")
    ap.add_argument("--out", required=True, help="output folder (<out>/<case>/<photo>.jpg)")
    args = ap.parse_args()

    count = 0
    for case in list_cases(args.data_root):
        for src in list_photos(case.photo_dir):
            dst = export_clean_photo(src, Path(args.out) / case.slug / f"{src.stem}.jpg")
            info = photo_info(dst)
            if info["has_datetime"] or info["has_gps"] or info["model"]:
                raise SystemExit(f"metadata survived in {dst}")
            count += 1
    print(f"{count} clean photos -> {args.out}")


if __name__ == "__main__":
    main()
