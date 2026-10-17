#!/usr/bin/env python
"""Stage 0: audit a data drop (CT series, photos, de-identification) -> audit.csv / audit.json."""
import argparse

from facerecon.audit import audit_dataset, write_audit


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data_root", help="folder with one sub-folder per case")
    ap.add_argument("--out", required=True, help="output folder")
    args = ap.parse_args()

    rows = audit_dataset(args.data_root)
    write_audit(rows, args.out)
    print(f"{len(rows)} cases audited -> {args.out}")
    for row in rows:
        if row["issues"]:
            print(f"  {row['slug']}: " + "; ".join(row["issues"]))


if __name__ == "__main__":
    main()
