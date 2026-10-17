# face_recon — RGB-to-3D Facial Surface Reconstruction (CT-validated)

Joint project between Texas A&M (Zihao Zhu, Dr. Fan) and UT Austin (Sebastian Kuriakose).

Goal: reconstruct a metric 3D facial soft-tissue surface from routine multi-view clinic photos,
and quantitatively validate it against the CT-derived skin surface of the same patient.
Research use only — not for clinical decision-making.

> **Data policy:** this repo contains code only. Patient data (CT DICOM, photos, derived meshes)
> stays on TAMU research storage under `data/` and is git-ignored. Never commit, upload, or paste
> patient data anywhere else. Access is limited to IRB-approved personnel.

## Data (v1, received 2026-10)

- 26 patients: IDs 1–18 and 27–35 (Patient 8 excluded: CT motion artifact)
- Per patient: one de-identified CT DICOM series + `photos/` (multi-view clinic photos from the visit closest to the CT)
- DICOM headers de-identified; dates truncated to year

Expected local layout (not tracked):

```
data/
  raw/            # untouched copy of the Box export
    Patient 1/
      <CT DICOM series>/
      photos/
  interim/        # CT skin meshes, cropped/undistorted images, manifests
  processed/      # aligned pairs ready for eval/training
```

## Pipeline

| Stage | What | Output |
|---|---|---|
| 0. Audit | Per-patient manifest: #views, image res/EXIF, CT spacing / slice thickness / FOV coverage | `data/interim/manifest.csv` |
| 1. CT skin surface | HU threshold at skin/air boundary + marching cubes, keep largest component, crop face region | skin mesh (mm) |
| 2. RGB → 3D | Sparse uncalibrated multi-view reconstruction (feed-forward pointmap models and/or 3DMM fitting) | face mesh (up to scale) |
| 3. Registration | Landmark init → similarity ICP (rigid + scale) → optional conservative non-rigid | aligned mesh + transform |
| 4. Evaluation | Point-to-surface error (mean / median / P95), per-region breakdown, error heatmaps, scale error reported separately | per-case + cohort tables |

## Repo layout (planned)

```
src/        library code (io, ct_surface, recon, registration, metrics)
scripts/    entry points per stage
configs/    experiment configs
docs/       notes, data plan, meeting notes
```
