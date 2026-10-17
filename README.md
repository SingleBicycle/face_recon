# face_recon: RGB-to-3D Facial Surface Reconstruction, validated against CT

Joint project between Texas A&M (Zihao Zhu, Dr. Fan) and UT Austin (Sebastian Kuriakose).

Goal: reconstruct a metric 3D facial soft-tissue surface from routine multi-view clinic photos
and measure how well it matches the CT-derived skin surface of the same patient.
Research use only, not for clinical decision-making.

> **Data policy.** This repo holds code only. Patient data, and anything derived from it (CT
> meshes, point clouds, renders, per-case reports), stays on TAMU research storage and is
> git-ignored. Do not commit, upload or paste it anywhere else, including AI tools. Only personnel
> on the IRB protocol may access it.

## Data

First drop ("Training Scans 9.2026"): 26 patients, IDs 1-18 and 27-35 (patient 8 was excluded
because of a CT motion artifact). Each patient has one CT series and 6-13 clinic photos.

On the lab server the data lives in `/DATA2/zihao/face_recon_data/` (mode 700), symlinked as `data/`:

```
data/
  raw/training_scans_2026-09/<id>/<CT series>/*.dcm   untouched copy of the Box folder
  raw/training_scans_2026-09/<id>/photos/*.jpg
  interim/audit/            audit.csv, audit.json
  interim/photos_clean/     upright, metadata-free photo copies (used by everything downstream)
  interim/ct_skin/          <case>_skin.ply (+ .json), patient LPS coordinates in mm
  interim/vggt_baseline/    per-case reconstruction, error heatmap, metrics; summary.csv
```

What the audit of this drop found:

- **CT series vary between patients:** 17 axial, 8 coronal and 1 sagittal reformat. Pixel size
  is 0.25-0.43 mm, slice spacing 0.62-3 mm, and kernels range from soft tissue to sharp bone.
  Pixel data is JPEG Lossless compressed, so reading it needs `pylibjpeg`.
- **Export quirks the loader handles:**
  - Patients 15, 16 and 17 have every DICOM file twice.
  - Patient 29's series also contains a stray reference image with a different orientation.
- **Photos:** Canon EOS Rebel T3i or T6 with a 100 mm lens. That is a 12.7° field of view,
  close to orthographic. Some photos were cropped or resized, so their EXIF focal length no longer
  matches the pixels.
- **Photo metadata:** all 214 photos carry EXIF capture dates, and 81 photos (9 patients) carry
  EXIF GPS. The pipeline only reads the metadata-free copies made by `clean_photos.py`. The site
  should strip this metadata at export.

## Setup

On the lab server the project uses a venv layered on the existing `mast3r_vggt` conda env, which
already has torch 2.10 + CUDA 12.8 and VGGT:

```bash
/DATA/zihao/miniconda3/envs/mast3r_vggt/bin/python -m venv --system-site-packages /DATA/zihao/envs/face_recon
PIP=/DATA/zihao/envs/face_recon/bin/pip
# mast3r_vggt contains a broken numpy-2.2.6.dist-info, so any pip resolve that touches numpy fails:
$PIP install --no-deps "scikit-image==0.25.2" tifffile lazy_loader pydicom rtree pylibjpeg pylibjpeg-libjpeg \
    transformers tokenizers regex
$PIP install --no-deps -U safetensors
$PIP install -U huggingface_hub
$PIP install --no-deps -e .
```

On any other machine, run `pip install -e ".[dev,recon]"` and install
[VGGT](https://github.com/facebookresearch/vggt).

## Pipeline

All scripts take `--help`. Run them from the repo root with the venv's python.

| Stage | Command | Output |
|---|---|---|
| Download | `scripts/download_box.sh "Training Scans 9.2026" training_scans_2026-09` | `raw/...`, verified against Box checksums |
| 0. Audit | `scripts/audit_data.py data/raw/training_scans_2026-09 --out data/interim/audit` | per-case CT / photo summary, de-identification flags |
| 0. Clean photos | `scripts/clean_photos.py data/raw/training_scans_2026-09 --out data/interim/photos_clean` | upright JPEGs, no EXIF |
| 1. CT skin | `scripts/ct_to_skin_mesh.py data/raw/training_scans_2026-09 --out data/interim/ct_skin` | skin mesh per case |
| 2-5. Baseline | `scripts/run_vggt_baseline.py data/interim/photos_clean data/interim/ct_skin --out data/interim/vggt_baseline` | VGGT reconstruction aligned to CT, error metrics |
| 4-5. Any method | `scripts/evaluate_surface.py pred.ply ct_skin.ply --init landmarks.json --out <dir>` | metrics.json, error heatmap |

## Method notes

- **CT skin surface** (`ct_surface.py`):
  - Threshold at -500 HU, the 50% edge between air and soft tissue.
  - Keep the largest connected component.
  - Fill holes plane by plane in the true axial plane, even for coronal or sagittal series. This
    closes the airway and sinuses even where they open through the nostrils.
  - Run marching cubes on the HU values themselves, so the surface has sub-voxel accuracy.
  - Leave the surface open where the scan ends; no artificial caps.
- **Alignment** (`registration.py`):
  - Start from face-parsing labels on the reconstruction: anterior is the face cap's normal on the
    side the nose protrudes, superior points from the mouth to the eyes. Scale comes from the
    point spread, translation from the nose tips.
  - Multi-start rigid ICP over yaw ±60° and pitch ±40°, then trimmed point-to-plane ICP with scale.
  - The scale is bounded to ±50% of its starting value. Point-to-plane residuals can always be
    driven to zero by collapsing the points onto one spot, and unbounded scale does exactly that.
- **Error:** signed distance from each reconstructed point to the CT skin (positive = outside the
  skin), over points within 90 mm of the CT nose tip.

## Status

- Stages 0, 1 and 4-5 are implemented and tested on synthetic phantoms. They have also been run
  on all 26 cases.
- The VGGT baseline has been run on all 26 cases with fully automatic alignment (no manual
  landmarks):
  - Per-case median surface error: 3.9 mm (median over cases), range 1.1-10.3 mm.
  - Points within 2 mm: 29% (median over cases).
  - The best three cases reach 1.1-1.5 mm, so the pipeline works when the reconstruction does.
  - VGGT's focal estimate averages about 1360 px, against about 2320 px expected for uncropped
    100 mm photos. Generic feed-forward models assume ordinary fields of view, whereas clinical
    photography is telephoto, and the reconstructions distort accordingly.
  - The CT drives the alignment, so these errors are optimistic for shape. They are a reference
    point, not a validation result.
- A photo capture protocol for future clinic sessions (15 fixed views, one camera setup, no
  metadata) is kept with its generator scripts under `docs/capture_protocol/`. It is not tracked in git.

Next:

1. Try methods that accept the known intrinsics or model weak perspective: MapAnything,
   MASt3R with fixed focal length, FLAME / 3DMM fitting.
2. Manually landmark a few cases (nasion, pronasale, exocanthions, cheilions, pogonion) to
   validate the automatic alignment and evaluation region.
3. Test how sensitive the CT skin surface is to the threshold (-300 to -600 HU).
4. Ask the site for the photo-to-CT interval in days, and to strip photo EXIF at export.

## Tests

```bash
/DATA/zihao/envs/face_recon/bin/python -m pytest -m "not gpu"            # synthetic data only, ~20 s
CUDA_VISIBLE_DEVICES=6 /DATA/zihao/envs/face_recon/bin/python -m pytest -m gpu   # model wrappers
```

## Layout

```
src/facerecon/   dicom_io, ct_surface, photos, audit, views, face_parsing, recon_vggt,
                 registration, metrics, evaluate
scripts/         one entry point per stage (see Pipeline)
tests/           pytest; synthetic.py builds the CT phantoms, DICOM series and photos
```
