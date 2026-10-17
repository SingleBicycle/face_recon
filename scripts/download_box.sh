#!/usr/bin/env bash
# Mirror a UT Box data drop to local storage and verify it against Box checksums.
# One-time setup: `rclone config create utbox box` (log in with the invited Box account).
#
#   scripts/download_box.sh "Training Scans 9.2026" training_scans_2026-09
set -euo pipefail

BOX_FOLDER=${1:?usage: download_box.sh "<folder under 2D to 3D Facial Model Project>" <local name>}
NAME=${2:?usage: download_box.sh "<folder>" <local name>}
SRC="utbox:2D to 3D Facial Model Project/${BOX_FOLDER}"
ROOT=${FACE_RECON_DATA:-/DATA2/zihao/face_recon_data}
DEST="${ROOT}/raw/${NAME}"

mkdir -p "$DEST"
chmod 700 "$ROOT"  # IRB: access limited to study personnel
printf 'Source: %s\nDownloaded with rclone on %s\n' "$SRC" "$(date +%F)" > "${ROOT}/raw/${NAME}.SOURCE.txt"
rclone copy "$SRC" "$DEST" --transfers 8 --checkers 16 --log-file "${ROOT}/rclone_copy.log" --log-level INFO
rclone check "$SRC" "$DEST" --one-way
