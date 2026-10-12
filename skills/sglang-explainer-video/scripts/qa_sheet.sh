#!/usr/bin/env bash
# Contact sheet(s) sampled from the RENDERED MP4 for the final frame-by-frame review
# (乱码 / typos / translation / layout). Always review the MP4 itself, not Studio snapshots.
#   bash qa_sheet.sh video.mp4 [out_dir] [every_seconds]
set -euo pipefail
VIDEO="$1"
OUT="${2:-$(dirname "$VIDEO")/qa}"
EVERY="${3:-2.5}"
mkdir -p "$OUT"
rm -f "$OUT"/sheet_*.png
ffmpeg -v error -i "$VIDEO" -vf "fps=1/${EVERY},scale=640:-1,tile=3x4" "$OUT/sheet_%02d.png"
ls "$OUT"/sheet_*.png
