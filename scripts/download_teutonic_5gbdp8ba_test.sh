#!/usr/bin/env bash
# Download dendrite/teutonic-5gbdp8ba-test-dv00s from Hippius Hub (revision: main).
# Hub: https://hub.hippius.com/models/dendrite/teutonic-5gbdp8ba-test-dv00s/main
#
# Usage:
#   chmod +x scripts/download_teutonic_5gbdp8ba_test.sh
#   ./scripts/download_teutonic_5gbdp8ba_test.sh
#
# Optional:
#   OUT_DIR=/data/models/teutonic-test ./scripts/download_teutonic_5gbdp8ba_test.sh

set -euo pipefail

REPO="mastertensor/teutonic-q3-10b-5ek5koe5-41021132322-rn"
REV="5Ek5KoE5-41021132322-rn"
OUT_DIR="${OUT_DIR:-/root/teutonic/s1-work/king1}"

FILES=(
  model-00001-of-00004.safetensors
  model-00002-of-00004.safetensors
  model-00003-of-00004.safetensors
  model-00004-of-00004.safetensors
  model.safetensors.index.json
  modeling_qwen3_5.py
  configuration_qwen3_5.py
  config.json
  generation_config.json
)

if ! command -v hippius-hub >/dev/null 2>&1; then
  echo "hippius-hub not found. Install with: pip install hippius-hub" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

echo "[download] repo=${REPO} revision=${REV}"
echo "[download] output=${OUT_DIR}"
echo "[download] ~32 GB total (9 weight shards)"
echo

for f in "${FILES[@]}"; do
  if [[ -f "$f" ]] && [[ "$(stat -c%s "$f" 2>/dev/null || echo 0)" -gt 1024 ]]; then
    echo "[skip] $f already present ($(numfmt --to=iec "$(stat -c%s "$f")" 2>/dev/null || stat -c%s "$f"))"
    continue
  fi
  echo "[get] $f ..."
  hippius-hub download "$REPO" "$f" --revision "$REV"
done

echo
echo "[done] files in ${OUT_DIR}:"
ls -lh
