#!/usr/bin/env bash
# Download dendrite/teutonic-x-genesis from Hippius Hub (revision: main).
# Hub: https://hub.hippius.com/models/dendrite/teutonic-x-genesis/main
#
# Usage:
#   chmod +x scripts/download_teutonic_x_genesis.sh
#   ./scripts/download_teutonic_x_genesis.sh
#
# Optional:
#   OUT_DIR=/data/models/teutonic-x-genesis ./scripts/download_teutonic_x_genesis.sh
#   HIPPIUS_TOKEN=... ./scripts/download_teutonic_x_genesis.sh   # if not logged in

set -euo pipefail

REPO="dendrite/teutonic-x-genesis"
REV="main"
OUT_DIR="${OUT_DIR:-/root/teutonic/models/teutonic-x-genesis}"

FILES=(
  model.safetensors
  tokenizer.json
  modeling_qwen3_5.py
  configuration_qwen3_5.py
  config.json
  tokenizer_config.json
  eval_only_512.json
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
