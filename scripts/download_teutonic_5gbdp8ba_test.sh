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
#   WORK_DIR=/root/teutonic/s1-work ./scripts/download_teutonic_5gbdp8ba_test.sh

set -euo pipefail

REPO="whiskey/teutonic-5dawwwmr-7971731779-cp1"
REV="5DaWwWmR-7971731779"
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

# Hippius kings often omit tokenizer files — copy from sibling or TEUTONIC_TOKENIZER_DIR
if [[ ! -f "${OUT_DIR}/tokenizer.json" ]]; then
  TOK_SRC="${TEUTONIC_TOKENIZER_DIR:-}"
  if [[ -z "${TOK_SRC}" && -f "$(dirname "${OUT_DIR}")/king1/tokenizer.json" ]]; then
    TOK_SRC="$(dirname "${OUT_DIR}")/king1"
  fi
  if [[ -n "${TOK_SRC}" && -f "${TOK_SRC}/tokenizer.json" ]]; then
    echo "[copy] tokenizer from ${TOK_SRC}"
    cp -av "${TOK_SRC}/tokenizer.json" "${TOK_SRC}/tokenizer_config.json" "${OUT_DIR}/" 2>/dev/null || \
      cp -av "${TOK_SRC}/tokenizer.json" "${OUT_DIR}/"
  else
    echo "[warn] no tokenizer.json in ${OUT_DIR}; set TEUTONIC_TOKENIZER_DIR or copy from king1 before training"
  fi
fi

CACHE_SNAPSHOT="${HOME}/.cache/hippius/hub/models--${REPO//\//--}/snapshots/${REV}"
WORK_DIR="${WORK_DIR:-/root/teutonic/s1-work}"

echo
echo "[copy] ${CACHE_SNAPSHOT} -> ${WORK_DIR}"
mkdir -p "$WORK_DIR"
cp -avL "$CACHE_SNAPSHOT" "$WORK_DIR"
