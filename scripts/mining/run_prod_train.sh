#!/usr/bin/env bash
# Production mining: local Qwen .npy shards + validator-aligned offline eval.
#
# Required env (edit before run):
#   LOCAL_DATASET_MANIFEST  path to manifest.json (5× ~2G Qwen shards)
#   LOCAL_KING_DIR          current on-chain king (sync from dashboard)
#   TEUTONIC_SIM_HOTKEY     hotkey ss58 you will submit with
#
# Usage:
#   chmod +x scripts/mining/run_prod_train.sh
#   ./scripts/mining/run_prod_train.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

: "${LOCAL_DATASET_MANIFEST:?Set LOCAL_DATASET_MANIFEST to your Qwen npy manifest.json}"
: "${LOCAL_KING_DIR:?Set LOCAL_KING_DIR to current king model directory}"
: "${TEUTONIC_SIM_HOTKEY:?Set TEUTONIC_SIM_HOTKEY to your submit hotkey ss58}"

export TEUTONIC_EVAL_DATASET_MODE=raw_hippius
export TEUTONIC_RAW_TOKENIZER_REPO="${TEUTONIC_RAW_TOKENIZER_REPO:-Qwen/Qwen3-4B}"
export TEUTONIC_RAW_MAX_FILES_PER_EVAL="${TEUTONIC_RAW_MAX_FILES_PER_EVAL:-32}"

WORK="${TEUTONIC_WORK_DIR:-/root/teutonic/s1-work-v2}"
REPORT="${TEUTONIC_VERDICT:-${WORK}/verdict.json}"

# 5 shards: train on 0-3, hold shard 4 for local pretokenized sanity only
exec python -u scripts/mining/train_challenger.py \
  --work "${WORK}" \
  --bundle scripts/training_bundle \
  --profile prod \
  --dataset-mode auto \
  --n-shards 4 \
  --shard-start 0 \
  --eval-shard 4 \
  --eval-mode validator \
  --sim-hotkey "${TEUTONIC_SIM_HOTKEY}" \
  --n-gpus "${TEUTONIC_N_GPUS:-1}" \
  --micro-batch 2 \
  --grad-accum 16 \
  --lr "${TEUTONIC_LR:-1e-4}" \
  --eval-gpus "${TEUTONIC_EVAL_GPUS:-0}" \
  --report-out "${REPORT}"
