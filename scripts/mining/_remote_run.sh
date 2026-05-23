#!/usr/bin/env bash
# Wrapper invoked inside the tmux session on the GPU box.
# Reads HF_TOKEN from /root/teutonic-mining/.hf_token (chmod 600).
#
# UPLOAD_REPO must contain the first 8 ss58 chars of your coldkey
# (case-insensitive substring, anywhere in account or basename).
# Without that the validator rejects with `coldkey_required` and the
# whole training run is wasted. run_pipeline.sh verifies this locally
# before launching us; if you invoke this script directly, double-check.
set -euo pipefail
cd /root/teutonic-mining
export HF_TOKEN="$(cat .hf_token)"

# Match production validator/eval (ecosystem.config.js): raw FineWeb-Edu on
# Hippius, tokenized at load time with Qwen3 — not v2 Gemma pretokenized shards.
export TEUTONIC_EVAL_DATASET_MODE="${TEUTONIC_EVAL_DATASET_MODE:-raw_hippius}"
export TEUTONIC_RAW_TOKENIZER_REPO="${TEUTONIC_RAW_TOKENIZER_REPO:-Qwen/Qwen3-4B}"
export TEUTONIC_RAW_MAX_FILES_PER_EVAL="${TEUTONIC_RAW_MAX_FILES_PER_EVAL:-8}"

# King is OCI on dashboard (sha256:…); HF snapshot_download usually fails.
# Set LOCAL_KING_DIR on the GPU box to a pre-synced merged/base model tree.
: "${LOCAL_KING_DIR:=}"

# Validator-aligned offline gate (same holdout path as eval_server).
: "${SIM_HOTKEY:?Set SIM_HOTKEY to the hotkey ss58 you will submit with}"

exec ./venv/bin/python -u train_challenger.py \
  --work /root/teutonic-mining/work \
  --bundle /root/teutonic-mining/bundle \
  --upload-repo "${UPLOAD_REPO:?UPLOAD_REPO must be set (matching the active chain.toml [chain].name)}" \
  --report-out /root/teutonic-mining/work/verdict.json \
  --hf-token "$HF_TOKEN" \
  --dataset-mode auto \
  --eval-mode validator \
  --sim-hotkey "${SIM_HOTKEY}" \
  --raw-max-files 32 \
  --n-eval 5000 \
  --n-eval-private 0 \
  --eval-batch-size 64 \
  --eval-gpus 0 \
  --n-score 8000 \
  --train-per-iter 7000 \
  --val-size 400 \
  --max-iters 4 \
  --target-mu 0.008 \
  --micro-batch 2 \
  --grad-accum 16 \
  --lr 1.5e-4 \
  --epochs 2 \
  --lora-r 16 \
  --lora-alpha 32 \
  --n-gpus 1
