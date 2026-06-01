#!/usr/bin/env bash
# Production mining: local Qwen .npy shards + validator-aligned offline eval.
#
# Usage:
#   cp scripts/mining/.env.example scripts/mining/.env
#   # Edit .env with your paths and hotkey
#   chmod +x scripts/mining/run_prod_train.sh
#   ./scripts/mining/run_prod_train.sh
#
# Or override via env:
#   LOCAL_DATASET_MANIFEST=/data/fw.json ./scripts/mining/run_prod_train.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

# Load .env if present (in scripts/mining/ or repo root)
for _env_path in scripts/mining/.env .env; do
  if [[ -f "$_env_path" ]]; then
    echo "[run_prod_train] loading env from $_env_path"
    set -a
    # shellcheck disable=SC1090
    source "$_env_path"
    set +a
    break
  fi
done

: "${LOCAL_DATASET_MANIFEST:?Set LOCAL_DATASET_MANIFEST in .env or environment}"
: "${TEUTONIC_SIM_HOTKEY:?Set TEUTONIC_SIM_HOTKEY in .env or environment}"

# Optional — if not set, the script downloads the king from the dashboard
LOCAL_KING_DIR="${LOCAL_KING_DIR:-}"

export TEUTONIC_EVAL_DATASET_MODE="${TEUTONIC_EVAL_DATASET_MODE:-raw_hippius}"
export TEUTONIC_RAW_TOKENIZER_REPO="${TEUTONIC_RAW_TOKENIZER_REPO:-Qwen/Qwen3-4B}"
export TEUTONIC_RAW_MAX_FILES_PER_EVAL="${TEUTONIC_RAW_MAX_FILES_PER_EVAL:-32}"

WORK="${TEUTONIC_WORK_DIR:-/root/teutonic/s1-work-prod}"
REPORT="${TEUTONIC_VERDICT:-${WORK}/verdict.json}"
N_GPUS="${N_GPUS:-1}"
PRESET="${CANDIDATE_PRESET:-main}"

echo "[run_prod_train] work=${WORK}"
echo "[run_prod_train] preset=${PRESET}  n_gpus=${N_GPUS}"
echo "[run_prod_train] manifest=${LOCAL_DATASET_MANIFEST}"

exec python -u scripts/mining/train_challenger.py \
  --work "${WORK}" \
  --bundle scripts/training_bundle \
  --dataset-mode auto \
  --local-dataset-manifest "${LOCAL_DATASET_MANIFEST}" \
  --candidate-preset "${PRESET}" \
  --n-shards 12 \
  --shard-start 0 \
  --n-score 64000 \
  --train-per-iter 32000 \
  --val-size 1500 \
  --max-iters 3 \
  --general-frac 0.70 \
  --hard-frac 0.20 \
  --easy-frac 0.10 \
  --micro-batch 2 \
  --grad-accum 16 \
  --n-gpus "${N_GPUS}" \
  --eval-mode validator \
  --sim-hotkey "${TEUTONIC_SIM_HOTKEY}" \
  --n-eval 25600 \
  --eval-batch-size 64 \
  --eval-gpus "0" \
  --fast-eval \
  --fast-eval-n 3000 \
  --acceptance-lcb-floor 0.0035 \
  --mean-delta-floor 0.0045 \
  --bootstrap 10000 \
  --alpha 0.001 \
  --use-local-score-cache \
  --report-out "${REPORT}"
