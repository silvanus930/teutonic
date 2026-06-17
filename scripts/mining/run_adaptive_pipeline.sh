#!/usr/bin/env bash
# Adaptive hyperparameter pipeline: screen → tune rounds → strong training.
#
# Usage:
#   export LOCAL_KING_DIR=/root/teutonic/s1-work/king2
#   export TEUTONIC_SIM_HOTKEY=...
#   ./scripts/mining/run_adaptive_pipeline.sh          # start pipeline
#   ./scripts/mining/run_adaptive_pipeline.sh status   # one-shot status
#   ./scripts/mining/run_adaptive_pipeline.sh watch    # refresh every 15s
#
# Status files (updated live during pipeline):
#   $ADAPTIVE_WORK/status.txt   human-readable summary
#   $ADAPTIVE_WORK/status.json  machine-readable
#   $ADAPTIVE_WORK/adaptive.log full subprocess log
#
# Env overrides:
#   ADAPTIVE_WORK   work directory (default: /root/teutonic/s1-work-adaptive)
#   MIX_CACHE       shard cache (default: /root/teutonic/s1-work/cache)
#   MAX_ROUNDS      tuning rounds (default: 5)
#   FORCE_STRONG=1  run strong even if below crown μ̂/LCB

set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

WORK="${ADAPTIVE_WORK:-/root/teutonic/s1-work-adaptive}"
CACHE="${MIX_CACHE:-/root/teutonic/s1-work/cache}"
MAX_ROUNDS="${MAX_ROUNDS:-5}"
CMD="${1:-run}"
shift || true

case "${CMD}" in
  status)
    exec python -u scripts/mining/show_pipeline_status.py --work "${WORK}" "$@"
    ;;
  watch)
    INTERVAL="${WATCH_INTERVAL:-15}"
    exec watch -n "${INTERVAL}" python -u scripts/mining/show_pipeline_status.py --work "${WORK}"
    ;;
  json)
    exec python -u scripts/mining/show_pipeline_status.py --work "${WORK}" --json "$@"
    ;;
  log|tail)
    exec tail -f "${WORK}/adaptive.log"
    ;;
  run|start|"")
  : "${TEUTONIC_SIM_HOTKEY:?Set TEUTONIC_SIM_HOTKEY}"

  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

  EXTRA=()
  if [[ "${RESUME:-0}" == "1" ]]; then
    EXTRA+=(--resume)
  fi
  if [[ "${LOCAL_SHARDS_ONLY:-0}" == "1" ]]; then
    EXTRA+=(--local-shards-only)
  fi
  if [[ "${PREFETCH_SHARDS:-1}" == "1" ]]; then
    EXTRA+=(--prefetch-shards)
  fi
  if [[ "${FORCE_STRONG:-0}" == "1" ]]; then
    EXTRA+=(--force-strong)
  fi
  if [[ "${SKIP_STRONG:-0}" == "1" ]]; then
    EXTRA+=(--skip-strong)
  fi

  echo "[adaptive] work=${WORK} cache=${CACHE} rounds=${MAX_ROUNDS}"
  echo "[adaptive] status: ${WORK}/status.txt  (./scripts/mining/run_adaptive_pipeline.sh watch)"

  exec python -u scripts/mining/run_adaptive_pipeline.py \
    --work "${WORK}" \
    --mix-shard-cache "${CACHE}" \
    --sim-hotkey "${TEUTONIC_SIM_HOTKEY}" \
    --max-rounds "${MAX_ROUNDS}" \
    --auto-bucket-mix \
    --mix-shards-per-dataset "${MIX_SHARDS_PER_DATASET:-12}" \
    --prefetch-shards-per-dataset "${PREFETCH_SHARDS_PER_DATASET:-48}" \
    "${EXTRA[@]}" \
    "$@"
    ;;
  *)
    echo "usage: $0 {run|status|watch|json|log}" >&2
    exit 2
    ;;
esac
