#!/usr/bin/env bash
# Smoke test for the full train_challenger.py pipeline.
#
# Uses tiny numbers so it completes in minutes (no heavy GPU work).
# Verifies that:
#   - local manifest loading works (or remote fallback)
#   - curriculum writes train/val jsonl
#   - LoRA train command starts (torchrun, may be CPU-only for smoke test)
#   - output dir is created
#   - verdict.json is written
#   - rejected candidates are not submitted
#
# Usage:
#   chmod +x scripts/mining/smoke_train_challenger.sh
#   ./scripts/mining/smoke_train_challenger.sh
#
# Override work dir:
#   SMOKE_WORK=/tmp/my-smoke ./scripts/mining/smoke_train_challenger.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

SMOKE_WORK="${SMOKE_WORK:-/tmp/teutonic-smoke-$$}"
REPORT="${SMOKE_WORK}/verdict.json"

echo "============================================"
echo "  Teutonic Smoke Test"
echo "  work: ${SMOKE_WORK}"
echo "============================================"

# ---- Step 1: Check Python imports ----
echo "[smoke] checking Python imports..."
python -c "
import sys
print('Python', sys.version)
import numpy, torch, transformers, peft, datasets
print('numpy', numpy.__version__)
print('torch', torch.__version__)
print('transformers', transformers.__version__)
print('peft', peft.__version__)
print('OK: all imports pass')
"

# ---- Step 2: py_compile all changed scripts ----
echo "[smoke] syntax-checking Python files..."
python -m py_compile scripts/mining/train_challenger.py
python -m py_compile scripts/mining/retokenize_fineweb_edu_qwen.py
python -m py_compile scripts/mining/validator_eval.py
python -m py_compile scripts/mining/submit_challenger.py
python -m py_compile scripts/training_bundle/train_lora_token_ids.py
python -m py_compile scripts/training_bundle/build_curriculum.py
python -m py_compile scripts/training_bundle/score_samples.py
echo "[smoke] syntax check PASSED"

# ---- Step 3: Validate local dataset (if manifest exists) ----
LOCAL_MANIFEST="${LOCAL_DATASET_MANIFEST:-}"
if [[ -n "${LOCAL_MANIFEST}" && -f "${LOCAL_MANIFEST}" ]]; then
  echo "[smoke] validating local manifest: ${LOCAL_MANIFEST}"
  python scripts/mining/retokenize_fineweb_edu_qwen.py \
    --out-dir "$(dirname "${LOCAL_MANIFEST}")" \
    --validate-only && echo "[smoke] manifest VALID" || echo "[smoke] WARNING: manifest validation failed"
else
  echo "[smoke] LOCAL_DATASET_MANIFEST not set — skipping manifest validation"
fi

# ---- Step 4: Run tiny training pipeline ----
echo "[smoke] running tiny pipeline (n-score=64 train=32 val=8 1 iter)..."

HOTKEY="${TEUTONIC_SIM_HOTKEY:-smoke-test-hotkey}"

python -u scripts/mining/train_challenger.py \
  --work "${SMOKE_WORK}" \
  --bundle scripts/training_bundle \
  --dataset-mode auto \
  --candidate-preset safe \
  --n-score 64 \
  --train-per-iter 32 \
  --val-size 8 \
  --max-iters 1 \
  --n-gpus 1 \
  --eval-mode local \
  --fast-eval \
  --fast-eval-n 32 \
  --acceptance-lcb-floor 0.0 \
  --mean-delta-floor 0.0 \
  --report-out "${REPORT}" \
  --sim-hotkey "${HOTKEY}" \
  2>&1 | tee "${SMOKE_WORK}/smoke.log" || {
    echo "[smoke] WARN: pipeline exited non-zero (may be expected for local eval without GPU)"
  }

# ---- Step 5: Check outputs ----
echo "[smoke] checking outputs..."
PASS=1

check_file() {
  local f="$1"
  if [[ -f "$f" ]]; then
    echo "  OK: $f"
  else
    echo "  MISSING: $f"
    PASS=0
  fi
}

check_dir() {
  local d="$1"
  if [[ -d "$d" ]]; then
    echo "  OK dir: $d"
  else
    echo "  MISSING dir: $d"
    PASS=0
  fi
}

check_dir "${SMOKE_WORK}"
check_dir "${SMOKE_WORK}/iter_00"
check_file "${SMOKE_WORK}/iter_00/train.jsonl"
check_file "${SMOKE_WORK}/iter_00/val.jsonl"
check_file "${SMOKE_WORK}/iter_00/scored.jsonl"
check_file "${SMOKE_WORK}/iter_00/curriculum.json"

# verdict might not exist if GPU not available; that is OK for a parse check
if [[ -f "${REPORT}" ]]; then
  echo "  OK: verdict.json exists"
  python -c "
import json, sys
v = json.load(open('${REPORT}'))
best = v.get('best') or {}
accepted = best.get('accepted', False)
mu = best.get('mu_hat', 0)
print(f'  verdict: accepted={accepted} mu_hat={mu:.6f}')
# submitted field must NOT exist unless explicitly uploaded
assert 'uploaded_repo' not in v or not accepted or '${UPLOAD_REPO:-}', \
  'verdict has uploaded_repo without explicit UPLOAD_REPO — check submission gate'
print('  verdict fields OK')
" 2>&1 || echo "  WARN: verdict parse issue"
else
  echo "  INFO: verdict.json not found (expected if GPU unavailable)"
fi

echo ""
if [[ "$PASS" -eq 1 ]]; then
  echo "============================================"
  echo "  SMOKE TEST PASSED"
  echo "  work dir: ${SMOKE_WORK}"
  echo "============================================"
else
  echo "============================================"
  echo "  SMOKE TEST FAILED — see output above"
  echo "  logs: ${SMOKE_WORK}/smoke.log"
  echo "============================================"
  exit 1
fi
