# Teutonic Mining — Commands Reference

## Overview

The production training flow is:

```text
retokenize_fineweb_edu_qwen.py  →  local .npy shards (one-time build)
train_challenger.py              →  score → curriculum → LoRA → merge → eval → verdict
submit_challenger.py             →  submit accepted model to chain
```

### When to use which preset

| Situation | Preset | Approx time |
| --------- | ------ | ----------- |
| New challenge, king on different dataset | `fast_first_strike` | ~20-30 min |
| New challenge, want safer first run | `safe_first_strike` | ~30-45 min |
| Follow-up after first submission | `strong_followup` | ~60-90 min |
| Normal competition, refine scoring | `main` | ~60-120 min |
| Conservative low-risk run | `safe` | ~90-120 min |
| Aggressive high-gain attempt | `aggressive` | ~60 min |

**First-strike presets** (`fast_first_strike`, `safe_first_strike`) use `n_score=0`:
they skip the expensive king forward-pass scoring step and sample directly from
clean local FineWeb-Edu shards. This is ideal when the king may be overfit to a
different distribution (e.g. old CulturaX-trained king vs new FineWeb-Edu challenge).

**`strong_followup`** uses `n_score=64000` to score with the king, build a proper
curriculum, and train on harder examples. Run this after your first-strike submission.

---

## A. Build Local FineWeb-Edu Shards (one-time)

Build 16 train + 2 eval shards from `sample-10BT` (~10B token dataset):

```bash
python scripts/mining/retokenize_fineweb_edu_qwen.py \
    --tokenizer-dir /root/teutonic/s1-work/king \
    --out-dir /data/fineweb_edu_qwen3_2048 \
    --dataset HuggingFaceFW/fineweb-edu \
    --config sample-10BT \
    --split train \
    --seq-len 2048 \
    --prod \
    --max-train-shards 16 \
    --max-eval-shards 2 \
    --shuffle-buffer-size 10000 \
    --quality-filter \
    --streaming \
    --resume \
    --seed 42
```

**Production shard size:** `--prod` sets `rows_per_shard=262144`, giving ~536M tokens/shard (~2 GiB).

Expand to 32 train + 4 eval shards from `sample-100BT`:

```bash
python scripts/mining/retokenize_fineweb_edu_qwen.py \
    --tokenizer-dir /root/teutonic/s1-work/king \
    --out-dir /data/fineweb_edu_qwen3_2048 \
    --dataset HuggingFaceFW/fineweb-edu \
    --config sample-100BT \
    --seq-len 2048 \
    --prod \
    --max-train-shards 32 \
    --max-eval-shards 4 \
    --streaming \
    --resume
```

---

## B. Validate Existing Local Shards

```bash
python scripts/mining/retokenize_fineweb_edu_qwen.py \
    --out-dir /data/fineweb_edu_qwen3_2048 \
    --validate-only
```

Checks:

- All shards in `manifest.json` exist
- Correct shape `(rows_per_shard, seq_len)`
- dtype is `uint32`
- Train/eval split is present
- Tokenizer metadata exists

---

## C. Set Up Environment

```bash
cp scripts/mining/.env.example scripts/mining/.env
# Edit .env — set LOCAL_DATASET_MANIFEST, TEUTONIC_SIM_HOTKEY, LOCAL_KING_DIR
```

Or export directly:

```bash
export LOCAL_DATASET_MANIFEST=/data/fineweb_edu_qwen3_2048/manifest.json
export LOCAL_KING_DIR=/root/teutonic/s1-work/king
export TEUTONIC_SIM_HOTKEY=5FxJ...YOUR_HOTKEY...
export N_GPUS=1
```

---

## D. First-Strike: 20-30 Minute Candidate (new challenge, unknown king distribution)

Use this when a new challenge starts and the current king may be overfit to a different
dataset. `n_score=0` skips the expensive king scoring step; training data is sampled
directly from clean local FineWeb-Edu shards.

### Fast First Strike (~20-30 min)

```bash
python -u scripts/mining/train_challenger.py \
  --work /root/teutonic/s1-fast \
  --bundle scripts/training_bundle \
  --dataset-mode local \
  --local-dataset-manifest /data/fineweb_edu_qwen3_2048/manifest.json \
  --candidate-preset fast_first_strike \
  --first-strike \
  --max-iters 1 \
  --n-gpus 1 \
  --micro-batch 2 \
  --grad-accum 8 \
  --eval-mode local \
  --sim-hotkey "${TEUTONIC_SIM_HOTKEY}" \
  --acceptance-lcb-floor 0.0025 \
  --report-out /root/teutonic/s1-fast/verdict.json
```

Preset values: `n_score=0, train_per_iter=8192, val_size=512, lr=8e-5, lora_r=32, epochs=1.0`

### Safe First Strike (~30-45 min)

```bash
python -u scripts/mining/train_challenger.py \
  --work /root/teutonic/s1-fast \
  --bundle scripts/training_bundle \
  --dataset-mode local \
  --local-dataset-manifest /data/fineweb_edu_qwen3_2048/manifest.json \
  --candidate-preset safe_first_strike \
  --first-strike \
  --max-iters 1 \
  --n-gpus 1 \
  --micro-batch 2 \
  --grad-accum 8 \
  --eval-mode local \
  --sim-hotkey "${TEUTONIC_SIM_HOTKEY}" \
  --acceptance-lcb-floor 0.0025 \
  --report-out /root/teutonic/s1-fast/verdict.json
```

Preset values: `n_score=0, train_per_iter=16384, val_size=1024, lr=5e-5, lora_r=32, epochs=1.0`

### After First-Strike: Strong Follow-Up (~60-90 min)

Run this after submitting the first-strike candidate to get a better, scoring-backed model:

```bash
python -u scripts/mining/train_challenger.py \
  --work /root/teutonic/s1-work-prod \
  --bundle scripts/training_bundle \
  --dataset-mode local \
  --local-dataset-manifest /data/fineweb_edu_qwen3_2048/manifest.json \
  --candidate-preset strong_followup \
  --max-iters 2 \
  --n-gpus 1 \
  --micro-batch 2 \
  --grad-accum 16 \
  --eval-mode validator \
  --sim-hotkey "${TEUTONIC_SIM_HOTKEY}" \
  --fast-eval \
  --acceptance-lcb-floor 0.0035 \
  --mean-delta-floor 0.0045 \
  --report-out /root/teutonic/s1-work-prod/verdict.json
```

Preset values: `n_score=64000, train_per_iter=32768, val_size=1500, lr=5e-5, lora_r=64, epochs=1.5`

---

## E. Run Production Training (recommended: via shell script)

```bash
./scripts/mining/run_prod_train.sh
```

Or manually with the `main` preset (recommended for the current 0.0025 gap):

```bash
export LOCAL_DATASET_MANIFEST=/data/fineweb_edu_qwen3_2048/manifest.json

python -u scripts/mining/train_challenger.py \
  --work /root/teutonic/s1-work-prod \
  --bundle scripts/training_bundle \
  --dataset-mode auto \
  --candidate-preset main \
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
  --n-gpus 1 \
  --eval-mode validator \
  --sim-hotkey "${TEUTONIC_SIM_HOTKEY}" \
  --fast-eval \
  --fast-eval-n 3000 \
  --acceptance-lcb-floor 0.0035 \
  --mean-delta-floor 0.0045 \
  --report-out /root/teutonic/s1-work-prod/verdict.json
```

---

## F. Preset Variants

### Safe (conservative, low risk)

```bash
python -u scripts/mining/train_challenger.py \
  --work /root/teutonic/s1-work-prod \
  --bundle scripts/training_bundle \
  --dataset-mode auto \
  --candidate-preset safe \
  --n-score 64000 \
  --train-per-iter 32000 \
  --val-size 1500 \
  --max-iters 3 \
  --n-gpus 1 \
  --eval-mode validator \
  --sim-hotkey "${TEUTONIC_SIM_HOTKEY}" \
  --fast-eval \
  --report-out /root/teutonic/s1-work-prod/verdict.json
```

Preset values: `lr=2e-5, lora_r=32, lora_alpha=64, epochs=1.5, target_mu=0.0045`

### Main (default, recommended)

Same as above but `--candidate-preset main`:
`lr=5e-5, lora_r=64, lora_alpha=128, epochs=1.5, target_mu=0.006`

### Aggressive (higher risk, potentially larger gain)

```bash
python -u scripts/mining/train_challenger.py \
  --work /root/teutonic/s1-work-prod \
  --bundle scripts/training_bundle \
  --dataset-mode auto \
  --candidate-preset aggressive \
  --n-score 64000 \
  --train-per-iter 32000 \
  --val-size 1500 \
  --max-iters 2 \
  --n-gpus 1 \
  --eval-mode validator \
  --sim-hotkey "${TEUTONIC_SIM_HOTKEY}" \
  --fast-eval \
  --report-out /root/teutonic/s1-work-prod/verdict.json
```

Preset values: `lr=8e-5, lora_r=64, lora_alpha=128, epochs=1.0, target_mu=0.007`

---

## G. Smoke Test (verify pipeline without GPU hours)

```bash
./scripts/mining/smoke_train_challenger.sh
```

Or manually:

```bash
python -u scripts/mining/train_challenger.py \
  --work /tmp/teutonic-smoke \
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
  --report-out /tmp/teutonic-smoke/verdict.json
```

---

## H. Submit Accepted Challenger

Only submit if `verdict.json` has `best.accepted = true`.

```bash
python scripts/mining/submit_challenger.py \
    --verdict /root/teutonic/s1-work-prod/verdict.json \
    --hotkey hotkey30 \
    --wallet-name silvanus-hs2
```

Dry-run first:

```bash
python scripts/mining/submit_challenger.py \
    --verdict /root/teutonic/s1-work-prod/verdict.json \
    --hotkey hotkey30 \
    --wallet-name silvanus-hs2 \
    --dry-run
```

Force-submit even if rejected (not recommended):

```bash
python scripts/mining/submit_challenger.py \
    --verdict /root/teutonic/s1-work-prod/verdict.json \
    --hotkey hotkey30 \
    --wallet-name silvanus-hs2 \
    --force
```

---

## I. Standalone Validator Eval

Run only the eval step against an existing model:

```bash
export TEUTONIC_EVAL_DATASET_MODE=raw_hippius
export TEUTONIC_RAW_TOKENIZER_REPO=Qwen/Qwen3-4B

python -u scripts/mining/validator_eval.py \
    --king /root/teutonic/s1-work-prod/king \
    --challenger /root/teutonic/s1-work-prod/iter_00/merged \
    --hotkey "${TEUTONIC_SIM_HOTKEY}" \
    --n-public 25600 \
    --batch-size 64 \
    --gpus 0 \
    --report-out /root/teutonic/s1-work-prod/verdict.json
```

Using local shards for eval (faster, no download):

```bash
python -u scripts/mining/validator_eval.py \
    --king /root/teutonic/s1-work-prod/king \
    --challenger /root/teutonic/s1-work-prod/iter_00/merged \
    --hotkey "${TEUTONIC_SIM_HOTKEY}" \
    --local-dataset /data/fineweb_edu_qwen3_2048/manifest.json \
    --n-public 5000 \
    --report-out /root/teutonic/s1-work-prod/verdict.json
```

---

## J. Score Cache

Score cache avoids re-running the king model on the same data across iterations.
Located at: `<work>/score_cache/king_<hash>/manifest_<hash>/`

- Cache is populated automatically on the first scoring run.
- Subsequent iterations of the same manifest reuse cached scores.
- Force re-score: `--force-rescore`
- Disable cache: `--no-use-local-score-cache`
- First-strike presets (`n_score=0`) do not use the score cache.

---

## K. Presets Reference

| Preset | lr | lora_r | lora_alpha | epochs | n_score | train/iter | target_mu |
| ------ | -- | ------ | ---------- | ------ | ------- | ---------- | --------- |
| fast_first_strike | 8e-5 | 32 | 64 | 1.0 | 0 | 8192 | 0.0035 |
| safe_first_strike | 5e-5 | 32 | 64 | 1.0 | 0 | 16384 | 0.004 |
| safe | 2e-5 | 32 | 64 | 1.5 | — | — | 0.0045 |
| main | 5e-5 | 64 | 128 | 1.5 | — | — | 0.006 |
| strong_followup | 5e-5 | 64 | 128 | 1.5 | 64000 | 32768 | 0.006 |
| aggressive | 8e-5 | 64 | 128 | 1.0 | — | — | 0.007 |
| custom | CLI | CLI | CLI | CLI | CLI | CLI | CLI |

`n_score=0` = skip king scoring, sample directly from local shards.
`—` = use CLI `--n-score` / `--train-per-iter` value (defaults: 4000/4000).

All preset values can be overridden by explicit CLI args
(e.g. `--candidate-preset main --lr 3e-5`).

---

## L. Output Directory Structure

```text
<work>/
  king/                   # current king model weights
  cache/                  # manifest + downloaded shards
  score_cache/            # per-king scored JSONL refs (no token lists)
    king_<hash>/
      manifest_<hash>/
        scored_shard_000000.jsonl
        ...
  iter_00/
    train.jsonl           # selected training sequences
    val.jsonl             # validation sequences
    scored.jsonl          # lightweight scored refs (n_score>0 only)
    scoring.json          # loss distribution stats
    curriculum.json       # curriculum selection report
    lora/                 # LoRA adapter output
      best_adapter/
      train_summary.json  # training metrics
    merged/               # merged full model
    eval_fast.json        # fast local eval result (if --fast-eval)
    verdict.json          # final eval verdict
  iter_01/
    ...
  best/
    verdict.json          # best iteration verdict
    best_model_path.txt   # path to best merged model
  race_summary.json       # full race report
  race_summary.md         # human-readable summary
  verdict.json            # final top-level verdict (--report-out)
```

cd /root/teutonic && source .venv/bin/activate

export LOCAL_KING_DIR="/root/teutonic/s1-work/king1"
export TEUTONIC_SIM_HOTKEY="5FhMoUmcE9ed4p1it7xebF1y1SHdC5hYFbD1Gk44wuiX88hv"
export TRANSFORMERS_TRUST_REMOTE_CODE=true

python -u scripts/mining/train_challenger.py \
  --work /root/teutonic/s1-work \
  --bundle /root/teutonic/scripts/training_bundle \
  --dataset-mix /root/teutonic/scripts/mining/dataset_mix_quasar_v4.json \
  --mix-shards-per-dataset 12 \
  --candidate-preset strong_followup \
  --n-score 32000 \
  --train-per-iter 20000 \
  --val-size 1500 \
  --hard-frac 0.35 \
  --general-frac 0.55 \
  --easy-frac 0.10 \
  --epochs 1.5 \
  --lr 5e-5 \
  --max-iters 1 \
  --n-gpus 1 \
  --micro-batch 2 \
  --grad-accum 8 \
  --eval-mode validator \
  --final-eval-n 1000 \
  --n-eval 1000 \
  --report-out /root/teutonic/s1-work/verdict.json