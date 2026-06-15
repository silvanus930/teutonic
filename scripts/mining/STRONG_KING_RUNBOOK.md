# Strong King Mining Runbook

When the current king is mature, do **not** run a basic r16 / 1-epoch / random-shard pipeline.
The validator accepts only when bootstrap **LCB > 0.0025** on paired cross-entropy (seq_len=2048).

## Decision tree

| Observation | Action |
|---|---|
| `mixture_mu_hat <= 0` after probe | **Stop.** Change data mix or LoRA config. Do not scale up. |
| `mu_hat > 0` but `lcb < 0` | Promising but unstable. Increase `n_eval` and unique samples. |
| `mu_hat ~ 0.003–0.006` | Weak; unlikely to pass final validator. |
| `mu_hat >= 0.0075` and `lcb > 0.0025` | Continue to strong run / merge. |
| `lcb >= 0.0035` | Submit-worthy (`READY_TO_UPLOAD`). |
| FineWebEdu regression warning | Increase finewebedu weight or reduce math-hard ratio. |

## Stage 1 — Probe (cheap beatability check)

```bash
python scripts/mining/train_challenger.py \
  --work /root/teutonic-mining/probe \
  --bundle scripts/training_bundle \
  --mode probe \
  --profile a100-80gb \
  --sim-hotkey YOUR_HOTKEY \
  --report-out /root/teutonic-mining/probe/verdict.json
```

`--mode probe` sets:
- `teutonic-mixture-v2` dataset (35/5/35/25)
- `n_score=10000`, `train=6000`, `val=800`, `n_eval=1000`
- LoRA sweep: `r32` and `r64` with `dropout=0.05`, `epochs=0.5`
- **`--abort-if-mu-hat-nonpositive`** (auto-enabled): exits before merge if best μ̂ ≤ 0

Check: `iter_00/lora_sweep/sweep_results.json`, `eval_fast.json`, `submit_decision`.

## Stage 2 — Strong run (A100 80GB)

```bash
python scripts/mining/train_challenger.py \
  --work /root/teutonic-mining/strong \
  --bundle scripts/training_bundle \
  --mode strong \
  --profile a100-80gb \
  --sim-hotkey YOUR_HOTKEY \
  --sim-block-hash YOUR_BLOCK_HASH \
  --report-out /root/teutonic-mining/strong/verdict.json
```

`--mode strong` sets:
- `n_score=50000`, `train=24000`, `val=3000`, `n_eval=5000`
- LoRA sweep: r32, r64×2, r128 (with dropout/epochs in spec)
- **`--dual-eval`**: mixture local eval (primary) + validator-style eval (secondary)

A100 40GB: use `--profile a100-40gb` (micro_batch=1, grad_accum=32).

## Custom sweep (comma-separated)

```bash
--lora-sweep "r32:a64:lr2e-4:d0.05:e0.5,r64:a128:lr1e-4:d0.05:e0.8"
```

## Per-dataset curriculum override

```bash
--bucket-mix finewebedu=0.75:0.15:0.10 \
--bucket-mix automathtext-v2=0.45:0.45:0.10
```

## Submit gate (automatic)

| Verdict | Meaning |
|---|---|
| `DO_NOT_SUBMIT` | LCB floor failed or regression on major datasets |
| `PROMISING_NEEDS_MORE_EVAL` | Passes floor but `n_eval < 3000` |
| `READY_TO_MERGE` | Passes floor, below preferred margin |
| `READY_TO_UPLOAD` | `lcb >= 0.0035`, `mu_hat >= 0.0075`, hygiene OK |

Upload only with explicit approval:

```bash
--upload-repo NAMESPACE/Teutonic-Q3-4B-PREFIX-suffix \
--coldkey-prefix YOUR8CHAR \
--upload-approved
```

## Key artifacts

| File | Contents |
|---|---|
| `allocation_summary.json` | Weighted sample counts per dataset |
| `lora_sweep/sweep_results.json` | Per-config train/eval/LCB |
| `eval_mixture_final.json` | Per-dataset + mixture LCB |
| `eval_validator.json` | Validator-style eval (dual-eval mode) |
| `probe_abort.json` | Written if probe aborts (μ̂ ≤ 0) |
| `run_report.md` | Human-readable summary |
