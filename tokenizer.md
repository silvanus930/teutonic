Yes. For Qwen3 4B, you need to **retokenize raw FineWeb-Edu text using your exact Qwen3 tokenizer**, then save the result as Teutonic-compatible `.npy` shards.

FineWeb-Edu is published as Parquet/text data on Hugging Face, with smaller configs like `sample-10BT`, `sample-100BT`, and `sample-350BT`. For your A100 test, start with `sample-10BT`, not the full dataset. ([Hugging Face][1])

Also, rotate the Hugging Face token you pasted in your previous log.

## 1. Install requirements

```bash
pip install -U datasets transformers tokenizers numpy tqdm pyarrow huggingface_hub
```

## 2. Create converter script

Save this as:

```bash
nano /root/teutonic/scripts/mining/retokenize_fineweb_edu_qwen.py
```

Paste:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(
        description="Retokenize FineWeb-Edu with a local Qwen tokenizer into Teutonic-style .npy shards."
    )
    p.add_argument(
        "--tokenizer-dir",
        required=True,
        help="Local Qwen/Qwen3 base model folder containing tokenizer.json/tokenizer_config.json.",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for .npy shards and manifest.json.",
    )
    p.add_argument(
        "--dataset",
        default="HuggingFaceFW/fineweb-edu",
        help="HF dataset name.",
    )
    p.add_argument(
        "--config",
        default="sample-10BT",
        help="Dataset config. Good start: sample-10BT.",
    )
    p.add_argument(
        "--split",
        default="train",
        help="Dataset split.",
    )
    p.add_argument(
        "--seq-len",
        type=int,
        default=2048,
        help="Sequence length expected by Teutonic script.",
    )
    p.add_argument(
        "--sequences-per-shard",
        type=int,
        default=8192,
        help="Number of 2048-token sequences per .npy shard. 8192 ~= 16.7M tokens/shard.",
    )
    p.add_argument(
        "--max-shards",
        type=int,
        default=4,
        help="Stop after writing this many shards. Start small on A100.",
    )
    p.add_argument(
        "--text-column",
        default="text",
        help="Column containing raw text.",
    )
    p.add_argument(
        "--min-chars",
        type=int,
        default=200,
        help="Skip very short documents.",
    )
    p.add_argument(
        "--streaming",
        action="store_true",
        help="Use HF streaming mode. Recommended for FineWeb-Edu.",
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to tokenizer if needed.",
    )
    return p.parse_args()


def save_shard(out_dir: Path, shard_idx: int, rows: list[list[int]], seq_len: int) -> dict:
    arr = np.asarray(rows, dtype=np.uint32)
    if arr.ndim != 2 or arr.shape[1] != seq_len:
        raise RuntimeError(f"Bad shard shape: {arr.shape}, expected (?, {seq_len})")

    name = f"shard_{shard_idx:06d}.npy"
    path = out_dir / name
    np.save(path, arr)

    return {
        "key": name,
        "num_sequences": int(arr.shape[0]),
        "num_tokens": int(arr.shape[0] * arr.shape[1]),
        "dtype": "uint32",
        "shape": [int(arr.shape[0]), int(arr.shape[1])],
        "bytes": int(path.stat().st_size),
    }


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer from: {args.tokenizer_dir}")
    tok = AutoTokenizer.from_pretrained(
        args.tokenizer_dir,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )

    if tok.eos_token_id is None:
        raise RuntimeError("Tokenizer has no eos_token_id. Cannot safely separate documents.")

    vocab_size = len(tok)
    eos = int(tok.eos_token_id)

    print("Tokenizer OK")
    print("vocab_size:", vocab_size)
    print("eos_token_id:", eos)
    print("out_dir:", out_dir)

    print(f"Loading dataset: {args.dataset} / {args.config} / {args.split}")
    ds = load_dataset(
        args.dataset,
        name=args.config,
        split=args.split,
        streaming=args.streaming,
    )

    seq_len = args.seq_len
    target_rows_per_shard = args.sequences_per_shard

    token_buffer: list[int] = []
    shard_rows: list[list[int]] = []
    shards: list[dict] = []

    docs_seen = 0
    docs_used = 0
    tokens_seen = 0
    t0 = time.time()

    pbar = tqdm(desc="Retokenizing", unit="doc")

    for row in ds:
        docs_seen += 1
        pbar.update(1)

        text = row.get(args.text_column)
        if not text or not isinstance(text, str):
            continue

        text = text.strip()
        if len(text) < args.min_chars:
            continue

        ids = tok.encode(text, add_special_tokens=False)

        if not ids:
            continue

        # Safety check: should never fail if tokenizer matches model.
        mx = max(ids)
        if mx >= vocab_size or min(ids) < 0:
            raise RuntimeError(
                f"Tokenizer produced out-of-range token id. min={min(ids)} max={mx} vocab_size={vocab_size}"
            )

        token_buffer.extend(ids)
        token_buffer.append(eos)

        docs_used += 1
        tokens_seen += len(ids) + 1

        while len(token_buffer) >= seq_len:
            seq = token_buffer[:seq_len]
            del token_buffer[:seq_len]
            shard_rows.append(seq)

            if len(shard_rows) >= target_rows_per_shard:
                meta = save_shard(out_dir, len(shards), shard_rows, seq_len)
                shards.append(meta)
                shard_rows = []

                elapsed = max(time.time() - t0, 1e-6)
                print(
                    f"\nSaved {meta['key']} | "
                    f"shards={len(shards)}/{args.max_shards} | "
                    f"docs_seen={docs_seen} docs_used={docs_used} | "
                    f"tokens={tokens_seen:,} | "
                    f"tok/s={tokens_seen / elapsed:,.0f}"
                )

                manifest = {
                    "dataset": args.dataset,
                    "config": args.config,
                    "split": args.split,
                    "tokenizer_dir": args.tokenizer_dir,
                    "seq_len": seq_len,
                    "eos_token_id": eos,
                    "vocab_size": vocab_size,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "shards": shards,
                }
                (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

                if len(shards) >= args.max_shards:
                    pbar.close()
                    print("Done.")
                    print("Manifest:", out_dir / "manifest.json")
                    return

    # Save final partial shard only if it has useful size.
    if shard_rows:
        meta = save_shard(out_dir, len(shards), shard_rows, seq_len)
        shards.append(meta)

    manifest = {
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "tokenizer_dir": args.tokenizer_dir,
        "seq_len": seq_len,
        "eos_token_id": eos,
        "vocab_size": vocab_size,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "shards": shards,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    pbar.close()

    print("Done.")
    print("Manifest:", out_dir / "manifest.json")
    print("Shards:", len(shards))


if __name__ == "__main__":
    main()
```

Make executable:

```bash
chmod +x /root/teutonic/scripts/mining/retokenize_fineweb_edu_qwen.py
```

## 3. Run small first

Use your local Qwen3 model/tokenizer folder:

```bash
TOKENIZER_DIR="/root/teutonic/s1-work/main"
OUT="/root/teutonic/qwen3-fineweb-edu-npy"

python /root/teutonic/scripts/mining/retokenize_fineweb_edu_qwen.py \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --out-dir "$OUT" \
  --config sample-10BT \
  --streaming \
  --seq-len 2048 \
  --sequences-per-shard 4096 \
  --max-shards 2
```

This will create:

```text
/root/teutonic/qwen3-fineweb-edu-npy/
├── manifest.json
├── shard_000000.npy
└── shard_000001.npy
```

Each shard will be shaped like:

```text
(4096, 2048)
```

That means around:

```text
4096 × 2048 = 8,388,608 tokens per shard
```

## 4. Verify the token range

Run:

```bash
python - <<'PY'
import numpy as np
from transformers import AutoConfig, AutoTokenizer

MODEL_DIR = "/root/teutonic/s1-work/main"
SHARD = "/root/teutonic/qwen3-fineweb-edu-npy/shard_000000.npy"

cfg = AutoConfig.from_pretrained(MODEL_DIR)
tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)

arr = np.load(SHARD, mmap_mode="r")

print("model vocab_size:", cfg.vocab_size)
print("tokenizer len:", len(tok))
print("shard shape:", arr.shape)
print("dtype:", arr.dtype)
print("min token:", int(arr.min()))
print("max token:", int(arr.max()))
print("bad >= vocab:", int((arr >= cfg.vocab_size).sum()))
PY
```

Expected:

```text
bad >= vocab: 0
```

## 5. Patch training script to use local retokenized manifest

Your current script always downloads:

```text
https://s3.hippius.com/teutonic-sn3/dataset/v2/manifest.json
```

and shard URLs. We need it to accept local shards.

Open:

```bash
nano /root/teutonic/scripts/mining/train_challenger.py
```

Find this function:

```python
def download_shard(shard_key: str, out: Path) -> Path:
    if out.exists() and out.stat().st_size > 1024:
        log.info("shard cached: %s (%.1f GB)", out, out.stat().st_size / 1e9)
        return out
    url = f"{HIPPIUS_BASE}/{shard_key}"
    log.info("downloading %s -> %s", url, out)
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["curl", "-fsSL", "-o", str(out), url])
    return out
```

Replace with:

```python
def download_shard(shard_key: str, out: Path) -> Path:
    # Local shard support for retokenized datasets.
    local_path = Path(shard_key)
    if local_path.exists():
        log.info("using local shard: %s", local_path)
        return local_path

    if out.exists() and out.stat().st_size > 1024:
        log.info("shard cached: %s (%.1f GB)", out, out.stat().st_size / 1e9)
        return out

    url = f"{HIPPIUS_BASE}/{shard_key}"
    log.info("downloading %s -> %s", url, out)
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["curl", "-fsSL", "-o", str(out), url])
    return out
```

Find:

```python
def fetch_manifest(cache: Path) -> dict:
    p = cache / "manifest.json"
    if not p.exists():
        url = f"{HIPPIUS_BASE}/dataset/v2/manifest.json"
        log.info("downloading manifest from %s", url)
        cache.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(["curl", "-fsSL", "-o", str(p), url])
    return json.loads(p.read_text())
```

Replace with:

```python
def fetch_manifest(cache: Path) -> dict:
    local_manifest = os.environ.get("LOCAL_DATASET_MANIFEST", "")
    if local_manifest:
        p = Path(local_manifest)
        if not p.exists():
            raise FileNotFoundError(f"LOCAL_DATASET_MANIFEST does not exist: {p}")
        log.info("loading local dataset manifest: %s", p)
        m = json.loads(p.read_text())

        # Convert relative shard names to absolute local paths.
        base = p.parent
        for s in m.get("shards", []):
            key = s["key"]
            kp = Path(key)
            if not kp.is_absolute():
                s["key"] = str(base / key)
        return m

    p = cache / "manifest.json"
    if not p.exists():
        url = f"{HIPPIUS_BASE}/dataset/v2/manifest.json"
        log.info("downloading manifest from %s", url)
        cache.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(["curl", "-fsSL", "-o", str(p), url])
    return json.loads(p.read_text())
```

## 6. Run training with local Qwen-tokenized FineWeb-Edu

```bash
export LOCAL_KING_DIR="/root/teutonic/s1-work/main"
export LOCAL_DATASET_MANIFEST="/root/teutonic/qwen3-fineweb-edu-npy/manifest.json"
export HF_XET_HIGH_PERFORMANCE=1

python /root/teutonic/scripts/mining/train_challenger.py \
  --work /root/teutonic/s1-work-qwen-fwe \
  --bundle /root/teutonic/scripts/training_bundle \
  --n-gpus 1 \
  --n-shards 1 \
  --shard-start 0 \
  --eval-shard 1 \
  --n-eval 200 \
  --n-score 500 \
  --train-per-iter 500 \
  --val-size 50 \
  --max-iters 1 \
  --micro-batch 1 \
  --grad-accum 16 \
  --lr 1e-4 \
  --epochs 1 \
  --lora-r 8 \
  --lora-alpha 16
```

Important: because you generated only 2 shards, use:

```bash
--n-shards 1
--shard-start 0
--eval-shard 1
```

## 7. Later, generate more shards

After the pipeline works:

```bash
python /root/teutonic/scripts/mining/retokenize_fineweb_edu_qwen.py \
  --tokenizer-dir "/root/teutonic/s1-work/main" \
  --out-dir "/root/teutonic/qwen3-fineweb-edu-npy-big" \
  --config sample-10BT \
  --streaming \
  --seq-len 2048 \
  --sequences-per-shard 8192 \
  --max-shards 16
```

Then train with more data:

```bash
export LOCAL_DATASET_MANIFEST="/root/teutonic/qwen3-fineweb-edu-npy-big/manifest.json"

python /root/teutonic/scripts/mining/train_challenger.py \
  --work /root/teutonic/s1-work-qwen-fwe-big \
  --bundle /root/teutonic/scripts/training_bundle \
  --n-gpus 1 \
  --n-shards 4 \
  --shard-start 0 \
  --eval-shard 8 \
  --n-eval 500 \
  --n-score 2000 \
  --train-per-iter 2000 \
  --val-size 200 \
  --max-iters 1 \
  --micro-batch 1 \
  --grad-accum 16 \
  --lr 1e-4 \
  --epochs 1 \
  --lora-r 8 \
  --lora-alpha 16
```

For your A100, start small first. The goal is to confirm: **no token mismatch, no CUDA assert, training completes, merge completes, eval runs**.

[1]: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu?utm_source=chatgpt.com "HuggingFaceFW/fineweb-edu · Datasets at ..."
