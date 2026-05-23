#!/usr/bin/env python3
"""Retokenize FineWeb-Edu with a local Qwen3 tokenizer into Teutonic-style .npy shards.

Produces 2D uint32 shards (n_sequences, seq_len) plus manifest.json for use with
train_challenger.py via LOCAL_DATASET_MANIFEST.

Usage:
    python scripts/mining/retokenize_fineweb_edu_qwen.py \\
        --tokenizer-dir /path/to/qwen3-model \\
        --out-dir /path/to/qwen3-fineweb-edu-npy \\
        --config sample-10BT --streaming --max-shards 2
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("retokenize_fineweb")

# Hippius v2 pretokenized shards: 536_870_912 tokens each -> 262144 x 2048 sequences.
PROD_SEQUENCES_PER_SHARD = 262144
CHECKPOINT_NAME = "retokenize_checkpoint.json"


def parse_args():
    p = argparse.ArgumentParser(
        description="Retokenize FineWeb-Edu with a local Qwen tokenizer into Teutonic-style .npy shards.",
    )
    p.add_argument("--tokenizer-dir", required=True,
                   help="Local Qwen3 folder with tokenizer.json and config.json.")
    p.add_argument("--out-dir", required=True,
                   help="Output directory for .npy shards and manifest.json.")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--config", default="sample-10BT",
                   help="HF config; sample-10BT is a good start on one GPU.")
    p.add_argument("--split", default="train")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--sequences-per-shard", type=int, default=8192,
                   help=f"Sequences per shard. Production (Hippius v2) uses {PROD_SEQUENCES_PER_SHARD}.")
    p.add_argument("--prod", action="store_true",
                   help=f"Use production shard size ({PROD_SEQUENCES_PER_SHARD} sequences, ~2 GiB/shard).")
    p.add_argument("--max-shards", type=int, default=4,
                   help="Total shards to write (including any resumed shards).")
    p.add_argument("--resume", action="store_true",
                   help="Continue in --out-dir using checkpoint + existing manifest shards.")
    p.add_argument("--text-column", default="text")
    p.add_argument("--min-chars", type=int, default=200)
    p.add_argument("--streaming", action="store_true",
                   help="Use HF streaming (recommended for FineWeb-Edu).")
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args()


def save_shard(out_dir: Path, shard_idx: int, rows: list[list[int]], seq_len: int) -> dict:
    arr = np.asarray(rows, dtype=np.uint32)
    if arr.ndim != 2 or arr.shape[1] != seq_len:
        raise RuntimeError(f"bad shard shape {arr.shape}, expected (?, {seq_len})")
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


def write_manifest(out_dir: Path, meta: dict) -> None:
    (out_dir / "manifest.json").write_text(json.dumps(meta, indent=2))


def load_resume_state(out_dir: Path, args) -> tuple[list[dict], int, int, int]:
    """Return (existing_shards, docs_seen, docs_used, tokens_seen)."""
    manifest_path = out_dir / "manifest.json"
    ckpt_path = out_dir / CHECKPOINT_NAME
    shards: list[dict] = []
    docs_seen = docs_used = tokens_seen = 0
    if manifest_path.is_file():
        shards = json.loads(manifest_path.read_text()).get("shards") or []
    if args.resume and ckpt_path.is_file():
        ckpt = json.loads(ckpt_path.read_text())
        docs_seen = int(ckpt.get("docs_seen", 0))
        docs_used = int(ckpt.get("docs_used", 0))
        tokens_seen = int(ckpt.get("tokens_seen", 0))
        log.info(
            "resume: %d existing shard(s), skipping %d streamed docs",
            len(shards), docs_seen,
        )
    elif args.resume and shards:
        raise RuntimeError(
            f"--resume requested but no {CHECKPOINT_NAME} in {out_dir}. "
            "Use a fresh --out-dir or rerun without --resume."
        )
    return shards, docs_seen, docs_used, tokens_seen


def save_checkpoint(out_dir: Path, docs_seen: int, docs_used: int, tokens_seen: int) -> None:
    (out_dir / CHECKPOINT_NAME).write_text(json.dumps({
        "docs_seen": docs_seen,
        "docs_used": docs_used,
        "tokens_seen": tokens_seen,
    }, indent=2))


def resolve_tokenizer_dir(path: str) -> Path:
    """Resolve a local model/tokenizer tree; HF hub rejects bare paths otherwise."""
    p = Path(path).expanduser().resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"tokenizer dir does not exist: {p}")
    needed = ("config.json",)
    tok_files = ("tokenizer.json", "tokenizer_config.json")
    if not (p / "config.json").is_file():
        raise FileNotFoundError(
            f"{p} is missing config.json — point --tokenizer-dir at a full HF model "
            "folder (e.g. s1-work/king or s1-work/main), not a weights-only subdir."
        )
    if not any((p / f).is_file() for f in tok_files):
        raise FileNotFoundError(
            f"{p} has no tokenizer.json / tokenizer_config.json"
        )
    return p


def main():
    args = parse_args()
    if args.prod:
        args.sequences_per_shard = PROD_SEQUENCES_PER_SHARD
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok_dir = resolve_tokenizer_dir(args.tokenizer_dir)
    log.info("loading tokenizer from %s", tok_dir)
    local_kw = {"local_files_only": True, "trust_remote_code": args.trust_remote_code}
    tok = AutoTokenizer.from_pretrained(str(tok_dir), use_fast=True, **local_kw)
    cfg = AutoConfig.from_pretrained(str(tok_dir), **local_kw)
    if tok.eos_token_id is None:
        raise RuntimeError("tokenizer has no eos_token_id")

    model_vocab = int(getattr(cfg, "vocab_size", 0) or len(tok))
    eos = int(tok.eos_token_id)
    log.info("model vocab_size=%d tokenizer len=%d eos=%d", model_vocab, len(tok), eos)

    log.info("loading dataset %s / %s / %s (streaming=%s)",
             args.dataset, args.config, args.split, args.streaming)
    ds = load_dataset(args.dataset, name=args.config, split=args.split, streaming=args.streaming)

    seq_len = args.seq_len
    target_rows = args.sequences_per_shard
    token_buffer: list[int] = []
    shard_rows: list[list[int]] = []
    shards, docs_seen, docs_used, tokens_seen = load_resume_state(out_dir, args)
    skip_docs = docs_seen if args.resume else 0
    if len(shards) >= args.max_shards:
        log.info("already have %d shard(s) >= --max-shards %d", len(shards), args.max_shards)
        return

    log.info(
        "target: %d shards x %d seq x %d tokens (~%.2f GiB/shard, ~%.1f GiB total)",
        args.max_shards, target_rows, seq_len,
        target_rows * seq_len * 4 / (1024 ** 3),
        args.max_shards * target_rows * seq_len * 4 / (1024 ** 3),
    )
    t0 = time.time()
    pbar = tqdm(desc="retokenizing", unit="doc", initial=skip_docs)

    def flush_shard():
        nonlocal shard_rows
        meta = save_shard(out_dir, len(shards), shard_rows, seq_len)
        shards.append(meta)
        shard_rows = []
        save_checkpoint(out_dir, docs_seen, docs_used, tokens_seen)
        elapsed = max(time.time() - t0, 1e-6)
        log.info(
            "saved %s | shards=%d/%d docs=%d/%d tokens=%s tok/s=%.0f",
            meta["key"], len(shards), args.max_shards,
            docs_used, docs_seen, f"{tokens_seen:,}", tokens_seen / elapsed,
        )
        manifest = {
            "version": "local-qwen-fineweb",
            "dataset": args.dataset,
            "config": args.config,
            "split": args.split,
            "tokenizer": str(tok_dir),
            "tokenizer_dir": str(tok_dir),
            "seq_len": seq_len,
            "sequences_per_shard": target_rows,
            "eos_token_id": eos,
            "vocab_size": model_vocab,
            "dtype": "uint32",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_shards": len(shards),
            "shards": shards,
        }
        write_manifest(out_dir, manifest)

    for row in ds:
        if docs_seen < skip_docs:
            docs_seen += 1
            if docs_seen % 50000 == 0:
                pbar.update(50000)
            continue
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
        mx, mn = max(ids), min(ids)
        if mn < 0 or mx >= model_vocab:
            raise RuntimeError(
                f"out-of-range token id min={mn} max={mx} model_vocab={model_vocab}"
            )

        token_buffer.extend(ids)
        token_buffer.append(eos)
        docs_used += 1
        tokens_seen += len(ids) + 1

        while len(token_buffer) >= seq_len:
            shard_rows.append(token_buffer[:seq_len])
            del token_buffer[:seq_len]
            if len(shard_rows) >= target_rows:
                flush_shard()
                if len(shards) >= args.max_shards:
                    pbar.close()
                    log.info("done: %s", out_dir / "manifest.json")
                    return

    if shard_rows and len(shards) < args.max_shards:
        flush_shard()
    save_checkpoint(out_dir, docs_seen, docs_used, tokens_seen)
    pbar.close()
    log.info("done: %d shard(s) -> %s", len(shards), out_dir / "manifest.json")


if __name__ == "__main__":
    main()
