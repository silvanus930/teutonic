#!/usr/bin/env python3
"""Retokenize FineWeb-Edu with a local Qwen tokenizer into reusable Teutonic .npy shards.

Produces a structured local dataset with high-quality, mixed, deduplicated shards:
  <out-dir>/
    manifest.json               # main index (compatible with train_challenger.py)
    train/
      shard_000000.npy          # (rows_per_shard, seq_len) uint32
      shard_000001.npy
      ...
    eval/
      holdout_000000.npy        # holdout shards — never used for training
      ...
    metadata/
      tokenizer_info.json
      build_stats.json
      shard_stats.jsonl
    score_cache/                # populated by train_challenger.py

Key features:
  - Uses HF dataset.shuffle(seed, buffer_size) for deterministic, unbiased ordering
  - Quality filtering: min chars, alpha ratio, repeat ratio, URL/boilerplate detection
  - Atomic writes (temp → rename) — crash-safe
  - Resume support: skips shards that already exist with correct shape/dtype
  - Train/eval split: eval shards are never used for training

Build production shards:
  python scripts/mining/retokenize_fineweb_edu_qwen.py \\
      --tokenizer-dir /root/teutonic/s1-work/king \\
      --out-dir /data/fineweb_edu_qwen3_2048 \\
      --dataset HuggingFaceFW/fineweb-edu \\
      --config sample-10BT \\
      --seq-len 2048 \\
      --prod \\
      --max-train-shards 16 \\
      --max-eval-shards 2 \\
      --streaming \\
      --resume

Validate existing shards:
  python scripts/mining/retokenize_fineweb_edu_qwen.py \\
      --out-dir /data/fineweb_edu_qwen3_2048 \\
      --validate-only

Expand to 32 train + 4 eval shards from sample-100BT:
  python scripts/mining/retokenize_fineweb_edu_qwen.py \\
      --tokenizer-dir /root/teutonic/s1-work/king \\
      --out-dir /data/fineweb_edu_qwen3_2048 \\
      --dataset HuggingFaceFW/fineweb-edu \\
      --config sample-100BT \\
      --max-train-shards 32 \\
      --max-eval-shards 4 \\
      --streaming \\
      --prod \\
      --resume
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("retokenize_fineweb")

PROD_ROWS_PER_SHARD = 262144       # 262144 * 2048 = ~536M tokens/shard (~2 GiB)
PROD_EVAL_ROWS_PER_SHARD = 16384   # ~32M tokens/eval shard (~128 MiB)

# Compiled regex patterns for quality filtering
_RE_URL = re.compile(
    r"https?://[^\s]{15,}|www\.[^\s]{10,}",
    re.IGNORECASE,
)
_RE_NAV = re.compile(
    r"(click here|skip to|jump to|back to top|read more|next page|"
    r"previous page|menu|navigation|breadcrumb|cookie policy|privacy policy)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Retokenize FineWeb-Edu with a local Qwen tokenizer into .npy shards.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Tokenizer / dataset
    p.add_argument("--tokenizer-dir", default="",
                   help="Local model folder (config.json + tokenizer files). "
                        "Required for build mode; omit with --validate-only.")
    p.add_argument("--out-dir", required=True,
                   help="Output directory for shards, manifest.json, metadata/.")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--config", default="sample-10BT",
                   help="HF dataset config: sample-10BT, sample-100BT, CC-MAIN-2024-10, etc.")
    p.add_argument("--split", default="train")
    p.add_argument("--text-column", default="text")

    # Shard geometry
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--rows-per-shard", type=int, default=8192,
                   help=f"Sequences per train shard. Production: {PROD_ROWS_PER_SHARD}.")
    p.add_argument("--eval-rows-per-shard", type=int, default=0,
                   help="Sequences per eval shard. 0 = same as --rows-per-shard.")
    p.add_argument("--prod", action="store_true",
                   help=f"Use production shard sizes "
                        f"({PROD_ROWS_PER_SHARD} train, {PROD_EVAL_ROWS_PER_SHARD} eval rows).")
    p.add_argument("--max-train-shards", type=int, default=4)
    p.add_argument("--max-eval-shards", type=int, default=1)

    # Shuffling / mixing
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for deterministic HF shuffle and document mixing.")
    p.add_argument("--shuffle-buffer-size", type=int, default=10000,
                   help="HF dataset shuffle buffer size (higher = better mixing, more RAM). "
                        "Applied before tokenization. 0 = no shuffle.")
    # Internal post-tokenization mixing buffer (for extra mixing within token buffer)
    p.add_argument("--mix-buffer", type=int, default=0,
                   help="Post-tokenization document mixing buffer size. "
                        "Usually 0 since HF shuffle handles mixing.")

    # Quality filtering
    p.add_argument("--quality-filter", action="store_true", default=True,
                   help="Enable quality filtering (default: on).")
    p.add_argument("--no-quality-filter", action="store_false", dest="quality_filter",
                   help="Disable all quality filtering.")
    p.add_argument("--min-text-chars", type=int, default=200,
                   help="Minimum character count (after strip). Short docs are skipped.")
    p.add_argument("--max-text-chars", type=int, default=0,
                   help="Maximum character count. 0 = no limit.")
    p.add_argument("--max-repeat-ratio", type=float, default=0.30,
                   help="Max fraction of repeated consecutive characters (e.g. 'aaaa'). "
                        "Docs above this are rejected.")
    p.add_argument("--min-alpha-ratio", type=float, default=0.40,
                   help="Minimum fraction of alphabetic characters in the text. "
                        "Docs below this (heavy symbols/numbers/URLs) are rejected. "
                        "0.40 avoids over-filtering math/code/science FineWeb-Edu pages.")
    p.add_argument("--max-url-density", type=float, default=0.10,
                   help="Max fraction of text covered by URL matches. 0 = no URL filtering.")
    p.add_argument("--max-nav-matches", type=int, default=5,
                   help="Max navigation/boilerplate phrase count before rejecting. "
                        "0 = no navigation filtering.")

    # Streaming / resume
    p.add_argument("--streaming", action="store_true",
                   help="Use HF streaming mode (recommended for FineWeb-Edu).")
    p.add_argument("--resume", action="store_true",
                   help="Skip shards that already exist with the correct shape.")
    p.add_argument("--trust-remote-code", action="store_true")

    # Validate only
    p.add_argument("--validate-only", action="store_true",
                   help="Check manifest + shard shapes/dtypes and exit. No tokenization.")

    # Backward-compat aliases
    p.add_argument("--sequences-per-shard", type=int, default=0,
                   help="[deprecated] Use --rows-per-shard instead.")
    p.add_argument("--max-shards", type=int, default=0,
                   help="[deprecated] Use --max-train-shards instead.")
    p.add_argument("--shuffle-buffer", type=int, default=0,
                   help="[deprecated] Use --shuffle-buffer-size instead.")
    p.add_argument("--min-chars", type=int, default=0,
                   help="[deprecated] Use --min-text-chars instead.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------

def resolve_tokenizer_dir(path: str) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"tokenizer dir not found: {p}")
    if not (p / "config.json").is_file():
        raise FileNotFoundError(
            f"{p} missing config.json — point --tokenizer-dir at a full HF model folder."
        )
    tok_files = ("tokenizer.json", "tokenizer_config.json")
    if not any((p / f).is_file() for f in tok_files):
        raise FileNotFoundError(f"{p} has no tokenizer.json / tokenizer_config.json")
    return p


def tokenizer_hash(tok_dir: Path) -> str:
    """Stable hash of the tokenizer vocabulary for cache keying."""
    h = hashlib.sha256()
    for name in sorted(["tokenizer.json", "tokenizer_config.json", "vocab.json",
                        "merges.txt", "special_tokens_map.json"]):
        f = tok_dir / name
        if f.is_file():
            h.update(name.encode())
            h.update(f.read_bytes()[:65536])  # first 64K for speed
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Atomic I/O
# ---------------------------------------------------------------------------

def atomic_save_npy(dest: Path, arr: np.ndarray) -> None:
    """Write arr to a temp file then atomically rename — crash-safe."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".tmp.npy")
    try:
        os.close(fd)
        np.save(tmp, arr)
        shutil.move(tmp, str(dest))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_json_atomic(dest: Path, obj: dict) -> None:
    tmp = dest.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(obj, indent=2))
    shutil.move(str(tmp), str(dest))


# ---------------------------------------------------------------------------
# Shard validity check
# ---------------------------------------------------------------------------

def shard_is_valid(path: Path, expected_rows: int, seq_len: int) -> bool:
    if not path.is_file() or path.stat().st_size < 1024:
        return False
    try:
        arr = np.load(path, mmap_mode="r")
        return (
            arr.shape == (expected_rows, seq_len)
            and arr.dtype == np.dtype("uint32")
        )
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Quality filtering
# ---------------------------------------------------------------------------

def check_quality(text: str, args: argparse.Namespace) -> tuple[bool, str]:
    """Return (accept, rejection_reason). Empty reason = accepted."""
    if not args.quality_filter:
        return True, ""

    n = len(text)

    # Min length
    if n < args.min_text_chars:
        return False, "too_short"

    # Max length
    if args.max_text_chars > 0 and n > args.max_text_chars:
        return False, "too_long"

    # Alphabetic ratio
    alpha_count = sum(1 for c in text if c.isalpha())
    alpha_ratio = alpha_count / max(1, n)
    if alpha_ratio < args.min_alpha_ratio:
        return False, "low_alpha_ratio"

    # Consecutive character repeat ratio
    if args.max_repeat_ratio > 0 and n > 1:
        repeats = sum(1 for i in range(1, n) if text[i] == text[i - 1])
        if repeats / (n - 1) > args.max_repeat_ratio:
            return False, "high_repeat_ratio"

    # URL density
    if args.max_url_density > 0:
        url_chars = sum(len(m.group()) for m in _RE_URL.finditer(text))
        if url_chars / max(1, n) > args.max_url_density:
            return False, "high_url_density"

    # Navigation / boilerplate
    if args.max_nav_matches > 0:
        nav_hits = len(_RE_NAV.findall(text))
        if nav_hits > args.max_nav_matches:
            return False, "navigation_boilerplate"

    return True, ""


# ---------------------------------------------------------------------------
# Manifest validation (--validate-only)
# ---------------------------------------------------------------------------

def validate_manifest(out_dir: Path) -> dict:
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest.json not found in {out_dir}")
    manifest = json.loads(manifest_path.read_text())
    seq_len = manifest.get("seq_len", 2048)
    shards = manifest.get("shards", [])

    errors: list[str] = []
    warnings_list: list[str] = []
    ok = 0
    for i, s in enumerate(shards):
        key = s.get("key", "")
        kp = Path(key) if Path(key).is_absolute() else (out_dir / key).resolve()
        split = s.get("split", "unknown")

        if not kp.is_file():
            errors.append(f"shard {i} ({split}/{key}): file missing")
            continue
        try:
            arr = np.load(kp, mmap_mode="r")
        except Exception as e:
            errors.append(f"shard {i} ({split}/{key}): unreadable — {e}")
            continue

        exp_rows = s.get("num_sequences") or s.get("rows")
        exp_dtype = s.get("dtype", "uint32")

        if arr.ndim != 2:
            errors.append(f"shard {i} ({split}/{key}): ndim={arr.ndim} (expected 2)")
        elif arr.shape[1] != seq_len:
            errors.append(f"shard {i} ({split}/{key}): shape {arr.shape}, expected seq_len={seq_len}")
        elif exp_rows and arr.shape[0] != exp_rows:
            errors.append(f"shard {i} ({split}/{key}): {arr.shape[0]} rows, expected {exp_rows}")
        elif arr.dtype != np.dtype(exp_dtype):
            errors.append(f"shard {i} ({split}/{key}): dtype={arr.dtype}, expected {exp_dtype}")
        else:
            ok += 1
        del arr

    # Check metadata
    tok_info_path = out_dir / "metadata" / "tokenizer_info.json"
    if not tok_info_path.is_file():
        warnings_list.append("metadata/tokenizer_info.json not found")

    train_shards = [s for s in shards if s.get("split") == "train"]
    eval_shards = [s for s in shards if s.get("split") == "eval"]
    if not train_shards:
        warnings_list.append("no train shards in manifest")
    if not eval_shards:
        warnings_list.append("no eval shards in manifest (holdout not built yet)")

    return {
        "total_shards": len(shards),
        "train_shards": len(train_shards),
        "eval_shards": len(eval_shards),
        "ok": ok,
        "failed": len(shards) - ok,
        "errors": errors,
        "warnings": warnings_list,
        "passed": len(errors) == 0,
        "manifest": str(manifest_path),
        "seq_len": seq_len,
    }


# ---------------------------------------------------------------------------
# Shard save helper
# ---------------------------------------------------------------------------

def save_shard_entry(
    out_dir: Path,
    rows: list[list[int]],
    shard_idx: int,
    split: str,
    seq_len: int,
) -> dict:
    prefix = "shard" if split == "train" else "holdout"
    rel = f"{split}/{prefix}_{shard_idx:06d}.npy"
    dest = out_dir / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(rows, dtype=np.uint32)
    if arr.ndim != 2 or arr.shape[1] != seq_len:
        raise RuntimeError(f"bad shard shape {arr.shape}, expected (?, {seq_len})")
    atomic_save_npy(dest, arr)
    token_count = int(arr.shape[0] * arr.shape[1])
    return {
        "key": rel,
        "split": split,
        "num_sequences": int(arr.shape[0]),
        "rows": int(arr.shape[0]),
        "seq_len": seq_len,
        "num_tokens": token_count,
        "token_count": token_count,
        "dtype": "uint32",
        "shape": [int(arr.shape[0]), int(arr.shape[1])],
        "bytes": int(dest.stat().st_size),
    }


# ---------------------------------------------------------------------------
# Manifest builder
# ---------------------------------------------------------------------------

def build_manifest(
    out_dir: Path,
    train_entries: list[dict],
    eval_entries: list[dict],
    tok_info: dict,
    args: argparse.Namespace,
) -> dict:
    all_entries = train_entries + eval_entries
    return {
        "version": "local-qwen-v3",
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "seq_len": tok_info["seq_len"],
        "rows_per_shard": args.rows_per_shard,
        "eval_rows_per_shard": args.eval_rows_per_shard,
        "seed": args.seed,
        "shuffle_buffer_size": args.shuffle_buffer_size,
        "quality_filter": args.quality_filter,
        "quality_filter_settings": {
            "min_text_chars": args.min_text_chars,
            "max_repeat_ratio": args.max_repeat_ratio,
            "min_alpha_ratio": args.min_alpha_ratio,
            "max_url_density": args.max_url_density,
            "max_nav_matches": args.max_nav_matches,
        },
        # "tokenizer" field is read by train_challenger.pretokenized_incompatible_with_king
        "tokenizer": tok_info["tokenizer_dir"],
        "tokenizer_dir": tok_info["tokenizer_dir"],
        "tokenizer_hash": tok_info.get("tokenizer_hash", ""),
        "eos_token_id": tok_info["eos_token_id"],
        "vocab_size": tok_info["vocab_size"],
        "dtype": "uint32",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_shards": len(all_entries),
        "total_train_shards": len(train_entries),
        "total_eval_shards": len(eval_entries),
        # "shards" is iterated by train_challenger.py
        "shards": all_entries,
        "train_shards": train_entries,
        "eval_shards": eval_entries,
    }


def write_metadata(
    out_dir: Path,
    tok_info: dict,
    train_entries: list[dict],
    eval_entries: list[dict],
    build_stats: dict,
) -> None:
    meta_dir = out_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(meta_dir / "tokenizer_info.json", tok_info)
    write_json_atomic(meta_dir / "build_stats.json", build_stats)
    stats_path = meta_dir / "shard_stats.jsonl"
    tmp = meta_dir / "shard_stats.jsonl.tmp"
    with open(tmp, "w") as f:
        for e in train_entries + eval_entries:
            f.write(json.dumps(e) + "\n")
    shutil.move(str(tmp), str(stats_path))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Backward-compat aliases
    if args.sequences_per_shard > 0:
        args.rows_per_shard = args.sequences_per_shard
    if args.max_shards > 0:
        args.max_train_shards = args.max_shards
    if args.shuffle_buffer > 0 and args.shuffle_buffer_size == 10000:
        args.shuffle_buffer_size = args.shuffle_buffer
    if args.min_chars > 0 and args.min_text_chars == 200:
        args.min_text_chars = args.min_chars
    if args.prod:
        args.rows_per_shard = PROD_ROWS_PER_SHARD
        if args.eval_rows_per_shard <= 0:
            args.eval_rows_per_shard = PROD_EVAL_ROWS_PER_SHARD
    if args.eval_rows_per_shard <= 0:
        args.eval_rows_per_shard = args.rows_per_shard

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("train", "eval", "metadata", "score_cache"):
        (out_dir / sub).mkdir(exist_ok=True)

    # ----------------------------------------------------------------
    # --validate-only
    # ----------------------------------------------------------------
    if args.validate_only:
        log.info("validate-only: checking %s", out_dir)
        result = validate_manifest(out_dir)
        ok_str = "PASS" if result["passed"] else "FAIL"
        log.info(
            "%s: %d/%d shards OK | train=%d eval=%d",
            ok_str, result["ok"], result["total_shards"],
            result.get("train_shards", 0), result.get("eval_shards", 0),
        )
        if result.get("warnings"):
            for w in result["warnings"]:
                log.warning("  warning: %s", w)
        if not result["passed"]:
            for e in result["errors"]:
                log.error("  error: %s", e)
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result["passed"] else 1)

    # ----------------------------------------------------------------
    # Build mode — tokenizer required
    # ----------------------------------------------------------------
    if not args.tokenizer_dir:
        raise ValueError(
            "--tokenizer-dir is required for build mode. "
            "Use --validate-only to check existing shards."
        )

    from transformers import AutoConfig, AutoTokenizer

    tok_dir = resolve_tokenizer_dir(args.tokenizer_dir)
    log.info("loading tokenizer from %s", tok_dir)
    tok = AutoTokenizer.from_pretrained(
        str(tok_dir), use_fast=True, local_files_only=True,
        trust_remote_code=args.trust_remote_code,
    )
    cfg = AutoConfig.from_pretrained(
        str(tok_dir), local_files_only=True,
        trust_remote_code=args.trust_remote_code,
    )
    if tok.eos_token_id is None:
        raise RuntimeError("tokenizer has no eos_token_id")

    model_vocab = int(getattr(cfg, "vocab_size", 0) or len(tok))
    eos = int(tok.eos_token_id)
    pad_id = int(tok.pad_token_id) if tok.pad_token_id is not None else None
    max_len = int(getattr(tok, "model_max_length", 0) or 0) or None
    tok_hash = tokenizer_hash(tok_dir)

    log.info(
        "tokenizer: vocab=%d eos=%d pad=%s model_max_length=%s hash=%s",
        model_vocab, eos, pad_id, max_len, tok_hash[:16],
    )

    tok_info = {
        "tokenizer_dir": str(tok_dir),
        "tokenizer_name": tok_dir.name,
        "tokenizer_hash": tok_hash,
        "vocab_size": model_vocab,
        "eos_token_id": eos,
        "pad_token_id": pad_id,
        "model_max_length": max_len,
        "seq_len": args.seq_len,
    }

    # ----------------------------------------------------------------
    # Resume: discover existing valid shards
    # ----------------------------------------------------------------
    train_entries: list[dict] = []
    eval_entries: list[dict] = []
    if args.resume and (out_dir / "manifest.json").is_file():
        try:
            old = json.loads((out_dir / "manifest.json").read_text())
            for e in old.get("train_shards") or old.get("shards") or []:
                if e.get("split", "train") != "train":
                    continue
                p = (out_dir / e["key"]).resolve()
                exp_rows = e.get("num_sequences") or e.get("rows", args.rows_per_shard)
                if shard_is_valid(p, exp_rows, args.seq_len):
                    log.info("resume: valid train shard: %s", e["key"])
                    train_entries.append(e)
            for e in old.get("eval_shards") or []:
                p = (out_dir / e["key"]).resolve()
                exp_rows = e.get("num_sequences") or e.get("rows", args.eval_rows_per_shard)
                if shard_is_valid(p, exp_rows, args.seq_len):
                    log.info("resume: valid eval shard: %s", e["key"])
                    eval_entries.append(e)
            log.info(
                "resume: found %d train + %d eval shards already valid",
                len(train_entries), len(eval_entries),
            )
        except Exception as exc:
            log.warning("could not parse existing manifest for resume: %s", exc)

    need_train = args.max_train_shards - len(train_entries)
    need_eval = args.max_eval_shards - len(eval_entries)

    if need_train <= 0 and need_eval <= 0:
        log.info(
            "all %d train + %d eval shards exist — nothing to build",
            len(train_entries), len(eval_entries),
        )
        _finalize(out_dir, train_entries, eval_entries, tok_info, args,
                  t0=time.time(), docs_seen=0, docs_accepted=0,
                  tokens_seen=0, reject_counts={})
        return

    log.info(
        "build plan: train=%d (need %d more, %d rows, %.2f GiB each) | "
        "eval=%d (need %d more, %d rows)",
        args.max_train_shards, need_train, args.rows_per_shard,
        args.rows_per_shard * args.seq_len * 4 / (1024 ** 3),
        args.max_eval_shards, need_eval, args.eval_rows_per_shard,
    )
    log.info(
        "quality filter: enabled=%s min_chars=%d min_alpha=%.2f "
        "max_repeat=%.2f max_url_density=%.2f max_nav=%d",
        args.quality_filter, args.min_text_chars, args.min_alpha_ratio,
        args.max_repeat_ratio, args.max_url_density, args.max_nav_matches,
    )

    # ----------------------------------------------------------------
    # Load + shuffle dataset
    # ----------------------------------------------------------------
    from datasets import load_dataset
    log.info(
        "loading dataset %s / %s / %s (streaming=%s, shuffle_buffer=%d, seed=%d)",
        args.dataset, args.config, args.split, args.streaming,
        args.shuffle_buffer_size, args.seed,
    )
    ds = load_dataset(
        args.dataset, name=args.config, split=args.split,
        streaming=args.streaming,
        trust_remote_code=args.trust_remote_code,
    )

    # Shuffle using HF's built-in deterministic shuffling
    # This prevents first-stream bias by randomizing document order before tokenization
    if args.shuffle_buffer_size > 0:
        ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer_size)
        log.info("applied dataset.shuffle(seed=%d, buffer_size=%d)",
                 args.seed, args.shuffle_buffer_size)

    # ----------------------------------------------------------------
    # Streaming tokenization loop
    # ----------------------------------------------------------------
    seq_len = args.seq_len
    token_buffer: list[int] = []
    shard_rows: list[list[int]] = []
    t0 = time.time()
    docs_seen = 0
    docs_accepted = 0
    tokens_seen = 0
    reject_counts: dict[str, int] = defaultdict(int)

    building_train = need_train > 0
    rows_target = args.rows_per_shard if building_train else args.eval_rows_per_shard

    # Optional post-tokenization mixing buffer (secondary mixing layer)
    mix_buf: list[list[int]] = []
    mix_rng = np.random.default_rng(args.seed + 1)

    def _drain_mix_buf(n: int) -> None:
        """Drain n docs from mix_buf into token_buffer in random order."""
        picks = mix_rng.choice(len(mix_buf), size=min(n, len(mix_buf)), replace=False)
        for pi in sorted(picks.tolist(), reverse=True):
            token_buffer.extend(mix_buf.pop(int(pi)))

    def _flush_shard() -> bool:
        """Save one shard. Returns True if all targets are met."""
        nonlocal building_train, rows_target, shard_rows

        chunk = shard_rows[:rows_target]
        shard_rows = shard_rows[rows_target:]

        if building_train:
            idx = len(train_entries)
            entry = save_shard_entry(out_dir, chunk, idx, "train", seq_len)
            train_entries.append(entry)
            elapsed = max(time.time() - t0, 1e-6)
            log.info(
                "saved train shard %d/%d | rows=%d | %.2f GiB | "
                "docs=%d accepted=%d (%.1f%%) | tok/s=%.0f",
                len(train_entries), args.max_train_shards,
                len(chunk), entry["bytes"] / (1024 ** 3),
                docs_seen, docs_accepted,
                100.0 * docs_accepted / max(1, docs_seen),
                tokens_seen / elapsed,
            )
            _update_manifest()
            if len(train_entries) >= args.max_train_shards:
                if need_eval <= 0 or args.max_eval_shards <= 0:
                    return True
                log.info("all train shards done — switching to eval phase")
                building_train = False
                rows_target = args.eval_rows_per_shard
        else:
            idx = len(eval_entries)
            entry = save_shard_entry(out_dir, chunk, idx, "eval", seq_len)
            eval_entries.append(entry)
            log.info(
                "saved eval shard %d/%d | rows=%d",
                len(eval_entries), args.max_eval_shards, len(chunk),
            )
            _update_manifest()
            if len(eval_entries) >= args.max_eval_shards:
                return True
        return False

    def _update_manifest() -> None:
        manifest = build_manifest(out_dir, train_entries, eval_entries, tok_info, args)
        write_json_atomic(out_dir / "manifest.json", manifest)

    done = False
    pbar = tqdm(desc="retokenizing", unit="doc")

    for row in ds:
        docs_seen += 1
        pbar.update(1)

        text = row.get(args.text_column)
        if not text or not isinstance(text, str):
            reject_counts["empty"] += 1
            continue
        text = text.strip()

        # Quality filter
        accept, reason = check_quality(text, args)
        if not accept:
            reject_counts[reason] += 1
            continue

        ids = tok.encode(text, add_special_tokens=False)
        if not ids:
            reject_counts["empty_tokenization"] += 1
            continue
        mn, mx = min(ids), max(ids)
        if mn < 0 or mx >= model_vocab:
            reject_counts["out_of_vocab"] += 1
            log.warning("out-of-range token id min=%d max=%d — skipping doc", mn, mx)
            continue

        doc_tokens: list[int] = ids + [eos]
        docs_accepted += 1
        tokens_seen += len(doc_tokens)

        # Post-tokenization mixing buffer (optional secondary layer)
        if args.mix_buffer > 0:
            mix_buf.append(doc_tokens)
            if len(mix_buf) >= args.mix_buffer:
                _drain_mix_buf(len(mix_buf) // 2)
        else:
            token_buffer.extend(doc_tokens)

        # Slice token_buffer into seq_len windows
        while len(token_buffer) >= seq_len:
            shard_rows.append(token_buffer[:seq_len])
            del token_buffer[:seq_len]
            if len(shard_rows) >= rows_target:
                if _flush_shard():
                    done = True
                    break
        if done:
            break

    if not done:
        # Drain mix buffer
        if mix_buf:
            for doc in mix_buf:
                token_buffer.extend(doc)
            mix_buf.clear()
        # Flush remaining complete shards
        while not done and len(shard_rows) >= rows_target:
            if _flush_shard():
                done = True

    pbar.close()

    # Log rejection summary
    if reject_counts:
        total_rejected = sum(reject_counts.values())
        log.info(
            "quality filter: rejected %d docs (%.1f%%) | reasons: %s",
            total_rejected,
            100.0 * total_rejected / max(1, docs_seen),
            dict(reject_counts),
        )

    _finalize(
        out_dir, train_entries, eval_entries, tok_info, args,
        t0=t0, docs_seen=docs_seen, docs_accepted=docs_accepted,
        tokens_seen=tokens_seen, reject_counts=dict(reject_counts),
    )


def _finalize(
    out_dir: Path,
    train_entries: list[dict],
    eval_entries: list[dict],
    tok_info: dict,
    args: argparse.Namespace,
    t0: float,
    docs_seen: int,
    docs_accepted: int,
    tokens_seen: int,
    reject_counts: dict,
) -> None:
    elapsed = max(time.time() - t0, 1e-6)
    manifest = build_manifest(out_dir, train_entries, eval_entries, tok_info, args)
    write_json_atomic(out_dir / "manifest.json", manifest)

    total_tokens = sum(e.get("num_tokens", 0) for e in train_entries + eval_entries)
    docs_rejected = docs_seen - docs_accepted
    build_stats = {
        "docs_seen": docs_seen,
        "docs_accepted": docs_accepted,
        "docs_rejected": docs_rejected,
        "reject_rate": round(docs_rejected / max(1, docs_seen), 4),
        "reject_counts": reject_counts,
        "tokens_streamed": tokens_seen,
        "tokens_in_shards": total_tokens,
        "train_shards_created": len(train_entries),
        "eval_shards_created": len(eval_entries),
        "avg_tokens_per_doc": round(tokens_seen / max(1, docs_accepted)),
        "elapsed_s": round(elapsed, 1),
        "tok_per_s": round(tokens_seen / elapsed),
        "seed": getattr(args, "seed", 42),
        "shuffle_buffer_size": getattr(args, "shuffle_buffer_size", 0),
        "quality_filter": getattr(args, "quality_filter", True),
        "quality_filter_settings": {
            "min_text_chars": getattr(args, "min_text_chars", 200),
            "max_repeat_ratio": getattr(args, "max_repeat_ratio", 0.30),
            "min_alpha_ratio": getattr(args, "min_alpha_ratio", 0.50),
            "max_url_density": getattr(args, "max_url_density", 0.10),
            "max_nav_matches": getattr(args, "max_nav_matches", 5),
        },
        "dataset": getattr(args, "dataset", ""),
        "config": getattr(args, "config", ""),
        "split": getattr(args, "split", "train"),
        "seq_len": getattr(args, "seq_len", 2048),
        "rows_per_shard": getattr(args, "rows_per_shard", PROD_ROWS_PER_SHARD),
        "eval_rows_per_shard": getattr(args, "eval_rows_per_shard", PROD_EVAL_ROWS_PER_SHARD),
    }
    write_metadata(out_dir, tok_info, train_entries, eval_entries, build_stats)

    log.info(
        "DONE: %d train + %d eval shards | "
        "docs seen=%d accepted=%d (%.1f%%) | "
        "tokens_in_shards=%s | elapsed=%.1fs | tok/s=%.0f",
        len(train_entries), len(eval_entries),
        docs_seen, docs_accepted, 100.0 * docs_accepted / max(1, docs_seen),
        f"{total_tokens:,}", elapsed, tokens_seen / elapsed,
    )
    log.info("manifest: %s", out_dir / "manifest.json")

    n_train_wanted = getattr(args, "max_train_shards", 0)
    if n_train_wanted > 0 and len(train_entries) < n_train_wanted:
        log.warning(
            "only %d/%d train shards built — dataset exhausted. "
            "Use a larger HF config (sample-100BT, CC-MAIN-*) for more data.",
            len(train_entries), n_train_wanted,
        )


if __name__ == "__main__":
    main()
