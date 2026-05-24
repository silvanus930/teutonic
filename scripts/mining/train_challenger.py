#!/usr/bin/env python3
"""Teutonic mining harness — train a challenger that beats the current king.

Runs on a multi-B200 box. Pipeline:
  1. Discover current king from R2 dashboard (repo + revision).
  2. Download king from HF (snapshot_download, pinned revision).
  3. Pull training data: auto-selects raw FineWeb-Edu (Qwen tokenizer) when v2
     Gemma pretokenized shards mismatch the king; else pretokenized .npy shards.
  4. Score sample sequences with the king (avg next-token loss).
  5. Build a curriculum (general / hard / easy buckets, drop suspicious).
  6. Train a LoRA adapter with torchrun multi-GPU on the chosen training mix.
  7. Merge LoRA into the base weights -> standalone candidate dir.
  8. Offline paired eval candidate vs king on a held-out shard slice
     (mirrors validator's compute_paired_losses + bootstrap LCB > delta).
  9. Emit a JSON verdict file. If accepted, optionally upload to HF.

Designed to be re-run iteratively (--max-iters): if first attempt's mu_hat
falls short, training is re-run with a different seed / more steps until
the budget is spent.

This script is meant to live on the GPU box (e.g. /root/teutonic-mining/)
and be invoked there. It does NOT touch bittensor — that step is handled
by submit_challenger.py on the templar host where the wallet lives.

REQUIRED — coldkey prefix in --upload-repo (since 2026-04-29):
  The validator rejects any HF repo whose name doesn't contain the first
  8 ss58 chars of the miner's coldkey (case-insensitive substring, in
  either the HF account or the model basename). This is an
  anti-impersonation gate — only the legit coldkey owner can publish a
  repo whose name embeds *their* coldkey. Imposters who lift somebody
  else's URL end up advertising the victim's coldkey on chain.

  This script doesn't have wallet access so it can't enforce locally —
  the orchestrator (run_pipeline.sh) and the on-chain submitter
  (submit_challenger.py) both check before they burn HF / TAO. Pass an
  --upload-repo matching the active chain (chain.toml -> [chain].name)
  with your coldkey prefix embedded, e.g.
      myaccount/<chain.name>-5DhAqMpd-v3
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import logging
import math
import os

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import HfApi, snapshot_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

# Mining harness lives at scripts/mining/; bootstrap repo root onto sys.path
# so the active arch (chain.toml -> [arch].module) registers with HF Auto*.
# When deployed to the GPU box (run_pipeline.sh tar-pushes only this script),
# the user must `pip install -e .` the teutonic repo there so chain_config /
# archs are importable.
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
import chain_config  # noqa: E402

chain_config.load_arch()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [train_challenger] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_challenger")

# ---------------------------------------------------------------------------
# Defaults (mirror validator constants where applicable)
# ---------------------------------------------------------------------------
SEQ_LEN = 2048
EVAL_ALPHA = 0.001
EVAL_DELTA = 0.0025  # validator's effect floor in nats/token; see eval/torch_runner.py
LM_HEAD_CHUNK = 256
DASHBOARD_URL = os.environ.get(
    "TEUTONIC_DASHBOARD_URL",
    "https://us-east-1.hippius.com/teutonic-sn3/dashboard.json",
)
HIPPIUS_BASE = "https://s3.hippius.com/teutonic-sn3"


# ---------------------------------------------------------------------------
# Shard I/O
# ---------------------------------------------------------------------------
def parse_npy_header(raw: bytes) -> tuple[int, dict]:
    buf = io.BytesIO(raw)
    if buf.read(6) != b"\x93NUMPY":
        raise ValueError("not a .npy file")
    ver = struct.unpack("BB", buf.read(2))
    hl = struct.unpack("<H" if ver[0] == 1 else "<I",
                       buf.read(2 if ver[0] == 1 else 4))[0]
    header = eval(buf.read(hl).decode("latin1").strip())
    return buf.tell(), header


def load_shard(path: Path, seq_len: int = SEQ_LEN) -> tuple[np.ndarray, int]:
    """Shards are 1D uint32 arrays (concatenated tokens). Reshape into
    (n_sequences, seq_len) by truncating tail so it divides cleanly.
    Matches validator's slicing semantics in eval_torch.fetch_sequences.
    """
    raw = path.read_bytes()
    data_offset, header = parse_npy_header(raw)
    shape = header["shape"]
    flat = np.frombuffer(raw[data_offset:], dtype="<u4")
    if len(shape) == 1:
        n_total = shape[0]
        n_seq = n_total // seq_len
        arr = flat[: n_seq * seq_len].reshape(n_seq, seq_len)
    elif len(shape) == 2:
        n_seq, seq_len = shape
        arr = flat.reshape(n_seq, seq_len)
    else:
        raise ValueError(f"unexpected shard shape {shape}")
    return arr, seq_len


def download_shard(shard_key: str, out: Path) -> Path:
    """Return path to shard bytes. Supports absolute local paths (retokenized datasets)."""
    local_path = Path(shard_key)
    if local_path.is_file() and local_path.stat().st_size > 1024:
        log.info("using local shard: %s (%.2f GB)", local_path, local_path.stat().st_size / 1e9)
        return local_path
    if out.exists() and out.stat().st_size > 1024:
        log.info("shard cached: %s (%.1f GB)", out, out.stat().st_size / 1e9)
        return out
    url = f"{HIPPIUS_BASE}/{shard_key}"
    log.info("downloading %s -> %s", url, out)
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["curl", "-fsSL", "-o", str(out), url])
    return out


def fetch_manifest(cache: Path) -> dict:
    local_manifest = os.environ.get("LOCAL_DATASET_MANIFEST", "")
    if local_manifest:
        p = Path(local_manifest)
        if not p.exists():
            raise FileNotFoundError(f"LOCAL_DATASET_MANIFEST does not exist: {p}")
        log.info("loading local dataset manifest: %s", p)
        m = json.loads(p.read_text())
        base = p.parent
        for s in m.get("shards", []):
            key = s.get("key", "")
            kp = Path(key)
            if key and not kp.is_absolute():
                s["key"] = str((base / key).resolve())
        return m

    p = cache / "manifest.json"
    if not p.exists():
        url = f"{HIPPIUS_BASE}/dataset/v2/manifest.json"
        log.info("downloading manifest from %s", url)
        cache.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(["curl", "-fsSL", "-o", str(p), url])
    return json.loads(p.read_text())


def king_vocab_size(king_dir: Path) -> int | None:
    cfg_path = king_dir / "config.json"
    if not cfg_path.exists():
        return None
    vocab = int(json.loads(cfg_path.read_text()).get("vocab_size", 0) or 0)
    return vocab or None


def validate_sequences_vocab(arr: np.ndarray, vocab_size: int, label: str = "shard") -> None:
    """Fail fast before CUDA embedding lookup on out-of-vocab token ids."""
    if vocab_size is None:
        return
    max_id = int(arr.max()) if arr.size else -1
    if max_id >= vocab_size:
        raise ValueError(
            f"{label}: token id {max_id} >= model vocab_size {vocab_size}. "
            "Pretokenized v2 shards use the Gemma tokenizer (manifest "
            f"tokenizer field); Qwen3 kings need --dataset-mode raw "
            "(FineWeb-Edu tokenized at load time with the Qwen tokenizer, "
            "same as production eval)."
        )


def pretokenized_incompatible_with_king(manifest: dict, king_dir: Path) -> bool:
    """v2 CulturaX shards are Gemma-tokenized; Qwen3 kings cannot consume them."""
    shard_tok = (manifest.get("tokenizer") or "").lower()
    cfg = json.loads((king_dir / "config.json").read_text())
    arch = " ".join(cfg.get("architectures") or []).lower()
    if "qwen" not in arch:
        return False
    return "gemma" in shard_tok or "unsloth" in shard_tok


RAW_MANIFEST_KEY = os.environ.get(
    "TEUTONIC_RAW_DATASET_MANIFEST",
    "hf-mirrors/HuggingFaceFW/fineweb-edu/data/_manifest.json",
)
RAW_MAX_FILES = int(os.environ.get("TEUTONIC_RAW_MAX_FILES_PER_EVAL", "4"))


def hippius_download(key: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 1024:
        return dest
    url = f"{HIPPIUS_BASE}/{key}"
    log.info("downloading %s -> %s", url, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["curl", "-fsSL", "-o", str(dest), url])
    return dest


def global_sample_shard_indices(
    shards: list[np.ndarray],
    n: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Pick n random (shard_idx, row_idx) pairs uniformly across all shards."""
    active = [(i, s) for i, s in enumerate(shards) if len(s) > 0]
    if not active:
        return []
    sizes = [len(s) for _, s in active]
    total = sum(sizes)
    n = min(n, total)
    gindices = rng.choice(total, size=n, replace=False)
    cumsum = np.zeros(len(sizes) + 1, dtype=np.int64)
    for i, sz in enumerate(sizes):
        cumsum[i + 1] = cumsum[i] + sz
    out: list[tuple[int, int]] = []
    for gi in gindices:
        fi = int(np.searchsorted(cumsum, gi, side="right") - 1)
        local = int(gi - cumsum[fi])
        shard_idx, _ = active[fi]
        out.append((shard_idx, local))
    return out


def load_raw_sequences(
    n_sequences: int,
    seq_len: int,
    seed_str: str,
    cache: Path,
    tokenizer_repo: str,
    max_files: int = RAW_MAX_FILES,
    model_vocab_size: int | None = None,
) -> np.ndarray:
    """Tokenize FineWeb-Edu parquet into cache, then global-random sample windows."""
    import urllib.request
    from eval.raw_dataset import RawDatasetConfig, _get_tokenized_npy, sample_global_from_token_mmaps

    manifest_url = f"{HIPPIUS_BASE}/{RAW_MANIFEST_KEY}"
    log.info("fetching raw dataset manifest %s", manifest_url)
    with urllib.request.urlopen(manifest_url, timeout=60) as r:
        manifest = json.loads(r.read())
    files = sorted(
        item["dest_key"] if "dest_key" in item else item["key"]
        for item in (manifest.get("files") or manifest.get("shards") or [])
        if (item.get("dest_key") or item.get("key", "")).endswith(".parquet")
    )
    if not files:
        raise RuntimeError(f"no parquet files in raw manifest {RAW_MANIFEST_KEY}")

    raw_cache = cache / "raw_parquet"
    raw_cache.mkdir(parents=True, exist_ok=True)
    token_cache = cache / "raw_tokens"
    token_cache.mkdir(parents=True, exist_ok=True)

    cfg = RawDatasetConfig(
        manifest_key=RAW_MANIFEST_KEY,
        prefix=os.environ.get("TEUTONIC_RAW_DATASET_PREFIX", "hf-mirrors/HuggingFaceFW/fineweb-edu/data").strip("/"),
        explicit_keys=tuple(),
        tokenizer_repo=tokenizer_repo,
        text_column=os.environ.get("TEUTONIC_RAW_TEXT_COLUMN", "text"),
        cache_dir=token_cache,
        max_files_per_eval=max_files,
        list_fallback=False,
    )

    class _HttpR2:
        ds_bucket = "hippius"
        ds_client = None

        def ds_get(self, _key: str):
            return None

    r2 = _HttpR2()
    orig_download = None
    try:
        from eval import raw_dataset as _raw_mod
        orig_download = _raw_mod._download_parquet

        def _http_download(_r2, _cfg, key: str) -> Path:
            digest = hashlib.sha256(key.encode()).hexdigest()[:24]
            return hippius_download(key, raw_cache / f"{digest}.parquet")

        _raw_mod._download_parquet = _http_download

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_repo, token=os.environ.get("HF_TOKEN") or None, use_fast=True,
        )
        eos_id = tokenizer.eos_token_id
        if eos_id is None:
            eos_id = tokenizer.sep_token_id

        npy_files = sorted(token_cache.glob("*.tokens.npy"))
        if len(npy_files) < max_files:
            seed = int.from_bytes(
                hashlib.blake2b(seed_str.encode(), digest_size=8).digest(), "little",
            )
            rng = np.random.Generator(np.random.PCG64(seed))
            start = int(rng.integers(0, len(files)))
            ordered = files[start:] + files[:start]
            for key in ordered[:max_files]:
                _get_tokenized_npy(r2, cfg, key, tokenizer, eos_id)
            npy_files = sorted(token_cache.glob("*.tokens.npy"))

        windows, meta = sample_global_from_token_mmaps(
            npy_files, n_sequences, seq_len, seed_str,
        )
    finally:
        if orig_download is not None:
            from eval import raw_dataset as _raw_mod
            _raw_mod._download_parquet = orig_download

    arr = np.asarray(windows, dtype=np.uint32)
    validate_sequences_vocab(arr, model_vocab_size, "raw")
    log.info(
        "raw dataset (global): %d sequences from %d token cache shard(s), max_id=%d meta=%s",
        len(arr), meta.get("total_shards", len(npy_files)), int(arr.max()), meta.get("mode"),
    )
    return arr


def resolve_dataset_mode(requested: str, manifest: dict, king_dir: Path) -> str:
    if requested != "auto":
        return requested
    if os.environ.get("LOCAL_DATASET_MANIFEST"):
        log.info(
            "auto dataset mode: LOCAL_DATASET_MANIFEST set -> pretokenized local shards "
            "(tokenizer=%r)",
            manifest.get("tokenizer") or manifest.get("tokenizer_dir"),
        )
        return "pretokenized"
    if pretokenized_incompatible_with_king(manifest, king_dir):
        log.warning(
            "auto dataset mode: v2 manifest tokenizer=%r is incompatible with Qwen king; "
            "using raw FineWeb-Edu (retokenize with scripts/mining/retokenize_fineweb_edu_qwen.py "
            "or set LOCAL_DATASET_MANIFEST to skip per-run parquet downloads)",
            manifest.get("tokenizer"),
        )
        return "raw"
    return "pretokenized"


# ---------------------------------------------------------------------------
# King discovery
# ---------------------------------------------------------------------------
def fetch_king() -> dict:
    import urllib.request
    log.info("fetching dashboard %s", DASHBOARD_URL)
    with urllib.request.urlopen(DASHBOARD_URL, timeout=30) as r:
        d = json.loads(r.read())

    k = d["king"]

    # New dashboard schema uses model_repo / king_digest.
    # Old script expected hf_repo / king_revision.
    repo = k.get("hf_repo") or k.get("model_repo") or k.get("previous_repo")
    revision = k.get("king_revision") or k.get("revision") or k.get("king_digest")

    if not repo:
        raise KeyError(f"Could not find king repo in dashboard king object. Keys: {list(k.keys())}")

    k["hf_repo"] = repo
    k["king_revision"] = revision

    log.info(
        "king: repo=%s revision=%s reign=%d hotkey=%s",
        repo,
        (revision or "HEAD")[:20],
        k.get("reign_number", 0),
        k.get("hotkey", "?")[:16],
    )
    return k


def sha256_dir(path: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(path.glob("*.safetensors")):
        with open(p, "rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Paired eval (mirrors eval_torch.compute_paired_losses + bootstrap)
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_per_seq_loss(model, token_batches, device, chunk=LM_HEAD_CHUNK):
    """Average per-token cross-entropy per sequence (matches eval_torch)."""
    input_ids = torch.tensor(token_batches, dtype=torch.long, device=device)
    # Reset stateful arch (Quasar latent memory) before each batch — see
    # eval_torch.compute_paired_losses for rationale. No-op for stock HF archs.
    if hasattr(model, "reset_state"):
        model.reset_state()
    out = model.model(input_ids)
    hidden = out.last_hidden_state
    lm_head = model.lm_head
    n_pos = input_ids.size(1) - 1
    total = torch.zeros(len(token_batches), device=device)
    for i in range(0, n_pos, chunk):
        end = min(i + chunk, n_pos)
        logits = lm_head(hidden[:, i:end, :])
        labels = input_ids[:, i + 1:end + 1]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            reduction="none",
        )
        total += loss.reshape(len(token_batches), -1).sum(dim=1)
        del logits, loss
    return (total / n_pos).cpu().tolist()


def paired_eval(king_dir: str, chall_dir: str, shard: np.ndarray,
                indices: list[int], device: str, batch_size: int = 8,
                n_bootstrap: int = 10000, alpha: float = EVAL_ALPHA) -> dict:
    """Mirrors validator's paired bootstrap test on a single GPU.

    Acceptance floor delta = EVAL_DELTA (default 0.0025 nats/token) matches
    the validator's restored fixed-effect-floor rule.
    """
    delta = EVAL_DELTA
    log.info("paired_eval: loading king %s on %s", king_dir, device)
    king = AutoModelForCausalLM.from_pretrained(
        king_dir, torch_dtype=torch.bfloat16, device_map={"": device},
        use_safetensors=True,
    )
    king.eval()
    log.info("paired_eval: loading challenger %s on %s", chall_dir, device)
    chall = AutoModelForCausalLM.from_pretrained(
        chall_dir, torch_dtype=torch.bfloat16, device_map={"": device},
        use_safetensors=True,
    )
    chall.eval()

    diffs = []
    king_sum = chall_sum = 0.0
    n_done = 0
    t0 = time.time()
    for i in range(0, len(indices), batch_size):
        batch_idx = indices[i:i + batch_size]
        toks = [shard[j].tolist() for j in batch_idx]
        kl = compute_per_seq_loss(king, toks, device)
        cl = compute_per_seq_loss(chall, toks, device)
        for k, c in zip(kl, cl):
            diffs.append(k - c)
            king_sum += k
            chall_sum += c
            n_done += 1
        if (i // batch_size) % 5 == 0:
            mu = float(np.mean(diffs))
            log.info("eval %d/%d | mu_hat=%.6f | king=%.4f chall=%.4f | %.1fs",
                     n_done, len(indices), mu,
                     king_sum / n_done, chall_sum / n_done, time.time() - t0)

    diffs = np.asarray(diffs, dtype=np.float64)
    mu_hat = float(diffs.mean())
    boot = np.empty(n_bootstrap)
    rng = np.random.default_rng(0xB007)
    for b in range(n_bootstrap):
        boot[b] = diffs[rng.integers(0, len(diffs), size=len(diffs))].mean()
    lcb = float(np.quantile(boot, alpha))
    accepted = lcb > delta
    res = {
        "eval_mode": "local",
        "n_eval": n_done,
        "mu_hat": mu_hat,
        "lcb": lcb,
        "delta": delta,
        "delta_threshold": delta,
        "alpha": alpha,
        "accepted": accepted,
        "avg_king_loss": king_sum / n_done,
        "avg_chall_loss": chall_sum / n_done,
        "elapsed_s": time.time() - t0,
        "note": (
            "local mode: holdout sampled from the mining data pool (not validator seeds); "
            "use --eval-mode validator to match eval_server"
        ),
    }
    log.info("paired_eval: mu_hat=%.6f lcb=%.6f accepted=%s",
             mu_hat, lcb, accepted)
    del king, chall
    torch.cuda.empty_cache()
    return res


# ---------------------------------------------------------------------------
# Sample scoring + curriculum (single-GPU; lifted from training_bundle)
# ---------------------------------------------------------------------------
def _loss_summary(losses: np.ndarray) -> dict:
    if len(losses) == 0:
        return {}
    return {
        "n": int(len(losses)),
        "min": float(losses.min()),
        "max": float(losses.max()),
        "mean": float(losses.mean()),
        "std": float(losses.std()),
        "p10": float(np.percentile(losses, 10)),
        "p25": float(np.percentile(losses, 25)),
        "p50": float(np.percentile(losses, 50)),
        "p75": float(np.percentile(losses, 75)),
        "p85": float(np.percentile(losses, 85)),
        "p90": float(np.percentile(losses, 90)),
    }


def _bucket_means(rows: list[dict], key: str = "loss") -> dict[str, float]:
    out: dict[str, float] = {}
    for b in ("general", "hard", "easy", "suspicious"):
        vals = [r[key] for r in rows if r.get("bucket") == b]
        if vals:
            out[b] = float(np.mean(vals))
    return out


def score_and_curate(king_dir: str, shards: list[np.ndarray],
                     n_score: int, train_per_iter: int, val_size: int,
                     seed: int, device: str, work: Path) -> tuple[Path, Path]:
    """Score `n_score` random samples on the king, bucket, write train/val jsonl."""
    rng = np.random.default_rng(seed)
    cands = global_sample_shard_indices(shards, n_score, rng)
    rng.shuffle(cands)

    log.info(
        "scoring: %d candidates (global sample across %d shard(s)), seed=%d, king=%s, device=%s",
        len(cands), len(shards), seed, king_dir, device,
    )
    t_score = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        king_dir, torch_dtype=torch.bfloat16, device_map={"": device},
        use_safetensors=True,
    )
    model.eval()

    rows = []
    BATCH = 8
    log_every = max(1, len(cands) // (BATCH * 10))  # ~10 progress lines
    for batch_i, i in enumerate(range(0, len(cands), BATCH)):
        chunk = cands[i:i + BATCH]
        toks = [shards[s][j].tolist() for s, j in chunk]
        losses = compute_per_seq_loss(model, toks, device)
        for (s_idx, j), tok, loss in zip(chunk, toks, losses):
            arr = np.asarray(tok)
            unique_r = float(len(set(tok)) / len(tok))
            rep_r = float(np.mean(arr[1:] == arr[:-1])) if len(arr) > 1 else 0.0
            ngrams = [tuple(tok[k:k + 4]) for k in range(len(tok) - 3)]
            rep_ng = 1.0 - len(set(ngrams)) / len(ngrams) if ngrams else 0.0
            rows.append({
                "shard": s_idx,
                "idx": j,
                "loss": float(loss),
                "unique_r": unique_r,
                "rep_r": rep_r,
                "rep_ng4": rep_ng,
                "tokens": tok,
            })
        if batch_i % log_every == 0 or i + BATCH >= len(cands):
            done = min(i + BATCH, len(cands))
            run_losses = np.asarray([r["loss"] for r in rows])
            ls = _loss_summary(run_losses)
            log.info(
                "scoring progress %d/%d (%.1f%%) | running loss mean=%.4f p50=%.4f | %.1fs",
                done, len(cands), 100.0 * done / len(cands),
                ls.get("mean", float("nan")), ls.get("p50", float("nan")),
                time.time() - t_score,
            )

    del model
    torch.cuda.empty_cache()
    log.info("scoring forward pass done in %.1fs (%d sequences)",
             time.time() - t_score, len(rows))

    losses = np.asarray([r["loss"] for r in rows])
    loss_stats = _loss_summary(losses)
    p50 = loss_stats["p50"]
    p85 = loss_stats["p85"]
    general_floor = p50 * 0.8
    log.info(
        "scoring loss distribution: min=%.4f max=%.4f mean=%.4f std=%.4f "
        "p10=%.4f p25=%.4f p50=%.4f p75=%.4f p85=%.4f p90=%.4f",
        loss_stats.get("min", 0), loss_stats.get("max", 0),
        loss_stats.get("mean", 0), loss_stats.get("std", 0),
        loss_stats.get("p10", 0), loss_stats.get("p25", 0), p50,
        loss_stats.get("p75", 0), p85, loss_stats.get("p90", 0),
    )
    log.info(
        "bucketing thresholds: suspicious if rep_r>0.2 or rep_ng4>0.5 or unique_r<0.05; "
        "hard if loss>=%.4f; general if loss>=%.4f; else easy",
        p85, general_floor,
    )

    def bucket(r):
        if r["rep_r"] > 0.2 or r["rep_ng4"] > 0.5 or r["unique_r"] < 0.05:
            return "suspicious"
        if r["loss"] >= p85:
            return "hard"
        if r["loss"] >= general_floor:
            return "general"
        return "easy"

    for r in rows:
        r["bucket"] = bucket(r)
    counts = {b: sum(1 for r in rows if r["bucket"] == b)
              for b in ("general", "hard", "easy", "suspicious")}
    bucket_loss = _bucket_means(rows)
    log.info(
        "scoring buckets: %s | mean loss by bucket: %s",
        counts,
        {k: f"{v:.4f}" for k, v in bucket_loss.items()},
    )

    # --- curriculum ---
    log.info("curriculum: building train/val from %d scored rows", len(rows))
    clean = [r for r in rows if r["bucket"] != "suspicious"]
    dropped = len(rows) - len(clean)
    log.info(
        "curriculum: dropped %d suspicious (%.1f%%), %d clean remain",
        dropped, 100.0 * dropped / max(1, len(rows)), len(clean),
    )

    rng2 = np.random.default_rng(seed + 1)
    rng2.shuffle(clean)
    val_rows = clean[:val_size]
    val_keys = {(r["shard"], r["idx"]) for r in val_rows}
    pool = [r for r in clean if (r["shard"], r["idx"]) not in val_keys]
    log.info(
        "curriculum: val=%d (requested %d), train pool=%d",
        len(val_rows), val_size, len(pool),
    )
    if val_rows:
        val_ls = _loss_summary(np.asarray([r["loss"] for r in val_rows]))
        log.info(
            "curriculum val loss: mean=%.4f p50=%.4f buckets=%s",
            val_ls.get("mean", 0), val_ls.get("p50", 0),
            {b: sum(1 for r in val_rows if r["bucket"] == b)
             for b in ("general", "hard", "easy")},
        )

    general = [r for r in pool if r["bucket"] == "general"]
    hard = [r for r in pool if r["bucket"] == "hard"]
    easy = [r for r in pool if r["bucket"] == "easy"]
    n_general = int(train_per_iter * 0.6)
    n_hard = int(train_per_iter * 0.3)
    n_easy = train_per_iter - n_general - n_hard
    log.info(
        "curriculum train mix target: general=%d hard=%d easy=%d (total=%d) | "
        "pool available: general=%d hard=%d easy=%d",
        n_general, n_hard, n_easy, train_per_iter,
        len(general), len(hard), len(easy),
    )

    train_rows = []
    picked = {}
    for label, src, n in (("general", general, n_general),
                          ("hard", hard, n_hard),
                          ("easy", easy, n_easy)):
        if not src:
            picked[label] = 0
            if n > 0:
                log.warning("curriculum: no %s samples in pool (wanted %d)", label, n)
            continue
        if n >= len(src):
            train_rows.extend(src)
            picked[label] = len(src)
            log.info(
                "curriculum: took all %d %s samples (wanted %d)",
                len(src), label, n,
            )
        else:
            sel = rng2.choice(len(src), size=n, replace=False)
            train_rows.extend(src[int(k)] for k in sel)
            picked[label] = n
            log.info("curriculum: sampled %d/%d %s", n, len(src), label)
    rng2.shuffle(train_rows)

    train_mix = {b: sum(1 for r in train_rows if r["bucket"] == b)
                 for b in ("general", "hard", "easy")}
    if train_rows:
        tr_ls = _loss_summary(np.asarray([r["loss"] for r in train_rows]))
        log.info(
            "curriculum train: %d sequences | mix %s | loss mean=%.4f p50=%.4f",
            len(train_rows), train_mix, tr_ls.get("mean", 0), tr_ls.get("p50", 0),
        )
    if len(train_rows) < train_per_iter:
        log.warning(
            "curriculum: train set undersized (%d < %d); increase --n-score or pool",
            len(train_rows), train_per_iter,
        )

    work.mkdir(parents=True, exist_ok=True)
    train_p = work / "train.jsonl"
    val_p = work / "val.jsonl"
    scored_p = work / "scored.jsonl"
    scoring_p = work / "scoring.json"
    curriculum_p = work / "curriculum.json"

    with open(scored_p, "w") as f:
        for r in rows:
            f.write(json.dumps({
                "shard": r["shard"], "idx": r["idx"],
                "loss": r["loss"], "unique_r": r["unique_r"],
                "rep_r": r["rep_r"], "rep_ng4": r["rep_ng4"], "bucket": r["bucket"],
            }) + "\n")
    with open(train_p, "w") as f:
        for r in train_rows:
            f.write(json.dumps({"input_ids": r["tokens"]}) + "\n")
    with open(val_p, "w") as f:
        for r in val_rows:
            f.write(json.dumps({"input_ids": r["tokens"]}) + "\n")

    scoring_report = {
        "seed": seed,
        "n_candidates": len(cands),
        "loss": loss_stats,
        "thresholds": {"p50": p50, "p85": p85, "general_floor": general_floor},
        "bucket_counts": counts,
        "bucket_mean_loss": bucket_loss,
        "elapsed_s": time.time() - t_score,
    }
    curriculum_report = {
        "seed": seed + 1,
        "train_per_iter": train_per_iter,
        "val_size": val_size,
        "suspicious_dropped": dropped,
        "clean_pool": len(clean),
        "train_pool": len(pool),
        "picked": picked,
        "train_mix": train_mix,
        "train_n": len(train_rows),
        "val_n": len(val_rows),
        "val_bucket_counts": {b: sum(1 for r in val_rows if r["bucket"] == b)
                              for b in ("general", "hard", "easy")},
        "train_loss": _loss_summary(np.asarray([r["loss"] for r in train_rows]))
        if train_rows else {},
        "val_loss": _loss_summary(np.asarray([r["loss"] for r in val_rows]))
        if val_rows else {},
    }
    json.dump(scoring_report, open(scoring_p, "w"), indent=2)
    json.dump(curriculum_report, open(curriculum_p, "w"), indent=2)

    log.info(
        "curriculum wrote train=%d val=%d | artifacts: %s %s %s %s",
        len(train_rows), len(val_rows), scored_p, scoring_p, curriculum_p, train_p,
    )
    return train_p, val_p


# ---------------------------------------------------------------------------
# Multi-GPU LoRA training (delegated to torchrun)
# ---------------------------------------------------------------------------
def run_lora_training(base_model: str, train_p: Path, val_p: Path,
                      out_dir: Path, n_gpus: int, args: argparse.Namespace,
                      bundle: Path) -> Path:
    """Spawn torchrun on the existing training_bundle script."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "torchrun", f"--nproc_per_node={n_gpus}",
        str(bundle / "train_lora_token_ids.py"),
        "--base-model", base_model,
        "--train-data", str(train_p),
        "--val-data", str(val_p),
        "--output-dir", str(out_dir),
        "--seq-len", "2048",
        "--micro-batch-size", str(args.micro_batch),
        "--grad-accum", str(args.grad_accum),
        "--learning-rate", str(args.lr),
        "--epochs", str(args.epochs),
        "--lora-r", str(args.lora_r),
        "--lora-alpha", str(args.lora_alpha),
        "--lora-dropout", "0.05",
    ]
    if args.resume_checkpoint:
        cmd.extend(["--resume-from-checkpoint", str(args.resume_checkpoint)])
    log.info("training: %s", " ".join(cmd))
    t0 = time.time()
    subprocess.check_call(cmd)
    log.info("training done in %.1fs", time.time() - t0)
    adapter = out_dir / "best_adapter"
    if not adapter.exists():
        # Trainer.save_model may have only put the adapter in the root output_dir.
        # Look for adapter_model files.
        if (out_dir / "adapter_model.safetensors").exists() or \
           (out_dir / "adapter_model.bin").exists():
            adapter = out_dir
        else:
            raise RuntimeError(f"no adapter found in {out_dir}")
    return adapter


def merge_lora(base_model: str, adapter: Path | str, out: Path | str) -> Path:
    adapter = Path(adapter)
    out = Path(out)
    log.info("merging LoRA %s into %s -> %s", adapter, base_model, out)
    from peft import PeftModel
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, use_safetensors=True,
    )
    merged = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out), safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    tok.save_pretrained(str(out))
    # Copy extra config files from base (local king dir or HF repo).
    base_path = Path(base_model).expanduser().resolve()
    for name in ("config.json", "generation_config.json"):
        if base_path.is_dir():
            src = base_path / name
        else:
            src = Path(snapshot_download(base_model, allow_patterns=[name])) / name
        if src.is_file():
            shutil.copy(src, out / name)
    del base, merged
    torch.cuda.empty_cache()
    log.info("merged model saved to %s", out)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/root/teutonic-mining/work",
                    help="Working dir on this box")
    ap.add_argument("--bundle", default="/root/teutonic-mining/bundle",
                    help="Path to training_bundle directory")
    ap.add_argument("--n-shards", type=int, default=0,
                    help="Training shards to load (0 = all manifest shards except --eval-shard)")
    ap.add_argument("--shard-start", type=int, default=0,
                    help="Index of first shard to use (other than eval shard)")
    ap.add_argument("--dataset-mode", choices=("auto", "pretokenized", "raw"), default="auto",
                    help="auto: use raw FineWeb if v2 Gemma shards mismatch Qwen king "
                         "(production eval path); pretokenized: v2 .npy shards; raw: force FineWeb")
    ap.add_argument("--raw-max-files", type=int, default=RAW_MAX_FILES,
                    help="Max FineWeb parquet files to scan when --dataset-mode raw")
    ap.add_argument("--eval-shard", type=int, default=-1,
                    help="Held-out shard index for offline paired eval (-1 = last shard)")
    ap.add_argument("--eval-mode", choices=("validator", "local"), default="validator",
                    help="validator: same holdout+sampling+bootstrap as eval_server; "
                         "local: fast eval on mining pool subset (optimistic)")
    ap.add_argument("--sim-block-hash", default=os.environ.get("TEUTONIC_SIM_BLOCK_HASH", ""),
                    help="Pinned block hash for validator-style holdout seeds "
                         "(default: 64 zero hex digits, or TEUTONIC_SIM_BLOCK_HASH)")
    ap.add_argument("--sim-hotkey", default=os.environ.get("TEUTONIC_SIM_HOTKEY", ""),
                    help="Hotkey ss58 for validator-style holdout seeds (TEUTONIC_SIM_HOTKEY)")
    ap.add_argument("--n-eval", type=int, default=25600,
                    help="Public holdout sequences (validator default TEUTONIC_EVAL_N_PUBLIC=25600)")
    ap.add_argument("--n-eval-private", type=int, default=0,
                    help="Private holdout sequences (validator TEUTONIC_EVAL_N_PRIVATE)")
    ap.add_argument("--eval-batch-size", type=int, default=64,
                    help="Batch size for validator-style paired eval")
    ap.add_argument("--eval-gpus", default="",
                    help="Comma-separated GPU ids for eval only (default: first --n-gpus GPU)")
    ap.add_argument("--n-score", type=int, default=4000)
    ap.add_argument("--train-per-iter", type=int, default=4000)
    ap.add_argument("--val-size", type=int, default=400)
    ap.add_argument("--max-iters", type=int, default=3,
                    help="Retry training with new seed if first attempt insufficient")
    ap.add_argument("--target-mu", type=float, default=0.05,
                    help="Stop training as soon as offline mu_hat exceeds this")
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-gpus", type=int, default=8)
    ap.add_argument("--upload-repo", default="",
                    help="If set + accepted, push merged model to this HF repo. "
                         "Must contain the first 8 ss58 chars of your coldkey "
                         "(case-insensitive substring) or the validator will "
                         "reject the eval with `coldkey_required`. See "
                         "submit_challenger.py for the gate that enforces this.")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    ap.add_argument("--report-out", default="",
                    help="Write a final JSON verdict to this path")
    ap.add_argument(
        "--resume-checkpoint", default="",
        help="Resume LoRA training from this Trainer checkpoint dir (skip scoring if "
             "iter dir already has train.jsonl; use with --from-iter)",
    )
    ap.add_argument(
        "--from-iter", type=int, default=-1,
        help="Only run merge+eval for this iter index (skip score/train). "
             "Use after interrupted training.",
    )
    ap.add_argument(
        "--skip-scoring", action="store_true",
        help="Reuse iter_XX/train.jsonl + val.jsonl (skip score/curate; skips shard reload "
             "when --eval-mode validator)",
    )
    ap.add_argument(
        "--profile", choices=("default", "prod"), default="default",
        help="prod: wide local-shard curriculum + validator eval defaults",
    )
    args = ap.parse_args()

    if args.profile == "prod":
        args.n_score = max(args.n_score, 12000)
        args.train_per_iter = max(args.train_per_iter, 10000)
        args.val_size = max(args.val_size, 500)
        args.n_eval = max(args.n_eval, 25600)
        args.epochs = max(args.epochs, 3.0)
        args.max_iters = max(args.max_iters, 5)
        args.lora_r = max(args.lora_r, 32)
        args.lora_alpha = max(args.lora_alpha, 64)
        args.lr = min(args.lr, 1e-4) if args.lr == 2e-4 else args.lr
        args.target_mu = min(args.target_mu, 0.015)
        args.eval_mode = "validator"
        args.raw_max_files = max(args.raw_max_files, 32)
        log.info("profile=prod applied (wide data + validator eval defaults)")

    min_score = args.train_per_iter + args.val_size
    if args.n_score < min_score:
        log.warning(
            "n_score=%d < train_per_iter+val_size=%d — curriculum will be undersized; "
            "use --n-score >= %d (prod profile does this automatically)",
            args.n_score, min_score, min_score + 500,
        )

    if args.eval_mode == "validator":
        os.environ.setdefault("TEUTONIC_EVAL_DATASET_MODE", "raw_hippius")
        os.environ.setdefault("TEUTONIC_RAW_TOKENIZER_REPO", "Qwen/Qwen3-4B")

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    cache = work / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    # 1. king
    king = fetch_king()
    king_dir = work / "king"

    local_king = os.environ.get("LOCAL_KING_DIR", "")

    if local_king:
        local_king = Path(local_king)
        if not local_king.exists():
            raise FileNotFoundError(f"LOCAL_KING_DIR does not exist: {local_king}")
        if king_dir.exists():
            shutil.rmtree(king_dir)
        log.info("copying local king from %s to %s", local_king, king_dir)
        shutil.copytree(local_king, king_dir, symlinks=False)
    else:
        if king_dir.exists():
            shutil.rmtree(king_dir)
        log.info("downloading king to %s", king_dir)
        snapshot_download(
            king["hf_repo"],
            local_dir=str(king_dir),
            revision=king.get("king_revision") or None,
            token=args.hf_token or None,
            max_workers=16,
        )

    king_hash = sha256_dir(king_dir)
    log.info("king sha256[:16]=%s", king_hash[:16])

    vocab_size = king_vocab_size(king_dir)
    shards: list = []
    eval_arr = None
    eval_indices = None

    skip_shard_load = (
        args.skip_scoring
        or args.from_iter >= 0
    ) and args.eval_mode == "validator"

    if skip_shard_load:
        log.info(
            "skipping dataset shard load (--skip-scoring or --from-iter, validator eval)"
        )
    else:
        # 2. dataset
        manifest = fetch_manifest(cache)
        dataset_mode = resolve_dataset_mode(args.dataset_mode, manifest, king_dir)
        log.info("dataset mode: %s (manifest tokenizer=%r, king vocab=%s)",
                 dataset_mode, manifest.get("tokenizer"), vocab_size)

    if not skip_shard_load and dataset_mode == "raw":
        n_needed = args.n_score + args.train_per_iter + args.val_size + 256
        if args.eval_mode == "local":
            n_needed += args.n_eval
        tokenizer_repo = os.environ.get(
            "TEUTONIC_RAW_TOKENIZER_REPO", chain_config.SEED_TOKENIZER_REPO,
        ) or str(king_dir)
        pool = load_raw_sequences(
            n_needed, SEQ_LEN, f"mining:{args.seed}", cache,
            tokenizer_repo, max_files=args.raw_max_files,
            model_vocab_size=vocab_size,
        )
        shards = [pool]
        eval_arr = None
        eval_indices = None
        if args.eval_mode == "local":
            eval_arr = pool
            rng_eval = np.random.default_rng(0xE1A)
            eval_indices = rng_eval.choice(
                len(eval_arr), size=min(args.n_eval, len(eval_arr)), replace=False,
            ).tolist()
            log.info(
                "raw local eval pool: %d sequences (sampling %d for paired eval)",
                len(eval_arr), len(eval_indices),
            )
        else:
            log.info(
                "raw validator eval: holdout will be sampled per duel seeds "
                "(not from mining pool)"
            )
    elif not skip_shard_load:
        n_manifest_shards = len(manifest.get("shards") or [])
        if n_manifest_shards == 0:
            raise ValueError("dataset manifest has no shards")

        def _check_shard_idx(idx: int, label: str) -> None:
            if idx < 0 or idx >= n_manifest_shards:
                raise ValueError(
                    f"{label} shard index {idx} out of range: manifest has "
                    f"{n_manifest_shards} shard(s) (valid indices 0.."
                    f"{n_manifest_shards - 1}). "
                    f"For a local retokenized dataset with 2 shards use "
                    f"--shard-start 0 --n-shards 1 --eval-shard 1."
                )

        if args.eval_shard < 0:
            args.eval_shard = n_manifest_shards - 1

        if args.n_shards <= 0:
            train_shard_idxs = [i for i in range(n_manifest_shards) if i != args.eval_shard]
            log.info(
                "loading all training shards: %d shard(s), holdout eval_shard=%d",
                len(train_shard_idxs), args.eval_shard,
            )
        else:
            train_shard_idxs = list(range(args.shard_start, args.shard_start + args.n_shards))
        for idx in train_shard_idxs:
            _check_shard_idx(idx, "training")
        _check_shard_idx(args.eval_shard, "eval")
        if args.eval_shard in train_shard_idxs:
            raise ValueError("eval_shard cannot overlap training shards")
        shards = []
        for idx in train_shard_idxs:
            key = manifest["shards"][idx]["key"]
            path = download_shard(key, cache / Path(key).name)
            arr, _ = load_shard(path)
            validate_sequences_vocab(arr, vocab_size, f"shard {idx}")
            log.info("loaded shard %d: %d sequences", idx, len(arr))
            shards.append(arr)

        eval_key = manifest["shards"][args.eval_shard]["key"]
        eval_path = download_shard(eval_key, cache / Path(eval_key).name)
        eval_arr, _ = load_shard(eval_path)
        validate_sequences_vocab(eval_arr, vocab_size, f"eval shard {args.eval_shard}")
        rng_eval = np.random.default_rng(0xE1A)
        eval_indices = rng_eval.choice(
            len(eval_arr), size=min(args.n_eval, len(eval_arr)), replace=False,
        ).tolist()
        log.info("held-out eval shard %d: %d sequences (sampling %d)",
                 args.eval_shard, len(eval_arr), len(eval_indices))

    best = None
    history = []
    if args.from_iter >= 0:
        iter_list = [args.from_iter]
        log.info("from-iter mode: only processing iter_%02d", args.from_iter)
    else:
        iter_list = list(range(args.max_iters))
    for it in iter_list:
        log.info("=" * 60)
        log.info("=== iteration %d/%d ===", it + 1, args.max_iters)
        log.info("=" * 60)
        seed = args.seed + 1000 * it

        iter_work = work / f"iter_{it:02d}"
        iter_work.mkdir(exist_ok=True)
        train_p = iter_work / "train.jsonl"
        val_p = iter_work / "val.jsonl"
        merge_only = args.from_iter >= 0
        out_dir = iter_work / "lora_out"
        merged_dir = iter_work / "merged"

        if merge_only:
            log.info("from-iter %d: merge+eval only (skip score+train)", it)
            if not train_p.exists():
                raise FileNotFoundError(
                    f"--from-iter {it} but missing {train_p}; train first or use --skip-scoring"
                )
        elif args.skip_scoring and train_p.exists() and val_p.exists():
            log.info(
                "skip-scoring: reusing %s (%d lines) and %s",
                train_p, sum(1 for _ in open(train_p)), val_p,
            )
        else:
            if args.skip_scoring:
                raise FileNotFoundError(
                    f"--skip-scoring set but missing {train_p} or {val_p}; "
                    "run scoring once or omit --skip-scoring"
                )
            train_p, val_p = score_and_curate(
                str(king_dir), shards, args.n_score,
                args.train_per_iter, args.val_size, seed, "cuda:0", iter_work,
            )

        if merge_only:
            ckpt = (args.resume_checkpoint or "").strip()
            if ckpt:
                adapter = Path(ckpt)
            elif (out_dir / "best_adapter").exists():
                adapter = out_dir / "best_adapter"
            else:
                cks = sorted(out_dir.glob("checkpoint-*"),
                             key=lambda p: int(p.name.split("-")[-1]))
                if not cks:
                    raise FileNotFoundError(
                        f"no checkpoint under {out_dir}; pass --resume-checkpoint"
                    )
                adapter = cks[-1]
                log.info("using latest checkpoint %s", adapter)
        else:
            resume_ckpt = (args.resume_checkpoint or "").strip() if it == iter_list[0] else ""
            if resume_ckpt:
                log.info("resuming LoRA from %s", resume_ckpt)
            adapter = run_lora_training(
                str(king_dir), train_p, val_p, out_dir, args.n_gpus, args,
                Path(args.bundle),
            )

        if not merged_dir.exists() or not (merged_dir / "config.json").is_file():
            merge_lora(str(king_dir), adapter, merged_dir)
        else:
            log.info("merged model already exists at %s", merged_dir)

        # 7. paired eval
        if args.eval_mode == "validator":
            _mining_dir = os.path.dirname(os.path.abspath(__file__))
            if _mining_dir not in sys.path:
                sys.path.insert(0, _mining_dir)
            from validator_eval import (  # noqa: E402
                DEFAULT_BLOCK_HASH,
                validator_style_paired_eval,
            )
            block_hash = (args.sim_block_hash or DEFAULT_BLOCK_HASH).strip()
            hotkey = (args.sim_hotkey or "").strip()
            if not hotkey:
                raise ValueError(
                    "validator eval mode requires --sim-hotkey or TEUTONIC_SIM_HOTKEY"
                )
            if args.eval_gpus:
                eval_gpu_ids = [int(x) for x in args.eval_gpus.split(",") if x.strip()]
            else:
                eval_gpu_ids = [0]
            verdict = validator_style_paired_eval(
                str(king_dir), str(merged_dir),
                block_hash=block_hash,
                hotkey=hotkey,
                n_public=args.n_eval,
                n_private=args.n_eval_private,
                seq_len=SEQ_LEN,
                gpu_ids=eval_gpu_ids,
                batch_size=args.eval_batch_size,
                eval_alpha=EVAL_ALPHA,
                delta_threshold=EVAL_DELTA,
                vocab_size=vocab_size,
            )
        else:
            if eval_arr is None or eval_indices is None:
                raise RuntimeError("local eval requires eval_arr (pretokenized or raw)")
            verdict = paired_eval(
                str(king_dir), str(merged_dir), eval_arr, eval_indices, "cuda:0",
            )
        verdict["iter"] = it
        verdict["seed"] = seed
        history.append(verdict)
        json.dump(verdict, open(iter_work / "verdict.json", "w"), indent=2)

        if best is None or verdict["mu_hat"] > best["mu_hat"]:
            best = {**verdict, "iter_dir": str(iter_work),
                    "merged_dir": str(merged_dir)}
        if verdict["mu_hat"] >= args.target_mu and verdict["accepted"]:
            log.info("target reached at iter %d", it)
            break

    final = {
        "king_repo": king["hf_repo"],
        "king_revision": king.get("king_revision"),
        "king_hash": king_hash,
        "best": best,
        "history": history,
        "ts": time.time(),
    }
    if args.report_out:
        Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(final, open(args.report_out, "w"), indent=2)
        log.info("wrote verdict to %s", args.report_out)

    # 8. optional upload
    if args.upload_repo and best and best["accepted"]:
        log.info("uploading %s -> %s", best["merged_dir"], args.upload_repo)
        api = HfApi(token=args.hf_token)
        api.create_repo(args.upload_repo, exist_ok=True, private=False)
        api.upload_folder(
            folder_path=best["merged_dir"],
            repo_id=args.upload_repo,
            commit_message=f"Teutonic challenger (mu_hat={best['mu_hat']:.6f})",
            allow_patterns=["*.safetensors", "config.json", "tokenizer*",
                            "special_tokens*", "generation_config.json"],
        )
        info = api.repo_info(args.upload_repo)
        final["uploaded_repo"] = args.upload_repo
        final["uploaded_revision"] = info.sha
        final["challenger_hash"] = sha256_dir(Path(best["merged_dir"]))
        if args.report_out:
            json.dump(final, open(args.report_out, "w"), indent=2)
        log.info("uploaded -> %s @ %s", args.upload_repo, info.sha[:12])
    elif args.upload_repo:
        log.warning("not uploading: best=%s", best)

    log.info("DONE — best mu_hat=%.6f accepted=%s",
             best["mu_hat"] if best else float("nan"),
             best["accepted"] if best else False)


if __name__ == "__main__":
    main()
