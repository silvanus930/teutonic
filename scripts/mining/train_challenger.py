#!/usr/bin/env python3
"""Teutonic mining harness — train a challenger that beats the current king.

Pipeline:
  1. Discover current king from R2 dashboard (repo + revision).
  2. Download king from HF (snapshot_download, pinned revision).
  3. Pull training data:
       --dataset-mode auto   → uses local Qwen .npy shards if LOCAL_DATASET_MANIFEST is set,
                               falls back to raw FineWeb-Edu download otherwise.
       --dataset-mode local  → local .npy shards from --local-dataset-manifest / env
       --dataset-mode remote → raw FineWeb-Edu (retokenize from Hippius parquet)
  4. Score sample sequences with the king (avg next-token loss).
       Score cache: scored refs saved to work/score_cache/king_<hash>/ — reused
       across iterations without re-forwarding through the king.
  5. Build a curriculum (general / hard / easy buckets, drop suspicious).
       Fractions: --general-frac --hard-frac --easy-frac (default 70/20/10).
  6. Train a LoRA adapter with torchrun multi-GPU on the chosen training mix.
       Presets: --candidate-preset {safe,main,aggressive} set lr/lora/epochs.
  7. Merge LoRA into the base weights → standalone candidate dir.
  8. Offline paired eval candidate vs king on a held-out shard slice
     (mirrors validator's compute_paired_losses + bootstrap LCB > delta).
       Acceptance gate: --acceptance-lcb-floor --mean-delta-floor
       Best candidate: selected by LCB (not just mu_hat).
  9. Emit a JSON verdict file. If accepted, optionally upload to HF.

REQUIRED — coldkey prefix in --upload-repo (since 2026-04-29):
  The validator rejects any HF repo whose name doesn't contain the first
  8 ss58 chars of the miner's coldkey. See submit_challenger.py.
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
# Constants
# ---------------------------------------------------------------------------
SEQ_LEN = 2048
EVAL_ALPHA = 0.001
EVAL_DELTA = 0.0025  # validator effect floor; see eval/torch_runner.py
LM_HEAD_CHUNK = 256
DASHBOARD_URL = os.environ.get(
    "TEUTONIC_DASHBOARD_URL",
    "https://us-east-1.hippius.com/teutonic-sn3/dashboard.json",
)
HIPPIUS_BASE = "https://s3.hippius.com/teutonic-sn3"

# Candidate training presets
# Keys with value None = not set (use CLI default).
# n_score=0 means skip king scoring; sample directly from local shards.
CANDIDATE_PRESETS: dict[str, dict] = {
    "safe": {
        "lr": 2e-5, "lora_r": 32, "lora_alpha": 64,
        "lora_dropout": 0.0, "epochs": 1.5, "target_mu": 0.0045,
        "lr_scheduler_type": "cosine", "warmup_ratio": 0.03,
        "min_lr_ratio": 0.10, "stable_ratio": 0.0,
        "adam_beta2": 0.95, "save_steps": 300, "eval_steps": 300,
    },
    "main": {
        "lr": 5e-5, "lora_r": 64, "lora_alpha": 128,
        "lora_dropout": 0.0, "epochs": 1.5, "target_mu": 0.006,
        "lr_scheduler_type": "cosine", "warmup_ratio": 0.03,
        "min_lr_ratio": 0.10, "stable_ratio": 0.0,
        "adam_beta2": 0.95, "save_steps": 300, "eval_steps": 300,
    },
    "aggressive": {
        "lr": 8e-5, "lora_r": 64, "lora_alpha": 128,
        "lora_dropout": 0.0, "epochs": 1.0, "target_mu": 0.007,
        "lr_scheduler_type": "cosine", "warmup_ratio": 0.03,
        "min_lr_ratio": 0.10, "stable_ratio": 0.0,
        "adam_beta2": 0.95, "save_steps": 200, "eval_steps": 200,
    },
    # Conservative run with WSD: warmup 3% → stable 65% → decay 32% → min_lr 10%
    "safe_strong": {
        "lr": 3e-5, "lora_r": 64, "lora_alpha": 128,
        "lora_dropout": 0.0, "epochs": 1.5, "target_mu": 0.006,
        "lr_scheduler_type": "wsd", "warmup_ratio": 0.03,
        "min_lr_ratio": 0.10, "stable_ratio": 0.65,
        "adam_beta2": 0.95, "save_steps": 300, "eval_steps": 300,
        "n_score": 64000, "train_per_iter": 32768, "val_size": 1500,
        "general_frac": 0.70, "hard_frac": 0.20, "easy_frac": 0.10,
        "fast_eval_n": 3000, "final_eval_n": 5000,
    },
    # ---- first-strike presets ----------------------------------------
    # Use when the king may be trained on a different dataset (e.g. old
    # CulturaX king vs new FineWeb-Edu challenge). Skip expensive king
    # scoring (n_score=0); sample directly from clean local shards and
    # train fast. Goal: get a positive submission in 20-30 minutes.
    "fast_first_strike": {
        "lr": 8e-5, "lora_r": 32, "lora_alpha": 64,
        "lora_dropout": 0.0, "epochs": 1.0, "target_mu": 0.0035,
        "lr_scheduler_type": "cosine", "warmup_steps": 30, "warmup_ratio": 0.0,
        "min_lr_ratio": 0.25, "stable_ratio": 0.0,
        "adam_beta2": 0.95, "save_steps": 200, "eval_steps": 200,
        "n_score": 0, "train_per_iter": 8192, "val_size": 512,
        "general_frac": 0.80, "hard_frac": 0.10, "easy_frac": 0.10,
        "fast_eval_n": 1000, "final_eval_n": 2000,
    },
    "safe_first_strike": {
        "lr": 5e-5, "lora_r": 32, "lora_alpha": 64,
        "lora_dropout": 0.0, "epochs": 1.0, "target_mu": 0.004,
        "lr_scheduler_type": "cosine", "warmup_steps": 30, "warmup_ratio": 0.0,
        "min_lr_ratio": 0.20, "stable_ratio": 0.0,
        "adam_beta2": 0.95, "save_steps": 200, "eval_steps": 200,
        "n_score": 0, "train_per_iter": 16384, "val_size": 1024,
        "general_frac": 0.80, "hard_frac": 0.10, "easy_frac": 0.10,
        "fast_eval_n": 1500, "final_eval_n": 3000,
    },
    # ---- follow-up preset -------------------------------------------
    # Run after an initial first-strike submission to get a better,
    # scoring-backed challenger with tighter LCB guarantees.
    "strong_followup": {
        "lr": 5e-5, "lora_r": 64, "lora_alpha": 128,
        "lora_dropout": 0.0, "epochs": 1.5, "target_mu": 0.006,
        "lr_scheduler_type": "cosine", "warmup_ratio": 0.03,
        "min_lr_ratio": 0.10, "stable_ratio": 0.0,
        "adam_beta2": 0.95, "save_steps": 300, "eval_steps": 300,
        "n_score": 64000, "train_per_iter": 32768, "val_size": 1500,
        "general_frac": 0.70, "hard_frac": 0.20, "easy_frac": 0.10,
        "fast_eval_n": 3000, "final_eval_n": 5000,
    },
    "custom": {},  # all values from explicit CLI args
}


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
    """Load a 1D or 2D uint32 .npy shard, reshape into (n_seq, seq_len)."""
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


def fetch_manifest(cache: Path, local_manifest_path: str = "") -> dict:
    local = local_manifest_path or os.environ.get("LOCAL_DATASET_MANIFEST", "")
    if local:
        p = Path(local)
        if not p.exists():
            raise FileNotFoundError(f"local dataset manifest not found: {p}")
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
    if vocab_size is None:
        return
    max_id = int(arr.max()) if arr.size else -1
    if max_id >= vocab_size:
        raise ValueError(
            f"{label}: token id {max_id} >= model vocab_size {vocab_size}. "
            "Pretokenized v2 shards use the Gemma tokenizer; Qwen3 kings need "
            "--dataset-mode remote (FineWeb-Edu retokenized at load time)."
        )


def pretokenized_incompatible_with_king(manifest: dict, king_dir: Path) -> bool:
    shard_tok = (
        manifest.get("tokenizer") or manifest.get("tokenizer_dir") or ""
    ).lower()
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
        prefix=os.environ.get(
            "TEUTONIC_RAW_DATASET_PREFIX",
            "hf-mirrors/HuggingFaceFW/fineweb-edu/data",
        ).strip("/"),
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
        def ds_get(self, _key):
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
        eos_id = tokenizer.eos_token_id or tokenizer.sep_token_id
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
        "raw dataset: %d sequences, max_id=%d, mode=%s",
        len(arr), int(arr.max()), meta.get("mode"),
    )
    return arr


def resolve_dataset_mode(requested: str, manifest: dict, king_dir: Path) -> str:
    # Normalize aliases
    if requested == "local":
        requested = "pretokenized"
    elif requested == "remote":
        requested = "raw"

    if requested != "auto":
        return requested
    if os.environ.get("LOCAL_DATASET_MANIFEST"):
        log.info(
            "auto dataset mode: LOCAL_DATASET_MANIFEST set → pretokenized local shards "
            "(tokenizer=%r)",
            manifest.get("tokenizer") or manifest.get("tokenizer_dir"),
        )
        return "pretokenized"
    if pretokenized_incompatible_with_king(manifest, king_dir):
        log.warning(
            "auto dataset mode: v2 manifest tokenizer=%r incompatible with Qwen king; "
            "using raw FineWeb-Edu (set LOCAL_DATASET_MANIFEST to skip per-run downloads)",
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
    repo = k.get("hf_repo") or k.get("model_repo") or k.get("previous_repo")
    revision = k.get("king_revision") or k.get("revision") or k.get("king_digest")
    if not repo:
        raise KeyError(f"no king repo in dashboard: {list(k.keys())}")
    k["hf_repo"] = repo
    k["king_revision"] = revision
    log.info(
        "king: repo=%s revision=%s reign=%d hotkey=%s",
        repo, (revision or "HEAD")[:20], k.get("reign_number", 0),
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


def manifest_hash(manifest: dict) -> str:
    """Stable hash of the manifest shard keys for score cache keying."""
    h = hashlib.sha256()
    for s in manifest.get("shards", []):
        h.update(s.get("key", "").encode())
        h.update(b"|")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Paired eval (mirrors eval_torch.compute_paired_losses + bootstrap)
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_per_seq_loss(model, token_batches, device, chunk=LM_HEAD_CHUNK):
    input_ids = torch.tensor(token_batches, dtype=torch.long, device=device)
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


def paired_eval(
    king_dir: str, chall_dir: str, shard: np.ndarray,
    indices: list[int], device: str, batch_size: int = 8,
    n_bootstrap: int = 10000, alpha: float = EVAL_ALPHA,
    acceptance_lcb_floor: float = EVAL_DELTA,
    mean_delta_floor: float = 0.0,
) -> dict:
    """Local paired bootstrap test mirroring the validator."""
    delta = EVAL_DELTA
    log.info("paired_eval: loading king %s on %s", king_dir, device)
    king = AutoModelForCausalLM.from_pretrained(
        king_dir, torch_dtype=torch.bfloat16, device_map={"": device},
        use_safetensors=True,
    )
    king.eval()
    log.info("paired_eval: loading challenger %s", chall_dir)
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

    # Acceptance: lcb must exceed floor AND mean_delta must exceed floor
    accepted = (
        lcb > acceptance_lcb_floor
        and mu_hat >= mean_delta_floor
    )
    rejection_reasons = []
    if lcb <= acceptance_lcb_floor:
        rejection_reasons.append(
            f"lcb={lcb:.6f} <= acceptance_lcb_floor={acceptance_lcb_floor:.6f}"
        )
    if mean_delta_floor > 0 and mu_hat < mean_delta_floor:
        rejection_reasons.append(
            f"mu_hat={mu_hat:.6f} < mean_delta_floor={mean_delta_floor:.6f}"
        )

    res = {
        "eval_mode": "local",
        "n_eval": n_done,
        "mu_hat": mu_hat,
        "mean_delta": mu_hat,
        "lcb": lcb,
        "delta": delta,
        "delta_threshold": delta,
        "alpha": alpha,
        "acceptance_lcb_floor": acceptance_lcb_floor,
        "mean_delta_floor": mean_delta_floor,
        "accepted": accepted,
        "rejection_reasons": rejection_reasons,
        "avg_king_loss": king_sum / max(1, n_done),
        "avg_chall_loss": chall_sum / max(1, n_done),
        "avg_challenger_loss": chall_sum / max(1, n_done),
        "elapsed_s": time.time() - t0,
        "note": "local mode: holdout from mining data pool (not validator seeds)",
    }
    log.info("paired_eval: mu_hat=%.6f lcb=%.6f accepted=%s rejection=%s",
             mu_hat, lcb, accepted, rejection_reasons or "none")
    del king, chall
    torch.cuda.empty_cache()
    return res


# ---------------------------------------------------------------------------
# Score cache
# ---------------------------------------------------------------------------
def _score_cache_dir(work: Path, king_hash: str, manifest: dict) -> Path:
    mhash = manifest_hash(manifest)[:8]
    return work / "score_cache" / f"king_{king_hash[:16]}" / f"manifest_{mhash}"


def _shard_key_hash(shard_key: str) -> str:
    """Stable 12-char hash of a manifest shard key for cache file naming.
    Uses shard key (not local list index) so cache is stable across --shard-start changes.
    """
    return hashlib.sha256(shard_key.encode()).hexdigest()[:12]


def _load_score_cache(
    cache_dir: Path, shard_keys: list[str],
) -> list[dict] | None:
    """Load scored rows for specified manifest shard keys.
    Returns None if any shard's cache file is missing.
    Normalises 'shard_id'/'shard_key' back to local 'shard' index.
    """
    key_to_local = {k: i for i, k in enumerate(shard_keys)}
    all_rows: list[dict] = []
    for shard_key in shard_keys:
        fname = f"scored_shard_{_shard_key_hash(shard_key)}.jsonl"
        p = cache_dir / fname
        if not p.is_file():
            log.info("score cache miss: %s (key=%s)", fname, shard_key)
            return None
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                # Map shard_key back to local shard index
                sk = r.get("shard_key", shard_key)
                r["shard"] = key_to_local.get(sk, r.get("shard", 0))
                if "idx" not in r and "row_idx" in r:
                    r["idx"] = r["row_idx"]
                all_rows.append(r)
    log.info("score cache hit: %d rows from %d shard(s)", len(all_rows), len(shard_keys))
    return all_rows


def _save_score_cache(
    cache_dir: Path, shard_keys: list[str], rows: list[dict],
) -> None:
    """Save scored rows keyed by manifest shard key hash (not local list index).
    This makes the cache stable across --shard-start / --n-shards changes.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Group rows by their local shard index
    by_local: dict[int, list[dict]] = {i: [] for i in range(len(shard_keys))}
    for r in rows:
        si = int(r.get("shard", 0))
        if si in by_local:
            by_local[si].append(r)
    for local_idx, shard_rows in by_local.items():
        shard_key = shard_keys[local_idx]
        khash = _shard_key_hash(shard_key)
        fname = f"scored_shard_{khash}.jsonl"
        p = cache_dir / fname
        tmp = cache_dir / f"{fname}.tmp"
        with open(tmp, "w") as f:
            for r in shard_rows:
                f.write(json.dumps({
                    "shard_key": shard_key,
                    "shard_key_hash": khash,
                    "row_idx": r.get("idx"),
                    "loss": r.get("loss"),
                    "bucket": r.get("bucket", ""),
                    "unique_r": r.get("unique_r"),
                    "rep_r": r.get("rep_r"),
                    "rep_ng4": r.get("rep_ng4"),
                }) + "\n")
        shutil.move(str(tmp), str(p))
    log.info("score cache saved: %d shard file(s) in %s", len(by_local), cache_dir)


# ---------------------------------------------------------------------------
# Scoring helpers
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


# ---------------------------------------------------------------------------
# Score + curriculum  (lightweight — no tokens stored in rows)
# ---------------------------------------------------------------------------
def score_and_curate(
    king_dir: str,
    shards: list[np.ndarray],
    shard_keys: list[str],
    manifest: dict,
    n_score: int,
    train_per_iter: int,
    val_size: int,
    seed: int,
    device: str,
    work: Path,
    general_frac: float = 0.70,
    hard_frac: float = 0.20,
    easy_frac: float = 0.10,
    max_suspicious_frac: float = 0.0,
    score_cache_path: Path | None = None,
    force_rescore: bool = False,
    king_hash: str = "",
) -> tuple[Path, Path]:
    """Score n_score random samples on the king, build curriculum, write train/val jsonl.

    LIGHTWEIGHT: token lists are NOT stored in scored rows. When writing jsonl
    files we load tokens directly from the shard arrays already in memory, so
    RAM usage is proportional to n_score * ~6 floats rather than n_score * seq_len * 4.

    shard_keys: manifest key for each shard (same order as shards list).
    Used to key the score cache by manifest identity, not local list index,
    so the cache is stable across --shard-start / --n-shards changes.
    """
    assert len(shard_keys) == len(shards), \
        f"shard_keys length {len(shard_keys)} != shards length {len(shards)}"
    rng = np.random.default_rng(seed)
    cands = global_sample_shard_indices(shards, n_score, rng)
    rng.shuffle(cands)
    # shard_keys for the actually sampled shards (unique local indices)
    sampled_local_idxs = sorted({s for s, _ in cands})
    sampled_keys = [shard_keys[i] for i in sampled_local_idxs]

    # --- score cache (keyed by manifest shard keys, not local indices) ---
    cached_rows: list[dict] | None = None
    if score_cache_path and not force_rescore and score_cache_path.is_dir():
        cached_rows = _load_score_cache(score_cache_path, sampled_keys)

    if cached_rows is not None:
        rows = cached_rows
        # Normalize field names from cache format
        for r in rows:
            if "shard_id" in r and "shard" not in r:
                r["shard"] = r["shard_id"]
            if "row_idx" in r and "idx" not in r:
                r["idx"] = r["row_idx"]
    else:
        log.info(
            "scoring: %d candidates across %d shard(s) | seed=%d | king=%s | device=%s",
            len(cands), len(shards), seed, king_dir, device,
        )
        t_score = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            king_dir, torch_dtype=torch.bfloat16, device_map={"": device},
            use_safetensors=True,
        )
        model.eval()

        rows: list[dict] = []
        BATCH = 8
        log_every = max(1, len(cands) // (BATCH * 10))
        for batch_i, i in enumerate(range(0, len(cands), BATCH)):
            chunk = cands[i:i + BATCH]
            toks = [shards[s][j].tolist() for s, j in chunk]
            losses = compute_per_seq_loss(model, toks, device)
            for (s_idx, j), tok, loss in zip(chunk, toks, losses):
                arr = np.asarray(tok)
                unique_r = float(len(set(tok)) / max(1, len(tok)))
                rep_r = float(np.mean(arr[1:] == arr[:-1])) if len(arr) > 1 else 0.0
                ngrams = [tuple(tok[k:k + 4]) for k in range(len(tok) - 3)]
                rep_ng = 1.0 - len(set(ngrams)) / max(1, len(ngrams)) if ngrams else 0.0
                rows.append({
                    "shard": s_idx,
                    "idx": j,
                    "loss": float(loss),
                    "unique_r": unique_r,
                    "rep_r": rep_r,
                    "rep_ng4": rep_ng,
                    # NOTE: no "tokens" field — load from shard arrays when writing jsonl
                })
            if batch_i % log_every == 0 or i + BATCH >= len(cands):
                done = min(i + BATCH, len(cands))
                run_losses = np.asarray([r["loss"] for r in rows])
                ls = _loss_summary(run_losses)
                log.info(
                    "scoring %d/%d (%.1f%%) | mean=%.4f p50=%.4f | %.1fs",
                    done, len(cands), 100.0 * done / max(1, len(cands)),
                    ls.get("mean", float("nan")), ls.get("p50", float("nan")),
                    time.time() - t_score,
                )

        del model
        torch.cuda.empty_cache()
        log.info("scoring done in %.1fs (%d rows)", time.time() - t_score, len(rows))

        if score_cache_path:
            _save_score_cache(score_cache_path, shard_keys, rows)

    # --- loss distribution ---
    losses = np.asarray([r["loss"] for r in rows])
    loss_stats = _loss_summary(losses)
    p50 = loss_stats.get("p50", float(np.median(losses)))
    p85 = loss_stats.get("p85", float(np.percentile(losses, 85)))
    general_floor = p50 * 0.8

    log.info(
        "loss dist: min=%.4f max=%.4f mean=%.4f std=%.4f "
        "p10=%.4f p25=%.4f p50=%.4f p75=%.4f p85=%.4f p90=%.4f",
        loss_stats.get("min", 0), loss_stats.get("max", 0),
        loss_stats.get("mean", 0), loss_stats.get("std", 0),
        loss_stats.get("p10", 0), loss_stats.get("p25", 0), p50,
        loss_stats.get("p75", 0), p85, loss_stats.get("p90", 0),
    )
    log.info(
        "bucket thresholds: suspicious=rep_r>0.2|rep_ng4>0.5|unique_r<0.05 | "
        "hard=loss>=%.4f | general=loss>=%.4f | else=easy",
        p85, general_floor,
    )

    def _bucket(r: dict) -> str:
        if r["rep_r"] > 0.2 or r["rep_ng4"] > 0.5 or r["unique_r"] < 0.05:
            return "suspicious"
        if math.isnan(r["loss"]) or math.isinf(r["loss"]):
            return "suspicious"
        if r["loss"] >= p85:
            return "hard"
        if r["loss"] >= general_floor:
            return "general"
        return "easy"

    for r in rows:
        if "bucket" not in r or not r["bucket"]:
            r["bucket"] = _bucket(r)

    counts = {b: sum(1 for r in rows if r.get("bucket") == b)
              for b in ("general", "hard", "easy", "suspicious")}
    bucket_loss = _bucket_means(rows)
    log.info(
        "buckets: %s | mean loss: %s",
        counts,
        {k: f"{v:.4f}" for k, v in bucket_loss.items()},
    )

    # --- curriculum ---
    clean = [r for r in rows if r.get("bucket") != "suspicious"]
    dropped = len(rows) - len(clean)
    log.info("dropped %d suspicious (%.1f%%), %d clean remain",
             dropped, 100.0 * dropped / max(1, len(rows)), len(clean))

    rng2 = np.random.default_rng(seed + 1)
    order = rng2.permutation(len(clean)).tolist()
    clean_shuffled = [clean[i] for i in order]
    val_rows = clean_shuffled[:val_size]
    val_keys = {(r["shard"], r["idx"]) for r in val_rows}
    pool = [r for r in clean_shuffled if (r["shard"], r["idx"]) not in val_keys]

    log.info("val=%d (requested %d), train pool=%d", len(val_rows), val_size, len(pool))

    # Normalize fractions
    total_frac = general_frac + hard_frac + easy_frac
    if total_frac > 0:
        gf = general_frac / total_frac
        hf = hard_frac / total_frac
        ef = easy_frac / total_frac
    else:
        gf, hf, ef = 0.70, 0.20, 0.10

    general = [r for r in pool if r.get("bucket") == "general"]
    hard = [r for r in pool if r.get("bucket") == "hard"]
    easy = [r for r in pool if r.get("bucket") == "easy"]
    n_general = int(train_per_iter * gf)
    n_hard = int(train_per_iter * hf)
    n_easy = train_per_iter - n_general - n_hard

    log.info(
        "curriculum mix target: general=%d (%.0f%%) hard=%d (%.0f%%) easy=%d (%.0f%%) | "
        "pool: general=%d hard=%d easy=%d",
        n_general, gf * 100, n_hard, hf * 100, n_easy, ef * 100,
        len(general), len(hard), len(easy),
    )

    train_rows: list[dict] = []
    picked: dict[str, int] = {}
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
        else:
            sel = rng2.choice(len(src), size=n, replace=False)
            train_rows.extend(src[int(k)] for k in sel)
            picked[label] = n
        log.info("curriculum: %s → picked %d/%d", label, picked[label], len(src))

    order2 = rng2.permutation(len(train_rows)).tolist()
    train_rows = [train_rows[i] for i in order2]

    train_mix = {b: sum(1 for r in train_rows if r.get("bucket") == b)
                 for b in ("general", "hard", "easy")}
    if train_rows:
        tr_ls = _loss_summary(np.asarray([r["loss"] for r in train_rows]))
        log.info(
            "curriculum train: %d sequences | mix %s | loss mean=%.4f p50=%.4f",
            len(train_rows), train_mix, tr_ls.get("mean", 0), tr_ls.get("p50", 0),
        )
    if len(train_rows) < train_per_iter:
        log.warning(
            "train set undersized (%d < %d); increase --n-score or data pool",
            len(train_rows), train_per_iter,
        )

    # Print curriculum summary
    print(
        f"\n{'='*50}\n"
        f"CURRICULUM SUMMARY\n"
        f"  scored: {len(rows)} | clean: {len(clean)} | "
        f"suspicious: {dropped} ({100.0*dropped/max(1,len(rows)):.1f}%)\n"
        f"  train: {len(train_rows)} | val: {len(val_rows)}\n"
        f"  mix: general={train_mix.get('general',0)} "
        f"hard={train_mix.get('hard',0)} easy={train_mix.get('easy',0)}\n"
        f"  loss p50={p50:.4f} p85={p85:.4f}\n"
        f"{'='*50}\n",
        flush=True,
    )

    # --- write jsonl ---
    work.mkdir(parents=True, exist_ok=True)
    train_p = work / "train.jsonl"
    val_p = work / "val.jsonl"
    scored_p = work / "scored.jsonl"
    scoring_p = work / "scoring.json"
    curriculum_p = work / "curriculum.json"

    # scored.jsonl — lightweight refs only
    with open(scored_p, "w") as f:
        for r in rows:
            f.write(json.dumps({
                "shard": r["shard"], "idx": r["idx"],
                "loss": r["loss"], "unique_r": r["unique_r"],
                "rep_r": r["rep_r"], "rep_ng4": r["rep_ng4"], "bucket": r.get("bucket", ""),
            }) + "\n")

    # train.jsonl — tokens loaded from shard arrays (never stored in rows)
    with open(train_p, "w") as f:
        for r in train_rows:
            toks = shards[r["shard"]][r["idx"]].tolist()
            f.write(json.dumps({"input_ids": toks}) + "\n")

    # val.jsonl — tokens loaded from shard arrays
    with open(val_p, "w") as f:
        for r in val_rows:
            toks = shards[r["shard"]][r["idx"]].tolist()
            f.write(json.dumps({"input_ids": toks}) + "\n")

    scoring_report = {
        "seed": seed,
        "n_candidates": len(cands),
        "loss": loss_stats,
        "thresholds": {"p50": p50, "p85": p85, "general_floor": general_floor},
        "bucket_counts": counts,
        "bucket_mean_loss": bucket_loss,
    }
    curriculum_report = {
        "seed": seed + 1,
        "train_per_iter": train_per_iter,
        "val_size": val_size,
        "general_frac": general_frac,
        "hard_frac": hard_frac,
        "easy_frac": easy_frac,
        "suspicious_dropped": dropped,
        "clean_pool": len(clean),
        "train_pool": len(pool),
        "picked": picked,
        "train_mix": train_mix,
        "train_n": len(train_rows),
        "val_n": len(val_rows),
        "val_bucket_counts": {b: sum(1 for r in val_rows if r.get("bucket") == b)
                              for b in ("general", "hard", "easy")},
        "train_loss": _loss_summary(np.asarray([r["loss"] for r in train_rows])) if train_rows else {},
        "val_loss": _loss_summary(np.asarray([r["loss"] for r in val_rows])) if val_rows else {},
        "shards_used": shard_indices,
    }
    json.dump(scoring_report, open(scoring_p, "w"), indent=2)
    json.dump(curriculum_report, open(curriculum_p, "w"), indent=2)

    log.info("wrote train=%d val=%d | %s %s", len(train_rows), len(val_rows), train_p, val_p)
    return train_p, val_p


# ---------------------------------------------------------------------------
# LoRA training
# ---------------------------------------------------------------------------
def run_lora_training(
    base_model: str, train_p: Path, val_p: Path,
    out_dir: Path, n_gpus: int, args: argparse.Namespace,
    bundle: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "torchrun", f"--nproc_per_node={n_gpus}",
        str(bundle / "train_lora_token_ids.py"),
        "--base-model", base_model,
        "--train-data", str(train_p),
        "--val-data", str(val_p),
        "--output-dir", str(out_dir),
        "--seq-len", str(args.seq_len),
        "--micro-batch-size", str(args.micro_batch),
        "--grad-accum", str(args.grad_accum),
        "--learning-rate", str(args.lr),
        "--epochs", str(args.epochs),
        "--lora-r", str(args.lora_r),
        "--lora-alpha", str(args.lora_alpha),
        "--lora-dropout", str(args.lora_dropout),
        "--weight-decay", str(args.weight_decay),
        "--warmup-ratio", str(args.warmup_ratio),
        "--warmup-steps", str(args.warmup_steps),
        "--min-lr-ratio", str(args.min_lr_ratio),
        "--stable-ratio", str(args.stable_ratio),
        "--adam-beta2", str(args.adam_beta2),
        "--max-grad-norm", str(args.max_grad_norm),
        "--dtype", args.dtype,
        "--logging-steps", str(args.logging_steps),
        "--save-steps", str(args.save_steps),
        "--eval-steps", str(args.eval_steps),
        "--lr-scheduler-type", args.lr_scheduler_type,
        "--average-top-k-lora-checkpoints", str(args.average_top_k_lora_checkpoints),
    ]
    if args.gradient_checkpointing:
        cmd.append("--gradient-checkpointing")
    else:
        cmd.append("--no-gradient-checkpointing")
    if args.resume_checkpoint:
        cmd.extend(["--resume-from-checkpoint", str(args.resume_checkpoint)])
    log.info("training: %s", " ".join(cmd))
    t0 = time.time()
    subprocess.check_call(cmd)
    log.info("training done in %.1fs", time.time() - t0)

    adapter = out_dir / "best_adapter"
    if not adapter.exists():
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
# Direct shard sampling (n_score=0 / first-strike mode)
# ---------------------------------------------------------------------------
def sample_direct_from_shards(
    shards: list[np.ndarray],
    train_per_iter: int,
    val_size: int,
    seed: int,
    work: Path,
) -> tuple[Path, Path]:
    """Sample train/val sequences directly from local shards without king scoring.

    Used when n_score=0 (first-strike presets). No king model forward pass;
    relies purely on token-level quality stats to filter suspicious windows.
    Tokens are never stored in memory — loaded from shard arrays at write time.
    """
    n_needed = train_per_iter + val_size
    # Oversample 4× to allow filtering out suspicious windows
    n_candidates = min(n_needed * 4, sum(len(s) for s in shards))
    rng = np.random.default_rng(seed)
    cands = global_sample_shard_indices(shards, n_candidates, rng)
    rng.shuffle(cands)

    log.info(
        "direct sampling: need=%d (train=%d val=%d), candidates=%d from %d shard(s)",
        n_needed, train_per_iter, val_size, len(cands), len(shards),
    )

    clean_refs: list[dict] = []
    n_suspicious = 0
    for s_idx, j in cands:
        if len(clean_refs) >= n_needed:
            break
        tok = shards[s_idx][j].tolist()
        arr = np.asarray(tok)
        unique_r = float(len(set(tok)) / max(1, len(tok)))
        rep_r = float(np.mean(arr[1:] == arr[:-1])) if len(arr) > 1 else 0.0
        ngrams = [tuple(tok[k:k + 4]) for k in range(len(tok) - 3)]
        rep_ng = 1.0 - len(set(ngrams)) / max(1, len(ngrams)) if ngrams else 0.0
        if rep_r > 0.2 or rep_ng > 0.5 or unique_r < 0.05:
            n_suspicious += 1
            continue
        clean_refs.append({"shard": s_idx, "idx": j,
                           "unique_r": unique_r, "rep_r": rep_r, "rep_ng4": rep_ng})

    log.info(
        "direct sampling: %d clean refs collected (filtered %d suspicious)",
        len(clean_refs), n_suspicious,
    )
    if len(clean_refs) < n_needed:
        log.warning(
            "direct sampling: only %d clean refs available, needed %d — "
            "consider adding more local shards",
            len(clean_refs), n_needed,
        )

    # Split train / val
    rng2 = np.random.default_rng(seed + 1)
    order = rng2.permutation(len(clean_refs)).tolist()
    shuffled = [clean_refs[i] for i in order]
    val_rows = shuffled[:val_size]
    train_rows = shuffled[val_size:val_size + train_per_iter]

    work.mkdir(parents=True, exist_ok=True)
    train_p = work / "train.jsonl"
    val_p = work / "val.jsonl"
    curriculum_p = work / "curriculum.json"

    with open(train_p, "w") as f:
        for r in train_rows:
            toks = shards[r["shard"]][r["idx"]].tolist()
            f.write(json.dumps({"input_ids": toks}) + "\n")
    with open(val_p, "w") as f:
        for r in val_rows:
            toks = shards[r["shard"]][r["idx"]].tolist()
            f.write(json.dumps({"input_ids": toks}) + "\n")

    json.dump({
        "mode": "direct_sample",
        "seed": seed,
        "n_candidates": len(cands),
        "n_suspicious_filtered": n_suspicious,
        "train_n": len(train_rows),
        "val_n": len(val_rows),
        "shards_used": sorted({r["shard"] for r in clean_refs}),
    }, open(curriculum_p, "w"), indent=2)

    log.info(
        "direct sample wrote train=%d val=%d | %s %s",
        len(train_rows), len(val_rows), train_p, val_p,
    )
    print(
        f"\n{'='*50}\n"
        f"DIRECT SAMPLE (n_score=0 — no king scoring)\n"
        f"  candidates: {len(cands)} | suspicious filtered: {n_suspicious}\n"
        f"  train: {len(train_rows)} | val: {len(val_rows)}\n"
        f"{'='*50}\n",
        flush=True,
    )
    return train_p, val_p


# ---------------------------------------------------------------------------
# Candidate ranking helper
# ---------------------------------------------------------------------------
def _rank_candidates(verdicts: list[dict]) -> dict | None:
    """Return the best verdict by: accepted > lcb > mean_delta > lowest challenger loss."""
    if not verdicts:
        return None
    def _sort_key(v: dict) -> tuple:
        return (
            int(v.get("accepted", False)),
            v.get("lcb", v.get("mu_hat", -999)),
            v.get("mean_delta", v.get("mu_hat", -999)),
            -v.get("avg_chall_loss", v.get("avg_challenger_loss", 999)),
        )
    return max(verdicts, key=_sort_key)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Work / bundle
    ap.add_argument("--work", default="/root/teutonic-mining/work")
    ap.add_argument("--bundle", default="/root/teutonic-mining/bundle")

    # Dataset
    ap.add_argument(
        "--dataset-mode",
        choices=("auto", "local", "pretokenized", "remote", "raw"),
        default="auto",
        help="auto: local shards if LOCAL_DATASET_MANIFEST set, else remote; "
             "local/pretokenized: .npy shards; remote/raw: FineWeb-Edu parquet",
    )
    ap.add_argument(
        "--local-dataset-manifest", default="",
        help="Path to local manifest.json (overrides LOCAL_DATASET_MANIFEST env var)",
    )
    ap.add_argument("--n-shards", type=int, default=0)
    ap.add_argument("--shard-start", type=int, default=0)
    ap.add_argument("--raw-max-files", type=int, default=RAW_MAX_FILES)
    ap.add_argument("--seq-len", type=int, default=2048)

    # Eval holdout
    ap.add_argument("--eval-shard", type=int, default=-1)
    ap.add_argument(
        "--eval-mode", choices=("validator", "local"), default="validator",
        help="validator: uses eval_server seed path; local: fast mining-pool eval",
    )
    ap.add_argument("--sim-block-hash", default=os.environ.get("TEUTONIC_SIM_BLOCK_HASH", ""))
    ap.add_argument("--sim-hotkey", default=os.environ.get("TEUTONIC_SIM_HOTKEY", ""))
    ap.add_argument("--n-eval", type=int, default=25600)
    ap.add_argument("--n-eval-private", type=int, default=0)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--eval-gpus", default="")

    # Two-stage eval
    ap.add_argument("--fast-eval", action="store_true",
                    help="Run a fast local eval pass before the final validator eval")
    ap.add_argument("--fast-eval-n", type=int, default=None,
                    help="Sequences for fast eval. Default from preset or 3000.")
    ap.add_argument("--no-fast-eval", action="store_false", dest="fast_eval")
    ap.add_argument("--final-eval-n", type=int, default=None,
                    help="Sequences for final eval (overrides --n-eval for final stage). "
                         "Default from preset.")

    # Scoring / curriculum
    ap.add_argument("--n-score", type=int, default=None,
                    help="Sequences to score with king. 0 = skip scoring, sample directly. "
                         "Default from preset or 4000.")
    ap.add_argument("--train-per-iter", type=int, default=None,
                    help="Training sequences per iteration. Default from preset or 4000.")
    ap.add_argument("--val-size", type=int, default=None,
                    help="Validation sequences. Default from preset or 400.")
    ap.add_argument("--seed", type=int, default=42)

    # Score cache
    ap.add_argument("--use-local-score-cache", action="store_true", default=True)
    ap.add_argument("--no-use-local-score-cache", action="store_false", dest="use_local_score_cache")
    ap.add_argument("--score-cache-dir", default="",
                    help="Score cache root dir. Default: <work>/score_cache")
    ap.add_argument("--force-rescore", action="store_true")

    # Curriculum fractions (None = from preset)
    ap.add_argument("--general-frac", type=float, default=None)
    ap.add_argument("--hard-frac", type=float, default=None)
    ap.add_argument("--easy-frac", type=float, default=None)
    ap.add_argument("--max-suspicious-frac", type=float, default=0.0)

    # Iteration control
    ap.add_argument("--max-iters", type=int, default=3)
    ap.add_argument("--target-mu", type=float, default=0.006)

    # Acceptance gate
    ap.add_argument("--acceptance-lcb-floor", type=float, default=EVAL_DELTA,
                    help="Min bootstrap LCB to accept a candidate (default=validator floor)")
    ap.add_argument("--mean-delta-floor", type=float, default=0.0)
    ap.add_argument("--min-shard-lcb-floor", type=float, default=0.0)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--alpha", type=float, default=EVAL_ALPHA)

    # First-strike mode
    ap.add_argument(
        "--first-strike", action="store_true",
        help="First-strike mode: requires local dataset manifest, skips expensive king "
             "scoring (uses n_score=0 direct sampling), enables fast eval, sets "
             "eval-mode=local unless --eval-mode validator is explicit. Designed for "
             "a quick 20-30 min candidate when the king may be on a different distribution.",
    )

    # Candidate preset
    ap.add_argument(
        "--candidate-preset",
        choices=(
            "safe", "main", "aggressive", "safe_strong",
            "fast_first_strike", "safe_first_strike", "strong_followup",
            "custom",
        ),
        default="main",
        help="Training hyperparameter preset. "
             "first-strike presets skip king scoring (n_score=0). "
             "safe_strong uses WSD scheduler. "
             "Explicit CLI args always override preset values.",
    )

    # Training hyperparams (explicit non-None values override preset)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--lora-r", type=int, default=None)
    ap.add_argument("--lora-alpha", type=int, default=None)
    ap.add_argument("--lora-dropout", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-ratio", type=float, default=None,
                    help="Warmup fraction of total steps. Default from preset or 0.03.")
    ap.add_argument("--warmup-steps", type=int, default=None,
                    help="Absolute warmup steps (overrides warmup-ratio when > 0). "
                         "Default from preset.")
    ap.add_argument("--min-lr-ratio", type=float, default=None,
                    help="Min LR at end of schedule = lr * min_lr_ratio. "
                         "Default from preset or 0.10.")
    ap.add_argument("--stable-ratio", type=float, default=None,
                    help="WSD: fraction of total steps for stable-LR phase. "
                         "Default from preset or 0.0.")
    ap.add_argument("--adam-beta2", type=float, default=None,
                    help="AdamW beta2. Default from preset or 0.95.")
    ap.add_argument("--max-grad-norm", type=float, default=0.3)
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--n-gpus", type=int, default=1)
    ap.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    ap.add_argument("--gradient-checkpointing", action="store_true", default=True)
    ap.add_argument("--no-gradient-checkpointing", action="store_false", dest="gradient_checkpointing")
    ap.add_argument("--lr-scheduler-type", default=None,
                    choices=("constant", "constant_with_warmup", "linear", "cosine", "wsd"),
                    help="LR scheduler type. Default from preset or cosine.")
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--save-steps", type=int, default=None,
                    help="Checkpoint save frequency. Default from preset or 300.")
    ap.add_argument("--eval-steps", type=int, default=None,
                    help="Eval frequency. Default from preset or 300.")
    # Checkpoint averaging
    ap.add_argument("--average-top-k-lora-checkpoints", type=int, default=0,
                    help="Average top-K LoRA adapters by eval_loss after training. "
                         "0 = disabled. Averaged model used only if it improves val loss.")

    # Resume / skip
    ap.add_argument("--resume-checkpoint", default="")
    ap.add_argument("--from-iter", type=int, default=-1)
    ap.add_argument("--skip-scoring", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="Re-run even if verdict.json already exists for that iter")

    # Profile (legacy backward compat)
    ap.add_argument("--profile", choices=("default", "prod"), default="default")

    # Upload / report
    ap.add_argument("--upload-repo", default="")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    ap.add_argument("--report-out", default="")

    args = ap.parse_args()

    # --- apply preset (explicit CLI args with non-None default override preset) ---
    preset = CANDIDATE_PRESETS.get(args.candidate_preset, {})

    def _p(attr: str, key: str | None = None, fallback=None):
        """Apply preset[key] to args.attr if args.attr is None."""
        k = key or attr
        if getattr(args, attr, None) is None:
            setattr(args, attr, preset.get(k, fallback))

    # Model / training hyperparams
    _p("lr", fallback=5e-5)
    _p("epochs", fallback=1.5)
    _p("lora_r", fallback=64)
    _p("lora_alpha", fallback=128)
    _p("lora_dropout", fallback=0.0)

    # Scheduler
    _p("lr_scheduler_type", fallback="cosine")
    _p("warmup_ratio", fallback=0.03)
    _p("warmup_steps", fallback=0)
    _p("min_lr_ratio", fallback=0.10)
    _p("stable_ratio", fallback=0.0)
    _p("adam_beta2", fallback=0.95)
    _p("save_steps", fallback=300)
    _p("eval_steps", fallback=300)

    # Data / curriculum (preset overrides when user hasn't explicitly set)
    _p("n_score", fallback=4000)
    _p("train_per_iter", fallback=4000)
    _p("val_size", fallback=400)
    _p("general_frac", fallback=0.70)
    _p("hard_frac", fallback=0.20)
    _p("easy_frac", fallback=0.10)
    _p("fast_eval_n", fallback=3000)
    _p("final_eval_n", "final_eval_n", fallback=None)

    # target_mu from preset unless default was already changed via CLI
    if args.target_mu == 0.006 and "target_mu" in preset:
        args.target_mu = preset["target_mu"]

    # --first-strike: imply local dataset, enable fast eval, local eval mode
    if args.first_strike:
        if args.dataset_mode == "auto":
            args.dataset_mode = "local"
        args.fast_eval = True
        # Only switch to local eval if user hasn't explicitly requested validator
        if args.eval_mode == "validator" and not any(
            "--eval-mode" in a for a in sys.argv
        ):
            args.eval_mode = "local"
            log.info("--first-strike: eval_mode set to local (pass --eval-mode validator to override)")

    # n_score=0 requires a local manifest (can't sample without local shards)
    if args.n_score == 0:
        local_m = args.local_dataset_manifest or os.environ.get("LOCAL_DATASET_MANIFEST", "")
        if not local_m or not Path(local_m).is_file():
            raise SystemExit(
                "ERROR: n_score=0 (direct sampling) requires a local dataset manifest.\n"
                "  Set --local-dataset-manifest <path/to/manifest.json>\n"
                "  or LOCAL_DATASET_MANIFEST env var.\n"
                "  Build shards with: python scripts/mining/retokenize_fineweb_edu_qwen.py ..."
            )

    log.info(
        "preset=%s → lr=%g epochs=%g lora_r=%d lora_alpha=%d lora_dropout=%g "
        "n_score=%d train_per_iter=%d val_size=%d "
        "fracs=%.0f/%.0f/%.0f target_mu=%g",
        args.candidate_preset, args.lr, args.epochs,
        args.lora_r, args.lora_alpha, args.lora_dropout,
        args.n_score, args.train_per_iter, args.val_size,
        args.general_frac * 100, args.hard_frac * 100, args.easy_frac * 100,
        args.target_mu,
    )

    # --- legacy prod profile ---
    if args.profile == "prod":
        args.n_score = max(args.n_score, 12000)
        args.train_per_iter = max(args.train_per_iter, 10000)
        args.val_size = max(args.val_size, 500)
        args.n_eval = max(args.n_eval, 25600)
        args.epochs = max(args.epochs, 3.0)
        args.max_iters = max(args.max_iters, 5)
        args.lora_r = max(args.lora_r, 32)
        args.lora_alpha = max(args.lora_alpha, 64)
        args.raw_max_files = max(args.raw_max_files, 32)
        if args.eval_mode != "local":
            args.eval_mode = "validator"
        log.info("legacy profile=prod applied")

    # --- propagate local manifest env ---
    if args.local_dataset_manifest:
        os.environ["LOCAL_DATASET_MANIFEST"] = args.local_dataset_manifest

    # --- sanity checks ---
    if args.n_score > 0:
        min_score = args.train_per_iter + args.val_size
        if args.n_score < min_score:
            log.warning(
                "n_score=%d < train_per_iter+val_size=%d — curriculum will be undersized",
                args.n_score, min_score,
            )
    else:
        log.info("n_score=0: skipping king scoring — using direct local shard sampling")

    if args.eval_mode == "validator":
        os.environ.setdefault("TEUTONIC_EVAL_DATASET_MODE", "raw_hippius")
        os.environ.setdefault("TEUTONIC_RAW_TOKENIZER_REPO", "Qwen/Qwen3-4B")

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    cache = work / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    # ---- 1. King ----
    king = fetch_king()
    king_dir = work / "king"
    local_king = os.environ.get("LOCAL_KING_DIR", "")
    if local_king:
        local_king_p = Path(local_king)
        if not local_king_p.exists():
            raise FileNotFoundError(f"LOCAL_KING_DIR not found: {local_king_p}")
        if king_dir.exists():
            shutil.rmtree(king_dir)
        log.info("copying local king %s -> %s", local_king_p, king_dir)
        shutil.copytree(local_king_p, king_dir, symlinks=False)
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

    # ---- 2. Dataset ----
    shards: list[np.ndarray] = []
    shard_keys: list[str] = []      # manifest key for each shard (same index as shards)
    eval_arr: np.ndarray | None = None
    eval_arr_per_shard: list[np.ndarray] = []
    eval_indices: list[int] | None = None
    manifest: dict = {}

    skip_shard_load = (
        (args.skip_scoring or args.from_iter >= 0) and args.eval_mode == "validator"
    )

    if skip_shard_load:
        log.info("skipping dataset shard load (skip_scoring/from_iter + validator eval)")
    else:
        manifest = fetch_manifest(cache, args.local_dataset_manifest)
        dataset_mode = resolve_dataset_mode(args.dataset_mode, manifest, king_dir)
        log.info("dataset mode: %s (manifest tokenizer=%r, king vocab=%s)",
                 dataset_mode, manifest.get("tokenizer"), vocab_size)

        if dataset_mode == "raw":
            n_needed = args.n_score + args.train_per_iter + args.val_size + 256
            if args.n_score == 0:
                n_needed = args.train_per_iter + args.val_size + 256
            if args.eval_mode == "local":
                n_needed += args.n_eval
            tokenizer_repo = os.environ.get(
                "TEUTONIC_RAW_TOKENIZER_REPO", chain_config.SEED_TOKENIZER_REPO,
            ) or str(king_dir)
            pool = load_raw_sequences(
                n_needed, args.seq_len, f"mining:{args.seed}", cache,
                tokenizer_repo, max_files=args.raw_max_files,
                model_vocab_size=vocab_size,
            )
            shards = [pool]
            if args.eval_mode == "local":
                eval_arr = pool
                rng_eval = np.random.default_rng(0xE1A)
                eval_indices = rng_eval.choice(
                    len(eval_arr), size=min(args.n_eval, len(eval_arr)), replace=False,
                ).tolist()
        else:
            # pretokenized / local
            # Use structured train_shards / eval_shards split if available (new-format manifests).
            # Fall back to legacy combined shards + eval_shard index for backward compat.
            has_structured_split = (
                "train_shards" in manifest and "eval_shards" in manifest
                and len(manifest["train_shards"]) > 0
            )

            if has_structured_split:
                train_entries = manifest["train_shards"]
                eval_entries = manifest.get("eval_shards") or []
                log.info(
                    "structured manifest split: %d train shards, %d eval shards",
                    len(train_entries), len(eval_entries),
                )

                # Select training entries
                if args.n_shards <= 0:
                    selected_train = train_entries[args.shard_start:]
                else:
                    end = args.shard_start + args.n_shards
                    selected_train = train_entries[args.shard_start:end]
                if not selected_train:
                    raise ValueError(
                        f"No train shards selected (shard_start={args.shard_start} "
                        f"n_shards={args.n_shards}, manifest has {len(train_entries)} "
                        f"train shards)"
                    )

                for e in selected_train:
                    key = e["key"]
                    path = download_shard(key, cache / Path(key).name)
                    arr, _ = load_shard(path, args.seq_len)
                    validate_sequences_vocab(arr, vocab_size, f"train shard {key}")
                    log.info("loaded train shard: %s (%d sequences)", key, len(arr))
                    shards.append(arr)
                    shard_keys.append(key)

                # Eval shards: load all available for multi-shard eval
                eval_arrays: list[np.ndarray] = []
                for e in eval_entries:
                    key = e["key"]
                    path = download_shard(key, cache / Path(key).name)
                    arr, _ = load_shard(path, args.seq_len)
                    validate_sequences_vocab(arr, vocab_size, f"eval shard {key}")
                    log.info("loaded eval shard: %s (%d sequences)", key, len(arr))
                    eval_arrays.append(arr)

                if eval_arrays:
                    # Primary eval array: concatenate all eval shards for global LCB
                    # Per-shard arrays kept separately for min_shard_lcb
                    eval_arr = eval_arrays[0] if len(eval_arrays) == 1 else np.concatenate(eval_arrays, axis=0)
                    eval_arr_per_shard = eval_arrays  # for multi-shard LCB
                else:
                    eval_arr = None
                    eval_arr_per_shard = []
                    log.warning("no eval shards in manifest — local eval unavailable")

            else:
                # Legacy: combined shards list, select eval by index
                eval_arr_per_shard = []
                all_entries = manifest.get("shards") or []
                n_manifest_shards = len(all_entries)
                if n_manifest_shards == 0:
                    raise ValueError("dataset manifest has no shards")

                def _check_idx(idx: int, label: str) -> None:
                    if idx < 0 or idx >= n_manifest_shards:
                        raise ValueError(
                            f"{label} shard index {idx} out of range (manifest has "
                            f"{n_manifest_shards} shard(s), valid 0..{n_manifest_shards - 1})"
                        )

                if args.eval_shard < 0:
                    args.eval_shard = n_manifest_shards - 1
                if args.n_shards <= 0:
                    train_shard_idxs = [i for i in range(n_manifest_shards)
                                        if i != args.eval_shard]
                else:
                    train_shard_idxs = list(range(args.shard_start,
                                                  args.shard_start + args.n_shards))

                for idx in train_shard_idxs:
                    _check_idx(idx, "training")
                _check_idx(args.eval_shard, "eval")
                if args.eval_shard in train_shard_idxs:
                    raise ValueError("eval_shard cannot overlap training shards")

                for idx in train_shard_idxs:
                    key = all_entries[idx]["key"]
                    path = download_shard(key, cache / Path(key).name)
                    arr, _ = load_shard(path, args.seq_len)
                    validate_sequences_vocab(arr, vocab_size, f"shard {idx}")
                    log.info("loaded shard %d: %d sequences", idx, len(arr))
                    shards.append(arr)
                    shard_keys.append(key)

                eval_key = all_entries[args.eval_shard]["key"]
                eval_path = download_shard(eval_key, cache / Path(eval_key).name)
                eval_arr, _ = load_shard(eval_path, args.seq_len)
                validate_sequences_vocab(eval_arr, vocab_size, f"eval shard {args.eval_shard}")

            # Build eval_indices covering fast+final eval
            if eval_arr is not None:
                rng_eval = np.random.default_rng(0xE1A)
                max_eval_n = max(
                    args.fast_eval_n if args.fast_eval else 0,
                    args.final_eval_n or args.n_eval,
                )
                eval_indices = rng_eval.choice(
                    len(eval_arr),
                    size=min(max_eval_n, len(eval_arr)),
                    replace=False,
                ).tolist()
                log.info("eval pool: %d sequences (sampling %d for eval)",
                         len(eval_arr), len(eval_indices))

    # ---- Score cache dir ----
    score_cache_dir: Path | None = None
    if args.use_local_score_cache:
        if args.score_cache_dir:
            score_cache_dir = Path(args.score_cache_dir) / f"king_{king_hash[:16]}"
        else:
            score_cache_dir = _score_cache_dir(work, king_hash, manifest)
        score_cache_dir.mkdir(parents=True, exist_ok=True)
        log.info("score cache dir: %s", score_cache_dir)

    # ---- Iteration loop ----
    best: dict | None = None
    history: list[dict] = []
    iter_list = [args.from_iter] if args.from_iter >= 0 else list(range(args.max_iters))

    for it in iter_list:
        log.info("=" * 60)
        log.info("=== ITERATION %d/%d ===", it + 1, args.max_iters)
        log.info("=" * 60)
        seed = args.seed + 1000 * it

        iter_work = work / f"iter_{it:02d}"
        iter_work.mkdir(exist_ok=True)
        train_p = iter_work / "train.jsonl"
        val_p = iter_work / "val.jsonl"
        out_dir = iter_work / "lora"
        merged_dir = iter_work / "merged"
        verdict_p = iter_work / "verdict.json"

        # Skip if already completed and --force not set
        if verdict_p.exists() and not args.force and args.from_iter < 0:
            try:
                v = json.loads(verdict_p.read_text())
                log.info("iter %d already has verdict.json (accepted=%s, mu_hat=%.6f) — skipping",
                         it, v.get("accepted"), v.get("mu_hat", 0))
                history.append(v)
                best = _rank_candidates(history + ([best] if best else []))
                continue
            except Exception:
                pass

        merge_only = args.from_iter >= 0

        # --- Score / curate ---
        if merge_only:
            log.info("from-iter %d: merge+eval only (skip score+train)", it)
            if not train_p.exists():
                raise FileNotFoundError(
                    f"--from-iter {it} but missing {train_p}; run scoring first"
                )
        elif args.skip_scoring and train_p.exists() and val_p.exists():
            log.info("skip-scoring: reusing existing train/val jsonl")
        else:
            if args.skip_scoring:
                raise FileNotFoundError(
                    f"--skip-scoring set but {train_p} or {val_p} missing"
                )
            if args.n_score == 0:
                # First-strike / direct sampling: no king forward pass, no loss buckets.
                # Clean random sampling with token-level quality filters only.
                if not shards:
                    raise RuntimeError(
                        "n_score=0 (direct sampling) requires local shards. "
                        "Use --dataset-mode local and --local-dataset-manifest."
                    )
                train_p, val_p = sample_direct_from_shards(
                    shards, args.train_per_iter, args.val_size, seed, iter_work,
                )
            else:
                train_p, val_p = score_and_curate(
                    str(king_dir), shards, shard_keys, manifest,
                    args.n_score, args.train_per_iter, args.val_size,
                    seed, "cuda:0", iter_work,
                    general_frac=args.general_frac,
                    hard_frac=args.hard_frac,
                    easy_frac=args.easy_frac,
                    max_suspicious_frac=args.max_suspicious_frac,
                    score_cache_path=score_cache_dir,
                    force_rescore=args.force_rescore,
                    king_hash=king_hash,
                )

        # --- Train ---
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
                    raise FileNotFoundError(f"no checkpoint under {out_dir}")
                adapter = cks[-1]
        else:
            resume = (args.resume_checkpoint or "").strip() if it == iter_list[0] else ""
            if resume:
                log.info("resuming LoRA from %s", resume)
            adapter = run_lora_training(
                str(king_dir), train_p, val_p, out_dir,
                args.n_gpus, args, Path(args.bundle),
            )

        # --- Merge ---
        if not merged_dir.exists() or not (merged_dir / "config.json").is_file():
            merge_lora(str(king_dir), adapter, merged_dir)
        else:
            log.info("merged model already at %s", merged_dir)

        # --- Fast local eval (optional, uses holdout shard) ---
        fast_verdict: dict | None = None
        if args.fast_eval and eval_arr is not None and eval_indices is not None:
            fast_n = min(args.fast_eval_n, len(eval_indices))
            log.info("fast local eval: %d sequences", fast_n)
            fast_verdict = paired_eval(
                str(king_dir), str(merged_dir), eval_arr,
                eval_indices[:fast_n], "cuda:0",
                batch_size=args.eval_batch_size,
                n_bootstrap=args.bootstrap,
                alpha=args.alpha,
                acceptance_lcb_floor=args.acceptance_lcb_floor,
                mean_delta_floor=args.mean_delta_floor,
            )
            fast_verdict["phase"] = "fast_eval"
            json.dump(fast_verdict, open(iter_work / "eval_fast.json", "w"), indent=2)
            log.info(
                "fast eval: mu_hat=%.6f lcb=%.6f accepted=%s",
                fast_verdict.get("mu_hat", 0),
                fast_verdict.get("lcb", 0),
                fast_verdict.get("accepted"),
            )

        # --- Final eval ---
        if args.eval_mode == "validator":
            _mining_dir = os.path.dirname(os.path.abspath(__file__))
            if _mining_dir not in sys.path:
                sys.path.insert(0, _mining_dir)
            from validator_eval import DEFAULT_BLOCK_HASH, validator_style_paired_eval
            block_hash = (args.sim_block_hash or DEFAULT_BLOCK_HASH).strip()
            hotkey = (args.sim_hotkey or "").strip()
            if not hotkey:
                raise ValueError("validator eval requires --sim-hotkey or TEUTONIC_SIM_HOTKEY")
            eval_gpu_ids = (
                [int(x) for x in args.eval_gpus.split(",") if x.strip()]
                if args.eval_gpus else [0]
            )
            verdict = validator_style_paired_eval(
                str(king_dir), str(merged_dir),
                block_hash=block_hash, hotkey=hotkey,
                n_public=args.n_eval, n_private=args.n_eval_private,
                seq_len=args.seq_len, gpu_ids=eval_gpu_ids,
                batch_size=args.eval_batch_size,
                eval_alpha=args.alpha,
                delta_threshold=EVAL_DELTA,
                n_bootstrap=args.bootstrap,
                vocab_size=vocab_size,
            )
            # Apply mining-side acceptance floors on top of validator result
            rejection_reasons = list(verdict.get("rejection_reasons") or [])
            lcb = verdict.get("lcb", 0)
            mu_hat = verdict.get("mu_hat", 0)
            if lcb <= args.acceptance_lcb_floor:
                rejection_reasons.append(
                    f"lcb={lcb:.6f} <= acceptance_lcb_floor={args.acceptance_lcb_floor}"
                )
            if args.mean_delta_floor > 0 and mu_hat < args.mean_delta_floor:
                rejection_reasons.append(
                    f"mu_hat={mu_hat:.6f} < mean_delta_floor={args.mean_delta_floor}"
                )
            if rejection_reasons:
                verdict["accepted"] = False
            verdict["rejection_reasons"] = rejection_reasons
            verdict["acceptance_lcb_floor"] = args.acceptance_lcb_floor
            verdict["mean_delta_floor"] = args.mean_delta_floor
        else:
            if eval_arr is None or eval_indices is None:
                raise RuntimeError("local eval requires eval_arr")
            n_final = min(args.final_eval_n or args.n_eval, len(eval_indices))
            log.info("final local eval: %d sequences", n_final)
            verdict = paired_eval(
                str(king_dir), str(merged_dir), eval_arr,
                eval_indices[:n_final], "cuda:0",
                batch_size=args.eval_batch_size,
                n_bootstrap=args.bootstrap,
                alpha=args.alpha,
                acceptance_lcb_floor=args.acceptance_lcb_floor,
                mean_delta_floor=args.mean_delta_floor,
            )

        verdict["iter"] = it
        verdict["seed"] = seed
        verdict["merged_dir"] = str(merged_dir)
        verdict["mean_delta"] = verdict.get("mu_hat", 0)
        if fast_verdict:
            verdict["fast_eval"] = fast_verdict

        json.dump(verdict, open(verdict_p, "w"), indent=2)
        history.append(verdict)

        # Update best using LCB-based ranking
        candidates = history + ([best] if best else [])
        best = _rank_candidates(candidates)
        if best and best.get("iter") == it:
            log.info("iter %d is new best (lcb=%.6f mu_hat=%.6f accepted=%s)",
                     it, best.get("lcb", 0), best.get("mu_hat", 0), best.get("accepted"))
            best["iter_dir"] = str(iter_work)

        if verdict.get("mu_hat", 0) >= args.target_mu and verdict.get("accepted"):
            log.info("target_mu=%.4f reached at iter %d — stopping", args.target_mu, it)
            break

    # ---- Write best/ dir ----
    best_dir = work / "best"
    best_dir.mkdir(exist_ok=True)
    if best:
        json.dump(best, open(best_dir / "verdict.json", "w"), indent=2)
        best_merged = best.get("merged_dir", "")
        if best_merged:
            (best_dir / "best_model_path.txt").write_text(best_merged)

    # ---- race_summary ----
    race_summary = {
        "king_repo": king["hf_repo"],
        "king_revision": king.get("king_revision"),
        "king_hash": king_hash,
        "preset": args.candidate_preset,
        "lr": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "general_frac": args.general_frac,
        "hard_frac": args.hard_frac,
        "easy_frac": args.easy_frac,
        "n_score": args.n_score,
        "train_per_iter": args.train_per_iter,
        "val_size": args.val_size,
        "n_iters_run": len(history),
        "best": best,
        "history": history,
        "ts": time.time(),
    }
    json.dump(race_summary, open(work / "race_summary.json", "w"), indent=2)

    # Human-readable summary
    if best:
        accepted_str = "ACCEPTED" if best.get("accepted") else "REJECTED"
        summary_lines = [
            f"=== RACE SUMMARY ===",
            f"king: {king['hf_repo']}",
            f"preset: {args.candidate_preset}  lr={args.lr}  lora_r={args.lora_r}",
            f"iters run: {len(history)}",
            f"best iter: {best.get('iter', '?')}  {accepted_str}",
            f"  mu_hat={best.get('mu_hat', 0):.6f}  "
            f"lcb={best.get('lcb', 0):.6f}  "
            f"king_loss={best.get('avg_king_loss', 0):.4f}  "
            f"chall_loss={best.get('avg_chall_loss', best.get('avg_challenger_loss', 0)):.4f}",
        ]
        if best.get("rejection_reasons"):
            summary_lines.append(f"  rejection: {best['rejection_reasons']}")
        if best.get("merged_dir"):
            summary_lines.append(f"  model: {best['merged_dir']}")
        summary_text = "\n".join(summary_lines) + "\n"
        print(summary_text, flush=True)
        (work / "race_summary.md").write_text(summary_text)

    # ---- Final report ----
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

    # ---- HuggingFace Upload (optional) ----
    if args.upload_repo and best and best.get("accepted"):
        best_merged = best.get("merged_dir", best.get("iter_dir", ""))
        if best_merged and Path(best_merged).is_dir():
            log.info("uploading %s -> HF %s", best_merged, args.upload_repo)
            api = HfApi(token=args.hf_token)
            api.create_repo(args.upload_repo, exist_ok=True, private=False)
            api.upload_folder(
                folder_path=best_merged,
                repo_id=args.upload_repo,
                commit_message=f"Teutonic challenger (mu_hat={best.get('mu_hat', 0):.6f})",
                allow_patterns=["*.safetensors", "config.json", "tokenizer*",
                                "special_tokens*", "generation_config.json"],
            )
            info = api.repo_info(args.upload_repo)
            final["hf_uploaded_repo"] = args.upload_repo
            final["hf_uploaded_revision"] = info.sha  # HF git SHA, NOT an OCI/Hippius digest
            final["challenger_hash"] = sha256_dir(Path(best_merged))
            # submit_challenger.py requires an OCI manifest digest (uploaded_digest),
            # not a HF git SHA. Hippius/OCI push must be done separately.
            final["uploaded_digest"] = None
            final["hippius_upload_required"] = True
            log.info("uploaded to HF -> %s @ %s", args.upload_repo, info.sha[:12])
            log.warning(
                "IMPORTANT: HF upload done, but submit_challenger.py needs an OCI/Hippius "
                "manifest digest (uploaded_digest), not a HF git SHA.\n"
                "  1. Push the model to Hippius: hippius push %s (or equivalent)\n"
                "  2. Set 'uploaded_digest' in %s to the OCI sha256:... digest\n"
                "  3. Then run: python scripts/mining/submit_challenger.py --verdict %s",
                best_merged, args.report_out or "verdict.json",
                args.report_out or "verdict.json",
            )
            if args.report_out:
                json.dump(final, open(args.report_out, "w"), indent=2)
    elif args.upload_repo:
        log.warning("not uploading: best=%s", best)

    log.info(
        "DONE — best mu_hat=%.6f accepted=%s",
        best.get("mu_hat", float("nan")) if best else float("nan"),
        best.get("accepted", False) if best else False,
    )


if __name__ == "__main__":
    main()
