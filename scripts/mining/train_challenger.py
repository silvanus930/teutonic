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
       Fractions: --general-frac --hard-frac --easy-frac (default 60/30/10).
  6. Train LoRA adapter(s) with optional --lora-sweep (adapter eval before merge).
       Modes: --mode {fast,strong} for preset sweeps.
  7. Merge only the best LoRA candidate after sweep.
  8. Offline paired eval candidate vs king on held-out shard slice
     (adapter-first during sweep; merged model for final eval).
       Pre-submit gate: submit_decision in {DO_NOT_SUBMIT, PROMISING_NEEDS_MORE_EVAL,
       READY_TO_MERGE, READY_TO_UPLOAD}. Upload requires --upload-approved.
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
from huggingface_hub import HfApi, snapshot_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

_mining_dir = os.path.dirname(os.path.abspath(__file__))
if _mining_dir not in sys.path:
    sys.path.insert(0, _mining_dir)
from dataset_mix import (  # noqa: E402
    MixConfig,
    MixedDatasetIndex,
    ShardStore,
    load_mix_config,
    mix_config_hash,
    refs_to_candidates,
    synthetic_manifest,
)
from hf_king_compat import (  # noqa: E402
    default_lora_target_modules_for_king,
    ensure_quasar_arch_registered,
    hf_remote_code_kwargs as _hf_remote_code_kwargs,
    prepare_quasar_model as _prepare_quasar_model,
    king_subprocess_env,
    patch_transformers_quasar_compat,
)
from cache_utils import (  # noqa: E402
    download_shard_cached,
    ensure_king_cached,
    king_digest_dir,
    load_score_cache,
    save_score_cache,
    score_cache_dir,
)
from curriculum import (  # noqa: E402
    DEFAULT_EASY_FRAC,
    DEFAULT_GENERAL_FRAC,
    DEFAULT_HARD_FRAC,
    assign_buckets,
    build_curriculum,
    build_mixture_curriculum,
    bucket_means,
    loss_summary as _loss_summary,
    save_curriculum_reports,
    write_curriculum_jsonl,
)
from dataset_mixture import (  # noqa: E402
    MixtureConfig,
    MixtureShardStore,
    build_mixture_allocations,
    build_mixture_config,
    parse_bucket_mix_arg,
    parse_dataset_manifest_arg,
    parse_dataset_weight_arg,
    prepare_mixture_eval_pools,
    save_allocation_summary,
)
from paired_eval import (  # noqa: E402
    compute_per_seq_loss,
    gpu_memory_stats,
    mixture_weighted_paired_eval,
    mixture_weighted_paired_eval_adapter,
    paired_eval_adapter,
    paired_eval_merged,
)
from preflight import (  # noqa: E402
    check_chain_match,
    pre_submit_decision,
    print_submit_verdict,
    validate_merged_model,
    validate_upload_repo,
)
from reporting import save_run_report  # noqa: E402

patch_transformers_quasar_compat()
ensure_quasar_arch_registered()

_repo_root = os.path.dirname(os.path.dirname(_mining_dir))
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
HIPPIUS_BASE = os.environ.get(
    "TEUTONIC_HIPPIUS_HTTP_BASE", "https://s3.hippius.com/teutonic-sn3",
).rstrip("/")
# ~4–5 safetensor files for 8.6B Quasar (~17 GB); Hippius-friendly upload size.
DEFAULT_MERGE_MAX_SHARD_SIZE = "3500MB"

# Fast / probe / strong mode presets (CLI overrides still apply)
MODE_PRESETS: dict[str, dict] = {
    "fast": {
        "n_score": 4000,
        "train_per_iter": 3000,
        "val_size": 400,
        "n_eval": 768,
        "epochs": 0.5,
        "lora_dropout": 0.05,
        "fast_eval": True,
        "fast_eval_n": 768,
        "final_eval_n": 768,
        "lora_sweep": ["r32:a64:lr2e-4:d0.05:e0.5"],
        "max_iters": 1,
    },
    "probe": {
        "dataset_preset": "teutonic-mixture-v2",
        "eval_mode": "local",
        "n_score": 10000,
        "train_per_iter": 6000,
        "val_size": 800,
        "n_eval": 1000,
        "epochs": 0.5,
        "lora_dropout": 0.05,
        "fast_eval": True,
        "fast_eval_n": 1000,
        "final_eval_n": 1000,
        "lora_sweep": [
            "r32:a64:lr2e-4:d0.05:e0.5",
            "r64:a128:lr1e-4:d0.05:e0.5",
        ],
        "abort_if_mu_hat_nonpositive": True,
        "max_iters": 1,
    },
    "strong": {
        "dataset_preset": "teutonic-mixture-v2",
        "eval_mode": "local",
        "n_score": 50000,
        "train_per_iter": 24000,
        "val_size": 3000,
        "n_eval": 5000,
        "epochs": 0.8,
        "lora_dropout": 0.05,
        "fast_eval": True,
        "fast_eval_n": 3000,
        "final_eval_n": 5000,
        "dual_eval": True,
        "lora_sweep": [
            "r32:a64:lr2e-4:d0.05:e0.8",
            "r64:a128:lr1e-4:d0.05:e0.8",
            "r64:a64:lr1e-4:d0.05:e0.8",
            "r128:a256:lr8e-5:d0.05:e0.7",
        ],
        "max_iters": 1,
    },
}

GPU_PROFILE_PRESETS: dict[str, dict] = {
    "a100-80gb": {
        "micro_batch": 1,
        "grad_accum": 16,
        "gradient_checkpointing": True,
    },
    "a100-40gb": {
        "micro_batch": 1,
        "grad_accum": 32,
        "gradient_checkpointing": True,
    },
}

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
        "general_frac": 0.60, "hard_frac": 0.30, "easy_frac": 0.10,
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
        "general_frac": 0.60, "hard_frac": 0.30, "easy_frac": 0.10,
        "fast_eval_n": 3000, "final_eval_n": 5000,
    },
    "custom": {},  # all values from explicit CLI args
}


@dataclasses.dataclass
class LoRASweepConfig:
    label: str
    lora_r: int
    lora_alpha: int
    lr: float
    lora_dropout: float | None = None
    epochs: float | None = None


def parse_lora_sweep(specs: list[str]) -> list[LoRASweepConfig]:
    """Parse sweep specs; supports comma lists and optional :d dropout / :e epochs.

    Examples:
      r64:a128:lr1e-4
      r64:a128:lr1e-4:d0.05:e0.8
      r32:a64:lr2e-4,r64:a128:lr1e-4
    """
    import re
    pattern = re.compile(
        r"^r(\d+):a(\d+):lr([\de.\-+]+)(?::d([\de.\-+]+))?(?::e([\de.\-+]+))?$",
        re.IGNORECASE,
    )
    out: list[LoRASweepConfig] = []
    for raw in specs:
        for spec in raw.split(","):
            spec = spec.strip()
            if not spec:
                continue
            m = pattern.match(spec)
            if not m:
                raise ValueError(
                    f"invalid LoRA sweep spec {spec!r}; "
                    "expected r64:a128:lr1e-4 or r64:a128:lr1e-4:d0.05:e0.8"
                )
            label = spec.replace(":", "_").replace(",", "").lower()
            dropout = float(m.group(4)) if m.group(4) is not None else None
            epochs = float(m.group(5)) if m.group(5) is not None else None
            out.append(LoRASweepConfig(
                label=label,
                lora_r=int(m.group(1)),
                lora_alpha=int(m.group(2)),
                lr=float(m.group(3)),
                lora_dropout=dropout,
                epochs=epochs,
            ))
    return out


def _sweep_rank_key(entry: dict) -> tuple:
    """Rank sweep candidates by mixture LCB, then mu_hat, then lower train eval loss."""
    return (
        float(entry.get("lcb") if entry.get("lcb") is not None else -999),
        float(entry.get("mu_hat") if entry.get("mu_hat") is not None else -999),
        -float(entry.get("eval_loss") if entry.get("eval_loss") is not None else 999),
    )


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
    return download_shard_cached(shard_key, out.parent, HIPPIUS_BASE)


def _resolve_local_manifest_paths(m: dict, base: Path) -> None:
    """Turn relative shard keys into absolute paths under the manifest directory."""
    for field in ("shards", "train_shards", "eval_shards"):
        for s in m.get(field) or []:
            key = (s.get("key") or "").strip()
            if not key:
                continue
            kp = Path(key)
            if not kp.is_absolute():
                s["key"] = str((base / key).resolve())


def fetch_manifest(cache: Path, local_manifest_path: str = "") -> dict:
    local = local_manifest_path or os.environ.get("LOCAL_DATASET_MANIFEST", "")
    if local:
        p = Path(local)
        if not p.exists():
            raise FileNotFoundError(f"local dataset manifest not found: {p}")
        log.info("loading local dataset manifest: %s", p)
        m = json.loads(p.read_text())
        _resolve_local_manifest_paths(m, p.parent)
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
    if "quasar" in shard_tok:
        return False
    cfg = json.loads((king_dir / "config.json").read_text())
    arch = " ".join(cfg.get("architectures") or []).lower()
    if "qwen" not in arch and "quasar" not in arch:
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
            tokenizer_repo,
            token=os.environ.get("HF_TOKEN") or None,
            use_fast=True,
            **_hf_remote_code_kwargs(tokenizer_repo),
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
def paired_eval(
    king_dir: str, chall_dir: str, shard: np.ndarray,
    indices: list[int], device: str, batch_size: int = 8,
    n_bootstrap: int = 10000, alpha: float = EVAL_ALPHA,
    acceptance_lcb_floor: float = EVAL_DELTA,
    mean_delta_floor: float = 0.0,
) -> dict:
    """Local paired bootstrap test mirroring the validator (merged challenger)."""
    result = paired_eval_merged(
        king_dir, chall_dir, shard, indices, device,
        batch_size=batch_size,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        acceptance_lcb_floor=acceptance_lcb_floor,
        mean_delta_floor=mean_delta_floor,
        hf_remote_code_kwargs=_hf_remote_code_kwargs,
        prepare_quasar_model=_prepare_quasar_model,
    )
    result["note"] = "local mode: holdout from mining data pool (not validator seeds)"
    return result


# Legacy wrappers — delegate to cache_utils
def _score_cache_dir(work: Path, king_hash: str, manifest: dict) -> Path:
    shard_keys = [s.get("key", "") for s in manifest.get("shards", []) if s.get("key")]
    return score_cache_dir(work, king_hash, shard_keys or ["legacy"], SEQ_LEN)


def _load_score_cache(cache_dir: Path, shard_keys: list[str]) -> list[dict] | None:
    return load_score_cache(cache_dir, shard_keys)


def _save_score_cache(cache_dir: Path, shard_keys: list[str], rows: list[dict]) -> None:
    save_score_cache(cache_dir, shard_keys, rows, include_input_ids=True)


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
    general_frac: float = DEFAULT_GENERAL_FRAC,
    hard_frac: float = DEFAULT_HARD_FRAC,
    easy_frac: float = DEFAULT_EASY_FRAC,
    max_suspicious_frac: float = 0.0,
    score_cache_path: Path | None = None,
    force_rescore: bool = False,
    king_hash: str = "",
    preset_cands: list[tuple[int, int]] | None = None,
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
    if preset_cands is not None:
        cands = list(preset_cands)
        rng.shuffle(cands)
    else:
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
            use_safetensors=True, **_hf_remote_code_kwargs(king_dir),
        )
        n_quasar = _prepare_quasar_model(model)
        if n_quasar:
            log.info("quasar compat: using PyTorch conv fallback on %d layer(s)", n_quasar)
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
                    "input_ids": tok,
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
            _save_score_cache(score_cache_path, sampled_keys, rows)

    # Backfill input_ids from shards when loading legacy cache entries
    for r in rows:
        if r.get("input_ids") is None and shards is not None:
            r["input_ids"] = shards[r["shard"]][r["idx"]].tolist()

    counts, p50, p85 = assign_buckets(rows)
    losses = np.asarray([r["loss"] for r in rows])
    loss_stats = _loss_summary(losses)
    bucket_loss = bucket_means(rows)
    log.info(
        "buckets: %s | mean loss: %s",
        counts,
        {k: f"{v:.4f}" for k, v in bucket_loss.items()},
    )

    train_rows, val_rows, curriculum_summary = build_curriculum(
        rows,
        train_per_iter=train_per_iter,
        val_size=val_size,
        seed=seed,
        general_frac=general_frac,
        hard_frac=hard_frac,
        easy_frac=easy_frac,
        max_suspicious_frac=max_suspicious_frac,
    )
    dropped = curriculum_summary.get("suspicious_dropped", 0)
    train_mix = curriculum_summary.get("train_mix", {})
    log.info(
        "curriculum train: %d sequences | mix %s",
        len(train_rows), train_mix,
    )
    if len(train_rows) < train_per_iter:
        log.warning(
            "train set undersized (%d < %d); increase --n-score or data pool",
            len(train_rows), train_per_iter,
        )

    print(
        f"\n{'='*50}\n"
        f"CURRICULUM SUMMARY\n"
        f"  scored: {len(rows)} | suspicious dropped: {dropped}\n"
        f"  train: {len(train_rows)} | val: {len(val_rows)}\n"
        f"  mix: general={train_mix.get('general',0)} "
        f"hard={train_mix.get('hard',0)} easy={train_mix.get('easy',0)}\n"
        f"  loss p50={p50:.4f} p85={p85:.4f}\n"
        f"{'='*50}\n",
        flush=True,
    )

    work.mkdir(parents=True, exist_ok=True)
    train_p, val_p = write_curriculum_jsonl(train_rows, val_rows, work, shards)
    save_curriculum_reports(
        work, rows, curriculum_summary,
        seed=seed,
        n_candidates=len(cands),
        shards_used=sampled_keys,
        shards_used_local_indices=sampled_local_idxs,
    )
    log.info("wrote train=%d val=%d | %s %s", len(train_rows), len(val_rows), train_p, val_p)
    return train_p, val_p


def _score_dataset_candidates(
    king_dir: str,
    shards: list[np.ndarray],
    shard_keys: list[str],
    cands: list[tuple[int, int]],
    device: str,
    score_cache_path: Path | None,
    force_rescore: bool,
) -> list[dict]:
    sampled_keys = [shard_keys[i] for i in sorted({s for s, _ in cands})]
    cached_rows: list[dict] | None = None
    if score_cache_path and not force_rescore and score_cache_path.is_dir():
        cached_rows = _load_score_cache(score_cache_path, sampled_keys)

    if cached_rows is not None:
        rows = cached_rows
        for r in rows:
            if "row_idx" in r and "idx" not in r:
                r["idx"] = r["row_idx"]
    else:
        model = AutoModelForCausalLM.from_pretrained(
            king_dir, torch_dtype=torch.bfloat16, device_map={"": device},
            use_safetensors=True, **_hf_remote_code_kwargs(king_dir),
        )
        _prepare_quasar_model(model)
        model.eval()
        rows = []
        batch = 8
        for i in range(0, len(cands), batch):
            chunk = cands[i:i + batch]
            toks = [shards[s][j].tolist() for s, j in chunk]
            losses = compute_per_seq_loss(model, toks, device)
            for (s_idx, j), tok, loss in zip(chunk, toks, losses):
                arr = np.asarray(tok)
                unique_r = float(len(set(tok)) / max(1, len(tok)))
                rep_r = float(np.mean(arr[1:] == arr[:-1])) if len(arr) > 1 else 0.0
                ngrams = [tuple(tok[k:k + 4]) for k in range(len(tok) - 3)]
                rep_ng = 1.0 - len(set(ngrams)) / max(1, len(ngrams)) if ngrams else 0.0
                rows.append({
                    "shard": s_idx, "idx": j, "loss": float(loss),
                    "unique_r": unique_r, "rep_r": rep_r, "rep_ng4": rep_ng,
                    "input_ids": tok,
                })
        del model
        torch.cuda.empty_cache()
        if score_cache_path:
            _save_score_cache(score_cache_path, sampled_keys, rows)

    for r in rows:
        if r.get("input_ids") is None:
            r["input_ids"] = shards[r["shard"]][r["idx"]].tolist()
    return rows


def score_and_curate_mixture(
    king_dir: str,
    store: MixtureShardStore,
    cfg: MixtureConfig,
    allocations: dict,
    seed: int,
    device: str,
    work: Path,
    king_hash: str,
    *,
    shards_per_dataset: int = 12,
    force_rescore: bool = False,
    max_suspicious_frac: float = 0.0,
    use_score_cache: bool = True,
) -> tuple[Path, Path, dict]:
    """Score and build curriculum across weighted mixture datasets."""
    rows_by_dataset: dict[str, list[dict]] = {}
    shards_used: list[str] = []

    for source in cfg.sources:
        n_score = allocations["n_score"].get(source.name, 0)
        if n_score <= 0:
            continue
        ds_seed = seed + hash(source.name) % 100000
        refs = store.sample_refs(source.name, n_score, ds_seed, shards_per_dataset)
        cands = store.refs_to_candidates(refs)
        shards_used.extend(sorted({r.shard_key for r in refs}))

        cache_dir = None
        if use_score_cache:
            work_root = work
            while work_root.name.startswith("iter_") and work_root.parent != work_root:
                work_root = work_root.parent
            cache_dir = store.score_cache_dir(work_root, king_hash, source.name)
            cache_dir.mkdir(parents=True, exist_ok=True)

        log.info("mixture scoring %s: n=%d candidates", source.name, len(cands))
        rows = _score_dataset_candidates(
            king_dir, store.arrays, store.keys, cands, device,
            cache_dir, force_rescore,
        )
        for r in rows:
            r["dataset"] = source.name
        assign_buckets(rows)
        rows_by_dataset[source.name] = rows

    train_rows, val_rows, curriculum_summary = build_mixture_curriculum(
        rows_by_dataset,
        train_alloc=allocations["train_per_iter"],
        val_alloc=allocations["val_size"],
        seed=seed,
        bucket_mix=cfg.bucket_mix,
        max_suspicious_frac=max_suspicious_frac,
    )

    all_rows = [r for rows in rows_by_dataset.values() for r in rows]
    work.mkdir(parents=True, exist_ok=True)
    train_p, val_p = write_curriculum_jsonl(train_rows, val_rows, work, store.arrays)
    save_curriculum_reports(
        work, all_rows, curriculum_summary,
        seed=seed,
        n_candidates=sum(len(v) for v in rows_by_dataset.values()),
        shards_used=shards_used,
        per_dataset_stats=curriculum_summary.get("per_dataset"),
    )
    save_allocation_summary(work, allocations)
    log.info(
        "mixture curriculum: train=%d val=%d across %d datasets",
        len(train_rows), len(val_rows), len(rows_by_dataset),
    )
    return train_p, val_p, curriculum_summary


# ---------------------------------------------------------------------------
# LoRA training
# ---------------------------------------------------------------------------
def run_lora_training(
    base_model: str, train_p: Path, val_p: Path,
    out_dir: Path, n_gpus: int, args: argparse.Namespace,
    bundle: Path,
    resume_checkpoint: str = "",
    learning_rate: float | None = None,
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
        "--learning-rate", str(learning_rate if learning_rate is not None else args.lr),
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
    if args.lora_target_modules:
        cmd.extend(["--lora-target-modules", args.lora_target_modules])
    if args.gradient_checkpointing:
        cmd.append("--gradient-checkpointing")
    else:
        cmd.append("--no-gradient-checkpointing")
    resume = (resume_checkpoint or args.resume_checkpoint or "").strip()
    if resume:
        cmd.extend(["--resume-from-checkpoint", resume])
    log.info("training: %s", " ".join(cmd))
    t0 = time.time()
    subprocess.check_call(cmd, env=king_subprocess_env(base_model))
    log.info("training done in %.1fs", time.time() - t0)

    adapter = out_dir / "best_adapter"
    if not adapter.exists():
        if (out_dir / "adapter_model.safetensors").exists() or \
           (out_dir / "adapter_model.bin").exists():
            adapter = out_dir
        else:
            raise RuntimeError(f"no adapter found in {out_dir}")
    return adapter


def run_lora_sweep(
    base_model: str,
    train_p: Path,
    val_p: Path,
    sweep_out: Path,
    sweep_configs: list[LoRASweepConfig],
    n_gpus: int,
    args: argparse.Namespace,
    bundle: Path,
    *,
    eval_arr: np.ndarray | None,
    eval_indices: list[int] | None,
    eval_batch_size: int,
    bootstrap: int,
    alpha: float,
    acceptance_lcb_floor: float,
    mean_delta_floor: float,
    fast_eval_n: int,
    mixture_eval_pools: dict[str, tuple[np.ndarray, list[int]]] | None = None,
    mixture_weights: dict[str, float] | None = None,
    regression_datasets: tuple[str, ...] = (),
    regression_mu_floor: float = -0.001,
    regression_lcb_floor: float = -0.001,
) -> tuple[LoRASweepConfig, Path, list[dict]]:
    """Train and adapter-eval each LoRA config; return best config + adapter path."""
    results: list[dict] = []
    best_cfg: LoRASweepConfig | None = None
    best_adapter: Path | None = None
    best_score: tuple = (-999.0, -999.0, -999.0)

    for cfg in sweep_configs:
        log.info(
            "=== LoRA sweep: %s (r=%d a=%d lr=%g d=%s e=%s) ===",
            cfg.label, cfg.lora_r, cfg.lora_alpha, cfg.lr,
            cfg.lora_dropout, cfg.epochs,
        )
        cfg_out = sweep_out / cfg.label
        cfg_args = argparse.Namespace(**vars(args))
        cfg_args.lora_r = cfg.lora_r
        cfg_args.lora_alpha = cfg.lora_alpha
        cfg_args.lr = cfg.lr
        if cfg.lora_dropout is not None:
            cfg_args.lora_dropout = cfg.lora_dropout
        if cfg.epochs is not None:
            cfg_args.epochs = cfg.epochs

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t0 = time.time()
        adapter = run_lora_training(
            base_model, train_p, val_p, cfg_out,
            n_gpus, cfg_args, bundle,
        )
        elapsed = time.time() - t0

        train_summary: dict = {}
        summary_path = cfg_out / "train_summary.json"
        if summary_path.is_file():
            train_summary = json.loads(summary_path.read_text())

        paired: dict | None = None
        if mixture_eval_pools and mixture_weights:
            paired = mixture_weighted_paired_eval_adapter(
                base_model, str(adapter), mixture_eval_pools, mixture_weights, "cuda:0",
                batch_size=eval_batch_size,
                n_bootstrap=bootstrap,
                alpha=alpha,
                acceptance_lcb_floor=acceptance_lcb_floor,
                mean_delta_floor=mean_delta_floor,
                hf_remote_code_kwargs=_hf_remote_code_kwargs,
                prepare_quasar_model=_prepare_quasar_model,
                regression_datasets=regression_datasets,
                regression_mu_floor=regression_mu_floor,
                regression_lcb_floor=regression_lcb_floor,
            )
            paired_path = cfg_out / "paired_eval_mixture_adapter.json"
            paired_path.write_text(json.dumps(paired, indent=2, default=str))
        elif eval_arr is not None and eval_indices is not None:
            n_eval = min(fast_eval_n, len(eval_indices))
            paired = paired_eval_adapter(
                base_model, str(adapter), eval_arr, eval_indices[:n_eval], "cuda:0",
                batch_size=eval_batch_size,
                n_bootstrap=bootstrap,
                alpha=alpha,
                acceptance_lcb_floor=acceptance_lcb_floor,
                mean_delta_floor=mean_delta_floor,
                hf_remote_code_kwargs=_hf_remote_code_kwargs,
                prepare_quasar_model=_prepare_quasar_model,
            )
            paired_path = cfg_out / "paired_eval_adapter.json"
            paired_path.write_text(json.dumps(paired, indent=2))

        entry = {
            "config": (
                f"r{cfg.lora_r}:a{cfg.lora_alpha}:lr{cfg.lr:g}"
                + (f":d{cfg.lora_dropout:g}" if cfg.lora_dropout is not None else "")
                + (f":e{cfg.epochs:g}" if cfg.epochs is not None else "")
            ),
            "label": cfg.label,
            "lora_r": cfg.lora_r,
            "lora_alpha": cfg.lora_alpha,
            "lr": cfg.lr,
            "lora_dropout": cfg.lora_dropout if cfg.lora_dropout is not None else args.lora_dropout,
            "epochs": cfg.epochs if cfg.epochs is not None else args.epochs,
            "train_loss": train_summary.get("final_train_loss"),
            "eval_loss": train_summary.get("final_eval_loss"),
            "best_checkpoint_eval_loss": train_summary.get("best_checkpoint_eval_loss"),
            "mu_hat": paired.get("mixture_mu_hat", paired.get("mu_hat")) if paired else None,
            "lcb": paired.get("mixture_lcb", paired.get("lcb")) if paired else None,
            "accepted": paired.get("accepted") if paired else False,
            "n_eval": paired.get("n_eval") if paired else 0,
            "elapsed_s": round(elapsed, 1),
            "adapter_dir": str(adapter),
            "gpu_memory": gpu_memory_stats(),
        }
        results.append(entry)
        log.info(
            "sweep %s: eval_loss=%s mu_hat=%s lcb=%s accepted=%s (%.1fs)",
            cfg.label, entry["eval_loss"], entry["mu_hat"], entry["lcb"],
            entry["accepted"], elapsed,
        )

        score = _sweep_rank_key(entry)
        if score > best_score:
            best_score = score
            best_cfg = cfg
            best_adapter = adapter

    if best_cfg is None or best_adapter is None:
        raise RuntimeError("LoRA sweep produced no candidates")

    sweep_report = sweep_out / "sweep_results.json"
    sweep_report.write_text(json.dumps({"results": results, "best": best_cfg.label}, indent=2))
    log.info("LoRA sweep best: %s -> %s", best_cfg.label, best_adapter)
    return best_cfg, best_adapter, results


def merge_lora(
    base_model: str, adapter: Path | str, out: Path | str,
    max_shard_size: str = DEFAULT_MERGE_MAX_SHARD_SIZE,
) -> Path:
    adapter = Path(adapter)
    out = Path(out)
    log.info("merging LoRA %s into %s -> %s", adapter, base_model, out)
    from peft import PeftModel
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, use_safetensors=True,
        **_hf_remote_code_kwargs(base_model),
    )
    _prepare_quasar_model(base)
    merged = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    save_kw: dict = {"safe_serialization": True}
    shard_cap = (max_shard_size or "").strip()
    if shard_cap:
        save_kw["max_shard_size"] = shard_cap
        log.info("merge save: sharding weights max_shard_size=%s", shard_cap)
    else:
        log.info("merge save: single model.safetensors (no max_shard_size)")
    merged.save_pretrained(str(out), **save_kw)
    weight_shards = sorted(out.glob("model-*-of-*.safetensors"))
    if weight_shards:
        for p in weight_shards:
            log.info("merge shard: %s (%.2f GB)", p.name, p.stat().st_size / 1e9)
        log.info("merge saved %d weight shard(s)", len(weight_shards))
    elif (out / "model.safetensors").is_file():
        p = out / "model.safetensors"
        log.info("merge shard: %s (%.2f GB)", p.name, p.stat().st_size / 1e9)
    tok = AutoTokenizer.from_pretrained(
        base_model, use_fast=True, **_hf_remote_code_kwargs(base_model),
    )
    tok.save_pretrained(str(out))
    base_path = Path(base_model).expanduser().resolve()
    for name in ("config.json", "generation_config.json"):
        if base_path.is_dir():
            src = base_path / name
        else:
            src = Path(snapshot_download(base_model, allow_patterns=[name])) / name
        if src.is_file():
            shutil.copy(src, out / name)
    for py_name in ("configuration_qwen3_5.py", "modeling_qwen3_5.py"):
        src_py = base_path / py_name
        if src_py.is_file():
            shutil.copy(src_py, out / py_name)
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
    ap.add_argument(
        "--dataset-mix", default="",
        help="JSON with multiple v4 Hippius manifests + weights "
             "(see scripts/mining/dataset_mix_quasar_v4.json). "
             "Enables mixture mode (alias for --dataset-preset teutonic-mixture-v2).",
    )
    ap.add_argument(
        "--dataset-preset",
        default="",
        help="Dataset preset: teutonic-mixture-v2 (validator mixture) or legacy (single manifest).",
    )
    ap.add_argument(
        "--dataset-manifest", action="append", default=[],
        help="Override/add dataset manifest: name=https://.../manifest.json",
    )
    ap.add_argument(
        "--dataset-weight", action="append", default=[],
        help="Override dataset weight: name=0.35",
    )
    ap.add_argument(
        "--dataset-names", default="",
        help="Comma-separated subset of dataset names to include in mixture.",
    )
    ap.add_argument(
        "--bucket-mix", action="append", default=[],
        help="Per-dataset curriculum bucket override: name=general:hard:easy "
             "(e.g. finewebedu=0.7:0.2:0.1)",
    )
    ap.add_argument(
        "--regression-datasets", default="automathtext-v2,ultradata-math,finewebedu",
        help="Comma-separated datasets checked for negative mu_hat/lcb regression.",
    )
    ap.add_argument("--regression-mu-floor", type=float, default=-0.001)
    ap.add_argument("--regression-lcb-floor", type=float, default=-0.001)
    ap.add_argument(
        "--no-block-on-regression", action="store_true",
        help="Warn on per-dataset regression but do not block submission.",
    )
    ap.add_argument(
        "--mix-shard-cache", default="",
        help="Cache dir for mixed-mode shard downloads (default: <work>/mix_cache)",
    )
    ap.add_argument(
        "--mix-shards-per-dataset", type=int, default=12,
        help="Mixed mode: shard pool per dataset (only these .npy files download).",
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

    # Fast / probe / strong modes and LoRA sweep
    ap.add_argument(
        "--mode", choices=("fast", "probe", "strong"), default="",
        help="Preset mode: fast, probe (cheap beatability check), or strong (A100 run).",
    )
    ap.add_argument(
        "--lora-sweep", action="append", default=[],
        help="LoRA sweep spec (repeatable or comma-separated). "
             "Example: r64:a128:lr1e-4:d0.05:e0.8",
    )
    ap.add_argument(
        "--merge-best-only", action="store_true", default=True,
        help="During sweep, merge only the best adapter (default: on).",
    )
    ap.add_argument(
        "--no-merge-best-only", action="store_false", dest="merge_best_only",
        help="Merge after each sweep config (legacy behavior).",
    )
    ap.add_argument(
        "--skip-chain-check", action="store_true",
        help="Skip chain.toml vs dashboard chain name verification.",
    )
    ap.add_argument(
        "--force-king-redownload", action="store_true",
        help="Re-download king even if digest cache exists.",
    )

    ap.add_argument(
        "--abort-if-mu-hat-nonpositive", action="store_true",
        help="After LoRA sweep, exit before merge if best mixture mu_hat <= 0.",
    )
    ap.add_argument(
        "--dual-eval", action="store_true",
        help="Run mixture local eval as primary verdict, then validator-style eval.",
    )

    # Pre-submit gate
    ap.add_argument("--preferred-lcb-margin", type=float, default=0.0035)
    ap.add_argument("--preferred-mu-hat", type=float, default=0.0075)
    ap.add_argument("--min-final-n-eval", type=int, default=3000)
    ap.add_argument(
        "--upload-approved", action="store_true",
        help="Allow HF upload when pre-submit gate passes (never uploads by default).",
    )
    ap.add_argument(
        "--submit-approved", action="store_true",
        help="Alias for --upload-approved (no on-chain reveal is performed here).",
    )
    ap.add_argument("--coldkey-prefix", default="", help="First 8 ss58 chars for upload repo check.")

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
    ap.add_argument("--lora-target-modules", default=None,
                    help="Comma-separated LoRA module suffixes. "
                         "Auto-detected for Quasar kings when omitted.")
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
    ap.add_argument("--resume-checkpoint", default="",
                    help="LoRA adapter dir to resume training from (overrides auto-chain).")
    ap.add_argument("--no-chain-lora", action="store_true",
                    help="Do not resume iter N>0 from iter_{N-1}/lora/best_adapter.")
    ap.add_argument("--chain-lr-ratio", type=float, default=0.25,
                    help="When chaining from adapter-only checkpoint (no trainer_state), "
                         "multiply base --lr by this ratio for follow-up iters.")
    ap.add_argument("--from-iter", type=int, default=-1)
    ap.add_argument("--skip-scoring", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="Re-run even if verdict.json already exists for that iter")

    # Profile (legacy + GPU presets)
    ap.add_argument(
        "--profile",
        choices=("default", "prod", "a100-80gb", "a100-40gb"),
        default="default",
        help="GPU / training profile. a100-80gb: micro_batch=1 grad_accum=16; "
             "a100-40gb: micro_batch=1 grad_accum=32.",
    )

    # Upload / report
    ap.add_argument("--upload-repo", default="")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    ap.add_argument("--report-out", default="")
    ap.add_argument(
        "--merge-max-shard-size",
        default=os.environ.get("TEUTONIC_MERGE_MAX_SHARD_SIZE", DEFAULT_MERGE_MAX_SHARD_SIZE),
        help="Shard merged safetensors on save (default: %(default)s → ~4–5 files for "
             "8.6B Quasar, Hippius upload). Set empty or TEUTONIC_MERGE_MAX_SHARD_SIZE= "
             "for a single model.safetensors.",
    )

    args = ap.parse_args()

    if args.submit_approved:
        args.upload_approved = True

    # --- apply --mode preset (explicit CLI args override via None defaults) ---
    mode_preset = MODE_PRESETS.get(args.mode, {}) if args.mode else {}
    for key, value in mode_preset.items():
        if key == "lora_sweep":
            if not args.lora_sweep:
                args.lora_sweep = list(value)
            continue
        if key == "fast_eval" and not any(a.startswith("--fast-eval") for a in sys.argv):
            args.fast_eval = bool(value)
            continue
        if key == "dual_eval" and not any(a.startswith("--dual-eval") for a in sys.argv):
            args.dual_eval = bool(value)
            continue
        if key == "abort_if_mu_hat_nonpositive" and not any(
            a.startswith("--abort-if-mu-hat") for a in sys.argv
        ):
            args.abort_if_mu_hat_nonpositive = bool(value)
            continue
        if key == "dataset_preset" and not args.dataset_preset:
            args.dataset_preset = str(value)
            continue
        if key == "eval_mode" and not any(a.startswith("--eval-mode") for a in sys.argv):
            args.eval_mode = str(value)
            continue
        if getattr(args, key, None) is None and hasattr(args, key):
            setattr(args, key, value)
        elif key in ("lora_dropout", "epochs", "micro_batch", "grad_accum") and hasattr(args, key):
            if not any(f"--{key.replace('_', '-')}" in a for a in sys.argv):
                setattr(args, key, value)

    # --- apply GPU profile ---
    if args.profile in GPU_PROFILE_PRESETS:
        for key, value in GPU_PROFILE_PRESETS[args.profile].items():
            if hasattr(args, key):
                setattr(args, key, value)
        log.info("GPU profile=%s applied (micro_batch=%d grad_accum=%d)",
                 args.profile, args.micro_batch, args.grad_accum)

    # --- apply candidate preset (explicit CLI args with non-None default override preset) ---
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
    _p("general_frac", fallback=DEFAULT_GENERAL_FRAC)
    _p("hard_frac", fallback=DEFAULT_HARD_FRAC)
    _p("easy_frac", fallback=DEFAULT_EASY_FRAC)
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

    local_manifest = (
        args.local_dataset_manifest or os.environ.get("LOCAL_DATASET_MANIFEST", "")
    ).strip()
    if args.dataset_mode in ("local", "pretokenized"):
        if not local_manifest or not Path(local_manifest).is_file():
            raise SystemExit(
                "ERROR: --dataset-mode local requires king-tokenized shards.\n"
                "  Pass an explicit path (do not rely on an unset env var):\n"
                "    --local-dataset-manifest /root/teutonic/s1-work/dataset/manifest.json\n"
                "  Or export LOCAL_DATASET_MANIFEST to that path before running."
            )

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

    if not args.skip_chain_check:
        check_chain_match(chain_config.NAME, DASHBOARD_URL, strict=True)

    regression_datasets = tuple(
        x.strip() for x in args.regression_datasets.split(",") if x.strip()
    )

    # --- mixture dataset config ---
    manifest_overrides = [parse_dataset_manifest_arg(m) for m in args.dataset_manifest]
    weight_overrides = dict(parse_dataset_weight_arg(w) for w in args.dataset_weight)
    bucket_mix_overrides = dict(parse_bucket_mix_arg(b) for b in args.bucket_mix)
    dataset_names = [x.strip() for x in args.dataset_names.split(",") if x.strip()] or None
    mix_json = (
        args.dataset_mix or os.environ.get("TEUTONIC_DATASET_MIX", "")
    ).strip()
    use_mixture = (
        args.dataset_preset == "teutonic-mixture-v2"
        or bool(mix_json)
        or bool(manifest_overrides)
        or bool(weight_overrides)
    )
    if args.dataset_preset == "legacy":
        use_mixture = False

    mixture_cfg: MixtureConfig | None = None
    mixture_store: MixtureShardStore | None = None
    mixture_allocations: dict | None = None
    mixture_eval_pools: dict[str, tuple[np.ndarray, list[int]]] | None = None

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    cache = work / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    if use_mixture:
        mixture_cfg = build_mixture_config(
            preset=args.dataset_preset or "teutonic-mixture-v2",
            cache_root=cache,
            seq_len=args.seq_len,
            dataset_manifests=manifest_overrides or None,
            dataset_weights=weight_overrides or None,
            dataset_names=dataset_names,
            mix_json_path=mix_json,
            bucket_mix_overrides=bucket_mix_overrides or None,
        )
        mixture_store = MixtureShardStore(mixture_cfg)
        log.info(
            "dataset preset=%s (%d sources, weights=%s)",
            mixture_cfg.preset,
            len(mixture_cfg.sources),
            ", ".join(f"{n}={w:.2f}" for n, w in mixture_cfg.weights.items()),
        )

    sweep_configs: list[LoRASweepConfig] = []
    if args.lora_sweep:
        sweep_configs = parse_lora_sweep(args.lora_sweep)
        log.info("LoRA sweep: %d config(s): %s", len(sweep_configs), args.lora_sweep)

    # ---- 1. King ----
    work_king_dir = work / "king"
    local_king = os.environ.get("LOCAL_KING_DIR", "")
    if local_king:
        local_king_p = Path(local_king)
        if not local_king_p.exists():
            raise FileNotFoundError(f"LOCAL_KING_DIR not found: {local_king_p}")
        king = {
            "hf_repo": str(local_king_p.resolve()),
            "king_revision": "local",
            "hotkey": "local",
            "reign_number": 0,
        }
        log.info("LOCAL_KING_DIR set — skipping dashboard fetch")
        # Use the pointed king directory in-place (avoid copying / overwriting).
        king_dir = local_king_p.resolve()
        if king_dir == work_king_dir.resolve():
            raise ValueError(
                f"LOCAL_KING_DIR must not be {work_king_dir} (use a real folder like "
                f"{work / 'king1'} to avoid symlink loops with hf download)"
            )
        log.info("using LOCAL_KING_DIR in-place: %s", king_dir)
        if not (king_dir / "config.json").is_file():
            hint = work / "king1"
            raise FileNotFoundError(
                f"LOCAL_KING_DIR={king_dir} has no config.json (not a model directory).\n"
                f"  Point LOCAL_KING_DIR at the folder with weights, e.g.:\n"
                f"    export LOCAL_KING_DIR={hint}"
            )
    else:
        king = fetch_king()
        rev = king.get("king_revision") or None
        if isinstance(rev, str) and rev.startswith("hf:"):
            rev = rev[len("hf:"):]
        cache_key = hashlib.sha256(
            f"{king['hf_repo']}|{rev or 'HEAD'}".encode(),
        ).hexdigest()[:16]
        meta_path = work / "king_cache" / f"meta_{cache_key}.json"
        king_dir = None
        if meta_path.is_file() and not args.force_king_redownload:
            try:
                meta = json.loads(meta_path.read_text())
                candidate = Path(meta.get("path", ""))
                if candidate.is_dir() and (candidate / "config.json").is_file():
                    king_dir = candidate
                    log.info("king digest cache hit: %s", king_dir)
            except Exception:
                pass
        if king_dir is None:
            staging = work / "king_cache" / f"staging_{cache_key}"
            ensure_king_cached(
                king["hf_repo"], rev, staging,
                hf_token=args.hf_token,
                force=args.force_king_redownload,
            )
            king_hash_staging = sha256_dir(staging)
            king_dir = king_digest_dir(work, king_hash_staging)
            if king_dir.exists() and not args.force_king_redownload:
                shutil.rmtree(staging)
                log.info("king digest dir already present: %s", king_dir)
            elif staging.resolve() != king_dir.resolve():
                if king_dir.exists():
                    shutil.rmtree(king_dir)
                shutil.move(str(staging), str(king_dir))
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path.write_text(json.dumps({
                "path": str(king_dir),
                "repo": king["hf_repo"],
                "revision": king.get("king_revision"),
                "digest": sha256_dir(king_dir),
            }, indent=2))
            log.info("king cached by digest: %s", king_dir)

    king_hash = sha256_dir(king_dir)
    log.info("king sha256[:16]=%s", king_hash[:16])
    if args.eval_mode == "validator":
        os.environ.setdefault("TEUTONIC_RAW_TOKENIZER_REPO", str(king_dir))
        log.info(
            "validator eval tokenizer: %s",
            os.environ["TEUTONIC_RAW_TOKENIZER_REPO"],
        )
    if not args.lora_target_modules:
        auto_lora = default_lora_target_modules_for_king(str(king_dir))
        if auto_lora:
            args.lora_target_modules = auto_lora
            log.info("auto lora targets (Quasar): %s", auto_lora)
    vocab_size = king_vocab_size(king_dir)

    # ---- 2. Dataset ----
    shards: list[np.ndarray] = []
    shard_keys: list[str] = []      # manifest key for each shard (same index as shards)
    eval_arr: np.ndarray | None = None
    eval_arr_per_shard: list[np.ndarray] = []
    eval_indices: list[int] | None = None
    manifest: dict = {}
    mix_cfg: MixConfig | None = None
    mix_index: MixedDatasetIndex | None = None
    mix_cache_dir: Path | None = None

    skip_shard_load = (
        (args.skip_scoring or args.from_iter >= 0) and args.eval_mode == "validator"
        and not use_mixture
    )

    if skip_shard_load:
        log.info("skipping dataset shard load (skip_scoring/from_iter + validator eval)")
    elif use_mixture and mixture_cfg and mixture_store:
        manifest = {
            "version": "mixture-v2",
            "preset": mixture_cfg.preset,
            "seq_len": mixture_cfg.seq_len,
            "datasets": [
                {"name": s.name, "manifest_url": s.manifest_url, "weight": mixture_cfg.weights[s.name]}
                for s in mixture_cfg.sources
            ],
        }
        mixture_allocations = build_mixture_allocations(
            mixture_cfg,
            n_score=args.n_score,
            train_per_iter=args.train_per_iter,
            val_size=args.val_size,
            n_eval=args.final_eval_n or args.n_eval,
        )
        save_allocation_summary(work, mixture_allocations)
        if args.eval_mode == "local":
            mixture_eval_pools = prepare_mixture_eval_pools(
                mixture_store,
                mixture_cfg,
                mixture_allocations["n_eval"],
                args.seed,
                args.mix_shards_per_dataset,
            )
            log.info(
                "mixture eval pools: %s",
                {k: len(v[1]) for k, v in mixture_eval_pools.items()},
            )
    else:
        manifest = fetch_manifest(cache, args.local_dataset_manifest)
        dataset_mode = resolve_dataset_mode(args.dataset_mode, manifest, king_dir)
        log.info("dataset mode: %s (manifest tokenizer=%r, king vocab=%s)",
                 dataset_mode, manifest.get("tokenizer"), vocab_size)

        if dataset_mode == "raw" and not mix_index:
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
            # Raw mode yields a single in-memory pool rather than manifest-addressed shards.
            # Still provide a stable synthetic shard key so score caching and curriculum
            # bookkeeping can work uniformly.
            # IMPORTANT: raw pools are sampled and can vary with seed, max_files,
            # and the number of sequences requested. Include these in the cache key
            # to avoid reusing cached (shard, idx) pairs that don't exist in a new pool.
            raw_key_material = json.dumps(
                {
                    "dataset_mode": "raw",
                    "manifest_tokenizer": manifest.get("tokenizer"),
                    "seq_len": args.seq_len,
                    "tokenizer_repo": tokenizer_repo,
                    "vocab_size": int(vocab_size),
                    "seed": int(args.seed),
                    "raw_max_files": int(args.raw_max_files),
                    "n_needed": int(n_needed),
                },
                sort_keys=True,
            ).encode("utf-8")
            shard_keys = [f"raw_pool_{hashlib.sha256(raw_key_material).hexdigest()[:16]}"]
            if args.eval_mode == "local":
                eval_arr = pool
                rng_eval = np.random.default_rng(0xE1A)
                eval_indices = rng_eval.choice(
                    len(eval_arr), size=min(args.n_eval, len(eval_arr)), replace=False,
                ).tolist()
        elif not mix_index:
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
    score_cache_dir_path: Path | None = None
    if args.use_local_score_cache:
        if args.score_cache_dir:
            score_cache_dir_path = Path(args.score_cache_dir)
        elif mix_cfg is not None:
            score_cache_dir_path = score_cache_dir(
                work, king_hash, [], args.seq_len,
                extra_tag=f"mix_{mix_config_hash(mix_cfg)[:8]}",
            )
        elif shard_keys:
            score_cache_dir_path = score_cache_dir(
                work, king_hash, shard_keys, args.seq_len,
            )
        else:
            score_cache_dir_path = _score_cache_dir(work, king_hash, manifest)
        score_cache_dir_path.mkdir(parents=True, exist_ok=True)
        log.info("score cache dir: %s", score_cache_dir_path)

    sweep_results_all: list[dict] = []
    curriculum_stats: dict = {}

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
                if use_mixture:
                    raise RuntimeError("n_score=0 is not supported with mixture datasets")
                if mix_index is not None:
                    raise RuntimeError(
                        "n_score=0 is not supported with --dataset-mix; use n_score > 0"
                    )
                if not shards:
                    raise RuntimeError(
                        "n_score=0 (direct sampling) requires local shards. "
                        "Use --dataset-mode local and --local-dataset-manifest."
                    )
                train_p, val_p = sample_direct_from_shards(
                    shards, args.train_per_iter, args.val_size, seed, iter_work,
                )
            elif use_mixture and mixture_store and mixture_cfg and mixture_allocations:
                train_p, val_p, curriculum_stats = score_and_curate_mixture(
                    str(king_dir), mixture_store, mixture_cfg, mixture_allocations,
                    seed, "cuda:0", iter_work, king_hash,
                    shards_per_dataset=args.mix_shards_per_dataset,
                    force_rescore=args.force_rescore,
                    max_suspicious_frac=args.max_suspicious_frac,
                    use_score_cache=args.use_local_score_cache,
                )
            else:
                preset_cands: list[tuple[int, int]] | None = None
                iter_shards = shards
                iter_keys = shard_keys
                if mix_index is not None and mix_cfg is not None and mix_cache_dir:
                    refs, _ = mix_index.sample_refs(
                        args.n_score, seed, args.mix_shards_per_dataset,
                    )
                    store = ShardStore(
                        mix_cache_dir, mix_cfg.hippius_base, mix_cfg.seq_len,
                    )
                    preset_cands = refs_to_candidates(refs, store)
                    iter_shards = store.arrays
                    iter_keys = store.keys
                    for key, arr in zip(iter_keys, iter_shards):
                        validate_sequences_vocab(arr, vocab_size, key)
                    log.info(
                        "mixed scoring: %d candidates, %d shard file(s) "
                        "(pool=%d/dataset)",
                        len(preset_cands), len(iter_keys),
                        args.mix_shards_per_dataset,
                    )
                train_p, val_p = score_and_curate(
                    str(king_dir), iter_shards, iter_keys, manifest,
                    args.n_score, args.train_per_iter, args.val_size,
                    seed, "cuda:0", iter_work,
                    general_frac=args.general_frac,
                    hard_frac=args.hard_frac,
                    easy_frac=args.easy_frac,
                    max_suspicious_frac=args.max_suspicious_frac,
                    score_cache_path=score_cache_dir_path,
                    force_rescore=args.force_rescore,
                    king_hash=king_hash,
                    preset_cands=preset_cands,
                )
            curriculum_path = iter_work / "curriculum.json"
            if curriculum_path.is_file():
                curriculum_stats = json.loads(curriculum_path.read_text())

        # --- Train ---
        sweep_out = iter_work / "lora_sweep"
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
            resume = (args.resume_checkpoint or "").strip()
            chain_lr: float | None = None
            if not resume and not args.no_chain_lora and it > 0:
                prev_adapter = work / f"iter_{it-1:02d}" / "lora" / "best_adapter"
                if prev_adapter.is_dir():
                    resume = str(prev_adapter)
                    log.info("chaining LoRA from iter_%02d: %s", it - 1, resume)
            elif resume:
                log.info("resuming LoRA from %s", resume)
            if resume and it > 0:
                resume_p = Path(resume)
                adapter_only = (
                    (resume_p / "adapter_model.safetensors").is_file()
                    or (resume_p / "adapter_model.bin").is_file()
                ) and not (resume_p / "trainer_state.json").is_file()
                if adapter_only:
                    chain_lr = args.lr * args.chain_lr_ratio
                    log.info(
                        "chain training: adapter-only resume, lr=%g (base %g × ratio %g)",
                        chain_lr, args.lr, args.chain_lr_ratio,
                    )
            if sweep_configs and not merge_only:
                best_cfg, adapter, sweep_results = run_lora_sweep(
                    str(king_dir), train_p, val_p, sweep_out, sweep_configs,
                    args.n_gpus, args, Path(args.bundle),
                    eval_arr=eval_arr,
                    eval_indices=eval_indices,
                    eval_batch_size=args.eval_batch_size,
                    bootstrap=args.bootstrap,
                    alpha=args.alpha,
                    acceptance_lcb_floor=args.acceptance_lcb_floor,
                    mean_delta_floor=args.mean_delta_floor,
                    fast_eval_n=args.fast_eval_n or 512,
                    mixture_eval_pools=mixture_eval_pools,
                    mixture_weights=mixture_cfg.weights if mixture_cfg else None,
                    regression_datasets=regression_datasets,
                    regression_mu_floor=args.regression_mu_floor,
                    regression_lcb_floor=args.regression_lcb_floor,
                )
                sweep_results_all = sweep_results
                best_mu = max(
                    float(r.get("mu_hat") or -999) for r in sweep_results
                )
                best_lcb = max(
                    float(r.get("lcb") or -999) for r in sweep_results
                )
                log.info(
                    "sweep summary: best mu_hat=%.6f best lcb=%.6f across %d config(s)",
                    best_mu, best_lcb, len(sweep_results),
                )
                if args.abort_if_mu_hat_nonpositive and best_mu <= 0:
                    msg = (
                        f"PROBE ABORT: best mixture mu_hat={best_mu:.6f} <= 0. "
                        "Do not merge or scale up — change LoRA/data/curriculum first."
                    )
                    log.error(msg)
                    abort_report = {
                        "abort_reason": msg,
                        "sweep_results": sweep_results,
                        "best_mu_hat": best_mu,
                        "best_lcb": best_lcb,
                    }
                    (iter_work / "probe_abort.json").write_text(
                        json.dumps(abort_report, indent=2),
                    )
                    raise SystemExit(2)
                best_link = out_dir / "best_adapter"
                if best_link.exists():
                    if best_link.is_symlink():
                        best_link.unlink()
                    else:
                        shutil.rmtree(best_link)
                shutil.copytree(str(adapter), str(best_link))
                log.info("sweep best adapter copied to %s", best_link)
            else:
                adapter = run_lora_training(
                    str(king_dir), train_p, val_p, out_dir,
                    args.n_gpus, args, Path(args.bundle),
                    resume_checkpoint=resume,
                    learning_rate=chain_lr,
                )

        # --- Merge (best only during sweep) ---
        should_merge = (
            not sweep_configs
            or args.merge_best_only
            or merge_only
        )
        if should_merge and (not merged_dir.exists() or not (merged_dir / "config.json").is_file()):
            merge_lora(
                str(king_dir), adapter, merged_dir,
                max_shard_size=(args.merge_max_shard_size or "").strip(),
            )
            hygiene_ok, hygiene_issues = validate_merged_model(merged_dir)
            if not hygiene_ok:
                log.warning("merged model hygiene issues: %s", hygiene_issues)
        elif should_merge:
            log.info("merged model already at %s", merged_dir)
        elif sweep_configs:
            log.info("sweep mode: skipping merge until best candidate selected")

        # --- Fast local eval (optional) ---
        fast_verdict: dict | None = None
        if args.fast_eval and mixture_eval_pools and mixture_cfg:
            log.info("fast mixture eval across %d datasets", len(mixture_eval_pools))
            fast_verdict = mixture_weighted_paired_eval(
                str(king_dir), str(merged_dir), mixture_eval_pools,
                mixture_cfg.weights, "cuda:0",
                batch_size=args.eval_batch_size,
                n_bootstrap=args.bootstrap,
                alpha=args.alpha,
                acceptance_lcb_floor=args.acceptance_lcb_floor,
                mean_delta_floor=args.mean_delta_floor,
                hf_remote_code_kwargs=_hf_remote_code_kwargs,
                prepare_quasar_model=_prepare_quasar_model,
                regression_datasets=regression_datasets,
                regression_mu_floor=args.regression_mu_floor,
                regression_lcb_floor=args.regression_lcb_floor,
            )
            fast_verdict["phase"] = "fast_eval"
            json.dump(fast_verdict, open(iter_work / "eval_fast.json", "w"), indent=2)
            log.info(
                "fast mixture eval: mu_hat=%.6f lcb=%.6f accepted=%s warnings=%d",
                fast_verdict.get("mixture_mu_hat", 0),
                fast_verdict.get("mixture_lcb", 0),
                fast_verdict.get("accepted"),
                len(fast_verdict.get("regression_warnings") or []),
            )
        elif args.fast_eval and eval_arr is not None and eval_indices is not None:
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
        verdict: dict | None = None
        validator_verdict: dict | None = None

        if args.dual_eval and mixture_eval_pools and mixture_cfg:
            log.info("dual eval: mixture local (primary decision)")
            verdict = mixture_weighted_paired_eval(
                str(king_dir), str(merged_dir), mixture_eval_pools,
                mixture_cfg.weights, "cuda:0",
                batch_size=args.eval_batch_size,
                n_bootstrap=args.bootstrap,
                alpha=args.alpha,
                acceptance_lcb_floor=args.acceptance_lcb_floor,
                mean_delta_floor=args.mean_delta_floor,
                hf_remote_code_kwargs=_hf_remote_code_kwargs,
                prepare_quasar_model=_prepare_quasar_model,
                regression_datasets=regression_datasets,
                regression_mu_floor=args.regression_mu_floor,
                regression_lcb_floor=args.regression_lcb_floor,
            )
            verdict["phase"] = "mixture_final"
            json.dump(verdict, open(iter_work / "eval_mixture_final.json", "w"), indent=2)

        if args.eval_mode == "validator" or args.dual_eval:
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
            log.info("running validator-style paired eval (n_public=%d)", args.n_eval)
            validator_verdict = validator_style_paired_eval(
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
            rejection_reasons = list(validator_verdict.get("rejection_reasons") or [])
            lcb = validator_verdict.get("lcb", 0)
            mu_hat = validator_verdict.get("mu_hat", 0)
            if lcb <= args.acceptance_lcb_floor:
                rejection_reasons.append(
                    f"lcb={lcb:.6f} <= acceptance_lcb_floor={args.acceptance_lcb_floor}"
                )
            if args.mean_delta_floor > 0 and mu_hat < args.mean_delta_floor:
                rejection_reasons.append(
                    f"mu_hat={mu_hat:.6f} < mean_delta_floor={args.mean_delta_floor}"
                )
            if rejection_reasons:
                validator_verdict["accepted"] = False
            validator_verdict["rejection_reasons"] = rejection_reasons
            validator_verdict["acceptance_lcb_floor"] = args.acceptance_lcb_floor
            validator_verdict["mean_delta_floor"] = args.mean_delta_floor
            validator_verdict["phase"] = "validator"
            json.dump(validator_verdict, open(iter_work / "eval_validator.json", "w"), indent=2)
            if verdict is None:
                verdict = validator_verdict
            else:
                verdict["validator_eval"] = validator_verdict
        elif verdict is None:
            if mixture_eval_pools and mixture_cfg:
                log.info("final mixture eval across %d datasets", len(mixture_eval_pools))
                verdict = mixture_weighted_paired_eval(
                    str(king_dir), str(merged_dir), mixture_eval_pools,
                    mixture_cfg.weights, "cuda:0",
                    batch_size=args.eval_batch_size,
                    n_bootstrap=args.bootstrap,
                    alpha=args.alpha,
                    acceptance_lcb_floor=args.acceptance_lcb_floor,
                    mean_delta_floor=args.mean_delta_floor,
                    hf_remote_code_kwargs=_hf_remote_code_kwargs,
                    prepare_quasar_model=_prepare_quasar_model,
                    regression_datasets=regression_datasets,
                    regression_mu_floor=args.regression_mu_floor,
                    regression_lcb_floor=args.regression_lcb_floor,
                )
            elif eval_arr is None or eval_indices is None:
                raise RuntimeError("local eval requires eval_arr or mixture_eval_pools")
            else:
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

        assert verdict is not None

        verdict["iter"] = it
        verdict["seed"] = seed
        verdict["merged_dir"] = str(merged_dir)
        verdict["mean_delta"] = verdict.get("mu_hat", 0)
        if fast_verdict:
            verdict["fast_eval"] = fast_verdict
        if sweep_results_all:
            verdict["sweep_results"] = sweep_results_all

        hygiene_ok, hygiene_issues = (
            validate_merged_model(merged_dir)
            if merged_dir.exists() else (False, ["merged dir missing"])
        )
        submit_decision, decision_reasons = pre_submit_decision(
            lcb=float(verdict.get("lcb", 0)),
            mu_hat=float(verdict.get("mu_hat", 0)),
            n_eval=int(verdict.get("n_eval", 0)),
            lcb_floor=args.acceptance_lcb_floor,
            preferred_lcb_margin=args.preferred_lcb_margin,
            preferred_mu_hat=args.preferred_mu_hat,
            min_n_eval=args.min_final_n_eval,
            merged_hygiene_ok=hygiene_ok if merged_dir.exists() else None,
            mixture_lcb=verdict.get("mixture_lcb"),
            mixture_mu_hat=verdict.get("mixture_mu_hat"),
            regression_warnings=verdict.get("regression_warnings"),
            block_on_regression=not args.no_block_on_regression,
        )
        verdict["submit_decision"] = submit_decision
        verdict["decision_reasons"] = decision_reasons
        verdict["merged_hygiene_ok"] = hygiene_ok
        verdict["merged_hygiene_issues"] = hygiene_issues
        print_submit_verdict(submit_decision, decision_reasons)

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
    run_report = {
        "chain_name": chain_config.NAME,
        "king_repo": king["hf_repo"],
        "king_revision": king.get("king_revision"),
        "king_digest": king_hash,
        "mode": args.mode or args.candidate_preset,
        "candidate_preset": args.candidate_preset,
        "dataset_preset": mixture_cfg.preset if mixture_cfg else ("legacy" if not use_mixture else ""),
        "dataset_weights": mixture_cfg.weights if mixture_cfg else {},
        "sample_allocations": mixture_allocations,
        "dataset_shards_used": shard_keys,
        "curriculum_stats": curriculum_stats,
        "curriculum_per_dataset": curriculum_stats.get("per_dataset") if curriculum_stats else {},
        "lora_configs_tested": sweep_results_all,
        "paired_eval": best,
        "per_dataset_eval": (best or {}).get("per_dataset") if best else {},
        "mixture_eval": {
            "mixture_mu_hat": (best or {}).get("mixture_mu_hat"),
            "mixture_lcb": (best or {}).get("mixture_lcb"),
            "regression_warnings": (best or {}).get("regression_warnings"),
        } if best else {},
        "validator_eval": (best or {}).get("validator_eval") if best else None,
        "best_config": (
            max(sweep_results_all, key=_sweep_rank_key)
            if sweep_results_all else None
        ),
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
        "final_decision": best.get("submit_decision") if best else "DO_NOT_SUBMIT",
        "decision_reasons": best.get("decision_reasons") if best else [],
        "merged_model_path": best.get("merged_dir") if best else None,
        "ts": time.time(),
    }
    save_run_report(work, run_report)

    race_summary = {
        "king_repo": king["hf_repo"],
        "king_revision": king.get("king_revision"),
        "king_hash": king_hash,
        "preset": args.candidate_preset,
        "mode": args.mode,
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
        "submit_decision": best.get("submit_decision") if best else None,
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

    # ---- HuggingFace Upload (optional, requires explicit approval) ----
    upload_allowed = (
        args.upload_approved
        and best
        and best.get("submit_decision") == "READY_TO_UPLOAD"
    )
    if args.upload_repo and upload_allowed:
        if args.coldkey_prefix:
            ok, msg = validate_upload_repo(args.upload_repo, args.coldkey_prefix)
            if not ok:
                raise ValueError(msg)
        best_merged = best.get("merged_dir", best.get("iter_dir", ""))
        if best_merged and Path(best_merged).is_dir():
            hygiene_ok, hygiene_issues = validate_merged_model(Path(best_merged))
            if not hygiene_ok:
                raise ValueError(f"upload blocked: merged model hygiene failed: {hygiene_issues}")
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
        log.warning(
            "not uploading: upload_approved=%s submit_decision=%s best=%s",
            args.upload_approved,
            best.get("submit_decision") if best else None,
            bool(best),
        )

    log.info(
        "DONE — best mu_hat=%.6f accepted=%s",
        best.get("mu_hat", float("nan")) if best else float("nan"),
        best.get("accepted", False) if best else False,
    )


if __name__ == "__main__":
    main()
