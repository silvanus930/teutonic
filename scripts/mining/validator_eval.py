"""Validator-aligned offline paired eval for the mining harness.

Uses the same code path as eval_server.py / validator dispatch:
  - derive_seeds(block_hash, hotkey) for public + bootstrap seeds
  - eval.raw_dataset.load_raw_sequences via sample_public_holdout
  - eval.torch_runner.run_paired_eval + compute_paired_losses

Usage:
    cd /root/teutonic
    export TEUTONIC_EVAL_DATASET_MODE=raw_hippius
    export TEUTONIC_RAW_TOKENIZER_REPO=Qwen/Qwen3-4B
    export TEUTONIC_SIM_HOTKEY=5FxJCGB1...

    python -u scripts/mining/validator_eval.py \\
        --king /root/teutonic/s1-work-prod/king \\
        --challenger /root/teutonic/s1-work-prod/iter_00/merged \\
        --report-out /root/teutonic/s1-work-prod/verdict.json
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Bootstrap repo root (scripts/mining/ -> teutonic/)
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

log = logging.getLogger("validator_eval")


def _ensure_logging() -> None:
    """Inline mining scripts often skip logging setup; make progress visible."""
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    # eval stack loggers
    for name in ("eval_torch", "eval_raw_dataset", "validator_eval"):
        logging.getLogger(name).setLevel(logging.INFO)


def _progress_printer(phase: str):
    """Return an on_progress callback for run_paired_eval."""
    def _on(info: dict) -> None:
        print(
            f"[validator eval] {phase}: {info.get('done', 0)}/{info.get('total', '?')} "
            f"mu_hat={info.get('mu_hat', 0):.6f} "
            f"king={info.get('avg_king_loss', 0):.4f} "
            f"chall={info.get('avg_challenger_loss', 0):.4f} "
            f"({info.get('seqs_per_sec', 0):.1f} seq/s)",
            flush=True,
        )
    return _on

HIPPIUS_BASE = os.environ.get(
    "TEUTONIC_HIPPIUS_HTTP_BASE", "https://s3.hippius.com/teutonic-sn3",
).rstrip("/")
RAW_SHARD_KEY = "raw:hippius:fineweb-edu"
DEFAULT_BLOCK_HASH = os.environ.get(
    "TEUTONIC_SIM_BLOCK_HASH",
    "0" * 64,
)
DEFAULT_HOTKEY = os.environ.get("TEUTONIC_SIM_HOTKEY", "")


def derive_eval_seeds(block_hash: str, hotkey: str) -> tuple[bytes, bytes, bytes]:
    """Same as eval_server._derive_seeds."""
    material = block_hash.encode() + hotkey.encode()
    public_seed = hashlib.blake2b(material + b"public", digest_size=8).digest()
    boot_seed = hashlib.blake2b(material + b"boot", digest_size=8).digest()
    private_seed = hashlib.blake2b(material + b"private", digest_size=8).digest()
    return public_seed, boot_seed, private_seed


class _HttpBody:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data


class _HippiusS3Client:
    """Minimal boto3 S3 client stand-in for public Hippius HTTP objects."""

    def __init__(self, base: str):
        self.base = base.rstrip("/")

    def _fetch(self, key: str, headers: dict[str, str] | None = None) -> bytes:
        url = f"{self.base}/{key.lstrip('/')}"
        req = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(req, timeout=300) as resp:
            return resp.read()

    def get_object(self, Bucket: str, Key: str, Range: str | None = None) -> dict:
        headers: dict[str, str] = {}
        if Range:
            headers["Range"] = Range
        return {"Body": _HttpBody(self._fetch(Key, headers))}

    def download_file(self, Bucket: str, Key: str, Filename: str, **kwargs) -> None:
        """Used by eval.raw_dataset._download_parquet via safe_download_file."""
        data = self._fetch(Key)
        with open(Filename, "wb") as f:
            f.write(data)


class HippiusHttpR2:
    """Dataset-store R2 stand-in when TEUTONIC_DS_* credentials are not set."""

    ds_bucket = "teutonic-sn3"

    def __init__(self, base: str = HIPPIUS_BASE):
        self.ds_client = _HippiusS3Client(base)

    def ds_get(self, key: str) -> dict | None:
        url = f"{HIPPIUS_BASE}/{key.lstrip('/')}"
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return json.loads(resp.read())
        except Exception:
            return None


def _resolve_local_npy_paths(dataset: Path) -> list[Path]:
    """Resolve pretokenized shard paths from a directory or manifest.json."""
    if dataset.is_file() and dataset.suffix.lower() == ".json":
        manifest = json.loads(dataset.read_text())
        base = dataset.parent
        paths: list[Path] = []
        for shard in manifest.get("shards") or []:
            key = (shard.get("key") or "").strip()
            if not key:
                continue
            p = Path(key)
            if not p.is_absolute():
                p = (base / key).resolve()
            if p.is_file():
                paths.append(p)
        if paths:
            return sorted(paths)
        dataset = base

    if not dataset.is_dir():
        raise FileNotFoundError(f"local dataset not found: {dataset}")

    for pattern in ("*.tokens.npy", "shard_*.npy", "*.npy"):
        paths = sorted(dataset.glob(pattern))
        if paths:
            return [p.resolve() for p in paths]
    raise RuntimeError(
        f"no .npy shards under {dataset} (expected *.tokens.npy, shard_*.npy, or manifest.json)"
    )


def sample_local_public_holdout(
    local_dataset: str | Path,
    public_seed: bytes,
    n_public: int,
    seq_len: int,
    *,
    mix_files: int = 0,
    vocab_size: int | None = None,
) -> tuple[torch.Tensor, str, dict]:
    """Sample holdout windows from local pretokenized .npy shards (1D or 2D uint32)."""
    from eval.raw_dataset import sample_global_from_token_mmaps

    dataset = Path(local_dataset).expanduser().resolve()
    all_paths = _resolve_local_npy_paths(dataset)
    npy_paths = all_paths
    seed_str = public_seed.hex()

    if mix_files and int(mix_files) > 0:
        want = min(int(mix_files), len(npy_paths))
        mix_seed = int.from_bytes(
            hashlib.blake2b((seed_str + "|mix").encode(), digest_size=8).digest(),
            "little",
        )
        rng = np.random.Generator(np.random.PCG64(mix_seed))
        picks = rng.choice(len(npy_paths), size=want, replace=False)
        npy_paths = [npy_paths[int(i)] for i in sorted(picks.tolist())]
        log.info(
            "local holdout: mix %d shard(s) from %s (%d available)",
            len(npy_paths), dataset, len(all_paths),
        )
    else:
        log.info(
            "local holdout: using all %d shard(s) under %s",
            len(npy_paths), dataset,
        )

    for p in npy_paths:
        log.info("  shard: %s (%.2f GB)", p.name, p.stat().st_size / 1e9)

    sequences, meta = sample_global_from_token_mmaps(npy_paths, n_public, seq_len, seed_str)
    meta["mode"] = "local_tokens_npy"
    meta["local_dataset"] = str(dataset)
    meta["used_shard_paths"] = [str(p) for p in npy_paths]
    meta["used_files"] = [p.name for p in npy_paths]
    meta["mixed_files_used"] = len(npy_paths)

    if vocab_size is not None and sequences:
        arr = np.asarray(sequences, dtype=np.uint32)
        max_id = int(arr.max()) if arr.size else -1
        if max_id >= vocab_size:
            raise ValueError(
                f"local holdout: token id {max_id} >= model vocab_size {vocab_size}"
            )

    h = hashlib.sha256()
    for key in sorted(meta["used_files"]):
        h.update(key.encode())
        h.update(b"\n")
    h.update(public_seed)
    indices_digest = h.hexdigest()
    return torch.tensor(sequences, dtype=torch.long), indices_digest, meta


def get_dataset_r2() -> Any:
    """Prefer real S3 dataset credentials; fall back to public Hippius HTTP."""
    if os.environ.get("TEUTONIC_DS_ENDPOINT") or os.environ.get("TEUTONIC_R2_ENDPOINT"):
        try:
            from eval.torch_runner import R2
            return R2()
        except Exception as exc:
            log.warning("S3 dataset client unavailable (%s); using Hippius HTTP", exc)
    log.info("validator eval: using Hippius HTTP mirror %s", HIPPIUS_BASE)
    return HippiusHttpR2()


def validator_style_paired_eval(
    king_dir: str,
    chall_dir: str,
    *,
    block_hash: str,
    hotkey: str,
    n_public: int,
    n_private: int = 0,
    seq_len: int = 2048,
    gpu_ids: list[int] | None = None,
    batch_size: int = 64,
    eval_alpha: float = 0.001,
    delta_threshold: float = 0.0025,
    n_bootstrap: int = 10000,
    vocab_size: int | None = None,
    mix_files: int = 0,
    local_dataset: str | Path | None = None,
) -> dict:
    """Run paired eval the same way eval_server does for a validator duel."""
    _ensure_logging()
    from eval.raw_dataset import sample_private_pool
    from eval.torch_runner import (
        MultiGPUEvaluator,
        run_paired_eval,
        sample_public_holdout,
    )

    use_local = bool(local_dataset and str(local_dataset).strip())
    if not use_local:
        os.environ.setdefault("TEUTONIC_EVAL_DATASET_MODE", "raw_hippius")
        if mix_files and int(mix_files) > 0:
            # raw_dataset.load_raw_sequences reads this env var
            os.environ["TEUTONIC_RAW_MIX_FILES"] = str(int(mix_files))

    if not hotkey:
        raise ValueError(
            "validator-style eval requires --sim-hotkey or TEUTONIC_SIM_HOTKEY "
            "(used with block_hash to derive holdout seeds)"
        )

    public_seed, boot_seed, private_seed = derive_eval_seeds(block_hash, hotkey)
    log.info(
        "validator eval: block_hash=%s… hotkey=%s… n_public=%d n_private=%d",
        block_hash[:16], hotkey[:16], n_public, n_private,
    )
    if use_local:
        log.info("validator eval: local dataset %s", local_dataset)
        if mix_files and int(mix_files) > 0:
            log.info("validator eval: local mix enabled (%d shards)", int(mix_files))
        print("[validator eval] holdout: sampling from local .npy shards...",
              flush=True)
        public_seqs, public_indices_digest, raw_meta = sample_local_public_holdout(
            local_dataset,
            public_seed,
            n_public,
            seq_len,
            mix_files=mix_files,
            vocab_size=vocab_size,
        )
    else:
        if mix_files and int(mix_files) > 0:
            log.info("validator eval: raw mix enabled (TEUTONIC_RAW_MIX_FILES=%s)", int(mix_files))
        r2 = get_dataset_r2()
        log.info("validator eval: building holdout (download + tokenize; may take 5-20 min on first run)")
        print("[validator eval] holdout: downloading/tokenizing parquet (no batch logs until this finishes)...",
              flush=True)
        public_seqs, public_indices_digest, raw_meta = sample_public_holdout(
            r2, RAW_SHARD_KEY, public_seed, n_public, seq_len, vocab_size=vocab_size,
        )

    n_shards = int((raw_meta or {}).get("mixed_files_used") or 0)
    if not n_shards:
        n_shards = len((raw_meta or {}).get("used_parquet_keys") or [])
    if not n_shards:
        n_shards = len((raw_meta or {}).get("used_files") or [])
    log.info(
        "validator eval: holdout ready — %d sequences from %d shard(s) (mode=%s)",
        int(public_seqs.shape[0]), n_shards, (raw_meta or {}).get("mode", "?"),
    )
    print(
        f"[validator eval] holdout ready: {public_seqs.shape[0]} seqs from "
        f"{n_shards} shard(s)",
        flush=True,
    )
    actual_public = int(public_seqs.shape[0])
    if actual_public < n_public:
        log.warning(
            "validator holdout undersized: got %d sequences, requested %d",
            actual_public, n_public,
        )

    if n_private > 0:
        import chain_config
        private_seqs, private_pool_digest = sample_private_pool(
            seq_len, n_private, chain_config.SEED_TOKENIZER_REPO,
            rng_seed=private_seed,
        )
        holdout = (
            public_seqs if private_seqs.shape[0] == 0
            else torch.cat([public_seqs, private_seqs], dim=0)
        )
    else:
        private_seqs = torch.zeros((0, seq_len), dtype=torch.int64)
        private_pool_digest = ""
        holdout = public_seqs

    if gpu_ids is None:
        gpu_ids = [0]

    king_eval = MultiGPUEvaluator(king_dir, gpu_ids, label="king")
    chall_eval = MultiGPUEvaluator(chall_dir, gpu_ids, label="challenger")
    log.info("validator eval: starting paired CE on %d sequences (batch_size=%d)",
             int(holdout.shape[0]), batch_size)
    print(f"[validator eval] paired CE starting: {holdout.shape[0]} sequences, batch_size={batch_size}",
          flush=True)
    try:
        verdict = run_paired_eval(
            king_eval, chall_eval,
            holdout, actual_public,
            boot_seed=boot_seed,
            eval_alpha=eval_alpha,
            delta_threshold=delta_threshold,
            n_bootstrap=n_bootstrap,
            batch_size=batch_size,
            on_progress=_progress_printer("paired"),
        )
    finally:
        king_eval.shutdown()
        chall_eval.shutdown()
        torch.cuda.empty_cache()

    verdict["eval_mode"] = "validator_local" if use_local else "validator"
    verdict["block_hash"] = block_hash
    verdict["hotkey"] = hotkey
    verdict["public_seed"] = public_seed.hex()
    verdict["boot_seed"] = boot_seed.hex()
    verdict["public_indices_digest"] = public_indices_digest
    verdict["private_pool_digest"] = private_pool_digest
    verdict["n_public_requested"] = n_public
    verdict["n_private_requested"] = n_private
    verdict["raw_meta"] = raw_meta
    verdict["delta_threshold"] = delta_threshold
    verdict["merged_dir"] = str(chall_dir)
    return verdict


def _parse_gpu_ids(spec: str) -> list[int]:
    spec = (spec or "0").strip()
    if not spec:
        return [0]
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def _check_local_model_weights(model_dir: Path, label: str) -> None:
    """Fail fast when safetensor shards look truncated (common after interrupted copy)."""
    idx_path = model_dir / "model.safetensors.index.json"
    if not idx_path.is_file():
        single = model_dir / "model.safetensors"
        if single.is_file() and single.stat().st_size < 1_000_000:
            raise RuntimeError(f"{label} model.safetensors looks truncated: {single}")
        return
    try:
        idx = json.loads(idx_path.read_text())
    except Exception:
        return
    shard_names = sorted(set(idx.get("weight_map", {}).values()))
    if not shard_names:
        return
    sizes = []
    for name in shard_names:
        p = model_dir / name
        if p.is_file():
            sizes.append(p.stat().st_size)
    if not sizes:
        return
    median = sorted(sizes)[len(sizes) // 2]
    for name in shard_names:
        p = model_dir / name
        if not p.is_file():
            raise RuntimeError(f"{label} missing weight shard: {p}")
        sz = p.stat().st_size
        if sz < median * 0.25:
            raise RuntimeError(
                f"{label} weight shard looks truncated: {p.name} "
                f"({sz / 1e9:.2f} GB; median shard ~{median / 1e9:.2f} GB). "
                "Re-copy the model with `cp -avL` from the Hippius HF cache snapshot."
            )


def _king_vocab_size(king_dir: str) -> int | None:
    cfg_path = Path(king_dir) / "config.json"
    if not cfg_path.is_file():
        return None
    try:
        cfg = json.loads(cfg_path.read_text())
        return int(cfg.get("vocab_size") or 0) or None
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Validator-aligned offline paired eval (same path as eval_server).",
    )
    ap.add_argument(
        "--king", default="/root/teutonic/s1-work-prod/king",
        help="Local king model directory (config.json + weights)",
    )
    ap.add_argument(
        "--challenger", default="/root/teutonic/s1-work-prod/iter_00/merged",
        help="Local challenger / merged model directory",
    )
    ap.add_argument(
        "--hotkey", default=os.environ.get("TEUTONIC_SIM_HOTKEY", ""),
        help="Submit hotkey ss58 (holdout seeds). Env: TEUTONIC_SIM_HOTKEY",
    )
    ap.add_argument(
        "--block-hash", default=os.environ.get("TEUTONIC_SIM_BLOCK_HASH", DEFAULT_BLOCK_HASH),
        help="Pinned block hash for holdout seeds (default: 64 zero hex digits)",
    )
    ap.add_argument("--n-public", type=int, default=1000,
                    help="Public holdout sequences (validator default 25600)")
    ap.add_argument("--n-private", type=int, default=0,
                    help="Private holdout sequences (0 on current finney config)")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--gpus", default="0", help="Comma-separated GPU ids for eval, e.g. 0 or 0,1")
    ap.add_argument(
        "--mix",
        type=int,
        default=0,
        help="Mix N shards: Hippius parquet files (default) or local .npy files with --local-dataset",
    )
    ap.add_argument(
        "--local-dataset",
        default=os.environ.get("LOCAL_DATASET_MANIFEST", ""),
        help="Local holdout source: directory of *.tokens.npy or manifest.json "
             "(env: LOCAL_DATASET_MANIFEST). Skips Hippius download.",
    )
    ap.add_argument("--alpha", type=float, default=0.001, help="Bootstrap LCB quantile")
    ap.add_argument("--delta", type=float, default=0.0025, help="Acceptance floor (nats/token)")
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument(
        "--report-out", default="/root/teutonic/s1-work-prod/verdict.json",
        help="Write verdict JSON (best.accepted gate for submit_challenger.py)",
    )
    args = ap.parse_args()

    _ensure_logging()

    king_dir = Path(args.king).expanduser().resolve()
    chall_dir = Path(args.challenger).expanduser().resolve()
    for label, path in (("king", king_dir), ("challenger", chall_dir)):
        if not (path / "config.json").is_file():
            ap.error(f"{label} path missing config.json: {path}")
        try:
            _check_local_model_weights(path, label)
        except RuntimeError as exc:
            ap.error(str(exc))

    hotkey = (args.hotkey or "").strip()
    if not hotkey:
        ap.error("--hotkey or TEUTONIC_SIM_HOTKEY is required")

    local_dataset = (args.local_dataset or "").strip()
    if not local_dataset:
        os.environ.setdefault("TEUTONIC_EVAL_DATASET_MODE", "raw_hippius")
        os.environ.setdefault("TEUTONIC_RAW_TOKENIZER_REPO", "Qwen/Qwen3-4B")

    log.info("king=%s", king_dir)
    log.info("challenger=%s", chall_dir)
    if local_dataset:
        log.info("local_dataset=%s", local_dataset)

    t0 = time.time()
    verdict = validator_style_paired_eval(
        str(king_dir),
        str(chall_dir),
        block_hash=args.block_hash.strip(),
        hotkey=hotkey,
        n_public=args.n_public,
        n_private=args.n_private,
        seq_len=args.seq_len,
        gpu_ids=_parse_gpu_ids(args.gpus),
        batch_size=args.batch_size,
        eval_alpha=args.alpha,
        delta_threshold=args.delta,
        n_bootstrap=args.n_bootstrap,
        vocab_size=_king_vocab_size(str(king_dir)),
        mix_files=int(args.mix or 0),
        local_dataset=local_dataset or None,
    )

    report = {
        "best": verdict,
        "history": [verdict],
        "king_dir": str(king_dir),
        "challenger_dir": str(chall_dir),
        "ts": time.time(),
        "elapsed_s": round(time.time() - t0, 1),
    }
    if args.report_out:
        out_path = Path(args.report_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        log.info("wrote verdict to %s", out_path)

    print(
        f"\n=== VALIDATOR EVAL DONE ===\n"
        f"accepted: {verdict['accepted']}\n"
        f"mu_hat:   {verdict['mu_hat']}\n"
        f"lcb:      {verdict['lcb']}\n"
        f"delta:    {verdict.get('delta_threshold', verdict.get('delta'))}\n"
        f"eval_mode: {verdict.get('eval_mode')}\n"
        f"elapsed:  {report['elapsed_s']}s\n",
        flush=True,
    )
    sys.exit(0 if verdict["accepted"] else 1)


if __name__ == "__main__":
    main()
