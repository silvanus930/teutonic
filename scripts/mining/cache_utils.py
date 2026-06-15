"""Digest-based caching for king models, dataset shards, and king-scored samples."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

from huggingface_hub import snapshot_download

log = logging.getLogger(__name__)

SCORE_CACHE_VERSION = 2


def king_digest_dir(work: Path, king_hash: str) -> Path:
    return work / "king_cache" / f"digest_{king_hash[:16]}"


def ensure_king_cached(
    hf_repo: str,
    revision: str | None,
    cache_dir: Path,
    hf_token: str = "",
    force: bool = False,
) -> Path:
    """Download king weights once per digest; reuse on subsequent runs."""
    cache_dir = Path(cache_dir)
    marker = cache_dir / ".complete"
    if marker.is_file() and (cache_dir / "config.json").is_file() and not force:
        log.info("king cache hit: %s", cache_dir)
        return cache_dir

    if cache_dir.exists() and force:
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    rev = revision
    if isinstance(rev, str) and rev.startswith("hf:"):
        rev = rev[len("hf:"):]
    if rev in ("local", "HEAD", ""):
        rev = None

    log.info("downloading king %s@%s -> %s", hf_repo, rev or "HEAD", cache_dir)
    snapshot_download(
        hf_repo,
        local_dir=str(cache_dir),
        revision=rev,
        token=hf_token or None,
        max_workers=16,
    )
    marker.write_text(json.dumps({"repo": hf_repo, "revision": revision or ""}))
    return cache_dir


def shard_cache_path(cache_root: Path, shard_key: str) -> Path:
    """Stable on-disk path for a downloaded dataset shard."""
    key_hash = hashlib.sha256(shard_key.encode()).hexdigest()[:16]
    name = Path(shard_key).name or "shard.npy"
    return Path(cache_root) / f"{key_hash}_{name}"


def download_shard_cached(shard_key: str, cache_root: Path, hippius_base: str) -> Path:
    """Return local shard path, downloading from Hippius only if missing."""
    local_path = Path(shard_key)
    if local_path.is_file() and local_path.stat().st_size > 1024:
        return local_path

    out = shard_cache_path(cache_root, shard_key)
    if out.exists() and out.stat().st_size > 1024:
        log.info("shard cache hit: %s", out)
        return out

    url = f"{hippius_base.rstrip('/')}/{shard_key}"
    log.info("downloading shard %s -> %s", url, out)
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["curl", "-fsSL", "-o", str(out), url])
    return out


def manifest_shard_hash(shard_keys: list[str], seq_len: int) -> str:
    h = hashlib.sha256()
    h.update(f"v{SCORE_CACHE_VERSION}".encode())
    h.update(str(seq_len).encode())
    for key in sorted(shard_keys):
        h.update(key.encode())
        h.update(b"|")
    return h.hexdigest()


def score_cache_dir(
    work: Path,
    king_hash: str,
    shard_keys: list[str],
    seq_len: int,
    extra_tag: str = "",
) -> Path:
    shard_hash = manifest_shard_hash(shard_keys, seq_len)[:12]
    base = work / "score_cache" / f"king_{king_hash[:16]}" / f"shards_{shard_hash}"
    if extra_tag:
        base = base / extra_tag
    return base


def _shard_key_hash(shard_key: str) -> str:
    return hashlib.sha256(shard_key.encode()).hexdigest()[:12]


def load_score_cache(cache_dir: Path, shard_keys: list[str]) -> list[dict] | None:
    """Load cached scored rows; returns None if any shard file is missing."""
    cache_dir = Path(cache_dir)
    key_to_local = {k: i for i, k in enumerate(shard_keys)}
    all_rows: list[dict] = []
    for shard_key in shard_keys:
        fname = f"scored_shard_{_shard_key_hash(shard_key)}.jsonl"
        path = cache_dir / fname
        if not path.is_file():
            log.info("score cache miss: %s (key=%s)", fname, shard_key)
            return None
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                sk = row.get("shard_key", shard_key)
                row["shard"] = key_to_local.get(sk, row.get("shard", 0))
                if "idx" not in row and "row_idx" in row:
                    row["idx"] = row["row_idx"]
                if "sample_index" not in row:
                    row["sample_index"] = row.get("idx")
                all_rows.append(row)
    log.info("score cache hit: %d rows from %d shard(s)", len(all_rows), len(shard_keys))
    return all_rows


def save_score_cache(
    cache_dir: Path,
    shard_keys: list[str],
    rows: list[dict],
    *,
    include_input_ids: bool = True,
) -> None:
    """Persist scored rows keyed by manifest shard identity."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    by_local: dict[int, list[dict]] = {i: [] for i in range(len(shard_keys))}
    for row in rows:
        si = int(row.get("shard", 0))
        if si in by_local:
            by_local[si].append(row)

    for local_idx, shard_rows in by_local.items():
        if not shard_rows:
            continue
        shard_key = shard_keys[local_idx]
        khash = _shard_key_hash(shard_key)
        fname = f"scored_shard_{khash}.jsonl"
        path = cache_dir / fname
        tmp = cache_dir / f"{fname}.tmp"
        with open(tmp, "w") as f:
            for row in shard_rows:
                payload = {
                    "shard_key": shard_key,
                    "shard_key_hash": khash,
                    "shard_index": local_idx,
                    "row_idx": row.get("idx"),
                    "sample_index": row.get("idx"),
                    "loss": row.get("loss"),
                    "bucket": row.get("bucket", ""),
                    "unique_r": row.get("unique_r"),
                    "unique_ratio": row.get("unique_r"),
                    "rep_r": row.get("rep_r"),
                    "repetition_ratio": row.get("rep_r"),
                    "rep_ng4": row.get("rep_ng4"),
                    "4gram_repetition": row.get("rep_ng4"),
                    "cache_version": SCORE_CACHE_VERSION,
                }
                if include_input_ids and row.get("input_ids") is not None:
                    payload["input_ids"] = row["input_ids"]
                f.write(json.dumps(payload) + "\n")
        shutil.move(str(tmp), str(path))

    meta = {
        "cache_version": SCORE_CACHE_VERSION,
        "n_rows": len(rows),
        "shard_keys": shard_keys,
    }
    (cache_dir / "cache_meta.json").write_text(json.dumps(meta, indent=2))
    log.info("score cache saved: %d shard file(s) in %s", len(by_local), cache_dir)
