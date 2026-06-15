#!/usr/bin/env python3
"""Prefetch Teutonic mining datasets (manifests + .npy shards) before training.

Uses the same cache layout as train_challenger.py so downloads are reused:

  <cache-dir>/manifests/<dataset>/manifest.json
  <cache-dir>/shards/<dataset>/<hash>_<name>.npy

Examples:

  # Mixture v2 (probe/strong default) — 12 shards per dataset
  python scripts/mining/download_dataset.py \\
    --work /root/teutonic-mining/probe

  # Download every shard in each manifest (large)
  python scripts/mining/download_dataset.py \\
    --work /root/teutonic-mining/strong \\
    --download-all --workers 8

  # Subset + dry-run
  python scripts/mining/download_dataset.py \\
    --datasets automathtext-v2,finewebedu \\
    --shards-per-dataset 20 --dry-run

  # Legacy single manifest
  python scripts/mining/download_dataset.py \\
    --dataset-preset legacy \\
    --cache-dir /data/teutonic-cache
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_mining_dir = os.path.dirname(os.path.abspath(__file__))
if _mining_dir not in sys.path:
    sys.path.insert(0, _mining_dir)

from cache_utils import (  # noqa: E402
    download_shard_cached,
    ensure_king_cached,
    shard_cache_path,
)
from dataset_mixture import (  # noqa: E402
    build_mixture_config,
    download_mixture_datasets,
    parse_dataset_manifest_arg,
    parse_dataset_weight_arg,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [download_dataset] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("download_dataset")

DASHBOARD_URL = os.environ.get(
    "TEUTONIC_DASHBOARD_URL",
    "https://us-east-1.hippius.com/teutonic-sn3/dashboard.json",
)
HIPPIUS_BASE = os.environ.get(
    "TEUTONIC_HIPPIUS_HTTP_BASE", "https://s3.hippius.com/teutonic-sn3",
).rstrip("/")


def _human_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} TiB"


def fetch_king_meta() -> dict:
    log.info("fetching dashboard %s", DASHBOARD_URL)
    with urllib.request.urlopen(DASHBOARD_URL, timeout=30) as resp:
        data = json.loads(resp.read())
    king = data["king"]
    repo = king.get("hf_repo") or king.get("model_repo") or king.get("previous_repo")
    revision = king.get("king_revision") or king.get("revision") or king.get("king_digest")
    if not repo:
        raise KeyError(f"no king repo in dashboard: {list(king.keys())}")
    return {"hf_repo": repo, "king_revision": revision}


def _resolve_local_manifest_paths(manifest: dict, base: Path) -> None:
    for field in ("shards", "train_shards", "eval_shards"):
        for entry in manifest.get(field) or []:
            key = (entry.get("key") or "").strip()
            if not key:
                continue
            key_path = Path(key)
            if not key_path.is_absolute():
                entry["key"] = str((base / key).resolve())


def load_legacy_manifest(cache_dir: Path, local_manifest: str) -> dict:
    local = (local_manifest or os.environ.get("LOCAL_DATASET_MANIFEST", "")).strip()
    if local:
        path = Path(local)
        if not path.is_file():
            raise FileNotFoundError(f"local manifest not found: {path}")
        manifest = json.loads(path.read_text())
        _resolve_local_manifest_paths(manifest, path.parent)
        return manifest

    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        url = f"{HIPPIUS_BASE}/dataset/v2/manifest.json"
        log.info("downloading legacy manifest %s", url)
        subprocess.check_call(["curl", "-fsSL", "-o", str(manifest_path), url])
    return json.loads(manifest_path.read_text())


def _legacy_shard_keys(
    manifest: dict,
    *,
    download_all: bool,
    n_shards: int,
    shard_start: int,
) -> list[str]:
    fields = ("train_shards", "eval_shards", "shards")
    keys: list[str] = []
    seen: set[str] = set()
    for field in fields:
        entries = manifest.get(field) or []
        if not entries:
            continue
        if field in ("train_shards", "shards") and not download_all and n_shards > 0:
            selected = entries[shard_start: shard_start + n_shards]
        else:
            selected = entries
        for entry in selected:
            key = str(entry.get("key") or "").strip()
            if key and key not in seen:
                seen.add(key)
                keys.append(key)
        if field == "train_shards" and keys:
            break
    if not keys and manifest.get("shards"):
        all_entries = manifest["shards"]
        if download_all or n_shards <= 0:
            keys = [str(e.get("key", "")) for e in all_entries if e.get("key")]
        else:
            keys = [
                str(e.get("key", ""))
                for e in all_entries[shard_start: shard_start + n_shards]
                if e.get("key")
            ]
    return keys


def _download_legacy_shard(
    shard_key: str,
    cache_dir: Path,
    *,
    force: bool,
) -> dict:
    local = Path(shard_key)
    if local.is_file() and local.stat().st_size > 1024:
        return {
            "shard_key": shard_key,
            "path": str(local),
            "status": "cached",
            "bytes": local.stat().st_size,
        }

    legacy_shard_dir = cache_dir / "shards" / "legacy"
    out = shard_cache_path(legacy_shard_dir, shard_key)
    if out.is_file() and out.stat().st_size > 1024 and not force:
        return {
            "shard_key": shard_key,
            "path": str(out),
            "status": "cached",
            "bytes": out.stat().st_size,
        }
    if force and out.is_file():
        out.unlink()

    try:
        path = download_shard_cached(shard_key, legacy_shard_dir, HIPPIUS_BASE)
        return {
            "shard_key": shard_key,
            "path": str(path),
            "status": "downloaded",
            "bytes": path.stat().st_size,
        }
    except Exception as exc:
        return {
            "shard_key": shard_key,
            "path": str(out),
            "status": "failed",
            "bytes": 0,
            "error": str(exc),
        }


def download_legacy_dataset(
    cache_dir: Path,
    *,
    local_manifest: str = "",
    download_all: bool = False,
    n_shards: int = 12,
    shard_start: int = 0,
    force: bool = False,
    workers: int = 4,
    dry_run: bool = False,
    manifests_only: bool = False,
) -> dict:
    t0 = time.time()
    manifest = load_legacy_manifest(cache_dir, local_manifest)
    shard_keys = _legacy_shard_keys(
        manifest,
        download_all=download_all,
        n_shards=n_shards,
        shard_start=shard_start,
    )
    log.info("legacy manifest: %d shard(s) planned", len(shard_keys))

    if dry_run or manifests_only:
        return {
            "mode": "legacy",
            "manifest_path": str(cache_dir / "manifest.json"),
            "shard_keys": shard_keys,
            "totals": {
                "shards_planned": len(shard_keys),
                "shards_downloaded": 0,
                "shards_cached": 0,
                "bytes": 0,
            },
            "dry_run": dry_run,
            "manifests_only": manifests_only,
            "elapsed_s": round(time.time() - t0, 2),
        }

    workers = max(1, workers)
    results: list[dict] = []
    if workers == 1:
        for key in shard_keys:
            results.append(_download_legacy_shard(key, cache_dir, force=force))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {
                pool.submit(_download_legacy_shard, key, cache_dir, force=force): key
                for key in shard_keys
            }
            for fut in as_completed(futs):
                results.append(fut.result())

    failed = [r for r in results if r.get("status") == "failed"]
    if failed:
        first = failed[0]
        raise RuntimeError(
            f"{len(failed)} shard download(s) failed; first: {first.get('shard_key')}: "
            f"{first.get('error')}"
        )

    downloaded = sum(1 for r in results if r.get("status") == "downloaded")
    cached = sum(1 for r in results if r.get("status") == "cached")
    total_bytes = sum(int(r.get("bytes") or 0) for r in results)
    return {
        "mode": "legacy",
        "manifest_path": str(cache_dir / "manifest.json"),
        "shard_keys": shard_keys,
        "totals": {
            "shards_planned": len(shard_keys),
            "shards_downloaded": downloaded,
            "shards_cached": cached,
            "bytes": total_bytes,
        },
        "results": results,
        "elapsed_s": round(time.time() - t0, 2),
    }


def download_king(work: Path, *, hf_token: str, force: bool) -> dict:
    king_meta = fetch_king_meta()
    rev = king_meta.get("king_revision")
    cache_key = hashlib.sha256(f"{king_meta['hf_repo']}|{rev or 'HEAD'}".encode()).hexdigest()[:16]
    staging = work / "king_cache" / f"staging_{cache_key}"
    path = ensure_king_cached(
        king_meta["hf_repo"],
        rev,
        staging,
        hf_token=hf_token,
        force=force,
    )
    return {
        "hf_repo": king_meta["hf_repo"],
        "revision": rev,
        "path": str(path),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Prefetch Teutonic dataset manifests and shards.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--work", default="/root/teutonic-mining/work",
        help="Work dir; cache defaults to <work>/cache (same as train_challenger).",
    )
    ap.add_argument(
        "--cache-dir", default="",
        help="Override cache root (default: <work>/cache).",
    )
    ap.add_argument(
        "--dataset-preset", default="teutonic-mixture-v2",
        choices=("teutonic-mixture-v2", "legacy"),
    )
    ap.add_argument(
        "--dataset-mix", default="",
        help="JSON mixture config (see dataset_mix_quasar_v4.json).",
    )
    ap.add_argument("--dataset-manifest", action="append", default=[])
    ap.add_argument("--dataset-weight", action="append", default=[])
    ap.add_argument(
        "--datasets", default="",
        help="Comma-separated subset of mixture dataset names.",
    )
    ap.add_argument(
        "--local-dataset-manifest", default="",
        help="Legacy mode: local manifest.json path.",
    )
    ap.add_argument(
        "--shards-per-dataset", type=int, default=12,
        help="Mixture mode: shard pool per dataset (matches --mix-shards-per-dataset).",
    )
    ap.add_argument("--n-shards", type=int, default=12, help="Legacy mode: train shards to fetch.")
    ap.add_argument("--shard-start", type=int, default=0)
    ap.add_argument(
        "--download-all", action="store_true",
        help="Download every shard listed in manifest(s) (large).",
    )
    ap.add_argument("--seed", type=int, default=42, help="Shard pool RNG seed (mixture mode).")
    ap.add_argument("--workers", type=int, default=4, help="Parallel shard downloads.")
    ap.add_argument("--manifests-only", action="store_true", help="Fetch manifests only.")
    ap.add_argument("--dry-run", action="store_true", help="Plan downloads without fetching shards.")
    ap.add_argument("--force", action="store_true", help="Re-download even if cached.")
    ap.add_argument(
        "--download-king", action="store_true",
        help="Also download current king model into <work>/king_cache/.",
    )
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    ap.add_argument(
        "--summary-out", default="",
        help="Write JSON summary (default: <cache-dir>/download_summary.json).",
    )
    args = ap.parse_args()

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else work / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "cache_dir": str(cache_dir),
        "work": str(work),
        "ts": time.time(),
    }

    if args.download_king:
        log.info("downloading king model...")
        summary["king"] = download_king(work, hf_token=args.hf_token, force=args.force)

    mix_json = (args.dataset_mix or os.environ.get("TEUTONIC_DATASET_MIX", "")).strip()
    use_mixture = args.dataset_preset != "legacy" or bool(mix_json) or bool(args.dataset_manifest)

    if use_mixture:
        manifest_overrides = [parse_dataset_manifest_arg(m) for m in args.dataset_manifest]
        weight_overrides = dict(parse_dataset_weight_arg(w) for w in args.dataset_weight)
        dataset_names = [x.strip() for x in args.datasets.split(",") if x.strip()] or None
        cfg = build_mixture_config(
            preset=args.dataset_preset or "teutonic-mixture-v2",
            cache_root=cache_dir,
            dataset_manifests=manifest_overrides or None,
            dataset_weights=weight_overrides or None,
            dataset_names=dataset_names,
            mix_json_path=mix_json,
        )
        if cfg is None:
            raise RuntimeError("mixture config is empty; check --dataset-preset / --dataset-mix")
        log.info(
            "mixture preset=%s datasets=%s weights=%s",
            cfg.preset,
            [s.name for s in cfg.sources],
            cfg.weights,
        )
        ds_summary = download_mixture_datasets(
            cfg,
            shards_per_dataset=args.shards_per_dataset,
            seed=args.seed,
            dataset_names=dataset_names,
            download_all=args.download_all,
            manifests_only=args.manifests_only,
            force=args.force,
            workers=args.workers,
            dry_run=args.dry_run,
        )
        summary.update(ds_summary)
    else:
        ds_summary = download_legacy_dataset(
            cache_dir,
            local_manifest=args.local_dataset_manifest,
            download_all=args.download_all,
            n_shards=args.n_shards,
            shard_start=args.shard_start,
            force=args.force,
            workers=args.workers,
            dry_run=args.dry_run,
            manifests_only=args.manifests_only,
        )
        summary.update(ds_summary)

    totals = summary.get("totals") or {}
    log.info(
        "done: planned=%s downloaded=%s cached=%s bytes=%s elapsed=%ss",
        totals.get("shards_planned", "?"),
        totals.get("shards_downloaded", "?"),
        totals.get("shards_cached", "?"),
        _human_bytes(int(totals.get("bytes") or 0)),
        summary.get("elapsed_s", "?"),
    )

    summary_path = Path(args.summary_out) if args.summary_out else cache_dir / "download_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info("summary written: %s", summary_path)


if __name__ == "__main__":
    main()
