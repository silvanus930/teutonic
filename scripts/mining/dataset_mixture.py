"""Multi-manifest weighted dataset mixture for validator-aligned mining.

Supports the teutonic-mixture-v2 preset (automathtext-v2, quasar-sn3,
ultradata-math, finewebedu) with per-dataset caching, allocation, scoring,
curriculum, and offline eval pools.
"""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dataset_mix import (
    SequenceRef,
    load_shard_array,
    shard_num_sequences,
)

log = logging.getLogger(__name__)

DEFAULT_SEQ_LEN = 2048

TEUTONIC_MIXTURE_V2: dict = {
    "preset": "teutonic-mixture-v2",
    "seq_len": DEFAULT_SEQ_LEN,
    "datasets": [
        {
            "name": "automathtext-v2",
            "manifest_url": (
                "https://eu-central-1.hippius.com/teutonic-sn3/dataset/"
                "automathtext-v2-quasar-10b/manifest.json"
            ),
            "weight": 0.35,
        },
        {
            "name": "quasar-sn3",
            "manifest_url": (
                "https://us-east-1.hippius.com/teutonic-sn3/dataset/"
                "quasar-sn3-retok/manifest.json"
            ),
            "weight": 0.05,
        },
        {
            "name": "ultradata-math",
            "manifest_url": (
                "https://eu-central-1.hippius.com/teutonic-sn3/dataset/"
                "ultradata-math-quasar-10b/manifest.json"
            ),
            "weight": 0.35,
        },
        {
            "name": "finewebedu",
            "manifest_url": (
                "https://eu-central-1.hippius.com/teutonic-sn3/dataset/"
                "finewebedu/manifest.json"
            ),
            "weight": 0.25,
        },
    ],
}

DEFAULT_BUCKET_MIX: dict[str, tuple[float, float, float]] = {
    "automathtext-v2": (0.50, 0.40, 0.10),
    "ultradata-math": (0.50, 0.40, 0.10),
    "finewebedu": (0.70, 0.20, 0.10),
    "quasar-sn3": (0.70, 0.20, 0.10),
}

MAJOR_DATASETS = ("automathtext-v2", "ultradata-math", "finewebedu")


@dataclass(frozen=True)
class DatasetSource:
    name: str
    manifest_url: str
    weight: float
    hippius_base: str
    manifest_cache: Path
    shard_cache_dir: Path


@dataclass
class MixtureConfig:
    preset: str
    seq_len: int
    sources: tuple[DatasetSource, ...]
    weights: dict[str, float]
    bucket_mix: dict[str, tuple[float, float, float]] = field(default_factory=dict)


def hippius_base_from_url(manifest_url: str) -> str:
    parsed = urllib.parse.urlparse(manifest_url)
    if parsed.scheme and parsed.netloc:
        path = parsed.path.split("/dataset/")[0].rstrip("/")
        return f"{parsed.scheme}://{parsed.netloc}{path}".rstrip("/")
    return "https://s3.hippius.com/teutonic-sn3"


def manifest_hash(manifest: dict) -> str:
    keys = sorted(str(s.get("key", "")) for s in (manifest.get("shards") or []))
    payload = json.dumps({"shards": keys, "version": manifest.get("version")}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def normalize_weights(raw: dict[str, float], *, warn: bool = True) -> dict[str, float]:
    total = sum(raw.values())
    if total <= 0:
        raise ValueError("dataset weights must sum to a positive value")
    if warn and abs(total - 1.0) > 1e-3:
        log.warning("dataset weights sum to %.4f — normalizing to 1.0", total)
    return {k: v / total for k, v in raw.items()}


def allocate_weighted_counts(
    total: int,
    weights: dict[str, float],
    *,
    min_per_source: int = 1,
) -> dict[str, int]:
    """Largest-remainder allocation; each source gets at least min_per_source when possible."""
    if total <= 0:
        return {k: 0 for k in weights}
    names = list(weights.keys())
    n = len(names)
    if total < n * min_per_source:
        min_per_source = 0
    floor = {name: min_per_source for name in names}
    remaining = total - sum(floor.values())
    if remaining < 0:
        raise ValueError(f"total={total} too small for min_per_source={min_per_source}")

    raw = {name: weights[name] * remaining for name in names}
    base = {name: floor[name] + int(raw[name]) for name in names}
    rem = total - sum(base.values())
    order = sorted(names, key=lambda k: raw[k] - int(raw[k]), reverse=True)
    for i in range(rem):
        base[order[i % len(order)]] += 1
    return base


def parse_dataset_manifest_arg(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"expected name=url, got {spec!r}")
    name, url = spec.split("=", 1)
    name, url = name.strip(), url.strip()
    if not name or not url:
        raise ValueError(f"invalid dataset manifest spec {spec!r}")
    return name, url


def parse_dataset_weight_arg(spec: str) -> tuple[str, float]:
    if "=" not in spec:
        raise ValueError(f"expected name=weight, got {spec!r}")
    name, val = spec.split("=", 1)
    return name.strip(), float(val.strip())


def parse_bucket_mix_arg(spec: str) -> tuple[str, tuple[float, float, float]]:
    """Parse per-dataset bucket mix: name=general:hard:easy (e.g. finewebedu=0.7:0.2:0.1)."""
    if "=" not in spec:
        raise ValueError(f"expected name=g:h:e, got {spec!r}")
    name, rest = spec.split("=", 1)
    parts = [float(x.strip()) for x in rest.split(":")]
    if len(parts) != 3:
        raise ValueError(f"bucket mix needs 3 fractions, got {spec!r}")
    return name.strip(), (parts[0], parts[1], parts[2])


def build_mixture_config(
    *,
    preset: str = "",
    cache_root: Path,
    seq_len: int = DEFAULT_SEQ_LEN,
    dataset_manifests: list[tuple[str, str]] | None = None,
    dataset_weights: dict[str, float] | None = None,
    dataset_names: list[str] | None = None,
    mix_json_path: str = "",
    bucket_mix_overrides: dict[str, tuple[float, float, float]] | None = None,
) -> MixtureConfig | None:
    """Return MixtureConfig for mixture presets, or None for legacy single-manifest mode."""
    if preset == "legacy":
        return None

    raw_cfg: dict
    preset_name = preset or "teutonic-mixture-v2"

    if mix_json_path:
        raw_cfg = json.loads(Path(mix_json_path).read_text())
        preset_name = raw_cfg.get("preset") or preset_name
    elif preset_name == "teutonic-mixture-v2":
        raw_cfg = dict(TEUTONIC_MIXTURE_V2)
    elif preset in ("", "teutonic-mixture-v2"):
        raw_cfg = dict(TEUTONIC_MIXTURE_V2)
    else:
        raise ValueError(f"unknown dataset preset {preset!r}; use teutonic-mixture-v2 or legacy")

    entries = list(raw_cfg.get("datasets") or [])
    overrides = {k: v for k, v in (dataset_manifests or [])}
    weights_override = dict(dataset_weights or {})

    sources: list[DatasetSource] = []
    weight_map: dict[str, float] = {}
    for item in entries:
        name = str(item.get("name") or "").strip()
        if dataset_names and name not in dataset_names:
            continue
        url = overrides.get(name) or str(item.get("manifest_url") or item.get("url") or "").strip()
        weight = weights_override.get(name, float(item.get("weight") or 0.0))
        if not name or not url or weight <= 0:
            continue
        base = hippius_base_from_url(url)
        manifest_cache = cache_root / "manifests" / name / "manifest.json"
        shard_cache_dir = cache_root / "shards" / name
        sources.append(DatasetSource(
            name=name,
            manifest_url=url,
            weight=weight,
            hippius_base=base,
            manifest_cache=manifest_cache,
            shard_cache_dir=shard_cache_dir,
        ))
        weight_map[name] = weight

    for name, url in overrides.items():
        if any(s.name == name for s in sources):
            continue
        weight = weights_override.get(name, 0.0)
        if weight <= 0:
            raise ValueError(f"--dataset-manifest {name}=... requires --dataset-weight {name}=...")
        base = hippius_base_from_url(url)
        sources.append(DatasetSource(
            name=name,
            manifest_url=url,
            weight=weight,
            hippius_base=base,
            manifest_cache=cache_root / "manifests" / name / "manifest.json",
            shard_cache_dir=cache_root / "shards" / name,
        ))
        weight_map[name] = weight

    if not sources:
        raise ValueError("no datasets configured for mixture")

    norm_weights = normalize_weights(weight_map)
    bucket_mix = dict(DEFAULT_BUCKET_MIX)
    if bucket_mix_overrides:
        bucket_mix.update(bucket_mix_overrides)
    return MixtureConfig(
        preset=preset_name,
        seq_len=int(raw_cfg.get("seq_len") or seq_len),
        sources=tuple(sources),
        weights=norm_weights,
        bucket_mix=bucket_mix,
    )


def fetch_manifest_cached(source: DatasetSource) -> dict:
    source.manifest_cache.parent.mkdir(parents=True, exist_ok=True)
    if not source.manifest_cache.is_file():
        log.info("fetching manifest %s -> %s", source.manifest_url, source.manifest_cache)
        req = urllib.request.Request(
            source.manifest_url, headers={"User-Agent": "teutonic-train/1"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            source.manifest_cache.write_bytes(resp.read())
    return json.loads(source.manifest_cache.read_text())


def resolve_shard_url(shard_key: str, hippius_base: str) -> str:
    key = shard_key.strip()
    if key.startswith("http://") or key.startswith("https://"):
        return key
    return f"{hippius_base.rstrip('/')}/{key.lstrip('/')}"


def download_shard_cached(
    source: DatasetSource, shard_key: str, *, force: bool = False,
) -> Path:
    local_path = Path(shard_key)
    if local_path.is_file() and local_path.stat().st_size > 1024:
        return local_path

    source.shard_cache_dir.mkdir(parents=True, exist_ok=True)
    key_hash = hashlib.sha256(shard_key.encode()).hexdigest()[:16]
    name = Path(shard_key).name or "shard.npy"
    out = source.shard_cache_dir / f"{key_hash}_{name}"
    if out.is_file() and out.stat().st_size > 1024:
        if not force:
            return out
        out.unlink()

    url = resolve_shard_url(shard_key, source.hippius_base)
    log.info("downloading %s shard %s", source.name, name)
    subprocess.check_call(["curl", "-fsSL", "-o", str(out), url])
    return out


class MixtureShardStore:
    """Lazy per-dataset shard loader with namespaced cache paths."""

    def __init__(self, cfg: MixtureConfig):
        self.cfg = cfg
        self._manifests: dict[str, dict] = {}
        self._shard_meta: dict[str, list[tuple[str, int]]] = {}
        self._arrays: dict[str, np.ndarray] = {}
        self._key_to_local: dict[str, int] = {}
        self._local_keys: list[str] = []

        for source in cfg.sources:
            manifest = fetch_manifest_cached(source)
            self._manifests[source.name] = manifest
            entries = []
            for entry in manifest.get("shards") or []:
                key = str(entry.get("key") or "").strip()
                if not key:
                    continue
                n_seq = shard_num_sequences(entry, cfg.seq_len)
                if n_seq > 0:
                    entries.append((key, n_seq))
            if not entries:
                raise RuntimeError(f"dataset {source.name} has no usable shards")
            self._shard_meta[source.name] = entries
            log.info(
                "mixture dataset %s: weight=%.2f shards=%d manifest_hash=%s",
                source.name, cfg.weights[source.name], len(entries),
                manifest_hash(manifest),
            )

    def select_shard_pool(
        self, dataset_name: str, seed: int, shards_per_dataset: int,
    ) -> list[tuple[str, int]]:
        rng = np.random.default_rng(seed)
        entries = self._shard_meta[dataset_name]
        n_pick = min(max(1, shards_per_dataset), len(entries))
        pick = rng.choice(len(entries), size=n_pick, replace=False)
        return [entries[int(i)] for i in pick]

    def sample_refs(
        self, dataset_name: str, n: int, seed: int, shards_per_dataset: int,
    ) -> list[SequenceRef]:
        rng = np.random.default_rng(seed)
        pool = self.select_shard_pool(dataset_name, seed, shards_per_dataset)
        refs: list[SequenceRef] = []
        for _ in range(n):
            shard_key, n_rows = pool[int(rng.integers(0, len(pool)))]
            row_i = int(rng.integers(0, n_rows))
            refs.append(SequenceRef(dataset_name, shard_key, row_i))
        return refs

    def ensure_loaded(self, shard_keys: set[str]) -> None:
        for key in sorted(shard_keys):
            if key not in self._arrays:
                self._load_key(key)

    def _load_key(self, shard_key: str) -> None:
        source = self._source_for_key(shard_key)
        path = download_shard_cached(source, shard_key)
        arr = load_shard_array(path, self.cfg.seq_len)
        self._arrays[shard_key] = arr
        if shard_key not in self._key_to_local:
            self._key_to_local[shard_key] = len(self._local_keys)
            self._local_keys.append(shard_key)
        log.info("loaded %s shard %s (%d seq)", source.name, Path(shard_key).name, len(arr))

    def _source_for_key(self, shard_key: str) -> DatasetSource:
        for source in self.cfg.sources:
            manifest = self._manifests[source.name]
            keys = {str(s.get("key", "")) for s in manifest.get("shards") or []}
            if shard_key in keys:
                return source
        for source in self.cfg.sources:
            if shard_key.startswith("http") and source.name in shard_key:
                return source
        return self.cfg.sources[0]

    def local_index(self, shard_key: str) -> int:
        if shard_key not in self._arrays:
            self._load_key(shard_key)
        return self._key_to_local[shard_key]

    @property
    def arrays(self) -> list[np.ndarray]:
        return [self._arrays[k] for k in self._local_keys]

    @property
    def keys(self) -> list[str]:
        return list(self._local_keys)

    def refs_to_candidates(self, refs: list[SequenceRef]) -> list[tuple[int, int]]:
        self.ensure_loaded({r.shard_key for r in refs})
        return [(self.local_index(r.shard_key), r.row_idx) for r in refs]

    def manifest_for(self, dataset_name: str) -> dict:
        return self._manifests[dataset_name]

    def score_cache_dir(
        self, work: Path, king_hash: str, dataset_name: str,
    ) -> Path:
        manifest = self._manifests[dataset_name]
        mhash = manifest_hash(manifest)
        return (
            work / "score_cache" / f"king_{king_hash[:16]}"
            / "datasets" / dataset_name / f"manifest_{mhash}" / f"seq_{self.cfg.seq_len}"
        )


def save_allocation_summary(work: Path, allocations: dict) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    path = work / "allocation_summary.json"
    path.write_text(json.dumps(allocations, indent=2))
    return path


def build_mixture_allocations(
    cfg: MixtureConfig,
    *,
    n_score: int,
    train_per_iter: int,
    val_size: int,
    n_eval: int,
) -> dict:
    alloc = {
        "preset": cfg.preset,
        "weights": cfg.weights,
        "n_score": allocate_weighted_counts(n_score, cfg.weights),
        "train_per_iter": allocate_weighted_counts(train_per_iter, cfg.weights),
        "val_size": allocate_weighted_counts(val_size, cfg.weights),
        "n_eval": allocate_weighted_counts(n_eval, cfg.weights),
    }
    alloc["totals"] = {
        "n_score": sum(alloc["n_score"].values()),
        "train_per_iter": sum(alloc["train_per_iter"].values()),
        "val_size": sum(alloc["val_size"].values()),
        "n_eval": sum(alloc["n_eval"].values()),
    }
    return alloc


def refs_to_eval_pool(
    store: MixtureShardStore, refs: list[SequenceRef],
) -> tuple[np.ndarray, list[int]]:
    """Materialize eval holdout rows from sequence refs."""
    if not refs:
        return np.empty((0, store.cfg.seq_len), dtype=np.uint32), []
    store.ensure_loaded({r.shard_key for r in refs})
    rows = [store._arrays[r.shard_key][r.row_idx] for r in refs]
    return np.stack(rows, axis=0), list(range(len(rows)))


def prepare_mixture_eval_pools(
    store: MixtureShardStore,
    cfg: MixtureConfig,
    n_eval_alloc: dict[str, int],
    seed: int,
    shards_per_dataset: int,
) -> dict[str, tuple[np.ndarray, list[int]]]:
    pools: dict[str, tuple[np.ndarray, list[int]]] = {}
    for source in cfg.sources:
        n = n_eval_alloc.get(source.name, 0)
        if n <= 0:
            continue
        refs = store.sample_refs(source.name, n, seed + 0xE1A + hash(source.name) % 100000, shards_per_dataset)
        pools[source.name] = refs_to_eval_pool(store, refs)
        log.info("eval pool %s: %d sequences", source.name, len(refs))
    return pools


@dataclass
class ShardDownloadResult:
    dataset: str
    shard_key: str
    path: str
    status: str
    bytes: int
    error: str = ""


def collect_mixture_shard_keys(
    cfg: MixtureConfig,
    *,
    shards_per_dataset: int = 12,
    seed: int = 42,
    dataset_names: list[str] | None = None,
    download_all: bool = False,
) -> dict[str, list[str]]:
    """Return shard keys to prefetch per dataset (same pool logic as training)."""
    store = MixtureShardStore(cfg)
    keys_by_dataset: dict[str, list[str]] = {}
    for source in cfg.sources:
        if dataset_names and source.name not in dataset_names:
            continue
        entries = store._shard_meta[source.name]
        if download_all:
            keys = [key for key, _ in entries]
        else:
            pool = store.select_shard_pool(source.name, seed, shards_per_dataset)
            keys = [key for key, _ in pool]
        keys_by_dataset[source.name] = keys
        log.info(
            "dataset %s: selected %d / %d shard(s)",
            source.name, len(keys), len(entries),
        )
    return keys_by_dataset


def _shard_download_status(
    source: DatasetSource, shard_key: str, *, force: bool,
) -> tuple[str, Path]:
    """Return (status, path) without downloading when status would be 'cached'."""
    local_path = Path(shard_key)
    if local_path.is_file() and local_path.stat().st_size > 1024:
        return "cached", local_path

    key_hash = hashlib.sha256(shard_key.encode()).hexdigest()[:16]
    name = Path(shard_key).name or "shard.npy"
    out = source.shard_cache_dir / f"{key_hash}_{name}"
    if out.is_file() and out.stat().st_size > 1024 and not force:
        return "cached", out
    return "pending", out


def _download_one_shard(
    source: DatasetSource, shard_key: str, *, force: bool,
) -> ShardDownloadResult:
    status_before, expected = _shard_download_status(source, shard_key, force=force)
    if status_before == "cached":
        return ShardDownloadResult(
            dataset=source.name,
            shard_key=shard_key,
            path=str(expected),
            status="cached",
            bytes=expected.stat().st_size,
        )
    try:
        path = download_shard_cached(source, shard_key, force=force)
        return ShardDownloadResult(
            dataset=source.name,
            shard_key=shard_key,
            path=str(path),
            status="downloaded",
            bytes=path.stat().st_size,
        )
    except Exception as exc:
        return ShardDownloadResult(
            dataset=source.name,
            shard_key=shard_key,
            path=str(expected),
            status="failed",
            bytes=0,
            error=str(exc),
        )


def download_mixture_datasets(
    cfg: MixtureConfig,
    *,
    shards_per_dataset: int = 12,
    seed: int = 42,
    dataset_names: list[str] | None = None,
    download_all: bool = False,
    manifests_only: bool = False,
    force: bool = False,
    workers: int = 4,
    dry_run: bool = False,
) -> dict:
    """Prefetch manifests and shards for a mixture config."""
    t0 = time.time()
    manifests: dict[str, dict] = {}
    for source in cfg.sources:
        if dataset_names and source.name not in dataset_names:
            continue
        manifests[source.name] = fetch_manifest_cached(source)

    keys_by_dataset = collect_mixture_shard_keys(
        cfg,
        shards_per_dataset=shards_per_dataset,
        seed=seed,
        dataset_names=dataset_names,
        download_all=download_all,
    )

    planned: list[tuple[DatasetSource, str]] = []
    for source in cfg.sources:
        if dataset_names and source.name not in dataset_names:
            continue
        for shard_key in keys_by_dataset.get(source.name, []):
            planned.append((source, shard_key))

    if dry_run or manifests_only:
        return {
            "preset": cfg.preset,
            "manifests_only": manifests_only,
            "dry_run": dry_run,
            "datasets": {
                name: {
                    "manifest_url": next(s.manifest_url for s in cfg.sources if s.name == name),
                    "manifest_cache": str(next(s.manifest_cache for s in cfg.sources if s.name == name)),
                    "manifest_hash": manifest_hash(manifests[name]),
                    "shard_count_manifest": len(manifests[name].get("shards") or []),
                    "shard_count_planned": len(keys_by_dataset.get(name, [])),
                    "shard_keys": keys_by_dataset.get(name, []),
                }
                for name in manifests
            },
            "totals": {
                "datasets": len(manifests),
                "shards_planned": len(planned),
                "shards_downloaded": 0,
                "shards_cached": 0,
                "shards_failed": 0,
                "bytes": 0,
            },
            "results": [],
            "elapsed_s": round(time.time() - t0, 2),
        }

    workers = max(1, workers)
    results: list[ShardDownloadResult] = []
    if workers == 1:
        for source, shard_key in planned:
            results.append(_download_one_shard(source, shard_key, force=force))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {
                pool.submit(_download_one_shard, source, key, force=force): (source, key)
                for source, key in planned
            }
            for fut in as_completed(futs):
                results.append(fut.result())

    failed = [r for r in results if r.status == "failed"]
    if failed:
        sample = failed[0]
        raise RuntimeError(
            f"{len(failed)} shard download(s) failed; first: {sample.dataset} "
            f"{sample.shard_key}: {sample.error}"
        )

    downloaded = sum(1 for r in results if r.status == "downloaded")
    cached = sum(1 for r in results if r.status == "cached")
    total_bytes = sum(r.bytes for r in results)
    return {
        "preset": cfg.preset,
        "weights": cfg.weights,
        "datasets": {
            name: {
                "manifest_url": next(s.manifest_url for s in cfg.sources if s.name == name),
                "manifest_cache": str(next(s.manifest_cache for s in cfg.sources if s.name == name)),
                "manifest_hash": manifest_hash(manifests[name]),
                "shard_count_manifest": len(manifests[name].get("shards") or []),
                "shard_count_downloaded": len(keys_by_dataset.get(name, [])),
                "shard_keys": keys_by_dataset.get(name, []),
            }
            for name in manifests
        },
        "totals": {
            "datasets": len(manifests),
            "shards_planned": len(planned),
            "shards_downloaded": downloaded,
            "shards_cached": cached,
            "shards_failed": len(failed),
            "bytes": total_bytes,
        },
        "results": [r.__dict__ for r in results],
        "elapsed_s": round(time.time() - t0, 2),
    }
