"""Weighted multi-manifest pretokenized dataset mixing for train_challenger.

Samples from v4 Quasar Hippius manifests without loading full corpora into RAM.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import struct
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger("dataset_mix")

DEFAULT_HIPPIUS_BASE = os.environ.get(
    "TEUTONIC_HIPPIUS_HTTP_BASE", "https://s3.hippius.com/teutonic-sn3",
).rstrip("/")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    manifest_url: str
    weight: float


@dataclass(frozen=True)
class MixConfig:
    seq_len: int
    hippius_base: str
    datasets: tuple[DatasetSpec, ...]


@dataclass(frozen=True)
class SequenceRef:
    dataset: str
    shard_key: str
    row_idx: int


def parse_npy_header(raw: bytes) -> tuple[int, dict]:
    buf = io.BytesIO(raw)
    if buf.read(6) != b"\x93NUMPY":
        raise ValueError("not a .npy file")
    ver = struct.unpack("BB", buf.read(2))
    hl = struct.unpack("<H" if ver[0] == 1 else "<I",
                       buf.read(2 if ver[0] == 1 else 4))[0]
    header = eval(buf.read(hl).decode("latin1").strip())
    return buf.tell(), header


def load_shard_array(path: Path, seq_len: int) -> np.ndarray:
    raw = path.read_bytes()
    data_offset, header = parse_npy_header(raw)
    shape = header["shape"]
    flat = np.frombuffer(raw[data_offset:], dtype="<u4")
    if len(shape) == 1:
        n_seq = int(shape[0]) // seq_len
        return flat[: n_seq * seq_len].reshape(n_seq, seq_len)
    if len(shape) == 2:
        return flat.reshape(shape[0], shape[1])
    raise ValueError(f"unexpected shard shape {shape}")


def shard_num_sequences(entry: dict, seq_len: int) -> int:
    if int(entry.get("n_samples") or 0) > 0:
        n = int(entry["n_samples"])
        if int(entry.get("n_tokens") or 0) > 0:
            expected = n * seq_len
            if abs(int(entry["n_tokens"]) - expected) <= seq_len:
                return n
    n_tokens = int(entry.get("n_tokens") or 0)
    if n_tokens <= 0:
        return 0
    return n_tokens // seq_len


def fetch_manifest_json(url: str, cache_dir: Path) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(url.encode()).hexdigest()[:16]
    path = cache_dir / f"manifest_{key}.json"
    if not path.is_file():
        log.info("fetching manifest %s", url)
        req = urllib.request.Request(url, headers={"User-Agent": "teutonic-train/1"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            path.write_bytes(resp.read())
    return json.loads(path.read_text())


def load_mix_config(path: str | Path) -> MixConfig:
    raw = json.loads(Path(path).read_text())
    seq_len = int(raw.get("seq_len") or 2048)
    hippius_base = (raw.get("hippius_base") or DEFAULT_HIPPIUS_BASE).rstrip("/")
    datasets: list[DatasetSpec] = []
    for item in raw.get("datasets") or []:
        name = str(item.get("name") or "").strip()
        url = str(item.get("manifest_url") or item.get("url") or "").strip()
        weight = float(item.get("weight") or 0.0)
        if not name or not url or weight <= 0:
            continue
        datasets.append(DatasetSpec(name=name, manifest_url=url, weight=weight))
    if not datasets:
        raise ValueError(f"no datasets in mix config {path}")
    total_w = sum(d.weight for d in datasets)
    norm = tuple(
        DatasetSpec(d.name, d.manifest_url, d.weight / total_w) for d in datasets
    )
    return MixConfig(seq_len=seq_len, hippius_base=hippius_base, datasets=norm)


def mix_config_hash(cfg: MixConfig) -> str:
    payload = {
        "seq_len": cfg.seq_len,
        "datasets": [
            {"name": d.name, "url": d.manifest_url, "weight": d.weight}
            for d in cfg.datasets
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode(),
    ).hexdigest()


class MixedDatasetIndex:
    """Weighted dataset → shard → row sampling over v4 manifests."""

    def __init__(self, cfg: MixConfig, cache_dir: Path):
        self.cfg = cfg
        self.cache_dir = cache_dir
        self.manifest_cache = cache_dir / "manifests"
        self.names: list[str] = []
        self.weights = np.asarray([], dtype=np.float64)
        self.shard_keys: list[list[str]] = []
        self.shard_rows: list[list[int]] = []

        for spec in cfg.datasets:
            manifest = fetch_manifest_json(spec.manifest_url, self.manifest_cache)
            entries = manifest.get("shards") or []
            keys: list[str] = []
            rows: list[int] = []
            for entry in entries:
                key = str(entry.get("key") or "").strip()
                if not key:
                    continue
                n_seq = shard_num_sequences(entry, cfg.seq_len)
                if n_seq <= 0:
                    continue
                keys.append(key)
                rows.append(n_seq)
            if not keys:
                raise RuntimeError(f"dataset {spec.name} has no usable shards")
            self.names.append(spec.name)
            self.weights = np.append(self.weights, spec.weight)
            self.shard_keys.append(keys)
            self.shard_rows.append(rows)
            total_seq = int(sum(rows))
            log.info(
                "mix dataset %s: weight=%.2f shards=%d sequences≈%s",
                spec.name, spec.weight, len(keys), f"{total_seq:,}",
            )
        self.weights /= self.weights.sum()

    def select_shard_pool(
        self, seed: int, shards_per_dataset: int,
    ) -> list[list[tuple[str, int]]]:
        rng = np.random.default_rng(seed)
        pools: list[list[tuple[str, int]]] = []
        for ds_i, name in enumerate(self.names):
            keys = self.shard_keys[ds_i]
            rows = self.shard_rows[ds_i]
            n_pick = min(max(1, shards_per_dataset), len(keys))
            pick = rng.choice(len(keys), size=n_pick, replace=False)
            pool = [(keys[int(j)], rows[int(j)]) for j in pick]
            pools.append(pool)
            log.info(
                "mix pool %s: %d/%d shard(s) selected for scoring",
                name, n_pick, len(keys),
            )
        return pools

    def sample_refs(
        self,
        n: int,
        seed: int,
        shards_per_dataset: int = 12,
    ) -> tuple[list[SequenceRef], list[list[tuple[str, int]]]]:
        rng = np.random.default_rng(seed)
        pools = self.select_shard_pool(seed, shards_per_dataset)
        refs: list[SequenceRef] = []
        for _ in range(n):
            ds_i = int(rng.choice(len(self.names), p=self.weights))
            shard_key, n_rows = pools[ds_i][int(rng.integers(0, len(pools[ds_i])))]
            row_i = int(rng.integers(0, n_rows))
            refs.append(SequenceRef(
                dataset=self.names[ds_i],
                shard_key=shard_key,
                row_idx=row_i,
            ))
        unique_keys = {r.shard_key for r in refs}
        log.info(
            "mix sample: %d sequence refs from %d unique shard(s) "
            "(pool cap %d/dataset)",
            n, len(unique_keys), shards_per_dataset,
        )
        return refs, pools


class ShardStore:
    """Lazy shard loader keyed by manifest shard path."""

    def __init__(self, cache_dir: Path, hippius_base: str, seq_len: int):
        self.cache_dir = cache_dir
        self.hippius_base = hippius_base.rstrip("/")
        self.seq_len = seq_len
        self._arrays: dict[str, np.ndarray] = {}
        self._local_keys: list[str] = []

    @property
    def arrays(self) -> list[np.ndarray]:
        return [self._arrays[k] for k in self._local_keys]

    @property
    def keys(self) -> list[str]:
        return list(self._local_keys)

    def local_index(self, shard_key: str) -> int:
        if shard_key not in self._arrays:
            self._load(shard_key)
        return self._local_keys.index(shard_key)

    def ensure_loaded(self, shard_keys: set[str]) -> None:
        for key in sorted(shard_keys):
            if key not in self._arrays:
                self._load(key)

    def _load(self, shard_key: str) -> None:
        local_path = Path(shard_key)
        if local_path.is_file():
            path = local_path
        else:
            out = self.cache_dir / "shards" / Path(shard_key).name
            if out.is_file() and out.stat().st_size > 1024:
                path = out
            else:
                url = f"{self.hippius_base}/{shard_key.lstrip('/')}"
                log.info("downloading shard %s", shard_key.rsplit("/", 1)[-1])
                out.parent.mkdir(parents=True, exist_ok=True)
                subprocess.check_call(["curl", "-fsSL", "-o", str(out), url])
                path = out
        arr = load_shard_array(path, self.seq_len)
        self._arrays[shard_key] = arr
        self._local_keys.append(shard_key)
        log.info("loaded shard %s (%d seq)", shard_key.rsplit("/", 1)[-1], len(arr))


def refs_to_candidates(
    refs: list[SequenceRef], store: ShardStore,
) -> list[tuple[int, int]]:
    store.ensure_loaded({r.shard_key for r in refs})
    return [(store.local_index(r.shard_key), r.row_idx) for r in refs]


def synthetic_manifest(cfg: MixConfig) -> dict:
    return {
        "version": "mixed-v4",
        "tokenizer": "silx-ai/Quasar-10B",
        "seq_len": cfg.seq_len,
        "datasets": [
            {"name": d.name, "manifest_url": d.manifest_url, "weight": d.weight}
            for d in cfg.datasets
        ],
    }
