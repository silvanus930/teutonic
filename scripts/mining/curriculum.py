"""Curriculum builder for king-scored training samples."""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_GENERAL_FRAC = 0.60
DEFAULT_HARD_FRAC = 0.30
DEFAULT_EASY_FRAC = 0.10


def loss_summary(losses: np.ndarray) -> dict:
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


def bucket_means(rows: list[dict], key: str = "loss") -> dict[str, float]:
    out: dict[str, float] = {}
    for bucket in ("general", "hard", "easy", "suspicious"):
        vals = [r[key] for r in rows if r.get("bucket") == bucket]
        if vals:
            out[bucket] = float(np.mean(vals))
    return out


def assign_buckets(rows: list[dict]) -> tuple[dict, float, float]:
    losses = np.asarray([r["loss"] for r in rows])
    stats = loss_summary(losses)
    p50 = stats.get("p50", float(np.median(losses)))
    p85 = stats.get("p85", float(np.percentile(losses, 85)))
    general_floor = p50 * 0.8

    def _bucket(row: dict) -> str:
        if row.get("rep_r", 0) > 0.2 or row.get("rep_ng4", 0) > 0.5 or row.get("unique_r", 1) < 0.05:
            return "suspicious"
        if math.isnan(row["loss"]) or math.isinf(row["loss"]):
            return "suspicious"
        if row["loss"] >= p85:
            return "hard"
        if row["loss"] >= general_floor:
            return "general"
        return "easy"

    for row in rows:
        if not row.get("bucket"):
            row["bucket"] = _bucket(row)

    counts = {b: sum(1 for r in rows if r.get("bucket") == b)
              for b in ("general", "hard", "easy", "suspicious")}
    return counts, p50, p85


def build_curriculum(
    rows: list[dict],
    *,
    train_per_iter: int,
    val_size: int,
    seed: int,
    general_frac: float = DEFAULT_GENERAL_FRAC,
    hard_frac: float = DEFAULT_HARD_FRAC,
    easy_frac: float = DEFAULT_EASY_FRAC,
    max_suspicious_frac: float = 0.0,
) -> tuple[list[dict], list[dict], dict]:
    """Split scored rows into train/val with bucket mix; train and val never overlap."""
    counts, p50, p85 = assign_buckets(rows)
    clean = [r for r in rows if r.get("bucket") != "suspicious"]
    dropped = len(rows) - len(clean)

    if max_suspicious_frac > 0 and rows:
        susp_frac = dropped / len(rows)
        if susp_frac > max_suspicious_frac:
            log.warning(
                "suspicious fraction %.1f%% exceeds max %.1f%%",
                100 * susp_frac, 100 * max_suspicious_frac,
            )

    rng = np.random.default_rng(seed + 1)
    order = rng.permutation(len(clean)).tolist()
    clean_shuffled = [clean[i] for i in order]
    val_rows = clean_shuffled[:val_size]
    val_keys = {(r["shard"], r["idx"]) for r in val_rows}
    pool = [r for r in clean_shuffled if (r["shard"], r["idx"]) not in val_keys]

    total_frac = general_frac + hard_frac + easy_frac
    if total_frac > 0:
        gf = general_frac / total_frac
        hf = hard_frac / total_frac
        ef = easy_frac / total_frac
    else:
        gf, hf, ef = DEFAULT_GENERAL_FRAC, DEFAULT_HARD_FRAC, DEFAULT_EASY_FRAC

    general = [r for r in pool if r.get("bucket") == "general"]
    hard = [r for r in pool if r.get("bucket") == "hard"]
    easy = [r for r in pool if r.get("bucket") == "easy"]
    n_general = int(train_per_iter * gf)
    n_hard = int(train_per_iter * hf)
    n_easy = train_per_iter - n_general - n_hard

    train_rows: list[dict] = []
    picked: dict[str, int] = {}
    for label, src, n in (("general", general, n_general),
                          ("hard", hard, n_hard),
                          ("easy", easy, n_easy)):
        if not src:
            picked[label] = 0
            continue
        if n >= len(src):
            train_rows.extend(src)
            picked[label] = len(src)
        else:
            sel = rng.choice(len(src), size=n, replace=False)
            train_rows.extend(src[int(k)] for k in sel)
            picked[label] = n

    order2 = rng.permutation(len(train_rows)).tolist()
    train_rows = [train_rows[i] for i in order2]
    train_keys = {(r["shard"], r["idx"]) for r in train_rows}
    overlap = train_keys & val_keys
    if overlap:
        raise RuntimeError(f"train/val overlap detected: {len(overlap)} samples")

    train_mix = {b: sum(1 for r in train_rows if r.get("bucket") == b)
                 for b in ("general", "hard", "easy")}
    losses = np.asarray([r["loss"] for r in rows])
    summary = {
        "seed": seed + 1,
        "train_per_iter": train_per_iter,
        "val_size": val_size,
        "general_frac": general_frac,
        "hard_frac": hard_frac,
        "easy_frac": easy_frac,
        "suspicious_dropped": dropped,
        "bucket_counts": counts,
        "picked": picked,
        "train_mix": train_mix,
        "train_n": len(train_rows),
        "val_n": len(val_rows),
        "train_val_overlap": 0,
        "thresholds": {"p50": p50, "p85": p85, "general_floor": p50 * 0.8},
        "train_loss": loss_summary(np.asarray([r["loss"] for r in train_rows])) if train_rows else {},
        "val_loss": loss_summary(np.asarray([r["loss"] for r in val_rows])) if val_rows else {},
        "val_bucket_counts": {b: sum(1 for r in val_rows if r.get("bucket") == b)
                              for b in ("general", "hard", "easy")},
    }
    return train_rows, val_rows, summary


def build_mixture_curriculum(
    rows_by_dataset: dict[str, list[dict]],
    *,
    train_alloc: dict[str, int],
    val_alloc: dict[str, int],
    seed: int,
    bucket_mix: dict[str, tuple[float, float, float]] | None = None,
    default_mix: tuple[float, float, float] = (DEFAULT_GENERAL_FRAC, DEFAULT_HARD_FRAC, DEFAULT_EASY_FRAC),
    max_suspicious_frac: float = 0.0,
) -> tuple[list[dict], list[dict], dict]:
    """Build per-dataset curricula, merge, and guarantee global train/val non-overlap."""
    all_train: list[dict] = []
    all_val: list[dict] = []
    per_dataset: dict[str, dict] = {}
    global_val_keys: set[tuple] = set()

    for ds_name, rows in rows_by_dataset.items():
        n_train = train_alloc.get(ds_name, 0)
        n_val = val_alloc.get(ds_name, 0)
        if n_train <= 0 and n_val <= 0:
            continue
        gf, hf, ef = bucket_mix.get(ds_name, default_mix) if bucket_mix else default_mix
        train_rows, val_rows, summary = build_curriculum(
            rows,
            train_per_iter=max(n_train, 1) if n_train > 0 else 1,
            val_size=max(n_val, 1) if n_val > 0 else 1,
            seed=seed + hash(ds_name) % 10000,
            general_frac=gf,
            hard_frac=hf,
            easy_frac=ef,
            max_suspicious_frac=max_suspicious_frac,
        )
        if n_train > 0:
            if len(train_rows) > n_train:
                rng = np.random.default_rng(seed + 7)
                sel = rng.choice(len(train_rows), size=n_train, replace=False)
                train_rows = [train_rows[int(i)] for i in sel]
            elif len(train_rows) < n_train:
                log.warning(
                    "dataset %s: train undersized %d < %d",
                    ds_name, len(train_rows), n_train,
                )
        else:
            train_rows = []

        if n_val > 0:
            if len(val_rows) > n_val:
                val_rows = val_rows[:n_val]
            elif len(val_rows) < n_val:
                log.warning(
                    "dataset %s: val undersized %d < %d",
                    ds_name, len(val_rows), n_val,
                )
        else:
            val_rows = []

        for row in train_rows:
            row["dataset"] = ds_name
        for row in val_rows:
            row["dataset"] = ds_name

        ds_val_keys = {(r.get("dataset", ds_name), r["shard"], r["idx"]) for r in val_rows}
        overlap = ds_val_keys & global_val_keys
        if overlap:
            raise RuntimeError(f"cross-dataset val overlap for {ds_name}: {len(overlap)}")
        global_val_keys |= ds_val_keys

        train_keys = {(r["shard"], r["idx"]) for r in train_rows}
        val_keys = {(r["shard"], r["idx"]) for r in val_rows}
        if train_keys & val_keys:
            raise RuntimeError(f"{ds_name}: train/val overlap {len(train_keys & val_keys)}")

        all_train.extend(train_rows)
        all_val.extend(val_rows)
        per_dataset[ds_name] = summary

    rng = np.random.default_rng(seed + 99)
    if all_train:
        order = rng.permutation(len(all_train)).tolist()
        all_train = [all_train[i] for i in order]
    if all_val:
        order = rng.permutation(len(all_val)).tolist()
        all_val = [all_val[i] for i in order]

    merged_summary = {
        "mode": "mixture",
        "train_n": len(all_train),
        "val_n": len(all_val),
        "per_dataset": per_dataset,
        "train_alloc": train_alloc,
        "val_alloc": val_alloc,
        "train_mix": {
            b: sum(1 for r in all_train if r.get("bucket") == b)
            for b in ("general", "hard", "easy")
        },
        "val_mix": {
            b: sum(1 for r in all_val if r.get("bucket") == b)
            for b in ("general", "hard", "easy")
        },
    }
    return all_train, all_val, merged_summary


def write_curriculum_jsonl(
    train_rows: list[dict],
    val_rows: list[dict],
    work: Path,
    shards: list[np.ndarray] | None = None,
) -> tuple[Path, Path]:
    """Write train.jsonl and val.jsonl using cached input_ids or shard arrays."""
    work.mkdir(parents=True, exist_ok=True)
    train_p = work / "train.jsonl"
    val_p = work / "val.jsonl"

    def _input_ids(row: dict) -> list[int]:
        if row.get("input_ids") is not None:
            return row["input_ids"]
        if shards is None:
            raise ValueError("row has no input_ids and shards not provided")
        return shards[row["shard"]][row["idx"]].tolist()

    with open(train_p, "w") as f:
        for row in train_rows:
            f.write(json.dumps({"input_ids": _input_ids(row)}) + "\n")
    with open(val_p, "w") as f:
        for row in val_rows:
            f.write(json.dumps({"input_ids": _input_ids(row)}) + "\n")
    return train_p, val_p


def save_curriculum_reports(
    work: Path,
    rows: list[dict],
    summary: dict,
    *,
    seed: int,
    n_candidates: int,
    shards_used: list[str],
    shards_used_local_indices: list[int] | None = None,
    per_dataset_stats: dict | None = None,
) -> None:
    losses = np.asarray([r["loss"] for r in rows])
    scoring_report = {
        "seed": seed,
        "n_candidates": n_candidates,
        "loss": loss_summary(losses),
        "bucket_counts": summary.get("bucket_counts", {}),
        "bucket_mean_loss": bucket_means(rows),
        "thresholds": summary.get("thresholds", {}),
    }
    curriculum_report = dict(summary)
    curriculum_report["shards_used"] = shards_used
    if shards_used_local_indices is not None:
        curriculum_report["shards_used_local_indices"] = shards_used_local_indices
    if per_dataset_stats is not None:
        curriculum_report["per_dataset"] = per_dataset_stats

    (work / "scoring.json").write_text(json.dumps(scoring_report, indent=2))
    (work / "curriculum.json").write_text(json.dumps(curriculum_report, indent=2))

    with open(work / "scored.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps({
                "shard": row.get("shard"),
                "idx": row.get("idx"),
                "dataset": row.get("dataset"),
                "loss": row.get("loss"),
                "unique_r": row.get("unique_r"),
                "rep_r": row.get("rep_r"),
                "rep_ng4": row.get("rep_ng4"),
                "bucket": row.get("bucket", ""),
                "has_input_ids": row.get("input_ids") is not None,
            }) + "\n")
