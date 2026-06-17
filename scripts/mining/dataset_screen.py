"""Per-dataset trainability analysis after king scoring.

Ranks mixture sources by data quality / headroom and suggests bucket mix,
weights, and shard prefetch targets before LoRA training.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from curriculum import loss_summary

log = logging.getLogger(__name__)


def _bucket_counts(rows: list[dict]) -> dict[str, int]:
    return {
        b: sum(1 for r in rows if r.get("bucket") == b)
        for b in ("general", "hard", "easy", "suspicious")
    }


def analyze_dataset(
    rows: list[dict],
    *,
    dataset_name: str,
    weight: float,
    train_alloc: int,
    bucket_mix: tuple[float, float, float],
    shards_cached: int = 0,
    shards_manifest: int = 0,
) -> dict:
    """Score one mixture source for trainability before LoRA."""
    counts = _bucket_counts(rows)
    clean_n = counts["general"] + counts["hard"] + counts["easy"]
    suspicious_n = counts["suspicious"]
    losses = np.asarray([r["loss"] for r in rows if r.get("bucket") != "suspicious"])
    stats = loss_summary(losses) if len(losses) else {}

    gf, hf, ef = bucket_mix
    total_frac = max(gf + hf + ef, 1e-9)
    gf, hf, ef = gf / total_frac, hf / total_frac, ef / total_frac

    hard_cap_frac = (counts["hard"] / clean_n) if clean_n else 0.0
    n_hard_want = int(train_alloc * hf) if train_alloc > 0 else 0
    n_hard_avail = counts["hard"]
    hard_saturated = n_hard_want > 0 and n_hard_avail <= n_hard_want

    # Achievable train size if hard/general/easy pools are the limit
    n_general_want = int(train_alloc * gf)
    n_easy_want = train_alloc - n_general_want - n_hard_want
    achievable_train = min(
        train_alloc,
        counts["general"] + counts["hard"] + counts["easy"],
        n_general_want + min(n_hard_want, n_hard_avail) + min(n_easy_want, counts["easy"]),
    )

    suspicious_frac = suspicious_n / len(rows) if rows else 0.0
    loss_spread = float(stats.get("p85", 0) - stats.get("p50", 0)) if stats else 0.0

    # Higher = more worth training on (headroom + supply + diversity)
    shard_factor = min(1.0, shards_cached / 12.0) if shards_cached else 0.2
    trainability = (
        0.35 * min(1.0, hard_cap_frac / 0.15)
        + 0.25 * min(1.0, loss_spread / 0.5)
        + 0.20 * shard_factor
        + 0.10 * min(1.0, clean_n / 1500.0)
        + 0.10 * max(0.0, 1.0 - suspicious_frac * 5.0)
    )

    issues: list[str] = []
    if shards_cached < 4:
        issues.append(f"low_shard_cache={shards_cached}")
    if hard_saturated:
        issues.append(f"hard_saturated({n_hard_avail}/{n_hard_want})")
    if achievable_train < train_alloc * 0.85:
        issues.append(f"train_undersized({achievable_train}/{train_alloc})")
    if suspicious_frac > 0.05:
        issues.append(f"suspicious_frac={suspicious_frac:.3f}")
    if clean_n < 200:
        issues.append(f"small_scored_pool={clean_n}")

    suggested_mix = suggest_bucket_mix_for_dataset(
        counts, train_alloc, bucket_mix,
    )

    return {
        "dataset": dataset_name,
        "weight": weight,
        "n_scored": len(rows),
        "clean_n": clean_n,
        "bucket_counts": counts,
        "loss": stats,
        "loss_spread_p85_p50": loss_spread,
        "bucket_mix_requested": {"general": gf, "hard": hf, "easy": ef},
        "hard_cap_frac": round(hard_cap_frac, 4),
        "hard_saturated": hard_saturated,
        "train_alloc": train_alloc,
        "achievable_train": achievable_train,
        "shards_cached": shards_cached,
        "shards_manifest": shards_manifest,
        "suspicious_frac": round(suspicious_frac, 4),
        "trainability_score": round(trainability, 4),
        "issues": issues,
        "suggested_bucket_mix": suggested_mix,
    }


def suggest_bucket_mix_for_dataset(
    counts: dict[str, int],
    train_alloc: int,
    bucket_mix: tuple[float, float, float],
) -> dict[str, float]:
    """Cap hard_frac to what the scored pool can actually supply."""
    clean_n = counts.get("general", 0) + counts.get("hard", 0) + counts.get("easy", 0)
    if clean_n <= 0 or train_alloc <= 0:
        return {"general": 0.60, "hard": 0.30, "easy": 0.10}

    gf, hf, ef = bucket_mix
    total = max(gf + hf + ef, 1e-9)
    gf, hf, ef = gf / total, hf / total, ef / total

    hard_cap = counts.get("hard", 0) / clean_n
    # Use at most 90% of hard pool; never request more hard than exists
    hf_adj = min(hf, hard_cap * 0.90)
    if hf > 0 and hf_adj < 0.05:
        hf_adj = min(hf, hard_cap)

    remainder = max(0.0, 1.0 - hf_adj)
    if remainder <= 0:
        return {"general": 0.10, "hard": round(hf_adj, 3), "easy": 0.0}

    gf_adj = gf / max(gf + ef, 1e-9) * remainder
    ef_adj = remainder - gf_adj
    return {
        "general": round(gf_adj, 3),
        "hard": round(hf_adj, 3),
        "easy": round(max(ef_adj, 0.0), 3),
    }


def suggest_weights(analyses: list[dict], *, min_weight: float = 0.03) -> dict[str, float]:
    """Reweight mixture toward more trainable datasets (scores must be positive)."""
    scores = {a["dataset"]: max(0.05, float(a["trainability_score"])) for a in analyses}
    total = sum(scores.values()) or 1.0
    raw = {k: v / total for k, v in scores.items()}
    # Blend 50% validator prior + 50% trainability so we don't swing too wild
    priors = {a["dataset"]: float(a["weight"]) for a in analyses}
    blended = {
        k: 0.5 * priors.get(k, 0.0) + 0.5 * raw.get(k, 0.0)
        for k in priors
    }
    t = sum(blended.values()) or 1.0
    out = {k: max(min_weight, v / t) for k, v in blended.items()}
    t2 = sum(out.values())
    return {k: round(v / t2, 4) for k, v in out.items()}


def build_trainability_report(
    rows_by_dataset: dict[str, list[dict]],
    *,
    allocations: dict,
    bucket_mix: dict[str, tuple[float, float, float]],
    shard_stats: dict[str, dict] | None = None,
) -> dict:
    analyses: list[dict] = []
    weights = allocations.get("weights", {})
    train_alloc = allocations.get("train_per_iter", {})
    default_mix = (0.60, 0.30, 0.10)

    for ds_name, rows in rows_by_dataset.items():
        stats = (shard_stats or {}).get(ds_name, {})
        mix = bucket_mix.get(ds_name, default_mix)
        analyses.append(analyze_dataset(
            rows,
            dataset_name=ds_name,
            weight=float(weights.get(ds_name, 0.0)),
            train_alloc=int(train_alloc.get(ds_name, 0)),
            bucket_mix=mix,
            shards_cached=int(stats.get("cached", 0)),
            shards_manifest=int(stats.get("manifest", 0)),
        ))

    analyses.sort(key=lambda a: a["trainability_score"], reverse=True)
    suggested_weights = suggest_weights(analyses)
    suggested_bucket_cli = [
        f"{a['dataset']}={a['suggested_bucket_mix']['general']}:"
        f"{a['suggested_bucket_mix']['hard']}:"
        f"{a['suggested_bucket_mix']['easy']}"
        for a in analyses
    ]

    blockers = []
    for a in analyses:
        if a["trainability_score"] < 0.35 or "low_shard_cache" in str(a["issues"]):
            blockers.append(a["dataset"])

    return {
        "datasets": analyses,
        "ranking": [a["dataset"] for a in analyses],
        "suggested_weights": suggested_weights,
        "suggested_bucket_mix_cli": suggested_bucket_cli,
        "likely_blockers": blockers,
        "recommendation": _recommendation_text(analyses, blockers),
    }


def _recommendation_text(analyses: list[dict], blockers: list[str]) -> str:
    if not analyses:
        return "No scored datasets — run mixture scoring first."
    best = analyses[0]["dataset"]
    lines = [
        f"Most trainable source: {best} (score={analyses[0]['trainability_score']}).",
    ]
    if blockers:
        lines.append(
            f"Prefetch more shards for: {', '.join(blockers)} "
            f"(drop --local-shards-only or raise --mix-shards-per-dataset).",
        )
    saturated = [a["dataset"] for a in analyses if a.get("hard_saturated")]
    if saturated:
        lines.append(
            f"Hard bucket saturated on {', '.join(saturated)} — "
            "raising --hard-frac will not add new sequences; get more shards.",
        )
    return " ".join(lines)


def shard_cache_stats_from_store(store) -> dict[str, dict]:
    from dataset_mixture import is_shard_cached

    out: dict[str, dict] = {}
    for source in store.cfg.sources:
        entries = store._shard_meta.get(source.name, [])
        manifest_n = len(entries)
        cached_n = sum(1 for key, _ in entries if is_shard_cached(source, key))
        out[source.name] = {"cached": cached_n, "manifest": manifest_n}
    return out


def write_trainability_report(work: Path, report: dict) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    path = work / "dataset_trainability.json"
    path.write_text(json.dumps(report, indent=2))
    log.info("dataset trainability report → %s", path)
    for ds in report.get("datasets", []):
        log.info(
            "  trainability %s: score=%.3f hard_cap=%.2f achievable_train=%d issues=%s",
            ds["dataset"], ds["trainability_score"], ds["hard_cap_frac"],
            ds["achievable_train"], ds.get("issues") or [],
        )
    if report.get("recommendation"):
        log.info("  → %s", report["recommendation"])
    return path


def format_next_train_cli(work: Path, report: dict, *, base_flags: str = "") -> str:
    """Human-readable CLI snippet for a follow-up train run."""
    bucket_flags = "\n  ".join(
        f'--bucket-mix {spec}' for spec in report.get("suggested_bucket_mix_cli", [])
    )
    top = report.get("ranking", [])[:2]
    lora_hint = (
        '--lora-sweep "r64:a64:lr5e-5:d0.0:e1.0"'
        if top else ""
    )
    return (
        f"python -u scripts/mining/train_challenger.py \\\n"
        f"  --work {work} \\\n"
        f"  --mode probe \\\n"
        f"  --skip-scoring \\\n"
        f"  {bucket_flags} \\\n"
        f"  {lora_hint} \\\n"
        f"  # focus datasets: {', '.join(top)}"
    )
