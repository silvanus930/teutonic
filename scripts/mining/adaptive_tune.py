"""Adaptive hyperparameter search for Teutonic LoRA training.

Generates small candidate sets per round from prior results, decides when to
escalate from micro-screen → confirm → strong training.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import re
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

SWEEP_PATTERN = re.compile(
    r"^r(\d+):a(\d+):lr([\de.\-+]+)(?::d([\de.\-+]+))?(?::e([\de.\-+]+))?$",
    re.IGNORECASE,
)

# Empirical crown band from on-chain dashboard (Jun 2026).
DEFAULT_CROWN_MU = 0.004
DEFAULT_CROWN_LCB = 0.003

SEED_SPECS: list[str] = [
    "r64:a64:lr5e-5:d0.0:e0.5",
    "r64:a128:lr5e-5:d0.0:e0.5",
    "r32:a64:lr5e-5:d0.0:e0.5",
    "r64:a64:lr8e-5:d0.0:e0.5",
    "r64:a64:lr3e-5:d0.0:e0.5",
]


@dataclasses.dataclass
class HyperparamConfig:
    lora_r: int
    lora_alpha: int
    lr: float
    lora_dropout: float = 0.0
    epochs: float = 0.5

    @property
    def label(self) -> str:
        return format_sweep_spec(self).replace(":", "_").lower()

    def to_spec(self) -> str:
        return format_sweep_spec(self)


def format_sweep_spec(
    cfg: HyperparamConfig,
    *,
    epochs: float | None = None,
) -> str:
    e = cfg.epochs if epochs is None else epochs
    return (
        f"r{cfg.lora_r}:a{cfg.lora_alpha}:lr{cfg.lr:g}"
        f":d{cfg.lora_dropout:g}:e{e:g}"
    )


def parse_sweep_spec(spec: str) -> HyperparamConfig:
    m = SWEEP_PATTERN.match(spec.strip())
    if not m:
        raise ValueError(f"invalid sweep spec: {spec!r}")
    return HyperparamConfig(
        lora_r=int(m.group(1)),
        lora_alpha=int(m.group(2)),
        lr=float(m.group(3)),
        lora_dropout=float(m.group(4)) if m.group(4) is not None else 0.0,
        epochs=float(m.group(5)) if m.group(5) is not None else 0.5,
    )


def rank_key(entry: dict) -> tuple[float, float, float]:
    """Higher is better: LCB, mu_hat, negative eval_loss."""
    return (
        float(entry.get("lcb") if entry.get("lcb") is not None else -999),
        float(entry.get("mu_hat") if entry.get("mu_hat") is not None else -999),
        -float(entry.get("eval_loss") if entry.get("eval_loss") is not None else 999),
    )


def best_of(results: list[dict]) -> dict | None:
    if not results:
        return None
    return max(results, key=rank_key)


@dataclasses.dataclass
class AdaptiveState:
    version: int = 1
    round: int = 0
    crown_mu: float = DEFAULT_CROWN_MU
    crown_lcb: float = DEFAULT_CROWN_LCB
    best: dict | None = None
    production_ready: bool = False
    history: list[dict] = dataclasses.field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> AdaptiveState:
        if not path.is_file():
            return cls()
        raw = json.loads(path.read_text())
        return cls(
            version=raw.get("version", 1),
            round=int(raw.get("round", 0)),
            crown_mu=float(raw.get("crown_mu", DEFAULT_CROWN_MU)),
            crown_lcb=float(raw.get("crown_lcb", DEFAULT_CROWN_LCB)),
            best=raw.get("best"),
            production_ready=bool(raw.get("production_ready")),
            history=list(raw.get("history") or []),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dataclasses.asdict(self), indent=2, default=str))

    def record_round(
        self,
        *,
        phase: str,
        results: list[dict],
        specs: list[str],
    ) -> dict | None:
        top = best_of(results)
        entry = {
            "round": self.round,
            "phase": phase,
            "specs": specs,
            "best": top,
            "n_results": len(results),
        }
        self.history.append(entry)
        if top is not None:
            if self.best is None or rank_key(top) > rank_key(self.best):
                self.best = {
                    **top,
                    "phase": phase,
                    "round": self.round,
                    "config": top.get("config") or top.get("label"),
                }
        return top

    def is_production_ready(self) -> bool:
        if not self.best:
            return False
        mu = float(self.best.get("mu_hat") or -999)
        lcb = float(self.best.get("lcb") or -999)
        phase = self.best.get("phase", "")
        # Only trust confirm/strong phases for production gate.
        if phase not in ("confirm", "strong"):
            return False
        return mu >= self.crown_mu and lcb >= self.crown_lcb


def _dedupe_specs(specs: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in specs:
        key = s.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(s.strip())
    return out


def _around(cfg: HyperparamConfig, lrs: list[float], epochs: list[float]) -> list[str]:
    specs: list[str] = []
    for lr in lrs:
        for e in epochs:
            specs.append(format_sweep_spec(HyperparamConfig(
                cfg.lora_r, cfg.lora_alpha, lr, cfg.lora_dropout, e,
            )))
    return specs


def _config_from_result(best: dict) -> HyperparamConfig:
    if best.get("lora_r") is not None:
        return HyperparamConfig(
            lora_r=int(best["lora_r"]),
            lora_alpha=int(best["lora_alpha"]),
            lr=float(best["lr"]),
            lora_dropout=float(best.get("lora_dropout") or 0.0),
            epochs=float(best.get("epochs") or 0.5),
        )
    spec = best.get("config") or best.get("label", "")
    if not spec:
        return parse_sweep_spec(SEED_SPECS[0])
    if "_" in spec and ":" not in spec:
        parts = spec.split("_")
        try:
            return HyperparamConfig(
                lora_r=int(parts[0][1:]),
                lora_alpha=int(parts[1][1:]),
                lr=float(parts[2].replace("lr", "")),
                lora_dropout=float(parts[3].replace("d", "")) if len(parts) > 3 else 0.0,
                epochs=float(parts[4].replace("e", "")) if len(parts) > 4 else 0.5,
            )
        except (ValueError, IndexError):
            return parse_sweep_spec(SEED_SPECS[0])
    return parse_sweep_spec(spec.replace("_", ":") if ":" not in spec else spec)


def generate_micro_candidates(
    state: AdaptiveState,
    *,
    max_candidates: int = 4,
) -> list[str]:
    """Build a small sweep list for the next micro-screen round."""
    if state.round == 0 and not state.best:
        return _dedupe_specs(SEED_SPECS)[:max_candidates]

    if not state.best:
        return _dedupe_specs(SEED_SPECS)[:max_candidates]

    cfg = _config_from_result(state.best)
    mu = float(state.best.get("mu_hat") or -1)
    specs: list[str] = []

    if mu < -0.001:
        specs.extend(_around(cfg, [cfg.lr * 0.6, cfg.lr * 0.8, cfg.lr], [0.5]))
        if cfg.lora_r == 64:
            specs.append(format_sweep_spec(HyperparamConfig(32, 64, cfg.lr, cfg.lora_dropout, 0.5)))
            specs.append(format_sweep_spec(HyperparamConfig(64, 128, cfg.lr * 0.8, cfg.lora_dropout, 0.5)))
        elif cfg.lora_r == 32:
            specs.append(format_sweep_spec(HyperparamConfig(64, 64, cfg.lr, cfg.lora_dropout, 0.5)))
    elif mu < 0:
        specs.extend(_around(
            cfg,
            [cfg.lr * 0.85, cfg.lr, cfg.lr * 1.15],
            [0.5, 0.8],
        ))
    else:
        specs.extend(_around(
            cfg,
            [cfg.lr * 0.9, cfg.lr, cfg.lr * 1.1],
            [0.8, 1.0],
        ))

    specs.insert(0, cfg.to_spec())
    return _dedupe_specs(specs)[:max_candidates]


def should_skip_confirm(micro_best: dict | None, *, min_mu: float = -0.002) -> bool:
    """Skip expensive confirm train when micro-screen is clearly hopeless."""
    if micro_best is None:
        return True
    mu = float(micro_best.get("mu_hat") or -999)
    return mu < min_mu


def confirm_specs_from_micro(
    micro_best: dict,
    *,
    top_k: int = 1,
    epochs_override: float | None = None,
) -> list[str]:
    """Pick 1–2 specs for full probe-length training."""
    if micro_best.get("lora_r") is not None:
        cfg = HyperparamConfig(
            lora_r=int(micro_best["lora_r"]),
            lora_alpha=int(micro_best["lora_alpha"]),
            lr=float(micro_best["lr"]),
            lora_dropout=float(micro_best.get("lora_dropout") or 0.0),
            epochs=float(micro_best.get("epochs") or 0.5),
        )
    else:
        cfg = _config_from_result(micro_best)

    mu = float(micro_best.get("mu_hat") or -1)
    if epochs_override is not None:
        e = epochs_override
    elif mu >= 0:
        e = max(cfg.epochs, 0.8)
    else:
        e = max(cfg.epochs, 0.5)

    specs = [format_sweep_spec(cfg, epochs=e)]
    if top_k > 1:
        specs.append(format_sweep_spec(cfg, epochs=min(e * 1.2, 1.5)))
    return _dedupe_specs(specs)[:top_k]


def strong_spec_from_best(best: dict) -> str:
    """Production strong-training spec (longer epochs, same core hparams)."""
    cfg = _config_from_result(best)
    return format_sweep_spec(cfg, epochs=max(cfg.epochs, 1.0))


def write_production_config(
    work: Path,
    state: AdaptiveState,
    *,
    strong_spec: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    out = work / "production_config.json"
    payload = {
        "production_ready": state.is_production_ready(),
        "strong_lora_sweep": strong_spec,
        "best": state.best,
        "crown_mu": state.crown_mu,
        "crown_lcb": state.crown_lcb,
        "rounds_completed": state.round,
        "recommendation": (
            "Run strong training with strong_lora_sweep."
            if state.is_production_ready()
            else "Best config found but below crown targets — strong run is optional."
        ),
    }
    if extra:
        payload.update(extra)
    out.write_text(json.dumps(payload, indent=2, default=str))
    log.info("production config → %s (ready=%s)", out, payload["production_ready"])
    return out


def count_cached_shards(cache_root: Path) -> dict[str, int]:
    shards_root = cache_root / "shards"
    counts: dict[str, int] = {}
    if not shards_root.is_dir():
        return counts
    for ds_dir in sorted(shards_root.iterdir()):
        if ds_dir.is_dir():
            counts[ds_dir.name] = sum(1 for _ in ds_dir.glob("*.npy"))
    return counts


def shards_need_prefetch(
    counts: dict[str, int],
    *,
    min_per_dataset: int = 24,
) -> tuple[bool, list[str]]:
    low = [name for name, n in counts.items() if n < min_per_dataset]
    return bool(low), low
