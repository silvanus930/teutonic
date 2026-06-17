"""Live status tracking for the adaptive training pipeline.

Writes atomically-updated files under <work>/:
  status.json  — machine-readable snapshot
  status.txt   — human-readable one-screen summary

Tail the log or run show_pipeline_status.py anytime.
"""
from __future__ import annotations

import dataclasses
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# train_challenger / adaptive log patterns
_RE_EVAL_PROGRESS = re.compile(
    r"eval\s+(\d+)/(\d+)\s*\|\s*mu_hat=([-\d.eE+]+)",
)
_RE_MICRO = re.compile(r"micro-screen[:\s]+([^\s:]+)", re.I)
_RE_SWEEP = re.compile(r"=== LoRA sweep:\s*(\S+)", re.I)
_RE_SCORING = re.compile(r"mixture scoring (\S+):\s*n=(\d+)", re.I)
_RE_SCORE_BATCH = re.compile(
    r"mixture scoring (\S+)\s+(\d+)/(\d+)\s*\(([\d.]+)%\)",
)
_RE_TRAIN_STEP = re.compile(r"(\d+)%\|.*?\[.*?(\d+)/(\d+)")
_RE_LOSS = re.compile(r"'loss':\s*'([\d.]+)'.*'epoch':\s*'([\d.]+)'")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _read_tail(path: Path, n: int = 40) -> list[str]:
    if not path.is_file():
        return []
    try:
        lines = path.read_text(errors="replace").splitlines()
        return lines[-n:]
    except OSError:
        return []


def parse_log_progress(lines: list[str]) -> dict[str, Any]:
    """Extract latest progress hints from log tail."""
    progress: dict[str, Any] = {}
    for line in reversed(lines):
        m = _RE_EVAL_PROGRESS.search(line)
        if m and "eval_done" not in progress:
            progress["eval_done"] = int(m.group(1))
            progress["eval_total"] = int(m.group(2))
            progress["mu_hat_live"] = float(m.group(3))
            progress["activity"] = "paired_eval"
        m = _RE_SCORE_BATCH.search(line)
        if m and "scoring_dataset" not in progress:
            progress["scoring_dataset"] = m.group(1)
            progress["scoring_done"] = int(m.group(2))
            progress["scoring_total"] = int(m.group(3))
            progress["scoring_pct"] = float(m.group(4))
            progress["activity"] = "king_scoring"
        m = _RE_TRAIN_STEP.search(line)
        if m and "train_pct" not in progress:
            progress["train_pct"] = float(m.group(1))
            progress["train_step"] = int(m.group(2))
            progress["train_steps"] = int(m.group(3))
            progress["activity"] = "lora_training"
        m = _RE_LOSS.search(line)
        if m and "train_loss" not in progress:
            progress["train_loss"] = float(m.group(1))
            progress["train_epoch"] = float(m.group(2))
        m = _RE_MICRO.search(line)
        if m and "config" not in progress:
            progress["config"] = m.group(1)
        m = _RE_SWEEP.search(line)
        if m:
            progress["config"] = m.group(1)
        m = _RE_SCORING.search(line)
        if m and "scoring_dataset" not in progress:
            progress["scoring_dataset"] = m.group(1)
            progress["scoring_candidates"] = int(m.group(2))
            progress["activity"] = "king_scoring"
    return progress


@dataclasses.dataclass
class PipelineStatus:
    pipeline: str = "adaptive"
    state: str = "starting"  # starting|running|done|failed|stopped
    phase: str = "init"
    round: int = 0
    max_rounds: int = 0
    message: str = ""
    started_at: str = ""
    updated_at: str = ""
    phase_started_at: str = ""
    elapsed_s: float = 0.0
    phase_elapsed_s: float = 0.0
    current_job: str = ""
    progress: dict[str, Any] = dataclasses.field(default_factory=dict)
    best: dict[str, Any] | None = None
    crown_mu: float = 0.004
    crown_lcb: float = 0.003
    production_ready: bool = False
    shard_cache: dict[str, int] = dataclasses.field(default_factory=dict)
    log_path: str = ""
    status_path: str = ""
    recent_log: list[str] = dataclasses.field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def format_text(self) -> str:
        lines = [
            "═" * 60,
            f"  ADAPTIVE PIPELINE STATUS — {self.state.upper()}",
            "═" * 60,
            f"  Phase:    {self.phase}  (round {self.round}/{self.max_rounds})",
            f"  Job:      {self.current_job or '—'}",
            f"  Message:  {self.message or '—'}",
            "",
            f"  Started:  {self.started_at}",
            f"  Updated:  {self.updated_at}",
            f"  Elapsed:  {self._fmt_s(self.elapsed_s)}  (phase {self._fmt_s(self.phase_elapsed_s)})",
        ]
        if self.progress:
            lines.append("")
            lines.append("  Progress:")
            p = self.progress
            act = p.get("activity", "")
            if act == "paired_eval":
                lines.append(
                    f"    eval {p.get('eval_done','?')}/{p.get('eval_total','?')}"
                    f"  μ̂_live={p.get('mu_hat_live', '—')}"
                )
            elif act == "king_scoring":
                ds = p.get("scoring_dataset", "?")
                if "scoring_done" in p:
                    lines.append(
                        f"    scoring {ds}: {p['scoring_done']}/{p['scoring_total']}"
                        f" ({p.get('scoring_pct', '?')}%)"
                    )
                else:
                    lines.append(f"    scoring {ds}: n={p.get('scoring_candidates', '?')}")
            elif act == "lora_training":
                lines.append(
                    f"    train {p.get('train_pct', '?')}%"
                    f"  step {p.get('train_step','?')}/{p.get('train_steps','?')}"
                    f"  loss={p.get('train_loss', '—')} epoch={p.get('train_epoch', '—')}"
                )
            if p.get("config"):
                lines.append(f"    config: {p['config']}")
        if self.best:
            lines.extend([
                "",
                "  Best so far:",
                f"    config: {self.best.get('config') or self.best.get('label', '—')}",
                f"    μ̂:      {self._fmt_f(self.best.get('mu_hat'))}",
                f"    LCB:    {self._fmt_f(self.best.get('lcb'))}",
                f"    phase:  {self.best.get('phase', '—')}",
            ])
        lines.extend([
            "",
            f"  Crown target:  μ̂ ≥ {self.crown_mu}  LCB ≥ {self.crown_lcb}",
            f"  Prod ready:    {self.production_ready}",
        ])
        if self.shard_cache:
            sc = ", ".join(f"{k}={v}" for k, v in sorted(self.shard_cache.items()))
            lines.append(f"  Shard cache:   {sc}")
        if self.error:
            lines.extend(["", f"  ERROR: {self.error}"])
        lines.extend([
            "",
            f"  Log: {self.log_path}",
            "─" * 60,
            "  Recent log:",
        ])
        for ln in self.recent_log[-12:]:
            lines.append(f"    {ln[:100]}")
        lines.append("═" * 60)
        return "\n".join(lines)

    @staticmethod
    def _fmt_s(s: float) -> str:
        s = int(s)
        if s < 60:
            return f"{s}s"
        if s < 3600:
            return f"{s // 60}m {s % 60}s"
        return f"{s // 3600}h {(s % 3600) // 60}m"

    @staticmethod
    def _fmt_f(v: Any) -> str:
        if v is None:
            return "—"
        try:
            return f"{float(v):.6f}"
        except (TypeError, ValueError):
            return str(v)


class StatusTracker:
    """Write status.json + status.txt under work dir."""

    def __init__(self, work: Path, *, max_rounds: int = 0) -> None:
        self.work = work.expanduser().resolve()
        self.work.mkdir(parents=True, exist_ok=True)
        self.log_path = self.work / "adaptive.log"
        self.status_path = self.work / "status.json"
        self.status_txt = self.work / "status.txt"
        self._started_mono = time.monotonic()
        self._phase_started_mono = time.monotonic()
        self._started_at = _utc_now()
        self._data = PipelineStatus(
            started_at=self._started_at,
            updated_at=self._started_at,
            phase_started_at=self._started_at,
            max_rounds=max_rounds,
            log_path=str(self.log_path),
            status_path=str(self.status_path),
        )
        self._load_existing()
        self.flush()

    def _load_existing(self) -> None:
        if self.status_path.is_file():
            try:
                raw = json.loads(self.status_path.read_text())
                for k, v in raw.items():
                    if hasattr(self._data, k) and k not in ("elapsed_s", "phase_elapsed_s"):
                        setattr(self._data, k, v)
            except (json.JSONDecodeError, OSError):
                pass

    def set_phase(
        self,
        phase: str,
        *,
        message: str = "",
        round: int | None = None,
        current_job: str = "",
        state: str = "running",
    ) -> None:
        self._phase_started_mono = time.monotonic()
        self._data.phase = phase
        self._data.state = state
        self._data.message = message
        self._data.phase_started_at = _utc_now()
        self._data.current_job = current_job
        self._data.progress = {}
        if round is not None:
            self._data.round = round
        self.flush()

    def update(
        self,
        *,
        message: str | None = None,
        progress: dict[str, Any] | None = None,
        best: dict[str, Any] | None = None,
        shard_cache: dict[str, int] | None = None,
        production_ready: bool | None = None,
        state: str | None = None,
        error: str | None = None,
    ) -> None:
        if message is not None:
            self._data.message = message
        if progress is not None:
            self._data.progress = {**self._data.progress, **progress}
        if best is not None:
            self._data.best = best
        if shard_cache is not None:
            self._data.shard_cache = shard_cache
        if production_ready is not None:
            self._data.production_ready = production_ready
        if state is not None:
            self._data.state = state
        if error is not None:
            self._data.error = error
        self.flush()

    def sync_from_log(self) -> None:
        """Refresh progress + recent_log from adaptive.log tail."""
        tail = _read_tail(self.log_path, 80)
        self._data.recent_log = tail[-20:]
        parsed = parse_log_progress(tail)
        if parsed:
            self._data.progress = {**self._data.progress, **parsed}

    def flush(self) -> None:
        now = time.monotonic()
        self._data.updated_at = _utc_now()
        self._data.elapsed_s = now - self._started_mono
        self._data.phase_elapsed_s = now - self._phase_started_mono
        self.sync_from_log()
        payload = self._data.to_dict()
        tmp = self.status_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str))
        tmp.replace(self.status_path)
        self.status_txt.write_text(self._data.format_text())

    def done(self, message: str = "pipeline complete") -> None:
        self._data.state = "done"
        self._data.message = message
        self.flush()

    def failed(self, error: str) -> None:
        self._data.state = "failed"
        self._data.error = error
        self._data.message = error
        self.flush()


def load_status(work: Path) -> PipelineStatus | None:
    path = work / "status.json"
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text())
        return PipelineStatus(**{k: v for k, v in raw.items() if k in PipelineStatus.__dataclass_fields__})
    except (json.JSONDecodeError, OSError, TypeError):
        return None


def print_status(work: Path) -> int:
    """Print status.txt or build from status.json. Returns exit code."""
    work = work.expanduser().resolve()
    txt = work / "status.txt"
    if txt.is_file():
        print(txt.read_text())
        return 0
    st = load_status(work)
    if st:
        print(st.format_text())
        return 0
    print(f"No status found under {work}")
    print("Start pipeline: ./scripts/mining/run_adaptive_pipeline.sh")
    return 1
