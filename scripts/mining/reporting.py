"""Run reports: JSON + Markdown summaries per mining run."""
from __future__ import annotations

import json
import time
from pathlib import Path


def save_run_report(work: Path, report: dict, *, basename: str = "run_report") -> tuple[Path, Path]:
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    report = dict(report)
    report.setdefault("ts", time.time())

    json_path = work / f"{basename}.json"
    json_path.write_text(json.dumps(report, indent=2, default=str))

    md_path = work / f"{basename}.md"
    md_path.write_text(render_markdown(report))
    return json_path, md_path


def render_markdown(report: dict) -> str:
    lines = [
        "# Teutonic Mining Run Report",
        "",
        f"- **Chain**: {report.get('chain_name', '?')}",
        f"- **King**: {report.get('king_repo', '?')} @ {str(report.get('king_digest', '?'))[:16]}",
        f"- **Mode**: {report.get('mode', report.get('candidate_preset', '?'))}",
        "",
        "## Dataset",
        f"- Preset: {report.get('dataset_preset', report.get('mode', '?'))}",
        f"- Weights: {report.get('dataset_weights', {})}",
        f"- Shards: {report.get('dataset_shards_used', report.get('shards_used', []))}",
    ]
    allocation = report.get("sample_allocations") or {}
    if allocation:
        lines.append(f"- Allocations: {allocation}")
    lines.extend(["", "## Curriculum"])
    curriculum = report.get("curriculum_stats") or {}
    per_ds = curriculum.get("per_dataset") or report.get("curriculum_per_dataset") or {}
    if per_ds:
        lines.append("- Per-dataset bucket stats:")
        for ds, stats in per_ds.items():
            lines.append(f"  - {ds}: train_n={stats.get('train_n')} val_n={stats.get('val_n')} mix={stats.get('train_mix')}")
    if curriculum and not per_ds:
        lines.extend([
            f"- Train: {curriculum.get('train_n', '?')} | Val: {curriculum.get('val_n', '?')}",
            f"- Mix: {curriculum.get('train_mix', {})}",
            f"- Buckets: {curriculum.get('bucket_counts', {})}",
        ])
    elif not per_ds and not curriculum:
        lines.append("- (not available)")

    lines.extend(["", "## LoRA Sweep"])
    sweep = report.get("lora_configs_tested") or []
    if sweep:
        for entry in sweep:
            cfg = entry.get("config", entry.get("label", "?"))
            lines.append(
                f"- `{cfg}`: train_loss={entry.get('train_loss')} "
                f"eval_loss={entry.get('eval_loss')} "
                f"mu_hat={entry.get('mu_hat')} lcb={entry.get('lcb')} "
                f"({entry.get('elapsed_s', '?')}s)"
            )
    else:
        lines.append(f"- Single config: r={report.get('lora_r')} a={report.get('lora_alpha')} lr={report.get('lr')}")

    best = report.get("best_config") or {}
    if best:
        lines.extend([
            "",
            "## Best Config",
            f"- `{best.get('label', best.get('config', '?'))}`",
            f"- mu_hat={best.get('mu_hat')} lcb={best.get('lcb')} accepted={best.get('accepted')}",
        ])

    paired = report.get("paired_eval") or report.get("best_paired_eval") or {}
    per_eval = report.get("per_dataset_eval") or (paired.get("per_dataset") if paired else {}) or {}
    if per_eval:
        lines.extend(["", "## Per-Dataset Eval"])
        for ds, ev in per_eval.items():
            lines.append(
                f"- **{ds}**: n={ev.get('n_eval')} mu_hat={ev.get('mu_hat')} "
                f"lcb={ev.get('lcb')} king={ev.get('avg_king_loss')} chall={ev.get('avg_chall_loss')}"
            )
    if paired:
        lines.extend([
            "",
            "## Mixture Eval",
            f"- n_eval={paired.get('n_eval')} mixture_mu_hat={paired.get('mixture_mu_hat', paired.get('mu_hat'))} "
            f"mixture_lcb={paired.get('mixture_lcb', paired.get('lcb'))}",
            f"- accepted={paired.get('accepted')} mode={paired.get('eval_mode', '?')}",
        ])
        warns = paired.get("regression_warnings") or []
        for w in warns:
            lines.append(f"- ⚠ {w}")

    validator = report.get("validator_eval")
    if validator:
        lines.extend([
            "",
            "## Validator Eval (dual-eval)",
            f"- n_eval={validator.get('n_eval')} mu_hat={validator.get('mu_hat')} lcb={validator.get('lcb')}",
            f"- accepted={validator.get('accepted')}",
        ])

    decision = report.get("final_decision") or report.get("submit_decision")
    if decision:
        lines.extend(["", "## Final Decision", f"**{decision}**"])
        reasons = report.get("decision_reasons") or []
        for reason in reasons:
            lines.append(f"- {reason}")

    merged = report.get("merged_model_path")
    if merged:
        lines.extend(["", "## Merged Model", f"`{merged}`"])

    lines.append("")
    return "\n".join(lines)
