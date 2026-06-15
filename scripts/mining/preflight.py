"""Preflight checks: chain match, model hygiene, pre-submit gate."""
from __future__ import annotations

import json
import logging
import re
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

SUBMIT_DECISIONS = (
    "DO_NOT_SUBMIT",
    "PROMISING_NEEDS_MORE_EVAL",
    "READY_TO_MERGE",
    "READY_TO_UPLOAD",
)


def fetch_dashboard_chain_name(dashboard_url: str) -> str:
    with urllib.request.urlopen(dashboard_url, timeout=30) as resp:
        data = json.loads(resp.read())
    return str(data.get("chain_name") or data.get("chain") or "")


def check_chain_match(
    expected_chain_name: str,
    dashboard_url: str,
    *,
    strict: bool = True,
) -> None:
    """Raise if chain.toml name does not match dashboard chain name."""
    expected = (expected_chain_name or "").strip()
    if not expected:
        return
    try:
        dashboard_chain = fetch_dashboard_chain_name(dashboard_url).strip()
    except Exception as exc:
        if strict:
            raise RuntimeError(f"failed to fetch dashboard chain name: {exc}") from exc
        log.warning("could not verify dashboard chain name: %s", exc)
        return

    if dashboard_chain and dashboard_chain != expected:
        msg = (
            f"Chain mismatch: chain.toml name={expected!r} "
            f"but dashboard chain_name={dashboard_chain!r}. "
            "Fix chain.toml or verify you are mining the correct subnet challenge."
        )
        if strict:
            raise RuntimeError(msg)
        log.warning(msg)


def validate_merged_model(model_dir: Path) -> tuple[bool, list[str]]:
    """Check merged model directory meets submission hygiene requirements."""
    model_dir = Path(model_dir)
    issues: list[str] = []

    if not (model_dir / "config.json").is_file():
        issues.append("missing config.json")

    tok_files = list(model_dir.glob("tokenizer.json")) + list(model_dir.glob("tokenizer_config.json"))
    if not tok_files:
        issues.append("missing tokenizer files (tokenizer.json or tokenizer_config.json)")

    single = model_dir / "model.safetensors"
    sharded = list(model_dir.glob("model-*-of-*.safetensors"))
    index = model_dir / "model.safetensors.index.json"
    if not single.is_file() and not sharded and not index.is_file():
        issues.append("missing model.safetensors or valid sharded safetensors index")

    py_files = list(model_dir.glob("*.py"))
    if py_files:
        issues.append(f"contains forbidden .py files: {[p.name for p in py_files]}")

    cfg_path = model_dir / "config.json"
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text())
        if cfg.get("auto_map"):
            issues.append("config.json contains auto_map (forbidden for submission)")

    return len(issues) == 0, issues


def validate_upload_repo(repo_id: str, coldkey_prefix: str) -> tuple[bool, str]:
    prefix = (coldkey_prefix or "").strip()[:8]
    if not prefix:
        return True, ""
    repo_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
    if prefix not in repo_name:
        return False, (
            f"upload repo {repo_id!r} must contain coldkey prefix {prefix!r} "
            "(first 8 ss58 chars of coldkey)"
        )
    return True, ""


def pre_submit_decision(
    *,
    lcb: float,
    mu_hat: float,
    n_eval: int,
    lcb_floor: float = 0.0025,
    preferred_lcb_margin: float = 0.0035,
    preferred_mu_hat: float = 0.0075,
    min_n_eval: int = 3000,
    merged_hygiene_ok: bool | None = None,
    mixture_lcb: float | None = None,
    mixture_mu_hat: float | None = None,
    regression_warnings: list[str] | None = None,
    block_on_regression: bool = True,
) -> tuple[str, list[str]]:
    """Conservative pre-submit verdict (supports mixture-weighted metrics)."""
    reasons: list[str] = []
    eff_lcb = mixture_lcb if mixture_lcb is not None else lcb
    eff_mu = mixture_mu_hat if mixture_mu_hat is not None else mu_hat

    if regression_warnings:
        for w in regression_warnings:
            reasons.append(w)
        if block_on_regression:
            return "DO_NOT_SUBMIT", reasons

    if eff_lcb <= lcb_floor:
        reasons.append(f"mixture_lcb={eff_lcb:.6f} <= floor={lcb_floor:.6f}")
        return "DO_NOT_SUBMIT", reasons

    if n_eval < min_n_eval:
        reasons.append(f"n_eval={n_eval} < min_n_eval={min_n_eval}")
        return "PROMISING_NEEDS_MORE_EVAL", reasons

    if eff_lcb >= preferred_lcb_margin and eff_mu >= preferred_mu_hat:
        if merged_hygiene_ok is False:
            reasons.append("merged model failed hygiene checks")
            return "READY_TO_MERGE", reasons
        reasons.append(
            f"strong mixture margins: lcb={eff_lcb:.6f}>={preferred_lcb_margin}, "
            f"mu_hat={eff_mu:.6f}>={preferred_mu_hat}"
        )
        return "READY_TO_UPLOAD", reasons

    reasons.append(
        f"passes floor (mixture_lcb={eff_lcb:.6f}) but below preferred margins "
        f"(lcb>={preferred_lcb_margin}, mu_hat>={preferred_mu_hat})"
    )
    return "READY_TO_MERGE", reasons


def print_submit_verdict(decision: str, reasons: list[str]) -> None:
    banner = "=" * 60
    print(f"\n{banner}\nPRE-SUBMIT VERDICT: {decision}\n{banner}", flush=True)
    for reason in reasons:
        print(f"  - {reason}", flush=True)
    if decision == "DO_NOT_SUBMIT":
        print("  → Do not merge for submission or upload.", flush=True)
    elif decision == "PROMISING_NEEDS_MORE_EVAL":
        print("  → Increase --n-eval or run strong mode before merge/submit.", flush=True)
    elif decision == "READY_TO_MERGE":
        print("  → Safe to merge best adapter. Re-eval before upload.", flush=True)
    elif decision == "READY_TO_UPLOAD":
        print("  → Candidate passes conservative gate. Upload only with --upload-approved.", flush=True)
    print(f"{banner}\n", flush=True)
