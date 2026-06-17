#!/usr/bin/env python3
"""Adaptive hyperparameter pipeline → production strong training.

Phases (automated loop):
  1. dataset screen + optional shard prefetch
  2. score/curate once (reused across rounds)
  3. per round: micro-screen (fast) → confirm (probe-length) → update state
  4. when crown targets met (or --force-strong): strong training + verdict

Time budget per round is kept small: few micro candidates, short max_steps,
confirm only top-1 unless micro μ̂ is near parity.

Example:
  export LOCAL_KING_DIR=/root/teutonic/s1-work/king2
  python -u scripts/mining/run_adaptive_pipeline.py \\
    --work /root/teutonic/s1-work-adaptive \\
    --mix-shard-cache /root/teutonic/s1-work/cache \\
    --sim-hotkey "$TEUTONIC_SIM_HOTKEY"
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

_mining = Path(__file__).resolve().parent
_repo = _mining.parent.parent
if str(_mining) not in sys.path:
    sys.path.insert(0, str(_mining))

from adaptive_tune import (  # noqa: E402
    AdaptiveState,
    confirm_specs_from_micro,
    count_cached_shards,
    generate_micro_candidates,
    rank_key,
    shards_need_prefetch,
    should_skip_confirm,
    strong_spec_from_best,
    write_production_config,
)
from pipeline_status import StatusTracker, print_status  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [adaptive] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("adaptive")


def _train_challenger() -> Path:
    return _mining / "train_challenger.py"


def _download_dataset() -> Path:
    return _mining / "download_dataset.py"


def run_cmd(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    log_file: Path | None = None,
    tracker: StatusTracker | None = None,
    job_label: str = "",
) -> None:
    log.info("exec: %s", " ".join(cmd))
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    log_path = log_file or (tracker.work / "adaptive.log" if tracker else None)
    if tracker and job_label:
        tracker.update(message=f"running: {job_label}")

    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as log_f:
            log_f.write(f"\n\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {job_label} ===\n")
            log_f.flush()
            proc = subprocess.Popen(
                cmd,
                env=merged_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            last_flush = time.monotonic()
            assert proc.stdout is not None
            for line in proc.stdout:
                log_f.write(line)
                log_f.flush()
                if tracker and time.monotonic() - last_flush >= 10:
                    tracker.flush()
                    last_flush = time.monotonic()
            rc = proc.wait()
            if tracker:
                tracker.flush()
            if rc != 0:
                if tracker:
                    tracker.failed(f"{job_label or 'command'} exited {rc}")
                raise subprocess.CalledProcessError(rc, cmd)
        return

    subprocess.run(cmd, check=True, env=merged_env)


def base_challenger_args(args: argparse.Namespace) -> list[str]:
    out = [
        sys.executable, "-u", str(_train_challenger()),
        "--work", str(args.work),
        "--bundle", str(args.bundle),
        "--mix-shard-cache", str(args.mix_shard_cache),
        "--mix-shards-per-dataset", str(args.mix_shards_per_dataset),
        "--skip-chain-check",
        "--profile", args.profile,
        "--sim-hotkey", args.sim_hotkey,
        "--max-iters", "1",
    ]
    if args.local_shards_only:
        out.append("--local-shards-only")
    if args.auto_bucket_mix:
        out.append("--auto-bucket-mix")
    for bm in args.bucket_mix:
        out.append("--bucket-mix")
        out.append(bm)
    return out


def copy_json(src: Path, dst: Path) -> None:
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def load_json(path: Path) -> dict | list | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def phase_screen(args: argparse.Namespace, env: dict[str, str], tracker: StatusTracker) -> None:
    marker = args.work / "dataset_trainability.json"
    if marker.is_file() and not args.force_screen:
        log.info("dataset screen: reusing %s", marker)
        tracker.update(message="dataset screen skipped (cached)")
        return
    tracker.set_phase("screen", message="dataset trainability screen")
    cmd = base_challenger_args(args) + [
        "--mode", "screen",
        "--n-score", str(args.n_score_screen),
    ]
    if args.prefetch_shards:
        cmd.append("--prefetch-mix-shards")
    run_cmd(cmd, env=env, log_file=args.work / "adaptive.log", tracker=tracker, job_label="screen")
    copy_json(args.work / "iter_00/dataset_trainability.json", marker)
    tracker.update(message="dataset screen complete")


def phase_prefetch(args: argparse.Namespace, env: dict[str, str], tracker: StatusTracker) -> None:
    counts = count_cached_shards(Path(args.mix_shard_cache))
    need, low = shards_need_prefetch(counts, min_per_dataset=args.min_shards_per_dataset)
    tracker.update(shard_cache=counts)
    log.info("shard cache: %s", counts)
    if not need and not args.force_prefetch:
        log.info("prefetch: skipped (all datasets >= %d shards)", args.min_shards_per_dataset)
        tracker.update(message=f"prefetch skipped (cache ok: {counts})")
        return
    tracker.set_phase("prefetch", message=f"downloading shards for {low}")
    log.info("prefetch: low cache on %s — downloading", low)
    cmd = [
        sys.executable, "-u", str(_download_dataset()),
        "--work", str(args.mix_shard_cache.parent),
        "--cache-dir", str(args.mix_shard_cache),
        "--dataset-preset", "teutonic-mixture-v2",
        "--shards-per-dataset", str(args.prefetch_shards_per_dataset),
        "--workers", str(args.prefetch_workers),
    ]
    run_cmd(cmd, env=env, log_file=args.work / "adaptive.log", tracker=tracker, job_label="prefetch")
    new_counts = count_cached_shards(Path(args.mix_shard_cache))
    (args.work / "shard_cache_after_prefetch.json").write_text(
        json.dumps(new_counts, indent=2),
    )
    tracker.update(shard_cache=new_counts, message=f"prefetch complete: {new_counts}")


def phase_score_curate(args: argparse.Namespace, env: dict[str, str], tracker: StatusTracker) -> None:
    train_p = args.work / "iter_00" / "train.jsonl"
    val_p = args.work / "iter_00" / "val.jsonl"
    if train_p.is_file() and val_p.is_file() and not args.force_rescore:
        log.info("score/curate: reusing %s", train_p)
        tracker.update(message="score/curate skipped (cached curriculum)")
        return
    tracker.set_phase("score", message="king scoring + curriculum build")
    cmd = base_challenger_args(args) + [
        "--mode", "probe",
        "--n-score", str(args.n_score),
        "--train-per-iter", str(args.train_per_iter),
        "--val-size", str(args.val_size),
        "--stop-after-scoring",
    ]
    if args.force_rescore:
        cmd.append("--force-rescore")
    run_cmd(cmd, env=env, log_file=args.work / "adaptive.log", tracker=tracker, job_label="score")
    tracker.update(message="score/curate complete")


def phase_micro(
    args: argparse.Namespace,
    env: dict[str, str],
    specs: list[str],
    round_dir: Path,
    tracker: StatusTracker,
) -> list[dict]:
    tracker.set_phase(
        "micro",
        round=int(round_dir.name.split("_")[-1]),
        message=f"micro-screen {len(specs)} config(s)",
        current_job=",".join(specs[:2]) + ("..." if len(specs) > 2 else ""),
    )
    cmd = base_challenger_args(args) + [
        "--mode", "micro",
        "--skip-scoring",
        "--micro-screen-only",
        "--micro-screen-max-steps", str(args.micro_max_steps),
        "--micro-screen-eval-n", str(args.micro_eval_n),
        "--micro-screen-bootstrap", str(args.micro_bootstrap),
        "--n-eval", str(args.micro_eval_n),
    ]
    for spec in specs:
        cmd.extend(["--lora-sweep", spec])
    if args.force_micro_rescreen:
        cmd.append("--force-micro-rescreen")
    run_cmd(cmd, env=env, log_file=args.work / "adaptive.log", tracker=tracker, job_label="micro-screen")

    report = load_json(args.work / "iter_00" / "micro_screen" / "micro_screen_results.json")
    results = list((report or {}).get("results") or [])
    copy_json(
        args.work / "iter_00" / "micro_screen" / "micro_screen_results.json",
        round_dir / "micro_screen_results.json",
    )
    tracker.update(message=f"micro-screen done ({len(results)} results)")
    return results


def phase_confirm(
    args: argparse.Namespace,
    env: dict[str, str],
    specs: list[str],
    round_dir: Path,
    tracker: StatusTracker,
) -> list[dict]:
    sweep_dir = args.work / "iter_00" / "lora_sweep"
    if sweep_dir.is_dir() and args.fresh_confirm:
        shutil.rmtree(sweep_dir)

    tracker.set_phase(
        "confirm",
        round=int(round_dir.name.split("_")[-1]),
        message=f"confirm train+eval: {specs}",
        current_job=specs[0] if specs else "",
    )
    cmd = base_challenger_args(args) + [
        "--mode", "probe",
        "--skip-scoring",
        "--n-eval", str(args.confirm_n_eval),
        "--fast-eval-n", str(args.confirm_n_eval),
        "--final-eval-n", str(args.confirm_n_eval),
    ]
    for spec in specs:
        cmd.extend(["--lora-sweep", spec])
    run_cmd(cmd, env=env, log_file=args.work / "adaptive.log", tracker=tracker, job_label="confirm")

    report = load_json(sweep_dir / "sweep_results.json")
    results = list((report or {}).get("results") or [])
    copy_json(sweep_dir / "sweep_results.json", round_dir / "confirm_results.json")
    tracker.update(message=f"confirm done ({len(results)} results)")
    return results


def phase_strong(
    args: argparse.Namespace,
    env: dict[str, str],
    spec: str,
    tracker: StatusTracker,
) -> dict | None:
    strong_work = args.work / "strong"
    strong_work.mkdir(parents=True, exist_ok=True)
    tracker.set_phase("strong", message=f"production strong train: {spec}", current_job=spec)
    # Reuse scored curriculum from base work dir.
    for name in ("train.jsonl", "val.jsonl", "curriculum.json"):
        src = args.work / "iter_00" / name
        dst = strong_work / "iter_00" / name
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists() or args.force_rescore:
                shutil.copy2(src, dst)

    cmd = [
        sys.executable, "-u", str(_train_challenger()),
        "--work", str(strong_work),
        "--bundle", str(args.bundle),
        "--mix-shard-cache", str(args.mix_shard_cache),
        "--mix-shards-per-dataset", str(args.mix_shards_per_dataset),
        "--skip-chain-check",
        "--profile", args.profile,
        "--sim-hotkey", args.sim_hotkey,
        "--mode", "strong",
        "--skip-scoring",
        "--max-iters", "1",
        "--lora-sweep", spec,
        "--report-out", str(strong_work / "verdict.json"),
    ]
    if args.local_shards_only:
        cmd.append("--local-shards-only")
    for bm in args.bucket_mix:
        cmd.extend(["--bucket-mix", bm])
    if args.auto_bucket_mix:
        cmd.append("--auto-bucket-mix")
    run_cmd(cmd, env=env, log_file=args.work / "adaptive.log", tracker=tracker, job_label="strong")

    verdict = load_json(strong_work / "verdict.json")
    copy_json(strong_work / "verdict.json", args.work / "strong_verdict.json")
    sweep = load_json(strong_work / "iter_00" / "lora_sweep" / "sweep_results.json")
    if isinstance(sweep, dict):
        results = sweep.get("results") or []
        if results:
            return max(results, key=rank_key)
    if isinstance(verdict, dict):
        return verdict
    return None


def bootstrap_state_from_work(work: Path, state: AdaptiveState) -> AdaptiveState:
    """Resume adaptive_state.json from completed artifacts after a crash."""
    sweep = load_json(work / "iter_00" / "lora_sweep" / "sweep_results.json")
    micro = load_json(work / "iter_00" / "micro_screen" / "micro_screen_results.json")
    results: list[dict] = []
    phase = "micro"
    if isinstance(micro, dict) and micro.get("results"):
        results = list(micro["results"])
        phase = "micro"
    elif isinstance(sweep, dict) and sweep.get("results"):
        results = list(sweep["results"])
        phase = "confirm" if any(r.get("n_eval", 0) >= 500 for r in results) else "micro"

    if not results:
        return state

    top = max(results, key=rank_key)
    state.record_round(phase=phase, results=results, specs=[])
    # If round_00 exists with micro results, advance to round 1 for next tune pass.
    if (work / "round_00").is_dir() and state.round < 1:
        state.round = 1
    state.production_ready = state.is_production_ready()
    log.info(
        "resume: bootstrapped state from %s results (best μ̂=%s, round=%d)",
        phase, (state.best or {}).get("mu_hat"), state.round,
    )
    return state


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Adaptive LoRA hyperparameter pipeline")
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--bundle", default=str(_repo / "scripts" / "training_bundle"))
    ap.add_argument("--mix-shard-cache", required=True, type=Path)
    ap.add_argument("--sim-hotkey", default=os.environ.get("TEUTONIC_SIM_HOTKEY", ""))
    ap.add_argument("--profile", default="a100-80gb")

    ap.add_argument("--max-rounds", type=int, default=5,
                    help="Max micro→confirm tuning rounds before stopping.")
    ap.add_argument("--max-micro-candidates", type=int, default=4)
    ap.add_argument("--confirm-top-k", type=int, default=1)
    ap.add_argument("--confirm-min-mu", type=float, default=-0.002,
                    help="Skip confirm if micro best μ̂ below this.")
    ap.add_argument("--crown-mu", type=float, default=0.004)
    ap.add_argument("--crown-lcb", type=float, default=0.003)

    ap.add_argument("--n-score-screen", type=int, default=5000)
    ap.add_argument("--n-score", type=int, default=10000)
    ap.add_argument("--train-per-iter", type=int, default=6000)
    ap.add_argument("--val-size", type=int, default=800)

    ap.add_argument("--mix-shards-per-dataset", type=int, default=12)
    ap.add_argument("--min-shards-per-dataset", type=int, default=24)
    ap.add_argument("--prefetch-shards-per-dataset", type=int, default=48)
    ap.add_argument("--prefetch-workers", type=int, default=4)
    ap.add_argument("--local-shards-only", action="store_true")
    ap.add_argument("--prefetch-shards", action="store_true",
                    help="Allow screen/prefetch phases to download shards.")
    ap.add_argument("--auto-bucket-mix", action="store_true")
    ap.add_argument("--bucket-mix", action="append", default=[])

    ap.add_argument("--micro-max-steps", type=int, default=50)
    ap.add_argument("--micro-eval-n", type=int, default=256)
    ap.add_argument("--micro-bootstrap", type=int, default=1000)
    ap.add_argument("--confirm-n-eval", type=int, default=1000)

    ap.add_argument("--skip-screen", action="store_true")
    ap.add_argument("--skip-prefetch", action="store_true")
    ap.add_argument("--skip-strong", action="store_true",
                    help="Stop after tuning; write production_config.json only.")
    ap.add_argument("--force-strong", action="store_true",
                    help="Run strong training with best config even if below crown targets.")
    ap.add_argument("--force-screen", action="store_true")
    ap.add_argument("--force-prefetch", action="store_true")
    ap.add_argument("--force-rescore", action="store_true")
    ap.add_argument("--force-micro-rescreen", action="store_true")
    ap.add_argument(
        "--resume", action="store_true",
        help="Skip screen/prefetch/score if cached; bootstrap adaptive_state from "
             "existing sweep/micro results.",
    )
    ap.add_argument("--fresh-confirm", action="store_true", default=True)
    ap.add_argument("--no-fresh-confirm", action="store_false", dest="fresh_confirm")
    ap.add_argument(
        "--status", action="store_true",
        help="Print current pipeline status and exit (no training).",
    )

    args = ap.parse_args()
    if args.status:
        raise SystemExit(print_status(args.work.expanduser().resolve()))
    if args.resume:
        args.skip_screen = True
        args.skip_prefetch = True
    if not args.sim_hotkey:
        ap.error("--sim-hotkey or TEUTONIC_SIM_HOTKEY required")
    args.work = args.work.expanduser().resolve()
    args.work.mkdir(parents=True, exist_ok=True)
    return args


def main() -> None:
    args = parse_args()
    state_path = args.work / "adaptive_state.json"
    state = AdaptiveState.load(state_path)
    state.crown_mu = args.crown_mu
    state.crown_lcb = args.crown_lcb
    if args.resume and not state.best:
        state = bootstrap_state_from_work(args.work, state)
        state.save(state_path)

    tracker = StatusTracker(args.work, max_rounds=args.max_rounds)
    tracker._data.crown_mu = args.crown_mu
    tracker._data.crown_lcb = args.crown_lcb
    if state.best:
        tracker._data.best = state.best
    tracker._data.production_ready = state.is_production_ready()
    tracker.set_phase("init", message="adaptive pipeline starting", state="running")

    env = {
        "LOCAL_KING_DIR": os.environ.get("LOCAL_KING_DIR", ""),
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get(
            "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True",
        ),
    }
    if not env["LOCAL_KING_DIR"]:
        log.warning("LOCAL_KING_DIR not set — train_challenger will fetch king from dashboard")

    log.info("adaptive pipeline work=%s rounds=%d", args.work, args.max_rounds)
    log.info("status files: %s  %s", tracker.status_path, tracker.status_txt)

    try:
        if not args.skip_screen:
            phase_screen(args, env, tracker)
        if not args.skip_prefetch:
            phase_prefetch(args, env, tracker)
        phase_score_curate(args, env, tracker)

        for r in range(state.round, args.max_rounds):
            state.round = r
            round_dir = args.work / f"round_{r:02d}"
            round_dir.mkdir(parents=True, exist_ok=True)
            log.info("=== ROUND %d/%d ===", r + 1, args.max_rounds)
            tracker.set_phase(
                "tune_round",
                round=r + 1,
                message=f"starting round {r + 1}/{args.max_rounds}",
            )

            specs = generate_micro_candidates(state, max_candidates=args.max_micro_candidates)
            (round_dir / "micro_specs.json").write_text(json.dumps(specs, indent=2))
            log.info("micro candidates: %s", specs)

            micro_results = phase_micro(args, env, specs, round_dir, tracker)
            micro_best = state.record_round(phase="micro", results=micro_results, specs=specs)
            state.save(state_path)
            if state.best:
                tracker.update(best=state.best, production_ready=state.is_production_ready())

            if micro_best:
                log.info(
                    "micro best: %s μ̂=%s LCB=%s",
                    micro_best.get("label"),
                    micro_best.get("mu_hat"),
                    micro_best.get("lcb"),
                )

            if should_skip_confirm(micro_best, min_mu=args.confirm_min_mu):
                log.info(
                    "confirm skipped (micro μ̂ < %g) — adapt candidates next round",
                    args.confirm_min_mu,
                )
                tracker.update(
                    message=f"round {r + 1}: confirm skipped (micro μ̂ too low)",
                )
                state.round = r + 1
                state.save(state_path)
                continue

            confirm_specs = confirm_specs_from_micro(
                micro_best or {},
                top_k=args.confirm_top_k,
                epochs_override=0.8 if float((micro_best or {}).get("mu_hat") or -1) >= 0 else 0.5,
            )
            (round_dir / "confirm_specs.json").write_text(json.dumps(confirm_specs, indent=2))
            log.info("confirm specs: %s", confirm_specs)

            confirm_results = phase_confirm(args, env, confirm_specs, round_dir, tracker)
            for row in confirm_results:
                row["phase"] = "confirm"
            confirm_best = state.record_round(phase="confirm", results=confirm_results, specs=confirm_specs)
            state.production_ready = state.is_production_ready()
            state.round = r + 1
            state.save(state_path)
            if state.best:
                tracker.update(best=state.best, production_ready=state.production_ready)

            if confirm_best:
                log.info(
                    "confirm best: %s μ̂=%s LCB=%s production_ready=%s",
                    confirm_best.get("label") or confirm_best.get("config"),
                    confirm_best.get("mu_hat"),
                    confirm_best.get("lcb"),
                    state.production_ready,
                )

            if state.production_ready:
                log.info("crown targets met — stopping tuning loop")
                tracker.update(message="crown targets met — proceeding to strong")
                break

        strong_spec = strong_spec_from_best(state.best or {})
        prod_path = write_production_config(
            args.work,
            state,
            strong_spec=strong_spec,
            extra={
                "bucket_mix_auto": args.auto_bucket_mix,
                "mix_shards_per_dataset": args.mix_shards_per_dataset,
            },
        )

        if args.skip_strong:
            log.info("skip-strong: production config at %s", prod_path)
            tracker.done("tuning complete (strong skipped)")
            raise SystemExit(0 if state.best else 2)

        if not state.production_ready and not args.force_strong:
            log.warning(
                "strong training skipped: best μ̂=%s LCB=%s below crown (%.4f / %.4f). "
                "Use --force-strong or add data and re-run.",
                (state.best or {}).get("mu_hat"),
                (state.best or {}).get("lcb"),
                args.crown_mu,
                args.crown_lcb,
            )
            tracker.done("tuning complete — below crown, strong skipped")
            raise SystemExit(2)

        log.info("=== STRONG TRAINING: %s ===", strong_spec)
        strong_result = phase_strong(args, env, strong_spec, tracker)
        if strong_result:
            state.record_round(phase="strong", results=[strong_result], specs=[strong_spec])
            state.production_ready = state.is_production_ready()
            state.save(state_path)
            write_production_config(args.work, state, strong_spec=strong_spec)
            tracker.update(best=state.best, production_ready=state.production_ready)

        tracker.done("pipeline complete")
        log.info("adaptive pipeline complete — verdict at %s", args.work / "strong_verdict.json")
        raise SystemExit(0 if state.production_ready or args.force_strong else 2)

    except subprocess.CalledProcessError as exc:
        tracker.failed(f"subprocess failed (exit {exc.returncode})")
        raise
    except SystemExit:
        raise
    except Exception as exc:
        tracker.failed(str(exc))
        raise


if __name__ == "__main__":
    main()
