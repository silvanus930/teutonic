#!/usr/bin/env python3
"""Merge a LoRA checkpoint and run validator-aligned eval (interrupted training).

Example:
    python -u scripts/mining/finish_iter.py \\
        --iter-dir /root/teutonic/s1-work-prod/iter_00 \\
        --checkpoint /root/teutonic/s1-work-prod/iter_00/lora_out/checkpoint-200 \\
        --king /root/teutonic/s1-work-prod/king \\
        --hotkey 5FxJCGB1... \\
        --report-out /root/teutonic/s1-work-prod/verdict.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
_mining = os.path.dirname(os.path.abspath(__file__))
if _mining not in sys.path:
    sys.path.insert(0, _mining)

from train_challenger import merge_lora  # noqa: E402
from validator_eval import validator_style_paired_eval  # noqa: E402


def _pick_checkpoint(iter_dir: Path, explicit: str) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_dir():
            raise FileNotFoundError(f"checkpoint not found: {p}")
        return p
    lora_out = iter_dir / "lora_out"
    best = lora_out / "best_adapter"
    if best.is_dir() and (best / "adapter_model.safetensors").is_file():
        return best
    cks = sorted(lora_out.glob("checkpoint-*"),
                 key=lambda x: int(x.name.split("-")[-1]))
    if not cks:
        raise FileNotFoundError(f"no checkpoints under {lora_out}")
    return cks[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge LoRA checkpoint + validator eval")
    ap.add_argument("--iter-dir", required=True)
    ap.add_argument("--checkpoint", default="", help="default: latest or best_adapter")
    ap.add_argument("--king", required=True)
    ap.add_argument("--hotkey", default=os.environ.get("TEUTONIC_SIM_HOTKEY", ""))
    ap.add_argument("--block-hash", default=os.environ.get("TEUTONIC_SIM_BLOCK_HASH", "0" * 64))
    ap.add_argument("--n-public", type=int, default=5000)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--report-out", default="")
    args = ap.parse_args()

    if not args.hotkey:
        ap.error("--hotkey or TEUTONIC_SIM_HOTKEY required")

    os.environ.setdefault("TEUTONIC_EVAL_DATASET_MODE", "raw_hippius")
    os.environ.setdefault("TEUTONIC_RAW_TOKENIZER_REPO", "Qwen/Qwen3-4B")
    os.environ.setdefault("TEUTONIC_RAW_MAX_FILES_PER_EVAL", "32")

    iter_dir = Path(args.iter_dir)
    king = Path(args.king)
    merged = iter_dir / "merged"
    ckpt = _pick_checkpoint(iter_dir, args.checkpoint)

    print(f"[finish_iter] merge {ckpt} -> {merged}", flush=True)
    merge_lora(str(king), ckpt, merged)

    gpu_ids = [int(x) for x in args.gpus.split(",") if x.strip()]
    t0 = time.time()
    verdict = validator_style_paired_eval(
        str(king), str(merged),
        block_hash=args.block_hash.strip(),
        hotkey=args.hotkey.strip(),
        n_public=args.n_public,
        gpu_ids=gpu_ids,
        batch_size=args.batch_size,
    )
    verdict["merged_dir"] = str(merged)
    verdict["checkpoint"] = str(ckpt)

    report = {"best": verdict, "history": [verdict], "elapsed_s": round(time.time() - t0, 1)}
    if args.report_out:
        Path(args.report_out).write_text(json.dumps(report, indent=2))
        print(f"[finish_iter] wrote {args.report_out}", flush=True)

    print(
        f"accepted={verdict['accepted']} mu_hat={verdict['mu_hat']} lcb={verdict['lcb']}",
        flush=True,
    )
    sys.exit(0 if verdict["accepted"] else 1)


if __name__ == "__main__":
    main()
