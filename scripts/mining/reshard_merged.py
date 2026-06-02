#!/usr/bin/env python3
"""Re-save a merged checkpoint with safetensors sharded by max file size.

Use when merge produced one large model.safetensors (e.g. ~17 GB) but Hippius
upload needs ~3.5 GB per file.

Example:
  python scripts/mining/reshard_merged.py \\
    --in /root/teutonic/smoke-test/iter_00/merged \\
    --out /root/teutonic/smoke-test/iter_00/merged_sharded \\
    --max-shard-size 3500MB
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_mining = Path(__file__).resolve().parent
if str(_mining) not in sys.path:
    sys.path.insert(0, str(_mining))
from hf_king_compat import hf_remote_code_kwargs, patch_transformers_quasar_compat  # noqa: E402

patch_transformers_quasar_compat()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [reshard] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("reshard")


def main() -> None:
    ap = argparse.ArgumentParser(description="Shard a merged model directory by max file size.")
    ap.add_argument("--in", dest="inp", required=True, help="Input merged model dir")
    ap.add_argument("--out", required=True, help="Output directory (created fresh)")
    ap.add_argument("--max-shard-size", default="3500MB", help="Per-shard cap, e.g. 3500MB or 3.5GB")
    ap.add_argument(
        "--strip-py", action="store_true",
        help="Do not copy modeling_*.py (for Hippius submit artifact)",
    )
    ap.add_argument(
        "--strip-auto-map", action="store_true",
        help="Remove auto_map from config.json in output",
    )
    args = ap.parse_args()

    inp = Path(args.inp).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    if not inp.is_dir():
        raise SystemExit(f"input not found: {inp}")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"output already exists and is non-empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    king_kw = hf_remote_code_kwargs(str(inp))
    log.info("loading %s (cpu, low_cpu_mem_usage)", inp)
    model = AutoModelForCausalLM.from_pretrained(
        str(inp),
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        use_safetensors=True,
        **king_kw,
    )
    log.info("saving sharded weights to %s (max_shard_size=%s)", out, args.max_shard_size)
    model.save_pretrained(
        str(out),
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tok = AutoTokenizer.from_pretrained(str(inp), use_fast=True, **king_kw)
    tok.save_pretrained(str(out))

    for name in ("generation_config.json",):
        src = inp / name
        if src.is_file():
            shutil.copy(src, out / name)

    cfg_src = inp / "config.json"
    if cfg_src.is_file():
        cfg = json.loads(cfg_src.read_text())
        if args.strip_auto_map:
            cfg.pop("auto_map", None)
        (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    if not args.strip_py:
        for py_name in ("configuration_qwen3_5.py", "modeling_qwen3_5.py"):
            src_py = inp / py_name
            if src_py.is_file():
                shutil.copy(src_py, out / py_name)

    for p in sorted(out.glob("*.safetensors")):
        log.info("  shard %s (%.2f GB)", p.name, p.stat().st_size / 1e9)
    log.info("done: %s", out)


if __name__ == "__main__":
    main()
