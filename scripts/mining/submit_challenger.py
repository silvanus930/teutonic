#!/usr/bin/env python3
"""Submit a pre-built challenger to the chain.

Two ways to provide the Hippius model reference:

1. **Simple submit** — pass repo + digest on the CLI (no verdict.json):

       python scripts/mining/submit_challenger.py \\
         --uploaded-repo yoko/teutonic-q3-4b-5fhmoumc-v1 \\
         --uploaded-digest sha256:... \\
         --wallet-name silvanus-hs2 --hotkey hotkey30

2. **Verdict file** — read from train_challenger.py / validator_eval.py output;
   CLI flags override fields in the file:

       python scripts/mining/submit_challenger.py \\
         --verdict /root/teutonic/s1-work-v3/verdict.json \\
         --wallet-name silvanus-hs2 --hotkey hotkey30

Posts a bittensor reveal commitment:
  `v4|{repo}|sha256:{manifest_digest}|{author_hotkey}`

Run on the host where the wallet lives.

IMPORTANT — coldkey gate (added 2026-04-29):
    The validator REJECTS any Hippius repo whose name does NOT contain the
    first 8 ss58 chars of your **coldkey** (case-insensitive substring).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import bittensor as bt

from model_store import ModelRef, build_reveal_v4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [submit] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("submit_challenger")

COLDKEY_PREFIX_LEN = 8


def _resolve_model_ref(
    verdict_path: str | None,
    cli_repo: str | None,
    cli_digest: str | None,
) -> tuple[str, str, dict | None, bool]:
    """Return (repo, digest, verdict_dict_or_none, simple_submit)."""
    v: dict | None = None
    if verdict_path:
        v = json.loads(Path(verdict_path).read_text())

    best = (v or {}).get("best") or {}
    repo = (
        cli_repo
        or (v or {}).get("uploaded_repo")
        or best.get("uploaded_repo")
        or (v or {}).get("model_repo")
        or best.get("model_repo")
    )
    digest = (
        cli_digest
        or (v or {}).get("uploaded_digest")
        or best.get("uploaded_digest")
        or (v or {}).get("model_digest")
        or best.get("model_digest")
    )
    simple = bool(cli_repo and cli_digest and not verdict_path)
    return (repo or "").strip(), (digest or "").strip(), v, simple


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Reveal a Hippius challenger model on-chain (v4 commitment).",
    )
    ap.add_argument(
        "--verdict",
        default="",
        help="Optional verdict.json from train_challenger.py / validator_eval.py",
    )
    ap.add_argument(
        "--uploaded-repo", "--uploaded_repo",
        dest="uploaded_repo", default="",
        help="Hippius repo id (simple submit; overrides verdict)",
    )
    ap.add_argument(
        "--uploaded-digest", "--uploaded_digest",
        dest="uploaded_digest", default="",
        help="OCI manifest digest sha256:... (simple submit; overrides verdict)",
    )
    ap.add_argument("--hotkey", default="h0")
    ap.add_argument("--wallet-name", default="teutonic")
    ap.add_argument("--netuid", type=int, default=3)
    ap.add_argument("--network", default="finney")
    ap.add_argument("--blocks-until-reveal", type=int, default=3)
    ap.add_argument(
        "--force",
        action="store_true",
        help="Submit even if verdict best.accepted is false (verdict mode only)",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    verdict_path = (args.verdict or "").strip() or None
    cli_repo = (args.uploaded_repo or "").strip() or None
    cli_digest = (args.uploaded_digest or "").strip() or None

    if not verdict_path and not (cli_repo and cli_digest):
        ap.error(
            "provide --uploaded-repo + --uploaded-digest (simple submit), "
            "or --verdict (verdict file), or both (CLI overrides verdict)"
        )
    if (cli_repo and not cli_digest) or (cli_digest and not cli_repo):
        ap.error("--uploaded-repo and --uploaded-digest must be given together")

    repo, digest, v, simple_submit = _resolve_model_ref(verdict_path, cli_repo, cli_digest)
    if not repo or not digest:
        log.error(
            "missing Hippius model ref — set --uploaded-repo + --uploaded-digest, "
            "or add uploaded_repo / uploaded_digest to verdict.json"
        )
        sys.exit(2)

    try:
        model_ref = ModelRef(repo, digest)
    except ValueError as exc:
        log.error("invalid Hippius model ref: %s", exc)
        sys.exit(2)

    best = (v or {}).get("best") or {}
    if v and not simple_submit and not best.get("accepted") and not args.force:
        log.error(
            "offline eval rejected (mu_hat=%.6f, lcb=%.6f, delta=%.6f) — "
            "refusing to burn TAO (use --force to override)",
            best.get("mu_hat", 0), best.get("lcb", 0), best.get("delta", 0),
        )
        sys.exit(3)
    if simple_submit:
        log.info("simple submit: repo=%s digest=%s…%s", repo, digest[:14], digest[-8:])
    elif cli_repo or cli_digest:
        log.info("verdict + CLI override: repo=%s", repo)

    wallet = bt.Wallet(name=args.wallet_name, hotkey=args.hotkey)
    log.info("wallet hotkey: %s", wallet.hotkey.ss58_address)

    coldkey_ss58 = wallet.coldkeypub.ss58_address
    expected_prefix = coldkey_ss58[:COLDKEY_PREFIX_LEN]
    if expected_prefix.lower() not in repo.lower():
        log.error(
            "Hippius repo '%s' does NOT contain your coldkey prefix '%s' "
            "(first %d chars of %s).\n"
            "    The validator will reject this submission with "
            "`coldkey_required` and your tx will be wasted.\n"
            "    Rename your Hippius repo so its full id contains '%s' "
            "(case-insensitive), e.g.\n"
            "        %s/<chain.name>-%s-v1",
            repo, expected_prefix, COLDKEY_PREFIX_LEN, coldkey_ss58,
            expected_prefix,
            repo.split("/", 1)[0] if "/" in repo else "<your-hippius-namespace>",
            expected_prefix,
        )
        sys.exit(6)
    log.info("coldkey gate ok: repo '%s' contains coldkey prefix '%s'",
             repo, expected_prefix)

    payload = build_reveal_v4(model_ref, wallet.hotkey.ss58_address)
    log.info("payload: %s", payload)

    if args.dry_run:
        log.info("[dry-run] not submitting")
        return

    sub = bt.Subtensor(network=args.network)
    meta = sub.metagraph(args.netuid)
    if wallet.hotkey.ss58_address not in meta.hotkeys:
        log.error("hotkey not registered on netuid %d", args.netuid)
        sys.exit(4)
    uid = meta.hotkeys.index(wallet.hotkey.ss58_address)
    log.info("registered as uid=%d", uid)

    resp = sub.set_reveal_commitment(
        wallet=wallet, netuid=args.netuid,
        data=payload, blocks_until_reveal=args.blocks_until_reveal,
        wait_for_revealed_execution=False,
    )
    if resp.success:
        log.info("reveal committed: %s -- validator should pick up after reveal", resp.message)
    else:
        log.error("commitment failed: %s", resp.message)
        sys.exit(5)


if __name__ == "__main__":
    main()
