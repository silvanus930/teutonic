#!/usr/bin/env python3
"""Print current adaptive pipeline status (status.txt / status.json)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_mining = Path(__file__).resolve().parent
if str(_mining) not in sys.path:
    sys.path.insert(0, str(_mining))

from pipeline_status import print_status  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Show adaptive pipeline status")
    ap.add_argument(
        "--work", type=Path,
        default=Path("/root/teutonic/s1-work-adaptive"),
        help="Pipeline work directory",
    )
    ap.add_argument("--json", action="store_true", help="Print status.json raw")
    args = ap.parse_args()
    work = args.work.expanduser().resolve()
    if args.json:
        p = work / "status.json"
        if p.is_file():
            print(p.read_text())
            raise SystemExit(0)
        raise SystemExit(1)
    raise SystemExit(print_status(work))


if __name__ == "__main__":
    main()
