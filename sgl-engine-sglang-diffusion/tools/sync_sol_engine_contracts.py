#!/usr/bin/env python3
"""Check or refresh hashes of reviewed Sol-Engine contract inputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from sgl_engine_sglang_diffusion.knowledge import (  # noqa: E402
    check_contract_hashes,
    write_contract_hashes,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--sol-checkout", type=Path, required=True)
    parser.add_argument("--hashes", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--update", action="store_true")
    args = parser.parse_args()

    if args.update:
        write_contract_hashes(args.source_lock, args.sol_checkout, args.hashes)
        print(f"updated {args.hashes}")
        return 0

    issues = check_contract_hashes(
        args.source_lock, args.sol_checkout, args.hashes
    )
    for issue in issues:
        print(issue, file=sys.stderr)
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
