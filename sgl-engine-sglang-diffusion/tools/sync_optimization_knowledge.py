#!/usr/bin/env python3
"""Snapshot one allowlisted optimization-knowledge source."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from sgl_engine_sglang_diffusion.knowledge import (  # noqa: E402
    load_registry,
    sync_source,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    registry = load_registry(args.registry)
    if args.source not in registry:
        parser.error(f"unknown knowledge source: {args.source}")
    snapshot = sync_source(
        name=args.source,
        checkout=args.checkout,
        commit=args.commit,
        patterns=registry[args.source],
        output_dir=args.output,
    )
    print(json.dumps(snapshot.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
