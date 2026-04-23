"""Compatibility entrypoint for unified LLM torch-profiler triage."""

from __future__ import annotations

import sys

from analyze_sglang_torch_profile import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
