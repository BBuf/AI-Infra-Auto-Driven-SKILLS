---
name: sglang-mimo-v2-flash-optimization
description: PR-backed optimization manual for MiMo-V2-Flash in SGLang. Use when Codex needs to audit, debug, extend, or document MiMo-V2-Flash inference-centric MoE runtime, flashinfer fused all-reduce, overlap, and reasoning parser behavior.
---

# SGLang MiMo-V2-Flash Optimization

## Overview

This skill covers MiMo-V2-Flash inference-centric MoE runtime, flashinfer fused all-reduce, overlap, and reasoning parser behavior.

Evidence snapshot:

- SGLang mainline checked around `c122d343adb969cd9bbd1af2ca86727a11be3845`
- sgl-cookbook checked around `e88b0fd8ac5b1caa6eb42766035029220053369b`
- Canonical PR notes: `references/pr-history.md`
- History mirrors: `model-pr-optimization-history/sglang/mimo-v2-flash/README.zh.md` and `README.en.md`

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.

## Runtime Surfaces

- `sglang/python/sglang/srt/models/mimo_v2_flash.py`
- `sglang/python/sglang/srt/models/mimo_v2_flash_nextn.py`

## Current Main Summary

- MiMo-V2-Flash is primarily a throughput-oriented MoE serving family.
- All-reduce fusion, overlap, and reasoning behavior matter more than generic text-only loader work.

## Key Landed PRs

- [#15207](https://github.com/sgl-project/sglang/pull/15207) `MiMo-V2-Flash day0 support`: Initial MiMo-V2-Flash landing.
- [#15464](https://github.com/sgl-project/sglang/pull/15464) `Optimize MiMo-V2-Flash by flashinfer fused allreduce`: Targeted decode-side communication cost.
- [#15488](https://github.com/sgl-project/sglang/pull/15488) `Respect `--swa-full-tokens-ratio``: Fixed a concrete runtime flag integration bug.
- [#17634](https://github.com/sgl-project/sglang/pull/17634) `Support two batch overlap`: Added overlap / throughput optimization.
- [#21414](https://github.com/sgl-project/sglang/pull/21414) `Add mimo reasoning parser`: Completed the parser path for thinking outputs.

## Validation Lanes

- Startup on the cookbook checkpoint or example route.
- Re-run the parser, quantization, or multimodal lane that matches this family.
- Re-check registered or manual tests after touching loader or processor code.
