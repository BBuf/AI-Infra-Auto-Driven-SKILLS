---
name: sglang-mixtral-quark-int4fp8-moe-optimization
description: PR-backed optimization manual for Mixtral-8x7B with SGLang's AMD-only quark_int4fp8_moe online MoE quantization. Use when Codex needs to audit or extend Mixtral AMD quantization, online INT4-to-FP8 MoE loading, AITER fused-MoE execution, or the registered GSM8K regression test.
---

# SGLang Mixtral Quark INT4-FP8 MoE Optimization

## Overview

`quark_int4fp8_moe` is an AMD-only online quantization path used by MoE checkpoints such as `mistralai/Mixtral-8x7B-Instruct-v0.1`. It loads high-precision MoE expert weights, quantizes them to packed INT4, stores scales, and executes with FP8-style MoE math on ROCm.

Current evidence snapshot:

- SGLang `origin/main`: `bca3dd958` on `2026-04-24`
- Quantization config: `python/sglang/srt/layers/quantization/quark_int4fp8_moe.py`
- Test: `test/registered/quant/test_int4fp8_moe.py`
- Docs: `docs_new/docs/advanced_features/quantization.mdx`
- Diff-reviewed PRs: #7392, #17116, #23455

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.

## Before You Change Anything

Capture:

- model id: `mistralai/Mixtral-8x7B-Instruct-v0.1`
- hardware: AMD CDNA3/CDNA4 only
- launch flags: `--quantization quark_int4fp8_moe`, `--attention-backend triton`, `--tp 2`
- test suite: `stage-b-test-2-gpu-large-amd`
- GSM8K threshold: score greater than `0.56`

## Core Principles

- This path is online quantization, not a pre-quantized checkpoint loader.
- It is HIP-only; non-AMD execution should fail loudly.
- MoE expert weights are sharded before online quantization when checkpoints are not pre-sharded.
- Packed INT4 weights and FP8/int4 scales are runtime contract fields on `FusedMoE`.
- Direct-file test execution matters because #23455 restored the test after missing `__main__` handling.

## Validation Lanes

- Launch Mixtral with `--quantization quark_int4fp8_moe --attention-backend triton --tp 2`.
- Run `test/registered/quant/test_int4fp8_moe.py` on AMD and verify GSM8K score above `0.56`.
- Check that quantization refuses non-HIP devices.
- Inspect online quantization progress and memory behavior during weight load.

## References

- `references/pr-history.md`: diff-reviewed Mixtral/quark INT4-FP8 MoE PR cards.
