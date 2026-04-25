---
name: sglang-gpt-oss-optimization
description: PR-backed optimization manual for GPT-OSS in SGLang. Use when Codex needs to audit, debug, extend, or document OpenAI GPT-OSS MoE, MXFP4/FP8 quantization, DP/EP, reasoning parser, tool calling, and Eagle/spec decode.
---

# SGLang GPT-OSS Optimization

## Overview

This skill covers OpenAI GPT-OSS MoE, MXFP4/FP8 quantization, DP/EP, reasoning parser, tool calling, and Eagle/spec decode.

Evidence snapshot:

- SGLang mainline checked around `c122d343adb969cd9bbd1af2ca86727a11be3845`
- sgl-cookbook checked around `e88b0fd8ac5b1caa6eb42766035029220053369b`
- Canonical PR notes: `references/pr-history.md`
- History mirrors: `model-pr-optimization-history/sglang/gpt-oss/README.zh.md` and `README.en.md`

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.

## Runtime Surfaces

- `sglang/python/sglang/srt/models/gpt_oss.py`

## Current Main Summary

- GPT-OSS is a flagship MoE family in SGLang.
- Quantization, expert-parallel topology, and reasoning/tool parser behavior all evolved quickly and need per-PR tracking.

## Key Landed PRs

- [#8843](https://github.com/sgl-project/sglang/pull/8843) `Support mxfp4 for GPT-OSS`: Added the headline quantized checkpoint path.
- [#8944](https://github.com/sgl-project/sglang/pull/8944) `Expert Parallelism for GPT-OSS`: Scaled GPT-OSS beyond pure tensor parallel.
- [#9043](https://github.com/sgl-project/sglang/pull/9043) `Implement Native GPT-OSS Tool Call Support`: Added native tool parser support instead of Harmony integration.
- [#9359](https://github.com/sgl-project/sglang/pull/9359) `Support DP attention with GPT-OSS`: Enabled larger topologies via DP attention.
- [#14920](https://github.com/sgl-project/sglang/pull/14920) `GPT-OSS Eagle v2 support`: Added speculative decoding support.
- [#18988](https://github.com/sgl-project/sglang/pull/18988) `Support fp8 online quantization for gpt-oss bf16`: Extended quantization coverage to online FP8.

## Validation Lanes

- Startup on the cookbook checkpoint or example route.
- Re-run the parser, quantization, or multimodal lane that matches this family.
- Re-check registered or manual tests after touching loader or processor code.
