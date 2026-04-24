---
name: sglang-hunyuan3-preview-optimization
description: PR-backed and current-main optimization manual for Tencent Hunyuan 3 Preview in SGLang. Use when Codex needs to audit or extend Hunyuan3 Preview cookbook recipes, BF16 MoE hardware sizing, H200/B200/B300/GB300 command generation, MTP/EAGLE flags, `hunyuan` reasoning/tool parsers, Blackwell attention backend selection, or trust-remote-code launch guidance.
---

# SGLang Hunyuan 3 Preview Optimization

## Overview

Hunyuan 3 Preview currently enters SGLang through docs/cookbook support with a copyable command generator. The PR is docs-only, but the generator records critical serving assumptions: BF16 weights, TP sizing by GPU memory, parser flags, MTP/EAGLE toggles, Blackwell `trtllm_mha`, and `--trust-remote-code`.

Current evidence snapshot:

- SGLang `origin/main`: `bca3dd958` on `2026-04-24`
- Primary PR: #23532
- Docs: `docs_new/cookbook/autoregressive/Tencent/Hunyuan3-Preview.mdx`
- Command generator: `docs_new/src/snippets/autoregressive/hunyuan3-preview-deployment.jsx`

## Before You Change Anything

Capture:

- hardware: H200, B200, B300, or GB300
- TP and memory fraction
- whether MTP is enabled and `SGLANG_ENABLE_SPEC_V2=1`
- parser flags: `--reasoning-parser hunyuan`, `--tool-call-parser hunyuan`
- whether Blackwell requires `--attention-backend trtllm_mha`
- whether `--trust-remote-code` is present

## Core Principle

Treat the command generator as deployment data.

- H200/B200 use TP=8 for BF16.
- B300/GB300 use TP=4 for BF16.
- A100/H100 80GB single-node is not supported by the documented recipe.
- Blackwell must pass `--attention-backend trtllm_mha`.

## PR Dossier Rule

Before adding Hunyuan3 Preview evidence, open the PR diff/source and update `references/pr-history.md` with motivation, implementation, code/config excerpts, reviewed files, and validation implications.

## Validation Lanes

- H200 and B200 command generation with TP=8.
- B300 and GB300 command generation with TP=4 and `trtllm_mha`.
- reasoning/tool parser toggles.
- MTP toggle adding `SGLANG_ENABLE_SPEC_V2=1` and EAGLE flags.
- `--trust-remote-code` always present.

## References

- `references/pr-history.md`: diff-reviewed Hunyuan3 Preview PR cards.
