---
name: sglang-qwen36-optimization
description: PR-backed and current-main optimization manual for Qwen3.6 in SGLang. Use when Codex needs to recover, extend, or audit Qwen3.6-35B-A3B/27B dense deployment, hybrid Gated Delta Network behavior, multimodal inputs, thinking preservation, Qwen3 reasoning plus Qwen3-Coder tool parser, MTP, Mamba scheduler strategy, FP8/BF16 commands, CPU offload, or cookbook parity.
---

# SGLang Qwen3.6 Optimization

## Overview

Qwen3.6 is split out because the current SGLang support is mostly a deployment/cookbook layer over hybrid Qwen infrastructure, with unique user-facing behavior: thinking preservation, multimodal input, Qwen3 reasoning parser, Qwen3-Coder tool parser, MTP, and Mamba scheduler strategy.

Current evidence snapshot:

- SGLang `origin/main`: `b3e6cf60a` on `2026-04-22`
- sgl-cookbook `origin/main`: `816bad5` on `2026-04-21`
- Three-pass completeness audit: `model-pr-optimization-history/sglang/qwen-glm-three-pass-audit-2026-04-23.md`
- Docs/snippet files: `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`, `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx`
- Adjacent runtime files: `qwen3_next.py`, `qwen3_5.py`, Qwen VLM processors, and Qwen3-Coder parser depending on the checkpoint path

## Before You Change Anything

Capture:

- checkpoint: Qwen3.6-35B-A3B BF16/FP8 or Qwen3.6-27B dense
- whether multimodal input is used
- reasoning parser: usually `qwen3`
- tool parser: usually `qwen3_coder`
- MTP enabled or disabled
- `--mamba-scheduler-strategy`: default `no_buffer` or `extra_buffer`
- whether `SGLANG_ENABLE_SPEC_V2=1` is set
- attention backend on B200: current docs add `--attention-backend trtllm_mha`

## Core Principle

Qwen3.6 work should preserve deployment semantics while reusing lower-level Qwen3-Next/Qwen3.5 machinery.

- Do not add Qwen3.6-specific runtime forks until the actual architecture mismatch is proven.
- Keep docs and command generators accurate; this is currently the primary source of Qwen3.6 behavior in the repo.
- Treat CPU offload and hybrid linear-attention bugs as shared hybrid-model issues.

## Main Runtime Surfaces

- `docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx`
- `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx`
- `docs_new/docs.json`
- `python/sglang/srt/models/qwen3_next.py`
- `python/sglang/srt/models/qwen3_5.py`
- `python/sglang/srt/function_call/qwen3_coder_detector.py`
- `python/sglang/srt/multimodal/processors/qwen_vl.py`

## Optimization Order

1. First verify that the desired behavior is absent from Qwen3-Next/Qwen3.5 shared code.
2. Keep Qwen3.6 docs and command generator in sync with any runtime flag change.
3. Validate reasoning and tool calling together because docs recommend both.
4. Validate multimodal requests separately from text-only MTP.
5. Only split runtime code if the checkpoint config requires a new class or loader rule.

## PR Dossier Rule

Before adding Qwen3.6 PR evidence, open the PR diff/source and write a full card in `references/pr-history.md` with motivation, key implementation, code excerpt, reviewed files, and validation implications.

Do not add title-only open PR radar lists. If an open PR matters for Qwen3.6, keep it in a clearly marked open-radar card only after reviewing the diff, as done for `#23474`.

## Validation Lanes

- Text-only BF16/FP8 server command from `qwen36-deployment.jsx`.
- Multimodal image and video request examples from `Qwen3.6.mdx`.
- Streaming reasoning plus tool-call parser together.
- MTP with `extra_buffer` and no-MTP with default scheduler.
