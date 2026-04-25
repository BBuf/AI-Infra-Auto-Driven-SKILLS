---
name: sglang-qwen3-coder-optimization
description: PR-backed and current-main optimization manual for Qwen3-Coder and Qwen3-Coder-Next in SGLang. Use when Codex needs to recover, extend, or audit Qwen3-Coder-480B-A35B, Qwen3-Coder-Next, tool-call parser behavior, incremental streaming tool arguments, NVFP4/FP8 loading, MoE fused configs, AMD/NPU/Blackwell recipes, or coding-agent deployment docs.
---

# SGLang Qwen3-Coder Optimization

## Overview

Qwen3-Coder needs its own skill because its optimization surface spans both model execution and parser behavior. The parser is used by Qwen3.6 docs as well, so changes here can affect multiple Qwen deployments.

Current evidence snapshot:

- SGLang `origin/main`: `b3e6cf60a` on `2026-04-22`
- sgl-cookbook `origin/main`: `816bad5` on `2026-04-21`
- Parser file: `python/sglang/srt/function_call/qwen3_coder_detector.py`
- Docs/snippets: `docs_new/cookbook/autoregressive/Qwen/Qwen3-Coder.mdx`, `Qwen3-Coder-Next.mdx`, and their deployment snippets
- Adjacent runtime: Qwen3 MoE and Qwen3-Next model files, depending on checkpoint

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.

## Before You Change Anything

Capture:

- checkpoint: Qwen3-Coder-480B-A35B or Qwen3-Coder-Next
- whether tool parser is enabled
- streaming or non-streaming tool-call mode
- quantization: BF16, FP8, NVFP4, or ModelOpt export
- hardware: H200/B200/GB200, AMD MI300/MI325/MI355, NPU
- whether MTP, EAGLE, or Qwen3-Next hybrid path is active

## Core Principle

Separate parser correctness from model performance.

- Tool-call streaming bugs should be reproduced with small local responses before changing model runtime.
- Qwen3-Coder-Next uses the hybrid Qwen3-Next lane; model fixes should cite that skill when they are not parser-specific.
- Cookbook command changes need parser, quantization, and hardware coverage.
- Every cited PR must be handled as a diff dossier: read the full patch, write the motivation, key implementation idea, core code snippet, reviewed files, and validation impact. Do not add title-only PR lists.

## Main Runtime Surfaces

- `python/sglang/srt/function_call/qwen3_coder_detector.py`
- `docs_new/cookbook/autoregressive/Qwen/Qwen3-Coder.mdx`
- `docs_new/cookbook/autoregressive/Qwen/Qwen3-Coder-Next.mdx`
- `docs_new/src/snippets/autoregressive/qwen3-coder-deployment.jsx`
- `docs_new/src/snippets/autoregressive/qwen3-coder-next-deployment.jsx`
- `python/sglang/srt/models/qwen3_next.py`
- `python/sglang/srt/models/qwen3_moe.py`

## Optimization Order

1. Reproduce parser behavior with deterministic streaming chunks.
2. Verify base model correctness without parser flags.
3. Enable tool parser and test complete JSON plus incremental deltas.
4. Add quantized loading tests for Coder/Next checkpoint.
5. Tune MoE or GDN kernels in the corresponding Qwen3/Qwen3-Next lane.
6. Update cookbook hardware recipes.

## PR Dossier Rule

Use `references/pr-history.md` as the authoritative ledger. It includes manually reviewed merged PRs and manually reviewed open design PRs. If you cite another Qwen3-Coder PR, first open its metadata and full diff, then add a dossier entry before using it as evidence.

Open PRs that are already diff-reviewed in the ledger:

- [#13411](https://github.com/sgl-project/sglang/pull/13411): schema-aware Qwen3-Coder parameter conversion.
- [#13979](https://github.com/sgl-project/sglang/pull/13979): Qwen3-Coder-480B nightly performance tests.
- [#21829](https://github.com/sgl-project/sglang/pull/21829): incremental streaming for tool-call arguments.

## Validation Lanes

- Parser unit tests with multiple tools, nested JSON, and incremental streaming deltas.
- Model smoke test with `--tool-call-parser qwen3_coder`.
- Qwen3-Coder-Next quantized loading on target hardware.
- Cookbook command parity for NVIDIA and AMD recipes.
