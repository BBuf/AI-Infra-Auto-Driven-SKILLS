---
name: sglang-deepseek-v4-optimization
description: PR-backed and current-main optimization manual for DeepSeek-V4 in SGLang. Use when an engineer needs to audit or extend DeepSeek-V4 Flash/Pro serving recipes, FP4-vs-FP8 checkpoint selection, H200/B200/GB300 launch commands, DeepEP dispatch-token budgets, context-parallel and PD-disaggregation recipes, MTP/EAGLE settings, or DeepSeek-V4 parser flags.
---

# SGLang DeepSeek-V4 Optimization

## Overview

DeepSeek-V4 is now both a current-main runtime lane and a cookbook/command-generator lane in SGLang. The latest PRs include the original deployment matrix, AMD/DeepSeek-V4 runtime integration, CUDA-graph support, DeepGEMM warmup, benchmarking scripts, parser/tool-call support, and model-level fixes.

Current evidence snapshot:

- SGLang `origin/main`: `6fbad22fe` on `2026-04-28`
- Main runtime: `python/sglang/srt/models/deepseek_v4.py`
- Main MTP runtime: `python/sglang/srt/models/deepseek_v4_nextn.py`
- Main attention backend: `python/sglang/srt/layers/attention/deepseek_v4_backend.py`
- Main docs: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`
- Command generator: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`
- Diff-reviewed PRs: #23605, #23617, #23628, #23622, #23634, #23684, #23689, #23690, #23691, #23697, #23698, #23725, #23737, #23742, #23756, #23776, #23787, #23810, #23817, #23832, #23883

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.

## Before You Change Anything

Capture:

- variant: DeepSeek-V4-Flash or DeepSeek-V4-Pro
- hardware: B200, GB300, or H200
- checkpoint dtype: Blackwell FP4 mixed checkpoint or H200 `sgl-project/*-FP8`
- recipe: low-latency, balanced, max-throughput, context-parallel, or PD-disagg
- parser flags: `--reasoning-parser deepseek-v4`, `--tool-call-parser deepseekv4`
- MTP settings and `SGLANG_ENABLE_SPEC_V2`
- DeepEP dispatch-token env budget and `--max-running-requests`

## Core Principle

Treat the DeepSeek-V4 docs as an executable deployment matrix, not ordinary prose.

- H200 must use `sgl-project/DeepSeek-V4-*-FP8`, not the default DeepSeek FP4-mixed repos.
- Blackwell uses the DeepSeek Flash/Pro repos directly.
- Unverified generator cells are intentionally rendered as commented shell no-ops.
- Recipe verification state is part of the serving contract.
- Runtime support is no longer docs-only: #23787 added the DeepSeek-V4 model,
  tokenizer/parser, compressed attention, memory pool, and JIT kernels; #23832
  adds CUDA-graph capture support for the DeepSeek-V4 attention/indexer path.
- #23776 adds the `swiglu_limit` clamp in `DeepseekV2MLP` for V4 checkpoints;
  keep that model-level fix in mind before debugging meaningless-number output.
- #23756/#23883 make DeepGEMM warmup behavior part of the deployment surface.

## PR Dossier Rule

Before adding DeepSeek-V4 evidence, open the PR diff/source and update `references/pr-history.md` with motivation, key implementation, short code/config excerpts, reviewed files, and validation implications. Docs-only PRs still need exact command/config lines.

## Validation Lanes

- B200 Flash/Pro low-latency, balanced, max-throughput, and CP recipe command generation.
- GB200/GB300 verified recipes, including Pro low-latency, CP, and PD-disagg cells.
- H200 Flash low-latency, balanced, and max-throughput command generation with `sgl-project/DeepSeek-V4-Flash-FP8`.
- H200 Pro command generation with `sgl-project/DeepSeek-V4-Pro-FP8` and TP=16 multinode note.
- Parser flags toggled on/off in generated commands.
- PD-disagg commands checked for router port and commented/uncommented state.
- Runtime smoke on `DeepseekV4ForCausalLM`, MTP/nextn, DSML parser, compressed attention, and CUDA-graph replay after changes to `deepseek_v4.py` or attention/indexer code.

## References

- `references/pr-history.md`: diff-reviewed DeepSeek-V4 PR cards.
