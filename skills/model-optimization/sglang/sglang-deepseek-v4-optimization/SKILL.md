---
name: sglang-deepseek-v4-optimization
description: PR-backed and current-main optimization manual for DeepSeek-V4 in SGLang. Use when an engineer needs to audit or extend DeepSeek-V4 Flash/Pro serving recipes, FP4-vs-FP8 checkpoint selection, H200/B200/GB300 launch commands, DeepEP dispatch-token budgets, context-parallel and PD-disaggregation recipes, MTP/EAGLE settings, or DeepSeek-V4 parser flags.
---

# SGLang DeepSeek-V4 Optimization

## Overview

DeepSeek-V4 is now both a current-main runtime lane and a cookbook/command-generator lane in SGLang. The latest PRs include the original deployment matrix, AMD/DeepSeek-V4 runtime integration, CUDA-graph support, DeepGEMM warmup, benchmarking scripts, parser/tool-call support, model-level fixes, DeepSeek-V4-specific JIT kernels, KV-compression v2, and W4A4 / MXFP4 quantized MoE lanes.

Current evidence snapshot:

- SGLang `origin/main`: `50f405816` on `2026-05-14`
- sgl-cookbook `origin/main`: `7b5bd9c` on `2026-05-01`
- Main runtime: `python/sglang/srt/models/deepseek_v4.py`
- Main MTP runtime: `python/sglang/srt/models/deepseek_v4_nextn.py`
- Main attention backend: `python/sglang/srt/layers/attention/deepseek_v4_backend.py`
- DSV4-specific attention / memory code:
  `python/sglang/srt/layers/attention/dsv4/*`,
  `python/sglang/srt/mem_cache/deepseek_v4_*`
- DSV4 JIT and C++ kernels:
  `python/sglang/jit_kernel/deepseek_v4.py`,
  `python/sglang/jit_kernel/csrc/deepseek_v4/*`,
  `python/sglang/jit_kernel/dsv4/*`
- MegaMoE path: `python/sglang/srt/layers/moe/mega_moe.py`
- Server hook surface: `python/sglang/srt/arg_groups/deepseek_v4_hook.py`
- Main docs: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`
- Command generator: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`
- Diff-reviewed PRs through the first V4 landing: #23605, #23617, #23628, #23622, #23634, #23684, #23689, #23690, #23691, #23697, #23698, #23725, #23737, #23742, #23756, #23776, #23787, #23810, #23817, #23832, #23883
- New source-reviewed follow-up tracks to account for before new work:
  #24367, #24775, #24793, #24816, #24890, #24897, #24925, #24949,
  #24986, #25001, #25052, #25152, #25243

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
- The latest mainline added more DSV4-specific runtime than the first landing:
  MHC pipeline optimization with DeepGEMM, fused norm, and fused hc_head
  (#24775); Tokenspeed MLA prefill/decode kernels for Blackwell FP8 KV cache
  (#24925); DeepSeek-V4-Pro shared-expert TP=1 handling (#24949); KV
  Compression V2 (#24890); fused SiLU+clamp+FP8 quant (#24897); FlashInfer
  SM90 CUTLASS MXFP4 MoE W4A16 (#24816); Hopper W4(MXFP4)A16 support
  (#24986); W4A4 MegaMoE (#25052); B300 Pro accuracy-verified serving config
  docs (#24367); H200 FP8 Flash max-throughput `SGLANG_JIT_DEEPGEMM_PRECOMPILE=0`
  guidance (#25152); and H100 deployment commands (#25243).
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
- Quantized runtime smoke for W4A4 MegaMoE, W4(MXFP4)A16 Hopper, FlashInfer
  SM90 CUTLASS MXFP4 MoE, and the Blackwell Tokenspeed MLA FP8-KV path when
  those feature gates or checkpoints are touched.

## References

- `references/pr-history.md`: diff-reviewed DeepSeek-V4 PR cards.
