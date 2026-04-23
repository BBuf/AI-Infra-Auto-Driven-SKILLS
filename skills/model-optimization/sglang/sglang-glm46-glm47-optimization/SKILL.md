---
name: sglang-glm46-glm47-optimization
description: PR-backed and current-main optimization manual for GLM-4.6, GLM-4.6V-adjacent text paths, GLM-4.7, and GLM-4.7-Flash in SGLang. Use when Codex needs to recover, extend, or audit GLM shared-expert fusion, dual-stream MoE GEMM overlap, GLM-4.7 tool parser, NVFP4/MTP, GLM4-MoE-Lite/Flash loading, AMD/NPU validation, or GLM-4.6/4.7 cookbook recipes.
---

# SGLang GLM-4.6/4.7 Optimization

## Overview

GLM-4.6 and GLM-4.7 share enough GLM4 MoE machinery to live in one skill, but this lane is still separate from GLM-4.5 and GLM-5 because it has its own shared-expert fusion, GLM-4.7 tool parser, GLM-4.7-Flash/lite loading, NVFP4/MTP, and AMD/NPU enablement.

Current evidence snapshot:

- SGLang `origin/main`: `b3e6cf60a` on `2026-04-22`
- sgl-cookbook `origin/main`: `816bad5` on `2026-04-21`
- Manual PR diff review date: `2026-04-23`
- Three-pass completeness audit: `model-pr-optimization-history/sglang/audits/qwen-glm-three-pass-audit-2026-04-23.md`
- Runtime files: `glm4_moe.py`, `glm4_moe_lite.py`, `glm4_moe_nextn.py`
- Parser files: `glm47_moe_detector.py`, `glm4_moe_detector.py`
- Docs/snippets: GLM-4.6, GLM-4.7, GLM-4.7-Flash cookbook pages and deployment snippets

## PR Dossier Rule

When producing or updating GLM-4.6/4.7 optimization documentation, every cited PR must be read from its GitHub diff before writing. Do not fill PR motivation, implementation, or code snippets from title search, a script-generated table, or a shallow summary.

For each cited PR, include:

- PR link, state, merge commit when merged, and diff stats.
- Motivation in concrete runtime terms.
- Key implementation path and why it fixes or optimizes the model.
- At least one code excerpt from the actual diff.
- Reviewed files.
- Validation implications.

Use `sglang-model-pr-diff-dossier` when producing new model PR-history docs.

## Before You Change Anything

Capture:

- model: GLM-4.6, GLM-4.7, or GLM-4.7-Flash
- whether lite/flash config is used
- quantization: BF16, FP8, NVFP4, compressed-tensors
- MTP enabled or disabled
- tool parser: GLM-4.7 parser vs GLM4 parser
- MoE backend: shared-expert fusion, dual-stream overlap, AITER, DeepEP, fused MoE
- hardware lane: NVIDIA, AMD, NPU

## Core Principle

GLM-4.6/4.7 optimization is MoE fusion plus parser/loading correctness, with MTP and hardware backend work layered on top.

- GLM-4.6 introduced shared-expert fusion and MoE GEMM overlap work.
- GLM-4.7 adds a new parser lane, NVFP4/MTP, and Flash/lite import formats.
- GLM-4.6V/4.7V visual issues belong to the VLM/OCR skill unless the shared text MoE class is the root cause.

## Main Runtime Surfaces

- `python/sglang/srt/models/glm4_moe.py`
- `python/sglang/srt/models/glm4_moe_lite.py`
- `python/sglang/srt/models/glm4_moe_nextn.py`
- `python/sglang/srt/function_call/glm47_moe_detector.py`
- `python/sglang/srt/function_call/glm4_moe_detector.py`
- `docs_new/cookbook/autoregressive/GLM/GLM-4.6.mdx`
- `docs_new/cookbook/autoregressive/GLM/GLM-4.7.mdx`
- `docs_new/cookbook/autoregressive/GLM/GLM-4.7-Flash.mdx`
- `test/registered/8-gpu-models/test_glm_46.py`

## Optimization Order

1. Verify GLM-4.6/4.7 BF16 output.
2. Validate parser behavior and chat template.
3. Validate Flash/lite import and quantized metadata.
4. Enable MTP and NextN only after base loading is stable.
5. Tune shared-expert fusion and dual-stream overlap.
6. Run AMD/NPU lanes and update cookbook.

## Key Diff-Reviewed PR Trails

Read `references/pr-history.md` before making a code or documentation claim. Important trails:

- Parser/tool-call correctness: `#12456`, `#13989`, `#15333`, `#15520`, `#15753`, `#15754`, `#20543`, open `#11951`, open `#23067`.
- Shared MoE optimization: `#13786`, `#13873`, `#14668`, `#21660`, `#21851`.
- GLM-4.7 MTP/NextN/quantization: `#17166`, `#19246`, open `#22315`, merged `#22823`.
- GLM-4.7-Flash/Lite: `#17247`, `#21851`, `#22509`, `#22720`, open `#19040`, open `#19106`.
- AMD/NPU backend: `#21403`, `#21534`, `#19246`, `#22509`, open `#17869`, open `#18930`, open `#22801`.

## Open PRs to Track

- [#11951](https://github.com/sgl-project/sglang/pull/11951): GLM4.6 tool-call streaming parser.
- [#17869](https://github.com/sgl-project/sglang/pull/17869): NPU GLM-4.7-Flash.
- [#18930](https://github.com/sgl-project/sglang/pull/18930): AMD MTP tests for GLM-4.7.
- [#19040](https://github.com/sgl-project/sglang/pull/19040): `Glm4MoeLiteConfig` and A2A MoE for GLM-4.7-Flash.
- [#19106](https://github.com/sgl-project/sglang/pull/19106): GLM4 MoE Lite compressed-tensors.
- [#22315](https://github.com/sgl-project/sglang/pull/22315): EAGLE accept length due draft quant config.
- [#22801](https://github.com/sgl-project/sglang/pull/22801): NPU dual-stream/DeepEP GLM-4.7-Flash.
- [#23067](https://github.com/sgl-project/sglang/pull/23067): `continue_final_message` kwargs for `glm45` reasoning parser.

## Validation Lanes

- GLM-4.6 8-GPU registered test.
- GLM-4.7/Flash parser and tool-call streaming.
- NVFP4/MTP target/draft.
- AMD/NPU accuracy and perf.

## References

- `references/pr-history.md`: manually diff-reviewed PR cards.
- `references/playbook.md`: symptom map, validation lanes, and production toggle checklist.
