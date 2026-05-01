---
name: sglang-internvl35-optimization
description: PR-backed optimization manual for InternVL3.5 in SGLang. Use when an engineer needs to audit, debug, extend, or document InternVL3.5 multimodal processor, video support, ViT DP / CUDA graph, and non-CUDA backend compatibility.
---

# SGLang InternVL3.5 Optimization

## Overview

This skill covers InternVL3.5 multimodal processor, video support, ViT DP / CUDA graph, and non-CUDA backend compatibility.

Evidence snapshot:

- SGLang mainline checked around `c122d343adb969cd9bbd1af2ca86727a11be3845`
- sgl-cookbook checked around `e88b0fd8ac5b1caa6eb42766035029220053369b`
- Canonical PR notes: `references/pr-history.md`
- History mirrors: `model-pr-optimization-history/sglang/internvl35/README.zh.md` and `README.en.md`

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.

## Runtime Surfaces

- `sglang/python/sglang/srt/models/internvl.py`

## Current Main Summary

- InternVL3.5 is mostly a processor / encoder / video problem in SGLang.
- Video handling, data-parallel vision execution, and backend compatibility dominate the risk surface.

## Key Landed PRs

- [#5350](https://github.com/sgl-project/sglang/pull/5350) `Support InternVL3`: Initial InternVL family support that later carried 3.5.
- [#13640](https://github.com/sgl-project/sglang/pull/13640) `Support Piecewise CUDA Graph for InternVL`: Added graph capture support on the encoder path.
- [#13925](https://github.com/sgl-project/sglang/pull/13925) `Support InternVL Vision Encoder Data Parallelism`: Opened the multi-GPU ViT path.
- [#15942](https://github.com/sgl-project/sglang/pull/15942) `Support Video for InternVL3_5`: Extended support to 3.5 video use cases.
- [#19127](https://github.com/sgl-project/sglang/pull/19127) `Support processor and embedding inputs for InternVL`: Hardened processor / embed input interoperability.

## Validation Lanes

- Startup on the cookbook checkpoint or example route.
- Re-run the parser, quantization, or multimodal lane that matches this family.
- Re-check registered or manual tests after touching loader or processor code.
