---
name: vllm-internvl35-optimization
description: PR-backed optimization manual for InternVL3.5 in vLLM. Use when an engineer needs to audit, debug, extend, or document InternVL3.5 multimodal processor, video support, ViT DP / torch.compile, and backend compatibility.
---

# vLLM InternVL3.5 Optimization

## Overview

This skill covers InternVL3.5 multimodal processor, video support, ViT DP / torch.compile, and backend compatibility.

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Canonical PR notes: `references/pr-history.md`
- History mirrors: `model-pr-optimization-history/vllm/internvl35/README.zh.md` and `README.en.md`

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.

## Runtime Surfaces

- `vllm/vllm/model_executor/models/internvl.py`

## Current Main Summary

- InternVL3.5 is mostly a processor / encoder / video problem in vLLM.
- Video handling, native HF loading, and backend compatibility dominate the risk surface.

## Key Landed PRs

- [#6514](https://github.com/vllm-project/vllm/pull/6514) `Initialize support for InternVL2 series models`: Historical base for current InternVL runtime code.
- [#18499](https://github.com/vllm-project/vllm/pull/18499) `Initialize video input support for InternVL models`: Added video processing to the family.
- [#23658](https://github.com/vllm-project/vllm/pull/23658) `Enable video support for InternVL3.5 models`: Carried video support into the 3.5 checkpoints.
- [#23742](https://github.com/vllm-project/vllm/pull/23742) `Enable native HF format InternVL support`: Removed reliance on ad hoc checkpoint rewrites.
- [#38049](https://github.com/vllm-project/vllm/pull/38049) `Add torch.compile support for InternVL vision encoder`: Modernized the encoder execution path.

## Validation Lanes

- Startup on the current vLLM mainline checkpoint.
- Re-run the parser, quantization, MTP, or multimodal lane that matches this family.
- Re-check tests after touching loader or processor code.
