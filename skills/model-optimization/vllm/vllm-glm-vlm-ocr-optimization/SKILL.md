---
name: vllm-glm-vlm-ocr-optimization
description: PR-backed optimization manual for GLM VLM / OCR in vLLM. Use when an engineer needs to audit, debug, extend, or document GLM-4V, GLM-4.1V, GLM-OCR, GLM visual processor, MRoPE, video, and OCR MTP behavior in vLLM.
---

# vLLM GLM VLM / OCR Optimization

## Overview

This skill covers GLM-4V, GLM-4.1V, GLM-OCR, GLM visual processor, MRoPE, video, and OCR MTP behavior in vLLM.

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Support status: supported on current mainline
- Canonical PR notes: `references/pr-history.md`
- History mirrors: `model-pr-optimization-history/vllm/glm-vlm-ocr/README.zh.md` and `README.en.md`

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.

## Runtime Surfaces

- `vllm/vllm/model_executor/models/glm4v.py`
- `vllm/vllm/model_executor/models/glm4_1v.py`
- `vllm/vllm/model_executor/models/glm_ocr.py`

## Current Main Summary

- GLM visual/OCR support in vLLM spans classic GLM4V, newer GLM4.1V, and GLM-OCR-specific processor paths.
- The main failures are processor-schema drift, MRoPE/video position handling, and OCR-specific weight or patch-merger mismatches.

## Key Landed PRs

- [#9242](https://github.com/vllm-project/vllm/pull/9242) `Add GLM-4v support`: Landed the original GLM4V multimodal model path.
- [#19331](https://github.com/vllm-project/vllm/pull/19331) `Add GLM4.1V model`: Extended the family to the newer GLM4.1V checkpoint layout and vision stack.
- [#27860](https://github.com/vllm-project/vllm/pull/27860) `Fix broken MRoPE for GLM-4.1V/GLM-4.5V`: Closed a positional-embedding bug with large practical accuracy impact on vision inputs.
- [#33005](https://github.com/vllm-project/vllm/pull/33005) `GLM-OCR with MTP Support`: Added OCR-specific draft / MTP support rather than text-only OCR loading.
- [#33350](https://github.com/vllm-project/vllm/pull/33350) `Fix broken GLM-OCR initialization`: Fixed startup failures in the GLM-OCR path after the first bring-up.
- [#37962](https://github.com/vllm-project/vllm/pull/37962) `GLM OCR Patch Merger context_dim`: Updated the patch-merger contract for newer OCR checkpoints.

## Open Optimization Items

- Re-run PR search before claiming new support beyond the checked mainline commit.

## Validation Lanes

- Single-image and multi-image GLM4V / GLM4.1V generation.
- GLM-OCR document extraction and MTP startup.
- Video / MRoPE correctness under transformers upgrades.
