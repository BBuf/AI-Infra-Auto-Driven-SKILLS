---
name: vllm-hunyuan3-preview-optimization
description: PR-backed optimization manual for Hunyuan 3 Preview in vLLM. Use when an engineer needs to audit, debug, extend, or document Adjacent Hunyuan dense / OCR / VL support in vLLM relevant to Hunyuan 3 Preview planning, without a dedicated `Hunyuan3Preview` mainline alias yet.
---

# vLLM Hunyuan 3 Preview Optimization

## Overview

This skill covers Adjacent Hunyuan dense / OCR / VL support in vLLM relevant to Hunyuan 3 Preview planning, without a dedicated `Hunyuan3Preview` mainline alias yet.

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Support status: partially supported or only adjacent architectures landed on current mainline
- Canonical PR notes: `references/pr-history.md`
- History mirrors: `model-pr-optimization-history/vllm/hunyuan3-preview/README.zh.md` and `README.en.md`

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.

## Runtime Surfaces

- `vllm/vllm/model_executor/models/hunyuan_v1.py`
- `vllm/vllm/model_executor/models/hunyuan_vision.py`

## Current Main Summary

- vLLM does not currently expose a dedicated Hunyuan 3 Preview model alias.
- The closest landed evidence is the Hunyuan dense, Hunyuan OCR, and HunyuanVL / Eagle work already in tree.

## Key Landed PRs

- [#21368](https://github.com/vllm-project/vllm/pull/21368) `Add Hunyuan V1 Dense Model support`: Brought the dense Hunyuan line into vLLM mainline.
- [#29327](https://github.com/vllm-project/vllm/pull/29327) `Add HunyuanOCR support`: Extended the family to OCR workloads instead of text-only generation.
- [#33035](https://github.com/vllm-project/vllm/pull/33035) `Eagle3 support for HunyuanVL & Hunyuan`: Added speculative decoding support on top of the Hunyuan family.

## Open Optimization Items

- Re-run PR search before claiming new support beyond the checked mainline commit.

## Validation Lanes

- Do not equate Hunyuan V1 / OCR / VL with a landed Hunyuan 3 Preview runtime.
- If exact Hunyuan 3 Preview parity matters, re-check open PR and registry state first.
