---
name: vllm-qwen3-next-optimization
description: PR-backed optimization manual for Qwen3-Next in vLLM. Use when an engineer needs to audit, debug, extend, or document Qwen3-Next GDN attention, MTP, packed module naming, PP, and cross-hardware tuned MoE configuration in vLLM.
---

# vLLM Qwen3-Next Optimization

## Overview

This skill covers Qwen3-Next GDN attention, MTP, packed module naming, PP, and cross-hardware tuned MoE configuration in vLLM.

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Support status: supported on current mainline
- Canonical PR notes: `references/pr-history.md`
- History mirrors: `model-pr-optimization-history/vllm/qwen3-next/README.zh.md` and `README.en.md`

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.

## Runtime Surfaces

- `vllm/vllm/model_executor/models/qwen3_next.py`
- `vllm/vllm/model_executor/models/qwen3_next_mtp.py`

## Current Main Summary

- Qwen3-Next is its own runtime family because of Gated DeltaNet attention and its MTP path.
- The practical risks are PP, MTP varlen handling, quantized shared-expert naming, and GDN-specific CUDA graph bugs.

## Key Landed PRs

- [#24709](https://github.com/vllm-project/vllm/pull/24709) `Fix Qwen3-Next PP`: Corrected pipeline-parallel execution on Qwen3-Next.
- [#24957](https://github.com/vllm-project/vllm/pull/24957) `Fix the varlen issue in qwen3-next MTP implementation`: Removed a concrete MTP correctness bug on variable-length batches.
- [#24960](https://github.com/vllm-project/vllm/pull/24960) `Add prefixes to shared_expert in qwen3-next`: Fixed ignored-parameter and quantized weight loading for shared experts.
- [#25743](https://github.com/vllm-project/vllm/pull/25743) `Fix cuda graph capture bug in GDN metadata and a stride bug`: Stabilized GDN execution under CUDA graphs.
- [#31722](https://github.com/vllm-project/vllm/pull/31722) `Speed-up of GDN attention decode part`: Improved decode throughput on the GDN attention path.
- [#33657](https://github.com/vllm-project/vllm/pull/33657) `Initial support for GDN attention on Qwen3-next/Qwen3.5 (XPU)`: Extended the family beyond CUDA with XPU GDN coverage.

## Open Optimization Items

- Re-run PR search before claiming new support beyond the checked mainline commit.

## Validation Lanes

- Qwen3-Next PP and MTP correctness.
- GDN decode throughput / CUDA graph capture.
- Quantized shared-expert loading.
- XPU GDN fallback if relevant.
