---
name: vllm-qwen3-core-optimization
description: PR-backed optimization manual for Qwen3 Core in vLLM. Use when an engineer needs to audit, debug, extend, or document Qwen3 dense, Qwen3 MoE, embeddings/rerankers, GGUF/GPTQ/ModelOpt quant paths, and Eagle3 speculative decoding in vLLM.
---

# vLLM Qwen3 Core Optimization

## Overview

This skill covers Qwen3 dense, Qwen3 MoE, embeddings/rerankers, GGUF/GPTQ/ModelOpt quant paths, and Eagle3 speculative decoding in vLLM.

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Support status: supported on current mainline
- Canonical PR notes: `references/pr-history.md`
- History mirrors: `model-pr-optimization-history/vllm/qwen3-core/README.zh.md` and `README.en.md`

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.

## Runtime Surfaces

- `vllm/vllm/model_executor/models/qwen3.py`
- `vllm/vllm/model_executor/models/qwen3_moe.py`
- `vllm/vllm/model_executor/models/voyage.py`

## Current Main Summary

- Qwen3 and Qwen3MoE are first-class mainline runtimes in vLLM.
- The highest-risk regressions show up in quantized MoE weight loading, packed-module mappings, and embedding / reranker special cases.

## Key Landed PRs

- [#15289](https://github.com/vllm-project/vllm/pull/15289) `Add Qwen3 and Qwen3MoE`: Initial Qwen3 dense and MoE support landed here.
- [#19260](https://github.com/vllm-project/vllm/pull/19260) `Support Qwen3 Embedding & Reranker`: Extended the family to bidirectional embedding / reranker models.
- [#19598](https://github.com/vllm-project/vllm/pull/19598) `Skip loading extra parameters for modelopt Qwen3 MoE model`: Fixed a concrete ModelOpt launch failure on Qwen3 MoE.
- [#22017](https://github.com/vllm-project/vllm/pull/22017) `KeyError for Qwen3-MoE with GPTQ on ROCm`: Closed a GPTQ loading failure in the Qwen3 MoE path.
- [#22785](https://github.com/vllm-project/vllm/pull/22785) `Fix GGUF loader for Qwen3 MoE`: Made the Qwen3 MoE loader accept GGUF weights again.
- [#23490](https://github.com/vllm-project/vllm/pull/23490) `Fix Qwen3 MoE GPTQ inference`: Patched runtime correctness after GPTQ startup succeeded.
- [#26485](https://github.com/vllm-project/vllm/pull/26485) `Add EAGLE-3 Speculative Decoding Support for Qwen3 MoE`: Enabled the draft-model path on top of the base Qwen3 MoE runtime.

## Open Optimization Items

- Re-run PR search before claiming new support beyond the checked mainline commit.

## Validation Lanes

- Dense and MoE BF16 launch.
- Embedding / reranker special-case loading.
- GPTQ / GGUF / ModelOpt startup on Qwen3 MoE.
- EAGLE-3 acceptance-rate sanity on supported checkpoints.
