---
name: vllm-deepseek-v32-optimization
description: PR-backed optimization manual for DeepSeek V3.2 in vLLM. Use when Codex needs to audit, debug, extend, or document DeepSeek V3.2 sparse-MLA / DSA runtime, indexer, tool parser, MTP fallback, and long-context decode kernels in vLLM.
---

# vLLM DeepSeek V3.2 Optimization

## Overview

This skill covers DeepSeek V3.2 sparse-MLA / DSA runtime, indexer, tool parser, MTP fallback, and long-context decode kernels in vLLM.

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Support status: supported on current mainline
- Canonical PR notes: `references/pr-history.md`
- History mirrors: `model-pr-optimization-history/vllm/deepseek-v32/README.zh.md` and `README.en.md`

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.

## Runtime Surfaces

- `vllm/vllm/model_executor/models/deepseek_v2.py`
- `vllm/vllm/model_executor/layers/rotary_embedding/mrope.py`

## Current Main Summary

- DeepSeek V3.2 is the major vLLM fork of the DeepSeek line because it adds sparse MLA and indexer logic.
- Most regressions land in the indexer builder, tokenizer / parser, and specialized decode kernels rather than the generic V3 loader.

## Key Landed PRs

- [#25896](https://github.com/vllm-project/vllm/pull/25896) `Support DeepSeek-V3.2`: Landed the initial V3.2 model registration, sparse-attention runtime, and benchmark hooks.
- [#25999](https://github.com/vllm-project/vllm/pull/25999) `Support indexer prefill chunking`: Made the V3.2 sparse indexer work with chunked prefill instead of eager-only behavior.
- [#26670](https://github.com/vllm-project/vllm/pull/26670) `Add AMD GPU support on DeepSeek v3.2 and SparseMLA`: Opened the ROCm SparseMLA lane for V3.2 deployments.
- [#29848](https://github.com/vllm-project/vllm/pull/29848) `Add DeepSeek-V3.2 tool parser`: Added the parser surface that cookbook-style V3.2 reasoning deployments depend on.
- [#33090](https://github.com/vllm-project/vllm/pull/33090) `Fix DeepseekV32 `AssertionError: num_kv_heads == 1``: Removed a hard failure triggered by newer V3.2 attention shapes.
- [#37421](https://github.com/vllm-project/vllm/pull/37421) `Persistent TopK scheduler for DeepSeek-V3.2 decode`: Modernized the decode scheduler with a CUDAGraph-safe persistent TopK kernel.

## Open Radar

- Re-run PR search before claiming new support beyond the checked mainline commit.

## Validation Lanes

- SparseMLA prefill plus decode on CUDA and ROCm.
- Chunked prefill with the V3.2 indexer enabled.
- Tool parser on long reasoning responses.
- MTP / eager-fallback checks where sparse kernels are unavailable.
