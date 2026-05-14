---
name: vllm-deepseek-v32-optimization
description: PR-backed optimization manual for DeepSeek V3.2 in vLLM. Use when an engineer needs to audit, debug, extend, or document DeepSeek V3.2 sparse-MLA / DSA runtime, indexer, tool parser, MTP fallback, and long-context decode kernels in vLLM.
---

# vLLM DeepSeek V3.2 Optimization

## Overview

This skill covers DeepSeek V3.2 sparse-MLA / DSA runtime, indexer, tool parser, MTP fallback, and long-context decode kernels in vLLM.

Evidence snapshot:

- vLLM `origin/main`: `f3d536059` on `2026-05-15`
- Support status: supported on current mainline
- Latest relevant follow-ups: `#41217` further optimizes ROCm sparse MLA,
  `#41835` enables TP4 AITER MLA, and `#42062` enables the ROCm AITER/Gluon
  paged-MQA logits path on gfx950 / MI355X sparse-MLA shapes.
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
- On AMD gfx950, sparse-MLA traces should now be checked against
  `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` before treating paged-MQA
  logits as a missing kernel.

## Key Landed PRs

- [#25896](https://github.com/vllm-project/vllm/pull/25896) `Support DeepSeek-V3.2`: Landed the initial V3.2 model registration, sparse-attention runtime, and benchmark hooks.
- [#25999](https://github.com/vllm-project/vllm/pull/25999) `Support indexer prefill chunking`: Made the V3.2 sparse indexer work with chunked prefill instead of eager-only behavior.
- [#26670](https://github.com/vllm-project/vllm/pull/26670) `Add AMD GPU support on DeepSeek v3.2 and SparseMLA`: Opened the ROCm SparseMLA lane for V3.2 deployments.
- [#29848](https://github.com/vllm-project/vllm/pull/29848) `Add DeepSeek-V3.2 tool parser`: Added the parser surface that cookbook-style V3.2 reasoning deployments depend on.
- [#33090](https://github.com/vllm-project/vllm/pull/33090) `Fix DeepseekV32 `AssertionError: num_kv_heads == 1``: Removed a hard failure triggered by newer V3.2 attention shapes.
- [#37421](https://github.com/vllm-project/vllm/pull/37421) `Persistent TopK scheduler for DeepSeek-V3.2 decode`: Modernized the decode scheduler with a CUDAGraph-safe persistent TopK kernel.
- [#41217](https://github.com/vllm-project/vllm/pull/41217) `dsv3.2 further optimization`: Updates ROCm sparse-MLA indexer and AITER sparse backend paths.
- [#41835](https://github.com/vllm-project/vllm/pull/41835) `Enable V3.2 TP4 AITER MLA`: Enables AITER MLA for TP4 DeepSeek-V3.2 on ROCm.
- [#42062](https://github.com/vllm-project/vllm/pull/42062) `Enable gluon paged MQA logits on gfx950`: Enables the ROCm/AITER paged-MQA logits path for MI355X-class sparse MLA.

## Open Optimization Items

- Re-run PR search before claiming new support beyond the checked mainline commit.

## Validation Lanes

- SparseMLA prefill plus decode on CUDA and ROCm.
- Chunked prefill with the V3.2 indexer enabled.
- Tool parser on long reasoning responses.
- MTP / eager-fallback checks where sparse kernels are unavailable.
