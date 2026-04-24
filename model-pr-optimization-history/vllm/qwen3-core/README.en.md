# vLLM Qwen3 Core Support and PR History

This note tracks the vLLM runtime, key PRs, and remaining risk areas for Qwen3 Core.

- Status: supported on current mainline

## Key Conclusions

- Qwen3 and Qwen3MoE are first-class mainline runtimes in vLLM.
- The highest-risk regressions show up in quantized MoE weight loading, packed-module mappings, and embedding / reranker special cases.

## Main Runtime Surfaces

- `vllm/vllm/model_executor/models/qwen3.py`
- `vllm/vllm/model_executor/models/qwen3_moe.py`
- `vllm/vllm/model_executor/models/voyage.py`

## Landed PRs

- [#15289](https://github.com/vllm-project/vllm/pull/15289) `Add Qwen3 and Qwen3MoE`: Initial Qwen3 dense and MoE support landed here.
- [#19260](https://github.com/vllm-project/vllm/pull/19260) `Support Qwen3 Embedding & Reranker`: Extended the family to bidirectional embedding / reranker models.
- [#19598](https://github.com/vllm-project/vllm/pull/19598) `Skip loading extra parameters for modelopt Qwen3 MoE model`: Fixed a concrete ModelOpt launch failure on Qwen3 MoE.
- [#22017](https://github.com/vllm-project/vllm/pull/22017) `KeyError for Qwen3-MoE with GPTQ on ROCm`: Closed a GPTQ loading failure in the Qwen3 MoE path.
- [#22785](https://github.com/vllm-project/vllm/pull/22785) `Fix GGUF loader for Qwen3 MoE`: Made the Qwen3 MoE loader accept GGUF weights again.
- [#23490](https://github.com/vllm-project/vllm/pull/23490) `Fix Qwen3 MoE GPTQ inference`: Patched runtime correctness after GPTQ startup succeeded.
- [#26485](https://github.com/vllm-project/vllm/pull/26485) `Add EAGLE-3 Speculative Decoding Support for Qwen3 MoE`: Enabled the draft-model path on top of the base Qwen3 MoE runtime.

## Open PR Radar

- No pinned open PR here; re-run PR search before claiming new support.

## Matching Skill

- `skills/model-optimization/vllm/vllm-qwen3-core-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-qwen3-core-optimization/references/pr-history.md`
