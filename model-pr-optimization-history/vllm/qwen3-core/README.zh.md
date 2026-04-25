# vLLM Qwen3 Core 支持与 PR 历史

本文记录 vLLM 中与 Qwen3 Core 相关的模型支持、关键 PR、以及仍需持续跟踪的风险点。

- 状态: 当前 mainline 已支持

## 核心结论

- Qwen3 and Qwen3MoE are first-class mainline runtimes in vLLM.
- The highest-risk regressions show up in quantized MoE weight loading, packed-module mappings, and embedding / reranker special cases.

## 主要代码面

- `vllm/vllm/model_executor/models/qwen3.py`
- `vllm/vllm/model_executor/models/qwen3_moe.py`
- `vllm/vllm/model_executor/models/voyage.py`

## 已合入 PR

- [#15289](https://github.com/vllm-project/vllm/pull/15289) `Add Qwen3 and Qwen3MoE`：Initial Qwen3 dense and MoE support landed here.
- [#19260](https://github.com/vllm-project/vllm/pull/19260) `Support Qwen3 Embedding & Reranker`：Extended the family to bidirectional embedding / reranker models.
- [#19598](https://github.com/vllm-project/vllm/pull/19598) `Skip loading extra parameters for modelopt Qwen3 MoE model`：Fixed a concrete ModelOpt launch failure on Qwen3 MoE.
- [#22017](https://github.com/vllm-project/vllm/pull/22017) `KeyError for Qwen3-MoE with GPTQ on ROCm`：Closed a GPTQ loading failure in the Qwen3 MoE path.
- [#22785](https://github.com/vllm-project/vllm/pull/22785) `Fix GGUF loader for Qwen3 MoE`：Made the Qwen3 MoE loader accept GGUF weights again.
- [#23490](https://github.com/vllm-project/vllm/pull/23490) `Fix Qwen3 MoE GPTQ inference`：Patched runtime correctness after GPTQ startup succeeded.
- [#26485](https://github.com/vllm-project/vllm/pull/26485) `Add EAGLE-3 Speculative Decoding Support for Qwen3 MoE`：Enabled the draft-model path on top of the base Qwen3 MoE runtime.

## Open PR 雷达

- 暂无固定 open PR；需要在声称新支持前重新搜索。

## 配套 skill

- `skills/model-optimization/vllm/vllm-qwen3-core-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-qwen3-core-optimization/references/pr-history.md`
