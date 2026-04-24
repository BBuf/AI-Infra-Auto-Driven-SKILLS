# vLLM DeepSeek V3.2 支持与 PR 历史

本文记录 vLLM 中与 DeepSeek V3.2 相关的模型支持、关键 PR、以及仍需持续跟踪的风险点。

- 状态: 当前 mainline 已支持

## 核心结论

- DeepSeek V3.2 is the major vLLM fork of the DeepSeek line because it adds sparse MLA and indexer logic.
- Most regressions land in the indexer builder, tokenizer / parser, and specialized decode kernels rather than the generic V3 loader.

## 主要代码面

- `vllm/vllm/model_executor/models/deepseek_v2.py`
- `vllm/vllm/model_executor/layers/rotary_embedding/mrope.py`

## 已合入 PR

- [#25896](https://github.com/vllm-project/vllm/pull/25896) `Support DeepSeek-V3.2`：Landed the initial V3.2 model registration, sparse-attention runtime, and benchmark hooks.
- [#25999](https://github.com/vllm-project/vllm/pull/25999) `Support indexer prefill chunking`：Made the V3.2 sparse indexer work with chunked prefill instead of eager-only behavior.
- [#26670](https://github.com/vllm-project/vllm/pull/26670) `Add AMD GPU support on DeepSeek v3.2 and SparseMLA`：Opened the ROCm SparseMLA lane for V3.2 deployments.
- [#29848](https://github.com/vllm-project/vllm/pull/29848) `Add DeepSeek-V3.2 tool parser`：Added the parser surface that cookbook-style V3.2 reasoning deployments depend on.
- [#33090](https://github.com/vllm-project/vllm/pull/33090) `Fix DeepseekV32 `AssertionError: num_kv_heads == 1``：Removed a hard failure triggered by newer V3.2 attention shapes.
- [#37421](https://github.com/vllm-project/vllm/pull/37421) `Persistent TopK scheduler for DeepSeek-V3.2 decode`：Modernized the decode scheduler with a CUDAGraph-safe persistent TopK kernel.

## Open PR 雷达

- 暂无固定 open PR；需要在声称新支持前重新搜索。

## 配套 skill

- `skills/model-optimization/vllm/vllm-deepseek-v32-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-deepseek-v32-optimization/references/pr-history.md`
