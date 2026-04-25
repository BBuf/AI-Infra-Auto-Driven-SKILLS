# vLLM Qwen3.5 支持与 PR 历史

本文记录 vLLM 中与 Qwen3.5 相关的模型支持、关键 PR、以及仍需持续跟踪的风险点。

- 状态: 当前 mainline 已支持

## 核心结论

- Qwen3.5 builds on the Qwen3-Next era work but has its own model registration and quantization details.
- The hot spots are GDN fusion, FP8/NVFP4 loading, LoRA target naming, and MoE EP precision.

## 主要代码面

- `vllm/vllm/model_executor/models/qwen3_5.py`
- `vllm/vllm/model_executor/models/qwen3_5_mtp.py`

## 已合入 PR

- [#34110](https://github.com/vllm-project/vllm/pull/34110) `Adding Support for Qwen3.5 Models`：Landed the Qwen3.5 runtime family.
- [#34697](https://github.com/vllm-project/vllm/pull/34697) `Redo Qwen3.5/Qwen3-Next GDN projector fusion`：Reworked an earlier fusion that had to be reverted.
- [#35289](https://github.com/vllm-project/vllm/pull/35289) `Fix Qwen3.5 FP8 quantization tuple shard_id weight loading`：Closed a concrete FP8 weight-loading failure.
- [#36658](https://github.com/vllm-project/vllm/pull/36658) `Add Eagle3 support for Qwen3.5`：Enabled the draft-model fast path.
- [#37975](https://github.com/vllm-project/vllm/pull/37975) `Extract GatedDeltaNetAttention into shared layer for Qwen3Next and Qwen3.5`：Reduced duplicated GDN logic across related families.
- [#39181](https://github.com/vllm-project/vllm/pull/39181) `Fix EP precision for Qwen3.5, Qwen3-Next`：Patched a serving-precision bug under expert parallelism.

## Open PR 雷达

- 暂无固定 open PR；需要在声称新支持前重新搜索。

## 配套 skill

- `skills/model-optimization/vllm/vllm-qwen35-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-qwen35-optimization/references/pr-history.md`
