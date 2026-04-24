# vLLM Qwen3.5 Support and PR History

This note tracks the vLLM runtime, key PRs, and remaining risk areas for Qwen3.5.

- Status: supported on current mainline

## Key Conclusions

- Qwen3.5 builds on the Qwen3-Next era work but has its own model registration and quantization details.
- The hot spots are GDN fusion, FP8/NVFP4 loading, LoRA target naming, and MoE EP precision.

## Main Runtime Surfaces

- `vllm/vllm/model_executor/models/qwen3_5.py`
- `vllm/vllm/model_executor/models/qwen3_5_mtp.py`

## Landed PRs

- [#34110](https://github.com/vllm-project/vllm/pull/34110) `Adding Support for Qwen3.5 Models`: Landed the Qwen3.5 runtime family.
- [#34697](https://github.com/vllm-project/vllm/pull/34697) `Redo Qwen3.5/Qwen3-Next GDN projector fusion`: Reworked an earlier fusion that had to be reverted.
- [#35289](https://github.com/vllm-project/vllm/pull/35289) `Fix Qwen3.5 FP8 quantization tuple shard_id weight loading`: Closed a concrete FP8 weight-loading failure.
- [#36658](https://github.com/vllm-project/vllm/pull/36658) `Add Eagle3 support for Qwen3.5`: Enabled the draft-model fast path.
- [#37975](https://github.com/vllm-project/vllm/pull/37975) `Extract GatedDeltaNetAttention into shared layer for Qwen3Next and Qwen3.5`: Reduced duplicated GDN logic across related families.
- [#39181](https://github.com/vllm-project/vllm/pull/39181) `Fix EP precision for Qwen3.5, Qwen3-Next`: Patched a serving-precision bug under expert parallelism.

## Open PR Radar

- No pinned open PR here; re-run PR search before claiming new support.

## Matching Skill

- `skills/model-optimization/vllm/vllm-qwen35-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-qwen35-optimization/references/pr-history.md`
