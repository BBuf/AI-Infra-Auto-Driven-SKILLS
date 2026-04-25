# vLLM DeepSeek V3.1 Support and PR History

This note tracks the vLLM runtime, key PRs, and remaining risk areas for DeepSeek V3.1.

- Status: supported on current mainline
- This family inherits the base runtime from `deepseek-v3-r1` and only records the delta here.

## Key Conclusions

- V3.1 mostly reuses the base V3 runtime and adds parser plus scale-format correctness work.
- The practical blast radius is in tool calling, DeepGEMM scale handling, and reasoning-parser behavior.

## Main Runtime Surfaces

- `vllm/vllm/model_executor/models/deepseek_v2.py`
- `vllm/vllm/model_executor/layers/quantization/utils/flashinfer_utils.py`

## Landed PRs

- [#23454](https://github.com/vllm-project/vllm/pull/23454) `Support DeepSeek-V3.1 tool call`: Added the first V3.1-specific tool-call parser surface to vLLM.
- [#23666](https://github.com/vllm-project/vllm/pull/23666) `Add Hopper DeepGEMM E8M0 for DeepSeekV3.1 scale_fmt`: Tuned the scale-format path used by DeepGEMM-based DeepSeek V3.1 kernels.
- [#25589](https://github.com/vllm-project/vllm/pull/25589) `Add DeepSeek-V3.1 reasoning parser`: Separated V3.1 reasoning output handling from generic DeepSeek parsing.
- [#32361](https://github.com/vllm-project/vllm/pull/32361) `Fix DeepSeek-V3.1 + DeepGEMM incompatible scale shapes`: Patched a concrete shape mismatch between newer checkpoints and DeepGEMM assumptions.

## Open PR Radar

- No pinned open PR here; re-run PR search before claiming new support.

## Matching Skill

- `skills/model-optimization/vllm/vllm-deepseek-v31-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-deepseek-v31-optimization/references/pr-history.md`
