# vLLM DeepSeek V3 / R1 Support and PR History

This note tracks the vLLM runtime, key PRs, and remaining risk areas for DeepSeek V3 / R1.

- Status: supported on current mainline

## Key Conclusions

- `DeepseekV2ForCausalLM` / `DeepseekV3ForCausalLM` remain the shared runtime for V3 and R1.
- The highest-risk regressions cluster around packed module mapping, quantized MLA/MoE weight loading, LoRA, and MTP draft paths.
- R1 validation should split BF16, FP8/ModelOpt, and compressed-tensors or ROCm lanes.

## Main Runtime Surfaces

- `vllm/vllm/model_executor/models/deepseek_v2.py`
- `vllm/vllm/model_executor/models/deepseek_eagle.py`
- `vllm/vllm/model_executor/models/deepseek_eagle3.py`
- `vllm/vllm/model_executor/models/deepseek_mtp.py`

## Landed PRs

- [#22352](https://github.com/vllm-project/vllm/pull/22352) `Add missing `packed_modules_mapping` to `DeepseekV2ForCausalLM``: Fixed quantized and packed-weight loading for DeepSeek V2/V3/R1 style checkpoints.
- [#23971](https://github.com/vllm-project/vllm/pull/23971) `Add LoRA support for DeepSeek models (V2, V3, R1-0528)`: Enabled adapter injection on the DeepSeek family rather than only base dense models.
- [#29545](https://github.com/vllm-project/vllm/pull/29545) `Fix DeepSeek R1 MTP weight loading`: Hardened R1 NextN / MTP draft loading after launch failures on draft weights.
- [#36247](https://github.com/vllm-project/vllm/pull/36247) `Fix compressed-tensors quantization failure for DeepSeek-R1 on MI300x`: Closed a production ROCm gap for compressed-tensors DeepSeek-R1 deployment.

## Open PR Radar

- No pinned open PR here; re-run PR search before claiming new support.

## Matching Skill

- `skills/model-optimization/vllm/vllm-deepseek-v3-r1-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-deepseek-v3-r1-optimization/references/pr-history.md`
