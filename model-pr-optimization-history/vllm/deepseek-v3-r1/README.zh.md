# vLLM DeepSeek V3 / R1 支持与 PR 历史

本文记录 vLLM 中与 DeepSeek V3 / R1 相关的模型支持、关键 PR、以及仍需持续跟踪的风险点。

- 状态: 当前 mainline 已支持

## 核心结论

- `DeepseekV2ForCausalLM` / `DeepseekV3ForCausalLM` remain the shared runtime for V3 and R1.
- The highest-risk regressions cluster around packed module mapping, quantized MLA/MoE weight loading, LoRA, and MTP draft paths.
- R1 validation should split BF16, FP8/ModelOpt, and compressed-tensors or ROCm lanes.

## 主要代码面

- `vllm/vllm/model_executor/models/deepseek_v2.py`
- `vllm/vllm/model_executor/models/deepseek_eagle.py`
- `vllm/vllm/model_executor/models/deepseek_eagle3.py`
- `vllm/vllm/model_executor/models/deepseek_mtp.py`

## 已合入 PR

- [#22352](https://github.com/vllm-project/vllm/pull/22352) `Add missing `packed_modules_mapping` to `DeepseekV2ForCausalLM``：Fixed quantized and packed-weight loading for DeepSeek V2/V3/R1 style checkpoints.
- [#23971](https://github.com/vllm-project/vllm/pull/23971) `Add LoRA support for DeepSeek models (V2, V3, R1-0528)`：Enabled adapter injection on the DeepSeek family rather than only base dense models.
- [#29545](https://github.com/vllm-project/vllm/pull/29545) `Fix DeepSeek R1 MTP weight loading`：Hardened R1 NextN / MTP draft loading after launch failures on draft weights.
- [#36247](https://github.com/vllm-project/vllm/pull/36247) `Fix compressed-tensors quantization failure for DeepSeek-R1 on MI300x`：Closed a production ROCm gap for compressed-tensors DeepSeek-R1 deployment.

## Open PR 雷达

- 暂无固定 open PR；需要在声称新支持前重新搜索。

## 配套 skill

- `skills/model-optimization/vllm/vllm-deepseek-v3-r1-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-deepseek-v3-r1-optimization/references/pr-history.md`
