# vLLM Mistral Small 4 支持与 PR 历史

本文记录 vLLM 中与 Mistral Small 4 相关的模型支持、关键 PR、以及 cookbook 对应的落点。

- 状态: 当前 mainline 已支持

## 核心结论

- Mistral Small 4 sits on top of the larger Mistral Large 3 / Ministral runtime work.
- MoE execution and multimodal projector behavior are the main risk areas.

## 主要代码面

- `vllm/vllm/model_executor/models/mistral_large_3.py`
- `vllm/vllm/model_executor/models/mistral_large_3_eagle.py`
- `vllm/vllm/model_executor/models/mistral3.py`

## 已合入 PR

- [#29757](https://github.com/vllm-project/vllm/pull/29757) `Add Mistral Large 3 and Ministral 3`：Landed the runtime family that Mistral Small 4 deployments build on in vLLM.
- [#33174](https://github.com/vllm-project/vllm/pull/33174) `Add support for Mistral Large 3 inference with Flashinfer MoE`：Improved the practical MoE serving path for the same family.

## 配套 skill

- `skills/model-optimization/vllm/vllm-mistral-small-4-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-mistral-small-4-optimization/references/pr-history.md`

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `vllm-project/vllm` 的 Pull Request API 和文件级 patch 重新审计 `Mistral Small 4 / Ministral 3`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2025-11-30 | [#29757](https://github.com/vllm-project/vllm/pull/29757) | merged | Add Mistral Large 3 and Ministral 3 | model wrapper, attention/backend, MoE/router, quantization, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/mistral_large_3_eagle.py`, `tests/tokenizers_/test_mistral.py`, `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_H200,dtype=fp8_w8a8.json` |
| 2026-01-27 | [#33174](https://github.com/vllm-project/vllm/pull/33174) | merged | Add support for Mistral Large 3 inference with Flashinfer MoE | model wrapper, attention/backend, MoE/router, quantization, kernel, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200,dtype=fp8_w8a8.json`, `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200.json`, `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_GB200,dtype=fp8_w8a8.json` |

### 逐 PR 代码 diff 阅读记录

### PR #29757 - Add Mistral Large 3 and Ministral 3

- 链接：https://github.com/vllm-project/vllm/pull/29757
- 状态/时间：`merged`，created 2025-11-30, merged 2025-12-02；作者 `juliendenize`。
- 代码 diff 已读范围：`16` 个文件，`+724/-30`；代码面：model wrapper, attention/backend, MoE/router, quantization, scheduler/runtime, tests/benchmarks, docs/config；关键词：config, spec, kv, attention, eagle, expert, fp8, moe, quant, test。
- 代码 diff 细节：
  - `vllm/model_executor/models/mistral_large_3_eagle.py` added +165/-0 (165 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: EagleMistralLarge3Model, __init__, forward, EagleMistralLarge3ForCausalLM
  - `tests/tokenizers_/test_mistral.py` modified +151/-7 (158 lines); hunk: ],; def test_decode(; 符号: test_prepare_apply_chat_template_tools_and_messages, test_decode, test_decode_empty, test_decode_int
  - `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_H200,dtype=fp8_w8a8.json` added +146/-0 (146 lines); hunk: +{
  - `vllm/transformers_utils/configs/mistral.py` modified +62/-12 (74 lines); hunk: def adapt_config_dict(; def _remap_general_mistral_args(config: dict) -> dict:; 符号: adapt_config_dict, _remap_general_mistral_args, _remap_mistral_quantization_args, _remap_mistral_audio_args
  - `vllm/model_executor/models/deepseek_v2.py` modified +59/-7 (66 lines); hunk: def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:; def __init__(; 符号: yarn_get_mscale, _get_llama_4_scaling, DeepseekV2Attention, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/mistral_large_3_eagle.py`, `tests/tokenizers_/test_mistral.py`, `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_H200,dtype=fp8_w8a8.json`；patch 关键词为 config, spec, kv, attention, eagle, expert。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/mistral_large_3_eagle.py`, `tests/tokenizers_/test_mistral.py`, `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_H200,dtype=fp8_w8a8.json` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #33174 - Add support for Mistral Large 3 inference with Flashinfer MoE

- 链接：https://github.com/vllm-project/vllm/pull/33174
- 状态/时间：`merged`，created 2026-01-27, merged 2026-01-31；作者 `dbari`。
- 代码 diff 已读范围：`16` 个文件，`+1104/-31`；代码面：model wrapper, attention/backend, MoE/router, quantization, kernel, scheduler/runtime, tests/benchmarks, docs/config；关键词：config, moe, triton, fp8, benchmark, expert, quant, topk。
- 代码 diff 细节：
  - `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200,dtype=fp8_w8a8.json` added +147/-0 (147 lines); hunk: +{
  - `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200.json` added +147/-0 (147 lines); hunk: +{
  - `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_GB200,dtype=fp8_w8a8.json` added +147/-0 (147 lines); hunk: +{
  - `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json` added +147/-0 (147 lines); hunk: +{
  - `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_H200.json` added +147/-0 (147 lines); hunk: +{
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200,dtype=fp8_w8a8.json`, `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200.json`, `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_GB200,dtype=fp8_w8a8.json`；patch 关键词为 config, moe, triton, fp8, benchmark, expert。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200,dtype=fp8_w8a8.json`, `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200.json`, `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_GB200,dtype=fp8_w8a8.json` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：2；open PR 数：0。
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
