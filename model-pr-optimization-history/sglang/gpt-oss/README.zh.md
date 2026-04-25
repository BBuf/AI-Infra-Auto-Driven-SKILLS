# SGLang GPT-OSS 支持与 PR 历史

本文记录 SGLang 中与 GPT-OSS 相关的模型支持、关键 PR、以及 cookbook 对应的落点。

- 状态: 当前 mainline 已支持

## 核心结论

- GPT-OSS is a flagship MoE family in SGLang.
- Quantization, expert-parallel topology, and reasoning/tool parser behavior all evolved quickly and need per-PR tracking.

## 主要代码面

- `sglang/python/sglang/srt/models/gpt_oss.py`

## 已合入 PR

- [#8843](https://github.com/sgl-project/sglang/pull/8843) `Support mxfp4 for GPT-OSS`：Added the headline quantized checkpoint path.
- [#8944](https://github.com/sgl-project/sglang/pull/8944) `Expert Parallelism for GPT-OSS`：Scaled GPT-OSS beyond pure tensor parallel.
- [#9043](https://github.com/sgl-project/sglang/pull/9043) `Implement Native GPT-OSS Tool Call Support`：Added native tool parser support instead of Harmony integration.
- [#9359](https://github.com/sgl-project/sglang/pull/9359) `Support DP attention with GPT-OSS`：Enabled larger topologies via DP attention.
- [#14920](https://github.com/sgl-project/sglang/pull/14920) `GPT-OSS Eagle v2 support`：Added speculative decoding support.
- [#18988](https://github.com/sgl-project/sglang/pull/18988) `Support fp8 online quantization for gpt-oss bf16`：Extended quantization coverage to online FP8.

## 配套 skill

- `skills/model-optimization/sglang/sglang-gpt-oss-optimization/SKILL.md`
- `skills/model-optimization/sglang/sglang-gpt-oss-optimization/references/pr-history.md`

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `sgl-project/sglang` 的 Pull Request API 和文件级 patch 重新审计 `GPT-OSS`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2025-08-06 | [#8843](https://github.com/sgl-project/sglang/pull/8843) | merged | Support mxfp4 for GPT-OSS | model wrapper, MoE/router, quantization, kernel | `python/sglang/srt/layers/quantization/mxfp4.py`, `python/sglang/srt/layers/quantization/fp4.py`, `python/sglang/srt/models/gpt_oss.py` |
| 2025-08-08 | [#8944](https://github.com/sgl-project/sglang/pull/8944) | merged | Expert Parallelism for GPT-OSS | model wrapper, MoE/router, quantization, kernel | `python/sglang/srt/layers/quantization/mxfp4.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/models/gpt_oss.py` |
| 2025-08-11 | [#9043](https://github.com/sgl-project/sglang/pull/9043) | merged | (gpt-oss, oai, chat): Remove Harmony Integration and Implement Native GPT-OSS Tool Call Support | docs/config | `python/sglang/srt/function_call/gpt_oss_detector.py`, `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/reasoning_parser.py` |
| 2025-08-19 | [#9359](https://github.com/sgl-project/sglang/pull/9359) | merged | Support DP attention with GPT-OSS | model wrapper | `python/sglang/srt/server_args.py`, `python/sglang/srt/models/gpt_oss.py` |
| 2025-12-11 | [#14920](https://github.com/sgl-project/sglang/pull/14920) | merged | Eagle: GPT-OSS Eagle v2 support | kernel, scheduler/runtime | `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/speculative/eagle_worker.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py` |
| 2026-02-18 | [#18988](https://github.com/sgl-project/sglang/pull/18988) | merged | [GPT-OSS] support fp8 online quantization for gpt-oss bf16 | quantization | `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/server_args.py` |

### 逐 PR 代码 diff 阅读记录

### PR #8843 - Support mxfp4 for GPT-OSS

- 链接：https://github.com/sgl-project/sglang/pull/8843
- 状态/时间：`merged`，created 2025-08-06, merged 2025-08-06；作者 `Ying1123`。
- 代码 diff 已读范围：`9` 个文件，`+791/-325`；代码面：model wrapper, MoE/router, quantization, kernel；关键词：config, fp4, moe, quant, triton, cuda, expert, fp8, attention, cache。
- 代码 diff 细节：
  - `python/sglang/srt/layers/quantization/mxfp4.py` added +443/-0 (443 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: _swizzle_mxfp4, _dequant_mxfp4, _dequant_mxfp4_fake, _quant_dequant_mxfp4
  - `python/sglang/srt/layers/quantization/fp4.py` modified +28/-293 (321 lines); hunk: OCP_MX_BLOCK_SIZE = 32; 符号: MxFp4Config, Mxfp4Config, __init__, __init__
  - `python/sglang/srt/models/gpt_oss.py` modified +209/-9 (218 lines); hunk: from transformers import PretrainedConfig; def __init__(; 符号: __init__, __init__, __init__, _get_default_weight_mapping
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +58/-6 (64 lines); hunk: def _load_w2(; def _load_w2(; 符号: _load_w2, _load_w2, weight_loader, _weight_loader_physical
  - `python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py` modified +25/-15 (40 lines); hunk: def triton_kernel_fused_experts(; def triton_kernel_moe_with_bias_forward(; 符号: triton_kernel_fused_experts, triton_kernel_moe_with_bias_forward, triton_kernel_moe_with_bias_forward, triton_kernel_moe_with_bias_forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/quantization/mxfp4.py`, `python/sglang/srt/layers/quantization/fp4.py`, `python/sglang/srt/models/gpt_oss.py`；patch 关键词为 config, fp4, moe, quant, triton, cuda。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/quantization/mxfp4.py`, `python/sglang/srt/layers/quantization/fp4.py`, `python/sglang/srt/models/gpt_oss.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #8944 - Expert Parallelism for GPT-OSS

- 链接：https://github.com/sgl-project/sglang/pull/8944
- 状态/时间：`merged`，created 2025-08-08, merged 2025-08-08；作者 `ch-wan`。
- 代码 diff 已读范围：`8` 个文件，`+269/-119`；代码面：model wrapper, MoE/router, quantization, kernel；关键词：moe, quant, triton, cuda, expert, topk, router, cache, config, flash。
- 代码 diff 细节：
  - `python/sglang/srt/layers/quantization/mxfp4.py` modified +80/-52 (132 lines); hunk: from typing import TYPE_CHECKING, List, Optional; is_cuda,; 符号: get_quant_method, Mxfp4MoEMethod, __init__, create_weights
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +101/-12 (113 lines); hunk: def fused_moe_kernel(; def fused_moe_kernel(; 符号: fused_moe_kernel, fused_moe_kernel, fused_moe_kernel, fused_moe_kernel
  - `python/sglang/srt/models/gpt_oss.py` modified +54/-47 (101 lines); hunk: get_moe_expert_parallel_rank,; def __init__(; 符号: __init__, _load_mxfp4_experts_weights, _load_mxfp4_experts_weights, _load_mxfp4_experts_weights
  - `python/sglang/srt/server_args.py` modified +10/-4 (14 lines); hunk: is_hip,; def print_deprecated_warning(message: str):; 符号: print_deprecated_warning
  - `python/sglang/srt/layers/quantization/unquant.py` modified +9/-2 (11 lines); hunk: def apply(; def create_weights(; 符号: apply, UnquantizedFusedMoEMethod, __init__, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/quantization/mxfp4.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/models/gpt_oss.py`；patch 关键词为 moe, quant, triton, cuda, expert, topk。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/quantization/mxfp4.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/models/gpt_oss.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #9043 - (gpt-oss, oai, chat): Remove Harmony Integration and Implement Native GPT-OSS Tool Call Support

- 链接：https://github.com/sgl-project/sglang/pull/9043
- 状态/时间：`merged`，created 2025-08-11, merged 2025-08-12；作者 `CatherineSue`。
- 代码 diff 已读范围：`10` 个文件，`+717/-409`；代码面：docs/config；关键词：cache, kv, spec, config, doc, lora。
- 代码 diff 细节：
  - `python/sglang/srt/function_call/gpt_oss_detector.py` added +331/-0 (331 lines); hunk: +import json; 符号: GptOssDetector, __init__, has_tool_call, detect_and_parse
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +61/-265 (326 lines); hunk: from fastapi import Request; class OpenAIServingChat(OpenAIServingBase):; 符号: OpenAIServingChat, __init__, _request_id_prefix, _validate_request
  - `python/sglang/srt/reasoning_parser.py` modified +316/-0 (316 lines); hunk: +import re; def __init__(self, stream_reasoning: bool = True, force_reasoning: bool = False); 符号: __init__, GptOssDetector, __init__, detect_and_parse
  - `python/sglang/srt/function_call/harmony_tool_parser.py` removed +0/-130 (130 lines); hunk: -# Copyright 2023-2024 SGLang Team; 符号: HarmonyToolCallParser:, extract_tool_calls_from_message, process_streaming_chunk
  - `python/sglang/srt/entrypoints/openai/protocol.py` modified +0/-9 (9 lines); hunk: class ResponseReasoningTextContent(BaseModel):; 符号: ResponseReasoningTextContent, ResponseReasoningItem
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/function_call/gpt_oss_detector.py`, `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/reasoning_parser.py`；patch 关键词为 cache, kv, spec, config, doc, lora。影响判断：文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/function_call/gpt_oss_detector.py`, `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/reasoning_parser.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #9359 - Support DP attention with GPT-OSS

- 链接：https://github.com/sgl-project/sglang/pull/9359
- 状态/时间：`merged`，created 2025-08-19, merged 2025-08-20；作者 `nvcastet`。
- 代码 diff 已读范围：`2` 个文件，`+6/-5`；代码面：model wrapper；关键词：attention, config, flash, fp4, quant, spec。
- 代码 diff 细节：
  - `python/sglang/srt/server_args.py` modified +5/-4 (9 lines); hunk: def model_specific_adjustments(self):; 符号: model_specific_adjustments
  - `python/sglang/srt/models/gpt_oss.py` modified +1/-1 (2 lines); hunk: def _load_normal_weights(; 符号: _load_normal_weights
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/server_args.py`, `python/sglang/srt/models/gpt_oss.py`；patch 关键词为 attention, config, flash, fp4, quant, spec。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/server_args.py`, `python/sglang/srt/models/gpt_oss.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #14920 - Eagle: GPT-OSS Eagle v2 support

- 链接：https://github.com/sgl-project/sglang/pull/14920
- 状态/时间：`merged`，created 2025-12-11, merged 2025-12-30；作者 `IzzyPutterman`。
- 代码 diff 已读范围：`4` 个文件，`+48/-25`；代码面：kernel, scheduler/runtime；关键词：eagle, spec, cuda, config, attention, moe, vision。
- 代码 diff 细节：
  - `python/sglang/srt/model_executor/model_runner.py` modified +30/-23 (53 lines); hunk: def __init__(; def initialize(self, min_per_gpu_memory: float):; 符号: __init__, initialize, _dummy_run
  - `python/sglang/srt/speculative/eagle_worker.py` modified +10/-0 (10 lines); hunk: def __init__(; def forward_draft_extend_after_decode(self, batch: ScheduleBatch):; 符号: __init__, forward_draft_extend_after_decode
  - `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +4/-1 (5 lines); hunk: def __init__(self, model_runner: ModelRunner):; 符号: __init__
  - `python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py` modified +4/-1 (5 lines); hunk: def __init__(self, eagle_worker: EAGLEWorker):; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/speculative/eagle_worker.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`；patch 关键词为 eagle, spec, cuda, config, attention, moe。影响判断：CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/speculative/eagle_worker.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #18988 - [GPT-OSS] support fp8 online quantization for gpt-oss bf16

- 链接：https://github.com/sgl-project/sglang/pull/18988
- 状态/时间：`merged`，created 2026-02-18, merged 2026-02-20；作者 `zminglei`。
- 代码 diff 已读范围：`2` 个文件，`+31/-1`；代码面：quantization；关键词：moe, quant, triton, config, expert, fp8, spec。
- 代码 diff 细节：
  - `python/sglang/srt/layers/quantization/fp8.py` modified +26/-0 (26 lines); hunk: def __init__(self, quant_config: Fp8Config):; def create_weights(; 符号: __init__, create_weights, create_weights, apply
  - `python/sglang/srt/server_args.py` modified +5/-1 (6 lines); hunk: def _handle_model_specific_adjustments(self):; 符号: _handle_model_specific_adjustments
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/server_args.py`；patch 关键词为 moe, quant, triton, config, expert, fp8。影响判断：量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/server_args.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：6；open PR 数：0。
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
