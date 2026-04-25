# SGLang Llama 4 支持与 PR 历史

本文记录 SGLang 中与 Llama 4 相关的模型支持、关键 PR、以及 cookbook 对应的落点。

- 状态: 当前 mainline 已支持

## 核心结论

- Llama4 is mature on the SGLang side but still sensitive to quantized MoE and long-context backend selection.
- The multimodal path adds a separate vision-rotary and processor validation surface.

## 主要代码面

- `sglang/python/sglang/srt/models/llama4.py`
- `sglang/python/sglang/srt/models/mllama4.py`

## 已合入 PR

- [#5092](https://github.com/sgl-project/sglang/pull/5092) `Add Llama4 support`：Initial Llama4 landing in SGLang.
- [#5194](https://github.com/sgl-project/sglang/pull/5194) `Support Llama4 fp8 inference`：Enabled the first production quantized lane.
- [#6162](https://github.com/sgl-project/sglang/pull/6162) `Fix Llama4 gibberish output with long context and CUDA graph`：Closed a major correctness bug.
- [#7129](https://github.com/sgl-project/sglang/pull/7129) `Enable ModelOpt Llama4 fp8 checkpoint deployment in SGLang`：Added the ModelOpt checkpoint path.
- [#13421](https://github.com/sgl-project/sglang/pull/13421) `Add Llama4 attention backend auto-selection`：Stabilized backend choice for real deployments.

## 配套 skill

- `skills/model-optimization/sglang/sglang-llama4-optimization/SKILL.md`
- `skills/model-optimization/sglang/sglang-llama4-optimization/references/pr-history.md`

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `sgl-project/sglang` 的 Pull Request API 和文件级 patch 重新审计 `Llama 4`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2025-04-05 | [#5092](https://github.com/sgl-project/sglang/pull/5092) | merged | Add Llama4 support | model wrapper, attention/backend, MoE/router, quantization, kernel, multimodal/processor, scheduler/runtime, docs/config | `python/sglang/srt/models/llama4.py`, `python/sglang/srt/layers/attention/flashattention_backend.py`, `python/sglang/srt/managers/multimodal_processors/mllama4.py` |
| 2025-04-09 | [#5194](https://github.com/sgl-project/sglang/pull/5194) | merged | Support Llama4 fp8 inference | model wrapper, MoE/router, quantization, kernel, tests/benchmarks | `test/srt/test_triton_moe_channel_fp8_kernel.py`, `python/sglang/srt/layers/quantization/w8a8_fp8.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` |
| 2025-05-09 | [#6162](https://github.com/sgl-project/sglang/pull/6162) | merged | [Bugfix] Fix Llama4 gibberish output with long context and CUDA graph | attention/backend | `python/sglang/srt/layers/attention/flashattention_backend.py` |
| 2025-06-12 | [#7129](https://github.com/sgl-project/sglang/pull/7129) | merged | Enable ModelOpt Llama4 fp8 checkpoint deployment in SGLang | model wrapper, MoE/router, quantization, kernel | `python/sglang/srt/models/mllama4.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` |
| 2025-11-17 | [#13421](https://github.com/sgl-project/sglang/pull/13421) | merged | Add Llama4 attention backend auto-selection | docs/config | `python/sglang/srt/server_args.py`, `docs/basic_usage/llama4.md` |

### 逐 PR 代码 diff 阅读记录

### PR #5092 - Add Llama4 support

- 链接：https://github.com/sgl-project/sglang/pull/5092
- 状态/时间：`merged`，created 2025-04-05, merged 2025-04-07；作者 `CatherineSue`。
- 代码 diff 已读范围：`27` 个文件，`+2213/-21`；代码面：model wrapper, attention/backend, MoE/router, quantization, kernel, multimodal/processor, scheduler/runtime, docs/config；关键词：config, moe, triton, attention, kv, spec, vision, expert, processor, quant。
- 代码 diff 细节：
  - `python/sglang/srt/models/llama4.py` added +420/-0 (420 lines); hunk: +# Copyright 2023-2024 SGLang Team; 符号: Llama4MoE, custom_routing_function, __init__, forward
  - `python/sglang/srt/layers/attention/flashattention_backend.py` modified +286/-9 (295 lines); hunk: from __future__ import annotations; class FlashAttentionMetadata:; 符号: FlashAttentionMetadata:, LocalAttentionMetadata:, make_local_attention_virtual_batches, cdiv
  - `python/sglang/srt/managers/multimodal_processors/mllama4.py` added +161/-0 (161 lines); hunk: +from typing import List, Mapping, Optional, Tuple, Union; 符号: Mllama4ImageProcessor, __init__, process_mm_data_async, get_patch_per_chunk
  - `python/sglang/srt/models/mllama4.py` added +154/-0 (154 lines); hunk: +# NOTE: add Aapted from vllm/mllama4.py; 符号: Llama4ForConditionalGeneration, __init__, forward, permute_qk_weight_for_rotary
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/E=128,N=512,device_name=NVIDIA_H100_80GB_HBM3.json` added +146/-0 (146 lines); hunk: +{
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/llama4.py`, `python/sglang/srt/layers/attention/flashattention_backend.py`, `python/sglang/srt/managers/multimodal_processors/mllama4.py`；patch 关键词为 config, moe, triton, attention, kv, spec。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/llama4.py`, `python/sglang/srt/layers/attention/flashattention_backend.py`, `python/sglang/srt/managers/multimodal_processors/mllama4.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #5194 - Support Llama4 fp8 inference

- 链接：https://github.com/sgl-project/sglang/pull/5194
- 状态/时间：`merged`，created 2025-04-09, merged 2025-04-09；作者 `HandH1998`。
- 代码 diff 已读范围：`14` 个文件，`+537/-106`；代码面：model wrapper, MoE/router, quantization, kernel, tests/benchmarks；关键词：quant, fp8, config, expert, moe, triton, cuda, kv, topk, processor。
- 代码 diff 细节：
  - `test/srt/test_triton_moe_channel_fp8_kernel.py` added +177/-0 (177 lines); hunk: +import itertools; 符号: native_w8a8_per_token_matmul, fp8_mask, torch_w8a8_per_column_moe, TestW8A8FP8FusedMoE
  - `python/sglang/srt/layers/quantization/w8a8_fp8.py` modified +154/-4 (158 lines); hunk: -from typing import Any, Dict, List, Optional; input_to_float8,; 符号: get_config_filenames, from_config, get_quant_method, get_quant_method
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +66/-45 (111 lines); hunk: def __init__(; def create_weights(; 符号: __init__, create_weights, create_weights, process_weights_after_loading
  - `python/sglang/srt/models/mllama4.py` modified +52/-18 (70 lines); hunk: from transformers import Llama4Config; class Llama4ForConditionalGeneration(nn.Module):; 符号: Llama4ForConditionalGeneration, __init__, load_weights, load_weights
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +33/-18 (51 lines); hunk: def fused_moe_kernel(; def fused_moe_kernel(; 符号: fused_moe_kernel, fused_moe_kernel, fused_moe_kernel, invoke_fused_moe_kernel
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/srt/test_triton_moe_channel_fp8_kernel.py`, `python/sglang/srt/layers/quantization/w8a8_fp8.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`；patch 关键词为 quant, fp8, config, expert, moe, triton。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/srt/test_triton_moe_channel_fp8_kernel.py`, `python/sglang/srt/layers/quantization/w8a8_fp8.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #6162 - [Bugfix] Fix Llama4 gibberish output with long context and CUDA graph

- 链接：https://github.com/sgl-project/sglang/pull/6162
- 状态/时间：`merged`，created 2025-05-09, merged 2025-05-09；作者 `CatherineSue`。
- 代码 diff 已读范围：`1` 个文件，`+125/-8`；代码面：attention/backend；关键词：attention, cache, cuda, eagle, flash, kv, test, topk。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/flashattention_backend.py` modified +125/-8 (133 lines); hunk: def forward_decode(; def forward_decode(; 符号: forward_decode, forward_decode, forward_decode, init_cuda_graph_state
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/flashattention_backend.py`；patch 关键词为 attention, cache, cuda, eagle, flash, kv。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/flashattention_backend.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #7129 - Enable ModelOpt Llama4 fp8 checkpoint deployment in SGLang

- 链接：https://github.com/sgl-project/sglang/pull/7129
- 状态/时间：`merged`，created 2025-06-12, merged 2025-07-08；作者 `Edwardf0t1`。
- 代码 diff 已读范围：`3` 个文件，`+643/-81`；代码面：model wrapper, MoE/router, quantization, kernel；关键词：expert, fp8, moe, quant, cache, config, fp4, kv, spec, triton。
- 代码 diff 细节：
  - `python/sglang/srt/models/mllama4.py` modified +360/-79 (439 lines); hunk: +import json as json_lib; from sglang.srt.utils import add_prefix, is_cpu; 符号: Llama4ForConditionalGeneration, __init__, _has_vision_weights, _check_vision_weights_in_index
  - `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +244/-1 (245 lines); hunk: from sglang.srt.layers.quantization.utils import (; def get_quant_method(; 符号: get_quant_method, get_quant_method, get_scaled_act_names, __init__
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +39/-1 (40 lines); hunk: def _load_w2(; def _load_w2(; 符号: _load_w2, _load_w2, weight_loader
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/mllama4.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`；patch 关键词为 expert, fp8, moe, quant, cache, config。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/mllama4.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #13421 - Add Llama4 attention backend auto-selection

- 链接：https://github.com/sgl-project/sglang/pull/13421
- 状态/时间：`merged`，created 2025-11-17, merged 2025-11-25；作者 `janbernloehr`。
- 代码 diff 已读范围：`2` 个文件，`+24/-5`；代码面：docs/config；关键词：attention, spec, triton, cache, doc, flash, fp8, kv, moe, quant。
- 代码 diff 细节：
  - `python/sglang/srt/server_args.py` modified +15/-5 (20 lines); hunk: def _handle_model_specific_adjustments(self):; 符号: _handle_model_specific_adjustments
  - `docs/basic_usage/llama4.md` modified +9/-0 (9 lines); hunk: python3 -m sglang.launch_server \; 符号: llama
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/server_args.py`, `docs/basic_usage/llama4.md`；patch 关键词为 attention, spec, triton, cache, doc, flash。影响判断：文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/server_args.py`, `docs/basic_usage/llama4.md` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：5；open PR 数：0。
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
