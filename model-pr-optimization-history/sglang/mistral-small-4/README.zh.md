# SGLang Mistral Small 4 支持与 PR 历史

本文记录 SGLang 中与 Mistral Small 4 相关的模型支持、关键 PR、以及 cookbook 对应的落点。

- 状态: 当前 mainline 已支持

## 核心结论

- Mistral Small 4 sits on top of the larger Mistral Large 3 / Ministral runtime work.
- Startup format mismatches, multimodal projector behavior, and Eagle / MoE integration are the main risk areas.

## 主要代码面

- `sglang/python/sglang/srt/models/mistral_large_3.py`
- `sglang/python/sglang/srt/models/mistral_large_3_eagle.py`
- `sglang/python/sglang/srt/models/mistral.py`
- `sglang/python/sglang/srt/models/ministral3.py`

## 已合入 PR

- [#14213](https://github.com/sgl-project/sglang/pull/14213) `Add Mistral Large 3 support`：Historical base runtime reused by later Small 4 work.
- [#14466](https://github.com/sgl-project/sglang/pull/14466) `Add Mistral Large 3 Eagle Support`：Enabled speculative decode on the underlying family.
- [#15049](https://github.com/sgl-project/sglang/pull/15049) `Mistral Large 3 NVFP4 TRTLLM MoE support`：Added the first serious quantized MoE path.
- [#20708](https://github.com/sgl-project/sglang/pull/20708) `Add Mistral Small 4 support`：Brought Mistral Small 4 / Pixtral-style runtime into mainline.
- [#21620](https://github.com/sgl-project/sglang/pull/21620) `Mistral Small 4 fails to start due to config/weight format mismatch`：Closed a startup regression after launch.

## 配套 skill

- `skills/model-optimization/sglang/sglang-mistral-small-4-optimization/SKILL.md`
- `skills/model-optimization/sglang/sglang-mistral-small-4-optimization/references/pr-history.md`

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `sgl-project/sglang` 的 Pull Request API 和文件级 patch 重新审计 `Mistral Small 4 / Ministral 3`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2025-12-01 | [#14213](https://github.com/sgl-project/sglang/pull/14213) | merged | Add Mistral Large 3 support. | model wrapper, attention/backend, MoE/router, quantization, kernel, multimodal/processor, tests/benchmarks, docs/config | `python/sglang/srt/models/pixtral.py`, `python/sglang/srt/utils/mistral_utils.py`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py` |
| 2025-12-05 | [#14466](https://github.com/sgl-project/sglang/pull/14466) | merged | Add Mistral Large 3 Eagle Support | model wrapper, attention/backend, quantization, docs/config | `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/models/mistral_large_3_eagle.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2025-12-13 | [#15049](https://github.com/sgl-project/sglang/pull/15049) | merged | Mistral Large 3 NVFP4 TRTLLM MoE support | MoE/router, quantization, kernel | `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `python/sglang/srt/layers/quantization/utils.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py` |
| 2026-03-16 | [#20708](https://github.com/sgl-project/sglang/pull/20708) | merged | Add Mistral Small 4 (Pixtral) support | model wrapper, attention/backend, MoE/router, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `python/sglang/srt/multimodal/processors/pixtral.py`, `python/sglang/srt/utils/hf_transformers_utils.py`, `benchmark/mmmu/bench_sglang.py` |
| 2026-03-29 | [#21620](https://github.com/sgl-project/sglang/pull/21620) | merged | fix: Mistral Small 4 fails to start due to config/weight format mismatch | model wrapper, tests/benchmarks | `python/sglang/srt/server_args.py`, `test/registered/models/test_ministral4_models.py` |

### 逐 PR 代码 diff 阅读记录

### PR #14213 - Add Mistral Large 3 support.

- 链接：https://github.com/sgl-project/sglang/pull/14213
- 状态/时间：`merged`，created 2025-12-01, merged 2025-12-04；作者 `dcampora`。
- 代码 diff 已读范围：`16` 个文件，`+1400/-120`；代码面：model wrapper, attention/backend, MoE/router, quantization, kernel, multimodal/processor, tests/benchmarks, docs/config；关键词：quant, fp8, config, kv, attention, expert, moe, vision, cache, fp4。
- 代码 diff 细节：
  - `python/sglang/srt/models/pixtral.py` modified +565/-3 (568 lines); hunk: Using mistral-community/pixtral-12b as reference.; from sglang.srt.layers.layernorm import RMSNorm; 符号: VisionEncoderArgs:, PixtralForConditionalGeneration, get_placeholder_str, __init__
  - `python/sglang/srt/utils/mistral_utils.py` added +295/-0 (295 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: adapt_config_dict, _remap_mistral_vision_args, _remap_mistral_yarn_args, _remap_general_mistral_args
  - `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py` modified +127/-63 (190 lines); hunk: # Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors; from sglang.srt.layers.quantization.fp; 符号: CompressedTensorsW8A8Fp8, __init__, CompressedTensorsW8A8Fp8, __init__
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +81/-1 (82 lines); hunk: from compressed_tensors import CompressionFormat; def get_moe_method(; 符号: get_moe_method, __init__, create_weights, create_weights
  - `python/sglang/srt/models/mistral_large_3.py` added +81/-0 (81 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: MistralLarge3ForCausalLM, load_weights, _iterable_remap_mistral_to_ds
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/pixtral.py`, `python/sglang/srt/utils/mistral_utils.py`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py`；patch 关键词为 quant, fp8, config, kv, attention, expert。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/pixtral.py`, `python/sglang/srt/utils/mistral_utils.py`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #14466 - Add Mistral Large 3 Eagle Support

- 链接：https://github.com/sgl-project/sglang/pull/14466
- 状态/时间：`merged`，created 2025-12-05, merged 2025-12-05；作者 `elvischenv`。
- 代码 diff 已读范围：`9` 个文件，`+313/-62`；代码面：model wrapper, attention/backend, quantization, docs/config；关键词：config, quant, attention, eagle, expert, fp8, kv, mla, moe, spec。
- 代码 diff 细节：
  - `python/sglang/srt/layers/quantization/fp8.py` modified +161/-36 (197 lines); hunk: def process_weights_after_loading(self, layer: Module) -> None:; def apply_with_router_logits(; 符号: process_weights_after_loading, process_weights_hip_int4, apply_with_router_logits, apply_with_router_logits
  - `python/sglang/srt/models/mistral_large_3_eagle.py` added +105/-0 (105 lines); hunk: +from typing import Optional; 符号: MistralLarge3Model, __init__, forward, MistralLarge3ForCausalLMEagle
  - `python/sglang/srt/models/deepseek_v2.py` modified +14/-6 (20 lines); hunk: def __init__(; def forward(; 符号: __init__, forward, DeepseekV2ForCausalLM, __init__
  - `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py` modified +6/-10 (16 lines); hunk: def create_weights(; def create_weights(; 符号: create_weights, create_weights, process_weights_after_loading, apply_weights
  - `python/sglang/srt/configs/model_config.py` modified +11/-1 (12 lines); hunk: def _derive_model_shapes(self):; def _verify_quantization(self) -> None:; 符号: _derive_model_shapes, _verify_quantization
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/models/mistral_large_3_eagle.py`, `python/sglang/srt/models/deepseek_v2.py`；patch 关键词为 config, quant, attention, eagle, expert, fp8。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/models/mistral_large_3_eagle.py`, `python/sglang/srt/models/deepseek_v2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #15049 - Mistral Large 3 NVFP4 TRTLLM MoE support

- 链接：https://github.com/sgl-project/sglang/pull/15049
- 状态/时间：`merged`，created 2025-12-13, merged 2025-12-18；作者 `elvischenv`。
- 代码 diff 已读范围：`7` 个文件，`+340/-151`；代码面：MoE/router, quantization, kernel；关键词：moe, fp8, quant, flash, config, expert, fp4, topk, cache, router。
- 代码 diff 细节：
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +193/-21 (214 lines); hunk: from compressed_tensors import CompressionFormat; from sglang.srt.layers.quantization.utils import (; 符号: __init__, create_weights, create_weights, process_weights_after_loading
  - `python/sglang/srt/layers/quantization/utils.py` modified +140/-0 (140 lines); hunk: def swizzle_blockscale(scale: torch.Tensor):; 符号: swizzle_blockscale, reorder_w1w3_to_w3w1, prepare_static_weights_for_trtllm_fp4_moe
  - `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +2/-125 (127 lines); hunk: convert_to_channelwise,; def create_weights(; 符号: create_weights, prepare_static_weights_for_kernel, process_weights_after_loading, _slice_scale
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +2/-1 (3 lines); hunk: def get_moe_impl_class(quant_config: Optional[QuantizationConfig]):; 符号: get_moe_impl_class
  - `python/sglang/srt/server_args.py` modified +2/-1 (3 lines); hunk: def _handle_moe_kernel_config(self):; 符号: _handle_moe_kernel_config
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `python/sglang/srt/layers/quantization/utils.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py`；patch 关键词为 moe, fp8, quant, flash, config, expert。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `python/sglang/srt/layers/quantization/utils.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #20708 - Add Mistral Small 4 (Pixtral) support

- 链接：https://github.com/sgl-project/sglang/pull/20708
- 状态/时间：`merged`，created 2026-03-16, merged 2026-03-18；作者 `JustinTong0323`。
- 代码 diff 已读范围：`18` 个文件，`+360/-124`；代码面：model wrapper, attention/backend, MoE/router, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config；关键词：config, processor, spec, vision, attention, benchmark, cache, eagle, expert, flash。
- 代码 diff 细节：
  - `python/sglang/srt/multimodal/processors/pixtral.py` modified +71/-44 (115 lines); hunk: -import asyncio; class PixtralProcessor(BaseMultimodalProcessor):; 符号: PixtralProcessor, get_patch_grid_size, __init__, defined
  - `python/sglang/srt/utils/hf_transformers_utils.py` modified +55/-6 (61 lines); hunk: import os; def _load_deepseek_v32_model(; 符号: _load_deepseek_v32_model, _load_mistral_large_3_for_causal_LM, _load_mistral_large_3_for_causal_LM, get_config
  - `benchmark/mmmu/bench_sglang.py` modified +49/-10 (59 lines); hunk: import argparse; def _get_prefix_suffix(prompt: str) -> Tuple[str, str]:; 符号: _get_prefix_suffix, process_sample, process_sample_with_semaphore, eval_mmmu
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +32/-13 (45 lines); hunk: def _process_messages(; def _apply_jinja_template(; 符号: _process_messages, and, _apply_jinja_template, _apply_jinja_template
  - `python/sglang/srt/utils/mistral_utils.py` modified +27/-3 (30 lines); hunk: def adapt_config_dict(; def _remap_mistral_yarn_args(config: dict) -> dict:; 符号: adapt_config_dict, _remap_mistral_yarn_args
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/multimodal/processors/pixtral.py`, `python/sglang/srt/utils/hf_transformers_utils.py`, `benchmark/mmmu/bench_sglang.py`；patch 关键词为 config, processor, spec, vision, attention, benchmark。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/multimodal/processors/pixtral.py`, `python/sglang/srt/utils/hf_transformers_utils.py`, `benchmark/mmmu/bench_sglang.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21620 - fix: Mistral Small 4 fails to start due to config/weight format mismatch

- 链接：https://github.com/sgl-project/sglang/pull/21620
- 状态/时间：`merged`，created 2026-03-29, merged 2026-03-30；作者 `LiYomi`。
- 代码 diff 已读范围：`2` 个文件，`+59/-7`；代码面：model wrapper, tests/benchmarks；关键词：attention, config, cuda, test。
- 代码 diff 细节：
  - `python/sglang/srt/server_args.py` modified +27/-7 (34 lines); hunk: def _handle_load_format(self):; 符号: _handle_load_format, _is_mistral_native_format, _check_format
  - `test/registered/models/test_ministral4_models.py` added +32/-0 (32 lines); hunk: +import unittest; 符号: TestMistralSmall4TextOnly, TestMistralSmall4MMMU
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/server_args.py`, `test/registered/models/test_ministral4_models.py`；patch 关键词为 attention, config, cuda, test。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/server_args.py`, `test/registered/models/test_ministral4_models.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：5；open PR 数：0。
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
