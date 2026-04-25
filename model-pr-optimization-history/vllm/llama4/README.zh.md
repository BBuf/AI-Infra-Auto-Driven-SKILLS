# vLLM Llama 4 支持与 PR 历史

本文记录 vLLM 中与 Llama 4 相关的模型支持、关键 PR、以及 cookbook 对应的落点。

- 状态: 当前 mainline 已支持

## 核心结论

- Llama4 is mature on the vLLM side but still sensitive to quantized MoE and long-context backend selection.
- The multimodal path adds a separate vision-rotary and processor validation surface.

## 主要代码面

- `vllm/vllm/model_executor/models/llama4.py`
- `vllm/vllm/model_executor/models/mllama4.py`
- `vllm/vllm/model_executor/models/llama4_eagle.py`

## 已合入 PR

- [#16104](https://github.com/vllm-project/vllm/pull/16104) `Support Llama4 in vLLM`：Initial Llama4 landing.
- [#20419](https://github.com/vllm-project/vllm/pull/20419) `Enable ModelOpt Llama4 fp8 checkpoint deployment`：Added ModelOpt FP8 coverage.
- [#20591](https://github.com/vllm-project/vllm/pull/20591) `Llama4 EAGLE Support`：Opened speculative decoding for Llama4.
- [#22511](https://github.com/vllm-project/vllm/pull/22511) `Fix Llama4 FlashInfer FP4 MoE issues`：Stabilized the FP4 MoE path.
- [#25889](https://github.com/vllm-project/vllm/pull/25889) `Fix misplaced dtype cast in Llama4VisionRotaryEmbedding`：Patched a multimodal rotary bug.

## 配套 skill

- `skills/model-optimization/vllm/vllm-llama4-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-llama4-optimization/references/pr-history.md`

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `vllm-project/vllm` 的 Pull Request API 和文件级 patch 重新审计 `Llama 4`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2025-04-05 | [#16104](https://github.com/vllm-project/vllm/pull/16104) | merged | [Model] Support Llama4 in vLLM | model wrapper, attention/backend, MoE/router, quantization, kernel, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/mllama4.py`, `vllm/model_executor/models/llama4.py`, `vllm/v1/attention/backends/flash_attn.py` |
| 2025-07-03 | [#20419](https://github.com/vllm-project/vllm/pull/20419) | merged | Enable ModelOpt Llama4 fp8 checkpoint deployment | model wrapper, MoE/router, quantization, scheduler/runtime | `vllm/model_executor/layers/quantization/modelopt.py`, `vllm/model_executor/models/mllama4.py`, `vllm/model_executor/models/llama4.py` |
| 2025-07-07 | [#20591](https://github.com/vllm-project/vllm/pull/20591) | merged | [Meta] Llama4 EAGLE Support | model wrapper, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/llama4_eagle.py`, `tests/v1/e2e/test_spec_decode.py`, `tests/models/registry.py` |
| 2025-08-08 | [#22511](https://github.com/vllm-project/vllm/pull/22511) | merged | Fix Llama4 FlashInfer FP4 MoE issues | attention/backend, MoE/router, quantization, scheduler/runtime | `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_prepare_finalize.py`, `vllm/model_executor/layers/quantization/modelopt.py`, `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py` |
| 2025-09-29 | [#25889](https://github.com/vllm-project/vllm/pull/25889) | merged | [Llama4] [multimodal] Fix misplaced dtype cast of `cos_sin_cache` in `Llama4VisionRotaryEmbedding` | multimodal/processor, scheduler/runtime | `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` |

### 逐 PR 代码 diff 阅读记录

### PR #16104 - [Model] Support Llama4 in vLLM

- 链接：https://github.com/vllm-project/vllm/pull/16104
- 状态/时间：`merged`，created 2025-04-05, merged 2025-04-06；作者 `houseroad`。
- 代码 diff 已读范围：`35` 个文件，`+2369/-142`；代码面：model wrapper, attention/backend, MoE/router, quantization, kernel, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config；关键词：cache, config, attention, kv, quant, moe, spec, vision, expert, processor。
- 代码 diff 细节：
  - `vllm/model_executor/models/mllama4.py` added +886/-0 (886 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: Llama4ImagePatchInputs, Llama4VisionMLP, __init__, forward
  - `vllm/model_executor/models/llama4.py` added +530/-0 (530 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: Llama4MoE, custom_routing_function, __init__, forward
  - `vllm/v1/attention/backends/flash_attn.py` modified +236/-14 (250 lines); hunk: class FlashAttentionMetadata:; def reorder_batch(self, input_batch: "InputBatch",; 符号: FlashAttentionMetadata:, LocalAttentionMetadata:, make_local_attention_virtual_batches, FlashAttentionMetadataBuilder:
  - `vllm/model_executor/layers/fused_moe/configs/E=16,N=1024,device_name=AMD_Instinct_MI300X.json` added +200/-0 (200 lines); hunk: +{
  - `tests/models/multimodal/processing/test_llama4.py` added +99/-0 (99 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: test_processor_override
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/mllama4.py`, `vllm/model_executor/models/llama4.py`, `vllm/v1/attention/backends/flash_attn.py`；patch 关键词为 cache, config, attention, kv, quant, moe。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/mllama4.py`, `vllm/model_executor/models/llama4.py`, `vllm/v1/attention/backends/flash_attn.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #20419 - Enable ModelOpt Llama4 fp8 checkpoint deployment

- 链接：https://github.com/vllm-project/vllm/pull/20419
- 状态/时间：`merged`，created 2025-07-03, merged 2025-07-12；作者 `Edwardf0t1`。
- 代码 diff 已读范围：`5` 个文件，`+501/-35`；代码面：model wrapper, MoE/router, quantization, scheduler/runtime；关键词：expert, fp8, kv, moe, config, quant, spec, attention, cache, fp4。
- 代码 diff 细节：
  - `vllm/model_executor/layers/quantization/modelopt.py` modified +261/-5 (266 lines); hunk: class ModelOptFp8Config(QuantizationConfig):; def get_config_filenames(cls) -> list[str]:; 符号: ModelOptFp8Config, __init__, get_config_filenames, from_config
  - `vllm/model_executor/models/mllama4.py` modified +144/-20 (164 lines); hunk: class Llama4ForConditionalGeneration(nn.Module, SupportsMultiModal,; def _consolidate_qkv_weights(; 符号: Llama4ForConditionalGeneration, _consolidate_qkv_weights, load_weights, _rename_weight_for_modelopt_checkpoint
  - `vllm/model_executor/models/llama4.py` modified +55/-4 (59 lines); hunk: RowParallelLinear); def load_weights(self, weights: Iterable[tuple[str,; 符号: load_weights, load_weights
  - `vllm/model_executor/layers/fused_moe/layer.py` modified +31/-6 (37 lines); hunk: def create_weights(self, layer: torch.nn.Module, num_experts: int,; def weight_loader(self,; 符号: create_weights, uses_weight_scale_2_pattern, maybe_make_prepare_finalize, weight_loader
  - `vllm/model_executor/model_loader/weight_utils.py` modified +10/-0 (10 lines); hunk: def maybe_remap_kv_scale_name(name: str, params_dict: dict) -> Optional[str]:; 符号: maybe_remap_kv_scale_name
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/layers/quantization/modelopt.py`, `vllm/model_executor/models/mllama4.py`, `vllm/model_executor/models/llama4.py`；patch 关键词为 expert, fp8, kv, moe, config, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/layers/quantization/modelopt.py`, `vllm/model_executor/models/mllama4.py`, `vllm/model_executor/models/llama4.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #20591 - [Meta] Llama4 EAGLE Support

- 链接：https://github.com/vllm-project/vllm/pull/20591
- 状态/时间：`merged`，created 2025-07-07, merged 2025-07-16；作者 `morgendave`。
- 代码 diff 已读范围：`6` 个文件，`+258/-18`；代码面：model wrapper, scheduler/runtime, tests/benchmarks, docs/config；关键词：eagle, config, spec, test, cache, cuda, kv, moe, processor, quant。
- 代码 diff 细节：
  - `vllm/model_executor/models/llama4_eagle.py` added +214/-0 (214 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: LlamaModel, __init__, forward, load_weights
  - `tests/v1/e2e/test_spec_decode.py` modified +31/-17 (48 lines); hunk: from typing import Any; def model_name():; 符号: model_name, eagle_model_name, eagle3_model_name, test_ngram_correctness
  - `tests/models/registry.py` modified +6/-1 (7 lines); hunk: def check_available_online(; def find_hf_info(self, model_id: str) -> _HfExamplesInfo:; 符号: check_available_online, find_hf_info
  - `tests/models/test_initialization.py` modified +5/-0 (5 lines); hunk: def test_can_initialize(model_arch: str, monkeypatch: pytest.MonkeyPatch):; 符号: test_can_initialize, hf_overrides
  - `examples/offline_inference/spec_decode.py` modified +1/-0 (1 lines); hunk: def main():; 符号: main
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/llama4_eagle.py`, `tests/v1/e2e/test_spec_decode.py`, `tests/models/registry.py`；patch 关键词为 eagle, config, spec, test, cache, cuda。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/llama4_eagle.py`, `tests/v1/e2e/test_spec_decode.py`, `tests/models/registry.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22511 - Fix Llama4 FlashInfer FP4 MoE issues

- 链接：https://github.com/vllm-project/vllm/pull/22511
- 状态/时间：`merged`，created 2025-08-08, merged 2025-08-12；作者 `nvpohanh`。
- 代码 diff 已读范围：`3` 个文件，`+9/-5`；代码面：attention/backend, MoE/router, quantization, scheduler/runtime；关键词：expert, flash, moe, quant, router, topk。
- 代码 diff 细节：
  - `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_prepare_finalize.py` modified +6/-1 (7 lines); hunk: def prepare(; 符号: prepare
  - `vllm/model_executor/layers/quantization/modelopt.py` modified +3/-2 (5 lines); hunk: def apply(; 符号: apply
  - `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py` modified +0/-2 (2 lines); hunk: def apply(; 符号: apply
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_prepare_finalize.py`, `vllm/model_executor/layers/quantization/modelopt.py`, `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py`；patch 关键词为 expert, flash, moe, quant, router, topk。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_prepare_finalize.py`, `vllm/model_executor/layers/quantization/modelopt.py`, `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #25889 - [Llama4] [multimodal] Fix misplaced dtype cast of `cos_sin_cache` in `Llama4VisionRotaryEmbedding`

- 链接：https://github.com/vllm-project/vllm/pull/25889
- 状态/时间：`merged`，created 2025-09-29, merged 2025-09-30；作者 `cjackal`。
- 代码 diff 已读范围：`1` 个文件，`+3/-1`；代码面：multimodal/processor, scheduler/runtime；关键词：cache, vision。
- 代码 diff 细节：
  - `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` modified +3/-1 (4 lines); hunk: def forward_native( # type: ignore[override]; 符号: forward_native
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py`；patch 关键词为 cache, vision。影响判断：多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：5；open PR 数：0。
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
