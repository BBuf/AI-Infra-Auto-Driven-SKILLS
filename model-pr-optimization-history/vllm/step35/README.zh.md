# vLLM Step3.5 / Step3-VL 支持与 PR 历史

本文记录 vLLM 中与 Step3.5 / Step3-VL 相关的模型支持、关键 PR、以及 cookbook 对应的落点。

- 状态: 当前 mainline 已支持

## 核心结论

- Step3.5 is split between text/MTP and VL processor work.
- NVFP4 and processor behavior are the main axes to track on the vLLM side.

## 主要代码面

- `vllm/vllm/model_executor/models/step3p5.py`
- `vllm/vllm/model_executor/models/step3p5_mtp.py`
- `vllm/vllm/model_executor/models/step3_vl.py`
- `vllm/vllm/model_executor/models/step3_text.py`

## 已合入 PR

- [#33755](https://github.com/vllm-project/vllm/pull/33755) `Enable Step3p5ForCausalLM testing`：Stabilized the core Step3.5 text runtime.
- [#34478](https://github.com/vllm-project/vllm/pull/34478) `Add NVFP4 quantization support for Step3.5-Flash`：Opened the practical quantized deployment path.
- [#37579](https://github.com/vllm-project/vllm/pull/37579) `Refactor Step3-VL processor to HF style`：Modernized the Step3-VL processor contract.

## 配套 skill

- `skills/model-optimization/vllm/vllm-step35-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-step35-optimization/references/pr-history.md`

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `vllm-project/vllm` 的 Pull Request API 和文件级 patch 重新审计 `Step 3.5`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2026-02-04 | [#33755](https://github.com/vllm-project/vllm/pull/33755) | merged | [Model] Enable Step3p5ForCausalLM testing | model wrapper, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/step3p5.py`, `tests/models/registry.py`, `docs/models/supported_models.md` |
| 2026-02-13 | [#34478](https://github.com/vllm-project/vllm/pull/34478) | merged | [Model] Add NVFP4 quantization support for Step3.5-Flash | model wrapper, MoE/router, quantization, kernel, scheduler/runtime, tests/benchmarks | `tests/kernels/moe/test_nvfp4_moe.py`, `vllm/model_executor/models/step3p5.py`, `vllm/model_executor/layers/fused_moe/cutlass_moe.py` |
| 2026-03-19 | [#37579](https://github.com/vllm-project/vllm/pull/37579) | merged | [Model] Refactor Step3-VL processor to HF style | model wrapper, multimodal/processor, scheduler/runtime | `vllm/transformers_utils/processors/step3_vl.py`, `vllm/model_executor/models/step3_vl.py`, `vllm/transformers_utils/processors/internvl.py` |

### 逐 PR 代码 diff 阅读记录

### PR #33755 - [Model] Enable Step3p5ForCausalLM testing

- 链接：https://github.com/vllm-project/vllm/pull/33755
- 状态/时间：`merged`，created 2026-02-04, merged 2026-02-07；作者 `jeejeelee`。
- 代码 diff 已读范围：`3` 个文件，`+28/-32`；代码面：model wrapper, scheduler/runtime, tests/benchmarks, docs/config；关键词：flash, moe, config, doc, expert, lora, processor, quant, spec, test。
- 代码 diff 细节：
  - `vllm/model_executor/models/step3p5.py` modified +12/-25 (37 lines); hunk: from vllm.model_executor.layers.quantization.base_config import QuantizationConfig; def __init__(; 符号: __init__, __init__
  - `tests/models/registry.py` modified +15/-6 (21 lines); hunk: def check_available_online(; def check_available_online(; 符号: check_available_online, check_available_online
  - `docs/models/supported_models.md` modified +1/-1 (2 lines); hunk: th {
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/step3p5.py`, `tests/models/registry.py`, `docs/models/supported_models.md`；patch 关键词为 flash, moe, config, doc, expert, lora。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/step3p5.py`, `tests/models/registry.py`, `docs/models/supported_models.md` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #34478 - [Model] Add NVFP4 quantization support for Step3.5-Flash

- 链接：https://github.com/vllm-project/vllm/pull/34478
- 状态/时间：`merged`，created 2026-02-13, merged 2026-02-22；作者 `tacos8me`。
- 代码 diff 已读范围：`5` 个文件，`+204/-4`；代码面：model wrapper, MoE/router, quantization, kernel, scheduler/runtime, tests/benchmarks；关键词：moe, quant, fp4, expert, config, flash, cuda, marlin, test, topk。
- 代码 diff 细节：
  - `tests/kernels/moe/test_nvfp4_moe.py` modified +126/-0 (126 lines); hunk: from vllm import _custom_ops as ops; def test_cutlass_fp4_moe_no_graph(; 符号: test_cutlass_fp4_moe_no_graph, test_cutlass_fp4_moe_swiglustep
  - `vllm/model_executor/models/step3p5.py` modified +71/-1 (72 lines); hunk: # SPDX-FileCopyrightText: Copyright contributors to the vLLM project; def __init__(; 符号: __init__, load_weights, load_weights, load_weights
  - `vllm/model_executor/layers/fused_moe/cutlass_moe.py` modified +4/-0 (4 lines); hunk: def _supports_quant_scheme(; 符号: _supports_quant_scheme, _supports_activation
  - `vllm/model_executor/layers/fused_moe/fused_marlin_moe.py` modified +3/-0 (3 lines); hunk: def _supports_quant_scheme(; 符号: _supports_quant_scheme, _supports_activation
  - `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +0/-3 (3 lines); hunk: def apply(; 符号: apply
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `tests/kernels/moe/test_nvfp4_moe.py`, `vllm/model_executor/models/step3p5.py`, `vllm/model_executor/layers/fused_moe/cutlass_moe.py`；patch 关键词为 moe, quant, fp4, expert, config, flash。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `tests/kernels/moe/test_nvfp4_moe.py`, `vllm/model_executor/models/step3p5.py`, `vllm/model_executor/layers/fused_moe/cutlass_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #37579 - [Model] Refactor Step3-VL processor to HF style

- 链接：https://github.com/vllm-project/vllm/pull/37579
- 状态/时间：`merged`，created 2026-03-19, merged 2026-03-20；作者 `DarkLight1337`。
- 代码 diff 已读范围：`4` 个文件，`+228/-160`；代码面：model wrapper, multimodal/processor, scheduler/runtime；关键词：processor, vision, config, spec。
- 代码 diff 细节：
  - `vllm/transformers_utils/processors/step3_vl.py` modified +197/-127 (324 lines); hunk: from PIL import Image; def get_num_patches(self, img_width: int, img_height: int) -> tuple[int, int]:; 符号: Step3VisionProcessor:, get_num_patches, __call__, __call__
  - `vllm/model_executor/models/step3_vl.py` modified +27/-29 (56 lines); hunk: ); class Step3VLImageEmbeddingInputs(TensorSchema):; 符号: Step3VLImageEmbeddingInputs, Step3VLProcessingInfo, get_image_processor, get_hf_processor
  - `vllm/transformers_utils/processors/internvl.py` modified +4/-3 (7 lines); hunk: def __call__(; 符号: __call__
  - `vllm/transformers_utils/processors/kimi_k25.py` modified +0/-1 (1 lines); hunk: def __init__(; 符号: __init__, __call__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/transformers_utils/processors/step3_vl.py`, `vllm/model_executor/models/step3_vl.py`, `vllm/transformers_utils/processors/internvl.py`；patch 关键词为 processor, vision, config, spec。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `vllm/transformers_utils/processors/step3_vl.py`, `vllm/model_executor/models/step3_vl.py`, `vllm/transformers_utils/processors/internvl.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：3；open PR 数：0。
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
