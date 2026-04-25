# vLLM Hunyuan 3 Preview 支持与 PR 历史

本文记录 vLLM 中与 Hunyuan 3 Preview 相关的模型支持、关键 PR、以及仍需持续跟踪的风险点。

- 状态: 当前 mainline 仅部分支持，或只有相邻架构已落地

## 核心结论

- vLLM does not currently expose a dedicated Hunyuan 3 Preview model alias.
- The closest landed evidence is the Hunyuan dense, Hunyuan OCR, and HunyuanVL / Eagle work already in tree.

## 主要代码面

- `vllm/vllm/model_executor/models/hunyuan_v1.py`
- `vllm/vllm/model_executor/models/hunyuan_vision.py`

## 已合入 PR

- [#21368](https://github.com/vllm-project/vllm/pull/21368) `Add Hunyuan V1 Dense Model support`：Brought the dense Hunyuan line into vLLM mainline.
- [#29327](https://github.com/vllm-project/vllm/pull/29327) `Add HunyuanOCR support`：Extended the family to OCR workloads instead of text-only generation.
- [#33035](https://github.com/vllm-project/vllm/pull/33035) `Eagle3 support for HunyuanVL & Hunyuan`：Added speculative decoding support on top of the Hunyuan family.

## Open PR 雷达

- 暂无固定 open PR；需要在声称新支持前重新搜索。

## 配套 skill

- `skills/model-optimization/vllm/vllm-hunyuan3-preview-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-hunyuan3-preview-optimization/references/pr-history.md`

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `vllm-project/vllm` 的 Pull Request API 和文件级 patch 重新审计 `Hunyuan3 Preview`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2025-07-22 | [#21368](https://github.com/vllm-project/vllm/pull/21368) | merged | [Model] add Hunyuan V1 Dense Model support. | model wrapper, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/hunyuan_v1.py`, `vllm/model_executor/models/registry.py`, `tests/models/registry.py` |
| 2025-11-24 | [#29327](https://github.com/vllm-project/vllm/pull/29327) | merged | [Model] Add HunyuanOCR support | model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/hunyuan_vision.py`, `vllm/transformers_utils/processors/hunyuan_vl_image.py`, `vllm/transformers_utils/configs/hunyuan_vl.py` |
| 2026-01-25 | [#33035](https://github.com/vllm-project/vllm/pull/33035) | merged | feature: support eagle3 for HunyuanVL & Hunyuan | model wrapper, multimodal/processor, scheduler/runtime, docs/config | `vllm/model_executor/models/hunyuan_v1.py`, `vllm/v1/spec_decode/eagle.py`, `vllm/config/speculative.py` |

### 逐 PR 代码 diff 阅读记录

### PR #21368 - [Model] add Hunyuan V1 Dense Model support.

- 链接：https://github.com/vllm-project/vllm/pull/21368
- 状态/时间：`merged`，created 2025-07-22, merged 2025-07-23；作者 `kzjeef`。
- 代码 diff 已读范围：`4` 个文件，`+57/-19`；代码面：model wrapper, scheduler/runtime, tests/benchmarks, docs/config；关键词：moe, expert, fp8, test, attention, config, doc, kv, lora, quant。
- 代码 diff 细节：
  - `vllm/model_executor/models/hunyuan_v1.py` renamed +52/-18 (70 lines); hunk: make_layers); def __init__(; 符号: _is_moe, _get_cla_factor, __init__, __init__
  - `vllm/model_executor/models/registry.py` modified +2/-1 (3 lines); hunk: "GraniteMoeSharedForCausalLM": ("granitemoeshared", "GraniteMoeSharedForCausalLM"), # noqa: E501
  - `tests/models/registry.py` modified +2/-0 (2 lines); hunk: def check_available_online(; 符号: check_available_online
  - `docs/models/supported_models.md` modified +1/-0 (1 lines); hunk: th {
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/hunyuan_v1.py`, `vllm/model_executor/models/registry.py`, `tests/models/registry.py`；patch 关键词为 moe, expert, fp8, test, attention, config。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/hunyuan_v1.py`, `vllm/model_executor/models/registry.py`, `tests/models/registry.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #29327 - [Model] Add HunyuanOCR support

- 链接：https://github.com/vllm-project/vllm/pull/29327
- 状态/时间：`merged`，created 2025-11-24, merged 2025-11-25；作者 `Isotr0py`。
- 代码 diff 已读范围：`18` 个文件，`+2415/-4`；代码面：model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config；关键词：spec, attention, vision, cache, config, processor, doc, kv, eagle, expert。
- 代码 diff 细节：
  - `vllm/model_executor/models/hunyuan_vision.py` added +1028/-0 (1028 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: HunYuanVLImagePixelInputs, HunYuanVLImageEmbeddingInputs, HunYuanVisionMLP, __init__
  - `vllm/transformers_utils/processors/hunyuan_vl_image.py` added +477/-0 (477 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: for, smart_resize, HunYuanVLImageProcessor, __init__
  - `vllm/transformers_utils/configs/hunyuan_vl.py` added +322/-0 (322 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: HunYuanVLVisionConfig, __init__, HunYuanVLTextConfig, to
  - `vllm/transformers_utils/processors/hunyuan_vl.py` added +233/-0 (233 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: HunYuanVLProcessor, __init__, __call__, batch_decode
  - `vllm/v1/worker/gpu_model_runner.py` modified +103/-1 (104 lines); hunk: from vllm.forward_context import BatchDescriptor, set_forward_context; def __init__(; 符号: __init__, __init__, _get_positions, _make_buffer
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/hunyuan_vision.py`, `vllm/transformers_utils/processors/hunyuan_vl_image.py`, `vllm/transformers_utils/configs/hunyuan_vl.py`；patch 关键词为 spec, attention, vision, cache, config, processor。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/hunyuan_vision.py`, `vllm/transformers_utils/processors/hunyuan_vl_image.py`, `vllm/transformers_utils/configs/hunyuan_vl.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #33035 - feature: support eagle3 for HunyuanVL & Hunyuan

- 链接：https://github.com/vllm-project/vllm/pull/33035
- 状态/时间：`merged`，created 2026-01-25, merged 2026-01-27；作者 `irisliu10`。
- 代码 diff 已读范围：`4` 个文件，`+49/-3`；代码面：model wrapper, multimodal/processor, scheduler/runtime, docs/config；关键词：eagle, config, lora, spec, attention, cuda, expert, kv, moe, quant。
- 代码 diff 细节：
  - `vllm/model_executor/models/hunyuan_v1.py` modified +17/-2 (19 lines); hunk: from vllm.sequence import IntermediateTensors; def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):; 符号: __init__, embed_input_ids, forward, forward
  - `vllm/v1/spec_decode/eagle.py` modified +15/-0 (15 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__, __init__, _get_positions
  - `vllm/config/speculative.py` modified +8/-1 (9 lines); hunk: def _verify_args(self) -> Self:; 符号: _verify_args
  - `vllm/model_executor/models/hunyuan_vision.py` modified +9/-0 (9 lines); hunk: from .interfaces import (; class HunYuanVLForConditionalGeneration(; 符号: HunYuanVLForConditionalGeneration, embed_multimodal, set_aux_hidden_state_layers, get_eagle3_aux_hidden_state_layers
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/hunyuan_v1.py`, `vllm/v1/spec_decode/eagle.py`, `vllm/config/speculative.py`；patch 关键词为 eagle, config, lora, spec, attention, cuda。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/hunyuan_v1.py`, `vllm/v1/spec_decode/eagle.py`, `vllm/config/speculative.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：3；open PR 数：0。
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
