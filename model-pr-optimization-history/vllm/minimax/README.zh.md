# vLLM MiniMax M1 / M2 / VL 支持与 PR 历史

本文记录 vLLM 中与 MiniMax M1 / M2 / VL 相关的模型支持、关键 PR、以及仍需持续跟踪的风险点。

- 状态: 当前 mainline 已支持

## 核心结论

- MiniMax support evolved from the text-01 path into M1/M2 and VL variants.
- Today the key production surfaces are linear-attention correctness, VL processor behavior, LoRA, and Eagle3 on M2.

## 主要代码面

- `vllm/vllm/model_executor/models/minimax_text_01.py`
- `vllm/vllm/model_executor/models/minimax_m2.py`
- `vllm/vllm/model_executor/models/minimax_vl_01.py`

## 已合入 PR

- [#13454](https://github.com/vllm-project/vllm/pull/13454) `Support MiniMaxText01 model inference`：Landed the original MiniMax text runtime.
- [#16328](https://github.com/vllm-project/vllm/pull/16328) `support MiniMax-VL-01 model`：Added the multimodal MiniMax-VL path.
- [#19677](https://github.com/vllm-project/vllm/pull/19677) `Add support for MiniMaxM1ForCausalLM`：Connected the M1 checkpoint alias to the shared MiniMax runtime.
- [#27535](https://github.com/vllm-project/vllm/pull/27535) `Support MiniMax-M2 Model`：Brought the M2 generation into mainline.
- [#32763](https://github.com/vllm-project/vllm/pull/32763) `Complete LoRA support for MiniMaxM2`：Finished missing adapter wiring in the M2 family.
- [#37512](https://github.com/vllm-project/vllm/pull/37512) `MiniMax-M2: add Eagle3 speculative decoding support`：Enabled the draft-model acceleration path for MiniMax M2.

## Open PR 雷达

- 暂无固定 open PR；需要在声称新支持前重新搜索。

## 配套 skill

- `skills/model-optimization/vllm/vllm-minimax-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-minimax-optimization/references/pr-history.md`

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `vllm-project/vllm` 的 Pull Request API 和文件级 patch 重新审计 `MiniMax M2 series`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2025-02-18 | [#13454](https://github.com/vllm-project/vllm/pull/13454) | merged | [Model][MiniMaxText01] Support MiniMaxText01 model inference | model wrapper, kernel, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/minimax_text_01.py`, `vllm/model_executor/layers/lightning_attn.py`, `tests/kernels/test_lightning_attn.py` |
| 2025-04-09 | [#16328](https://github.com/vllm-project/vllm/pull/16328) | merged | [Model] support MiniMax-VL-01 model | model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/minimax_vl_01.py`, `tests/models/multimodal/processing/test_minimax_vl_01.py`, `vllm/transformers_utils/configs/minimax_vl_01.py` |
| 2025-06-16 | [#19677](https://github.com/vllm-project/vllm/pull/19677) | merged | [Model] Add support for MiniMaxM1ForCausalLM (shares architecture with MiniMaxText01ForCausalLM) | model wrapper, scheduler/runtime, tests/benchmarks, docs/config | `tests/models/registry.py`, `docs/models/supported_models.md`, `vllm/model_executor/models/registry.py` |
| 2025-10-26 | [#27535](https://github.com/vllm-project/vllm/pull/27535) | merged | [Model][MiniMax-M2] Support MiniMax-M2 Model | model wrapper, scheduler/runtime, tests/benchmarks | `vllm/entrypoints/openai/tool_parsers/minimax_m2_tool_parser.py`, `vllm/model_executor/models/minimax_m2.py`, `vllm/reasoning/minimax_m2_reasoning_parser.py` |
| 2026-01-21 | [#32763](https://github.com/vllm-project/vllm/pull/32763) | merged | feat: Complete LoRA support for MiniMaxM2 Fixes #32736 | model wrapper, scheduler/runtime, docs/config | `vllm/model_executor/models/minimax_m2.py`, `docs/models/supported_models.md` |
| 2026-03-19 | [#37512](https://github.com/vllm-project/vllm/pull/37512) | merged | MiniMax-M2: add Eagle3 speculative decoding support | model wrapper, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/minimax_m2.py`, `tests/models/registry.py`, `vllm/config/speculative.py` |

### 逐 PR 代码 diff 阅读记录

### PR #13454 - [Model][MiniMaxText01] Support MiniMaxText01 model inference

- 链接：https://github.com/vllm-project/vllm/pull/13454
- 状态/时间：`merged`，created 2025-02-18, merged 2025-04-01；作者 `ZZBoom`。
- 代码 diff 已读范围：`11` 个文件，`+2440/-130`；代码面：model wrapper, kernel, scheduler/runtime, tests/benchmarks, docs/config；关键词：cache, attention, cuda, config, kv, triton, flash, scheduler, expert, lora。
- 代码 diff 细节：
  - `vllm/model_executor/models/minimax_text_01.py` added +1273/-0 (1273 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: replace_weight_name, weight_loader_with_alias, wrapper, inner_func
  - `vllm/model_executor/layers/lightning_attn.py` added +651/-0 (651 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: _fwd_diag_kernel, _fwd_kv_parallel, _fwd_kv_reduce, _fwd_none_diag_kernel
  - `tests/kernels/test_lightning_attn.py` added +286/-0 (286 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: reference_lightning_attention, reference_linear_decode, test_linear_decode_forward_triton, test_linear_decode_forward_triton_with_padding
  - `vllm/model_executor/models/constant_size_cache.py` added +136/-0 (136 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: ConstantSizeCache, for, __init__, cache
  - `vllm/model_executor/models/mamba_cache.py` modified +21/-111 (132 lines); hunk: # SPDX-License-Identifier: Apache-2.0; def at_layer_idx(self, layer_idx):; 符号: at_layer_idx, MambaCacheManager:, MambaCacheManager, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/minimax_text_01.py`, `vllm/model_executor/layers/lightning_attn.py`, `tests/kernels/test_lightning_attn.py`；patch 关键词为 cache, attention, cuda, config, kv, triton。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/minimax_text_01.py`, `vllm/model_executor/layers/lightning_attn.py`, `tests/kernels/test_lightning_attn.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #16328 - [Model] support MiniMax-VL-01 model

- 链接：https://github.com/vllm-project/vllm/pull/16328
- 状态/时间：`merged`，created 2025-04-09, merged 2025-04-29；作者 `qscqesze`。
- 代码 diff 已读范围：`11` 个文件，`+954/-19`；代码面：model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config；关键词：config, vision, attention, processor, test, cache, expert, kv, spec, flash。
- 代码 diff 细节：
  - `vllm/model_executor/models/minimax_vl_01.py` added +615/-0 (615 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: MaxImageTokenMeta:, MiniMaxVL01ImagePixelInputs, MiniMaxVL01ImageEmbeddingInputs, image_size_to_num_patches
  - `tests/models/multimodal/processing/test_minimax_vl_01.py` added +99/-0 (99 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: test_processor_override, _validate_image_prompt_replacements_one, _test_image_prompt_replacements, test_processor_prompt_replacements_regression
  - `vllm/transformers_utils/configs/minimax_vl_01.py` added +70/-0 (70 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: MiniMaxVL01Config, __init__
  - `vllm/transformers_utils/configs/minimax_text_01.py` added +69/-0 (69 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: MiniMaxText01Config, __init__
  - `vllm/model_executor/models/minimax_text_01.py` modified +53/-14 (67 lines); hunk: import copy; def _forward(; 符号: _forward, forward, _prefill_and_mix_infer, _prefill_and_mix_infer
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/minimax_vl_01.py`, `tests/models/multimodal/processing/test_minimax_vl_01.py`, `vllm/transformers_utils/configs/minimax_vl_01.py`；patch 关键词为 config, vision, attention, processor, test, cache。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/minimax_vl_01.py`, `tests/models/multimodal/processing/test_minimax_vl_01.py`, `vllm/transformers_utils/configs/minimax_vl_01.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19677 - [Model] Add support for MiniMaxM1ForCausalLM (shares architecture with MiniMaxText01ForCausalLM)

- 链接：https://github.com/vllm-project/vllm/pull/19677
- 状态/时间：`merged`，created 2025-06-16, merged 2025-06-16；作者 `qscqesze`。
- 代码 diff 已读范围：`3` 个文件，`+4/-0`；代码面：model wrapper, scheduler/runtime, tests/benchmarks, docs/config；关键词：doc, spec, test。
- 代码 diff 细节：
  - `tests/models/registry.py` modified +2/-0 (2 lines); hunk: def check_available_online(; 符号: check_available_online
  - `docs/models/supported_models.md` modified +1/-0 (1 lines); hunk: Specified using `--task generate`.
  - `vllm/model_executor/models/registry.py` modified +1/-0 (1 lines); hunk: "AquilaForCausalLM": ("llama", "LlamaForCausalLM"), # AquilaChat2; 符号: name, name
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `tests/models/registry.py`, `docs/models/supported_models.md`, `vllm/model_executor/models/registry.py`；patch 关键词为 doc, spec, test。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `tests/models/registry.py`, `docs/models/supported_models.md`, `vllm/model_executor/models/registry.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #27535 - [Model][MiniMax-M2] Support MiniMax-M2 Model

- 链接：https://github.com/vllm-project/vllm/pull/27535
- 状态/时间：`merged`，created 2025-10-26, merged 2025-10-26；作者 `rogeryoungh`。
- 代码 diff 已读范围：`7` 个文件，`+1306/-0`；代码面：model wrapper, scheduler/runtime, tests/benchmarks；关键词：config, attention, cache, expert, flash, fp8, kv, moe, processor, quant。
- 代码 diff 细节：
  - `vllm/entrypoints/openai/tool_parsers/minimax_m2_tool_parser.py` added +644/-0 (644 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: MinimaxM2ToolParser, __init__, type, _generate_tool_call_id
  - `vllm/model_executor/models/minimax_m2.py` added +585/-0 (585 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: MiniMaxM2MoE, __init__, ebias_weight_loader, forward
  - `vllm/reasoning/minimax_m2_reasoning_parser.py` added +69/-0 (69 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: MiniMaxM2ReasoningParser, start_token, end_token, MiniMaxM2AppendThinkReasoningParser
  - `tests/models/registry.py` modified +3/-0 (3 lines); hunk: def check_available_online(; 符号: check_available_online
  - `vllm/entrypoints/openai/tool_parsers/__init__.py` modified +2/-0 (2 lines); hunk: from .llama4_pythonic_tool_parser import Llama4PythonicToolParser; "SeedOssToolParser",
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/entrypoints/openai/tool_parsers/minimax_m2_tool_parser.py`, `vllm/model_executor/models/minimax_m2.py`, `vllm/reasoning/minimax_m2_reasoning_parser.py`；patch 关键词为 config, attention, cache, expert, flash, fp8。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `vllm/entrypoints/openai/tool_parsers/minimax_m2_tool_parser.py`, `vllm/model_executor/models/minimax_m2.py`, `vllm/reasoning/minimax_m2_reasoning_parser.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #32763 - feat: Complete LoRA support for MiniMaxM2 Fixes #32736

- 链接：https://github.com/vllm-project/vllm/pull/32763
- 状态/时间：`merged`，created 2026-01-21, merged 2026-01-24；作者 `Chenhao-Guan`。
- 代码 diff 已读范围：`2` 个文件，`+11/-3`；代码面：model wrapper, scheduler/runtime, docs/config；关键词：config, doc, flash, kv, lora。
- 代码 diff 细节：
  - `vllm/model_executor/models/minimax_m2.py` modified +10/-2 (12 lines); hunk: ); def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:; 符号: load_weights, MiniMaxM2ForCausalLM, MiniMaxM2ForCausalLM, __init__
  - `docs/models/supported_models.md` modified +1/-1 (2 lines); hunk: th {
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/minimax_m2.py`, `docs/models/supported_models.md`；patch 关键词为 config, doc, flash, kv, lora。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/minimax_m2.py`, `docs/models/supported_models.md` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #37512 - MiniMax-M2: add Eagle3 speculative decoding support

- 链接：https://github.com/vllm-project/vllm/pull/37512
- 状态/时间：`merged`，created 2026-03-19, merged 2026-04-06；作者 `liuchenbing2026`。
- 代码 diff 已读范围：`4` 个文件，`+24/-5`；代码面：model wrapper, scheduler/runtime, tests/benchmarks, docs/config；关键词：eagle, config, flash, spec, expert, kv, lora, test。
- 代码 diff 细节：
  - `vllm/model_executor/models/minimax_m2.py` modified +16/-5 (21 lines); hunk: """Inference-only MiniMaxM2 model."""; ); 符号: forward, MiniMaxM2Model, MiniMaxM2Model, __init__
  - `tests/models/registry.py` modified +6/-0 (6 lines); hunk: def check_available_online(; 符号: check_available_online
  - `vllm/config/speculative.py` modified +1/-0 (1 lines); hunk: def _verify_args(self) -> Self:; 符号: _verify_args
  - `vllm/model_executor/models/registry.py` modified +1/-0 (1 lines); hunk: "EagleMiniCPMForCausalLM": ("minicpm_eagle", "EagleMiniCPMForCausalLM"),
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/minimax_m2.py`, `tests/models/registry.py`, `vllm/config/speculative.py`；patch 关键词为 eagle, config, flash, spec, expert, kv。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/minimax_m2.py`, `tests/models/registry.py`, `vllm/config/speculative.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：6；open PR 数：0。
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
