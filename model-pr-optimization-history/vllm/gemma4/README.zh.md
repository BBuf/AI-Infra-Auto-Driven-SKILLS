# vLLM Gemma 4 支持与 PR 历史

本文记录 vLLM 中与 Gemma 4 相关的模型支持、关键 PR、以及 cookbook 对应的落点。

- 状态: 当前 mainline 已支持

## 核心结论

- Gemma 4 is a genuinely multi-surface family: text, MoE, multimodal, tool calling, and speculative decoding all matter.
- Fast prefill, quantized MoE, and tool-parser correctness are the main areas that changed rapidly after bring-up.

## 主要代码面

- `vllm/vllm/model_executor/models/gemma4.py`
- `vllm/vllm/model_executor/models/gemma4_mm.py`

## 已合入 PR

- [#38826](https://github.com/vllm-project/vllm/pull/38826) `Implement Google Gemma 4 architecture support`：Initial Gemma 4 text/MoE/multimodal landing.
- [#38879](https://github.com/vllm-project/vllm/pull/38879) `Enable Fast Prefill Optimization`：Added YOCO KV-sharing based fast prefill for Gemma4.
- [#39045](https://github.com/vllm-project/vllm/pull/39045) `Support quantized MoE`：Extended Gemma4 to quantized MoE checkpoints.
- [#38844](https://github.com/vllm-project/vllm/pull/38844) `Enable Gemma4ForCausalLM to load LoRA adapters correctly`：Fixed adapter naming/load behavior.
- [#39450](https://github.com/vllm-project/vllm/pull/39450) `Add Gemma4 Eagle3 support`：Enabled speculative decode for Gemma4.

## 配套 skill

- `skills/model-optimization/vllm/vllm-gemma4-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-gemma4-optimization/references/pr-history.md`

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `vllm-project/vllm` 的 Pull Request API 和文件级 patch 重新审计 `Gemma 4`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2026-04-02 | [#38826](https://github.com/vllm-project/vllm/pull/38826) | merged | feat(models): implement Google Gemma 4 architecture support (MoE, Multimodal, Reasoning, Tool-Use) | model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/gemma4_mm.py`, `vllm/model_executor/models/gemma4.py`, `vllm/tool_parsers/gemma4_tool_parser.py` |
| 2026-04-02 | [#38844](https://github.com/vllm-project/vllm/pull/38844) | merged | [Gemma4][Bugfix]: Enable Gemma4ForCasualLM to load lora adapters correctly | model wrapper, scheduler/runtime, tests/benchmarks | `tests/lora/test_lora_checkpoints.py`, `vllm/model_executor/models/gemma4.py` |
| 2026-04-03 | [#38879](https://github.com/vllm-project/vllm/pull/38879) | merged | [Gemma4] Enable Fast Prefill Optimization | model wrapper, scheduler/runtime | `vllm/model_executor/models/gemma4.py` |
| 2026-04-05 | [#39045](https://github.com/vllm-project/vllm/pull/39045) | merged | [Gemma4] Support quantized MoE | model wrapper, scheduler/runtime | `vllm/model_executor/models/gemma4.py` |
| 2026-04-09 | [#39450](https://github.com/vllm-project/vllm/pull/39450) | merged | Add Gemma4 Eagle3 support | model wrapper, attention/backend, scheduler/runtime, docs/config | `vllm/model_executor/models/gemma4.py`, `vllm/model_executor/models/gemma4_mm.py`, `vllm/v1/core/single_type_kv_cache_manager.py` |

### 逐 PR 代码 diff 阅读记录

### PR #38826 - feat(models): implement Google Gemma 4 architecture support (MoE, Multimodal, Reasoning, Tool-Use)

- 链接：https://github.com/vllm-project/vllm/pull/38826
- 状态/时间：`merged`，created 2026-04-02, merged 2026-04-02；作者 `lucianommartins`。
- 代码 diff 已读范围：`20` 个文件，`+5051/-1`；代码面：model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config；关键词：spec, config, attention, cache, expert, kv, lora, moe, processor, quant。
- 代码 diff 细节：
  - `vllm/model_executor/models/gemma4_mm.py` added +1341/-0 (1341 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: Gemma4ImagePixelInputs, Gemma4AudioInputs, Gemma4VideoInputs, Gemma4ProcessingInfo
  - `vllm/model_executor/models/gemma4.py` added +1239/-0 (1239 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: _get_text_config, Gemma4MLP, __init__, forward
  - `vllm/tool_parsers/gemma4_tool_parser.py` added +724/-0 (724 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: _parse_gemma4_value, _parse_gemma4_args, _parse_gemma4_array, Gemma4ToolParser
  - `tests/tool_parsers/test_gemma4_tool_parser.py` added +504/-0 (504 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: mock_tokenizer, parser, mock_request, TestParseGemma4Args:
  - `vllm/model_executor/models/gemma4_utils.py` added +292/-0 (292 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: parse_thinking_output, _strip_thought_label, _clean_answer, _parse_tool_arguments
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/gemma4_mm.py`, `vllm/model_executor/models/gemma4.py`, `vllm/tool_parsers/gemma4_tool_parser.py`；patch 关键词为 spec, config, attention, cache, expert, kv。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/gemma4_mm.py`, `vllm/model_executor/models/gemma4.py`, `vllm/tool_parsers/gemma4_tool_parser.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #38844 - [Gemma4][Bugfix]: Enable Gemma4ForCasualLM to load lora adapters correctly

- 链接：https://github.com/vllm-project/vllm/pull/38844
- 状态/时间：`merged`，created 2026-04-02, merged 2026-04-11；作者 `ShubyM`。
- 代码 diff 已读范围：`2` 个文件，`+40/-0`；代码面：model wrapper, scheduler/runtime, tests/benchmarks；关键词：expert, lora, moe, attention, eagle, kv, test。
- 代码 diff 细节：
  - `tests/lora/test_lora_checkpoints.py` modified +23/-0 (23 lines); hunk: from vllm.lora.lora_model import LoRAModel; def test_lora_weights_mapping(baichuan_lora_files):; 符号: test_lora_weights_mapping, test_gemma4_lora_weights_mapping, test_gemma4_moe_lora_weights_mapping
  - `vllm/model_executor/models/gemma4.py` modified +17/-0 (17 lines); hunk: ); def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:; 符号: load_weights, Gemma4ForCausalLM
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `tests/lora/test_lora_checkpoints.py`, `vllm/model_executor/models/gemma4.py`；patch 关键词为 expert, lora, moe, attention, eagle, kv。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `tests/lora/test_lora_checkpoints.py`, `vllm/model_executor/models/gemma4.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #38879 - [Gemma4] Enable Fast Prefill Optimization

- 链接：https://github.com/vllm-project/vllm/pull/38879
- 状态/时间：`merged`，created 2026-04-03, merged 2026-04-06；作者 `LucasWilkinson`。
- 代码 diff 已读范围：`1` 个文件，`+369/-47`；代码面：model wrapper, scheduler/runtime；关键词：attention, cache, config, cuda, expert, kv, lora, scheduler。
- 代码 diff 细节：
  - `vllm/model_executor/models/gemma4.py` modified +369/-47 (416 lines); hunk: """Gemma 4 model implementation for vLLM."""; get_tensor_model_parallel_rank,; 符号: forward, _run_decoder_layers, Gemma4SelfDecoderLayers, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/gemma4.py`；patch 关键词为 attention, cache, config, cuda, expert, kv。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/gemma4.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #39045 - [Gemma4] Support quantized MoE

- 链接：https://github.com/vllm-project/vllm/pull/39045
- 状态/时间：`merged`，created 2026-04-05, merged 2026-04-09；作者 `dsikka`。
- 代码 diff 已读范围：`1` 个文件，`+34/-14`；代码面：model wrapper, scheduler/runtime；关键词：config, expert, moe, quant。
- 代码 diff 细节：
  - `vllm/model_executor/models/gemma4.py` modified +34/-14 (48 lines); hunk: def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:; def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str; 符号: load_weights, load_weights, load_weights, _weight_iterator
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/gemma4.py`；patch 关键词为 config, expert, moe, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/gemma4.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #39450 - Add Gemma4 Eagle3 support

- 链接：https://github.com/vllm-project/vllm/pull/39450
- 状态/时间：`merged`，created 2026-04-09, merged 2026-04-10；作者 `fynnsu`。
- 代码 diff 已读范围：`5` 个文件，`+43/-10`；代码面：model wrapper, attention/backend, scheduler/runtime, docs/config；关键词：eagle, config, kv, spec, cache, attention, expert, flash, lora, moe。
- 代码 diff 细节：
  - `vllm/model_executor/models/gemma4.py` modified +20/-5 (25 lines); hunk: from vllm.sequence import IntermediateTensors; def forward(; 符号: forward, Gemma4Model, Gemma4Model, __init__
  - `vllm/model_executor/models/gemma4_mm.py` modified +12/-2 (14 lines); hunk: from vllm.sequence import IntermediateTensors; def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:; 符号: forward, Gemma4ForConditionalGeneration, Gemma4ForConditionalGeneration
  - `vllm/v1/core/single_type_kv_cache_manager.py` modified +9/-3 (12 lines); hunk: def find_longest_cache_hit(; 符号: find_longest_cache_hit, get_num_skipped_tokens
  - `vllm/config/speculative.py` modified +1/-0 (1 lines); hunk: def _verify_args(self) -> Self:; 符号: _verify_args
  - `vllm/v1/spec_decode/eagle.py` modified +1/-0 (1 lines); hunk: def load_model(self, target_model: nn.Module) -> None:; 符号: load_model
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/gemma4.py`, `vllm/model_executor/models/gemma4_mm.py`, `vllm/v1/core/single_type_kv_cache_manager.py`；patch 关键词为 eagle, config, kv, spec, cache, attention。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/gemma4.py`, `vllm/model_executor/models/gemma4_mm.py`, `vllm/v1/core/single_type_kv_cache_manager.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：5；open PR 数：0。
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
