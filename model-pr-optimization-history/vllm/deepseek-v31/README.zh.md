# vLLM DeepSeek V3.1 支持与 PR 历史

本文记录 vLLM 中与 DeepSeek V3.1 相关的模型支持、关键 PR、以及仍需持续跟踪的风险点。

- 状态: 当前 mainline 已支持
- 该家族继承 `deepseek-v3-r1` 的基础 runtime，这里只记录增量 PR。

## 核心结论

- V3.1 mostly reuses the base V3 runtime and adds parser plus scale-format correctness work.
- The practical blast radius is in tool calling, DeepGEMM scale handling, and reasoning-parser behavior.

## 主要代码面

- `vllm/vllm/model_executor/models/deepseek_v2.py`
- `vllm/vllm/model_executor/layers/quantization/utils/flashinfer_utils.py`

## 已合入 PR

- [#23454](https://github.com/vllm-project/vllm/pull/23454) `Support DeepSeek-V3.1 tool call`：Added the first V3.1-specific tool-call parser surface to vLLM.
- [#23666](https://github.com/vllm-project/vllm/pull/23666) `Add Hopper DeepGEMM E8M0 for DeepSeekV3.1 scale_fmt`：Tuned the scale-format path used by DeepGEMM-based DeepSeek V3.1 kernels.
- [#25589](https://github.com/vllm-project/vllm/pull/25589) `Add DeepSeek-V3.1 reasoning parser`：Separated V3.1 reasoning output handling from generic DeepSeek parsing.
- [#32361](https://github.com/vllm-project/vllm/pull/32361) `Fix DeepSeek-V3.1 + DeepGEMM incompatible scale shapes`：Patched a concrete shape mismatch between newer checkpoints and DeepGEMM assumptions.

## Open PR 雷达

- 暂无固定 open PR；需要在声称新支持前重新搜索。

## 配套 skill

- `skills/model-optimization/vllm/vllm-deepseek-v31-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-deepseek-v31-optimization/references/pr-history.md`

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `vllm-project/vllm` 的 Pull Request API 和文件级 patch 重新审计 `DeepSeek V3.1`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2025-08-23 | [#23454](https://github.com/vllm-project/vllm/pull/23454) | merged | Support DeepSeek-V3.1 tool call | docs/config | `vllm/entrypoints/openai/tool_parsers/deepseekv31_tool_parser.py`, `examples/tool_chat_template_deepseekv31.jinja`, `docs/features/tool_calling.md` |
| 2025-08-26 | [#23666](https://github.com/vllm-project/vllm/pull/23666) | merged | [Feature] Add Hopper DeepGEMM E8M0 for DeepSeekV3.1 scale_fmt | MoE/router, quantization, kernel, scheduler/runtime, tests/benchmarks, docs/config | `vllm/utils/deep_gemm.py`, `vllm/transformers_utils/config.py`, `vllm/model_executor/layers/quantization/fp8.py` |
| 2025-09-24 | [#25589](https://github.com/vllm-project/vllm/pull/25589) | merged | [Model] Add DeepSeek-V3.1 reasoning parser (split from PR #24972) | tests/benchmarks, docs/config | `tests/reasoning/test_deepseekv3_reasoning_parser.py`, `vllm/reasoning/deepseek_v3_reasoning_parser.py`, `vllm/reasoning/identity_reasoning_parser.py` |
| 2026-01-15 | [#32361](https://github.com/vllm-project/vllm/pull/32361) | merged | [BugFix] Fix DeepSeek-V3.1 + DeepGEMM incompatible scale shapes | quantization, scheduler/runtime | `vllm/model_executor/layers/quantization/utils/quant_utils.py` |

### 逐 PR 代码 diff 阅读记录

### PR #23454 - Support DeepSeek-V3.1 tool call

- 链接：https://github.com/vllm-project/vllm/pull/23454
- 状态/时间：`merged`，created 2025-08-23, merged 2025-08-23；作者 `Xu-Wenqing`。
- 代码 diff 已读范围：`4` 个文件，`+468/-0`；代码面：docs/config；关键词：kv, doc, moe。
- 代码 diff 细节：
  - `vllm/entrypoints/openai/tool_parsers/deepseekv31_tool_parser.py` added +367/-0 (367 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: DeepSeekV31ToolParser, __init__, extract_tool_calls, extract_tool_calls_streaming
  - `examples/tool_chat_template_deepseekv31.jinja` added +91/-0 (91 lines); hunk: +{% if not add_generation_prompt is defined %}
  - `docs/features/tool_calling.md` modified +8/-0 (8 lines); hunk: Supported models:
  - `vllm/entrypoints/openai/tool_parsers/__init__.py` modified +2/-0 (2 lines); hunk: from .abstract_tool_parser import ToolParser, ToolParserManager; "PythonicToolParser",
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/entrypoints/openai/tool_parsers/deepseekv31_tool_parser.py`, `examples/tool_chat_template_deepseekv31.jinja`, `docs/features/tool_calling.md`；patch 关键词为 kv, doc, moe。影响判断：文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/entrypoints/openai/tool_parsers/deepseekv31_tool_parser.py`, `examples/tool_chat_template_deepseekv31.jinja`, `docs/features/tool_calling.md` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #23666 - [Feature] Add Hopper DeepGEMM E8M0 for DeepSeekV3.1 scale_fmt

- 链接：https://github.com/vllm-project/vllm/pull/23666
- 状态/时间：`merged`，created 2025-08-26, merged 2025-08-27；作者 `yewentao256`。
- 代码 diff 已读范围：`10` 个文件，`+68/-53`；代码面：MoE/router, quantization, kernel, scheduler/runtime, tests/benchmarks, docs/config；关键词：moe, fp8, expert, quant, topk, triton, config, cuda, flash, spec。
- 代码 diff 细节：
  - `vllm/utils/deep_gemm.py` modified +24/-29 (53 lines); hunk: def is_deep_gemm_supported() -> bool:; def fp8_gemm_nt(*args, **kwargs):; 符号: is_deep_gemm_supported, is_blackwell_deep_gemm_e8m0_used, is_deep_gemm_e8m0_used, GPU
  - `vllm/transformers_utils/config.py` modified +18/-0 (18 lines); hunk: def get_config(; 符号: get_config
  - `vllm/model_executor/layers/quantization/fp8.py` modified +4/-5 (9 lines); hunk: from vllm.platforms import current_platform; def process_weights_after_loading(self, layer: Module) -> None:; 符号: process_weights_after_loading, process_weights_after_loading, process_weights_after_loading
  - `vllm/envs.py` modified +7/-1 (8 lines); hunk: VLLM_TPU_USING_PATHWAYS: bool = False; def get_vllm_port() -> Optional[int]:; 符号: get_vllm_port, compute_hash
  - `tests/kernels/moe/test_deepep_deepgemm_moe.py` modified +3/-4 (7 lines); hunk: FusedMoEModularKernel); def _test_deepep_deepgemm_moe(; 符号: _test_deepep_deepgemm_moe, test_ht_deepep_deepgemm_moe, test_ht_deepep_deepgemm_moe, test_ll_deepep_deepgemm_moe
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/utils/deep_gemm.py`, `vllm/transformers_utils/config.py`, `vllm/model_executor/layers/quantization/fp8.py`；patch 关键词为 moe, fp8, expert, quant, topk, triton。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/utils/deep_gemm.py`, `vllm/transformers_utils/config.py`, `vllm/model_executor/layers/quantization/fp8.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #25589 - [Model] Add DeepSeek-V3.1 reasoning parser (split from PR #24972)

- 链接：https://github.com/vllm-project/vllm/pull/25589
- 状态/时间：`merged`，created 2025-09-24, merged 2025-10-15；作者 `taohui`。
- 代码 diff 已读范围：`6` 个文件，`+215/-3`；代码面：tests/benchmarks, docs/config；关键词：kv, doc, moe, spec, test。
- 代码 diff 细节：
  - `tests/reasoning/test_deepseekv3_reasoning_parser.py` added +76/-0 (76 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: tokenizer, test_parser_selection, test_identity_reasoning_parser_basic
  - `vllm/reasoning/deepseek_v3_reasoning_parser.py` added +66/-0 (66 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: DeepSeekV3ReasoningParser, __init__, is_reasoning_end, extract_content_ids
  - `vllm/reasoning/identity_reasoning_parser.py` added +58/-0 (58 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: IdentityReasoningParser, __init__, is_reasoning_end, extract_content_ids
  - `vllm/entrypoints/openai/serving_chat.py` modified +8/-2 (10 lines); hunk: async def chat_completion_stream_generator(; async def chat_completion_full_generator(; 符号: chat_completion_stream_generator, chat_completion_full_generator
  - `docs/features/reasoning_outputs.md` modified +3/-1 (4 lines); hunk: vLLM currently supports the following reasoning models:; vLLM currently supports the following reasoning models:
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `tests/reasoning/test_deepseekv3_reasoning_parser.py`, `vllm/reasoning/deepseek_v3_reasoning_parser.py`, `vllm/reasoning/identity_reasoning_parser.py`；patch 关键词为 kv, doc, moe, spec, test。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `tests/reasoning/test_deepseekv3_reasoning_parser.py`, `vllm/reasoning/deepseek_v3_reasoning_parser.py`, `vllm/reasoning/identity_reasoning_parser.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #32361 - [BugFix] Fix DeepSeek-V3.1 + DeepGEMM incompatible scale shapes

- 链接：https://github.com/vllm-project/vllm/pull/32361
- 状态/时间：`merged`，created 2026-01-15, merged 2026-01-15；作者 `LucasWilkinson`。
- 代码 diff 已读范围：`1` 个文件，`+3/-0`；代码面：quantization, scheduler/runtime；关键词：fp8, marlin, quant。
- 代码 diff 细节：
  - `vllm/model_executor/layers/quantization/utils/quant_utils.py` modified +3/-0 (3 lines); hunk: def get_and_maybe_dequant_weights(; 符号: get_and_maybe_dequant_weights
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/layers/quantization/utils/quant_utils.py`；patch 关键词为 fp8, marlin, quant。影响判断：量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/layers/quantization/utils/quant_utils.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：4；open PR 数：0。
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
