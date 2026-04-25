# vLLM GPT-OSS 支持与 PR 历史

本文记录 vLLM 中与 GPT-OSS 相关的模型支持、关键 PR、以及 cookbook 对应的落点。

- 状态: 当前 mainline 已支持

## 核心结论

- GPT-OSS is a flagship MoE family in vLLM.
- Quantization, expert-parallel topology, and reasoning/tool parser behavior all evolved quickly and need per-PR tracking.

## 主要代码面

- `vllm/vllm/model_executor/models/gpt_oss.py`

## 已合入 PR

- [#22327](https://github.com/vllm-project/vllm/pull/22327) `Add GPT-OSS model code and config`：Initial GPT-OSS landing in vLLM.
- [#23819](https://github.com/vllm-project/vllm/pull/23819) `Support DP+EP for GPT-OSS with FlashInfer trtllm-gen MoE`：Opened large-scale GPT-OSS serving topologies.
- [#25246](https://github.com/vllm-project/vllm/pull/25246) `Enable Eagle3 speculative decoding for GPT-OSS model`：Added draft-model acceleration.
- [#25515](https://github.com/vllm-project/vllm/pull/25515) `Structure_Tag support for gpt-oss tool-call in cot`：Improved tool calling in reasoning-mode outputs.
- [#30647](https://github.com/vllm-project/vllm/pull/30647) `Eliminate padding and slicing op for GPT-OSS with Flashinfer MXFP4 MXFP8 MoE`：Targeted the hot MXFP4/MXFP8 path for throughput.

## 配套 skill

- `skills/model-optimization/vllm/vllm-gpt-oss-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-gpt-oss-optimization/references/pr-history.md`

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `vllm-project/vllm` 的 Pull Request API 和文件级 patch 重新审计 `GPT-OSS`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2025-08-06 | [#22327](https://github.com/vllm-project/vllm/pull/22327) | merged | Add GPT-OSS model code and config [1/N] | model wrapper, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/gpt_oss.py`, `vllm/model_executor/models/config.py`, `tests/models/registry.py` |
| 2025-08-28 | [#23819](https://github.com/vllm-project/vllm/pull/23819) | merged | [Model][gpt-oss] Support DP+EP for GPT-OSS with FlashInfer trtllm-gen MoE | MoE/router, quantization, scheduler/runtime, docs/config | `vllm/model_executor/layers/fused_moe/config.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/quantization/mxfp4.py` |
| 2025-09-19 | [#25246](https://github.com/vllm-project/vllm/pull/25246) | merged | Enable Eagle3 speculative decoding for GPT-OSS model | model wrapper, scheduler/runtime, docs/config | `vllm/v1/spec_decode/eagle.py`, `vllm/model_executor/models/gpt_oss.py`, `vllm/config/speculative.py` |
| 2025-09-23 | [#25515](https://github.com/vllm-project/vllm/pull/25515) | merged | [GPT-OSS] Structure_Tag support for gpt-oss tool-call in cot | tests/benchmarks, docs/config | `tests/entrypoints/openai/test_gptoss_structural_tags_integration.py`, `tests/v1/structured_output/test_reasoning_structured_output.py`, `tests/v1/structured_output/test_gptoss_structural_tags.py` |
| 2025-12-14 | [#30647](https://github.com/vllm-project/vllm/pull/30647) | merged | [Perf] Eliminate padding and slicing op for GPT-OSS with Flashinfer MXFP4 MXFP8 MoE | MoE/router, quantization, scheduler/runtime, tests/benchmarks | `vllm/model_executor/layers/quantization/mxfp4.py`, `tests/compile/fusions_e2e/models.py`, `vllm/model_executor/layers/fused_moe/fused_moe_method_base.py` |

### 逐 PR 代码 diff 阅读记录

### PR #22327 - Add GPT-OSS model code and config [1/N]

- 链接：https://github.com/vllm-project/vllm/pull/22327
- 状态/时间：`merged`，created 2025-08-06, merged 2025-08-06；作者 `WoosukKwon`。
- 代码 diff 已读范围：`4` 个文件，`+503/-0`；代码面：model wrapper, scheduler/runtime, tests/benchmarks, docs/config；关键词：moe, attention, config, cuda, test, cache, expert, fp4, kv, processor。
- 代码 diff 细节：
  - `vllm/model_executor/models/gpt_oss.py` added +472/-0 (472 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: OAIAttention, __init__, forward, MLPBlock
  - `vllm/model_executor/models/config.py` modified +29/-0 (29 lines); hunk: def verify_and_update_config(vllm_config: "VllmConfig") -> None:; def verify_and_update_config(cls, vllm_config: "VllmConfig") -> None:; 符号: verify_and_update_config, GptOssConfig, verify_and_update_config, HybridAttentionMambaModelConfig
  - `tests/models/registry.py` modified +1/-0 (1 lines); hunk: def check_available_online(; 符号: check_available_online
  - `vllm/model_executor/models/registry.py` modified +1/-0 (1 lines); hunk: "GlmForCausalLM": ("glm", "GlmForCausalLM"),
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/gpt_oss.py`, `vllm/model_executor/models/config.py`, `tests/models/registry.py`；patch 关键词为 moe, attention, config, cuda, test, cache。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/gpt_oss.py`, `vllm/model_executor/models/config.py`, `tests/models/registry.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #23819 - [Model][gpt-oss] Support DP+EP for GPT-OSS with FlashInfer trtllm-gen MoE

- 链接：https://github.com/vllm-project/vllm/pull/23819
- 状态/时间：`merged`，created 2025-08-28, merged 2025-08-28；作者 `nvpohanh`。
- 代码 diff 已读范围：`3` 个文件，`+14/-15`；代码面：MoE/router, quantization, scheduler/runtime, docs/config；关键词：flash, moe, config, deepep, expert, fp4, quant, fp8, router, topk。
- 代码 diff 细节：
  - `vllm/model_executor/layers/fused_moe/config.py` modified +8/-7 (15 lines); hunk: def use_deepep_ll_kernels(self):; def use_deepep_ll_kernels(self):; 符号: use_deepep_ll_kernels, use_flashinfer_cutlass_kernels, make, use_deepep_ll_kernels
  - `vllm/model_executor/layers/fused_moe/layer.py` modified +4/-4 (8 lines); hunk: def __init__(; def use_deepep_ll_kernels(self):; 符号: __init__, use_deepep_ll_kernels, use_flashinfer_cutlass_kernels, update_expert_map
  - `vllm/model_executor/layers/quantization/mxfp4.py` modified +2/-4 (6 lines); hunk: def apply(; def apply(; 符号: apply, apply
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/layers/fused_moe/config.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/quantization/mxfp4.py`；patch 关键词为 flash, moe, config, deepep, expert, fp4。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/layers/fused_moe/config.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/quantization/mxfp4.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #25246 - Enable Eagle3 speculative decoding for GPT-OSS model

- 链接：https://github.com/vllm-project/vllm/pull/25246
- 状态/时间：`merged`，created 2025-09-19, merged 2025-09-22；作者 `eldarkurtic`。
- 代码 diff 已读范围：`3` 个文件，`+41/-12`；代码面：model wrapper, scheduler/runtime, docs/config；关键词：eagle, config, spec, fp4, kv。
- 代码 diff 细节：
  - `vllm/v1/spec_decode/eagle.py` modified +23/-9 (32 lines); hunk: def load_model(self, target_model: nn.Module) -> None:; 符号: load_model
  - `vllm/model_executor/models/gpt_oss.py` modified +17/-2 (19 lines); hunk: from vllm.sequence import IntermediateTensors; def __init__(; 符号: __init__, get_input_embeddings, forward, _load_weights_mxfp4
  - `vllm/config/speculative.py` modified +1/-1 (2 lines); hunk: def _verify_args(self) -> Self:; 符号: _verify_args
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/v1/spec_decode/eagle.py`, `vllm/model_executor/models/gpt_oss.py`, `vllm/config/speculative.py`；patch 关键词为 eagle, config, spec, fp4, kv。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/v1/spec_decode/eagle.py`, `vllm/model_executor/models/gpt_oss.py`, `vllm/config/speculative.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #25515 - [GPT-OSS] Structure_Tag support for gpt-oss tool-call in cot

- 链接：https://github.com/vllm-project/vllm/pull/25515
- 状态/时间：`merged`，created 2025-09-23, merged 2025-10-18；作者 `Hanchenli`。
- 代码 diff 已读范围：`14` 个文件，`+911/-32`；代码面：tests/benchmarks, docs/config；关键词：spec, test, config, scheduler, vision。
- 代码 diff 细节：
  - `tests/entrypoints/openai/test_gptoss_structural_tags_integration.py` added +280/-0 (280 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: TestGptOssStructuralTagsIntegration:, mock_tokenizer, gptoss_parser, tool_server_with_python
  - `tests/v1/structured_output/test_reasoning_structured_output.py` added +207/-0 (207 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: TestReasoningStructuredOutput:, mock_model_config, mock_scheduler_config, mock_vllm_config
  - `tests/v1/structured_output/test_gptoss_structural_tags.py` added +172/-0 (172 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: TestGptOssReasoningParser:, mock_tokenizer, reasoning_parser, mock_tool_server_empty
  - `vllm/reasoning/gptoss_reasoning_parser.py` modified +75/-1 (76 lines); hunk: # SPDX-License-Identifier: Apache-2.0; def extract_reasoning_content(; 符号: from_builtin_tool_to_tag, tag_with_builtin_funcs, GptOssReasoningParser, extract_reasoning_content
  - `vllm/v1/structured_output/backend_xgrammar.py` modified +28/-24 (52 lines); hunk: def compile_grammar(; def validate_xgrammar_grammar(sampling_params: SamplingParams) -> None:; 符号: compile_grammar, validate_xgrammar_grammar
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `tests/entrypoints/openai/test_gptoss_structural_tags_integration.py`, `tests/v1/structured_output/test_reasoning_structured_output.py`, `tests/v1/structured_output/test_gptoss_structural_tags.py`；patch 关键词为 spec, test, config, scheduler, vision。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `tests/entrypoints/openai/test_gptoss_structural_tags_integration.py`, `tests/v1/structured_output/test_reasoning_structured_output.py`, `tests/v1/structured_output/test_gptoss_structural_tags.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #30647 - [Perf] Eliminate padding and slicing op for GPT-OSS with Flashinfer MXFP4 MXFP8 MoE

- 链接：https://github.com/vllm-project/vllm/pull/30647
- 状态/时间：`merged`，created 2025-12-14, merged 2026-03-18；作者 `elvischenv`。
- 代码 diff 已读范围：`6` 个文件，`+40/-3`；代码面：MoE/router, quantization, scheduler/runtime, tests/benchmarks；关键词：moe, flash, fp4, fp8, test, config, quant, attention, cache, expert。
- 代码 diff 细节：
  - `vllm/model_executor/layers/quantization/mxfp4.py` modified +16/-1 (17 lines); hunk: def __init__(self, moe: FusedMoEConfig):; def apply_monolithic(; 符号: __init__, skip_forward_padding, create_weights, apply_monolithic
  - `tests/compile/fusions_e2e/models.py` modified +9/-0 (9 lines); hunk: # async_tp=n_layers * 2,
  - `vllm/model_executor/layers/fused_moe/fused_moe_method_base.py` modified +5/-0 (5 lines); hunk: def topk_indices_dtype(self) -> torch.dtype \| None:; 符号: topk_indices_dtype, skip_forward_padding, supports_eplb
  - `vllm/model_executor/layers/fused_moe/runner/default_moe_runner.py` modified +4/-1 (5 lines); hunk: def forward(; 符号: forward
  - `tests/compile/fusions_e2e/conftest.py` modified +4/-0 (4 lines); hunk: def run(; 符号: run
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/layers/quantization/mxfp4.py`, `tests/compile/fusions_e2e/models.py`, `vllm/model_executor/layers/fused_moe/fused_moe_method_base.py`；patch 关键词为 moe, flash, fp4, fp8, test, config。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/layers/quantization/mxfp4.py`, `tests/compile/fusions_e2e/models.py`, `vllm/model_executor/layers/fused_moe/fused_moe_method_base.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：5；open PR 数：0。
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
