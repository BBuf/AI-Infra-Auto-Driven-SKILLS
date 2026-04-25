# vLLM GPT-OSS Support and PR History

This note tracks the vLLM runtime, key PRs, and cookbook-facing touchpoints for GPT-OSS.

- Status: supported on current mainline

## Key Conclusions

- GPT-OSS is a flagship MoE family in vLLM.
- Quantization, expert-parallel topology, and reasoning/tool parser behavior all evolved quickly and need per-PR tracking.

## Main Runtime Surfaces

- `vllm/vllm/model_executor/models/gpt_oss.py`

## Landed PRs

- [#22327](https://github.com/vllm-project/vllm/pull/22327) `Add GPT-OSS model code and config`: Initial GPT-OSS landing in vLLM.
- [#23819](https://github.com/vllm-project/vllm/pull/23819) `Support DP+EP for GPT-OSS with FlashInfer trtllm-gen MoE`: Opened large-scale GPT-OSS serving topologies.
- [#25246](https://github.com/vllm-project/vllm/pull/25246) `Enable Eagle3 speculative decoding for GPT-OSS model`: Added draft-model acceleration.
- [#25515](https://github.com/vllm-project/vllm/pull/25515) `Structure_Tag support for gpt-oss tool-call in cot`: Improved tool calling in reasoning-mode outputs.
- [#30647](https://github.com/vllm-project/vllm/pull/30647) `Eliminate padding and slicing op for GPT-OSS with Flashinfer MXFP4 MXFP8 MoE`: Targeted the hot MXFP4/MXFP8 path for throughput.

## Matching Skill

- `skills/model-optimization/vllm/vllm-gpt-oss-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-gpt-oss-optimization/references/pr-history.md`

<!-- MODEL_PR_DIFF_AUDIT:START en -->

## PR Diff Audit Cards (2026-04-25 rebuild)

This section re-audits `GPT-OSS` against `vllm-project/vllm` Pull Request metadata and file-level patches. Acceptance rule: every PR needs status, code surface, file-level diff digest, support/optimization interpretation, and verification risk notes; if no public PR is found, keep an explicit no-match conclusion instead of inventing history.

### Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2025-08-06 | [#22327](https://github.com/vllm-project/vllm/pull/22327) | merged | Add GPT-OSS model code and config [1/N] | model wrapper, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/gpt_oss.py`, `vllm/model_executor/models/config.py`, `tests/models/registry.py` |
| 2025-08-28 | [#23819](https://github.com/vllm-project/vllm/pull/23819) | merged | [Model][gpt-oss] Support DP+EP for GPT-OSS with FlashInfer trtllm-gen MoE | MoE/router, quantization, scheduler/runtime, docs/config | `vllm/model_executor/layers/fused_moe/config.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/quantization/mxfp4.py` |
| 2025-09-19 | [#25246](https://github.com/vllm-project/vllm/pull/25246) | merged | Enable Eagle3 speculative decoding for GPT-OSS model | model wrapper, scheduler/runtime, docs/config | `vllm/v1/spec_decode/eagle.py`, `vllm/model_executor/models/gpt_oss.py`, `vllm/config/speculative.py` |
| 2025-09-23 | [#25515](https://github.com/vllm-project/vllm/pull/25515) | merged | [GPT-OSS] Structure_Tag support for gpt-oss tool-call in cot | tests/benchmarks, docs/config | `tests/entrypoints/openai/test_gptoss_structural_tags_integration.py`, `tests/v1/structured_output/test_reasoning_structured_output.py`, `tests/v1/structured_output/test_gptoss_structural_tags.py` |
| 2025-12-14 | [#30647](https://github.com/vllm-project/vllm/pull/30647) | merged | [Perf] Eliminate padding and slicing op for GPT-OSS with Flashinfer MXFP4 MXFP8 MoE | MoE/router, quantization, scheduler/runtime, tests/benchmarks | `vllm/model_executor/layers/quantization/mxfp4.py`, `tests/compile/fusions_e2e/models.py`, `vllm/model_executor/layers/fused_moe/fused_moe_method_base.py` |

### File-level PR diff reading notes

### PR #22327 - Add GPT-OSS model code and config [1/N]

- Link: https://github.com/vllm-project/vllm/pull/22327
- Status/date: `merged`, created 2025-08-06, merged 2025-08-06; author `WoosukKwon`.
- Diff scope read: `4` files, `+503/-0`; areas: model wrapper, scheduler/runtime, tests/benchmarks, docs/config; keywords: moe, attention, config, cuda, test, cache, expert, fp4, kv, processor.
- Code diff details:
  - `vllm/model_executor/models/gpt_oss.py` added +472/-0 (472 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: OAIAttention, __init__, forward, MLPBlock
  - `vllm/model_executor/models/config.py` modified +29/-0 (29 lines); hunks: def verify_and_update_config(vllm_config: "VllmConfig") -> None:; def verify_and_update_config(cls, vllm_config: "VllmConfig") -> None:; symbols: verify_and_update_config, GptOssConfig, verify_and_update_config, HybridAttentionMambaModelConfig
  - `tests/models/registry.py` modified +1/-0 (1 lines); hunks: def check_available_online(; symbols: check_available_online
  - `vllm/model_executor/models/registry.py` modified +1/-0 (1 lines); hunks: "GlmForCausalLM": ("glm", "GlmForCausalLM"),
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/gpt_oss.py`, `vllm/model_executor/models/config.py`, `tests/models/registry.py`; keywords observed in patches: moe, attention, config, cuda, test, cache. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/gpt_oss.py`, `vllm/model_executor/models/config.py`, `tests/models/registry.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23819 - [Model][gpt-oss] Support DP+EP for GPT-OSS with FlashInfer trtllm-gen MoE

- Link: https://github.com/vllm-project/vllm/pull/23819
- Status/date: `merged`, created 2025-08-28, merged 2025-08-28; author `nvpohanh`.
- Diff scope read: `3` files, `+14/-15`; areas: MoE/router, quantization, scheduler/runtime, docs/config; keywords: flash, moe, config, deepep, expert, fp4, quant, fp8, router, topk.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/config.py` modified +8/-7 (15 lines); hunks: def use_deepep_ll_kernels(self):; def use_deepep_ll_kernels(self):; symbols: use_deepep_ll_kernels, use_flashinfer_cutlass_kernels, make, use_deepep_ll_kernels
  - `vllm/model_executor/layers/fused_moe/layer.py` modified +4/-4 (8 lines); hunks: def __init__(; def use_deepep_ll_kernels(self):; symbols: __init__, use_deepep_ll_kernels, use_flashinfer_cutlass_kernels, update_expert_map
  - `vllm/model_executor/layers/quantization/mxfp4.py` modified +2/-4 (6 lines); hunks: def apply(; def apply(; symbols: apply, apply
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/layers/fused_moe/config.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/quantization/mxfp4.py`; keywords observed in patches: flash, moe, config, deepep, expert, fp4. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/layers/fused_moe/config.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/quantization/mxfp4.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #25246 - Enable Eagle3 speculative decoding for GPT-OSS model

- Link: https://github.com/vllm-project/vllm/pull/25246
- Status/date: `merged`, created 2025-09-19, merged 2025-09-22; author `eldarkurtic`.
- Diff scope read: `3` files, `+41/-12`; areas: model wrapper, scheduler/runtime, docs/config; keywords: eagle, config, spec, fp4, kv.
- Code diff details:
  - `vllm/v1/spec_decode/eagle.py` modified +23/-9 (32 lines); hunks: def load_model(self, target_model: nn.Module) -> None:; symbols: load_model
  - `vllm/model_executor/models/gpt_oss.py` modified +17/-2 (19 lines); hunks: from vllm.sequence import IntermediateTensors; def __init__(; symbols: __init__, get_input_embeddings, forward, _load_weights_mxfp4
  - `vllm/config/speculative.py` modified +1/-1 (2 lines); hunks: def _verify_args(self) -> Self:; symbols: _verify_args
- Optimization/support interpretation: The concrete diff surface is `vllm/v1/spec_decode/eagle.py`, `vllm/model_executor/models/gpt_oss.py`, `vllm/config/speculative.py`; keywords observed in patches: eagle, config, spec, fp4, kv. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/v1/spec_decode/eagle.py`, `vllm/model_executor/models/gpt_oss.py`, `vllm/config/speculative.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #25515 - [GPT-OSS] Structure_Tag support for gpt-oss tool-call in cot

- Link: https://github.com/vllm-project/vllm/pull/25515
- Status/date: `merged`, created 2025-09-23, merged 2025-10-18; author `Hanchenli`.
- Diff scope read: `14` files, `+911/-32`; areas: tests/benchmarks, docs/config; keywords: spec, test, config, scheduler, vision.
- Code diff details:
  - `tests/entrypoints/openai/test_gptoss_structural_tags_integration.py` added +280/-0 (280 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: TestGptOssStructuralTagsIntegration:, mock_tokenizer, gptoss_parser, tool_server_with_python
  - `tests/v1/structured_output/test_reasoning_structured_output.py` added +207/-0 (207 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: TestReasoningStructuredOutput:, mock_model_config, mock_scheduler_config, mock_vllm_config
  - `tests/v1/structured_output/test_gptoss_structural_tags.py` added +172/-0 (172 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: TestGptOssReasoningParser:, mock_tokenizer, reasoning_parser, mock_tool_server_empty
  - `vllm/reasoning/gptoss_reasoning_parser.py` modified +75/-1 (76 lines); hunks: # SPDX-License-Identifier: Apache-2.0; def extract_reasoning_content(; symbols: from_builtin_tool_to_tag, tag_with_builtin_funcs, GptOssReasoningParser, extract_reasoning_content
  - `vllm/v1/structured_output/backend_xgrammar.py` modified +28/-24 (52 lines); hunks: def compile_grammar(; def validate_xgrammar_grammar(sampling_params: SamplingParams) -> None:; symbols: compile_grammar, validate_xgrammar_grammar
- Optimization/support interpretation: The concrete diff surface is `tests/entrypoints/openai/test_gptoss_structural_tags_integration.py`, `tests/v1/structured_output/test_reasoning_structured_output.py`, `tests/v1/structured_output/test_gptoss_structural_tags.py`; keywords observed in patches: spec, test, config, scheduler, vision. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `tests/entrypoints/openai/test_gptoss_structural_tags_integration.py`, `tests/v1/structured_output/test_reasoning_structured_output.py`, `tests/v1/structured_output/test_gptoss_structural_tags.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #30647 - [Perf] Eliminate padding and slicing op for GPT-OSS with Flashinfer MXFP4 MXFP8 MoE

- Link: https://github.com/vllm-project/vllm/pull/30647
- Status/date: `merged`, created 2025-12-14, merged 2026-03-18; author `elvischenv`.
- Diff scope read: `6` files, `+40/-3`; areas: MoE/router, quantization, scheduler/runtime, tests/benchmarks; keywords: moe, flash, fp4, fp8, test, config, quant, attention, cache, expert.
- Code diff details:
  - `vllm/model_executor/layers/quantization/mxfp4.py` modified +16/-1 (17 lines); hunks: def __init__(self, moe: FusedMoEConfig):; def apply_monolithic(; symbols: __init__, skip_forward_padding, create_weights, apply_monolithic
  - `tests/compile/fusions_e2e/models.py` modified +9/-0 (9 lines); hunks: # async_tp=n_layers * 2,
  - `vllm/model_executor/layers/fused_moe/fused_moe_method_base.py` modified +5/-0 (5 lines); hunks: def topk_indices_dtype(self) -> torch.dtype \| None:; symbols: topk_indices_dtype, skip_forward_padding, supports_eplb
  - `vllm/model_executor/layers/fused_moe/runner/default_moe_runner.py` modified +4/-1 (5 lines); hunks: def forward(; symbols: forward
  - `tests/compile/fusions_e2e/conftest.py` modified +4/-0 (4 lines); hunks: def run(; symbols: run
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/layers/quantization/mxfp4.py`, `tests/compile/fusions_e2e/models.py`, `vllm/model_executor/layers/fused_moe/fused_moe_method_base.py`; keywords observed in patches: moe, flash, fp4, fp8, test, config. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/layers/quantization/mxfp4.py`, `tests/compile/fusions_e2e/models.py`, `vllm/model_executor/layers/fused_moe/fused_moe_method_base.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


### Gap and optimization follow-up

- Covered PRs: 5; open PRs: 0.
- Any future PR must add both the timeline row and the file-level diff card; title-only summaries are not acceptable.

<!-- MODEL_PR_DIFF_AUDIT:END en -->
