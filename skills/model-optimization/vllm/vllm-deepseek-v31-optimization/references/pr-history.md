# vLLM DeepSeek V3.1 PR History

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Support status: supported on current mainline
- Scope: DeepSeek V3.1 parser, scale-format, DeepGEMM, and reasoning-tooling deltas layered on top of the base DeepSeek V3 runtime.

This family inherits the base runtime context from `deepseek-v3-r1`; this file records only the delta that is specific to `deepseek-v31`.

## Landed PRs

### PR #23454 - Support DeepSeek-V3.1 tool call

- Link: https://github.com/vllm-project/vllm/pull/23454
- Why it mattered: Added the first V3.1-specific tool-call parser surface to vLLM.
- Runtime path: vllm/vllm/model_executor/models/deepseek_v2.py, vllm/vllm/model_executor/layers/quantization/utils/flashinfer_utils.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #23666 - Add Hopper DeepGEMM E8M0 for DeepSeekV3.1 scale_fmt

- Link: https://github.com/vllm-project/vllm/pull/23666
- Why it mattered: Tuned the scale-format path used by DeepGEMM-based DeepSeek V3.1 kernels.
- Runtime path: vllm/vllm/model_executor/models/deepseek_v2.py, vllm/vllm/model_executor/layers/quantization/utils/flashinfer_utils.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #25589 - Add DeepSeek-V3.1 reasoning parser

- Link: https://github.com/vllm-project/vllm/pull/25589
- Why it mattered: Separated V3.1 reasoning output handling from generic DeepSeek parsing.
- Runtime path: vllm/vllm/model_executor/models/deepseek_v2.py, vllm/vllm/model_executor/layers/quantization/utils/flashinfer_utils.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #32361 - Fix DeepSeek-V3.1 + DeepGEMM incompatible scale shapes

- Link: https://github.com/vllm-project/vllm/pull/32361
- Why it mattered: Patched a concrete shape mismatch between newer checkpoints and DeepGEMM assumptions.
- Runtime path: vllm/vllm/model_executor/models/deepseek_v2.py, vllm/vllm/model_executor/layers/quantization/utils/flashinfer_utils.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

## Open PR Radar

- No specific open PR is pinned here; rerun GitHub PR search before asserting new support.

<!-- MODEL_PR_DIFF_AUDIT:START reference -->

# VLLM DeepSeek V3.1 PR Diff Audit Reference

This reference is rebuilt from the same audited PR metadata used by `model-pr-optimization-history`. It is intentionally concise but keeps a file-level diff digest for every indexed PR.

## Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2025-08-23 | [#23454](https://github.com/vllm-project/vllm/pull/23454) | merged | Support DeepSeek-V3.1 tool call | docs/config | `vllm/entrypoints/openai/tool_parsers/deepseekv31_tool_parser.py`, `examples/tool_chat_template_deepseekv31.jinja`, `docs/features/tool_calling.md` |
| 2025-08-26 | [#23666](https://github.com/vllm-project/vllm/pull/23666) | merged | [Feature] Add Hopper DeepGEMM E8M0 for DeepSeekV3.1 scale_fmt | MoE/router, quantization, kernel, scheduler/runtime, tests/benchmarks, docs/config | `vllm/utils/deep_gemm.py`, `vllm/transformers_utils/config.py`, `vllm/model_executor/layers/quantization/fp8.py` |
| 2025-09-24 | [#25589](https://github.com/vllm-project/vllm/pull/25589) | merged | [Model] Add DeepSeek-V3.1 reasoning parser (split from PR #24972) | tests/benchmarks, docs/config | `tests/reasoning/test_deepseekv3_reasoning_parser.py`, `vllm/reasoning/deepseek_v3_reasoning_parser.py`, `vllm/reasoning/identity_reasoning_parser.py` |
| 2026-01-15 | [#32361](https://github.com/vllm-project/vllm/pull/32361) | merged | [BugFix] Fix DeepSeek-V3.1 + DeepGEMM incompatible scale shapes | quantization, scheduler/runtime | `vllm/model_executor/layers/quantization/utils/quant_utils.py` |

## Diff Cards

### PR #23454 - Support DeepSeek-V3.1 tool call

- Link: https://github.com/vllm-project/vllm/pull/23454
- Status/date: `merged`, created 2025-08-23, merged 2025-08-23; author `Xu-Wenqing`.
- Diff scope read: `4` files, `+468/-0`; areas: docs/config; keywords: kv, doc, moe.
- Code diff details:
  - `vllm/entrypoints/openai/tool_parsers/deepseekv31_tool_parser.py` added +367/-0 (367 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: DeepSeekV31ToolParser, __init__, extract_tool_calls, extract_tool_calls_streaming
  - `examples/tool_chat_template_deepseekv31.jinja` added +91/-0 (91 lines); hunks: +{% if not add_generation_prompt is defined %}
  - `docs/features/tool_calling.md` modified +8/-0 (8 lines); hunks: Supported models:
  - `vllm/entrypoints/openai/tool_parsers/__init__.py` modified +2/-0 (2 lines); hunks: from .abstract_tool_parser import ToolParser, ToolParserManager; "PythonicToolParser",
- Optimization/support interpretation: The concrete diff surface is `vllm/entrypoints/openai/tool_parsers/deepseekv31_tool_parser.py`, `examples/tool_chat_template_deepseekv31.jinja`, `docs/features/tool_calling.md`; keywords observed in patches: kv, doc, moe. Impact reading: docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/entrypoints/openai/tool_parsers/deepseekv31_tool_parser.py`, `examples/tool_chat_template_deepseekv31.jinja`, `docs/features/tool_calling.md`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23666 - [Feature] Add Hopper DeepGEMM E8M0 for DeepSeekV3.1 scale_fmt

- Link: https://github.com/vllm-project/vllm/pull/23666
- Status/date: `merged`, created 2025-08-26, merged 2025-08-27; author `yewentao256`.
- Diff scope read: `10` files, `+68/-53`; areas: MoE/router, quantization, kernel, scheduler/runtime, tests/benchmarks, docs/config; keywords: moe, fp8, expert, quant, topk, triton, config, cuda, flash, spec.
- Code diff details:
  - `vllm/utils/deep_gemm.py` modified +24/-29 (53 lines); hunks: def is_deep_gemm_supported() -> bool:; def fp8_gemm_nt(*args, **kwargs):; symbols: is_deep_gemm_supported, is_blackwell_deep_gemm_e8m0_used, is_deep_gemm_e8m0_used, GPU
  - `vllm/transformers_utils/config.py` modified +18/-0 (18 lines); hunks: def get_config(; symbols: get_config
  - `vllm/model_executor/layers/quantization/fp8.py` modified +4/-5 (9 lines); hunks: from vllm.platforms import current_platform; def process_weights_after_loading(self, layer: Module) -> None:; symbols: process_weights_after_loading, process_weights_after_loading, process_weights_after_loading
  - `vllm/envs.py` modified +7/-1 (8 lines); hunks: VLLM_TPU_USING_PATHWAYS: bool = False; def get_vllm_port() -> Optional[int]:; symbols: get_vllm_port, compute_hash
  - `tests/kernels/moe/test_deepep_deepgemm_moe.py` modified +3/-4 (7 lines); hunks: FusedMoEModularKernel); def _test_deepep_deepgemm_moe(; symbols: _test_deepep_deepgemm_moe, test_ht_deepep_deepgemm_moe, test_ht_deepep_deepgemm_moe, test_ll_deepep_deepgemm_moe
- Optimization/support interpretation: The concrete diff surface is `vllm/utils/deep_gemm.py`, `vllm/transformers_utils/config.py`, `vllm/model_executor/layers/quantization/fp8.py`; keywords observed in patches: moe, fp8, expert, quant, topk, triton. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/utils/deep_gemm.py`, `vllm/transformers_utils/config.py`, `vllm/model_executor/layers/quantization/fp8.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #25589 - [Model] Add DeepSeek-V3.1 reasoning parser (split from PR #24972)

- Link: https://github.com/vllm-project/vllm/pull/25589
- Status/date: `merged`, created 2025-09-24, merged 2025-10-15; author `taohui`.
- Diff scope read: `6` files, `+215/-3`; areas: tests/benchmarks, docs/config; keywords: kv, doc, moe, spec, test.
- Code diff details:
  - `tests/reasoning/test_deepseekv3_reasoning_parser.py` added +76/-0 (76 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: tokenizer, test_parser_selection, test_identity_reasoning_parser_basic
  - `vllm/reasoning/deepseek_v3_reasoning_parser.py` added +66/-0 (66 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: DeepSeekV3ReasoningParser, __init__, is_reasoning_end, extract_content_ids
  - `vllm/reasoning/identity_reasoning_parser.py` added +58/-0 (58 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: IdentityReasoningParser, __init__, is_reasoning_end, extract_content_ids
  - `vllm/entrypoints/openai/serving_chat.py` modified +8/-2 (10 lines); hunks: async def chat_completion_stream_generator(; async def chat_completion_full_generator(; symbols: chat_completion_stream_generator, chat_completion_full_generator
  - `docs/features/reasoning_outputs.md` modified +3/-1 (4 lines); hunks: vLLM currently supports the following reasoning models:; vLLM currently supports the following reasoning models:
- Optimization/support interpretation: The concrete diff surface is `tests/reasoning/test_deepseekv3_reasoning_parser.py`, `vllm/reasoning/deepseek_v3_reasoning_parser.py`, `vllm/reasoning/identity_reasoning_parser.py`; keywords observed in patches: kv, doc, moe, spec, test. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `tests/reasoning/test_deepseekv3_reasoning_parser.py`, `vllm/reasoning/deepseek_v3_reasoning_parser.py`, `vllm/reasoning/identity_reasoning_parser.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #32361 - [BugFix] Fix DeepSeek-V3.1 + DeepGEMM incompatible scale shapes

- Link: https://github.com/vllm-project/vllm/pull/32361
- Status/date: `merged`, created 2026-01-15, merged 2026-01-15; author `LucasWilkinson`.
- Diff scope read: `1` files, `+3/-0`; areas: quantization, scheduler/runtime; keywords: fp8, marlin, quant.
- Code diff details:
  - `vllm/model_executor/layers/quantization/utils/quant_utils.py` modified +3/-0 (3 lines); hunks: def get_and_maybe_dequant_weights(; symbols: get_and_maybe_dequant_weights
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/layers/quantization/utils/quant_utils.py`; keywords observed in patches: fp8, marlin, quant. Impact reading: quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/layers/quantization/utils/quant_utils.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


<!-- MODEL_PR_DIFF_AUDIT:END reference -->
