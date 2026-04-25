# vLLM MiniMax M1 / M2 / VL PR History

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Support status: supported on current mainline
- Scope: MiniMaxText01, MiniMax-M1, MiniMax-M2, MiniMax-VL-01, LoRA, and Eagle3 support in vLLM.

## Landed PRs

### PR #13454 - Support MiniMaxText01 model inference

- Link: https://github.com/vllm-project/vllm/pull/13454
- Why it mattered: Landed the original MiniMax text runtime.
- Runtime path: vllm/vllm/model_executor/models/minimax_text_01.py, vllm/vllm/model_executor/models/minimax_m2.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #16328 - support MiniMax-VL-01 model

- Link: https://github.com/vllm-project/vllm/pull/16328
- Why it mattered: Added the multimodal MiniMax-VL path.
- Runtime path: vllm/vllm/model_executor/models/minimax_text_01.py, vllm/vllm/model_executor/models/minimax_m2.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #19677 - Add support for MiniMaxM1ForCausalLM

- Link: https://github.com/vllm-project/vllm/pull/19677
- Why it mattered: Connected the M1 checkpoint alias to the shared MiniMax runtime.
- Runtime path: vllm/vllm/model_executor/models/minimax_text_01.py, vllm/vllm/model_executor/models/minimax_m2.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #27535 - Support MiniMax-M2 Model

- Link: https://github.com/vllm-project/vllm/pull/27535
- Why it mattered: Brought the M2 generation into mainline.
- Runtime path: vllm/vllm/model_executor/models/minimax_text_01.py, vllm/vllm/model_executor/models/minimax_m2.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #32763 - Complete LoRA support for MiniMaxM2

- Link: https://github.com/vllm-project/vllm/pull/32763
- Why it mattered: Finished missing adapter wiring in the M2 family.
- Runtime path: vllm/vllm/model_executor/models/minimax_text_01.py, vllm/vllm/model_executor/models/minimax_m2.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #37512 - MiniMax-M2: add Eagle3 speculative decoding support

- Link: https://github.com/vllm-project/vllm/pull/37512
- Why it mattered: Enabled the draft-model acceleration path for MiniMax M2.
- Runtime path: vllm/vllm/model_executor/models/minimax_text_01.py, vllm/vllm/model_executor/models/minimax_m2.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

## Open PR Radar

- No specific open PR is pinned here; rerun GitHub PR search before asserting new support.

<!-- MODEL_PR_DIFF_AUDIT:START reference -->

# VLLM MiniMax M2 series PR Diff Audit Reference

This reference is rebuilt from the same audited PR metadata used by `model-pr-optimization-history`. It is intentionally concise but keeps a file-level diff digest for every indexed PR.

## Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2025-02-18 | [#13454](https://github.com/vllm-project/vllm/pull/13454) | merged | [Model][MiniMaxText01] Support MiniMaxText01 model inference | model wrapper, kernel, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/minimax_text_01.py`, `vllm/model_executor/layers/lightning_attn.py`, `tests/kernels/test_lightning_attn.py` |
| 2025-04-09 | [#16328](https://github.com/vllm-project/vllm/pull/16328) | merged | [Model] support MiniMax-VL-01 model | model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/minimax_vl_01.py`, `tests/models/multimodal/processing/test_minimax_vl_01.py`, `vllm/transformers_utils/configs/minimax_vl_01.py` |
| 2025-06-16 | [#19677](https://github.com/vllm-project/vllm/pull/19677) | merged | [Model] Add support for MiniMaxM1ForCausalLM (shares architecture with MiniMaxText01ForCausalLM) | model wrapper, scheduler/runtime, tests/benchmarks, docs/config | `tests/models/registry.py`, `docs/models/supported_models.md`, `vllm/model_executor/models/registry.py` |
| 2025-10-26 | [#27535](https://github.com/vllm-project/vllm/pull/27535) | merged | [Model][MiniMax-M2] Support MiniMax-M2 Model | model wrapper, scheduler/runtime, tests/benchmarks | `vllm/entrypoints/openai/tool_parsers/minimax_m2_tool_parser.py`, `vllm/model_executor/models/minimax_m2.py`, `vllm/reasoning/minimax_m2_reasoning_parser.py` |
| 2026-01-21 | [#32763](https://github.com/vllm-project/vllm/pull/32763) | merged | feat: Complete LoRA support for MiniMaxM2 Fixes #32736 | model wrapper, scheduler/runtime, docs/config | `vllm/model_executor/models/minimax_m2.py`, `docs/models/supported_models.md` |
| 2026-03-19 | [#37512](https://github.com/vllm-project/vllm/pull/37512) | merged | MiniMax-M2: add Eagle3 speculative decoding support | model wrapper, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/minimax_m2.py`, `tests/models/registry.py`, `vllm/config/speculative.py` |

## Diff Cards

### PR #13454 - [Model][MiniMaxText01] Support MiniMaxText01 model inference

- Link: https://github.com/vllm-project/vllm/pull/13454
- Status/date: `merged`, created 2025-02-18, merged 2025-04-01; author `ZZBoom`.
- Diff scope read: `11` files, `+2440/-130`; areas: model wrapper, kernel, scheduler/runtime, tests/benchmarks, docs/config; keywords: cache, attention, cuda, config, kv, triton, flash, scheduler, expert, lora.
- Code diff details:
  - `vllm/model_executor/models/minimax_text_01.py` added +1273/-0 (1273 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: replace_weight_name, weight_loader_with_alias, wrapper, inner_func
  - `vllm/model_executor/layers/lightning_attn.py` added +651/-0 (651 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: _fwd_diag_kernel, _fwd_kv_parallel, _fwd_kv_reduce, _fwd_none_diag_kernel
  - `tests/kernels/test_lightning_attn.py` added +286/-0 (286 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: reference_lightning_attention, reference_linear_decode, test_linear_decode_forward_triton, test_linear_decode_forward_triton_with_padding
  - `vllm/model_executor/models/constant_size_cache.py` added +136/-0 (136 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: ConstantSizeCache, for, __init__, cache
  - `vllm/model_executor/models/mamba_cache.py` modified +21/-111 (132 lines); hunks: # SPDX-License-Identifier: Apache-2.0; def at_layer_idx(self, layer_idx):; symbols: at_layer_idx, MambaCacheManager:, MambaCacheManager, __init__
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/minimax_text_01.py`, `vllm/model_executor/layers/lightning_attn.py`, `tests/kernels/test_lightning_attn.py`; keywords observed in patches: cache, attention, cuda, config, kv, triton. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/minimax_text_01.py`, `vllm/model_executor/layers/lightning_attn.py`, `tests/kernels/test_lightning_attn.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #16328 - [Model] support MiniMax-VL-01 model

- Link: https://github.com/vllm-project/vllm/pull/16328
- Status/date: `merged`, created 2025-04-09, merged 2025-04-29; author `qscqesze`.
- Diff scope read: `11` files, `+954/-19`; areas: model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config; keywords: config, vision, attention, processor, test, cache, expert, kv, spec, flash.
- Code diff details:
  - `vllm/model_executor/models/minimax_vl_01.py` added +615/-0 (615 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: MaxImageTokenMeta:, MiniMaxVL01ImagePixelInputs, MiniMaxVL01ImageEmbeddingInputs, image_size_to_num_patches
  - `tests/models/multimodal/processing/test_minimax_vl_01.py` added +99/-0 (99 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: test_processor_override, _validate_image_prompt_replacements_one, _test_image_prompt_replacements, test_processor_prompt_replacements_regression
  - `vllm/transformers_utils/configs/minimax_vl_01.py` added +70/-0 (70 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: MiniMaxVL01Config, __init__
  - `vllm/transformers_utils/configs/minimax_text_01.py` added +69/-0 (69 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: MiniMaxText01Config, __init__
  - `vllm/model_executor/models/minimax_text_01.py` modified +53/-14 (67 lines); hunks: import copy; def _forward(; symbols: _forward, forward, _prefill_and_mix_infer, _prefill_and_mix_infer
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/minimax_vl_01.py`, `tests/models/multimodal/processing/test_minimax_vl_01.py`, `vllm/transformers_utils/configs/minimax_vl_01.py`; keywords observed in patches: config, vision, attention, processor, test, cache. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/minimax_vl_01.py`, `tests/models/multimodal/processing/test_minimax_vl_01.py`, `vllm/transformers_utils/configs/minimax_vl_01.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19677 - [Model] Add support for MiniMaxM1ForCausalLM (shares architecture with MiniMaxText01ForCausalLM)

- Link: https://github.com/vllm-project/vllm/pull/19677
- Status/date: `merged`, created 2025-06-16, merged 2025-06-16; author `qscqesze`.
- Diff scope read: `3` files, `+4/-0`; areas: model wrapper, scheduler/runtime, tests/benchmarks, docs/config; keywords: doc, spec, test.
- Code diff details:
  - `tests/models/registry.py` modified +2/-0 (2 lines); hunks: def check_available_online(; symbols: check_available_online
  - `docs/models/supported_models.md` modified +1/-0 (1 lines); hunks: Specified using `--task generate`.
  - `vllm/model_executor/models/registry.py` modified +1/-0 (1 lines); hunks: "AquilaForCausalLM": ("llama", "LlamaForCausalLM"), # AquilaChat2; symbols: name, name
- Optimization/support interpretation: The concrete diff surface is `tests/models/registry.py`, `docs/models/supported_models.md`, `vllm/model_executor/models/registry.py`; keywords observed in patches: doc, spec, test. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `tests/models/registry.py`, `docs/models/supported_models.md`, `vllm/model_executor/models/registry.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #27535 - [Model][MiniMax-M2] Support MiniMax-M2 Model

- Link: https://github.com/vllm-project/vllm/pull/27535
- Status/date: `merged`, created 2025-10-26, merged 2025-10-26; author `rogeryoungh`.
- Diff scope read: `7` files, `+1306/-0`; areas: model wrapper, scheduler/runtime, tests/benchmarks; keywords: config, attention, cache, expert, flash, fp8, kv, moe, processor, quant.
- Code diff details:
  - `vllm/entrypoints/openai/tool_parsers/minimax_m2_tool_parser.py` added +644/-0 (644 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: MinimaxM2ToolParser, __init__, type, _generate_tool_call_id
  - `vllm/model_executor/models/minimax_m2.py` added +585/-0 (585 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: MiniMaxM2MoE, __init__, ebias_weight_loader, forward
  - `vllm/reasoning/minimax_m2_reasoning_parser.py` added +69/-0 (69 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: MiniMaxM2ReasoningParser, start_token, end_token, MiniMaxM2AppendThinkReasoningParser
  - `tests/models/registry.py` modified +3/-0 (3 lines); hunks: def check_available_online(; symbols: check_available_online
  - `vllm/entrypoints/openai/tool_parsers/__init__.py` modified +2/-0 (2 lines); hunks: from .llama4_pythonic_tool_parser import Llama4PythonicToolParser; "SeedOssToolParser",
- Optimization/support interpretation: The concrete diff surface is `vllm/entrypoints/openai/tool_parsers/minimax_m2_tool_parser.py`, `vllm/model_executor/models/minimax_m2.py`, `vllm/reasoning/minimax_m2_reasoning_parser.py`; keywords observed in patches: config, attention, cache, expert, flash, fp8. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `vllm/entrypoints/openai/tool_parsers/minimax_m2_tool_parser.py`, `vllm/model_executor/models/minimax_m2.py`, `vllm/reasoning/minimax_m2_reasoning_parser.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #32763 - feat: Complete LoRA support for MiniMaxM2 Fixes #32736

- Link: https://github.com/vllm-project/vllm/pull/32763
- Status/date: `merged`, created 2026-01-21, merged 2026-01-24; author `Chenhao-Guan`.
- Diff scope read: `2` files, `+11/-3`; areas: model wrapper, scheduler/runtime, docs/config; keywords: config, doc, flash, kv, lora.
- Code diff details:
  - `vllm/model_executor/models/minimax_m2.py` modified +10/-2 (12 lines); hunks: ); def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:; symbols: load_weights, MiniMaxM2ForCausalLM, MiniMaxM2ForCausalLM, __init__
  - `docs/models/supported_models.md` modified +1/-1 (2 lines); hunks: th {
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/minimax_m2.py`, `docs/models/supported_models.md`; keywords observed in patches: config, doc, flash, kv, lora. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/minimax_m2.py`, `docs/models/supported_models.md`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #37512 - MiniMax-M2: add Eagle3 speculative decoding support

- Link: https://github.com/vllm-project/vllm/pull/37512
- Status/date: `merged`, created 2026-03-19, merged 2026-04-06; author `liuchenbing2026`.
- Diff scope read: `4` files, `+24/-5`; areas: model wrapper, scheduler/runtime, tests/benchmarks, docs/config; keywords: eagle, config, flash, spec, expert, kv, lora, test.
- Code diff details:
  - `vllm/model_executor/models/minimax_m2.py` modified +16/-5 (21 lines); hunks: """Inference-only MiniMaxM2 model."""; ); symbols: forward, MiniMaxM2Model, MiniMaxM2Model, __init__
  - `tests/models/registry.py` modified +6/-0 (6 lines); hunks: def check_available_online(; symbols: check_available_online
  - `vllm/config/speculative.py` modified +1/-0 (1 lines); hunks: def _verify_args(self) -> Self:; symbols: _verify_args
  - `vllm/model_executor/models/registry.py` modified +1/-0 (1 lines); hunks: "EagleMiniCPMForCausalLM": ("minicpm_eagle", "EagleMiniCPMForCausalLM"),
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/minimax_m2.py`, `tests/models/registry.py`, `vllm/config/speculative.py`; keywords observed in patches: eagle, config, flash, spec, expert, kv. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/minimax_m2.py`, `tests/models/registry.py`, `vllm/config/speculative.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


<!-- MODEL_PR_DIFF_AUDIT:END reference -->
