# vLLM MiMo-V2-Flash PR History

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Scope: MiMo-V2-Flash inference-centric MoE runtime, MTP behavior, and the transition from older MiMo checkpoints in vLLM.

## Landed PRs

### PR #17433 - Support MiMo-7B inference with MTP

- Link: https://github.com/vllm-project/vllm/pull/17433
- Why it mattered: Historical base for the MiMo family.
- Runtime path: vllm/vllm/model_executor/models/mimo_v2_flash.py, vllm/vllm/model_executor/models/mimo.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #25136 - Fix MTP inference path for MiMo-7B model

- Link: https://github.com/vllm-project/vllm/pull/25136
- Why it mattered: Closed a concrete draft-path bug.
- Runtime path: vllm/vllm/model_executor/models/mimo_v2_flash.py, vllm/vllm/model_executor/models/mimo.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #30836 - Add MiMo-V2-Flash support

- Link: https://github.com/vllm-project/vllm/pull/30836
- Why it mattered: Landed the dedicated V2-Flash runtime.
- Runtime path: vllm/vllm/model_executor/models/mimo_v2_flash.py, vllm/vllm/model_executor/models/mimo.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

<!-- MODEL_PR_DIFF_AUDIT:START reference -->

# VLLM MiMo-V2-Flash PR Diff Audit Reference

This reference is rebuilt from the same audited PR metadata used by `model-pr-optimization-history`. It is intentionally concise but keeps a file-level diff digest for every indexed PR.

## Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2025-04-30 | [#17433](https://github.com/vllm-project/vllm/pull/17433) | merged | [Model] Support MiMo-7B inference with MTP | model wrapper, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/mimo_mtp.py`, `vllm/model_executor/models/mimo.py`, `vllm/config.py` |
| 2025-09-18 | [#25136](https://github.com/vllm-project/vllm/pull/25136) | merged | [spec decode] Fix MTP inference path for MiMo-7B model | model wrapper, scheduler/runtime, docs/config | `vllm/model_executor/models/mimo_mtp.py`, `examples/offline_inference/spec_decode.py`, `vllm/config/speculative.py` |
| 2025-12-17 | [#30836](https://github.com/vllm-project/vllm/pull/30836) | merged | [Model] Add MiMo-V2-Flash support | model wrapper, quantization, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/mimo_v2_flash.py`, `vllm/model_executor/layers/linear.py`, `vllm/model_executor/layers/quantization/utils/fp8_utils.py` |

## Diff Cards

### PR #17433 - [Model] Support MiMo-7B inference with MTP

- Link: https://github.com/vllm-project/vllm/pull/17433
- Status/date: `merged`, created 2025-04-30, merged 2025-05-12; author `bwshen-mi`.
- Diff scope read: `7` files, `+507/-4`; areas: model wrapper, scheduler/runtime, tests/benchmarks, docs/config; keywords: spec, config, eagle, cache, kv, processor, quant, attention, doc, expert.
- Code diff details:
  - `vllm/model_executor/models/mimo_mtp.py` added +283/-0 (283 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: MiMoMultiTokenPredictorLayer, __init__, forward, MiMoMultiTokenPredictor
  - `vllm/model_executor/models/mimo.py` added +190/-0 (190 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: MiMoModel, forward, load_weights, MiMoForCausalLM
  - `vllm/config.py` modified +17/-3 (20 lines); hunks: def get_num_attention_heads(self,; def hf_config_override(hf_config: PretrainedConfig) -> PretrainedConfig:; symbols: get_num_attention_heads, get_layers_start_end_indices, hf_config_override, __post_init__
  - `vllm/worker/worker.py` modified +5/-1 (6 lines); hunks: def __init__(; symbols: __init__
  - `docs/source/models/supported_models.md` modified +5/-0 (5 lines); hunks: See this page (#generative-models) for more information on how to use generativ
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/mimo_mtp.py`, `vllm/model_executor/models/mimo.py`, `vllm/config.py`; keywords observed in patches: spec, config, eagle, cache, kv, processor. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/mimo_mtp.py`, `vllm/model_executor/models/mimo.py`, `vllm/config.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #25136 - [spec decode] Fix MTP inference path for MiMo-7B model

- Link: https://github.com/vllm-project/vllm/pull/25136
- Status/date: `merged`, created 2025-09-18, merged 2025-09-18; author `zixi-qi`.
- Diff scope read: `3` files, `+20/-6`; areas: model wrapper, scheduler/runtime, docs/config; keywords: config, spec, eagle.
- Code diff details:
  - `vllm/model_executor/models/mimo_mtp.py` modified +14/-4 (18 lines); hunks: def load_weights(self, weights: Iterable[tuple[str,; symbols: load_weights, map_model_name_to_mtp_param_name, _rewrite_spec_layer_name
  - `examples/offline_inference/spec_decode.py` modified +5/-1 (6 lines); hunks: def parse_args():; def main():; symbols: parse_args, main
  - `vllm/config/speculative.py` modified +1/-1 (2 lines); hunks: SpeculativeMethod = Literal["ngram", "eagle", "eagle3", "medusa",
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/mimo_mtp.py`, `examples/offline_inference/spec_decode.py`, `vllm/config/speculative.py`; keywords observed in patches: config, spec, eagle. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/mimo_mtp.py`, `examples/offline_inference/spec_decode.py`, `vllm/config/speculative.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #30836 - [Model] Add MiMo-V2-Flash support

- Link: https://github.com/vllm-project/vllm/pull/30836
- Status/date: `merged`, created 2025-12-17, merged 2025-12-19; author `Abatom`.
- Diff scope read: `8` files, `+789/-13`; areas: model wrapper, quantization, scheduler/runtime, tests/benchmarks, docs/config; keywords: config, flash, fp8, quant, kv, attention, cache, doc, eagle, expert.
- Code diff details:
  - `vllm/model_executor/models/mimo_v2_flash.py` added +720/-0 (720 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: MiMoV2MLP, __init__, forward, MiMoV2MoE
  - `vllm/model_executor/layers/linear.py` modified +49/-13 (62 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__, __init__, _maybe_allow_fp8_block_shape_mismatch
  - `vllm/model_executor/layers/quantization/utils/fp8_utils.py` modified +8/-0 (8 lines); hunks: def validate_fp8_block_shape(; symbols: validate_fp8_block_shape
  - `vllm/config/model.py` modified +5/-0 (5 lines); hunks: def try_match_architecture_defaults(; symbols: try_match_architecture_defaults, str_dtype_to_torch_dtype
  - `tests/models/registry.py` modified +3/-0 (3 lines); hunks: def check_available_online(; symbols: check_available_online
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/mimo_v2_flash.py`, `vllm/model_executor/layers/linear.py`, `vllm/model_executor/layers/quantization/utils/fp8_utils.py`; keywords observed in patches: config, flash, fp8, quant, kv, attention. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/mimo_v2_flash.py`, `vllm/model_executor/layers/linear.py`, `vllm/model_executor/layers/quantization/utils/fp8_utils.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


<!-- MODEL_PR_DIFF_AUDIT:END reference -->
