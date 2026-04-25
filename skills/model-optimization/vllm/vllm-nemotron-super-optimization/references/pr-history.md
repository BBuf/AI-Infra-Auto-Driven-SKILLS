# vLLM Nemotron Super / Nano Hybrid PR History

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Scope: NemotronH, Nemotron 3 Super, Nemotron Nano hybrid Mamba+Attention+MoE, MTP, NVFP4, and VL adjacencies.

## Landed PRs

### PR #18863 - NemotronH support

- Link: https://github.com/vllm-project/vllm/pull/18863
- Why it mattered: Initial NemotronH landing in vLLM.
- Runtime path: vllm/vllm/model_executor/models/nemotron_h.py, vllm/vllm/model_executor/models/nemotron_h_mtp.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #25863 - Add MoE support for NemotronH

- Link: https://github.com/vllm-project/vllm/pull/25863
- Why it mattered: Extended the hybrid family to routed experts.
- Runtime path: vllm/vllm/model_executor/models/nemotron_h.py, vllm/vllm/model_executor/models/nemotron_h_mtp.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #33726 - Nemotron-H MTP and Mamba Speculative Decoding Support

- Link: https://github.com/vllm-project/vllm/pull/33726
- Why it mattered: Opened the MTP / spec-decode path.
- Runtime path: vllm/vllm/model_executor/models/nemotron_h.py, vllm/vllm/model_executor/models/nemotron_h_mtp.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #36803 - E2E Nemotron-3-Super tests

- Link: https://github.com/vllm-project/vllm/pull/36803
- Why it mattered: Added direct Super-family regression coverage.
- Runtime path: vllm/vllm/model_executor/models/nemotron_h.py, vllm/vllm/model_executor/models/nemotron_h_mtp.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #37803 - Enable NemotronHPuzzle + NemotronHMTP

- Link: https://github.com/vllm-project/vllm/pull/37803
- Why it mattered: Expanded hybrid and MTP coverage for the family.
- Runtime path: vllm/vllm/model_executor/models/nemotron_h.py, vllm/vllm/model_executor/models/nemotron_h_mtp.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

<!-- MODEL_PR_DIFF_AUDIT:START reference -->

# VLLM Nemotron Super / Nano PR Diff Audit Reference

This reference is rebuilt from the same audited PR metadata used by `model-pr-optimization-history`. It is intentionally concise but keeps a file-level diff digest for every indexed PR.

## Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2025-05-28 | [#18863](https://github.com/vllm-project/vllm/pull/18863) | merged | [Model] NemotronH support | model wrapper, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/nemotron_h.py`, `vllm/transformers_utils/configs/nemotron_h.py`, `tests/models/registry.py` |
| 2025-09-29 | [#25863](https://github.com/vllm-project/vllm/pull/25863) | merged | [Model] Add MoE support for NemotronH | model wrapper, MoE/router, quantization, scheduler/runtime, docs/config | `vllm/model_executor/models/nemotron_h.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/quantization/modelopt.py` |
| 2026-02-03 | [#33726](https://github.com/vllm-project/vllm/pull/33726) | merged | [Model][Spec Decode] Nemotron-H MTP and Mamba Speculative Decoding Support | model wrapper, attention/backend, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/nemotron_h_mtp.py`, `vllm/v1/attention/backends/mamba_attn.py`, `vllm/model_executor/layers/mamba/mamba_mixer2.py` |
| 2026-03-11 | [#36803](https://github.com/vllm-project/vllm/pull/36803) | merged | [Test] E2E Nemotron-3-Super tests | quantization, tests/benchmarks, docs/config | `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-BF16.yaml`, `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-FP8.yaml`, `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-NVFP4.yaml` |
| 2026-03-22 | [#37803](https://github.com/vllm-project/vllm/pull/37803) | merged | Enable `NemotronHPuzzle` + `NemotronHMTP` | model wrapper, scheduler/runtime, docs/config | `vllm/model_executor/models/nemotron_h_mtp.py`, `vllm/config/speculative.py` |

## Diff Cards

### PR #18863 - [Model] NemotronH support

- Link: https://github.com/vllm-project/vllm/pull/18863
- Status/date: `merged`, created 2025-05-28, merged 2025-06-05; author `vegaluisjose`.
- Diff scope read: `6` files, `+829/-0`; areas: model wrapper, scheduler/runtime, tests/benchmarks, docs/config; keywords: config, moe, spec, attention, cache, cuda, doc, fp8, kv, lora.
- Code diff details:
  - `vllm/model_executor/models/nemotron_h.py` added +565/-0 (565 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: NemotronHMLP, __init__, forward, NemotronHMLPDecoderLayer
  - `vllm/transformers_utils/configs/nemotron_h.py` added +258/-0 (258 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: NemotronHConfig, to, __init__, layers_block_type
  - `tests/models/registry.py` modified +2/-0 (2 lines); hunks: def check_available_online(; symbols: check_available_online
  - `vllm/transformers_utils/configs/__init__.py` modified +2/-0 (2 lines); hunks: from vllm.transformers_utils.configs.moonvit import MoonViTConfig; "MoonViTConfig",
  - `docs/models/supported_models.md` modified +1/-0 (1 lines); hunks: Specified using `--task generate`.
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/nemotron_h.py`, `vllm/transformers_utils/configs/nemotron_h.py`, `tests/models/registry.py`; keywords observed in patches: config, moe, spec, attention, cache, cuda. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/nemotron_h.py`, `vllm/transformers_utils/configs/nemotron_h.py`, `tests/models/registry.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #25863 - [Model] Add MoE support for NemotronH

- Link: https://github.com/vllm-project/vllm/pull/25863
- Status/date: `merged`, created 2025-09-29, merged 2025-10-23; author `tomeras91`.
- Diff scope read: `7` files, `+413/-39`; areas: model wrapper, MoE/router, quantization, scheduler/runtime, docs/config; keywords: moe, config, expert, quant, attention, cache, fp8, router, topk, cuda.
- Code diff details:
  - `vllm/model_executor/models/nemotron_h.py` modified +329/-27 (356 lines); hunks: # limitations under the License.; from vllm.model_executor.models.interfaces import (; symbols: NemotronHMLP, __init__, forward, NemotronHMoE
  - `vllm/model_executor/layers/fused_moe/layer.py` modified +30/-5 (35 lines); hunks: def create_weights(; def create_weights(; symbols: create_weights, create_weights, __init__, __init__
  - `vllm/model_executor/layers/quantization/modelopt.py` modified +26/-5 (31 lines); hunks: def __init__(; def create_weights(; symbols: __init__, create_weights, create_weights, process_weights_after_loading
  - `vllm/transformers_utils/configs/nemotron_h.py` modified +20/-0 (20 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__, layers_block_type
  - `vllm/model_executor/layers/fused_moe/fused_moe.py` modified +3/-1 (4 lines); hunks: def fused_experts(; def fused_experts_impl(; symbols: fused_experts, _get_config_quant_dtype, fused_experts_impl
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/nemotron_h.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/quantization/modelopt.py`; keywords observed in patches: moe, config, expert, quant, attention, cache. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/nemotron_h.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/quantization/modelopt.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #33726 - [Model][Spec Decode] Nemotron-H MTP and Mamba Speculative Decoding Support

- Link: https://github.com/vllm-project/vllm/pull/33726
- Status/date: `merged`, created 2026-02-03, merged 2026-02-24; author `benchislett`.
- Diff scope read: `19` files, `+800/-158`; areas: model wrapper, attention/backend, scheduler/runtime, tests/benchmarks, docs/config; keywords: cache, kv, config, spec, attention, cuda, moe, eagle, expert, processor.
- Code diff details:
  - `vllm/model_executor/models/nemotron_h_mtp.py` added +503/-0 (503 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: NemotronHMTPAttentionDecoderLayer, __init__, forward, NemotronHMTPMoEDecoderLayer
  - `vllm/v1/attention/backends/mamba_attn.py` modified +193/-85 (278 lines); hunks: # SPDX-FileCopyrightText: Copyright contributors to the vLLM project; class BaseMambaAttentionMetadata:; symbols: BaseMambaAttentionMetadata:, BaseMambaAttentionMetadata:, BaseMambaAttentionMetadataBuilder, __init__
  - `vllm/model_executor/layers/mamba/mamba_mixer2.py` modified +27/-19 (46 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__, conv_ssm_forward, conv_ssm_forward
  - `vllm/model_executor/layers/mamba/mamba_mixer.py` modified +2/-18 (20 lines); hunks: def forward_impl(self, hidden_states: torch.Tensor, output: torch.Tensor):; def forward_impl(self, hidden_states: torch.Tensor, output: torch.Tensor):; symbols: forward_impl, forward_impl, PrefillDecodeSplit, split_batch_to_prefill_and_decode
  - `vllm/config/speculative.py` modified +17/-2 (19 lines); hunks: "glm4_moe_lite_mtp",; def hf_config_override(hf_config: PretrainedConfig) -> PretrainedConfig:; symbols: hf_config_override, __post_init__, __post_init__, _verify_args
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/nemotron_h_mtp.py`, `vllm/v1/attention/backends/mamba_attn.py`, `vllm/model_executor/layers/mamba/mamba_mixer2.py`; keywords observed in patches: cache, kv, config, spec, attention, cuda. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/nemotron_h_mtp.py`, `vllm/v1/attention/backends/mamba_attn.py`, `vllm/model_executor/layers/mamba/mamba_mixer2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #36803 - [Test] E2E Nemotron-3-Super tests

- Link: https://github.com/vllm-project/vllm/pull/36803
- Status/date: `merged`, created 2026-03-11, merged 2026-03-24; author `roikoren755`.
- Diff scope read: `6` files, `+37/-0`; areas: quantization, tests/benchmarks, docs/config; keywords: config, test, expert, fp8, spec, fp4, moe.
- Code diff details:
  - `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-BF16.yaml` added +11/-0 (11 lines); hunks: +model_name: "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
  - `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-FP8.yaml` added +11/-0 (11 lines); hunks: +model_name: "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8"
  - `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-NVFP4.yaml` added +11/-0 (11 lines); hunks: +model_name: "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
  - `tests/evals/gsm8k/configs/models-h200.txt` modified +2/-0 (2 lines); hunks: DeepSeek-R1-TP.yaml
  - `.buildkite/test_areas/lm_eval.yaml` modified +1/-0 (1 lines); hunks: steps:
- Optimization/support interpretation: The concrete diff surface is `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-BF16.yaml`, `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-FP8.yaml`, `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-NVFP4.yaml`; keywords observed in patches: config, test, expert, fp8, spec, fp4. Impact reading: quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-BF16.yaml`, `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-FP8.yaml`, `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-NVFP4.yaml`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #37803 - Enable `NemotronHPuzzle` + `NemotronHMTP`

- Link: https://github.com/vllm-project/vllm/pull/37803
- Status/date: `merged`, created 2026-03-22, merged 2026-03-22; author `netanel-haber`.
- Diff scope read: `2` files, `+6/-3`; areas: model wrapper, scheduler/runtime, docs/config; keywords: config, expert, moe, spec.
- Code diff details:
  - `vllm/model_executor/models/nemotron_h_mtp.py` modified +5/-2 (7 lines); hunks: def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:; symbols: load_weights
  - `vllm/config/speculative.py` modified +1/-1 (2 lines); hunks: def hf_config_override(hf_config: PretrainedConfig) -> PretrainedConfig:; symbols: hf_config_override
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/nemotron_h_mtp.py`, `vllm/config/speculative.py`; keywords observed in patches: config, expert, moe, spec. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/nemotron_h_mtp.py`, `vllm/config/speculative.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


<!-- MODEL_PR_DIFF_AUDIT:END reference -->
