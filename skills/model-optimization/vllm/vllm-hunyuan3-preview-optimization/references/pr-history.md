# vLLM Hunyuan 3 Preview PR History

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Support status: partially supported or only adjacent architectures landed on current mainline
- Scope: Adjacent Hunyuan dense / OCR / VL support in vLLM relevant to Hunyuan 3 Preview planning, without a dedicated `Hunyuan3Preview` mainline alias yet.

## Landed PRs

### PR #21368 - Add Hunyuan V1 Dense Model support

- Link: https://github.com/vllm-project/vllm/pull/21368
- Why it mattered: Brought the dense Hunyuan line into vLLM mainline.
- Runtime path: vllm/vllm/model_executor/models/hunyuan_v1.py, vllm/vllm/model_executor/models/hunyuan_vision.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #29327 - Add HunyuanOCR support

- Link: https://github.com/vllm-project/vllm/pull/29327
- Why it mattered: Extended the family to OCR workloads instead of text-only generation.
- Runtime path: vllm/vllm/model_executor/models/hunyuan_v1.py, vllm/vllm/model_executor/models/hunyuan_vision.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #33035 - Eagle3 support for HunyuanVL & Hunyuan

- Link: https://github.com/vllm-project/vllm/pull/33035
- Why it mattered: Added speculative decoding support on top of the Hunyuan family.
- Runtime path: vllm/vllm/model_executor/models/hunyuan_v1.py, vllm/vllm/model_executor/models/hunyuan_vision.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

## Open PR Radar

- No specific open PR is pinned here; rerun GitHub PR search before asserting new support.

<!-- MODEL_PR_DIFF_AUDIT:START reference -->

# VLLM Hunyuan3 Preview PR Diff Audit Reference

This reference is rebuilt from the same audited PR metadata used by `model-pr-optimization-history`. It is intentionally concise but keeps a file-level diff digest for every indexed PR.

## Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2025-07-22 | [#21368](https://github.com/vllm-project/vllm/pull/21368) | merged | [Model] add Hunyuan V1 Dense Model support. | model wrapper, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/hunyuan_v1.py`, `vllm/model_executor/models/registry.py`, `tests/models/registry.py` |
| 2025-11-24 | [#29327](https://github.com/vllm-project/vllm/pull/29327) | merged | [Model] Add HunyuanOCR support | model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/hunyuan_vision.py`, `vllm/transformers_utils/processors/hunyuan_vl_image.py`, `vllm/transformers_utils/configs/hunyuan_vl.py` |
| 2026-01-25 | [#33035](https://github.com/vllm-project/vllm/pull/33035) | merged | feature: support eagle3 for HunyuanVL & Hunyuan | model wrapper, multimodal/processor, scheduler/runtime, docs/config | `vllm/model_executor/models/hunyuan_v1.py`, `vllm/v1/spec_decode/eagle.py`, `vllm/config/speculative.py` |

## Diff Cards

### PR #21368 - [Model] add Hunyuan V1 Dense Model support.

- Link: https://github.com/vllm-project/vllm/pull/21368
- Status/date: `merged`, created 2025-07-22, merged 2025-07-23; author `kzjeef`.
- Diff scope read: `4` files, `+57/-19`; areas: model wrapper, scheduler/runtime, tests/benchmarks, docs/config; keywords: moe, expert, fp8, test, attention, config, doc, kv, lora, quant.
- Code diff details:
  - `vllm/model_executor/models/hunyuan_v1.py` renamed +52/-18 (70 lines); hunks: make_layers); def __init__(; symbols: _is_moe, _get_cla_factor, __init__, __init__
  - `vllm/model_executor/models/registry.py` modified +2/-1 (3 lines); hunks: "GraniteMoeSharedForCausalLM": ("granitemoeshared", "GraniteMoeSharedForCausalLM"), # noqa: E501
  - `tests/models/registry.py` modified +2/-0 (2 lines); hunks: def check_available_online(; symbols: check_available_online
  - `docs/models/supported_models.md` modified +1/-0 (1 lines); hunks: th {
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/hunyuan_v1.py`, `vllm/model_executor/models/registry.py`, `tests/models/registry.py`; keywords observed in patches: moe, expert, fp8, test, attention, config. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/hunyuan_v1.py`, `vllm/model_executor/models/registry.py`, `tests/models/registry.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #29327 - [Model] Add HunyuanOCR support

- Link: https://github.com/vllm-project/vllm/pull/29327
- Status/date: `merged`, created 2025-11-24, merged 2025-11-25; author `Isotr0py`.
- Diff scope read: `18` files, `+2415/-4`; areas: model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config; keywords: spec, attention, vision, cache, config, processor, doc, kv, eagle, expert.
- Code diff details:
  - `vllm/model_executor/models/hunyuan_vision.py` added +1028/-0 (1028 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: HunYuanVLImagePixelInputs, HunYuanVLImageEmbeddingInputs, HunYuanVisionMLP, __init__
  - `vllm/transformers_utils/processors/hunyuan_vl_image.py` added +477/-0 (477 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: for, smart_resize, HunYuanVLImageProcessor, __init__
  - `vllm/transformers_utils/configs/hunyuan_vl.py` added +322/-0 (322 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: HunYuanVLVisionConfig, __init__, HunYuanVLTextConfig, to
  - `vllm/transformers_utils/processors/hunyuan_vl.py` added +233/-0 (233 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: HunYuanVLProcessor, __init__, __call__, batch_decode
  - `vllm/v1/worker/gpu_model_runner.py` modified +103/-1 (104 lines); hunks: from vllm.forward_context import BatchDescriptor, set_forward_context; def __init__(; symbols: __init__, __init__, _get_positions, _make_buffer
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/hunyuan_vision.py`, `vllm/transformers_utils/processors/hunyuan_vl_image.py`, `vllm/transformers_utils/configs/hunyuan_vl.py`; keywords observed in patches: spec, attention, vision, cache, config, processor. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/hunyuan_vision.py`, `vllm/transformers_utils/processors/hunyuan_vl_image.py`, `vllm/transformers_utils/configs/hunyuan_vl.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #33035 - feature: support eagle3 for HunyuanVL & Hunyuan

- Link: https://github.com/vllm-project/vllm/pull/33035
- Status/date: `merged`, created 2026-01-25, merged 2026-01-27; author `irisliu10`.
- Diff scope read: `4` files, `+49/-3`; areas: model wrapper, multimodal/processor, scheduler/runtime, docs/config; keywords: eagle, config, lora, spec, attention, cuda, expert, kv, moe, quant.
- Code diff details:
  - `vllm/model_executor/models/hunyuan_v1.py` modified +17/-2 (19 lines); hunks: from vllm.sequence import IntermediateTensors; def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):; symbols: __init__, embed_input_ids, forward, forward
  - `vllm/v1/spec_decode/eagle.py` modified +15/-0 (15 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__, __init__, _get_positions
  - `vllm/config/speculative.py` modified +8/-1 (9 lines); hunks: def _verify_args(self) -> Self:; symbols: _verify_args
  - `vllm/model_executor/models/hunyuan_vision.py` modified +9/-0 (9 lines); hunks: from .interfaces import (; class HunYuanVLForConditionalGeneration(; symbols: HunYuanVLForConditionalGeneration, embed_multimodal, set_aux_hidden_state_layers, get_eagle3_aux_hidden_state_layers
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/hunyuan_v1.py`, `vllm/v1/spec_decode/eagle.py`, `vllm/config/speculative.py`; keywords observed in patches: eagle, config, lora, spec, attention, cuda. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/hunyuan_v1.py`, `vllm/v1/spec_decode/eagle.py`, `vllm/config/speculative.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


<!-- MODEL_PR_DIFF_AUDIT:END reference -->
