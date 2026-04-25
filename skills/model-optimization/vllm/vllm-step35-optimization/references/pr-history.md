# vLLM Step3.5 / Step3-VL PR History

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Scope: Step3.5-Flash and Step3-VL serving, NVFP4, tool/reasoning parser, and HF-style processor evolution.

## Landed PRs

### PR #33755 - Enable Step3p5ForCausalLM testing

- Link: https://github.com/vllm-project/vllm/pull/33755
- Why it mattered: Stabilized the core Step3.5 text runtime.
- Runtime path: vllm/vllm/model_executor/models/step3p5.py, vllm/vllm/model_executor/models/step3p5_mtp.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #34478 - Add NVFP4 quantization support for Step3.5-Flash

- Link: https://github.com/vllm-project/vllm/pull/34478
- Why it mattered: Opened the practical quantized deployment path.
- Runtime path: vllm/vllm/model_executor/models/step3p5.py, vllm/vllm/model_executor/models/step3p5_mtp.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #37579 - Refactor Step3-VL processor to HF style

- Link: https://github.com/vllm-project/vllm/pull/37579
- Why it mattered: Modernized the Step3-VL processor contract.
- Runtime path: vllm/vllm/model_executor/models/step3p5.py, vllm/vllm/model_executor/models/step3p5_mtp.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

<!-- MODEL_PR_DIFF_AUDIT:START reference -->

# VLLM Step 3.5 PR Diff Audit Reference

This reference is rebuilt from the same audited PR metadata used by `model-pr-optimization-history`. It is intentionally concise but keeps a file-level diff digest for every indexed PR.

## Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2026-02-04 | [#33755](https://github.com/vllm-project/vllm/pull/33755) | merged | [Model] Enable Step3p5ForCausalLM testing | model wrapper, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/step3p5.py`, `tests/models/registry.py`, `docs/models/supported_models.md` |
| 2026-02-13 | [#34478](https://github.com/vllm-project/vllm/pull/34478) | merged | [Model] Add NVFP4 quantization support for Step3.5-Flash | model wrapper, MoE/router, quantization, kernel, scheduler/runtime, tests/benchmarks | `tests/kernels/moe/test_nvfp4_moe.py`, `vllm/model_executor/models/step3p5.py`, `vllm/model_executor/layers/fused_moe/cutlass_moe.py` |
| 2026-03-19 | [#37579](https://github.com/vllm-project/vllm/pull/37579) | merged | [Model] Refactor Step3-VL processor to HF style | model wrapper, multimodal/processor, scheduler/runtime | `vllm/transformers_utils/processors/step3_vl.py`, `vllm/model_executor/models/step3_vl.py`, `vllm/transformers_utils/processors/internvl.py` |

## Diff Cards

### PR #33755 - [Model] Enable Step3p5ForCausalLM testing

- Link: https://github.com/vllm-project/vllm/pull/33755
- Status/date: `merged`, created 2026-02-04, merged 2026-02-07; author `jeejeelee`.
- Diff scope read: `3` files, `+28/-32`; areas: model wrapper, scheduler/runtime, tests/benchmarks, docs/config; keywords: flash, moe, config, doc, expert, lora, processor, quant, spec, test.
- Code diff details:
  - `vllm/model_executor/models/step3p5.py` modified +12/-25 (37 lines); hunks: from vllm.model_executor.layers.quantization.base_config import QuantizationConfig; def __init__(; symbols: __init__, __init__
  - `tests/models/registry.py` modified +15/-6 (21 lines); hunks: def check_available_online(; def check_available_online(; symbols: check_available_online, check_available_online
  - `docs/models/supported_models.md` modified +1/-1 (2 lines); hunks: th {
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/step3p5.py`, `tests/models/registry.py`, `docs/models/supported_models.md`; keywords observed in patches: flash, moe, config, doc, expert, lora. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/step3p5.py`, `tests/models/registry.py`, `docs/models/supported_models.md`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #34478 - [Model] Add NVFP4 quantization support for Step3.5-Flash

- Link: https://github.com/vllm-project/vllm/pull/34478
- Status/date: `merged`, created 2026-02-13, merged 2026-02-22; author `tacos8me`.
- Diff scope read: `5` files, `+204/-4`; areas: model wrapper, MoE/router, quantization, kernel, scheduler/runtime, tests/benchmarks; keywords: moe, quant, fp4, expert, config, flash, cuda, marlin, test, topk.
- Code diff details:
  - `tests/kernels/moe/test_nvfp4_moe.py` modified +126/-0 (126 lines); hunks: from vllm import _custom_ops as ops; def test_cutlass_fp4_moe_no_graph(; symbols: test_cutlass_fp4_moe_no_graph, test_cutlass_fp4_moe_swiglustep
  - `vllm/model_executor/models/step3p5.py` modified +71/-1 (72 lines); hunks: # SPDX-FileCopyrightText: Copyright contributors to the vLLM project; def __init__(; symbols: __init__, load_weights, load_weights, load_weights
  - `vllm/model_executor/layers/fused_moe/cutlass_moe.py` modified +4/-0 (4 lines); hunks: def _supports_quant_scheme(; symbols: _supports_quant_scheme, _supports_activation
  - `vllm/model_executor/layers/fused_moe/fused_marlin_moe.py` modified +3/-0 (3 lines); hunks: def _supports_quant_scheme(; symbols: _supports_quant_scheme, _supports_activation
  - `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +0/-3 (3 lines); hunks: def apply(; symbols: apply
- Optimization/support interpretation: The concrete diff surface is `tests/kernels/moe/test_nvfp4_moe.py`, `vllm/model_executor/models/step3p5.py`, `vllm/model_executor/layers/fused_moe/cutlass_moe.py`; keywords observed in patches: moe, quant, fp4, expert, config, flash. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `tests/kernels/moe/test_nvfp4_moe.py`, `vllm/model_executor/models/step3p5.py`, `vllm/model_executor/layers/fused_moe/cutlass_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #37579 - [Model] Refactor Step3-VL processor to HF style

- Link: https://github.com/vllm-project/vllm/pull/37579
- Status/date: `merged`, created 2026-03-19, merged 2026-03-20; author `DarkLight1337`.
- Diff scope read: `4` files, `+228/-160`; areas: model wrapper, multimodal/processor, scheduler/runtime; keywords: processor, vision, config, spec.
- Code diff details:
  - `vllm/transformers_utils/processors/step3_vl.py` modified +197/-127 (324 lines); hunks: from PIL import Image; def get_num_patches(self, img_width: int, img_height: int) -> tuple[int, int]:; symbols: Step3VisionProcessor:, get_num_patches, __call__, __call__
  - `vllm/model_executor/models/step3_vl.py` modified +27/-29 (56 lines); hunks: ); class Step3VLImageEmbeddingInputs(TensorSchema):; symbols: Step3VLImageEmbeddingInputs, Step3VLProcessingInfo, get_image_processor, get_hf_processor
  - `vllm/transformers_utils/processors/internvl.py` modified +4/-3 (7 lines); hunks: def __call__(; symbols: __call__
  - `vllm/transformers_utils/processors/kimi_k25.py` modified +0/-1 (1 lines); hunks: def __init__(; symbols: __init__, __call__
- Optimization/support interpretation: The concrete diff surface is `vllm/transformers_utils/processors/step3_vl.py`, `vllm/model_executor/models/step3_vl.py`, `vllm/transformers_utils/processors/internvl.py`; keywords observed in patches: processor, vision, config, spec. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `vllm/transformers_utils/processors/step3_vl.py`, `vllm/model_executor/models/step3_vl.py`, `vllm/transformers_utils/processors/internvl.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


<!-- MODEL_PR_DIFF_AUDIT:END reference -->
