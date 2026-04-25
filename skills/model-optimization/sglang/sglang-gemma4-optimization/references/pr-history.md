# SGLang Gemma 4 PR History

Evidence snapshot:

- SGLang mainline checked around `c122d343adb969cd9bbd1af2ca86727a11be3845`
- sgl-cookbook checked around `e88b0fd8ac5b1caa6eb42766035029220053369b`
- Scope: Gemma 4 text, MoE, multimodal, reasoning, tool use, and quantized MoE serving.

## Landed PRs

### PR #21952 - New Model: Gemma 4

- Link: https://github.com/sgl-project/sglang/pull/21952
- Why it mattered: Initial Gemma 4 support in SGLang.
- Runtime path: sglang/python/sglang/srt/models/gemma4_causal.py, sglang/python/sglang/srt/models/gemma4_mm.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #22079 - Gemma4 nvfp4 fix

- Link: https://github.com/sgl-project/sglang/pull/22079
- Why it mattered: Fixed the NVFP4 launch path.
- Runtime path: sglang/python/sglang/srt/models/gemma4_causal.py, sglang/python/sglang/srt/models/gemma4_mm.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #22408 - Adding Gemma 4 to Nightly CI

- Link: https://github.com/sgl-project/sglang/pull/22408
- Why it mattered: Added model-family regression coverage.
- Runtime path: sglang/python/sglang/srt/models/gemma4_causal.py, sglang/python/sglang/srt/models/gemma4_mm.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

<!-- MODEL_PR_DIFF_AUDIT:START reference -->

# SGLANG Gemma 4 PR Diff Audit Reference

This reference is rebuilt from the same audited PR metadata used by `model-pr-optimization-history`. It is intentionally concise but keeps a file-level diff digest for every indexed PR.

## Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2026-04-02 | [#21952](https://github.com/sgl-project/sglang/pull/21952) | merged | [New Model] Gemma 4 | model wrapper, attention/backend, MoE/router, quantization, kernel, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `python/sglang/srt/models/gemma4_causal.py`, `python/sglang/srt/models/gemma4_mm.py`, `python/sglang/srt/models/gemma4_audio.py` |
| 2026-04-03 | [#22079](https://github.com/sgl-project/sglang/pull/22079) | merged | [nvidia] Gemma4 nvfp4 fix | attention/backend, kernel | `python/sglang/srt/layers/attention/triton_ops/extend_attention.py` |
| 2026-04-09 | [#22408](https://github.com/sgl-project/sglang/pull/22408) | merged | [CI] Adding Gemma 4 to Nightly CI | tests/benchmarks | `test/registered/eval/test_vlms_mmmu_eval.py` |

## Diff Cards

### PR #21952 - [New Model] Gemma 4

- Link: https://github.com/sgl-project/sglang/pull/21952
- Status/date: `merged`, created 2026-04-02, merged 2026-04-07; author `JustinTong0323`.
- Diff scope read: `35` files, `+6007/-70`; areas: model wrapper, attention/backend, MoE/router, quantization, kernel, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config; keywords: kv, spec, attention, config, quant, cuda, moe, processor, vision, cache.
- Code diff details:
  - `python/sglang/srt/models/gemma4_causal.py` added +1009/-0 (1009 lines); hunks: +# Copyright 2025 SGLang Team; symbols: get_attention_sliding_window_size, Gemma4Router, __init__, fuse_scale
  - `python/sglang/srt/models/gemma4_mm.py` added +878/-0 (878 lines); hunks: +# Copyright 2025 SGLang Team; symbols: Gemma4ImagePixelInputs, Gemma4AudioInputs, Gemma4MultimodalEmbedder, __init__
  - `python/sglang/srt/models/gemma4_audio.py` added +873/-0 (873 lines); hunks: +# Copyright 2025 SGLang Team; symbols: Gemma4AudioRelativePositionEmbedding, __init__, _get_timing_signal_1d_pos, _relative_shift
  - `python/sglang/srt/models/gemma4_vision.py` added +599/-0 (599 lines); hunks: +# Copyright 2025 SGLang Team; symbols: _rotate_half, _apply_rotary, Gemma4VisionRotaryEmbedding, __init__
  - `python/sglang/srt/function_call/gemma4_detector.py` added +445/-0 (445 lines); hunks: +import json; symbols: _parse_gemma4_value, _parse_gemma4_array, _parse_gemma4_args, _find_matching_brace
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/gemma4_causal.py`, `python/sglang/srt/models/gemma4_mm.py`, `python/sglang/srt/models/gemma4_audio.py`; keywords observed in patches: kv, spec, attention, config, quant, cuda. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/gemma4_causal.py`, `python/sglang/srt/models/gemma4_mm.py`, `python/sglang/srt/models/gemma4_audio.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22079 - [nvidia] Gemma4 nvfp4 fix

- Link: https://github.com/sgl-project/sglang/pull/22079
- Status/date: `merged`, created 2026-04-03, merged 2026-04-10; author `wenscarl`.
- Diff scope read: `1` files, `+8/-0`; areas: attention/backend, kernel; keywords: attention, cuda, triton.
- Code diff details:
  - `python/sglang/srt/layers/attention/triton_ops/extend_attention.py` modified +8/-0 (8 lines); hunks: def _get_block_sizes_for_extend_attention(Lq: int, Lv: int):; symbols: _get_block_sizes_for_extend_attention
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/triton_ops/extend_attention.py`; keywords observed in patches: attention, cuda, triton. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/triton_ops/extend_attention.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22408 - [CI] Adding Gemma 4 to Nightly CI

- Link: https://github.com/sgl-project/sglang/pull/22408
- Status/date: `merged`, created 2026-04-09, merged 2026-04-17; author `kpham-sgl`.
- Diff scope read: `1` files, `+6/-3`; areas: tests/benchmarks; keywords: test.
- Code diff details:
  - `test/registered/eval/test_vlms_mmmu_eval.py` modified +6/-3 (9 lines); hunks: ModelLaunchSettings("Efficient-Large-Model/NVILA-Lite-2B-hf"): ModelEvalMetrics(
- Optimization/support interpretation: The concrete diff surface is `test/registered/eval/test_vlms_mmmu_eval.py`; keywords observed in patches: test. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/eval/test_vlms_mmmu_eval.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


<!-- MODEL_PR_DIFF_AUDIT:END reference -->
