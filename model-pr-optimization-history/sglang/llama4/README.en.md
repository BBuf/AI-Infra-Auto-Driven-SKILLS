# SGLang Llama 4 Support and PR History

This note tracks the SGLang runtime, key PRs, and cookbook-facing touchpoints for Llama 4.

- Status: 当前 mainline 已支持

## Key Conclusions

- Llama4 is mature on the SGLang side but still sensitive to quantized MoE and long-context backend selection.
- The multimodal path adds a separate vision-rotary and processor validation surface.

## Main Runtime Surfaces

- `sglang/python/sglang/srt/models/llama4.py`
- `sglang/python/sglang/srt/models/mllama4.py`

## Landed PRs

- [#5092](https://github.com/sgl-project/sglang/pull/5092) `Add Llama4 support`: Initial Llama4 landing in SGLang.
- [#5194](https://github.com/sgl-project/sglang/pull/5194) `Support Llama4 fp8 inference`: Enabled the first production quantized lane.
- [#6162](https://github.com/sgl-project/sglang/pull/6162) `Fix Llama4 gibberish output with long context and CUDA graph`: Closed a major correctness bug.
- [#7129](https://github.com/sgl-project/sglang/pull/7129) `Enable ModelOpt Llama4 fp8 checkpoint deployment in SGLang`: Added the ModelOpt checkpoint path.
- [#13421](https://github.com/sgl-project/sglang/pull/13421) `Add Llama4 attention backend auto-selection`: Stabilized backend choice for real deployments.

## Matching Skill

- `skills/model-optimization/sglang/sglang-llama4-optimization/SKILL.md`
- `skills/model-optimization/sglang/sglang-llama4-optimization/references/pr-history.md`

<!-- MODEL_PR_DIFF_AUDIT:START en -->

## PR Diff Audit Cards (2026-04-25 rebuild)

This section re-audits `Llama 4` against `sgl-project/sglang` Pull Request metadata and file-level patches. Acceptance rule: every PR needs status, code surface, file-level diff digest, support/optimization interpretation, and verification risk notes; if no public PR is found, keep an explicit no-match conclusion instead of inventing history.

### Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2025-04-05 | [#5092](https://github.com/sgl-project/sglang/pull/5092) | merged | Add Llama4 support | model wrapper, attention/backend, MoE/router, quantization, kernel, multimodal/processor, scheduler/runtime, docs/config | `python/sglang/srt/models/llama4.py`, `python/sglang/srt/layers/attention/flashattention_backend.py`, `python/sglang/srt/managers/multimodal_processors/mllama4.py` |
| 2025-04-09 | [#5194](https://github.com/sgl-project/sglang/pull/5194) | merged | Support Llama4 fp8 inference | model wrapper, MoE/router, quantization, kernel, tests/benchmarks | `test/srt/test_triton_moe_channel_fp8_kernel.py`, `python/sglang/srt/layers/quantization/w8a8_fp8.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` |
| 2025-05-09 | [#6162](https://github.com/sgl-project/sglang/pull/6162) | merged | [Bugfix] Fix Llama4 gibberish output with long context and CUDA graph | attention/backend | `python/sglang/srt/layers/attention/flashattention_backend.py` |
| 2025-06-12 | [#7129](https://github.com/sgl-project/sglang/pull/7129) | merged | Enable ModelOpt Llama4 fp8 checkpoint deployment in SGLang | model wrapper, MoE/router, quantization, kernel | `python/sglang/srt/models/mllama4.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` |
| 2025-11-17 | [#13421](https://github.com/sgl-project/sglang/pull/13421) | merged | Add Llama4 attention backend auto-selection | docs/config | `python/sglang/srt/server_args.py`, `docs/basic_usage/llama4.md` |

### File-level PR diff reading notes

### PR #5092 - Add Llama4 support

- Link: https://github.com/sgl-project/sglang/pull/5092
- Status/date: `merged`, created 2025-04-05, merged 2025-04-07; author `CatherineSue`.
- Diff scope read: `27` files, `+2213/-21`; areas: model wrapper, attention/backend, MoE/router, quantization, kernel, multimodal/processor, scheduler/runtime, docs/config; keywords: config, moe, triton, attention, kv, spec, vision, expert, processor, quant.
- Code diff details:
  - `python/sglang/srt/models/llama4.py` added +420/-0 (420 lines); hunks: +# Copyright 2023-2024 SGLang Team; symbols: Llama4MoE, custom_routing_function, __init__, forward
  - `python/sglang/srt/layers/attention/flashattention_backend.py` modified +286/-9 (295 lines); hunks: from __future__ import annotations; class FlashAttentionMetadata:; symbols: FlashAttentionMetadata:, LocalAttentionMetadata:, make_local_attention_virtual_batches, cdiv
  - `python/sglang/srt/managers/multimodal_processors/mllama4.py` added +161/-0 (161 lines); hunks: +from typing import List, Mapping, Optional, Tuple, Union; symbols: Mllama4ImageProcessor, __init__, process_mm_data_async, get_patch_per_chunk
  - `python/sglang/srt/models/mllama4.py` added +154/-0 (154 lines); hunks: +# NOTE: add Aapted from vllm/mllama4.py; symbols: Llama4ForConditionalGeneration, __init__, forward, permute_qk_weight_for_rotary
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/E=128,N=512,device_name=NVIDIA_H100_80GB_HBM3.json` added +146/-0 (146 lines); hunks: +{
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/llama4.py`, `python/sglang/srt/layers/attention/flashattention_backend.py`, `python/sglang/srt/managers/multimodal_processors/mllama4.py`; keywords observed in patches: config, moe, triton, attention, kv, spec. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/llama4.py`, `python/sglang/srt/layers/attention/flashattention_backend.py`, `python/sglang/srt/managers/multimodal_processors/mllama4.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #5194 - Support Llama4 fp8 inference

- Link: https://github.com/sgl-project/sglang/pull/5194
- Status/date: `merged`, created 2025-04-09, merged 2025-04-09; author `HandH1998`.
- Diff scope read: `14` files, `+537/-106`; areas: model wrapper, MoE/router, quantization, kernel, tests/benchmarks; keywords: quant, fp8, config, expert, moe, triton, cuda, kv, topk, processor.
- Code diff details:
  - `test/srt/test_triton_moe_channel_fp8_kernel.py` added +177/-0 (177 lines); hunks: +import itertools; symbols: native_w8a8_per_token_matmul, fp8_mask, torch_w8a8_per_column_moe, TestW8A8FP8FusedMoE
  - `python/sglang/srt/layers/quantization/w8a8_fp8.py` modified +154/-4 (158 lines); hunks: -from typing import Any, Dict, List, Optional; input_to_float8,; symbols: get_config_filenames, from_config, get_quant_method, get_quant_method
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +66/-45 (111 lines); hunks: def __init__(; def create_weights(; symbols: __init__, create_weights, create_weights, process_weights_after_loading
  - `python/sglang/srt/models/mllama4.py` modified +52/-18 (70 lines); hunks: from transformers import Llama4Config; class Llama4ForConditionalGeneration(nn.Module):; symbols: Llama4ForConditionalGeneration, __init__, load_weights, load_weights
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +33/-18 (51 lines); hunks: def fused_moe_kernel(; def fused_moe_kernel(; symbols: fused_moe_kernel, fused_moe_kernel, fused_moe_kernel, invoke_fused_moe_kernel
- Optimization/support interpretation: The concrete diff surface is `test/srt/test_triton_moe_channel_fp8_kernel.py`, `python/sglang/srt/layers/quantization/w8a8_fp8.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`; keywords observed in patches: quant, fp8, config, expert, moe, triton. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/srt/test_triton_moe_channel_fp8_kernel.py`, `python/sglang/srt/layers/quantization/w8a8_fp8.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #6162 - [Bugfix] Fix Llama4 gibberish output with long context and CUDA graph

- Link: https://github.com/sgl-project/sglang/pull/6162
- Status/date: `merged`, created 2025-05-09, merged 2025-05-09; author `CatherineSue`.
- Diff scope read: `1` files, `+125/-8`; areas: attention/backend; keywords: attention, cache, cuda, eagle, flash, kv, test, topk.
- Code diff details:
  - `python/sglang/srt/layers/attention/flashattention_backend.py` modified +125/-8 (133 lines); hunks: def forward_decode(; def forward_decode(; symbols: forward_decode, forward_decode, forward_decode, init_cuda_graph_state
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/flashattention_backend.py`; keywords observed in patches: attention, cache, cuda, eagle, flash, kv. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/flashattention_backend.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #7129 - Enable ModelOpt Llama4 fp8 checkpoint deployment in SGLang

- Link: https://github.com/sgl-project/sglang/pull/7129
- Status/date: `merged`, created 2025-06-12, merged 2025-07-08; author `Edwardf0t1`.
- Diff scope read: `3` files, `+643/-81`; areas: model wrapper, MoE/router, quantization, kernel; keywords: expert, fp8, moe, quant, cache, config, fp4, kv, spec, triton.
- Code diff details:
  - `python/sglang/srt/models/mllama4.py` modified +360/-79 (439 lines); hunks: +import json as json_lib; from sglang.srt.utils import add_prefix, is_cpu; symbols: Llama4ForConditionalGeneration, __init__, _has_vision_weights, _check_vision_weights_in_index
  - `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +244/-1 (245 lines); hunks: from sglang.srt.layers.quantization.utils import (; def get_quant_method(; symbols: get_quant_method, get_quant_method, get_scaled_act_names, __init__
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +39/-1 (40 lines); hunks: def _load_w2(; def _load_w2(; symbols: _load_w2, _load_w2, weight_loader
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/mllama4.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`; keywords observed in patches: expert, fp8, moe, quant, cache, config. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/mllama4.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #13421 - Add Llama4 attention backend auto-selection

- Link: https://github.com/sgl-project/sglang/pull/13421
- Status/date: `merged`, created 2025-11-17, merged 2025-11-25; author `janbernloehr`.
- Diff scope read: `2` files, `+24/-5`; areas: docs/config; keywords: attention, spec, triton, cache, doc, flash, fp8, kv, moe, quant.
- Code diff details:
  - `python/sglang/srt/server_args.py` modified +15/-5 (20 lines); hunks: def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments
  - `docs/basic_usage/llama4.md` modified +9/-0 (9 lines); hunks: python3 -m sglang.launch_server \; symbols: llama
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/server_args.py`, `docs/basic_usage/llama4.md`; keywords observed in patches: attention, spec, triton, cache, doc, flash. Impact reading: docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/server_args.py`, `docs/basic_usage/llama4.md`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


### Gap and optimization follow-up

- Covered PRs: 5; open PRs: 0.
- Any future PR must add both the timeline row and the file-level diff card; title-only summaries are not acceptable.

<!-- MODEL_PR_DIFF_AUDIT:END en -->
