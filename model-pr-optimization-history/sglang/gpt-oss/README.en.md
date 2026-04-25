# SGLang GPT-OSS Support and PR History

This note tracks the SGLang runtime, key PRs, and cookbook-facing touchpoints for GPT-OSS.

- Status: 当前 mainline 已支持

## Key Conclusions

- GPT-OSS is a flagship MoE family in SGLang.
- Quantization, expert-parallel topology, and reasoning/tool parser behavior all evolved quickly and need per-PR tracking.

## Main Runtime Surfaces

- `sglang/python/sglang/srt/models/gpt_oss.py`

## Landed PRs

- [#8843](https://github.com/sgl-project/sglang/pull/8843) `Support mxfp4 for GPT-OSS`: Added the headline quantized checkpoint path.
- [#8944](https://github.com/sgl-project/sglang/pull/8944) `Expert Parallelism for GPT-OSS`: Scaled GPT-OSS beyond pure tensor parallel.
- [#9043](https://github.com/sgl-project/sglang/pull/9043) `Implement Native GPT-OSS Tool Call Support`: Added native tool parser support instead of Harmony integration.
- [#9359](https://github.com/sgl-project/sglang/pull/9359) `Support DP attention with GPT-OSS`: Enabled larger topologies via DP attention.
- [#14920](https://github.com/sgl-project/sglang/pull/14920) `GPT-OSS Eagle v2 support`: Added speculative decoding support.
- [#18988](https://github.com/sgl-project/sglang/pull/18988) `Support fp8 online quantization for gpt-oss bf16`: Extended quantization coverage to online FP8.

## Matching Skill

- `skills/model-optimization/sglang/sglang-gpt-oss-optimization/SKILL.md`
- `skills/model-optimization/sglang/sglang-gpt-oss-optimization/references/pr-history.md`

<!-- MODEL_PR_DIFF_AUDIT:START en -->

## PR Diff Audit Cards (2026-04-25 rebuild)

This section re-audits `GPT-OSS` against `sgl-project/sglang` Pull Request metadata and file-level patches. Acceptance rule: every PR needs status, code surface, file-level diff digest, support/optimization interpretation, and verification risk notes; if no public PR is found, keep an explicit no-match conclusion instead of inventing history.

### Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2025-08-06 | [#8843](https://github.com/sgl-project/sglang/pull/8843) | merged | Support mxfp4 for GPT-OSS | model wrapper, MoE/router, quantization, kernel | `python/sglang/srt/layers/quantization/mxfp4.py`, `python/sglang/srt/layers/quantization/fp4.py`, `python/sglang/srt/models/gpt_oss.py` |
| 2025-08-08 | [#8944](https://github.com/sgl-project/sglang/pull/8944) | merged | Expert Parallelism for GPT-OSS | model wrapper, MoE/router, quantization, kernel | `python/sglang/srt/layers/quantization/mxfp4.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/models/gpt_oss.py` |
| 2025-08-11 | [#9043](https://github.com/sgl-project/sglang/pull/9043) | merged | (gpt-oss, oai, chat): Remove Harmony Integration and Implement Native GPT-OSS Tool Call Support | docs/config | `python/sglang/srt/function_call/gpt_oss_detector.py`, `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/reasoning_parser.py` |
| 2025-08-19 | [#9359](https://github.com/sgl-project/sglang/pull/9359) | merged | Support DP attention with GPT-OSS | model wrapper | `python/sglang/srt/server_args.py`, `python/sglang/srt/models/gpt_oss.py` |
| 2025-12-11 | [#14920](https://github.com/sgl-project/sglang/pull/14920) | merged | Eagle: GPT-OSS Eagle v2 support | kernel, scheduler/runtime | `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/speculative/eagle_worker.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py` |
| 2026-02-18 | [#18988](https://github.com/sgl-project/sglang/pull/18988) | merged | [GPT-OSS] support fp8 online quantization for gpt-oss bf16 | quantization | `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/server_args.py` |

### File-level PR diff reading notes

### PR #8843 - Support mxfp4 for GPT-OSS

- Link: https://github.com/sgl-project/sglang/pull/8843
- Status/date: `merged`, created 2025-08-06, merged 2025-08-06; author `Ying1123`.
- Diff scope read: `9` files, `+791/-325`; areas: model wrapper, MoE/router, quantization, kernel; keywords: config, fp4, moe, quant, triton, cuda, expert, fp8, attention, cache.
- Code diff details:
  - `python/sglang/srt/layers/quantization/mxfp4.py` added +443/-0 (443 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: _swizzle_mxfp4, _dequant_mxfp4, _dequant_mxfp4_fake, _quant_dequant_mxfp4
  - `python/sglang/srt/layers/quantization/fp4.py` modified +28/-293 (321 lines); hunks: OCP_MX_BLOCK_SIZE = 32; symbols: MxFp4Config, Mxfp4Config, __init__, __init__
  - `python/sglang/srt/models/gpt_oss.py` modified +209/-9 (218 lines); hunks: from transformers import PretrainedConfig; def __init__(; symbols: __init__, __init__, __init__, _get_default_weight_mapping
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +58/-6 (64 lines); hunks: def _load_w2(; def _load_w2(; symbols: _load_w2, _load_w2, weight_loader, _weight_loader_physical
  - `python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py` modified +25/-15 (40 lines); hunks: def triton_kernel_fused_experts(; def triton_kernel_moe_with_bias_forward(; symbols: triton_kernel_fused_experts, triton_kernel_moe_with_bias_forward, triton_kernel_moe_with_bias_forward, triton_kernel_moe_with_bias_forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/quantization/mxfp4.py`, `python/sglang/srt/layers/quantization/fp4.py`, `python/sglang/srt/models/gpt_oss.py`; keywords observed in patches: config, fp4, moe, quant, triton, cuda. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/quantization/mxfp4.py`, `python/sglang/srt/layers/quantization/fp4.py`, `python/sglang/srt/models/gpt_oss.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #8944 - Expert Parallelism for GPT-OSS

- Link: https://github.com/sgl-project/sglang/pull/8944
- Status/date: `merged`, created 2025-08-08, merged 2025-08-08; author `ch-wan`.
- Diff scope read: `8` files, `+269/-119`; areas: model wrapper, MoE/router, quantization, kernel; keywords: moe, quant, triton, cuda, expert, topk, router, cache, config, flash.
- Code diff details:
  - `python/sglang/srt/layers/quantization/mxfp4.py` modified +80/-52 (132 lines); hunks: from typing import TYPE_CHECKING, List, Optional; is_cuda,; symbols: get_quant_method, Mxfp4MoEMethod, __init__, create_weights
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +101/-12 (113 lines); hunks: def fused_moe_kernel(; def fused_moe_kernel(; symbols: fused_moe_kernel, fused_moe_kernel, fused_moe_kernel, fused_moe_kernel
  - `python/sglang/srt/models/gpt_oss.py` modified +54/-47 (101 lines); hunks: get_moe_expert_parallel_rank,; def __init__(; symbols: __init__, _load_mxfp4_experts_weights, _load_mxfp4_experts_weights, _load_mxfp4_experts_weights
  - `python/sglang/srt/server_args.py` modified +10/-4 (14 lines); hunks: is_hip,; def print_deprecated_warning(message: str):; symbols: print_deprecated_warning
  - `python/sglang/srt/layers/quantization/unquant.py` modified +9/-2 (11 lines); hunks: def apply(; def create_weights(; symbols: apply, UnquantizedFusedMoEMethod, __init__, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/quantization/mxfp4.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/models/gpt_oss.py`; keywords observed in patches: moe, quant, triton, cuda, expert, topk. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/quantization/mxfp4.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/models/gpt_oss.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #9043 - (gpt-oss, oai, chat): Remove Harmony Integration and Implement Native GPT-OSS Tool Call Support

- Link: https://github.com/sgl-project/sglang/pull/9043
- Status/date: `merged`, created 2025-08-11, merged 2025-08-12; author `CatherineSue`.
- Diff scope read: `10` files, `+717/-409`; areas: docs/config; keywords: cache, kv, spec, config, doc, lora.
- Code diff details:
  - `python/sglang/srt/function_call/gpt_oss_detector.py` added +331/-0 (331 lines); hunks: +import json; symbols: GptOssDetector, __init__, has_tool_call, detect_and_parse
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +61/-265 (326 lines); hunks: from fastapi import Request; class OpenAIServingChat(OpenAIServingBase):; symbols: OpenAIServingChat, __init__, _request_id_prefix, _validate_request
  - `python/sglang/srt/reasoning_parser.py` modified +316/-0 (316 lines); hunks: +import re; def __init__(self, stream_reasoning: bool = True, force_reasoning: bool = False); symbols: __init__, GptOssDetector, __init__, detect_and_parse
  - `python/sglang/srt/function_call/harmony_tool_parser.py` removed +0/-130 (130 lines); hunks: -# Copyright 2023-2024 SGLang Team; symbols: HarmonyToolCallParser:, extract_tool_calls_from_message, process_streaming_chunk
  - `python/sglang/srt/entrypoints/openai/protocol.py` modified +0/-9 (9 lines); hunks: class ResponseReasoningTextContent(BaseModel):; symbols: ResponseReasoningTextContent, ResponseReasoningItem
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/function_call/gpt_oss_detector.py`, `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/reasoning_parser.py`; keywords observed in patches: cache, kv, spec, config, doc, lora. Impact reading: docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/function_call/gpt_oss_detector.py`, `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/reasoning_parser.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #9359 - Support DP attention with GPT-OSS

- Link: https://github.com/sgl-project/sglang/pull/9359
- Status/date: `merged`, created 2025-08-19, merged 2025-08-20; author `nvcastet`.
- Diff scope read: `2` files, `+6/-5`; areas: model wrapper; keywords: attention, config, flash, fp4, quant, spec.
- Code diff details:
  - `python/sglang/srt/server_args.py` modified +5/-4 (9 lines); hunks: def model_specific_adjustments(self):; symbols: model_specific_adjustments
  - `python/sglang/srt/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: def _load_normal_weights(; symbols: _load_normal_weights
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/server_args.py`, `python/sglang/srt/models/gpt_oss.py`; keywords observed in patches: attention, config, flash, fp4, quant, spec. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/server_args.py`, `python/sglang/srt/models/gpt_oss.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #14920 - Eagle: GPT-OSS Eagle v2 support

- Link: https://github.com/sgl-project/sglang/pull/14920
- Status/date: `merged`, created 2025-12-11, merged 2025-12-30; author `IzzyPutterman`.
- Diff scope read: `4` files, `+48/-25`; areas: kernel, scheduler/runtime; keywords: eagle, spec, cuda, config, attention, moe, vision.
- Code diff details:
  - `python/sglang/srt/model_executor/model_runner.py` modified +30/-23 (53 lines); hunks: def __init__(; def initialize(self, min_per_gpu_memory: float):; symbols: __init__, initialize, _dummy_run
  - `python/sglang/srt/speculative/eagle_worker.py` modified +10/-0 (10 lines); hunks: def __init__(; def forward_draft_extend_after_decode(self, batch: ScheduleBatch):; symbols: __init__, forward_draft_extend_after_decode
  - `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +4/-1 (5 lines); hunks: def __init__(self, model_runner: ModelRunner):; symbols: __init__
  - `python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py` modified +4/-1 (5 lines); hunks: def __init__(self, eagle_worker: EAGLEWorker):; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/speculative/eagle_worker.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`; keywords observed in patches: eagle, spec, cuda, config, attention, moe. Impact reading: CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/speculative/eagle_worker.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18988 - [GPT-OSS] support fp8 online quantization for gpt-oss bf16

- Link: https://github.com/sgl-project/sglang/pull/18988
- Status/date: `merged`, created 2026-02-18, merged 2026-02-20; author `zminglei`.
- Diff scope read: `2` files, `+31/-1`; areas: quantization; keywords: moe, quant, triton, config, expert, fp8, spec.
- Code diff details:
  - `python/sglang/srt/layers/quantization/fp8.py` modified +26/-0 (26 lines); hunks: def __init__(self, quant_config: Fp8Config):; def create_weights(; symbols: __init__, create_weights, create_weights, apply
  - `python/sglang/srt/server_args.py` modified +5/-1 (6 lines); hunks: def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/server_args.py`; keywords observed in patches: moe, quant, triton, config, expert, fp8. Impact reading: quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/server_args.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


### Gap and optimization follow-up

- Covered PRs: 6; open PRs: 0.
- Any future PR must add both the timeline row and the file-level diff card; title-only summaries are not acceptable.

<!-- MODEL_PR_DIFF_AUDIT:END en -->
