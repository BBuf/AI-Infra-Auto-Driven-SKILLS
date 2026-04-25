# SGLang MiMo-V2-Flash PR History

Evidence snapshot:

- SGLang mainline checked around `c122d343adb969cd9bbd1af2ca86727a11be3845`
- sgl-cookbook checked around `e88b0fd8ac5b1caa6eb42766035029220053369b`
- Scope: MiMo-V2-Flash inference-centric MoE runtime, flashinfer fused all-reduce, overlap, and reasoning parser behavior.

## Landed PRs

### PR #15207 - MiMo-V2-Flash day0 support

- Link: https://github.com/sgl-project/sglang/pull/15207
- Why it mattered: Initial MiMo-V2-Flash landing.
- Runtime path: sglang/python/sglang/srt/models/mimo_v2_flash.py, sglang/python/sglang/srt/models/mimo_v2_flash_nextn.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #15464 - Optimize MiMo-V2-Flash by flashinfer fused allreduce

- Link: https://github.com/sgl-project/sglang/pull/15464
- Why it mattered: Targeted decode-side communication cost.
- Runtime path: sglang/python/sglang/srt/models/mimo_v2_flash.py, sglang/python/sglang/srt/models/mimo_v2_flash_nextn.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #15488 - Respect `--swa-full-tokens-ratio`

- Link: https://github.com/sgl-project/sglang/pull/15488
- Why it mattered: Fixed a concrete runtime flag integration bug.
- Runtime path: sglang/python/sglang/srt/models/mimo_v2_flash.py, sglang/python/sglang/srt/models/mimo_v2_flash_nextn.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #17634 - Support two batch overlap

- Link: https://github.com/sgl-project/sglang/pull/17634
- Why it mattered: Added overlap / throughput optimization.
- Runtime path: sglang/python/sglang/srt/models/mimo_v2_flash.py, sglang/python/sglang/srt/models/mimo_v2_flash_nextn.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #21414 - Add mimo reasoning parser

- Link: https://github.com/sgl-project/sglang/pull/21414
- Why it mattered: Completed the parser path for thinking outputs.
- Runtime path: sglang/python/sglang/srt/models/mimo_v2_flash.py, sglang/python/sglang/srt/models/mimo_v2_flash_nextn.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

<!-- MODEL_PR_DIFF_AUDIT:START reference -->

# SGLANG MiMo-V2-Flash PR Diff Audit Reference

This reference is rebuilt from the same audited PR metadata used by `model-pr-optimization-history`. It is intentionally concise but keeps a file-level diff digest for every indexed PR.

## Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2025-12-15 | [#15207](https://github.com/sgl-project/sglang/pull/15207) | merged | [Feature] Xiaomi `MiMo-V2-Flash` day0 support | model wrapper, attention/backend, quantization, kernel, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `python/sglang/srt/speculative/mtp_worker.py`, `python/sglang/srt/models/mimo_v2_flash.py`, `python/sglang/srt/speculative/mtp_worker_v2.py` |
| 2025-12-19 | [#15464](https://github.com/sgl-project/sglang/pull/15464) | merged | Optimize MiMo-V2-Flash by flashinfer fused allreduce | model wrapper | `python/sglang/srt/models/mimo_v2_flash.py` |
| 2025-12-19 | [#15488](https://github.com/sgl-project/sglang/pull/15488) | merged | [MiMoV2Flash] fix: respect --swa-full-tokens-ratio arg | scheduler/runtime | `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/server_args.py` |
| 2026-01-23 | [#17634](https://github.com/sgl-project/sglang/pull/17634) | merged | [MiMoV2Flash] [feat]: support two batch overlap | model wrapper | `python/sglang/srt/models/mimo_v2_flash.py`, `python/sglang/srt/batch_overlap/operations_strategy.py` |
| 2026-03-25 | [#21414](https://github.com/sgl-project/sglang/pull/21414) | merged | fix(MiMo-V2-Flash): add mimo reasoning parser | misc | `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/parser/reasoning_parser.py` |

## Diff Cards

### PR #15207 - [Feature] Xiaomi `MiMo-V2-Flash` day0 support

- Link: https://github.com/sgl-project/sglang/pull/15207
- Status/date: `merged`, created 2025-12-15, merged 2025-12-19; author `acelyc111`.
- Diff scope read: `38` files, `+5396/-169`; areas: model wrapper, attention/backend, quantization, kernel, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config; keywords: spec, cache, attention, config, cuda, kv, topk, moe, processor, eagle.
- Code diff details:
  - `python/sglang/srt/speculative/mtp_worker.py` added +989/-0 (989 lines); hunks: +# Copyright 2023-2024 SGLang Team; symbols: MTPWorker, __init__, init_attention_backend, init_cuda_graphs
  - `python/sglang/srt/models/mimo_v2_flash.py` added +927/-0 (927 lines); hunks: +# Copyright 2023-2024 SGLang Team; symbols: MiMoV2MLP, __init__, forward, MoEGate
  - `python/sglang/srt/speculative/mtp_worker_v2.py` added +750/-0 (750 lines); hunks: +# Copyright 2023-2024 SGLang Team; symbols: _get_plan_stream, MTPDraftWorker, __init__, mtp_model_runner
  - `python/sglang/srt/speculative/mtp_draft_extend_cuda_graph_runner.py` added +655/-0 (655 lines); hunks: +# Copyright 2023-2024 SGLang Team; symbols: MTPDraftExtendCudaGraphRunner:, __init__, init_buffers_and_capture, can_run
  - `test/registered/function_call/test_function_call_parser.py` modified +441/-0 (441 lines); hunks: from sglang.srt.function_call.json_array_parser import JsonArrayParser; def check_single_todos(tool_result, expected):; symbols: check_single_todos, TestMiMoDetector, setUp, test_has_tool_call
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/speculative/mtp_worker.py`, `python/sglang/srt/models/mimo_v2_flash.py`, `python/sglang/srt/speculative/mtp_worker_v2.py`; keywords observed in patches: spec, cache, attention, config, cuda, kv. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/speculative/mtp_worker.py`, `python/sglang/srt/models/mimo_v2_flash.py`, `python/sglang/srt/speculative/mtp_worker_v2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #15464 - Optimize MiMo-V2-Flash by flashinfer fused allreduce

- Link: https://github.com/sgl-project/sglang/pull/15464
- Status/date: `merged`, created 2025-12-19, merged 2025-12-20; author `yuan-luo`.
- Diff scope read: `1` files, `+66/-10`; areas: model wrapper; keywords: attention, config, deepep, eagle, expert, flash, fp4, moe, processor, quant.
- Code diff details:
  - `python/sglang/srt/models/mimo_v2_flash.py` modified +66/-10 (76 lines); hunks: # ==============================================================================; RowParallelLinear,; symbols: __init__, forward, forward, forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/mimo_v2_flash.py`; keywords observed in patches: attention, config, deepep, eagle, expert, flash. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/mimo_v2_flash.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #15488 - [MiMoV2Flash] fix: respect --swa-full-tokens-ratio arg

- Link: https://github.com/sgl-project/sglang/pull/15488
- Status/date: `merged`, created 2025-12-19, merged 2025-12-25; author `acelyc111`.
- Diff scope read: `2` files, `+16/-16`; areas: scheduler/runtime; keywords: cache, flash, kv, attention, config, eagle, spec.
- Code diff details:
  - `python/sglang/srt/model_executor/model_runner.py` modified +10/-12 (22 lines); hunks: def __init__(; def profile_max_num_token(self, total_gpu_memory: int):; symbols: __init__, profile_max_num_token, handle_max_mamba_cache, set_num_token_hybrid
  - `python/sglang/srt/server_args.py` modified +6/-4 (10 lines); hunks: def _handle_model_specific_adjustments(self):; def _handle_cache_compatibility(self):; symbols: _handle_model_specific_adjustments, _handle_cache_compatibility, _handle_deterministic_inference
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/server_args.py`; keywords observed in patches: cache, flash, kv, attention, config, eagle. Impact reading: scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/server_args.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17634 - [MiMoV2Flash] [feat]: support two batch overlap

- Link: https://github.com/sgl-project/sglang/pull/17634
- Status/date: `merged`, created 2026-01-23, merged 2026-02-02; author `TZHelloWorld`.
- Diff scope read: `2` files, `+292/-8`; areas: model wrapper; keywords: config, deepep, expert, moe, attention, cache, cuda, flash, kv, processor.
- Code diff details:
  - `python/sglang/srt/models/mimo_v2_flash.py` modified +208/-8 (216 lines); hunks: import torch.nn.functional as F; kv_cache_scales_loader,; symbols: forward_deepep, op_gate, op_select_experts, op_dispatch_a
  - `python/sglang/srt/batch_overlap/operations_strategy.py` modified +84/-0 (84 lines); hunks: def init_new_tbo(; def _compute_moe_qwen3_decode(layer):; symbols: init_new_tbo, _compute_moe_qwen3_decode, _compute_moe_mimov2_layer_operations_strategy_tbo, _compute_moe_mimov2_prefill
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/mimo_v2_flash.py`, `python/sglang/srt/batch_overlap/operations_strategy.py`; keywords observed in patches: config, deepep, expert, moe, attention, cache. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/mimo_v2_flash.py`, `python/sglang/srt/batch_overlap/operations_strategy.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21414 - fix(MiMo-V2-Flash): add mimo reasoning parser

- Link: https://github.com/sgl-project/sglang/pull/21414
- Status/date: `merged`, created 2026-03-25, merged 2026-04-01; author `alphabetc1`.
- Diff scope read: `2` files, `+7/-0`; areas: misc; keywords: n/a.
- Code diff details:
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +6/-0 (6 lines); hunks: def _get_reasoning_from_request(self, request: ChatCompletionRequest) -> bool:; symbols: _get_reasoning_from_request
  - `python/sglang/srt/parser/reasoning_parser.py` modified +1/-0 (1 lines); hunks: class ReasoningParser:; symbols: ReasoningParser:
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/parser/reasoning_parser.py`; keywords observed in patches: n/a. Impact reading: the patch is in miscellaneous paths; infer the actual impact from the touched files.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/parser/reasoning_parser.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


<!-- MODEL_PR_DIFF_AUDIT:END reference -->
