# SGLang Step3.5 / Step3-VL PR History

Evidence snapshot:

- SGLang mainline checked around `c122d343adb969cd9bbd1af2ca86727a11be3845`
- sgl-cookbook checked around `e88b0fd8ac5b1caa6eb42766035029220053369b`
- Scope: Step3.5-Flash and Step3-VL-10B serving, MTP, MoE all-reduce, tool/reasoning parser, and processor evolution.

## Landed PRs

### PR #8583 - Support Step3V

- Link: https://github.com/sgl-project/sglang/pull/8583
- Why it mattered: Initial Step3 visual model support.
- Runtime path: sglang/python/sglang/srt/models/step3p5.py, sglang/python/sglang/srt/models/step3p5_mtp.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #8699 - Support DP Attention for step3_vl

- Link: https://github.com/sgl-project/sglang/pull/8699
- Why it mattered: Enabled multi-GPU VL serving.
- Runtime path: sglang/python/sglang/srt/models/step3p5.py, sglang/python/sglang/srt/models/step3p5_mtp.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #9695 - Add step3 tool parser

- Link: https://github.com/sgl-project/sglang/pull/9695
- Why it mattered: Added tool-call parsing.
- Runtime path: sglang/python/sglang/srt/models/step3p5.py, sglang/python/sglang/srt/models/step3p5_mtp.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #18564 - Implement the standard multi-layer MTP for step3p5

- Link: https://github.com/sgl-project/sglang/pull/18564
- Why it mattered: Added Step3.5 draft-model support.
- Runtime path: sglang/python/sglang/srt/models/step3p5.py, sglang/python/sglang/srt/models/step3p5_mtp.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #22773 - Optimize allreduce in MoE layers

- Link: https://github.com/sgl-project/sglang/pull/22773
- Why it mattered: Targeted the Step3.5 MoE hot path.
- Runtime path: sglang/python/sglang/srt/models/step3p5.py, sglang/python/sglang/srt/models/step3p5_mtp.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

<!-- MODEL_PR_DIFF_AUDIT:START reference -->

# SGLANG Step 3.5 PR Diff Audit Reference

This reference is rebuilt from the same audited PR metadata used by `model-pr-optimization-history`. It is intentionally concise but keeps a file-level diff digest for every indexed PR.

## Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2025-07-30 | [#8583](https://github.com/sgl-project/sglang/pull/8583) | merged | model: support Step3V | model wrapper, multimodal/processor, tests/benchmarks, docs/config | `python/sglang/srt/models/step3_vl.py`, `python/sglang/srt/multimodal/processors/step3_vl.py`, `python/sglang/srt/function_call/step3_detector.py` |
| 2025-08-02 | [#8699](https://github.com/sgl-project/sglang/pull/8699) | merged | feat: Support DP Attention for step3_vl | model wrapper, attention/backend, multimodal/processor | `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/models/step3_vl.py`, `python/sglang/srt/multimodal/processors/step3_vl.py` |
| 2025-08-27 | [#9695](https://github.com/sgl-project/sglang/pull/9695) | merged | [router] add step3 tool parser | MoE/router, tests/benchmarks | `sgl-router/src/tool_parser/parsers/step3_parser.rs`, `sgl-router/tests/tool_parser_step3.rs`, `sgl-router/src/tool_parser/registry.rs` |
| 2026-02-10 | [#18564](https://github.com/sgl-project/sglang/pull/18564) | merged | [Feature] implement the standard multi-layer MTP for step3p5 | kernel, scheduler/runtime | `python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py`, `python/sglang/srt/speculative/multi_layer_eagle_draft_extend_cuda_graph_runner.py` |
| 2026-04-14 | [#22773](https://github.com/sgl-project/sglang/pull/22773) | merged | [Step3p5] Optimize allreduce in MoE layers | model wrapper | `python/sglang/srt/models/step3p5.py` |

## Diff Cards

### PR #8583 - model: support Step3V

- Link: https://github.com/sgl-project/sglang/pull/8583
- Status/date: `merged`, created 2025-07-30, merged 2025-07-31; author `CatherineSue`.
- Diff scope read: `16` files, `+2340/-23`; areas: model wrapper, multimodal/processor, tests/benchmarks, docs/config; keywords: config, spec, vision, attention, cuda, expert, moe, processor, deepep, kv.
- Code diff details:
  - `python/sglang/srt/models/step3_vl.py` added +994/-0 (994 lines); hunks: +import logging; symbols: Step3TextMLP, __init__, forward, Step3TextMoEMLP
  - `python/sglang/srt/multimodal/processors/step3_vl.py` added +515/-0 (515 lines); hunks: +import math; symbols: GPUToTensor, forward, Step3VisionProcessor:, __init__
  - `python/sglang/srt/function_call/step3_detector.py` added +436/-0 (436 lines); hunks: +import ast; symbols: get_argument_type, parse_arguments, Step3Detector, __init__
  - `python/sglang/srt/configs/step3_vl.py` added +172/-0 (172 lines); hunks: +from typing import Any, Optional, Union; symbols: Step3VisionEncoderConfig, __init__, Step3TextConfig, __init__
  - `test/srt/test_reasoning_parser.py` modified +112/-0 (112 lines); hunks: def test_qwen3_thinking_streaming_scenario(self):; symbols: test_qwen3_thinking_streaming_scenario, TestBufferLossBugFix, test_partial_end_tag_buffer_loss_bug, test_partial_start_tag_buffer_preservation
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/step3_vl.py`, `python/sglang/srt/multimodal/processors/step3_vl.py`, `python/sglang/srt/function_call/step3_detector.py`; keywords observed in patches: config, spec, vision, attention, cuda, expert. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/step3_vl.py`, `python/sglang/srt/multimodal/processors/step3_vl.py`, `python/sglang/srt/function_call/step3_detector.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #8699 - feat: Support DP Attention for step3_vl

- Link: https://github.com/sgl-project/sglang/pull/8699
- Status/date: `merged`, created 2025-08-02, merged 2025-08-03; author `yhyang201`.
- Diff scope read: `3` files, `+25/-6`; areas: model wrapper, attention/backend, multimodal/processor; keywords: config, attention, quant, vision, cuda, kv, processor.
- Code diff details:
  - `python/sglang/srt/layers/attention/vision.py` modified +13/-5 (18 lines); hunks: import torch.nn.functional as F; def __init__(; symbols: __init__, __init__, __init__
  - `python/sglang/srt/models/step3_vl.py` modified +9/-0 (9 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__
  - `python/sglang/srt/multimodal/processors/step3_vl.py` modified +3/-1 (4 lines); hunks: from PIL import Image; def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/models/step3_vl.py`, `python/sglang/srt/multimodal/processors/step3_vl.py`; keywords observed in patches: config, attention, quant, vision, cuda, kv. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/models/step3_vl.py`, `python/sglang/srt/multimodal/processors/step3_vl.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #9695 - [router] add step3 tool parser

- Link: https://github.com/sgl-project/sglang/pull/9695
- Status/date: `merged`, created 2025-08-27, merged 2025-08-27; author `slin1237`.
- Diff scope read: `5` files, `+600/-2`; areas: MoE/router, tests/benchmarks; keywords: router, config, test, spec.
- Code diff details:
  - `sgl-router/src/tool_parser/parsers/step3_parser.rs` added +348/-0 (348 lines); hunks: +use async_trait::async_trait;; symbols: Step3Parser
  - `sgl-router/tests/tool_parser_step3.rs` added +245/-0 (245 lines); hunks: +//! Step3 Parser Integration Tests
  - `sgl-router/src/tool_parser/registry.rs` modified +3/-1 (4 lines); hunks: use crate::tool_parser::parsers::{; impl ParserRegistry {
  - `sgl-router/src/tool_parser/parsers/mod.rs` modified +3/-0 (3 lines); hunks: pub mod llama_parser;
  - `sgl-router/src/tool_parser/mod.rs` modified +1/-1 (2 lines); hunks: pub use types::{FunctionCall, PartialToolCall, StreamResult, TokenConfig, ToolCa
- Optimization/support interpretation: The concrete diff surface is `sgl-router/src/tool_parser/parsers/step3_parser.rs`, `sgl-router/tests/tool_parser_step3.rs`, `sgl-router/src/tool_parser/registry.rs`; keywords observed in patches: router, config, test, spec. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `sgl-router/src/tool_parser/parsers/step3_parser.rs`, `sgl-router/tests/tool_parser_step3.rs`, `sgl-router/src/tool_parser/registry.rs`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18564 - [Feature] implement the standard multi-layer MTP for step3p5

- Link: https://github.com/sgl-project/sglang/pull/18564
- Status/date: `merged`, created 2026-02-10, merged 2026-03-04; author `zhaziqwe`.
- Diff scope read: `2` files, `+31/-2`; areas: kernel, scheduler/runtime; keywords: eagle, spec, triton, cache, config, cuda, kv, topk.
- Code diff details:
  - `python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py` modified +21/-2 (23 lines); hunks: def __init__(; def _draft_extend_for_prefill(; symbols: __init__, _draft_extend_for_prefill, forward_batch_generation
  - `python/sglang/srt/speculative/multi_layer_eagle_draft_extend_cuda_graph_runner.py` modified +10/-0 (10 lines); hunks: def run_once():; symbols: run_once
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py`, `python/sglang/srt/speculative/multi_layer_eagle_draft_extend_cuda_graph_runner.py`; keywords observed in patches: eagle, spec, triton, cache, config, cuda. Impact reading: CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py`, `python/sglang/srt/speculative/multi_layer_eagle_draft_extend_cuda_graph_runner.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22773 - [Step3p5] Optimize allreduce in MoE layers

- Link: https://github.com/sgl-project/sglang/pull/22773
- Status/date: `merged`, created 2026-04-14, merged 2026-04-16; author `yhyang201`.
- Diff scope read: `1` files, `+59/-57`; areas: model wrapper; keywords: attention, config, cuda, expert, moe, quant.
- Code diff details:
  - `python/sglang/srt/models/step3p5.py` modified +59/-57 (116 lines); hunks: -import logging; Step3p5Config = None; symbols: __init__, __init__, __init__, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/step3p5.py`; keywords observed in patches: attention, config, cuda, expert, moe, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/step3p5.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


<!-- MODEL_PR_DIFF_AUDIT:END reference -->
