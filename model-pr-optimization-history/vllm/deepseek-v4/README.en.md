# vLLM DeepSeek V4 Support and PR History

This note tracks the vLLM status for DeepSeek V4 at commit
`0f7be0f2f76814f80f9091220a5fbbb53912ad00`.

- Status: not supported on current mainline; only open PR evidence exists

## Key Conclusions

- Current mainline still does not register `DeepseekV4ForCausalLM` in
  `vllm/model_executor/models/registry.py`.
- The real bring-up work is concentrated in open PR `#40760`, which spans the
  model, MTP draft path, tokenizer, renderer, parser, tests, and spec-decode
  glue.
- Two additional open PRs are already relevant even before merge:
  `#40811` for BF16 persistent top-k and `#40806` for DSML streaming safety.

## Main Runtime Surfaces

- Current-main check point: `vllm/vllm/model_executor/models/registry.py`
- Open-radar files:
  `vllm/vllm/model_executor/models/deepseek_v4.py`,
  `vllm/vllm/model_executor/models/deepseek_v4_mtp.py`,
  `vllm/vllm/tokenizers/deepseek_v4.py`,
  `vllm/vllm/renderers/deepseek_v4.py`,
  `vllm/vllm/tool_parsers/deepseekv4_tool_parser.py`,
  `vllm/vllm/v1/spec_decode/eagle.py`,
  `vllm/csrc/persistent_topk.cuh`

## Open PR Radar

- [#40760](https://github.com/vllm-project/vllm/pull/40760)
  `[New Model] Support DeepseekV4`
  Diff reviewed: `156` files, `16193` additions, `760` deletions.
  The PR adds the proposed model alias, V4 MTP class, tokenizer, renderer,
  parser, config mapping, and speculative-decode wiring.
- [#40811](https://github.com/vllm-project/vllm/pull/40811)
  `[Perf][Kernel] BF16 input support for persistent topK - DeepSeekV4`
  Diff reviewed: `3` files, `886` additions, `330` deletions.
  The patch teaches the sparse top-k kernel to handle BF16 ordered keys and adds
  BF16 kernel tests.
- [#40806](https://github.com/vllm-project/vllm/pull/40806)
  `[Bugfix] Fix the DSML token leakage in DSV4/3.2`
  Diff reviewed: `2` files, `30` additions, `1` deletion.
  The parser now buffers partial DSML sentinels instead of leaking them as plain
  text during chunked streaming.

## Current Contract

Do not claim DeepSeek V4 support in vLLM until the model alias appears on
mainline and the tokenizer/parser path merges with it. If these PRs merge,
validate model load, tool calling, speculative decoding, and BF16 sparse top-k
as one connected stack.

<!-- MODEL_PR_DIFF_AUDIT:START en -->

## PR Diff Audit Cards (2026-04-25 rebuild)

This section re-audits `DeepSeek V4` against `vllm-project/vllm` Pull Request metadata and file-level patches. Acceptance rule: every PR needs status, code surface, file-level diff digest, support/optimization interpretation, and verification risk notes; if no public PR is found, keep an explicit no-match conclusion instead of inventing history.

### Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2026-04-24 | [#40760](https://github.com/vllm-project/vllm/pull/40760) | open | [New Model] Support DeepseekV4 | model wrapper, attention/backend, MoE/router, quantization, kernel, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`, `tests/kernels/test_fused_inv_rope_fp8_quant.py` |
| 2026-04-24 | [#40806](https://github.com/vllm-project/vllm/pull/40806) | open | [Bugfix] Fix the DSML token leakage in DSV4/3.2 | tests/benchmarks | `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py` |
| 2026-04-24 | [#40811](https://github.com/vllm-project/vllm/pull/40811) | open | [Perf][Kernel] BF16 input support for persistent topK - DeepSeekV4 | MoE/router, kernel, tests/benchmarks | `csrc/persistent_topk.cuh`, `csrc/topk.cu`, `tests/kernels/test_top_k_per_row.py` |

### File-level PR diff reading notes

### PR #40760 - [New Model] Support DeepseekV4

- Link: https://github.com/vllm-project/vllm/pull/40760
- Status/date: `open`, created 2026-04-24; author `zyongye`.
- Diff scope read: `158` files, `+16954/-760`; areas: model wrapper, attention/backend, MoE/router, quantization, kernel, scheduler/runtime, tests/benchmarks, docs/config; keywords: kv, attention, cache, cuda, fp8, quant, config, spec, topk, triton.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v4.py` added +1423/-0 (1423 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: DeepseekV4FP8Config, __init__, get_name, override_quantization_method
  - `vllm/model_executor/layers/deepseek_v4_attention.py` added +1062/-0 (1062 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: DeepseekV4MLAModules:, DeepseekV4MultiHeadLatentAttentionWrapper, takes, does
  - `tests/kernels/test_fused_inv_rope_fp8_quant.py` added +998/-0 (998 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: assert_dequant_close, rotate_gptj, make_cos_sin_cache, reference_inv_rope
  - `vllm/tokenizers/deepseek_v4_encoding.py` added +757/-0 (757 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: to_json, tools_from_openai_format, tool_calls_from_openai_format, tool_calls_to_openai_format
  - `csrc/moe/topk_softplus_sqrt_kernels.cu` added +715/-0 (715 lines); hunks: +/*; symbols: alignas, int, int, int
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`, `tests/kernels/test_fused_inv_rope_fp8_quant.py`; keywords observed in patches: kv, attention, cache, cuda, fp8, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`, `tests/kernels/test_fused_inv_rope_fp8_quant.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #40806 - [Bugfix] Fix the DSML token leakage in DSV4/3.2

- Link: https://github.com/vllm-project/vllm/pull/40806
- Status/date: `open`, created 2026-04-24; author `chaunceyjiang`.
- Diff scope read: `2` files, `+76/-23`; areas: tests/benchmarks; keywords: kv, test.
- Code diff details:
  - `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +52/-0 (52 lines); hunks: def test_no_emission_while_incomplete(self, parser):; symbols: test_no_emission_while_incomplete, test_no_marker_leak_chunked, test_no_marker_leak_with_prefix_chunked, test_no_marker_leak_char_by_char
  - `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +24/-23 (47 lines); hunks: Tool,; def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] \| None = None):; symbols: __init__, extract_tool_calls, _reset_streaming_state, _extract_delta_tool_calls
- Optimization/support interpretation: The concrete diff surface is `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py`; keywords observed in patches: kv, test. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #40811 - [Perf][Kernel] BF16 input support for persistent topK - DeepSeekV4

- Link: https://github.com/vllm-project/vllm/pull/40811
- Status/date: `open`, created 2026-04-24; author `LopezCastroRoberto`.
- Diff scope read: `3` files, `+886/-330`; areas: MoE/router, kernel, tests/benchmarks; keywords: cuda, topk, attention, config, flash, kv, mla, processor, spec, test.
- Code diff details:
  - `csrc/persistent_topk.cuh` modified +593/-218 (811 lines); hunks: #define PERSISTENT_TOPK_CUH_; __device__ __forceinline__ auto convert_to_uint8(float x) -> uint8_t {; symbols: TopKDTypeTraits, TopKDTypeTraits, int, int
  - `csrc/topk.cu` modified +156/-112 (268 lines); hunks: -// Persistent TopK kernel for DeepSeek V3 sparse attention indexer.; #include "persistent_topk.cuh"; symbols: int, size_t, bool, size_t
  - `tests/kernels/test_top_k_per_row.py` modified +137/-0 (137 lines); hunks: def run_large_context_topk_test(; def run_large_context_topk_test(; symbols: run_large_context_topk_test, run_large_context_topk_test, run_large_context_topk_test, test_persistent_topk_padded_stride
- Optimization/support interpretation: The concrete diff surface is `csrc/persistent_topk.cuh`, `csrc/topk.cu`, `tests/kernels/test_top_k_per_row.py`; keywords observed in patches: cuda, topk, attention, config, flash, kv. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `csrc/persistent_topk.cuh`, `csrc/topk.cu`, `tests/kernels/test_top_k_per_row.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


### Gap and optimization follow-up

- Covered PRs: 3; open PRs: 3.
- Open PRs to keep tracking: [#40760](https://github.com/vllm-project/vllm/pull/40760), [#40806](https://github.com/vllm-project/vllm/pull/40806), [#40811](https://github.com/vllm-project/vllm/pull/40811)
- Any future PR must add both the timeline row and the file-level diff card; title-only summaries are not acceptable.

<!-- MODEL_PR_DIFF_AUDIT:END en -->
