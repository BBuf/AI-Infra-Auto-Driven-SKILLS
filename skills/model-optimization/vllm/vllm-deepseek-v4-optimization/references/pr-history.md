# vLLM DeepSeek V4 PR History

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Support status: not landed on current mainline
- Scope: open-radar DeepSeek V4 bring-up, BF16 sparse top-k, and DSML parser
  correctness before merge

## Landed PRs

No merged DeepSeek V4 family PR is present in the checked mainline commit.
Treat every item below as open radar.

## Open PR Radar

### Open PR #40760 - Support DeepseekV4

- Link: https://github.com/vllm-project/vllm/pull/40760
- State: open
- Diff coverage: full file list reviewed via GitHub API, `156` files, `16193`
  additions, `760` deletions
- Motivation:
  - vLLM had DeepSeek V3/V3.2 paths but no DeepSeek V4 architecture alias,
    config class, tokenizer/renderer, DSML tool parser, or MTP draft model.
  - The PR is trying to land the whole bring-up stack instead of only adding a
    registry alias.
- Key implementation:
  - Adds `DeepseekV4ForCausalLM` and `DeepSeekV4MTPModel` to the registry.
  - Introduces new runtime files `deepseek_v4.py` and `deepseek_v4_mtp.py`.
  - Adds `DeepseekV4Config`, `DeepseekV4Tokenizer`, `DeepseekV4Renderer`, and a
    dedicated `DeepSeekV4ToolParser`.
  - Extends spec-decode buffers for the V4 hyper-compressed residual path and
    shares the target `lm_head` into MTP `shared_head.head`.
- Key code excerpts:

```diff
+    "DeepseekV4ForCausalLM": ("deepseek_v4", "DeepseekV4ForCausalLM"),
+    "DeepSeekV4MTPModel": ("deepseek_v4_mtp", "DeepSeekV4MTP"),
```

```diff
+class DeepSeekV4ToolParser(DeepSeekV32ToolParser):
+    tool_call_start_token: str = "<｜DSML｜tool_calls>"
+    tool_call_end_token: str = "</｜DSML｜tool_calls>"
```

```diff
+if hasattr(draft_hf_config, "compress_ratios") and hasattr(
+    draft_hf_config, "hc_mult"
+):
+    self.hidden_size = self.hidden_size * draft_hf_config.hc_mult
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v4.py`,
    `vllm/model_executor/models/deepseek_v4_mtp.py`,
    `vllm/model_executor/models/registry.py`,
    `vllm/model_executor/models/config.py`
  - tokenizer/renderer: `vllm/tokenizers/deepseek_v4.py`,
    `vllm/renderers/deepseek_v4.py`,
    `vllm/renderers/registry.py`,
    `vllm/tokenizers/registry.py`
  - tool calling: `vllm/tool_parsers/deepseekv4_tool_parser.py`,
    `vllm/tool_parsers/__init__.py`,
    `tests/tool_parsers/test_deepseekv4_tool_parser.py`
  - spec decode: `vllm/v1/spec_decode/eagle.py`,
    `vllm/v1/worker/gpu/spec_decode/eagle/speculator.py`,
    `vllm/v1/worker/gpu/spec_decode/eagle/utils.py`
  - tests: `tests/models/registry.py`, `tests/tokenizers_/test_deepseek_v4.py`
- Validation implications:
  - Do not present DeepSeek V4 as supported until the registry alias and parser
    are merged on mainline.
  - After merge, validate model load, chat templating, tool calls, and MTP
    speculative decode together because the PR spans all of those layers.

### Open PR #40811 - BF16 input support for persistent topK - DeepSeekV4

- Link: https://github.com/vllm-project/vllm/pull/40811
- State: open
- Diff coverage: full diff reviewed via GitHub API, `3` files, `886`
  additions, `330` deletions
- Motivation:
  - The persistent top-k path used by the DeepSeek sparse indexer assumed FP32
    ordering logic. BF16 inputs need ordered-key conversion and dedicated test
    coverage.
- Key implementation:
  - Adds BF16 ordered-key helpers and dtype traits to `persistent_topk.cuh`.
  - Makes `topk.cu` explicitly support both `float32` and `bfloat16`.
  - Extends `tests/kernels/test_top_k_per_row.py` with BF16 decode and long
    context cases.
- Key code excerpts:

```diff
+__device__ __forceinline__ auto convert_to_uint16_bf16(__nv_bfloat16 x)
+    -> uint16_t {
+  uint16_t bits;
+  memcpy(&bits, &x, sizeof(uint16_t));
+  return (bits & 0x8000) ? static_cast<uint16_t>(~bits)
+                         : static_cast<uint16_t>(bits | 0x8000);
+}
```

```diff
-// Persistent TopK kernel for DeepSeek V3 sparse attention indexer.
+// Persistent TopK kernel for DeepSeek V3/V4 sparse attention indexer.
+// Supports float32 and bfloat16 input dtypes.
```

- Reviewed files:
  - kernel: `csrc/persistent_topk.cuh`, `csrc/topk.cu`
  - tests: `tests/kernels/test_top_k_per_row.py`
- Validation implications:
  - Re-run the kernel suite on CUDA after merge.
  - Keep BF16 and FP32 lanes separate when debugging DeepSeek V4 indexer output,
    because this patch changes the dtype-ordering logic rather than only adding
    a new call site.

### Open PR #40806 - Fix the DSML token leakage in DSV4/3.2

- Link: https://github.com/vllm-project/vllm/pull/40806
- State: open
- Diff coverage: full diff reviewed via GitHub API, `2` files, `30` additions,
  `1` deletion
- Motivation:
  - When the `<｜DSML｜function_calls>` start token arrived split across streaming
    chunks, the parser could emit partial sentinel bytes as plain content.
  - DeepSeek V4 uses a sibling DSML grammar, so the parser had to buffer partial
    prefixes instead of forwarding them.
- Key implementation:
  - Precomputes all strict prefixes of the start token.
  - Adds `_is_potential_tool_call_prefix(...)` and returns `None` while the
    sentinel is still incomplete.
  - Adds a chunk-size-1 regression test to guarantee no token leakage.
- Key code excerpts:

```diff
+self._start_prefixes = {
+    self.tool_call_start_token[:i]
+    for i in range(1, len(self.tool_call_start_token))
+}
...
+elif self._is_potential_tool_call_prefix(current_text):
+    return None
```

```diff
+def test_no_leakage_during_chunked_streaming(self, parser):
+    full_text = build_tool_call("get_weather", {"location": "SF"})
+    deltas = self._stream_chunked(parser, full_text, chunk_size=1)
+    assert "<｜DSML｜function_calls>" not in content
```

- Reviewed files:
  - runtime: `vllm/tool_parsers/deepseekv32_tool_parser.py`
  - tests: `tests/tool_parsers/test_deepseekv32_tool_parser.py`
- Validation implications:
  - Even before a DeepSeek V4 parser lands, this is relevant because V4 tooling
    builds on the same DSML streaming behavior.
  - Streaming tool-call tests need to cover partial-sentinel chunk boundaries,
    not only full-token boundaries.

<!-- MODEL_PR_DIFF_AUDIT:START reference -->

# VLLM DeepSeek V4 PR Diff Audit Reference

This reference is rebuilt from the same audited PR metadata used by `model-pr-optimization-history`. It is intentionally concise but keeps a file-level diff digest for every indexed PR.

## Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2026-04-24 | [#40760](https://github.com/vllm-project/vllm/pull/40760) | open | [New Model] Support DeepseekV4 | model wrapper, attention/backend, MoE/router, quantization, kernel, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`, `tests/kernels/test_fused_inv_rope_fp8_quant.py` |
| 2026-04-24 | [#40806](https://github.com/vllm-project/vllm/pull/40806) | open | [Bugfix] Fix the DSML token leakage in DSV4/3.2 | tests/benchmarks | `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py` |
| 2026-04-24 | [#40811](https://github.com/vllm-project/vllm/pull/40811) | open | [Perf][Kernel] BF16 input support for persistent topK - DeepSeekV4 | MoE/router, kernel, tests/benchmarks | `csrc/persistent_topk.cuh`, `csrc/topk.cu`, `tests/kernels/test_top_k_per_row.py` |

## Diff Cards

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


<!-- MODEL_PR_DIFF_AUDIT:END reference -->
