# vllm DeepSeek V4 PR Diff Audit Reference

- Rebuilt on: 2026-04-28
- Source baseline: `vllm-project/vllm` `origin/main` commit `fd74c90d9`
- Collection: model implementation files were traced with `git log --name-only -- <model-files>`, filtered by model keywords in commit subjects, then every PR card was populated from the GitHub Pull Request files API.
- Extra preserved PRs from prior docs: 2
- Rule: use this evidence file before changing model-specific skill guidance; it is not only PR titles.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `vllm/model_executor/models/deepseek_v4.py` | [#40860](https://github.com/vllm-project/vllm/pull/40860), [#41006](https://github.com/vllm-project/vllm/pull/41006) |
| `vllm/model_executor/models/deepseek_v4_mtp.py` | [#40860](https://github.com/vllm-project/vllm/pull/40860), [#41006](https://github.com/vllm-project/vllm/pull/41006) |
| `vllm/model_executor/layers/deepseek_v4_attention.py` | [#40860](https://github.com/vllm-project/vllm/pull/40860) |
| `vllm/tokenizers/deepseek_v4.py` | [#40860](https://github.com/vllm-project/vllm/pull/40860) |
| `vllm/renderers/deepseek_v4.py` | [#40860](https://github.com/vllm-project/vllm/pull/40860) |
| `vllm/tool_parsers/deepseekv4_tool_parser.py` | [#40860](https://github.com/vllm-project/vllm/pull/40860) |
| `vllm/model_executor/models/registry.py` | [#40860](https://github.com/vllm-project/vllm/pull/40860) |

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-04-24 | [#40760](https://github.com/vllm-project/vllm/pull/40760) | closed | [New Model] Support DeepseekV4 | `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/tokenizers/deepseek_v4_encoding.py` |
| 2026-04-26 | [#40806](https://github.com/vllm-project/vllm/pull/40806) | merged | [Bugfix] Fix the DSML token leakage in DSV4/3.2 | `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py` |
| 2026-04-27 | [#40860](https://github.com/vllm-project/vllm/pull/40860) | merged | [Feat] DeepSeek V4 Rebased | `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/models/deepseek_v4_mtp.py`, `vllm/model_executor/models/registry.py` |
| 2026-04-28 | [#41006](https://github.com/vllm-project/vllm/pull/41006) | merged | [Model] Support DSV4 base | `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/models/deepseek_v4_mtp.py` |
| 2026-04-24 | [#40811](https://github.com/vllm-project/vllm/pull/40811) | open | [Perf][Kernel] BF16 input support for persistent topK - DeepSeekV4 | `csrc/persistent_topk.cuh`, `csrc/topk.cu`, `tests/kernels/test_top_k_per_row.py` |

## Per-PR Diff Audit Cards

### PR #40760 - [New Model] Support DeepseekV4

- Link: https://github.com/vllm-project/vllm/pull/40760
- Status/date: closed / 2026-04-27T01:51:00Z
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 158 files, +16968/-760; this card is historical because the branch was closed and superseded by merged PR #40860.
- Motivation: Title: "[New Model] Support DeepseekV4"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/tokenizers/deepseek_v4_encoding.py`; PR body summary: Congratulations on Deepseek-ai to release the model. Thanks for all Inferact member's effort for support this. Note: This model implementation is highly optimized. All the compo....
- Key implementation: `vllm/model_executor/models/deepseek_v4.py` added +1423/-0 (1423 lines); hunks: -0,0 +1,1423; symbols: DeepseekV4FP8Config, __init__, get_name, override_quantization_method, touching `DeepseekV4FP8Config, __init__, get_name`; `vllm/model_executor/layers/deepseek_v4_attention.py` added +1062/-0 (1062 lines); hunks: -0,0 +1,1062; symbols: DeepseekV4MLAModules, DeepseekV4MultiHeadLatentAttentionWrapper, takes, does, touching `DeepseekV4MLAModules, DeepseekV4MultiHeadLatentAttentionWrapper, takes`; `vllm/tokenizers/deepseek_v4_encoding.py` added +757/-0 (757 lines); hunks: -0,0 +1,757; symbols: to_json, tools_from_openai_format, tool_calls_from_openai_format, tool_calls_to_openai_format, touching `to_json, tools_from_openai_format, tool_calls_from_openai_format`; `vllm/model_executor/models/deepseek_v4_mtp.py` added +472/-0 (472 lines); hunks: -0,0 +1,472; symbols: DeepSeekV4MultiTokenPredictorLayer, __init__, forward, DeepSeekV4MultiTokenPredictor, touching `DeepSeekV4MultiTokenPredictorLayer, __init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v4.py` added +1423/-0 (1423 lines); hunks: -0,0 +1,1423; symbols: DeepseekV4FP8Config, __init__, get_name, override_quantization_method
  - `vllm/model_executor/layers/deepseek_v4_attention.py` added +1062/-0 (1062 lines); hunks: -0,0 +1,1062; symbols: DeepseekV4MLAModules, DeepseekV4MultiHeadLatentAttentionWrapper, takes, does
  - `vllm/tokenizers/deepseek_v4_encoding.py` added +757/-0 (757 lines); hunks: -0,0 +1,757; symbols: to_json, tools_from_openai_format, tool_calls_from_openai_format, tool_calls_to_openai_format
  - `vllm/model_executor/models/deepseek_v4_mtp.py` added +472/-0 (472 lines); hunks: -0,0 +1,472; symbols: DeepSeekV4MultiTokenPredictorLayer, __init__, forward, DeepSeekV4MultiTokenPredictor
  - `vllm/model_executor/layers/deepseek_compressor.py` added +436/-0 (436 lines); hunks: -0,0 +1,436; symbols: CompressorBackend, __init__, get_name, get_supported_kernel_block_sizes
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v4.py
@@ -0,0 +1,1423 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import typing
+from collections.abc import Callable, Iterable
+from itertools import islice
+import regex as re
diff -- vllm/model_executor/layers/deepseek_v4_attention.py
@@ -0,0 +1,1062 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""
+DeepseekV4 MLA Attention Layer
+"""
+from dataclasses import dataclass
diff -- vllm/tokenizers/deepseek_v4_encoding.py
@@ -0,0 +1,757 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v4.py` added +1423/-0; `vllm/model_executor/layers/deepseek_v4_attention.py` added +1062/-0; `vllm/tokenizers/deepseek_v4_encoding.py` added +757/-0; `vllm/model_executor/models/deepseek_v4_mtp.py` added +472/-0; `vllm/model_executor/layers/deepseek_compressor.py` added +436/-0; `vllm/model_executor/layers/mhc.py` added +436/-0
- Risk and verification: Historical predecessor only; validate current-main behavior against #40860/#41006 instead of this closed branch.

### PR #40806 - [Bugfix] Fix the DSML token leakage in DSV4/3.2

- Link: https://github.com/vllm-project/vllm/pull/40806
- Status/date: merged / 2026-04-26T00:58:50Z
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +76/-23, 144 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix the DSML token leakage in DSV4/3.2"; model line: DeepSeek V4; category: bug fix; main diff: `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py`; PR body summary: Co-authored-by: @Windswithyou before after.
- Key implementation: `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +52/-0 (52 lines); hunks: -484,6 +484,58 @@ def test_no_emission_while_incomplete(self, parser):; symbols: test_no_emission_while_incomplete, test_no_marker_leak_chunked, test_no_marker_leak_with_prefix_chunked, test_no_marker_leak_char_by_char, touching `test_no_emission_while_incomplete, test_no_marker_leak_chunked, test_no_marker_leak_with_prefix_chunked`; `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +24/-23 (47 lines); hunks: -26,6 +26,7; -54,8 +55,8 @@ def __init__(self, tokenizer: TokenizerLike, tools: list[Tool]...; symbols: __init__, extract_tool_calls, _reset_streaming_state, _extract_delta_tool_calls, touching `__init__, extract_tool_calls, _reset_streaming_state`.
- Code diff details:
  - `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +52/-0 (52 lines); hunks: -484,6 +484,58 @@ def test_no_emission_while_incomplete(self, parser):; symbols: test_no_emission_while_incomplete, test_no_marker_leak_chunked, test_no_marker_leak_with_prefix_chunked, test_no_marker_leak_char_by_char
  - `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +24/-23 (47 lines); hunks: -26,6 +26,7; -54,8 +55,8 @@ def __init__(self, tokenizer: TokenizerLike, tools: list[Tool]...; symbols: __init__, extract_tool_calls, _reset_streaming_state, _extract_delta_tool_calls
- Key code excerpts:

```diff
diff -- tests/tool_parsers/test_deepseekv32_tool_parser.py
@@ -484,6 +484,58 @@ def test_no_emission_while_incomplete(self, parser):
+    def test_no_marker_leak_chunked(self, parser):
+        """Chunked streaming must NOT leak DSML start-marker fragments
+        as content (GitHub #40801)."""
+        full_text = build_tool_call("fn", {"k": "v"})
+        deltas = self._stream_chunked(parser, full_text, chunk_size=5)
+        content = "".join(d.content for d in deltas if d.content is not None)
diff -- vllm/tool_parsers/deepseekv32_tool_parser.py
@@ -26,6 +26,7 @@
+from vllm.tool_parsers.utils import partial_tag_overlap
@@ -54,8 +55,8 @@ def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
-        self.is_tool_call_started: bool = False
+        self._sent_content_idx: int = 0
@@ -219,7 +220,7 @@ def extract_tool_calls(
-        self.is_tool_call_started = False
```

- Reviewed files:
  - tests: `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +52/-0
  - runtime: `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +24/-23
- Risk and verification: The diff ships test coverage in `tests/tool_parsers/test_deepseekv32_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #40860 - [Feat] DeepSeek V4 Rebased

- Link: https://github.com/vllm-project/vllm/pull/40860
- Status/date: merged / 2026-04-27T01:31:09Z
- Trace source: current-main implementation files and registry aliases.
- Diff scope read: GitHub Pull Request files API returned 150 files, +16313/-717; this card prioritizes model, registry, tokenizer, renderer, parser, and MTP files.
- Motivation: Title: "[Feat] DeepSeek V4 Rebased"; model line: DeepSeek V4; category: model support/runtime entry; main diff: `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/models/deepseek_v4_mtp.py`, `vllm/model_executor/models/registry.py`; supersedes the closed #40760 branch.
- Key implementation: `vllm/model_executor/models/deepseek_v4.py` adds the causal LM, FP8/MXFP4-aware quant class, MegaMoE expert path, and weight mapper; `vllm/model_executor/models/deepseek_v4_mtp.py` adds MTP support; `registry.py` registers `DeepseekV4ForCausalLM` and `DeepSeekV4MTPModel`; tokenizer, renderer, and tool parser files add DeepSeek V4 OpenAI-compatible behavior.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v4.py` added the main model, quant dispatch, MoE blocks, MLA plumbing, and `DeepseekV4ForCausalLM`.
  - `vllm/model_executor/models/deepseek_v4_mtp.py` added the MTP draft model and weight remapping.
  - `vllm/model_executor/models/registry.py` added the DeepSeek V4 model and MTP aliases.
  - `vllm/tool_parsers/deepseekv4_tool_parser.py` derives the DSML parser from the DSV3.2 parser with V4 sentinels.
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/registry.py
+    "DeepseekV4ForCausalLM": ("deepseek_v4", "DeepseekV4ForCausalLM"),
+    "DeepSeekV4MTPModel": ("deepseek_v4_mtp", "DeepSeekV4MTP"),
diff -- vllm/tool_parsers/deepseekv4_tool_parser.py
+class DeepSeekV4ToolParser(DeepSeekV32ToolParser):
+    tool_call_start_token: str = "<｜DSML｜tool_calls>"
+    tool_call_end_token: str = "</｜DSML｜tool_calls>"
diff -- vllm/model_executor/models/deepseek_v4.py
+class DeepseekV4FP8Config(Fp8Config):
+class DeepseekV4ForCausalLM(nn.Module):
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/models/deepseek_v4_mtp.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/model_executor/layers/deepseek_compressor.py`
  - parser/tokenizer: `vllm/tool_parsers/deepseekv4_tool_parser.py`, `vllm/tokenizers/deepseek_v4.py`, `vllm/renderers/deepseek_v4.py`
  - registry: `vllm/model_executor/models/registry.py`
- Risk and verification: Validate model load, base generation, MTP speculative decode, DSML tool parsing, and quantized expert routing; parser tests from #40806 should remain green.

### PR #41006 - [Model] Support DSV4 base

- Link: https://github.com/vllm-project/vllm/pull/41006
- Status/date: merged / 2026-04-28T00:16:56Z
- Trace source: current-main follow-up touching `deepseek_v4.py` and `deepseek_v4_mtp.py`.
- Diff scope read: GitHub Pull Request files API returned 2 files, +111/-23; this card covers the complete runtime diff.
- Motivation: Title: "[Model] Support DSV4 base"; model line: DeepSeek V4; category: bug fix/model support; main diff: `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/models/deepseek_v4_mtp.py`; fixes Flash-Base / FP8-expert checkpoints that should not take the MXFP4 expert path.
- Key implementation: introduces `_DEEPSEEK_V4_EXPERT_DTYPES = ("fp4", "fp8")`; resolves `expert_dtype` lazily from the active vLLM config; routes FP4 experts to `Mxfp4MoEMethod` and FP8 experts to `Fp8Config`; builds the weight mapper with different scale suffixes for FP4 vs FP8 experts; mirrors the suffix choice in the MTP loader.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v4.py` modified quant dispatch, MegaMoE validation, and per-instance weight mapper selection.
  - `vllm/model_executor/models/deepseek_v4_mtp.py` modified expert scale suffix mapping for FP4 vs FP8 experts.
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v4.py
+_DEEPSEEK_V4_EXPERT_DTYPES = ("fp4", "fp8")
+    def expert_dtype(self) -> str:
+        expert_dtype = getattr(hf_config, "expert_dtype", "fp4")
+    def is_scale_e8m0(self) -> bool:
+        return self.expert_dtype == "fp4"
+            if self.expert_dtype == "fp4":
+                return Mxfp4MoEMethod(layer.moe_config)
diff -- vllm/model_executor/models/deepseek_v4_mtp.py
+        expert_scale_suffix = (
+            ".weight_scale"
+            if getattr(self.config, "expert_dtype", "fp4") == "fp4"
+            else ".weight_scale_inv"
+        )
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/models/deepseek_v4_mtp.py`
- Risk and verification: Re-test both FP4 expert checkpoints and FP8 expert/base checkpoints; the main risk is silently applying the wrong scale suffix or MoE quant method.

### PR #40811 - [Perf][Kernel] BF16 input support for persistent topK - DeepSeekV4

- Link: https://github.com/vllm-project/vllm/pull/40811
- Status/date: open / 2026-04-24
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +886/-330, 1650 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Perf][Kernel] BF16 input support for persistent topK - DeepSeekV4"; model line: DeepSeek V4; category: performance/backend optimization; main diff: `csrc/persistent_topk.cuh`, `csrc/topk.cu`, `tests/kernels/test_top_k_per_row.py`; PR body summary: > We further quantize the index scores 𝐼:,: from FP32 to BF16 during this QAT process. This optimization achieves a 2× speedup for the top-k selector, while preserving a 99.7% r....
- Key implementation: `csrc/persistent_topk.cuh` modified +593/-218 (811 lines); hunks: -6,10 +6,12; -59,6 +61,76 @@ __device__ __forceinline__ auto convert_to_uint8(float x) ->...; `csrc/topk.cu` modified +156/-112 (268 lines); hunks: -1,5 +1,6; -10,14 +11,153; `tests/kernels/test_top_k_per_row.py` modified +137/-0 (137 lines); hunks: -383,6 +383,7 @@ def run_large_context_topk_test(; -393,6 +394,7 @@ def run_large_context_topk_test(; symbols: run_large_context_topk_test, test_persistent_topk_padded_stride, test_persistent_topk_bf16, touching `run_large_context_topk_test, test_persistent_topk_padded_stride, test_persistent_topk_bf16`.
- Code diff details:
  - `csrc/persistent_topk.cuh` modified +593/-218 (811 lines); hunks: -6,10 +6,12; -59,6 +61,76 @@ __device__ __forceinline__ auto convert_to_uint8(float x) ->...
  - `csrc/topk.cu` modified +156/-112 (268 lines); hunks: -1,5 +1,6; -10,14 +11,153
  - `tests/kernels/test_top_k_per_row.py` modified +137/-0 (137 lines); hunks: -383,6 +383,7 @@ def run_large_context_topk_test(; -393,6 +394,7 @@ def run_large_context_topk_test(; symbols: run_large_context_topk_test, test_persistent_topk_padded_stride, test_persistent_topk_bf16
- Key code excerpts:

```diff
diff -- csrc/persistent_topk.cuh
@@ -6,10 +6,12 @@
+#include <cuda_bf16.h>
+#include <type_traits>
@@ -59,6 +61,76 @@ __device__ __forceinline__ auto convert_to_uint8(float x) -> uint8_t {
+// BF16 ordered key: sign-magnitude to ordered unsigned (16-bit)
+// Uses memcpy to avoid dependency on __CUDA_NO_BFLOAT16_CONVERSIONS__
+__device__ __forceinline__ auto convert_to_uint16_bf16(__nv_bfloat16 x)
diff -- csrc/topk.cu
@@ -1,5 +1,6 @@
-// Persistent TopK kernel for DeepSeek V3 sparse attention indexer.
+// Persistent TopK kernel for DeepSeek V3/V4 sparse attention indexer.
+// Supports float32 and bfloat16 input dtypes.
@@ -10,14 +11,153 @@
+#ifndef USE_ROCM
+namespace {
diff -- tests/kernels/test_top_k_per_row.py
@@ -383,6 +383,7 @@ def run_large_context_topk_test(
```

- Reviewed files:
  - other: `csrc/persistent_topk.cuh` modified +593/-218; `csrc/topk.cu` modified +156/-112
  - tests: `tests/kernels/test_top_k_per_row.py` modified +137/-0
- Risk and verification: The diff ships test coverage in `tests/kernels/test_top_k_per_row.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.
