# vllm DeepSeek V4 PR Diff Audit Reference

- Rebuilt on: 2026-04-25
- Source baseline: `vllm-project/vllm` trace worktree commit `95995bbef8`
- Collection: model implementation files were traced with `git log --name-only -- <model-files>`, filtered by model keywords in commit subjects, then every PR card was populated from the GitHub Pull Request files API.
- Extra preserved PRs from prior docs: 3
- Rule: use this as the backing dossier for the skill, not only PR titles.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| - | No matching implementation file on current main |

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-04-24 | [#40760](https://github.com/vllm-project/vllm/pull/40760) | open | [New Model] Support DeepseekV4 | `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/tokenizers/deepseek_v4_encoding.py` |
| 2026-04-24 | [#40806](https://github.com/vllm-project/vllm/pull/40806) | open | [Bugfix] Fix the DSML token leakage in DSV4/3.2 | `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py` |
| 2026-04-24 | [#40811](https://github.com/vllm-project/vllm/pull/40811) | open | [Perf][Kernel] BF16 input support for persistent topK - DeepSeekV4 | `csrc/persistent_topk.cuh`, `csrc/topk.cu`, `tests/kernels/test_top_k_per_row.py` |

## Per-PR Diff Audit Cards

### PR #40760 - [New Model] Support DeepseekV4

- Link: https://github.com/vllm-project/vllm/pull/40760
- Status/date: open / 2026-04-24
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 158 files, +16954/-760, 21384 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V4, this PR adds or enables a model support/runtime surface. Title: "[New Model] Support DeepseekV4". The diff centers on `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/tokenizers/deepseek_v4_encoding.py`. PR body context: Congratulations on Deepseek-ai to release the model. Thanks for all Inferact member's effort for support this. Note: This model implementation is highly optimized. All the compo...
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
- Risk and verification: The diff ships test coverage in `tests/kernels/attention/test_use_trtllm_attention.py`, `tests/kernels/core/test_fused_q_kv_rmsnorm.py`, `tests/kernels/moe/test_deepgemm.py`, `tests/kernels/moe/test_ocp_mx_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #40806 - [Bugfix] Fix the DSML token leakage in DSV4/3.2

- Link: https://github.com/vllm-project/vllm/pull/40806
- Status/date: open / 2026-04-24
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +76/-23, 144 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V4, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Bugfix] Fix the DSML token leakage in DSV4/3.2". The diff centers on `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py`. PR body context: Co-authored-by: @Windswithyou ## Purpose ## Test Plan ## Test Result before after --- Essential Elements of an Effective PR Description Checklist - [ ] The purpose of the PR, su...
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

### PR #40811 - [Perf][Kernel] BF16 input support for persistent topK - DeepSeekV4

- Link: https://github.com/vllm-project/vllm/pull/40811
- Status/date: open / 2026-04-24
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +886/-330, 1650 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V4, this PR adds or enables a model support/runtime surface. Title: "[Perf][Kernel] BF16 input support for persistent topK - DeepSeekV4". The diff centers on `csrc/persistent_topk.cuh`, `csrc/topk.cu`, `tests/kernels/test_top_k_per_row.py`. PR body context: ## Motivation > We further quantize the index scores 𝐼:,: from FP32 to BF16 during this QAT process. This optimization achieves a 2× speedup for the top-k selector, while preser...
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
