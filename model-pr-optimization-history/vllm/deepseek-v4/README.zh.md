# vllm DeepSeek V4 模型 PR 优化历史

## 文档口径

- 重做日期: 2026-04-25
- 源码基线: `vllm-project/vllm` 当前追溯 worktree commit `95995bbef8`
- PR 收集规则: 先从模型实现、配置、processor、parser、docs/tests 等相关文件执行 `git log --name-only -- <model-files>`，再按 commit subject 的模型关键词过滤，最后用 GitHub Pull Request files API 读取每个 PR 的最终 diff。
- 额外保留规则: 原 history/skill 已显式引用但未出现在当前实现文件 git trace 中的 PR 会保留，并在卡片里标注来源。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| - | 当前主线没有匹配到实现文件 |

## PR 覆盖总览

- git 追溯 PR 数: 0
- 原文档显式引用补充 PR 数: 3
- 当前文档总 PR 数: 3
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-04-24 | [#40760](https://github.com/vllm-project/vllm/pull/40760) | open | [New Model] Support DeepseekV4 | `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/tokenizers/deepseek_v4_encoding.py` |
| 2026-04-24 | [#40806](https://github.com/vllm-project/vllm/pull/40806) | open | [Bugfix] Fix the DSML token leakage in DSV4/3.2 | `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py` |
| 2026-04-24 | [#40811](https://github.com/vllm-project/vllm/pull/40811) | open | [Perf][Kernel] BF16 input support for persistent topK - DeepSeekV4 | `csrc/persistent_topk.cuh`, `csrc/topk.cu`, `tests/kernels/test_top_k_per_row.py` |

## 逐 PR diff 审计卡

### PR #40760 - [New Model] Support DeepseekV4

- 链接: https://github.com/vllm-project/vllm/pull/40760
- 状态/时间: open / 2026-04-24
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 158 个文件，+16954/-760，可读 patch 21384 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[New Model] Support DeepseekV4」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/layers/deepseek_v4_attention.py`, `vllm/tokenizers/deepseek_v4_encoding.py`；PR 正文摘要: Congratulations on Deepseek-ai to release the model. Thanks for all Inferact member's effort for support this. Note: This model implementation is highly optimized. All the compo...。
- 实现要点: `vllm/model_executor/models/deepseek_v4.py` added +1423/-0 (1423 lines); hunks: -0,0 +1,1423; symbols: DeepseekV4FP8Config, __init__, get_name, override_quantization_method，涉及 `DeepseekV4FP8Config, __init__, get_name`；`vllm/model_executor/layers/deepseek_v4_attention.py` added +1062/-0 (1062 lines); hunks: -0,0 +1,1062; symbols: DeepseekV4MLAModules, DeepseekV4MultiHeadLatentAttentionWrapper, takes, does，涉及 `DeepseekV4MLAModules, DeepseekV4MultiHeadLatentAttentionWrapper, takes`；`vllm/tokenizers/deepseek_v4_encoding.py` added +757/-0 (757 lines); hunks: -0,0 +1,757; symbols: to_json, tools_from_openai_format, tool_calls_from_openai_format, tool_calls_to_openai_format，涉及 `to_json, tools_from_openai_format, tool_calls_from_openai_format`；`vllm/model_executor/models/deepseek_v4_mtp.py` added +472/-0 (472 lines); hunks: -0,0 +1,472; symbols: DeepSeekV4MultiTokenPredictorLayer, __init__, forward, DeepSeekV4MultiTokenPredictor，涉及 `DeepSeekV4MultiTokenPredictorLayer, __init__, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v4.py` added +1423/-0 (1423 lines); hunks: -0,0 +1,1423; symbols: DeepseekV4FP8Config, __init__, get_name, override_quantization_method
  - `vllm/model_executor/layers/deepseek_v4_attention.py` added +1062/-0 (1062 lines); hunks: -0,0 +1,1062; symbols: DeepseekV4MLAModules, DeepseekV4MultiHeadLatentAttentionWrapper, takes, does
  - `vllm/tokenizers/deepseek_v4_encoding.py` added +757/-0 (757 lines); hunks: -0,0 +1,757; symbols: to_json, tools_from_openai_format, tool_calls_from_openai_format, tool_calls_to_openai_format
  - `vllm/model_executor/models/deepseek_v4_mtp.py` added +472/-0 (472 lines); hunks: -0,0 +1,472; symbols: DeepSeekV4MultiTokenPredictorLayer, __init__, forward, DeepSeekV4MultiTokenPredictor
  - `vllm/model_executor/layers/deepseek_compressor.py` added +436/-0 (436 lines); hunks: -0,0 +1,436; symbols: CompressorBackend, __init__, get_name, get_supported_kernel_block_sizes
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v4.py` added +1423/-0; `vllm/model_executor/layers/deepseek_v4_attention.py` added +1062/-0; `vllm/tokenizers/deepseek_v4_encoding.py` added +757/-0; `vllm/model_executor/models/deepseek_v4_mtp.py` added +472/-0; `vllm/model_executor/layers/deepseek_compressor.py` added +436/-0; `vllm/model_executor/layers/mhc.py` added +436/-0
- 验证与风险: diff 自带测试面 `tests/kernels/attention/test_use_trtllm_attention.py`, `tests/kernels/core/test_fused_q_kv_rmsnorm.py`, `tests/kernels/moe/test_deepgemm.py`, `tests/kernels/moe/test_ocp_mx_moe.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #40806 - [Bugfix] Fix the DSML token leakage in DSV4/3.2

- 链接: https://github.com/vllm-project/vllm/pull/40806
- 状态/时间: open / 2026-04-24
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+76/-23，可读 patch 144 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix the DSML token leakage in DSV4/3.2」；模型线: DeepSeek V4；类别: 缺陷修复；主要 diff: `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py`；PR 正文摘要: Co-authored-by: @Windswithyou before after。
- 实现要点: `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +52/-0 (52 lines); hunks: -484,6 +484,58 @@ def test_no_emission_while_incomplete(self, parser):; symbols: test_no_emission_while_incomplete, test_no_marker_leak_chunked, test_no_marker_leak_with_prefix_chunked, test_no_marker_leak_char_by_char，涉及 `test_no_emission_while_incomplete, test_no_marker_leak_chunked, test_no_marker_leak_with_prefix_chunked`；`vllm/tool_parsers/deepseekv32_tool_parser.py` modified +24/-23 (47 lines); hunks: -26,6 +26,7; -54,8 +55,8 @@ def __init__(self, tokenizer: TokenizerLike, tools: list[Tool]...; symbols: __init__, extract_tool_calls, _reset_streaming_state, _extract_delta_tool_calls，涉及 `__init__, extract_tool_calls, _reset_streaming_state`。
- 代码 diff 细节:
  - `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +52/-0 (52 lines); hunks: -484,6 +484,58 @@ def test_no_emission_while_incomplete(self, parser):; symbols: test_no_emission_while_incomplete, test_no_marker_leak_chunked, test_no_marker_leak_with_prefix_chunked, test_no_marker_leak_char_by_char
  - `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +24/-23 (47 lines); hunks: -26,6 +26,7; -54,8 +55,8 @@ def __init__(self, tokenizer: TokenizerLike, tools: list[Tool]...; symbols: __init__, extract_tool_calls, _reset_streaming_state, _extract_delta_tool_calls
- 关键代码摘录:

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

- 已读文件:
  - tests: `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +52/-0
  - runtime: `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +24/-23
- 验证与风险: diff 自带测试面 `tests/tool_parsers/test_deepseekv32_tool_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #40811 - [Perf][Kernel] BF16 input support for persistent topK - DeepSeekV4

- 链接: https://github.com/vllm-project/vllm/pull/40811
- 状态/时间: open / 2026-04-24
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+886/-330，可读 patch 1650 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Perf][Kernel] BF16 input support for persistent topK - DeepSeekV4」；模型线: DeepSeek V4；类别: 性能/后端优化；主要 diff: `csrc/persistent_topk.cuh`, `csrc/topk.cu`, `tests/kernels/test_top_k_per_row.py`；PR 正文摘要: > We further quantize the index scores 𝐼:,: from FP32 to BF16 during this QAT process. This optimization achieves a 2× speedup for the top-k selector, while preserving a 99.7% r...。
- 实现要点: `csrc/persistent_topk.cuh` modified +593/-218 (811 lines); hunks: -6,10 +6,12; -59,6 +61,76 @@ __device__ __forceinline__ auto convert_to_uint8(float x) ->...；`csrc/topk.cu` modified +156/-112 (268 lines); hunks: -1,5 +1,6; -10,14 +11,153；`tests/kernels/test_top_k_per_row.py` modified +137/-0 (137 lines); hunks: -383,6 +383,7 @@ def run_large_context_topk_test(; -393,6 +394,7 @@ def run_large_context_topk_test(; symbols: run_large_context_topk_test, test_persistent_topk_padded_stride, test_persistent_topk_bf16，涉及 `run_large_context_topk_test, test_persistent_topk_padded_stride, test_persistent_topk_bf16`。
- 代码 diff 细节:
  - `csrc/persistent_topk.cuh` modified +593/-218 (811 lines); hunks: -6,10 +6,12; -59,6 +61,76 @@ __device__ __forceinline__ auto convert_to_uint8(float x) ->...
  - `csrc/topk.cu` modified +156/-112 (268 lines); hunks: -1,5 +1,6; -10,14 +11,153
  - `tests/kernels/test_top_k_per_row.py` modified +137/-0 (137 lines); hunks: -383,6 +383,7 @@ def run_large_context_topk_test(; -393,6 +394,7 @@ def run_large_context_topk_test(; symbols: run_large_context_topk_test, test_persistent_topk_padded_stride, test_persistent_topk_bf16
- 关键代码摘录:

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

- 已读文件:
  - other: `csrc/persistent_topk.cuh` modified +593/-218; `csrc/topk.cu` modified +156/-112
  - tests: `tests/kernels/test_top_k_per_row.py` modified +137/-0
- 验证与风险: diff 自带测试面 `tests/kernels/test_top_k_per_row.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
