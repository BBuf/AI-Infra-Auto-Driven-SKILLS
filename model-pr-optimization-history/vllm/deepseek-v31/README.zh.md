# vllm DeepSeek V3.1 模型 PR 优化历史

## 文档口径

- 重做日期: 2026-04-25
- 源码基线: `vllm-project/vllm` 当前追溯 worktree commit `95995bbef8`
- PR 收集规则: 先从模型实现、配置、processor、parser、docs/tests 等相关文件执行 `git log --name-only -- <model-files>`，再按 commit subject 的模型关键词过滤，最后用 GitHub Pull Request files API 读取每个 PR 的最终 diff。
- 额外保留规则: 原 history/skill 已显式引用但未出现在当前实现文件 git trace 中的 PR 会保留，并在卡片里标注来源。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `examples/online_serving/elastic_ep/serve_deepseek_v2.sh` | 无直接 PR 号提交 |
| `examples/tool_chat_template_deepseekv31.jinja` | [#23454](https://github.com/vllm-project/vllm/pull/23454) |
| `tests/tool_parsers/test_deepseekv31_tool_parser.py` | 无直接 PR 号提交 |
| `vllm/model_executor/models/deepseek_mtp.py` | 无直接 PR 号提交 |
| `vllm/model_executor/models/deepseek_v2.py` | 无直接 PR 号提交 |
| `vllm/tool_parsers/deepseekv31_tool_parser.py` | 无直接 PR 号提交 |

## PR 覆盖总览

- git 追溯 PR 数: 1
- 原文档显式引用补充 PR 数: 4
- 当前文档总 PR 数: 5
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-08-23 | [#23454](https://github.com/vllm-project/vllm/pull/23454) | merged | Support DeepSeek-V3.1 tool call | `examples/tool_chat_template_deepseekv31.jinja` |
| 2025-08-27 | [#23666](https://github.com/vllm-project/vllm/pull/23666) | merged | [Feature] Add Hopper DeepGEMM E8M0 for DeepSeekV3.1 scale_fmt | `vllm/model_executor/layers/quantization/fp8.py`, `vllm/model_executor/layers/fused_moe/fused_moe.py`, `vllm/model_executor/layers/fused_moe/triton_deep_gemm_moe.py` |
| 2025-10-15 | [#25589](https://github.com/vllm-project/vllm/pull/25589) | merged | [Model] Add DeepSeek-V3.1 reasoning parser (split from PR #24972) | `tests/reasoning/test_deepseekv3_reasoning_parser.py`, `vllm/reasoning/deepseek_v3_reasoning_parser.py`, `vllm/reasoning/identity_reasoning_parser.py` |
| 2026-01-13 | [#29867](https://github.com/vllm-project/vllm/pull/29867) | merged | [Quantization] fix: overflow with static per-tensor scaling | `vllm/model_executor/layers/quantization/utils/quant_utils.py`, `vllm/v1/attention/backends/mla/common.py` |
| 2026-01-15 | [#32361](https://github.com/vllm-project/vllm/pull/32361) | merged | [BugFix] Fix DeepSeek-V3.1 + DeepGEMM incompatible scale shapes | `vllm/model_executor/layers/quantization/utils/quant_utils.py` |

## 逐 PR diff 审计卡

### PR #23454 - Support DeepSeek-V3.1 tool call

- 链接: https://github.com/vllm-project/vllm/pull/23454
- 状态/时间: merged / 2025-08-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `examples/tool_chat_template_deepseekv31.jinja`；关联提交 `b8f17f5d980e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+468/-0，可读 patch 491 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support DeepSeek-V3.1 tool call」；模型线: DeepSeek V3.1；类别: 模型支持/运行时入口；主要 diff: `examples/tool_chat_template_deepseekv31.jinja`；PR 正文摘要: Support DeepSeek-V3.1 tool call. The tool call format of DeepSeek-V3.1 is different from DeepSeek-V3/R1: DeepSeek-V3.1: tool_call_name tool_call_arguments DeepSeek-R1/V3: functi...。
- 实现要点: `examples/tool_chat_template_deepseekv31.jinja` added +91/-0 (91 lines); hunks: -0,0 +1,91。
- 代码 diff 细节:
  - `examples/tool_chat_template_deepseekv31.jinja` added +91/-0 (91 lines); hunks: -0,0 +1,91
- 关键代码摘录:

```diff
diff -- examples/tool_chat_template_deepseekv31.jinja
@@ -0,0 +1,91 @@
+{% if not add_generation_prompt is defined %}
+  {% set add_generation_prompt = false %}
+{% endif %}
+{% if not thinking is defined %}
+  {% set thinking = false %}
+{% endif %}
```

- 已读文件:
  - docs: `examples/tool_chat_template_deepseekv31.jinja` added +91/-0
- 验证与风险: runtime 路径改动集中在 `vllm/entrypoints/openai/tool_parsers/__init__.py`, `vllm/entrypoints/openai/tool_parsers/deepseekv31_tool_parser.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23666 - [Feature] Add Hopper DeepGEMM E8M0 for DeepSeekV3.1 scale_fmt

- 链接: https://github.com/vllm-project/vllm/pull/23666
- 状态/时间: merged / 2025-08-27
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+68/-53，可读 patch 322 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Feature] Add Hopper DeepGEMM E8M0 for DeepSeekV3.1 scale_fmt」；模型线: DeepSeek V3.1；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/quantization/fp8.py`, `vllm/model_executor/layers/fused_moe/fused_moe.py`, `vllm/model_executor/layers/fused_moe/triton_deep_gemm_moe.py`；PR 正文摘要: Recently DeepGEMM has supported E8M0 scale on Hopper, and this is also required by DeepSeekV3.1 This PR adds the support for it Test Unit Test Accuracy `VLLM_USE_DEEP_GEMM=1 lm_...。
- 实现要点: `vllm/model_executor/layers/quantization/fp8.py` modified +4/-5 (9 lines); hunks: -48,8 +48,7; -427,7 +426,7 @@ def process_weights_after_loading(self, layer: Module) -> None:; symbols: process_weights_after_loading，涉及 `process_weights_after_loading`；`vllm/model_executor/layers/fused_moe/fused_moe.py` modified +3/-4 (7 lines); hunks: -40,7 +40,7; -1431,9 +1431,8 @@ def fused_experts(hidden_states: torch.Tensor,; symbols: fused_experts，涉及 `fused_experts`；`vllm/model_executor/layers/fused_moe/triton_deep_gemm_moe.py` modified +3/-3 (6 lines); hunks: -10,7 +10,7; -107,7 +107,7 @@ def workspace_shapes(; symbols: TritonOrDeepGemmExperts, workspace_shapes, apply，涉及 `TritonOrDeepGemmExperts, workspace_shapes, apply`；`vllm/model_executor/layers/fused_moe/batched_deep_gemm_moe.py` modified +2/-2 (4 lines); hunks: -12,7 +12,7; -174,7 +174,7 @@ def silu_mul_fp8_quant_deep_gemm(; symbols: silu_mul_fp8_quant_deep_gemm，涉及 `silu_mul_fp8_quant_deep_gemm`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/fp8.py` modified +4/-5 (9 lines); hunks: -48,8 +48,7; -427,7 +426,7 @@ def process_weights_after_loading(self, layer: Module) -> None:; symbols: process_weights_after_loading
  - `vllm/model_executor/layers/fused_moe/fused_moe.py` modified +3/-4 (7 lines); hunks: -40,7 +40,7; -1431,9 +1431,8 @@ def fused_experts(hidden_states: torch.Tensor,; symbols: fused_experts
  - `vllm/model_executor/layers/fused_moe/triton_deep_gemm_moe.py` modified +3/-3 (6 lines); hunks: -10,7 +10,7; -107,7 +107,7 @@ def workspace_shapes(; symbols: TritonOrDeepGemmExperts, workspace_shapes, apply
  - `vllm/model_executor/layers/fused_moe/batched_deep_gemm_moe.py` modified +2/-2 (4 lines); hunks: -12,7 +12,7; -174,7 +174,7 @@ def silu_mul_fp8_quant_deep_gemm(; symbols: silu_mul_fp8_quant_deep_gemm
  - `vllm/model_executor/layers/quantization/utils/fp8_utils.py` modified +2/-2 (4 lines); hunks: -20,7 +20,7; -385,7 +385,7 @@ def per_token_group_quant_fp8(; symbols: per_token_group_quant_fp8
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/fp8.py
@@ -48,8 +48,7 @@
-from vllm.utils.deep_gemm import (is_blackwell_deep_gemm_e8m0_used,
-                                  is_deep_gemm_supported)
+from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used, is_deep_gemm_supported
@@ -427,7 +426,7 @@ def process_weights_after_loading(self, layer: Module) -> None:
-        if is_blackwell_deep_gemm_e8m0_used():
+        if is_deep_gemm_e8m0_used():
diff -- vllm/model_executor/layers/fused_moe/fused_moe.py
@@ -40,7 +40,7 @@
-from vllm.utils.deep_gemm import is_blackwell_deep_gemm_e8m0_used
+from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used
@@ -1431,9 +1431,8 @@ def fused_experts(hidden_states: torch.Tensor,
-    if (allow_deep_gemm and use_fp8_w8a8
-            and (is_blackwell_deep_gemm_e8m0_used()
-                 or _valid_deep_gemm(hidden_states, w1, w2))):
diff -- vllm/model_executor/layers/fused_moe/triton_deep_gemm_moe.py
@@ -10,7 +10,7 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/fp8.py` modified +4/-5; `vllm/model_executor/layers/fused_moe/fused_moe.py` modified +3/-4; `vllm/model_executor/layers/fused_moe/triton_deep_gemm_moe.py` modified +3/-3; `vllm/model_executor/layers/fused_moe/batched_deep_gemm_moe.py` modified +2/-2; `vllm/model_executor/layers/quantization/utils/fp8_utils.py` modified +2/-2; `vllm/utils/deep_gemm.py` modified +24/-29
- 验证与风险: diff 自带测试面 `tests/kernels/moe/test_block_fp8.py`, `tests/kernels/moe/test_deepep_deepgemm_moe.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #25589 - [Model] Add DeepSeek-V3.1 reasoning parser (split from PR #24972)

- 链接: https://github.com/vllm-project/vllm/pull/25589
- 状态/时间: merged / 2025-10-15
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+215/-3，可读 patch 269 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Add DeepSeek-V3.1 reasoning parser (split from PR #24972)」；模型线: DeepSeek V3.1；类别: 模型支持/运行时入口；主要 diff: `tests/reasoning/test_deepseekv3_reasoning_parser.py`, `vllm/reasoning/deepseek_v3_reasoning_parser.py`, `vllm/reasoning/identity_reasoning_parser.py`；PR 正文摘要: This PR adds a new reasoning parser for the DeepSeek-V3.1 model, named deepseek_v3. Unlike previous models such as deepseek_r1, the reasoning parser for DeepSeek-V3.1 is determi...。
- 实现要点: `tests/reasoning/test_deepseekv3_reasoning_parser.py` added +76/-0 (76 lines); hunks: -0,0 +1,76; symbols: tokenizer, test_parser_selection, test_identity_reasoning_parser_basic，涉及 `tokenizer, test_parser_selection, test_identity_reasoning_parser_basic`；`vllm/reasoning/deepseek_v3_reasoning_parser.py` added +66/-0 (66 lines); hunks: -0,0 +1,66; symbols: DeepSeekV3ReasoningParser, __init__, is_reasoning_end, extract_content_ids，涉及 `DeepSeekV3ReasoningParser, __init__, is_reasoning_end`；`vllm/reasoning/identity_reasoning_parser.py` added +58/-0 (58 lines); hunks: -0,0 +1,58; symbols: IdentityReasoningParser, __init__, is_reasoning_end, extract_content_ids，涉及 `IdentityReasoningParser, __init__, is_reasoning_end`；`vllm/entrypoints/openai/serving_chat.py` modified +8/-2 (10 lines); hunks: -573,7 +573,10 @@ async def chat_completion_stream_generator(; -1342,7 +1345,10 @@ async def chat_completion_full_generator(; symbols: chat_completion_stream_generator, chat_completion_full_generator，涉及 `chat_completion_stream_generator, chat_completion_full_generator`。
- 代码 diff 细节:
  - `tests/reasoning/test_deepseekv3_reasoning_parser.py` added +76/-0 (76 lines); hunks: -0,0 +1,76; symbols: tokenizer, test_parser_selection, test_identity_reasoning_parser_basic
  - `vllm/reasoning/deepseek_v3_reasoning_parser.py` added +66/-0 (66 lines); hunks: -0,0 +1,66; symbols: DeepSeekV3ReasoningParser, __init__, is_reasoning_end, extract_content_ids
  - `vllm/reasoning/identity_reasoning_parser.py` added +58/-0 (58 lines); hunks: -0,0 +1,58; symbols: IdentityReasoningParser, __init__, is_reasoning_end, extract_content_ids
  - `vllm/entrypoints/openai/serving_chat.py` modified +8/-2 (10 lines); hunks: -573,7 +573,10 @@ async def chat_completion_stream_generator(; -1342,7 +1345,10 @@ async def chat_completion_full_generator(; symbols: chat_completion_stream_generator, chat_completion_full_generator
  - `docs/features/reasoning_outputs.md` modified +3/-1 (4 lines); hunks: -11,6 +11,7 @@ vLLM currently supports the following reasoning models:; -20,8 +21,9 @@ vLLM currently supports the following reasoning models:
- 关键代码摘录:

```diff
diff -- tests/reasoning/test_deepseekv3_reasoning_parser.py
@@ -0,0 +1,76 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import pytest
+from transformers import AutoTokenizer
+from vllm.entrypoints.openai.protocol import ChatCompletionRequest, DeltaMessage
+from vllm.reasoning import (
diff -- vllm/reasoning/deepseek_v3_reasoning_parser.py
@@ -0,0 +1,66 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from collections.abc import Sequence
+from transformers import PreTrainedTokenizerBase
+from vllm.entrypoints.openai.protocol import ChatCompletionRequest, DeltaMessage
+from vllm.logger import init_logger
diff -- vllm/reasoning/identity_reasoning_parser.py
@@ -0,0 +1,58 @@
```

- 已读文件:
  - tests: `tests/reasoning/test_deepseekv3_reasoning_parser.py` added +76/-0
  - runtime: `vllm/reasoning/deepseek_v3_reasoning_parser.py` added +66/-0; `vllm/reasoning/identity_reasoning_parser.py` added +58/-0; `vllm/entrypoints/openai/serving_chat.py` modified +8/-2; `vllm/reasoning/__init__.py` modified +4/-0
  - docs: `docs/features/reasoning_outputs.md` modified +3/-1
- 验证与风险: diff 自带测试面 `tests/reasoning/test_deepseekv3_reasoning_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #29867 - [Quantization] fix: overflow with static per-tensor scaling

- 链接: https://github.com/vllm-project/vllm/pull/29867
- 状态/时间: merged / 2026-01-13
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+71/-56，可读 patch 182 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Quantization] fix: overflow with static per-tensor scaling」；模型线: DeepSeek V3.1；类别: 缺陷修复；主要 diff: `vllm/model_executor/layers/quantization/utils/quant_utils.py`, `vllm/v1/attention/backends/mla/common.py`；PR 正文摘要: When dequantizing weights with the `eye` method, the static scaling factor may actually push the 1s out of float8 range. Don't use that method when there's static scaling factors.。
- 实现要点: `vllm/model_executor/layers/quantization/utils/quant_utils.py` modified +61/-2 (63 lines); hunks: -5,7 +5,7; -15,6 +15,9; symbols: scaled_dequantize, get_attribute_fallback, get_and_maybe_dequant_weights, pack_quantized_values_into_int32，涉及 `scaled_dequantize, get_attribute_fallback, get_and_maybe_dequant_weights`；`vllm/v1/attention/backends/mla/common.py` modified +10/-54 (64 lines); hunks: -207,8 +207,9; -1184,35 +1185,13 @@ def __init__(; symbols: __init__, process_weights_after_loading, get_layer_weight, get_and_maybe_dequant_weights，涉及 `__init__, process_weights_after_loading, get_layer_weight`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/utils/quant_utils.py` modified +61/-2 (63 lines); hunks: -5,7 +5,7; -15,6 +15,9; symbols: scaled_dequantize, get_attribute_fallback, get_and_maybe_dequant_weights, pack_quantized_values_into_int32
  - `vllm/v1/attention/backends/mla/common.py` modified +10/-54 (64 lines); hunks: -207,8 +207,9; -1184,35 +1185,13 @@ def __init__(; symbols: __init__, process_weights_after_loading, get_layer_weight, get_and_maybe_dequant_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/utils/quant_utils.py
@@ -5,7 +5,7 @@
-from typing import ClassVar, NamedTuple
+from typing import TYPE_CHECKING, ClassVar, NamedTuple
@@ -15,6 +15,9 @@
+if TYPE_CHECKING:
+    from vllm.model_executor.layers.linear import LinearBase
@@ -239,7 +242,7 @@ def scaled_dequantize(
diff -- vllm/v1/attention/backends/mla/common.py
@@ -207,8 +207,9 @@
-    LinearBase,
-    UnquantizedLinearMethod,
+)
+from vllm.model_executor.layers.quantization.utils.quant_utils import (
+    get_and_maybe_dequant_weights,
@@ -1184,35 +1185,13 @@ def __init__(
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/utils/quant_utils.py` modified +61/-2; `vllm/v1/attention/backends/mla/common.py` modified +10/-54
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/quantization/utils/quant_utils.py`, `vllm/v1/attention/backends/mla/common.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #32361 - [BugFix] Fix DeepSeek-V3.1 + DeepGEMM incompatible scale shapes

- 链接: https://github.com/vllm-project/vllm/pull/32361
- 状态/时间: merged / 2026-01-15
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-0，可读 patch 10 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[BugFix] Fix DeepSeek-V3.1 + DeepGEMM incompatible scale shapes」；模型线: DeepSeek V3.1；类别: 缺陷修复；主要 diff: `vllm/model_executor/layers/quantization/utils/quant_utils.py`；PR 正文摘要: https://github.com/vllm-project/vllm/pull/29867 broke with For DeepGEMM revert to the behavior before https://github.com/vllm-project/vllm/pull/29867 Credit to: Eldar Kurtić for...。
- 实现要点: `vllm/model_executor/layers/quantization/utils/quant_utils.py` modified +3/-0 (3 lines); hunks: -299,6 +299,9 @@ def get_and_maybe_dequant_weights(; symbols: get_and_maybe_dequant_weights，涉及 `get_and_maybe_dequant_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/utils/quant_utils.py` modified +3/-0 (3 lines); hunks: -299,6 +299,9 @@ def get_and_maybe_dequant_weights(; symbols: get_and_maybe_dequant_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/utils/quant_utils.py
@@ -299,6 +299,9 @@ def get_and_maybe_dequant_weights(
+        # DeepGEMM transforms the scales using `transform_sf_into_required_layout` into
+        # a layout that is not compatible with `scaled_dequantize`.
+        and not layer.quant_method.use_deep_gemm
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/utils/quant_utils.py` modified +3/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/quantization/utils/quant_utils.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
