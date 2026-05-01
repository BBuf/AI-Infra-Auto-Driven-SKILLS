# sglang GLM-4.6/4.7 模型 PR 优化历史

## 覆盖范围

- 重做日期: 2026-05-01
- 源码基线: `sgl-project/sglang` 当前追溯 worktree commit `4197c55968`
- PR 收集规则: 先从模型实现、配置、processor、parser、docs/tests 等相关文件执行 `git log --name-only -- <model-files>`，再按 commit subject 的模型关键词过滤，最后用 GitHub Pull Request files API 读取每个 PR 的最终 diff。
- 额外保留规则: 原 history/skill 已显式引用但未出现在当前实现文件 git trace 中的 PR 会保留，并在卡片里标注来源。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `docs_new/cookbook/autoregressive/GLM/GLM-4.6.mdx` | 无直接 PR 号提交 |
| `docs_new/cookbook/autoregressive/GLM/GLM-4.6V.mdx` | 无直接 PR 号提交 |
| `docs_new/cookbook/autoregressive/GLM/GLM-4.7-Flash.mdx` | 无直接 PR 号提交 |
| `docs_new/cookbook/autoregressive/GLM/GLM-4.7.mdx` | 无直接 PR 号提交 |
| `python/sglang/srt/function_call/glm47_moe_detector.py` | [#15333](https://github.com/sgl-project/sglang/pull/15333), [#15753](https://github.com/sgl-project/sglang/pull/15753) |
| `python/sglang/srt/function_call/glm4_moe_detector.py` | [#13989](https://github.com/sgl-project/sglang/pull/13989), [#15333](https://github.com/sgl-project/sglang/pull/15333), [#15753](https://github.com/sgl-project/sglang/pull/15753) |
| `python/sglang/srt/models/glm4_moe.py` | [#13873](https://github.com/sgl-project/sglang/pull/13873), [#14585](https://github.com/sgl-project/sglang/pull/14585), [#15333](https://github.com/sgl-project/sglang/pull/15333), [#17166](https://github.com/sgl-project/sglang/pull/17166), [#21403](https://github.com/sgl-project/sglang/pull/21403), [#21660](https://github.com/sgl-project/sglang/pull/21660), [#21851](https://github.com/sgl-project/sglang/pull/21851) |
| `python/sglang/srt/models/glm4_moe_lite.py` | [#21851](https://github.com/sgl-project/sglang/pull/21851), [#22509](https://github.com/sgl-project/sglang/pull/22509) |
| `python/sglang/srt/models/glm4_moe_nextn.py` | [#13873](https://github.com/sgl-project/sglang/pull/13873) |
| `test/registered/amd/accuracy/mi35x/test_glm47_fp8_eval_mi35x.py` | [#21534](https://github.com/sgl-project/sglang/pull/21534) |
| `test/registered/moe/test_glm4_moe_models.py` | 无直接 PR 号提交 |

## PR 覆盖总览

- git 追溯 PR 数: 11
- 原文档显式引用补充 PR 数: 21
- 当前文档总 PR 数: 32
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-10-22 | [#11951](https://github.com/sgl-project/sglang/pull/11951) | open | WIP: Fix glm-4.6 tool call streaming parse | `sgl-router/src/tool_parser/parsers/glm4_moe_parser.rs`, `python/sglang/srt/function_call/glm4_moe_detector.py`, `sgl-router/tests/tool_parser_glm4_moe.rs` |
| 2025-11-05 | [#12456](https://github.com/sgl-project/sglang/pull/12456) | merged | [fix] Handle escaped characters in GLM tool call parser to prevent double serialization | `test/srt/test_function_call_parser.py`, `python/sglang/srt/function_call/glm4_moe_detector.py` |
| 2025-11-25 | [#13786](https://github.com/sgl-project/sglang/pull/13786) | merged | Overlap glm moe gemms in two cuda streams | `python/sglang/srt/models/glm4_moe.py` |
| 2025-12-01 | [#13873](https://github.com/sgl-project/sglang/pull/13873) | merged | Feat: GLM-4.6 supports shared experts fusion | `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_nextn.py` |
| 2025-12-08 | [#14568](https://github.com/sgl-project/sglang/pull/14568) | closed | update custom_logit_processor.py for. GLM-4.5 and GLM-4.6 support | `python/sglang/srt/models/glm4v_moe.py`, `docs/basic_usage/glmv.md`, `docs/basic_usage/glm45.md` |
| 2025-12-08 | [#14585](https://github.com/sgl-project/sglang/pull/14585) | merged | [Glm46v] Bug fix for accuracy drop and unable to launch server | `python/sglang/srt/models/glm4_moe.py` |
| 2025-12-13 | [#13989](https://github.com/sgl-project/sglang/pull/13989) | merged | Fix GLM-4.6 tool calls don't support streaming output for arguments i… | `python/sglang/srt/function_call/glm4_moe_detector.py` |
| 2025-12-20 | [#15333](https://github.com/sgl-project/sglang/pull/15333) | merged | [GLM-4.7] GLM-4.7 Tool Parser and Doc Update | `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`, `python/sglang/srt/models/glm4_moe.py` |
| 2025-12-21 | [#15520](https://github.com/sgl-project/sglang/pull/15520) | merged | [model-gateway]: Tool parser for glm47 | `sgl-model-gateway/tests/tool_parser_glm47_moe.rs`, `sgl-model-gateway/src/tool_parser/parsers/glm4_moe.rs`, `sgl-model-gateway/tests/tool_parser_glm4_moe.rs` |
| 2025-12-30 | [#15754](https://github.com/sgl-project/sglang/pull/15754) | merged | Fix: Handle empty func_name and None values in GLM MoE detectors | `test/registered/function_call/test_glm47_moe_detector.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py` |
| 2026-01-09 | [#15753](https://github.com/sgl-project/sglang/pull/15753) | merged | Fix GLM-4.7 MoE Detector complex JSON Schema type parsing | `test/registered/function_call/test_glm47_moe_detector.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py` |
| 2026-01-20 | [#17247](https://github.com/sgl-project/sglang/pull/17247) | merged | [New Model] GLM4.7-Flash | `python/sglang/srt/models/glm4_moe_lite.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` |
| 2026-01-21 | [#17166](https://github.com/sgl-project/sglang/pull/17166) | merged | [Fix] GLM 4.7 + NVFP4 + MTP | `python/sglang/srt/models/glm4_moe.py` |
| 2026-01-24 | [#14668](https://github.com/sgl-project/sglang/pull/14668) | merged | [NVIDIA] Add flashinfer all-to-all MOE dispatcher | `python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py`, `python/sglang/srt/layers/moe/token_dispatcher/flashinfer_utils.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py` |
| 2026-01-28 | [#17869](https://github.com/sgl-project/sglang/pull/17869) | open | [NPU]Support model GLM-4.7-Flash for npu, accuracy 81% | `test/registered/ascend/llm_models/test_ascend_glm4_7_flash.py`, `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` |
| 2026-02-07 | [#18383](https://github.com/sgl-project/sglang/pull/18383) | open | [Bug Fix] Add missing use_mla guard in aiter_backend draft_extend CUD… | `python/sglang/srt/layers/attention/aiter_backend.py` |
| 2026-02-17 | [#18930](https://github.com/sgl-project/sglang/pull/18930) | open | [AMD] Unit tests for mtp in GLM-4.7 | `python/sglang/srt/layers/attention/aiter_backend.py`, `test/registered/amd/test_glm4v_fp8_mtp.py` |
| 2026-02-20 | [#19040](https://github.com/sgl-project/sglang/pull/19040) | open | feat: add Glm4MoeLiteConfig and fix enable_a2a_moe for GLM-4.7-Flash | `python/sglang/srt/configs/glm4_moe_lite.py`, `python/sglang/srt/configs/__init__.py`, `python/sglang/srt/models/glm4_moe_lite.py` |
| 2026-02-21 | [#19106](https://github.com/sgl-project/sglang/pull/19106) | open | Fix GLM4 MoE Lite CompressedTensors serving and transformers version checks | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/glm4_moe_lite.py`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` |
| 2026-03-26 | [#21135](https://github.com/sgl-project/sglang/pull/21135) | merged | fix: use get_rope_config() to support models without rope_parameters | `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4.py`, `python/sglang/srt/models/grok.py` |
| 2026-03-28 | [#21534](https://github.com/sgl-project/sglang/pull/21534) | merged | [AMD] Add GLM-4.7-FP8 accuracy CI test for MI35x | `test/registered/amd/accuracy/mi35x/test_glm47_fp8_eval_mi35x.py` |
| 2026-03-30 | [#21660](https://github.com/sgl-project/sglang/pull/21660) | merged | [GLM-V and GLM-4.7] Cast to FP32 before gate projection for GLM model. | `python/sglang/srt/models/glm4_moe.py` |
| 2026-04-03 | [#19246](https://github.com/sgl-project/sglang/pull/19246) | merged | [NPU] optimize glm4.7 | `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_nextn.py`, `python/sglang/srt/layers/quantization/modelslim/modelslim.py` |
| 2026-04-04 | [#21851](https://github.com/sgl-project/sglang/pull/21851) | merged | GLM-4.7 and GLM-4.7-Flash Loading and import format | `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_lite.py` |
| 2026-04-08 | [#22315](https://github.com/sgl-project/sglang/pull/22315) | open | [Bugfix] Fix GLM-4.7-FP8 EAGLE accept_len=1.00 due to draft model loading with incorrect quant_config | `python/sglang/srt/models/glm4_moe_nextn.py` |
| 2026-04-09 | [#20543](https://github.com/sgl-project/sglang/pull/20543) | merged | fix: do not strip whitespace from GLM tool call values | `test/registered/unit/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py` |
| 2026-04-11 | [#21403](https://github.com/sgl-project/sglang/pull/21403) | merged | [AMD] Fuse RMSNorm + FP8 per-token quant for GLM-4.7-FP8 | `python/sglang/srt/models/glm4_moe.py` |
| 2026-04-13 | [#22720](https://github.com/sgl-project/sglang/pull/22720) | merged | fix[glm4.7 flash]: properly detect `gfx95_quant_format` | `python/sglang/srt/models/glm4_moe_lite.py` |
| 2026-04-14 | [#22801](https://github.com/sgl-project/sglang/pull/22801) | open | [NPU]add dual-stream and deepep support for GLM-4.7-Flash | `python/sglang/srt/models/glm4_moe_lite.py`, `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` |
| 2026-04-15 | [#22823](https://github.com/sgl-project/sglang/pull/22823) | merged | [Bugfix] Preserve auto-detected quant_config for GLM NextN draft model | `python/sglang/srt/models/glm4_moe_nextn.py` |
| 2026-04-17 | [#23067](https://github.com/sgl-project/sglang/pull/23067) | open | Fix: forward continue_final_message kwargs in Glm45Detector | `test/registered/unit/parser/test_reasoning_parser.py`, `python/sglang/srt/parser/reasoning_parser.py` |
| 2026-04-22 | [#22509](https://github.com/sgl-project/sglang/pull/22509) | merged | [NPU]Fix GLM-4.7-Flash failed on NPU | `python/sglang/srt/models/glm4_moe_lite.py` |

## 逐 PR diff 审计卡

### PR #11951 - WIP: Fix glm-4.6 tool call streaming parse

- 链接: https://github.com/sgl-project/sglang/pull/11951
- 状态/时间: open / 2025-10-22
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+450/-105，可读 patch 660 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「WIP: Fix glm-4.6 tool call streaming parse」；模型线: GLM-4.6/4.7；类别: 缺陷修复；主要 diff: `sgl-router/src/tool_parser/parsers/glm4_moe_parser.rs`, `python/sglang/srt/function_call/glm4_moe_detector.py`, `sgl-router/tests/tool_parser_glm4_moe.rs`；技术摘要: 覆盖「WIP: Fix glm-4.6 tool call streaming parse」；主要实现面是 `sgl-router/src/tool_parser/parsers/glm4_moe_parser.rs`, `python/sglang/srt/function_call/glm4_moe_detector.py`, `sgl-router/tests/tool_parser_glm4_moe.rs`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `sgl-router/src/tool_parser/parsers/glm4_moe_parser.rs` modified +198/-86 (284 lines); hunks: -41,6 +41,9 @@ pub struct Glm4MoeParser {; -67,6 +70,7 @@ impl Glm4MoeParser {；`python/sglang/srt/function_call/glm4_moe_detector.py` modified +180/-19 (199 lines); hunks: -6,7 +6,11; -99,6 +103,7 @@ def parse_streaming_increment(; symbols: parse_streaming_increment, _parse_partial_tool_call, _find_common_prefix, supports_structural_tag，涉及 `parse_streaming_increment, _parse_partial_tool_call, _find_common_prefix`；`sgl-router/tests/tool_parser_glm4_moe.rs` modified +72/-0 (72 lines); hunks: -167,3 +167,75 @@ async fn test_glm4_nested_json_in_arg_values() {。
- 代码 diff 细节:
  - `sgl-router/src/tool_parser/parsers/glm4_moe_parser.rs` modified +198/-86 (284 lines); hunks: -41,6 +41,9 @@ pub struct Glm4MoeParser {; -67,6 +70,7 @@ impl Glm4MoeParser {
  - `python/sglang/srt/function_call/glm4_moe_detector.py` modified +180/-19 (199 lines); hunks: -6,7 +6,11; -99,6 +103,7 @@ def parse_streaming_increment(; symbols: parse_streaming_increment, _parse_partial_tool_call, _find_common_prefix, supports_structural_tag
  - `sgl-router/tests/tool_parser_glm4_moe.rs` modified +72/-0 (72 lines); hunks: -167,3 +167,75 @@ async fn test_glm4_nested_json_in_arg_values() {
- 关键代码摘录:

```diff
diff -- sgl-router/src/tool_parser/parsers/glm4_moe_parser.rs
@@ -41,6 +41,9 @@ pub struct Glm4MoeParser {
+    /// Whether the current tool's name has been sent (for streaming)
+    current_tool_name_sent: bool,
@@ -67,6 +70,7 @@ impl Glm4MoeParser {
+            current_tool_name_sent: false,
@@ -154,6 +158,79 @@ impl Glm4MoeParser {
+    /// Parse partial tool call from buffer (for streaming)
diff -- python/sglang/srt/function_call/glm4_moe_detector.py
@@ -6,7 +6,11 @@
-from sglang.srt.function_call.core_types import StreamingParseResult, _GetInfoFunc
+from sglang.srt.function_call.core_types import (
+    StreamingParseResult,
+    ToolCallItem,
+    _GetInfoFunc,
+)
diff -- sgl-router/tests/tool_parser_glm4_moe.rs
@@ -167,3 +167,75 @@ async fn test_glm4_nested_json_in_arg_values() {
```

- 已读文件:
  - runtime: `sgl-router/src/tool_parser/parsers/glm4_moe_parser.rs` modified +198/-86; `python/sglang/srt/function_call/glm4_moe_detector.py` modified +180/-19
  - tests: `sgl-router/tests/tool_parser_glm4_moe.rs` modified +72/-0
- 验证与风险: diff 自带测试面 `sgl-router/tests/tool_parser_glm4_moe.rs`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #12456 - [fix] Handle escaped characters in GLM tool call parser to prevent double serialization

- 链接: https://github.com/sgl-project/sglang/pull/12456
- 状态/时间: merged / 2025-11-05
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+127/-13，可读 patch 172 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[fix] Handle escaped characters in GLM tool call parser to prevent double serialization」；模型线: GLM-4.6/4.7；类别: 缺陷修复；主要 diff: `test/srt/test_function_call_parser.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`；技术摘要: 覆盖「[fix] Handle escaped characters in GLM tool call parser to prevent double serialization」；主要实现面是 `test/srt/test_function_call_parser.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/srt/test_function_call_parser.py` modified +103/-0 (103 lines); hunks: -2191,6 +2191,109 @@ def test_partial_tool_call(self):; symbols: test_partial_tool_call, test_array_argument_with_escaped_json, check_params, check_single_todos，涉及 `test_partial_tool_call, test_array_argument_with_escaped_json, check_params`；`python/sglang/srt/function_call/glm4_moe_detector.py` modified +24/-13 (37 lines); hunks: -24,13 +24,23 @@ def get_argument_type(func_name: str, arg_key: str, defined_...; -45,8 +55,13 @@ def __init__(self):; symbols: get_argument_type, parse_arguments, Glm4MoeDetector, __init__，涉及 `get_argument_type, parse_arguments, Glm4MoeDetector`。
- 代码 diff 细节:
  - `test/srt/test_function_call_parser.py` modified +103/-0 (103 lines); hunks: -2191,6 +2191,109 @@ def test_partial_tool_call(self):; symbols: test_partial_tool_call, test_array_argument_with_escaped_json, check_params, check_single_todos
  - `python/sglang/srt/function_call/glm4_moe_detector.py` modified +24/-13 (37 lines); hunks: -24,13 +24,23 @@ def get_argument_type(func_name: str, arg_key: str, defined_...; -45,8 +55,13 @@ def __init__(self):; symbols: get_argument_type, parse_arguments, Glm4MoeDetector, __init__
- 关键代码摘录:

```diff
diff -- test/srt/test_function_call_parser.py
@@ -2191,6 +2191,109 @@ def test_partial_tool_call(self):
+    def test_array_argument_with_escaped_json(self):
+        """Test that array arguments with escaped JSON are properly handled without double-escaping."""
+        # Add a tool with array parameter
+        tools_with_array = [
+            Tool(
+                type="function",
diff -- python/sglang/srt/function_call/glm4_moe_detector.py
@@ -24,13 +24,23 @@ def get_argument_type(func_name: str, arg_key: str, defined_tools: list):
-        try:
-            parsed_value = json.loads(json_value)
-        except:
-            parsed_value = ast.literal_eval(json_value)
+        parsed_value = json.loads(json_value)
-        return json_value, False
```

- 已读文件:
  - tests: `test/srt/test_function_call_parser.py` modified +103/-0
  - runtime: `python/sglang/srt/function_call/glm4_moe_detector.py` modified +24/-13
- 验证与风险: diff 自带测试面 `test/srt/test_function_call_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #13786 - Overlap glm moe gemms in two cuda streams

- 链接: https://github.com/sgl-project/sglang/pull/13786
- 状态/时间: merged / 2025-11-25
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+47/-3，可读 patch 60 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Overlap glm moe gemms in two cuda streams」；模型线: GLM-4.6/4.7；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/glm4_moe.py`；技术摘要: 覆盖「Overlap glm moe gemms in two cuda streams」；主要实现面是 `python/sglang/srt/models/glm4_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/glm4_moe.py` modified +47/-3 (50 lines); hunks: -448,12 +448,56 @@ def forward(; symbols: forward, forward_normal_dual_stream, forward_normal，涉及 `forward, forward_normal_dual_stream, forward_normal`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4_moe.py` modified +47/-3 (50 lines); hunks: -448,12 +448,56 @@ def forward(; symbols: forward, forward_normal_dual_stream, forward_normal
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4_moe.py
@@ -448,12 +448,56 @@ def forward(
-            return self.forward_normal(
-                hidden_states, should_allreduce_fusion, use_reduce_scatter
-            )
+            if (
+                self.alt_stream is not None
+                and hidden_states.shape[0] > 0
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4_moe.py` modified +47/-3
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/glm4_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #13873 - Feat: GLM-4.6 supports shared experts fusion

- 链接: https://github.com/sgl-project/sglang/pull/13873
- 状态/时间: merged / 2025-12-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_nextn.py`；关联提交 `982db4ebac26`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+252/-24，可读 patch 431 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Feat: GLM-4.6 supports shared experts fusion」；模型线: GLM-4.6/4.7；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_nextn.py`；技术摘要: 覆盖「Feat: GLM-4.6 supports shared experts fusion」；主要实现面是 `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_nextn.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/glm4_moe.py` modified +74/-19 (93 lines); hunks: -85,6 +85,7; -352,8 +353,14 @@ def __init__(; symbols: __init__, forward, forward_normal_dual_stream，涉及 `__init__, forward, forward_normal_dual_stream`；`python/sglang/srt/models/glm4_moe_nextn.py` modified +4/-0 (4 lines); hunks: -139,6 +139,10 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4_moe.py` modified +74/-19 (93 lines); hunks: -85,6 +85,7; -352,8 +353,14 @@ def __init__(; symbols: __init__, forward, forward_normal_dual_stream
  - `python/sglang/srt/models/glm4_moe_nextn.py` modified +4/-0 (4 lines); hunks: -139,6 +139,10 @@ def __init__(; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4_moe.py
@@ -85,6 +85,7 @@
+    log_info_on_rank0,
@@ -352,8 +353,14 @@ def __init__(
+        self.moe_ep_size = get_moe_expert_parallel_world_size()
+        self.num_fused_shared_experts = (
+            0
+            if get_global_server_args().disable_shared_experts_fusion
diff -- python/sglang/srt/models/glm4_moe_nextn.py
@@ -139,6 +139,10 @@ def __init__(
+        self.num_fused_shared_experts = (
+            0 if get_global_server_args().disable_shared_experts_fusion else 1
+        )
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4_moe.py` modified +74/-19; `python/sglang/srt/models/glm4_moe_nextn.py` modified +4/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=161,N=192,device_name=NVIDIA_H200,dtype=fp8_w8a8,per_channel_quant=True.json`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #14568 - update custom_logit_processor.py for. GLM-4.5 and GLM-4.6 support

- 链接: https://github.com/sgl-project/sglang/pull/14568
- 状态/时间: closed / 2025-12-08
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+222/-1，可读 patch 269 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「update custom_logit_processor.py for. GLM-4.5 and GLM-4.6 support」；模型线: GLM-4.6/4.7；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/glm4v_moe.py`, `docs/basic_usage/glmv.md`, `docs/basic_usage/glm45.md`；未提供可用技术摘要。
- 实现要点: `python/sglang/srt/models/glm4v_moe.py` modified +4/-0 (4 lines); hunks: -7,6 +7,7; -36,7 +37,9 @@ def __init__(; symbols: __init__，涉及 `__init__`；`docs/basic_usage/glmv.md` added +136/-0 (136 lines); hunks: -0,0 +1,136；`docs/basic_usage/glm45.md` added +70/-0 (70 lines); hunks: -0,0 +1,70；`python/sglang/srt/sampling/custom_logit_processor.py` modified +8/-0 (8 lines); hunks: -112,6 +112,14 @@ def __call__(self, logits, custom_param_list: list[dict[str...; symbols: __call__, Glm4MoeThinkingBudgetLogitProcessor, Qwen3ThinkingBudgetLogitProcessor，涉及 `__call__, Glm4MoeThinkingBudgetLogitProcessor, Qwen3ThinkingBudgetLogitProcessor`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4v_moe.py` modified +4/-0 (4 lines); hunks: -7,6 +7,7; -36,7 +37,9 @@ def __init__(; symbols: __init__
  - `docs/basic_usage/glmv.md` added +136/-0 (136 lines); hunks: -0,0 +1,136
  - `docs/basic_usage/glm45.md` added +70/-0 (70 lines); hunks: -0,0 +1,70
  - `python/sglang/srt/sampling/custom_logit_processor.py` modified +8/-0 (8 lines); hunks: -112,6 +112,14 @@ def __call__(self, logits, custom_param_list: list[dict[str...; symbols: __call__, Glm4MoeThinkingBudgetLogitProcessor, Qwen3ThinkingBudgetLogitProcessor
  - `docs/basic_usage/popular_model_usage.rst` modified +3/-1 (4 lines); hunks: -1,11 +1,13
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4v_moe.py
@@ -7,6 +7,7 @@
+from sglang.srt.distributed.parallel_state import get_pp_group
@@ -36,7 +37,9 @@ def __init__(
+        self.pp_group = get_pp_group()
+        self.use_data_parallel = get_global_server_args().mm_enable_dp_encoder
@@ -55,6 +58,7 @@ def __init__(
+            use_data_parallel=self.use_data_parallel,
diff -- docs/basic_usage/glmv.md
@@ -0,0 +1,136 @@
+# GLM-4.6V / GLM-4.5V Usage
+## Launch commands for SGLang
+Below are suggested launch commands tailored for different hardware / precision modes
+### FP8 (quantised) mode
+For high memory-efficiency and latency optimized deployments (e.g., on H100, H200) where FP8 checkpoint is supported:
+'''bash
diff -- docs/basic_usage/glm45.md
@@ -0,0 +1,70 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4v_moe.py` modified +4/-0; `python/sglang/srt/sampling/custom_logit_processor.py` modified +8/-0
  - docs: `docs/basic_usage/glmv.md` added +136/-0; `docs/basic_usage/glm45.md` added +70/-0; `docs/basic_usage/popular_model_usage.rst` modified +3/-1; `docs/advanced_features/dp_for_multi_modal_encoder.md` modified +1/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/sampling/custom_logit_processor.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #14585 - [Glm46v] Bug fix for accuracy drop and unable to launch server

- 链接: https://github.com/sgl-project/sglang/pull/14585
- 状态/时间: merged / 2025-12-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/glm4_moe.py`；关联提交 `cf0478d602ce`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 12 个文件，+308/-29，可读 patch 530 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Glm46v] Bug fix for accuracy drop and unable to launch server」；模型线: GLM-4.6/4.7；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/glm4_moe.py`；技术摘要: 覆盖「[Glm46v] Bug fix for accuracy drop and unable to launch server」；主要实现面是 `python/sglang/srt/models/glm4_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/glm4_moe.py` modified +1/-0 (1 lines); hunks: -361,6 +361,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4_moe.py` modified +1/-0 (1 lines); hunks: -361,6 +361,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4_moe.py
@@ -361,6 +361,7 @@ def __init__(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4_moe.py` modified +1/-0
- 验证与风险: runtime 路径改动集中在 `python/pyproject.toml`, `python/sglang/srt/configs/qwen3_omni.py`, `python/sglang/srt/configs/qwen3_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #13989 - Fix GLM-4.6 tool calls don't support streaming output for arguments i…

- 链接: https://github.com/sgl-project/sglang/pull/13989
- 状态/时间: merged / 2025-12-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/function_call/glm4_moe_detector.py`；关联提交 `80554598d33b`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+527/-81，可读 patch 700 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix GLM-4.6 tool calls don't support streaming output for arguments i…」；模型线: GLM-4.6/4.7；类别: 缺陷修复；主要 diff: `python/sglang/srt/function_call/glm4_moe_detector.py`；技术摘要: 覆盖「Fix GLM-4.6 tool calls don't support streaming output for arguments i…」；主要实现面是 `python/sglang/srt/function_call/glm4_moe_detector.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/function_call/glm4_moe_detector.py` modified +498/-66 (564 lines); hunks: -2,16 +2,43; -21,32 +48,90 @@ def get_argument_type(func_name: str, arg_key: str, defined_...; symbols: get_argument_type, StreamState, parse_arguments，涉及 `get_argument_type, StreamState, parse_arguments`。
- 代码 diff 细节:
  - `python/sglang/srt/function_call/glm4_moe_detector.py` modified +498/-66 (564 lines); hunks: -2,16 +2,43; -21,32 +48,90 @@ def get_argument_type(func_name: str, arg_key: str, defined_...; symbols: get_argument_type, StreamState, parse_arguments
- 关键代码摘录:

```diff
diff -- python/sglang/srt/function_call/glm4_moe_detector.py
@@ -2,16 +2,43 @@
-from typing import List
+from enum import Enum
+from typing import Any, Dict, List, Optional, Tuple
-from sglang.srt.function_call.core_types import StreamingParseResult, _GetInfoFunc
+from sglang.srt.function_call.core_types import (
+    StreamingParseResult,
```

- 已读文件:
  - runtime: `python/sglang/srt/function_call/glm4_moe_detector.py` modified +498/-66
- 验证与风险: diff 自带测试面 `test/registered/function_call/test_function_call_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #15333 - [GLM-4.7] GLM-4.7 Tool Parser and Doc Update

- 链接: https://github.com/sgl-project/sglang/pull/15333
- 状态/时间: merged / 2025-12-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`, `python/sglang/srt/models/glm4_moe.py`；关联提交 `b82c7a0ae744`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+809/-394，可读 patch 1356 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[GLM-4.7] GLM-4.7 Tool Parser and Doc Update」；模型线: GLM-4.6/4.7；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`, `python/sglang/srt/models/glm4_moe.py`；技术摘要: 覆盖「[GLM-4.7] GLM-4.7 Tool Parser and Doc Update」；主要实现面是 `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`, `python/sglang/srt/models/glm4_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/function_call/glm47_moe_detector.py` added +584/-0 (584 lines); hunks: -0,0 +1,584; symbols: StreamState, get_argument_type, _convert_to_number, parse_arguments，涉及 `StreamState, get_argument_type, _convert_to_number`；`python/sglang/srt/function_call/glm4_moe_detector.py` modified +5/-2 (7 lines); hunks: -43,9 +43,12 @@ def get_argument_type(; symbols: get_argument_type, _convert_to_number，涉及 `get_argument_type, _convert_to_number`；`python/sglang/srt/models/glm4_moe.py` modified +1/-1 (2 lines); hunks: -12,7 +12,7。
- 代码 diff 细节:
  - `python/sglang/srt/function_call/glm47_moe_detector.py` added +584/-0 (584 lines); hunks: -0,0 +1,584; symbols: StreamState, get_argument_type, _convert_to_number, parse_arguments
  - `python/sglang/srt/function_call/glm4_moe_detector.py` modified +5/-2 (7 lines); hunks: -43,9 +43,12 @@ def get_argument_type(; symbols: get_argument_type, _convert_to_number
  - `python/sglang/srt/models/glm4_moe.py` modified +1/-1 (2 lines); hunks: -12,7 +12,7
- 关键代码摘录:

```diff
diff -- python/sglang/srt/function_call/glm47_moe_detector.py
@@ -0,0 +1,584 @@
+import ast
+import json
+import logging
+import re
+from enum import Enum
+from typing import Any, Dict, List, Optional, Tuple
diff -- python/sglang/srt/function_call/glm4_moe_detector.py
@@ -43,9 +43,12 @@ def get_argument_type(
-    if arg_key not in tool.function.parameters["properties"]:
+    properties = (tool.function.parameters or {}).get("properties", {})
+    if not isinstance(properties, dict):
+        properties = {}
+    if arg_key not in properties:
-    return tool.function.parameters["properties"][arg_key].get("type", None)
diff -- python/sglang/srt/models/glm4_moe.py
@@ -12,7 +12,7 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/function_call/glm47_moe_detector.py` added +584/-0; `python/sglang/srt/function_call/glm4_moe_detector.py` modified +5/-2; `python/sglang/srt/models/glm4_moe.py` modified +1/-1
- 验证与风险: diff 自带测试面 `test/registered/function_call/test_function_call_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #15520 - [model-gateway]: Tool parser for glm47

- 链接: https://github.com/sgl-project/sglang/pull/15520
- 状态/时间: merged / 2025-12-21
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+179/-26，可读 patch 392 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[model-gateway]: Tool parser for glm47」；模型线: GLM-4.6/4.7；类别: 模型支持/运行时入口；主要 diff: `sgl-model-gateway/tests/tool_parser_glm47_moe.rs`, `sgl-model-gateway/src/tool_parser/parsers/glm4_moe.rs`, `sgl-model-gateway/tests/tool_parser_glm4_moe.rs`；技术摘要: 覆盖「[model-gateway]: Tool parser for glm47」；主要实现面是 `sgl-model-gateway/tests/tool_parser_glm47_moe.rs`, `sgl-model-gateway/src/tool_parser/parsers/glm4_moe.rs`, `sgl-model-gateway/tests/tool_parser_glm4_moe.rs`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `sgl-model-gateway/tests/tool_parser_glm47_moe.rs` added +132/-0 (132 lines); hunks: -0,0 +1,132；`sgl-model-gateway/src/tool_parser/parsers/glm4_moe.rs` modified +22/-8 (30 lines); hunks: -14,8 +14,9 @@ use crate::{; -47,13 +48,17 @@ pub struct Glm4MoeParser {；`sgl-model-gateway/tests/tool_parser_glm4_moe.rs` modified +7/-7 (14 lines); hunks: -7,7 +7,7 @@ use common::create_test_tools;; -30,7 +30,7 @@ The weather will be..."#;；`sgl-model-gateway/src/tool_parser/factory.rs` modified +5/-3 (8 lines); hunks: -239,7 +239,8 @@ impl ParserFactory {; -281,8 +282,9 @@ impl ParserFactory {。
- 代码 diff 细节:
  - `sgl-model-gateway/tests/tool_parser_glm47_moe.rs` added +132/-0 (132 lines); hunks: -0,0 +1,132
  - `sgl-model-gateway/src/tool_parser/parsers/glm4_moe.rs` modified +22/-8 (30 lines); hunks: -14,8 +14,9 @@ use crate::{; -47,13 +48,17 @@ pub struct Glm4MoeParser {
  - `sgl-model-gateway/tests/tool_parser_glm4_moe.rs` modified +7/-7 (14 lines); hunks: -7,7 +7,7 @@ use common::create_test_tools;; -30,7 +30,7 @@ The weather will be..."#;
  - `sgl-model-gateway/src/tool_parser/factory.rs` modified +5/-3 (8 lines); hunks: -239,7 +239,8 @@ impl ParserFactory {; -281,8 +282,9 @@ impl ParserFactory {
  - `sgl-model-gateway/benches/tool_parser_benchmark.rs` modified +5/-2 (7 lines); hunks: -80,7 +80,7 @@ Let me examine the scan results and provide recommendations."#;; -94,6 +94,8 @@ analyze_customer_behavior
- 关键代码摘录:

```diff
diff -- sgl-model-gateway/tests/tool_parser_glm47_moe.rs
@@ -0,0 +1,132 @@
+//! GLM-4.7 MoE Parser Integration Tests
+use sgl_model_gateway::tool_parser::{Glm4MoeParser, ToolParser};
+mod common;
+use common::create_test_tools;
+#[tokio::test]
+async fn test_glm47_complete_parsing() {
diff -- sgl-model-gateway/src/tool_parser/parsers/glm4_moe.rs
@@ -14,8 +14,9 @@ use crate::{
-/// Handles the GLM-4 MoE specific format:
-/// `<tool_call>{name}\n<arg_key>{key}</arg_key>\n<arg_value>{value}</arg_value>\n</tool_call>`
+/// Handles both GLM-4 MoE and GLM-4.7 MoE formats:
+/// - GLM-4: `<tool_call>{name}\n<arg_key>{key}</arg_key>\n<arg_value>{value}</arg_value>\n</tool_call>`
+/// - GLM-4.7: `<tool_call>{name}<arg_key>{key}</arg_key><arg_value>{value}</arg_value></tool_call>`
@@ -47,13 +48,17 @@ pub struct Glm4MoeParser {
diff -- sgl-model-gateway/tests/tool_parser_glm4_moe.rs
@@ -7,7 +7,7 @@ use common::create_test_tools;
```

- 已读文件:
  - tests: `sgl-model-gateway/tests/tool_parser_glm47_moe.rs` added +132/-0; `sgl-model-gateway/tests/tool_parser_glm4_moe.rs` modified +7/-7
  - runtime: `sgl-model-gateway/src/tool_parser/parsers/glm4_moe.rs` modified +22/-8; `sgl-model-gateway/src/tool_parser/factory.rs` modified +5/-3; `sgl-model-gateway/benches/tool_parser_benchmark.rs` modified +5/-2; `sgl-model-gateway/src/reasoning_parser/README.md` modified +4/-3; `sgl-model-gateway/src/reasoning_parser/factory.rs` modified +1/-0
  - other: `sgl-model-gateway/README.md` modified +3/-3
- 验证与风险: diff 自带测试面 `sgl-model-gateway/tests/tool_parser_glm47_moe.rs`, `sgl-model-gateway/tests/tool_parser_glm4_moe.rs`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #15754 - Fix: Handle empty func_name and None values in GLM MoE detectors

- 链接: https://github.com/sgl-project/sglang/pull/15754
- 状态/时间: merged / 2025-12-30
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+1513/-140，可读 patch 1786 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix: Handle empty func_name and None values in GLM MoE detectors」；模型线: GLM-4.6/4.7；类别: 缺陷修复；主要 diff: `test/registered/function_call/test_glm47_moe_detector.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`；技术摘要: 覆盖「Fix: Handle empty func_name and None values in GLM MoE detectors」；主要实现面是 `test/registered/function_call/test_glm47_moe_detector.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/function_call/test_glm47_moe_detector.py` added +1176/-0 (1176 lines); hunks: -0,0 +1,1176; symbols: TestGlm47MoeDetector, setUp, test_single_tool_call, test_multiple_tool_calls，涉及 `TestGlm47MoeDetector, setUp, test_single_tool_call`；`python/sglang/srt/function_call/glm47_moe_detector.py` modified +303/-132 (435 lines); hunks: -40,15 +40,27 @@ def get_argument_type(; -143,6 +155,10 @@ def __init__(self):; symbols: get_argument_type, _convert_to_number, __init__, _reset_streaming_state，涉及 `get_argument_type, _convert_to_number, __init__`；`python/sglang/srt/function_call/glm4_moe_detector.py` modified +19/-8 (27 lines); hunks: -189,8 +189,10 @@ def detect_and_parse(self, text: str, tools: List[Tool]) ->...; -426,10 +428,19 @@ def parse_streaming_increment(; symbols: detect_and_parse, parse_streaming_increment，涉及 `detect_and_parse, parse_streaming_increment`；`test/registered/function_call/test_function_call_parser.py` modified +15/-0 (15 lines); hunks: -2257,6 +2257,21 @@ def check_single_todos(tool_result, expected):; symbols: check_single_todos, test_empty_function_name_handling, TestGlm47MoeDetector, setUp，涉及 `check_single_todos, test_empty_function_name_handling, TestGlm47MoeDetector`。
- 代码 diff 细节:
  - `test/registered/function_call/test_glm47_moe_detector.py` added +1176/-0 (1176 lines); hunks: -0,0 +1,1176; symbols: TestGlm47MoeDetector, setUp, test_single_tool_call, test_multiple_tool_calls
  - `python/sglang/srt/function_call/glm47_moe_detector.py` modified +303/-132 (435 lines); hunks: -40,15 +40,27 @@ def get_argument_type(; -143,6 +155,10 @@ def __init__(self):; symbols: get_argument_type, _convert_to_number, __init__, _reset_streaming_state
  - `python/sglang/srt/function_call/glm4_moe_detector.py` modified +19/-8 (27 lines); hunks: -189,8 +189,10 @@ def detect_and_parse(self, text: str, tools: List[Tool]) ->...; -426,10 +428,19 @@ def parse_streaming_increment(; symbols: detect_and_parse, parse_streaming_increment
  - `test/registered/function_call/test_function_call_parser.py` modified +15/-0 (15 lines); hunks: -2257,6 +2257,21 @@ def check_single_todos(tool_result, expected):; symbols: check_single_todos, test_empty_function_name_handling, TestGlm47MoeDetector, setUp
- 关键代码摘录:

```diff
diff -- test/registered/function_call/test_glm47_moe_detector.py
@@ -0,0 +1,1176 @@
+import json
+import unittest
+from sglang.srt.entrypoints.openai.protocol import Function, Tool
+from sglang.srt.function_call.core_types import StreamingParseResult
+from sglang.srt.function_call.glm47_moe_detector import Glm47MoeDetector
+from sglang.test.ci.ci_register import register_cpu_ci
diff -- python/sglang/srt/function_call/glm47_moe_detector.py
@@ -40,15 +40,27 @@ def get_argument_type(
-    if func_name not in name2tool:
+    # Check if function exists
+    tool = name2tool.get(func_name)
+    if not tool:
+        return None
+    # Get parameters safely using getattr
diff -- python/sglang/srt/function_call/glm4_moe_detector.py
@@ -189,8 +189,10 @@ def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult
```

- 已读文件:
  - tests: `test/registered/function_call/test_glm47_moe_detector.py` added +1176/-0; `test/registered/function_call/test_function_call_parser.py` modified +15/-0
  - runtime: `python/sglang/srt/function_call/glm47_moe_detector.py` modified +303/-132; `python/sglang/srt/function_call/glm4_moe_detector.py` modified +19/-8
- 验证与风险: diff 自带测试面 `test/registered/function_call/test_function_call_parser.py`, `test/registered/function_call/test_glm47_moe_detector.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #15753 - Fix GLM-4.7 MoE Detector complex JSON Schema type parsing

- 链接: https://github.com/sgl-project/sglang/pull/15753
- 状态/时间: merged / 2026-01-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`；关联提交 `8ef5b9052825`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+869/-20，可读 patch 989 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix GLM-4.7 MoE Detector complex JSON Schema type parsing」；模型线: GLM-4.6/4.7；类别: 缺陷修复；主要 diff: `test/registered/function_call/test_glm47_moe_detector.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`；技术摘要: 覆盖「Fix GLM-4.7 MoE Detector complex JSON Schema type parsing」；主要实现面是 `test/registered/function_call/test_glm47_moe_detector.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/function_call/test_glm47_moe_detector.py` modified +678/-3 (681 lines); hunks: -3,7 +3,11; -1172,5 +1176,676 @@ def test_streamed_raw_length_multiple_empty_returns(self):; symbols: test_streamed_raw_length_multiple_empty_returns, TestGlm4ComplexJsonSchema, setUp, test_get_argument_type_simple_type，涉及 `test_streamed_raw_length_multiple_empty_returns, TestGlm4ComplexJsonSchema, setUp`；`python/sglang/srt/function_call/glm47_moe_detector.py` modified +43/-10 (53 lines); hunks: -12,6 +12,7; -31,6 +32,14 @@ def get_argument_type(; symbols: get_argument_type, _get_value_type, _format_value_complete，涉及 `get_argument_type, _get_value_type, _format_value_complete`；`python/sglang/srt/function_call/glm4_moe_detector.py` modified +44/-6 (50 lines); hunks: -12,6 +12,7; -31,6 +32,14 @@ def get_argument_type(; symbols: get_argument_type, _convert_to_number, _get_value_type, _format_value_complete，涉及 `get_argument_type, _convert_to_number, _get_value_type`。
- 代码 diff 细节:
  - `test/registered/function_call/test_glm47_moe_detector.py` modified +678/-3 (681 lines); hunks: -3,7 +3,11; -1172,5 +1176,676 @@ def test_streamed_raw_length_multiple_empty_returns(self):; symbols: test_streamed_raw_length_multiple_empty_returns, TestGlm4ComplexJsonSchema, setUp, test_get_argument_type_simple_type
  - `python/sglang/srt/function_call/glm47_moe_detector.py` modified +43/-10 (53 lines); hunks: -12,6 +12,7; -31,6 +32,14 @@ def get_argument_type(; symbols: get_argument_type, _get_value_type, _format_value_complete
  - `python/sglang/srt/function_call/glm4_moe_detector.py` modified +44/-6 (50 lines); hunks: -12,6 +12,7; -31,6 +32,14 @@ def get_argument_type(; symbols: get_argument_type, _convert_to_number, _get_value_type, _format_value_complete
- 关键代码摘录:

```diff
diff -- test/registered/function_call/test_glm47_moe_detector.py
@@ -3,7 +3,11 @@
-from sglang.srt.function_call.glm47_moe_detector import Glm47MoeDetector
+from sglang.srt.function_call.glm4_moe_detector import Glm4MoeDetector
+from sglang.srt.function_call.glm47_moe_detector import (
+    Glm47MoeDetector,
+    get_argument_type,
+)
diff -- python/sglang/srt/function_call/glm47_moe_detector.py
@@ -12,6 +12,7 @@
+from sglang.srt.function_call.utils import infer_type_from_json_schema
@@ -31,6 +32,14 @@ def get_argument_type(
+    Supports complex JSON Schema definitions including:
+    - Direct type field (including type arrays)
+    - anyOf/oneOf: parameter can be any of multiple types
+    - enum: parameter must be one of enum values
diff -- python/sglang/srt/function_call/glm4_moe_detector.py
@@ -12,6 +12,7 @@
```

- 已读文件:
  - tests: `test/registered/function_call/test_glm47_moe_detector.py` modified +678/-3
  - runtime: `python/sglang/srt/function_call/glm47_moe_detector.py` modified +43/-10; `python/sglang/srt/function_call/glm4_moe_detector.py` modified +44/-6
- 验证与风险: diff 自带测试面 `test/registered/function_call/test_glm47_moe_detector.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #17247 - [New Model] GLM4.7-Flash

- 链接: https://github.com/sgl-project/sglang/pull/17247
- 状态/时间: merged / 2026-01-20
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+842/-12，可读 patch 940 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[New Model] GLM4.7-Flash」；模型线: GLM-4.6/4.7；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/glm4_moe_lite.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`；技术摘要: 覆盖「[New Model] GLM4.7-Flash」；主要实现面是 `python/sglang/srt/models/glm4_moe_lite.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/glm4_moe_lite.py` added +808/-0 (808 lines); hunks: -0,0 +1,808; symbols: Glm4MoeLiteMLP, __init__, forward, Glm4MoeLiteGate，涉及 `Glm4MoeLiteMLP, __init__, forward`；`python/sglang/srt/configs/model_config.py` modified +19/-9 (28 lines); hunks: -269,7 +269,10 @@ def _config_draft_model(self):; -375,6 +378,7 @@ def _derive_model_shapes(self):; symbols: _config_draft_model, _derive_model_shapes，涉及 `_config_draft_model, _derive_model_shapes`；`python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` modified +7/-2 (9 lines); hunks: -17,7 +17,7; -472,7 +472,12 @@ def _concat_and_cast_mha_k(; symbols: _concat_and_cast_mha_k，涉及 `_concat_and_cast_mha_k`；`python/sglang/srt/models/glm4_moe.py` modified +3/-1 (4 lines); hunks: -685,6 +685,8 @@ def __init__(; -699,7 +701,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4_moe_lite.py` added +808/-0 (808 lines); hunks: -0,0 +1,808; symbols: Glm4MoeLiteMLP, __init__, forward, Glm4MoeLiteGate
  - `python/sglang/srt/configs/model_config.py` modified +19/-9 (28 lines); hunks: -269,7 +269,10 @@ def _config_draft_model(self):; -375,6 +378,7 @@ def _derive_model_shapes(self):; symbols: _config_draft_model, _derive_model_shapes
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` modified +7/-2 (9 lines); hunks: -17,7 +17,7; -472,7 +472,12 @@ def _concat_and_cast_mha_k(; symbols: _concat_and_cast_mha_k
  - `python/sglang/srt/models/glm4_moe.py` modified +3/-1 (4 lines); hunks: -685,6 +685,8 @@ def __init__(; -699,7 +701,7 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +2/-0 (2 lines); hunks: -454,6 +454,7 @@ def _apply_jinja_template(; -476,6 +477,7 @@ def _apply_jinja_template(; symbols: _apply_jinja_template
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4_moe_lite.py
@@ -0,0 +1,808 @@
+# Copyright 2025-2026 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/configs/model_config.py
@@ -269,7 +269,10 @@ def _config_draft_model(self):
-        if is_draft_model and self.hf_config.architectures[0] == "Glm4MoeForCausalLM":
+        if is_draft_model and self.hf_config.architectures[0] in [
+            "Glm4MoeForCausalLM",
+            "Glm4MoeLiteForCausalLM",
+        ]:
@@ -375,6 +378,7 @@ def _derive_model_shapes(self):
diff -- python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py
@@ -17,7 +17,7 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4_moe_lite.py` added +808/-0; `python/sglang/srt/configs/model_config.py` modified +19/-9; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` modified +7/-2; `python/sglang/srt/models/glm4_moe.py` modified +3/-1; `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +2/-0; `python/sglang/srt/server_args.py` modified +3/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #17166 - [Fix] GLM 4.7 + NVFP4 + MTP

- 链接: https://github.com/sgl-project/sglang/pull/17166
- 状态/时间: merged / 2026-01-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/glm4_moe.py`；关联提交 `2ff0880a0ed1`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+114/-9，可读 patch 206 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Fix] GLM 4.7 + NVFP4 + MTP」；模型线: GLM-4.6/4.7；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/glm4_moe.py`；技术摘要: 覆盖「[Fix] GLM 4.7 + NVFP4 + MTP」；主要实现面是 `python/sglang/srt/models/glm4_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/glm4_moe.py` modified +5/-1 (6 lines); hunks: -63,7 +63,10; -376,6 +379,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4_moe.py` modified +5/-1 (6 lines); hunks: -63,7 +63,10; -376,6 +379,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4_moe.py
@@ -63,7 +63,10 @@
-from sglang.srt.layers.moe.utils import filter_moe_weight_param_global_expert
+from sglang.srt.layers.moe.utils import (
+    RoutingMethodType,
+    filter_moe_weight_param_global_expert,
+)
@@ -376,6 +379,7 @@ def __init__(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4_moe.py` modified +5/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/model_loader/loader.py`, `python/sglang/srt/model_loader/weight_utils.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #14668 - [NVIDIA] Add flashinfer all-to-all MOE dispatcher

- 链接: https://github.com/sgl-project/sglang/pull/14668
- 状态/时间: merged / 2026-01-24
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 14 个文件，+723/-16，可读 patch 935 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NVIDIA] Add flashinfer all-to-all MOE dispatcher」；模型线: GLM-4.6/4.7；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py`, `python/sglang/srt/layers/moe/token_dispatcher/flashinfer_utils.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py`；技术摘要: 覆盖「[NVIDIA] Add flashinfer all-to-all MOE dispatcher」；主要实现面是 `python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py`, `python/sglang/srt/layers/moe/token_dispatcher/flashinfer_utils.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py` added +263/-0 (263 lines); hunks: -0,0 +1,263; symbols: FlashinferDispatchOutput, format, FlashinferCombineInput, FlashinferDispatcher，涉及 `FlashinferDispatchOutput, format, FlashinferCombineInput`；`python/sglang/srt/layers/moe/token_dispatcher/flashinfer_utils.py` added +47/-0 (47 lines); hunks: -0,0 +1,47; symbols: CommBackend, when, TorchDistributedCommBackend, __init__，涉及 `CommBackend, when, TorchDistributedCommBackend`；`python/sglang/srt/layers/quantization/modelopt_quant.py` modified +23/-14 (37 lines); hunks: -18,6 +18,7; -1479,6 +1480,7 @@ def _slice_scale(w):; symbols: _slice_scale, apply，涉及 `_slice_scale, apply`；`python/sglang/srt/layers/moe/token_dispatcher/base.py` modified +19/-0 (19 lines); hunks: -25,6 +25,8; -149,12 +151,19 @@ def format_is_deepep(; symbols: format_is_deepep, format_is_flashinfer, DispatchOutputFormat, is_standard，涉及 `format_is_deepep, format_is_flashinfer, DispatchOutputFormat`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py` added +263/-0 (263 lines); hunks: -0,0 +1,263; symbols: FlashinferDispatchOutput, format, FlashinferCombineInput, FlashinferDispatcher
  - `python/sglang/srt/layers/moe/token_dispatcher/flashinfer_utils.py` added +47/-0 (47 lines); hunks: -0,0 +1,47; symbols: CommBackend, when, TorchDistributedCommBackend, __init__
  - `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +23/-14 (37 lines); hunks: -18,6 +18,7; -1479,6 +1480,7 @@ def _slice_scale(w):; symbols: _slice_scale, apply
  - `python/sglang/srt/layers/moe/token_dispatcher/base.py` modified +19/-0 (19 lines); hunks: -25,6 +25,8; -149,12 +151,19 @@ def format_is_deepep(; symbols: format_is_deepep, format_is_flashinfer, DispatchOutputFormat, is_standard
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +9/-0 (9 lines); hunks: -37,6 +37,7; -117,6 +118,14 @@ def create_moe_dispatcher(moe_runner_config: MoeRunnerConfi...; symbols: create_moe_dispatcher
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py
@@ -0,0 +1,263 @@
+from __future__ import annotations
+import logging
+from typing import NamedTuple, Optional
+import torch
+from sglang.srt.environ import envs
+from sglang.srt.layers.dp_attention import get_dp_global_num_tokens
diff -- python/sglang/srt/layers/moe/token_dispatcher/flashinfer_utils.py
@@ -0,0 +1,47 @@
+import torch.distributed as dist
+from sglang.srt.utils import is_flashinfer_available
+if is_flashinfer_available():
+    from flashinfer.comm.mnnvl import CommBackend
+else:
+    class CommBackend:
diff -- python/sglang/srt/layers/quantization/modelopt_quant.py
@@ -18,6 +18,7 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py` added +263/-0; `python/sglang/srt/layers/moe/token_dispatcher/flashinfer_utils.py` added +47/-0; `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +23/-14; `python/sglang/srt/layers/moe/token_dispatcher/base.py` modified +19/-0; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +9/-0; `python/sglang/srt/layers/moe/token_dispatcher/__init__.py` modified +6/-0
- 验证与风险: diff 自带测试面 `python/sglang/test/test_flashinfer_dispatcher.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #17869 - [NPU]Support model GLM-4.7-Flash for npu, accuracy 81%

- 链接: https://github.com/sgl-project/sglang/pull/17869
- 状态/时间: open / 2026-01-28
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+86/-5，可读 patch 113 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU]Support model GLM-4.7-Flash for npu, accuracy 81%」；模型线: GLM-4.6/4.7；类别: 性能/后端优化；主要 diff: `test/registered/ascend/llm_models/test_ascend_glm4_7_flash.py`, `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`；技术摘要: 覆盖「[NPU]Support model GLM-4.7-Flash for npu, accuracy 81%」；主要实现面是 `test/registered/ascend/llm_models/test_ascend_glm4_7_flash.py`, `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/ascend/llm_models/test_ascend_glm4_7_flash.py` added +54/-0 (54 lines); hunks: -0,0 +1,54; symbols: TestGLM47Flash，涉及 `TestGLM47Flash`；`python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py` modified +30/-4 (34 lines); hunks: -1001,10 +1001,36 @@ def forward_extend(; symbols: forward_extend，涉及 `forward_extend`；`python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` modified +1/-1 (2 lines); hunks: -103,7 +103,7 @@ def forward_mha_prepare_npu(; symbols: forward_mha_prepare_npu，涉及 `forward_mha_prepare_npu`；`python/sglang/test/ascend/test_ascend_utils.py` modified +1/-0 (1 lines); hunks: -110,6 +110,7。
- 代码 diff 细节:
  - `test/registered/ascend/llm_models/test_ascend_glm4_7_flash.py` added +54/-0 (54 lines); hunks: -0,0 +1,54; symbols: TestGLM47Flash
  - `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py` modified +30/-4 (34 lines); hunks: -1001,10 +1001,36 @@ def forward_extend(; symbols: forward_extend
  - `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` modified +1/-1 (2 lines); hunks: -103,7 +103,7 @@ def forward_mha_prepare_npu(; symbols: forward_mha_prepare_npu
  - `python/sglang/test/ascend/test_ascend_utils.py` modified +1/-0 (1 lines); hunks: -110,6 +110,7
- 关键代码摘录:

```diff
diff -- test/registered/ascend/llm_models/test_ascend_glm4_7_flash.py
@@ -0,0 +1,54 @@
+import os
+import unittest
+from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
+from sglang.test.ascend.test_ascend_utils import GLM_4_7_FLASH_WEIGHTS_PATH
+from sglang.test.ci.ci_register import register_npu_ci
+from sglang.test.test_utils import CustomTestCase
diff -- python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py
@@ -1001,10 +1001,36 @@ def forward_extend(
-            assert (
-                layer.qk_head_dim != layer.v_head_dim
-            ), "FIA only supports qk_head_dim != v_head_dim"
-            if layer.v_head_dim in [256]:
+            if layer.qk_head_dim == layer.v_head_dim:
+                """FIA only supports qk_head_dim != v_head_dim"""
diff -- python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py
@@ -103,7 +103,7 @@ def forward_mha_prepare_npu(
```

- 已读文件:
  - tests: `test/registered/ascend/llm_models/test_ascend_glm4_7_flash.py` added +54/-0; `python/sglang/test/ascend/test_ascend_utils.py` modified +1/-0
  - runtime: `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py` modified +30/-4; `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` modified +1/-1
- 验证与风险: diff 自带测试面 `python/sglang/test/ascend/test_ascend_utils.py`, `test/registered/ascend/llm_models/test_ascend_glm4_7_flash.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #18383 - [Bug Fix] Add missing use_mla guard in aiter_backend draft_extend CUD…

- 链接: https://github.com/sgl-project/sglang/pull/18383
- 状态/时间: open / 2026-02-07
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bug Fix] Add missing use_mla guard in aiter_backend draft_extend CUD…」；模型线: GLM-4.6/4.7；类别: 缺陷修复；主要 diff: `python/sglang/srt/layers/attention/aiter_backend.py`；技术摘要: 覆盖「[Bug Fix] Add missing use_mla guard in aiter_backend draft_extend CUD…」；主要实现面是 `python/sglang/srt/layers/attention/aiter_backend.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/aiter_backend.py` modified +1/-1 (2 lines); hunks: -865,7 +865,7 @@ def init_forward_metadata_capture_cuda_graph(; symbols: init_forward_metadata_capture_cuda_graph，涉及 `init_forward_metadata_capture_cuda_graph`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/aiter_backend.py` modified +1/-1 (2 lines); hunks: -865,7 +865,7 @@ def init_forward_metadata_capture_cuda_graph(; symbols: init_forward_metadata_capture_cuda_graph
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/aiter_backend.py
@@ -865,7 +865,7 @@ def init_forward_metadata_capture_cuda_graph(
-            if _use_mla_ps_kernel:
+            if self.use_mla and _use_mla_ps_kernel:
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/aiter_backend.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/aiter_backend.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #18930 - [AMD] Unit tests for mtp in GLM-4.7

- 链接: https://github.com/sgl-project/sglang/pull/18930
- 状态/时间: open / 2026-02-17
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+120/-1，可读 patch 129 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Unit tests for mtp in GLM-4.7」；模型线: GLM-4.6/4.7；类别: 缺陷修复；主要 diff: `python/sglang/srt/layers/attention/aiter_backend.py`, `test/registered/amd/test_glm4v_fp8_mtp.py`；技术摘要: 覆盖「[AMD] Unit tests for mtp in GLM-4.7」；主要实现面是 `python/sglang/srt/layers/attention/aiter_backend.py`, `test/registered/amd/test_glm4v_fp8_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/aiter_backend.py` modified +2/-1 (3 lines); hunks: -999,7 +999,8 @@ def init_forward_metadata_capture_cuda_graph(; symbols: init_forward_metadata_capture_cuda_graph，涉及 `init_forward_metadata_capture_cuda_graph`；`test/registered/amd/test_glm4v_fp8_mtp.py` added +118/-0 (118 lines); hunks: -0,0 +1,118; symbols: TestGLM47FP8TPMTP, setUpClass, tearDownClass, test_a_gsm8k，涉及 `TestGLM47FP8TPMTP, setUpClass, tearDownClass`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/aiter_backend.py` modified +2/-1 (3 lines); hunks: -999,7 +999,8 @@ def init_forward_metadata_capture_cuda_graph(; symbols: init_forward_metadata_capture_cuda_graph
  - `test/registered/amd/test_glm4v_fp8_mtp.py` added +118/-0 (118 lines); hunks: -0,0 +1,118; symbols: TestGLM47FP8TPMTP, setUpClass, tearDownClass, test_a_gsm8k
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/aiter_backend.py
@@ -999,7 +999,8 @@ def init_forward_metadata_capture_cuda_graph(
-                if _use_mla_ps_kernel:
+                # https://github.com/sgl-project/sglang/pull/18383/changes
+                if self.use_mla and _use_mla_ps_kernel:
diff -- test/registered/amd/test_glm4v_fp8_mtp.py
@@ -0,0 +1,118 @@
+import unittest
+from types import SimpleNamespace
+import requests
+from sglang.srt.utils import kill_process_tree
+from sglang.test.ci.ci_register import register_amd_ci
+from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/aiter_backend.py` modified +2/-1
  - tests: `test/registered/amd/test_glm4v_fp8_mtp.py` added +118/-0
- 验证与风险: diff 自带测试面 `test/registered/amd/test_glm4v_fp8_mtp.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #19040 - feat: add Glm4MoeLiteConfig and fix enable_a2a_moe for GLM-4.7-Flash

- 链接: https://github.com/sgl-project/sglang/pull/19040
- 状态/时间: open / 2026-02-20
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+52/-0，可读 patch 88 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「feat: add Glm4MoeLiteConfig and fix enable_a2a_moe for GLM-4.7-Flash」；模型线: GLM-4.6/4.7；类别: 缺陷修复；主要 diff: `python/sglang/srt/configs/glm4_moe_lite.py`, `python/sglang/srt/configs/__init__.py`, `python/sglang/srt/models/glm4_moe_lite.py`；技术摘要: 覆盖「feat: add Glm4MoeLiteConfig and fix enable_a2a_moe for GLM-4.7-Flash」；主要实现面是 `python/sglang/srt/configs/glm4_moe_lite.py`, `python/sglang/srt/configs/__init__.py`, `python/sglang/srt/models/glm4_moe_lite.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/configs/glm4_moe_lite.py` added +47/-0 (47 lines); hunks: -0,0 +1,47; symbols: Glm4MoeLiteConfig, with, __init__，涉及 `Glm4MoeLiteConfig, with, __init__`；`python/sglang/srt/configs/__init__.py` modified +2/-0 (2 lines); hunks: -7,6 +7,7; -53,6 +54,7；`python/sglang/srt/models/glm4_moe_lite.py` modified +1/-0 (1 lines); hunks: -435,6 +435,7 @@ def __init__(; symbols: __init__，涉及 `__init__`；`python/sglang/srt/utils/hf_transformers_utils.py` modified +2/-0 (2 lines); hunks: -53,6 +53,7; -93,6 +94,7。
- 代码 diff 细节:
  - `python/sglang/srt/configs/glm4_moe_lite.py` added +47/-0 (47 lines); hunks: -0,0 +1,47; symbols: Glm4MoeLiteConfig, with, __init__
  - `python/sglang/srt/configs/__init__.py` modified +2/-0 (2 lines); hunks: -7,6 +7,7; -53,6 +54,7
  - `python/sglang/srt/models/glm4_moe_lite.py` modified +1/-0 (1 lines); hunks: -435,6 +435,7 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/utils/hf_transformers_utils.py` modified +2/-0 (2 lines); hunks: -53,6 +53,7; -93,6 +94,7
- 关键代码摘录:

```diff
diff -- python/sglang/srt/configs/glm4_moe_lite.py
@@ -0,0 +1,47 @@
+# Copyright 2025-2026 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/configs/__init__.py
@@ -7,6 +7,7 @@
+from sglang.srt.configs.glm4_moe_lite import Glm4MoeLiteConfig
@@ -53,6 +54,7 @@
+    "Glm4MoeLiteConfig",
diff -- python/sglang/srt/models/glm4_moe_lite.py
@@ -435,6 +435,7 @@ def __init__(
+        self.enable_a2a_moe = False  # Glm4MoeLite does not use all-to-all MoE dispatch
diff -- python/sglang/srt/utils/hf_transformers_utils.py
@@ -53,6 +53,7 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/configs/glm4_moe_lite.py` added +47/-0; `python/sglang/srt/configs/__init__.py` modified +2/-0; `python/sglang/srt/models/glm4_moe_lite.py` modified +1/-0; `python/sglang/srt/utils/hf_transformers_utils.py` modified +2/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/__init__.py`, `python/sglang/srt/configs/glm4_moe_lite.py`, `python/sglang/srt/models/glm4_moe_lite.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #19106 - Fix GLM4 MoE Lite CompressedTensors serving and transformers version checks

- 链接: https://github.com/sgl-project/sglang/pull/19106
- 状态/时间: open / 2026-02-21
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 12 个文件，+505/-37，可读 patch 677 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix GLM4 MoE Lite CompressedTensors serving and transformers version checks」；模型线: GLM-4.6/4.7；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/glm4_moe_lite.py`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`；技术摘要: 覆盖「Fix GLM4 MoE Lite CompressedTensors serving and transformers version checks」；主要实现面是 `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/glm4_moe_lite.py`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v2.py` modified +52/-27 (79 lines); hunks: -1275,40 +1275,66 @@ def __init__(; -2791,8 +2817,18 @@ def forward(; symbols: __init__, forward, DeepseekV2ForCausalLM，涉及 `__init__, forward, DeepseekV2ForCausalLM`；`python/sglang/srt/models/glm4_moe_lite.py` modified +52/-8 (60 lines); hunks: -132,16 +132,13 @@ def forward(; -467,6 +464,17 @@ def __init__(; symbols: forward, __init__, Glm4MoeLiteForCausalLM, determine_num_fused_shared_experts，涉及 `forward, __init__, Glm4MoeLiteForCausalLM`；`python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +54/-0 (54 lines); hunks: -35,6 +35,7; -93,6 +94,55 @@ class DeepseekV2WeightLoaderMixin:; symbols: DeepseekV2WeightLoaderMixin, _dequantize_ct_wna16_weight, do_load_weights, post_load_weights，涉及 `DeepseekV2WeightLoaderMixin, _dequantize_ct_wna16_weight, do_load_weights`；`python/sglang/srt/models/glm4_moe.py` modified +16/-0 (16 lines); hunks: -1001,6 +1001,13 @@ def forward(; -1047,6 +1054,15 @@ def determine_num_fused_shared_experts(self):; symbols: forward, Glm4MoeForCausalLM, __init__, determine_num_fused_shared_experts，涉及 `forward, Glm4MoeForCausalLM, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v2.py` modified +52/-27 (79 lines); hunks: -1275,40 +1275,66 @@ def __init__(; -2791,8 +2817,18 @@ def forward(; symbols: __init__, forward, DeepseekV2ForCausalLM
  - `python/sglang/srt/models/glm4_moe_lite.py` modified +52/-8 (60 lines); hunks: -132,16 +132,13 @@ def forward(; -467,6 +464,17 @@ def __init__(; symbols: forward, __init__, Glm4MoeLiteForCausalLM, determine_num_fused_shared_experts
  - `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +54/-0 (54 lines); hunks: -35,6 +35,7; -93,6 +94,55 @@ class DeepseekV2WeightLoaderMixin:; symbols: DeepseekV2WeightLoaderMixin, _dequantize_ct_wna16_weight, do_load_weights, post_load_weights
  - `python/sglang/srt/models/glm4_moe.py` modified +16/-0 (16 lines); hunks: -1001,6 +1001,13 @@ def forward(; -1047,6 +1054,15 @@ def determine_num_fused_shared_experts(self):; symbols: forward, Glm4MoeForCausalLM, __init__, determine_num_fused_shared_experts
  - `python/sglang/srt/configs/model_config.py` modified +14/-1 (15 lines); hunks: -1009,7 +1009,20 @@ def _verify_transformers_version(self):; symbols: _verify_transformers_version
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -1275,40 +1275,66 @@ def __init__(
-        # If we have self.fused_qkv_a_proj_with_mqa and we're running on CPU, we will choose the torch.ops.sgl_kernel.qkv_proj_with_rope_fused_weight kernel
-        # which requires self.w_kc and self.w_vc to be packed.
-        # If not, we will use torch.bmm and weight shouldn't be packed in this case
-        has_fused_proj = hasattr(self, "fused_qkv_a_proj_with_mqa")
+        # If we have self.fused_qkv_a_proj_with_mqa and we're running on CPU,
+        # we will choose the torch.ops.sgl_kernel.qkv_proj_with_rope_fused_weight
diff -- python/sglang/srt/models/glm4_moe_lite.py
@@ -132,16 +132,13 @@ def forward(
-        # Some quantization wrappers store the underlying parameter as `weight_packed`.
-        if not hasattr(self.gate_up_proj, "weight"):
-            self.gate_up_proj.weight = getattr(self.gate_up_proj, "weight_packed")
-        if not hasattr(self.down_proj, "weight"):
-            self.down_proj.weight = getattr(self.down_proj, "weight_packed")
+        gate_up_proj_weight = getattr(self.gate_up_proj, "weight", None)
diff -- python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py
@@ -35,6 +35,7 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +52/-27; `python/sglang/srt/models/glm4_moe_lite.py` modified +52/-8; `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +54/-0; `python/sglang/srt/models/glm4_moe.py` modified +16/-0; `python/sglang/srt/configs/model_config.py` modified +14/-1; `python/sglang/srt/models/deepseek_common/attention_backend_handler.py` modified +6/-1
  - tests: `test/registered/core/test_deepseek_weight_loader.py` added +86/-0; `test/registered/core/test_model_config_transformers_version.py` added +84/-0
- 验证与风险: diff 自带测试面 `test/registered/core/test_deepseek_attention_backend_handler.py`, `test/registered/core/test_deepseek_packed_modules_mapping.py`, `test/registered/core/test_deepseek_weight_loader.py`, `test/registered/core/test_glm4_moe_lite_shared_experts_fusion.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #21135 - fix: use get_rope_config() to support models without rope_parameters

- 链接: https://github.com/sgl-project/sglang/pull/21135
- 状态/时间: merged / 2026-03-26
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 18 个文件，+44/-42，可读 patch 342 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix: use get_rope_config() to support models without rope_parameters」；模型线: GLM-4.6/4.7；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4.py`, `python/sglang/srt/models/grok.py`；技术摘要: 覆盖「fix: use get_rope_config() to support models without rope_parameters」；主要实现面是 `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4.py`, `python/sglang/srt/models/grok.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/glm4_moe.py` modified +5/-5 (10 lines); hunks: -94,6 +94,7; -684,11 +685,10 @@ def __init__(; symbols: __init__，涉及 `__init__`；`python/sglang/srt/models/glm4.py` modified +5/-3 (8 lines); hunks: -52,6 +52,7; -217,9 +218,10 @@ def __init__(; symbols: __init__，涉及 `__init__`；`python/sglang/srt/models/grok.py` modified +2/-5 (7 lines); hunks: -61,6 +61,7; -477,11 +478,7 @@ def __init__(; symbols: __init__，涉及 `__init__`；`python/sglang/srt/models/llada2.py` modified +4/-2 (6 lines); hunks: -84,6 +84,7; -486,12 +487,13 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4_moe.py` modified +5/-5 (10 lines); hunks: -94,6 +94,7; -684,11 +685,10 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/glm4.py` modified +5/-3 (8 lines); hunks: -52,6 +52,7; -217,9 +218,10 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/grok.py` modified +2/-5 (7 lines); hunks: -61,6 +61,7; -477,11 +478,7 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/llada2.py` modified +4/-2 (6 lines); hunks: -84,6 +84,7; -486,12 +487,13 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/deepseek.py` modified +2/-2 (4 lines); hunks: -49,6 +49,7; -310,8 +311,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4_moe.py
@@ -94,6 +94,7 @@
+from sglang.srt.utils.hf_transformers_utils import get_rope_config
@@ -684,11 +685,10 @@ def __init__(
-        rope_theta = config.rope_parameters["rope_theta"]
-        rope_scaling = config.rope_parameters
-        partial_rotary_factor = getattr(
-            getattr(config, "rope_parameters", None), "partial_rotary_factor", None
diff -- python/sglang/srt/models/glm4.py
@@ -52,6 +52,7 @@
+from sglang.srt.utils.hf_transformers_utils import get_rope_config
@@ -217,9 +218,10 @@ def __init__(
-        rope_theta = config.rope_parameters["rope_theta"]
-        rope_scaling = config.rope_parameters
-        partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 0.5)
+        rope_theta, rope_scaling = get_rope_config(config)
diff -- python/sglang/srt/models/grok.py
@@ -61,6 +61,7 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4_moe.py` modified +5/-5; `python/sglang/srt/models/glm4.py` modified +5/-3; `python/sglang/srt/models/grok.py` modified +2/-5; `python/sglang/srt/models/llada2.py` modified +4/-2; `python/sglang/srt/models/deepseek.py` modified +2/-2; `python/sglang/srt/models/ernie4.py` modified +2/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/baichuan.py`, `python/sglang/srt/models/deepseek.py`, `python/sglang/srt/models/ernie4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21534 - [AMD] Add GLM-4.7-FP8 accuracy CI test for MI35x

- 链接: https://github.com/sgl-project/sglang/pull/21534
- 状态/时间: merged / 2026-03-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/amd/accuracy/mi35x/test_glm47_fp8_eval_mi35x.py`；关联提交 `7078e385ea13`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+96/-0，可读 patch 118 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Add GLM-4.7-FP8 accuracy CI test for MI35x」；模型线: GLM-4.6/4.7；类别: 性能/后端优化；主要 diff: `test/registered/amd/accuracy/mi35x/test_glm47_fp8_eval_mi35x.py`；技术摘要: 覆盖「[AMD] Add GLM-4.7-FP8 accuracy CI test for MI35x」；主要实现面是 `test/registered/amd/accuracy/mi35x/test_glm47_fp8_eval_mi35x.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/amd/accuracy/mi35x/test_glm47_fp8_eval_mi35x.py` added +61/-0 (61 lines); hunks: -0,0 +1,61; symbols: TestGLM47FP8EvalMI35x, test_glm_47_fp8，涉及 `TestGLM47FP8EvalMI35x, test_glm_47_fp8`。
- 代码 diff 细节:
  - `test/registered/amd/accuracy/mi35x/test_glm47_fp8_eval_mi35x.py` added +61/-0 (61 lines); hunks: -0,0 +1,61; symbols: TestGLM47FP8EvalMI35x, test_glm_47_fp8
- 关键代码摘录:

```diff
diff -- test/registered/amd/accuracy/mi35x/test_glm47_fp8_eval_mi35x.py
@@ -0,0 +1,61 @@
+"""MI35x GLM-4.7-FP8 GSM8K Accuracy Evaluation Test (8-GPU)
+Tests GLM-4.7-FP8 accuracy using GSM8K benchmark on MI35x.
+Registry: nightly-amd-8-gpu-mi35x-glm47-fp8 suite
+"""
+import os
+# Set HF cache for MI35x
```

- 已读文件:
  - tests: `test/registered/amd/accuracy/mi35x/test_glm47_fp8_eval_mi35x.py` added +61/-0
- 验证与风险: diff 自带测试面 `test/registered/amd/accuracy/mi35x/test_glm47_fp8_eval_mi35x.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #21660 - [GLM-V and GLM-4.7] Cast to FP32 before gate projection for GLM model.

- 链接: https://github.com/sgl-project/sglang/pull/21660
- 状态/时间: merged / 2026-03-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/glm4_moe.py`；关联提交 `ad064c2f4e33`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+6/-1，可读 patch 16 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[GLM-V and GLM-4.7] Cast to FP32 before gate projection for GLM model.」；模型线: GLM-4.6/4.7；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/glm4_moe.py`；技术摘要: 覆盖「[GLM-V and GLM-4.7] Cast to FP32 before gate projection for GLM model.」；主要实现面是 `python/sglang/srt/models/glm4_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/glm4_moe.py` modified +6/-1 (7 lines); hunks: -327,9 +327,14 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4_moe.py` modified +6/-1 (7 lines); hunks: -327,9 +327,14 @@ def __init__(; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4_moe.py
@@ -327,9 +327,14 @@ def __init__(
+        # GLM requires FP32 gate projection; cache to avoid per-forward cast.
+        # FIXME: if gate weight is updated at runtime (e.g. expert rebalancing), _weight_fp32 must be invalidated.
+        self.register_buffer("_weight_fp32", None, persistent=False)
-        logits = F.linear(hidden_states, self.weight, None)
+        if self._weight_fp32 is None:
+            self._weight_fp32 = self.weight.data.to(torch.float32)
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4_moe.py` modified +6/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/glm4_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #19246 - [NPU] optimize glm4.7

- 链接: https://github.com/sgl-project/sglang/pull/19246
- 状态/时间: merged / 2026-04-03
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+146/-15，可读 patch 259 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU] optimize glm4.7」；模型线: GLM-4.6/4.7；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_nextn.py`, `python/sglang/srt/layers/quantization/modelslim/modelslim.py`；技术摘要: 覆盖「[NPU] optimize glm4.7」；主要实现面是 `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_nextn.py`, `python/sglang/srt/layers/quantization/modelslim/modelslim.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/glm4_moe.py` modified +61/-11 (72 lines); hunks: -34,6 +34,7; -91,6 +92,7; symbols: Glm4MoeMLP, __init__, forward_prepare, forward_deepep，涉及 `Glm4MoeMLP, __init__, forward_prepare`；`python/sglang/srt/models/glm4_moe_nextn.py` modified +19/-2 (21 lines); hunks: -14,6 +14,7; -22,6 +23,7; symbols: __init__, forward，涉及 `__init__, forward`；`python/sglang/srt/layers/quantization/modelslim/modelslim.py` modified +2/-2 (4 lines); hunks: -58,13 +58,13 @@ def _rmsnorm_forward_oot(; symbols: _rmsnorm_forward_oot，涉及 `_rmsnorm_forward_oot`；`python/sglang/srt/hardware_backend/npu/utils.py` modified +64/-0 (64 lines); hunks: -178,3 +178,67 @@ def get_indexer_weight_stream():; symbols: get_indexer_weight_stream, get_share_stream, set_share_stream, get_routed_stream，涉及 `get_indexer_weight_stream, get_share_stream, set_share_stream`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4_moe.py` modified +61/-11 (72 lines); hunks: -34,6 +34,7; -91,6 +92,7; symbols: Glm4MoeMLP, __init__, forward_prepare, forward_deepep
  - `python/sglang/srt/models/glm4_moe_nextn.py` modified +19/-2 (21 lines); hunks: -14,6 +14,7; -22,6 +23,7; symbols: __init__, forward
  - `python/sglang/srt/layers/quantization/modelslim/modelslim.py` modified +2/-2 (4 lines); hunks: -58,13 +58,13 @@ def _rmsnorm_forward_oot(; symbols: _rmsnorm_forward_oot
  - `python/sglang/srt/hardware_backend/npu/utils.py` modified +64/-0 (64 lines); hunks: -178,3 +178,67 @@ def get_indexer_weight_stream():; symbols: get_indexer_weight_stream, get_share_stream, set_share_stream, get_routed_stream
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4_moe.py
@@ -34,6 +34,7 @@
+from sglang.srt.environ import envs
@@ -91,6 +92,7 @@
+    is_npu,
@@ -102,10 +104,19 @@
+_is_npu = is_npu()
+if _is_npu:
diff -- python/sglang/srt/models/glm4_moe_nextn.py
@@ -14,6 +14,7 @@
+import contextlib
@@ -22,6 +23,7 @@
+from sglang.srt.environ import temp_set_env
@@ -126,7 +128,10 @@ def __init__(
-        self.quant_config = quant_config
+        self.needs_quant_draft = (
diff -- python/sglang/srt/layers/quantization/modelslim/modelslim.py
@@ -58,13 +58,13 @@ def _rmsnorm_forward_oot(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4_moe.py` modified +61/-11; `python/sglang/srt/models/glm4_moe_nextn.py` modified +19/-2; `python/sglang/srt/layers/quantization/modelslim/modelslim.py` modified +2/-2; `python/sglang/srt/hardware_backend/npu/utils.py` modified +64/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/hardware_backend/npu/utils.py`, `python/sglang/srt/layers/quantization/modelslim/modelslim.py`, `python/sglang/srt/models/glm4_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21851 - GLM-4.7 and GLM-4.7-Flash Loading and import format

- 链接: https://github.com/sgl-project/sglang/pull/21851
- 状态/时间: merged / 2026-04-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_lite.py`；关联提交 `b7ae3b5a9a57`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+139/-86，可读 patch 486 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「GLM-4.7 and GLM-4.7-Flash Loading and import format」；模型线: GLM-4.6/4.7；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_lite.py`；技术摘要: 覆盖「GLM-4.7 and GLM-4.7-Flash Loading and import format」；主要实现面是 `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_lite.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/glm4_moe.py` modified +130/-57 (187 lines); hunks: -15,13 +15,15; -62,6 +64,7; symbols: __init__, get_moe_weights，涉及 `__init__, get_moe_weights`；`python/sglang/srt/models/glm4_moe_lite.py` modified +9/-29 (38 lines); hunks: -12,13 +12,15; -29,12 +31,14; symbols: forward, __init__, load_weights，涉及 `forward, __init__, load_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4_moe.py` modified +130/-57 (187 lines); hunks: -15,13 +15,15; -62,6 +64,7; symbols: __init__, get_moe_weights
  - `python/sglang/srt/models/glm4_moe_lite.py` modified +9/-29 (38 lines); hunks: -12,13 +12,15; -29,12 +31,14; symbols: forward, __init__, load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4_moe.py
@@ -15,13 +15,15 @@
+import re
+from sglang.srt.batch_overlap.single_batch_overlap import SboFlags
@@ -62,6 +64,7 @@
+from sglang.srt.layers.moe.kt_ep_wrapper import KTEPWrapperMethod
@@ -172,7 +175,7 @@ def __init__(
-        rope_theta: float = 10000,
diff -- python/sglang/srt/models/glm4_moe_lite.py
@@ -12,13 +12,15 @@
-"""Inference-only GLM-Lite model compatible with HuggingFace weights"""
+"""Inference-only GLM-4.7-Flash model compatible with HuggingFace weights"""
+import re
+from sgl_kernel import dsv3_router_gemm
@@ -29,12 +31,14 @@
+from sglang.srt.layers.attention.nsa.utils import is_nsa_enable_prefill_cp
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4_moe.py` modified +130/-57; `python/sglang/srt/models/glm4_moe_lite.py` modified +9/-29
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_lite.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22315 - [Bugfix] Fix GLM-4.7-FP8 EAGLE accept_len=1.00 due to draft model loading with incorrect quant_config

- 链接: https://github.com/sgl-project/sglang/pull/22315
- 状态/时间: open / 2026-04-08
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+7/-5，可读 patch 26 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix GLM-4.7-FP8 EAGLE accept_len=1.00 due to draft model loading with incorrect quant_config」；模型线: GLM-4.6/4.7；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/glm4_moe_nextn.py`；技术摘要: 覆盖「[Bugfix] Fix GLM-4.7-FP8 EAGLE accept_len=1.00 due to draft model loading with incorrect quant_config」；主要实现面是 `python/sglang/srt/models/glm4_moe_nextn.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/glm4_moe_nextn.py` modified +7/-5 (12 lines); hunks: -36,7 +36,7; -128,10 +128,12 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4_moe_nextn.py` modified +7/-5 (12 lines); hunks: -36,7 +36,7; -128,10 +128,12 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4_moe_nextn.py
@@ -36,7 +36,7 @@
-from sglang.srt.utils import add_prefix
+from sglang.srt.utils import add_prefix, is_npu
@@ -128,10 +128,12 @@ def __init__(
-        self.needs_quant_draft = (
-            get_global_server_args().speculative_draft_model_quantization
-        )
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4_moe_nextn.py` modified +7/-5
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/glm4_moe_nextn.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #20543 - fix: do not strip whitespace from GLM tool call values

- 链接: https://github.com/sgl-project/sglang/pull/20543
- 状态/时间: merged / 2026-04-09
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+66/-2，可读 patch 96 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix: do not strip whitespace from GLM tool call values」；模型线: GLM-4.6/4.7；类别: 缺陷修复；主要 diff: `test/registered/unit/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`；技术摘要: 覆盖「fix: do not strip whitespace from GLM tool call values」；主要实现面是 `test/registered/unit/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/glm47_moe_detector.py`, `python/sglang/srt/function_call/glm4_moe_detector.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/unit/function_call/test_function_call_parser.py` modified +66/-0 (66 lines); hunks: -2270,6 +2270,39 @@ def test_empty_function_name_handling(self):; -2546,6 +2579,39 @@ def check_single_todos(tool_result, expected):; symbols: test_empty_function_name_handling, test_whitespace_preserved_in_arg_values, TestGlm47MoeDetector, setUp，涉及 `test_empty_function_name_handling, test_whitespace_preserved_in_arg_values, TestGlm47MoeDetector`；`python/sglang/srt/function_call/glm47_moe_detector.py` modified +0/-1 (1 lines); hunks: -759,7 +759,6 @@ def _parse_argument_pairs(; symbols: _parse_argument_pairs，涉及 `_parse_argument_pairs`；`python/sglang/srt/function_call/glm4_moe_detector.py` modified +0/-1 (1 lines); hunks: -613,7 +613,6 @@ def _parse_argument_pairs(; symbols: _parse_argument_pairs，涉及 `_parse_argument_pairs`。
- 代码 diff 细节:
  - `test/registered/unit/function_call/test_function_call_parser.py` modified +66/-0 (66 lines); hunks: -2270,6 +2270,39 @@ def test_empty_function_name_handling(self):; -2546,6 +2579,39 @@ def check_single_todos(tool_result, expected):; symbols: test_empty_function_name_handling, test_whitespace_preserved_in_arg_values, TestGlm47MoeDetector, setUp
  - `python/sglang/srt/function_call/glm47_moe_detector.py` modified +0/-1 (1 lines); hunks: -759,7 +759,6 @@ def _parse_argument_pairs(; symbols: _parse_argument_pairs
  - `python/sglang/srt/function_call/glm4_moe_detector.py` modified +0/-1 (1 lines); hunks: -613,7 +613,6 @@ def _parse_argument_pairs(; symbols: _parse_argument_pairs
- 关键代码摘录:

```diff
diff -- test/registered/unit/function_call/test_function_call_parser.py
@@ -2270,6 +2270,39 @@ def test_empty_function_name_handling(self):
+    def test_whitespace_preserved_in_arg_values(self):
+        """Test that leading/trailing whitespace in arg values is not stripped."""
+        tools_with_string = [
+            Tool(
+                type="function",
+                function=Function(
diff -- python/sglang/srt/function_call/glm47_moe_detector.py
@@ -759,7 +759,6 @@ def _parse_argument_pairs(
-            arg_value = arg_value.strip()
diff -- python/sglang/srt/function_call/glm4_moe_detector.py
@@ -613,7 +613,6 @@ def _parse_argument_pairs(
-            arg_value = arg_value.strip()
```

- 已读文件:
  - tests: `test/registered/unit/function_call/test_function_call_parser.py` modified +66/-0
  - runtime: `python/sglang/srt/function_call/glm47_moe_detector.py` modified +0/-1; `python/sglang/srt/function_call/glm4_moe_detector.py` modified +0/-1
- 验证与风险: diff 自带测试面 `test/registered/unit/function_call/test_function_call_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #21403 - [AMD] Fuse RMSNorm + FP8 per-token quant for GLM-4.7-FP8

- 链接: https://github.com/sgl-project/sglang/pull/21403
- 状态/时间: merged / 2026-04-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/glm4_moe.py`；关联提交 `7e4e1dcd7ac8`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+149/-13，可读 patch 269 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Fuse RMSNorm + FP8 per-token quant for GLM-4.7-FP8」；模型线: GLM-4.6/4.7；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/glm4_moe.py`；技术摘要: 覆盖「[AMD] Fuse RMSNorm + FP8 per-token quant for GLM-4.7-FP8」；主要实现面是 `python/sglang/srt/models/glm4_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/glm4_moe.py` modified +58/-3 (61 lines); hunks: -289,7 +289,9 @@ def forward_prepare(; -865,6 +867,51 @@ def __init__(; symbols: forward_prepare, __init__, _detect_fp8_per_token_quant, _detect_attn_quant_format，涉及 `forward_prepare, __init__, _detect_fp8_per_token_quant`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4_moe.py` modified +58/-3 (61 lines); hunks: -289,7 +289,9 @@ def forward_prepare(; -865,6 +867,51 @@ def __init__(; symbols: forward_prepare, __init__, _detect_fp8_per_token_quant, _detect_attn_quant_format
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4_moe.py
@@ -289,7 +289,9 @@ def forward_prepare(
-        if hidden_states.shape[0] == 0:
+        # hidden_states can be a (fp8_tensor, scale) tuple from fused RMSNorm+Quant
+        hs = hidden_states[0] if isinstance(hidden_states, tuple) else hidden_states
+        if hs.shape[0] == 0:
@@ -865,6 +867,51 @@ def __init__(
+        # Detect if QKV uses aiter FP8 per-token quant so we can fuse
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4_moe.py` modified +58/-3
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/layers/quantization/fp8_utils.py`, `python/sglang/srt/models/glm4_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22720 - fix[glm4.7 flash]: properly detect `gfx95_quant_format`

- 链接: https://github.com/sgl-project/sglang/pull/22720
- 状态/时间: merged / 2026-04-13
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-0，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix[glm4.7 flash]: properly detect `gfx95_quant_format`」；模型线: GLM-4.6/4.7；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/glm4_moe_lite.py`；技术摘要: 覆盖「fix[glm4.7 flash]: properly detect `gfx95_quant_format`」；主要实现面是 `python/sglang/srt/models/glm4_moe_lite.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/glm4_moe_lite.py` modified +2/-0 (2 lines); hunks: -403,6 +403,8 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4_moe_lite.py` modified +2/-0 (2 lines); hunks: -403,6 +403,8 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4_moe_lite.py
@@ -403,6 +403,8 @@ def __init__(
+        self._gfx95_quant_format = self._detect_gfx95_quant_format()
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4_moe_lite.py` modified +2/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/glm4_moe_lite.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22801 - [NPU]add dual-stream and deepep support for GLM-4.7-Flash

- 链接: https://github.com/sgl-project/sglang/pull/22801
- 状态/时间: open / 2026-04-14
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+14/-3，可读 patch 52 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU]add dual-stream and deepep support for GLM-4.7-Flash」；模型线: GLM-4.6/4.7；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/glm4_moe_lite.py`, `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`；技术摘要: 覆盖「[NPU]add dual-stream and deepep support for GLM-4.7-Flash」；主要实现面是 `python/sglang/srt/models/glm4_moe_lite.py`, `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/glm4_moe_lite.py` modified +13/-2 (15 lines); hunks: -30,6 +30,7; -58,6 +59,7; symbols: __init__, forward，涉及 `__init__, forward`；`python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +1/-1 (2 lines); hunks: -609,7 +609,7 @@ def _dispatch_core(; symbols: _dispatch_core，涉及 `_dispatch_core`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4_moe_lite.py` modified +13/-2 (15 lines); hunks: -30,6 +30,7; -58,6 +59,7; symbols: __init__, forward
  - `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +1/-1 (2 lines); hunks: -609,7 +609,7 @@ def _dispatch_core(; symbols: _dispatch_core
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4_moe_lite.py
@@ -30,6 +30,7 @@
+from sglang.srt.environ import envs
@@ -58,6 +59,7 @@
+from sglang.srt.model_executor.forward_batch_info import ForwardBatch
@@ -178,7 +180,12 @@ def __init__(
-    def forward(self, hidden_states, gemm_output_zero_allocator: BumpAllocator = None):
+    def forward(
diff -- python/sglang/srt/layers/moe/token_dispatcher/deepep.py
@@ -609,7 +609,7 @@ def _dispatch_core(
-        else:
+        elif not envs.SGLANG_DEEPEP_BF16_DISPATCH.get():
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4_moe_lite.py` modified +13/-2; `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`, `python/sglang/srt/models/glm4_moe_lite.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22823 - [Bugfix] Preserve auto-detected quant_config for GLM NextN draft model

- 链接: https://github.com/sgl-project/sglang/pull/22823
- 状态/时间: merged / 2026-04-15
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-1，可读 patch 10 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Preserve auto-detected quant_config for GLM NextN draft model」；模型线: GLM-4.6/4.7；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/glm4_moe_nextn.py`；技术摘要: 覆盖「[Bugfix] Preserve auto-detected quant_config for GLM NextN draft model」；主要实现面是 `python/sglang/srt/models/glm4_moe_nextn.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/glm4_moe_nextn.py` modified +2/-1 (3 lines); hunks: -129,7 +129,8 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4_moe_nextn.py` modified +2/-1 (3 lines); hunks: -129,7 +129,8 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4_moe_nextn.py
@@ -129,7 +129,8 @@ def __init__(
-            get_global_server_args().speculative_draft_model_quantization
+            get_global_server_args().speculative_draft_model_quantization is not None
+            or quant_config is not None
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4_moe_nextn.py` modified +2/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/glm4_moe_nextn.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23067 - Fix: forward continue_final_message kwargs in Glm45Detector

- 链接: https://github.com/sgl-project/sglang/pull/23067
- 状态/时间: open / 2026-04-17
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+66/-1，可读 patch 94 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix: forward continue_final_message kwargs in Glm45Detector」；模型线: GLM-4.6/4.7；类别: 缺陷修复；主要 diff: `test/registered/unit/parser/test_reasoning_parser.py`, `python/sglang/srt/parser/reasoning_parser.py`；技术摘要: 覆盖「Fix: forward continue_final_message kwargs in Glm45Detector」；主要实现面是 `test/registered/unit/parser/test_reasoning_parser.py`, `python/sglang/srt/parser/reasoning_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/unit/parser/test_reasoning_parser.py` modified +57/-0 (57 lines); hunks: -518,6 +518,39 @@ def test_forced_reasoning_mode(self):; -1248,6 +1281,30 @@ def test_continue_final_message_with_request(self):; symbols: test_forced_reasoning_mode, test_continue_final_message_accepts_kwargs, test_continue_final_message_think_start_in_previous, test_continue_final_message_think_end_in_previous，涉及 `test_forced_reasoning_mode, test_continue_final_message_accepts_kwargs, test_continue_final_message_think_start_in_previous`；`python/sglang/srt/parser/reasoning_parser.py` modified +9/-1 (10 lines); hunks: -314,13 +314,21 @@ class Glm45Detector(BaseReasoningFormatDetector):; symbols: Glm45Detector, __init__，涉及 `Glm45Detector, __init__`。
- 代码 diff 细节:
  - `test/registered/unit/parser/test_reasoning_parser.py` modified +57/-0 (57 lines); hunks: -518,6 +518,39 @@ def test_forced_reasoning_mode(self):; -1248,6 +1281,30 @@ def test_continue_final_message_with_request(self):; symbols: test_forced_reasoning_mode, test_continue_final_message_accepts_kwargs, test_continue_final_message_think_start_in_previous, test_continue_final_message_think_end_in_previous
  - `python/sglang/srt/parser/reasoning_parser.py` modified +9/-1 (10 lines); hunks: -314,13 +314,21 @@ class Glm45Detector(BaseReasoningFormatDetector):; symbols: Glm45Detector, __init__
- 关键代码摘录:

```diff
diff -- test/registered/unit/parser/test_reasoning_parser.py
@@ -518,6 +518,39 @@ def test_forced_reasoning_mode(self):
+    def test_continue_final_message_accepts_kwargs(self):
+        """Regression: Glm45Detector must accept continue_final_message and
+        previous_content kwargs (forwarded by ReasoningParser when the request
+        sets continue_final_message=True with a trailing assistant message)."""
+        detector = Glm45Detector(
+            continue_final_message=True,
diff -- python/sglang/srt/parser/reasoning_parser.py
@@ -314,13 +314,21 @@ class Glm45Detector(BaseReasoningFormatDetector):
-    def __init__(self, stream_reasoning: bool = True, force_reasoning: bool = False):
+    def __init__(
+        self,
+        stream_reasoning: bool = True,
+        force_reasoning: bool = False,
+        continue_final_message: bool = False,
```

- 已读文件:
  - tests: `test/registered/unit/parser/test_reasoning_parser.py` modified +57/-0
  - runtime: `python/sglang/srt/parser/reasoning_parser.py` modified +9/-1
- 验证与风险: diff 自带测试面 `test/registered/unit/parser/test_reasoning_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22509 - [NPU]Fix GLM-4.7-Flash failed on NPU

- 链接: https://github.com/sgl-project/sglang/pull/22509
- 状态/时间: merged / 2026-04-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/glm4_moe_lite.py`；关联提交 `92f28e9ba80b`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+4/-2，可读 patch 27 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU]Fix GLM-4.7-Flash failed on NPU」；模型线: GLM-4.6/4.7；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/glm4_moe_lite.py`；技术摘要: 覆盖「[NPU]Fix GLM-4.7-Flash failed on NPU」；主要实现面是 `python/sglang/srt/models/glm4_moe_lite.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/glm4_moe_lite.py` modified +3/-1 (4 lines); hunks: -20,7 +20,6; -81,6 +80,9。
- 代码 diff 细节:
  - `python/sglang/srt/models/glm4_moe_lite.py` modified +3/-1 (4 lines); hunks: -20,7 +20,6; -81,6 +80,9
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/glm4_moe_lite.py
@@ -20,7 +20,6 @@
-from sgl_kernel import dsv3_router_gemm
@@ -81,6 +80,9 @@
+if _is_cuda:
+    from sgl_kernel import dsv3_router_gemm
```

- 已读文件:
  - runtime: `python/sglang/srt/models/glm4_moe_lite.py` modified +3/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/glm4_moe_lite.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
