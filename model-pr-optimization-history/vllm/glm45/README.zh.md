# vllm GLM-4.5 模型 PR 优化历史

## 2026-06-05 PR 补漏复核

已于 2026-06-05 按 vllm 上游 `origin/main@c66b19800` 复核；自上次时效基准（2025-11-17）以来，共有 39 个带 PR 编号的合并改动到所跟踪的实现文件，这些 PR 尚未并入下方时间线 / 逐 PR diff 审计卡，应在下次完整重生成时补齐。

| 合并日期 | PR | 标题 | 改动到的跟踪文件 |
| --- | --- | --- | --- |
| 2026-06-03 | [#44346](https://github.com/vllm-project/vllm/pull/44346) | [Refactor] Suppress SyntaxWarning from ast.literal_eval in tool parsers | `glm4_moe_tool_parser.py` |
| 2026-05-20 | [#39601](https://github.com/vllm-project/vllm/pull/39601) | [Bugfix] Fix glm4_moe_tool_parser._is_string_type for /v1/responses FunctionTool format | `test_glm4_moe_tool_parser.py`, `glm4_moe_tool_parser.py` |
| 2026-05-08 | [#42026](https://github.com/vllm-project/vllm/pull/42026) | [Bugfix] Preserve leading/trailing whitespace in GLM non-streaming tool parser | `test_glm4_moe_tool_parser.py`, `glm4_moe_tool_parser.py` |
| 2026-05-07 | [#41755](https://github.com/vllm-project/vllm/pull/41755) | [Bugfix] Fix GLM4-MoE weight loading for NVFP4 quantized checkpoints | `glm4_moe.py` |
| 2026-04-23 | [#40671](https://github.com/vllm-project/vllm/pull/40671) | [MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping | `glm4_moe.py`, `glm4_moe_lite.py`, `glm4_moe_lite_mtp.py`, … (+1) |
| 2026-04-21 | [#35782](https://github.com/vllm-project/vllm/pull/35782) | [MoE Refactor] Remove SharedFusedMoE class | `glm4_moe.py`, `glm4_moe_lite.py`, `glm4_moe_lite_mtp.py` |
| 2026-04-20 | [#35949](https://github.com/vllm-project/vllm/pull/35949) | [MoE Refactor] Move the shared/fused expert output sum into MoERunnerBase | `glm4_moe.py` |
| 2026-04-17 | [#39870](https://github.com/vllm-project/vllm/pull/39870) | [BugFix] Support custom tool parsers when tool_choice is `required` and named function | `glm4_moe_tool_parser.py` |
| 2026-04-13 | [#39253](https://github.com/vllm-project/vllm/pull/39253) | [Bugfix] Fix GLM tool parser streaming with MTP or stream interval | `test_glm4_moe_tool_parser.py`, `glm4_moe_tool_parser.py` |
| 2026-04-01 | [#38172](https://github.com/vllm-project/vllm/pull/38172) | [Misc] Add 20 regression tests for 11 tool parser bug fixes | `test_glm4_moe_tool_parser.py` |
| 2026-03-31 | [#38264](https://github.com/vllm-project/vllm/pull/38264) | [Mypy] Fix adjust_request typing | `glm4_moe_tool_parser.py` |
| 2026-03-31 | [#38189](https://github.com/vllm-project/vllm/pull/38189) | [Tool Parser][2/3] Use self.tools instead of request.tools in tool parsers | `test_glm4_moe_tool_parser.py`, `glm4_moe_tool_parser.py` |
| 2026-03-26 | [#38029](https://github.com/vllm-project/vllm/pull/38029) | [Tool Parser][1/3] Pass tools to ToolParser constructor | `glm4_moe_tool_parser.py` |
| 2026-03-18 | [#37386](https://github.com/vllm-project/vllm/pull/37386) | fix(glm47): improve tool call parsing and content normalization | `test_glm4_moe_tool_parser.py`, `glm4_moe_tool_parser.py` |
| 2026-03-16 | [#35208](https://github.com/vllm-project/vllm/pull/35208) | GLM4 tool parser: fix streaming mode | `test_glm4_moe_tool_parser.py`, `glm4_moe_tool_parser.py` |
| 2026-03-04 | [#35640](https://github.com/vllm-project/vllm/pull/35640) | [MISC] fixed tool_parser mypy errors | `glm4_moe_tool_parser.py` |
| 2026-02-24 | [#34905](https://github.com/vllm-project/vllm/pull/34905) | Fix GLM4 parser tests | `test_glm4_moe_tool_parser.py` |
| 2026-02-02 | [#33525](https://github.com/vllm-project/vllm/pull/33525) | Update get_expert_mapping to include self parameter | `glm4_moe_lite.py` |
| 2026-02-02 | [#33218](https://github.com/vllm-project/vllm/pull/33218) | [Bugfix] GLM-4 tool parser: incremental string streaming | `test_glm4_moe_tool_parser.py`, `glm4_moe_tool_parser.py` |
| 2026-01-27 | [#32064](https://github.com/vllm-project/vllm/pull/32064) | [5/N][Attention] Finish eliminating `vllm/attention` folder | `glm4_moe.py` |
| 2026-01-26 | [#33063](https://github.com/vllm-project/vllm/pull/33063) | [Chore] Update type annotation of `input_ids` in model forward | `glm4_moe.py`, `glm4_moe_lite.py`, `glm4_moe_lite_mtp.py`, … (+1) |
| 2026-01-19 | [#31386](https://github.com/vllm-project/vllm/pull/31386) | [GLM-4.7] GLM Model support for GLM-Lite | `glm4_moe_lite.py`, `glm4_moe_lite_mtp.py`, `glm4_moe_mtp.py` |
| 2026-01-15 | [#32321](https://github.com/vllm-project/vllm/pull/32321) | fix: avoid crash on zero-arg tool calls in glm4 parser | `glm4_moe_tool_parser.py` |
| 2026-01-13 | [#32240](https://github.com/vllm-project/vllm/pull/32240) | [Refactor] [6/N] to simplify the vLLM openai chat_completion serving architecture | `test_glm4_moe_tool_parser.py`, `glm4_moe_tool_parser.py` |
| 2026-01-12 | [#32150](https://github.com/vllm-project/vllm/pull/32150) | [Model] Remove incorrect `SupportsPP` from MTP models | `glm4_moe_mtp.py` |
| 2026-01-10 | [#32101](https://github.com/vllm-project/vllm/pull/32101) | [MTP][GLM][Bugfix] Fixed .weight_scale loading logic that dropped MTP prediction accuracy with fp8+mtp | `glm4_moe_mtp.py` |
| 2026-01-07 | [#31869](https://github.com/vllm-project/vllm/pull/31869) | [Model] Cleanup: Remove redundant manual definition of `make_empty_intermediate_tensors` in GLM-4-MoE | `glm4_moe.py` |
| 2026-01-07 | [#31757](https://github.com/vllm-project/vllm/pull/31757) | [Bugfix][MTP] Fix GLM4 MoE fp8 loading with MTP on | `glm4_moe_mtp.py` |
| 2026-01-07 | [#31104](https://github.com/vllm-project/vllm/pull/31104) | [BugFix] LoRA: Support loading base_layer of experts | `glm4_moe.py`, `glm4_moe_mtp.py` |
| 2026-01-06 | [#31055](https://github.com/vllm-project/vllm/pull/31055) | [Bugfix] Fix GLM-4 MoE router logits dtype for data parallel chunking | `glm4_moe.py` |
| 2026-01-05 | [#31622](https://github.com/vllm-project/vllm/pull/31622) | Fix GLM-4.6v flash tool calling in transformers 5.x | `glm4_moe_tool_parser.py` |
| 2025-12-20 | [#30876](https://github.com/vllm-project/vllm/pull/30876) | GLM-4.7 Tool Parser and Doc Update | `glm4_moe.py` |
| 2025-12-18 | [#30920](https://github.com/vllm-project/vllm/pull/30920) | [Bugfix] Fix Unicode issues in GLM-4 tool calling | `glm4_moe_tool_parser.py` |
| 2025-12-15 | [#30693](https://github.com/vllm-project/vllm/pull/30693) | [Refactor] [3/N] Move tool parser tests and run on CPU | `test_glm4_moe_tool_parser.py` |
| 2025-12-15 | [#30675](https://github.com/vllm-project/vllm/pull/30675) | [Refactor] [2/N] Move tool parsers into the vLLM main directory | `glm4_moe_tool_parser.py` |
| 2025-12-11 | [#30389](https://github.com/vllm-project/vllm/pull/30389) | Standardise `get_rope` to use `rope_parameters["partial_rotary_factor"]`, not `rotary_dim` | `glm4_moe.py` |
| 2025-12-04 | [#29966](https://github.com/vllm-project/vllm/pull/29966) | Access `partial_rotary_factor` from `rope_parameters` | `glm4_moe.py` |
| 2025-11-26 | [#29342](https://github.com/vllm-project/vllm/pull/29342) | [Attention] Remove imports from `vllm/attention/__init__.py` | `glm4_moe.py` |
| 2025-11-19 | [#28542](https://github.com/vllm-project/vllm/pull/28542) | Update `rope_scaling` to `rope_parameters` in preparation for Transformers v5 | `glm4_moe.py` |


## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `tests/reasoning/test_glm4_moe_reasoning_parser.py` | 无直接 PR 号提交 |
| `tests/tool_parsers/test_glm4_moe_tool_parser.py` | 无直接 PR 号提交 |
| `vllm/model_executor/models/glm4_moe.py` | [#21435](https://github.com/vllm-project/vllm/pull/21435), [#22143](https://github.com/vllm-project/vllm/pull/22143), [#22203](https://github.com/vllm-project/vllm/pull/22203), [#22460](https://github.com/vllm-project/vllm/pull/22460), [#22520](https://github.com/vllm-project/vllm/pull/22520), [#22832](https://github.com/vllm-project/vllm/pull/22832), [#24849](https://github.com/vllm-project/vllm/pull/24849), [#25830](https://github.com/vllm-project/vllm/pull/25830) |
| `vllm/model_executor/models/glm4_moe_lite.py` | 无直接 PR 号提交 |
| `vllm/model_executor/models/glm4_moe_lite_mtp.py` | 无直接 PR 号提交 |
| `vllm/model_executor/models/glm4_moe_mtp.py` | [#27597](https://github.com/vllm-project/vllm/pull/27597), [#28805](https://github.com/vllm-project/vllm/pull/28805) |
| `vllm/tool_parsers/glm4_moe_tool_parser.py` | 无直接 PR 号提交 |

## PR 覆盖总览

- git 追溯 PR 数: 10
- 原文档显式引用补充 PR 数: 3
- 当前文档总 PR 数: 13
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-07-24 | [#21435](https://github.com/vllm-project/vllm/pull/21435) | merged | remove GLM-4 quantization wrong Code | `vllm/model_executor/models/glm4_moe.py` |
| 2025-08-03 | [#22143](https://github.com/vllm-project/vllm/pull/22143) | merged | fuse fp32 for GLM-4.5 e_score_correction_bias | `vllm/model_executor/models/glm4_moe.py` |
| 2025-08-04 | [#22171](https://github.com/vllm-project/vllm/pull/22171) | merged | [Misc] Modify the organization of GLM series | `docs/models/supported_models.md`, `tests/models/registry.py`, `tests/models/multimodal/generation/test_common.py` |
| 2025-08-05 | [#22203](https://github.com/vllm-project/vllm/pull/22203) | merged | self.gate dtype update for GLM-4.5 | `vllm/model_executor/models/glm4_moe.py` |
| 2025-08-08 | [#22460](https://github.com/vllm-project/vllm/pull/22460) | merged | not tie_word_embeddings for glm-4.5 and glm-4.5v | `vllm/model_executor/models/glm4_moe.py` |
| 2025-08-09 | [#22520](https://github.com/vllm-project/vllm/pull/22520) | merged | GLM-4.5V with new class name at transformers | `vllm/model_executor/models/glm4_moe.py` |
| 2025-08-14 | [#22832](https://github.com/vllm-project/vllm/pull/22832) | merged | [Model] Modify the gate implementation of glm4_moe | `vllm/model_executor/models/glm4_moe.py` |
| 2025-08-27 | [#23695](https://github.com/vllm-project/vllm/pull/23695) | merged | feat: add triton fused moe config for GLM-4.5-Air-FP8 on B200 | `vllm/model_executor/layers/fused_moe/configs/E=128,N=704,device_name=NVIDIA_B200,dtype=fp8_w8a8.json` |
| 2025-09-10 | [#24589](https://github.com/vllm-project/vllm/pull/24589) | merged | [Doc] Add documentation for GLM-4.5 series models: tool-calling and reasoning parser | `docs/features/reasoning_outputs.md`, `docs/features/tool_calling.md` |
| 2025-09-17 | [#24849](https://github.com/vllm-project/vllm/pull/24849) | merged | [Model] Apply SharedFusedMoE to glm4_moe. | `vllm/model_executor/models/glm4_moe.py` |
| 2025-09-28 | [#25830](https://github.com/vllm-project/vllm/pull/25830) | merged | Update GLM-4.5 Doc transformers version | `vllm/model_executor/models/glm4_moe.py` |
| 2025-11-12 | [#27597](https://github.com/vllm-project/vllm/pull/27597) | merged | [Model] fix glm4_moe_mtp load weights with GLM-4.6 checkpoint. | `vllm/model_executor/models/glm4_moe_mtp.py` |
| 2025-11-17 | [#28805](https://github.com/vllm-project/vllm/pull/28805) | merged | [BugFix] Fix glm4_moe_mtp load weights bug | `vllm/model_executor/models/glm4_moe_mtp.py` |

## 逐 PR diff 审计卡

### PR #21435 - remove GLM-4 quantization wrong Code

- 链接: https://github.com/vllm-project/vllm/pull/21435
- 状态/时间: merged / 2025-07-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/glm4_moe.py`；关联提交 `85bda9e7d053`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+2/-3，可读 patch 26 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「remove GLM-4 quantization wrong Code」；模型线: GLM-4.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/glm4_moe.py`；技术摘要: 覆盖「remove GLM-4 quantization wrong Code」；主要实现面是 `vllm/model_executor/models/glm4_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/glm4_moe.py` modified +0/-1 (1 lines); hunks: -390,7 +390,6 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/glm4_moe.py` modified +0/-1 (1 lines); hunks: -390,7 +390,6 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/glm4_moe.py
@@ -390,7 +390,6 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
-                quant_config=quant_config,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/glm4_moe.py` modified +0/-1
- 验证与风险: runtime 路径改动集中在 `vllm/entrypoints/openai/tool_parsers/glm4_moe_tool_parser.py`, `vllm/model_executor/models/glm4_moe.py`, `vllm/reasoning/glm4_moe_reasoning_parser.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22143 - fuse fp32 for GLM-4.5 e_score_correction_bias

- 链接: https://github.com/vllm-project/vllm/pull/22143
- 状态/时间: merged / 2025-08-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/glm4_moe.py`；关联提交 `d3c18c9cb0b6`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-3，可读 patch 12 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fuse fp32 for GLM-4.5 e_score_correction_bias」；模型线: GLM-4.5；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/glm4_moe.py`；未提供可用技术摘要。
- 实现要点: `vllm/model_executor/models/glm4_moe.py` modified +2/-3 (5 lines); hunks: -125,9 +125,8 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/glm4_moe.py` modified +2/-3 (5 lines); hunks: -125,9 +125,8 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/glm4_moe.py
@@ -125,9 +125,8 @@ def __init__(
-        # noaux_tc is not set in transformers new config now
-        self.gate.e_score_correction_bias = (nn.Parameter(
-            torch.empty(config.n_routed_experts)))
+        self.gate.e_score_correction_bias = nn.Parameter(
+            torch.empty(config.n_routed_experts, dtype=torch.float32))
```

- 已读文件:
  - runtime: `vllm/model_executor/models/glm4_moe.py` modified +2/-3
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/glm4_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22171 - [Misc] Modify the organization of GLM series

- 链接: https://github.com/vllm-project/vllm/pull/22171
- 状态/时间: merged / 2025-08-04
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 16 个文件，+31/-31，可读 patch 241 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Misc] Modify the organization of GLM series」；模型线: GLM-4.5；类别: 模型实现调整；主要 diff: `docs/models/supported_models.md`, `tests/models/registry.py`, `tests/models/multimodal/generation/test_common.py`；未提供可用技术摘要。
- 实现要点: `docs/models/supported_models.md` modified +5/-5 (10 lines); hunks: -328,7 +328,7 @@ th {; -348,8 +348,8 @@ th {；`tests/models/registry.py` modified +5/-5 (10 lines); hunks: -153,7 +153,7 @@ def check_available_online(; -187,8 +187,8 @@ def check_available_online(; symbols: check_available_online，涉及 `check_available_online`；`tests/models/multimodal/generation/test_common.py` modified +3/-3 (6 lines); hunks: -355,7 +355,7; -374,7 +374,7；`vllm/model_executor/models/chatglm.py` modified +3/-3 (6 lines); hunks: -1,7 +1,7; -86,10 +86,10 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `docs/models/supported_models.md` modified +5/-5 (10 lines); hunks: -328,7 +328,7 @@ th {; -348,8 +348,8 @@ th {
  - `tests/models/registry.py` modified +5/-5 (10 lines); hunks: -153,7 +153,7 @@ def check_available_online(; -187,8 +187,8 @@ def check_available_online(; symbols: check_available_online
  - `tests/models/multimodal/generation/test_common.py` modified +3/-3 (6 lines); hunks: -355,7 +355,7; -374,7 +374,7
  - `vllm/model_executor/models/chatglm.py` modified +3/-3 (6 lines); hunks: -1,7 +1,7; -86,10 +86,10 @@ def __init__(; symbols: __init__
  - `tests/models/multimodal/processing/test_common.py` modified +2/-2 (4 lines); hunks: -271,8 +271,8 @@ def _test_processing_correctness_one(; symbols: _test_processing_correctness_one
- 关键代码摘录:

```diff
diff -- docs/models/supported_models.md
@@ -328,7 +328,7 @@ th {
-| `ChatGLMModel`, `ChatGLMForConditionalGeneration` | ChatGLM | `THUDM/chatglm2-6b`, `THUDM/chatglm3-6b`, `ShieldLM-6B-chatglm3`, etc. | ✅︎ | ✅︎ | ✅︎ |
+| `ChatGLMModel`, `ChatGLMForConditionalGeneration` | ChatGLM | `zai-org/chatglm2-6b`, `zai-org/chatglm3-6b`, `ShieldLM-6B-chatglm3`, etc. | ✅︎ | ✅︎ | ✅︎ |
@@ -348,8 +348,8 @@ th {
-| `GlmForCausalLM` | GLM-4 | `THUDM/glm-4-9b-chat-hf`, etc. | ✅︎ | ✅︎ | ✅︎ |
-| `Glm4ForCausalLM` | GLM-4-0414 | `THUDM/GLM-4-32B-0414`, etc. | ✅︎ | ✅︎ | ✅︎ |
+| `GlmForCausalLM` | GLM-4 | `zai-org/glm-4-9b-chat-hf`, etc. | ✅︎ | ✅︎ | ✅︎ |
diff -- tests/models/registry.py
@@ -153,7 +153,7 @@ def check_available_online(
-    "ChatGLMModel": _HfExamplesInfo("THUDM/chatglm3-6b",
+    "ChatGLMModel": _HfExamplesInfo("zai-org/chatglm3-6b",
@@ -187,8 +187,8 @@ def check_available_online(
-    "GlmForCausalLM": _HfExamplesInfo("THUDM/glm-4-9b-chat-hf"),
-    "Glm4ForCausalLM": _HfExamplesInfo("THUDM/GLM-4-9B-0414"),
+    "GlmForCausalLM": _HfExamplesInfo("zai-org/glm-4-9b-chat-hf"),
diff -- tests/models/multimodal/generation/test_common.py
@@ -355,7 +355,7 @@
```

- 已读文件:
  - docs: `docs/models/supported_models.md` modified +5/-5
  - tests: `tests/models/registry.py` modified +5/-5; `tests/models/multimodal/generation/test_common.py` modified +3/-3; `tests/models/multimodal/processing/test_common.py` modified +2/-2; `tests/models/language/generation/test_common.py` modified +1/-1; `tests/models/multimodal/processing/test_glm4_1v.py` modified +1/-1; `tests/tokenization/test_cached_tokenizer.py` modified +1/-1
  - runtime: `vllm/model_executor/models/chatglm.py` modified +3/-3
- 验证与风险: diff 自带测试面 `tests/distributed/test_pipeline_parallel.py`, `tests/lora/test_add_lora.py`, `tests/lora/test_chatglm3_tp.py`, `tests/models/language/generation/test_common.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22203 - self.gate dtype update for GLM-4.5

- 链接: https://github.com/vllm-project/vllm/pull/22203
- 状态/时间: merged / 2025-08-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/glm4_moe.py`；关联提交 `6fa41e0c32f3`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+4/-3，可读 patch 35 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「self.gate dtype update for GLM-4.5」；模型线: GLM-4.5；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/glm4_moe.py`；技术摘要: 覆盖「self.gate dtype update for GLM-4.5」；主要实现面是 `vllm/model_executor/models/glm4_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/glm4_moe.py` modified +2/-1 (3 lines); hunks: -123,6 +123,7 @@ def __init__(; -180,7 +181,7 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/glm4_moe.py` modified +2/-1 (3 lines); hunks: -123,6 +123,7 @@ def __init__(; -180,7 +181,7 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/glm4_moe.py
@@ -123,6 +123,7 @@ def __init__(
+                                     params_dtype=torch.float32,
@@ -180,7 +181,7 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
-        router_logits, _ = self.gate(hidden_states)
+        router_logits, _ = self.gate(hidden_states.to(dtype=torch.float32))
```

- 已读文件:
  - runtime: `vllm/model_executor/models/glm4_moe.py` modified +2/-1
- 验证与风险: diff 自带测试面 `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22460 - not tie_word_embeddings for glm-4.5 and glm-4.5v

- 链接: https://github.com/vllm-project/vllm/pull/22460
- 状态/时间: merged / 2025-08-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/glm4_moe.py`；关联提交 `c152e2a8a0f4`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+0/-2，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「not tie_word_embeddings for glm-4.5 and glm-4.5v」；模型线: GLM-4.5；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/glm4_moe.py`；未提供可用技术摘要。
- 实现要点: `vllm/model_executor/models/glm4_moe.py` modified +0/-2 (2 lines); hunks: -601,8 +601,6 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/glm4_moe.py` modified +0/-2 (2 lines); hunks: -601,8 +601,6 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/glm4_moe.py
@@ -601,8 +601,6 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
-        if self.config.tie_word_embeddings:
-            self.lm_head.weight = self.model.embed_tokens.weight
```

- 已读文件:
  - runtime: `vllm/model_executor/models/glm4_moe.py` modified +0/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/glm4_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22520 - GLM-4.5V with new class name at transformers

- 链接: https://github.com/vllm-project/vllm/pull/22520
- 状态/时间: merged / 2025-08-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/glm4_moe.py`；关联提交 `a6022e6fbcbd`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+13/-6，可读 patch 61 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「GLM-4.5V with new class name at transformers」；模型线: GLM-4.5；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/glm4_moe.py`；技术摘要: 覆盖「GLM-4.5V with new class name at transformers」；主要实现面是 `vllm/model_executor/models/glm4_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/glm4_moe.py` modified +7/-1 (8 lines); hunks: -372,7 +372,13 @@ def forward(; symbols: forward, Glm4MoeModel, __init__，涉及 `forward, Glm4MoeModel, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/glm4_moe.py` modified +7/-1 (8 lines); hunks: -372,7 +372,13 @@ def forward(; symbols: forward, Glm4MoeModel, __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/glm4_moe.py
@@ -372,7 +372,13 @@ def forward(
-@support_torch_compile
+@support_torch_compile(
+    dynamic_arg_dims={
+        "input_ids": 0,
+        "positions": -1,
+        "intermediate_tensors": 0,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/glm4_moe.py` modified +7/-1
- 验证与风险: diff 自带测试面 `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22832 - [Model] Modify the gate implementation of glm4_moe

- 链接: https://github.com/vllm-project/vllm/pull/22832
- 状态/时间: merged / 2025-08-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/glm4_moe.py`；关联提交 `92ff41abea13`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+11/-11，可读 patch 50 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Modify the gate implementation of glm4_moe」；模型线: GLM-4.5；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/glm4_moe.py`；未提供可用技术摘要。
- 实现要点: `vllm/model_executor/models/glm4_moe.py` modified +10/-10 (20 lines); hunks: -41,7 +41,6; -118,14 +117,15 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/glm4_moe.py` modified +10/-10 (20 lines); hunks: -41,7 +41,6; -118,14 +117,15 @@ def __init__(; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/glm4_moe.py
@@ -41,7 +41,6 @@
-                                               ReplicatedLinear,
@@ -118,14 +117,15 @@ def __init__(
-        self.gate = ReplicatedLinear(config.hidden_size,
-                                     config.n_routed_experts,
-                                     bias=False,
-                                     quant_config=None,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/glm4_moe.py` modified +10/-10
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/glm4_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23695 - feat: add triton fused moe config for GLM-4.5-Air-FP8 on B200

- 链接: https://github.com/vllm-project/vllm/pull/23695
- 状态/时间: merged / 2025-08-27
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+146/-0，可读 patch 147 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「feat: add triton fused moe config for GLM-4.5-Air-FP8 on B200」；模型线: GLM-4.5；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/fused_moe/configs/E=128,N=704,device_name=NVIDIA_B200,dtype=fp8_w8a8.json`；技术摘要: 覆盖「feat: add triton fused moe config for GLM-4.5-Air-FP8 on B200」；主要实现面是 `vllm/model_executor/layers/fused_moe/configs/E=128,N=704,device_name=NVIDIA_B200,dtype=fp8_w8a8.json`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/fused_moe/configs/E=128,N=704,device_name=NVIDIA_B200,dtype=fp8_w8a8.json` added +146/-0 (146 lines); hunks: -0,0 +1,146。
- 代码 diff 细节:
  - `vllm/model_executor/layers/fused_moe/configs/E=128,N=704,device_name=NVIDIA_B200,dtype=fp8_w8a8.json` added +146/-0 (146 lines); hunks: -0,0 +1,146
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/fused_moe/configs/E=128,N=704,device_name=NVIDIA_B200,dtype=fp8_w8a8.json
@@ -0,0 +1,146 @@
+{
+    "1": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/fused_moe/configs/E=128,N=704,device_name=NVIDIA_B200,dtype=fp8_w8a8.json` added +146/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/fused_moe/configs/E=128,N=704,device_name=NVIDIA_B200,dtype=fp8_w8a8.json`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #24589 - [Doc] Add documentation for GLM-4.5 series models: tool-calling and reasoning parser

- 链接: https://github.com/vllm-project/vllm/pull/24589
- 状态/时间: merged / 2025-09-10
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+10/-0，可读 patch 24 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Doc] Add documentation for GLM-4.5 series models: tool-calling and reasoning parser」；模型线: GLM-4.5；类别: 文档/测试/CI；主要 diff: `docs/features/reasoning_outputs.md`, `docs/features/tool_calling.md`；技术摘要: 覆盖「[Doc] Add documentation for GLM-4.5 series models: tool-calling and reasoning parser」；主要实现面是 `docs/features/reasoning_outputs.md`, `docs/features/tool_calling.md`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs/features/reasoning_outputs.md` modified +1/-0 (1 lines); hunks: -15,6 +15,7 @@ vLLM currently supports the following reasoning models:；`docs/features/tool_calling.md` modified +9/-0 (9 lines); hunks: -311,6 +311,15 @@ Flags:。
- 代码 diff 细节:
  - `docs/features/reasoning_outputs.md` modified +1/-0 (1 lines); hunks: -15,6 +15,7 @@ vLLM currently supports the following reasoning models:
  - `docs/features/tool_calling.md` modified +9/-0 (9 lines); hunks: -311,6 +311,15 @@ Flags:
- 关键代码摘录:

```diff
diff -- docs/features/reasoning_outputs.md
@@ -15,6 +15,7 @@ vLLM currently supports the following reasoning models:
+| [GLM-4.5 series](https://huggingface.co/collections/zai-org/glm-45-687c621d34bda8c9e4bf503b) | `glm45` | `guided_json`, `guided_regex` | ✅ |
diff -- docs/features/tool_calling.md
@@ -311,6 +311,15 @@ Flags:
+### GLM-4.5 Models (`glm45`)
+Supported models:
+* `ZhipuAI/GLM-4.5`
+* `ZhipuAI/GLM-4.5-Air`
+Flags: `--tool-call-parser glm45`
```

- 已读文件:
  - docs: `docs/features/reasoning_outputs.md` modified +1/-0; `docs/features/tool_calling.md` modified +9/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs/features/reasoning_outputs.md`, `docs/features/tool_calling.md`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #24849 - [Model] Apply SharedFusedMoE to glm4_moe.

- 链接: https://github.com/vllm-project/vllm/pull/24849
- 状态/时间: merged / 2025-09-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/glm4_moe.py`；关联提交 `c15309a730fa`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+55/-30，可读 patch 114 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Apply SharedFusedMoE to glm4_moe.」；模型线: GLM-4.5；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/glm4_moe.py`；技术摘要: 覆盖「[Model] Apply SharedFusedMoE to glm4_moe.」；主要实现面是 `vllm/model_executor/models/glm4_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/glm4_moe.py` modified +55/-30 (85 lines); hunks: -46,6 +46,7; -146,25 +147,6 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/glm4_moe.py` modified +55/-30 (85 lines); hunks: -46,6 +46,7; -146,25 +147,6 @@ def __init__(; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/glm4_moe.py
@@ -46,6 +46,7 @@
+from vllm.model_executor.layers.shared_fused_moe import SharedFusedMoE
@@ -146,25 +147,6 @@ def __init__(
-        self.experts = FusedMoE(
-            num_experts=config.n_routed_experts,
-            top_k=config.num_experts_per_tok,
-            hidden_size=config.hidden_size,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/glm4_moe.py` modified +55/-30
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/glm4_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #25830 - Update GLM-4.5 Doc transformers version

- 链接: https://github.com/vllm-project/vllm/pull/25830
- 状态/时间: merged / 2025-09-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/glm4_moe.py`；关联提交 `b1ded114b976`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+7/-5，可读 patch 40 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Update GLM-4.5 Doc transformers version」；模型线: GLM-4.5；类别: 文档/测试/CI；主要 diff: `vllm/model_executor/models/glm4_moe.py`；技术摘要: 覆盖「Update GLM-4.5 Doc transformers version」；主要实现面是 `vllm/model_executor/models/glm4_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/glm4_moe.py` modified +1/-1 (2 lines); hunks: -21,7 +21,7。
- 代码 diff 细节:
  - `vllm/model_executor/models/glm4_moe.py` modified +1/-1 (2 lines); hunks: -21,7 +21,7
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/glm4_moe.py
@@ -21,7 +21,7 @@
-"""Inference-only GLM-4.5 model compatible with HuggingFace weights."""
+"""Inference-only GLM-4.5, GLM-4.6 model compatible with HuggingFace weights."""
```

- 已读文件:
  - runtime: `vllm/model_executor/models/glm4_moe.py` modified +1/-1
- 验证与风险: diff 自带测试面 `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #27597 - [Model] fix glm4_moe_mtp load weights with GLM-4.6 checkpoint.

- 链接: https://github.com/vllm-project/vllm/pull/27597
- 状态/时间: merged / 2025-11-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/glm4_moe_mtp.py`；关联提交 `d3ade61e429f`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+11/-4，可读 patch 23 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] fix glm4_moe_mtp load weights with GLM-4.6 checkpoint.」；模型线: GLM-4.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/glm4_moe_mtp.py`；技术摘要: 覆盖「[Model] fix glm4_moe_mtp load weights with GLM-4.6 checkpoint.」；主要实现面是 `vllm/model_executor/models/glm4_moe_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/glm4_moe_mtp.py` modified +11/-4 (15 lines); hunks: -256,11 +256,18 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/models/glm4_moe_mtp.py` modified +11/-4 (15 lines); hunks: -256,11 +256,18 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: load_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/glm4_moe_mtp.py
@@ -256,11 +256,18 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
+        spec_layer = self.model.mtp_start_layer_idx
-            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
-            if spec_layer is None:
-                continue
-            name = self._rewrite_spec_layer_name(spec_layer, name)
+            if name == "lm_head.weight":
```

- 已读文件:
  - runtime: `vllm/model_executor/models/glm4_moe_mtp.py` modified +11/-4
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/glm4_moe_mtp.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #28805 - [BugFix] Fix glm4_moe_mtp load weights bug

- 链接: https://github.com/vllm-project/vllm/pull/28805
- 状态/时间: merged / 2025-11-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/glm4_moe_mtp.py`；关联提交 `ab01cd14e5e2`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-4，可读 patch 17 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[BugFix] Fix glm4_moe_mtp load weights bug」；模型线: GLM-4.5；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/glm4_moe_mtp.py`；技术摘要: 覆盖「[BugFix] Fix glm4_moe_mtp load weights bug」；主要实现面是 `vllm/model_executor/models/glm4_moe_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/glm4_moe_mtp.py` modified +3/-4 (7 lines); hunks: -256,13 +256,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/models/glm4_moe_mtp.py` modified +3/-4 (7 lines); hunks: -256,13 +256,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: load_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/glm4_moe_mtp.py
@@ -256,13 +256,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
-        spec_layer = self.model.mtp_start_layer_idx
-                name = f"model.layers.{spec_layer}.shard_head.head.weight"
+                spec_layer = self.model.mtp_start_layer_idx
+                name = f"model.layers.{spec_layer}.shared_head.head.weight"
-                # This name is same with local model, rewriting is not needed.
-                pass
```

- 已读文件:
  - runtime: `vllm/model_executor/models/glm4_moe_mtp.py` modified +3/-4
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/glm4_moe_mtp.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
