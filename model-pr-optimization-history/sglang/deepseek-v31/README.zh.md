# sglang DeepSeek V3.1 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `examples/chat_template/tool_chat_template_deepseekv31.jinja` | [#9446](https://github.com/sgl-project/sglang/pull/9446), [#9895](https://github.com/sgl-project/sglang/pull/9895), [#14837](https://github.com/sgl-project/sglang/pull/14837) |
| `python/sglang/srt/function_call/deepseekv31_detector.py` | [#9446](https://github.com/sgl-project/sglang/pull/9446), [#11589](https://github.com/sgl-project/sglang/pull/11589), [#13394](https://github.com/sgl-project/sglang/pull/13394) |
| `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` | 无直接 PR 号提交 |
| `python/sglang/srt/models/deepseek_common/__init__.py` | 无直接 PR 号提交 |
| `python/sglang/srt/models/deepseek_common/attention_backend_handler.py` | 无直接 PR 号提交 |
| `python/sglang/srt/models/deepseek_common/attention_forward_methods/__init__.py` | 无直接 PR 号提交 |
| `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_methods.py` | 无直接 PR 号提交 |
| `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` | 无直接 PR 号提交 |
| `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` | 无直接 PR 号提交 |
| `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_cpu.py` | 无直接 PR 号提交 |
| `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_rocm.py` | 无直接 PR 号提交 |
| `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` | 无直接 PR 号提交 |
| `python/sglang/srt/models/deepseek_common/utils.py` | 无直接 PR 号提交 |
| `python/sglang/srt/models/deepseek_v2.py` | [#13954](https://github.com/sgl-project/sglang/pull/13954) |
| `test/manual/nightly/test_deepseek_v31_perf.py` | 无直接 PR 号提交 |
| `test/manual/test_deepseek_v31.py` | 无直接 PR 号提交 |
| `test/registered/amd/accuracy/mi30x/test_deepseek_v31_eval_amd.py` | 无直接 PR 号提交 |
| `test/registered/amd/perf/mi30x/test_deepseek_v31_perf.py` | 无直接 PR 号提交 |

## PR 覆盖总览

- git 追溯 PR 数: 6
- 原文档显式引用补充 PR 数: 26
- 当前文档总 PR 数: 31
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-08-21 | [#9464](https://github.com/sgl-project/sglang/pull/9464) | merged | Add deepseek v3.1 thinking parser support and update docs | `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/reasoning_parser.py` |
| 2025-08-23 | [#9544](https://github.com/sgl-project/sglang/pull/9544) | merged | [doc] deepseekv31 support | `benchmark/deepseek_v3/README.md`, `docs/basic_usage/deepseek.md` |
| 2025-08-27 | [#9446](https://github.com/sgl-project/sglang/pull/9446) | merged | Support DeepSeek-V3.1 tool call | `python/sglang/srt/function_call/deepseekv31_detector.py`, `examples/chat_template/tool_chat_template_deepseekv31.jinja` |
| 2025-09-03 | [#9895](https://github.com/sgl-project/sglang/pull/9895) | merged | Update tool_chat_template_deepseekv31.jinja | `examples/chat_template/tool_chat_template_deepseekv31.jinja` |
| 2025-09-27 | [#10550](https://github.com/sgl-project/sglang/pull/10550) | merged | Use jsonschema to constrain required or specific tool choice | `test/srt/function_call/test_json_schema_constraint.py`, `test/srt/openai_server/function_call/test_tool_choice.py`, `test/srt/test_function_call_parser.py` |
| 2025-09-29 | [#10875](https://github.com/sgl-project/sglang/pull/10875) | merged | feat(reasoning): improve enable thinking from request | `python/sglang/srt/entrypoints/openai/serving_chat.py` |
| 2025-10-03 | [#11189](https://github.com/sgl-project/sglang/pull/11189) | merged | Add --thinking-mode to run_eval | `python/sglang/test/run_eval.py`, `python/sglang/test/simple_eval_common.py` |
| 2025-10-07 | [#11223](https://github.com/sgl-project/sglang/pull/11223) | merged | Update tool parser and related documentation | `python/sglang/srt/function_call/function_call_parser.py`, `python/sglang/srt/server_args.py` |
| 2025-10-30 | [#12123](https://github.com/sgl-project/sglang/pull/12123) | merged | Fix DeepSeek chat templates to handle tool call arguments type checking (#11700) | `test/srt/test_deepseek_chat_templates.py`, `examples/chat_template/tool_chat_template_deepseekv3.jinja`, `examples/chat_template/tool_chat_template_deepseekv31.jinja` |
| 2025-11-13 | [#13190](https://github.com/sgl-project/sglang/pull/13190) | merged | Remove enable_dp_attention in deepseek nightly tests | `test/srt/nightly/test_deepseek_v32_perf.py`, `test/srt/nightly/test_deepseek_v31_perf.py` |
| 2025-11-14 | [#11589](https://github.com/sgl-project/sglang/pull/11589) | merged | [Tool Call] Steamline function arguments when tool_choice="auto" for deepseekv31_detector | `python/sglang/srt/function_call/deepseekv31_detector.py` |
| 2025-11-26 | [#13954](https://github.com/sgl-project/sglang/pull/13954) | merged | Fix Deepseek v3.1 loading issue | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-12-10 | [#14837](https://github.com/sgl-project/sglang/pull/14837) | merged | [Auto Sync] Update tool_chat_template_deepseekv31.jinja (20251210) | `examples/chat_template/tool_chat_template_deepseekv31.jinja` |
| 2025-12-31 | [#13394](https://github.com/sgl-project/sglang/pull/13394) | merged | Fix DeepSeekV31's structural tag trigger | `python/sglang/srt/function_call/deepseekv31_detector.py` |
| 2026-01-07 | [#16660](https://github.com/sgl-project/sglang/pull/16660) | merged | [CI] Enable dpsk v31 test on nightly H200 | `test/registered/8-gpu-models/test_deepseek_v31.py` |
| 2026-01-16 | [#17178](https://github.com/sgl-project/sglang/pull/17178) | merged | Remove deepseek-r1 from THINKING_MODE_CHOICES in run_eval.py | `python/sglang/test/run_eval.py` |
| 2026-01-16 | [#17133](https://github.com/sgl-project/sglang/pull/17133) | merged | [DeepSeek V3.1/V3.2] Optimize fused moe configs for H20 & H20-3E based on swapab | `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json` |
| 2026-01-19 | [#17320](https://github.com/sgl-project/sglang/pull/17320) | closed | fix: Fixed the issue where "finish_reason":"stop" appeared when calling the tool and the tool was in the content. | `python/sglang/srt/function_call/deepseekv32_detector.py`, `examples/chat_template/tool_chat_template_deepseekv32.jinja` |
| 2026-01-22 | [#17141](https://github.com/sgl-project/sglang/pull/17141) | closed | fix: Fixed the issue where "finish_reason":"stop" appeared when calling the tool and the tool was in the content. | `python/sglang/srt/function_call/deepseekv32_detector.py`, `examples/chat_template/tool_chat_template_deepseekv32.jinja` |
| 2026-01-24 | [#17558](https://github.com/sgl-project/sglang/pull/17558) | closed | fix: Fixed the issue where "finish_reason":"stop" appeared when calling the tool and the tool was in the content. | `python/sglang/srt/function_call/deepseekv32_detector.py`, `examples/chat_template/tool_chat_template_deepseekv32.jinja` |
| 2026-01-26 | [#17761](https://github.com/sgl-project/sglang/pull/17761) | open | fix: missing Assistant token after tool output in DeepSeek v3.1/v3.2 chat templates | `test/manual/test_deepseek_chat_templates.py`, `examples/chat_template/tool_chat_template_deepseekv31.jinja`, `examples/chat_template/tool_chat_template_deepseekv32.jinja` |
| 2026-02-04 | [#18236](https://github.com/sgl-project/sglang/pull/18236) | open | Fix function call arguments missing in streaming mode for DeepSeek V3.1 | `python/sglang/srt/function_call/deepseekv31_detector.py` |
| 2026-03-31 | [#21739](https://github.com/sgl-project/sglang/pull/21739) | open | [NPU] Update DeepSeek-V3.1 and DeepSeek-V3.2 model deployment instructions in documentation | `docs/platforms/ascend/ascend_npu_best_practice.md` |
| 2026-04-09 | [#22433](https://github.com/sgl-project/sglang/pull/22433) | open | [Test] Add unit tests for DeepSeekV31Detector | `test/registered/unit/function_call/test_deepseekv31_detector.py` |
| 2026-04-11 | [#21593](https://github.com/sgl-project/sglang/pull/21593) | merged | Fix tool call constrained decoding and parsing for models with native formats | `test/registered/unit/function_call/test_function_call_parser.py`, `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/function_call/function_call_parser.py` |
| 2026-04-16 | [#22981](https://github.com/sgl-project/sglang/pull/22981) | open | [Test] Add unit tests for 7 missing function call detectors | `test/registered/unit/function_call/test_function_call_parser.py`, `test/registered/openai_server/function_call/test_tool_choice.py`, `test/registered/unit/function_call/test_kimik2_detector.py` |
| 2026-04-17 | [#22128](https://github.com/sgl-project/sglang/pull/22128) | merged | Allow piecewise CUDA graph with speculative decoding | `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`, `python/sglang/srt/model_executor/model_runner.py`, `test/registered/piecewise_cuda_graph/test_pcg_with_speculative_decoding.py` |
| 2026-04-20 | [#21599](https://github.com/sgl-project/sglang/pull/21599) | merged | [SPEC][1/N] feat: add adaptive speculative_num_steps for EAGLE topk=1 | `python/sglang/srt/model_executor/cuda_graph_runner.py`, `benchmark/bench_adaptive_speculative.py`, `test/registered/unit/spec/test_adaptive_spec_params.py` |
| 2026-04-21 | [#23315](https://github.com/sgl-project/sglang/pull/23315) | merged | Opt-in strip of thinking tokens from radix cache | `test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py`, `python/sglang/srt/managers/schedule_batch.py`, `python/sglang/srt/server_args.py` |
| 2026-04-21 | [#22950](https://github.com/sgl-project/sglang/pull/22950) | closed | [fix] Parser-gated two-phase cache stripping for reasoning radix caches (fixes #22373) | `python/sglang/srt/parser/reasoning_parser.py`, `python/sglang/srt/configs/model_config.py`, `test/registered/unit/mem_cache/test_radix_cache_thinking.py` |
| 2026-04-21 | [#23336](https://github.com/sgl-project/sglang/pull/23336) | open | [SPEC V2][2/N] feat: adaptive spec support spec v2 | `python/sglang/srt/speculative/eagle_worker_v2.py`, `python/sglang/srt/speculative/eagle_info_v2.py`, `python/sglang/srt/managers/scheduler_output_processor_mixin.py` |

## 逐 PR diff 审计卡

### PR #9464 - Add deepseek v3.1 thinking parser support and update docs

- 链接: https://github.com/sgl-project/sglang/pull/9464
- 状态/时间: merged / 2025-08-21
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+136/-78，可读 patch 245 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add deepseek v3.1 thinking parser support and update docs」；模型线: DeepSeek V3.1；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/reasoning_parser.py`；技术摘要: 覆盖「Add deepseek v3.1 thinking parser support and update docs」；主要实现面是 `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/reasoning_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +9/-6 (15 lines); hunks: -859,12 +859,15 @@ def _get_enable_thinking_from_request(self, request: ChatC...; symbols: _get_enable_thinking_from_request, _process_tool_call_stream，涉及 `_get_enable_thinking_from_request, _process_tool_call_stream`；`python/sglang/srt/reasoning_parser.py` modified +4/-3 (7 lines); hunks: -513,12 +513,13 @@ class ReasoningParser:; symbols: ReasoningParser, __init__，涉及 `ReasoningParser, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +9/-6 (15 lines); hunks: -859,12 +859,15 @@ def _get_enable_thinking_from_request(self, request: ChatC...; symbols: _get_enable_thinking_from_request, _process_tool_call_stream
  - `python/sglang/srt/reasoning_parser.py` modified +4/-3 (7 lines); hunks: -513,12 +513,13 @@ class ReasoningParser:; symbols: ReasoningParser, __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/entrypoints/openai/serving_chat.py
@@ -859,12 +859,15 @@ def _get_enable_thinking_from_request(self, request: ChatCompletionRequest) -> b
-        if (
-            hasattr(request, "chat_template_kwargs")
-            and request.chat_template_kwargs
-            and request.chat_template_kwargs.get("enable_thinking") is not None
-        ):
-            return request.chat_template_kwargs.get("enable_thinking")
diff -- python/sglang/srt/reasoning_parser.py
@@ -513,12 +513,13 @@ class ReasoningParser:
-        "qwen3": Qwen3Detector,
-        "qwen3-thinking": Qwen3Detector,
+        "deepseek-v3": Qwen3Detector,
+        "gpt-oss": GptOssDetector,
+        "qwen3": Qwen3Detector,
+        "qwen3-thinking": Qwen3Detector,
```

- 已读文件:
  - runtime: `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +9/-6; `python/sglang/srt/reasoning_parser.py` modified +4/-3
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/reasoning_parser.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #9544 - [doc] deepseekv31 support

- 链接: https://github.com/sgl-project/sglang/pull/9544
- 状态/时间: merged / 2025-08-23
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+82/-4，可读 patch 112 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[doc] deepseekv31 support」；模型线: DeepSeek V3.1；类别: 性能/后端优化；主要 diff: `benchmark/deepseek_v3/README.md`, `docs/basic_usage/deepseek.md`；技术摘要: 覆盖「[doc] deepseekv31 support」；主要实现面是 `benchmark/deepseek_v3/README.md`, `docs/basic_usage/deepseek.md`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `benchmark/deepseek_v3/README.md` modified +80/-2 (82 lines); hunks: -1,4 +1,4; -50,7 +50,9 @@ Add performance optimization options as nee；`docs/basic_usage/deepseek.md` modified +2/-2 (4 lines); hunks: -5,9 +5,9 @@ SGLang provides many optimizations specifically designed for the...。
- 代码 diff 细节:
  - `benchmark/deepseek_v3/README.md` modified +80/-2 (82 lines); hunks: -1,4 +1,4; -50,7 +50,9 @@ Add performance optimization options as nee
  - `docs/basic_usage/deepseek.md` modified +2/-2 (4 lines); hunks: -5,9 +5,9 @@ SGLang provides many optimizations specifically designed for the...
- 关键代码摘录:

```diff
diff -- benchmark/deepseek_v3/README.md
@@ -1,4 +1,4 @@
-# DeepSeek V3 Support
+# DeepSeek V3.1/V3/R1 Support
@@ -50,7 +50,9 @@ Add [performance optimization options](#performance-optimization-options) as nee
-### Example: Sending requests with OpenAI API
+### Usage: Chat with DeepSeek
+#### DeepSeek V3/R1
diff -- docs/basic_usage/deepseek.md
@@ -5,9 +5,9 @@ SGLang provides many optimizations specifically designed for the DeepSeek models
-## Launch DeepSeek V3 with SGLang
+## Launch DeepSeek V3.1/V3/R1 with SGLang
-To run DeepSeek V3/R1 models, the requirements are as follows:
+To run DeepSeek V3.1/V3/R1 models, the recommended settings are as follows:
```

- 已读文件:
  - other: `benchmark/deepseek_v3/README.md` modified +80/-2
  - docs: `docs/basic_usage/deepseek.md` modified +2/-2
- 验证与风险: 该 PR 主要落在文档/示例 `docs/basic_usage/deepseek.md`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #9446 - Support DeepSeek-V3.1 tool call

- 链接: https://github.com/sgl-project/sglang/pull/9446
- 状态/时间: merged / 2025-08-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `examples/chat_template/tool_chat_template_deepseekv31.jinja`, `python/sglang/srt/function_call/deepseekv31_detector.py`；关联提交 `b9683be6538e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+315/-0，可读 patch 331 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support DeepSeek-V3.1 tool call」；模型线: DeepSeek V3.1；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/function_call/deepseekv31_detector.py`, `examples/chat_template/tool_chat_template_deepseekv31.jinja`；技术摘要: 覆盖「Support DeepSeek-V3.1 tool call」；主要实现面是 `python/sglang/srt/function_call/deepseekv31_detector.py`, `examples/chat_template/tool_chat_template_deepseekv31.jinja`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/function_call/deepseekv31_detector.py` added +222/-0 (222 lines); hunks: -0,0 +1,222; symbols: DeepSeekV31Detector, __init__, has_tool_call, detect_and_parse，涉及 `DeepSeekV31Detector, __init__, has_tool_call`；`examples/chat_template/tool_chat_template_deepseekv31.jinja` added +91/-0 (91 lines); hunks: -0,0 +1,91。
- 代码 diff 细节:
  - `python/sglang/srt/function_call/deepseekv31_detector.py` added +222/-0 (222 lines); hunks: -0,0 +1,222; symbols: DeepSeekV31Detector, __init__, has_tool_call, detect_and_parse
  - `examples/chat_template/tool_chat_template_deepseekv31.jinja` added +91/-0 (91 lines); hunks: -0,0 +1,91
- 关键代码摘录:

```diff
diff -- python/sglang/srt/function_call/deepseekv31_detector.py
@@ -0,0 +1,222 @@
+import json
+import logging
+import re
+from typing import List
+from sglang.srt.entrypoints.openai.protocol import Tool
+from sglang.srt.function_call.base_format_detector import BaseFormatDetector
diff -- examples/chat_template/tool_chat_template_deepseekv31.jinja
@@ -0,0 +1,91 @@
+{% if not add_generation_prompt is defined %}
+  {% set add_generation_prompt = false %}
+{% endif %}
+{% if not thinking is defined %}
+  {% set thinking = false %}
+{% endif %}
```

- 已读文件:
  - runtime: `python/sglang/srt/function_call/deepseekv31_detector.py` added +222/-0
  - docs: `examples/chat_template/tool_chat_template_deepseekv31.jinja` added +91/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/function_call/deepseekv31_detector.py`, `python/sglang/srt/function_call/function_call_parser.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #9895 - Update tool_chat_template_deepseekv31.jinja

- 链接: https://github.com/sgl-project/sglang/pull/9895
- 状态/时间: merged / 2025-09-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `examples/chat_template/tool_chat_template_deepseekv31.jinja`；关联提交 `cc9a31c66226`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-3，可读 patch 17 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Update tool_chat_template_deepseekv31.jinja」；模型线: DeepSeek V3.1；类别: 模型实现调整；主要 diff: `examples/chat_template/tool_chat_template_deepseekv31.jinja`；技术摘要: 覆盖「Update tool_chat_template_deepseekv31.jinja」；主要实现面是 `examples/chat_template/tool_chat_template_deepseekv31.jinja`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `examples/chat_template/tool_chat_template_deepseekv31.jinja` modified +3/-3 (6 lines); hunks: -43,13 +43,13。
- 代码 diff 细节:
  - `examples/chat_template/tool_chat_template_deepseekv31.jinja` modified +3/-3 (6 lines); hunks: -43,13 +43,13
- 关键代码摘录:

```diff
diff -- examples/chat_template/tool_chat_template_deepseekv31.jinja
@@ -43,13 +43,13 @@
-          {{'<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>'+ tool['function']['name'] + '<｜tool▁sep｜>' + tool['function']['arguments'] + '<｜tool▁call▁end｜>'}}
+          {{'<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>'+ tool['function']['name'] + '<｜tool▁sep｜>' + tool['function']['arguments']|tojson + '<｜tool▁call▁end｜>'}}
-          {{message['content'] + '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['function']['name'] + '<｜tool▁sep｜>' + tool['function']['arguments'] + '<｜tool▁call▁end｜>'}}
+          {{message['content'] + '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['function']['name'] + '<｜tool▁sep｜>' + tool['function']['arguments']|tojson + '<｜tool▁call▁end｜>'
-        {{'<｜tool▁call▁begin｜>'+ tool['function']['name'] + '<｜tool▁sep｜>' + tool['function']['arguments'] + '<｜tool▁call▁end｜>'}}
+        {{'<｜tool▁call▁begin｜>'+ tool['function']['name'] + '<｜tool▁sep｜>' + tool['function']['arguments']|tojson + '<｜tool▁call▁end｜>'}}
```

- 已读文件:
  - docs: `examples/chat_template/tool_chat_template_deepseekv31.jinja` modified +3/-3
- 验证与风险: 该 PR 主要落在文档/示例 `examples/chat_template/tool_chat_template_deepseekv31.jinja`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #10550 - Use jsonschema to constrain required or specific tool choice

- 链接: https://github.com/sgl-project/sglang/pull/10550
- 状态/时间: merged / 2025-09-27
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 12 个文件，+1558/-50，可读 patch 1876 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Use jsonschema to constrain required or specific tool choice」；模型线: DeepSeek V3.1；类别: 文档/测试/CI；主要 diff: `test/srt/function_call/test_json_schema_constraint.py`, `test/srt/openai_server/function_call/test_tool_choice.py`, `test/srt/test_function_call_parser.py`；技术摘要: 覆盖「Use jsonschema to constrain required or specific tool choice」；主要实现面是 `test/srt/function_call/test_json_schema_constraint.py`, `test/srt/openai_server/function_call/test_tool_choice.py`, `test/srt/test_function_call_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/srt/function_call/test_json_schema_constraint.py` added +618/-0 (618 lines); hunks: -0,0 +1,618; symbols: TestJsonSchemaConstraint, setUp, test_required_tool_choice_schema, test_specific_tool_choice_schema，涉及 `TestJsonSchemaConstraint, setUp, test_required_tool_choice_schema`；`test/srt/openai_server/function_call/test_tool_choice.py` modified +319/-14 (333 lines); hunks: -343,6 +343,142 @@ def test_tool_choice_specific_function_streaming(self):; -408,6 +544,10 @@ def test_multi_tool_scenario_required(self):; symbols: test_tool_choice_specific_function_streaming, test_required_streaming_arguments_chunks_json, test_complex_parameters_required_non_streaming, test_multi_tool_scenario_auto，涉及 `test_tool_choice_specific_function_streaming, test_required_streaming_arguments_chunks_json, test_complex_parameters_required_non_streaming`；`test/srt/test_function_call_parser.py` modified +319/-0 (319 lines); hunks: -5,8 +5,10; -2190,5 +2192,322 @@ def test_partial_tool_call(self):; symbols: test_partial_tool_call, TestJsonArrayParser, setUp, test_json_detector_ebnf，涉及 `test_partial_tool_call, TestJsonArrayParser, setUp`；`python/sglang/srt/entrypoints/openai/serving_chat.py` modified +115/-22 (137 lines); hunks: -9,6 +9,7; -25,6 +26,8; symbols: _validate_request, _process_messages, _build_sampling_params, _build_chat_response，涉及 `_validate_request, _process_messages, _build_sampling_params`。
- 代码 diff 细节:
  - `test/srt/function_call/test_json_schema_constraint.py` added +618/-0 (618 lines); hunks: -0,0 +1,618; symbols: TestJsonSchemaConstraint, setUp, test_required_tool_choice_schema, test_specific_tool_choice_schema
  - `test/srt/openai_server/function_call/test_tool_choice.py` modified +319/-14 (333 lines); hunks: -343,6 +343,142 @@ def test_tool_choice_specific_function_streaming(self):; -408,6 +544,10 @@ def test_multi_tool_scenario_required(self):; symbols: test_tool_choice_specific_function_streaming, test_required_streaming_arguments_chunks_json, test_complex_parameters_required_non_streaming, test_multi_tool_scenario_auto
  - `test/srt/test_function_call_parser.py` modified +319/-0 (319 lines); hunks: -5,8 +5,10; -2190,5 +2192,322 @@ def test_partial_tool_call(self):; symbols: test_partial_tool_call, TestJsonArrayParser, setUp, test_json_detector_ebnf
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +115/-22 (137 lines); hunks: -9,6 +9,7; -25,6 +26,8; symbols: _validate_request, _process_messages, _build_sampling_params, _build_chat_response
  - `python/sglang/srt/function_call/utils.py` modified +96/-5 (101 lines); hunks: -1,10 +1,13; -37,10 +40,12 @@ def _partial_json_loads(input_str: str, flags: Allow) -> Tup...; symbols: _find_common_prefix, _partial_json_loads, _is_complete_json, _get_tool_schema_defs
- 关键代码摘录:

```diff
diff -- test/srt/function_call/test_json_schema_constraint.py
@@ -0,0 +1,618 @@
+"""
+Tests for JSON schema constraint functionality used by JsonArrayParser
+"""
+import json
+import unittest
+import jsonschema
diff -- test/srt/openai_server/function_call/test_tool_choice.py
@@ -343,6 +343,142 @@ def test_tool_choice_specific_function_streaming(self):
+    def test_required_streaming_arguments_chunks_json(self):
+        """In streaming required mode, complete tool call arguments should be valid JSON when all chunks are combined"""
+        tools = self.get_test_tools()
+        messages = self.get_test_messages()
+        response = self.client.chat.completions.create(
+            model=self.model_name,
diff -- test/srt/test_function_call_parser.py
@@ -5,8 +5,10 @@
```

- 已读文件:
  - tests: `test/srt/function_call/test_json_schema_constraint.py` added +618/-0; `test/srt/openai_server/function_call/test_tool_choice.py` modified +319/-14; `test/srt/test_function_call_parser.py` modified +319/-0; `test/srt/openai_server/function_call/test_openai_function_calling.py` modified +4/-4
  - runtime: `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +115/-22; `python/sglang/srt/function_call/utils.py` modified +96/-5; `python/sglang/srt/function_call/json_array_parser.py` added +63/-0; `python/sglang/srt/entrypoints/openai/protocol.py` modified +12/-2
- 验证与风险: diff 自带测试面 `test/srt/function_call/test_json_schema_constraint.py`, `test/srt/openai_server/basic/test_serving_chat.py`, `test/srt/openai_server/function_call/test_openai_function_calling.py`, `test/srt/openai_server/function_call/test_tool_choice.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #10875 - feat(reasoning): improve enable thinking from request

- 链接: https://github.com/sgl-project/sglang/pull/10875
- 状态/时间: merged / 2025-09-29
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+8/-10，可读 patch 54 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「feat(reasoning): improve enable thinking from request」；模型线: DeepSeek V3.1；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/entrypoints/openai/serving_chat.py`；技术摘要: 覆盖「feat(reasoning): improve enable thinking from request」；主要实现面是 `python/sglang/srt/entrypoints/openai/serving_chat.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +8/-10 (18 lines); hunks: -64,6 +64,7 @@ def __init__(; -563,10 +564,7 @@ async def _generate_chat_stream(; symbols: __init__, _request_id_prefix, _generate_chat_stream, _build_chat_response，涉及 `__init__, _request_id_prefix, _generate_chat_stream`。
- 代码 diff 细节:
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +8/-10 (18 lines); hunks: -64,6 +64,7 @@ def __init__(; -563,10 +564,7 @@ async def _generate_chat_stream(; symbols: __init__, _request_id_prefix, _generate_chat_stream, _build_chat_response
- 关键代码摘录:

```diff
diff -- python/sglang/srt/entrypoints/openai/serving_chat.py
@@ -64,6 +64,7 @@ def __init__(
+        self.reasoning_parser = self.tokenizer_manager.server_args.reasoning_parser
@@ -563,10 +564,7 @@ async def _generate_chat_stream(
-                if (
-                    self.tokenizer_manager.server_args.reasoning_parser
-                    and request.separate_reasoning
-                ):
```

- 已读文件:
  - runtime: `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +8/-10
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/entrypoints/openai/serving_chat.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #11189 - Add --thinking-mode to run_eval

- 链接: https://github.com/sgl-project/sglang/pull/11189
- 状态/时间: merged / 2025-10-03
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+29/-1，可读 patch 81 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add --thinking-mode to run_eval」；模型线: DeepSeek V3.1；类别: 模型支持/运行时入口；主要 diff: `python/sglang/test/run_eval.py`, `python/sglang/test/simple_eval_common.py`；技术摘要: 覆盖「Add --thinking-mode to run_eval」；主要实现面是 `python/sglang/test/run_eval.py`, `python/sglang/test/simple_eval_common.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/test/run_eval.py` modified +25/-0 (25 lines); hunks: -16,13 +16,29; -136,6 +152,8 @@ def run_eval(args):; symbols: get_thinking_kwargs, run_eval_once, run_eval，涉及 `get_thinking_kwargs, run_eval_once, run_eval`；`python/sglang/test/simple_eval_common.py` modified +4/-1 (5 lines); hunks: -93,6 +93,7 @@ def __init__(; -104,9 +105,10 @@ def __init__(; symbols: __init__, _handle_image, __call__，涉及 `__init__, _handle_image, __call__`。
- 代码 diff 细节:
  - `python/sglang/test/run_eval.py` modified +25/-0 (25 lines); hunks: -16,13 +16,29; -136,6 +152,8 @@ def run_eval(args):; symbols: get_thinking_kwargs, run_eval_once, run_eval
  - `python/sglang/test/simple_eval_common.py` modified +4/-1 (5 lines); hunks: -93,6 +93,7 @@ def __init__(; -104,9 +105,10 @@ def __init__(; symbols: __init__, _handle_image, __call__
- 关键代码摘录:

```diff
diff -- python/sglang/test/run_eval.py
@@ -16,13 +16,29 @@
+def get_thinking_kwargs(args):
+    if args.thinking_mode in THINKING_MODE_CHOICES:
+        thinking_param = (
+            "thinking" if args.thinking_mode == "deepseek-v3" else "enable_thinking"
+        )
+        return {
diff -- python/sglang/test/simple_eval_common.py
@@ -93,6 +93,7 @@ def __init__(
+        extra_body: Optional[Dict[str, Any]] = None,
@@ -104,9 +105,10 @@ def __init__(
+        self.extra_body = extra_body
-            f"ChatCompletionSampler initialized with {self.system_message=} {self.temperature=} {self.max_tokens=} {self.reasoning_effort=}"
+            f"ChatCompletionSampler initialized with {self.system_message=} {self.temperature=} {self.max_tokens=} {self.reasoning_effort=} {self.extra_body=}"
@@ -144,6 +146,7 @@ def __call__(self, message_list: MessageList) -> str:
```

- 已读文件:
  - tests: `python/sglang/test/run_eval.py` modified +25/-0; `python/sglang/test/simple_eval_common.py` modified +4/-1
- 验证与风险: diff 自带测试面 `python/sglang/test/run_eval.py`, `python/sglang/test/simple_eval_common.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #11223 - Update tool parser and related documentation

- 链接: https://github.com/sgl-project/sglang/pull/11223
- 状态/时间: merged / 2025-10-07
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+24/-12，可读 patch 65 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Update tool parser and related documentation」；模型线: DeepSeek V3.1；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/function_call/function_call_parser.py`, `python/sglang/srt/server_args.py`；技术摘要: 覆盖「Update tool parser and related documentation」；主要实现面是 `python/sglang/srt/function_call/function_call_parser.py`, `python/sglang/srt/server_args.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/function_call/function_call_parser.py` modified +8/-6 (14 lines); hunks: -35,17 +35,19 @@ class FunctionCallParser:; symbols: FunctionCallParser, __init__，涉及 `FunctionCallParser, __init__`；`python/sglang/srt/server_args.py` modified +7/-1 (8 lines); hunks: -527,7 +527,13 @@ def __post_init__(self):; symbols: __post_init__, _handle_deprecated_args, _handle_missing_default_values，涉及 `__post_init__, _handle_deprecated_args, _handle_missing_default_values`。
- 代码 diff 细节:
  - `python/sglang/srt/function_call/function_call_parser.py` modified +8/-6 (14 lines); hunks: -35,17 +35,19 @@ class FunctionCallParser:; symbols: FunctionCallParser, __init__
  - `python/sglang/srt/server_args.py` modified +7/-1 (8 lines); hunks: -527,7 +527,13 @@ def __post_init__(self):; symbols: __post_init__, _handle_deprecated_args, _handle_missing_default_values
- 关键代码摘录:

```diff
diff -- python/sglang/srt/function_call/function_call_parser.py
@@ -35,17 +35,19 @@ class FunctionCallParser:
-        "llama3": Llama32Detector,
-        "qwen25": Qwen25Detector,
-        "mistral": MistralDetector,
-        "pythonic": PythonicDetector,
+        "glm": Glm4MoeDetector,
+        "glm45": Glm4MoeDetector,
diff -- python/sglang/srt/server_args.py
@@ -527,7 +527,13 @@ def __post_init__(self):
-        pass
+        # handle deprecated tool call parsers
+        deprecated_tool_call_parsers = {"qwen25": "qwen", "glm45": "glm"}
+        if self.tool_call_parser in deprecated_tool_call_parsers:
+            logger.warning(
+                f"The tool_call_parser '{self.tool_call_parser}' is deprecated. Please use '{deprecated_tool_call_parsers[self.tool_call_parser]}' instead."
```

- 已读文件:
  - runtime: `python/sglang/srt/function_call/function_call_parser.py` modified +8/-6; `python/sglang/srt/server_args.py` modified +7/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/function_call/function_call_parser.py`, `python/sglang/srt/server_args.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #12123 - Fix DeepSeek chat templates to handle tool call arguments type checking (#11700)

- 链接: https://github.com/sgl-project/sglang/pull/12123
- 状态/时间: merged / 2025-10-30
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+331/-9，可读 patch 380 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix DeepSeek chat templates to handle tool call arguments type checking (#11700)」；模型线: DeepSeek V3.1；类别: 缺陷修复；主要 diff: `test/srt/test_deepseek_chat_templates.py`, `examples/chat_template/tool_chat_template_deepseekv3.jinja`, `examples/chat_template/tool_chat_template_deepseekv31.jinja`；技术摘要: 覆盖「Fix DeepSeek chat templates to handle tool call arguments type checking (#11700)」；主要实现面是 `test/srt/test_deepseek_chat_templates.py`, `examples/chat_template/tool_chat_template_deepseekv3.jinja`, `examples/chat_template/tool_chat_template_deepseekv31.jinja`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/srt/test_deepseek_chat_templates.py` added +319/-0 (319 lines); hunks: -0,0 +1,319; symbols: TestDeepSeekChatTemplateToolCalls, setUpClass, _render_template, test_tool_arguments_as_dict，涉及 `TestDeepSeekChatTemplateToolCalls, setUpClass, _render_template`；`examples/chat_template/tool_chat_template_deepseekv3.jinja` modified +4/-3 (7 lines); hunks: -47,15 +47,16；`examples/chat_template/tool_chat_template_deepseekv31.jinja` modified +4/-3 (7 lines); hunks: -41,15 +41,16；`examples/chat_template/tool_chat_template_deepseekv32.jinja` modified +4/-3 (7 lines); hunks: -42,15 +42,16。
- 代码 diff 细节:
  - `test/srt/test_deepseek_chat_templates.py` added +319/-0 (319 lines); hunks: -0,0 +1,319; symbols: TestDeepSeekChatTemplateToolCalls, setUpClass, _render_template, test_tool_arguments_as_dict
  - `examples/chat_template/tool_chat_template_deepseekv3.jinja` modified +4/-3 (7 lines); hunks: -47,15 +47,16
  - `examples/chat_template/tool_chat_template_deepseekv31.jinja` modified +4/-3 (7 lines); hunks: -41,15 +41,16
  - `examples/chat_template/tool_chat_template_deepseekv32.jinja` modified +4/-3 (7 lines); hunks: -42,15 +42,16
- 关键代码摘录:

```diff
diff -- test/srt/test_deepseek_chat_templates.py
@@ -0,0 +1,319 @@
+"""
+Unit tests for DeepSeek chat template tool call handling.
+Tests verify that the DeepSeek chat templates (v3, v3.1, v3.2) correctly handle
+both dict and string types for tool['function']['arguments'] without double-escaping,
+addressing issue #11700.
+"""
diff -- examples/chat_template/tool_chat_template_deepseekv3.jinja
@@ -47,15 +47,16 @@
+            {%- set formatted_args = tool['function']['arguments'] if tool['function']['arguments'] is string else tool['function']['arguments']|tojson %}
-                    {{- '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\n' + ''''json' + '\n' + tool['function']['argument
+                    {{- '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\n' + ''''json' + '\n' + formatted_args + '\n' + '`
-                    {{- message['content'] + '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\n' + ''''json' + '\n' + tool[
+                    {{- message['content'] + '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\n' + ''''json' + '\n' + forma
-                {{- '\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\n' + ''''json' + '\n' + tool['function']['arguments']|tojson + '\n'
diff -- examples/chat_template/tool_chat_template_deepseekv31.jinja
@@ -41,15 +41,16 @@
```

- 已读文件:
  - tests: `test/srt/test_deepseek_chat_templates.py` added +319/-0
  - docs: `examples/chat_template/tool_chat_template_deepseekv3.jinja` modified +4/-3; `examples/chat_template/tool_chat_template_deepseekv31.jinja` modified +4/-3; `examples/chat_template/tool_chat_template_deepseekv32.jinja` modified +4/-3
- 验证与风险: diff 自带测试面 `test/srt/test_deepseek_chat_templates.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #13190 - Remove enable_dp_attention in deepseek nightly tests

- 链接: https://github.com/sgl-project/sglang/pull/13190
- 状态/时间: merged / 2025-11-13
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+0/-5，可读 patch 40 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Remove enable_dp_attention in deepseek nightly tests」；模型线: DeepSeek V3.1；类别: 文档/测试/CI；主要 diff: `test/srt/nightly/test_deepseek_v32_perf.py`, `test/srt/nightly/test_deepseek_v31_perf.py`；未提供可用技术摘要。
- 实现要点: `test/srt/nightly/test_deepseek_v32_perf.py` modified +0/-3 (3 lines); hunks: -27,7 +27,6 @@ def setUpClass(cls):; -38,7 +37,6 @@ def setUpClass(cls):; symbols: setUpClass，涉及 `setUpClass`；`test/srt/nightly/test_deepseek_v31_perf.py` modified +0/-2 (2 lines); hunks: -27,7 +27,6 @@ def setUpClass(cls):; -38,7 +37,6 @@ def setUpClass(cls):; symbols: setUpClass，涉及 `setUpClass`。
- 代码 diff 细节:
  - `test/srt/nightly/test_deepseek_v32_perf.py` modified +0/-3 (3 lines); hunks: -27,7 +27,6 @@ def setUpClass(cls):; -38,7 +37,6 @@ def setUpClass(cls):; symbols: setUpClass
  - `test/srt/nightly/test_deepseek_v31_perf.py` modified +0/-2 (2 lines); hunks: -27,7 +27,6 @@ def setUpClass(cls):; -38,7 +37,6 @@ def setUpClass(cls):; symbols: setUpClass
- 关键代码摘录:

```diff
diff -- test/srt/nightly/test_deepseek_v32_perf.py
@@ -27,7 +27,6 @@ def setUpClass(cls):
-                    "--enable-dp-attention",
@@ -38,7 +37,6 @@ def setUpClass(cls):
-                    "--enable-dp-attention",
@@ -59,7 +57,6 @@ def setUpClass(cls):
-                    "--enable-dp-attention",
diff -- test/srt/nightly/test_deepseek_v31_perf.py
@@ -27,7 +27,6 @@ def setUpClass(cls):
-                    "--enable-dp-attention",
@@ -38,7 +37,6 @@ def setUpClass(cls):
-                    "--enable-dp-attention",
```

- 已读文件:
  - tests: `test/srt/nightly/test_deepseek_v32_perf.py` modified +0/-3; `test/srt/nightly/test_deepseek_v31_perf.py` modified +0/-2
- 验证与风险: diff 自带测试面 `test/srt/nightly/test_deepseek_v31_perf.py`, `test/srt/nightly/test_deepseek_v32_perf.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #11589 - [Tool Call] Steamline function arguments when tool_choice="auto" for deepseekv31_detector

- 链接: https://github.com/sgl-project/sglang/pull/11589
- 状态/时间: merged / 2025-11-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/function_call/deepseekv31_detector.py`；关联提交 `fc5da1e80b78`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+4/-9，可读 patch 34 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Tool Call] Steamline function arguments when tool_choice="auto" for deepseekv31_detector」；模型线: DeepSeek V3.1；类别: 模型实现调整；主要 diff: `python/sglang/srt/function_call/deepseekv31_detector.py`；技术摘要: 覆盖「[Tool Call] Steamline function arguments when tool_choice="auto" for deepseekv31_detector」；主要实现面是 `python/sglang/srt/function_call/deepseekv31_detector.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/function_call/deepseekv31_detector.py` modified +4/-9 (13 lines); hunks: -115,13 +115,14 @@ def parse_streaming_increment(; -180,15 +181,9 @@ def parse_streaming_increment(; symbols: parse_streaming_increment，涉及 `parse_streaming_increment`。
- 代码 diff 细节:
  - `python/sglang/srt/function_call/deepseekv31_detector.py` modified +4/-9 (13 lines); hunks: -115,13 +115,14 @@ def parse_streaming_increment(; -180,15 +181,9 @@ def parse_streaming_increment(; symbols: parse_streaming_increment
- 关键代码摘录:

```diff
diff -- python/sglang/srt/function_call/deepseekv31_detector.py
@@ -115,13 +115,14 @@ def parse_streaming_increment(
-                pattern=r"<｜tool▁call▁begin｜>(.*)<｜tool▁sep｜>(.*)<｜tool▁call▁end｜>",
+                pattern=r"<｜tool▁call▁begin｜>(.*)<｜tool▁sep｜>(.*?)(<｜tool▁call▁end｜>|$)",
+                is_tool_end = partial_match.group(3)
@@ -180,15 +181,9 @@ def parse_streaming_increment(
-                        tool_call_end_pattern = (
-                            r"<｜tool▁call▁begin｜>.*?<｜tool▁call▁end｜>"
```

- 已读文件:
  - runtime: `python/sglang/srt/function_call/deepseekv31_detector.py` modified +4/-9
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/function_call/deepseekv31_detector.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #13954 - Fix Deepseek v3.1 loading issue

- 链接: https://github.com/sgl-project/sglang/pull/13954
- 状态/时间: merged / 2025-11-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/deepseek_v2.py`；关联提交 `13e5beeab499`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix Deepseek v3.1 loading issue」；模型线: DeepSeek V3.1；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/deepseek_v2.py`；技术摘要: 覆盖「Fix Deepseek v3.1 loading issue」；主要实现面是 `python/sglang/srt/models/deepseek_v2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/deepseek_v2.py` modified +1/-1 (2 lines); hunks: -3568,7 +3568,7 @@ def post_load_weights(self, is_nextn=False, weight_names=N...; symbols: post_load_weights，涉及 `post_load_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v2.py` modified +1/-1 (2 lines); hunks: -3568,7 +3568,7 @@ def post_load_weights(self, is_nextn=False, weight_names=N...; symbols: post_load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -3568,7 +3568,7 @@ def post_load_weights(self, is_nextn=False, weight_names=None):
-                        and self_attn.kv_b_proj.executed_weight_requant_ue8m0
+                        and weight_scale.format_ue8m0
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #14837 - [Auto Sync] Update tool_chat_template_deepseekv31.jinja (20251210)

- 链接: https://github.com/sgl-project/sglang/pull/14837
- 状态/时间: merged / 2025-12-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `examples/chat_template/tool_chat_template_deepseekv31.jinja`；关联提交 `ef1ab2302ab2`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+5/-1，可读 patch 13 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Auto Sync] Update tool_chat_template_deepseekv31.jinja (20251210)」；模型线: DeepSeek V3.1；类别: 模型实现调整；主要 diff: `examples/chat_template/tool_chat_template_deepseekv31.jinja`；技术摘要: 覆盖「[Auto Sync] Update tool_chat_template_deepseekv31.jinja (20251210)」；主要实现面是 `examples/chat_template/tool_chat_template_deepseekv31.jinja`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `examples/chat_template/tool_chat_template_deepseekv31.jinja` modified +5/-1 (6 lines); hunks: -19,7 +19,11。
- 代码 diff 细节:
  - `examples/chat_template/tool_chat_template_deepseekv31.jinja` modified +5/-1 (6 lines); hunks: -19,7 +19,11
- 关键代码摘录:

```diff
diff -- examples/chat_template/tool_chat_template_deepseekv31.jinja
@@ -19,7 +19,11 @@
-    {% set tool_ns.text = tool_ns.text + '\n### ' + tool.function.name + '\nDescription: ' + tool.function.description + '\n\nParameters: ' + (tool.function.parameters | tojson) +
+    {% if tool.function.description is not none %}
+      {% set tool_ns.text = tool_ns.text + '\n### ' + tool.function.name + '\nDescription: ' + tool.function.description + '\n\nParameters: ' + (tool.function.parameters | tojson)
+    {% else %}
+      {% set tool_ns.text = tool_ns.text + '\n### ' + tool.function.name + '\n\nParameters: ' + (tool.function.parameters | tojson) + '\n' %}
+    {% endif %}
```

- 已读文件:
  - docs: `examples/chat_template/tool_chat_template_deepseekv31.jinja` modified +5/-1
- 验证与风险: 该 PR 主要落在文档/示例 `examples/chat_template/tool_chat_template_deepseekv31.jinja`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #13394 - Fix DeepSeekV31's structural tag trigger

- 链接: https://github.com/sgl-project/sglang/pull/13394
- 状态/时间: merged / 2025-12-31
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/function_call/deepseekv31_detector.py`；关联提交 `2667c857a78f`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 7 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix DeepSeekV31's structural tag trigger」；模型线: DeepSeek V3.1；类别: 缺陷修复；主要 diff: `python/sglang/srt/function_call/deepseekv31_detector.py`；技术摘要: 覆盖「Fix DeepSeekV31's structural tag trigger」；主要实现面是 `python/sglang/srt/function_call/deepseekv31_detector.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/function_call/deepseekv31_detector.py` modified +1/-1 (2 lines); hunks: -202,5 +202,5 @@ def structure_info(self) -> _GetInfoFunc:; symbols: structure_info，涉及 `structure_info`。
- 代码 diff 细节:
  - `python/sglang/srt/function_call/deepseekv31_detector.py` modified +1/-1 (2 lines); hunks: -202,5 +202,5 @@ def structure_info(self) -> _GetInfoFunc:; symbols: structure_info
- 关键代码摘录:

```diff
diff -- python/sglang/srt/function_call/deepseekv31_detector.py
@@ -202,5 +202,5 @@ def structure_info(self) -> _GetInfoFunc:
-            trigger="<｜tool▁call▁begin｜>" + name + "<｜tool▁sep｜>",
+            trigger="<｜tool▁call▁begin｜>",
```

- 已读文件:
  - runtime: `python/sglang/srt/function_call/deepseekv31_detector.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/function_call/deepseekv31_detector.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #16660 - [CI] Enable dpsk v31 test on nightly H200

- 链接: https://github.com/sgl-project/sglang/pull/16660
- 状态/时间: merged / 2026-01-07
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-2，可读 patch 17 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI] Enable dpsk v31 test on nightly H200」；模型线: DeepSeek V3.1；类别: 文档/测试/CI；主要 diff: `test/registered/8-gpu-models/test_deepseek_v31.py`；未提供可用技术摘要。
- 实现要点: `test/registered/8-gpu-models/test_deepseek_v31.py` modified +1/-2 (3 lines); hunks: -4,15 +4,14; symbols: TestDeepseekV31Unified, for，涉及 `TestDeepseekV31Unified, for`。
- 代码 diff 细节:
  - `test/registered/8-gpu-models/test_deepseek_v31.py` modified +1/-2 (3 lines); hunks: -4,15 +4,14; symbols: TestDeepseekV31Unified, for
- 关键代码摘录:

```diff
diff -- test/registered/8-gpu-models/test_deepseek_v31.py
@@ -4,15 +4,14 @@
-from sglang.test.test_utils import ModelLaunchSettings, is_blackwell_system
+from sglang.test.test_utils import ModelLaunchSettings
-@unittest.skipIf(not is_blackwell_system(), "Requires B200")
```

- 已读文件:
  - tests: `test/registered/8-gpu-models/test_deepseek_v31.py` modified +1/-2
- 验证与风险: diff 自带测试面 `test/registered/8-gpu-models/test_deepseek_v31.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #17178 - Remove deepseek-r1 from THINKING_MODE_CHOICES in run_eval.py

- 链接: https://github.com/sgl-project/sglang/pull/17178
- 状态/时间: merged / 2026-01-16
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-2，可读 patch 26 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Remove deepseek-r1 from THINKING_MODE_CHOICES in run_eval.py」；模型线: DeepSeek V3.1；类别: 模型支持/运行时入口；主要 diff: `python/sglang/test/run_eval.py`；技术摘要: 覆盖「Remove deepseek-r1 from THINKING_MODE_CHOICES in run_eval.py」；主要实现面是 `python/sglang/test/run_eval.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/test/run_eval.py` modified +3/-2 (5 lines); hunks: -22,6 +22,7 @@ def get_thinking_kwargs(args):; -203,7 +204,7 @@ def run_eval(args):; symbols: get_thinking_kwargs, run_eval，涉及 `get_thinking_kwargs, run_eval`。
- 代码 diff 细节:
  - `python/sglang/test/run_eval.py` modified +3/-2 (5 lines); hunks: -22,6 +22,7 @@ def get_thinking_kwargs(args):; -203,7 +204,7 @@ def run_eval(args):; symbols: get_thinking_kwargs, run_eval
- 关键代码摘录:

```diff
diff -- python/sglang/test/run_eval.py
@@ -22,6 +22,7 @@ def get_thinking_kwargs(args):
+            # Qwen3
@@ -203,7 +204,7 @@ def run_eval(args):
-THINKING_MODE_CHOICES = ["deepseek-r1", "deepseek-v3", "qwen3"]
+THINKING_MODE_CHOICES = ["deepseek-v3", "qwen3"]
@@ -241,7 +242,7 @@ def run_eval(args):
-        help="Enable thinking mode in Deepseek R1, V3.1/3.2, or Qwen3",
```

- 已读文件:
  - tests: `python/sglang/test/run_eval.py` modified +3/-2
- 验证与风险: diff 自带测试面 `python/sglang/test/run_eval.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #17133 - [DeepSeek V3.1/V3.2] Optimize fused moe configs for H20 & H20-3E based on swapab

- 链接: https://github.com/sgl-project/sglang/pull/17133
- 状态/时间: merged / 2026-01-16
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+959/-217，可读 patch 1311 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DeepSeek V3.1/V3.2] Optimize fused moe configs for H20 & H20-3E based on swapab」；模型线: DeepSeek V3.1；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json`；技术摘要: 覆盖「[DeepSeek V3.1/V3.2] Optimize fused moe configs for H20 & H20-3E based on swapab」；主要实现面是 `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128]_down.json` added +164/-0 (164 lines); hunks: -0,0 +1,164；`python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128]_down.json` added +164/-0 (164 lines); hunks: -0,0 +1,164；`python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: -0,0 +1,146；`python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: -0,0 +1,146。
- 代码 diff 细节:
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128]_down.json` added +164/-0 (164 lines); hunks: -0,0 +1,164
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128]_down.json` added +164/-0 (164 lines); hunks: -0,0 +1,164
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: -0,0 +1,146
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: -0,0 +1,146
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py` modified +2/-2 (4 lines); hunks: -744,8 +744,8 @@ def invoke_fused_moe_kernel(; symbols: invoke_fused_moe_kernel
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128]_down.json
@@ -0,0 +1,164 @@
+{
+    "1": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 64,
+        "GROUP_SIZE_M": 64,
diff -- python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128]_down.json
@@ -0,0 +1,164 @@
+{
+    "1": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 64,
+        "GROUP_SIZE_M": 32,
diff -- python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json
@@ -0,0 +1,146 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128]_down.json` added +164/-0; `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128]_down.json` added +164/-0; `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0; `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0; `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py` modified +2/-2
  - other: `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py` modified +337/-215
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128].json`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #17320 - fix: Fixed the issue where "finish_reason":"stop" appeared when calling the tool and the tool was in the content.

- 链接: https://github.com/sgl-project/sglang/pull/17320
- 状态/时间: closed / 2026-01-19
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+14/-16，可读 patch 56 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix: Fixed the issue where "finish_reason":"stop" appeared when calling the tool and the tool was in the content.」；模型线: DeepSeek V3.1；类别: 缺陷修复；主要 diff: `python/sglang/srt/function_call/deepseekv32_detector.py`, `examples/chat_template/tool_chat_template_deepseekv32.jinja`；技术摘要: 覆盖「fix: Fixed the issue where "finish_reason":"stop" appeared when calling the tool and the tool was in the content.」；主要实现面是 `python/sglang/srt/function_call/deepseekv32_detector.py`, `examples/chat_template/tool_chat_template_deepseekv32.jinja`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/function_call/deepseekv32_detector.py` modified +8/-4 (12 lines); hunks: -195,12 +195,16 @@ def detect_and_parse(self, text: str, tools: list[Tool]) -...; symbols: detect_and_parse，涉及 `detect_and_parse`；`examples/chat_template/tool_chat_template_deepseekv32.jinja` modified +6/-12 (18 lines); hunks: -22,7 +22,7; -41,20 +41,14。
- 代码 diff 细节:
  - `python/sglang/srt/function_call/deepseekv32_detector.py` modified +8/-4 (12 lines); hunks: -195,12 +195,16 @@ def detect_and_parse(self, text: str, tools: list[Tool]) -...; symbols: detect_and_parse
  - `examples/chat_template/tool_chat_template_deepseekv32.jinja` modified +6/-12 (18 lines); hunks: -22,7 +22,7; -41,20 +41,14
- 关键代码摘录:

```diff
diff -- python/sglang/srt/function_call/deepseekv32_detector.py
@@ -195,12 +195,16 @@ def detect_and_parse(self, text: str, tools: list[Tool]) -> StreamingParseResult
-            for func_name, invoke_content, _ in invoke_matches:
+            for i, (func_name, invoke_content, _) in enumerate(invoke_matches):
-                # construct match_result for parse_base_json
-                match_result = {"name": func_name, "parameters": func_args}
-                calls.extend(self.parse_base_json(match_result, tools))
+                calls.append(
diff -- examples/chat_template/tool_chat_template_deepseekv32.jinja
@@ -22,7 +22,7 @@
-  {% set tool_ns.text = tool_ns.text + "\nIMPORTANT: ALWAYS adhere to this exact format for tool use:\n<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>tool_call_name<｜tool▁sep｜>tool_call_a
+  {% set tool_ns.text = tool_ns.text + "\nIMPORTANT: ALWAYS adhere to this exact format for tool use:\n<｜DSML｜function_calls>\n<｜DSML｜invoke name=\"tool_call_name\">\n<｜DSML｜param
@@ -41,20 +41,14 @@
+    {{'<｜DSML｜function_calls>'}}
-      {%- if not ns.is_first %}
-        {%- if message['content'] is none %}
```

- 已读文件:
  - runtime: `python/sglang/srt/function_call/deepseekv32_detector.py` modified +8/-4
  - docs: `examples/chat_template/tool_chat_template_deepseekv32.jinja` modified +6/-12
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/function_call/deepseekv32_detector.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #17141 - fix: Fixed the issue where "finish_reason":"stop" appeared when calling the tool and the tool was in the content.

- 链接: https://github.com/sgl-project/sglang/pull/17141
- 状态/时间: closed / 2026-01-22
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+14/-16，可读 patch 56 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix: Fixed the issue where "finish_reason":"stop" appeared when calling the tool and the tool was in the content.」；模型线: DeepSeek V3.1；类别: 缺陷修复；主要 diff: `python/sglang/srt/function_call/deepseekv32_detector.py`, `examples/chat_template/tool_chat_template_deepseekv32.jinja`；技术摘要: 覆盖「fix: Fixed the issue where "finish_reason":"stop" appeared when calling the tool and the tool was in the content.」；主要实现面是 `python/sglang/srt/function_call/deepseekv32_detector.py`, `examples/chat_template/tool_chat_template_deepseekv32.jinja`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/function_call/deepseekv32_detector.py` modified +8/-4 (12 lines); hunks: -195,12 +195,16 @@ def detect_and_parse(self, text: str, tools: list[Tool]) -...; symbols: detect_and_parse，涉及 `detect_and_parse`；`examples/chat_template/tool_chat_template_deepseekv32.jinja` modified +6/-12 (18 lines); hunks: -22,7 +22,7; -41,20 +41,14。
- 代码 diff 细节:
  - `python/sglang/srt/function_call/deepseekv32_detector.py` modified +8/-4 (12 lines); hunks: -195,12 +195,16 @@ def detect_and_parse(self, text: str, tools: list[Tool]) -...; symbols: detect_and_parse
  - `examples/chat_template/tool_chat_template_deepseekv32.jinja` modified +6/-12 (18 lines); hunks: -22,7 +22,7; -41,20 +41,14
- 关键代码摘录:

```diff
diff -- python/sglang/srt/function_call/deepseekv32_detector.py
@@ -195,12 +195,16 @@ def detect_and_parse(self, text: str, tools: list[Tool]) -> StreamingParseResult
-            for func_name, invoke_content, _ in invoke_matches:
+            for i, (func_name, invoke_content, _) in enumerate(invoke_matches):
-                # construct match_result for parse_base_json
-                match_result = {"name": func_name, "parameters": func_args}
-                calls.extend(self.parse_base_json(match_result, tools))
+                calls.append(
diff -- examples/chat_template/tool_chat_template_deepseekv32.jinja
@@ -22,7 +22,7 @@
-  {% set tool_ns.text = tool_ns.text + "\nIMPORTANT: ALWAYS adhere to this exact format for tool use:\n<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>tool_call_name<｜tool▁sep｜>tool_call_a
+  {% set tool_ns.text = tool_ns.text + "\nIMPORTANT: ALWAYS adhere to this exact format for tool use:\n<｜DSML｜function_calls>\n<｜DSML｜invoke name=\"tool_call_name\">\n<｜DSML｜param
@@ -41,20 +41,14 @@
+    {{'<｜DSML｜function_calls>'}}
-      {%- if not ns.is_first %}
-        {%- if message['content'] is none %}
```

- 已读文件:
  - runtime: `python/sglang/srt/function_call/deepseekv32_detector.py` modified +8/-4
  - docs: `examples/chat_template/tool_chat_template_deepseekv32.jinja` modified +6/-12
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/function_call/deepseekv32_detector.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #17558 - fix: Fixed the issue where "finish_reason":"stop" appeared when calling the tool and the tool was in the content.

- 链接: https://github.com/sgl-project/sglang/pull/17558
- 状态/时间: closed / 2026-01-24
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+14/-16，可读 patch 56 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix: Fixed the issue where "finish_reason":"stop" appeared when calling the tool and the tool was in the content.」；模型线: DeepSeek V3.1；类别: 缺陷修复；主要 diff: `python/sglang/srt/function_call/deepseekv32_detector.py`, `examples/chat_template/tool_chat_template_deepseekv32.jinja`；技术摘要: 覆盖「fix: Fixed the issue where "finish_reason":"stop" appeared when calling the tool and the tool was in the content.」；主要实现面是 `python/sglang/srt/function_call/deepseekv32_detector.py`, `examples/chat_template/tool_chat_template_deepseekv32.jinja`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/function_call/deepseekv32_detector.py` modified +8/-4 (12 lines); hunks: -195,12 +195,16 @@ def detect_and_parse(self, text: str, tools: list[Tool]) -...; symbols: detect_and_parse，涉及 `detect_and_parse`；`examples/chat_template/tool_chat_template_deepseekv32.jinja` modified +6/-12 (18 lines); hunks: -22,7 +22,7; -41,20 +41,14。
- 代码 diff 细节:
  - `python/sglang/srt/function_call/deepseekv32_detector.py` modified +8/-4 (12 lines); hunks: -195,12 +195,16 @@ def detect_and_parse(self, text: str, tools: list[Tool]) -...; symbols: detect_and_parse
  - `examples/chat_template/tool_chat_template_deepseekv32.jinja` modified +6/-12 (18 lines); hunks: -22,7 +22,7; -41,20 +41,14
- 关键代码摘录:

```diff
diff -- python/sglang/srt/function_call/deepseekv32_detector.py
@@ -195,12 +195,16 @@ def detect_and_parse(self, text: str, tools: list[Tool]) -> StreamingParseResult
-            for func_name, invoke_content, _ in invoke_matches:
+            for i, (func_name, invoke_content, _) in enumerate(invoke_matches):
-                # construct match_result for parse_base_json
-                match_result = {"name": func_name, "parameters": func_args}
-                calls.extend(self.parse_base_json(match_result, tools))
+                calls.append(
diff -- examples/chat_template/tool_chat_template_deepseekv32.jinja
@@ -22,7 +22,7 @@
-  {% set tool_ns.text = tool_ns.text + "\nIMPORTANT: ALWAYS adhere to this exact format for tool use:\n<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>tool_call_name<｜tool▁sep｜>tool_call_a
+  {% set tool_ns.text = tool_ns.text + "\nIMPORTANT: ALWAYS adhere to this exact format for tool use:\n<｜DSML｜function_calls>\n<｜DSML｜invoke name=\"tool_call_name\">\n<｜DSML｜param
@@ -41,20 +41,14 @@
+    {{'<｜DSML｜function_calls>'}}
-      {%- if not ns.is_first %}
-        {%- if message['content'] is none %}
```

- 已读文件:
  - runtime: `python/sglang/srt/function_call/deepseekv32_detector.py` modified +8/-4
  - docs: `examples/chat_template/tool_chat_template_deepseekv32.jinja` modified +6/-12
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/function_call/deepseekv32_detector.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #17761 - fix: missing Assistant token after tool output in DeepSeek v3.1/v3.2 chat templates

- 链接: https://github.com/sgl-project/sglang/pull/17761
- 状态/时间: open / 2026-01-26
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+79/-2，可读 patch 102 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix: missing Assistant token after tool output in DeepSeek v3.1/v3.2 chat templates」；模型线: DeepSeek V3.1；类别: 缺陷修复；主要 diff: `test/manual/test_deepseek_chat_templates.py`, `examples/chat_template/tool_chat_template_deepseekv31.jinja`, `examples/chat_template/tool_chat_template_deepseekv32.jinja`；技术摘要: 覆盖「fix: missing Assistant token after tool output in DeepSeek v3.1/v3.2 chat templates」；主要实现面是 `test/manual/test_deepseek_chat_templates.py`, `examples/chat_template/tool_chat_template_deepseekv31.jinja`, `examples/chat_template/tool_chat_template_deepseekv32.jinja`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/manual/test_deepseek_chat_templates.py` modified +77/-0 (77 lines); hunks: -313,6 +313,83 @@ def test_tool_call_with_content(self):; symbols: test_tool_call_with_content, test_assistant_marker_after_tool_output，涉及 `test_tool_call_with_content, test_assistant_marker_after_tool_output`；`examples/chat_template/tool_chat_template_deepseekv31.jinja` modified +1/-1 (2 lines); hunks: -60,7 +60,7；`examples/chat_template/tool_chat_template_deepseekv32.jinja` modified +1/-1 (2 lines); hunks: -57,7 +57,7。
- 代码 diff 细节:
  - `test/manual/test_deepseek_chat_templates.py` modified +77/-0 (77 lines); hunks: -313,6 +313,83 @@ def test_tool_call_with_content(self):; symbols: test_tool_call_with_content, test_assistant_marker_after_tool_output
  - `examples/chat_template/tool_chat_template_deepseekv31.jinja` modified +1/-1 (2 lines); hunks: -60,7 +60,7
  - `examples/chat_template/tool_chat_template_deepseekv32.jinja` modified +1/-1 (2 lines); hunks: -57,7 +57,7
- 关键代码摘录:

```diff
diff -- test/manual/test_deepseek_chat_templates.py
@@ -313,6 +313,83 @@ def test_tool_call_with_content(self):
+    def test_assistant_marker_after_tool_output(self):
+        """Test that Assistant marker is present after tool output in multi-turn conversation."""
+        # This tests that when an assistant responds after receiving tool output,
+        # the <｜Assistant｜> marker is correctly added
+        for version in ["v3.1", "v3.2"]:
+            with self.subTest(version=version):
diff -- examples/chat_template/tool_chat_template_deepseekv31.jinja
@@ -60,7 +60,7 @@
-    {%- if ns.is_last_user %}
+    {%- if ns.is_last_user or ns.is_tool %}
diff -- examples/chat_template/tool_chat_template_deepseekv32.jinja
@@ -57,7 +57,7 @@
-    {%- if ns.is_last_user %}
+    {%- if ns.is_last_user or ns.is_tool %}
```

- 已读文件:
  - tests: `test/manual/test_deepseek_chat_templates.py` modified +77/-0
  - docs: `examples/chat_template/tool_chat_template_deepseekv31.jinja` modified +1/-1; `examples/chat_template/tool_chat_template_deepseekv32.jinja` modified +1/-1
- 验证与风险: diff 自带测试面 `test/manual/test_deepseek_chat_templates.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #18236 - Fix function call arguments missing in streaming mode for DeepSeek V3.1

- 链接: https://github.com/sgl-project/sglang/pull/18236
- 状态/时间: open / 2026-02-04
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+21/-3，可读 patch 57 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix function call arguments missing in streaming mode for DeepSeek V3.1」；模型线: DeepSeek V3.1；类别: 缺陷修复；主要 diff: `python/sglang/srt/function_call/deepseekv31_detector.py`；技术摘要: 覆盖「Fix function call arguments missing in streaming mode for DeepSeek V3.1」；主要实现面是 `python/sglang/srt/function_call/deepseekv31_detector.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/function_call/deepseekv31_detector.py` modified +21/-3 (24 lines); hunks: -52,6 +52,7 @@ def __init__(self):; -111,6 +112,18 @@ def parse_streaming_increment(; symbols: __init__, has_tool_call, parse_streaming_increment，涉及 `__init__, has_tool_call, parse_streaming_increment`。
- 代码 diff 细节:
  - `python/sglang/srt/function_call/deepseekv31_detector.py` modified +21/-3 (24 lines); hunks: -52,6 +52,7 @@ def __init__(self):; -111,6 +112,18 @@ def parse_streaming_increment(; symbols: __init__, has_tool_call, parse_streaming_increment
- 关键代码摘录:

```diff
diff -- python/sglang/srt/function_call/deepseekv31_detector.py
@@ -52,6 +52,7 @@ def __init__(self):
+        self._normal_text_sent = False
@@ -111,6 +112,18 @@ def parse_streaming_increment(
+        # Extract normal text before tool call on first detection
+        normal_text_to_return = ""
+        if not self._normal_text_sent:
+            # Find the first tool call marker
```

- 已读文件:
  - runtime: `python/sglang/srt/function_call/deepseekv31_detector.py` modified +21/-3
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/function_call/deepseekv31_detector.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21739 - [NPU] Update DeepSeek-V3.1 and DeepSeek-V3.2 model deployment instructions in documentation

- 链接: https://github.com/sgl-project/sglang/pull/21739
- 状态/时间: open / 2026-03-31
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+163/-19，可读 patch 270 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU] Update DeepSeek-V3.1 and DeepSeek-V3.2 model deployment instructions in documentation」；模型线: DeepSeek V3.1；类别: 文档/测试/CI；主要 diff: `docs/platforms/ascend/ascend_npu_best_practice.md`；技术摘要: 覆盖「[NPU] Update DeepSeek-V3.1 and DeepSeek-V3.2 model deployment instructions in documentation」；主要实现面是 `docs/platforms/ascend/ascend_npu_best_practice.md`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs/platforms/ascend/ascend_npu_best_practice.md` modified +163/-19 (182 lines); hunks: -20,6 +20,7 @@ you encounter issues or have any questions, please [open an is...; -177,7 +178,148 @@ We tested it based on the `RANDOM` dataset.。
- 代码 diff 细节:
  - `docs/platforms/ascend/ascend_npu_best_practice.md` modified +163/-19 (182 lines); hunks: -20,6 +20,7 @@ you encounter issues or have any questions, please [open an is...; -177,7 +178,148 @@ We tested it based on the `RANDOM` dataset.
- 关键代码摘录:

```diff
diff -- docs/platforms/ascend/ascend_npu_best_practice.md
@@ -20,6 +20,7 @@ you encounter issues or have any questions, please [open an issue](https://githu
+| Deepseek-R1 | Atlas 800I A3 | 24    | PD Separation | 2K+2K     | 50ms | W8A8 INT8    | [Optimal Configuration](#deepseek-r1-2k-2k-50ms-on-a3-24-cards-separation-mode) |
@@ -177,7 +178,148 @@ We tested it based on the `RANDOM` dataset.
+### DeepSeek-R1 2K-2K 50ms on A3 24 Cards Separation Mode
+Model: Deepseek R1
+Hardware: Atlas 800I A3 24Card
+DeployMode: PD Separation
```

- 已读文件:
  - docs: `docs/platforms/ascend/ascend_npu_best_practice.md` modified +163/-19
- 验证与风险: 该 PR 主要落在文档/示例 `docs/platforms/ascend/ascend_npu_best_practice.md`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #22433 - [Test] Add unit tests for DeepSeekV31Detector

- 链接: https://github.com/sgl-project/sglang/pull/22433
- 状态/时间: open / 2026-04-09
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+314/-0，可读 patch 315 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Test] Add unit tests for DeepSeekV31Detector」；模型线: DeepSeek V3.1；类别: 文档/测试/CI；主要 diff: `test/registered/unit/function_call/test_deepseekv31_detector.py`；技术摘要: 覆盖「[Test] Add unit tests for DeepSeekV31Detector」；主要实现面是 `test/registered/unit/function_call/test_deepseekv31_detector.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/unit/function_call/test_deepseekv31_detector.py` added +314/-0 (314 lines); hunks: -0,0 +1,314; symbols: _wrap_single, _make_tools, TestDeepSeekV31DetectorHasToolCall, setUp，涉及 `_wrap_single, _make_tools, TestDeepSeekV31DetectorHasToolCall`。
- 代码 diff 细节:
  - `test/registered/unit/function_call/test_deepseekv31_detector.py` added +314/-0 (314 lines); hunks: -0,0 +1,314; symbols: _wrap_single, _make_tools, TestDeepSeekV31DetectorHasToolCall, setUp
- 关键代码摘录:

```diff
diff -- test/registered/unit/function_call/test_deepseekv31_detector.py
@@ -0,0 +1,314 @@
+"""Unit tests for DeepSeekV31Detector — no server, no model loading.
+Covers the DeepSeek V3.1 function-call format:
+    <｜tool▁calls▁begin｜>
+      <｜tool▁call▁begin｜>{name}<｜tool▁sep｜>{json_args}<｜tool▁call▁end｜>
+      ...
+    <｜tool▁calls▁end｜>
```

- 已读文件:
  - tests: `test/registered/unit/function_call/test_deepseekv31_detector.py` added +314/-0
- 验证与风险: diff 自带测试面 `test/registered/unit/function_call/test_deepseekv31_detector.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #21593 - Fix tool call constrained decoding and parsing for models with native formats

- 链接: https://github.com/sgl-project/sglang/pull/21593
- 状态/时间: merged / 2026-04-11
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+306/-61，可读 patch 516 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix tool call constrained decoding and parsing for models with native formats」；模型线: DeepSeek V3.1；类别: 缺陷修复；主要 diff: `test/registered/unit/function_call/test_function_call_parser.py`, `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/function_call/function_call_parser.py`；技术摘要: 覆盖「Fix tool call constrained decoding and parsing for models with native formats」；主要实现面是 `test/registered/unit/function_call/test_function_call_parser.py`, `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/function_call/function_call_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/unit/function_call/test_function_call_parser.py` modified +113/-0 (113 lines); hunks: -3859,6 +3859,119 @@ def test_streaming_function_call_marker_json_split_at_qu...; symbols: test_streaming_function_call_marker_json_split_at_quotes, TestGetStructureConstraint, _make_tools, _make_parser，涉及 `test_streaming_function_call_marker_json_split_at_quotes, TestGetStructureConstraint, _make_tools`；`python/sglang/srt/entrypoints/openai/serving_chat.py` modified +67/-43 (110 lines); hunks: -361,9 +361,11 @@ def _process_messages(; -1136,22 +1138,56 @@ def _process_tool_calls(; symbols: _process_messages, _process_tool_calls, _process_tool_call_stream，涉及 `_process_messages, _process_tool_calls, _process_tool_call_stream`；`python/sglang/srt/function_call/function_call_parser.py` modified +35/-11 (46 lines); hunks: -3,6 +3,7; -32,7 +33,10; symbols: parse_stream_chunk, get_structure_tag, get_structure_constraint，涉及 `parse_stream_chunk, get_structure_tag, get_structure_constraint`；`test/registered/openai_server/function_call/test_tool_choice.py` modified +8/-2 (10 lines); hunks: -348,8 +348,12 @@ def test_tool_choice_specific_function_streaming(self):; -406,13 +410,15 @@ def test_required_streaming_arguments_chunks_json(self):; symbols: test_tool_choice_specific_function_streaming, test_required_streaming_arguments_chunks_json, test_complex_parameters_required_non_streaming，涉及 `test_tool_choice_specific_function_streaming, test_required_streaming_arguments_chunks_json, test_complex_parameters_required_non_streaming`。
- 代码 diff 细节:
  - `test/registered/unit/function_call/test_function_call_parser.py` modified +113/-0 (113 lines); hunks: -3859,6 +3859,119 @@ def test_streaming_function_call_marker_json_split_at_qu...; symbols: test_streaming_function_call_marker_json_split_at_quotes, TestGetStructureConstraint, _make_tools, _make_parser
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +67/-43 (110 lines); hunks: -361,9 +361,11 @@ def _process_messages(; -1136,22 +1138,56 @@ def _process_tool_calls(; symbols: _process_messages, _process_tool_calls, _process_tool_call_stream
  - `python/sglang/srt/function_call/function_call_parser.py` modified +35/-11 (46 lines); hunks: -3,6 +3,7; -32,7 +33,10; symbols: parse_stream_chunk, get_structure_tag, get_structure_constraint
  - `test/registered/openai_server/function_call/test_tool_choice.py` modified +8/-2 (10 lines); hunks: -348,8 +348,12 @@ def test_tool_choice_specific_function_streaming(self):; -406,13 +410,15 @@ def test_required_streaming_arguments_chunks_json(self):; symbols: test_tool_choice_specific_function_streaming, test_required_streaming_arguments_chunks_json, test_complex_parameters_required_non_streaming
  - `python/sglang/srt/function_call/deepseekv3_detector.py` modified +5/-3 (8 lines); hunks: -203,7 +203,9 @@ def parse_streaming_increment(; symbols: parse_streaming_increment, structure_info
- 关键代码摘录:

```diff
diff -- test/registered/unit/function_call/test_function_call_parser.py
@@ -3859,6 +3859,119 @@ def test_streaming_function_call_marker_json_split_at_quotes(self):
+class TestGetStructureConstraint(unittest.TestCase):
+    """Tests for FunctionCallParser.get_structure_constraint() logic.
+    Verifies that detectors supporting structural_tag use it for required/named
+    tool_choice, and that the generic json_schema fallback is used otherwise.
+    """
+    def _make_tools(self, strict=False):
diff -- python/sglang/srt/entrypoints/openai/serving_chat.py
@@ -361,9 +361,11 @@ def _process_messages(
-            # Handle JSON schema constraint directly for required or named tool choice
-            if request.tool_choice == "required" or isinstance(
-                request.tool_choice, ToolChoice
+            # Fallback: use generic JSON schema for required/named tool choice
+            # only when no parser-specific constraint was set
+            if tool_call_constraint is None and (
diff -- python/sglang/srt/function_call/function_call_parser.py
@@ -3,6 +3,7 @@
```

- 已读文件:
  - tests: `test/registered/unit/function_call/test_function_call_parser.py` modified +113/-0; `test/registered/openai_server/function_call/test_tool_choice.py` modified +8/-2; `test/registered/openai_server/basic/test_serving_chat.py` modified +72/-0
  - runtime: `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +67/-43; `python/sglang/srt/function_call/function_call_parser.py` modified +35/-11; `python/sglang/srt/function_call/deepseekv3_detector.py` modified +5/-3; `python/sglang/srt/function_call/base_format_detector.py` modified +1/-1; `python/sglang/srt/entrypoints/openai/protocol.py` modified +1/-0
- 验证与风险: diff 自带测试面 `test/registered/openai_server/basic/test_serving_chat.py`, `test/registered/openai_server/function_call/test_tool_choice.py`, `test/registered/unit/function_call/test_function_call_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22981 - [Test] Add unit tests for 7 missing function call detectors

- 链接: https://github.com/sgl-project/sglang/pull/22981
- 状态/时间: open / 2026-04-16
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+1017/-1，可读 patch 1063 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Test] Add unit tests for 7 missing function call detectors」；模型线: DeepSeek V3.1；类别: 文档/测试/CI；主要 diff: `test/registered/unit/function_call/test_function_call_parser.py`, `test/registered/openai_server/function_call/test_tool_choice.py`, `test/registered/unit/function_call/test_kimik2_detector.py`；技术摘要: 覆盖「[Test] Add unit tests for 7 missing function call detectors」；主要实现面是 `test/registered/unit/function_call/test_function_call_parser.py`, `test/registered/openai_server/function_call/test_tool_choice.py`, `test/registered/unit/function_call/test_kimik2_detector.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/unit/function_call/test_function_call_parser.py` modified +960/-1 (961 lines); hunks: -2,9 +2,11; -15,16 +17,22; symbols: TestPythonicDetector, setUp, test_has_tool_call, test_detect_and_parse_single，涉及 `TestPythonicDetector, setUp, test_has_tool_call`；`test/registered/openai_server/function_call/test_tool_choice.py` modified +57/-0 (57 lines); hunks: -894,5 +894,62 @@ def setUpClass(cls):; symbols: setUpClass, TestToolChoiceWithConstrainedDecoding, test_tool_choice_required_strict_finish_reason，涉及 `setUpClass, TestToolChoiceWithConstrainedDecoding, test_tool_choice_required_strict_finish_reason`；`test/registered/unit/function_call/test_kimik2_detector.py` renamed +0/-0 (0 lines)。
- 代码 diff 细节:
  - `test/registered/unit/function_call/test_function_call_parser.py` modified +960/-1 (961 lines); hunks: -2,9 +2,11; -15,16 +17,22; symbols: TestPythonicDetector, setUp, test_has_tool_call, test_detect_and_parse_single
  - `test/registered/openai_server/function_call/test_tool_choice.py` modified +57/-0 (57 lines); hunks: -894,5 +894,62 @@ def setUpClass(cls):; symbols: setUpClass, TestToolChoiceWithConstrainedDecoding, test_tool_choice_required_strict_finish_reason
  - `test/registered/unit/function_call/test_kimik2_detector.py` renamed +0/-0 (0 lines)
- 关键代码摘录:

```diff
diff -- test/registered/unit/function_call/test_function_call_parser.py
@@ -2,9 +2,11 @@
+from sglang.srt.environ import envs
+from sglang.srt.function_call.deepseekv31_detector import DeepSeekV31Detector
@@ -15,16 +17,22 @@
+from sglang.srt.function_call.gpt_oss_detector import GptOssDetector
+from sglang.srt.function_call.internlm_detector import InternlmDetector
+from sglang.srt.function_call.mimo_detector import MiMoDetector
diff -- test/registered/openai_server/function_call/test_tool_choice.py
@@ -894,5 +894,62 @@ def setUpClass(cls):
+class TestToolChoiceWithConstrainedDecoding(TestToolChoiceLlama32):
+    """Test tool_choice with grammar backend (structural_tag + constrained decoding).
+    Verifies that tool_choice="required" with strict=True produces valid
+    tool calls when the grammar backend is enabled.
+    """
+    @classmethod
```

- 已读文件:
  - tests: `test/registered/unit/function_call/test_function_call_parser.py` modified +960/-1; `test/registered/openai_server/function_call/test_tool_choice.py` modified +57/-0; `test/registered/unit/function_call/test_kimik2_detector.py` renamed +0/-0
- 验证与风险: diff 自带测试面 `test/registered/openai_server/function_call/test_tool_choice.py`, `test/registered/unit/function_call/test_function_call_parser.py`, `test/registered/unit/function_call/test_kimik2_detector.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22128 - Allow piecewise CUDA graph with speculative decoding

- 链接: https://github.com/sgl-project/sglang/pull/22128
- 状态/时间: merged / 2026-04-17
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+272/-18，可读 patch 344 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Allow piecewise CUDA graph with speculative decoding」；模型线: DeepSeek V3.1；类别: 性能/后端优化；主要 diff: `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`, `python/sglang/srt/model_executor/model_runner.py`, `test/registered/piecewise_cuda_graph/test_pcg_with_speculative_decoding.py`；技术摘要: 覆盖「Allow piecewise CUDA graph with speculative decoding」；主要实现面是 `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`, `python/sglang/srt/model_executor/model_runner.py`, `test/registered/piecewise_cuda_graph/test_pcg_with_speculative_decoding.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` modified +10/-0 (10 lines); hunks: -417,6 +417,16 @@ def can_run(self, forward_batch: ForwardBatch):; symbols: can_run，涉及 `can_run`；`python/sglang/srt/model_executor/model_runner.py` modified +4/-0 (4 lines); hunks: -2554,6 +2554,10 @@ def init_piecewise_cuda_graphs(self):; symbols: init_piecewise_cuda_graphs，涉及 `init_piecewise_cuda_graphs`；`test/registered/piecewise_cuda_graph/test_pcg_with_speculative_decoding.py` added +243/-0 (243 lines); hunks: -0,0 +1,243; symbols: TestPCGWithMTP, setUpClass, tearDownClass, test_gsm8k，涉及 `TestPCGWithMTP, setUpClass, tearDownClass`；`python/sglang/srt/server_args.py` modified +15/-18 (33 lines); hunks: -1113,56 +1113,53 @@ def _handle_piecewise_cuda_graph(self):; symbols: _handle_piecewise_cuda_graph，涉及 `_handle_piecewise_cuda_graph`。
- 代码 diff 细节:
  - `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` modified +10/-0 (10 lines); hunks: -417,6 +417,16 @@ def can_run(self, forward_batch: ForwardBatch):; symbols: can_run
  - `python/sglang/srt/model_executor/model_runner.py` modified +4/-0 (4 lines); hunks: -2554,6 +2554,10 @@ def init_piecewise_cuda_graphs(self):; symbols: init_piecewise_cuda_graphs
  - `test/registered/piecewise_cuda_graph/test_pcg_with_speculative_decoding.py` added +243/-0 (243 lines); hunks: -0,0 +1,243; symbols: TestPCGWithMTP, setUpClass, tearDownClass, test_gsm8k
  - `python/sglang/srt/server_args.py` modified +15/-18 (33 lines); hunks: -1113,56 +1113,53 @@ def _handle_piecewise_cuda_graph(self):; symbols: _handle_piecewise_cuda_graph
- 关键代码摘录:

```diff
diff -- python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py
@@ -417,6 +417,16 @@ def can_run(self, forward_batch: ForwardBatch):
+        # PCG graphs are captured with ForwardMode.EXTEND and spec_info=None.
+        # TARGET_VERIFY has different spec_info and capture_hidden_mode,
+        # so it must not use PCG-captured graphs.
+        if forward_batch.forward_mode.is_target_verify():
+            return False
+        # PCG graphs are captured with the runner's capture_hidden_mode.
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -2554,6 +2554,10 @@ def init_piecewise_cuda_graphs(self):
+        # Draft models use decode CUDA graphs, not PCG
+        if self.is_draft_worker:
+            return
diff -- test/registered/piecewise_cuda_graph/test_pcg_with_speculative_decoding.py
@@ -0,0 +1,243 @@
+"""Test piecewise CUDA graph coexisting with speculative decoding.
+PCG handles prefill/extend path while speculative decoding (MTP/EAGLE3/STANDALONE/NGRAM)
+uses decode CUDA graphs. This test verifies they don't interfere with each other.
```

- 已读文件:
  - runtime: `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` modified +10/-0; `python/sglang/srt/model_executor/model_runner.py` modified +4/-0; `python/sglang/srt/server_args.py` modified +15/-18
  - tests: `test/registered/piecewise_cuda_graph/test_pcg_with_speculative_decoding.py` added +243/-0
- 验证与风险: diff 自带测试面 `test/registered/piecewise_cuda_graph/test_pcg_with_speculative_decoding.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #21599 - [SPEC][1/N] feat: add adaptive speculative_num_steps for EAGLE topk=1

- 链接: https://github.com/sgl-project/sglang/pull/21599
- 状态/时间: merged / 2026-04-20
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 13 个文件，+1296/-33，可读 patch 1579 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[SPEC][1/N] feat: add adaptive speculative_num_steps for EAGLE topk=1」；模型线: DeepSeek V3.1；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/model_executor/cuda_graph_runner.py`, `benchmark/bench_adaptive_speculative.py`, `test/registered/unit/spec/test_adaptive_spec_params.py`；技术摘要: 覆盖「[SPEC][1/N] feat: add adaptive speculative_num_steps for EAGLE topk=1」；主要实现面是 `python/sglang/srt/model_executor/cuda_graph_runner.py`, `benchmark/bench_adaptive_speculative.py`, `test/registered/unit/spec/test_adaptive_spec_params.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +26/-12 (38 lines); hunks: -512,7 +512,14 @@ def set_global_graph_memory_pool(val):; -551,6 +558,17 @@ def __init__(self, model_runner: ModelRunner):; symbols: set_global_graph_memory_pool, CudaGraphRunner, __init__，涉及 `set_global_graph_memory_pool, CudaGraphRunner, __init__`；`benchmark/bench_adaptive_speculative.py` added +263/-0 (263 lines); hunks: -0,0 +1,263; symbols: build_phase_plan, send_request, run_phase, summarize_phases，涉及 `build_phase_plan, send_request, run_phase`；`test/registered/unit/spec/test_adaptive_spec_params.py` added +195/-0 (195 lines); hunks: -0,0 +1,195; symbols: TestAdaptiveSpeculativeParams, test_initial_steps_snap_to_nearest_candidate_preferring_larger_step, test_update_respects_warmup_and_interval, test_empty_batches_do_not_consume_warmup_or_shift_steps，涉及 `TestAdaptiveSpeculativeParams, test_initial_steps_snap_to_nearest_candidate_preferring_larger_step, test_update_respects_warmup_and_interval`；`test/registered/spec/eagle/test_adaptive_speculative.py` added +170/-0 (170 lines); hunks: -0,0 +1,170; symbols: TestAdaptiveSpeculativeServer, setUpClass, tearDownClass, _get_internal_state，涉及 `TestAdaptiveSpeculativeServer, setUpClass, tearDownClass`。
- 代码 diff 细节:
  - `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +26/-12 (38 lines); hunks: -512,7 +512,14 @@ def set_global_graph_memory_pool(val):; -551,6 +558,17 @@ def __init__(self, model_runner: ModelRunner):; symbols: set_global_graph_memory_pool, CudaGraphRunner, __init__
  - `benchmark/bench_adaptive_speculative.py` added +263/-0 (263 lines); hunks: -0,0 +1,263; symbols: build_phase_plan, send_request, run_phase, summarize_phases
  - `test/registered/unit/spec/test_adaptive_spec_params.py` added +195/-0 (195 lines); hunks: -0,0 +1,195; symbols: TestAdaptiveSpeculativeParams, test_initial_steps_snap_to_nearest_candidate_preferring_larger_step, test_update_respects_warmup_and_interval, test_empty_batches_do_not_consume_warmup_or_shift_steps
  - `test/registered/spec/eagle/test_adaptive_speculative.py` added +170/-0 (170 lines); hunks: -0,0 +1,170; symbols: TestAdaptiveSpeculativeServer, setUpClass, tearDownClass, _get_internal_state
  - `python/sglang/srt/speculative/eagle_worker.py` modified +162/-4 (166 lines); hunks: -1,5 +1,6; -24,6 +25,7; symbols: __init__, init_cuda_graphs, apply_runtime_state, build_adaptive_runtime_state
- 关键代码摘录:

```diff
diff -- python/sglang/srt/model_executor/cuda_graph_runner.py
@@ -512,7 +512,14 @@ def set_global_graph_memory_pool(val):
-    def __init__(self, model_runner: ModelRunner):
+    def __init__(
+        self,
+        model_runner: ModelRunner,
+        *,
+        attn_backend=None,
diff -- benchmark/bench_adaptive_speculative.py
@@ -0,0 +1,263 @@
+"""Benchmark adaptive speculative decoding against static baselines.
+Run the same workload against one adaptive server and one or more static
+servers, then compare throughput, latency, and acceptance length.
+Workloads:
+- low: steady-state low-acceptance generation
+- high: steady-state high-acceptance generation
diff -- test/registered/unit/spec/test_adaptive_spec_params.py
@@ -0,0 +1,195 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +26/-12; `python/sglang/srt/speculative/eagle_worker.py` modified +162/-4; `python/sglang/srt/speculative/adaptive_spec_params.py` added +133/-0; `python/sglang/srt/speculative/adaptive_runtime_state.py` added +121/-0
  - other: `benchmark/bench_adaptive_speculative.py` added +263/-0
  - tests: `test/registered/unit/spec/test_adaptive_spec_params.py` added +195/-0; `test/registered/spec/eagle/test_adaptive_speculative.py` added +170/-0
  - docs: `docs/advanced_features/adaptive_speculative_decoding.md` added +156/-0
- 验证与风险: diff 自带测试面 `test/registered/spec/eagle/test_adaptive_speculative.py`, `test/registered/unit/spec/test_adaptive_spec_params.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #23315 - Opt-in strip of thinking tokens from radix cache

- 链接: https://github.com/sgl-project/sglang/pull/23315
- 状态/时间: merged / 2026-04-21
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+72/-4，可读 patch 131 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Opt-in strip of thinking tokens from radix cache」；模型线: DeepSeek V3.1；类别: 缺陷修复；主要 diff: `test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py`, `python/sglang/srt/managers/schedule_batch.py`, `python/sglang/srt/server_args.py`；技术摘要: 覆盖「Opt-in strip of thinking tokens from radix cache」；主要实现面是 `test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py`, `python/sglang/srt/managers/schedule_batch.py`, `python/sglang/srt/server_args.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py` modified +52/-1 (53 lines); hunks: -30,7 +30,11; -485,6 +489,53 @@ def test_cache_finished_req_insert(self):; symbols: test_cache_finished_req_insert, test_cache_finished_req_strips_thinking, test_cache_finished_req_no_insert，涉及 `test_cache_finished_req_insert, test_cache_finished_req_strips_thinking, test_cache_finished_req_no_insert`；`python/sglang/srt/managers/schedule_batch.py` modified +9/-2 (11 lines); hunks: -903,13 +903,20 @@ def output_ids_through_stop(self) -> List[int]:; -921,7 +928,7 @@ def pop_overallocated_kv_cache(self) -> Tuple[int, int]:; symbols: output_ids_through_stop, _cache_commit_len, pop_committed_kv_cache, pop_overallocated_kv_cache，涉及 `output_ids_through_stop, _cache_commit_len, pop_committed_kv_cache`；`python/sglang/srt/server_args.py` modified +8/-0 (8 lines); hunks: -436,6 +436,7 @@ class ServerArgs:; -4879,6 +4880,13 @@ def add_cli_args(parser: argparse.ArgumentParser):; symbols: ServerArgs, add_cli_args，涉及 `ServerArgs, add_cli_args`；`python/sglang/srt/mem_cache/common.py` modified +3/-1 (4 lines); hunks: -489,7 +489,9 @@ def release_kv_cache(req: Req, tree_cache: BasePrefixCache,...; symbols: release_kv_cache，涉及 `release_kv_cache`。
- 代码 diff 细节:
  - `test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py` modified +52/-1 (53 lines); hunks: -30,7 +30,11; -485,6 +489,53 @@ def test_cache_finished_req_insert(self):; symbols: test_cache_finished_req_insert, test_cache_finished_req_strips_thinking, test_cache_finished_req_no_insert
  - `python/sglang/srt/managers/schedule_batch.py` modified +9/-2 (11 lines); hunks: -903,13 +903,20 @@ def output_ids_through_stop(self) -> List[int]:; -921,7 +928,7 @@ def pop_overallocated_kv_cache(self) -> Tuple[int, int]:; symbols: output_ids_through_stop, _cache_commit_len, pop_committed_kv_cache, pop_overallocated_kv_cache
  - `python/sglang/srt/server_args.py` modified +8/-0 (8 lines); hunks: -436,6 +436,7 @@ class ServerArgs:; -4879,6 +4880,13 @@ def add_cli_args(parser: argparse.ArgumentParser):; symbols: ServerArgs, add_cli_args
  - `python/sglang/srt/mem_cache/common.py` modified +3/-1 (4 lines); hunks: -489,7 +489,9 @@ def release_kv_cache(req: Req, tree_cache: BasePrefixCache,...; symbols: release_kv_cache
- 关键代码摘录:

```diff
diff -- test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py
@@ -30,7 +30,11 @@
-from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
+from sglang.srt.server_args import (
+    ServerArgs,
+    get_global_server_args,
+    set_global_server_args_for_scheduler,
+)
diff -- python/sglang/srt/managers/schedule_batch.py
@@ -903,13 +903,20 @@ def output_ids_through_stop(self) -> List[int]:
+    def _cache_commit_len(self) -> int:
+        # Report only the prompt prefix so thinking + answer fall into the
+        # overallocated range and are reclaimed by release_kv_cache. #22373.
+        if get_global_server_args().strip_thinking_cache and self.reasoning_tokens > 0:
+            return min(self.kv_committed_len, len(self.origin_input_ids))
+        return self.kv_committed_len
diff -- python/sglang/srt/server_args.py
@@ -436,6 +436,7 @@ class ServerArgs:
```

- 已读文件:
  - tests: `test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py` modified +52/-1
  - runtime: `python/sglang/srt/managers/schedule_batch.py` modified +9/-2; `python/sglang/srt/server_args.py` modified +8/-0; `python/sglang/srt/mem_cache/common.py` modified +3/-1
- 验证与风险: diff 自带测试面 `test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22950 - [fix] Parser-gated two-phase cache stripping for reasoning radix caches (fixes #22373)

- 链接: https://github.com/sgl-project/sglang/pull/22950
- 状态/时间: closed / 2026-04-21
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 11 个文件，+597/-64，可读 patch 850 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[fix] Parser-gated two-phase cache stripping for reasoning radix caches (fixes #22373)」；模型线: DeepSeek V3.1；类别: 缺陷修复；主要 diff: `python/sglang/srt/parser/reasoning_parser.py`, `python/sglang/srt/configs/model_config.py`, `test/registered/unit/mem_cache/test_radix_cache_thinking.py`；技术摘要: 覆盖「[fix] Parser-gated two-phase cache stripping for reasoning radix caches (fixes #22373)」；主要实现面是 `python/sglang/srt/parser/reasoning_parser.py`, `python/sglang/srt/configs/model_config.py`, `test/registered/unit/mem_cache/test_radix_cache_thinking.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/parser/reasoning_parser.py` modified +8/-0 (8 lines); hunks: -19,6 +19,10 @@ def __init__(; -395,6 +399,10 @@ class MiniMaxAppendThinkDetector(BaseReasoningFormatDetector):; symbols: __init__, BaseReasoningFormatDetector, providing, MiniMaxAppendThinkDetector，涉及 `__init__, BaseReasoningFormatDetector, providing`；`python/sglang/srt/configs/model_config.py` modified +1/-0 (1 lines); hunks: -242,6 +242,7 @@ def __init__(; symbols: __init__，涉及 `__init__`；`test/registered/unit/mem_cache/test_radix_cache_thinking.py` added +238/-0 (238 lines); hunks: -0,0 +1,238; symbols: _MockReqToTokenPool, __init__, write, _MockAllocator，涉及 `_MockReqToTokenPool, __init__, write`；`test/registered/unit/mem_cache/test_radix_cache_thinking_gated.py` added +220/-0 (220 lines); hunks: -0,0 +1,220; symbols: _MockReqToTokenPool, __init__, write, _MockAllocator，涉及 `_MockReqToTokenPool, __init__, write`。
- 代码 diff 细节:
  - `python/sglang/srt/parser/reasoning_parser.py` modified +8/-0 (8 lines); hunks: -19,6 +19,10 @@ def __init__(; -395,6 +399,10 @@ class MiniMaxAppendThinkDetector(BaseReasoningFormatDetector):; symbols: __init__, BaseReasoningFormatDetector, providing, MiniMaxAppendThinkDetector
  - `python/sglang/srt/configs/model_config.py` modified +1/-0 (1 lines); hunks: -242,6 +242,7 @@ def __init__(; symbols: __init__
  - `test/registered/unit/mem_cache/test_radix_cache_thinking.py` added +238/-0 (238 lines); hunks: -0,0 +1,238; symbols: _MockReqToTokenPool, __init__, write, _MockAllocator
  - `test/registered/unit/mem_cache/test_radix_cache_thinking_gated.py` added +220/-0 (220 lines); hunks: -0,0 +1,220; symbols: _MockReqToTokenPool, __init__, write, _MockAllocator
  - `python/sglang/srt/mem_cache/mamba_radix_cache.py` modified +62/-50 (112 lines); hunks: -28,7 +28,6; -45,6 +44,7; symbols: cache_finished_req, _skip_cache_unfinished_req
- 关键代码摘录:

```diff
diff -- python/sglang/srt/parser/reasoning_parser.py
@@ -19,6 +19,10 @@ def __init__(
+    # Most reasoning parsers separate hidden thinking from visible assistant
+    # content, so those tokens should not be cached across turns.
+    strip_thinking_from_cache: bool = True
@@ -395,6 +399,10 @@ class MiniMaxAppendThinkDetector(BaseReasoningFormatDetector):
+    # MiniMax appends thinking into visible assistant content, so future turns
+    # may include it verbatim and the full output should stay cacheable.
diff -- python/sglang/srt/configs/model_config.py
@@ -242,6 +242,7 @@ def __init__(
+        self.strip_thinking_from_cache: bool = True
diff -- test/registered/unit/mem_cache/test_radix_cache_thinking.py
@@ -0,0 +1,238 @@
+import unittest
+import torch
+from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
+from sglang.srt.mem_cache.cache_init_params import CacheInitParams
+from sglang.srt.mem_cache.common import maybe_strip_thinking_tokens
```

- 已读文件:
  - runtime: `python/sglang/srt/parser/reasoning_parser.py` modified +8/-0; `python/sglang/srt/configs/model_config.py` modified +1/-0; `python/sglang/srt/mem_cache/mamba_radix_cache.py` modified +62/-50; `python/sglang/srt/mem_cache/radix_cache_cpp.py` modified +27/-14; `python/sglang/srt/mem_cache/common.py` modified +22/-0; `python/sglang/srt/mem_cache/radix_cache.py` modified +7/-0
  - tests: `test/registered/unit/mem_cache/test_radix_cache_thinking.py` added +238/-0; `test/registered/unit/mem_cache/test_radix_cache_thinking_gated.py` added +220/-0
- 验证与风险: diff 自带测试面 `test/registered/unit/mem_cache/test_radix_cache_thinking.py`, `test/registered/unit/mem_cache/test_radix_cache_thinking_gated.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #23336 - [SPEC V2][2/N] feat: adaptive spec support spec v2

- 链接: https://github.com/sgl-project/sglang/pull/23336
- 状态/时间: open / 2026-04-21
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+193/-10，可读 patch 290 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[SPEC V2][2/N] feat: adaptive spec support spec v2」；模型线: DeepSeek V3.1；类别: 性能/后端优化；主要 diff: `python/sglang/srt/speculative/eagle_worker_v2.py`, `python/sglang/srt/speculative/eagle_info_v2.py`, `python/sglang/srt/managers/scheduler_output_processor_mixin.py`；技术摘要: 覆盖「[SPEC V2][2/N] feat: adaptive spec support spec v2」；主要实现面是 `python/sglang/srt/speculative/eagle_worker_v2.py`, `python/sglang/srt/speculative/eagle_info_v2.py`, `python/sglang/srt/managers/scheduler_output_processor_mixin.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/speculative/eagle_worker_v2.py` modified +173/-0 (173 lines); hunks: -30,8 +30,13; -671,6 +676,13 @@ def __init__(; symbols: __init__, target_worker, forward_batch_generation, on_verify_complete_cpu，涉及 `__init__, target_worker, forward_batch_generation`；`python/sglang/srt/speculative/eagle_info_v2.py` modified +8/-4 (12 lines); hunks: -114,14 +114,18 @@ def prepare_for_decode(self: EagleDraftInput, batch: Sched...; -163,7 +167,7 @@ def prepare_for_decode(self: EagleDraftInput, batch: Schedul...; symbols: prepare_for_decode, prepare_for_v2_draft，涉及 `prepare_for_decode, prepare_for_v2_draft`；`python/sglang/srt/managers/scheduler_output_processor_mixin.py` modified +10/-1 (11 lines); hunks: -360,8 +360,17 @@ def _resolve_spec_overlap_token_ids(; symbols: _resolve_spec_overlap_token_ids，涉及 `_resolve_spec_overlap_token_ids`；`python/sglang/srt/speculative/adaptive_spec_params.py` modified +0/-5 (5 lines); hunks: -32,11 +32,6 @@ def adaptive_unsupported_reason(server_args: ServerArgs) -> s...; symbols: adaptive_unsupported_reason，涉及 `adaptive_unsupported_reason`。
- 代码 diff 细节:
  - `python/sglang/srt/speculative/eagle_worker_v2.py` modified +173/-0 (173 lines); hunks: -30,8 +30,13; -671,6 +676,13 @@ def __init__(; symbols: __init__, target_worker, forward_batch_generation, on_verify_complete_cpu
  - `python/sglang/srt/speculative/eagle_info_v2.py` modified +8/-4 (12 lines); hunks: -114,14 +114,18 @@ def prepare_for_decode(self: EagleDraftInput, batch: Sched...; -163,7 +167,7 @@ def prepare_for_decode(self: EagleDraftInput, batch: Schedul...; symbols: prepare_for_decode, prepare_for_v2_draft
  - `python/sglang/srt/managers/scheduler_output_processor_mixin.py` modified +10/-1 (11 lines); hunks: -360,8 +360,17 @@ def _resolve_spec_overlap_token_ids(; symbols: _resolve_spec_overlap_token_ids
  - `python/sglang/srt/speculative/adaptive_spec_params.py` modified +0/-5 (5 lines); hunks: -32,11 +32,6 @@ def adaptive_unsupported_reason(server_args: ServerArgs) -> s...; symbols: adaptive_unsupported_reason
  - `python/sglang/srt/managers/utils.py` modified +1/-0 (1 lines); hunks: -27,6 +27,7 @@ class GenerationBatchResult:; symbols: GenerationBatchResult
- 关键代码摘录:

```diff
diff -- python/sglang/srt/speculative/eagle_worker_v2.py
@@ -30,8 +30,13 @@
+from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
+from sglang.srt.speculative.adaptive_runtime_state import (
+    AdaptiveController,
+    SpecRuntimeState,
+)
@@ -671,6 +676,13 @@ def __init__(
diff -- python/sglang/srt/speculative/eagle_info_v2.py
@@ -114,14 +114,18 @@ def prepare_for_decode(self: EagleDraftInput, batch: ScheduleBatch):
+        current_kv_lens_cpu = batch.seq_lens.to(device="cpu")
-        for r in batch.reqs:
-            # Over-allocation happens here
-            x = r.kv_committed_len + 2 * alloc_len_per_decode - r.kv_allocated_len
+        for i, r in enumerate(batch.reqs):
+            cur_kv_len = current_kv_lens_cpu[i].item()
diff -- python/sglang/srt/managers/scheduler_output_processor_mixin.py
@@ -360,8 +360,17 @@ def _resolve_spec_overlap_token_ids(
```

- 已读文件:
  - runtime: `python/sglang/srt/speculative/eagle_worker_v2.py` modified +173/-0; `python/sglang/srt/speculative/eagle_info_v2.py` modified +8/-4; `python/sglang/srt/managers/scheduler_output_processor_mixin.py` modified +10/-1; `python/sglang/srt/speculative/adaptive_spec_params.py` modified +0/-5; `python/sglang/srt/managers/utils.py` modified +1/-0; `python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py` modified +1/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/managers/scheduler_output_processor_mixin.py`, `python/sglang/srt/managers/utils.py`, `python/sglang/srt/speculative/adaptive_spec_params.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
