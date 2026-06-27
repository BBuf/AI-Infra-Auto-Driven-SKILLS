# vllm Llama 3.1 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `examples/tool_chat_template_llama3.1_json.jinja` | [#8343](https://github.com/vllm-project/vllm/pull/8343) |

## PR 覆盖总览

- git 追溯 PR 数: 1
- 原文档显式引用补充 PR 数: 18
- 当前文档总 PR 数: 19
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2024-09-27 | [#8343](https://github.com/vllm-project/vllm/pull/8343) | merged | [Feature] Add support for Llama 3.1 and 3.2 tool use | `examples/tool_chat_template_llama3.1_json.jinja` |
| 2024-11-23 | [#10164](https://github.com/vllm-project/vllm/pull/10164) | merged | [Bugfix][Frontend] Update Llama Chat Templates to also support Non-Tool use | `tests/entrypoints/test_chat_utils.py`, `examples/tool_chat_template_llama3.2_json.jinja`, `examples/tool_chat_template_llama3.1_json.jinja` |
| 2025-10-30 | [#25786](https://github.com/vllm-project/vllm/pull/25786) | merged | [Benchmark] Cleanup deprecated nightly benchmark and adjust the docstring for performance benchmark | `.buildkite/nightly-benchmarks/scripts/download-tokenizer.py`, `.buildkite/nightly-benchmarks/scripts/run-nightly-benchmarks.sh`, `.buildkite/nightly-benchmarks/nightly-pipeline.yaml` |
| 2026-02-06 | [#33731](https://github.com/vllm-project/vllm/pull/33731) | merged | [torch.compile] Reorganize vllm/compilation and tests/compile (0/N for vLLM IR) | `vllm/compilation/passes/fusion/collective_fusion.py`, `vllm/compilation/passes/fusion/allreduce_rms_fusion.py`, `tests/compile/passes/distributed/test_async_tp.py` |
| 2026-02-12 | [#34128](https://github.com/vllm-project/vllm/pull/34128) | merged | Vllm CPU benchmark suite improvement | `.buildkite/performance-benchmarks/scripts/compare-json-results.py`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json`, `.buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh` |
| 2026-02-17 | [#34716](https://github.com/vllm-project/vllm/pull/34716) | merged | [BugFix] Fix sp tests | `tests/compile/correctness_e2e/test_sequence_parallel.py` |
| 2026-03-04 | [#35871](https://github.com/vllm-project/vllm/pull/35871) | merged | [CI] Add Blackwell AsyncTP correctness test | `.buildkite/test_areas/compile.yaml`, `tests/compile/correctness_e2e/test_async_tp.py` |
| 2026-03-07 | [#36216](https://github.com/vllm-project/vllm/pull/36216) | merged | [V0 Deprecation] Remove unused swap_space parameter | `vllm/entrypoints/llm.py`, `vllm/config/cache.py`, `docs/design/metrics.md` |
| 2026-03-12 | [#35086](https://github.com/vllm-project/vllm/pull/35086) | merged | more models for vLLM Benchmark Suite | `.buildkite/performance-benchmarks/scripts/compare-json-results.py`, `.buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json` |
| 2026-03-31 | [#38576](https://github.com/vllm-project/vllm/pull/38576) | merged | vLLM Benchmark Suite perf regression after PR#32723 | `.buildkite/performance-benchmarks/tests/serving-tests-hpu.json`, `.buildkite/performance-benchmarks/tests/serving-tests.json`, `.buildkite/performance-benchmarks/tests/serving-tests-arm64-cpu.json` |
| 2026-04-26 | [#38373](https://github.com/vllm-project/vllm/pull/38373) | merged | [torch.compile]: Disable Sequence Parallelism (SP) for piecewise compilation | `tests/compile/test_config.py`, `vllm/config/vllm.py`, `vllm/compilation/passes/fusion/sequence_parallelism.py` |
| 2026-05-10 | [#41882](https://github.com/vllm-project/vllm/pull/41882) | merged | Add NVFP4 all-gather GEMM fusion for AsyncTP | `vllm/compilation/passes/fusion/collective_fusion.py`, `vllm/compilation/passes/fusion/sequence_parallelism.py`, `tests/compile/correctness_e2e/test_async_tp.py` |
| 2026-05-10 | [#33322](https://github.com/vllm-project/vllm/pull/33322) | merged | [Bugfix] Fix SP pass for multimodal models and PP+SP residual handling | `vllm/compilation/passes/fusion/sequence_parallelism.py`, `tests/compile/correctness_e2e/test_sequence_parallel.py`, `vllm/v1/worker/gpu_model_runner.py` |
| 2026-05-10 | [#42197](https://github.com/vllm-project/vllm/pull/42197) | merged | Fix mypy failure on main | `tests/compile/correctness_e2e/test_sequence_parallel.py` |
| 2026-05-15 | [#42607](https://github.com/vllm-project/vllm/pull/42607) | merged | Update Intel Xeon model list and vLLM Benchmark Suite BKMs | `docs/models/hardware_supported_models/cpu.md`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json` |
| 2026-05-21 | [#43262](https://github.com/vllm-project/vllm/pull/43262) | merged | update GPU json file based on h200 recipes | `.buildkite/performance-benchmarks/tests/serving-tests.json` |
| 2026-05-23 | [#43233](https://github.com/vllm-project/vllm/pull/43233) | merged | [Model Runner v2] Force v1 runner for tests | `tests/models/quantization/test_bitsandbytes.py`, `tests/compile/correctness_e2e/test_async_tp.py`, `tests/utils.py` |
| 2026-06-03 | [#44128](https://github.com/vllm-project/vllm/pull/44128) | merged | [Misc] Remove dead VLLM_RPC_TIMEOUT env var and fix profiling doc that references it | `vllm/envs.py`, `docs/contributing/profiling.md`, `.buildkite/performance-benchmarks/tests/latency-tests-arm64-cpu.json` |
| 2026-06-11 | [#44992](https://github.com/vllm-project/vllm/pull/44992) | merged | Deprecations for v0.23 and v0.24 | `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`, `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py`, `vllm/model_executor/kernels/linear/__init__.py` |

## 逐 PR diff 审计卡

### PR #8343 - [Feature] Add support for Llama 3.1 and 3.2 tool use

- 链接: https://github.com/vllm-project/vllm/pull/8343
- 状态/时间: merged / 2024-09-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `examples/tool_chat_template_llama3.1_json.jinja`；关联提交 `344cd2b6f4c2`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+576/-27，可读 patch 741 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Feature] Add support for Llama 3.1 and 3.2 tool use」；模型线: Llama 3.1；类别: 模型支持/运行时入口；主要 diff: `examples/tool_chat_template_llama3.1_json.jinja`；技术摘要: 覆盖「[Feature] Add support for Llama 3.1 and 3.2 tool use」；主要实现面是 `examples/tool_chat_template_llama3.1_json.jinja`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `examples/tool_chat_template_llama3.1_json.jinja` added +94/-0 (94 lines); hunks: -0,0 +1,94。
- 代码 diff 细节:
  - `examples/tool_chat_template_llama3.1_json.jinja` added +94/-0 (94 lines); hunks: -0,0 +1,94
- 关键代码摘录:

```diff
diff -- examples/tool_chat_template_llama3.1_json.jinja
@@ -0,0 +1,94 @@
+{{- bos_token }}
+{%- if custom_tools is defined %}
+    {%- set tools = custom_tools %}
+{%- endif %}
+{%- if not tools_in_user_message is defined %}
+    {#- Llama 3.1 doesn't pass all tests if the tools are in the system prompt #}
```

- 已读文件:
  - docs: `examples/tool_chat_template_llama3.1_json.jinja` added +94/-0
- 验证与风险: diff 自带测试面 `tests/tool_use/test_chat_completions.py`, `tests/tool_use/test_parallel_tool_calls.py`, `tests/tool_use/utils.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #10164 - [Bugfix][Frontend] Update Llama Chat Templates to also support Non-Tool use

- 链接: https://github.com/vllm-project/vllm/pull/10164
- 状态/时间: merged / 2024-11-23
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+110/-36，可读 patch 240 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix][Frontend] Update Llama Chat Templates to also support Non-Tool use」；模型线: Llama 3.1；类别: 缺陷修复；主要 diff: `tests/entrypoints/test_chat_utils.py`, `examples/tool_chat_template_llama3.2_json.jinja`, `examples/tool_chat_template_llama3.1_json.jinja`；技术摘要: 覆盖「[Bugfix][Frontend] Update Llama Chat Templates to also support Non-Tool use」；主要实现面是 `tests/entrypoints/test_chat_utils.py`, `examples/tool_chat_template_llama3.2_json.jinja`, `examples/tool_chat_template_llama3.1_json.jinja`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/entrypoints/test_chat_utils.py` modified +2/-2 (4 lines); hunks: -766,8 +766,8 @@ def test_resolve_content_format_hf_defined(model, expected_f...; symbols: test_resolve_content_format_hf_defined，涉及 `test_resolve_content_format_hf_defined`；`examples/tool_chat_template_llama3.2_json.jinja` modified +72/-24 (96 lines); hunks: -16,46 +16,78; -66,7 +98,19；`examples/tool_chat_template_llama3.1_json.jinja` modified +36/-10 (46 lines); hunks: -19,10 +19,18; -33,8 +41,8。
- 代码 diff 细节:
  - `tests/entrypoints/test_chat_utils.py` modified +2/-2 (4 lines); hunks: -766,8 +766,8 @@ def test_resolve_content_format_hf_defined(model, expected_f...; symbols: test_resolve_content_format_hf_defined
  - `examples/tool_chat_template_llama3.2_json.jinja` modified +72/-24 (96 lines); hunks: -16,46 +16,78; -66,7 +98,19
  - `examples/tool_chat_template_llama3.1_json.jinja` modified +36/-10 (46 lines); hunks: -19,10 +19,18; -33,8 +41,8
- 关键代码摘录:

```diff
diff -- tests/entrypoints/test_chat_utils.py
@@ -766,8 +766,8 @@ def test_resolve_content_format_hf_defined(model, expected_format):
-     ("tool_chat_template_llama3.1_json.jinja", "string"),
-     ("tool_chat_template_llama3.2_json.jinja", "string"),
+     ("tool_chat_template_llama3.1_json.jinja", "openai"),
+     ("tool_chat_template_llama3.2_json.jinja", "openai"),
diff -- examples/tool_chat_template_llama3.2_json.jinja
@@ -16,46 +16,78 @@
+{#- Find out if there are any images #}
+{% set image_ns = namespace(has_images=false) %}
+{%- for message in messages %}
+    {%- for content in message['content'] %}
+        {%- if content['type'] == 'image' %}
+            {%- set image_ns.has_images = true %}
diff -- examples/tool_chat_template_llama3.1_json.jinja
@@ -19,10 +19,18 @@
-    {%- set system_message = messages[0]['content']|trim %}
+    {%- if messages[0]['content'] is string %}
```

- 已读文件:
  - tests: `tests/entrypoints/test_chat_utils.py` modified +2/-2
  - docs: `examples/tool_chat_template_llama3.2_json.jinja` modified +72/-24; `examples/tool_chat_template_llama3.1_json.jinja` modified +36/-10
- 验证与风险: diff 自带测试面 `tests/entrypoints/test_chat_utils.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #25786 - [Benchmark] Cleanup deprecated nightly benchmark and adjust the docstring for performance benchmark

- 链接: https://github.com/vllm-project/vllm/pull/25786
- 状态/时间: merged / 2025-10-30
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 29 个文件，+10/-1289，可读 patch 1387 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Benchmark] Cleanup deprecated nightly benchmark and adjust the docstring for performance benchmark」；模型线: Llama 3.1；类别: 性能/后端优化；主要 diff: `.buildkite/nightly-benchmarks/scripts/download-tokenizer.py`, `.buildkite/nightly-benchmarks/scripts/run-nightly-benchmarks.sh`, `.buildkite/nightly-benchmarks/nightly-pipeline.yaml`；技术摘要: 覆盖「[Benchmark] Cleanup deprecated nightly benchmark and adjust the docstring for performance benchmark」；主要实现面是 `.buildkite/nightly-benchmarks/scripts/download-tokenizer.py`, `.buildkite/nightly-benchmarks/scripts/run-nightly-benchmarks.sh`, `.buildkite/nightly-benchmarks/nightly-pipeline.yaml`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `.buildkite/nightly-benchmarks/scripts/download-tokenizer.py` removed +0/-26 (26 lines); hunks: -1,26 +0,0; symbols: main，涉及 `main`；`.buildkite/nightly-benchmarks/scripts/run-nightly-benchmarks.sh` removed +0/-464 (464 lines); hunks: -1,464 +0,0；`.buildkite/nightly-benchmarks/nightly-pipeline.yaml` removed +0/-196 (196 lines); hunks: -1,196 +0,0；`.buildkite/nightly-benchmarks/benchmark-pipeline.yaml` removed +0/-184 (184 lines); hunks: -1,184 +0,0。
- 代码 diff 细节:
  - `.buildkite/nightly-benchmarks/scripts/download-tokenizer.py` removed +0/-26 (26 lines); hunks: -1,26 +0,0; symbols: main
  - `.buildkite/nightly-benchmarks/scripts/run-nightly-benchmarks.sh` removed +0/-464 (464 lines); hunks: -1,464 +0,0
  - `.buildkite/nightly-benchmarks/nightly-pipeline.yaml` removed +0/-196 (196 lines); hunks: -1,196 +0,0
  - `.buildkite/nightly-benchmarks/benchmark-pipeline.yaml` removed +0/-184 (184 lines); hunks: -1,184 +0,0
  - `.buildkite/nightly-benchmarks/scripts/generate-nightly-markdown.py` removed +0/-97 (97 lines); hunks: -1,97 +0,0; symbols: parse_arguments, get_perf, get_perf_w_std, main
- 关键代码摘录:

```diff
diff -- .buildkite/nightly-benchmarks/scripts/download-tokenizer.py
@@ -1,26 +0,0 @@
-# SPDX-License-Identifier: Apache-2.0
-# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-import argparse
-from transformers import AutoTokenizer
-def main(model, cachedir):
-    # Load the tokenizer and save it to the specified directory
diff -- .buildkite/nightly-benchmarks/scripts/run-nightly-benchmarks.sh
@@ -1,464 +0,0 @@
-#!/bin/bash
-set -o pipefail
-set -x
-check_gpus() {
-  # check the number of GPUs and GPU type.
-  declare -g gpu_count=$(nvidia-smi --list-gpus | wc -l)
diff -- .buildkite/nightly-benchmarks/nightly-pipeline.yaml
@@ -1,196 +0,0 @@
```

- 已读文件:
  - runtime: `.buildkite/nightly-benchmarks/scripts/download-tokenizer.py` removed +0/-26
  - other: `.buildkite/nightly-benchmarks/scripts/run-nightly-benchmarks.sh` removed +0/-464; `.buildkite/nightly-benchmarks/nightly-pipeline.yaml` removed +0/-196; `.buildkite/nightly-benchmarks/benchmark-pipeline.yaml` removed +0/-184; `.buildkite/nightly-benchmarks/scripts/generate-nightly-markdown.py` removed +0/-97; `.buildkite/nightly-benchmarks/scripts/summary-nightly-results.py` removed +0/-82; `.buildkite/nightly-benchmarks/scripts/nightly-annotate.sh` removed +0/-78
- 验证与风险: diff 自带测试面 `.buildkite/performance-benchmarks/tests/genai-perf-tests.json`, `.buildkite/performance-benchmarks/tests/latency-tests-cpu.json`, `.buildkite/performance-benchmarks/tests/latency-tests.json`, `.buildkite/performance-benchmarks/tests/nightly-tests.json`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #33731 - [torch.compile] Reorganize vllm/compilation and tests/compile (0/N for vLLM IR)

- 链接: https://github.com/vllm-project/vllm/pull/33731
- 状态/时间: merged / 2026-02-06
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 47 个文件，+717/-651，可读 patch 1985 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[torch.compile] Reorganize vllm/compilation and tests/compile (0/N for vLLM IR)」；模型线: Llama 3.1；类别: 文档/测试/CI；主要 diff: `vllm/compilation/passes/fusion/collective_fusion.py`, `vllm/compilation/passes/fusion/allreduce_rms_fusion.py`, `tests/compile/passes/distributed/test_async_tp.py`；技术摘要: 覆盖「[torch.compile] Reorganize vllm/compilation and tests/compile (0/N for vLLM IR)」；主要实现面是 `vllm/compilation/passes/fusion/collective_fusion.py`, `vllm/compilation/passes/fusion/allreduce_rms_fusion.py`, `tests/compile/passes/distributed/test_async_tp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/compilation/passes/fusion/collective_fusion.py` added +423/-0 (423 lines); hunks: -0,0 +1,423; symbols: BasePattern, __init__, GEMMReduceScatterPattern, get_inputs，涉及 `BasePattern, __init__, GEMMReduceScatterPattern`；`vllm/compilation/passes/fusion/allreduce_rms_fusion.py` renamed +13/-403 (416 lines); hunks: -8,7 +8,6; -24,12 +23,14; symbols: BasePattern, __init__, GEMMReduceScatterPattern, get_inputs，涉及 `BasePattern, __init__, GEMMReduceScatterPattern`；`tests/compile/passes/distributed/test_async_tp.py` renamed +5/-75 (80 lines); hunks: -1,16 +1,18; -29,14 +31,6; symbols: async_tp_pass_on_test_model, test_async_tp_pass_correctness，涉及 `async_tp_pass_on_test_model, test_async_tp_pass_correctness`；`tests/compile/correctness_e2e/test_async_tp.py` added +79/-0 (79 lines); hunks: -0,0 +1,79; symbols: test_async_tp_pass_correctness，涉及 `test_async_tp_pass_correctness`。
- 代码 diff 细节:
  - `vllm/compilation/passes/fusion/collective_fusion.py` added +423/-0 (423 lines); hunks: -0,0 +1,423; symbols: BasePattern, __init__, GEMMReduceScatterPattern, get_inputs
  - `vllm/compilation/passes/fusion/allreduce_rms_fusion.py` renamed +13/-403 (416 lines); hunks: -8,7 +8,6; -24,12 +23,14; symbols: BasePattern, __init__, GEMMReduceScatterPattern, get_inputs
  - `tests/compile/passes/distributed/test_async_tp.py` renamed +5/-75 (80 lines); hunks: -1,16 +1,18; -29,14 +31,6; symbols: async_tp_pass_on_test_model, test_async_tp_pass_correctness
  - `tests/compile/correctness_e2e/test_async_tp.py` added +79/-0 (79 lines); hunks: -0,0 +1,79; symbols: test_async_tp_pass_correctness
  - `.buildkite/test_areas/compile.yaml` modified +24/-18 (42 lines); hunks: -2,7 +2,7 @@ group: Compile; -11,37 +11,43 @@ steps:
- 关键代码摘录:

```diff
diff -- vllm/compilation/passes/fusion/collective_fusion.py
@@ -0,0 +1,423 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import torch
+import torch._inductor.pattern_matcher as pm
+import torch.fx as fx
+from torch._inductor.pattern_matcher import PatternMatcherPass
diff -- vllm/compilation/passes/fusion/allreduce_rms_fusion.py
@@ -8,7 +8,6 @@
-from torch.distributed._symmetric_memory import enable_symm_mem_for_group
@@ -24,12 +23,14 @@
-from .inductor_pass import enable_fake_mode
+from ..inductor_pass import enable_fake_mode
+from ..vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass
-from .vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass
diff -- tests/compile/passes/distributed/test_async_tp.py
@@ -1,16 +1,18 @@
```

- 已读文件:
  - runtime: `vllm/compilation/passes/fusion/collective_fusion.py` added +423/-0; `vllm/compilation/passes/fusion/allreduce_rms_fusion.py` renamed +13/-403
  - tests: `tests/compile/passes/distributed/test_async_tp.py` renamed +5/-75; `tests/compile/correctness_e2e/test_async_tp.py` added +79/-0; `tests/compile/passes/test_fusion.py` renamed +16/-11; `tests/compile/passes/test_functionalization.py` renamed +12/-9
  - other: `.buildkite/test_areas/compile.yaml` modified +24/-18; `.buildkite/test-amd.yaml` modified +19/-19
- 验证与风险: diff 自带测试面 `tests/compile/backend.py`, `tests/compile/correctness_e2e/__init__.py`, `tests/compile/correctness_e2e/test_async_tp.py`, `tests/compile/correctness_e2e/test_sequence_parallel.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #34128 - Vllm CPU benchmark suite improvement

- 链接: https://github.com/vllm-project/vllm/pull/34128
- 状态/时间: merged / 2026-02-12
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+802/-254，可读 patch 1243 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Vllm CPU benchmark suite improvement」；模型线: Llama 3.1；类别: 性能/后端优化；主要 diff: `.buildkite/performance-benchmarks/scripts/compare-json-results.py`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json`, `.buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh`；技术摘要: 覆盖「Vllm CPU benchmark suite improvement」；主要实现面是 `.buildkite/performance-benchmarks/scripts/compare-json-results.py`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json`, `.buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `.buildkite/performance-benchmarks/scripts/compare-json-results.py` modified +368/-77 (445 lines); hunks: -9,8 +9,10; -275,6 +277,131 @@ def _apply_two_decimals(; symbols: _apply_two_decimals, _sanitize_sheet_name, _group_to_sheet_base, _write_tables_to_excel_sheet，涉及 `_apply_two_decimals, _sanitize_sheet_name, _group_to_sheet_base`；`.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json` added +283/-0 (283 lines); hunks: -0,0 +1,283；`.buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh` modified +87/-46 (133 lines); hunks: -1,6 +1,4; -9,6 +7,11；`.buildkite/performance-benchmarks/tests/serving-tests-cpu.json` modified +0/-130 (130 lines); hunks: -148,136 +148,6。
- 代码 diff 细节:
  - `.buildkite/performance-benchmarks/scripts/compare-json-results.py` modified +368/-77 (445 lines); hunks: -9,8 +9,10; -275,6 +277,131 @@ def _apply_two_decimals(; symbols: _apply_two_decimals, _sanitize_sheet_name, _group_to_sheet_base, _write_tables_to_excel_sheet
  - `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json` added +283/-0 (283 lines); hunks: -0,0 +1,283
  - `.buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh` modified +87/-46 (133 lines); hunks: -1,6 +1,4; -9,6 +7,11
  - `.buildkite/performance-benchmarks/tests/serving-tests-cpu.json` modified +0/-130 (130 lines); hunks: -148,136 +148,6
  - `.buildkite/performance-benchmarks/tests/serving-tests-cpu-embed.json` added +41/-0 (41 lines); hunks: -0,0 +1,41
- 关键代码摘录:

```diff
diff -- .buildkite/performance-benchmarks/scripts/compare-json-results.py
@@ -9,8 +9,10 @@
+from pathlib import Path
+import regex as re
@@ -275,6 +277,131 @@ def _apply_two_decimals(
+# -----------------------------
+# Export helpers (Excel + CSV)
+# -----------------------------
diff -- .buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json
@@ -0,0 +1,283 @@
+{
+  "defaults": {
+    "qps_list": [
+      "inf"
+    ],
+    "max_concurrency_list": [12, 16, 24, 32, 64, 128, 200],
diff -- .buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh
@@ -1,6 +1,4 @@
```

- 已读文件:
  - other: `.buildkite/performance-benchmarks/scripts/compare-json-results.py` modified +368/-77; `.buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh` modified +87/-46
  - tests: `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json` added +283/-0; `.buildkite/performance-benchmarks/tests/serving-tests-cpu.json` modified +0/-130; `.buildkite/performance-benchmarks/tests/serving-tests-cpu-embed.json` added +41/-0
  - docs: `docs/getting_started/installation/cpu.md` modified +23/-1
- 验证与风险: diff 自带测试面 `.buildkite/performance-benchmarks/tests/serving-tests-cpu-embed.json`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu.json`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #34716 - [BugFix] Fix sp tests

- 链接: https://github.com/vllm-project/vllm/pull/34716
- 状态/时间: merged / 2026-02-17
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[BugFix] Fix sp tests」；模型线: Llama 3.1；类别: 缺陷修复；主要 diff: `tests/compile/correctness_e2e/test_sequence_parallel.py`；技术摘要: 覆盖「[BugFix] Fix sp tests」；主要实现面是 `tests/compile/correctness_e2e/test_sequence_parallel.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/compile/correctness_e2e/test_sequence_parallel.py` modified +1/-1 (2 lines); hunks: -229,7 +229,7 @@ def _compare_sp(; symbols: _compare_sp，涉及 `_compare_sp`。
- 代码 diff 细节:
  - `tests/compile/correctness_e2e/test_sequence_parallel.py` modified +1/-1 (2 lines); hunks: -229,7 +229,7 @@ def _compare_sp(; symbols: _compare_sp
- 关键代码摘录:

```diff
diff -- tests/compile/correctness_e2e/test_sequence_parallel.py
@@ -229,7 +229,7 @@ def _compare_sp(
-        common_args.append("--enforce-eager")
+        common_args.append("-cc.cudagraph_mode=none")
```

- 已读文件:
  - tests: `tests/compile/correctness_e2e/test_sequence_parallel.py` modified +1/-1
- 验证与风险: diff 自带测试面 `tests/compile/correctness_e2e/test_sequence_parallel.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #35871 - [CI] Add Blackwell AsyncTP correctness test

- 链接: https://github.com/vllm-project/vllm/pull/35871
- 状态/时间: merged / 2026-03-04
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+15/-0，可读 patch 30 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI] Add Blackwell AsyncTP correctness test」；模型线: Llama 3.1；类别: 文档/测试/CI；主要 diff: `.buildkite/test_areas/compile.yaml`, `tests/compile/correctness_e2e/test_async_tp.py`；技术摘要: 覆盖「[CI] Add Blackwell AsyncTP correctness test」；主要实现面是 `.buildkite/test_areas/compile.yaml`, `tests/compile/correctness_e2e/test_async_tp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `.buildkite/test_areas/compile.yaml` modified +10/-0 (10 lines); hunks: -36,6 +36,16 @@ steps:；`tests/compile/correctness_e2e/test_async_tp.py` modified +5/-0 (5 lines); hunks: -31,7 +31,12 @@ def test_async_tp_pass_correctness(; symbols: test_async_tp_pass_correctness，涉及 `test_async_tp_pass_correctness`。
- 代码 diff 细节:
  - `.buildkite/test_areas/compile.yaml` modified +10/-0 (10 lines); hunks: -36,6 +36,16 @@ steps:
  - `tests/compile/correctness_e2e/test_async_tp.py` modified +5/-0 (5 lines); hunks: -31,7 +31,12 @@ def test_async_tp_pass_correctness(; symbols: test_async_tp_pass_correctness
- 关键代码摘录:

```diff
diff -- .buildkite/test_areas/compile.yaml
@@ -36,6 +36,16 @@ steps:
+- label: AsyncTP Correctness Tests (B200)
+  timeout_in_minutes: 50
+  working_dir: "/vllm-workspace/"
+  device: b200
+  optional: true
+  num_devices: 2
diff -- tests/compile/correctness_e2e/test_async_tp.py
@@ -31,7 +31,12 @@ def test_async_tp_pass_correctness(
+    monkeypatch,
+    # Disable FlashInfer FP8 scaled_mm kernel as it is incompatible with
+    # async TP patterns. No-op on H100 (kernel requires CC >= 100).
+    monkeypatch.setenv("VLLM_DISABLED_KERNELS", "FlashInferFP8ScaledMMLinearKernel")
```

- 已读文件:
  - other: `.buildkite/test_areas/compile.yaml` modified +10/-0
  - tests: `tests/compile/correctness_e2e/test_async_tp.py` modified +5/-0
- 验证与风险: diff 自带测试面 `tests/compile/correctness_e2e/test_async_tp.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #36216 - [V0 Deprecation] Remove unused swap_space parameter

- 链接: https://github.com/vllm-project/vllm/pull/36216
- 状态/时间: merged / 2026-03-07
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 22 个文件，+19/-79，可读 patch 395 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[V0 Deprecation] Remove unused swap_space parameter」；模型线: Llama 3.1；类别: 文档/测试/CI；主要 diff: `vllm/entrypoints/llm.py`, `vllm/config/cache.py`, `docs/design/metrics.md`；技术摘要: 覆盖「[V0 Deprecation] Remove unused swap_space parameter」；主要实现面是 `vllm/entrypoints/llm.py`, `vllm/config/cache.py`, `docs/design/metrics.md`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/entrypoints/llm.py` modified +11/-8 (19 lines); hunks: -164,12 +164,6 @@ class LLM:; -240,7 +234,6 @@ def __init__(; symbols: LLM, __init__, _make_config，涉及 `LLM, __init__, _make_config`；`vllm/config/cache.py` modified +1/-33 (34 lines); hunks: -1,21 +1,13; -53,8 +45,6 @@ class CacheConfig:; symbols: CacheConfig, compute_hash, _validate_cache_dtype, verify_with_parallel_config，涉及 `CacheConfig, compute_hash, _validate_cache_dtype`；`docs/design/metrics.md` modified +4/-4 (8 lines); hunks: -507,10 +507,10 @@ longer relevant in v1:；`.buildkite/performance-benchmarks/tests/serving-tests-hpu.json` modified +0/-4 (4 lines); hunks: -10,7 +10,6; -37,7 +36,6。
- 代码 diff 细节:
  - `vllm/entrypoints/llm.py` modified +11/-8 (19 lines); hunks: -164,12 +164,6 @@ class LLM:; -240,7 +234,6 @@ def __init__(; symbols: LLM, __init__, _make_config
  - `vllm/config/cache.py` modified +1/-33 (34 lines); hunks: -1,21 +1,13; -53,8 +45,6 @@ class CacheConfig:; symbols: CacheConfig, compute_hash, _validate_cache_dtype, verify_with_parallel_config
  - `docs/design/metrics.md` modified +4/-4 (8 lines); hunks: -507,10 +507,10 @@ longer relevant in v1:
  - `.buildkite/performance-benchmarks/tests/serving-tests-hpu.json` modified +0/-4 (4 lines); hunks: -10,7 +10,6; -37,7 +36,6
  - `.buildkite/performance-benchmarks/tests/serving-tests.json` modified +0/-4 (4 lines); hunks: -5,7 +5,6; -23,7 +22,6
- 关键代码摘录:

```diff
diff -- vllm/entrypoints/llm.py
@@ -164,12 +164,6 @@ class LLM:
-        swap_space: The size (GiB) of CPU memory per GPU to use as swap space.
-            This can be used for temporarily storing the states of the requests
-            when their `best_of` sampling parameters are larger than 1. If all
-            requests will have `best_of=1`, you can safely set this to 0.
-            Noting that `best_of` is only supported in V0. Otherwise, too small
-            values may cause out-of-memory (OOM) errors.
diff -- vllm/config/cache.py
@@ -1,21 +1,13 @@
-import math
-from typing import TYPE_CHECKING, Any, Literal
+from typing import Literal
-from vllm.utils.mem_constants import GiB_bytes
-from vllm.utils.mem_utils import format_gib, get_cpu_memory
-if TYPE_CHECKING:
diff -- docs/design/metrics.md
@@ -507,10 +507,10 @@ longer relevant in v1:
```

- 已读文件:
  - runtime: `vllm/entrypoints/llm.py` modified +11/-8; `vllm/config/cache.py` modified +1/-33
  - docs: `docs/design/metrics.md` modified +4/-4
  - tests: `.buildkite/performance-benchmarks/tests/serving-tests-hpu.json` modified +0/-4; `.buildkite/performance-benchmarks/tests/serving-tests.json` modified +0/-4; `tests/distributed/test_torchrun_example.py` modified +1/-2; `tests/distributed/test_torchrun_example_moe.py` modified +1/-2; `tests/v1/worker/test_gpu_model_runner.py` modified +0/-3
- 验证与风险: diff 自带测试面 `.buildkite/performance-benchmarks/tests/serving-tests-hpu.json`, `.buildkite/performance-benchmarks/tests/serving-tests.json`, `tests/conftest.py`, `tests/distributed/test_torchrun_example.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #35086 - more models for vLLM Benchmark Suite

- 链接: https://github.com/vllm-project/vllm/pull/35086
- 状态/时间: merged / 2026-03-12
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+800/-119，可读 patch 1301 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「more models for vLLM Benchmark Suite」；模型线: Llama 3.1；类别: 性能/后端优化；主要 diff: `.buildkite/performance-benchmarks/scripts/compare-json-results.py`, `.buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json`；技术摘要: 覆盖「more models for vLLM Benchmark Suite」；主要实现面是 `.buildkite/performance-benchmarks/scripts/compare-json-results.py`, `.buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `.buildkite/performance-benchmarks/scripts/compare-json-results.py` modified +301/-90 (391 lines); hunks: -7,12 +7,12; -33,6 +33,45; symbols: _find_concurrency_col, _normalize_concurrency_in_df, compare_data_columns，涉及 `_find_concurrency_col, _normalize_concurrency_in_df, compare_data_columns`；`.buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh` modified +361/-4 (365 lines); hunks: -12,6 +12,13 @@ DRY_RUN="${DRY_RUN:-0}"; -183,6 +190,304 @@ upload_to_buildkite() {；`.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json` modified +72/-0 (72 lines); hunks: -149,6 +149,39; -188,6 +221,45；`.buildkite/performance-benchmarks/tests/serving-tests-cpu-asr.json` added +37/-0 (37 lines); hunks: -0,0 +1,37。
- 代码 diff 细节:
  - `.buildkite/performance-benchmarks/scripts/compare-json-results.py` modified +301/-90 (391 lines); hunks: -7,12 +7,12; -33,6 +33,45; symbols: _find_concurrency_col, _normalize_concurrency_in_df, compare_data_columns
  - `.buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh` modified +361/-4 (365 lines); hunks: -12,6 +12,13 @@ DRY_RUN="${DRY_RUN:-0}"; -183,6 +190,304 @@ upload_to_buildkite() {
  - `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json` modified +72/-0 (72 lines); hunks: -149,6 +149,39; -188,6 +221,45
  - `.buildkite/performance-benchmarks/tests/serving-tests-cpu-asr.json` added +37/-0 (37 lines); hunks: -0,0 +1,37
  - `.buildkite/performance-benchmarks/tests/serving-tests-cpu.json` modified +12/-23 (35 lines); hunks: -72,17 +72,6; -106,20 +95,20
- 关键代码摘录:

```diff
diff -- .buildkite/performance-benchmarks/scripts/compare-json-results.py
@@ -7,12 +7,12 @@
+from contextlib import nullcontext
-import regex as re
@@ -33,6 +33,45 @@
+# -----------------------------
+# Concurrency normalization (NEW, small)
+# -----------------------------
diff -- .buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh
@@ -12,6 +12,13 @@ DRY_RUN="${DRY_RUN:-0}"
+# Adaptive search controls
+ENABLE_ADAPTIVE_CONCURRENCY="${ENABLE_ADAPTIVE_CONCURRENCY:-0}"
+SLA_TTFT_MS="${SLA_TTFT_MS:-3000}"
+SLA_TPOT_MS="${SLA_TPOT_MS:-100}"
+ADAPTIVE_MAX_PROBES="${ADAPTIVE_MAX_PROBES:-8}"
+ADAPTIVE_MAX_CONCURRENCY="${ADAPTIVE_MAX_CONCURRENCY:-1024}"
diff -- .buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json
@@ -149,6 +149,39 @@
```

- 已读文件:
  - other: `.buildkite/performance-benchmarks/scripts/compare-json-results.py` modified +301/-90; `.buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh` modified +361/-4; `requirements/test.txt` modified +7/-1; `requirements/test.in` modified +4/-1
  - tests: `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json` modified +72/-0; `.buildkite/performance-benchmarks/tests/serving-tests-cpu-asr.json` added +37/-0; `.buildkite/performance-benchmarks/tests/serving-tests-cpu.json` modified +12/-23
  - docs: `docs/benchmarking/dashboard.md` modified +6/-0
- 验证与风险: diff 自带测试面 `.buildkite/performance-benchmarks/tests/serving-tests-cpu-asr.json`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu.json`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #38576 - vLLM Benchmark Suite perf regression after PR#32723

- 链接: https://github.com/vllm-project/vllm/pull/38576
- 状态/时间: merged / 2026-03-31
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+15/-1，可读 patch 119 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「vLLM Benchmark Suite perf regression after PR#32723」；模型线: Llama 3.1；类别: 缺陷修复；主要 diff: `.buildkite/performance-benchmarks/tests/serving-tests-hpu.json`, `.buildkite/performance-benchmarks/tests/serving-tests.json`, `.buildkite/performance-benchmarks/tests/serving-tests-arm64-cpu.json`；技术摘要: 覆盖「vLLM Benchmark Suite perf regression after PR#32723」；主要实现面是 `.buildkite/performance-benchmarks/tests/serving-tests-hpu.json`, `.buildkite/performance-benchmarks/tests/serving-tests.json`, `.buildkite/performance-benchmarks/tests/serving-tests-arm64-cpu.json`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `.buildkite/performance-benchmarks/tests/serving-tests-hpu.json` modified +6/-0 (6 lines); hunks: -21,6 +21,7; -47,6 +48,7；`.buildkite/performance-benchmarks/tests/serving-tests.json` modified +4/-0 (4 lines); hunks: -13,6 +13,7; -30,6 +31,7；`.buildkite/performance-benchmarks/tests/serving-tests-arm64-cpu.json` modified +2/-1 (3 lines); hunks: -36,6 +36,7; -127,4 +128,4；`.buildkite/performance-benchmarks/tests/serving-tests-cpu-asr.json` modified +1/-0 (1 lines); hunks: -22,6 +22,7。
- 代码 diff 细节:
  - `.buildkite/performance-benchmarks/tests/serving-tests-hpu.json` modified +6/-0 (6 lines); hunks: -21,6 +21,7; -47,6 +48,7
  - `.buildkite/performance-benchmarks/tests/serving-tests.json` modified +4/-0 (4 lines); hunks: -13,6 +13,7; -30,6 +31,7
  - `.buildkite/performance-benchmarks/tests/serving-tests-arm64-cpu.json` modified +2/-1 (3 lines); hunks: -36,6 +36,7; -127,4 +128,4
  - `.buildkite/performance-benchmarks/tests/serving-tests-cpu-asr.json` modified +1/-0 (1 lines); hunks: -22,6 +22,7
  - `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json` modified +1/-0 (1 lines); hunks: -26,6 +26,7
- 关键代码摘录:

```diff
diff -- .buildkite/performance-benchmarks/tests/serving-tests-hpu.json
@@ -21,6 +21,7 @@
+            "temperature": 0,
@@ -47,6 +48,7 @@
+            "temperature": 0,
@@ -73,6 +75,7 @@
+            "temperature": 0,
@@ -100,6 +103,7 @@
diff -- .buildkite/performance-benchmarks/tests/serving-tests.json
@@ -13,6 +13,7 @@
+            "temperature": 0,
@@ -30,6 +31,7 @@
+            "temperature": 0,
@@ -47,6 +49,7 @@
+            "temperature": 0,
@@ -67,6 +70,7 @@
diff -- .buildkite/performance-benchmarks/tests/serving-tests-arm64-cpu.json
@@ -36,6 +36,7 @@
```

- 已读文件:
  - tests: `.buildkite/performance-benchmarks/tests/serving-tests-hpu.json` modified +6/-0; `.buildkite/performance-benchmarks/tests/serving-tests.json` modified +4/-0; `.buildkite/performance-benchmarks/tests/serving-tests-arm64-cpu.json` modified +2/-1; `.buildkite/performance-benchmarks/tests/serving-tests-cpu-asr.json` modified +1/-0; `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json` modified +1/-0; `.buildkite/performance-benchmarks/tests/serving-tests-cpu.json` modified +1/-0
- 验证与风险: diff 自带测试面 `.buildkite/performance-benchmarks/tests/serving-tests-arm64-cpu.json`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu-asr.json`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu.json`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #38373 - [torch.compile]: Disable Sequence Parallelism (SP) for piecewise compilation

- 链接: https://github.com/vllm-project/vllm/pull/38373
- 状态/时间: merged / 2026-04-26
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+223/-80，可读 patch 450 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[torch.compile]: Disable Sequence Parallelism (SP) for piecewise compilation」；模型线: Llama 3.1；类别: 文档/测试/CI；主要 diff: `tests/compile/test_config.py`, `vllm/config/vllm.py`, `vllm/compilation/passes/fusion/sequence_parallelism.py`；技术摘要: 覆盖「[torch.compile]: Disable Sequence Parallelism (SP) for piecewise compilation」；主要实现面是 `tests/compile/test_config.py`, `vllm/config/vllm.py`, `vllm/compilation/passes/fusion/sequence_parallelism.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/compile/test_config.py` modified +118/-1 (119 lines); hunks: -407,7 +407,7 @@ def test_should_split():; -465,6 +465,123 @@ def test_cudagraph_sizes_post_init(; symbols: test_should_split, test_cudagraph_sizes_post_init, test_sequence_parallelism_requires_full_graph_compilation, test_cached_compilation_config，涉及 `test_should_split, test_cudagraph_sizes_post_init, test_sequence_parallelism_requires_full_graph_compilation`；`vllm/config/vllm.py` modified +17/-28 (45 lines); hunks: -983,19 +983,16 @@ def has_blocked_weights():; -1015,8 +1012,8 @@ def has_blocked_weights():; symbols: has_blocked_weights，涉及 `has_blocked_weights`；`vllm/compilation/passes/fusion/sequence_parallelism.py` modified +16/-26 (42 lines); hunks: -341,22 +341,18 @@ class SequenceParallelismPass(VllmPatternMatcherPass):; -419,19 +415,13 @@ def is_applicable_for_range(self, compile_range: Range) ->...; symbols: SequenceParallelismPass, is_applicable_for_range，涉及 `SequenceParallelismPass, is_applicable_for_range`；`vllm/v1/worker/utils.py` modified +8/-15 (23 lines); hunks: -519,12 +519,8 @@ def is_residual_scattered_for_sp(; -534,16 +530,13 @@ def is_residual_scattered_for_sp(; symbols: is_residual_scattered_for_sp，涉及 `is_residual_scattered_for_sp`。
- 代码 diff 细节:
  - `tests/compile/test_config.py` modified +118/-1 (119 lines); hunks: -407,7 +407,7 @@ def test_should_split():; -465,6 +465,123 @@ def test_cudagraph_sizes_post_init(; symbols: test_should_split, test_cudagraph_sizes_post_init, test_sequence_parallelism_requires_full_graph_compilation, test_cached_compilation_config
  - `vllm/config/vllm.py` modified +17/-28 (45 lines); hunks: -983,19 +983,16 @@ def has_blocked_weights():; -1015,8 +1012,8 @@ def has_blocked_weights():; symbols: has_blocked_weights
  - `vllm/compilation/passes/fusion/sequence_parallelism.py` modified +16/-26 (42 lines); hunks: -341,22 +341,18 @@ class SequenceParallelismPass(VllmPatternMatcherPass):; -419,19 +415,13 @@ def is_applicable_for_range(self, compile_range: Range) ->...; symbols: SequenceParallelismPass, is_applicable_for_range
  - `vllm/v1/worker/utils.py` modified +8/-15 (23 lines); hunks: -519,12 +519,8 @@ def is_residual_scattered_for_sp(; -534,16 +530,13 @@ def is_residual_scattered_for_sp(; symbols: is_residual_scattered_for_sp
  - `tests/compile/passes/distributed/test_sequence_parallelism.py` modified +19/-0 (19 lines); hunks: -22,6 +22,7; -216,6 +217,24 @@ def run_torch_spawn(fn, nprocs):; symbols: run_torch_spawn, test_sequence_parallelism_pass_requires_full_graph_compilation, sequence_parallelism_pass_on_test_model
- 关键代码摘录:

```diff
diff -- tests/compile/test_config.py
@@ -407,7 +407,7 @@ def test_should_split():
-        # filtered out 15 due to SP
+        # SP forces full-graph compilation, sizes are filtered by TP
@@ -465,6 +465,123 @@ def test_cudagraph_sizes_post_init(
+@pytest.mark.skipif(
+    not current_platform.support_static_graph_mode(),
+    reason="Skip if not cudagraph mode supported",
diff -- vllm/config/vllm.py
@@ -983,19 +983,16 @@ def has_blocked_weights():
-        # async tp is built on top of sequence parallelism
-        # and requires it to be enabled.
-        if self.compilation_config.pass_config.fuse_gemm_comms:
-            self.compilation_config.pass_config.enable_sp = True
-        if self.compilation_config.pass_config.enable_sp:
+        # async tp is built on top of sequence parallelism and requires it.
diff -- vllm/compilation/passes/fusion/sequence_parallelism.py
@@ -341,22 +341,18 @@ class SequenceParallelismPass(VllmPatternMatcherPass):
```

- 已读文件:
  - tests: `tests/compile/test_config.py` modified +118/-1; `tests/compile/passes/distributed/test_sequence_parallelism.py` modified +19/-0; `tests/compile/passes/distributed/test_async_tp.py` modified +17/-0
  - runtime: `vllm/config/vllm.py` modified +17/-28; `vllm/compilation/passes/fusion/sequence_parallelism.py` modified +16/-26; `vllm/v1/worker/utils.py` modified +8/-15; `vllm/config/compilation.py` modified +19/-0; `vllm/compilation/passes/fusion/collective_fusion.py` modified +7/-10
- 验证与风险: diff 自带测试面 `tests/compile/correctness_e2e/test_sequence_parallel.py`, `tests/compile/passes/distributed/test_async_tp.py`, `tests/compile/passes/distributed/test_sequence_parallelism.py`, `tests/compile/test_config.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #41882 - Add NVFP4 all-gather GEMM fusion for AsyncTP

- 链接: https://github.com/vllm-project/vllm/pull/41882
- 状态/时间: merged / 2026-05-10
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+605/-6，可读 patch 781 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add NVFP4 all-gather GEMM fusion for AsyncTP」；模型线: Llama 3.1；类别: 性能/后端优化；主要 diff: `vllm/compilation/passes/fusion/collective_fusion.py`, `vllm/compilation/passes/fusion/sequence_parallelism.py`, `tests/compile/correctness_e2e/test_async_tp.py`；技术摘要: 覆盖「Add NVFP4 all-gather GEMM fusion for AsyncTP」；主要实现面是 `vllm/compilation/passes/fusion/collective_fusion.py`, `vllm/compilation/passes/fusion/sequence_parallelism.py`, `tests/compile/correctness_e2e/test_async_tp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/compilation/passes/fusion/collective_fusion.py` modified +243/-0 (243 lines); hunks: -74,6 +74,36 @@ def _flashinfer_scaled_mm_out(; -197,6 +227,90 @@ def fused_all_gather_flashinfer_scaled_matmul(; symbols: _flashinfer_scaled_mm_out, _flashinfer_fp4_mm_out, fused_flashinfer_scaled_matmul_reduce_scatter_fake, fused_all_gather_flashinfer_scaled_matmul，涉及 `_flashinfer_scaled_mm_out, _flashinfer_fp4_mm_out, fused_flashinfer_scaled_matmul_reduce_scatter_fake`；`vllm/compilation/passes/fusion/sequence_parallelism.py` modified +136/-0 (136 lines); hunks: -8,6 +8,7; -27,6 +28,10; symbols: replacement, FirstAllReduceRMSNormStaticNVFP4Pattern, get_inputs, register，涉及 `replacement, FirstAllReduceRMSNormStaticNVFP4Pattern, get_inputs`；`tests/compile/correctness_e2e/test_async_tp.py` modified +73/-0 (73 lines); hunks: -13,6 +13,17; -82,3 +93,65 @@ def test_async_tp_pass_correctness(; symbols: test_async_tp_pass_correctness, test_async_tp_pass_nvfp4_correctness，涉及 `test_async_tp_pass_correctness, test_async_tp_pass_nvfp4_correctness`；`tests/compile/fusions_e2e/test_tp2_async_tp.py` modified +65/-0 (65 lines); hunks: -13,11 +13,13; -90,6 +92,69 @@ def test_tp2_async_tp_fp8_fusions(; symbols: test_tp2_async_tp_fp8_fusions, test_tp2_async_tp_nvfp4_fusions，涉及 `test_tp2_async_tp_fp8_fusions, test_tp2_async_tp_nvfp4_fusions`。
- 代码 diff 细节:
  - `vllm/compilation/passes/fusion/collective_fusion.py` modified +243/-0 (243 lines); hunks: -74,6 +74,36 @@ def _flashinfer_scaled_mm_out(; -197,6 +227,90 @@ def fused_all_gather_flashinfer_scaled_matmul(; symbols: _flashinfer_scaled_mm_out, _flashinfer_fp4_mm_out, fused_flashinfer_scaled_matmul_reduce_scatter_fake, fused_all_gather_flashinfer_scaled_matmul
  - `vllm/compilation/passes/fusion/sequence_parallelism.py` modified +136/-0 (136 lines); hunks: -8,6 +8,7; -27,6 +28,10; symbols: replacement, FirstAllReduceRMSNormStaticNVFP4Pattern, get_inputs, register
  - `tests/compile/correctness_e2e/test_async_tp.py` modified +73/-0 (73 lines); hunks: -13,6 +13,17; -82,3 +93,65 @@ def test_async_tp_pass_correctness(; symbols: test_async_tp_pass_correctness, test_async_tp_pass_nvfp4_correctness
  - `tests/compile/fusions_e2e/test_tp2_async_tp.py` modified +65/-0 (65 lines); hunks: -13,11 +13,13; -90,6 +92,69 @@ def test_tp2_async_tp_fp8_fusions(; symbols: test_tp2_async_tp_fp8_fusions, test_tp2_async_tp_nvfp4_fusions
  - `tests/compile/correctness_e2e/test_sequence_parallel.py` modified +44/-5 (49 lines); hunks: -21,12 +21,14; -41,6 +43,7 @@ class ParallelSetup(NamedTuple):; symbols: ParallelSetup, SPTestOptions, _compare_sp
- 关键代码摘录:

```diff
diff -- vllm/compilation/passes/fusion/collective_fusion.py
@@ -74,6 +74,36 @@ def _flashinfer_scaled_mm_out(
+def _flashinfer_fp4_mm_out(
+    A: torch.Tensor,
+    B: torch.Tensor,
+    *,
+    scale_a: torch.Tensor,
+    scale_b: torch.Tensor,
diff -- vllm/compilation/passes/fusion/sequence_parallelism.py
@@ -8,6 +8,7 @@
+from torch._higher_order_ops.auto_functionalize import auto_functionalized
@@ -27,6 +28,10 @@
+if hasattr(torch.ops._C, "scaled_fp4_quant"):
+    SCALED_FP4_QUANT_OUT_OVERLOAD = torch.ops._C.scaled_fp4_quant.out
+    SCALED_FP4_QUANT_DEFAULT_OVERLOAD = torch.ops._C.scaled_fp4_quant.default
@@ -332,6 +337,129 @@ def replacement(
diff -- tests/compile/correctness_e2e/test_async_tp.py
@@ -13,6 +13,17 @@
```

- 已读文件:
  - runtime: `vllm/compilation/passes/fusion/collective_fusion.py` modified +243/-0; `vllm/compilation/passes/fusion/sequence_parallelism.py` modified +136/-0; `vllm/utils/flashinfer.py` modified +42/-0
  - tests: `tests/compile/correctness_e2e/test_async_tp.py` modified +73/-0; `tests/compile/fusions_e2e/test_tp2_async_tp.py` modified +65/-0; `tests/compile/correctness_e2e/test_sequence_parallel.py` modified +44/-5; `tests/compile/fullgraph/test_toy_llama.py` modified +2/-1
- 验证与风险: diff 自带测试面 `tests/compile/correctness_e2e/test_async_tp.py`, `tests/compile/correctness_e2e/test_sequence_parallel.py`, `tests/compile/fullgraph/test_toy_llama.py`, `tests/compile/fusions_e2e/test_tp2_async_tp.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #33322 - [Bugfix] Fix SP pass for multimodal models and PP+SP residual handling

- 链接: https://github.com/vllm-project/vllm/pull/33322
- 状态/时间: merged / 2026-05-10
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+116/-34，可读 patch 260 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix SP pass for multimodal models and PP+SP residual handling」；模型线: Llama 3.1；类别: 缺陷修复；主要 diff: `vllm/compilation/passes/fusion/sequence_parallelism.py`, `tests/compile/correctness_e2e/test_sequence_parallel.py`, `vllm/v1/worker/gpu_model_runner.py`；技术摘要: 覆盖「[Bugfix] Fix SP pass for multimodal models and PP+SP residual handling」；主要实现面是 `vllm/compilation/passes/fusion/sequence_parallelism.py`, `tests/compile/correctness_e2e/test_sequence_parallel.py`, `vllm/v1/worker/gpu_model_runner.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/compilation/passes/fusion/sequence_parallelism.py` modified +55/-20 (75 lines); hunks: -14,7 +14,10; -117,6 +120,7 @@ def __init__(; symbols: __init__, _all_reduce, replacement, SequenceParallelismPass，涉及 `__init__, _all_reduce, replacement`；`tests/compile/correctness_e2e/test_sequence_parallel.py` modified +48/-0 (48 lines); hunks: -167,6 +167,7 @@ def _compare_sp(; -248,6 +249,8 @@ def _compare_sp(; symbols: _compare_sp, test_tp_sp_generation, test_tp_sp_generation_prompt_embeds，涉及 `_compare_sp, test_tp_sp_generation, test_tp_sp_generation_prompt_embeds`；`vllm/v1/worker/gpu_model_runner.py` modified +13/-14 (27 lines); hunks: -3098,7 +3098,7 @@ def get_supported_tasks(self) -> tuple[SupportedTask, ...]:; -3109,24 +3109,23 @@ def sync_and_slice_intermediate_tensors(; symbols: get_supported_tasks, sync_and_slice_intermediate_tensors, sync_and_gather_intermediate_tensors, eplb_step，涉及 `get_supported_tasks, sync_and_slice_intermediate_tensors, sync_and_gather_intermediate_tensors`。
- 代码 diff 细节:
  - `vllm/compilation/passes/fusion/sequence_parallelism.py` modified +55/-20 (75 lines); hunks: -14,7 +14,10; -117,6 +120,7 @@ def __init__(; symbols: __init__, _all_reduce, replacement, SequenceParallelismPass
  - `tests/compile/correctness_e2e/test_sequence_parallel.py` modified +48/-0 (48 lines); hunks: -167,6 +167,7 @@ def _compare_sp(; -248,6 +249,8 @@ def _compare_sp(; symbols: _compare_sp, test_tp_sp_generation, test_tp_sp_generation_prompt_embeds
  - `vllm/v1/worker/gpu_model_runner.py` modified +13/-14 (27 lines); hunks: -3098,7 +3098,7 @@ def get_supported_tasks(self) -> tuple[SupportedTask, ...]:; -3109,24 +3109,23 @@ def sync_and_slice_intermediate_tensors(; symbols: get_supported_tasks, sync_and_slice_intermediate_tensors, sync_and_gather_intermediate_tensors, eplb_step
- 关键代码摘录:

```diff
diff -- vllm/compilation/passes/fusion/sequence_parallelism.py
@@ -14,7 +14,10 @@
-from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size
+from vllm.distributed.parallel_state import (
+    get_tensor_model_parallel_rank,
+    get_tensor_model_parallel_world_size,
+)
@@ -117,6 +120,7 @@ def __init__(
diff -- tests/compile/correctness_e2e/test_sequence_parallel.py
@@ -167,6 +167,7 @@ def _compare_sp(
+    enable_prompt_embeds: bool,
@@ -248,6 +249,8 @@ def _compare_sp(
+    elif enable_prompt_embeds:
+        common_args.append("--enable-prompt-embeds")
@@ -257,7 +260,9 @@ def _compare_sp(
+            "fuse_allreduce_rms": False,
diff -- vllm/v1/worker/gpu_model_runner.py
@@ -3098,7 +3098,7 @@ def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
```

- 已读文件:
  - runtime: `vllm/compilation/passes/fusion/sequence_parallelism.py` modified +55/-20; `vllm/v1/worker/gpu_model_runner.py` modified +13/-14
  - tests: `tests/compile/correctness_e2e/test_sequence_parallel.py` modified +48/-0
- 验证与风险: diff 自带测试面 `tests/compile/correctness_e2e/test_sequence_parallel.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #42197 - Fix mypy failure on main

- 链接: https://github.com/vllm-project/vllm/pull/42197
- 状态/时间: merged / 2026-05-10
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-0，可读 patch 8 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix mypy failure on main」；模型线: Llama 3.1；类别: 缺陷修复；主要 diff: `tests/compile/correctness_e2e/test_sequence_parallel.py`；技术摘要: 覆盖「Fix mypy failure on main」；主要实现面是 `tests/compile/correctness_e2e/test_sequence_parallel.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/compile/correctness_e2e/test_sequence_parallel.py` modified +1/-0 (1 lines); hunks: -435,6 +435,7 @@ def test_tp_sp_nvfp4_generation(num_gpus_available: int):; symbols: test_tp_sp_nvfp4_generation，涉及 `test_tp_sp_nvfp4_generation`。
- 代码 diff 细节:
  - `tests/compile/correctness_e2e/test_sequence_parallel.py` modified +1/-0 (1 lines); hunks: -435,6 +435,7 @@ def test_tp_sp_nvfp4_generation(num_gpus_available: int):; symbols: test_tp_sp_nvfp4_generation
- 关键代码摘录:

```diff
diff -- tests/compile/correctness_e2e/test_sequence_parallel.py
@@ -435,6 +435,7 @@ def test_tp_sp_nvfp4_generation(num_gpus_available: int):
+        enable_prompt_embeds=False,
```

- 已读文件:
  - tests: `tests/compile/correctness_e2e/test_sequence_parallel.py` modified +1/-0
- 验证与风险: diff 自带测试面 `tests/compile/correctness_e2e/test_sequence_parallel.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #42607 - Update Intel Xeon model list and vLLM Benchmark Suite BKMs

- 链接: https://github.com/vllm-project/vllm/pull/42607
- 状态/时间: merged / 2026-05-15
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+118/-159，可读 patch 465 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Update Intel Xeon model list and vLLM Benchmark Suite BKMs」；模型线: Llama 3.1；类别: 性能/后端优化；主要 diff: `docs/models/hardware_supported_models/cpu.md`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json`；技术摘要: 覆盖「Update Intel Xeon model list and vLLM Benchmark Suite BKMs」；主要实现面是 `docs/models/hardware_supported_models/cpu.md`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs/models/hardware_supported_models/cpu.md` modified +42/-16 (58 lines); hunks: -11,24 +11,50；`.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json` modified +76/-143 (219 lines); hunks: -31,30 +31,9; -63,290 +42,244。
- 代码 diff 细节:
  - `docs/models/hardware_supported_models/cpu.md` modified +42/-16 (58 lines); hunks: -11,24 +11,50
  - `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json` modified +76/-143 (219 lines); hunks: -31,30 +31,9; -63,290 +42,244
- 关键代码摘录:

```diff
diff -- docs/models/hardware_supported_models/cpu.md
@@ -11,24 +11,50 @@
-| Model                                | Architecture                             | Supported |
+| Model | Architecture | Supported |
-| meta-llama/Llama-3.1-8B-Instruct     | LlamaForCausalLM                         | ✅        |
-| meta-llama/Llama-3.2-3B-Instruct     | LlamaForCausalLM                         | ✅        |
-| ibm-granite/granite-3.2-2b-instruct  | GraniteForCausalLM                       | ✅        |
-| Qwen/Qwen3-1.7B                      | Qwen3ForCausalLM                         | ✅        |
diff -- .buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json
@@ -31,30 +31,9 @@
-    {
-      "test_name": "serving_llama8B_tp1_sharegpt",
-      "server_parameters": {
-        "tensor_parallel_size": 1
-      },
-      "client_parameters": {
```

- 已读文件:
  - docs: `docs/models/hardware_supported_models/cpu.md` modified +42/-16
  - tests: `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json` modified +76/-143
- 验证与风险: diff 自带测试面 `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #43262 - update GPU json file based on h200 recipes

- 链接: https://github.com/vllm-project/vllm/pull/43262
- 状态/时间: merged / 2026-05-21
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+104/-69，可读 patch 182 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「update GPU json file based on h200 recipes」；模型线: Llama 3.1；类别: 性能/后端优化；主要 diff: `.buildkite/performance-benchmarks/tests/serving-tests.json`；技术摘要: 覆盖「update GPU json file based on h200 recipes」；主要实现面是 `.buildkite/performance-benchmarks/tests/serving-tests.json`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `.buildkite/performance-benchmarks/tests/serving-tests.json` modified +104/-69 (173 lines); hunks: -1,77 +1,112。
- 代码 diff 细节:
  - `.buildkite/performance-benchmarks/tests/serving-tests.json` modified +104/-69 (173 lines); hunks: -1,77 +1,112
- 关键代码摘录:

```diff
diff -- .buildkite/performance-benchmarks/tests/serving-tests.json
@@ -1,77 +1,112 @@
-[
+{
+  "defaults": {
+    "qps_list": [
+      "inf"
+    ],
```

- 已读文件:
  - tests: `.buildkite/performance-benchmarks/tests/serving-tests.json` modified +104/-69
- 验证与风险: diff 自带测试面 `.buildkite/performance-benchmarks/tests/serving-tests.json`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #43233 - [Model Runner v2] Force v1 runner for tests

- 链接: https://github.com/vllm-project/vllm/pull/43233
- 状态/时间: merged / 2026-05-23
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+50/-6，可读 patch 136 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model Runner v2] Force v1 runner for tests」；模型线: Llama 3.1；类别: 文档/测试/CI；主要 diff: `tests/models/quantization/test_bitsandbytes.py`, `tests/compile/correctness_e2e/test_async_tp.py`, `tests/utils.py`；技术摘要: 覆盖「[Model Runner v2] Force v1 runner for tests」；主要实现面是 `tests/models/quantization/test_bitsandbytes.py`, `tests/compile/correctness_e2e/test_async_tp.py`, `tests/utils.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/models/quantization/test_bitsandbytes.py` modified +5/-1 (6 lines); hunks: -137,7 +137,11 @@ def test_load_pp_4bit_bnb_model(model_name, description) ->...; symbols: test_load_pp_4bit_bnb_model，涉及 `test_load_pp_4bit_bnb_model`；`tests/compile/correctness_e2e/test_async_tp.py` modified +14/-2 (16 lines); hunks: -92,7 +92,13 @@ def test_async_tp_pass_correctness(; -154,4 +160,10 @@ def test_async_tp_pass_nvfp4_correctness(num_gpus_available...; symbols: test_async_tp_pass_correctness, test_async_tp_pass_nvfp4_correctness，涉及 `test_async_tp_pass_correctness, test_async_tp_pass_nvfp4_correctness`；`tests/utils.py` modified +14/-0 (14 lines); hunks: -1116,6 +1116,7 @@ def compare_two_settings(; -1129,6 +1130,9 @@ def compare_two_settings(; symbols: compare_two_settings, compare_all_settings，涉及 `compare_two_settings, compare_all_settings`；`tests/distributed/test_pipeline_parallel.py` modified +8/-1 (9 lines); hunks: -349,7 +349,14 @@ def _compare_tp(; symbols: _compare_tp，涉及 `_compare_tp`。
- 代码 diff 细节:
  - `tests/models/quantization/test_bitsandbytes.py` modified +5/-1 (6 lines); hunks: -137,7 +137,11 @@ def test_load_pp_4bit_bnb_model(model_name, description) ->...; symbols: test_load_pp_4bit_bnb_model
  - `tests/compile/correctness_e2e/test_async_tp.py` modified +14/-2 (16 lines); hunks: -92,7 +92,13 @@ def test_async_tp_pass_correctness(; -154,4 +160,10 @@ def test_async_tp_pass_nvfp4_correctness(num_gpus_available...; symbols: test_async_tp_pass_correctness, test_async_tp_pass_nvfp4_correctness
  - `tests/utils.py` modified +14/-0 (14 lines); hunks: -1116,6 +1116,7 @@ def compare_two_settings(; -1129,6 +1130,9 @@ def compare_two_settings(; symbols: compare_two_settings, compare_all_settings
  - `tests/distributed/test_pipeline_parallel.py` modified +8/-1 (9 lines); hunks: -349,7 +349,14 @@ def _compare_tp(; symbols: _compare_tp
  - `tests/compile/correctness_e2e/test_sequence_parallel.py` modified +7/-1 (8 lines); hunks: -294,7 +294,13 @@ def _compare_sp(; symbols: _compare_sp
- 关键代码摘录:

```diff
diff -- tests/models/quantization/test_bitsandbytes.py
@@ -137,7 +137,11 @@ def test_load_pp_4bit_bnb_model(model_name, description) -> None:
-    compare_two_settings(model_name, common_args, pp_args)
+    compare_two_settings(
+        model_name,
+        common_args,
+        pp_args,
+    )
diff -- tests/compile/correctness_e2e/test_async_tp.py
@@ -92,7 +92,13 @@ def test_async_tp_pass_correctness(
-    compare_two_settings(model_id, async_tp_args, tp_args, method="generate")
+    compare_two_settings(
+        model_id,
+        async_tp_args,
+        tp_args,
+        method="generate",
diff -- tests/utils.py
@@ -1116,6 +1116,7 @@ def compare_two_settings(
```

- 已读文件:
  - tests: `tests/models/quantization/test_bitsandbytes.py` modified +5/-1; `tests/compile/correctness_e2e/test_async_tp.py` modified +14/-2; `tests/utils.py` modified +14/-0; `tests/distributed/test_pipeline_parallel.py` modified +8/-1; `tests/compile/correctness_e2e/test_sequence_parallel.py` modified +7/-1; `tests/compile/fullgraph/test_basic_correctness.py` modified +2/-1
- 验证与风险: diff 自带测试面 `tests/compile/correctness_e2e/test_async_tp.py`, `tests/compile/correctness_e2e/test_sequence_parallel.py`, `tests/compile/fullgraph/test_basic_correctness.py`, `tests/distributed/test_pipeline_parallel.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #44128 - [Misc] Remove dead VLLM_RPC_TIMEOUT env var and fix profiling doc that references it

- 链接: https://github.com/vllm-project/vllm/pull/44128
- 状态/时间: merged / 2026-06-03
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 11 个文件，+1/-15，可读 patch 100 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Misc] Remove dead VLLM_RPC_TIMEOUT env var and fix profiling doc that references it」；模型线: Llama 3.1；类别: 缺陷修复；主要 diff: `vllm/envs.py`, `docs/contributing/profiling.md`, `.buildkite/performance-benchmarks/tests/latency-tests-arm64-cpu.json`；技术摘要: 覆盖「[Misc] Remove dead VLLM_RPC_TIMEOUT env var and fix profiling doc that references it」；主要实现面是 `vllm/envs.py`, `docs/contributing/profiling.md`, `.buildkite/performance-benchmarks/tests/latency-tests-arm64-cpu.json`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/envs.py` modified +0/-4 (4 lines); hunks: -95,7 +95,6; -1015,9 +1014,6 @@ def _resolve_rust_frontend_path() -> str | None:; symbols: _resolve_rust_frontend_path，涉及 `_resolve_rust_frontend_path`；`docs/contributing/profiling.md` modified +1/-2 (3 lines); hunks: -35,8 +35,7 @@ Traces can be visualized using .；`.buildkite/performance-benchmarks/tests/latency-tests-arm64-cpu.json` modified +0/-1 (1 lines); hunks: -2,7 +2,6；`.buildkite/performance-benchmarks/tests/latency-tests-cpu.json` modified +0/-1 (1 lines); hunks: -2,7 +2,6。
- 代码 diff 细节:
  - `vllm/envs.py` modified +0/-4 (4 lines); hunks: -95,7 +95,6; -1015,9 +1014,6 @@ def _resolve_rust_frontend_path() -> str | None:; symbols: _resolve_rust_frontend_path
  - `docs/contributing/profiling.md` modified +1/-2 (3 lines); hunks: -35,8 +35,7 @@ Traces can be visualized using .
  - `.buildkite/performance-benchmarks/tests/latency-tests-arm64-cpu.json` modified +0/-1 (1 lines); hunks: -2,7 +2,6
  - `.buildkite/performance-benchmarks/tests/latency-tests-cpu.json` modified +0/-1 (1 lines); hunks: -2,7 +2,6
  - `.buildkite/performance-benchmarks/tests/serving-tests-arm64-cpu.json` modified +0/-1 (1 lines); hunks: -13,7 +13,6
- 关键代码摘录:

```diff
diff -- vllm/envs.py
@@ -95,7 +95,6 @@
-    VLLM_RPC_TIMEOUT: int = 10000  # ms
@@ -1015,9 +1014,6 @@ def _resolve_rust_frontend_path() -> str | None:
-    # Time in ms for the zmq client to wait for a response from the backend
-    # server for simple data operations
-    "VLLM_RPC_TIMEOUT": lambda: int(os.getenv("VLLM_RPC_TIMEOUT", "10000")),
diff -- docs/contributing/profiling.md
@@ -35,8 +35,7 @@ Traces can be visualized using <https://ui.perfetto.dev/>.
-    Set the env variable VLLM_RPC_TIMEOUT to a big number before you start the server. Say something like 30 minutes.
-    `export VLLM_RPC_TIMEOUT=1800000`
+    The engine client waits for this flush to complete without timing out, so simply allow the stop call to run to completion.
diff -- .buildkite/performance-benchmarks/tests/latency-tests-arm64-cpu.json
@@ -2,7 +2,6 @@
-            "VLLM_RPC_TIMEOUT": 100000,
diff -- .buildkite/performance-benchmarks/tests/latency-tests-cpu.json
@@ -2,7 +2,6 @@
-            "VLLM_RPC_TIMEOUT": 100000,
```

- 已读文件:
  - runtime: `vllm/envs.py` modified +0/-4
  - docs: `docs/contributing/profiling.md` modified +1/-2
  - tests: `.buildkite/performance-benchmarks/tests/latency-tests-arm64-cpu.json` modified +0/-1; `.buildkite/performance-benchmarks/tests/latency-tests-cpu.json` modified +0/-1; `.buildkite/performance-benchmarks/tests/serving-tests-arm64-cpu.json` modified +0/-1; `.buildkite/performance-benchmarks/tests/serving-tests-cpu-asr.json` modified +0/-1; `.buildkite/performance-benchmarks/tests/serving-tests-cpu-embed.json` modified +0/-1; `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json` modified +0/-1
- 验证与风险: diff 自带测试面 `.buildkite/performance-benchmarks/tests/latency-tests-arm64-cpu.json`, `.buildkite/performance-benchmarks/tests/latency-tests-cpu.json`, `.buildkite/performance-benchmarks/tests/serving-tests-arm64-cpu.json`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu-asr.json`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #44992 - Deprecations for v0.23 and v0.24

- 链接: https://github.com/vllm-project/vllm/pull/44992
- 状态/时间: merged / 2026-06-11
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 35 个文件，+102/-676，可读 patch 1334 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Deprecations for v0.23 and v0.24」；模型线: Llama 3.1；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`, `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py`, `vllm/model_executor/kernels/linear/__init__.py`；技术摘要: 覆盖「Deprecations for v0.23 and v0.24」；主要实现面是 `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`, `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py`, `vllm/model_executor/kernels/linear/__init__.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` modified +0/-69 (69 lines); hunks: -6,7 +6,6; -465,74 +464,6 @@ def select_mxfp4_moe_backend(; symbols: select_mxfp4_moe_backend，涉及 `select_mxfp4_moe_backend`；`vllm/model_executor/layers/fused_moe/oracle/nvfp4.py` modified +0/-59 (59 lines); hunks: -22,10 +22,6; -58,12 +54,6 @@ class NvFp4MoeBackend(Enum):; symbols: NvFp4MoeBackend, is_global_sf_supported_for_nvfp4_backend, _return_or_raise，涉及 `NvFp4MoeBackend, is_global_sf_supported_for_nvfp4_backend, _return_or_raise`；`vllm/model_executor/kernels/linear/__init__.py` modified +8/-47 (55 lines); hunks: -212,6 +212,9 @@ def _get_linear_backend() -> str:; -392,7 +395,7 @@ def _filter_kernels_by_backend(; symbols: _get_linear_backend, _filter_kernels_by_backend, init_mxfp4_linear_kernel, init_wfp8_a16_linear_kernel，涉及 `_get_linear_backend, _filter_kernels_by_backend, init_mxfp4_linear_kernel`；`vllm/model_executor/layers/fused_moe/oracle/fp8.py` modified +0/-50 (50 lines); hunks: -20,8 +20,6; -321,54 +319,6 @@ def _return_or_raise(; symbols: _return_or_raise，涉及 `_return_or_raise`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` modified +0/-69 (69 lines); hunks: -6,7 +6,6; -465,74 +464,6 @@ def select_mxfp4_moe_backend(; symbols: select_mxfp4_moe_backend
  - `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py` modified +0/-59 (59 lines); hunks: -22,10 +22,6; -58,12 +54,6 @@ class NvFp4MoeBackend(Enum):; symbols: NvFp4MoeBackend, is_global_sf_supported_for_nvfp4_backend, _return_or_raise
  - `vllm/model_executor/kernels/linear/__init__.py` modified +8/-47 (55 lines); hunks: -212,6 +212,9 @@ def _get_linear_backend() -> str:; -392,7 +395,7 @@ def _filter_kernels_by_backend(; symbols: _get_linear_backend, _filter_kernels_by_backend, init_mxfp4_linear_kernel, init_wfp8_a16_linear_kernel
  - `vllm/model_executor/layers/fused_moe/oracle/fp8.py` modified +0/-50 (50 lines); hunks: -20,8 +20,6; -321,54 +319,6 @@ def _return_or_raise(; symbols: _return_or_raise
  - `vllm/model_executor/layers/fused_moe/oracle/unquantized.py` modified +0/-45 (45 lines); hunks: -19,9 +19,7; -230,49 +228,6 @@ def _return_or_raise(; symbols: _return_or_raise
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/fused_moe/oracle/mxfp4.py
@@ -6,7 +6,6 @@
-from vllm import envs
@@ -465,74 +464,6 @@ def select_mxfp4_moe_backend(
-    # Handle explicit FlashInfer MXFP4 BF16 configuration.
-    if envs.is_set("VLLM_USE_FLASHINFER_MOE_MXFP4_BF16"):
-        if not envs.VLLM_USE_FLASHINFER_MOE_MXFP4_BF16:
-            for _b in (
diff -- vllm/model_executor/layers/fused_moe/oracle/nvfp4.py
@@ -22,10 +22,6 @@
-from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
-    FlashinferMoeBackend,
-    get_flashinfer_moe_backend,
-)
@@ -58,12 +54,6 @@ class NvFp4MoeBackend(Enum):
-fi_2_vllm_backend_map: dict[FlashinferMoeBackend, NvFp4MoeBackend] = {
diff -- vllm/model_executor/kernels/linear/__init__.py
@@ -212,6 +212,9 @@ def _get_linear_backend() -> str:
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` modified +0/-69; `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py` modified +0/-59; `vllm/model_executor/kernels/linear/__init__.py` modified +8/-47; `vllm/model_executor/layers/fused_moe/oracle/fp8.py` modified +0/-50; `vllm/model_executor/layers/fused_moe/oracle/unquantized.py` modified +0/-45; `vllm/entrypoints/pooling/offline.py` modified +0/-44
- 验证与风险: diff 自带测试面 `tests/compile/correctness_e2e/test_async_tp.py`, `tests/conftest.py`, `tests/distributed/test_eplb_fused_moe_layer_dep_nvfp4.py`, `tests/entrypoints/pooling/reward/test_token_reward_offline.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
