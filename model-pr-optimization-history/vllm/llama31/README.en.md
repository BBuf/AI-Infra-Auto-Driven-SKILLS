# vllm Llama 3.1 Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `examples/tool_chat_template_llama3.1_json.jinja` | [#8343](https://github.com/vllm-project/vllm/pull/8343) |

## PR Coverage Summary

- Git-traced PRs: 1
- Extra PRs preserved from existing docs: 18
- Total PRs in this document: 19
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
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

## Per-PR Diff Audit Cards

### PR #8343 - [Feature] Add support for Llama 3.1 and 3.2 tool use

- Link: https://github.com/vllm-project/vllm/pull/8343
- Status/date: merged / 2024-09-27
- Trace source: `git log --name-only -- <model-files>` found it through `examples/tool_chat_template_llama3.1_json.jinja`; associated commits `344cd2b6f4c2`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +576/-27, 741 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feature] Add support for Llama 3.1 and 3.2 tool use"; model line: Llama 3.1; category: model support/runtime entry; main diff: `examples/tool_chat_template_llama3.1_json.jinja`; technical summary: Covers "[Feature] Add support for Llama 3.1 and 3.2 tool use"; the main implementation surface is `examples/tool_chat_template_llama3.1_json.jinja`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `examples/tool_chat_template_llama3.1_json.jinja` added +94/-0 (94 lines); hunks: -0,0 +1,94.
- Code diff details:
  - `examples/tool_chat_template_llama3.1_json.jinja` added +94/-0 (94 lines); hunks: -0,0 +1,94
- Key code excerpts:

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

- Reviewed files:
  - docs: `examples/tool_chat_template_llama3.1_json.jinja` added +94/-0
- Risk and verification: The diff ships test coverage in `tests/tool_use/test_chat_completions.py`, `tests/tool_use/test_parallel_tool_calls.py`, `tests/tool_use/utils.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #10164 - [Bugfix][Frontend] Update Llama Chat Templates to also support Non-Tool use

- Link: https://github.com/vllm-project/vllm/pull/10164
- Status/date: merged / 2024-11-23
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +110/-36, 240 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix][Frontend] Update Llama Chat Templates to also support Non-Tool use"; model line: Llama 3.1; category: bug fix; main diff: `tests/entrypoints/test_chat_utils.py`, `examples/tool_chat_template_llama3.2_json.jinja`, `examples/tool_chat_template_llama3.1_json.jinja`; technical summary: Covers "[Bugfix][Frontend] Update Llama Chat Templates to also support Non-Tool use"; the main implementation surface is `tests/entrypoints/test_chat_utils.py`, `examples/tool_chat_template_llama3.2_json.jinja`, `examples/tool_chat_template_llama3.1_json.jinja`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/entrypoints/test_chat_utils.py` modified +2/-2 (4 lines); hunks: -766,8 +766,8 @@ def test_resolve_content_format_hf_defined(model, expected_f...; symbols: test_resolve_content_format_hf_defined, touching `test_resolve_content_format_hf_defined`; `examples/tool_chat_template_llama3.2_json.jinja` modified +72/-24 (96 lines); hunks: -16,46 +16,78; -66,7 +98,19; `examples/tool_chat_template_llama3.1_json.jinja` modified +36/-10 (46 lines); hunks: -19,10 +19,18; -33,8 +41,8.
- Code diff details:
  - `tests/entrypoints/test_chat_utils.py` modified +2/-2 (4 lines); hunks: -766,8 +766,8 @@ def test_resolve_content_format_hf_defined(model, expected_f...; symbols: test_resolve_content_format_hf_defined
  - `examples/tool_chat_template_llama3.2_json.jinja` modified +72/-24 (96 lines); hunks: -16,46 +16,78; -66,7 +98,19
  - `examples/tool_chat_template_llama3.1_json.jinja` modified +36/-10 (46 lines); hunks: -19,10 +19,18; -33,8 +41,8
- Key code excerpts:

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

- Reviewed files:
  - tests: `tests/entrypoints/test_chat_utils.py` modified +2/-2
  - docs: `examples/tool_chat_template_llama3.2_json.jinja` modified +72/-24; `examples/tool_chat_template_llama3.1_json.jinja` modified +36/-10
- Risk and verification: The diff ships test coverage in `tests/entrypoints/test_chat_utils.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #25786 - [Benchmark] Cleanup deprecated nightly benchmark and adjust the docstring for performance benchmark

- Link: https://github.com/vllm-project/vllm/pull/25786
- Status/date: merged / 2025-10-30
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 29 files, +10/-1289, 1387 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Benchmark] Cleanup deprecated nightly benchmark and adjust the docstring for performance benchmark"; model line: Llama 3.1; category: performance/backend optimization; main diff: `.buildkite/nightly-benchmarks/scripts/download-tokenizer.py`, `.buildkite/nightly-benchmarks/scripts/run-nightly-benchmarks.sh`, `.buildkite/nightly-benchmarks/nightly-pipeline.yaml`; technical summary: Covers "[Benchmark] Cleanup deprecated nightly benchmark and adjust the docstring for performance benchmark"; the main implementation surface is `.buildkite/nightly-benchmarks/scripts/download-tokenizer.py`, `.buildkite/nightly-benchmarks/scripts/run-nightly-benchmarks.sh`, `.buildkite/nightly-benchmarks/nightly-pipeline.yaml`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `.buildkite/nightly-benchmarks/scripts/download-tokenizer.py` removed +0/-26 (26 lines); hunks: -1,26 +0,0; symbols: main, touching `main`; `.buildkite/nightly-benchmarks/scripts/run-nightly-benchmarks.sh` removed +0/-464 (464 lines); hunks: -1,464 +0,0; `.buildkite/nightly-benchmarks/nightly-pipeline.yaml` removed +0/-196 (196 lines); hunks: -1,196 +0,0; `.buildkite/nightly-benchmarks/benchmark-pipeline.yaml` removed +0/-184 (184 lines); hunks: -1,184 +0,0.
- Code diff details:
  - `.buildkite/nightly-benchmarks/scripts/download-tokenizer.py` removed +0/-26 (26 lines); hunks: -1,26 +0,0; symbols: main
  - `.buildkite/nightly-benchmarks/scripts/run-nightly-benchmarks.sh` removed +0/-464 (464 lines); hunks: -1,464 +0,0
  - `.buildkite/nightly-benchmarks/nightly-pipeline.yaml` removed +0/-196 (196 lines); hunks: -1,196 +0,0
  - `.buildkite/nightly-benchmarks/benchmark-pipeline.yaml` removed +0/-184 (184 lines); hunks: -1,184 +0,0
  - `.buildkite/nightly-benchmarks/scripts/generate-nightly-markdown.py` removed +0/-97 (97 lines); hunks: -1,97 +0,0; symbols: parse_arguments, get_perf, get_perf_w_std, main
- Key code excerpts:

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

- Reviewed files:
  - runtime: `.buildkite/nightly-benchmarks/scripts/download-tokenizer.py` removed +0/-26
  - other: `.buildkite/nightly-benchmarks/scripts/run-nightly-benchmarks.sh` removed +0/-464; `.buildkite/nightly-benchmarks/nightly-pipeline.yaml` removed +0/-196; `.buildkite/nightly-benchmarks/benchmark-pipeline.yaml` removed +0/-184; `.buildkite/nightly-benchmarks/scripts/generate-nightly-markdown.py` removed +0/-97; `.buildkite/nightly-benchmarks/scripts/summary-nightly-results.py` removed +0/-82; `.buildkite/nightly-benchmarks/scripts/nightly-annotate.sh` removed +0/-78
- Risk and verification: The diff ships test coverage in `.buildkite/performance-benchmarks/tests/genai-perf-tests.json`, `.buildkite/performance-benchmarks/tests/latency-tests-cpu.json`, `.buildkite/performance-benchmarks/tests/latency-tests.json`, `.buildkite/performance-benchmarks/tests/nightly-tests.json`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #33731 - [torch.compile] Reorganize vllm/compilation and tests/compile (0/N for vLLM IR)

- Link: https://github.com/vllm-project/vllm/pull/33731
- Status/date: merged / 2026-02-06
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 47 files, +717/-651, 1985 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[torch.compile] Reorganize vllm/compilation and tests/compile (0/N for vLLM IR)"; model line: Llama 3.1; category: docs/tests/CI; main diff: `vllm/compilation/passes/fusion/collective_fusion.py`, `vllm/compilation/passes/fusion/allreduce_rms_fusion.py`, `tests/compile/passes/distributed/test_async_tp.py`; technical summary: Covers "[torch.compile] Reorganize vllm/compilation and tests/compile (0/N for vLLM IR)"; the main implementation surface is `vllm/compilation/passes/fusion/collective_fusion.py`, `vllm/compilation/passes/fusion/allreduce_rms_fusion.py`, `tests/compile/passes/distributed/test_async_tp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/compilation/passes/fusion/collective_fusion.py` added +423/-0 (423 lines); hunks: -0,0 +1,423; symbols: BasePattern, __init__, GEMMReduceScatterPattern, get_inputs, touching `BasePattern, __init__, GEMMReduceScatterPattern`; `vllm/compilation/passes/fusion/allreduce_rms_fusion.py` renamed +13/-403 (416 lines); hunks: -8,7 +8,6; -24,12 +23,14; symbols: BasePattern, __init__, GEMMReduceScatterPattern, get_inputs, touching `BasePattern, __init__, GEMMReduceScatterPattern`; `tests/compile/passes/distributed/test_async_tp.py` renamed +5/-75 (80 lines); hunks: -1,16 +1,18; -29,14 +31,6; symbols: async_tp_pass_on_test_model, test_async_tp_pass_correctness, touching `async_tp_pass_on_test_model, test_async_tp_pass_correctness`; `tests/compile/correctness_e2e/test_async_tp.py` added +79/-0 (79 lines); hunks: -0,0 +1,79; symbols: test_async_tp_pass_correctness, touching `test_async_tp_pass_correctness`.
- Code diff details:
  - `vllm/compilation/passes/fusion/collective_fusion.py` added +423/-0 (423 lines); hunks: -0,0 +1,423; symbols: BasePattern, __init__, GEMMReduceScatterPattern, get_inputs
  - `vllm/compilation/passes/fusion/allreduce_rms_fusion.py` renamed +13/-403 (416 lines); hunks: -8,7 +8,6; -24,12 +23,14; symbols: BasePattern, __init__, GEMMReduceScatterPattern, get_inputs
  - `tests/compile/passes/distributed/test_async_tp.py` renamed +5/-75 (80 lines); hunks: -1,16 +1,18; -29,14 +31,6; symbols: async_tp_pass_on_test_model, test_async_tp_pass_correctness
  - `tests/compile/correctness_e2e/test_async_tp.py` added +79/-0 (79 lines); hunks: -0,0 +1,79; symbols: test_async_tp_pass_correctness
  - `.buildkite/test_areas/compile.yaml` modified +24/-18 (42 lines); hunks: -2,7 +2,7 @@ group: Compile; -11,37 +11,43 @@ steps:
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/compilation/passes/fusion/collective_fusion.py` added +423/-0; `vllm/compilation/passes/fusion/allreduce_rms_fusion.py` renamed +13/-403
  - tests: `tests/compile/passes/distributed/test_async_tp.py` renamed +5/-75; `tests/compile/correctness_e2e/test_async_tp.py` added +79/-0; `tests/compile/passes/test_fusion.py` renamed +16/-11; `tests/compile/passes/test_functionalization.py` renamed +12/-9
  - other: `.buildkite/test_areas/compile.yaml` modified +24/-18; `.buildkite/test-amd.yaml` modified +19/-19
- Risk and verification: The diff ships test coverage in `tests/compile/backend.py`, `tests/compile/correctness_e2e/__init__.py`, `tests/compile/correctness_e2e/test_async_tp.py`, `tests/compile/correctness_e2e/test_sequence_parallel.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #34128 - Vllm CPU benchmark suite improvement

- Link: https://github.com/vllm-project/vllm/pull/34128
- Status/date: merged / 2026-02-12
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +802/-254, 1243 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Vllm CPU benchmark suite improvement"; model line: Llama 3.1; category: performance/backend optimization; main diff: `.buildkite/performance-benchmarks/scripts/compare-json-results.py`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json`, `.buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh`; technical summary: Covers "Vllm CPU benchmark suite improvement"; the main implementation surface is `.buildkite/performance-benchmarks/scripts/compare-json-results.py`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json`, `.buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `.buildkite/performance-benchmarks/scripts/compare-json-results.py` modified +368/-77 (445 lines); hunks: -9,8 +9,10; -275,6 +277,131 @@ def _apply_two_decimals(; symbols: _apply_two_decimals, _sanitize_sheet_name, _group_to_sheet_base, _write_tables_to_excel_sheet, touching `_apply_two_decimals, _sanitize_sheet_name, _group_to_sheet_base`; `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json` added +283/-0 (283 lines); hunks: -0,0 +1,283; `.buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh` modified +87/-46 (133 lines); hunks: -1,6 +1,4; -9,6 +7,11; `.buildkite/performance-benchmarks/tests/serving-tests-cpu.json` modified +0/-130 (130 lines); hunks: -148,136 +148,6.
- Code diff details:
  - `.buildkite/performance-benchmarks/scripts/compare-json-results.py` modified +368/-77 (445 lines); hunks: -9,8 +9,10; -275,6 +277,131 @@ def _apply_two_decimals(; symbols: _apply_two_decimals, _sanitize_sheet_name, _group_to_sheet_base, _write_tables_to_excel_sheet
  - `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json` added +283/-0 (283 lines); hunks: -0,0 +1,283
  - `.buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh` modified +87/-46 (133 lines); hunks: -1,6 +1,4; -9,6 +7,11
  - `.buildkite/performance-benchmarks/tests/serving-tests-cpu.json` modified +0/-130 (130 lines); hunks: -148,136 +148,6
  - `.buildkite/performance-benchmarks/tests/serving-tests-cpu-embed.json` added +41/-0 (41 lines); hunks: -0,0 +1,41
- Key code excerpts:

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

- Reviewed files:
  - other: `.buildkite/performance-benchmarks/scripts/compare-json-results.py` modified +368/-77; `.buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh` modified +87/-46
  - tests: `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json` added +283/-0; `.buildkite/performance-benchmarks/tests/serving-tests-cpu.json` modified +0/-130; `.buildkite/performance-benchmarks/tests/serving-tests-cpu-embed.json` added +41/-0
  - docs: `docs/getting_started/installation/cpu.md` modified +23/-1
- Risk and verification: The diff ships test coverage in `.buildkite/performance-benchmarks/tests/serving-tests-cpu-embed.json`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu.json`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #34716 - [BugFix] Fix sp tests

- Link: https://github.com/vllm-project/vllm/pull/34716
- Status/date: merged / 2026-02-17
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BugFix] Fix sp tests"; model line: Llama 3.1; category: bug fix; main diff: `tests/compile/correctness_e2e/test_sequence_parallel.py`; technical summary: Covers "[BugFix] Fix sp tests"; the main implementation surface is `tests/compile/correctness_e2e/test_sequence_parallel.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/compile/correctness_e2e/test_sequence_parallel.py` modified +1/-1 (2 lines); hunks: -229,7 +229,7 @@ def _compare_sp(; symbols: _compare_sp, touching `_compare_sp`.
- Code diff details:
  - `tests/compile/correctness_e2e/test_sequence_parallel.py` modified +1/-1 (2 lines); hunks: -229,7 +229,7 @@ def _compare_sp(; symbols: _compare_sp
- Key code excerpts:

```diff
diff -- tests/compile/correctness_e2e/test_sequence_parallel.py
@@ -229,7 +229,7 @@ def _compare_sp(
-        common_args.append("--enforce-eager")
+        common_args.append("-cc.cudagraph_mode=none")
```

- Reviewed files:
  - tests: `tests/compile/correctness_e2e/test_sequence_parallel.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `tests/compile/correctness_e2e/test_sequence_parallel.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #35871 - [CI] Add Blackwell AsyncTP correctness test

- Link: https://github.com/vllm-project/vllm/pull/35871
- Status/date: merged / 2026-03-04
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +15/-0, 30 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CI] Add Blackwell AsyncTP correctness test"; model line: Llama 3.1; category: docs/tests/CI; main diff: `.buildkite/test_areas/compile.yaml`, `tests/compile/correctness_e2e/test_async_tp.py`; technical summary: Covers "[CI] Add Blackwell AsyncTP correctness test"; the main implementation surface is `.buildkite/test_areas/compile.yaml`, `tests/compile/correctness_e2e/test_async_tp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `.buildkite/test_areas/compile.yaml` modified +10/-0 (10 lines); hunks: -36,6 +36,16 @@ steps:; `tests/compile/correctness_e2e/test_async_tp.py` modified +5/-0 (5 lines); hunks: -31,7 +31,12 @@ def test_async_tp_pass_correctness(; symbols: test_async_tp_pass_correctness, touching `test_async_tp_pass_correctness`.
- Code diff details:
  - `.buildkite/test_areas/compile.yaml` modified +10/-0 (10 lines); hunks: -36,6 +36,16 @@ steps:
  - `tests/compile/correctness_e2e/test_async_tp.py` modified +5/-0 (5 lines); hunks: -31,7 +31,12 @@ def test_async_tp_pass_correctness(; symbols: test_async_tp_pass_correctness
- Key code excerpts:

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

- Reviewed files:
  - other: `.buildkite/test_areas/compile.yaml` modified +10/-0
  - tests: `tests/compile/correctness_e2e/test_async_tp.py` modified +5/-0
- Risk and verification: The diff ships test coverage in `tests/compile/correctness_e2e/test_async_tp.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #36216 - [V0 Deprecation] Remove unused swap_space parameter

- Link: https://github.com/vllm-project/vllm/pull/36216
- Status/date: merged / 2026-03-07
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 22 files, +19/-79, 395 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[V0 Deprecation] Remove unused swap_space parameter"; model line: Llama 3.1; category: docs/tests/CI; main diff: `vllm/entrypoints/llm.py`, `vllm/config/cache.py`, `docs/design/metrics.md`; technical summary: Covers "[V0 Deprecation] Remove unused swap_space parameter"; the main implementation surface is `vllm/entrypoints/llm.py`, `vllm/config/cache.py`, `docs/design/metrics.md`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/entrypoints/llm.py` modified +11/-8 (19 lines); hunks: -164,12 +164,6 @@ class LLM:; -240,7 +234,6 @@ def __init__(; symbols: LLM, __init__, _make_config, touching `LLM, __init__, _make_config`; `vllm/config/cache.py` modified +1/-33 (34 lines); hunks: -1,21 +1,13; -53,8 +45,6 @@ class CacheConfig:; symbols: CacheConfig, compute_hash, _validate_cache_dtype, verify_with_parallel_config, touching `CacheConfig, compute_hash, _validate_cache_dtype`; `docs/design/metrics.md` modified +4/-4 (8 lines); hunks: -507,10 +507,10 @@ longer relevant in v1:; `.buildkite/performance-benchmarks/tests/serving-tests-hpu.json` modified +0/-4 (4 lines); hunks: -10,7 +10,6; -37,7 +36,6.
- Code diff details:
  - `vllm/entrypoints/llm.py` modified +11/-8 (19 lines); hunks: -164,12 +164,6 @@ class LLM:; -240,7 +234,6 @@ def __init__(; symbols: LLM, __init__, _make_config
  - `vllm/config/cache.py` modified +1/-33 (34 lines); hunks: -1,21 +1,13; -53,8 +45,6 @@ class CacheConfig:; symbols: CacheConfig, compute_hash, _validate_cache_dtype, verify_with_parallel_config
  - `docs/design/metrics.md` modified +4/-4 (8 lines); hunks: -507,10 +507,10 @@ longer relevant in v1:
  - `.buildkite/performance-benchmarks/tests/serving-tests-hpu.json` modified +0/-4 (4 lines); hunks: -10,7 +10,6; -37,7 +36,6
  - `.buildkite/performance-benchmarks/tests/serving-tests.json` modified +0/-4 (4 lines); hunks: -5,7 +5,6; -23,7 +22,6
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/entrypoints/llm.py` modified +11/-8; `vllm/config/cache.py` modified +1/-33
  - docs: `docs/design/metrics.md` modified +4/-4
  - tests: `.buildkite/performance-benchmarks/tests/serving-tests-hpu.json` modified +0/-4; `.buildkite/performance-benchmarks/tests/serving-tests.json` modified +0/-4; `tests/distributed/test_torchrun_example.py` modified +1/-2; `tests/distributed/test_torchrun_example_moe.py` modified +1/-2; `tests/v1/worker/test_gpu_model_runner.py` modified +0/-3
- Risk and verification: The diff ships test coverage in `.buildkite/performance-benchmarks/tests/serving-tests-hpu.json`, `.buildkite/performance-benchmarks/tests/serving-tests.json`, `tests/conftest.py`, `tests/distributed/test_torchrun_example.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #35086 - more models for vLLM Benchmark Suite

- Link: https://github.com/vllm-project/vllm/pull/35086
- Status/date: merged / 2026-03-12
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +800/-119, 1301 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "more models for vLLM Benchmark Suite"; model line: Llama 3.1; category: performance/backend optimization; main diff: `.buildkite/performance-benchmarks/scripts/compare-json-results.py`, `.buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json`; technical summary: Covers "more models for vLLM Benchmark Suite"; the main implementation surface is `.buildkite/performance-benchmarks/scripts/compare-json-results.py`, `.buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `.buildkite/performance-benchmarks/scripts/compare-json-results.py` modified +301/-90 (391 lines); hunks: -7,12 +7,12; -33,6 +33,45; symbols: _find_concurrency_col, _normalize_concurrency_in_df, compare_data_columns, touching `_find_concurrency_col, _normalize_concurrency_in_df, compare_data_columns`; `.buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh` modified +361/-4 (365 lines); hunks: -12,6 +12,13 @@ DRY_RUN="${DRY_RUN:-0}"; -183,6 +190,304 @@ upload_to_buildkite() {; `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json` modified +72/-0 (72 lines); hunks: -149,6 +149,39; -188,6 +221,45; `.buildkite/performance-benchmarks/tests/serving-tests-cpu-asr.json` added +37/-0 (37 lines); hunks: -0,0 +1,37.
- Code diff details:
  - `.buildkite/performance-benchmarks/scripts/compare-json-results.py` modified +301/-90 (391 lines); hunks: -7,12 +7,12; -33,6 +33,45; symbols: _find_concurrency_col, _normalize_concurrency_in_df, compare_data_columns
  - `.buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh` modified +361/-4 (365 lines); hunks: -12,6 +12,13 @@ DRY_RUN="${DRY_RUN:-0}"; -183,6 +190,304 @@ upload_to_buildkite() {
  - `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json` modified +72/-0 (72 lines); hunks: -149,6 +149,39; -188,6 +221,45
  - `.buildkite/performance-benchmarks/tests/serving-tests-cpu-asr.json` added +37/-0 (37 lines); hunks: -0,0 +1,37
  - `.buildkite/performance-benchmarks/tests/serving-tests-cpu.json` modified +12/-23 (35 lines); hunks: -72,17 +72,6; -106,20 +95,20
- Key code excerpts:

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

- Reviewed files:
  - other: `.buildkite/performance-benchmarks/scripts/compare-json-results.py` modified +301/-90; `.buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh` modified +361/-4; `requirements/test.txt` modified +7/-1; `requirements/test.in` modified +4/-1
  - tests: `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json` modified +72/-0; `.buildkite/performance-benchmarks/tests/serving-tests-cpu-asr.json` added +37/-0; `.buildkite/performance-benchmarks/tests/serving-tests-cpu.json` modified +12/-23
  - docs: `docs/benchmarking/dashboard.md` modified +6/-0
- Risk and verification: The diff ships test coverage in `.buildkite/performance-benchmarks/tests/serving-tests-cpu-asr.json`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu.json`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #38576 - vLLM Benchmark Suite perf regression after PR#32723

- Link: https://github.com/vllm-project/vllm/pull/38576
- Status/date: merged / 2026-03-31
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +15/-1, 119 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "vLLM Benchmark Suite perf regression after PR#32723"; model line: Llama 3.1; category: bug fix; main diff: `.buildkite/performance-benchmarks/tests/serving-tests-hpu.json`, `.buildkite/performance-benchmarks/tests/serving-tests.json`, `.buildkite/performance-benchmarks/tests/serving-tests-arm64-cpu.json`; technical summary: Covers "vLLM Benchmark Suite perf regression after PR#32723"; the main implementation surface is `.buildkite/performance-benchmarks/tests/serving-tests-hpu.json`, `.buildkite/performance-benchmarks/tests/serving-tests.json`, `.buildkite/performance-benchmarks/tests/serving-tests-arm64-cpu.json`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `.buildkite/performance-benchmarks/tests/serving-tests-hpu.json` modified +6/-0 (6 lines); hunks: -21,6 +21,7; -47,6 +48,7; `.buildkite/performance-benchmarks/tests/serving-tests.json` modified +4/-0 (4 lines); hunks: -13,6 +13,7; -30,6 +31,7; `.buildkite/performance-benchmarks/tests/serving-tests-arm64-cpu.json` modified +2/-1 (3 lines); hunks: -36,6 +36,7; -127,4 +128,4; `.buildkite/performance-benchmarks/tests/serving-tests-cpu-asr.json` modified +1/-0 (1 lines); hunks: -22,6 +22,7.
- Code diff details:
  - `.buildkite/performance-benchmarks/tests/serving-tests-hpu.json` modified +6/-0 (6 lines); hunks: -21,6 +21,7; -47,6 +48,7
  - `.buildkite/performance-benchmarks/tests/serving-tests.json` modified +4/-0 (4 lines); hunks: -13,6 +13,7; -30,6 +31,7
  - `.buildkite/performance-benchmarks/tests/serving-tests-arm64-cpu.json` modified +2/-1 (3 lines); hunks: -36,6 +36,7; -127,4 +128,4
  - `.buildkite/performance-benchmarks/tests/serving-tests-cpu-asr.json` modified +1/-0 (1 lines); hunks: -22,6 +22,7
  - `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json` modified +1/-0 (1 lines); hunks: -26,6 +26,7
- Key code excerpts:

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

- Reviewed files:
  - tests: `.buildkite/performance-benchmarks/tests/serving-tests-hpu.json` modified +6/-0; `.buildkite/performance-benchmarks/tests/serving-tests.json` modified +4/-0; `.buildkite/performance-benchmarks/tests/serving-tests-arm64-cpu.json` modified +2/-1; `.buildkite/performance-benchmarks/tests/serving-tests-cpu-asr.json` modified +1/-0; `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json` modified +1/-0; `.buildkite/performance-benchmarks/tests/serving-tests-cpu.json` modified +1/-0
- Risk and verification: The diff ships test coverage in `.buildkite/performance-benchmarks/tests/serving-tests-arm64-cpu.json`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu-asr.json`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu.json`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #38373 - [torch.compile]: Disable Sequence Parallelism (SP) for piecewise compilation

- Link: https://github.com/vllm-project/vllm/pull/38373
- Status/date: merged / 2026-04-26
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +223/-80, 450 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[torch.compile]: Disable Sequence Parallelism (SP) for piecewise compilation"; model line: Llama 3.1; category: docs/tests/CI; main diff: `tests/compile/test_config.py`, `vllm/config/vllm.py`, `vllm/compilation/passes/fusion/sequence_parallelism.py`; technical summary: Covers "[torch.compile]: Disable Sequence Parallelism (SP) for piecewise compilation"; the main implementation surface is `tests/compile/test_config.py`, `vllm/config/vllm.py`, `vllm/compilation/passes/fusion/sequence_parallelism.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/compile/test_config.py` modified +118/-1 (119 lines); hunks: -407,7 +407,7 @@ def test_should_split():; -465,6 +465,123 @@ def test_cudagraph_sizes_post_init(; symbols: test_should_split, test_cudagraph_sizes_post_init, test_sequence_parallelism_requires_full_graph_compilation, test_cached_compilation_config, touching `test_should_split, test_cudagraph_sizes_post_init, test_sequence_parallelism_requires_full_graph_compilation`; `vllm/config/vllm.py` modified +17/-28 (45 lines); hunks: -983,19 +983,16 @@ def has_blocked_weights():; -1015,8 +1012,8 @@ def has_blocked_weights():; symbols: has_blocked_weights, touching `has_blocked_weights`; `vllm/compilation/passes/fusion/sequence_parallelism.py` modified +16/-26 (42 lines); hunks: -341,22 +341,18 @@ class SequenceParallelismPass(VllmPatternMatcherPass):; -419,19 +415,13 @@ def is_applicable_for_range(self, compile_range: Range) ->...; symbols: SequenceParallelismPass, is_applicable_for_range, touching `SequenceParallelismPass, is_applicable_for_range`; `vllm/v1/worker/utils.py` modified +8/-15 (23 lines); hunks: -519,12 +519,8 @@ def is_residual_scattered_for_sp(; -534,16 +530,13 @@ def is_residual_scattered_for_sp(; symbols: is_residual_scattered_for_sp, touching `is_residual_scattered_for_sp`.
- Code diff details:
  - `tests/compile/test_config.py` modified +118/-1 (119 lines); hunks: -407,7 +407,7 @@ def test_should_split():; -465,6 +465,123 @@ def test_cudagraph_sizes_post_init(; symbols: test_should_split, test_cudagraph_sizes_post_init, test_sequence_parallelism_requires_full_graph_compilation, test_cached_compilation_config
  - `vllm/config/vllm.py` modified +17/-28 (45 lines); hunks: -983,19 +983,16 @@ def has_blocked_weights():; -1015,8 +1012,8 @@ def has_blocked_weights():; symbols: has_blocked_weights
  - `vllm/compilation/passes/fusion/sequence_parallelism.py` modified +16/-26 (42 lines); hunks: -341,22 +341,18 @@ class SequenceParallelismPass(VllmPatternMatcherPass):; -419,19 +415,13 @@ def is_applicable_for_range(self, compile_range: Range) ->...; symbols: SequenceParallelismPass, is_applicable_for_range
  - `vllm/v1/worker/utils.py` modified +8/-15 (23 lines); hunks: -519,12 +519,8 @@ def is_residual_scattered_for_sp(; -534,16 +530,13 @@ def is_residual_scattered_for_sp(; symbols: is_residual_scattered_for_sp
  - `tests/compile/passes/distributed/test_sequence_parallelism.py` modified +19/-0 (19 lines); hunks: -22,6 +22,7; -216,6 +217,24 @@ def run_torch_spawn(fn, nprocs):; symbols: run_torch_spawn, test_sequence_parallelism_pass_requires_full_graph_compilation, sequence_parallelism_pass_on_test_model
- Key code excerpts:

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

- Reviewed files:
  - tests: `tests/compile/test_config.py` modified +118/-1; `tests/compile/passes/distributed/test_sequence_parallelism.py` modified +19/-0; `tests/compile/passes/distributed/test_async_tp.py` modified +17/-0
  - runtime: `vllm/config/vllm.py` modified +17/-28; `vllm/compilation/passes/fusion/sequence_parallelism.py` modified +16/-26; `vllm/v1/worker/utils.py` modified +8/-15; `vllm/config/compilation.py` modified +19/-0; `vllm/compilation/passes/fusion/collective_fusion.py` modified +7/-10
- Risk and verification: The diff ships test coverage in `tests/compile/correctness_e2e/test_sequence_parallel.py`, `tests/compile/passes/distributed/test_async_tp.py`, `tests/compile/passes/distributed/test_sequence_parallelism.py`, `tests/compile/test_config.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #41882 - Add NVFP4 all-gather GEMM fusion for AsyncTP

- Link: https://github.com/vllm-project/vllm/pull/41882
- Status/date: merged / 2026-05-10
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +605/-6, 781 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add NVFP4 all-gather GEMM fusion for AsyncTP"; model line: Llama 3.1; category: performance/backend optimization; main diff: `vllm/compilation/passes/fusion/collective_fusion.py`, `vllm/compilation/passes/fusion/sequence_parallelism.py`, `tests/compile/correctness_e2e/test_async_tp.py`; technical summary: Covers "Add NVFP4 all-gather GEMM fusion for AsyncTP"; the main implementation surface is `vllm/compilation/passes/fusion/collective_fusion.py`, `vllm/compilation/passes/fusion/sequence_parallelism.py`, `tests/compile/correctness_e2e/test_async_tp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/compilation/passes/fusion/collective_fusion.py` modified +243/-0 (243 lines); hunks: -74,6 +74,36 @@ def _flashinfer_scaled_mm_out(; -197,6 +227,90 @@ def fused_all_gather_flashinfer_scaled_matmul(; symbols: _flashinfer_scaled_mm_out, _flashinfer_fp4_mm_out, fused_flashinfer_scaled_matmul_reduce_scatter_fake, fused_all_gather_flashinfer_scaled_matmul, touching `_flashinfer_scaled_mm_out, _flashinfer_fp4_mm_out, fused_flashinfer_scaled_matmul_reduce_scatter_fake`; `vllm/compilation/passes/fusion/sequence_parallelism.py` modified +136/-0 (136 lines); hunks: -8,6 +8,7; -27,6 +28,10; symbols: replacement, FirstAllReduceRMSNormStaticNVFP4Pattern, get_inputs, register, touching `replacement, FirstAllReduceRMSNormStaticNVFP4Pattern, get_inputs`; `tests/compile/correctness_e2e/test_async_tp.py` modified +73/-0 (73 lines); hunks: -13,6 +13,17; -82,3 +93,65 @@ def test_async_tp_pass_correctness(; symbols: test_async_tp_pass_correctness, test_async_tp_pass_nvfp4_correctness, touching `test_async_tp_pass_correctness, test_async_tp_pass_nvfp4_correctness`; `tests/compile/fusions_e2e/test_tp2_async_tp.py` modified +65/-0 (65 lines); hunks: -13,11 +13,13; -90,6 +92,69 @@ def test_tp2_async_tp_fp8_fusions(; symbols: test_tp2_async_tp_fp8_fusions, test_tp2_async_tp_nvfp4_fusions, touching `test_tp2_async_tp_fp8_fusions, test_tp2_async_tp_nvfp4_fusions`.
- Code diff details:
  - `vllm/compilation/passes/fusion/collective_fusion.py` modified +243/-0 (243 lines); hunks: -74,6 +74,36 @@ def _flashinfer_scaled_mm_out(; -197,6 +227,90 @@ def fused_all_gather_flashinfer_scaled_matmul(; symbols: _flashinfer_scaled_mm_out, _flashinfer_fp4_mm_out, fused_flashinfer_scaled_matmul_reduce_scatter_fake, fused_all_gather_flashinfer_scaled_matmul
  - `vllm/compilation/passes/fusion/sequence_parallelism.py` modified +136/-0 (136 lines); hunks: -8,6 +8,7; -27,6 +28,10; symbols: replacement, FirstAllReduceRMSNormStaticNVFP4Pattern, get_inputs, register
  - `tests/compile/correctness_e2e/test_async_tp.py` modified +73/-0 (73 lines); hunks: -13,6 +13,17; -82,3 +93,65 @@ def test_async_tp_pass_correctness(; symbols: test_async_tp_pass_correctness, test_async_tp_pass_nvfp4_correctness
  - `tests/compile/fusions_e2e/test_tp2_async_tp.py` modified +65/-0 (65 lines); hunks: -13,11 +13,13; -90,6 +92,69 @@ def test_tp2_async_tp_fp8_fusions(; symbols: test_tp2_async_tp_fp8_fusions, test_tp2_async_tp_nvfp4_fusions
  - `tests/compile/correctness_e2e/test_sequence_parallel.py` modified +44/-5 (49 lines); hunks: -21,12 +21,14; -41,6 +43,7 @@ class ParallelSetup(NamedTuple):; symbols: ParallelSetup, SPTestOptions, _compare_sp
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/compilation/passes/fusion/collective_fusion.py` modified +243/-0; `vllm/compilation/passes/fusion/sequence_parallelism.py` modified +136/-0; `vllm/utils/flashinfer.py` modified +42/-0
  - tests: `tests/compile/correctness_e2e/test_async_tp.py` modified +73/-0; `tests/compile/fusions_e2e/test_tp2_async_tp.py` modified +65/-0; `tests/compile/correctness_e2e/test_sequence_parallel.py` modified +44/-5; `tests/compile/fullgraph/test_toy_llama.py` modified +2/-1
- Risk and verification: The diff ships test coverage in `tests/compile/correctness_e2e/test_async_tp.py`, `tests/compile/correctness_e2e/test_sequence_parallel.py`, `tests/compile/fullgraph/test_toy_llama.py`, `tests/compile/fusions_e2e/test_tp2_async_tp.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #33322 - [Bugfix] Fix SP pass for multimodal models and PP+SP residual handling

- Link: https://github.com/vllm-project/vllm/pull/33322
- Status/date: merged / 2026-05-10
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +116/-34, 260 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix SP pass for multimodal models and PP+SP residual handling"; model line: Llama 3.1; category: bug fix; main diff: `vllm/compilation/passes/fusion/sequence_parallelism.py`, `tests/compile/correctness_e2e/test_sequence_parallel.py`, `vllm/v1/worker/gpu_model_runner.py`; technical summary: Covers "[Bugfix] Fix SP pass for multimodal models and PP+SP residual handling"; the main implementation surface is `vllm/compilation/passes/fusion/sequence_parallelism.py`, `tests/compile/correctness_e2e/test_sequence_parallel.py`, `vllm/v1/worker/gpu_model_runner.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/compilation/passes/fusion/sequence_parallelism.py` modified +55/-20 (75 lines); hunks: -14,7 +14,10; -117,6 +120,7 @@ def __init__(; symbols: __init__, _all_reduce, replacement, SequenceParallelismPass, touching `__init__, _all_reduce, replacement`; `tests/compile/correctness_e2e/test_sequence_parallel.py` modified +48/-0 (48 lines); hunks: -167,6 +167,7 @@ def _compare_sp(; -248,6 +249,8 @@ def _compare_sp(; symbols: _compare_sp, test_tp_sp_generation, test_tp_sp_generation_prompt_embeds, touching `_compare_sp, test_tp_sp_generation, test_tp_sp_generation_prompt_embeds`; `vllm/v1/worker/gpu_model_runner.py` modified +13/-14 (27 lines); hunks: -3098,7 +3098,7 @@ def get_supported_tasks(self) -> tuple[SupportedTask, ...]:; -3109,24 +3109,23 @@ def sync_and_slice_intermediate_tensors(; symbols: get_supported_tasks, sync_and_slice_intermediate_tensors, sync_and_gather_intermediate_tensors, eplb_step, touching `get_supported_tasks, sync_and_slice_intermediate_tensors, sync_and_gather_intermediate_tensors`.
- Code diff details:
  - `vllm/compilation/passes/fusion/sequence_parallelism.py` modified +55/-20 (75 lines); hunks: -14,7 +14,10; -117,6 +120,7 @@ def __init__(; symbols: __init__, _all_reduce, replacement, SequenceParallelismPass
  - `tests/compile/correctness_e2e/test_sequence_parallel.py` modified +48/-0 (48 lines); hunks: -167,6 +167,7 @@ def _compare_sp(; -248,6 +249,8 @@ def _compare_sp(; symbols: _compare_sp, test_tp_sp_generation, test_tp_sp_generation_prompt_embeds
  - `vllm/v1/worker/gpu_model_runner.py` modified +13/-14 (27 lines); hunks: -3098,7 +3098,7 @@ def get_supported_tasks(self) -> tuple[SupportedTask, ...]:; -3109,24 +3109,23 @@ def sync_and_slice_intermediate_tensors(; symbols: get_supported_tasks, sync_and_slice_intermediate_tensors, sync_and_gather_intermediate_tensors, eplb_step
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/compilation/passes/fusion/sequence_parallelism.py` modified +55/-20; `vllm/v1/worker/gpu_model_runner.py` modified +13/-14
  - tests: `tests/compile/correctness_e2e/test_sequence_parallel.py` modified +48/-0
- Risk and verification: The diff ships test coverage in `tests/compile/correctness_e2e/test_sequence_parallel.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #42197 - Fix mypy failure on main

- Link: https://github.com/vllm-project/vllm/pull/42197
- Status/date: merged / 2026-05-10
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-0, 8 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix mypy failure on main"; model line: Llama 3.1; category: bug fix; main diff: `tests/compile/correctness_e2e/test_sequence_parallel.py`; technical summary: Covers "Fix mypy failure on main"; the main implementation surface is `tests/compile/correctness_e2e/test_sequence_parallel.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/compile/correctness_e2e/test_sequence_parallel.py` modified +1/-0 (1 lines); hunks: -435,6 +435,7 @@ def test_tp_sp_nvfp4_generation(num_gpus_available: int):; symbols: test_tp_sp_nvfp4_generation, touching `test_tp_sp_nvfp4_generation`.
- Code diff details:
  - `tests/compile/correctness_e2e/test_sequence_parallel.py` modified +1/-0 (1 lines); hunks: -435,6 +435,7 @@ def test_tp_sp_nvfp4_generation(num_gpus_available: int):; symbols: test_tp_sp_nvfp4_generation
- Key code excerpts:

```diff
diff -- tests/compile/correctness_e2e/test_sequence_parallel.py
@@ -435,6 +435,7 @@ def test_tp_sp_nvfp4_generation(num_gpus_available: int):
+        enable_prompt_embeds=False,
```

- Reviewed files:
  - tests: `tests/compile/correctness_e2e/test_sequence_parallel.py` modified +1/-0
- Risk and verification: The diff ships test coverage in `tests/compile/correctness_e2e/test_sequence_parallel.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #42607 - Update Intel Xeon model list and vLLM Benchmark Suite BKMs

- Link: https://github.com/vllm-project/vllm/pull/42607
- Status/date: merged / 2026-05-15
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +118/-159, 465 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Update Intel Xeon model list and vLLM Benchmark Suite BKMs"; model line: Llama 3.1; category: performance/backend optimization; main diff: `docs/models/hardware_supported_models/cpu.md`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json`; technical summary: Covers "Update Intel Xeon model list and vLLM Benchmark Suite BKMs"; the main implementation surface is `docs/models/hardware_supported_models/cpu.md`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs/models/hardware_supported_models/cpu.md` modified +42/-16 (58 lines); hunks: -11,24 +11,50; `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json` modified +76/-143 (219 lines); hunks: -31,30 +31,9; -63,290 +42,244.
- Code diff details:
  - `docs/models/hardware_supported_models/cpu.md` modified +42/-16 (58 lines); hunks: -11,24 +11,50
  - `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json` modified +76/-143 (219 lines); hunks: -31,30 +31,9; -63,290 +42,244
- Key code excerpts:

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

- Reviewed files:
  - docs: `docs/models/hardware_supported_models/cpu.md` modified +42/-16
  - tests: `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json` modified +76/-143
- Risk and verification: The diff ships test coverage in `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #43262 - update GPU json file based on h200 recipes

- Link: https://github.com/vllm-project/vllm/pull/43262
- Status/date: merged / 2026-05-21
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +104/-69, 182 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "update GPU json file based on h200 recipes"; model line: Llama 3.1; category: performance/backend optimization; main diff: `.buildkite/performance-benchmarks/tests/serving-tests.json`; technical summary: Covers "update GPU json file based on h200 recipes"; the main implementation surface is `.buildkite/performance-benchmarks/tests/serving-tests.json`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `.buildkite/performance-benchmarks/tests/serving-tests.json` modified +104/-69 (173 lines); hunks: -1,77 +1,112.
- Code diff details:
  - `.buildkite/performance-benchmarks/tests/serving-tests.json` modified +104/-69 (173 lines); hunks: -1,77 +1,112
- Key code excerpts:

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

- Reviewed files:
  - tests: `.buildkite/performance-benchmarks/tests/serving-tests.json` modified +104/-69
- Risk and verification: The diff ships test coverage in `.buildkite/performance-benchmarks/tests/serving-tests.json`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #43233 - [Model Runner v2] Force v1 runner for tests

- Link: https://github.com/vllm-project/vllm/pull/43233
- Status/date: merged / 2026-05-23
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +50/-6, 136 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model Runner v2] Force v1 runner for tests"; model line: Llama 3.1; category: docs/tests/CI; main diff: `tests/models/quantization/test_bitsandbytes.py`, `tests/compile/correctness_e2e/test_async_tp.py`, `tests/utils.py`; technical summary: Covers "[Model Runner v2] Force v1 runner for tests"; the main implementation surface is `tests/models/quantization/test_bitsandbytes.py`, `tests/compile/correctness_e2e/test_async_tp.py`, `tests/utils.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/models/quantization/test_bitsandbytes.py` modified +5/-1 (6 lines); hunks: -137,7 +137,11 @@ def test_load_pp_4bit_bnb_model(model_name, description) ->...; symbols: test_load_pp_4bit_bnb_model, touching `test_load_pp_4bit_bnb_model`; `tests/compile/correctness_e2e/test_async_tp.py` modified +14/-2 (16 lines); hunks: -92,7 +92,13 @@ def test_async_tp_pass_correctness(; -154,4 +160,10 @@ def test_async_tp_pass_nvfp4_correctness(num_gpus_available...; symbols: test_async_tp_pass_correctness, test_async_tp_pass_nvfp4_correctness, touching `test_async_tp_pass_correctness, test_async_tp_pass_nvfp4_correctness`; `tests/utils.py` modified +14/-0 (14 lines); hunks: -1116,6 +1116,7 @@ def compare_two_settings(; -1129,6 +1130,9 @@ def compare_two_settings(; symbols: compare_two_settings, compare_all_settings, touching `compare_two_settings, compare_all_settings`; `tests/distributed/test_pipeline_parallel.py` modified +8/-1 (9 lines); hunks: -349,7 +349,14 @@ def _compare_tp(; symbols: _compare_tp, touching `_compare_tp`.
- Code diff details:
  - `tests/models/quantization/test_bitsandbytes.py` modified +5/-1 (6 lines); hunks: -137,7 +137,11 @@ def test_load_pp_4bit_bnb_model(model_name, description) ->...; symbols: test_load_pp_4bit_bnb_model
  - `tests/compile/correctness_e2e/test_async_tp.py` modified +14/-2 (16 lines); hunks: -92,7 +92,13 @@ def test_async_tp_pass_correctness(; -154,4 +160,10 @@ def test_async_tp_pass_nvfp4_correctness(num_gpus_available...; symbols: test_async_tp_pass_correctness, test_async_tp_pass_nvfp4_correctness
  - `tests/utils.py` modified +14/-0 (14 lines); hunks: -1116,6 +1116,7 @@ def compare_two_settings(; -1129,6 +1130,9 @@ def compare_two_settings(; symbols: compare_two_settings, compare_all_settings
  - `tests/distributed/test_pipeline_parallel.py` modified +8/-1 (9 lines); hunks: -349,7 +349,14 @@ def _compare_tp(; symbols: _compare_tp
  - `tests/compile/correctness_e2e/test_sequence_parallel.py` modified +7/-1 (8 lines); hunks: -294,7 +294,13 @@ def _compare_sp(; symbols: _compare_sp
- Key code excerpts:

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

- Reviewed files:
  - tests: `tests/models/quantization/test_bitsandbytes.py` modified +5/-1; `tests/compile/correctness_e2e/test_async_tp.py` modified +14/-2; `tests/utils.py` modified +14/-0; `tests/distributed/test_pipeline_parallel.py` modified +8/-1; `tests/compile/correctness_e2e/test_sequence_parallel.py` modified +7/-1; `tests/compile/fullgraph/test_basic_correctness.py` modified +2/-1
- Risk and verification: The diff ships test coverage in `tests/compile/correctness_e2e/test_async_tp.py`, `tests/compile/correctness_e2e/test_sequence_parallel.py`, `tests/compile/fullgraph/test_basic_correctness.py`, `tests/distributed/test_pipeline_parallel.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #44128 - [Misc] Remove dead VLLM_RPC_TIMEOUT env var and fix profiling doc that references it

- Link: https://github.com/vllm-project/vllm/pull/44128
- Status/date: merged / 2026-06-03
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 11 files, +1/-15, 100 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Misc] Remove dead VLLM_RPC_TIMEOUT env var and fix profiling doc that references it"; model line: Llama 3.1; category: bug fix; main diff: `vllm/envs.py`, `docs/contributing/profiling.md`, `.buildkite/performance-benchmarks/tests/latency-tests-arm64-cpu.json`; technical summary: Covers "[Misc] Remove dead VLLM_RPC_TIMEOUT env var and fix profiling doc that references it"; the main implementation surface is `vllm/envs.py`, `docs/contributing/profiling.md`, `.buildkite/performance-benchmarks/tests/latency-tests-arm64-cpu.json`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/envs.py` modified +0/-4 (4 lines); hunks: -95,7 +95,6; -1015,9 +1014,6 @@ def _resolve_rust_frontend_path() -> str | None:; symbols: _resolve_rust_frontend_path, touching `_resolve_rust_frontend_path`; `docs/contributing/profiling.md` modified +1/-2 (3 lines); hunks: -35,8 +35,7 @@ Traces can be visualized using .; `.buildkite/performance-benchmarks/tests/latency-tests-arm64-cpu.json` modified +0/-1 (1 lines); hunks: -2,7 +2,6; `.buildkite/performance-benchmarks/tests/latency-tests-cpu.json` modified +0/-1 (1 lines); hunks: -2,7 +2,6.
- Code diff details:
  - `vllm/envs.py` modified +0/-4 (4 lines); hunks: -95,7 +95,6; -1015,9 +1014,6 @@ def _resolve_rust_frontend_path() -> str | None:; symbols: _resolve_rust_frontend_path
  - `docs/contributing/profiling.md` modified +1/-2 (3 lines); hunks: -35,8 +35,7 @@ Traces can be visualized using .
  - `.buildkite/performance-benchmarks/tests/latency-tests-arm64-cpu.json` modified +0/-1 (1 lines); hunks: -2,7 +2,6
  - `.buildkite/performance-benchmarks/tests/latency-tests-cpu.json` modified +0/-1 (1 lines); hunks: -2,7 +2,6
  - `.buildkite/performance-benchmarks/tests/serving-tests-arm64-cpu.json` modified +0/-1 (1 lines); hunks: -13,7 +13,6
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/envs.py` modified +0/-4
  - docs: `docs/contributing/profiling.md` modified +1/-2
  - tests: `.buildkite/performance-benchmarks/tests/latency-tests-arm64-cpu.json` modified +0/-1; `.buildkite/performance-benchmarks/tests/latency-tests-cpu.json` modified +0/-1; `.buildkite/performance-benchmarks/tests/serving-tests-arm64-cpu.json` modified +0/-1; `.buildkite/performance-benchmarks/tests/serving-tests-cpu-asr.json` modified +0/-1; `.buildkite/performance-benchmarks/tests/serving-tests-cpu-embed.json` modified +0/-1; `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json` modified +0/-1
- Risk and verification: The diff ships test coverage in `.buildkite/performance-benchmarks/tests/latency-tests-arm64-cpu.json`, `.buildkite/performance-benchmarks/tests/latency-tests-cpu.json`, `.buildkite/performance-benchmarks/tests/serving-tests-arm64-cpu.json`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu-asr.json`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #44992 - Deprecations for v0.23 and v0.24

- Link: https://github.com/vllm-project/vllm/pull/44992
- Status/date: merged / 2026-06-11
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 35 files, +102/-676, 1334 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Deprecations for v0.23 and v0.24"; model line: Llama 3.1; category: performance/backend optimization; main diff: `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`, `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py`, `vllm/model_executor/kernels/linear/__init__.py`; technical summary: Covers "Deprecations for v0.23 and v0.24"; the main implementation surface is `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`, `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py`, `vllm/model_executor/kernels/linear/__init__.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` modified +0/-69 (69 lines); hunks: -6,7 +6,6; -465,74 +464,6 @@ def select_mxfp4_moe_backend(; symbols: select_mxfp4_moe_backend, touching `select_mxfp4_moe_backend`; `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py` modified +0/-59 (59 lines); hunks: -22,10 +22,6; -58,12 +54,6 @@ class NvFp4MoeBackend(Enum):; symbols: NvFp4MoeBackend, is_global_sf_supported_for_nvfp4_backend, _return_or_raise, touching `NvFp4MoeBackend, is_global_sf_supported_for_nvfp4_backend, _return_or_raise`; `vllm/model_executor/kernels/linear/__init__.py` modified +8/-47 (55 lines); hunks: -212,6 +212,9 @@ def _get_linear_backend() -> str:; -392,7 +395,7 @@ def _filter_kernels_by_backend(; symbols: _get_linear_backend, _filter_kernels_by_backend, init_mxfp4_linear_kernel, init_wfp8_a16_linear_kernel, touching `_get_linear_backend, _filter_kernels_by_backend, init_mxfp4_linear_kernel`; `vllm/model_executor/layers/fused_moe/oracle/fp8.py` modified +0/-50 (50 lines); hunks: -20,8 +20,6; -321,54 +319,6 @@ def _return_or_raise(; symbols: _return_or_raise, touching `_return_or_raise`.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` modified +0/-69 (69 lines); hunks: -6,7 +6,6; -465,74 +464,6 @@ def select_mxfp4_moe_backend(; symbols: select_mxfp4_moe_backend
  - `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py` modified +0/-59 (59 lines); hunks: -22,10 +22,6; -58,12 +54,6 @@ class NvFp4MoeBackend(Enum):; symbols: NvFp4MoeBackend, is_global_sf_supported_for_nvfp4_backend, _return_or_raise
  - `vllm/model_executor/kernels/linear/__init__.py` modified +8/-47 (55 lines); hunks: -212,6 +212,9 @@ def _get_linear_backend() -> str:; -392,7 +395,7 @@ def _filter_kernels_by_backend(; symbols: _get_linear_backend, _filter_kernels_by_backend, init_mxfp4_linear_kernel, init_wfp8_a16_linear_kernel
  - `vllm/model_executor/layers/fused_moe/oracle/fp8.py` modified +0/-50 (50 lines); hunks: -20,8 +20,6; -321,54 +319,6 @@ def _return_or_raise(; symbols: _return_or_raise
  - `vllm/model_executor/layers/fused_moe/oracle/unquantized.py` modified +0/-45 (45 lines); hunks: -19,9 +19,7; -230,49 +228,6 @@ def _return_or_raise(; symbols: _return_or_raise
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` modified +0/-69; `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py` modified +0/-59; `vllm/model_executor/kernels/linear/__init__.py` modified +8/-47; `vllm/model_executor/layers/fused_moe/oracle/fp8.py` modified +0/-50; `vllm/model_executor/layers/fused_moe/oracle/unquantized.py` modified +0/-45; `vllm/entrypoints/pooling/offline.py` modified +0/-44
- Risk and verification: The diff ships test coverage in `tests/compile/correctness_e2e/test_async_tp.py`, `tests/conftest.py`, `tests/distributed/test_eplb_fused_moe_layer_dep_nvfp4.py`, `tests/entrypoints/pooling/reward/test_token_reward_offline.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
