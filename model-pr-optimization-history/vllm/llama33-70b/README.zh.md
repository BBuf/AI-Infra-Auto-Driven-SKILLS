# vllm Llama 3.3 70B 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| - | 当前主线没有匹配到实现文件 |

## PR 覆盖总览

- git 追溯 PR 数: 0
- 原文档显式引用补充 PR 数: 10
- 当前文档总 PR 数: 10
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-11-19 | [#28697](https://github.com/vllm-project/vllm/pull/28697) | merged | Add CPU support model | `docs/models/hardware_supported_models/cpu.md` |
| 2025-11-27 | [#29380](https://github.com/vllm-project/vllm/pull/29380) | merged | add xpu supported model and model id for cpu | `docs/models/hardware_supported_models/xpu.md`, `docs/models/hardware_supported_models/cpu.md` |
| 2026-01-24 | [#32963](https://github.com/vllm-project/vllm/pull/32963) | merged | Update CPU doc according to feedback | `docs/models/hardware_supported_models/cpu.md`, `docs/models/hardware_supported_models/xpu.md`, `docs/benchmarking/dashboard.md` |
| 2026-02-12 | [#34128](https://github.com/vllm-project/vllm/pull/34128) | merged | Vllm CPU benchmark suite improvement | `.buildkite/performance-benchmarks/scripts/compare-json-results.py`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json`, `.buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh` |
| 2026-03-09 | [#36398](https://github.com/vllm-project/vllm/pull/36398) | merged | Allow `markdownlint` to run locally | `docs/models/hardware_supported_models/xpu.md`, `docs/models/supported_models.md`, `docs/models/hardware_supported_models/cpu.md` |
| 2026-03-12 | [#35086](https://github.com/vllm-project/vllm/pull/35086) | merged | more models for vLLM Benchmark Suite | `.buildkite/performance-benchmarks/scripts/compare-json-results.py`, `.buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json` |
| 2026-03-31 | [#38576](https://github.com/vllm-project/vllm/pull/38576) | merged | vLLM Benchmark Suite perf regression after PR#32723 | `.buildkite/performance-benchmarks/tests/serving-tests-hpu.json`, `.buildkite/performance-benchmarks/tests/serving-tests.json`, `.buildkite/performance-benchmarks/tests/serving-tests-arm64-cpu.json` |
| 2026-05-15 | [#42607](https://github.com/vllm-project/vllm/pull/42607) | merged | Update Intel Xeon model list and vLLM Benchmark Suite BKMs | `docs/models/hardware_supported_models/cpu.md`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json` |
| 2026-06-03 | [#44128](https://github.com/vllm-project/vllm/pull/44128) | merged | [Misc] Remove dead VLLM_RPC_TIMEOUT env var and fix profiling doc that references it | `vllm/envs.py`, `docs/contributing/profiling.md`, `.buildkite/performance-benchmarks/tests/latency-tests-arm64-cpu.json` |
| 2026-06-16 | [#42726](https://github.com/vllm-project/vllm/pull/42726) | merged | [ZenCPU] Add zencpu Platform Runtime Logging and Docs | `tests/model_executor/test_cpu_unquantized_gemm_dispatch.py`, `vllm/model_executor/layers/utils.py`, `docs/models/hardware_supported_models/cpu.md` |

## 逐 PR diff 审计卡

### PR #28697 - Add CPU support model

- 链接: https://github.com/vllm-project/vllm/pull/28697
- 状态/时间: merged / 2025-11-19
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+26/-0，可读 patch 27 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add CPU support model」；模型线: Llama 3.3 70B；类别: 文档/测试/CI；主要 diff: `docs/models/hardware_supported_models/cpu.md`；技术摘要: 覆盖「Add CPU support model」；主要实现面是 `docs/models/hardware_supported_models/cpu.md`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs/models/hardware_supported_models/cpu.md` added +26/-0 (26 lines); hunks: -0,0 +1,26。
- 代码 diff 细节:
  - `docs/models/hardware_supported_models/cpu.md` added +26/-0 (26 lines); hunks: -0,0 +1,26
- 关键代码摘录:

```diff
diff -- docs/models/hardware_supported_models/cpu.md
@@ -0,0 +1,26 @@
+# CPU - Intel® Xeon®
+## Supported Models
+### Text-only Language Models
+| Model                                | Architecture                             | Supported |
+|--------------------------------------|-------------------------------------------|-----------|
+| meta-llama/Llama-3.1 / 3.3           | LlamaForCausalLM                          | ✅        |
```

- 已读文件:
  - docs: `docs/models/hardware_supported_models/cpu.md` added +26/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs/models/hardware_supported_models/cpu.md`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #29380 - add xpu supported model and model id for cpu

- 链接: https://github.com/vllm-project/vllm/pull/29380
- 状态/时间: merged / 2025-11-27
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+82/-9，可读 patch 109 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「add xpu supported model and model id for cpu」；模型线: Llama 3.3 70B；类别: 文档/测试/CI；主要 diff: `docs/models/hardware_supported_models/xpu.md`, `docs/models/hardware_supported_models/cpu.md`；技术摘要: 覆盖「add xpu supported model and model id for cpu」；主要实现面是 `docs/models/hardware_supported_models/xpu.md`, `docs/models/hardware_supported_models/cpu.md`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs/models/hardware_supported_models/xpu.md` added +65/-0 (65 lines); hunks: -0,0 +1,65；`docs/models/hardware_supported_models/cpu.md` modified +17/-9 (26 lines); hunks: -1,25 +1,33。
- 代码 diff 细节:
  - `docs/models/hardware_supported_models/xpu.md` added +65/-0 (65 lines); hunks: -0,0 +1,65
  - `docs/models/hardware_supported_models/cpu.md` modified +17/-9 (26 lines); hunks: -1,25 +1,33
- 关键代码摘录:

```diff
diff -- docs/models/hardware_supported_models/xpu.md
@@ -0,0 +1,65 @@
+# XPU - Intel® GPUs
+## Validated Hardware
+| Hardware                                 |
+| ----------------------------------------- |
+| [Intel® Arc™ Pro B-Series Graphics](https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/arc/workstations/b-series/overview.html)                   |
+## Supported Models
diff -- docs/models/hardware_supported_models/cpu.md
@@ -1,25 +1,33 @@
+## Validated Hardware
+| Hardware                                 |
+| ----------------------------------------- |
+| [Intel® Xeon® 6 Processors](https://www.intel.com/content/www/us/en/products/details/processors/xeon.html)                   |
+| [Intel® Xeon® 5 Processors](https://www.intel.com/content/www/us/en/products/docs/processors/xeon/5th-gen-xeon-scalable-processors.html)              |
-| meta-llama/Llama-3.1 / 3.3           | LlamaForCausalLM                          | ✅        |
```

- 已读文件:
  - docs: `docs/models/hardware_supported_models/xpu.md` added +65/-0; `docs/models/hardware_supported_models/cpu.md` modified +17/-9
- 验证与风险: 该 PR 主要落在文档/示例 `docs/models/hardware_supported_models/cpu.md`, `docs/models/hardware_supported_models/xpu.md`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #32963 - Update CPU doc according to feedback

- 链接: https://github.com/vllm-project/vllm/pull/32963
- 状态/时间: merged / 2026-01-24
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+4/-4，可读 patch 35 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Update CPU doc according to feedback」；模型线: Llama 3.3 70B；类别: 文档/测试/CI；主要 diff: `docs/models/hardware_supported_models/cpu.md`, `docs/models/hardware_supported_models/xpu.md`, `docs/benchmarking/dashboard.md`；技术摘要: 覆盖「Update CPU doc according to feedback」；主要实现面是 `docs/models/hardware_supported_models/cpu.md`, `docs/models/hardware_supported_models/xpu.md`, `docs/benchmarking/dashboard.md`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs/models/hardware_supported_models/cpu.md` modified +1/-1 (2 lines); hunks: -7,7 +7,7；`docs/models/hardware_supported_models/xpu.md` modified +1/-1 (2 lines); hunks: -6,7 +6,7；`docs/benchmarking/dashboard.md` modified +2/-2 (4 lines); hunks: -13,14 +13,14 @@ For x86 CPU environment, please use the image with "-cpu" po...。
- 代码 diff 细节:
  - `docs/models/hardware_supported_models/cpu.md` modified +1/-1 (2 lines); hunks: -7,7 +7,7
  - `docs/models/hardware_supported_models/xpu.md` modified +1/-1 (2 lines); hunks: -6,7 +6,7
  - `docs/benchmarking/dashboard.md` modified +2/-2 (4 lines); hunks: -13,14 +13,14 @@ For x86 CPU environment, please use the image with "-cpu" po...
- 关键代码摘录:

```diff
diff -- docs/models/hardware_supported_models/cpu.md
@@ -7,7 +7,7 @@
-## Supported Models
+## Recommended Models
diff -- docs/models/hardware_supported_models/xpu.md
@@ -6,7 +6,7 @@
-## Supported Models
+## Recommended Models
diff -- docs/benchmarking/dashboard.md
@@ -13,14 +13,14 @@ For x86 CPU environment, please use the image with "-cpu" postfix. For AArch64 C
-export VLLM_COMMIT=1da94e673c257373280026f75ceb4effac80e892 # use full commit hash from the main branch
+export VLLM_COMMIT=7f42dc20bb2800d09faa72b26f25d54e26f1b694 # use full commit hash from the main branch
-docker run -it --entrypoint /bin/bash -v /data/huggingface:/root/.cache/huggingface -e HF_TOKEN=$HF_TOKEN -e ON_ARM64_CPU=1 --shm-size=16g --name vllm-cpu-ci public.ecr.aws/q9t5s3
+docker run -it --entrypoint /bin/bash -v /data/huggingface:/root/.cache/huggingface -e HF_TOKEN=$HF_TOKEN -e ON_CPU=1 --shm-size=16g --name vllm-cpu-ci public.ecr.aws/q9t5s3a7/vll
```

- 已读文件:
  - docs: `docs/models/hardware_supported_models/cpu.md` modified +1/-1; `docs/models/hardware_supported_models/xpu.md` modified +1/-1; `docs/benchmarking/dashboard.md` modified +2/-2
- 验证与风险: 该 PR 主要落在文档/示例 `docs/benchmarking/dashboard.md`, `docs/models/hardware_supported_models/cpu.md`, `docs/models/hardware_supported_models/xpu.md`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #34128 - Vllm CPU benchmark suite improvement

- 链接: https://github.com/vllm-project/vllm/pull/34128
- 状态/时间: merged / 2026-02-12
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+802/-254，可读 patch 1243 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Vllm CPU benchmark suite improvement」；模型线: Llama 3.3 70B；类别: 性能/后端优化；主要 diff: `.buildkite/performance-benchmarks/scripts/compare-json-results.py`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json`, `.buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh`；技术摘要: 覆盖「Vllm CPU benchmark suite improvement」；主要实现面是 `.buildkite/performance-benchmarks/scripts/compare-json-results.py`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json`, `.buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #36398 - Allow `markdownlint` to run locally

- 链接: https://github.com/vllm-project/vllm/pull/36398
- 状态/时间: merged / 2026-03-09
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 47 个文件，+394/-392，可读 patch 1933 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Allow `markdownlint` to run locally」；模型线: Llama 3.3 70B；类别: 文档/测试/CI；主要 diff: `docs/models/hardware_supported_models/xpu.md`, `docs/models/supported_models.md`, `docs/models/hardware_supported_models/cpu.md`；技术摘要: 覆盖「Allow `markdownlint` to run locally」；主要实现面是 `docs/models/hardware_supported_models/xpu.md`, `docs/models/supported_models.md`, `docs/models/hardware_supported_models/cpu.md`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docs/models/hardware_supported_models/xpu.md` modified +40/-40 (80 lines); hunks: -2,63 +2,63；`docs/models/supported_models.md` modified +31/-31 (62 lines); hunks: -179,7 +179,7 @@ class MyConfig(PretrainedConfig):; -363,7 +363,7 @@ th {; symbols: MyConfig，涉及 `MyConfig`；`docs/models/hardware_supported_models/cpu.md` modified +16/-16 (32 lines); hunks: -2,32 +2,32；`docs/models/pooling_models.md` modified +7/-7 (14 lines); hunks: -31,7 +31,7 @@ vLLM will attempt to automatically convert the model according...; -46,7 +46,7 @@ Each pooling model in vLLM supports one or more of these tasks...。
- 代码 diff 细节:
  - `docs/models/hardware_supported_models/xpu.md` modified +40/-40 (80 lines); hunks: -2,63 +2,63
  - `docs/models/supported_models.md` modified +31/-31 (62 lines); hunks: -179,7 +179,7 @@ class MyConfig(PretrainedConfig):; -363,7 +363,7 @@ th {; symbols: MyConfig
  - `docs/models/hardware_supported_models/cpu.md` modified +16/-16 (32 lines); hunks: -2,32 +2,32
  - `docs/models/pooling_models.md` modified +7/-7 (14 lines); hunks: -31,7 +31,7 @@ vLLM will attempt to automatically convert the model according...; -46,7 +46,7 @@ Each pooling model in vLLM supports one or more of these tasks...
  - `docs/features/reasoning_outputs.md` modified +1/-1 (2 lines); hunks: -12,7 +12,7 @@ Reasoning models return an additional `reasoning` field in the...
- 关键代码摘录:

```diff
diff -- docs/models/hardware_supported_models/xpu.md
@@ -2,63 +2,63 @@
-| Hardware                                 |
-| ----------------------------------------- |
-| [Intel® Arc™ Pro B-Series Graphics](https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/arc/workstations/b-series/overview.html)                   |
+| Hardware |
+| -------- |
+| [Intel® Arc™ Pro B-Series Graphics](https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/arc/workstations/b-series/overview.html) |
diff -- docs/models/supported_models.md
@@ -179,7 +179,7 @@ class MyConfig(PretrainedConfig):
-|--------------|--------|-------------------|
+| ------------ | ------ | ----------------- |
@@ -363,7 +363,7 @@ th {
-|--------------|--------|-------------------|----------------------|---------------------------|
+| ------------ | ------ | ----------------- | -------------------- | ------------------------- |
@@ -387,7 +387,7 @@ th {
diff -- docs/models/hardware_supported_models/cpu.md
@@ -2,32 +2,32 @@
```

- 已读文件:
  - docs: `docs/models/hardware_supported_models/xpu.md` modified +40/-40; `docs/models/supported_models.md` modified +31/-31; `docs/models/hardware_supported_models/cpu.md` modified +16/-16; `docs/models/pooling_models.md` modified +7/-7; `docs/features/reasoning_outputs.md` modified +1/-1; `docs/getting_started/installation/cpu.arm.inc.md` modified +27/-25
- 验证与风险: runtime 路径改动集中在 `vllm/lora/ops/triton_ops/README_TUNING.md`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #35086 - more models for vLLM Benchmark Suite

- 链接: https://github.com/vllm-project/vllm/pull/35086
- 状态/时间: merged / 2026-03-12
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+800/-119，可读 patch 1301 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「more models for vLLM Benchmark Suite」；模型线: Llama 3.3 70B；类别: 性能/后端优化；主要 diff: `.buildkite/performance-benchmarks/scripts/compare-json-results.py`, `.buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json`；技术摘要: 覆盖「more models for vLLM Benchmark Suite」；主要实现面是 `.buildkite/performance-benchmarks/scripts/compare-json-results.py`, `.buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「vLLM Benchmark Suite perf regression after PR#32723」；模型线: Llama 3.3 70B；类别: 缺陷修复；主要 diff: `.buildkite/performance-benchmarks/tests/serving-tests-hpu.json`, `.buildkite/performance-benchmarks/tests/serving-tests.json`, `.buildkite/performance-benchmarks/tests/serving-tests-arm64-cpu.json`；技术摘要: 覆盖「vLLM Benchmark Suite perf regression after PR#32723」；主要实现面是 `.buildkite/performance-benchmarks/tests/serving-tests-hpu.json`, `.buildkite/performance-benchmarks/tests/serving-tests.json`, `.buildkite/performance-benchmarks/tests/serving-tests-arm64-cpu.json`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #42607 - Update Intel Xeon model list and vLLM Benchmark Suite BKMs

- 链接: https://github.com/vllm-project/vllm/pull/42607
- 状态/时间: merged / 2026-05-15
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+118/-159，可读 patch 465 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Update Intel Xeon model list and vLLM Benchmark Suite BKMs」；模型线: Llama 3.3 70B；类别: 性能/后端优化；主要 diff: `docs/models/hardware_supported_models/cpu.md`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json`；技术摘要: 覆盖「Update Intel Xeon model list and vLLM Benchmark Suite BKMs」；主要实现面是 `docs/models/hardware_supported_models/cpu.md`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #44128 - [Misc] Remove dead VLLM_RPC_TIMEOUT env var and fix profiling doc that references it

- 链接: https://github.com/vllm-project/vllm/pull/44128
- 状态/时间: merged / 2026-06-03
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 11 个文件，+1/-15，可读 patch 100 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Misc] Remove dead VLLM_RPC_TIMEOUT env var and fix profiling doc that references it」；模型线: Llama 3.3 70B；类别: 缺陷修复；主要 diff: `vllm/envs.py`, `docs/contributing/profiling.md`, `.buildkite/performance-benchmarks/tests/latency-tests-arm64-cpu.json`；技术摘要: 覆盖「[Misc] Remove dead VLLM_RPC_TIMEOUT env var and fix profiling doc that references it」；主要实现面是 `vllm/envs.py`, `docs/contributing/profiling.md`, `.buildkite/performance-benchmarks/tests/latency-tests-arm64-cpu.json`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #42726 - [ZenCPU] Add zencpu Platform Runtime Logging and Docs

- 链接: https://github.com/vllm-project/vllm/pull/42726
- 状态/时间: merged / 2026-06-16
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+108/-3，可读 patch 200 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[ZenCPU] Add zencpu Platform Runtime Logging and Docs」；模型线: Llama 3.3 70B；类别: 文档/测试/CI；主要 diff: `tests/model_executor/test_cpu_unquantized_gemm_dispatch.py`, `vllm/model_executor/layers/utils.py`, `docs/models/hardware_supported_models/cpu.md`；技术摘要: 覆盖「[ZenCPU] Add zencpu Platform Runtime Logging and Docs」；主要实现面是 `tests/model_executor/test_cpu_unquantized_gemm_dispatch.py`, `vllm/model_executor/layers/utils.py`, `docs/models/hardware_supported_models/cpu.md`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/model_executor/test_cpu_unquantized_gemm_dispatch.py` modified +23/-0 (23 lines); hunks: -66,3 +66,26 @@ def test_dispatch_cpu_unquantized_gemm_zen_remove_weight(monk...; symbols: test_dispatch_cpu_unquantized_gemm_zen_remove_weight, test_dispatch_cpu_unquantized_gemm_logs_zentorch_dispatch，涉及 `test_dispatch_cpu_unquantized_gemm_zen_remove_weight, test_dispatch_cpu_unquantized_gemm_logs_zentorch_dispatch`；`vllm/model_executor/layers/utils.py` modified +11/-0 (11 lines); hunks: -272,6 +272,10 @@ def dispatch_cpu_unquantized_gemm(; -285,6 +289,9 @@ def dispatch_cpu_unquantized_gemm(; symbols: dispatch_cpu_unquantized_gemm, cpu_unquantized_gemm，涉及 `dispatch_cpu_unquantized_gemm, cpu_unquantized_gemm`；`docs/models/hardware_supported_models/cpu.md` modified +3/-0 (3 lines); hunks: -1,5 +1,8；`docs/getting_started/installation/cpu.x86.inc.md` modified +45/-2 (47 lines); hunks: -1,4 +1,4; -200,7 +200,19 @@ docker build -f docker/Dockerfile.cpu \。
- 代码 diff 细节:
  - `tests/model_executor/test_cpu_unquantized_gemm_dispatch.py` modified +23/-0 (23 lines); hunks: -66,3 +66,26 @@ def test_dispatch_cpu_unquantized_gemm_zen_remove_weight(monk...; symbols: test_dispatch_cpu_unquantized_gemm_zen_remove_weight, test_dispatch_cpu_unquantized_gemm_logs_zentorch_dispatch
  - `vllm/model_executor/layers/utils.py` modified +11/-0 (11 lines); hunks: -272,6 +272,10 @@ def dispatch_cpu_unquantized_gemm(; -285,6 +289,9 @@ def dispatch_cpu_unquantized_gemm(; symbols: dispatch_cpu_unquantized_gemm, cpu_unquantized_gemm
  - `docs/models/hardware_supported_models/cpu.md` modified +3/-0 (3 lines); hunks: -1,5 +1,8
  - `docs/getting_started/installation/cpu.x86.inc.md` modified +45/-2 (47 lines); hunks: -1,4 +1,4; -200,7 +200,19 @@ docker build -f docker/Dockerfile.cpu \
  - `docs/getting_started/installation/cpu.md` modified +25/-0 (25 lines); hunks: -142,19 +142,25 @@ VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_VARIANT=cpu...; -227,6 +233,25 @@ By providing MODEL_FILTER and DTYPE_FILTER, only commands f...
- 关键代码摘录:

```diff
diff -- tests/model_executor/test_cpu_unquantized_gemm_dispatch.py
@@ -66,3 +66,26 @@ def test_dispatch_cpu_unquantized_gemm_zen_remove_weight(monkeypatch):
+@pytest.mark.usefixtures("_mock_zentorch_linear_unary")
+def test_dispatch_cpu_unquantized_gemm_logs_zentorch_dispatch(monkeypatch):
+    monkeypatch.setattr(current_platform, "is_zen_cpu", lambda: True)
+    expected_prepacked = bool(utils.envs.VLLM_ZENTORCH_WEIGHT_PREPACK) and hasattr(
+        torch.ops.zentorch, "zentorch_weight_prepack_for_linear"
+    )
diff -- vllm/model_executor/layers/utils.py
@@ -272,6 +272,10 @@ def dispatch_cpu_unquantized_gemm(
+        logger.debug_once(
+            "CPU unquantized GEMM dispatch: using zentorch_linear_unary (prepacked=%s)",
+            is_prepacked,
+        )
@@ -285,6 +289,9 @@ def dispatch_cpu_unquantized_gemm(
+        logger.debug_once(
diff -- docs/models/hardware_supported_models/cpu.md
@@ -1,5 +1,8 @@
```

- 已读文件:
  - tests: `tests/model_executor/test_cpu_unquantized_gemm_dispatch.py` modified +23/-0
  - runtime: `vllm/model_executor/layers/utils.py` modified +11/-0; `vllm/platforms/__init__.py` modified +1/-1
  - docs: `docs/models/hardware_supported_models/cpu.md` modified +3/-0; `docs/getting_started/installation/cpu.x86.inc.md` modified +45/-2; `docs/getting_started/installation/cpu.md` modified +25/-0
- 验证与风险: diff 自带测试面 `tests/model_executor/test_cpu_unquantized_gemm_dispatch.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
