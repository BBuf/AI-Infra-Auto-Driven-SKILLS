# vllm Llama 3.3 70B 模型 PR 优化历史

## 2026-06-05 PR 补漏复核

已于 2026-06-05 按 vllm 上游 `origin/main@c66b19800` 复核；自上次时效基准（2026-05-19）以来，共有 1 个带 PR 编号的合并改动到所跟踪的实现文件，这些 PR 尚未并入下方时间线 / 逐 PR diff 审计卡，应在下次完整重生成时补齐。

| 合并日期 | PR | 标题 | 改动到的跟踪文件 |
| --- | --- | --- | --- |
| 2026-06-03 | [#44128](https://github.com/vllm-project/vllm/pull/44128) | [Misc] Remove dead VLLM_RPC_TIMEOUT env var and fix profiling doc that references it | `serving-tests-cpu-text.json` |


## 2026-05-19 新增覆盖

按 vllm 上游 `origin/main@ef54a4d604`、模型相关文件的 `git log --name-only -- <model-files>` 以及 GitHub Pull Request files API 生成。本页用于补齐 sgl-cookbook 中 `Llama 3.3 70B` 缺失的历史 PR 优化文档。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json` | [#42607](https://github.com/vllm-project/vllm/pull/42607), [#38576](https://github.com/vllm-project/vllm/pull/38576), [#35086](https://github.com/vllm-project/vllm/pull/35086), [#34128](https://github.com/vllm-project/vllm/pull/34128) |
| `docs/models/hardware_supported_models/cpu.md` | [#42607](https://github.com/vllm-project/vllm/pull/42607), [#36398](https://github.com/vllm-project/vllm/pull/36398), [#32963](https://github.com/vllm-project/vllm/pull/32963), [#29380](https://github.com/vllm-project/vllm/pull/29380), [#28697](https://github.com/vllm-project/vllm/pull/28697) |

## PR 覆盖总览

- git 追溯 PR 数: 8
- 关键词/补充 PR 数: 0
- 当前文档总 PR 数: 8
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-11-19 | [#28697](https://github.com/vllm-project/vllm/pull/28697) | merged | Add CPU support model | `docs/models/hardware_supported_models/cpu.md` |
| 2025-11-27 | [#29380](https://github.com/vllm-project/vllm/pull/29380) | merged | add xpu supported model and model id for cpu | `docs/models/hardware_supported_models/cpu.md`, `docs/models/hardware_supported_models/xpu.md` |
| 2026-01-24 | [#32963](https://github.com/vllm-project/vllm/pull/32963) | merged | Update CPU doc according to feedback | `docs/benchmarking/dashboard.md`, `docs/models/hardware_supported_models/cpu.md`, `docs/models/hardware_supported_models/xpu.md` |
| 2026-02-12 | [#34128](https://github.com/vllm-project/vllm/pull/34128) | merged | Vllm CPU benchmark suite improvement | `.buildkite/performance-benchmarks/scripts/compare-json-results.py`, `.buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu-embed.json` |
| 2026-03-09 | [#36398](https://github.com/vllm-project/vllm/pull/36398) | merged | Allow `markdownlint` to run locally | `.github/mergify.yml`, `.pre-commit-config.yaml`, `benchmarks/attention_benchmarks/README.md` |
| 2026-03-12 | [#35086](https://github.com/vllm-project/vllm/pull/35086) | merged | more models for vLLM Benchmark Suite | `.buildkite/performance-benchmarks/scripts/compare-json-results.py`, `.buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu-asr.json` |
| 2026-03-31 | [#38576](https://github.com/vllm-project/vllm/pull/38576) | merged | vLLM Benchmark Suite perf regression after PR#32723 | `.buildkite/performance-benchmarks/tests/serving-tests-arm64-cpu.json`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu-asr.json`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json` |
| 2026-05-15 | [#42607](https://github.com/vllm-project/vllm/pull/42607) | merged | Update Intel Xeon model list and vLLM Benchmark Suite BKMs | `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json`, `docs/models/hardware_supported_models/cpu.md` |

## 逐 PR diff 审计卡

### PR #28697 - Add CPU support model

- 链接: https://github.com/vllm-project/vllm/pull/28697
- 状态/时间: merged / 2025-11-19
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+26/-0，可读 patch 27 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add CPU support model」；模型线: Llama 3.3 70B；类别: 模型支持/运行时入口；主要 diff: `docs/models/hardware_supported_models/cpu.md`。
- 实现要点:
  - `docs/models/hardware_supported_models/cpu.md` added +26/-0
- 代码 diff 细节:
  - `docs/models/hardware_supported_models/cpu.md` added +26/-0
- 关键代码摘录:

```diff
diff -- docs/models/hardware_supported_models/cpu.md
@@ -0,0 +1,26 @@
+# CPU - Intel® Xeon®
+
+## Supported Models
+
+### Text-only Language Models
+
+| Model                                | Architecture                             | Supported |
+|--------------------------------------|-------------------------------------------|-----------|
+| meta-llama/Llama-3.1 / 3.3           | LlamaForCausalLM                          | ✅        |
+| meta-llama/Llama-4-Scout             | Llama4ForConditionalGeneration            | ✅        |
+| meta-llama/Llama-4-Maverick          | Llama4ForConditionalGeneration            | ✅        |
+| ibm-granite/granite (Granite-MOE)    | GraniteMoeForCausalLM                     | ✅        |
+| Qwen/Qwen3                           | Qwen3ForCausalLM                          | ✅        |
```
- 已读文件:
  - docs/bench: `docs/models/hardware_supported_models/cpu.md` added +26/-0
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #29380 - add xpu supported model and model id for cpu

- 链接: https://github.com/vllm-project/vllm/pull/29380
- 状态/时间: merged / 2025-11-27
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+82/-9，可读 patch 109 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「add xpu supported model and model id for cpu」；模型线: Llama 3.3 70B；类别: 模型支持/运行时入口；主要 diff: `docs/models/hardware_supported_models/cpu.md`, `docs/models/hardware_supported_models/xpu.md`。
- 实现要点:
  - `docs/models/hardware_supported_models/cpu.md` modified +17/-9
  - `docs/models/hardware_supported_models/xpu.md` added +65/-0
- 代码 diff 细节:
  - `docs/models/hardware_supported_models/cpu.md` modified +17/-9
  - `docs/models/hardware_supported_models/xpu.md` added +65/-0
- 关键代码摘录:

```diff
diff -- docs/models/hardware_supported_models/cpu.md
@@ -1,25 +1,33 @@
 # CPU - Intel® Xeon®

+## Validated Hardware
+
+| Hardware                                 |
+| ----------------------------------------- |
+| [Intel® Xeon® 6 Processors](https://www.intel.com/content/www/us/en/products/details/processors/xeon.html)                   |
+| [Intel® Xeon® 5 Processors](https://www.intel.com/content/www/us/en/products/docs/processors/xeon/5th-gen-xeon-scalable-processors.html)              |
+
 ## Supported Models

 ### Text-only Language Models

diff -- docs/models/hardware_supported_models/xpu.md
@@ -0,0 +1,65 @@
+# XPU - Intel® GPUs
+
+## Validated Hardware
+
+| Hardware                                 |
+| ----------------------------------------- |
+| [Intel® Arc™ Pro B-Series Graphics](https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/arc/workstations/b-series/overview.html)                   |
+
+## Supported Models
+
+### Text-only Language Models
+
+| Model                                     | Architecture                                         | FP16 | Dynamic FP8 | MXFP4 |
```
- 已读文件:
  - docs/bench: `docs/models/hardware_supported_models/cpu.md` modified +17/-9; `docs/models/hardware_supported_models/xpu.md` added +65/-0
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #32963 - Update CPU doc according to feedback

- 链接: https://github.com/vllm-project/vllm/pull/32963
- 状态/时间: merged / 2026-01-24
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+4/-4，可读 patch 35 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Update CPU doc according to feedback」；模型线: Llama 3.3 70B；类别: 文档/测试/CI；主要 diff: `docs/benchmarking/dashboard.md`, `docs/models/hardware_supported_models/cpu.md`, `docs/models/hardware_supported_models/xpu.md`。
- 实现要点:
  - `docs/benchmarking/dashboard.md` modified +2/-2
  - `docs/models/hardware_supported_models/cpu.md` modified +1/-1
  - `docs/models/hardware_supported_models/xpu.md` modified +1/-1
- 代码 diff 细节:
  - `docs/benchmarking/dashboard.md` modified +2/-2
  - `docs/models/hardware_supported_models/cpu.md` modified +1/-1
  - `docs/models/hardware_supported_models/xpu.md` modified +1/-1
- 关键代码摘录:

```diff
diff -- docs/benchmarking/dashboard.md
@@ -13,14 +13,14 @@ For x86 CPU environment, please use the image with "-cpu" postfix. For AArch64 C
 Here is an example for docker run command for CPU. For GPUs skip setting the `ON_CPU` env var.

 ```bash
-export VLLM_COMMIT=1da94e673c257373280026f75ceb4effac80e892 # use full commit hash from the main branch
+export VLLM_COMMIT=7f42dc20bb2800d09faa72b26f25d54e26f1b694 # use full commit hash from the main branch
 export HF_TOKEN=<valid Hugging Face token>
 if [[ "$(uname -m)" == aarch64 || "$(uname -m)" == arm64 ]]; then
   IMG_SUFFIX="arm64-cpu"
 else
   IMG_SUFFIX="cpu"
 fi
-docker run -it --entrypoint /bin/bash -v /data/huggingface:/root/.cache/huggingface -e HF_TOKEN=$HF_TOKEN -e ON_ARM64_CPU=1 --shm-size=16g --name vllm-cpu-ci public.ecr.aws/q9t5s3a7/vllm-ci-test-repo:${VLLM_COMMIT}-${IMG_SUFFIX}
+docker run -it --entrypoint /bin/bash -v /data/huggingface:/root/.cache/huggingface -e HF_TOKEN=$HF_TOKEN -e ON_CPU=1 --shm-size=16g --name vllm-cpu-ci public.ecr.aws/q9t5s3a7/vllm-ci-test-repo:${VLLM_COMMIT}-${IMG_SUFFIX}
diff -- docs/models/hardware_supported_models/cpu.md
@@ -7,7 +7,7 @@
 | [Intel® Xeon® 6 Processors](https://www.intel.com/content/www/us/en/products/details/processors/xeon.html)                   |
 | [Intel® Xeon® 5 Processors](https://www.intel.com/content/www/us/en/products/docs/processors/xeon/5th-gen-xeon-scalable-processors.html)              |

-## Supported Models
+## Recommended Models

 ### Text-only Language Models

```
- 已读文件:
  - docs/bench: `docs/benchmarking/dashboard.md` modified +2/-2; `docs/models/hardware_supported_models/cpu.md` modified +1/-1; `docs/models/hardware_supported_models/xpu.md` modified +1/-1
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #34128 - Vllm CPU benchmark suite improvement

- 链接: https://github.com/vllm-project/vllm/pull/34128
- 状态/时间: merged / 2026-02-12
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+802/-254，可读 patch 1243 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Vllm CPU benchmark suite improvement」；模型线: Llama 3.3 70B；类别: 文档/测试/CI；主要 diff: `.buildkite/performance-benchmarks/scripts/compare-json-results.py`, `.buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu-embed.json`。
- 实现要点:
  - `.buildkite/performance-benchmarks/scripts/compare-json-results.py` modified +368/-77；symbols: _sanitize_sheet_name, _group_to_sheet_base, _write_tables_to_excel_sheet, _safe_filename
  - `.buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh` modified +87/-46
  - `.buildkite/performance-benchmarks/tests/serving-tests-cpu-embed.json` added +41/-0
  - `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json` added +283/-0
- 代码 diff 细节:
  - `.buildkite/performance-benchmarks/scripts/compare-json-results.py` modified +368/-77
  - `.buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh` modified +87/-46
  - `.buildkite/performance-benchmarks/tests/serving-tests-cpu-embed.json` added +41/-0
  - `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json` added +283/-0
- 关键代码摘录:

```diff
diff -- .buildkite/performance-benchmarks/scripts/compare-json-results.py
@@ -9,8 +9,10 @@
 import os
 from dataclasses import dataclass
 from importlib import util
+from pathlib import Path

 import pandas as pd
+import regex as re

 pd.options.display.float_format = "{:.2f}".format
 plotly_found = util.find_spec("plotly.express") is not None
@@ -275,6 +277,131 @@ def _apply_two_decimals(
     return styler.format({c: "{:.2f}" for c in num_cols}, na_rep="")

diff -- .buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh
@@ -1,6 +1,4 @@
 #!/bin/bash
-
-# This script should be run inside the CI process
 # This script assumes that we are already inside the vllm/ directory
 # Benchmarking results will be available inside vllm/benchmarks/results/

@@ -9,6 +7,11 @@
 set -x
 set -o pipefail

+# Environment-driven debug controls (like ON_CPU=1)
+DRY_RUN="${DRY_RUN:-0}"
+MODEL_FILTER="${MODEL_FILTER:-}"
```
- 已读文件:
  - docs/bench: `.buildkite/performance-benchmarks/scripts/compare-json-results.py` modified +368/-77; `.buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh` modified +87/-46; `.buildkite/performance-benchmarks/tests/serving-tests-cpu-embed.json` added +41/-0; `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json` added +283/-0; `.buildkite/performance-benchmarks/tests/serving-tests-cpu.json` modified +0/-130; `docs/getting_started/installation/cpu.md` modified +23/-1
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #36398 - Allow `markdownlint` to run locally

- 链接: https://github.com/vllm-project/vllm/pull/36398
- 状态/时间: merged / 2026-03-09
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 47 个文件，+394/-392，可读 patch 1933 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Allow `markdownlint` to run locally」；模型线: Llama 3.3 70B；类别: 文档/测试/CI；主要 diff: `.github/mergify.yml`, `.pre-commit-config.yaml`, `benchmarks/attention_benchmarks/README.md`。
- 实现要点:
  - `.github/mergify.yml` modified +2/-4
  - `.pre-commit-config.yaml` modified +5/-5
  - `benchmarks/attention_benchmarks/README.md` modified +1/-1
  - `benchmarks/auto_tune/README.md` modified +1/-1
- 代码 diff 细节:
  - `.github/mergify.yml` modified +2/-4
  - `.pre-commit-config.yaml` modified +5/-5
  - `benchmarks/attention_benchmarks/README.md` modified +1/-1
  - `benchmarks/auto_tune/README.md` modified +1/-1
- 关键代码摘录:

```diff
diff -- .github/mergify.yml
@@ -38,15 +38,13 @@ pull_request_rules:

         > [!TIP]
         > <details>
-        > <summary>Is <code>mypy</code> or <code>markdownlint</code> failing?</summary>
+        > <summary>Is <code>mypy</code> failing?</summary>
         > <br/>
-        > <code>mypy</code> and <code>markdownlint</code> are run differently in CI. If the failure is related to either of these checks, please use the following commands to run them locally:
+        > <code>mypy</code> is run differently in CI. If the failure is related to this check, please use the following command to run it locally:
         >
         > ```bash
         > # For mypy (substitute "3.10" with the failing version if needed)
         > pre-commit run --hook-stage manual mypy-3.10
-        > # For markdownlint
diff -- .pre-commit-config.yaml
@@ -24,12 +24,12 @@ repos:
     exclude: 'csrc/(moe/topk_softmax_kernels.cu|quantization/gguf/(ggml-common.h|dequantize.cuh|vecdotq.cuh|mmq.cuh|mmvq.cuh))|vllm/third_party/.*'
     types_or: [c++, cuda]
     args: [--style=file, --verbose]
-- repo: https://github.com/igorshubovych/markdownlint-cli
-  rev: v0.45.0
+- repo: https://github.com/DavidAnson/markdownlint-cli2
+  rev: v0.21.0
   hooks:
-  - id: markdownlint
-    exclude: '.*\.inc\.md'
-    stages: [manual] # Only run in CI
+  - id: markdownlint-cli2
+    language_version: lts
```
- 已读文件:
  - docs/bench: `benchmarks/attention_benchmarks/README.md` modified +1/-1; `benchmarks/auto_tune/README.md` modified +1/-1; `docs/benchmarking/cli.md` modified +9/-9; `docs/benchmarking/dashboard.md` modified +6/-6; `docs/cli/bench/mm_processor.md` modified +1/-1; `docs/cli/json_tip.inc.md` modified +2/-1; `docs/configuration/optimization.md` modified +1/-1; `docs/contributing/README.md` modified +0/-1
  - other: `.github/mergify.yml` modified +2/-4; `.pre-commit-config.yaml` modified +5/-5
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #35086 - more models for vLLM Benchmark Suite

- 链接: https://github.com/vllm-project/vllm/pull/35086
- 状态/时间: merged / 2026-03-12
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+800/-119，可读 patch 1301 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「more models for vLLM Benchmark Suite」；模型线: Llama 3.3 70B；类别: 文档/测试/CI；主要 diff: `.buildkite/performance-benchmarks/scripts/compare-json-results.py`, `.buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu-asr.json`。
- 实现要点:
  - `.buildkite/performance-benchmarks/scripts/compare-json-results.py` modified +301/-90；symbols: _find_concurrency_col, _normalize_concurrency_in_df, _cell
  - `.buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh` modified +361/-4
  - `.buildkite/performance-benchmarks/tests/serving-tests-cpu-asr.json` added +37/-0
  - `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json` modified +72/-0
- 代码 diff 细节:
  - `.buildkite/performance-benchmarks/scripts/compare-json-results.py` modified +301/-90
  - `.buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh` modified +361/-4
  - `.buildkite/performance-benchmarks/tests/serving-tests-cpu-asr.json` added +37/-0
  - `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json` modified +72/-0
- 关键代码摘录:

```diff
diff -- .buildkite/performance-benchmarks/scripts/compare-json-results.py
@@ -7,12 +7,12 @@
 import html as _html
 import json
 import os
+from contextlib import nullcontext
 from dataclasses import dataclass
 from importlib import util
 from pathlib import Path

 import pandas as pd
-import regex as re

 pd.options.display.float_format = "{:.2f}".format
 plotly_found = util.find_spec("plotly.express") is not None
diff -- .buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh
@@ -12,6 +12,13 @@ DRY_RUN="${DRY_RUN:-0}"
 MODEL_FILTER="${MODEL_FILTER:-}"
 DTYPE_FILTER="${DTYPE_FILTER:-}"

+# Adaptive search controls
+ENABLE_ADAPTIVE_CONCURRENCY="${ENABLE_ADAPTIVE_CONCURRENCY:-0}"
+SLA_TTFT_MS="${SLA_TTFT_MS:-3000}"
+SLA_TPOT_MS="${SLA_TPOT_MS:-100}"
+ADAPTIVE_MAX_PROBES="${ADAPTIVE_MAX_PROBES:-8}"
+ADAPTIVE_MAX_CONCURRENCY="${ADAPTIVE_MAX_CONCURRENCY:-1024}"
+
 check_gpus() {
   if command -v nvidia-smi; then
     # check the number of GPUs and GPU type.
```
- 已读文件:
  - docs/bench: `.buildkite/performance-benchmarks/scripts/compare-json-results.py` modified +301/-90; `.buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh` modified +361/-4; `.buildkite/performance-benchmarks/tests/serving-tests-cpu-asr.json` added +37/-0; `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json` modified +72/-0; `.buildkite/performance-benchmarks/tests/serving-tests-cpu.json` modified +12/-23; `docs/benchmarking/dashboard.md` modified +6/-0
  - other: `requirements/test.in` modified +4/-1; `requirements/test.txt` modified +7/-1
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #38576 - vLLM Benchmark Suite perf regression after PR#32723

- 链接: https://github.com/vllm-project/vllm/pull/38576
- 状态/时间: merged / 2026-03-31
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+15/-1，可读 patch 119 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「vLLM Benchmark Suite perf regression after PR#32723」；模型线: Llama 3.3 70B；类别: 性能/后端优化；主要 diff: `.buildkite/performance-benchmarks/tests/serving-tests-arm64-cpu.json`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu-asr.json`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json`。
- 实现要点:
  - `.buildkite/performance-benchmarks/tests/serving-tests-arm64-cpu.json` modified +2/-1
  - `.buildkite/performance-benchmarks/tests/serving-tests-cpu-asr.json` modified +1/-0
  - `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json` modified +1/-0
  - `.buildkite/performance-benchmarks/tests/serving-tests-cpu.json` modified +1/-0
- 代码 diff 细节:
  - `.buildkite/performance-benchmarks/tests/serving-tests-arm64-cpu.json` modified +2/-1
  - `.buildkite/performance-benchmarks/tests/serving-tests-cpu-asr.json` modified +1/-0
  - `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json` modified +1/-0
  - `.buildkite/performance-benchmarks/tests/serving-tests-cpu.json` modified +1/-0
- 关键代码摘录:

```diff
diff -- .buildkite/performance-benchmarks/tests/serving-tests-arm64-cpu.json
@@ -36,6 +36,7 @@
       "model": "meta-llama/Llama-3.1-8B-Instruct",
       "backend": "vllm",
       "ignore-eos": "",
+      "temperature": 0,
       "num_prompts": 200
     }
   },
@@ -127,4 +128,4 @@
       }
     }
   ]
-}
\ No newline at end of file
diff -- .buildkite/performance-benchmarks/tests/serving-tests-cpu-asr.json
@@ -22,6 +22,7 @@
       "hf_split": "test",
       "no_stream": "",
       "no_oversample": "",
+      "temperature": 0,
       "num_prompts": 200
     }
   },
```
- 已读文件:
  - docs/bench: `.buildkite/performance-benchmarks/tests/serving-tests-arm64-cpu.json` modified +2/-1; `.buildkite/performance-benchmarks/tests/serving-tests-cpu-asr.json` modified +1/-0; `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json` modified +1/-0; `.buildkite/performance-benchmarks/tests/serving-tests-cpu.json` modified +1/-0; `.buildkite/performance-benchmarks/tests/serving-tests-hpu.json` modified +6/-0; `.buildkite/performance-benchmarks/tests/serving-tests.json` modified +4/-0
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #42607 - Update Intel Xeon model list and vLLM Benchmark Suite BKMs

- 链接: https://github.com/vllm-project/vllm/pull/42607
- 状态/时间: merged / 2026-05-15
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+118/-159，可读 patch 465 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Update Intel Xeon model list and vLLM Benchmark Suite BKMs」；模型线: Llama 3.3 70B；类别: 文档/测试/CI；主要 diff: `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json`, `docs/models/hardware_supported_models/cpu.md`。
- 实现要点:
  - `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json` modified +76/-143
  - `docs/models/hardware_supported_models/cpu.md` modified +42/-16
- 代码 diff 细节:
  - `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json` modified +76/-143
  - `docs/models/hardware_supported_models/cpu.md` modified +42/-16
- 关键代码摘录:

```diff
diff -- .buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json
@@ -31,30 +31,9 @@
     }
   },
   "tests": [
-    {
-      "test_name": "serving_llama8B_tp1_sharegpt",
-      "server_parameters": {
-        "tensor_parallel_size": 1
-      },
-      "client_parameters": {
-        "dataset_name": "sharegpt",
-        "dataset_path": "./ShareGPT_V3_unfiltered_cleaned_split.json"
-      }
-    },
diff -- docs/models/hardware_supported_models/cpu.md
@@ -11,24 +11,50 @@

 ### Text-only Language Models

-| Model                                | Architecture                             | Supported |
+| Model | Architecture | Supported |
 | ------------------------------------ | ---------------------------------------- | --------- |
-| meta-llama/Llama-3.1-8B-Instruct     | LlamaForCausalLM                         | ✅        |
-| meta-llama/Llama-3.2-3B-Instruct     | LlamaForCausalLM                         | ✅        |
-| ibm-granite/granite-3.2-2b-instruct  | GraniteForCausalLM                       | ✅        |
-| Qwen/Qwen3-1.7B                      | Qwen3ForCausalLM                         | ✅        |
-| Qwen/Qwen3-4B                        | Qwen3ForCausalLM                         | ✅        |
-| Qwen/Qwen3-8B                        | Qwen3ForCausalLM                         | ✅        |
-| zai-org/glm-4-9b-hf                  | GLMForCausalLM                           | ✅        |
```
- 已读文件:
  - docs/bench: `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json` modified +76/-143; `docs/models/hardware_supported_models/cpu.md` modified +42/-16
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。
