# sglang Llama 3.1 模型 PR 优化历史

## 2026-06-05 PR 补漏复核

已于 2026-06-05 按 sglang 上游 `origin/main@6cfdc1858` 复核；自上次时效基准（2026-05-19）以来，没有新的带 PR 编号的合并改动到所跟踪的实现文件，上方覆盖信息保持最新。


## 2026-05-19 新增覆盖

按 sglang 上游 `origin/main@5073c82a37`、模型相关文件的 `git log --name-only -- <model-files>` 以及 GitHub Pull Request files API 生成。本页用于补齐 sgl-cookbook 中 `Llama 3.1` 缺失的历史 PR 优化文档。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `skills/llm-serving-auto-benchmark/configs/cookbook-llm/llama-3.1-70b-instruct.yaml` | [#24250](https://github.com/sgl-project/sglang/pull/24250) |
| `docs_new/cookbook/autoregressive/Llama/Llama3.1.mdx` | [#23337](https://github.com/sgl-project/sglang/pull/23337), [#23001](https://github.com/sgl-project/sglang/pull/23001) |
| `docs_new/src/snippets/autoregressive/llama31-deployment.jsx` | [#23001](https://github.com/sgl-project/sglang/pull/23001) |
| `examples/chat_template/tool_chat_template_llama3.1_json.jinja` | [#13938](https://github.com/sgl-project/sglang/pull/13938), [#13935](https://github.com/sgl-project/sglang/pull/13935) |
| `test/registered/eval/test_text_models_gsm8k_eval.py` | [#21931](https://github.com/sgl-project/sglang/pull/21931), [#18886](https://github.com/sgl-project/sglang/pull/18886), [#15582](https://github.com/sgl-project/sglang/pull/15582) |
| `test/registered/amd/accuracy/mi30x/test_gsm8k_eval_amd.py` | [#21931](https://github.com/sgl-project/sglang/pull/21931), [#18911](https://github.com/sgl-project/sglang/pull/18911), [#17799](https://github.com/sgl-project/sglang/pull/17799), [#17895](https://github.com/sgl-project/sglang/pull/17895) |
| `test/registered/amd/perf/mi30x/test_text_models_perf_amd.py` | [#17895](https://github.com/sgl-project/sglang/pull/17895) |
| `test/manual/nightly/test_text_models_gsm8k_eval.py` | [#13610](https://github.com/sgl-project/sglang/pull/13610) |
| `test/manual/nightly/test_text_models_perf.py` | [#19778](https://github.com/sgl-project/sglang/pull/19778), [#13610](https://github.com/sgl-project/sglang/pull/13610) |

## PR 覆盖总览

- git 追溯 PR 数: 13
- 关键词/补充 PR 数: 0
- 当前文档总 PR 数: 13
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-11-20 | [#13610](https://github.com/sgl-project/sglang/pull/13610) | merged | Test reorganization: Move tests to manual/ | `test/manual/ascend/test_ascend_w8a8_quantization.py`, `test/manual/ascend/test_mindspore_models.py`, `test/manual/cpu/test_comm.py` |
| 2025-11-25 | [#13935](https://github.com/sgl-project/sglang/pull/13935) | merged | [misc] add llama3.1 chat template | `examples/chat_template/tool_chat_template_llama3.1_json.jinja` |
| 2025-11-25 | [#13938](https://github.com/sgl-project/sglang/pull/13938) | merged | [Minor] Fix lint | `examples/chat_template/tool_chat_template_llama3.1_json.jinja` |
| 2025-12-23 | [#15582](https://github.com/sgl-project/sglang/pull/15582) | merged | [CI] Migrate nightly tests to test/registered/ | `.github/workflows/nightly-test-nvidia.yml`, `python/sglang/test/accuracy_test_runner.py`, `python/sglang/test/ascend/__init__.py` |
| 2026-02-04 | [#17895](https://github.com/sgl-project/sglang/pull/17895) | merged | [AMD] Add kimi mi35x nightly test, folder organization and several stability fixes | `.github/workflows/nightly-test-amd.yml`, `python/sglang/test/nightly_utils.py`, `test/registered/amd/accuracy/mi30x/test_deepseek_r1_eval_amd.py` |
| 2026-02-12 | [#17799](https://github.com/sgl-project/sglang/pull/17799) | merged | [AMD] rocm 7.2 image release, PR test, Nightly Test | `.github/workflows/nightly-test-amd-rocm720.yml`, `.github/workflows/pr-test-amd-rocm720.yml`, `.github/workflows/pr-test-amd.yml` |
| 2026-02-17 | [#18886](https://github.com/sgl-project/sglang/pull/18886) | merged | Fix eval tests not capturing server launch failures | `python/sglang/srt/model_loader/ci_weight_validation.py`, `python/sglang/test/nightly_utils.py`, `test/registered/eval/test_text_models_gsm8k_eval.py` |
| 2026-02-25 | [#18911](https://github.com/sgl-project/sglang/pull/18911) | merged | [AMD] [GLM-5 Day 0] Add GLM-5 nightly test | `.github/workflows/nightly-test-amd-rocm720.yml`, `.github/workflows/nightly-test-amd.yml`, `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py` |
| 2026-03-07 | [#19778](https://github.com/sgl-project/sglang/pull/19778) | merged | Adding correct path for module not found error while collecting test | `python/sglang/multimodal_gen/csrc/attn/vmoba_attn/tests/test_vmoba_attn.py`, `test/manual/nightly/test_deepseek_v31_perf.py`, `test/manual/nightly/test_deepseek_v32_perf.py` |
| 2026-04-07 | [#21931](https://github.com/sgl-project/sglang/pull/21931) | merged | [CI] Migrate mgsm_en eval to gsm8k to remove openaipublic dependency | `test/registered/amd/accuracy/mi30x/test_gsm8k_eval_amd.py`, `test/registered/distributed/test_dp_attention_large.py`, `test/registered/distributed/test_pp_single_node.py` |
| 2026-04-20 | [#23001](https://github.com/sgl-project/sglang/pull/23001) | merged | Add new Mintlify documentation site (docs_new/) | `.gitignore`, `docs_new/.github/workflows/sync-lmsys-sglang-blogs.yml`, `docs_new/.gitignore` |
| 2026-04-21 | [#23337](https://github.com/sgl-project/sglang/pull/23337) | merged | [Docs] Sync docs_new with legacy docs and update migration redirects | `.pre-commit-config.yaml`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-Math-V2.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR-2.mdx` |
| 2026-05-02 | [#24250](https://github.com/sgl-project/sglang/pull/24250) | merged | [SKILL] Upgrade sglang profile and auto_benchmark skills | `agent-skills/llm-serving-auto-benchmark/SKILL.md`, `skills/llm-serving-auto-benchmark/configs/cookbook-llm/README.md`, `skills/llm-serving-auto-benchmark/configs/cookbook-llm/deepseek-math-v2.yaml` |

## 逐 PR diff 审计卡

### PR #13610 - Test reorganization: Move tests to manual/

- 链接: https://github.com/sgl-project/sglang/pull/13610
- 状态/时间: merged / 2025-11-20
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 74 个文件，+0/-74，可读 patch 87 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Test reorganization: Move tests to manual/」；模型线: Llama 3.1；类别: 文档/测试/CI；主要 diff: `test/manual/ascend/test_ascend_w8a8_quantization.py`, `test/manual/ascend/test_mindspore_models.py`, `test/manual/cpu/test_comm.py`。
- 实现要点:
  - `test/manual/ascend/test_ascend_w8a8_quantization.py` renamed +0/-0
  - `test/manual/ascend/test_mindspore_models.py` renamed +0/-0
  - `test/manual/cpu/test_comm.py` renamed +0/-0
  - `test/manual/debug_utils/test_log_parser.py` renamed +0/-0
- 代码 diff 细节:
  - `test/manual/ascend/test_ascend_w8a8_quantization.py` renamed +0/-0
  - `test/manual/ascend/test_mindspore_models.py` renamed +0/-0
  - `test/manual/cpu/test_comm.py` renamed +0/-0
  - `test/manual/debug_utils/test_log_parser.py` renamed +0/-0
- 关键代码摘录:

```diff
diff -- test/srt/run_suite.py
@@ -230,86 +230,12 @@
     ],
     "nightly-8-gpu-h20": [],
     "__not_in_ci__": [
-        TestFile("ascend/test_ascend_w8a8_quantization.py"),
-        TestFile("ascend/test_mindspore_models.py"),
-        TestFile("cpu/test_comm.py"),
-        TestFile("debug_utils/test_log_parser.py", 5),
-        TestFile("test_deepseek_v3_cutedsl_4gpu.py"),
-        TestFile("entrypoints/http_server/test_abort_request.py"),
-        TestFile("hicache/test_disaggregation_hicache.py"),
-        TestFile("layers/attention/nsa/test_act_quant_triton.py"),
-        TestFile("layers/moe/test_moe_runners.py"),
-        TestFile("lora/test_chunked_sgmv_backend.py"),
```
- 已读文件:
  - tests: `test/manual/ascend/test_ascend_w8a8_quantization.py` renamed +0/-0; `test/manual/ascend/test_mindspore_models.py` renamed +0/-0; `test/manual/cpu/test_comm.py` renamed +0/-0; `test/manual/debug_utils/test_log_parser.py` renamed +0/-0; `test/manual/entrypoints/http_server/test_abort_request.py` renamed +0/-0; `test/manual/hicache/test_disaggregation_hicache.py` renamed +0/-0; `test/manual/layers/attention/nsa/test_act_quant_triton.py` renamed +0/-0; `test/manual/layers/moe/test_moe_runners.py` renamed +0/-0
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #13935 - [misc] add llama3.1 chat template

- 链接: https://github.com/sgl-project/sglang/pull/13935
- 状态/时间: merged / 2025-11-25
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+121/-0，可读 patch 123 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[misc] add llama3.1 chat template」；模型线: Llama 3.1；类别: 模型支持/运行时入口；主要 diff: `examples/chat_template/tool_chat_template_llama3.1_json.jinja`。
- 实现要点:
  - `examples/chat_template/tool_chat_template_llama3.1_json.jinja` added +121/-0
- 代码 diff 细节:
  - `examples/chat_template/tool_chat_template_llama3.1_json.jinja` added +121/-0
- 关键代码摘录:

```diff
diff -- examples/chat_template/tool_chat_template_llama3.1_json.jinja
@@ -0,0 +1,121 @@
+{# Copied from https://github.com/vllm-project/vllm/blob/main/examples/tool_chat_template_llama3.1_json.jinja to enable better model response. #}
+{{- bos_token }}
+{%- if custom_tools is defined %}
+    {%- set tools = custom_tools %}
+{%- endif %}
+{%- if not tools_in_user_message is defined %}
+    {#- Llama 3.1 doesn't pass all tests if the tools are in the system prompt #}
+    {%- set tools_in_user_message = true %}
+{%- endif %}
+{%- if not date_string is defined %}
+    {%- if strftime_now is defined %}
+        {%- set date_string = strftime_now("%d %b %Y") %}
+    {%- else %}
```
- 已读文件:
  - docs/bench: `examples/chat_template/tool_chat_template_llama3.1_json.jinja` added +121/-0
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #13938 - [Minor] Fix lint

- 链接: https://github.com/sgl-project/sglang/pull/13938
- 状态/时间: merged / 2025-11-25
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 7 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Minor] Fix lint」；模型线: Llama 3.1；类别: 缺陷修复；主要 diff: `examples/chat_template/tool_chat_template_llama3.1_json.jinja`。
- 实现要点:
  - `examples/chat_template/tool_chat_template_llama3.1_json.jinja` modified +1/-1
- 代码 diff 细节:
  - `examples/chat_template/tool_chat_template_llama3.1_json.jinja` modified +1/-1
- 关键代码摘录:

```diff
diff -- examples/chat_template/tool_chat_template_llama3.1_json.jinja
@@ -118,4 +118,4 @@
 {%- endfor %}
 {%- if add_generation_prompt %}
     {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
-{%- endif %}
\ No newline at end of file
+{%- endif %}
```
- 已读文件:
  - docs/bench: `examples/chat_template/tool_chat_template_llama3.1_json.jinja` modified +1/-1
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #15582 - [CI] Migrate nightly tests to test/registered/

- 链接: https://github.com/sgl-project/sglang/pull/15582
- 状态/时间: merged / 2025-12-23
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 64 个文件，+93/-140，可读 patch 611 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI] Migrate nightly tests to test/registered/」；模型线: Llama 3.1；类别: 文档/测试/CI；主要 diff: `.github/workflows/nightly-test-nvidia.yml`, `python/sglang/test/accuracy_test_runner.py`, `python/sglang/test/ascend/__init__.py`。
- 实现要点:
  - `.github/workflows/nightly-test-nvidia.yml` modified +4/-4
  - `python/sglang/test/accuracy_test_runner.py` renamed +0/-0
  - `python/sglang/test/ascend/__init__.py` added +0/-0
  - `python/sglang/test/ascend/gsm8k_ascend_mixin.py` renamed +0/-0
- 代码 diff 细节:
  - `.github/workflows/nightly-test-nvidia.yml` modified +4/-4
  - `python/sglang/test/accuracy_test_runner.py` renamed +0/-0
  - `python/sglang/test/ascend/__init__.py` added +0/-0
  - `python/sglang/test/ascend/gsm8k_ascend_mixin.py` renamed +0/-0
- 关键代码摘录:

```diff
diff -- .github/workflows/nightly-test-nvidia.yml
@@ -191,7 +191,7 @@ jobs:
         timeout-minutes: 120
         run: |
           cd test
-          python3 nightly/test_text_models_gsm8k_eval.py
+          python3 run_suite.py --hw cuda --suite nightly-eval-text-2-gpu --nightly --continue-on-error --timeout-per-file 4500

   # Text model performance tests
   nightly-test-text-perf-2-gpu-runner:
@@ -216,7 +216,7 @@ jobs:
         run: |
           cd test
           rm -rf performance_profiles_text_models/
-          python3 nightly/test_text_models_perf.py
diff -- python/sglang/test/ascend/vlm_utils.py
@@ -87,7 +87,7 @@ def run_mmmu_eval(
             "--limit",
             limit,
             "--config",
-            "/__w/sglang/sglang/test/nightly/ascend/vlm_models/mmmu-val.yaml",
+            "/__w/sglang/sglang/test/registered/ascend/vlm_models/mmmu-val.yaml",
         ]

         subprocess.run(
```
- 已读文件:
  - runtime: `python/sglang/test/accuracy_test_runner.py` renamed +0/-0; `python/sglang/test/ascend/__init__.py` added +0/-0; `python/sglang/test/ascend/gsm8k_ascend_mixin.py` renamed +0/-0; `python/sglang/test/ascend/vlm_utils.py` renamed +1/-1; `python/sglang/test/nightly_utils.py` renamed +0/-0; `python/sglang/test/performance_test_runner.py` renamed +1/-2; `python/sglang/test/run_combined_tests.py` renamed +3/-4
  - tests: `test/registered/8-gpu-models/test_deepseek_v31.py` modified +3/-9; `test/registered/8-gpu-models/test_deepseek_v32.py` modified +3/-9; `test/registered/8-gpu-models/test_glm_46.py` modified +3/-9; `test/registered/8-gpu-models/test_kimi_k2.py` modified +3/-9; `test/registered/8-gpu-models/test_minimax_m2.py` modified +3/-9; `test/registered/8-gpu-models/test_mistral_large3.py` modified +3/-9; `test/registered/8-gpu-models/test_qwen3_235b.py` modified +3/-9; `test/registered/ascend/embedding_models/test_ascend_embedding_models.py` renamed +0/-0
  - other: `.github/workflows/nightly-test-nvidia.yml` modified +4/-4
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #17895 - [AMD] Add kimi mi35x nightly test, folder organization and several stability fixes

- 链接: https://github.com/sgl-project/sglang/pull/17895
- 状态/时间: merged / 2026-02-04
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 34 个文件，+184/-14，可读 patch 414 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Add kimi mi35x nightly test, folder organization and several stability fixes」；模型线: Llama 3.1；类别: 缺陷修复；主要 diff: `.github/workflows/nightly-test-amd.yml`, `python/sglang/test/nightly_utils.py`, `test/registered/amd/accuracy/mi30x/test_deepseek_r1_eval_amd.py`。
- 实现要点:
  - `.github/workflows/nightly-test-amd.yml` modified +40/-5
  - `python/sglang/test/nightly_utils.py` modified +16/-4
  - `test/registered/amd/accuracy/mi30x/test_deepseek_r1_eval_amd.py` renamed +0/-0
  - `test/registered/amd/accuracy/mi30x/test_deepseek_v31_eval_amd.py` renamed +0/-0
- 代码 diff 细节:
  - `.github/workflows/nightly-test-amd.yml` modified +40/-5
  - `python/sglang/test/nightly_utils.py` modified +16/-4
  - `test/registered/amd/accuracy/mi30x/test_deepseek_r1_eval_amd.py` renamed +0/-0
  - `test/registered/amd/accuracy/mi30x/test_deepseek_v31_eval_amd.py` renamed +0/-0
- 关键代码摘录:

```diff
diff -- .github/workflows/nightly-test-amd.yml
@@ -34,6 +34,7 @@ on:
           - 'nightly-8-gpu-kimi-k2'
           # MI35x jobs
           - 'nightly-test-1-gpu-mi35x'
+          - 'nightly-8-gpu-mi35x-kimi-k2'
           - 'nightly-accuracy-8-gpu-mi35x'
           - 'nightly-8-gpu-mi35x-grok1-int4'
           - 'nightly-8-gpu-mi35x-grok2'
@@ -582,13 +583,13 @@ jobs:
           bash scripts/ci/amd/amd_ci_exec.sh pip install tabulate

       - name: Accuracy Test MI35x (8-GPU Grok1-INT4)
-        timeout-minutes: 60
+        timeout-minutes: 90
diff -- python/sglang/test/nightly_utils.py
@@ -94,6 +94,7 @@ def build_benchmark_command(
         json_output_file: str,
         extra_args: Optional[List[str]] = None,
         server_args: Optional[List[str]] = None,
+        enable_profile: bool = True,
     ) -> List[str]:
         """Build the benchmark command with all required arguments.

@@ -106,6 +107,7 @@ def build_benchmark_command(
             json_output_file: Path to JSON output file
             extra_args: Optional extra arguments to append to command
             server_args: Optional server launch arguments to record in metrics
+            enable_profile: Whether to enable profiling (default True for NVIDIA)

```
- 已读文件:
  - runtime: `python/sglang/test/nightly_utils.py` modified +16/-4
  - tests: `test/registered/amd/accuracy/mi30x/test_deepseek_r1_eval_amd.py` renamed +0/-0; `test/registered/amd/accuracy/mi30x/test_deepseek_v31_eval_amd.py` renamed +0/-0; `test/registered/amd/accuracy/mi30x/test_deepseek_v32_dp_eval_amd.py` renamed +0/-0; `test/registered/amd/accuracy/mi30x/test_deepseek_v32_eval_amd.py` renamed +0/-0; `test/registered/amd/accuracy/mi30x/test_deepseek_v32_mtp_eval_amd.py` renamed +0/-0; `test/registered/amd/accuracy/mi30x/test_deepseek_v32_tc_eval_amd.py` renamed +0/-0; `test/registered/amd/accuracy/mi30x/test_gpt_oss_eval_amd.py` renamed +0/-0; `test/registered/amd/accuracy/mi30x/test_grok1_fp8_eval_amd.py` renamed +0/-0
  - other: `.github/workflows/nightly-test-amd.yml` modified +40/-5
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #17799 - [AMD] rocm 7.2 image release, PR test, Nightly Test

- 链接: https://github.com/sgl-project/sglang/pull/17799
- 状态/时间: merged / 2026-02-12
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 26 个文件，+2719/-156，可读 patch 3314 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] rocm 7.2 image release, PR test, Nightly Test」；模型线: Llama 3.1；类别: 文档/测试/CI；主要 diff: `.github/workflows/nightly-test-amd-rocm720.yml`, `.github/workflows/pr-test-amd-rocm720.yml`, `.github/workflows/pr-test-amd.yml`。
- 实现要点:
  - `.github/workflows/nightly-test-amd-rocm720.yml` added +868/-0
  - `.github/workflows/pr-test-amd-rocm720.yml` added +793/-0
  - `.github/workflows/pr-test-amd.yml` modified +20/-2
  - `.github/workflows/release-docker-amd-rocm720-nightly-preview.yml` added +82/-0
- 代码 diff 细节:
  - `.github/workflows/nightly-test-amd-rocm720.yml` added +868/-0
  - `.github/workflows/pr-test-amd-rocm720.yml` added +793/-0
  - `.github/workflows/pr-test-amd.yml` modified +20/-2
  - `.github/workflows/release-docker-amd-rocm720-nightly-preview.yml` added +82/-0
- 关键代码摘录:

```diff
diff -- .github/workflows/nightly-test-amd-rocm720.yml
@@ -0,0 +1,868 @@
+name: Nightly Test (AMD ROCm 7.2)
+
+on:
+  schedule:
+    - cron: '0 2 * * *'
+  push:
+    branches:
+      - main
+    paths:
+      - "python/sglang/version.py"
+  workflow_dispatch:
+    inputs:
+      job_filter:
diff -- .github/workflows/pr-test-amd-rocm720.yml
@@ -0,0 +1,793 @@
+name: PR Test ROCm 7.2 (AMD)
+# Dynamic run-name for /rerun-stage commands to enable URL lookup
+# Format: "[stage-name] sha" for fork PRs, "[stage-name]" for non-fork, default for normal runs
+run-name: ${{ inputs.target_stage && (inputs.pr_head_sha && format('[{0}] {1}', inputs.target_stage, inputs.pr_head_sha) || format('[{0}]', inputs.target_stage)) || '' }}
+
+on:
+  # run rocm 720 pr tests once a day at 2am UTC to avoid overwhelming the CI system
+  schedule:
+    - cron: '0 2 * * *'
+  # push:
+  #   branches: [ main ]
+  #   paths:
+  #     - "python/**"
```
- 已读文件:
  - runtime: `python/sglang/srt/layers/layernorm.py` modified +14/-1; `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +43/-12; `python/sglang/srt/layers/moe/moe_runner/triton.py` modified +16/-2; `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +45/-4; `python/sglang/srt/layers/quantization/unquant.py` modified +8/-2; `python/sglang/srt/models/deepseek_janus_pro.py` modified +1/-1; `python/sglang/srt/server_args.py` modified +7/-0; `python/sglang/test/gpt_oss_common.py` modified +2/-1
  - tests: `test/registered/amd/accuracy/mi30x/test_gpt_oss_eval_amd.py` modified +2/-2; `test/registered/amd/accuracy/mi30x/test_gsm8k_eval_amd.py` modified +4/-4; `test/registered/amd/accuracy/mi30x/test_vlms_mmmu_eval_amd.py` modified +3/-3; `test/registered/amd/accuracy/mi35x/test_deepseek_v32_eval_mi35x.py` modified +3/-3; `test/registered/amd/accuracy/mi35x/test_deepseek_v32_mtp_eval_mi35x.py` modified +12/-10; `test/registered/amd/accuracy/mi35x/test_gpt_oss_eval_mi35x.py` modified +2/-6; `test/registered/amd/perf/mi35x/test_deepseek_v32_basic_perf_mi35x.py` modified +1/-0; `test/registered/layers/mamba/test_mamba_ssm_ssd.py` modified +5/-0
  - other: `.github/workflows/nightly-test-amd-rocm720.yml` added +868/-0; `.github/workflows/pr-test-amd-rocm720.yml` added +793/-0; `.github/workflows/pr-test-amd.yml` modified +20/-2; `.github/workflows/release-docker-amd-rocm720-nightly-preview.yml` added +82/-0; `docker/rocm720.Dockerfile` added +502/-0; `scripts/ci/amd/amd_ci_install_dependency.sh` modified +109/-89; `scripts/ci/amd/amd_ci_start_container.sh` modified +86/-11
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #18886 - Fix eval tests not capturing server launch failures

- 链接: https://github.com/sgl-project/sglang/pull/18886
- 状态/时间: merged / 2026-02-17
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+137/-95，可读 patch 410 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix eval tests not capturing server launch failures」；模型线: Llama 3.1；类别: 缺陷修复；主要 diff: `python/sglang/srt/model_loader/ci_weight_validation.py`, `python/sglang/test/nightly_utils.py`, `test/registered/eval/test_text_models_gsm8k_eval.py`。
- 实现要点:
  - `python/sglang/srt/model_loader/ci_weight_validation.py` modified +91/-57；symbols: _get_lock_file_path
  - `python/sglang/test/nightly_utils.py` modified +16/-12
  - `test/registered/eval/test_text_models_gsm8k_eval.py` modified +15/-14
  - `test/registered/eval/test_vlms_mmmu_eval.py` modified +15/-12
- 代码 diff 细节:
  - `python/sglang/srt/model_loader/ci_weight_validation.py` modified +91/-57
  - `python/sglang/test/nightly_utils.py` modified +16/-12
  - `test/registered/eval/test_text_models_gsm8k_eval.py` modified +15/-14
  - `test/registered/eval/test_vlms_mmmu_eval.py` modified +15/-12
- 关键代码摘录:

```diff
diff -- python/sglang/srt/model_loader/ci_weight_validation.py
@@ -1730,30 +1730,44 @@ def _validate_weights_after_download(
     return True


-def _get_lock_file_path(model_name_or_path: str) -> str:
+def _get_lock_file_path(
+    model_name_or_path: str, cache_dir: Optional[str] = None
+) -> str:
     """
     Generate a unique lock file path for download coordination.

-    Uses file-based locking (fcntl.flock) to ensure only one process downloads
-    while others wait. This works regardless of how processes are spawned
-    (mp.Process, torchrun, etc.).
diff -- python/sglang/test/nightly_utils.py
@@ -259,18 +259,21 @@ def run_benchmark_for_model(
         avg_spec_accept_length = None
         model_description = f"{model_path}" + (f" ({variant})" if variant else "")

-        # Launch server
-        process = popen_launch_server(
-            model=model_path,
-            base_url=self.base_url,
-            other_args=other_args or [],
-            timeout=(
-                timeout if timeout is not None else DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
-            ),
-            env=env,
-        )
```
- 已读文件:
  - runtime: `python/sglang/srt/model_loader/ci_weight_validation.py` modified +91/-57; `python/sglang/test/nightly_utils.py` modified +16/-12
  - tests: `test/registered/eval/test_text_models_gsm8k_eval.py` modified +15/-14; `test/registered/eval/test_vlms_mmmu_eval.py` modified +15/-12
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #18911 - [AMD] [GLM-5 Day 0] Add GLM-5 nightly test

- 链接: https://github.com/sgl-project/sglang/pull/18911
- 状态/时间: merged / 2026-02-25
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+635/-1，可读 patch 725 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] [GLM-5 Day 0] Add GLM-5 nightly test」；模型线: Llama 3.1；类别: 模型支持/运行时入口；主要 diff: `.github/workflows/nightly-test-amd-rocm720.yml`, `.github/workflows/nightly-test-amd.yml`, `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py`。
- 实现要点:
  - `.github/workflows/nightly-test-amd-rocm720.yml` modified +71/-0
  - `.github/workflows/nightly-test-amd.yml` modified +70/-0
  - `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py` added +244/-0；symbols: ModelConfig, get_display_name, get_one_example, get_few_shot_examples
  - `test/registered/amd/accuracy/mi30x/test_gsm8k_eval_amd.py` modified +1/-1
- 代码 diff 细节:
  - `.github/workflows/nightly-test-amd-rocm720.yml` modified +71/-0
  - `.github/workflows/nightly-test-amd.yml` modified +70/-0
  - `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py` added +244/-0
  - `test/registered/amd/accuracy/mi30x/test_gsm8k_eval_amd.py` modified +1/-1
- 关键代码摘录:

```diff
diff -- .github/workflows/nightly-test-amd-rocm720.yml
@@ -32,6 +32,7 @@ on:
           - 'nightly-8-gpu-deepseek-v32-rocm720'
           - 'nightly-8-gpu-deepseek-v32-mtp-rocm720'
           - 'nightly-8-gpu-kimi-k25-rocm720'
+          - 'nightly-8-gpu-glm5-rocm720'
           # MI35x ROCm 7.2 jobs
           - 'nightly-test-1-gpu-mi35x-rocm720'
           - 'nightly-accuracy-8-gpu-mi35x-rocm720'
@@ -43,6 +44,7 @@ on:
           - 'nightly-perf-8-gpu-mi35x-deepseek-v32-basic-rocm720'
           - 'nightly-perf-8-gpu-mi35x-deepseek-v32-mtp-rocm720'
           - 'nightly-8-gpu-mi35x-kimi-k25-rocm720'
+          - 'nightly-8-gpu-mi35x-glm5-rocm720'
   workflow_call:
diff -- .github/workflows/nightly-test-amd.yml
@@ -32,9 +32,11 @@ on:
           - 'nightly-8-gpu-deepseek-v32'
           - 'nightly-8-gpu-deepseek-v32-mtp'
           - 'nightly-8-gpu-kimi-k25'
+          - 'nightly-8-gpu-glm5'
           # MI35x jobs
           - 'nightly-test-1-gpu-mi35x'
           - 'nightly-8-gpu-mi35x-kimi-k25'
+          - 'nightly-8-gpu-mi35x-glm5'
           - 'nightly-accuracy-8-gpu-mi35x'
           - 'nightly-8-gpu-mi35x-grok1-int4'
           - 'nightly-8-gpu-mi35x-grok2'
@@ -494,6 +496,38 @@ jobs:
           echo "$(<github_summary.md )" >> $GITHUB_STEP_SUMMARY || true
```
- 已读文件:
  - tests: `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py` added +244/-0; `test/registered/amd/accuracy/mi30x/test_gsm8k_eval_amd.py` modified +1/-1; `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py` added +249/-0
  - other: `.github/workflows/nightly-test-amd-rocm720.yml` modified +71/-0; `.github/workflows/nightly-test-amd.yml` modified +70/-0
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #19778 - Adding correct path for module not found error while collecting test

- 链接: https://github.com/sgl-project/sglang/pull/19778
- 状态/时间: merged / 2026-03-07
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+8/-13，可读 patch 63 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Adding correct path for module not found error while collecting test」；模型线: Llama 3.1；类别: 模型支持/运行时入口；主要 diff: `python/sglang/multimodal_gen/csrc/attn/vmoba_attn/tests/test_vmoba_attn.py`, `test/manual/nightly/test_deepseek_v31_perf.py`, `test/manual/nightly/test_deepseek_v32_perf.py`。
- 实现要点:
  - `python/sglang/multimodal_gen/csrc/attn/vmoba_attn/tests/test_vmoba_attn.py` modified +1/-2
  - `test/manual/nightly/test_deepseek_v31_perf.py` modified +1/-2
  - `test/manual/nightly/test_deepseek_v32_perf.py` modified +1/-2
  - `test/manual/nightly/test_text_models_perf.py` modified +1/-2
- 代码 diff 细节:
  - `python/sglang/multimodal_gen/csrc/attn/vmoba_attn/tests/test_vmoba_attn.py` modified +1/-2
  - `test/manual/nightly/test_deepseek_v31_perf.py` modified +1/-2
  - `test/manual/nightly/test_deepseek_v32_perf.py` modified +1/-2
  - `test/manual/nightly/test_text_models_perf.py` modified +1/-2
- 关键代码摘录:

```diff
diff -- python/sglang/multimodal_gen/csrc/attn/vmoba_attn/tests/test_vmoba_attn.py
@@ -4,8 +4,7 @@

 import pytest
 import torch
-from csrc.attn.vmoba_attn.vmoba import moba_attn_varlen
-
+from sglang.multimodal_gen.csrc.attn.vmoba_attn.vmoba import moba_attn_varlen

 def generate_test_data(
     batch_size, total_seqlen, num_heads, head_dim, dtype, device="cuda"
diff -- test/manual/nightly/test_deepseek_v31_perf.py
@@ -1,7 +1,6 @@
 import unittest

-from nightly_utils import NightlyBenchmarkRunner
-
+from sglang.test.nightly_utils import NightlyBenchmarkRunner
 from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, _parse_int_list_env

 DEEPSEEK_V31_MODEL_PATH = "deepseek-ai/DeepSeek-V3.1"
```
- 已读文件:
  - runtime: `python/sglang/multimodal_gen/csrc/attn/vmoba_attn/tests/test_vmoba_attn.py` modified +1/-2
  - tests: `test/manual/nightly/test_deepseek_v31_perf.py` modified +1/-2; `test/manual/nightly/test_deepseek_v32_perf.py` modified +1/-2; `test/manual/nightly/test_text_models_perf.py` modified +1/-2; `test/manual/nightly/test_vlms_perf.py` modified +1/-2; `test/manual/test_two_batch_overlap.py` modified +3/-3
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #21931 - [CI] Migrate mgsm_en eval to gsm8k to remove openaipublic dependency

- 链接: https://github.com/sgl-project/sglang/pull/21931
- 状态/时间: merged / 2026-04-07
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+82/-77，可读 patch 336 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI] Migrate mgsm_en eval to gsm8k to remove openaipublic dependency」；模型线: Llama 3.1；类别: 文档/测试/CI；主要 diff: `test/registered/amd/accuracy/mi30x/test_gsm8k_eval_amd.py`, `test/registered/distributed/test_dp_attention_large.py`, `test/registered/distributed/test_pp_single_node.py`。
- 实现要点:
  - `test/registered/amd/accuracy/mi30x/test_gsm8k_eval_amd.py` modified +27/-26；symbols: test_gsm8k_all_models
  - `test/registered/distributed/test_dp_attention_large.py` modified +2/-2；symbols: test_gsm8k
  - `test/registered/distributed/test_pp_single_node.py` modified +2/-2；symbols: test_gsm8k
  - `test/registered/eval/test_text_models_gsm8k_eval.py` modified +22/-21；symbols: test_gsm8k_all_models
- 代码 diff 细节:
  - `test/registered/amd/accuracy/mi30x/test_gsm8k_eval_amd.py` modified +27/-26
  - `test/registered/distributed/test_dp_attention_large.py` modified +2/-2
  - `test/registered/distributed/test_pp_single_node.py` modified +2/-2
  - `test/registered/eval/test_text_models_gsm8k_eval.py` modified +22/-21
- 关键代码摘录:

```diff
diff -- test/registered/amd/accuracy/mi30x/test_gsm8k_eval_amd.py
@@ -1,7 +1,7 @@
 """
 AMD GSM8K Evaluation Test (Migrated from test/srt/nightly/)

-This test evaluates instruction-tuned models on the mgsm_en benchmark using chat completions.
+This test evaluates instruction-tuned models on the gsm8k benchmark using chat completions.
 Models are tested with various TP configurations on AMD GPUs.

 Registry: nightly-amd suite (2-GPU tests)
@@ -35,34 +35,35 @@
 register_amd_ci(est_time=3600, suite="nightly-amd", nightly=True)

 MODEL_SCORE_THRESHOLDS = {
+    # Thresholds set at 5% below reported GSM8K (5-shot/CoT) scores
diff -- test/registered/distributed/test_dp_attention_large.py
@@ -56,11 +56,11 @@ def setUpClass(cls):
     def tearDownClass(cls):
         kill_process_tree(cls.process.pid)

-    def test_mgsm_en(self):
+    def test_gsm8k(self):
         args = SimpleNamespace(
             base_url=self.base_url,
             model=self.model,
-            eval_name="mgsm_en",
+            eval_name="gsm8k",
             num_examples=None,
             num_threads=1024,
         )
```
- 已读文件:
  - tests: `test/registered/amd/accuracy/mi30x/test_gsm8k_eval_amd.py` modified +27/-26; `test/registered/distributed/test_dp_attention_large.py` modified +2/-2; `test/registered/distributed/test_pp_single_node.py` modified +2/-2; `test/registered/eval/test_text_models_gsm8k_eval.py` modified +22/-21; `test/registered/piecewise_cuda_graph/test_piecewise_cuda_graph_support_1_gpu.py` modified +16/-16; `test/registered/quant/test_quantization.py` modified +8/-5; `test/registered/scheduler/test_prefill_delayer.py` modified +5/-5
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #23001 - Add new Mintlify documentation site (docs_new/)

- 链接: https://github.com/sgl-project/sglang/pull/23001
- 状态/时间: merged / 2026-04-20
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 100 个文件，+19458/-0，可读 patch 19508 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add new Mintlify documentation site (docs_new/)」；模型线: Llama 3.1；类别: 模型支持/运行时入口；主要 diff: `.gitignore`, `docs_new/.github/workflows/sync-lmsys-sglang-blogs.yml`, `docs_new/.gitignore`。
- 实现要点:
  - `.gitignore` modified +1/-0
  - `docs_new/.github/workflows/sync-lmsys-sglang-blogs.yml` added +39/-0
  - `docs_new/.gitignore` added +30/-0
  - `docs_new/.mintignore` added +7/-0
- 代码 diff 细节:
  - `.gitignore` modified +1/-0
  - `docs_new/.github/workflows/sync-lmsys-sglang-blogs.yml` added +39/-0
  - `docs_new/.gitignore` added +30/-0
  - `docs_new/.mintignore` added +7/-0
- 关键代码摘录:

```diff
diff -- .gitignore
@@ -192,6 +192,7 @@ work_dirs/
 *.csv

 !logo.png
+!docs_new/images/*.png

 # Prerequisites
 *.d
diff -- docs_new/.github/workflows/sync-lmsys-sglang-blogs.yml
@@ -0,0 +1,39 @@
+name: Sync LMSYS SGLang blogs
+
+on:
+  workflow_dispatch:
+  schedule:
+    - cron: "0 */12 * * *"
+
+permissions:
+  contents: write
+
+jobs:
+  sync:
+    runs-on: ubuntu-latest
```
- 已读文件:
  - docs/bench: `docs_new/.github/workflows/sync-lmsys-sglang-blogs.yml` added +39/-0; `docs_new/.gitignore` added +30/-0; `docs_new/.mintignore` added +7/-0; `docs_new/AGENTS.md` added +381/-0; `docs_new/CONTRIBUTING.md` added +34/-0; `docs_new/LICENSE` added +201/-0; `docs_new/README.md` added +126/-0; `docs_new/cards/Autoregressive-benchmark-card.png` added +0/-0
  - other: `.gitignore` modified +1/-0
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #23337 - [Docs] Sync docs_new with legacy docs and update migration redirects

- 链接: https://github.com/sgl-project/sglang/pull/23337
- 状态/时间: merged / 2026-04-21
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 100 个文件，+3881/-1454，可读 patch 7195 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Docs] Sync docs_new with legacy docs and update migration redirects」；模型线: Llama 3.1；类别: 文档/测试/CI；主要 diff: `.pre-commit-config.yaml`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-Math-V2.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR-2.mdx`。
- 实现要点:
  - `.pre-commit-config.yaml` modified +7/-0
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-Math-V2.mdx` modified +1/-1
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR-2.mdx` modified +1/-1
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR.mdx` modified +1/-1
- 代码 diff 细节:
  - `.pre-commit-config.yaml` modified +7/-0
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-Math-V2.mdx` modified +1/-1
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR-2.mdx` modified +1/-1
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR.mdx` modified +1/-1
- 关键代码摘录:

```diff
diff -- .pre-commit-config.yaml
@@ -93,6 +93,13 @@ repos:
         language: system
         files: ^test/registered/.*\.py$
         pass_filenames: false
+      - id: check-no-docs-changes
+        name: reject changes under legacy docs/
+        entry: python3 scripts/ci/check_no_docs_changes.py
+        language: system
+        pass_filenames: false
+        always_run: true
+        stages: [pre-commit]
   - repo: https://github.com/lycheeverse/lychee.git
     rev: lychee-v0.22.0
     hooks:
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-Math-V2.mdx
@@ -26,7 +26,7 @@ To use DeepSeek-Math-V2, you must agree to DeepSeek's Community License. See [LI

 ## 2. SGLang Installation

-Please refer to the [official SGLang installation guide](../../../docs/get-started/installation) for installation instructions.
+Please refer to the [official SGLang installation guide](../../../docs/get-started/install) for installation instructions.

 ## 3. Model Deployment

```
- 已读文件:
  - docs/bench: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-Math-V2.mdx` modified +1/-1; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR-2.mdx` modified +1/-1; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR.mdx` modified +1/-1; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-R1.mdx` modified +1/-1; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3.mdx` modified +1/-1; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_1.mdx` modified +1/-1; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx` modified +1/-1; `docs_new/cookbook/autoregressive/Ernie/Ernie4.5.mdx` modified +1/-1
  - other: `.pre-commit-config.yaml` modified +7/-0
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #24250 - [SKILL] Upgrade sglang profile and auto_benchmark skills

- 链接: https://github.com/sgl-project/sglang/pull/24250
- 状态/时间: merged / 2026-05-02
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 100 个文件，+9334/-3813，可读 patch 13573 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[SKILL] Upgrade sglang profile and auto_benchmark skills」；模型线: Llama 3.1；类别: 文档/测试/CI；主要 diff: `agent-skills/llm-serving-auto-benchmark/SKILL.md`, `skills/llm-serving-auto-benchmark/configs/cookbook-llm/README.md`, `skills/llm-serving-auto-benchmark/configs/cookbook-llm/deepseek-math-v2.yaml`。
- 实现要点:
  - `agent-skills/llm-serving-auto-benchmark/SKILL.md` added +527/-0
  - `skills/llm-serving-auto-benchmark/configs/cookbook-llm/README.md` added +17/-0
  - `skills/llm-serving-auto-benchmark/configs/cookbook-llm/deepseek-math-v2.yaml` added +130/-0
  - `skills/llm-serving-auto-benchmark/configs/cookbook-llm/deepseek-r1-0528.yaml` added +133/-0
- 代码 diff 细节:
  - `agent-skills/llm-serving-auto-benchmark/SKILL.md` added +527/-0
  - `skills/llm-serving-auto-benchmark/configs/cookbook-llm/README.md` added +17/-0
  - `skills/llm-serving-auto-benchmark/configs/cookbook-llm/deepseek-math-v2.yaml` added +130/-0
  - `skills/llm-serving-auto-benchmark/configs/cookbook-llm/deepseek-r1-0528.yaml` added +133/-0
- 关键代码摘录:

```diff
diff -- agent-skills/llm-serving-auto-benchmark/SKILL.md
@@ -0,0 +1,527 @@
+---
+name: llm-serving-auto-benchmark
+description: Framework-independent LLM serving benchmark skill for comparing SGLang, vLLM, TensorRT-LLM, or another serving framework. Use when a user wants to find the best deployment command for one model across multiple serving frameworks under the same workload, GPU budget, and latency SLA.
+---
+
+# LLM Serving Auto Benchmark
+
+## Overview
+
+Use this skill to compare LLM serving frameworks such as SGLang, vLLM, and
+TensorRT-LLM for the same model and workload.
+
+Use a config-driven workflow:
diff -- skills/llm-serving-auto-benchmark/configs/cookbook-llm/README.md
@@ -0,0 +1,17 @@
+# Cookbook LLM Configs
+
+These configs define a framework-neutral LLM serving cookbook model set and translate each model into a three-framework run plan for SGLang, vLLM, and TensorRT-LLM.
+
+Scope:
+- SGLang can preserve source-recipe `base_flags` and `search_space` where applicable; if a sequence limit is smaller than the default synthetic scenario, the config raises that limit so the shipped workload can run.
+- vLLM uses framework-native `vllm serve` flags. The translation keeps the same model, tokenizer, dataset shape, GPU count, and high-impact batching/prefix-cache knobs; it does not copy SGLang-only parser or scheduler flags.
+- TensorRT-LLM uses `trtllm-serve serve` with `backend: pytorch` fixed in `base_server_flags`. Backend choice is never searched.
+- The two default random scenarios remain aligned pairs: `chat` uses `1000 -> 1000`, and `summarization` uses `8000 -> 1000`.
+
+Before a real run, capture the target framework `--help` output and validate the configs:
+
+```bash
```
- 已读文件:
  - docs/bench: `skills/llm-serving-auto-benchmark/configs/cookbook-llm/README.md` added +17/-0; `skills/llm-serving-auto-benchmark/configs/cookbook-llm/deepseek-math-v2.yaml` added +130/-0; `skills/llm-serving-auto-benchmark/configs/cookbook-llm/deepseek-r1-0528.yaml` added +133/-0; `skills/llm-serving-auto-benchmark/configs/cookbook-llm/deepseek-v3.1.yaml` added +132/-0; `skills/llm-serving-auto-benchmark/configs/cookbook-llm/deepseek-v3.2.yaml` added +132/-0; `skills/llm-serving-auto-benchmark/configs/cookbook-llm/deepseek-v3.yaml` added +133/-0; `skills/llm-serving-auto-benchmark/configs/cookbook-llm/devstral-small-2-24b-instruct-2512.yaml` added +123/-0; `skills/llm-serving-auto-benchmark/configs/cookbook-llm/ernie-4.5-21b-a3b-pt.yaml` added +117/-0
  - other: `agent-skills/llm-serving-auto-benchmark/SKILL.md` added +527/-0
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。
