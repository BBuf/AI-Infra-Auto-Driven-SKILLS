# vllm Llama 3.1 模型 PR 优化历史

## 2026-06-26 最新源码扫描

已按 vLLM 上游 `vllm-project/vllm@abc71548ef029132c3316b902207f254a246d593` 重新扫描本文下方列出的 tracked files。
文件级匹配使用 GitHub mirror 的 `git log --name-only`；PR 标题、链接和合并时间通过 GitHub GraphQL Pull Request API 批量复核。上一时效锚点：`2026-06-05`。

结果：发现 1 个额外 PR-numbered merge 触及 tracked files，但尚未提升为下方完整逐 PR diff audit card。此节只作为 freshness index；需要引用实现细节时，仍应先人工阅读 PR diff 再补完整卡片。

| 合并日期 | PR | 标题 | 命中的 tracked files |
| --- | --- | --- | --- |
| 2026-06-11 | [#44992](https://github.com/vllm-project/vllm/pull/44992) | Deprecations for v0.23 and v0.24 | `test_async_tp.py` |

## 2026-06-05 PR 补漏复核

已于 2026-06-05 按 vllm 上游 `origin/main@c66b19800` 复核；自上次时效基准（2026-05-19）以来，共有 3 个带 PR 编号的合并改动到所跟踪的实现文件，这些 PR 尚未并入下方时间线 / 逐 PR diff 审计卡，应在下次完整重生成时补齐。

| 合并日期 | PR | 标题 | 改动到的跟踪文件 |
| --- | --- | --- | --- |
| 2026-06-03 | [#44128](https://github.com/vllm-project/vllm/pull/44128) | [Misc] Remove dead VLLM_RPC_TIMEOUT env var and fix profiling doc that references it | `serving-tests-cpu-text.json` |
| 2026-05-23 | [#43233](https://github.com/vllm-project/vllm/pull/43233) | [Model Runner v2] Force v1 runner for tests | `test_async_tp.py`, `test_sequence_parallel.py` |
| 2026-05-21 | [#43262](https://github.com/vllm-project/vllm/pull/43262) | update GPU json file based on h200 recipes | `serving-tests.json` |


## 2026-05-19 新增覆盖

按 vllm 上游 `origin/main@ef54a4d604`、模型相关文件的 `git log --name-only -- <model-files>` 以及 GitHub Pull Request files API 生成。本页用于补齐 sgl-cookbook 中 `Llama 3.1` 缺失的历史 PR 优化文档。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `examples/tool_chat_template_llama3.1_json.jinja` | [#10164](https://github.com/vllm-project/vllm/pull/10164), [#8343](https://github.com/vllm-project/vllm/pull/8343) |
| `.buildkite/performance-benchmarks/tests/serving-tests.json` | [#38576](https://github.com/vllm-project/vllm/pull/38576), [#36216](https://github.com/vllm-project/vllm/pull/36216), [#25786](https://github.com/vllm-project/vllm/pull/25786) |
| `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json` | [#42607](https://github.com/vllm-project/vllm/pull/42607), [#38576](https://github.com/vllm-project/vllm/pull/38576), [#35086](https://github.com/vllm-project/vllm/pull/35086), [#34128](https://github.com/vllm-project/vllm/pull/34128) |
| `tests/compile/correctness_e2e/test_sequence_parallel.py` | [#42197](https://github.com/vllm-project/vllm/pull/42197), [#33322](https://github.com/vllm-project/vllm/pull/33322), [#41882](https://github.com/vllm-project/vllm/pull/41882), [#38373](https://github.com/vllm-project/vllm/pull/38373), [#34716](https://github.com/vllm-project/vllm/pull/34716), [#33731](https://github.com/vllm-project/vllm/pull/33731) |
| `tests/compile/correctness_e2e/test_async_tp.py` | [#41882](https://github.com/vllm-project/vllm/pull/41882), [#35871](https://github.com/vllm-project/vllm/pull/35871), [#33731](https://github.com/vllm-project/vllm/pull/33731) |

## PR 覆盖总览

- git 追溯 PR 数: 15
- 关键词/补充 PR 数: 0
- 当前文档总 PR 数: 15
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2024-09-27 | [#8343](https://github.com/vllm-project/vllm/pull/8343) | merged | [Feature] Add support for Llama 3.1 and 3.2 tool use | `docs/source/serving/openai_compatible_server.md`, `examples/tool_chat_template_llama3.1_json.jinja`, `examples/tool_chat_template_llama3.2_json.jinja` |
| 2024-11-23 | [#10164](https://github.com/vllm-project/vllm/pull/10164) | merged | [Bugfix][Frontend] Update Llama Chat Templates to also support Non-Tool use | `examples/tool_chat_template_llama3.1_json.jinja`, `examples/tool_chat_template_llama3.2_json.jinja`, `tests/entrypoints/test_chat_utils.py` |
| 2025-10-30 | [#25786](https://github.com/vllm-project/vllm/pull/25786) | merged | [Benchmark] Cleanup deprecated nightly benchmark and adjust the docstring for performance benchmark | `.buildkite/nightly-benchmarks/benchmark-pipeline.yaml`, `.buildkite/nightly-benchmarks/nightly-annotation.md`, `.buildkite/nightly-benchmarks/nightly-descriptions.md` |
| 2026-02-06 | [#33731](https://github.com/vllm-project/vllm/pull/33731) | merged | [torch.compile] Reorganize vllm/compilation and tests/compile (0/N for vLLM IR) | `.buildkite/test-amd.yaml`, `.buildkite/test-pipeline.yaml`, `.buildkite/test_areas/compile.yaml` |
| 2026-02-12 | [#34128](https://github.com/vllm-project/vllm/pull/34128) | merged | Vllm CPU benchmark suite improvement | `.buildkite/performance-benchmarks/scripts/compare-json-results.py`, `.buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu-embed.json` |
| 2026-02-17 | [#34716](https://github.com/vllm-project/vllm/pull/34716) | merged | [BugFix] Fix sp tests | `tests/compile/correctness_e2e/test_sequence_parallel.py` |
| 2026-03-04 | [#35871](https://github.com/vllm-project/vllm/pull/35871) | merged | [CI] Add Blackwell AsyncTP correctness test | `.buildkite/test_areas/compile.yaml`, `tests/compile/correctness_e2e/test_async_tp.py` |
| 2026-03-07 | [#36216](https://github.com/vllm-project/vllm/pull/36216) | merged | [V0 Deprecation] Remove unused swap_space parameter | `.buildkite/performance-benchmarks/README.md`, `.buildkite/performance-benchmarks/tests/serving-tests-hpu.json`, `.buildkite/performance-benchmarks/tests/serving-tests.json` |
| 2026-03-12 | [#35086](https://github.com/vllm-project/vllm/pull/35086) | merged | more models for vLLM Benchmark Suite | `.buildkite/performance-benchmarks/scripts/compare-json-results.py`, `.buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu-asr.json` |
| 2026-03-31 | [#38576](https://github.com/vllm-project/vllm/pull/38576) | merged | vLLM Benchmark Suite perf regression after PR#32723 | `.buildkite/performance-benchmarks/tests/serving-tests-arm64-cpu.json`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu-asr.json`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json` |
| 2026-04-26 | [#38373](https://github.com/vllm-project/vllm/pull/38373) | merged | [torch.compile]: Disable Sequence Parallelism (SP) for piecewise compilation | `tests/compile/correctness_e2e/test_sequence_parallel.py`, `tests/compile/passes/distributed/test_async_tp.py`, `tests/compile/passes/distributed/test_sequence_parallelism.py` |
| 2026-05-10 | [#33322](https://github.com/vllm-project/vllm/pull/33322) | merged | [Bugfix] Fix SP pass for multimodal models and PP+SP residual handling | `tests/compile/correctness_e2e/test_sequence_parallel.py`, `vllm/compilation/passes/fusion/sequence_parallelism.py`, `vllm/v1/worker/gpu_model_runner.py` |
| 2026-05-10 | [#41882](https://github.com/vllm-project/vllm/pull/41882) | merged | Add NVFP4 all-gather GEMM fusion for AsyncTP | `tests/compile/correctness_e2e/test_async_tp.py`, `tests/compile/correctness_e2e/test_sequence_parallel.py`, `tests/compile/fullgraph/test_toy_llama.py` |
| 2026-05-10 | [#42197](https://github.com/vllm-project/vllm/pull/42197) | merged | Fix mypy failure on main | `tests/compile/correctness_e2e/test_sequence_parallel.py` |
| 2026-05-15 | [#42607](https://github.com/vllm-project/vllm/pull/42607) | merged | Update Intel Xeon model list and vLLM Benchmark Suite BKMs | `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json`, `docs/models/hardware_supported_models/cpu.md` |

## 逐 PR diff 审计卡

### PR #8343 - [Feature] Add support for Llama 3.1 and 3.2 tool use

- 链接: https://github.com/vllm-project/vllm/pull/8343
- 状态/时间: merged / 2024-09-27
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+576/-27，可读 patch 741 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Feature] Add support for Llama 3.1 and 3.2 tool use」；模型线: Llama 3.1；类别: 模型支持/运行时入口；主要 diff: `docs/source/serving/openai_compatible_server.md`, `examples/tool_chat_template_llama3.1_json.jinja`, `examples/tool_chat_template_llama3.2_json.jinja`。
- 实现要点:
  - `docs/source/serving/openai_compatible_server.md` modified +24/-2
  - `examples/tool_chat_template_llama3.1_json.jinja` added +94/-0
  - `examples/tool_chat_template_llama3.2_json.jinja` added +93/-0
  - `tests/tool_use/test_chat_completions.py` modified +10/-7；symbols: test_chat_completion_without_tools, test_chat_completion_with_tools
- 代码 diff 细节:
  - `docs/source/serving/openai_compatible_server.md` modified +24/-2
  - `examples/tool_chat_template_llama3.1_json.jinja` added +94/-0
  - `examples/tool_chat_template_llama3.2_json.jinja` added +93/-0
  - `tests/tool_use/test_chat_completions.py` modified +10/-7
- 关键代码摘录:

```diff
diff -- docs/source/serving/openai_compatible_server.md
@@ -157,10 +157,10 @@ vLLM will use guided decoding to ensure the response matches the tool parameter
 To enable this feature, you should set the following flags:
 * `--enable-auto-tool-choice` -- **mandatory** Auto tool choice. tells vLLM that you want to enable the model to generate its own tool calls when it
 deems appropriate.
-* `--tool-call-parser` -- select the tool parser to use - currently either `hermes` or `mistral`. Additional tool parsers
+* `--tool-call-parser` -- select the tool parser to use - currently either `hermes`, `mistral` or `llama3_json`. Additional tool parsers
 will continue to be added in the future.
 * `--chat-template` -- **optional** for auto tool choice. the path to the chat template which handles `tool`-role messages and `assistant`-role messages
-that contain previously generated tool calls. Hermes and Mistral models have tool-compatible chat templates in their
+that contain previously generated tool calls. Hermes, Mistral and Llama models have tool-compatible chat templates in their
 `tokenizer_config.json` files, but you can specify a custom template. This argument can be set to `tool_use` if your model has a tool use-specific chat
 template configured in the `tokenizer_config.json`. In this case, it will be used per the `transformers` specification. More on this [here](https://huggingface.co/docs/transformers/en/chat_templating#why-do-some-models-have-multiple-templates)
 from HuggingFace; and you can find an example of this in a `tokenizer_config.json` [here](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B/blob/main/tokenizer_config.json)
@@ -197,3 +197,25 @@ when tools are provided, that results in much better reliability when working wi
diff -- examples/tool_chat_template_llama3.1_json.jinja
@@ -0,0 +1,94 @@
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
+        {%- set date_string = "26 Jul 2024" %}
```
- 已读文件:
  - runtime: `vllm/entrypoints/openai/cli_args.py` modified +1/-1; `vllm/entrypoints/openai/serving_chat.py` modified +3/-0; `vllm/entrypoints/openai/tool_parsers/__init__.py` modified +5/-1; `vllm/entrypoints/openai/tool_parsers/llama_tool_parser.py` added +273/-0
  - tests: `tests/tool_use/test_chat_completions.py` modified +10/-7; `tests/tool_use/test_parallel_tool_calls.py` modified +15/-3; `tests/tool_use/utils.py` modified +58/-13
  - docs/bench: `docs/source/serving/openai_compatible_server.md` modified +24/-2; `examples/tool_chat_template_llama3.1_json.jinja` added +94/-0; `examples/tool_chat_template_llama3.2_json.jinja` added +93/-0
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #10164 - [Bugfix][Frontend] Update Llama Chat Templates to also support Non-Tool use

- 链接: https://github.com/vllm-project/vllm/pull/10164
- 状态/时间: merged / 2024-11-23
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+110/-36，可读 patch 240 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix][Frontend] Update Llama Chat Templates to also support Non-Tool use」；模型线: Llama 3.1；类别: 缺陷修复；主要 diff: `examples/tool_chat_template_llama3.1_json.jinja`, `examples/tool_chat_template_llama3.2_json.jinja`, `tests/entrypoints/test_chat_utils.py`。
- 实现要点:
  - `examples/tool_chat_template_llama3.1_json.jinja` modified +36/-10
  - `examples/tool_chat_template_llama3.2_json.jinja` modified +72/-24
  - `tests/entrypoints/test_chat_utils.py` modified +2/-2
- 代码 diff 细节:
  - `examples/tool_chat_template_llama3.1_json.jinja` modified +36/-10
  - `examples/tool_chat_template_llama3.2_json.jinja` modified +72/-24
  - `tests/entrypoints/test_chat_utils.py` modified +2/-2
- 关键代码摘录:

```diff
diff -- examples/tool_chat_template_llama3.1_json.jinja
@@ -19,10 +19,18 @@

 {#- This block extracts the system message, so we can slot it into the right place. #}
 {%- if messages[0]['role'] == 'system' %}
-    {%- set system_message = messages[0]['content']|trim %}
+    {%- if messages[0]['content'] is string %}
+        {%- set system_message = messages[0]['content']|trim %}
+    {%- else %}
+        {%- set system_message = messages[0]['content'][0]['text']|trim %}
+    {%- endif %}
     {%- set messages = messages[1:] %}
 {%- else %}
-    {%- set system_message = "You are a helpful assistant with tool calling capabilities. Only reply with a tool call if the function exists in the library provided by the user. If it doesn't exist, just reply directly in natural language. When you receive a tool call response, use the output to format an answer to the original user question." %}
+    {%- if tools is not none %}
diff -- examples/tool_chat_template_llama3.2_json.jinja
@@ -16,46 +16,78 @@
     {%- set tools = none %}
 {%- endif %}

+{#- Find out if there are any images #}
+{% set image_ns = namespace(has_images=false) %}
+{%- for message in messages %}
+    {%- for content in message['content'] %}
+        {%- if content['type'] == 'image' %}
+            {%- set image_ns.has_images = true %}
+        {%- endif %}
+    {%- endfor %}
+{%- endfor %}
+
```
- 已读文件:
  - tests: `tests/entrypoints/test_chat_utils.py` modified +2/-2
  - docs/bench: `examples/tool_chat_template_llama3.1_json.jinja` modified +36/-10; `examples/tool_chat_template_llama3.2_json.jinja` modified +72/-24
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #25786 - [Benchmark] Cleanup deprecated nightly benchmark and adjust the docstring for performance benchmark

- 链接: https://github.com/vllm-project/vllm/pull/25786
- 状态/时间: merged / 2025-10-30
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 29 个文件，+10/-1289，可读 patch 1387 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Benchmark] Cleanup deprecated nightly benchmark and adjust the docstring for performance benchmark」；模型线: Llama 3.1；类别: 性能/后端优化；主要 diff: `.buildkite/nightly-benchmarks/benchmark-pipeline.yaml`, `.buildkite/nightly-benchmarks/nightly-annotation.md`, `.buildkite/nightly-benchmarks/nightly-descriptions.md`。
- 实现要点:
  - `.buildkite/nightly-benchmarks/benchmark-pipeline.yaml` removed +0/-184
  - `.buildkite/nightly-benchmarks/nightly-annotation.md` removed +0/-28
  - `.buildkite/nightly-benchmarks/nightly-descriptions.md` removed +0/-39
  - `.buildkite/nightly-benchmarks/nightly-pipeline.yaml` removed +0/-196
- 代码 diff 细节:
  - `.buildkite/nightly-benchmarks/benchmark-pipeline.yaml` removed +0/-184
  - `.buildkite/nightly-benchmarks/nightly-annotation.md` removed +0/-28
  - `.buildkite/nightly-benchmarks/nightly-descriptions.md` removed +0/-39
  - `.buildkite/nightly-benchmarks/nightly-pipeline.yaml` removed +0/-196
- 关键代码摘录:

```diff
diff -- .buildkite/nightly-benchmarks/benchmark-pipeline.yaml
@@ -1,184 +0,0 @@
-steps:
-  - label: "Wait for container to be ready"
-    key: wait-for-container-image
-    agents:
-      queue: A100
-    plugins:
-    - kubernetes:
-        podSpec:
-          containers:
-          - image: badouralix/curl-jq
-            command:
-            - sh .buildkite/nightly-benchmarks/scripts/wait-for-image.sh
-  - label: "Cleanup H100"
diff -- .buildkite/nightly-benchmarks/nightly-annotation.md
@@ -1,28 +0,0 @@
-# Nightly benchmark annotation
-
-## Description
-
-This file contains the downloading link for benchmarking results.
-
-- [benchmarking pipeline](artifact://nightly-pipeline.yaml)
-- [benchmarking results](artifact://results.zip)
-- [benchmarking code](artifact://nightly-benchmarks.zip)
-
-Please download the visualization scripts in the post
-
-## Results reproduction
```
- 已读文件:
  - docs/bench: `.buildkite/nightly-benchmarks/benchmark-pipeline.yaml` removed +0/-184; `.buildkite/nightly-benchmarks/nightly-annotation.md` removed +0/-28; `.buildkite/nightly-benchmarks/nightly-descriptions.md` removed +0/-39; `.buildkite/nightly-benchmarks/nightly-pipeline.yaml` removed +0/-196; `.buildkite/nightly-benchmarks/scripts/download-tokenizer.py` removed +0/-26; `.buildkite/nightly-benchmarks/scripts/generate-nightly-markdown.py` removed +0/-97; `.buildkite/nightly-benchmarks/scripts/get-lmdeploy-modelname.py` removed +0/-9; `.buildkite/nightly-benchmarks/scripts/nightly-annotate.sh` removed +0/-78
  - other: `.github/mergify.yml` modified +1/-1
- 验证与风险: diff 未直接暴露测试文件，后续改同一模型时应补最小 launch、tokenizer/MM processor 或 accuracy smoke。

### PR #33731 - [torch.compile] Reorganize vllm/compilation and tests/compile (0/N for vLLM IR)

- 链接: https://github.com/vllm-project/vllm/pull/33731
- 状态/时间: merged / 2026-02-06
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 47 个文件，+717/-651，可读 patch 1985 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[torch.compile] Reorganize vllm/compilation and tests/compile (0/N for vLLM IR)」；模型线: Llama 3.1；类别: 文档/测试/CI；主要 diff: `.buildkite/test-amd.yaml`, `.buildkite/test-pipeline.yaml`, `.buildkite/test_areas/compile.yaml`。
- 实现要点:
  - `.buildkite/test-amd.yaml` modified +19/-19
  - `.buildkite/test-pipeline.yaml` modified +6/-5
  - `.buildkite/test_areas/compile.yaml` modified +24/-18
  - `.buildkite/test_areas/pytorch.yaml` modified +9/-1
- 代码 diff 细节:
  - `.buildkite/test-amd.yaml` modified +19/-19
  - `.buildkite/test-pipeline.yaml` modified +6/-5
  - `.buildkite/test_areas/compile.yaml` modified +24/-18
  - `.buildkite/test_areas/pytorch.yaml` modified +9/-1
- 关键代码摘录:

```diff
diff -- .buildkite/test-amd.yaml
@@ -544,7 +544,7 @@ steps:
 - label: LoRA Test %N # 20min each
   timeout_in_minutes: 30
   mirror_hardwares: [amdexperimental]
-  agent_pool: mi325_1
+  agent_pool: mi325_8
   # grade: Blocking
   source_file_dependencies:
   - vllm/lora
@@ -640,7 +640,7 @@ steps:
 - label: Kernels Attention Test %N # 23min
   timeout_in_minutes: 35
   mirror_hardwares: [amdexperimental, amdproduction]
-  agent_pool: mi325_1
diff -- .buildkite/test-pipeline.yaml
@@ -512,6 +512,7 @@ steps:
   # However, find does not normally propagate error codes, so we combine it with xargs
   # (using -0 for proper path handling)
   - "find compile/ -maxdepth 1 -name 'test_*.py' -print0 | xargs -0 -n1 -I{} pytest -s -v '{}'"
+  - pytest -s -v compile/passes --ignore compile/passes/distributed

 - label: PyTorch Fullgraph Smoke Test # 15min
   timeout_in_minutes: 30
@@ -1072,14 +1073,14 @@ steps:
   - vllm/model_executor/layers/quantization/input_quant_fp8.py
   - tests/compile/test_fusion_attn.py
   - tests/compile/test_silu_mul_quant_fusion.py
-  - tests/compile/distributed/test_fusion_all_reduce.py
+  - tests/compile/passes/distributed/test_fusion_all_reduce.py
```
- 已读文件:
  - runtime: `vllm/compilation/backends.py` modified +6/-7; `vllm/compilation/passes/__init__.py` added +0/-0; `vllm/compilation/passes/fusion/__init__.py` added +0/-0; `vllm/compilation/passes/fusion/act_quant_fusion.py` renamed +3/-3; `vllm/compilation/passes/fusion/allreduce_rms_fusion.py` renamed +13/-403
  - tests: `tests/compile/backend.py` modified +4/-4; `tests/compile/correctness_e2e/__init__.py` renamed +0/-0; `tests/compile/correctness_e2e/test_async_tp.py` added +79/-0; `tests/compile/correctness_e2e/test_sequence_parallel.py` renamed +2/-2; `tests/compile/fusions_e2e/common.py` modified +6/-8; `tests/compile/passes/__init__.py` added +0/-0; `tests/compile/passes/distributed/__init__.py` added +0/-0; `tests/compile/passes/distributed/test_async_tp.py` renamed +5/-75
  - docs/bench: `.buildkite/test-amd.yaml` modified +19/-19; `.buildkite/test-pipeline.yaml` modified +6/-5; `.buildkite/test_areas/compile.yaml` modified +24/-18; `.buildkite/test_areas/pytorch.yaml` modified +9/-1
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #34128 - Vllm CPU benchmark suite improvement

- 链接: https://github.com/vllm-project/vllm/pull/34128
- 状态/时间: merged / 2026-02-12
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+802/-254，可读 patch 1243 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Vllm CPU benchmark suite improvement」；模型线: Llama 3.1；类别: 文档/测试/CI；主要 diff: `.buildkite/performance-benchmarks/scripts/compare-json-results.py`, `.buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu-embed.json`。
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

### PR #34716 - [BugFix] Fix sp tests

- 链接: https://github.com/vllm-project/vllm/pull/34716
- 状态/时间: merged / 2026-02-17
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[BugFix] Fix sp tests」；模型线: Llama 3.1；类别: 缺陷修复；主要 diff: `tests/compile/correctness_e2e/test_sequence_parallel.py`。
- 实现要点:
  - `tests/compile/correctness_e2e/test_sequence_parallel.py` modified +1/-1
- 代码 diff 细节:
  - `tests/compile/correctness_e2e/test_sequence_parallel.py` modified +1/-1
- 关键代码摘录:

```diff
diff -- tests/compile/correctness_e2e/test_sequence_parallel.py
@@ -229,7 +229,7 @@ def _compare_sp(
     if chunked_prefill:
         common_args.append("--enable-chunked-prefill")
     if eager_mode:
-        common_args.append("--enforce-eager")
+        common_args.append("-cc.cudagraph_mode=none")
     if runner != "auto":
         common_args.extend(["--runner", runner])
     if trust_remote_code:
```
- 已读文件:
  - tests: `tests/compile/correctness_e2e/test_sequence_parallel.py` modified +1/-1
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #35871 - [CI] Add Blackwell AsyncTP correctness test

- 链接: https://github.com/vllm-project/vllm/pull/35871
- 状态/时间: merged / 2026-03-04
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+15/-0，可读 patch 30 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI] Add Blackwell AsyncTP correctness test」；模型线: Llama 3.1；类别: 模型支持/运行时入口；主要 diff: `.buildkite/test_areas/compile.yaml`, `tests/compile/correctness_e2e/test_async_tp.py`。
- 实现要点:
  - `.buildkite/test_areas/compile.yaml` modified +10/-0
  - `tests/compile/correctness_e2e/test_async_tp.py` modified +5/-0
- 代码 diff 细节:
  - `.buildkite/test_areas/compile.yaml` modified +10/-0
  - `tests/compile/correctness_e2e/test_async_tp.py` modified +5/-0
- 关键代码摘录:

```diff
diff -- .buildkite/test_areas/compile.yaml
@@ -36,6 +36,16 @@ steps:
   - export VLLM_TEST_CLEAN_GPU_MEMORY=1
   - pytest -v -s tests/compile/correctness_e2e/test_async_tp.py

+- label: AsyncTP Correctness Tests (B200)
+  timeout_in_minutes: 50
+  working_dir: "/vllm-workspace/"
+  device: b200
+  optional: true
+  num_devices: 2
+  commands:
+  - export VLLM_TEST_CLEAN_GPU_MEMORY=1
+  - pytest -v -s tests/compile/correctness_e2e/test_async_tp.py
+
diff -- tests/compile/correctness_e2e/test_async_tp.py
@@ -31,7 +31,12 @@ def test_async_tp_pass_correctness(
     distributed_backend: str,
     eager_mode: bool,
     num_gpus_available: int,
+    monkeypatch,
 ):
+    # Disable FlashInfer FP8 scaled_mm kernel as it is incompatible with
+    # async TP patterns. No-op on H100 (kernel requires CC >= 100).
+    monkeypatch.setenv("VLLM_DISABLED_KERNELS", "FlashInferFP8ScaledMMLinearKernel")
+
     model_info = HF_EXAMPLE_MODELS.find_hf_info(model_id)
     model_info.check_transformers_version(on_fail="skip")
     model_info.check_available_online(on_fail="skip")
```
- 已读文件:
  - tests: `tests/compile/correctness_e2e/test_async_tp.py` modified +5/-0
  - docs/bench: `.buildkite/test_areas/compile.yaml` modified +10/-0
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #36216 - [V0 Deprecation] Remove unused swap_space parameter

- 链接: https://github.com/vllm-project/vllm/pull/36216
- 状态/时间: merged / 2026-03-07
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 22 个文件，+19/-79，可读 patch 395 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[V0 Deprecation] Remove unused swap_space parameter」；模型线: Llama 3.1；类别: 文档/测试/CI；主要 diff: `.buildkite/performance-benchmarks/README.md`, `.buildkite/performance-benchmarks/tests/serving-tests-hpu.json`, `.buildkite/performance-benchmarks/tests/serving-tests.json`。
- 实现要点:
  - `.buildkite/performance-benchmarks/README.md` modified +0/-1
  - `.buildkite/performance-benchmarks/tests/serving-tests-hpu.json` modified +0/-4
  - `.buildkite/performance-benchmarks/tests/serving-tests.json` modified +0/-4
  - `benchmarks/attention_benchmarks/mla_runner.py` modified +0/-1
- 代码 diff 细节:
  - `.buildkite/performance-benchmarks/README.md` modified +0/-1
  - `.buildkite/performance-benchmarks/tests/serving-tests-hpu.json` modified +0/-4
  - `.buildkite/performance-benchmarks/tests/serving-tests.json` modified +0/-4
  - `benchmarks/attention_benchmarks/mla_runner.py` modified +0/-1
- 关键代码摘录:

```diff
diff -- .buildkite/performance-benchmarks/README.md
@@ -83,7 +83,6 @@ We test the throughput by using `vllm bench serve` with request rate = inf to co
         "server_parameters": {
             "model": "meta-llama/Meta-Llama-3-8B",
             "tensor_parallel_size": 1,
-            "swap_space": 16,
             "disable_log_stats": "",
             "load_format": "dummy"
         },
diff -- .buildkite/performance-benchmarks/tests/serving-tests-hpu.json
@@ -10,7 +10,6 @@
         "server_parameters": {
             "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
             "tensor_parallel_size": 1,
-            "swap_space": 16,
             "disable_log_stats": "",
             "load_format": "dummy",
             "max-model-len": 2048,
@@ -37,7 +36,6 @@
         "server_parameters": {
             "model": "meta-llama/Meta-Llama-3.1-70B-Instruct",
             "tensor_parallel_size": 4,
-            "swap_space": 16,
             "disable_log_stats": "",
```
- 已读文件:
  - runtime: `vllm/config/cache.py` modified +1/-33; `vllm/config/vllm.py` modified +0/-2; `vllm/engine/arg_utils.py` modified +0/-3; `vllm/entrypoints/llm.py` modified +11/-8
  - tests: `tests/conftest.py` modified +0/-2; `tests/distributed/test_torchrun_example.py` modified +1/-2; `tests/distributed/test_torchrun_example_moe.py` modified +1/-2; `tests/lora/test_worker.py` modified +0/-1; `tests/v1/attention/utils.py` modified +0/-1; `tests/v1/core/test_scheduler.py` modified +0/-2; `tests/v1/core/utils.py` modified +0/-1; `tests/v1/engine/test_engine_core.py` modified +0/-1
  - docs/bench: `.buildkite/performance-benchmarks/README.md` modified +0/-1; `.buildkite/performance-benchmarks/tests/serving-tests-hpu.json` modified +0/-4; `.buildkite/performance-benchmarks/tests/serving-tests.json` modified +0/-4; `benchmarks/attention_benchmarks/mla_runner.py` modified +0/-1; `benchmarks/attention_benchmarks/runner.py` modified +0/-1; `docs/design/metrics.md` modified +4/-4; `docs/serving/integrations/llamaindex.md` modified +1/-1
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #35086 - more models for vLLM Benchmark Suite

- 链接: https://github.com/vllm-project/vllm/pull/35086
- 状态/时间: merged / 2026-03-12
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+800/-119，可读 patch 1301 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「more models for vLLM Benchmark Suite」；模型线: Llama 3.1；类别: 文档/测试/CI；主要 diff: `.buildkite/performance-benchmarks/scripts/compare-json-results.py`, `.buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu-asr.json`。
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
- 动机: 标题「vLLM Benchmark Suite perf regression after PR#32723」；模型线: Llama 3.1；类别: 性能/后端优化；主要 diff: `.buildkite/performance-benchmarks/tests/serving-tests-arm64-cpu.json`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu-asr.json`, `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json`。
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

### PR #38373 - [torch.compile]: Disable Sequence Parallelism (SP) for piecewise compilation

- 链接: https://github.com/vllm-project/vllm/pull/38373
- 状态/时间: merged / 2026-04-26
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+223/-80，可读 patch 450 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[torch.compile]: Disable Sequence Parallelism (SP) for piecewise compilation」；模型线: Llama 3.1；类别: 文档/测试/CI；主要 diff: `tests/compile/correctness_e2e/test_sequence_parallel.py`, `tests/compile/passes/distributed/test_async_tp.py`, `tests/compile/passes/distributed/test_sequence_parallelism.py`。
- 实现要点:
  - `tests/compile/correctness_e2e/test_sequence_parallel.py` modified +2/-0
  - `tests/compile/passes/distributed/test_async_tp.py` modified +17/-0；symbols: test_async_tp_pass_requires_full_graph_compilation
  - `tests/compile/passes/distributed/test_sequence_parallelism.py` modified +19/-0；symbols: test_sequence_parallelism_pass_requires_full_graph_compilation
  - `tests/compile/test_config.py` modified +118/-1；symbols: test_sequence_parallelism_requires_full_graph_compilation
- 代码 diff 细节:
  - `tests/compile/correctness_e2e/test_sequence_parallel.py` modified +2/-0
  - `tests/compile/passes/distributed/test_async_tp.py` modified +17/-0
  - `tests/compile/passes/distributed/test_sequence_parallelism.py` modified +19/-0
  - `tests/compile/test_config.py` modified +118/-1
- 关键代码摘录:

```diff
diff -- tests/compile/correctness_e2e/test_sequence_parallel.py
@@ -261,6 +261,8 @@ def _compare_sp(
         },
         "use_inductor_graph_partition": use_inductor_graph_partition,
     }
+    if not use_inductor_graph_partition:
+        compilation_config["splitting_ops"] = []

     tp_sp_args = [
         *common_args,
diff -- tests/compile/passes/distributed/test_async_tp.py
@@ -19,6 +19,7 @@
     VllmConfig,
     set_current_vllm_config,
 )
+from vllm.config.utils import Range
 from vllm.distributed import (
     tensor_model_parallel_all_gather,
     tensor_model_parallel_reduce_scatter,
@@ -288,6 +289,22 @@ def run_torch_spawn(fn, nprocs):
     run_torch_spawn(async_tp_pass_on_test_model, num_processes)


+def test_async_tp_pass_requires_full_graph_compilation():
+    vllm_config = VllmConfig()
```
- 已读文件:
  - runtime: `vllm/compilation/passes/fusion/collective_fusion.py` modified +7/-10; `vllm/compilation/passes/fusion/sequence_parallelism.py` modified +16/-26; `vllm/config/compilation.py` modified +19/-0; `vllm/config/vllm.py` modified +17/-28; `vllm/v1/worker/utils.py` modified +8/-15
  - tests: `tests/compile/correctness_e2e/test_sequence_parallel.py` modified +2/-0; `tests/compile/passes/distributed/test_async_tp.py` modified +17/-0; `tests/compile/passes/distributed/test_sequence_parallelism.py` modified +19/-0; `tests/compile/test_config.py` modified +118/-1
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #33322 - [Bugfix] Fix SP pass for multimodal models and PP+SP residual handling

- 链接: https://github.com/vllm-project/vllm/pull/33322
- 状态/时间: merged / 2026-05-10
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+116/-34，可读 patch 260 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix SP pass for multimodal models and PP+SP residual handling」；模型线: Llama 3.1；类别: 缺陷修复；主要 diff: `tests/compile/correctness_e2e/test_sequence_parallel.py`, `vllm/compilation/passes/fusion/sequence_parallelism.py`, `vllm/v1/worker/gpu_model_runner.py`。
- 实现要点:
  - `tests/compile/correctness_e2e/test_sequence_parallel.py` modified +48/-0；symbols: test_tp_sp_generation_prompt_embeds
  - `vllm/compilation/passes/fusion/sequence_parallelism.py` modified +55/-20
  - `vllm/v1/worker/gpu_model_runner.py` modified +13/-14；symbols: sync_and_gather_intermediate_tensors
- 代码 diff 细节:
  - `tests/compile/correctness_e2e/test_sequence_parallel.py` modified +48/-0
  - `vllm/compilation/passes/fusion/sequence_parallelism.py` modified +55/-20
  - `vllm/v1/worker/gpu_model_runner.py` modified +13/-14
- 关键代码摘录:

```diff
diff -- tests/compile/correctness_e2e/test_sequence_parallel.py
@@ -167,6 +167,7 @@ def _compare_sp(
     num_gpus_available: int,
     use_inductor_graph_partition: bool,
     fuse_gemm_comms: bool,
+    enable_prompt_embeds: bool,
     *,
     method: Literal["generate", "encode"],
     is_multimodal: bool,
@@ -248,6 +249,8 @@ def _compare_sp(
                 "--enable-mm-embeds",
             ]
         )
+    elif enable_prompt_embeds:
+        common_args.append("--enable-prompt-embeds")
diff -- vllm/compilation/passes/fusion/sequence_parallelism.py
@@ -14,7 +14,10 @@
 from vllm.config import VllmConfig
 from vllm.config.utils import Range
 from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce
-from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size
+from vllm.distributed.parallel_state import (
+    get_tensor_model_parallel_rank,
+    get_tensor_model_parallel_world_size,
+)
 from vllm.logger import init_logger
 from vllm.model_executor.layers.quantization.utils.quant_utils import (
     kFp8StaticTensorSym,
@@ -117,6 +120,7 @@ def __init__(
         self.device = device
```
- 已读文件:
  - runtime: `vllm/compilation/passes/fusion/sequence_parallelism.py` modified +55/-20; `vllm/v1/worker/gpu_model_runner.py` modified +13/-14
  - tests: `tests/compile/correctness_e2e/test_sequence_parallel.py` modified +48/-0
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #41882 - Add NVFP4 all-gather GEMM fusion for AsyncTP

- 链接: https://github.com/vllm-project/vllm/pull/41882
- 状态/时间: merged / 2026-05-10
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+605/-6，可读 patch 781 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add NVFP4 all-gather GEMM fusion for AsyncTP」；模型线: Llama 3.1；类别: 模型支持/运行时入口；主要 diff: `tests/compile/correctness_e2e/test_async_tp.py`, `tests/compile/correctness_e2e/test_sequence_parallel.py`, `tests/compile/fullgraph/test_toy_llama.py`。
- 实现要点:
  - `tests/compile/correctness_e2e/test_async_tp.py` modified +73/-0；symbols: test_async_tp_pass_nvfp4_correctness
  - `tests/compile/correctness_e2e/test_sequence_parallel.py` modified +44/-5；symbols: test_tp_sp_nvfp4_generation
  - `tests/compile/fullgraph/test_toy_llama.py` modified +2/-1
  - `tests/compile/fusions_e2e/test_tp2_async_tp.py` modified +65/-0；symbols: test_tp2_async_tp_nvfp4_fusions
- 代码 diff 细节:
  - `tests/compile/correctness_e2e/test_async_tp.py` modified +73/-0
  - `tests/compile/correctness_e2e/test_sequence_parallel.py` modified +44/-5
  - `tests/compile/fullgraph/test_toy_llama.py` modified +2/-1
  - `tests/compile/fusions_e2e/test_tp2_async_tp.py` modified +65/-0
- 关键代码摘录:

```diff
diff -- tests/compile/correctness_e2e/test_async_tp.py
@@ -13,6 +13,17 @@
 from vllm.config import (
     CompilationMode,
 )
+from vllm.platforms import current_platform
+from vllm.utils.flashinfer import has_flashinfer
+
+NVFP4_MODEL_ID = "nvidia/Llama-3.1-8B-Instruct-NVFP4"
+NVFP4_HF_OVERRIDES = {
+    "num_hidden_layers": 4,
+    "hidden_size": 512,
+    "intermediate_size": 800,
+    "num_attention_heads": 4,
+    "num_key_value_heads": 1,
diff -- tests/compile/correctness_e2e/test_sequence_parallel.py
@@ -21,12 +21,14 @@
 from vllm.platforms import current_platform
 from vllm.utils.torch_utils import is_torch_equal_or_newer

-from ...models.registry import HF_EXAMPLE_MODELS
+from ...models.registry import HF_EXAMPLE_MODELS, _HfExamplesInfo
 from ...utils import compare_two_settings, create_new_process_for_each_test

 logger = init_logger("test_sequence_parallel")

 VLLM_MULTI_NODE = os.getenv("VLLM_MULTI_NODE", "0") == "1"
+NVFP4_MODEL_ID = "nvidia/Llama-3.1-8B-Instruct-NVFP4"
+NVFP4_MODEL_INFO = _HfExamplesInfo(NVFP4_MODEL_ID)

```
- 已读文件:
  - runtime: `vllm/compilation/passes/fusion/collective_fusion.py` modified +243/-0; `vllm/compilation/passes/fusion/sequence_parallelism.py` modified +136/-0; `vllm/utils/flashinfer.py` modified +42/-0
  - tests: `tests/compile/correctness_e2e/test_async_tp.py` modified +73/-0; `tests/compile/correctness_e2e/test_sequence_parallel.py` modified +44/-5; `tests/compile/fullgraph/test_toy_llama.py` modified +2/-1; `tests/compile/fusions_e2e/test_tp2_async_tp.py` modified +65/-0
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #42197 - Fix mypy failure on main

- 链接: https://github.com/vllm-project/vllm/pull/42197
- 状态/时间: merged / 2026-05-10
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-0，可读 patch 8 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix mypy failure on main」；模型线: Llama 3.1；类别: 缺陷修复；主要 diff: `tests/compile/correctness_e2e/test_sequence_parallel.py`。
- 实现要点:
  - `tests/compile/correctness_e2e/test_sequence_parallel.py` modified +1/-0
- 代码 diff 细节:
  - `tests/compile/correctness_e2e/test_sequence_parallel.py` modified +1/-0
- 关键代码摘录:

```diff
diff -- tests/compile/correctness_e2e/test_sequence_parallel.py
@@ -435,6 +435,7 @@ def test_tp_sp_nvfp4_generation(num_gpus_available: int):
         num_gpus_available,
         use_inductor_graph_partition=False,
         fuse_gemm_comms=False,
+        enable_prompt_embeds=False,
         method="generate",
         is_multimodal=False,
         dtype="bfloat16",
```
- 已读文件:
  - tests: `tests/compile/correctness_e2e/test_sequence_parallel.py` modified +1/-0
- 验证与风险: diff 自带测试/基准路径，后续改同一模型优先复跑相关测试并补一个最小 launch/accuracy smoke。

### PR #42607 - Update Intel Xeon model list and vLLM Benchmark Suite BKMs

- 链接: https://github.com/vllm-project/vllm/pull/42607
- 状态/时间: merged / 2026-05-15
- 反查来源: `git log --name-only -- <model-files>` 或模型关键词补充；本卡按 GitHub Pull Request files API 审计。
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+118/-159，可读 patch 465 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Update Intel Xeon model list and vLLM Benchmark Suite BKMs」；模型线: Llama 3.1；类别: 文档/测试/CI；主要 diff: `.buildkite/performance-benchmarks/tests/serving-tests-cpu-text.json`, `docs/models/hardware_supported_models/cpu.md`。
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
