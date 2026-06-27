# sglang Llama 3.1 Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `docs_new/cookbook/autoregressive/Llama/Llama3.1.mdx` | no direct PR-number commit |
| `docs_new/src/snippets/autoregressive/llama31-deployment.jsx` | no direct PR-number commit |
| `examples/chat_template/tool_chat_template_llama3.1_json.jinja` | [#13935](https://github.com/sgl-project/sglang/pull/13935) |
| `test/manual/quant/kv_cache_scales_llama3_1_8b.json` | no direct PR-number commit |

## PR Coverage Summary

- Git-traced PRs: 1
- Extra PRs preserved from existing docs: 13
- Total PRs in this document: 13
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2025-11-20 | [#13610](https://github.com/sgl-project/sglang/pull/13610) | merged | Test reorganization: Move tests to manual/ | `test/manual/entrypoints/http_server/test_abort_request.py`, `test/manual/layers/attention/nsa/test_act_quant_triton.py`, `test/manual/layers/moe/test_moe_runners.py` |
| 2025-11-25 | [#13935](https://github.com/sgl-project/sglang/pull/13935) | merged | [misc] add llama3.1 chat template | `examples/chat_template/tool_chat_template_llama3.1_json.jinja` |
| 2025-11-25 | [#13938](https://github.com/sgl-project/sglang/pull/13938) | merged | [Minor] Fix lint | `examples/chat_template/tool_chat_template_llama3.1_json.jinja` |
| 2025-12-23 | [#15582](https://github.com/sgl-project/sglang/pull/15582) | merged | [CI] Migrate nightly tests to test/registered/ | `test/registered/ascend/llm_models/test_ascend_phi_4_multimodal.py`, `test/registered/ascend/vlm_models/test_ascend_phi4_multimodal_instruct.py`, `test/run_suite.py` |
| 2026-02-04 | [#17895](https://github.com/sgl-project/sglang/pull/17895) | merged | [AMD] Add kimi mi35x nightly test, folder organization and several stability fixes | `test/registered/amd/accuracy/mi35x/test_kimi_k2_eval_mi35x.py`, `.github/workflows/nightly-test-amd.yml`, `python/sglang/test/nightly_utils.py` |
| 2026-02-12 | [#17799](https://github.com/sgl-project/sglang/pull/17799) | merged | [AMD] rocm 7.2 image release, PR test, Nightly Test | `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/layers/moe/moe_runner/triton.py` |
| 2026-02-17 | [#18886](https://github.com/sgl-project/sglang/pull/18886) | merged | Fix eval tests not capturing server launch failures | `python/sglang/srt/model_loader/ci_weight_validation.py`, `test/registered/eval/test_text_models_gsm8k_eval.py`, `python/sglang/test/nightly_utils.py` |
| 2026-02-25 | [#18911](https://github.com/sgl-project/sglang/pull/18911) | merged | [AMD] [GLM-5 Day 0] Add GLM-5 nightly test | `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py`, `.github/workflows/nightly-test-amd-rocm720.yml` |
| 2026-03-07 | [#19778](https://github.com/sgl-project/sglang/pull/19778) | merged | Adding correct path for module not found error while collecting test | `test/manual/test_two_batch_overlap.py`, `test/manual/nightly/test_deepseek_v31_perf.py`, `test/manual/nightly/test_deepseek_v32_perf.py` |
| 2026-04-07 | [#21931](https://github.com/sgl-project/sglang/pull/21931) | merged | [CI] Migrate mgsm_en eval to gsm8k to remove openaipublic dependency | `test/registered/amd/accuracy/mi30x/test_gsm8k_eval_amd.py`, `test/registered/eval/test_text_models_gsm8k_eval.py`, `test/registered/piecewise_cuda_graph/test_piecewise_cuda_graph_support_1_gpu.py` |
| 2026-04-20 | [#23001](https://github.com/sgl-project/sglang/pull/23001) | merged | Add new Mintlify documentation site (docs_new/) | `docs_new/docs/advanced_features/tool_parser.mdx`, `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx`, `docs_new/docs/advanced_features/separate_reasoning.mdx` |
| 2026-04-21 | [#23337](https://github.com/sgl-project/sglang/pull/23337) | merged | [Docs] Sync docs_new with legacy docs and update migration redirects | `docs_new/docs/supported-models/multimodal_language_models.mdx`, `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx` |
| 2026-06-09 | [#27342](https://github.com/sgl-project/sglang/pull/27342) | merged | test: fix gemma GSM8K thresholds in nightly text eval | `test/registered/eval/test_text_models_gsm8k_eval.py` |

## Per-PR Diff Audit Cards

### PR #13610 - Test reorganization: Move tests to manual/

- Link: https://github.com/sgl-project/sglang/pull/13610
- Status/date: merged / 2025-11-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 74 files, +0/-74, 87 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Test reorganization: Move tests to manual/"; model line: Llama 3.1; category: performance/backend optimization; main diff: `test/manual/entrypoints/http_server/test_abort_request.py`, `test/manual/layers/attention/nsa/test_act_quant_triton.py`, `test/manual/layers/moe/test_moe_runners.py`; technical summary: Covers "Test reorganization: Move tests to manual/"; the main implementation surface is `test/manual/entrypoints/http_server/test_abort_request.py`, `test/manual/layers/attention/nsa/test_act_quant_triton.py`, `test/manual/layers/moe/test_moe_runners.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/manual/entrypoints/http_server/test_abort_request.py` renamed +0/-0 (0 lines); `test/manual/layers/attention/nsa/test_act_quant_triton.py` renamed +0/-0 (0 lines); `test/manual/layers/moe/test_moe_runners.py` renamed +0/-0 (0 lines); `test/manual/models/test_clip_models.py` renamed +0/-0 (0 lines).
- Code diff details:
  - `test/manual/entrypoints/http_server/test_abort_request.py` renamed +0/-0 (0 lines)
  - `test/manual/layers/attention/nsa/test_act_quant_triton.py` renamed +0/-0 (0 lines)
  - `test/manual/layers/moe/test_moe_runners.py` renamed +0/-0 (0 lines)
  - `test/manual/models/test_clip_models.py` renamed +0/-0 (0 lines)
  - `test/manual/models/test_dummy_grok_models.py` renamed +0/-0 (0 lines)
- Key code excerpts:

```diff
No textual patch was returned by GitHub for the selected changed files.
```

- Reviewed files:
  - tests: `test/manual/entrypoints/http_server/test_abort_request.py` renamed +0/-0; `test/manual/layers/attention/nsa/test_act_quant_triton.py` renamed +0/-0; `test/manual/layers/moe/test_moe_runners.py` renamed +0/-0; `test/manual/models/test_clip_models.py` renamed +0/-0; `test/manual/models/test_dummy_grok_models.py` renamed +0/-0; `test/manual/models/test_falcon_h1_models.py` renamed +0/-0
- Risk and verification: The diff ships test coverage in `test/manual/ascend/test_ascend_w8a8_quantization.py`, `test/manual/ascend/test_mindspore_models.py`, `test/manual/cpu/test_comm.py`, `test/manual/debug_utils/test_log_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #13935 - [misc] add llama3.1 chat template

- Link: https://github.com/sgl-project/sglang/pull/13935
- Status/date: merged / 2025-11-25
- Trace source: `git log --name-only -- <model-files>` found it through `examples/chat_template/tool_chat_template_llama3.1_json.jinja`; associated commits `4852aa054cf5`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +121/-0, 123 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[misc] add llama3.1 chat template"; model line: Llama 3.1; category: model support/runtime entry; main diff: `examples/chat_template/tool_chat_template_llama3.1_json.jinja`; technical summary: Covers "[misc] add llama3.1 chat template"; the main implementation surface is `examples/chat_template/tool_chat_template_llama3.1_json.jinja`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `examples/chat_template/tool_chat_template_llama3.1_json.jinja` added +121/-0 (121 lines); hunks: -0,0 +1,121.
- Code diff details:
  - `examples/chat_template/tool_chat_template_llama3.1_json.jinja` added +121/-0 (121 lines); hunks: -0,0 +1,121
- Key code excerpts:

```diff
diff -- examples/chat_template/tool_chat_template_llama3.1_json.jinja
@@ -0,0 +1,121 @@
+{# Copied from https://github.com/vllm-project/vllm/blob/main/examples/tool_chat_template_llama3.1_json.jinja to enable better model response. #}
+{{- bos_token }}
+{%- if custom_tools is defined %}
+    {%- set tools = custom_tools %}
+{%- endif %}
+{%- if not tools_in_user_message is defined %}
```

- Reviewed files:
  - docs: `examples/chat_template/tool_chat_template_llama3.1_json.jinja` added +121/-0
- Risk and verification: This is mostly docs/examples in `examples/chat_template/tool_chat_template_llama3.1_json.jinja`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #13938 - [Minor] Fix lint

- Link: https://github.com/sgl-project/sglang/pull/13938
- Status/date: merged / 2025-11-25
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 7 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Minor] Fix lint"; model line: Llama 3.1; category: bug fix; main diff: `examples/chat_template/tool_chat_template_llama3.1_json.jinja`; technical summary: Covers "[Minor] Fix lint"; the main implementation surface is `examples/chat_template/tool_chat_template_llama3.1_json.jinja`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `examples/chat_template/tool_chat_template_llama3.1_json.jinja` modified +1/-1 (2 lines); hunks: -118,4 +118,4.
- Code diff details:
  - `examples/chat_template/tool_chat_template_llama3.1_json.jinja` modified +1/-1 (2 lines); hunks: -118,4 +118,4
- Key code excerpts:

```diff
diff -- examples/chat_template/tool_chat_template_llama3.1_json.jinja
@@ -118,4 +118,4 @@
-{%- endif %}
+{%- endif %}
```

- Reviewed files:
  - docs: `examples/chat_template/tool_chat_template_llama3.1_json.jinja` modified +1/-1
- Risk and verification: This is mostly docs/examples in `examples/chat_template/tool_chat_template_llama3.1_json.jinja`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #15582 - [CI] Migrate nightly tests to test/registered/

- Link: https://github.com/sgl-project/sglang/pull/15582
- Status/date: merged / 2025-12-23
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 64 files, +93/-140, 611 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CI] Migrate nightly tests to test/registered/"; model line: Llama 3.1; category: docs/tests/CI; main diff: `test/registered/ascend/llm_models/test_ascend_phi_4_multimodal.py`, `test/registered/ascend/vlm_models/test_ascend_phi4_multimodal_instruct.py`, `test/run_suite.py`; technical summary: Covers "[CI] Migrate nightly tests to test/registered/"; the main implementation surface is `test/registered/ascend/llm_models/test_ascend_phi_4_multimodal.py`, `test/registered/ascend/vlm_models/test_ascend_phi4_multimodal_instruct.py`, `test/run_suite.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/ascend/llm_models/test_ascend_phi_4_multimodal.py` renamed +1/-2 (3 lines); hunks: -1,7 +1,6; `test/registered/ascend/vlm_models/test_ascend_phi4_multimodal_instruct.py` renamed +1/-2 (3 lines); hunks: -1,7 +1,6; `test/run_suite.py` modified +9/-8 (17 lines); hunks: -43,6 +43,11; -149,14 +154,10 @@ def run_a_suite(args):; symbols: run_a_suite, touching `run_a_suite`; `test/registered/8-gpu-models/test_deepseek_v31.py` modified +3/-9 (12 lines); hunks: -1,15 +1,9.
- Code diff details:
  - `test/registered/ascend/llm_models/test_ascend_phi_4_multimodal.py` renamed +1/-2 (3 lines); hunks: -1,7 +1,6
  - `test/registered/ascend/vlm_models/test_ascend_phi4_multimodal_instruct.py` renamed +1/-2 (3 lines); hunks: -1,7 +1,6
  - `test/run_suite.py` modified +9/-8 (17 lines); hunks: -43,6 +43,11; -149,14 +154,10 @@ def run_a_suite(args):; symbols: run_a_suite
  - `test/registered/8-gpu-models/test_deepseek_v31.py` modified +3/-9 (12 lines); hunks: -1,15 +1,9
  - `test/registered/8-gpu-models/test_deepseek_v32.py` modified +3/-9 (12 lines); hunks: -1,15 +1,9
- Key code excerpts:

```diff
diff -- test/registered/ascend/llm_models/test_ascend_phi_4_multimodal.py
@@ -1,7 +1,6 @@
-from gsm8k_ascend_mixin import GSM8KAscendMixin
+from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
diff -- test/registered/ascend/vlm_models/test_ascend_phi4_multimodal_instruct.py
@@ -1,7 +1,6 @@
-from test_vlm_utils import TestVLMModels
+from sglang.test.ascend.vlm_utils import TestVLMModels
diff -- test/run_suite.py
@@ -43,6 +43,11 @@
+        # Eval and perf suites (2-gpu)
+        "nightly-eval-text-2-gpu",
+        "nightly-eval-vlm-2-gpu",
+        "nightly-perf-text-2-gpu",
+        "nightly-perf-vlm-2-gpu",
@@ -149,14 +154,10 @@ def run_a_suite(args):
diff -- test/registered/8-gpu-models/test_deepseek_v31.py
@@ -1,15 +1,9 @@
```

- Reviewed files:
  - tests: `test/registered/ascend/llm_models/test_ascend_phi_4_multimodal.py` renamed +1/-2; `test/registered/ascend/vlm_models/test_ascend_phi4_multimodal_instruct.py` renamed +1/-2; `test/run_suite.py` modified +9/-8; `test/registered/8-gpu-models/test_deepseek_v31.py` modified +3/-9; `test/registered/8-gpu-models/test_deepseek_v32.py` modified +3/-9; `test/registered/8-gpu-models/test_glm_46.py` modified +3/-9
- Risk and verification: The diff ships test coverage in `python/sglang/test/accuracy_test_runner.py`, `python/sglang/test/ascend/__init__.py`, `python/sglang/test/ascend/gsm8k_ascend_mixin.py`, `python/sglang/test/ascend/vlm_utils.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17895 - [AMD] Add kimi mi35x nightly test, folder organization and several stability fixes

- Link: https://github.com/sgl-project/sglang/pull/17895
- Status/date: merged / 2026-02-04
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 34 files, +184/-14, 414 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Add kimi mi35x nightly test, folder organization and several stability fixes"; model line: Llama 3.1; category: bug fix; main diff: `test/registered/amd/accuracy/mi35x/test_kimi_k2_eval_mi35x.py`, `.github/workflows/nightly-test-amd.yml`, `python/sglang/test/nightly_utils.py`; technical summary: Covers "[AMD] Add kimi mi35x nightly test, folder organization and several stability fixes"; the main implementation surface is `test/registered/amd/accuracy/mi35x/test_kimi_k2_eval_mi35x.py`, `.github/workflows/nightly-test-amd.yml`, `python/sglang/test/nightly_utils.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/amd/accuracy/mi35x/test_kimi_k2_eval_mi35x.py` added +105/-0 (105 lines); hunks: -0,0 +1,105; symbols: TestKimiK2EvalMI35x, setUpClass, test_kimi_k2_gsm8k_accuracy, touching `TestKimiK2EvalMI35x, setUpClass, test_kimi_k2_gsm8k_accuracy`; `.github/workflows/nightly-test-amd.yml` modified +40/-5 (45 lines); hunks: -34,6 +34,7 @@ on:; -582,13 +583,13 @@ jobs:; `python/sglang/test/nightly_utils.py` modified +16/-4 (20 lines); hunks: -94,6 +94,7 @@ def build_benchmark_command(; -106,6 +107,7 @@ def build_benchmark_command(; symbols: build_benchmark_command, run_benchmark_for_model, touching `build_benchmark_command, run_benchmark_for_model`; `test/registered/amd/accuracy/mi35x/test_gpt_oss_eval_mi35x.py` modified +6/-2 (8 lines); hunks: -75,7 +75,9 @@ def __post_init__(self):; -93,7 +95,9 @@ def __post_init__(self):; symbols: __post_init__, touching `__post_init__`.
- Code diff details:
  - `test/registered/amd/accuracy/mi35x/test_kimi_k2_eval_mi35x.py` added +105/-0 (105 lines); hunks: -0,0 +1,105; symbols: TestKimiK2EvalMI35x, setUpClass, test_kimi_k2_gsm8k_accuracy
  - `.github/workflows/nightly-test-amd.yml` modified +40/-5 (45 lines); hunks: -34,6 +34,7 @@ on:; -582,13 +583,13 @@ jobs:
  - `python/sglang/test/nightly_utils.py` modified +16/-4 (20 lines); hunks: -94,6 +94,7 @@ def build_benchmark_command(; -106,6 +107,7 @@ def build_benchmark_command(; symbols: build_benchmark_command, run_benchmark_for_model
  - `test/registered/amd/accuracy/mi35x/test_gpt_oss_eval_mi35x.py` modified +6/-2 (8 lines); hunks: -75,7 +75,9 @@ def __post_init__(self):; -93,7 +95,9 @@ def __post_init__(self):; symbols: __post_init__
  - `test/registered/amd/accuracy/mi35x/test_grok1_int4_eval_mi35x.py` modified +2/-2 (4 lines); hunks: -23,9 +23,9
- Key code excerpts:

```diff
diff -- test/registered/amd/accuracy/mi35x/test_kimi_k2_eval_mi35x.py
@@ -0,0 +1,105 @@
+"""MI35x Kimi-K2 GSM8K Completion Evaluation Test (8-GPU)
+Tests moonshotai/Kimi-K2-Instruct-0905 with GSM8K few-shot benchmark on MI35x.
+Registry: nightly-amd-accuracy-8-gpu-mi35x-kimi-k2 suite
+"""
+import os
+import unittest
diff -- .github/workflows/nightly-test-amd.yml
@@ -34,6 +34,7 @@ on:
+          - 'nightly-8-gpu-mi35x-kimi-k2'
@@ -582,13 +583,13 @@ jobs:
-        timeout-minutes: 60
+        timeout-minutes: 90
-            python3 run_suite.py --hw amd --suite nightly-amd-accuracy-8-gpu-mi35x-grok1-int4 --nightly --timeout-per-file 3600 || TEST_EXIT_CODE=$?
+            python3 run_suite.py --hw amd --suite nightly-amd-accuracy-8-gpu-mi35x-grok1-int4 --nightly --timeout-per-file 5400 || TEST_EXIT_CODE=$?
diff -- python/sglang/test/nightly_utils.py
@@ -94,6 +94,7 @@ def build_benchmark_command(
```

- Reviewed files:
  - tests: `test/registered/amd/accuracy/mi35x/test_kimi_k2_eval_mi35x.py` added +105/-0; `python/sglang/test/nightly_utils.py` modified +16/-4; `test/registered/amd/accuracy/mi35x/test_gpt_oss_eval_mi35x.py` modified +6/-2; `test/registered/amd/accuracy/mi35x/test_grok1_int4_eval_mi35x.py` modified +2/-2; `test/registered/amd/accuracy/mi35x/test_grok2_eval_mi35x.py` modified +1/-1; `test/registered/amd/perf/mi30x/test_deepseek_v31_perf.py` renamed +1/-0
  - ci: `.github/workflows/nightly-test-amd.yml` modified +40/-5
- Risk and verification: The diff ships test coverage in `python/sglang/test/nightly_utils.py`, `test/registered/amd/accuracy/mi30x/test_deepseek_r1_eval_amd.py`, `test/registered/amd/accuracy/mi30x/test_deepseek_v31_eval_amd.py`, `test/registered/amd/accuracy/mi30x/test_deepseek_v32_dp_eval_amd.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17799 - [AMD] rocm 7.2 image release, PR test, Nightly Test

- Link: https://github.com/sgl-project/sglang/pull/17799
- Status/date: merged / 2026-02-12
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 26 files, +2719/-156, 3314 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] rocm 7.2 image release, PR test, Nightly Test"; model line: Llama 3.1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/layers/moe/moe_runner/triton.py`; technical summary: Covers "[AMD] rocm 7.2 image release, PR test, Nightly Test"; the main implementation surface is `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/layers/moe/moe_runner/triton.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +43/-12 (55 lines); hunks: -57,11 +57,22; -513,9 +524,15 @@ def fused_experts_impl(; symbols: fused_experts_impl, touching `fused_experts_impl`; `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +45/-4 (49 lines); hunks: -64,6 +64,7; -76,8 +77,11; symbols: per_token_group_quant_mla_deep_gemm_masked_fp8, _native_dynamic_per_token_quant_fp8, _native_dynamic_per_tensor_quant_fp8, _native_static_quant_fp8, touching `per_token_group_quant_mla_deep_gemm_masked_fp8, _native_dynamic_per_token_quant_fp8, _native_dynamic_per_tensor_quant_fp8`; `python/sglang/srt/layers/moe/moe_runner/triton.py` modified +16/-2 (18 lines); hunks: -41,6 +41,7; -49,7 +50,13; symbols: run, touching `run`; `python/sglang/srt/layers/layernorm.py` modified +14/-1 (15 lines); hunks: -64,11 +64,20; -181,6 +190,10 @@ def forward_hip(; symbols: forward_hip, touching `forward_hip`.
- Code diff details:
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +43/-12 (55 lines); hunks: -57,11 +57,22; -513,9 +524,15 @@ def fused_experts_impl(; symbols: fused_experts_impl
  - `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +45/-4 (49 lines); hunks: -64,6 +64,7; -76,8 +77,11; symbols: per_token_group_quant_mla_deep_gemm_masked_fp8, _native_dynamic_per_token_quant_fp8, _native_dynamic_per_tensor_quant_fp8, _native_static_quant_fp8
  - `python/sglang/srt/layers/moe/moe_runner/triton.py` modified +16/-2 (18 lines); hunks: -41,6 +41,7; -49,7 +50,13; symbols: run
  - `python/sglang/srt/layers/layernorm.py` modified +14/-1 (15 lines); hunks: -64,11 +64,20; -181,6 +190,10 @@ def forward_hip(; symbols: forward_hip
  - `python/sglang/srt/layers/quantization/unquant.py` modified +8/-2 (10 lines); hunks: -224,7 +224,10 @@ def create_weights(; -383,7 +386,10 @@ def forward_cuda(; symbols: create_weights, process_weights_after_loading, forward_cuda
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py
@@ -57,11 +57,22 @@
-    else:
-        from vllm import _custom_ops as vllm_ops
+    # Note: vllm_ops is not needed for HIP when _use_aiter=False
+    # because the code uses moe_sum_reduce_triton as fallback (line 619)
+# Try to import vllm_ops for non-CUDA/HIP/XPU platforms
+_has_vllm_ops = False
diff -- python/sglang/srt/layers/quantization/fp8_kernel.py
@@ -64,6 +64,7 @@
+    _has_vllm = False
@@ -76,8 +77,11 @@
+            _has_vllm = True
-            raise ImportError("vllm is required when SGLANG_USE_AITER is set to False")
+            # Fallback: vllm not available, will use native PyTorch implementation
+            _has_vllm = False
diff -- python/sglang/srt/layers/moe/moe_runner/triton.py
@@ -41,6 +41,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +43/-12; `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +45/-4; `python/sglang/srt/layers/moe/moe_runner/triton.py` modified +16/-2; `python/sglang/srt/layers/layernorm.py` modified +14/-1; `python/sglang/srt/layers/quantization/unquant.py` modified +8/-2; `python/sglang/srt/models/deepseek_janus_pro.py` modified +1/-1
  - tests: `test/registered/layers/mamba/test_mamba_ssm_ssd.py` modified +5/-0
  - ci: `.github/workflows/nightly-test-amd-rocm720.yml` added +868/-0
- Risk and verification: The diff ships test coverage in `python/sglang/test/gpt_oss_common.py`, `python/sglang/test/nightly_utils.py`, `test/registered/amd/accuracy/mi30x/test_gpt_oss_eval_amd.py`, `test/registered/amd/accuracy/mi30x/test_gsm8k_eval_amd.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18886 - Fix eval tests not capturing server launch failures

- Link: https://github.com/sgl-project/sglang/pull/18886
- Status/date: merged / 2026-02-17
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +137/-95, 410 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix eval tests not capturing server launch failures"; model line: Llama 3.1; category: bug fix; main diff: `python/sglang/srt/model_loader/ci_weight_validation.py`, `test/registered/eval/test_text_models_gsm8k_eval.py`, `python/sglang/test/nightly_utils.py`; technical summary: Covers "Fix eval tests not capturing server launch failures"; the main implementation surface is `python/sglang/srt/model_loader/ci_weight_validation.py`, `test/registered/eval/test_text_models_gsm8k_eval.py`, `python/sglang/test/nightly_utils.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/model_loader/ci_weight_validation.py` modified +91/-57 (148 lines); hunks: -1730,30 +1730,44 @@ def _validate_weights_after_download(; -1826,13 +1840,10 @@ def ci_download_with_validation_and_retry(; symbols: _validate_weights_after_download, _get_lock_file_path, ci_download_with_validation_and_retry, touching `_validate_weights_after_download, _get_lock_file_path, ci_download_with_validation_and_retry`; `test/registered/eval/test_text_models_gsm8k_eval.py` modified +15/-14 (29 lines); hunks: -11,7 +11,6; -20,6 +19,10; symbols: test_mgsm_en_all_models, touching `test_mgsm_en_all_models`; `python/sglang/test/nightly_utils.py` modified +16/-12 (28 lines); hunks: -259,18 +259,21 @@ def run_benchmark_for_model(; -311,7 +314,8 @@ def run_benchmark_for_model(; symbols: run_benchmark_for_model, _get_spec_accept_length, touching `run_benchmark_for_model, _get_spec_accept_length`; `test/registered/eval/test_vlms_mmmu_eval.py` modified +15/-12 (27 lines); hunks: -7,7 +7,6; -16,6 +15,10; symbols: test_mmmu_vlm_models, touching `test_mmmu_vlm_models`.
- Code diff details:
  - `python/sglang/srt/model_loader/ci_weight_validation.py` modified +91/-57 (148 lines); hunks: -1730,30 +1730,44 @@ def _validate_weights_after_download(; -1826,13 +1840,10 @@ def ci_download_with_validation_and_retry(; symbols: _validate_weights_after_download, _get_lock_file_path, ci_download_with_validation_and_retry
  - `test/registered/eval/test_text_models_gsm8k_eval.py` modified +15/-14 (29 lines); hunks: -11,7 +11,6; -20,6 +19,10; symbols: test_mgsm_en_all_models
  - `python/sglang/test/nightly_utils.py` modified +16/-12 (28 lines); hunks: -259,18 +259,21 @@ def run_benchmark_for_model(; -311,7 +314,8 @@ def run_benchmark_for_model(; symbols: run_benchmark_for_model, _get_spec_accept_length
  - `test/registered/eval/test_vlms_mmmu_eval.py` modified +15/-12 (27 lines); hunks: -7,7 +7,6; -16,6 +15,10; symbols: test_mmmu_vlm_models
- Key code excerpts:

```diff
diff -- python/sglang/srt/model_loader/ci_weight_validation.py
@@ -1730,30 +1730,44 @@ def _validate_weights_after_download(
-def _get_lock_file_path(model_name_or_path: str) -> str:
+def _get_lock_file_path(
+    model_name_or_path: str, cache_dir: Optional[str] = None
+) -> str:
-    Uses file-based locking (fcntl.flock) to ensure only one process downloads
-    while others wait. This works regardless of how processes are spawned
diff -- test/registered/eval/test_text_models_gsm8k_eval.py
@@ -11,7 +11,6 @@
-    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
@@ -20,6 +19,10 @@
+# Nightly eval tests run large models (up to 70B+ params) that may need
+# downloading on cache miss. Use a longer timeout than the default 600s.
+NIGHTLY_EVAL_SERVER_TIMEOUT = 1800
@@ -72,19 +75,19 @@ def test_mgsm_en_all_models(self):
diff -- python/sglang/test/nightly_utils.py
@@ -259,18 +259,21 @@ def run_benchmark_for_model(
```

- Reviewed files:
  - runtime: `python/sglang/srt/model_loader/ci_weight_validation.py` modified +91/-57
  - tests: `test/registered/eval/test_text_models_gsm8k_eval.py` modified +15/-14; `python/sglang/test/nightly_utils.py` modified +16/-12; `test/registered/eval/test_vlms_mmmu_eval.py` modified +15/-12
- Risk and verification: The diff ships test coverage in `python/sglang/test/nightly_utils.py`, `test/registered/eval/test_text_models_gsm8k_eval.py`, `test/registered/eval/test_vlms_mmmu_eval.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18911 - [AMD] [GLM-5 Day 0] Add GLM-5 nightly test

- Link: https://github.com/sgl-project/sglang/pull/18911
- Status/date: merged / 2026-02-25
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +635/-1, 725 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] [GLM-5 Day 0] Add GLM-5 nightly test"; model line: Llama 3.1; category: docs/tests/CI; main diff: `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py`, `.github/workflows/nightly-test-amd-rocm720.yml`; technical summary: Covers "[AMD] [GLM-5 Day 0] Add GLM-5 nightly test"; the main implementation surface is `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py`, `.github/workflows/nightly-test-amd-rocm720.yml`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py` added +249/-0 (249 lines); hunks: -0,0 +1,249; symbols: ModelConfig, get_display_name, get_one_example, get_few_shot_examples, touching `ModelConfig, get_display_name, get_one_example`; `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py` added +244/-0 (244 lines); hunks: -0,0 +1,244; symbols: ModelConfig, get_display_name, get_one_example, get_few_shot_examples, touching `ModelConfig, get_display_name, get_one_example`; `.github/workflows/nightly-test-amd-rocm720.yml` modified +71/-0 (71 lines); hunks: -32,6 +32,7 @@ on:; -43,6 +44,7 @@ on:; `.github/workflows/nightly-test-amd.yml` modified +70/-0 (70 lines); hunks: -32,9 +32,11 @@ on:; -494,6 +496,38 @@ jobs:.
- Code diff details:
  - `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py` added +249/-0 (249 lines); hunks: -0,0 +1,249; symbols: ModelConfig, get_display_name, get_one_example, get_few_shot_examples
  - `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py` added +244/-0 (244 lines); hunks: -0,0 +1,244; symbols: ModelConfig, get_display_name, get_one_example, get_few_shot_examples
  - `.github/workflows/nightly-test-amd-rocm720.yml` modified +71/-0 (71 lines); hunks: -32,6 +32,7 @@ on:; -43,6 +44,7 @@ on:
  - `.github/workflows/nightly-test-amd.yml` modified +70/-0 (70 lines); hunks: -32,9 +32,11 @@ on:; -494,6 +496,38 @@ jobs:
  - `test/registered/amd/accuracy/mi30x/test_gsm8k_eval_amd.py` modified +1/-1 (2 lines); hunks: -42,7 +42,7
- Key code excerpts:

```diff
diff -- test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py
@@ -0,0 +1,249 @@
+"""MI35x GLM-5 GSM8K Completion Evaluation Test (8-GPU)
+Tests GLM-5 with NSA attention backend using few-shot completion
+benchmark on MI35x.
+Registry: nightly-amd-8-gpu-mi35x-glm5 suite
+"""
+import ast
diff -- test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py
@@ -0,0 +1,244 @@
+"""AMD GLM-5 GSM8K Completion Evaluation Test (8-GPU)
+Tests GLM-5 with NSA attention backend using few-shot completion
+benchmark on MI325/MI300X.
+Registry: nightly-amd-accuracy-8-gpu-glm5 suite
+"""
+import ast
diff -- .github/workflows/nightly-test-amd-rocm720.yml
@@ -32,6 +32,7 @@ on:
```

- Reviewed files:
  - tests: `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py` added +249/-0; `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py` added +244/-0; `test/registered/amd/accuracy/mi30x/test_gsm8k_eval_amd.py` modified +1/-1
  - ci: `.github/workflows/nightly-test-amd-rocm720.yml` modified +71/-0; `.github/workflows/nightly-test-amd.yml` modified +70/-0
- Risk and verification: The diff ships test coverage in `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py`, `test/registered/amd/accuracy/mi30x/test_gsm8k_eval_amd.py`, `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #19778 - Adding correct path for module not found error while collecting test

- Link: https://github.com/sgl-project/sglang/pull/19778
- Status/date: merged / 2026-03-07
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +8/-13, 63 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Adding correct path for module not found error while collecting test"; model line: Llama 3.1; category: bug fix; main diff: `test/manual/test_two_batch_overlap.py`, `test/manual/nightly/test_deepseek_v31_perf.py`, `test/manual/nightly/test_deepseek_v32_perf.py`; technical summary: Covers "Adding correct path for module not found error while collecting test"; the main implementation surface is `test/manual/test_two_batch_overlap.py`, `test/manual/nightly/test_deepseek_v31_perf.py`, `test/manual/nightly/test_deepseek_v32_perf.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/manual/test_two_batch_overlap.py` modified +3/-3 (6 lines); hunks: -3,12 +3,12; `test/manual/nightly/test_deepseek_v31_perf.py` modified +1/-2 (3 lines); hunks: -1,7 +1,6; `test/manual/nightly/test_deepseek_v32_perf.py` modified +1/-2 (3 lines); hunks: -1,7 +1,6; `test/manual/nightly/test_text_models_perf.py` modified +1/-2 (3 lines); hunks: -1,7 +1,6.
- Code diff details:
  - `test/manual/test_two_batch_overlap.py` modified +3/-3 (6 lines); hunks: -3,12 +3,12
  - `test/manual/nightly/test_deepseek_v31_perf.py` modified +1/-2 (3 lines); hunks: -1,7 +1,6
  - `test/manual/nightly/test_deepseek_v32_perf.py` modified +1/-2 (3 lines); hunks: -1,7 +1,6
  - `test/manual/nightly/test_text_models_perf.py` modified +1/-2 (3 lines); hunks: -1,7 +1,6
  - `test/manual/nightly/test_vlms_perf.py` modified +1/-2 (3 lines); hunks: -2,8 +2,7
- Key code excerpts:

```diff
diff -- test/manual/test_two_batch_overlap.py
@@ -3,12 +3,12 @@
-from sglang.srt.environ import envs
-from sglang.srt.model_executor.forward_batch_info import ForwardMode
-from sglang.srt.two_batch_overlap import (
+from sglang.srt.batch_overlap.two_batch_overlap import (
+from sglang.srt.environ import envs
+from sglang.srt.model_executor.forward_batch_info import ForwardMode
diff -- test/manual/nightly/test_deepseek_v31_perf.py
@@ -1,7 +1,6 @@
-from nightly_utils import NightlyBenchmarkRunner
+from sglang.test.nightly_utils import NightlyBenchmarkRunner
diff -- test/manual/nightly/test_deepseek_v32_perf.py
@@ -1,7 +1,6 @@
-from nightly_utils import NightlyBenchmarkRunner
+from sglang.test.nightly_utils import NightlyBenchmarkRunner
diff -- test/manual/nightly/test_text_models_perf.py
@@ -1,7 +1,6 @@
```

- Reviewed files:
  - tests: `test/manual/test_two_batch_overlap.py` modified +3/-3; `test/manual/nightly/test_deepseek_v31_perf.py` modified +1/-2; `test/manual/nightly/test_deepseek_v32_perf.py` modified +1/-2; `test/manual/nightly/test_text_models_perf.py` modified +1/-2; `test/manual/nightly/test_vlms_perf.py` modified +1/-2
- Risk and verification: The diff ships test coverage in `test/manual/nightly/test_deepseek_v31_perf.py`, `test/manual/nightly/test_deepseek_v32_perf.py`, `test/manual/nightly/test_text_models_perf.py`, `test/manual/nightly/test_vlms_perf.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #21931 - [CI] Migrate mgsm_en eval to gsm8k to remove openaipublic dependency

- Link: https://github.com/sgl-project/sglang/pull/21931
- Status/date: merged / 2026-04-07
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +82/-77, 336 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CI] Migrate mgsm_en eval to gsm8k to remove openaipublic dependency"; model line: Llama 3.1; category: performance/backend optimization; main diff: `test/registered/amd/accuracy/mi30x/test_gsm8k_eval_amd.py`, `test/registered/eval/test_text_models_gsm8k_eval.py`, `test/registered/piecewise_cuda_graph/test_piecewise_cuda_graph_support_1_gpu.py`; technical summary: Covers "[CI] Migrate mgsm_en eval to gsm8k to remove openaipublic dependency"; the main implementation surface is `test/registered/amd/accuracy/mi30x/test_gsm8k_eval_amd.py`, `test/registered/eval/test_text_models_gsm8k_eval.py`, `test/registered/piecewise_cuda_graph/test_piecewise_cuda_graph_support_1_gpu.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/amd/accuracy/mi30x/test_gsm8k_eval_amd.py` modified +27/-26 (53 lines); hunks: -1,7 +1,7; -35,34 +35,35; symbols: check_model_scores, TestNightlyGsm8KEval, setUpClass, touching `check_model_scores, TestNightlyGsm8KEval, setUpClass`; `test/registered/eval/test_text_models_gsm8k_eval.py` modified +22/-21 (43 lines); hunks: -26,28 +26,29; -66,7 +67,7 @@ def setUpClass(cls):; symbols: TestNightlyGsm8KEval, setUpClass, test_mgsm_en_all_models, test_gsm8k_all_models, touching `TestNightlyGsm8KEval, setUpClass, test_mgsm_en_all_models`; `test/registered/piecewise_cuda_graph/test_piecewise_cuda_graph_support_1_gpu.py` modified +16/-16 (32 lines); hunks: -41,21 +41,19 @@ def setUpClass(cls):; -79,21 +77,23 @@ def setUpClass(cls):; symbols: setUpClass, tearDownClass, test_mgsm_accuracy, test_gsm8k_accuracy, touching `setUpClass, tearDownClass, test_mgsm_accuracy`; `test/registered/quant/test_quantization.py` modified +8/-5 (13 lines); hunks: -19,9 +19,12; -93,7 +96,7 @@ def setUpClass(cls):; symbols: setUpClass, test_mgsm_en_all_models, test_gsm8k_all_models, touching `setUpClass, test_mgsm_en_all_models, test_gsm8k_all_models`.
- Code diff details:
  - `test/registered/amd/accuracy/mi30x/test_gsm8k_eval_amd.py` modified +27/-26 (53 lines); hunks: -1,7 +1,7; -35,34 +35,35; symbols: check_model_scores, TestNightlyGsm8KEval, setUpClass
  - `test/registered/eval/test_text_models_gsm8k_eval.py` modified +22/-21 (43 lines); hunks: -26,28 +26,29; -66,7 +67,7 @@ def setUpClass(cls):; symbols: TestNightlyGsm8KEval, setUpClass, test_mgsm_en_all_models, test_gsm8k_all_models
  - `test/registered/piecewise_cuda_graph/test_piecewise_cuda_graph_support_1_gpu.py` modified +16/-16 (32 lines); hunks: -41,21 +41,19 @@ def setUpClass(cls):; -79,21 +77,23 @@ def setUpClass(cls):; symbols: setUpClass, tearDownClass, test_mgsm_accuracy, test_gsm8k_accuracy
  - `test/registered/quant/test_quantization.py` modified +8/-5 (13 lines); hunks: -19,9 +19,12; -93,7 +96,7 @@ def setUpClass(cls):; symbols: setUpClass, test_mgsm_en_all_models, test_gsm8k_all_models
  - `test/registered/scheduler/test_prefill_delayer.py` modified +5/-5 (10 lines); hunks: -428,10 +428,10 @@ async def send_normal_request(dp_rank, req_idx):; -454,14 +454,14 @@ def _run_accuracy_test(self, prefill_delayer: bool):; symbols: send_normal_request, TestPrefillDelayerAccuracy, test_1_mgsm_en_has_prefill_delayer, test_1_gsm8k_has_prefill_delayer
- Key code excerpts:

```diff
diff -- test/registered/amd/accuracy/mi30x/test_gsm8k_eval_amd.py
@@ -1,7 +1,7 @@
-This test evaluates instruction-tuned models on the mgsm_en benchmark using chat completions.
+This test evaluates instruction-tuned models on the gsm8k benchmark using chat completions.
@@ -35,34 +35,35 @@
+    # Thresholds set at 5% below reported GSM8K (5-shot/CoT) scores
-    "meta-llama/Llama-3.1-8B-Instruct": 0.82,
-    "meta-llama/Llama-3.1-70B-Instruct": 0.95,
diff -- test/registered/eval/test_text_models_gsm8k_eval.py
@@ -26,28 +26,29 @@
-    "meta-llama/Llama-3.1-8B-Instruct": 0.82,
-    "mistralai/Mistral-7B-Instruct-v0.3": 0.58,
-    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct": 0.85,
-    "google/gemma-2-27b-it": 0.91,
-    "meta-llama/Llama-3.1-70B-Instruct": 0.95,
-    "mistralai/Mixtral-8x7B-Instruct-v0.1": 0.616,
diff -- test/registered/piecewise_cuda_graph/test_piecewise_cuda_graph_support_1_gpu.py
@@ -41,21 +41,19 @@ def setUpClass(cls):
```

- Reviewed files:
  - tests: `test/registered/amd/accuracy/mi30x/test_gsm8k_eval_amd.py` modified +27/-26; `test/registered/eval/test_text_models_gsm8k_eval.py` modified +22/-21; `test/registered/piecewise_cuda_graph/test_piecewise_cuda_graph_support_1_gpu.py` modified +16/-16; `test/registered/quant/test_quantization.py` modified +8/-5; `test/registered/scheduler/test_prefill_delayer.py` modified +5/-5; `test/registered/distributed/test_dp_attention_large.py` modified +2/-2
- Risk and verification: The diff ships test coverage in `test/registered/amd/accuracy/mi30x/test_gsm8k_eval_amd.py`, `test/registered/distributed/test_dp_attention_large.py`, `test/registered/distributed/test_pp_single_node.py`, `test/registered/eval/test_text_models_gsm8k_eval.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #23001 - Add new Mintlify documentation site (docs_new/)

- Link: https://github.com/sgl-project/sglang/pull/23001
- Status/date: merged / 2026-04-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 330 files, +80364/-0, 68714 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add new Mintlify documentation site (docs_new/)"; model line: Llama 3.1; category: docs/tests/CI; main diff: `docs_new/docs/advanced_features/tool_parser.mdx`, `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx`, `docs_new/docs/advanced_features/separate_reasoning.mdx`; technical summary: Covers "Add new Mintlify documentation site (docs_new/)"; the main implementation surface is `docs_new/docs/advanced_features/tool_parser.mdx`, `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx`, `docs_new/docs/advanced_features/separate_reasoning.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/docs/advanced_features/tool_parser.mdx` added +740/-0 (740 lines); hunks: -0,0 +1,740; symbols: get_messages, get_current_weather, convert_dict_to_tool, in, touching `get_messages, get_current_weather, convert_dict_to_tool`; `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx` added +663/-0 (663 lines); hunks: -0,0 +1,663; symbols: CapitalInfo, get_messages, touching `CapitalInfo, get_messages`; `docs_new/docs/advanced_features/separate_reasoning.mdx` added +317/-0 (317 lines); hunks: -0,0 +1,317; `docs_new/docs/hardware-platforms/ascend-npus/Support-Features-on-Ascend-NPU.mdx` added +3327/-0 (3327 lines).
- Code diff details:
  - `docs_new/docs/advanced_features/tool_parser.mdx` added +740/-0 (740 lines); hunks: -0,0 +1,740; symbols: get_messages, get_current_weather, convert_dict_to_tool, in
  - `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx` added +663/-0 (663 lines); hunks: -0,0 +1,663; symbols: CapitalInfo, get_messages
  - `docs_new/docs/advanced_features/separate_reasoning.mdx` added +317/-0 (317 lines); hunks: -0,0 +1,317
  - `docs_new/docs/hardware-platforms/ascend-npus/Support-Features-on-Ascend-NPU.mdx` added +3327/-0 (3327 lines)
  - `docs_new/docs/hardware-platforms/ascend-npus/Best-Practice-on-Ascend-NPU.mdx` added +2911/-0 (2911 lines)
- Key code excerpts:

```diff
diff -- docs_new/docs/advanced_features/tool_parser.mdx
@@ -0,0 +1,740 @@
+---
+title: "Tool Parser"
+metatags:
+    description: "SGLang function calling: tool parsers for DeepSeek, Llama, Qwen, Mistral, GLM, Kimi K2. OpenAI-compatible tool use API."
+---
+This guide demonstrates how to use SGLang’s [Function calling](https://platform.openai.com/docs/guides/function-calling) functionality.
diff -- docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx
@@ -0,0 +1,663 @@
+---
+title: "Structured Outputs For Reasoning Models"
+metatags:
+    description: "SGLang structured outputs for reasoning models: free-form thinking with constrained final output for DeepSeek R1, QwQ models."
+---
+When working with reasoning models that use special tokens like `&lt;think&gt;...&lt;/think&gt;` to denote reasoning sections, you might want to allow free-form text within these
diff -- docs_new/docs/advanced_features/separate_reasoning.mdx
@@ -0,0 +1,317 @@
```

- Reviewed files:
  - docs: `docs_new/docs/advanced_features/tool_parser.mdx` added +740/-0; `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx` added +663/-0; `docs_new/docs/advanced_features/separate_reasoning.mdx` added +317/-0; `docs_new/docs/hardware-platforms/ascend-npus/Support-Features-on-Ascend-NPU.mdx` added +3327/-0; `docs_new/docs/hardware-platforms/ascend-npus/Best-Practice-on-Ascend-NPU.mdx` added +2911/-0; `docs_new/docs/advanced_features/server_arguments.mdx` added +2871/-0
- Risk and verification: This is mostly docs/examples in `docs_new/.github/workflows/sync-lmsys-sglang-blogs.yml`, `docs_new/.gitignore`, `docs_new/.mintignore`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23337 - [Docs] Sync docs_new with legacy docs and update migration redirects

- Link: https://github.com/sgl-project/sglang/pull/23337
- Status/date: merged / 2026-04-21
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 179 files, +16004/-8152, 23604 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Docs] Sync docs_new with legacy docs and update migration redirects"; model line: Llama 3.1; category: docs/tests/CI; main diff: `docs_new/docs/supported-models/multimodal_language_models.mdx`, `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx`; technical summary: Covers "[Docs] Sync docs_new with legacy docs and update migration redirects"; the main implementation surface is `docs_new/docs/supported-models/multimodal_language_models.mdx`, `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/docs/supported-models/multimodal_language_models.mdx` renamed +73/-14 (87 lines); hunks: -1,15 +1,18; -19,11 +22,9 @@ Below the supported models are summarized in a table.; `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx` modified +23/-23 (46 lines); hunks: -3,17 +3,17 @@ title: "Structured Outputs For Reasoning Models"; -252,9 +252,9 @@ If a you choose to call a function ONLY reply in the followi...; `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx` renamed +1809/-463 (2272 lines); `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_support_features.mdx` renamed +671/-1418 (2089 lines).
- Code diff details:
  - `docs_new/docs/supported-models/multimodal_language_models.mdx` renamed +73/-14 (87 lines); hunks: -1,15 +1,18; -19,11 +22,9 @@ Below the supported models are summarized in a table.
  - `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx` modified +23/-23 (46 lines); hunks: -3,17 +3,17 @@ title: "Structured Outputs For Reasoning Models"; -252,9 +252,9 @@ If a you choose to call a function ONLY reply in the followi...
  - `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx` renamed +1809/-463 (2272 lines)
  - `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_support_features.mdx` renamed +671/-1418 (2089 lines)
  - `docs_new/docs/advanced_features/server_arguments.mdx` modified +446/-486 (932 lines)
- Key code excerpts:

```diff
diff -- docs_new/docs/supported-models/multimodal_language_models.mdx
@@ -1,15 +1,18 @@
+---
+title: "Multimodal Language Models"
+metatags:
+  description: "Documentation for Multimodal Language Models"
+---
-<CodeGroup>
diff -- docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx
@@ -3,17 +3,17 @@ title: "Structured Outputs For Reasoning Models"
-When working with reasoning models that use special tokens like `&lt;think&gt;...&lt;/think&gt;` to denote reasoning sections, you might want to allow free-form text within these
+When working with reasoning models that use special tokens like `<think>...</think>` to denote reasoning sections, you might want to allow free-form text within these sections whi
-To enable this feature, use the `--reasoning-parser` flag which decide the think_end_token, such as `&lt;/think&gt;`, when launching the server. You can also specify the reasoning
+To enable this feature, use the `--reasoning-parser` flag which decide the think_end_token, such as `</think>`, when launching the server. You can also specify the reasoning parse
-- [DeepSeek R1 series](https://huggingface.co/collections/deepseek-ai/deepseek-r1-678e1e131c0169c0bc89728d): The reasoning content is wrapped with `&lt;think&gt;` and `&lt;/think&
-- [QwQ](https://huggingface.co/Qwen/QwQ-32B): The reasoning content is wrapped with `&lt;think&gt;` and `&lt;/think&gt;` tags.
diff -- docs_new/docs/hardware-platforms/tpu.mdx
@@ -2,65 +2,67 @@
```

- Reviewed files:
  - docs: `docs_new/docs/supported-models/multimodal_language_models.mdx` renamed +73/-14; `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx` modified +23/-23; `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx` renamed +1809/-463; `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_support_features.mdx` renamed +671/-1418; `docs_new/docs/advanced_features/server_arguments.mdx` modified +446/-486; `docs_new/docs/hardware-platforms/tpu.mdx` modified +425/-468
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-Math-V2.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR-2.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR.mdx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #27342 - test: fix gemma GSM8K thresholds in nightly text eval

- Link: https://github.com/sgl-project/sglang/pull/27342
- Status/date: merged / 2026-06-09
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-4, 20 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "test: fix gemma GSM8K thresholds in nightly text eval"; model line: Llama 3.1; category: bug fix; main diff: `test/registered/eval/test_text_models_gsm8k_eval.py`; technical summary: Covers "test: fix gemma GSM8K thresholds in nightly text eval"; the main implementation surface is `test/registered/eval/test_text_models_gsm8k_eval.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/eval/test_text_models_gsm8k_eval.py` modified +2/-4 (6 lines); hunks: -30,17 +30,15.
- Code diff details:
  - `test/registered/eval/test_text_models_gsm8k_eval.py` modified +2/-4 (6 lines); hunks: -30,17 +30,15
- Key code excerpts:

```diff
diff -- test/registered/eval/test_text_models_gsm8k_eval.py
@@ -30,17 +30,15 @@
-    "google/gemma-2-27b-it": 0.86,  # 90.7% - 5%
+    "google/gemma-2-27b-it": 0.81,  # 85.5% measured - 5%
-    # GSM8K baseline for gemma-2-2b is ~40-45%; threshold set at 5% below.
-    # (Previously 0.50 based on MGSM-EN; tracked regression: https://github.com/sgl-project/sglang/issues/4324)
-    "neuralmagic/gemma-2-2b-it-FP8": 0.38,  # ~43%  - 5%
+    "neuralmagic/gemma-2-2b-it-FP8": 0.53,  # 58.4% measured - 5%
```

- Reviewed files:
  - tests: `test/registered/eval/test_text_models_gsm8k_eval.py` modified +2/-4
- Risk and verification: The diff ships test coverage in `test/registered/eval/test_text_models_gsm8k_eval.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
