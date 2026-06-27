# sglang GPT-OSS Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `benchmark/gpt_oss/README.md` | [#9728](https://github.com/sgl-project/sglang/pull/9728) |
| `docs/basic_usage/gpt_oss.md` | [#9497](https://github.com/sgl-project/sglang/pull/9497), [#9613](https://github.com/sgl-project/sglang/pull/9613), [#9626](https://github.com/sgl-project/sglang/pull/9626) |
| `docs_new/cookbook/autoregressive/OpenAI/GPT-OSS.mdx` | no direct PR-number commit |
| `docs_new/src/snippets/autoregressive/gpt-oss-deployment.jsx` | no direct PR-number commit |
| `python/sglang/srt/function_call/gpt_oss_detector.py` | [#9043](https://github.com/sgl-project/sglang/pull/9043), [#9190](https://github.com/sgl-project/sglang/pull/9190), [#9657](https://github.com/sgl-project/sglang/pull/9657) |
| `python/sglang/srt/models/gpt_oss.py` | [#8824](https://github.com/sgl-project/sglang/pull/8824), [#8843](https://github.com/sgl-project/sglang/pull/8843), [#8944](https://github.com/sgl-project/sglang/pull/8944), [#9028](https://github.com/sgl-project/sglang/pull/9028), [#9146](https://github.com/sgl-project/sglang/pull/9146), [#9161](https://github.com/sgl-project/sglang/pull/9161), [#9359](https://github.com/sgl-project/sglang/pull/9359), [#9433](https://github.com/sgl-project/sglang/pull/9433), [#9469](https://github.com/sgl-project/sglang/pull/9469), [#9783](https://github.com/sgl-project/sglang/pull/9783), [#14197](https://github.com/sgl-project/sglang/pull/14197), [#16775](https://github.com/sgl-project/sglang/pull/16775), ... (19 total) |
| `python/sglang/test/gpt_oss_common.py` | [#16426](https://github.com/sgl-project/sglang/pull/16426) |
| `test/manual/core/test_gpt_oss_1gpu.py` | no direct PR-number commit |
| `test/registered/8-gpu-models/test_gpt_oss_120b.py` | [#18134](https://github.com/sgl-project/sglang/pull/18134) |
| `test/registered/amd/accuracy/mi30x/test_gpt_oss_eval_amd.py` | no direct PR-number commit |
| `test/registered/amd/accuracy/mi35x/test_gpt_oss_eval_mi35x.py` | [#26884](https://github.com/sgl-project/sglang/pull/26884) |
| `test/registered/lora/test_lora_gpt_oss_20b_logprob_diff.py` | [#21570](https://github.com/sgl-project/sglang/pull/21570) |
| `test/registered/models_e2e/test_gpt_oss_4gpu_bf16.py` | no direct PR-number commit |
| `test/registered/models_e2e/test_gpt_oss_4gpu_mxfp4.py` | no direct PR-number commit |
| `test/registered/models_e2e/test_gpt_oss_sm120.py` | no direct PR-number commit |
| `test/registered/perf/test_gpt_oss_4gpu_perf.py` | no direct PR-number commit |

## PR Coverage Summary

- Git-traced PRs: 29
- Extra PRs preserved from existing docs: 16
- Total PRs in this document: 45
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2025-08-05 | [#8824](https://github.com/sgl-project/sglang/pull/8824) | merged | Add initial support for gpt-oss | `python/sglang/srt/models/gpt_oss.py` |
| 2025-08-06 | [#8843](https://github.com/sgl-project/sglang/pull/8843) | merged | Support mxfp4 for GPT-OSS | `python/sglang/srt/models/gpt_oss.py` |
| 2025-08-08 | [#8944](https://github.com/sgl-project/sglang/pull/8944) | merged | Expert Parallelism for GPT-OSS | `python/sglang/srt/models/gpt_oss.py` |
| 2025-08-12 | [#9043](https://github.com/sgl-project/sglang/pull/9043) | merged | (gpt-oss, oai, chat): Remove Harmony Integration and Implement Native GPT-OSS Tool Call Support | `python/sglang/srt/function_call/gpt_oss_detector.py` |
| 2025-08-13 | [#9146](https://github.com/sgl-project/sglang/pull/9146) | merged | Fix gpt-oss ~2x memory consumption issue | `python/sglang/srt/models/gpt_oss.py` |
| 2025-08-13 | [#9028](https://github.com/sgl-project/sglang/pull/9028) | merged | Support FA3 backend for gpt-oss | `python/sglang/srt/models/gpt_oss.py` |
| 2025-08-13 | [#9161](https://github.com/sgl-project/sglang/pull/9161) | merged | Fix broken trtllm_mha attn backend with gpt-oss | `python/sglang/srt/models/gpt_oss.py` |
| 2025-08-20 | [#9359](https://github.com/sgl-project/sglang/pull/9359) | merged | Support DP attention with GPT-OSS | `python/sglang/srt/models/gpt_oss.py` |
| 2025-08-21 | [#9433](https://github.com/sgl-project/sglang/pull/9433) | merged | [fix] Fix mxfp4 weight loading bug with TP sharding in GPT-OSS | `python/sglang/srt/models/gpt_oss.py` |
| 2025-08-22 | [#9469](https://github.com/sgl-project/sglang/pull/9469) | merged | fix: tmp revert gpt oss tp sharding on hopper | `python/sglang/srt/models/gpt_oss.py` |
| 2025-08-22 | [#9497](https://github.com/sgl-project/sglang/pull/9497) | merged | [Docs] Add doc and quick demo for gpt-oss responses api & buildin tools | `docs/basic_usage/gpt_oss.md` |
| 2025-08-25 | [#9190](https://github.com/sgl-project/sglang/pull/9190) | merged | Fix Harmony reasoning parser for and auto-separation for gpt-oss models | `python/sglang/srt/function_call/gpt_oss_detector.py` |
| 2025-08-25 | [#9613](https://github.com/sgl-project/sglang/pull/9613) | merged | [docs] Refactor, remove compiled results and add gpt-oss | `docs/basic_usage/gpt_oss.md` |
| 2025-08-28 | [#9728](https://github.com/sgl-project/sglang/pull/9728) | merged | gpt-oss blog reproduction document | `benchmark/gpt_oss/README.md` |
| 2025-09-01 | [#9783](https://github.com/sgl-project/sglang/pull/9783) | merged | support fp8 kvcache for hybrid attn backend on GPT-OSS | `python/sglang/srt/models/gpt_oss.py` |
| 2025-09-15 | [#9626](https://github.com/sgl-project/sglang/pull/9626) | merged | Add reasoning examples for GPT-OSS in Markdown examples | `docs/basic_usage/gpt_oss.md` |
| 2025-09-15 | [#9657](https://github.com/sgl-project/sglang/pull/9657) | merged | fix: gpt-oss streaming dropping normal content when tools are provided but not used | `python/sglang/srt/function_call/gpt_oss_detector.py` |
| 2025-12-30 | [#14920](https://github.com/sgl-project/sglang/pull/14920) | merged | Eagle: GPT-OSS Eagle v2 support | `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`, `python/sglang/srt/speculative/eagle_worker.py` |
| 2026-01-07 | [#16426](https://github.com/sgl-project/sglang/pull/16426) | merged | Fix gpt_oss_common import path and migrate core tests | `python/sglang/test/gpt_oss_common.py` |
| 2026-01-18 | [#14197](https://github.com/sgl-project/sglang/pull/14197) | merged | [NPU]Support GPT-OSS for NPU | `python/sglang/srt/models/gpt_oss.py` |
| 2026-01-22 | [#17553](https://github.com/sgl-project/sglang/pull/17553) | merged | [NPU] [Bug Fix] Fix typo in npu device check in gpt_oss.py | `python/sglang/srt/models/gpt_oss.py` |
| 2026-02-03 | [#18134](https://github.com/sgl-project/sglang/pull/18134) | merged | feature: adding gpt-oss 120b nightly test | `test/registered/8-gpu-models/test_gpt_oss_120b.py` |
| 2026-02-12 | [#18405](https://github.com/sgl-project/sglang/pull/18405) | merged | [PCG] GPT OSS Triton Kernel Support | `python/sglang/srt/models/gpt_oss.py` |
| 2026-02-16 | [#18869](https://github.com/sgl-project/sglang/pull/18869) | merged | [CI] Remove `--mem-fraction-static 0.93` from gpt-oss test | `test/registered/4-gpu-models/test_gpt_oss_4gpu.py` |
| 2026-02-20 | [#18988](https://github.com/sgl-project/sglang/pull/18988) | merged | [GPT-OSS] support fp8 online quantization for gpt-oss bf16 | `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/server_args.py` |
| 2026-03-06 | [#20056](https://github.com/sgl-project/sglang/pull/20056) | merged | [CI] Add GPT-OSS test for SM120 | `test/registered/core/test_gpt_oss_sm120.py` |
| 2026-03-24 | [#20755](https://github.com/sgl-project/sglang/pull/20755) | merged | Use FlashInfer tinygemm for GPT-OSS MoE router on SM90+ | `python/sglang/srt/models/gpt_oss.py` |
| 2026-04-02 | [#21570](https://github.com/sgl-project/sglang/pull/21570) | merged | [4/n] Support gpt oss 20b lora | `python/sglang/srt/models/gpt_oss.py`, `test/registered/lora/test_lora_gpt_oss_20b_logprob_diff.py` |
| 2026-04-08 | [#22237](https://github.com/sgl-project/sglang/pull/22237) | merged | [CI] Relax gpt-oss 4GPU accuracy threshold from 0.60 to 0.58 | `test/registered/4-gpu-models/test_gpt_oss_4gpu.py` |
| 2026-05-15 | [#25335](https://github.com/sgl-project/sglang/pull/25335) | merged | [Fix] Fix gpt oss triton kernels and upgrade flashinfer back to 0.6.11.post1 | `python/sglang/srt/layers/quantization/mxfp4.py`, `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/layers/flashinfer_comm_fusion.py` |
| 2026-05-20 | [#25831](https://github.com/sgl-project/sglang/pull/25831) | merged | [Test] Stage-a sanity kits; consolidate core/ + models_e2e/ tests | `test/manual/models/test_nvidia_nemotron_3_nano_archived.py`, `python/sglang/test/kits/server_sanity_kit.py`, `python/sglang/test/kits/basic_scheduler_stress_kit.py` |
| 2026-05-24 | [#26205](https://github.com/sgl-project/sglang/pull/26205) | merged | Clean up server startup log noise | `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/layers/attention/flashinfer_backend.py`, `python/sglang/srt/utils/hf_transformers/tokenizer.py` |
| 2026-05-28 | [#26610](https://github.com/sgl-project/sglang/pull/26610) | merged | test/registered: cleanup pure model e2e tests (moves, splits, dedup, kit) | `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py`, `python/sglang/test/kits/unified_radix_cache_kit.py`, `test/registered/models_e2e/test_step3p5_flash_chain_mtp.py` |
| 2026-05-29 | [#16775](https://github.com/sgl-project/sglang/pull/16775) | merged | [CPU] Add GPT-OSS model optimization for CPU | `python/sglang/srt/models/gpt_oss.py` |
| 2026-06-02 | [#25813](https://github.com/sgl-project/sglang/pull/25813) | merged | docs(cookbook): port popular model usage guides into cookbook pages | `docs_new/docs/basic_usage/deepseek_v32.mdx`, `docs_new/docs/basic_usage/deepseek_v3.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx` |
| 2026-06-02 | [#26884](https://github.com/sgl-project/sglang/pull/26884) | merged | [AMD] Fix GPT-OSS MXFP4 accuracy on ROCm AITER path | `test/registered/amd/accuracy/mi35x/test_gpt_oss_eval_mi35x.py` |
| 2026-06-03 | [#27001](https://github.com/sgl-project/sglang/pull/27001) | merged | [AMD] [CI] Remove hardcoded model/cache paths from MI35x nightly tests | `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_kv_fp8_perf_mi35x.py` |
| 2026-06-06 | [#27201](https://github.com/sgl-project/sglang/pull/27201) | merged | [AMD][WA] force to use gate_mode interleaved to fix tp2/tp4/tp8 acc issue | `python/sglang/srt/layers/quantization/mxfp4.py`, `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/server_args.py` |
| 2026-06-08 | [#27063](https://github.com/sgl-project/sglang/pull/27063) | merged | [AMD] Optimize gpt-oss-120B performance | `python/sglang/srt/models/gpt_oss.py` |
| 2026-06-08 | [#27528](https://github.com/sgl-project/sglang/pull/27528) | merged | Fix GPT-OSS MXFP4 hidden size reshape on SM10X | `python/sglang/srt/models/gpt_oss.py` |
| 2026-06-10 | [#23906](https://github.com/sgl-project/sglang/pull/23906) | merged | [Refactor] Cuda Graph Runner/Backend Refactor | `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` |
| 2026-06-11 | [#27964](https://github.com/sgl-project/sglang/pull/27964) | merged | [Spec] Retire Spec V1 | `test/registered/ep/test_deepep_large.py`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx`, `python/sglang/srt/arg_groups/speculative_hook.py` |
| 2026-06-12 | [#27941](https://github.com/sgl-project/sglang/pull/27941) | merged | Enable PDL for GPT-OSS tinygemm router | `python/sglang/srt/models/gpt_oss.py` |
| 2026-06-18 | [#28567](https://github.com/sgl-project/sglang/pull/28567) | merged | Add get_parallel(): a structured accessor for parallel-topology state | `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py` |
| 2026-06-19 | [#28697](https://github.com/sgl-project/sglang/pull/28697) | merged | [docs] Add B300 cookbook deployment options | `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx`, `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx` |

## Per-PR Diff Audit Cards

### PR #8824 - Add initial support for gpt-oss

- Link: https://github.com/sgl-project/sglang/pull/8824
- Status/date: merged / 2025-08-05
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/gpt_oss.py`; associated commits `c1d2061f97ae`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 12 files, +1595/-47, 2185 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add initial support for gpt-oss"; model line: GPT-OSS; category: model support/runtime entry; main diff: `python/sglang/srt/models/gpt_oss.py`; technical summary: Covers "Add initial support for gpt-oss"; the main implementation surface is `python/sglang/srt/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/gpt_oss.py` added +923/-0 (923 lines); hunks: -0,0 +1,923; symbols: GptOssConfig, __init__, get_attention_sliding_window_size, GptOssSparseMoeBlock, touching `GptOssConfig, __init__, get_attention_sliding_window_size`.
- Code diff details:
  - `python/sglang/srt/models/gpt_oss.py` added +923/-0 (923 lines); hunks: -0,0 +1,923; symbols: GptOssConfig, __init__, get_attention_sliding_window_size, GptOssSparseMoeBlock
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -0,0 +1,923 @@
+# Copyright 2023-2024 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/gpt_oss.py` added +923/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/triton_backend.py`, `python/sglang/srt/layers/attention/triton_ops/decode_attention.py`, `python/sglang/srt/layers/attention/triton_ops/extend_attention.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #8843 - Support mxfp4 for GPT-OSS

- Link: https://github.com/sgl-project/sglang/pull/8843
- Status/date: merged / 2025-08-06
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/gpt_oss.py`; associated commits `168033d5fb1e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +791/-325, 1320 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support mxfp4 for GPT-OSS"; model line: GPT-OSS; category: performance/backend optimization; main diff: `python/sglang/srt/models/gpt_oss.py`; technical summary: Covers "Support mxfp4 for GPT-OSS"; the main implementation surface is `python/sglang/srt/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/gpt_oss.py` modified +209/-9 (218 lines); hunks: -25,6 +25,8; -108,11 +110,15 @@ def __init__(; symbols: __init__, _get_default_weight_mapping, load_weights, touching `__init__, _get_default_weight_mapping, load_weights`.
- Code diff details:
  - `python/sglang/srt/models/gpt_oss.py` modified +209/-9 (218 lines); hunks: -25,6 +25,8; -108,11 +110,15 @@ def __init__(; symbols: __init__, _get_default_weight_mapping, load_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -25,6 +25,8 @@
+    get_moe_expert_parallel_rank,
+    get_moe_expert_parallel_world_size,
@@ -108,11 +110,15 @@ def __init__(
+            quant_config_name = (
+                quant_config.get_name() if quant_config is not None else None
+            )
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +209/-9
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py`, `python/sglang/srt/layers/quantization/__init__.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #8944 - Expert Parallelism for GPT-OSS

- Link: https://github.com/sgl-project/sglang/pull/8944
- Status/date: merged / 2025-08-08
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/gpt_oss.py`; associated commits `1d24db834803`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +269/-119, 956 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Expert Parallelism for GPT-OSS"; model line: GPT-OSS; category: model implementation change; main diff: `python/sglang/srt/models/gpt_oss.py`; technical summary: Covers "Expert Parallelism for GPT-OSS"; the main implementation surface is `python/sglang/srt/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/gpt_oss.py` modified +54/-47 (101 lines); hunks: -28,6 +28,7; -96,11 +97,6 @@ def __init__(; symbols: __init__, _load_mxfp4_experts_weights, touching `__init__, _load_mxfp4_experts_weights`.
- Code diff details:
  - `python/sglang/srt/models/gpt_oss.py` modified +54/-47 (101 lines); hunks: -28,6 +28,7; -96,11 +97,6 @@ def __init__(; symbols: __init__, _load_mxfp4_experts_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -28,6 +28,7 @@
+    get_moe_tensor_parallel_world_size,
@@ -96,11 +97,6 @@ def __init__(
-        if self.tp_size > config.num_local_experts:
-            raise ValueError(
-                f"Tensor parallel size {self.tp_size} is greater than "
-                f"the number of experts {config.num_local_experts}."
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +54/-47
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9043 - (gpt-oss, oai, chat): Remove Harmony Integration and Implement Native GPT-OSS Tool Call Support

- Link: https://github.com/sgl-project/sglang/pull/9043
- Status/date: merged / 2025-08-12
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/function_call/gpt_oss_detector.py`; associated commits `a21849013607`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +717/-409, 1293 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "(gpt-oss, oai, chat): Remove Harmony Integration and Implement Native GPT-OSS Tool Call Support"; model line: GPT-OSS; category: model support/runtime entry; main diff: `python/sglang/srt/function_call/gpt_oss_detector.py`; technical summary: Covers "(gpt-oss, oai, chat): Remove Harmony Integration and Implement Native GPT-OSS Tool Call Support"; the main implementation surface is `python/sglang/srt/function_call/gpt_oss_detector.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/function_call/gpt_oss_detector.py` added +331/-0 (331 lines); hunks: -0,0 +1,331; symbols: GptOssDetector, __init__, has_tool_call, detect_and_parse, touching `GptOssDetector, __init__, has_tool_call`.
- Code diff details:
  - `python/sglang/srt/function_call/gpt_oss_detector.py` added +331/-0 (331 lines); hunks: -0,0 +1,331; symbols: GptOssDetector, __init__, has_tool_call, detect_and_parse
- Key code excerpts:

```diff
diff -- python/sglang/srt/function_call/gpt_oss_detector.py
@@ -0,0 +1,331 @@
+import json
+import logging
+import re
+from typing import List
+from sglang.srt.entrypoints.openai.protocol import Tool
+from sglang.srt.function_call.base_format_detector import BaseFormatDetector
```

- Reviewed files:
  - runtime: `python/sglang/srt/function_call/gpt_oss_detector.py` added +331/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/entrypoints/harmony_utils.py`, `python/sglang/srt/entrypoints/http_server.py`, `python/sglang/srt/entrypoints/openai/protocol.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9146 - Fix gpt-oss ~2x memory consumption issue

- Link: https://github.com/sgl-project/sglang/pull/9146
- Status/date: merged / 2025-08-13
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/gpt_oss.py`; associated commits `9394ed63867d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +19/-7, 47 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix gpt-oss ~2x memory consumption issue"; model line: GPT-OSS; category: bug fix; main diff: `python/sglang/srt/models/gpt_oss.py`; technical summary: Covers "Fix gpt-oss ~2x memory consumption issue"; the main implementation surface is `python/sglang/srt/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/gpt_oss.py` modified +19/-7 (26 lines); hunks: -64,7 +64,13; -655,6 +661,18 @@ def __init__(; symbols: __init__, routed_experts_weights_of_layer, forward, _load_normal_weights, touching `__init__, routed_experts_weights_of_layer, forward`.
- Code diff details:
  - `python/sglang/srt/models/gpt_oss.py` modified +19/-7 (26 lines); hunks: -64,7 +64,13; -655,6 +661,18 @@ def __init__(; symbols: __init__, routed_experts_weights_of_layer, forward, _load_normal_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -64,7 +64,13 @@
-from sglang.srt.utils import add_prefix, is_cuda, is_flashinfer_available, make_layers
+from sglang.srt.utils import (
+    LazyValue,
+    add_prefix,
+    is_cuda,
+    is_flashinfer_available,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +19/-7
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/gpt_oss.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9028 - Support FA3 backend for gpt-oss

- Link: https://github.com/sgl-project/sglang/pull/9028
- Status/date: merged / 2025-08-13
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/gpt_oss.py`; associated commits `0ff6d1fce122`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +24/-6, 121 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support FA3 backend for gpt-oss"; model line: GPT-OSS; category: model support/runtime entry; main diff: `python/sglang/srt/models/gpt_oss.py`; technical summary: Covers "Support FA3 backend for gpt-oss"; the main implementation surface is `python/sglang/srt/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -294,7 +294,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -294,7 +294,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -294,7 +294,7 @@ def __init__(
-            torch.empty(self.num_heads, dtype=torch.float32), requires_grad=False
+            torch.empty(self.num_heads, dtype=torch.bfloat16), requires_grad=False
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/pyproject.toml`, `python/sglang/srt/layers/attention/flashattention_backend.py`, `python/sglang/srt/models/gpt_oss.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9161 - Fix broken trtllm_mha attn backend with gpt-oss

- Link: https://github.com/sgl-project/sglang/pull/9161
- Status/date: merged / 2025-08-13
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/gpt_oss.py`; associated commits `6b7c24712cda`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-1, 14 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix broken trtllm_mha attn backend with gpt-oss"; model line: GPT-OSS; category: bug fix; main diff: `python/sglang/srt/models/gpt_oss.py`; technical summary: Covers "Fix broken trtllm_mha attn backend with gpt-oss"; the main implementation surface is `python/sglang/srt/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/gpt_oss.py` modified +5/-1 (6 lines); hunks: -293,8 +293,12 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/gpt_oss.py` modified +5/-1 (6 lines); hunks: -293,8 +293,12 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -293,8 +293,12 @@ def __init__(
+        # Choose dtype of sinks based on attention backend: trtllm_mha requires float32,
+        # others can use bfloat16
+        attn_backend = global_server_args_dict.get("attention_backend")
+        sinks_dtype = torch.float32 if attn_backend == "trtllm_mha" else torch.bfloat16
-            torch.empty(self.num_heads, dtype=torch.bfloat16), requires_grad=False
+            torch.empty(self.num_heads, dtype=sinks_dtype), requires_grad=False
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +5/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/gpt_oss.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9359 - Support DP attention with GPT-OSS

- Link: https://github.com/sgl-project/sglang/pull/9359
- Status/date: merged / 2025-08-20
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/gpt_oss.py`; associated commits `c10b8e6a0f2a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +6/-5, 25 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support DP attention with GPT-OSS"; model line: GPT-OSS; category: model support/runtime entry; main diff: `python/sglang/srt/models/gpt_oss.py`; technical summary: Covers "Support DP attention with GPT-OSS"; the main implementation surface is `python/sglang/srt/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -1123,7 +1123,7 @@ def _load_normal_weights(; symbols: _load_normal_weights, touching `_load_normal_weights`.
- Code diff details:
  - `python/sglang/srt/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -1123,7 +1123,7 @@ def _load_normal_weights(; symbols: _load_normal_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -1123,7 +1123,7 @@ def _load_normal_weights(
-                            start = tp_rank * param.numel()
+                            start = get_attention_tp_rank() * param.numel()
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/gpt_oss.py`, `python/sglang/srt/server_args.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9433 - [fix] Fix mxfp4 weight loading bug with TP sharding in GPT-OSS

- Link: https://github.com/sgl-project/sglang/pull/9433
- Status/date: merged / 2025-08-21
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/gpt_oss.py`; associated commits `dae9a80f43e8`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +11/-3, 46 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[fix] Fix mxfp4 weight loading bug with TP sharding in GPT-OSS"; model line: GPT-OSS; category: bug fix; main diff: `python/sglang/srt/models/gpt_oss.py`; technical summary: Covers "[fix] Fix mxfp4 weight loading bug with TP sharding in GPT-OSS"; the main implementation surface is `python/sglang/srt/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/gpt_oss.py` modified +9/-1 (10 lines); hunks: -16,6 +16,7; -788,18 +789,25 @@ def _load_mxfp4_experts_weights(self, weights):; symbols: _load_mxfp4_experts_weights, touching `_load_mxfp4_experts_weights`.
- Code diff details:
  - `python/sglang/srt/models/gpt_oss.py` modified +9/-1 (10 lines); hunks: -16,6 +16,7; -788,18 +789,25 @@ def _load_mxfp4_experts_weights(self, weights):; symbols: _load_mxfp4_experts_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -16,6 +16,7 @@
+import math
@@ -788,18 +789,25 @@ def _load_mxfp4_experts_weights(self, weights):
+        assert (
+            intermediate_size % mxfp4_block == 0
+        ), f"{intermediate_size=} must be divisible by {mxfp4_block=}"
-        per_rank_intermediate_size_block = intermediate_size_block // moe_tp_size
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +9/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/entrypoints/openai/protocol.py`, `python/sglang/srt/models/gpt_oss.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9469 - fix: tmp revert gpt oss tp sharding on hopper

- Link: https://github.com/sgl-project/sglang/pull/9469
- Status/date: merged / 2025-08-22
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/gpt_oss.py`; associated commits `849957bc76c3`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +6/-3, 16 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix: tmp revert gpt oss tp sharding on hopper"; model line: GPT-OSS; category: bug fix; main diff: `python/sglang/srt/models/gpt_oss.py`; technical summary: Covers "fix: tmp revert gpt oss tp sharding on hopper"; the main implementation surface is `python/sglang/srt/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/gpt_oss.py` modified +6/-3 (9 lines); hunks: -793,9 +793,12 @@ def _load_mxfp4_experts_weights(self, weights):; symbols: _load_mxfp4_experts_weights, touching `_load_mxfp4_experts_weights`.
- Code diff details:
  - `python/sglang/srt/models/gpt_oss.py` modified +6/-3 (9 lines); hunks: -793,9 +793,12 @@ def _load_mxfp4_experts_weights(self, weights):; symbols: _load_mxfp4_experts_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -793,9 +793,12 @@ def _load_mxfp4_experts_weights(self, weights):
-        per_rank_intermediate_size_block = math.ceil(
-            intermediate_size_block / moe_tp_size
-        )
+        if _is_sm100_supported:
+            per_rank_intermediate_size_block = math.ceil(
+                intermediate_size_block / moe_tp_size
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +6/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/gpt_oss.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9497 - [Docs] Add doc and quick demo for gpt-oss responses api & buildin tools

- Link: https://github.com/sgl-project/sglang/pull/9497
- Status/date: merged / 2025-08-22
- Trace source: `git log --name-only -- <model-files>` found it through `docs/basic_usage/gpt_oss.md`; associated commits `fedfe91c1a6e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +106/-0, 110 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Docs] Add doc and quick demo for gpt-oss responses api & buildin tools"; model line: GPT-OSS; category: docs/tests/CI; main diff: `docs/basic_usage/gpt_oss.md`; technical summary: Covers "[Docs] Add doc and quick demo for gpt-oss responses api & buildin tools"; the main implementation surface is `docs/basic_usage/gpt_oss.md`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs/basic_usage/gpt_oss.md` modified +106/-0 (106 lines); hunks: -1,3 +1,109.
- Code diff details:
  - `docs/basic_usage/gpt_oss.md` modified +106/-0 (106 lines); hunks: -1,3 +1,109
- Key code excerpts:

```diff
diff -- docs/basic_usage/gpt_oss.md
@@ -1,3 +1,109 @@
+## Responses API & Built-in Tools
+### Responses API
+GPT‑OSS is compatible with the OpenAI Responses API. Use `client.responses.create(...)` with `model`, `instructions`, `input`, and optional `tools` to enable built‑in tool use.
+### Built-in Tools
+GPT‑OSS can call built‑in tools for web search and Python execution. You can use the demo tool server or connect to external MCP tool servers.
+#### Python Tool
```

- Reviewed files:
  - docs: `docs/basic_usage/gpt_oss.md` modified +106/-0
- Risk and verification: This is mostly docs/examples in `docs/basic_usage/gpt_oss.md`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #9190 - Fix Harmony reasoning parser for and auto-separation for gpt-oss models

- Link: https://github.com/sgl-project/sglang/pull/9190
- Status/date: merged / 2025-08-25
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/function_call/gpt_oss_detector.py`; associated commits `a0a77d937b99`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +1681/-556, 2406 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix Harmony reasoning parser for and auto-separation for gpt-oss models"; model line: GPT-OSS; category: bug fix; main diff: `python/sglang/srt/function_call/gpt_oss_detector.py`; technical summary: Covers "Fix Harmony reasoning parser for and auto-separation for gpt-oss models"; the main implementation surface is `python/sglang/srt/function_call/gpt_oss_detector.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/function_call/gpt_oss_detector.py` modified +144/-256 (400 lines); hunks: -1,7 +1,7; -10,60 +10,31; symbols: GptOssDetector, __init__, has_tool_call, detect_and_parse, touching `GptOssDetector, __init__, has_tool_call`.
- Code diff details:
  - `python/sglang/srt/function_call/gpt_oss_detector.py` modified +144/-256 (400 lines); hunks: -1,7 +1,7; -10,60 +10,31; symbols: GptOssDetector, __init__, has_tool_call, detect_and_parse
- Key code excerpts:

```diff
diff -- python/sglang/srt/function_call/gpt_oss_detector.py
@@ -1,7 +1,7 @@
-from typing import List
+from typing import List, Optional
@@ -10,60 +10,31 @@
+from sglang.srt.harmony_parser import HarmonyParser
-    Detector for T4-style function calls with channel format.
+    Detector for T4-style function calls using HarmonyParser.
```

- Reviewed files:
  - runtime: `python/sglang/srt/function_call/gpt_oss_detector.py` modified +144/-256
- Risk and verification: The diff ships test coverage in `test/srt/run_suite.py`, `test/srt/test_harmony_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #9613 - [docs] Refactor, remove compiled results and add gpt-oss

- Link: https://github.com/sgl-project/sglang/pull/9613
- Status/date: merged / 2025-08-25
- Trace source: `git log --name-only -- <model-files>` found it through `docs/basic_usage/gpt_oss.md`; associated commits `9b08d975a0a5`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +166/-611, 638 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[docs] Refactor, remove compiled results and add gpt-oss"; model line: GPT-OSS; category: docs/tests/CI; main diff: `docs/basic_usage/gpt_oss.md`; technical summary: Covers "[docs] Refactor, remove compiled results and add gpt-oss"; the main implementation surface is `docs/basic_usage/gpt_oss.md`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs/basic_usage/gpt_oss.md` modified +5/-0 (5 lines); hunks: -23,6 +23,11 @@ GPT‑OSS can call built‑in tools for web search and Python exe....
- Code diff details:
  - `docs/basic_usage/gpt_oss.md` modified +5/-0 (5 lines); hunks: -23,6 +23,11 @@ GPT‑OSS can call built‑in tools for web search and Python exe...
- Key code excerpts:

```diff
diff -- docs/basic_usage/gpt_oss.md
@@ -23,6 +23,11 @@ GPT‑OSS can call built‑in tools for web search and Python execution. You can
+### Tool & Reasoning Parser
+- We support OpenAI Reasoning and Tool Call parser, as well as our SGLang native api for tool call and reasoning. Refer to [reasoning parser](../advanced_features/separate_reasoni
```

- Reviewed files:
  - docs: `docs/basic_usage/gpt_oss.md` modified +5/-0
- Risk and verification: This is mostly docs/examples in `docs/basic_usage/gpt_oss.md`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #9728 - gpt-oss blog reproduction document

- Link: https://github.com/sgl-project/sglang/pull/9728
- Status/date: merged / 2025-08-28
- Trace source: `git log --name-only -- <model-files>` found it through `benchmark/gpt_oss/README.md`; associated commits `d0934a519257`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +163/-0, 164 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "gpt-oss blog reproduction document"; model line: GPT-OSS; category: docs/tests/CI; main diff: `benchmark/gpt_oss/README.md`; technical summary: Covers "gpt-oss blog reproduction document"; the main implementation surface is `benchmark/gpt_oss/README.md`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `benchmark/gpt_oss/README.md` added +163/-0 (163 lines); hunks: -0,0 +1,163.
- Code diff details:
  - `benchmark/gpt_oss/README.md` added +163/-0 (163 lines); hunks: -0,0 +1,163
- Key code excerpts:

```diff
diff -- benchmark/gpt_oss/README.md
@@ -0,0 +1,163 @@
+# How to reproduce the result of GPT-OSS with SGLang
+### Install the latest SGLang
+'''bash
+git clone https://github.com/sgl-project/sglang.git
+cd sglang
+git checkout v0.5.1.post3
```

- Reviewed files:
  - other: `benchmark/gpt_oss/README.md` added +163/-0
- Risk and verification: No explicit test file appears in the diff; future edits should add or run model loading, short generation, and parser/multimodal regression checks.

### PR #9783 - support fp8 kvcache for hybrid attn backend on GPT-OSS

- Link: https://github.com/sgl-project/sglang/pull/9783
- Status/date: merged / 2025-09-01
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/gpt_oss.py`; associated commits `9db8025376b2`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-4, 30 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "support fp8 kvcache for hybrid attn backend on GPT-OSS"; model line: GPT-OSS; category: performance/backend optimization; main diff: `python/sglang/srt/models/gpt_oss.py`; technical summary: Covers "support fp8 kvcache for hybrid attn backend on GPT-OSS"; the main implementation surface is `python/sglang/srt/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/gpt_oss.py` modified +5/-4 (9 lines); hunks: -193,8 +193,9 @@ def forward_normal(; -341,7 +342,7 @@ def forward_prepare(; symbols: forward_normal, _enable_fused_set_kv_buffer, forward_prepare, forward_core, touching `forward_normal, _enable_fused_set_kv_buffer, forward_prepare`.
- Code diff details:
  - `python/sglang/srt/models/gpt_oss.py` modified +5/-4 (9 lines); hunks: -193,8 +193,9 @@ def forward_normal(; -341,7 +342,7 @@ def forward_prepare(; symbols: forward_normal, _enable_fused_set_kv_buffer, forward_prepare, forward_core
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -193,8 +193,9 @@ def forward_normal(
-def _enable_fused_set_kv_buffer():
-    return _is_cuda
+def _enable_fused_set_kv_buffer(forward_batch: ForwardBatch):
+    """Enable fused set_kv_buffer only on CUDA with bfloat16 KV cache."""
+    return _is_cuda and forward_batch.token_to_kv_pool.dtype == torch.bfloat16
@@ -341,7 +342,7 @@ def forward_prepare(
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +5/-4
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/gpt_oss.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9626 - Add reasoning examples for GPT-OSS in Markdown examples

- Link: https://github.com/sgl-project/sglang/pull/9626
- Status/date: merged / 2025-09-15
- Trace source: `git log --name-only -- <model-files>` found it through `docs/basic_usage/gpt_oss.md`; associated commits `0b14159fc4e0`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +12/-2, 35 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add reasoning examples for GPT-OSS in Markdown examples"; model line: GPT-OSS; category: docs/tests/CI; main diff: `docs/basic_usage/gpt_oss.md`; technical summary: Covers "Add reasoning examples for GPT-OSS in Markdown examples"; the main implementation surface is `docs/basic_usage/gpt_oss.md`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs/basic_usage/gpt_oss.md` modified +11/-1 (12 lines); hunks: -6,7 +6,7 @@ Please refer to [https://github.com/sgl-project/sglang/issues/88...; -69,6 +69,16 @@ tools = [.
- Code diff details:
  - `docs/basic_usage/gpt_oss.md` modified +11/-1 (12 lines); hunks: -6,7 +6,7 @@ Please refer to [https://github.com/sgl-project/sglang/issues/88...; -69,6 +69,16 @@ tools = [
- Key code excerpts:

```diff
diff -- docs/basic_usage/gpt_oss.md
@@ -6,7 +6,7 @@ Please refer to [https://github.com/sgl-project/sglang/issues/8833](https://gith
-GPT‑OSS is compatible with the OpenAI Responses API. Use `client.responses.create(...)` with `model`, `instructions`, `input`, and optional `tools` to enable built‑in tool use.
+GPT‑OSS is compatible with the OpenAI Responses API. Use `client.responses.create(...)` with `model`, `instructions`, `input`, and optional `tools` to enable built‑in tool use. Yo
@@ -69,6 +69,16 @@ tools = [
+# Reasoning level example
+response = client.responses.create(
+    model="openai/gpt-oss-120b",
```

- Reviewed files:
  - docs: `docs/basic_usage/gpt_oss.md` modified +11/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/entrypoints/openai/protocol.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9657 - fix: gpt-oss streaming dropping normal content when tools are provided but not used

- Link: https://github.com/sgl-project/sglang/pull/9657
- Status/date: merged / 2025-09-15
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/function_call/gpt_oss_detector.py`; associated commits `28c79dc84ab8`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +23/-0, 30 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix: gpt-oss streaming dropping normal content when tools are provided but not used"; model line: GPT-OSS; category: bug fix; main diff: `python/sglang/srt/function_call/gpt_oss_detector.py`; technical summary: Covers "fix: gpt-oss streaming dropping normal content when tools are provided but not used"; the main implementation surface is `python/sglang/srt/function_call/gpt_oss_detector.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/function_call/gpt_oss_detector.py` modified +23/-0 (23 lines); hunks: -81,6 +81,29 @@ def parse_streaming_increment(; symbols: parse_streaming_increment, touching `parse_streaming_increment`.
- Code diff details:
  - `python/sglang/srt/function_call/gpt_oss_detector.py` modified +23/-0 (23 lines); hunks: -81,6 +81,29 @@ def parse_streaming_increment(; symbols: parse_streaming_increment
- Key code excerpts:

```diff
diff -- python/sglang/srt/function_call/gpt_oss_detector.py
@@ -81,6 +81,29 @@ def parse_streaming_increment(
+        # If there are no parsed events and the chunk contains no Harmony structural
+        # markers, treat it as plain text and pass it through. This fixes a bug where
+        # normal content was held in the buffer when tools were provided but not used.
+        if not events:
+            has_harmony_markers = any(
+                marker in self._buffer
```

- Reviewed files:
  - runtime: `python/sglang/srt/function_call/gpt_oss_detector.py` modified +23/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/function_call/gpt_oss_detector.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #14920 - Eagle: GPT-OSS Eagle v2 support

- Link: https://github.com/sgl-project/sglang/pull/14920
- Status/date: merged / 2025-12-30
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +48/-25, 124 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Eagle: GPT-OSS Eagle v2 support"; model line: GPT-OSS; category: performance/backend optimization; main diff: `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`, `python/sglang/srt/speculative/eagle_worker.py`; technical summary: Covers "Eagle: GPT-OSS Eagle v2 support"; the main implementation surface is `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`, `python/sglang/srt/speculative/eagle_worker.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/model_executor/model_runner.py` modified +30/-23 (53 lines); hunks: -345,6 +345,32 @@ def __init__(; -593,30 +619,11 @@ def initialize(self, min_per_gpu_memory: float):; symbols: __init__, initialize, _dummy_run, touching `__init__, initialize, _dummy_run`; `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +4/-1 (5 lines); hunks: -349,7 +349,10 @@ def __init__(self, model_runner: ModelRunner):; symbols: __init__, touching `__init__`; `python/sglang/srt/speculative/eagle_worker.py` modified +10/-0 (10 lines); hunks: -186,6 +186,15 @@ def __init__(; -897,6 +906,7 @@ def forward_draft_extend_after_decode(self, batch: ScheduleB...; symbols: __init__, forward_draft_extend_after_decode, touching `__init__, forward_draft_extend_after_decode`; `python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py` modified +4/-1 (5 lines); hunks: -100,7 +100,10 @@ def __init__(self, eagle_worker: EAGLEWorker):; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/model_executor/model_runner.py` modified +30/-23 (53 lines); hunks: -345,6 +345,32 @@ def __init__(; -593,30 +619,11 @@ def initialize(self, min_per_gpu_memory: float):; symbols: __init__, initialize, _dummy_run
  - `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +4/-1 (5 lines); hunks: -349,7 +349,10 @@ def __init__(self, model_runner: ModelRunner):; symbols: __init__
  - `python/sglang/srt/speculative/eagle_worker.py` modified +10/-0 (10 lines); hunks: -186,6 +186,15 @@ def __init__(; -897,6 +906,7 @@ def forward_draft_extend_after_decode(self, batch: ScheduleB...; symbols: __init__, forward_draft_extend_after_decode
  - `python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py` modified +4/-1 (5 lines); hunks: -100,7 +100,10 @@ def __init__(self, eagle_worker: EAGLEWorker):; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -345,6 +345,32 @@ def __init__(
+        # auxiliary hidden capture mode. TODO: expose this to server args?
+        self.eagle_use_aux_hidden_state = False
+        if self.spec_algorithm.is_eagle3() and not self.is_draft_worker:
+            # load draft config
+            draft_model_config = ModelConfig.from_server_args(
+                server_args,
diff -- python/sglang/srt/model_executor/cuda_graph_runner.py
@@ -349,7 +349,10 @@ def __init__(self, model_runner: ModelRunner):
-        if model_runner.spec_algorithm.is_eagle3():
+        if (
+            model_runner.spec_algorithm.is_eagle3()
+            and model_runner.eagle_use_aux_hidden_state
+        ):
diff -- python/sglang/srt/speculative/eagle_worker.py
@@ -186,6 +186,15 @@ def __init__(
+        self.eagle_use_aux_hidden_state = False
```

- Reviewed files:
  - runtime: `python/sglang/srt/model_executor/model_runner.py` modified +30/-23; `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +4/-1; `python/sglang/srt/speculative/eagle_worker.py` modified +10/-0; `python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py` modified +4/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/model_executor/cuda_graph_runner.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #16426 - Fix gpt_oss_common import path and migrate core tests

- Link: https://github.com/sgl-project/sglang/pull/16426
- Status/date: merged / 2026-01-07
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/test/gpt_oss_common.py`; associated commits `0c474273c514`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 15 files, +48/-26, 255 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix gpt_oss_common import path and migrate core tests"; model line: GPT-OSS; category: bug fix; main diff: `python/sglang/test/gpt_oss_common.py`; technical summary: Covers "Fix gpt_oss_common import path and migrate core tests"; the main implementation surface is `python/sglang/test/gpt_oss_common.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/test/gpt_oss_common.py` renamed +0/-0 (0 lines).
- Code diff details:
  - `python/sglang/test/gpt_oss_common.py` renamed +0/-0 (0 lines)
- Key code excerpts:

```diff
No textual patch was returned by GitHub for the selected changed files.
```

- Reviewed files:
  - tests: `python/sglang/test/gpt_oss_common.py` renamed +0/-0
- Risk and verification: The diff ships test coverage in `python/sglang/test/gpt_oss_common.py`, `test/registered/core/test_deterministic.py`, `test/registered/core/test_gpt_oss_1gpu.py`, `test/registered/core/test_hidden_states.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #14197 - [NPU]Support GPT-OSS for NPU

- Link: https://github.com/sgl-project/sglang/pull/14197
- Status/date: merged / 2026-01-18
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/gpt_oss.py`; associated commits `733de6be31e2`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +96/-17, 244 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NPU]Support GPT-OSS for NPU"; model line: GPT-OSS; category: model support/runtime entry; main diff: `python/sglang/srt/models/gpt_oss.py`; technical summary: Covers "[NPU]Support GPT-OSS for NPU"; the main implementation surface is `python/sglang/srt/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/gpt_oss.py` modified +20/-15 (35 lines); hunks: -71,9 +71,10; -129,6 +130,7 @@ def __init__(; symbols: __init__, forward_prepare, touching `__init__, forward_prepare`.
- Code diff details:
  - `python/sglang/srt/models/gpt_oss.py` modified +20/-15 (35 lines); hunks: -71,9 +71,10; -129,6 +130,7 @@ def __init__(; symbols: __init__, forward_prepare
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -71,9 +71,10 @@
-from sglang.srt.utils import LazyValue, add_prefix, is_cuda, make_layers
+from sglang.srt.utils import LazyValue, add_prefix, is_cuda, is_npu, make_layers
+_is_npu = is_npu()
@@ -129,6 +130,7 @@ def __init__(
@@ -305,20 +307,20 @@ def forward_prepare(
-        q, k = self.rotary_emb(
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +20/-15
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/layers/quantization/unquant.py`, `python/sglang/srt/models/gpt_oss.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17553 - [NPU] [Bug Fix] Fix typo in npu device check in gpt_oss.py

- Link: https://github.com/sgl-project/sglang/pull/17553
- Status/date: merged / 2026-01-22
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/gpt_oss.py`; associated commits `61abff66c150`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NPU] [Bug Fix] Fix typo in npu device check in gpt_oss.py"; model line: GPT-OSS; category: bug fix; main diff: `python/sglang/srt/models/gpt_oss.py`; technical summary: Covers "[NPU] [Bug Fix] Fix typo in npu device check in gpt_oss.py"; the main implementation surface is `python/sglang/srt/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -492,7 +492,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/gpt_oss.py` modified +1/-1 (2 lines); hunks: -492,7 +492,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -492,7 +492,7 @@ def __init__(
-        if is_npu:
+        if _is_npu:
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/gpt_oss.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18134 - feature: adding gpt-oss 120b nightly test

- Link: https://github.com/sgl-project/sglang/pull/18134
- Status/date: merged / 2026-02-03
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/8-gpu-models/test_gpt_oss_120b.py`; associated commits `c8da307d7e63`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +88/-4, 121 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "feature: adding gpt-oss 120b nightly test"; model line: GPT-OSS; category: docs/tests/CI; main diff: `test/registered/8-gpu-models/test_gpt_oss_120b.py`; technical summary: Covers "feature: adding gpt-oss 120b nightly test"; the main implementation surface is `test/registered/8-gpu-models/test_gpt_oss_120b.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/8-gpu-models/test_gpt_oss_120b.py` added +84/-0 (84 lines); hunks: -0,0 +1,84; symbols: TestGptOss120B, for, test_gpt_oss_120b_all_variants, touching `TestGptOss120B, for, test_gpt_oss_120b_all_variants`.
- Code diff details:
  - `test/registered/8-gpu-models/test_gpt_oss_120b.py` added +84/-0 (84 lines); hunks: -0,0 +1,84; symbols: TestGptOss120B, for, test_gpt_oss_120b_all_variants
- Key code excerpts:

```diff
diff -- test/registered/8-gpu-models/test_gpt_oss_120b.py
@@ -0,0 +1,84 @@
+import unittest
+from sglang.test.ci.ci_register import register_cuda_ci
+from sglang.test.performance_test_runner import PerformanceTestParams
+from sglang.test.run_combined_tests import run_combined_tests
+from sglang.test.test_utils import ModelLaunchSettings
+# Runs on both H200 and B200 via nightly-8-gpu-common suite
```

- Reviewed files:
  - tests: `test/registered/8-gpu-models/test_gpt_oss_120b.py` added +84/-0
- Risk and verification: The diff ships test coverage in `test/registered/8-gpu-models/test_gpt_oss_120b.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18405 - [PCG] GPT OSS Triton Kernel Support

- Link: https://github.com/sgl-project/sglang/pull/18405
- Status/date: merged / 2026-02-12
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/gpt_oss.py`; associated commits `2bd8363486e4`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +68/-32, 228 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[PCG] GPT OSS Triton Kernel Support"; model line: GPT-OSS; category: performance/backend optimization; main diff: `python/sglang/srt/models/gpt_oss.py`; technical summary: Covers "[PCG] GPT OSS Triton Kernel Support"; the main implementation surface is `python/sglang/srt/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/gpt_oss.py` modified +21/-4 (25 lines); hunks: -25,6 +25,10; -72,6 +76,7; symbols: forward_normal, moe_impl, GptOssAttention, __init__, touching `forward_normal, moe_impl, GptOssAttention`.
- Code diff details:
  - `python/sglang/srt/models/gpt_oss.py` modified +21/-4 (25 lines); hunks: -25,6 +25,10; -72,6 +76,7; symbols: forward_normal, moe_impl, GptOssAttention, __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -25,6 +25,10 @@
+from sglang.srt.compilation.piecewise_context_manager import (
+    get_forward_context,
+    is_in_piecewise_cuda_graph,
+)
@@ -72,6 +76,7 @@
+from sglang.srt.utils.custom_op import register_custom_op
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +21/-4
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/compilation/piecewise_context_manager.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/model_executor/model_runner.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18869 - [CI] Remove `--mem-fraction-static 0.93` from gpt-oss test

- Link: https://github.com/sgl-project/sglang/pull/18869
- Status/date: merged / 2026-02-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +0/-2, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CI] Remove `--mem-fraction-static 0.93` from gpt-oss test"; model line: GPT-OSS; category: docs/tests/CI; main diff: `test/registered/4-gpu-models/test_gpt_oss_4gpu.py`; technical summary: Covers "[CI] Remove `--mem-fraction-static 0.93` from gpt-oss test"; the main implementation surface is `test/registered/4-gpu-models/test_gpt_oss_4gpu.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/4-gpu-models/test_gpt_oss_4gpu.py` modified +0/-2 (2 lines); hunks: -30,8 +30,6 @@ def test_mxfp4_120b(self):; symbols: test_mxfp4_120b, touching `test_mxfp4_120b`.
- Code diff details:
  - `test/registered/4-gpu-models/test_gpt_oss_4gpu.py` modified +0/-2 (2 lines); hunks: -30,8 +30,6 @@ def test_mxfp4_120b(self):; symbols: test_mxfp4_120b
- Key code excerpts:

```diff
diff -- test/registered/4-gpu-models/test_gpt_oss_4gpu.py
@@ -30,8 +30,6 @@ def test_mxfp4_120b(self):
-                "--mem-fraction-static",
-                "0.93",
```

- Reviewed files:
  - tests: `test/registered/4-gpu-models/test_gpt_oss_4gpu.py` modified +0/-2
- Risk and verification: The diff ships test coverage in `test/registered/4-gpu-models/test_gpt_oss_4gpu.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18988 - [GPT-OSS] support fp8 online quantization for gpt-oss bf16

- Link: https://github.com/sgl-project/sglang/pull/18988
- Status/date: merged / 2026-02-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +31/-1, 69 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[GPT-OSS] support fp8 online quantization for gpt-oss bf16"; model line: GPT-OSS; category: performance/backend optimization; main diff: `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/server_args.py`; technical summary: Covers "[GPT-OSS] support fp8 online quantization for gpt-oss bf16"; the main implementation surface is `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/server_args.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/quantization/fp8.py` modified +26/-0 (26 lines); hunks: -677,6 +677,7 @@ def __init__(self, quant_config: Fp8Config):; -706,8 +707,10 @@ def create_weights(; symbols: __init__, create_weights, apply, touching `__init__, create_weights, apply`; `python/sglang/srt/server_args.py` modified +5/-1 (6 lines); hunks: -1386,7 +1386,11 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments, touching `_handle_model_specific_adjustments`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/fp8.py` modified +26/-0 (26 lines); hunks: -677,6 +677,7 @@ def __init__(self, quant_config: Fp8Config):; -706,8 +707,10 @@ def create_weights(; symbols: __init__, create_weights, apply
  - `python/sglang/srt/server_args.py` modified +5/-1 (6 lines); hunks: -1386,7 +1386,11 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/fp8.py
@@ -677,6 +677,7 @@ def __init__(self, quant_config: Fp8Config):
+        self.with_bias = False
@@ -706,8 +707,10 @@ def create_weights(
+        with_bias: bool = False,
+        self.with_bias = with_bias
@@ -782,6 +785,27 @@ def create_weights(
+        # BIAS (optional, e.g. GPT-OSS)
diff -- python/sglang/srt/server_args.py
@@ -1386,7 +1386,11 @@ def _handle_model_specific_adjustments(self):
-                elif self.ep_size == 1 and is_triton_kernels_available():
+                elif (
+                    self.ep_size == 1
+                    and is_triton_kernels_available()
+                    and self.quantization is None
+                ):
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/fp8.py` modified +26/-0; `python/sglang/srt/server_args.py` modified +5/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/server_args.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #20056 - [CI] Add GPT-OSS test for SM120

- Link: https://github.com/sgl-project/sglang/pull/20056
- Status/date: merged / 2026-03-06
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +34/-0, 35 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CI] Add GPT-OSS test for SM120"; model line: GPT-OSS; category: docs/tests/CI; main diff: `test/registered/core/test_gpt_oss_sm120.py`; technical summary: Covers "[CI] Add GPT-OSS test for SM120"; the main implementation surface is `test/registered/core/test_gpt_oss_sm120.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/core/test_gpt_oss_sm120.py` added +34/-0 (34 lines); hunks: -0,0 +1,34; symbols: TestGptOssSm120, setUpClass, test_mxfp4_20b, touching `TestGptOssSm120, setUpClass, test_mxfp4_20b`.
- Code diff details:
  - `test/registered/core/test_gpt_oss_sm120.py` added +34/-0 (34 lines); hunks: -0,0 +1,34; symbols: TestGptOssSm120, setUpClass, test_mxfp4_20b
- Key code excerpts:

```diff
diff -- test/registered/core/test_gpt_oss_sm120.py
@@ -0,0 +1,34 @@
+import unittest
+import torch
+from sglang.test.ci.ci_register import register_cuda_ci
+from sglang.test.gpt_oss_common import BaseTestGptOss
+register_cuda_ci(est_time=500, suite="stage-b-test-small-1-gpu")
+@unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available")
```

- Reviewed files:
  - tests: `test/registered/core/test_gpt_oss_sm120.py` added +34/-0
- Risk and verification: The diff ships test coverage in `test/registered/core/test_gpt_oss_sm120.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #20755 - Use FlashInfer tinygemm for GPT-OSS MoE router on SM90+

- Link: https://github.com/sgl-project/sglang/pull/20755
- Status/date: merged / 2026-03-24
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/gpt_oss.py`; associated commits `bbe25b24126d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +65/-2, 91 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Use FlashInfer tinygemm for GPT-OSS MoE router on SM90+"; model line: GPT-OSS; category: performance/backend optimization; main diff: `python/sglang/srt/models/gpt_oss.py`; technical summary: Covers "Use FlashInfer tinygemm for GPT-OSS MoE router on SM90+"; the main implementation surface is `python/sglang/srt/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/gpt_oss.py` modified +65/-2 (67 lines); hunks: -75,10 +75,34; -97,6 +121,45 @@ def get_attention_sliding_window_size(config):; symbols: GptOssConfig, get_attention_sliding_window_size, TinyGemmLinear, __init__, touching `GptOssConfig, get_attention_sliding_window_size, TinyGemmLinear`.
- Code diff details:
  - `python/sglang/srt/models/gpt_oss.py` modified +65/-2 (67 lines); hunks: -75,10 +75,34; -97,6 +121,45 @@ def get_attention_sliding_window_size(config):; symbols: GptOssConfig, get_attention_sliding_window_size, TinyGemmLinear, __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -75,10 +75,34 @@
-from sglang.srt.utils import LazyValue, add_prefix, is_npu, make_layers
+from sglang.srt.utils import (
+    LazyValue,
+    add_prefix,
+    is_blackwell_supported,
+    is_cuda,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +65/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/gpt_oss.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21570 - [4/n] Support gpt oss 20b lora

- Link: https://github.com/sgl-project/sglang/pull/21570
- Status/date: merged / 2026-04-02
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/gpt_oss.py`, `test/registered/lora/test_lora_gpt_oss_20b_logprob_diff.py`; associated commits `566b4a4f1ccc`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +195/-24, 328 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[4/n] Support gpt oss 20b lora"; model line: GPT-OSS; category: docs/tests/CI; main diff: `python/sglang/srt/models/gpt_oss.py`, `test/registered/lora/test_lora_gpt_oss_20b_logprob_diff.py`; technical summary: Covers "[4/n] Support gpt oss 20b lora"; the main implementation surface is `python/sglang/srt/models/gpt_oss.py`, `test/registered/lora/test_lora_gpt_oss_20b_logprob_diff.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/gpt_oss.py` modified +8/-0 (8 lines); hunks: -17,6 +17,7; -651,6 +652,13 @@ def forward(; symbols: forward, GptOssForCausalLM, should_apply_lora, __init__, touching `forward, GptOssForCausalLM, should_apply_lora`; `test/registered/lora/test_lora_gpt_oss_20b_logprob_diff.py` added +151/-0 (151 lines); hunks: -0,0 +1,151; symbols: kl_v2, get_prompt_logprobs, TestLoRAGptOss20BLogprobDiff, test_lora_gpt_oss_20b_logprob_accuracy, touching `kl_v2, get_prompt_logprobs, TestLoRAGptOss20BLogprobDiff`.
- Code diff details:
  - `python/sglang/srt/models/gpt_oss.py` modified +8/-0 (8 lines); hunks: -17,6 +17,7; -651,6 +652,13 @@ def forward(; symbols: forward, GptOssForCausalLM, should_apply_lora, __init__
  - `test/registered/lora/test_lora_gpt_oss_20b_logprob_diff.py` added +151/-0 (151 lines); hunks: -0,0 +1,151; symbols: kl_v2, get_prompt_logprobs, TestLoRAGptOss20BLogprobDiff, test_lora_gpt_oss_20b_logprob_accuracy
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -17,6 +17,7 @@
+import re
@@ -651,6 +652,13 @@ def forward(
+    _lora_pattern_moe = re.compile(
+        r"^(?:model\.layers\.\d+\.(?:self_attn\.(?:qkv_proj|o_proj)|mlp\.experts)|lm_head|model\.embed_tokens)$"
+    )
+    def should_apply_lora(self, module_name: str) -> bool:
diff -- test/registered/lora/test_lora_gpt_oss_20b_logprob_diff.py
@@ -0,0 +1,151 @@
+# Copyright 2023-2025 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +8/-0
  - tests: `test/registered/lora/test_lora_gpt_oss_20b_logprob_diff.py` added +151/-0
- Risk and verification: The diff ships test coverage in `test/registered/lora/test_lora_gpt_oss_20b_logprob_diff.py`, `test/registered/lora/test_lora_qwen3_8b_logprob_diff.py`, `test/registered/lora/test_lora_qwen3_vl_30b_a3b_instruct_logprob_diff.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22237 - [CI] Relax gpt-oss 4GPU accuracy threshold from 0.60 to 0.58

- Link: https://github.com/sgl-project/sglang/pull/22237
- Status/date: merged / 2026-04-08
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-2, 18 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CI] Relax gpt-oss 4GPU accuracy threshold from 0.60 to 0.58"; model line: GPT-OSS; category: docs/tests/CI; main diff: `test/registered/4-gpu-models/test_gpt_oss_4gpu.py`; technical summary: Covers "[CI] Relax gpt-oss 4GPU accuracy threshold from 0.60 to 0.58"; the main implementation surface is `test/registered/4-gpu-models/test_gpt_oss_4gpu.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/4-gpu-models/test_gpt_oss_4gpu.py` modified +2/-2 (4 lines); hunks: -13,7 +13,7 @@ def test_bf16_120b(self):; -23,7 +23,7 @@ def test_mxfp4_120b(self):; symbols: test_bf16_120b, test_mxfp4_120b, touching `test_bf16_120b, test_mxfp4_120b`.
- Code diff details:
  - `test/registered/4-gpu-models/test_gpt_oss_4gpu.py` modified +2/-2 (4 lines); hunks: -13,7 +13,7 @@ def test_bf16_120b(self):; -23,7 +23,7 @@ def test_mxfp4_120b(self):; symbols: test_bf16_120b, test_mxfp4_120b
- Key code excerpts:

```diff
diff -- test/registered/4-gpu-models/test_gpt_oss_4gpu.py
@@ -13,7 +13,7 @@ def test_bf16_120b(self):
-                "low": 0.60,
+                "low": 0.58,
@@ -23,7 +23,7 @@ def test_mxfp4_120b(self):
-                "low": 0.60,
+                "low": 0.58,
```

- Reviewed files:
  - tests: `test/registered/4-gpu-models/test_gpt_oss_4gpu.py` modified +2/-2
- Risk and verification: The diff ships test coverage in `test/registered/4-gpu-models/test_gpt_oss_4gpu.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #25335 - [Fix] Fix gpt oss triton kernels and upgrade flashinfer back to 0.6.11.post1

- Link: https://github.com/sgl-project/sglang/pull/25335
- Status/date: merged / 2026-05-15
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 13 files, +147/-53, 404 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Fix] Fix gpt oss triton kernels and upgrade flashinfer back to 0.6.11.post1"; model line: GPT-OSS; category: bug fix; main diff: `python/sglang/srt/layers/quantization/mxfp4.py`, `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/layers/flashinfer_comm_fusion.py`; technical summary: Covers "[Fix] Fix gpt oss triton kernels and upgrade flashinfer back to 0.6.11.post1"; the main implementation surface is `python/sglang/srt/layers/quantization/mxfp4.py`, `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/layers/flashinfer_comm_fusion.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/quantization/mxfp4.py` modified +46/-3 (49 lines); hunks: -141,6 +141,7 @@ def _get_flashinfer_mxfp4_device_permute_indices(; -156,6 +157,49 @@ def _get_flashinfer_mxfp4_device_permute_indices(; symbols: _get_flashinfer_mxfp4_device_permute_indices, _patch_sm120_mxfp4_min_warps, _compute_num_warps_sm120_mxfp4, _swizzle_mxfp4, touching `_get_flashinfer_mxfp4_device_permute_indices, _patch_sm120_mxfp4_min_warps, _compute_num_warps_sm120_mxfp4`; `python/sglang/srt/layers/moe/topk.py` modified +44/-1 (45 lines); hunks: -32,7 +32,50; symbols: routing, touching `routing`; `python/sglang/srt/layers/flashinfer_comm_fusion.py` modified +11/-13 (24 lines); hunks: -383,6 +383,11 @@ def initialize(; -515,8 +520,6 @@ def ensure_workspace_initialized(; symbols: initialize, ensure_workspace_initialized, touching `initialize, ensure_workspace_initialized`; `python/sglang/srt/layers/quantization/fp4_utils.py` modified +7/-7 (14 lines); hunks: -34,13 +34,13 @@ def _flashinfer_fp4_quantize_impl(; symbols: _flashinfer_fp4_quantize_impl, touching `_flashinfer_fp4_quantize_impl`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/mxfp4.py` modified +46/-3 (49 lines); hunks: -141,6 +141,7 @@ def _get_flashinfer_mxfp4_device_permute_indices(; -156,6 +157,49 @@ def _get_flashinfer_mxfp4_device_permute_indices(; symbols: _get_flashinfer_mxfp4_device_permute_indices, _patch_sm120_mxfp4_min_warps, _compute_num_warps_sm120_mxfp4, _swizzle_mxfp4
  - `python/sglang/srt/layers/moe/topk.py` modified +44/-1 (45 lines); hunks: -32,7 +32,50; symbols: routing
  - `python/sglang/srt/layers/flashinfer_comm_fusion.py` modified +11/-13 (24 lines); hunks: -383,6 +383,11 @@ def initialize(; -515,8 +520,6 @@ def ensure_workspace_initialized(; symbols: initialize, ensure_workspace_initialized
  - `python/sglang/srt/layers/quantization/fp4_utils.py` modified +7/-7 (14 lines); hunks: -34,13 +34,13 @@ def _flashinfer_fp4_quantize_impl(; symbols: _flashinfer_fp4_quantize_impl
  - `python/sglang/srt/layers/moe/moe_runner/triton_kernels.py` modified +6/-2 (8 lines); hunks: -19,8 +19,12
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/mxfp4.py
@@ -141,6 +141,7 @@ def _get_flashinfer_mxfp4_device_permute_indices(
+_sm120_mxfp4_min_warps_patched = False
@@ -156,6 +157,49 @@ def _get_flashinfer_mxfp4_device_permute_indices(
+def _patch_sm120_mxfp4_min_warps():
+    global _sm120_mxfp4_min_warps_patched
+    if _sm120_mxfp4_min_warps_patched:
+        return
diff -- python/sglang/srt/layers/moe/topk.py
@@ -32,7 +32,50 @@
-    from triton_kernels.routing import GatherIndx, RoutingData, ScatterIndx, routing
+    from triton_kernels.matmul_ogs import GatherIndx, RoutingData, ScatterIndx
+    from triton_kernels.tensor import make_ragged_tensor_metadata
+    from triton_kernels.topk import topk as triton_kernels_topk
+    def routing(
+        logits,
diff -- python/sglang/srt/layers/flashinfer_comm_fusion.py
@@ -383,6 +383,11 @@ def initialize(
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/mxfp4.py` modified +46/-3; `python/sglang/srt/layers/moe/topk.py` modified +44/-1; `python/sglang/srt/layers/flashinfer_comm_fusion.py` modified +11/-13; `python/sglang/srt/layers/quantization/fp4_utils.py` modified +7/-7; `python/sglang/srt/layers/moe/moe_runner/triton_kernels.py` modified +6/-2; `python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py` modified +4/-3
  - tests: `test/registered/unit/layers/quantization/test_mxfp4_sm90_cutlass.py` modified +1/-6
- Risk and verification: The diff ships test coverage in `test/registered/dsv4/test_deepseek_v4_flash_fp4_h200.py`, `test/registered/moe/test_cutedsl_moe.py`, `test/registered/unit/layers/quantization/test_mxfp4_sm90_cutlass.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #25831 - [Test] Stage-a sanity kits; consolidate core/ + models_e2e/ tests

- Link: https://github.com/sgl-project/sglang/pull/25831
- Status/date: merged / 2026-05-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 36 files, +572/-639, 1504 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Test] Stage-a sanity kits; consolidate core/ + models_e2e/ tests"; model line: GPT-OSS; category: docs/tests/CI; main diff: `test/manual/models/test_nvidia_nemotron_3_nano_archived.py`, `python/sglang/test/kits/server_sanity_kit.py`, `python/sglang/test/kits/basic_scheduler_stress_kit.py`; technical summary: Covers "[Test] Stage-a sanity kits; consolidate core/ + models_e2e/ tests"; the main implementation surface is `test/manual/models/test_nvidia_nemotron_3_nano_archived.py`, `python/sglang/test/kits/server_sanity_kit.py`, `python/sglang/test/kits/basic_scheduler_stress_kit.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/manual/models/test_nvidia_nemotron_3_nano_archived.py` modified +1/-1 (2 lines); hunks: -1,4 +1,4; `python/sglang/test/kits/server_sanity_kit.py` removed +0/-228 (228 lines); hunks: -1,228 +0,0; symbols: ServerSanityMixin, _sanity_generate, test_health, test_health_generate, touching `ServerSanityMixin, _sanity_generate, test_health`; `python/sglang/test/kits/basic_scheduler_stress_kit.py` added +135/-0 (135 lines); hunks: -0,0 +1,135; symbols: BasicSchedulerStressMixin, _stress_generate, test_streaming_response, test_concurrent_requests, touching `BasicSchedulerStressMixin, _stress_generate, test_streaming_response`; `python/sglang/test/kits/basic_decode_correctness_kit.py` added +114/-0 (114 lines); hunks: -0,0 +1,114; symbols: BasicDecodeCorrectnessMixin, _decode_generate, test_capital_france, test_basic_math, touching `BasicDecodeCorrectnessMixin, _decode_generate, test_capital_france`.
- Code diff details:
  - `test/manual/models/test_nvidia_nemotron_3_nano_archived.py` modified +1/-1 (2 lines); hunks: -1,4 +1,4
  - `python/sglang/test/kits/server_sanity_kit.py` removed +0/-228 (228 lines); hunks: -1,228 +0,0; symbols: ServerSanityMixin, _sanity_generate, test_health, test_health_generate
  - `python/sglang/test/kits/basic_scheduler_stress_kit.py` added +135/-0 (135 lines); hunks: -0,0 +1,135; symbols: BasicSchedulerStressMixin, _stress_generate, test_streaming_response, test_concurrent_requests
  - `python/sglang/test/kits/basic_decode_correctness_kit.py` added +114/-0 (114 lines); hunks: -0,0 +1,114; symbols: BasicDecodeCorrectnessMixin, _decode_generate, test_capital_france, test_basic_math
  - `test/registered/language/test_srt_backend.py` removed +0/-94 (94 lines); hunks: -1,94 +0,0; symbols: TestSRTBackend, setUpClass, tearDownClass, test_few_shot_qa
- Key code excerpts:

```diff
diff -- test/manual/models/test_nvidia_nemotron_3_nano_archived.py
@@ -1,4 +1,4 @@
-"""Archived test classes split out of test/registered/models/test_nvidia_nemotron_3_nano.py.
+"""Archived test classes split out of test/registered/models_e2e/test_nvidia_nemotron_3_nano.py.
diff -- python/sglang/test/kits/server_sanity_kit.py
@@ -1,228 +0,0 @@
-"""Black-box server sanity prompts: cheap checks that catch silent
-correctness regressions (gibberish / repetition collapse / encoding),
-streaming/concurrent path bugs, and endpoint health.
-Mix into any ``CustomTestCase`` subclass that exposes ``self.base_url``
-and ``self.process``. Each test is independent and fast (≤ 5 s after
-warmup); the whole kit completes in < 1 min."""
diff -- python/sglang/test/kits/basic_scheduler_stress_kit.py
@@ -0,0 +1,135 @@
+"""Basic scheduler / cache / streaming stress sanity kit.
+Probes that catch bugs which only fire under multi-request or large-
+prompt conditions: scheduler hangs, radix prefix-cache cross-
+contamination, chunked-prefill multi-chunk kernel crashes, and SSE
```

- Reviewed files:
  - tests: `test/manual/models/test_nvidia_nemotron_3_nano_archived.py` modified +1/-1; `python/sglang/test/kits/server_sanity_kit.py` removed +0/-228; `python/sglang/test/kits/basic_scheduler_stress_kit.py` added +135/-0; `python/sglang/test/kits/basic_decode_correctness_kit.py` added +114/-0; `test/registered/language/test_srt_backend.py` removed +0/-94; `test/registered/core/test_engine_child_pids.py` modified +40/-51
- Risk and verification: The diff ships test coverage in `python/sglang/test/kits/basic_api_contract_kit.py`, `python/sglang/test/kits/basic_decode_correctness_kit.py`, `python/sglang/test/kits/basic_scheduler_stress_kit.py`, `python/sglang/test/kits/server_sanity_kit.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #26205 - Clean up server startup log noise

- Link: https://github.com/sgl-project/sglang/pull/26205
- Status/date: merged / 2026-05-24
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +157/-68, 380 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Clean up server startup log noise"; model line: GPT-OSS; category: performance/backend optimization; main diff: `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/layers/attention/flashinfer_backend.py`, `python/sglang/srt/utils/hf_transformers/tokenizer.py`; technical summary: Covers "Clean up server startup log noise"; the main implementation surface is `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/layers/attention/flashinfer_backend.py`, `python/sglang/srt/utils/hf_transformers/tokenizer.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/configs/model_config.py` modified +3/-3 (6 lines); hunks: -1450,15 +1450,15 @@ def _get_and_verify_dtype(; symbols: _get_and_verify_dtype, touching `_get_and_verify_dtype`; `python/sglang/srt/layers/attention/flashinfer_backend.py` modified +2/-4 (6 lines); hunks: -243,12 +243,10 @@ def __init__(; symbols: __init__, touching `__init__`; `python/sglang/srt/utils/hf_transformers/tokenizer.py` modified +3/-3 (6 lines); hunks: -105,7 +105,7 @@ def _load_tokenizer_by_declared_class(tokenizer_name, *args,...; -208,7 +208,7 @@ def _resolve_tokenizers_backend(tokenizer_name, *args, **com...; symbols: _load_tokenizer_by_declared_class, _resolve_tokenizers_backend, touching `_load_tokenizer_by_declared_class, _resolve_tokenizers_backend`; `python/sglang/srt/models/gpt_oss.py` modified +2/-2 (4 lines); hunks: -219,7 +219,7 @@ def __init__(; -468,7 +468,7 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `python/sglang/srt/configs/model_config.py` modified +3/-3 (6 lines); hunks: -1450,15 +1450,15 @@ def _get_and_verify_dtype(; symbols: _get_and_verify_dtype
  - `python/sglang/srt/layers/attention/flashinfer_backend.py` modified +2/-4 (6 lines); hunks: -243,12 +243,10 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/utils/hf_transformers/tokenizer.py` modified +3/-3 (6 lines); hunks: -105,7 +105,7 @@ def _load_tokenizer_by_declared_class(tokenizer_name, *args,...; -208,7 +208,7 @@ def _resolve_tokenizers_backend(tokenizer_name, *args, **com...; symbols: _load_tokenizer_by_declared_class, _resolve_tokenizers_backend
  - `python/sglang/srt/models/gpt_oss.py` modified +2/-2 (4 lines); hunks: -219,7 +219,7 @@ def __init__(; -468,7 +468,7 @@ def __init__(; symbols: __init__, forward
  - `python/sglang/srt/model_executor/model_runner.py` modified +0/-2 (2 lines); hunks: -2214,8 +2214,6 @@ def configure_kv_cache_dtype(self):; symbols: configure_kv_cache_dtype, init_cublas
- Key code excerpts:

```diff
diff -- python/sglang/srt/configs/model_config.py
@@ -1450,15 +1450,15 @@ def _get_and_verify_dtype(
-            logger.info("Upcasting %s to %s.", config_dtype, torch_dtype)
+            logger.debug("Upcasting %s to %s.", config_dtype, torch_dtype)
-            logger.info("Downcasting %s to %s.", config_dtype, torch_dtype)
+            logger.debug("Downcasting %s to %s.", config_dtype, torch_dtype)
-            logger.warning("Casting %s to %s.", config_dtype, torch_dtype)
+            logger.debug("Casting %s to %s.", config_dtype, torch_dtype)
diff -- python/sglang/srt/layers/attention/flashinfer_backend.py
@@ -243,12 +243,10 @@ def __init__(
-            # Disable CUTLASS backend when piecewise cuda graph is enabled
-            # due to TMA descriptor initialization issues on B200
-                logger.warning(
+                logger.info(
-                    "due to TMA descriptor initialization issues on B200. "
+                    "due to TMA descriptor initialization issues on SM100 GPUs. "
diff -- python/sglang/srt/utils/hf_transformers/tokenizer.py
@@ -105,7 +105,7 @@ def _load_tokenizer_by_declared_class(tokenizer_name, *args, **kwargs):
```

- Reviewed files:
  - runtime: `python/sglang/srt/configs/model_config.py` modified +3/-3; `python/sglang/srt/layers/attention/flashinfer_backend.py` modified +2/-4; `python/sglang/srt/utils/hf_transformers/tokenizer.py` modified +3/-3; `python/sglang/srt/models/gpt_oss.py` modified +2/-2; `python/sglang/srt/model_executor/model_runner.py` modified +0/-2; `python/sglang/srt/managers/template_manager.py` modified +7/-6
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/layers/attention/flashinfer_backend.py`, `python/sglang/srt/managers/template_detection.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #26610 - test/registered: cleanup pure model e2e tests (moves, splits, dedup, kit)

- Link: https://github.com/sgl-project/sglang/pull/26610
- Status/date: merged / 2026-05-28
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 26 files, +611/-816, 1566 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "test/registered: cleanup pure model e2e tests (moves, splits, dedup, kit)"; model line: GPT-OSS; category: performance/backend optimization; main diff: `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py`, `python/sglang/test/kits/unified_radix_cache_kit.py`, `test/registered/models_e2e/test_step3p5_flash_chain_mtp.py`; technical summary: Covers "test/registered: cleanup pure model e2e tests (moves, splits, dedup, kit)"; the main implementation surface is `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py`, `python/sglang/test/kits/unified_radix_cache_kit.py`, `test/registered/models_e2e/test_step3p5_flash_chain_mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py` removed +0/-212 (212 lines); hunks: -1,212 +0,0; symbols: TestDeepseekV32FP4DPSpecV2, setUpClass, tearDownClass, test_a_gsm8k, touching `TestDeepseekV32FP4DPSpecV2, setUpClass, tearDownClass`; `python/sglang/test/kits/unified_radix_cache_kit.py` renamed +1/-133 (134 lines); hunks: -1,25 +1,12; -28,18 +15,8 @@ def _random_suffixes(n, length, seed):; symbols: _random_suffixes, UnifiedRadixTreeTestMixin, test_multiturn_decode_cache_hit_branching, TestUnifiedFullRadixCache, touching `_random_suffixes, UnifiedRadixTreeTestMixin, test_multiturn_decode_cache_hit_branching`; `test/registered/models_e2e/test_step3p5_flash_chain_mtp.py` renamed +33/-78 (111 lines); hunks: -1,28 +1,20; -31,75 +23,38 @@ class TestStep3p5FlashChainMTP(CustomTestCase):; symbols: TestStep3p5FlashChainMTP, setUpClass, tearDownClass, touching `TestStep3p5FlashChainMTP, setUpClass, tearDownClass`; `test/registered/8-gpu-models/test_deepseek_v3_mtp.py` removed +0/-110 (110 lines); hunks: -1,110 +0,0; symbols: TestDeepseekV3MTP, setUpClass, tearDownClass, test_a_gsm8k, touching `TestDeepseekV3MTP, setUpClass, tearDownClass`.
- Code diff details:
  - `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py` removed +0/-212 (212 lines); hunks: -1,212 +0,0; symbols: TestDeepseekV32FP4DPSpecV2, setUpClass, tearDownClass, test_a_gsm8k
  - `python/sglang/test/kits/unified_radix_cache_kit.py` renamed +1/-133 (134 lines); hunks: -1,25 +1,12; -28,18 +15,8 @@ def _random_suffixes(n, length, seed):; symbols: _random_suffixes, UnifiedRadixTreeTestMixin, test_multiturn_decode_cache_hit_branching, TestUnifiedFullRadixCache
  - `test/registered/models_e2e/test_step3p5_flash_chain_mtp.py` renamed +33/-78 (111 lines); hunks: -1,28 +1,20; -31,75 +23,38 @@ class TestStep3p5FlashChainMTP(CustomTestCase):; symbols: TestStep3p5FlashChainMTP, setUpClass, tearDownClass
  - `test/registered/8-gpu-models/test_deepseek_v3_mtp.py` removed +0/-110 (110 lines); hunks: -1,110 +0,0; symbols: TestDeepseekV3MTP, setUpClass, tearDownClass, test_a_gsm8k
  - `test/registered/4-gpu-models/test_qwen35_models.py` removed +0/-105 (105 lines); hunks: -1,105 +0,0; symbols: TestQwen35FP4MTPV2, setUpClass, tearDownClass, test_gsm8k
- Key code excerpts:

```diff
diff -- test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py
@@ -1,212 +0,0 @@
-import unittest
-from types import SimpleNamespace
-import requests
-from sglang.srt.utils import kill_process_tree
-from sglang.test.ci.ci_register import register_cuda_ci
-from sglang.test.run_eval import run_eval
diff -- python/sglang/test/kits/unified_radix_cache_kit.py
@@ -1,25 +1,12 @@
-import unittest
-from sglang.srt.utils import kill_process_tree
-from sglang.test.ci.ci_register import register_cuda_ci
-    get_input_ids,
-    make_mamba_decode_assert,
-    make_mamba_prefill_assert,
diff -- test/registered/models_e2e/test_step3p5_flash_chain_mtp.py
@@ -1,28 +1,20 @@
```

- Reviewed files:
  - tests: `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py` removed +0/-212; `python/sglang/test/kits/unified_radix_cache_kit.py` renamed +1/-133; `test/registered/models_e2e/test_step3p5_flash_chain_mtp.py` renamed +33/-78; `test/registered/8-gpu-models/test_deepseek_v3_mtp.py` removed +0/-110; `test/registered/4-gpu-models/test_qwen35_models.py` removed +0/-105; `test/registered/quant/test_deepseek_v3_fp4_4gpu.py` removed +0/-80
- Risk and verification: The diff ships test coverage in `python/sglang/test/kits/unified_radix_cache_kit.py`, `test/manual/core/test_dsv4_hicache_swa_translation_cache.py`, `test/registered/4-gpu-models/test_qwen35_models.py`, `test/registered/8-gpu-models/test_deepseek_v3_mtp.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #16775 - [CPU] Add GPT-OSS model optimization for CPU

- Link: https://github.com/sgl-project/sglang/pull/16775
- Status/date: merged / 2026-05-29
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/gpt_oss.py`; associated commits `3ecf2c76ad1b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 35 files, +2023/-553, 3996 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CPU] Add GPT-OSS model optimization for CPU"; model line: GPT-OSS; category: performance/backend optimization; main diff: `python/sglang/srt/models/gpt_oss.py`; technical summary: Covers "[CPU] Add GPT-OSS model optimization for CPU"; the main implementation surface is `python/sglang/srt/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/gpt_oss.py` modified +20/-1 (21 lines); hunks: -81,6 +81,7; -89,6 +90,7; symbols: _load_mxfp4_experts_weights, _load_normal_weights, touching `_load_mxfp4_experts_weights, _load_normal_weights`.
- Code diff details:
  - `python/sglang/srt/models/gpt_oss.py` modified +20/-1 (21 lines); hunks: -81,6 +81,7; -89,6 +90,7; symbols: _load_mxfp4_experts_weights, _load_normal_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -81,6 +81,7 @@
+    is_cpu,
@@ -89,6 +90,7 @@
+_is_cpu = is_cpu()
@@ -881,7 +883,8 @@ def _load_mxfp4_experts_weights(self, weights):
-            weight = weight.cuda()
+            if _is_cuda:
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +20/-1
- Risk and verification: The diff ships test coverage in `test/registered/cpu/test_decode.py`, `test/registered/cpu/test_extend.py`, `test/registered/cpu/test_gemm.py`, `test/registered/cpu/test_mla.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #25813 - docs(cookbook): port popular model usage guides into cookbook pages

- Link: https://github.com/sgl-project/sglang/pull/25813
- Status/date: merged / 2026-06-02
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 47 files, +1262/-2154, 4187 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "docs(cookbook): port popular model usage guides into cookbook pages"; model line: GPT-OSS; category: docs/tests/CI; main diff: `docs_new/docs/basic_usage/deepseek_v32.mdx`, `docs_new/docs/basic_usage/deepseek_v3.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx`; technical summary: Covers "docs(cookbook): port popular model usage guides into cookbook pages"; the main implementation surface is `docs_new/docs/basic_usage/deepseek_v32.mdx`, `docs_new/docs/basic_usage/deepseek_v3.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/docs/basic_usage/deepseek_v32.mdx` removed +0/-601 (601 lines); hunks: -1,601 +0,0; `docs_new/docs/basic_usage/deepseek_v3.mdx` removed +0/-375 (375 lines); hunks: -1,375 +0,0; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx` modified +244/-3 (247 lines); hunks: -24,6 +24,27 @@ SGLang offers multiple installation methods. You can choose t...; -37,7 +58,18 @@ import { DeepSeekV32Deployment } from "/src/snippets/autoregr...; `docs_new/cookbook/autoregressive/GLM/GLM-4.6V.mdx` modified +156/-26 (182 lines); hunks: -10,7 +10,7 @@ GLM-4.6V series model includes two versions: GLM-4.6V (106B),...; -70,14 +70,56 @@ import { GLM46VDeployment } from "/src/snippets/autoregressi...; symbols: image_to_base64, touching `image_to_base64`.
- Code diff details:
  - `docs_new/docs/basic_usage/deepseek_v32.mdx` removed +0/-601 (601 lines); hunks: -1,601 +0,0
  - `docs_new/docs/basic_usage/deepseek_v3.mdx` removed +0/-375 (375 lines); hunks: -1,375 +0,0
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx` modified +244/-3 (247 lines); hunks: -24,6 +24,27 @@ SGLang offers multiple installation methods. You can choose t...; -37,7 +58,18 @@ import { DeepSeekV32Deployment } from "/src/snippets/autoregr...
  - `docs_new/cookbook/autoregressive/GLM/GLM-4.6V.mdx` modified +156/-26 (182 lines); hunks: -10,7 +10,7 @@ GLM-4.6V series model includes two versions: GLM-4.6V (106B),...; -70,14 +70,56 @@ import { GLM46VDeployment } from "/src/snippets/autoregressi...; symbols: image_to_base64
  - `docs_new/docs/basic_usage/gpt_oss.mdx` removed +0/-181 (181 lines); hunks: -1,181 +0,0
- Key code excerpts:

```diff
diff -- docs_new/docs/basic_usage/deepseek_v32.mdx
@@ -1,601 +0,0 @@
-title: "DeepSeek V3.2/GLM-5 Usage"
-metatags:
-    description: "Deploy DeepSeek V3.2/GLM-5 with SGLang: DeepSeek Sparse Attention (DSA), long-context optimization, MTP speculative decoding, function calling. Supports H200, B2
-DeepSeek-V3.2 model family equips DeepSeek-V3.1-Terminus with DeepSeek Sparse Attention (DSA) through continued training. With DSA, a fine-grained sparse attention mechanism power
-Note: This document is originally written for the usage of [DeepSeek-V3.2-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) model. The usage of [DeepSeek-V3.2](https://hu
-## Installation
diff -- docs_new/docs/basic_usage/deepseek_v3.mdx
@@ -1,375 +0,0 @@
-title: "DeepSeek V3/V3.1/R1 Usage"
-metatags:
-    description: "Deploy DeepSeek V3/R1 with SGLang: MLA optimization, FP8 quantization, multi-node TP, DP attention, MTP speculative decoding. Supports H200, B200, MI300X, A100."
-SGLang provides many optimizations specifically designed for the DeepSeek models, making it the inference engine recommended by the official [DeepSeek team](https://github.com/dee
-This document outlines current optimizations for DeepSeek.
-For an overview of the implemented features see the completed [Roadmap](https://github.com/sgl-project/sglang/issues/2591).
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx
@@ -24,6 +24,27 @@ SGLang offers multiple installation methods. You can choose the most suitable in
```

- Reviewed files:
  - docs: `docs_new/docs/basic_usage/deepseek_v32.mdx` removed +0/-601; `docs_new/docs/basic_usage/deepseek_v3.mdx` removed +0/-375; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx` modified +244/-3; `docs_new/cookbook/autoregressive/GLM/GLM-4.6V.mdx` modified +156/-26; `docs_new/docs/basic_usage/gpt_oss.mdx` removed +0/-181; `docs_new/docs/basic_usage/glmv.mdx` removed +0/-139
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR-2.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-R1.mdx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #26884 - [AMD] Fix GPT-OSS MXFP4 accuracy on ROCm AITER path

- Link: https://github.com/sgl-project/sglang/pull/26884
- Status/date: merged / 2026-06-02
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/amd/accuracy/mi35x/test_gpt_oss_eval_mi35x.py`; associated commits `4226a6f13aa6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +87/-20, 206 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Fix GPT-OSS MXFP4 accuracy on ROCm AITER path"; model line: GPT-OSS; category: bug fix; main diff: `test/registered/amd/accuracy/mi35x/test_gpt_oss_eval_mi35x.py`; technical summary: Covers "[AMD] Fix GPT-OSS MXFP4 accuracy on ROCm AITER path"; the main implementation surface is `test/registered/amd/accuracy/mi35x/test_gpt_oss_eval_mi35x.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/amd/accuracy/mi35x/test_gpt_oss_eval_mi35x.py` modified +12/-2 (14 lines); hunks: -75,7 +75,14 @@ def __post_init__(self):; -93,7 +100,10 @@ def __post_init__(self):; symbols: __post_init__, touching `__post_init__`.
- Code diff details:
  - `test/registered/amd/accuracy/mi35x/test_gpt_oss_eval_mi35x.py` modified +12/-2 (14 lines); hunks: -75,7 +75,14 @@ def __post_init__(self):; -93,7 +100,10 @@ def __post_init__(self):; symbols: __post_init__
- Key code excerpts:

```diff
diff -- test/registered/amd/accuracy/mi35x/test_gpt_oss_eval_mi35x.py
@@ -75,7 +75,14 @@ def __post_init__(self):
-        env_vars={"SGLANG_USE_AITER": "1"},
+        # AITER MXFP4 fused-MoE for gpt-oss uses the SEPARATED gate/up tile
+        # layout (matches `gptoss_fp4_tuned_fmoe.csv` flydsl entries and the
+        # Mxfp4MoEMethod weight shuffle). Other AITER MXFP4 callers default
+        # to INTERLEAVE, so opt out explicitly here.
+        env_vars={
```

- Reviewed files:
  - tests: `test/registered/amd/accuracy/mi35x/test_gpt_oss_eval_mi35x.py` modified +12/-2
- Risk and verification: The diff ships test coverage in `test/registered/amd/accuracy/mi35x/test_gpt_oss_eval_mi35x.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #27001 - [AMD] [CI] Remove hardcoded model/cache paths from MI35x nightly tests

- Link: https://github.com/sgl-project/sglang/pull/27001
- Status/date: merged / 2026-06-03
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 27 files, +11/-471, 936 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] [CI] Remove hardcoded model/cache paths from MI35x nightly tests"; model line: GPT-OSS; category: performance/backend optimization; main diff: `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_kv_fp8_perf_mi35x.py`; technical summary: Covers "[AMD] [CI] Remove hardcoded model/cache paths from MI35x nightly tests"; the main implementation surface is `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_kv_fp8_perf_mi35x.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py` modified +1/-45 (46 lines); hunks: -2,19 +2,10; -60,26 +51,9 @@ def generate_simple_markdown_report(results: List[BenchmarkRe...; symbols: generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4PerfMI35x, setUpClass, touching `generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4PerfMI35x`; `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py` modified +1/-43 (44 lines); hunks: -3,19 +3,10; -63,26 +54,9 @@ def generate_simple_markdown_report(results: List[BenchmarkRe...; symbols: generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4ArFusionPerfMI35x, setUpClass, touching `generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4ArFusionPerfMI35x`; `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_kv_fp8_perf_mi35x.py` modified +1/-43 (44 lines); hunks: -3,19 +3,10; -63,26 +54,9 @@ def generate_simple_markdown_report(results: List[BenchmarkRe...; symbols: generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4KvFp8PerfMI35x, setUpClass, touching `generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4KvFp8PerfMI35x`; `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_ar_fusion_eval_mi35x.py` modified +1/-35 (36 lines); hunks: -8,11 +8,6; -41,21 +36,6; symbols: get_model_path, ModelConfig, get_display_name, get_mxfp4_models, touching `get_model_path, ModelConfig, get_display_name`.
- Code diff details:
  - `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py` modified +1/-45 (46 lines); hunks: -2,19 +2,10; -60,26 +51,9 @@ def generate_simple_markdown_report(results: List[BenchmarkRe...; symbols: generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4PerfMI35x, setUpClass
  - `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py` modified +1/-43 (44 lines); hunks: -3,19 +3,10; -63,26 +54,9 @@ def generate_simple_markdown_report(results: List[BenchmarkRe...; symbols: generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4ArFusionPerfMI35x, setUpClass
  - `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_kv_fp8_perf_mi35x.py` modified +1/-43 (44 lines); hunks: -3,19 +3,10; -63,26 +54,9 @@ def generate_simple_markdown_report(results: List[BenchmarkRe...; symbols: generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4KvFp8PerfMI35x, setUpClass
  - `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_ar_fusion_eval_mi35x.py` modified +1/-35 (36 lines); hunks: -8,11 +8,6; -41,21 +36,6; symbols: get_model_path, ModelConfig, get_display_name, get_mxfp4_models
  - `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_eval_mi35x.py` modified +1/-35 (36 lines); hunks: -8,11 +8,6; -39,21 +34,6; symbols: get_model_path, ModelConfig, get_display_name, get_mxfp4_models
- Key code excerpts:

```diff
diff -- test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py
@@ -2,19 +2,10 @@
-The model path can be configured via DEEPSEEK_R1_MXFP4_MODEL_PATH environment variable.
-Example usage:
-    DEEPSEEK_R1_MXFP4_MODEL_PATH=/data2/models/amd-DeepSeek-R1-MXFP4-Preview python -m pytest test_deepseek_r1_mxfp4_perf_mi35x.py -v
-# Set HF cache to /data2/models/ for MI35x so HF models download there
-os.environ.setdefault("HF_HOME", "/data2/models/huggingface")
-os.environ.setdefault("HF_HUB_CACHE", "/data2/models/huggingface/hub")
diff -- test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py
@@ -3,19 +3,10 @@
-The model path can be configured via DEEPSEEK_R1_MXFP4_MODEL_PATH environment variable.
-Example usage:
-    DEEPSEEK_R1_MXFP4_MODEL_PATH=/data2/models/amd-DeepSeek-R1-MXFP4-Preview python -m pytest test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py -v
-# Set HF cache to /data2/models/ for MI35x so HF models download there
-os.environ.setdefault("HF_HOME", "/data2/models/huggingface")
-os.environ.setdefault("HF_HUB_CACHE", "/data2/models/huggingface/hub")
diff -- test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_kv_fp8_perf_mi35x.py
@@ -3,19 +3,10 @@
```

- Reviewed files:
  - tests: `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py` modified +1/-45; `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py` modified +1/-43; `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_kv_fp8_perf_mi35x.py` modified +1/-43; `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_ar_fusion_eval_mi35x.py` modified +1/-35; `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_eval_mi35x.py` modified +1/-35; `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_kv_fp8_eval_mi35x.py` modified +1/-35
- Risk and verification: The diff ships test coverage in `test/registered/amd/accuracy/mi35x/test_deepseek_r1_eval_mi35x.py`, `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_ar_fusion_eval_mi35x.py`, `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_eval_mi35x.py`, `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_kv_fp8_eval_mi35x.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #27201 - [AMD][WA] force to use gate_mode interleaved to fix tp2/tp4/tp8 acc issue

- Link: https://github.com/sgl-project/sglang/pull/27201
- Status/date: merged / 2026-06-06
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +73/-55, 233 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD][WA] force to use gate_mode interleaved to fix tp2/tp4/tp8 acc issue"; model line: GPT-OSS; category: bug fix; main diff: `python/sglang/srt/layers/quantization/mxfp4.py`, `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/server_args.py`; technical summary: Covers "[AMD][WA] force to use gate_mode interleaved to fix tp2/tp4/tp8 acc issue"; the main implementation surface is `python/sglang/srt/layers/quantization/mxfp4.py`, `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/server_args.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/quantization/mxfp4.py` modified +54/-43 (97 lines); hunks: -28,11 +28,11; -155,7 +155,9 @@ def _get_flashinfer_mxfp4_device_permute_indices(; symbols: _get_flashinfer_mxfp4_device_permute_indices, swap_every_two_rows, apply, touching `_get_flashinfer_mxfp4_device_permute_indices, swap_every_two_rows, apply`; `python/sglang/srt/layers/attention/aiter_backend.py` modified +10/-3 (13 lines); hunks: -899,10 +899,12 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; -1378,10 +1380,13 @@ def init_forward_metadata(self, forward_batch: ForwardBa...; symbols: init_forward_metadata, _apply_cuda_graph_metadata, touching `init_forward_metadata, _apply_cuda_graph_metadata`; `python/sglang/srt/server_args.py` modified +7/-7 (14 lines); hunks: -2179,13 +2179,13 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments, touching `_handle_model_specific_adjustments`; `test/registered/amd/accuracy/mi35x/test_gpt_oss_eval_mi35x.py` modified +2/-2 (4 lines); hunks: -76,7 +76,7 @@ def __post_init__(self):; -97,7 +97,7 @@ def __post_init__(self):; symbols: __post_init__, touching `__post_init__`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/mxfp4.py` modified +54/-43 (97 lines); hunks: -28,11 +28,11; -155,7 +155,9 @@ def _get_flashinfer_mxfp4_device_permute_indices(; symbols: _get_flashinfer_mxfp4_device_permute_indices, swap_every_two_rows, apply
  - `python/sglang/srt/layers/attention/aiter_backend.py` modified +10/-3 (13 lines); hunks: -899,10 +899,12 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; -1378,10 +1380,13 @@ def init_forward_metadata(self, forward_batch: ForwardBa...; symbols: init_forward_metadata, _apply_cuda_graph_metadata
  - `python/sglang/srt/server_args.py` modified +7/-7 (14 lines); hunks: -2179,13 +2179,13 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments
  - `test/registered/amd/accuracy/mi35x/test_gpt_oss_eval_mi35x.py` modified +2/-2 (4 lines); hunks: -76,7 +76,7 @@ def __post_init__(self):; -97,7 +97,7 @@ def __post_init__(self):; symbols: __post_init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/mxfp4.py
@@ -28,11 +28,11 @@
+from sglang.srt.environ import envs
@@ -155,7 +155,9 @@ def _get_flashinfer_mxfp4_device_permute_indices(
+            shuffle_scale_a16w4,
+            shuffle_weight_a16w4,
@@ -773,20 +775,13 @@ def swap_every_two_rows(x, axis=-1):
-            # Bias must be fp32 for the AITER kernels.
diff -- python/sglang/srt/layers/attention/aiter_backend.py
@@ -899,10 +899,12 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):
+                        # AITER attention kernels require int32 page indices;
+                        # full_to_swa_index_mapping is stored as int64.
-                            )
+                            ).to(torch.int32)
@@ -1378,10 +1380,13 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):
+                    # AITER attention kernels (e.g. mha_batch_prefill_func)
diff -- python/sglang/srt/server_args.py
@@ -2179,13 +2179,13 @@ def _handle_model_specific_adjustments(self):
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/mxfp4.py` modified +54/-43; `python/sglang/srt/layers/attention/aiter_backend.py` modified +10/-3; `python/sglang/srt/server_args.py` modified +7/-7
  - tests: `test/registered/amd/accuracy/mi35x/test_gpt_oss_eval_mi35x.py` modified +2/-2
- Risk and verification: The diff ships test coverage in `test/registered/amd/accuracy/mi35x/test_gpt_oss_eval_mi35x.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #27063 - [AMD] Optimize gpt-oss-120B performance

- Link: https://github.com/sgl-project/sglang/pull/27063
- Status/date: merged / 2026-06-08
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/gpt_oss.py`; associated commits `1c73ff8ad3fd`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 11 files, +1874/-54, 2160 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Optimize gpt-oss-120B performance"; model line: GPT-OSS; category: performance/backend optimization; main diff: `python/sglang/srt/models/gpt_oss.py`; technical summary: Covers "[AMD] Optimize gpt-oss-120B performance"; the main implementation surface is `python/sglang/srt/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/gpt_oss.py` modified +74/-5 (79 lines); hunks: -84,6 +84,7; -92,6 +93,7; symbols: forward, _resolve_moe_input_pad_multiple, GptOssSparseMoeBlock, __init__, touching `forward, _resolve_moe_input_pad_multiple, GptOssSparseMoeBlock`.
- Code diff details:
  - `python/sglang/srt/models/gpt_oss.py` modified +74/-5 (79 lines); hunks: -84,6 +84,7; -92,6 +93,7; symbols: forward, _resolve_moe_input_pad_multiple, GptOssSparseMoeBlock, __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -84,6 +84,7 @@
+    is_hip,
@@ -92,6 +93,7 @@
+_is_hip = is_hip()
@@ -165,6 +167,36 @@ def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]
+def _resolve_moe_input_pad_multiple(
+    quant_config: Optional[QuantizationConfig],
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +74/-5
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/environ.py`, `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/layers/attention/aiter_utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27528 - Fix GPT-OSS MXFP4 hidden size reshape on SM10X

- Link: https://github.com/sgl-project/sglang/pull/27528
- Status/date: merged / 2026-06-08
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/gpt_oss.py`; associated commits `dc24a2682190`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-1, 17 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix GPT-OSS MXFP4 hidden size reshape on SM10X"; model line: GPT-OSS; category: bug fix; main diff: `python/sglang/srt/models/gpt_oss.py`; technical summary: Covers "Fix GPT-OSS MXFP4 hidden size reshape on SM10X"; the main implementation surface is `python/sglang/srt/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/gpt_oss.py` modified +2/-1 (3 lines); hunks: -208,6 +208,7 @@ def __init__(; -291,7 +292,7 @@ def forward_normal(; symbols: __init__, forward_normal, touching `__init__, forward_normal`.
- Code diff details:
  - `python/sglang/srt/models/gpt_oss.py` modified +2/-1 (3 lines); hunks: -208,6 +208,7 @@ def __init__(; -291,7 +292,7 @@ def forward_normal(; symbols: __init__, forward_normal
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -208,6 +208,7 @@ def __init__(
+        self.hidden_size = config.hidden_size
@@ -291,7 +292,7 @@ def forward_normal(
-        hidden_dim_unpadded = self.experts.hidden_size
+        hidden_dim_unpadded = self.hidden_size
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +2/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/gpt_oss.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23906 - [Refactor] Cuda Graph Runner/Backend Refactor

- Link: https://github.com/sgl-project/sglang/pull/23906
- Status/date: merged / 2026-06-10
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 160 files, +5197/-3068, 12233 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Refactor] Cuda Graph Runner/Backend Refactor"; model line: GPT-OSS; category: performance/backend optimization; main diff: `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py`; technical summary: Covers "[Refactor] Cuda Graph Runner/Backend Refactor"; the main implementation surface is `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` removed +0/-860 (860 lines); hunks: -1,860 +0,0; symbols: freeze_gc, _to_torch, patch_model, get_global_graph_memory_pool, touching `freeze_gc, _to_torch, patch_model`; `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py` added +846/-0 (846 lines); hunks: -0,0 +1,846; symbols: PrefillCudaGraphRunner, __init__, _is_mamba_track_enabled, _cache_loc_dtype, touching `PrefillCudaGraphRunner, __init__, _is_mamba_track_enabled`; `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` renamed +294/-463 (757 lines); hunks: -1,4 +1,4; -11,33 +11,36; symbols: _make_graph_key, build_replay_fb_view, _allocate_decode_buffers, get_is_capture_mode, touching `_make_graph_key, build_replay_fb_view, _allocate_decode_buffers`; `python/sglang/srt/model_executor/breakable_cuda_graph_runner.py` removed +0/-541 (541 lines); hunks: -1,541 +0,0; symbols: BreakableCudaGraphRunner, __init__, _has_inactive_dp_rank, _init_buffers, touching `BreakableCudaGraphRunner, __init__, _has_inactive_dp_rank`.
- Code diff details:
  - `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` removed +0/-860 (860 lines); hunks: -1,860 +0,0; symbols: freeze_gc, _to_torch, patch_model, get_global_graph_memory_pool
  - `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py` added +846/-0 (846 lines); hunks: -0,0 +1,846; symbols: PrefillCudaGraphRunner, __init__, _is_mamba_track_enabled, _cache_loc_dtype
  - `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` renamed +294/-463 (757 lines); hunks: -1,4 +1,4; -11,33 +11,36; symbols: _make_graph_key, build_replay_fb_view, _allocate_decode_buffers, get_is_capture_mode
  - `python/sglang/srt/model_executor/breakable_cuda_graph_runner.py` removed +0/-541 (541 lines); hunks: -1,541 +0,0; symbols: BreakableCudaGraphRunner, __init__, _has_inactive_dp_rank, _init_buffers
  - `python/sglang/srt/model_executor/runner_utils/buffers.py` added +442/-0 (442 lines); hunks: -0,0 +1,442; symbols: _grouped_foreach_copy_, foreach_copy, DecodeInputBuffers, create
- Key code excerpts:

```diff
diff -- python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py
@@ -1,860 +0,0 @@
-# Copyright 2023-2024 SGLang Team
-# Licensed under the Apache License, Version 2.0 (the "License");
-# you may not use this file except in compliance with the License.
-# You may obtain a copy of the License at
-#
-#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py
@@ -0,0 +1,846 @@
+# Copyright 2023-2026 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py
@@ -1,4 +1,4 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` removed +0/-860; `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py` added +846/-0; `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` renamed +294/-463; `python/sglang/srt/model_executor/breakable_cuda_graph_runner.py` removed +0/-541; `python/sglang/srt/model_executor/runner_utils/buffers.py` added +442/-0; `python/sglang/srt/model_executor/runner_backend/tc_piecewise_cuda_graph_backend.py` added +225/-0
- Risk and verification: The diff ships test coverage in `python/sglang/test/doc_patch.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dense_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dsa_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dsv4_attention.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #27964 - [Spec] Retire Spec V1

- Link: https://github.com/sgl-project/sglang/pull/27964
- Status/date: merged / 2026-06-11
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 46 files, +111/-252, 1422 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Spec] Retire Spec V1"; model line: GPT-OSS; category: performance/backend optimization; main diff: `test/registered/ep/test_deepep_large.py`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx`, `python/sglang/srt/arg_groups/speculative_hook.py`; technical summary: Covers "[Spec] Retire Spec V1"; the main implementation surface is `test/registered/ep/test_deepep_large.py`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx`, `python/sglang/srt/arg_groups/speculative_hook.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/ep/test_deepep_large.py` modified +43/-44 (87 lines); hunks: -3,7 +3,6; -87,49 +86,49 @@ class TestDeepseekMTP(CustomTestCase):; symbols: TestDeepseekMTP, setUpClass, tearDownClass, touching `TestDeepseekMTP, setUpClass, tearDownClass`; `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx` modified +0/-64 (64 lines); hunks: -1108,7 +1108,6 @@ do; -1227,7 +1226,6 @@ do; `python/sglang/srt/arg_groups/speculative_hook.py` modified +10/-26 (36 lines); hunks: -1,9 +1,8; -63,6 +62,15 @@ def handle_speculative_decoding(server_args: "ServerArgs") ->...; symbols: handle_speculative_decoding, _handle_dflash, _handle_frozen_kv_mtp, _handle_eagle_family, touching `handle_speculative_decoding, _handle_dflash, _handle_frozen_kv_mtp`; `docs_new/docs/advanced_features/speculative_decoding.mdx` modified +4/-21 (25 lines); hunks: -33,7 +33,6 @@ SGLang provides several speculative decoding options, includin...; -101,13 +100,6 @@ SGLang provides several speculative decoding options, inclu....
- Code diff details:
  - `test/registered/ep/test_deepep_large.py` modified +43/-44 (87 lines); hunks: -3,7 +3,6; -87,49 +86,49 @@ class TestDeepseekMTP(CustomTestCase):; symbols: TestDeepseekMTP, setUpClass, tearDownClass
  - `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx` modified +0/-64 (64 lines); hunks: -1108,7 +1108,6 @@ do; -1227,7 +1226,6 @@ do
  - `python/sglang/srt/arg_groups/speculative_hook.py` modified +10/-26 (36 lines); hunks: -1,9 +1,8; -63,6 +62,15 @@ def handle_speculative_decoding(server_args: "ServerArgs") ->...; symbols: handle_speculative_decoding, _handle_dflash, _handle_frozen_kv_mtp, _handle_eagle_family
  - `docs_new/docs/advanced_features/speculative_decoding.mdx` modified +4/-21 (25 lines); hunks: -33,7 +33,6 @@ SGLang provides several speculative decoding options, includin...; -101,13 +100,6 @@ SGLang provides several speculative decoding options, inclu...
  - `test/registered/spec/eagle/test_eagle_constrained_decoding.py` modified +11/-10 (21 lines); hunks: -1,6 +1,5; -31,7 +30,8 @@ class TestEagleConstrainedDecoding(; symbols: TestEagleConstrainedDecoding, setUpClass, tearDownClass, TestEagleConstrainedDecodingV2
- Key code excerpts:

```diff
diff -- test/registered/ep/test_deepep_large.py
@@ -3,7 +3,6 @@
-from sglang.srt.environ import envs
@@ -87,49 +86,49 @@ class TestDeepseekMTP(CustomTestCase):
-        with envs.SGLANG_ENABLE_SPEC_V2.override(False):
-            cls.process = popen_launch_server(
-                cls.model,
-                cls.base_url,
diff -- docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx
@@ -1108,7 +1108,6 @@ do
-        export SGLANG_ENABLE_SPEC_V2=1
@@ -1227,7 +1226,6 @@ do
-        export SGLANG_ENABLE_SPEC_V2=1
@@ -1351,7 +1349,6 @@ do
-        export SGLANG_ENABLE_SPEC_V2=1
@@ -1476,7 +1473,6 @@ do
diff -- python/sglang/srt/arg_groups/speculative_hook.py
@@ -1,9 +1,8 @@
```

- Reviewed files:
  - tests: `test/registered/ep/test_deepep_large.py` modified +43/-44; `test/registered/spec/eagle/test_eagle_constrained_decoding.py` modified +11/-10; `python/sglang/test/server_fixtures/standalone_fixture.py` modified +7/-8; `python/sglang/test/server_fixtures/spec_eagle_fixture.py` modified +6/-6
  - docs: `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx` modified +0/-64; `docs_new/docs/advanced_features/speculative_decoding.mdx` modified +4/-21; `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_optimization.mdx` modified +3/-8
  - runtime: `python/sglang/srt/arg_groups/speculative_hook.py` modified +10/-26
- Risk and verification: The diff ships test coverage in `python/sglang/test/server_fixtures/spec_eagle_fixture.py`, `python/sglang/test/server_fixtures/standalone_fixture.py`, `test/manual/dsv4/test_dsv4_flash_mtp_tp8.py`, `test/manual/dsv4/test_dsv4_pro_mtp.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #27941 - Enable PDL for GPT-OSS tinygemm router

- Link: https://github.com/sgl-project/sglang/pull/27941
- Status/date: merged / 2026-06-12
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/gpt_oss.py`; associated commits `cb9140ee6108`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-2, 25 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Enable PDL for GPT-OSS tinygemm router"; model line: GPT-OSS; category: model support/runtime entry; main diff: `python/sglang/srt/models/gpt_oss.py`; technical summary: Covers "Enable PDL for GPT-OSS tinygemm router"; the main implementation surface is `python/sglang/srt/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/gpt_oss.py` modified +2/-2 (4 lines); hunks: -26,6 +26,7; -161,7 +162,7 @@ def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Op...; symbols: forward, _load_normal_weights, touching `forward, _load_normal_weights`.
- Code diff details:
  - `python/sglang/srt/models/gpt_oss.py` modified +2/-2 (4 lines); hunks: -26,6 +26,7; -161,7 +162,7 @@ def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Op...; symbols: forward, _load_normal_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -26,6 +26,7 @@
+from sglang.jit_kernel.utils import is_arch_support_pdl
@@ -161,7 +162,7 @@ def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]
-            tinygemm_bf16(x, self.weight, out, self.bias)
+            tinygemm_bf16(x, self.weight, out, self.bias, use_pdl=is_arch_support_pdl())
@@ -1094,7 +1095,6 @@ def _load_normal_weights(
-        tp_rank = get_tensor_model_parallel_rank()
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/gpt_oss.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #28567 - Add get_parallel(): a structured accessor for parallel-topology state

- Link: https://github.com/sgl-project/sglang/pull/28567
- Status/date: merged / 2026-06-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 184 files, +1865/-1727, 8932 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add get_parallel(): a structured accessor for parallel-topology state"; model line: GPT-OSS; category: model support/runtime entry; main diff: `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py`; technical summary: Covers "Add get_parallel(): a structured accessor for parallel-topology state"; the main implementation surface is `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/apertus.py` modified +686/-687 (1373 lines); hunks: -1,687 +1,686; symbols: ApertusMLP, __init__, forward, ApertusAttention, touching `ApertusMLP, __init__, forward`; `python/sglang/srt/models/solar.py` modified +28/-27 (55 lines); hunks: -1,37 +1,14; -54,6 +31,30; symbols: __init__, forward, load_kv_cache_scales, touching `__init__, forward, load_kv_cache_scales`; `python/sglang/srt/models/gpt_oss.py` modified +17/-24 (41 lines); hunks: -28,21 +28,13; -76,6 +68,7; symbols: _resolve_moe_input_pad_multiple, __init__, touching `_resolve_moe_input_pad_multiple, __init__`; `python/sglang/srt/models/deepseek_v2.py` modified +14/-23 (37 lines); hunks: -47,9 +47,7; -72,12 +70,6; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/apertus.py` modified +686/-687 (1373 lines); hunks: -1,687 +1,686; symbols: ApertusMLP, __init__, forward, ApertusAttention
  - `python/sglang/srt/models/solar.py` modified +28/-27 (55 lines); hunks: -1,37 +1,14; -54,6 +31,30; symbols: __init__, forward, load_kv_cache_scales
  - `python/sglang/srt/models/gpt_oss.py` modified +17/-24 (41 lines); hunks: -28,21 +28,13; -76,6 +68,7; symbols: _resolve_moe_input_pad_multiple, __init__
  - `python/sglang/srt/models/deepseek_v2.py` modified +14/-23 (37 lines); hunks: -47,9 +47,7; -72,12 +70,6; symbols: __init__
  - `python/sglang/srt/layers/communicator.py` modified +13/-19 (32 lines); hunks: -23,8 +23,6; -44,12 +42,7; symbols: apply_aiter_all_reduce_fusion, init_context, should_fuse_mlp_allreduce_with_next_layer, is_same_group_size
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/apertus.py
@@ -1,687 +1,686 @@
-# SPDX-License-Identifier: Apache-2.0
-# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-# Copyright 2025 The SwissAI Initiative
-# Copyright 2023-2024 SGLang Team
-# Licensed under the Apache License, Version 2.0 (the "License");
-# you may not use this file except in compliance with the License.
diff -- python/sglang/srt/models/solar.py
@@ -1,37 +1,14 @@
-# Adapted from
-# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
-# Copyright 2023 The vLLM team.
-# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
-#
-# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
diff -- python/sglang/srt/models/gpt_oss.py
@@ -28,21 +28,13 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/apertus.py` modified +686/-687; `python/sglang/srt/models/solar.py` modified +28/-27; `python/sglang/srt/models/gpt_oss.py` modified +17/-24; `python/sglang/srt/models/deepseek_v2.py` modified +14/-23; `python/sglang/srt/layers/communicator.py` modified +13/-19; `python/sglang/srt/models/qwen3_moe.py` modified +12/-18
- Risk and verification: The diff ships test coverage in `python/sglang/test/kits/attention_unittest/attention_methods/dense_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dsa_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dsv4_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dual_chunk_attention.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #28697 - [docs] Add B300 cookbook deployment options

- Link: https://github.com/sgl-project/sglang/pull/28697
- Status/date: merged / 2026-06-19
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 27 files, +503/-69, 1291 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[docs] Add B300 cookbook deployment options"; model line: GPT-OSS; category: performance/backend optimization; main diff: `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx`, `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx`; technical summary: Covers "[docs] Add B300 cookbook deployment options"; the main implementation surface is `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx`, `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx` added +167/-0 (167 lines); hunks: -0,0 +1,167; `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx` modified +68/-2 (70 lines); hunks: -9,6 +9,11 @@ const lookupData = {; -182,6 +187,66 @@ const lookupData = {; `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx` modified +40/-16 (56 lines); hunks: -4,6 +4,7 @@ export const GLM5Deployment = () => {; -13,6 +14,7 @@ export const GLM5Deployment = () => {; `docs_new/src/snippets/autoregressive/deepseek-v32-deployment.jsx` modified +29/-10 (39 lines); hunks: -3,7 +3,7 @@ export const DeepSeekV32Deployment = () => {; -12,6 +12,7 @@ export const DeepSeekV32Deployment = () => {.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx` added +167/-0 (167 lines); hunks: -0,0 +1,167
  - `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx` modified +68/-2 (70 lines); hunks: -9,6 +9,11 @@ const lookupData = {; -182,6 +187,66 @@ const lookupData = {
  - `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx` modified +40/-16 (56 lines); hunks: -4,6 +4,7 @@ export const GLM5Deployment = () => {; -13,6 +14,7 @@ export const GLM5Deployment = () => {
  - `docs_new/src/snippets/autoregressive/deepseek-v32-deployment.jsx` modified +29/-10 (39 lines); hunks: -3,7 +3,7 @@ export const DeepSeekV32Deployment = () => {; -12,6 +12,7 @@ export const DeepSeekV32Deployment = () => {
  - `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx` modified +23/-15 (38 lines); hunks: -8,19 +8,19 @@ export const Qwen35Deployment = () => {; -149,7 +149,7 @@ export const Qwen35Deployment = () => {
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx
@@ -0,0 +1,167 @@
+export const InternS1Deployment = () => {
+  const options = {
+    hardware: {
+      name: 'hardware',
+      title: 'Hardware Platform',
+      items: [
diff -- docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx
@@ -9,6 +9,11 @@ const lookupData = {
+      {
+        "id": "b300",
+        "label": "B300",
+        "default": false
+      },
@@ -182,6 +187,66 @@ const lookupData = {
diff -- docs_new/src/snippets/autoregressive/glm-5-deployment.jsx
@@ -4,6 +4,7 @@ export const GLM5Deployment = () => {
```

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx` added +167/-0; `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx` modified +68/-2; `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx` modified +40/-16; `docs_new/src/snippets/autoregressive/deepseek-v32-deployment.jsx` modified +29/-10; `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx` modified +23/-15; `docs_new/cookbook/autoregressive/InternLM/Intern-S1.mdx` modified +16/-13
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/InternLM/Intern-S1.mdx`, `docs_new/src/snippets/autoregressive/deepseek-math-v2-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
