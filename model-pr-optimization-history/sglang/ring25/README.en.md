# sglang Ring 2.5 Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `docs_new/cookbook/autoregressive/InclusionAI/Ring-2.5-1T.mdx` | no direct PR-number commit |
| `docs_new/src/snippets/autoregressive/ring-25-1t-deployment.jsx` | no direct PR-number commit |
| `test/registered/8-gpu-models/test_ring_2_5_1t.py` | no direct PR-number commit |

## PR Coverage Summary

- Git-traced PRs: 0
- Extra PRs preserved from existing docs: 48
- Total PRs in this document: 47
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2025-08-06 | [#8680](https://github.com/sgl-project/sglang/pull/8680) | merged | Support bailing moe | `python/sglang/srt/models/bailing_moe.py`, `test/srt/models/test_generation_models.py`, `docs/supported_models/generative_models.md` |
| 2025-09-12 | [#10359](https://github.com/sgl-project/sglang/pull/10359) | merged | Support LingV2 model | `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/bailing_moe_nextn.py`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=256,N=512,device_name=NVIDIA_H20.json` |
| 2025-09-12 | [#10362](https://github.com/sgl-project/sglang/pull/10362) | merged | Fix Bailing MoE model bugs | `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/server_args.py` |
| 2025-09-15 | [#9338](https://github.com/sgl-project/sglang/pull/9338) | merged | Refactor TopK to ensure readability and extensibility | `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` |
| 2025-09-24 | [#10860](https://github.com/sgl-project/sglang/pull/10860) | merged | fix bailing_moe with enable_dp_attention | `python/sglang/srt/models/bailing_moe.py` |
| 2025-09-26 | [#10749](https://github.com/sgl-project/sglang/pull/10749) | merged | Fuse write kv buffer into rope for qwen3 moe & bailing moe | `python/sglang/srt/models/utils.py`, `python/sglang/srt/models/gpt_oss.py`, `python/sglang/srt/models/bailing_moe.py` |
| 2025-10-12 | [#11465](https://github.com/sgl-project/sglang/pull/11465) | merged | bailingMoE: Fix Key error of deepep_mode | `python/sglang/srt/models/bailing_moe.py` |
| 2025-10-12 | [#11331](https://github.com/sgl-project/sglang/pull/11331) | merged | Deprecate `global_server_args_dict` | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/models/glm4_moe.py` |
| 2025-10-13 | [#11520](https://github.com/sgl-project/sglang/pull/11520) | merged | Revert "Deprecate `global_server_args_dict`" | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/models/glm4_moe.py` |
| 2025-10-13 | [#11528](https://github.com/sgl-project/sglang/pull/11528) | merged | Depreate `global_server_args_dict` | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/models/glm4_moe.py` |
| 2025-10-17 | [#11685](https://github.com/sgl-project/sglang/pull/11685) | merged | [Lint] Add `python/sglang` to ruff F401 checks and remove unused imports in files | `python/sglang/srt/layers/quantization/w8a8_int8.py`, `python/sglang/srt/models/qwen2_audio.py`, `python/sglang/srt/models/longcat_flash.py` |
| 2025-10-20 | [#11847](https://github.com/sgl-project/sglang/pull/11847) | merged | [9/N] MoE Refactor: cleanup dispatcher interfaces | `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` |
| 2025-10-31 | [#12369](https://github.com/sgl-project/sglang/pull/12369) | merged | Enable bailing_moe to support TP=16 | `python/sglang/srt/models/bailing_moe.py` |
| 2025-12-07 | [#14337](https://github.com/sgl-project/sglang/pull/14337) | merged | remove unecessary dual stream token threshold from the rest of models (qwen moe, kimi linear, etc.) | `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/kimi_linear.py`, `python/sglang/srt/models/llada2.py` |
| 2025-12-12 | [#13730](https://github.com/sgl-project/sglang/pull/13730) | merged | [bugfix] fix TBO crashes when attn_tp_size > 1 | `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/model_executor/forward_batch_info.py`, `python/sglang/srt/models/bailing_moe.py` |
| 2025-12-21 | [#15526](https://github.com/sgl-project/sglang/pull/15526) | merged | Optimize Bailing-MoE with FlashInfer Fused All-Reduce | `python/sglang/srt/models/bailing_moe.py` |
| 2025-12-28 | [#15835](https://github.com/sgl-project/sglang/pull/15835) | merged | [Feature] JIT Fused QK norm + qk norm clean up | `python/sglang/srt/models/utils.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen3.py` |
| 2026-01-10 | [#13715](https://github.com/sgl-project/sglang/pull/13715) | merged | Fix EPLB + FP4 Quantization Compatibility Issue | `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/qwen2_moe.py` |
| 2026-01-24 | [#17570](https://github.com/sgl-project/sglang/pull/17570) | merged | Use attn tp group in embedding for more models | `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/bailing_moe_nextn.py`, `python/sglang/srt/models/falcon_h1.py` |
| 2026-01-30 | [#17808](https://github.com/sgl-project/sglang/pull/17808) | merged | Fix the scenario where eh_proj is quantized in the bailing moe nextn weights | `python/sglang/srt/models/bailing_moe_nextn.py` |
| 2026-02-01 | [#15119](https://github.com/sgl-project/sglang/pull/15119) | merged | feat: Add Ling Flash v2.0 support for Eagle3 | `python/sglang/srt/models/bailing_moe.py` |
| 2026-02-13 | [#18598](https://github.com/sgl-project/sglang/pull/18598) | merged | Support LingV2_5 model | `python/sglang/srt/models/bailing_moe_linear.py`, `python/sglang/srt/layers/attention/linear/seg_la.py`, `python/sglang/srt/layers/attention/linear/lightning_attn.py` |
| 2026-02-13 | [#18793](https://github.com/sgl-project/sglang/pull/18793) | merged | Cleanup debug log for Ring model | `python/sglang/srt/models/bailing_moe_linear.py`, `python/sglang/srt/configs/mamba_utils.py` |
| 2026-02-14 | [#18829](https://github.com/sgl-project/sglang/pull/18829) | merged | Add ci test for ring model | `test/registered/8-gpu-models/test_ring_2_5_1t.py`, `python/sglang/test/accuracy_test_runner.py` |
| 2026-02-15 | [#18860](https://github.com/sgl-project/sglang/pull/18860) | merged | update pre-commit config | `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/models/pixtral.py`, `python/sglang/srt/layers/quantization/modelslim/modelslim_moe.py` |
| 2026-02-25 | [#19201](https://github.com/sgl-project/sglang/pull/19201) | merged | Add server CUDA graph warmup CI step for cold H200 nodes | `scripts/ci/cuda/warmup_deep_gemm.py`, `scripts/ci/cuda/warmup_server.py`, `.github/workflows/pr-test.yml` |
| 2026-03-18 | [#17784](https://github.com/sgl-project/sglang/pull/17784) | merged | Upgrade transformers==5.3.0 | `python/sglang/srt/models/gemma3_causal.py`, `python/sglang/srt/layers/rotary_embedding/factory.py`, `python/sglang/srt/configs/model_config.py` |
| 2026-03-19 | [#9744](https://github.com/sgl-project/sglang/pull/9744) | merged | [CPU] Add FP8 Bmm support | `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/models/longcat_flash.py`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` |
| 2026-03-23 | [#20316](https://github.com/sgl-project/sglang/pull/20316) | merged | fix fused_set_kv_buffer for rope with Ling-v2 | `python/sglang/srt/models/bailing_moe.py` |
| 2026-03-31 | [#21751](https://github.com/sgl-project/sglang/pull/21751) | merged | [CI] Fix ring test timeout | `test/registered/8-gpu-models/test_ring_2_5_1t.py` |
| 2026-04-03 | [#22045](https://github.com/sgl-project/sglang/pull/22045) | merged | [CI] Adjust CI server launch timeout | `python/sglang/test/accuracy_test_runner.py`, `python/sglang/test/test_utils.py`, `test/registered/8-gpu-models/test_ring_2_5_1t.py` |
| 2026-04-07 | [#22267](https://github.com/sgl-project/sglang/pull/22267) | merged | Move ring test to nightly | `test/registered/8-gpu-models/test_ring_2_5_1t.py` |
| 2026-04-10 | [#22305](https://github.com/sgl-project/sglang/pull/22305) | merged | [CI] Update est_time for 64 tests based on actual elapsed times | `test/registered/models/test_nvidia_nemotron_3_nano.py`, `test/registered/models/test_transformers_backend_eval.py`, `test/registered/models/test_transformers_models.py` |
| 2026-04-20 | [#23001](https://github.com/sgl-project/sglang/pull/23001) | merged | Add new Mintlify documentation site (docs_new/) | `docs_new/docs/advanced_features/tool_parser.mdx`, `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx`, `docs_new/docs/advanced_features/separate_reasoning.mdx` |
| 2026-04-21 | [#23337](https://github.com/sgl-project/sglang/pull/23337) | merged | [Docs] Sync docs_new with legacy docs and update migration redirects | `docs_new/docs/supported-models/multimodal_language_models.mdx`, `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx` |
| 2026-04-26 | [#23732](https://github.com/sgl-project/sglang/pull/23732) | merged | Apply should_use_dp_reduce_scatterv guard to remaining MoE models (follow-up to #23731) | `python/sglang/srt/models/llada2.py`, `python/sglang/srt/models/hunyuan_v3.py`, `python/sglang/srt/models/bailing_moe_linear.py` |
| 2026-04-27 | [#23748](https://github.com/sgl-project/sglang/pull/23748) | merged | refactor(moe): centralize post-experts all-reduce skip predicate | `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/models/sarvam_moe.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2026-04-30 | [#21126](https://github.com/sgl-project/sglang/pull/21126) | merged | [4/N] Quantization Refactor: AWQ schemes and Kernel call and weight init split | `python/sglang/srt/layers/quantization/awq.py`, `python/sglang/srt/layers/quantization/awq/awq.py`, `python/sglang/srt/layers/quantization/awq/schemes/awq_moe.py` |
| 2026-05-04 | [#24333](https://github.com/sgl-project/sglang/pull/24333) | merged | nextn subclass owns post_load_weights is_nextn | `python/sglang/srt/models/bailing_moe_nextn.py`, `python/sglang/srt/models/deepseek_nextn.py`, `python/sglang/srt/model_loader/loader.py` |
| 2026-05-27 | [#23837](https://github.com/sgl-project/sglang/pull/23837) | merged | Add Ling_2_6 | `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20-3e_down.json`, `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20_down.json`, `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20-3e.json` |
| 2026-05-29 | [#26474](https://github.com/sgl-project/sglang/pull/26474) | merged | [HotFix][Ling 2.6] Fix HybridLinearAttn dispatcher for Ling-2.6 | `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/models/bailing_moe_linear.py`, `test/registered/8-gpu-models/test_ling_2_6_flash.py` |
| 2026-06-02 | [#26623](https://github.com/sgl-project/sglang/pull/26623) | merged | Fix hybrid linear attention misrouting plain-RadixAttention linear layers to the full backend (Ring-2.5-1T) | `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/models/bailing_moe_linear.py`, `test/registered/8-gpu-models/test_ling_2_6_flash.py` |
| 2026-06-03 | [#27116](https://github.com/sgl-project/sglang/pull/27116) | merged | Revert "Fix hybrid linear attention misrouting plain-RadixAttention linear layers to the full backend (Ring-2.5-1T)" | `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/models/bailing_moe_linear.py` |
| 2026-06-03 | [#27120](https://github.com/sgl-project/sglang/pull/27120) | merged | Fix hybrid linear attention dispatch by layer id with draft-worker awareness | `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/layers/attention/attention_registry.py`, `python/sglang/srt/models/bailing_moe_linear.py` |
| 2026-06-10 | [#23906](https://github.com/sgl-project/sglang/pull/23906) | merged | [Refactor] Cuda Graph Runner/Backend Refactor | `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` |
| 2026-06-18 | [#28567](https://github.com/sgl-project/sglang/pull/28567) | merged | Add get_parallel(): a structured accessor for parallel-topology state | `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py` |
| 2026-06-19 | [#28697](https://github.com/sgl-project/sglang/pull/28697) | merged | [docs] Add B300 cookbook deployment options | `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx`, `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx` |

## Per-PR Diff Audit Cards

### PR #8680 - Support bailing moe

- Link: https://github.com/sgl-project/sglang/pull/8680
- Status/date: merged / 2025-08-06
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +427/-0, 441 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support bailing moe"; model line: Ring 2.5; category: docs/tests/CI; main diff: `python/sglang/srt/models/bailing_moe.py`, `test/srt/models/test_generation_models.py`, `docs/supported_models/generative_models.md`; technical summary: Covers "Support bailing moe"; the main implementation surface is `python/sglang/srt/models/bailing_moe.py`, `test/srt/models/test_generation_models.py`, `docs/supported_models/generative_models.md`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/bailing_moe.py` added +425/-0 (425 lines); hunks: -0,0 +1,425; symbols: BailingAttention, __init__, forward, BailingMLP, touching `BailingAttention, __init__, forward`; `test/srt/models/test_generation_models.py` modified +1/-0 (1 lines); hunks: -67,6 +67,7 @@ class ModelCase:; symbols: ModelCase, touching `ModelCase`; `docs/supported_models/generative_models.md` modified +1/-0 (1 lines); hunks: -47,5 +47,6 @@ in the GitHub search bar..
- Code diff details:
  - `python/sglang/srt/models/bailing_moe.py` added +425/-0 (425 lines); hunks: -0,0 +1,425; symbols: BailingAttention, __init__, forward, BailingMLP
  - `test/srt/models/test_generation_models.py` modified +1/-0 (1 lines); hunks: -67,6 +67,7 @@ class ModelCase:; symbols: ModelCase
  - `docs/supported_models/generative_models.md` modified +1/-0 (1 lines); hunks: -47,5 +47,6 @@ in the GitHub search bar.
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/bailing_moe.py
@@ -0,0 +1,425 @@
+# Copyright 2023-2024 SGLang Team
+# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/bailing_moe.py
+from collections.abc import Iterable
+from typing import Optional, Tuple
+import torch
+import torch.nn.functional as F
diff -- test/srt/models/test_generation_models.py
@@ -67,6 +67,7 @@ class ModelCase:
+    ModelCase("inclusionAI/Ling-lite", trust_remote_code=True),
diff -- docs/supported_models/generative_models.md
@@ -47,5 +47,6 @@ in the GitHub search bar.
+| **Ling** (16.8B–290B) | `inclusionAI/Ling-lite`, `inclusionAI/Ling-plus` | InclusionAI’s open MoE models. Ling-Lite has 16.8B total / 2.75B active parameters, and Ling-Plus has
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/bailing_moe.py` added +425/-0
  - tests: `test/srt/models/test_generation_models.py` modified +1/-0
  - docs: `docs/supported_models/generative_models.md` modified +1/-0
- Risk and verification: The diff ships test coverage in `test/srt/models/test_generation_models.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #10359 - Support LingV2 model

- Link: https://github.com/sgl-project/sglang/pull/10359
- Status/date: merged / 2025-09-12
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +1165/-221, 1642 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support LingV2 model"; model line: Ring 2.5; category: performance/backend optimization; main diff: `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/bailing_moe_nextn.py`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=256,N=512,device_name=NVIDIA_H20.json`; technical summary: Covers "Support LingV2 model"; the main implementation surface is `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/bailing_moe_nextn.py`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=256,N=512,device_name=NVIDIA_H20.json`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/bailing_moe.py` modified +795/-218 (1013 lines); hunks: -1,19 +1,51; -22,356 +54,828; symbols: BailingAttention, BailingMoEMLP, __init__, forward, touching `BailingAttention, BailingMoEMLP, __init__`; `python/sglang/srt/models/bailing_moe_nextn.py` added +168/-0 (168 lines); hunks: -0,0 +1,168; symbols: BailingMoEModelNextN, __init__, forward, BailingMoeForCausalLMNextN, touching `BailingMoEModelNextN, __init__, forward`; `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=256,N=512,device_name=NVIDIA_H20.json` added +146/-0 (146 lines); hunks: -0,0 +1,146; `python/sglang/srt/layers/linear.py` modified +32/-0 (32 lines); hunks: -893,6 +893,35 @@ def _load_fused_module_from_checkpoint(; -906,6 +935,9 @@ def weight_loader_v2(; symbols: _load_fused_module_from_checkpoint, _load_qkv_block_scale, weight_loader_v2, touching `_load_fused_module_from_checkpoint, _load_qkv_block_scale, weight_loader_v2`.
- Code diff details:
  - `python/sglang/srt/models/bailing_moe.py` modified +795/-218 (1013 lines); hunks: -1,19 +1,51; -22,356 +54,828; symbols: BailingAttention, BailingMoEMLP, __init__, forward
  - `python/sglang/srt/models/bailing_moe_nextn.py` added +168/-0 (168 lines); hunks: -0,0 +1,168; symbols: BailingMoEModelNextN, __init__, forward, BailingMoeForCausalLMNextN
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=256,N=512,device_name=NVIDIA_H20.json` added +146/-0 (146 lines); hunks: -0,0 +1,146
  - `python/sglang/srt/layers/linear.py` modified +32/-0 (32 lines); hunks: -893,6 +893,35 @@ def _load_fused_module_from_checkpoint(; -906,6 +935,9 @@ def weight_loader_v2(; symbols: _load_fused_module_from_checkpoint, _load_qkv_block_scale, weight_loader_v2
  - `python/sglang/srt/configs/model_config.py` modified +5/-0 (5 lines); hunks: -141,6 +141,11 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/bailing_moe.py
@@ -1,19 +1,51 @@
-# Copyright 2023-2024 SGLang Team
-# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/bailing_moe.py
-from collections.abc import Iterable
-from typing import Optional, Tuple
+# coding=utf-8
+# Copyright 2023 Antgroup and The HuggingFace Inc. team. All rights reserved.
diff -- python/sglang/srt/models/bailing_moe_nextn.py
@@ -0,0 +1,168 @@
+# coding=utf-8
+# Copyright 2023 Antgroup and The HuggingFace Inc. team. All rights reserved.
+#
+# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
+# and OPT implementations in this library. It has been modified from its
+# original forms to accommodate minor architectural differences compared
diff -- python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=256,N=512,device_name=NVIDIA_H20.json
@@ -0,0 +1,146 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/bailing_moe.py` modified +795/-218; `python/sglang/srt/models/bailing_moe_nextn.py` added +168/-0; `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=256,N=512,device_name=NVIDIA_H20.json` added +146/-0; `python/sglang/srt/layers/linear.py` modified +32/-0; `python/sglang/srt/configs/model_config.py` modified +5/-0; `python/sglang/srt/server_args.py` modified +8/-1
  - other: `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` modified +11/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/layers/linear.py`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=256,N=512,device_name=NVIDIA_H20.json`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #10362 - Fix Bailing MoE model bugs

- Link: https://github.com/sgl-project/sglang/pull/10362
- Status/date: merged / 2025-09-12
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +8/-5, 41 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix Bailing MoE model bugs"; model line: Ring 2.5; category: bug fix; main diff: `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/server_args.py`; technical summary: Covers "Fix Bailing MoE model bugs"; the main implementation surface is `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/server_args.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/bailing_moe.py` modified +7/-4 (11 lines); hunks: -128,7 +128,9 @@ def forward(; -328,7 +330,7 @@ def forward_normal_dual_stream(; symbols: forward, forward_normal_dual_stream, forward_normal, touching `forward, forward_normal_dual_stream, forward_normal`; `python/sglang/srt/server_args.py` modified +1/-1 (2 lines); hunks: -757,7 +757,7 @@ def __post_init__(self):; symbols: __post_init__, touching `__post_init__`.
- Code diff details:
  - `python/sglang/srt/models/bailing_moe.py` modified +7/-4 (11 lines); hunks: -128,7 +128,9 @@ def forward(; -328,7 +330,7 @@ def forward_normal_dual_stream(; symbols: forward, forward_normal_dual_stream, forward_normal
  - `python/sglang/srt/server_args.py` modified +1/-1 (2 lines); hunks: -757,7 +757,7 @@ def __post_init__(self):; symbols: __post_init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/bailing_moe.py
@@ -128,7 +128,9 @@ def forward(
-        hidden_states, _ = self.down_proj(hidden_states)
+        hidden_states, _ = self.down_proj(
+            hidden_states, skip_all_reduce=use_reduce_scatter
+        )
@@ -328,7 +330,7 @@ def forward_normal_dual_stream(
-        shared_output = self._forward_shared_experts(hidden_states)
diff -- python/sglang/srt/server_args.py
@@ -757,7 +757,7 @@ def __post_init__(self):
-                "BailingMoeV2ForCausalLM",
+                "BailingMoeForCausalLM",
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/bailing_moe.py` modified +7/-4; `python/sglang/srt/server_args.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/server_args.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9338 - Refactor TopK to ensure readability and extensibility

- Link: https://github.com/sgl-project/sglang/pull/9338
- Status/date: merged / 2025-09-15
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 14 files, +52/-47, 296 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Refactor TopK to ensure readability and extensibility"; model line: Ring 2.5; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`; technical summary: Covers "Refactor TopK to ensure readability and extensibility"; the main implementation surface is `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/moe/topk.py` modified +30/-9 (39 lines); hunks: -19,6 +19,7; -51,6 +52,9; symbols: TopKConfig, __init__, forward_native, touching `TopKConfig, __init__, forward_native`; `python/sglang/srt/models/deepseek_v2.py` modified +7/-12 (19 lines); hunks: -65,14 +65,10; -375,21 +371,20 @@ def __init__(; symbols: __init__, touching `__init__`; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +0/-10 (10 lines); hunks: -74,16 +74,6; symbols: _is_fp4_quantization_enabled, selection, _get_tile_tokens_dim, touching `_is_fp4_quantization_enabled, selection, _get_tile_tokens_dim`; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +4/-4 (8 lines); hunks: -888,7 +888,7 @@ def _forward_ll(dispatch_output: DeepEPLLOutput):; -901,8 +901,7 @@ def get_moe_impl_class(quant_config: Optional[QuantizationCo...; symbols: _forward_ll, get_moe_impl_class, touching `_forward_ll, get_moe_impl_class`.
- Code diff details:
  - `python/sglang/srt/layers/moe/topk.py` modified +30/-9 (39 lines); hunks: -19,6 +19,7; -51,6 +52,9; symbols: TopKConfig, __init__, forward_native
  - `python/sglang/srt/models/deepseek_v2.py` modified +7/-12 (19 lines); hunks: -65,14 +65,10; -375,21 +371,20 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +0/-10 (10 lines); hunks: -74,16 +74,6; symbols: _is_fp4_quantization_enabled, selection, _get_tile_tokens_dim
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +4/-4 (8 lines); hunks: -888,7 +888,7 @@ def _forward_ll(dispatch_output: DeepEPLLOutput):; -901,8 +901,7 @@ def get_moe_impl_class(quant_config: Optional[QuantizationCo...; symbols: _forward_ll, get_moe_impl_class
  - `python/sglang/srt/models/longcat_flash.py` modified +2/-2 (4 lines); hunks: -260,7 +260,7 @@ def __init__(; -853,7 +853,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: __init__, load_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/topk.py
@@ -19,6 +19,7 @@
+    TYPE_CHECKING,
@@ -51,6 +52,9 @@
+if TYPE_CHECKING:
+    from sglang.srt.layers.quantization import QuantizationConfig
@@ -94,6 +98,7 @@ class TopKConfig:
+    output_format: Optional[TopKOutputFormat] = None
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -65,14 +65,10 @@
-    should_use_flashinfer_trtllm_moe,
-from sglang.srt.layers.moe.fused_moe_triton.layer import (
-    FusedMoE,
-    _is_fp4_quantization_enabled,
-)
-from sglang.srt.layers.moe.topk import TopK
diff -- python/sglang/srt/layers/moe/fused_moe_triton/layer.py
@@ -74,16 +74,6 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/topk.py` modified +30/-9; `python/sglang/srt/models/deepseek_v2.py` modified +7/-12; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +0/-10; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +4/-4; `python/sglang/srt/models/longcat_flash.py` modified +2/-2; `python/sglang/srt/models/qwen3_next.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/layers/moe/topk.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #10860 - fix bailing_moe with enable_dp_attention

- Link: https://github.com/sgl-project/sglang/pull/10860
- Status/date: merged / 2025-09-24
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-2, 23 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix bailing_moe with enable_dp_attention"; model line: Ring 2.5; category: bug fix; main diff: `python/sglang/srt/models/bailing_moe.py`; technical summary: Covers "fix bailing_moe with enable_dp_attention"; the main implementation surface is `python/sglang/srt/models/bailing_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/bailing_moe.py` modified +2/-2 (4 lines); hunks: -45,12 +45,12; -702,7 +702,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/bailing_moe.py` modified +2/-2 (4 lines); hunks: -45,12 +45,12; -702,7 +702,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/bailing_moe.py
@@ -45,12 +45,12 @@
+    is_dp_attention_enabled,
-    ReplicatedLinear,
@@ -702,7 +702,7 @@ def __init__(
-                use_attn_tp_group=global_server_args_dict["enable_dp_lm_head"],
+                enable_tp=not is_dp_attention_enabled(),
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/bailing_moe.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/bailing_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #10749 - Fuse write kv buffer into rope for qwen3 moe & bailing moe

- Link: https://github.com/sgl-project/sglang/pull/10749
- Status/date: merged / 2025-09-26
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +105/-34, 207 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fuse write kv buffer into rope for qwen3 moe & bailing moe"; model line: Ring 2.5; category: performance/backend optimization; main diff: `python/sglang/srt/models/utils.py`, `python/sglang/srt/models/gpt_oss.py`, `python/sglang/srt/models/bailing_moe.py`; technical summary: Covers "Fuse write kv buffer into rope for qwen3 moe & bailing moe"; the main implementation surface is `python/sglang/srt/models/utils.py`, `python/sglang/srt/models/gpt_oss.py`, `python/sglang/srt/models/bailing_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/utils.py` added +51/-0 (51 lines); hunks: -0,0 +1,51; symbols: enable_fused_set_kv_buffer, create_fused_set_kv_buffer_arg, touching `enable_fused_set_kv_buffer, create_fused_set_kv_buffer_arg`; `python/sglang/srt/models/gpt_oss.py` modified +7/-30 (37 lines); hunks: -66,6 +66,10; -193,33 +197,6 @@ def forward_normal(; symbols: forward_normal, _enable_fused_set_kv_buffer, _create_fused_set_kv_buffer_arg, GptOssAttention, touching `forward_normal, _enable_fused_set_kv_buffer, _create_fused_set_kv_buffer_arg`; `python/sglang/srt/models/bailing_moe.py` modified +25/-2 (27 lines); hunks: -72,6 +72,10; -555,8 +559,27 @@ def forward(; symbols: forward, touching `forward`; `python/sglang/srt/models/qwen3_moe.py` modified +22/-2 (24 lines); hunks: -60,6 +60,10; -412,15 +416,31 @@ def forward_prepare(; symbols: forward_prepare, forward_core, touching `forward_prepare, forward_core`.
- Code diff details:
  - `python/sglang/srt/models/utils.py` added +51/-0 (51 lines); hunks: -0,0 +1,51; symbols: enable_fused_set_kv_buffer, create_fused_set_kv_buffer_arg
  - `python/sglang/srt/models/gpt_oss.py` modified +7/-30 (37 lines); hunks: -66,6 +66,10; -193,33 +197,6 @@ def forward_normal(; symbols: forward_normal, _enable_fused_set_kv_buffer, _create_fused_set_kv_buffer_arg, GptOssAttention
  - `python/sglang/srt/models/bailing_moe.py` modified +25/-2 (27 lines); hunks: -72,6 +72,10; -555,8 +559,27 @@ def forward(; symbols: forward
  - `python/sglang/srt/models/qwen3_moe.py` modified +22/-2 (24 lines); hunks: -60,6 +60,10; -412,15 +416,31 @@ def forward_prepare(; symbols: forward_prepare, forward_core
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/utils.py
@@ -0,0 +1,51 @@
+# Copyright 2023-2025 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/models/gpt_oss.py
@@ -66,6 +66,10 @@
+from sglang.srt.models.utils import (
+    create_fused_set_kv_buffer_arg,
+    enable_fused_set_kv_buffer,
+)
@@ -193,33 +197,6 @@ def forward_normal(
-def _enable_fused_set_kv_buffer(forward_batch: ForwardBatch):
diff -- python/sglang/srt/models/bailing_moe.py
@@ -72,6 +72,10 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/utils.py` added +51/-0; `python/sglang/srt/models/gpt_oss.py` modified +7/-30; `python/sglang/srt/models/bailing_moe.py` modified +25/-2; `python/sglang/srt/models/qwen3_moe.py` modified +22/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/gpt_oss.py`, `python/sglang/srt/models/qwen3_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #11465 - bailingMoE: Fix Key error of deepep_mode

- Link: https://github.com/sgl-project/sglang/pull/11465
- Status/date: merged / 2025-10-12
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-2, 18 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "bailingMoE: Fix Key error of deepep_mode"; model line: Ring 2.5; category: bug fix; main diff: `python/sglang/srt/models/bailing_moe.py`; technical summary: Covers "bailingMoE: Fix Key error of deepep_mode"; the main implementation surface is `python/sglang/srt/models/bailing_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/bailing_moe.py` modified +2/-2 (4 lines); hunks: -54,7 +54,7; -293,7 +293,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/bailing_moe.py` modified +2/-2 (4 lines); hunks: -54,7 +54,7; -293,7 +293,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/bailing_moe.py
@@ -54,7 +54,7 @@
-from sglang.srt.layers.moe import get_moe_a2a_backend
+from sglang.srt.layers.moe import get_deepep_mode, get_moe_a2a_backend
@@ -293,7 +293,7 @@ def __init__(
-                deepep_mode=DeepEPMode[global_server_args_dict["deepep_mode"]],
+                deepep_mode=get_deepep_mode(),
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/bailing_moe.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/bailing_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #11331 - Deprecate `global_server_args_dict`

- Link: https://github.com/sgl-project/sglang/pull/11331
- Status/date: merged / 2025-10-12
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 54 files, +240/-321, 1946 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Deprecate `global_server_args_dict`"; model line: Ring 2.5; category: model implementation change; main diff: `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/models/glm4_moe.py`; technical summary: Covers "Deprecate `global_server_args_dict`"; the main implementation surface is `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/models/glm4_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +23/-21 (44 lines); hunks: -35,7 +35,6; -108,10 +107,11; symbols: __init__, touching `__init__`; `python/sglang/srt/model_executor/model_runner.py` modified +16/-21 (37 lines); hunks: -83,10 +83,6; -125,7 +121,11; symbols: __init__, initialize, _get_attention_backend, _get_attention_backend_from_str, touching `__init__, initialize, _get_attention_backend`; `python/sglang/srt/models/glm4_moe.py` modified +8/-12 (20 lines); hunks: -56,18 +56,13; -77,6 +72,7; symbols: __init__, determine_num_fused_shared_experts, touching `__init__, determine_num_fused_shared_experts`; `python/sglang/srt/layers/logits_processor.py` modified +6/-10 (16 lines); hunks: -38,17 +38,15; -230,8 +228,8 @@ def __init__(; symbols: __init__, compute_logprobs_for_multi_item_scoring, forward, touching `__init__, compute_logprobs_for_multi_item_scoring, forward`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +23/-21 (44 lines); hunks: -35,7 +35,6; -108,10 +107,11; symbols: __init__
  - `python/sglang/srt/model_executor/model_runner.py` modified +16/-21 (37 lines); hunks: -83,10 +83,6; -125,7 +121,11; symbols: __init__, initialize, _get_attention_backend, _get_attention_backend_from_str
  - `python/sglang/srt/models/glm4_moe.py` modified +8/-12 (20 lines); hunks: -56,18 +56,13; -77,6 +72,7; symbols: __init__, determine_num_fused_shared_experts
  - `python/sglang/srt/layers/logits_processor.py` modified +6/-10 (16 lines); hunks: -38,17 +38,15; -230,8 +228,8 @@ def __init__(; symbols: __init__, compute_logprobs_for_multi_item_scoring, forward
  - `python/sglang/srt/models/qwen3_vl_moe.py` modified +3/-11 (14 lines); hunks: -38,20 +38,12
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -35,7 +35,6 @@
-from sglang.srt.debug_utils.dumper import dumper
@@ -108,10 +107,11 @@
-from sglang.srt.managers.schedule_batch import global_server_args_dict
+from sglang.srt.server_args import get_global_server_args
+from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
@@ -520,7 +520,7 @@ def __init__(
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -83,10 +83,6 @@
-from sglang.srt.managers.schedule_batch import (
-    GLOBAL_SERVER_ARGS_KEYS,
-    global_server_args_dict,
-)
@@ -125,7 +121,11 @@
-from sglang.srt.server_args import ServerArgs
diff -- python/sglang/srt/models/glm4_moe.py
@@ -56,18 +56,13 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +23/-21; `python/sglang/srt/model_executor/model_runner.py` modified +16/-21; `python/sglang/srt/models/glm4_moe.py` modified +8/-12; `python/sglang/srt/layers/logits_processor.py` modified +6/-10; `python/sglang/srt/models/qwen3_vl_moe.py` modified +3/-11; `python/sglang/srt/layers/communicator.py` modified +8/-5
- Risk and verification: The diff ships test coverage in `test/srt/rl/test_fp32_lm_head.py`, `test/srt/test_gptqmodel_dynamic.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #11520 - Revert "Deprecate `global_server_args_dict`"

- Link: https://github.com/sgl-project/sglang/pull/11520
- Status/date: merged / 2025-10-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 54 files, +321/-240, 1946 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Revert "Deprecate `global_server_args_dict`""; model line: Ring 2.5; category: model implementation change; main diff: `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/models/glm4_moe.py`; technical summary: Covers "Revert "Deprecate `global_server_args_dict`""; the main implementation surface is `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/models/glm4_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +21/-23 (44 lines); hunks: -35,6 +35,7; -107,11 +108,10; symbols: __init__, touching `__init__`; `python/sglang/srt/model_executor/model_runner.py` modified +21/-16 (37 lines); hunks: -83,6 +83,10; -121,11 +125,7; symbols: __init__, initialize, _get_attention_backend, _get_attention_backend_from_str, touching `__init__, initialize, _get_attention_backend`; `python/sglang/srt/models/glm4_moe.py` modified +12/-8 (20 lines); hunks: -56,13 +56,18; -72,7 +77,6; symbols: __init__, determine_num_fused_shared_experts, touching `__init__, determine_num_fused_shared_experts`; `python/sglang/srt/layers/logits_processor.py` modified +10/-6 (16 lines); hunks: -38,15 +38,17; -228,8 +230,8 @@ def __init__(; symbols: __init__, compute_logprobs_for_multi_item_scoring, forward, touching `__init__, compute_logprobs_for_multi_item_scoring, forward`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +21/-23 (44 lines); hunks: -35,6 +35,7; -107,11 +108,10; symbols: __init__
  - `python/sglang/srt/model_executor/model_runner.py` modified +21/-16 (37 lines); hunks: -83,6 +83,10; -121,11 +125,7; symbols: __init__, initialize, _get_attention_backend, _get_attention_backend_from_str
  - `python/sglang/srt/models/glm4_moe.py` modified +12/-8 (20 lines); hunks: -56,13 +56,18; -72,7 +77,6; symbols: __init__, determine_num_fused_shared_experts
  - `python/sglang/srt/layers/logits_processor.py` modified +10/-6 (16 lines); hunks: -38,15 +38,17; -228,8 +230,8 @@ def __init__(; symbols: __init__, compute_logprobs_for_multi_item_scoring, forward
  - `python/sglang/srt/models/qwen3_vl_moe.py` modified +11/-3 (14 lines); hunks: -38,12 +38,20
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -35,6 +35,7 @@
+from sglang.srt.debug_utils.dumper import dumper
@@ -107,11 +108,10 @@
+from sglang.srt.managers.schedule_batch import global_server_args_dict
-from sglang.srt.server_args import get_global_server_args
-from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
@@ -520,7 +520,7 @@ def __init__(
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -83,6 +83,10 @@
+from sglang.srt.managers.schedule_batch import (
+    GLOBAL_SERVER_ARGS_KEYS,
+    global_server_args_dict,
+)
@@ -121,11 +125,7 @@
-from sglang.srt.server_args import (
diff -- python/sglang/srt/models/glm4_moe.py
@@ -56,13 +56,18 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +21/-23; `python/sglang/srt/model_executor/model_runner.py` modified +21/-16; `python/sglang/srt/models/glm4_moe.py` modified +12/-8; `python/sglang/srt/layers/logits_processor.py` modified +10/-6; `python/sglang/srt/models/qwen3_vl_moe.py` modified +11/-3; `python/sglang/srt/layers/communicator.py` modified +5/-8
- Risk and verification: The diff ships test coverage in `test/srt/rl/test_fp32_lm_head.py`, `test/srt/test_gptqmodel_dynamic.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #11528 - Depreate `global_server_args_dict`

- Link: https://github.com/sgl-project/sglang/pull/11528
- Status/date: merged / 2025-10-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 54 files, +240/-321, 1946 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Depreate `global_server_args_dict`"; model line: Ring 2.5; category: model implementation change; main diff: `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/models/glm4_moe.py`; technical summary: Covers "Depreate `global_server_args_dict`"; the main implementation surface is `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/models/glm4_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +23/-21 (44 lines); hunks: -35,7 +35,6; -108,10 +107,11; symbols: __init__, touching `__init__`; `python/sglang/srt/model_executor/model_runner.py` modified +16/-21 (37 lines); hunks: -83,10 +83,6; -125,7 +121,11; symbols: __init__, initialize, _get_attention_backend, _get_attention_backend_from_str, touching `__init__, initialize, _get_attention_backend`; `python/sglang/srt/models/glm4_moe.py` modified +8/-12 (20 lines); hunks: -56,18 +56,13; -77,6 +72,7; symbols: __init__, determine_num_fused_shared_experts, touching `__init__, determine_num_fused_shared_experts`; `python/sglang/srt/layers/logits_processor.py` modified +6/-10 (16 lines); hunks: -38,17 +38,15; -230,8 +228,8 @@ def __init__(; symbols: __init__, compute_logprobs_for_multi_item_scoring, forward, touching `__init__, compute_logprobs_for_multi_item_scoring, forward`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +23/-21 (44 lines); hunks: -35,7 +35,6; -108,10 +107,11; symbols: __init__
  - `python/sglang/srt/model_executor/model_runner.py` modified +16/-21 (37 lines); hunks: -83,10 +83,6; -125,7 +121,11; symbols: __init__, initialize, _get_attention_backend, _get_attention_backend_from_str
  - `python/sglang/srt/models/glm4_moe.py` modified +8/-12 (20 lines); hunks: -56,18 +56,13; -77,6 +72,7; symbols: __init__, determine_num_fused_shared_experts
  - `python/sglang/srt/layers/logits_processor.py` modified +6/-10 (16 lines); hunks: -38,17 +38,15; -230,8 +228,8 @@ def __init__(; symbols: __init__, compute_logprobs_for_multi_item_scoring, forward
  - `python/sglang/srt/models/qwen3_vl_moe.py` modified +3/-11 (14 lines); hunks: -38,20 +38,12
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -35,7 +35,6 @@
-from sglang.srt.debug_utils.dumper import dumper
@@ -108,10 +107,11 @@
-from sglang.srt.managers.schedule_batch import global_server_args_dict
+from sglang.srt.server_args import get_global_server_args
+from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
@@ -520,7 +520,7 @@ def __init__(
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -83,10 +83,6 @@
-from sglang.srt.managers.schedule_batch import (
-    GLOBAL_SERVER_ARGS_KEYS,
-    global_server_args_dict,
-)
@@ -125,7 +121,11 @@
-from sglang.srt.server_args import ServerArgs
diff -- python/sglang/srt/models/glm4_moe.py
@@ -56,18 +56,13 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +23/-21; `python/sglang/srt/model_executor/model_runner.py` modified +16/-21; `python/sglang/srt/models/glm4_moe.py` modified +8/-12; `python/sglang/srt/layers/logits_processor.py` modified +6/-10; `python/sglang/srt/models/qwen3_vl_moe.py` modified +3/-11; `python/sglang/srt/layers/communicator.py` modified +8/-5
- Risk and verification: The diff ships test coverage in `test/srt/rl/test_fp32_lm_head.py`, `test/srt/test_gptqmodel_dynamic.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #11685 - [Lint] Add `python/sglang` to ruff F401 checks and remove unused imports in files

- Link: https://github.com/sgl-project/sglang/pull/11685
- Status/date: merged / 2025-10-17
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 151 files, +124/-406, 1915 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Lint] Add `python/sglang` to ruff F401 checks and remove unused imports in files"; model line: Ring 2.5; category: performance/backend optimization; main diff: `python/sglang/srt/layers/quantization/w8a8_int8.py`, `python/sglang/srt/models/qwen2_audio.py`, `python/sglang/srt/models/longcat_flash.py`; technical summary: Covers "[Lint] Add `python/sglang` to ruff F401 checks and remove unused imports in files"; the main implementation surface is `python/sglang/srt/layers/quantization/w8a8_int8.py`, `python/sglang/srt/models/qwen2_audio.py`, `python/sglang/srt/models/longcat_flash.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/quantization/w8a8_int8.py` modified +2/-18 (20 lines); hunks: -1,28 +1,12; `python/sglang/srt/models/qwen2_audio.py` modified +2/-15 (17 lines); hunks: -23,30 +23,18; -60,7 +48,6; `python/sglang/srt/models/longcat_flash.py` modified +1/-14 (15 lines); hunks: -44,9 +44,7; -87,20 +85,15; `python/sglang/srt/models/longcat_flash_nextn.py` modified +2/-13 (15 lines); hunks: -32,14 +32,10; -75,7 +71,6.
- Code diff details:
  - `python/sglang/srt/layers/quantization/w8a8_int8.py` modified +2/-18 (20 lines); hunks: -1,28 +1,12
  - `python/sglang/srt/models/qwen2_audio.py` modified +2/-15 (17 lines); hunks: -23,30 +23,18; -60,7 +48,6
  - `python/sglang/srt/models/longcat_flash.py` modified +1/-14 (15 lines); hunks: -44,9 +44,7; -87,20 +85,15
  - `python/sglang/srt/models/longcat_flash_nextn.py` modified +2/-13 (15 lines); hunks: -32,14 +32,10; -75,7 +71,6
  - `python/sglang/srt/models/mimo.py` modified +2/-13 (15 lines); hunks: -1,28 +1,17
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/w8a8_int8.py
@@ -1,28 +1,12 @@
-import importlib
-import sys
-from typing import (
-    TYPE_CHECKING,
-    Any,
-    Callable,
diff -- python/sglang/srt/models/qwen2_audio.py
@@ -23,30 +23,18 @@
-import math
-from functools import lru_cache, partial
-from typing import Any, Iterable, List, Optional, Tuple, Type, TypedDict
+from typing import Any, Iterable, List, Optional, Tuple
-import torch.nn.functional as F
-from einops import rearrange
diff -- python/sglang/srt/models/longcat_flash.py
@@ -44,9 +44,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/w8a8_int8.py` modified +2/-18; `python/sglang/srt/models/qwen2_audio.py` modified +2/-15; `python/sglang/srt/models/longcat_flash.py` modified +1/-14; `python/sglang/srt/models/longcat_flash_nextn.py` modified +2/-13; `python/sglang/srt/models/mimo.py` modified +2/-13; `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +3/-10
- Risk and verification: The diff ships test coverage in `python/sglang/test/attention/test_flashattn_mla_backend.py`, `python/sglang/test/attention/test_prefix_chunk_info.py`, `python/sglang/test/few_shot_gsm8k_engine.py`, `python/sglang/test/simple_eval_gpqa.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #11847 - [9/N] MoE Refactor: cleanup dispatcher interfaces

- Link: https://github.com/sgl-project/sglang/pull/11847
- Status/date: merged / 2025-10-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 24 files, +394/-428, 1948 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[9/N] MoE Refactor: cleanup dispatcher interfaces"; model line: Ring 2.5; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`; technical summary: Covers "[9/N] MoE Refactor: cleanup dispatcher interfaces"; the main implementation surface is `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +86/-91 (177 lines); hunks: -7,6 +7,7; -15,6 +16,7; symbols: DeepEPNormalOutput, format, DeepEPLLOutput, __init__, touching `DeepEPNormalOutput, format, DeepEPLLOutput`; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +69/-99 (168 lines); hunks: -20,18 +20,14; -109,23 +105,6 @@ def __init__(; symbols: __init__, forward, dispatch, touching `__init__, forward, dispatch`; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +44/-35 (79 lines); hunks: -11,14 +11,19; -32,6 +37,7; symbols: _get_tile_tokens_dim, create_moe_dispatcher, FusedMoeWeightScaleSupported, __init__, touching `_get_tile_tokens_dim, create_moe_dispatcher, FusedMoeWeightScaleSupported`; `python/sglang/srt/layers/moe/token_dispatcher/mooncake.py` modified +37/-39 (76 lines); hunks: -5,13 +5,15; -27,16 +29,15; symbols: MooncakeDispatchOutput, __init__, dispatch_a, dispatch_b, touching `MooncakeDispatchOutput, __init__, dispatch_a`.
- Code diff details:
  - `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +86/-91 (177 lines); hunks: -7,6 +7,7; -15,6 +16,7; symbols: DeepEPNormalOutput, format, DeepEPLLOutput, __init__
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +69/-99 (168 lines); hunks: -20,18 +20,14; -109,23 +105,6 @@ def __init__(; symbols: __init__, forward, dispatch
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +44/-35 (79 lines); hunks: -11,14 +11,19; -32,6 +37,7; symbols: _get_tile_tokens_dim, create_moe_dispatcher, FusedMoeWeightScaleSupported, __init__
  - `python/sglang/srt/layers/moe/token_dispatcher/mooncake.py` modified +37/-39 (76 lines); hunks: -5,13 +5,15; -27,16 +29,15; symbols: MooncakeDispatchOutput, __init__, dispatch_a, dispatch_b
  - `python/sglang/srt/models/deepseek_v2.py` modified +14/-46 (60 lines); hunks: -74,7 +74,6; -113,10 +112,7; symbols: __init__, forward_deepep, _forward_shared_experts_and_put_results, op_select_experts
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/token_dispatcher/deepep.py
@@ -7,6 +7,7 @@
+from sglang.srt.layers.dp_attention import get_is_extend_in_batch
@@ -15,6 +16,7 @@
+from sglang.srt.layers.moe.topk import TopKOutput
@@ -51,8 +53,6 @@
-from sglang.srt.model_executor.forward_batch_info import ForwardBatch
@@ -61,9 +61,9 @@
diff -- python/sglang/srt/layers/moe/ep_moe/layer.py
@@ -20,18 +20,14 @@
+from sglang.srt.layers.moe.topk import TopKOutput
-from sglang.srt.layers.quantization.modelopt_quant import (
-    CUTEDSL_MOE_NVFP4_DISPATCH,
-    ModelOptNvFp4FusedMoEMethod,
-)
-from sglang.srt.model_executor.forward_batch_info import ForwardBatch
diff -- python/sglang/srt/layers/moe/fused_moe_triton/layer.py
@@ -11,14 +11,19 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +86/-91; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +69/-99; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +44/-35; `python/sglang/srt/layers/moe/token_dispatcher/mooncake.py` modified +37/-39; `python/sglang/srt/models/deepseek_v2.py` modified +14/-46; `python/sglang/srt/layers/moe/token_dispatcher/standard.py` modified +46/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/dp_attention.py`, `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #12369 - Enable bailing_moe to support TP=16

- Link: https://github.com/sgl-project/sglang/pull/12369
- Status/date: merged / 2025-10-31
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +9/-2, 24 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Enable bailing_moe to support TP=16"; model line: Ring 2.5; category: model support/runtime entry; main diff: `python/sglang/srt/models/bailing_moe.py`; technical summary: Covers "Enable bailing_moe to support TP=16"; the main implementation surface is `python/sglang/srt/models/bailing_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/bailing_moe.py` modified +9/-2 (11 lines); hunks: -420,14 +420,21 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/bailing_moe.py` modified +9/-2 (11 lines); hunks: -420,14 +420,21 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/bailing_moe.py
@@ -420,14 +420,21 @@ def __init__(
-        assert self.total_kv_heads % attn_tp_size == 0
+        if self.total_kv_heads >= attn_tp_size:
+            # Number of KV heads is greater than TP size, so we partition
+            # the KV heads across multiple tensor parallel GPUs.
+            assert self.total_kv_heads % attn_tp_size == 0
+        else:
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/bailing_moe.py` modified +9/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/bailing_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #14337 - remove unecessary dual stream token threshold from the rest of models (qwen moe, kimi linear, etc.)

- Link: https://github.com/sgl-project/sglang/pull/14337
- Status/date: merged / 2025-12-07
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +0/-8, 50 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "remove unecessary dual stream token threshold from the rest of models (qwen moe, kimi linear, etc.)"; model line: Ring 2.5; category: model implementation change; main diff: `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/kimi_linear.py`, `python/sglang/srt/models/llada2.py`; technical summary: Covers "remove unecessary dual stream token threshold from the rest of models (qwen moe, kimi linear, etc.)"; the main implementation surface is `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/kimi_linear.py`, `python/sglang/srt/models/llada2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/bailing_moe.py` modified +0/-2 (2 lines); hunks: -349,11 +349,9 @@ def forward_normal(; symbols: forward_normal, touching `forward_normal`; `python/sglang/srt/models/kimi_linear.py` modified +0/-2 (2 lines); hunks: -125,13 +125,11 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Te...; symbols: forward, touching `forward`; `python/sglang/srt/models/llada2.py` modified +0/-2 (2 lines); hunks: -349,11 +349,9 @@ def forward_normal(; symbols: forward_normal, touching `forward_normal`; `python/sglang/srt/models/qwen2_moe.py` modified +0/-2 (2 lines); hunks: -275,11 +275,9 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/bailing_moe.py` modified +0/-2 (2 lines); hunks: -349,11 +349,9 @@ def forward_normal(; symbols: forward_normal
  - `python/sglang/srt/models/kimi_linear.py` modified +0/-2 (2 lines); hunks: -125,13 +125,11 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Te...; symbols: forward
  - `python/sglang/srt/models/llada2.py` modified +0/-2 (2 lines); hunks: -349,11 +349,9 @@ def forward_normal(; symbols: forward_normal
  - `python/sglang/srt/models/qwen2_moe.py` modified +0/-2 (2 lines); hunks: -275,11 +275,9 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/bailing_moe.py
@@ -349,11 +349,9 @@ def forward_normal(
-        DUAL_STREAM_TOKEN_THRESHOLD = 1024
-            and hidden_states.shape[0] <= DUAL_STREAM_TOKEN_THRESHOLD
diff -- python/sglang/srt/models/kimi_linear.py
@@ -125,13 +125,11 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
-        DUAL_STREAM_TOKEN_THRESHOLD = 1024
-            and hidden_states.shape[0] <= DUAL_STREAM_TOKEN_THRESHOLD
diff -- python/sglang/srt/models/llada2.py
@@ -349,11 +349,9 @@ def forward_normal(
-        DUAL_STREAM_TOKEN_THRESHOLD = 1024
-            and hidden_states.shape[0] <= DUAL_STREAM_TOKEN_THRESHOLD
diff -- python/sglang/srt/models/qwen2_moe.py
@@ -275,11 +275,9 @@ def forward(
-        DUAL_STREAM_TOKEN_THRESHOLD = 1024
-            and hidden_states.shape[0] <= DUAL_STREAM_TOKEN_THRESHOLD
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/bailing_moe.py` modified +0/-2; `python/sglang/srt/models/kimi_linear.py` modified +0/-2; `python/sglang/srt/models/llada2.py` modified +0/-2; `python/sglang/srt/models/qwen2_moe.py` modified +0/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/kimi_linear.py`, `python/sglang/srt/models/llada2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #13730 - [bugfix] fix TBO crashes when attn_tp_size > 1

- Link: https://github.com/sgl-project/sglang/pull/13730
- Status/date: merged / 2025-12-12
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 20 files, +285/-16, 617 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[bugfix] fix TBO crashes when attn_tp_size > 1"; model line: Ring 2.5; category: bug fix; main diff: `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/model_executor/forward_batch_info.py`, `python/sglang/srt/models/bailing_moe.py`; technical summary: Covers "[bugfix] fix TBO crashes when attn_tp_size > 1"; the main implementation surface is `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/model_executor/forward_batch_info.py`, `python/sglang/srt/models/bailing_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/communicator.py` modified +14/-1 (15 lines); hunks: -217,14 +217,16 @@ class _LayerModeComputationContext:; -273,6 +275,15 @@ def _compute_mlp_mode(cls, context: _LayerModeComputationCo...; symbols: _LayerModeComputationContext, previous_layer, _compute_mlp_mode, _should_gather_for_tbo, touching `_LayerModeComputationContext, previous_layer, _compute_mlp_mode`; `python/sglang/srt/model_executor/forward_batch_info.py` modified +9/-0 (9 lines); hunks: -376,6 +376,7 @@ class ForwardBatch:; -852,6 +853,14 @@ def prepare_mlp_sync_batch(self, model_runner: ModelRunner):; symbols: ForwardBatch, prepare_mlp_sync_batch, _pad_inputs_to_size, touching `ForwardBatch, prepare_mlp_sync_batch, _pad_inputs_to_size`; `python/sglang/srt/models/bailing_moe.py` modified +4/-0 (4 lines); hunks: -582,12 +582,16 @@ def __init__(; symbols: __init__, touching `__init__`; `python/sglang/srt/models/falcon_h1.py` modified +3/-1 (4 lines); hunks: -198,15 +198,17 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/layers/communicator.py` modified +14/-1 (15 lines); hunks: -217,14 +217,16 @@ class _LayerModeComputationContext:; -273,6 +275,15 @@ def _compute_mlp_mode(cls, context: _LayerModeComputationCo...; symbols: _LayerModeComputationContext, previous_layer, _compute_mlp_mode, _should_gather_for_tbo
  - `python/sglang/srt/model_executor/forward_batch_info.py` modified +9/-0 (9 lines); hunks: -376,6 +376,7 @@ class ForwardBatch:; -852,6 +853,14 @@ def prepare_mlp_sync_batch(self, model_runner: ModelRunner):; symbols: ForwardBatch, prepare_mlp_sync_batch, _pad_inputs_to_size
  - `python/sglang/srt/models/bailing_moe.py` modified +4/-0 (4 lines); hunks: -582,12 +582,16 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/falcon_h1.py` modified +3/-1 (4 lines); hunks: -198,15 +198,17 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/longcat_flash.py` modified +4/-0 (4 lines); hunks: -380,6 +380,8 @@ def __init__(; -398,6 +400,8 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/communicator.py
@@ -217,14 +217,16 @@ class _LayerModeComputationContext:
+    is_next_layer_sparse: Optional[bool]
+            num_layers=self.num_layers,
-            num_layers=self.num_layers,
+            is_next_layer_sparse=self.is_layer_sparse,
@@ -273,6 +275,15 @@ def _compute_mlp_mode(cls, context: _LayerModeComputationContext):
+    @classmethod
diff -- python/sglang/srt/model_executor/forward_batch_info.py
@@ -376,6 +376,7 @@ class ForwardBatch:
+    tbo_padded_len: Optional[int] = None
@@ -852,6 +853,14 @@ def prepare_mlp_sync_batch(self, model_runner: ModelRunner):
+        # TODO: The following is added to make sure sub-batch input_ids are padded
+        # to the multiple of attn_tp_size. It can likely be removed after this
+        # function is refactored and merged into the Scheduler.
+        if self.tbo_children:
diff -- python/sglang/srt/models/bailing_moe.py
@@ -582,12 +582,16 @@ def __init__(
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/communicator.py` modified +14/-1; `python/sglang/srt/model_executor/forward_batch_info.py` modified +9/-0; `python/sglang/srt/models/bailing_moe.py` modified +4/-0; `python/sglang/srt/models/falcon_h1.py` modified +3/-1; `python/sglang/srt/models/longcat_flash.py` modified +4/-0; `python/sglang/srt/models/qwen3_next.py` modified +4/-0
- Risk and verification: The diff ships test coverage in `test/srt/ep/test_deepep_small.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #15526 - Optimize Bailing-MoE with FlashInfer Fused All-Reduce

- Link: https://github.com/sgl-project/sglang/pull/15526
- Status/date: merged / 2025-12-21
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +58/-20, 182 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Optimize Bailing-MoE with FlashInfer Fused All-Reduce"; model line: Ring 2.5; category: performance/backend optimization; main diff: `python/sglang/srt/models/bailing_moe.py`; technical summary: Covers "Optimize Bailing-MoE with FlashInfer Fused All-Reduce"; the main implementation surface is `python/sglang/srt/models/bailing_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/bailing_moe.py` modified +58/-20 (78 lines); hunks: -19,7 +19,7; -54,7 +54,11; symbols: forward, forward_normal_dual_stream, forward_normal, touching `forward, forward_normal_dual_stream, forward_normal`.
- Code diff details:
  - `python/sglang/srt/models/bailing_moe.py` modified +58/-20 (78 lines); hunks: -19,7 +19,7; -54,7 +54,11; symbols: forward, forward_normal_dual_stream, forward_normal
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/bailing_moe.py
@@ -19,7 +19,7 @@
-from typing import Iterable, Optional, Tuple, Union
+from typing import Iterable, List, Optional, Tuple, Union
@@ -54,7 +54,11 @@
-from sglang.srt.layers.moe import get_deepep_mode, get_moe_a2a_backend
+from sglang.srt.layers.moe import (
+    get_deepep_mode,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/bailing_moe.py` modified +58/-20
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/bailing_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #15835 - [Feature] JIT Fused QK norm + qk norm clean up

- Link: https://github.com/sgl-project/sglang/pull/15835
- Status/date: merged / 2025-12-28
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 15 files, +827/-127, 1151 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feature] JIT Fused QK norm + qk norm clean up"; model line: Ring 2.5; category: performance/backend optimization; main diff: `python/sglang/srt/models/utils.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen3.py`; technical summary: Covers "[Feature] JIT Fused QK norm + qk norm clean up"; the main implementation surface is `python/sglang/srt/models/utils.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/utils.py` modified +80/-5 (85 lines); hunks: -11,24 +11,27; -113,6 +116,8 @@ def create_fused_set_kv_buffer_arg(; symbols: create_fused_set_kv_buffer_arg, rot_pos_ids, apply_qk_norm, touching `create_fused_set_kv_buffer_arg, rot_pos_ids, apply_qk_norm`; `python/sglang/srt/models/qwen3_moe.py` modified +9/-27 (36 lines); hunks: -57,12 +57,12; -498,31 +498,6 @@ def __init__(; symbols: __init__, _apply_qk_norm, op_prepare, apply_qk_norm_rope, touching `__init__, _apply_qk_norm, op_prepare`; `python/sglang/srt/models/qwen3.py` modified +9/-24 (33 lines); hunks: -21,14 +21,14; -138,32 +138,17 @@ def __init__(; symbols: __init__, _apply_qk_norm, forward_prepare_native, touching `__init__, _apply_qk_norm, forward_prepare_native`; `python/sglang/srt/models/bailing_moe.py` modified +9/-23 (32 lines); hunks: -75,6 +75,7; -507,28 +508,6 @@ def __init__(; symbols: __init__, _apply_qk_norm, forward, touching `__init__, _apply_qk_norm, forward`.
- Code diff details:
  - `python/sglang/srt/models/utils.py` modified +80/-5 (85 lines); hunks: -11,24 +11,27; -113,6 +116,8 @@ def create_fused_set_kv_buffer_arg(; symbols: create_fused_set_kv_buffer_arg, rot_pos_ids, apply_qk_norm
  - `python/sglang/srt/models/qwen3_moe.py` modified +9/-27 (36 lines); hunks: -57,12 +57,12; -498,31 +498,6 @@ def __init__(; symbols: __init__, _apply_qk_norm, op_prepare, apply_qk_norm_rope
  - `python/sglang/srt/models/qwen3.py` modified +9/-24 (33 lines); hunks: -21,14 +21,14; -138,32 +138,17 @@ def __init__(; symbols: __init__, _apply_qk_norm, forward_prepare_native
  - `python/sglang/srt/models/bailing_moe.py` modified +9/-23 (32 lines); hunks: -75,6 +75,7; -507,28 +508,6 @@ def __init__(; symbols: __init__, _apply_qk_norm, forward
  - `python/sglang/srt/models/glm4_moe.py` modified +9/-23 (32 lines); hunks: -75,6 +75,7; -250,28 +251,6 @@ def __init__(; symbols: __init__, _apply_qk_norm, op_prepare, forward_prepare
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/utils.py
@@ -11,24 +11,27 @@
+from __future__ import annotations
-from typing import Any, Optional
+from typing import TYPE_CHECKING, Any, Optional, Tuple
+from sglang.jit_kernel.norm import can_use_fused_inplace_qknorm, fused_inplace_qknorm
+from sglang.jit_kernel.utils import register_jit_op
+from sglang.srt.environ import envs
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -57,12 +57,12 @@
-from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
+    apply_qk_norm,
@@ -498,31 +498,6 @@ def __init__(
-    def _apply_qk_norm(
-        self, q: torch.Tensor, k: torch.Tensor
-    ) -> Tuple[torch.Tensor, torch.Tensor]:
diff -- python/sglang/srt/models/qwen3.py
@@ -21,14 +21,14 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/utils.py` modified +80/-5; `python/sglang/srt/models/qwen3_moe.py` modified +9/-27; `python/sglang/srt/models/qwen3.py` modified +9/-24; `python/sglang/srt/models/bailing_moe.py` modified +9/-23; `python/sglang/srt/models/glm4_moe.py` modified +9/-23; `python/sglang/srt/models/llada2.py` modified +9/-23
- Risk and verification: The diff ships test coverage in `python/sglang/jit_kernel/tests/test_qknorm.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #13715 - Fix EPLB + FP4 Quantization Compatibility Issue

- Link: https://github.com/sgl-project/sglang/pull/13715
- Status/date: merged / 2026-01-10
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +49/-3, 157 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix EPLB + FP4 Quantization Compatibility Issue"; model line: Ring 2.5; category: bug fix; main diff: `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/qwen2_moe.py`; technical summary: Covers "Fix EPLB + FP4 Quantization Compatibility Issue"; the main implementation surface is `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/qwen2_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/moe/utils.py` modified +12/-0 (12 lines); hunks: -249,6 +249,18 @@ def get_tbo_token_distribution_threshold() -> float:; symbols: get_tbo_token_distribution_threshold, filter_moe_weight_param_global_expert, should_use_flashinfer_cutlass_moe_fp4_allgather, touching `get_tbo_token_distribution_threshold, filter_moe_weight_param_global_expert, should_use_flashinfer_cutlass_moe_fp4_allgather`; `python/sglang/srt/models/deepseek_v2.py` modified +7/-1 (8 lines); hunks: -103,7 +103,10; -587,6 +590,9 @@ def get_moe_weights(self):; symbols: get_moe_weights, forward, touching `get_moe_weights, forward`; `python/sglang/srt/models/qwen2_moe.py` modified +7/-1 (8 lines); hunks: -58,7 +58,10; -223,6 +226,9 @@ def get_moe_weights(self):; symbols: get_moe_weights, _forward_shared_experts, touching `get_moe_weights, _forward_shared_experts`; `python/sglang/srt/models/qwen3_moe.py` modified +7/-1 (8 lines); hunks: -51,7 +51,10; -281,6 +284,9 @@ def get_moe_weights(self):; symbols: get_moe_weights, forward_normal, touching `get_moe_weights, forward_normal`.
- Code diff details:
  - `python/sglang/srt/layers/moe/utils.py` modified +12/-0 (12 lines); hunks: -249,6 +249,18 @@ def get_tbo_token_distribution_threshold() -> float:; symbols: get_tbo_token_distribution_threshold, filter_moe_weight_param_global_expert, should_use_flashinfer_cutlass_moe_fp4_allgather
  - `python/sglang/srt/models/deepseek_v2.py` modified +7/-1 (8 lines); hunks: -103,7 +103,10; -587,6 +590,9 @@ def get_moe_weights(self):; symbols: get_moe_weights, forward
  - `python/sglang/srt/models/qwen2_moe.py` modified +7/-1 (8 lines); hunks: -58,7 +58,10; -223,6 +226,9 @@ def get_moe_weights(self):; symbols: get_moe_weights, _forward_shared_experts
  - `python/sglang/srt/models/qwen3_moe.py` modified +7/-1 (8 lines); hunks: -51,7 +51,10; -281,6 +284,9 @@ def get_moe_weights(self):; symbols: get_moe_weights, forward_normal
  - `python/sglang/srt/models/bailing_moe.py` modified +4/-0 (4 lines); hunks: -63,6 +63,7; -324,6 +325,9 @@ def get_moe_weights(self):; symbols: get_moe_weights, _forward_shared_experts
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/utils.py
@@ -249,6 +249,18 @@ def get_tbo_token_distribution_threshold() -> float:
+def filter_moe_weight_param_global_expert(name, x, num_local_experts):
+    """
+    Filter out for MoE expert parameters that requires global expert.
+    """
+    return (
+        not getattr(x, "_sglang_require_global_experts", False)
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -103,7 +103,10 @@
-from sglang.srt.layers.moe.utils import RoutingMethodType
+from sglang.srt.layers.moe.utils import (
+    RoutingMethodType,
+    filter_moe_weight_param_global_expert,
+)
@@ -587,6 +590,9 @@ def get_moe_weights(self):
diff -- python/sglang/srt/models/qwen2_moe.py
@@ -58,7 +58,10 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/utils.py` modified +12/-0; `python/sglang/srt/models/deepseek_v2.py` modified +7/-1; `python/sglang/srt/models/qwen2_moe.py` modified +7/-1; `python/sglang/srt/models/qwen3_moe.py` modified +7/-1; `python/sglang/srt/models/bailing_moe.py` modified +4/-0; `python/sglang/srt/models/glm4_moe.py` modified +4/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17570 - Use attn tp group in embedding for more models

- Link: https://github.com/sgl-project/sglang/pull/17570
- Status/date: merged / 2026-01-24
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 19 files, +19/-19, 171 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Use attn tp group in embedding for more models"; model line: Ring 2.5; category: model implementation change; main diff: `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/bailing_moe_nextn.py`, `python/sglang/srt/models/falcon_h1.py`; technical summary: Covers "Use attn tp group in embedding for more models"; the main implementation surface is `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/bailing_moe_nextn.py`, `python/sglang/srt/models/falcon_h1.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/bailing_moe.py` modified +1/-1 (2 lines); hunks: -717,7 +717,7 @@ def __init__(; symbols: __init__, touching `__init__`; `python/sglang/srt/models/bailing_moe_nextn.py` modified +1/-1 (2 lines); hunks: -62,7 +62,7 @@ def __init__(; symbols: __init__, touching `__init__`; `python/sglang/srt/models/falcon_h1.py` modified +1/-1 (2 lines); hunks: -394,7 +394,7 @@ def __init__(; symbols: __init__, get_layer, touching `__init__, get_layer`; `python/sglang/srt/models/glm4.py` modified +1/-1 (2 lines); hunks: -307,7 +307,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/bailing_moe.py` modified +1/-1 (2 lines); hunks: -717,7 +717,7 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/bailing_moe_nextn.py` modified +1/-1 (2 lines); hunks: -62,7 +62,7 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/falcon_h1.py` modified +1/-1 (2 lines); hunks: -394,7 +394,7 @@ def __init__(; symbols: __init__, get_layer
  - `python/sglang/srt/models/glm4.py` modified +1/-1 (2 lines); hunks: -307,7 +307,7 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/glm4_moe.py` modified +1/-1 (2 lines); hunks: -895,7 +895,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/bailing_moe.py
@@ -717,7 +717,7 @@ def __init__(
-                enable_tp=not is_dp_attention_enabled(),
+                use_attn_tp_group=is_dp_attention_enabled(),
diff -- python/sglang/srt/models/bailing_moe_nextn.py
@@ -62,7 +62,7 @@ def __init__(
-            enable_tp=not is_dp_attention_enabled(),
+            use_attn_tp_group=is_dp_attention_enabled(),
diff -- python/sglang/srt/models/falcon_h1.py
@@ -394,7 +394,7 @@ def __init__(
-            enable_tp=not is_dp_attention_enabled(),
+            use_attn_tp_group=is_dp_attention_enabled(),
diff -- python/sglang/srt/models/glm4.py
@@ -307,7 +307,7 @@ def __init__(
-                enable_tp=not is_dp_attention_enabled(),
+                use_attn_tp_group=is_dp_attention_enabled(),
diff -- python/sglang/srt/models/glm4_moe.py
@@ -895,7 +895,7 @@ def __init__(
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/bailing_moe.py` modified +1/-1; `python/sglang/srt/models/bailing_moe_nextn.py` modified +1/-1; `python/sglang/srt/models/falcon_h1.py` modified +1/-1; `python/sglang/srt/models/glm4.py` modified +1/-1; `python/sglang/srt/models/glm4_moe.py` modified +1/-1; `python/sglang/srt/models/glm4_moe_lite.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/bailing_moe_nextn.py`, `python/sglang/srt/models/falcon_h1.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17808 - Fix the scenario where eh_proj is quantized in the bailing moe nextn weights

- Link: https://github.com/sgl-project/sglang/pull/17808
- Status/date: merged / 2026-01-30
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +9/-2, 32 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix the scenario where eh_proj is quantized in the bailing moe nextn weights"; model line: Ring 2.5; category: bug fix; main diff: `python/sglang/srt/models/bailing_moe_nextn.py`; technical summary: Covers "Fix the scenario where eh_proj is quantized in the bailing moe nextn weights"; the main implementation surface is `python/sglang/srt/models/bailing_moe_nextn.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/bailing_moe_nextn.py` modified +9/-2 (11 lines); hunks: -28,6 +28,7; -69,7 +70,13 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `python/sglang/srt/models/bailing_moe_nextn.py` modified +9/-2 (11 lines); hunks: -28,6 +28,7; -69,7 +70,13 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/bailing_moe_nextn.py
@@ -28,6 +28,7 @@
+from sglang.srt.layers.linear import ReplicatedLinear
@@ -69,7 +70,13 @@ def __init__(
-        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
+        self.eh_proj = ReplicatedLinear(
+            2 * config.hidden_size,
+            config.hidden_size,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/bailing_moe_nextn.py` modified +9/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/bailing_moe_nextn.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #15119 - feat: Add Ling Flash v2.0 support for Eagle3

- Link: https://github.com/sgl-project/sglang/pull/15119
- Status/date: merged / 2026-02-01
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +30/-1, 76 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "feat: Add Ling Flash v2.0 support for Eagle3"; model line: Ring 2.5; category: performance/backend optimization; main diff: `python/sglang/srt/models/bailing_moe.py`; technical summary: Covers "feat: Add Ling Flash v2.0 support for Eagle3"; the main implementation surface is `python/sglang/srt/models/bailing_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/bailing_moe.py` modified +30/-1 (31 lines); hunks: -738,6 +738,8 @@ def __init__(; -760,6 +762,10 @@ def forward(; symbols: __init__, forward, BailingMoEForCausalLM, touching `__init__, forward, BailingMoEForCausalLM`.
- Code diff details:
  - `python/sglang/srt/models/bailing_moe.py` modified +30/-1 (31 lines); hunks: -738,6 +738,8 @@ def __init__(; -760,6 +762,10 @@ def forward(; symbols: __init__, forward, BailingMoEForCausalLM
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/bailing_moe.py
@@ -738,6 +738,8 @@ def __init__(
+        self.layers_to_capture = []
@@ -760,6 +762,10 @@ def forward(
+                if i in self.layers_to_capture:
+                    aux_hidden_states.append(
+                        hidden_states if residual is None else hidden_states + residual
+                    )
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/bailing_moe.py` modified +30/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/bailing_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18598 - Support LingV2_5 model

- Link: https://github.com/sgl-project/sglang/pull/18598
- Status/date: merged / 2026-02-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 16 files, +4042/-23, 4377 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support LingV2_5 model"; model line: Ring 2.5; category: model support/runtime entry; main diff: `python/sglang/srt/models/bailing_moe_linear.py`, `python/sglang/srt/layers/attention/linear/seg_la.py`, `python/sglang/srt/layers/attention/linear/lightning_attn.py`; technical summary: Covers "Support LingV2_5 model"; the main implementation surface is `python/sglang/srt/models/bailing_moe_linear.py`, `python/sglang/srt/layers/attention/linear/seg_la.py`, `python/sglang/srt/layers/attention/linear/lightning_attn.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/bailing_moe_linear.py` added +1571/-0 (1571 lines); hunks: -0,0 +1,1571; symbols: DsV3MLA, __init__, is_linear_layer, is_pp_missing_parameter, touching `DsV3MLA, __init__, is_linear_layer`; `python/sglang/srt/layers/attention/linear/seg_la.py` added +909/-0 (909 lines); hunks: -0,0 +1,909; symbols: SegLaMeta, seg_la_kernel, seg_la_p_kernel, seg_la_s_kernel, touching `SegLaMeta, seg_la_kernel, seg_la_p_kernel`; `python/sglang/srt/layers/attention/linear/lightning_attn.py` added +767/-0 (767 lines); hunks: -0,0 +1,767; symbols: _fwd_diag_kernel, _fwd_kv_parallel, _fwd_kv_reduce, _fwd_none_diag_kernel, touching `_fwd_diag_kernel, _fwd_kv_parallel, _fwd_kv_reduce`; `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +369/-0 (369 lines); hunks: -1,3 +1,5; -14,6 +16,12; symbols: forward_extend, LightningAttentionBackend, __init__, init_forward_metadata, touching `forward_extend, LightningAttentionBackend, __init__`.
- Code diff details:
  - `python/sglang/srt/models/bailing_moe_linear.py` added +1571/-0 (1571 lines); hunks: -0,0 +1,1571; symbols: DsV3MLA, __init__, is_linear_layer, is_pp_missing_parameter
  - `python/sglang/srt/layers/attention/linear/seg_la.py` added +909/-0 (909 lines); hunks: -0,0 +1,909; symbols: SegLaMeta, seg_la_kernel, seg_la_p_kernel, seg_la_s_kernel
  - `python/sglang/srt/layers/attention/linear/lightning_attn.py` added +767/-0 (767 lines); hunks: -0,0 +1,767; symbols: _fwd_diag_kernel, _fwd_kv_parallel, _fwd_kv_reduce, _fwd_none_diag_kernel
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +369/-0 (369 lines); hunks: -1,3 +1,5; -14,6 +16,12; symbols: forward_extend, LightningAttentionBackend, __init__, init_forward_metadata
  - `python/sglang/srt/configs/bailing_hybrid.py` added +188/-0 (188 lines); hunks: -0,0 +1,188; symbols: HybridLayerType, BailingHybridConfig, __init__, layers_block_type
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/bailing_moe_linear.py
@@ -0,0 +1,1571 @@
+# coding=utf-8
+# Copyright 2023 Antgroup and The HuggingFace Inc. team. All rights reserved.
+import copy
+import logging
+from typing import Callable, Iterable, Optional, Set, Tuple, Union
+import torch
diff -- python/sglang/srt/layers/attention/linear/seg_la.py
@@ -0,0 +1,909 @@
+# -*- coding: utf-8 -*-
+"""
+Copyright (c) Ant Financial Service Group and its affiliates.
+"""
+# Copied from https://code.alipay.com/pia/PainlessInferenceAcceleration/blob/v0.0.6/flood/flood/ops/seg_la.py
+from dataclasses import dataclass
diff -- python/sglang/srt/layers/attention/linear/lightning_attn.py
@@ -0,0 +1,767 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/bailing_moe_linear.py` added +1571/-0; `python/sglang/srt/layers/attention/linear/seg_la.py` added +909/-0; `python/sglang/srt/layers/attention/linear/lightning_attn.py` added +767/-0; `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +369/-0; `python/sglang/srt/configs/bailing_hybrid.py` added +188/-0; `python/sglang/srt/models/bailing_moe_nextn.py` modified +90/-15
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/__init__.py`, `python/sglang/srt/configs/bailing_hybrid.py`, `python/sglang/srt/configs/model_config.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18793 - Cleanup debug log for Ring model

- Link: https://github.com/sgl-project/sglang/pull/18793
- Status/date: merged / 2026-02-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +9/-11, 69 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Cleanup debug log for Ring model"; model line: Ring 2.5; category: bug fix; main diff: `python/sglang/srt/models/bailing_moe_linear.py`, `python/sglang/srt/configs/mamba_utils.py`; technical summary: Covers "Cleanup debug log for Ring model"; the main implementation surface is `python/sglang/srt/models/bailing_moe_linear.py`, `python/sglang/srt/configs/mamba_utils.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/bailing_moe_linear.py` modified +8/-10 (18 lines); hunks: -77,6 +77,7; -423,7 +424,7 @@ def __init__(; symbols: __init__, layer_fn, post_load_weights, touching `__init__, layer_fn, post_load_weights`; `python/sglang/srt/configs/mamba_utils.py` modified +1/-1 (2 lines); hunks: -102,7 +102,7 @@ def mamba2_state_dtype(config=None) -> Mamba2StateDType:; symbols: mamba2_state_dtype, touching `mamba2_state_dtype`.
- Code diff details:
  - `python/sglang/srt/models/bailing_moe_linear.py` modified +8/-10 (18 lines); hunks: -77,6 +77,7; -423,7 +424,7 @@ def __init__(; symbols: __init__, layer_fn, post_load_weights
  - `python/sglang/srt/configs/mamba_utils.py` modified +1/-1 (2 lines); hunks: -102,7 +102,7 @@ def mamba2_state_dtype(config=None) -> Mamba2StateDType:; symbols: mamba2_state_dtype
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/bailing_moe_linear.py
@@ -77,6 +77,7 @@
+from sglang.srt.utils.common import rank0_log
@@ -423,7 +424,7 @@ def __init__(
-        logger.info(f"linear_backend in bailing_moe_linear: {self.linear_backend}")
+        logger.debug(f"linear_backend in bailing_moe_linear: {self.linear_backend}")
@@ -740,7 +741,7 @@ def __init__(
-                logger.info(f"==={layer_id=} use gqa")
diff -- python/sglang/srt/configs/mamba_utils.py
@@ -102,7 +102,7 @@ def mamba2_state_dtype(config=None) -> Mamba2StateDType:
-    logger.info(f"Mamba2 state dtype: conv_dtype={conv_dtype}, ssm_dtype={ssm_dtype}")
+    logger.debug(f"Mamba2 state dtype: conv_dtype={conv_dtype}, ssm_dtype={ssm_dtype}")
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/bailing_moe_linear.py` modified +8/-10; `python/sglang/srt/configs/mamba_utils.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/mamba_utils.py`, `python/sglang/srt/models/bailing_moe_linear.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18829 - Add ci test for ring model

- Link: https://github.com/sgl-project/sglang/pull/18829
- Status/date: merged / 2026-02-14
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +66/-1, 103 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add ci test for ring model"; model line: Ring 2.5; category: docs/tests/CI; main diff: `test/registered/8-gpu-models/test_ring_2_5_1t.py`, `python/sglang/test/accuracy_test_runner.py`; technical summary: Covers "Add ci test for ring model"; the main implementation surface is `test/registered/8-gpu-models/test_ring_2_5_1t.py`, `python/sglang/test/accuracy_test_runner.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/8-gpu-models/test_ring_2_5_1t.py` added +53/-0 (53 lines); hunks: -0,0 +1,53; symbols: TestRing2_5_1T, test_ring_2_5_1t, touching `TestRing2_5_1T, test_ring_2_5_1t`; `python/sglang/test/accuracy_test_runner.py` modified +13/-1 (14 lines); hunks: -26,6 +26,7 @@ class AccuracyTestParams:; -81,6 +82,7 @@ def _run_simple_eval(; symbols: AccuracyTestParams, _run_simple_eval, run_accuracy_test, touching `AccuracyTestParams, _run_simple_eval, run_accuracy_test`.
- Code diff details:
  - `test/registered/8-gpu-models/test_ring_2_5_1t.py` added +53/-0 (53 lines); hunks: -0,0 +1,53; symbols: TestRing2_5_1T, test_ring_2_5_1t
  - `python/sglang/test/accuracy_test_runner.py` modified +13/-1 (14 lines); hunks: -26,6 +26,7 @@ class AccuracyTestParams:; -81,6 +82,7 @@ def _run_simple_eval(; symbols: AccuracyTestParams, _run_simple_eval, run_accuracy_test
- Key code excerpts:

```diff
diff -- test/registered/8-gpu-models/test_ring_2_5_1t.py
@@ -0,0 +1,53 @@
+import unittest
+from sglang.test.accuracy_test_runner import AccuracyTestParams
+from sglang.test.ci.ci_register import register_cuda_ci
+from sglang.test.run_combined_tests import run_combined_tests
+from sglang.test.test_utils import ModelLaunchSettings
+# register_cuda_ci(est_time=1000, suite="nightly-8-gpu-common", nightly=True)
diff -- python/sglang/test/accuracy_test_runner.py
@@ -26,6 +26,7 @@ class AccuracyTestParams:
+    top_p: Optional[float] = None
@@ -81,6 +82,7 @@ def _run_simple_eval(
+    top_p: Optional[float] = None,
@@ -117,6 +119,9 @@ def _run_simple_eval(
+        if top_p is not None:
+            args.top_p = top_p
```

- Reviewed files:
  - tests: `test/registered/8-gpu-models/test_ring_2_5_1t.py` added +53/-0; `python/sglang/test/accuracy_test_runner.py` modified +13/-1
- Risk and verification: The diff ships test coverage in `python/sglang/test/accuracy_test_runner.py`, `test/registered/8-gpu-models/test_ring_2_5_1t.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18860 - update pre-commit config

- Link: https://github.com/sgl-project/sglang/pull/18860
- Status/date: merged / 2026-02-15
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 135 files, +239/-198, 1632 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "update pre-commit config"; model line: Ring 2.5; category: model implementation change; main diff: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/models/pixtral.py`, `python/sglang/srt/layers/quantization/modelslim/modelslim_moe.py`; technical summary: Covers "update pre-commit config"; the main implementation surface is `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/models/pixtral.py`, `python/sglang/srt/layers/quantization/modelslim/modelslim_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +6/-6 (12 lines); hunks: -670,9 +670,9 @@ def forward_decode(; -744,9 +744,9 @@ def forward_extend(; symbols: forward_decode, forward_extend, touching `forward_decode, forward_extend`; `python/sglang/srt/models/pixtral.py` modified +6/-2 (8 lines); hunks: -23,11 +23,15; `python/sglang/srt/layers/quantization/modelslim/modelslim_moe.py` modified +2/-4 (6 lines); hunks: -63,11 +63,9 @@ def get_moe_method(; symbols: get_moe_method, touching `get_moe_method`; `test/registered/layers/test_fla_layernorm_guard.py` modified +4/-1 (5 lines); hunks: -10,7 +10,10.
- Code diff details:
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +6/-6 (12 lines); hunks: -670,9 +670,9 @@ def forward_decode(; -744,9 +744,9 @@ def forward_extend(; symbols: forward_decode, forward_extend
  - `python/sglang/srt/models/pixtral.py` modified +6/-2 (8 lines); hunks: -23,11 +23,15
  - `python/sglang/srt/layers/quantization/modelslim/modelslim_moe.py` modified +2/-4 (6 lines); hunks: -63,11 +63,9 @@ def get_moe_method(; symbols: get_moe_method
  - `test/registered/layers/test_fla_layernorm_guard.py` modified +4/-1 (5 lines); hunks: -10,7 +10,10
  - `python/sglang/srt/layers/attention/dual_chunk_flashattention_backend.py` modified +2/-2 (4 lines); hunks: -1,6 +1,6
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py
@@ -670,9 +670,9 @@ def forward_decode(
-        (q_proj_states, k_proj_states, v_proj_states) = mixed_qkv
-        (q_conv_weights, k_conv_weights, v_conv_weights) = layer.conv_weights
-        (q_conv_bias, k_conv_bias, v_conv_bias) = layer.bias
+        q_proj_states, k_proj_states, v_proj_states = mixed_qkv
+        q_conv_weights, k_conv_weights, v_conv_weights = layer.conv_weights
+        q_conv_bias, k_conv_bias, v_conv_bias = layer.bias
diff -- python/sglang/srt/models/pixtral.py
@@ -23,11 +23,15 @@
-from transformers.models.pixtral.modeling_pixtral import PixtralRotaryEmbedding
+from transformers.models.pixtral.modeling_pixtral import (
+    PixtralRotaryEmbedding,
+)
-from transformers.models.pixtral.modeling_pixtral import position_ids_in_meshgrid
+from transformers.models.pixtral.modeling_pixtral import (
diff -- python/sglang/srt/layers/quantization/modelslim/modelslim_moe.py
@@ -63,11 +63,9 @@ def get_moe_method(
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +6/-6; `python/sglang/srt/models/pixtral.py` modified +6/-2; `python/sglang/srt/layers/quantization/modelslim/modelslim_moe.py` modified +2/-4; `python/sglang/srt/layers/attention/dual_chunk_flashattention_backend.py` modified +2/-2; `python/sglang/srt/models/qwen3_next.py` modified +2/-2; `python/sglang/srt/multimodal/processors/ernie45_vl.py` modified +3/-1
  - tests: `test/registered/layers/test_fla_layernorm_guard.py` modified +4/-1
- Risk and verification: The diff ships test coverage in `sgl-kernel/tests/test_per_token_group_quant_8bit.py`, `test/manual/test_vlm_accuracy.py`, `test/registered/attention/test_triton_sliding_window.py`, `test/registered/layers/test_fla_layernorm_guard.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #19201 - Add server CUDA graph warmup CI step for cold H200 nodes

- Link: https://github.com/sgl-project/sglang/pull/19201
- Status/date: merged / 2026-02-25
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +739/-7, 771 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add server CUDA graph warmup CI step for cold H200 nodes"; model line: Ring 2.5; category: performance/backend optimization; main diff: `scripts/ci/cuda/warmup_deep_gemm.py`, `scripts/ci/cuda/warmup_server.py`, `.github/workflows/pr-test.yml`; technical summary: Covers "Add server CUDA graph warmup CI step for cold H200 nodes"; the main implementation surface is `scripts/ci/cuda/warmup_deep_gemm.py`, `scripts/ci/cuda/warmup_server.py`, `.github/workflows/pr-test.yml`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `scripts/ci/cuda/warmup_deep_gemm.py` added +399/-0 (399 lines); hunks: -0,0 +1,399; symbols: get_config_json, is_deepseek_v2v3, compute_deepseek_v2v3_shapes, get_architecture_key, touching `get_config_json, is_deepseek_v2v3, compute_deepseek_v2v3_shapes`; `scripts/ci/cuda/warmup_server.py` added +313/-0 (313 lines); hunks: -0,0 +1,313; symbols: get_version_key, get_marker_path, check_marker, write_marker, touching `get_version_key, get_marker_path, check_marker`; `.github/workflows/pr-test.yml` modified +27/-6 (33 lines); hunks: -1326,14 +1326,22 @@ jobs:; -1472,6 +1480,19 @@ jobs:; `test/registered/8-gpu-models/test_ring_2_5_1t.py` modified +0/-1 (1 lines); hunks: -20,7 +20,6 @@ class TestRing2_5_1T(unittest.TestCase):; symbols: TestRing2_5_1T, test_ring_2_5_1t, touching `TestRing2_5_1T, test_ring_2_5_1t`.
- Code diff details:
  - `scripts/ci/cuda/warmup_deep_gemm.py` added +399/-0 (399 lines); hunks: -0,0 +1,399; symbols: get_config_json, is_deepseek_v2v3, compute_deepseek_v2v3_shapes, get_architecture_key
  - `scripts/ci/cuda/warmup_server.py` added +313/-0 (313 lines); hunks: -0,0 +1,313; symbols: get_version_key, get_marker_path, check_marker, write_marker
  - `.github/workflows/pr-test.yml` modified +27/-6 (33 lines); hunks: -1326,14 +1326,22 @@ jobs:; -1472,6 +1480,19 @@ jobs:
  - `test/registered/8-gpu-models/test_ring_2_5_1t.py` modified +0/-1 (1 lines); hunks: -20,7 +20,6 @@ class TestRing2_5_1T(unittest.TestCase):; symbols: TestRing2_5_1T, test_ring_2_5_1t
- Key code excerpts:

```diff
diff -- scripts/ci/cuda/warmup_deep_gemm.py
@@ -0,0 +1,399 @@
+"""
+Lightweight DeepGEMM JIT compilation warmup without loading model weights.
+Reads model config.json from HF cache to derive kernel shapes, then compiles
+DeepGEMM kernels directly. This avoids the expensive model weight loading step
+that the full `sglang.compile_deep_gemm` requires.
+Supports DeepSeek V2/V3 family models. Falls back to `sglang.compile_deep_gemm`
diff -- scripts/ci/cuda/warmup_server.py
@@ -0,0 +1,313 @@
+"""
+Full server warmup to pre-warm Triton autotuning and CUDA graph capture.
+On cold H200 nodes (new nodes or after container recreation), CUDA graph capture
+triggers Triton autotuning which takes ~330s per server launch. This script
+launches actual servers with CUDA graphs enabled to cache the autotuned kernels,
+so subsequent test launches are fast (~30-60s).
diff -- .github/workflows/pr-test.yml
@@ -1326,14 +1326,22 @@ jobs:
```

- Reviewed files:
  - other: `scripts/ci/cuda/warmup_deep_gemm.py` added +399/-0; `scripts/ci/cuda/warmup_server.py` added +313/-0
  - ci: `.github/workflows/pr-test.yml` modified +27/-6
  - tests: `test/registered/8-gpu-models/test_ring_2_5_1t.py` modified +0/-1
- Risk and verification: The diff ships test coverage in `test/registered/8-gpu-models/test_ring_2_5_1t.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17784 - Upgrade transformers==5.3.0

- Link: https://github.com/sgl-project/sglang/pull/17784
- Status/date: merged / 2026-03-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 95 files, +1136/-343, 2752 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Upgrade transformers==5.3.0"; model line: Ring 2.5; category: model implementation change; main diff: `python/sglang/srt/models/gemma3_causal.py`, `python/sglang/srt/layers/rotary_embedding/factory.py`, `python/sglang/srt/configs/model_config.py`; technical summary: Covers "Upgrade transformers==5.3.0"; the main implementation surface is `python/sglang/srt/models/gemma3_causal.py`, `python/sglang/srt/layers/rotary_embedding/factory.py`, `python/sglang/srt/configs/model_config.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/gemma3_causal.py` modified +87/-14 (101 lines); hunks: -166,18 +166,36 @@ def __init__(; -325,9 +343,10 @@ class Gemma3RotaryEmbedding(nn.Module):; symbols: __init__, Gemma3RotaryEmbedding, _dynamic_frequency_update, touching `__init__, Gemma3RotaryEmbedding, _dynamic_frequency_update`; `python/sglang/srt/layers/rotary_embedding/factory.py` modified +63/-13 (76 lines); hunks: -2,6 +2,7; -26,6 +27,29; symbols: _get_rope_param, get_rope, touching `_get_rope_param, get_rope`; `python/sglang/srt/configs/model_config.py` modified +38/-18 (56 lines); hunks: -51,10 +51,20 @@ class ModelImpl(str, Enum):; -63,7 +73,7 @@ def is_deepseek_nsa(config: PretrainedConfig) -> bool:; symbols: ModelImpl, is_deepseek_nsa, _derive_model_shapes, touching `ModelImpl, is_deepseek_nsa, _derive_model_shapes`; `python/sglang/srt/models/qwen3_moe.py` modified +14/-7 (21 lines); hunks: -115,12 +115,19 @@ def compute_yarn_parameters(; -130,7 +137,7 @@ def compute_yarn_parameters(; symbols: compute_yarn_parameters, forward_prepare_native, apply_qk_norm_rope, __init__, touching `compute_yarn_parameters, forward_prepare_native, apply_qk_norm_rope`.
- Code diff details:
  - `python/sglang/srt/models/gemma3_causal.py` modified +87/-14 (101 lines); hunks: -166,18 +166,36 @@ def __init__(; -325,9 +343,10 @@ class Gemma3RotaryEmbedding(nn.Module):; symbols: __init__, Gemma3RotaryEmbedding, _dynamic_frequency_update
  - `python/sglang/srt/layers/rotary_embedding/factory.py` modified +63/-13 (76 lines); hunks: -2,6 +2,7; -26,6 +27,29; symbols: _get_rope_param, get_rope
  - `python/sglang/srt/configs/model_config.py` modified +38/-18 (56 lines); hunks: -51,10 +51,20 @@ class ModelImpl(str, Enum):; -63,7 +73,7 @@ def is_deepseek_nsa(config: PretrainedConfig) -> bool:; symbols: ModelImpl, is_deepseek_nsa, _derive_model_shapes
  - `python/sglang/srt/models/qwen3_moe.py` modified +14/-7 (21 lines); hunks: -115,12 +115,19 @@ def compute_yarn_parameters(; -130,7 +137,7 @@ def compute_yarn_parameters(; symbols: compute_yarn_parameters, forward_prepare_native, apply_qk_norm_rope, __init__
  - `python/sglang/srt/models/midashenglm.py` modified +6/-14 (20 lines); hunks: -476,20 +476,12 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/gemma3_causal.py
@@ -166,18 +166,36 @@ def __init__(
+        # In transformers v5, rope_parameters is nested per layer type:
+        #   {"sliding_attention": {"rope_theta": 10000}, "full_attention": {"rope_theta": 1000000}}
+        # In v4 it was flat: {"rope_type": "default", "rope_theta": ...}
+        rope_params = config.rope_parameters
+        is_nested = isinstance(rope_params, dict) and "full_attention" in rope_params
-            self.rope_theta = config.rope_local_base_freq
diff -- python/sglang/srt/layers/rotary_embedding/factory.py
@@ -2,6 +2,7 @@
+import logging
@@ -26,6 +27,29 @@
+logger = logging.getLogger(__name__)
+def _get_rope_param(rope_scaling, key, default, scaling_type):
+    """Get a parameter from rope_scaling dict, warn if missing.
+    In transformers v5, config.rope_scaling is an alias for rope_parameters
diff -- python/sglang/srt/configs/model_config.py
@@ -51,10 +51,20 @@ class ModelImpl(str, Enum):
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/gemma3_causal.py` modified +87/-14; `python/sglang/srt/layers/rotary_embedding/factory.py` modified +63/-13; `python/sglang/srt/configs/model_config.py` modified +38/-18; `python/sglang/srt/models/qwen3_moe.py` modified +14/-7; `python/sglang/srt/models/midashenglm.py` modified +6/-14; `python/sglang/srt/models/glm4.py` modified +3/-14
- Risk and verification: The diff ships test coverage in `python/sglang/test/runners.py`, `test/registered/core/test_score_api.py`, `test/registered/quant/test_awq.py`, `test/registered/rl/test_multi_instance_release_memory_occupation.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #9744 - [CPU] Add FP8 Bmm support

- Link: https://github.com/sgl-project/sglang/pull/9744
- Status/date: merged / 2026-03-19
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 14 files, +585/-84, 1014 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CPU] Add FP8 Bmm support"; model line: Ring 2.5; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/models/longcat_flash.py`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`; technical summary: Covers "[CPU] Add FP8 Bmm support"; the main implementation surface is `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/models/longcat_flash.py`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +44/-28 (72 lines); hunks: -16,6 +16,7; -268,18 +269,24 @@ def forward_absorb_prepare(; symbols: forward_absorb_prepare, forward_absorb_core, touching `forward_absorb_prepare, forward_absorb_core`; `python/sglang/srt/models/longcat_flash.py` modified +0/-12 (12 lines); hunks: -760,18 +760,6 @@ def post_load_weights(self, weight_names=None):; symbols: post_load_weights, touching `post_load_weights`; `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +0/-10 (10 lines); hunks: -46,8 +46,6; -583,14 +581,6 @@ def post_load_weights(; symbols: post_load_weights, touching `post_load_weights`; `python/sglang/srt/models/bailing_moe_linear.py` modified +0/-8 (8 lines); hunks: -1208,14 +1208,6 @@ def post_load_weights(self, is_nextn=False, weight_names=...; symbols: post_load_weights, touching `post_load_weights`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +44/-28 (72 lines); hunks: -16,6 +16,7; -268,18 +269,24 @@ def forward_absorb_prepare(; symbols: forward_absorb_prepare, forward_absorb_core
  - `python/sglang/srt/models/longcat_flash.py` modified +0/-12 (12 lines); hunks: -760,18 +760,6 @@ def post_load_weights(self, weight_names=None):; symbols: post_load_weights
  - `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +0/-10 (10 lines); hunks: -46,8 +46,6; -583,14 +581,6 @@ def post_load_weights(; symbols: post_load_weights
  - `python/sglang/srt/models/bailing_moe_linear.py` modified +0/-8 (8 lines); hunks: -1208,14 +1208,6 @@ def post_load_weights(self, is_nextn=False, weight_names=...; symbols: post_load_weights
  - `python/sglang/srt/models/longcat_flash_nextn.py` modified +0/-4 (4 lines); hunks: -426,10 +426,6 @@ def post_load_weights(self):; symbols: post_load_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py
@@ -16,6 +16,7 @@
+    _is_cpu,
@@ -268,18 +269,24 @@ def forward_absorb_prepare(
-            # fix bmm_fp8 error under cublas12.9 caused by bumpallocator, detail in pr#11612
-            q_nope_val, q_nope_scale = per_tensor_quant_mla_fp8(
-                q_nope.transpose(0, 1),
-                (
diff -- python/sglang/srt/models/longcat_flash.py
@@ -760,18 +760,6 @@ def post_load_weights(self, weight_names=None):
-                    # TODO: remove this after adding FP8 support in bmm cpu kernel
-                    if (
-                        _is_cpu
-                        and _is_cpu_amx_available
-                        and w.dtype == torch.float8_e4m3fn
-                    ):
diff -- python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py
@@ -46,8 +46,6 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +44/-28; `python/sglang/srt/models/longcat_flash.py` modified +0/-12; `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +0/-10; `python/sglang/srt/models/bailing_moe_linear.py` modified +0/-8; `python/sglang/srt/models/longcat_flash_nextn.py` modified +0/-4; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_cpu.py` modified +2/-1
  - other: `sgl-kernel/csrc/cpu/gemm_fp8.cpp` modified +310/-1; `sgl-kernel/csrc/cpu/bmm.cpp` modified +93/-11
- Risk and verification: The diff ships test coverage in `test/srt/cpu/test_bmm.py`, `test/srt/cpu/test_qkv_proj_with_rope.py`, `test/srt/run_suite.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #20316 - fix fused_set_kv_buffer for rope with Ling-v2

- Link: https://github.com/sgl-project/sglang/pull/20316
- Status/date: merged / 2026-03-23
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +6/-2, 29 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix fused_set_kv_buffer for rope with Ling-v2"; model line: Ring 2.5; category: bug fix; main diff: `python/sglang/srt/models/bailing_moe.py`; technical summary: Covers "fix fused_set_kv_buffer for rope with Ling-v2"; the main implementation surface is `python/sglang/srt/models/bailing_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/bailing_moe.py` modified +6/-2 (8 lines); hunks: -532,6 +532,10 @@ def forward(; -542,7 +546,7 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/bailing_moe.py` modified +6/-2 (8 lines); hunks: -532,6 +532,10 @@ def forward(; -542,7 +546,7 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/bailing_moe.py
@@ -532,6 +532,10 @@ def forward(
+        can_fuse_set_kv = (
+            self.head_dim == self.rotary_emb.rotary_dim
+            and enable_fused_set_kv_buffer(forward_batch)
+        )
@@ -542,7 +546,7 @@ def forward(
-                if enable_fused_set_kv_buffer(forward_batch)
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/bailing_moe.py` modified +6/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/bailing_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21751 - [CI] Fix ring test timeout

- Link: https://github.com/sgl-project/sglang/pull/21751
- Status/date: merged / 2026-03-31
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-0, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CI] Fix ring test timeout"; model line: Ring 2.5; category: bug fix; main diff: `test/registered/8-gpu-models/test_ring_2_5_1t.py`; technical summary: Covers "[CI] Fix ring test timeout"; the main implementation surface is `test/registered/8-gpu-models/test_ring_2_5_1t.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/8-gpu-models/test_ring_2_5_1t.py` modified +2/-0 (2 lines); hunks: -23,6 +23,8 @@ def test_ring_2_5_1t(self):; symbols: test_ring_2_5_1t, touching `test_ring_2_5_1t`.
- Code diff details:
  - `test/registered/8-gpu-models/test_ring_2_5_1t.py` modified +2/-0 (2 lines); hunks: -23,6 +23,8 @@ def test_ring_2_5_1t(self):; symbols: test_ring_2_5_1t
- Key code excerpts:

```diff
diff -- test/registered/8-gpu-models/test_ring_2_5_1t.py
@@ -23,6 +23,8 @@ def test_ring_2_5_1t(self):
+            "--watchdog-timeout",
+            "1800",
```

- Reviewed files:
  - tests: `test/registered/8-gpu-models/test_ring_2_5_1t.py` modified +2/-0
- Risk and verification: The diff ships test coverage in `test/registered/8-gpu-models/test_ring_2_5_1t.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22045 - [CI] Adjust CI server launch timeout

- Link: https://github.com/sgl-project/sglang/pull/22045
- Status/date: merged / 2026-04-03
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +5/-2, 41 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CI] Adjust CI server launch timeout"; model line: Ring 2.5; category: docs/tests/CI; main diff: `python/sglang/test/accuracy_test_runner.py`, `python/sglang/test/test_utils.py`, `test/registered/8-gpu-models/test_ring_2_5_1t.py`; technical summary: Covers "[CI] Adjust CI server launch timeout"; the main implementation surface is `python/sglang/test/accuracy_test_runner.py`, `python/sglang/test/test_utils.py`, `test/registered/8-gpu-models/test_ring_2_5_1t.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/test/accuracy_test_runner.py` modified +2/-2 (4 lines); hunks: -98,7 +98,7 @@ def _run_simple_eval(; -275,7 +275,7 @@ def _run_nemo_skills_eval(; symbols: _run_simple_eval, _run_nemo_skills_eval, touching `_run_simple_eval, _run_nemo_skills_eval`; `python/sglang/test/test_utils.py` modified +2/-0 (2 lines); hunks: -2181,12 +2181,14 @@ def __init__(; symbols: __init__, touching `__init__`; `test/registered/8-gpu-models/test_ring_2_5_1t.py` modified +1/-0 (1 lines); hunks: -33,6 +33,7 @@ def test_ring_2_5_1t(self):; symbols: test_ring_2_5_1t, touching `test_ring_2_5_1t`.
- Code diff details:
  - `python/sglang/test/accuracy_test_runner.py` modified +2/-2 (4 lines); hunks: -98,7 +98,7 @@ def _run_simple_eval(; -275,7 +275,7 @@ def _run_nemo_skills_eval(; symbols: _run_simple_eval, _run_nemo_skills_eval
  - `python/sglang/test/test_utils.py` modified +2/-0 (2 lines); hunks: -2181,12 +2181,14 @@ def __init__(; symbols: __init__
  - `test/registered/8-gpu-models/test_ring_2_5_1t.py` modified +1/-0 (1 lines); hunks: -33,6 +33,7 @@ def test_ring_2_5_1t(self):; symbols: test_ring_2_5_1t
- Key code excerpts:

```diff
diff -- python/sglang/test/accuracy_test_runner.py
@@ -98,7 +98,7 @@ def _run_simple_eval(
-            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
+            timeout=model.launch_timeout or DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
@@ -275,7 +275,7 @@ def _run_nemo_skills_eval(
-            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
+            timeout=model.launch_timeout or DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
diff -- python/sglang/test/test_utils.py
@@ -2181,12 +2181,14 @@ def __init__(
+        launch_timeout: Optional[float] = None,
+        self.launch_timeout = launch_timeout
diff -- test/registered/8-gpu-models/test_ring_2_5_1t.py
@@ -33,6 +33,7 @@ def test_ring_2_5_1t(self):
+                launch_timeout=1800,
```

- Reviewed files:
  - tests: `python/sglang/test/accuracy_test_runner.py` modified +2/-2; `python/sglang/test/test_utils.py` modified +2/-0; `test/registered/8-gpu-models/test_ring_2_5_1t.py` modified +1/-0
- Risk and verification: The diff ships test coverage in `python/sglang/test/accuracy_test_runner.py`, `python/sglang/test/test_utils.py`, `test/registered/8-gpu-models/test_ring_2_5_1t.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22267 - Move ring test to nightly

- Link: https://github.com/sgl-project/sglang/pull/22267
- Status/date: merged / 2026-04-07
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-2, 19 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Move ring test to nightly"; model line: Ring 2.5; category: docs/tests/CI; main diff: `test/registered/8-gpu-models/test_ring_2_5_1t.py`; technical summary: Covers "Move ring test to nightly"; the main implementation surface is `test/registered/8-gpu-models/test_ring_2_5_1t.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/8-gpu-models/test_ring_2_5_1t.py` modified +3/-2 (5 lines); hunks: -5,8 +5,7; -25,6 +24,8 @@ def test_ring_2_5_1t(self):; symbols: test_ring_2_5_1t, touching `test_ring_2_5_1t`.
- Code diff details:
  - `test/registered/8-gpu-models/test_ring_2_5_1t.py` modified +3/-2 (5 lines); hunks: -5,8 +5,7; -25,6 +24,8 @@ def test_ring_2_5_1t(self):; symbols: test_ring_2_5_1t
- Key code excerpts:

```diff
diff -- test/registered/8-gpu-models/test_ring_2_5_1t.py
@@ -5,8 +5,7 @@
-# register_cuda_ci(est_time=1000, suite="nightly-8-gpu-common", nightly=True)
-register_cuda_ci(est_time=1000, suite="stage-c-test-8-gpu-h200")
+register_cuda_ci(est_time=1800, suite="nightly-8-gpu-common", nightly=True)
@@ -25,6 +24,8 @@ def test_ring_2_5_1t(self):
+            "--soft-watchdog-timeout",
+            "1800",
```

- Reviewed files:
  - tests: `test/registered/8-gpu-models/test_ring_2_5_1t.py` modified +3/-2
- Risk and verification: The diff ships test coverage in `test/registered/8-gpu-models/test_ring_2_5_1t.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22305 - [CI] Update est_time for 64 tests based on actual elapsed times

- Link: https://github.com/sgl-project/sglang/pull/22305
- Status/date: merged / 2026-04-10
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 61 files, +61/-61, 546 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CI] Update est_time for 64 tests based on actual elapsed times"; model line: Ring 2.5; category: docs/tests/CI; main diff: `test/registered/models/test_nvidia_nemotron_3_nano.py`, `test/registered/models/test_transformers_backend_eval.py`, `test/registered/models/test_transformers_models.py`; technical summary: Covers "[CI] Update est_time for 64 tests based on actual elapsed times"; the main implementation surface is `test/registered/models/test_nvidia_nemotron_3_nano.py`, `test/registered/models/test_transformers_backend_eval.py`, `test/registered/models/test_transformers_models.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/models/test_nvidia_nemotron_3_nano.py` modified +1/-1 (2 lines); hunks: -4,7 +4,7; `test/registered/models/test_transformers_backend_eval.py` modified +1/-1 (2 lines); hunks: -7,7 +7,7; symbols: TestTransformersBackendEval, touching `TestTransformersBackendEval`; `test/registered/models/test_transformers_models.py` modified +1/-1 (2 lines); hunks: -21,7 +21,7; `test/registered/openai_server/function_call/test_anthropic_tool_use.py` modified +1/-1 (2 lines); hunks: -25,7 +25,7.
- Code diff details:
  - `test/registered/models/test_nvidia_nemotron_3_nano.py` modified +1/-1 (2 lines); hunks: -4,7 +4,7
  - `test/registered/models/test_transformers_backend_eval.py` modified +1/-1 (2 lines); hunks: -7,7 +7,7; symbols: TestTransformersBackendEval
  - `test/registered/models/test_transformers_models.py` modified +1/-1 (2 lines); hunks: -21,7 +21,7
  - `test/registered/openai_server/function_call/test_anthropic_tool_use.py` modified +1/-1 (2 lines); hunks: -25,7 +25,7
  - `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py` modified +1/-1 (2 lines); hunks: -11,7 +11,7
- Key code excerpts:

```diff
diff -- test/registered/models/test_nvidia_nemotron_3_nano.py
@@ -4,7 +4,7 @@
-register_cuda_ci(est_time=660, suite="stage-b-test-2-gpu-large")
+register_cuda_ci(est_time=540, suite="stage-b-test-2-gpu-large")
diff -- test/registered/models/test_transformers_backend_eval.py
@@ -7,7 +7,7 @@
-register_cuda_ci(est_time=180, suite="stage-b-test-1-gpu-small")
+register_cuda_ci(est_time=40, suite="stage-b-test-1-gpu-small")
diff -- test/registered/models/test_transformers_models.py
@@ -21,7 +21,7 @@
-register_cuda_ci(est_time=450, suite="stage-b-test-1-gpu-small")
+register_cuda_ci(est_time=260, suite="stage-b-test-1-gpu-small")
diff -- test/registered/openai_server/function_call/test_anthropic_tool_use.py
@@ -25,7 +25,7 @@
-register_cuda_ci(est_time=120, suite="stage-b-test-1-gpu-large")
+register_cuda_ci(est_time=50, suite="stage-b-test-1-gpu-large")
diff -- test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py
@@ -11,7 +11,7 @@
```

- Reviewed files:
  - tests: `test/registered/models/test_nvidia_nemotron_3_nano.py` modified +1/-1; `test/registered/models/test_transformers_backend_eval.py` modified +1/-1; `test/registered/models/test_transformers_models.py` modified +1/-1; `test/registered/openai_server/function_call/test_anthropic_tool_use.py` modified +1/-1; `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py` modified +1/-1; `test/registered/4-gpu-models/test_qwen35_hicache.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`, `test/registered/4-gpu-models/test_qwen35_hicache.py`, `test/registered/4-gpu-models/test_qwen3_30b.py`, `test/registered/4-gpu-models/test_qwen3_next_models.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #23001 - Add new Mintlify documentation site (docs_new/)

- Link: https://github.com/sgl-project/sglang/pull/23001
- Status/date: merged / 2026-04-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 330 files, +80364/-0, 68714 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add new Mintlify documentation site (docs_new/)"; model line: Ring 2.5; category: docs/tests/CI; main diff: `docs_new/docs/advanced_features/tool_parser.mdx`, `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx`, `docs_new/docs/advanced_features/separate_reasoning.mdx`; technical summary: Covers "Add new Mintlify documentation site (docs_new/)"; the main implementation surface is `docs_new/docs/advanced_features/tool_parser.mdx`, `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx`, `docs_new/docs/advanced_features/separate_reasoning.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "[Docs] Sync docs_new with legacy docs and update migration redirects"; model line: Ring 2.5; category: docs/tests/CI; main diff: `docs_new/docs/supported-models/multimodal_language_models.mdx`, `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx`; technical summary: Covers "[Docs] Sync docs_new with legacy docs and update migration redirects"; the main implementation surface is `docs_new/docs/supported-models/multimodal_language_models.mdx`, `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #23732 - Apply should_use_dp_reduce_scatterv guard to remaining MoE models (follow-up to #23731)

- Link: https://github.com/sgl-project/sglang/pull/23732
- Status/date: merged / 2026-04-26
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 13 files, +59/-12, 290 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Apply should_use_dp_reduce_scatterv guard to remaining MoE models (follow-up to #23731)"; model line: Ring 2.5; category: model implementation change; main diff: `python/sglang/srt/models/llada2.py`, `python/sglang/srt/models/hunyuan_v3.py`, `python/sglang/srt/models/bailing_moe_linear.py`; technical summary: Covers "Apply should_use_dp_reduce_scatterv guard to remaining MoE models (follow-up to #23731)"; the main implementation surface is `python/sglang/srt/models/llada2.py`, `python/sglang/srt/models/hunyuan_v3.py`, `python/sglang/srt/models/bailing_moe_linear.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/llada2.py` modified +10/-2 (12 lines); hunks: -55,7 +55,11; -379,7 +383,11 @@ def forward_normal(; symbols: forward_normal, touching `forward_normal`; `python/sglang/srt/models/hunyuan_v3.py` modified +7/-4 (11 lines); hunks: -34,6 +34,7; -191,10 +192,11 @@ def _forward_single_stream(self, hidden_states: torch.Tens...; symbols: _forward_single_stream, _forward_dual_stream, touching `_forward_single_stream, _forward_dual_stream`; `python/sglang/srt/models/bailing_moe_linear.py` modified +7/-1 (8 lines); hunks: -34,6 +34,7; -347,7 +348,12 @@ def forward(; symbols: forward, touching `forward`; `python/sglang/srt/models/exaone_moe.py` modified +6/-2 (8 lines); hunks: -47,7 +47,7; -300,7 +300,11 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/llada2.py` modified +10/-2 (12 lines); hunks: -55,7 +55,11; -379,7 +383,11 @@ def forward_normal(; symbols: forward_normal
  - `python/sglang/srt/models/hunyuan_v3.py` modified +7/-4 (11 lines); hunks: -34,6 +34,7; -191,10 +192,11 @@ def _forward_single_stream(self, hidden_states: torch.Tens...; symbols: _forward_single_stream, _forward_dual_stream
  - `python/sglang/srt/models/bailing_moe_linear.py` modified +7/-1 (8 lines); hunks: -34,6 +34,7; -347,7 +348,12 @@ def forward(; symbols: forward
  - `python/sglang/srt/models/exaone_moe.py` modified +6/-2 (8 lines); hunks: -47,7 +47,7; -300,7 +300,11 @@ def forward(; symbols: forward
  - `python/sglang/srt/models/llama4.py` modified +6/-1 (7 lines); hunks: -39,6 +39,7; -145,7 +146,11 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/llada2.py
@@ -55,7 +55,11 @@
-from sglang.srt.layers.moe import get_deepep_mode, get_moe_a2a_backend
+from sglang.srt.layers.moe import (
+    get_deepep_mode,
+    get_moe_a2a_backend,
+    should_use_dp_reduce_scatterv,
+)
diff -- python/sglang/srt/models/hunyuan_v3.py
@@ -34,6 +34,7 @@
+from sglang.srt.layers.moe import should_use_dp_reduce_scatterv
@@ -191,10 +192,11 @@ def _forward_single_stream(self, hidden_states: torch.Tensor) -> torch.Tensor:
-        if self.ep_size > 1:
+        skip_post_reduce = should_use_dp_reduce_scatterv()
+        if self.ep_size > 1 and not skip_post_reduce:
-        if self.tp_size > 1:
diff -- python/sglang/srt/models/bailing_moe_linear.py
@@ -34,6 +34,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/llada2.py` modified +10/-2; `python/sglang/srt/models/hunyuan_v3.py` modified +7/-4; `python/sglang/srt/models/bailing_moe_linear.py` modified +7/-1; `python/sglang/srt/models/exaone_moe.py` modified +6/-2; `python/sglang/srt/models/llama4.py` modified +6/-1; `python/sglang/srt/models/sarvam_moe.py` modified +6/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/bailing_moe_linear.py`, `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23748 - refactor(moe): centralize post-experts all-reduce skip predicate

- Link: https://github.com/sgl-project/sglang/pull/23748
- Status/date: merged / 2026-04-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 17 files, +134/-132, 532 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "refactor(moe): centralize post-experts all-reduce skip predicate"; model line: Ring 2.5; category: model implementation change; main diff: `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/models/sarvam_moe.py`, `python/sglang/srt/models/deepseek_v2.py`; technical summary: Covers "refactor(moe): centralize post-experts all-reduce skip predicate"; the main implementation surface is `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/models/sarvam_moe.py`, `python/sglang/srt/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/moe/utils.py` modified +33/-0 (33 lines); hunks: -346,6 +346,39 @@ def should_use_dp_reduce_scatterv():; symbols: should_use_dp_reduce_scatterv, should_skip_post_experts_all_reduce, speculative_moe_backend_context, touching `should_use_dp_reduce_scatterv, should_skip_post_experts_all_reduce, speculative_moe_backend_context`; `python/sglang/srt/models/sarvam_moe.py` modified +9/-16 (25 lines); hunks: -39,10 +39,7; -373,12 +370,10 @@ def forward_normal_dual_stream(; symbols: forward_normal_dual_stream, forward_normal, touching `forward_normal_dual_stream, forward_normal`; `python/sglang/srt/models/deepseek_v2.py` modified +9/-13 (22 lines); hunks: -85,7 +85,7; -651,12 +651,10 @@ def forward_normal_dual_stream(; symbols: forward_normal_dual_stream, _post_combine_hook, touching `forward_normal_dual_stream, _post_combine_hook`; `python/sglang/srt/models/glm4_moe.py` modified +9/-13 (22 lines); hunks: -61,7 +61,7; -594,12 +594,10 @@ def forward_normal_dual_stream(; symbols: forward_normal_dual_stream, forward_normal, touching `forward_normal_dual_stream, forward_normal`.
- Code diff details:
  - `python/sglang/srt/layers/moe/utils.py` modified +33/-0 (33 lines); hunks: -346,6 +346,39 @@ def should_use_dp_reduce_scatterv():; symbols: should_use_dp_reduce_scatterv, should_skip_post_experts_all_reduce, speculative_moe_backend_context
  - `python/sglang/srt/models/sarvam_moe.py` modified +9/-16 (25 lines); hunks: -39,10 +39,7; -373,12 +370,10 @@ def forward_normal_dual_stream(; symbols: forward_normal_dual_stream, forward_normal
  - `python/sglang/srt/models/deepseek_v2.py` modified +9/-13 (22 lines); hunks: -85,7 +85,7; -651,12 +651,10 @@ def forward_normal_dual_stream(; symbols: forward_normal_dual_stream, _post_combine_hook
  - `python/sglang/srt/models/glm4_moe.py` modified +9/-13 (22 lines); hunks: -61,7 +61,7; -594,12 +594,10 @@ def forward_normal_dual_stream(; symbols: forward_normal_dual_stream, forward_normal
  - `python/sglang/srt/models/qwen3_moe.py` modified +9/-13 (22 lines); hunks: -50,8 +50,7; -332,20 +331,17 @@ def forward_normal(; symbols: forward_normal
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/utils.py
@@ -346,6 +346,39 @@ def should_use_dp_reduce_scatterv():
+def should_skip_post_experts_all_reduce(
+    *,
+    is_tp_path: bool,
+    use_reduce_scatter: bool = False,
+    should_allreduce_fusion: bool = False,
+) -> bool:
diff -- python/sglang/srt/models/sarvam_moe.py
@@ -39,10 +39,7 @@
-from sglang.srt.layers.moe import (
-    should_use_dp_reduce_scatterv,
-    should_use_flashinfer_cutlass_moe_fp4_allgather,
-)
+from sglang.srt.layers.moe import should_skip_post_experts_all_reduce
@@ -373,12 +370,10 @@ def forward_normal_dual_stream(
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -85,7 +85,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/utils.py` modified +33/-0; `python/sglang/srt/models/sarvam_moe.py` modified +9/-16; `python/sglang/srt/models/deepseek_v2.py` modified +9/-13; `python/sglang/srt/models/glm4_moe.py` modified +9/-13; `python/sglang/srt/models/qwen3_moe.py` modified +9/-13; `python/sglang/srt/models/hunyuan_v3.py` modified +13/-7
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/__init__.py`, `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/models/bailing_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21126 - [4/N] Quantization Refactor: AWQ schemes and Kernel call and weight init split

- Link: https://github.com/sgl-project/sglang/pull/21126
- Status/date: merged / 2026-04-30
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 20 files, +1419/-1031, 2590 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[4/N] Quantization Refactor: AWQ schemes and Kernel call and weight init split"; model line: Ring 2.5; category: model implementation change; main diff: `python/sglang/srt/layers/quantization/awq.py`, `python/sglang/srt/layers/quantization/awq/awq.py`, `python/sglang/srt/layers/quantization/awq/schemes/awq_moe.py`; technical summary: Covers "[4/N] Quantization Refactor: AWQ schemes and Kernel call and weight init split"; the main implementation surface is `python/sglang/srt/layers/quantization/awq.py`, `python/sglang/srt/layers/quantization/awq/awq.py`, `python/sglang/srt/layers/quantization/awq/schemes/awq_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/quantization/awq.py` removed +0/-966 (966 lines); hunks: -1,966 +0,0; symbols: is_layer_skipped_awq, AWQConfig, for, __init__, touching `is_layer_skipped_awq, AWQConfig, for`; `python/sglang/srt/layers/quantization/awq/awq.py` added +484/-0 (484 lines); hunks: -0,0 +1,484; symbols: is_layer_skipped_awq, AWQConfig, for, __init__, touching `is_layer_skipped_awq, AWQConfig, for`; `python/sglang/srt/layers/quantization/awq/schemes/awq_moe.py` added +156/-0 (156 lines); hunks: -0,0 +1,156; symbols: AWQMoEScheme, __init__, _init_kernel, create_weights, touching `AWQMoEScheme, __init__, _init_kernel`; `python/sglang/srt/layers/quantization/awq/schemes/awq_linear.py` added +110/-0 (110 lines); hunks: -0,0 +1,110; symbols: AWQLinearScheme, __init__, _init_kernel, create_weights, touching `AWQLinearScheme, __init__, _init_kernel`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/awq.py` removed +0/-966 (966 lines); hunks: -1,966 +0,0; symbols: is_layer_skipped_awq, AWQConfig, for, __init__
  - `python/sglang/srt/layers/quantization/awq/awq.py` added +484/-0 (484 lines); hunks: -0,0 +1,484; symbols: is_layer_skipped_awq, AWQConfig, for, __init__
  - `python/sglang/srt/layers/quantization/awq/schemes/awq_moe.py` added +156/-0 (156 lines); hunks: -0,0 +1,156; symbols: AWQMoEScheme, __init__, _init_kernel, create_weights
  - `python/sglang/srt/layers/quantization/awq/schemes/awq_linear.py` added +110/-0 (110 lines); hunks: -0,0 +1,110; symbols: AWQLinearScheme, __init__, _init_kernel, create_weights
  - `python/sglang/srt/layers/quantization/awq/schemes/awq_marlin.py` added +105/-0 (105 lines); hunks: -0,0 +1,105; symbols: AWQMarlinLinearScheme, __init__, create_weights, process_weights_after_loading
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/awq.py
@@ -1,966 +0,0 @@
-# SPDX-License-Identifier: Apache-2.0
-from __future__ import annotations
-import logging
-import warnings
-from typing import TYPE_CHECKING, Any, Dict, List, Optional
-import torch
diff -- python/sglang/srt/layers/quantization/awq/awq.py
@@ -0,0 +1,484 @@
+# SPDX-License-Identifier: Apache-2.0
+from __future__ import annotations
+import logging
+import warnings
+from typing import TYPE_CHECKING, Any, Dict, List, Optional
+import torch
diff -- python/sglang/srt/layers/quantization/awq/schemes/awq_moe.py
@@ -0,0 +1,156 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/awq.py` removed +0/-966; `python/sglang/srt/layers/quantization/awq/awq.py` added +484/-0; `python/sglang/srt/layers/quantization/awq/schemes/awq_moe.py` added +156/-0; `python/sglang/srt/layers/quantization/awq/schemes/awq_linear.py` added +110/-0; `python/sglang/srt/layers/quantization/awq/schemes/awq_marlin.py` added +105/-0; `python/sglang/srt/layers/quantization/awq/schemes/awq_cpu.py` renamed +35/-51
- Risk and verification: The diff ships test coverage in `test/registered/quant/test_awq_dequant.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #24333 - nextn subclass owns post_load_weights is_nextn

- Link: https://github.com/sgl-project/sglang/pull/24333
- Status/date: merged / 2026-05-04
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +30/-25, 138 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "nextn subclass owns post_load_weights is_nextn"; model line: Ring 2.5; category: model implementation change; main diff: `python/sglang/srt/models/bailing_moe_nextn.py`, `python/sglang/srt/models/deepseek_nextn.py`, `python/sglang/srt/model_loader/loader.py`; technical summary: Covers "nextn subclass owns post_load_weights is_nextn"; the main implementation surface is `python/sglang/srt/models/bailing_moe_nextn.py`, `python/sglang/srt/models/deepseek_nextn.py`, `python/sglang/srt/model_loader/loader.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/bailing_moe_nextn.py` modified +9/-3 (12 lines); hunks: -217,7 +217,8 @@ def __init__(; -243,8 +244,13 @@ def set_embed_and_head(self, embed, head):; symbols: __init__, forward, set_embed_and_head, load_weights, touching `__init__, forward, set_embed_and_head`; `python/sglang/srt/models/deepseek_nextn.py` modified +6/-0 (6 lines); hunks: -313,5 +313,11 @@ def forward(; symbols: forward, load_weights, post_load_weights, touching `forward, load_weights, post_load_weights`; `python/sglang/srt/model_loader/loader.py` modified +15/-10 (25 lines); hunks: -78,7 +78,6; -286,6 +285,15 @@ def _initialize_model(; symbols: _initialize_model, _post_load_weights, BaseModelLoader, for, touching `_initialize_model, _post_load_weights, BaseModelLoader`; `python/sglang/srt/model_loader/utils.py` modified +0/-12 (12 lines); hunks: -247,18 +247,6 @@ def get_architecture_class_name(model_config: ModelConfig)...; symbols: get_architecture_class_name, post_load_weights, should_deepgemm_weight_requant_ue8m0, touching `get_architecture_class_name, post_load_weights, should_deepgemm_weight_requant_ue8m0`.
- Code diff details:
  - `python/sglang/srt/models/bailing_moe_nextn.py` modified +9/-3 (12 lines); hunks: -217,7 +217,8 @@ def __init__(; -243,8 +244,13 @@ def set_embed_and_head(self, embed, head):; symbols: __init__, forward, set_embed_and_head, load_weights
  - `python/sglang/srt/models/deepseek_nextn.py` modified +6/-0 (6 lines); hunks: -313,5 +313,11 @@ def forward(; symbols: forward, load_weights, post_load_weights
  - `python/sglang/srt/model_loader/loader.py` modified +15/-10 (25 lines); hunks: -78,7 +78,6; -286,6 +285,15 @@ def _initialize_model(; symbols: _initialize_model, _post_load_weights, BaseModelLoader, for
  - `python/sglang/srt/model_loader/utils.py` modified +0/-12 (12 lines); hunks: -247,18 +247,6 @@ def get_architecture_class_name(model_config: ModelConfig)...; symbols: get_architecture_class_name, post_load_weights, should_deepgemm_weight_requant_ue8m0
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/bailing_moe_nextn.py
@@ -217,7 +217,8 @@ def __init__(
-            self.post_load_weights_func = BailingMoEForCausalLM.post_load_weights
+            # V1 BailingMoeAttention is standard QKV (no kv_b_proj), no fixup needed.
+            self.post_load_weights_func = None
@@ -243,8 +244,13 @@ def set_embed_and_head(self, embed, head):
-    def post_load_weights(self, is_nextn=False, weight_names=None):
-        self.post_load_weights_func(self, is_nextn=is_nextn, weight_names=weight_names)
diff -- python/sglang/srt/models/deepseek_nextn.py
@@ -313,5 +313,11 @@ def forward(
+    def post_load_weights(self, is_nextn=True, weight_names=None):
+        # `is_nextn` is pinned to True for the NextN subclass; the parameter is kept
+        # only because the mixin's `do_load_weights` calls `self.post_load_weights`
+        # with `is_nextn=...` as a kwarg.
+        super().post_load_weights(is_nextn=True, weight_names=weight_names)
diff -- python/sglang/srt/model_loader/loader.py
@@ -78,7 +78,6 @@
-    post_load_weights,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/bailing_moe_nextn.py` modified +9/-3; `python/sglang/srt/models/deepseek_nextn.py` modified +6/-0; `python/sglang/srt/model_loader/loader.py` modified +15/-10; `python/sglang/srt/model_loader/utils.py` modified +0/-12
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/model_loader/loader.py`, `python/sglang/srt/model_loader/utils.py`, `python/sglang/srt/models/bailing_moe_nextn.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23837 - Add Ling_2_6

- Link: https://github.com/sgl-project/sglang/pull/23837
- Status/date: merged / 2026-05-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 12 files, +813/-68, 1107 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add Ling_2_6"; model line: Ring 2.5; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20-3e_down.json`, `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20_down.json`, `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20-3e.json`; technical summary: Covers "Add Ling_2_6"; the main implementation surface is `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20-3e_down.json`, `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20_down.json`, `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20-3e.json`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20-3e_down.json` added +164/-0 (164 lines); hunks: -0,0 +1,164; `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20_down.json` added +164/-0 (164 lines); hunks: -0,0 +1,164; `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20-3e.json` added +146/-0 (146 lines); hunks: -0,0 +1,146; `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20.json` added +146/-0 (146 lines); hunks: -0,0 +1,146.
- Code diff details:
  - `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20-3e_down.json` added +164/-0 (164 lines); hunks: -0,0 +1,164
  - `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20_down.json` added +164/-0 (164 lines); hunks: -0,0 +1,164
  - `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20-3e.json` added +146/-0 (146 lines); hunks: -0,0 +1,146
  - `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20.json` added +146/-0 (146 lines); hunks: -0,0 +1,146
  - `python/sglang/srt/models/bailing_moe_linear.py` modified +78/-34 (112 lines); hunks: -57,6 +57,7; -243,9 +244,11 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20-3e_down.json
@@ -0,0 +1,164 @@
+{
+    "1": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 32,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
diff -- python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20_down.json
@@ -0,0 +1,164 @@
+{
+    "1": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 32,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
diff -- python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20-3e.json
@@ -0,0 +1,146 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20-3e_down.json` added +164/-0; `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20_down.json` added +164/-0; `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20-3e.json` added +146/-0; `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20.json` added +146/-0; `python/sglang/srt/models/bailing_moe_linear.py` modified +78/-34; `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` modified +80/-24
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/layers/attention/linear/lightning_backend.py`, `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20-3e.json`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #26474 - [HotFix][Ling 2.6] Fix HybridLinearAttn dispatcher for Ling-2.6

- Link: https://github.com/sgl-project/sglang/pull/26474
- Status/date: merged / 2026-05-29
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +70/-1, 88 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[HotFix][Ling 2.6] Fix HybridLinearAttn dispatcher for Ling-2.6"; model line: Ring 2.5; category: bug fix; main diff: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/models/bailing_moe_linear.py`, `test/registered/8-gpu-models/test_ling_2_6_flash.py`; technical summary: Covers "[HotFix][Ling 2.6] Fix HybridLinearAttn dispatcher for Ling-2.6"; the main implementation surface is `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/models/bailing_moe_linear.py`, `test/registered/8-gpu-models/test_ling_2_6_flash.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +10/-1 (11 lines); hunks: -811,9 +811,18 @@ def __init__(; symbols: __init__, _is_full_attn, touching `__init__, _is_full_attn`; `python/sglang/srt/models/bailing_moe_linear.py` modified +6/-0 (6 lines); hunks: -508,6 +508,12 @@ def __init__(; symbols: __init__, touching `__init__`; `test/registered/8-gpu-models/test_ling_2_6_flash.py` added +54/-0 (54 lines); hunks: -0,0 +1,54; symbols: TestLing26Flash, touching `TestLing26Flash`.
- Code diff details:
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +10/-1 (11 lines); hunks: -811,9 +811,18 @@ def __init__(; symbols: __init__, _is_full_attn
  - `python/sglang/srt/models/bailing_moe_linear.py` modified +6/-0 (6 lines); hunks: -508,6 +508,12 @@ def __init__(; symbols: __init__
  - `test/registered/8-gpu-models/test_ling_2_6_flash.py` added +54/-0 (54 lines); hunks: -0,0 +1,54; symbols: TestLing26Flash
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py
@@ -811,9 +811,18 @@ def __init__(
-        # Dispatch by the layer's runtime type
+        # Explicit linear-attention subclass → strong linear signal (KDA, GDN,
+        # Qwen3-Next, Qwen3.5 main linear layers).
+        # Some hybrid models (Ling-2.5/2.6) wrap their linear layers in plain
+        # `RadixAttention` rather than `RadixLinearAttention`. Those wrappers
+        # set `_is_linear_attention=True` on the attn module so we can
diff -- python/sglang/srt/models/bailing_moe_linear.py
@@ -508,6 +508,12 @@ def __init__(
+        # Marker for HybridLinearAttnBackend._is_full_attn: Bailing wraps
+        # linear-attention layers in a plain RadixAttention, so the
+        # dispatcher can't tell from the type alone that this is a linear
+        # layer (would otherwise default to the full-attn backend, e.g. the
+        # same way MTP/NEXTN draft layers are routed).
+        self.attn._is_linear_attention = True
diff -- test/registered/8-gpu-models/test_ling_2_6_flash.py
@@ -0,0 +1,54 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +10/-1; `python/sglang/srt/models/bailing_moe_linear.py` modified +6/-0
  - tests: `test/registered/8-gpu-models/test_ling_2_6_flash.py` added +54/-0
- Risk and verification: The diff ships test coverage in `test/registered/8-gpu-models/test_ling_2_6_flash.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #26623 - Fix hybrid linear attention misrouting plain-RadixAttention linear layers to the full backend (Ring-2.5-1T)

- Link: https://github.com/sgl-project/sglang/pull/26623
- Status/date: merged / 2026-06-02
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +10/-21, 62 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix hybrid linear attention misrouting plain-RadixAttention linear layers to the full backend (Ring-2.5-1T)"; model line: Ring 2.5; category: bug fix; main diff: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/models/bailing_moe_linear.py`, `test/registered/8-gpu-models/test_ling_2_6_flash.py`; technical summary: Covers "Fix hybrid linear attention misrouting plain-RadixAttention linear layers to the full backend (Ring-2.5-1T)"; the main implementation surface is `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/models/bailing_moe_linear.py`, `test/registered/8-gpu-models/test_ling_2_6_flash.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +8/-13 (21 lines); hunks: -825,22 +825,17 @@ def __init__(; symbols: __init__, _is_full_attn, touching `__init__, _is_full_attn`; `python/sglang/srt/models/bailing_moe_linear.py` modified +0/-6 (6 lines); hunks: -508,12 +508,6 @@ def __init__(; symbols: __init__, touching `__init__`; `test/registered/8-gpu-models/test_ling_2_6_flash.py` modified +2/-2 (4 lines); hunks: -3,7 +3,7; -12,7 +12,7; symbols: TestLing26Flash, touching `TestLing26Flash`.
- Code diff details:
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +8/-13 (21 lines); hunks: -825,22 +825,17 @@ def __init__(; symbols: __init__, _is_full_attn
  - `python/sglang/srt/models/bailing_moe_linear.py` modified +0/-6 (6 lines); hunks: -508,12 +508,6 @@ def __init__(; symbols: __init__
  - `test/registered/8-gpu-models/test_ling_2_6_flash.py` modified +2/-2 (4 lines); hunks: -3,7 +3,7; -12,7 +12,7; symbols: TestLing26Flash
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py
@@ -825,22 +825,17 @@ def __init__(
-        self, layer: Optional[RadixAttention], layer_id: Optional[int] = None
+        self,
+        layer: Optional[Union[RadixAttention, RadixLinearAttention]],
+        layer_id: Optional[int] = None,
-        # Explicit linear-attention subclass → strong linear signal (KDA, GDN,
-        # Qwen3-Next, Qwen3.5 main linear layers).
diff -- python/sglang/srt/models/bailing_moe_linear.py
@@ -508,12 +508,6 @@ def __init__(
-        # Marker for HybridLinearAttnBackend._is_full_attn: Bailing wraps
-        # linear-attention layers in a plain RadixAttention, so the
-        # dispatcher can't tell from the type alone that this is a linear
-        # layer (would otherwise default to the full-attn backend, e.g. the
-        # same way MTP/NEXTN draft layers are routed).
-        self.attn._is_linear_attention = True
diff -- test/registered/8-gpu-models/test_ling_2_6_flash.py
@@ -3,7 +3,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +8/-13; `python/sglang/srt/models/bailing_moe_linear.py` modified +0/-6
  - tests: `test/registered/8-gpu-models/test_ling_2_6_flash.py` modified +2/-2
- Risk and verification: The diff ships test coverage in `test/registered/8-gpu-models/test_ling_2_6_flash.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #27116 - Revert "Fix hybrid linear attention misrouting plain-RadixAttention linear layers to the full backend (Ring-2.5-1T)"

- Link: https://github.com/sgl-project/sglang/pull/27116
- Status/date: merged / 2026-06-03
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +19/-8, 44 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Revert "Fix hybrid linear attention misrouting plain-RadixAttention linear layers to the full backend (Ring-2.5-1T)""; model line: Ring 2.5; category: bug fix; main diff: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/models/bailing_moe_linear.py`; technical summary: Covers "Revert "Fix hybrid linear attention misrouting plain-RadixAttention linear layers to the full backend (Ring-2.5-1T)""; the main implementation surface is `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/models/bailing_moe_linear.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +13/-8 (21 lines); hunks: -782,17 +782,22 @@ def __init__(; symbols: __init__, _is_full_attn, touching `__init__, _is_full_attn`; `python/sglang/srt/models/bailing_moe_linear.py` modified +6/-0 (6 lines); hunks: -508,6 +508,12 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +13/-8 (21 lines); hunks: -782,17 +782,22 @@ def __init__(; symbols: __init__, _is_full_attn
  - `python/sglang/srt/models/bailing_moe_linear.py` modified +6/-0 (6 lines); hunks: -508,6 +508,12 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py
@@ -782,17 +782,22 @@ def __init__(
-        self,
-        layer: Optional[Union[RadixAttention, RadixLinearAttention]],
-        layer_id: Optional[int] = None,
+        self, layer: Optional[RadixAttention], layer_id: Optional[int] = None
-        # RadixLinearAttention is unambiguously a linear-attention layer.
-        # Everything else (including plain RadixAttention) must be classified by
diff -- python/sglang/srt/models/bailing_moe_linear.py
@@ -508,6 +508,12 @@ def __init__(
+        # Marker for HybridLinearAttnBackend._is_full_attn: Bailing wraps
+        # linear-attention layers in a plain RadixAttention, so the
+        # dispatcher can't tell from the type alone that this is a linear
+        # layer (would otherwise default to the full-attn backend, e.g. the
+        # same way MTP/NEXTN draft layers are routed).
+        self.attn._is_linear_attention = True
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +13/-8; `python/sglang/srt/models/bailing_moe_linear.py` modified +6/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/models/bailing_moe_linear.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27120 - Fix hybrid linear attention dispatch by layer id with draft-worker awareness

- Link: https://github.com/sgl-project/sglang/pull/27120
- Status/date: merged / 2026-06-03
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +5/-23, 56 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix hybrid linear attention dispatch by layer id with draft-worker awareness"; model line: Ring 2.5; category: bug fix; main diff: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/layers/attention/attention_registry.py`, `python/sglang/srt/models/bailing_moe_linear.py`; technical summary: Covers "Fix hybrid linear attention dispatch by layer id with draft-worker awareness"; the main implementation surface is `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/layers/attention/attention_registry.py`, `python/sglang/srt/models/bailing_moe_linear.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +0/-16 (16 lines); hunks: -16,7 +16,6; -784,21 +783,6 @@ def __init__(; symbols: __init__, _is_full_attn, touching `__init__, _is_full_attn`; `python/sglang/srt/layers/attention/attention_registry.py` modified +5/-1 (6 lines); hunks: -314,7 +314,11 @@ def attn_backend_wrapper(runner: "ModelRunner", full_attn_b...; symbols: attn_backend_wrapper, touching `attn_backend_wrapper`; `python/sglang/srt/models/bailing_moe_linear.py` modified +0/-6 (6 lines); hunks: -508,12 +508,6 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +0/-16 (16 lines); hunks: -16,7 +16,6; -784,21 +783,6 @@ def __init__(; symbols: __init__, _is_full_attn
  - `python/sglang/srt/layers/attention/attention_registry.py` modified +5/-1 (6 lines); hunks: -314,7 +314,11 @@ def attn_backend_wrapper(runner: "ModelRunner", full_attn_b...; symbols: attn_backend_wrapper
  - `python/sglang/srt/models/bailing_moe_linear.py` modified +0/-6 (6 lines); hunks: -508,12 +508,6 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py
@@ -16,7 +16,6 @@
-from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
@@ -784,21 +783,6 @@ def __init__(
-        # Explicit linear-attention subclass → strong linear signal (KDA, GDN,
-        # Qwen3-Next, Qwen3.5 main linear layers).
-        if isinstance(layer, RadixLinearAttention):
-            return False
diff -- python/sglang/srt/layers/attention/attention_registry.py
@@ -314,7 +314,11 @@ def attn_backend_wrapper(runner: "ModelRunner", full_attn_backend: "AttentionBac
-        full_attn_layers = cfg.full_attention_layer_ids
+        if runner.is_draft_worker:
+            # FIXME: we assume that MTP/NEXTN always use full-attention.
+            full_attn_layers = [0]
+        else:
+            full_attn_layers = cfg.full_attention_layer_ids
diff -- python/sglang/srt/models/bailing_moe_linear.py
@@ -508,12 +508,6 @@ def __init__(
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +0/-16; `python/sglang/srt/layers/attention/attention_registry.py` modified +5/-1; `python/sglang/srt/models/bailing_moe_linear.py` modified +0/-6
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/attention_registry.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/models/bailing_moe_linear.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23906 - [Refactor] Cuda Graph Runner/Backend Refactor

- Link: https://github.com/sgl-project/sglang/pull/23906
- Status/date: merged / 2026-06-10
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 160 files, +5197/-3068, 12233 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Refactor] Cuda Graph Runner/Backend Refactor"; model line: Ring 2.5; category: performance/backend optimization; main diff: `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py`; technical summary: Covers "[Refactor] Cuda Graph Runner/Backend Refactor"; the main implementation surface is `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #28567 - Add get_parallel(): a structured accessor for parallel-topology state

- Link: https://github.com/sgl-project/sglang/pull/28567
- Status/date: merged / 2026-06-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 184 files, +1865/-1727, 8932 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add get_parallel(): a structured accessor for parallel-topology state"; model line: Ring 2.5; category: model support/runtime entry; main diff: `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py`; technical summary: Covers "Add get_parallel(): a structured accessor for parallel-topology state"; the main implementation surface is `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Motivation: Title: "[docs] Add B300 cookbook deployment options"; model line: Ring 2.5; category: performance/backend optimization; main diff: `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx`, `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx`; technical summary: Covers "[docs] Add B300 cookbook deployment options"; the main implementation surface is `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx`, `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
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
