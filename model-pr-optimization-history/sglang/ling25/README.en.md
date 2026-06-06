# sglang Ling 2.5 1T Model PR Optimization History

## 2026-06-05 PR Backfill Audit

Rechecked sglang upstream `origin/main@6cfdc1858` on 2026-06-05; 5 additional PR-numbered merge(s) touched the tracked implementation files after the previous freshness cutoff (2026-05-19). These are not yet reflected in the timeline / diff-audit cards below and should be folded in on the next full regeneration.

| Merged | PR | Title | Tracked files touched |
| --- | --- | --- | --- |
| 2026-06-03 | [#27120](https://github.com/sgl-project/sglang/pull/27120) | Fix hybrid linear attention dispatch by layer id with draft-worker awareness | `bailing_moe_linear.py` |
| 2026-06-02 | [#27116](https://github.com/sgl-project/sglang/pull/27116) | Revert "Fix hybrid linear attention misrouting plain-RadixAttention linear layers to the full backend (Ring-2.5-1T)" | `bailing_moe_linear.py` |
| 2026-06-02 | [#26623](https://github.com/sgl-project/sglang/pull/26623) | Fix hybrid linear attention misrouting plain-RadixAttention linear layers to the full backend (Ring-2.5-1T) | `bailing_moe_linear.py` |
| 2026-05-29 | [#26474](https://github.com/sgl-project/sglang/pull/26474) | [HotFix][Ling 2.6] Fix HybridLinearAttn dispatcher for Ling-2.6 | `bailing_moe_linear.py` |
| 2026-05-27 | [#23837](https://github.com/sgl-project/sglang/pull/23837) | Add Ling_2_6 | `bailing_moe_linear.py` |


## 2026-05-19 Coverage Addition

Generated from sglang upstream `origin/main@5073c82a37`, `git log --name-only -- <model-files>` over model-related paths, and the GitHub Pull Request files API. This page fills the missing `Ling 2.5 1T` history entry found from sgl-cookbook coverage.

## Implementation File Coverage

| File | PRs traced by git |
| --- | --- |
| `skills/llm-serving-auto-benchmark/configs/cookbook-llm/ling-2.5-1t.yaml` | [#24250](https://github.com/sgl-project/sglang/pull/24250) |
| `docs_new/cookbook/autoregressive/InclusionAI/Ling-2.5-1T.mdx` | [#23337](https://github.com/sgl-project/sglang/pull/23337), [#23001](https://github.com/sgl-project/sglang/pull/23001) |
| `docs_new/src/snippets/autoregressive/ling-25-1t-deployment.jsx` | [#24977](https://github.com/sgl-project/sglang/pull/24977), [#23001](https://github.com/sgl-project/sglang/pull/23001) |
| `python/sglang/srt/configs/bailing_hybrid.py` | [#18598](https://github.com/sgl-project/sglang/pull/18598) |
| `python/sglang/srt/models/bailing_moe.py` | [#23748](https://github.com/sgl-project/sglang/pull/23748), [#23732](https://github.com/sgl-project/sglang/pull/23732), [#20316](https://github.com/sgl-project/sglang/pull/20316), [#17784](https://github.com/sgl-project/sglang/pull/17784), [#18860](https://github.com/sgl-project/sglang/pull/18860), [#15119](https://github.com/sgl-project/sglang/pull/15119), [#17570](https://github.com/sgl-project/sglang/pull/17570), [#13715](https://github.com/sgl-project/sglang/pull/13715), [#15835](https://github.com/sgl-project/sglang/pull/15835), [#15526](https://github.com/sgl-project/sglang/pull/15526), [#13730](https://github.com/sgl-project/sglang/pull/13730), [#14337](https://github.com/sgl-project/sglang/pull/14337), ... (25 total) |
| `python/sglang/srt/models/bailing_moe_linear.py` | [#21126](https://github.com/sgl-project/sglang/pull/21126), [#23748](https://github.com/sgl-project/sglang/pull/23748), [#23732](https://github.com/sgl-project/sglang/pull/23732), [#9744](https://github.com/sgl-project/sglang/pull/9744), [#18793](https://github.com/sgl-project/sglang/pull/18793), [#18598](https://github.com/sgl-project/sglang/pull/18598) |
| `python/sglang/srt/models/bailing_moe_nextn.py` | [#24333](https://github.com/sgl-project/sglang/pull/24333), [#18860](https://github.com/sgl-project/sglang/pull/18860), [#18598](https://github.com/sgl-project/sglang/pull/18598), [#17808](https://github.com/sgl-project/sglang/pull/17808), [#17570](https://github.com/sgl-project/sglang/pull/17570), [#11528](https://github.com/sgl-project/sglang/pull/11528), [#11520](https://github.com/sgl-project/sglang/pull/11520), [#11331](https://github.com/sgl-project/sglang/pull/11331), [#10359](https://github.com/sgl-project/sglang/pull/10359) |
| `test/registered/ascend/llm_models/test_npu_ling_lite.py` | [#20751](https://github.com/sgl-project/sglang/pull/20751), [#19382](https://github.com/sgl-project/sglang/pull/19382) |

## PR Coverage Summary

- git-traced PR count: 37
- keyword/supplemental PR count: 0
- total PR count in this document: 37
- file trace command: `git log --name-only -- <model-files>`
- diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | Status | Title | Main files |
| --- | --- | --- | --- | --- |
| 2025-08-06 | [#8680](https://github.com/sgl-project/sglang/pull/8680) | merged | Support bailing moe | `docs/supported_models/generative_models.md`, `python/sglang/srt/models/bailing_moe.py`, `test/srt/models/test_generation_models.py` |
| 2025-09-12 | [#10359](https://github.com/sgl-project/sglang/pull/10359) | merged | Support LingV2 model | `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/layers/linear.py` |
| 2025-09-12 | [#10362](https://github.com/sgl-project/sglang/pull/10362) | merged | Fix Bailing MoE model bugs | `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/server_args.py` |
| 2025-09-15 | [#9338](https://github.com/sgl-project/sglang/pull/9338) | merged | Refactor TopK to ensure readability and extensibility | `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/layers/moe/topk.py` |
| 2025-09-24 | [#10860](https://github.com/sgl-project/sglang/pull/10860) | merged | fix bailing_moe with enable_dp_attention | `python/sglang/srt/models/bailing_moe.py` |
| 2025-09-26 | [#10749](https://github.com/sgl-project/sglang/pull/10749) | merged | Fuse write kv buffer into rope for qwen3 moe & bailing moe | `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/gpt_oss.py`, `python/sglang/srt/models/qwen3_moe.py` |
| 2025-10-12 | [#11331](https://github.com/sgl-project/sglang/pull/11331) | merged | Deprecate `global_server_args_dict` | `python/sglang/global_config.py`, `python/sglang/srt/distributed/device_communicators/pynccl_allocator.py`, `python/sglang/srt/eplb/expert_location_dispatch.py` |
| 2025-10-12 | [#11465](https://github.com/sgl-project/sglang/pull/11465) | merged | bailingMoE: Fix Key error of deepep_mode | `python/sglang/srt/models/bailing_moe.py` |
| 2025-10-13 | [#11520](https://github.com/sgl-project/sglang/pull/11520) | merged | Revert "Deprecate `global_server_args_dict`" | `python/sglang/global_config.py`, `python/sglang/srt/distributed/device_communicators/pynccl_allocator.py`, `python/sglang/srt/eplb/expert_location_dispatch.py` |
| 2025-10-13 | [#11528](https://github.com/sgl-project/sglang/pull/11528) | merged | Depreate `global_server_args_dict` | `python/sglang/global_config.py`, `python/sglang/srt/distributed/device_communicators/pynccl_allocator.py`, `python/sglang/srt/eplb/expert_location_dispatch.py` |
| 2025-10-17 | [#11685](https://github.com/sgl-project/sglang/pull/11685) | merged | [Lint] Add `python/sglang` to ruff F401 checks and remove unused imports in files | `.pre-commit-config.yaml`, `python/sglang/srt/_custom_ops.py`, `python/sglang/srt/compilation/cuda_piecewise_backend.py` |
| 2025-10-20 | [#11847](https://github.com/sgl-project/sglang/pull/11847) | merged | [9/N] MoE Refactor: cleanup dispatcher interfaces | `python/sglang/srt/layers/dp_attention.py`, `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py` |
| 2025-10-31 | [#12369](https://github.com/sgl-project/sglang/pull/12369) | merged | Enable bailing_moe to support TP=16 | `python/sglang/srt/models/bailing_moe.py` |
| 2025-12-07 | [#14337](https://github.com/sgl-project/sglang/pull/14337) | merged | remove unecessary dual stream token threshold from the rest of models (qwen moe, kimi linear, etc.) | `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/kimi_linear.py`, `python/sglang/srt/models/llada2.py` |
| 2025-12-12 | [#13730](https://github.com/sgl-project/sglang/pull/13730) | merged | [bugfix] fix TBO crashes when attn_tp_size > 1 | `python/sglang/srt/batch_overlap/operations.py`, `python/sglang/srt/batch_overlap/two_batch_overlap.py`, `python/sglang/srt/layers/communicator.py` |
| 2025-12-21 | [#15526](https://github.com/sgl-project/sglang/pull/15526) | merged | Optimize Bailing-MoE with FlashInfer Fused All-Reduce | `python/sglang/srt/models/bailing_moe.py` |
| 2025-12-28 | [#15835](https://github.com/sgl-project/sglang/pull/15835) | merged | [Feature] JIT Fused QK norm + qk norm clean up | `python/sglang/jit_kernel/benchmark/bench_qknorm.py`, `python/sglang/jit_kernel/csrc/norm.cuh`, `python/sglang/jit_kernel/include/sgl_kernel/runtime.cuh` |
| 2026-01-10 | [#13715](https://github.com/sgl-project/sglang/pull/13715) | merged | Fix EPLB + FP4 Quantization Compatibility Issue | `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2026-01-24 | [#17570](https://github.com/sgl-project/sglang/pull/17570) | merged | Use attn tp group in embedding for more models | `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/bailing_moe_nextn.py`, `python/sglang/srt/models/falcon_h1.py` |
| 2026-01-30 | [#17808](https://github.com/sgl-project/sglang/pull/17808) | merged | Fix the scenario where eh_proj is quantized in the bailing moe nextn weights | `python/sglang/srt/models/bailing_moe_nextn.py` |
| 2026-02-01 | [#15119](https://github.com/sgl-project/sglang/pull/15119) | merged | feat: Add Ling Flash v2.0 support for Eagle3 | `python/sglang/srt/models/bailing_moe.py` |
| 2026-02-13 | [#18598](https://github.com/sgl-project/sglang/pull/18598) | merged | Support LingV2_5 model | `python/sglang/srt/configs/__init__.py`, `python/sglang/srt/configs/bailing_hybrid.py`, `python/sglang/srt/configs/model_config.py` |
| 2026-02-13 | [#18793](https://github.com/sgl-project/sglang/pull/18793) | merged | Cleanup debug log for Ring model | `python/sglang/srt/configs/mamba_utils.py`, `python/sglang/srt/models/bailing_moe_linear.py` |
| 2026-02-15 | [#18860](https://github.com/sgl-project/sglang/pull/18860) | merged | update pre-commit config | `.github/workflows/lint.yml`, `.pre-commit-config.yaml`, `3rdparty/amd/tuning/benchmark_moe_rocm.py` |
| 2026-03-16 | [#19382](https://github.com/sgl-project/sglang/pull/19382) | merged | Add NPU basic function testcases | `.github/workflows/nightly-test-npu.yml`, `python/sglang/test/ascend/disaggregation_utils.py`, `python/sglang/test/ascend/test_ascend_utils.py` |
| 2026-03-18 | [#17784](https://github.com/sgl-project/sglang/pull/17784) | merged | Upgrade transformers==5.3.0 | `docs/advanced_features/vlm_query.ipynb`, `python/pyproject.toml`, `python/pyproject_cpu.toml` |
| 2026-03-19 | [#9744](https://github.com/sgl-project/sglang/pull/9744) | merged | [CPU] Add FP8 Bmm support | `python/sglang/srt/models/bailing_moe_linear.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_cpu.py` |
| 2026-03-23 | [#20316](https://github.com/sgl-project/sglang/pull/20316) | merged | fix fused_set_kv_buffer for rope with Ling-v2 | `python/sglang/srt/models/bailing_moe.py` |
| 2026-04-01 | [#20751](https://github.com/sgl-project/sglang/pull/20751) | merged | [NPU]Add a full test pipeline on NPU, resolve issues in the NPU test architecture | `.github/workflows/full-test-npu.yml`, `.github/workflows/nightly-test-npu.yml`, `.github/workflows/pr-test-npu.yml` |
| 2026-04-20 | [#23001](https://github.com/sgl-project/sglang/pull/23001) | merged | Add new Mintlify documentation site (docs_new/) | `.gitignore`, `docs_new/.github/workflows/sync-lmsys-sglang-blogs.yml`, `docs_new/.gitignore` |
| 2026-04-21 | [#23337](https://github.com/sgl-project/sglang/pull/23337) | merged | [Docs] Sync docs_new with legacy docs and update migration redirects | `.pre-commit-config.yaml`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-Math-V2.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR-2.mdx` |
| 2026-04-26 | [#23732](https://github.com/sgl-project/sglang/pull/23732) | merged | Apply should_use_dp_reduce_scatterv guard to remaining MoE models (follow-up to #23731) | `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/bailing_moe_linear.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2026-04-27 | [#23748](https://github.com/sgl-project/sglang/pull/23748) | merged | refactor(moe): centralize post-experts all-reduce skip predicate | `python/sglang/srt/layers/moe/__init__.py`, `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/models/bailing_moe.py` |
| 2026-04-30 | [#21126](https://github.com/sgl-project/sglang/pull/21126) | merged | [4/N] Quantization Refactor: AWQ schemes and Kernel call and weight init split | `python/sglang/srt/hardware_backend/gpu/quantization/awq_kernels.py`, `python/sglang/srt/hardware_backend/npu/quantization/awq_kernels.py`, `python/sglang/srt/layers/linear.py` |
| 2026-05-02 | [#24250](https://github.com/sgl-project/sglang/pull/24250) | merged | [SKILL] Upgrade sglang profile and auto_benchmark skills | `agent-skills/llm-serving-auto-benchmark/SKILL.md`, `skills/llm-serving-auto-benchmark/configs/cookbook-llm/README.md`, `skills/llm-serving-auto-benchmark/configs/cookbook-llm/deepseek-math-v2.yaml` |
| 2026-05-04 | [#24333](https://github.com/sgl-project/sglang/pull/24333) | merged | nextn subclass owns post_load_weights is_nextn | `python/sglang/srt/model_loader/loader.py`, `python/sglang/srt/model_loader/utils.py`, `python/sglang/srt/models/bailing_moe_nextn.py` |
| 2026-05-11 | [#24977](https://github.com/sgl-project/sglang/pull/24977) | merged | fix gb envs in deployment guide | `docs_new/src/snippets/autoregressive/ling-25-1t-deployment.jsx` |

## Per-PR Diff Audit Cards

### PR #8680 - Support bailing moe

- Link: https://github.com/sgl-project/sglang/pull/8680
- Status/date: merged / 2025-08-06
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 3 files, +427/-0, with 441 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support bailing moe"; model line: Ling 2.5 1T; category: model support/runtime entry; main diff: `docs/supported_models/generative_models.md`, `python/sglang/srt/models/bailing_moe.py`, `test/srt/models/test_generation_models.py`.
- Key implementation:
  - `docs/supported_models/generative_models.md` modified +1/-0
  - `python/sglang/srt/models/bailing_moe.py` added +425/-0; symbols: BailingAttention, __init__, forward, BailingMLP
  - `test/srt/models/test_generation_models.py` modified +1/-0
- Code diff details:
  - `docs/supported_models/generative_models.md` modified +1/-0
  - `python/sglang/srt/models/bailing_moe.py` added +425/-0
  - `test/srt/models/test_generation_models.py` modified +1/-0
- Key code excerpts:

```diff
diff -- docs/supported_models/generative_models.md
@@ -47,5 +47,6 @@ in the GitHub search bar.
 | **MiMo** (7B series)               | `XiaomiMiMo/MiMo-7B-RL`                         | Xiaomi's reasoning-optimized model series, leverages Multiple-Token Prediction for faster inference. |
 | **Arcee AFM-4.5B**               | `arcee-ai/AFM-4.5B-Base`                         | Arcee's foundational model series for real world reliability and edge deployments. |
 | **Persimmon** (8B)               | `adept/persimmon-8b-chat`                         | Adept’s open 8B model with a 16K context window and fast inference; trained for broad usability and licensed under Apache 2.0. |
+| **Ling** (16.8B–290B) | `inclusionAI/Ling-lite`, `inclusionAI/Ling-plus` | InclusionAI’s open MoE models. Ling-Lite has 16.8B total / 2.75B active parameters, and Ling-Plus has 290B total / 28.8B active parameters. They are designed for high performance on NLP and complex reasoning tasks. |
 | **Granite 3.0, 3.1** (IBM)               | `ibm-granite/granite-3.1-8b-instruct`                          | IBM's open dense foundation models optimized for reasoning, code, and business AI use cases. Integrated with Red Hat and watsonx systems. |
 | **Granite 3.0 MoE** (IBM)               | `ibm-granite/granite-3.0-3b-a800m-instruct`                          | IBM’s Mixture-of-Experts models offering strong performance with cost-efficiency. MoE expert routing designed for enterprise deployment at scale. |
diff -- python/sglang/srt/models/bailing_moe.py
@@ -0,0 +1,425 @@
+# Copyright 2023-2024 SGLang Team
+# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/bailing_moe.py
+
+from collections.abc import Iterable
+from typing import Optional, Tuple
+
+import torch
+import torch.nn.functional as F
+from torch import nn
+from transformers.configuration_utils import PretrainedConfig
+
+from sglang.srt.distributed import (
+    get_tensor_model_parallel_world_size,
```
- Reviewed files:
  - runtime: `python/sglang/srt/models/bailing_moe.py` added +425/-0
  - tests: `test/srt/models/test_generation_models.py` modified +1/-0
  - docs/bench: `docs/supported_models/generative_models.md` modified +1/-0
- Risk and verification: The diff includes test or benchmark paths; rerun those checks plus a minimal launch/accuracy smoke before changing this model again.

### PR #10359 - Support LingV2 model

- Link: https://github.com/sgl-project/sglang/pull/10359
- Status/date: merged / 2025-09-12
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 7 files, +1165/-221, with 1642 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support LingV2 model"; model line: Ling 2.5 1T; category: model support/runtime entry; main diff: `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/layers/linear.py`.
- Key implementation:
  - `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` modified +11/-2
  - `python/sglang/srt/configs/model_config.py` modified +5/-0
  - `python/sglang/srt/layers/linear.py` modified +32/-0; symbols: _load_qkv_block_scale
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=256,N=512,device_name=NVIDIA_H20.json` added +146/-0
- Code diff details:
  - `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` modified +11/-2
  - `python/sglang/srt/configs/model_config.py` modified +5/-0
  - `python/sglang/srt/layers/linear.py` modified +32/-0
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=256,N=512,device_name=NVIDIA_H20.json` added +146/-0
- Key code excerpts:

```diff
diff -- benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py
@@ -13,8 +13,8 @@
 from transformers import AutoConfig

 from sglang.srt.layers.moe.fused_moe_triton import override_config
-from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
-    fused_moe,
+from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_moe
+from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_config import (
     get_config_dtype_str,
     get_config_file_name,
     get_default_config,
@@ -441,6 +441,15 @@ def main(args: argparse.Namespace):
         topk = config.num_experts_per_tok
         intermediate_size = config.moe_intermediate_size
diff -- python/sglang/srt/configs/model_config.py
@@ -141,6 +141,11 @@ def __init__(

         if is_draft_model and self.hf_config.architectures[0] == "MiMoForCausalLM":
             self.hf_config.architectures[0] = "MiMoMTP"
+        if is_draft_model and self.hf_config.architectures[0] in [
+            "BailingMoeV2ForCausalLM",
+            "BailingMoeForCausalLM",
+        ]:
+            self.hf_config.architectures[0] = "BailingMoeForCausalLMNextN"
         if (
             is_draft_model
             and self.hf_config.architectures[0] == "Ernie4_5_MoeForCausalLM"
```
- Reviewed files:
  - runtime: `python/sglang/srt/configs/model_config.py` modified +5/-0; `python/sglang/srt/layers/linear.py` modified +32/-0; `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=256,N=512,device_name=NVIDIA_H20.json` added +146/-0; `python/sglang/srt/models/bailing_moe.py` modified +795/-218; `python/sglang/srt/models/bailing_moe_nextn.py` added +168/-0; `python/sglang/srt/server_args.py` modified +8/-1
  - docs/bench: `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` modified +11/-2
- Risk and verification: The diff includes test or benchmark paths; rerun those checks plus a minimal launch/accuracy smoke before changing this model again.

### PR #10362 - Fix Bailing MoE model bugs

- Link: https://github.com/sgl-project/sglang/pull/10362
- Status/date: merged / 2025-09-12
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 2 files, +8/-5, with 41 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix Bailing MoE model bugs"; model line: Ling 2.5 1T; category: bug fix; main diff: `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/server_args.py`.
- Key implementation:
  - `python/sglang/srt/models/bailing_moe.py` modified +7/-4
  - `python/sglang/srt/server_args.py` modified +1/-1
- Code diff details:
  - `python/sglang/srt/models/bailing_moe.py` modified +7/-4
  - `python/sglang/srt/server_args.py` modified +1/-1
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/bailing_moe.py
@@ -128,7 +128,9 @@ def forward(

         gate_up, _ = self.gate_up_proj(hidden_states)
         hidden_states = self.act_fn(gate_up)
-        hidden_states, _ = self.down_proj(hidden_states)
+        hidden_states, _ = self.down_proj(
+            hidden_states, skip_all_reduce=use_reduce_scatter
+        )
         return hidden_states


@@ -328,7 +330,7 @@ def forward_normal_dual_stream(
     ) -> torch.Tensor:
         current_stream = torch.cuda.current_stream()
diff -- python/sglang/srt/server_args.py
@@ -757,7 +757,7 @@ def __post_init__(self):
             if model_arch in [
                 "DeepseekV3ForCausalLM",
                 "Glm4MoeForCausalLM",
-                "BailingMoeV2ForCausalLM",
+                "BailingMoeForCausalLM",
                 "BailingMoeV2ForCausalLM",
             ]:
                 # Auto set draft_model_path DeepSeek-V3/R1
```
- Reviewed files:
  - runtime: `python/sglang/srt/models/bailing_moe.py` modified +7/-4; `python/sglang/srt/server_args.py` modified +1/-1
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.

### PR #9338 - Refactor TopK to ensure readability and extensibility

- Link: https://github.com/sgl-project/sglang/pull/9338
- Status/date: merged / 2025-09-15
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 14 files, +52/-47, with 296 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Refactor TopK to ensure readability and extensibility"; model line: Ling 2.5 1T; category: model implementation change; main diff: `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/layers/moe/topk.py`.
- Key implementation:
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +4/-4; symbols: get_moe_impl_class
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +0/-10
  - `python/sglang/srt/layers/moe/topk.py` modified +30/-9
  - `python/sglang/srt/managers/schedule_batch.py` modified +0/-1
- Code diff details:
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +4/-4
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +0/-10
  - `python/sglang/srt/layers/moe/topk.py` modified +30/-9
  - `python/sglang/srt/managers/schedule_batch.py` modified +0/-1
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/ep_moe/layer.py
@@ -888,7 +888,7 @@ def _forward_ll(dispatch_output: DeepEPLLOutput):
             raise ValueError(f"Not Supported DeepEP format {dispatch_output.format}")


-def get_moe_impl_class(quant_config: Optional[QuantizationConfig] = None):
+def get_moe_impl_class(quant_config: Optional[QuantizationConfig]):
     if get_moe_a2a_backend().is_deepep():
         return DeepEPMoE

@@ -901,8 +901,7 @@ def get_moe_impl_class(quant_config: Optional[QuantizationConfig] = None):
             return FusedMoE
         try:
             # Check the quantization argument directly
-            quantization = global_server_args_dict.get("quantization")
diff -- python/sglang/srt/layers/moe/fused_moe_triton/layer.py
@@ -74,16 +74,6 @@
 logger = logging.getLogger(__name__)


-def _is_fp4_quantization_enabled():
-    """Check if ModelOpt FP4 quantization is enabled."""
-    try:
-        # Use the same simple check that works for class selection
-        quantization = global_server_args_dict.get("quantization")
-        return quantization == "modelopt_fp4"
-    except:
-        return False
-
-
```
- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +4/-4; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +0/-10; `python/sglang/srt/layers/moe/topk.py` modified +30/-9; `python/sglang/srt/managers/schedule_batch.py` modified +0/-1; `python/sglang/srt/models/bailing_moe.py` modified +1/-1; `python/sglang/srt/models/deepseek_v2.py` modified +7/-12; `python/sglang/srt/models/ernie4.py` modified +1/-1; `python/sglang/srt/models/glm4_moe.py` modified +1/-1
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.

### PR #10860 - fix bailing_moe with enable_dp_attention

- Link: https://github.com/sgl-project/sglang/pull/10860
- Status/date: merged / 2025-09-24
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-2, with 23 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix bailing_moe with enable_dp_attention"; model line: Ling 2.5 1T; category: bug fix; main diff: `python/sglang/srt/models/bailing_moe.py`.
- Key implementation:
  - `python/sglang/srt/models/bailing_moe.py` modified +2/-2
- Code diff details:
  - `python/sglang/srt/models/bailing_moe.py` modified +2/-2
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/bailing_moe.py
@@ -45,12 +45,12 @@
     get_attention_dp_size,
     get_attention_tp_rank,
     get_attention_tp_size,
+    is_dp_attention_enabled,
 )
 from sglang.srt.layers.layernorm import RMSNorm
 from sglang.srt.layers.linear import (
     MergedColumnParallelLinear,
     QKVParallelLinear,
-    ReplicatedLinear,
     RowParallelLinear,
 )
 from sglang.srt.layers.logits_processor import LogitsProcessor
```
- Reviewed files:
  - runtime: `python/sglang/srt/models/bailing_moe.py` modified +2/-2
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.

### PR #10749 - Fuse write kv buffer into rope for qwen3 moe & bailing moe

- Link: https://github.com/sgl-project/sglang/pull/10749
- Status/date: merged / 2025-09-26
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 4 files, +105/-34, with 207 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fuse write kv buffer into rope for qwen3 moe & bailing moe"; model line: Ling 2.5 1T; category: performance/backend optimization; main diff: `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/gpt_oss.py`, `python/sglang/srt/models/qwen3_moe.py`.
- Key implementation:
  - `python/sglang/srt/models/bailing_moe.py` modified +25/-2
  - `python/sglang/srt/models/gpt_oss.py` modified +7/-30
  - `python/sglang/srt/models/qwen3_moe.py` modified +22/-2
  - `python/sglang/srt/models/utils.py` added +51/-0; symbols: enable_fused_set_kv_buffer, create_fused_set_kv_buffer_arg
- Code diff details:
  - `python/sglang/srt/models/bailing_moe.py` modified +25/-2
  - `python/sglang/srt/models/gpt_oss.py` modified +7/-30
  - `python/sglang/srt/models/qwen3_moe.py` modified +22/-2
  - `python/sglang/srt/models/utils.py` added +51/-0
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/bailing_moe.py
@@ -72,6 +72,10 @@
 from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
 from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
 from sglang.srt.model_loader.weight_utils import default_weight_loader
+from sglang.srt.models.utils import (
+    create_fused_set_kv_buffer_arg,
+    enable_fused_set_kv_buffer,
+)
 from sglang.srt.utils import add_prefix, is_cuda, is_non_idle_and_non_empty, make_layers

 LoraConfig = None
@@ -555,8 +559,27 @@ def forward(
         q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
         if self.use_qk_norm:
diff -- python/sglang/srt/models/gpt_oss.py
@@ -66,6 +66,10 @@
 from sglang.srt.managers.schedule_batch import global_server_args_dict
 from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
 from sglang.srt.model_loader.weight_utils import default_weight_loader
+from sglang.srt.models.utils import (
+    create_fused_set_kv_buffer_arg,
+    enable_fused_set_kv_buffer,
+)
 from sglang.srt.utils import (
     LazyValue,
     add_prefix,
@@ -193,33 +197,6 @@ def forward_normal(
         return ans

```
- Reviewed files:
  - runtime: `python/sglang/srt/models/bailing_moe.py` modified +25/-2; `python/sglang/srt/models/gpt_oss.py` modified +7/-30; `python/sglang/srt/models/qwen3_moe.py` modified +22/-2; `python/sglang/srt/models/utils.py` added +51/-0
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.

### PR #11331 - Deprecate `global_server_args_dict`

- Link: https://github.com/sgl-project/sglang/pull/11331
- Status/date: merged / 2025-10-12
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 54 files, +240/-321, with 1946 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Deprecate `global_server_args_dict`"; model line: Ling 2.5 1T; category: model implementation change; main diff: `python/sglang/global_config.py`, `python/sglang/srt/distributed/device_communicators/pynccl_allocator.py`, `python/sglang/srt/eplb/expert_location_dispatch.py`.
- Key implementation:
  - `python/sglang/global_config.py` modified +0/-3
  - `python/sglang/srt/distributed/device_communicators/pynccl_allocator.py` modified +2/-2
  - `python/sglang/srt/eplb/expert_location_dispatch.py` modified +2/-2
  - `python/sglang/srt/eplb/expert_location_updater.py` modified +2/-2
- Code diff details:
  - `python/sglang/global_config.py` modified +0/-3
  - `python/sglang/srt/distributed/device_communicators/pynccl_allocator.py` modified +2/-2
  - `python/sglang/srt/eplb/expert_location_dispatch.py` modified +2/-2
  - `python/sglang/srt/eplb/expert_location_updater.py` modified +2/-2
- Key code excerpts:

```diff
diff -- python/sglang/global_config.py
@@ -6,9 +6,6 @@
 class GlobalConfig:
     """
     Store some global constants.
-
-    See also python/sglang/srt/managers/schedule_batch.py::global_server_args_dict, which stores
-    many global runtime arguments as well.
     """

     def __init__(self):
diff -- python/sglang/srt/distributed/device_communicators/pynccl_allocator.py
@@ -5,7 +5,7 @@
 from torch.cuda.memory import CUDAPluggableAllocator

 from sglang.srt.distributed.parallel_state import GroupCoordinator
-from sglang.srt.managers.schedule_batch import global_server_args_dict
+from sglang.srt.server_args import get_global_server_args

 nccl_allocator_source = """
 #include <nccl.h>
@@ -32,7 +32,7 @@


 def is_symmetric_memory_enabled():
-    return global_server_args_dict["enable_symm_mem"]
```
- Reviewed files:
  - runtime: `python/sglang/global_config.py` modified +0/-3; `python/sglang/srt/distributed/device_communicators/pynccl_allocator.py` modified +2/-2; `python/sglang/srt/eplb/expert_location_dispatch.py` modified +2/-2; `python/sglang/srt/eplb/expert_location_updater.py` modified +2/-2; `python/sglang/srt/layers/attention/double_sparsity_backend.py` modified +2/-2; `python/sglang/srt/layers/attention/flashattention_backend.py` modified +2/-2; `python/sglang/srt/layers/attention/flashinfer_mla_backend.py` modified +5/-5; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +2/-2
- Risk and verification: The diff includes test or benchmark paths; rerun those checks plus a minimal launch/accuracy smoke before changing this model again.

### PR #11465 - bailingMoE: Fix Key error of deepep_mode

- Link: https://github.com/sgl-project/sglang/pull/11465
- Status/date: merged / 2025-10-12
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-2, with 18 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "bailingMoE: Fix Key error of deepep_mode"; model line: Ling 2.5 1T; category: bug fix; main diff: `python/sglang/srt/models/bailing_moe.py`.
- Key implementation:
  - `python/sglang/srt/models/bailing_moe.py` modified +2/-2
- Code diff details:
  - `python/sglang/srt/models/bailing_moe.py` modified +2/-2
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/bailing_moe.py
@@ -54,7 +54,7 @@
     RowParallelLinear,
 )
 from sglang.srt.layers.logits_processor import LogitsProcessor
-from sglang.srt.layers.moe import get_moe_a2a_backend
+from sglang.srt.layers.moe import get_deepep_mode, get_moe_a2a_backend
 from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
 from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
 from sglang.srt.layers.moe.token_dispatcher import DeepEPDispatcher
@@ -293,7 +293,7 @@ def __init__(
                 num_local_experts=config.num_experts // self.tp_size,
                 hidden_size=config.hidden_size,
                 params_dtype=config.torch_dtype,
-                deepep_mode=DeepEPMode[global_server_args_dict["deepep_mode"]],
```
- Reviewed files:
  - runtime: `python/sglang/srt/models/bailing_moe.py` modified +2/-2
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.

### PR #11520 - Revert "Deprecate `global_server_args_dict`"

- Link: https://github.com/sgl-project/sglang/pull/11520
- Status/date: merged / 2025-10-13
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 54 files, +321/-240, with 1946 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Revert "Deprecate `global_server_args_dict`""; model line: Ling 2.5 1T; category: bug fix; main diff: `python/sglang/global_config.py`, `python/sglang/srt/distributed/device_communicators/pynccl_allocator.py`, `python/sglang/srt/eplb/expert_location_dispatch.py`.
- Key implementation:
  - `python/sglang/global_config.py` modified +3/-0
  - `python/sglang/srt/distributed/device_communicators/pynccl_allocator.py` modified +2/-2
  - `python/sglang/srt/eplb/expert_location_dispatch.py` modified +2/-2
  - `python/sglang/srt/eplb/expert_location_updater.py` modified +2/-2
- Code diff details:
  - `python/sglang/global_config.py` modified +3/-0
  - `python/sglang/srt/distributed/device_communicators/pynccl_allocator.py` modified +2/-2
  - `python/sglang/srt/eplb/expert_location_dispatch.py` modified +2/-2
  - `python/sglang/srt/eplb/expert_location_updater.py` modified +2/-2
- Key code excerpts:

```diff
diff -- python/sglang/global_config.py
@@ -6,6 +6,9 @@
 class GlobalConfig:
     """
     Store some global constants.
+
+    See also python/sglang/srt/managers/schedule_batch.py::global_server_args_dict, which stores
+    many global runtime arguments as well.
     """

     def __init__(self):
diff -- python/sglang/srt/distributed/device_communicators/pynccl_allocator.py
@@ -5,7 +5,7 @@
 from torch.cuda.memory import CUDAPluggableAllocator

 from sglang.srt.distributed.parallel_state import GroupCoordinator
-from sglang.srt.server_args import get_global_server_args
+from sglang.srt.managers.schedule_batch import global_server_args_dict

 nccl_allocator_source = """
 #include <nccl.h>
@@ -32,7 +32,7 @@


 def is_symmetric_memory_enabled():
-    return get_global_server_args().enable_symm_mem
```
- Reviewed files:
  - runtime: `python/sglang/global_config.py` modified +3/-0; `python/sglang/srt/distributed/device_communicators/pynccl_allocator.py` modified +2/-2; `python/sglang/srt/eplb/expert_location_dispatch.py` modified +2/-2; `python/sglang/srt/eplb/expert_location_updater.py` modified +2/-2; `python/sglang/srt/layers/attention/double_sparsity_backend.py` modified +2/-2; `python/sglang/srt/layers/attention/flashattention_backend.py` modified +2/-2; `python/sglang/srt/layers/attention/flashinfer_mla_backend.py` modified +5/-5; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +2/-2
- Risk and verification: The diff includes test or benchmark paths; rerun those checks plus a minimal launch/accuracy smoke before changing this model again.

### PR #11528 - Depreate `global_server_args_dict`

- Link: https://github.com/sgl-project/sglang/pull/11528
- Status/date: merged / 2025-10-13
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 54 files, +240/-321, with 1946 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Depreate `global_server_args_dict`"; model line: Ling 2.5 1T; category: model implementation change; main diff: `python/sglang/global_config.py`, `python/sglang/srt/distributed/device_communicators/pynccl_allocator.py`, `python/sglang/srt/eplb/expert_location_dispatch.py`.
- Key implementation:
  - `python/sglang/global_config.py` modified +0/-3
  - `python/sglang/srt/distributed/device_communicators/pynccl_allocator.py` modified +2/-2
  - `python/sglang/srt/eplb/expert_location_dispatch.py` modified +2/-2
  - `python/sglang/srt/eplb/expert_location_updater.py` modified +2/-2
- Code diff details:
  - `python/sglang/global_config.py` modified +0/-3
  - `python/sglang/srt/distributed/device_communicators/pynccl_allocator.py` modified +2/-2
  - `python/sglang/srt/eplb/expert_location_dispatch.py` modified +2/-2
  - `python/sglang/srt/eplb/expert_location_updater.py` modified +2/-2
- Key code excerpts:

```diff
diff -- python/sglang/global_config.py
@@ -6,9 +6,6 @@
 class GlobalConfig:
     """
     Store some global constants.
-
-    See also python/sglang/srt/managers/schedule_batch.py::global_server_args_dict, which stores
-    many global runtime arguments as well.
     """

     def __init__(self):
diff -- python/sglang/srt/distributed/device_communicators/pynccl_allocator.py
@@ -5,7 +5,7 @@
 from torch.cuda.memory import CUDAPluggableAllocator

 from sglang.srt.distributed.parallel_state import GroupCoordinator
-from sglang.srt.managers.schedule_batch import global_server_args_dict
+from sglang.srt.server_args import get_global_server_args

 nccl_allocator_source = """
 #include <nccl.h>
@@ -32,7 +32,7 @@


 def is_symmetric_memory_enabled():
-    return global_server_args_dict["enable_symm_mem"]
```
- Reviewed files:
  - runtime: `python/sglang/global_config.py` modified +0/-3; `python/sglang/srt/distributed/device_communicators/pynccl_allocator.py` modified +2/-2; `python/sglang/srt/eplb/expert_location_dispatch.py` modified +2/-2; `python/sglang/srt/eplb/expert_location_updater.py` modified +2/-2; `python/sglang/srt/layers/attention/double_sparsity_backend.py` modified +2/-2; `python/sglang/srt/layers/attention/flashattention_backend.py` modified +2/-2; `python/sglang/srt/layers/attention/flashinfer_mla_backend.py` modified +5/-5; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +2/-2
- Risk and verification: The diff includes test or benchmark paths; rerun those checks plus a minimal launch/accuracy smoke before changing this model again.

### PR #11685 - [Lint] Add `python/sglang` to ruff F401 checks and remove unused imports in files

- Link: https://github.com/sgl-project/sglang/pull/11685
- Status/date: merged / 2025-10-17
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 100 files, +90/-235, with 1164 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Lint] Add `python/sglang` to ruff F401 checks and remove unused imports in files"; model line: Ling 2.5 1T; category: model support/runtime entry; main diff: `.pre-commit-config.yaml`, `python/sglang/srt/_custom_ops.py`, `python/sglang/srt/compilation/cuda_piecewise_backend.py`.
- Key implementation:
  - `.pre-commit-config.yaml` modified +3/-3
  - `python/sglang/srt/_custom_ops.py` modified +1/-1
  - `python/sglang/srt/compilation/cuda_piecewise_backend.py` modified +0/-1
  - `python/sglang/srt/configs/deepseekvl2.py` modified +0/-1
- Code diff details:
  - `.pre-commit-config.yaml` modified +3/-3
  - `python/sglang/srt/_custom_ops.py` modified +1/-1
  - `python/sglang/srt/compilation/cuda_piecewise_backend.py` modified +0/-1
  - `python/sglang/srt/configs/deepseekvl2.py` modified +0/-1
- Key code excerpts:

```diff
diff -- .pre-commit-config.yaml
@@ -27,9 +27,9 @@ repos:
     rev: v0.11.7
     hooks:
       - id: ruff
-        args: [--select=F401, --fixable=F401]
-        files: ^(benchmark/|docs/|examples/)
-        exclude: \.ipynb$|^python/sglang/srt/grpc/.*_pb2\.py$|^python/sglang/srt/grpc/.*_pb2_grpc\.py$|^python/sglang/srt/grpc/.*_pb2\.pyi$|^python/sglang/srt/grpc/.*_pb2_grpc\.pyi$
+        args: [--select=F401,F821, --fixable=F401]
+        files: ^(benchmark/|docs/|examples/|python/sglang/)
+        exclude: __init__\.py$|\.ipynb$|^python/sglang/srt/grpc/.*_pb2\.py$|^python/sglang/srt/grpc/.*_pb2_grpc\.py$|^python/sglang/srt/grpc/.*_pb2\.pyi$|^python/sglang/srt/grpc/.*_pb2_grpc\.pyi$
   - repo: https://github.com/psf/black
     rev: 24.10.0
     hooks:
diff -- python/sglang/srt/_custom_ops.py
@@ -15,7 +15,7 @@
     # ROCm does not use vllm custom allreduce
     if use_vllm_custom_allreduce and not is_hip():
         try:
-            import vllm._C
+            import vllm._C  # noqa: F401
         except ImportError as e:
             logger.warning("Failed to import from vllm._C with %r", e)
     else:
```
- Reviewed files:
  - runtime: `python/sglang/srt/_custom_ops.py` modified +1/-1; `python/sglang/srt/compilation/cuda_piecewise_backend.py` modified +0/-1; `python/sglang/srt/configs/deepseekvl2.py` modified +0/-1; `python/sglang/srt/configs/dots_vlm.py` modified +2/-7; `python/sglang/srt/configs/falcon_h1.py` modified +1/-6; `python/sglang/srt/configs/qwen3_next.py` modified +0/-1; `python/sglang/srt/connector/remote_instance.py` modified +1/-1; `python/sglang/srt/disaggregation/ascend/transfer_engine.py` modified +1/-1
  - other: `.pre-commit-config.yaml` modified +3/-3
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.

### PR #11847 - [9/N] MoE Refactor: cleanup dispatcher interfaces

- Link: https://github.com/sgl-project/sglang/pull/11847
- Status/date: merged / 2025-10-20
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 24 files, +394/-428, with 1948 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[9/N] MoE Refactor: cleanup dispatcher interfaces"; model line: Ling 2.5 1T; category: model implementation change; main diff: `python/sglang/srt/layers/dp_attention.py`, `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`.
- Key implementation:
  - `python/sglang/srt/layers/dp_attention.py` modified +17/-0; symbols: set_is_extend_in_batch, get_is_extend_in_batch
  - `python/sglang/srt/layers/moe/ep_moe/kernels.py` modified +3/-1
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +69/-99; symbols: run_moe_core
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +44/-35; symbols: create_moe_dispatcher, run_moe_core
- Code diff details:
  - `python/sglang/srt/layers/dp_attention.py` modified +17/-0
  - `python/sglang/srt/layers/moe/ep_moe/kernels.py` modified +3/-1
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +69/-99
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +44/-35
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/dp_attention.py
@@ -87,6 +87,7 @@ class _DpGatheredBufferWrapper:
     _global_dp_buffer_len: int
     _local_dp_buffer_len: int
     _global_num_tokens: Optional[List[int]]
+    _is_extend_in_batch: bool

     @classmethod
     def set_metadata(cls, hidden_size: int, dtype: torch.dtype, device: torch.device):
@@ -145,6 +146,14 @@ def get_dp_dtype(cls) -> torch.dtype:
     def get_dp_device(cls) -> torch.device:
         return cls._device

+    @classmethod
+    def set_is_extend_in_batch(cls, is_extend_in_batch: bool):
diff -- python/sglang/srt/layers/moe/ep_moe/kernels.py
@@ -566,7 +566,9 @@ def ep_scatter(
         scale_hidden_size = ceil_div(scale_hidden_size, 4)

     assert m_indices.shape[0] % BLOCK_E == 0
-    assert recv_x_scale.dtype == output_tensor_scale.dtype
+    assert (
+        recv_x_scale.dtype == output_tensor_scale.dtype
+    ), f"recv_x_scale.dtype: {recv_x_scale.dtype}, output_tensor_scale.dtype: {output_tensor_scale.dtype}"
     assert recv_x_scale.shape[1] == output_tensor_scale.shape[1] == scale_hidden_size

     _fwd_kernel_ep_scatter_1[(grid,)](
```
- Reviewed files:
  - runtime: `python/sglang/srt/layers/dp_attention.py` modified +17/-0; `python/sglang/srt/layers/moe/ep_moe/kernels.py` modified +3/-1; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +69/-99; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +44/-35; `python/sglang/srt/layers/moe/token_dispatcher/__init__.py` modified +2/-0; `python/sglang/srt/layers/moe/token_dispatcher/base.py` modified +1/-1; `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +86/-91; `python/sglang/srt/layers/moe/token_dispatcher/mooncake.py` modified +37/-39
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.

### PR #12369 - Enable bailing_moe to support TP=16

- Link: https://github.com/sgl-project/sglang/pull/12369
- Status/date: merged / 2025-10-31
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 1 files, +9/-2, with 24 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Enable bailing_moe to support TP=16"; model line: Ling 2.5 1T; category: model support/runtime entry; main diff: `python/sglang/srt/models/bailing_moe.py`.
- Key implementation:
  - `python/sglang/srt/models/bailing_moe.py` modified +9/-2
- Code diff details:
  - `python/sglang/srt/models/bailing_moe.py` modified +9/-2
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/bailing_moe.py
@@ -420,14 +420,21 @@ def __init__(
         attn_tp_size = get_attention_tp_size()

         assert self.total_num_heads % attn_tp_size == 0
-        assert self.total_kv_heads % attn_tp_size == 0
+        if self.total_kv_heads >= attn_tp_size:
+            # Number of KV heads is greater than TP size, so we partition
+            # the KV heads across multiple tensor parallel GPUs.
+            assert self.total_kv_heads % attn_tp_size == 0
+        else:
+            # Number of KV heads is less than TP size, so we replicate
+            # the KV heads across multiple tensor parallel GPUs.
+            assert attn_tp_size % self.total_kv_heads == 0
         assert self.total_num_heads >= self.total_kv_heads
```
- Reviewed files:
  - runtime: `python/sglang/srt/models/bailing_moe.py` modified +9/-2
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.

### PR #14337 - remove unecessary dual stream token threshold from the rest of models (qwen moe, kimi linear, etc.)

- Link: https://github.com/sgl-project/sglang/pull/14337
- Status/date: merged / 2025-12-07
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 4 files, +0/-8, with 50 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "remove unecessary dual stream token threshold from the rest of models (qwen moe, kimi linear, etc.)"; model line: Ling 2.5 1T; category: model implementation change; main diff: `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/kimi_linear.py`, `python/sglang/srt/models/llada2.py`.
- Key implementation:
  - `python/sglang/srt/models/bailing_moe.py` modified +0/-2
  - `python/sglang/srt/models/kimi_linear.py` modified +0/-2
  - `python/sglang/srt/models/llada2.py` modified +0/-2
  - `python/sglang/srt/models/qwen2_moe.py` modified +0/-2
- Code diff details:
  - `python/sglang/srt/models/bailing_moe.py` modified +0/-2
  - `python/sglang/srt/models/kimi_linear.py` modified +0/-2
  - `python/sglang/srt/models/llada2.py` modified +0/-2
  - `python/sglang/srt/models/qwen2_moe.py` modified +0/-2
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/bailing_moe.py
@@ -349,11 +349,9 @@ def forward_normal(
         num_tokens, hidden_size = hidden_states.shape
         hidden_states = hidden_states.view(-1, hidden_size)

-        DUAL_STREAM_TOKEN_THRESHOLD = 1024
         if (
             self.alt_stream is not None
             and hidden_states.shape[0] > 0
-            and hidden_states.shape[0] <= DUAL_STREAM_TOKEN_THRESHOLD
             and get_is_capture_mode()
         ):
             final_hidden_states, shared_output = self.forward_normal_dual_stream(
diff -- python/sglang/srt/models/kimi_linear.py
@@ -125,13 +125,11 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
         hidden_states = hidden_states.view(-1, hidden_size)

         shared_output = None
-        DUAL_STREAM_TOKEN_THRESHOLD = 1024

         if (
             self.alt_stream is not None
             and self.num_shared_experts is not None
             and hidden_states.shape[0] > 0
-            and hidden_states.shape[0] <= DUAL_STREAM_TOKEN_THRESHOLD
             and get_is_capture_mode()
         ):
             current_stream = torch.cuda.current_stream()
```
- Reviewed files:
  - runtime: `python/sglang/srt/models/bailing_moe.py` modified +0/-2; `python/sglang/srt/models/kimi_linear.py` modified +0/-2; `python/sglang/srt/models/llada2.py` modified +0/-2; `python/sglang/srt/models/qwen2_moe.py` modified +0/-2
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.

### PR #13730 - [bugfix] fix TBO crashes when attn_tp_size > 1

- Link: https://github.com/sgl-project/sglang/pull/13730
- Status/date: merged / 2025-12-12
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 20 files, +285/-16, with 617 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[bugfix] fix TBO crashes when attn_tp_size > 1"; model line: Ling 2.5 1T; category: bug fix; main diff: `python/sglang/srt/batch_overlap/operations.py`, `python/sglang/srt/batch_overlap/two_batch_overlap.py`, `python/sglang/srt/layers/communicator.py`.
- Key implementation:
  - `python/sglang/srt/batch_overlap/operations.py` modified +10/-8
  - `python/sglang/srt/batch_overlap/two_batch_overlap.py` modified +39/-6; symbols: _pad, _model_forward_tbo_merge_outputs
  - `python/sglang/srt/layers/communicator.py` modified +14/-1; symbols: _should_gather_for_tbo
  - `python/sglang/srt/model_executor/forward_batch_info.py` modified +9/-0
- Code diff details:
  - `python/sglang/srt/batch_overlap/operations.py` modified +10/-8
  - `python/sglang/srt/batch_overlap/two_batch_overlap.py` modified +39/-6
  - `python/sglang/srt/layers/communicator.py` modified +14/-1
  - `python/sglang/srt/model_executor/forward_batch_info.py` modified +9/-0
- Key code excerpts:

```diff
diff -- python/sglang/srt/batch_overlap/operations.py
@@ -83,7 +83,7 @@ def __init__(self, debug_name: str, stages: List[Stage], inputs: dict):
         # handling DP attention
         forward_batch: ForwardBatch = inputs["forward_batch"]
         self._global_dp_buffer_len = forward_batch.global_dp_buffer_len
-        self._local_dp_buffer_len = forward_batch.input_ids.shape[0]
+        self._local_dp_buffer_len = forward_batch.tbo_padded_len
         self._global_num_tokens = forward_batch.global_num_tokens_cpu
         self._is_dp_max_padding = forward_batch.dp_padding_mode.is_max_len()

@@ -92,13 +92,15 @@ def next(self):

         stage = self._stages[self._index]

-        if self._global_dp_buffer_len is not None:
diff -- python/sglang/srt/batch_overlap/two_batch_overlap.py
@@ -20,6 +20,7 @@
     CommunicateSummableTensorPairFn,
     ScatterMode,
 )
+from sglang.srt.layers.dp_attention import get_attention_tp_size
 from sglang.srt.layers.moe import (
     get_deepep_mode,
     get_moe_a2a_backend,
@@ -630,6 +631,11 @@ def filter_batch(
             ), f"{key=} {old_value=} {num_tokens=} {batch=}"
             output_dict[key] = old_value[start_token_index:end_token_index]

+        attention_tp_size = get_attention_tp_size()
+        output_dict["tbo_padded_len"] = (
```
- Reviewed files:
  - runtime: `python/sglang/srt/batch_overlap/operations.py` modified +10/-8; `python/sglang/srt/batch_overlap/two_batch_overlap.py` modified +39/-6; `python/sglang/srt/layers/communicator.py` modified +14/-1; `python/sglang/srt/model_executor/forward_batch_info.py` modified +9/-0; `python/sglang/srt/models/bailing_moe.py` modified +4/-0; `python/sglang/srt/models/deepseek_v2.py` modified +2/-0; `python/sglang/srt/models/falcon_h1.py` modified +3/-1; `python/sglang/srt/models/glm4_moe.py` modified +2/-0
  - tests: `test/srt/ep/test_deepep_small.py` modified +178/-0
- Risk and verification: The diff includes test or benchmark paths; rerun those checks plus a minimal launch/accuracy smoke before changing this model again.

### PR #15526 - Optimize Bailing-MoE with FlashInfer Fused All-Reduce

- Link: https://github.com/sgl-project/sglang/pull/15526
- Status/date: merged / 2025-12-21
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 1 files, +58/-20, with 182 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Optimize Bailing-MoE with FlashInfer Fused All-Reduce"; model line: Ling 2.5 1T; category: performance/backend optimization; main diff: `python/sglang/srt/models/bailing_moe.py`.
- Key implementation:
  - `python/sglang/srt/models/bailing_moe.py` modified +58/-20
- Code diff details:
  - `python/sglang/srt/models/bailing_moe.py` modified +58/-20
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/bailing_moe.py
@@ -19,7 +19,7 @@
 # limitations under the License.
 """SGLang BailingMoE model."""
 import logging
-from typing import Iterable, Optional, Tuple, Union
+from typing import Iterable, List, Optional, Tuple, Union

 import torch
 import torch.nn.functional as F
@@ -54,7 +54,11 @@
     RowParallelLinear,
 )
 from sglang.srt.layers.logits_processor import LogitsProcessor
-from sglang.srt.layers.moe import get_deepep_mode, get_moe_a2a_backend
```
- Reviewed files:
  - runtime: `python/sglang/srt/models/bailing_moe.py` modified +58/-20
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.

### PR #15835 - [Feature] JIT Fused QK norm + qk norm clean up

- Link: https://github.com/sgl-project/sglang/pull/15835
- Status/date: merged / 2025-12-28
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 15 files, +827/-127, with 1151 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feature] JIT Fused QK norm + qk norm clean up"; model line: Ling 2.5 1T; category: performance/backend optimization; main diff: `python/sglang/jit_kernel/benchmark/bench_qknorm.py`, `python/sglang/jit_kernel/csrc/norm.cuh`, `python/sglang/jit_kernel/include/sgl_kernel/runtime.cuh`.
- Key implementation:
  - `python/sglang/jit_kernel/benchmark/bench_qknorm.py` added +130/-0; symbols: sglang_aot_qknorm, sglang_jit_qknorm, flashinfer_qknorm, torch_impl_qknorm
  - `python/sglang/jit_kernel/csrc/norm.cuh` added +202/-0; symbols: void, auto, int64_t, tvm
  - `python/sglang/jit_kernel/include/sgl_kernel/runtime.cuh` added +26/-0
  - `python/sglang/jit_kernel/include/sgl_kernel/tensor.h` modified +10/-0
- Code diff details:
  - `python/sglang/jit_kernel/benchmark/bench_qknorm.py` added +130/-0
  - `python/sglang/jit_kernel/csrc/norm.cuh` added +202/-0
  - `python/sglang/jit_kernel/include/sgl_kernel/runtime.cuh` added +26/-0
  - `python/sglang/jit_kernel/include/sgl_kernel/tensor.h` modified +10/-0
- Key code excerpts:

```diff
diff -- python/sglang/jit_kernel/benchmark/bench_qknorm.py
@@ -0,0 +1,130 @@
+import itertools
+import os
+from typing import Tuple
+
+import torch
+import triton
+import triton.testing
+
+IS_CI = (
+    os.getenv("CI", "false").lower() == "true"
+    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
+)
+
diff -- python/sglang/jit_kernel/csrc/norm.cuh
@@ -0,0 +1,202 @@
+#include <sgl_kernel/runtime.cuh>
+#include <sgl_kernel/tensor.h>
+#include <sgl_kernel/utils.cuh>
+#include <sgl_kernel/utils.h>
+#include <sgl_kernel/warp.cuh>
+
+#include <cuda_bf16.h>
+#include <cuda_fp16.h>
+#include <dlpack/dlpack.h>
+#include <tvm/ffi/container/tensor.h>
+
+#include <cstdint>
+#include <type_traits>
```
- Reviewed files:
  - runtime: `python/sglang/jit_kernel/benchmark/bench_qknorm.py` added +130/-0; `python/sglang/jit_kernel/csrc/norm.cuh` added +202/-0; `python/sglang/jit_kernel/include/sgl_kernel/runtime.cuh` added +26/-0; `python/sglang/jit_kernel/include/sgl_kernel/tensor.h` modified +10/-0; `python/sglang/jit_kernel/include/sgl_kernel/utils.cuh` modified +31/-1; `python/sglang/jit_kernel/include/sgl_kernel/warp.cuh` added +14/-0; `python/sglang/jit_kernel/norm.py` added +55/-0; `python/sglang/jit_kernel/tests/test_qknorm.py` added +85/-0
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.

### PR #13715 - Fix EPLB + FP4 Quantization Compatibility Issue

- Link: https://github.com/sgl-project/sglang/pull/13715
- Status/date: merged / 2026-01-10
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 8 files, +49/-3, with 157 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix EPLB + FP4 Quantization Compatibility Issue"; model line: Ling 2.5 1T; category: bug fix; main diff: `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/deepseek_v2.py`.
- Key implementation:
  - `python/sglang/srt/layers/moe/utils.py` modified +12/-0; symbols: filter_moe_weight_param_global_expert
  - `python/sglang/srt/models/bailing_moe.py` modified +4/-0
  - `python/sglang/srt/models/deepseek_v2.py` modified +7/-1
  - `python/sglang/srt/models/glm4_moe.py` modified +4/-0
- Code diff details:
  - `python/sglang/srt/layers/moe/utils.py` modified +12/-0
  - `python/sglang/srt/models/bailing_moe.py` modified +4/-0
  - `python/sglang/srt/models/deepseek_v2.py` modified +7/-1
  - `python/sglang/srt/models/glm4_moe.py` modified +4/-0
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/utils.py
@@ -249,6 +249,18 @@ def get_tbo_token_distribution_threshold() -> float:
     return TBO_TOKEN_DISTRIBUTION_THRESHOLD


+def filter_moe_weight_param_global_expert(name, x, num_local_experts):
+    """
+    Filter out for MoE expert parameters that requires global expert.
+    """
+    return (
+        not getattr(x, "_sglang_require_global_experts", False)
+        and not name.endswith("_blockscale_swizzled")
+        and x.data.ndim > 0
+        and x.data.shape[0] == num_local_experts
+    )
diff -- python/sglang/srt/models/bailing_moe.py
@@ -63,6 +63,7 @@
 from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
 from sglang.srt.layers.moe.token_dispatcher import DeepEPDispatcher
 from sglang.srt.layers.moe.topk import TopK
+from sglang.srt.layers.moe.utils import filter_moe_weight_param_global_expert
 from sglang.srt.layers.quantization.base_config import QuantizationConfig
 from sglang.srt.layers.radix_attention import RadixAttention
 from sglang.srt.layers.rotary_embedding import get_rope
@@ -324,6 +325,9 @@ def get_moe_weights(self):
             x.data
             for name, x in self.experts.named_parameters()
             if name not in ["correction_bias"]
+            and filter_moe_weight_param_global_expert(
+                name, x, self.experts.num_local_experts
```
- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/utils.py` modified +12/-0; `python/sglang/srt/models/bailing_moe.py` modified +4/-0; `python/sglang/srt/models/deepseek_v2.py` modified +7/-1; `python/sglang/srt/models/glm4_moe.py` modified +4/-0; `python/sglang/srt/models/gpt_oss.py` modified +4/-0; `python/sglang/srt/models/longcat_flash.py` modified +4/-0; `python/sglang/srt/models/qwen2_moe.py` modified +7/-1; `python/sglang/srt/models/qwen3_moe.py` modified +7/-1
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.

### PR #17570 - Use attn tp group in embedding for more models

- Link: https://github.com/sgl-project/sglang/pull/17570
- Status/date: merged / 2026-01-24
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 19 files, +19/-19, with 171 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Use attn tp group in embedding for more models"; model line: Ling 2.5 1T; category: model implementation change; main diff: `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/bailing_moe_nextn.py`, `python/sglang/srt/models/falcon_h1.py`.
- Key implementation:
  - `python/sglang/srt/models/bailing_moe.py` modified +1/-1
  - `python/sglang/srt/models/bailing_moe_nextn.py` modified +1/-1
  - `python/sglang/srt/models/falcon_h1.py` modified +1/-1
  - `python/sglang/srt/models/glm4.py` modified +1/-1
- Code diff details:
  - `python/sglang/srt/models/bailing_moe.py` modified +1/-1
  - `python/sglang/srt/models/bailing_moe_nextn.py` modified +1/-1
  - `python/sglang/srt/models/falcon_h1.py` modified +1/-1
  - `python/sglang/srt/models/glm4.py` modified +1/-1
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/bailing_moe.py
@@ -717,7 +717,7 @@ def __init__(
                 self.embed_dim,
                 quant_config=quant_config,
                 prefix=add_prefix("word_embeddings", prefix),
-                enable_tp=not is_dp_attention_enabled(),
+                use_attn_tp_group=is_dp_attention_enabled(),
             )
         else:
             self.word_embeddings = PPMissingLayer()
diff -- python/sglang/srt/models/bailing_moe_nextn.py
@@ -62,7 +62,7 @@ def __init__(
         self.word_embeddings = VocabParallelEmbedding(
             config.vocab_size,
             config.hidden_size,
-            enable_tp=not is_dp_attention_enabled(),
+            use_attn_tp_group=is_dp_attention_enabled(),
             prefix=add_prefix("word_embeddings", prefix),
         )

```
- Reviewed files:
  - runtime: `python/sglang/srt/models/bailing_moe.py` modified +1/-1; `python/sglang/srt/models/bailing_moe_nextn.py` modified +1/-1; `python/sglang/srt/models/falcon_h1.py` modified +1/-1; `python/sglang/srt/models/glm4.py` modified +1/-1; `python/sglang/srt/models/glm4_moe.py` modified +1/-1; `python/sglang/srt/models/glm4_moe_lite.py` modified +1/-1; `python/sglang/srt/models/glm4_moe_nextn.py` modified +1/-1; `python/sglang/srt/models/gpt_oss.py` modified +1/-1
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.

### PR #17808 - Fix the scenario where eh_proj is quantized in the bailing moe nextn weights

- Link: https://github.com/sgl-project/sglang/pull/17808
- Status/date: merged / 2026-01-30
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 1 files, +9/-2, with 32 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix the scenario where eh_proj is quantized in the bailing moe nextn weights"; model line: Ling 2.5 1T; category: bug fix; main diff: `python/sglang/srt/models/bailing_moe_nextn.py`.
- Key implementation:
  - `python/sglang/srt/models/bailing_moe_nextn.py` modified +9/-2
- Code diff details:
  - `python/sglang/srt/models/bailing_moe_nextn.py` modified +9/-2
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/bailing_moe_nextn.py
@@ -28,6 +28,7 @@
 from sglang.srt.distributed import get_tensor_model_parallel_world_size
 from sglang.srt.layers.dp_attention import is_dp_attention_enabled
 from sglang.srt.layers.layernorm import RMSNorm
+from sglang.srt.layers.linear import ReplicatedLinear
 from sglang.srt.layers.logits_processor import LogitsProcessor
 from sglang.srt.layers.quantization.base_config import QuantizationConfig
 from sglang.srt.layers.vocab_parallel_embedding import (
@@ -69,7 +70,13 @@ def __init__(
         self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
         self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

-        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
+        self.eh_proj = ReplicatedLinear(
```
- Reviewed files:
  - runtime: `python/sglang/srt/models/bailing_moe_nextn.py` modified +9/-2
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.

### PR #15119 - feat: Add Ling Flash v2.0 support for Eagle3

- Link: https://github.com/sgl-project/sglang/pull/15119
- Status/date: merged / 2026-02-01
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 1 files, +30/-1, with 76 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "feat: Add Ling Flash v2.0 support for Eagle3"; model line: Ling 2.5 1T; category: model support/runtime entry; main diff: `python/sglang/srt/models/bailing_moe.py`.
- Key implementation:
  - `python/sglang/srt/models/bailing_moe.py` modified +30/-1; symbols: set_eagle3_layers_to_capture
- Code diff details:
  - `python/sglang/srt/models/bailing_moe.py` modified +30/-1
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/bailing_moe.py
@@ -738,6 +738,8 @@ def __init__(
         else:
             self.norm = PPMissingLayer(return_tuple=True)

+        self.layers_to_capture = []
+
     def forward(
         self,
         input_ids: torch.Tensor,
@@ -760,6 +762,10 @@ def forward(
         aux_hidden_states = []
         for i in range(self.start_layer, self.end_layer):
             with get_global_expert_distribution_recorder().with_current_layer(i):
+                if i in self.layers_to_capture:
```
- Reviewed files:
  - runtime: `python/sglang/srt/models/bailing_moe.py` modified +30/-1
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.

### PR #18598 - Support LingV2_5 model

- Link: https://github.com/sgl-project/sglang/pull/18598
- Status/date: merged / 2026-02-13
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 16 files, +4042/-23, with 4377 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support LingV2_5 model"; model line: Ling 2.5 1T; category: model support/runtime entry; main diff: `python/sglang/srt/configs/__init__.py`, `python/sglang/srt/configs/bailing_hybrid.py`, `python/sglang/srt/configs/model_config.py`.
- Key implementation:
  - `python/sglang/srt/configs/__init__.py` modified +2/-0
  - `python/sglang/srt/configs/bailing_hybrid.py` added +188/-0; symbols: HybridLayerType, BailingHybridConfig, __init__, layers_block_type
  - `python/sglang/srt/configs/model_config.py` modified +20/-0
  - `python/sglang/srt/layers/attention/attention_registry.py` modified +3/-0
- Code diff details:
  - `python/sglang/srt/configs/__init__.py` modified +2/-0
  - `python/sglang/srt/configs/bailing_hybrid.py` added +188/-0
  - `python/sglang/srt/configs/model_config.py` modified +20/-0
  - `python/sglang/srt/layers/attention/attention_registry.py` modified +3/-0
- Key code excerpts:

```diff
diff -- python/sglang/srt/configs/__init__.py
@@ -1,4 +1,5 @@
 from sglang.srt.configs.afmoe import AfmoeConfig
+from sglang.srt.configs.bailing_hybrid import BailingHybridConfig
 from sglang.srt.configs.chatglm import ChatGLMConfig
 from sglang.srt.configs.dbrx import DbrxConfig
 from sglang.srt.configs.deepseekvl2 import DeepseekVL2Config
@@ -30,6 +31,7 @@

 __all__ = [
     "AfmoeConfig",
+    "BailingHybridConfig",
     "ExaoneConfig",
     "ChatGLMConfig",
     "DbrxConfig",
diff -- python/sglang/srt/configs/bailing_hybrid.py
@@ -0,0 +1,188 @@
+# coding=utf-8
+# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
+#
+# Unless required by applicable law or agreed to in writing, software
+# distributed under the License is distributed on an "AS IS" BASIS,
+# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
+# See the License for the specific language governing permissions and
```
- Reviewed files:
  - runtime: `python/sglang/srt/configs/__init__.py` modified +2/-0; `python/sglang/srt/configs/bailing_hybrid.py` added +188/-0; `python/sglang/srt/configs/model_config.py` modified +20/-0; `python/sglang/srt/layers/attention/attention_registry.py` modified +3/-0; `python/sglang/srt/layers/attention/fla/layernorm_gated.py` modified +26/-5; `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +369/-0; `python/sglang/srt/layers/attention/linear/lightning_attn.py` added +767/-0; `python/sglang/srt/layers/attention/linear/linear_metadata.py` added +70/-0
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.

### PR #18793 - Cleanup debug log for Ring model

- Link: https://github.com/sgl-project/sglang/pull/18793
- Status/date: merged / 2026-02-13
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 2 files, +9/-11, with 69 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Cleanup debug log for Ring model"; model line: Ling 2.5 1T; category: bug fix; main diff: `python/sglang/srt/configs/mamba_utils.py`, `python/sglang/srt/models/bailing_moe_linear.py`.
- Key implementation:
  - `python/sglang/srt/configs/mamba_utils.py` modified +1/-1
  - `python/sglang/srt/models/bailing_moe_linear.py` modified +8/-10
- Code diff details:
  - `python/sglang/srt/configs/mamba_utils.py` modified +1/-1
  - `python/sglang/srt/models/bailing_moe_linear.py` modified +8/-10
- Key code excerpts:

```diff
diff -- python/sglang/srt/configs/mamba_utils.py
@@ -102,7 +102,7 @@ def mamba2_state_dtype(config=None) -> Mamba2StateDType:
         else:
             ssm_dtype = dtype_map[env_ssm_dtype]

-    logger.info(f"Mamba2 state dtype: conv_dtype={conv_dtype}, ssm_dtype={ssm_dtype}")
+    logger.debug(f"Mamba2 state dtype: conv_dtype={conv_dtype}, ssm_dtype={ssm_dtype}")

     return Mamba2StateDType(conv=conv_dtype, temporal=ssm_dtype)

diff -- python/sglang/srt/models/bailing_moe_linear.py
@@ -77,6 +77,7 @@
     is_sm100_supported,
     make_layers,
 )
+from sglang.srt.utils.common import rank0_log

 _is_hip = is_hip()
 _is_cuda = is_cuda()
@@ -423,7 +424,7 @@ def __init__(
         # minimax / seg_la / fla
         # TODO support fla
         self.linear_backend = getattr(config, "linear_backend", "seg_la")
-        logger.info(f"linear_backend in bailing_moe_linear: {self.linear_backend}")
+        logger.debug(f"linear_backend in bailing_moe_linear: {self.linear_backend}")
```
- Reviewed files:
  - runtime: `python/sglang/srt/configs/mamba_utils.py` modified +1/-1; `python/sglang/srt/models/bailing_moe_linear.py` modified +8/-10
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.

### PR #18860 - update pre-commit config

- Link: https://github.com/sgl-project/sglang/pull/18860
- Status/date: merged / 2026-02-15
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 100 files, +170/-159, with 1254 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "update pre-commit config"; model line: Ling 2.5 1T; category: docs/tests/CI; main diff: `.github/workflows/lint.yml`, `.pre-commit-config.yaml`, `3rdparty/amd/tuning/benchmark_moe_rocm.py`.
- Key implementation:
  - `.github/workflows/lint.yml` modified +2/-2
  - `.pre-commit-config.yaml` modified +6/-6
  - `3rdparty/amd/tuning/benchmark_moe_rocm.py` modified +2/-4
  - `benchmark/fla/benchmark_layernorm_gated.py` modified +3/-1
- Code diff details:
  - `.github/workflows/lint.yml` modified +2/-2
  - `.pre-commit-config.yaml` modified +6/-6
  - `3rdparty/amd/tuning/benchmark_moe_rocm.py` modified +2/-4
  - `benchmark/fla/benchmark_layernorm_gated.py` modified +3/-1
- Key code excerpts:

```diff
diff -- .github/workflows/lint.yml
@@ -26,9 +26,9 @@ jobs:
         run: SKIP=no-commit-to-branch pre-commit run --all-files --show-diff-on-failure

       - name: Run sgl-kernel clang-format checks
-        uses: DoozyX/clang-format-lint-action@v0.18.1
+        uses: DoozyX/clang-format-lint-action@v0.20
         with:
           source: sgl-kernel
           extensions: h,c,cpp,hpp,cu,cuh,cc
-          clangFormatVersion: 18
+          clangFormatVersion: 20
           style: file
diff -- .pre-commit-config.yaml
@@ -3,7 +3,7 @@ exclude: ^(python/sglang/multimodal_gen/csrc|python/sglang/jit_kernel/flash_atte

 repos:
   - repo: https://github.com/pre-commit/pre-commit-hooks
-    rev: v5.0.0
+    rev: v6.0.0
     hooks:
       - id: check-symlinks
       - id: destroyed-symlinks
@@ -21,12 +21,12 @@ repos:
       - id: debug-statements
       - id: no-commit-to-branch
   - repo: https://github.com/PyCQA/isort
-    rev: 5.13.2
```
- Reviewed files:
  - runtime: `python/sglang/jit_kernel/include/sgl_kernel/type.cuh` modified +16/-15; `python/sglang/multimodal_gen/apps/webui/main.py` modified +2/-4; `python/sglang/multimodal_gen/configs/models/encoders/qwen3.py` modified +1/-0; `python/sglang/multimodal_gen/runtime/cache/__init__.py` modified +1/-0; `python/sglang/multimodal_gen/runtime/distributed/parallel_state.py` modified +2/-1; `python/sglang/multimodal_gen/runtime/layers/activation.py` modified +1/-0; `python/sglang/multimodal_gen/runtime/layers/layernorm.py` modified +1/-0; `python/sglang/multimodal_gen/runtime/layers/rotary_embedding.py` modified +1/-0
  - docs/bench: `benchmark/fla/benchmark_layernorm_gated.py` modified +3/-1; `benchmark/tip_suggestion/bench_other.py` modified +2/-8; `benchmark/tip_suggestion/bench_sglang.py` modified +2/-8; `benchmark/tip_suggestion/lmql_funcs.py` modified +2/-8; `docs/advanced_features/lora.ipynb` modified +10/-20; `docs/advanced_features/structured_outputs.ipynb` modified +0/-1; `docs/advanced_features/structured_outputs_for_reasoning_models.ipynb` modified +0/-1; `docs/advanced_features/vlm_query.ipynb` modified +0/-1
  - other: `.github/workflows/lint.yml` modified +2/-2; `.pre-commit-config.yaml` modified +6/-6; `3rdparty/amd/tuning/benchmark_moe_rocm.py` modified +2/-4
- Risk and verification: The diff includes test or benchmark paths; rerun those checks plus a minimal launch/accuracy smoke before changing this model again.

### PR #19382 - Add NPU basic function testcases

- Link: https://github.com/sgl-project/sglang/pull/19382
- Status/date: merged / 2026-03-16
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 87 files, +4587/-333, with 5347 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add NPU basic function testcases"; model line: Ling 2.5 1T; category: model support/runtime entry; main diff: `.github/workflows/nightly-test-npu.yml`, `python/sglang/test/ascend/disaggregation_utils.py`, `python/sglang/test/ascend/test_ascend_utils.py`.
- Key implementation:
  - `.github/workflows/nightly-test-npu.yml` modified +37/-17
  - `python/sglang/test/ascend/disaggregation_utils.py` added +153/-0; symbols: TestDisaggregationBase, setUpClass, launch_lb, wait_server_ready
  - `python/sglang/test/ascend/test_ascend_utils.py` modified +495/-54; symbols: ModelTestConfig, run_command, get_benchmark_args, run_bench_serving
  - `test/registered/ascend/basic_function/HiCache/test_npu_hierarchical_cache.py` added +127/-0; symbols: TestNPUHierarchicalCache, setUpClass, tearDownClass, test_hierarchical_cache_reused_long_identical
- Code diff details:
  - `.github/workflows/nightly-test-npu.yml` modified +37/-17
  - `python/sglang/test/ascend/disaggregation_utils.py` added +153/-0
  - `python/sglang/test/ascend/test_ascend_utils.py` modified +495/-54
  - `test/registered/ascend/basic_function/HiCache/test_npu_hierarchical_cache.py` added +127/-0
- Key code excerpts:

```diff
diff -- .github/workflows/nightly-test-npu.yml
@@ -50,7 +50,6 @@ jobs:
           pip config set global.index-url http://${CACHING_URL}/pypi/simple
           pip config set global.extra-index-url "https://pypi.tuna.tsinghua.edu.cn/simple"
           pip config set global.trusted-host "${CACHING_URL} pypi.tuna.tsinghua.edu.cn"
-
           bash scripts/ci/npu/npu_ci_install_dependency.sh a3
           # copy required file from our daily cache
           cp ~/.cache/modelscope/hub/datasets/otavia/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json /tmp
@@ -71,7 +70,18 @@ jobs:
           PYTORCH_NPU_ALLOC_CONF: "expandable_segments:True"
           STREAMS_PER_DEVICE: 32
         run: |
-          pip install sentence_transformers accelerate
+          pip install sglang_router
diff -- python/sglang/test/ascend/disaggregation_utils.py
@@ -0,0 +1,153 @@
+import logging
+import os
+import time
+import warnings
+from urllib.parse import urlparse
+
+import requests
+
+from sglang.srt.environ import envs
+from sglang.srt.utils import kill_process_tree
+from sglang.test.test_utils import (
+    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
+    DEFAULT_URL_FOR_TEST,
```
- Reviewed files:
  - runtime: `python/sglang/test/ascend/disaggregation_utils.py` added +153/-0; `python/sglang/test/ascend/test_ascend_utils.py` modified +495/-54
  - tests: `test/registered/ascend/basic_function/HiCache/test_npu_hierarchical_cache.py` added +127/-0; `test/registered/ascend/basic_function/HiCache/test_npu_hierarchical_cache_mla.py` added +97/-0; `test/registered/ascend/basic_function/HiCache/test_npu_hierarchical_cache_mutually_exclusive.py` added +68/-0; `test/registered/ascend/basic_function/HiCache/test_npu_hierarchical_cache_ttft_mha.py` added +89/-0; `test/registered/ascend/basic_function/HiCache/test_npu_radix_cache.py` added +132/-0; `test/registered/ascend/basic_function/parallel_strategy/expert_parallelism/test_npu_deepep_auto_deepseek_v3_2_w8a8.py` added +108/-0; `test/registered/ascend/basic_function/parallel_strategy/expert_parallelism/test_npu_deepep_auto_qwen3_480b.py` added +128/-0; `test/registered/ascend/basic_function/parallel_strategy/expert_parallelism/test_npu_deepep_auto_qwen3_next.py` added +118/-0
  - other: `.github/workflows/nightly-test-npu.yml` modified +37/-17
- Risk and verification: The diff includes test or benchmark paths; rerun those checks plus a minimal launch/accuracy smoke before changing this model again.

### PR #17784 - Upgrade transformers==5.3.0

- Link: https://github.com/sgl-project/sglang/pull/17784
- Status/date: merged / 2026-03-18
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 95 files, +1136/-343, with 2752 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Upgrade transformers==5.3.0"; model line: Ling 2.5 1T; category: docs/tests/CI; main diff: `docs/advanced_features/vlm_query.ipynb`, `python/pyproject.toml`, `python/pyproject_cpu.toml`.
- Key implementation:
  - `docs/advanced_features/vlm_query.ipynb` modified +3/-3
  - `python/pyproject.toml` modified +4/-4
  - `python/pyproject_cpu.toml` modified +3/-4
  - `python/pyproject_npu.toml` modified +3/-4
- Code diff details:
  - `docs/advanced_features/vlm_query.ipynb` modified +3/-3
  - `python/pyproject.toml` modified +4/-4
  - `python/pyproject_cpu.toml` modified +3/-4
  - `python/pyproject_npu.toml` modified +3/-4
- Key code excerpts:

```diff
diff -- docs/advanced_features/vlm_query.ipynb
@@ -182,9 +182,8 @@
     "from transformers import Qwen2_5_VLForConditionalGeneration\n",
     "\n",
     "processor = AutoProcessor.from_pretrained(model_path, use_fast=True)\n",
-    "vision = (\n",
-    "    Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path).eval().visual.cuda()\n",
-    ")"
+    "model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path).eval()\n",
+    "vision = model.model.visual.cuda()"
    ]
   },
   {
@@ -203,6 +202,7 @@
     "precomputed_embeddings = vision(\n",
diff -- python/pyproject.toml
@@ -30,8 +30,6 @@ dependencies = [
   "flashinfer_python==0.6.6", # keep it aligned with jit-cache version in Dockerfile
   "flashinfer_cubin==0.6.6",
   "gguf",
-  "hf_transfer",
-  "huggingface_hub",
   "interegular",
   "llguidance>=0.7.11,<0.8.0",
   "modelscope",
@@ -72,7 +70,8 @@ dependencies = [
   "av ; sys_platform == 'linux' and (platform_machine == 'aarch64' or platform_machine == 'arm64' or platform_machine == 'armv7l')",
   "torchvision",
   "tqdm",
-  "transformers==4.57.1",
```
- Reviewed files:
  - runtime: `python/pyproject.toml` modified +4/-4; `python/pyproject_cpu.toml` modified +3/-4; `python/pyproject_npu.toml` modified +3/-4; `python/pyproject_other.toml` modified +3/-4; `python/pyproject_xpu.toml` modified +3/-4; `python/sglang/check_env.py` modified +0/-1; `python/sglang/multimodal_gen/runtime/loader/weight_utils.py` modified +0/-16; `python/sglang/multimodal_gen/runtime/models/encoders/llama.py` modified +2/-2
  - docs/bench: `docs/advanced_features/vlm_query.ipynb` modified +3/-3
- Risk and verification: The diff includes test or benchmark paths; rerun those checks plus a minimal launch/accuracy smoke before changing this model again.

### PR #9744 - [CPU] Add FP8 Bmm support

- Link: https://github.com/sgl-project/sglang/pull/9744
- Status/date: merged / 2026-03-19
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 14 files, +585/-84, with 1014 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CPU] Add FP8 Bmm support"; model line: Ling 2.5 1T; category: model support/runtime entry; main diff: `python/sglang/srt/models/bailing_moe_linear.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_cpu.py`.
- Key implementation:
  - `python/sglang/srt/models/bailing_moe_linear.py` modified +0/-8
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +44/-28
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_cpu.py` modified +2/-1
  - `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +0/-10
- Code diff details:
  - `python/sglang/srt/models/bailing_moe_linear.py` modified +0/-8
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +44/-28
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_cpu.py` modified +2/-1
  - `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +0/-10
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/bailing_moe_linear.py
@@ -1208,14 +1208,6 @@ def post_load_weights(self, is_nextn=False, weight_names=None):
                     )
                     if _is_hip:
                         self_attn.w_scale *= 2.0
-                # TODO: remove this after adding FP8 support in bmm cpu kernel
-                if _is_cpu and _is_cpu_amx_available and w.dtype == torch.float8_e4m3fn:
-                    self_attn.w_kc = (
-                        self_attn.w_kc.to(torch.bfloat16) * self_attn.w_scale
-                    )
-                    self_attn.w_vc = (
-                        self_attn.w_vc.to(torch.bfloat16) * self_attn.w_scale
-                    )
             else:
                 num_tiles_k = self_attn.qk_nope_head_dim // weight_block_size[1]
diff -- python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py
@@ -16,6 +16,7 @@
 from sglang.srt.model_executor.forward_batch_info import ForwardBatch
 from sglang.srt.models.deepseek_common.utils import (
     FORWARD_ABSORB_CORE_ATTENTION_BACKENDS,
+    _is_cpu,
     _is_cublas_ge_129,
     _is_cuda,
     _is_gfx95_supported,
@@ -268,18 +269,24 @@ def forward_absorb_prepare(
                     )

         elif self.w_kc.dtype == torch.float8_e4m3fn:
-            # fix bmm_fp8 error under cublas12.9 caused by bumpallocator, detail in pr#11612
-            q_nope_val, q_nope_scale = per_tensor_quant_mla_fp8(
```
- Reviewed files:
  - runtime: `python/sglang/srt/models/bailing_moe_linear.py` modified +0/-8; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +44/-28; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_cpu.py` modified +2/-1; `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +0/-10; `python/sglang/srt/models/longcat_flash.py` modified +0/-12; `python/sglang/srt/models/longcat_flash_nextn.py` modified +0/-4; `sgl-kernel/csrc/cpu/bmm.cpp` modified +93/-11; `sgl-kernel/csrc/cpu/gemm.h` modified +18/-0
  - tests: `test/srt/cpu/test_bmm.py` added +95/-0; `test/srt/cpu/test_qkv_proj_with_rope.py` modified +13/-4; `test/srt/run_suite.py` modified +1/-0
- Risk and verification: The diff includes test or benchmark paths; rerun those checks plus a minimal launch/accuracy smoke before changing this model again.

### PR #20316 - fix fused_set_kv_buffer for rope with Ling-v2

- Link: https://github.com/sgl-project/sglang/pull/20316
- Status/date: merged / 2026-03-23
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 1 files, +6/-2, with 29 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix fused_set_kv_buffer for rope with Ling-v2"; model line: Ling 2.5 1T; category: performance/backend optimization; main diff: `python/sglang/srt/models/bailing_moe.py`.
- Key implementation:
  - `python/sglang/srt/models/bailing_moe.py` modified +6/-2
- Code diff details:
  - `python/sglang/srt/models/bailing_moe.py` modified +6/-2
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/bailing_moe.py
@@ -532,6 +532,10 @@ def forward(
                 head_dim=self.head_dim,
                 alt_stream=self.alt_stream,
             )
+        can_fuse_set_kv = (
+            self.head_dim == self.rotary_emb.rotary_dim
+            and enable_fused_set_kv_buffer(forward_batch)
+        )
         q, k = self.rotary_emb(
             positions,
             q,
@@ -542,7 +546,7 @@ def forward(
                     layer=self.attn,
                     forward_batch=forward_batch,
```
- Reviewed files:
  - runtime: `python/sglang/srt/models/bailing_moe.py` modified +6/-2
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.

### PR #20751 - [NPU]Add a full test pipeline on NPU, resolve issues in the NPU test architecture

- Link: https://github.com/sgl-project/sglang/pull/20751
- Status/date: merged / 2026-04-01
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 43 files, +673/-106, with 1465 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NPU]Add a full test pipeline on NPU, resolve issues in the NPU test architecture"; model line: Ling 2.5 1T; category: model support/runtime entry; main diff: `.github/workflows/full-test-npu.yml`, `.github/workflows/nightly-test-npu.yml`, `.github/workflows/pr-test-npu.yml`.
- Key implementation:
  - `.github/workflows/full-test-npu.yml` added +355/-0
  - `.github/workflows/nightly-test-npu.yml` modified +124/-36
  - `.github/workflows/pr-test-npu.yml` modified +70/-40
  - `.github/workflows/release-docker-npu-nightly.yml` modified +1/-1
- Code diff details:
  - `.github/workflows/full-test-npu.yml` added +355/-0
  - `.github/workflows/nightly-test-npu.yml` modified +124/-36
  - `.github/workflows/pr-test-npu.yml` modified +70/-40
  - `.github/workflows/release-docker-npu-nightly.yml` modified +1/-1
- Key code excerpts:

```diff
diff -- .github/workflows/full-test-npu.yml
@@ -0,0 +1,355 @@
+name: Full Test (NPU)
+
+on:
+#  pull_request:
+#    branches:
+#      - main
+#    paths:
+#      - ".github/workflows/full-test-npu.yml"
+  workflow_dispatch:
+    inputs:
+      ref:
+        description: 'Git ref (branch, tag, or SHA) to test. If not provided, uses the default branch.'
+        required: false
diff -- .github/workflows/nightly-test-npu.yml
@@ -2,7 +2,7 @@ name: Nightly Test (NPU)

 on:
   schedule:
-    - cron: '0 17 * * *'  # Execute at 1:00 a.m. Beijing Time every day
+    - cron: '0 18 * * *'  # Execute at 2:00 a.m. Beijing Time every day
   pull_request:
     branches:
       - main
@@ -21,40 +21,95 @@ on:
         required: false
         type: string
         default: 'all'
+      image_a3:
```
- Reviewed files:
  - runtime: `python/pyproject_npu.toml` modified +2/-0; `python/sglang/test/ascend/test_ascend_utils.py` modified +9/-9
  - tests: `test/registered/ascend/basic_function/HiCache/test_npu_hicache_mha.py` renamed +4/-0; `test/registered/ascend/basic_function/HiCache/test_npu_hicache_mla.py` renamed +4/-0; `test/registered/ascend/basic_function/backends/test_npu_sampling_backend.py` renamed +4/-0; `test/registered/ascend/basic_function/dllm/test_npu_llada2_mini.py` renamed +4/-0; `test/registered/ascend/basic_function/optimization_debug/test_npu_compile_graph_tp1_bf16.py` renamed +4/-0; `test/registered/ascend/basic_function/optimization_debug/test_npu_graph_tp1_bf16.py` renamed +4/-0; `test/registered/ascend/basic_function/optimization_debug/test_npu_graph_tp2_bf16.py` renamed +4/-0; `test/registered/ascend/basic_function/optimization_debug/test_npu_piecewise_graph_prefill.py` renamed +4/-0
  - other: `.github/workflows/full-test-npu.yml` added +355/-0; `.github/workflows/nightly-test-npu.yml` modified +124/-36; `.github/workflows/pr-test-npu.yml` modified +70/-40; `.github/workflows/release-docker-npu-nightly.yml` modified +1/-1
- Risk and verification: The diff includes test or benchmark paths; rerun those checks plus a minimal launch/accuracy smoke before changing this model again.

### PR #23001 - Add new Mintlify documentation site (docs_new/)

- Link: https://github.com/sgl-project/sglang/pull/23001
- Status/date: merged / 2026-04-20
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 100 files, +19458/-0, with 19508 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add new Mintlify documentation site (docs_new/)"; model line: Ling 2.5 1T; category: model support/runtime entry; main diff: `.gitignore`, `docs_new/.github/workflows/sync-lmsys-sglang-blogs.yml`, `docs_new/.gitignore`.
- Key implementation:
  - `.gitignore` modified +1/-0
  - `docs_new/.github/workflows/sync-lmsys-sglang-blogs.yml` added +39/-0
  - `docs_new/.gitignore` added +30/-0
  - `docs_new/.mintignore` added +7/-0
- Code diff details:
  - `.gitignore` modified +1/-0
  - `docs_new/.github/workflows/sync-lmsys-sglang-blogs.yml` added +39/-0
  - `docs_new/.gitignore` added +30/-0
  - `docs_new/.mintignore` added +7/-0
- Key code excerpts:

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
- Reviewed files:
  - docs/bench: `docs_new/.github/workflows/sync-lmsys-sglang-blogs.yml` added +39/-0; `docs_new/.gitignore` added +30/-0; `docs_new/.mintignore` added +7/-0; `docs_new/AGENTS.md` added +381/-0; `docs_new/CONTRIBUTING.md` added +34/-0; `docs_new/LICENSE` added +201/-0; `docs_new/README.md` added +126/-0; `docs_new/cards/Autoregressive-benchmark-card.png` added +0/-0
  - other: `.gitignore` modified +1/-0
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.

### PR #23337 - [Docs] Sync docs_new with legacy docs and update migration redirects

- Link: https://github.com/sgl-project/sglang/pull/23337
- Status/date: merged / 2026-04-21
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 100 files, +3881/-1454, with 7195 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Docs] Sync docs_new with legacy docs and update migration redirects"; model line: Ling 2.5 1T; category: docs/tests/CI; main diff: `.pre-commit-config.yaml`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-Math-V2.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR-2.mdx`.
- Key implementation:
  - `.pre-commit-config.yaml` modified +7/-0
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-Math-V2.mdx` modified +1/-1
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR-2.mdx` modified +1/-1
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR.mdx` modified +1/-1
- Code diff details:
  - `.pre-commit-config.yaml` modified +7/-0
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-Math-V2.mdx` modified +1/-1
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR-2.mdx` modified +1/-1
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR.mdx` modified +1/-1
- Key code excerpts:

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
- Reviewed files:
  - docs/bench: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-Math-V2.mdx` modified +1/-1; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR-2.mdx` modified +1/-1; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR.mdx` modified +1/-1; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-R1.mdx` modified +1/-1; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3.mdx` modified +1/-1; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_1.mdx` modified +1/-1; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx` modified +1/-1; `docs_new/cookbook/autoregressive/Ernie/Ernie4.5.mdx` modified +1/-1
  - other: `.pre-commit-config.yaml` modified +7/-0
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.

### PR #23732 - Apply should_use_dp_reduce_scatterv guard to remaining MoE models (follow-up to #23731)

- Link: https://github.com/sgl-project/sglang/pull/23732
- Status/date: merged / 2026-04-26
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 13 files, +59/-12, with 290 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Apply should_use_dp_reduce_scatterv guard to remaining MoE models (follow-up to #23731)"; model line: Ling 2.5 1T; category: model implementation change; main diff: `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/bailing_moe_linear.py`, `python/sglang/srt/models/deepseek_v2.py`.
- Key implementation:
  - `python/sglang/srt/models/bailing_moe.py` modified +2/-0
  - `python/sglang/srt/models/bailing_moe_linear.py` modified +7/-1
  - `python/sglang/srt/models/deepseek_v2.py` modified +3/-0
  - `python/sglang/srt/models/exaone_moe.py` modified +6/-2
- Code diff details:
  - `python/sglang/srt/models/bailing_moe.py` modified +2/-0
  - `python/sglang/srt/models/bailing_moe_linear.py` modified +7/-1
  - `python/sglang/srt/models/deepseek_v2.py` modified +3/-0
  - `python/sglang/srt/models/exaone_moe.py` modified +6/-2
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/bailing_moe.py
@@ -58,6 +58,7 @@
 from sglang.srt.layers.moe import (
     get_deepep_mode,
     get_moe_a2a_backend,
+    should_use_dp_reduce_scatterv,
     should_use_flashinfer_cutlass_moe_fp4_allgather,
 )
 from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
@@ -386,6 +387,7 @@ def forward_normal(
             and not should_allreduce_fusion
             and not use_reduce_scatter
             and not should_use_flashinfer_cutlass_moe_fp4_allgather()
+            and not should_use_dp_reduce_scatterv()
         ):
diff -- python/sglang/srt/models/bailing_moe_linear.py
@@ -34,6 +34,7 @@
     RowParallelLinear,
 )
 from sglang.srt.layers.logits_processor import LogitsProcessor
+from sglang.srt.layers.moe import should_use_dp_reduce_scatterv
 from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE, get_moe_impl_class
 from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
 from sglang.srt.layers.moe.topk import TopK
@@ -347,7 +348,12 @@ def forward(
         if self.num_shared_experts > 0:
             final_hidden_states = final_hidden_states + shared_output

-        if self.tp_size > 1 and not use_reduce_scatter and not should_allreduce_fusion:
+        if (
```
- Reviewed files:
  - runtime: `python/sglang/srt/models/bailing_moe.py` modified +2/-0; `python/sglang/srt/models/bailing_moe_linear.py` modified +7/-1; `python/sglang/srt/models/deepseek_v2.py` modified +3/-0; `python/sglang/srt/models/exaone_moe.py` modified +6/-2; `python/sglang/srt/models/glm4_moe.py` modified +3/-0; `python/sglang/srt/models/hunyuan_v3.py` modified +7/-4; `python/sglang/srt/models/llada2.py` modified +10/-2; `python/sglang/srt/models/llama4.py` modified +6/-1
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.

### PR #23748 - refactor(moe): centralize post-experts all-reduce skip predicate

- Link: https://github.com/sgl-project/sglang/pull/23748
- Status/date: merged / 2026-04-27
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 17 files, +134/-132, with 532 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "refactor(moe): centralize post-experts all-reduce skip predicate"; model line: Ling 2.5 1T; category: model implementation change; main diff: `python/sglang/srt/layers/moe/__init__.py`, `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/models/bailing_moe.py`.
- Key implementation:
  - `python/sglang/srt/layers/moe/__init__.py` modified +2/-0
  - `python/sglang/srt/layers/moe/utils.py` modified +33/-0; symbols: should_skip_post_experts_all_reduce
  - `python/sglang/srt/models/bailing_moe.py` modified +5/-8
  - `python/sglang/srt/models/bailing_moe_linear.py` modified +5/-6
- Code diff details:
  - `python/sglang/srt/layers/moe/__init__.py` modified +2/-0
  - `python/sglang/srt/layers/moe/utils.py` modified +33/-0
  - `python/sglang/srt/models/bailing_moe.py` modified +5/-8
  - `python/sglang/srt/models/bailing_moe_linear.py` modified +5/-6
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/__init__.py
@@ -10,6 +10,7 @@
     get_tbo_token_distribution_threshold,
     initialize_moe_config,
     is_tbo_enabled,
+    should_skip_post_experts_all_reduce,
     should_use_dp_reduce_scatterv,
     should_use_flashinfer_cutlass_moe_fp4_allgather,
 )
@@ -24,6 +25,7 @@
     "get_moe_a2a_backend",
     "get_moe_runner_backend",
     "get_deepep_mode",
+    "should_skip_post_experts_all_reduce",
     "should_use_dp_reduce_scatterv",
diff -- python/sglang/srt/layers/moe/utils.py
@@ -346,6 +346,39 @@ def should_use_dp_reduce_scatterv():
     )


+def should_skip_post_experts_all_reduce(
+    *,
+    is_tp_path: bool,
+    use_reduce_scatter: bool = False,
+    should_allreduce_fusion: bool = False,
+) -> bool:
+    """Whether to skip the post-experts all-reduce (EP or TP) because a
+    downstream component will fuse, replace, or absorb it.
+
+    Skip reasons, in order:
```
- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/__init__.py` modified +2/-0; `python/sglang/srt/layers/moe/utils.py` modified +33/-0; `python/sglang/srt/models/bailing_moe.py` modified +5/-8; `python/sglang/srt/models/bailing_moe_linear.py` modified +5/-6; `python/sglang/srt/models/deepseek_v2.py` modified +9/-13; `python/sglang/srt/models/exaone_moe.py` modified +7/-5; `python/sglang/srt/models/glm4_moe.py` modified +9/-13; `python/sglang/srt/models/hunyuan_v3.py` modified +13/-7
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.

### PR #21126 - [4/N] Quantization Refactor: AWQ schemes and Kernel call and weight init split

- Link: https://github.com/sgl-project/sglang/pull/21126
- Status/date: merged / 2026-04-30
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 20 files, +1419/-1031, with 2590 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[4/N] Quantization Refactor: AWQ schemes and Kernel call and weight init split"; model line: Ling 2.5 1T; category: performance/backend optimization; main diff: `python/sglang/srt/hardware_backend/gpu/quantization/awq_kernels.py`, `python/sglang/srt/hardware_backend/npu/quantization/awq_kernels.py`, `python/sglang/srt/layers/linear.py`.
- Key implementation:
  - `python/sglang/srt/hardware_backend/gpu/quantization/awq_kernels.py` added +255/-0; symbols: _unsupported_awq_dequantize, AWQLinearKernel, __init__, process_weights_after_loading
  - `python/sglang/srt/hardware_backend/npu/quantization/awq_kernels.py` added +156/-0; symbols: AWQAscendLinearKernel, __init__, process_weights_after_loading, apply
  - `python/sglang/srt/layers/linear.py` modified +0/-3
  - `python/sglang/srt/layers/quantization/__init__.py` modified +2/-3
- Code diff details:
  - `python/sglang/srt/hardware_backend/gpu/quantization/awq_kernels.py` added +255/-0
  - `python/sglang/srt/hardware_backend/npu/quantization/awq_kernels.py` added +156/-0
  - `python/sglang/srt/layers/linear.py` modified +0/-3
  - `python/sglang/srt/layers/quantization/__init__.py` modified +2/-3
- Key code excerpts:

```diff
diff -- python/sglang/srt/hardware_backend/gpu/quantization/awq_kernels.py
@@ -0,0 +1,255 @@
+from __future__ import annotations
+
+from typing import TYPE_CHECKING, Optional
+
+import torch
+
+from sglang.srt.layers.moe import MoeRunner
+from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo
+from sglang.srt.layers.quantization.marlin_utils import (
+    apply_awq_marlin_linear,
+    awq_to_marlin_zero_points,
+    marlin_make_empty_g_idx,
+    marlin_make_workspace,
diff -- python/sglang/srt/hardware_backend/npu/quantization/awq_kernels.py
@@ -0,0 +1,156 @@
+from __future__ import annotations
+
+from typing import TYPE_CHECKING, Optional
+
+import torch
+
+from sglang.srt.hardware_backend.npu.quantization.fused_moe_method_npu import (
+    NPUW4A16Int4DynamicMoEMethod,
+)
+from sglang.srt.layers.quantization.utils import replace_parameter
+
+if TYPE_CHECKING:
+    from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput
```
- Reviewed files:
  - runtime: `python/sglang/srt/hardware_backend/gpu/quantization/awq_kernels.py` added +255/-0; `python/sglang/srt/hardware_backend/npu/quantization/awq_kernels.py` added +156/-0; `python/sglang/srt/layers/linear.py` modified +0/-3; `python/sglang/srt/layers/quantization/__init__.py` modified +2/-3; `python/sglang/srt/layers/quantization/auto_round.py` modified +5/-2; `python/sglang/srt/layers/quantization/awq.py` removed +0/-966; `python/sglang/srt/layers/quantization/awq/__init__.py` added +32/-0; `python/sglang/srt/layers/quantization/awq/awq.py` added +484/-0
  - tests: `test/registered/quant/test_awq_dequant.py` modified +1/-1
- Risk and verification: The diff includes test or benchmark paths; rerun those checks plus a minimal launch/accuracy smoke before changing this model again.

### PR #24250 - [SKILL] Upgrade sglang profile and auto_benchmark skills

- Link: https://github.com/sgl-project/sglang/pull/24250
- Status/date: merged / 2026-05-02
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 100 files, +9334/-3813, with 13573 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[SKILL] Upgrade sglang profile and auto_benchmark skills"; model line: Ling 2.5 1T; category: docs/tests/CI; main diff: `agent-skills/llm-serving-auto-benchmark/SKILL.md`, `skills/llm-serving-auto-benchmark/configs/cookbook-llm/README.md`, `skills/llm-serving-auto-benchmark/configs/cookbook-llm/deepseek-math-v2.yaml`.
- Key implementation:
  - `agent-skills/llm-serving-auto-benchmark/SKILL.md` added +527/-0
  - `skills/llm-serving-auto-benchmark/configs/cookbook-llm/README.md` added +17/-0
  - `skills/llm-serving-auto-benchmark/configs/cookbook-llm/deepseek-math-v2.yaml` added +130/-0
  - `skills/llm-serving-auto-benchmark/configs/cookbook-llm/deepseek-r1-0528.yaml` added +133/-0
- Code diff details:
  - `agent-skills/llm-serving-auto-benchmark/SKILL.md` added +527/-0
  - `skills/llm-serving-auto-benchmark/configs/cookbook-llm/README.md` added +17/-0
  - `skills/llm-serving-auto-benchmark/configs/cookbook-llm/deepseek-math-v2.yaml` added +130/-0
  - `skills/llm-serving-auto-benchmark/configs/cookbook-llm/deepseek-r1-0528.yaml` added +133/-0
- Key code excerpts:

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
- Reviewed files:
  - docs/bench: `skills/llm-serving-auto-benchmark/configs/cookbook-llm/README.md` added +17/-0; `skills/llm-serving-auto-benchmark/configs/cookbook-llm/deepseek-math-v2.yaml` added +130/-0; `skills/llm-serving-auto-benchmark/configs/cookbook-llm/deepseek-r1-0528.yaml` added +133/-0; `skills/llm-serving-auto-benchmark/configs/cookbook-llm/deepseek-v3.1.yaml` added +132/-0; `skills/llm-serving-auto-benchmark/configs/cookbook-llm/deepseek-v3.2.yaml` added +132/-0; `skills/llm-serving-auto-benchmark/configs/cookbook-llm/deepseek-v3.yaml` added +133/-0; `skills/llm-serving-auto-benchmark/configs/cookbook-llm/devstral-small-2-24b-instruct-2512.yaml` added +123/-0; `skills/llm-serving-auto-benchmark/configs/cookbook-llm/ernie-4.5-21b-a3b-pt.yaml` added +117/-0
  - other: `agent-skills/llm-serving-auto-benchmark/SKILL.md` added +527/-0
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.

### PR #24333 - nextn subclass owns post_load_weights is_nextn

- Link: https://github.com/sgl-project/sglang/pull/24333
- Status/date: merged / 2026-05-04
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 4 files, +30/-25, with 138 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "nextn subclass owns post_load_weights is_nextn"; model line: Ling 2.5 1T; category: model implementation change; main diff: `python/sglang/srt/model_loader/loader.py`, `python/sglang/srt/model_loader/utils.py`, `python/sglang/srt/models/bailing_moe_nextn.py`.
- Key implementation:
  - `python/sglang/srt/model_loader/loader.py` modified +15/-10; symbols: _post_load_weights
  - `python/sglang/srt/model_loader/utils.py` modified +0/-12
  - `python/sglang/srt/models/bailing_moe_nextn.py` modified +9/-3; symbols: post_load_weights
  - `python/sglang/srt/models/deepseek_nextn.py` modified +6/-0; symbols: post_load_weights
- Code diff details:
  - `python/sglang/srt/model_loader/loader.py` modified +15/-10
  - `python/sglang/srt/model_loader/utils.py` modified +0/-12
  - `python/sglang/srt/models/bailing_moe_nextn.py` modified +9/-3
  - `python/sglang/srt/models/deepseek_nextn.py` modified +6/-0
- Key code excerpts:

```diff
diff -- python/sglang/srt/model_loader/loader.py
@@ -78,7 +78,6 @@
 )
 from sglang.srt.model_loader.utils import (
     get_model_architecture,
-    post_load_weights,
     set_default_torch_dtype,
 )

@@ -286,6 +285,15 @@ def _initialize_model(
     return model_class(**kwargs)


+def _post_load_weights(model: nn.Module) -> None:
+    # Loaders that bypass `model.load_weights()` (dummy / sharded state / remote instance /
diff -- python/sglang/srt/model_loader/utils.py
@@ -247,18 +247,6 @@ def get_architecture_class_name(model_config: ModelConfig) -> str:
     return get_model_architecture(model_config)[1]


-def post_load_weights(model: nn.Module, model_config: ModelConfig):
-    # Model weight loading consists of two stages:
-    # 1. Initial weight loading.
-    # 2. Post-processing of weights, including assigning specific member variables.
-    # For `dummy_init`, only the second stage is required.
-    if hasattr(model, "post_load_weights"):
-        if model_config.hf_config.architectures[0] == "DeepseekV3ForCausalLMNextN":
-            model.post_load_weights(is_nextn=True)
-        else:
-            model.post_load_weights()
```
- Reviewed files:
  - runtime: `python/sglang/srt/model_loader/loader.py` modified +15/-10; `python/sglang/srt/model_loader/utils.py` modified +0/-12; `python/sglang/srt/models/bailing_moe_nextn.py` modified +9/-3; `python/sglang/srt/models/deepseek_nextn.py` modified +6/-0
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.

### PR #24977 - fix gb envs in deployment guide

- Link: https://github.com/sgl-project/sglang/pull/24977
- Status/date: merged / 2026-05-11
- Trace source: `git log --name-only -- <model-files>` or model-keyword supplement; this card was audited through the GitHub Pull Request files API.
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, with 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix gb envs in deployment guide"; model line: Ling 2.5 1T; category: bug fix; main diff: `docs_new/src/snippets/autoregressive/ling-25-1t-deployment.jsx`.
- Key implementation:
  - `docs_new/src/snippets/autoregressive/ling-25-1t-deployment.jsx` modified +1/-1; symbols: envPrefix
- Code diff details:
  - `docs_new/src/snippets/autoregressive/ling-25-1t-deployment.jsx` modified +1/-1
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/autoregressive/ling-25-1t-deployment.jsx
@@ -81,7 +81,7 @@ export const Ling251TDeployment = () => {
     const { hardware, parallelism, toolcall } = values;

     const isGB = hardware === 'gb200' || hardware === 'gb300';
-    const envPrefix = isGB ? 'NCCL_IB_DISABLE=1 ' : '';
+    const envPrefix = isGB ? 'NCCL_MNNVL_ENABLE=1 NCCL_CUMEM_ENABLE=1 ' : '';

     let tp, pp;
     if (isGB && parallelism === 'tp8') {
```
- Reviewed files:
  - docs/bench: `docs_new/src/snippets/autoregressive/ling-25-1t-deployment.jsx` modified +1/-1
- Risk and verification: The diff does not expose direct test files; future work should add a minimal launch, tokenizer/MM processor, or accuracy smoke.
