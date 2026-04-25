# sglang Qwen3 Next Model PR Optimization History

## Scope

- Rebuilt on: 2026-04-25
- Source baseline: `sgl-project/sglang` trace worktree commit `880599cd43`
- PR collection rule: run `git log --name-only -- <model-files>` on model implementation, config, processor, parser, docs/tests, filter by model keywords in commit subjects, then read each PR's final diff through the GitHub Pull Request files API.
- Preservation rule: PRs explicitly cited by the previous history/skill are retained even if current implementation files no longer trace to them, and the card marks that source.
- Diffusion model families have been removed from this history set and are no longer part of model optimization skills.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `python/sglang/srt/configs/qwen3_next.py` | [#10233](https://github.com/sgl-project/sglang/pull/10233), [#11585](https://github.com/sgl-project/sglang/pull/11585), [#12525](https://github.com/sgl-project/sglang/pull/12525) |
| `python/sglang/srt/models/qwen3_next.py` | [#10233](https://github.com/sgl-project/sglang/pull/10233), [#10322](https://github.com/sgl-project/sglang/pull/10322), [#10379](https://github.com/sgl-project/sglang/pull/10379), [#10622](https://github.com/sgl-project/sglang/pull/10622), [#11969](https://github.com/sgl-project/sglang/pull/11969), [#12525](https://github.com/sgl-project/sglang/pull/12525), [#13081](https://github.com/sgl-project/sglang/pull/13081), [#14607](https://github.com/sgl-project/sglang/pull/14607), [#16164](https://github.com/sgl-project/sglang/pull/16164), [#17016](https://github.com/sgl-project/sglang/pull/17016), [#17373](https://github.com/sgl-project/sglang/pull/17373), [#17613](https://github.com/sgl-project/sglang/pull/17613), ... (19 total) |
| `python/sglang/srt/models/qwen3_next_mtp.py` | [#10233](https://github.com/sgl-project/sglang/pull/10233), [#10322](https://github.com/sgl-project/sglang/pull/10322), [#10392](https://github.com/sgl-project/sglang/pull/10392) |
| `test/registered/4-gpu-models/test_qwen3_next_models.py` | no direct PR-number commit |
| `test/registered/4-gpu-models/test_qwen3_next_models_mtp.py` | no direct PR-number commit |
| `test/registered/ascend/basic_function/parallel_strategy/expert_parallelism/test_npu_deepep_auto_qwen3_next.py` | no direct PR-number commit |
| `test/registered/ascend/basic_function/parallel_strategy/expert_parallelism/test_npu_deepep_low_latency_qwen3_next.py` | no direct PR-number commit |
| `test/registered/core/test_qwen3_next_deterministic.py` | no direct PR-number commit |
| `test/registered/models/test_qwen3_next_models_fp4.py` | [#17627](https://github.com/sgl-project/sglang/pull/17627) |

## PR Coverage Summary

- Git-traced PRs: 21
- Extra PRs preserved from existing docs: 35
- Total PRs in this document: 56
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2025-09-11 | [#10233](https://github.com/sgl-project/sglang/pull/10233) | merged | Qwen3-Next support | `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/configs/qwen3_next.py`, `python/sglang/srt/models/qwen3_next_mtp.py` |
| 2025-09-11 | [#10322](https://github.com/sgl-project/sglang/pull/10322) | merged | [bugfix] fix norm type error in qwen3_next model | `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/models/qwen3_next_mtp.py` |
| 2025-09-12 | [#10379](https://github.com/sgl-project/sglang/pull/10379) | merged | Support Qwen3-Next on Ascend NPU | `python/sglang/srt/models/qwen3_next.py` |
| 2025-09-13 | [#10392](https://github.com/sgl-project/sglang/pull/10392) | merged | [Fix] Support qwen3-next MTP+DP | `python/sglang/srt/models/qwen3_next_mtp.py` |
| 2025-09-16 | [#10466](https://github.com/sgl-project/sglang/pull/10466) | merged | feat: update support for qwen3next model | `python/sglang/srt/layers/attention/fla/fused_recurrent.py`, `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py` |
| 2025-09-18 | [#10622](https://github.com/sgl-project/sglang/pull/10622) | merged | support qwen3-next-fp8 deepep | `python/sglang/srt/models/qwen3_next.py` |
| 2025-09-19 | [#10657](https://github.com/sgl-project/sglang/pull/10657) | open | feat: add eagle3 support for qwen3-next model | `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/model_executor/model_runner.py` |
| 2025-10-15 | [#11585](https://github.com/sgl-project/sglang/pull/11585) | merged | Clean up some Qwen3-Next and deterministic code | `python/sglang/srt/configs/qwen3_next.py` |
| 2025-10-16 | [#10912](https://github.com/sgl-project/sglang/pull/10912) | merged | [PD] Add PD support for hybrid model (Qwen3-Next, DeepSeek V3.2 Exp) | `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/disaggregation/mooncake/conn.py` |
| 2025-10-21 | [#11487](https://github.com/sgl-project/sglang/pull/11487) | merged | init support for KTransformers Heterogeneous Computing | `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2025-10-30 | [#11969](https://github.com/sgl-project/sglang/pull/11969) | merged | [NPU] bugfix for Qwen3-Next and performance update | `python/sglang/srt/models/qwen3_next.py` |
| 2025-11-06 | [#12508](https://github.com/sgl-project/sglang/pull/12508) | merged | [GDN] Fuse b.sigmoid(), fused_gdn_gating and unsqueeze into one kernel: up to 0.85% e2e speedup | `python/sglang/srt/layers/attention/fla/fused_gdn_gating.py`, `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` |
| 2025-11-08 | [#12892](https://github.com/sgl-project/sglang/pull/12892) | open | [GDN/Qwen3-Next] Avoid SSM and conv state copy for speculative decoding - up to 9.47% e2e speedup | `python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/layers/attention/fla/fused_recurrent.py` |
| 2025-11-21 | [#13708](https://github.com/sgl-project/sglang/pull/13708) | merged | [Fix] Qwen3Next lmhead dtype | `python/sglang/srt/models/qwen3_next.py` |
| 2025-11-25 | [#13081](https://github.com/sgl-project/sglang/pull/13081) | merged | Support piecewise cuda graph for Qwen3-next | `python/sglang/srt/models/qwen3_next.py` |
| 2025-11-26 | [#13964](https://github.com/sgl-project/sglang/pull/13964) | open | [Performance]Qwen3 Next kernel performance optimize | `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py` |
| 2025-12-05 | [#14502](https://github.com/sgl-project/sglang/pull/14502) | open | [Qwen3-Next]Optimize piecewise CUDA graph for Qwen3-Next | `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` |
| 2025-12-13 | [#14855](https://github.com/sgl-project/sglang/pull/14855) | merged | Clean up GDN Init | `python/sglang/srt/models/qwen3_next.py` |
| 2026-01-03 | [#16164](https://github.com/sgl-project/sglang/pull/16164) | merged | [NPU] Adapt qwen3-next W8A8 on NPU | `python/sglang/srt/models/qwen3_next.py` |
| 2026-01-05 | [#16488](https://github.com/sgl-project/sglang/pull/16488) | open | Two-Batch Overlap (TBO) support to Qwen3-Next Models | `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` |
| 2026-01-10 | [#16863](https://github.com/sgl-project/sglang/pull/16863) | merged | tiny refactor pcg split op registration | `python/sglang/srt/layers/radix_attention.py`, `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/compilation/compilation_config.py` |
| 2026-01-16 | [#17016](https://github.com/sgl-project/sglang/pull/17016) | merged | [bugfix] fix qwen3-next alt_stream none issue | `python/sglang/srt/models/qwen3_next.py` |
| 2026-01-18 | [#15631](https://github.com/sgl-project/sglang/pull/15631) | merged | [jit-kernel] Add CuTe DSL GDN Decode Kernel | `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/jit_kernel/cutedsl_gdn.py`, `python/sglang/jit_kernel/tests/test_cutedsl_gdn.py` |
| 2026-01-22 | [#17373](https://github.com/sgl-project/sglang/pull/17373) | merged | refactor Qwen3-Next with a new RadixLinearAttention | `python/sglang/srt/models/qwen3_next.py` |
| 2026-01-24 | [#17570](https://github.com/sgl-project/sglang/pull/17570) | merged | Use attn tp group in embedding for more models | `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/bailing_moe_nextn.py`, `python/sglang/srt/models/falcon_h1.py` |
| 2026-01-27 | [#17660](https://github.com/sgl-project/sglang/pull/17660) | merged | [hybrid-model] clean up and consolidate redundant fields in RadixLinearAttention | `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/layers/radix_linear_attention.py`, `python/sglang/srt/models/kimi_linear.py` |
| 2026-01-30 | [#12525](https://github.com/sgl-project/sglang/pull/12525) | merged | [CPU] Optimize Qwen3-next model on CPU | `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/configs/qwen3_next.py` |
| 2026-01-30 | [#17981](https://github.com/sgl-project/sglang/pull/17981) | open | [Qwen3-Next] Add cutedsl decode/mtp kernel with transposed ssm_state and prefill gluon kernel for blackwell. | `python/sglang/srt/layers/attention/linear/kernels/gdn_cutedsl_transpose.py`, `python/sglang/srt/layers/attention/fla/chunk_delta_h.py`, `python/sglang/srt/layers/attention/linear/gdn_backend.py` |
| 2026-01-30 | [#17983](https://github.com/sgl-project/sglang/pull/17983) | open | [Qwen3-Next] Optimize Prefill Kernel, add GDN Gluon kernel and optimize cumsum kernel | `python/sglang/srt/layers/attention/fla/gluon/chunk_delta_h_gluon.py`, `python/sglang/srt/layers/attention/fla/gluon/wy_fast_gluon.py`, `python/sglang/srt/layers/attention/fla/gluon/chunk_o_gluon.py` |
| 2026-02-01 | [#14607](https://github.com/sgl-project/sglang/pull/14607) | merged | support qwen3-next eagle3 | `python/sglang/srt/models/qwen3_next.py` |
| 2026-02-08 | [#18224](https://github.com/sgl-project/sglang/pull/18224) | merged | [ModelOPT] Support Qwen 3 Next Coder NVFP4 | `python/sglang/srt/models/qwen3_next.py` |
| 2026-02-09 | [#18489](https://github.com/sgl-project/sglang/pull/18489) | merged | [MODEL] Adding Support for Qwen3.5 Models | `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/configs/qwen3_5.py` |
| 2026-02-14 | [#17613](https://github.com/sgl-project/sglang/pull/17613) | merged | [Perf] refactor piecewise cuda graph support of Qwen3-Next | `python/sglang/srt/models/qwen3_next.py` |
| 2026-02-22 | [#18917](https://github.com/sgl-project/sglang/pull/18917) | merged | [Qwen3-Next] Enable fused_qkvzba_split_reshape_cat also for prefill | `python/sglang/srt/models/qwen3_next.py` |
| 2026-02-25 | [#18355](https://github.com/sgl-project/sglang/pull/18355) | merged | [AMD] Support Qwen3-Coder-Next on AMD platform | `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/models/qwen3_next.py` |
| 2026-02-26 | [#19220](https://github.com/sgl-project/sglang/pull/19220) | merged | [PCG] fix piecewise cuda graph for Qwen3.5 | `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/layers/quantization/fp8_utils.py` |
| 2026-02-27 | [#19434](https://github.com/sgl-project/sglang/pull/19434) | merged | [Qwen3-Next] Support gdn fused_rms_norm_gated | `python/sglang/srt/models/qwen3_next.py` |
| 2026-02-28 | [#17627](https://github.com/sgl-project/sglang/pull/17627) | merged | [feat] Support nvfp4 quantized model of Qwen3-Next | `test/registered/models/test_qwen3_next_models_fp4.py`, `python/sglang/srt/models/qwen3_next.py` |
| 2026-03-04 | [#19812](https://github.com/sgl-project/sglang/pull/19812) | open | Fix Qwen3.5/Qwen3Next MTP EPLB compatibility | `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen2_moe.py` |
| 2026-03-09 | [#19767](https://github.com/sgl-project/sglang/pull/19767) | merged | Fix qwen3.5 mtp eplb related issues | `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/models/qwen3_next_mtp.py` |
| 2026-03-12 | [#20397](https://github.com/sgl-project/sglang/pull/20397) | open | [NPU] Qwen3 next Ascend Support MTP | `python/sglang/srt/layers/attention/mamba/mamba_state_scatter_triton.py`, `python/sglang/srt/layers/attention/linear/gdn_backend.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` |
| 2026-03-20 | [#19321](https://github.com/sgl-project/sglang/pull/19321) | merged | [Qwen3-Next] Fuse Qwen3-Next GDN's qkvz_proj and ba_proj | `python/sglang/srt/models/qwen3_next.py` |
| 2026-03-23 | [#21019](https://github.com/sgl-project/sglang/pull/21019) | merged | [Qwen3.5] Fuse split/reshape/cat ops in GDN projection with Triton kernel | `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_next.py`, `python/sglang/jit_kernel/triton/gdn_fused_proj.py` |
| 2026-03-26 | [#21313](https://github.com/sgl-project/sglang/pull/21313) | merged | bugfix for weight loading for qwen3-next | `python/sglang/srt/models/qwen3_next.py` |
| 2026-03-26 | [#21496](https://github.com/sgl-project/sglang/pull/21496) | merged | Revert "bugfix for weight loading for qwen3-next" | `python/sglang/srt/models/qwen3_next.py` |
| 2026-03-30 | [#21662](https://github.com/sgl-project/sglang/pull/21662) | merged | [Fix] Fix weight_loader property assignment for qwen3-next FP8 models | `python/sglang/srt/models/qwen3_next.py` |
| 2026-03-30 | [#21684](https://github.com/sgl-project/sglang/pull/21684) | open | [bugfix] fix Qwen3-next memory leak | `python/sglang/srt/mem_cache/allocator.py`, `python/sglang/srt/mem_cache/memory_pool.py` |
| 2026-03-30 | [#21698](https://github.com/sgl-project/sglang/pull/21698) | open | [npu]fix: qwen3-next w8a8 precision bugs | `python/sglang/srt/models/qwen3_next.py` |
| 2026-04-07 | [#22073](https://github.com/sgl-project/sglang/pull/22073) | merged | [Feature] Adding Qwen3-asr Model Support | `python/sglang/srt/models/qwen3_asr.py`, `python/sglang/srt/configs/qwen3_asr.py`, `python/sglang/srt/multimodal/processors/qwen3_asr.py` |
| 2026-04-09 | [#22358](https://github.com/sgl-project/sglang/pull/22358) | merged | Enable DFLASH support for additional model backends | `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/models/qwen3_next.py` |
| 2026-04-10 | [#22458](https://github.com/sgl-project/sglang/pull/22458) | merged | Fix NCCL AllGather hanging issue for Qwen3 Next MTP | `python/sglang/srt/speculative/eagle_info.py`, `python/sglang/srt/speculative/eagle_info_v2.py` |
| 2026-04-15 | [#22876](https://github.com/sgl-project/sglang/pull/22876) | open | Fix: Raise ValueError when --enable-mixed-chunk and --mamba-scheduler-strategy extra_buffer cause ac | `test/registered/unit/server_args/test_server_args.py`, `python/sglang/srt/server_args.py` |
| 2026-04-17 | [#23075](https://github.com/sgl-project/sglang/pull/23075) | open | [Fix] Mixed chunk query_start_loc and mamba_cache_indices to the prefill-only prefix so that the tracking helpers see a consistent, prefill-only view. | `python/sglang/srt/layers/attention/mamba/mamba2_metadata.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/managers/schedule_batch.py` |
| 2026-04-18 | [#22664](https://github.com/sgl-project/sglang/pull/22664) | merged | Qwen3next flashinfer allreduce auto enable | `python/sglang/srt/server_args.py` |
| 2026-04-20 | [#23273](https://github.com/sgl-project/sglang/pull/23273) | open | [NVIDIA] [GDN] Enable FlashInfer MTP verify on SM100+ (Blackwell) | `python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py`, `python/sglang/srt/server_args.py` |
| 2026-04-22 | [#23474](https://github.com/sgl-project/sglang/pull/23474) | open | [Bugfix] Try to fix --cpu-offload-gb on hybrid linear-attn models | `test/registered/unit/utils/test_offloader_tied_params.py`, `python/sglang/srt/utils/offloader.py` |

## Per-PR Diff Audit Cards

### PR #10233 - Qwen3-Next support

- Link: https://github.com/sgl-project/sglang/pull/10233
- Status/date: merged / 2025-09-11
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/configs/qwen3_next.py`, `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/models/qwen3_next_mtp.py`; associated commits `30c6e1f56967`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 19 files, +3224/-8, 3452 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR adds or enables a model support/runtime surface. Title: "Qwen3-Next support". The diff centers on `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/configs/qwen3_next.py`, `python/sglang/srt/models/qwen3_next_mtp.py`. PR body context: ## Motivation ref #10306 support qwen3-next/qwen3-next-mtp ## Modifications 1. add `MambaPool` / `HybridReqTokenPool` to allocate mamba cache 2. add `HybridLinearKVPool` to avoi...
- Key implementation: `python/sglang/srt/models/qwen3_next.py` added +1072/-0 (1072 lines); hunks: -0,0 +1,1072; symbols: fused_qkvzba_split_reshape_cat_kernel, fused_qkvzba_split_reshape_cat, fused_gdn_gating_kernel, fused_gdn_gating, touching `fused_qkvzba_split_reshape_cat_kernel, fused_qkvzba_split_reshape_cat, fused_gdn_gating_kernel`; `python/sglang/srt/configs/qwen3_next.py` added +326/-0 (326 lines); hunks: -0,0 +1,326; symbols: HybridLayerType, Qwen3NextConfig, to, __init__, touching `HybridLayerType, Qwen3NextConfig, to`; `python/sglang/srt/models/qwen3_next_mtp.py` added +117/-0 (117 lines); hunks: -0,0 +1,117; symbols: Qwen3NextForCausalLMMTP, __init__, forward, load_weights, touching `Qwen3NextForCausalLMMTP, __init__, forward`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_next.py` added +1072/-0 (1072 lines); hunks: -0,0 +1,1072; symbols: fused_qkvzba_split_reshape_cat_kernel, fused_qkvzba_split_reshape_cat, fused_gdn_gating_kernel, fused_gdn_gating
  - `python/sglang/srt/configs/qwen3_next.py` added +326/-0 (326 lines); hunks: -0,0 +1,326; symbols: HybridLayerType, Qwen3NextConfig, to, __init__
  - `python/sglang/srt/models/qwen3_next_mtp.py` added +117/-0 (117 lines); hunks: -0,0 +1,117; symbols: Qwen3NextForCausalLMMTP, __init__, forward, load_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -0,0 +1,1072 @@
+import enum
+import logging
+from typing import Any, Dict, Iterable, Optional, Set, Tuple
+import torch
+import torch.nn.functional as F
+from torch import nn
diff -- python/sglang/srt/configs/qwen3_next.py
@@ -0,0 +1,326 @@
+# coding=utf-8
+# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
diff -- python/sglang/srt/models/qwen3_next_mtp.py
@@ -0,0 +1,117 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_next.py` added +1072/-0; `python/sglang/srt/configs/qwen3_next.py` added +326/-0; `python/sglang/srt/models/qwen3_next_mtp.py` added +117/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/__init__.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/configs/qwen3_next.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #10322 - [bugfix] fix norm type error in qwen3_next model

- Link: https://github.com/sgl-project/sglang/pull/10322
- Status/date: merged / 2025-09-11
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/models/qwen3_next_mtp.py`; associated commits `4a0e0be2a2b4`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +10/-51, 89 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR adds or enables a model support/runtime surface. Title: "[bugfix] fix norm type error in qwen3_next model". The diff centers on `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/models/qwen3_next_mtp.py`. PR body context: ## Motivation ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [ ] Format your code according to the Format code with pre-commit. - [ ] Add unit t...
- Key implementation: `python/sglang/srt/models/qwen3_next.py` modified +9/-42 (51 lines); hunks: -518,24 +518,10 @@ def __init__(; -685,23 +671,10 @@ def __init__(; symbols: __init__, get_layer, forward, touching `__init__, get_layer, forward`; `python/sglang/srt/models/qwen3_next_mtp.py` modified +1/-9 (10 lines); hunks: -54,15 +54,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_next.py` modified +9/-42 (51 lines); hunks: -518,24 +518,10 @@ def __init__(; -685,23 +671,10 @@ def __init__(; symbols: __init__, get_layer, forward
  - `python/sglang/srt/models/qwen3_next_mtp.py` modified +1/-9 (10 lines); hunks: -54,15 +54,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -518,24 +518,10 @@ def __init__(
-        if getattr(
-            config, "use_gemma_rms_norm", getattr(config, "apply_layernorm_1p", False)
-        ):
-            logger.warning_once(
-                "Using Gemma RMSNorm for input normalization and post attn normalization."
-            )
diff -- python/sglang/srt/models/qwen3_next_mtp.py
@@ -54,15 +54,7 @@ def __init__(
-        if getattr(
-            config, "use_gemma_rms_norm", getattr(config, "apply_layernorm_1p", False)
-        ):
-            logger.warning_once(
-                "Using Gemma RMSNorm for input normalization and post attn normalization."
-            )
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +9/-42; `python/sglang/srt/models/qwen3_next_mtp.py` modified +1/-9
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/models/qwen3_next_mtp.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #10379 - Support Qwen3-Next on Ascend NPU

- Link: https://github.com/sgl-project/sglang/pull/10379
- Status/date: merged / 2025-09-12
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_next.py`; associated commits `16cd550c8554`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +79/-26, 281 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR adds or enables a model support/runtime surface. Title: "Support Qwen3-Next on Ascend NPU". The diff centers on `python/sglang/srt/models/qwen3_next.py`. PR body context: ## Motivation ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [x] Format your code according to the Format code with pre-commit. - [ ] Add unit t...
- Key implementation: `python/sglang/srt/models/qwen3_next.py` modified +6/-3 (9 lines); hunks: -46,10 +46,11; -327,7 +328,7 @@ def __init__(; symbols: __init__, fix_query_key_value_ordering, _forward_input_proj, forward, touching `__init__, fix_query_key_value_ordering, _forward_input_proj`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_next.py` modified +6/-3 (9 lines); hunks: -46,10 +46,11; -327,7 +328,7 @@ def __init__(; symbols: __init__, fix_query_key_value_ordering, _forward_input_proj, forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -46,10 +46,11 @@
-from sglang.srt.utils import add_prefix, is_cuda, make_layers, set_weight_attrs
+from sglang.srt.utils import add_prefix, is_cuda, is_npu, make_layers, set_weight_attrs
+_is_npu = is_npu()
@@ -327,7 +328,7 @@ def __init__(
-            device=torch.cuda.current_device(),
+            device=torch.get_device_module().current_device(),
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +6/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/fla/layernorm_gated.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/mem_cache/memory_pool.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #10392 - [Fix] Support qwen3-next MTP+DP

- Link: https://github.com/sgl-project/sglang/pull/10392
- Status/date: merged / 2025-09-13
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_next_mtp.py`; associated commits `9752861002d7`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +29/-18, 120 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR adds or enables a model support/runtime surface. Title: "[Fix] Support qwen3-next MTP+DP". The diff centers on `python/sglang/srt/models/qwen3_next_mtp.py`. PR body context: ## Motivation When MTP/SpecDecoding and DP are both enabled in qwen3-next, it will fail at cuda graph capture & forward: ## Modifications - Fix `set_dp_buffer_len` logic in logi...
- Key implementation: `python/sglang/srt/models/qwen3_next_mtp.py` modified +5/-2 (7 lines); hunks: -85,8 +85,11 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_next_mtp.py` modified +5/-2 (7 lines); hunks: -85,8 +85,11 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_next_mtp.py
@@ -85,8 +85,11 @@ def forward(
-        input_embeds = self.pre_fc_norm_embedding(input_embeds)
-        hidden_states = self.pre_fc_norm_hidden(forward_batch.spec_info.hidden_states)
+        hidden_states = forward_batch.spec_info.hidden_states
+        # Some idle batch has 0 batch size. GemmaRMSNorm.forward would fail due to bs=0.
+        if not forward_batch.forward_mode.is_idle():
+            input_embeds = self.pre_fc_norm_embedding(input_embeds)
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_next_mtp.py` modified +5/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/layers/logits_processor.py`, `python/sglang/srt/mem_cache/memory_pool.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #10466 - feat: update support for qwen3next model

- Link: https://github.com/sgl-project/sglang/pull/10466
- Status/date: merged / 2025-09-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +11/-7, 74 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR adds or enables a model support/runtime surface. Title: "feat: update support for qwen3next model". The diff centers on `python/sglang/srt/layers/attention/fla/fused_recurrent.py`, `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py`. PR body context: ## Motivation 1、fix the l2norm accuracy issue 2、add support for the upcoming FP8 Qwen3Next model Note: DeepGEMM is currently not supported for Qwen3Next FP8, please set `SGL_ENA...
- Key implementation: `python/sglang/srt/layers/attention/fla/fused_recurrent.py` modified +4/-4 (8 lines); hunks: -86,8 +86,8 @@ def fused_recurrent_gated_delta_rule_fwd_kernel(; -411,8 +411,8 @@ def fused_recurrent_gated_delta_rule_update_fwd_kernel(; symbols: fused_recurrent_gated_delta_rule_fwd_kernel, fused_recurrent_gated_delta_rule_update_fwd_kernel, touching `fused_recurrent_gated_delta_rule_fwd_kernel, fused_recurrent_gated_delta_rule_update_fwd_kernel`; `python/sglang/srt/models/qwen3_next.py` modified +5/-1 (6 lines); hunks: -239,6 +239,7 @@ def __init__(; -278,13 +279,15 @@ def __init__(; symbols: __init__, touching `__init__`; `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py` modified +2/-2 (4 lines); hunks: -119,8 +119,8 @@ def fused_sigmoid_gating_delta_rule_update_kernel(; symbols: fused_sigmoid_gating_delta_rule_update_kernel, touching `fused_sigmoid_gating_delta_rule_update_kernel`.
- Code diff details:
  - `python/sglang/srt/layers/attention/fla/fused_recurrent.py` modified +4/-4 (8 lines); hunks: -86,8 +86,8 @@ def fused_recurrent_gated_delta_rule_fwd_kernel(; -411,8 +411,8 @@ def fused_recurrent_gated_delta_rule_update_fwd_kernel(; symbols: fused_recurrent_gated_delta_rule_fwd_kernel, fused_recurrent_gated_delta_rule_update_fwd_kernel
  - `python/sglang/srt/models/qwen3_next.py` modified +5/-1 (6 lines); hunks: -239,6 +239,7 @@ def __init__(; -278,13 +279,15 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py` modified +2/-2 (4 lines); hunks: -119,8 +119,8 @@ def fused_sigmoid_gating_delta_rule_update_kernel(; symbols: fused_sigmoid_gating_delta_rule_update_kernel
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/fla/fused_recurrent.py
@@ -86,8 +86,8 @@ def fused_recurrent_gated_delta_rule_fwd_kernel(
-            b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q)) + 1e-6)
-            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k)) + 1e-6)
+            b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6))
+            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))
@@ -411,8 +411,8 @@ def fused_recurrent_gated_delta_rule_update_fwd_kernel(
-            b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q)) + 1e-6)
diff -- python/sglang/srt/models/qwen3_next.py
@@ -239,6 +239,7 @@ def __init__(
+        quant_config: Optional[QuantizationConfig] = None,
@@ -278,13 +279,15 @@ def __init__(
+            quant_config=quant_config,
+            quant_config=None,
@@ -336,6 +339,7 @@ def __init__(
+            quant_config=quant_config,
diff -- python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py
@@ -119,8 +119,8 @@ def fused_sigmoid_gating_delta_rule_update_kernel(
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/fla/fused_recurrent.py` modified +4/-4; `python/sglang/srt/models/qwen3_next.py` modified +5/-1; `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/fla/fused_recurrent.py`, `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py`, `python/sglang/srt/models/qwen3_next.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #10622 - support qwen3-next-fp8 deepep

- Link: https://github.com/sgl-project/sglang/pull/10622
- Status/date: merged / 2025-09-18
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_next.py`; associated commits `1344ebc8333d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +93/-9, 204 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR adds or enables a model support/runtime surface. Title: "support qwen3-next-fp8 deepep". The diff centers on `python/sglang/srt/models/qwen3_next.py`. PR body context: ## Motivation need merge https://github.com/sgl-project/sglang/pull/10624 first ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [ ] Format your c...
- Key implementation: `python/sglang/srt/models/qwen3_next.py` modified +29/-8 (37 lines); hunks: -13,6 +13,7; -46,7 +47,14; symbols: forward, __init__, routed_experts_weights_of_layer, touching `forward, __init__, routed_experts_weights_of_layer`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_next.py` modified +29/-8 (37 lines); hunks: -13,6 +13,7; -46,7 +47,14; symbols: forward, __init__, routed_experts_weights_of_layer
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -13,6 +13,7 @@
+from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
@@ -46,7 +47,14 @@
-from sglang.srt.utils import add_prefix, is_cuda, is_npu, make_layers, set_weight_attrs
+from sglang.srt.utils import (
+    LazyValue,
+    add_prefix,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +29/-8
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_next.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #10657 - feat: add eagle3 support for qwen3-next model

- Link: https://github.com/sgl-project/sglang/pull/10657
- Status/date: open / 2025-09-19
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +45/-3, 113 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR adds or enables a model support/runtime surface. Title: "feat: add eagle3 support for qwen3-next model". The diff centers on `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/model_executor/model_runner.py`. PR body context: ## Motivation Add eagle3 speculative alg support for qwen3-next model ## Modifications 1. support eagle3 func/params in qwen3-next model file; 2. for hybrid_gdn model, save atte...
- Key implementation: `python/sglang/srt/models/qwen3_next.py` modified +38/-3 (41 lines); hunks: -1,6 +1,6; -837,6 +837,9 @@ def get_layer(idx: int, prefix: str):; symbols: get_layer, forward, HybridLayerType, touching `get_layer, forward, HybridLayerType`; `python/sglang/srt/model_executor/model_runner.py` modified +7/-0 (7 lines); hunks: -347,6 +347,7 @@ def initialize(self, min_per_gpu_memory: float):; -1743,6 +1744,12 @@ def _get_attention_backend(self):; symbols: initialize, _get_attention_backend, touching `initialize, _get_attention_backend`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_next.py` modified +38/-3 (41 lines); hunks: -1,6 +1,6; -837,6 +837,9 @@ def get_layer(idx: int, prefix: str):; symbols: get_layer, forward, HybridLayerType
  - `python/sglang/srt/model_executor/model_runner.py` modified +7/-0 (7 lines); hunks: -347,6 +347,7 @@ def initialize(self, min_per_gpu_memory: float):; -1743,6 +1744,12 @@ def _get_attention_backend(self):; symbols: initialize, _get_attention_backend
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -1,6 +1,6 @@
-from typing import Any, Dict, Iterable, Optional, Set, Tuple
+from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
@@ -837,6 +837,9 @@ def get_layer(idx: int, prefix: str):
+        # For EAGLE3 support
+        self.layers_to_capture = []
@@ -855,9 +858,16 @@ def forward(
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -347,6 +347,7 @@ def initialize(self, min_per_gpu_memory: float):
+            self.server_args.full_attention_backend = self.server_args.attention_backend
@@ -1743,6 +1744,12 @@ def _get_attention_backend(self):
+        elif self.is_draft_worker and hasattr(
+            self.server_args, "full_attention_backend"
+        ):
+            attn_backend = self._get_attention_backend_from_str(
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +38/-3; `python/sglang/srt/model_executor/model_runner.py` modified +7/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/models/qwen3_next.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #11585 - Clean up some Qwen3-Next and deterministic code

- Link: https://github.com/sgl-project/sglang/pull/11585
- Status/date: merged / 2025-10-15
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/configs/qwen3_next.py`; associated commits `6b143d62a297`
- Diff scope read: GitHub Pull Request files API returned 4 files, +1/-15, 47 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR extends deployment docs, tests, or CI coverage. Title: "Clean up some Qwen3-Next and deterministic code". The diff centers on `python/sglang/srt/configs/qwen3_next.py`. PR body context: ## Motivation Clean up Qwen3-Next and deterministic code ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [ ] Format your code according to the Fo...
- Key implementation: `python/sglang/srt/configs/qwen3_next.py` modified +0/-3 (3 lines); hunks: -27,12 +27,9; symbols: HybridLayerType, Qwen3NextConfig, touching `HybridLayerType, Qwen3NextConfig`.
- Code diff details:
  - `python/sglang/srt/configs/qwen3_next.py` modified +0/-3 (3 lines); hunks: -27,12 +27,9; symbols: HybridLayerType, Qwen3NextConfig
- Key code excerpts:

```diff
diff -- python/sglang/srt/configs/qwen3_next.py
@@ -27,12 +27,9 @@
-# NOTE: HybridLayerType
-    swa_attention = "swa_attention"
-    mamba2 = "mamba"
```

- Reviewed files:
  - runtime: `python/sglang/srt/configs/qwen3_next.py` modified +0/-3
- Risk and verification: The diff ships test coverage in `python/sglang/test/test_deterministic.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #10912 - [PD] Add PD support for hybrid model (Qwen3-Next, DeepSeek V3.2 Exp)

- Link: https://github.com/sgl-project/sglang/pull/10912
- Status/date: merged / 2025-10-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 13 files, +727/-186, 1433 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR adds or enables a model support/runtime surface. Title: "[PD] Add PD support for hybrid model (Qwen3-Next, DeepSeek V3.2 Exp)". The diff centers on `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/disaggregation/mooncake/conn.py`. PR body context: ## Motivation and Modifications To support hybrid attention models like Qwen3-Next with mamba state or DeepSeek V3.2 Exp with indexer cache, this PR extends the current PD disag...
- Key implementation: `python/sglang/srt/model_executor/model_runner.py` modified +25/-9 (34 lines); hunks: -1670,19 +1670,34 @@ def init_memory_pool(; -1808,6 +1823,7 @@ def init_memory_pool(; symbols: init_memory_pool, touching `init_memory_pool`; `python/sglang/srt/mem_cache/memory_pool.py` modified +248/-137 (385 lines); hunks: -142,72 +142,93 @@ def __init__(; -253,6 +274,22 @@ def fork_from(self, src_index: torch.Tensor) -> Optional[to...; symbols: __init__, for, get_speculative_mamba2_params_all_layers, fork_from, touching `__init__, for, get_speculative_mamba2_params_all_layers`; `python/sglang/srt/disaggregation/mooncake/conn.py` modified +148/-17 (165 lines); hunks: -58,6 +58,7 @@ class TransferKVChunk:; -69,6 +70,7 @@ class TransferInfo:; symbols: TransferKVChunk, TransferInfo, from_zmq, KVArgsRegisterInfo, touching `TransferKVChunk, TransferInfo, from_zmq`; `python/sglang/srt/disaggregation/decode.py` modified +113/-8 (121 lines); hunks: -25,11 +25,12; -47,9 +48,19; symbols: clear, HybridMambaDecodeReqToTokenPool, __init__, DecodeRequest, touching `clear, HybridMambaDecodeReqToTokenPool, __init__`.
- Code diff details:
  - `python/sglang/srt/model_executor/model_runner.py` modified +25/-9 (34 lines); hunks: -1670,19 +1670,34 @@ def init_memory_pool(; -1808,6 +1823,7 @@ def init_memory_pool(; symbols: init_memory_pool
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +248/-137 (385 lines); hunks: -142,72 +142,93 @@ def __init__(; -253,6 +274,22 @@ def fork_from(self, src_index: torch.Tensor) -> Optional[to...; symbols: __init__, for, get_speculative_mamba2_params_all_layers, fork_from
  - `python/sglang/srt/disaggregation/mooncake/conn.py` modified +148/-17 (165 lines); hunks: -58,6 +58,7 @@ class TransferKVChunk:; -69,6 +70,7 @@ class TransferInfo:; symbols: TransferKVChunk, TransferInfo, from_zmq, KVArgsRegisterInfo
  - `python/sglang/srt/disaggregation/decode.py` modified +113/-8 (121 lines); hunks: -25,11 +25,12; -47,9 +48,19; symbols: clear, HybridMambaDecodeReqToTokenPool, __init__, DecodeRequest
  - `test/srt/test_disaggregation_hybrid_attention.py` added +83/-0 (83 lines); hunks: -0,0 +1,83; symbols: TestDisaggregationHybridAttentionMamba, setUpClass, start_prefill, start_decode
- Key code excerpts:

```diff
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -1670,19 +1670,34 @@ def init_memory_pool(
-                from sglang.srt.disaggregation.decode import DecodeReqToTokenPool
+                from sglang.srt.disaggregation.decode import (
+                    DecodeReqToTokenPool,
+                    HybridMambaDecodeReqToTokenPool,
+                )
-                self.req_to_token_pool = DecodeReqToTokenPool(
diff -- python/sglang/srt/mem_cache/memory_pool.py
@@ -142,72 +142,93 @@ def __init__(
-        # assume conv_state = (dim, state_len)
-        assert conv_state_shape[0] > conv_state_shape[1]
-        conv_state = torch.zeros(
-            size=(num_mamba_layers, size + 1) + conv_state_shape,
-            dtype=conv_dtype,
-            device=device,
diff -- python/sglang/srt/disaggregation/mooncake/conn.py
@@ -58,6 +58,7 @@ class TransferKVChunk:
```

- Reviewed files:
  - runtime: `python/sglang/srt/model_executor/model_runner.py` modified +25/-9; `python/sglang/srt/mem_cache/memory_pool.py` modified +248/-137; `python/sglang/srt/disaggregation/mooncake/conn.py` modified +148/-17; `python/sglang/srt/disaggregation/decode.py` modified +113/-8; `python/sglang/srt/disaggregation/prefill.py` modified +71/-1; `python/sglang/srt/disaggregation/base/conn.py` modified +17/-4
  - tests: `test/srt/test_disaggregation_hybrid_attention.py` added +83/-0
- Risk and verification: The diff ships test coverage in `test/srt/run_suite.py`, `test/srt/test_disaggregation_hybrid_attention.py`, `test/srt/test_mamba_unittest.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #11487 - init support for KTransformers Heterogeneous Computing

- Link: https://github.com/sgl-project/sglang/pull/11487
- Status/date: merged / 2025-10-21
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +547/-17, 842 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR adds or enables a model support/runtime surface. Title: "init support for KTransformers Heterogeneous Computing". The diff centers on `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/models/deepseek_v2.py`. PR body context: ## Motivation KTransformers Integration to Support CPU/GPU Hybrid Inference for MoE Models Only support CompressedTensor format. Will support other formats later. ## Modificatio...
- Key implementation: `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +408/-8 (416 lines); hunks: -4,24 +4,47; -51,6 +74,18; symbols: _mask_topk_ids_cpu_experts, mask_cpu_expert_ids, GPTQMarlinState, __new__, touching `_mask_topk_ids_cpu_experts, mask_cpu_expert_ids, GPTQMarlinState`; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +25/-3 (28 lines); hunks: -28,6 +28,11; -123,7 +128,6 @@ def __init__(; symbols: __init__, _weight_loader_physical, _weight_loader_impl, forward, touching `__init__, _weight_loader_physical, _weight_loader_impl`; `python/sglang/srt/models/deepseek_v2.py` modified +21/-5 (26 lines); hunks: -44,6 +44,7; -82,7 +83,12; symbols: forward_normal_dual_stream, __init__, post_load_weights, touching `forward_normal_dual_stream, __init__, post_load_weights`; `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py` modified +10/-1 (11 lines); hunks: -19,11 +19,13; -38,6 +40,7; symbols: to_int, CompressedTensorsConfig, __init__, get_quant_method, touching `to_int, CompressedTensorsConfig, __init__`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +408/-8 (416 lines); hunks: -4,24 +4,47; -51,6 +74,18; symbols: _mask_topk_ids_cpu_experts, mask_cpu_expert_ids, GPTQMarlinState, __new__
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +25/-3 (28 lines); hunks: -28,6 +28,11; -123,7 +128,6 @@ def __init__(; symbols: __init__, _weight_loader_physical, _weight_loader_impl, forward
  - `python/sglang/srt/models/deepseek_v2.py` modified +21/-5 (26 lines); hunks: -44,6 +44,7; -82,7 +83,12; symbols: forward_normal_dual_stream, __init__, post_load_weights
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py` modified +10/-1 (11 lines); hunks: -19,11 +19,13; -38,6 +40,7; symbols: to_int, CompressedTensorsConfig, __init__, get_quant_method
  - `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +9/-0 (9 lines); hunks: -64,6 +64,13; -247,6 +254,8 @@ def __init__(self, model_runner: ModelRunner):; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py
@@ -4,24 +4,47 @@
+import re
+try:
+    from sgl_kernel import fused_marlin_moe
+    FUSED_MARLIN_MOE_AVAILABLE = True
+except ImportError:
+    FUSED_MARLIN_MOE_AVAILABLE = False
diff -- python/sglang/srt/layers/moe/fused_moe_triton/layer.py
@@ -28,6 +28,11 @@
+from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors_moe import (
+    CompressedTensorsWNA16AMXEPMoEMethod,
+    CompressedTensorsWNA16AMXMoEMethod,
+    CompressedTensorsWNA16MoEMethod,
+)
@@ -123,7 +128,6 @@ def __init__(
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -44,6 +44,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +408/-8; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +25/-3; `python/sglang/srt/models/deepseek_v2.py` modified +21/-5; `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py` modified +10/-1; `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +9/-0; `python/sglang/srt/layers/quantization/compressed_tensors/__init__.py` modified +7/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/environ.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/layers/quantization/compressed_tensors/__init__.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #11969 - [NPU] bugfix for Qwen3-Next and performance update

- Link: https://github.com/sgl-project/sglang/pull/11969
- Status/date: merged / 2025-10-30
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_next.py`; associated commits `cafebef1546e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +68/-21, 162 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[NPU] bugfix for Qwen3-Next and performance update". The diff centers on `python/sglang/srt/models/qwen3_next.py`. PR body context: ## Motivation This PR aims to solve model issues and improve model performance and throughput on Ascend NPUs. We have reached a better performance result on a single Altas 800I...
- Key implementation: `python/sglang/srt/models/qwen3_next.py` modified +7/-0 (7 lines); hunks: -478,6 +478,13 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_next.py` modified +7/-0 (7 lines); hunks: -478,6 +478,13 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -478,6 +478,13 @@ def forward(
+        # Add padding for DP-Attn
+        if is_dp_attention_enabled():
+            core_attn_out_pad = torch.zeros_like(z)
+            core_attn_out_pad[: core_attn_out.shape[0], :] = core_attn_out
+            core_attn_out = core_attn_out_pad
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +7/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/fla/layernorm_gated.py`, `python/sglang/srt/layers/attention/mamba/mamba.py`, `python/sglang/srt/layers/moe/topk.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #12508 - [GDN] Fuse b.sigmoid(), fused_gdn_gating and unsqueeze into one kernel: up to 0.85% e2e speedup

- Link: https://github.com/sgl-project/sglang/pull/12508
- Status/date: merged / 2025-11-06
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +71/-51, 151 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR optimizes an inference path or backend selection. Title: "[GDN] Fuse b.sigmoid(), fused_gdn_gating and unsqueeze into one kernel: up to 0.85% e2e speedup". The diff centers on `python/sglang/srt/layers/attention/fla/fused_gdn_gating.py`, `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`. PR body context: ## Motivation Target Extend: 89us -> 79us: 10us saving -> Verify: 3.5us -> 1.4us: 2.1us saving -> ## Accuracy Tests Accuracy: 0.950 ## Benchmarking and Profiling H100 x 4 from 3...
- Key implementation: `python/sglang/srt/layers/attention/fla/fused_gdn_gating.py` added +69/-0 (69 lines); hunks: -0,0 +1,69; symbols: fused_gdn_gating_kernel, fused_gdn_gating, touching `fused_gdn_gating_kernel, fused_gdn_gating`; `python/sglang/srt/models/qwen3_next.py` modified +0/-45 (45 lines); hunks: -190,51 +190,6 @@ def fused_qkvzba_split_reshape_cat(; symbols: fused_qkvzba_split_reshape_cat, fused_gdn_gating_kernel, fused_gdn_gating, Qwen3GatedDeltaNet, touching `fused_qkvzba_split_reshape_cat, fused_gdn_gating_kernel, fused_gdn_gating`; `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +2/-6 (8 lines); hunks: -5,6 +5,7; -30,7 +31,6; symbols: forward_extend, touching `forward_extend`.
- Code diff details:
  - `python/sglang/srt/layers/attention/fla/fused_gdn_gating.py` added +69/-0 (69 lines); hunks: -0,0 +1,69; symbols: fused_gdn_gating_kernel, fused_gdn_gating
  - `python/sglang/srt/models/qwen3_next.py` modified +0/-45 (45 lines); hunks: -190,51 +190,6 @@ def fused_qkvzba_split_reshape_cat(; symbols: fused_qkvzba_split_reshape_cat, fused_gdn_gating_kernel, fused_gdn_gating, Qwen3GatedDeltaNet
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +2/-6 (8 lines); hunks: -5,6 +5,7; -30,7 +31,6; symbols: forward_extend
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/fla/fused_gdn_gating.py
@@ -0,0 +1,69 @@
+from typing import Tuple
+import torch
+import triton
+import triton.language as tl
+# g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
+# beta_output = b.sigmoid()
diff -- python/sglang/srt/models/qwen3_next.py
@@ -190,51 +190,6 @@ def fused_qkvzba_split_reshape_cat(
-# g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
-@triton.jit
-def fused_gdn_gating_kernel(
-    g,
-    A_log,
-    a,
diff -- python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py
@@ -5,6 +5,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/fla/fused_gdn_gating.py` added +69/-0; `python/sglang/srt/models/qwen3_next.py` modified +0/-45; `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +2/-6
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/fla/fused_gdn_gating.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/models/qwen3_next.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #12892 - [GDN/Qwen3-Next] Avoid SSM and conv state copy for speculative decoding - up to 9.47% e2e speedup

- Link: https://github.com/sgl-project/sglang/pull/12892
- Status/date: open / 2025-11-08
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +172/-241, 840 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR changes model-related implementation. Title: "[GDN/Qwen3-Next] Avoid SSM and conv state copy for speculative decoding - up to 9.47% e2e speedup". The diff centers on `python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/layers/attention/fla/fused_recurrent.py`. PR body context: ## Motivation Replace intermediate state buffers with last_steps for speculative decoding. All conv and SSM states are now lazily updated inside their kernels based on last_step...
- Key implementation: `python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py` modified +53/-126 (179 lines); hunks: -460,14 +460,14 @@ def causal_conv1d_fn(; -576,8 +576,7 @@ def _causal_conv1d_update_kernel(; symbols: causal_conv1d_fn, _causal_conv1d_update_kernel, touching `causal_conv1d_fn, _causal_conv1d_update_kernel`; `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +33/-55 (88 lines); hunks: -36,24 +36,21; -630,50 +627,39 @@ def forward_extend(; symbols: forward_extend, forward, update_mamba_state_after_mtp_verify, touching `forward_extend, forward, update_mamba_state_after_mtp_verify`; `python/sglang/srt/layers/attention/fla/fused_recurrent.py` modified +28/-22 (50 lines); hunks: -348,8 +348,7 @@ def fused_recurrent_gated_delta_rule(; -366,9 +365,9 @@ def fused_recurrent_gated_delta_rule_update_fwd_kernel(; symbols: fused_recurrent_gated_delta_rule, fused_recurrent_gated_delta_rule_update_fwd_kernel, touching `fused_recurrent_gated_delta_rule, fused_recurrent_gated_delta_rule_update_fwd_kernel`; `python/sglang/srt/mem_cache/memory_pool.py` modified +55/-35 (90 lines); hunks: -140,8 +140,16 @@ def mem_usage_bytes(self):; -171,32 +179,40 @@ def __init__(; symbols: mem_usage_bytes, SpeculativeState, at_layer_idx, __init__, touching `mem_usage_bytes, SpeculativeState, at_layer_idx`.
- Code diff details:
  - `python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py` modified +53/-126 (179 lines); hunks: -460,14 +460,14 @@ def causal_conv1d_fn(; -576,8 +576,7 @@ def _causal_conv1d_update_kernel(; symbols: causal_conv1d_fn, _causal_conv1d_update_kernel
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +33/-55 (88 lines); hunks: -36,24 +36,21; -630,50 +627,39 @@ def forward_extend(; symbols: forward_extend, forward, update_mamba_state_after_mtp_verify
  - `python/sglang/srt/layers/attention/fla/fused_recurrent.py` modified +28/-22 (50 lines); hunks: -348,8 +348,7 @@ def fused_recurrent_gated_delta_rule(; -366,9 +365,9 @@ def fused_recurrent_gated_delta_rule_update_fwd_kernel(; symbols: fused_recurrent_gated_delta_rule, fused_recurrent_gated_delta_rule_update_fwd_kernel
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +55/-35 (90 lines); hunks: -140,8 +140,16 @@ def mem_usage_bytes(self):; -171,32 +179,40 @@ def __init__(; symbols: mem_usage_bytes, SpeculativeState, at_layer_idx, __init__
  - `sgl-kernel/csrc/mamba/causal_conv1d.cu` modified +2/-2 (4 lines); hunks: -182,8 +182,8 @@ void causal_conv1d_fwd(const at::Tensor &x, const at::Tensor...
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py
@@ -460,14 +460,14 @@ def causal_conv1d_fn(
+        stride_istate_seq = conv_states.stride(0)
+        stride_istate_dim = conv_states.stride(-2)
+        stride_istate_token = conv_states.stride(-1)
-            and dim == conv_states.shape[1]
-            and width - 1 <= conv_states.shape[2]
+            and dim == conv_states.shape[-2]
diff -- python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py
@@ -36,24 +36,21 @@
-    from sglang.srt.layers.attention.mamba.causal_conv1d import (
-        causal_conv1d_fn as causal_conv1d_fn_cuda,
-    )
+    pass
-    causal_conv1d_fn = causal_conv1d_fn_cuda
+    # TODO: add this back. This is temporarily commented to wait for sgl-kernel new version release
diff -- python/sglang/srt/layers/attention/fla/fused_recurrent.py
@@ -348,8 +348,7 @@ def fused_recurrent_gated_delta_rule(
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py` modified +53/-126; `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +33/-55; `python/sglang/srt/layers/attention/fla/fused_recurrent.py` modified +28/-22; `python/sglang/srt/mem_cache/memory_pool.py` modified +55/-35; `python/sglang/srt/speculative/eagle_worker.py` modified +1/-1
  - other: `sgl-kernel/csrc/mamba/causal_conv1d.cu` modified +2/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/fla/fused_recurrent.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #13708 - [Fix] Qwen3Next lmhead dtype

- Link: https://github.com/sgl-project/sglang/pull/13708
- Status/date: merged / 2025-11-21
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +0/-1, 8 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Fix] Qwen3Next lmhead dtype". The diff centers on `python/sglang/srt/models/qwen3_next.py`. PR body context: ## Motivation The default dtype of `Qwen3-Next-80B-A3B` lmhead should be bf16? See lmhead dtype in vllm: ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Chec...
- Key implementation: `python/sglang/srt/models/qwen3_next.py` modified +0/-1 (1 lines); hunks: -864,7 +864,6 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_next.py` modified +0/-1 (1 lines); hunks: -864,7 +864,6 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -864,7 +864,6 @@ def __init__(
-        self.lm_head = self.lm_head.float()
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +0/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3_next.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #13081 - Support piecewise cuda graph for Qwen3-next

- Link: https://github.com/sgl-project/sglang/pull/13081
- Status/date: merged / 2025-11-25
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_next.py`; associated commits `d64bf6c6ce70`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +112/-3, 191 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR adds or enables a model support/runtime surface. Title: "Support piecewise cuda graph for Qwen3-next". The diff centers on `python/sglang/srt/models/qwen3_next.py`. PR body context: ## Motivation Support piecewise cuda graph for Qwen3-next https://github.com/sgl-project/sglang/issues/11490 ## Modifications 1. Temporarily split the entire GDN attention due t...
- Key implementation: `python/sglang/srt/models/qwen3_next.py` modified +62/-1 (63 lines); hunks: -5,6 +5,7; -53,9 +54,13; symbols: fused_qkvzba_split_reshape_cat_kernel, fix_query_key_value_ordering, _forward_input_proj, forward, touching `fused_qkvzba_split_reshape_cat_kernel, fix_query_key_value_ordering, _forward_input_proj`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_next.py` modified +62/-1 (63 lines); hunks: -5,6 +5,7; -53,9 +54,13; symbols: fused_qkvzba_split_reshape_cat_kernel, fix_query_key_value_ordering, _forward_input_proj, forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -5,6 +5,7 @@
+from sglang.srt.compilation.piecewise_context_manager import get_forward_context
@@ -53,9 +54,13 @@
+from sglang.srt.compilation.piecewise_context_manager import get_forward_context
+from sglang.srt.utils import direct_register_custom_op
@@ -349,7 +354,11 @@ def fix_query_key_value_ordering(self, mixed_qkvz, mixed_ba):
-        DUAL_STREAM_TOKEN_THRESHOLD = 1024 if not _is_npu else 0
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +62/-1
- Risk and verification: The diff ships test coverage in `test/srt/models/test_qwen3_next_models.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #13964 - [Performance]Qwen3 Next kernel performance optimize

- Link: https://github.com/sgl-project/sglang/pull/13964
- Status/date: open / 2025-11-26
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +34/-24, 121 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR optimizes an inference path or backend selection. Title: "[Performance]Qwen3 Next kernel performance optimize". The diff centers on `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py`. PR body context: ## Motivation Update from: https://github.com/sgl-project/sglang/pull/13924 Improve the fused_sigmoid_gating_delta_rule_update_kernel performance. ## Modifications 1. autotune a...
- Key implementation: `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py` modified +34/-24 (58 lines); hunks: -7,6 +7,17; -71,6 +82,10 @@ def fused_sigmoid_gating_delta_rule_update_kernel(; symbols: fused_sigmoid_gating_delta_rule_update_kernel, fused_sigmoid_gating_delta_rule_update, touching `fused_sigmoid_gating_delta_rule_update_kernel, fused_sigmoid_gating_delta_rule_update`.
- Code diff details:
  - `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py` modified +34/-24 (58 lines); hunks: -7,6 +7,17; -71,6 +82,10 @@ def fused_sigmoid_gating_delta_rule_update_kernel(; symbols: fused_sigmoid_gating_delta_rule_update_kernel, fused_sigmoid_gating_delta_rule_update
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py
@@ -7,6 +7,17 @@
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=2, num_stages=3),
+        triton.Config({}, num_warps=2, num_stages=4),
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py` modified +34/-24
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #14502 - [Qwen3-Next]Optimize piecewise CUDA graph for Qwen3-Next

- Link: https://github.com/sgl-project/sglang/pull/14502
- Status/date: open / 2025-12-05
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +248/-123, 498 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR adds or enables a model support/runtime surface. Title: "[Qwen3-Next]Optimize piecewise CUDA graph for Qwen3-Next". The diff centers on `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`. PR body context: ## Motivation Based on https://github.com/sgl-project/sglang/pull/13081, this PR reduce TTFT when piecewise CUDA graph is enabled. The previous version kept the entire linear-at...
- Key implementation: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +219/-75 (294 lines); hunks: -5,6 +5,7; -37,6 +38,7; symbols: forward_extend, _causal_conv1d_gdn_core, touching `forward_extend, _causal_conv1d_gdn_core`; `python/sglang/srt/models/qwen3_next.py` modified +0/-41 (41 lines); hunks: -5,7 +5,6; -49,7 +48,6; symbols: fused_qkvzba_split_reshape_cat_kernel, forward, _forward, get_model_config_for_expert_location, touching `fused_qkvzba_split_reshape_cat_kernel, forward, _forward`; `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` modified +12/-4 (16 lines); hunks: -330,9 +330,13 @@ def warmup_torch_compile(self, num_tokens: int):; -477,9 +481,13 @@ def capture_one_batch_size(self, num_tokens: int):; symbols: warmup_torch_compile, capture_one_batch_size, touching `warmup_torch_compile, capture_one_batch_size`; `python/sglang/srt/mem_cache/memory_pool.py` modified +16/-2 (18 lines); hunks: -131,10 +131,15 @@ class State:; -151,6 +156,15 @@ class SpeculativeState(State):; symbols: State, at_layer_idx, SpeculativeState, __init__, touching `State, at_layer_idx, SpeculativeState`.
- Code diff details:
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +219/-75 (294 lines); hunks: -5,6 +5,7; -37,6 +38,7; symbols: forward_extend, _causal_conv1d_gdn_core
  - `python/sglang/srt/models/qwen3_next.py` modified +0/-41 (41 lines); hunks: -5,7 +5,6; -49,7 +48,6; symbols: fused_qkvzba_split_reshape_cat_kernel, forward, _forward, get_model_config_for_expert_location
  - `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` modified +12/-4 (16 lines); hunks: -330,9 +330,13 @@ def warmup_torch_compile(self, num_tokens: int):; -477,9 +481,13 @@ def capture_one_batch_size(self, num_tokens: int):; symbols: warmup_torch_compile, capture_one_batch_size
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +16/-2 (18 lines); hunks: -131,10 +131,15 @@ class State:; -151,6 +156,15 @@ class SpeculativeState(State):; symbols: State, at_layer_idx, SpeculativeState, __init__
  - `python/sglang/srt/compilation/compilation_config.py` modified +1/-1 (2 lines); hunks: -17,7 +17,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py
@@ -5,6 +5,7 @@
+from sglang.srt.compilation.piecewise_context_manager import get_forward_context
@@ -37,6 +38,7 @@
+from sglang.srt.utils.custom_op import register_custom_op
@@ -957,6 +959,9 @@ def forward_extend(
+        g, beta = fused_gdn_gating(A_log, a, b, dt_bias)
@@ -968,13 +973,11 @@ def forward_extend(
diff -- python/sglang/srt/models/qwen3_next.py
@@ -5,7 +5,6 @@
-from sglang.srt.compilation.piecewise_context_manager import get_forward_context
@@ -49,7 +48,6 @@
-from sglang.srt.utils.custom_op import register_custom_op
@@ -59,8 +57,6 @@
-from sglang.srt.compilation.piecewise_context_manager import get_forward_context
@@ -373,22 +369,6 @@ def forward(
diff -- python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py
@@ -330,9 +330,13 @@ def warmup_torch_compile(self, num_tokens: int):
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +219/-75; `python/sglang/srt/models/qwen3_next.py` modified +0/-41; `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` modified +12/-4; `python/sglang/srt/mem_cache/memory_pool.py` modified +16/-2; `python/sglang/srt/compilation/compilation_config.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/compilation/compilation_config.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/mem_cache/memory_pool.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #14855 - Clean up GDN Init

- Link: https://github.com/sgl-project/sglang/pull/14855
- Status/date: merged / 2025-12-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-13, 47 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR extends deployment docs, tests, or CI coverage. Title: "Clean up GDN Init". The diff centers on `python/sglang/srt/models/qwen3_next.py`. PR body context: ## Motivation torch.log is very confusing to developer and we can actually just delete it ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [ ] For...
- Key implementation: `python/sglang/srt/models/qwen3_next.py` modified +5/-13 (18 lines); hunks: -7,7 +7,7; -221,7 +221,6 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_next.py` modified +5/-13 (18 lines); hunks: -7,7 +7,7; -221,7 +221,6 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -7,7 +7,7 @@
-from sglang.srt.distributed import divide, get_pp_group
+from sglang.srt.distributed import get_pp_group
@@ -221,7 +221,6 @@ def __init__(
-        # QKV
@@ -232,7 +231,6 @@ def __init__(
-        # projection of the input hidden states
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +5/-13
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3_next.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #16164 - [NPU] Adapt qwen3-next W8A8 on NPU

- Link: https://github.com/sgl-project/sglang/pull/16164
- Status/date: merged / 2026-01-03
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_next.py`; associated commits `6bc5a52fd2d4`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +18/-5, 121 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[NPU] Adapt qwen3-next W8A8 on NPU". The diff centers on `python/sglang/srt/models/qwen3_next.py`. PR body context: ## Motivation Fix the bugs on NPU when running TP+EP on the qwen3-next model, improve the performance, and adapt to W8A8 quantization. ## Modifications The qwen3-next model was...
- Key implementation: `python/sglang/srt/models/qwen3_next.py` modified +18/-5 (23 lines); hunks: -202,6 +202,7 @@ def __init__(; -229,6 +230,7 @@ def __init__(; symbols: __init__, fix_query_key_value_ordering, touching `__init__, fix_query_key_value_ordering`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_next.py` modified +18/-5 (23 lines); hunks: -202,6 +202,7 @@ def __init__(; -229,6 +230,7 @@ def __init__(; symbols: __init__, fix_query_key_value_ordering
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -202,6 +202,7 @@ def __init__(
+        prefix: str = "",
@@ -229,6 +230,7 @@ def __init__(
+            prefix=add_prefix("conv1d", prefix),
@@ -241,14 +243,16 @@ def __init__(
+            prefix=add_prefix("in_proj_qkvz", prefix),
-            quant_config=None,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +18/-5
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3_next.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #16488 - Two-Batch Overlap (TBO) support to Qwen3-Next Models

- Link: https://github.com/sgl-project/sglang/pull/16488
- Status/date: open / 2026-01-05
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +484/-13, 594 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR adds or enables a model support/runtime surface. Title: "Two-Batch Overlap (TBO) support to Qwen3-Next Models". The diff centers on `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`. PR body context: ## Motivation This PR adds Two-Batch Overlap (TBO) support to Qwen3Next models. The optimization takes effect when `--enable-piecewise-cuda-graph` is not enabled, providing perf...
- Key implementation: `python/sglang/srt/models/qwen3_next.py` modified +293/-11 (304 lines); hunks: -12,7 +12,7; -50,6 +50,8; symbols: _forward, op_prepare, op_core, Qwen3HybridLinearDecoderLayer, touching `_forward, op_prepare, op_core`; `python/sglang/srt/models/qwen2_moe.py` modified +91/-0 (91 lines); hunks: -78,6 +78,7; -324,6 +325,96 @@ def forward(; symbols: forward, op_gate, op_shared_experts, op_select_experts, touching `forward, op_gate, op_shared_experts`; `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +5/-1 (6 lines); hunks: -223,6 +223,9 @@ def _forward_metadata(self, forward_batch: ForwardBatch):; -1040,7 +1043,8 @@ def forward_extend(; symbols: _forward_metadata, forward_extend, touching `_forward_metadata, forward_extend`; `python/sglang/srt/batch_overlap/operations_strategy.py` modified +85/-0 (85 lines); hunks: -51,6 +51,16 @@ def init_new_tbo(; -209,3 +219,78 @@ def _compute_moe_qwen3_decode(layer):; symbols: init_new_tbo, _compute_moe_qwen3_decode, _compute_moe_qwen3_next_layer_operations_strategy_tbo, _compute_moe_qwen3_next_prefill, touching `init_new_tbo, _compute_moe_qwen3_decode, _compute_moe_qwen3_next_layer_operations_strategy_tbo`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_next.py` modified +293/-11 (304 lines); hunks: -12,7 +12,7; -50,6 +50,8; symbols: _forward, op_prepare, op_core, Qwen3HybridLinearDecoderLayer
  - `python/sglang/srt/models/qwen2_moe.py` modified +91/-0 (91 lines); hunks: -78,6 +78,7; -324,6 +325,96 @@ def forward(; symbols: forward, op_gate, op_shared_experts, op_select_experts
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +5/-1 (6 lines); hunks: -223,6 +223,9 @@ def _forward_metadata(self, forward_batch: ForwardBatch):; -1040,7 +1043,8 @@ def forward_extend(; symbols: _forward_metadata, forward_extend
  - `python/sglang/srt/batch_overlap/operations_strategy.py` modified +85/-0 (85 lines); hunks: -51,6 +51,16 @@ def init_new_tbo(; -209,3 +219,78 @@ def _compute_moe_qwen3_decode(layer):; symbols: init_new_tbo, _compute_moe_qwen3_decode, _compute_moe_qwen3_next_layer_operations_strategy_tbo, _compute_moe_qwen3_next_prefill
  - `python/sglang/srt/batch_overlap/two_batch_overlap.py` modified +9/-0 (9 lines); hunks: -79,6 +79,15 @@ def compute_split_seq_index(; symbols: compute_split_seq_index
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -12,7 +12,7 @@
-from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes
+from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes, ScatterMode
@@ -50,6 +50,8 @@
+from sglang.srt.batch_overlap.two_batch_overlap import model_forward_maybe_tbo
@@ -468,7 +470,100 @@ def _forward(
+    def op_prepare(self, state):
diff -- python/sglang/srt/models/qwen2_moe.py
@@ -78,6 +78,7 @@
+    is_non_idle_and_non_empty
@@ -324,6 +325,96 @@ def forward(
+    def op_gate(self, state):
+        if is_non_idle_and_non_empty(
+            state.forward_batch.forward_mode, state.hidden_states_mlp_input
+        ):
diff -- python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py
@@ -223,6 +223,9 @@ def _forward_metadata(self, forward_batch: ForwardBatch):
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +293/-11; `python/sglang/srt/models/qwen2_moe.py` modified +91/-0; `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +5/-1; `python/sglang/srt/batch_overlap/operations_strategy.py` modified +85/-0; `python/sglang/srt/batch_overlap/two_batch_overlap.py` modified +9/-0
  - tests: `test/manual/test_two_batch_overlap.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `test/manual/test_two_batch_overlap.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #16863 - tiny refactor pcg split op registration

- Link: https://github.com/sgl-project/sglang/pull/16863
- Status/date: merged / 2026-01-10
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +20/-6, 81 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR adds or enables a model support/runtime surface. Title: "tiny refactor pcg split op registration". The diff centers on `python/sglang/srt/layers/radix_attention.py`, `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/compilation/compilation_config.py`. PR body context: ## Motivation ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [ ] Format your code according to the Format code with pre-commit. - [ ] Add unit t...
- Key implementation: `python/sglang/srt/layers/radix_attention.py` modified +2/-0 (2 lines); hunks: -20,6 +20,7; -132,6 +133,7 @@ def forward(; symbols: forward, unified_attention_with_output, touching `forward, unified_attention_with_output`; `python/sglang/srt/models/qwen3_next.py` modified +2/-0 (2 lines); hunks: -5,6 +5,7; -1060,6 +1061,7 @@ def get_model_config_for_expert_location(cls, config):; symbols: get_model_config_for_expert_location, gdn_with_output, touching `get_model_config_for_expert_location, gdn_with_output`; `python/sglang/srt/compilation/compilation_config.py` modified +14/-6 (20 lines); hunks: -1,6 +1,17; -15,11 +26,8 @@ def __init__(; symbols: register_split_op, decorator, __init__, add_split_op, touching `register_split_op, decorator, __init__`; `python/sglang/srt/distributed/parallel_state.py` modified +2/-0 (2 lines); hunks: -39,6 +39,7; -125,6 +126,7 @@ def _register_group(group: "GroupCoordinator") -> None:; symbols: _register_group, inplace_all_reduce, touching `_register_group, inplace_all_reduce`.
- Code diff details:
  - `python/sglang/srt/layers/radix_attention.py` modified +2/-0 (2 lines); hunks: -20,6 +20,7; -132,6 +133,7 @@ def forward(; symbols: forward, unified_attention_with_output
  - `python/sglang/srt/models/qwen3_next.py` modified +2/-0 (2 lines); hunks: -5,6 +5,7; -1060,6 +1061,7 @@ def get_model_config_for_expert_location(cls, config):; symbols: get_model_config_for_expert_location, gdn_with_output
  - `python/sglang/srt/compilation/compilation_config.py` modified +14/-6 (20 lines); hunks: -1,6 +1,17; -15,11 +26,8 @@ def __init__(; symbols: register_split_op, decorator, __init__, add_split_op
  - `python/sglang/srt/distributed/parallel_state.py` modified +2/-0 (2 lines); hunks: -39,6 +39,7; -125,6 +126,7 @@ def _register_group(group: "GroupCoordinator") -> None:; symbols: _register_group, inplace_all_reduce
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/radix_attention.py
@@ -20,6 +20,7 @@
+from sglang.srt.compilation.compilation_config import register_split_op
@@ -132,6 +133,7 @@ def forward(
+@register_split_op()
diff -- python/sglang/srt/models/qwen3_next.py
@@ -5,6 +5,7 @@
+from sglang.srt.compilation.compilation_config import register_split_op
@@ -1060,6 +1061,7 @@ def get_model_config_for_expert_location(cls, config):
+@register_split_op()
diff -- python/sglang/srt/compilation/compilation_config.py
@@ -1,6 +1,17 @@
-from typing import List
+from typing import Callable, List, Optional
+SPLIT_OPS = []
+def register_split_op(op_name: Optional[str] = None):
+    def decorator(op_func: Callable):
+        name = op_name or op_func.__name__
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/radix_attention.py` modified +2/-0; `python/sglang/srt/models/qwen3_next.py` modified +2/-0; `python/sglang/srt/compilation/compilation_config.py` modified +14/-6; `python/sglang/srt/distributed/parallel_state.py` modified +2/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/compilation/compilation_config.py`, `python/sglang/srt/distributed/parallel_state.py`, `python/sglang/srt/layers/radix_attention.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17016 - [bugfix] fix qwen3-next alt_stream none issue

- Link: https://github.com/sgl-project/sglang/pull/17016
- Status/date: merged / 2026-01-16
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_next.py`; associated commits `6f10e17b4a68`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-1, 13 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[bugfix] fix qwen3-next alt_stream none issue". The diff centers on `python/sglang/srt/models/qwen3_next.py`. PR body context: ## Motivation This patch is to solve the alt_stream None issue for qwen3-next on other non-cuda platform The command for how to reproduce on AMD MI300X: The error trace: The roo...
- Key implementation: `python/sglang/srt/models/qwen3_next.py` modified +5/-1 (6 lines); hunks: -358,7 +358,11 @@ def _forward_input_proj(self, hidden_states: torch.Tensor):; symbols: _forward_input_proj, touching `_forward_input_proj`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_next.py` modified +5/-1 (6 lines); hunks: -358,7 +358,11 @@ def _forward_input_proj(self, hidden_states: torch.Tensor):; symbols: _forward_input_proj
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -358,7 +358,11 @@ def _forward_input_proj(self, hidden_states: torch.Tensor):
-        if seq_len < DUAL_STREAM_TOKEN_THRESHOLD:
+        if (
+            seq_len < DUAL_STREAM_TOKEN_THRESHOLD
+            and self.alt_stream is not None
+            and get_is_capture_mode()
+        ):
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +5/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3_next.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #15631 - [jit-kernel] Add CuTe DSL GDN Decode Kernel

- Link: https://github.com/sgl-project/sglang/pull/15631
- Status/date: merged / 2026-01-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +1804/-1, 1842 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR adds or enables a model support/runtime surface. Title: "[jit-kernel] Add CuTe DSL GDN Decode Kernel". The diff centers on `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/jit_kernel/cutedsl_gdn.py`, `python/sglang/jit_kernel/tests/test_cutedsl_gdn.py`. PR body context: Co-author: @zhou9402, @HongliMi, @xutizhou ## Motivation ## Modifications Add CuTe DSL GDN Decode Kernel - hybrid_linear_attn_backend.py - Add CuTe DSL decode branch (env: SGLAN...
- Key implementation: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +12/-1 (13 lines); hunks: -5,6 +5,8; -38,6 +40,7; symbols: __init__, forward_decode, touching `__init__, forward_decode`; `python/sglang/jit_kernel/cutedsl_gdn.py` added +1494/-0 (1494 lines); hunks: -0,0 +1,1494; symbols: _define_kernels, gdn_kernel_small_batch, gdn_kernel_small_batch_varlen, gdn_kernel_large_batch, touching `_define_kernels, gdn_kernel_small_batch, gdn_kernel_small_batch_varlen`; `python/sglang/jit_kernel/tests/test_cutedsl_gdn.py` added +295/-0 (295 lines); hunks: -0,0 +1,295; symbols: run_triton_kernel, test_cutedsl_gdn_precision, test_cutedsl_gdn_performance, run_cutedsl, touching `run_triton_kernel, test_cutedsl_gdn_precision, test_cutedsl_gdn_performance`; `python/sglang/srt/environ.py` modified +3/-0 (3 lines); hunks: -175,6 +175,9 @@ class Envs:; symbols: Envs, touching `Envs`.
- Code diff details:
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +12/-1 (13 lines); hunks: -5,6 +5,8; -38,6 +40,7; symbols: __init__, forward_decode
  - `python/sglang/jit_kernel/cutedsl_gdn.py` added +1494/-0 (1494 lines); hunks: -0,0 +1,1494; symbols: _define_kernels, gdn_kernel_small_batch, gdn_kernel_small_batch_varlen, gdn_kernel_large_batch
  - `python/sglang/jit_kernel/tests/test_cutedsl_gdn.py` added +295/-0 (295 lines); hunks: -0,0 +1,295; symbols: run_triton_kernel, test_cutedsl_gdn_precision, test_cutedsl_gdn_performance, run_cutedsl
  - `python/sglang/srt/environ.py` modified +3/-0 (3 lines); hunks: -175,6 +175,9 @@ class Envs:; symbols: Envs
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py
@@ -5,6 +5,8 @@
+from sglang.jit_kernel.cutedsl_gdn import cutedsl_fused_sigmoid_gating_delta_rule_update
+from sglang.srt.environ import Envs
@@ -38,6 +40,7 @@
+from sglang.srt.utils.common import rank0_log
@@ -843,6 +846,14 @@ def __init__(self, model_runner: ModelRunner):
+        use_cutedsl = Envs.SGLANG_USE_CUTEDSL_GDN_DECODE.get()
diff -- python/sglang/jit_kernel/cutedsl_gdn.py
@@ -0,0 +1,1494 @@
+"""CuTe DSL Fused Sigmoid Gating Delta Rule Kernel for GDN Decode."""
+import logging
+from typing import Dict, Optional, Tuple
+import cuda.bindings.driver as cuda
+import cutlass
+import cutlass.cute as cute
diff -- python/sglang/jit_kernel/tests/test_cutedsl_gdn.py
@@ -0,0 +1,295 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +12/-1; `python/sglang/jit_kernel/cutedsl_gdn.py` added +1494/-0; `python/sglang/srt/environ.py` modified +3/-0
  - tests: `python/sglang/jit_kernel/tests/test_cutedsl_gdn.py` added +295/-0
- Risk and verification: The diff ships test coverage in `python/sglang/jit_kernel/tests/test_cutedsl_gdn.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17373 - refactor Qwen3-Next with a new RadixLinearAttention

- Link: https://github.com/sgl-project/sglang/pull/17373
- Status/date: merged / 2026-01-22
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_next.py`; associated commits `419bbcee10a5`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +200/-106, 444 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR changes model-related implementation. Title: "refactor Qwen3-Next with a new RadixLinearAttention". The diff centers on `python/sglang/srt/models/qwen3_next.py`. PR body context: ## Motivation Refactor: Abstract linear attention backend calls through `RadixLinearAttention` ## Modifications - `qwen3_next.py`: Replaced direct forward_batch.attn_backend.for...
- Key implementation: `python/sglang/srt/models/qwen3_next.py` modified +21/-37 (58 lines); hunks: -29,6 +29,7; -60,8 +61,6; symbols: fused_qkvzba_split_reshape_cat_kernel, __init__, fix_query_key_value_ordering, forward, touching `fused_qkvzba_split_reshape_cat_kernel, __init__, fix_query_key_value_ordering`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_next.py` modified +21/-37 (58 lines); hunks: -29,6 +29,7; -60,8 +61,6; symbols: fused_qkvzba_split_reshape_cat_kernel, __init__, fix_query_key_value_ordering, forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -29,6 +29,7 @@
+from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
@@ -60,8 +61,6 @@
-from sglang.srt.compilation.piecewise_context_manager import get_forward_context
@@ -305,6 +304,20 @@ def __init__(
+        self.linear_attn = RadixLinearAttention(
+            layer_id=layer_id,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +21/-37
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/layers/radix_linear_attention.py`, `python/sglang/srt/models/qwen3_next.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17570 - Use attn tp group in embedding for more models

- Link: https://github.com/sgl-project/sglang/pull/17570
- Status/date: merged / 2026-01-24
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 19 files, +19/-19, 171 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR extends deployment docs, tests, or CI coverage. Title: "Use attn tp group in embedding for more models". The diff centers on `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/bailing_moe_nextn.py`, `python/sglang/srt/models/falcon_h1.py`. PR body context: ## Motivation Follow up of https://github.com/sgl-project/sglang/pull/17403. cc: @ch-wan ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [ ] Form...
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

### PR #17660 - [hybrid-model] clean up and consolidate redundant fields in RadixLinearAttention

- Link: https://github.com/sgl-project/sglang/pull/17660
- Status/date: merged / 2026-01-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +54/-105, 298 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR extends deployment docs, tests, or CI coverage. Title: "[hybrid-model] clean up and consolidate redundant fields in RadixLinearAttention". The diff centers on `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/layers/radix_linear_attention.py`, `python/sglang/srt/models/kimi_linear.py`. PR body context: ## Motivation clean up and consolidate redundant fields in `RadixLinearAttention` ## Modifications ## Accuracy Tests Qwen3-Next Kimi Linear ## Benchmarking and Profiling ## Chec...
- Key implementation: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +34/-81 (115 lines); hunks: -635,15 +635,7 @@ def forward_decode(; -678,18 +670,18 @@ def forward_decode(; symbols: forward_decode, forward_extend, touching `forward_decode, forward_extend`; `python/sglang/srt/layers/radix_linear_attention.py` modified +12/-18 (30 lines); hunks: -31,36 +31,30 @@ class RadixLinearAttention(nn.Module):; symbols: RadixLinearAttention, __init__, touching `RadixLinearAttention, __init__`; `python/sglang/srt/models/kimi_linear.py` modified +4/-3 (7 lines); hunks: -315,11 +315,12 @@ def __init__(; symbols: __init__, touching `__init__`; `python/sglang/srt/models/qwen3_next.py` modified +4/-3 (7 lines); hunks: -306,11 +306,12 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +34/-81 (115 lines); hunks: -635,15 +635,7 @@ def forward_decode(; -678,18 +670,18 @@ def forward_decode(; symbols: forward_decode, forward_extend
  - `python/sglang/srt/layers/radix_linear_attention.py` modified +12/-18 (30 lines); hunks: -31,36 +31,30 @@ class RadixLinearAttention(nn.Module):; symbols: RadixLinearAttention, __init__
  - `python/sglang/srt/models/kimi_linear.py` modified +4/-3 (7 lines); hunks: -315,11 +315,12 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/qwen3_next.py` modified +4/-3 (7 lines); hunks: -306,11 +306,12 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py
@@ -635,15 +635,7 @@ def forward_decode(
-        head_dim = layer.head_qk_dim
-        layer_id = layer.layer_id
-        beta = b
-        g = a
-        A_log = layer.A_log
-        dt_bias = layer.dt_bias
diff -- python/sglang/srt/layers/radix_linear_attention.py
@@ -31,36 +31,30 @@ class RadixLinearAttention(nn.Module):
-        num_qk_heads: int,
+        num_q_heads: int,
+        num_k_heads: int,
-        head_qk_dim: int,
+        head_q_dim: int,
+        head_k_dim: int,
diff -- python/sglang/srt/models/kimi_linear.py
@@ -315,11 +315,12 @@ def __init__(
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +34/-81; `python/sglang/srt/layers/radix_linear_attention.py` modified +12/-18; `python/sglang/srt/models/kimi_linear.py` modified +4/-3; `python/sglang/srt/models/qwen3_next.py` modified +4/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/layers/radix_linear_attention.py`, `python/sglang/srt/models/kimi_linear.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #12525 - [CPU] Optimize Qwen3-next model on CPU

- Link: https://github.com/sgl-project/sglang/pull/12525
- Status/date: merged / 2026-01-30
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/configs/qwen3_next.py`, `python/sglang/srt/models/qwen3_next.py`; associated commits `336dc4579e94`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 13 files, +366/-41, 685 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR adds or enables a model support/runtime surface. Title: "[CPU] Optimize Qwen3-next model on CPU". The diff centers on `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/configs/qwen3_next.py`. PR body context: This PR adds unified CPU optimizations for Qwen3-next models, including: 1. Add CPU paths to call optimized kernels, which is depending on below sgl-kernels: a. chunk_gated_delt...
- Key implementation: `python/sglang/srt/models/qwen3_next.py` modified +31/-5 (36 lines); hunks: -46,6 +46,8; -56,6 +58,8; symbols: __init__, fix_query_key_value_ordering, _forward_input_proj, _forward, touching `__init__, fix_query_key_value_ordering, _forward_input_proj`; `python/sglang/srt/configs/qwen3_next.py` modified +7/-0 (7 lines); hunks: -20,8 +20,11; -276,6 +279,10 @@ def full_attention_layer_ids(self):; symbols: HybridLayerType, full_attention_layer_ids, mamba2_cache_params, touching `HybridLayerType, full_attention_layer_ids, mamba2_cache_params`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_next.py` modified +31/-5 (36 lines); hunks: -46,6 +46,8; -56,6 +58,8; symbols: __init__, fix_query_key_value_ordering, _forward_input_proj, _forward
  - `python/sglang/srt/configs/qwen3_next.py` modified +7/-0 (7 lines); hunks: -20,8 +20,11; -276,6 +279,10 @@ def full_attention_layer_ids(self):; symbols: HybridLayerType, full_attention_layer_ids, mamba2_cache_params
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -46,6 +46,8 @@
+    cpu_has_amx_support,
+    is_cpu,
@@ -56,6 +58,8 @@
+_is_cpu = is_cpu()
+_is_amx_available = cpu_has_amx_support()
@@ -209,8 +213,16 @@ def __init__(
diff -- python/sglang/srt/configs/qwen3_next.py
@@ -20,8 +20,11 @@
+from sglang.srt.configs.update_config import adjust_tp_num_heads_if_necessary
+from sglang.srt.utils import is_cpu
+_is_cpu = is_cpu()
@@ -276,6 +279,10 @@ def full_attention_layer_ids(self):
+        if _is_cpu:
+            world_size = get_attention_tp_size()
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +31/-5; `python/sglang/srt/configs/qwen3_next.py` modified +7/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/qwen3_next.py`, `python/sglang/srt/configs/update_config.py`, `python/sglang/srt/layers/amx_utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17981 - [Qwen3-Next] Add cutedsl decode/mtp kernel with transposed ssm_state and prefill gluon kernel for blackwell.

- Link: https://github.com/sgl-project/sglang/pull/17981
- Status/date: open / 2026-01-30
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +2128/-88, 2504 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR adds or enables a model support/runtime surface. Title: "[Qwen3-Next] Add cutedsl decode/mtp kernel with transposed ssm_state and prefill gluon kernel for blackwell.". The diff centers on `python/sglang/srt/layers/attention/linear/kernels/gdn_cutedsl_transpose.py`, `python/sglang/srt/layers/attention/fla/chunk_delta_h.py`, `python/sglang/srt/layers/attention/linear/gdn_backend.py`. PR body context: ## Motivation For Qwen3-Next model, we observed current preill/decode kernels cannot fully utilize the hardware efficiency of Blackwell. Thus, we implement gluon based prefill k...
- Key implementation: `python/sglang/srt/layers/attention/linear/kernels/gdn_cutedsl_transpose.py` added +115/-0 (115 lines); hunks: -0,0 +1,115; symbols: CuteDSLGDNTransposeKernel, decode, extend, target_verify, touching `CuteDSLGDNTransposeKernel, decode, extend`; `python/sglang/srt/layers/attention/fla/chunk_delta_h.py` modified +79/-28 (107 lines); hunks: -55,6 +55,7 @@ def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(; -101,23 +102,47 @@ def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(; symbols: chunk_gated_delta_rule_fwd_kernel_h_blockdim64, chunk_gated_delta_rule_fwd_h, touching `chunk_gated_delta_rule_fwd_kernel_h_blockdim64, chunk_gated_delta_rule_fwd_h`; `python/sglang/srt/layers/attention/linear/gdn_backend.py` modified +23/-1 (24 lines); hunks: -4,6 +4,9; -57,6 +60,7 @@ def __init__(; symbols: __init__, touching `__init__`; `python/sglang/srt/layers/attention/fla/chunk.py` modified +6/-0 (6 lines); hunks: -33,6 +33,7 @@ def chunk_gated_delta_rule_fwd(; -56,6 +57,7 @@ def chunk_gated_delta_rule_fwd(; symbols: chunk_gated_delta_rule_fwd, forward, chunk_gated_delta_rule, touching `chunk_gated_delta_rule_fwd, forward, chunk_gated_delta_rule`.
- Code diff details:
  - `python/sglang/srt/layers/attention/linear/kernels/gdn_cutedsl_transpose.py` added +115/-0 (115 lines); hunks: -0,0 +1,115; symbols: CuteDSLGDNTransposeKernel, decode, extend, target_verify
  - `python/sglang/srt/layers/attention/fla/chunk_delta_h.py` modified +79/-28 (107 lines); hunks: -55,6 +55,7 @@ def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(; -101,23 +102,47 @@ def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(; symbols: chunk_gated_delta_rule_fwd_kernel_h_blockdim64, chunk_gated_delta_rule_fwd_h
  - `python/sglang/srt/layers/attention/linear/gdn_backend.py` modified +23/-1 (24 lines); hunks: -4,6 +4,9; -57,6 +60,7 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/layers/attention/fla/chunk.py` modified +6/-0 (6 lines); hunks: -33,6 +33,7 @@ def chunk_gated_delta_rule_fwd(; -56,6 +57,7 @@ def chunk_gated_delta_rule_fwd(; symbols: chunk_gated_delta_rule_fwd, forward, chunk_gated_delta_rule
  - `python/sglang/srt/layers/attention/linear/utils.py` modified +4/-0 (4 lines); hunks: -15,13 +15,17; symbols: LinearAttnKernelBackend, is_triton, is_cutedsl, is_cutedsl_transpose
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/linear/kernels/gdn_cutedsl_transpose.py
@@ -0,0 +1,115 @@
+import logging
+import torch
+from sglang.jit_kernel.cutedsl_gdn_transpose import (
+    cutedsl_transpose_fused_recurrent_gated_delta_rule_update,
+    cutedsl_transpose_fused_sigmoid_gated_delta_rule_update,
+)
diff -- python/sglang/srt/layers/attention/fla/chunk_delta_h.py
@@ -55,6 +55,7 @@ def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
+    TRANSPOSE_STATE: tl.constexpr,
@@ -101,23 +102,47 @@ def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
-        p_h0_1 = tl.make_block_ptr(h0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
-        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
-        if K > 64:
-            p_h0_2 = tl.make_block_ptr(
diff -- python/sglang/srt/layers/attention/linear/gdn_backend.py
@@ -4,6 +4,9 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/linear/kernels/gdn_cutedsl_transpose.py` added +115/-0; `python/sglang/srt/layers/attention/fla/chunk_delta_h.py` modified +79/-28; `python/sglang/srt/layers/attention/linear/gdn_backend.py` modified +23/-1; `python/sglang/srt/layers/attention/fla/chunk.py` modified +6/-0; `python/sglang/srt/layers/attention/linear/utils.py` modified +4/-0; `python/sglang/jit_kernel/cutedsl_gdn_transpose.py` added +1038/-0
  - tests: `python/sglang/jit_kernel/tests/test_cutedsl_gdn.py` modified +858/-57
- Risk and verification: The diff ships test coverage in `python/sglang/jit_kernel/tests/test_cutedsl_gdn.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17983 - [Qwen3-Next] Optimize Prefill Kernel, add GDN Gluon kernel and optimize cumsum kernel

- Link: https://github.com/sgl-project/sglang/pull/17983
- Status/date: open / 2026-01-30
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +1248/-97, 1469 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR adds or enables a model support/runtime surface. Title: "[Qwen3-Next] Optimize Prefill Kernel, add GDN Gluon kernel and optimize cumsum kernel". The diff centers on `python/sglang/srt/layers/attention/fla/gluon/chunk_delta_h_gluon.py`, `python/sglang/srt/layers/attention/fla/gluon/wy_fast_gluon.py`, `python/sglang/srt/layers/attention/fla/gluon/chunk_o_gluon.py`. PR body context: ## Motivation This PR optimizes the prefill kernel for Qwen3-Next (Gated Delta Rule) models, focusing on key improvements: - Blackwell GPU Performance: Add Gluon kernels that le...
- Key implementation: `python/sglang/srt/layers/attention/fla/gluon/chunk_delta_h_gluon.py` added +293/-0 (293 lines); hunks: -0,0 +1,293; symbols: chunk_gated_delta_rule_fwd_kernel_h_blockdim64_gluon, touching `chunk_gated_delta_rule_fwd_kernel_h_blockdim64_gluon`; `python/sglang/srt/layers/attention/fla/gluon/wy_fast_gluon.py` added +245/-0 (245 lines); hunks: -0,0 +1,245; symbols: recompute_w_u_fwd_kernel_gluon, touching `recompute_w_u_fwd_kernel_gluon`; `python/sglang/srt/layers/attention/fla/gluon/chunk_o_gluon.py` added +210/-0 (210 lines); hunks: -0,0 +1,210; symbols: _mask_scalar, _apply_causal_mask, chunk_fwd_kernel_o_gluon, touching `_mask_scalar, _apply_causal_mask, chunk_fwd_kernel_o_gluon`; `python/sglang/srt/layers/attention/fla/chunk_delta_h.py` modified +178/-29 (207 lines); hunks: -13,7 +13,14; -55,6 +62,7 @@ def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(; symbols: chunk_gated_delta_rule_fwd_kernel_h_blockdim64, chunk_gated_delta_rule_fwd_h, touching `chunk_gated_delta_rule_fwd_kernel_h_blockdim64, chunk_gated_delta_rule_fwd_h`.
- Code diff details:
  - `python/sglang/srt/layers/attention/fla/gluon/chunk_delta_h_gluon.py` added +293/-0 (293 lines); hunks: -0,0 +1,293; symbols: chunk_gated_delta_rule_fwd_kernel_h_blockdim64_gluon
  - `python/sglang/srt/layers/attention/fla/gluon/wy_fast_gluon.py` added +245/-0 (245 lines); hunks: -0,0 +1,245; symbols: recompute_w_u_fwd_kernel_gluon
  - `python/sglang/srt/layers/attention/fla/gluon/chunk_o_gluon.py` added +210/-0 (210 lines); hunks: -0,0 +1,210; symbols: _mask_scalar, _apply_causal_mask, chunk_fwd_kernel_o_gluon
  - `python/sglang/srt/layers/attention/fla/chunk_delta_h.py` modified +178/-29 (207 lines); hunks: -13,7 +13,14; -55,6 +62,7 @@ def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(; symbols: chunk_gated_delta_rule_fwd_kernel_h_blockdim64, chunk_gated_delta_rule_fwd_h
  - `python/sglang/srt/layers/attention/fla/cumsum.py` modified +106/-18 (124 lines); hunks: -9,11 +9,77; -178,23 +244,45 @@ def chunk_local_cumsum_scalar(; symbols: chunk_local_cumsum_scalar_vectorization_kernel, chunk_local_cumsum_scalar
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/fla/gluon/chunk_delta_h_gluon.py
@@ -0,0 +1,293 @@
+from sglang.srt.layers.attention.fla.gluon import (
+    TensorMemoryLayout,
+    allocate_tensor_memory,
+    fence_async_shared,
+    get_tmem_reg_layout,
+    gl,
diff -- python/sglang/srt/layers/attention/fla/gluon/wy_fast_gluon.py
@@ -0,0 +1,245 @@
+from sglang.srt.layers.attention.fla.gluon import (
+    TensorMemoryLayout,
+    allocate_tensor_memory,
+    fence_async_shared,
+    get_tmem_reg_layout,
+    gl,
diff -- python/sglang/srt/layers/attention/fla/gluon/chunk_o_gluon.py
@@ -0,0 +1,210 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/fla/gluon/chunk_delta_h_gluon.py` added +293/-0; `python/sglang/srt/layers/attention/fla/gluon/wy_fast_gluon.py` added +245/-0; `python/sglang/srt/layers/attention/fla/gluon/chunk_o_gluon.py` added +210/-0; `python/sglang/srt/layers/attention/fla/chunk_delta_h.py` modified +178/-29; `python/sglang/srt/layers/attention/fla/cumsum.py` modified +106/-18; `python/sglang/srt/layers/attention/fla/chunk_o.py` modified +87/-28
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/fla/chunk_delta_h.py`, `python/sglang/srt/layers/attention/fla/chunk_o.py`, `python/sglang/srt/layers/attention/fla/cumsum.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #14607 - support qwen3-next eagle3

- Link: https://github.com/sgl-project/sglang/pull/14607
- Status/date: merged / 2026-02-01
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_next.py`; associated commits `3ca29dffc72a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +73/-6, 156 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR adds or enables a model support/runtime surface. Title: "support qwen3-next eagle3". The diff centers on `python/sglang/srt/models/qwen3_next.py`. PR body context: ## Motivation support qwen3 next eagle3,https://huggingface.co/lukeysong/qwen3-next-draft/tree/main ## Benchmarking and Profiling use Specfoge tp4 h20 |**Benchmark**|**Latency (...
- Key implementation: `python/sglang/srt/models/qwen3_next.py` modified +73/-6 (79 lines); hunks: -531,12 +531,18 @@ def forward(; -753,10 +759,16 @@ def forward(; symbols: forward, get_layer, set_eagle3_layers_to_capture, touching `forward, get_layer, set_eagle3_layers_to_capture`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_next.py` modified +73/-6 (79 lines); hunks: -531,12 +531,18 @@ def forward(; -753,10 +759,16 @@ def forward(; symbols: forward, get_layer, set_eagle3_layers_to_capture
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -531,12 +531,18 @@ def forward(
+        captured_last_layer_outputs: Optional[list[torch.Tensor]] = None,
-        hidden_states, residual = self.layer_communicator.prepare_attn(
-            hidden_states, residual, forward_batch
+        hidden_states, residual = (
+            self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
+                hidden_states,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +73/-6
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3_next.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18224 - [ModelOPT] Support Qwen 3 Next Coder NVFP4

- Link: https://github.com/sgl-project/sglang/pull/18224
- Status/date: merged / 2026-02-08
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +35/-6, 95 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR adds or enables a model support/runtime surface. Title: "[ModelOPT] Support Qwen 3 Next Coder NVFP4". The diff centers on `python/sglang/srt/models/qwen3_next.py`. PR body context: ## Motivation This branch include important bugfix for qwen 3 coder next nvfp4 # B300 `sglang serve --model vincentzed-hf/Qwen3-Coder-Next-NVFP4 --quantization modelopt_fp4` **W...
- Key implementation: `python/sglang/srt/models/qwen3_next.py` modified +35/-6 (41 lines); hunks: -665,6 +665,7 @@ def __init__(; -921,6 +922,15 @@ class HybridLayerType(enum.Enum):; symbols: __init__, HybridLayerType, Qwen3NextForCausalLM, touching `__init__, HybridLayerType, Qwen3NextForCausalLM`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_next.py` modified +35/-6 (41 lines); hunks: -665,6 +665,7 @@ def __init__(; -921,6 +922,15 @@ class HybridLayerType(enum.Enum):; symbols: __init__, HybridLayerType, Qwen3NextForCausalLM
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -665,6 +665,7 @@ def __init__(
+            quant_config=quant_config,
@@ -921,6 +922,15 @@ class HybridLayerType(enum.Enum):
+    # Map fused module names to their checkpoint (unfused) counterparts.
+    # This is needed so the quantization exclusion logic can match
+    # checkpoint-style names (e.g. "q_proj") against the fused sglang
+    # module names (e.g. "qkv_proj").
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +35/-6
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3_next.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18489 - [MODEL] Adding Support for Qwen3.5 Models

- Link: https://github.com/sgl-project/sglang/pull/18489
- Status/date: merged / 2026-02-09
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 17 files, +1923/-9, 2159 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR adds or enables a model support/runtime surface. Title: "[MODEL] Adding Support for Qwen3.5 Models". The diff centers on `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/configs/qwen3_5.py`. PR body context: ## Purpose This PR adds model support for the upcoming Qwen3.5 models, including both dense and MoE variants. Special thanks to @cao1zhg, @yizhang2077, and @attack204 for their...
- Key implementation: `python/sglang/srt/models/qwen3_5.py` added +1310/-0 (1310 lines); hunks: -0,0 +1,1310; symbols: Qwen3_5GatedDeltaNet, __init__, fix_query_key_value_ordering, forward, touching `Qwen3_5GatedDeltaNet, __init__, fix_query_key_value_ordering`; `python/sglang/srt/models/qwen3_5_mtp.py` added +415/-0 (415 lines); hunks: -0,0 +1,415; symbols: Qwen3_5MultiTokenPredictor, __init__, embed_input_ids, forward, touching `Qwen3_5MultiTokenPredictor, __init__, embed_input_ids`; `python/sglang/srt/configs/qwen3_5.py` added +113/-0 (113 lines); hunks: -0,0 +1,113; symbols: Qwen3_5VisionConfig, Qwen3_5TextConfig, __init__, Qwen3_5Config, touching `Qwen3_5VisionConfig, Qwen3_5TextConfig, __init__`; `python/sglang/srt/model_executor/model_runner.py` modified +14/-3 (17 lines); hunks: -38,6 +38,8; -1498,8 +1500,15 @@ def qwen3_next_config(self):; symbols: qwen3_next_config, hybrid_gdn_config, compute_logprobs_only, model_is_mrope, touching `qwen3_next_config, hybrid_gdn_config, compute_logprobs_only`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` added +1310/-0 (1310 lines); hunks: -0,0 +1,1310; symbols: Qwen3_5GatedDeltaNet, __init__, fix_query_key_value_ordering, forward
  - `python/sglang/srt/models/qwen3_5_mtp.py` added +415/-0 (415 lines); hunks: -0,0 +1,415; symbols: Qwen3_5MultiTokenPredictor, __init__, embed_input_ids, forward
  - `python/sglang/srt/configs/qwen3_5.py` added +113/-0 (113 lines); hunks: -0,0 +1,113; symbols: Qwen3_5VisionConfig, Qwen3_5TextConfig, __init__, Qwen3_5Config
  - `python/sglang/srt/model_executor/model_runner.py` modified +14/-3 (17 lines); hunks: -38,6 +38,8; -1498,8 +1500,15 @@ def qwen3_next_config(self):; symbols: qwen3_next_config, hybrid_gdn_config, compute_logprobs_only, model_is_mrope
  - `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +16/-1 (17 lines); hunks: -7,6 +7,7; -15,6 +16,10; symbols: preprocess_video, QwenVLImageProcessor, process_mm_data_async
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -0,0 +1,1310 @@
+# Copyright 2025 Qwen Team
+# Copyright 2025 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
diff -- python/sglang/srt/models/qwen3_5_mtp.py
@@ -0,0 +1,415 @@
+# Copyright 2023-2024 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/configs/qwen3_5.py
@@ -0,0 +1,113 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_5.py` added +1310/-0; `python/sglang/srt/models/qwen3_5_mtp.py` added +415/-0; `python/sglang/srt/configs/qwen3_5.py` added +113/-0; `python/sglang/srt/model_executor/model_runner.py` modified +14/-3; `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +16/-1; `python/sglang/srt/layers/logits_processor.py` modified +11/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/__init__.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/configs/qwen3_5.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17613 - [Perf] refactor piecewise cuda graph support of Qwen3-Next

- Link: https://github.com/sgl-project/sglang/pull/17613
- Status/date: merged / 2026-02-14
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_next.py`; associated commits `8be18c655d0f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +80/-34, 196 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR adds or enables a model support/runtime surface. Title: "[Perf] refactor piecewise cuda graph support of Qwen3-Next". The diff centers on `python/sglang/srt/models/qwen3_next.py`. PR body context: ## Motivation Currently in Qwen3Next, we hide all ops inside `Qwen3GatedDeltaNet.forward` in a custom fake op, so that even with piecewise cuda graph, the whole forward of Qwen3...
- Key implementation: `python/sglang/srt/models/qwen3_next.py` modified +2/-19 (21 lines); hunks: -316,7 +316,7 @@ def __init__(; -405,23 +405,6 @@ def forward(; symbols: __init__, forward, _forward, touching `__init__, forward, _forward`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_next.py` modified +2/-19 (21 lines); hunks: -316,7 +316,7 @@ def __init__(; -405,23 +405,6 @@ def forward(; symbols: __init__, forward, _forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -316,7 +316,7 @@ def __init__(
-        self.linear_attn = RadixLinearAttention(
+        self.attn = RadixLinearAttention(
@@ -405,23 +405,6 @@ def forward(
-        if forward_batch.forward_mode.is_extend() and get_forward_context() is not None:
-            output = torch.empty_like(hidden_states)
-            gdn_with_output(
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +2/-19
- Risk and verification: The diff ships test coverage in `test/registered/models/test_qwen3_next_models_pcg.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18917 - [Qwen3-Next] Enable fused_qkvzba_split_reshape_cat also for prefill

- Link: https://github.com/sgl-project/sglang/pull/18917
- Status/date: merged / 2026-02-22
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_next.py`; associated commits `5995bfec6368`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-7, 19 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR adds or enables a model support/runtime surface. Title: "[Qwen3-Next] Enable fused_qkvzba_split_reshape_cat also for prefill". The diff centers on `python/sglang/srt/models/qwen3_next.py`. PR body context: ## Motivation In the GDN (Gated Delta Net) prefill path of Qwen3-Next, the `fix_query_key_value_ordering` function uses a chain of PyTorch `view` / `split` / `reshape` / `cat` o...
- Key implementation: `python/sglang/srt/models/qwen3_next.py` modified +1/-7 (8 lines); hunks: -405,17 +405,11 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_next.py` modified +1/-7 (8 lines); hunks: -405,17 +405,11 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -405,17 +405,11 @@ def forward(
-        is_cuda_graph = forward_batch.forward_mode.is_cuda_graph()
-        if (
-            self.num_v_heads // self.num_k_heads in [1, 2, 4]
-            and is_cuda_graph
-            and not _is_cpu
-        ):
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +1/-7
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3_next.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18355 - [AMD] Support Qwen3-Coder-Next on AMD platform

- Link: https://github.com/sgl-project/sglang/pull/18355
- Status/date: merged / 2026-02-25
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +213/-74, 395 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR adds or enables a model support/runtime surface. Title: "[AMD] Support Qwen3-Coder-Next on AMD platform". The diff centers on `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/models/qwen3_next.py`. PR body context: ## Motivation Enable Qwen3-Coder-Next model on AMD GPU platform. With this PR, we are able to support non-MTP (fp8 kv cache) and MTP on Qwen3-Coder-Next. ## Modifications - aite...
- Key implementation: `python/sglang/srt/layers/attention/aiter_backend.py` modified +211/-72 (283 lines); hunks: -89,6 +89,9 @@ class ForwardMetadata:; -123,7 +126,6 @@ def __init__(; symbols: ForwardMetadata, __init__, init_forward_metadata, touching `ForwardMetadata, __init__, init_forward_metadata`; `python/sglang/srt/models/qwen3_next.py` modified +2/-2 (4 lines); hunks: -385,9 +385,9 @@ def _forward_input_proj(self, hidden_states: torch.Tensor):; symbols: _forward_input_proj, touching `_forward_input_proj`.
- Code diff details:
  - `python/sglang/srt/layers/attention/aiter_backend.py` modified +211/-72 (283 lines); hunks: -89,6 +89,9 @@ class ForwardMetadata:; -123,7 +126,6 @@ def __init__(; symbols: ForwardMetadata, __init__, init_forward_metadata
  - `python/sglang/srt/models/qwen3_next.py` modified +2/-2 (4 lines); hunks: -385,9 +385,9 @@ def _forward_input_proj(self, hidden_states: torch.Tensor):; symbols: _forward_input_proj
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/aiter_backend.py
@@ -89,6 +89,9 @@ class ForwardMetadata:
+    custom_mask: Optional[torch.Tensor] = None
+    mask_indptr: Optional[torch.Tensor] = None
+    max_extend_len: Optional[int] = None
@@ -123,7 +126,6 @@ def __init__(
-        self.v_head_dim = model_runner.token_to_kv_pool.get_value_buffer(0).shape[-1]
@@ -133,6 +135,21 @@ def __init__(
diff -- python/sglang/srt/models/qwen3_next.py
@@ -385,9 +385,9 @@ def _forward_input_proj(self, hidden_states: torch.Tensor):
-            seq_len < DUAL_STREAM_TOKEN_THRESHOLD
-            and self.alt_stream is not None
+            self.alt_stream is not None
+            and seq_len < DUAL_STREAM_TOKEN_THRESHOLD
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/aiter_backend.py` modified +211/-72; `python/sglang/srt/models/qwen3_next.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/models/qwen3_next.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19220 - [PCG] fix piecewise cuda graph for Qwen3.5

- Link: https://github.com/sgl-project/sglang/pull/19220
- Status/date: merged / 2026-02-26
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +9/-46, 115 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[PCG] fix piecewise cuda graph for Qwen3.5". The diff centers on `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/layers/quantization/fp8_utils.py`. PR body context: ## Motivation fix piecewise cuda graph for Qwen3.5 ## Modifications 1. fix piecewise cuda graph for Qwen3.5 2. clean up legacy code `gdn_with_output` as it's not used anymore. #...
- Key implementation: `python/sglang/srt/models/qwen3_next.py` modified +0/-25 (25 lines); hunks: -5,8 +5,6; -53,7 +51,6; symbols: set_eagle3_layers_to_capture, gdn_with_output, touching `set_eagle3_layers_to_capture, gdn_with_output`; `python/sglang/srt/models/qwen3_5.py` modified +1/-21 (22 lines); hunks: -22,9 +22,6; -72,7 +69,6; symbols: forward, _forward, touching `forward, _forward`; `python/sglang/srt/layers/quantization/fp8_utils.py` modified +7/-0 (7 lines); hunks: -71,6 +71,13 @@ def _fp8_scaled_mm_abstract(mat_a, mat_b, scales_a, scales_b,...; symbols: _fp8_scaled_mm_abstract, _fp8_blockwise_scaled_mm_abstract, touching `_fp8_scaled_mm_abstract, _fp8_blockwise_scaled_mm_abstract`; `python/sglang/srt/models/qwen3_vl.py` modified +1/-0 (1 lines); hunks: -987,6 +987,7 @@ def get_input_embeddings(self):; symbols: get_input_embeddings, should_apply_lora, forward, touching `get_input_embeddings, should_apply_lora, forward`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_next.py` modified +0/-25 (25 lines); hunks: -5,8 +5,6; -53,7 +51,6; symbols: set_eagle3_layers_to_capture, gdn_with_output
  - `python/sglang/srt/models/qwen3_5.py` modified +1/-21 (22 lines); hunks: -22,9 +22,6; -72,7 +69,6; symbols: forward, _forward
  - `python/sglang/srt/layers/quantization/fp8_utils.py` modified +7/-0 (7 lines); hunks: -71,6 +71,13 @@ def _fp8_scaled_mm_abstract(mat_a, mat_b, scales_a, scales_b,...; symbols: _fp8_scaled_mm_abstract, _fp8_blockwise_scaled_mm_abstract
  - `python/sglang/srt/models/qwen3_vl.py` modified +1/-0 (1 lines); hunks: -987,6 +987,7 @@ def get_input_embeddings(self):; symbols: get_input_embeddings, should_apply_lora, forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -5,8 +5,6 @@
-from sglang.srt.compilation.compilation_config import register_split_op
-from sglang.srt.compilation.piecewise_context_manager import get_forward_context
@@ -53,7 +51,6 @@
-from sglang.srt.utils.custom_op import register_custom_op
@@ -1149,25 +1146,3 @@ def set_eagle3_layers_to_capture(self, layer_ids: Optional[list[int]] = None):
-@register_custom_op(mutates_args=["output"])
diff -- python/sglang/srt/models/qwen3_5.py
@@ -22,9 +22,6 @@
-# Model Executor
-from sglang.srt.compilation.piecewise_context_manager import get_forward_context
@@ -72,7 +69,6 @@
-from sglang.srt.models.qwen3_next import gdn_with_output
@@ -253,22 +249,6 @@ def forward(
-    ):
diff -- python/sglang/srt/layers/quantization/fp8_utils.py
@@ -71,6 +71,13 @@ def _fp8_scaled_mm_abstract(mat_a, mat_b, scales_a, scales_b, out_dtype, bias=No
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +0/-25; `python/sglang/srt/models/qwen3_5.py` modified +1/-21; `python/sglang/srt/layers/quantization/fp8_utils.py` modified +7/-0; `python/sglang/srt/models/qwen3_vl.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/quantization/fp8_utils.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_next.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19434 - [Qwen3-Next] Support gdn fused_rms_norm_gated

- Link: https://github.com/sgl-project/sglang/pull/19434
- Status/date: merged / 2026-02-27
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_next.py`; associated commits `d2885a9094c3`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +411/-299, 758 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR adds or enables a model support/runtime surface. Title: "[Qwen3-Next] Support gdn fused_rms_norm_gated". The diff centers on `python/sglang/srt/models/qwen3_next.py`. PR body context: ## Motivation This PR is to refactor Qwen3-Next GDN fused_rms_norm_gate. The ground is in attention / state-space style architectures, each layer and each step does a norm + gat...
- Key implementation: `python/sglang/srt/models/qwen3_next.py` modified +20/-8 (28 lines); hunks: -53,6 +53,9; -291,14 +294,23 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_next.py` modified +20/-8 (28 lines); hunks: -53,6 +53,9; -291,14 +294,23 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -53,6 +53,9 @@
+from sglang.srt.layers.attention.fla.fused_norm_gate import FusedRMSNormGated
@@ -291,14 +294,23 @@ def __init__(
-        self.norm = RMSNormGated(
-            self.head_v_dim,
-            eps=self.layer_norm_epsilon,
-            group_size=None,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +20/-8
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/fla/fused_norm_gate.py`, `python/sglang/srt/layers/attention/fla/kda.py`, `python/sglang/srt/models/kimi_linear.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17627 - [feat] Support nvfp4 quantized model of Qwen3-Next

- Link: https://github.com/sgl-project/sglang/pull/17627
- Status/date: merged / 2026-02-28
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_next.py`, `test/registered/models/test_qwen3_next_models_fp4.py`; associated commits `a2ea5941d562`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +83/-1, 105 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR adds or enables a model support/runtime surface. Title: "[feat] Support nvfp4 quantized model of Qwen3-Next". The diff centers on `test/registered/models/test_qwen3_next_models_fp4.py`, `python/sglang/srt/models/qwen3_next.py`. PR body context: ## Motivation Support https://huggingface.co/nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4 ## Modifications - The `qkv_proj` is not quantized in the NVFP4 quantized version, so is sp...
- Key implementation: `test/registered/models/test_qwen3_next_models_fp4.py` added +71/-0 (71 lines); hunks: -0,0 +1,71; symbols: TestQwen3NextFp4, setUpClass, tearDownClass, test_gsm8k, touching `TestQwen3NextFp4, setUpClass, tearDownClass`; `python/sglang/srt/models/qwen3_next.py` modified +12/-1 (13 lines); hunks: -625,13 +625,19 @@ def __init__(; -1123,6 +1129,11 @@ def load_weights(; symbols: __init__, load_weights, touching `__init__, load_weights`.
- Code diff details:
  - `test/registered/models/test_qwen3_next_models_fp4.py` added +71/-0 (71 lines); hunks: -0,0 +1,71; symbols: TestQwen3NextFp4, setUpClass, tearDownClass, test_gsm8k
  - `python/sglang/srt/models/qwen3_next.py` modified +12/-1 (13 lines); hunks: -625,13 +625,19 @@ def __init__(; -1123,6 +1129,11 @@ def load_weights(; symbols: __init__, load_weights
- Key code excerpts:

```diff
diff -- test/registered/models/test_qwen3_next_models_fp4.py
@@ -0,0 +1,71 @@
+import unittest
+from types import SimpleNamespace
+from sglang.srt.utils import get_device_sm, kill_process_tree
+from sglang.test.ci.ci_register import register_cuda_ci
+from sglang.test.few_shot_gsm8k import run_eval
+from sglang.test.test_utils import (
diff -- python/sglang/srt/models/qwen3_next.py
@@ -625,13 +625,19 @@ def __init__(
+        # qkv_proj is not quantized for fp4
-            quant_config=quant_config,
+            quant_config=(
+                quant_config
+                if quant_config is not None
+                and quant_config.get_name() != "modelopt_fp4"
```

- Reviewed files:
  - tests: `test/registered/models/test_qwen3_next_models_fp4.py` added +71/-0
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +12/-1
- Risk and verification: The diff ships test coverage in `test/registered/models/test_qwen3_next_models_fp4.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #19812 - Fix Qwen3.5/Qwen3Next MTP EPLB compatibility

- Link: https://github.com/sgl-project/sglang/pull/19812
- Status/date: open / 2026-03-04
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +26/-0, 47 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR adds or enables a model support/runtime surface. Title: "Fix Qwen3.5/Qwen3Next MTP EPLB compatibility". The diff centers on `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen2_moe.py`. PR body context: ## Summary - Fix three bugs when EPLB is enabled with Qwen3.5/Qwen3Next MTP (speculative decoding): - MTP layers created `ExpertLocationDispatchInfo` with wrong layer IDs (MTP h...
- Key implementation: `python/sglang/srt/models/qwen3_5.py` modified +25/-0 (25 lines); hunks: -888,6 +888,27 @@ def __init__(; -1200,6 +1221,10 @@ def __init__(; symbols: __init__, routed_experts_weights_of_layer, get_model_config_for_expert_location, load_weights, touching `__init__, routed_experts_weights_of_layer, get_model_config_for_expert_location`; `python/sglang/srt/models/qwen2_moe.py` modified +1/-0 (1 lines); hunks: -156,6 +156,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` modified +25/-0 (25 lines); hunks: -888,6 +888,27 @@ def __init__(; -1200,6 +1221,10 @@ def __init__(; symbols: __init__, routed_experts_weights_of_layer, get_model_config_for_expert_location, load_weights
  - `python/sglang/srt/models/qwen2_moe.py` modified +1/-0 (1 lines); hunks: -156,6 +156,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -888,6 +888,27 @@ def __init__(
+        self._routed_experts_weights_of_layer = LazyValue(
+            lambda: {
+                layer_id: layer.mlp.get_moe_weights()
+                for layer_id, layer in enumerate(self.layers)
+                if isinstance(layer.mlp, Qwen2MoeSparseMoeBlock)
+            }
diff -- python/sglang/srt/models/qwen2_moe.py
@@ -156,6 +156,7 @@ def __init__(
+        self.is_nextn = is_nextn
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +25/-0; `python/sglang/srt/models/qwen2_moe.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_5.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19767 - Fix qwen3.5 mtp eplb related issues

- Link: https://github.com/sgl-project/sglang/pull/19767
- Status/date: merged / 2026-03-09
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +79/-16, 272 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR adds or enables a model support/runtime surface. Title: "Fix qwen3.5 mtp eplb related issues". The diff centers on `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/models/qwen3_next_mtp.py`. PR body context: ## Motivation support eplb with mtp for Qwen3.5 & Qwen3-Next ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [ ] Format your code according to th...
- Key implementation: `python/sglang/srt/models/qwen3_5.py` modified +34/-1 (35 lines); hunks: -72,7 +72,14; -294,6 +301,7 @@ def __init__(; symbols: __init__, touching `__init__`; `python/sglang/srt/models/qwen3_5_mtp.py` modified +19/-6 (25 lines); hunks: -22,6 +22,8; -69,6 +71,7 @@ def __init__(; symbols: __init__, get_model_config_for_expert_location, get_embed_and_head, forward, touching `__init__, get_model_config_for_expert_location, get_embed_and_head`; `python/sglang/srt/models/qwen3_next_mtp.py` modified +12/-7 (19 lines); hunks: -22,6 +22,7; -62,7 +63,10 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`; `python/sglang/srt/models/qwen2_moe.py` modified +8/-2 (10 lines); hunks: -150,6 +150,7 @@ def __init__(; -220,6 +221,7 @@ def __init__(; symbols: __init__, get_moe_weights, _forward_deepep, touching `__init__, get_moe_weights, _forward_deepep`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` modified +34/-1 (35 lines); hunks: -72,7 +72,14; -294,6 +301,7 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/qwen3_5_mtp.py` modified +19/-6 (25 lines); hunks: -22,6 +22,8; -69,6 +71,7 @@ def __init__(; symbols: __init__, get_model_config_for_expert_location, get_embed_and_head, forward
  - `python/sglang/srt/models/qwen3_next_mtp.py` modified +12/-7 (19 lines); hunks: -22,6 +22,7; -62,7 +63,10 @@ def __init__(; symbols: __init__, forward
  - `python/sglang/srt/models/qwen2_moe.py` modified +8/-2 (10 lines); hunks: -150,6 +150,7 @@ def __init__(; -220,6 +221,7 @@ def __init__(; symbols: __init__, get_moe_weights, _forward_deepep
  - `python/sglang/srt/models/qwen3_next.py` modified +6/-0 (6 lines); hunks: -485,6 +485,7 @@ def __init__(; -513,6 +514,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -72,7 +72,14 @@
-from sglang.srt.utils import add_prefix, is_cuda, is_npu, make_layers, set_weight_attrs
+from sglang.srt.utils import (
+    LazyValue,
+    add_prefix,
+    is_cuda,
+    is_npu,
diff -- python/sglang/srt/models/qwen3_5_mtp.py
@@ -22,6 +22,8 @@
+from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
+from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
@@ -69,6 +71,7 @@ def __init__(
+            is_nextn=True,
@@ -84,6 +87,15 @@ def __init__(
+    @classmethod
diff -- python/sglang/srt/models/qwen3_next_mtp.py
@@ -22,6 +22,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +34/-1; `python/sglang/srt/models/qwen3_5_mtp.py` modified +19/-6; `python/sglang/srt/models/qwen3_next_mtp.py` modified +12/-7; `python/sglang/srt/models/qwen2_moe.py` modified +8/-2; `python/sglang/srt/models/qwen3_next.py` modified +6/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #20397 - [NPU] Qwen3 next Ascend Support MTP

- Link: https://github.com/sgl-project/sglang/pull/20397
- Status/date: open / 2026-03-12
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 11 files, +985/-94, 1352 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR adds or enables a model support/runtime surface. Title: "[NPU] Qwen3 next Ascend Support MTP". The diff centers on `python/sglang/srt/layers/attention/mamba/mamba_state_scatter_triton.py`, `python/sglang/srt/layers/attention/linear/gdn_backend.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`. PR body context: ## Motivation ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [ ] Format your code according to the Format code with pre-commit. - [ ] Add unit t...
- Key implementation: `python/sglang/srt/layers/attention/mamba/mamba_state_scatter_triton.py` modified +537/-0 (537 lines); hunks: -188,3 +188,540 @@ def fused_mamba_state_scatter_with_mask(; symbols: fused_mamba_state_scatter_with_mask, fused_qkvzba_split_reshape_cat_kernel, fused_qkvzba_split_reshape_cat_npu, move_cache_dynamic_last_kernel_h_block, touching `fused_mamba_state_scatter_with_mask, fused_qkvzba_split_reshape_cat_kernel, fused_qkvzba_split_reshape_cat_npu`; `python/sglang/srt/layers/attention/linear/gdn_backend.py` modified +282/-61 (343 lines); hunks: -1,7 +1,7; -37,9 +37,20; symbols: vllm_causal_conv1d_update, GDNKernelDispatcher, forward_decode, forward_extend, touching `vllm_causal_conv1d_update, GDNKernelDispatcher, forward_decode`; `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +94/-3 (97 lines); hunks: -22,13 +22,16; -142,6 +145,9 @@ def __init__(self, model_runner: ModelRunner):; symbols: __init__, _forward_metadata, prepare_gdn_inputs, init_forward_metadata, touching `__init__, _forward_metadata, prepare_gdn_inputs`; `python/sglang/srt/models/qwen3_next_mtp.py` modified +9/-3 (12 lines); hunks: -16,7 +16,7; -46,6 +46,8 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `python/sglang/srt/layers/attention/mamba/mamba_state_scatter_triton.py` modified +537/-0 (537 lines); hunks: -188,3 +188,540 @@ def fused_mamba_state_scatter_with_mask(; symbols: fused_mamba_state_scatter_with_mask, fused_qkvzba_split_reshape_cat_kernel, fused_qkvzba_split_reshape_cat_npu, move_cache_dynamic_last_kernel_h_block
  - `python/sglang/srt/layers/attention/linear/gdn_backend.py` modified +282/-61 (343 lines); hunks: -1,7 +1,7; -37,9 +37,20; symbols: vllm_causal_conv1d_update, GDNKernelDispatcher, forward_decode, forward_extend
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +94/-3 (97 lines); hunks: -22,13 +22,16; -142,6 +145,9 @@ def __init__(self, model_runner: ModelRunner):; symbols: __init__, _forward_metadata, prepare_gdn_inputs, init_forward_metadata
  - `python/sglang/srt/models/qwen3_next_mtp.py` modified +9/-3 (12 lines); hunks: -16,7 +16,7; -46,6 +46,8 @@ def __init__(; symbols: __init__, forward
  - `python/sglang/srt/layers/layernorm.py` modified +4/-3 (7 lines); hunks: -84,6 +84,7; -508,11 +509,11 @@ def forward_npu(; symbols: RMSNorm, forward_npu, forward_xpu
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/mamba/mamba_state_scatter_triton.py
@@ -188,3 +188,540 @@ def fused_mamba_state_scatter_with_mask(
+# todo: move to sgl_kernel_npu
+from sgl_kernel_npu.utils.triton_utils import get_device_properties
+MAX_ROWS_PER_ITER = 64
+@triton.jit(do_not_specialize=["total_rows", "rows_per_vec"])
+def fused_qkvzba_split_reshape_cat_kernel(
+    mixed_qkv,
diff -- python/sglang/srt/layers/attention/linear/gdn_backend.py
@@ -1,7 +1,7 @@
-from typing import Tuple, Union
+from typing import Tuple, Union, Optional
+import torch.nn.functional as F
@@ -37,9 +37,20 @@
+    from sgl_kernel_npu.fla.fused_gdn_gating import fused_gdn_gating_npu
+    from sglang.srt.layers.attention.fla.l2norm import l2norm_fwd
diff -- python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py
@@ -22,13 +22,16 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/mamba/mamba_state_scatter_triton.py` modified +537/-0; `python/sglang/srt/layers/attention/linear/gdn_backend.py` modified +282/-61; `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +94/-3; `python/sglang/srt/models/qwen3_next_mtp.py` modified +9/-3; `python/sglang/srt/layers/layernorm.py` modified +4/-3; `python/sglang/srt/models/qwen3_next.py` modified +5/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/hardware_backend/npu/memory_pool_npu.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19321 - [Qwen3-Next] Fuse Qwen3-Next GDN's qkvz_proj and ba_proj

- Link: https://github.com/sgl-project/sglang/pull/19321
- Status/date: merged / 2026-03-20
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_next.py`; associated commits `d9794ef9f760`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +107/-17, 207 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR optimizes an inference path or backend selection. Title: "[Qwen3-Next] Fuse Qwen3-Next GDN's qkvz_proj and ba_proj". The diff centers on `python/sglang/srt/models/qwen3_next.py`. PR body context: ## Motivation This PR is to fuse Qwen3-Next GDN's qkvz_proj and ba_proj with MergedColumnParallelLinear in order to improve performance. TTFT speedup 2.6%. (Stably) E2E throughp...
- Key implementation: `python/sglang/srt/models/qwen3_next.py` modified +83/-11 (94 lines); hunks: -20,6 +20,7; -245,28 +246,38 @@ def __init__(; symbols: __init__, fix_query_key_value_ordering, _make_packed_weight_loader, weight_loader, touching `__init__, fix_query_key_value_ordering, _make_packed_weight_loader`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_next.py` modified +83/-11 (94 lines); hunks: -20,6 +20,7; -245,28 +246,38 @@ def __init__(; symbols: __init__, fix_query_key_value_ordering, _make_packed_weight_loader, weight_loader
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -20,6 +20,7 @@
+    MergedColumnParallelLinear,
@@ -245,28 +246,38 @@ def __init__(
-        projection_size_qkvz = self.key_dim * 2 + self.value_dim * 2
-        projection_size_ba = self.num_v_heads * 2
-        self.in_proj_qkvz = ColumnParallelLinear(
-            input_size=self.hidden_size,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +83/-11
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/linear.py`, `python/sglang/srt/models/qwen3_next.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21019 - [Qwen3.5] Fuse split/reshape/cat ops in GDN projection with Triton kernel

- Link: https://github.com/sgl-project/sglang/pull/21019
- Status/date: merged / 2026-03-23
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +597/-202, 953 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR optimizes an inference path or backend selection. Title: "[Qwen3.5] Fuse split/reshape/cat ops in GDN projection with Triton kernel". The diff centers on `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_next.py`, `python/sglang/jit_kernel/triton/gdn_fused_proj.py`. PR body context: ## Motivation In PR https://github.com/sgl-project/sglang/pull/19321 we fused Qwen3-Next GDN's qkvz_proj and ba_proj. This PR is a follow up. The background that Qwen3-Next and...
- Key implementation: `python/sglang/srt/models/qwen3_5.py` modified +285/-65 (350 lines); hunks: -20,6 +20,11; -54,6 +59,10; symbols: __init__, touching `__init__`; `python/sglang/srt/models/qwen3_next.py` modified +2/-137 (139 lines); hunks: -3,6 +3,7; -55,6 +56,7; symbols: fused_qkvzba_split_reshape_cat_kernel, fused_qkvzba_split_reshape_cat, Qwen3GatedDeltaNet, __init__, touching `fused_qkvzba_split_reshape_cat_kernel, fused_qkvzba_split_reshape_cat, Qwen3GatedDeltaNet`; `python/sglang/jit_kernel/triton/gdn_fused_proj.py` added +310/-0 (310 lines); hunks: -0,0 +1,310; symbols: fused_qkvzba_split_reshape_cat_kernel, fused_qkvzba_split_reshape_cat, fused_qkvzba_split_reshape_cat_contiguous_kernel, fused_qkvzba_split_reshape_cat_contiguous, touching `fused_qkvzba_split_reshape_cat_kernel, fused_qkvzba_split_reshape_cat, fused_qkvzba_split_reshape_cat_contiguous_kernel`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` modified +285/-65 (350 lines); hunks: -20,6 +20,11; -54,6 +59,10; symbols: __init__
  - `python/sglang/srt/models/qwen3_next.py` modified +2/-137 (139 lines); hunks: -3,6 +3,7; -55,6 +56,7; symbols: fused_qkvzba_split_reshape_cat_kernel, fused_qkvzba_split_reshape_cat, Qwen3GatedDeltaNet, __init__
  - `python/sglang/jit_kernel/triton/gdn_fused_proj.py` added +310/-0 (310 lines); hunks: -0,0 +1,310; symbols: fused_qkvzba_split_reshape_cat_kernel, fused_qkvzba_split_reshape_cat, fused_qkvzba_split_reshape_cat_contiguous_kernel, fused_qkvzba_split_reshape_cat_contiguous
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -20,6 +20,11 @@
+import triton
+from sglang.jit_kernel.triton.gdn_fused_proj import (
+    fused_qkvzba_split_reshape_cat_contiguous,
+)
@@ -54,6 +59,10 @@
+from sglang.srt.layers.parameter import (
diff -- python/sglang/srt/models/qwen3_next.py
@@ -3,6 +3,7 @@
+import triton
@@ -55,6 +56,7 @@
+from sglang.jit_kernel.triton.gdn_fused_proj import fused_qkvzba_split_reshape_cat
@@ -63,143 +65,6 @@
-import triton
-import triton.language as tl
diff -- python/sglang/jit_kernel/triton/gdn_fused_proj.py
@@ -0,0 +1,310 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +285/-65; `python/sglang/srt/models/qwen3_next.py` modified +2/-137; `python/sglang/jit_kernel/triton/gdn_fused_proj.py` added +310/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/jit_kernel/triton/gdn_fused_proj.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_next.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21313 - bugfix for weight loading for qwen3-next

- Link: https://github.com/sgl-project/sglang/pull/21313
- Status/date: merged / 2026-03-26
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_next.py`; associated commits `0906e45cec97`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-2, 13 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR fixes a launch, loading, parsing, or numerical issue. Title: "bugfix for weight loading for qwen3-next". The diff centers on `python/sglang/srt/models/qwen3_next.py`. PR body context: ## Motivation previously, When loading quantization weight such as w8a8, as weight_loader of self.in_proj_qkvz.weight is denoted by @property, which means it doesn't have a sett...
- Key implementation: `python/sglang/srt/models/qwen3_next.py` modified +2/-2 (4 lines); hunks: -135,10 +135,10 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_next.py` modified +2/-2 (4 lines); hunks: -135,10 +135,10 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -135,10 +135,10 @@ def __init__(
-        self.in_proj_qkvz.weight.weight_loader = self._make_packed_weight_loader(
+        self.in_proj_qkvz.weight._weight_loader = self._make_packed_weight_loader(
-        self.in_proj_ba.weight.weight_loader = self._make_packed_weight_loader(
+        self.in_proj_ba.weight._weight_loader = self._make_packed_weight_loader(
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3_next.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21496 - Revert "bugfix for weight loading for qwen3-next"

- Link: https://github.com/sgl-project/sglang/pull/21496
- Status/date: merged / 2026-03-26
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_next.py`; associated commits `a93065679b63`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-2, 13 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR fixes a launch, loading, parsing, or numerical issue. Title: "Revert "bugfix for weight loading for qwen3-next"". The diff centers on `python/sglang/srt/models/qwen3_next.py`. PR body context: Reverts sgl-project/sglang#21313
- Key implementation: `python/sglang/srt/models/qwen3_next.py` modified +2/-2 (4 lines); hunks: -135,10 +135,10 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_next.py` modified +2/-2 (4 lines); hunks: -135,10 +135,10 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -135,10 +135,10 @@ def __init__(
-        self.in_proj_qkvz.weight._weight_loader = self._make_packed_weight_loader(
+        self.in_proj_qkvz.weight.weight_loader = self._make_packed_weight_loader(
-        self.in_proj_ba.weight._weight_loader = self._make_packed_weight_loader(
+        self.in_proj_ba.weight.weight_loader = self._make_packed_weight_loader(
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3_next.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21662 - [Fix] Fix weight_loader property assignment for qwen3-next FP8 models

- Link: https://github.com/sgl-project/sglang/pull/21662
- Status/date: merged / 2026-03-30
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_next.py`; associated commits `62a63eeff76d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +17/-4, 36 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Fix] Fix weight_loader property assignment for qwen3-next FP8 models". The diff centers on `python/sglang/srt/models/qwen3_next.py`. PR body context: ## Summary - Fix `AttributeError: property 'weight_loader' of 'ModelWeightParameter' object has no setter` when loading Qwen3-Coder-Next-FP8 weights - `BasevLLMParameter.weight_...
- Key implementation: `python/sglang/srt/models/qwen3_next.py` modified +17/-4 (21 lines); hunks: -135,11 +135,11 @@ def __init__(; -216,6 +216,19 @@ def __init__(; symbols: __init__, _override_weight_loader, _make_packed_weight_loader, touching `__init__, _override_weight_loader, _make_packed_weight_loader`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_next.py` modified +17/-4 (21 lines); hunks: -135,11 +135,11 @@ def __init__(; -216,6 +216,19 @@ def __init__(; symbols: __init__, _override_weight_loader, _make_packed_weight_loader
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -135,11 +135,11 @@ def __init__(
-        self.in_proj_qkvz.weight.weight_loader = self._make_packed_weight_loader(
-            self.in_proj_qkvz
+        self._override_weight_loader(
+            self.in_proj_qkvz, self._make_packed_weight_loader(self.in_proj_qkvz)
-        self.in_proj_ba.weight.weight_loader = self._make_packed_weight_loader(
-            self.in_proj_ba
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +17/-4
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3_next.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21684 - [bugfix] fix Qwen3-next memory leak

- Link: https://github.com/sgl-project/sglang/pull/21684
- Status/date: open / 2026-03-30
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +2/-2, 18 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR adds or enables a model support/runtime surface. Title: "[bugfix] fix Qwen3-next memory leak". The diff centers on `python/sglang/srt/mem_cache/allocator.py`, `python/sglang/srt/mem_cache/memory_pool.py`. PR body context: ## Motivation ## Modifications ## Accuracy Tests ## Speed Tests and Profiling ## Checklist - [ ] Format your code according to the Format code with pre-commit. - [ ] Add unit te...
- Key implementation: `python/sglang/srt/mem_cache/allocator.py` modified +1/-1 (2 lines); hunks: -150,7 +150,7 @@ def alloc(self, need_size: int):; symbols: alloc, free, touching `alloc, free`; `python/sglang/srt/mem_cache/memory_pool.py` modified +1/-1 (2 lines); hunks: -356,7 +356,7 @@ def alloc(self, need_size: int) -> Optional[torch.Tensor]:; symbols: alloc, free, touching `alloc, free`.
- Code diff details:
  - `python/sglang/srt/mem_cache/allocator.py` modified +1/-1 (2 lines); hunks: -150,7 +150,7 @@ def alloc(self, need_size: int):; symbols: alloc, free
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +1/-1 (2 lines); hunks: -356,7 +356,7 @@ def alloc(self, need_size: int) -> Optional[torch.Tensor]:; symbols: alloc, free
- Key code excerpts:

```diff
diff -- python/sglang/srt/mem_cache/allocator.py
@@ -150,7 +150,7 @@ def alloc(self, need_size: int):
-        return select_index
+        return select_index.clone()
diff -- python/sglang/srt/mem_cache/memory_pool.py
@@ -356,7 +356,7 @@ def alloc(self, need_size: int) -> Optional[torch.Tensor]:
-        return select_index
+        return select_index.clone()
```

- Reviewed files:
  - runtime: `python/sglang/srt/mem_cache/allocator.py` modified +1/-1; `python/sglang/srt/mem_cache/memory_pool.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/mem_cache/allocator.py`, `python/sglang/srt/mem_cache/memory_pool.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21698 - [npu]fix: qwen3-next w8a8 precision bugs

- Link: https://github.com/sgl-project/sglang/pull/21698
- Status/date: open / 2026-03-30
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +22/-5, 41 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[npu]fix: qwen3-next w8a8 precision bugs". The diff centers on `python/sglang/srt/models/qwen3_next.py`. PR body context: ## Motivation The change at https://github.com/sgl-project/sglang/pull/19321 caused qwen3-next to fail in running the w8a8 model on NPU due to incorrect loading of the in_proj_q...
- Key implementation: `python/sglang/srt/models/qwen3_next.py` modified +22/-5 (27 lines); hunks: -65,6 +65,14; -223,11 +231,20 @@ def _override_weight_loader(module, new_loader):; symbols: Qwen3GatedDeltaNet, __init__, _override_weight_loader, _make_packed_weight_loader, touching `Qwen3GatedDeltaNet, __init__, _override_weight_loader`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_next.py` modified +22/-5 (27 lines); hunks: -65,6 +65,14; -223,11 +231,20 @@ def _override_weight_loader(module, new_loader):; symbols: Qwen3GatedDeltaNet, __init__, _override_weight_loader, _make_packed_weight_loader
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -65,6 +65,14 @@
+if _is_npu:
+    from sgl_kernel_npu.fla.utils import (
+        fused_qkvzba_split_reshape_cat as fused_qkvzba_split_reshape_cat_npu,
+    )
+    fused_qkvzba_split_reshape_cat = fused_qkvzba_split_reshape_cat_npu
@@ -223,11 +231,20 @@ def _override_weight_loader(module, new_loader):
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +22/-5
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3_next.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22073 - [Feature] Adding Qwen3-asr Model Support

- Link: https://github.com/sgl-project/sglang/pull/22073
- Status/date: merged / 2026-04-07
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +571/-11, 689 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR adds or enables a model support/runtime surface. Title: "[Feature] Adding Qwen3-asr Model Support". The diff centers on `python/sglang/srt/models/qwen3_asr.py`, `python/sglang/srt/configs/qwen3_asr.py`, `python/sglang/srt/multimodal/processors/qwen3_asr.py`. PR body context: ## Motivation Issue : https://github.com/sgl-project/sglang/issues/22025 This PR adds support so users can serve Qwen3-ASR via the existing `/v1/audio/transcriptions` endpoint....
- Key implementation: `python/sglang/srt/models/qwen3_asr.py` added +199/-0 (199 lines); hunks: -0,0 +1,199; symbols: Qwen3ASRForConditionalGeneration, __init__, pad_input_ids, get_audio_feature, touching `Qwen3ASRForConditionalGeneration, __init__, pad_input_ids`; `python/sglang/srt/configs/qwen3_asr.py` added +172/-0 (172 lines); hunks: -0,0 +1,172; symbols: Qwen3ASRThinkerConfig, __init__, Qwen3ASRConfig, get_text_config, touching `Qwen3ASRThinkerConfig, __init__, Qwen3ASRConfig`; `python/sglang/srt/multimodal/processors/qwen3_asr.py` added +95/-0 (95 lines); hunks: -0,0 +1,95; symbols: Qwen3ASRMultimodalProcessor, __init__, _build_transcription_prompt, compute_mrope_positions, touching `Qwen3ASRMultimodalProcessor, __init__, _build_transcription_prompt`; `python/sglang/srt/entrypoints/openai/serving_transcription.py` modified +57/-7 (64 lines); hunks: -50,12 +50,22; -71,6 +81,27 @@ def _convert_to_internal_request(; symbols: _detect_model_family, OpenAIServingTranscription, __init__, _request_id_prefix, touching `_detect_model_family, OpenAIServingTranscription, __init__`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_asr.py` added +199/-0 (199 lines); hunks: -0,0 +1,199; symbols: Qwen3ASRForConditionalGeneration, __init__, pad_input_ids, get_audio_feature
  - `python/sglang/srt/configs/qwen3_asr.py` added +172/-0 (172 lines); hunks: -0,0 +1,172; symbols: Qwen3ASRThinkerConfig, __init__, Qwen3ASRConfig, get_text_config
  - `python/sglang/srt/multimodal/processors/qwen3_asr.py` added +95/-0 (95 lines); hunks: -0,0 +1,95; symbols: Qwen3ASRMultimodalProcessor, __init__, _build_transcription_prompt, compute_mrope_positions
  - `python/sglang/srt/entrypoints/openai/serving_transcription.py` modified +57/-7 (64 lines); hunks: -50,12 +50,22; -71,6 +81,27 @@ def _convert_to_internal_request(; symbols: _detect_model_family, OpenAIServingTranscription, __init__, _request_id_prefix
  - `docs/supported_models/text_generation/multimodal_language_models.md` modified +29/-0 (29 lines); hunks: -51,9 +51,38 @@ in the GitHub search bar.
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_asr.py
@@ -0,0 +1,199 @@
+"""Qwen3-ASR model compatible with HuggingFace weights"""
+import logging
+from typing import Any, Iterable, List, Optional, Tuple
+import torch
+import torch.nn as nn
+from sglang.srt.configs.qwen3_asr import Qwen3ASRConfig
diff -- python/sglang/srt/configs/qwen3_asr.py
@@ -0,0 +1,172 @@
+import torch
+from transformers import (
+    AutoConfig,
+    AutoFeatureExtractor,
+    AutoTokenizer,
+    PretrainedConfig,
diff -- python/sglang/srt/multimodal/processors/qwen3_asr.py
@@ -0,0 +1,95 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_asr.py` added +199/-0; `python/sglang/srt/configs/qwen3_asr.py` added +172/-0; `python/sglang/srt/multimodal/processors/qwen3_asr.py` added +95/-0; `python/sglang/srt/entrypoints/openai/serving_transcription.py` modified +57/-7; `python/sglang/srt/configs/model_config.py` modified +12/-2; `python/sglang/srt/configs/__init__.py` modified +2/-0
  - docs: `docs/supported_models/text_generation/multimodal_language_models.md` modified +29/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/__init__.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/configs/qwen3_asr.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22358 - Enable DFLASH support for additional model backends

- Link: https://github.com/sgl-project/sglang/pull/22358
- Status/date: merged / 2026-04-09
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +152/-5, 299 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR adds or enables a model support/runtime surface. Title: "Enable DFLASH support for additional model backends". The diff centers on `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/models/qwen3_next.py`. PR body context: ## Summary Enable DFLASH for additional supported models from the z-lab collection: https://huggingface.co/collections/z-lab/dflash Based on #20547, landing this early to enable...
- Key implementation: `python/sglang/srt/models/qwen3_5.py` modified +34/-5 (39 lines); hunks: -574,8 +574,15 @@ def forward(; -825,10 +832,16 @@ def forward(; symbols: forward, get_layer, get_input_embeddings, set_dflash_layers_to_capture, touching `forward, get_layer, get_input_embeddings`; `python/sglang/srt/models/kimi_k25.py` modified +24/-0 (24 lines); hunks: -849,6 +849,30 @@ def set_eagle3_layers_to_capture(; symbols: set_eagle3_layers_to_capture, set_dflash_layers_to_capture, get_input_embeddings, lm_head, touching `set_eagle3_layers_to_capture, set_dflash_layers_to_capture, get_input_embeddings`; `python/sglang/srt/models/qwen3_next.py` modified +20/-0 (20 lines); hunks: -813,6 +813,11 @@ def set_eagle3_layers_to_capture(self, layers_to_capture: l...; -947,6 +952,9 @@ def forward(; symbols: set_eagle3_layers_to_capture, set_dflash_layers_to_capture, forward, get_embed_and_head, touching `set_eagle3_layers_to_capture, set_dflash_layers_to_capture, forward`; `python/sglang/srt/models/qwen3_moe.py` modified +17/-0 (17 lines); hunks: -924,6 +924,11 @@ def __init__(; -1079,6 +1084,18 @@ def set_eagle3_layers_to_capture(self, layer_ids: Optiona...; symbols: __init__, set_dflash_layers_to_capture, Qwen3MoeForCausalLM, set_eagle3_layers_to_capture, touching `__init__, set_dflash_layers_to_capture, Qwen3MoeForCausalLM`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` modified +34/-5 (39 lines); hunks: -574,8 +574,15 @@ def forward(; -825,10 +832,16 @@ def forward(; symbols: forward, get_layer, get_input_embeddings, set_dflash_layers_to_capture
  - `python/sglang/srt/models/kimi_k25.py` modified +24/-0 (24 lines); hunks: -849,6 +849,30 @@ def set_eagle3_layers_to_capture(; symbols: set_eagle3_layers_to_capture, set_dflash_layers_to_capture, get_input_embeddings, lm_head
  - `python/sglang/srt/models/qwen3_next.py` modified +20/-0 (20 lines); hunks: -813,6 +813,11 @@ def set_eagle3_layers_to_capture(self, layers_to_capture: l...; -947,6 +952,9 @@ def forward(; symbols: set_eagle3_layers_to_capture, set_dflash_layers_to_capture, forward, get_embed_and_head
  - `python/sglang/srt/models/qwen3_moe.py` modified +17/-0 (17 lines); hunks: -924,6 +924,11 @@ def __init__(; -1079,6 +1084,18 @@ def set_eagle3_layers_to_capture(self, layer_ids: Optiona...; symbols: __init__, set_dflash_layers_to_capture, Qwen3MoeForCausalLM, set_eagle3_layers_to_capture
  - `python/sglang/srt/models/qwen3_vl.py` modified +16/-0 (16 lines); hunks: -1122,6 +1122,7 @@ def __init__(; -1246,19 +1247,34 @@ def forward(; symbols: __init__, forward, set_dflash_layers_to_capture, load_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -574,8 +574,15 @@ def forward(
-        hidden_states, residual = self.layer_communicator.prepare_attn(
-            hidden_states, residual, forward_batch
+        hidden_states, residual = (
+            self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
+                hidden_states,
+                residual,
diff -- python/sglang/srt/models/kimi_k25.py
@@ -849,6 +849,30 @@ def set_eagle3_layers_to_capture(
+    def set_dflash_layers_to_capture(self, layer_ids: List[int]) -> None:
+        """Set the layers to capture for DFLASH draft model training."""
+        if not hasattr(self.language_model, "set_dflash_layers_to_capture"):
+            raise AttributeError(
+                "language_model does not support DFLASH layer capture."
+            )
diff -- python/sglang/srt/models/qwen3_next.py
@@ -813,6 +813,11 @@ def set_eagle3_layers_to_capture(self, layers_to_capture: list[int]):
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +34/-5; `python/sglang/srt/models/kimi_k25.py` modified +24/-0; `python/sglang/srt/models/qwen3_next.py` modified +20/-0; `python/sglang/srt/models/qwen3_moe.py` modified +17/-0; `python/sglang/srt/models/qwen3_vl.py` modified +16/-0; `python/sglang/srt/models/gpt_oss.py` modified +15/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/gpt_oss.py`, `python/sglang/srt/models/kimi_k25.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22458 - Fix NCCL AllGather hanging issue for Qwen3 Next MTP

- Link: https://github.com/sgl-project/sglang/pull/22458
- Status/date: merged / 2026-04-10
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +38/-0, 68 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR fixes a launch, loading, parsing, or numerical issue. Title: "Fix NCCL AllGather hanging issue for Qwen3 Next MTP". The diff centers on `python/sglang/srt/speculative/eagle_info.py`, `python/sglang/srt/speculative/eagle_info_v2.py`. PR body context: ## Motivation Fix hanging issue (AllGather size mismatch) in EAGLE speculative decoding (both V1 and V2) when running with TP>1 and non-greedy sampling. (https://github.com/sgl-...
- Key implementation: `python/sglang/srt/speculative/eagle_info.py` modified +19/-0 (19 lines); hunks: -7,8 +7,13; -377,6 +382,20 @@ def verify(; symbols: verify, touching `verify`; `python/sglang/srt/speculative/eagle_info_v2.py` modified +19/-0 (19 lines); hunks: -8,6 +8,11; -370,6 +375,20 @@ def sample(; symbols: sample, touching `sample`.
- Code diff details:
  - `python/sglang/srt/speculative/eagle_info.py` modified +19/-0 (19 lines); hunks: -7,8 +7,13; -377,6 +382,20 @@ def verify(; symbols: verify
  - `python/sglang/srt/speculative/eagle_info_v2.py` modified +19/-0 (19 lines); hunks: -8,6 +8,11; -370,6 +375,20 @@ def sample(; symbols: sample
- Key code excerpts:

```diff
diff -- python/sglang/srt/speculative/eagle_info.py
@@ -7,8 +7,13 @@
+from sglang.srt.distributed import get_tp_group
+from sglang.srt.layers.dp_attention import (
+    get_attention_tp_group,
+    is_dp_attention_enabled,
+)
@@ -377,6 +382,20 @@ def verify(
diff -- python/sglang/srt/speculative/eagle_info_v2.py
@@ -8,6 +8,11 @@
+from sglang.srt.distributed import get_tp_group
+from sglang.srt.layers.dp_attention import (
+    get_attention_tp_group,
+    is_dp_attention_enabled,
+)
@@ -370,6 +375,20 @@ def sample(
```

- Reviewed files:
  - runtime: `python/sglang/srt/speculative/eagle_info.py` modified +19/-0; `python/sglang/srt/speculative/eagle_info_v2.py` modified +19/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/speculative/eagle_info.py`, `python/sglang/srt/speculative/eagle_info_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22876 - Fix: Raise ValueError when --enable-mixed-chunk and --mamba-scheduler-strategy extra_buffer cause ac

- Link: https://github.com/sgl-project/sglang/pull/22876
- Status/date: open / 2026-04-15
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +42/-0, 55 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR adds or enables a model support/runtime surface. Title: "Fix: Raise ValueError when --enable-mixed-chunk and --mamba-scheduler-strategy extra_buffer cause ac". The diff centers on `test/registered/unit/server_args/test_server_args.py`, `python/sglang/srt/server_args.py`. PR body context: ## Motivation In recent tests, it was found that when both --enable-mixed-chunk and --mamba-scheduler-strategy extra_buffer are enabled, the model exhibits a drop in accuracy un...
- Key implementation: `test/registered/unit/server_args/test_server_args.py` modified +35/-0 (35 lines); hunks: -477,5 +477,40 @@ def test_external_corpus_max_tokens_must_be_positive(self):; symbols: test_external_corpus_max_tokens_must_be_positive, TestMambaRadixCacheArgs, _make_dummy_mamba_args, test_mamba_extra_buffer_rejects_mixed_chunk_before_cuda_check, touching `test_external_corpus_max_tokens_must_be_positive, TestMambaRadixCacheArgs, _make_dummy_mamba_args`; `python/sglang/srt/server_args.py` modified +7/-0 (7 lines); hunks: -2250,6 +2250,13 @@ def _handle_mamba_radix_cache(; symbols: _handle_mamba_radix_cache, touching `_handle_mamba_radix_cache`.
- Code diff details:
  - `test/registered/unit/server_args/test_server_args.py` modified +35/-0 (35 lines); hunks: -477,5 +477,40 @@ def test_external_corpus_max_tokens_must_be_positive(self):; symbols: test_external_corpus_max_tokens_must_be_positive, TestMambaRadixCacheArgs, _make_dummy_mamba_args, test_mamba_extra_buffer_rejects_mixed_chunk_before_cuda_check
  - `python/sglang/srt/server_args.py` modified +7/-0 (7 lines); hunks: -2250,6 +2250,13 @@ def _handle_mamba_radix_cache(; symbols: _handle_mamba_radix_cache
- Key code excerpts:

```diff
diff -- test/registered/unit/server_args/test_server_args.py
@@ -477,5 +477,40 @@ def test_external_corpus_max_tokens_must_be_positive(self):
+class TestMambaRadixCacheArgs(CustomTestCase):
+    def _make_dummy_mamba_args(self, **overrides) -> ServerArgs:
+        args = ServerArgs(model_path="dummy")
+        args.mamba_scheduler_strategy = "extra_buffer"
+        args.disable_radix_cache = False
+        args.enable_mixed_chunk = False
diff -- python/sglang/srt/server_args.py
@@ -2250,6 +2250,13 @@ def _handle_mamba_radix_cache(
+            if self.enable_mixed_chunk:
+                raise ValueError(
+                    "mamba extra_buffer is not compatible with --enable-mixed-chunk "
+                    "because this combination may reduce model accuracy. "
+                    "Please disable --enable-mixed-chunk or use "
+                    "--mamba-scheduler-strategy no_buffer instead."
```

- Reviewed files:
  - tests: `test/registered/unit/server_args/test_server_args.py` modified +35/-0
  - runtime: `python/sglang/srt/server_args.py` modified +7/-0
- Risk and verification: The diff ships test coverage in `test/registered/unit/server_args/test_server_args.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #23075 - [Fix] Mixed chunk query_start_loc and mamba_cache_indices to the prefill-only prefix so that the tracking helpers see a consistent, prefill-only view.

- Link: https://github.com/sgl-project/sglang/pull/23075
- Status/date: open / 2026-04-17
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +51/-13, 128 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR adds or enables a model support/runtime surface. Title: "[Fix] Mixed chunk query_start_loc and mamba_cache_indices to the prefill-only prefix so that the tracking helpers see a consistent, prefill-only view.". The diff centers on `python/sglang/srt/layers/attention/mamba/mamba2_metadata.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/managers/schedule_batch.py`. PR body context: ## Motivation In recent tests, it was found that when both --enable-mixed-chunk and --mamba-scheduler-strategy extra_buffer are enabled, the model exhibits a drop in accuracy un...
- Key implementation: `python/sglang/srt/layers/attention/mamba/mamba2_metadata.py` modified +19/-6 (25 lines); hunks: -195,14 +195,21 @@ def prepare_mixed(; -213,6 +220,12 @@ def prepare_mixed(; symbols: prepare_mixed, touching `prepare_mixed`; `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +21/-2 (23 lines); hunks: -1,4 +1,5; -205,16 +206,34 @@ def _forward_metadata(self, forward_batch: ForwardBatch):; symbols: _forward_metadata, touching `_forward_metadata`; `python/sglang/srt/managers/schedule_batch.py` modified +11/-5 (16 lines); hunks: -1946,7 +1946,7 @@ def mix_with_running(self, running_batch: "ScheduleBatch"):; -2341,7 +2341,13 @@ def filter_batch(; symbols: mix_with_running, filter_batch, merge_batch, touching `mix_with_running, filter_batch, merge_batch`.
- Code diff details:
  - `python/sglang/srt/layers/attention/mamba/mamba2_metadata.py` modified +19/-6 (25 lines); hunks: -195,14 +195,21 @@ def prepare_mixed(; -213,6 +220,12 @@ def prepare_mixed(; symbols: prepare_mixed
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +21/-2 (23 lines); hunks: -1,4 +1,5; -205,16 +206,34 @@ def _forward_metadata(self, forward_batch: ForwardBatch):; symbols: _forward_metadata
  - `python/sglang/srt/managers/schedule_batch.py` modified +11/-5 (16 lines); hunks: -1946,7 +1946,7 @@ def mix_with_running(self, running_batch: "ScheduleBatch"):; -2341,7 +2341,13 @@ def filter_batch(; symbols: mix_with_running, filter_batch, merge_batch
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/mamba/mamba2_metadata.py
@@ -195,14 +195,21 @@ def prepare_mixed(
-        num_prefills = len(forward_batch.extend_seq_lens)
-        num_prefill_tokens = forward_batch.extend_num_tokens
-        num_decodes = len(forward_batch.seq_lens) - num_prefills
+        # In MIXED mode (enable_mixed_chunk),We derive num_prefills by subtracting the number
+        # of decode requests (= total seq_lens count - extend_seq_lens count) from
+        # the total extend count.  In non-MIXED mode the two counts are equal.
diff -- python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py
@@ -1,4 +1,5 @@
+import types
@@ -205,16 +206,34 @@ def _forward_metadata(self, forward_batch: ForwardBatch):
+                    # In MIXED mode (enable_mixed_chunk), slice query_start_loc and
+                    # mamba_cache_indices to the prefill-only prefix so that the
+                    # tracking helpers see a consistent, prefill-only view.
+                    if forward_batch.forward_mode.is_mixed():
diff -- python/sglang/srt/managers/schedule_batch.py
@@ -1946,7 +1946,7 @@ def mix_with_running(self, running_batch: "ScheduleBatch"):
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/mamba/mamba2_metadata.py` modified +19/-6; `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +21/-2; `python/sglang/srt/managers/schedule_batch.py` modified +11/-5
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/layers/attention/mamba/mamba2_metadata.py`, `python/sglang/srt/managers/schedule_batch.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22664 - Qwen3next flashinfer allreduce auto enable

- Link: https://github.com/sgl-project/sglang/pull/22664
- Status/date: merged / 2026-04-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-1, 18 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR adds or enables a model support/runtime surface. Title: "Qwen3next flashinfer allreduce auto enable". The diff centers on `python/sglang/srt/server_args.py`. PR body context: Made with @Codex ## Summary Enable FlashInfer allreduce fusion by default for `Qwen3NextForCausalLM` on supported single-node SM90/SM100 TP runs. ## Why `Qwen/Qwen3-Coder-Next`...
- Key implementation: `python/sglang/srt/server_args.py` modified +3/-1 (4 lines); hunks: -2171,7 +2171,8 @@ def _handle_model_specific_adjustments(self):; -2189,6 +2190,7 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments, touching `_handle_model_specific_adjustments`.
- Code diff details:
  - `python/sglang/srt/server_args.py` modified +3/-1 (4 lines); hunks: -2171,7 +2171,8 @@ def _handle_model_specific_adjustments(self):; -2189,6 +2190,7 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments
- Key code excerpts:

```diff
diff -- python/sglang/srt/server_args.py
@@ -2171,7 +2171,8 @@ def _handle_model_specific_adjustments(self):
-        # for models with explicit support (DeepseekV3, GptOss, Glm4Moe, Qwen3Moe)
+        # for models with explicit support (DeepseekV3, GptOss, Glm4Moe,
+        # Qwen3/Qwen3Next/Qwen3.5 MoE families)
@@ -2189,6 +2190,7 @@ def _handle_model_specific_adjustments(self):
+                "Qwen3NextForCausalLM",
```

- Reviewed files:
  - runtime: `python/sglang/srt/server_args.py` modified +3/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/server_args.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23273 - [NVIDIA] [GDN] Enable FlashInfer MTP verify on SM100+ (Blackwell)

- Link: https://github.com/sgl-project/sglang/pull/23273
- Status/date: open / 2026-04-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +54/-22, 154 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR adds or enables a model support/runtime surface. Title: "[NVIDIA] [GDN] Enable FlashInfer MTP verify on SM100+ (Blackwell)". The diff centers on `python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py`, `python/sglang/srt/server_args.py`. PR body context: ## [GDN] Enable FlashInfer MTP verify on SM100+ (Blackwell) co-authored by @YAMY1234 (main contributor) ### Summary Enables FlashInfer GDN MTP (speculative decoding) verify on S...
- Key implementation: `python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py` modified +51/-16 (67 lines); hunks: -3,7 +3,7; -27,14 +27,15; symbols: _get_flashinfer_gdn_kernels, FlashInferGDNKernel, __init__, touching `_get_flashinfer_gdn_kernels, FlashInferGDNKernel, __init__`; `python/sglang/srt/server_args.py` modified +3/-6 (9 lines); hunks: -2715,17 +2715,14 @@ def _handle_mamba_backend(self):; symbols: _handle_mamba_backend, _handle_linear_attn_backend, touching `_handle_mamba_backend, _handle_linear_attn_backend`.
- Code diff details:
  - `python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py` modified +51/-16 (67 lines); hunks: -3,7 +3,7; -27,14 +27,15; symbols: _get_flashinfer_gdn_kernels, FlashInferGDNKernel, __init__
  - `python/sglang/srt/server_args.py` modified +3/-6 (9 lines); hunks: -2715,17 +2715,14 @@ def _handle_mamba_backend(self):; symbols: _handle_mamba_backend, _handle_linear_attn_backend
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py
@@ -3,7 +3,7 @@
-SM100+ (Blackwell+): decode-only with bf16 state.  More support on the way.
+SM100+ (Blackwell+): decode + MTP verify with bf16 state.  Prefill on the way.
@@ -27,14 +27,15 @@
+_flashinfer_gated_delta_rule_mtp_bf16 = None
-    Returns (available, prefill_fn, mtp_fn, decode_fn).
+    Returns (available, prefill_fn, mtp_fn, decode_fn, mtp_bf16_fn).
diff -- python/sglang/srt/server_args.py
@@ -2715,17 +2715,14 @@ def _handle_mamba_backend(self):
-        # SM100+: default to FlashInfer GDN decode when the user hasn't
-        # explicitly chosen a decode backend and mamba-ssm-dtype is bf16
-        # (required by FlashInfer GDN on SM100+).
+        # SM100+: default to FlashInfer GDN decode (and MTP verify, via pool API)
+        # when the user hasn't explicitly chosen a decode backend and
+        # mamba-ssm-dtype is bf16 (required by FlashInfer GDN on SM100+).
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py` modified +51/-16; `python/sglang/srt/server_args.py` modified +3/-6
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py`, `python/sglang/srt/server_args.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23474 - [Bugfix] Try to fix --cpu-offload-gb on hybrid linear-attn models

- Link: https://github.com/sgl-project/sglang/pull/23474
- Status/date: open / 2026-04-22
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +284/-8, 330 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For Qwen3 Next, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Bugfix] Try to fix --cpu-offload-gb on hybrid linear-attn models". The diff centers on `test/registered/unit/utils/test_offloader_tied_params.py`, `python/sglang/srt/utils/offloader.py`. PR body context: ## Motivation Fixes #23150. `--cpu-offload-gb > 0` was broken on hybrid linear-attention models (Qwen3-Next, Qwen3.5, Kimi-Linear): the first `/v1/chat/completions` request rais...
- Key implementation: `test/registered/unit/utils/test_offloader_tied_params.py` added +199/-0 (199 lines); hunks: -0,0 +1,199; symbols: _TiedChild, __init__, forward, _TiedParent, touching `_TiedChild, __init__, forward`; `python/sglang/srt/utils/offloader.py` modified +85/-8 (93 lines); hunks: -1,7 +1,7; -106,16 +106,52 @@ def maybe_offload_to_cpu(self, module: torch.nn.Module) ->...; symbols: maybe_offload_to_cpu, forward, touching `maybe_offload_to_cpu, forward`.
- Code diff details:
  - `test/registered/unit/utils/test_offloader_tied_params.py` added +199/-0 (199 lines); hunks: -0,0 +1,199; symbols: _TiedChild, __init__, forward, _TiedParent
  - `python/sglang/srt/utils/offloader.py` modified +85/-8 (93 lines); hunks: -1,7 +1,7; -106,16 +106,52 @@ def maybe_offload_to_cpu(self, module: torch.nn.Module) ->...; symbols: maybe_offload_to_cpu, forward
- Key code excerpts:

```diff
diff -- test/registered/unit/utils/test_offloader_tied_params.py
@@ -0,0 +1,199 @@
+"""Tests for OffloaderV1 with tied parameters and view aliases (see issue #23150).
+Two failure modes caused the Qwen3-Next / Qwen3.5 CPU-offload regression:
+1. **Tied parameters**: a single nn.Parameter is registered under both a parent
+   and a child module (Qwen3GatedDeltaNet + RadixLinearAttention share
+   ``A_log`` / ``dt_bias``). state_dict() then lists the same tensor under
+   multiple keys, and functional_call(..., tie_weights=True) rejects it when
diff -- python/sglang/srt/utils/offloader.py
@@ -1,7 +1,7 @@
-from typing import Callable, Generator, List, Optional
+from typing import Callable, Dict, Generator, List, Optional
@@ -106,16 +106,52 @@ def maybe_offload_to_cpu(self, module: torch.nn.Module) -> torch.nn.Module:
+        # Record tensor views that alias each parameter's *original* storage
+        # BEFORE we rebind .data to pinned CPU memory. Some hybrid linear-attn
+        # models (e.g. Qwen3-Next) cache such views, which would otherwise point
```

- Reviewed files:
  - tests: `test/registered/unit/utils/test_offloader_tied_params.py` added +199/-0
  - runtime: `python/sglang/srt/utils/offloader.py` modified +85/-8
- Risk and verification: The diff ships test coverage in `test/registered/unit/utils/test_offloader_tied_params.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- This version rejects title-only PR lists; every PR must include trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
