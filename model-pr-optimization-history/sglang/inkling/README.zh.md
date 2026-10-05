# SGLang Inkling 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `docs/cookbook/autoregressive/ThinkingMachines/Inkling-Small.mdx` | [#34250](https://github.com/sgl-project/sglang/pull/34250) |
| `docs/cookbook/autoregressive/ThinkingMachines/Inkling.mdx` | 无直接 PR 号提交 |
| `docs/src/snippets/configs/thinkingmachines/inkling-benchmarks.jsx` | 无直接 PR 号提交 |
| `docs/src/snippets/configs/thinkingmachines/inkling-small-benchmarks.jsx` | 无直接 PR 号提交 |
| `docs/src/snippets/configs/thinkingmachines/inkling-small.jsx` | [#34250](https://github.com/sgl-project/sglang/pull/34250) |
| `docs/src/snippets/configs/thinkingmachines/inkling.jsx` | 无直接 PR 号提交 |
| `python/sglang/kernels/jit/csrc/inkling/causal_conv1d.cuh` | 无直接 PR 号提交 |
| `python/sglang/kernels/jit/csrc/inkling/draft_extend_sconv.cuh` | 无直接 PR 号提交 |
| `python/sglang/kernels/jit/csrc/inkling/fused_decode_update.cuh` | 无直接 PR 号提交 |
| `python/sglang/kernels/jit/csrc/inkling/gather_scatter_sconv.cuh` | 无直接 PR 号提交 |
| `python/sglang/kernels/jit/csrc/inkling/inkling_all_reduce.cuh` | 无直接 PR 号提交 |
| `python/sglang/kernels/jit/csrc/inkling/inkling_ar_barrier.cuh` | 无直接 PR 号提交 |
| `python/sglang/kernels/jit/csrc/inkling/inkling_ar_fused_decode.cuh` | 无直接 PR 号提交 |
| `python/sglang/kernels/jit/csrc/inkling/inkling_ar_scattered_sconv.cuh` | 无直接 PR 号提交 |
| `python/sglang/kernels/jit/csrc/inkling/inkling_attn_prologue_fused.cuh` | 无直接 PR 号提交 |
| `python/sglang/kernels/jit/csrc/inkling/inkling_rel_proj.cuh` | 无直接 PR 号提交 |
| `python/sglang/kernels/jit/csrc/inkling/inkling_row_scale.cuh` | 无直接 PR 号提交 |
| `python/sglang/kernels/jit/csrc/inkling/update_sconv_cache.cuh` | 无直接 PR 号提交 |
| `python/sglang/kernels/jit/csrc/moe/inkling_gate_topk_renorm.cuh` | 无直接 PR 号提交 |
| `python/sglang/kernels/ops/attention/inkling_attn_prologue.py` | 无直接 PR 号提交 |
| `python/sglang/kernels/ops/communication/inkling_all_reduce.py` | 无直接 PR 号提交 |
| `python/sglang/kernels/ops/communication/inkling_ar_fused.py` | 无直接 PR 号提交 |
| `python/sglang/kernels/ops/communication/inkling_ar_scattered_sconv.py` | 无直接 PR 号提交 |
| `python/sglang/kernels/ops/gemm/inkling_rel_proj.py` | 无直接 PR 号提交 |
| `python/sglang/kernels/ops/mamba/inkling_sconv.py` | 无直接 PR 号提交 |
| `python/sglang/kernels/ops/moe/inkling_gate_topk_renorm.py` | 无直接 PR 号提交 |
| `python/sglang/kernels/ops/moe/inkling_moe.py` | [#33108](https://github.com/sgl-project/sglang/pull/33108), [#33903](https://github.com/sgl-project/sglang/pull/33903) |
| `python/sglang/srt/arg_groups/model_overrides/inkling.py` | 无直接 PR 号提交 |
| `python/sglang/srt/configs/inkling.py` | [#31681](https://github.com/sgl-project/sglang/pull/31681) |
| `python/sglang/srt/function_call/inkling_detector.py` | [#31681](https://github.com/sgl-project/sglang/pull/31681), [#32861](https://github.com/sgl-project/sglang/pull/32861) |
| `python/sglang/srt/layers/attention/linear/inkling_sconv_backend.py` | [#33023](https://github.com/sgl-project/sglang/pull/33023), [#33116](https://github.com/sgl-project/sglang/pull/33116), [#38169](https://github.com/sgl-project/sglang/pull/38169), [#38229](https://github.com/sgl-project/sglang/pull/38229), [#41144](https://github.com/sgl-project/sglang/pull/41144) |
| `python/sglang/srt/lora/trtllm_lora_temp/inkling_dense.py` | [#31681](https://github.com/sgl-project/sglang/pull/31681) |
| `python/sglang/srt/models/inkling.py` | [#31681](https://github.com/sgl-project/sglang/pull/31681), [#31840](https://github.com/sgl-project/sglang/pull/31840), [#33023](https://github.com/sgl-project/sglang/pull/33023) |
| `python/sglang/srt/models/inkling_common/__init__.py` | [#31681](https://github.com/sgl-project/sglang/pull/31681) |
| `python/sglang/srt/models/inkling_common/attn.py` | [#31681](https://github.com/sgl-project/sglang/pull/31681), [#32076](https://github.com/sgl-project/sglang/pull/32076), [#33417](https://github.com/sgl-project/sglang/pull/33417), [#35161](https://github.com/sgl-project/sglang/pull/35161), [#38229](https://github.com/sgl-project/sglang/pull/38229) |
| `python/sglang/srt/models/inkling_common/dense_mlp.py` | [#31681](https://github.com/sgl-project/sglang/pull/31681) |
| `python/sglang/srt/models/inkling_common/hmlp.py` | [#31681](https://github.com/sgl-project/sglang/pull/31681) |
| `python/sglang/srt/models/inkling_common/kernels/__init__.py` | [#31681](https://github.com/sgl-project/sglang/pull/31681) |
| `python/sglang/srt/models/inkling_common/kernels/comm.py` | [#31681](https://github.com/sgl-project/sglang/pull/31681), [#32076](https://github.com/sgl-project/sglang/pull/32076), [#33023](https://github.com/sgl-project/sglang/pull/33023), [#33417](https://github.com/sgl-project/sglang/pull/33417), [#38229](https://github.com/sgl-project/sglang/pull/38229) |
| `python/sglang/srt/models/inkling_common/kernels/sconv.py` | [#31681](https://github.com/sgl-project/sglang/pull/31681), [#33023](https://github.com/sgl-project/sglang/pull/33023) |
| `python/sglang/srt/models/inkling_common/lora.py` | [#31681](https://github.com/sgl-project/sglang/pull/31681) |
| `python/sglang/srt/models/inkling_common/moe.py` | [#31681](https://github.com/sgl-project/sglang/pull/31681), [#33903](https://github.com/sgl-project/sglang/pull/33903) |
| `python/sglang/srt/models/inkling_common/norm.py` | [#31681](https://github.com/sgl-project/sglang/pull/31681) |
| `python/sglang/srt/models/inkling_common/quantization/__init__.py` | [#31681](https://github.com/sgl-project/sglang/pull/31681) |
| `python/sglang/srt/models/inkling_common/quantization/config.py` | [#31681](https://github.com/sgl-project/sglang/pull/31681), [#33750](https://github.com/sgl-project/sglang/pull/33750) |
| `python/sglang/srt/models/inkling_common/quantization/quant.py` | [#31681](https://github.com/sgl-project/sglang/pull/31681) |
| `python/sglang/srt/models/inkling_common/sconv.py` | [#31681](https://github.com/sgl-project/sglang/pull/31681), [#33023](https://github.com/sgl-project/sglang/pull/33023), [#33116](https://github.com/sgl-project/sglang/pull/33116), [#38229](https://github.com/sgl-project/sglang/pull/38229) |
| `python/sglang/srt/models/inkling_common/util.py` | [#31681](https://github.com/sgl-project/sglang/pull/31681) |
| `python/sglang/srt/multimodal/inkling/__init__.py` | [#31681](https://github.com/sgl-project/sglang/pull/31681) |
| `python/sglang/srt/multimodal/inkling/feature_extraction.py` | [#31681](https://github.com/sgl-project/sglang/pull/31681) |
| `python/sglang/srt/multimodal/inkling/image_processing.py` | [#31681](https://github.com/sgl-project/sglang/pull/31681) |
| `python/sglang/srt/multimodal/inkling/image_processing_rust.py` | [#31681](https://github.com/sgl-project/sglang/pull/31681) |
| `python/sglang/srt/multimodal/inkling/processing_inkling.py` | [#31681](https://github.com/sgl-project/sglang/pull/31681) |
| `python/sglang/srt/multimodal/processors/inkling.py` | [#31681](https://github.com/sgl-project/sglang/pull/31681) |
| `python/sglang/srt/parser/inkling_renderer.py` | [#31681](https://github.com/sgl-project/sglang/pull/31681), [#33898](https://github.com/sgl-project/sglang/pull/33898) |
| `python/sglang/srt/parser/inkling_tokenizer.py` | [#31681](https://github.com/sgl-project/sglang/pull/31681) |
| `test/registered/disaggregation/test_disaggregation_inkling_mxfp8.py` | [#35840](https://github.com/sgl-project/sglang/pull/35840) |
| `test/registered/e2e/models/test_inkling.py` | 无直接 PR 号提交 |
| `test/registered/e2e/models/test_inkling_small_nvfp4.py` | 无直接 PR 号提交 |
| `test/registered/e2e/models/test_inkling_unified.py` | 无直接 PR 号提交 |
| `test/registered/e2e/models_large/test_inkling_nvfp4_nightly.py` | 无直接 PR 号提交 |
| `test/registered/kernels/ops/attention/test_inkling_attn_prologue_tau.py` | 无直接 PR 号提交 |
| `test/registered/kernels/ops/attention/test_inkling_checkpoint_indices.py` | [#38229](https://github.com/sgl-project/sglang/pull/38229) |
| `test/registered/kernels/ops/gemm/test_inkling_rel_proj.py` | 无直接 PR 号提交 |
| `test/registered/kernels/ops/moe/test_inkling_silu_and_mul.py` | [#33903](https://github.com/sgl-project/sglang/pull/33903) |
| `test/registered/unit/lora/test_inkling_linearized_lora_unit.py` | [#31681](https://github.com/sgl-project/sglang/pull/31681), [#33752](https://github.com/sgl-project/sglang/pull/33752) |
| `test/registered/unit/mem_cache/test_inkling_sconv_strided_conv_state.py` | 无直接 PR 号提交 |
| `test/registered/unit/models/test_inkling_per_expert_sync.py` | [#40725](https://github.com/sgl-project/sglang/pull/40725) |
| `test/registered/unit/multimodal/rust/inkling/test_bindings.py` | 无直接 PR 号提交 |
| `test/registered/unit/parser/test_inkling_renderer.py` | [#31681](https://github.com/sgl-project/sglang/pull/31681), [#33898](https://github.com/sgl-project/sglang/pull/33898) |

## PR 覆盖总览

- git 追溯 PR 数: 19
- 原文档显式引用补充 PR 数: 0
- 当前文档总 PR 数: 19
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-07-20 | [#31681](https://github.com/sgl-project/sglang/pull/31681) | merged | Add Inkling model support | `python/sglang/srt/models/inkling.py`, `python/sglang/srt/models/inkling_common/kernels/comm.py`, `python/sglang/srt/models/inkling_common/kernels/sconv.py` |
| 2026-07-22 | [#32076](https://github.com/sgl-project/sglang/pull/32076) | merged | Fix Inkling kernel imports after migration | `python/sglang/srt/models/inkling_common/kernels/comm.py`, `python/sglang/srt/models/inkling_common/attn.py` |
| 2026-07-27 | [#31840](https://github.com/sgl-project/sglang/pull/31840) | merged | [Inkling] Add minimal DFLASH support | `python/sglang/srt/models/inkling.py` |
| 2026-07-30 | [#32861](https://github.com/sgl-project/sglang/pull/32861) | merged | Fix Inkling tool-call parsing recovery, content handling, and streaming | `python/sglang/srt/function_call/inkling_detector.py` |
| 2026-07-31 | [#33023](https://github.com/sgl-project/sglang/pull/33023) | merged | feat(inkling): migrate short convs onto the ShortConv attention backend | `python/sglang/srt/layers/attention/linear/inkling_sconv_backend.py`, `python/sglang/srt/models/inkling_common/sconv.py`, `python/sglang/srt/models/inkling_common/kernels/sconv.py` |
| 2026-08-01 | [#33116](https://github.com/sgl-project/sglang/pull/33116) | merged | [Inkling] Hold the short-conv per-step state on one metadata struct | `python/sglang/srt/layers/attention/linear/inkling_sconv_backend.py`, `python/sglang/srt/models/inkling_common/sconv.py` |
| 2026-08-05 | [#33752](https://github.com/sgl-project/sglang/pull/33752) | merged | [test] Re-enable a pruned Inkling LoRA unit-test set (68 -> 9 cases) | `test/registered/unit/lora/test_inkling_linearized_lora_unit.py` |
| 2026-08-05 | [#33108](https://github.com/sgl-project/sglang/pull/33108) | merged | feat(dgx-spark): add inkling-small MoE support for sm_121 | `python/sglang/kernels/ops/moe/inkling_moe.py` |
| 2026-08-06 | [#33750](https://github.com/sgl-project/sglang/pull/33750) | merged | [Fix] Inkling works with gs:// runai_streamer paths | `python/sglang/srt/models/inkling_common/quantization/config.py` |
| 2026-08-07 | [#33417](https://github.com/sgl-project/sglang/pull/33417) | merged | Fix deterministic inference for Inkling | `python/sglang/srt/models/inkling_common/attn.py`, `python/sglang/srt/models/inkling_common/kernels/comm.py` |
| 2026-08-08 | [#33898](https://github.com/sgl-project/sglang/pull/33898) | merged | [inkling] Render tool-result media instead of coercing content to str | `test/registered/unit/parser/test_inkling_renderer.py`, `python/sglang/srt/parser/inkling_renderer.py` |
| 2026-08-08 | [#33903](https://github.com/sgl-project/sglang/pull/33903) | merged | [Inkling] silu_and_mul: replace helion kernels with plain Triton | `python/sglang/srt/models/inkling_common/moe.py`, `python/sglang/kernels/ops/moe/inkling_moe.py`, `test/registered/kernels/ops/moe/test_inkling_silu_and_mul.py` |
| 2026-08-10 | [#34250](https://github.com/sgl-project/sglang/pull/34250) | merged | Update dspark draft path in Inkling small cookbook | `docs/src/snippets/configs/thinkingmachines/inkling-small.jsx`, `docs/cookbook/autoregressive/ThinkingMachines/Inkling-Small.mdx` |
| 2026-08-18 | [#35161](https://github.com/sgl-project/sglang/pull/35161) | merged | Skip inkling sheared bias under batch invariance | `python/sglang/srt/models/inkling_common/attn.py` |
| 2026-08-24 | [#35840](https://github.com/sgl-project/sglang/pull/35840) | merged | Add PD test for inkling with mxfp8 KV | `test/registered/disaggregation/test_disaggregation_inkling_mxfp8.py`, `python/sglang/srt/mem_cache/kv_cache_builder.py`, `python/sglang/srt/managers/schedule_batch.py` |
| 2026-09-10 | [#38169](https://github.com/sgl-project/sglang/pull/38169) | merged | [Spec] Stage Inkling MTP draft metadata before verify | `python/sglang/srt/layers/attention/linear/inkling_sconv_backend.py` |
| 2026-09-22 | [#40725](https://github.com/sgl-project/sglang/pull/40725) | merged | Fix the Inkling per-expert sync test and collect it in the weekly CPU run | `test/registered/unit/models/test_inkling_per_expert_sync.py` |
| 2026-09-28 | [#41144](https://github.com/sgl-project/sglang/pull/41144) | merged | [Unified Memory] Fix Inkling conv-checkpoint track ids written to virtual slot numbers | `python/sglang/srt/layers/attention/linear/inkling_sconv_backend.py` |
| 2026-10-04 | [#38229](https://github.com/sgl-project/sglang/pull/38229) | merged | fix(inkling): translate unified-memory checkpoint destinations to physical slots | `python/sglang/srt/layers/attention/linear/inkling_sconv_backend.py`, `python/sglang/srt/models/inkling_common/sconv.py`, `python/sglang/srt/models/inkling_common/kernels/comm.py` |

## 逐 PR diff 审计卡

### PR #31681 - Add Inkling model support

- 链接: https://github.com/sgl-project/sglang/pull/31681
- 状态/时间: merged / 2026-07-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/configs/inkling.py`, `python/sglang/srt/function_call/inkling_detector.py`, `python/sglang/srt/lora/trtllm_lora_temp/inkling_dense.py`, `python/sglang/srt/models/inkling.py`, `python/sglang/srt/models/inkling_common/__init__.py` 等 29 个文件；关联提交 `02236fa38cb0`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 279 个文件，+56899/-778，可读 patch 51462 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/sglang/srt/models/inkling.py` added +1972/-0 (1972 lines); hunks: -0,0 +1,1972; symbols: _shard_full_to_local, _normalize_mm_weight_name, _is_unsupported_mm_weight_name, InklingDecoderLayer，涉及 `_shard_full_to_local, _normalize_mm_weight_name, _is_unsupported_mm_weight_name`；`python/sglang/srt/models/inkling_common/kernels/comm.py` added +1195/-0 (1195 lines); hunks: -0,0 +1,1195; symbols: _InklingArResources, _ar_jit, _ar_fused_jit, _get_inkling_ar_resources，涉及 `_InklingArResources, _ar_jit, _ar_fused_jit`；`python/sglang/srt/models/inkling_common/kernels/sconv.py` added +1160/-0 (1160 lines); hunks: -0,0 +1,1160; symbols: SconvDecodeMetadata, SconvExtendMetadata, _conv_prefix_autotune_configs, _causal_conv1d_fwd_with_prefix_kernel，涉及 `SconvDecodeMetadata, SconvExtendMetadata, _conv_prefix_autotune_configs`；`python/sglang/srt/models/inkling_common/moe.py` added +1098/-0 (1098 lines); hunks: -0,0 +1,1098; symbols: _mm_fp32, _addmm_fp32, _load_gate_weight_padded, inkling_fused_gate_linear_with_fp32_out，涉及 `_mm_fp32, _addmm_fp32, _load_gate_weight_padded`。
- 代码 diff 细节:
  - `python/sglang/srt/models/inkling.py` added +1972/-0 (1972 lines); hunks: -0,0 +1,1972; symbols: _shard_full_to_local, _normalize_mm_weight_name, _is_unsupported_mm_weight_name, InklingDecoderLayer
  - `python/sglang/srt/models/inkling_common/kernels/comm.py` added +1195/-0 (1195 lines); hunks: -0,0 +1,1195; symbols: _InklingArResources, _ar_jit, _ar_fused_jit, _get_inkling_ar_resources
  - `python/sglang/srt/models/inkling_common/kernels/sconv.py` added +1160/-0 (1160 lines); hunks: -0,0 +1,1160; symbols: SconvDecodeMetadata, SconvExtendMetadata, _conv_prefix_autotune_configs, _causal_conv1d_fwd_with_prefix_kernel
  - `python/sglang/srt/models/inkling_common/moe.py` added +1098/-0 (1098 lines); hunks: -0,0 +1,1098; symbols: _mm_fp32, _addmm_fp32, _load_gate_weight_padded, inkling_fused_gate_linear_with_fp32_out
  - `python/sglang/srt/models/inkling_common/sconv.py` added +1047/-0 (1047 lines); hunks: -0,0 +1,1047; symbols: SconvType, ShortConvolution, implements, __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/inkling.py
@@ -0,0 +1,1972 @@
+from __future__ import annotations
+import copy
+import logging
+import re
+from typing import Iterable, Optional, Set, Tuple
+import torch
diff -- python/sglang/srt/models/inkling_common/kernels/comm.py
@@ -0,0 +1,1195 @@
+from __future__ import annotations
+import functools
+from typing import TYPE_CHECKING
+import msgspec
+import torch
+from sglang.srt.environ import envs
diff -- python/sglang/srt/models/inkling_common/kernels/sconv.py
@@ -0,0 +1,1160 @@
```

- 提取文件（未人工审阅）:
  - runtime: `python/sglang/srt/models/inkling.py` added +1972/-0; `python/sglang/srt/models/inkling_common/kernels/comm.py` added +1195/-0; `python/sglang/srt/models/inkling_common/kernels/sconv.py` added +1160/-0; `python/sglang/srt/models/inkling_common/moe.py` added +1098/-0; `python/sglang/srt/models/inkling_common/sconv.py` added +1047/-0; `python/sglang/srt/models/inkling_common/attn.py` added +978/-0
- 验证与风险: diff 自带测试面 `python/sglang/jit_kernel/tests/test_moe_preprocess.py`, `python/sglang/jit_kernel/tests/test_sconv_decode_metadata.py`, `python/sglang/jit_kernel/tests/test_sconv_extend_metadata.py`, `python/sglang/test/kits/attention_unittest/runner_modes/cuda_graph_decode_runner.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #32076 - Fix Inkling kernel imports after migration

- 链接: https://github.com/sgl-project/sglang/pull/32076
- 状态/时间: merged / 2026-07-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/inkling_common/attn.py`, `python/sglang/srt/models/inkling_common/kernels/comm.py`；关联提交 `b855efd9e66a`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+9/-6，可读 patch 57 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/sglang/srt/models/inkling_common/kernels/comm.py` modified +7/-5 (12 lines); hunks: -94,7 +94,7 @@ def _ar_jit():; -103,7 +103,7 @@ def _ar_jit():; symbols: _ar_jit, _ar_fused_jit, ar_sconv_norm_fusable, all_gather_hidden，涉及 `_ar_jit, _ar_fused_jit, ar_sconv_norm_fusable`；`python/sglang/srt/models/inkling_common/attn.py` modified +2/-1 (3 lines); hunks: -364,7 +364,8 @@ def _project_qkvr(; symbols: _project_qkvr, _fused_attn_prologue_verify，涉及 `_project_qkvr, _fused_attn_prologue_verify`。
- 代码 diff 细节:
  - `python/sglang/srt/models/inkling_common/kernels/comm.py` modified +7/-5 (12 lines); hunks: -94,7 +94,7 @@ def _ar_jit():; -103,7 +103,7 @@ def _ar_jit():; symbols: _ar_jit, _ar_fused_jit, ar_sconv_norm_fusable, all_gather_hidden
  - `python/sglang/srt/models/inkling_common/attn.py` modified +2/-1 (3 lines); hunks: -364,7 +364,8 @@ def _project_qkvr(; symbols: _project_qkvr, _fused_attn_prologue_verify
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/inkling_common/kernels/comm.py
@@ -94,7 +94,7 @@ def _ar_jit():
-    from sglang.jit_kernel import inkling_all_reduce
+    from sglang.kernels.ops.model.inkling import inkling_all_reduce
@@ -103,7 +103,7 @@ def _ar_jit():
-    from sglang.jit_kernel import inkling_ar_fused
+    from sglang.kernels.ops.model.inkling import inkling_ar_fused
@@ -240,7 +240,8 @@ def ar_sconv_norm_fusable(
diff -- python/sglang/srt/models/inkling_common/attn.py
@@ -364,7 +364,8 @@ def _project_qkvr(
-        (jit_kernel/inkling_attn_prologue.py); returns ``(q, k, v, did_store)``.
+        (kernels/ops/model/inkling/inkling_attn_prologue.py); returns
+        ``(q, k, v, did_store)``.
```

- 提取文件（未人工审阅）:
  - runtime: `python/sglang/srt/models/inkling_common/kernels/comm.py` modified +7/-5; `python/sglang/srt/models/inkling_common/attn.py` modified +2/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/inkling_common/attn.py`, `python/sglang/srt/models/inkling_common/kernels/comm.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #31840 - [Inkling] Add minimal DFLASH support

- 链接: https://github.com/sgl-project/sglang/pull/31840
- 状态/时间: merged / 2026-07-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/inkling.py`；关联提交 `1da062f018ff`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 6 个文件，+186/-26，可读 patch 355 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/sglang/srt/models/inkling.py` modified +86/-6 (92 lines); hunks: -677,6 +677,32 @@ def get_layer(idx: int, prefix: str) -> InklingDecoderLayer:; -778,6 +804,12 @@ def forward(; symbols: get_layer, set_dflash_layers_to_capture, get_input_embeddings, forward，涉及 `get_layer, set_dflash_layers_to_capture, get_input_embeddings`。
- 代码 diff 细节:
  - `python/sglang/srt/models/inkling.py` modified +86/-6 (92 lines); hunks: -677,6 +677,32 @@ def get_layer(idx: int, prefix: str) -> InklingDecoderLayer:; -778,6 +804,12 @@ def forward(; symbols: get_layer, set_dflash_layers_to_capture, get_input_embeddings, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/inkling.py
@@ -677,6 +677,32 @@ def get_layer(idx: int, prefix: str) -> InklingDecoderLayer:
+        self._dflash_layers_to_capture: set[int] = set()
+    def set_dflash_layers_to_capture(self, layer_ids: list[int]) -> None:
+        """Capture post-layer hidden states consumed by a DFLASH drafter."""
+        if layer_ids is None:
+            raise ValueError("DFLASH requires explicit target layer IDs.")
+        if len(layer_ids) != len(set(layer_ids)):
```

- 提取文件（未人工审阅）:
  - runtime: `python/sglang/srt/models/inkling.py` modified +86/-6
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/arg_groups/speculative_hook.py`, `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`, `python/sglang/srt/models/dflash.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #32861 - Fix Inkling tool-call parsing recovery, content handling, and streaming

- 链接: https://github.com/sgl-project/sglang/pull/32861
- 状态/时间: merged / 2026-07-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/function_call/inkling_detector.py`；关联提交 `07a087bf45f4`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 4 个文件，+322/-278，可读 patch 789 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/sglang/srt/function_call/inkling_detector.py` modified +169/-225 (394 lines); hunks: -1,11 +1,8; -16,9 +13,9; symbols: _reject_nonfinite_number, InklingDetector, __init__, has_tool_call，涉及 `_reject_nonfinite_number, InklingDetector, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/function_call/inkling_detector.py` modified +169/-225 (394 lines); hunks: -1,11 +1,8; -16,9 +13,9; symbols: _reject_nonfinite_number, InklingDetector, __init__, has_tool_call
- 关键代码摘录:

```diff
diff -- python/sglang/srt/function_call/inkling_detector.py
@@ -1,11 +1,8 @@
-import re
-from partial_json_parser.core.exceptions import MalformedJSON
-from partial_json_parser.core.options import Allow
@@ -16,9 +13,9 @@
-from sglang.srt.function_call.utils import _is_complete_json, _partial_json_loads
+    CONTENT_INVOKE_TOOL_TEXT,
```

- 提取文件（未人工审阅）:
  - runtime: `python/sglang/srt/function_call/inkling_detector.py` modified +169/-225
- 验证与风险: diff 自带测试面 `test/registered/unit/function_call/test_function_call_parser.py`, `test/registered/unit/parser/test_reasoning_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #33023 - feat(inkling): migrate short convs onto the ShortConv attention backend

- 链接: https://github.com/sgl-project/sglang/pull/33023
- 状态/时间: merged / 2026-07-31
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/layers/attention/linear/inkling_sconv_backend.py`, `python/sglang/srt/models/inkling.py`, `python/sglang/srt/models/inkling_common/kernels/comm.py`, `python/sglang/srt/models/inkling_common/kernels/sconv.py`, `python/sglang/srt/models/inkling_common/sconv.py`；关联提交 `77c77a3da879`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 16 个文件，+1197/-437，可读 patch 1983 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/sglang/srt/layers/attention/linear/inkling_sconv_backend.py` added +572/-0 (572 lines); hunks: -0,0 +1,572; symbols: InklingShortConvMetadata, InklingShortConvAttnBackend, __init__, _alloc_graph_buffers，涉及 `InklingShortConvMetadata, InklingShortConvAttnBackend, __init__`；`python/sglang/srt/models/inkling_common/sconv.py` modified +61/-342 (403 lines); hunks: -1,5 +1,4; -10,24 +9,16; symbols: SconvType, ShortConvolution, weight_loader, _owns_extend_metadata，涉及 `SconvType, ShortConvolution, weight_loader`；`python/sglang/srt/models/inkling_common/kernels/sconv.py` modified +69/-20 (89 lines); hunks: -23,6 +23,49 @@ class SconvExtendMetadata(TypedDict):; -260,23 +303,25 @@ def _fused_decode_metadata_kernel(; symbols: SconvExtendMetadata, SconvMetadataOut, _metadata_out, _fused_decode_metadata_kernel，涉及 `SconvExtendMetadata, SconvMetadataOut, _metadata_out`；`python/sglang/srt/models/inkling.py` modified +0/-32 (32 lines); hunks: -1198,38 +1198,6 @@ def forward(; symbols: forward, update_conv_state_after_mtp_verify, _load_regular_param，涉及 `forward, update_conv_state_after_mtp_verify, _load_regular_param`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/linear/inkling_sconv_backend.py` added +572/-0 (572 lines); hunks: -0,0 +1,572; symbols: InklingShortConvMetadata, InklingShortConvAttnBackend, __init__, _alloc_graph_buffers
  - `python/sglang/srt/models/inkling_common/sconv.py` modified +61/-342 (403 lines); hunks: -1,5 +1,4; -10,24 +9,16; symbols: SconvType, ShortConvolution, weight_loader, _owns_extend_metadata
  - `python/sglang/srt/models/inkling_common/kernels/sconv.py` modified +69/-20 (89 lines); hunks: -23,6 +23,49 @@ class SconvExtendMetadata(TypedDict):; -260,23 +303,25 @@ def _fused_decode_metadata_kernel(; symbols: SconvExtendMetadata, SconvMetadataOut, _metadata_out, _fused_decode_metadata_kernel
  - `python/sglang/srt/models/inkling.py` modified +0/-32 (32 lines); hunks: -1198,38 +1198,6 @@ def forward(; symbols: forward, update_conv_state_after_mtp_verify, _load_regular_param
  - `python/sglang/srt/models/inkling_common/kernels/comm.py` modified +1/-1 (2 lines); hunks: -1003,7 +1003,7 @@ def ar_scattered_sconv_fused(; symbols: ar_scattered_sconv_fused
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/linear/inkling_sconv_backend.py
@@ -0,0 +1,572 @@
+# Copyright 2023-2026 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/models/inkling_common/sconv.py
@@ -1,5 +1,4 @@
-from typing import Any
@@ -10,24 +9,16 @@
-from sglang.srt.model_executor.forward_context import get_req_to_token_pool
+from sglang.srt.model_executor.forward_context import get_attn_backend
-    HIS_ONES,
-    HIS_PREFIX,
diff -- python/sglang/srt/models/inkling_common/kernels/sconv.py
@@ -23,6 +23,49 @@ class SconvExtendMetadata(TypedDict):
```

- 提取文件（未人工审阅）:
  - runtime: `python/sglang/srt/layers/attention/linear/inkling_sconv_backend.py` added +572/-0; `python/sglang/srt/models/inkling_common/sconv.py` modified +61/-342; `python/sglang/srt/models/inkling_common/kernels/sconv.py` modified +69/-20; `python/sglang/srt/models/inkling.py` modified +0/-32; `python/sglang/srt/models/inkling_common/kernels/comm.py` modified +1/-1
- 验证与风险: diff 自带测试面 `test/registered/kernels/ops/mamba/test_sconv_decode_metadata.py`, `test/registered/kernels/ops/mamba/test_sconv_extend_metadata.py`, `test/registered/unit/models/test_inkling_sconv_metadata_once.py`, `test/registered/unit/spec/test_ngram_mamba_verify_update.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #33116 - [Inkling] Hold the short-conv per-step state on one metadata struct

- 链接: https://github.com/sgl-project/sglang/pull/33116
- 状态/时间: merged / 2026-08-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/layers/attention/linear/inkling_sconv_backend.py`, `python/sglang/srt/models/inkling_common/sconv.py`；关联提交 `934a13ce3e60`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 4 个文件，+80/-424，可读 patch 699 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/sglang/srt/layers/attention/linear/inkling_sconv_backend.py` modified +47/-37 (84 lines); hunks: -15,7 +15,8; -34,8 +35,9; symbols: InklingShortConvMetadata, InklingShortConvAttnBackend, __init__，涉及 `InklingShortConvMetadata, InklingShortConvAttnBackend, __init__`；`python/sglang/srt/models/inkling_common/sconv.py` modified +19/-19 (38 lines); hunks: -7,7 +7,6; -123,16 +122,20 @@ def weight_loader(self, param: Parameter, loaded_weight: t...; symbols: weight_loader, _conv_state, _sconv_cache, _intermediate_window，涉及 `weight_loader, _conv_state, _sconv_cache`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/linear/inkling_sconv_backend.py` modified +47/-37 (84 lines); hunks: -15,7 +15,8; -34,8 +35,9; symbols: InklingShortConvMetadata, InklingShortConvAttnBackend, __init__
  - `python/sglang/srt/models/inkling_common/sconv.py` modified +19/-19 (38 lines); hunks: -7,7 +7,6; -123,16 +122,20 @@ def weight_loader(self, param: Parameter, loaded_weight: t...; symbols: weight_loader, _conv_state, _sconv_cache, _intermediate_window
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/linear/inkling_sconv_backend.py
@@ -15,7 +15,8 @@
-``MambaPool``; the model reaches this via :meth:`conv_state_metadata`, never
+``MambaPool``; the model reaches this via :meth:`conv_state_metadata` for the
+step's metadata and :meth:`sconv_state` for a layer's own conv stream, never
@@ -34,8 +35,9 @@
-from typing import TYPE_CHECKING, Any, NamedTuple, Optional
+from typing import TYPE_CHECKING, Optional
diff -- python/sglang/srt/models/inkling_common/sconv.py
@@ -7,7 +7,6 @@
-from sglang.srt.mem_cache.memory_pool import MambaPool
@@ -123,16 +122,20 @@ def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
-        """This layer's conv-state handle for the current step.
-        ``InklingShortConvAttnBackend`` resolved the whole step-global metadata set
-        once during metadata prep, so this is a pure read shared by every conv
-        module in the step.
```

- 提取文件（未人工审阅）:
  - runtime: `python/sglang/srt/layers/attention/linear/inkling_sconv_backend.py` modified +47/-37; `python/sglang/srt/models/inkling_common/sconv.py` modified +19/-19
- 验证与风险: diff 自带测试面 `test/registered/unit/models/test_inkling_sconv_metadata_once.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #33752 - [test] Re-enable a pruned Inkling LoRA unit-test set (68 -> 9 cases)

- 链接: https://github.com/sgl-project/sglang/pull/33752
- 状态/时间: merged / 2026-08-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/unit/lora/test_inkling_linearized_lora_unit.py`；关联提交 `b9d572ee0245`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 9 个文件，+717/-3162，可读 patch 4348 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/registered/unit/lora/test_inkling_linearized_lora_unit.py` modified +425/-1089 (1514 lines); hunks: -1,1173 +1,509; symbols: _Flag, __init__, get, is_set，涉及 `_Flag, __init__, get`。
- 代码 diff 细节:
  - `test/registered/unit/lora/test_inkling_linearized_lora_unit.py` modified +425/-1089 (1514 lines); hunks: -1,1173 +1,509; symbols: _Flag, __init__, get, is_set
- 关键代码摘录:

```diff
diff -- test/registered/unit/lora/test_inkling_linearized_lora_unit.py
@@ -1,1173 +1,509 @@
-"""Regression tests for Inkling's linearized shared-sink LoRA path.
-The production LoRA module has optional GPU/runtime imports that are unavailable
-in lightweight unit-test environments. The tests compile selected production
-methods directly so every tensor operation remains the real implementation.
-"""
+"""CPU-only regression tests for Inkling's linearized shared-sink LoRA path:
```

- 提取文件（未人工审阅）:
  - tests: `test/registered/unit/lora/test_inkling_linearized_lora_unit.py` modified +425/-1089
- 验证与风险: diff 自带测试面 `test/registered/unit/lora/test_experimental_sgl_marlin_alignment.py`, `test/registered/unit/lora/test_experimental_sgl_marlin_direct_decode.py`, `test/registered/unit/lora/test_experimental_sgl_marlin_multi_prefill.py`, `test/registered/unit/lora/test_experimental_sgl_marlin_policy.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #33108 - feat(dgx-spark): add inkling-small MoE support for sm_121

- 链接: https://github.com/sgl-project/sglang/pull/33108
- 状态/时间: merged / 2026-08-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/kernels/ops/moe/inkling_moe.py`；关联提交 `02cd44c59a69`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 3 个文件，+52/-1，可读 patch 82 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/sglang/kernels/ops/moe/inkling_moe.py` modified +6/-1 (7 lines); hunks: -11,6 +11,7; -884,7 +885,10 @@ def grouped_gemm_triton(; symbols: grouped_gemm_triton，涉及 `grouped_gemm_triton`。
- 代码 diff 细节:
  - `python/sglang/kernels/ops/moe/inkling_moe.py` modified +6/-1 (7 lines); hunks: -11,6 +11,7; -884,7 +885,10 @@ def grouped_gemm_triton(; symbols: grouped_gemm_triton
- 关键代码摘录:

```diff
diff -- python/sglang/kernels/ops/moe/inkling_moe.py
@@ -11,6 +11,7 @@
+from sglang.srt.utils.common import is_sm121
@@ -884,7 +885,10 @@ def grouped_gemm_triton(
-            "num_stages": 4,
+            # sm_121 (GB10 / DGX Spark) caps shared memory at 99 KB per block and
+            # num_stages=4 needs 108 KB. The BLOCK_SIZE_M=128 branch below fits at
+            # its default 3 (96 KB) and needs no gate.
```

- 提取文件（未人工审阅）:
  - runtime: `python/sglang/kernels/ops/moe/inkling_moe.py` modified +6/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/kernels/ops/moe/inkling_moe.py`, `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/silu_and_mul_interleaved_sm_121.json`, `python/sglang/srt/utils/common.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #33750 - [Fix] Inkling works with gs:// runai_streamer paths

- 链接: https://github.com/sgl-project/sglang/pull/33750
- 状态/时间: merged / 2026-08-06
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/inkling_common/quantization/config.py`；关联提交 `269d51ed4be1`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+6/-2，可读 patch 24 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/sglang/srt/models/inkling_common/quantization/config.py` modified +6/-2 (8 lines); hunks: -18,6 +18,7; -43,10 +44,13 @@ def _get_raw_quant_config(; symbols: _get_raw_quant_config，涉及 `_get_raw_quant_config`。
- 代码 diff 细节:
  - `python/sglang/srt/models/inkling_common/quantization/config.py` modified +6/-2 (8 lines); hunks: -18,6 +18,7; -43,10 +44,13 @@ def _get_raw_quant_config(; symbols: _get_raw_quant_config
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/inkling_common/quantization/config.py
@@ -18,6 +18,7 @@
+from sglang.srt.utils.runai_utils import ObjectStorageModel, is_runai_obj_uri
@@ -43,10 +44,13 @@ def _get_raw_quant_config(
-    # A local path holds hf_quant_config.json directly; a remote HF repo id must
-    # first resolve its JSON configs from the hub (mirrors weight_utils.get_quant_config).
+    # A local path holds hf_quant_config.json directly; anything else resolves its
+    # JSON configs first. An object-storage URL is neither a directory nor a valid
```

- 提取文件（未人工审阅）:
  - runtime: `python/sglang/srt/models/inkling_common/quantization/config.py` modified +6/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/inkling_common/quantization/config.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #33417 - Fix deterministic inference for Inkling

- 链接: https://github.com/sgl-project/sglang/pull/33417
- 状态/时间: merged / 2026-08-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/inkling_common/attn.py`, `python/sglang/srt/models/inkling_common/kernels/comm.py`；关联提交 `ce84df0fa111`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 11 个文件，+590/-5，可读 patch 733 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/sglang/srt/models/inkling_common/attn.py` modified +11/-3 (14 lines); hunks: -121,7 +121,7 @@ def _rel_proj_kernel_eligible(r: torch.Tensor) -> bool:; -134,7 +134,11 @@ def __init__(self, d_rel: int, rel_extent: int):; symbols: _rel_proj_kernel_eligible, RelLogitsProj, __init__，涉及 `_rel_proj_kernel_eligible, RelLogitsProj, __init__`；`python/sglang/srt/models/inkling_common/kernels/comm.py` modified +3/-0 (3 lines); hunks: -455,9 +455,12 @@ def symm_mem_all_reduce(; symbols: symm_mem_all_reduce，涉及 `symm_mem_all_reduce`。
- 代码 diff 细节:
  - `python/sglang/srt/models/inkling_common/attn.py` modified +11/-3 (14 lines); hunks: -121,7 +121,7 @@ def _rel_proj_kernel_eligible(r: torch.Tensor) -> bool:; -134,7 +134,11 @@ def __init__(self, d_rel: int, rel_extent: int):; symbols: _rel_proj_kernel_eligible, RelLogitsProj, __init__
  - `python/sglang/srt/models/inkling_common/kernels/comm.py` modified +3/-0 (3 lines); hunks: -455,9 +455,12 @@ def symm_mem_all_reduce(; symbols: symm_mem_all_reduce
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/inkling_common/attn.py
@@ -121,7 +121,7 @@ def _rel_proj_kernel_eligible(r: torch.Tensor) -> bool:
-    def __init__(self, d_rel: int, rel_extent: int):
+    def __init__(self, d_rel: int, rel_extent: int, *, deterministic: bool = False):
@@ -134,7 +134,11 @@ def __init__(self, d_rel: int, rel_extent: int):
-        self._proj_dispatch = envs.SGLANG_OPT_USE_INKLING_REL_PROJ_DISPATCH.get()
+        # The dispatch keys off the token count, so a row's kernel would follow
+        # the batch composition.
diff -- python/sglang/srt/models/inkling_common/kernels/comm.py
@@ -455,9 +455,12 @@ def symm_mem_all_reduce(
+        # select_ar_config() keys off the token count; plain multimem below
+        # reduces in a shape-independent order.
+            and not get_exec().deterministic.enable_deterministic_inference
```

- 提取文件（未人工审阅）:
  - runtime: `python/sglang/srt/models/inkling_common/attn.py` modified +11/-3; `python/sglang/srt/models/inkling_common/kernels/comm.py` modified +3/-0
- 验证与风险: diff 自带测试面 `python/sglang/test/cache_consistency_jitter.py`, `test/registered/models_e2e/test_inkling.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #33898 - [inkling] Render tool-result media instead of coercing content to str

- 链接: https://github.com/sgl-project/sglang/pull/33898
- 状态/时间: merged / 2026-08-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/parser/inkling_renderer.py`, `test/registered/unit/parser/test_inkling_renderer.py`；关联提交 `c69d59395b6c`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+147/-20，可读 patch 205 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/registered/unit/parser/test_inkling_renderer.py` modified +128/-0 (128 lines); hunks: -3,15 +3,18; -184,6 +187,131 @@ def test_empty_assistant_message_does_not_emit_bare_termin...; symbols: test_empty_assistant_message_does_not_emit_bare_terminator, test_tool_result_renders_image_parts_alongside_text, test_tool_result_placeholder_count_matches_image_parts, test_tool_result_with_multiple_text_blocks_renders_each，涉及 `test_empty_assistant_message_does_not_emit_bare_terminator, test_tool_result_renders_image_parts_alongside_text, test_tool_result_placeholder_count_matches_image_parts`；`python/sglang/srt/parser/inkling_renderer.py` modified +19/-20 (39 lines); hunks: -99,17 +99,26 @@ def append_effort() -> None:; -249,16 +258,6 @@ def _format_reasoning_effort(reasoning_effort: float) -> str:; symbols: append_effort, _format_reasoning_effort, _expect_string_content, _expect_role，涉及 `append_effort, _format_reasoning_effort, _expect_string_content`。
- 代码 diff 细节:
  - `test/registered/unit/parser/test_inkling_renderer.py` modified +128/-0 (128 lines); hunks: -3,15 +3,18; -184,6 +187,131 @@ def test_empty_assistant_message_does_not_emit_bare_termin...; symbols: test_empty_assistant_message_does_not_emit_bare_terminator, test_tool_result_renders_image_parts_alongside_text, test_tool_result_placeholder_count_matches_image_parts, test_tool_result_with_multiple_text_blocks_renders_each
  - `python/sglang/srt/parser/inkling_renderer.py` modified +19/-20 (39 lines); hunks: -99,17 +99,26 @@ def append_effort() -> None:; -249,16 +258,6 @@ def _format_reasoning_effort(reasoning_effort: float) -> str:; symbols: append_effort, _format_reasoning_effort, _expect_string_content, _expect_role
- 关键代码摘录:

```diff
diff -- test/registered/unit/parser/test_inkling_renderer.py
@@ -3,15 +3,18 @@
+    CONTENT_IMAGE,
+    IMAGE_TOKEN_ID,
+    MESSAGE_TOOL,
@@ -184,6 +187,131 @@ def test_empty_assistant_message_does_not_emit_bare_terminator(self):
+    def test_tool_result_renders_image_parts_alongside_text(self):
+        """Bug regression: the tool branch coerced content to a string, so a
diff -- python/sglang/srt/parser/inkling_renderer.py
@@ -99,17 +99,26 @@ def append_effort() -> None:
-            tool_name = message.get("name") or tool_call_id_to_name.get(
-                message.get("tool_call_id") or "", ""
-            )
-            _append_message(
-                input_ids,
-                tokenizer,
```

- 提取文件（未人工审阅）:
  - tests: `test/registered/unit/parser/test_inkling_renderer.py` modified +128/-0
  - runtime: `python/sglang/srt/parser/inkling_renderer.py` modified +19/-20
- 验证与风险: diff 自带测试面 `test/registered/unit/parser/test_inkling_renderer.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #33903 - [Inkling] silu_and_mul: replace helion kernels with plain Triton

- 链接: https://github.com/sgl-project/sglang/pull/33903
- 状态/时间: merged / 2026-08-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/kernels/ops/moe/inkling_moe.py`, `python/sglang/srt/models/inkling_common/moe.py`, `test/registered/kernels/ops/moe/test_inkling_silu_and_mul.py`；关联提交 `afb4f37ca509`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 13 个文件，+368/-700，可读 patch 1146 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/sglang/srt/models/inkling_common/moe.py` modified +2/-2 (4 lines); hunks: -25,7 +25,7; -572,7 +572,7 @@ def activation(; symbols: activation，涉及 `activation`；`python/sglang/kernels/ops/moe/inkling_moe.py` modified +207/-123 (330 lines); hunks: -1,163 +1,247; symbols: silu_and_mul_key, silu_and_mul_inputs, _silu_and_mul_helion_interleaved_kernel, silu_and_mul_interleaved_kernel，涉及 `silu_and_mul_key, silu_and_mul_inputs, _silu_and_mul_helion_interleaved_kernel`；`test/registered/kernels/ops/moe/test_inkling_silu_and_mul.py` added +159/-0 (159 lines); hunks: -0,0 +1,159; symbols: _reference, _ulp_diff_bf16, _check, test_matches_reference，涉及 `_reference, _ulp_diff_bf16, _check`。
- 代码 diff 细节:
  - `python/sglang/srt/models/inkling_common/moe.py` modified +2/-2 (4 lines); hunks: -25,7 +25,7; -572,7 +572,7 @@ def activation(; symbols: activation
  - `python/sglang/kernels/ops/moe/inkling_moe.py` modified +207/-123 (330 lines); hunks: -1,163 +1,247; symbols: silu_and_mul_key, silu_and_mul_inputs, _silu_and_mul_helion_interleaved_kernel, silu_and_mul_interleaved_kernel
  - `test/registered/kernels/ops/moe/test_inkling_silu_and_mul.py` added +159/-0 (159 lines); hunks: -0,0 +1,159; symbols: _reference, _ulp_diff_bf16, _check, test_matches_reference
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/inkling_common/moe.py
@@ -25,7 +25,7 @@
-    silu_and_mul_helion,
+    silu_and_mul,
@@ -572,7 +572,7 @@ def activation(
-        return silu_and_mul_helion(
+        return silu_and_mul(
diff -- python/sglang/kernels/ops/moe/inkling_moe.py
@@ -1,163 +1,247 @@
-from functools import partial
-import helion
-import helion.language as hl
-from sglang.srt.layers.moe.moe_runner.triton_utils.helion_utils import (
-    get_model_depths,
-    helion_aot_autotune,
diff -- test/registered/kernels/ops/moe/test_inkling_silu_and_mul.py
@@ -0,0 +1,159 @@
+"""Numerics tests for the plain-Triton silu_and_mul (former helion kernels).
```

- 提取文件（未人工审阅）:
  - runtime: `python/sglang/srt/models/inkling_common/moe.py` modified +2/-2; `python/sglang/kernels/ops/moe/inkling_moe.py` modified +207/-123
  - tests: `test/registered/kernels/ops/moe/test_inkling_silu_and_mul.py` added +159/-0
- 验证与风险: diff 自带测试面 `test/registered/kernels/ops/moe/test_inkling_silu_and_mul.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #34250 - Update dspark draft path in Inkling small cookbook

- 链接: https://github.com/sgl-project/sglang/pull/34250
- 状态/时间: merged / 2026-08-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/cookbook/autoregressive/ThinkingMachines/Inkling-Small.mdx`, `docs/src/snippets/configs/thinkingmachines/inkling-small.jsx`；关联提交 `430f38ea2530`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+5/-5，可读 patch 44 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `docs/src/snippets/configs/thinkingmachines/inkling-small.jsx` modified +4/-4 (8 lines); hunks: -751,7 +751,7 @@ export const config = {; -786,7 +786,7 @@ export const config = {；`docs/cookbook/autoregressive/ThinkingMachines/Inkling-Small.mdx` modified +1/-1 (2 lines); hunks: -307,6 +307,6 @@ To try it, select the **Long Context** strategy in the Deplo...。
- 代码 diff 细节:
  - `docs/src/snippets/configs/thinkingmachines/inkling-small.jsx` modified +4/-4 (8 lines); hunks: -751,7 +751,7 @@ export const config = {; -786,7 +786,7 @@ export const config = {
  - `docs/cookbook/autoregressive/ThinkingMachines/Inkling-Small.mdx` modified +1/-1 (2 lines); hunks: -307,6 +307,6 @@ To try it, select the **Long Context** strategy in the Deplo...
- 关键代码摘录:

```diff
diff -- docs/src/snippets/configs/thinkingmachines/inkling-small.jsx
@@ -751,7 +751,7 @@ export const config = {
-        "--speculative-draft-model-path RadixArk/Inkling-Small-DSpark-Preview",
+        "--speculative-draft-model-path RadixArk/Inkling-Small-DSpark",
@@ -786,7 +786,7 @@ export const config = {
-        "--speculative-draft-model-path RadixArk/Inkling-Small-DSpark-Preview",
+        "--speculative-draft-model-path RadixArk/Inkling-Small-DSpark",
@@ -821,7 +821,7 @@ export const config = {
diff -- docs/cookbook/autoregressive/ThinkingMachines/Inkling-Small.mdx
@@ -307,6 +307,6 @@ To try it, select the **Long Context** strategy in the Deploy panel above for an
-The **DSpark** deploy strategy is the second speculative-decoding path for Inkling-Small. Unlike **MTP**, which drives Inkling-Small's own multi-layer draft head, DSpark runs a **
+The **DSpark** deploy strategy is the second speculative-decoding path for Inkling-Small. Unlike **MTP**, which drives Inkling-Small's own multi-layer draft head, DSpark runs a **
```

- 提取文件（未人工审阅）:
  - docs: `docs/src/snippets/configs/thinkingmachines/inkling-small.jsx` modified +4/-4; `docs/cookbook/autoregressive/ThinkingMachines/Inkling-Small.mdx` modified +1/-1
- 验证与风险: 该 PR 主要落在文档/示例 `docs/cookbook/autoregressive/ThinkingMachines/Inkling-Small.mdx`, `docs/src/snippets/configs/thinkingmachines/inkling-small.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #35161 - Skip inkling sheared bias under batch invariance

- 链接: https://github.com/sgl-project/sglang/pull/35161
- 状态/时间: merged / 2026-08-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/inkling_common/attn.py`；关联提交 `d528192bf983`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+20/-17，可读 patch 65 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/sglang/srt/models/inkling_common/attn.py` modified +12/-1 (13 lines); hunks: -6,6 +6,9; -943,7 +946,15 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/inkling_common/attn.py` modified +12/-1 (13 lines); hunks: -6,6 +6,9; -943,7 +946,15 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/inkling_common/attn.py
@@ -6,6 +6,9 @@
+from sglang.kernels.ops.attention.flash_attn.cute.batch_invariance import (
+    is_batch_invariant,
+)
@@ -943,7 +946,15 @@ def forward(
-        if envs.SGLANG_OPT_USE_INKLING_SHEARED_BIAS.get() and fa4:
+        # The sheared-bias kernel is not batch invariant: its bias tile geometry
```

- 提取文件（未人工审阅）:
  - runtime: `python/sglang/srt/models/inkling_common/attn.py` modified +12/-1
- 验证与风险: diff 自带测试面 `test/registered/radix_cache/unified_radix_tree/test_unified_radix_cache_kl_hybrid_bitexact.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #35840 - Add PD test for inkling with mxfp8 KV

- 链接: https://github.com/sgl-project/sglang/pull/35840
- 状态/时间: merged / 2026-08-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/disaggregation/test_disaggregation_inkling_mxfp8.py`；关联提交 `586211bc461c`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 6 个文件，+260/-13，可读 patch 346 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/registered/disaggregation/test_disaggregation_inkling_mxfp8.py` added +132/-0 (132 lines); hunks: -0,0 +1,132; symbols: TestDisaggregationInklingMXFP8, setUpClass, start_prefill, start_decode，涉及 `TestDisaggregationInklingMXFP8, setUpClass, start_prefill`；`python/sglang/srt/mem_cache/kv_cache_builder.py` modified +20/-11 (31 lines); hunks: -146,6 +146,18 @@ def _register_legacy_hicache_draft(; -165,8 +177,13 @@ def resolve_decode_retraction_backup(*, tp_worker: BaseTpWo...; symbols: _register_legacy_hicache_draft, uses_ssm_state, resolve_decode_retraction_backup, build_kv_cache，涉及 `_register_legacy_hicache_draft, uses_ssm_state, resolve_decode_retraction_backup`；`python/sglang/srt/managers/schedule_batch.py` modified +22/-2 (24 lines); hunks: -105,7 +105,7; -1717,15 +1717,30 @@ def reset_for_retract(self):; symbols: reset_for_retract, _mamba_pool_needing_backup, offload_kv_cache, load_kv_cache，涉及 `reset_for_retract, _mamba_pool_needing_backup, offload_kv_cache`；`python/sglang/srt/mem_cache/memory_pool.py` modified +5/-0 (5 lines); hunks: -1628,6 +1628,9 @@ def item_len_bytes(self, page_size: int) -> int:; -3654,6 +3657,8 @@ def get_kv_size_bytes(self):; symbols: item_len_bytes, KVCache, __init__, get_kv_size_bytes，涉及 `item_len_bytes, KVCache, __init__`。
- 代码 diff 细节:
  - `test/registered/disaggregation/test_disaggregation_inkling_mxfp8.py` added +132/-0 (132 lines); hunks: -0,0 +1,132; symbols: TestDisaggregationInklingMXFP8, setUpClass, start_prefill, start_decode
  - `python/sglang/srt/mem_cache/kv_cache_builder.py` modified +20/-11 (31 lines); hunks: -146,6 +146,18 @@ def _register_legacy_hicache_draft(; -165,8 +177,13 @@ def resolve_decode_retraction_backup(*, tp_worker: BaseTpWo...; symbols: _register_legacy_hicache_draft, uses_ssm_state, resolve_decode_retraction_backup, build_kv_cache
  - `python/sglang/srt/managers/schedule_batch.py` modified +22/-2 (24 lines); hunks: -105,7 +105,7; -1717,15 +1717,30 @@ def reset_for_retract(self):; symbols: reset_for_retract, _mamba_pool_needing_backup, offload_kv_cache, load_kv_cache
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +5/-0 (5 lines); hunks: -1628,6 +1628,9 @@ def item_len_bytes(self, page_size: int) -> int:; -3654,6 +3657,8 @@ def get_kv_size_bytes(self):; symbols: item_len_bytes, KVCache, __init__, get_kv_size_bytes
  - `python/sglang/srt/mem_cache/common.py` modified +2/-0 (2 lines); hunks: -35,6 +35,8 @@ class RetractionBackup(NamedTuple):; symbols: RetractionBackup, kv_to_page_indices
- 关键代码摘录:

```diff
diff -- test/registered/disaggregation/test_disaggregation_inkling_mxfp8.py
@@ -0,0 +1,132 @@
+"""PD disaggregation for Inkling with MXFP8 KV and a hierarchical prefill cache.
+Inkling ships three heterogeneous state components -- full-attention KV,
+sliding-window KV and ShortConv state -- and MXFP8 adds a block-scale component
+per KV sub-pool, each addressed like the KV it describes. A transfer that drops
+or misaligns any of them collapses generation rather than shaving accuracy.
+MXFP8 KV needs SM100+, so this is Blackwell-only.
diff -- python/sglang/srt/mem_cache/kv_cache_builder.py
@@ -146,6 +146,18 @@ def _register_legacy_hicache_draft(
+def uses_ssm_state(model_config) -> bool:
+    """Whether the model keeps recurrent/conv state alongside its attention KV."""
+    spec = linear_attn_model_spec(model_config)
+    return (
+        hybrid_gdn_config(model_config) is not None
+        or mamba2_config(model_config) is not None
diff -- python/sglang/srt/managers/schedule_batch.py
@@ -105,7 +105,7 @@
```

- 提取文件（未人工审阅）:
  - tests: `test/registered/disaggregation/test_disaggregation_inkling_mxfp8.py` added +132/-0
  - runtime: `python/sglang/srt/mem_cache/kv_cache_builder.py` modified +20/-11; `python/sglang/srt/managers/schedule_batch.py` modified +22/-2; `python/sglang/srt/mem_cache/memory_pool.py` modified +5/-0; `python/sglang/srt/mem_cache/common.py` modified +2/-0
- 验证与风险: diff 自带测试面 `test/registered/disaggregation/test_disaggregation_inkling_mxfp8.py`, `test/registered/unit/mem_cache/test_retraction_mamba_backup.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #38169 - [Spec] Stage Inkling MTP draft metadata before verify

- 链接: https://github.com/sgl-project/sglang/pull/38169
- 状态/时间: merged / 2026-09-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/layers/attention/linear/inkling_sconv_backend.py`；关联提交 `bb15be6d79d3`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 5 个文件，+205/-35，可读 patch 395 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/sglang/srt/layers/attention/linear/inkling_sconv_backend.py` modified +54/-8 (62 lines); hunks: -501,16 +501,22 @@ def commit_conv_state_after_mtp_verify(; -548,6 +554,43 @@ class InklingShortConvHybridAttnBackend(ShortConvHybridAttn...; symbols: commit_conv_state_after_mtp_verify, InklingShortConvHybridAttnBackend, supports_draft_extend_metadata_staging, init_forward_metadata_out_graph，涉及 `commit_conv_state_after_mtp_verify, InklingShortConvHybridAttnBackend, supports_draft_extend_metadata_staging`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/linear/inkling_sconv_backend.py` modified +54/-8 (62 lines); hunks: -501,16 +501,22 @@ def commit_conv_state_after_mtp_verify(; -548,6 +554,43 @@ class InklingShortConvHybridAttnBackend(ShortConvHybridAttn...; symbols: commit_conv_state_after_mtp_verify, InklingShortConvHybridAttnBackend, supports_draft_extend_metadata_staging, init_forward_metadata_out_graph
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/linear/inkling_sconv_backend.py
@@ -501,16 +501,22 @@ def commit_conv_state_after_mtp_verify(
-        """Commit the TARGET_VERIFY conv windows at each request's last accepted step.
-        Slot ids come from ``req_pool_indices``, not the per-step
-        ``self._cache_indices``: this runs after the forward context exits, so that
-        buffer may already belong to a later forward.
-        """
+        """Commit the TARGET_VERIFY conv windows at each request's last accepted step."""
```

- 提取文件（未人工审阅）:
  - runtime: `python/sglang/srt/layers/attention/linear/inkling_sconv_backend.py` modified +54/-8
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/base_attn_backend.py`, `python/sglang/srt/layers/attention/flashattention_backend.py`, `python/sglang/srt/layers/attention/linear/inkling_sconv_backend.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #40725 - Fix the Inkling per-expert sync test and collect it in the weekly CPU run

- 链接: https://github.com/sgl-project/sglang/pull/40725
- 状态/时间: merged / 2026-09-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/unit/models/test_inkling_per_expert_sync.py`；关联提交 `db73f35f4a52`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+21/-20，可读 patch 75 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/registered/unit/models/test_inkling_per_expert_sync.py` renamed +21/-20 (41 lines); hunks: -1,7 +1,7; -11,7 +11,7; symbols: _expected_stacks, TestPerExpertSync, setUp, tearDown，涉及 `_expected_stacks, TestPerExpertSync, setUp`。
- 代码 diff 细节:
  - `test/registered/unit/models/test_inkling_per_expert_sync.py` renamed +21/-20 (41 lines); hunks: -1,7 +1,7; -11,7 +11,7; symbols: _expected_stacks, TestPerExpertSync, setUp, tearDown
- 关键代码摘录:

```diff
diff -- test/registered/unit/models/test_inkling_per_expert_sync.py
@@ -1,7 +1,7 @@
-Exercises ``_load_per_expert_param`` on a simulated EP x MoE-TP grid (parallel
-helpers monkeypatched, no process groups) and checks every (ep_rank, tp_rank)
+Exercises ``_load_per_expert_param`` on a simulated EP x MoE-TP grid (topology
+published on the runtime context, no process groups) and checks every (ep_rank, tp_rank)
@@ -11,7 +11,7 @@
-Run: python3 test/srt/models/test_inkling_per_expert_sync.py
```

- 提取文件（未人工审阅）:
  - tests: `test/registered/unit/models/test_inkling_per_expert_sync.py` renamed +21/-20
- 验证与风险: diff 自带测试面 `test/registered/unit/models/test_inkling_per_expert_sync.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #41144 - [Unified Memory] Fix Inkling conv-checkpoint track ids written to virtual slot numbers

- 链接: https://github.com/sgl-project/sglang/pull/41144
- 状态/时间: merged / 2026-09-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/layers/attention/linear/inkling_sconv_backend.py`；关联提交 `da3eb6db32a1`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+47/-1，可读 patch 83 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/sglang/srt/layers/attention/linear/inkling_sconv_backend.py` modified +18/-0 (18 lines); hunks: -155,6 +155,8 @@ def _alloc_graph_buffers(self):; -269,6 +271,22 @@ def _prepare_slot_indices(self, forward_batch: ForwardBatch):; symbols: _alloc_graph_buffers, _prepare_slot_indices, _translate_track_indices, _refresh_sconv_metadata，涉及 `_alloc_graph_buffers, _prepare_slot_indices, _translate_track_indices`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/linear/inkling_sconv_backend.py` modified +18/-0 (18 lines); hunks: -155,6 +155,8 @@ def _alloc_graph_buffers(self):; -269,6 +271,22 @@ def _prepare_slot_indices(self, forward_batch: ForwardBatch):; symbols: _alloc_graph_buffers, _prepare_slot_indices, _translate_track_indices, _refresh_sconv_metadata
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/linear/inkling_sconv_backend.py
@@ -155,6 +155,8 @@ def _alloc_graph_buffers(self):
+        # Graph-static: captured track scatters read this address.
+        self._track_indices_buf = torch.zeros(max_bs, dtype=torch.int64, device=dev)
@@ -269,6 +271,22 @@ def _prepare_slot_indices(self, forward_batch: ForwardBatch):
+        if not self._slot_gather_recordable:
+            self._translate_track_indices(forward_batch)
+    def _translate_track_indices(self, forward_batch: ForwardBatch):
```

- 提取文件（未人工审阅）:
  - runtime: `python/sglang/srt/layers/attention/linear/inkling_sconv_backend.py` modified +18/-0
- 验证与风险: diff 自带测试面 `test/registered/radix_cache/unified_radix_tree/test_unified_radix_cache_kl_hybrid_bitexact.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #38229 - fix(inkling): translate unified-memory checkpoint destinations to physical slots

- 链接: https://github.com/sgl-project/sglang/pull/38229
- 状态/时间: merged / 2026-10-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/layers/attention/linear/inkling_sconv_backend.py`, `python/sglang/srt/models/inkling_common/attn.py`, `python/sglang/srt/models/inkling_common/kernels/comm.py`, `python/sglang/srt/models/inkling_common/sconv.py`, `test/registered/kernels/ops/attention/test_inkling_checkpoint_indices.py`；关联提交 `1d02a36bb7ec`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 8 个文件，+1099/-24，可读 patch 1330 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/sglang/srt/layers/attention/linear/inkling_sconv_backend.py` modified +32/-17 (49 lines); hunks: -85,6 +85,8 @@ class InklingShortConvMetadata(msgspec.Struct):; -144,7 +146,10 @@ def _alloc_graph_buffers(self):; symbols: InklingShortConvMetadata, InklingShortConvAttnBackend, _alloc_graph_buffers, init_forward_metadata_capture_cpu_graph，涉及 `InklingShortConvMetadata, InklingShortConvAttnBackend, _alloc_graph_buffers`；`python/sglang/srt/models/inkling_common/sconv.py` modified +4/-4 (8 lines); hunks: -198,7 +198,7 @@ def _apply_decode_sconv_kernel(; -216,7 +216,7 @@ def _prepare_extend_sconv_cache(; symbols: _apply_decode_sconv_kernel, _prepare_extend_sconv_cache, _save_intermediate_conv_windows, _update_sconv_cache_for_draft_extend，涉及 `_apply_decode_sconv_kernel, _prepare_extend_sconv_cache, _save_intermediate_conv_windows`；`python/sglang/srt/models/inkling_common/kernels/comm.py` modified +2/-2 (4 lines); hunks: -359,7 +359,7 @@ def ar_sconv_norm_fused(; -834,7 +834,7 @@ def ar_scattered_sconv_fused(; symbols: ar_sconv_norm_fused, ar_scattered_sconv_fused，涉及 `ar_sconv_norm_fused, ar_scattered_sconv_fused`；`python/sglang/srt/models/inkling_common/attn.py` modified +1/-1 (2 lines); hunks: -725,7 +725,7 @@ def _fused_attn_prologue_decode(self, q, k, v, forward_batch...; symbols: _fused_attn_prologue_decode，涉及 `_fused_attn_prologue_decode`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/linear/inkling_sconv_backend.py` modified +32/-17 (49 lines); hunks: -85,6 +85,8 @@ class InklingShortConvMetadata(msgspec.Struct):; -144,7 +146,10 @@ def _alloc_graph_buffers(self):; symbols: InklingShortConvMetadata, InklingShortConvAttnBackend, _alloc_graph_buffers, init_forward_metadata_capture_cpu_graph
  - `python/sglang/srt/models/inkling_common/sconv.py` modified +4/-4 (8 lines); hunks: -198,7 +198,7 @@ def _apply_decode_sconv_kernel(; -216,7 +216,7 @@ def _prepare_extend_sconv_cache(; symbols: _apply_decode_sconv_kernel, _prepare_extend_sconv_cache, _save_intermediate_conv_windows, _update_sconv_cache_for_draft_extend
  - `python/sglang/srt/models/inkling_common/kernels/comm.py` modified +2/-2 (4 lines); hunks: -359,7 +359,7 @@ def ar_sconv_norm_fused(; -834,7 +834,7 @@ def ar_scattered_sconv_fused(; symbols: ar_sconv_norm_fused, ar_scattered_sconv_fused
  - `python/sglang/srt/models/inkling_common/attn.py` modified +1/-1 (2 lines); hunks: -725,7 +725,7 @@ def _fused_attn_prologue_decode(self, q, k, v, forward_batch...; symbols: _fused_attn_prologue_decode
  - `test/registered/kernels/ops/attention/test_inkling_checkpoint_indices.py` added +997/-0 (997 lines); hunks: -0,0 +1,997; symbols: TestInklingCheckpointIndices, make_backend, batch, scatter
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/linear/inkling_sconv_backend.py
@@ -85,6 +85,8 @@ class InklingShortConvMetadata(msgspec.Struct):
+    # Kernel-facing checkpoint slots. ForwardBatch retains the virtual source.
+    track_cache_indices: Optional[torch.Tensor] = None
@@ -144,7 +146,10 @@ def _alloc_graph_buffers(self):
-        # Inert track fields for graph capture: a capture warmup batch that
+        # Prefill can capture before init_cuda_graph_state. Keep the translated
+        # checkpoint destinations at one address across both graph phases.
diff -- python/sglang/srt/models/inkling_common/sconv.py
@@ -198,7 +198,7 @@ def _apply_decode_sconv_kernel(
-            track_indices=forward_batch.mamba_track_indices,
+            track_indices=self._conv_state(forward_batch).track_cache_indices,
@@ -216,7 +216,7 @@ def _prepare_extend_sconv_cache(
-                dst_indices=forward_batch.mamba_track_indices,
+                dst_indices=self._conv_state(forward_batch).track_cache_indices,
@@ -269,7 +269,7 @@ def _update_sconv_cache_for_draft_extend(
diff -- python/sglang/srt/models/inkling_common/kernels/comm.py
@@ -359,7 +359,7 @@ def ar_sconv_norm_fused(
```

- 提取文件（未人工审阅）:
  - runtime: `python/sglang/srt/layers/attention/linear/inkling_sconv_backend.py` modified +32/-17; `python/sglang/srt/models/inkling_common/sconv.py` modified +4/-4; `python/sglang/srt/models/inkling_common/kernels/comm.py` modified +2/-2; `python/sglang/srt/models/inkling_common/attn.py` modified +1/-1
  - tests: `test/registered/kernels/ops/attention/test_inkling_checkpoint_indices.py` added +997/-0
- 验证与风险: diff 自带测试面 `test/registered/kernels/ops/attention/test_inkling_checkpoint_indices.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
