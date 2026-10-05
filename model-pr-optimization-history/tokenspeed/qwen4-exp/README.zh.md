# TokenSpeed Qwen4-Exp (Qwen3.8-Flash-Next) 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `python/tokenspeed/runtime/configs/qwen4_exp_config.py` | [#1257](https://github.com/lightseekorg/tokenspeed/pull/1257), [#1362](https://github.com/lightseekorg/tokenspeed/pull/1362) |
| `python/tokenspeed/runtime/layers/attention/backends/specific/qwen4_exp.py` | [#1416](https://github.com/lightseekorg/tokenspeed/pull/1416) |
| `python/tokenspeed/runtime/layers/attention/backends/specific/qwen4_exp_ple.py` | [#1416](https://github.com/lightseekorg/tokenspeed/pull/1416), [#1797](https://github.com/lightseekorg/tokenspeed/pull/1797) |
| `python/tokenspeed/runtime/layers/attention/kv_cache/qwen4_exp.py` | [#1257](https://github.com/lightseekorg/tokenspeed/pull/1257), [#1278](https://github.com/lightseekorg/tokenspeed/pull/1278) |
| `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/qwen4_exp.py` | [#1257](https://github.com/lightseekorg/tokenspeed/pull/1257), [#1278](https://github.com/lightseekorg/tokenspeed/pull/1278), [#1416](https://github.com/lightseekorg/tokenspeed/pull/1416), [#1797](https://github.com/lightseekorg/tokenspeed/pull/1797) |
| `python/tokenspeed/runtime/layers/qwen4_exp_ple.py` | [#1361](https://github.com/lightseekorg/tokenspeed/pull/1361), [#1362](https://github.com/lightseekorg/tokenspeed/pull/1362), [#1416](https://github.com/lightseekorg/tokenspeed/pull/1416) |
| `python/tokenspeed/runtime/models/qwen4_exp.py` | [#1257](https://github.com/lightseekorg/tokenspeed/pull/1257), [#1361](https://github.com/lightseekorg/tokenspeed/pull/1361), [#1362](https://github.com/lightseekorg/tokenspeed/pull/1362), [#1416](https://github.com/lightseekorg/tokenspeed/pull/1416) |
| `python/tokenspeed/runtime/models/qwen4_exp_nextn.py` | [#1257](https://github.com/lightseekorg/tokenspeed/pull/1257), [#1416](https://github.com/lightseekorg/tokenspeed/pull/1416) |
| `test/runtime/test_qwen4_exp.py` | [#1257](https://github.com/lightseekorg/tokenspeed/pull/1257), [#1278](https://github.com/lightseekorg/tokenspeed/pull/1278), [#1323](https://github.com/lightseekorg/tokenspeed/pull/1323), [#1361](https://github.com/lightseekorg/tokenspeed/pull/1361), [#1362](https://github.com/lightseekorg/tokenspeed/pull/1362), [#1416](https://github.com/lightseekorg/tokenspeed/pull/1416) |
| `test/runtime/test_qwen4_exp_cache_without_gdn.py` | [#1416](https://github.com/lightseekorg/tokenspeed/pull/1416) |
| `test/runtime/test_qwen4_exp_ple_backend.py` | [#1416](https://github.com/lightseekorg/tokenspeed/pull/1416), [#1797](https://github.com/lightseekorg/tokenspeed/pull/1797) |
| `test/runtime/test_qwen4_exp_ple_prefetch_timing.py` | [#1362](https://github.com/lightseekorg/tokenspeed/pull/1362) |
| `tokenspeed-kernel/test/ops/test_qwen4_exp_qsa.py` | [#1257](https://github.com/lightseekorg/tokenspeed/pull/1257), [#1416](https://github.com/lightseekorg/tokenspeed/pull/1416) |

## PR 覆盖总览

- git 追溯 PR 数: 7
- 原文档显式引用补充 PR 数: 0
- 当前文档总 PR 数: 7
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-08-27 | [#1257](https://github.com/lightseekorg/tokenspeed/pull/1257) | merged | feat: add Qwen3.8-flash-next support | `python/tokenspeed/runtime/models/qwen4_exp.py`, `python/tokenspeed/runtime/models/qwen4_exp_nextn.py`, `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/qwen4_exp.py` |
| 2026-08-28 | [#1278](https://github.com/lightseekorg/tokenspeed/pull/1278) | merged | fix(L2): support L2 cache for qwen3.8-flash-next | `python/tokenspeed/runtime/layers/attention/kv_cache/qwen4_exp.py`, `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/qwen4_exp.py`, `python/tokenspeed/runtime/layers/attention/backends/qwen4_exp.py` |
| 2026-09-01 | [#1323](https://github.com/lightseekorg/tokenspeed/pull/1323) | merged | perf(qwen4): optimize gated residual kernels and PDL | `test/runtime/test_qwen4_exp.py`, `python/tokenspeed/runtime/layers/hyperconnection.py`, `tokenspeed-kernel/python/tokenspeed_kernel/ops/hyperconnection/triton.py` |
| 2026-09-03 | [#1361](https://github.com/lightseekorg/tokenspeed/pull/1361) | merged | refactor(ple): reorganize kernels and optimize qwen4-exp execution | `python/tokenspeed/runtime/layers/qwen4_exp_ple.py`, `python/tokenspeed/runtime/models/qwen4_exp.py`, `python/tokenspeed/runtime/layers/attention/backends/qwen4_exp.py` |
| 2026-09-13 | [#1416](https://github.com/lightseekorg/tokenspeed/pull/1416) | merged | refactor(attention): compose Qwen4-Exp QSA and PLE backends | `python/tokenspeed/runtime/layers/attention/backends/specific/qwen4_exp_ple.py`, `python/tokenspeed/runtime/layers/attention/backends/specific/qwen4_exp.py`, `python/tokenspeed/runtime/layers/qwen4_exp_ple.py` |
| 2026-09-24 | [#1362](https://github.com/lightseekorg/tokenspeed/pull/1362) | merged | feat(ple): support qwen3.8-flash-next PLE offloading | `python/tokenspeed/runtime/layers/qwen4_exp_ple.py`, `python/tokenspeed/runtime/models/qwen4_exp.py`, `python/tokenspeed/runtime/configs/qwen4_exp_config.py` |
| 2026-09-26 | [#1797](https://github.com/lightseekorg/tokenspeed/pull/1797) | merged | feat(attention): let Qwen4-Exp's PLE and QSA indexer take a replacement pool | `python/tokenspeed/runtime/layers/attention/backends/specific/qwen4_exp_ple.py`, `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/qwen4_exp.py`, `test/runtime/test_qwen4_exp_ple_backend.py` |

## 逐 PR diff 审计卡

### PR #1257 - feat: add Qwen3.8-flash-next support

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1257
- 状态/时间: merged / 2026-08-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/configs/qwen4_exp_config.py`, `python/tokenspeed/runtime/layers/attention/kv_cache/qwen4_exp.py`, `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/qwen4_exp.py`, `python/tokenspeed/runtime/models/qwen4_exp.py`, `python/tokenspeed/runtime/models/qwen4_exp_nextn.py` 等 7 个文件；关联提交 `ce67c25fe344`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 53 个文件，+12462/-84，可读 patch 13245 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/qwen4_exp.py` added +905/-0 (905 lines); hunks: -0,0 +1,905; symbols: _qwen4_exp_uses_sigmoid_output_gate, _qwen4_exp_uses_sparse_moe, _build_qwen4_exp_mlp, _Qwen4ExpRMSNormGated，涉及 `_qwen4_exp_uses_sigmoid_output_gate, _qwen4_exp_uses_sparse_moe, _build_qwen4_exp_mlp`；`python/tokenspeed/runtime/models/qwen4_exp_nextn.py` added +351/-0 (351 lines); hunks: -0,0 +1,351; symbols: _resolve_mtp_quant_config, _mtp_index_sharing_enabled, _build_mtp_lm_head, Qwen4ExpDraftAttentionDecoderLayer，涉及 `_resolve_mtp_quant_config, _mtp_index_sharing_enabled, _build_mtp_lm_head`；`python/tokenspeed/runtime/layers/attention/kv_cache/recipes/qwen4_exp.py` added +255/-0 (255 lines); hunks: -0,0 +1,255; symbols: Qwen4ExpRecipe, prefix_granularity, max_padding_fraction, _ple_fields，涉及 `Qwen4ExpRecipe, prefix_granularity, max_padding_fraction`；`python/tokenspeed/runtime/configs/qwen4_exp_config.py` added +192/-0 (192 lines); hunks: -0,0 +1,192; symbols: Qwen4ExpVisionConfig, Qwen4ExpTextConfig, __init__, layers_block_type，涉及 `Qwen4ExpVisionConfig, Qwen4ExpTextConfig, __init__`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/qwen4_exp.py` added +905/-0 (905 lines); hunks: -0,0 +1,905; symbols: _qwen4_exp_uses_sigmoid_output_gate, _qwen4_exp_uses_sparse_moe, _build_qwen4_exp_mlp, _Qwen4ExpRMSNormGated
  - `python/tokenspeed/runtime/models/qwen4_exp_nextn.py` added +351/-0 (351 lines); hunks: -0,0 +1,351; symbols: _resolve_mtp_quant_config, _mtp_index_sharing_enabled, _build_mtp_lm_head, Qwen4ExpDraftAttentionDecoderLayer
  - `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/qwen4_exp.py` added +255/-0 (255 lines); hunks: -0,0 +1,255; symbols: Qwen4ExpRecipe, prefix_granularity, max_padding_fraction, _ple_fields
  - `python/tokenspeed/runtime/configs/qwen4_exp_config.py` added +192/-0 (192 lines); hunks: -0,0 +1,192; symbols: Qwen4ExpVisionConfig, Qwen4ExpTextConfig, __init__, layers_block_type
  - `python/tokenspeed/runtime/layers/attention/backends/qwen4_exp.py` added +154/-0 (154 lines); hunks: -0,0 +1,154; symbols: qwen4_exp_linear_backend, Qwen4ExpMambaAttnBackend, __init__, _preallocate_aux_verify_workspace
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/qwen4_exp.py
@@ -0,0 +1,905 @@
+# Copyright (c) 2026 LightSeek Foundation
+#
+# Permission is hereby granted, free of charge, to any person obtaining a copy
+# of this software and associated documentation files (the "Software"), to deal
+# in the Software without restriction, including without limitation the rights
+# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
diff -- python/tokenspeed/runtime/models/qwen4_exp_nextn.py
@@ -0,0 +1,351 @@
+# Copyright (c) 2026 LightSeek Foundation
+#
+# Permission is hereby granted, free of charge, to any person obtaining a copy
+# of this software and associated documentation files (the "Software"), to deal
+# in the Software without restriction, including without limitation the rights
+# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
diff -- python/tokenspeed/runtime/layers/attention/kv_cache/recipes/qwen4_exp.py
@@ -0,0 +1,255 @@
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/qwen4_exp.py` added +905/-0; `python/tokenspeed/runtime/models/qwen4_exp_nextn.py` added +351/-0; `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/qwen4_exp.py` added +255/-0; `python/tokenspeed/runtime/configs/qwen4_exp_config.py` added +192/-0; `python/tokenspeed/runtime/layers/attention/backends/qwen4_exp.py` added +154/-0; `python/tokenspeed/runtime/layers/attention/kv_cache/qwen4_exp.py` added +72/-0
  - tests: `test/runtime/test_qwen4_exp.py` added +2424/-0; `tokenspeed-kernel/test/ops/test_qwen4_exp_qsa.py` added +1360/-0
- 验证与风险: diff 自带测试面 `test/runtime/execution/test_breakable_cuda_graph.py`, `test/runtime/layers/test_vocab_parallel_embedding.py`, `test/runtime/test_drafter_accept_indexing.py`, `test/runtime/test_fp8_linear_online_block_quant.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1278 - fix(L2): support L2 cache for qwen3.8-flash-next

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1278
- 状态/时间: merged / 2026-08-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/layers/attention/kv_cache/qwen4_exp.py`, `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/qwen4_exp.py`, `test/runtime/test_qwen4_exp.py`；关联提交 `6bfbd61f5870`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 19 个文件，+221/-55，可读 patch 749 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/layers/attention/kv_cache/qwen4_exp.py` modified +8/-3 (11 lines); hunks: -26,18 +26,23; -60,13 +65,13 @@ def qsa_rope_position_field(layer_id: int) -> str:; symbols: qwen4_exp_ple_context_field, qwen4_exp_ple_conv_field, qsa_raw_key_field, qsa_rope_position_field，涉及 `qwen4_exp_ple_context_field, qwen4_exp_ple_conv_field, qsa_raw_key_field`；`python/tokenspeed/runtime/layers/attention/kv_cache/recipes/qwen4_exp.py` modified +6/-4 (10 lines); hunks: -30,14 +30,14; -83,10 +83,12 @@ def max_padding_fraction(self) -> float:; symbols: max_padding_fraction, _ple_fields，涉及 `max_padding_fraction, _ple_fields`；`python/tokenspeed/runtime/layers/attention/backends/qwen4_exp.py` modified +2/-3 (5 lines); hunks: -33,7 +33,6; -96,10 +95,10 @@ def _ensure_ple_verify_scratch(self, max_bs: int, draft_toke...; symbols: _ensure_ple_verify_scratch, ple_verify_scratch，涉及 `_ensure_ple_verify_scratch, ple_verify_scratch`；`test/runtime/test_qwen4_exp.py` modified +49/-13 (62 lines); hunks: -25,6 +25,7; -69,11 +70,11; symbols: test_qwen4_exp_qsa_publishes_and_reuses_context_topk, fields，涉及 `test_qwen4_exp_qsa_publishes_and_reuses_context_topk, fields`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/layers/attention/kv_cache/qwen4_exp.py` modified +8/-3 (11 lines); hunks: -26,18 +26,23; -60,13 +65,13 @@ def qsa_rope_position_field(layer_id: int) -> str:; symbols: qwen4_exp_ple_context_field, qwen4_exp_ple_conv_field, qsa_raw_key_field, qsa_rope_position_field
  - `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/qwen4_exp.py` modified +6/-4 (10 lines); hunks: -30,14 +30,14; -83,10 +83,12 @@ def max_padding_fraction(self) -> float:; symbols: max_padding_fraction, _ple_fields
  - `python/tokenspeed/runtime/layers/attention/backends/qwen4_exp.py` modified +2/-3 (5 lines); hunks: -33,7 +33,6; -96,10 +95,10 @@ def _ensure_ple_verify_scratch(self, max_bs: int, draft_toke...; symbols: _ensure_ple_verify_scratch, ple_verify_scratch
  - `test/runtime/test_qwen4_exp.py` modified +49/-13 (62 lines); hunks: -25,6 +25,7; -69,11 +70,11; symbols: test_qwen4_exp_qsa_publishes_and_reuses_context_topk, fields
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/layers/attention/kv_cache/qwen4_exp.py
@@ -26,18 +26,23 @@
-QWEN4_EXP_PLE_CONTEXT_FIELD = "qwen4_exp.ple.context"
+def qwen4_exp_ple_context_field(layer_id: int) -> str:
+    """Return the shared PLE context field owned by its first consumer."""
+    return f"layer.{layer_id}.qwen4_exp.ple.context"
-    return f"qwen4_exp.ple.layer.{layer_id}.conv"
+    return f"layer.{layer_id}.qwen4_exp.ple.conv"
diff -- python/tokenspeed/runtime/layers/attention/kv_cache/recipes/qwen4_exp.py
@@ -30,14 +30,14 @@
-    QWEN4_EXP_PLE_CONTEXT_FIELD,
+    qwen4_exp_ple_context_field,
@@ -83,10 +83,12 @@ def max_padding_fraction(self) -> float:
+        ple_layer_ids = tuple(self._text_config.short_conv_layer_ids)
+        context_field = qwen4_exp_ple_context_field(min(ple_layer_ids))
-                QWEN4_EXP_PLE_CONTEXT_FIELD,
diff -- python/tokenspeed/runtime/layers/attention/backends/qwen4_exp.py
@@ -33,7 +33,6 @@
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/layers/attention/kv_cache/qwen4_exp.py` modified +8/-3; `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/qwen4_exp.py` modified +6/-4; `python/tokenspeed/runtime/layers/attention/backends/qwen4_exp.py` modified +2/-3
  - tests: `test/runtime/test_qwen4_exp.py` modified +49/-13
- 验证与风险: diff 自带测试面 `test/ci/README.md`, `test/ci/eval/qwen3.8-flash-next-fp8-evalscope-gsm8k.yaml`, `test/ci_system/test_eval_configs.py`, `test/runtime/test_cache_setup.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1323 - perf(qwen4): optimize gated residual kernels and PDL

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1323
- 状态/时间: merged / 2026-09-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/runtime/test_qwen4_exp.py`；关联提交 `a89524374466`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 14 个文件，+2232/-337，可读 patch 2870 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/runtime/test_qwen4_exp.py` modified +24/-83 (107 lines); hunks: -79,7 +79,7; -226,6 +226,7 @@ def test_qwen4_exp_flat_config_preserves_text_rope_parameter...; symbols: test_qwen4_exp_flat_config_preserves_text_rope_parameters, test_hyperconnection_mix_and_combine_shapes, test_hyperconnection_norm_for_reuses_the_mix_time_norm，涉及 `test_qwen4_exp_flat_config_preserves_text_rope_parameters, test_hyperconnection_mix_and_combine_shapes, test_hyperconnection_norm_for_reuses_the_mix_time_norm`；`python/tokenspeed/runtime/layers/hyperconnection.py` modified +63/-240 (303 lines); hunks: -26,9 +26,12; -43,6 +46,12 @@ class HyperConnectionConfig:; symbols: HyperConnectionConfig, __post_init__, _matching_rows, GroupedGemmaRMSNorm，涉及 `HyperConnectionConfig, __post_init__, _matching_rows`；`tokenspeed-kernel/python/tokenspeed_kernel/ops/hyperconnection/triton.py` added +610/-0 (610 lines); hunks: -0,0 +1,610; symbols: _projection_epilogue_kernel, _mix_epilogue_kernel, _combine_kernel, _grid_barrier，涉及 `_projection_epilogue_kernel, _mix_epilogue_kernel, _combine_kernel`；`tokenspeed-kernel/python/tokenspeed_kernel/ops/hyperconnection/__init__.py` added +297/-0 (297 lines); hunks: -0,0 +1,297; symbols: _flatten_rows, _same_tensor_contract, prepare_gated_residual_weight_cache, gated_residual_mix，涉及 `_flatten_rows, _same_tensor_contract, prepare_gated_residual_weight_cache`。
- 代码 diff 细节:
  - `test/runtime/test_qwen4_exp.py` modified +24/-83 (107 lines); hunks: -79,7 +79,7; -226,6 +226,7 @@ def test_qwen4_exp_flat_config_preserves_text_rope_parameter...; symbols: test_qwen4_exp_flat_config_preserves_text_rope_parameters, test_hyperconnection_mix_and_combine_shapes, test_hyperconnection_norm_for_reuses_the_mix_time_norm
  - `python/tokenspeed/runtime/layers/hyperconnection.py` modified +63/-240 (303 lines); hunks: -26,9 +26,12; -43,6 +46,12 @@ class HyperConnectionConfig:; symbols: HyperConnectionConfig, __post_init__, _matching_rows, GroupedGemmaRMSNorm
  - `tokenspeed-kernel/python/tokenspeed_kernel/ops/hyperconnection/triton.py` added +610/-0 (610 lines); hunks: -0,0 +1,610; symbols: _projection_epilogue_kernel, _mix_epilogue_kernel, _combine_kernel, _grid_barrier
  - `tokenspeed-kernel/python/tokenspeed_kernel/ops/hyperconnection/__init__.py` added +297/-0 (297 lines); hunks: -0,0 +1,297; symbols: _flatten_rows, _same_tensor_contract, prepare_gated_residual_weight_cache, gated_residual_mix
  - `tokenspeed-kernel/python/tokenspeed_kernel/ops/hyperconnection/cute_dsl.py` added +226/-0 (226 lines); hunks: -0,0 +1,226; symbols: _round_up, _CachedPaddedWeight, _copy_padded_up_weight, _prepare_padded_up_weight
- 关键代码摘录:

```diff
diff -- test/runtime/test_qwen4_exp.py
@@ -79,7 +79,7 @@
-    not torch.cuda.is_available(), reason="QSA indexer kernels require CUDA"
+    not torch.cuda.is_available(), reason="requires a CUDA device"
@@ -226,6 +226,7 @@ def test_qwen4_exp_flat_config_preserves_text_rope_parameters() -> None:
+@_requires_cuda
@@ -234,17 +235,18 @@ def test_hyperconnection_mix_and_combine_shapes() -> None:
-    )
diff -- python/tokenspeed/runtime/layers/hyperconnection.py
@@ -26,9 +26,12 @@
-import triton
-import triton.language as tl
-from tokenspeed_kernel.platform import pdl_enabled
+from tokenspeed_kernel import (
+    gated_residual_combine,
+    gated_residual_mix,
diff -- tokenspeed-kernel/python/tokenspeed_kernel/ops/hyperconnection/triton.py
@@ -0,0 +1,610 @@
```

- 提取文件（未人工审阅）:
  - tests: `test/runtime/test_qwen4_exp.py` modified +24/-83
  - runtime: `python/tokenspeed/runtime/layers/hyperconnection.py` modified +63/-240; `tokenspeed-kernel/python/tokenspeed_kernel/ops/hyperconnection/triton.py` added +610/-0; `tokenspeed-kernel/python/tokenspeed_kernel/ops/hyperconnection/__init__.py` added +297/-0; `tokenspeed-kernel/python/tokenspeed_kernel/ops/hyperconnection/cute_dsl.py` added +226/-0; `tokenspeed-kernel/python/tokenspeed_kernel/ops/layernorm/triton.py` modified +103/-0; `tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cute_dsl/ll_bf16/__init__.py` modified +57/-12
- 验证与风险: diff 自带测试面 `test/runtime/test_hyperconnection_kernel_boundary.py`, `test/runtime/test_qwen4_exp.py`, `tokenspeed-kernel/test/ops/bench_hyperconnection.py`, `tokenspeed-kernel/test/ops/gemm/test_ll_bf16_router.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1361 - refactor(ple): reorganize kernels and optimize qwen4-exp execution

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1361
- 状态/时间: merged / 2026-09-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/layers/qwen4_exp_ple.py`, `python/tokenspeed/runtime/models/qwen4_exp.py`, `test/runtime/test_qwen4_exp.py`；关联提交 `88d2e7a5b841`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 9 个文件，+1446/-664，可读 patch 2399 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/layers/qwen4_exp_ple.py` renamed +124/-587 (711 lines); hunks: -26,9 +26,13; -56,6 +60,15; symbols: _is_prime, quantize_ple_embedding_rows, _ngram_ids_kernel, _ple_page_gather_kernel，涉及 `_is_prime, quantize_ple_embedding_rows, _ngram_ids_kernel`；`python/tokenspeed/runtime/models/qwen4_exp.py` modified +5/-5 (10 lines); hunks: -55,6 +55,11; -73,11 +78,6；`python/tokenspeed/runtime/layers/attention/backends/qwen4_exp.py` modified +1/-1 (2 lines); hunks: -43,7 +43,7; symbols: qwen4_exp_linear_backend，涉及 `qwen4_exp_linear_backend`；`test/runtime/test_qwen4_exp.py` modified +42/-14 (56 lines); hunks: -59,6 +59,15; -68,15 +77,6; symbols: test_ngram_ids_anchor_rewrite_matches_legacy, test_ngram_ids_flat_kernel_matches_legacy, test_ple_final_context_matches_token_contexts, matches，涉及 `test_ngram_ids_anchor_rewrite_matches_legacy, test_ngram_ids_flat_kernel_matches_legacy, test_ple_final_context_matches_token_contexts`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/layers/qwen4_exp_ple.py` renamed +124/-587 (711 lines); hunks: -26,9 +26,13; -56,6 +60,15; symbols: _is_prime, quantize_ple_embedding_rows, _ngram_ids_kernel, _ple_page_gather_kernel
  - `python/tokenspeed/runtime/models/qwen4_exp.py` modified +5/-5 (10 lines); hunks: -55,6 +55,11; -73,11 +78,6
  - `python/tokenspeed/runtime/layers/attention/backends/qwen4_exp.py` modified +1/-1 (2 lines); hunks: -43,7 +43,7; symbols: qwen4_exp_linear_backend
  - `test/runtime/test_qwen4_exp.py` modified +42/-14 (56 lines); hunks: -59,6 +59,15; -68,15 +77,6; symbols: test_ngram_ids_anchor_rewrite_matches_legacy, test_ngram_ids_flat_kernel_matches_legacy, test_ple_final_context_matches_token_contexts, matches
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/layers/qwen4_exp_ple.py
@@ -26,9 +26,13 @@
-import triton
-import triton.language as tl
-from tokenspeed_kernel.platform import pdl_enabled
+from tokenspeed_kernel.ops.ple import (
+    ple_conv_sequences,
+    ple_gate_norm,
diff -- python/tokenspeed/runtime/models/qwen4_exp.py
@@ -55,6 +55,11 @@
+from tokenspeed.runtime.layers.qwen4_exp_ple import (
+    Qwen4ExpNGramEmbedding,
+    Qwen4ExpPLELayer,
+    quantize_ple_embedding_rows,
+)
@@ -73,11 +78,6 @@
diff -- python/tokenspeed/runtime/layers/attention/backends/qwen4_exp.py
@@ -43,7 +43,7 @@
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/layers/qwen4_exp_ple.py` renamed +124/-587; `python/tokenspeed/runtime/models/qwen4_exp.py` modified +5/-5; `python/tokenspeed/runtime/layers/attention/backends/qwen4_exp.py` modified +1/-1
  - tests: `test/runtime/test_qwen4_exp.py` modified +42/-14
- 验证与风险: diff 自带测试面 `test/gemm_tuning/tune_route.py`, `test/runtime/test_qwen4_exp.py`, `tokenspeed-kernel/test/ops/gemm/test_routed_gemv.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1416 - refactor(attention): compose Qwen4-Exp QSA and PLE backends

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1416
- 状态/时间: merged / 2026-09-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/layers/attention/backends/specific/qwen4_exp.py`, `python/tokenspeed/runtime/layers/attention/backends/specific/qwen4_exp_ple.py`, `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/qwen4_exp.py`, `python/tokenspeed/runtime/layers/qwen4_exp_ple.py`, `python/tokenspeed/runtime/models/qwen4_exp.py` 等 10 个文件；关联提交 `d3b2998c79e1`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 55 个文件，+5564/-2131，可读 patch 9777 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/layers/attention/backends/specific/qwen4_exp_ple.py` added +473/-0 (473 lines); hunks: -0,0 +1,473; symbols: PLEForwardMetadata, Qwen4ExpPLEBackend, __init__, _cache_fields，涉及 `PLEForwardMetadata, Qwen4ExpPLEBackend, __init__`；`python/tokenspeed/runtime/layers/attention/backends/specific/qwen4_exp.py` modified +172/-138 (310 lines); hunks: -18,155 +18,189; symbols: qwen4_exp_linear_backend, Qwen4ExpMambaAttnBackend, __init__, _preallocate_aux_verify_workspace，涉及 `qwen4_exp_linear_backend, Qwen4ExpMambaAttnBackend, __init__`；`python/tokenspeed/runtime/layers/qwen4_exp_ple.py` modified +25/-104 (129 lines); hunks: -23,7 +23,6; -45,7 +44,11; symbols: __init__, _load_kv_proj_shard, _linear_backend, _ple_backend，涉及 `__init__, _load_kv_proj_shard, _linear_backend`；`python/tokenspeed/runtime/layers/attention/kv_cache/recipes/qwen4_exp.py` modified +70/-24 (94 lines); hunks: -136,18 +136,23 @@ def _qsa_compress_ratio(self) -> int:; -167,10 +172,7 @@ def _qsa_fields(; symbols: _qsa_compress_ratio, _qsa_target_layers, _qsa_layers, _qsa_fields，涉及 `_qsa_compress_ratio, _qsa_target_layers, _qsa_layers`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/layers/attention/backends/specific/qwen4_exp_ple.py` added +473/-0 (473 lines); hunks: -0,0 +1,473; symbols: PLEForwardMetadata, Qwen4ExpPLEBackend, __init__, _cache_fields
  - `python/tokenspeed/runtime/layers/attention/backends/specific/qwen4_exp.py` modified +172/-138 (310 lines); hunks: -18,155 +18,189; symbols: qwen4_exp_linear_backend, Qwen4ExpMambaAttnBackend, __init__, _preallocate_aux_verify_workspace
  - `python/tokenspeed/runtime/layers/qwen4_exp_ple.py` modified +25/-104 (129 lines); hunks: -23,7 +23,6; -45,7 +44,11; symbols: __init__, _load_kv_proj_shard, _linear_backend, _ple_backend
  - `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/qwen4_exp.py` modified +70/-24 (94 lines); hunks: -136,18 +136,23 @@ def _qsa_compress_ratio(self) -> int:; -167,10 +172,7 @@ def _qsa_fields(; symbols: _qsa_compress_ratio, _qsa_target_layers, _qsa_layers, _qsa_fields
  - `python/tokenspeed/runtime/models/qwen4_exp_nextn.py` modified +28/-23 (51 lines); hunks: -95,42 +95,47 @@ def _attn(; symbols: _attn, _qsa_attention, forward
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/layers/attention/backends/specific/qwen4_exp_ple.py
@@ -0,0 +1,473 @@
+# Copyright (c) 2026 LightSeek Foundation
+#
+# Permission is hereby granted, free of charge, to any person obtaining a copy
+# of this software and associated documentation files (the "Software"), to deal
+# in the Software without restriction, including without limitation the rights
+# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
diff -- python/tokenspeed/runtime/layers/attention/backends/specific/qwen4_exp.py
@@ -18,155 +18,189 @@
-"""Qwen4-Exp extensions for the hybrid GDN attention backend."""
+"""Qwen4-Exp composition of attention, PLE and QSA cache consumers."""
-from collections.abc import Iterable
-from typing_extensions import override
-from tokenspeed.runtime.layers.attention.backends.base import CudaGraphSupport
-from tokenspeed.runtime.layers.attention.backends.state.mamba import (
diff -- python/tokenspeed/runtime/layers/qwen4_exp_ple.py
@@ -23,7 +23,6 @@
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/layers/attention/backends/specific/qwen4_exp_ple.py` added +473/-0; `python/tokenspeed/runtime/layers/attention/backends/specific/qwen4_exp.py` modified +172/-138; `python/tokenspeed/runtime/layers/qwen4_exp_ple.py` modified +25/-104; `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/qwen4_exp.py` modified +70/-24; `python/tokenspeed/runtime/models/qwen4_exp_nextn.py` modified +28/-23; `python/tokenspeed/runtime/models/qwen4_exp.py` modified +6/-35
  - tests: `test/runtime/test_qwen4_exp.py` modified +495/-500; `test/runtime/test_qwen4_exp_ple_backend.py` added +343/-0
- 验证与风险: diff 自带测试面 `test/runtime/distributed/test_draft_moe_capture_global_bs.py`, `test/runtime/models/test_qwen3_moe_models.py`, `test/runtime/test_cache_group_router.py`, `test/runtime/test_cache_setup.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1362 - feat(ple): support qwen3.8-flash-next PLE offloading

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1362
- 状态/时间: merged / 2026-09-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/configs/qwen4_exp_config.py`, `python/tokenspeed/runtime/layers/qwen4_exp_ple.py`, `python/tokenspeed/runtime/models/qwen4_exp.py`, `test/runtime/test_qwen4_exp.py`, `test/runtime/test_qwen4_exp_ple_prefetch_timing.py`；关联提交 `b19403208c0e`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 16 个文件，+2818/-327，可读 patch 3949 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/layers/qwen4_exp_ple.py` modified +251/-179 (430 lines); hunks: -23,6 +23,7; -31,12 +32,13; symbols: _nth_prime_after, quantize_ple_embedding_rows, Qwen4ExpNGramEmbedding, __init__，涉及 `_nth_prime_after, quantize_ple_embedding_rows, Qwen4ExpNGramEmbedding`；`python/tokenspeed/runtime/models/qwen4_exp.py` modified +34/-63 (97 lines); hunks: -23,7 +23,6; -37,6 +36,10; symbols: load_kv_cache_scales, _start_ple_prefetch, forward, _copy_ple_shard，涉及 `load_kv_cache_scales, _start_ple_prefetch, forward`；`python/tokenspeed/runtime/configs/qwen4_exp_config.py` modified +12/-0 (12 lines); hunks: -22,6 +22,8; -55,6 +57,7 @@ def __init__(; symbols: __init__，涉及 `__init__`；`test/runtime/test_qwen4_exp.py` modified +588/-20 (608 lines); hunks: -20,10 +20,16; -34,6 +40,7; symbols: test_qwen4_exp_config_normalizes_layer_and_ple_geometry, test_qwen4_exp_ple_offload_default, test_qwen4_exp_flat_config_preserves_text_rope_parameters, _ple_layer_stub，涉及 `test_qwen4_exp_config_normalizes_layer_and_ple_geometry, test_qwen4_exp_ple_offload_default, test_qwen4_exp_flat_config_preserves_text_rope_parameters`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/layers/qwen4_exp_ple.py` modified +251/-179 (430 lines); hunks: -23,6 +23,7; -31,12 +32,13; symbols: _nth_prime_after, quantize_ple_embedding_rows, Qwen4ExpNGramEmbedding, __init__
  - `python/tokenspeed/runtime/models/qwen4_exp.py` modified +34/-63 (97 lines); hunks: -23,7 +23,6; -37,6 +36,10; symbols: load_kv_cache_scales, _start_ple_prefetch, forward, _copy_ple_shard
  - `python/tokenspeed/runtime/configs/qwen4_exp_config.py` modified +12/-0 (12 lines); hunks: -22,6 +22,8; -55,6 +57,7 @@ def __init__(; symbols: __init__
  - `test/runtime/test_qwen4_exp.py` modified +588/-20 (608 lines); hunks: -20,10 +20,16; -34,6 +40,7; symbols: test_qwen4_exp_config_normalizes_layer_and_ple_geometry, test_qwen4_exp_ple_offload_default, test_qwen4_exp_flat_config_preserves_text_rope_parameters, _ple_layer_stub
  - `test/runtime/test_qwen4_exp_ple_prefetch_timing.py` added +130/-0 (130 lines); hunks: -0,0 +1,130; symbols: _RecordingPLE, __init__, start_prefetch, _RecordingLayer
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/layers/qwen4_exp_ple.py
@@ -23,6 +23,7 @@
+from typing import NamedTuple
@@ -31,12 +32,13 @@
+    ple_page_gather_pair,
+    prepare_ngram_reciprocals,
-from tokenspeed.runtime.distributed.comm_ops import all_reduce
@@ -57,12 +59,15 @@
diff -- python/tokenspeed/runtime/models/qwen4_exp.py
@@ -23,7 +23,6 @@
-import math
@@ -37,6 +36,10 @@
+from tokenspeed.runtime.execution.breakable_cuda_graph import (
+    BreakableCapture,
+    current_forward_ctx,
+)
diff -- python/tokenspeed/runtime/configs/qwen4_exp_config.py
@@ -22,6 +22,8 @@
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/layers/qwen4_exp_ple.py` modified +251/-179; `python/tokenspeed/runtime/models/qwen4_exp.py` modified +34/-63; `python/tokenspeed/runtime/configs/qwen4_exp_config.py` modified +12/-0
  - tests: `test/runtime/test_qwen4_exp.py` modified +588/-20; `test/runtime/test_qwen4_exp_ple_prefetch_timing.py` added +130/-0
- 验证与风险: diff 自带测试面 `test/runtime/test_hyperconnection_kernel_boundary.py`, `test/runtime/test_ple_global_lookup.py`, `test/runtime/test_ple_optimization_perf.py`, `test/runtime/test_qwen4_exp.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1797 - feat(attention): let Qwen4-Exp's PLE and QSA indexer take a replacement pool

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1797
- 状态/时间: merged / 2026-09-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/layers/attention/backends/specific/qwen4_exp_ple.py`, `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/qwen4_exp.py`, `test/runtime/test_qwen4_exp_ple_backend.py`；关联提交 `b4f5fc883db6`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 10 个文件，+57/-65，可读 patch 265 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/layers/attention/backends/specific/qwen4_exp_ple.py` modified +4/-2 (6 lines); hunks: -150,8 +150,6 @@ def _cache_fields(; -164,6 +162,10 @@ def _publish_cache_pool(self, cache_pool: CachePool) -> None:; symbols: _cache_fields, validate_cache_pool, _publish_cache_pool, _block_rows，涉及 `_cache_fields, validate_cache_pool, _publish_cache_pool`；`python/tokenspeed/runtime/layers/attention/kv_cache/recipes/qwen4_exp.py` modified +0/-5 (5 lines); hunks: -202,11 +202,6 @@ def _qsa_fields(; symbols: _qsa_fields, backends_accept_pool_replacement, workspace_bytes，涉及 `_qsa_fields, backends_accept_pool_replacement, workspace_bytes`；`test/runtime/test_qwen4_exp_ple_backend.py` modified +17/-5 (22 lines); hunks: -94,17 +94,29 @@ def backend():; symbols: backend, test_ple_rebind_rejection_preserves_verify_workspace, test_ple_rebind_rebuilds_the_commit_tables_on_the_new_arena, test_ple_invalid_fields_do_not_publish_a_pool，涉及 `backend, test_ple_rebind_rejection_preserves_verify_workspace, test_ple_rebind_rebuilds_the_commit_tables_on_the_new_arena`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/layers/attention/backends/specific/qwen4_exp_ple.py` modified +4/-2 (6 lines); hunks: -150,8 +150,6 @@ def _cache_fields(; -164,6 +162,10 @@ def _publish_cache_pool(self, cache_pool: CachePool) -> None:; symbols: _cache_fields, validate_cache_pool, _publish_cache_pool, _block_rows
  - `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/qwen4_exp.py` modified +0/-5 (5 lines); hunks: -202,11 +202,6 @@ def _qsa_fields(; symbols: _qsa_fields, backends_accept_pool_replacement, workspace_bytes
  - `test/runtime/test_qwen4_exp_ple_backend.py` modified +17/-5 (22 lines); hunks: -94,17 +94,29 @@ def backend():; symbols: backend, test_ple_rebind_rejection_preserves_verify_workspace, test_ple_rebind_rebuilds_the_commit_tables_on_the_new_arena, test_ple_invalid_fields_do_not_publish_a_pool
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/layers/attention/backends/specific/qwen4_exp_ple.py
@@ -150,8 +150,6 @@ def _cache_fields(
-        if self.cache_pool is not None and self.cache_pool is not cache_pool:
-            raise RuntimeError("PLE backend cannot be rebound to another cache pool")
@@ -164,6 +162,10 @@ def _publish_cache_pool(self, cache_pool: CachePool) -> None:
+        # The commit tables point into the old arena; preallocation rebuilds them.
+        self._ple_verify_tables = None
+        self._ple_commit_rows = None
diff -- python/tokenspeed/runtime/layers/attention/kv_cache/recipes/qwen4_exp.py
@@ -202,11 +202,6 @@ def _qsa_fields(
-    @override
-    def backends_accept_pool_replacement(self) -> bool:
-        """The PLE and QSA indexer children latch their pool at construction."""
-        return False
diff -- test/runtime/test_qwen4_exp_ple_backend.py
@@ -94,17 +94,29 @@ def backend():
-def test_ple_rebind_rejection_preserves_verify_workspace(backend):
+def test_ple_rebind_rebuilds_the_commit_tables_on_the_new_arena(backend):
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/layers/attention/backends/specific/qwen4_exp_ple.py` modified +4/-2; `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/qwen4_exp.py` modified +0/-5
  - tests: `test/runtime/test_qwen4_exp_ple_backend.py` modified +17/-5
- 验证与风险: diff 自带测试面 `test/runtime/test_cudagraph_probe_arena.py`, `test/runtime/test_qsa_backend.py`, `test/runtime/test_qwen4_exp_ple_backend.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
