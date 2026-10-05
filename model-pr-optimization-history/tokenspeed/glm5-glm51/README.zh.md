# TokenSpeed GLM-5 Series (5/5.1/5.2/5.3-Flash) 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `python/tokenspeed/runtime/configs/glm53_flash_config.py` | [#1259](https://github.com/lightseekorg/tokenspeed/pull/1259) |
| `python/tokenspeed/runtime/layers/attention/kv_cache/hybrid_glm53_flash.py` | [#1259](https://github.com/lightseekorg/tokenspeed/pull/1259) |
| `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/glm53_flash.py` | [#1259](https://github.com/lightseekorg/tokenspeed/pull/1259) |
| `python/tokenspeed/runtime/models/glm5.py` | [#348](https://github.com/lightseekorg/tokenspeed/pull/348), [#509](https://github.com/lightseekorg/tokenspeed/pull/509), [#528](https://github.com/lightseekorg/tokenspeed/pull/528), [#533](https://github.com/lightseekorg/tokenspeed/pull/533), [#554](https://github.com/lightseekorg/tokenspeed/pull/554), [#586](https://github.com/lightseekorg/tokenspeed/pull/586), [#590](https://github.com/lightseekorg/tokenspeed/pull/590), [#599](https://github.com/lightseekorg/tokenspeed/pull/599), [#835](https://github.com/lightseekorg/tokenspeed/pull/835), [#1240](https://github.com/lightseekorg/tokenspeed/pull/1240), [#1259](https://github.com/lightseekorg/tokenspeed/pull/1259), [#1840](https://github.com/lightseekorg/tokenspeed/pull/1840) |
| `python/tokenspeed/runtime/models/glm53_flash.py` | [#1259](https://github.com/lightseekorg/tokenspeed/pull/1259), [#1429](https://github.com/lightseekorg/tokenspeed/pull/1429) |
| `python/tokenspeed/runtime/models/glm53_flash_nextn.py` | [#1259](https://github.com/lightseekorg/tokenspeed/pull/1259) |
| `python/tokenspeed/runtime/models/glm_moe_dsa_nextn.py` | [#1259](https://github.com/lightseekorg/tokenspeed/pull/1259) |
| `test/agentic_benchmark/glm5.2/tokenspeed/agentic_bench.sh` | [#532](https://github.com/lightseekorg/tokenspeed/pull/532) |
| `test/agentic_benchmark/glm5.2/tokenspeed/collect_outputs.py` | [#532](https://github.com/lightseekorg/tokenspeed/pull/532) |
| `test/agentic_benchmark/glm5.2/tokenspeed/configs/attn_tp4_moe_ep4.sh` | [#532](https://github.com/lightseekorg/tokenspeed/pull/532) |
| `test/agentic_benchmark/glm5.2/tokenspeed/configs/attn_tp4_moe_tp4.sh` | [#532](https://github.com/lightseekorg/tokenspeed/pull/532) |
| `test/agentic_benchmark/glm5.2/tokenspeed/configs/attn_tp8_moe_ep8.sh` | [#532](https://github.com/lightseekorg/tokenspeed/pull/532) |
| `test/agentic_benchmark/glm5.2/tokenspeed/configs/attn_tp8_moe_tp8.sh` | [#532](https://github.com/lightseekorg/tokenspeed/pull/532) |
| `test/agentic_benchmark/glm5.2/trtllm/agentic_bench.sh` | [#532](https://github.com/lightseekorg/tokenspeed/pull/532), [#581](https://github.com/lightseekorg/tokenspeed/pull/581) |
| `test/agentic_benchmark/glm5.2/trtllm/collect_outputs.py` | [#532](https://github.com/lightseekorg/tokenspeed/pull/532) |
| `test/agentic_benchmark/glm5.2/trtllm/configs/attn_tp4_moe_ep4.yaml` | [#532](https://github.com/lightseekorg/tokenspeed/pull/532) |
| `test/agentic_benchmark/glm5.2/trtllm/configs/attn_tp4_moe_tp4.yaml` | [#532](https://github.com/lightseekorg/tokenspeed/pull/532) |
| `test/agentic_benchmark/glm5.2/trtllm/configs/attn_tp8_moe_ep8.yaml` | [#532](https://github.com/lightseekorg/tokenspeed/pull/532) |
| `test/agentic_benchmark/glm5.2/trtllm/configs/attn_tp8_moe_tp8.yaml` | [#532](https://github.com/lightseekorg/tokenspeed/pull/532) |
| `test/ci/eval/glm-5.2-nvfp4-mtp-evalscope-aime26.yaml` | [#572](https://github.com/lightseekorg/tokenspeed/pull/572), [#1942](https://github.com/lightseekorg/tokenspeed/pull/1942) |
| `test/ci/eval/glm-5.3-flash-fp8-mtp-evalscope-aime26-amd.yaml` | [#1259](https://github.com/lightseekorg/tokenspeed/pull/1259) |
| `test/ci/eval/glm-5.3-flash-fp8-mtp-evalscope-aime26.yaml` | [#1259](https://github.com/lightseekorg/tokenspeed/pull/1259), [#1581](https://github.com/lightseekorg/tokenspeed/pull/1581) |
| `test/runtime/models/test_glm53_flash_models.py` | [#1259](https://github.com/lightseekorg/tokenspeed/pull/1259), [#1429](https://github.com/lightseekorg/tokenspeed/pull/1429) |
| `test/runtime/test_glm53_flash_cache_pool.py` | [#1259](https://github.com/lightseekorg/tokenspeed/pull/1259) |
| `test/runtime/test_glm53_flash_cache_spec.py` | [#1259](https://github.com/lightseekorg/tokenspeed/pull/1259) |
| `test/runtime/test_glm53_flash_config.py` | [#1259](https://github.com/lightseekorg/tokenspeed/pull/1259) |
| `tokenspeed-kernel/benchmarks/amd/gfx950/glm53_flash/dsa.json` | [#1721](https://github.com/lightseekorg/tokenspeed/pull/1721) |
| `tokenspeed-kernel/benchmarks/amd/gfx950/glm53_flash/kda.json` | [#1721](https://github.com/lightseekorg/tokenspeed/pull/1721) |
| `tokenspeed-kernel/benchmarks/amd/gfx950/glm53_flash/moe.json` | [#1723](https://github.com/lightseekorg/tokenspeed/pull/1723) |

## PR 覆盖总览

- git 追溯 PR 数: 20
- 原文档显式引用补充 PR 数: 0
- 当前文档总 PR 数: 20
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-06-22 | [#348](https://github.com/lightseekorg/tokenspeed/pull/348) | merged | [Model] GLM-5 support: DSA sparse attention, MTP speculative decoding, CUDA graph decode | `python/tokenspeed/runtime/models/glm5.py` |
| 2026-06-25 | [#509](https://github.com/lightseekorg/tokenspeed/pull/509) | merged | Fix: GLM5 DSA MTP TP4 IMA Error | `python/tokenspeed/runtime/models/glm5.py` |
| 2026-06-26 | [#533](https://github.com/lightseekorg/tokenspeed/pull/533) | merged | fix(glm5.2): fix indexer_topk_prefil params for batched long prefill IMA | `python/tokenspeed/runtime/models/glm5.py` |
| 2026-06-29 | [#554](https://github.com/lightseekorg/tokenspeed/pull/554) | merged | fix(glm-5.2): remove unnecessary DSA KV-cache and flashmla DSA path | `python/tokenspeed/runtime/models/glm5.py` |
| 2026-07-01 | [#528](https://github.com/lightseekorg/tokenspeed/pull/528) | merged | Initial glm 5.2 support on amd | `python/tokenspeed/runtime/models/glm5.py` |
| 2026-07-01 | [#572](https://github.com/lightseekorg/tokenspeed/pull/572) | merged | feat(glm-5.2): support hierarchical (host/L2) KV cache for DSA | `test/ci/eval/glm-5.2-nvfp4-mtp-evalscope-aime26.yaml`, `python/tokenspeed/runtime/layers/attention/kv_cache/dsa.py`, `python/tokenspeed/runtime/layers/attention/configs/dsa.py` |
| 2026-07-02 | [#532](https://github.com/lightseekorg/tokenspeed/pull/532) | merged | test: glm-5.2 agentic bench | `test/agentic_benchmark/glm5.2/tokenspeed/configs/attn_tp4_moe_ep4.sh`, `test/agentic_benchmark/glm5.2/tokenspeed/configs/attn_tp4_moe_tp4.sh`, `test/agentic_benchmark/glm5.2/tokenspeed/configs/attn_tp8_moe_ep8.sh` |
| 2026-07-02 | [#581](https://github.com/lightseekorg/tokenspeed/pull/581) | merged | test: fix glm-5.2 agentic bench | `test/agentic_benchmark/glm5.2/trtllm/agentic_bench.sh` |
| 2026-07-03 | [#586](https://github.com/lightseekorg/tokenspeed/pull/586) | merged | perf(glm-5.2): drop full-topk path and hadamard transform, use flashinfer LayerNorm for DSA decode path | `python/tokenspeed/runtime/models/glm5.py` |
| 2026-07-04 | [#590](https://github.com/lightseekorg/tokenspeed/pull/590) | merged | chore(glm-5.2): cleanup bf16 index_k cache for DSA | `python/tokenspeed/runtime/models/glm5.py` |
| 2026-07-07 | [#599](https://github.com/lightseekorg/tokenspeed/pull/599) | merged | perf(glm-5.2): remove per-token expansion of seq_lens and block_table for DSA decode path | `python/tokenspeed/runtime/models/glm5.py` |
| 2026-07-29 | [#835](https://github.com/lightseekorg/tokenspeed/pull/835) | merged | fix(glm5.2): dsa with bf16 kv cache on amd | `python/tokenspeed/runtime/models/glm5.py` |
| 2026-08-27 | [#1240](https://github.com/lightseekorg/tokenspeed/pull/1240) | merged | perf(glm5): reuse DSA prefill plans across indexer layers | `python/tokenspeed/runtime/models/glm5.py` |
| 2026-09-01 | [#1259](https://github.com/lightseekorg/tokenspeed/pull/1259) | merged | feat(glm-5.3-flash): add day-0 serving support | `python/tokenspeed/runtime/models/glm53_flash.py`, `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/glm53_flash.py`, `test/runtime/models/test_glm53_flash_models.py` |
| 2026-09-08 | [#1429](https://github.com/lightseekorg/tokenspeed/pull/1429) | merged | perf(amd): Optimize GLM-5.3-Flash KPool decode and tail handling | `test/runtime/models/test_glm53_flash_models.py`, `python/tokenspeed/runtime/models/glm53_flash.py` |
| 2026-09-15 | [#1581](https://github.com/lightseekorg/tokenspeed/pull/1581) | merged | Update SMG pins with GLM-5.3 Flash grammar compatibility | `test/ci/eval/glm-5.3-flash-fp8-mtp-evalscope-aime26.yaml`, `python/pyproject.toml` |
| 2026-09-23 | [#1721](https://github.com/lightseekorg/tokenspeed/pull/1721) | merged | ci(amd-kernel): Add GLM 5.3 Flash DSA Kernel Benchmarks | `tokenspeed-kernel/benchmarks/amd/gfx950/glm53_flash/dsa.json`, `tokenspeed-kernel/benchmarks/amd/gfx950/glm53_flash/kda.json` |
| 2026-09-23 | [#1723](https://github.com/lightseekorg/tokenspeed/pull/1723) | merged | ci(amd-kernel): Add GLM 5.3 Flash MoE Kernel Benchmarks | `tokenspeed-kernel/benchmarks/amd/gfx950/glm53_flash/moe.json` |
| 2026-09-28 | [#1840](https://github.com/lightseekorg/tokenspeed/pull/1840) | merged | fix(glm5): pass explicit batch invariance to DSA top-k | `python/tokenspeed/runtime/models/glm5.py` |
| 2026-10-03 | [#1942](https://github.com/lightseekorg/tokenspeed/pull/1942) | merged | ci: make glm-5.2 nvfp4 aime26 eval manual only | `test/ci/eval/glm-5.2-nvfp4-mtp-evalscope-aime26.yaml` |

## 逐 PR diff 审计卡

### PR #348 - [Model] GLM-5 support: DSA sparse attention, MTP speculative decoding, CUDA graph decode

- 链接: https://github.com/lightseekorg/tokenspeed/pull/348
- 状态/时间: merged / 2026-06-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/glm5.py`；关联提交 `1492030a2a02`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 23 个文件，+5363/-50，可读 patch 5779 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/glm5.py` added +2108/-0 (2108 lines); hunks: -0,0 +1,2108; symbols: GlmDsaIndexerOutput, GlmDsaPrefillTopK, GlmDsaDecodeTopK, GlmDsaDecodeWindow，涉及 `GlmDsaIndexerOutput, GlmDsaPrefillTopK, GlmDsaDecodeTopK`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/glm5.py` added +2108/-0 (2108 lines); hunks: -0,0 +1,2108; symbols: GlmDsaIndexerOutput, GlmDsaPrefillTopK, GlmDsaDecodeTopK, GlmDsaDecodeWindow
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/glm5.py
@@ -0,0 +1,2108 @@
+# Copyright (c) 2026 LightSeek Foundation
+#
+# Permission is hereby granted, free of charge, to any person obtaining a copy
+# of this software and associated documentation files (the "Software"), to deal
+# in the Software without restriction, including without limitation the rights
+# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/glm5.py` added +2108/-0
- 验证与风险: diff 自带测试面 `test/runtime/test_dp_sampling_routing_metadata.py`, `tokenspeed-kernel/test/test_kernel_api_selection.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #509 - Fix: GLM5 DSA MTP TP4 IMA Error

- 链接: https://github.com/lightseekorg/tokenspeed/pull/509
- 状态/时间: merged / 2026-06-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/glm5.py`；关联提交 `38ad3ed9b70f`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 7 个文件，+215/-39，可读 patch 423 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/glm5.py` modified +17/-4 (21 lines); hunks: -1089,7 +1089,14 @@ def _compute_decode_topk_indices_deepgemm(; -1106,6 +1113,11 @@ def _compute_decode_topk_indices_deepgemm(; symbols: _compute_decode_topk_indices_deepgemm, forward，涉及 `_compute_decode_topk_indices_deepgemm, forward`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/glm5.py` modified +17/-4 (21 lines); hunks: -1089,7 +1089,14 @@ def _compute_decode_topk_indices_deepgemm(; -1106,6 +1113,11 @@ def _compute_decode_topk_indices_deepgemm(; symbols: _compute_decode_topk_indices_deepgemm, forward
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/glm5.py
@@ -1089,7 +1089,14 @@ def _compute_decode_topk_indices_deepgemm(
-        if schedule_metadata is None:
+        schedule_shape = tuple(seq_lens_2d.shape)
+        if (
+            schedule_metadata is None
+            or getattr(decode_metadata, "_dsa_paged_mqa_schedule_q_len", None)
+            != q_len_per_req
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/glm5.py` modified +17/-4
- 验证与风险: diff 自带测试面 `test/runtime/test_deepseek_v4_config.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #533 - fix(glm5.2): fix indexer_topk_prefil params for batched long prefill IMA

- 链接: https://github.com/lightseekorg/tokenspeed/pull/533
- 状态/时间: merged / 2026-06-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/glm5.py`；关联提交 `3e7fd51bfb4a`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+6/-4，可读 patch 30 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/glm5.py` modified +6/-4 (10 lines); hunks: -1352,14 +1352,16 @@ def _compute_prefill_topk_indices_deepgemm(; -1370,8 +1372,8 @@ def _compute_prefill_topk_indices_deepgemm(; symbols: _compute_prefill_topk_indices_deepgemm，涉及 `_compute_prefill_topk_indices_deepgemm`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/glm5.py` modified +6/-4 (10 lines); hunks: -1352,14 +1352,16 @@ def _compute_prefill_topk_indices_deepgemm(; -1370,8 +1372,8 @@ def _compute_prefill_topk_indices_deepgemm(; symbols: _compute_prefill_topk_indices_deepgemm
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/glm5.py
@@ -1352,14 +1352,16 @@ def _compute_prefill_topk_indices_deepgemm(
+        local_starts_i32 = torch.zeros_like(row_starts_i32)
+        causal_lens_i32 = causal_lens.to(torch.int32).contiguous()
-                row_starts_i32[start:end].contiguous(),
-                row_ends_i32[start:end].contiguous(),
+                row_starts_i32[start:end],
+                row_ends_i32[start:end],
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/glm5.py` modified +6/-4
- 验证与风险: runtime 路径改动集中在 `python/tokenspeed/runtime/models/glm5.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #554 - fix(glm-5.2): remove unnecessary DSA KV-cache and flashmla DSA path

- 链接: https://github.com/lightseekorg/tokenspeed/pull/554
- 状态/时间: merged / 2026-06-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/glm5.py`；关联提交 `8a9a2c656045`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 8 个文件，+82/-693，可读 patch 1114 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/glm5.py` modified +10/-18 (28 lines); hunks: -1036,8 +1036,8 @@ def _compute_decode_topk_indices_deepgemm(; -1118,14 +1118,12 @@ def _compute_decode_topk_indices_deepgemm(; symbols: _compute_decode_topk_indices_deepgemm, _compute_prefill_topk_indices, _compute_prefill_topk_indices_deepgemm，涉及 `_compute_decode_topk_indices_deepgemm, _compute_prefill_topk_indices, _compute_prefill_topk_indices_deepgemm`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/glm5.py` modified +10/-18 (28 lines); hunks: -1036,8 +1036,8 @@ def _compute_decode_topk_indices_deepgemm(; -1118,14 +1118,12 @@ def _compute_decode_topk_indices_deepgemm(; symbols: _compute_decode_topk_indices_deepgemm, _compute_prefill_topk_indices, _compute_prefill_topk_indices_deepgemm
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/glm5.py
@@ -1036,8 +1036,8 @@ def _compute_decode_topk_indices_deepgemm(
-            not hasattr(ctx.token_to_kv_pool, "has_index_k_with_scale_buffer")
-            or not ctx.token_to_kv_pool.has_index_k_with_scale_buffer()
+            not hasattr(ctx.token_to_kv_pool, "has_index_k_buffer")
+            or not ctx.token_to_kv_pool.has_index_k_buffer()
@@ -1118,14 +1118,12 @@ def _compute_decode_topk_indices_deepgemm(
-        index_k_with_scale_cache = ctx.token_to_kv_pool.get_index_k_with_scale_buffer(
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/glm5.py` modified +10/-18
- 验证与风险: runtime 路径改动集中在 `python/tokenspeed/runtime/layers/attention/backends/base.py`, `python/tokenspeed/runtime/layers/attention/backends/dsa.py`, `python/tokenspeed/runtime/layers/attention/backends/tokenspeed_mla.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #528 - Initial glm 5.2 support on amd

- 链接: https://github.com/lightseekorg/tokenspeed/pull/528
- 状态/时间: merged / 2026-07-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/glm5.py`；关联提交 `20e9dad7127a`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 44 个文件，+6435/-1291，可读 patch 8843 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/glm5.py` modified +108/-320 (428 lines); hunks: -27,28 +27,18; -151,7 +141,7 @@ def _build_prefill_kv_workspace_slots(; symbols: _build_prefill_kv_workspace_slots, _glm_dsa_rope_scaling, _glm_dsa_hadamard_rotate，涉及 `_build_prefill_kv_workspace_slots, _glm_dsa_rope_scaling, _glm_dsa_hadamard_rotate`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/glm5.py` modified +108/-320 (428 lines); hunks: -27,28 +27,18; -151,7 +141,7 @@ def _build_prefill_kv_workspace_slots(; symbols: _build_prefill_kv_workspace_slots, _glm_dsa_rope_scaling, _glm_dsa_hadamard_rotate
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/glm5.py
@@ -27,28 +27,18 @@
+from tokenspeed_kernel.ops.attention import (
+    dsa_decode_topk,
+    dsa_plan,
+    dsa_prefill_topk,
+)
-    local_topk_to_global_slots,
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/glm5.py` modified +108/-320
- 验证与风险: diff 自带测试面 `test/runtime/test_deepseek_v4_attention_ops.py`, `tokenspeed-kernel/test/ops/test_attention.py`, `tokenspeed-kernel/test/ops/test_attention_dsa.py`, `tokenspeed-kernel/test/ops/test_attention_mla.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #572 - feat(glm-5.2): support hierarchical (host/L2) KV cache for DSA

- 链接: https://github.com/lightseekorg/tokenspeed/pull/572
- 状态/时间: merged / 2026-07-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/ci/eval/glm-5.2-nvfp4-mtp-evalscope-aime26.yaml`；关联提交 `99d20dd8270d`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 8 个文件，+412/-151，可读 patch 790 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/ci/eval/glm-5.2-nvfp4-mtp-evalscope-aime26.yaml` modified +1/-1 (2 lines); hunks: -21,7 +21,7 @@ server:；`python/tokenspeed/runtime/layers/attention/kv_cache/dsa.py` modified +30/-53 (83 lines); hunks: -34,39 +34,32; -75,9 +68,7 @@ def _get_page_size_bytes(self):; symbols: DSATokenToKVPool, __init__, _get_page_size_bytes, get_kv_size_bytes，涉及 `DSATokenToKVPool, __init__, _get_page_size_bytes`；`python/tokenspeed/runtime/layers/attention/configs/dsa.py` modified +2/-3 (5 lines); hunks: -35,10 +35,9; symbols: dsa_index_k_row_bytes，涉及 `dsa_index_k_row_bytes`；`python/tokenspeed/runtime/cache/kv_cache_host.py` modified +129/-0 (129 lines); hunks: -34,6 +34,7; -797,3 +798,131 @@ def get_page_buffer_meta(self, indices):; symbols: get_page_buffer_meta, DSATokenToKVPoolHost, __init__, get_size_per_token，涉及 `get_page_buffer_meta, DSATokenToKVPoolHost, __init__`。
- 代码 diff 细节:
  - `test/ci/eval/glm-5.2-nvfp4-mtp-evalscope-aime26.yaml` modified +1/-1 (2 lines); hunks: -21,7 +21,7 @@ server:
  - `python/tokenspeed/runtime/layers/attention/kv_cache/dsa.py` modified +30/-53 (83 lines); hunks: -34,39 +34,32; -75,9 +68,7 @@ def _get_page_size_bytes(self):; symbols: DSATokenToKVPool, __init__, _get_page_size_bytes, get_kv_size_bytes
  - `python/tokenspeed/runtime/layers/attention/configs/dsa.py` modified +2/-3 (5 lines); hunks: -35,10 +35,9; symbols: dsa_index_k_row_bytes
  - `python/tokenspeed/runtime/cache/kv_cache_host.py` modified +129/-0 (129 lines); hunks: -34,6 +34,7; -797,3 +798,131 @@ def get_page_buffer_meta(self, indices):; symbols: get_page_buffer_meta, DSATokenToKVPoolHost, __init__, get_size_per_token
  - `tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cuda/csrc/kvcacheio_transfer.cu` modified +62/-49 (111 lines); hunks: -108,14 +108,15 @@ inline void copy_token_span(; -203,7 +204,7 @@ __device__ __forceinline__ T* get_global_offset_ph(
- 关键代码摘录:

```diff
diff -- test/ci/eval/glm-5.2-nvfp4-mtp-evalscope-aime26.yaml
@@ -21,7 +21,7 @@ server:
-    --gpu-memory-utilization 0.9
+    --gpu-memory-utilization 0.95
diff -- python/tokenspeed/runtime/layers/attention/kv_cache/dsa.py
@@ -34,39 +34,32 @@
-    # KVStore currently transfers only the base MLA KV rows. DSA also needs
-    # sparse/indexer cache rows to stay coherent with the reused KV pages.
-    supports_hierarchical_kv_cache = False
-        self.index_k_row_bytes = dsa_index_k_row_bytes(
-            self.index_head_dim,
-        )
diff -- python/tokenspeed/runtime/layers/attention/configs/dsa.py
@@ -35,10 +35,9 @@
-    if index_head_dim % _INDEX_K_FP8_GROUP_SIZE != 0:
+    if index_head_dim <= 0 or index_head_dim % _INDEX_K_FP8_GROUP_SIZE != 0:
-            "DSA index_head_dim must be divisible by "
-            f"{_INDEX_K_FP8_GROUP_SIZE}, got {index_head_dim}"
```

- 提取文件（未人工审阅）:
  - tests: `test/ci/eval/glm-5.2-nvfp4-mtp-evalscope-aime26.yaml` modified +1/-1
  - runtime: `python/tokenspeed/runtime/layers/attention/kv_cache/dsa.py` modified +30/-53; `python/tokenspeed/runtime/layers/attention/configs/dsa.py` modified +2/-3; `python/tokenspeed/runtime/cache/kv_cache_host.py` modified +129/-0; `tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cuda/csrc/kvcacheio_transfer.cu` modified +62/-49; `python/tokenspeed/runtime/cache/executor/memory_executor.py` modified +26/-27; `python/tokenspeed/cli/serve_smg.py` modified +13/-18
- 验证与风险: diff 自带测试面 `test/ci/eval/glm-5.2-nvfp4-mtp-evalscope-aime26.yaml`, `test/runtime/cache/test_dsa_kvstore_host.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #532 - test: glm-5.2 agentic bench

- 链接: https://github.com/lightseekorg/tokenspeed/pull/532
- 状态/时间: merged / 2026-07-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/agentic_benchmark/glm5.2/tokenspeed/agentic_bench.sh`, `test/agentic_benchmark/glm5.2/tokenspeed/collect_outputs.py`, `test/agentic_benchmark/glm5.2/tokenspeed/configs/attn_tp4_moe_ep4.sh`, `test/agentic_benchmark/glm5.2/tokenspeed/configs/attn_tp4_moe_tp4.sh`, `test/agentic_benchmark/glm5.2/tokenspeed/configs/attn_tp8_moe_ep8.sh` 等 12 个文件；关联提交 `1f42267db0e4`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 41 个文件，+617/-3，可读 patch 653 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/agentic_benchmark/glm5.2/tokenspeed/configs/attn_tp4_moe_ep4.sh` added +26/-0 (26 lines); hunks: -0,0 +1,26；`test/agentic_benchmark/glm5.2/tokenspeed/configs/attn_tp4_moe_tp4.sh` added +26/-0 (26 lines); hunks: -0,0 +1,26；`test/agentic_benchmark/glm5.2/tokenspeed/configs/attn_tp8_moe_ep8.sh` added +26/-0 (26 lines); hunks: -0,0 +1,26；`test/agentic_benchmark/glm5.2/tokenspeed/configs/attn_tp8_moe_tp8.sh` added +26/-0 (26 lines); hunks: -0,0 +1,26。
- 代码 diff 细节:
  - `test/agentic_benchmark/glm5.2/tokenspeed/configs/attn_tp4_moe_ep4.sh` added +26/-0 (26 lines); hunks: -0,0 +1,26
  - `test/agentic_benchmark/glm5.2/tokenspeed/configs/attn_tp4_moe_tp4.sh` added +26/-0 (26 lines); hunks: -0,0 +1,26
  - `test/agentic_benchmark/glm5.2/tokenspeed/configs/attn_tp8_moe_ep8.sh` added +26/-0 (26 lines); hunks: -0,0 +1,26
  - `test/agentic_benchmark/glm5.2/tokenspeed/configs/attn_tp8_moe_tp8.sh` added +26/-0 (26 lines); hunks: -0,0 +1,26
  - `test/agentic_benchmark/glm5.2/trtllm/configs/attn_tp4_moe_ep4.yaml` added +13/-0 (13 lines); hunks: -0,0 +1,13
- 关键代码摘录:

```diff
diff -- test/agentic_benchmark/glm5.2/tokenspeed/configs/attn_tp4_moe_ep4.sh
@@ -0,0 +1,26 @@
+#!/usr/bin/bash
+set -euo pipefail
+exec ts serve \
+    --model nvidia/GLM-5.2-NVFP4 \
+    --attn-tp-size 4 \
+    --ep-size 4 \
diff -- test/agentic_benchmark/glm5.2/tokenspeed/configs/attn_tp4_moe_tp4.sh
@@ -0,0 +1,26 @@
+#!/usr/bin/bash
+set -euo pipefail
+exec ts serve \
+    --model nvidia/GLM-5.2-NVFP4 \
+    --attn-tp-size 4 \
+    --moe-tp-size 4 \
diff -- test/agentic_benchmark/glm5.2/tokenspeed/configs/attn_tp8_moe_ep8.sh
@@ -0,0 +1,26 @@
```

- 提取文件（未人工审阅）:
  - tests: `test/agentic_benchmark/glm5.2/tokenspeed/configs/attn_tp4_moe_ep4.sh` added +26/-0; `test/agentic_benchmark/glm5.2/tokenspeed/configs/attn_tp4_moe_tp4.sh` added +26/-0; `test/agentic_benchmark/glm5.2/tokenspeed/configs/attn_tp8_moe_ep8.sh` added +26/-0; `test/agentic_benchmark/glm5.2/tokenspeed/configs/attn_tp8_moe_tp8.sh` added +26/-0; `test/agentic_benchmark/glm5.2/trtllm/configs/attn_tp4_moe_ep4.yaml` added +13/-0; `test/agentic_benchmark/glm5.2/trtllm/configs/attn_tp4_moe_tp4.yaml` added +13/-0
- 验证与风险: diff 自带测试面 `test/agentic_benchmark/glm5.2/tokenspeed/agentic_bench.sh`, `test/agentic_benchmark/glm5.2/tokenspeed/collect_outputs.py`, `test/agentic_benchmark/glm5.2/tokenspeed/configs/attn_tp4_moe_ep4.sh`, `test/agentic_benchmark/glm5.2/tokenspeed/configs/attn_tp4_moe_tp4.sh`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #581 - test: fix glm-5.2 agentic bench

- 链接: https://github.com/lightseekorg/tokenspeed/pull/581
- 状态/时间: merged / 2026-07-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/agentic_benchmark/glm5.2/trtllm/agentic_bench.sh`；关联提交 `676f19eb57a6`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+5/-5，可读 patch 44 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/agentic_benchmark/glm5.2/trtllm/agentic_bench.sh` modified +5/-5 (10 lines); hunks: -1,6 +1,6; -13,7 +13,7 @@ pip install "evalscope[perf] @ git+https://github.com/modelsco...。
- 代码 diff 细节:
  - `test/agentic_benchmark/glm5.2/trtllm/agentic_bench.sh` modified +5/-5 (10 lines); hunks: -1,6 +1,6; -13,7 +13,7 @@ pip install "evalscope[perf] @ git+https://github.com/modelsco...
- 关键代码摘录:

```diff
diff -- test/agentic_benchmark/glm5.2/trtllm/agentic_bench.sh
@@ -1,6 +1,6 @@
-# Tested on TensorRT-LLM commit: 70c5e43
+# Tested on TensorRT-LLM commit: 281acfd
@@ -13,7 +13,7 @@ pip install "evalscope[perf] @ git+https://github.com/modelscope/evalscope.git@$
-    --model zai-org/GLM-5.1 \
+    --model zai-org/GLM-5.2 \
@@ -37,7 +37,7 @@ SERVER_LOG=
```

- 提取文件（未人工审阅）:
  - tests: `test/agentic_benchmark/glm5.2/trtllm/agentic_bench.sh` modified +5/-5
- 验证与风险: diff 自带测试面 `test/agentic_benchmark/glm5.2/trtllm/agentic_bench.sh`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #586 - perf(glm-5.2): drop full-topk path and hadamard transform, use flashinfer LayerNorm for DSA decode path

- 链接: https://github.com/lightseekorg/tokenspeed/pull/586
- 状态/时间: merged / 2026-07-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/glm5.py`；关联提交 `4241bb9ba291`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 10 个文件，+303/-1027，可读 patch 1576 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/glm5.py` modified +42/-464 (506 lines); hunks: -32,10 +32,6; -44,7 +40,7; symbols: GlmDsaDecodeWindow, _glm_dsa_is_decode_token_mode, _glm_dsa_is_pure_decode_token_mode, _glm_dsa_skip_indexer_topk，涉及 `GlmDsaDecodeWindow, _glm_dsa_is_decode_token_mode, _glm_dsa_is_pure_decode_token_mode`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/glm5.py` modified +42/-464 (506 lines); hunks: -32,10 +32,6; -44,7 +40,7; symbols: GlmDsaDecodeWindow, _glm_dsa_is_decode_token_mode, _glm_dsa_is_pure_decode_token_mode, _glm_dsa_skip_indexer_topk
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/glm5.py
@@ -32,10 +32,6 @@
-from tokenspeed_kernel.ops.attention.triton.dsa_sparse_layout import (
-    full_context_topk_to_global_slots,
-)
-from tokenspeed_kernel.ops.transform import hadamard_transform
@@ -44,7 +40,7 @@
-from tokenspeed.runtime.layers.layernorm import FusedRMSNorm, RMSNorm
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/glm5.py` modified +42/-464
- 验证与风险: diff 自带测试面 `tokenspeed-kernel/test/ops/test_attention_dsa.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #590 - chore(glm-5.2): cleanup bf16 index_k cache for DSA

- 链接: https://github.com/lightseekorg/tokenspeed/pull/590
- 状态/时间: merged / 2026-07-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/glm5.py`；关联提交 `7b50e13f605f`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 6 个文件，+52/-762，可读 patch 1167 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/glm5.py` modified +4/-28 (32 lines); hunks: -648,18 +648,8 @@ def _compute_decode_topk_indices_portable(; -687,7 +677,6 @@ def _compute_decode_topk_indices_portable(; symbols: _compute_decode_topk_indices_portable, _compute_prefill_topk_indices_portable，涉及 `_compute_decode_topk_indices_portable, _compute_prefill_topk_indices_portable`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/glm5.py` modified +4/-28 (32 lines); hunks: -648,18 +648,8 @@ def _compute_decode_topk_indices_portable(; -687,7 +677,6 @@ def _compute_decode_topk_indices_portable(; symbols: _compute_decode_topk_indices_portable, _compute_prefill_topk_indices_portable
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/glm5.py
@@ -648,18 +648,8 @@ def _compute_decode_topk_indices_portable(
-        index_k_cache = (
-            ctx.token_to_kv_pool.get_index_k_buffer(self.attn_mqa.layer_id)
-            if hasattr(ctx.token_to_kv_pool, "get_index_k_buffer")
-            else None
-        )
-        index_k_with_scale_cache = (
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/glm5.py` modified +4/-28
- 验证与风险: diff 自带测试面 `tokenspeed-kernel/test/ops/test_attention_dsa.py`, `tokenspeed-kernel/test/test_kernel_api_selection.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #599 - perf(glm-5.2): remove per-token expansion of seq_lens and block_table for DSA decode path

- 链接: https://github.com/lightseekorg/tokenspeed/pull/599
- 状态/时间: merged / 2026-07-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/glm5.py`；关联提交 `c8a92feb697b`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 13 个文件，+294/-306，可读 patch 1171 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/glm5.py` modified +17/-99 (116 lines); hunks: -29,7 +29,6; -496,38 +495,6 @@ def _retire_decode_workspace(self, buffer: torch.Tensor) ->...; symbols: _retire_decode_workspace, _expand_decode_seq_lens_per_token, _check_decode_q_len_per_req, _compute_decode_topk_indices，涉及 `_retire_decode_workspace, _expand_decode_seq_lens_per_token, _check_decode_q_len_per_req`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/glm5.py` modified +17/-99 (116 lines); hunks: -29,7 +29,6; -496,38 +495,6 @@ def _retire_decode_workspace(self, buffer: torch.Tensor) ->...; symbols: _retire_decode_workspace, _expand_decode_seq_lens_per_token, _check_decode_q_len_per_req, _compute_decode_topk_indices
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/glm5.py
@@ -29,7 +29,6 @@
-    dsa_plan,
@@ -496,38 +495,6 @@ def _retire_decode_workspace(self, buffer: torch.Tensor) -> None:
-    @staticmethod
-    def _expand_decode_seq_lens_per_token(
-        seq_lens: torch.Tensor,
-        q_len_per_req: int,
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/glm5.py` modified +17/-99
- 验证与风险: diff 自带测试面 `tokenspeed-kernel/test/ops/test_attention_dsa.py`, `tokenspeed-kernel/test/ops/test_dsa_topk.py`, `tokenspeed-kernel/test/test_kernel_api_selection.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #835 - fix(glm5.2): dsa with bf16 kv cache on amd

- 链接: https://github.com/lightseekorg/tokenspeed/pull/835
- 状态/时间: merged / 2026-07-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/glm5.py`；关联提交 `246fd901aeaf`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/glm5.py` modified +1/-1 (2 lines); hunks: -988,7 +988,7 @@ def forward_absorb_attn_v_proj(; symbols: forward_absorb_attn_v_proj，涉及 `forward_absorb_attn_v_proj`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/glm5.py` modified +1/-1 (2 lines); hunks: -988,7 +988,7 @@ def forward_absorb_attn_v_proj(; symbols: forward_absorb_attn_v_proj
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/glm5.py
@@ -988,7 +988,7 @@ def forward_absorb_attn_v_proj(
-            K[..., : self.kv_lora_rank],
+            K[..., : self.kv_lora_rank] if K is not None else None,
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/glm5.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `python/tokenspeed/runtime/models/glm5.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #1240 - perf(glm5): reuse DSA prefill plans across indexer layers

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1240
- 状态/时间: merged / 2026-08-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/glm5.py`；关联提交 `327deb8096f5`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 7 个文件，+185/-45，可读 patch 386 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/glm5.py` modified +74/-43 (117 lines); hunks: -87,6 +87,9 @@ class GlmDsaPrefillTopK:; -603,61 +606,112 @@ def _compute_prefill_topk_indices(; symbols: GlmDsaPrefillTopK, _compute_prefill_topk_indices, _compute_prefill_topk_indices_portable，涉及 `GlmDsaPrefillTopK, _compute_prefill_topk_indices, _compute_prefill_topk_indices_portable`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/glm5.py` modified +74/-43 (117 lines); hunks: -87,6 +87,9 @@ class GlmDsaPrefillTopK:; -603,61 +606,112 @@ def _compute_prefill_topk_indices(; symbols: GlmDsaPrefillTopK, _compute_prefill_topk_indices, _compute_prefill_topk_indices_portable
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/glm5.py
@@ -87,6 +87,9 @@ class GlmDsaPrefillTopK:
+    row_starts: torch.Tensor
+    row_ends: torch.Tensor
+    candidate_lens_cpu: torch.Tensor
@@ -603,61 +606,112 @@ def _compute_prefill_topk_indices(
+        cached = ctx.dsa_prefill_topk
+        if (
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/glm5.py` modified +74/-43
- 验证与风险: diff 自带测试面 `test/runtime/test_page_table_conversion.py`, `tokenspeed-kernel/test/test_kernel_api_selection.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1259 - feat(glm-5.3-flash): add day-0 serving support

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1259
- 状态/时间: merged / 2026-09-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/configs/glm53_flash_config.py`, `python/tokenspeed/runtime/layers/attention/kv_cache/hybrid_glm53_flash.py`, `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/glm53_flash.py`, `python/tokenspeed/runtime/models/glm5.py`, `python/tokenspeed/runtime/models/glm53_flash.py` 等 13 个文件；关联提交 `268f06b30e40`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 120 个文件，+18860/-814，可读 patch 22699 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/glm53_flash.py` added +1933/-0 (1933 lines); hunks: -0,0 +1,1933; symbols: Glm53FlashVisionPatchEmbed, __init__, forward, Glm53FlashVisionRotaryEmbedding，涉及 `Glm53FlashVisionPatchEmbed, __init__, forward`；`python/tokenspeed/runtime/layers/attention/kv_cache/recipes/glm53_flash.py` added +457/-0 (457 lines); hunks: -0,0 +1,457; symbols: Glm53FlashPoolOptions, tail_width, workspace_bytes, _require_non_negative_int，涉及 `Glm53FlashPoolOptions, tail_width, workspace_bytes`；`test/runtime/models/test_glm53_flash_models.py` added +426/-0 (426 lines); hunks: -0,0 +1,426; symbols: _FakeBackbone, __init__, get_input_embeddings, _FakeLanguageModel，涉及 `_FakeBackbone, __init__, get_input_embeddings`；`python/tokenspeed/runtime/configs/glm53_flash_config.py` added +386/-0 (386 lines); hunks: -0,0 +1,386; symbols: Glm53FlashVisionConfig, __init__, Glm53FlashTextConfig, is_kda_layer，涉及 `Glm53FlashVisionConfig, __init__, Glm53FlashTextConfig`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/glm53_flash.py` added +1933/-0 (1933 lines); hunks: -0,0 +1,1933; symbols: Glm53FlashVisionPatchEmbed, __init__, forward, Glm53FlashVisionRotaryEmbedding
  - `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/glm53_flash.py` added +457/-0 (457 lines); hunks: -0,0 +1,457; symbols: Glm53FlashPoolOptions, tail_width, workspace_bytes, _require_non_negative_int
  - `test/runtime/models/test_glm53_flash_models.py` added +426/-0 (426 lines); hunks: -0,0 +1,426; symbols: _FakeBackbone, __init__, get_input_embeddings, _FakeLanguageModel
  - `python/tokenspeed/runtime/configs/glm53_flash_config.py` added +386/-0 (386 lines); hunks: -0,0 +1,386; symbols: Glm53FlashVisionConfig, __init__, Glm53FlashTextConfig, is_kda_layer
  - `python/tokenspeed/runtime/models/glm53_flash_nextn.py` added +336/-0 (336 lines); hunks: -0,0 +1,336; symbols: Glm53FlashModelNextN, __init__, forward, Glm53FlashForConditionalGenerationNextN
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/glm53_flash.py
@@ -0,0 +1,1933 @@
+# Copyright (c) 2026 LightSeek Foundation
+#
+# Permission is hereby granted, free of charge, to any person obtaining a copy
+# of this software and associated documentation files (the "Software"), to deal
+# in the Software without restriction, including without limitation the rights
+# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
diff -- python/tokenspeed/runtime/layers/attention/kv_cache/recipes/glm53_flash.py
@@ -0,0 +1,457 @@
+# Copyright (c) 2026 LightSeek Foundation
+#
+# Permission is hereby granted, free of charge, to any person obtaining a copy
+# of this software and associated documentation files (the "Software"), to deal
+# in the Software without restriction, including without limitation the rights
+# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
diff -- test/runtime/models/test_glm53_flash_models.py
@@ -0,0 +1,426 @@
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/glm53_flash.py` added +1933/-0; `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/glm53_flash.py` added +457/-0; `python/tokenspeed/runtime/configs/glm53_flash_config.py` added +386/-0; `python/tokenspeed/runtime/models/glm53_flash_nextn.py` added +336/-0; `python/tokenspeed/runtime/layers/attention/kv_cache/hybrid_glm53_flash.py` added +213/-0; `python/tokenspeed/runtime/models/glm_moe_dsa_nextn.py` renamed +6/-61
  - tests: `test/runtime/models/test_glm53_flash_models.py` added +426/-0
- 验证与风险: diff 自带测试面 `test/ci/README.md`, `test/ci/eval/glm-5.3-flash-fp8-mtp-evalscope-aime26-amd.yaml`, `test/ci/eval/glm-5.3-flash-fp8-mtp-evalscope-aime26.yaml`, `test/ci/eval/gpt-oss-120b-mxfp4-evalscope-gpqa-diamond.yaml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1429 - perf(amd): Optimize GLM-5.3-Flash KPool decode and tail handling

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1429
- 状态/时间: merged / 2026-09-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/glm53_flash.py`, `test/runtime/models/test_glm53_flash_models.py`；关联提交 `ae43c5bc5fa5`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 7 个文件，+463/-24，可读 patch 667 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/runtime/models/test_glm53_flash_models.py` modified +121/-0 (121 lines); hunks: -4,6 +4,7; -13,6 +14,8; symbols: _tiny_config, test_kpool_decode_forwards_configured_context_bound, fake_kpool_decode_topk, test_glm53_flash_decode_topk_skips_only_overwritten_workspace_fills，涉及 `_tiny_config, test_kpool_decode_forwards_configured_context_bound, fake_kpool_decode_topk`；`python/tokenspeed/runtime/models/glm53_flash.py` modified +3/-0 (3 lines); hunks: -1178,15 +1178,18 @@ def _compute_decode_topk_indices_portable(; symbols: _compute_decode_topk_indices_portable，涉及 `_compute_decode_topk_indices_portable`。
- 代码 diff 细节:
  - `test/runtime/models/test_glm53_flash_models.py` modified +121/-0 (121 lines); hunks: -4,6 +4,7; -13,6 +14,8; symbols: _tiny_config, test_kpool_decode_forwards_configured_context_bound, fake_kpool_decode_topk, test_glm53_flash_decode_topk_skips_only_overwritten_workspace_fills
  - `python/tokenspeed/runtime/models/glm53_flash.py` modified +3/-0 (3 lines); hunks: -1178,15 +1178,18 @@ def _compute_decode_topk_indices_portable(; symbols: _compute_decode_topk_indices_portable
- 关键代码摘录:

```diff
diff -- test/runtime/models/test_glm53_flash_models.py
@@ -4,6 +4,7 @@
+import pytest
@@ -13,6 +14,8 @@
+from tokenspeed.runtime.layers.attention import kpool as kpool_runtime
+from tokenspeed.runtime.layers.attention.kpool import KPoolRuntime
@@ -22,6 +25,7 @@
+    Glm53FlashIndexerOutput,
diff -- python/tokenspeed/runtime/models/glm53_flash.py
@@ -1178,15 +1178,18 @@ def _compute_decode_topk_indices_portable(
+        writes_full_workspace = decode_start == 0 and num_decode_tokens == num_tokens
+            fill_value=None if writes_full_workspace else -1,
+            fill=not writes_full_workspace,
```

- 提取文件（未人工审阅）:
  - tests: `test/runtime/models/test_glm53_flash_models.py` modified +121/-0
  - runtime: `python/tokenspeed/runtime/models/glm53_flash.py` modified +3/-0
- 验证与风险: diff 自带测试面 `test/runtime/models/test_glm53_flash_models.py`, `tokenspeed-kernel/test/ops/test_kpool_cache.py`, `tokenspeed-kernel/test/ops/test_kpool_select.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1581 - Update SMG pins with GLM-5.3 Flash grammar compatibility

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1581
- 状态/时间: merged / 2026-09-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/ci/eval/glm-5.3-flash-fp8-mtp-evalscope-aime26.yaml`；关联提交 `b7f46ff23f39`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+4/-3，可读 patch 21 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/ci/eval/glm-5.3-flash-fp8-mtp-evalscope-aime26.yaml` modified +1/-0 (1 lines); hunks: -31,6 +31,7 @@ server:；`python/pyproject.toml` modified +3/-3 (6 lines); hunks: -72,9 +72,9 @@ dependencies = [。
- 代码 diff 细节:
  - `test/ci/eval/glm-5.3-flash-fp8-mtp-evalscope-aime26.yaml` modified +1/-0 (1 lines); hunks: -31,6 +31,7 @@ server:
  - `python/pyproject.toml` modified +3/-3 (6 lines); hunks: -72,9 +72,9 @@ dependencies = [
- 关键代码摘录:

```diff
diff -- test/ci/eval/glm-5.3-flash-fp8-mtp-evalscope-aime26.yaml
@@ -31,6 +31,7 @@ server:
+    --grammar-backend xgrammar
diff -- python/pyproject.toml
@@ -72,9 +72,9 @@ dependencies = [
-    "tokenspeed-smg==1.10.1.post20260913",
-    "tokenspeed-smg-grpc-proto==0.4.18.post20260913",
-    "tokenspeed-smg-grpc-servicer==0.9.1.post20260913",
+    "tokenspeed-smg==1.10.1.post20260915",
+    "tokenspeed-smg-grpc-proto==0.4.18.post20260915",
+    "tokenspeed-smg-grpc-servicer==0.9.1.post20260915",
```

- 提取文件（未人工审阅）:
  - tests: `test/ci/eval/glm-5.3-flash-fp8-mtp-evalscope-aime26.yaml` modified +1/-0
  - runtime: `python/pyproject.toml` modified +3/-3
- 验证与风险: diff 自带测试面 `test/ci/eval/glm-5.3-flash-fp8-mtp-evalscope-aime26.yaml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1721 - ci(amd-kernel): Add GLM 5.3 Flash DSA Kernel Benchmarks

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1721
- 状态/时间: merged / 2026-09-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tokenspeed-kernel/benchmarks/amd/gfx950/glm53_flash/dsa.json`, `tokenspeed-kernel/benchmarks/amd/gfx950/glm53_flash/kda.json`；关联提交 `297e1540e060`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 11 个文件，+1879/-4，可读 patch 1949 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tokenspeed-kernel/benchmarks/amd/gfx950/glm53_flash/dsa.json` added +530/-0 (530 lines); hunks: -0,0 +1,530；`tokenspeed-kernel/benchmarks/amd/gfx950/glm53_flash/kda.json` renamed +0/-0 (0 lines)。
- 代码 diff 细节:
  - `tokenspeed-kernel/benchmarks/amd/gfx950/glm53_flash/dsa.json` added +530/-0 (530 lines); hunks: -0,0 +1,530
  - `tokenspeed-kernel/benchmarks/amd/gfx950/glm53_flash/kda.json` renamed +0/-0 (0 lines)
- 关键代码摘录:

```diff
diff -- tokenspeed-kernel/benchmarks/amd/gfx950/glm53_flash/dsa.json
@@ -0,0 +1,530 @@
+{
+  "schema_version": 1,
+  "common_parameters": {
+    "model_profile": "glm53_flash_tp4",
+    "dtype": "bfloat16",
+    "kv_cache_dtype": "bfloat16",
```

- 提取文件（未人工审阅）:
  - runtime: `tokenspeed-kernel/benchmarks/amd/gfx950/glm53_flash/dsa.json` added +530/-0; `tokenspeed-kernel/benchmarks/amd/gfx950/glm53_flash/kda.json` renamed +0/-0
- 验证与风险: diff 自带测试面 `tokenspeed-kernel/test/test_benchmark_ci.py`, `tokenspeed-kernel/test/test_benchmark_dsa_generator.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1723 - ci(amd-kernel): Add GLM 5.3 Flash MoE Kernel Benchmarks

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1723
- 状态/时间: merged / 2026-09-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tokenspeed-kernel/benchmarks/amd/gfx950/glm53_flash/moe.json`；关联提交 `a657e4770cb6`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 5 个文件，+1072/-1，可读 patch 1097 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tokenspeed-kernel/benchmarks/amd/gfx950/glm53_flash/moe.json` added +283/-0 (283 lines); hunks: -0,0 +1,283。
- 代码 diff 细节:
  - `tokenspeed-kernel/benchmarks/amd/gfx950/glm53_flash/moe.json` added +283/-0 (283 lines); hunks: -0,0 +1,283
- 关键代码摘录:

```diff
diff -- tokenspeed-kernel/benchmarks/amd/gfx950/glm53_flash/moe.json
@@ -0,0 +1,283 @@
+{
+  "schema_version": 1,
+  "common_parameters": {
+    "model_profile": "glm53_flash_tp4",
+    "num_experts": 288,
+    "topk": 8,
```

- 提取文件（未人工审阅）:
  - runtime: `tokenspeed-kernel/benchmarks/amd/gfx950/glm53_flash/moe.json` added +283/-0
- 验证与风险: diff 自带测试面 `tokenspeed-kernel/test/test_benchmark_moe_generator.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1840 - fix(glm5): pass explicit batch invariance to DSA top-k

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1840
- 状态/时间: merged / 2026-09-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/glm5.py`；关联提交 `3dc3ca120384`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+2/-0，可读 patch 16 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/glm5.py` modified +2/-0 (2 lines); hunks: -627,6 +627,7 @@ def _compute_decode_topk_indices_portable(; -807,6 +808,7 @@ def _compute_prefill_topk_indices_portable(; symbols: _compute_decode_topk_indices_portable, _compute_prefill_topk_indices_portable，涉及 `_compute_decode_topk_indices_portable, _compute_prefill_topk_indices_portable`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/glm5.py` modified +2/-0 (2 lines); hunks: -627,6 +627,7 @@ def _compute_decode_topk_indices_portable(; -807,6 +808,7 @@ def _compute_prefill_topk_indices_portable(; symbols: _compute_decode_topk_indices_portable, _compute_prefill_topk_indices_portable
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/glm5.py
@@ -627,6 +627,7 @@ def _compute_decode_topk_indices_portable(
+                batch_invariant=False,
@@ -807,6 +808,7 @@ def _compute_prefill_topk_indices_portable(
+                batch_invariant=False,
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/glm5.py` modified +2/-0
- 验证与风险: runtime 路径改动集中在 `python/tokenspeed/runtime/models/glm5.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #1942 - ci: make glm-5.2 nvfp4 aime26 eval manual only

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1942
- 状态/时间: merged / 2026-10-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/ci/eval/glm-5.2-nvfp4-mtp-evalscope-aime26.yaml`；关联提交 `80660378e929`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+0/-1，可读 patch 8 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/ci/eval/glm-5.2-nvfp4-mtp-evalscope-aime26.yaml` modified +0/-1 (1 lines); hunks: -3,7 +3,6 @@ name: eval-glm-5.2-nvfp4-mtp-aime26。
- 代码 diff 细节:
  - `test/ci/eval/glm-5.2-nvfp4-mtp-evalscope-aime26.yaml` modified +0/-1 (1 lines); hunks: -3,7 +3,6 @@ name: eval-glm-5.2-nvfp4-mtp-aime26
- 关键代码摘录:

```diff
diff -- test/ci/eval/glm-5.2-nvfp4-mtp-evalscope-aime26.yaml
@@ -3,7 +3,6 @@ name: eval-glm-5.2-nvfp4-mtp-aime26
-  - per-commit
```

- 提取文件（未人工审阅）:
  - tests: `test/ci/eval/glm-5.2-nvfp4-mtp-evalscope-aime26.yaml` modified +0/-1
- 验证与风险: diff 自带测试面 `test/ci/eval/glm-5.2-nvfp4-mtp-evalscope-aime26.yaml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
