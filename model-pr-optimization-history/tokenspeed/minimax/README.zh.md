# TokenSpeed MiniMax M2/M3 Series 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `python/tokenspeed/runtime/configs/minimax_m3_config.py` | [#733](https://github.com/lightseekorg/tokenspeed/pull/733) |
| `python/tokenspeed/runtime/models/minimax_m3.py` | [#733](https://github.com/lightseekorg/tokenspeed/pull/733), [#779](https://github.com/lightseekorg/tokenspeed/pull/779), [#781](https://github.com/lightseekorg/tokenspeed/pull/781) |
| `test/agentic_benchmark/minimax_m3/tokenspeed/agentic_bench.sh` | [#785](https://github.com/lightseekorg/tokenspeed/pull/785) |
| `test/agentic_benchmark/minimax_m3/tokenspeed/collect_outputs.py` | [#785](https://github.com/lightseekorg/tokenspeed/pull/785) |
| `test/agentic_benchmark/minimax_m3/tokenspeed/configs/attn_tp4_moe_ep4.sh` | [#785](https://github.com/lightseekorg/tokenspeed/pull/785) |
| `test/agentic_benchmark/minimax_m3/tokenspeed/configs/attn_tp4_moe_tp4.sh` | [#785](https://github.com/lightseekorg/tokenspeed/pull/785) |
| `test/ci/eval/minimax-m3-nvfp4-dspark-evalscope-aime26.yaml` | 无直接 PR 号提交 |
| `test/ci/eval/minimax-m3-nvfp4-evalscope-aime25.yaml` | [#785](https://github.com/lightseekorg/tokenspeed/pull/785) |
| `test/runtime/models/test_minimax_m3.py` | [#733](https://github.com/lightseekorg/tokenspeed/pull/733), [#781](https://github.com/lightseekorg/tokenspeed/pull/781) |
| `test/runtime/test_minimax_m3_fused_qkv_indexer.py` | [#781](https://github.com/lightseekorg/tokenspeed/pull/781) |
| `tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/triton/minimax_topk.py` | 无直接 PR 号提交 |
| `tokenspeed-kernel/test/ops/moe/test_minimax_biased_grouped_topk.py` | 无直接 PR 号提交 |

## PR 覆盖总览

- git 追溯 PR 数: 4
- 原文档显式引用补充 PR 数: 0
- 当前文档总 PR 数: 4
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-07-21 | [#733](https://github.com/lightseekorg/tokenspeed/pull/733) | merged | feat: add basic MiniMax M3 support | `python/tokenspeed/runtime/models/minimax_m3.py`, `python/tokenspeed/runtime/configs/minimax_m3_config.py`, `test/runtime/models/test_minimax_m3.py` |
| 2026-07-23 | [#779](https://github.com/lightseekorg/tokenspeed/pull/779) | merged | perf(m3): optimize MiniMax M3 MoE and MSA | `python/tokenspeed/runtime/models/minimax_m3.py` |
| 2026-07-23 | [#781](https://github.com/lightseekorg/tokenspeed/pull/781) | merged | perf(m3): optimize MiniMax M3 attention module | `python/tokenspeed/runtime/models/minimax_m3.py`, `test/runtime/models/test_minimax_m3.py`, `test/runtime/test_minimax_m3_fused_qkv_indexer.py` |
| 2026-07-24 | [#785](https://github.com/lightseekorg/tokenspeed/pull/785) | merged | test(m3): enable MiniMax-M3 CI (replacing M2.7) and add agentic benchmark | `test/agentic_benchmark/minimax_m3/tokenspeed/configs/attn_tp4_moe_ep4.sh`, `test/agentic_benchmark/minimax_m3/tokenspeed/configs/attn_tp4_moe_tp4.sh`, `test/agentic_benchmark/minimax_m3/tokenspeed/agentic_bench.sh` |

## 逐 PR diff 审计卡

### PR #733 - feat: add basic MiniMax M3 support

- 链接: https://github.com/lightseekorg/tokenspeed/pull/733
- 状态/时间: merged / 2026-07-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/configs/minimax_m3_config.py`, `python/tokenspeed/runtime/models/minimax_m3.py`, `test/runtime/models/test_minimax_m3.py`；关联提交 `c73daee9f957`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 33 个文件，+5150/-27，可读 patch 5629 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/minimax_m3.py` added +1403/-0 (1403 lines); hunks: -0,0 +1,1403; symbols: MiniMaxM3MLP, __init__, forward, MiniMaxM3SparseMoeBlock，涉及 `MiniMaxM3MLP, __init__, forward`；`python/tokenspeed/runtime/configs/minimax_m3_config.py` added +229/-0 (229 lines); hunks: -0,0 +1,229; symbols: MiniMaxM3VisionConfig, __init__, MiniMaxM3Config，涉及 `MiniMaxM3VisionConfig, __init__, MiniMaxM3Config`；`test/runtime/models/test_minimax_m3.py` added +226/-0 (226 lines); hunks: -0,0 +1,226; symbols: _tiny_config, _mxfp8_config, _tp4_mapping, _build_model，涉及 `_tiny_config, _mxfp8_config, _tp4_mapping`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/minimax_m3.py` added +1403/-0 (1403 lines); hunks: -0,0 +1,1403; symbols: MiniMaxM3MLP, __init__, forward, MiniMaxM3SparseMoeBlock
  - `python/tokenspeed/runtime/configs/minimax_m3_config.py` added +229/-0 (229 lines); hunks: -0,0 +1,229; symbols: MiniMaxM3VisionConfig, __init__, MiniMaxM3Config
  - `test/runtime/models/test_minimax_m3.py` added +226/-0 (226 lines); hunks: -0,0 +1,226; symbols: _tiny_config, _mxfp8_config, _tp4_mapping, _build_model
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/minimax_m3.py
@@ -0,0 +1,1403 @@
+# Copyright (c) 2026 LightSeek Foundation
+#
+# Permission is hereby granted, free of charge, to any person obtaining a copy
+# of this software and associated documentation files (the "Software"), to deal
+# in the Software without restriction, including without limitation the rights
+# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
diff -- python/tokenspeed/runtime/configs/minimax_m3_config.py
@@ -0,0 +1,229 @@
+# Copyright (c) 2026 LightSeek Foundation
+#
+# Permission is hereby granted, free of charge, to any person obtaining a copy
+# of this software and associated documentation files (the "Software"), to deal
+# in the Software without restriction, including without limitation the rights
+# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
diff -- test/runtime/models/test_minimax_m3.py
@@ -0,0 +1,226 @@
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/minimax_m3.py` added +1403/-0; `python/tokenspeed/runtime/configs/minimax_m3_config.py` added +229/-0
  - tests: `test/runtime/models/test_minimax_m3.py` added +226/-0
- 验证与风险: diff 自带测试面 `test/ci_system/install_deps.sh`, `test/runtime/models/test_minimax_m3.py`, `tokenspeed-kernel/test/ops/test_activation.py`, `tokenspeed-kernel/test/ops/test_attention_msa.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #779 - perf(m3): optimize MiniMax M3 MoE and MSA

- 链接: https://github.com/lightseekorg/tokenspeed/pull/779
- 状态/时间: merged / 2026-07-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/minimax_m3.py`；关联提交 `eba72d130727`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 19 个文件，+207/-701，可读 patch 1263 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/minimax_m3.py` modified +88/-20 (108 lines); hunks: -27,7 +27,10; -39,6 +42,7; symbols: forward, MiniMaxM3SparseMoeBlock, __init__，涉及 `forward, MiniMaxM3SparseMoeBlock, __init__`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/minimax_m3.py` modified +88/-20 (108 lines); hunks: -27,7 +27,10; -39,6 +42,7; symbols: forward, MiniMaxM3SparseMoeBlock, __init__
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/minimax_m3.py
@@ -27,7 +27,10 @@
+from tokenspeed_kernel.ops.gemm.cuda import dsv3_router_gemm
+from tokenspeed_kernel.ops.moe.cuda import moe_finalize_fuse_shared
+from tokenspeed_kernel.platform import current_platform
@@ -39,6 +42,7 @@
+from tokenspeed.runtime.execution.cuda_graph_wrapper import get_is_capture_mode
@@ -76,8 +80,10 @@
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/minimax_m3.py` modified +88/-20
- 验证与风险: diff 自带测试面 `tokenspeed-kernel/test/thirdparty/test_cuda.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #781 - perf(m3): optimize MiniMax M3 attention module

- 链接: https://github.com/lightseekorg/tokenspeed/pull/781
- 状态/时间: merged / 2026-07-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/minimax_m3.py`, `test/runtime/models/test_minimax_m3.py`, `test/runtime/test_minimax_m3_fused_qkv_indexer.py`；关联提交 `6da4cf59eee1`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 17 个文件，+1720/-81，可读 patch 2344 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/minimax_m3.py` modified +324/-56 (380 lines); hunks: -31,6 +31,9; -41,6 +44,7; symbols: forward, MinimaxM3QKVParallelLinearWithIndexer, __init__, _get_shard_offset_mapping，涉及 `forward, MinimaxM3QKVParallelLinearWithIndexer, __init__`；`test/runtime/models/test_minimax_m3.py` modified +6/-3 (9 lines); hunks: -295,14 +295,17 @@ def test_minimax_m3_mixed_precision_quant_dispatch(; symbols: test_minimax_m3_mixed_precision_quant_dispatch，涉及 `test_minimax_m3_mixed_precision_quant_dispatch`；`test/runtime/test_minimax_m3_fused_qkv_indexer.py` added +218/-0 (218 lines); hunks: -0,0 +1,218; symbols: _build, _ref_weights, test_output_sizes_layout, test_forward_matches_independent_projections_tp1，涉及 `_build, _ref_weights, test_output_sizes_layout`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/minimax_m3.py` modified +324/-56 (380 lines); hunks: -31,6 +31,9; -41,6 +44,7; symbols: forward, MinimaxM3QKVParallelLinearWithIndexer, __init__, _get_shard_offset_mapping
  - `test/runtime/models/test_minimax_m3.py` modified +6/-3 (9 lines); hunks: -295,14 +295,17 @@ def test_minimax_m3_mixed_precision_quant_dispatch(; symbols: test_minimax_m3_mixed_precision_quant_dispatch
  - `test/runtime/test_minimax_m3_fused_qkv_indexer.py` added +218/-0 (218 lines); hunks: -0,0 +1,218; symbols: _build, _ref_weights, test_output_sizes_layout, test_forward_matches_independent_projections_tp1
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/minimax_m3.py
@@ -31,6 +31,9 @@
+from tokenspeed_kernel.thirdparty.cuda.minimax_m3_fused import (
+    fused_qknorm_rope_kv_insert,
+)
@@ -41,6 +44,7 @@
+from tokenspeed.runtime.distributed.utils import divide
@@ -62,6 +66,10 @@
diff -- test/runtime/models/test_minimax_m3.py
@@ -295,14 +295,17 @@ def test_minimax_m3_mixed_precision_quant_dispatch(
+    from tokenspeed.runtime.models.minimax_m3 import (
+        MinimaxM3QKVParallelLinearWithIndexer,
+    )
+    # Sparse layers fuse q/k/v/index_q/index_k into one projection; its MXFP8
+    # dispatch confirms the index members resolved to MXFP8 via the qkv unfuse.
+    assert isinstance(attn.qkv_proj, MinimaxM3QKVParallelLinearWithIndexer)
diff -- test/runtime/test_minimax_m3_fused_qkv_indexer.py
@@ -0,0 +1,218 @@
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/minimax_m3.py` modified +324/-56
  - tests: `test/runtime/models/test_minimax_m3.py` modified +6/-3; `test/runtime/test_minimax_m3_fused_qkv_indexer.py` added +218/-0
- 验证与风险: diff 自带测试面 `test/runtime/models/test_minimax_m3.py`, `test/runtime/test_minimax_m3_fused_qkv_indexer.py`, `tokenspeed-kernel/test/ops/test_minimax_m3_fused_qknorm_rope_kv_insert.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #785 - test(m3): enable MiniMax-M3 CI (replacing M2.7) and add agentic benchmark

- 链接: https://github.com/lightseekorg/tokenspeed/pull/785
- 状态/时间: merged / 2026-07-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/agentic_benchmark/minimax_m3/tokenspeed/agentic_bench.sh`, `test/agentic_benchmark/minimax_m3/tokenspeed/collect_outputs.py`, `test/agentic_benchmark/minimax_m3/tokenspeed/configs/attn_tp4_moe_ep4.sh`, `test/agentic_benchmark/minimax_m3/tokenspeed/configs/attn_tp4_moe_tp4.sh`, `test/ci/eval/minimax-m3-nvfp4-evalscope-aime25.yaml`；关联提交 `0064dfefdf07`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 7 个文件，+308/-25，可读 patch 384 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/agentic_benchmark/minimax_m3/tokenspeed/configs/attn_tp4_moe_ep4.sh` added +30/-0 (30 lines); hunks: -0,0 +1,30；`test/agentic_benchmark/minimax_m3/tokenspeed/configs/attn_tp4_moe_tp4.sh` added +30/-0 (30 lines); hunks: -0,0 +1,30；`test/agentic_benchmark/minimax_m3/tokenspeed/agentic_bench.sh` added +140/-0 (140 lines); hunks: -0,0 +1,140；`test/agentic_benchmark/minimax_m3/tokenspeed/collect_outputs.py` added +82/-0 (82 lines); hunks: -0,0 +1,82; symbols: num_gpus_from_config, collect, main，涉及 `num_gpus_from_config, collect, main`。
- 代码 diff 细节:
  - `test/agentic_benchmark/minimax_m3/tokenspeed/configs/attn_tp4_moe_ep4.sh` added +30/-0 (30 lines); hunks: -0,0 +1,30
  - `test/agentic_benchmark/minimax_m3/tokenspeed/configs/attn_tp4_moe_tp4.sh` added +30/-0 (30 lines); hunks: -0,0 +1,30
  - `test/agentic_benchmark/minimax_m3/tokenspeed/agentic_bench.sh` added +140/-0 (140 lines); hunks: -0,0 +1,140
  - `test/agentic_benchmark/minimax_m3/tokenspeed/collect_outputs.py` added +82/-0 (82 lines); hunks: -0,0 +1,82; symbols: num_gpus_from_config, collect, main
  - `test/ci/eval/minimax-m3-nvfp4-evalscope-aime25.yaml` renamed +26/-14 (40 lines); hunks: -1,30 +1,39; -36,15 +45,18 @@ server:
- 关键代码摘录:

```diff
diff -- test/agentic_benchmark/minimax_m3/tokenspeed/configs/attn_tp4_moe_ep4.sh
@@ -0,0 +1,30 @@
+#!/usr/bin/bash
+set -euo pipefail
+exec ts serve \
+    --model nvidia/MiniMax-M3-NVFP4 \
+    --attn-tp-size 4 \
+    --ep-size 4 \
diff -- test/agentic_benchmark/minimax_m3/tokenspeed/configs/attn_tp4_moe_tp4.sh
@@ -0,0 +1,30 @@
+#!/usr/bin/bash
+set -euo pipefail
+exec ts serve \
+    --model nvidia/MiniMax-M3-NVFP4 \
+    --attn-tp-size 4 \
+    --moe-tp-size 4 \
diff -- test/agentic_benchmark/minimax_m3/tokenspeed/agentic_bench.sh
@@ -0,0 +1,140 @@
```

- 提取文件（未人工审阅）:
  - tests: `test/agentic_benchmark/minimax_m3/tokenspeed/configs/attn_tp4_moe_ep4.sh` added +30/-0; `test/agentic_benchmark/minimax_m3/tokenspeed/configs/attn_tp4_moe_tp4.sh` added +30/-0; `test/agentic_benchmark/minimax_m3/tokenspeed/agentic_bench.sh` added +140/-0; `test/agentic_benchmark/minimax_m3/tokenspeed/collect_outputs.py` added +82/-0; `test/ci/eval/minimax-m3-nvfp4-evalscope-aime25.yaml` renamed +26/-14
- 验证与风险: diff 自带测试面 `test/agentic_benchmark/minimax_m3/tokenspeed/agentic_bench.sh`, `test/agentic_benchmark/minimax_m3/tokenspeed/collect_outputs.py`, `test/agentic_benchmark/minimax_m3/tokenspeed/configs/attn_tp4_moe_ep4.sh`, `test/agentic_benchmark/minimax_m3/tokenspeed/configs/attn_tp4_moe_tp4.sh`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
