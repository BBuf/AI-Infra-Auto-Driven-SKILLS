# TokenSpeed MiniMax M2/M3 Series Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `python/tokenspeed/runtime/configs/minimax_m3_config.py` | [#733](https://github.com/lightseekorg/tokenspeed/pull/733) |
| `python/tokenspeed/runtime/models/minimax_m3.py` | [#733](https://github.com/lightseekorg/tokenspeed/pull/733), [#779](https://github.com/lightseekorg/tokenspeed/pull/779), [#781](https://github.com/lightseekorg/tokenspeed/pull/781) |
| `test/agentic_benchmark/minimax_m3/tokenspeed/agentic_bench.sh` | [#785](https://github.com/lightseekorg/tokenspeed/pull/785) |
| `test/agentic_benchmark/minimax_m3/tokenspeed/collect_outputs.py` | [#785](https://github.com/lightseekorg/tokenspeed/pull/785) |
| `test/agentic_benchmark/minimax_m3/tokenspeed/configs/attn_tp4_moe_ep4.sh` | [#785](https://github.com/lightseekorg/tokenspeed/pull/785) |
| `test/agentic_benchmark/minimax_m3/tokenspeed/configs/attn_tp4_moe_tp4.sh` | [#785](https://github.com/lightseekorg/tokenspeed/pull/785) |
| `test/ci/eval/minimax-m3-nvfp4-dspark-evalscope-aime26.yaml` | no direct PR-number commit |
| `test/ci/eval/minimax-m3-nvfp4-evalscope-aime25.yaml` | [#785](https://github.com/lightseekorg/tokenspeed/pull/785) |
| `test/runtime/models/test_minimax_m3.py` | [#733](https://github.com/lightseekorg/tokenspeed/pull/733), [#781](https://github.com/lightseekorg/tokenspeed/pull/781) |
| `test/runtime/test_minimax_m3_fused_qkv_indexer.py` | [#781](https://github.com/lightseekorg/tokenspeed/pull/781) |
| `tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/triton/minimax_topk.py` | no direct PR-number commit |
| `tokenspeed-kernel/test/ops/moe/test_minimax_biased_grouped_topk.py` | no direct PR-number commit |

## PR Coverage Summary

- Git-traced PRs: 4
- Extra PRs preserved from existing docs: 0
- Total PRs in this document: 4
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-07-21 | [#733](https://github.com/lightseekorg/tokenspeed/pull/733) | merged | feat: add basic MiniMax M3 support | `python/tokenspeed/runtime/models/minimax_m3.py`, `python/tokenspeed/runtime/configs/minimax_m3_config.py`, `test/runtime/models/test_minimax_m3.py` |
| 2026-07-23 | [#779](https://github.com/lightseekorg/tokenspeed/pull/779) | merged | perf(m3): optimize MiniMax M3 MoE and MSA | `python/tokenspeed/runtime/models/minimax_m3.py` |
| 2026-07-23 | [#781](https://github.com/lightseekorg/tokenspeed/pull/781) | merged | perf(m3): optimize MiniMax M3 attention module | `python/tokenspeed/runtime/models/minimax_m3.py`, `test/runtime/models/test_minimax_m3.py`, `test/runtime/test_minimax_m3_fused_qkv_indexer.py` |
| 2026-07-24 | [#785](https://github.com/lightseekorg/tokenspeed/pull/785) | merged | test(m3): enable MiniMax-M3 CI (replacing M2.7) and add agentic benchmark | `test/agentic_benchmark/minimax_m3/tokenspeed/configs/attn_tp4_moe_ep4.sh`, `test/agentic_benchmark/minimax_m3/tokenspeed/configs/attn_tp4_moe_tp4.sh`, `test/agentic_benchmark/minimax_m3/tokenspeed/agentic_bench.sh` |

## Per-PR Diff Audit Cards

### PR #733 - feat: add basic MiniMax M3 support

- Link: https://github.com/lightseekorg/tokenspeed/pull/733
- Status/date: merged / 2026-07-21
- Trace source: `git log --name-only -- <model-files>` found it through `python/tokenspeed/runtime/configs/minimax_m3_config.py`, `python/tokenspeed/runtime/models/minimax_m3.py`, `test/runtime/models/test_minimax_m3.py`; associated commits `c73daee9f957`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 33 files, +5150/-27, 5629 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/tokenspeed/runtime/models/minimax_m3.py` added +1403/-0 (1403 lines); hunks: -0,0 +1,1403; symbols: MiniMaxM3MLP, __init__, forward, MiniMaxM3SparseMoeBlock, touching `MiniMaxM3MLP, __init__, forward`; `python/tokenspeed/runtime/configs/minimax_m3_config.py` added +229/-0 (229 lines); hunks: -0,0 +1,229; symbols: MiniMaxM3VisionConfig, __init__, MiniMaxM3Config, touching `MiniMaxM3VisionConfig, __init__, MiniMaxM3Config`; `test/runtime/models/test_minimax_m3.py` added +226/-0 (226 lines); hunks: -0,0 +1,226; symbols: _tiny_config, _mxfp8_config, _tp4_mapping, _build_model, touching `_tiny_config, _mxfp8_config, _tp4_mapping`.
- Code diff details:
  - `python/tokenspeed/runtime/models/minimax_m3.py` added +1403/-0 (1403 lines); hunks: -0,0 +1,1403; symbols: MiniMaxM3MLP, __init__, forward, MiniMaxM3SparseMoeBlock
  - `python/tokenspeed/runtime/configs/minimax_m3_config.py` added +229/-0 (229 lines); hunks: -0,0 +1,229; symbols: MiniMaxM3VisionConfig, __init__, MiniMaxM3Config
  - `test/runtime/models/test_minimax_m3.py` added +226/-0 (226 lines); hunks: -0,0 +1,226; symbols: _tiny_config, _mxfp8_config, _tp4_mapping, _build_model
- Key code excerpts:

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

- Extracted files (not manually reviewed):
  - runtime: `python/tokenspeed/runtime/models/minimax_m3.py` added +1403/-0; `python/tokenspeed/runtime/configs/minimax_m3_config.py` added +229/-0
  - tests: `test/runtime/models/test_minimax_m3.py` added +226/-0
- Risk and verification: The diff ships test coverage in `test/ci_system/install_deps.sh`, `test/runtime/models/test_minimax_m3.py`, `tokenspeed-kernel/test/ops/test_activation.py`, `tokenspeed-kernel/test/ops/test_attention_msa.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #779 - perf(m3): optimize MiniMax M3 MoE and MSA

- Link: https://github.com/lightseekorg/tokenspeed/pull/779
- Status/date: merged / 2026-07-23
- Trace source: `git log --name-only -- <model-files>` found it through `python/tokenspeed/runtime/models/minimax_m3.py`; associated commits `eba72d130727`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 19 files, +207/-701, 1263 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/tokenspeed/runtime/models/minimax_m3.py` modified +88/-20 (108 lines); hunks: -27,7 +27,10; -39,6 +42,7; symbols: forward, MiniMaxM3SparseMoeBlock, __init__, touching `forward, MiniMaxM3SparseMoeBlock, __init__`.
- Code diff details:
  - `python/tokenspeed/runtime/models/minimax_m3.py` modified +88/-20 (108 lines); hunks: -27,7 +27,10; -39,6 +42,7; symbols: forward, MiniMaxM3SparseMoeBlock, __init__
- Key code excerpts:

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

- Extracted files (not manually reviewed):
  - runtime: `python/tokenspeed/runtime/models/minimax_m3.py` modified +88/-20
- Risk and verification: The diff ships test coverage in `tokenspeed-kernel/test/thirdparty/test_cuda.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #781 - perf(m3): optimize MiniMax M3 attention module

- Link: https://github.com/lightseekorg/tokenspeed/pull/781
- Status/date: merged / 2026-07-23
- Trace source: `git log --name-only -- <model-files>` found it through `python/tokenspeed/runtime/models/minimax_m3.py`, `test/runtime/models/test_minimax_m3.py`, `test/runtime/test_minimax_m3_fused_qkv_indexer.py`; associated commits `6da4cf59eee1`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 17 files, +1720/-81, 2344 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/tokenspeed/runtime/models/minimax_m3.py` modified +324/-56 (380 lines); hunks: -31,6 +31,9; -41,6 +44,7; symbols: forward, MinimaxM3QKVParallelLinearWithIndexer, __init__, _get_shard_offset_mapping, touching `forward, MinimaxM3QKVParallelLinearWithIndexer, __init__`; `test/runtime/models/test_minimax_m3.py` modified +6/-3 (9 lines); hunks: -295,14 +295,17 @@ def test_minimax_m3_mixed_precision_quant_dispatch(; symbols: test_minimax_m3_mixed_precision_quant_dispatch, touching `test_minimax_m3_mixed_precision_quant_dispatch`; `test/runtime/test_minimax_m3_fused_qkv_indexer.py` added +218/-0 (218 lines); hunks: -0,0 +1,218; symbols: _build, _ref_weights, test_output_sizes_layout, test_forward_matches_independent_projections_tp1, touching `_build, _ref_weights, test_output_sizes_layout`.
- Code diff details:
  - `python/tokenspeed/runtime/models/minimax_m3.py` modified +324/-56 (380 lines); hunks: -31,6 +31,9; -41,6 +44,7; symbols: forward, MinimaxM3QKVParallelLinearWithIndexer, __init__, _get_shard_offset_mapping
  - `test/runtime/models/test_minimax_m3.py` modified +6/-3 (9 lines); hunks: -295,14 +295,17 @@ def test_minimax_m3_mixed_precision_quant_dispatch(; symbols: test_minimax_m3_mixed_precision_quant_dispatch
  - `test/runtime/test_minimax_m3_fused_qkv_indexer.py` added +218/-0 (218 lines); hunks: -0,0 +1,218; symbols: _build, _ref_weights, test_output_sizes_layout, test_forward_matches_independent_projections_tp1
- Key code excerpts:

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

- Extracted files (not manually reviewed):
  - runtime: `python/tokenspeed/runtime/models/minimax_m3.py` modified +324/-56
  - tests: `test/runtime/models/test_minimax_m3.py` modified +6/-3; `test/runtime/test_minimax_m3_fused_qkv_indexer.py` added +218/-0
- Risk and verification: The diff ships test coverage in `test/runtime/models/test_minimax_m3.py`, `test/runtime/test_minimax_m3_fused_qkv_indexer.py`, `tokenspeed-kernel/test/ops/test_minimax_m3_fused_qknorm_rope_kv_insert.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #785 - test(m3): enable MiniMax-M3 CI (replacing M2.7) and add agentic benchmark

- Link: https://github.com/lightseekorg/tokenspeed/pull/785
- Status/date: merged / 2026-07-24
- Trace source: `git log --name-only -- <model-files>` found it through `test/agentic_benchmark/minimax_m3/tokenspeed/agentic_bench.sh`, `test/agentic_benchmark/minimax_m3/tokenspeed/collect_outputs.py`, `test/agentic_benchmark/minimax_m3/tokenspeed/configs/attn_tp4_moe_ep4.sh`, `test/agentic_benchmark/minimax_m3/tokenspeed/configs/attn_tp4_moe_tp4.sh`, `test/ci/eval/minimax-m3-nvfp4-evalscope-aime25.yaml`; associated commits `0064dfefdf07`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 7 files, +308/-25, 384 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `test/agentic_benchmark/minimax_m3/tokenspeed/configs/attn_tp4_moe_ep4.sh` added +30/-0 (30 lines); hunks: -0,0 +1,30; `test/agentic_benchmark/minimax_m3/tokenspeed/configs/attn_tp4_moe_tp4.sh` added +30/-0 (30 lines); hunks: -0,0 +1,30; `test/agentic_benchmark/minimax_m3/tokenspeed/agentic_bench.sh` added +140/-0 (140 lines); hunks: -0,0 +1,140; `test/agentic_benchmark/minimax_m3/tokenspeed/collect_outputs.py` added +82/-0 (82 lines); hunks: -0,0 +1,82; symbols: num_gpus_from_config, collect, main, touching `num_gpus_from_config, collect, main`.
- Code diff details:
  - `test/agentic_benchmark/minimax_m3/tokenspeed/configs/attn_tp4_moe_ep4.sh` added +30/-0 (30 lines); hunks: -0,0 +1,30
  - `test/agentic_benchmark/minimax_m3/tokenspeed/configs/attn_tp4_moe_tp4.sh` added +30/-0 (30 lines); hunks: -0,0 +1,30
  - `test/agentic_benchmark/minimax_m3/tokenspeed/agentic_bench.sh` added +140/-0 (140 lines); hunks: -0,0 +1,140
  - `test/agentic_benchmark/minimax_m3/tokenspeed/collect_outputs.py` added +82/-0 (82 lines); hunks: -0,0 +1,82; symbols: num_gpus_from_config, collect, main
  - `test/ci/eval/minimax-m3-nvfp4-evalscope-aime25.yaml` renamed +26/-14 (40 lines); hunks: -1,30 +1,39; -36,15 +45,18 @@ server:
- Key code excerpts:

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

- Extracted files (not manually reviewed):
  - tests: `test/agentic_benchmark/minimax_m3/tokenspeed/configs/attn_tp4_moe_ep4.sh` added +30/-0; `test/agentic_benchmark/minimax_m3/tokenspeed/configs/attn_tp4_moe_tp4.sh` added +30/-0; `test/agentic_benchmark/minimax_m3/tokenspeed/agentic_bench.sh` added +140/-0; `test/agentic_benchmark/minimax_m3/tokenspeed/collect_outputs.py` added +82/-0; `test/ci/eval/minimax-m3-nvfp4-evalscope-aime25.yaml` renamed +26/-14
- Risk and verification: The diff ships test coverage in `test/agentic_benchmark/minimax_m3/tokenspeed/agentic_bench.sh`, `test/agentic_benchmark/minimax_m3/tokenspeed/collect_outputs.py`, `test/agentic_benchmark/minimax_m3/tokenspeed/configs/attn_tp4_moe_ep4.sh`, `test/agentic_benchmark/minimax_m3/tokenspeed/configs/attn_tp4_moe_tp4.sh`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
