# TokenSpeed Qwen3 Core Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `python/tokenspeed/runtime/configs/qwen3_moe_config.py` | [#181](https://github.com/lightseekorg/tokenspeed/pull/181) |
| `python/tokenspeed/runtime/models/qwen3.py` | [#1281](https://github.com/lightseekorg/tokenspeed/pull/1281) |
| `python/tokenspeed/runtime/models/qwen3_moe.py` | [#181](https://github.com/lightseekorg/tokenspeed/pull/181), [#1713](https://github.com/lightseekorg/tokenspeed/pull/1713) |
| `test/runtime/models/test_qwen3_moe_models.py` | [#181](https://github.com/lightseekorg/tokenspeed/pull/181) |

## PR Coverage Summary

- Git-traced PRs: 3
- Extra PRs preserved from existing docs: 0
- Total PRs in this document: 3
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-05-19 | [#181](https://github.com/lightseekorg/tokenspeed/pull/181) | merged | feat(qwen3): add Qwen3 MoE causal LM support | `python/tokenspeed/runtime/models/qwen3_moe.py`, `test/runtime/models/test_qwen3_moe_models.py`, `python/tokenspeed/runtime/configs/qwen3_moe_config.py` |
| 2026-08-29 | [#1281](https://github.com/lightseekorg/tokenspeed/pull/1281) | merged | feat(npu): support qwen3-0.6B on ascend | `python/tokenspeed/runtime/models/qwen3.py` |
| 2026-09-22 | [#1713](https://github.com/lightseekorg/tokenspeed/pull/1713) | merged | fix(qwen3.5 moe): shard Qwen shared experts over the MoE reduction group | `python/tokenspeed/runtime/models/qwen3_moe.py` |

## Per-PR Diff Audit Cards

### PR #181 - feat(qwen3): add Qwen3 MoE causal LM support

- Link: https://github.com/lightseekorg/tokenspeed/pull/181
- Status/date: merged / 2026-05-19
- Trace source: `git log --name-only -- <model-files>` found it through `python/tokenspeed/runtime/configs/qwen3_moe_config.py`, `python/tokenspeed/runtime/models/qwen3_moe.py`, `test/runtime/models/test_qwen3_moe_models.py`; associated commits `0db115ff6004`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 6 files, +610/-0, 645 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/tokenspeed/runtime/models/qwen3_moe.py` added +309/-0 (309 lines); hunks: -0,0 +1,309; symbols: Qwen3MoeDecoderLayer, __init__, forward, forward_mlp, touching `Qwen3MoeDecoderLayer, __init__, forward`; `test/runtime/models/test_qwen3_moe_models.py` added +223/-0 (223 lines); hunks: -0,0 +1,223; symbols: _tiny_qwen3_moe_config, _single_rank_mapping, _ep_rank_mapping, TestQwen3MoeConfig, touching `_tiny_qwen3_moe_config, _single_rank_mapping, _ep_rank_mapping`; `python/tokenspeed/runtime/configs/qwen3_moe_config.py` added +56/-0 (56 lines); hunks: -0,0 +1,56; symbols: Qwen3MoeConfig, __init__, touching `Qwen3MoeConfig, __init__`.
- Code diff details:
  - `python/tokenspeed/runtime/models/qwen3_moe.py` added +309/-0 (309 lines); hunks: -0,0 +1,309; symbols: Qwen3MoeDecoderLayer, __init__, forward, forward_mlp
  - `test/runtime/models/test_qwen3_moe_models.py` added +223/-0 (223 lines); hunks: -0,0 +1,223; symbols: _tiny_qwen3_moe_config, _single_rank_mapping, _ep_rank_mapping, TestQwen3MoeConfig
  - `python/tokenspeed/runtime/configs/qwen3_moe_config.py` added +56/-0 (56 lines); hunks: -0,0 +1,56; symbols: Qwen3MoeConfig, __init__
- Key code excerpts:

```diff
diff -- python/tokenspeed/runtime/models/qwen3_moe.py
@@ -0,0 +1,309 @@
+# Copyright (c) 2026 LightSeek Foundation
+#
+# Permission is hereby granted, free of charge, to any person obtaining a copy
+# of this software and associated documentation files (the "Software"), to deal
+# in the Software without restriction, including without limitation the rights
+# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
diff -- test/runtime/models/test_qwen3_moe_models.py
@@ -0,0 +1,223 @@
+import json
+import unittest
+import torch
+from torch import nn
+from tokenspeed.runtime.configs.qwen3_moe_config import Qwen3MoeConfig
+from tokenspeed.runtime.distributed.mapping import Mapping
diff -- python/tokenspeed/runtime/configs/qwen3_moe_config.py
@@ -0,0 +1,56 @@
```

- Extracted files (not manually reviewed):
  - runtime: `python/tokenspeed/runtime/models/qwen3_moe.py` added +309/-0; `python/tokenspeed/runtime/configs/qwen3_moe_config.py` added +56/-0
  - tests: `test/runtime/models/test_qwen3_moe_models.py` added +223/-0
- Risk and verification: The diff ships test coverage in `test/runtime/models/test_qwen3_moe_models.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #1281 - feat(npu): support qwen3-0.6B on ascend

- Link: https://github.com/lightseekorg/tokenspeed/pull/1281
- Status/date: merged / 2026-08-29
- Trace source: `git log --name-only -- <model-files>` found it through `python/tokenspeed/runtime/models/qwen3.py`; associated commits `c006baf2dc07`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 55 files, +2339/-138, 3091 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/tokenspeed/runtime/models/qwen3.py` modified +2/-1 (3 lines); hunks: -26,7 +26,7; -180,6 +180,7 @@ def __init__(; symbols: __init__, _apply_qk_norm, touching `__init__, _apply_qk_norm`.
- Code diff details:
  - `python/tokenspeed/runtime/models/qwen3.py` modified +2/-1 (3 lines); hunks: -26,7 +26,7; -180,6 +180,7 @@ def __init__(; symbols: __init__, _apply_qk_norm
- Key code excerpts:

```diff
diff -- python/tokenspeed/runtime/models/qwen3.py
@@ -26,7 +26,7 @@
-from tokenspeed_kernel.ops.layernorm.triton import qk_rmsnorm
+from tokenspeed_kernel.ops.layernorm import qk_rmsnorm
@@ -180,6 +180,7 @@ def __init__(
+            group_id=config.layer_types[layer_id],
```

- Extracted files (not manually reviewed):
  - runtime: `python/tokenspeed/runtime/models/qwen3.py` modified +2/-1
- Risk and verification: The diff ships test coverage in `test/ci_system/install_triton_ascend.sh`, `test/ci_system/verify_triton_ascend.py`, `test/runtime/distributed/test_hccl_backend.py`, `test/runtime/layers/test_npu_attention_backend.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #1713 - fix(qwen3.5 moe): shard Qwen shared experts over the MoE reduction group

- Link: https://github.com/lightseekorg/tokenspeed/pull/1713
- Status/date: merged / 2026-09-22
- Trace source: `git log --name-only -- <model-files>` found it through `python/tokenspeed/runtime/models/qwen3_moe.py`; associated commits `0722ff403d8b`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 5 files, +88/-1, 169 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/tokenspeed/runtime/models/qwen3_moe.py` modified +1/-0 (1 lines); hunks: -77,6 +77,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/tokenspeed/runtime/models/qwen3_moe.py` modified +1/-0 (1 lines); hunks: -77,6 +77,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/tokenspeed/runtime/models/qwen3_moe.py
@@ -77,6 +77,7 @@ def __init__(
+                parallelism="dense",
```

- Extracted files (not manually reviewed):
  - runtime: `python/tokenspeed/runtime/models/qwen3_moe.py` modified +1/-0
- Risk and verification: The diff ships test coverage in `test/runtime/models/test_qwen3_5_shared_expert_dp.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
