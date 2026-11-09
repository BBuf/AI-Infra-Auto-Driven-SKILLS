# TokenSpeed Qwen3 Core 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `python/tokenspeed/runtime/configs/qwen3_moe_config.py` | [#181](https://github.com/lightseekorg/tokenspeed/pull/181) |
| `python/tokenspeed/runtime/models/qwen3.py` | [#1281](https://github.com/lightseekorg/tokenspeed/pull/1281) |
| `python/tokenspeed/runtime/models/qwen3_moe.py` | [#181](https://github.com/lightseekorg/tokenspeed/pull/181), [#1713](https://github.com/lightseekorg/tokenspeed/pull/1713) |
| `test/runtime/models/test_qwen3_moe_models.py` | [#181](https://github.com/lightseekorg/tokenspeed/pull/181) |

## PR 覆盖总览

- git 追溯 PR 数: 3
- 原文档显式引用补充 PR 数: 0
- 当前文档总 PR 数: 3
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-05-19 | [#181](https://github.com/lightseekorg/tokenspeed/pull/181) | merged | feat(qwen3): add Qwen3 MoE causal LM support | `python/tokenspeed/runtime/models/qwen3_moe.py`, `test/runtime/models/test_qwen3_moe_models.py`, `python/tokenspeed/runtime/configs/qwen3_moe_config.py` |
| 2026-08-29 | [#1281](https://github.com/lightseekorg/tokenspeed/pull/1281) | merged | feat(npu): support qwen3-0.6B on ascend | `python/tokenspeed/runtime/models/qwen3.py` |
| 2026-09-22 | [#1713](https://github.com/lightseekorg/tokenspeed/pull/1713) | merged | fix(qwen3.5 moe): shard Qwen shared experts over the MoE reduction group | `python/tokenspeed/runtime/models/qwen3_moe.py` |

## 逐 PR diff 审计卡

### PR #181 - feat(qwen3): add Qwen3 MoE causal LM support

- 链接: https://github.com/lightseekorg/tokenspeed/pull/181
- 状态/时间: merged / 2026-05-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/configs/qwen3_moe_config.py`, `python/tokenspeed/runtime/models/qwen3_moe.py`, `test/runtime/models/test_qwen3_moe_models.py`；关联提交 `0db115ff6004`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 6 个文件，+610/-0，可读 patch 645 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/qwen3_moe.py` added +309/-0 (309 lines); hunks: -0,0 +1,309; symbols: Qwen3MoeDecoderLayer, __init__, forward, forward_mlp，涉及 `Qwen3MoeDecoderLayer, __init__, forward`；`test/runtime/models/test_qwen3_moe_models.py` added +223/-0 (223 lines); hunks: -0,0 +1,223; symbols: _tiny_qwen3_moe_config, _single_rank_mapping, _ep_rank_mapping, TestQwen3MoeConfig，涉及 `_tiny_qwen3_moe_config, _single_rank_mapping, _ep_rank_mapping`；`python/tokenspeed/runtime/configs/qwen3_moe_config.py` added +56/-0 (56 lines); hunks: -0,0 +1,56; symbols: Qwen3MoeConfig, __init__，涉及 `Qwen3MoeConfig, __init__`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/qwen3_moe.py` added +309/-0 (309 lines); hunks: -0,0 +1,309; symbols: Qwen3MoeDecoderLayer, __init__, forward, forward_mlp
  - `test/runtime/models/test_qwen3_moe_models.py` added +223/-0 (223 lines); hunks: -0,0 +1,223; symbols: _tiny_qwen3_moe_config, _single_rank_mapping, _ep_rank_mapping, TestQwen3MoeConfig
  - `python/tokenspeed/runtime/configs/qwen3_moe_config.py` added +56/-0 (56 lines); hunks: -0,0 +1,56; symbols: Qwen3MoeConfig, __init__
- 关键代码摘录:

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

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/qwen3_moe.py` added +309/-0; `python/tokenspeed/runtime/configs/qwen3_moe_config.py` added +56/-0
  - tests: `test/runtime/models/test_qwen3_moe_models.py` added +223/-0
- 验证与风险: diff 自带测试面 `test/runtime/models/test_qwen3_moe_models.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1281 - feat(npu): support qwen3-0.6B on ascend

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1281
- 状态/时间: merged / 2026-08-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/qwen3.py`；关联提交 `c006baf2dc07`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 55 个文件，+2339/-138，可读 patch 3091 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/qwen3.py` modified +2/-1 (3 lines); hunks: -26,7 +26,7; -180,6 +180,7 @@ def __init__(; symbols: __init__, _apply_qk_norm，涉及 `__init__, _apply_qk_norm`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/qwen3.py` modified +2/-1 (3 lines); hunks: -26,7 +26,7; -180,6 +180,7 @@ def __init__(; symbols: __init__, _apply_qk_norm
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/qwen3.py
@@ -26,7 +26,7 @@
-from tokenspeed_kernel.ops.layernorm.triton import qk_rmsnorm
+from tokenspeed_kernel.ops.layernorm import qk_rmsnorm
@@ -180,6 +180,7 @@ def __init__(
+            group_id=config.layer_types[layer_id],
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/qwen3.py` modified +2/-1
- 验证与风险: diff 自带测试面 `test/ci_system/install_triton_ascend.sh`, `test/ci_system/verify_triton_ascend.py`, `test/runtime/distributed/test_hccl_backend.py`, `test/runtime/layers/test_npu_attention_backend.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1713 - fix(qwen3.5 moe): shard Qwen shared experts over the MoE reduction group

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1713
- 状态/时间: merged / 2026-09-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/qwen3_moe.py`；关联提交 `0722ff403d8b`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 5 个文件，+88/-1，可读 patch 169 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/qwen3_moe.py` modified +1/-0 (1 lines); hunks: -77,6 +77,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/qwen3_moe.py` modified +1/-0 (1 lines); hunks: -77,6 +77,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/qwen3_moe.py
@@ -77,6 +77,7 @@ def __init__(
+                parallelism="dense",
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/qwen3_moe.py` modified +1/-0
- 验证与风险: diff 自带测试面 `test/runtime/models/test_qwen3_5_shared_expert_dp.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
