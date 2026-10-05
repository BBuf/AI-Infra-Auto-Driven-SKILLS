# SGLang Hunyuan V4 (Hy4) 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `docs/cookbook/autoregressive/Tencent/Hy4-Preview.mdx` | [#36804](https://github.com/sgl-project/sglang/pull/36804), [#36808](https://github.com/sgl-project/sglang/pull/36808), [#36823](https://github.com/sgl-project/sglang/pull/36823) |
| `docs/src/snippets/configs/tencent/hy4-preview-benchmarks.jsx` | [#36804](https://github.com/sgl-project/sglang/pull/36804) |
| `docs/src/snippets/configs/tencent/hy4-preview.jsx` | [#36804](https://github.com/sgl-project/sglang/pull/36804), [#36808](https://github.com/sgl-project/sglang/pull/36808) |
| `python/sglang/kernels/ops/layernorm/hy4_ihc.py` | [#36805](https://github.com/sgl-project/sglang/pull/36805) |
| `python/sglang/srt/configs/hy_v4.py` | [#36805](https://github.com/sgl-project/sglang/pull/36805) |
| `python/sglang/srt/models/hunyuan_v4.py` | [#36805](https://github.com/sgl-project/sglang/pull/36805), [#42477](https://github.com/sgl-project/sglang/pull/42477) |
| `python/sglang/srt/models/hunyuan_v4_nextn.py` | [#36805](https://github.com/sgl-project/sglang/pull/36805), [#42477](https://github.com/sgl-project/sglang/pull/42477) |
| `test/registered/kernels/ops/attention/test_hy4_hpc_gated_mla.py` | [#36805](https://github.com/sgl-project/sglang/pull/36805) |
| `test/registered/kernels/ops/layernorm/test_hy4_decode_kernels.py` | [#36805](https://github.com/sgl-project/sglang/pull/36805) |
| `test/registered/kernels/ops/layernorm/test_hy4_hpc_ihc.py` | [#36805](https://github.com/sgl-project/sglang/pull/36805) |
| `test/registered/unit/models/test_hunyuan_v4.py` | [#36805](https://github.com/sgl-project/sglang/pull/36805) |
| `test/registered/unit/models/test_hunyuan_v4_nextn_weight_loading.py` | [#36805](https://github.com/sgl-project/sglang/pull/36805) |

## PR 覆盖总览

- git 追溯 PR 数: 5
- 原文档显式引用补充 PR 数: 0
- 当前文档总 PR 数: 5
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-08-28 | [#36804](https://github.com/sgl-project/sglang/pull/36804) | merged | [Cookbook] Add the Hy4-Preview model page (Tencent) | `docs/src/snippets/configs/tencent/hy4-preview.jsx`, `docs/src/snippets/configs/tencent/hy4-preview-benchmarks.jsx`, `docs/cookbook/autoregressive/Tencent/Hy4-Preview.mdx` |
| 2026-08-28 | [#36808](https://github.com/sgl-project/sglang/pull/36808) | merged | [Cookbook] Hy4-Preview follow-ups: runtime-accurate recipes + released-model info | `docs/src/snippets/configs/tencent/hy4-preview.jsx`, `docs/cookbook/autoregressive/Tencent/Hy4-Preview.mdx` |
| 2026-08-28 | [#36823](https://github.com/sgl-project/sglang/pull/36823) | merged | [Docs] Rename Tencent cookbook page titles to "Hy4 preview" / "Hy3 preview" | `docs/cookbook/autoregressive/Tencent/Hy4-Preview.mdx` |
| 2026-09-05 | [#36805](https://github.com/sgl-project/sglang/pull/36805) | merged | Support Hy4-preview | `python/sglang/srt/models/hunyuan_v4.py`, `python/sglang/srt/models/hunyuan_v4_nextn.py`, `python/sglang/srt/configs/hy_v4.py` |
| 2026-10-04 | [#42477](https://github.com/sgl-project/sglang/pull/42477) | merged | [Refactor] Build the Hunyuan V4 decoder from stage boundaries | `python/sglang/srt/models/hunyuan_v4.py`, `python/sglang/srt/models/hunyuan_v4_nextn.py` |

## 逐 PR diff 审计卡

### PR #36804 - [Cookbook] Add the Hy4-Preview model page (Tencent)

- 链接: https://github.com/sgl-project/sglang/pull/36804
- 状态/时间: merged / 2026-08-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/cookbook/autoregressive/Tencent/Hy4-Preview.mdx`, `docs/src/snippets/configs/tencent/hy4-preview-benchmarks.jsx`, `docs/src/snippets/configs/tencent/hy4-preview.jsx`；关联提交 `1948b61ad4d7`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 6 个文件，+888/-2，可读 patch 914 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `docs/src/snippets/configs/tencent/hy4-preview.jsx` added +574/-0 (574 lines); hunks: -0,0 +1,574；`docs/src/snippets/configs/tencent/hy4-preview-benchmarks.jsx` added +33/-0 (33 lines); hunks: -0,0 +1,33；`docs/cookbook/autoregressive/Tencent/Hy4-Preview.mdx` added +279/-0 (279 lines); hunks: -0,0 +1,279。
- 代码 diff 细节:
  - `docs/src/snippets/configs/tencent/hy4-preview.jsx` added +574/-0 (574 lines); hunks: -0,0 +1,574
  - `docs/src/snippets/configs/tencent/hy4-preview-benchmarks.jsx` added +33/-0 (33 lines); hunks: -0,0 +1,33
  - `docs/cookbook/autoregressive/Tencent/Hy4-Preview.mdx` added +279/-0 (279 lines); hunks: -0,0 +1,279
- 关键代码摘录:

```diff
diff -- docs/src/snippets/configs/tencent/hy4-preview.jsx
@@ -0,0 +1,574 @@
+// Hy4-Preview cookbook config. Consumed by _deployment.jsx + _playground.jsx;
+// see _deployment.jsx header for the field contract.
+//
+// Sizing (drives the TP/nodes choices below):
+// ~760B total / ~40B active MoE. BF16 weights ≈ 1.5TB → TP16 on H200/B200
+// (2x8 multi-node) or TP8 on B300 (single 8-GPU node) / GB300 (2x4 multi-node
diff -- docs/src/snippets/configs/tencent/hy4-preview-benchmarks.jsx
@@ -0,0 +1,33 @@
+// Hy4-Preview benchmark data — one entry per config cell `match` tuple.
+//
+// All entries are bare-match stubs (the card shows "pending"). When a cell's
+// numbers land, fill the entry with the measured data and set
+// `sglang_version` to the exact commit/tag they were measured on (a
+// reproducible anchor — never a moving ref like "main").
diff -- docs/cookbook/autoregressive/Tencent/Hy4-Preview.mdx
@@ -0,0 +1,279 @@
```

- 提取文件（未人工审阅）:
  - docs: `docs/src/snippets/configs/tencent/hy4-preview.jsx` added +574/-0; `docs/src/snippets/configs/tencent/hy4-preview-benchmarks.jsx` added +33/-0; `docs/cookbook/autoregressive/Tencent/Hy4-Preview.mdx` added +279/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs/cookbook/autoregressive/Tencent/Hy3.mdx`, `docs/cookbook/autoregressive/Tencent/Hy4-Preview.mdx`, `docs/cookbook/autoregressive/intro.mdx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #36808 - [Cookbook] Hy4-Preview follow-ups: runtime-accurate recipes + released-model info

- 链接: https://github.com/sgl-project/sglang/pull/36808
- 状态/时间: merged / 2026-08-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/cookbook/autoregressive/Tencent/Hy4-Preview.mdx`, `docs/src/snippets/configs/tencent/hy4-preview.jsx`；关联提交 `2960d696228e`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+42/-59，可读 patch 285 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `docs/src/snippets/configs/tencent/hy4-preview.jsx` modified +24/-43 (67 lines); hunks: -2,7 +2,7; -12,10 +12,11；`docs/cookbook/autoregressive/Tencent/Hy4-Preview.mdx` modified +18/-16 (34 lines); hunks: -1,6 +1,6; -10,7 +10,7 @@ tag: NEW。
- 代码 diff 细节:
  - `docs/src/snippets/configs/tencent/hy4-preview.jsx` modified +24/-43 (67 lines); hunks: -2,7 +2,7; -12,10 +12,11
  - `docs/cookbook/autoregressive/Tencent/Hy4-Preview.mdx` modified +18/-16 (34 lines); hunks: -1,6 +1,6; -10,7 +10,7 @@ tag: NEW
- 关键代码摘录:

```diff
diff -- docs/src/snippets/configs/tencent/hy4-preview.jsx
@@ -2,7 +2,7 @@
-// ~760B total / ~40B active MoE. BF16 weights ≈ 1.5TB → TP16 on H200/B200
+// 770B total / 49B active MoE. BF16 weights ≈ 1.5TB → TP16 on H200/B200
@@ -12,10 +12,11 @@
-// Every cell carries `verificationStatus: "in-progress"`. When a recipe's
-// end-to-end verification lands, REPLACE that line with `verified: true` —
-// `verificationStatus` takes precedence over `verified` in the engine, so
diff -- docs/cookbook/autoregressive/Tencent/Hy4-Preview.mdx
@@ -1,6 +1,6 @@
-description: "Deploy Tencent Hy4-Preview with SGLang — launch recipes for the 760B-parameter Mixture-of-Experts model with MLA, DeepSeek Sparse Attention (DSA), and MTP speculativ
+description: "Deploy Tencent Hy4-Preview with SGLang — launch recipes for the 770B-parameter Mixture-of-Experts model with MLA, DeepSeek Sparse Attention (DSA), and MTP speculativ
@@ -10,7 +10,7 @@ tag: NEW
-For all methods and hardware platforms, see the [official SGLang installation guide](/docs/get-started/install). The two paths below match the **Python / Docker** toggle in the co
+For all methods and hardware platforms, see the [official SGLang installation guide](/docs/get-started/install). The Docker path below matches the **Docker** framing in the comman
@@ -50,7 +50,7 @@ import { benchmarks } from "/src/snippets/configs/tencent/hy4-preview-benchmarks
```

- 提取文件（未人工审阅）:
  - docs: `docs/src/snippets/configs/tencent/hy4-preview.jsx` modified +24/-43; `docs/cookbook/autoregressive/Tencent/Hy4-Preview.mdx` modified +18/-16
- 验证与风险: 该 PR 主要落在文档/示例 `docs/cookbook/autoregressive/Tencent/Hy4-Preview.mdx`, `docs/src/snippets/configs/tencent/hy4-preview.jsx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #36823 - [Docs] Rename Tencent cookbook page titles to "Hy4 preview" / "Hy3 preview"

- 链接: https://github.com/sgl-project/sglang/pull/36823
- 状态/时间: merged / 2026-08-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/cookbook/autoregressive/Tencent/Hy4-Preview.mdx`；关联提交 `989e51ba9c4c`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+12/-12，可读 patch 87 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `docs/cookbook/autoregressive/Tencent/Hy4-Preview.mdx` modified +9/-9 (18 lines); hunks: -1,6 +1,6; -78,7 +78,7 @@ import { Playground } from "/src/snippets/_playground.jsx";。
- 代码 diff 细节:
  - `docs/cookbook/autoregressive/Tencent/Hy4-Preview.mdx` modified +9/-9 (18 lines); hunks: -1,6 +1,6; -78,7 +78,7 @@ import { Playground } from "/src/snippets/_playground.jsx";
- 关键代码摘录:

```diff
diff -- docs/cookbook/autoregressive/Tencent/Hy4-Preview.mdx
@@ -1,6 +1,6 @@
-title: Hy4-Preview
-description: "Deploy Tencent Hy4-Preview with SGLang — launch recipes for the 770B-parameter Mixture-of-Experts model with MLA, DeepSeek Sparse Attention (DSA), and MTP speculativ
+title: Hy4 preview
+description: "Deploy Tencent Hy4 preview with SGLang — launch recipes for the 770B-parameter Mixture-of-Experts model with MLA, DeepSeek Sparse Attention (DSA), and MTP speculativ
@@ -78,7 +78,7 @@ import { Playground } from "/src/snippets/_playground.jsx";
-**Hy4-Preview** is Tencent's next-generation flagship Mixture-of-Experts language model: 770B total parameters with 49B active per token, pairing a DeepSeek-style MLA + sparse-att
```

- 提取文件（未人工审阅）:
  - docs: `docs/cookbook/autoregressive/Tencent/Hy4-Preview.mdx` modified +9/-9
- 验证与风险: 该 PR 主要落在文档/示例 `docs/cookbook/autoregressive/Tencent/Hunyuan3-Preview.mdx`, `docs/cookbook/autoregressive/Tencent/Hy4-Preview.mdx`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #36805 - Support Hy4-preview

- 链接: https://github.com/sgl-project/sglang/pull/36805
- 状态/时间: merged / 2026-09-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/kernels/ops/layernorm/hy4_ihc.py`, `python/sglang/srt/configs/hy_v4.py`, `python/sglang/srt/models/hunyuan_v4.py`, `python/sglang/srt/models/hunyuan_v4_nextn.py`, `test/registered/kernels/ops/attention/test_hy4_hpc_gated_mla.py` 等 9 个文件；关联提交 `55bf3380e073`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 47 个文件，+3387/-103，可读 patch 4334 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/sglang/srt/models/hunyuan_v4.py` added +734/-0 (734 lines); hunks: -0,0 +1,734; symbols: _hpc_ihc_available, permute_hyv4_indexer_weight, HYV4HCPreLayer, __init__，涉及 `_hpc_ihc_available, permute_hyv4_indexer_weight, HYV4HCPreLayer`；`python/sglang/srt/models/hunyuan_v4_nextn.py` added +229/-0 (229 lines); hunks: -0,0 +1,229; symbols: _mtp_quant_config, HYV4MTPDecoderLayer, __init__, forward，涉及 `_mtp_quant_config, HYV4MTPDecoderLayer, __init__`；`python/sglang/srt/configs/hy_v4.py` added +152/-0 (152 lines); hunks: -0,0 +1,152; symbols: HYV4Config, __init__, _validate_hy_v4，涉及 `HYV4Config, __init__, _validate_hy_v4`；`test/registered/unit/models/test_hunyuan_v4.py` added +119/-0 (119 lines); hunks: -0,0 +1,119; symbols: test_attention_gate_uses_attention_tp, fake_attention_init, FakeColumnParallelLinear, __init__，涉及 `test_attention_gate_uses_attention_tp, fake_attention_init, FakeColumnParallelLinear`。
- 代码 diff 细节:
  - `python/sglang/srt/models/hunyuan_v4.py` added +734/-0 (734 lines); hunks: -0,0 +1,734; symbols: _hpc_ihc_available, permute_hyv4_indexer_weight, HYV4HCPreLayer, __init__
  - `python/sglang/srt/models/hunyuan_v4_nextn.py` added +229/-0 (229 lines); hunks: -0,0 +1,229; symbols: _mtp_quant_config, HYV4MTPDecoderLayer, __init__, forward
  - `python/sglang/srt/configs/hy_v4.py` added +152/-0 (152 lines); hunks: -0,0 +1,152; symbols: HYV4Config, __init__, _validate_hy_v4
  - `test/registered/unit/models/test_hunyuan_v4.py` added +119/-0 (119 lines); hunks: -0,0 +1,119; symbols: test_attention_gate_uses_attention_tp, fake_attention_init, FakeColumnParallelLinear, __init__
  - `test/registered/unit/models/test_hunyuan_v4_nextn_weight_loading.py` added +42/-0 (42 lines); hunks: -0,0 +1,42; symbols: TestHunyuanV4NextNWeightLoading, test_indexer_checkpoint_layout_is_permuted
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/hunyuan_v4.py
@@ -0,0 +1,734 @@
+import functools
+import logging
+from typing import Iterable, Tuple
+import torch
+from torch import nn
+from transformers import PretrainedConfig
diff -- python/sglang/srt/models/hunyuan_v4_nextn.py
@@ -0,0 +1,229 @@
+import copy
+from typing import Iterable, Tuple
+import torch
+from torch import nn
+from sglang.srt.distributed import get_pp_group
+from sglang.srt.layers.attention.index_topk_share import IndexTopKShareState
diff -- python/sglang/srt/configs/hy_v4.py
@@ -0,0 +1,152 @@
```

- 提取文件（未人工审阅）:
  - runtime: `python/sglang/srt/models/hunyuan_v4.py` added +734/-0; `python/sglang/srt/models/hunyuan_v4_nextn.py` added +229/-0; `python/sglang/srt/configs/hy_v4.py` added +152/-0; `python/sglang/kernels/ops/layernorm/hy4_ihc.py` added +443/-0
  - tests: `test/registered/unit/models/test_hunyuan_v4.py` added +119/-0; `test/registered/unit/models/test_hunyuan_v4_nextn_weight_loading.py` added +42/-0; `test/registered/kernels/ops/layernorm/test_hy4_decode_kernels.py` added +229/-0; `test/registered/kernels/ops/attention/test_hy4_hpc_gated_mla.py` added +145/-0
- 验证与风险: diff 自带测试面 `test/registered/kernels/ops/attention/test_dsa_sink_pad_cache.py`, `test/registered/kernels/ops/attention/test_hy4_hpc_gated_mla.py`, `test/registered/kernels/ops/layernorm/test_hy4_decode_kernels.py`, `test/registered/kernels/ops/layernorm/test_hy4_hpc_ihc.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #42477 - [Refactor] Build the Hunyuan V4 decoder from stage boundaries

- 链接: https://github.com/sgl-project/sglang/pull/42477
- 状态/时间: merged / 2026-10-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/hunyuan_v4.py`, `python/sglang/srt/models/hunyuan_v4_nextn.py`；关联提交 `92d60351e2eb`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 5 个文件，+531/-34，可读 patch 705 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/sglang/srt/models/hunyuan_v4.py` modified +101/-20 (121 lines); hunks: -7,7 +7,14; -342,7 +349,8 @@ def __init__(; symbols: __init__, dispatch_attn_forward_method, _HeadNorm, __call__，涉及 `__init__, dispatch_attn_forward_method, _HeadNorm`；`python/sglang/srt/models/hunyuan_v4_nextn.py` modified +29/-14 (43 lines); hunks: -5,7 +5,13; -61,9 +67,19 @@ def __init__(self, config, quant_config=None, prefix="", alt_...; symbols: __init__, forward, HYV4ModelNextN，涉及 `__init__, forward, HYV4ModelNextN`。
- 代码 diff 细节:
  - `python/sglang/srt/models/hunyuan_v4.py` modified +101/-20 (121 lines); hunks: -7,7 +7,14; -342,7 +349,8 @@ def __init__(; symbols: __init__, dispatch_attn_forward_method, _HeadNorm, __call__
  - `python/sglang/srt/models/hunyuan_v4_nextn.py` modified +29/-14 (43 lines); hunks: -5,7 +5,13; -61,9 +67,19 @@ def __init__(self, config, quant_config=None, prefix="", alt_...; symbols: __init__, forward, HYV4ModelNextN
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/hunyuan_v4.py
@@ -7,7 +7,14 @@
-from sglang.srt.layers.layer_boundary import AttentionInputs, get_attn_tp_context
+from sglang.srt.layers.layer_boundary import (
+    IHCState,
+    declare_attn,
+    declare_ffn,
+    get_attn_tp_context,
diff -- python/sglang/srt/models/hunyuan_v4_nextn.py
@@ -5,7 +5,13 @@
-from sglang.srt.layers.layer_boundary import AttentionInputs, get_attn_tp_context
+from sglang.srt.layers.layer_boundary import (
+    declare_attn,
+    declare_ffn,
+    get_attn_tp_context,
+    make_stages,
```

- 提取文件（未人工审阅）:
  - runtime: `python/sglang/srt/models/hunyuan_v4.py` modified +101/-20; `python/sglang/srt/models/hunyuan_v4_nextn.py` modified +29/-14
- 验证与风险: diff 自带测试面 `test/registered/unit/layer_boundary/test_ihc_residual_ops.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
