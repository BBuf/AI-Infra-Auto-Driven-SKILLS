# SGLang EXAONE 4/4.5/K-EXAONE 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `python/sglang/srt/models/exaone4.py` | [#8205](https://github.com/sgl-project/sglang/pull/8205), [#42303](https://github.com/sgl-project/sglang/pull/42303) |
| `python/sglang/srt/models/exaone_moe.py` | [#16294](https://github.com/sgl-project/sglang/pull/16294), [#42303](https://github.com/sgl-project/sglang/pull/42303), [#42304](https://github.com/sgl-project/sglang/pull/42304), [#42305](https://github.com/sgl-project/sglang/pull/42305) |
| `python/sglang/srt/models/exaone_moe_mtp.py` | [#16294](https://github.com/sgl-project/sglang/pull/16294) |

## PR 覆盖总览

- git 追溯 PR 数: 5
- 原文档显式引用补充 PR 数: 0
- 当前文档总 PR 数: 5
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-01-27 | [#8205](https://github.com/sgl-project/sglang/pull/8205) | merged | [Model] Add support for EXAONE-4.0 Model | `python/sglang/srt/models/exaone4.py` |
| 2026-01-30 | [#16294](https://github.com/sgl-project/sglang/pull/16294) | merged | [Model] Add K-EXAONE model support | `python/sglang/srt/models/exaone_moe.py`, `python/sglang/srt/models/exaone_moe_mtp.py` |
| 2026-10-03 | [#42303](https://github.com/sgl-project/sglang/pull/42303) | merged | [Fix] EXAONE and ERNIE 4.5 VL MoE architecture, backend and PP issues | `python/sglang/srt/models/exaone_moe.py`, `python/sglang/srt/models/exaone4.py` |
| 2026-10-03 | [#42304](https://github.com/sgl-project/sglang/pull/42304) | merged | [Refactor] Build the ERNIE 4.5 VL MoE and EXAONE MoE decoders from stage boundaries | `python/sglang/srt/models/exaone_moe.py` |
| 2026-10-03 | [#42305](https://github.com/sgl-project/sglang/pull/42305) | merged | [Fix] EXAONE MoE under DP attention and DeepEP | `python/sglang/srt/models/exaone_moe.py` |

## 逐 PR diff 审计卡

### PR #8205 - [Model] Add support for EXAONE-4.0 Model

- 链接: https://github.com/sgl-project/sglang/pull/8205
- 状态/时间: merged / 2026-01-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/exaone4.py`；关联提交 `81c0f5c5adf0`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+728/-0，可读 patch 736 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/sglang/srt/models/exaone4.py` added +719/-0 (719 lines); hunks: -0,0 +1,719; symbols: get_attention_sliding_window_size, Exaone4GatedMLP, __init__, forward，涉及 `get_attention_sliding_window_size, Exaone4GatedMLP, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/exaone4.py` added +719/-0 (719 lines); hunks: -0,0 +1,719; symbols: get_attention_sliding_window_size, Exaone4GatedMLP, __init__, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/exaone4.py
@@ -0,0 +1,719 @@
+from collections.abc import Iterable
+from typing import Any, List, Optional, Tuple, Union
+import torch
+from torch import nn
+from transformers import Exaone4Config
+from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size
```

- 提取文件（未人工审阅）:
  - runtime: `python/sglang/srt/models/exaone4.py` added +719/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/exaone4.py`, `python/sglang/srt/server_args.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #16294 - [Model] Add K-EXAONE model support

- 链接: https://github.com/sgl-project/sglang/pull/16294
- 状态/时间: merged / 2026-01-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/exaone_moe.py`, `python/sglang/srt/models/exaone_moe_mtp.py`；关联提交 `c04efe030acc`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 4 个文件，+1000/-7，可读 patch 1025 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/sglang/srt/models/exaone_moe.py` added +881/-0 (881 lines); hunks: -0,0 +1,881; symbols: ExaoneMoEMLP, __init__, forward, ExaoneMoESparseMoEBlock，涉及 `ExaoneMoEMLP, __init__, forward`；`python/sglang/srt/models/exaone_moe_mtp.py` added +106/-0 (106 lines); hunks: -0,0 +1,106; symbols: ExaoneMoEForCausalLMMTP, __init__, forward, load_weights，涉及 `ExaoneMoEForCausalLMMTP, __init__, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/exaone_moe.py` added +881/-0 (881 lines); hunks: -0,0 +1,881; symbols: ExaoneMoEMLP, __init__, forward, ExaoneMoESparseMoEBlock
  - `python/sglang/srt/models/exaone_moe_mtp.py` added +106/-0 (106 lines); hunks: -0,0 +1,106; symbols: ExaoneMoEForCausalLMMTP, __init__, forward, load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/exaone_moe.py
@@ -0,0 +1,881 @@
+# Copyright 2025 The LG AI Research Team
+# Copyright 2023-2024 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
diff -- python/sglang/srt/models/exaone_moe_mtp.py
@@ -0,0 +1,106 @@
+# Copyright 2025 The LG AI Research Team
+# Copyright 2023-2024 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
```

- 提取文件（未人工审阅）:
  - runtime: `python/sglang/srt/models/exaone_moe.py` added +881/-0; `python/sglang/srt/models/exaone_moe_mtp.py` added +106/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/models/exaone_moe.py`, `python/sglang/srt/models/exaone_moe_mtp.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #42303 - [Fix] EXAONE and ERNIE 4.5 VL MoE architecture, backend and PP issues

- 链接: https://github.com/sgl-project/sglang/pull/42303
- 状态/时间: merged / 2026-10-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/exaone4.py`, `python/sglang/srt/models/exaone_moe.py`；关联提交 `f89ead192f3c`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 8 个文件，+65/-10，可读 patch 168 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/sglang/srt/models/exaone_moe.py` modified +12/-1 (13 lines); hunks: -786,6 +786,7 @@ def load_weights(; -807,7 +808,11 @@ def load_weights(; symbols: load_weights, set_eagle3_layers_to_capture, ExaoneMoeForCausalLM，涉及 `load_weights, set_eagle3_layers_to_capture, ExaoneMoeForCausalLM`；`python/sglang/srt/models/exaone4.py` modified +10/-1 (11 lines); hunks: -435,7 +435,7 @@ def __init__(; -575,6 +575,15 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: __init__, load_weights，涉及 `__init__, load_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/exaone_moe.py` modified +12/-1 (13 lines); hunks: -786,6 +786,7 @@ def load_weights(; -807,7 +808,11 @@ def load_weights(; symbols: load_weights, set_eagle3_layers_to_capture, ExaoneMoeForCausalLM
  - `python/sglang/srt/models/exaone4.py` modified +10/-1 (11 lines); hunks: -435,7 +435,7 @@ def __init__(; -575,6 +575,15 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: __init__, load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/exaone_moe.py
@@ -786,6 +786,7 @@ def load_weights(
+            is_expert_weight = False
@@ -807,7 +808,11 @@ def load_weights(
+                    is_expert_weight = True
+                    if name not in params_dict:
+                        # The expert's layer lives on another pipeline rank.
+                        continue
diff -- python/sglang/srt/models/exaone4.py
@@ -435,7 +435,7 @@ def __init__(
-        if config.tie_word_embeddings:
+        if config.tie_word_embeddings and self.pp_group.world_size == 1:
@@ -575,6 +575,15 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
+            if (
+                name == "model.embed_tokens.weight"
+                and self.config.tie_word_embeddings
```

- 提取文件（未人工审阅）:
  - runtime: `python/sglang/srt/models/exaone_moe.py` modified +12/-1; `python/sglang/srt/models/exaone4.py` modified +10/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/arg_groups/model_hook.py`, `python/sglang/srt/arg_groups/model_overrides/__init__.py`, `python/sglang/srt/arg_groups/model_overrides/ernie45_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #42304 - [Refactor] Build the ERNIE 4.5 VL MoE and EXAONE MoE decoders from stage boundaries

- 链接: https://github.com/sgl-project/sglang/pull/42304
- 状态/时间: merged / 2026-10-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/exaone_moe.py`；关联提交 `2fa17a64671b`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+126/-100，可读 patch 403 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/sglang/srt/models/exaone_moe.py` modified +60/-42 (102 lines); hunks: -28,9 +28,16; -39,10 +46,7; symbols: forward, __init__，涉及 `forward, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/exaone_moe.py` modified +60/-42 (102 lines); hunks: -28,9 +28,16; -39,10 +46,7; symbols: forward, __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/exaone_moe.py
@@ -28,9 +28,16 @@
+from sglang.srt.layers.aux_hidden_states import AuxHiddenStateList
+from sglang.srt.layers.layer_boundary import (
+    declare_attn,
+    declare_ffn,
+    make_stages,
+)
```

- 提取文件（未人工审阅）:
  - runtime: `python/sglang/srt/models/exaone_moe.py` modified +60/-42
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/ernie45_moe_vl.py`, `python/sglang/srt/models/exaone_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #42305 - [Fix] EXAONE MoE under DP attention and DeepEP

- 链接: https://github.com/sgl-project/sglang/pull/42305
- 状态/时间: merged / 2026-10-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/exaone_moe.py`；关联提交 `b7fc51698601`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+3/-1，可读 patch 18 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/sglang/srt/models/exaone_moe.py` modified +3/-1 (4 lines); hunks: -406,6 +406,8 @@ def forward(; -532,7 +534,7 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/exaone_moe.py` modified +3/-1 (4 lines); hunks: -406,6 +406,8 @@ def forward(; -532,7 +534,7 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/exaone_moe.py
@@ -406,6 +406,8 @@ def forward(
+        if hidden_states.shape[0] == 0:
+            return hidden_states
@@ -532,7 +534,7 @@ def forward(
-            hidden_states = self.mlp(hidden_states)
+            hidden_states = self.mlp(hidden_states, forward_batch)
```

- 提取文件（未人工审阅）:
  - runtime: `python/sglang/srt/models/exaone_moe.py` modified +3/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/exaone_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
