# SGLang ERNIE 4.5 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `docs/cookbook/autoregressive/Ernie/Ernie4.5-VL.mdx` | 无直接 PR 号提交 |
| `docs/cookbook/autoregressive/Ernie/Ernie4.5.mdx` | 无直接 PR 号提交 |
| `python/sglang/srt/arg_groups/model_overrides/ernie45_vl.py` | [#42303](https://github.com/sgl-project/sglang/pull/42303) |
| `python/sglang/srt/models/ernie4.py` | [#7657](https://github.com/sgl-project/sglang/pull/7657), [#26038](https://github.com/sgl-project/sglang/pull/26038), [#35222](https://github.com/sgl-project/sglang/pull/35222) |
| `python/sglang/srt/models/ernie45_moe_vl.py` | [#15679](https://github.com/sgl-project/sglang/pull/15679), [#42303](https://github.com/sgl-project/sglang/pull/42303), [#42304](https://github.com/sgl-project/sglang/pull/42304) |
| `python/sglang/srt/models/ernie45_vl.py` | [#15679](https://github.com/sgl-project/sglang/pull/15679), [#19743](https://github.com/sgl-project/sglang/pull/19743) |
| `python/sglang/srt/models/ernie4_eagle.py` | [#7657](https://github.com/sgl-project/sglang/pull/7657) |
| `python/sglang/srt/multimodal/processors/ernie45_vl.py` | [#15679](https://github.com/sgl-project/sglang/pull/15679) |

## PR 覆盖总览

- git 追溯 PR 数: 7
- 原文档显式引用补充 PR 数: 0
- 当前文档总 PR 数: 7
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-08-08 | [#7657](https://github.com/sgl-project/sglang/pull/7657) | merged | Add ernie4.py for ERNIE-4.5 | `python/sglang/srt/models/ernie4.py`, `python/sglang/srt/models/ernie4_eagle.py` |
| 2026-01-26 | [#15679](https://github.com/sgl-project/sglang/pull/15679) | merged | [Model] Add Ernie4.5 VL model support | `python/sglang/srt/models/ernie45_vl.py`, `python/sglang/srt/models/ernie45_moe_vl.py`, `python/sglang/srt/multimodal/processors/ernie45_vl.py` |
| 2026-03-04 | [#19743](https://github.com/sgl-project/sglang/pull/19743) | merged | [VLM] Support cos sin cache for Ernie4.5-VL | `python/sglang/srt/models/ernie45_vl.py` |
| 2026-05-28 | [#26038](https://github.com/sgl-project/sglang/pull/26038) | merged | [NPU] fix model ERNIE-4.5-21B-A3B-PT bias need 1D error | `python/sglang/srt/models/ernie4.py` |
| 2026-08-27 | [#35222](https://github.com/sgl-project/sglang/pull/35222) | merged | [CPU] Enable ERNIE models on CPU | `python/sglang/srt/models/ernie4.py` |
| 2026-10-03 | [#42303](https://github.com/sgl-project/sglang/pull/42303) | merged | [Fix] EXAONE and ERNIE 4.5 VL MoE architecture, backend and PP issues | `python/sglang/srt/models/ernie45_moe_vl.py`, `python/sglang/srt/arg_groups/model_overrides/ernie45_vl.py` |
| 2026-10-03 | [#42304](https://github.com/sgl-project/sglang/pull/42304) | merged | [Refactor] Build the ERNIE 4.5 VL MoE and EXAONE MoE decoders from stage boundaries | `python/sglang/srt/models/ernie45_moe_vl.py` |

## 逐 PR diff 审计卡

### PR #7657 - Add ernie4.py for ERNIE-4.5

- 链接: https://github.com/sgl-project/sglang/pull/7657
- 状态/时间: merged / 2025-08-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/ernie4.py`, `python/sglang/srt/models/ernie4_eagle.py`；关联提交 `113254749659`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 4 个文件，+635/-0，可读 patch 651 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/sglang/srt/models/ernie4.py` added +426/-0 (426 lines); hunks: -0,0 +1,426; symbols: MoEGate, __init__, forward, Ernie4Moe，涉及 `MoEGate, __init__, forward`；`python/sglang/srt/models/ernie4_eagle.py` added +203/-0 (203 lines); hunks: -0,0 +1,203; symbols: Ernie4ModelMTP, __init__, forward, Ernie4_5_MoeForCausalLMMTP，涉及 `Ernie4ModelMTP, __init__, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/ernie4.py` added +426/-0 (426 lines); hunks: -0,0 +1,426; symbols: MoEGate, __init__, forward, Ernie4Moe
  - `python/sglang/srt/models/ernie4_eagle.py` added +203/-0 (203 lines); hunks: -0,0 +1,203; symbols: Ernie4ModelMTP, __init__, forward, Ernie4_5_MoeForCausalLMMTP
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/ernie4.py
@@ -0,0 +1,426 @@
+# Copyright 2023-2025 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/models/ernie4_eagle.py
@@ -0,0 +1,203 @@
+# Copyright 2023-2025 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
```

- 提取文件（未人工审阅）:
  - runtime: `python/sglang/srt/models/ernie4.py` added +426/-0; `python/sglang/srt/models/ernie4_eagle.py` added +203/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/models/ernie4.py`, `python/sglang/srt/models/ernie4_eagle.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #15679 - [Model] Add Ernie4.5 VL model support

- 链接: https://github.com/sgl-project/sglang/pull/15679
- 状态/时间: merged / 2026-01-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/ernie45_moe_vl.py`, `python/sglang/srt/models/ernie45_vl.py`, `python/sglang/srt/multimodal/processors/ernie45_vl.py`；关联提交 `1a19b3987dca`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 6 个文件，+2072/-0，可读 patch 2103 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/sglang/srt/models/ernie45_vl.py` added +845/-0 (845 lines); hunks: -0,0 +1,845; symbols: Ernie4_5_VisionMLP, __init__, forward, Ernie4_5_VisionBlock，涉及 `Ernie4_5_VisionMLP, __init__, forward`；`python/sglang/srt/models/ernie45_moe_vl.py` added +552/-0 (552 lines); hunks: -0,0 +1,552; symbols: Ernie4_5_VLMoeAttention, __init__, forward, Ernie4_5_VLMoeMoE，涉及 `Ernie4_5_VLMoeAttention, __init__, forward`；`python/sglang/srt/multimodal/processors/ernie45_vl.py` added +417/-0 (417 lines); hunks: -0,0 +1,417; symbols: smart_resize, resize_image, round_by_factor, ceil_by_factor，涉及 `smart_resize, resize_image, round_by_factor`。
- 代码 diff 细节:
  - `python/sglang/srt/models/ernie45_vl.py` added +845/-0 (845 lines); hunks: -0,0 +1,845; symbols: Ernie4_5_VisionMLP, __init__, forward, Ernie4_5_VisionBlock
  - `python/sglang/srt/models/ernie45_moe_vl.py` added +552/-0 (552 lines); hunks: -0,0 +1,552; symbols: Ernie4_5_VLMoeAttention, __init__, forward, Ernie4_5_VLMoeMoE
  - `python/sglang/srt/multimodal/processors/ernie45_vl.py` added +417/-0 (417 lines); hunks: -0,0 +1,417; symbols: smart_resize, resize_image, round_by_factor, ceil_by_factor
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/ernie45_vl.py
@@ -0,0 +1,845 @@
+# Copyright 2023-2025 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/models/ernie45_moe_vl.py
@@ -0,0 +1,552 @@
+# Copyright 2023-2025 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/multimodal/processors/ernie45_vl.py
@@ -0,0 +1,417 @@
```

- 提取文件（未人工审阅）:
  - runtime: `python/sglang/srt/models/ernie45_vl.py` added +845/-0; `python/sglang/srt/models/ernie45_moe_vl.py` added +552/-0; `python/sglang/srt/multimodal/processors/ernie45_vl.py` added +417/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/layers/rotary_embedding.py`, `python/sglang/srt/models/ernie45_moe_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #19743 - [VLM] Support cos sin cache for Ernie4.5-VL

- 链接: https://github.com/sgl-project/sglang/pull/19743
- 状态/时间: merged / 2026-03-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/ernie45_vl.py`；关联提交 `82e7139c06a3`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+34/-12，可读 patch 102 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/sglang/srt/models/ernie45_vl.py` modified +34/-12 (46 lines); hunks: -30,6 +30,7; -120,14 +121,16 @@ def forward(; symbols: forward, __init__, dtype, device，涉及 `forward, __init__, dtype`。
- 代码 diff 细节:
  - `python/sglang/srt/models/ernie45_vl.py` modified +34/-12 (46 lines); hunks: -30,6 +30,7; -120,14 +121,16 @@ def forward(; symbols: forward, __init__, dtype, device
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/ernie45_vl.py
@@ -30,6 +30,7 @@
+from sglang.srt.layers.rotary_embedding import get_rope
@@ -120,14 +121,16 @@ def forward(
-        position_embeddings: torch.Tensor,
+        rotary_pos_emb_cos: torch.Tensor,
+        rotary_pos_emb_sin: torch.Tensor,
-            position_embeddings=position_embeddings,
```

- 提取文件（未人工审阅）:
  - runtime: `python/sglang/srt/models/ernie45_vl.py` modified +34/-12
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/ernie45_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #26038 - [NPU] fix model ERNIE-4.5-21B-A3B-PT bias need 1D error

- 链接: https://github.com/sgl-project/sglang/pull/26038
- 状态/时间: merged / 2026-05-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/ernie4.py`；关联提交 `a245cae3d1e7`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+9/-2，可读 patch 39 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/sglang/srt/models/ernie4.py` modified +8/-2 (10 lines); hunks: -42,9 +42,11; -86,12 +88,16 @@ def __init__(; symbols: MoEGate, __init__，涉及 `MoEGate, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/ernie4.py` modified +8/-2 (10 lines); hunks: -42,9 +42,11; -86,12 +88,16 @@ def __init__(; symbols: MoEGate, __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/ernie4.py
@@ -42,9 +42,11 @@
-from sglang.srt.utils import add_prefix, make_layers
+from sglang.srt.utils import add_prefix, is_npu, make_layers
+_is_npu = is_npu()
@@ -86,12 +88,16 @@ def __init__(
+        correction_bias = self.gate.e_score_correction_bias
+        # npu only supports 1D, but current correction_bias is 2D
```

- 提取文件（未人工审阅）:
  - runtime: `python/sglang/srt/models/ernie4.py` modified +8/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/ernie4.py`, `python/sglang/srt/models/llama.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #35222 - [CPU] Enable ERNIE models on CPU

- 链接: https://github.com/sgl-project/sglang/pull/35222
- 状态/时间: merged / 2026-08-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/ernie4.py`；关联提交 `5adc2880f996`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+6/-3，可读 patch 32 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/sglang/srt/models/ernie4.py` modified +4/-3 (7 lines); hunks: -42,9 +42,10; -89,8 +90,8 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/ernie4.py` modified +4/-3 (7 lines); hunks: -42,9 +42,10; -89,8 +90,8 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/ernie4.py
@@ -42,9 +42,10 @@
-from sglang.srt.utils import add_prefix, is_npu, make_layers
+from sglang.srt.utils import add_prefix, is_cpu, is_npu, make_layers
+_is_cpu = is_cpu()
@@ -89,8 +90,8 @@ def __init__(
-        # npu only supports 1D, but current correction_bias is 2D
-        if _is_npu:
```

- 提取文件（未人工审阅）:
  - runtime: `python/sglang/srt/models/ernie4.py` modified +4/-3
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/models/ernie4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #42303 - [Fix] EXAONE and ERNIE 4.5 VL MoE architecture, backend and PP issues

- 链接: https://github.com/sgl-project/sglang/pull/42303
- 状态/时间: merged / 2026-10-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/arg_groups/model_overrides/ernie45_vl.py`, `python/sglang/srt/models/ernie45_moe_vl.py`；关联提交 `f89ead192f3c`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 8 个文件，+65/-10，可读 patch 168 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/sglang/srt/models/ernie45_moe_vl.py` modified +0/-2 (2 lines); hunks: -25,7 +25,6; -479,7 +478,6 @@ def __init__(; symbols: __init__，涉及 `__init__`；`python/sglang/srt/arg_groups/model_overrides/ernie45_vl.py` added +19/-0 (19 lines); hunks: -0,0 +1,19; symbols: _ernie45_vl_overrides，涉及 `_ernie45_vl_overrides`。
- 代码 diff 细节:
  - `python/sglang/srt/models/ernie45_moe_vl.py` modified +0/-2 (2 lines); hunks: -25,7 +25,6; -479,7 +478,6 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/arg_groups/model_overrides/ernie45_vl.py` added +19/-0 (19 lines); hunks: -0,0 +1,19; symbols: _ernie45_vl_overrides
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/ernie45_moe_vl.py
@@ -25,7 +25,6 @@
-from sglang.srt.layers.dp_attention import is_dp_attention_enabled
@@ -479,7 +478,6 @@ def __init__(
-                enable_tp=not is_dp_attention_enabled(),
diff -- python/sglang/srt/arg_groups/model_overrides/ernie45_vl.py
@@ -0,0 +1,19 @@
+"""Config-time override declarations for ernie45_vl."""
+from typing import Any
+from sglang.srt.arg_groups.model_override_base import (
+    _register_for,
+    resolving_view,
+)
```

- 提取文件（未人工审阅）:
  - runtime: `python/sglang/srt/models/ernie45_moe_vl.py` modified +0/-2; `python/sglang/srt/arg_groups/model_overrides/ernie45_vl.py` added +19/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/arg_groups/model_hook.py`, `python/sglang/srt/arg_groups/model_overrides/__init__.py`, `python/sglang/srt/arg_groups/model_overrides/ernie45_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #42304 - [Refactor] Build the ERNIE 4.5 VL MoE and EXAONE MoE decoders from stage boundaries

- 链接: https://github.com/sgl-project/sglang/pull/42304
- 状态/时间: merged / 2026-10-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/ernie45_moe_vl.py`；关联提交 `2fa17a64671b`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+126/-100，可读 patch 403 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/sglang/srt/models/ernie45_moe_vl.py` modified +66/-58 (124 lines); hunks: -16,15 +16,18; -107,6 +110,7 @@ def __init__(; symbols: __init__, forward, _is_moe_layer, Ernie4_5_VLMoeDecoderLayer，涉及 `__init__, forward, _is_moe_layer`。
- 代码 diff 细节:
  - `python/sglang/srt/models/ernie45_moe_vl.py` modified +66/-58 (124 lines); hunks: -16,15 +16,18; -107,6 +110,7 @@ def __init__(; symbols: __init__, forward, _is_moe_layer, Ernie4_5_VLMoeDecoderLayer
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/ernie45_moe_vl.py
@@ -16,15 +16,18 @@
-from typing import Any, Dict, Optional, Tuple, Union
+from typing import Any, Dict, Optional, Union
-from sglang.srt.distributed import (
-    tensor_model_parallel_all_reduce,
+from sglang.srt.layers.layer_boundary import (
+    declare_attn,
```

- 提取文件（未人工审阅）:
  - runtime: `python/sglang/srt/models/ernie45_moe_vl.py` modified +66/-58
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/ernie45_moe_vl.py`, `python/sglang/srt/models/exaone_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
