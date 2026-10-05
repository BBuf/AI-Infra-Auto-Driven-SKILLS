# TensorRT-LLM EXAONE 4/4.5/K-EXAONE 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `tensorrt_llm/_torch/models/checkpoints/hf/exaone4_5_weight_mapper.py` | [#12873](https://github.com/NVIDIA/TensorRT-LLM/pull/12873) |
| `tensorrt_llm/_torch/models/checkpoints/hf/exaone_moe_weight_mapper.py` | [#10796](https://github.com/NVIDIA/TensorRT-LLM/pull/10796), [#11862](https://github.com/NVIDIA/TensorRT-LLM/pull/11862) |
| `tensorrt_llm/_torch/models/modeling_exaone4.py` | [#5696](https://github.com/NVIDIA/TensorRT-LLM/pull/5696), [#6397](https://github.com/NVIDIA/TensorRT-LLM/pull/6397), [#8429](https://github.com/NVIDIA/TensorRT-LLM/pull/8429) |
| `tensorrt_llm/_torch/models/modeling_exaone4_5.py` | [#12873](https://github.com/NVIDIA/TensorRT-LLM/pull/12873), [#16992](https://github.com/NVIDIA/TensorRT-LLM/pull/16992) |
| `tensorrt_llm/_torch/models/modeling_exaone_moe.py` | [#10355](https://github.com/NVIDIA/TensorRT-LLM/pull/10355), [#10796](https://github.com/NVIDIA/TensorRT-LLM/pull/10796) |
| `tests/unittest/_torch/modeling/test_modeling_exaone4_5.py` | [#12873](https://github.com/NVIDIA/TensorRT-LLM/pull/12873), [#16992](https://github.com/NVIDIA/TensorRT-LLM/pull/16992), [#19246](https://github.com/NVIDIA/TensorRT-LLM/pull/19246) |
| `tests/unittest/_torch/modeling/test_modeling_exaone_moe.py` | [#10355](https://github.com/NVIDIA/TensorRT-LLM/pull/10355) |

## PR 覆盖总览

- git 追溯 PR 数: 9
- 原文档显式引用补充 PR 数: 0
- 当前文档总 PR 数: 9
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-07-14 | [#5696](https://github.com/NVIDIA/TensorRT-LLM/pull/5696) | merged | feat: EXAONE4.0 support | `tensorrt_llm/_torch/models/modeling_exaone4.py` |
| 2025-08-04 | [#6397](https://github.com/NVIDIA/TensorRT-LLM/pull/6397) | merged | chore: add EXAONE4 accuracy test | `tensorrt_llm/_torch/models/modeling_exaone4.py`, `tests/unittest/_torch/modeling/test_modeling_exaone4.py` |
| 2025-10-21 | [#8429](https://github.com/NVIDIA/TensorRT-LLM/pull/8429) | merged | [https://nvbugs/5569713][fix] Disable fp8 deep gemm for EXAONE-4.0-32B-FP8 | `tensorrt_llm/_torch/models/modeling_exaone4.py`, `tests/unittest/_torch/modeling/test_modeling_exaone4.py` |
| 2026-01-11 | [#10355](https://github.com/NVIDIA/TensorRT-LLM/pull/10355) | merged | [TRTLLM-10195][feat] K-EXAONE support | `tensorrt_llm/_torch/models/modeling_exaone_moe.py`, `tests/unittest/_torch/modeling/test_modeling_exaone_moe.py` |
| 2026-01-22 | [#10796](https://github.com/NVIDIA/TensorRT-LLM/pull/10796) | merged | [None][feat] K-EXAONE MTP support | `tensorrt_llm/_torch/models/modeling_exaone_moe.py`, `tensorrt_llm/_torch/models/checkpoints/hf/exaone_moe_weight_mapper.py` |
| 2026-03-05 | [#11862](https://github.com/NVIDIA/TensorRT-LLM/pull/11862) | merged | [None][fix] Prevent RuntimeError from dict mutation during iteration in EXAONE MoE weight mapper | `tensorrt_llm/_torch/models/checkpoints/hf/exaone_moe_weight_mapper.py` |
| 2026-05-21 | [#12873](https://github.com/NVIDIA/TensorRT-LLM/pull/12873) | merged | [None][feat] EXAONE-4.5 Support | `tensorrt_llm/_torch/models/modeling_exaone4_5.py`, `tensorrt_llm/_torch/models/checkpoints/hf/exaone4_5_weight_mapper.py`, `tests/unittest/_torch/modeling/test_modeling_exaone4_5.py` |
| 2026-08-06 | [#16992](https://github.com/NVIDIA/TensorRT-LLM/pull/16992) | merged | [https://nvbugs/6327149][fix] Handle EXAONE 4.5 33B memory constraints | `tensorrt_llm/_torch/models/modeling_exaone4_5.py`, `tests/unittest/_torch/modeling/test_modeling_exaone4_5.py` |
| 2026-09-17 | [#19246](https://github.com/NVIDIA/TensorRT-LLM/pull/19246) | merged | [None][test] Add coverage for Exaone4ForCausalLM | `tests/unittest/_torch/modeling/test_modeling_exaone4_5.py` |

## 逐 PR diff 审计卡

### PR #5696 - feat: EXAONE4.0 support

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/5696
- 状态/时间: merged / 2025-07-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_exaone4.py`；关联提交 `63139fdcff31`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 3 个文件，+392/-1，可读 patch 432 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_exaone4.py` added +322/-0 (322 lines); hunks: -0,0 +1,322; symbols: Exaone4Config, check_is_sliding, Exaone4Attention, __init__，涉及 `Exaone4Config, check_is_sliding, Exaone4Attention`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_exaone4.py` added +322/-0 (322 lines); hunks: -0,0 +1,322; symbols: Exaone4Config, check_is_sliding, Exaone4Attention, __init__
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_exaone4.py
@@ -0,0 +1,322 @@
+from typing import Optional, Tuple
+import torch
+from torch import nn
+from tensorrt_llm._torch.distributed import AllReduceParams
+from tensorrt_llm.functional import PositionEmbeddingType
+from ..attention_backend import AttentionMetadata
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_exaone4.py` added +322/-0
- 验证与风险: runtime 路径改动集中在 `tensorrt_llm/_torch/models/__init__.py`, `tensorrt_llm/_torch/models/modeling_exaone4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #6397 - chore: add EXAONE4 accuracy test

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/6397
- 状态/时间: merged / 2025-08-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_exaone4.py`；关联提交 `ee6ab5be962b`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 8 个文件，+412/-16，可读 patch 529 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_exaone4.py` modified +5/-16 (21 lines); hunks: -3,7 +3,6; -55,7 +54,6 @@ class Exaone4Attention(Attention):; symbols: Exaone4Attention, __init__, apply_rope, forward，涉及 `Exaone4Attention, __init__, apply_rope`；`tests/unittest/_torch/modeling/test_modeling_exaone4.py` added +384/-0 (384 lines); hunks: -0,0 +1,384; symbols: Exaone4Config, Scenario, __repr__, TestEXAONE4，涉及 `Exaone4Config, Scenario, __repr__`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_exaone4.py` modified +5/-16 (21 lines); hunks: -3,7 +3,6; -55,7 +54,6 @@ class Exaone4Attention(Attention):; symbols: Exaone4Attention, __init__, apply_rope, forward
  - `tests/unittest/_torch/modeling/test_modeling_exaone4.py` added +384/-0 (384 lines); hunks: -0,0 +1,384; symbols: Exaone4Config, Scenario, __repr__, TestEXAONE4
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_exaone4.py
@@ -3,7 +3,6 @@
-from tensorrt_llm._torch.distributed import AllReduceParams
@@ -55,7 +54,6 @@ class Exaone4Attention(Attention):
-                 is_sliding: bool,
@@ -64,9 +62,10 @@ def __init__(self,
-        self.is_sliding = is_sliding
+        self.sliding_window = config.sliding_window
diff -- tests/unittest/_torch/modeling/test_modeling_exaone4.py
@@ -0,0 +1,384 @@
+import unittest
+from copy import deepcopy
+from dataclasses import dataclass
+import torch
+from parameterized import parameterized
+try:
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_exaone4.py` modified +5/-16
  - tests: `tests/unittest/_torch/modeling/test_modeling_exaone4.py` added +384/-0
- 验证与风险: diff 自带测试面 `tests/integration/defs/accuracy/references/gsm8k.yaml`, `tests/integration/defs/accuracy/references/mmlu.yaml`, `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/integration/test_lists/qa/examples_test_list.txt`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #8429 - [https://nvbugs/5569713][fix] Disable fp8 deep gemm for EXAONE-4.0-32B-FP8

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/8429
- 状态/时间: merged / 2025-10-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_exaone4.py`；关联提交 `ee6944bfa2be`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 3 个文件，+56/-3，可读 patch 126 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_exaone4.py` modified +13/-1 (14 lines); hunks: -5,6 +5,7; -54,7 +55,8 @@ class Exaone4Attention(QKNormRoPEAttention):; symbols: Exaone4Attention, __init__, forward，涉及 `Exaone4Attention, __init__, forward`；`tests/unittest/_torch/modeling/test_modeling_exaone4.py` modified +42/-2 (44 lines); hunks: -1,3 +1,6; -51,8 +54,9 @@ class Exaone4Config(PretrainedConfig):; symbols: Exaone4Config, Scenario, run_forward, test_llm_load，涉及 `Exaone4Config, Scenario, run_forward`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_exaone4.py` modified +13/-1 (14 lines); hunks: -5,6 +5,7; -54,7 +55,8 @@ class Exaone4Attention(QKNormRoPEAttention):; symbols: Exaone4Attention, __init__, forward
  - `tests/unittest/_torch/modeling/test_modeling_exaone4.py` modified +42/-2 (44 lines); hunks: -1,3 +1,6; -51,8 +54,9 @@ class Exaone4Config(PretrainedConfig):; symbols: Exaone4Config, Scenario, run_forward, test_llm_load
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_exaone4.py
@@ -5,6 +5,7 @@
+from tensorrt_llm.quantization import QuantAlgo
@@ -54,7 +55,8 @@ class Exaone4Attention(QKNormRoPEAttention):
-                 fuse_qk_norm_rope: bool = False):
+                 fuse_qk_norm_rope: bool = False,
+                 disable_deep_gemm: bool = False):
@@ -88,6 +90,7 @@ def __init__(self,
diff -- tests/unittest/_torch/modeling/test_modeling_exaone4.py
@@ -1,3 +1,6 @@
+import json
+import os
+import shutil
@@ -51,8 +54,9 @@ class Exaone4Config(PretrainedConfig):
-    "num_hidden_layers":
-    4,  #NOTE: For testing, we use 4 instead of 64(all layers)
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_exaone4.py` modified +13/-1
  - tests: `tests/unittest/_torch/modeling/test_modeling_exaone4.py` modified +42/-2
- 验证与风险: diff 自带测试面 `tests/integration/test_lists/test-db/l0_b200.yml`, `tests/unittest/_torch/modeling/test_modeling_exaone4.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #10355 - [TRTLLM-10195][feat] K-EXAONE support

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/10355
- 状态/时间: merged / 2026-01-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_exaone_moe.py`, `tests/unittest/_torch/modeling/test_modeling_exaone_moe.py`；关联提交 `8e0d20d90153`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 6 个文件，+1330/-63，可读 patch 1562 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_exaone_moe.py` added +581/-0 (581 lines); hunks: -0,0 +1,581; symbols: ExaoneMoEConfig, check_is_moe, enable_attn_allreduce, ExaoneMoeAttention，涉及 `ExaoneMoEConfig, check_is_moe, enable_attn_allreduce`；`tests/unittest/_torch/modeling/test_modeling_exaone_moe.py` added +406/-0 (406 lines); hunks: -0,0 +1,406; symbols: Scenario, __repr__, TestExaoneMoe, test_exaone_moe_sanity，涉及 `Scenario, __repr__, TestExaoneMoe`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_exaone_moe.py` added +581/-0 (581 lines); hunks: -0,0 +1,581; symbols: ExaoneMoEConfig, check_is_moe, enable_attn_allreduce, ExaoneMoeAttention
  - `tests/unittest/_torch/modeling/test_modeling_exaone_moe.py` added +406/-0 (406 lines); hunks: -0,0 +1,406; symbols: Scenario, __repr__, TestExaoneMoe, test_exaone_moe_sanity
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_exaone_moe.py
@@ -0,0 +1,581 @@
+import math
+import os
+import re
+from typing import Dict, List, Optional, Tuple
+import torch
+from torch import nn
diff -- tests/unittest/_torch/modeling/test_modeling_exaone_moe.py
@@ -0,0 +1,406 @@
+import unittest
+from copy import deepcopy
+from dataclasses import dataclass
+import torch
+from _torch.helpers import create_mock_cuda_graph_runner
+from parameterized import parameterized
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_exaone_moe.py` added +581/-0
  - tests: `tests/unittest/_torch/modeling/test_modeling_exaone_moe.py` added +406/-0
- 验证与风险: diff 自带测试面 `tests/integration/test_lists/test-db/l0_b200.yml`, `tests/unittest/_torch/modeling/test_modeling_exaone_moe.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #10796 - [None][feat] K-EXAONE MTP support

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/10796
- 状态/时间: merged / 2026-01-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/checkpoints/hf/exaone_moe_weight_mapper.py`, `tensorrt_llm/_torch/models/modeling_exaone_moe.py`；关联提交 `70caa779a469`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 9 个文件，+385/-61，可读 patch 631 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_exaone_moe.py` modified +181/-44 (225 lines); hunks: -1,6 +1,5; -32,15 +31,15; symbols: ExaoneMoEConfig, check_is_moe, enable_attn_allreduce, __init__，涉及 `ExaoneMoEConfig, check_is_moe, enable_attn_allreduce`；`tensorrt_llm/_torch/models/checkpoints/hf/exaone_moe_weight_mapper.py` added +67/-0 (67 lines); hunks: -0,0 +1,67; symbols: ExaoneMoeWeightMapper, __init__, preprocess_weights, is_special_instance_module，涉及 `ExaoneMoeWeightMapper, __init__, preprocess_weights`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_exaone_moe.py` modified +181/-44 (225 lines); hunks: -1,6 +1,5; -32,15 +31,15; symbols: ExaoneMoEConfig, check_is_moe, enable_attn_allreduce, __init__
  - `tensorrt_llm/_torch/models/checkpoints/hf/exaone_moe_weight_mapper.py` added +67/-0 (67 lines); hunks: -0,0 +1,67; symbols: ExaoneMoeWeightMapper, __init__, preprocess_weights, is_special_instance_module
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_exaone_moe.py
@@ -1,6 +1,5 @@
-import re
@@ -32,15 +31,15 @@
-from ..modules.linear import TensorParallelMode
+from ..modules.linear import Linear, TensorParallelMode
+from ..modules.multi_stream_utils import maybe_execute_in_parallel
-from ..utils import AuxStreamType, Fp4QuantizedTensor
diff -- tensorrt_llm/_torch/models/checkpoints/hf/exaone_moe_weight_mapper.py
@@ -0,0 +1,67 @@
+from torch import nn
+from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import HfWeightMapper
+from tensorrt_llm._torch.models.modeling_utils import register_mapper
+from tensorrt_llm._torch.modules.fused_moe.interface import MoE
+@register_mapper("HF", "ExaoneMoEForCausalLM")
+class ExaoneMoeWeightMapper(HfWeightMapper):
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_exaone_moe.py` modified +181/-44; `tensorrt_llm/_torch/models/checkpoints/hf/exaone_moe_weight_mapper.py` added +67/-0
- 验证与风险: runtime 路径改动集中在 `tensorrt_llm/_torch/model_config.py`, `tensorrt_llm/_torch/models/checkpoints/hf/exaone_moe_weight_mapper.py`, `tensorrt_llm/_torch/models/modeling_auto.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #11862 - [None][fix] Prevent RuntimeError from dict mutation during iteration in EXAONE MoE weight mapper

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/11862
- 状态/时间: merged / 2026-03-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/checkpoints/hf/exaone_moe_weight_mapper.py`；关联提交 `5b0e8a9290bd`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/checkpoints/hf/exaone_moe_weight_mapper.py` modified +1/-1 (2 lines); hunks: -28,7 +28,7 @@ def __init__(self):; symbols: __init__, preprocess_weights，涉及 `__init__, preprocess_weights`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/checkpoints/hf/exaone_moe_weight_mapper.py` modified +1/-1 (2 lines); hunks: -28,7 +28,7 @@ def __init__(self):; symbols: __init__, preprocess_weights
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/checkpoints/hf/exaone_moe_weight_mapper.py
@@ -28,7 +28,7 @@ def __init__(self):
-        for name in weights.keys():
+        for name in list(weights.keys()):
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/checkpoints/hf/exaone_moe_weight_mapper.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `tensorrt_llm/_torch/models/checkpoints/hf/exaone_moe_weight_mapper.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #12873 - [None][feat] EXAONE-4.5 Support

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/12873
- 状态/时间: merged / 2026-05-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/checkpoints/hf/exaone4_5_weight_mapper.py`, `tensorrt_llm/_torch/models/modeling_exaone4_5.py`, `tests/unittest/_torch/modeling/test_modeling_exaone4_5.py`；关联提交 `7279d6322d72`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 18 个文件，+1269/-338，可读 patch 2020 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_exaone4_5.py` added +267/-0 (267 lines); hunks: -0,0 +1,267; symbols: Exaone4_5_VisionConfig, Exaone4_5Config, __init__, Exaone4_5InputProcessor，涉及 `Exaone4_5_VisionConfig, Exaone4_5Config, __init__`；`tensorrt_llm/_torch/models/checkpoints/hf/exaone4_5_weight_mapper.py` added +38/-0 (38 lines); hunks: -0,0 +1,38; symbols: Exaone4_5HfWeightMapper, preprocess_weights，涉及 `Exaone4_5HfWeightMapper, preprocess_weights`；`tests/unittest/_torch/modeling/test_modeling_exaone4_5.py` added +255/-0 (255 lines); hunks: -0,0 +1,255; symbols: TestExaone4_5Scenario, when, TestExaone4_5, skip_hf_inference，涉及 `TestExaone4_5Scenario, when, TestExaone4_5`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_exaone4_5.py` added +267/-0 (267 lines); hunks: -0,0 +1,267; symbols: Exaone4_5_VisionConfig, Exaone4_5Config, __init__, Exaone4_5InputProcessor
  - `tensorrt_llm/_torch/models/checkpoints/hf/exaone4_5_weight_mapper.py` added +38/-0 (38 lines); hunks: -0,0 +1,38; symbols: Exaone4_5HfWeightMapper, preprocess_weights
  - `tests/unittest/_torch/modeling/test_modeling_exaone4_5.py` added +255/-0 (255 lines); hunks: -0,0 +1,255; symbols: TestExaone4_5Scenario, when, TestExaone4_5, skip_hf_inference
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_exaone4_5.py
@@ -0,0 +1,267 @@
+# SPDX-License-Identifier: Apache-2.0
+# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+import copy
+from typing import List, Optional, Tuple, Union
+import torch
+from transformers import AutoConfig, AutoTokenizer, PretrainedConfig, PreTrainedModel
diff -- tensorrt_llm/_torch/models/checkpoints/hf/exaone4_5_weight_mapper.py
@@ -0,0 +1,38 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
diff -- tests/unittest/_torch/modeling/test_modeling_exaone4_5.py
@@ -0,0 +1,255 @@
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_exaone4_5.py` added +267/-0; `tensorrt_llm/_torch/models/checkpoints/hf/exaone4_5_weight_mapper.py` added +38/-0
  - tests: `tests/unittest/_torch/modeling/test_modeling_exaone4_5.py` added +255/-0
- 验证与风险: diff 自带测试面 `tests/integration/defs/accuracy/references/mmmu.yaml`, `tests/integration/defs/accuracy/test_llm_api_pytorch_multimodal.py`, `tests/integration/test_lists/qa/llm_function_core.txt`, `tests/unittest/_torch/modeling/test_modeling_exaone4_5.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #16992 - [https://nvbugs/6327149][fix] Handle EXAONE 4.5 33B memory constraints

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/16992
- 状态/时间: merged / 2026-08-06
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_exaone4_5.py`, `tests/unittest/_torch/modeling/test_modeling_exaone4_5.py`；关联提交 `a99395276420`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 4 个文件，+59/-2，可读 patch 116 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_exaone4_5.py` modified +27/-0 (27 lines); hunks: -10,6 +10,7; -36,6 +37,31; symbols: _normalize_exaone4_5_mtp_layer_types, __init__，涉及 `_normalize_exaone4_5_mtp_layer_types, __init__`；`tests/unittest/_torch/modeling/test_modeling_exaone4_5.py` modified +31/-0 (31 lines); hunks: -1,12 +1,14; -110,6 +112,35; symbols: test_exaone4_5_config_normalizes_trailing_mtp_layer_types, test_exaone4_5_config_preserves_unexpected_layer_type_mismatch, TestExaone4_5Scenario，涉及 `test_exaone4_5_config_normalizes_trailing_mtp_layer_types, test_exaone4_5_config_preserves_unexpected_layer_type_mismatch, TestExaone4_5Scenario`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_exaone4_5.py` modified +27/-0 (27 lines); hunks: -10,6 +10,7; -36,6 +37,31; symbols: _normalize_exaone4_5_mtp_layer_types, __init__
  - `tests/unittest/_torch/modeling/test_modeling_exaone4_5.py` modified +31/-0 (31 lines); hunks: -1,12 +1,14; -110,6 +112,35; symbols: test_exaone4_5_config_normalizes_trailing_mtp_layer_types, test_exaone4_5_config_preserves_unexpected_layer_type_mismatch, TestExaone4_5Scenario
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_exaone4_5.py
@@ -10,6 +10,7 @@
+from tensorrt_llm.logger import logger
@@ -36,6 +37,31 @@
+def _normalize_exaone4_5_mtp_layer_types(text_config: dict) -> None:
+    """Remove MTP-only entries from the base decoder layer layout."""
+    layer_types = text_config.get("layer_types")
+    num_hidden_layers = text_config.get("num_hidden_layers")
diff -- tests/unittest/_torch/modeling/test_modeling_exaone4_5.py
@@ -1,12 +1,14 @@
+import copy
+from huggingface_hub.errors import StrictDataclassClassValidationError
@@ -110,6 +112,35 @@
+def test_exaone4_5_config_normalizes_trailing_mtp_layer_types():
+    config = copy.deepcopy(EXAONE_4_5_TEST_CONFIG)
+    text_config = config["text_config"]
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_exaone4_5.py` modified +27/-0
  - tests: `tests/unittest/_torch/modeling/test_modeling_exaone4_5.py` modified +31/-0
- 验证与风险: diff 自带测试面 `tests/integration/defs/accuracy/test_llm_api_pytorch_multimodal.py`, `tests/integration/test_lists/waives.txt`, `tests/unittest/_torch/modeling/test_modeling_exaone4_5.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #19246 - [None][test] Add coverage for Exaone4ForCausalLM

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/19246
- 状态/时间: merged / 2026-09-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/unittest/_torch/modeling/test_modeling_exaone4_5.py`；关联提交 `2b0421eae607`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+100/-0，可读 patch 114 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/unittest/_torch/modeling/test_modeling_exaone4_5.py` modified +97/-0 (97 lines); hunks: -112,6 +112,103; symbols: test_exaone4_construction_and_forward, test_exaone4_5_config_normalizes_trailing_mtp_layer_types，涉及 `test_exaone4_construction_and_forward, test_exaone4_5_config_normalizes_trailing_mtp_layer_types`。
- 代码 diff 细节:
  - `tests/unittest/_torch/modeling/test_modeling_exaone4_5.py` modified +97/-0 (97 lines); hunks: -112,6 +112,103; symbols: test_exaone4_construction_and_forward, test_exaone4_5_config_normalizes_trailing_mtp_layer_types
- 关键代码摘录:

```diff
diff -- tests/unittest/_torch/modeling/test_modeling_exaone4_5.py
@@ -112,6 +112,103 @@
+@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
+@torch.inference_mode()
+def test_exaone4_construction_and_forward() -> None:
+    """Check EXAONE 4 construction, BF16 weights, and forward behavior."""
+    from transformers import Exaone4Config
+    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
```

- 提取文件（未人工审阅）:
  - tests: `tests/unittest/_torch/modeling/test_modeling_exaone4_5.py` modified +97/-0
- 验证与风险: diff 自带测试面 `tests/integration/test_lists/test-db/l0_h100.yml`, `tests/unittest/_torch/modeling/test_modeling_exaone4_5.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
