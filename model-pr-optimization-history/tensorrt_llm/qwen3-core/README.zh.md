# TensorRT-LLM Qwen3 Core 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `docs/source/deployment-guide/deployment-guide-for-qwen3-on-trtllm.md` | [#9488](https://github.com/NVIDIA/TensorRT-LLM/pull/9488) |
| `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_moe_weight_mapper.py` | [#10962](https://github.com/NVIDIA/TensorRT-LLM/pull/10962) |
| `tensorrt_llm/_torch/models/modeling_qwen3.py` | [#4010](https://github.com/NVIDIA/TensorRT-LLM/pull/4010), [#5879](https://github.com/NVIDIA/TensorRT-LLM/pull/5879), [#6785](https://github.com/NVIDIA/TensorRT-LLM/pull/6785), [#7616](https://github.com/NVIDIA/TensorRT-LLM/pull/7616), [#7618](https://github.com/NVIDIA/TensorRT-LLM/pull/7618), [#7765](https://github.com/NVIDIA/TensorRT-LLM/pull/7765), [#7780](https://github.com/NVIDIA/TensorRT-LLM/pull/7780), [#7892](https://github.com/NVIDIA/TensorRT-LLM/pull/7892), [#8030](https://github.com/NVIDIA/TensorRT-LLM/pull/8030), [#8087](https://github.com/NVIDIA/TensorRT-LLM/pull/8087), [#9060](https://github.com/NVIDIA/TensorRT-LLM/pull/9060), [#9689](https://github.com/NVIDIA/TensorRT-LLM/pull/9689), ... (13 total) |
| `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` | [#4010](https://github.com/NVIDIA/TensorRT-LLM/pull/4010), [#4058](https://github.com/NVIDIA/TensorRT-LLM/pull/4058), [#4141](https://github.com/NVIDIA/TensorRT-LLM/pull/4141), [#4304](https://github.com/NVIDIA/TensorRT-LLM/pull/4304), [#4530](https://github.com/NVIDIA/TensorRT-LLM/pull/4530), [#4575](https://github.com/NVIDIA/TensorRT-LLM/pull/4575), [#5206](https://github.com/NVIDIA/TensorRT-LLM/pull/5206), [#5369](https://github.com/NVIDIA/TensorRT-LLM/pull/5369), [#5459](https://github.com/NVIDIA/TensorRT-LLM/pull/5459), [#6199](https://github.com/NVIDIA/TensorRT-LLM/pull/6199), [#6235](https://github.com/NVIDIA/TensorRT-LLM/pull/6235), [#7443](https://github.com/NVIDIA/TensorRT-LLM/pull/7443), ... (17 total) |

## PR 覆盖总览

- git 追溯 PR 数: 28
- 原文档显式引用补充 PR 数: 0
- 当前文档总 PR 数: 28
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-05-01 | [#4010](https://github.com/NVIDIA/TensorRT-LLM/pull/4010) | merged | model: support Qwen3 | `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`, `tensorrt_llm/_torch/models/modeling_qwen3.py` |
| 2025-05-06 | [#4058](https://github.com/NVIDIA/TensorRT-LLM/pull/4058) | merged | Fix: fix bug of qwen3 moe | `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` |
| 2025-05-09 | [#4141](https://github.com/NVIDIA/TensorRT-LLM/pull/4141) | merged | [TRTLLM-5147][Qwen3] fix: fix bug of attention dp on qwen3_moe model | `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` |
| 2025-05-15 | [#4304](https://github.com/NVIDIA/TensorRT-LLM/pull/4304) | merged | Add allreduce and rmsnorm fusion for qwen3 | `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` |
| 2025-05-23 | [#4575](https://github.com/NVIDIA/TensorRT-LLM/pull/4575) | merged | [Fix][Qwen3] fix bug of qwen3 fp4 workflow with EP | `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` |
| 2025-05-23 | [#4530](https://github.com/NVIDIA/TensorRT-LLM/pull/4530) | merged | Qwen3 supports TRTLLM FP4 MoE backend | `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` |
| 2025-06-17 | [#5206](https://github.com/NVIDIA/TensorRT-LLM/pull/5206) | merged | [feat] Add EAGLE3 support for Qwen3 | `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` |
| 2025-06-25 | [#5369](https://github.com/NVIDIA/TensorRT-LLM/pull/5369) | merged | fix: fix bug of qwen3 + eagle3 + finalize_moe_fusion | `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` |
| 2025-06-30 | [#5459](https://github.com/NVIDIA/TensorRT-LLM/pull/5459) | merged | feat : support duplicate_kv_weight for qwen3 blockwise scale | `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` |
| 2025-07-18 | [#5879](https://github.com/NVIDIA/TensorRT-LLM/pull/5879) | merged | feat(eagle3):support qwen3 dense model | `tensorrt_llm/_torch/models/modeling_qwen3.py` |
| 2025-07-24 | [#6235](https://github.com/NVIDIA/TensorRT-LLM/pull/6235) | merged | [Fix][nvbug 5401163][nvbug 5404726][Qwen3] Fix bug of MoE on tp > 1 with trtllm moe backend | `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` |
| 2025-08-06 | [#6199](https://github.com/NVIDIA/TensorRT-LLM/pull/6199) | merged | Qwen3: Fix eagle hidden states | `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` |
| 2025-08-16 | [#6785](https://github.com/NVIDIA/TensorRT-LLM/pull/6785) | merged | [None][feat] Support Yarn on Qwen3 | `tensorrt_llm/_torch/models/modeling_qwen3.py` |
| 2025-09-10 | [#7616](https://github.com/NVIDIA/TensorRT-LLM/pull/7616) | merged | [https://nvbugs/5505402] [fix] Disable deep_gemm for Qwen3 QKNormRoPEAttention and Linear layers due to accuracy issues | `tensorrt_llm/_torch/models/modeling_qwen3.py` |
| 2025-09-16 | [#7618](https://github.com/NVIDIA/TensorRT-LLM/pull/7618) | merged | [None][feat] support attention dp for qwen3 dense model | `tensorrt_llm/_torch/models/modeling_qwen3.py` |
| 2025-09-16 | [#7724](https://github.com/NVIDIA/TensorRT-LLM/pull/7724) | merged | [https://nvbugs/5355219][fix] Fix trtllm moe backend test config and Qwen3 MoE multi node | `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` |
| 2025-09-16 | [#7765](https://github.com/NVIDIA/TensorRT-LLM/pull/7765) | merged | Revert "[None][feat] support attention dp for qwen3 dense model" | `tensorrt_llm/_torch/models/modeling_qwen3.py` |
| 2025-09-18 | [#7780](https://github.com/NVIDIA/TensorRT-LLM/pull/7780) | merged | [None][fix] Revert "Revert "[None][feat] support attention dp for qwen3 dense model"" | `tensorrt_llm/_torch/models/modeling_qwen3.py` |
| 2025-09-19 | [#7443](https://github.com/NVIDIA/TensorRT-LLM/pull/7443) | merged | [None][feat] Support EPLB in Qwen3 MoE | `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` |
| 2025-09-28 | [#8030](https://github.com/NVIDIA/TensorRT-LLM/pull/8030) | merged | [https://nvbugs/5461712] [fix] Use DG for Qwen3 Linear layers | `tensorrt_llm/_torch/models/modeling_qwen3.py` |
| 2025-09-29 | [#7892](https://github.com/NVIDIA/TensorRT-LLM/pull/7892) | merged | [None][feat] Support Qwen3 next | `tensorrt_llm/_torch/models/modeling_qwen3.py` |
| 2025-10-03 | [#8075](https://github.com/NVIDIA/TensorRT-LLM/pull/8075) | merged | [None][fix] Fix Qwen3 FP8 per-tensor when requesting TRTLLM-GEN MoE backend | `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` |
| 2025-10-14 | [#8087](https://github.com/NVIDIA/TensorRT-LLM/pull/8087) | merged | [None][fix] Disable DeepGEMM for Qwen3 MoE Attention layers | `tensorrt_llm/_torch/models/modeling_qwen3.py`, `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` |
| 2025-11-27 | [#9488](https://github.com/NVIDIA/TensorRT-LLM/pull/9488) | merged | [TRTLLM-9513][docs] Qwen3 deployment guide | `docs/source/deployment-guide/deployment-guide-for-qwen3-on-trtllm.md` |
| 2025-12-16 | [#9689](https://github.com/NVIDIA/TensorRT-LLM/pull/9689) | merged | [TRTLLM-8310][feat] Add Qwen3-VL-MoE | `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`, `tensorrt_llm/_torch/models/modeling_qwen3.py` |
| 2025-12-31 | [#9060](https://github.com/NVIDIA/TensorRT-LLM/pull/9060) | merged | [None][feat] support Qwen3-VL dense model in pytorch backend | `tensorrt_llm/_torch/models/modeling_qwen3.py` |
| 2026-01-28 | [#10962](https://github.com/NVIDIA/TensorRT-LLM/pull/10962) | merged | [https://nvbugs/5835925][fix] Add EPD disagg support for Qwen3 VL MoE | `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_moe_weight_mapper.py` |
| 2026-04-08 | [#12785](https://github.com/NVIDIA/TensorRT-LLM/pull/12785) | merged | [None][fix] Fix LoRA support for Qwen3 models | `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`, `tensorrt_llm/_torch/models/modeling_qwen3.py` |

## 逐 PR diff 审计卡

### PR #4010 - model: support Qwen3

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/4010
- 状态/时间: merged / 2025-05-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_qwen3.py`, `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`；关联提交 `129bf199807a`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 6 个文件，+596/-5，可读 patch 657 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` added +312/-0 (312 lines); hunks: -0,0 +1,312; symbols: Qwen3MoE, __init__, forward, Qwen3MoEAttention，涉及 `Qwen3MoE, __init__, forward`；`tensorrt_llm/_torch/models/modeling_qwen3.py` added +244/-0 (244 lines); hunks: -0,0 +1,244; symbols: Qwen3Attention, __init__, Qwen3DecoderLayer, forward，涉及 `Qwen3Attention, __init__, Qwen3DecoderLayer`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` added +312/-0 (312 lines); hunks: -0,0 +1,312; symbols: Qwen3MoE, __init__, forward, Qwen3MoEAttention
  - `tensorrt_llm/_torch/models/modeling_qwen3.py` added +244/-0 (244 lines); hunks: -0,0 +1,244; symbols: Qwen3Attention, __init__, Qwen3DecoderLayer, forward
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3_moe.py
@@ -0,0 +1,312 @@
+from typing import Dict, Optional
+import torch
+from torch import nn
+from tqdm import tqdm
+from transformers import Qwen3MoeConfig
+from tensorrt_llm.functional import PositionEmbeddingType
diff -- tensorrt_llm/_torch/models/modeling_qwen3.py
@@ -0,0 +1,244 @@
+from typing import Optional, Tuple
+import torch
+from torch import nn
+from transformers import Qwen3Config
+from tensorrt_llm.functional import PositionEmbeddingType
+from ..attention_backend import AttentionMetadata
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` added +312/-0; `tensorrt_llm/_torch/models/modeling_qwen3.py` added +244/-0
- 验证与风险: runtime 路径改动集中在 `tensorrt_llm/_torch/models/__init__.py`, `tensorrt_llm/_torch/models/modeling_qwen3.py`, `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #4058 - Fix: fix bug of qwen3 moe

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/4058
- 状态/时间: merged / 2025-05-06
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`；关联提交 `e053cb651bac`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 5 个文件，+9/-11，可读 patch 57 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +2/-2 (4 lines); hunks: -13,7 +13,7; -48,7 +48,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +2/-2 (4 lines); hunks: -13,7 +13,7; -48,7 +48,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3_moe.py
@@ -13,7 +13,7 @@
-from ..modules.fused_moe import DefaultMoeRoutingMethod, FusedMoE
+from ..modules.fused_moe import FusedMoE, RenormalizeMoeRoutingMethod
@@ -48,7 +48,7 @@ def __init__(
-            routing_method=DefaultMoeRoutingMethod(top_k=self.top_k),
+            routing_method=RenormalizeMoeRoutingMethod(top_k=self.top_k),
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +2/-2
- 验证与风险: diff 自带测试面 `tests/integration/defs/accuracy/references/cnn_dailymail.yaml`, `tests/integration/defs/accuracy/references/gsm8k.yaml`, `tests/integration/defs/accuracy/references/mmlu.yaml`, `tests/integration/defs/accuracy/test_llm_api_pytorch.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #4141 - [TRTLLM-5147][Qwen3] fix: fix bug of attention dp on qwen3_moe model

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/4141
- 状态/时间: merged / 2025-05-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`；关联提交 `700d09ab6540`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+25/-11，可读 patch 64 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +25/-11 (36 lines); hunks: -73,8 +73,10 @@ def forward(; -183,14 +185,23 @@ def __init__(self, model_config: ModelConfig[Qwen3MoeConfi...; symbols: forward, __init__, load_weights, filter_weights，涉及 `forward, __init__, load_weights`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +25/-11 (36 lines); hunks: -73,8 +73,10 @@ def forward(; -183,14 +185,23 @@ def __init__(self, model_config: ModelConfig[Qwen3MoeConfi...; symbols: forward, __init__, load_weights, filter_weights
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3_moe.py
@@ -73,8 +73,10 @@ def forward(
-        final_hidden_states = self.experts(hidden_states, router_logits,
-                                           all_rank_num_tokens)
+        final_hidden_states = self.experts(
+            hidden_states,
+            router_logits,
+            all_rank_num_tokens=all_rank_num_tokens)
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +25/-11
- 验证与风险: runtime 路径改动集中在 `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #4304 - Add allreduce and rmsnorm fusion for qwen3

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/4304
- 状态/时间: merged / 2025-05-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`；关联提交 `f0ca60a95da5`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+75/-13，可读 patch 183 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +75/-13 (88 lines); hunks: -1,3 +1,4; -6,15 +7,18; symbols: Qwen3MoE, __init__, forward，涉及 `Qwen3MoE, __init__, forward`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +75/-13 (88 lines); hunks: -1,3 +1,4; -6,15 +7,18; symbols: Qwen3MoE, __init__, forward
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3_moe.py
@@ -1,3 +1,4 @@
+import os
@@ -6,15 +7,18 @@
+from ..distributed import AllReduce, AllReduceFusionOp, AllReduceParams
+from ..models.modeling_utils import MissingLayer
-                             duplicate_kv_weight, register_auto_model)
+                             EagerFusionConfig, duplicate_kv_weight,
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +75/-13
- 验证与风险: runtime 路径改动集中在 `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #4575 - [Fix][Qwen3] fix bug of qwen3 fp4 workflow with EP

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/4575
- 状态/时间: merged / 2025-05-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`；关联提交 `d69c6622151f`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+48/-4，可读 patch 98 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +48/-4 (52 lines); hunks: -6,14 +6,18; -32,12 +36,15 @@ def __init__(; symbols: __init__, should_enable_alltoall, forward，涉及 `__init__, should_enable_alltoall, forward`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +48/-4 (52 lines); hunks: -6,14 +6,18; -32,12 +36,15 @@ def __init__(; symbols: __init__, should_enable_alltoall, forward
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3_moe.py
@@ -6,14 +6,18 @@
+from tensorrt_llm._mnnvl_utils import MnnvlMemory
-from ..distributed import AllReduce, AllReduceFusionOp, AllReduceParams
+from ..distributed import (AllReduce, AllReduceFusionOp, AllReduceParams,
+                           allgather)
+from ..utils import disable_fp4_allgather
@@ -32,12 +36,15 @@ def __init__(
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +48/-4
- 验证与风险: runtime 路径改动集中在 `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #4530 - Qwen3 supports TRTLLM FP4 MoE backend

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/4530
- 状态/时间: merged / 2025-05-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`；关联提交 `bbea2647b1ba`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 15 个文件，+1939/-166，可读 patch 2663 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +62/-11 (73 lines); hunks: -1,5 +1,5; -11,21 +11,69; symbols: Qwen3Gate, __init__, forward, load_weights，涉及 `Qwen3Gate, __init__, forward`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +62/-11 (73 lines); hunks: -1,5 +1,5; -11,21 +11,69; symbols: Qwen3Gate, __init__, forward, load_weights
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3_moe.py
@@ -1,5 +1,5 @@
-from typing import Dict, Optional
+from typing import Dict, List, Optional
@@ -11,21 +11,69 @@
-from ..modules.fused_moe import FusedMoE, RenormalizeMoeRoutingMethod
-from ..modules.linear import Linear, TensorParallelMode
+from ..modules.fused_moe import (BaseMoeRoutingMethod, FusedMoE,
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +62/-11
- 验证与风险: diff 自带测试面 `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/integration/test_lists/test-db/l0_b200.yml`, `tests/unittest/_torch/thop/test_moe.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #5206 - [feat] Add EAGLE3 support for Qwen3

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/5206
- 状态/时间: merged / 2025-06-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`；关联提交 `498fadceb4eb`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+16/-10，可读 patch 81 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +16/-10 (26 lines); hunks: -18,11 +18,13; -203,6 +205,7 @@ def forward(; symbols: Qwen3Gate, forward，涉及 `Qwen3Gate, forward`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +16/-10 (26 lines); hunks: -18,11 +18,13; -203,6 +205,7 @@ def forward(; symbols: Qwen3Gate, forward
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3_moe.py
@@ -18,11 +18,13 @@
+from ..speculative import SpecMetadata
-from .modeling_utils import (DecoderModel, DecoderModelForCausalLM,
-                             EagerFusionConfig, duplicate_kv_weight,
-                             filter_weights, register_auto_model)
+from .modeling_speculative import SpecDecOneEngineForCausalLM
+from .modeling_utils import (DecoderModel, EagerFusionConfig,
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +16/-10
- 验证与风险: runtime 路径改动集中在 `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #5369 - fix: fix bug of qwen3 + eagle3 + finalize_moe_fusion

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/5369
- 状态/时间: merged / 2025-06-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`；关联提交 `73ba4fc32057`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+11/-3，可读 patch 31 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +11/-3 (14 lines); hunks: -263,11 +263,11 @@ def forward(; -296,7 +296,15 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +11/-3 (14 lines); hunks: -263,11 +263,11 @@ def forward(; -296,7 +296,15 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3_moe.py
@@ -263,11 +263,11 @@ def forward(
-        if spec_metadata:
-            spec_metadata.maybe_capture_hidden_states(self.layer_idx,
-                                                      hidden_states, residual)
+                if spec_metadata:
+                    spec_metadata.maybe_capture_hidden_states(
+                        self.layer_idx, hidden_states, residual)
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +11/-3
- 验证与风险: runtime 路径改动集中在 `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #5459 - feat : support duplicate_kv_weight for qwen3 blockwise scale

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/5459
- 状态/时间: merged / 2025-06-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`；关联提交 `852b79053d51`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 6 个文件，+36/-30，可读 patch 181 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +5/-4 (9 lines); hunks: -394,9 +394,7 @@ def load_weights(self, weights: Dict):; -419,11 +417,14 @@ def load_weights(self, weights: Dict):; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +5/-4 (9 lines); hunks: -394,9 +394,7 @@ def load_weights(self, weights: Dict):; -419,11 +417,14 @@ def load_weights(self, weights: Dict):; symbols: load_weights
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3_moe.py
@@ -394,9 +394,7 @@ def load_weights(self, weights: Dict):
-        head_dim = getattr(
-            self.config, "head_dim",
-            self.config.hidden_size // self.config.num_attention_heads)
+        num_kv_heads = self.config.num_key_value_heads
@@ -419,11 +417,14 @@ def load_weights(self, weights: Dict):
+                        if module.quant_config.quant_mode.has_fp8_block_scales(
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +5/-4
- 验证与风险: runtime 路径改动集中在 `tensorrt_llm/_torch/models/modeling_gemma3.py`, `tensorrt_llm/_torch/models/modeling_mllama.py`, `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #5879 - feat(eagle3):support qwen3 dense model

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/5879
- 状态/时间: merged / 2025-07-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_qwen3.py`；关联提交 `28858c871143`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 4 个文件，+39/-32，可读 patch 139 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +12/-32 (44 lines); hunks: -16,8 +16,9; -148,6 +149,7 @@ def forward(; symbols: Qwen3Attention, forward，涉及 `Qwen3Attention, forward`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +12/-32 (44 lines); hunks: -16,8 +16,9; -148,6 +149,7 @@ def forward(; symbols: Qwen3Attention, forward
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3.py
@@ -16,8 +16,9 @@
-from .modeling_utils import (DecoderModel, DecoderModelForCausalLM,
-                             register_auto_model)
+from ..speculative import SpecMetadata
+from .modeling_speculative import SpecDecOneEngineForCausalLM
+from .modeling_utils import DecoderModel, register_auto_model
@@ -148,6 +149,7 @@ def forward(
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +12/-32
- 验证与风险: diff 自带测试面 `tests/integration/defs/accuracy/references/mmlu.yaml`, `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/integration/test_lists/test-db/l0_h100.yml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #6235 - [Fix][nvbug 5401163][nvbug 5404726][Qwen3] Fix bug of MoE on tp > 1 with trtllm moe backend

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/6235
- 状态/时间: merged / 2025-07-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`；关联提交 `7b6aadc80056`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 5 个文件，+36/-8，可读 patch 100 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +8/-0 (8 lines); hunks: -309,6 +309,13 @@ def __init__(self, model_config: ModelConfig[Qwen3MoeConfig]):; -381,6 +388,7 @@ def __init__(; symbols: __init__, load_weights，涉及 `__init__, load_weights`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +8/-0 (8 lines); hunks: -309,6 +309,13 @@ def __init__(self, model_config: ModelConfig[Qwen3MoeConfig]):; -381,6 +388,7 @@ def __init__(; symbols: __init__, load_weights
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3_moe.py
@@ -309,6 +309,13 @@ def __init__(self, model_config: ModelConfig[Qwen3MoeConfig]):
+        self.preload_weight_modules = []
+        if config.moe_backend == "TRTLLM":
+            self.preload_weight_modules = [
+                "experts",
+                "routing_method",
+                "all_reduce",
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +8/-0
- 验证与风险: diff 自带测试面 `tests/integration/defs/accuracy/references/gsm8k.yaml`, `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/integration/test_lists/waives.txt`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #6199 - Qwen3: Fix eagle hidden states

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/6199
- 状态/时间: merged / 2025-08-06
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`；关联提交 `7e0158b58334`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+4/-9，可读 patch 35 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +4/-9 (13 lines); hunks: -214,7 +214,9 @@ def forward(; -257,9 +259,6 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +4/-9 (13 lines); hunks: -214,7 +214,9 @@ def forward(; -257,9 +259,6 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3_moe.py
@@ -214,7 +214,9 @@ def forward(
+        if spec_metadata is not None and spec_metadata.is_layer_capture(
+                self.layer_idx):
+            self.fusion_config.POST_MOE_FUSION = False
@@ -257,9 +259,6 @@ def forward(
-                if spec_metadata:
-                    spec_metadata.maybe_capture_hidden_states(
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +4/-9
- 验证与风险: runtime 路径改动集中在 `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #6785 - [None][feat] Support Yarn on Qwen3

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/6785
- 状态/时间: merged / 2025-08-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_qwen3.py`；关联提交 `85cbd0263be9`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 5 个文件，+208/-31，可读 patch 360 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +121/-5 (126 lines); hunks: -1,8 +1,9; -21,6 +22,111; symbols: compute_yarn_parameters, get_mscale, find_correction_dim, find_correction_range，涉及 `compute_yarn_parameters, get_mscale, find_correction_dim`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +121/-5 (126 lines); hunks: -1,8 +1,9; -21,6 +22,111; symbols: compute_yarn_parameters, get_mscale, find_correction_dim, find_correction_range
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3.py
@@ -1,8 +1,9 @@
+import math
-from transformers import Qwen3Config
+from transformers import PretrainedConfig, Qwen3Config
@@ -21,6 +22,111 @@
+# Move out from this class
+def compute_yarn_parameters(
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +121/-5
- 验证与风险: diff 自带测试面 `tests/unittest/_torch/thop/test_fused_qk_norm_rope.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #7616 - [https://nvbugs/5505402] [fix] Disable deep_gemm for Qwen3 QKNormRoPEAttention and Linear layers due to accuracy issues

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/7616
- 状态/时间: merged / 2025-09-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_qwen3.py`；关联提交 `fc9d426589ad`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 7 个文件，+44/-21，可读 patch 190 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +9/-2 (11 lines); hunks: -48,8 +48,9 @@ def __init__(; -63,6 +64,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +9/-2 (11 lines); hunks: -48,8 +48,9 @@ def __init__(; -63,6 +64,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3.py
@@ -48,8 +48,9 @@ def __init__(
-        # Qwen3 has accuracy issues with deep_gemm (see: https://nvbugspro.nvidia.com/bug/5461712)
-        # TODO: Consider adding disable_deep_gemm support to QKNormRoPEAttention if accuracy still remains
+        # Qwen3 has accuracy issues with deep_gemm (see: https://nvbugspro.nvidia.com/bug/5461712
+        # and https://nvbugspro.nvidia.com/bug/5505402)
+        disable_deep_gemm = True
@@ -63,6 +64,7 @@ def __init__(
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +9/-2
- 验证与风险: diff 自带测试面 `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/integration/test_lists/test-db/l0_b200.yml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #7618 - [None][feat] support attention dp for qwen3 dense model

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/7618
- 状态/时间: merged / 2025-09-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_qwen3.py`；关联提交 `96f11b10ae83`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+16/-1，可读 patch 57 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +16/-1 (17 lines); hunks: -8,6 +8,7; -80,12 +81,15 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +16/-1 (17 lines); hunks: -8,6 +8,7; -80,12 +81,15 @@ def __init__(; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3.py
@@ -8,6 +8,7 @@
+from ..distributed import AllReduceParams
@@ -80,12 +81,15 @@ def __init__(
+        self.mapping = model_config.mapping
+        self.enable_attention_dp = self.mapping.enable_attention_dp
+            overridden_tp_size=1 if self.enable_attention_dp else None,
@@ -95,6 +99,8 @@ def __init__(
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +16/-1
- 验证与风险: runtime 路径改动集中在 `tensorrt_llm/_torch/models/modeling_qwen3.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #7724 - [https://nvbugs/5355219][fix] Fix trtllm moe backend test config and Qwen3 MoE multi node

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/7724
- 状态/时间: merged / 2025-09-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`；关联提交 `f9c9c3f50a64`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 6 个文件，+17/-7，可读 patch 85 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +8/-5 (13 lines); hunks: -5,6 +5,7; -187,6 +188,8 @@ def __init__(self, model_config: ModelConfig[Qwen3MoeConfig],; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +8/-5 (13 lines); hunks: -5,6 +5,7; -187,6 +188,8 @@ def __init__(self, model_config: ModelConfig[Qwen3MoeConfig],; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3_moe.py
@@ -5,6 +5,7 @@
+from tensorrt_llm._ipc_utils import can_access_peer
@@ -187,6 +188,8 @@ def __init__(self, model_config: ModelConfig[Qwen3MoeConfig],
+        self.is_p2p_supported = can_access_peer(model_config.mapping)
@@ -242,11 +245,11 @@ def forward(
-        do_finalize = not (hidden_states.shape[0]
-                           <= self.moe_allreduce.max_token
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +8/-5
- 验证与风险: diff 自带测试面 `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/integration/test_lists/qa/llm_function_full.txt`, `tests/integration/test_lists/test-db/l0_gb200_multi_nodes.yml`, `tests/integration/test_lists/waives.txt`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #7765 - Revert "[None][feat] support attention dp for qwen3 dense model"

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/7765
- 状态/时间: merged / 2025-09-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_qwen3.py`；关联提交 `8226ef23dc20`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+1/-16，可读 patch 58 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +1/-16 (17 lines); hunks: -8,7 +8,6; -83,8 +82,6 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +1/-16 (17 lines); hunks: -8,7 +8,6; -83,8 +82,6 @@ def __init__(; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3.py
@@ -8,7 +8,6 @@
-from ..distributed import AllReduceParams
@@ -83,8 +82,6 @@ def __init__(
-        self.mapping = model_config.mapping
-        self.enable_attention_dp = self.mapping.enable_attention_dp
@@ -95,7 +92,6 @@ def __init__(
-            overridden_tp_size=1 if self.enable_attention_dp else None,
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +1/-16
- 验证与风险: runtime 路径改动集中在 `tensorrt_llm/_torch/models/modeling_qwen3.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #7780 - [None][fix] Revert "Revert "[None][feat] support attention dp for qwen3 dense model""

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/7780
- 状态/时间: merged / 2025-09-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_qwen3.py`；关联提交 `c65457db8a09`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+15/-1，可读 patch 57 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +15/-1 (16 lines); hunks: -8,6 +8,7; -82,6 +83,8 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +15/-1 (16 lines); hunks: -8,6 +8,7; -82,6 +83,8 @@ def __init__(; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3.py
@@ -8,6 +8,7 @@
+from ..distributed import AllReduceParams
@@ -82,6 +83,8 @@ def __init__(
+        self.mapping = model_config.mapping
+        self.enable_attention_dp = self.mapping.enable_attention_dp
@@ -92,6 +95,7 @@ def __init__(
+            overridden_tp_size=1 if self.enable_attention_dp else None,
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +15/-1
- 验证与风险: runtime 路径改动集中在 `tensorrt_llm/_torch/models/modeling_qwen3.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #7443 - [None][feat] Support EPLB in Qwen3 MoE

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/7443
- 状态/时间: merged / 2025-09-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`；关联提交 `0e72e8f7e655`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+10/-6，可读 patch 58 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +10/-6 (16 lines); hunks: -78,7 +78,7 @@ class Qwen3MoE(nn.Module):; -108,7 +108,7 @@ def __init__(; symbols: Qwen3MoE, __init__, forward, Qwen3MoEDecoderLayer，涉及 `Qwen3MoE, __init__, forward`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +10/-6 (16 lines); hunks: -78,7 +78,7 @@ class Qwen3MoE(nn.Module):; -108,7 +108,7 @@ def __init__(; symbols: Qwen3MoE, __init__, forward, Qwen3MoEDecoderLayer
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3_moe.py
@@ -78,7 +78,7 @@ class Qwen3MoE(nn.Module):
-        aux_stream: torch.cuda.Stream,
+        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
@@ -108,7 +108,7 @@ def __init__(
-            aux_stream_dict={AuxStreamType.MoeChunkingOverlap: aux_stream},
+            aux_stream_dict=aux_stream_dict,
@@ -160,7 +160,8 @@ def forward(
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +10/-6
- 验证与风险: runtime 路径改动集中在 `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #8030 - [https://nvbugs/5461712] [fix] Use DG for Qwen3 Linear layers

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/8030
- 状态/时间: merged / 2025-09-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_qwen3.py`；关联提交 `77b68d9d7d59`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+25/-10，可读 patch 77 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +25/-10 (35 lines); hunks: -2,9 +2,13; -49,10 +53,6 @@ def __init__(; symbols: __init__, post_load_weights，涉及 `__init__, post_load_weights`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +25/-10 (35 lines); hunks: -2,9 +2,13; -49,10 +53,6 @@ def __init__(; symbols: __init__, post_load_weights
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3.py
@@ -2,9 +2,13 @@
+from tqdm import tqdm
+from tensorrt_llm._utils import is_sm_100f
+from tensorrt_llm.quantization.utils.fp8_utils import (
+    resmooth_to_fp8_e8m0, transform_sf_into_required_layout)
@@ -49,10 +53,6 @@ def __init__(
-        # Qwen3 has accuracy issues with deep_gemm (see: https://nvbugspro.nvidia.com/bug/5461712
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +25/-10
- 验证与风险: runtime 路径改动集中在 `tensorrt_llm/_torch/models/modeling_qwen3.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #7892 - [None][feat] Support Qwen3 next

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/7892
- 状态/时间: merged / 2025-09-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_qwen3.py`；关联提交 `38d6e4e60b1a`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 30 个文件，+5286/-39，可读 patch 5588 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +8/-2 (10 lines); hunks: -32,8 +32,12 @@ def __init__(; -58,13 +62,15 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +8/-2 (10 lines); hunks: -32,8 +32,12 @@ def __init__(; -58,13 +62,15 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3.py
@@ -32,8 +32,12 @@ def __init__(
+        attn_output_gate: bool = False,
+        use_gemma_rms_norm: bool = False,
+        self.pretrained_config = config
+        self.attn_output_gate = attn_output_gate
@@ -58,13 +62,15 @@ def __init__(
-            bias=config.attention_bias,
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +8/-2
- 验证与风险: runtime 路径改动集中在 `cpp/tensorrt_llm/kernels/fusedQKNormRopeKernel.cu`, `tensorrt_llm/_torch/custom_ops/__init__.py`, `tensorrt_llm/_torch/custom_ops/flashinfer_custom_ops.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #8075 - [None][fix] Fix Qwen3 FP8 per-tensor when requesting TRTLLM-GEN MoE backend

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/8075
- 状态/时间: merged / 2025-10-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`；关联提交 `9db436690325`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+15/-19，可读 patch 93 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +14/-15 (29 lines); hunks: -1,5 +1,5; -15,10 +15,12; symbols: __init__, load_weights, routing_method，涉及 `__init__, load_weights, routing_method`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +14/-15 (29 lines); hunks: -1,5 +1,5; -15,10 +15,12; symbols: __init__, load_weights, routing_method
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3_moe.py
@@ -1,5 +1,5 @@
-from typing import Dict, List, Optional
+from typing import Dict, List, Optional, Type
@@ -15,10 +15,12 @@
-from ..modules.fused_moe import (BaseMoeRoutingMethod,
+from ..modules.fused_moe import (BaseMoeRoutingMethod, CutlassFusedMoE,
-                                 RoutingMethodType, create_moe)
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +14/-15
- 验证与风险: runtime 路径改动集中在 `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`, `tensorrt_llm/_torch/modules/fused_moe/create_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #8087 - [None][fix] Disable DeepGEMM for Qwen3 MoE Attention layers

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/8087
- 状态/时间: merged / 2025-10-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_qwen3.py`, `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`；关联提交 `9bc055faf1e5`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+3/-0，可读 patch 24 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +2/-0 (2 lines); hunks: -34,6 +34,7 @@ def __init__(; -71,6 +72,7 @@ def __init__(; symbols: __init__，涉及 `__init__`；`tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +1/-0 (1 lines); hunks: -168,6 +168,7 @@ def __init__(self, model_config: ModelConfig[Qwen3MoeConfig],; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +2/-0 (2 lines); hunks: -34,6 +34,7 @@ def __init__(; -71,6 +72,7 @@ def __init__(; symbols: __init__
  - `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +1/-0 (1 lines); hunks: -168,6 +168,7 @@ def __init__(self, model_config: ModelConfig[Qwen3MoeConfig],; symbols: __init__
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3.py
@@ -34,6 +34,7 @@ def __init__(
+        disable_deep_gemm: bool = False,
@@ -71,6 +72,7 @@ def __init__(
+            disable_deep_gemm=disable_deep_gemm,
diff -- tensorrt_llm/_torch/models/modeling_qwen3_moe.py
@@ -168,6 +168,7 @@ def __init__(self, model_config: ModelConfig[Qwen3MoeConfig],
+            disable_deep_gemm=True,
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +2/-0; `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +1/-0
- 验证与风险: runtime 路径改动集中在 `tensorrt_llm/_torch/models/modeling_qwen3.py`, `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #9488 - [TRTLLM-9513][docs] Qwen3 deployment guide

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/9488
- 状态/时间: merged / 2025-11-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `docs/source/deployment-guide/deployment-guide-for-qwen3-on-trtllm.md`；关联提交 `5425d9675738`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+257/-0，可读 patch 263 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `docs/source/deployment-guide/deployment-guide-for-qwen3-on-trtllm.md` added +256/-0 (256 lines); hunks: -0,0 +1,256。
- 代码 diff 细节:
  - `docs/source/deployment-guide/deployment-guide-for-qwen3-on-trtllm.md` added +256/-0 (256 lines); hunks: -0,0 +1,256
- 关键代码摘录:

```diff
diff -- docs/source/deployment-guide/deployment-guide-for-qwen3-on-trtllm.md
@@ -0,0 +1,256 @@
+# Deployment Guide for Qwen3 on TensorRT LLM - Blackwell & Hopper Hardware
+## Introduction
+This is a functional quick-start guide for running the Qwen3 model on TensorRT LLM. It focuses on a working setup with recommended defaults. Additional performance optimizations a
+## Prerequisites
+* GPU: NVIDIA Blackwell or Hopper Architecture
+* OS: Linux
```

- 提取文件（未人工审阅）:
  - docs: `docs/source/deployment-guide/deployment-guide-for-qwen3-on-trtllm.md` added +256/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs/source/deployment-guide/deployment-guide-for-qwen3-on-trtllm.md`, `docs/source/deployment-guide/index.rst`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #9689 - [TRTLLM-8310][feat] Add Qwen3-VL-MoE

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/9689
- 状态/时间: merged / 2025-12-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_qwen3.py`, `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`；关联提交 `8ba8699f66b6`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 31 个文件，+1630/-160，可读 patch 2437 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +20/-6 (26 lines); hunks: -18,7 +18,7; -114,6 +114,7 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`；`tensorrt_llm/_torch/models/modeling_qwen3.py` modified +6/-1 (7 lines); hunks: -48,7 +48,11 @@ def __init__(; -64,6 +68,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +20/-6 (26 lines); hunks: -18,7 +18,7; -114,6 +114,7 @@ def __init__(; symbols: __init__, forward
  - `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +6/-1 (7 lines); hunks: -48,7 +48,11 @@ def __init__(; -64,6 +68,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3_moe.py
@@ -18,7 +18,7 @@
-from ..modules.fused_moe.interface import MoE
+from ..modules.fused_moe.interface import MoE, MoEWeightLoadingMode
@@ -114,6 +114,7 @@ def __init__(
+        self.weight_loading_mode = MoEWeightLoadingMode.FUSED_GATE_UP_PROJ if config.model_type == "qwen3_vl_moe_text" else MoEWeightLoadingMode.VANILLA
@@ -124,6 +125,7 @@ def __init__(
+            weight_loading_mode=self.weight_loading_mode,
diff -- tensorrt_llm/_torch/models/modeling_qwen3.py
@@ -48,7 +48,11 @@ def __init__(
-            )
+                mrope_section=config.rope_scaling.get("mrope_section", None),
+                mrope_interleaved=config.rope_scaling.get(
+                    "mrope_interleaved", False))
+            if config.rope_scaling.get("mrope_interleaved", False):
+                fuse_qk_norm_rope = False
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +20/-6; `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +6/-1
- 验证与风险: diff 自带测试面 `tests/integration/defs/accuracy/references/mmmu.yaml`, `tests/integration/defs/accuracy/test_llm_api_pytorch_multimodal.py`, `tests/integration/test_lists/qa/llm_function_core.txt`, `tests/integration/test_lists/test-db/l0_l40s.yml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #9060 - [None][feat] support Qwen3-VL dense model in pytorch backend

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/9060
- 状态/时间: merged / 2025-12-31
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_qwen3.py`；关联提交 `73870ae4ad12`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 6 个文件，+360/-23，可读 patch 479 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +11/-2 (13 lines); hunks: -121,6 +121,8 @@ def forward(; -137,6 +139,7 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +11/-2 (13 lines); hunks: -121,6 +121,8 @@ def forward(; -137,6 +139,7 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3.py
@@ -121,6 +121,8 @@ def forward(
+        mrope_config: Optional[dict] = None,
+        deepstack_embeds: Optional[list[torch.Tensor]] = None,
@@ -137,6 +139,7 @@ def forward(
+            mrope_config=mrope_config,
@@ -150,6 +153,9 @@ def forward(
+        if deepstack_embeds is not None and self.layer_idx in range(
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +11/-2
- 验证与风险: diff 自带测试面 `tests/unittest/_torch/modeling/test_modeling_qwen3vl.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #10962 - [https://nvbugs/5835925][fix] Add EPD disagg support for Qwen3 VL MoE

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/10962
- 状态/时间: merged / 2026-01-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_moe_weight_mapper.py`；关联提交 `abb8106c016a`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 4 个文件，+134/-13，可读 patch 268 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_moe_weight_mapper.py` modified +8/-3 (11 lines); hunks: -16,9 +16,6 @@ def init_model_and_config(self, model: Union[nn.Module,; -49,3 +46,11 @@ def _duplicate_kv_weights(self, module: nn.Module, new_name:...; symbols: init_model_and_config, should_skip_module, _duplicate_kv_weights, _num_kv_heads，涉及 `init_model_and_config, should_skip_module, _duplicate_kv_weights`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_moe_weight_mapper.py` modified +8/-3 (11 lines); hunks: -16,9 +16,6 @@ def init_model_and_config(self, model: Union[nn.Module,; -49,3 +46,11 @@ def _duplicate_kv_weights(self, module: nn.Module, new_name:...; symbols: init_model_and_config, should_skip_module, _duplicate_kv_weights, _num_kv_heads
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/checkpoints/hf/qwen3_moe_weight_mapper.py
@@ -16,9 +16,6 @@ def init_model_and_config(self, model: Union[nn.Module,
-        self._num_kv_heads = model.config.num_key_value_heads if hasattr(
-            model.config, 'num_key_value_heads'
-        ) and model.config.num_key_value_heads is not None else model.config.num_attention_heads
@@ -49,3 +46,11 @@ def _duplicate_kv_weights(self, module: nn.Module, new_name: str,
+    @property
+    def _num_kv_heads(self) -> int:
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_moe_weight_mapper.py` modified +8/-3
- 验证与风险: diff 自带测试面 `tests/unittest/_torch/multimodal/test_mm_encoder_standalone.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #12785 - [None][fix] Fix LoRA support for Qwen3 models

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/12785
- 状态/时间: merged / 2026-04-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_qwen3.py`, `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`；关联提交 `ae86b91e12b0`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 3 个文件，+204/-1，可读 patch 220 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +2/-1 (3 lines); hunks: -398,7 +398,8 @@ def forward(; symbols: forward，涉及 `forward`；`tensorrt_llm/_torch/models/modeling_qwen3.py` modified +1/-0 (1 lines); hunks: -121,6 +121,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +2/-1 (3 lines); hunks: -398,7 +398,8 @@ def forward(; symbols: forward
  - `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +1/-0 (1 lines); hunks: -121,6 +121,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3_moe.py
@@ -398,7 +398,8 @@ def forward(
-                deepstack_embeds=deepstack_embeds)
+                deepstack_embeds=deepstack_embeds,
+                **kwargs)
diff -- tensorrt_llm/_torch/models/modeling_qwen3.py
@@ -121,6 +121,7 @@ def __init__(
+            layer_idx=layer_idx,
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` modified +2/-1; `tensorrt_llm/_torch/models/modeling_qwen3.py` modified +1/-0
- 验证与风险: diff 自带测试面 `tests/unittest/_torch/modules/tests_lora_modules/test_qwen3_sanity.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
