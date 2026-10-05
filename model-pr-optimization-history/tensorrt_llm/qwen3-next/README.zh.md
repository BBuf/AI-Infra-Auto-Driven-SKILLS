# TensorRT-LLM Qwen3 Next 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `examples/configs/curated/qwen3-next.yaml` | 无直接 PR 号提交 |
| `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_next_weight_mapper.py` | [#7892](https://github.com/NVIDIA/TensorRT-LLM/pull/7892), [#10218](https://github.com/NVIDIA/TensorRT-LLM/pull/10218), [#11370](https://github.com/NVIDIA/TensorRT-LLM/pull/11370), [#16314](https://github.com/NVIDIA/TensorRT-LLM/pull/16314) |
| `tensorrt_llm/_torch/models/modeling_qwen3_next.py` | [#7892](https://github.com/NVIDIA/TensorRT-LLM/pull/7892), [#8064](https://github.com/NVIDIA/TensorRT-LLM/pull/8064), [#8902](https://github.com/NVIDIA/TensorRT-LLM/pull/8902), [#9691](https://github.com/NVIDIA/TensorRT-LLM/pull/9691), [#10218](https://github.com/NVIDIA/TensorRT-LLM/pull/10218), [#10228](https://github.com/NVIDIA/TensorRT-LLM/pull/10228), [#11370](https://github.com/NVIDIA/TensorRT-LLM/pull/11370), [#15194](https://github.com/NVIDIA/TensorRT-LLM/pull/15194) |
| `tests/unittest/_torch/models/test_qwen3_next_eager_fusion.py` | [#15194](https://github.com/NVIDIA/TensorRT-LLM/pull/15194) |
| `tests/unittest/_torch/models/test_qwen3_next_moe_quant.py` | 无直接 PR 号提交 |

## PR 覆盖总览

- git 追溯 PR 数: 9
- 原文档显式引用补充 PR 数: 0
- 当前文档总 PR 数: 9
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-09-29 | [#7892](https://github.com/NVIDIA/TensorRT-LLM/pull/7892) | merged | [None][feat] Support Qwen3 next | `tensorrt_llm/_torch/models/modeling_qwen3_next.py`, `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_next_weight_mapper.py` |
| 2025-09-30 | [#8064](https://github.com/NVIDIA/TensorRT-LLM/pull/8064) | merged | [None][chore] Refine qwen3-next implementation. | `tensorrt_llm/_torch/models/modeling_qwen3_next.py` |
| 2025-12-02 | [#8902](https://github.com/NVIDIA/TensorRT-LLM/pull/8902) | merged | [None][chroe] Polish qwen3-next modeling code. | `tensorrt_llm/_torch/models/modeling_qwen3_next.py` |
| 2025-12-23 | [#9691](https://github.com/NVIDIA/TensorRT-LLM/pull/9691) | merged | [TRTLLM-9432][feat] Reduce synchronization and recompilation for qwen3-next | `tensorrt_llm/_torch/models/modeling_qwen3_next.py` |
| 2025-12-27 | [#10228](https://github.com/NVIDIA/TensorRT-LLM/pull/10228) | merged | [TRTLLM-8577][feat] Clean the Qwen3-next code by removing Qwen3NextCo… | `tensorrt_llm/_torch/models/modeling_qwen3_next.py` |
| 2026-03-16 | [#10218](https://github.com/NVIDIA/TensorRT-LLM/pull/10218) | merged | [TRTLLM-9767][feat] Enable attention dp for qwen3-next. | `tensorrt_llm/_torch/models/modeling_qwen3_next.py`, `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_next_weight_mapper.py` |
| 2026-04-03 | [#11370](https://github.com/NVIDIA/TensorRT-LLM/pull/11370) | merged | [None][feat] Qwen3-Next MTP | `tensorrt_llm/_torch/models/modeling_qwen3_next.py`, `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_next_weight_mapper.py` |
| 2026-07-14 | [#16314](https://github.com/NVIDIA/TensorRT-LLM/pull/16314) | merged | [TRTLLM-14054][perf] Load Qwen3-Next GDN in_proj in dense layout and add a multi-row gated RMSNorm | `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_next_weight_mapper.py` |
| 2026-07-24 | [#15194](https://github.com/NVIDIA/TensorRT-LLM/pull/15194) | merged | [TRTLLM-13349][perf] Fuse gemma RMSNorm into AllReduce for Qwen3-Next/Qwen3.5… | `tests/unittest/_torch/models/test_qwen3_next_eager_fusion.py`, `tensorrt_llm/_torch/models/modeling_qwen3_next.py` |

## 逐 PR diff 审计卡

### PR #7892 - [None][feat] Support Qwen3 next

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/7892
- 状态/时间: merged / 2025-09-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_next_weight_mapper.py`, `tensorrt_llm/_torch/models/modeling_qwen3_next.py`；关联提交 `38d6e4e60b1a`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 30 个文件，+5286/-39，可读 patch 5588 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_qwen3_next.py` added +1519/-0 (1519 lines); hunks: -0,0 +1,1519; symbols: ensure_divisibility, divide, Qwen3NextConfig, to，涉及 `ensure_divisibility, divide, Qwen3NextConfig`；`tensorrt_llm/_torch/models/checkpoints/hf/qwen3_next_weight_mapper.py` added +105/-0 (105 lines); hunks: -0,0 +1,105; symbols: Qwen3NextHfWeightMapper, init_model_and_config, should_skip_module, _duplicate_kv_weights，涉及 `Qwen3NextHfWeightMapper, init_model_and_config, should_skip_module`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_qwen3_next.py` added +1519/-0 (1519 lines); hunks: -0,0 +1,1519; symbols: ensure_divisibility, divide, Qwen3NextConfig, to
  - `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_next_weight_mapper.py` added +105/-0 (105 lines); hunks: -0,0 +1,105; symbols: Qwen3NextHfWeightMapper, init_model_and_config, should_skip_module, _duplicate_kv_weights
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3_next.py
@@ -0,0 +1,1519 @@
+# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py
+# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/configs/qwen3_next.py
+# coding=utf-8
+# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
diff -- tensorrt_llm/_torch/models/checkpoints/hf/qwen3_next_weight_mapper.py
@@ -0,0 +1,105 @@
+from typing import Union
+import torch
+from torch import nn
+from tensorrt_llm._torch.model_config import ModelConfig
+from tensorrt_llm._torch.models.checkpoints.hf.qwen2_moe_weight_mapper import \
+    Qwen2MoeHfWeightMapper
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_next.py` added +1519/-0; `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_next_weight_mapper.py` added +105/-0
- 验证与风险: runtime 路径改动集中在 `cpp/tensorrt_llm/kernels/fusedQKNormRopeKernel.cu`, `tensorrt_llm/_torch/custom_ops/__init__.py`, `tensorrt_llm/_torch/custom_ops/flashinfer_custom_ops.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #8064 - [None][chore] Refine qwen3-next implementation.

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/8064
- 状态/时间: merged / 2025-09-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_qwen3_next.py`；关联提交 `b4be0d2e4c37`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 5 个文件，+45/-47，可读 patch 220 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_qwen3_next.py` modified +11/-16 (27 lines); hunks: -1104,17 +1104,15 @@ def __init__(; -1266,17 +1264,15 @@ def __init__(self, model_config: ModelConfig[Qwen3NextCo...; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_qwen3_next.py` modified +11/-16 (27 lines); hunks: -1104,17 +1104,15 @@ def __init__(; -1266,17 +1264,15 @@ def __init__(self, model_config: ModelConfig[Qwen3NextCo...; symbols: __init__
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3_next.py
@@ -1104,17 +1104,15 @@ def __init__(
-        use_gemma_rms_norm = True
-                                       use_gemma_rms_norm=use_gemma_rms_norm)
+                                       use_gemma=True)
-        self.post_attention_layernorm = RMSNorm(
-            hidden_size=config.hidden_size,
-            eps=config.rms_norm_eps,
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_next.py` modified +11/-16
- 验证与风险: runtime 路径改动集中在 `tensorrt_llm/_torch/models/modeling_qwen3_next.py`, `tensorrt_llm/_torch/modules/attention.py`, `tensorrt_llm/_torch/modules/qk_norm_attention.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #8902 - [None][chroe] Polish qwen3-next modeling code.

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/8902
- 状态/时间: merged / 2025-12-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_qwen3_next.py`；关联提交 `6fbe87c8b54c`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+96/-95，可读 patch 296 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_qwen3_next.py` modified +96/-93 (189 lines); hunks: -50,9 +50,10; -387,6 +388,7 @@ def __init__(; symbols: __init__, forward, _compute_routed_output，涉及 `__init__, forward, _compute_routed_output`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_qwen3_next.py` modified +96/-93 (189 lines); hunks: -50,9 +50,10; -387,6 +388,7 @@ def __init__(; symbols: __init__, forward, _compute_routed_output
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3_next.py
@@ -50,9 +50,10 @@
+from ..modules.multi_stream_utils import maybe_execute_in_parallel
-from ..utils import AuxStreamType
+from ..utils import AuxStreamType, EventType
@@ -387,6 +388,7 @@ def __init__(
+        self.aux_stream = aux_stream
@@ -425,6 +427,11 @@ def __init__(
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_next.py` modified +96/-93
- 验证与风险: runtime 路径改动集中在 `tensorrt_llm/_torch/models/modeling_qwen3_next.py`, `tensorrt_llm/_torch/modules/fla/chunk.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #9691 - [TRTLLM-9432][feat] Reduce synchronization and recompilation for qwen3-next

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/9691
- 状态/时间: merged / 2025-12-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_qwen3_next.py`；关联提交 `648196f8aea8`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 4 个文件，+33/-34，可读 patch 183 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_qwen3_next.py` modified +11/-28 (39 lines); hunks: -826,17 +826,13 @@ def forward_decode(; -870,7 +866,7 @@ def forward_decode(; symbols: forward_decode, forward_extend，涉及 `forward_decode, forward_extend`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_qwen3_next.py` modified +11/-28 (39 lines); hunks: -826,17 +826,13 @@ def forward_decode(; -870,7 +866,7 @@ def forward_decode(; symbols: forward_decode, forward_extend
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3_next.py
@@ -826,17 +826,13 @@ def forward_decode(
-        num_decodes,
-        cu_seqlens,
+        query_start_loc_long,
-        query_start_loc = torch.arange(0,
-                                       num_decodes + 1,
-                                       device=cu_seqlens.device).to(torch.long)
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_next.py` modified +11/-28
- 验证与风险: runtime 路径改动集中在 `tensorrt_llm/_torch/models/modeling_qwen3_next.py`, `tensorrt_llm/_torch/modules/fla/l2norm.py`, `tensorrt_llm/_torch/modules/mamba/mamba2_metadata.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #10228 - [TRTLLM-8577][feat] Clean the Qwen3-next code by removing Qwen3NextCo…

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/10228
- 状态/时间: merged / 2025-12-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_qwen3_next.py`；关联提交 `1865020b6f7c`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+1/-250，可读 patch 265 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_qwen3_next.py` modified +1/-250 (251 lines); hunks: -23,8 +23,7; -71,254 +70,6 @@ def divide(numerator, denominator):; symbols: divide, Qwen3NextConfig, to, __init__，涉及 `divide, Qwen3NextConfig, to`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_qwen3_next.py` modified +1/-250 (251 lines); hunks: -23,8 +23,7; -71,254 +70,6 @@ def divide(numerator, denominator):; symbols: divide, Qwen3NextConfig, to, __init__
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3_next.py
@@ -23,8 +23,7 @@
-from transformers.configuration_utils import PretrainedConfig
-from transformers.modeling_rope_utils import rope_config_validation
+from transformers import Qwen3NextConfig
@@ -71,254 +70,6 @@ def divide(numerator, denominator):
-class Qwen3NextConfig(PretrainedConfig):
-    r"""
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_next.py` modified +1/-250
- 验证与风险: runtime 路径改动集中在 `tensorrt_llm/_torch/models/modeling_qwen3_next.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #10218 - [TRTLLM-9767][feat] Enable attention dp for qwen3-next.

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/10218
- 状态/时间: merged / 2026-03-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_next_weight_mapper.py`, `tensorrt_llm/_torch/models/modeling_qwen3_next.py`；关联提交 `677cdf673ae2`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 10 个文件，+76/-57，可读 patch 402 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_qwen3_next.py` modified +37/-37 (74 lines); hunks: -36,19 +36,19; -138,8 +138,10 @@ def __init__(; symbols: __init__, forward, _compute_routed_output, _compute_shared_output，涉及 `__init__, forward, _compute_routed_output`；`tensorrt_llm/_torch/models/checkpoints/hf/qwen3_next_weight_mapper.py` modified +2/-0 (2 lines); hunks: -57,6 +57,8 @@ def preprocess_weights(self, weights: dict) -> dict:; symbols: preprocess_weights，涉及 `preprocess_weights`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_qwen3_next.py` modified +37/-37 (74 lines); hunks: -36,19 +36,19; -138,8 +138,10 @@ def __init__(; symbols: __init__, forward, _compute_routed_output, _compute_shared_output
  - `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_next_weight_mapper.py` modified +2/-0 (2 lines); hunks: -57,6 +57,8 @@ def preprocess_weights(self, weights: dict) -> dict:; symbols: preprocess_weights
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3_next.py
@@ -36,19 +36,19 @@
+from tensorrt_llm._utils import get_sm_version
-                           MoEAllReduce, MoEAllReduceParams, allgather)
+                           MoEAllReduce, MoEAllReduceParams)
-                                 RoutingMethodType, TRTLLMGenFusedMoE,
-                                 create_moe)
+                                 RoutingMethodType, create_moe)
diff -- tensorrt_llm/_torch/models/checkpoints/hf/qwen3_next_weight_mapper.py
@@ -57,6 +57,8 @@ def preprocess_weights(self, weights: dict) -> dict:
+        if self.config.mapping.enable_attention_dp:
+            tp_size = 1
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_next.py` modified +37/-37; `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_next_weight_mapper.py` modified +2/-0
- 验证与风险: diff 自带测试面 `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/integration/test_lists/test-db/l0_dgx_h100.yml`, `tests/integration/test_lists/test-db/l0_dgx_h200.yml`, `tests/integration/test_lists/test-db/l0_gb200_multi_gpus.yml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #11370 - [None][feat] Qwen3-Next MTP

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/11370
- 状态/时间: merged / 2026-04-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_next_weight_mapper.py`, `tensorrt_llm/_torch/models/modeling_qwen3_next.py`；关联提交 `1045f3858e4d`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 6 个文件，+1135/-619，可读 patch 1888 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_qwen3_next.py` modified +190/-619 (809 lines); hunks: -25,24 +25,17; -55,30 +48,16; symbols: ensure_divisibility, divide, Qwen3NextGate, __init__，涉及 `ensure_divisibility, divide, Qwen3NextGate`；`tensorrt_llm/_torch/models/checkpoints/hf/qwen3_next_weight_mapper.py` modified +19/-0 (19 lines); hunks: -56,6 +56,7 @@ def preprocess_weights(self, weights: dict) -> dict:; -66,10 +67,28 @@ def preprocess_weights(self, weights: dict) -> dict:; symbols: preprocess_weights，涉及 `preprocess_weights`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_qwen3_next.py` modified +190/-619 (809 lines); hunks: -25,24 +25,17; -55,30 +48,16; symbols: ensure_divisibility, divide, Qwen3NextGate, __init__
  - `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_next_weight_mapper.py` modified +19/-0 (19 lines); hunks: -56,6 +56,7 @@ def preprocess_weights(self, weights: dict) -> dict:; -66,10 +67,28 @@ def preprocess_weights(self, weights: dict) -> dict:; symbols: preprocess_weights
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3_next.py
@@ -25,24 +25,17 @@
-import triton
-import triton.language as tl
-from tensorrt_llm._torch.modules.fla.chunk import chunk_gated_delta_rule
-from tensorrt_llm._torch.modules.fla.fused_sigmoid_gating_recurrent import \
-    fused_sigmoid_gating_delta_rule_update
-from tensorrt_llm._torch.pyexecutor.mamba_cache_manager import \
diff -- tensorrt_llm/_torch/models/checkpoints/hf/qwen3_next_weight_mapper.py
@@ -56,6 +56,7 @@ def preprocess_weights(self, weights: dict) -> dict:
+        mtp_layer_offset = config.num_hidden_layers
@@ -66,10 +67,28 @@ def preprocess_weights(self, weights: dict) -> dict:
+        mtp_mapping = {
+            "mtp.fc": "fc",
+            "mtp.norm": "shared_head.norm",
+            "mtp.pre_fc_norm_embedding": "pre_fc_norm_embedding",
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_next.py` modified +190/-619; `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_next_weight_mapper.py` modified +19/-0
- 验证与风险: diff 自带测试面 `tests/integration/defs/accuracy/test_llm_api_pytorch.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #16314 - [TRTLLM-14054][perf] Load Qwen3-Next GDN in_proj in dense layout and add a multi-row gated RMSNorm

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/16314
- 状态/时间: merged / 2026-07-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_next_weight_mapper.py`；关联提交 `0f05df4f4c5f`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 5 个文件，+548/-231，可读 patch 1015 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_next_weight_mapper.py` modified +124/-3 (127 lines); hunks: -1,3 +1,5; -6,6 +8,86; symbols: grouped_to_dense_in_proj_qkvz_perm, grouped_to_dense_in_proj_ba_perm, _rows_to_scale_block_perm, _permute_rows，涉及 `grouped_to_dense_in_proj_qkvz_perm, grouped_to_dense_in_proj_ba_perm, _rows_to_scale_block_perm`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_next_weight_mapper.py` modified +124/-3 (127 lines); hunks: -1,3 +1,5; -6,6 +8,86; symbols: grouped_to_dense_in_proj_qkvz_perm, grouped_to_dense_in_proj_ba_perm, _rows_to_scale_block_perm, _permute_rows
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/checkpoints/hf/qwen3_next_weight_mapper.py
@@ -1,3 +1,5 @@
+import math
@@ -6,6 +8,86 @@
+# 2D block edge of FP8 block-scale tensors (weight_scale_inv), matching
+# FP8BlockScalesLinearMethod.
+_FP8_BLOCK_SIZE = 128
+def grouped_to_dense_in_proj_qkvz_perm(num_k_heads: int, head_k_dim: int,
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_next_weight_mapper.py` modified +124/-3
- 验证与风险: diff 自带测试面 `tests/unittest/_torch/modules/mamba/test_gdn_kernel_optimizations.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #15194 - [TRTLLM-13349][perf] Fuse gemma RMSNorm into AllReduce for Qwen3-Next/Qwen3.5…

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/15194
- 状态/时间: merged / 2026-07-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_qwen3_next.py`, `tests/unittest/_torch/models/test_qwen3_next_eager_fusion.py`；关联提交 `1fae43cc6c0d`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 3 个文件，+278/-29，可读 patch 405 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/unittest/_torch/models/test_qwen3_next_eager_fusion.py` added +167/-0 (167 lines); hunks: -0,0 +1,167; symbols: _new_causal_lm, test_setup_aliases_does_not_read_meta_weights, test_cache_derived_state_refreshes_gemma_norm_weight, test_eager_fusion_is_enabled_for_gdn_by_default，涉及 `_new_causal_lm, test_setup_aliases_does_not_read_meta_weights, test_cache_derived_state_refreshes_gemma_norm_weight`；`tensorrt_llm/_torch/models/modeling_qwen3_next.py` modified +93/-29 (122 lines); hunks: -62,6 +62,58; -368,17 +420,15 @@ def __init__(; symbols: _fused_norm_weight, _precompute_fused_norm_weights, _eager_fusion_enabled, Qwen3NextGate，涉及 `_fused_norm_weight, _precompute_fused_norm_weights, _eager_fusion_enabled`。
- 代码 diff 细节:
  - `tests/unittest/_torch/models/test_qwen3_next_eager_fusion.py` added +167/-0 (167 lines); hunks: -0,0 +1,167; symbols: _new_causal_lm, test_setup_aliases_does_not_read_meta_weights, test_cache_derived_state_refreshes_gemma_norm_weight, test_eager_fusion_is_enabled_for_gdn_by_default
  - `tensorrt_llm/_torch/models/modeling_qwen3_next.py` modified +93/-29 (122 lines); hunks: -62,6 +62,58; -368,17 +420,15 @@ def __init__(; symbols: _fused_norm_weight, _precompute_fused_norm_weights, _eager_fusion_enabled, Qwen3NextGate
- 关键代码摘录:

```diff
diff -- tests/unittest/_torch/models/test_qwen3_next_eager_fusion.py
@@ -0,0 +1,167 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
diff -- tensorrt_llm/_torch/models/modeling_qwen3_next.py
@@ -62,6 +62,58 @@
+def _fused_norm_weight(norm: RMSNorm) -> torch.Tensor:
+    """Weight to feed the fused AllReduce+RMSNorm op for ``norm``.
+    Gemma RMSNorm scales by ``(1 + weight)`` (see RMSNorm.forward), but the
+    fused AR+RMSNorm kernels (and the NCCL / NCCL_SYMMETRIC fallbacks the
+    AUTO strategy may pick) only apply ``weight``. Baking the ``+1`` into the
+    weight makes EVERY allreduce backend produce the correct gemma result
```

- 提取文件（未人工审阅）:
  - tests: `tests/unittest/_torch/models/test_qwen3_next_eager_fusion.py` added +167/-0
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_next.py` modified +93/-29
- 验证与风险: diff 自带测试面 `tests/unittest/_torch/models/test_qwen3_next_eager_fusion.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
