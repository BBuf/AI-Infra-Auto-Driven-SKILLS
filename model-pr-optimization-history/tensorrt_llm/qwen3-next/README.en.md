# TensorRT-LLM Qwen3 Next Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `examples/configs/curated/qwen3-next.yaml` | no direct PR-number commit |
| `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_next_weight_mapper.py` | [#7892](https://github.com/NVIDIA/TensorRT-LLM/pull/7892), [#10218](https://github.com/NVIDIA/TensorRT-LLM/pull/10218), [#11370](https://github.com/NVIDIA/TensorRT-LLM/pull/11370), [#16314](https://github.com/NVIDIA/TensorRT-LLM/pull/16314) |
| `tensorrt_llm/_torch/models/modeling_qwen3_next.py` | [#7892](https://github.com/NVIDIA/TensorRT-LLM/pull/7892), [#8064](https://github.com/NVIDIA/TensorRT-LLM/pull/8064), [#8902](https://github.com/NVIDIA/TensorRT-LLM/pull/8902), [#9691](https://github.com/NVIDIA/TensorRT-LLM/pull/9691), [#10218](https://github.com/NVIDIA/TensorRT-LLM/pull/10218), [#10228](https://github.com/NVIDIA/TensorRT-LLM/pull/10228), [#11370](https://github.com/NVIDIA/TensorRT-LLM/pull/11370), [#15194](https://github.com/NVIDIA/TensorRT-LLM/pull/15194) |
| `tests/unittest/_torch/models/test_qwen3_next_eager_fusion.py` | [#15194](https://github.com/NVIDIA/TensorRT-LLM/pull/15194) |
| `tests/unittest/_torch/models/test_qwen3_next_moe_quant.py` | no direct PR-number commit |

## PR Coverage Summary

- Git-traced PRs: 9
- Extra PRs preserved from existing docs: 0
- Total PRs in this document: 9
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
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

## Per-PR Diff Audit Cards

### PR #7892 - [None][feat] Support Qwen3 next

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/7892
- Status/date: merged / 2025-09-29
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_next_weight_mapper.py`, `tensorrt_llm/_torch/models/modeling_qwen3_next.py`; associated commits `38d6e4e60b1a`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 30 files, +5286/-39, 5588 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_qwen3_next.py` added +1519/-0 (1519 lines); hunks: -0,0 +1,1519; symbols: ensure_divisibility, divide, Qwen3NextConfig, to, touching `ensure_divisibility, divide, Qwen3NextConfig`; `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_next_weight_mapper.py` added +105/-0 (105 lines); hunks: -0,0 +1,105; symbols: Qwen3NextHfWeightMapper, init_model_and_config, should_skip_module, _duplicate_kv_weights, touching `Qwen3NextHfWeightMapper, init_model_and_config, should_skip_module`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_qwen3_next.py` added +1519/-0 (1519 lines); hunks: -0,0 +1,1519; symbols: ensure_divisibility, divide, Qwen3NextConfig, to
  - `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_next_weight_mapper.py` added +105/-0 (105 lines); hunks: -0,0 +1,105; symbols: Qwen3NextHfWeightMapper, init_model_and_config, should_skip_module, _duplicate_kv_weights
- Key code excerpts:

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

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_next.py` added +1519/-0; `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_next_weight_mapper.py` added +105/-0
- Risk and verification: Runtime changes concentrate in `cpp/tensorrt_llm/kernels/fusedQKNormRopeKernel.cu`, `tensorrt_llm/_torch/custom_ops/__init__.py`, `tensorrt_llm/_torch/custom_ops/flashinfer_custom_ops.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #8064 - [None][chore] Refine qwen3-next implementation.

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/8064
- Status/date: merged / 2025-09-30
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_qwen3_next.py`; associated commits `b4be0d2e4c37`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 5 files, +45/-47, 220 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_qwen3_next.py` modified +11/-16 (27 lines); hunks: -1104,17 +1104,15 @@ def __init__(; -1266,17 +1264,15 @@ def __init__(self, model_config: ModelConfig[Qwen3NextCo...; symbols: __init__, touching `__init__`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_qwen3_next.py` modified +11/-16 (27 lines); hunks: -1104,17 +1104,15 @@ def __init__(; -1266,17 +1264,15 @@ def __init__(self, model_config: ModelConfig[Qwen3NextCo...; symbols: __init__
- Key code excerpts:

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

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_next.py` modified +11/-16
- Risk and verification: Runtime changes concentrate in `tensorrt_llm/_torch/models/modeling_qwen3_next.py`, `tensorrt_llm/_torch/modules/attention.py`, `tensorrt_llm/_torch/modules/qk_norm_attention.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #8902 - [None][chroe] Polish qwen3-next modeling code.

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/8902
- Status/date: merged / 2025-12-02
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_qwen3_next.py`; associated commits `6fbe87c8b54c`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +96/-95, 296 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_qwen3_next.py` modified +96/-93 (189 lines); hunks: -50,9 +50,10; -387,6 +388,7 @@ def __init__(; symbols: __init__, forward, _compute_routed_output, touching `__init__, forward, _compute_routed_output`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_qwen3_next.py` modified +96/-93 (189 lines); hunks: -50,9 +50,10; -387,6 +388,7 @@ def __init__(; symbols: __init__, forward, _compute_routed_output
- Key code excerpts:

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

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_next.py` modified +96/-93
- Risk and verification: Runtime changes concentrate in `tensorrt_llm/_torch/models/modeling_qwen3_next.py`, `tensorrt_llm/_torch/modules/fla/chunk.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9691 - [TRTLLM-9432][feat] Reduce synchronization and recompilation for qwen3-next

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/9691
- Status/date: merged / 2025-12-23
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_qwen3_next.py`; associated commits `648196f8aea8`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 4 files, +33/-34, 183 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_qwen3_next.py` modified +11/-28 (39 lines); hunks: -826,17 +826,13 @@ def forward_decode(; -870,7 +866,7 @@ def forward_decode(; symbols: forward_decode, forward_extend, touching `forward_decode, forward_extend`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_qwen3_next.py` modified +11/-28 (39 lines); hunks: -826,17 +826,13 @@ def forward_decode(; -870,7 +866,7 @@ def forward_decode(; symbols: forward_decode, forward_extend
- Key code excerpts:

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

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_next.py` modified +11/-28
- Risk and verification: Runtime changes concentrate in `tensorrt_llm/_torch/models/modeling_qwen3_next.py`, `tensorrt_llm/_torch/modules/fla/l2norm.py`, `tensorrt_llm/_torch/modules/mamba/mamba2_metadata.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #10228 - [TRTLLM-8577][feat] Clean the Qwen3-next code by removing Qwen3NextCo…

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/10228
- Status/date: merged / 2025-12-27
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_qwen3_next.py`; associated commits `1865020b6f7c`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +1/-250, 265 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_qwen3_next.py` modified +1/-250 (251 lines); hunks: -23,8 +23,7; -71,254 +70,6 @@ def divide(numerator, denominator):; symbols: divide, Qwen3NextConfig, to, __init__, touching `divide, Qwen3NextConfig, to`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_qwen3_next.py` modified +1/-250 (251 lines); hunks: -23,8 +23,7; -71,254 +70,6 @@ def divide(numerator, denominator):; symbols: divide, Qwen3NextConfig, to, __init__
- Key code excerpts:

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

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_next.py` modified +1/-250
- Risk and verification: Runtime changes concentrate in `tensorrt_llm/_torch/models/modeling_qwen3_next.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #10218 - [TRTLLM-9767][feat] Enable attention dp for qwen3-next.

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/10218
- Status/date: merged / 2026-03-16
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_next_weight_mapper.py`, `tensorrt_llm/_torch/models/modeling_qwen3_next.py`; associated commits `677cdf673ae2`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 10 files, +76/-57, 402 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_qwen3_next.py` modified +37/-37 (74 lines); hunks: -36,19 +36,19; -138,8 +138,10 @@ def __init__(; symbols: __init__, forward, _compute_routed_output, _compute_shared_output, touching `__init__, forward, _compute_routed_output`; `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_next_weight_mapper.py` modified +2/-0 (2 lines); hunks: -57,6 +57,8 @@ def preprocess_weights(self, weights: dict) -> dict:; symbols: preprocess_weights, touching `preprocess_weights`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_qwen3_next.py` modified +37/-37 (74 lines); hunks: -36,19 +36,19; -138,8 +138,10 @@ def __init__(; symbols: __init__, forward, _compute_routed_output, _compute_shared_output
  - `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_next_weight_mapper.py` modified +2/-0 (2 lines); hunks: -57,6 +57,8 @@ def preprocess_weights(self, weights: dict) -> dict:; symbols: preprocess_weights
- Key code excerpts:

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

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_next.py` modified +37/-37; `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_next_weight_mapper.py` modified +2/-0
- Risk and verification: The diff ships test coverage in `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/integration/test_lists/test-db/l0_dgx_h100.yml`, `tests/integration/test_lists/test-db/l0_dgx_h200.yml`, `tests/integration/test_lists/test-db/l0_gb200_multi_gpus.yml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #11370 - [None][feat] Qwen3-Next MTP

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/11370
- Status/date: merged / 2026-04-03
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_next_weight_mapper.py`, `tensorrt_llm/_torch/models/modeling_qwen3_next.py`; associated commits `1045f3858e4d`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 6 files, +1135/-619, 1888 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_qwen3_next.py` modified +190/-619 (809 lines); hunks: -25,24 +25,17; -55,30 +48,16; symbols: ensure_divisibility, divide, Qwen3NextGate, __init__, touching `ensure_divisibility, divide, Qwen3NextGate`; `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_next_weight_mapper.py` modified +19/-0 (19 lines); hunks: -56,6 +56,7 @@ def preprocess_weights(self, weights: dict) -> dict:; -66,10 +67,28 @@ def preprocess_weights(self, weights: dict) -> dict:; symbols: preprocess_weights, touching `preprocess_weights`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_qwen3_next.py` modified +190/-619 (809 lines); hunks: -25,24 +25,17; -55,30 +48,16; symbols: ensure_divisibility, divide, Qwen3NextGate, __init__
  - `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_next_weight_mapper.py` modified +19/-0 (19 lines); hunks: -56,6 +56,7 @@ def preprocess_weights(self, weights: dict) -> dict:; -66,10 +67,28 @@ def preprocess_weights(self, weights: dict) -> dict:; symbols: preprocess_weights
- Key code excerpts:

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

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_next.py` modified +190/-619; `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_next_weight_mapper.py` modified +19/-0
- Risk and verification: The diff ships test coverage in `tests/integration/defs/accuracy/test_llm_api_pytorch.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #16314 - [TRTLLM-14054][perf] Load Qwen3-Next GDN in_proj in dense layout and add a multi-row gated RMSNorm

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/16314
- Status/date: merged / 2026-07-14
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_next_weight_mapper.py`; associated commits `0f05df4f4c5f`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 5 files, +548/-231, 1015 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_next_weight_mapper.py` modified +124/-3 (127 lines); hunks: -1,3 +1,5; -6,6 +8,86; symbols: grouped_to_dense_in_proj_qkvz_perm, grouped_to_dense_in_proj_ba_perm, _rows_to_scale_block_perm, _permute_rows, touching `grouped_to_dense_in_proj_qkvz_perm, grouped_to_dense_in_proj_ba_perm, _rows_to_scale_block_perm`.
- Code diff details:
  - `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_next_weight_mapper.py` modified +124/-3 (127 lines); hunks: -1,3 +1,5; -6,6 +8,86; symbols: grouped_to_dense_in_proj_qkvz_perm, grouped_to_dense_in_proj_ba_perm, _rows_to_scale_block_perm, _permute_rows
- Key code excerpts:

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

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_next_weight_mapper.py` modified +124/-3
- Risk and verification: The diff ships test coverage in `tests/unittest/_torch/modules/mamba/test_gdn_kernel_optimizations.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #15194 - [TRTLLM-13349][perf] Fuse gemma RMSNorm into AllReduce for Qwen3-Next/Qwen3.5…

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/15194
- Status/date: merged / 2026-07-24
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_qwen3_next.py`, `tests/unittest/_torch/models/test_qwen3_next_eager_fusion.py`; associated commits `1fae43cc6c0d`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +278/-29, 405 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/unittest/_torch/models/test_qwen3_next_eager_fusion.py` added +167/-0 (167 lines); hunks: -0,0 +1,167; symbols: _new_causal_lm, test_setup_aliases_does_not_read_meta_weights, test_cache_derived_state_refreshes_gemma_norm_weight, test_eager_fusion_is_enabled_for_gdn_by_default, touching `_new_causal_lm, test_setup_aliases_does_not_read_meta_weights, test_cache_derived_state_refreshes_gemma_norm_weight`; `tensorrt_llm/_torch/models/modeling_qwen3_next.py` modified +93/-29 (122 lines); hunks: -62,6 +62,58; -368,17 +420,15 @@ def __init__(; symbols: _fused_norm_weight, _precompute_fused_norm_weights, _eager_fusion_enabled, Qwen3NextGate, touching `_fused_norm_weight, _precompute_fused_norm_weights, _eager_fusion_enabled`.
- Code diff details:
  - `tests/unittest/_torch/models/test_qwen3_next_eager_fusion.py` added +167/-0 (167 lines); hunks: -0,0 +1,167; symbols: _new_causal_lm, test_setup_aliases_does_not_read_meta_weights, test_cache_derived_state_refreshes_gemma_norm_weight, test_eager_fusion_is_enabled_for_gdn_by_default
  - `tensorrt_llm/_torch/models/modeling_qwen3_next.py` modified +93/-29 (122 lines); hunks: -62,6 +62,58; -368,17 +420,15 @@ def __init__(; symbols: _fused_norm_weight, _precompute_fused_norm_weights, _eager_fusion_enabled, Qwen3NextGate
- Key code excerpts:

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

- Extracted files (not manually reviewed):
  - tests: `tests/unittest/_torch/models/test_qwen3_next_eager_fusion.py` added +167/-0
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_next.py` modified +93/-29
- Risk and verification: The diff ships test coverage in `tests/unittest/_torch/models/test_qwen3_next_eager_fusion.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
