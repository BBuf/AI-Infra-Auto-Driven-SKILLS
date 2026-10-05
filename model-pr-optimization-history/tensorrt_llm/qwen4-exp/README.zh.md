# TensorRT-LLM Qwen4-Exp (Qwen3.8-Flash-Next) 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `tensorrt_llm/_torch/configs/qwen4_exp.py` | [#18585](https://github.com/NVIDIA/TensorRT-LLM/pull/18585) |
| `tensorrt_llm/_torch/models/checkpoints/hf/qwen4_exp_weight_mapper.py` | [#18585](https://github.com/NVIDIA/TensorRT-LLM/pull/18585) |
| `tensorrt_llm/_torch/models/modeling_qwen4_exp.py` | [#18585](https://github.com/NVIDIA/TensorRT-LLM/pull/18585), [#18823](https://github.com/NVIDIA/TensorRT-LLM/pull/18823), [#18883](https://github.com/NVIDIA/TensorRT-LLM/pull/18883) |
| `tensorrt_llm/_torch/modules/qwen4_exp/__init__.py` | [#18585](https://github.com/NVIDIA/TensorRT-LLM/pull/18585) |
| `tensorrt_llm/_torch/modules/qwen4_exp/hyper_connection.py` | [#18585](https://github.com/NVIDIA/TensorRT-LLM/pull/18585), [#18883](https://github.com/NVIDIA/TensorRT-LLM/pull/18883) |
| `tensorrt_llm/_torch/modules/qwen4_exp/hyper_connection_kernels.py` | [#18585](https://github.com/NVIDIA/TensorRT-LLM/pull/18585), [#18883](https://github.com/NVIDIA/TensorRT-LLM/pull/18883) |
| `tensorrt_llm/_torch/modules/qwen4_exp/ple.py` | [#18585](https://github.com/NVIDIA/TensorRT-LLM/pull/18585), [#18823](https://github.com/NVIDIA/TensorRT-LLM/pull/18823) |
| `tensorrt_llm/_torch/modules/qwen4_exp/ple_kernels.py` | [#18585](https://github.com/NVIDIA/TensorRT-LLM/pull/18585) |
| `tests/unittest/_torch/modeling/test_qwen4_exp_support.py` | [#18585](https://github.com/NVIDIA/TensorRT-LLM/pull/18585), [#18883](https://github.com/NVIDIA/TensorRT-LLM/pull/18883), [#18921](https://github.com/NVIDIA/TensorRT-LLM/pull/18921) |
| `tests/unittest/_torch/modules/test_qwen4_exp_hyper_connection.py` | [#18585](https://github.com/NVIDIA/TensorRT-LLM/pull/18585), [#18883](https://github.com/NVIDIA/TensorRT-LLM/pull/18883) |
| `tests/unittest/_torch/modules/test_qwen4_exp_ple.py` | [#18585](https://github.com/NVIDIA/TensorRT-LLM/pull/18585) |
| `tests/unittest/_torch/modules/test_qwen4_exp_ple_kernels.py` | [#18585](https://github.com/NVIDIA/TensorRT-LLM/pull/18585) |
| `tests/unittest/_torch/modules/test_qwen4_exp_ple_offload.py` | [#18585](https://github.com/NVIDIA/TensorRT-LLM/pull/18585) |

## PR 覆盖总览

- git 追溯 PR 数: 4
- 原文档显式引用补充 PR 数: 0
- 当前文档总 PR 数: 4
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-09-04 | [#18585](https://github.com/NVIDIA/TensorRT-LLM/pull/18585) | merged | [None][feat] add Qwen3.8-Flash-Next support | `tensorrt_llm/_torch/models/modeling_qwen4_exp.py`, `tensorrt_llm/_torch/models/checkpoints/hf/qwen4_exp_weight_mapper.py`, `tensorrt_llm/_torch/configs/qwen4_exp.py` |
| 2026-09-09 | [#18823](https://github.com/NVIDIA/TensorRT-LLM/pull/18823) | merged | [TRTLLM-16182][fix] load the mixed-precision Qwen3.8-Flash-Next NVFP4 checkpoint | `tensorrt_llm/_torch/models/modeling_qwen4_exp.py`, `tensorrt_llm/_torch/modules/qwen4_exp/ple.py` |
| 2026-09-10 | [#18883](https://github.com/NVIDIA/TensorRT-LLM/pull/18883) | merged | [None][perf] fuse decode kernel seams for Qwen3.8-Flash-Next and de-vendor the low-M GEMMs | `tensorrt_llm/_torch/models/modeling_qwen4_exp.py`, `tensorrt_llm/_torch/modules/qwen4_exp/hyper_connection_kernels.py`, `tests/unittest/_torch/modules/test_qwen4_exp_hyper_connection.py` |
| 2026-09-15 | [#18921](https://github.com/NVIDIA/TensorRT-LLM/pull/18921) | merged | [None][feat] support disaggregated serving for Qwen3.8-Flash-Next | `tests/unittest/_torch/modeling/test_qwen4_exp_support.py`, `tensorrt_llm/_torch/disaggregation/resource/kv_extractor.py`, `tensorrt_llm/_torch/disaggregation/native/mixers/ssm/peer.py` |

## 逐 PR diff 审计卡

### PR #18585 - [None][feat] add Qwen3.8-Flash-Next support

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/18585
- 状态/时间: merged / 2026-09-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/configs/qwen4_exp.py`, `tensorrt_llm/_torch/models/checkpoints/hf/qwen4_exp_weight_mapper.py`, `tensorrt_llm/_torch/models/modeling_qwen4_exp.py`, `tensorrt_llm/_torch/modules/qwen4_exp/__init__.py`, `tensorrt_llm/_torch/modules/qwen4_exp/hyper_connection.py` 等 13 个文件；关联提交 `02746c52d0b1`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 62 个文件，+14598/-43，可读 patch 15436 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_qwen4_exp.py` added +1291/-0 (1291 lines); hunks: -0,0 +1,1291; symbols: _qwen4_exp_tp_output_reduction_enabled, _validate_qwen4_exp_runtime_config, Qwen4ExpGatedDeltaNet, _postprocess_gdn_output，涉及 `_qwen4_exp_tp_output_reduction_enabled, _validate_qwen4_exp_runtime_config, Qwen4ExpGatedDeltaNet`；`tensorrt_llm/_torch/models/checkpoints/hf/qwen4_exp_weight_mapper.py` added +650/-0 (650 lines); hunks: -0,0 +1,650; symbols: _rank_block, _normalize_moe_module_weights, Qwen4ExpHfWeightMapper, should_skip_module，涉及 `_rank_block, _normalize_moe_module_weights, Qwen4ExpHfWeightMapper`；`tensorrt_llm/_torch/configs/qwen4_exp.py` added +201/-0 (201 lines); hunks: -0,0 +1,201; symbols: _flatten_qwen4_exp_rope, _normalize_qwen4_exp_layer_types, Qwen4ExpTextConfig, __init__，涉及 `_flatten_qwen4_exp_rope, _normalize_qwen4_exp_layer_types, Qwen4ExpTextConfig`；`tensorrt_llm/_torch/configs/__init__.py` modified +11/-0 (11 lines); hunks: -27,6 +27,11; -73,6 +78,9 @@ def _register_custom_configs_with_transformers() -> None:; symbols: _register_custom_configs_with_transformers，涉及 `_register_custom_configs_with_transformers`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_qwen4_exp.py` added +1291/-0 (1291 lines); hunks: -0,0 +1,1291; symbols: _qwen4_exp_tp_output_reduction_enabled, _validate_qwen4_exp_runtime_config, Qwen4ExpGatedDeltaNet, _postprocess_gdn_output
  - `tensorrt_llm/_torch/models/checkpoints/hf/qwen4_exp_weight_mapper.py` added +650/-0 (650 lines); hunks: -0,0 +1,650; symbols: _rank_block, _normalize_moe_module_weights, Qwen4ExpHfWeightMapper, should_skip_module
  - `tensorrt_llm/_torch/configs/qwen4_exp.py` added +201/-0 (201 lines); hunks: -0,0 +1,201; symbols: _flatten_qwen4_exp_rope, _normalize_qwen4_exp_layer_types, Qwen4ExpTextConfig, __init__
  - `tensorrt_llm/_torch/configs/__init__.py` modified +11/-0 (11 lines); hunks: -27,6 +27,11; -73,6 +78,9 @@ def _register_custom_configs_with_transformers() -> None:; symbols: _register_custom_configs_with_transformers
  - `tensorrt_llm/_torch/models/checkpoints/__init__.py` modified +6/-1 (7 lines); hunks: -1,3 +1,6; -16,6 +19,7
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen4_exp.py
@@ -0,0 +1,1291 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
diff -- tensorrt_llm/_torch/models/checkpoints/hf/qwen4_exp_weight_mapper.py
@@ -0,0 +1,650 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
diff -- tensorrt_llm/_torch/configs/qwen4_exp.py
@@ -0,0 +1,201 @@
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen4_exp.py` added +1291/-0; `tensorrt_llm/_torch/models/checkpoints/hf/qwen4_exp_weight_mapper.py` added +650/-0; `tensorrt_llm/_torch/configs/qwen4_exp.py` added +201/-0; `tensorrt_llm/_torch/configs/__init__.py` modified +11/-0; `tensorrt_llm/_torch/models/checkpoints/__init__.py` modified +6/-1; `tensorrt_llm/_torch/models/__init__.py` modified +2/-0
  - tests: `tests/unittest/_torch/modeling/test_qwen4_exp_support.py` added +1384/-0
- 验证与风险: diff 自带测试面 `tests/integration/defs/accuracy/references/gsm8k.yaml`, `tests/integration/defs/accuracy/references/mmlu.yaml`, `tests/integration/defs/accuracy/references/mmmu.yaml`, `tests/integration/defs/accuracy/test_llm_api_pytorch.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #18823 - [TRTLLM-16182][fix] load the mixed-precision Qwen3.8-Flash-Next NVFP4 checkpoint

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/18823
- 状态/时间: merged / 2026-09-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_qwen4_exp.py`, `tensorrt_llm/_torch/modules/qwen4_exp/ple.py`；关联提交 `e1e6bcca3b7c`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 11 个文件，+129/-59，可读 patch 367 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_qwen4_exp.py` modified +2/-1 (3 lines); hunks: -50,7 +50,7; -1119,6 +1119,7 @@ class Qwen4ExpForCausalLM(SpecDecOneEngineForCausalLM[Qwen...; symbols: Qwen4ExpForCausalLM, __init__，涉及 `Qwen4ExpForCausalLM, __init__`；`tensorrt_llm/_torch/modules/qwen4_exp/ple.py` modified +17/-5 (22 lines); hunks: -28,6 +28,8; -48,6 +50,8; symbols: _uses_ple_host_offload, _uses_scaled_fp8_ngram_table, _first_eos_token_id，涉及 `_uses_ple_host_offload, _uses_scaled_fp8_ngram_table, _first_eos_token_id`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_qwen4_exp.py` modified +2/-1 (3 lines); hunks: -50,7 +50,7; -1119,6 +1119,7 @@ class Qwen4ExpForCausalLM(SpecDecOneEngineForCausalLM[Qwen...; symbols: Qwen4ExpForCausalLM, __init__
  - `tensorrt_llm/_torch/modules/qwen4_exp/ple.py` modified +17/-5 (22 lines); hunks: -28,6 +28,8; -48,6 +50,8; symbols: _uses_ple_host_offload, _uses_scaled_fp8_ngram_table, _first_eos_token_id
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen4_exp.py
@@ -50,7 +50,7 @@
-from .modeling_qwen3_5 import _normalize_qwen35_exclude_modules
+from .modeling_qwen3_5 import _normalize_qwen35_exclude_modules, _normalize_qwen35_quant_config_dict
@@ -1119,6 +1119,7 @@ class Qwen4ExpForCausalLM(SpecDecOneEngineForCausalLM[Qwen4ExpModel, PretrainedC
+        _normalize_qwen35_quant_config_dict(model_config)
diff -- tensorrt_llm/_torch/modules/qwen4_exp/ple.py
@@ -28,6 +28,8 @@
+from tensorrt_llm.quantization.mode import QuantAlgo
+from tensorrt_llm.quantization.modelopt_config import canonicalize_quant_algo
@@ -48,6 +50,8 @@
+# Substring both quantization schemas use to name the PLE n-gram table.
+_NGRAM_TABLE_MARKER = "ple.ple_embedding.ngram_embedding"
@@ -65,13 +69,21 @@ def _uses_scaled_fp8_ngram_table(config: object) -> bool:
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen4_exp.py` modified +2/-1; `tensorrt_llm/_torch/modules/qwen4_exp/ple.py` modified +17/-5
- 验证与风险: diff 自带测试面 `tests/integration/defs/accuracy/references/gsm8k.yaml`, `tests/integration/defs/accuracy/references/mmlu.yaml`, `tests/integration/defs/accuracy/references/mmmu.yaml`, `tests/integration/defs/accuracy/test_llm_api_pytorch.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #18883 - [None][perf] fuse decode kernel seams for Qwen3.8-Flash-Next and de-vendor the low-M GEMMs

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/18883
- 状态/时间: merged / 2026-09-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tensorrt_llm/_torch/models/modeling_qwen4_exp.py`, `tensorrt_llm/_torch/modules/qwen4_exp/hyper_connection.py`, `tensorrt_llm/_torch/modules/qwen4_exp/hyper_connection_kernels.py`, `tests/unittest/_torch/modeling/test_qwen4_exp_support.py`, `tests/unittest/_torch/modules/test_qwen4_exp_hyper_connection.py`；关联提交 `f5fabd47d554`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 18 个文件，+878/-1538，可读 patch 2953 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tensorrt_llm/_torch/models/modeling_qwen4_exp.py` modified +29/-9 (38 lines); hunks: -15,7 +15,7; -51,7 +51,7; symbols: _qwen4_exp_tp_output_reduction_enabled, forward, skip_forward，涉及 `_qwen4_exp_tp_output_reduction_enabled, forward, skip_forward`；`tensorrt_llm/_torch/modules/qwen4_exp/hyper_connection_kernels.py` modified +188/-1 (189 lines); hunks: -446,4 +446,191 @@ def _(; symbols: _, _hc_moe_finalize_combine_norm_kernel, hc_moe_finalize_combine_norm，涉及 `_, _hc_moe_finalize_combine_norm_kernel, hc_moe_finalize_combine_norm`；`tests/unittest/_torch/modules/test_qwen4_exp_hyper_connection.py` modified +127/-1 (128 lines); hunks: -6,7 +6,17; -298,3 +308,119 @@ def test_combine_norm_rejects_a_reshaped_norm_weight():; symbols: _reference_mix, test_combine_norm_rejects_a_reshaped_norm_weight, test_deferred_moe_finalize_matches_staged_combine_norm_and_graph，涉及 `_reference_mix, test_combine_norm_rejects_a_reshaped_norm_weight, test_deferred_moe_finalize_matches_staged_combine_norm_and_graph`；`tests/unittest/_torch/modeling/test_qwen4_exp_support.py` modified +102/-0 (102 lines); hunks: -377,6 +377,108 @@ def test_text_model_is_eligible_for_online_eplb() -> None:; symbols: test_text_model_is_eligible_for_online_eplb, test_shared_expert_finalize_defers_only_when_no_collective_follows, finalize, _AllReduce，涉及 `test_text_model_is_eligible_for_online_eplb, test_shared_expert_finalize_defers_only_when_no_collective_follows, finalize`。
- 代码 diff 细节:
  - `tensorrt_llm/_torch/models/modeling_qwen4_exp.py` modified +29/-9 (38 lines); hunks: -15,7 +15,7; -51,7 +51,7; symbols: _qwen4_exp_tp_output_reduction_enabled, forward, skip_forward
  - `tensorrt_llm/_torch/modules/qwen4_exp/hyper_connection_kernels.py` modified +188/-1 (189 lines); hunks: -446,4 +446,191 @@ def _(; symbols: _, _hc_moe_finalize_combine_norm_kernel, hc_moe_finalize_combine_norm
  - `tests/unittest/_torch/modules/test_qwen4_exp_hyper_connection.py` modified +127/-1 (128 lines); hunks: -6,7 +6,17; -298,3 +308,119 @@ def test_combine_norm_rejects_a_reshaped_norm_weight():; symbols: _reference_mix, test_combine_norm_rejects_a_reshaped_norm_weight, test_deferred_moe_finalize_matches_staged_combine_norm_and_graph
  - `tests/unittest/_torch/modeling/test_qwen4_exp_support.py` modified +102/-0 (102 lines); hunks: -377,6 +377,108 @@ def test_text_model_is_eligible_for_online_eplb() -> None:; symbols: test_text_model_is_eligible_for_online_eplb, test_shared_expert_finalize_defers_only_when_no_collective_follows, finalize, _AllReduce
  - `tensorrt_llm/_torch/modules/qwen4_exp/hyper_connection.py` modified +78/-1 (79 lines); hunks: -12,12 +12,23; -393,6 +404,72 @@ def combine_and_mix(; symbols: GroupedRMSNorm, combine_and_mix, can_fuse_deferred_moe, combine_deferred_moe_and_mix
- 关键代码摘录:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen4_exp.py
@@ -15,7 +15,7 @@
-from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple
+from typing import TYPE_CHECKING, Dict, List, Literal, Optional
@@ -51,7 +51,7 @@
-from .modeling_qwen3_next import Qwen3NextSparseMoeBlock
+from .modeling_qwen3_next import Qwen3NextSparseMoeBlock, _DeferredSharedExpertFinalize
@@ -66,6 +66,9 @@
diff -- tensorrt_llm/_torch/modules/qwen4_exp/hyper_connection_kernels.py
@@ -446,4 +446,191 @@ def _(
-__all__ = ["hc_combine", "hc_combine_norm", "hc_gate_mix", "hc_silu"]
+@triton.jit
+def _hc_moe_finalize_combine_norm_kernel(
+    residual_ptr,
+    routed_ptr,
+    shared_ptr,
diff -- tests/unittest/_torch/modules/test_qwen4_exp_hyper_connection.py
@@ -6,7 +6,17 @@
```

- 提取文件（未人工审阅）:
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen4_exp.py` modified +29/-9; `tensorrt_llm/_torch/modules/qwen4_exp/hyper_connection_kernels.py` modified +188/-1; `tensorrt_llm/_torch/modules/qwen4_exp/hyper_connection.py` modified +78/-1
  - tests: `tests/unittest/_torch/modules/test_qwen4_exp_hyper_connection.py` modified +127/-1; `tests/unittest/_torch/modeling/test_qwen4_exp_support.py` modified +102/-0
- 验证与风险: diff 自带测试面 `tests/unittest/_torch/attention/sparse/qsa/test_qsa_sparse.py`, `tests/unittest/_torch/modeling/test_qsa_runtime_wiring.py`, `tests/unittest/_torch/modeling/test_qwen4_exp_support.py`, `tests/unittest/_torch/modules/test_low_m_gemm.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #18921 - [None][feat] support disaggregated serving for Qwen3.8-Flash-Next

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/18921
- 状态/时间: merged / 2026-09-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/unittest/_torch/modeling/test_qwen4_exp_support.py`；关联提交 `64fbd92b1071`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 32 个文件，+1840/-404，可读 patch 3190 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `tests/unittest/_torch/modeling/test_qwen4_exp_support.py` modified +23/-5 (28 lines); hunks: -532,13 +532,26 @@ def test_ple_cache_layout_excludes_separate_mtp_draft() ->...; -547,7 +560,13 @@ def test_v2_cache_estimator_counts_ple_lifecycle_state() ->...; symbols: test_ple_cache_layout_excludes_separate_mtp_draft, test_v2_cache_estimator_counts_ple_lifecycle_state，涉及 `test_ple_cache_layout_excludes_separate_mtp_draft, test_v2_cache_estimator_counts_ple_lifecycle_state`；`tensorrt_llm/_torch/disaggregation/resource/kv_extractor.py` modified +202/-212 (414 lines); hunks: -14,10 +14,9; -30,6 +29,7; symbols: KVRegionExtractorV1, __init__, _slot_layout, page_table，涉及 `KVRegionExtractorV1, __init__, _slot_layout`；`tensorrt_llm/_torch/disaggregation/native/mixers/ssm/peer.py` modified +135/-73 (208 lines); hunks: -28,6 +28,11; -343,9 +348,18 @@ class MambaPolicy:; symbols: MambaPolicy, __init__, should_send, build_mapper，涉及 `MambaPolicy, __init__, should_send`；`tensorrt_llm/_torch/disaggregation/resource/page.py` modified +66/-20 (86 lines); hunks: -17,14 +17,16; -64,9 +66,11 @@ class MapperKind(IntEnum):; symbols: MapperKind, RoleLayout, PhysicalPool, PoolView，涉及 `MapperKind, RoleLayout, PhysicalPool`。
- 代码 diff 细节:
  - `tests/unittest/_torch/modeling/test_qwen4_exp_support.py` modified +23/-5 (28 lines); hunks: -532,13 +532,26 @@ def test_ple_cache_layout_excludes_separate_mtp_draft() ->...; -547,7 +560,13 @@ def test_v2_cache_estimator_counts_ple_lifecycle_state() ->...; symbols: test_ple_cache_layout_excludes_separate_mtp_draft, test_v2_cache_estimator_counts_ple_lifecycle_state
  - `tensorrt_llm/_torch/disaggregation/resource/kv_extractor.py` modified +202/-212 (414 lines); hunks: -14,10 +14,9; -30,6 +29,7; symbols: KVRegionExtractorV1, __init__, _slot_layout, page_table
  - `tensorrt_llm/_torch/disaggregation/native/mixers/ssm/peer.py` modified +135/-73 (208 lines); hunks: -28,6 +28,11; -343,9 +348,18 @@ class MambaPolicy:; symbols: MambaPolicy, __init__, should_send, build_mapper
  - `tensorrt_llm/_torch/disaggregation/resource/page.py` modified +66/-20 (86 lines); hunks: -17,14 +17,16; -64,9 +66,11 @@ class MapperKind(IntEnum):; symbols: MapperKind, RoleLayout, PhysicalPool, PoolView
  - `tensorrt_llm/_torch/disaggregation/resource/utils.py` modified +46/-1 (47 lines); hunks: -15,14 +15,15; -156,6 +157,50 @@ def get_global_layer_ids(layer_group: AttentionLayerGroup)...; symbols: get_global_layer_ids, get_hosted_global_layer_ids, get_replicated_role_layers, find_replicated_role_mismatch
- 关键代码摘录:

```diff
diff -- tests/unittest/_torch/modeling/test_qwen4_exp_support.py
@@ -532,13 +532,26 @@ def test_ple_cache_layout_excludes_separate_mtp_draft() -> None:
-def test_v2_cache_estimator_counts_ple_lifecycle_state() -> None:
+@pytest.mark.parametrize(
+    "offsets_from_start,offsets_from_end,expected_state_slots",
+    [
+        # Two live request slots plus one non-speculative CUDA-graph dummy slot.
+        ([], [], 3),
diff -- tensorrt_llm/_torch/disaggregation/resource/kv_extractor.py
@@ -14,10 +14,9 @@
-from typing import Dict, List, Optional, Sequence
+from typing import Dict, List, Optional, Tuple
-import torch
@@ -30,6 +29,7 @@
+    CacheKind,
@@ -38,9 +38,11 @@
diff -- tensorrt_llm/_torch/disaggregation/native/mixers/ssm/peer.py
@@ -28,6 +28,11 @@
```

- 提取文件（未人工审阅）:
  - tests: `tests/unittest/_torch/modeling/test_qwen4_exp_support.py` modified +23/-5
  - runtime: `tensorrt_llm/_torch/disaggregation/resource/kv_extractor.py` modified +202/-212; `tensorrt_llm/_torch/disaggregation/native/mixers/ssm/peer.py` modified +135/-73; `tensorrt_llm/_torch/disaggregation/resource/page.py` modified +66/-20; `tensorrt_llm/_torch/disaggregation/resource/utils.py` modified +46/-1; `tensorrt_llm/_torch/disaggregation/native/peer.py` modified +20/-21; `tensorrt_llm/_torch/pyexecutor/kv_cache/mamba_cache_manager.py` modified +39/-0
- 验证与风险: diff 自带测试面 `tests/integration/defs/accuracy/test_disaggregated_serving.py`, `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/integration/test_lists/qa/llm_function_core.txt`, `tests/integration/test_lists/test-db/l0_gb300.yml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
