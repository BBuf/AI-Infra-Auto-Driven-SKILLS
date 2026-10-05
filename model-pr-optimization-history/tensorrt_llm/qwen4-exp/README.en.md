# TensorRT-LLM Qwen4-Exp (Qwen3.8-Flash-Next) Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
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

## PR Coverage Summary

- Git-traced PRs: 4
- Extra PRs preserved from existing docs: 0
- Total PRs in this document: 4
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-09-04 | [#18585](https://github.com/NVIDIA/TensorRT-LLM/pull/18585) | merged | [None][feat] add Qwen3.8-Flash-Next support | `tensorrt_llm/_torch/models/modeling_qwen4_exp.py`, `tensorrt_llm/_torch/models/checkpoints/hf/qwen4_exp_weight_mapper.py`, `tensorrt_llm/_torch/configs/qwen4_exp.py` |
| 2026-09-09 | [#18823](https://github.com/NVIDIA/TensorRT-LLM/pull/18823) | merged | [TRTLLM-16182][fix] load the mixed-precision Qwen3.8-Flash-Next NVFP4 checkpoint | `tensorrt_llm/_torch/models/modeling_qwen4_exp.py`, `tensorrt_llm/_torch/modules/qwen4_exp/ple.py` |
| 2026-09-10 | [#18883](https://github.com/NVIDIA/TensorRT-LLM/pull/18883) | merged | [None][perf] fuse decode kernel seams for Qwen3.8-Flash-Next and de-vendor the low-M GEMMs | `tensorrt_llm/_torch/models/modeling_qwen4_exp.py`, `tensorrt_llm/_torch/modules/qwen4_exp/hyper_connection_kernels.py`, `tests/unittest/_torch/modules/test_qwen4_exp_hyper_connection.py` |
| 2026-09-15 | [#18921](https://github.com/NVIDIA/TensorRT-LLM/pull/18921) | merged | [None][feat] support disaggregated serving for Qwen3.8-Flash-Next | `tests/unittest/_torch/modeling/test_qwen4_exp_support.py`, `tensorrt_llm/_torch/disaggregation/resource/kv_extractor.py`, `tensorrt_llm/_torch/disaggregation/native/mixers/ssm/peer.py` |

## Per-PR Diff Audit Cards

### PR #18585 - [None][feat] add Qwen3.8-Flash-Next support

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/18585
- Status/date: merged / 2026-09-04
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/configs/qwen4_exp.py`, `tensorrt_llm/_torch/models/checkpoints/hf/qwen4_exp_weight_mapper.py`, `tensorrt_llm/_torch/models/modeling_qwen4_exp.py`, `tensorrt_llm/_torch/modules/qwen4_exp/__init__.py`, `tensorrt_llm/_torch/modules/qwen4_exp/hyper_connection.py` and 13 files; associated commits `02746c52d0b1`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 62 files, +14598/-43, 15436 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_qwen4_exp.py` added +1291/-0 (1291 lines); hunks: -0,0 +1,1291; symbols: _qwen4_exp_tp_output_reduction_enabled, _validate_qwen4_exp_runtime_config, Qwen4ExpGatedDeltaNet, _postprocess_gdn_output, touching `_qwen4_exp_tp_output_reduction_enabled, _validate_qwen4_exp_runtime_config, Qwen4ExpGatedDeltaNet`; `tensorrt_llm/_torch/models/checkpoints/hf/qwen4_exp_weight_mapper.py` added +650/-0 (650 lines); hunks: -0,0 +1,650; symbols: _rank_block, _normalize_moe_module_weights, Qwen4ExpHfWeightMapper, should_skip_module, touching `_rank_block, _normalize_moe_module_weights, Qwen4ExpHfWeightMapper`; `tensorrt_llm/_torch/configs/qwen4_exp.py` added +201/-0 (201 lines); hunks: -0,0 +1,201; symbols: _flatten_qwen4_exp_rope, _normalize_qwen4_exp_layer_types, Qwen4ExpTextConfig, __init__, touching `_flatten_qwen4_exp_rope, _normalize_qwen4_exp_layer_types, Qwen4ExpTextConfig`; `tensorrt_llm/_torch/configs/__init__.py` modified +11/-0 (11 lines); hunks: -27,6 +27,11; -73,6 +78,9 @@ def _register_custom_configs_with_transformers() -> None:; symbols: _register_custom_configs_with_transformers, touching `_register_custom_configs_with_transformers`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_qwen4_exp.py` added +1291/-0 (1291 lines); hunks: -0,0 +1,1291; symbols: _qwen4_exp_tp_output_reduction_enabled, _validate_qwen4_exp_runtime_config, Qwen4ExpGatedDeltaNet, _postprocess_gdn_output
  - `tensorrt_llm/_torch/models/checkpoints/hf/qwen4_exp_weight_mapper.py` added +650/-0 (650 lines); hunks: -0,0 +1,650; symbols: _rank_block, _normalize_moe_module_weights, Qwen4ExpHfWeightMapper, should_skip_module
  - `tensorrt_llm/_torch/configs/qwen4_exp.py` added +201/-0 (201 lines); hunks: -0,0 +1,201; symbols: _flatten_qwen4_exp_rope, _normalize_qwen4_exp_layer_types, Qwen4ExpTextConfig, __init__
  - `tensorrt_llm/_torch/configs/__init__.py` modified +11/-0 (11 lines); hunks: -27,6 +27,11; -73,6 +78,9 @@ def _register_custom_configs_with_transformers() -> None:; symbols: _register_custom_configs_with_transformers
  - `tensorrt_llm/_torch/models/checkpoints/__init__.py` modified +6/-1 (7 lines); hunks: -1,3 +1,6; -16,6 +19,7
- Key code excerpts:

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

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen4_exp.py` added +1291/-0; `tensorrt_llm/_torch/models/checkpoints/hf/qwen4_exp_weight_mapper.py` added +650/-0; `tensorrt_llm/_torch/configs/qwen4_exp.py` added +201/-0; `tensorrt_llm/_torch/configs/__init__.py` modified +11/-0; `tensorrt_llm/_torch/models/checkpoints/__init__.py` modified +6/-1; `tensorrt_llm/_torch/models/__init__.py` modified +2/-0
  - tests: `tests/unittest/_torch/modeling/test_qwen4_exp_support.py` added +1384/-0
- Risk and verification: The diff ships test coverage in `tests/integration/defs/accuracy/references/gsm8k.yaml`, `tests/integration/defs/accuracy/references/mmlu.yaml`, `tests/integration/defs/accuracy/references/mmmu.yaml`, `tests/integration/defs/accuracy/test_llm_api_pytorch.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18823 - [TRTLLM-16182][fix] load the mixed-precision Qwen3.8-Flash-Next NVFP4 checkpoint

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/18823
- Status/date: merged / 2026-09-09
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_qwen4_exp.py`, `tensorrt_llm/_torch/modules/qwen4_exp/ple.py`; associated commits `e1e6bcca3b7c`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 11 files, +129/-59, 367 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_qwen4_exp.py` modified +2/-1 (3 lines); hunks: -50,7 +50,7; -1119,6 +1119,7 @@ class Qwen4ExpForCausalLM(SpecDecOneEngineForCausalLM[Qwen...; symbols: Qwen4ExpForCausalLM, __init__, touching `Qwen4ExpForCausalLM, __init__`; `tensorrt_llm/_torch/modules/qwen4_exp/ple.py` modified +17/-5 (22 lines); hunks: -28,6 +28,8; -48,6 +50,8; symbols: _uses_ple_host_offload, _uses_scaled_fp8_ngram_table, _first_eos_token_id, touching `_uses_ple_host_offload, _uses_scaled_fp8_ngram_table, _first_eos_token_id`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_qwen4_exp.py` modified +2/-1 (3 lines); hunks: -50,7 +50,7; -1119,6 +1119,7 @@ class Qwen4ExpForCausalLM(SpecDecOneEngineForCausalLM[Qwen...; symbols: Qwen4ExpForCausalLM, __init__
  - `tensorrt_llm/_torch/modules/qwen4_exp/ple.py` modified +17/-5 (22 lines); hunks: -28,6 +28,8; -48,6 +50,8; symbols: _uses_ple_host_offload, _uses_scaled_fp8_ngram_table, _first_eos_token_id
- Key code excerpts:

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

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen4_exp.py` modified +2/-1; `tensorrt_llm/_torch/modules/qwen4_exp/ple.py` modified +17/-5
- Risk and verification: The diff ships test coverage in `tests/integration/defs/accuracy/references/gsm8k.yaml`, `tests/integration/defs/accuracy/references/mmlu.yaml`, `tests/integration/defs/accuracy/references/mmmu.yaml`, `tests/integration/defs/accuracy/test_llm_api_pytorch.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18883 - [None][perf] fuse decode kernel seams for Qwen3.8-Flash-Next and de-vendor the low-M GEMMs

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/18883
- Status/date: merged / 2026-09-10
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_qwen4_exp.py`, `tensorrt_llm/_torch/modules/qwen4_exp/hyper_connection.py`, `tensorrt_llm/_torch/modules/qwen4_exp/hyper_connection_kernels.py`, `tests/unittest/_torch/modeling/test_qwen4_exp_support.py`, `tests/unittest/_torch/modules/test_qwen4_exp_hyper_connection.py`; associated commits `f5fabd47d554`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 18 files, +878/-1538, 2953 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_qwen4_exp.py` modified +29/-9 (38 lines); hunks: -15,7 +15,7; -51,7 +51,7; symbols: _qwen4_exp_tp_output_reduction_enabled, forward, skip_forward, touching `_qwen4_exp_tp_output_reduction_enabled, forward, skip_forward`; `tensorrt_llm/_torch/modules/qwen4_exp/hyper_connection_kernels.py` modified +188/-1 (189 lines); hunks: -446,4 +446,191 @@ def _(; symbols: _, _hc_moe_finalize_combine_norm_kernel, hc_moe_finalize_combine_norm, touching `_, _hc_moe_finalize_combine_norm_kernel, hc_moe_finalize_combine_norm`; `tests/unittest/_torch/modules/test_qwen4_exp_hyper_connection.py` modified +127/-1 (128 lines); hunks: -6,7 +6,17; -298,3 +308,119 @@ def test_combine_norm_rejects_a_reshaped_norm_weight():; symbols: _reference_mix, test_combine_norm_rejects_a_reshaped_norm_weight, test_deferred_moe_finalize_matches_staged_combine_norm_and_graph, touching `_reference_mix, test_combine_norm_rejects_a_reshaped_norm_weight, test_deferred_moe_finalize_matches_staged_combine_norm_and_graph`; `tests/unittest/_torch/modeling/test_qwen4_exp_support.py` modified +102/-0 (102 lines); hunks: -377,6 +377,108 @@ def test_text_model_is_eligible_for_online_eplb() -> None:; symbols: test_text_model_is_eligible_for_online_eplb, test_shared_expert_finalize_defers_only_when_no_collective_follows, finalize, _AllReduce, touching `test_text_model_is_eligible_for_online_eplb, test_shared_expert_finalize_defers_only_when_no_collective_follows, finalize`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_qwen4_exp.py` modified +29/-9 (38 lines); hunks: -15,7 +15,7; -51,7 +51,7; symbols: _qwen4_exp_tp_output_reduction_enabled, forward, skip_forward
  - `tensorrt_llm/_torch/modules/qwen4_exp/hyper_connection_kernels.py` modified +188/-1 (189 lines); hunks: -446,4 +446,191 @@ def _(; symbols: _, _hc_moe_finalize_combine_norm_kernel, hc_moe_finalize_combine_norm
  - `tests/unittest/_torch/modules/test_qwen4_exp_hyper_connection.py` modified +127/-1 (128 lines); hunks: -6,7 +6,17; -298,3 +308,119 @@ def test_combine_norm_rejects_a_reshaped_norm_weight():; symbols: _reference_mix, test_combine_norm_rejects_a_reshaped_norm_weight, test_deferred_moe_finalize_matches_staged_combine_norm_and_graph
  - `tests/unittest/_torch/modeling/test_qwen4_exp_support.py` modified +102/-0 (102 lines); hunks: -377,6 +377,108 @@ def test_text_model_is_eligible_for_online_eplb() -> None:; symbols: test_text_model_is_eligible_for_online_eplb, test_shared_expert_finalize_defers_only_when_no_collective_follows, finalize, _AllReduce
  - `tensorrt_llm/_torch/modules/qwen4_exp/hyper_connection.py` modified +78/-1 (79 lines); hunks: -12,12 +12,23; -393,6 +404,72 @@ def combine_and_mix(; symbols: GroupedRMSNorm, combine_and_mix, can_fuse_deferred_moe, combine_deferred_moe_and_mix
- Key code excerpts:

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

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen4_exp.py` modified +29/-9; `tensorrt_llm/_torch/modules/qwen4_exp/hyper_connection_kernels.py` modified +188/-1; `tensorrt_llm/_torch/modules/qwen4_exp/hyper_connection.py` modified +78/-1
  - tests: `tests/unittest/_torch/modules/test_qwen4_exp_hyper_connection.py` modified +127/-1; `tests/unittest/_torch/modeling/test_qwen4_exp_support.py` modified +102/-0
- Risk and verification: The diff ships test coverage in `tests/unittest/_torch/attention/sparse/qsa/test_qsa_sparse.py`, `tests/unittest/_torch/modeling/test_qsa_runtime_wiring.py`, `tests/unittest/_torch/modeling/test_qwen4_exp_support.py`, `tests/unittest/_torch/modules/test_low_m_gemm.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18921 - [None][feat] support disaggregated serving for Qwen3.8-Flash-Next

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/18921
- Status/date: merged / 2026-09-15
- Trace source: `git log --name-only -- <model-files>` found it through `tests/unittest/_torch/modeling/test_qwen4_exp_support.py`; associated commits `64fbd92b1071`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 32 files, +1840/-404, 3190 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/unittest/_torch/modeling/test_qwen4_exp_support.py` modified +23/-5 (28 lines); hunks: -532,13 +532,26 @@ def test_ple_cache_layout_excludes_separate_mtp_draft() ->...; -547,7 +560,13 @@ def test_v2_cache_estimator_counts_ple_lifecycle_state() ->...; symbols: test_ple_cache_layout_excludes_separate_mtp_draft, test_v2_cache_estimator_counts_ple_lifecycle_state, touching `test_ple_cache_layout_excludes_separate_mtp_draft, test_v2_cache_estimator_counts_ple_lifecycle_state`; `tensorrt_llm/_torch/disaggregation/resource/kv_extractor.py` modified +202/-212 (414 lines); hunks: -14,10 +14,9; -30,6 +29,7; symbols: KVRegionExtractorV1, __init__, _slot_layout, page_table, touching `KVRegionExtractorV1, __init__, _slot_layout`; `tensorrt_llm/_torch/disaggregation/native/mixers/ssm/peer.py` modified +135/-73 (208 lines); hunks: -28,6 +28,11; -343,9 +348,18 @@ class MambaPolicy:; symbols: MambaPolicy, __init__, should_send, build_mapper, touching `MambaPolicy, __init__, should_send`; `tensorrt_llm/_torch/disaggregation/resource/page.py` modified +66/-20 (86 lines); hunks: -17,14 +17,16; -64,9 +66,11 @@ class MapperKind(IntEnum):; symbols: MapperKind, RoleLayout, PhysicalPool, PoolView, touching `MapperKind, RoleLayout, PhysicalPool`.
- Code diff details:
  - `tests/unittest/_torch/modeling/test_qwen4_exp_support.py` modified +23/-5 (28 lines); hunks: -532,13 +532,26 @@ def test_ple_cache_layout_excludes_separate_mtp_draft() ->...; -547,7 +560,13 @@ def test_v2_cache_estimator_counts_ple_lifecycle_state() ->...; symbols: test_ple_cache_layout_excludes_separate_mtp_draft, test_v2_cache_estimator_counts_ple_lifecycle_state
  - `tensorrt_llm/_torch/disaggregation/resource/kv_extractor.py` modified +202/-212 (414 lines); hunks: -14,10 +14,9; -30,6 +29,7; symbols: KVRegionExtractorV1, __init__, _slot_layout, page_table
  - `tensorrt_llm/_torch/disaggregation/native/mixers/ssm/peer.py` modified +135/-73 (208 lines); hunks: -28,6 +28,11; -343,9 +348,18 @@ class MambaPolicy:; symbols: MambaPolicy, __init__, should_send, build_mapper
  - `tensorrt_llm/_torch/disaggregation/resource/page.py` modified +66/-20 (86 lines); hunks: -17,14 +17,16; -64,9 +66,11 @@ class MapperKind(IntEnum):; symbols: MapperKind, RoleLayout, PhysicalPool, PoolView
  - `tensorrt_llm/_torch/disaggregation/resource/utils.py` modified +46/-1 (47 lines); hunks: -15,14 +15,15; -156,6 +157,50 @@ def get_global_layer_ids(layer_group: AttentionLayerGroup)...; symbols: get_global_layer_ids, get_hosted_global_layer_ids, get_replicated_role_layers, find_replicated_role_mismatch
- Key code excerpts:

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

- Extracted files (not manually reviewed):
  - tests: `tests/unittest/_torch/modeling/test_qwen4_exp_support.py` modified +23/-5
  - runtime: `tensorrt_llm/_torch/disaggregation/resource/kv_extractor.py` modified +202/-212; `tensorrt_llm/_torch/disaggregation/native/mixers/ssm/peer.py` modified +135/-73; `tensorrt_llm/_torch/disaggregation/resource/page.py` modified +66/-20; `tensorrt_llm/_torch/disaggregation/resource/utils.py` modified +46/-1; `tensorrt_llm/_torch/disaggregation/native/peer.py` modified +20/-21; `tensorrt_llm/_torch/pyexecutor/kv_cache/mamba_cache_manager.py` modified +39/-0
- Risk and verification: The diff ships test coverage in `tests/integration/defs/accuracy/test_disaggregated_serving.py`, `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/integration/test_lists/qa/llm_function_core.txt`, `tests/integration/test_lists/test-db/l0_gb300.yml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
