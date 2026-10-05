# TokenSpeed Nemotron Super Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `python/tokenspeed/runtime/configs/nemotron_h_config.py` | [#1918](https://github.com/lightseekorg/tokenspeed/pull/1918) |
| `python/tokenspeed/runtime/models/nemotron_h.py` | [#1918](https://github.com/lightseekorg/tokenspeed/pull/1918), [#1965](https://github.com/lightseekorg/tokenspeed/pull/1965) |
| `python/tokenspeed/runtime/models/nemotron_h_nextn.py` | [#1918](https://github.com/lightseekorg/tokenspeed/pull/1918), [#1965](https://github.com/lightseekorg/tokenspeed/pull/1965) |
| `test/runtime/test_nemotron_h.py` | [#1918](https://github.com/lightseekorg/tokenspeed/pull/1918), [#1965](https://github.com/lightseekorg/tokenspeed/pull/1965) |

## PR Coverage Summary

- Git-traced PRs: 2
- Extra PRs preserved from existing docs: 0
- Total PRs in this document: 2
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-10-02 | [#1918](https://github.com/lightseekorg/tokenspeed/pull/1918) | merged | feat(model): support nemotron-3-super nvfp4 | `python/tokenspeed/runtime/models/nemotron_h.py`, `python/tokenspeed/runtime/models/nemotron_h_nextn.py`, `python/tokenspeed/runtime/configs/nemotron_h_config.py` |
| 2026-10-04 | [#1965](https://github.com/lightseekorg/tokenspeed/pull/1965) | merged | perf(nemotron): trim launches and host time from Nemotron-3 Super serving | `python/tokenspeed/runtime/models/nemotron_h.py`, `python/tokenspeed/runtime/models/nemotron_h_nextn.py`, `test/runtime/test_nemotron_h.py` |

## Per-PR Diff Audit Cards

### PR #1918 - feat(model): support nemotron-3-super nvfp4

- Link: https://github.com/lightseekorg/tokenspeed/pull/1918
- Status/date: merged / 2026-10-02
- Trace source: `git log --name-only -- <model-files>` found it through `python/tokenspeed/runtime/configs/nemotron_h_config.py`, `python/tokenspeed/runtime/models/nemotron_h.py`, `python/tokenspeed/runtime/models/nemotron_h_nextn.py`, `test/runtime/test_nemotron_h.py`; associated commits `52f9151305b1`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 52 files, +6883/-148, 8044 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/tokenspeed/runtime/models/nemotron_h.py` added +815/-0 (815 lines); hunks: -0,0 +1,815; symbols: _static_fp8_scale, _fc2_reduce_group, require_single_tp_group, expert_checkpoint_loader, touching `_static_fp8_scale, _fc2_reduce_group, require_single_tp_group`; `python/tokenspeed/runtime/models/nemotron_h_nextn.py` added +229/-0 (229 lines); hunks: -0,0 +1,229; symbols: _mtp_param_name, NemotronHForCausalLMNextN, __init__, get_hot_token_id, touching `_mtp_param_name, NemotronHForCausalLMNextN, __init__`; `python/tokenspeed/runtime/configs/nemotron_h_config.py` added +85/-0 (85 lines); hunks: -0,0 +1,85; symbols: NemotronHConfig, cache_layer_ids, cache_layer_types, linear_layer_ids, touching `NemotronHConfig, cache_layer_ids, cache_layer_types`; `test/runtime/test_nemotron_h.py` added +1064/-0 (1064 lines); hunks: -0,0 +1,1064; symbols: _config, test_cache_layers_are_the_mamba_and_attention_blocks_in_order, test_mamba2_component_maps_ssd_geometry_onto_the_linear_fields, test_non_gated_expert_plan_loads_up_proj_as_all_of_w13, touching `_config, test_cache_layers_are_the_mamba_and_attention_blocks_in_order, test_mamba2_component_maps_ssd_geometry_onto_the_linear_fields`.
- Code diff details:
  - `python/tokenspeed/runtime/models/nemotron_h.py` added +815/-0 (815 lines); hunks: -0,0 +1,815; symbols: _static_fp8_scale, _fc2_reduce_group, require_single_tp_group, expert_checkpoint_loader
  - `python/tokenspeed/runtime/models/nemotron_h_nextn.py` added +229/-0 (229 lines); hunks: -0,0 +1,229; symbols: _mtp_param_name, NemotronHForCausalLMNextN, __init__, get_hot_token_id
  - `python/tokenspeed/runtime/configs/nemotron_h_config.py` added +85/-0 (85 lines); hunks: -0,0 +1,85; symbols: NemotronHConfig, cache_layer_ids, cache_layer_types, linear_layer_ids
  - `test/runtime/test_nemotron_h.py` added +1064/-0 (1064 lines); hunks: -0,0 +1,1064; symbols: _config, test_cache_layers_are_the_mamba_and_attention_blocks_in_order, test_mamba2_component_maps_ssd_geometry_onto_the_linear_fields, test_non_gated_expert_plan_loads_up_proj_as_all_of_w13
- Key code excerpts:

```diff
diff -- python/tokenspeed/runtime/models/nemotron_h.py
@@ -0,0 +1,815 @@
+# Copyright (c) 2026 LightSeek Foundation
+#
+# Permission is hereby granted, free of charge, to any person obtaining a copy
+# of this software and associated documentation files (the "Software"), to deal
+# in the Software without restriction, including without limitation the rights
+# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
diff -- python/tokenspeed/runtime/models/nemotron_h_nextn.py
@@ -0,0 +1,229 @@
+# Copyright (c) 2026 LightSeek Foundation
+#
+# Permission is hereby granted, free of charge, to any person obtaining a copy
+# of this software and associated documentation files (the "Software"), to deal
+# in the Software without restriction, including without limitation the rights
+# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
diff -- python/tokenspeed/runtime/configs/nemotron_h_config.py
@@ -0,0 +1,85 @@
```

- Extracted files (not manually reviewed):
  - runtime: `python/tokenspeed/runtime/models/nemotron_h.py` added +815/-0; `python/tokenspeed/runtime/models/nemotron_h_nextn.py` added +229/-0; `python/tokenspeed/runtime/configs/nemotron_h_config.py` added +85/-0
  - tests: `test/runtime/test_nemotron_h.py` added +1064/-0
- Risk and verification: The diff ships test coverage in `test/runtime/test_cudagraph_probe_arena.py`, `test/runtime/test_gdn_state_paging.py`, `test/runtime/test_kimi_k3_kda.py`, `test/runtime/test_moe_ispp_padding.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #1965 - perf(nemotron): trim launches and host time from Nemotron-3 Super serving

- Link: https://github.com/lightseekorg/tokenspeed/pull/1965
- Status/date: merged / 2026-10-04
- Trace source: `git log --name-only -- <model-files>` found it through `python/tokenspeed/runtime/models/nemotron_h.py`, `python/tokenspeed/runtime/models/nemotron_h_nextn.py`, `test/runtime/test_nemotron_h.py`; associated commits `45db70ffef47`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 23 files, +932/-110, 1589 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/tokenspeed/runtime/models/nemotron_h.py` modified +34/-8 (42 lines); hunks: -44,6 +44,10; -76,6 +80,7; symbols: __init__, input_fp8_scale, forward, _routed, touching `__init__, input_fp8_scale, forward`; `python/tokenspeed/runtime/models/nemotron_h_nextn.py` modified +2/-0 (2 lines); hunks: -123,6 +123,7 @@ def __init__(; -132,6 +133,7 @@ def __init__(; symbols: __init__, touching `__init__`; `test/runtime/test_nemotron_h.py` modified +34/-0 (34 lines); hunks: -709,6 +709,40 @@ def test_quantized_fc2_sees_the_reduced_latent_and_counts_o...; symbols: test_quantized_fc2_sees_the_reduced_latent_and_counts_once, test_router_and_shared_expert_run_beside_fc1_only_in_decode_graphs, record, touching `test_quantized_fc2_sees_the_reduced_latent_and_counts_once, test_router_and_shared_expert_run_beside_fc1_only_in_decode_graphs, record`.
- Code diff details:
  - `python/tokenspeed/runtime/models/nemotron_h.py` modified +34/-8 (42 lines); hunks: -44,6 +44,10; -76,6 +80,7; symbols: __init__, input_fp8_scale, forward, _routed
  - `python/tokenspeed/runtime/models/nemotron_h_nextn.py` modified +2/-0 (2 lines); hunks: -123,6 +123,7 @@ def __init__(; -132,6 +133,7 @@ def __init__(; symbols: __init__
  - `test/runtime/test_nemotron_h.py` modified +34/-0 (34 lines); hunks: -709,6 +709,40 @@ def test_quantized_fc2_sees_the_reduced_latent_and_counts_o...; symbols: test_quantized_fc2_sees_the_reduced_latent_and_counts_once, test_router_and_shared_expert_run_beside_fc1_only_in_decode_graphs, record
- Key code excerpts:

```diff
diff -- python/tokenspeed/runtime/models/nemotron_h.py
@@ -44,6 +44,10 @@
+from tokenspeed.runtime.execution.forward_step import (
+    get_is_capture_mode,
+    get_is_cuda_graph_phase,
+)
@@ -76,6 +80,7 @@
+from tokenspeed.runtime.utils.cuda_stream import StreamFork
diff -- python/tokenspeed/runtime/models/nemotron_h_nextn.py
@@ -123,6 +123,7 @@ def __init__(
+        alt_stream = torch.cuda.Stream()
@@ -132,6 +133,7 @@ def __init__(
+                alt_stream,
diff -- test/runtime/test_nemotron_h.py
@@ -709,6 +709,40 @@ def test_quantized_fc2_sees_the_reduced_latent_and_counts_once(
+@pytest.mark.parametrize("graph_phase,capture", [(False, False), (True, True)])
+def test_router_and_shared_expert_run_beside_fc1_only_in_decode_graphs(
+    monkeypatch: pytest.MonkeyPatch, graph_phase: bool, capture: bool
```

- Extracted files (not manually reviewed):
  - runtime: `python/tokenspeed/runtime/models/nemotron_h.py` modified +34/-8; `python/tokenspeed/runtime/models/nemotron_h_nextn.py` modified +2/-0
  - tests: `test/runtime/test_nemotron_h.py` modified +34/-0
- Risk and verification: The diff ships test coverage in `test/runtime/layers/test_gdn_qkv_split_fused.py`, `test/runtime/test_gdn_state_paging.py`, `test/runtime/test_nemotron_h.py`, `tokenspeed-kernel/test/nvidia/ops/gemm/test_fp8_skinny_m1.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
