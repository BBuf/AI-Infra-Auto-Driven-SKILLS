# TokenSpeed Nemotron Super 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `python/tokenspeed/runtime/configs/nemotron_h_config.py` | [#1918](https://github.com/lightseekorg/tokenspeed/pull/1918) |
| `python/tokenspeed/runtime/models/nemotron_h.py` | [#1918](https://github.com/lightseekorg/tokenspeed/pull/1918), [#1965](https://github.com/lightseekorg/tokenspeed/pull/1965) |
| `python/tokenspeed/runtime/models/nemotron_h_nextn.py` | [#1918](https://github.com/lightseekorg/tokenspeed/pull/1918), [#1965](https://github.com/lightseekorg/tokenspeed/pull/1965) |
| `test/runtime/test_nemotron_h.py` | [#1918](https://github.com/lightseekorg/tokenspeed/pull/1918), [#1965](https://github.com/lightseekorg/tokenspeed/pull/1965) |

## PR 覆盖总览

- git 追溯 PR 数: 2
- 原文档显式引用补充 PR 数: 0
- 当前文档总 PR 数: 2
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-10-02 | [#1918](https://github.com/lightseekorg/tokenspeed/pull/1918) | merged | feat(model): support nemotron-3-super nvfp4 | `python/tokenspeed/runtime/models/nemotron_h.py`, `python/tokenspeed/runtime/models/nemotron_h_nextn.py`, `python/tokenspeed/runtime/configs/nemotron_h_config.py` |
| 2026-10-04 | [#1965](https://github.com/lightseekorg/tokenspeed/pull/1965) | merged | perf(nemotron): trim launches and host time from Nemotron-3 Super serving | `python/tokenspeed/runtime/models/nemotron_h.py`, `python/tokenspeed/runtime/models/nemotron_h_nextn.py`, `test/runtime/test_nemotron_h.py` |

## 逐 PR diff 审计卡

### PR #1918 - feat(model): support nemotron-3-super nvfp4

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1918
- 状态/时间: merged / 2026-10-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/configs/nemotron_h_config.py`, `python/tokenspeed/runtime/models/nemotron_h.py`, `python/tokenspeed/runtime/models/nemotron_h_nextn.py`, `test/runtime/test_nemotron_h.py`；关联提交 `52f9151305b1`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 52 个文件，+6883/-148，可读 patch 8044 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/nemotron_h.py` added +815/-0 (815 lines); hunks: -0,0 +1,815; symbols: _static_fp8_scale, _fc2_reduce_group, require_single_tp_group, expert_checkpoint_loader，涉及 `_static_fp8_scale, _fc2_reduce_group, require_single_tp_group`；`python/tokenspeed/runtime/models/nemotron_h_nextn.py` added +229/-0 (229 lines); hunks: -0,0 +1,229; symbols: _mtp_param_name, NemotronHForCausalLMNextN, __init__, get_hot_token_id，涉及 `_mtp_param_name, NemotronHForCausalLMNextN, __init__`；`python/tokenspeed/runtime/configs/nemotron_h_config.py` added +85/-0 (85 lines); hunks: -0,0 +1,85; symbols: NemotronHConfig, cache_layer_ids, cache_layer_types, linear_layer_ids，涉及 `NemotronHConfig, cache_layer_ids, cache_layer_types`；`test/runtime/test_nemotron_h.py` added +1064/-0 (1064 lines); hunks: -0,0 +1,1064; symbols: _config, test_cache_layers_are_the_mamba_and_attention_blocks_in_order, test_mamba2_component_maps_ssd_geometry_onto_the_linear_fields, test_non_gated_expert_plan_loads_up_proj_as_all_of_w13，涉及 `_config, test_cache_layers_are_the_mamba_and_attention_blocks_in_order, test_mamba2_component_maps_ssd_geometry_onto_the_linear_fields`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/nemotron_h.py` added +815/-0 (815 lines); hunks: -0,0 +1,815; symbols: _static_fp8_scale, _fc2_reduce_group, require_single_tp_group, expert_checkpoint_loader
  - `python/tokenspeed/runtime/models/nemotron_h_nextn.py` added +229/-0 (229 lines); hunks: -0,0 +1,229; symbols: _mtp_param_name, NemotronHForCausalLMNextN, __init__, get_hot_token_id
  - `python/tokenspeed/runtime/configs/nemotron_h_config.py` added +85/-0 (85 lines); hunks: -0,0 +1,85; symbols: NemotronHConfig, cache_layer_ids, cache_layer_types, linear_layer_ids
  - `test/runtime/test_nemotron_h.py` added +1064/-0 (1064 lines); hunks: -0,0 +1,1064; symbols: _config, test_cache_layers_are_the_mamba_and_attention_blocks_in_order, test_mamba2_component_maps_ssd_geometry_onto_the_linear_fields, test_non_gated_expert_plan_loads_up_proj_as_all_of_w13
- 关键代码摘录:

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

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/nemotron_h.py` added +815/-0; `python/tokenspeed/runtime/models/nemotron_h_nextn.py` added +229/-0; `python/tokenspeed/runtime/configs/nemotron_h_config.py` added +85/-0
  - tests: `test/runtime/test_nemotron_h.py` added +1064/-0
- 验证与风险: diff 自带测试面 `test/runtime/test_cudagraph_probe_arena.py`, `test/runtime/test_gdn_state_paging.py`, `test/runtime/test_kimi_k3_kda.py`, `test/runtime/test_moe_ispp_padding.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1965 - perf(nemotron): trim launches and host time from Nemotron-3 Super serving

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1965
- 状态/时间: merged / 2026-10-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/nemotron_h.py`, `python/tokenspeed/runtime/models/nemotron_h_nextn.py`, `test/runtime/test_nemotron_h.py`；关联提交 `45db70ffef47`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 23 个文件，+932/-110，可读 patch 1589 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/nemotron_h.py` modified +34/-8 (42 lines); hunks: -44,6 +44,10; -76,6 +80,7; symbols: __init__, input_fp8_scale, forward, _routed，涉及 `__init__, input_fp8_scale, forward`；`python/tokenspeed/runtime/models/nemotron_h_nextn.py` modified +2/-0 (2 lines); hunks: -123,6 +123,7 @@ def __init__(; -132,6 +133,7 @@ def __init__(; symbols: __init__，涉及 `__init__`；`test/runtime/test_nemotron_h.py` modified +34/-0 (34 lines); hunks: -709,6 +709,40 @@ def test_quantized_fc2_sees_the_reduced_latent_and_counts_o...; symbols: test_quantized_fc2_sees_the_reduced_latent_and_counts_once, test_router_and_shared_expert_run_beside_fc1_only_in_decode_graphs, record，涉及 `test_quantized_fc2_sees_the_reduced_latent_and_counts_once, test_router_and_shared_expert_run_beside_fc1_only_in_decode_graphs, record`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/nemotron_h.py` modified +34/-8 (42 lines); hunks: -44,6 +44,10; -76,6 +80,7; symbols: __init__, input_fp8_scale, forward, _routed
  - `python/tokenspeed/runtime/models/nemotron_h_nextn.py` modified +2/-0 (2 lines); hunks: -123,6 +123,7 @@ def __init__(; -132,6 +133,7 @@ def __init__(; symbols: __init__
  - `test/runtime/test_nemotron_h.py` modified +34/-0 (34 lines); hunks: -709,6 +709,40 @@ def test_quantized_fc2_sees_the_reduced_latent_and_counts_o...; symbols: test_quantized_fc2_sees_the_reduced_latent_and_counts_once, test_router_and_shared_expert_run_beside_fc1_only_in_decode_graphs, record
- 关键代码摘录:

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

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/nemotron_h.py` modified +34/-8; `python/tokenspeed/runtime/models/nemotron_h_nextn.py` modified +2/-0
  - tests: `test/runtime/test_nemotron_h.py` modified +34/-0
- 验证与风险: diff 自带测试面 `test/runtime/layers/test_gdn_qkv_split_fused.py`, `test/runtime/test_gdn_state_paging.py`, `test/runtime/test_nemotron_h.py`, `tokenspeed-kernel/test/nvidia/ops/gemm/test_fp8_skinny_m1.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
