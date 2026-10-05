# TokenSpeed Inkling 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `python/tokenspeed/runtime/configs/inkling_config.py` | [#689](https://github.com/lightseekorg/tokenspeed/pull/689), [#827](https://github.com/lightseekorg/tokenspeed/pull/827), [#1110](https://github.com/lightseekorg/tokenspeed/pull/1110) |
| `python/tokenspeed/runtime/layers/attention/backends/specific/inkling.py` | 无直接 PR 号提交 |
| `python/tokenspeed/runtime/layers/attention/kv_cache/hybrid_inkling.py` | 无直接 PR 号提交 |
| `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/inkling.py` | [#996](https://github.com/lightseekorg/tokenspeed/pull/996), [#1040](https://github.com/lightseekorg/tokenspeed/pull/1040), [#1110](https://github.com/lightseekorg/tokenspeed/pull/1110) |
| `python/tokenspeed/runtime/models/inkling.py` | [#689](https://github.com/lightseekorg/tokenspeed/pull/689), [#771](https://github.com/lightseekorg/tokenspeed/pull/771), [#996](https://github.com/lightseekorg/tokenspeed/pull/996), [#1040](https://github.com/lightseekorg/tokenspeed/pull/1040), [#1110](https://github.com/lightseekorg/tokenspeed/pull/1110) |
| `python/tokenspeed/runtime/models/inkling_nextn.py` | [#689](https://github.com/lightseekorg/tokenspeed/pull/689) |
| `test/agentic_benchmark/inkling/tokenspeed/agentic_bench.sh` | 无直接 PR 号提交 |
| `test/agentic_benchmark/inkling/tokenspeed/collect_outputs.py` | 无直接 PR 号提交 |
| `test/agentic_benchmark/inkling/tokenspeed/configs/attn_tp4_moe_ep4.sh` | 无直接 PR 号提交 |
| `test/agentic_benchmark/inkling/tokenspeed/configs/attn_tp4_moe_tp4.sh` | 无直接 PR 号提交 |
| `test/ci/eval/inkling-mxfp4-mtp-evalscope-aime25.yaml` | [#1150](https://github.com/lightseekorg/tokenspeed/pull/1150), [#1960](https://github.com/lightseekorg/tokenspeed/pull/1960) |
| `test/ci/eval/inkling-nvfp4-mtp-evalscope-aime25.yaml` | [#1150](https://github.com/lightseekorg/tokenspeed/pull/1150) |
| `test/runtime/models/inkling_fixtures.py` | [#689](https://github.com/lightseekorg/tokenspeed/pull/689), [#827](https://github.com/lightseekorg/tokenspeed/pull/827) |
| `test/runtime/models/test_inkling_models.py` | [#689](https://github.com/lightseekorg/tokenspeed/pull/689), [#827](https://github.com/lightseekorg/tokenspeed/pull/827), [#996](https://github.com/lightseekorg/tokenspeed/pull/996), [#1040](https://github.com/lightseekorg/tokenspeed/pull/1040) |
| `test/runtime/models/test_inkling_multimodal.py` | [#689](https://github.com/lightseekorg/tokenspeed/pull/689), [#771](https://github.com/lightseekorg/tokenspeed/pull/771), [#827](https://github.com/lightseekorg/tokenspeed/pull/827) |
| `test/runtime/test_inkling_activation_parity.py` | [#689](https://github.com/lightseekorg/tokenspeed/pull/689), [#827](https://github.com/lightseekorg/tokenspeed/pull/827), [#996](https://github.com/lightseekorg/tokenspeed/pull/996), [#1040](https://github.com/lightseekorg/tokenspeed/pull/1040), [#1110](https://github.com/lightseekorg/tokenspeed/pull/1110) |
| `test/runtime/test_inkling_load_weights.py` | [#689](https://github.com/lightseekorg/tokenspeed/pull/689), [#827](https://github.com/lightseekorg/tokenspeed/pull/827), [#1110](https://github.com/lightseekorg/tokenspeed/pull/1110) |
| `test/runtime/test_inkling_mtp_conv_state.py` | [#689](https://github.com/lightseekorg/tokenspeed/pull/689), [#996](https://github.com/lightseekorg/tokenspeed/pull/996), [#1040](https://github.com/lightseekorg/tokenspeed/pull/1040) |
| `test/runtime/test_inkling_mtp_load_weights.py` | [#689](https://github.com/lightseekorg/tokenspeed/pull/689), [#827](https://github.com/lightseekorg/tokenspeed/pull/827) |
| `test/runtime/test_inkling_mtp_text_config.py` | [#689](https://github.com/lightseekorg/tokenspeed/pull/689), [#827](https://github.com/lightseekorg/tokenspeed/pull/827), [#1110](https://github.com/lightseekorg/tokenspeed/pull/1110) |
| `test/runtime/test_inkling_real_checkpoint_load.py` | [#689](https://github.com/lightseekorg/tokenspeed/pull/689) |
| `test/runtime/test_inkling_reference_parity.py` | [#689](https://github.com/lightseekorg/tokenspeed/pull/689), [#827](https://github.com/lightseekorg/tokenspeed/pull/827) |
| `tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/triton/inkling_topk.py` | [#689](https://github.com/lightseekorg/tokenspeed/pull/689) |
| `tokenspeed-kernel/test/ops/test_inkling_topk.py` | [#689](https://github.com/lightseekorg/tokenspeed/pull/689) |

## PR 覆盖总览

- git 追溯 PR 数: 8
- 原文档显式引用补充 PR 数: 0
- 当前文档总 PR 数: 8
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-07-16 | [#689](https://github.com/lightseekorg/tokenspeed/pull/689) | merged | Add TML Inkling support | `python/tokenspeed/runtime/models/inkling.py`, `python/tokenspeed/runtime/layers/attention/backends/inkling.py`, `python/tokenspeed/runtime/configs/inkling_config.py` |
| 2026-07-22 | [#771](https://github.com/lightseekorg/tokenspeed/pull/771) | merged | fix(inkling): fuse dMel lookup+sum to bound audio encode memory | `test/runtime/models/test_inkling_multimodal.py`, `python/tokenspeed/runtime/models/inkling.py` |
| 2026-07-28 | [#827](https://github.com/lightseekorg/tokenspeed/pull/827) | merged | chore(inkling): fix hard-coded config, fix and register unit tests | `test/runtime/models/inkling_fixtures.py`, `test/runtime/models/test_inkling_models.py`, `test/runtime/models/test_inkling_multimodal.py` |
| 2026-08-08 | [#996](https://github.com/lightseekorg/tokenspeed/pull/996) | merged | refactor(inkling): ring sconv state with in-kernel persistence and speculative checkpoint publish | `python/tokenspeed/runtime/layers/attention/backends/inkling.py`, `python/tokenspeed/runtime/models/inkling.py`, `test/runtime/models/test_inkling_models.py` |
| 2026-08-11 | [#1040](https://github.com/lightseekorg/tokenspeed/pull/1040) | merged | feat(inkling): paged sconv checkpoints for the MTP draft + unified fused sconv kernels | `python/tokenspeed/runtime/layers/attention/backends/inkling.py`, `test/runtime/models/test_inkling_models.py`, `python/tokenspeed/runtime/models/inkling.py` |
| 2026-08-15 | [#1110](https://github.com/lightseekorg/tokenspeed/pull/1110) | merged | perf(inkling): eliminate per-step elementwise/copy kernels on the decode path | `python/tokenspeed/runtime/models/inkling.py`, `python/tokenspeed/runtime/configs/inkling_config.py`, `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/inkling.py` |
| 2026-08-19 | [#1150](https://github.com/lightseekorg/tokenspeed/pull/1150) | merged | ci(inkling): switch Inkling CI from GSM8K to AIME25 | `test/ci/eval/inkling-mxfp4-mtp-evalscope-aime25.yaml`, `test/ci/eval/inkling-nvfp4-mtp-evalscope-aime25.yaml` |
| 2026-10-03 | [#1960](https://github.com/lightseekorg/tokenspeed/pull/1960) | merged | ci(amd): Run all Inkling AIME25 problems concurrently | `test/ci/eval/inkling-mxfp4-mtp-evalscope-aime25.yaml` |

## 逐 PR diff 审计卡

### PR #689 - Add TML Inkling support

- 链接: https://github.com/lightseekorg/tokenspeed/pull/689
- 状态/时间: merged / 2026-07-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/configs/inkling_config.py`, `python/tokenspeed/runtime/models/inkling.py`, `python/tokenspeed/runtime/models/inkling_nextn.py`, `test/runtime/models/inkling_fixtures.py`, `test/runtime/models/test_inkling_models.py` 等 15 个文件；关联提交 `93ecb3ae8dea`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 131 个文件，+21501/-747，可读 patch 24597 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/inkling.py` added +2145/-0 (2145 lines); hunks: -0,0 +1,2145; symbols: _translate_inkling_quant_pattern, _translate_quant_exclusions, translate, _deinterleave_w13，涉及 `_translate_inkling_quant_pattern, _translate_quant_exclusions, translate`；`python/tokenspeed/runtime/layers/attention/backends/inkling.py` added +1173/-0 (1173 lines); hunks: -0,0 +1,1173; symbols: InklingConvMetadata, InklingConvStatePool, __init__, layer_state，涉及 `InklingConvMetadata, InklingConvStatePool, __init__`；`python/tokenspeed/runtime/configs/inkling_config.py` added +575/-0 (575 lines); hunks: -0,0 +1,575; symbols: InklingConvStream, inkling_conv_stream_layout, inkling_kv_heads_for_layer, inkling_mtp_text_config，涉及 `InklingConvStream, inkling_conv_stream_layout, inkling_kv_heads_for_layer`；`test/runtime/models/test_inkling_multimodal.py` added +498/-0 (498 lines); hunks: -0,0 +1,498; symbols: _has_blackwell, _naive_hmlp_forward, fold, TestInklingPlanOutScales，涉及 `_has_blackwell, _naive_hmlp_forward, fold`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/inkling.py` added +2145/-0 (2145 lines); hunks: -0,0 +1,2145; symbols: _translate_inkling_quant_pattern, _translate_quant_exclusions, translate, _deinterleave_w13
  - `python/tokenspeed/runtime/layers/attention/backends/inkling.py` added +1173/-0 (1173 lines); hunks: -0,0 +1,1173; symbols: InklingConvMetadata, InklingConvStatePool, __init__, layer_state
  - `python/tokenspeed/runtime/configs/inkling_config.py` added +575/-0 (575 lines); hunks: -0,0 +1,575; symbols: InklingConvStream, inkling_conv_stream_layout, inkling_kv_heads_for_layer, inkling_mtp_text_config
  - `test/runtime/models/test_inkling_multimodal.py` added +498/-0 (498 lines); hunks: -0,0 +1,498; symbols: _has_blackwell, _naive_hmlp_forward, fold, TestInklingPlanOutScales
  - `python/tokenspeed/runtime/models/inkling_nextn.py` added +438/-0 (438 lines); hunks: -0,0 +1,438; symbols: _draft_text_config, InklingMultiTokenPredictorLayer, __init__, forward
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/inkling.py
@@ -0,0 +1,2145 @@
+# Copyright (c) 2026 LightSeek Foundation
+#
+# Permission is hereby granted, free of charge, to any person obtaining a copy
+# of this software and associated documentation files (the "Software"), to deal
+# in the Software without restriction, including without limitation the rights
+# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
diff -- python/tokenspeed/runtime/layers/attention/backends/inkling.py
@@ -0,0 +1,1173 @@
+# Copyright (c) 2026 LightSeek Foundation
+#
+# Permission is hereby granted, free of charge, to any person obtaining a copy
+# of this software and associated documentation files (the "Software"), to deal
+# in the Software without restriction, including without limitation the rights
+# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
diff -- python/tokenspeed/runtime/configs/inkling_config.py
@@ -0,0 +1,575 @@
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/inkling.py` added +2145/-0; `python/tokenspeed/runtime/layers/attention/backends/inkling.py` added +1173/-0; `python/tokenspeed/runtime/configs/inkling_config.py` added +575/-0; `python/tokenspeed/runtime/models/inkling_nextn.py` added +438/-0
  - tests: `test/runtime/models/test_inkling_multimodal.py` added +498/-0; `test/runtime/models/test_inkling_models.py` added +315/-0; `test/runtime/models/inkling_fixtures.py` added +186/-0; `test/runtime/test_inkling_load_weights.py` added +745/-0
- 验证与风险: diff 自带测试面 `test/cli/test_argsplit.py`, `test/cli/test_serve_smg_unit.py`, `test/runtime/models/inkling_fixtures.py`, `test/runtime/models/test_inkling_models.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #771 - fix(inkling): fuse dMel lookup+sum to bound audio encode memory

- 链接: https://github.com/lightseekorg/tokenspeed/pull/771
- 状态/时间: merged / 2026-07-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/inkling.py`, `test/runtime/models/test_inkling_multimodal.py`；关联提交 `9a20ec9be67d`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+52/-3，可读 patch 69 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/runtime/models/test_inkling_multimodal.py` modified +46/-0 (46 lines); hunks: -234,6 +234,52 @@ def test_no_norm_variant(self):; symbols: test_no_norm_variant, test_peak_memory_does_not_scale_with_n_mel_bins, TestInklingHMLPPatchEncoder，涉及 `test_no_norm_variant, test_peak_memory_does_not_scale_with_n_mel_bins, TestInklingHMLPPatchEncoder`；`python/tokenspeed/runtime/models/inkling.py` modified +6/-3 (9 lines); hunks: -1442,9 +1442,12 @@ def forward(self, audio_features: torch.Tensor) -> torch....; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `test/runtime/models/test_inkling_multimodal.py` modified +46/-0 (46 lines); hunks: -234,6 +234,52 @@ def test_no_norm_variant(self):; symbols: test_no_norm_variant, test_peak_memory_does_not_scale_with_n_mel_bins, TestInklingHMLPPatchEncoder
  - `python/tokenspeed/runtime/models/inkling.py` modified +6/-3 (9 lines); hunks: -1442,9 +1442,12 @@ def forward(self, audio_features: torch.Tensor) -> torch....; symbols: forward
- 关键代码摘录:

```diff
diff -- test/runtime/models/test_inkling_multimodal.py
@@ -234,6 +234,52 @@ def test_no_norm_variant(self):
+    def test_peak_memory_does_not_scale_with_n_mel_bins(self):
+        """dMel embedding must fuse lookup+sum.
+        The unfused ``encoder(idx).reshape(...).sum(1)`` form materializes an
+        intermediate ``n_mel_bins`` times the size of the output (~0.95 MB per
+        audio token at the released ``decoder_dmodel=6144``), which lets a
+        single long clip OOM the engine. Assert the peak stays within a small
diff -- python/tokenspeed/runtime/models/inkling.py
@@ -1442,9 +1442,12 @@ def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
-        hidden_states = self.encoder((bin_offsets + dmel).reshape(-1))
-        hidden_states = hidden_states.reshape(dmel.shape[0], self.n_mel_bins, -1).sum(
-            dim=1
+        # Fused lookup+sum: the unfused form materializes an intermediate of
+        # ``(num_tokens * n_mel_bins, decoder_dmodel)``, i.e. ``n_mel_bins``
+        # (80) times the output, which dominates peak memory on long clips --
```

- 提取文件（未人工审阅）:
  - tests: `test/runtime/models/test_inkling_multimodal.py` modified +46/-0
  - runtime: `python/tokenspeed/runtime/models/inkling.py` modified +6/-3
- 验证与风险: diff 自带测试面 `test/runtime/models/test_inkling_multimodal.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #827 - chore(inkling): fix hard-coded config, fix and register unit tests

- 链接: https://github.com/lightseekorg/tokenspeed/pull/827
- 状态/时间: merged / 2026-07-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/configs/inkling_config.py`, `test/runtime/models/inkling_fixtures.py`, `test/runtime/models/test_inkling_models.py`, `test/runtime/models/test_inkling_multimodal.py`, `test/runtime/test_inkling_activation_parity.py` 等 9 个文件；关联提交 `38f08e341aff`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 12 个文件，+475/-780，可读 patch 1907 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/runtime/models/inkling_fixtures.py` modified +90/-174 (264 lines); hunks: -1,186 +1,102; symbols: _write_synthetic_tokenizer, has_blackwell, truncate_text_config, make_inkling_dummy_checkpoint，涉及 `_write_synthetic_tokenizer, has_blackwell, truncate_text_config`；`test/runtime/models/test_inkling_models.py` modified +94/-48 (142 lines); hunks: -1,13 +1,13; -26,31 +26,57; symbols: TestInklingConfigRegistry, test_config_registry, test_get_config_loads_tiny_fixture, test_get_config_loads_hub_snapshot，涉及 `TestInklingConfigRegistry, test_config_registry, test_get_config_loads_tiny_fixture`；`test/runtime/models/test_inkling_multimodal.py` modified +76/-59 (135 lines); hunks: -7,14 +7,14; -24,7 +24,6; symbols: test_arch_flags, test_placeholder_token_ids, test_mrope_is_noop_for_tml, TestInklingAudioTower，涉及 `test_arch_flags, test_placeholder_token_ids, test_mrope_is_noop_for_tml`；`python/tokenspeed/runtime/configs/inkling_config.py` modified +4/-2 (6 lines); hunks: -109,8 +109,10 @@ def inkling_kv_heads_for_layer(; symbols: inkling_kv_heads_for_layer，涉及 `inkling_kv_heads_for_layer`。
- 代码 diff 细节:
  - `test/runtime/models/inkling_fixtures.py` modified +90/-174 (264 lines); hunks: -1,186 +1,102; symbols: _write_synthetic_tokenizer, has_blackwell, truncate_text_config, make_inkling_dummy_checkpoint
  - `test/runtime/models/test_inkling_models.py` modified +94/-48 (142 lines); hunks: -1,13 +1,13; -26,31 +26,57; symbols: TestInklingConfigRegistry, test_config_registry, test_get_config_loads_tiny_fixture, test_get_config_loads_hub_snapshot
  - `test/runtime/models/test_inkling_multimodal.py` modified +76/-59 (135 lines); hunks: -7,14 +7,14; -24,7 +24,6; symbols: test_arch_flags, test_placeholder_token_ids, test_mrope_is_noop_for_tml, TestInklingAudioTower
  - `python/tokenspeed/runtime/configs/inkling_config.py` modified +4/-2 (6 lines); hunks: -109,8 +109,10 @@ def inkling_kv_heads_for_layer(; symbols: inkling_kv_heads_for_layer
  - `test/runtime/test_inkling_load_weights.py` modified +89/-66 (155 lines); hunks: -1,48 +1,54; -59,6 +65,22 @@ def _deinterleave_rows(w: torch.Tensor) -> torch.Tensor:; symbols: _build_model, _deinterleave_rows, _make_tower_tensors, _make_checkpoint_tensors
- 关键代码摘录:

```diff
diff -- test/runtime/models/inkling_fixtures.py
@@ -1,186 +1,102 @@
-"""Test fixtures for the Inkling model: dummy checkpoint directories.
+"""Shared Inkling test config: the released hub checkpoints, truncated in-memory.
-The tiny variant is fully synthetic (safe to commit). The full-size variant
-reads the confidential reference ``config.json``/tokenizer from the directory
-named by the ``INKLING_REF_DIR`` env var at runtime and is never embedded here.
+Tests load the real config/tokenizer from the released snapshots (weightless
diff -- test/runtime/models/test_inkling_models.py
@@ -1,13 +1,13 @@
-"""Inkling model unit tests: config registration, scheduler blindness, fixtures.
+"""Inkling model unit tests: config registration, scheduler blindness, modules.
-Model-module tests (gate math, sconv parity, shapes) are added alongside the
-model implementation. NOTE: intentionally NOT registered in CI suites while
-the Inkling port is confidential/local-only.
+All configs come from the released hub snapshots (``inkling_fixtures``),
diff -- test/runtime/models/test_inkling_multimodal.py
@@ -7,14 +7,14 @@
```

- 提取文件（未人工审阅）:
  - tests: `test/runtime/models/inkling_fixtures.py` modified +90/-174; `test/runtime/models/test_inkling_models.py` modified +94/-48; `test/runtime/models/test_inkling_multimodal.py` modified +76/-59; `test/runtime/test_inkling_load_weights.py` modified +89/-66; `test/runtime/test_inkling_mtp_load_weights.py` modified +25/-24; `test/runtime/test_inkling_reference_parity.py` modified +22/-16
  - runtime: `python/tokenspeed/runtime/configs/inkling_config.py` modified +4/-2
- 验证与风险: diff 自带测试面 `test/runtime/models/inkling_fixtures.py`, `test/runtime/models/test_inkling_models.py`, `test/runtime/models/test_inkling_multimodal.py`, `test/runtime/test_inkling_activation_parity.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #996 - refactor(inkling): ring sconv state with in-kernel persistence and speculative checkpoint publish

- 链接: https://github.com/lightseekorg/tokenspeed/pull/996
- 状态/时间: merged / 2026-08-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/inkling.py`, `python/tokenspeed/runtime/models/inkling.py`, `test/runtime/models/test_inkling_models.py`, `test/runtime/test_inkling_activation_parity.py`, `test/runtime/test_inkling_mtp_conv_state.py`；关联提交 `27008e78184c`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 10 个文件，+1338/-2426，可读 patch 4704 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/layers/attention/backends/inkling.py` modified +122/-555 (677 lines); hunks: -21,8 +21,11; -51,7 +54,7; symbols: InklingConvMetadata, InklingConvStatePool, __init__, layer_state，涉及 `InklingConvMetadata, InklingConvStatePool, __init__`；`python/tokenspeed/runtime/models/inkling.py` modified +35/-149 (184 lines); hunks: -81,12 +81,7; -144,10 +139,6; symbols: _sconv_apply，涉及 `_sconv_apply`；`test/runtime/models/test_inkling_models.py` modified +21/-13 (34 lines); hunks: -271,20 +271,19 @@ def _make_ctx(self, num_layers, num_slots, conv_dim, kerne...; -325,6 +324,8 @@ def test_prefill_then_decode_parity(self):; symbols: _make_ctx, test_prefill_then_decode_parity，涉及 `_make_ctx, test_prefill_then_decode_parity`；`python/tokenspeed/runtime/layers/attention/kv_cache/recipes/inkling.py` modified +10/-17 (27 lines); hunks: -205,26 +205,22 @@ def _workspace_bytes(; -254,7 +250,7 @@ def prepare_inkling_cache(; symbols: _workspace_bytes, prepare_inkling_cache，涉及 `_workspace_bytes, prepare_inkling_cache`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/layers/attention/backends/inkling.py` modified +122/-555 (677 lines); hunks: -21,8 +21,11; -51,7 +54,7; symbols: InklingConvMetadata, InklingConvStatePool, __init__, layer_state
  - `python/tokenspeed/runtime/models/inkling.py` modified +35/-149 (184 lines); hunks: -81,12 +81,7; -144,10 +139,6; symbols: _sconv_apply
  - `test/runtime/models/test_inkling_models.py` modified +21/-13 (34 lines); hunks: -271,20 +271,19 @@ def _make_ctx(self, num_layers, num_slots, conv_dim, kerne...; -325,6 +324,8 @@ def test_prefill_then_decode_parity(self):; symbols: _make_ctx, test_prefill_then_decode_parity
  - `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/inkling.py` modified +10/-17 (27 lines); hunks: -205,26 +205,22 @@ def _workspace_bytes(; -254,7 +250,7 @@ def prepare_inkling_cache(; symbols: _workspace_bytes, prepare_inkling_cache
  - `test/runtime/test_inkling_mtp_conv_state.py` modified +561/-245 (806 lines); hunks: -1,12 +1,14; -35,46 +37,11 @@ class HistoryBackend:; symbols: HistoryBackend, test_checkpoint_publication_masks_padded_rows_before_indexing, TestInklingConvSpecState, TestInklingConvRingState
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/layers/attention/backends/inkling.py
@@ -21,8 +21,11 @@
-sconv rolling state — four short-causal-conv streams per decoder block,
-window ``W-1`` states per request — is managed entirely engine-side:
+sconv working state — four short-causal-conv streams per decoder block, a
+ring of the last ``R`` input rows per request — is managed entirely
+engine-side. The ring row of absolute position ``p`` is ``p % R``; positions
+derive from the through-chunk ``seq_lens``, so there is no stored cursor and
diff -- python/tokenspeed/runtime/models/inkling.py
@@ -81,12 +81,7 @@
-from tokenspeed_kernel.ops.conv import (
-    sconv_decode,
-    sconv_decode_paged,
-    sconv_prefill,
-    sconv_prefill_paged,
-)
diff -- test/runtime/models/test_inkling_models.py
@@ -271,20 +271,19 @@ def _make_ctx(self, num_layers, num_slots, conv_dim, kernel_size, device):
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/layers/attention/backends/inkling.py` modified +122/-555; `python/tokenspeed/runtime/models/inkling.py` modified +35/-149; `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/inkling.py` modified +10/-17
  - tests: `test/runtime/models/test_inkling_models.py` modified +21/-13; `test/runtime/test_inkling_mtp_conv_state.py` modified +561/-245; `test/runtime/test_inkling_activation_parity.py` modified +2/-0
- 验证与风险: diff 自带测试面 `test/runtime/models/test_inkling_models.py`, `test/runtime/test_inkling_activation_parity.py`, `test/runtime/test_inkling_mtp_conv_state.py`, `tokenspeed-kernel/test/ops/test_conv.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1040 - feat(inkling): paged sconv checkpoints for the MTP draft + unified fused sconv kernels

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1040
- 状态/时间: merged / 2026-08-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/inkling.py`, `python/tokenspeed/runtime/models/inkling.py`, `test/runtime/models/test_inkling_models.py`, `test/runtime/test_inkling_activation_parity.py`, `test/runtime/test_inkling_mtp_conv_state.py`；关联提交 `6739113f945e`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 14 个文件，+1159/-1201，可读 patch 3165 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/layers/attention/backends/inkling.py` modified +91/-446 (537 lines); hunks: -71,18 +71,6; -108,16 +96,14 @@ class InklingConvMetadata:; symbols: ShortConvCheckpointMetadata, InklingConvMetadata, InklingConvStatePool, __init__，涉及 `ShortConvCheckpointMetadata, InklingConvMetadata, InklingConvStatePool`；`test/runtime/models/test_inkling_models.py` modified +0/-112 (112 lines); hunks: -253,117 +253,5 @@ def test_deferred_matches_finalized(self):; symbols: test_deferred_matches_finalized, _ref_sconv, TestInklingShortConvolution, _make_ctx，涉及 `test_deferred_matches_finalized, _ref_sconv, TestInklingShortConvolution`；`python/tokenspeed/runtime/models/inkling.py` modified +29/-61 (90 lines); hunks: -81,7 +81,7; -367,66 +367,44 @@ def _sconv_apply(; symbols: _sconv_apply, InklingRelLogitsProj，涉及 `_sconv_apply, InklingRelLogitsProj`；`python/tokenspeed/runtime/layers/attention/kv_cache/recipes/inkling.py` modified +5/-15 (20 lines); hunks: -4,7 +4,6; -268,20 +267,11 @@ def prepare_inkling_cache(; symbols: prepare_inkling_cache，涉及 `prepare_inkling_cache`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/layers/attention/backends/inkling.py` modified +91/-446 (537 lines); hunks: -71,18 +71,6; -108,16 +96,14 @@ class InklingConvMetadata:; symbols: ShortConvCheckpointMetadata, InklingConvMetadata, InklingConvStatePool, __init__
  - `test/runtime/models/test_inkling_models.py` modified +0/-112 (112 lines); hunks: -253,117 +253,5 @@ def test_deferred_matches_finalized(self):; symbols: test_deferred_matches_finalized, _ref_sconv, TestInklingShortConvolution, _make_ctx
  - `python/tokenspeed/runtime/models/inkling.py` modified +29/-61 (90 lines); hunks: -81,7 +81,7; -367,66 +367,44 @@ def _sconv_apply(; symbols: _sconv_apply, InklingRelLogitsProj
  - `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/inkling.py` modified +5/-15 (20 lines); hunks: -4,7 +4,6; -268,20 +267,11 @@ def prepare_inkling_cache(; symbols: prepare_inkling_cache
  - `test/runtime/test_inkling_mtp_conv_state.py` modified +114/-124 (238 lines); hunks: -20,6 +20,13; -130,6 +137,10 @@ def test_ring_holds_window_for_every_accept(self):; symbols: _inert_publish, TestInklingCacheContract, test_wrapper_consumes_history_and_checkpoint_state, test_ring_holds_window_for_every_accept
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/layers/attention/backends/inkling.py
@@ -71,18 +71,6 @@
-@dataclass
-class ShortConvCheckpointMetadata:
-    """Device indices for restoring and publishing boundary checkpoints."""
-    restore_pages: dict[str, torch.Tensor]
-    write_pages: dict[str, torch.Tensor]
-    write_requests: torch.Tensor
diff -- test/runtime/models/test_inkling_models.py
@@ -253,117 +253,5 @@ def test_deferred_matches_finalized(self):
-def _ref_sconv(x, weight, prefix, use_residual=True):
-    """Torch reference: residual causal FIR, current token = last tap."""
-    import torch
-    W = weight.shape[1]
-    xp = torch.cat([prefix.float(), x.float()])
-    y = sum(xp[w : w + len(x)] * weight[:, w].float() for w in range(W))
diff -- python/tokenspeed/runtime/models/inkling.py
@@ -81,7 +81,7 @@
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/layers/attention/backends/inkling.py` modified +91/-446; `python/tokenspeed/runtime/models/inkling.py` modified +29/-61; `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/inkling.py` modified +5/-15
  - tests: `test/runtime/models/test_inkling_models.py` modified +0/-112; `test/runtime/test_inkling_mtp_conv_state.py` modified +114/-124; `test/runtime/test_inkling_activation_parity.py` modified +72/-2
- 验证与风险: diff 自带测试面 `test/runtime/models/test_inkling_models.py`, `test/runtime/test_cache_memory_plan.py`, `test/runtime/test_cache_setup.py`, `test/runtime/test_inkling_activation_parity.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1110 - perf(inkling): eliminate per-step elementwise/copy kernels on the decode path

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1110
- 状态/时间: merged / 2026-08-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/configs/inkling_config.py`, `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/inkling.py`, `python/tokenspeed/runtime/models/inkling.py`, `test/runtime/test_inkling_activation_parity.py`, `test/runtime/test_inkling_load_weights.py` 等 6 个文件；关联提交 `21cfc7c4199e`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 20 个文件，+359/-201，可读 patch 1418 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/inkling.py` modified +18/-39 (57 lines); hunks: -44,8 +44,8; -81,6 +81,9; symbols: _load_block_param, compute_log_scaling_tau, _apply_log_scaling_tau, InklingShortConvolution，涉及 `_load_block_param, compute_log_scaling_tau, _apply_log_scaling_tau`；`python/tokenspeed/runtime/configs/inkling_config.py` modified +2/-29 (31 lines); hunks: -101,35 +101,6 @@ def inkling_conv_stream_layout(; -184,6 +155,7 @@ class InklingModelConfig(PretrainedConfig):; symbols: inkling_conv_stream_layout, inkling_kv_heads_for_layer, inkling_mtp_text_config, InklingModelConfig，涉及 `inkling_conv_stream_layout, inkling_kv_heads_for_layer, inkling_mtp_text_config`；`python/tokenspeed/runtime/layers/attention/kv_cache/recipes/inkling.py` modified +8/-3 (11 lines); hunks: -148,11 +148,16 @@ def inkling_cache_fields(; symbols: inkling_cache_fields, inkling_layer_kv_head_counts，涉及 `inkling_cache_fields, inkling_layer_kv_head_counts`；`python/tokenspeed/runtime/layers/attention/backends/inkling.py` modified +5/-0 (5 lines); hunks: -642,6 +642,7 @@ def forward_decode(; -695,6 +696,7 @@ def forward_decode(; symbols: forward_decode, forward_extend，涉及 `forward_decode, forward_extend`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/inkling.py` modified +18/-39 (57 lines); hunks: -44,8 +44,8; -81,6 +81,9; symbols: _load_block_param, compute_log_scaling_tau, _apply_log_scaling_tau, InklingShortConvolution
  - `python/tokenspeed/runtime/configs/inkling_config.py` modified +2/-29 (31 lines); hunks: -101,35 +101,6 @@ def inkling_conv_stream_layout(; -184,6 +155,7 @@ class InklingModelConfig(PretrainedConfig):; symbols: inkling_conv_stream_layout, inkling_kv_heads_for_layer, inkling_mtp_text_config, InklingModelConfig
  - `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/inkling.py` modified +8/-3 (11 lines); hunks: -148,11 +148,16 @@ def inkling_cache_fields(; symbols: inkling_cache_fields, inkling_layer_kv_head_counts
  - `python/tokenspeed/runtime/layers/attention/backends/inkling.py` modified +5/-0 (5 lines); hunks: -642,6 +642,7 @@ def forward_decode(; -695,6 +696,7 @@ def forward_decode(; symbols: forward_decode, forward_extend
  - `test/runtime/test_inkling_load_weights.py` modified +12/-44 (56 lines); hunks: -211,50 +211,24 @@ def test_full_parameter_coverage(self):; -458,24 +432,18 @@ def _param(self, model, name):; symbols: test_full_parameter_coverage, test_qkvr_fusion_and_kv_replication, test_qkvr_fusion, test_dense_mlp_deinterleave_and_scale
- 关键代码摘录:

```diff
diff -- python/tokenspeed/runtime/models/inkling.py
@@ -44,8 +44,8 @@
-head count (unconditional; the INKLING_HETERO_KV gate is retired) — uniform serving with
-full-layer KV replicated at load (see the config docstring).
+head count — uniform serving with full-layer KV replicated at load (see the
+config docstring).
@@ -81,6 +81,9 @@
+from tokenspeed_kernel.ops.attention.triton.log_scaling import (
diff -- python/tokenspeed/runtime/configs/inkling_config.py
@@ -101,35 +101,6 @@ def inkling_conv_stream_layout(
-def inkling_kv_heads_for_layer(
-    config: "InklingModelConfig", layer_id: int, hetero: bool
-) -> int:
-    """Served KV head count for one layer.
-    Uniform mode replicates every layer to ``num_key_value_heads`` (the max
-    over layer kinds). Heterogeneous mode (byte-uniform slots, #647) serves
diff -- python/tokenspeed/runtime/layers/attention/kv_cache/recipes/inkling.py
@@ -148,11 +148,16 @@ def inkling_cache_fields(
```

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/inkling.py` modified +18/-39; `python/tokenspeed/runtime/configs/inkling_config.py` modified +2/-29; `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/inkling.py` modified +8/-3; `python/tokenspeed/runtime/layers/attention/backends/inkling.py` modified +5/-0
  - tests: `test/runtime/test_inkling_load_weights.py` modified +12/-44; `test/runtime/test_inkling_mtp_text_config.py` modified +1/-15; `test/runtime/test_inkling_activation_parity.py` modified +6/-5
- 验证与风险: diff 自带测试面 `test/runtime/test_inkling_activation_parity.py`, `test/runtime/test_inkling_load_weights.py`, `test/runtime/test_inkling_mtp_text_config.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1150 - ci(inkling): switch Inkling CI from GSM8K to AIME25

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1150
- 状态/时间: merged / 2026-08-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/ci/eval/inkling-mxfp4-mtp-evalscope-aime25.yaml`, `test/ci/eval/inkling-nvfp4-mtp-evalscope-aime25.yaml`；关联提交 `2edd9436d22d`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 4 个文件，+17/-15，可读 patch 86 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/ci/eval/inkling-mxfp4-mtp-evalscope-aime25.yaml` renamed +7/-6 (13 lines); hunks: -1,5 +1,5; -18,7 +18,7 @@ server:；`test/ci/eval/inkling-nvfp4-mtp-evalscope-aime25.yaml` renamed +7/-6 (13 lines); hunks: -1,5 +1,5; -18,7 +18,7 @@ server:。
- 代码 diff 细节:
  - `test/ci/eval/inkling-mxfp4-mtp-evalscope-aime25.yaml` renamed +7/-6 (13 lines); hunks: -1,5 +1,5; -18,7 +18,7 @@ server:
  - `test/ci/eval/inkling-nvfp4-mtp-evalscope-aime25.yaml` renamed +7/-6 (13 lines); hunks: -1,5 +1,5; -18,7 +18,7 @@ server:
- 关键代码摘录:

```diff
diff -- test/ci/eval/inkling-mxfp4-mtp-evalscope-aime25.yaml
@@ -1,5 +1,5 @@
-name: eval-inkling-mxfp4-mtp-gsm8k
+name: eval-inkling-mxfp4-mtp-aime25
@@ -18,7 +18,7 @@ server:
-    --max-model-len 81920
+    --max-model-len 102400
@@ -48,11 +48,12 @@ eval:
diff -- test/ci/eval/inkling-nvfp4-mtp-evalscope-aime25.yaml
@@ -1,5 +1,5 @@
-name: eval-inkling-nvfp4-mtp-gsm8k
+name: eval-inkling-nvfp4-mtp-aime25
@@ -18,7 +18,7 @@ server:
-    --max-model-len 81920
+    --max-model-len 102400
@@ -50,11 +50,12 @@ eval:
```

- 提取文件（未人工审阅）:
  - tests: `test/ci/eval/inkling-mxfp4-mtp-evalscope-aime25.yaml` renamed +7/-6; `test/ci/eval/inkling-nvfp4-mtp-evalscope-aime25.yaml` renamed +7/-6
- 验证与风险: diff 自带测试面 `test/ci/eval/inkling-mxfp4-mtp-evalscope-aime25.yaml`, `test/ci/eval/inkling-nvfp4-mtp-evalscope-aime25.yaml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1960 - ci(amd): Run all Inkling AIME25 problems concurrently

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1960
- 状态/时间: merged / 2026-10-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/ci/eval/inkling-mxfp4-mtp-evalscope-aime25.yaml`；关联提交 `0697a710cc68`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+4/-2，可读 patch 26 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/ci/eval/inkling-mxfp4-mtp-evalscope-aime25.yaml` modified +4/-2 (6 lines); hunks: -13,13 +13,15 @@ env:; -49,7 +51,7 @@ eval:。
- 代码 diff 细节:
  - `test/ci/eval/inkling-mxfp4-mtp-evalscope-aime25.yaml` modified +4/-2 (6 lines); hunks: -13,13 +13,15 @@ env:; -49,7 +51,7 @@ eval:
- 关键代码摘录:

```diff
diff -- test/ci/eval/inkling-mxfp4-mtp-evalscope-aime25.yaml
@@ -13,13 +13,15 @@ env:
+  # AIME25 has 30 problems. Admit them all at once (32 seqs) so the run is
+  # bounded by the longest single answer rather than by a late second wave.
-    --max-num-seqs 16
+    --max-num-seqs 32
@@ -49,7 +51,7 @@ eval:
-    --eval-batch-size 16
```

- 提取文件（未人工审阅）:
  - tests: `test/ci/eval/inkling-mxfp4-mtp-evalscope-aime25.yaml` modified +4/-2
- 验证与风险: diff 自带测试面 `test/ci/eval/inkling-mxfp4-mtp-evalscope-aime25.yaml`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
