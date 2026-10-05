# TokenSpeed Inkling Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `python/tokenspeed/runtime/configs/inkling_config.py` | [#689](https://github.com/lightseekorg/tokenspeed/pull/689), [#827](https://github.com/lightseekorg/tokenspeed/pull/827), [#1110](https://github.com/lightseekorg/tokenspeed/pull/1110) |
| `python/tokenspeed/runtime/layers/attention/backends/specific/inkling.py` | no direct PR-number commit |
| `python/tokenspeed/runtime/layers/attention/kv_cache/hybrid_inkling.py` | no direct PR-number commit |
| `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/inkling.py` | [#996](https://github.com/lightseekorg/tokenspeed/pull/996), [#1040](https://github.com/lightseekorg/tokenspeed/pull/1040), [#1110](https://github.com/lightseekorg/tokenspeed/pull/1110) |
| `python/tokenspeed/runtime/models/inkling.py` | [#689](https://github.com/lightseekorg/tokenspeed/pull/689), [#771](https://github.com/lightseekorg/tokenspeed/pull/771), [#996](https://github.com/lightseekorg/tokenspeed/pull/996), [#1040](https://github.com/lightseekorg/tokenspeed/pull/1040), [#1110](https://github.com/lightseekorg/tokenspeed/pull/1110) |
| `python/tokenspeed/runtime/models/inkling_nextn.py` | [#689](https://github.com/lightseekorg/tokenspeed/pull/689) |
| `test/agentic_benchmark/inkling/tokenspeed/agentic_bench.sh` | no direct PR-number commit |
| `test/agentic_benchmark/inkling/tokenspeed/collect_outputs.py` | no direct PR-number commit |
| `test/agentic_benchmark/inkling/tokenspeed/configs/attn_tp4_moe_ep4.sh` | no direct PR-number commit |
| `test/agentic_benchmark/inkling/tokenspeed/configs/attn_tp4_moe_tp4.sh` | no direct PR-number commit |
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

## PR Coverage Summary

- Git-traced PRs: 8
- Extra PRs preserved from existing docs: 0
- Total PRs in this document: 8
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-07-16 | [#689](https://github.com/lightseekorg/tokenspeed/pull/689) | merged | Add TML Inkling support | `python/tokenspeed/runtime/models/inkling.py`, `python/tokenspeed/runtime/layers/attention/backends/inkling.py`, `python/tokenspeed/runtime/configs/inkling_config.py` |
| 2026-07-22 | [#771](https://github.com/lightseekorg/tokenspeed/pull/771) | merged | fix(inkling): fuse dMel lookup+sum to bound audio encode memory | `test/runtime/models/test_inkling_multimodal.py`, `python/tokenspeed/runtime/models/inkling.py` |
| 2026-07-28 | [#827](https://github.com/lightseekorg/tokenspeed/pull/827) | merged | chore(inkling): fix hard-coded config, fix and register unit tests | `test/runtime/models/inkling_fixtures.py`, `test/runtime/models/test_inkling_models.py`, `test/runtime/models/test_inkling_multimodal.py` |
| 2026-08-08 | [#996](https://github.com/lightseekorg/tokenspeed/pull/996) | merged | refactor(inkling): ring sconv state with in-kernel persistence and speculative checkpoint publish | `python/tokenspeed/runtime/layers/attention/backends/inkling.py`, `python/tokenspeed/runtime/models/inkling.py`, `test/runtime/models/test_inkling_models.py` |
| 2026-08-11 | [#1040](https://github.com/lightseekorg/tokenspeed/pull/1040) | merged | feat(inkling): paged sconv checkpoints for the MTP draft + unified fused sconv kernels | `python/tokenspeed/runtime/layers/attention/backends/inkling.py`, `test/runtime/models/test_inkling_models.py`, `python/tokenspeed/runtime/models/inkling.py` |
| 2026-08-15 | [#1110](https://github.com/lightseekorg/tokenspeed/pull/1110) | merged | perf(inkling): eliminate per-step elementwise/copy kernels on the decode path | `python/tokenspeed/runtime/models/inkling.py`, `python/tokenspeed/runtime/configs/inkling_config.py`, `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/inkling.py` |
| 2026-08-19 | [#1150](https://github.com/lightseekorg/tokenspeed/pull/1150) | merged | ci(inkling): switch Inkling CI from GSM8K to AIME25 | `test/ci/eval/inkling-mxfp4-mtp-evalscope-aime25.yaml`, `test/ci/eval/inkling-nvfp4-mtp-evalscope-aime25.yaml` |
| 2026-10-03 | [#1960](https://github.com/lightseekorg/tokenspeed/pull/1960) | merged | ci(amd): Run all Inkling AIME25 problems concurrently | `test/ci/eval/inkling-mxfp4-mtp-evalscope-aime25.yaml` |

## Per-PR Diff Audit Cards

### PR #689 - Add TML Inkling support

- Link: https://github.com/lightseekorg/tokenspeed/pull/689
- Status/date: merged / 2026-07-16
- Trace source: `git log --name-only -- <model-files>` found it through `python/tokenspeed/runtime/configs/inkling_config.py`, `python/tokenspeed/runtime/models/inkling.py`, `python/tokenspeed/runtime/models/inkling_nextn.py`, `test/runtime/models/inkling_fixtures.py`, `test/runtime/models/test_inkling_models.py` and 15 files; associated commits `93ecb3ae8dea`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 131 files, +21501/-747, 24597 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/tokenspeed/runtime/models/inkling.py` added +2145/-0 (2145 lines); hunks: -0,0 +1,2145; symbols: _translate_inkling_quant_pattern, _translate_quant_exclusions, translate, _deinterleave_w13, touching `_translate_inkling_quant_pattern, _translate_quant_exclusions, translate`; `python/tokenspeed/runtime/layers/attention/backends/inkling.py` added +1173/-0 (1173 lines); hunks: -0,0 +1,1173; symbols: InklingConvMetadata, InklingConvStatePool, __init__, layer_state, touching `InklingConvMetadata, InklingConvStatePool, __init__`; `python/tokenspeed/runtime/configs/inkling_config.py` added +575/-0 (575 lines); hunks: -0,0 +1,575; symbols: InklingConvStream, inkling_conv_stream_layout, inkling_kv_heads_for_layer, inkling_mtp_text_config, touching `InklingConvStream, inkling_conv_stream_layout, inkling_kv_heads_for_layer`; `test/runtime/models/test_inkling_multimodal.py` added +498/-0 (498 lines); hunks: -0,0 +1,498; symbols: _has_blackwell, _naive_hmlp_forward, fold, TestInklingPlanOutScales, touching `_has_blackwell, _naive_hmlp_forward, fold`.
- Code diff details:
  - `python/tokenspeed/runtime/models/inkling.py` added +2145/-0 (2145 lines); hunks: -0,0 +1,2145; symbols: _translate_inkling_quant_pattern, _translate_quant_exclusions, translate, _deinterleave_w13
  - `python/tokenspeed/runtime/layers/attention/backends/inkling.py` added +1173/-0 (1173 lines); hunks: -0,0 +1,1173; symbols: InklingConvMetadata, InklingConvStatePool, __init__, layer_state
  - `python/tokenspeed/runtime/configs/inkling_config.py` added +575/-0 (575 lines); hunks: -0,0 +1,575; symbols: InklingConvStream, inkling_conv_stream_layout, inkling_kv_heads_for_layer, inkling_mtp_text_config
  - `test/runtime/models/test_inkling_multimodal.py` added +498/-0 (498 lines); hunks: -0,0 +1,498; symbols: _has_blackwell, _naive_hmlp_forward, fold, TestInklingPlanOutScales
  - `python/tokenspeed/runtime/models/inkling_nextn.py` added +438/-0 (438 lines); hunks: -0,0 +1,438; symbols: _draft_text_config, InklingMultiTokenPredictorLayer, __init__, forward
- Key code excerpts:

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

- Extracted files (not manually reviewed):
  - runtime: `python/tokenspeed/runtime/models/inkling.py` added +2145/-0; `python/tokenspeed/runtime/layers/attention/backends/inkling.py` added +1173/-0; `python/tokenspeed/runtime/configs/inkling_config.py` added +575/-0; `python/tokenspeed/runtime/models/inkling_nextn.py` added +438/-0
  - tests: `test/runtime/models/test_inkling_multimodal.py` added +498/-0; `test/runtime/models/test_inkling_models.py` added +315/-0; `test/runtime/models/inkling_fixtures.py` added +186/-0; `test/runtime/test_inkling_load_weights.py` added +745/-0
- Risk and verification: The diff ships test coverage in `test/cli/test_argsplit.py`, `test/cli/test_serve_smg_unit.py`, `test/runtime/models/inkling_fixtures.py`, `test/runtime/models/test_inkling_models.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #771 - fix(inkling): fuse dMel lookup+sum to bound audio encode memory

- Link: https://github.com/lightseekorg/tokenspeed/pull/771
- Status/date: merged / 2026-07-22
- Trace source: `git log --name-only -- <model-files>` found it through `python/tokenspeed/runtime/models/inkling.py`, `test/runtime/models/test_inkling_multimodal.py`; associated commits `9a20ec9be67d`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +52/-3, 69 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `test/runtime/models/test_inkling_multimodal.py` modified +46/-0 (46 lines); hunks: -234,6 +234,52 @@ def test_no_norm_variant(self):; symbols: test_no_norm_variant, test_peak_memory_does_not_scale_with_n_mel_bins, TestInklingHMLPPatchEncoder, touching `test_no_norm_variant, test_peak_memory_does_not_scale_with_n_mel_bins, TestInklingHMLPPatchEncoder`; `python/tokenspeed/runtime/models/inkling.py` modified +6/-3 (9 lines); hunks: -1442,9 +1442,12 @@ def forward(self, audio_features: torch.Tensor) -> torch....; symbols: forward, touching `forward`.
- Code diff details:
  - `test/runtime/models/test_inkling_multimodal.py` modified +46/-0 (46 lines); hunks: -234,6 +234,52 @@ def test_no_norm_variant(self):; symbols: test_no_norm_variant, test_peak_memory_does_not_scale_with_n_mel_bins, TestInklingHMLPPatchEncoder
  - `python/tokenspeed/runtime/models/inkling.py` modified +6/-3 (9 lines); hunks: -1442,9 +1442,12 @@ def forward(self, audio_features: torch.Tensor) -> torch....; symbols: forward
- Key code excerpts:

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

- Extracted files (not manually reviewed):
  - tests: `test/runtime/models/test_inkling_multimodal.py` modified +46/-0
  - runtime: `python/tokenspeed/runtime/models/inkling.py` modified +6/-3
- Risk and verification: The diff ships test coverage in `test/runtime/models/test_inkling_multimodal.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #827 - chore(inkling): fix hard-coded config, fix and register unit tests

- Link: https://github.com/lightseekorg/tokenspeed/pull/827
- Status/date: merged / 2026-07-28
- Trace source: `git log --name-only -- <model-files>` found it through `python/tokenspeed/runtime/configs/inkling_config.py`, `test/runtime/models/inkling_fixtures.py`, `test/runtime/models/test_inkling_models.py`, `test/runtime/models/test_inkling_multimodal.py`, `test/runtime/test_inkling_activation_parity.py` and 9 files; associated commits `38f08e341aff`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 12 files, +475/-780, 1907 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `test/runtime/models/inkling_fixtures.py` modified +90/-174 (264 lines); hunks: -1,186 +1,102; symbols: _write_synthetic_tokenizer, has_blackwell, truncate_text_config, make_inkling_dummy_checkpoint, touching `_write_synthetic_tokenizer, has_blackwell, truncate_text_config`; `test/runtime/models/test_inkling_models.py` modified +94/-48 (142 lines); hunks: -1,13 +1,13; -26,31 +26,57; symbols: TestInklingConfigRegistry, test_config_registry, test_get_config_loads_tiny_fixture, test_get_config_loads_hub_snapshot, touching `TestInklingConfigRegistry, test_config_registry, test_get_config_loads_tiny_fixture`; `test/runtime/models/test_inkling_multimodal.py` modified +76/-59 (135 lines); hunks: -7,14 +7,14; -24,7 +24,6; symbols: test_arch_flags, test_placeholder_token_ids, test_mrope_is_noop_for_tml, TestInklingAudioTower, touching `test_arch_flags, test_placeholder_token_ids, test_mrope_is_noop_for_tml`; `python/tokenspeed/runtime/configs/inkling_config.py` modified +4/-2 (6 lines); hunks: -109,8 +109,10 @@ def inkling_kv_heads_for_layer(; symbols: inkling_kv_heads_for_layer, touching `inkling_kv_heads_for_layer`.
- Code diff details:
  - `test/runtime/models/inkling_fixtures.py` modified +90/-174 (264 lines); hunks: -1,186 +1,102; symbols: _write_synthetic_tokenizer, has_blackwell, truncate_text_config, make_inkling_dummy_checkpoint
  - `test/runtime/models/test_inkling_models.py` modified +94/-48 (142 lines); hunks: -1,13 +1,13; -26,31 +26,57; symbols: TestInklingConfigRegistry, test_config_registry, test_get_config_loads_tiny_fixture, test_get_config_loads_hub_snapshot
  - `test/runtime/models/test_inkling_multimodal.py` modified +76/-59 (135 lines); hunks: -7,14 +7,14; -24,7 +24,6; symbols: test_arch_flags, test_placeholder_token_ids, test_mrope_is_noop_for_tml, TestInklingAudioTower
  - `python/tokenspeed/runtime/configs/inkling_config.py` modified +4/-2 (6 lines); hunks: -109,8 +109,10 @@ def inkling_kv_heads_for_layer(; symbols: inkling_kv_heads_for_layer
  - `test/runtime/test_inkling_load_weights.py` modified +89/-66 (155 lines); hunks: -1,48 +1,54; -59,6 +65,22 @@ def _deinterleave_rows(w: torch.Tensor) -> torch.Tensor:; symbols: _build_model, _deinterleave_rows, _make_tower_tensors, _make_checkpoint_tensors
- Key code excerpts:

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

- Extracted files (not manually reviewed):
  - tests: `test/runtime/models/inkling_fixtures.py` modified +90/-174; `test/runtime/models/test_inkling_models.py` modified +94/-48; `test/runtime/models/test_inkling_multimodal.py` modified +76/-59; `test/runtime/test_inkling_load_weights.py` modified +89/-66; `test/runtime/test_inkling_mtp_load_weights.py` modified +25/-24; `test/runtime/test_inkling_reference_parity.py` modified +22/-16
  - runtime: `python/tokenspeed/runtime/configs/inkling_config.py` modified +4/-2
- Risk and verification: The diff ships test coverage in `test/runtime/models/inkling_fixtures.py`, `test/runtime/models/test_inkling_models.py`, `test/runtime/models/test_inkling_multimodal.py`, `test/runtime/test_inkling_activation_parity.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #996 - refactor(inkling): ring sconv state with in-kernel persistence and speculative checkpoint publish

- Link: https://github.com/lightseekorg/tokenspeed/pull/996
- Status/date: merged / 2026-08-08
- Trace source: `git log --name-only -- <model-files>` found it through `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/inkling.py`, `python/tokenspeed/runtime/models/inkling.py`, `test/runtime/models/test_inkling_models.py`, `test/runtime/test_inkling_activation_parity.py`, `test/runtime/test_inkling_mtp_conv_state.py`; associated commits `27008e78184c`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 10 files, +1338/-2426, 4704 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/tokenspeed/runtime/layers/attention/backends/inkling.py` modified +122/-555 (677 lines); hunks: -21,8 +21,11; -51,7 +54,7; symbols: InklingConvMetadata, InklingConvStatePool, __init__, layer_state, touching `InklingConvMetadata, InklingConvStatePool, __init__`; `python/tokenspeed/runtime/models/inkling.py` modified +35/-149 (184 lines); hunks: -81,12 +81,7; -144,10 +139,6; symbols: _sconv_apply, touching `_sconv_apply`; `test/runtime/models/test_inkling_models.py` modified +21/-13 (34 lines); hunks: -271,20 +271,19 @@ def _make_ctx(self, num_layers, num_slots, conv_dim, kerne...; -325,6 +324,8 @@ def test_prefill_then_decode_parity(self):; symbols: _make_ctx, test_prefill_then_decode_parity, touching `_make_ctx, test_prefill_then_decode_parity`; `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/inkling.py` modified +10/-17 (27 lines); hunks: -205,26 +205,22 @@ def _workspace_bytes(; -254,7 +250,7 @@ def prepare_inkling_cache(; symbols: _workspace_bytes, prepare_inkling_cache, touching `_workspace_bytes, prepare_inkling_cache`.
- Code diff details:
  - `python/tokenspeed/runtime/layers/attention/backends/inkling.py` modified +122/-555 (677 lines); hunks: -21,8 +21,11; -51,7 +54,7; symbols: InklingConvMetadata, InklingConvStatePool, __init__, layer_state
  - `python/tokenspeed/runtime/models/inkling.py` modified +35/-149 (184 lines); hunks: -81,12 +81,7; -144,10 +139,6; symbols: _sconv_apply
  - `test/runtime/models/test_inkling_models.py` modified +21/-13 (34 lines); hunks: -271,20 +271,19 @@ def _make_ctx(self, num_layers, num_slots, conv_dim, kerne...; -325,6 +324,8 @@ def test_prefill_then_decode_parity(self):; symbols: _make_ctx, test_prefill_then_decode_parity
  - `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/inkling.py` modified +10/-17 (27 lines); hunks: -205,26 +205,22 @@ def _workspace_bytes(; -254,7 +250,7 @@ def prepare_inkling_cache(; symbols: _workspace_bytes, prepare_inkling_cache
  - `test/runtime/test_inkling_mtp_conv_state.py` modified +561/-245 (806 lines); hunks: -1,12 +1,14; -35,46 +37,11 @@ class HistoryBackend:; symbols: HistoryBackend, test_checkpoint_publication_masks_padded_rows_before_indexing, TestInklingConvSpecState, TestInklingConvRingState
- Key code excerpts:

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

- Extracted files (not manually reviewed):
  - runtime: `python/tokenspeed/runtime/layers/attention/backends/inkling.py` modified +122/-555; `python/tokenspeed/runtime/models/inkling.py` modified +35/-149; `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/inkling.py` modified +10/-17
  - tests: `test/runtime/models/test_inkling_models.py` modified +21/-13; `test/runtime/test_inkling_mtp_conv_state.py` modified +561/-245; `test/runtime/test_inkling_activation_parity.py` modified +2/-0
- Risk and verification: The diff ships test coverage in `test/runtime/models/test_inkling_models.py`, `test/runtime/test_inkling_activation_parity.py`, `test/runtime/test_inkling_mtp_conv_state.py`, `tokenspeed-kernel/test/ops/test_conv.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #1040 - feat(inkling): paged sconv checkpoints for the MTP draft + unified fused sconv kernels

- Link: https://github.com/lightseekorg/tokenspeed/pull/1040
- Status/date: merged / 2026-08-11
- Trace source: `git log --name-only -- <model-files>` found it through `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/inkling.py`, `python/tokenspeed/runtime/models/inkling.py`, `test/runtime/models/test_inkling_models.py`, `test/runtime/test_inkling_activation_parity.py`, `test/runtime/test_inkling_mtp_conv_state.py`; associated commits `6739113f945e`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 14 files, +1159/-1201, 3165 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/tokenspeed/runtime/layers/attention/backends/inkling.py` modified +91/-446 (537 lines); hunks: -71,18 +71,6; -108,16 +96,14 @@ class InklingConvMetadata:; symbols: ShortConvCheckpointMetadata, InklingConvMetadata, InklingConvStatePool, __init__, touching `ShortConvCheckpointMetadata, InklingConvMetadata, InklingConvStatePool`; `test/runtime/models/test_inkling_models.py` modified +0/-112 (112 lines); hunks: -253,117 +253,5 @@ def test_deferred_matches_finalized(self):; symbols: test_deferred_matches_finalized, _ref_sconv, TestInklingShortConvolution, _make_ctx, touching `test_deferred_matches_finalized, _ref_sconv, TestInklingShortConvolution`; `python/tokenspeed/runtime/models/inkling.py` modified +29/-61 (90 lines); hunks: -81,7 +81,7; -367,66 +367,44 @@ def _sconv_apply(; symbols: _sconv_apply, InklingRelLogitsProj, touching `_sconv_apply, InklingRelLogitsProj`; `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/inkling.py` modified +5/-15 (20 lines); hunks: -4,7 +4,6; -268,20 +267,11 @@ def prepare_inkling_cache(; symbols: prepare_inkling_cache, touching `prepare_inkling_cache`.
- Code diff details:
  - `python/tokenspeed/runtime/layers/attention/backends/inkling.py` modified +91/-446 (537 lines); hunks: -71,18 +71,6; -108,16 +96,14 @@ class InklingConvMetadata:; symbols: ShortConvCheckpointMetadata, InklingConvMetadata, InklingConvStatePool, __init__
  - `test/runtime/models/test_inkling_models.py` modified +0/-112 (112 lines); hunks: -253,117 +253,5 @@ def test_deferred_matches_finalized(self):; symbols: test_deferred_matches_finalized, _ref_sconv, TestInklingShortConvolution, _make_ctx
  - `python/tokenspeed/runtime/models/inkling.py` modified +29/-61 (90 lines); hunks: -81,7 +81,7; -367,66 +367,44 @@ def _sconv_apply(; symbols: _sconv_apply, InklingRelLogitsProj
  - `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/inkling.py` modified +5/-15 (20 lines); hunks: -4,7 +4,6; -268,20 +267,11 @@ def prepare_inkling_cache(; symbols: prepare_inkling_cache
  - `test/runtime/test_inkling_mtp_conv_state.py` modified +114/-124 (238 lines); hunks: -20,6 +20,13; -130,6 +137,10 @@ def test_ring_holds_window_for_every_accept(self):; symbols: _inert_publish, TestInklingCacheContract, test_wrapper_consumes_history_and_checkpoint_state, test_ring_holds_window_for_every_accept
- Key code excerpts:

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

- Extracted files (not manually reviewed):
  - runtime: `python/tokenspeed/runtime/layers/attention/backends/inkling.py` modified +91/-446; `python/tokenspeed/runtime/models/inkling.py` modified +29/-61; `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/inkling.py` modified +5/-15
  - tests: `test/runtime/models/test_inkling_models.py` modified +0/-112; `test/runtime/test_inkling_mtp_conv_state.py` modified +114/-124; `test/runtime/test_inkling_activation_parity.py` modified +72/-2
- Risk and verification: The diff ships test coverage in `test/runtime/models/test_inkling_models.py`, `test/runtime/test_cache_memory_plan.py`, `test/runtime/test_cache_setup.py`, `test/runtime/test_inkling_activation_parity.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #1110 - perf(inkling): eliminate per-step elementwise/copy kernels on the decode path

- Link: https://github.com/lightseekorg/tokenspeed/pull/1110
- Status/date: merged / 2026-08-15
- Trace source: `git log --name-only -- <model-files>` found it through `python/tokenspeed/runtime/configs/inkling_config.py`, `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/inkling.py`, `python/tokenspeed/runtime/models/inkling.py`, `test/runtime/test_inkling_activation_parity.py`, `test/runtime/test_inkling_load_weights.py` and 6 files; associated commits `21cfc7c4199e`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 20 files, +359/-201, 1418 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/tokenspeed/runtime/models/inkling.py` modified +18/-39 (57 lines); hunks: -44,8 +44,8; -81,6 +81,9; symbols: _load_block_param, compute_log_scaling_tau, _apply_log_scaling_tau, InklingShortConvolution, touching `_load_block_param, compute_log_scaling_tau, _apply_log_scaling_tau`; `python/tokenspeed/runtime/configs/inkling_config.py` modified +2/-29 (31 lines); hunks: -101,35 +101,6 @@ def inkling_conv_stream_layout(; -184,6 +155,7 @@ class InklingModelConfig(PretrainedConfig):; symbols: inkling_conv_stream_layout, inkling_kv_heads_for_layer, inkling_mtp_text_config, InklingModelConfig, touching `inkling_conv_stream_layout, inkling_kv_heads_for_layer, inkling_mtp_text_config`; `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/inkling.py` modified +8/-3 (11 lines); hunks: -148,11 +148,16 @@ def inkling_cache_fields(; symbols: inkling_cache_fields, inkling_layer_kv_head_counts, touching `inkling_cache_fields, inkling_layer_kv_head_counts`; `python/tokenspeed/runtime/layers/attention/backends/inkling.py` modified +5/-0 (5 lines); hunks: -642,6 +642,7 @@ def forward_decode(; -695,6 +696,7 @@ def forward_decode(; symbols: forward_decode, forward_extend, touching `forward_decode, forward_extend`.
- Code diff details:
  - `python/tokenspeed/runtime/models/inkling.py` modified +18/-39 (57 lines); hunks: -44,8 +44,8; -81,6 +81,9; symbols: _load_block_param, compute_log_scaling_tau, _apply_log_scaling_tau, InklingShortConvolution
  - `python/tokenspeed/runtime/configs/inkling_config.py` modified +2/-29 (31 lines); hunks: -101,35 +101,6 @@ def inkling_conv_stream_layout(; -184,6 +155,7 @@ class InklingModelConfig(PretrainedConfig):; symbols: inkling_conv_stream_layout, inkling_kv_heads_for_layer, inkling_mtp_text_config, InklingModelConfig
  - `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/inkling.py` modified +8/-3 (11 lines); hunks: -148,11 +148,16 @@ def inkling_cache_fields(; symbols: inkling_cache_fields, inkling_layer_kv_head_counts
  - `python/tokenspeed/runtime/layers/attention/backends/inkling.py` modified +5/-0 (5 lines); hunks: -642,6 +642,7 @@ def forward_decode(; -695,6 +696,7 @@ def forward_decode(; symbols: forward_decode, forward_extend
  - `test/runtime/test_inkling_load_weights.py` modified +12/-44 (56 lines); hunks: -211,50 +211,24 @@ def test_full_parameter_coverage(self):; -458,24 +432,18 @@ def _param(self, model, name):; symbols: test_full_parameter_coverage, test_qkvr_fusion_and_kv_replication, test_qkvr_fusion, test_dense_mlp_deinterleave_and_scale
- Key code excerpts:

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

- Extracted files (not manually reviewed):
  - runtime: `python/tokenspeed/runtime/models/inkling.py` modified +18/-39; `python/tokenspeed/runtime/configs/inkling_config.py` modified +2/-29; `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/inkling.py` modified +8/-3; `python/tokenspeed/runtime/layers/attention/backends/inkling.py` modified +5/-0
  - tests: `test/runtime/test_inkling_load_weights.py` modified +12/-44; `test/runtime/test_inkling_mtp_text_config.py` modified +1/-15; `test/runtime/test_inkling_activation_parity.py` modified +6/-5
- Risk and verification: The diff ships test coverage in `test/runtime/test_inkling_activation_parity.py`, `test/runtime/test_inkling_load_weights.py`, `test/runtime/test_inkling_mtp_text_config.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #1150 - ci(inkling): switch Inkling CI from GSM8K to AIME25

- Link: https://github.com/lightseekorg/tokenspeed/pull/1150
- Status/date: merged / 2026-08-19
- Trace source: `git log --name-only -- <model-files>` found it through `test/ci/eval/inkling-mxfp4-mtp-evalscope-aime25.yaml`, `test/ci/eval/inkling-nvfp4-mtp-evalscope-aime25.yaml`; associated commits `2edd9436d22d`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 4 files, +17/-15, 86 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `test/ci/eval/inkling-mxfp4-mtp-evalscope-aime25.yaml` renamed +7/-6 (13 lines); hunks: -1,5 +1,5; -18,7 +18,7 @@ server:; `test/ci/eval/inkling-nvfp4-mtp-evalscope-aime25.yaml` renamed +7/-6 (13 lines); hunks: -1,5 +1,5; -18,7 +18,7 @@ server:.
- Code diff details:
  - `test/ci/eval/inkling-mxfp4-mtp-evalscope-aime25.yaml` renamed +7/-6 (13 lines); hunks: -1,5 +1,5; -18,7 +18,7 @@ server:
  - `test/ci/eval/inkling-nvfp4-mtp-evalscope-aime25.yaml` renamed +7/-6 (13 lines); hunks: -1,5 +1,5; -18,7 +18,7 @@ server:
- Key code excerpts:

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

- Extracted files (not manually reviewed):
  - tests: `test/ci/eval/inkling-mxfp4-mtp-evalscope-aime25.yaml` renamed +7/-6; `test/ci/eval/inkling-nvfp4-mtp-evalscope-aime25.yaml` renamed +7/-6
- Risk and verification: The diff ships test coverage in `test/ci/eval/inkling-mxfp4-mtp-evalscope-aime25.yaml`, `test/ci/eval/inkling-nvfp4-mtp-evalscope-aime25.yaml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #1960 - ci(amd): Run all Inkling AIME25 problems concurrently

- Link: https://github.com/lightseekorg/tokenspeed/pull/1960
- Status/date: merged / 2026-10-03
- Trace source: `git log --name-only -- <model-files>` found it through `test/ci/eval/inkling-mxfp4-mtp-evalscope-aime25.yaml`; associated commits `0697a710cc68`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +4/-2, 26 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `test/ci/eval/inkling-mxfp4-mtp-evalscope-aime25.yaml` modified +4/-2 (6 lines); hunks: -13,13 +13,15 @@ env:; -49,7 +51,7 @@ eval:.
- Code diff details:
  - `test/ci/eval/inkling-mxfp4-mtp-evalscope-aime25.yaml` modified +4/-2 (6 lines); hunks: -13,13 +13,15 @@ env:; -49,7 +51,7 @@ eval:
- Key code excerpts:

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

- Extracted files (not manually reviewed):
  - tests: `test/ci/eval/inkling-mxfp4-mtp-evalscope-aime25.yaml` modified +4/-2
- Risk and verification: The diff ships test coverage in `test/ci/eval/inkling-mxfp4-mtp-evalscope-aime25.yaml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
