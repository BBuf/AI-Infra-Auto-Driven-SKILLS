# SGLang Qwen4-Exp (Qwen3.8-Flash-Next) Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `python/sglang/srt/arg_groups/model_overrides/qwen4_exp.py` | [#39614](https://github.com/sgl-project/sglang/pull/39614) |
| `python/sglang/srt/configs/qwen4_exp.py` | no direct PR-number commit |
| `python/sglang/srt/models/qwen4_exp.py` | [#38878](https://github.com/sgl-project/sglang/pull/38878), [#39928](https://github.com/sgl-project/sglang/pull/39928), [#40501](https://github.com/sgl-project/sglang/pull/40501), [#42478](https://github.com/sgl-project/sglang/pull/42478) |
| `python/sglang/srt/models/qwen4_exp_mtp.py` | no direct PR-number commit |
| `python/sglang/srt/models/qwen4_exp_ple_table.py` | no direct PR-number commit |
| `test/registered/e2e/models/test_qwen4_exp_models.py` | [#39662](https://github.com/sgl-project/sglang/pull/39662) |
| `test/registered/unit/layer_boundary/test_qwen4_exp_ple_rows.py` | [#42478](https://github.com/sgl-project/sglang/pull/42478) |
| `test/registered/unit/models/test_qwen4_exp_ple_table.py` | no direct PR-number commit |

## PR Coverage Summary

- Git-traced PRs: 6
- Extra PRs preserved from existing docs: 0
- Total PRs in this document: 6
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-09-16 | [#39662](https://github.com/sgl-project/sglang/pull/39662) | merged | [qwen 3.8 next] change the testing model in test_qwen4_exp_models.py | `test/registered/e2e/models/test_qwen4_exp_models.py` |
| 2026-09-17 | [#38878](https://github.com/sgl-project/sglang/pull/38878) | merged | [AMD] Load fused shared experts for Qwen4-Exp and Qwen3.5 MTP | `python/sglang/srt/models/qwen4_exp.py` |
| 2026-09-20 | [#39928](https://github.com/sgl-project/sglang/pull/39928) | merged | [Qwen4-Exp] Build the offloaded PLE table on the meta device so --ple-offload-embedding never materialises it on the accelerator | `python/sglang/srt/models/qwen4_exp.py` |
| 2026-09-22 | [#40501](https://github.com/sgl-project/sglang/pull/40501) | merged | [Qwen3.8-Next] Pipeline-parallel serving and PD-prefill MTP for Qwen4-Exp | `python/sglang/srt/models/qwen4_exp.py` |
| 2026-09-30 | [#39614](https://github.com/sgl-project/sglang/pull/39614) | merged | [Qwen4-Exp] Optional fp8 (e4m3) storage for the compressed QSA indexer cache | `python/sglang/srt/arg_groups/model_overrides/qwen4_exp.py` |
| 2026-10-04 | [#42478](https://github.com/sgl-project/sglang/pull/42478) | merged | [Refactor] Build the Qwen4 experimental decoders from stage boundaries | `python/sglang/srt/models/qwen4_exp.py`, `test/registered/unit/layer_boundary/test_qwen4_exp_ple_rows.py` |

## Per-PR Diff Audit Cards

### PR #39662 - [qwen 3.8 next] change the testing model in test_qwen4_exp_models.py

- Link: https://github.com/sgl-project/sglang/pull/39662
- Status/date: merged / 2026-09-16
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/e2e/models/test_qwen4_exp_models.py`; associated commits `43390f63f57c`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `test/registered/e2e/models/test_qwen4_exp_models.py` modified +1/-1 (2 lines); hunks: -15,7 +15,7.
- Code diff details:
  - `test/registered/e2e/models/test_qwen4_exp_models.py` modified +1/-1 (2 lines); hunks: -15,7 +15,7
- Key code excerpts:

```diff
diff -- test/registered/e2e/models/test_qwen4_exp_models.py
@@ -15,7 +15,7 @@
-MODEL = "RadixArk/Qwen3.8-Flash-Next-NVFP4"
+MODEL = "nvidia/Qwen3.8-Flash-Next-NVFP4"
```

- Extracted files (not manually reviewed):
  - tests: `test/registered/e2e/models/test_qwen4_exp_models.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `test/registered/e2e/models/test_qwen4_exp_models.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #38878 - [AMD] Load fused shared experts for Qwen4-Exp and Qwen3.5 MTP

- Link: https://github.com/sgl-project/sglang/pull/38878
- Status/date: merged / 2026-09-17
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen4_exp.py`; associated commits `11c35b8433e8`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +37/-10, 96 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/models/qwen4_exp.py` modified +24/-2 (26 lines); hunks: -61,7 +61,9; -1786,12 +1788,21 @@ def load_weights(self, weights: Iterable[Tuple[str, torc...; symbols: load_weights, load_qwen4_exp_ple_shard, touching `load_weights, load_qwen4_exp_ple_shard`.
- Code diff details:
  - `python/sglang/srt/models/qwen4_exp.py` modified +24/-2 (26 lines); hunks: -61,7 +61,9; -1786,12 +1788,21 @@ def load_weights(self, weights: Iterable[Tuple[str, torc...; symbols: load_weights, load_qwen4_exp_ple_shard
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen4_exp.py
@@ -61,7 +61,9 @@
-from sglang.srt.utils import logger
+from sglang.srt.utils import get_bool_env_var, is_hip, logger
+_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and is_hip()
@@ -1786,12 +1788,21 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
+        # A fused shared expert lives in routed slot `num_experts`, so the
+        # mapping has to cover one more expert than the config declares.
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/models/qwen4_exp.py` modified +24/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/models/qwen4_exp.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #39928 - [Qwen4-Exp] Build the offloaded PLE table on the meta device so --ple-offload-embedding never materialises it on the accelerator

- Link: https://github.com/sgl-project/sglang/pull/39928
- Status/date: merged / 2026-09-20
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen4_exp.py`; associated commits `e97614d10c8e`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +74/-18, 136 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/models/qwen4_exp.py` modified +26/-18 (44 lines); hunks: -508,20 +508,32 @@ def __init__(; -771,6 +783,8 @@ class Qwen4ExpPinnedHostEmbedding(VocabParallelEmbedding):; symbols: __init__, _splitmix64, Qwen4ExpPinnedHostEmbedding, touching `__init__, _splitmix64, Qwen4ExpPinnedHostEmbedding`.
- Code diff details:
  - `python/sglang/srt/models/qwen4_exp.py` modified +26/-18 (44 lines); hunks: -508,20 +508,32 @@ def __init__(; -771,6 +783,8 @@ class Qwen4ExpPinnedHostEmbedding(VocabParallelEmbedding):; symbols: __init__, _splitmix64, Qwen4ExpPinnedHostEmbedding
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen4_exp.py
@@ -508,20 +508,32 @@ def __init__(
-        self.ngram_embedding = VocabParallelEmbedding(
-            padded_vocab_size,
-            self.head_dim_per_ngram,
-            params_dtype=(
-                torch.float8_e4m3fn
-                if _ple_table_is_fp8(config, quant_config, ngram_prefix)
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/models/qwen4_exp.py` modified +26/-18
- Risk and verification: The diff ships test coverage in `test/registered/kernels/ops/embeddings/test_qwen4_ple_offload.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #40501 - [Qwen3.8-Next] Pipeline-parallel serving and PD-prefill MTP for Qwen4-Exp

- Link: https://github.com/sgl-project/sglang/pull/40501
- Status/date: merged / 2026-09-22
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen4_exp.py`; associated commits `6fd98c98b95f`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 8 files, +194/-43, 424 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/models/qwen4_exp.py` modified +75/-16 (91 lines); hunks: -2,7 +2,7; -46,9 +46,13; symbols: Qwen4ExpModel, _build_embed_tokens, __init__, forward, touching `Qwen4ExpModel, _build_embed_tokens, __init__`.
- Code diff details:
  - `python/sglang/srt/models/qwen4_exp.py` modified +75/-16 (91 lines); hunks: -2,7 +2,7; -46,9 +46,13; symbols: Qwen4ExpModel, _build_embed_tokens, __init__, forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen4_exp.py
@@ -2,7 +2,7 @@
-from typing import Any, Iterable, Optional, Set, Tuple
+from typing import Any, Iterable, Optional, Set, Tuple, Union
@@ -46,9 +46,13 @@
-from sglang.srt.layers.utils import get_layer_id
+from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
-from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/models/qwen4_exp.py` modified +75/-16
- Risk and verification: The diff ships test coverage in `test/registered/unit/disaggregation/test_disaggregation_wire.py`, `test/registered/unit/server_args/test_server_args.py`, `test/registered/unit/spec/test_eagle_worker_v2_topk1_fastpath.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #39614 - [Qwen4-Exp] Optional fp8 (e4m3) storage for the compressed QSA indexer cache

- Link: https://github.com/sgl-project/sglang/pull/39614
- Status/date: merged / 2026-09-30
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/arg_groups/model_overrides/qwen4_exp.py`; associated commits `eb9c9ee99d47`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 13 files, +416/-54, 937 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/arg_groups/model_overrides/qwen4_exp.py` modified +11/-0 (11 lines); hunks: -75,6 +75,17 @@ def _qwen4_exp_overrides(server_args: Any, hf_config: Any) ->...; symbols: _qwen4_exp_overrides, touching `_qwen4_exp_overrides`.
- Code diff details:
  - `python/sglang/srt/arg_groups/model_overrides/qwen4_exp.py` modified +11/-0 (11 lines); hunks: -75,6 +75,17 @@ def _qwen4_exp_overrides(server_args: Any, hf_config: Any) ->...; symbols: _qwen4_exp_overrides
- Key code excerpts:

```diff
diff -- python/sglang/srt/arg_groups/model_overrides/qwen4_exp.py
@@ -75,6 +75,17 @@ def _qwen4_exp_overrides(server_args: Any, hf_config: Any) -> dict:
+    if cfg.qsa_indexer_dtype == "fp8_e4m3":
+        # fp8 scoring runs on TileLang fp8 GEMMs, validated on Hopper and Blackwell.
+        platform = get_platform()
+        if not (platform.is_cuda and (platform.is_sm90 or platform.is_sm100)):
+            raise ValueError(
+                "--qsa-indexer-dtype fp8_e4m3 requires a CUDA SM90/SM100 GPU"
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/arg_groups/model_overrides/qwen4_exp.py` modified +11/-0
- Risk and verification: The diff ships test coverage in `test/registered/kernels/ops/attention/qsa/test_qsa_indexer.py`, `test/registered/unit/disaggregation/test_disaggregation_wire.py`, `test/registered/unit/test_model_overrides.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #42478 - [Refactor] Build the Qwen4 experimental decoders from stage boundaries

- Link: https://github.com/sgl-project/sglang/pull/42478
- Status/date: merged / 2026-10-04
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen4_exp.py`, `test/registered/unit/layer_boundary/test_qwen4_exp_ple_rows.py`; associated commits `f07c1a398efd`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 10 files, +857/-307, 1453 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/models/qwen4_exp.py` modified +176/-161 (337 lines); hunks: -26,24 +26,27; -70,6 +73,7; symbols: forward, Qwen4ExpLayerExtensionMixin, _MixStreams, __init__, touching `forward, Qwen4ExpLayerExtensionMixin, _MixStreams`; `test/registered/unit/layer_boundary/test_qwen4_exp_ple_rows.py` added +64/-0 (64 lines); hunks: -0,0 +1,64; symbols: _residual_ops, unused, _sliced_over_attention_tp, TestQwen4ExpPleRows, touching `_residual_ops, unused, _sliced_over_attention_tp`.
- Code diff details:
  - `python/sglang/srt/models/qwen4_exp.py` modified +176/-161 (337 lines); hunks: -26,24 +26,27; -70,6 +73,7; symbols: forward, Qwen4ExpLayerExtensionMixin, _MixStreams, __init__
  - `test/registered/unit/layer_boundary/test_qwen4_exp_ple_rows.py` added +64/-0 (64 lines); hunks: -0,0 +1,64; symbols: _residual_ops, unused, _sliced_over_attention_tp, TestQwen4ExpPleRows
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen4_exp.py
@@ -26,24 +26,27 @@
-    attn_tp_all_gather,
-    get_dp_global_num_tokens,
-    get_global_dp_buffer,
-    get_local_dp_buffer,
-from sglang.srt.layers.layer_boundary import get_attn_tp_context
+from sglang.srt.layers.layer_boundary import (
diff -- test/registered/unit/layer_boundary/test_qwen4_exp_ple_rows.py
@@ -0,0 +1,64 @@
+"""A Qwen4-exp PLE layer reads its input on full attention rows."""
+import unittest
+from types import SimpleNamespace
+import test_declared_decoder_boundary as fixture
+from sglang.srt.layers.layer_boundary.contracts import BatchVariant
+from sglang.srt.layers.layer_boundary.layout import TokenAxis
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/models/qwen4_exp.py` modified +176/-161
  - tests: `test/registered/unit/layer_boundary/test_qwen4_exp_ple_rows.py` added +64/-0
- Risk and verification: The diff ships test coverage in `test/registered/unit/layer_boundary/test_gated_residual_ops.py`, `test/registered/unit/layer_boundary/test_moe_output_reduction.py`, `test/registered/unit/layer_boundary/test_qwen4_exp_ple_rows.py`, `test/registered/unit/layer_boundary/test_stage_producers_leave_sums.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
