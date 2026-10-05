# SGLang LongCat-Flash Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `docs/cookbook/autoregressive/Meituan/LongCat-2.0.mdx` | no direct PR-number commit |
| `docs/src/snippets/configs/meituan-longcat/longcat-2.0-benchmarks.jsx` | no direct PR-number commit |
| `docs/src/snippets/configs/meituan-longcat/longcat-2.0.jsx` | no direct PR-number commit |
| `python/sglang/srt/configs/longcat_flash.py` | [#9824](https://github.com/sgl-project/sglang/pull/9824), [#17838](https://github.com/sgl-project/sglang/pull/17838), [#30275](https://github.com/sgl-project/sglang/pull/30275) |
| `python/sglang/srt/models/longcat_flash.py` | [#9824](https://github.com/sgl-project/sglang/pull/9824), [#9916](https://github.com/sgl-project/sglang/pull/9916), [#14007](https://github.com/sgl-project/sglang/pull/14007), [#14161](https://github.com/sgl-project/sglang/pull/14161), [#17838](https://github.com/sgl-project/sglang/pull/17838), [#30247](https://github.com/sgl-project/sglang/pull/30247), [#30275](https://github.com/sgl-project/sglang/pull/30275), [#31311](https://github.com/sgl-project/sglang/pull/31311), [#40799](https://github.com/sgl-project/sglang/pull/40799), [#41436](https://github.com/sgl-project/sglang/pull/41436) |
| `python/sglang/srt/models/longcat_flash_nextn.py` | [#9824](https://github.com/sgl-project/sglang/pull/9824), [#9916](https://github.com/sgl-project/sglang/pull/9916), [#32125](https://github.com/sgl-project/sglang/pull/32125) |
| `test/registered/e2e/models_large/test_longcat_flash_lite_fp8.py` | no direct PR-number commit |
| `test/registered/unit/layer_boundary/test_longcat_flash_shortcut.py` | no direct PR-number commit |
| `test/registered/unit/models/test_longcat_flash_router_hpc_gemm.py` | [#30247](https://github.com/sgl-project/sglang/pull/30247) |

## PR Coverage Summary

- Git-traced PRs: 11
- Extra PRs preserved from existing docs: 0
- Total PRs in this document: 11
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2025-08-31 | [#9824](https://github.com/sgl-project/sglang/pull/9824) | merged | [Model] Support Meituan LongCat-Flash && LongCat-Flash-MTP | `python/sglang/srt/models/longcat_flash.py`, `python/sglang/srt/models/longcat_flash_nextn.py`, `python/sglang/srt/configs/longcat_flash.py` |
| 2025-09-02 | [#9916](https://github.com/sgl-project/sglang/pull/9916) | merged | [Fix] fix the issue encountered when inference LongCat-Flash/MTP EP MoE on b200 | `python/sglang/srt/models/longcat_flash.py`, `python/sglang/srt/models/longcat_flash_nextn.py` |
| 2025-11-26 | [#14007](https://github.com/sgl-project/sglang/pull/14007) | merged | fix: cuda graph issue while running longcat_flash | `python/sglang/srt/models/longcat_flash.py` |
| 2025-11-30 | [#14161](https://github.com/sgl-project/sglang/pull/14161) | merged | feat: longcat flash add aux layers capture for eagle3 | `python/sglang/srt/models/longcat_flash.py` |
| 2026-03-09 | [#17838](https://github.com/sgl-project/sglang/pull/17838) | merged | Feature/support longcat flash lite | `python/sglang/srt/models/longcat_flash.py`, `python/sglang/srt/configs/longcat_flash.py` |
| 2026-07-07 | [#30275](https://github.com/sgl-project/sglang/pull/30275) | merged | [Model] Support LongCat 2.0 FP8 | `python/sglang/srt/models/longcat_flash.py`, `python/sglang/srt/configs/longcat_flash.py` |
| 2026-07-20 | [#31311](https://github.com/sgl-project/sglang/pull/31311) | merged | Fix LongCat-2.0 real EP (deepep): double all-reduce + ScMoE RoPE crash | `python/sglang/srt/models/longcat_flash.py` |
| 2026-07-21 | [#30247](https://github.com/sgl-project/sglang/pull/30247) | merged | Optimize LongCat-Flash router GEMM with the HPC-Ops bf16xfp32 kernel | `test/registered/unit/models/test_longcat_flash_router_hpc_gemm.py`, `python/sglang/srt/models/longcat_flash.py` |
| 2026-07-24 | [#32125](https://github.com/sgl-project/sglang/pull/32125) | merged | ci: add LongCat-Flash-Lite-FP8 8-GPU nightly test + fix NextN rope_theta | `python/sglang/srt/models/longcat_flash_nextn.py` |
| 2026-09-23 | [#40799](https://github.com/sgl-project/sglang/pull/40799) | merged | [Fix] Avoid duplicate residual in LongCat MoE shortcut | `python/sglang/srt/models/longcat_flash.py` |
| 2026-09-27 | [#41436](https://github.com/sgl-project/sglang/pull/41436) | merged | [Fix] LongCat-Flash under attention DP: branch and merge the dense FFNs through the communicators | `python/sglang/srt/models/longcat_flash.py` |

## Per-PR Diff Audit Cards

### PR #9824 - [Model] Support Meituan LongCat-Flash && LongCat-Flash-MTP

- Link: https://github.com/sgl-project/sglang/pull/9824
- Status/date: merged / 2025-08-31
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/configs/longcat_flash.py`, `python/sglang/srt/models/longcat_flash.py`, `python/sglang/srt/models/longcat_flash_nextn.py`; associated commits `5e194b21437f`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 10 files, +1940/-11, 2043 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/models/longcat_flash.py` added +1015/-0 (1015 lines); hunks: -0,0 +1,1015; symbols: LongcatFlashMLP, __init__, forward, LongcatFlashRouter, touching `LongcatFlashMLP, __init__, forward`; `python/sglang/srt/models/longcat_flash_nextn.py` added +691/-0 (691 lines); hunks: -0,0 +1,691; symbols: LongcatFlashDenseDecoderLayer, __init__, forward, LongcatFlashModelNextN, touching `LongcatFlashDenseDecoderLayer, __init__, forward`; `python/sglang/srt/configs/longcat_flash.py` added +104/-0 (104 lines); hunks: -0,0 +1,104; symbols: LongcatFlashConfig, __init__, touching `LongcatFlashConfig, __init__`.
- Code diff details:
  - `python/sglang/srt/models/longcat_flash.py` added +1015/-0 (1015 lines); hunks: -0,0 +1,1015; symbols: LongcatFlashMLP, __init__, forward, LongcatFlashRouter
  - `python/sglang/srt/models/longcat_flash_nextn.py` added +691/-0 (691 lines); hunks: -0,0 +1,691; symbols: LongcatFlashDenseDecoderLayer, __init__, forward, LongcatFlashModelNextN
  - `python/sglang/srt/configs/longcat_flash.py` added +104/-0 (104 lines); hunks: -0,0 +1,104; symbols: LongcatFlashConfig, __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/longcat_flash.py
@@ -0,0 +1,1015 @@
+# Apache License, Version 2.0:
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/models/longcat_flash_nextn.py
@@ -0,0 +1,691 @@
+# Apache License, Version 2.0:
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/configs/longcat_flash.py
@@ -0,0 +1,104 @@
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/models/longcat_flash.py` added +1015/-0; `python/sglang/srt/models/longcat_flash_nextn.py` added +691/-0; `python/sglang/srt/configs/longcat_flash.py` added +104/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/__init__.py`, `python/sglang/srt/configs/longcat_flash.py`, `python/sglang/srt/configs/model_config.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9916 - [Fix] fix the issue encountered when inference LongCat-Flash/MTP EP MoE on b200

- Link: https://github.com/sgl-project/sglang/pull/9916
- Status/date: merged / 2025-09-02
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/longcat_flash.py`, `python/sglang/srt/models/longcat_flash_nextn.py`; associated commits `b7361cc4441d`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +49/-30, 126 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/models/longcat_flash.py` modified +26/-15 (41 lines); hunks: -651,9 +651,6 @@ def post_load_weights(self, weight_names=None):; -790,6 +787,9 @@ def post_load_weights(self, weight_names=None):; symbols: post_load_weights, _weight_requant_ue8m0, touching `post_load_weights, _weight_requant_ue8m0`; `python/sglang/srt/models/longcat_flash_nextn.py` modified +23/-15 (38 lines); hunks: -344,9 +344,6 @@ def post_load_weights(self):; -480,24 +477,35 @@ def post_load_weights(self):; symbols: post_load_weights, _weight_requant_ue8m0, load_weights, touching `post_load_weights, _weight_requant_ue8m0, load_weights`.
- Code diff details:
  - `python/sglang/srt/models/longcat_flash.py` modified +26/-15 (41 lines); hunks: -651,9 +651,6 @@ def post_load_weights(self, weight_names=None):; -790,6 +787,9 @@ def post_load_weights(self, weight_names=None):; symbols: post_load_weights, _weight_requant_ue8m0
  - `python/sglang/srt/models/longcat_flash_nextn.py` modified +23/-15 (38 lines); hunks: -344,9 +344,6 @@ def post_load_weights(self):; -480,24 +477,35 @@ def post_load_weights(self):; symbols: post_load_weights, _weight_requant_ue8m0, load_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/longcat_flash.py
@@ -651,9 +651,6 @@ def post_load_weights(self, weight_names=None):
-                # NOTE(HandH1998): Since `bmm_fp8` only supports per-tensor scale, we have to requantize `self_attn.kv_b_proj`.
-                # This may affect the accuracy of fp8 model.
-                # Fix deepseek v3 blockwise bmm by using deep_gemm
@@ -790,6 +787,9 @@ def post_load_weights(self, weight_names=None):
+        # TODO(linguoyuan) EPMoE not support DEEPGEMM_BLACKWELL, DeepEP needs to be supported in the future
+        deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0 = False
diff -- python/sglang/srt/models/longcat_flash_nextn.py
@@ -344,9 +344,6 @@ def post_load_weights(self):
-        # NOTE(HandH1998): Since `bmm_fp8` only supports per-tensor scale, we have to requantize `self_attn.kv_b_proj`.
-        # This may affect the accuracy of fp8 model.
-        # Fix deepseek v3 blockwise bmm by using deep_gemm
@@ -480,24 +477,35 @@ def post_load_weights(self):
-        for module in [
-            layer.self_attn.fused_qkv_a_proj_with_mqa,
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/models/longcat_flash.py` modified +26/-15; `python/sglang/srt/models/longcat_flash_nextn.py` modified +23/-15
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/longcat_flash.py`, `python/sglang/srt/models/longcat_flash_nextn.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #14007 - fix: cuda graph issue while running longcat_flash

- Link: https://github.com/sgl-project/sglang/pull/14007
- Status/date: merged / 2025-11-26
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/longcat_flash.py`; associated commits `685b9d82bd01`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +2/-0, 16 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/models/longcat_flash.py` modified +2/-0 (2 lines); hunks: -388,6 +388,7 @@ def __init__(; -402,6 +403,7 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `python/sglang/srt/models/longcat_flash.py` modified +2/-0 (2 lines); hunks: -388,6 +388,7 @@ def __init__(; -402,6 +403,7 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/longcat_flash.py
@@ -388,6 +388,7 @@ def __init__(
+                qkv_latent_func=self.self_attn[i].prepare_qkv_latent,
@@ -402,6 +403,7 @@ def __init__(
+            qkv_latent_func=self.self_attn[0].prepare_qkv_latent,
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/models/longcat_flash.py` modified +2/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/longcat_flash.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #14161 - feat: longcat flash add aux layers capture for eagle3

- Link: https://github.com/sgl-project/sglang/pull/14161
- Status/date: merged / 2025-11-30
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/longcat_flash.py`; associated commits `67e6ef4b2d24`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +25/-3, 78 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/models/longcat_flash.py` modified +25/-3 (28 lines); hunks: -32,7 +32,7; -511,6 +511,7 @@ def __init__(; symbols: __init__, get_input_embeddings, forward, LongcatFlashForCausalLM, touching `__init__, get_input_embeddings, forward`.
- Code diff details:
  - `python/sglang/srt/models/longcat_flash.py` modified +25/-3 (28 lines); hunks: -32,7 +32,7; -511,6 +511,7 @@ def __init__(; symbols: __init__, get_input_embeddings, forward, LongcatFlashForCausalLM
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/longcat_flash.py
@@ -32,7 +32,7 @@
-from typing import Iterable, Optional, Tuple
+from typing import Iterable, List, Optional, Tuple
@@ -511,6 +511,7 @@ def __init__(
+        self.layers_to_capture = []
@@ -536,7 +537,10 @@ def forward(
+        aux_hidden_states = []
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/models/longcat_flash.py` modified +25/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/longcat_flash.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17838 - Feature/support longcat flash lite

- Link: https://github.com/sgl-project/sglang/pull/17838
- Status/date: merged / 2026-03-09
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/configs/longcat_flash.py`, `python/sglang/srt/models/longcat_flash.py`; associated commits `eb4ba1bde254`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 16 files, +838/-15, 1156 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/models/longcat_flash.py` modified +29/-8 (37 lines); hunks: -64,6 +64,7; -329,7 +330,7 @@ def __init__(; symbols: __init__, forward, load_weights, touching `__init__, forward, load_weights`; `python/sglang/srt/configs/longcat_flash.py` modified +8/-0 (8 lines); hunks: -53,6 +53,9 @@ def __init__(; -102,3 +105,8 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/longcat_flash.py` modified +29/-8 (37 lines); hunks: -64,6 +64,7; -329,7 +330,7 @@ def __init__(; symbols: __init__, forward, load_weights
  - `python/sglang/srt/configs/longcat_flash.py` modified +8/-0 (8 lines); hunks: -53,6 +53,9 @@ def __init__(; -102,3 +105,8 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/longcat_flash.py
@@ -64,6 +64,7 @@
+from sglang.srt.layers.n_gram_embedding import NgramEmbedding
@@ -329,7 +330,7 @@ def __init__(
-                    rope_scaling=None,
+                    rope_scaling=getattr(config, "rope_scaling", None),
@@ -500,11 +501,22 @@ def __init__(
-        self.embed_tokens = VocabParallelEmbedding(
diff -- python/sglang/srt/configs/longcat_flash.py
@@ -53,6 +53,9 @@ def __init__(
+        ngram_vocab_size_ratio=None,
+        emb_neighbor_num=None,
+        emb_split_num=None,
@@ -102,3 +105,8 @@ def __init__(
+        self.use_ngram_embedding = ngram_vocab_size_ratio is not None
+        if self.use_ngram_embedding:
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/models/longcat_flash.py` modified +29/-8; `python/sglang/srt/configs/longcat_flash.py` modified +8/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/jit_kernel/csrc/ngram_embedding.cuh`, `python/sglang/jit_kernel/ngram_embedding.py`, `python/sglang/srt/configs/longcat_flash.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #30275 - [Model] Support LongCat 2.0 FP8

- Link: https://github.com/sgl-project/sglang/pull/30275
- Status/date: merged / 2026-07-07
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/configs/longcat_flash.py`, `python/sglang/srt/models/longcat_flash.py`; associated commits `e339c83f82e4`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 23 files, +481/-91, 1125 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/models/longcat_flash.py` modified +40/-11 (51 lines); hunks: -326,8 +326,8 @@ def __init__(; -420,18 +420,24 @@ def forward(; symbols: __init__, forward, forward_mlp, touching `__init__, forward, forward_mlp`; `python/sglang/srt/configs/longcat_flash.py` modified +12/-0 (12 lines); hunks: -56,6 +56,9 @@ def __init__(; -105,6 +108,15 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/longcat_flash.py` modified +40/-11 (51 lines); hunks: -326,8 +326,8 @@ def __init__(; -420,18 +420,24 @@ def forward(; symbols: __init__, forward, forward_mlp
  - `python/sglang/srt/configs/longcat_flash.py` modified +12/-0 (12 lines); hunks: -56,6 +56,9 @@ def __init__(; -105,6 +108,15 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/longcat_flash.py
@@ -326,8 +326,8 @@ def __init__(
-                    rope_theta=config.rope_parameters["rope_theta"],
-                    rope_scaling=None,
+                    rope_theta=config.rope_theta,
+                    rope_scaling=config.rope_scaling,
@@ -420,18 +420,24 @@ def forward(
+        prev_topk_indices: Optional[torch.Tensor],
diff -- python/sglang/srt/configs/longcat_flash.py
@@ -56,6 +56,9 @@ def __init__(
+        oe_vocab_size_ratio=None,
+        oe_neighbor_num=None,
+        oe_split_num=None,
@@ -105,6 +108,15 @@ def __init__(
+        if ngram_vocab_size_ratio is None:
+            ngram_vocab_size_ratio = oe_vocab_size_ratio
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/models/longcat_flash.py` modified +40/-11; `python/sglang/srt/configs/longcat_flash.py` modified +12/-0
- Risk and verification: The diff ships test coverage in `test/registered/jit/benchmark/bench_ngram_compute_decode.py`, `test/registered/jit/test_ngram_embedding.py`, `test/registered/unit/model_executor/test_ngram_token_table.py`, `test/registered/unit/test_model_overrides.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #31311 - Fix LongCat-2.0 real EP (deepep): double all-reduce + ScMoE RoPE crash

- Link: https://github.com/sgl-project/sglang/pull/31311
- Status/date: merged / 2026-07-20
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/longcat_flash.py`; associated commits `1843384c7a59`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +44/-1, 73 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/models/longcat_flash.py` modified +44/-1 (45 lines); hunks: -123,6 +123,24; -284,7 +302,12 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Ten...; symbols: _scmoe_align_rows, LongcatFlashMLP, __init__, forward, touching `_scmoe_align_rows, LongcatFlashMLP, __init__`.
- Code diff details:
  - `python/sglang/srt/models/longcat_flash.py` modified +44/-1 (45 lines); hunks: -123,6 +123,24; -284,7 +302,12 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Ten...; symbols: _scmoe_align_rows, LongcatFlashMLP, __init__, forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/longcat_flash.py
@@ -123,6 +123,24 @@
+def _scmoe_align_rows(t, target):
+    """Align a [rows,H] tensor to `target` rows across the attn-tp group:
+    all_gather when target>rows (target==rows*attn_tp_size), or take this rank's
+    contiguous segment when target<rows."""
+    if t is None or t.shape[0] == target:
+        return t
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/models/longcat_flash.py` modified +44/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/longcat_flash.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #30247 - Optimize LongCat-Flash router GEMM with the HPC-Ops bf16xfp32 kernel

- Link: https://github.com/sgl-project/sglang/pull/30247
- Status/date: merged / 2026-07-21
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/longcat_flash.py`, `test/registered/unit/models/test_longcat_flash_router_hpc_gemm.py`; associated commits `e4eea7ce2ffa`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 4 files, +336/-7, 393 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `test/registered/unit/models/test_longcat_flash_router_hpc_gemm.py` added +116/-0 (116 lines); hunks: -0,0 +1,116; symbols: _longcat_config, TestLongcatFlashRouterHpcGemm, _assert_dispatches_to_hpc_gemm, _assert_uses_classifier, touching `_longcat_config, TestLongcatFlashRouterHpcGemm, _assert_dispatches_to_hpc_gemm`; `python/sglang/srt/models/longcat_flash.py` modified +29/-2 (31 lines); hunks: -37,6 +37,7; -122,6 +123,15; symbols: LongcatFlashMLP, __init__, forward, touching `LongcatFlashMLP, __init__, forward`.
- Code diff details:
  - `test/registered/unit/models/test_longcat_flash_router_hpc_gemm.py` added +116/-0 (116 lines); hunks: -0,0 +1,116; symbols: _longcat_config, TestLongcatFlashRouterHpcGemm, _assert_dispatches_to_hpc_gemm, _assert_uses_classifier
  - `python/sglang/srt/models/longcat_flash.py` modified +29/-2 (31 lines); hunks: -37,6 +37,7; -122,6 +123,15; symbols: LongcatFlashMLP, __init__, forward
- Key code excerpts:

```diff
diff -- test/registered/unit/models/test_longcat_flash_router_hpc_gemm.py
@@ -0,0 +1,116 @@
+"""Unit tests for LongCat-Flash router GEMM dispatch to the HPC-Ops bf16xfp32 kernel."""
+import unittest
+from types import SimpleNamespace
+from unittest.mock import patch
+import torch
+from sglang.test.ci.ci_register import register_cpu_ci
diff -- python/sglang/srt/models/longcat_flash.py
@@ -37,6 +37,7 @@
+from sglang.jit_kernel.dsv4 import linear_bf16_fp32
@@ -122,6 +123,15 @@
+# Minimum m (num_tokens) from which the JIT bf16xfp32 router GEMM beats
+# cublas, benchmarked per router shape (hidden_size, n_routed_experts) on H200.
+_LONGCAT_FLASH_ROUTER_HPC_GEMM_MIN_M = {
+    # LongCat-Flash-Chat-FP8: 6144 hidden size, 512 routed experts + 256 zero experts.
```

- Extracted files (not manually reviewed):
  - tests: `test/registered/unit/models/test_longcat_flash_router_hpc_gemm.py` added +116/-0
  - runtime: `python/sglang/srt/models/longcat_flash.py` modified +29/-2
- Risk and verification: The diff ships test coverage in `test/registered/gemm/test_linear_bf16_fp32_hpc.py`, `test/registered/unit/models/test_longcat_flash_router_hpc_gemm.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #32125 - ci: add LongCat-Flash-Lite-FP8 8-GPU nightly test + fix NextN rope_theta

- Link: https://github.com/sgl-project/sglang/pull/32125
- Status/date: merged / 2026-07-24
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/longcat_flash_nextn.py`; associated commits `8389d79e43af`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +89/-2, 99 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/models/longcat_flash_nextn.py` modified +6/-2 (8 lines); hunks: -131,8 +131,12 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/longcat_flash_nextn.py` modified +6/-2 (8 lines); hunks: -131,8 +131,12 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/longcat_flash_nextn.py
@@ -131,8 +131,12 @@ def __init__(
-            rope_theta=config.rope_parameters["rope_theta"],
-            rope_scaling=None,
+            rope_theta=(
+                config.rope_parameters["rope_theta"]
+                if "rope_theta" in getattr(config, "rope_parameters", {})
+                else config.rope_theta
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/models/longcat_flash_nextn.py` modified +6/-2
- Risk and verification: The diff ships test coverage in `test/registered/8-gpu-models/test_longcat_flash_lite_fp8.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #40799 - [Fix] Avoid duplicate residual in LongCat MoE shortcut

- Link: https://github.com/sgl-project/sglang/pull/40799
- Status/date: merged / 2026-09-23
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/longcat_flash.py`; associated commits `0e80c73c925e`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +86/-1, 95 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/models/longcat_flash.py` modified +2/-1 (3 lines); hunks: -501,7 +501,8 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/longcat_flash.py` modified +2/-1 (3 lines); hunks: -501,7 +501,8 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/longcat_flash.py
@@ -501,7 +501,8 @@ def forward(
-        moe_residual = residual.clone()
+        # The final gather adds its residual; the dense branch already carries it.
+        moe_residual = torch.zeros_like(residual)
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/models/longcat_flash.py` modified +2/-1
- Risk and verification: The diff ships test coverage in `test/registered/unit/models/test_longcat_flash_shortcut.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #41436 - [Fix] LongCat-Flash under attention DP: branch and merge the dense FFNs through the communicators

- Link: https://github.com/sgl-project/sglang/pull/41436
- Status/date: merged / 2026-09-27
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/longcat_flash.py`; associated commits `0971450c168f`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 5 files, +298/-83, 503 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/sglang/srt/models/longcat_flash.py` modified +21/-56 (77 lines); hunks: -90,9 +90,8; -140,23 +139,6; symbols: _scmoe_align_rows, LongcatFlashMLP, __init__, touching `_scmoe_align_rows, LongcatFlashMLP, __init__`.
- Code diff details:
  - `python/sglang/srt/models/longcat_flash.py` modified +21/-56 (77 lines); hunks: -90,9 +90,8; -140,23 +139,6; symbols: _scmoe_align_rows, LongcatFlashMLP, __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/longcat_flash.py
@@ -90,9 +90,8 @@
-from sglang.srt.runtime_context import get_parallel
-from sglang.srt.runtime_context import get_parallel as _gp
+    get_parallel,
@@ -140,23 +139,6 @@
-def _scmoe_align_rows(t, target):
-    """Align a [rows,H] tensor to `target` rows across the attn-tp group:
```

- Extracted files (not manually reviewed):
  - runtime: `python/sglang/srt/models/longcat_flash.py` modified +21/-56
- Risk and verification: The diff ships test coverage in `test/registered/unit/layers/test_communicator_ffn_exit.py`, `test/registered/unit/layers/test_declared_decoder_boundary.py`, `test/registered/unit/models/test_longcat_flash_shortcut.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
