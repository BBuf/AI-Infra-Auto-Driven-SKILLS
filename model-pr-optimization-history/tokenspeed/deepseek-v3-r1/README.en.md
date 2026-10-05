# TokenSpeed DeepSeek V3/R1 Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `python/tokenspeed/runtime/models/deepseek_nextn.py` | [#217](https://github.com/lightseekorg/tokenspeed/pull/217), [#544](https://github.com/lightseekorg/tokenspeed/pull/544) |
| `python/tokenspeed/runtime/models/deepseek_v3.py` | [#217](https://github.com/lightseekorg/tokenspeed/pull/217), [#544](https://github.com/lightseekorg/tokenspeed/pull/544), [#602](https://github.com/lightseekorg/tokenspeed/pull/602), [#840](https://github.com/lightseekorg/tokenspeed/pull/840) |
| `test/runtime/models/test_deepseek_v3_loader.py` | [#602](https://github.com/lightseekorg/tokenspeed/pull/602) |

## PR Coverage Summary

- Git-traced PRs: 4
- Extra PRs preserved from existing docs: 0
- Total PRs in this document: 4
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-05-28 | [#217](https://github.com/lightseekorg/tokenspeed/pull/217) | merged | perf(Spec Decode): skip dead-position compute in draft catch-up step(decode) | `python/tokenspeed/runtime/models/deepseek_v3.py`, `python/tokenspeed/runtime/models/deepseek_nextn.py` |
| 2026-07-07 | [#602](https://github.com/lightseekorg/tokenspeed/pull/602) | merged | fix(deepseek-v3): Route DeepSeek merge state through unified backend | `test/runtime/models/test_deepseek_v3_loader.py`, `python/tokenspeed/runtime/models/deepseek_v3.py` |
| 2026-07-10 | [#544](https://github.com/lightseekorg/tokenspeed/pull/544) | merged | refactor(spec-decode): simplify deepseekV3/GLM attention path for #217 (3/3) | `python/tokenspeed/runtime/models/deepseek_v3.py`, `python/tokenspeed/runtime/models/deepseek_nextn.py` |
| 2026-07-29 | [#840](https://github.com/lightseekorg/tokenspeed/pull/840) | merged | refactor: backend seq_lens ownership and fix dsv3 | `python/tokenspeed/runtime/models/deepseek_v3.py` |

## Per-PR Diff Audit Cards

### PR #217 - perf(Spec Decode): skip dead-position compute in draft catch-up step(decode)

- Link: https://github.com/lightseekorg/tokenspeed/pull/217
- Status/date: merged / 2026-05-28
- Trace source: `git log --name-only -- <model-files>` found it through `python/tokenspeed/runtime/models/deepseek_nextn.py`, `python/tokenspeed/runtime/models/deepseek_v3.py`; associated commits `27e99fb86fa7`, `a9bc2188501c`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 9 files, +109/-38, 286 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/tokenspeed/runtime/models/deepseek_v3.py` modified +10/-0 (10 lines); hunks: -675,6 +675,10 @@ def forward(; -1153,6 +1157,9 @@ def forward(; symbols: forward, touching `forward`; `python/tokenspeed/runtime/models/deepseek_nextn.py` modified +4/-3 (7 lines); hunks: -141,11 +141,12 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/tokenspeed/runtime/models/deepseek_v3.py` modified +10/-0 (10 lines); hunks: -675,6 +675,10 @@ def forward(; -1153,6 +1157,9 @@ def forward(; symbols: forward
  - `python/tokenspeed/runtime/models/deepseek_nextn.py` modified +4/-3 (7 lines); hunks: -141,11 +141,12 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- python/tokenspeed/runtime/models/deepseek_v3.py
@@ -675,6 +675,10 @@ def forward(
+        if ctx.draft_first_step_reduce:
+            # KV already written; drop dead-position rows so o_proj / MLP /
+            # post-norms only run on one live row per request.
+            attn_output = attn_output.index_select(0, ctx.gather_ids)
@@ -1153,6 +1157,9 @@ def forward(
+            if ctx.draft_first_step_reduce:
diff -- python/tokenspeed/runtime/models/deepseek_nextn.py
@@ -141,11 +141,12 @@ def forward(
-            hidden_states, _ = self.shared_head.norm(hidden_states, residual)
-                hidden_states, _ = self.decoder.comm_manager.post_final_norm_comm(
-                    hidden_states, residual, ctx
+                hidden_states = self.decoder.comm_manager.final_norm(
+                    hidden_states, residual, ctx, self.shared_head.norm
+            else:
```

- Extracted files (not manually reviewed):
  - runtime: `python/tokenspeed/runtime/models/deepseek_v3.py` modified +10/-0; `python/tokenspeed/runtime/models/deepseek_nextn.py` modified +4/-3
- Risk and verification: Runtime changes concentrate in `python/tokenspeed/runtime/distributed/comm_manager.py`, `python/tokenspeed/runtime/execution/context.py`, `python/tokenspeed/runtime/execution/drafter/eagle.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #602 - fix(deepseek-v3): Route DeepSeek merge state through unified backend

- Link: https://github.com/lightseekorg/tokenspeed/pull/602
- Status/date: merged / 2026-07-07
- Trace source: `git log --name-only -- <model-files>` found it through `python/tokenspeed/runtime/models/deepseek_v3.py`, `test/runtime/models/test_deepseek_v3_loader.py`; associated commits `9741cb27de3b`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 7 files, +75/-10, 218 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `test/runtime/models/test_deepseek_v3_loader.py` modified +7/-0 (7 lines); hunks: -1,10 +1,17; symbols: TestDeepseekV3Loader, test_cached_prefix_merge_uses_attention_dispatcher, test_missing_checkpoint_scale_params_are_silent, touching `TestDeepseekV3Loader, test_cached_prefix_merge_uses_attention_dispatcher, test_missing_checkpoint_scale_params_are_silent`; `python/tokenspeed/runtime/models/deepseek_v3.py` modified +3/-3 (6 lines); hunks: -31,6 +31,7; -41,7 +42,6; symbols: forward_normal_chunked_kv_core, touching `forward_normal_chunked_kv_core`.
- Code diff details:
  - `test/runtime/models/test_deepseek_v3_loader.py` modified +7/-0 (7 lines); hunks: -1,10 +1,17; symbols: TestDeepseekV3Loader, test_cached_prefix_merge_uses_attention_dispatcher, test_missing_checkpoint_scale_params_are_silent
  - `python/tokenspeed/runtime/models/deepseek_v3.py` modified +3/-3 (6 lines); hunks: -31,6 +31,7; -41,7 +42,6; symbols: forward_normal_chunked_kv_core
- Key code excerpts:

```diff
diff -- test/runtime/models/test_deepseek_v3_loader.py
@@ -1,10 +1,17 @@
+from tokenspeed_kernel.ops.attention import attn_merge_state
+from tokenspeed.runtime.models import deepseek_v3
+    def test_cached_prefix_merge_uses_attention_dispatcher(self):
+        self.assertIs(deepseek_v3.attn_merge_state, attn_merge_state)
+        self.assertFalse(hasattr(deepseek_v3, "merge_state"))
diff -- python/tokenspeed/runtime/models/deepseek_v3.py
@@ -31,6 +31,7 @@
+from tokenspeed_kernel.ops.attention import attn_merge_state
@@ -41,7 +42,6 @@
-from tokenspeed_kernel.thirdparty.cuda.merge_state import merge_state
@@ -1003,7 +1003,7 @@ def forward_normal_chunked_kv_core(
-        # chunk's merge accumulates in place via merge_state(inplace=True).
+        # chunk's merge accumulates in place via attn_merge_state(inplace=True).
```

- Extracted files (not manually reviewed):
  - tests: `test/runtime/models/test_deepseek_v3_loader.py` modified +7/-0
  - runtime: `python/tokenspeed/runtime/models/deepseek_v3.py` modified +3/-3
- Risk and verification: The diff ships test coverage in `test/runtime/models/test_deepseek_v3_loader.py`, `tokenspeed-kernel/test/ops/test_attention.py`, `tokenspeed-kernel/test/test_kernel_api_selection.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #544 - refactor(spec-decode): simplify deepseekV3/GLM attention path for #217 (3/3)

- Link: https://github.com/lightseekorg/tokenspeed/pull/544
- Status/date: merged / 2026-07-10
- Trace source: `git log --name-only -- <model-files>` found it through `python/tokenspeed/runtime/models/deepseek_nextn.py`, `python/tokenspeed/runtime/models/deepseek_v3.py`; associated commits `27e99fb86fa7`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 12 files, +300/-115, 748 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/tokenspeed/runtime/models/deepseek_v3.py` modified +121/-32 (153 lines); hunks: -70,7 +70,10; -646,13 +649,28 @@ def forward(; symbols: forward, _project_q_latent, _attention_core, _attn, touching `forward, _project_q_latent, _attention_core`; `python/tokenspeed/runtime/models/deepseek_nextn.py` modified +80/-9 (89 lines); hunks: -30,7 +30,10; -53,12 +56,79; symbols: DeepseekV3DraftDecoderLayer, attention_cls, _maybe_narrow_residual, forward, touching `DeepseekV3DraftDecoderLayer, attention_cls, _maybe_narrow_residual`.
- Code diff details:
  - `python/tokenspeed/runtime/models/deepseek_v3.py` modified +121/-32 (153 lines); hunks: -70,7 +70,10; -646,13 +649,28 @@ def forward(; symbols: forward, _project_q_latent, _attention_core, _attn
  - `python/tokenspeed/runtime/models/deepseek_nextn.py` modified +80/-9 (89 lines); hunks: -30,7 +30,10; -53,12 +56,79; symbols: DeepseekV3DraftDecoderLayer, attention_cls, _maybe_narrow_residual, forward
- Key code excerpts:

```diff
diff -- python/tokenspeed/runtime/models/deepseek_v3.py
@@ -70,7 +70,10 @@
-from tokenspeed.runtime.execution.context import ForwardContext
+from tokenspeed.runtime.execution.context import (
+    ForwardContext,
+    report_collective_sizing,
+)
@@ -646,13 +649,28 @@ def forward(
diff -- python/tokenspeed/runtime/models/deepseek_nextn.py
@@ -30,7 +30,10 @@
-from tokenspeed.runtime.execution.context import ForwardContext
+from tokenspeed.runtime.execution.context import (
+    ForwardContext,
+    report_collective_sizing,
+)
@@ -53,12 +56,79 @@
```

- Extracted files (not manually reviewed):
  - runtime: `python/tokenspeed/runtime/models/deepseek_v3.py` modified +121/-32; `python/tokenspeed/runtime/models/deepseek_nextn.py` modified +80/-9
- Risk and verification: Runtime changes concentrate in `python/tokenspeed/runtime/distributed/comm_manager.py`, `python/tokenspeed/runtime/execution/context.py`, `python/tokenspeed/runtime/execution/cuda_graph_wrapper.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #840 - refactor: backend seq_lens ownership and fix dsv3

- Link: https://github.com/lightseekorg/tokenspeed/pull/840
- Status/date: merged / 2026-07-29
- Trace source: `git log --name-only -- <model-files>` found it through `python/tokenspeed/runtime/models/deepseek_v3.py`; associated commits `7f24a36e1654`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 24 files, +583/-408, 1643 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `python/tokenspeed/runtime/models/deepseek_v3.py` modified +5/-1 (6 lines); hunks: -997,7 +997,9 @@ def forward_normal_chunked_kv_prepare(; -1230,6 +1232,8 @@ def _apply_correction(self, ctx: ForwardContext) -> None:; symbols: forward_normal_chunked_kv_prepare, _apply_correction, DeepseekV3DecoderLayer, touching `forward_normal_chunked_kv_prepare, _apply_correction, DeepseekV3DecoderLayer`.
- Code diff details:
  - `python/tokenspeed/runtime/models/deepseek_v3.py` modified +5/-1 (6 lines); hunks: -997,7 +997,9 @@ def forward_normal_chunked_kv_prepare(; -1230,6 +1232,8 @@ def _apply_correction(self, ctx: ForwardContext) -> None:; symbols: forward_normal_chunked_kv_prepare, _apply_correction, DeepseekV3DecoderLayer
- Key code excerpts:

```diff
diff -- python/tokenspeed/runtime/models/deepseek_v3.py
@@ -997,7 +997,9 @@ def forward_normal_chunked_kv_prepare(
-        kv = self.kv_b_proj(kv_a)[0]
+        # kv_a is a split view of latent_cache (non-contiguous); the fp8 online-quant
+        # GEMM in kv_b_proj asserts contiguous input.
+        kv = self.kv_b_proj(kv_a.contiguous())[0]
@@ -1230,6 +1232,8 @@ def _apply_correction(self, ctx: ForwardContext) -> None:
+        # Publish: the backend owns its buffer, so in-graph edits need a copy.
```

- Extracted files (not manually reviewed):
  - runtime: `python/tokenspeed/runtime/models/deepseek_v3.py` modified +5/-1
- Risk and verification: The diff ships test coverage in `test/runtime/test_draft_advance_seqlens.py`, `test/runtime/test_flat_group_write_locs.py`, `test/runtime/test_hybrid_cudagraph_kwargs.py`, `test/runtime/test_kimi_k3_flat_cudagraph.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
