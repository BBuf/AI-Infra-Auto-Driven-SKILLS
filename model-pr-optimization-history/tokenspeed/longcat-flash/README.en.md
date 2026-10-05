# TokenSpeed LongCat-Flash Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `python/tokenspeed/runtime/models/longcat_flash.py` | [#1582](https://github.com/lightseekorg/tokenspeed/pull/1582), [#1952](https://github.com/lightseekorg/tokenspeed/pull/1952), [#1973](https://github.com/lightseekorg/tokenspeed/pull/1973) |
| `test/runtime/models/test_longcat_flash.py` | [#1582](https://github.com/lightseekorg/tokenspeed/pull/1582), [#1952](https://github.com/lightseekorg/tokenspeed/pull/1952), [#1973](https://github.com/lightseekorg/tokenspeed/pull/1973) |
| `tokenspeed-kernel/test/ops/test_longcat_lsa_topk.py` | no direct PR-number commit |

## PR Coverage Summary

- Git-traced PRs: 3
- Extra PRs preserved from existing docs: 0
- Total PRs in this document: 3
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-09-19 | [#1582](https://github.com/lightseekorg/tokenspeed/pull/1582) | merged | fix(longcat): select precomputed SwiGLU expert routing | `test/runtime/models/test_longcat_flash.py`, `python/tokenspeed/runtime/models/longcat_flash.py` |
| 2026-10-04 | [#1952](https://github.com/lightseekorg/tokenspeed/pull/1952) | merged | fix(longcat): keep the attention branches in the dense row layout under attention DP | `test/runtime/models/test_longcat_flash.py`, `python/tokenspeed/runtime/models/longcat_flash.py` |
| 2026-10-04 | [#1973](https://github.com/lightseekorg/tokenspeed/pull/1973) | merged | fix(longcat): count the identity zero-expert residual once under MoE TP/EP | `test/runtime/models/test_longcat_flash.py`, `python/tokenspeed/runtime/models/longcat_flash.py` |

## Per-PR Diff Audit Cards

### PR #1582 - fix(longcat): select precomputed SwiGLU expert routing

- Link: https://github.com/lightseekorg/tokenspeed/pull/1582
- Status/date: merged / 2026-09-19
- Trace source: `git log --name-only -- <model-files>` found it through `python/tokenspeed/runtime/models/longcat_flash.py`, `test/runtime/models/test_longcat_flash.py`; associated commits `d1b101b39dd4`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +86/-1, 115 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `test/runtime/models/test_longcat_flash.py` modified +77/-1 (78 lines); hunks: -6,7 +6,7; -139,6 +139,82 @@ def test_identity_zero_expert_masks_and_adds_hidden_state(s...; symbols: test_identity_zero_expert_masks_and_adds_hidden_state, TestLongcatMoePlan, test_blackwell_ep4_plans_accept_zero_expert_routing, TestLongcatCheckpointLoading, touching `test_identity_zero_expert_masks_and_adds_hidden_state, TestLongcatMoePlan, test_blackwell_ep4_plans_accept_zero_expert_routing`; `python/tokenspeed/runtime/models/longcat_flash.py` modified +3/-0 (3 lines); hunks: -245,6 +245,9 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `test/runtime/models/test_longcat_flash.py` modified +77/-1 (78 lines); hunks: -6,7 +6,7; -139,6 +139,82 @@ def test_identity_zero_expert_masks_and_adds_hidden_state(s...; symbols: test_identity_zero_expert_masks_and_adds_hidden_state, TestLongcatMoePlan, test_blackwell_ep4_plans_accept_zero_expert_routing, TestLongcatCheckpointLoading
  - `python/tokenspeed/runtime/models/longcat_flash.py` modified +3/-0 (3 lines); hunks: -245,6 +245,9 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- test/runtime/models/test_longcat_flash.py
@@ -6,7 +6,7 @@
-from tokenspeed.runtime.layers.moe.topk import StandardTopKOutput
+from tokenspeed.runtime.layers.moe.topk import StandardTopKOutput, TopKOutputFormat
@@ -139,6 +139,82 @@ def test_identity_zero_expert_masks_and_adds_hidden_state(self):
+class TestLongcatMoePlan(unittest.TestCase):
+    def test_blackwell_ep4_plans_accept_zero_expert_routing(self):
+        from tokenspeed_kernel.platform import current_platform
diff -- python/tokenspeed/runtime/models/longcat_flash.py
@@ -245,6 +245,9 @@ def __init__(
+            # LongCat applies its own zero-expert routing to gated SiLU experts.
+            activation="swiglu",
+            routing_mode="precomputed_topk",
```

- Extracted files (not manually reviewed):
  - tests: `test/runtime/models/test_longcat_flash.py` modified +77/-1
  - runtime: `python/tokenspeed/runtime/models/longcat_flash.py` modified +3/-0
- Risk and verification: The diff ships test coverage in `test/runtime/models/test_longcat_flash.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #1952 - fix(longcat): keep the attention branches in the dense row layout under attention DP

- Link: https://github.com/lightseekorg/tokenspeed/pull/1952
- Status/date: merged / 2026-10-04
- Trace source: `git log --name-only -- <model-files>` found it through `python/tokenspeed/runtime/models/longcat_flash.py`, `test/runtime/models/test_longcat_flash.py`; associated commits `179b482fc279`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 8 files, +472/-13, 610 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `test/runtime/models/test_longcat_flash.py` modified +287/-0 (287 lines); hunks: -215,6 +215,293 @@ def test_blackwell_ep4_plans_accept_zero_expert_routing(se...; symbols: test_blackwell_ep4_plans_accept_zero_expert_routing, _fake_comm_ops, all_reduce, token_all_gather, touching `test_blackwell_ep4_plans_accept_zero_expert_routing, _fake_comm_ops, all_reduce`; `python/tokenspeed/runtime/models/longcat_flash.py` modified +92/-7 (99 lines); hunks: -362,6 +362,18 @@ def forward(; -448,13 +460,21 @@ def __init__(; symbols: forward, _RuntimeLongcatDecoderLayer, __init__, _init_comm, touching `forward, _RuntimeLongcatDecoderLayer, __init__`.
- Code diff details:
  - `test/runtime/models/test_longcat_flash.py` modified +287/-0 (287 lines); hunks: -215,6 +215,293 @@ def test_blackwell_ep4_plans_accept_zero_expert_routing(se...; symbols: test_blackwell_ep4_plans_accept_zero_expert_routing, _fake_comm_ops, all_reduce, token_all_gather
  - `python/tokenspeed/runtime/models/longcat_flash.py` modified +92/-7 (99 lines); hunks: -362,6 +362,18 @@ def forward(; -448,13 +460,21 @@ def __init__(; symbols: forward, _RuntimeLongcatDecoderLayer, __init__, _init_comm
- Key code excerpts:

```diff
diff -- test/runtime/models/test_longcat_flash.py
@@ -215,6 +215,293 @@ def test_blackwell_ep4_plans_accept_zero_expert_routing(self):
+def _fake_comm_ops(rank: int, gathers: list | None = None) -> dict:
+    """Single-process stand-ins for the comm ops, keyed by this rank's slot.
+    ``gathers`` records the group of every all-gather when given.
+    """
+    def all_reduce(tensor, group, **_):
+        return tensor
diff -- python/tokenspeed/runtime/models/longcat_flash.py
@@ -362,6 +362,18 @@ def forward(
+    """One LongCat layer: two attention/dense-MLP branches beside one MoE.
+    Row layout: the residual stream runs through the dense branches
+    (attention 0 -> MLP 0 -> attention 1 -> MLP 1), so the layer's rows follow
+    the dense comm pattern -- all-reduce (every attention-TP rank holds every
+    row of its attention DP group) or RSAG (each rank holds its scattered
+    share). The MoE is a side branch fed from attention 0's output; its own
```

- Extracted files (not manually reviewed):
  - tests: `test/runtime/models/test_longcat_flash.py` modified +287/-0
  - runtime: `python/tokenspeed/runtime/models/longcat_flash.py` modified +92/-7
- Risk and verification: The diff ships test coverage in `test/runtime/models/test_longcat_flash.py`, `test/runtime/test_batch_log.py`, `test/runtime/test_hyperconnection_kernel_boundary.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #1973 - fix(longcat): count the identity zero-expert residual once under MoE TP/EP

- Link: https://github.com/lightseekorg/tokenspeed/pull/1973
- Status/date: merged / 2026-10-04
- Trace source: `git log --name-only -- <model-files>` found it through `python/tokenspeed/runtime/models/longcat_flash.py`, `test/runtime/models/test_longcat_flash.py`; associated commits `33c646730ea8`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +110/-18, 206 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `test/runtime/models/test_longcat_flash.py` modified +76/-13 (89 lines); hunks: -103,21 +103,31 @@ def test_moe_layer_rejects_partially_ignored_experts(self):; -138,8 +148,64 @@ def test_identity_zero_expert_masks_and_adds_hidden_state(s...; symbols: test_moe_layer_rejects_partially_ignored_experts, _zero_expert_moe, _zero_expert_topk, TestLongcatZeroExpert, touching `test_moe_layer_rejects_partially_ignored_experts, _zero_expert_moe, _zero_expert_topk`; `python/tokenspeed/runtime/models/longcat_flash.py` modified +34/-5 (39 lines); hunks: -52,6 +52,9; -219,13 +222,30 @@ def __init__(; symbols: __init__, get_moe_routed_weights, _apply_zero_experts, touching `__init__, get_moe_routed_weights, _apply_zero_experts`.
- Code diff details:
  - `test/runtime/models/test_longcat_flash.py` modified +76/-13 (89 lines); hunks: -103,21 +103,31 @@ def test_moe_layer_rejects_partially_ignored_experts(self):; -138,8 +148,64 @@ def test_identity_zero_expert_masks_and_adds_hidden_state(s...; symbols: test_moe_layer_rejects_partially_ignored_experts, _zero_expert_moe, _zero_expert_topk, TestLongcatZeroExpert
  - `python/tokenspeed/runtime/models/longcat_flash.py` modified +34/-5 (39 lines); hunks: -52,6 +52,9; -219,13 +222,30 @@ def __init__(; symbols: __init__, get_moe_routed_weights, _apply_zero_experts
- Key code excerpts:

```diff
diff -- test/runtime/models/test_longcat_flash.py
@@ -103,21 +103,31 @@ def test_moe_layer_rejects_partially_ignored_experts(self):
+def _zero_expert_moe(*, adds_residual: bool) -> _RuntimeLongcatMoE:
+    moe = object.__new__(_RuntimeLongcatMoE)
+    moe.zero_expert_num = 1
+    moe.n_routed_experts = 3
+    moe.zero_expert_type = "identity"
+    moe.adds_zero_expert_residual = adds_residual
diff -- python/tokenspeed/runtime/models/longcat_flash.py
@@ -52,6 +52,9 @@
+from tokenspeed.runtime.layers.moe.utils import (
+    get_all2all_backend as _get_all2all_backend,
+)
@@ -219,13 +222,30 @@ def __init__(
+        # The routed output leaves this module as one partial per MoE TP-EP
+        # rank and post_moe_comm sums the group (all-reduce or reduce-scatter),
```

- Extracted files (not manually reviewed):
  - tests: `test/runtime/models/test_longcat_flash.py` modified +76/-13
  - runtime: `python/tokenspeed/runtime/models/longcat_flash.py` modified +34/-5
- Risk and verification: The diff ships test coverage in `test/runtime/models/test_longcat_flash.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
