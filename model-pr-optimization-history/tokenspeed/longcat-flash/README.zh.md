# TokenSpeed LongCat-Flash 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `python/tokenspeed/runtime/models/longcat_flash.py` | [#1582](https://github.com/lightseekorg/tokenspeed/pull/1582), [#1952](https://github.com/lightseekorg/tokenspeed/pull/1952), [#1973](https://github.com/lightseekorg/tokenspeed/pull/1973) |
| `test/runtime/models/test_longcat_flash.py` | [#1582](https://github.com/lightseekorg/tokenspeed/pull/1582), [#1952](https://github.com/lightseekorg/tokenspeed/pull/1952), [#1973](https://github.com/lightseekorg/tokenspeed/pull/1973) |
| `tokenspeed-kernel/test/ops/test_longcat_lsa_topk.py` | 无直接 PR 号提交 |

## PR 覆盖总览

- git 追溯 PR 数: 3
- 原文档显式引用补充 PR 数: 0
- 当前文档总 PR 数: 3
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-09-19 | [#1582](https://github.com/lightseekorg/tokenspeed/pull/1582) | merged | fix(longcat): select precomputed SwiGLU expert routing | `test/runtime/models/test_longcat_flash.py`, `python/tokenspeed/runtime/models/longcat_flash.py` |
| 2026-10-04 | [#1952](https://github.com/lightseekorg/tokenspeed/pull/1952) | merged | fix(longcat): keep the attention branches in the dense row layout under attention DP | `test/runtime/models/test_longcat_flash.py`, `python/tokenspeed/runtime/models/longcat_flash.py` |
| 2026-10-04 | [#1973](https://github.com/lightseekorg/tokenspeed/pull/1973) | merged | fix(longcat): count the identity zero-expert residual once under MoE TP/EP | `test/runtime/models/test_longcat_flash.py`, `python/tokenspeed/runtime/models/longcat_flash.py` |

## 逐 PR diff 审计卡

### PR #1582 - fix(longcat): select precomputed SwiGLU expert routing

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1582
- 状态/时间: merged / 2026-09-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/longcat_flash.py`, `test/runtime/models/test_longcat_flash.py`；关联提交 `d1b101b39dd4`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 3 个文件，+86/-1，可读 patch 115 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/runtime/models/test_longcat_flash.py` modified +77/-1 (78 lines); hunks: -6,7 +6,7; -139,6 +139,82 @@ def test_identity_zero_expert_masks_and_adds_hidden_state(s...; symbols: test_identity_zero_expert_masks_and_adds_hidden_state, TestLongcatMoePlan, test_blackwell_ep4_plans_accept_zero_expert_routing, TestLongcatCheckpointLoading，涉及 `test_identity_zero_expert_masks_and_adds_hidden_state, TestLongcatMoePlan, test_blackwell_ep4_plans_accept_zero_expert_routing`；`python/tokenspeed/runtime/models/longcat_flash.py` modified +3/-0 (3 lines); hunks: -245,6 +245,9 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `test/runtime/models/test_longcat_flash.py` modified +77/-1 (78 lines); hunks: -6,7 +6,7; -139,6 +139,82 @@ def test_identity_zero_expert_masks_and_adds_hidden_state(s...; symbols: test_identity_zero_expert_masks_and_adds_hidden_state, TestLongcatMoePlan, test_blackwell_ep4_plans_accept_zero_expert_routing, TestLongcatCheckpointLoading
  - `python/tokenspeed/runtime/models/longcat_flash.py` modified +3/-0 (3 lines); hunks: -245,6 +245,9 @@ def __init__(; symbols: __init__
- 关键代码摘录:

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

- 提取文件（未人工审阅）:
  - tests: `test/runtime/models/test_longcat_flash.py` modified +77/-1
  - runtime: `python/tokenspeed/runtime/models/longcat_flash.py` modified +3/-0
- 验证与风险: diff 自带测试面 `test/runtime/models/test_longcat_flash.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1952 - fix(longcat): keep the attention branches in the dense row layout under attention DP

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1952
- 状态/时间: merged / 2026-10-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/longcat_flash.py`, `test/runtime/models/test_longcat_flash.py`；关联提交 `179b482fc279`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 8 个文件，+472/-13，可读 patch 610 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/runtime/models/test_longcat_flash.py` modified +287/-0 (287 lines); hunks: -215,6 +215,293 @@ def test_blackwell_ep4_plans_accept_zero_expert_routing(se...; symbols: test_blackwell_ep4_plans_accept_zero_expert_routing, _fake_comm_ops, all_reduce, token_all_gather，涉及 `test_blackwell_ep4_plans_accept_zero_expert_routing, _fake_comm_ops, all_reduce`；`python/tokenspeed/runtime/models/longcat_flash.py` modified +92/-7 (99 lines); hunks: -362,6 +362,18 @@ def forward(; -448,13 +460,21 @@ def __init__(; symbols: forward, _RuntimeLongcatDecoderLayer, __init__, _init_comm，涉及 `forward, _RuntimeLongcatDecoderLayer, __init__`。
- 代码 diff 细节:
  - `test/runtime/models/test_longcat_flash.py` modified +287/-0 (287 lines); hunks: -215,6 +215,293 @@ def test_blackwell_ep4_plans_accept_zero_expert_routing(se...; symbols: test_blackwell_ep4_plans_accept_zero_expert_routing, _fake_comm_ops, all_reduce, token_all_gather
  - `python/tokenspeed/runtime/models/longcat_flash.py` modified +92/-7 (99 lines); hunks: -362,6 +362,18 @@ def forward(; -448,13 +460,21 @@ def __init__(; symbols: forward, _RuntimeLongcatDecoderLayer, __init__, _init_comm
- 关键代码摘录:

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

- 提取文件（未人工审阅）:
  - tests: `test/runtime/models/test_longcat_flash.py` modified +287/-0
  - runtime: `python/tokenspeed/runtime/models/longcat_flash.py` modified +92/-7
- 验证与风险: diff 自带测试面 `test/runtime/models/test_longcat_flash.py`, `test/runtime/test_batch_log.py`, `test/runtime/test_hyperconnection_kernel_boundary.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #1973 - fix(longcat): count the identity zero-expert residual once under MoE TP/EP

- 链接: https://github.com/lightseekorg/tokenspeed/pull/1973
- 状态/时间: merged / 2026-10-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/longcat_flash.py`, `test/runtime/models/test_longcat_flash.py`；关联提交 `33c646730ea8`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+110/-18，可读 patch 206 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/runtime/models/test_longcat_flash.py` modified +76/-13 (89 lines); hunks: -103,21 +103,31 @@ def test_moe_layer_rejects_partially_ignored_experts(self):; -138,8 +148,64 @@ def test_identity_zero_expert_masks_and_adds_hidden_state(s...; symbols: test_moe_layer_rejects_partially_ignored_experts, _zero_expert_moe, _zero_expert_topk, TestLongcatZeroExpert，涉及 `test_moe_layer_rejects_partially_ignored_experts, _zero_expert_moe, _zero_expert_topk`；`python/tokenspeed/runtime/models/longcat_flash.py` modified +34/-5 (39 lines); hunks: -52,6 +52,9; -219,13 +222,30 @@ def __init__(; symbols: __init__, get_moe_routed_weights, _apply_zero_experts，涉及 `__init__, get_moe_routed_weights, _apply_zero_experts`。
- 代码 diff 细节:
  - `test/runtime/models/test_longcat_flash.py` modified +76/-13 (89 lines); hunks: -103,21 +103,31 @@ def test_moe_layer_rejects_partially_ignored_experts(self):; -138,8 +148,64 @@ def test_identity_zero_expert_masks_and_adds_hidden_state(s...; symbols: test_moe_layer_rejects_partially_ignored_experts, _zero_expert_moe, _zero_expert_topk, TestLongcatZeroExpert
  - `python/tokenspeed/runtime/models/longcat_flash.py` modified +34/-5 (39 lines); hunks: -52,6 +52,9; -219,13 +222,30 @@ def __init__(; symbols: __init__, get_moe_routed_weights, _apply_zero_experts
- 关键代码摘录:

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

- 提取文件（未人工审阅）:
  - tests: `test/runtime/models/test_longcat_flash.py` modified +76/-13
  - runtime: `python/tokenspeed/runtime/models/longcat_flash.py` modified +34/-5
- 验证与风险: diff 自带测试面 `test/runtime/models/test_longcat_flash.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
