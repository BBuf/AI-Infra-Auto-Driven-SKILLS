# TokenSpeed DeepSeek V3/R1 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `python/tokenspeed/runtime/models/deepseek_nextn.py` | [#217](https://github.com/lightseekorg/tokenspeed/pull/217), [#544](https://github.com/lightseekorg/tokenspeed/pull/544) |
| `python/tokenspeed/runtime/models/deepseek_v3.py` | [#217](https://github.com/lightseekorg/tokenspeed/pull/217), [#544](https://github.com/lightseekorg/tokenspeed/pull/544), [#602](https://github.com/lightseekorg/tokenspeed/pull/602), [#840](https://github.com/lightseekorg/tokenspeed/pull/840) |
| `test/runtime/models/test_deepseek_v3_loader.py` | [#602](https://github.com/lightseekorg/tokenspeed/pull/602) |

## PR 覆盖总览

- git 追溯 PR 数: 4
- 原文档显式引用补充 PR 数: 0
- 当前文档总 PR 数: 4
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-05-28 | [#217](https://github.com/lightseekorg/tokenspeed/pull/217) | merged | perf(Spec Decode): skip dead-position compute in draft catch-up step(decode) | `python/tokenspeed/runtime/models/deepseek_v3.py`, `python/tokenspeed/runtime/models/deepseek_nextn.py` |
| 2026-07-07 | [#602](https://github.com/lightseekorg/tokenspeed/pull/602) | merged | fix(deepseek-v3): Route DeepSeek merge state through unified backend | `test/runtime/models/test_deepseek_v3_loader.py`, `python/tokenspeed/runtime/models/deepseek_v3.py` |
| 2026-07-10 | [#544](https://github.com/lightseekorg/tokenspeed/pull/544) | merged | refactor(spec-decode): simplify deepseekV3/GLM attention path for #217 (3/3) | `python/tokenspeed/runtime/models/deepseek_v3.py`, `python/tokenspeed/runtime/models/deepseek_nextn.py` |
| 2026-07-29 | [#840](https://github.com/lightseekorg/tokenspeed/pull/840) | merged | refactor: backend seq_lens ownership and fix dsv3 | `python/tokenspeed/runtime/models/deepseek_v3.py` |

## 逐 PR diff 审计卡

### PR #217 - perf(Spec Decode): skip dead-position compute in draft catch-up step(decode)

- 链接: https://github.com/lightseekorg/tokenspeed/pull/217
- 状态/时间: merged / 2026-05-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/deepseek_nextn.py`, `python/tokenspeed/runtime/models/deepseek_v3.py`；关联提交 `27e99fb86fa7`, `a9bc2188501c`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 9 个文件，+109/-38，可读 patch 286 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/deepseek_v3.py` modified +10/-0 (10 lines); hunks: -675,6 +675,10 @@ def forward(; -1153,6 +1157,9 @@ def forward(; symbols: forward，涉及 `forward`；`python/tokenspeed/runtime/models/deepseek_nextn.py` modified +4/-3 (7 lines); hunks: -141,11 +141,12 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/deepseek_v3.py` modified +10/-0 (10 lines); hunks: -675,6 +675,10 @@ def forward(; -1153,6 +1157,9 @@ def forward(; symbols: forward
  - `python/tokenspeed/runtime/models/deepseek_nextn.py` modified +4/-3 (7 lines); hunks: -141,11 +141,12 @@ def forward(; symbols: forward
- 关键代码摘录:

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

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/deepseek_v3.py` modified +10/-0; `python/tokenspeed/runtime/models/deepseek_nextn.py` modified +4/-3
- 验证与风险: runtime 路径改动集中在 `python/tokenspeed/runtime/distributed/comm_manager.py`, `python/tokenspeed/runtime/execution/context.py`, `python/tokenspeed/runtime/execution/drafter/eagle.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #602 - fix(deepseek-v3): Route DeepSeek merge state through unified backend

- 链接: https://github.com/lightseekorg/tokenspeed/pull/602
- 状态/时间: merged / 2026-07-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/deepseek_v3.py`, `test/runtime/models/test_deepseek_v3_loader.py`；关联提交 `9741cb27de3b`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 7 个文件，+75/-10，可读 patch 218 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/runtime/models/test_deepseek_v3_loader.py` modified +7/-0 (7 lines); hunks: -1,10 +1,17; symbols: TestDeepseekV3Loader, test_cached_prefix_merge_uses_attention_dispatcher, test_missing_checkpoint_scale_params_are_silent，涉及 `TestDeepseekV3Loader, test_cached_prefix_merge_uses_attention_dispatcher, test_missing_checkpoint_scale_params_are_silent`；`python/tokenspeed/runtime/models/deepseek_v3.py` modified +3/-3 (6 lines); hunks: -31,6 +31,7; -41,7 +42,6; symbols: forward_normal_chunked_kv_core，涉及 `forward_normal_chunked_kv_core`。
- 代码 diff 细节:
  - `test/runtime/models/test_deepseek_v3_loader.py` modified +7/-0 (7 lines); hunks: -1,10 +1,17; symbols: TestDeepseekV3Loader, test_cached_prefix_merge_uses_attention_dispatcher, test_missing_checkpoint_scale_params_are_silent
  - `python/tokenspeed/runtime/models/deepseek_v3.py` modified +3/-3 (6 lines); hunks: -31,6 +31,7; -41,7 +42,6; symbols: forward_normal_chunked_kv_core
- 关键代码摘录:

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

- 提取文件（未人工审阅）:
  - tests: `test/runtime/models/test_deepseek_v3_loader.py` modified +7/-0
  - runtime: `python/tokenspeed/runtime/models/deepseek_v3.py` modified +3/-3
- 验证与风险: diff 自带测试面 `test/runtime/models/test_deepseek_v3_loader.py`, `tokenspeed-kernel/test/ops/test_attention.py`, `tokenspeed-kernel/test/test_kernel_api_selection.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #544 - refactor(spec-decode): simplify deepseekV3/GLM attention path for #217 (3/3)

- 链接: https://github.com/lightseekorg/tokenspeed/pull/544
- 状态/时间: merged / 2026-07-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/deepseek_nextn.py`, `python/tokenspeed/runtime/models/deepseek_v3.py`；关联提交 `27e99fb86fa7`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 12 个文件，+300/-115，可读 patch 748 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/deepseek_v3.py` modified +121/-32 (153 lines); hunks: -70,7 +70,10; -646,13 +649,28 @@ def forward(; symbols: forward, _project_q_latent, _attention_core, _attn，涉及 `forward, _project_q_latent, _attention_core`；`python/tokenspeed/runtime/models/deepseek_nextn.py` modified +80/-9 (89 lines); hunks: -30,7 +30,10; -53,12 +56,79; symbols: DeepseekV3DraftDecoderLayer, attention_cls, _maybe_narrow_residual, forward，涉及 `DeepseekV3DraftDecoderLayer, attention_cls, _maybe_narrow_residual`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/deepseek_v3.py` modified +121/-32 (153 lines); hunks: -70,7 +70,10; -646,13 +649,28 @@ def forward(; symbols: forward, _project_q_latent, _attention_core, _attn
  - `python/tokenspeed/runtime/models/deepseek_nextn.py` modified +80/-9 (89 lines); hunks: -30,7 +30,10; -53,12 +56,79; symbols: DeepseekV3DraftDecoderLayer, attention_cls, _maybe_narrow_residual, forward
- 关键代码摘录:

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

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/deepseek_v3.py` modified +121/-32; `python/tokenspeed/runtime/models/deepseek_nextn.py` modified +80/-9
- 验证与风险: runtime 路径改动集中在 `python/tokenspeed/runtime/distributed/comm_manager.py`, `python/tokenspeed/runtime/execution/context.py`, `python/tokenspeed/runtime/execution/cuda_graph_wrapper.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #840 - refactor: backend seq_lens ownership and fix dsv3

- 链接: https://github.com/lightseekorg/tokenspeed/pull/840
- 状态/时间: merged / 2026-07-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/tokenspeed/runtime/models/deepseek_v3.py`；关联提交 `7f24a36e1654`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 24 个文件，+583/-408，可读 patch 1643 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/tokenspeed/runtime/models/deepseek_v3.py` modified +5/-1 (6 lines); hunks: -997,7 +997,9 @@ def forward_normal_chunked_kv_prepare(; -1230,6 +1232,8 @@ def _apply_correction(self, ctx: ForwardContext) -> None:; symbols: forward_normal_chunked_kv_prepare, _apply_correction, DeepseekV3DecoderLayer，涉及 `forward_normal_chunked_kv_prepare, _apply_correction, DeepseekV3DecoderLayer`。
- 代码 diff 细节:
  - `python/tokenspeed/runtime/models/deepseek_v3.py` modified +5/-1 (6 lines); hunks: -997,7 +997,9 @@ def forward_normal_chunked_kv_prepare(; -1230,6 +1232,8 @@ def _apply_correction(self, ctx: ForwardContext) -> None:; symbols: forward_normal_chunked_kv_prepare, _apply_correction, DeepseekV3DecoderLayer
- 关键代码摘录:

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

- 提取文件（未人工审阅）:
  - runtime: `python/tokenspeed/runtime/models/deepseek_v3.py` modified +5/-1
- 验证与风险: diff 自带测试面 `test/runtime/test_draft_advance_seqlens.py`, `test/runtime/test_flat_group_write_locs.py`, `test/runtime/test_hybrid_cudagraph_kwargs.py`, `test/runtime/test_kimi_k3_flat_cudagraph.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
