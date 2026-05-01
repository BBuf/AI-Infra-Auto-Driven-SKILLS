# sglang ERNIE 4.5 模型 PR 优化历史

## 覆盖范围

- 重做日期: 2026-05-01
- 源码基线: `sgl-project/sglang` 当前追溯 worktree commit `4197c55968`
- PR 收集规则: 先从模型实现、配置、processor、parser、docs/tests 等相关文件执行 `git log --name-only -- <model-files>`，再按 commit subject 的模型关键词过滤，最后用 GitHub Pull Request files API 读取每个 PR 的最终 diff。
- 额外保留规则: 原 history/skill 已显式引用但未出现在当前实现文件 git trace 中的 PR 会保留，并在卡片里标注来源。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `python/sglang/srt/models/ernie45_moe_vl.py` | 无直接 PR 号提交 |
| `python/sglang/srt/models/ernie45_vl.py` | 无直接 PR 号提交 |
| `python/sglang/srt/multimodal/processors/ernie45_vl.py` | 无直接 PR 号提交 |

## PR 覆盖总览

- git 追溯 PR 数: 0
- 原文档显式引用补充 PR 数: 3
- 当前文档总 PR 数: 3
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-01-26 | [#15679](https://github.com/sgl-project/sglang/pull/15679) | merged | [Model] Add Ernie4.5 VL model support | `python/sglang/srt/models/ernie45_vl.py`, `python/sglang/srt/models/ernie45_moe_vl.py`, `python/sglang/srt/multimodal/processors/ernie45_vl.py` |
| 2026-02-16 | [#18856](https://github.com/sgl-project/sglang/pull/18856) | merged | [VLM] Optimize Ernie4.5-VL rotary embedding with fused triton kernel | `python/sglang/srt/layers/rotary_embedding.py` |
| 2026-03-04 | [#19743](https://github.com/sgl-project/sglang/pull/19743) | merged | [VLM] Support cos sin cache for Ernie4.5-VL | `python/sglang/srt/models/ernie45_vl.py` |

## 逐 PR diff 审计卡

### PR #15679 - [Model] Add Ernie4.5 VL model support

- 链接: https://github.com/sgl-project/sglang/pull/15679
- 状态/时间: merged / 2026-01-26
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+2072/-0，可读 patch 2103 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Add Ernie4.5 VL model support」；模型线: ERNIE 4.5；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/ernie45_vl.py`, `python/sglang/srt/models/ernie45_moe_vl.py`, `python/sglang/srt/multimodal/processors/ernie45_vl.py`；技术摘要: 覆盖「[Model] Add Ernie4.5 VL model support」；主要实现面是 `python/sglang/srt/models/ernie45_vl.py`, `python/sglang/srt/models/ernie45_moe_vl.py`, `python/sglang/srt/multimodal/processors/ernie45_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/ernie45_vl.py` added +845/-0 (845 lines); hunks: -0,0 +1,845; symbols: Ernie4_5_VisionMLP, __init__, forward, Ernie4_5_VisionBlock，涉及 `Ernie4_5_VisionMLP, __init__, forward`；`python/sglang/srt/models/ernie45_moe_vl.py` added +552/-0 (552 lines); hunks: -0,0 +1,552; symbols: Ernie4_5_VLMoeAttention, __init__, forward, Ernie4_5_VLMoeMoE，涉及 `Ernie4_5_VLMoeAttention, __init__, forward`；`python/sglang/srt/multimodal/processors/ernie45_vl.py` added +417/-0 (417 lines); hunks: -0,0 +1,417; symbols: smart_resize, resize_image, round_by_factor, ceil_by_factor，涉及 `smart_resize, resize_image, round_by_factor`；`python/sglang/srt/layers/rotary_embedding.py` modified +256/-0 (256 lines); hunks: -2284,6 +2284,177 @@ def get_rope_index_glm4v(; -2323,6 +2494,91 @@ def _get_llm_pos_ids_for_vision(; symbols: get_rope_index_glm4v, get_rope_index_ernie45, _get_feat_extract_output_lengths, _get_llm_pos_ids_for_vision，涉及 `get_rope_index_glm4v, get_rope_index_ernie45, _get_feat_extract_output_lengths`。
- 代码 diff 细节:
  - `python/sglang/srt/models/ernie45_vl.py` added +845/-0 (845 lines); hunks: -0,0 +1,845; symbols: Ernie4_5_VisionMLP, __init__, forward, Ernie4_5_VisionBlock
  - `python/sglang/srt/models/ernie45_moe_vl.py` added +552/-0 (552 lines); hunks: -0,0 +1,552; symbols: Ernie4_5_VLMoeAttention, __init__, forward, Ernie4_5_VLMoeMoE
  - `python/sglang/srt/multimodal/processors/ernie45_vl.py` added +417/-0 (417 lines); hunks: -0,0 +1,417; symbols: smart_resize, resize_image, round_by_factor, ceil_by_factor
  - `python/sglang/srt/layers/rotary_embedding.py` modified +256/-0 (256 lines); hunks: -2284,6 +2284,177 @@ def get_rope_index_glm4v(; -2323,6 +2494,91 @@ def _get_llm_pos_ids_for_vision(; symbols: get_rope_index_glm4v, get_rope_index_ernie45, _get_feat_extract_output_lengths, _get_llm_pos_ids_for_vision
  - `docs/supported_models/multimodal_language_models.md` modified +1/-0 (1 lines); hunks: -46,6 +46,7 @@ in the GitHub search bar.
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/ernie45_vl.py
@@ -0,0 +1,845 @@
+# Copyright 2023-2025 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/models/ernie45_moe_vl.py
@@ -0,0 +1,552 @@
+# Copyright 2023-2025 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/multimodal/processors/ernie45_vl.py
@@ -0,0 +1,417 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/ernie45_vl.py` added +845/-0; `python/sglang/srt/models/ernie45_moe_vl.py` added +552/-0; `python/sglang/srt/multimodal/processors/ernie45_vl.py` added +417/-0; `python/sglang/srt/layers/rotary_embedding.py` modified +256/-0; `python/sglang/srt/configs/model_config.py` modified +1/-0
  - docs: `docs/supported_models/multimodal_language_models.md` modified +1/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/layers/rotary_embedding.py`, `python/sglang/srt/models/ernie45_moe_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #18856 - [VLM] Optimize Ernie4.5-VL rotary embedding with fused triton kernel

- 链接: https://github.com/sgl-project/sglang/pull/18856
- 状态/时间: merged / 2026-02-16
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+268/-3，可读 patch 308 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[VLM] Optimize Ernie4.5-VL rotary embedding with fused triton kernel」；模型线: ERNIE 4.5；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/rotary_embedding.py`；技术摘要: 覆盖「[VLM] Optimize Ernie4.5-VL rotary embedding with fused triton kernel」；主要实现面是 `python/sglang/srt/layers/rotary_embedding.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/rotary_embedding.py` modified +268/-3 (271 lines); hunks: -2787,9 +2787,238 @@ def _compute_cos_sin_cache(self) -> torch.Tensor:; -2834,14 +3063,16 @@ def forward_native( # type: ignore[override]; symbols: _compute_cos_sin_cache, _triton_ernie45_rope_qk_fused, triton_ernie45_rope_fused_inplace, Ernie4_5_VLRotaryEmbedding，涉及 `_compute_cos_sin_cache, _triton_ernie45_rope_qk_fused, triton_ernie45_rope_fused_inplace`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/rotary_embedding.py` modified +268/-3 (271 lines); hunks: -2787,9 +2787,238 @@ def _compute_cos_sin_cache(self) -> torch.Tensor:; -2834,14 +3063,16 @@ def forward_native( # type: ignore[override]; symbols: _compute_cos_sin_cache, _triton_ernie45_rope_qk_fused, triton_ernie45_rope_fused_inplace, Ernie4_5_VLRotaryEmbedding
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/rotary_embedding.py
@@ -2787,9 +2787,238 @@ def _compute_cos_sin_cache(self) -> torch.Tensor:
+@triton.jit
+def _triton_ernie45_rope_qk_fused(
+    q_ptr,
+    k_ptr,
+    cos_sin_cache_ptr,
+    positions_ptr,  # [3, num_tokens]  (t/h/w)
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/rotary_embedding.py` modified +268/-3
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/rotary_embedding.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #19743 - [VLM] Support cos sin cache for Ernie4.5-VL

- 链接: https://github.com/sgl-project/sglang/pull/19743
- 状态/时间: merged / 2026-03-04
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+34/-12，可读 patch 102 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[VLM] Support cos sin cache for Ernie4.5-VL」；模型线: ERNIE 4.5；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/models/ernie45_vl.py`；技术摘要: 覆盖「[VLM] Support cos sin cache for Ernie4.5-VL」；主要实现面是 `python/sglang/srt/models/ernie45_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/ernie45_vl.py` modified +34/-12 (46 lines); hunks: -30,6 +30,7; -120,14 +121,16 @@ def forward(; symbols: forward, __init__, dtype, device，涉及 `forward, __init__, dtype`。
- 代码 diff 细节:
  - `python/sglang/srt/models/ernie45_vl.py` modified +34/-12 (46 lines); hunks: -30,6 +30,7; -120,14 +121,16 @@ def forward(; symbols: forward, __init__, dtype, device
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/ernie45_vl.py
@@ -30,6 +30,7 @@
+from sglang.srt.layers.rotary_embedding import get_rope
@@ -120,14 +121,16 @@ def forward(
-        position_embeddings: torch.Tensor,
+        rotary_pos_emb_cos: torch.Tensor,
+        rotary_pos_emb_sin: torch.Tensor,
-            position_embeddings=position_embeddings,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/ernie45_vl.py` modified +34/-12
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/ernie45_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
