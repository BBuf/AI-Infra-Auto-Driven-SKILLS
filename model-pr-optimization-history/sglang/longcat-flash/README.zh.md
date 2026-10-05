# SGLang LongCat-Flash 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `docs/cookbook/autoregressive/Meituan/LongCat-2.0.mdx` | 无直接 PR 号提交 |
| `docs/src/snippets/configs/meituan-longcat/longcat-2.0-benchmarks.jsx` | 无直接 PR 号提交 |
| `docs/src/snippets/configs/meituan-longcat/longcat-2.0.jsx` | 无直接 PR 号提交 |
| `python/sglang/srt/configs/longcat_flash.py` | [#9824](https://github.com/sgl-project/sglang/pull/9824), [#17838](https://github.com/sgl-project/sglang/pull/17838), [#30275](https://github.com/sgl-project/sglang/pull/30275) |
| `python/sglang/srt/models/longcat_flash.py` | [#9824](https://github.com/sgl-project/sglang/pull/9824), [#9916](https://github.com/sgl-project/sglang/pull/9916), [#14007](https://github.com/sgl-project/sglang/pull/14007), [#14161](https://github.com/sgl-project/sglang/pull/14161), [#17838](https://github.com/sgl-project/sglang/pull/17838), [#30247](https://github.com/sgl-project/sglang/pull/30247), [#30275](https://github.com/sgl-project/sglang/pull/30275), [#31311](https://github.com/sgl-project/sglang/pull/31311), [#40799](https://github.com/sgl-project/sglang/pull/40799), [#41436](https://github.com/sgl-project/sglang/pull/41436) |
| `python/sglang/srt/models/longcat_flash_nextn.py` | [#9824](https://github.com/sgl-project/sglang/pull/9824), [#9916](https://github.com/sgl-project/sglang/pull/9916), [#32125](https://github.com/sgl-project/sglang/pull/32125) |
| `test/registered/e2e/models_large/test_longcat_flash_lite_fp8.py` | 无直接 PR 号提交 |
| `test/registered/unit/layer_boundary/test_longcat_flash_shortcut.py` | 无直接 PR 号提交 |
| `test/registered/unit/models/test_longcat_flash_router_hpc_gemm.py` | [#30247](https://github.com/sgl-project/sglang/pull/30247) |

## PR 覆盖总览

- git 追溯 PR 数: 11
- 原文档显式引用补充 PR 数: 0
- 当前文档总 PR 数: 11
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
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

## 逐 PR diff 审计卡

### PR #9824 - [Model] Support Meituan LongCat-Flash && LongCat-Flash-MTP

- 链接: https://github.com/sgl-project/sglang/pull/9824
- 状态/时间: merged / 2025-08-31
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/configs/longcat_flash.py`, `python/sglang/srt/models/longcat_flash.py`, `python/sglang/srt/models/longcat_flash_nextn.py`；关联提交 `5e194b21437f`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 10 个文件，+1940/-11，可读 patch 2043 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/sglang/srt/models/longcat_flash.py` added +1015/-0 (1015 lines); hunks: -0,0 +1,1015; symbols: LongcatFlashMLP, __init__, forward, LongcatFlashRouter，涉及 `LongcatFlashMLP, __init__, forward`；`python/sglang/srt/models/longcat_flash_nextn.py` added +691/-0 (691 lines); hunks: -0,0 +1,691; symbols: LongcatFlashDenseDecoderLayer, __init__, forward, LongcatFlashModelNextN，涉及 `LongcatFlashDenseDecoderLayer, __init__, forward`；`python/sglang/srt/configs/longcat_flash.py` added +104/-0 (104 lines); hunks: -0,0 +1,104; symbols: LongcatFlashConfig, __init__，涉及 `LongcatFlashConfig, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/longcat_flash.py` added +1015/-0 (1015 lines); hunks: -0,0 +1,1015; symbols: LongcatFlashMLP, __init__, forward, LongcatFlashRouter
  - `python/sglang/srt/models/longcat_flash_nextn.py` added +691/-0 (691 lines); hunks: -0,0 +1,691; symbols: LongcatFlashDenseDecoderLayer, __init__, forward, LongcatFlashModelNextN
  - `python/sglang/srt/configs/longcat_flash.py` added +104/-0 (104 lines); hunks: -0,0 +1,104; symbols: LongcatFlashConfig, __init__
- 关键代码摘录:

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

- 提取文件（未人工审阅）:
  - runtime: `python/sglang/srt/models/longcat_flash.py` added +1015/-0; `python/sglang/srt/models/longcat_flash_nextn.py` added +691/-0; `python/sglang/srt/configs/longcat_flash.py` added +104/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/__init__.py`, `python/sglang/srt/configs/longcat_flash.py`, `python/sglang/srt/configs/model_config.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #9916 - [Fix] fix the issue encountered when inference LongCat-Flash/MTP EP MoE on b200

- 链接: https://github.com/sgl-project/sglang/pull/9916
- 状态/时间: merged / 2025-09-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/longcat_flash.py`, `python/sglang/srt/models/longcat_flash_nextn.py`；关联提交 `b7361cc4441d`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+49/-30，可读 patch 126 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/sglang/srt/models/longcat_flash.py` modified +26/-15 (41 lines); hunks: -651,9 +651,6 @@ def post_load_weights(self, weight_names=None):; -790,6 +787,9 @@ def post_load_weights(self, weight_names=None):; symbols: post_load_weights, _weight_requant_ue8m0，涉及 `post_load_weights, _weight_requant_ue8m0`；`python/sglang/srt/models/longcat_flash_nextn.py` modified +23/-15 (38 lines); hunks: -344,9 +344,6 @@ def post_load_weights(self):; -480,24 +477,35 @@ def post_load_weights(self):; symbols: post_load_weights, _weight_requant_ue8m0, load_weights，涉及 `post_load_weights, _weight_requant_ue8m0, load_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/longcat_flash.py` modified +26/-15 (41 lines); hunks: -651,9 +651,6 @@ def post_load_weights(self, weight_names=None):; -790,6 +787,9 @@ def post_load_weights(self, weight_names=None):; symbols: post_load_weights, _weight_requant_ue8m0
  - `python/sglang/srt/models/longcat_flash_nextn.py` modified +23/-15 (38 lines); hunks: -344,9 +344,6 @@ def post_load_weights(self):; -480,24 +477,35 @@ def post_load_weights(self):; symbols: post_load_weights, _weight_requant_ue8m0, load_weights
- 关键代码摘录:

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

- 提取文件（未人工审阅）:
  - runtime: `python/sglang/srt/models/longcat_flash.py` modified +26/-15; `python/sglang/srt/models/longcat_flash_nextn.py` modified +23/-15
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/longcat_flash.py`, `python/sglang/srt/models/longcat_flash_nextn.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #14007 - fix: cuda graph issue while running longcat_flash

- 链接: https://github.com/sgl-project/sglang/pull/14007
- 状态/时间: merged / 2025-11-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/longcat_flash.py`；关联提交 `685b9d82bd01`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+2/-0，可读 patch 16 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/sglang/srt/models/longcat_flash.py` modified +2/-0 (2 lines); hunks: -388,6 +388,7 @@ def __init__(; -402,6 +403,7 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/longcat_flash.py` modified +2/-0 (2 lines); hunks: -388,6 +388,7 @@ def __init__(; -402,6 +403,7 @@ def __init__(; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/longcat_flash.py
@@ -388,6 +388,7 @@ def __init__(
+                qkv_latent_func=self.self_attn[i].prepare_qkv_latent,
@@ -402,6 +403,7 @@ def __init__(
+            qkv_latent_func=self.self_attn[0].prepare_qkv_latent,
```

- 提取文件（未人工审阅）:
  - runtime: `python/sglang/srt/models/longcat_flash.py` modified +2/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/longcat_flash.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #14161 - feat: longcat flash add aux layers capture for eagle3

- 链接: https://github.com/sgl-project/sglang/pull/14161
- 状态/时间: merged / 2025-11-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/longcat_flash.py`；关联提交 `67e6ef4b2d24`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+25/-3，可读 patch 78 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/sglang/srt/models/longcat_flash.py` modified +25/-3 (28 lines); hunks: -32,7 +32,7; -511,6 +511,7 @@ def __init__(; symbols: __init__, get_input_embeddings, forward, LongcatFlashForCausalLM，涉及 `__init__, get_input_embeddings, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/longcat_flash.py` modified +25/-3 (28 lines); hunks: -32,7 +32,7; -511,6 +511,7 @@ def __init__(; symbols: __init__, get_input_embeddings, forward, LongcatFlashForCausalLM
- 关键代码摘录:

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

- 提取文件（未人工审阅）:
  - runtime: `python/sglang/srt/models/longcat_flash.py` modified +25/-3
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/longcat_flash.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #17838 - Feature/support longcat flash lite

- 链接: https://github.com/sgl-project/sglang/pull/17838
- 状态/时间: merged / 2026-03-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/configs/longcat_flash.py`, `python/sglang/srt/models/longcat_flash.py`；关联提交 `eb4ba1bde254`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 16 个文件，+838/-15，可读 patch 1156 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/sglang/srt/models/longcat_flash.py` modified +29/-8 (37 lines); hunks: -64,6 +64,7; -329,7 +330,7 @@ def __init__(; symbols: __init__, forward, load_weights，涉及 `__init__, forward, load_weights`；`python/sglang/srt/configs/longcat_flash.py` modified +8/-0 (8 lines); hunks: -53,6 +53,9 @@ def __init__(; -102,3 +105,8 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/longcat_flash.py` modified +29/-8 (37 lines); hunks: -64,6 +64,7; -329,7 +330,7 @@ def __init__(; symbols: __init__, forward, load_weights
  - `python/sglang/srt/configs/longcat_flash.py` modified +8/-0 (8 lines); hunks: -53,6 +53,9 @@ def __init__(; -102,3 +105,8 @@ def __init__(; symbols: __init__
- 关键代码摘录:

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

- 提取文件（未人工审阅）:
  - runtime: `python/sglang/srt/models/longcat_flash.py` modified +29/-8; `python/sglang/srt/configs/longcat_flash.py` modified +8/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/jit_kernel/csrc/ngram_embedding.cuh`, `python/sglang/jit_kernel/ngram_embedding.py`, `python/sglang/srt/configs/longcat_flash.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #30275 - [Model] Support LongCat 2.0 FP8

- 链接: https://github.com/sgl-project/sglang/pull/30275
- 状态/时间: merged / 2026-07-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/configs/longcat_flash.py`, `python/sglang/srt/models/longcat_flash.py`；关联提交 `e339c83f82e4`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 23 个文件，+481/-91，可读 patch 1125 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/sglang/srt/models/longcat_flash.py` modified +40/-11 (51 lines); hunks: -326,8 +326,8 @@ def __init__(; -420,18 +420,24 @@ def forward(; symbols: __init__, forward, forward_mlp，涉及 `__init__, forward, forward_mlp`；`python/sglang/srt/configs/longcat_flash.py` modified +12/-0 (12 lines); hunks: -56,6 +56,9 @@ def __init__(; -105,6 +108,15 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/longcat_flash.py` modified +40/-11 (51 lines); hunks: -326,8 +326,8 @@ def __init__(; -420,18 +420,24 @@ def forward(; symbols: __init__, forward, forward_mlp
  - `python/sglang/srt/configs/longcat_flash.py` modified +12/-0 (12 lines); hunks: -56,6 +56,9 @@ def __init__(; -105,6 +108,15 @@ def __init__(; symbols: __init__
- 关键代码摘录:

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

- 提取文件（未人工审阅）:
  - runtime: `python/sglang/srt/models/longcat_flash.py` modified +40/-11; `python/sglang/srt/configs/longcat_flash.py` modified +12/-0
- 验证与风险: diff 自带测试面 `test/registered/jit/benchmark/bench_ngram_compute_decode.py`, `test/registered/jit/test_ngram_embedding.py`, `test/registered/unit/model_executor/test_ngram_token_table.py`, `test/registered/unit/test_model_overrides.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #31311 - Fix LongCat-2.0 real EP (deepep): double all-reduce + ScMoE RoPE crash

- 链接: https://github.com/sgl-project/sglang/pull/31311
- 状态/时间: merged / 2026-07-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/longcat_flash.py`；关联提交 `1843384c7a59`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+44/-1，可读 patch 73 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/sglang/srt/models/longcat_flash.py` modified +44/-1 (45 lines); hunks: -123,6 +123,24; -284,7 +302,12 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Ten...; symbols: _scmoe_align_rows, LongcatFlashMLP, __init__, forward，涉及 `_scmoe_align_rows, LongcatFlashMLP, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/longcat_flash.py` modified +44/-1 (45 lines); hunks: -123,6 +123,24; -284,7 +302,12 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Ten...; symbols: _scmoe_align_rows, LongcatFlashMLP, __init__, forward
- 关键代码摘录:

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

- 提取文件（未人工审阅）:
  - runtime: `python/sglang/srt/models/longcat_flash.py` modified +44/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/longcat_flash.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #30247 - Optimize LongCat-Flash router GEMM with the HPC-Ops bf16xfp32 kernel

- 链接: https://github.com/sgl-project/sglang/pull/30247
- 状态/时间: merged / 2026-07-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/longcat_flash.py`, `test/registered/unit/models/test_longcat_flash_router_hpc_gemm.py`；关联提交 `e4eea7ce2ffa`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 4 个文件，+336/-7，可读 patch 393 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/registered/unit/models/test_longcat_flash_router_hpc_gemm.py` added +116/-0 (116 lines); hunks: -0,0 +1,116; symbols: _longcat_config, TestLongcatFlashRouterHpcGemm, _assert_dispatches_to_hpc_gemm, _assert_uses_classifier，涉及 `_longcat_config, TestLongcatFlashRouterHpcGemm, _assert_dispatches_to_hpc_gemm`；`python/sglang/srt/models/longcat_flash.py` modified +29/-2 (31 lines); hunks: -37,6 +37,7; -122,6 +123,15; symbols: LongcatFlashMLP, __init__, forward，涉及 `LongcatFlashMLP, __init__, forward`。
- 代码 diff 细节:
  - `test/registered/unit/models/test_longcat_flash_router_hpc_gemm.py` added +116/-0 (116 lines); hunks: -0,0 +1,116; symbols: _longcat_config, TestLongcatFlashRouterHpcGemm, _assert_dispatches_to_hpc_gemm, _assert_uses_classifier
  - `python/sglang/srt/models/longcat_flash.py` modified +29/-2 (31 lines); hunks: -37,6 +37,7; -122,6 +123,15; symbols: LongcatFlashMLP, __init__, forward
- 关键代码摘录:

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

- 提取文件（未人工审阅）:
  - tests: `test/registered/unit/models/test_longcat_flash_router_hpc_gemm.py` added +116/-0
  - runtime: `python/sglang/srt/models/longcat_flash.py` modified +29/-2
- 验证与风险: diff 自带测试面 `test/registered/gemm/test_linear_bf16_fp32_hpc.py`, `test/registered/unit/models/test_longcat_flash_router_hpc_gemm.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #32125 - ci: add LongCat-Flash-Lite-FP8 8-GPU nightly test + fix NextN rope_theta

- 链接: https://github.com/sgl-project/sglang/pull/32125
- 状态/时间: merged / 2026-07-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/longcat_flash_nextn.py`；关联提交 `8389d79e43af`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+89/-2，可读 patch 99 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/sglang/srt/models/longcat_flash_nextn.py` modified +6/-2 (8 lines); hunks: -131,8 +131,12 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/longcat_flash_nextn.py` modified +6/-2 (8 lines); hunks: -131,8 +131,12 @@ def __init__(; symbols: __init__
- 关键代码摘录:

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

- 提取文件（未人工审阅）:
  - runtime: `python/sglang/srt/models/longcat_flash_nextn.py` modified +6/-2
- 验证与风险: diff 自带测试面 `test/registered/8-gpu-models/test_longcat_flash_lite_fp8.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #40799 - [Fix] Avoid duplicate residual in LongCat MoE shortcut

- 链接: https://github.com/sgl-project/sglang/pull/40799
- 状态/时间: merged / 2026-09-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/longcat_flash.py`；关联提交 `0e80c73c925e`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+86/-1，可读 patch 95 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/sglang/srt/models/longcat_flash.py` modified +2/-1 (3 lines); hunks: -501,7 +501,8 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/longcat_flash.py` modified +2/-1 (3 lines); hunks: -501,7 +501,8 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/longcat_flash.py
@@ -501,7 +501,8 @@ def forward(
-        moe_residual = residual.clone()
+        # The final gather adds its residual; the dense branch already carries it.
+        moe_residual = torch.zeros_like(residual)
```

- 提取文件（未人工审阅）:
  - runtime: `python/sglang/srt/models/longcat_flash.py` modified +2/-1
- 验证与风险: diff 自带测试面 `test/registered/unit/models/test_longcat_flash_shortcut.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #41436 - [Fix] LongCat-Flash under attention DP: branch and merge the dense FFNs through the communicators

- 链接: https://github.com/sgl-project/sglang/pull/41436
- 状态/时间: merged / 2026-09-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/longcat_flash.py`；关联提交 `0971450c168f`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 5 个文件，+298/-83，可读 patch 503 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/sglang/srt/models/longcat_flash.py` modified +21/-56 (77 lines); hunks: -90,9 +90,8; -140,23 +139,6; symbols: _scmoe_align_rows, LongcatFlashMLP, __init__，涉及 `_scmoe_align_rows, LongcatFlashMLP, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/longcat_flash.py` modified +21/-56 (77 lines); hunks: -90,9 +90,8; -140,23 +139,6; symbols: _scmoe_align_rows, LongcatFlashMLP, __init__
- 关键代码摘录:

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

- 提取文件（未人工审阅）:
  - runtime: `python/sglang/srt/models/longcat_flash.py` modified +21/-56
- 验证与风险: diff 自带测试面 `test/registered/unit/layers/test_communicator_ffn_exit.py`, `test/registered/unit/layers/test_declared_decoder_boundary.py`, `test/registered/unit/models/test_longcat_flash_shortcut.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
