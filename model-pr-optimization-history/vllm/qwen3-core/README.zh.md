# vllm Qwen3 Core 模型 PR 优化历史

## 文档口径

- 重做日期: 2026-04-25
- 源码基线: `vllm-project/vllm` 当前追溯 worktree commit `95995bbef8`
- PR 收集规则: 先从模型实现、配置、processor、parser、docs/tests 等相关文件执行 `git log --name-only -- <model-files>`，再按 commit subject 的模型关键词过滤，最后用 GitHub Pull Request files API 读取每个 PR 的最终 diff。
- 额外保留规则: 原 history/skill 已显式引用但未出现在当前实现文件 git trace 中的 PR 会保留，并在卡片里标注来源。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `tests/models/multimodal/pooling/test_colqwen3.py` | [#34398](https://github.com/vllm-project/vllm/pull/34398), [#34574](https://github.com/vllm-project/vllm/pull/34574) |
| `vllm/model_executor/models/colqwen3.py` | [#34398](https://github.com/vllm-project/vllm/pull/34398), [#34574](https://github.com/vllm-project/vllm/pull/34574) |
| `vllm/model_executor/models/qwen3.py` | [#15289](https://github.com/vllm-project/vllm/pull/15289), [#17735](https://github.com/vllm-project/vllm/pull/17735), [#19260](https://github.com/vllm-project/vllm/pull/19260), [#21924](https://github.com/vllm-project/vllm/pull/21924), [#29816](https://github.com/vllm-project/vllm/pull/29816) |
| `vllm/model_executor/models/qwen3_dflash.py` | 无直接 PR 号提交 |
| `vllm/model_executor/models/qwen3_moe.py` | [#15289](https://github.com/vllm-project/vllm/pull/15289), [#16203](https://github.com/vllm-project/vllm/pull/16203), [#17735](https://github.com/vllm-project/vllm/pull/17735), [#18118](https://github.com/vllm-project/vllm/pull/18118), [#19598](https://github.com/vllm-project/vllm/pull/19598), [#19860](https://github.com/vllm-project/vllm/pull/19860), [#20101](https://github.com/vllm-project/vllm/pull/20101), [#20815](https://github.com/vllm-project/vllm/pull/20815), [#21924](https://github.com/vllm-project/vllm/pull/21924), [#22017](https://github.com/vllm-project/vllm/pull/22017), [#22785](https://github.com/vllm-project/vllm/pull/22785), [#23169](https://github.com/vllm-project/vllm/pull/23169), ... (24 total) |
| `vllm/transformers_utils/configs/colqwen3.py` | [#34398](https://github.com/vllm-project/vllm/pull/34398) |

## PR 覆盖总览

- git 追溯 PR 数: 27
- 原文档显式引用补充 PR 数: 0
- 当前文档总 PR 数: 27
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-04-07 | [#15289](https://github.com/vllm-project/vllm/pull/15289) | merged | [Model] Add Qwen3 and Qwen3MoE | `vllm/model_executor/models/qwen3_moe.py`, `vllm/model_executor/models/qwen3.py` |
| 2025-04-08 | [#16203](https://github.com/vllm-project/vllm/pull/16203) | merged | [Model] use AutoWeightsLoader for phimoe,qwen2_moe,qwen3_moe | `vllm/model_executor/models/qwen3_moe.py` |
| 2025-05-07 | [#17735](https://github.com/vllm-project/vllm/pull/17735) | merged | [Kernel] Use fused rmsnorm for some models like qwen3 series | `vllm/model_executor/models/qwen3.py`, `vllm/model_executor/models/qwen3_moe.py` |
| 2025-05-14 | [#18118](https://github.com/vllm-project/vllm/pull/18118) | merged | [Model] Add packed_modules_mapping for Qwen3-MOE | `vllm/model_executor/models/qwen3_moe.py` |
| 2025-06-11 | [#19260](https://github.com/vllm-project/vllm/pull/19260) | merged | [New Model]: Support Qwen3 Embedding & Reranker | `vllm/model_executor/models/qwen3.py` |
| 2025-06-20 | [#19860](https://github.com/vllm-project/vllm/pull/19860) | merged | [Chore]: qwen3-moe-type-hints-mistake | `vllm/model_executor/models/qwen3_moe.py` |
| 2025-06-30 | [#19598](https://github.com/vllm-project/vllm/pull/19598) | merged | [Bugfix] Skip loading extra parameters for modelopt Qwen3 MoE model | `vllm/model_executor/models/qwen3_moe.py` |
| 2025-07-30 | [#20815](https://github.com/vllm-project/vllm/pull/20815) | merged | [Feature][EPLB] Add eplb support for Qwen3 | `vllm/model_executor/models/qwen3_moe.py` |
| 2025-08-07 | [#21924](https://github.com/vllm-project/vllm/pull/21924) | merged | [Qwen3] Enable dual-chunk-attention support for Qwen3 models. | `vllm/model_executor/models/qwen3.py`, `vllm/model_executor/models/qwen3_moe.py` |
| 2025-08-08 | [#20101](https://github.com/vllm-project/vllm/pull/20101) | merged | Add ModelOpt Qwen3 nvfp4 support | `vllm/model_executor/models/qwen3_moe.py` |
| 2025-08-11 | [#22017](https://github.com/vllm-project/vllm/pull/22017) | merged | [BUGFIX] KeyError 'layers.14.mlp.gate.g_idx' for Qwen3-MoE with GPTQ on ROCm | `vllm/model_executor/models/qwen3_moe.py` |
| 2025-08-13 | [#22785](https://github.com/vllm-project/vllm/pull/22785) | merged | Fix GGUF loader for Qwen3 MoE. | `vllm/model_executor/models/qwen3_moe.py` |
| 2025-08-19 | [#23169](https://github.com/vllm-project/vllm/pull/23169) | merged | [Model] Removes redundant all-reduce operation in Qwen3MoeSparseMoeBlock | `vllm/model_executor/models/qwen3_moe.py` |
| 2025-08-25 | [#23490](https://github.com/vllm-project/vllm/pull/23490) | merged | [Bugfix] Fix Qwen3 MoE GPTQ inference | `vllm/model_executor/models/qwen3_moe.py` |
| 2025-09-01 | [#23994](https://github.com/vllm-project/vllm/pull/23994) | merged | [BUGFIX] GPTQ quantization compatibility for Qwen3 MOE models (AutoGPTQ and AutoRound-GPTQ) | `vllm/model_executor/models/qwen3_moe.py` |
| 2025-09-17 | [#24727](https://github.com/vllm-project/vllm/pull/24727) | merged | [Model] Support Qwen3-VL Model Series | `vllm/model_executor/models/qwen3_moe.py` |
| 2025-09-27 | [#24982](https://github.com/vllm-project/vllm/pull/24982) | merged | [Bugfix][WideEP] Apply TP Attn + EP MoE fix to other models | `vllm/model_executor/models/qwen3_moe.py` |
| 2025-09-28 | [#25814](https://github.com/vllm-project/vllm/pull/25814) | merged | [Bugfix] Fix Qwen3-VL regression from #24982 | `vllm/model_executor/models/qwen3_moe.py` |
| 2025-10-11 | [#26485](https://github.com/vllm-project/vllm/pull/26485) | merged | Add EAGLE-3 Speculative Decoding Support for Qwen3 MoE | `vllm/model_executor/models/qwen3_moe.py` |
| 2025-11-10 | [#27492](https://github.com/vllm-project/vllm/pull/27492) | merged | [Performance] Support FP8 flashinfer TRTLLM MOE on Qwen3 and Qwen-3next | `vllm/model_executor/models/qwen3_moe.py` |
| 2025-12-10 | [#30308](https://github.com/vllm-project/vllm/pull/30308) | merged | [bugfix][quantization] fix quark qwen3 kv_cache quantization | `vllm/model_executor/models/qwen3_moe.py` |
| 2026-01-24 | [#32082](https://github.com/vllm-project/vllm/pull/32082) | merged | [Models] Add `SharedFusedMoE` support to Qwen3MoE | `vllm/model_executor/models/qwen3_moe.py` |
| 2026-02-06 | [#29816](https://github.com/vllm-project/vllm/pull/29816) | merged | [Bugfix][Model] Support LoRA on Qwen3 Output Embedding | `vllm/model_executor/models/qwen3.py`, `vllm/model_executor/models/qwen3_moe.py` |
| 2026-02-14 | [#34398](https://github.com/vllm-project/vllm/pull/34398) | merged | [new model] add COLQwen3 code & Inference | `vllm/model_executor/models/colqwen3.py`, `tests/models/multimodal/pooling/test_colqwen3.py`, `vllm/transformers_utils/configs/colqwen3.py` |
| 2026-02-21 | [#34574](https://github.com/vllm-project/vllm/pull/34574) | merged | [Frontend] Support multimodal inputs for late-interaction scoring (ColQwen3) + NewModel: nvidia/nemotron-colembed | `tests/models/multimodal/pooling/test_colqwen3.py`, `vllm/model_executor/models/colqwen3.py` |
| 2026-03-04 | [#35656](https://github.com/vllm-project/vllm/pull/35656) | merged | [Bugfix][Model] Fix FP8 k_scale/v_scale not loaded for Qwen3-MoE | `vllm/model_executor/models/qwen3_moe.py` |
| 2026-04-23 | [#40664](https://github.com/vllm-project/vllm/pull/40664) | merged | [BugFix]fix Qwen3 MoE call gate twice | `vllm/model_executor/models/qwen3_moe.py` |

## 逐 PR diff 审计卡

### PR #15289 - [Model] Add Qwen3 and Qwen3MoE

- 链接: https://github.com/vllm-project/vllm/pull/15289
- 状态/时间: merged / 2025-04-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3.py`, `vllm/model_executor/models/qwen3_moe.py`；关联提交 `7699258ef013`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+893/-5，可读 patch 937 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Add Qwen3 and Qwen3MoE」；模型线: Qwen3 Core；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/qwen3_moe.py`, `vllm/model_executor/models/qwen3.py`；PR 正文摘要: Description Recently, I have submitted a pull request to Hugging Face Transformers containing the implementation of the Qwen3 and Qwen3MoE model. I would also like to contribute...。
- 实现要点: `vllm/model_executor/models/qwen3_moe.py` added +531/-0 (531 lines); hunks: -0,0 +1,531; symbols: Qwen3MoeMLP, __init__, forward, Qwen3MoeSparseMoeBlock，涉及 `Qwen3MoeMLP, __init__, forward`；`vllm/model_executor/models/qwen3.py` added +329/-0 (329 lines); hunks: -0,0 +1,329; symbols: Qwen3Attention, __init__, forward, Qwen3DecoderLayer，涉及 `Qwen3Attention, __init__, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_moe.py` added +531/-0 (531 lines); hunks: -0,0 +1,531; symbols: Qwen3MoeMLP, __init__, forward, Qwen3MoeSparseMoeBlock
  - `vllm/model_executor/models/qwen3.py` added +329/-0 (329 lines); hunks: -0,0 +1,329; symbols: Qwen3Attention, __init__, forward, Qwen3DecoderLayer
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_moe.py
@@ -0,0 +1,531 @@
+# SPDX-License-Identifier: Apache-2.0
+# Copyright 2024 The Qwen team.
+# Copyright 2023 The vLLM team.
+# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
+#
+# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
diff -- vllm/model_executor/models/qwen3.py
@@ -0,0 +1,329 @@
+# SPDX-License-Identifier: Apache-2.0
+# Copyright 2024 The Qwen team.
+# Copyright 2023 The vLLM team.
+# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
+#
+# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_moe.py` added +531/-0; `vllm/model_executor/models/qwen3.py` added +329/-0
- 验证与风险: diff 自带测试面 `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #16203 - [Model] use AutoWeightsLoader for phimoe,qwen2_moe,qwen3_moe

- 链接: https://github.com/vllm-project/vllm/pull/16203
- 状态/时间: merged / 2025-04-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_moe.py`；关联提交 `5a1e1c8353b9`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+220/-198，可读 patch 514 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] use AutoWeightsLoader for phimoe,qwen2_moe,qwen3_moe」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_moe.py`；PR 正文摘要: FIX https://github.com/vllm-project/vllm/issues/15697 - Qwen3MoeForCausalLM Because can't download Qwen/Qwen3-MoE-15B-A2B this model, so not test. - Qwen2MoeForCausalLM - PhiMoE...。
- 实现要点: `vllm/model_executor/models/qwen3_moe.py` modified +65/-58 (123 lines); hunks: -52,7 +52,8; -326,7 +327,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, forward, Qwen3MoeForCausalLM, get_input_embeddings，涉及 `__init__, forward, Qwen3MoeForCausalLM`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_moe.py` modified +65/-58 (123 lines); hunks: -52,7 +52,8; -326,7 +327,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, forward, Qwen3MoeForCausalLM, get_input_embeddings
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_moe.py
@@ -52,7 +52,8 @@
-from .utils import (extract_layer_index, is_pp_missing_parameter,
+from .utils import (AutoWeightsLoader, extract_layer_index,
+                    is_pp_missing_parameter,
@@ -326,7 +327,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
+        self.config = config
@@ -375,60 +376,6 @@ def forward(
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_moe.py` modified +65/-58
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/phimoe.py`, `vllm/model_executor/models/qwen2_moe.py`, `vllm/model_executor/models/qwen3_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #17735 - [Kernel] Use fused rmsnorm for some models like qwen3 series

- 链接: https://github.com/vllm-project/vllm/pull/17735
- 状态/时间: merged / 2025-05-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3.py`, `vllm/model_executor/models/qwen3_moe.py`；关联提交 `f80ae5bdcfa7`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+19/-15，可读 patch 97 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Kernel] Use fused rmsnorm for some models like qwen3 series」；模型线: Qwen3 Core；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/qwen3.py`, `vllm/model_executor/models/qwen3_moe.py`；PR 正文摘要: Some models like qwen3 series still use forward_native method when q and k need to be normalized. The reason why fused rmsnorm kernel is not being used is that before q norm and...。
- 实现要点: `vllm/model_executor/models/qwen3.py` modified +2/-2 (4 lines); hunks: -133,11 +133,11 @@ def forward(; symbols: forward，涉及 `forward`；`vllm/model_executor/models/qwen3_moe.py` modified +2/-2 (4 lines); hunks: -225,12 +225,12 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3.py` modified +2/-2 (4 lines); hunks: -133,11 +133,11 @@ def forward(; symbols: forward
  - `vllm/model_executor/models/qwen3_moe.py` modified +2/-2 (4 lines); hunks: -225,12 +225,12 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3.py
@@ -133,11 +133,11 @@ def forward(
-        q_by_head = self.q_norm.forward_native(q_by_head)
+        q_by_head = self.q_norm(q_by_head)
-        k_by_head = self.k_norm.forward_native(k_by_head)
+        k_by_head = self.k_norm(k_by_head)
diff -- vllm/model_executor/models/qwen3_moe.py
@@ -225,12 +225,12 @@ def forward(
-        q_by_head = self.q_norm.forward_native(q_by_head)
+        q_by_head = self.q_norm(q_by_head)
-        k_by_head = self.k_norm.forward_native(k_by_head)
+        k_by_head = self.k_norm(k_by_head)
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3.py` modified +2/-2; `vllm/model_executor/models/qwen3_moe.py` modified +2/-2
- 验证与风险: runtime 路径改动集中在 `vllm/_custom_ops.py`, `vllm/model_executor/models/intern_vit.py`, `vllm/model_executor/models/molmo.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #18118 - [Model] Add packed_modules_mapping for Qwen3-MOE

- 链接: https://github.com/vllm-project/vllm/pull/18118
- 状态/时间: merged / 2025-05-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_moe.py`；关联提交 `63dc3426e078`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+11/-0，可读 patch 18 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Add packed_modules_mapping for Qwen3-MOE」；模型线: Qwen3 Core；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/qwen3_moe.py`；PR 正文未提供可用摘要。
- 实现要点: `vllm/model_executor/models/qwen3_moe.py` modified +11/-0 (11 lines); hunks: -475,6 +475,17 @@ def load_weights(self, weights: Iterable[Tuple[str,; symbols: load_weights, Qwen3MoeForCausalLM，涉及 `load_weights, Qwen3MoeForCausalLM`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_moe.py` modified +11/-0 (11 lines); hunks: -475,6 +475,17 @@ def load_weights(self, weights: Iterable[Tuple[str,; symbols: load_weights, Qwen3MoeForCausalLM
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_moe.py
@@ -475,6 +475,17 @@ def load_weights(self, weights: Iterable[Tuple[str,
+    packed_modules_mapping = {
+        "qkv_proj": [
+            "q_proj",
+            "k_proj",
+            "v_proj",
+        ],
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_moe.py` modified +11/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #19260 - [New Model]: Support Qwen3 Embedding & Reranker

- 链接: https://github.com/vllm-project/vllm/pull/19260
- 状态/时间: merged / 2025-06-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3.py`；关联提交 `3952731e8f25`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+396/-19，可读 patch 470 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[New Model]: Support Qwen3 Embedding & Reranker」；模型线: Qwen3 Core；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/qwen3.py`；PR 正文摘要: - Qwen3 Embedding - Qwen/Qwen3-Embedding-0.6B - Qwen/Qwen3-Embedding-4B - Qwen/Qwen3-Embedding-8B - Qwen3 Reranker - Qwen/Qwen3-Reranker-0.6B - Qwen/Qwen3-Reranker-4B - Qwen/Qwe...。
- 实现要点: `vllm/model_executor/models/qwen3.py` modified +123/-2 (125 lines); hunks: -38,13 +38,15; -319,3 +321,122 @@ def load_weights(self, weights: Iterable[tuple[str,; symbols: load_weights, Qwen3ForSequenceClassification, __init__, forward，涉及 `load_weights, Qwen3ForSequenceClassification, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3.py` modified +123/-2 (125 lines); hunks: -38,13 +38,15; -319,3 +321,122 @@ def load_weights(self, weights: Iterable[tuple[str,; symbols: load_weights, Qwen3ForSequenceClassification, __init__, forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3.py
@@ -38,13 +38,15 @@
+from vllm.model_executor.layers.pooler import Pooler, PoolingType
+from vllm.model_executor.pooling_metadata import PoolingMetadata
-from vllm.sequence import IntermediateTensors
+from vllm.sequence import IntermediateTensors, PoolerOutput
-from .interfaces import SupportsLoRA, SupportsPP
+from .interfaces import SupportsCrossEncoding, SupportsLoRA, SupportsPP
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3.py` modified +123/-2
- 验证与风险: diff 自带测试面 `tests/models/language/pooling/test_gte.py`, `tests/models/language/pooling/test_qwen3_reranker.py`, `tests/models/language/pooling/test_qwen3_reranker_seq_cls.py`, `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #19860 - [Chore]: qwen3-moe-type-hints-mistake

- 链接: https://github.com/vllm-project/vllm/pull/19860
- 状态/时间: merged / 2025-06-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_moe.py`；关联提交 `e41bf15cd04e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Chore]: qwen3-moe-type-hints-mistake」；模型线: Qwen3 Core；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/qwen3_moe.py`；PR 正文未提供可用摘要。
- 实现要点: `vllm/model_executor/models/qwen3_moe.py` modified +1/-1 (2 lines); hunks: -294,7 +294,7 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_moe.py` modified +1/-1 (2 lines); hunks: -294,7 +294,7 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_moe.py
@@ -294,7 +294,7 @@ def forward(
-    ) -> torch.Tensor:
+    ) -> tuple[torch.Tensor, torch.Tensor]:
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_moe.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #19598 - [Bugfix] Skip loading extra parameters for modelopt Qwen3 MoE model

- 链接: https://github.com/vllm-project/vllm/pull/19598
- 状态/时间: merged / 2025-06-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_moe.py`；关联提交 `f5dfa0753163`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+15/-9，可读 patch 53 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Skip loading extra parameters for modelopt Qwen3 MoE model」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_moe.py`；PR 正文未提供可用摘要。
- 实现要点: `vllm/model_executor/models/qwen3_moe.py` modified +15/-9 (24 lines); hunks: -386,6 +386,11 @@ def load_weights(self, weights: Iterable[tuple[str,; -410,10 +415,11 @@ def load_weights(self, weights: Iterable[tuple[str,; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_moe.py` modified +15/-9 (24 lines); hunks: -386,6 +386,11 @@ def load_weights(self, weights: Iterable[tuple[str,; -410,10 +415,11 @@ def load_weights(self, weights: Iterable[tuple[str,; symbols: load_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_moe.py
@@ -386,6 +386,11 @@ def load_weights(self, weights: Iterable[tuple[str,
+        # Skip loading extra parameters for GPTQ/modelopt models.
+        ignore_suffixes = (".bias", "_bias", ".k_scale", "_k_scale",
+                           ".v_scale", "_v_scale", ".weight_scale",
+                           "_weight_scale", ".input_scale", "_input_scale")
@@ -410,10 +415,11 @@ def load_weights(self, weights: Iterable[tuple[str,
-                # Skip loading extra bias for GPTQ models.
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_moe.py` modified +15/-9
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #20815 - [Feature][EPLB] Add eplb support for Qwen3

- 链接: https://github.com/vllm-project/vllm/pull/20815
- 状态/时间: merged / 2025-07-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_moe.py`；关联提交 `d979dd6bebb1`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+142/-24，可读 patch 273 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Feature][EPLB] Add eplb support for Qwen3」；模型线: Qwen3 Core；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/qwen3_moe.py`；PR 正文未提供可用摘要。
- 实现要点: `vllm/model_executor/models/qwen3_moe.py` modified +142/-24 (166 lines); hunks: -22,7 +22,8; -31,8 +32,9; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_moe.py` modified +142/-24 (166 lines); hunks: -22,7 +22,8; -31,8 +32,9; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_moe.py
@@ -22,7 +22,8 @@
-from collections.abc import Iterable
+import typing
+from collections.abc import Callable, Iterable
@@ -31,8 +32,9 @@
-from vllm.config import CacheConfig, VllmConfig
-from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_moe.py` modified +142/-24
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21924 - [Qwen3] Enable dual-chunk-attention support for Qwen3 models.

- 链接: https://github.com/vllm-project/vllm/pull/21924
- 状态/时间: merged / 2025-08-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3.py`, `vllm/model_executor/models/qwen3_moe.py`；关联提交 `7377131a2ccb`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+60/-31，可读 patch 176 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Qwen3] Enable dual-chunk-attention support for Qwen3 models.」；模型线: Qwen3 Core；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/qwen3.py`, `vllm/model_executor/models/qwen3_moe.py`；PR 正文未提供可用摘要。
- 实现要点: `vllm/model_executor/models/qwen3.py` modified +40/-24 (64 lines); hunks: -23,7 +23,7; -47,27 +47,31; symbols: Qwen3Attention, __init__，涉及 `Qwen3Attention, __init__`；`vllm/model_executor/models/qwen3_moe.py` modified +20/-7 (27 lines); hunks: -159,6 +159,7 @@ def __init__(; -182,6 +183,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3.py` modified +40/-24 (64 lines); hunks: -23,7 +23,7; -47,27 +47,31; symbols: Qwen3Attention, __init__
  - `vllm/model_executor/models/qwen3_moe.py` modified +20/-7 (27 lines); hunks: -159,6 +159,7 @@ def __init__(; -182,6 +183,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3.py
@@ -23,7 +23,7 @@
-from typing import Optional, Union
+from typing import Any, Optional, Union
@@ -47,27 +47,31 @@
-from .utils import AutoWeightsLoader, PPMissingLayer, maybe_prefix
+from .utils import (AutoWeightsLoader, PPMissingLayer, extract_layer_index,
+                    maybe_prefix)
diff -- vllm/model_executor/models/qwen3_moe.py
@@ -159,6 +159,7 @@ def __init__(
+        dual_chunk_attention_config: Optional[dict[str, Any]] = None,
@@ -182,6 +183,7 @@ def __init__(
+        self.dual_chunk_attention_config = dual_chunk_attention_config
@@ -203,14 +205,21 @@ def __init__(
+            dual_chunk_attention_config=dual_chunk_attention_config,
+        )
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3.py` modified +40/-24; `vllm/model_executor/models/qwen3_moe.py` modified +20/-7
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3.py`, `vllm/model_executor/models/qwen3_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #20101 - Add ModelOpt Qwen3 nvfp4 support

- 链接: https://github.com/vllm-project/vllm/pull/20101
- 状态/时间: merged / 2025-08-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_moe.py`；关联提交 `d57dc2364e88`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+58/-37，可读 patch 129 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add ModelOpt Qwen3 nvfp4 support」；模型线: Qwen3 Core；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/qwen3_moe.py`；PR 正文未提供可用摘要。
- 实现要点: `vllm/model_executor/models/qwen3_moe.py` modified +13/-3 (16 lines); hunks: -48,7 +48,8; -471,12 +472,21 @@ def load_weights(self, weights: Iterable[tuple[str,; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_moe.py` modified +13/-3 (16 lines); hunks: -48,7 +48,8; -471,12 +472,21 @@ def load_weights(self, weights: Iterable[tuple[str,; symbols: load_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_moe.py
@@ -48,7 +48,8 @@
-from vllm.model_executor.model_loader.weight_utils import default_weight_loader
+from vllm.model_executor.model_loader.weight_utils import (
+    default_weight_loader, maybe_remap_kv_scale_name)
@@ -471,12 +472,21 @@ def load_weights(self, weights: Iterable[tuple[str,
+                if name.endswith("scale"):
+                    # Remapping the name of FP8 kv-scale.
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_moe.py` modified +13/-3
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/model_loader/weight_utils.py`, `vllm/model_executor/models/qwen2.py`, `vllm/model_executor/models/qwen3_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22017 - [BUGFIX] KeyError 'layers.14.mlp.gate.g_idx' for Qwen3-MoE with GPTQ on ROCm

- 链接: https://github.com/vllm-project/vllm/pull/22017
- 状态/时间: merged / 2025-08-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_moe.py`；关联提交 `1e55dfa7e552`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[BUGFIX] KeyError 'layers.14.mlp.gate.g_idx' for Qwen3-MoE with GPTQ on ROCm」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_moe.py`；PR 正文摘要: Description: This PR fixes a KeyError for 'layers.14.mlp.gate.g_idx' that was popping up when loading GPTQ-quantized Qwen3-MoE models, particularly in ROCm setups. The problem s...。
- 实现要点: `vllm/model_executor/models/qwen3_moe.py` modified +1/-1 (2 lines); hunks: -122,7 +122,7 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_moe.py` modified +1/-1 (2 lines); hunks: -122,7 +122,7 @@ def __init__(; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_moe.py
@@ -122,7 +122,7 @@ def __init__(
-                                     quant_config=None,
+                                     quant_config=quant_config,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_moe.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22785 - Fix GGUF loader for Qwen3 MoE.

- 链接: https://github.com/vllm-project/vllm/pull/22785
- 状态/时间: merged / 2025-08-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_moe.py`；关联提交 `b159c0a67aaa`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+12/-0，可读 patch 26 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix GGUF loader for Qwen3 MoE.」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_moe.py`；PR 正文摘要: Despite upstream repositories (gguf-py, transformers) having added support for Qwen3 MoE GGUF quantization, vLLM GGUF loading is still broken. This PR aims to fix the GGUF loade...。
- 实现要点: `vllm/model_executor/models/qwen3_moe.py` modified +1/-0 (1 lines); hunks: -375,6 +375,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_moe.py` modified +1/-0 (1 lines); hunks: -375,6 +375,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_moe.py
@@ -375,6 +375,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
+            quant_config=quant_config,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_moe.py` modified +1/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/model_loader/gguf_loader.py`, `vllm/model_executor/models/qwen3_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23169 - [Model] Removes redundant all-reduce operation in Qwen3MoeSparseMoeBlock

- 链接: https://github.com/vllm-project/vllm/pull/23169
- 状态/时间: merged / 2025-08-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_moe.py`；关联提交 `4f510bc2a175`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-5，可读 patch 20 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Removes redundant all-reduce operation in Qwen3MoeSparseMoeBlock」；模型线: Qwen3 Core；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/qwen3_moe.py`；PR 正文摘要: Remove redundant all-reduce operation and reuse the all-reduce implementation provided by `FusedMoE` (see `vllm/model_executor/layers/fused_moe/layer.py#L1672`). The existing te...。
- 实现要点: `vllm/model_executor/models/qwen3_moe.py` modified +1/-5 (6 lines); hunks: -139,7 +139,7 @@ def __init__(; -163,10 +163,6 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Ten...; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_moe.py` modified +1/-5 (6 lines); hunks: -139,7 +139,7 @@ def __init__(; -163,10 +163,6 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Ten...; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_moe.py
@@ -139,7 +139,7 @@ def __init__(
-                                reduce_results=False,
+                                reduce_results=True,
@@ -163,10 +163,6 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
-        if self.tp_size > 1:
-            final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(  # noqa E501
-                final_hidden_states)
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_moe.py` modified +1/-5
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23490 - [Bugfix] Fix Qwen3 MoE GPTQ inference

- 链接: https://github.com/vllm-project/vllm/pull/23490
- 状态/时间: merged / 2025-08-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_moe.py`；关联提交 `a9082a4d144e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+18/-6，可读 patch 43 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix Qwen3 MoE GPTQ inference」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_moe.py`；PR 正文摘要: - Fix #23467 (Optional) Documentation Update。
- 实现要点: `vllm/model_executor/models/qwen3_moe.py` modified +18/-6 (24 lines); hunks: -45,6 +45,9; -146,11 +149,20 @@ def __init__(; symbols: __init__, _maybe_ignore_quant_config, forward, load_weights，涉及 `__init__, _maybe_ignore_quant_config, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_moe.py` modified +18/-6 (24 lines); hunks: -45,6 +45,9; -146,11 +149,20 @@ def __init__(; symbols: __init__, _maybe_ignore_quant_config, forward, load_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_moe.py
@@ -45,6 +45,9 @@
+from vllm.model_executor.layers.quantization.gptq import GPTQConfig
+from vllm.model_executor.layers.quantization.gptq_marlin import (
+    GPTQMarlinConfig)
@@ -146,11 +149,20 @@ def __init__(
-        self.gate = ReplicatedLinear(config.hidden_size,
-                                     config.num_experts,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_moe.py` modified +18/-6
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23994 - [BUGFIX] GPTQ quantization compatibility for Qwen3 MOE models (AutoGPTQ and AutoRound-GPTQ)

- 链接: https://github.com/vllm-project/vllm/pull/23994
- 状态/时间: merged / 2025-09-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_moe.py`；关联提交 `183a70967a90`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+17/-4，可读 patch 57 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[BUGFIX] GPTQ quantization compatibility for Qwen3 MOE models (AutoGPTQ and AutoRound-GPTQ)」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_moe.py`；PR 正文摘要: The following PR attempts to make the quantized MOE model chart compatible with AutoGPTQ and Autoround-GPTQ. The PR: https://github.com/vllm-project/vllm/issues/23467 attempted...。
- 实现要点: `vllm/model_executor/models/qwen3_moe.py` modified +7/-3 (10 lines); hunks: -159,9 +159,13 @@ def __init__(; symbols: __init__, _maybe_ignore_quant_config，涉及 `__init__, _maybe_ignore_quant_config`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_moe.py` modified +7/-3 (10 lines); hunks: -159,9 +159,13 @@ def __init__(; symbols: __init__, _maybe_ignore_quant_config
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_moe.py
@@ -159,9 +159,13 @@ def __init__(
-        # seems to avoid gate quantization.
-        # See: https://huggingface.co/Qwen/Qwen3-30B-A3B-GPTQ-Int4
-        if isinstance(quant_config, (GPTQConfig, GPTQMarlinConfig)):
+        # seems to avoid gate quantization while AutoRound does.
+        # See: https://huggingface.co/Qwen/Qwen3-30B-A3B-GPTQ-Int4,
+        # and https://huggingface.co/jart25/Qwen3-Coder-30B-A3B-Instruct-Int4-gptq
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_moe.py` modified +7/-3
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/quantization/gptq.py`, `vllm/model_executor/layers/quantization/gptq_marlin.py`, `vllm/model_executor/models/qwen3_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #24727 - [Model] Support Qwen3-VL Model Series

- 链接: https://github.com/vllm-project/vllm/pull/24727
- 状态/时间: merged / 2025-09-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_moe.py`；关联提交 `0f7acdd73ca6`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 13 个文件，+2084/-17，可读 patch 2262 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Support Qwen3-VL Model Series」；模型线: Qwen3 Core；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/qwen3_moe.py`；PR 正文摘要: This PR adds model support for the upcoming Qwen3-VL models, including both dense and MoE variants. Reference HF implementation - https://github.com/huggingface/transformers/pul...。
- 实现要点: `vllm/model_executor/models/qwen3_moe.py` modified +1/-1 (2 lines); hunks: -378,7 +378,7 @@ class Qwen3MoeModel(nn.Module):; symbols: Qwen3MoeModel, __init__，涉及 `Qwen3MoeModel, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_moe.py` modified +1/-1 (2 lines); hunks: -378,7 +378,7 @@ class Qwen3MoeModel(nn.Module):; symbols: Qwen3MoeModel, __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_moe.py
@@ -378,7 +378,7 @@ class Qwen3MoeModel(nn.Module):
-        config = vllm_config.model_config.hf_config
+        config = vllm_config.model_config.hf_config.get_text_config()
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_moe.py` modified +1/-1
- 验证与风险: diff 自带测试面 `tests/models/multimodal/processing/test_common.py`, `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #24982 - [Bugfix][WideEP] Apply TP Attn + EP MoE fix to other models

- 链接: https://github.com/vllm-project/vllm/pull/24982
- 状态/时间: merged / 2025-09-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_moe.py`；关联提交 `614475401466`, `a5354b3ed247`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 23 个文件，+541/-376，可读 patch 1804 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix][WideEP] Apply TP Attn + EP MoE fix to other models」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_moe.py`；PR 正文摘要: Prior to this PR, in many cases, using TP Attn and EP MoEs with `--tensor-parallel-size N --data-parallel-size M --enable-expert-parallel` would result in factor `N` redundant w...。
- 实现要点: `vllm/model_executor/models/qwen3_moe.py` modified +33/-27 (60 lines); hunks: -29,13 +29,13; -51,6 +51,7; symbols: Qwen3MoeSparseMoeBlock, __init__, forward，涉及 `Qwen3MoeSparseMoeBlock, __init__, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_moe.py` modified +33/-27 (60 lines); hunks: -29,13 +29,13; -51,6 +51,7; symbols: Qwen3MoeSparseMoeBlock, __init__, forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_moe.py
@@ -29,13 +29,13 @@
-from transformers import Qwen3MoeConfig
-                              get_tensor_model_parallel_world_size)
+                              get_tensor_model_parallel_world_size,
+                              tensor_model_parallel_all_gather)
@@ -51,6 +51,7 @@
+from vllm.model_executor.models.utils import sequence_parallel_chunk
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_moe.py` modified +33/-27
- 验证与风险: runtime 路径改动集中在 `vllm/config/parallel.py`, `vllm/distributed/device_communicators/all2all.py`, `vllm/distributed/device_communicators/base_device_communicator.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #25814 - [Bugfix] Fix Qwen3-VL regression from #24982

- 链接: https://github.com/vllm-project/vllm/pull/25814
- 状态/时间: merged / 2025-09-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_moe.py`；关联提交 `614475401466`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+4/-4，可读 patch 36 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix Qwen3-VL regression from #24982」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_moe.py`；PR 正文摘要: 24982 caused a regression since the Qwen3-VL needs to access the text config of Qwen3-MoE via `get_text_config()` Tested locally to make sure both Qwen3-VL and Qwen3-MoE can loa...。
- 实现要点: `vllm/model_executor/models/qwen3_moe.py` modified +4/-4 (8 lines); hunks: -107,7 +107,7 @@ def __init__(; -293,7 +293,7 @@ class Qwen3MoeDecoderLayer(nn.Module):; symbols: __init__, Qwen3MoeDecoderLayer, Qwen3MoeModel，涉及 `__init__, Qwen3MoeDecoderLayer, Qwen3MoeModel`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_moe.py` modified +4/-4 (8 lines); hunks: -107,7 +107,7 @@ def __init__(; -293,7 +293,7 @@ class Qwen3MoeDecoderLayer(nn.Module):; symbols: __init__, Qwen3MoeDecoderLayer, Qwen3MoeModel
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_moe.py
@@ -107,7 +107,7 @@ def __init__(
-        config = vllm_config.model_config.hf_config
+        config = vllm_config.model_config.hf_text_config
@@ -293,7 +293,7 @@ class Qwen3MoeDecoderLayer(nn.Module):
-        config = vllm_config.model_config.hf_config
+        config = vllm_config.model_config.hf_text_config
@@ -372,7 +372,7 @@ class Qwen3MoeModel(nn.Module):
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_moe.py` modified +4/-4
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #26485 - Add EAGLE-3 Speculative Decoding Support for Qwen3 MoE

- 链接: https://github.com/vllm-project/vllm/pull/26485
- 状态/时间: merged / 2025-10-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_moe.py`；关联提交 `d2a71530c159`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+33/-4，可读 patch 85 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add EAGLE-3 Speculative Decoding Support for Qwen3 MoE」；模型线: Qwen3 Core；类别: 文档/测试/CI；主要 diff: `vllm/model_executor/models/qwen3_moe.py`；PR 正文摘要: This PR adds support for EAGLE-3 speculative decoding to the `Qwen3MoeForCausalLM` model, enabling faster inference with draft models like `nm-testing/Mockup-qwen235-eagle3-fp16...。
- 实现要点: `vllm/model_executor/models/qwen3_moe.py` modified +33/-4 (37 lines); hunks: -64,7 +64,7; -422,6 +422,8 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, get_input_embeddings, forward, get_expert_mapping，涉及 `__init__, get_input_embeddings, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_moe.py` modified +33/-4 (37 lines); hunks: -64,7 +64,7; -422,6 +422,8 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, get_input_embeddings, forward, get_expert_mapping
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_moe.py
@@ -64,7 +64,7 @@
-from .interfaces import MixtureOfExperts, SupportsLoRA, SupportsPP
+from .interfaces import MixtureOfExperts, SupportsEagle3, SupportsLoRA, SupportsPP
@@ -422,6 +422,8 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
+        # Track layers for auxiliary hidden state outputs (EAGLE3)
+        self.aux_hidden_state_layers: tuple[int, ...] = ()
@@ -432,7 +434,9 @@ def forward(
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_moe.py` modified +33/-4
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #27492 - [Performance] Support FP8 flashinfer TRTLLM MOE on Qwen3 and Qwen-3next

- 链接: https://github.com/vllm-project/vllm/pull/27492
- 状态/时间: merged / 2025-11-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_moe.py`；关联提交 `34553b9d2702`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+78/-30，可读 patch 251 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Performance] Support FP8 flashinfer TRTLLM MOE on Qwen3 and Qwen-3next」；模型线: Qwen3 Core；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/qwen3_moe.py`；PR 正文摘要: - Integrate multiple routing methods for FP8 flashinfer trtllm MOE, currently only DS and Llama4 - Add FP8 flashinfer trtllm MOE support on Qwen3 and Qwen3-next **Qwen3-Next-80B...。
- 实现要点: `vllm/model_executor/models/qwen3_moe.py` modified +2/-0 (2 lines); hunks: -43,6 +43,7; -171,6 +172,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_moe.py` modified +2/-0 (2 lines); hunks: -43,6 +43,7; -171,6 +172,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_moe.py
@@ -43,6 +43,7 @@
+from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
@@ -171,6 +172,7 @@ def __init__(
+            routing_method_type=RoutingMethodType.Renormalize,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_moe.py` modified +2/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/fused_moe/config.py`, `vllm/model_executor/layers/fused_moe/flashinfer_trtllm_moe.py`, `vllm/model_executor/layers/fused_moe/layer.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #30308 - [bugfix][quantization] fix quark qwen3 kv_cache quantization

- 链接: https://github.com/vllm-project/vllm/pull/30308
- 状态/时间: merged / 2025-12-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_moe.py`；关联提交 `06462392e40f`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+14/-0，可读 patch 28 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[bugfix][quantization] fix quark qwen3 kv_cache quantization」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_moe.py`；PR 正文摘要: Qwen3moe cannot effectively recognize the kv_cache scale. We need to call the get_cache_scale method in the base class to identify it. test scripts:。
- 实现要点: `vllm/model_executor/models/qwen3_moe.py` modified +14/-0 (14 lines); hunks: -403,6 +403,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; -505,6 +506,19 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: __init__, load_weights，涉及 `__init__, load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_moe.py` modified +14/-0 (14 lines); hunks: -403,6 +403,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; -505,6 +506,19 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: __init__, load_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_moe.py
@@ -403,6 +403,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
+        self.quant_config = quant_config
@@ -505,6 +506,19 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
+            if self.quant_config is not None and (
+                scale_name := self.quant_config.get_cache_scale(name)
+            ):
+                # Loading kv cache quantization scales
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_moe.py` modified +14/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #32082 - [Models] Add `SharedFusedMoE` support to Qwen3MoE

- 链接: https://github.com/vllm-project/vllm/pull/32082
- 状态/时间: merged / 2026-01-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_moe.py`；关联提交 `8edaf3857027`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+56/-16，可读 patch 143 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Models] Add `SharedFusedMoE` support to Qwen3MoE」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_moe.py`；PR 正文摘要: - Fix https://github.com/vllm-project/vllm-omni/pull/560#discussion_r2655368800 - Qwen3-omni's MoE talker has share experts in its sparse moe block, while vLLM's Qwen3MoE impl a...。
- 实现要点: `vllm/model_executor/models/qwen3_moe.py` modified +56/-16 (72 lines); hunks: -29,6 +29,7; -42,7 +43,7; symbols: __init__, forward, Qwen3MoeSparseMoeBlock，涉及 `__init__, forward, Qwen3MoeSparseMoeBlock`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_moe.py` modified +56/-16 (72 lines); hunks: -29,6 +29,7; -42,7 +43,7; symbols: __init__, forward, Qwen3MoeSparseMoeBlock
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_moe.py
@@ -29,6 +29,7 @@
+import torch.nn.functional as F
@@ -42,7 +43,7 @@
-from vllm.model_executor.layers.fused_moe import FusedMoE
+from vllm.model_executor.layers.fused_moe import SharedFusedMoE
@@ -86,6 +87,7 @@ def __init__(
+        expert_gate: torch.nn.Linear | None = None,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_moe.py` modified +56/-16
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #29816 - [Bugfix][Model] Support LoRA on Qwen3 Output Embedding

- 链接: https://github.com/vllm-project/vllm/pull/29816
- 状态/时间: merged / 2026-02-06
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3.py`, `vllm/model_executor/models/qwen3_moe.py`；关联提交 `2991dd3d2241`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+132/-13，可读 patch 188 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix][Model] Support LoRA on Qwen3 Output Embedding」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3.py`, `vllm/model_executor/models/qwen3_moe.py`；PR 正文摘要: This PR adds support for LoRA on the embed/unembed layers for Qwen3 dense/MoE models. It is a simplified version of #26115 that removes the changes for supporting zero-padded vo...。
- 实现要点: `vllm/model_executor/models/qwen3.py` modified +5/-0 (5 lines); hunks: -263,6 +263,11 @@ class Qwen3ForCausalLM(nn.Module, SupportsLoRA, SupportsPP,...; symbols: Qwen3ForCausalLM, __init__，涉及 `Qwen3ForCausalLM, __init__`；`vllm/model_executor/models/qwen3_moe.py` modified +5/-0 (5 lines); hunks: -689,6 +689,11 @@ class Qwen3MoeForCausalLM(; symbols: Qwen3MoeForCausalLM, __init__，涉及 `Qwen3MoeForCausalLM, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3.py` modified +5/-0 (5 lines); hunks: -263,6 +263,11 @@ class Qwen3ForCausalLM(nn.Module, SupportsLoRA, SupportsPP,...; symbols: Qwen3ForCausalLM, __init__
  - `vllm/model_executor/models/qwen3_moe.py` modified +5/-0 (5 lines); hunks: -689,6 +689,11 @@ class Qwen3MoeForCausalLM(; symbols: Qwen3MoeForCausalLM, __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3.py
@@ -263,6 +263,11 @@ class Qwen3ForCausalLM(nn.Module, SupportsLoRA, SupportsPP, SupportsEagle3):
+    embedding_modules = {
+        "embed_tokens": "input_embeddings",
+        "lm_head": "output_embeddings",
+    }
diff -- vllm/model_executor/models/qwen3_moe.py
@@ -689,6 +689,11 @@ class Qwen3MoeForCausalLM(
+    embedding_modules = {
+        "embed_tokens": "input_embeddings",
+        "lm_head": "output_embeddings",
+    }
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3.py` modified +5/-0; `vllm/model_executor/models/qwen3_moe.py` modified +5/-0
- 验证与风险: diff 自带测试面 `tests/lora/test_qwen3_unembed.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #34398 - [new model] add COLQwen3 code & Inference

- 链接: https://github.com/vllm-project/vllm/pull/34398
- 状态/时间: merged / 2026-02-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/multimodal/pooling/test_colqwen3.py`, `vllm/model_executor/models/colqwen3.py`, `vllm/transformers_utils/configs/colqwen3.py`；关联提交 `d1ea65d0a1c6`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+935/-0，可读 patch 982 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[new model] add COLQwen3 code & Inference」；模型线: Qwen3 Core；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/colqwen3.py`, `tests/models/multimodal/pooling/test_colqwen3.py`, `vllm/transformers_utils/configs/colqwen3.py`；PR 正文摘要: Add native support for **ColQwen3** multi-modal late interaction models in vLLM. ColPali (arXiv:2407.01449) introduces a ColBERT-style multi-vector retrieval approach for vision...。
- 实现要点: `vllm/model_executor/models/colqwen3.py` added +306/-0 (306 lines); hunks: -0,0 +1,306; symbols: ColQwen3ProcessingInfo, get_hf_config, get_hf_processor, _supports_video，涉及 `ColQwen3ProcessingInfo, get_hf_config, get_hf_processor`；`tests/models/multimodal/pooling/test_colqwen3.py` added +156/-0 (156 lines); hunks: -0,0 +1,156; symbols: _run_token_embed_test, _run_late_interaction_test, _run_relevance_test, test_colqwen3_token_embed，涉及 `_run_token_embed_test, _run_late_interaction_test, _run_relevance_test`；`vllm/transformers_utils/configs/colqwen3.py` added +58/-0 (58 lines); hunks: -0,0 +1,58; symbols: that, ColQwen3Config, for, __init__，涉及 `that, ColQwen3Config, for`。
- 代码 diff 细节:
  - `vllm/model_executor/models/colqwen3.py` added +306/-0 (306 lines); hunks: -0,0 +1,306; symbols: ColQwen3ProcessingInfo, get_hf_config, get_hf_processor, _supports_video
  - `tests/models/multimodal/pooling/test_colqwen3.py` added +156/-0 (156 lines); hunks: -0,0 +1,156; symbols: _run_token_embed_test, _run_late_interaction_test, _run_relevance_test, test_colqwen3_token_embed
  - `vllm/transformers_utils/configs/colqwen3.py` added +58/-0 (58 lines); hunks: -0,0 +1,58; symbols: that, ColQwen3Config, for, __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/colqwen3.py
@@ -0,0 +1,306 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""
+ColQwen3 late interaction model for multi-modal retrieval and reranking.
+ColQwen3 extends Qwen3-VL with a ColBERT-style late interaction head,
+producing per-token embeddings for both text and image inputs. It uses
diff -- tests/models/multimodal/pooling/test_colqwen3.py
@@ -0,0 +1,156 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Tests for ColQwen3 late interaction model for multi-modal retrieval.
+ColQwen3 is a multi-vector retrieval model based on Qwen3-VL backbone with
+ColBERT-style late interaction scoring (MaxSim). It produces per-token
+embeddings for both text and image inputs.
diff -- vllm/transformers_utils/configs/colqwen3.py
@@ -0,0 +1,58 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/models/colqwen3.py` added +306/-0; `vllm/transformers_utils/configs/colqwen3.py` added +58/-0
  - tests: `tests/models/multimodal/pooling/test_colqwen3.py` added +156/-0
- 验证与风险: diff 自带测试面 `tests/models/multimodal/pooling/test_colqwen3.py`, `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #34574 - [Frontend] Support multimodal inputs for late-interaction scoring (ColQwen3) + NewModel: nvidia/nemotron-colembed

- 链接: https://github.com/vllm-project/vllm/pull/34574
- 状态/时间: merged / 2026-02-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/multimodal/pooling/test_colqwen3.py`, `vllm/model_executor/models/colqwen3.py`；关联提交 `5719a4e4e601`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+532/-66，可读 patch 843 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Frontend] Support multimodal inputs for late-interaction scoring (ColQwen3) + NewModel: nvidia/nemotron-colembed」；模型线: Qwen3 Core；类别: 模型支持/运行时入口；主要 diff: `tests/models/multimodal/pooling/test_colqwen3.py`, `vllm/model_executor/models/colqwen3.py`；PR 正文摘要: Follow-up to #34398 (`[new model] add COLQwen3 code & Inference`). PR #34398 added native support for ColQwen3 multi-modal late interaction models, but the `/score` and `/rerank...。
- 实现要点: `tests/models/multimodal/pooling/test_colqwen3.py` modified +191/-0 (191 lines); hunks: -7,19 +7,31; -33,6 +45,43; symbols: _make_base64_image, _make_image_mm_param, _make_text_mm_param, _run_token_embed_test，涉及 `_make_base64_image, _make_image_mm_param, _make_text_mm_param`；`vllm/model_executor/models/colqwen3.py` modified +8/-6 (14 lines); hunks: -16,6 +16,7; -229,13 +230,14 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `tests/models/multimodal/pooling/test_colqwen3.py` modified +191/-0 (191 lines); hunks: -7,19 +7,31; -33,6 +45,43; symbols: _make_base64_image, _make_image_mm_param, _make_text_mm_param, _run_token_embed_test
  - `vllm/model_executor/models/colqwen3.py` modified +8/-6 (14 lines); hunks: -16,6 +16,7; -229,13 +230,14 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- tests/models/multimodal/pooling/test_colqwen3.py
@@ -7,19 +7,31 @@
+import base64
+from io import BytesIO
+from PIL import Image
+from vllm.entrypoints.chat_utils import (
+    ChatCompletionContentPartImageParam,
+    ChatCompletionContentPartTextParam,
diff -- vllm/model_executor/models/colqwen3.py
@@ -16,6 +16,7 @@
+- nvidia/nemotron-colembed-vl-4b-v2
@@ -229,13 +230,14 @@ def forward(
-        proj_dtype = self.custom_text_proj.weight.dtype  # type: ignore
-        if hidden_states.dtype != proj_dtype:
-            hidden_states = hidden_states.to(proj_dtype)
+        if self.custom_text_proj is not None:
```

- 已读文件:
  - tests: `tests/models/multimodal/pooling/test_colqwen3.py` modified +191/-0
  - runtime: `vllm/model_executor/models/colqwen3.py` modified +8/-6
- 验证与风险: diff 自带测试面 `tests/models/multimodal/pooling/test_colqwen3.py`, `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #35656 - [Bugfix][Model] Fix FP8 k_scale/v_scale not loaded for Qwen3-MoE

- 链接: https://github.com/vllm-project/vllm/pull/35656
- 状态/时间: merged / 2026-03-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_moe.py`；关联提交 `c8c3935b7013`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+129/-36，可读 patch 221 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix][Model] Fix FP8 k_scale/v_scale not loaded for Qwen3-MoE」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_moe.py`；PR 正文摘要: FP8 KV cache scales from llm-compressor checkpoints (e.g. `qkv_proj.k_scale`) were silently dropped during weight loading in Qwen3MoeModel, causing fallback to scale=1.0 and acc...。
- 实现要点: `vllm/model_executor/models/qwen3_moe.py` modified +6/-18 (24 lines); hunks: -535,10 +535,6 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; -562,6 +558,10 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_moe.py` modified +6/-18 (24 lines); hunks: -535,10 +535,6 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; -562,6 +558,10 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: load_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_moe.py
@@ -535,10 +535,6 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
-            ".k_scale",
-            "_k_scale",
-            ".v_scale",
-            "_v_scale",
@@ -562,6 +558,10 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
+            if "scale" in name or "zero_point" in name:
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_moe.py` modified +6/-18
- 验证与风险: diff 自带测试面 `tests/model_executor/test_weight_utils.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #40664 - [BugFix]fix Qwen3 MoE call gate twice

- 链接: https://github.com/vllm-project/vllm/pull/40664
- 状态/时间: merged / 2026-04-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_moe.py`；关联提交 `342c58bc548f`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+13/-5，可读 patch 25 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[BugFix]fix Qwen3 MoE call gate twice」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_moe.py`；PR 正文摘要: Qwen3 MoE model will call gate gemm twice, we find this in xpu kernel profiling. thanks @zufangzhu raise this. see discussion here https://github.com/vllm-project/vllm/pull/3532...。
- 实现要点: `vllm/model_executor/models/qwen3_moe.py` modified +13/-5 (18 lines); hunks: -231,11 +231,19 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Te...; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/qwen3_moe.py` modified +13/-5 (18 lines); hunks: -231,11 +231,19 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Te...; symbols: forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/qwen3_moe.py
@@ -231,11 +231,19 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
-        # router_logits: (num_tokens, n_experts)
-        router_logits, _ = self.gate(hidden_states)
-        final_hidden_states = self.experts(
-            hidden_states=hidden_states, router_logits=router_logits
-        )
+        if self.experts.is_internal_router:
```

- 已读文件:
  - runtime: `vllm/model_executor/models/qwen3_moe.py` modified +13/-5
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/qwen3_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
