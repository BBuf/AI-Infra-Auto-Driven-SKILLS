# vllm Qwen3 Core 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `tests/models/multimodal/pooling/test_colqwen3.py` | [#34398](https://github.com/vllm-project/vllm/pull/34398), [#34574](https://github.com/vllm-project/vllm/pull/34574) |
| `tests/parser/engine/test_qwen3.py` | [#45413](https://github.com/vllm-project/vllm/pull/45413), [#46047](https://github.com/vllm-project/vllm/pull/46047), [#46351](https://github.com/vllm-project/vllm/pull/46351), [#48846](https://github.com/vllm-project/vllm/pull/48846) |
| `vllm/model_executor/models/colqwen3.py` | [#34398](https://github.com/vllm-project/vllm/pull/34398), [#34574](https://github.com/vllm-project/vllm/pull/34574) |
| `vllm/model_executor/models/qwen3.py` | [#15289](https://github.com/vllm-project/vllm/pull/15289), [#17735](https://github.com/vllm-project/vllm/pull/17735), [#19260](https://github.com/vllm-project/vllm/pull/19260), [#21924](https://github.com/vllm-project/vllm/pull/21924), [#29816](https://github.com/vllm-project/vllm/pull/29816) |
| `vllm/model_executor/models/qwen3_dflash.py` | 无直接 PR 号提交 |
| `vllm/model_executor/models/qwen3_moe.py` | [#15289](https://github.com/vllm-project/vllm/pull/15289), [#16203](https://github.com/vllm-project/vllm/pull/16203), [#17735](https://github.com/vllm-project/vllm/pull/17735), [#18118](https://github.com/vllm-project/vllm/pull/18118), [#19598](https://github.com/vllm-project/vllm/pull/19598), [#19860](https://github.com/vllm-project/vllm/pull/19860), [#20101](https://github.com/vllm-project/vllm/pull/20101), [#20815](https://github.com/vllm-project/vllm/pull/20815), [#21924](https://github.com/vllm-project/vllm/pull/21924), [#22017](https://github.com/vllm-project/vllm/pull/22017), [#22785](https://github.com/vllm-project/vllm/pull/22785), [#23169](https://github.com/vllm-project/vllm/pull/23169), ... (24 total) |
| `vllm/parser/qwen3.py` | [#45413](https://github.com/vllm-project/vllm/pull/45413), [#45763](https://github.com/vllm-project/vllm/pull/45763), [#46047](https://github.com/vllm-project/vllm/pull/46047), [#46314](https://github.com/vllm-project/vllm/pull/46314), [#46351](https://github.com/vllm-project/vllm/pull/46351), [#48846](https://github.com/vllm-project/vllm/pull/48846) |
| `vllm/transformers_utils/configs/colqwen3.py` | [#34398](https://github.com/vllm-project/vllm/pull/34398) |

## PR 覆盖总览

- git 追溯 PR 数: 33
- 原文档显式引用补充 PR 数: 4
- 当前文档总 PR 数: 37
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
| 2026-04-23 | [#40671](https://github.com/vllm-project/vllm/pull/40671) | merged | [MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping | `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/llama4.py`, `vllm/model_executor/models/glm4_moe_lite.py` |
| 2026-05-11 | [#42280](https://github.com/vllm-project/vllm/pull/42280) | merged | [Model] Fix missing `maybe_prefix` | `vllm/model_executor/models/arcee.py`, `vllm/model_executor/models/cohere_asr.py`, `vllm/model_executor/models/hunyuan_v1.py` |
| 2026-06-05 | [#43167](https://github.com/vllm-project/vllm/pull/43167) | merged | Remove KV cache scale boilerplate from model weight loading methods | `tests/model_executor/test_eagle_quantization.py`, `vllm/model_executor/models/gpt_oss.py`, `vllm/model_executor/layers/quantization/kv_cache.py` |
| 2026-06-10 | [#39419](https://github.com/vllm-project/vllm/pull/39419) | merged | [SpecDecode] Reduce TP communication for large-vocab draft models speculative decoding | `vllm/model_executor/models/interfaces.py`, `vllm/model_executor/models/llama4_eagle.py`, `vllm/model_executor/models/qwen3.py` |
| 2026-06-15 | [#45413](https://github.com/vllm-project/vllm/pull/45413) | merged | [Frontend] Add Streaming Parser Engine and new Qwen3 Parser | `tests/parser/engine/test_qwen3.py`, `vllm/parser/qwen3.py` |
| 2026-06-16 | [#45763](https://github.com/vllm-project/vllm/pull/45763) | merged | [Bugfix] Fix Qwen3 prompt tool-call reasoning false positive | `vllm/parser/qwen3.py` |
| 2026-06-18 | [#46047](https://github.com/vllm-project/vllm/pull/46047) | merged | [Bugfix] [Parser] Fix Qwen3 latent bug in partial params dropping values containing `<` | `tests/parser/engine/test_qwen3.py`, `vllm/parser/qwen3.py` |
| 2026-06-23 | [#46351](https://github.com/vllm-project/vllm/pull/46351) | merged | fix: stream Qwen3 tool call string arguments | `tests/parser/engine/test_qwen3.py`, `vllm/parser/qwen3.py` |
| 2026-06-25 | [#46314](https://github.com/vllm-project/vllm/pull/46314) | merged | [Frontend] Port seed_oss to the streaming parser engine as a Qwen3 subclass | `vllm/parser/qwen3.py` |
| 2026-07-17 | [#48846](https://github.com/vllm-project/vllm/pull/48846) | merged | [Bugfix][Tool Parser] Preserve whitespace in parameter values (MiniMax M2, Qwen3, MiniCPM5 XML) | `vllm/parser/qwen3.py`, `tests/parser/engine/test_qwen3.py` |

## 逐 PR diff 审计卡

### PR #15289 - [Model] Add Qwen3 and Qwen3MoE

- 链接: https://github.com/vllm-project/vllm/pull/15289
- 状态/时间: merged / 2025-04-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3.py`, `vllm/model_executor/models/qwen3_moe.py`；关联提交 `7699258ef013`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+893/-5，可读 patch 937 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Add Qwen3 and Qwen3MoE」；模型线: Qwen3 Core；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/qwen3_moe.py`, `vllm/model_executor/models/qwen3.py`；技术摘要: 覆盖「[Model] Add Qwen3 and Qwen3MoE」；主要实现面是 `vllm/model_executor/models/qwen3_moe.py`, `vllm/model_executor/models/qwen3.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[Model] use AutoWeightsLoader for phimoe,qwen2_moe,qwen3_moe」；模型线: Qwen3 Core；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/qwen3_moe.py`；技术摘要: 覆盖「[Model] use AutoWeightsLoader for phimoe,qwen2_moe,qwen3_moe」；主要实现面是 `vllm/model_executor/models/qwen3_moe.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[Kernel] Use fused rmsnorm for some models like qwen3 series」；模型线: Qwen3 Core；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/qwen3.py`, `vllm/model_executor/models/qwen3_moe.py`；技术摘要: 覆盖「[Kernel] Use fused rmsnorm for some models like qwen3 series」；主要实现面是 `vllm/model_executor/models/qwen3.py`, `vllm/model_executor/models/qwen3_moe.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[Model] Add packed_modules_mapping for Qwen3-MOE」；模型线: Qwen3 Core；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/qwen3_moe.py`；技术摘要: 覆盖「[Model] Add packed_modules_mapping for Qwen3-MOE」；主要实现面是 `vllm/model_executor/models/qwen3_moe.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[New Model]: Support Qwen3 Embedding & Reranker」；模型线: Qwen3 Core；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/qwen3.py`；技术摘要: 覆盖「[New Model]: Support Qwen3 Embedding & Reranker」；主要实现面是 `vllm/model_executor/models/qwen3.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[Chore]: qwen3-moe-type-hints-mistake」；模型线: Qwen3 Core；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/qwen3_moe.py`；技术摘要: 覆盖「[Chore]: qwen3-moe-type-hints-mistake」；主要实现面是 `vllm/model_executor/models/qwen3_moe.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[Bugfix] Skip loading extra parameters for modelopt Qwen3 MoE model」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_moe.py`；技术摘要: 覆盖「[Bugfix] Skip loading extra parameters for modelopt Qwen3 MoE model」；主要实现面是 `vllm/model_executor/models/qwen3_moe.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[Feature][EPLB] Add eplb support for Qwen3」；模型线: Qwen3 Core；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/qwen3_moe.py`；技术摘要: 覆盖「[Feature][EPLB] Add eplb support for Qwen3」；主要实现面是 `vllm/model_executor/models/qwen3_moe.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[Qwen3] Enable dual-chunk-attention support for Qwen3 models.」；模型线: Qwen3 Core；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/qwen3.py`, `vllm/model_executor/models/qwen3_moe.py`；技术摘要: 覆盖「[Qwen3] Enable dual-chunk-attention support for Qwen3 models.」；主要实现面是 `vllm/model_executor/models/qwen3.py`, `vllm/model_executor/models/qwen3_moe.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「Add ModelOpt Qwen3 nvfp4 support」；模型线: Qwen3 Core；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/qwen3_moe.py`；技术摘要: 覆盖「Add ModelOpt Qwen3 nvfp4 support」；主要实现面是 `vllm/model_executor/models/qwen3_moe.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[BUGFIX] KeyError 'layers.14.mlp.gate.g_idx' for Qwen3-MoE with GPTQ on ROCm」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_moe.py`；技术摘要: 覆盖「[BUGFIX] KeyError 'layers.14.mlp.gate.g_idx' for Qwen3-MoE with GPTQ on ROCm」；主要实现面是 `vllm/model_executor/models/qwen3_moe.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「Fix GGUF loader for Qwen3 MoE.」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_moe.py`；技术摘要: 覆盖「Fix GGUF loader for Qwen3 MoE.」；主要实现面是 `vllm/model_executor/models/qwen3_moe.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[Model] Removes redundant all-reduce operation in Qwen3MoeSparseMoeBlock」；模型线: Qwen3 Core；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/qwen3_moe.py`；技术摘要: 覆盖「[Model] Removes redundant all-reduce operation in Qwen3MoeSparseMoeBlock」；主要实现面是 `vllm/model_executor/models/qwen3_moe.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_moe.py`；关联提交 `a9082a4d144e`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+18/-6，可读 patch 43 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix Qwen3 MoE GPTQ inference」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_moe.py`；技术摘要: 覆盖「[Bugfix] Fix Qwen3 MoE GPTQ inference」；主要实现面是 `vllm/model_executor/models/qwen3_moe.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_moe.py`；关联提交 `183a70967a90`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+17/-4，可读 patch 57 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[BUGFIX] GPTQ quantization compatibility for Qwen3 MOE models (AutoGPTQ and AutoRound-GPTQ)」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_moe.py`；技术摘要: 覆盖「[BUGFIX] GPTQ quantization compatibility for Qwen3 MOE models (AutoGPTQ and AutoRound-GPTQ)」；主要实现面是 `vllm/model_executor/models/qwen3_moe.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[Model] Support Qwen3-VL Model Series」；模型线: Qwen3 Core；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/qwen3_moe.py`；技术摘要: 覆盖「[Model] Support Qwen3-VL Model Series」；主要实现面是 `vllm/model_executor/models/qwen3_moe.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_moe.py`；关联提交 `614475401466`, `a5354b3ed247`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 23 个文件，+541/-376，可读 patch 1804 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix][WideEP] Apply TP Attn + EP MoE fix to other models」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_moe.py`；技术摘要: 覆盖「[Bugfix][WideEP] Apply TP Attn + EP MoE fix to other models」；主要实现面是 `vllm/model_executor/models/qwen3_moe.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_moe.py`；关联提交 `614475401466`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+4/-4，可读 patch 36 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix Qwen3-VL regression from #24982」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_moe.py`；技术摘要: 覆盖「[Bugfix] Fix Qwen3-VL regression from #24982」；主要实现面是 `vllm/model_executor/models/qwen3_moe.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_moe.py`；关联提交 `d2a71530c159`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+33/-4，可读 patch 85 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add EAGLE-3 Speculative Decoding Support for Qwen3 MoE」；模型线: Qwen3 Core；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/qwen3_moe.py`；技术摘要: 覆盖「Add EAGLE-3 Speculative Decoding Support for Qwen3 MoE」；主要实现面是 `vllm/model_executor/models/qwen3_moe.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_moe.py`；关联提交 `34553b9d2702`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+78/-30，可读 patch 251 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Performance] Support FP8 flashinfer TRTLLM MOE on Qwen3 and Qwen-3next」；模型线: Qwen3 Core；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/qwen3_moe.py`；技术摘要: 覆盖「[Performance] Support FP8 flashinfer TRTLLM MOE on Qwen3 and Qwen-3next」；主要实现面是 `vllm/model_executor/models/qwen3_moe.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_moe.py`；关联提交 `06462392e40f`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+14/-0，可读 patch 28 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[bugfix][quantization] fix quark qwen3 kv_cache quantization」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_moe.py`；技术摘要: 覆盖「[bugfix][quantization] fix quark qwen3 kv_cache quantization」；主要实现面是 `vllm/model_executor/models/qwen3_moe.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_moe.py`；关联提交 `8edaf3857027`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+56/-16，可读 patch 143 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Models] Add `SharedFusedMoE` support to Qwen3MoE」；模型线: Qwen3 Core；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/qwen3_moe.py`；技术摘要: 覆盖「[Models] Add `SharedFusedMoE` support to Qwen3MoE」；主要实现面是 `vllm/model_executor/models/qwen3_moe.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[Bugfix][Model] Support LoRA on Qwen3 Output Embedding」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3.py`, `vllm/model_executor/models/qwen3_moe.py`；技术摘要: 覆盖「[Bugfix][Model] Support LoRA on Qwen3 Output Embedding」；主要实现面是 `vllm/model_executor/models/qwen3.py`, `vllm/model_executor/models/qwen3_moe.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[new model] add COLQwen3 code & Inference」；模型线: Qwen3 Core；类别: 文档/测试/CI；主要 diff: `vllm/model_executor/models/colqwen3.py`, `tests/models/multimodal/pooling/test_colqwen3.py`, `vllm/transformers_utils/configs/colqwen3.py`；技术摘要: 覆盖「[new model] add COLQwen3 code & Inference」；主要实现面是 `vllm/model_executor/models/colqwen3.py`, `tests/models/multimodal/pooling/test_colqwen3.py`, `vllm/transformers_utils/configs/colqwen3.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 动机: 标题「[Frontend] Support multimodal inputs for late-interaction scoring (ColQwen3) + NewModel: nvidia/nemotron-colembed」；模型线: Qwen3 Core；类别: 文档/测试/CI；主要 diff: `tests/models/multimodal/pooling/test_colqwen3.py`, `vllm/model_executor/models/colqwen3.py`；技术摘要: 覆盖「[Frontend] Support multimodal inputs for late-interaction scoring (ColQwen3) + NewModel: nvidia/nemotron-colembed」；主要实现面是 `tests/models/multimodal/pooling/test_colqwen3.py`, `vllm/model_executor/models/colqwen3.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_moe.py`；关联提交 `c8c3935b7013`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+129/-36，可读 patch 221 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix][Model] Fix FP8 k_scale/v_scale not loaded for Qwen3-MoE」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_moe.py`；技术摘要: 覆盖「[Bugfix][Model] Fix FP8 k_scale/v_scale not loaded for Qwen3-MoE」；主要实现面是 `vllm/model_executor/models/qwen3_moe.py`。下方保留文件级证据、代码摘录和验证风险。
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
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/qwen3_moe.py`；关联提交 `342c58bc548f`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+13/-5，可读 patch 25 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[BugFix]fix Qwen3 MoE call gate twice」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/qwen3_moe.py`；技术摘要: 覆盖「[BugFix]fix Qwen3 MoE call gate twice」；主要实现面是 `vllm/model_executor/models/qwen3_moe.py`。下方保留文件级证据、代码摘录和验证风险。
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

### PR #40671 - [MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping

- 链接: https://github.com/vllm-project/vllm/pull/40671
- 状态/时间: merged / 2026-04-23
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 53 个文件，+254/-98，可读 patch 1073 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping」；模型线: Qwen3 Core；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/llama4.py`, `vllm/model_executor/models/glm4_moe_lite.py`；技术摘要: 覆盖「[MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping」；主要实现面是 `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/llama4.py`, `vllm/model_executor/models/glm4_moe_lite.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/fused_moe/layer.py` modified +19/-0 (19 lines); hunks: -1618,6 +1618,25 @@ def extra_repr(self) -> str:; symbols: extra_repr, fused_moe_make_expert_params_mapping，涉及 `extra_repr, fused_moe_make_expert_params_mapping`；`vllm/model_executor/models/llama4.py` modified +7/-4 (11 lines); hunks: -36,7 +36,10; -414,7 +417,7 @@ def load_moe_expert_weights(; symbols: load_moe_expert_weights, load_weights，涉及 `load_moe_expert_weights, load_weights`；`vllm/model_executor/models/glm4_moe_lite.py` modified +6/-4 (10 lines); hunks: -41,7 +41,9; -308,7 +310,7 @@ def make_empty_intermediate_tensors(; symbols: make_empty_intermediate_tensors, get_expert_mapping, load_weights, compute_logits，涉及 `make_empty_intermediate_tensors, get_expert_mapping, load_weights`；`vllm/model_executor/models/AXK1.py` modified +6/-3 (9 lines); hunks: -42,7 +42,10; -916,7 +919,7 @@ def compute_logits(; symbols: compute_logits, get_expert_mapping, load_weights，涉及 `compute_logits, get_expert_mapping, load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/fused_moe/layer.py` modified +19/-0 (19 lines); hunks: -1618,6 +1618,25 @@ def extra_repr(self) -> str:; symbols: extra_repr, fused_moe_make_expert_params_mapping
  - `vllm/model_executor/models/llama4.py` modified +7/-4 (11 lines); hunks: -36,7 +36,10; -414,7 +417,7 @@ def load_moe_expert_weights(; symbols: load_moe_expert_weights, load_weights
  - `vllm/model_executor/models/glm4_moe_lite.py` modified +6/-4 (10 lines); hunks: -41,7 +41,9; -308,7 +310,7 @@ def make_empty_intermediate_tensors(; symbols: make_empty_intermediate_tensors, get_expert_mapping, load_weights, compute_logits
  - `vllm/model_executor/models/AXK1.py` modified +6/-3 (9 lines); hunks: -42,7 +42,10; -916,7 +919,7 @@ def compute_logits(; symbols: compute_logits, get_expert_mapping, load_weights
  - `vllm/model_executor/models/afmoe.py` modified +5/-2 (7 lines); hunks: -18,7 +18,10; -479,7 +482,7 @@ def make_empty_intermediate_tensors(; symbols: make_empty_intermediate_tensors, get_expert_mapping
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/fused_moe/layer.py
@@ -1618,6 +1618,25 @@ def extra_repr(self) -> str:
+# This is a temporary forwarding method which will be removed/modified layer.
+def fused_moe_make_expert_params_mapping(
+    model: torch.nn.Module,
+    ckpt_gate_proj_name: str,
+    ckpt_down_proj_name: str,
+    ckpt_up_proj_name: str,
diff -- vllm/model_executor/models/llama4.py
@@ -36,7 +36,10 @@
-from vllm.model_executor.layers.fused_moe import FusedMoE
+from vllm.model_executor.layers.fused_moe import (
+    FusedMoE,
+    fused_moe_make_expert_params_mapping,
+)
@@ -414,7 +417,7 @@ def load_moe_expert_weights(
diff -- vllm/model_executor/models/glm4_moe_lite.py
@@ -41,7 +41,9 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/fused_moe/layer.py` modified +19/-0; `vllm/model_executor/models/llama4.py` modified +7/-4; `vllm/model_executor/models/glm4_moe_lite.py` modified +6/-4; `vllm/model_executor/models/AXK1.py` modified +6/-3; `vllm/model_executor/models/afmoe.py` modified +5/-2; `vllm/model_executor/models/bailing_moe.py` modified +5/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/fused_moe/__init__.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/AXK1.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #42280 - [Model] Fix missing `maybe_prefix`

- 链接: https://github.com/vllm-project/vllm/pull/42280
- 状态/时间: merged / 2026-05-11
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 25 个文件，+49/-29，可读 patch 302 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Fix missing `maybe_prefix`」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/arcee.py`, `vllm/model_executor/models/cohere_asr.py`, `vllm/model_executor/models/hunyuan_v1.py`；技术摘要: 覆盖「[Model] Fix missing `maybe_prefix`」；主要实现面是 `vllm/model_executor/models/arcee.py`, `vllm/model_executor/models/cohere_asr.py`, `vllm/model_executor/models/hunyuan_v1.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/arcee.py` modified +6/-2 (8 lines); hunks: -45,6 +45,7; -367,7 +368,10 @@ def __init__(self, *, vllm_config, prefix: str = "") -> None:; symbols: __init__，涉及 `__init__`；`vllm/model_executor/models/cohere_asr.py` modified +3/-2 (5 lines); hunks: -64,7 +64,7; -1717,7 +1717,8 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__，涉及 `__init__`；`vllm/model_executor/models/hunyuan_v1.py` modified +4/-1 (5 lines); hunks: -930,7 +930,10 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__，涉及 `__init__`；`vllm/model_executor/models/deepseek_eagle.py` modified +3/-1 (4 lines); hunks: -198,7 +198,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/arcee.py` modified +6/-2 (8 lines); hunks: -45,6 +45,7; -367,7 +368,10 @@ def __init__(self, *, vllm_config, prefix: str = "") -> None:; symbols: __init__
  - `vllm/model_executor/models/cohere_asr.py` modified +3/-2 (5 lines); hunks: -64,7 +64,7; -1717,7 +1717,8 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__
  - `vllm/model_executor/models/hunyuan_v1.py` modified +4/-1 (5 lines); hunks: -930,7 +930,10 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__
  - `vllm/model_executor/models/deepseek_eagle.py` modified +3/-1 (4 lines); hunks: -198,7 +198,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__
  - `vllm/model_executor/models/deepseek_eagle3.py` modified +3/-1 (4 lines); hunks: -318,7 +318,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/arcee.py
@@ -45,6 +45,7 @@
+    maybe_prefix,
@@ -367,7 +368,10 @@ def __init__(self, *, vllm_config, prefix: str = "") -> None:
-        self.model = ArceeModel(vllm_config=vllm_config, prefix=f"{prefix}.model")
+        self.model = ArceeModel(
+            vllm_config=vllm_config,
+            prefix=maybe_prefix(prefix, "model"),
diff -- vllm/model_executor/models/cohere_asr.py
@@ -64,7 +64,7 @@
-from .utils import AutoWeightsLoader, WeightsMapper, make_layers
+from .utils import AutoWeightsLoader, WeightsMapper, make_layers, maybe_prefix
@@ -1717,7 +1717,8 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
-            vllm_config=vllm_config, prefix=f"{prefix}.decoder"
+            vllm_config=vllm_config,
+            prefix=maybe_prefix(prefix, "decoder"),
diff -- vllm/model_executor/models/hunyuan_v1.py
@@ -930,7 +930,10 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
```

- 已读文件:
  - runtime: `vllm/model_executor/models/arcee.py` modified +6/-2; `vllm/model_executor/models/cohere_asr.py` modified +3/-2; `vllm/model_executor/models/hunyuan_v1.py` modified +4/-1; `vllm/model_executor/models/deepseek_eagle.py` modified +3/-1; `vllm/model_executor/models/deepseek_eagle3.py` modified +3/-1; `vllm/model_executor/models/granite_speech.py` modified +2/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/arcee.py`, `vllm/model_executor/models/aria.py`, `vllm/model_executor/models/blip2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #43167 - Remove KV cache scale boilerplate from model weight loading methods

- 链接: https://github.com/vllm-project/vllm/pull/43167
- 状态/时间: merged / 2026-06-05
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 56 个文件，+88/-731，可读 patch 1251 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Remove KV cache scale boilerplate from model weight loading methods」；模型线: Qwen3 Core；类别: 文档/测试/CI；主要 diff: `tests/model_executor/test_eagle_quantization.py`, `vllm/model_executor/models/gpt_oss.py`, `vllm/model_executor/layers/quantization/kv_cache.py`；技术摘要: 覆盖「Remove KV cache scale boilerplate from model weight loading methods」；主要实现面是 `tests/model_executor/test_eagle_quantization.py`, `vllm/model_executor/models/gpt_oss.py`, `vllm/model_executor/layers/quantization/kv_cache.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/model_executor/test_eagle_quantization.py` modified +0/-56 (56 lines); hunks: -100,32 +100,6 @@ def test_fc_layer_quant_config_usage(default_vllm_config, d...; -183,33 +157,3 @@ def test_eagle3_lm_head_receives_quant_config():; symbols: test_fc_layer_quant_config_usage, test_kv_cache_scale_name_handling, test_kv_cache_scale_name_no_scale, test_maybe_remap_kv_scale_name，涉及 `test_fc_layer_quant_config_usage, test_kv_cache_scale_name_handling, test_kv_cache_scale_name_no_scale`；`vllm/model_executor/models/gpt_oss.py` modified +0/-46 (46 lines); hunks: -635,52 +635,6 @@ def _get_moe_weight_dtype(layer_id: int = 0) -> str | None:; symbols: _get_moe_weight_dtype, kv_cache_scale_loader，涉及 `_get_moe_weight_dtype, kv_cache_scale_loader`；`vllm/model_executor/layers/quantization/kv_cache.py` modified +28/-4 (32 lines); hunks: -15,6 +15,30; -37,11 +61,11 @@ def create_weights(self, layer: torch.nn.Module):; symbols: KVCacheScaleParameter, __new__, weight_loader, BaseKVCacheMethod，涉及 `KVCacheScaleParameter, __new__, weight_loader`；`vllm/model_executor/layers/quantization/fp8.py` modified +12/-19 (31 lines); hunks: -207,25 +207,18 @@ def get_quant_method(; symbols: get_quant_method, get_cache_scale, get_cache_scale_mapper, CopyNumelCounter，涉及 `get_quant_method, get_cache_scale, get_cache_scale_mapper`。
- 代码 diff 细节:
  - `tests/model_executor/test_eagle_quantization.py` modified +0/-56 (56 lines); hunks: -100,32 +100,6 @@ def test_fc_layer_quant_config_usage(default_vllm_config, d...; -183,33 +157,3 @@ def test_eagle3_lm_head_receives_quant_config():; symbols: test_fc_layer_quant_config_usage, test_kv_cache_scale_name_handling, test_kv_cache_scale_name_no_scale, test_maybe_remap_kv_scale_name
  - `vllm/model_executor/models/gpt_oss.py` modified +0/-46 (46 lines); hunks: -635,52 +635,6 @@ def _get_moe_weight_dtype(layer_id: int = 0) -> str | None:; symbols: _get_moe_weight_dtype, kv_cache_scale_loader
  - `vllm/model_executor/layers/quantization/kv_cache.py` modified +28/-4 (32 lines); hunks: -15,6 +15,30; -37,11 +61,11 @@ def create_weights(self, layer: torch.nn.Module):; symbols: KVCacheScaleParameter, __new__, weight_loader, BaseKVCacheMethod
  - `vllm/model_executor/layers/quantization/fp8.py` modified +12/-19 (31 lines); hunks: -207,25 +207,18 @@ def get_quant_method(; symbols: get_quant_method, get_cache_scale, get_cache_scale_mapper, CopyNumelCounter
  - `vllm/model_executor/layers/quantization/quark/quark.py` modified +10/-20 (30 lines); hunks: -646,26 +646,16 @@ def get_scheme(; symbols: get_scheme, get_cache_scale, get_cache_scale_mapper, QuarkLinearMethod
- 关键代码摘录:

```diff
diff -- tests/model_executor/test_eagle_quantization.py
@@ -100,32 +100,6 @@ def test_fc_layer_quant_config_usage(default_vllm_config, dist_init, device) ->
-def test_kv_cache_scale_name_handling():
-    # Mock a quant config that supports cache scales
-    mock_quant_config = Mock()
-    mock_quant_config.get_cache_scale = Mock(return_value="layers.0.self_attn.kv_scale")
-    # Condition check in load_weights
-    name = "layers.0.self_attn.k_proj.weight"
diff -- vllm/model_executor/models/gpt_oss.py
@@ -635,52 +635,6 @@ def _get_moe_weight_dtype(layer_id: int = 0) -> str | None:
-            def kv_cache_scale_loader(
-                quant_config: QuantizationConfig,
-                name: str,
-                params_dict: dict[str, typing.Any],
-                weight: torch.Tensor,
-                default_weight_loader: Callable[..., None],
diff -- vllm/model_executor/layers/quantization/kv_cache.py
@@ -15,6 +15,30 @@
```

- 已读文件:
  - tests: `tests/model_executor/test_eagle_quantization.py` modified +0/-56
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +0/-46; `vllm/model_executor/layers/quantization/kv_cache.py` modified +28/-4; `vllm/model_executor/layers/quantization/fp8.py` modified +12/-19; `vllm/model_executor/layers/quantization/quark/quark.py` modified +10/-20; `vllm/model_executor/models/llama4.py` modified +3/-18; `vllm/model_executor/models/glm_ocr_mtp.py` modified +4/-13
- 验证与风险: diff 自带测试面 `tests/model_executor/test_eagle_quantization.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #39419 - [SpecDecode] Reduce TP communication for large-vocab draft models speculative decoding

- 链接: https://github.com/vllm-project/vllm/pull/39419
- 状态/时间: merged / 2026-06-10
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+53/-39，可读 patch 169 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[SpecDecode] Reduce TP communication for large-vocab draft models speculative decoding」；模型线: Qwen3 Core；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/interfaces.py`, `vllm/model_executor/models/llama4_eagle.py`, `vllm/model_executor/models/qwen3.py`；技术摘要: 覆盖「[SpecDecode] Reduce TP communication for large-vocab draft models speculative decoding」；主要实现面是 `vllm/model_executor/models/interfaces.py`, `vllm/model_executor/models/llama4_eagle.py`, `vllm/model_executor/models/qwen3.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/interfaces.py` modified +35/-0 (35 lines); hunks: -1282,6 +1282,41 @@ def supports_any_eagle(; symbols: supports_any_eagle, LocalArgmaxMixin, get_top_tokens, EagleModelMixin，涉及 `supports_any_eagle, LocalArgmaxMixin, get_top_tokens`；`vllm/model_executor/models/llama4_eagle.py` modified +0/-17 (17 lines); hunks: -208,23 +208,6 @@ def forward(; symbols: forward, get_top_tokens, load_weights, transform，涉及 `forward, get_top_tokens, load_weights`；`vllm/model_executor/models/qwen3.py` modified +8/-2 (10 lines); hunks: -48,7 +48,13; -259,7 +265,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, Qwen3ForCausalLM，涉及 `__init__, Qwen3ForCausalLM`；`vllm/model_executor/models/deepseek_eagle3.py` modified +2/-1 (3 lines); hunks: -31,6 +31,7; -309,7 +310,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights, Eagle3DeepseekV2ForCausalLM, __init__，涉及 `load_weights, Eagle3DeepseekV2ForCausalLM, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/interfaces.py` modified +35/-0 (35 lines); hunks: -1282,6 +1282,41 @@ def supports_any_eagle(; symbols: supports_any_eagle, LocalArgmaxMixin, get_top_tokens, EagleModelMixin
  - `vllm/model_executor/models/llama4_eagle.py` modified +0/-17 (17 lines); hunks: -208,23 +208,6 @@ def forward(; symbols: forward, get_top_tokens, load_weights, transform
  - `vllm/model_executor/models/qwen3.py` modified +8/-2 (10 lines); hunks: -48,7 +48,13; -259,7 +265,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, Qwen3ForCausalLM
  - `vllm/model_executor/models/deepseek_eagle3.py` modified +2/-1 (3 lines); hunks: -31,6 +31,7; -309,7 +310,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights, Eagle3DeepseekV2ForCausalLM, __init__
  - `vllm/model_executor/models/llama.py` modified +2/-1 (3 lines); hunks: -62,6 +62,7; -487,7 +488,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights, LlamaForCausalLM
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/interfaces.py
@@ -1282,6 +1282,41 @@ def supports_any_eagle(
+class LocalArgmaxMixin:
+    """Mixin for draft model heads in speculative decoding.
+    Provides a D2T-aware ``get_top_tokens`` that preserves the
+    local-argmax communication reduction even when the draft vocabulary
+    is smaller than the target vocabulary.
+    When ``draft_id_to_target_id`` is present (shape ``(draft_vocab_size,)``,
diff -- vllm/model_executor/models/llama4_eagle.py
@@ -208,23 +208,6 @@ def forward(
-    def get_top_tokens(
-        self,
-        hidden_states: torch.Tensor,
-    ) -> torch.Tensor:
-        """Vocab-parallel argmax without all-gathering full logits.
-        Falls back to full logits when draft_id_to_target_id remapping is
diff -- vllm/model_executor/models/qwen3.py
@@ -48,7 +48,13 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/models/interfaces.py` modified +35/-0; `vllm/model_executor/models/llama4_eagle.py` modified +0/-17; `vllm/model_executor/models/qwen3.py` modified +8/-2; `vllm/model_executor/models/deepseek_eagle3.py` modified +2/-1; `vllm/model_executor/models/llama.py` modified +2/-1; `vllm/model_executor/models/qwen3_5_mtp.py` modified +2/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_eagle3.py`, `vllm/model_executor/models/interfaces.py`, `vllm/model_executor/models/llama.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #45413 - [Frontend] Add Streaming Parser Engine and new Qwen3 Parser

- 链接: https://github.com/vllm-project/vllm/pull/45413
- 状态/时间: merged / 2026-06-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/parser/engine/test_qwen3.py`, `vllm/parser/qwen3.py`；关联提交 `c4a3f9d13709`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 33 个文件，+8492/-902，可读 patch 9786 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Frontend] Add Streaming Parser Engine and new Qwen3 Parser」；模型线: Qwen3 Core；类别: 文档/测试/CI；主要 diff: `tests/parser/engine/test_qwen3.py`, `vllm/parser/qwen3.py`；技术摘要: 覆盖「[Frontend] Add Streaming Parser Engine and new Qwen3 Parser」；主要实现面是 `tests/parser/engine/test_qwen3.py`, `vllm/parser/qwen3.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/parser/engine/test_qwen3.py` added +1095/-0 (1095 lines); hunks: -0,0 +1,1095; symbols: mock_tokenizer, parser, TestNonStreaming, test_no_tool_calls，涉及 `mock_tokenizer, parser, TestNonStreaming`；`vllm/parser/qwen3.py` added +218/-0 (218 lines); hunks: -0,0 +1,218; symbols: _qwen3_arg_converter, qwen3_config, Qwen3Parser, __init__，涉及 `_qwen3_arg_converter, qwen3_config, Qwen3Parser`。
- 代码 diff 细节:
  - `tests/parser/engine/test_qwen3.py` added +1095/-0 (1095 lines); hunks: -0,0 +1,1095; symbols: mock_tokenizer, parser, TestNonStreaming, test_no_tool_calls
  - `vllm/parser/qwen3.py` added +218/-0 (218 lines); hunks: -0,0 +1,218; symbols: _qwen3_arg_converter, qwen3_config, Qwen3Parser, __init__
- 关键代码摘录:

```diff
diff -- tests/parser/engine/test_qwen3.py
@@ -0,0 +1,1095 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Tests for the engine-based Qwen3 tool call parser.
+These validate that the engine-driven parser correctly handles
+Qwen3 XML-style tool calls.
+"""
diff -- vllm/parser/qwen3.py
@@ -0,0 +1,218 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Qwen3 parser for tool calls and reasoning.
+Qwen3 XML tool call format::
+    <tool_call>
+    <function=func_name>
```

- 已读文件:
  - tests: `tests/parser/engine/test_qwen3.py` added +1095/-0
  - runtime: `vllm/parser/qwen3.py` added +218/-0
- 验证与风险: diff 自带测试面 `tests/parser/engine/__init__.py`, `tests/parser/engine/conftest.py`, `tests/parser/engine/replay_harness.py`, `tests/parser/engine/streaming_helpers.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #45763 - [Bugfix] Fix Qwen3 prompt tool-call reasoning false positive

- 链接: https://github.com/vllm-project/vllm/pull/45763
- 状态/时间: merged / 2026-06-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/parser/qwen3.py`；关联提交 `7d567172fcb7`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+40/-0，可读 patch 78 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix Qwen3 prompt tool-call reasoning false positive」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `vllm/parser/qwen3.py`；技术摘要: 覆盖「[Bugfix] Fix Qwen3 prompt tool-call reasoning false positive」；主要实现面是 `vllm/parser/qwen3.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/parser/qwen3.py` modified +6/-0 (6 lines); hunks: -206,8 +206,14 @@ def is_reasoning_end(self, input_ids: list[int]) -> bool:; symbols: is_reasoning_end，涉及 `is_reasoning_end`。
- 代码 diff 细节:
  - `vllm/parser/qwen3.py` modified +6/-0 (6 lines); hunks: -206,8 +206,14 @@ def is_reasoning_end(self, input_ids: list[int]) -> bool:; symbols: is_reasoning_end
- 关键代码摘录:

```diff
diff -- vllm/parser/qwen3.py
@@ -206,8 +206,14 @@ def is_reasoning_end(self, input_ids: list[int]) -> bool:
+        reasoning_start_id = self._reasoning_start_token_id
+                if (
+                    reasoning_start_id is not None
+                    and input_ids[i] == reasoning_start_id
+                ):
+                    return False
```

- 已读文件:
  - runtime: `vllm/parser/qwen3.py` modified +6/-0
- 验证与风险: diff 自带测试面 `tests/parser/engine/test_qwen3_reasoning.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #46047 - [Bugfix] [Parser] Fix Qwen3 latent bug in partial params dropping values containing `<`

- 链接: https://github.com/vllm-project/vllm/pull/46047
- 状态/时间: merged / 2026-06-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/parser/engine/test_qwen3.py`, `vllm/parser/qwen3.py`；关联提交 `09f3cd5c1080`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+19/-1，可读 patch 34 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] [Parser] Fix Qwen3 latent bug in partial params dropping values containing `<`」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `tests/parser/engine/test_qwen3.py`, `vllm/parser/qwen3.py`；技术摘要: 覆盖「[Bugfix] [Parser] Fix Qwen3 latent bug in partial params dropping values containing `<`」；主要实现面是 `tests/parser/engine/test_qwen3.py`, `vllm/parser/qwen3.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/parser/engine/test_qwen3.py` modified +18/-0 (18 lines); hunks: -615,6 +615,24 @@ def test_partial_multiline(self):; symbols: test_partial_multiline, test_partial_value_with_angle_bracket, test_partial_value_with_angle_bracket_and_complete_param, TestSchemaAwareTypeCoercion，涉及 `test_partial_multiline, test_partial_value_with_angle_bracket, test_partial_value_with_angle_bracket_and_complete_param`；`vllm/parser/qwen3.py` modified +1/-1 (2 lines); hunks: -49,7 +49,7; symbols: _qwen3_arg_converter，涉及 `_qwen3_arg_converter`。
- 代码 diff 细节:
  - `tests/parser/engine/test_qwen3.py` modified +18/-0 (18 lines); hunks: -615,6 +615,24 @@ def test_partial_multiline(self):; symbols: test_partial_multiline, test_partial_value_with_angle_bracket, test_partial_value_with_angle_bracket_and_complete_param, TestSchemaAwareTypeCoercion
  - `vllm/parser/qwen3.py` modified +1/-1 (2 lines); hunks: -49,7 +49,7; symbols: _qwen3_arg_converter
- 关键代码摘录:

```diff
diff -- tests/parser/engine/test_qwen3.py
@@ -615,6 +615,24 @@ def test_partial_multiline(self):
+    def test_partial_value_with_angle_bracket(self):
+        from vllm.parser.qwen3 import (
+            _qwen3_arg_converter,
+        )
+        raw = "<parameter=expr>x<5"
+        result = json.loads(_qwen3_arg_converter(raw, partial=True))
diff -- vllm/parser/qwen3.py
@@ -49,7 +49,7 @@
-_PARTIAL_PARAM_RE = re.compile(r"<\s*parameter\s*=\s*([^>]+)>([^<]*)$", re.DOTALL)
+_PARTIAL_PARAM_RE = re.compile(r"<\s*parameter\s*=\s*([^>]+)>(.*)$", re.DOTALL)
```

- 已读文件:
  - tests: `tests/parser/engine/test_qwen3.py` modified +18/-0
  - runtime: `vllm/parser/qwen3.py` modified +1/-1
- 验证与风险: diff 自带测试面 `tests/parser/engine/test_qwen3.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #46351 - fix: stream Qwen3 tool call string arguments

- 链接: https://github.com/vllm-project/vllm/pull/46351
- 状态/时间: merged / 2026-06-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/parser/engine/test_qwen3.py`, `vllm/parser/qwen3.py`；关联提交 `8db12169a474`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+177/-9，可读 patch 326 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix: stream Qwen3 tool call string arguments」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `tests/parser/engine/test_qwen3.py`, `vllm/parser/qwen3.py`；技术摘要: 覆盖「fix: stream Qwen3 tool call string arguments」；主要实现面是 `tests/parser/engine/test_qwen3.py`, `vllm/parser/qwen3.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/parser/engine/test_qwen3.py` modified +65/-1 (66 lines); hunks: -346,6 +346,49 @@ def test_streaming_args_arrive_incrementally(self, parser,...; -395,6 +438,27 @@ def test_streaming_split_parameter_tag(self, parser, mock_r...; symbols: test_streaming_args_arrive_incrementally, test_streaming_long_string_arg_before_parameter_end, test_streaming_text_before_tool, test_streaming_split_parameter_tag，涉及 `test_streaming_args_arrive_incrementally, test_streaming_long_string_arg_before_parameter_end, test_streaming_text_before_tool`；`vllm/parser/qwen3.py` modified +13/-1 (14 lines); hunks: -42,6 +42,8; -67,7 +69,7 @@ def _qwen3_arg_converter(raw_args: str, partial: bool) -> str:; symbols: _qwen3_arg_converter, qwen3_config，涉及 `_qwen3_arg_converter, qwen3_config`。
- 代码 diff 细节:
  - `tests/parser/engine/test_qwen3.py` modified +65/-1 (66 lines); hunks: -346,6 +346,49 @@ def test_streaming_args_arrive_incrementally(self, parser,...; -395,6 +438,27 @@ def test_streaming_split_parameter_tag(self, parser, mock_r...; symbols: test_streaming_args_arrive_incrementally, test_streaming_long_string_arg_before_parameter_end, test_streaming_text_before_tool, test_streaming_split_parameter_tag
  - `vllm/parser/qwen3.py` modified +13/-1 (14 lines); hunks: -42,6 +42,8; -67,7 +69,7 @@ def _qwen3_arg_converter(raw_args: str, partial: bool) -> str:; symbols: _qwen3_arg_converter, qwen3_config
- 关键代码摘录:

```diff
diff -- tests/parser/engine/test_qwen3.py
@@ -346,6 +346,49 @@ def test_streaming_args_arrive_incrementally(self, parser, mock_request):
+    def test_streaming_long_string_arg_before_parameter_end(self, parser, mock_request):
+        """Long string arguments should stream before the closing parameter tag."""
+        chunks = [
+            "<tool_call>\n",
+            "<function=write_report>\n",
+            "<parameter=content>",
diff -- vllm/parser/qwen3.py
@@ -42,6 +42,8 @@
+PARAM_START = "<parameter="
+PARAM_END = "</parameter>"
@@ -67,7 +69,7 @@ def _qwen3_arg_converter(raw_args: str, partial: bool) -> str:
-                params[name] = value
+                params[name] = value.strip()
@@ -86,6 +88,8 @@ def qwen3_config(thinking: bool = True) -> ParserEngineConfig:
```

- 已读文件:
  - tests: `tests/parser/engine/test_qwen3.py` modified +65/-1
  - runtime: `vllm/parser/qwen3.py` modified +13/-1
- 验证与风险: diff 自带测试面 `tests/parser/engine/test_parser_engine.py`, `tests/parser/engine/test_qwen3.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #46314 - [Frontend] Port seed_oss to the streaming parser engine as a Qwen3 subclass

- 链接: https://github.com/vllm-project/vllm/pull/46314
- 状态/时间: merged / 2026-06-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/parser/qwen3.py`；关联提交 `cd347298e86c`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 13 个文件，+338/-1435，可读 patch 1879 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Frontend] Port seed_oss to the streaming parser engine as a Qwen3 subclass」；模型线: Qwen3 Core；类别: 模型实现调整；主要 diff: `vllm/parser/qwen3.py`；技术摘要: 覆盖「[Frontend] Port seed_oss to the streaming parser engine as a Qwen3 subclass」；主要实现面是 `vllm/parser/qwen3.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/parser/qwen3.py` modified +40/-13 (53 lines); hunks: -38,6 +38,8; -75,28 +77,36 @@ def _qwen3_arg_converter(raw_args: str, partial: bool) -> str:; symbols: _qwen3_arg_converter, qwen3_config, Qwen3Parser, attributes，涉及 `_qwen3_arg_converter, qwen3_config, Qwen3Parser`。
- 代码 diff 细节:
  - `vllm/parser/qwen3.py` modified +40/-13 (53 lines); hunks: -38,6 +38,8; -75,28 +77,36 @@ def _qwen3_arg_converter(raw_args: str, partial: bool) -> str:; symbols: _qwen3_arg_converter, qwen3_config, Qwen3Parser, attributes
- 关键代码摘录:

```diff
diff -- vllm/parser/qwen3.py
@@ -38,6 +38,8 @@
+THINK_START = "<think>"
+THINK_END = "</think>"
@@ -75,28 +77,36 @@ def _qwen3_arg_converter(raw_args: str, partial: bool) -> str:
-def qwen3_config(thinking: bool = True) -> ParserEngineConfig:
+def qwen3_config(
+    thinking: bool = True,
```

- 已读文件:
  - runtime: `vllm/parser/qwen3.py` modified +40/-13
- 验证与风险: diff 自带测试面 `tests/parser/engine/test_seed_oss.py`, `tests/parser/engine/trace_builder.py`, `tests/reasoning/test_seedoss_reasoning_parser.py`, `tests/tool_parsers/test_seed_oss_tool_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #48846 - [Bugfix][Tool Parser] Preserve whitespace in parameter values (MiniMax M2, Qwen3, MiniCPM5 XML)

- 链接: https://github.com/vllm-project/vllm/pull/48846
- 状态/时间: merged / 2026-07-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/parser/engine/test_qwen3.py`, `vllm/parser/qwen3.py`；关联提交 `11d291511a35`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+164/-9，可读 patch 249 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix][Tool Parser] Preserve whitespace in parameter values (MiniMax M2, Qwen3, MiniCPM5 XML)」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `vllm/parser/qwen3.py`, `tests/parser/engine/test_qwen3.py`；技术摘要: 覆盖「[Bugfix][Tool Parser] Preserve whitespace in parameter values (MiniMax M2, Qwen3, MiniCPM5 XML)」；主要实现面是 `vllm/parser/qwen3.py`, `tests/parser/engine/test_qwen3.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/parser/qwen3.py` modified +11/-2 (13 lines); hunks: -56,13 +56,22; -71,7 +80,7 @@ def _qwen3_arg_converter(raw_args: str, partial: bool) -> str:; symbols: _trim_wrapping_newlines, _qwen3_arg_converter，涉及 `_trim_wrapping_newlines, _qwen3_arg_converter`；`tests/parser/engine/test_qwen3.py` modified +2/-2 (4 lines); hunks: -454,10 +454,10 @@ def test_streaming_split_next_parameter_tag_is_buffered(se...; symbols: test_streaming_split_next_parameter_tag_is_buffered, test_streaming_numeric_values，涉及 `test_streaming_split_next_parameter_tag_is_buffered, test_streaming_numeric_values`。
- 代码 diff 细节:
  - `vllm/parser/qwen3.py` modified +11/-2 (13 lines); hunks: -56,13 +56,22; -71,7 +80,7 @@ def _qwen3_arg_converter(raw_args: str, partial: bool) -> str:; symbols: _trim_wrapping_newlines, _qwen3_arg_converter
  - `tests/parser/engine/test_qwen3.py` modified +2/-2 (4 lines); hunks: -454,10 +454,10 @@ def test_streaming_split_next_parameter_tag_is_buffered(se...; symbols: test_streaming_split_next_parameter_tag_is_buffered, test_streaming_numeric_values
- 关键代码摘录:

```diff
diff -- vllm/parser/qwen3.py
@@ -56,13 +56,22 @@
+def _trim_wrapping_newlines(value: str) -> str:
+    """Strip one leading and one trailing newline (the Qwen3 template markup)."""
+    if value.startswith("\n"):
+        value = value[1:]
+    if value.endswith("\n"):
+        value = value[:-1]
diff -- tests/parser/engine/test_qwen3.py
@@ -454,10 +454,10 @@ def test_streaming_split_next_parameter_tag_is_buffered(self, parser, mock_reque
-        assert args_after_partial_tag == '{"query": "hello'
+        assert args_after_partial_tag == '{"query": "hello '
-        assert json.loads(args_text) == {"query": "hello", "limit": "10"}
+        assert json.loads(args_text) == {"query": "hello ", "limit": "10"}
```

- 已读文件:
  - runtime: `vllm/parser/qwen3.py` modified +11/-2
  - tests: `tests/parser/engine/test_qwen3.py` modified +2/-2
- 验证与风险: diff 自带测试面 `tests/parser/engine/test_qwen3.py`, `tests/tool_parsers/test_minicpm5xml_tool_parser.py`, `tests/tool_parsers/test_minimax_m2_tool_parser.py`, `tests/tool_parsers/test_qwen3coder_tool_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
