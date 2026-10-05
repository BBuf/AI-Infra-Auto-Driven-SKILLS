# vLLM LongCat-Flash 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `tests/tool_parsers/test_longcat_tool_parser.py` | 无直接 PR 号提交 |
| `vllm/model_executor/models/longcat_flash.py` | [#23991](https://github.com/vllm-project/vllm/pull/23991), [#28891](https://github.com/vllm-project/vllm/pull/28891), [#41448](https://github.com/vllm-project/vllm/pull/41448), [#47857](https://github.com/vllm-project/vllm/pull/47857) |
| `vllm/model_executor/models/longcat_flash_mtp.py` | [#23991](https://github.com/vllm-project/vllm/pull/23991), [#47857](https://github.com/vllm-project/vllm/pull/47857) |
| `vllm/model_executor/models/longcat_flash_ngram.py` | [#47857](https://github.com/vllm-project/vllm/pull/47857) |
| `vllm/tool_parsers/longcat_tool_parser.py` | 无直接 PR 号提交 |

## PR 覆盖总览

- git 追溯 PR 数: 4
- 原文档显式引用补充 PR 数: 0
- 当前文档总 PR 数: 4
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-09-25 | [#23991](https://github.com/vllm-project/vllm/pull/23991) | merged | [Model] Add LongCat-Flash | `vllm/model_executor/models/longcat_flash.py`, `vllm/model_executor/models/longcat_flash_mtp.py` |
| 2025-12-20 | [#28891](https://github.com/vllm-project/vllm/pull/28891) | merged | [MoE Refactor][5/N] Isolate zero expert to LongCatFlash | `vllm/model_executor/models/longcat_flash.py` |
| 2026-05-01 | [#41448](https://github.com/vllm-project/vllm/pull/41448) | merged | Refractor longcat loading to use AutoWeightsLoader | `vllm/model_executor/models/longcat_flash.py` |
| 2026-07-10 | [#47857](https://github.com/vllm-project/vllm/pull/47857) | merged | [Model] Add LongCat-Flash-Lite (n-gram embedding) | `vllm/model_executor/models/longcat_flash_ngram.py`, `vllm/model_executor/models/longcat_flash_mtp.py`, `vllm/model_executor/models/longcat_flash.py` |

## 逐 PR diff 审计卡

### PR #23991 - [Model] Add LongCat-Flash

- 链接: https://github.com/vllm-project/vllm/pull/23991
- 状态/时间: merged / 2025-09-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/longcat_flash.py`, `vllm/model_executor/models/longcat_flash_mtp.py`；关联提交 `845adb3ec6d7`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 31 个文件，+1357/-66，可读 patch 2009 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/model_executor/models/longcat_flash.py` added +712/-0 (712 lines); hunks: -0,0 +1,712; symbols: FlashConfig, __init__, FlashMLP, forward，涉及 `FlashConfig, __init__, FlashMLP`；`vllm/model_executor/models/longcat_flash_mtp.py` added +352/-0 (352 lines); hunks: -0,0 +1,352; symbols: LongCatMultiTokenPredictorLayer, __init__, forward, LongCatMultiTokenPredictor，涉及 `LongCatMultiTokenPredictorLayer, __init__, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/longcat_flash.py` added +712/-0 (712 lines); hunks: -0,0 +1,712; symbols: FlashConfig, __init__, FlashMLP, forward
  - `vllm/model_executor/models/longcat_flash_mtp.py` added +352/-0 (352 lines); hunks: -0,0 +1,352; symbols: LongCatMultiTokenPredictorLayer, __init__, forward, LongCatMultiTokenPredictor
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/longcat_flash.py
@@ -0,0 +1,712 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Apache License, Version 2.0:
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
diff -- vllm/model_executor/models/longcat_flash_mtp.py
@@ -0,0 +1,352 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Adapted from
+# https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/model_executor/models/deepseek_mtp.py
+from collections.abc import Iterable
+from typing import Optional
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/model_executor/models/longcat_flash.py` added +712/-0; `vllm/model_executor/models/longcat_flash_mtp.py` added +352/-0
- 验证与风险: diff 自带测试面 `tests/kernels/moe/test_flashinfer.py`, `tests/models/registry.py`, `tests/models/utils.py`, `tests/test_routing_simulator.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #28891 - [MoE Refactor][5/N] Isolate zero expert to LongCatFlash

- 链接: https://github.com/vllm-project/vllm/pull/28891
- 状态/时间: merged / 2025-12-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/longcat_flash.py`；关联提交 `54c892438479`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 19 个文件，+263/-108，可读 patch 709 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/model_executor/models/longcat_flash.py` modified +37/-14 (51 lines); hunks: -46,7 +46,7; -179,7 +179,7 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/longcat_flash.py` modified +37/-14 (51 lines); hunks: -46,7 +46,7; -179,7 +179,7 @@ def __init__(; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/longcat_flash.py
@@ -46,7 +46,7 @@
-from vllm.model_executor.layers.fused_moe import FusedMoE
+from vllm.model_executor.layers.fused_moe import FusedMoE, ZeroExpertFusedMoE
@@ -179,7 +179,7 @@ def __init__(
-            else self.intermediate_size
+            else intermediate_size
@@ -280,48 +280,69 @@ def __init__(
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/model_executor/models/longcat_flash.py` modified +37/-14
- 验证与风险: diff 自带测试面 `tests/test_routing_simulator.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #41448 - Refractor longcat loading to use AutoWeightsLoader

- 链接: https://github.com/vllm-project/vllm/pull/41448
- 状态/时间: merged / 2026-05-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/longcat_flash.py`；关联提交 `0dbaf9daad20`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+82/-73，可读 patch 188 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/model_executor/models/longcat_flash.py` modified +82/-73 (155 lines); hunks: -69,6 +69,7; -485,6 +486,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, forward, LongcatFlashForCausalLM, embed_input_ids，涉及 `__init__, forward, LongcatFlashForCausalLM`。
- 代码 diff 细节:
  - `vllm/model_executor/models/longcat_flash.py` modified +82/-73 (155 lines); hunks: -69,6 +69,7; -485,6 +486,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, forward, LongcatFlashForCausalLM, embed_input_ids
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/longcat_flash.py
@@ -69,6 +69,7 @@
+    AutoWeightsLoader,
@@ -485,6 +486,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
+        self.quant_config = quant_config
@@ -551,77 +553,6 @@ def forward(
-class LongcatFlashForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
-    """Flash model for causal language modeling."""
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/model_executor/models/longcat_flash.py` modified +82/-73
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/longcat_flash.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #47857 - [Model] Add LongCat-Flash-Lite (n-gram embedding)

- 链接: https://github.com/vllm-project/vllm/pull/47857
- 状态/时间: merged / 2026-07-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/longcat_flash.py`, `vllm/model_executor/models/longcat_flash_mtp.py`, `vllm/model_executor/models/longcat_flash_ngram.py`；关联提交 `08dfd68610d2`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 16 个文件，+630/-12，可读 patch 782 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `vllm/model_executor/models/longcat_flash_ngram.py` added +405/-0 (405 lines); hunks: -0,0 +1,405; symbols: uses_ngram_embedding, _config_dtype, NgramEmbedding, __init__，涉及 `uses_ngram_embedding, _config_dtype, NgramEmbedding`；`vllm/model_executor/models/longcat_flash_mtp.py` modified +14/-4 (18 lines); hunks: -126,8 +126,10 @@ def forward(; -287,14 +289,22 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: forward, LongCatFlashMTP, __init__, load_weights，涉及 `forward, LongCatFlashMTP, __init__`；`vllm/model_executor/models/longcat_flash.py` modified +12/-3 (15 lines); hunks: -318,7 +318,7 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:; -687,14 +687,23 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: forward, load_weights，涉及 `forward, load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/models/longcat_flash_ngram.py` added +405/-0 (405 lines); hunks: -0,0 +1,405; symbols: uses_ngram_embedding, _config_dtype, NgramEmbedding, __init__
  - `vllm/model_executor/models/longcat_flash_mtp.py` modified +14/-4 (18 lines); hunks: -126,8 +126,10 @@ def forward(; -287,14 +289,22 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: forward, LongCatFlashMTP, __init__, load_weights
  - `vllm/model_executor/models/longcat_flash.py` modified +12/-3 (15 lines); hunks: -318,7 +318,7 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:; -687,14 +687,23 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: forward, load_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/longcat_flash_ngram.py
@@ -0,0 +1,405 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Inference-only LongCat-Flash-Lite (n-gram embedding) model.
+``LongcatFlashNgramForCausalLM`` is LongCat-Flash (MLA dual-attention +
+zero-expert MoE + YaRN) plus an n-gram embedding input layer: each position's
+embedding fuses the token embedding with hashed embeddings of the preceding
diff -- vllm/model_executor/models/longcat_flash_mtp.py
@@ -126,8 +126,10 @@ def forward(
-        # LongCat MTP without MoE layers
-        vllm_config.model_config.hf_config.n_routed_experts = None
+        # LongCat MTP has no MoE layers: clear n_routed_experts so the predictor
+        # builds a dense MLP. object.__setattr__ bypasses the ngram remote
+        # config's strict validation (it rejects setting the int field to None).
+        object.__setattr__(vllm_config.model_config.hf_config, "n_routed_experts", None)
diff -- vllm/model_executor/models/longcat_flash.py
@@ -318,7 +318,7 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
```

- 提取文件（未人工审阅）:
  - runtime: `vllm/model_executor/models/longcat_flash_ngram.py` added +405/-0; `vllm/model_executor/models/longcat_flash_mtp.py` modified +14/-4; `vllm/model_executor/models/longcat_flash.py` modified +12/-3
- 验证与风险: diff 自带测试面 `tests/models/registry.py`, `tests/models/utils.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
