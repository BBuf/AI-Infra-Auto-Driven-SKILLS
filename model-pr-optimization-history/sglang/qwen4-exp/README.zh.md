# SGLang Qwen4-Exp (Qwen3.8-Flash-Next) 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `python/sglang/srt/arg_groups/model_overrides/qwen4_exp.py` | [#39614](https://github.com/sgl-project/sglang/pull/39614) |
| `python/sglang/srt/configs/qwen4_exp.py` | 无直接 PR 号提交 |
| `python/sglang/srt/models/qwen4_exp.py` | [#38878](https://github.com/sgl-project/sglang/pull/38878), [#39928](https://github.com/sgl-project/sglang/pull/39928), [#40501](https://github.com/sgl-project/sglang/pull/40501), [#42478](https://github.com/sgl-project/sglang/pull/42478) |
| `python/sglang/srt/models/qwen4_exp_mtp.py` | 无直接 PR 号提交 |
| `python/sglang/srt/models/qwen4_exp_ple_table.py` | 无直接 PR 号提交 |
| `test/registered/e2e/models/test_qwen4_exp_models.py` | [#39662](https://github.com/sgl-project/sglang/pull/39662) |
| `test/registered/unit/layer_boundary/test_qwen4_exp_ple_rows.py` | [#42478](https://github.com/sgl-project/sglang/pull/42478) |
| `test/registered/unit/models/test_qwen4_exp_ple_table.py` | 无直接 PR 号提交 |

## PR 覆盖总览

- git 追溯 PR 数: 6
- 原文档显式引用补充 PR 数: 0
- 当前文档总 PR 数: 6
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-09-16 | [#39662](https://github.com/sgl-project/sglang/pull/39662) | merged | [qwen 3.8 next] change the testing model in test_qwen4_exp_models.py | `test/registered/e2e/models/test_qwen4_exp_models.py` |
| 2026-09-17 | [#38878](https://github.com/sgl-project/sglang/pull/38878) | merged | [AMD] Load fused shared experts for Qwen4-Exp and Qwen3.5 MTP | `python/sglang/srt/models/qwen4_exp.py` |
| 2026-09-20 | [#39928](https://github.com/sgl-project/sglang/pull/39928) | merged | [Qwen4-Exp] Build the offloaded PLE table on the meta device so --ple-offload-embedding never materialises it on the accelerator | `python/sglang/srt/models/qwen4_exp.py` |
| 2026-09-22 | [#40501](https://github.com/sgl-project/sglang/pull/40501) | merged | [Qwen3.8-Next] Pipeline-parallel serving and PD-prefill MTP for Qwen4-Exp | `python/sglang/srt/models/qwen4_exp.py` |
| 2026-09-30 | [#39614](https://github.com/sgl-project/sglang/pull/39614) | merged | [Qwen4-Exp] Optional fp8 (e4m3) storage for the compressed QSA indexer cache | `python/sglang/srt/arg_groups/model_overrides/qwen4_exp.py` |
| 2026-10-04 | [#42478](https://github.com/sgl-project/sglang/pull/42478) | merged | [Refactor] Build the Qwen4 experimental decoders from stage boundaries | `python/sglang/srt/models/qwen4_exp.py`, `test/registered/unit/layer_boundary/test_qwen4_exp_ple_rows.py` |

## 逐 PR diff 审计卡

### PR #39662 - [qwen 3.8 next] change the testing model in test_qwen4_exp_models.py

- 链接: https://github.com/sgl-project/sglang/pull/39662
- 状态/时间: merged / 2026-09-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/registered/e2e/models/test_qwen4_exp_models.py`；关联提交 `43390f63f57c`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `test/registered/e2e/models/test_qwen4_exp_models.py` modified +1/-1 (2 lines); hunks: -15,7 +15,7。
- 代码 diff 细节:
  - `test/registered/e2e/models/test_qwen4_exp_models.py` modified +1/-1 (2 lines); hunks: -15,7 +15,7
- 关键代码摘录:

```diff
diff -- test/registered/e2e/models/test_qwen4_exp_models.py
@@ -15,7 +15,7 @@
-MODEL = "RadixArk/Qwen3.8-Flash-Next-NVFP4"
+MODEL = "nvidia/Qwen3.8-Flash-Next-NVFP4"
```

- 提取文件（未人工审阅）:
  - tests: `test/registered/e2e/models/test_qwen4_exp_models.py` modified +1/-1
- 验证与风险: diff 自带测试面 `test/registered/e2e/models/test_qwen4_exp_models.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #38878 - [AMD] Load fused shared experts for Qwen4-Exp and Qwen3.5 MTP

- 链接: https://github.com/sgl-project/sglang/pull/38878
- 状态/时间: merged / 2026-09-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen4_exp.py`；关联提交 `11c35b8433e8`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+37/-10，可读 patch 96 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/sglang/srt/models/qwen4_exp.py` modified +24/-2 (26 lines); hunks: -61,7 +61,9; -1786,12 +1788,21 @@ def load_weights(self, weights: Iterable[Tuple[str, torc...; symbols: load_weights, load_qwen4_exp_ple_shard，涉及 `load_weights, load_qwen4_exp_ple_shard`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen4_exp.py` modified +24/-2 (26 lines); hunks: -61,7 +61,9; -1786,12 +1788,21 @@ def load_weights(self, weights: Iterable[Tuple[str, torc...; symbols: load_weights, load_qwen4_exp_ple_shard
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen4_exp.py
@@ -61,7 +61,9 @@
-from sglang.srt.utils import logger
+from sglang.srt.utils import get_bool_env_var, is_hip, logger
+_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and is_hip()
@@ -1786,12 +1788,21 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
+        # A fused shared expert lives in routed slot `num_experts`, so the
+        # mapping has to cover one more expert than the config declares.
```

- 提取文件（未人工审阅）:
  - runtime: `python/sglang/srt/models/qwen4_exp.py` modified +24/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/models/qwen4_exp.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #39928 - [Qwen4-Exp] Build the offloaded PLE table on the meta device so --ple-offload-embedding never materialises it on the accelerator

- 链接: https://github.com/sgl-project/sglang/pull/39928
- 状态/时间: merged / 2026-09-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen4_exp.py`；关联提交 `e97614d10c8e`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 2 个文件，+74/-18，可读 patch 136 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/sglang/srt/models/qwen4_exp.py` modified +26/-18 (44 lines); hunks: -508,20 +508,32 @@ def __init__(; -771,6 +783,8 @@ class Qwen4ExpPinnedHostEmbedding(VocabParallelEmbedding):; symbols: __init__, _splitmix64, Qwen4ExpPinnedHostEmbedding，涉及 `__init__, _splitmix64, Qwen4ExpPinnedHostEmbedding`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen4_exp.py` modified +26/-18 (44 lines); hunks: -508,20 +508,32 @@ def __init__(; -771,6 +783,8 @@ class Qwen4ExpPinnedHostEmbedding(VocabParallelEmbedding):; symbols: __init__, _splitmix64, Qwen4ExpPinnedHostEmbedding
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen4_exp.py
@@ -508,20 +508,32 @@ def __init__(
-        self.ngram_embedding = VocabParallelEmbedding(
-            padded_vocab_size,
-            self.head_dim_per_ngram,
-            params_dtype=(
-                torch.float8_e4m3fn
-                if _ple_table_is_fp8(config, quant_config, ngram_prefix)
```

- 提取文件（未人工审阅）:
  - runtime: `python/sglang/srt/models/qwen4_exp.py` modified +26/-18
- 验证与风险: diff 自带测试面 `test/registered/kernels/ops/embeddings/test_qwen4_ple_offload.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #40501 - [Qwen3.8-Next] Pipeline-parallel serving and PD-prefill MTP for Qwen4-Exp

- 链接: https://github.com/sgl-project/sglang/pull/40501
- 状态/时间: merged / 2026-09-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen4_exp.py`；关联提交 `6fd98c98b95f`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 8 个文件，+194/-43，可读 patch 424 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/sglang/srt/models/qwen4_exp.py` modified +75/-16 (91 lines); hunks: -2,7 +2,7; -46,9 +46,13; symbols: Qwen4ExpModel, _build_embed_tokens, __init__, forward，涉及 `Qwen4ExpModel, _build_embed_tokens, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen4_exp.py` modified +75/-16 (91 lines); hunks: -2,7 +2,7; -46,9 +46,13; symbols: Qwen4ExpModel, _build_embed_tokens, __init__, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen4_exp.py
@@ -2,7 +2,7 @@
-from typing import Any, Iterable, Optional, Set, Tuple
+from typing import Any, Iterable, Optional, Set, Tuple, Union
@@ -46,9 +46,13 @@
-from sglang.srt.layers.utils import get_layer_id
+from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
-from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
```

- 提取文件（未人工审阅）:
  - runtime: `python/sglang/srt/models/qwen4_exp.py` modified +75/-16
- 验证与风险: diff 自带测试面 `test/registered/unit/disaggregation/test_disaggregation_wire.py`, `test/registered/unit/server_args/test_server_args.py`, `test/registered/unit/spec/test_eagle_worker_v2_topk1_fastpath.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #39614 - [Qwen4-Exp] Optional fp8 (e4m3) storage for the compressed QSA indexer cache

- 链接: https://github.com/sgl-project/sglang/pull/39614
- 状态/时间: merged / 2026-09-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/arg_groups/model_overrides/qwen4_exp.py`；关联提交 `eb9c9ee99d47`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 13 个文件，+416/-54，可读 patch 937 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/sglang/srt/arg_groups/model_overrides/qwen4_exp.py` modified +11/-0 (11 lines); hunks: -75,6 +75,17 @@ def _qwen4_exp_overrides(server_args: Any, hf_config: Any) ->...; symbols: _qwen4_exp_overrides，涉及 `_qwen4_exp_overrides`。
- 代码 diff 细节:
  - `python/sglang/srt/arg_groups/model_overrides/qwen4_exp.py` modified +11/-0 (11 lines); hunks: -75,6 +75,17 @@ def _qwen4_exp_overrides(server_args: Any, hf_config: Any) ->...; symbols: _qwen4_exp_overrides
- 关键代码摘录:

```diff
diff -- python/sglang/srt/arg_groups/model_overrides/qwen4_exp.py
@@ -75,6 +75,17 @@ def _qwen4_exp_overrides(server_args: Any, hf_config: Any) -> dict:
+    if cfg.qsa_indexer_dtype == "fp8_e4m3":
+        # fp8 scoring runs on TileLang fp8 GEMMs, validated on Hopper and Blackwell.
+        platform = get_platform()
+        if not (platform.is_cuda and (platform.is_sm90 or platform.is_sm100)):
+            raise ValueError(
+                "--qsa-indexer-dtype fp8_e4m3 requires a CUDA SM90/SM100 GPU"
```

- 提取文件（未人工审阅）:
  - runtime: `python/sglang/srt/arg_groups/model_overrides/qwen4_exp.py` modified +11/-0
- 验证与风险: diff 自带测试面 `test/registered/kernels/ops/attention/qsa/test_qsa_indexer.py`, `test/registered/unit/disaggregation/test_disaggregation_wire.py`, `test/registered/unit/test_model_overrides.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #42478 - [Refactor] Build the Qwen4 experimental decoders from stage boundaries

- 链接: https://github.com/sgl-project/sglang/pull/42478
- 状态/时间: merged / 2026-10-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen4_exp.py`, `test/registered/unit/layer_boundary/test_qwen4_exp_ple_rows.py`；关联提交 `f07c1a398efd`
- 提取的 diff 范围（不是人工审计）: GitHub Pull Request files API 返回 10 个文件，+857/-307，可读 patch 1453 行；API patch 可能被截断或缺失，用作优化证据前须人工阅读完整 diff。
- 动机: 待人工核验；标题和文件清单仅供发现 PR，不构成已核验的动机。
- 实现变更清单（机器提取）: `python/sglang/srt/models/qwen4_exp.py` modified +176/-161 (337 lines); hunks: -26,24 +26,27; -70,6 +73,7; symbols: forward, Qwen4ExpLayerExtensionMixin, _MixStreams, __init__，涉及 `forward, Qwen4ExpLayerExtensionMixin, _MixStreams`；`test/registered/unit/layer_boundary/test_qwen4_exp_ple_rows.py` added +64/-0 (64 lines); hunks: -0,0 +1,64; symbols: _residual_ops, unused, _sliced_over_attention_tp, TestQwen4ExpPleRows，涉及 `_residual_ops, unused, _sliced_over_attention_tp`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen4_exp.py` modified +176/-161 (337 lines); hunks: -26,24 +26,27; -70,6 +73,7; symbols: forward, Qwen4ExpLayerExtensionMixin, _MixStreams, __init__
  - `test/registered/unit/layer_boundary/test_qwen4_exp_ple_rows.py` added +64/-0 (64 lines); hunks: -0,0 +1,64; symbols: _residual_ops, unused, _sliced_over_attention_tp, TestQwen4ExpPleRows
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen4_exp.py
@@ -26,24 +26,27 @@
-    attn_tp_all_gather,
-    get_dp_global_num_tokens,
-    get_global_dp_buffer,
-    get_local_dp_buffer,
-from sglang.srt.layers.layer_boundary import get_attn_tp_context
+from sglang.srt.layers.layer_boundary import (
diff -- test/registered/unit/layer_boundary/test_qwen4_exp_ple_rows.py
@@ -0,0 +1,64 @@
+"""A Qwen4-exp PLE layer reads its input on full attention rows."""
+import unittest
+from types import SimpleNamespace
+import test_declared_decoder_boundary as fixture
+from sglang.srt.layers.layer_boundary.contracts import BatchVariant
+from sglang.srt.layers.layer_boundary.layout import TokenAxis
```

- 提取文件（未人工审阅）:
  - runtime: `python/sglang/srt/models/qwen4_exp.py` modified +176/-161
  - tests: `test/registered/unit/layer_boundary/test_qwen4_exp_ple_rows.py` added +64/-0
- 验证与风险: diff 自带测试面 `test/registered/unit/layer_boundary/test_gated_residual_ops.py`, `test/registered/unit/layer_boundary/test_moe_output_reduction.py`, `test/registered/unit/layer_boundary/test_qwen4_exp_ple_rows.py`, `test/registered/unit/layer_boundary/test_stage_producers_leave_sums.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
