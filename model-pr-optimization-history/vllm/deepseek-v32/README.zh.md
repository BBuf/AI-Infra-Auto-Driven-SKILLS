# vllm DeepSeek V3.2 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `examples/online_serving/elastic_ep/serve_deepseek_v2.sh` | 无直接 PR 号提交 |
| `tests/evals/gsm8k/configs/DeepSeek-V3.2-DP.yaml` | [#33566](https://github.com/vllm-project/vllm/pull/33566) |
| `tests/evals/gsm8k/configs/DeepSeek-V3.2-DP_MI325.yaml` | 无直接 PR 号提交 |
| `tests/evals/gsm8k/configs/DeepSeek-V3.2-TP.yaml` | [#33566](https://github.com/vllm-project/vllm/pull/33566) |
| `tests/evals/gsm8k/configs/DeepSeek-V3.2-TP_MI325.yaml` | 无直接 PR 号提交 |
| `tests/tool_parsers/test_deepseekv32_tool_parser.py` | [#33703](https://github.com/vllm-project/vllm/pull/33703), [#36056](https://github.com/vllm-project/vllm/pull/36056), [#41198](https://github.com/vllm-project/vllm/pull/41198) |
| `vllm/model_executor/models/deepseek_mtp.py` | [#25896](https://github.com/vllm-project/vllm/pull/25896), [#38684](https://github.com/vllm-project/vllm/pull/38684), [#38870](https://github.com/vllm-project/vllm/pull/38870) |
| `vllm/model_executor/models/deepseek_v2.py` | [#25896](https://github.com/vllm-project/vllm/pull/25896), [#25999](https://github.com/vllm-project/vllm/pull/25999), [#26456](https://github.com/vllm-project/vllm/pull/26456), [#26465](https://github.com/vllm-project/vllm/pull/26465), [#26670](https://github.com/vllm-project/vllm/pull/26670), [#26763](https://github.com/vllm-project/vllm/pull/26763), [#27532](https://github.com/vllm-project/vllm/pull/27532), [#27568](https://github.com/vllm-project/vllm/pull/27568), [#28968](https://github.com/vllm-project/vllm/pull/28968), [#29287](https://github.com/vllm-project/vllm/pull/29287), [#30841](https://github.com/vllm-project/vllm/pull/30841), [#31046](https://github.com/vllm-project/vllm/pull/31046), ... (17 total) |
| `vllm/renderers/deepseek_v32.py` | [#33855](https://github.com/vllm-project/vllm/pull/33855) |
| `vllm/tokenizers/deepseek_v32.py` | [#30658](https://github.com/vllm-project/vllm/pull/30658), [#33855](https://github.com/vllm-project/vllm/pull/33855), [#37004](https://github.com/vllm-project/vllm/pull/37004) |
| `vllm/tokenizers/deepseek_v32_encoding.py` | [#29837](https://github.com/vllm-project/vllm/pull/29837), [#30025](https://github.com/vllm-project/vllm/pull/30025), [#31147](https://github.com/vllm-project/vllm/pull/31147), [#32884](https://github.com/vllm-project/vllm/pull/32884) |
| `vllm/tool_parsers/deepseekv32_tool_parser.py` | [#33703](https://github.com/vllm-project/vllm/pull/33703), [#33964](https://github.com/vllm-project/vllm/pull/33964), [#36056](https://github.com/vllm-project/vllm/pull/36056), [#41198](https://github.com/vllm-project/vllm/pull/41198) |

## PR 覆盖总览

- git 追溯 PR 数: 29
- 原文档显式引用补充 PR 数: 6
- 当前文档总 PR 数: 35
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-09-30 | [#25896](https://github.com/vllm-project/vllm/pull/25896) | merged | [New Model] DeepSeek-V3.2 (Rebased to Main) | `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py` |
| 2025-10-02 | [#25999](https://github.com/vllm-project/vllm/pull/25999) | merged | [Deepseek v3.2] Support indexer prefill chunking | `vllm/model_executor/models/deepseek_v2.py` |
| 2025-10-15 | [#26456](https://github.com/vllm-project/vllm/pull/26456) | merged | [Deepseek-V3.2][Kernel] Integrate cuda indexer k cache gather | `vllm/model_executor/models/deepseek_v2.py` |
| 2025-10-21 | [#26763](https://github.com/vllm-project/vllm/pull/26763) | merged | [Deepseek v3.2] Optimize top_k_per_row | `vllm/model_executor/models/deepseek_v2.py` |
| 2025-10-21 | [#26465](https://github.com/vllm-project/vllm/pull/26465) | merged | [Deepseek v3.2] Remove extra logics in indexer | `vllm/model_executor/models/deepseek_v2.py` |
| 2025-11-19 | [#28968](https://github.com/vllm-project/vllm/pull/28968) | merged | [DeepSeek] Fix DeepSeek V3.2 Rope Embedding | `vllm/model_executor/models/deepseek_v2.py` |
| 2025-11-20 | [#26670](https://github.com/vllm-project/vllm/pull/26670) | merged | [ROCm] Add AMD GPU support on Deepseek v3.2 and SparseMLA | `vllm/model_executor/models/deepseek_v2.py` |
| 2025-12-03 | [#29837](https://github.com/vllm-project/vllm/pull/29837) | merged | [Frontend] supports deepseekv32 chat template | `vllm/tokenizers/deepseek_v32_encoding.py` |
| 2025-12-04 | [#30025](https://github.com/vllm-project/vllm/pull/30025) | merged | [Bugfix] fixed deepseekv32 tool calling error | `vllm/tokenizers/deepseek_v32_encoding.py` |
| 2025-12-04 | [#29848](https://github.com/vllm-project/vllm/pull/29848) | merged | Add DeepSeek-V3.2 tool parser. | `vllm/entrypoints/openai/tool_parsers/deepseekv32_tool_parser.py`, `vllm/entrypoints/openai/tool_parsers/__init__.py` |
| 2025-12-08 | [#27568](https://github.com/vllm-project/vllm/pull/27568) | merged | [DeepSeek v3.2] Make top-k work for any logit values. | `vllm/model_executor/models/deepseek_v2.py` |
| 2025-12-12 | [#27532](https://github.com/vllm-project/vllm/pull/27532) | merged | [Attention] Use sparse prefill kernel for fp8 kv-cache in DeepSeek-v3.2 | `vllm/model_executor/models/deepseek_v2.py` |
| 2025-12-13 | [#30609](https://github.com/vllm-project/vllm/pull/30609) | merged | [Refactor] `TokenizerRegistry` only uses lazy imports | `vllm/tokenizers/registry.py`, `tests/tokenizers_/test_basic.py`, `vllm/tokenizers/deepseekv32.py` |
| 2025-12-15 | [#30658](https://github.com/vllm-project/vllm/pull/30658) | merged | [Bugfix] Fix deepseek_v32 tokenizer_mode | `vllm/tokenizers/deepseek_v32.py` |
| 2025-12-17 | [#30841](https://github.com/vllm-project/vllm/pull/30841) | merged | [Bugfix] deepseek-V3.2 self.weights_proj has no bias | `vllm/model_executor/models/deepseek_v2.py` |
| 2025-12-19 | [#31046](https://github.com/vllm-project/vllm/pull/31046) | merged | [Bug] Fix `error 'Dynamo failed to run FX node with fake tensors` for Deepseek V3.2 | `vllm/model_executor/models/deepseek_v2.py` |
| 2025-12-24 | [#31160](https://github.com/vllm-project/vllm/pull/31160) | merged | [Bug] Fix `Number of dimensions of tensors must match.` for Deepseek V3.2 | `vllm/model_executor/models/deepseek_v2.py` |
| 2026-01-05 | [#31147](https://github.com/vllm-project/vllm/pull/31147) | merged | Add chat prefix completion feature to DeepSeek v3.2 | `vllm/tokenizers/deepseek_v32_encoding.py` |
| 2026-01-16 | [#32175](https://github.com/vllm-project/vllm/pull/32175) | merged | [Bugfix] [DeepSeek-V3.2] fix sparse_attn_indexer padding | `vllm/model_executor/models/deepseek_v2.py` |
| 2026-01-21 | [#29287](https://github.com/vllm-project/vllm/pull/29287) | merged | [ROCm][Deepseekv3.2] Refactor Sparse Indexer as CustomOp | `vllm/model_executor/models/deepseek_v2.py` |
| 2026-01-22 | [#30207](https://github.com/vllm-project/vllm/pull/30207) | merged | Enable Cross layers KV cache layout at NIXL Connector | `tests/v1/kv_connector/unit/test_nixl_connector.py`, `vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py`, `vllm/distributed/kv_transfer/kv_connector/utils.py` |
| 2026-01-23 | [#32884](https://github.com/vllm-project/vllm/pull/32884) | merged | [BugFix] deepseek_v32_encoding: Replace asserts with proper exceptions | `vllm/tokenizers/deepseek_v32_encoding.py` |
| 2026-01-27 | [#33086](https://github.com/vllm-project/vllm/pull/33086) | closed | [Bugfix] Fix DeepseekV32 AssertionError: num_kv_heads == 1 | `vllm/v1/attention/backends/mla/indexer.py` |
| 2026-01-27 | [#33090](https://github.com/vllm-project/vllm/pull/33090) | merged | [Bugfix] Fix DeepseekV32 `AssertionError: num_kv_heads == 1` | `vllm/distributed/kv_transfer/kv_connector/utils.py` |
| 2026-02-02 | [#33566](https://github.com/vllm-project/vllm/pull/33566) | merged | [CI] Add DeepSeek V3.2 nightly eval | `tests/evals/gsm8k/configs/DeepSeek-V3.2-DP.yaml`, `tests/evals/gsm8k/configs/DeepSeek-V3.2-TP.yaml` |
| 2026-02-06 | [#33964](https://github.com/vllm-project/vllm/pull/33964) | merged | [Bugfix] Fix the issue where tool calling does not work when using fast detokenization with dsv32 | `vllm/tool_parsers/deepseekv32_tool_parser.py` |
| 2026-02-08 | [#33855](https://github.com/vllm-project/vllm/pull/33855) | merged | [Perf] Simplify DeepseekV32 tokenizer, ensure fast detokenization used | `vllm/tokenizers/deepseek_v32.py`, `vllm/renderers/deepseek_v32.py` |
| 2026-03-13 | [#37004](https://github.com/vllm-project/vllm/pull/37004) | merged | [Bugfix] Fix DeepSeek-V3.2 tokenizer stripping spaces | `vllm/tokenizers/deepseek_v32.py` |
| 2026-03-19 | [#36056](https://github.com/vllm-project/vllm/pull/36056) | merged | [Bugfix] Fix Deepseekv32 tool parser when stream interval > 1 | `vllm/tool_parsers/deepseekv32_tool_parser.py`, `tests/tool_parsers/test_deepseekv32_tool_parser.py` |
| 2026-03-30 | [#33703](https://github.com/vllm-project/vllm/pull/33703) | merged | [Bugfix] Support multi-type params parsing for DeepSeek v3.2 | `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py` |
| 2026-04-02 | [#38684](https://github.com/vllm-project/vllm/pull/38684) | merged | [Perf] DSV3.2 Indexer Fused Weights Projection | `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py` |
| 2026-04-04 | [#38870](https://github.com/vllm-project/vllm/pull/38870) | merged | [Bugfix] Fix DSV32 weight loading | `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py` |
| 2026-04-08 | [#37421](https://github.com/vllm-project/vllm/pull/37421) | merged | [Perf][Kernel] Persistent TopK scheduler: unified CUDAGraph-safe kernel with dynamic per-row dispatch - DeepSeek-V3.2 DSA decode | `vllm/model_executor/models/deepseek_v2.py` |
| 2026-04-27 | [#35968](https://github.com/vllm-project/vllm/pull/35968) | closed | [Performance] DeepSeek V3.2 multi-stream indexer overlap | `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/layers/layernorm.py`, `tests/utils_/test_indexer_dual_stream.py` |
| 2026-04-29 | [#41198](https://github.com/vllm-project/vllm/pull/41198) | merged | [Bugfix] DSV32/V4 add missing type conversion for non-streaming tool calls | `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py` |

## 逐 PR diff 审计卡

### PR #25896 - [New Model] DeepSeek-V3.2 (Rebased to Main)

- 链接: https://github.com/vllm-project/vllm/pull/25896
- 状态/时间: merged / 2025-09-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/deepseek_mtp.py`, `vllm/model_executor/models/deepseek_v2.py`；关联提交 `fa7e254a7f3e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 71 个文件，+3918/-221，可读 patch 5400 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[New Model] DeepSeek-V3.2 (Rebased to Main)」；模型线: DeepSeek V3.2；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`；技术摘要: 覆盖「[New Model] DeepSeek-V3.2 (Rebased to Main)」；主要实现面是 `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +445/-4 (449 lines); hunks: -33,36 +33,57; -276,6 +297,7 @@ class DeepseekV2Attention(nn.Module):; symbols: DeepseekV2MLP, DeepseekV2Attention, __init__，涉及 `DeepseekV2MLP, DeepseekV2Attention, __init__`；`vllm/model_executor/models/deepseek_mtp.py` modified +13/-1 (14 lines); hunks: -53,8 +53,20 @@ def __init__(self, vllm_config: VllmConfig, prefix: str) -> N...; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +445/-4 (449 lines); hunks: -33,36 +33,57; -276,6 +297,7 @@ class DeepseekV2Attention(nn.Module):; symbols: DeepseekV2MLP, DeepseekV2Attention, __init__
  - `vllm/model_executor/models/deepseek_mtp.py` modified +13/-1 (14 lines); hunks: -53,8 +53,20 @@ def __init__(self, vllm_config: VllmConfig, prefix: str) -> N...; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -33,36 +33,57 @@
+from vllm.attention.backends.abstract import AttentionBackend
+from vllm.attention.ops.common import pack_seq_triton, unpack_seq_triton
-from vllm.config import CacheConfig, ParallelConfig, VllmConfig
+from vllm.config import (CacheConfig, ParallelConfig, VllmConfig,
+                         get_current_vllm_config)
+from vllm.forward_context import get_forward_context
diff -- vllm/model_executor/models/deepseek_mtp.py
@@ -53,8 +53,20 @@ def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
+        self.is_v32 = hasattr(config, "index_topk")
+        if self.is_v32:
+            topk_tokens = config.index_topk
+            topk_indices_buffer = torch.empty(
+                vllm_config.scheduler_config.max_num_batched_tokens,
+                topk_tokens,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +445/-4; `vllm/model_executor/models/deepseek_mtp.py` modified +13/-1
- 验证与风险: diff 自带测试面 `tests/compile/test_fusion_attn.py`, `tests/kernels/attention/test_cache.py`, `tests/kernels/attention/test_deepgemm_attention.py`, `tests/kernels/attention/test_flashmla.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #25999 - [Deepseek v3.2] Support indexer prefill chunking

- 链接: https://github.com/vllm-project/vllm/pull/25999
- 状态/时间: merged / 2025-10-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/deepseek_v2.py`；关联提交 `1e50f1be7058`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+149/-79，可读 patch 324 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Deepseek v3.2] Support indexer prefill chunking」；模型线: DeepSeek V3.2；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/deepseek_v2.py`；技术摘要: 覆盖「[Deepseek v3.2] Support indexer prefill chunking」；主要实现面是 `vllm/model_executor/models/deepseek_v2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +37/-38 (75 lines); hunks: -583,44 +583,43 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer，涉及 `sparse_attn_indexer`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +37/-38 (75 lines); hunks: -583,44 +583,43 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -583,44 +583,43 @@ def sparse_attn_indexer(
-        num_prefills = attn_metadata.num_prefills
-        k_fp8 = torch.empty([prefill_metadata.total_seq_lens, head_dim],
-                            device=k.device,
-                            dtype=torch.float8_e4m3fn)
-        k_scale = torch.empty([prefill_metadata.total_seq_lens, 1],
-                              device=k.device,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +37/-38
- 验证与风险: diff 自带测试面 `tests/v1/attention/test_sparse_mla_backends.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #26456 - [Deepseek-V3.2][Kernel] Integrate cuda indexer k cache gather

- 链接: https://github.com/vllm-project/vllm/pull/26456
- 状态/时间: merged / 2025-10-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/deepseek_v2.py`；关联提交 `f5ed68ef63d0`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+6/-68，可读 patch 104 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Deepseek-V3.2][Kernel] Integrate cuda indexer k cache gather」；模型线: DeepSeek V3.2；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/deepseek_v2.py`；技术摘要: 覆盖「[Deepseek-V3.2][Kernel] Integrate cuda indexer k cache gather」；主要实现面是 `vllm/model_executor/models/deepseek_v2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +6/-68 (74 lines); hunks: -75,7 +75,7; -483,69 +483,6 @@ def get_attn_backend(self) -> AttentionBackend:; symbols: get_attn_backend, cp_gather_indexer_k_quant_cache, sparse_attn_indexer，涉及 `get_attn_backend, cp_gather_indexer_k_quant_cache, sparse_attn_indexer`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +6/-68 (74 lines); hunks: -75,7 +75,7; -483,69 +483,6 @@ def get_attn_backend(self) -> AttentionBackend:; symbols: get_attn_backend, cp_gather_indexer_k_quant_cache, sparse_attn_indexer
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -75,7 +75,7 @@
-from vllm.utils import cdiv, direct_register_custom_op
+from vllm.utils import direct_register_custom_op
@@ -483,69 +483,6 @@ def get_attn_backend(self) -> AttentionBackend:
-@torch.inference_mode()
-def cp_gather_indexer_k_quant_cache(
-    kv_cache,  # [num_blocks, block_size, head_dim + 1]
```

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +6/-68
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #26763 - [Deepseek v3.2] Optimize top_k_per_row

- 链接: https://github.com/vllm-project/vllm/pull/26763
- 状态/时间: merged / 2025-10-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/deepseek_v2.py`；关联提交 `80e94529845d`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+13/-49，可读 patch 203 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Deepseek v3.2] Optimize top_k_per_row」；模型线: DeepSeek V3.2；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/deepseek_v2.py`；技术摘要: 覆盖「[Deepseek v3.2] Optimize top_k_per_row」；主要实现面是 `vllm/model_executor/models/deepseek_v2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +0/-8 (8 lines); hunks: -577,15 +577,11 @@ def sparse_attn_indexer(; -642,15 +638,11 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer，涉及 `sparse_attn_indexer`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +0/-8 (8 lines); hunks: -577,15 +577,11 @@ def sparse_attn_indexer(; -642,15 +638,11 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -577,15 +577,11 @@ def sparse_attn_indexer(
-            topk_values = torch.empty(
-                num_rows, topk_tokens, dtype=logits.dtype, device=logits.device
-            )
-                topk_values,
@@ -642,15 +638,11 @@ def sparse_attn_indexer(
-        topk_values = torch.empty(
```

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +0/-8
- 验证与风险: diff 自带测试面 `tests/kernels/test_top_k_per_row.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #26465 - [Deepseek v3.2] Remove extra logics in indexer

- 链接: https://github.com/vllm-project/vllm/pull/26465
- 状态/时间: merged / 2025-10-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/deepseek_v2.py`；关联提交 `09a7e6f6179b`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+141/-40，可读 patch 272 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Deepseek v3.2] Remove extra logics in indexer」；模型线: DeepSeek V3.2；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/deepseek_v2.py`；技术摘要: 覆盖「[Deepseek v3.2] Remove extra logics in indexer」；主要实现面是 `vllm/model_executor/models/deepseek_v2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +11/-26 (37 lines); hunks: -574,9 +574,9 @@ def sparse_attn_indexer(; -586,9 +586,6 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer，涉及 `sparse_attn_indexer`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +11/-26 (37 lines); hunks: -574,9 +574,9 @@ def sparse_attn_indexer(; -586,9 +586,6 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -574,9 +574,9 @@ def sparse_attn_indexer(
-            topk_indices = torch.empty(
-                num_rows, topk_tokens, dtype=torch.int32, device=logits.device
-            )
+            topk_indices = topk_indices_buffer[
+                chunk.token_start : chunk.token_end, :topk_tokens
+            ]
```

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +11/-26
- 验证与风险: diff 自带测试面 `tests/kernels/test_top_k_per_row.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #28968 - [DeepSeek] Fix DeepSeek V3.2 Rope Embedding

- 链接: https://github.com/vllm-project/vllm/pull/28968
- 状态/时间: merged / 2025-11-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/deepseek_v2.py`；关联提交 `88f5b19f0bc6`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+17/-3，可读 patch 69 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DeepSeek] Fix DeepSeek V3.2 Rope Embedding」；模型线: DeepSeek V3.2；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/deepseek_v2.py`；技术摘要: 覆盖「[DeepSeek] Fix DeepSeek V3.2 Rope Embedding」；主要实现面是 `vllm/model_executor/models/deepseek_v2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +12/-2 (14 lines); hunks: -846,8 +846,8 @@ def forward(; -1000,6 +1000,14 @@ def __init__(; symbols: forward, __init__，涉及 `forward, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +12/-2 (14 lines); hunks: -846,8 +846,8 @@ def forward(; -1000,6 +1000,14 @@ def __init__(; symbols: forward, __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -846,8 +846,8 @@ def forward(
-        q = torch.cat([q_pe, q_nope], dim=-1)
-        k = torch.cat([k_pe.squeeze(1), k_nope], dim=-1)
+        q = torch.cat([q_pe.squeeze(0), q_nope], dim=-1)
+        k = torch.cat([k_pe.squeeze((0, 2)), k_nope], dim=-1)
@@ -1000,6 +1000,14 @@ def __init__(
+            self.indexer_rope_emb = get_rope(
```

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +12/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/mla.py`, `vllm/model_executor/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #26670 - [ROCm] Add AMD GPU support on Deepseek v3.2 and SparseMLA

- 链接: https://github.com/vllm-project/vllm/pull/26670
- 状态/时间: merged / 2025-11-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/deepseek_v2.py`；关联提交 `06c20c990464`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+583/-15，可读 patch 700 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[ROCm] Add AMD GPU support on Deepseek v3.2 and SparseMLA」；模型线: DeepSeek V3.2；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/deepseek_v2.py`；技术摘要: 覆盖「[ROCm] Add AMD GPU support on Deepseek v3.2 and SparseMLA」；主要实现面是 `vllm/model_executor/models/deepseek_v2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +18/-4 (22 lines); hunks: -591,6 +591,7 @@ def sparse_attn_indexer(; -630,7 +631,7 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer, sparse_attn_indexer_fake，涉及 `sparse_attn_indexer, sparse_attn_indexer_fake`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +18/-4 (22 lines); hunks: -591,6 +591,7 @@ def sparse_attn_indexer(; -630,7 +631,7 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer, sparse_attn_indexer_fake
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -591,6 +591,7 @@ def sparse_attn_indexer(
+    fp8_dtype = current_platform.fp8_dtype()
@@ -630,7 +631,7 @@ def sparse_attn_indexer(
-                dtype=torch.float8_e4m3fn,
+                dtype=fp8_dtype,
@@ -644,7 +645,12 @@ def sparse_attn_indexer(
-            logits = fp8_mqa_logits(
```

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +18/-4
- 验证与风险: runtime 路径改动集中在 `vllm/attention/ops/rocm_aiter_mla_sparse.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/platforms/rocm.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #29837 - [Frontend] supports deepseekv32 chat template

- 链接: https://github.com/vllm-project/vllm/pull/29837
- 状态/时间: merged / 2025-12-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/tokenizers/deepseek_v32_encoding.py`；关联提交 `b78772c43351`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+616/-2，可读 patch 660 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Frontend] supports deepseekv32 chat template」；模型线: DeepSeek V3.2；类别: 文档/测试/CI；主要 diff: `vllm/tokenizers/deepseek_v32_encoding.py`；技术摘要: 覆盖「[Frontend] supports deepseekv32 chat template」；主要实现面是 `vllm/tokenizers/deepseek_v32_encoding.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/tokenizers/deepseek_v32_encoding.py` added +456/-0 (456 lines); hunks: -0,0 +1,456; symbols: to_json, tools_from_openai_format, tool_calls_from_openai_format, tool_calls_to_openai_format，涉及 `to_json, tools_from_openai_format, tool_calls_from_openai_format`。
- 代码 diff 细节:
  - `vllm/tokenizers/deepseek_v32_encoding.py` added +456/-0 (456 lines); hunks: -0,0 +1,456; symbols: to_json, tools_from_openai_format, tool_calls_from_openai_format, tool_calls_to_openai_format
- 关键代码摘录:

```diff
diff -- vllm/tokenizers/deepseek_v32_encoding.py
@@ -0,0 +1,456 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# copy from https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/encoding/encoding_dsv32.py
+import copy
+import json
+import re
```

- 已读文件:
  - runtime: `vllm/tokenizers/deepseek_v32_encoding.py` added +456/-0
- 验证与风险: runtime 路径改动集中在 `vllm/config/model.py`, `vllm/entrypoints/openai/serving_engine.py`, `vllm/tokenizers/__init__.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #30025 - [Bugfix] fixed deepseekv32 tool calling error

- 链接: https://github.com/vllm-project/vllm/pull/30025
- 状态/时间: merged / 2025-12-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/tokenizers/deepseek_v32_encoding.py`；关联提交 `82a64b3d8f93`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+6/-3，可读 patch 23 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] fixed deepseekv32 tool calling error」；模型线: DeepSeek V3.2；类别: 缺陷修复；主要 diff: `vllm/tokenizers/deepseek_v32_encoding.py`；技术摘要: 覆盖「[Bugfix] fixed deepseekv32 tool calling error」；主要实现面是 `vllm/tokenizers/deepseek_v32_encoding.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/tokenizers/deepseek_v32_encoding.py` modified +4/-2 (6 lines); hunks: -95,8 +95,10 @@ def tool_calls_to_openai_format(tool_calls):; symbols: tool_calls_to_openai_format, encode_arguments_to_dsml，涉及 `tool_calls_to_openai_format, encode_arguments_to_dsml`。
- 代码 diff 细节:
  - `vllm/tokenizers/deepseek_v32_encoding.py` modified +4/-2 (6 lines); hunks: -95,8 +95,10 @@ def tool_calls_to_openai_format(tool_calls):; symbols: tool_calls_to_openai_format, encode_arguments_to_dsml
- 关键代码摘录:

```diff
diff -- vllm/tokenizers/deepseek_v32_encoding.py
@@ -95,8 +95,10 @@ def tool_calls_to_openai_format(tool_calls):
-    arguments = json.loads(tool_call["arguments"])
+    if isinstance(tool_call["arguments"], str):
+        arguments = json.loads(tool_call["arguments"])
+    else:
+        arguments = tool_call["arguments"]
```

- 已读文件:
  - runtime: `vllm/tokenizers/deepseek_v32_encoding.py` modified +4/-2
- 验证与风险: runtime 路径改动集中在 `vllm/tokenizers/deepseek_v32_encoding.py`, `vllm/tokenizers/deepseekv32.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #29848 - Add DeepSeek-V3.2 tool parser.

- 链接: https://github.com/vllm-project/vllm/pull/29848
- 状态/时间: merged / 2025-12-04
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+595/-0，可读 patch 603 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add DeepSeek-V3.2 tool parser.」；模型线: DeepSeek V3.2；类别: 文档/测试/CI；主要 diff: `vllm/entrypoints/openai/tool_parsers/deepseekv32_tool_parser.py`, `vllm/entrypoints/openai/tool_parsers/__init__.py`；技术摘要: 覆盖「Add DeepSeek-V3.2 tool parser.」；主要实现面是 `vllm/entrypoints/openai/tool_parsers/deepseekv32_tool_parser.py`, `vllm/entrypoints/openai/tool_parsers/__init__.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/entrypoints/openai/tool_parsers/deepseekv32_tool_parser.py` added +591/-0 (591 lines); hunks: -0,0 +1,591; symbols: DeepSeekV32ToolParser, __init__, type, _generate_tool_call_id，涉及 `DeepSeekV32ToolParser, __init__, type`；`vllm/entrypoints/openai/tool_parsers/__init__.py` modified +4/-0 (4 lines); hunks: -30,6 +30,10。
- 代码 diff 细节:
  - `vllm/entrypoints/openai/tool_parsers/deepseekv32_tool_parser.py` added +591/-0 (591 lines); hunks: -0,0 +1,591; symbols: DeepSeekV32ToolParser, __init__, type, _generate_tool_call_id
  - `vllm/entrypoints/openai/tool_parsers/__init__.py` modified +4/-0 (4 lines); hunks: -30,6 +30,10
- 关键代码摘录:

```diff
diff -- vllm/entrypoints/openai/tool_parsers/deepseekv32_tool_parser.py
@@ -0,0 +1,591 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import json
+import uuid
+from collections.abc import Sequence
+from typing import Any
diff -- vllm/entrypoints/openai/tool_parsers/__init__.py
@@ -30,6 +30,10 @@
+    "deepseek_v32": (
+        "deepseekv32_tool_parser",
+        "DeepSeekV32ToolParser",
+    ),
```

- 已读文件:
  - runtime: `vllm/entrypoints/openai/tool_parsers/deepseekv32_tool_parser.py` added +591/-0; `vllm/entrypoints/openai/tool_parsers/__init__.py` modified +4/-0
- 验证与风险: runtime 路径改动集中在 `vllm/entrypoints/openai/tool_parsers/__init__.py`, `vllm/entrypoints/openai/tool_parsers/deepseekv32_tool_parser.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #27568 - [DeepSeek v3.2] Make top-k work for any logit values.

- 链接: https://github.com/vllm-project/vllm/pull/27568
- 状态/时间: merged / 2025-12-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/deepseek_v2.py`；关联提交 `184076c3fecf`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+629/-210，可读 patch 1067 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[DeepSeek v3.2] Make top-k work for any logit values.」；模型线: DeepSeek V3.2；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/deepseek_v2.py`；技术摘要: 覆盖「[DeepSeek v3.2] Make top-k work for any logit values.」；主要实现面是 `vllm/model_executor/models/deepseek_v2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +3/-3 (6 lines); hunks: -684,18 +684,18 @@ def sparse_attn_indexer(; -738,7 +738,6 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer，涉及 `sparse_attn_indexer`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +3/-3 (6 lines); hunks: -684,18 +684,18 @@ def sparse_attn_indexer(; -738,7 +738,6 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -684,18 +684,18 @@ def sparse_attn_indexer(
-            assert topk_tokens == 2048, "top_k_per_row assumes size 2048"
-            torch.ops._C.top_k_per_row(
+            torch.ops._C.top_k_per_row_prefill(
+                topk_tokens,
@@ -738,7 +738,6 @@ def sparse_attn_indexer(
-        assert topk_tokens == 2048, "top_k_per_row assumes size 2048"
```

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +3/-3
- 验证与风险: diff 自带测试面 `tests/kernels/test_top_k_per_row.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #27532 - [Attention] Use sparse prefill kernel for fp8 kv-cache in DeepSeek-v3.2

- 链接: https://github.com/vllm-project/vllm/pull/27532
- 状态/时间: merged / 2025-12-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/deepseek_v2.py`；关联提交 `3e41992fecdc`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 30 个文件，+1372/-256，可读 patch 2323 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Attention] Use sparse prefill kernel for fp8 kv-cache in DeepSeek-v3.2」；模型线: DeepSeek V3.2；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/deepseek_v2.py`；技术摘要: 覆盖「[Attention] Use sparse prefill kernel for fp8 kv-cache in DeepSeek-v3.2」；主要实现面是 `vllm/model_executor/models/deepseek_v2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +18/-19 (37 lines); hunks: -83,6 +83,7; -618,8 +619,15 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer, sparse_attn_indexer_fake，涉及 `sparse_attn_indexer, sparse_attn_indexer_fake`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +18/-19 (37 lines); hunks: -83,6 +83,7; -618,8 +619,15 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer, sparse_attn_indexer_fake
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -83,6 +83,7 @@
+from vllm.v1.worker.workspace import current_workspace_manager
@@ -618,8 +619,15 @@ def sparse_attn_indexer(
+        # Reserve workspace for indexer during profiling run
+        current_workspace_manager().get_simultaneous(
+            ((total_seq_lens, head_dim), torch.float8_e4m3fn),
+            ((total_seq_lens, 4), torch.uint8),
```

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +18/-19
- 验证与风险: diff 自带测试面 `tests/conftest.py`, `tests/kernels/moe/test_batched_deepgemm.py`, `tests/kernels/moe/test_batched_moe.py`, `tests/kernels/moe/test_block_fp8.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #30609 - [Refactor] `TokenizerRegistry` only uses lazy imports

- 链接: https://github.com/vllm-project/vllm/pull/30609
- 状态/时间: merged / 2025-12-13
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 14 个文件，+202/-176，可读 patch 707 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Refactor] `TokenizerRegistry` only uses lazy imports」；模型线: DeepSeek V3.2；类别: 文档/测试/CI；主要 diff: `vllm/tokenizers/registry.py`, `tests/tokenizers_/test_basic.py`, `vllm/tokenizers/deepseekv32.py`；技术摘要: 覆盖「[Refactor] `TokenizerRegistry` only uses lazy imports」；主要实现面是 `vllm/tokenizers/registry.py`, `tests/tokenizers_/test_basic.py`, `vllm/tokenizers/deepseekv32.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/tokenizers/registry.py` modified +100/-100 (200 lines); hunks: -1,13 +1,13; -24,46 +24,25; symbols: TokenizerRegistry, register, _TokenizerRegistry，涉及 `TokenizerRegistry, register, _TokenizerRegistry`；`tests/tokenizers_/test_basic.py` modified +24/-23 (47 lines); hunks: -3,38 +3,39; symbols: _get_missing_attrs, _assert_tokenizer_like, test_tokenizer_like_protocol，涉及 `_get_missing_attrs, _assert_tokenizer_like, test_tokenizer_like_protocol`；`vllm/tokenizers/deepseekv32.py` modified +33/-14 (47 lines); hunks: -2,24 +2,18; -40,7 +34,21 @@ def from_pretrained(; symbols: DeepseekV32Tokenizer, __init__, from_pretrained，涉及 `DeepseekV32Tokenizer, __init__, from_pretrained`；`tests/tokenizers_/test_registry.py` modified +21/-2 (23 lines); hunks: -2,7 +2,14; -40,10 +47,22 @@ def is_fast(self) -> bool:; symbols: TestTokenizer, is_fast, test_resolve_tokenizer_args_idempotent, test_customized_tokenizer，涉及 `TestTokenizer, is_fast, test_resolve_tokenizer_args_idempotent`。
- 代码 diff 细节:
  - `vllm/tokenizers/registry.py` modified +100/-100 (200 lines); hunks: -1,13 +1,13; -24,46 +24,25; symbols: TokenizerRegistry, register, _TokenizerRegistry
  - `tests/tokenizers_/test_basic.py` modified +24/-23 (47 lines); hunks: -3,38 +3,39; symbols: _get_missing_attrs, _assert_tokenizer_like, test_tokenizer_like_protocol
  - `vllm/tokenizers/deepseekv32.py` modified +33/-14 (47 lines); hunks: -2,24 +2,18; -40,7 +34,21 @@ def from_pretrained(; symbols: DeepseekV32Tokenizer, __init__, from_pretrained
  - `tests/tokenizers_/test_registry.py` modified +21/-2 (23 lines); hunks: -2,7 +2,14; -40,10 +47,22 @@ def is_fast(self) -> bool:; symbols: TestTokenizer, is_fast, test_resolve_tokenizer_args_idempotent, test_customized_tokenizer
  - `vllm/tokenizers/hf.py` modified +7/-12 (19 lines); hunks: -3,22 +3,18; -65,11 +61,10 @@ def __reduce__(self):; symbols: get_cached_tokenizer, __reduce__, HfTokenizer, CachedHfTokenizer
- 关键代码摘录:

```diff
diff -- vllm/tokenizers/registry.py
@@ -1,13 +1,13 @@
-from collections.abc import Callable
+from dataclasses import dataclass, field
-from typing import TYPE_CHECKING, TypeVar, overload
+from typing import TYPE_CHECKING
-from typing_extensions import assert_never
+from typing_extensions import TypeVar, assert_never, deprecated
diff -- tests/tokenizers_/test_basic.py
@@ -3,38 +3,39 @@
-from transformers import PreTrainedTokenizerBase
+from transformers import (
+    PreTrainedTokenizer,
+    PreTrainedTokenizerBase,
+    PreTrainedTokenizerFast,
+)
diff -- vllm/tokenizers/deepseekv32.py
@@ -2,24 +2,18 @@
```

- 已读文件:
  - runtime: `vllm/tokenizers/registry.py` modified +100/-100; `vllm/tokenizers/deepseekv32.py` modified +33/-14; `vllm/tokenizers/hf.py` modified +7/-12; `vllm/tokenizers/mistral.py` modified +2/-5; `vllm/tokenizers/__init__.py` modified +0/-6; `vllm/transformers_utils/tokenizer.py` modified +3/-3
  - tests: `tests/tokenizers_/test_basic.py` modified +24/-23; `tests/tokenizers_/test_registry.py` modified +21/-2
- 验证与风险: diff 自带测试面 `tests/test_inputs.py`, `tests/tokenizers_/test_basic.py`, `tests/tokenizers_/test_registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #30658 - [Bugfix] Fix deepseek_v32 tokenizer_mode

- 链接: https://github.com/vllm-project/vllm/pull/30658
- 状态/时间: merged / 2025-12-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/tokenizers/deepseek_v32.py`；关联提交 `a524d1ba0af4`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+3/-3，可读 patch 27 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix deepseek_v32 tokenizer_mode」；模型线: DeepSeek V3.2；类别: 缺陷修复；主要 diff: `vllm/tokenizers/deepseek_v32.py`；技术摘要: 覆盖「[Bugfix] Fix deepseek_v32 tokenizer_mode」；主要实现面是 `vllm/tokenizers/deepseek_v32.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/tokenizers/deepseek_v32.py` renamed +0/-0 (0 lines)。
- 代码 diff 细节:
  - `vllm/tokenizers/deepseek_v32.py` renamed +0/-0 (0 lines)
- 关键代码摘录:

```diff
No textual patch was returned by GitHub for the selected changed files.
```

- 已读文件:
  - runtime: `vllm/tokenizers/deepseek_v32.py` renamed +0/-0
- 验证与风险: runtime 路径改动集中在 `vllm/entrypoints/openai/serving_engine.py`, `vllm/tokenizers/deepseek_v32.py`, `vllm/tokenizers/registry.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #30841 - [Bugfix] deepseek-V3.2 self.weights_proj has no bias

- 链接: https://github.com/vllm-project/vllm/pull/30841
- 状态/时间: merged / 2025-12-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/deepseek_v2.py`；关联提交 `84896fda22d3`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+5/-1，可读 patch 13 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] deepseek-V3.2 self.weights_proj has no bias」；模型线: DeepSeek V3.2；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/deepseek_v2.py`；技术摘要: 覆盖「[Bugfix] deepseek-V3.2 self.weights_proj has no bias」；主要实现面是 `vllm/model_executor/models/deepseek_v2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +5/-1 (6 lines); hunks: -835,7 +835,11 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +5/-1 (6 lines); hunks: -835,7 +835,11 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -835,7 +835,11 @@ def __init__(
-            hidden_size, self.n_head, quant_config=None, prefix=f"{prefix}.weights_proj"
+            hidden_size,
+            self.n_head,
+            bias=False,
+            quant_config=None,
+            prefix=f"{prefix}.weights_proj",
```

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +5/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #31046 - [Bug] Fix `error 'Dynamo failed to run FX node with fake tensors` for Deepseek V3.2

- 链接: https://github.com/vllm-project/vllm/pull/31046
- 状态/时间: merged / 2025-12-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/deepseek_v2.py`；关联提交 `4cf9429897c1`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+5/-2，可读 patch 14 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bug] Fix `error 'Dynamo failed to run FX node with fake tensors` for Deepseek V3.2」；模型线: DeepSeek V3.2；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/deepseek_v2.py`；技术摘要: 覆盖「[Bug] Fix `error 'Dynamo failed to run FX node with fake tensors` for Deepseek V3.2」；主要实现面是 `vllm/model_executor/models/deepseek_v2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +5/-2 (7 lines); hunks: -878,8 +878,11 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +5/-2 (7 lines); hunks: -878,8 +878,11 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -878,8 +878,11 @@ def forward(
-        q = torch.cat([q_pe.squeeze(0), q_nope], dim=-1)
-        k = torch.cat([k_pe.squeeze((0, 2)), k_nope], dim=-1)
+        # `rotary_emb` is shape-preserving; `q_pe` is already
+        # [num_tokens, n_head, rope_dim].
+        q = torch.cat([q_pe, q_nope], dim=-1)
+        # `k_pe` is [num_tokens, 1, rope_dim] (MQA).
```

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +5/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #31160 - [Bug] Fix `Number of dimensions of tensors must match.` for Deepseek V3.2

- 链接: https://github.com/vllm-project/vllm/pull/31160
- 状态/时间: merged / 2025-12-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/deepseek_v2.py`；关联提交 `76e6a951925b`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+6/-3，可读 patch 18 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bug] Fix `Number of dimensions of tensors must match.` for Deepseek V3.2」；模型线: DeepSeek V3.2；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/deepseek_v2.py`；技术摘要: 覆盖「[Bug] Fix `Number of dimensions of tensors must match.` for Deepseek V3.2」；主要实现面是 `vllm/model_executor/models/deepseek_v2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +6/-3 (9 lines); hunks: -878,11 +878,14 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +6/-3 (9 lines); hunks: -878,11 +878,14 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -878,11 +878,14 @@ def forward(
-        # `rotary_emb` is shape-preserving; `q_pe` is already
-        # [num_tokens, n_head, rope_dim].
+        # Note: RoPE (NeoX) can introduce extra leading dimensions during compilation
+        # so we need to reshape back to token-flattened shapes
+        q_pe = q_pe.reshape(-1, self.n_head, self.rope_dim)
+        k_pe = k_pe.reshape(-1, 1, self.rope_dim)
```

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +6/-3
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #31147 - Add chat prefix completion feature to DeepSeek v3.2

- 链接: https://github.com/vllm-project/vllm/pull/31147
- 状态/时间: merged / 2026-01-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/tokenizers/deepseek_v32_encoding.py`；关联提交 `346e56455a3b`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+9/-5，可读 patch 28 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add chat prefix completion feature to DeepSeek v3.2」；模型线: DeepSeek V3.2；类别: 缺陷修复；主要 diff: `vllm/tokenizers/deepseek_v32_encoding.py`；技术摘要: 覆盖「Add chat prefix completion feature to DeepSeek v3.2」；主要实现面是 `vllm/tokenizers/deepseek_v32_encoding.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/tokenizers/deepseek_v32_encoding.py` modified +9/-5 (14 lines); hunks: -169,6 +169,7 @@ def render_message(; -273,11 +274,14 @@ def render_message(; symbols: render_message，涉及 `render_message`。
- 代码 diff 细节:
  - `vllm/tokenizers/deepseek_v32_encoding.py` modified +9/-5 (14 lines); hunks: -169,6 +169,7 @@ def render_message(; -273,11 +274,14 @@ def render_message(; symbols: render_message
- 关键代码摘录:

```diff
diff -- vllm/tokenizers/deepseek_v32_encoding.py
@@ -169,6 +169,7 @@ def render_message(
+    is_prefix = msg.get("prefix", False)
@@ -273,11 +274,14 @@ def render_message(
-        prompt += assistant_msg_template.format(
-            reasoning=thinking_part,
-            content=summary_content,
-            tool_calls=tool_calls_content,
```

- 已读文件:
  - runtime: `vllm/tokenizers/deepseek_v32_encoding.py` modified +9/-5
- 验证与风险: runtime 路径改动集中在 `vllm/tokenizers/deepseek_v32_encoding.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #32175 - [Bugfix] [DeepSeek-V3.2] fix sparse_attn_indexer padding

- 链接: https://github.com/vllm-project/vllm/pull/32175
- 状态/时间: merged / 2026-01-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/deepseek_v2.py`；关联提交 `5de6dd0662da`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+9/-2，可读 patch 38 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] [DeepSeek-V3.2] fix sparse_attn_indexer padding」；模型线: DeepSeek V3.2；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/deepseek_v2.py`；技术摘要: 覆盖「[Bugfix] [DeepSeek-V3.2] fix sparse_attn_indexer padding」；主要实现面是 `vllm/model_executor/models/deepseek_v2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +9/-2 (11 lines); hunks: -717,13 +717,20 @@ def sparse_attn_indexer(; -739,14 +746,14 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer，涉及 `sparse_attn_indexer`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +9/-2 (11 lines); hunks: -717,13 +717,20 @@ def sparse_attn_indexer(; -739,14 +746,14 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -717,13 +717,20 @@ def sparse_attn_indexer(
+            # [num_decode_tokens, n_head, head_dim] -> [bs, 1+next_n, n_head, head_dim]
+            # [num_decode_tokens, n_head] -> [bs, 1+next_n, n_head]
+            padded_weights = pack_seq_triton(weights[:num_decode_tokens], decode_lens)
+            # [bs, 1+next_n, n_head] -> [bs * next_n, n_head]
+            padded_weights = padded_weights.flatten(0, 1)
+            padded_weights = weights
```

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +9/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #29287 - [ROCm][Deepseekv3.2] Refactor Sparse Indexer as CustomOp

- 链接: https://github.com/vllm-project/vllm/pull/29287
- 状态/时间: merged / 2026-01-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/deepseek_v2.py`；关联提交 `6c20e89c0209`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+982/-323，可读 patch 1521 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[ROCm][Deepseekv3.2] Refactor Sparse Indexer as CustomOp」；模型线: DeepSeek V3.2；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/deepseek_v2.py`；技术摘要: 覆盖「[ROCm][Deepseekv3.2] Refactor Sparse Indexer as CustomOp」；主要实现面是 `vllm/model_executor/models/deepseek_v2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +14/-233 (247 lines); hunks: -43,7 +43,6; -63,6 +62,7; symbols: get_attn_backend, sparse_attn_indexer, sparse_attn_indexer_fake, Indexer，涉及 `get_attn_backend, sparse_attn_indexer, sparse_attn_indexer_fake`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +14/-233 (247 lines); hunks: -43,7 +43,6; -63,6 +62,7; symbols: get_attn_backend, sparse_attn_indexer, sparse_attn_indexer_fake, Indexer
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -43,7 +43,6 @@
-from vllm.forward_context import get_forward_context
@@ -63,6 +62,7 @@
+from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
@@ -74,16 +74,11 @@
-from vllm.utils.deep_gemm import fp8_mqa_logits, fp8_paged_mqa_logits
-from vllm.utils.torch_utils import direct_register_custom_op
```

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +14/-233
- 验证与风险: runtime 路径改动集中在 `vllm/_aiter_ops.py`, `vllm/config/compilation.py`, `vllm/model_executor/layers/sparse_attn_indexer.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #30207 - Enable Cross layers KV cache layout at NIXL Connector

- 链接: https://github.com/vllm-project/vllm/pull/30207
- 状态/时间: merged / 2026-01-22
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+308/-89，可读 patch 729 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Enable Cross layers KV cache layout at NIXL Connector」；模型线: DeepSeek V3.2；类别: 性能/后端优化；主要 diff: `tests/v1/kv_connector/unit/test_nixl_connector.py`, `vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py`, `vllm/distributed/kv_transfer/kv_connector/utils.py`；技术摘要: 覆盖「Enable Cross layers KV cache layout at NIXL Connector」；主要实现面是 `tests/v1/kv_connector/unit/test_nixl_connector.py`, `vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py`, `vllm/distributed/kv_transfer/kv_connector/utils.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/v1/kv_connector/unit/test_nixl_connector.py` modified +178/-47 (225 lines); hunks: -18,8 +18,12; -48,8 +52,11; symbols: test_kv_transfer_handshake, __init__, _nixl_handshake, req_id，涉及 `test_kv_transfer_handshake, __init__, _nixl_handshake`；`vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py` modified +73/-38 (111 lines); hunks: -54,7 +54,7; -173,7 +173,7 @@ class NixlHandshakePayload(KVConnectorHandshakeMetadata):; symbols: NixlHandshakePayload, compute_nixl_compatibility_hash, add_new_req_to_recv, NixlConnector，涉及 `NixlHandshakePayload, compute_nixl_compatibility_hash, add_new_req_to_recv`；`vllm/distributed/kv_transfer/kv_connector/utils.py` modified +39/-2 (41 lines); hunks: -316,27 +316,56 @@ class TpKVTopology:; -346,6 +375,14 @@ def tp_size(self) -> int:; symbols: TpKVTopology, __post_init__, is_kv_layout_blocks_first, split_k_and_v，涉及 `TpKVTopology, __post_init__, is_kv_layout_blocks_first`；`tests/v1/kv_connector/nixl_integration/run_accuracy_test.sh` modified +9/-2 (11 lines); hunks: -34,11 +34,18 @@ else。
- 代码 diff 细节:
  - `tests/v1/kv_connector/unit/test_nixl_connector.py` modified +178/-47 (225 lines); hunks: -18,8 +18,12; -48,8 +52,11; symbols: test_kv_transfer_handshake, __init__, _nixl_handshake, req_id
  - `vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py` modified +73/-38 (111 lines); hunks: -54,7 +54,7; -173,7 +173,7 @@ class NixlHandshakePayload(KVConnectorHandshakeMetadata):; symbols: NixlHandshakePayload, compute_nixl_compatibility_hash, add_new_req_to_recv, NixlConnector
  - `vllm/distributed/kv_transfer/kv_connector/utils.py` modified +39/-2 (41 lines); hunks: -316,27 +316,56 @@ class TpKVTopology:; -346,6 +375,14 @@ def tp_size(self) -> int:; symbols: TpKVTopology, __post_init__, is_kv_layout_blocks_first, split_k_and_v
  - `tests/v1/kv_connector/nixl_integration/run_accuracy_test.sh` modified +9/-2 (11 lines); hunks: -34,11 +34,18 @@ else
  - `docs/features/nixl_connector_usage.md` modified +9/-0 (9 lines); hunks: -184,6 +184,15 @@ Support use case: Prefill with 'HND' and decode with 'NHD'...
- 关键代码摘录:

```diff
diff -- tests/v1/kv_connector/unit/test_nixl_connector.py
@@ -18,8 +18,12 @@
-from vllm.config import KVTransferConfig
-from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
+from vllm.config import KVTransferConfig, set_current_vllm_config
+from vllm.distributed.kv_transfer.kv_connector.utils import (
+    KVOutputAggregator,
+    TpKVTopology,
diff -- vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py
@@ -54,7 +54,7 @@
-from vllm.v1.attention.backend import AttentionMetadata
+from vllm.v1.attention.backend import AttentionBackend, AttentionMetadata
@@ -173,7 +173,7 @@ class NixlHandshakePayload(KVConnectorHandshakeMetadata):
-    vllm_config: VllmConfig, attn_backend_name: str
+    vllm_config: VllmConfig, attn_backend_name: str, cross_layers_blocks: bool
@@ -216,6 +216,7 @@ def compute_nixl_compatibility_hash(
diff -- vllm/distributed/kv_transfer/kv_connector/utils.py
@@ -316,27 +316,56 @@ class TpKVTopology:
```

- 已读文件:
  - tests: `tests/v1/kv_connector/unit/test_nixl_connector.py` modified +178/-47; `tests/v1/kv_connector/nixl_integration/run_accuracy_test.sh` modified +9/-2
  - runtime: `vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py` modified +73/-38; `vllm/distributed/kv_transfer/kv_connector/utils.py` modified +39/-2
  - docs: `docs/features/nixl_connector_usage.md` modified +9/-0
- 验证与风险: diff 自带测试面 `tests/v1/kv_connector/nixl_integration/run_accuracy_test.sh`, `tests/v1/kv_connector/unit/test_nixl_connector.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #32884 - [BugFix] deepseek_v32_encoding: Replace asserts with proper exceptions

- 链接: https://github.com/vllm-project/vllm/pull/32884
- 状态/时间: merged / 2026-01-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/tokenizers/deepseek_v32_encoding.py`；关联提交 `f61c9da711d8`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+39/-28，可读 patch 160 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[BugFix] deepseek_v32_encoding: Replace asserts with proper exceptions」；模型线: DeepSeek V3.2；类别: 缺陷修复；主要 diff: `vllm/tokenizers/deepseek_v32_encoding.py`；技术摘要: 覆盖「[BugFix] deepseek_v32_encoding: Replace asserts with proper exceptions」；主要实现面是 `vllm/tokenizers/deepseek_v32_encoding.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/tokenizers/deepseek_v32_encoding.py` modified +39/-28 (67 lines); hunks: -154,10 +154,12 @@ def find_last_user_index(messages: list[dict[str, Any]]) -...; -187,7 +189,8 @@ def render_message(; symbols: find_last_user_index, render_message，涉及 `find_last_user_index, render_message`。
- 代码 diff 细节:
  - `vllm/tokenizers/deepseek_v32_encoding.py` modified +39/-28 (67 lines); hunks: -154,10 +154,12 @@ def find_last_user_index(messages: list[dict[str, Any]]) -...; -187,7 +189,8 @@ def render_message(; symbols: find_last_user_index, render_message
- 关键代码摘录:

```diff
diff -- vllm/tokenizers/deepseek_v32_encoding.py
@@ -154,10 +154,12 @@ def find_last_user_index(messages: list[dict[str, Any]]) -> int:
-    assert 0 <= index < len(messages)
-    assert thinking_mode in ["chat", "thinking"], (
-        f"Invalid thinking_mode `{thinking_mode}`"
-    )
+    if not (0 <= index < len(messages)):
+        raise ValueError(
```

- 已读文件:
  - runtime: `vllm/tokenizers/deepseek_v32_encoding.py` modified +39/-28
- 验证与风险: runtime 路径改动集中在 `vllm/tokenizers/deepseek_v32_encoding.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #33086 - [Bugfix] Fix DeepseekV32 AssertionError: num_kv_heads == 1

- 链接: https://github.com/vllm-project/vllm/pull/33086
- 状态/时间: closed / 2026-01-27
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+0/-1，可读 patch 8 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix DeepseekV32 AssertionError: num_kv_heads == 1」；模型线: DeepSeek V3.2；类别: 缺陷修复；主要 diff: `vllm/v1/attention/backends/mla/indexer.py`；技术摘要: 覆盖「[Bugfix] Fix DeepseekV32 AssertionError: num_kv_heads == 1」；主要实现面是 `vllm/v1/attention/backends/mla/indexer.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/v1/attention/backends/mla/indexer.py` modified +0/-1 (1 lines); hunks: -49,7 +49,6 @@ def get_kv_cache_shape(; symbols: get_kv_cache_shape，涉及 `get_kv_cache_shape`。
- 代码 diff 细节:
  - `vllm/v1/attention/backends/mla/indexer.py` modified +0/-1 (1 lines); hunks: -49,7 +49,6 @@ def get_kv_cache_shape(; symbols: get_kv_cache_shape
- 关键代码摘录:

```diff
diff -- vllm/v1/attention/backends/mla/indexer.py
@@ -49,7 +49,6 @@ def get_kv_cache_shape(
-        assert num_kv_heads == 1
```

- 已读文件:
  - runtime: `vllm/v1/attention/backends/mla/indexer.py` modified +0/-1
- 验证与风险: runtime 路径改动集中在 `vllm/v1/attention/backends/mla/indexer.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #33090 - [Bugfix] Fix DeepseekV32 `AssertionError: num_kv_heads == 1`

- 链接: https://github.com/vllm-project/vllm/pull/33090
- 状态/时间: merged / 2026-01-27
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix DeepseekV32 `AssertionError: num_kv_heads == 1`」；模型线: DeepSeek V3.2；类别: 缺陷修复；主要 diff: `vllm/distributed/kv_transfer/kv_connector/utils.py`；技术摘要: 覆盖「[Bugfix] Fix DeepseekV32 `AssertionError: num_kv_heads == 1`」；主要实现面是 `vllm/distributed/kv_transfer/kv_connector/utils.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/distributed/kv_transfer/kv_connector/utils.py` modified +1/-1 (2 lines); hunks: -322,7 +322,7 @@ def __post_init__(self):; symbols: __post_init__，涉及 `__post_init__`。
- 代码 diff 细节:
  - `vllm/distributed/kv_transfer/kv_connector/utils.py` modified +1/-1 (2 lines); hunks: -322,7 +322,7 @@ def __post_init__(self):; symbols: __post_init__
- 关键代码摘录:

```diff
diff -- vllm/distributed/kv_transfer/kv_connector/utils.py
@@ -322,7 +322,7 @@ def __post_init__(self):
-            num_blocks=1, block_size=16, num_kv_heads=4, head_size=1
+            num_blocks=1, block_size=16, num_kv_heads=1, head_size=1
```

- 已读文件:
  - runtime: `vllm/distributed/kv_transfer/kv_connector/utils.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `vllm/distributed/kv_transfer/kv_connector/utils.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #33566 - [CI] Add DeepSeek V3.2 nightly eval

- 链接: https://github.com/vllm-project/vllm/pull/33566
- 状态/时间: merged / 2026-02-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/evals/gsm8k/configs/DeepSeek-V3.2-DP.yaml`, `tests/evals/gsm8k/configs/DeepSeek-V3.2-TP.yaml`；关联提交 `9f8cb81b44ce`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+24/-0，可读 patch 29 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI] Add DeepSeek V3.2 nightly eval」；模型线: DeepSeek V3.2；类别: 文档/测试/CI；主要 diff: `tests/evals/gsm8k/configs/DeepSeek-V3.2-DP.yaml`, `tests/evals/gsm8k/configs/DeepSeek-V3.2-TP.yaml`；技术摘要: 覆盖「[CI] Add DeepSeek V3.2 nightly eval」；主要实现面是 `tests/evals/gsm8k/configs/DeepSeek-V3.2-DP.yaml`, `tests/evals/gsm8k/configs/DeepSeek-V3.2-TP.yaml`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/evals/gsm8k/configs/DeepSeek-V3.2-DP.yaml` added +11/-0 (11 lines); hunks: -0,0 +1,11；`tests/evals/gsm8k/configs/DeepSeek-V3.2-TP.yaml` added +11/-0 (11 lines); hunks: -0,0 +1,11。
- 代码 diff 细节:
  - `tests/evals/gsm8k/configs/DeepSeek-V3.2-DP.yaml` added +11/-0 (11 lines); hunks: -0,0 +1,11
  - `tests/evals/gsm8k/configs/DeepSeek-V3.2-TP.yaml` added +11/-0 (11 lines); hunks: -0,0 +1,11
- 关键代码摘录:

```diff
diff -- tests/evals/gsm8k/configs/DeepSeek-V3.2-DP.yaml
@@ -0,0 +1,11 @@
+model_name: "deepseek-ai/DeepSeek-V3.2"
+accuracy_threshold: 0.95
+num_questions: 1319
+num_fewshot: 5
+startup_max_wait_seconds: 1200
+server_args: >-
diff -- tests/evals/gsm8k/configs/DeepSeek-V3.2-TP.yaml
@@ -0,0 +1,11 @@
+model_name: "deepseek-ai/DeepSeek-V3.2"
+accuracy_threshold: 0.95
+num_questions: 1319
+num_fewshot: 5
+startup_max_wait_seconds: 1200
+server_args: >-
```

- 已读文件:
  - tests: `tests/evals/gsm8k/configs/DeepSeek-V3.2-DP.yaml` added +11/-0; `tests/evals/gsm8k/configs/DeepSeek-V3.2-TP.yaml` added +11/-0
- 验证与风险: diff 自带测试面 `tests/evals/gsm8k/configs/DeepSeek-V3.2-DP.yaml`, `tests/evals/gsm8k/configs/DeepSeek-V3.2-TP.yaml`, `tests/evals/gsm8k/configs/models-h200.txt`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #33964 - [Bugfix] Fix the issue where tool calling does not work when using fast detokenization with dsv32

- 链接: https://github.com/vllm-project/vllm/pull/33964
- 状态/时间: merged / 2026-02-06
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/tool_parsers/deepseekv32_tool_parser.py`；关联提交 `7bec4351305f`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+12/-0，可读 patch 19 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix the issue where tool calling does not work when using fast detokenization with dsv32」；模型线: DeepSeek V3.2；类别: 缺陷修复；主要 diff: `vllm/tool_parsers/deepseekv32_tool_parser.py`；技术摘要: 覆盖「[Bugfix] Fix the issue where tool calling does not work when using fast detokenization with dsv32」；主要实现面是 `vllm/tool_parsers/deepseekv32_tool_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +12/-0 (12 lines); hunks: -110,6 +110,18 @@ def _generate_tool_call_id(self) -> str:; symbols: _generate_tool_call_id, adjust_request, _reset_streaming_state，涉及 `_generate_tool_call_id, adjust_request, _reset_streaming_state`。
- 代码 diff 细节:
  - `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +12/-0 (12 lines); hunks: -110,6 +110,18 @@ def _generate_tool_call_id(self) -> str:; symbols: _generate_tool_call_id, adjust_request, _reset_streaming_state
- 关键代码摘录:

```diff
diff -- vllm/tool_parsers/deepseekv32_tool_parser.py
@@ -110,6 +110,18 @@ def _generate_tool_call_id(self) -> str:
+    def adjust_request(self, request):
+        request = super().adjust_request(request)
+        if request.tools and request.tool_choice != "none":
+            # Ensure tool call tokens
+            # (<｜DSML｜function_calls>, </｜DSML｜function_calls>)
+            # are not skippedduring decoding.
```

- 已读文件:
  - runtime: `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +12/-0
- 验证与风险: runtime 路径改动集中在 `vllm/tool_parsers/deepseekv32_tool_parser.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #33855 - [Perf] Simplify DeepseekV32 tokenizer, ensure fast detokenization used

- 链接: https://github.com/vllm-project/vllm/pull/33855
- 状态/时间: merged / 2026-02-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/renderers/deepseek_v32.py`, `vllm/tokenizers/deepseek_v32.py`；关联提交 `a96197f564cb`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+88/-203，可读 patch 348 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Perf] Simplify DeepseekV32 tokenizer, ensure fast detokenization used」；模型线: DeepSeek V3.2；类别: 性能/后端优化；主要 diff: `vllm/tokenizers/deepseek_v32.py`, `vllm/renderers/deepseek_v32.py`；技术摘要: 覆盖「[Perf] Simplify DeepseekV32 tokenizer, ensure fast detokenization used」；主要实现面是 `vllm/tokenizers/deepseek_v32.py`, `vllm/renderers/deepseek_v32.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/tokenizers/deepseek_v32.py` modified +77/-179 (256 lines); hunks: -1,191 +1,89; symbols: DeepseekV32Tokenizer, from_pretrained, get_deepseek_v32_tokenizer, _DeepseekV32Tokenizer，涉及 `DeepseekV32Tokenizer, from_pretrained, get_deepseek_v32_tokenizer`；`vllm/renderers/deepseek_v32.py` modified +3/-2 (5 lines); hunks: -13,6 +13,7; -48,10 +49,10 @@ def __init__(; symbols: __init__, tokenizer, get_tokenizer，涉及 `__init__, tokenizer, get_tokenizer`。
- 代码 diff 细节:
  - `vllm/tokenizers/deepseek_v32.py` modified +77/-179 (256 lines); hunks: -1,191 +1,89; symbols: DeepseekV32Tokenizer, from_pretrained, get_deepseek_v32_tokenizer, _DeepseekV32Tokenizer
  - `vllm/renderers/deepseek_v32.py` modified +3/-2 (5 lines); hunks: -13,6 +13,7; -48,10 +49,10 @@ def __init__(; symbols: __init__, tokenizer, get_tokenizer
- 关键代码摘录:

```diff
diff -- vllm/tokenizers/deepseek_v32.py
@@ -1,191 +1,89 @@
+import copy
+from typing import Any
-from pathlib import Path
-from typing import Any, overload
-from transformers import BatchEncoding
+from transformers import AutoTokenizer
diff -- vllm/renderers/deepseek_v32.py
@@ -13,6 +13,7 @@
+from ..tokenizers.hf import HfTokenizer
@@ -48,10 +49,10 @@ def __init__(
-    def tokenizer(self) -> DeepseekV32Tokenizer | None:
+    def tokenizer(self) -> HfTokenizer | None:
-    def get_tokenizer(self) -> DeepseekV32Tokenizer:
+    def get_tokenizer(self) -> HfTokenizer:
```

- 已读文件:
  - runtime: `vllm/tokenizers/deepseek_v32.py` modified +77/-179; `vllm/renderers/deepseek_v32.py` modified +3/-2
- 验证与风险: diff 自带测试面 `tests/tokenizers_/test_basic.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #37004 - [Bugfix] Fix DeepSeek-V3.2 tokenizer stripping spaces

- 链接: https://github.com/vllm-project/vllm/pull/37004
- 状态/时间: merged / 2026-03-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/tokenizers/deepseek_v32.py`；关联提交 `9efc4db9658a`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+4/-2，可读 patch 25 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix DeepSeek-V3.2 tokenizer stripping spaces」；模型线: DeepSeek V3.2；类别: 缺陷修复；主要 diff: `vllm/tokenizers/deepseek_v32.py`；技术摘要: 覆盖「[Bugfix] Fix DeepSeek-V3.2 tokenizer stripping spaces」；主要实现面是 `vllm/tokenizers/deepseek_v32.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/tokenizers/deepseek_v32.py` modified +2/-2 (4 lines); hunks: -3,7 +3,7; -85,5 +85,5 @@ def __reduce__(self):; symbols: __reduce__, DeepseekV32Tokenizer, from_pretrained，涉及 `__reduce__, DeepseekV32Tokenizer, from_pretrained`。
- 代码 diff 细节:
  - `vllm/tokenizers/deepseek_v32.py` modified +2/-2 (4 lines); hunks: -3,7 +3,7; -85,5 +85,5 @@ def __reduce__(self):; symbols: __reduce__, DeepseekV32Tokenizer, from_pretrained
- 关键代码摘录:

```diff
diff -- vllm/tokenizers/deepseek_v32.py
@@ -3,7 +3,7 @@
-from transformers import AutoTokenizer
+from transformers import PreTrainedTokenizerFast
@@ -85,5 +85,5 @@ def __reduce__(self):
-        tokenizer = AutoTokenizer.from_pretrained(*args, **kwargs)
+        tokenizer = PreTrainedTokenizerFast.from_pretrained(*args, **kwargs)
```

- 已读文件:
  - runtime: `vllm/tokenizers/deepseek_v32.py` modified +2/-2
- 验证与风险: runtime 路径改动集中在 `vllm/config/model.py`, `vllm/tokenizers/deepseek_v32.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #36056 - [Bugfix] Fix Deepseekv32 tool parser when stream interval > 1

- 链接: https://github.com/vllm-project/vllm/pull/36056
- 状态/时间: merged / 2026-03-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py`；关联提交 `be12afd284f3`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+622/-437，可读 patch 1113 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix Deepseekv32 tool parser when stream interval > 1」；模型线: DeepSeek V3.2；类别: 缺陷修复；主要 diff: `vllm/tool_parsers/deepseekv32_tool_parser.py`, `tests/tool_parsers/test_deepseekv32_tool_parser.py`；技术摘要: 覆盖「[Bugfix] Fix Deepseekv32 tool parser when stream interval > 1」；主要实现面是 `vllm/tool_parsers/deepseekv32_tool_parser.py`, `tests/tool_parsers/test_deepseekv32_tool_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +146/-437 (583 lines); hunks: -48,41 +48,12 @@ def __init__(self, tokenizer: TokenizerLike):; -106,10 +77,6 @@ def __init__(self, tokenizer: TokenizerLike):; symbols: __init__, type, _generate_tool_call_id, adjust_request，涉及 `__init__, type, _generate_tool_call_id`；`tests/tool_parsers/test_deepseekv32_tool_parser.py` added +476/-0 (476 lines); hunks: -0,0 +1,476; symbols: make_parser, make_tool_param, make_request, build_tool_call，涉及 `make_parser, make_tool_param, make_request`。
- 代码 diff 细节:
  - `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +146/-437 (583 lines); hunks: -48,41 +48,12 @@ def __init__(self, tokenizer: TokenizerLike):; -106,10 +77,6 @@ def __init__(self, tokenizer: TokenizerLike):; symbols: __init__, type, _generate_tool_call_id, adjust_request
  - `tests/tool_parsers/test_deepseekv32_tool_parser.py` added +476/-0 (476 lines); hunks: -0,0 +1,476; symbols: make_parser, make_tool_param, make_request, build_tool_call
- 关键代码摘录:

```diff
diff -- vllm/tool_parsers/deepseekv32_tool_parser.py
@@ -48,41 +48,12 @@ def __init__(self, tokenizer: TokenizerLike):
-        # Sentinel tokens
-        self.dsml_token: str = "｜DSML｜"
-        self.dsml_start_check: str = "<" + self.dsml_token
+        # Sentinel token
-        self.tool_call_end_token: str = "</｜DSML｜function_calls>"
-        self.invoke_start_prefix: str = "<｜DSML｜invoke name="
diff -- tests/tool_parsers/test_deepseekv32_tool_parser.py
@@ -0,0 +1,476 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Unit tests for DeepSeekV32ToolParser.
+These tests use a minimal mock tokenizer so no real model weights are required.
+"""
+import json
```

- 已读文件:
  - runtime: `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +146/-437
  - tests: `tests/tool_parsers/test_deepseekv32_tool_parser.py` added +476/-0
- 验证与风险: diff 自带测试面 `tests/tool_parsers/test_deepseekv32_tool_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #33703 - [Bugfix] Support multi-type params parsing for DeepSeek v3.2

- 链接: https://github.com/vllm-project/vllm/pull/33703
- 状态/时间: merged / 2026-03-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py`；关联提交 `a6db99ba02ec`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+201/-18，可读 patch 250 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Support multi-type params parsing for DeepSeek v3.2」；模型线: DeepSeek V3.2；类别: 缺陷修复；主要 diff: `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py`；技术摘要: 覆盖「[Bugfix] Support multi-type params parsing for DeepSeek v3.2」；主要实现面是 `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +181/-0 (181 lines); hunks: -11,6 +11,7; -474,3 +475,183 @@ def test_no_emission_while_incomplete(self, parser):; symbols: test_no_emission_while_incomplete, deepseekv32_tokenizer, parser, test_convert_param_value_single_types，涉及 `test_no_emission_while_incomplete, deepseekv32_tokenizer, parser`；`vllm/tool_parsers/deepseekv32_tool_parser.py` modified +20/-18 (38 lines); hunks: -100,7 +100,7 @@ def _parse_invoke_params(self, invoke_str: str) -> dict:; -109,29 +109,31 @@ def _convert_param_value(self, value: str, param_type: str...; symbols: _parse_invoke_params, _convert_param_value, _convert_param_value_checked，涉及 `_parse_invoke_params, _convert_param_value, _convert_param_value_checked`。
- 代码 diff 细节:
  - `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +181/-0 (181 lines); hunks: -11,6 +11,7; -474,3 +475,183 @@ def test_no_emission_while_incomplete(self, parser):; symbols: test_no_emission_while_incomplete, deepseekv32_tokenizer, parser, test_convert_param_value_single_types
  - `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +20/-18 (38 lines); hunks: -100,7 +100,7 @@ def _parse_invoke_params(self, invoke_str: str) -> dict:; -109,29 +109,31 @@ def _convert_param_value(self, value: str, param_type: str...; symbols: _parse_invoke_params, _convert_param_value, _convert_param_value_checked
- 关键代码摘录:

```diff
diff -- tests/tool_parsers/test_deepseekv32_tool_parser.py
@@ -11,6 +11,7 @@
+from vllm.tokenizers import get_tokenizer
@@ -474,3 +475,183 @@ def test_no_emission_while_incomplete(self, parser):
+@pytest.fixture(scope="module")
+def deepseekv32_tokenizer():
+    return get_tokenizer(tokenizer_name="deepseek-ai/DeepSeek-V3.2")
+@pytest.fixture
diff -- vllm/tool_parsers/deepseekv32_tool_parser.py
@@ -100,7 +100,7 @@ def _parse_invoke_params(self, invoke_str: str) -> dict:
-    def _convert_param_value(self, value: str, param_type: str) -> Any:
+    def _convert_param_value_checked(self, value: str, param_type: str) -> Any:
@@ -109,29 +109,31 @@ def _convert_param_value(self, value: str, param_type: str) -> Any:
-            try:
-                return int(value)
-            except (ValueError, TypeError):
```

- 已读文件:
  - tests: `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +181/-0
  - runtime: `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +20/-18
- 验证与风险: diff 自带测试面 `tests/tool_parsers/test_deepseekv32_tool_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #38684 - [Perf] DSV3.2 Indexer Fused Weights Projection

- 链接: https://github.com/vllm-project/vllm/pull/38684
- 状态/时间: merged / 2026-04-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/deepseek_mtp.py`, `vllm/model_executor/models/deepseek_v2.py`；关联提交 `5f96f9aff10f`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+25/-14，可读 patch 79 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Perf] DSV3.2 Indexer Fused Weights Projection」；模型线: DeepSeek V3.2；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`；技术摘要: 覆盖「[Perf] DSV3.2 Indexer Fused Weights Projection」；主要实现面是 `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +22/-14 (36 lines); hunks: -639,21 +639,19 @@ def __init__(; -694,7 +692,11 @@ def forward(; symbols: __init__, forward, load_weights，涉及 `__init__, forward, load_weights`；`vllm/model_executor/models/deepseek_mtp.py` modified +3/-0 (3 lines); hunks: -241,6 +241,9 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +22/-14 (36 lines); hunks: -639,21 +639,19 @@ def __init__(; -694,7 +692,11 @@ def forward(; symbols: __init__, forward, load_weights
  - `vllm/model_executor/models/deepseek_mtp.py` modified +3/-0 (3 lines); hunks: -241,6 +241,9 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -639,21 +639,19 @@ def __init__(
-        self.wk = ReplicatedLinear(
+        # Fused wk + weights_proj: single GEMM producing [head_dim + n_head].
+        # weights_proj does not get quantized, so we run both with quant_config=None
+        # wk may be upcasted from the default quant; experiments show fusion is always
+        # faster unless WK proj is in FP4, which is not the case for all known quants.
+        self.wk_weights_proj = MergedColumnParallelLinear(
diff -- vllm/model_executor/models/deepseek_mtp.py
@@ -241,6 +241,9 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
+            # Fused indexer wk + weights_proj
+            ("wk_weights_proj", "wk", 0),
+            ("wk_weights_proj", "weights_proj", 1),
```

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +22/-14; `vllm/model_executor/models/deepseek_mtp.py` modified +3/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_mtp.py`, `vllm/model_executor/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #38870 - [Bugfix] Fix DSV32 weight loading

- 链接: https://github.com/vllm-project/vllm/pull/38870
- 状态/时间: merged / 2026-04-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/deepseek_mtp.py`, `vllm/model_executor/models/deepseek_v2.py`；关联提交 `8617f8676b5a`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+68/-27，可读 patch 158 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix DSV32 weight loading」；模型线: DeepSeek V3.2；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`；技术摘要: 覆盖「[Bugfix] Fix DSV32 weight loading」；主要实现面是 `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +55/-24 (79 lines); hunks: -625,6 +625,11 @@ def __init__(; -639,18 +644,36 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`；`vllm/model_executor/models/deepseek_mtp.py` modified +13/-3 (16 lines); hunks: -184,11 +184,16 @@ class DeepSeekMTP(nn.Module, DeepseekV2MixtureOfExperts):; -241,11 +246,16 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: DeepSeekMTP, __init__, set_moe_parameters, load_weights，涉及 `DeepSeekMTP, __init__, set_moe_parameters`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +55/-24 (79 lines); hunks: -625,6 +625,11 @@ def __init__(; -639,18 +644,36 @@ def __init__(; symbols: __init__, forward
  - `vllm/model_executor/models/deepseek_mtp.py` modified +13/-3 (16 lines); hunks: -184,11 +184,16 @@ class DeepSeekMTP(nn.Module, DeepseekV2MixtureOfExperts):; -241,11 +246,16 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: DeepSeekMTP, __init__, set_moe_parameters, load_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -625,6 +625,11 @@ def __init__(
+        self.quant_config = quant_config
+        self.is_fp4_ckpt = (
+            self.quant_config is not None
+            and self.quant_config.get_name() == "modelopt_fp4"
+        )
@@ -639,18 +644,36 @@ def __init__(
diff -- vllm/model_executor/models/deepseek_mtp.py
@@ -184,11 +184,16 @@ class DeepSeekMTP(nn.Module, DeepseekV2MixtureOfExperts):
+        self.quant_config = vllm_config.quant_config
+        self.is_fp4_ckpt = (
+            self.quant_config is not None
+            and self.quant_config.get_name() == "modelopt_fp4"
+        )
@@ -241,11 +246,16 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
```

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +55/-24; `vllm/model_executor/models/deepseek_mtp.py` modified +13/-3
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_mtp.py`, `vllm/model_executor/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #37421 - [Perf][Kernel] Persistent TopK scheduler: unified CUDAGraph-safe kernel with dynamic per-row dispatch - DeepSeek-V3.2 DSA decode

- 链接: https://github.com/vllm-project/vllm/pull/37421
- 状态/时间: merged / 2026-04-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/deepseek_v2.py`；关联提交 `b55d830ec782`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+2039/-483，可读 patch 2698 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Perf][Kernel] Persistent TopK scheduler: unified CUDAGraph-safe kernel with dynamic per-row dispatch - DeepSeek-V3.2 DSA decode」；模型线: DeepSeek V3.2；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/deepseek_v2.py`；技术摘要: 覆盖「[Perf][Kernel] Persistent TopK scheduler: unified CUDAGraph-safe kernel with dynamic per-row dispatch - DeepSeek-V3.2 DSA decode」；主要实现面是 `vllm/model_executor/models/deepseek_v2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +6/-2 (8 lines); hunks: -67,7 +67,9; -1203,7 +1205,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +6/-2 (8 lines); hunks: -67,7 +67,9; -1203,7 +1205,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -67,7 +67,9 @@
-from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
+from vllm.model_executor.layers.sparse_attn_indexer import (
+    SparseAttnIndexer,
+)
@@ -1203,7 +1205,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
-                vllm_config, prefix, topk_indices_buffer=topk_indices_buffer
```

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +6/-2
- 验证与风险: diff 自带测试面 `tests/kernels/test_top_k_per_row.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #35968 - [Performance] DeepSeek V3.2 multi-stream indexer overlap

- 链接: https://github.com/vllm-project/vllm/pull/35968
- 状态/时间: closed / 2026-04-27
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+187/-11，可读 patch 255 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Performance] DeepSeek V3.2 multi-stream indexer overlap」；模型线: DeepSeek V3.2；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/layers/layernorm.py`, `tests/utils_/test_indexer_dual_stream.py`；技术摘要: 覆盖「[Performance] DeepSeek V3.2 multi-stream indexer overlap」；主要实现面是 `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/layers/layernorm.py`, `tests/utils_/test_indexer_dual_stream.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +84/-8 (92 lines); hunks: -79,7 +79,8; -625,6 +626,11 @@ def __init__(; symbols: __init__, _compute_k, forward，涉及 `__init__, _compute_k, forward`；`vllm/model_executor/layers/layernorm.py` modified +20/-3 (23 lines); hunks: -615,7 +615,24 @@ def __init__(self, dim: int, eps: float = 1e-6):; symbols: __init__, _forward_static, forward，涉及 `__init__, _forward_static, forward`；`tests/utils_/test_indexer_dual_stream.py` added +83/-0 (83 lines); hunks: -0,0 +1,83; symbols: _indexer_weights_and_k_proj_fake, TestIndexerWeightsAndKProjOp, test_fake_output_shapes_and_strides, test_fake_output_shapes_parametrized，涉及 `_indexer_weights_and_k_proj_fake, TestIndexerWeightsAndKProjOp, test_fake_output_shapes_and_strides`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +84/-8 (92 lines); hunks: -79,7 +79,8; -625,6 +626,11 @@ def __init__(; symbols: __init__, _compute_k, forward
  - `vllm/model_executor/layers/layernorm.py` modified +20/-3 (23 lines); hunks: -615,7 +615,24 @@ def __init__(self, dim: int, eps: float = 1e-6):; symbols: __init__, _forward_static, forward
  - `tests/utils_/test_indexer_dual_stream.py` added +83/-0 (83 lines); hunks: -0,0 +1,83; symbols: _indexer_weights_and_k_proj_fake, TestIndexerWeightsAndKProjOp, test_fake_output_shapes_and_strides, test_fake_output_shapes_parametrized
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -79,7 +79,8 @@
-from vllm.utils.torch_utils import direct_register_custom_op
+from vllm.utils.multi_stream_utils import maybe_execute_in_parallel
+from vllm.utils.torch_utils import aux_stream, direct_register_custom_op
@@ -625,6 +626,11 @@ def __init__(
+        self.events = (
+            [torch.cuda.Event(), torch.cuda.Event()]
diff -- vllm/model_executor/layers/layernorm.py
@@ -615,7 +615,24 @@ def __init__(self, dim: int, eps: float = 1e-6):
+    @staticmethod
+    def _forward_static(
+        weight: torch.Tensor,
+        bias: torch.Tensor,
+        dim: int,
+        eps: float,
diff -- tests/utils_/test_indexer_dual_stream.py
@@ -0,0 +1,83 @@
```

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +84/-8; `vllm/model_executor/layers/layernorm.py` modified +20/-3
  - tests: `tests/utils_/test_indexer_dual_stream.py` added +83/-0
- 验证与风险: diff 自带测试面 `tests/utils_/test_indexer_dual_stream.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #41198 - [Bugfix] DSV32/V4 add missing type conversion for non-streaming tool calls

- 链接: https://github.com/vllm-project/vllm/pull/41198
- 状态/时间: merged / 2026-04-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py`；关联提交 `762022cafb1a`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+26/-1，可读 patch 46 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] DSV32/V4 add missing type conversion for non-streaming tool calls」；模型线: DeepSeek V3.2；类别: 缺陷修复；主要 diff: `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py`；技术摘要: 覆盖「[Bugfix] DSV32/V4 add missing type conversion for non-streaming tool calls」；主要实现面是 `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +24/-0 (24 lines); hunks: -188,6 +188,30 @@ def test_multiple_tools(self, parser):; symbols: test_multiple_tools, test_type_conversion_in_non_streaming，涉及 `test_multiple_tools, test_type_conversion_in_non_streaming`；`vllm/tool_parsers/deepseekv32_tool_parser.py` modified +2/-1 (3 lines); hunks: -191,12 +191,13 @@ def extract_tool_calls(; symbols: extract_tool_calls，涉及 `extract_tool_calls`。
- 代码 diff 细节:
  - `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +24/-0 (24 lines); hunks: -188,6 +188,30 @@ def test_multiple_tools(self, parser):; symbols: test_multiple_tools, test_type_conversion_in_non_streaming
  - `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +2/-1 (3 lines); hunks: -191,12 +191,13 @@ def extract_tool_calls(; symbols: extract_tool_calls
- 关键代码摘录:

```diff
diff -- tests/tool_parsers/test_deepseekv32_tool_parser.py
@@ -188,6 +188,30 @@ def test_multiple_tools(self, parser):
+    def test_type_conversion_in_non_streaming(self):
+        """Non-streaming extraction must convert params using the tool schema."""
+        tool = ChatCompletionToolsParam(
+            function=FunctionDefinition(
+                name="toggle",
+                parameters={
diff -- vllm/tool_parsers/deepseekv32_tool_parser.py
@@ -191,12 +191,13 @@ def extract_tool_calls(
+                    params = self._convert_params_with_schema(invoke_name, param_dict)
-                                arguments=json.dumps(param_dict, ensure_ascii=False),
+                                arguments=json.dumps(params, ensure_ascii=False),
```

- 已读文件:
  - tests: `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +24/-0
  - runtime: `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +2/-1
- 验证与风险: diff 自带测试面 `tests/tool_parsers/test_deepseekv32_tool_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
