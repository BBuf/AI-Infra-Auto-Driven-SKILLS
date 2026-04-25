# vllm DeepSeek V3.2 PR Diff Audit Reference

- Rebuilt on: 2026-04-25
- Source baseline: `vllm-project/vllm` trace worktree commit `95995bbef8`
- Collection: model implementation files were traced with `git log --name-only -- <model-files>`, filtered by model keywords in commit subjects, then every PR card was populated from the GitHub Pull Request files API.
- Extra preserved PRs from prior docs: 2
- Rule: use this as the backing dossier for the skill, not only PR titles.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `examples/online_serving/elastic_ep/serve_deepseek_v2.sh` | no direct PR-number commit |
| `tests/evals/gsm8k/configs/DeepSeek-V3.2-DP.yaml` | [#33566](https://github.com/vllm-project/vllm/pull/33566) |
| `tests/evals/gsm8k/configs/DeepSeek-V3.2-DP_MI325.yaml` | no direct PR-number commit |
| `tests/evals/gsm8k/configs/DeepSeek-V3.2-TP.yaml` | [#33566](https://github.com/vllm-project/vllm/pull/33566) |
| `tests/evals/gsm8k/configs/DeepSeek-V3.2-TP_MI325.yaml` | no direct PR-number commit |
| `tests/tool_parsers/test_deepseekv32_tool_parser.py` | [#33703](https://github.com/vllm-project/vllm/pull/33703), [#36056](https://github.com/vllm-project/vllm/pull/36056) |
| `vllm/model_executor/models/deepseek_mtp.py` | [#25896](https://github.com/vllm-project/vllm/pull/25896), [#38684](https://github.com/vllm-project/vllm/pull/38684), [#38870](https://github.com/vllm-project/vllm/pull/38870) |
| `vllm/model_executor/models/deepseek_v2.py` | [#25896](https://github.com/vllm-project/vllm/pull/25896), [#25999](https://github.com/vllm-project/vllm/pull/25999), [#26456](https://github.com/vllm-project/vllm/pull/26456), [#26465](https://github.com/vllm-project/vllm/pull/26465), [#26670](https://github.com/vllm-project/vllm/pull/26670), [#26763](https://github.com/vllm-project/vllm/pull/26763), [#27532](https://github.com/vllm-project/vllm/pull/27532), [#27568](https://github.com/vllm-project/vllm/pull/27568), [#28968](https://github.com/vllm-project/vllm/pull/28968), [#29287](https://github.com/vllm-project/vllm/pull/29287), [#30841](https://github.com/vllm-project/vllm/pull/30841), [#31046](https://github.com/vllm-project/vllm/pull/31046), ... (17 total) |
| `vllm/renderers/deepseek_v32.py` | [#33855](https://github.com/vllm-project/vllm/pull/33855) |
| `vllm/tokenizers/deepseek_v32.py` | [#30658](https://github.com/vllm-project/vllm/pull/30658), [#33855](https://github.com/vllm-project/vllm/pull/33855), [#37004](https://github.com/vllm-project/vllm/pull/37004) |
| `vllm/tokenizers/deepseek_v32_encoding.py` | [#29837](https://github.com/vllm-project/vllm/pull/29837), [#30025](https://github.com/vllm-project/vllm/pull/30025), [#31147](https://github.com/vllm-project/vllm/pull/31147), [#32884](https://github.com/vllm-project/vllm/pull/32884) |
| `vllm/tool_parsers/deepseekv32_tool_parser.py` | [#33703](https://github.com/vllm-project/vllm/pull/33703), [#33964](https://github.com/vllm-project/vllm/pull/33964), [#36056](https://github.com/vllm-project/vllm/pull/36056) |

## Timeline

| Date | PR | State | Title | Main files |
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
| 2025-12-15 | [#30658](https://github.com/vllm-project/vllm/pull/30658) | merged | [Bugfix] Fix deepseek_v32 tokenizer_mode | `vllm/tokenizers/deepseek_v32.py` |
| 2025-12-17 | [#30841](https://github.com/vllm-project/vllm/pull/30841) | merged | [Bugfix] deepseek-V3.2 self.weights_proj has no bias | `vllm/model_executor/models/deepseek_v2.py` |
| 2025-12-19 | [#31046](https://github.com/vllm-project/vllm/pull/31046) | merged | [Bug] Fix `error 'Dynamo failed to run FX node with fake tensors` for Deepseek V3.2 | `vllm/model_executor/models/deepseek_v2.py` |
| 2025-12-24 | [#31160](https://github.com/vllm-project/vllm/pull/31160) | merged | [Bug] Fix `Number of dimensions of tensors must match.` for Deepseek V3.2 | `vllm/model_executor/models/deepseek_v2.py` |
| 2026-01-05 | [#31147](https://github.com/vllm-project/vllm/pull/31147) | merged | Add chat prefix completion feature to DeepSeek v3.2 | `vllm/tokenizers/deepseek_v32_encoding.py` |
| 2026-01-16 | [#32175](https://github.com/vllm-project/vllm/pull/32175) | merged | [Bugfix] [DeepSeek-V3.2] fix sparse_attn_indexer padding | `vllm/model_executor/models/deepseek_v2.py` |
| 2026-01-21 | [#29287](https://github.com/vllm-project/vllm/pull/29287) | merged | [ROCm][Deepseekv3.2] Refactor Sparse Indexer as CustomOp | `vllm/model_executor/models/deepseek_v2.py` |
| 2026-01-23 | [#32884](https://github.com/vllm-project/vllm/pull/32884) | merged | [BugFix] deepseek_v32_encoding: Replace asserts with proper exceptions | `vllm/tokenizers/deepseek_v32_encoding.py` |
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

## Per-PR Diff Audit Cards

### PR #25896 - [New Model] DeepSeek-V3.2 (Rebased to Main)

- Link: https://github.com/vllm-project/vllm/pull/25896
- Status/date: merged / 2025-09-30
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_mtp.py`, `vllm/model_executor/models/deepseek_v2.py`; associated commits `fa7e254a7f3e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 71 files, +3918/-221, 5400 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR changes model-related implementation. Title: "[New Model] DeepSeek-V3.2 (Rebased to Main)". The diff centers on `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`. PR body context: Rebased dsv32, based on #25869 Run command gsm8k gsm8k, 20-shot
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +445/-4 (449 lines); hunks: -33,36 +33,57; -276,6 +297,7 @@ class DeepseekV2Attention(nn.Module):; symbols: DeepseekV2MLP, DeepseekV2Attention, __init__, touching `DeepseekV2MLP, DeepseekV2Attention, __init__`; `vllm/model_executor/models/deepseek_mtp.py` modified +13/-1 (14 lines); hunks: -53,8 +53,20 @@ def __init__(self, vllm_config: VllmConfig, prefix: str) -> N...; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +445/-4 (449 lines); hunks: -33,36 +33,57; -276,6 +297,7 @@ class DeepseekV2Attention(nn.Module):; symbols: DeepseekV2MLP, DeepseekV2Attention, __init__
  - `vllm/model_executor/models/deepseek_mtp.py` modified +13/-1 (14 lines); hunks: -53,8 +53,20 @@ def __init__(self, vllm_config: VllmConfig, prefix: str) -> N...; symbols: __init__, forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +445/-4; `vllm/model_executor/models/deepseek_mtp.py` modified +13/-1
- Risk and verification: The diff ships test coverage in `tests/compile/test_fusion_attn.py`, `tests/kernels/attention/test_cache.py`, `tests/kernels/attention/test_deepgemm_attention.py`, `tests/kernels/attention/test_flashmla.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #25999 - [Deepseek v3.2] Support indexer prefill chunking

- Link: https://github.com/vllm-project/vllm/pull/25999
- Status/date: merged / 2025-10-02
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `1e50f1be7058`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +149/-79, 324 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[Deepseek v3.2] Support indexer prefill chunking". The diff centers on `vllm/model_executor/models/deepseek_v2.py`. PR body context: ## Purpose Split the prefill to multiple steps, with each step contains a subset of prefill requests. With this approach, we can avoid the large output caused by gather kv cache...
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +37/-38 (75 lines); hunks: -583,44 +583,43 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer, touching `sparse_attn_indexer`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +37/-38 (75 lines); hunks: -583,44 +583,43 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +37/-38
- Risk and verification: The diff ships test coverage in `tests/v1/attention/test_sparse_mla_backends.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #26456 - [Deepseek-V3.2][Kernel] Integrate cuda indexer k cache gather

- Link: https://github.com/vllm-project/vllm/pull/26456
- Status/date: merged / 2025-10-15
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `f5ed68ef63d0`
- Diff scope read: GitHub Pull Request files API returned 1 files, +6/-68, 104 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR optimizes an inference path or backend selection. Title: "[Deepseek-V3.2][Kernel] Integrate cuda indexer k cache gather". The diff centers on `vllm/model_executor/models/deepseek_v2.py`. PR body context: Replace torch `cp_gather_indexer_k_quant_cache` to cuda op. Follow up for #25931 gsm8k 20 shots
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +6/-68 (74 lines); hunks: -75,7 +75,7; -483,69 +483,6 @@ def get_attn_backend(self) -> AttentionBackend:; symbols: get_attn_backend, cp_gather_indexer_k_quant_cache, sparse_attn_indexer, touching `get_attn_backend, cp_gather_indexer_k_quant_cache, sparse_attn_indexer`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +6/-68 (74 lines); hunks: -75,7 +75,7; -483,69 +483,6 @@ def get_attn_backend(self) -> AttentionBackend:; symbols: get_attn_backend, cp_gather_indexer_k_quant_cache, sparse_attn_indexer
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +6/-68
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #26763 - [Deepseek v3.2] Optimize top_k_per_row

- Link: https://github.com/vllm-project/vllm/pull/26763
- Status/date: merged / 2025-10-21
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `80e94529845d`
- Diff scope read: GitHub Pull Request files API returned 5 files, +13/-49, 203 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR optimizes an inference path or backend selection. Title: "[Deepseek v3.2] Optimize top_k_per_row". The diff centers on `vllm/model_executor/models/deepseek_v2.py`. PR body context: ## Purpose This PR optimizes kernel top_k_per_row. Local testing shows it is about 2.5x from its previous version. ## Test Plan ## Test Result --- Essential Elements of an Effec...
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +0/-8 (8 lines); hunks: -577,15 +577,11 @@ def sparse_attn_indexer(; -642,15 +638,11 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer, touching `sparse_attn_indexer`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +0/-8 (8 lines); hunks: -577,15 +577,11 @@ def sparse_attn_indexer(; -642,15 +638,11 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +0/-8
- Risk and verification: The diff ships test coverage in `tests/kernels/test_top_k_per_row.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #26465 - [Deepseek v3.2] Remove extra logics in indexer

- Link: https://github.com/vllm-project/vllm/pull/26465
- Status/date: merged / 2025-10-21
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `09a7e6f6179b`
- Diff scope read: GitHub Pull Request files API returned 5 files, +141/-40, 272 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR extends deployment docs, tests, or CI coverage. Title: "[Deepseek v3.2] Remove extra logics in indexer". The diff centers on `vllm/model_executor/models/deepseek_v2.py`. PR body context: ## Purpose Remove extra logics around `index_end_pos` in the indexer. CC @dcampora ## Test Plan `lm-eval --model local-completions --tasks gsm8k --model_args model=DeepSeek-V3.2...
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +11/-26 (37 lines); hunks: -574,9 +574,9 @@ def sparse_attn_indexer(; -586,9 +586,6 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer, touching `sparse_attn_indexer`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +11/-26 (37 lines); hunks: -574,9 +574,9 @@ def sparse_attn_indexer(; -586,9 +586,6 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +11/-26
- Risk and verification: The diff ships test coverage in `tests/kernels/test_top_k_per_row.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #28968 - [DeepSeek] Fix DeepSeek V3.2 Rope Embedding

- Link: https://github.com/vllm-project/vllm/pull/28968
- Status/date: merged / 2025-11-19
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `88f5b19f0bc6`
- Diff scope read: GitHub Pull Request files API returned 2 files, +17/-3, 69 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[DeepSeek] Fix DeepSeek V3.2 Rope Embedding". The diff centers on `vllm/model_executor/models/deepseek_v2.py`. PR body context: ## Purpose Deepseek recently find error in their official implementation that ROPE in indexer shouldn't be interleaved. ## Test Plan gsm8k 20-shots ## Test Result --- Essential...
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +12/-2 (14 lines); hunks: -846,8 +846,8 @@ def forward(; -1000,6 +1000,14 @@ def __init__(; symbols: forward, __init__, touching `forward, __init__`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +12/-2 (14 lines); hunks: -846,8 +846,8 @@ def forward(; -1000,6 +1000,14 @@ def __init__(; symbols: forward, __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +12/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/mla.py`, `vllm/model_executor/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #26670 - [ROCm] Add AMD GPU support on Deepseek v3.2 and SparseMLA

- Link: https://github.com/vllm-project/vllm/pull/26670
- Status/date: merged / 2025-11-20
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `06c20c990464`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +583/-15, 700 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[ROCm] Add AMD GPU support on Deepseek v3.2 and SparseMLA". The diff centers on `vllm/model_executor/models/deepseek_v2.py`. PR body context: ## Purpose The PR add Deepseek v3.2 support on ROCm platforms. The main change in this PR include: - Replace all hardcode float8_e4m3fn to platform supported fp8 dtype, and add...
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +18/-4 (22 lines); hunks: -591,6 +591,7 @@ def sparse_attn_indexer(; -630,7 +631,7 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer, sparse_attn_indexer_fake, touching `sparse_attn_indexer, sparse_attn_indexer_fake`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +18/-4 (22 lines); hunks: -591,6 +591,7 @@ def sparse_attn_indexer(; -630,7 +631,7 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer, sparse_attn_indexer_fake
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +18/-4
- Risk and verification: Runtime changes concentrate in `vllm/attention/ops/rocm_aiter_mla_sparse.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/platforms/rocm.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #29837 - [Frontend] supports deepseekv32 chat template

- Link: https://github.com/vllm-project/vllm/pull/29837
- Status/date: merged / 2025-12-03
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/tokenizers/deepseek_v32_encoding.py`; associated commits `b78772c43351`
- Diff scope read: GitHub Pull Request files API returned 5 files, +616/-2, 660 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[Frontend] supports deepseekv32 chat template". The diff centers on `vllm/tokenizers/deepseek_v32_encoding.py`. PR body context: ## Purpose Test Plan Result Test tool call Result Test reasoning: Result : --- Essential Elements of an Effective PR Description Checklist - [ ] The purpose of the PR, such as "...
- Key implementation: `vllm/tokenizers/deepseek_v32_encoding.py` added +456/-0 (456 lines); hunks: -0,0 +1,456; symbols: to_json, tools_from_openai_format, tool_calls_from_openai_format, tool_calls_to_openai_format, touching `to_json, tools_from_openai_format, tool_calls_from_openai_format`.
- Code diff details:
  - `vllm/tokenizers/deepseek_v32_encoding.py` added +456/-0 (456 lines); hunks: -0,0 +1,456; symbols: to_json, tools_from_openai_format, tool_calls_from_openai_format, tool_calls_to_openai_format
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/tokenizers/deepseek_v32_encoding.py` added +456/-0
- Risk and verification: Runtime changes concentrate in `vllm/config/model.py`, `vllm/entrypoints/openai/serving_engine.py`, `vllm/tokenizers/__init__.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #30025 - [Bugfix] fixed deepseekv32 tool calling error

- Link: https://github.com/vllm-project/vllm/pull/30025
- Status/date: merged / 2025-12-04
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/tokenizers/deepseek_v32_encoding.py`; associated commits `82a64b3d8f93`
- Diff scope read: GitHub Pull Request files API returned 2 files, +6/-3, 23 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[Bugfix] fixed deepseekv32 tool calling error". The diff centers on `vllm/tokenizers/deepseek_v32_encoding.py`. PR body context: ## Purpose we should use `conversation` instead of `messages`. `conversation` is parsed by the vLLM parser and supports some vLLM-specific formats. ## Test Plan ## Test Result -...
- Key implementation: `vllm/tokenizers/deepseek_v32_encoding.py` modified +4/-2 (6 lines); hunks: -95,8 +95,10 @@ def tool_calls_to_openai_format(tool_calls):; symbols: tool_calls_to_openai_format, encode_arguments_to_dsml, touching `tool_calls_to_openai_format, encode_arguments_to_dsml`.
- Code diff details:
  - `vllm/tokenizers/deepseek_v32_encoding.py` modified +4/-2 (6 lines); hunks: -95,8 +95,10 @@ def tool_calls_to_openai_format(tool_calls):; symbols: tool_calls_to_openai_format, encode_arguments_to_dsml
- Key code excerpts:

```diff
diff -- vllm/tokenizers/deepseek_v32_encoding.py
@@ -95,8 +95,10 @@ def tool_calls_to_openai_format(tool_calls):
-    arguments = json.loads(tool_call["arguments"])
+    if isinstance(tool_call["arguments"], str):
+        arguments = json.loads(tool_call["arguments"])
+    else:
+        arguments = tool_call["arguments"]
```

- Reviewed files:
  - runtime: `vllm/tokenizers/deepseek_v32_encoding.py` modified +4/-2
- Risk and verification: Runtime changes concentrate in `vllm/tokenizers/deepseek_v32_encoding.py`, `vllm/tokenizers/deepseekv32.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #29848 - Add DeepSeek-V3.2 tool parser.

- Link: https://github.com/vllm-project/vllm/pull/29848
- Status/date: merged / 2025-12-04
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +595/-0, 603 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "Add DeepSeek-V3.2 tool parser.". The diff centers on `vllm/entrypoints/openai/tool_parsers/deepseekv32_tool_parser.py`, `vllm/entrypoints/openai/tool_parsers/__init__.py`. PR body context: ## Purpose Support DeepSeek-V3.2 tool call wait for this PR: https://github.com/vllm-project/vllm/pull/29837 ## Test Plan Test (Non-Streaming) Test (Streaming) ## Test Result Re...
- Key implementation: `vllm/entrypoints/openai/tool_parsers/deepseekv32_tool_parser.py` added +591/-0 (591 lines); hunks: -0,0 +1,591; symbols: DeepSeekV32ToolParser, __init__, type, _generate_tool_call_id, touching `DeepSeekV32ToolParser, __init__, type`; `vllm/entrypoints/openai/tool_parsers/__init__.py` modified +4/-0 (4 lines); hunks: -30,6 +30,10.
- Code diff details:
  - `vllm/entrypoints/openai/tool_parsers/deepseekv32_tool_parser.py` added +591/-0 (591 lines); hunks: -0,0 +1,591; symbols: DeepSeekV32ToolParser, __init__, type, _generate_tool_call_id
  - `vllm/entrypoints/openai/tool_parsers/__init__.py` modified +4/-0 (4 lines); hunks: -30,6 +30,10
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/entrypoints/openai/tool_parsers/deepseekv32_tool_parser.py` added +591/-0; `vllm/entrypoints/openai/tool_parsers/__init__.py` modified +4/-0
- Risk and verification: Runtime changes concentrate in `vllm/entrypoints/openai/tool_parsers/__init__.py`, `vllm/entrypoints/openai/tool_parsers/deepseekv32_tool_parser.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27568 - [DeepSeek v3.2] Make top-k work for any logit values.

- Link: https://github.com/vllm-project/vllm/pull/27568
- Status/date: merged / 2025-12-08
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `184076c3fecf`
- Diff scope read: GitHub Pull Request files API returned 5 files, +629/-210, 1067 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR changes model-related implementation. Title: "[DeepSeek v3.2] Make top-k work for any logit values.". The diff centers on `vllm/model_executor/models/deepseek_v2.py`. PR body context: ## Purpose This PR allows top_k_per_row work for any values in logits. Even if the logits differ only in the least significant bytes, top-k is now guaranteed to always give a co...
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +3/-3 (6 lines); hunks: -684,18 +684,18 @@ def sparse_attn_indexer(; -738,7 +738,6 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer, touching `sparse_attn_indexer`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +3/-3 (6 lines); hunks: -684,18 +684,18 @@ def sparse_attn_indexer(; -738,7 +738,6 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +3/-3
- Risk and verification: The diff ships test coverage in `tests/kernels/test_top_k_per_row.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #27532 - [Attention] Use sparse prefill kernel for fp8 kv-cache in DeepSeek-v3.2

- Link: https://github.com/vllm-project/vllm/pull/27532
- Status/date: merged / 2025-12-12
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `3e41992fecdc`
- Diff scope read: GitHub Pull Request files API returned 30 files, +1372/-256, 2323 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[Attention] Use sparse prefill kernel for fp8 kv-cache in DeepSeek-v3.2". The diff centers on `vllm/model_executor/models/deepseek_v2.py`. PR body context: When doing prefill up-convert the kv-cache from fp8 to bf16 and call the bf16 prefill kernel instead of the decode kernel. This PR introduce global workspace management to have...
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +18/-19 (37 lines); hunks: -83,6 +83,7; -618,8 +619,15 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer, sparse_attn_indexer_fake, touching `sparse_attn_indexer, sparse_attn_indexer_fake`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +18/-19 (37 lines); hunks: -83,6 +83,7; -618,8 +619,15 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer, sparse_attn_indexer_fake
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +18/-19
- Risk and verification: The diff ships test coverage in `tests/conftest.py`, `tests/kernels/moe/test_batched_deepgemm.py`, `tests/kernels/moe/test_batched_moe.py`, `tests/kernels/moe/test_block_fp8.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #30658 - [Bugfix] Fix deepseek_v32 tokenizer_mode

- Link: https://github.com/vllm-project/vllm/pull/30658
- Status/date: merged / 2025-12-15
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/tokenizers/deepseek_v32.py`; associated commits `a524d1ba0af4`
- Diff scope read: GitHub Pull Request files API returned 4 files, +3/-3, 27 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Bugfix] Fix deepseek_v32 tokenizer_mode". The diff centers on `vllm/tokenizers/deepseek_v32.py`. PR body context: ## Purpose https://github.com/vllm-project/vllm/pull/30609 changed the tokenizer_mode of deepseek_v32 ,and will raise the following error: This PR revert tokenizer_mode name fro...
- Key implementation: `vllm/tokenizers/deepseek_v32.py` renamed +0/-0 (0 lines).
- Code diff details:
  - `vllm/tokenizers/deepseek_v32.py` renamed +0/-0 (0 lines)
- Key code excerpts:

```diff
No textual patch was returned by GitHub for the selected changed files.
```

- Reviewed files:
  - runtime: `vllm/tokenizers/deepseek_v32.py` renamed +0/-0
- Risk and verification: Runtime changes concentrate in `vllm/entrypoints/openai/serving_engine.py`, `vllm/tokenizers/deepseek_v32.py`, `vllm/tokenizers/registry.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #30841 - [Bugfix] deepseek-V3.2 self.weights_proj has no bias

- Link: https://github.com/vllm-project/vllm/pull/30841
- Status/date: merged / 2025-12-17
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `84896fda22d3`
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-1, 13 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[Bugfix] deepseek-V3.2 self.weights_proj has no bias". The diff centers on `vllm/model_executor/models/deepseek_v2.py`. PR body context: ## Purpose self.weights_proj has no bias,some other hardware bias maybe not initial with 0 maybe not correct H20 bias initial with 0 !image kunlun bias not initial with 0 !image...
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +5/-1 (6 lines); hunks: -835,7 +835,11 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +5/-1 (6 lines); hunks: -835,7 +835,11 @@ def __init__(; symbols: __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +5/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #31046 - [Bug] Fix `error 'Dynamo failed to run FX node with fake tensors` for Deepseek V3.2

- Link: https://github.com/vllm-project/vllm/pull/31046
- Status/date: merged / 2025-12-19
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `4cf9429897c1`
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-2, 14 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[Bug] Fix `error 'Dynamo failed to run FX node with fake tensors` for Deepseek V3.2". The diff centers on `vllm/model_executor/models/deepseek_v2.py`. PR body context: ## Purpose `export MODEL="deepseek-ai/DeepSeek-V3.2"` `vllm serve "$MODEL" -tp 8 --port 9256 --enable-expert-parallel -cc '{"mode":3,"pass_config":{"fuse_norm_quant":true,"elimi...
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +5/-2 (7 lines); hunks: -878,8 +878,11 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +5/-2 (7 lines); hunks: -878,8 +878,11 @@ def forward(; symbols: forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +5/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #31160 - [Bug] Fix `Number of dimensions of tensors must match.` for Deepseek V3.2

- Link: https://github.com/vllm-project/vllm/pull/31160
- Status/date: merged / 2025-12-24
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `76e6a951925b`
- Diff scope read: GitHub Pull Request files API returned 1 files, +6/-3, 18 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[Bug] Fix `Number of dimensions of tensors must match.` for Deepseek V3.2". The diff centers on `vllm/model_executor/models/deepseek_v2.py`. PR body context: ## Purpose `export MODEL="deepseek-ai/DeepSeek-V3.2"` `vllm serve "$MODEL" -tp 8 --port 9256 --enable-expert-parallel` Will trigger error The root cause: Number of dimensions of...
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +6/-3 (9 lines); hunks: -878,11 +878,14 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +6/-3 (9 lines); hunks: -878,11 +878,14 @@ def forward(; symbols: forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +6/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #31147 - Add chat prefix completion feature to DeepSeek v3.2

- Link: https://github.com/vllm-project/vllm/pull/31147
- Status/date: merged / 2026-01-05
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/tokenizers/deepseek_v32_encoding.py`; associated commits `346e56455a3b`
- Diff scope read: GitHub Pull Request files API returned 1 files, +9/-5, 28 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "Add chat prefix completion feature to DeepSeek v3.2". The diff centers on `vllm/tokenizers/deepseek_v32_encoding.py`. PR body context: Add prefix continuation feature to DeepSeek v3.2
- Key implementation: `vllm/tokenizers/deepseek_v32_encoding.py` modified +9/-5 (14 lines); hunks: -169,6 +169,7 @@ def render_message(; -273,11 +274,14 @@ def render_message(; symbols: render_message, touching `render_message`.
- Code diff details:
  - `vllm/tokenizers/deepseek_v32_encoding.py` modified +9/-5 (14 lines); hunks: -169,6 +169,7 @@ def render_message(; -273,11 +274,14 @@ def render_message(; symbols: render_message
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/tokenizers/deepseek_v32_encoding.py` modified +9/-5
- Risk and verification: Runtime changes concentrate in `vllm/tokenizers/deepseek_v32_encoding.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #32175 - [Bugfix] [DeepSeek-V3.2] fix sparse_attn_indexer padding

- Link: https://github.com/vllm-project/vllm/pull/32175
- Status/date: merged / 2026-01-16
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `5de6dd0662da`
- Diff scope read: GitHub Pull Request files API returned 1 files, +9/-2, 38 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Bugfix] [DeepSeek-V3.2] fix sparse_attn_indexer padding". The diff centers on `vllm/model_executor/models/deepseek_v2.py`. PR body context: ## Purpose Fix https://github.com/vllm-project/vllm/issues/32172 ## Test Plan ## Test Result vllm bench ok. lm_eval: Main: This PR: --- Essential Elements of an Effective PR Des...
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +9/-2 (11 lines); hunks: -717,13 +717,20 @@ def sparse_attn_indexer(; -739,14 +746,14 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer, touching `sparse_attn_indexer`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +9/-2 (11 lines); hunks: -717,13 +717,20 @@ def sparse_attn_indexer(; -739,14 +746,14 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +9/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #29287 - [ROCm][Deepseekv3.2] Refactor Sparse Indexer as CustomOp

- Link: https://github.com/vllm-project/vllm/pull/29287
- Status/date: merged / 2026-01-21
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `6c20e89c0209`
- Diff scope read: GitHub Pull Request files API returned 8 files, +982/-323, 1521 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR optimizes an inference path or backend selection. Title: "[ROCm][Deepseekv3.2] Refactor Sparse Indexer as CustomOp". The diff centers on `vllm/model_executor/models/deepseek_v2.py`. PR body context: ## Purpose This PR optimize the deepseekv3.2's performance on AMD's device, and separate `SparseAttnIndexer` out as a `CustomOp` as it contains lots of heavy kernels like `fp8_m...
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +14/-233 (247 lines); hunks: -43,7 +43,6; -63,6 +62,7; symbols: get_attn_backend, sparse_attn_indexer, sparse_attn_indexer_fake, Indexer, touching `get_attn_backend, sparse_attn_indexer, sparse_attn_indexer_fake`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +14/-233 (247 lines); hunks: -43,7 +43,6; -63,6 +62,7; symbols: get_attn_backend, sparse_attn_indexer, sparse_attn_indexer_fake, Indexer
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +14/-233
- Risk and verification: Runtime changes concentrate in `vllm/_aiter_ops.py`, `vllm/config/compilation.py`, `vllm/model_executor/layers/sparse_attn_indexer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #32884 - [BugFix] deepseek_v32_encoding: Replace asserts with proper exceptions

- Link: https://github.com/vllm-project/vllm/pull/32884
- Status/date: merged / 2026-01-23
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/tokenizers/deepseek_v32_encoding.py`; associated commits `f61c9da711d8`
- Diff scope read: GitHub Pull Request files API returned 1 files, +39/-28, 160 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[BugFix] deepseek_v32_encoding: Replace asserts with proper exceptions". The diff centers on `vllm/tokenizers/deepseek_v32_encoding.py`. PR body context: Resolves: https://github.com/vllm-project/vllm/issues/32874 Replace validation asserts with ValueError and parsing asserts with RuntimeError to return 400 Bad Request instead of...
- Key implementation: `vllm/tokenizers/deepseek_v32_encoding.py` modified +39/-28 (67 lines); hunks: -154,10 +154,12 @@ def find_last_user_index(messages: list[dict[str, Any]]) -...; -187,7 +189,8 @@ def render_message(; symbols: find_last_user_index, render_message, touching `find_last_user_index, render_message`.
- Code diff details:
  - `vllm/tokenizers/deepseek_v32_encoding.py` modified +39/-28 (67 lines); hunks: -154,10 +154,12 @@ def find_last_user_index(messages: list[dict[str, Any]]) -...; -187,7 +189,8 @@ def render_message(; symbols: find_last_user_index, render_message
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/tokenizers/deepseek_v32_encoding.py` modified +39/-28
- Risk and verification: Runtime changes concentrate in `vllm/tokenizers/deepseek_v32_encoding.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #33090 - [Bugfix] Fix DeepseekV32 `AssertionError: num_kv_heads == 1`

- Link: https://github.com/vllm-project/vllm/pull/33090
- Status/date: merged / 2026-01-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Bugfix] Fix DeepseekV32 `AssertionError: num_kv_heads == 1`". The diff centers on `vllm/distributed/kv_transfer/kv_connector/utils.py`. PR body context: Fix https://github.com/vllm-project/vllm/pull/33086
- Key implementation: `vllm/distributed/kv_transfer/kv_connector/utils.py` modified +1/-1 (2 lines); hunks: -322,7 +322,7 @@ def __post_init__(self):; symbols: __post_init__, touching `__post_init__`.
- Code diff details:
  - `vllm/distributed/kv_transfer/kv_connector/utils.py` modified +1/-1 (2 lines); hunks: -322,7 +322,7 @@ def __post_init__(self):; symbols: __post_init__
- Key code excerpts:

```diff
diff -- vllm/distributed/kv_transfer/kv_connector/utils.py
@@ -322,7 +322,7 @@ def __post_init__(self):
-            num_blocks=1, block_size=16, num_kv_heads=4, head_size=1
+            num_blocks=1, block_size=16, num_kv_heads=1, head_size=1
```

- Reviewed files:
  - runtime: `vllm/distributed/kv_transfer/kv_connector/utils.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/distributed/kv_transfer/kv_connector/utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #33566 - [CI] Add DeepSeek V3.2 nightly eval

- Link: https://github.com/vllm-project/vllm/pull/33566
- Status/date: merged / 2026-02-02
- Trace source: `git log --name-only -- <model-files>` found it through `tests/evals/gsm8k/configs/DeepSeek-V3.2-DP.yaml`, `tests/evals/gsm8k/configs/DeepSeek-V3.2-TP.yaml`; associated commits `9f8cb81b44ce`
- Diff scope read: GitHub Pull Request files API returned 3 files, +24/-0, 29 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[CI] Add DeepSeek V3.2 nightly eval". The diff centers on `tests/evals/gsm8k/configs/DeepSeek-V3.2-DP.yaml`, `tests/evals/gsm8k/configs/DeepSeek-V3.2-TP.yaml`. PR body context: ## Purpose Adds DeepSeek V3.2 to nightly lm eval on H200 to catch issues like https://github.com/vllm-project/vllm/issues/33546 ## Test Plan LM Eval Large Models (H200) ## Test...
- Key implementation: `tests/evals/gsm8k/configs/DeepSeek-V3.2-DP.yaml` added +11/-0 (11 lines); hunks: -0,0 +1,11; `tests/evals/gsm8k/configs/DeepSeek-V3.2-TP.yaml` added +11/-0 (11 lines); hunks: -0,0 +1,11.
- Code diff details:
  - `tests/evals/gsm8k/configs/DeepSeek-V3.2-DP.yaml` added +11/-0 (11 lines); hunks: -0,0 +1,11
  - `tests/evals/gsm8k/configs/DeepSeek-V3.2-TP.yaml` added +11/-0 (11 lines); hunks: -0,0 +1,11
- Key code excerpts:

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

- Reviewed files:
  - tests: `tests/evals/gsm8k/configs/DeepSeek-V3.2-DP.yaml` added +11/-0; `tests/evals/gsm8k/configs/DeepSeek-V3.2-TP.yaml` added +11/-0
- Risk and verification: The diff ships test coverage in `tests/evals/gsm8k/configs/DeepSeek-V3.2-DP.yaml`, `tests/evals/gsm8k/configs/DeepSeek-V3.2-TP.yaml`, `tests/evals/gsm8k/configs/models-h200.txt`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #33964 - [Bugfix] Fix the issue where tool calling does not work when using fast detokenization with dsv32

- Link: https://github.com/vllm-project/vllm/pull/33964
- Status/date: merged / 2026-02-06
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/tool_parsers/deepseekv32_tool_parser.py`; associated commits `7bec4351305f`
- Diff scope read: GitHub Pull Request files API returned 1 files, +12/-0, 19 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Bugfix] Fix the issue where tool calling does not work when using fast detokenization with dsv32". The diff centers on `vllm/tool_parsers/deepseekv32_tool_parser.py`. PR body context: ## Purpose Fix the issue where tool calling does not work when using fast detokenization with dsv32 dsv32 uses special tokens like `｜DSML｜function_calls>`, which are skipped by...
- Key implementation: `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +12/-0 (12 lines); hunks: -110,6 +110,18 @@ def _generate_tool_call_id(self) -> str:; symbols: _generate_tool_call_id, adjust_request, _reset_streaming_state, touching `_generate_tool_call_id, adjust_request, _reset_streaming_state`.
- Code diff details:
  - `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +12/-0 (12 lines); hunks: -110,6 +110,18 @@ def _generate_tool_call_id(self) -> str:; symbols: _generate_tool_call_id, adjust_request, _reset_streaming_state
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +12/-0
- Risk and verification: Runtime changes concentrate in `vllm/tool_parsers/deepseekv32_tool_parser.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #33855 - [Perf] Simplify DeepseekV32 tokenizer, ensure fast detokenization used

- Link: https://github.com/vllm-project/vllm/pull/33855
- Status/date: merged / 2026-02-08
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/renderers/deepseek_v32.py`, `vllm/tokenizers/deepseek_v32.py`; associated commits `a96197f564cb`
- Diff scope read: GitHub Pull Request files API returned 4 files, +88/-203, 348 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR optimizes an inference path or backend selection. Title: "[Perf] Simplify DeepseekV32 tokenizer, ensure fast detokenization used". The diff centers on `vllm/tokenizers/deepseek_v32.py`, `vllm/renderers/deepseek_v32.py`. PR body context: Noticed while looking into another issue. When `tokenizer_mode=deepseek_v32` is used, the fast detokenization path wasn't being taken. Also avoids the need for special-case xgra...
- Key implementation: `vllm/tokenizers/deepseek_v32.py` modified +77/-179 (256 lines); hunks: -1,191 +1,89; symbols: DeepseekV32Tokenizer, from_pretrained, get_deepseek_v32_tokenizer, _DeepseekV32Tokenizer, touching `DeepseekV32Tokenizer, from_pretrained, get_deepseek_v32_tokenizer`; `vllm/renderers/deepseek_v32.py` modified +3/-2 (5 lines); hunks: -13,6 +13,7; -48,10 +49,10 @@ def __init__(; symbols: __init__, tokenizer, get_tokenizer, touching `__init__, tokenizer, get_tokenizer`.
- Code diff details:
  - `vllm/tokenizers/deepseek_v32.py` modified +77/-179 (256 lines); hunks: -1,191 +1,89; symbols: DeepseekV32Tokenizer, from_pretrained, get_deepseek_v32_tokenizer, _DeepseekV32Tokenizer
  - `vllm/renderers/deepseek_v32.py` modified +3/-2 (5 lines); hunks: -13,6 +13,7; -48,10 +49,10 @@ def __init__(; symbols: __init__, tokenizer, get_tokenizer
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/tokenizers/deepseek_v32.py` modified +77/-179; `vllm/renderers/deepseek_v32.py` modified +3/-2
- Risk and verification: The diff ships test coverage in `tests/tokenizers_/test_basic.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #37004 - [Bugfix] Fix DeepSeek-V3.2 tokenizer stripping spaces

- Link: https://github.com/vllm-project/vllm/pull/37004
- Status/date: merged / 2026-03-13
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/tokenizers/deepseek_v32.py`; associated commits `9efc4db9658a`
- Diff scope read: GitHub Pull Request files API returned 2 files, +4/-2, 25 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Bugfix] Fix DeepSeek-V3.2 tokenizer stripping spaces". The diff centers on `vllm/tokenizers/deepseek_v32.py`. PR body context: DISCLAIMER: Generated with claude code ## Purpose DeepSeek-V3.2's `tokenizer_config.json` specifies `LlamaTokenizerFast`, but the tokenizer is byte-level BPE. `LlamaTokenizer` r...
- Key implementation: `vllm/tokenizers/deepseek_v32.py` modified +2/-2 (4 lines); hunks: -3,7 +3,7; -85,5 +85,5 @@ def __reduce__(self):; symbols: __reduce__, DeepseekV32Tokenizer, from_pretrained, touching `__reduce__, DeepseekV32Tokenizer, from_pretrained`.
- Code diff details:
  - `vllm/tokenizers/deepseek_v32.py` modified +2/-2 (4 lines); hunks: -3,7 +3,7; -85,5 +85,5 @@ def __reduce__(self):; symbols: __reduce__, DeepseekV32Tokenizer, from_pretrained
- Key code excerpts:

```diff
diff -- vllm/tokenizers/deepseek_v32.py
@@ -3,7 +3,7 @@
-from transformers import AutoTokenizer
+from transformers import PreTrainedTokenizerFast
@@ -85,5 +85,5 @@ def __reduce__(self):
-        tokenizer = AutoTokenizer.from_pretrained(*args, **kwargs)
+        tokenizer = PreTrainedTokenizerFast.from_pretrained(*args, **kwargs)
```

- Reviewed files:
  - runtime: `vllm/tokenizers/deepseek_v32.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `vllm/config/model.py`, `vllm/tokenizers/deepseek_v32.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #36056 - [Bugfix] Fix Deepseekv32 tool parser when stream interval > 1

- Link: https://github.com/vllm-project/vllm/pull/36056
- Status/date: merged / 2026-03-19
- Trace source: `git log --name-only -- <model-files>` found it through `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py`; associated commits `be12afd284f3`
- Diff scope read: GitHub Pull Request files API returned 2 files, +622/-437, 1113 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Bugfix] Fix Deepseekv32 tool parser when stream interval > 1". The diff centers on `vllm/tool_parsers/deepseekv32_tool_parser.py`, `tests/tool_parsers/test_deepseekv32_tool_parser.py`. PR body context: ## Purpose The deepseek 3.2 tool parser used an incremental state machine (~20 instance variables) that parsed DSML tags character-by-character as tokens streamed in. With strea...
- Key implementation: `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +146/-437 (583 lines); hunks: -48,41 +48,12 @@ def __init__(self, tokenizer: TokenizerLike):; -106,10 +77,6 @@ def __init__(self, tokenizer: TokenizerLike):; symbols: __init__, type, _generate_tool_call_id, adjust_request, touching `__init__, type, _generate_tool_call_id`; `tests/tool_parsers/test_deepseekv32_tool_parser.py` added +476/-0 (476 lines); hunks: -0,0 +1,476; symbols: make_parser, make_tool_param, make_request, build_tool_call, touching `make_parser, make_tool_param, make_request`.
- Code diff details:
  - `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +146/-437 (583 lines); hunks: -48,41 +48,12 @@ def __init__(self, tokenizer: TokenizerLike):; -106,10 +77,6 @@ def __init__(self, tokenizer: TokenizerLike):; symbols: __init__, type, _generate_tool_call_id, adjust_request
  - `tests/tool_parsers/test_deepseekv32_tool_parser.py` added +476/-0 (476 lines); hunks: -0,0 +1,476; symbols: make_parser, make_tool_param, make_request, build_tool_call
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +146/-437
  - tests: `tests/tool_parsers/test_deepseekv32_tool_parser.py` added +476/-0
- Risk and verification: The diff ships test coverage in `tests/tool_parsers/test_deepseekv32_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #33703 - [Bugfix] Support multi-type params parsing for DeepSeek v3.2

- Link: https://github.com/vllm-project/vllm/pull/33703
- Status/date: merged / 2026-03-30
- Trace source: `git log --name-only -- <model-files>` found it through `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py`; associated commits `a6db99ba02ec`
- Diff scope read: GitHub Pull Request files API returned 2 files, +201/-18, 250 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[Bugfix] Support multi-type params parsing for DeepSeek v3.2". The diff centers on `tests/tool_parsers/test_deepseekv32_tool_parser.py`, `vllm/tool_parsers/deepseekv32_tool_parser.py`. PR body context: ## Purpose Kilo code uses multi typed params for some reason, and such calls fails to render with exception `'list' object has no attribute 'lowercase'` when Kilo code passes `t...
- Key implementation: `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +181/-0 (181 lines); hunks: -11,6 +11,7; -474,3 +475,183 @@ def test_no_emission_while_incomplete(self, parser):; symbols: test_no_emission_while_incomplete, deepseekv32_tokenizer, parser, test_convert_param_value_single_types, touching `test_no_emission_while_incomplete, deepseekv32_tokenizer, parser`; `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +20/-18 (38 lines); hunks: -100,7 +100,7 @@ def _parse_invoke_params(self, invoke_str: str) -> dict:; -109,29 +109,31 @@ def _convert_param_value(self, value: str, param_type: str...; symbols: _parse_invoke_params, _convert_param_value, _convert_param_value_checked, touching `_parse_invoke_params, _convert_param_value, _convert_param_value_checked`.
- Code diff details:
  - `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +181/-0 (181 lines); hunks: -11,6 +11,7; -474,3 +475,183 @@ def test_no_emission_while_incomplete(self, parser):; symbols: test_no_emission_while_incomplete, deepseekv32_tokenizer, parser, test_convert_param_value_single_types
  - `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +20/-18 (38 lines); hunks: -100,7 +100,7 @@ def _parse_invoke_params(self, invoke_str: str) -> dict:; -109,29 +109,31 @@ def _convert_param_value(self, value: str, param_type: str...; symbols: _parse_invoke_params, _convert_param_value, _convert_param_value_checked
- Key code excerpts:

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

- Reviewed files:
  - tests: `tests/tool_parsers/test_deepseekv32_tool_parser.py` modified +181/-0
  - runtime: `vllm/tool_parsers/deepseekv32_tool_parser.py` modified +20/-18
- Risk and verification: The diff ships test coverage in `tests/tool_parsers/test_deepseekv32_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #38684 - [Perf] DSV3.2 Indexer Fused Weights Projection

- Link: https://github.com/vllm-project/vllm/pull/38684
- Status/date: merged / 2026-04-02
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_mtp.py`, `vllm/model_executor/models/deepseek_v2.py`; associated commits `5f96f9aff10f`
- Diff scope read: GitHub Pull Request files API returned 2 files, +25/-14, 79 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR optimizes an inference path or backend selection. Title: "[Perf] DSV3.2 Indexer Fused Weights Projection". The diff centers on `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`. PR body context: ## Purpose Fuse the WK and Weights_Proj projections in the DSV3.2 Indexer. This is an alternative optimization to https://github.com/vllm-project/vllm/pull/35968, which overlaps...
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +22/-14 (36 lines); hunks: -639,21 +639,19 @@ def __init__(; -694,7 +692,11 @@ def forward(; symbols: __init__, forward, load_weights, touching `__init__, forward, load_weights`; `vllm/model_executor/models/deepseek_mtp.py` modified +3/-0 (3 lines); hunks: -241,6 +241,9 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +22/-14 (36 lines); hunks: -639,21 +639,19 @@ def __init__(; -694,7 +692,11 @@ def forward(; symbols: __init__, forward, load_weights
  - `vllm/model_executor/models/deepseek_mtp.py` modified +3/-0 (3 lines); hunks: -241,6 +241,9 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +22/-14; `vllm/model_executor/models/deepseek_mtp.py` modified +3/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_mtp.py`, `vllm/model_executor/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #38870 - [Bugfix] Fix DSV32 weight loading

- Link: https://github.com/vllm-project/vllm/pull/38870
- Status/date: merged / 2026-04-04
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_mtp.py`, `vllm/model_executor/models/deepseek_v2.py`; associated commits `8617f8676b5a`
- Diff scope read: GitHub Pull Request files API returned 2 files, +68/-27, 158 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Bugfix] Fix DSV32 weight loading". The diff centers on `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`. PR body context: ## Purpose #38684 intorude bug on the fp8 checkpoint ## Test Plan gsm8k score ## Test Result --- Essential Elements of an Effective PR Description Checklist - [x] The purpose of...
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +55/-24 (79 lines); hunks: -625,6 +625,11 @@ def __init__(; -639,18 +644,36 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`; `vllm/model_executor/models/deepseek_mtp.py` modified +13/-3 (16 lines); hunks: -184,11 +184,16 @@ class DeepSeekMTP(nn.Module, DeepseekV2MixtureOfExperts):; -241,11 +246,16 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: DeepSeekMTP, __init__, set_moe_parameters, load_weights, touching `DeepSeekMTP, __init__, set_moe_parameters`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +55/-24 (79 lines); hunks: -625,6 +625,11 @@ def __init__(; -639,18 +644,36 @@ def __init__(; symbols: __init__, forward
  - `vllm/model_executor/models/deepseek_mtp.py` modified +13/-3 (16 lines); hunks: -184,11 +184,16 @@ class DeepSeekMTP(nn.Module, DeepseekV2MixtureOfExperts):; -241,11 +246,16 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: DeepSeekMTP, __init__, set_moe_parameters, load_weights
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +55/-24; `vllm/model_executor/models/deepseek_mtp.py` modified +13/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_mtp.py`, `vllm/model_executor/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #37421 - [Perf][Kernel] Persistent TopK scheduler: unified CUDAGraph-safe kernel with dynamic per-row dispatch - DeepSeek-V3.2 DSA decode

- Link: https://github.com/vllm-project/vllm/pull/37421
- Status/date: merged / 2026-04-08
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/deepseek_v2.py`; associated commits `b55d830ec782`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +2039/-483, 2698 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR optimizes an inference path or backend selection. Title: "[Perf][Kernel] Persistent TopK scheduler: unified CUDAGraph-safe kernel with dynamic per-row dispatch - DeepSeek-V3.2 DSA decode". The diff centers on `vllm/model_executor/models/deepseek_v2.py`. PR body context: ## Summary Redesigns the persistent TopK kernel used by DSA as a true persistent scheduler with dynamic per-row path selection. This supersedes and closes #34265, which took a C...
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +6/-2 (8 lines); hunks: -67,7 +67,9; -1203,7 +1205,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +6/-2 (8 lines); hunks: -67,7 +67,9; -1203,7 +1205,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +6/-2
- Risk and verification: The diff ships test coverage in `tests/kernels/test_top_k_per_row.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.
