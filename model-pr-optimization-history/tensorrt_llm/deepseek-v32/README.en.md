# TensorRT-LLM DeepSeek V3.2 Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `docs/source/blogs/tech_blog/blog15_Optimizing_DeepSeek_V32_on_NVIDIA_Blackwell_GPUs.md` | [#10565](https://github.com/NVIDIA/TensorRT-LLM/pull/10565) |
| `examples/models/core/deepseek_v3/README.md` | [#9141](https://github.com/NVIDIA/TensorRT-LLM/pull/9141), [#9383](https://github.com/NVIDIA/TensorRT-LLM/pull/9383) |
| `examples/serve/chat_templates/tool_chat_template_deepseekv31.jinja` | no direct PR-number commit |
| `tensorrt_llm/_torch/_experimental/modeling_v2/models/deepseek_v3/__init__.py` | no direct PR-number commit |
| `tensorrt_llm/_torch/_experimental/modeling_v2/models/deepseek_v3/r1_0528_nvfp4__sm_103__dep4/__init__.py` | no direct PR-number commit |
| `tensorrt_llm/_torch/_experimental/modeling_v2/models/deepseek_v3/r1_0528_nvfp4__sm_103__dep4/modeling.py` | no direct PR-number commit |
| `tensorrt_llm/_torch/_experimental/modeling_v2/models/deepseek_v3/r1_0528_nvfp4__sm_103__dep4/weights.py` | no direct PR-number commit |
| `tensorrt_llm/_torch/_experimental/modeling_v2/models/deepseek_v3/routing.py` | no direct PR-number commit |
| `tensorrt_llm/_torch/configs/deepseek_v3.py` | [#8405](https://github.com/NVIDIA/TensorRT-LLM/pull/8405) |
| `tensorrt_llm/_torch/models/modeling_deepseekv3.py` | [#8405](https://github.com/NVIDIA/TensorRT-LLM/pull/8405), [#11507](https://github.com/NVIDIA/TensorRT-LLM/pull/11507), [#11989](https://github.com/NVIDIA/TensorRT-LLM/pull/11989), [#14848](https://github.com/NVIDIA/TensorRT-LLM/pull/14848) |
| `tensorrt_llm/serve/tool_parser/deepseekv31_parser.py` | no direct PR-number commit |
| `tensorrt_llm/serve/tool_parser/deepseekv32_parser.py` | [#10126](https://github.com/NVIDIA/TensorRT-LLM/pull/10126), [#11935](https://github.com/NVIDIA/TensorRT-LLM/pull/11935) |
| `tensorrt_llm/serve/tool_parser/deepseekv3_parser.py` | no direct PR-number commit |
| `tensorrt_llm/tokenizer/deepseek_v32/__init__.py` | [#9814](https://github.com/NVIDIA/TensorRT-LLM/pull/9814) |
| `tensorrt_llm/tokenizer/deepseek_v32/encoding.py` | [#9814](https://github.com/NVIDIA/TensorRT-LLM/pull/9814), [#11935](https://github.com/NVIDIA/TensorRT-LLM/pull/11935) |
| `tensorrt_llm/tokenizer/deepseek_v32/tokenizer.py` | [#9814](https://github.com/NVIDIA/TensorRT-LLM/pull/9814) |
| `tests/integration/defs/accuracy/test_modeling_v2_deepseek_v3.py` | no direct PR-number commit |
| `tests/integration/defs/disaggregated/test_configs/disagg_config_cache_aware_balance_deepseek_v3.yaml` | no direct PR-number commit |
| `tests/integration/defs/disaggregated/test_configs/disagg_config_cache_reuse_deepseek_v3.yaml` | no direct PR-number commit |
| `tests/integration/defs/disaggregated/test_configs/disagg_config_conditional_deepseek_v3.yaml` | no direct PR-number commit |
| `tests/integration/defs/disaggregated/test_configs/disagg_config_conditional_deepseek_v3_v2.yaml` | no direct PR-number commit |
| `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp1_gentp1_deepseek_v3_lite.yaml` | no direct PR-number commit |
| `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp1_gentp1_deepseek_v3_lite_one_mtp.yaml` | no direct PR-number commit |
| `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp1_gentp1_deepseek_v3_lite_one_mtp_attention_dp_overlap.yaml` | no direct PR-number commit |
| `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp1_gentp1_deepseek_v3_lite_one_mtp_ctxpp2_gentp2.yaml` | no direct PR-number commit |
| `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp1_gentp1_deepseek_v3_lite_two_mtp.yaml` | no direct PR-number commit |
| `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp1_gentp1_deepseek_v3_lite_ucx.yaml` | no direct PR-number commit |
| `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp2_gentp1cp2_deepseek_v3_lite_bf16_tllm_gen.yaml` | no direct PR-number commit |
| `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp2_gentp2_deepseek_v3_lite.yaml` | no direct PR-number commit |
| `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp2_gentp2_deepseek_v3_lite_attention_dp.yaml` | no direct PR-number commit |
| `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp2_gentp2_deepseek_v3_lite_attention_dp_one.yaml` | no direct PR-number commit |
| `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp2_gentp2_deepseek_v3_lite_attention_dp_one_mtp.yaml` | no direct PR-number commit |
| `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp2_gentp2_deepseek_v3_lite_attention_dp_overlap.yaml` | no direct PR-number commit |
| `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp2_gentp2_deepseek_v3_lite_attention_dp_overlap_cuda_graph.yaml` | no direct PR-number commit |
| `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp2_gentp2_deepseek_v3_lite_mpi.yaml` | no direct PR-number commit |
| `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp2_gentp2_deepseek_v3_lite_nixl.yaml` | no direct PR-number commit |
| `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp2_gentp2_deepseek_v3_lite_overlap_cuda_graph.yaml` | no direct PR-number commit |
| `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp2ep2pp2_gentp4_deepseek_v3_lite_one_mtp_block_reuse.yaml` | no direct PR-number commit |
| `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp2ep2pp2_gentp4_deepseek_v3_lite_one_mtp_block_reuse_chunked.yaml` | no direct PR-number commit |
| `tests/integration/defs/disaggregated/test_configs/disagg_config_deepseek_v3_lite_empty_batch.yaml` | no direct PR-number commit |
| `tests/integration/defs/disaggregated/test_configs/disagg_config_gentp2_deepseek_v3_lite_attention_dp_gen_only.yaml` | no direct PR-number commit |
| `tests/scripts/perf-sanity/aggregated/deepseek_v32_fp4_blackwell.yaml` | no direct PR-number commit |
| `tests/scripts/perf-sanity/aggregated/deepseek_v32_fp4_grace_blackwell.yaml` | no direct PR-number commit |
| `tests/scripts/perf-sanity/aggregated/dynamo_deepseek_v32_fp4_2_nodes_grace_blackwell.yaml` | no direct PR-number commit |
| `tests/scripts/perf-sanity/aggregated/host_perf_deepseek_v3_lite.yaml` | no direct PR-number commit |
| `tests/unittest/_torch/modeling/test_modeling_deepseekv3.py` | no direct PR-number commit |
| `tests/unittest/_torch/modeling/test_modeling_deepseekv32.py` | [#19332](https://github.com/NVIDIA/TensorRT-LLM/pull/19332) |

## PR Coverage Summary

- Git-traced PRs: 11
- Extra PRs preserved from existing docs: 0
- Total PRs in this document: 11
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2025-10-24 | [#8405](https://github.com/NVIDIA/TensorRT-LLM/pull/8405) | merged | [TRTLLM-8535][feat] Support DeepSeek V3.2 with FP8 + BF16 KV cache/NVFP4 + BF16 KV cache | `tensorrt_llm/_torch/configs/deepseek_v3.py`, `tensorrt_llm/_torch/models/modeling_deepseekv3.py` |
| 2025-11-14 | [#9141](https://github.com/NVIDIA/TensorRT-LLM/pull/9141) | merged | [None][doc] Add DeepSeek-V3.2-Exp document | `examples/models/core/deepseek_v3/README.md` |
| 2025-12-02 | [#9383](https://github.com/NVIDIA/TensorRT-LLM/pull/9383) | merged | [None][feat] Add support for KVCache reuse for DSv32 | `examples/models/core/deepseek_v3/README.md`, `tensorrt_llm/_torch/attention_backend/sparse/dsa.py`, `cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp` |
| 2025-12-19 | [#9814](https://github.com/NVIDIA/TensorRT-LLM/pull/9814) | merged | [TRTLLM-9654][feat] Support DeepSeek-V32 chat template | `tensorrt_llm/tokenizer/deepseek_v32/encoding.py`, `tensorrt_llm/llmapi/tokenizer.py`, `tensorrt_llm/tokenizer/tokenizer.py` |
| 2025-12-23 | [#10126](https://github.com/NVIDIA/TensorRT-LLM/pull/10126) | merged | [TRTLLM-9677][feat] Support DeepSeek-V3.2 tool parser | `tensorrt_llm/serve/tool_parser/deepseekv32_parser.py` |
| 2026-01-09 | [#10565](https://github.com/NVIDIA/TensorRT-LLM/pull/10565) | merged | [None][doc] blog: Optimizing DeepSeek-V3.2 on NVIDIA Blackwell GPUs | `docs/source/blogs/tech_blog/blog15_Optimizing_DeepSeek_V32_on_NVIDIA_Blackwell_GPUs.md` |
| 2026-03-06 | [#11507](https://github.com/NVIDIA/TensorRT-LLM/pull/11507) | merged | [TRTLLM-11057][feat] Add Helix CP support for DSV3.2 | `tensorrt_llm/_torch/models/modeling_deepseekv3.py` |
| 2026-03-10 | [#11935](https://github.com/NVIDIA/TensorRT-LLM/pull/11935) | merged | [https://nvbugs/5937478][fix] Fix DS v32 tool calling type and parse error | `tensorrt_llm/serve/tool_parser/deepseekv32_parser.py`, `tensorrt_llm/tokenizer/deepseek_v32/encoding.py` |
| 2026-03-17 | [#11989](https://github.com/NVIDIA/TensorRT-LLM/pull/11989) | merged | [TRTLLM-9521][feat] Unfuse indexer.wk from attention GEMM for DS-V3.2 NVFP4 | `tensorrt_llm/_torch/models/modeling_deepseekv3.py` |
| 2026-07-15 | [#14848](https://github.com/NVIDIA/TensorRT-LLM/pull/14848) | merged | [TRTLLM-12373][feat] RMSNorm nvfp4 quant fusion for DS V3.2 / Kimi-K2.5 | `tensorrt_llm/_torch/models/modeling_deepseekv3.py` |
| 2026-09-21 | [#19332](https://github.com/NVIDIA/TensorRT-LLM/pull/19332) | merged | [None][test] Add coverage for DeepseekV32ForCausalLM | `tests/unittest/_torch/modeling/test_modeling_deepseekv32.py` |

## Per-PR Diff Audit Cards

### PR #8405 - [TRTLLM-8535][feat] Support DeepSeek V3.2 with FP8 + BF16 KV cache/NVFP4 + BF16 KV cache

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/8405
- Status/date: merged / 2025-10-24
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/configs/deepseek_v3.py`, `tensorrt_llm/_torch/models/modeling_deepseekv3.py`; associated commits `e47c787dd702`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 43 files, +4914/-153, 5772 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/configs/deepseek_v3.py` added +103/-0 (103 lines); hunks: -0,0 +1,103; symbols: DeepseekV3Config, __init__, touching `DeepseekV3Config, __init__`; `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +1/-0 (1 lines); hunks: -1468,6 +1468,7 @@ def forward(; symbols: forward, DeepseekV3ForCausalLM, touching `forward, DeepseekV3ForCausalLM`.
- Code diff details:
  - `tensorrt_llm/_torch/configs/deepseek_v3.py` added +103/-0 (103 lines); hunks: -0,0 +1,103; symbols: DeepseekV3Config, __init__
  - `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +1/-0 (1 lines); hunks: -1468,6 +1468,7 @@ def forward(; symbols: forward, DeepseekV3ForCausalLM
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/configs/deepseek_v3.py
@@ -0,0 +1,103 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from transformers.configuration_utils import PretrainedConfig
+from transformers.utils import logging
+logger = logging.get_logger(__name__)
+# This is a temporary workaround for DeepSeek-V3.2 model as HF does not support it yet
diff -- tensorrt_llm/_torch/models/modeling_deepseekv3.py
@@ -1468,6 +1468,7 @@ def forward(
+@register_auto_model("DeepseekV32ForCausalLM")
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/configs/deepseek_v3.py` added +103/-0; `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +1/-0
- Risk and verification: The diff ships test coverage in `tests/integration/defs/accuracy/references/gpqa_diamond.yaml`, `tests/integration/defs/accuracy/references/gsm8k.yaml`, `tests/integration/defs/accuracy/references/mmlu.yaml`, `tests/integration/defs/accuracy/test_llm_api_pytorch.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #9141 - [None][doc] Add DeepSeek-V3.2-Exp document

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/9141
- Status/date: merged / 2025-11-14
- Trace source: `git log --name-only -- <model-files>` found it through `examples/models/core/deepseek_v3/README.md`; associated commits `25bd2e691790`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +21/-10, 88 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `examples/models/core/deepseek_v3/README.md` modified +21/-10 (31 lines); hunks: -1,7 +1,8; -14,7 +15,7 @@ Please refer to [this guide](https://nvidia.github.io/TensorRT....
- Code diff details:
  - `examples/models/core/deepseek_v3/README.md` modified +21/-10 (31 lines); hunks: -1,7 +1,8; -14,7 +15,7 @@ Please refer to [this guide](https://nvidia.github.io/TensorRT...
- Key code excerpts:

```diff
diff -- examples/models/core/deepseek_v3/README.md
@@ -1,7 +1,8 @@
-# DeepSeek‑V3 and DeepSeek-R1
+# DeepSeek‑V3, DeepSeek-R1, and DeepSeek-V3.2-Exp
+This guide walks you through the examples to run the DeepSeek‑V3/DeepSeek-R1/DeepSeek-V3.2-Exp models using NVIDIA's TensorRT LLM framework with the PyTorch backend.
+**DeepSeek-R1 and DeepSeek-V3 share exact same model architecture other than weights differences, and share same code path in TensorRT LLM. DeepSeek-V3.2-Exp features DeepSeek Spa
-This guide walks you through the examples to run the DeepSeek‑V3/DeepSeek-R1 models using NVIDIA's TensorRT LLM framework with the PyTorch backend.
-**DeepSeek-R1 and DeepSeek-V3 share exact same model architecture other than weights differences, and share same code path in TensorRT-LLM, for brevity we only provide one model e
```

- Extracted files (not manually reviewed):
  - docs: `examples/models/core/deepseek_v3/README.md` modified +21/-10
- Risk and verification: This is mostly docs/examples in `examples/models/core/deepseek_v3/README.md`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #9383 - [None][feat] Add support for KVCache reuse for DSv32

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/9383
- Status/date: merged / 2025-12-02
- Trace source: `git log --name-only -- <model-files>` found it through `examples/models/core/deepseek_v3/README.md`; associated commits `356a52edf56e`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 7 files, +14/-38, 146 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `examples/models/core/deepseek_v3/README.md` modified +0/-9 (9 lines); hunks: -881,12 +881,3 @@ python quickstart_advanced.py --model_dir --enable_chunked_...; `tensorrt_llm/_torch/attention_backend/sparse/dsa.py` modified +4/-8 (12 lines); hunks: -930,15 +930,15 @@ def prepare(metadata: DSAtrtllmAttentionMetadata):; -1018,9 +1018,9 @@ def prepare(metadata: DSAtrtllmAttentionMetadata):; symbols: prepare, __init__, touching `prepare, __init__`; `cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp` modified +1/-8 (9 lines); hunks: -876,14 +876,7 @@ void WindowBlockManager::allocatePools(bool useUvm); `cpp/include/tensorrt_llm/batch_manager/kvCacheUtils.h` modified +4/-0 (4 lines); hunks: -183,6 +183,10 @@ class BlockRange; symbols: BlockRange, touching `BlockRange`.
- Code diff details:
  - `examples/models/core/deepseek_v3/README.md` modified +0/-9 (9 lines); hunks: -881,12 +881,3 @@ python quickstart_advanced.py --model_dir --enable_chunked_...
  - `tensorrt_llm/_torch/attention_backend/sparse/dsa.py` modified +4/-8 (12 lines); hunks: -930,15 +930,15 @@ def prepare(metadata: DSAtrtllmAttentionMetadata):; -1018,9 +1018,9 @@ def prepare(metadata: DSAtrtllmAttentionMetadata):; symbols: prepare, __init__
  - `cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp` modified +1/-8 (9 lines); hunks: -876,14 +876,7 @@ void WindowBlockManager::allocatePools(bool useUvm)
  - `cpp/include/tensorrt_llm/batch_manager/kvCacheUtils.h` modified +4/-0 (4 lines); hunks: -183,6 +183,10 @@ class BlockRange; symbols: BlockRange
  - `cpp/tensorrt_llm/batch_manager/dataTransceiver.cpp` modified +1/-1 (2 lines); hunks: -806,7 +806,7 @@ class CacheReceiver::Impl; symbols: CacheReceiver
- Key code excerpts:

```diff
diff -- examples/models/core/deepseek_v3/README.md
@@ -881,12 +881,3 @@ python quickstart_advanced.py --model_dir <YOUR_MODEL_DIR> --enable_chunked_pref
-## Known Issues
-- Support for KV Cache Reuse and Chunked Prefill in DeepSeek-V3.2-Exp is currently under development. When running `quickstart_advanced.py`, please include `--disable_kv_cache_reu
-'''
-kv_cache_config:
-    enable_block_reuse: false
-    tokens_per_block: 64
diff -- tensorrt_llm/_torch/attention_backend/sparse/dsa.py
@@ -930,15 +930,15 @@ def prepare(metadata: DSAtrtllmAttentionMetadata):
-                if len(chunk_groups) > 1:
+                if len(chunk_groups
+                       ) > 1 or metadata.enable_context_mla_with_cached_kv:
-                    # Single chunk - use non-chunked fallback path
@@ -1018,9 +1018,9 @@ def prepare(metadata: DSAtrtllmAttentionMetadata):
-        # Only when MLA chunked prefill is enabled, we need to gather the full KV for indexer's logit computation.
diff -- cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp
@@ -876,14 +876,7 @@ void WindowBlockManager::allocatePools(bool useUvm)
```

- Extracted files (not manually reviewed):
  - docs: `examples/models/core/deepseek_v3/README.md` modified +0/-9
  - runtime: `tensorrt_llm/_torch/attention_backend/sparse/dsa.py` modified +4/-8; `cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp` modified +1/-8; `cpp/include/tensorrt_llm/batch_manager/kvCacheUtils.h` modified +4/-0; `cpp/tensorrt_llm/batch_manager/dataTransceiver.cpp` modified +1/-1
- Risk and verification: The diff ships test coverage in `tests/integration/defs/accuracy/test_disaggregated_serving.py`, `tests/integration/defs/accuracy/test_llm_api_pytorch.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #9814 - [TRTLLM-9654][feat] Support DeepSeek-V32 chat template

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/9814
- Status/date: merged / 2025-12-19
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/tokenizer/deepseek_v32/__init__.py`, `tensorrt_llm/tokenizer/deepseek_v32/encoding.py`, `tensorrt_llm/tokenizer/deepseek_v32/tokenizer.py`; associated commits `31bc14b3507d`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 12 files, +1098/-389, 1591 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/tokenizer/deepseek_v32/encoding.py` added +425/-0 (425 lines); hunks: -0,0 +1,425; symbols: to_json, tools_from_openai_format, tool_calls_from_openai_format, tool_calls_to_openai_format, touching `to_json, tools_from_openai_format, tool_calls_from_openai_format`; `tensorrt_llm/llmapi/tokenizer.py` modified +21/-365 (386 lines); hunks: -1,365 +1,21; symbols: TokenizerBase, TransformersTokenizer, __init__, __call__, touching `TokenizerBase, TransformersTokenizer, __init__`; `tensorrt_llm/tokenizer/tokenizer.py` added +365/-0 (365 lines); hunks: -0,0 +1,365; symbols: TokenizerBase, TransformersTokenizer, __init__, __call__, touching `TokenizerBase, TransformersTokenizer, __init__`; `tensorrt_llm/tokenizer/deepseek_v32/tokenizer.py` added +147/-0 (147 lines); hunks: -0,0 +1,147; symbols: DeepseekV32Tokenizer, __init__, from_pretrained, apply_chat_template, touching `DeepseekV32Tokenizer, __init__, from_pretrained`.
- Code diff details:
  - `tensorrt_llm/tokenizer/deepseek_v32/encoding.py` added +425/-0 (425 lines); hunks: -0,0 +1,425; symbols: to_json, tools_from_openai_format, tool_calls_from_openai_format, tool_calls_to_openai_format
  - `tensorrt_llm/llmapi/tokenizer.py` modified +21/-365 (386 lines); hunks: -1,365 +1,21; symbols: TokenizerBase, TransformersTokenizer, __init__, __call__
  - `tensorrt_llm/tokenizer/tokenizer.py` added +365/-0 (365 lines); hunks: -0,0 +1,365; symbols: TokenizerBase, TransformersTokenizer, __init__, __call__
  - `tensorrt_llm/tokenizer/deepseek_v32/tokenizer.py` added +147/-0 (147 lines); hunks: -0,0 +1,147; symbols: DeepseekV32Tokenizer, __init__, from_pretrained, apply_chat_template
  - `tensorrt_llm/tokenizer/__init__.py` added +21/-0 (21 lines); hunks: -0,0 +1,21
- Key code excerpts:

```diff
diff -- tensorrt_llm/tokenizer/deepseek_v32/encoding.py
@@ -0,0 +1,425 @@
+# copy from https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/encoding/encoding_dsv32.py
+# ruff: noqa: E501
+import copy
+import json
+import re
+from typing import Any, Dict, List, Optional, Tuple, Union
diff -- tensorrt_llm/llmapi/tokenizer.py
@@ -1,365 +1,21 @@
-import os
-from pathlib import Path
-from typing import Any, Dict, List, Optional, Tuple, Union
-from transformers import (AutoTokenizer, PreTrainedTokenizerBase,
-                          PreTrainedTokenizerFast)
-from .._utils import nvtx_range_debug
diff -- tensorrt_llm/tokenizer/tokenizer.py
@@ -0,0 +1,365 @@
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/tokenizer/deepseek_v32/encoding.py` added +425/-0; `tensorrt_llm/llmapi/tokenizer.py` modified +21/-365; `tensorrt_llm/tokenizer/tokenizer.py` added +365/-0; `tensorrt_llm/tokenizer/deepseek_v32/tokenizer.py` added +147/-0; `tensorrt_llm/tokenizer/__init__.py` added +21/-0; `tensorrt_llm/tokenizer/deepseek_v32/__init__.py` added +14/-0
- Risk and verification: The diff ships test coverage in `tests/unittest/api_stability/references/llm.yaml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #10126 - [TRTLLM-9677][feat] Support DeepSeek-V3.2 tool parser

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/10126
- Status/date: merged / 2025-12-23
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/serve/tool_parser/deepseekv32_parser.py`; associated commits `0d2500c631d2`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 4 files, +444/-0, 480 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/serve/tool_parser/deepseekv32_parser.py` added +296/-0 (296 lines); hunks: -0,0 +1,296; symbols: DeepSeekV32Parser, __init__, has_tool_call, _parse_parameters_from_xml, touching `DeepSeekV32Parser, __init__, has_tool_call`.
- Code diff details:
  - `tensorrt_llm/serve/tool_parser/deepseekv32_parser.py` added +296/-0 (296 lines); hunks: -0,0 +1,296; symbols: DeepSeekV32Parser, __init__, has_tool_call, _parse_parameters_from_xml
- Key code excerpts:

```diff
diff -- tensorrt_llm/serve/tool_parser/deepseekv32_parser.py
@@ -0,0 +1,296 @@
+# Adapted from https://github.com/sgl-project/sglang/blob/0071fe9c407ad59f2803cc319e1bcaa3ac2021f1/python/sglang/srt/function_call/deepseekv32_detector.py
+import json
+import re
+from typing import List
+from tensorrt_llm.logger import logger
+from ..openai_protocol import ChatCompletionToolsParam as Tool
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/serve/tool_parser/deepseekv32_parser.py` added +296/-0
- Risk and verification: The diff ships test coverage in `tests/unittest/llmapi/apps/test_tool_parsers.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #10565 - [None][doc] blog: Optimizing DeepSeek-V3.2 on NVIDIA Blackwell GPUs

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/10565
- Status/date: merged / 2026-01-09
- Trace source: `git log --name-only -- <model-files>` found it through `docs/source/blogs/tech_blog/blog15_Optimizing_DeepSeek_V32_on_NVIDIA_Blackwell_GPUs.md`; associated commits `4632a8642d4d`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 5 files, +423/-0, 424 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `docs/source/blogs/tech_blog/blog15_Optimizing_DeepSeek_V32_on_NVIDIA_Blackwell_GPUs.md` added +423/-0 (423 lines); hunks: -0,0 +1,423.
- Code diff details:
  - `docs/source/blogs/tech_blog/blog15_Optimizing_DeepSeek_V32_on_NVIDIA_Blackwell_GPUs.md` added +423/-0 (423 lines); hunks: -0,0 +1,423
- Key code excerpts:

```diff
diff -- docs/source/blogs/tech_blog/blog15_Optimizing_DeepSeek_V32_on_NVIDIA_Blackwell_GPUs.md
@@ -0,0 +1,423 @@
+# Optimizing DeepSeek-V3.2 on NVIDIA Blackwell GPUs
+By NVIDIA TensorRT LLM team
+## Table of Contents
+- [Optimizing DeepSeek-V3.2 on NVIDIA Blackwell GPUs](#optimizing-deepseek-v32-on-nvidia-blackwell-gpus)
+    - [Table of Contents](#table-of-contents)
+    - [Introduction](#introduction)
```

- Extracted files (not manually reviewed):
  - docs: `docs/source/blogs/tech_blog/blog15_Optimizing_DeepSeek_V32_on_NVIDIA_Blackwell_GPUs.md` added +423/-0
- Risk and verification: This is mostly docs/examples in `docs/source/blogs/tech_blog/blog15_Optimizing_DeepSeek_V32_on_NVIDIA_Blackwell_GPUs.md`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #11507 - [TRTLLM-11057][feat] Add Helix CP support for DSV3.2

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/11507
- Status/date: merged / 2026-03-06
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_deepseekv3.py`; associated commits `ac8bc6ed1112`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 7 files, +178/-26, 363 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +13/-11 (24 lines); hunks: -764,6 +764,7 @@ def __init__(; -789,6 +790,7 @@ def __init__(; symbols: __init__, _compute_shared_expert_tp_size, _create_ideal_expert_load_balanced_logits, touching `__init__, _compute_shared_expert_tp_size, _create_ideal_expert_load_balanced_logits`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +13/-11 (24 lines); hunks: -764,6 +764,7 @@ def __init__(; -789,6 +790,7 @@ def __init__(; symbols: __init__, _compute_shared_expert_tp_size, _create_ideal_expert_load_balanced_logits
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_deepseekv3.py
@@ -764,6 +764,7 @@ def __init__(
+        mapping_with_cp: Optional[Mapping] = None,
@@ -789,6 +790,7 @@ def __init__(
+                         mapping_with_cp=mapping_with_cp,
@@ -1008,7 +1010,7 @@ def __init__(self,
-                dtype=dtype)
+                dtype=torch.float32)
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +13/-11
- Risk and verification: The diff ships test coverage in `cpp/tests/unit_tests/multi_gpu/cacheTransceiverTest.cpp`, `tests/integration/defs/accuracy/test_disaggregated_serving.py`, `tests/integration/test_lists/qa/llm_function_core.txt`, `tests/integration/test_lists/test-db/l0_dgx_b200.yml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #11935 - [https://nvbugs/5937478][fix] Fix DS v32 tool calling type and parse error

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/11935
- Status/date: merged / 2026-03-10
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/serve/tool_parser/deepseekv32_parser.py`, `tensorrt_llm/tokenizer/deepseek_v32/encoding.py`; associated commits `a20de8832494`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 4 files, +18/-2, 69 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/serve/tool_parser/deepseekv32_parser.py` modified +7/-1 (8 lines); hunks: -61,6 +61,10 @@ class DeepSeekV32Parser(BaseToolParser):; -118,6 +122,8 @@ def detect_and_parse(self, text: str, tools: List[Tool]) ->...; symbols: DeepSeekV32Parser, __init__, detect_and_parse, parse_streaming_increment, touching `DeepSeekV32Parser, __init__, detect_and_parse`; `tensorrt_llm/tokenizer/deepseek_v32/encoding.py` modified +2/-1 (3 lines); hunks: -91,7 +91,8 @@ def encode_arguments_to_dsml(tool_call: Dict[str, str]) -> str:; symbols: encode_arguments_to_dsml, touching `encode_arguments_to_dsml`.
- Code diff details:
  - `tensorrt_llm/serve/tool_parser/deepseekv32_parser.py` modified +7/-1 (8 lines); hunks: -61,6 +61,10 @@ class DeepSeekV32Parser(BaseToolParser):; -118,6 +122,8 @@ def detect_and_parse(self, text: str, tools: List[Tool]) ->...; symbols: DeepSeekV32Parser, __init__, detect_and_parse, parse_streaming_increment
  - `tensorrt_llm/tokenizer/deepseek_v32/encoding.py` modified +2/-1 (3 lines); hunks: -91,7 +91,8 @@ def encode_arguments_to_dsml(tool_call: Dict[str, str]) -> str:; symbols: encode_arguments_to_dsml
- Key code excerpts:

```diff
diff -- tensorrt_llm/serve/tool_parser/deepseekv32_parser.py
@@ -61,6 +61,10 @@ class DeepSeekV32Parser(BaseToolParser):
+    needs_raw_special_tokens = True
+    _eos_token = "<｜end▁of▁sentence｜>"  # nosec B105
@@ -118,6 +122,8 @@ def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult
+        if self._eos_token in text:
+            text = text.replace(self._eos_token, "")
@@ -177,7 +183,7 @@ def parse_streaming_increment(self, new_text: str, tools: List[Tool]) -> Streami
diff -- tensorrt_llm/tokenizer/deepseek_v32/encoding.py
@@ -91,7 +91,8 @@ def encode_arguments_to_dsml(tool_call: Dict[str, str]) -> str:
-    arguments = json.loads(tool_call["arguments"])
+    raw_args = tool_call["arguments"]
+    arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/serve/tool_parser/deepseekv32_parser.py` modified +7/-1; `tensorrt_llm/tokenizer/deepseek_v32/encoding.py` modified +2/-1
- Risk and verification: Runtime changes concentrate in `tensorrt_llm/serve/openai_server.py`, `tensorrt_llm/serve/tool_parser/base_tool_parser.py`, `tensorrt_llm/serve/tool_parser/deepseekv32_parser.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #11989 - [TRTLLM-9521][feat] Unfuse indexer.wk from attention GEMM for DS-V3.2 NVFP4

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/11989
- Status/date: merged / 2026-03-17
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_deepseekv3.py`; associated commits `2f45640c1991`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 6 files, +37/-46, 174 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +7/-39 (46 lines); hunks: -414,7 +414,12 @@ def split_kv_b_proj(kv_b_proj: torch.Tensor,; -497,8 +502,6 @@ def split_kv_b_proj(kv_b_proj: torch.Tensor,; symbols: split_kv_b_proj, __init__, touching `split_kv_b_proj, __init__`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +7/-39 (46 lines); hunks: -414,7 +414,12 @@ def split_kv_b_proj(kv_b_proj: torch.Tensor,; -497,8 +502,6 @@ def split_kv_b_proj(kv_b_proj: torch.Tensor,; symbols: split_kv_b_proj, __init__
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_deepseekv3.py
@@ -414,7 +414,12 @@ def split_kv_b_proj(kv_b_proj: torch.Tensor,
-                        f"{'.'.join(names[:-1])}.kv_a_proj_with_mqa.weight"].dtype == fp4_utils.float4_e2m1x2 and weights[
+                        f"{'.'.join(names[:-1])}.kv_a_proj_with_mqa.weight"].dtype == fp4_utils.float4_e2m1x2
+                    # Non-lite models (V3, R1, V3.2) fuse q_a_proj into
+                    # kv_a_proj_with_mqa, so both must be NVFP4 for the fused
+                    # path. Lite models (V3-Lite) have no q_a_proj.
+                    if not is_lite:
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +7/-39
- Risk and verification: The diff ships test coverage in `tests/integration/defs/accuracy/references/gsm8k.yaml`, `tests/integration/defs/accuracy/references/mmlu.yaml`, `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/integration/test_lists/test-db/l0_dgx_b200.yml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #14848 - [TRTLLM-12373][feat] RMSNorm nvfp4 quant fusion for DS V3.2 / Kimi-K2.5

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/14848
- Status/date: merged / 2026-07-15
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_deepseekv3.py`; associated commits `501777ac89b1`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 16 files, +1993/-101, 2418 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +98/-12 (110 lines); hunks: -66,7 +66,8; -145,6 +146,18 @@ def moe_reduce_add_shared_output(routed_output, shared_outp...; symbols: moe_reduce_add_shared_output, _static_nvfp4_input_scale, DeepseekV3WeightLoader, __init__, touching `moe_reduce_add_shared_output, _static_nvfp4_input_scale, DeepseekV3WeightLoader`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +98/-12 (110 lines); hunks: -66,7 +66,8; -145,6 +146,18 @@ def moe_reduce_add_shared_output(routed_output, shared_outp...; symbols: moe_reduce_add_shared_output, _static_nvfp4_input_scale, DeepseekV3WeightLoader, __init__
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_deepseekv3.py
@@ -66,7 +66,8 @@
-from ..modules.linear import Linear, TensorParallelMode, WeightsLoadingConfig
+from ..modules.linear import (Linear, TensorParallelMode, WeightsLoadingConfig,
+                              is_static_nvfp4_input_eligible)
@@ -145,6 +146,18 @@ def moe_reduce_add_shared_output(routed_output, shared_output, out=None):
+def _static_nvfp4_input_scale(linear):
+    """Return `linear`'s calibrated NVFP4 input_scale if it is eligible to be
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_deepseekv3.py` modified +98/-12
- Risk and verification: The diff ships test coverage in `tests/integration/test_lists/test-db/l0_b200.yml`, `tests/unittest/_torch/modules/test_fp4_num_tokens_slice.py`, `tests/unittest/_torch/modules/test_fused_rmsnorm_fp4_quantize.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #19332 - [None][test] Add coverage for DeepseekV32ForCausalLM

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/19332
- Status/date: merged / 2026-09-21
- Trace source: `git log --name-only -- <model-files>` found it through `tests/unittest/_torch/modeling/test_modeling_deepseekv32.py`; associated commits `a7eae4beee5b`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +202/-0, 210 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/unittest/_torch/modeling/test_modeling_deepseekv32.py` added +201/-0 (201 lines); hunks: -0,0 +1,201; symbols: test_deepseek_v32_context_forward, fresh_metadata, touching `test_deepseek_v32_context_forward, fresh_metadata`.
- Code diff details:
  - `tests/unittest/_torch/modeling/test_modeling_deepseekv32.py` added +201/-0 (201 lines); hunks: -0,0 +1,201; symbols: test_deepseek_v32_context_forward, fresh_metadata
- Key code excerpts:

```diff
diff -- tests/unittest/_torch/modeling/test_modeling_deepseekv32.py
@@ -0,0 +1,201 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
```

- Extracted files (not manually reviewed):
  - tests: `tests/unittest/_torch/modeling/test_modeling_deepseekv32.py` added +201/-0
- Risk and verification: The diff ships test coverage in `tests/integration/test_lists/test-db/l0_h100.yml`, `tests/unittest/_torch/modeling/test_modeling_deepseekv32.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
