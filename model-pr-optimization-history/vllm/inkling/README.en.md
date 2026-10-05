# vLLM Inkling Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `benchmarks/kernels/benchmark_inkling_qkvr_prep.py` | [#48841](https://github.com/vllm-project/vllm/pull/48841) |
| `tests/models/inkling/rocm/conftest.py` | [#48841](https://github.com/vllm-project/vllm/pull/48841) |
| `tests/models/inkling/rocm/test_model_alignment.py` | [#48841](https://github.com/vllm-project/vllm/pull/48841) |
| `tests/models/inkling/rocm/test_mxfp4_load.py` | [#48841](https://github.com/vllm-project/vllm/pull/48841) |
| `tests/models/inkling/rocm/test_rel_attention.py` | [#48841](https://github.com/vllm-project/vllm/pull/48841) |
| `tests/models/inkling/rocm/test_rocm_mtp_input_fusion.py` | [#48841](https://github.com/vllm-project/vllm/pull/48841) |
| `tests/models/inkling/rocm/test_sconv_cache_layout.py` | [#48841](https://github.com/vllm-project/vllm/pull/48841) |
| `tests/models/inkling/test_contract_validation.py` | [#48799](https://github.com/vllm-project/vllm/pull/48799), [#48822](https://github.com/vllm-project/vllm/pull/48822), [#48869](https://github.com/vllm-project/vllm/pull/48869), [#49485](https://github.com/vllm-project/vllm/pull/49485) |
| `tests/models/inkling/test_fa4_rel_attention.py` | [#48799](https://github.com/vllm-project/vllm/pull/48799), [#48858](https://github.com/vllm-project/vllm/pull/48858), [#49315](https://github.com/vllm-project/vllm/pull/49315) |
| `tests/models/inkling/test_fa4_warmup.py` | [#48799](https://github.com/vllm-project/vllm/pull/48799), [#49315](https://github.com/vllm-project/vllm/pull/49315) |
| `tests/models/inkling/test_mm_towers.py` | [#48799](https://github.com/vllm-project/vllm/pull/48799) |
| `tests/models/inkling/test_moe_weight_layout.py` | [#48799](https://github.com/vllm-project/vllm/pull/48799), [#48876](https://github.com/vllm-project/vllm/pull/48876), [#48884](https://github.com/vllm-project/vllm/pull/48884), [#48990](https://github.com/vllm-project/vllm/pull/48990), [#49258](https://github.com/vllm-project/vllm/pull/49258) |
| `tests/models/inkling/test_mtp_input_fusion.py` | [#48869](https://github.com/vllm-project/vllm/pull/48869) |
| `tests/models/inkling/test_qkvr_prep.py` | [#48799](https://github.com/vllm-project/vllm/pull/48799) |
| `tests/models/inkling/test_sconv_metadata.py` | [#48799](https://github.com/vllm-project/vllm/pull/48799) |
| `tests/parser/engine/test_inkling.py` | [#48799](https://github.com/vllm-project/vllm/pull/48799), [#49876](https://github.com/vllm-project/vllm/pull/49876), [#50403](https://github.com/vllm-project/vllm/pull/50403), [#50528](https://github.com/vllm-project/vllm/pull/50528), [#51391](https://github.com/vllm-project/vllm/pull/51391), [#58792](https://github.com/vllm-project/vllm/pull/58792) |
| `tests/renderers/test_inkling.py` | [#48799](https://github.com/vllm-project/vllm/pull/48799) |
| `vllm/models/inkling/__init__.py` | [#48799](https://github.com/vllm-project/vllm/pull/48799), [#48841](https://github.com/vllm-project/vllm/pull/48841), [#48869](https://github.com/vllm-project/vllm/pull/48869) |
| `vllm/models/inkling/amd/__init__.py` | [#48841](https://github.com/vllm-project/vllm/pull/48841) |
| `vllm/models/inkling/amd/attention.py` | [#48841](https://github.com/vllm-project/vllm/pull/48841) |
| `vllm/models/inkling/amd/layernorm.py` | [#48841](https://github.com/vllm-project/vllm/pull/48841) |
| `vllm/models/inkling/amd/logits_processor.py` | [#48841](https://github.com/vllm-project/vllm/pull/48841) |
| `vllm/models/inkling/amd/mlp.py` | [#48841](https://github.com/vllm-project/vllm/pull/48841) |
| `vllm/models/inkling/amd/model.py` | [#48841](https://github.com/vllm-project/vllm/pull/48841) |
| `vllm/models/inkling/amd/moe.py` | [#48841](https://github.com/vllm-project/vllm/pull/48841) |
| `vllm/models/inkling/amd/mtp.py` | [#48841](https://github.com/vllm-project/vllm/pull/48841), [#50806](https://github.com/vllm-project/vllm/pull/50806) |
| `vllm/models/inkling/amd/ops/__init__.py` | [#48841](https://github.com/vllm-project/vllm/pull/48841) |
| `vllm/models/inkling/amd/ops/fa4_rel_attention.py` | [#48841](https://github.com/vllm-project/vllm/pull/48841) |
| `vllm/models/inkling/amd/ops/fa4_warmup.py` | [#48841](https://github.com/vllm-project/vllm/pull/48841) |
| `vllm/models/inkling/amd/ops/gluon/__init__.py` | [#48841](https://github.com/vllm-project/vllm/pull/48841) |
| `vllm/models/inkling/amd/ops/gluon/rel_mha_decode_gfx950.py` | [#48841](https://github.com/vllm-project/vllm/pull/48841) |
| `vllm/models/inkling/amd/ops/gluon/rel_mha_extend_gfx950.py` | [#48841](https://github.com/vllm-project/vllm/pull/48841) |
| `vllm/models/inkling/amd/ops/gluon/utils.py` | [#48841](https://github.com/vllm-project/vllm/pull/48841) |
| `vllm/models/inkling/amd/ops/norm.py` | [#48841](https://github.com/vllm-project/vllm/pull/48841) |
| `vllm/models/inkling/amd/ops/qkvr_prep.py` | [#48841](https://github.com/vllm-project/vllm/pull/48841) |
| `vllm/models/inkling/amd/ops/rel_attention_decode.py` | [#48841](https://github.com/vllm-project/vllm/pull/48841) |
| `vllm/models/inkling/amd/ops/sconv.py` | [#48841](https://github.com/vllm-project/vllm/pull/48841) |
| `vllm/models/inkling/amd/ops/silu_and_mul.py` | [#48841](https://github.com/vllm-project/vllm/pull/48841) |
| `vllm/models/inkling/amd/sconv_swa_attn.py` | [#48841](https://github.com/vllm-project/vllm/pull/48841) |
| `vllm/models/inkling/amd/short_conv.py` | [#48841](https://github.com/vllm-project/vllm/pull/48841) |
| `vllm/models/inkling/common/__init__.py` | [#48799](https://github.com/vllm-project/vllm/pull/48799) |
| `vllm/models/inkling/common/mm_preprocess.py` | [#48799](https://github.com/vllm-project/vllm/pull/48799) |
| `vllm/models/inkling/common/towers.py` | [#48799](https://github.com/vllm-project/vllm/pull/48799), [#49485](https://github.com/vllm-project/vllm/pull/49485) |
| `vllm/models/inkling/configs.py` | [#48799](https://github.com/vllm-project/vllm/pull/48799), [#48869](https://github.com/vllm-project/vllm/pull/48869), [#51850](https://github.com/vllm-project/vllm/pull/51850) |
| `vllm/models/inkling/nvidia/__init__.py` | [#48799](https://github.com/vllm-project/vllm/pull/48799) |
| `vllm/models/inkling/nvidia/attention.py` | [#48799](https://github.com/vllm-project/vllm/pull/48799), [#48822](https://github.com/vllm-project/vllm/pull/48822), [#49315](https://github.com/vllm-project/vllm/pull/49315) |
| `vllm/models/inkling/nvidia/layernorm.py` | [#48799](https://github.com/vllm-project/vllm/pull/48799) |
| `vllm/models/inkling/nvidia/logits_processor.py` | [#48799](https://github.com/vllm-project/vllm/pull/48799), [#48884](https://github.com/vllm-project/vllm/pull/48884) |
| `vllm/models/inkling/nvidia/mlp.py` | [#48799](https://github.com/vllm-project/vllm/pull/48799), [#49487](https://github.com/vllm-project/vllm/pull/49487) |
| `vllm/models/inkling/nvidia/model.py` | [#48799](https://github.com/vllm-project/vllm/pull/48799), [#48869](https://github.com/vllm-project/vllm/pull/48869), [#48884](https://github.com/vllm-project/vllm/pull/48884), [#48990](https://github.com/vllm-project/vllm/pull/48990), [#49258](https://github.com/vllm-project/vllm/pull/49258), [#50697](https://github.com/vllm-project/vllm/pull/50697) |
| `vllm/models/inkling/nvidia/moe.py` | [#48799](https://github.com/vllm-project/vllm/pull/48799), [#48876](https://github.com/vllm-project/vllm/pull/48876), [#48884](https://github.com/vllm-project/vllm/pull/48884), [#48990](https://github.com/vllm-project/vllm/pull/48990), [#49258](https://github.com/vllm-project/vllm/pull/49258), [#49487](https://github.com/vllm-project/vllm/pull/49487), [#50697](https://github.com/vllm-project/vllm/pull/50697) |
| `vllm/models/inkling/nvidia/mtp.py` | [#48869](https://github.com/vllm-project/vllm/pull/48869), [#48990](https://github.com/vllm-project/vllm/pull/48990) |
| `vllm/models/inkling/nvidia/ops/__init__.py` | [#48799](https://github.com/vllm-project/vllm/pull/48799), [#49315](https://github.com/vllm-project/vllm/pull/49315) |
| `vllm/models/inkling/nvidia/ops/fa4_rel_attention.py` | [#48799](https://github.com/vllm-project/vllm/pull/48799), [#48858](https://github.com/vllm-project/vllm/pull/48858), [#49315](https://github.com/vllm-project/vllm/pull/49315) |
| `vllm/models/inkling/nvidia/ops/lamport.py` | [#48799](https://github.com/vllm-project/vllm/pull/48799), [#50697](https://github.com/vllm-project/vllm/pull/50697) |
| `vllm/models/inkling/nvidia/ops/mm_towers.py` | [#48799](https://github.com/vllm-project/vllm/pull/48799) |
| `vllm/models/inkling/nvidia/ops/norm.py` | [#48799](https://github.com/vllm-project/vllm/pull/48799), [#48869](https://github.com/vllm-project/vllm/pull/48869) |
| `vllm/models/inkling/nvidia/ops/qkvr_prep.py` | [#48799](https://github.com/vllm-project/vllm/pull/48799) |
| `vllm/models/inkling/nvidia/ops/sconv.py` | [#48799](https://github.com/vllm-project/vllm/pull/48799), [#48822](https://github.com/vllm-project/vllm/pull/48822) |
| `vllm/models/inkling/nvidia/ops/silu_and_mul.py` | [#48799](https://github.com/vllm-project/vllm/pull/48799) |
| `vllm/models/inkling/nvidia/sconv_swa_attn.py` | [#48799](https://github.com/vllm-project/vllm/pull/48799), [#48822](https://github.com/vllm-project/vllm/pull/48822) |
| `vllm/models/inkling/nvidia/short_conv.py` | [#48799](https://github.com/vllm-project/vllm/pull/48799), [#48822](https://github.com/vllm-project/vllm/pull/48822) |
| `vllm/parser/inkling.py` | [#48799](https://github.com/vllm-project/vllm/pull/48799), [#49876](https://github.com/vllm-project/vllm/pull/49876), [#50403](https://github.com/vllm-project/vllm/pull/50403), [#50528](https://github.com/vllm-project/vllm/pull/50528), [#58792](https://github.com/vllm-project/vllm/pull/58792) |
| `vllm/reasoning/inkling_reasoning_parser.py` | [#48799](https://github.com/vllm-project/vllm/pull/48799) |
| `vllm/renderers/inkling.py` | [#48799](https://github.com/vllm-project/vllm/pull/48799) |
| `vllm/renderers/inkling_encoding.py` | [#48799](https://github.com/vllm-project/vllm/pull/48799) |
| `vllm/tool_parsers/inkling_tool_parser.py` | [#48799](https://github.com/vllm-project/vllm/pull/48799) |
| `vllm/transformers_utils/processors/inkling.py` | [#48799](https://github.com/vllm-project/vllm/pull/48799) |

## PR Coverage Summary

- Git-traced PRs: 20
- Extra PRs preserved from existing docs: 0
- Total PRs in this document: 20
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-07-16 | [#48799](https://github.com/vllm-project/vllm/pull/48799) | merged | [Model] Add Inkling model support [1/N] | `vllm/models/inkling/nvidia/ops/qkvr_prep.py`, `vllm/models/inkling/nvidia/ops/lamport.py`, `vllm/models/inkling/nvidia/model.py` |
| 2026-07-16 | [#48822](https://github.com/vllm-project/vllm/pull/48822) | merged | [Model] Add PW CUDA graph support for Inkling [2/N] | `tests/models/inkling/test_contract_validation.py`, `vllm/models/inkling/nvidia/ops/sconv.py`, `vllm/models/inkling/nvidia/sconv_swa_attn.py` |
| 2026-07-16 | [#48858](https://github.com/vllm-project/vllm/pull/48858) | merged | [Model] Add Hopper FA4 relative attention for Inkling | `vllm/models/inkling/nvidia/ops/fa4_rel_attention.py`, `tests/models/inkling/test_fa4_rel_attention.py` |
| 2026-07-16 | [#48869](https://github.com/vllm-project/vllm/pull/48869) | merged | [Model] Add Inkling MTP=1 support [3/N] | `vllm/models/inkling/nvidia/mtp.py`, `tests/models/inkling/test_mtp_input_fusion.py`, `vllm/models/inkling/nvidia/ops/norm.py` |
| 2026-07-17 | [#48884](https://github.com/vllm-project/vllm/pull/48884) | merged | [Model] Add Inkling LoRA support [4/N] | `vllm/models/inkling/nvidia/moe.py`, `vllm/lora/layers/fused_moe.py`, `vllm/models/inkling/nvidia/logits_processor.py` |
| 2026-07-18 | [#48990](https://github.com/vllm-project/vllm/pull/48990) | merged | [Model] Use standard ModelOpt config for Inkling NVFP4 | `vllm/models/inkling/nvidia/moe.py`, `tests/models/inkling/test_moe_weight_layout.py`, `vllm/models/inkling/nvidia/model.py` |
| 2026-07-23 | [#49485](https://github.com/vllm-project/vllm/pull/49485) | merged | [Bugfix][Model] Remove SciPy dependency from Inkling scale planning | `tests/models/inkling/test_contract_validation.py`, `vllm/models/inkling/common/towers.py` |
| 2026-07-23 | [#49487](https://github.com/vllm-project/vllm/pull/49487) | merged | [Performance][Model] Avoid transient Inkling result allocations (performance, and OOM prevention on smaller memory configurations) | `vllm/models/inkling/nvidia/mlp.py`, `vllm/models/inkling/nvidia/moe.py` |
| 2026-07-24 | [#49258](https://github.com/vllm-project/vllm/pull/49258) | merged | [Model] Support llm-compressor Inkling NVFP4 weights | `tests/models/inkling/test_moe_weight_layout.py`, `vllm/models/inkling/nvidia/moe.py`, `vllm/models/inkling/nvidia/model.py` |
| 2026-07-27 | [#48841](https://github.com/vllm-project/vllm/pull/48841) | merged | [ROCm] [Model] Enable TML inkling | `vllm/models/inkling/amd/ops/gluon/rel_mha_decode_gfx950.py`, `vllm/models/inkling/amd/ops/qkvr_prep.py`, `vllm/models/inkling/amd/ops/gluon/rel_mha_extend_gfx950.py` |
| 2026-07-29 | [#48876](https://github.com/vllm-project/vllm/pull/48876) | merged | [Model] Add Inkling compressed-tensors dynamic FP8 support | `tests/models/inkling/test_moe_weight_layout.py`, `vllm/models/inkling/nvidia/moe.py` |
| 2026-07-30 | [#50403](https://github.com/vllm-project/vllm/pull/50403) | merged | [Frontend] Preserve bare Inkling text in Python and Rust parsers | `tests/parser/engine/test_inkling.py`, `vllm/parser/inkling.py` |
| 2026-08-04 | [#50697](https://github.com/vllm-project/vllm/pull/50697) | merged | [Kernel][Inkling] Fuse shared-expert partial addition into the Lamport collective | `vllm/models/inkling/nvidia/ops/lamport.py`, `vllm/models/inkling/nvidia/model.py`, `vllm/models/inkling/nvidia/moe.py` |
| 2026-08-05 | [#50806](https://github.com/vllm-project/vllm/pull/50806) | merged | [ROCm] Restore Inkling MTP backend parity | `vllm/models/inkling/amd/mtp.py` |
| 2026-08-08 | [#51391](https://github.com/vllm-project/vllm/pull/51391) | merged | [Bugfix][Parser] Prevent Inkling block-end leakage with tools | `tests/parser/engine/test_inkling.py`, `vllm/parser/engine/streaming_parser_engine.py` |
| 2026-08-08 | [#49876](https://github.com/vllm-project/vllm/pull/49876) | merged | [Bugfix][Parser] Confirm reasoning end when an Inkling content block opens | `tests/parser/engine/test_inkling.py`, `vllm/parser/inkling.py` |
| 2026-08-10 | [#50528](https://github.com/vllm-project/vllm/pull/50528) | merged | [Bugfix][Parser] Emit REASONING_END for Inkling tool calls that follow no thinking block | `tests/parser/engine/test_inkling.py`, `vllm/parser/inkling.py` |
| 2026-08-11 | [#49315](https://github.com/vllm-project/vllm/pull/49315) | merged | [2/N][Feat][Perf] Add new warmup infrastructure for JITs. Add predicate filtering for JIT warmup, and migrate Inkling FA4 | `vllm/models/inkling/nvidia/ops/fa4_rel_attention.py`, `tests/models/inkling/test_fa4_warmup.py`, `vllm/models/inkling/nvidia/attention.py` |
| 2026-08-11 | [#51850](https://github.com/vllm-project/vllm/pull/51850) | merged | [Bugfix] Support HF-config compat for Inkling | `vllm/models/inkling/configs.py` |
| 2026-09-26 | [#58792](https://github.com/vllm-project/vllm/pull/58792) | merged | [Bugfix][Frontend] Fix Inkling tool name leaking into content after reasoning | `tests/parser/engine/test_inkling.py`, `vllm/parser/inkling.py` |

## Per-PR Diff Audit Cards

### PR #48799 - [Model] Add Inkling model support [1/N]

- Link: https://github.com/vllm-project/vllm/pull/48799
- Status/date: merged / 2026-07-16
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/inkling/test_contract_validation.py`, `tests/models/inkling/test_fa4_rel_attention.py`, `tests/models/inkling/test_fa4_warmup.py`, `tests/models/inkling/test_mm_towers.py`, `tests/models/inkling/test_moe_weight_layout.py` and 37 files; associated commits `6570c9800cda`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 95 files, +12137/-62, 12887 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/inkling/nvidia/ops/qkvr_prep.py` added +918/-0 (918 lines); hunks: -0,0 +1,918; symbols: _rel_proj_low_latency_kernel, _rel_proj_throughput_kernel, use_rel_proj_throughput, qkvr_rel_proj, touching `_rel_proj_low_latency_kernel, _rel_proj_throughput_kernel, use_rel_proj_throughput`; `vllm/models/inkling/nvidia/ops/lamport.py` added +766/-0 (766 lines); hunks: -0,0 +1,766; symbols: _pack_bf16_pairs, _unpack_bf16_pairs, _wait_pairs, _publish_input_kernel, touching `_pack_bf16_pairs, _unpack_bf16_pairs, _wait_pairs`; `vllm/models/inkling/nvidia/model.py` added +678/-0 (678 lines); hunks: -0,0 +1,678; symbols: _layer_id, _sconv_add_norm, InklingDecoderLayer, __init__, touching `_layer_id, _sconv_add_norm, InklingDecoderLayer`; `vllm/models/inkling/nvidia/moe.py` added +578/-0 (578 lines); hunks: -0,0 +1,578; symbols: _linear_with_fp32_out, _inkling_gate_select_kernel, inkling_gate_select, InklingGate, touching `_linear_with_fp32_out, _inkling_gate_select_kernel, inkling_gate_select`.
- Code diff details:
  - `vllm/models/inkling/nvidia/ops/qkvr_prep.py` added +918/-0 (918 lines); hunks: -0,0 +1,918; symbols: _rel_proj_low_latency_kernel, _rel_proj_throughput_kernel, use_rel_proj_throughput, qkvr_rel_proj
  - `vllm/models/inkling/nvidia/ops/lamport.py` added +766/-0 (766 lines); hunks: -0,0 +1,766; symbols: _pack_bf16_pairs, _unpack_bf16_pairs, _wait_pairs, _publish_input_kernel
  - `vllm/models/inkling/nvidia/model.py` added +678/-0 (678 lines); hunks: -0,0 +1,678; symbols: _layer_id, _sconv_add_norm, InklingDecoderLayer, __init__
  - `vllm/models/inkling/nvidia/moe.py` added +578/-0 (578 lines); hunks: -0,0 +1,578; symbols: _linear_with_fp32_out, _inkling_gate_select_kernel, inkling_gate_select, InklingGate
  - `vllm/transformers_utils/processors/inkling.py` added +504/-0 (504 lines); hunks: -0,0 +1,504; symbols: _validate_image_rescale, _scaled_image_dimensions, scale, _load_image_bytes
- Key code excerpts:

```diff
diff -- vllm/models/inkling/nvidia/ops/qkvr_prep.py
@@ -0,0 +1,918 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import torch
+from vllm.triton_utils import tl, triton
+from vllm.utils.torch_utils import aux_stream
+LOW_BLOCK_M = 32
diff -- vllm/models/inkling/nvidia/ops/lamport.py
@@ -0,0 +1,766 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Deadlock-free fused RS + short-conv + AG + residual + RMSNorm.
+The public integration surface is ``LamportRSConv.rs_sconv_ag_add_norm``.
+Liveness
+--------
diff -- vllm/models/inkling/nvidia/model.py
@@ -0,0 +1,678 @@
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/inkling/nvidia/ops/qkvr_prep.py` added +918/-0; `vllm/models/inkling/nvidia/ops/lamport.py` added +766/-0; `vllm/models/inkling/nvidia/model.py` added +678/-0; `vllm/models/inkling/nvidia/moe.py` added +578/-0; `vllm/transformers_utils/processors/inkling.py` added +504/-0; `vllm/models/inkling/configs.py` added +373/-0
  - tests: `tests/models/inkling/test_fa4_rel_attention.py` added +381/-0
- Risk and verification: The diff ships test coverage in `rust/src/chat/tests/roundtrip.rs`, `tests/kernels/test_ll_bf16_gemm.py`, `tests/models/inkling/test_contract_validation.py`, `tests/models/inkling/test_fa4_rel_attention.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #48822 - [Model] Add PW CUDA graph support for Inkling [2/N]

- Link: https://github.com/vllm-project/vllm/pull/48822
- Status/date: merged / 2026-07-16
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/inkling/test_contract_validation.py`, `vllm/models/inkling/nvidia/attention.py`, `vllm/models/inkling/nvidia/ops/sconv.py`, `vllm/models/inkling/nvidia/sconv_swa_attn.py`, `vllm/models/inkling/nvidia/short_conv.py`; associated commits `251f7e478e8e`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 10 files, +98/-32, 280 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/inkling/test_contract_validation.py` modified +5/-5 (10 lines); hunks: -34,17 +34,17 @@ def test_inkling_raw_2d_audio_is_rejected_as_ambiguous():; symbols: test_inkling_raw_2d_audio_is_rejected_as_ambiguous, test_inkling_supports_full_decode_only_cudagraphs, test_inkling_supports_piecewise_cudagraphs, touching `test_inkling_raw_2d_audio_is_rejected_as_ambiguous, test_inkling_supports_full_decode_only_cudagraphs, test_inkling_supports_piecewise_cudagraphs`; `vllm/models/inkling/nvidia/ops/sconv.py` modified +4/-4 (8 lines); hunks: -21,8 +21,8; -167,8 +167,8 @@ def fused_sconv(; symbols: fused_sconv, touching `fused_sconv`; `vllm/models/inkling/nvidia/sconv_swa_attn.py` modified +1/-3 (4 lines); hunks: -49,9 +49,7 @@ class InklingSconvMetadata(AttentionMetadata):; symbols: InklingSconvMetadata, InklingSconvMetadataBuilder, __init__, touching `InklingSconvMetadata, InklingSconvMetadataBuilder, __init__`; `vllm/models/inkling/nvidia/short_conv.py` modified +2/-2 (4 lines); hunks: -17,8 +17,8.
- Code diff details:
  - `tests/models/inkling/test_contract_validation.py` modified +5/-5 (10 lines); hunks: -34,17 +34,17 @@ def test_inkling_raw_2d_audio_is_rejected_as_ambiguous():; symbols: test_inkling_raw_2d_audio_is_rejected_as_ambiguous, test_inkling_supports_full_decode_only_cudagraphs, test_inkling_supports_piecewise_cudagraphs
  - `vllm/models/inkling/nvidia/ops/sconv.py` modified +4/-4 (8 lines); hunks: -21,8 +21,8; -167,8 +167,8 @@ def fused_sconv(; symbols: fused_sconv
  - `vllm/models/inkling/nvidia/sconv_swa_attn.py` modified +1/-3 (4 lines); hunks: -49,9 +49,7 @@ class InklingSconvMetadata(AttentionMetadata):; symbols: InklingSconvMetadata, InklingSconvMetadataBuilder, __init__
  - `vllm/models/inkling/nvidia/short_conv.py` modified +2/-2 (4 lines); hunks: -17,8 +17,8
  - `vllm/models/inkling/nvidia/attention.py` modified +2/-0 (2 lines); hunks: -7,6 +7,7; -286,6 +287,7 @@ def forward(; symbols: forward, _attention
- Key code excerpts:

```diff
diff -- tests/models/inkling/test_contract_validation.py
@@ -34,17 +34,17 @@ def test_inkling_raw_2d_audio_is_rejected_as_ambiguous():
-def test_inkling_supports_full_decode_only_cudagraphs():
+def test_inkling_supports_piecewise_cudagraphs():
-    assert support(None, None) == AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
+    assert support(None, None) == AttentionCGSupport.UNIFORM_BATCH
-        cudagraph_mode=CUDAGraphMode.FULL,
+        cudagraph_mode=CUDAGraphMode.PIECEWISE,
diff -- vllm/models/inkling/nvidia/ops/sconv.py
@@ -21,8 +21,8 @@
-decode path replays correctly under a full CUDA graph without any
-data-dependent shape or branch.
+path replays correctly under eager, breakable PIECEWISE, and FULL cudagraphs
+without any data-dependent shape or branch.
@@ -167,8 +167,8 @@ def fused_sconv(
-    it is race-free in one launch for prefill / decode / spec and supports full
diff -- vllm/models/inkling/nvidia/sconv_swa_attn.py
@@ -49,9 +49,7 @@ class InklingSconvMetadata(AttentionMetadata):
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/inkling/test_contract_validation.py` modified +5/-5
  - runtime: `vllm/models/inkling/nvidia/ops/sconv.py` modified +4/-4; `vllm/models/inkling/nvidia/sconv_swa_attn.py` modified +1/-3; `vllm/models/inkling/nvidia/short_conv.py` modified +2/-2; `vllm/models/inkling/nvidia/attention.py` modified +2/-0
- Risk and verification: The diff ships test coverage in `tests/models/inkling/test_contract_validation.py`, `tests/v1/cudagraph/test_breakable_cudagraph.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #48858 - [Model] Add Hopper FA4 relative attention for Inkling

- Link: https://github.com/vllm-project/vllm/pull/48858
- Status/date: merged / 2026-07-16
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/inkling/test_fa4_rel_attention.py`, `vllm/models/inkling/nvidia/ops/fa4_rel_attention.py`; associated commits `f61163e6c736`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +109/-5, 195 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/inkling/nvidia/ops/fa4_rel_attention.py` modified +62/-5 (67 lines); hunks: -2,6 +2,10; -12,6 +16,47 @@ def bucket_max_seqlen_q(max_seqlen_q: int) -> int:; symbols: bucket_max_seqlen_q, _use_sheared_bias, _get_score_mod, score_mod_rel_bias, touching `bucket_max_seqlen_q, _use_sheared_bias, _get_score_mod`; `tests/models/inkling/test_fa4_rel_attention.py` modified +47/-0 (47 lines); hunks: -13,6 +13,8; -21,6 +23,7; symbols: test_num_splits_hopper_is_unsplit, test_sheared_bias_architecture_selection, blackwell_platform, _run_case, touching `test_num_splits_hopper_is_unsplit, test_sheared_bias_architecture_selection, blackwell_platform`.
- Code diff details:
  - `vllm/models/inkling/nvidia/ops/fa4_rel_attention.py` modified +62/-5 (67 lines); hunks: -2,6 +2,10; -12,6 +16,47 @@ def bucket_max_seqlen_q(max_seqlen_q: int) -> int:; symbols: bucket_max_seqlen_q, _use_sheared_bias, _get_score_mod, score_mod_rel_bias
  - `tests/models/inkling/test_fa4_rel_attention.py` modified +47/-0 (47 lines); hunks: -13,6 +13,8; -21,6 +23,7; symbols: test_num_splits_hopper_is_unsplit, test_sheared_bias_architecture_selection, blackwell_platform, _run_case
- Key code excerpts:

```diff
diff -- vllm/models/inkling/nvidia/ops/fa4_rel_attention.py
@@ -2,6 +2,10 @@
+from collections.abc import Callable
+from functools import cache
+from typing import Any
@@ -12,6 +16,47 @@ def bucket_max_seqlen_q(max_seqlen_q: int) -> int:
+@cache
+def _use_sheared_bias() -> bool:
diff -- tests/models/inkling/test_fa4_rel_attention.py
@@ -13,6 +13,8 @@
+import importlib
@@ -21,6 +23,7 @@
+    _use_sheared_bias,
@@ -78,6 +81,23 @@ def test_num_splits_hopper_is_unsplit(monkeypatch):
+@pytest.mark.parametrize(
+    ("major", "expected"),
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/inkling/nvidia/ops/fa4_rel_attention.py` modified +62/-5
  - tests: `tests/models/inkling/test_fa4_rel_attention.py` modified +47/-0
- Risk and verification: The diff ships test coverage in `tests/models/inkling/test_fa4_rel_attention.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #48869 - [Model] Add Inkling MTP=1 support [3/N]

- Link: https://github.com/vllm-project/vllm/pull/48869
- Status/date: merged / 2026-07-16
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/inkling/test_contract_validation.py`, `tests/models/inkling/test_mtp_input_fusion.py`, `vllm/models/inkling/__init__.py`, `vllm/models/inkling/configs.py`, `vllm/models/inkling/nvidia/model.py` and 7 files; associated commits `fb5ec0dc9edf`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 11 files, +710/-6, 854 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/inkling/nvidia/mtp.py` added +408/-0 (408 lines); hunks: -0,0 +1,408; symbols: _mtp_depth_from_name, InklingMTPDepthLayer, __init__, forward, touching `_mtp_depth_from_name, InklingMTPDepthLayer, __init__`; `tests/models/inkling/test_mtp_input_fusion.py` added +106/-0 (106 lines); hunks: -0,0 +1,106; symbols: _ref, test_embed_dual_rmsnorm_cat, test_embed_rmsnorm, touching `_ref, test_embed_dual_rmsnorm_cat, test_embed_rmsnorm`; `vllm/models/inkling/nvidia/ops/norm.py` modified +99/-0 (99 lines); hunks: -268,6 +268,105 @@ def embed_rmsnorm(; symbols: embed_rmsnorm, _embed_dual_rmsnorm_cat_kernel, embed_dual_rmsnorm_cat, rmsnorm, touching `embed_rmsnorm, _embed_dual_rmsnorm_cat_kernel, embed_dual_rmsnorm_cat`; `vllm/models/inkling/nvidia/model.py` modified +14/-5 (19 lines); hunks: -123,6 +123,7 @@ def __init__(; -161,7 +162,7 @@ def __init__(; symbols: __init__, forward, InklingReplicatedEmbedding, touching `__init__, forward, InklingReplicatedEmbedding`.
- Code diff details:
  - `vllm/models/inkling/nvidia/mtp.py` added +408/-0 (408 lines); hunks: -0,0 +1,408; symbols: _mtp_depth_from_name, InklingMTPDepthLayer, __init__, forward
  - `tests/models/inkling/test_mtp_input_fusion.py` added +106/-0 (106 lines); hunks: -0,0 +1,106; symbols: _ref, test_embed_dual_rmsnorm_cat, test_embed_rmsnorm
  - `vllm/models/inkling/nvidia/ops/norm.py` modified +99/-0 (99 lines); hunks: -268,6 +268,105 @@ def embed_rmsnorm(; symbols: embed_rmsnorm, _embed_dual_rmsnorm_cat_kernel, embed_dual_rmsnorm_cat, rmsnorm
  - `vllm/models/inkling/nvidia/model.py` modified +14/-5 (19 lines); hunks: -123,6 +123,7 @@ def __init__(; -161,7 +162,7 @@ def __init__(; symbols: __init__, forward, InklingReplicatedEmbedding
  - `vllm/models/inkling/__init__.py` modified +6/-0 (6 lines); hunks: -7,14 +7,20; symbols: __getattr__
- Key code excerpts:

```diff
diff -- vllm/models/inkling/nvidia/mtp.py
@@ -0,0 +1,408 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Inkling MTP (Multi-Token Prediction) draft model (NVIDIA).
+Implements the first MTP depth from the reference ``mtp_model.py`` shipped with
+the checkpoint. It owns ``hidden_norm`` / ``embed_norm`` RMSNorms, a ``2H -> H``
+input projection, and a full Inkling transformer block with a dense bf16 MLP.
diff -- tests/models/inkling/test_mtp_input_fusion.py
@@ -0,0 +1,106 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Bit-exactness tests for the fused MTP depth-layer input kernel.
+``embed_dual_rmsnorm_cat`` must match the unfused module sequence exactly:
+each rmsnorm computes in fp32 and rounds to bf16 at the same points as the
+vendored ``rmsnorm`` kernel (including the bf16 round-trip between the
diff -- vllm/models/inkling/nvidia/ops/norm.py
@@ -268,6 +268,105 @@ def embed_rmsnorm(
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/inkling/nvidia/mtp.py` added +408/-0; `vllm/models/inkling/nvidia/ops/norm.py` modified +99/-0; `vllm/models/inkling/nvidia/model.py` modified +14/-5; `vllm/models/inkling/__init__.py` modified +6/-0; `vllm/models/inkling/configs.py` modified +6/-0
  - tests: `tests/models/inkling/test_mtp_input_fusion.py` added +106/-0; `tests/models/inkling/test_contract_validation.py` modified +5/-0
- Risk and verification: The diff ships test coverage in `tests/config/test_speculative_draft_hf_overrides.py`, `tests/models/inkling/test_contract_validation.py`, `tests/models/inkling/test_mtp_input_fusion.py`, `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #48884 - [Model] Add Inkling LoRA support [4/N]

- Link: https://github.com/vllm-project/vllm/pull/48884
- Status/date: merged / 2026-07-17
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/inkling/test_moe_weight_layout.py`, `vllm/models/inkling/nvidia/logits_processor.py`, `vllm/models/inkling/nvidia/model.py`, `vllm/models/inkling/nvidia/moe.py`; associated commits `f3e9497e921a`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 12 files, +338/-48, 659 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/inkling/nvidia/moe.py` modified +71/-1 (72 lines); hunks: -343,6 +343,71 @@ def forward(self, x: torch.Tensor, gammas: torch.Tensor) ->...; -417,7 +482,12 @@ def __init__(; symbols: forward, InklingSinkExpertsLinear, __init__, _gamma_expand, touching `forward, InklingSinkExpertsLinear, __init__`; `vllm/lora/layers/fused_moe.py` modified +51/-15 (66 lines); hunks: -57,6 +57,9 @@ def __init__(self, base_layer: MoERunner) -> None:; -150,10 +153,21 @@ def _build_lora_context(self):; symbols: __init__, _build_lora_context, _w13_a_num_experts, _w2_b_num_experts, touching `__init__, _build_lora_context, _w13_a_num_experts`; `vllm/models/inkling/nvidia/logits_processor.py` modified +57/-1 (58 lines); hunks: -1,6 +1,22; -47,6 +63,46 @@ def forward(; symbols: forward, _lora_forward, _base_forward, touching `forward, _lora_forward, _base_forward`; `tests/models/inkling/test_moe_weight_layout.py` modified +25/-0 (25 lines); hunks: -6,7 +6,9; -62,6 +64,29 @@ def fake_ll_bf16_gemm(x, weight):; symbols: fake_ll_bf16_gemm, test_gate_is_not_a_lora_target, test_custom_embedding_is_not_a_lora_target, test_moe_loads_calibrated_input_scale, touching `fake_ll_bf16_gemm, test_gate_is_not_a_lora_target, test_custom_embedding_is_not_a_lora_target`.
- Code diff details:
  - `vllm/models/inkling/nvidia/moe.py` modified +71/-1 (72 lines); hunks: -343,6 +343,71 @@ def forward(self, x: torch.Tensor, gammas: torch.Tensor) ->...; -417,7 +482,12 @@ def __init__(; symbols: forward, InklingSinkExpertsLinear, __init__, _gamma_expand
  - `vllm/lora/layers/fused_moe.py` modified +51/-15 (66 lines); hunks: -57,6 +57,9 @@ def __init__(self, base_layer: MoERunner) -> None:; -150,10 +153,21 @@ def _build_lora_context(self):; symbols: __init__, _build_lora_context, _w13_a_num_experts, _w2_b_num_experts
  - `vllm/models/inkling/nvidia/logits_processor.py` modified +57/-1 (58 lines); hunks: -1,6 +1,22; -47,6 +63,46 @@ def forward(; symbols: forward, _lora_forward, _base_forward
  - `tests/models/inkling/test_moe_weight_layout.py` modified +25/-0 (25 lines); hunks: -6,7 +6,9; -62,6 +64,29 @@ def fake_ll_bf16_gemm(x, weight):; symbols: fake_ll_bf16_gemm, test_gate_is_not_a_lora_target, test_custom_embedding_is_not_a_lora_target, test_moe_loads_calibrated_input_scale
  - `vllm/models/inkling/nvidia/model.py` modified +11/-1 (12 lines); hunks: -24,6 +24,7; -360,7 +361,7 @@ def forward(; symbols: forward, _TmlForCausalLMBase
- Key code excerpts:

```diff
diff -- vllm/models/inkling/nvidia/moe.py
@@ -343,6 +343,71 @@ def forward(self, x: torch.Tensor, gammas: torch.Tensor) -> torch.Tensor:
+class InklingSinkExpertsLinear(nn.Module):
+    """LoRA-capable implementation of the Inkling sink experts."""
+    def __init__(
+        self,
+        n_experts: int,
+        d_model: int,
diff -- vllm/lora/layers/fused_moe.py
@@ -57,6 +57,9 @@ def __init__(self, base_layer: MoERunner) -> None:
+        # Set from lora_config.enable_moe_shared_loras in create_lora_weights.
+        # When True, w13 lora_A and w2 lora_B are stored once for all experts.
+        self.enable_moe_shared_loras = False
@@ -150,10 +153,21 @@ def _build_lora_context(self):
+            enable_moe_shared_loras=self.enable_moe_shared_loras,
+    @property
diff -- vllm/models/inkling/nvidia/logits_processor.py
@@ -1,6 +1,22 @@
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/inkling/nvidia/moe.py` modified +71/-1; `vllm/lora/layers/fused_moe.py` modified +51/-15; `vllm/models/inkling/nvidia/logits_processor.py` modified +57/-1; `vllm/models/inkling/nvidia/model.py` modified +11/-1
  - tests: `tests/models/inkling/test_moe_weight_layout.py` modified +25/-0
- Risk and verification: The diff ships test coverage in `tests/models/inkling/test_moe_weight_layout.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #48990 - [Model] Use standard ModelOpt config for Inkling NVFP4

- Link: https://github.com/vllm-project/vllm/pull/48990
- Status/date: merged / 2026-07-18
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/inkling/test_moe_weight_layout.py`, `vllm/models/inkling/nvidia/model.py`, `vllm/models/inkling/nvidia/moe.py`, `vllm/models/inkling/nvidia/mtp.py`; associated commits `02c01f442b8b`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 7 files, +56/-110, 309 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/inkling/nvidia/moe.py` modified +5/-28 (33 lines); hunks: -43,20 +43,19; -412,10 +411,9 @@ class InklingMoE(nn.Module):; symbols: _linear_with_fp32_out, InklingMoE, __init__, touching `_linear_with_fp32_out, InklingMoE, __init__`; `tests/models/inkling/test_moe_weight_layout.py` modified +25/-0 (25 lines); hunks: -7,6 +7,7; -87,6 +88,30 @@ def test_custom_embedding_is_not_a_lora_target() -> None:; symbols: test_custom_embedding_is_not_a_lora_target, test_inkling_mapper_maps_modelopt_exclusions, test_moe_loads_calibrated_input_scale, touching `test_custom_embedding_is_not_a_lora_target, test_inkling_mapper_maps_modelopt_exclusions, test_moe_loads_calibrated_input_scale`; `vllm/models/inkling/nvidia/model.py` modified +2/-15 (17 lines); hunks: -47,7 +47,6; -123,7 +122,6 @@ def __init__(; symbols: __init__, get_layer, touching `__init__, get_layer`; `vllm/models/inkling/nvidia/mtp.py` modified +0/-1 (1 lines); hunks: -73,7 +73,6 @@ def __init__(self, config: InklingModelConfig, prefix: str, is...; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/models/inkling/nvidia/moe.py` modified +5/-28 (33 lines); hunks: -43,20 +43,19; -412,10 +411,9 @@ class InklingMoE(nn.Module):; symbols: _linear_with_fp32_out, InklingMoE, __init__
  - `tests/models/inkling/test_moe_weight_layout.py` modified +25/-0 (25 lines); hunks: -7,6 +7,7; -87,6 +88,30 @@ def test_custom_embedding_is_not_a_lora_target() -> None:; symbols: test_custom_embedding_is_not_a_lora_target, test_inkling_mapper_maps_modelopt_exclusions, test_moe_loads_calibrated_input_scale
  - `vllm/models/inkling/nvidia/model.py` modified +2/-15 (17 lines); hunks: -47,7 +47,6; -123,7 +122,6 @@ def __init__(; symbols: __init__, get_layer
  - `vllm/models/inkling/nvidia/mtp.py` modified +0/-1 (1 lines); hunks: -73,7 +73,6 @@ def __init__(self, config: InklingModelConfig, prefix: str, is...; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/models/inkling/nvidia/moe.py
@@ -43,20 +43,19 @@
-from ..nvfp4 import FLOAT4_E2M1_MAX, FLOAT8_E4M3_MAX
-    from ..nvfp4 import InklingNvfp4Config
+    from vllm.model_executor.layers.quantization import QuantizationConfig
+_NVFP4_INPUT_SCALE_DENOMINATOR = torch.finfo(torch.float8_e4m3fn).max * 6.0
@@ -412,10 +411,9 @@ class InklingMoE(nn.Module):
-        layer_id: int,
diff -- tests/models/inkling/test_moe_weight_layout.py
@@ -7,6 +7,7 @@
+from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
@@ -87,6 +88,30 @@ def test_custom_embedding_is_not_a_lora_target() -> None:
+def test_inkling_mapper_maps_modelopt_exclusions() -> None:
+    quant_config = ModelOptNvFp4Config.from_config(
+        {
+            "quantization": {
diff -- vllm/models/inkling/nvidia/model.py
@@ -47,7 +47,6 @@
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/inkling/nvidia/moe.py` modified +5/-28; `vllm/models/inkling/nvidia/model.py` modified +2/-15; `vllm/models/inkling/nvidia/mtp.py` modified +0/-1
  - tests: `tests/models/inkling/test_moe_weight_layout.py` modified +25/-0
- Risk and verification: The diff ships test coverage in `tests/config/test_model_arch_config.py`, `tests/models/inkling/test_moe_weight_layout.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #49485 - [Bugfix][Model] Remove SciPy dependency from Inkling scale planning

- Link: https://github.com/vllm-project/vllm/pull/49485
- Status/date: merged / 2026-07-23
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/inkling/test_contract_validation.py`, `vllm/models/inkling/common/towers.py`; associated commits `4080263bb2c5`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +32/-2, 69 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/inkling/test_contract_validation.py` modified +17/-0 (17 lines); hunks: -6,6 +6,7; -17,6 +18,22; symbols: test_vision_scale_plan_matches_released_config, test_vision_scale_plan_breaks_assignment_ties_in_order, touching `test_vision_scale_plan_matches_released_config, test_vision_scale_plan_breaks_assignment_ties_in_order`; `vllm/models/inkling/common/towers.py` modified +15/-2 (17 lines); hunks: -9,6 +9,7; -45,6 +46,20 @@ def _prime_factors(n: int) -> list[int]:; symbols: _prime_factors, linear_sum_assignment, plan_out_scales, _round_up, touching `_prime_factors, linear_sum_assignment, plan_out_scales`.
- Code diff details:
  - `tests/models/inkling/test_contract_validation.py` modified +17/-0 (17 lines); hunks: -6,6 +6,7; -17,6 +18,22; symbols: test_vision_scale_plan_matches_released_config, test_vision_scale_plan_breaks_assignment_ties_in_order
  - `vllm/models/inkling/common/towers.py` modified +15/-2 (17 lines); hunks: -9,6 +9,7; -45,6 +46,20 @@ def _prime_factors(n: int) -> list[int]:; symbols: _prime_factors, linear_sum_assignment, plan_out_scales, _round_up
- Key code excerpts:

```diff
diff -- tests/models/inkling/test_contract_validation.py
@@ -6,6 +6,7 @@
+from vllm.models.inkling.common.towers import plan_out_scales
@@ -17,6 +18,22 @@
+def test_vision_scale_plan_matches_released_config():
+    assert plan_out_scales(2, 40, 4) == [
+        (1, 1, 1, 3),
+        (1, 5, 5, 128),
diff -- vllm/models/inkling/common/towers.py
@@ -9,6 +9,7 @@
+from itertools import combinations
@@ -45,6 +46,20 @@ def _prime_factors(n: int) -> list[int]:
+def linear_sum_assignment(
+    cost_matrix: np.ndarray,
+) -> tuple[np.ndarray, np.ndarray]:
+    """Implement SciPy's assignment for Inkling's ordered L1 cost matrix."""
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/inkling/test_contract_validation.py` modified +17/-0
  - runtime: `vllm/models/inkling/common/towers.py` modified +15/-2
- Risk and verification: The diff ships test coverage in `tests/models/inkling/test_contract_validation.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #49487 - [Performance][Model] Avoid transient Inkling result allocations (performance, and OOM prevention on smaller memory configurations)

- Link: https://github.com/vllm-project/vllm/pull/49487
- Status/date: merged / 2026-07-23
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/inkling/nvidia/mlp.py`, `vllm/models/inkling/nvidia/moe.py`; associated commits `9a698f3255a7`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +2/-2, 17 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/inkling/nvidia/mlp.py` modified +1/-1 (2 lines); hunks: -58,6 +58,6 @@ def forward(self, x: torch.Tensor) -> torch.Tensor:; symbols: forward, touching `forward`; `vllm/models/inkling/nvidia/moe.py` modified +1/-1 (2 lines); hunks: -528,7 +528,7 @@ def forward(self, x: torch.Tensor) -> torch.Tensor | None:; symbols: forward, touching `forward`.
- Code diff details:
  - `vllm/models/inkling/nvidia/mlp.py` modified +1/-1 (2 lines); hunks: -58,6 +58,6 @@ def forward(self, x: torch.Tensor) -> torch.Tensor:; symbols: forward
  - `vllm/models/inkling/nvidia/moe.py` modified +1/-1 (2 lines); hunks: -528,7 +528,7 @@ def forward(self, x: torch.Tensor) -> torch.Tensor | None:; symbols: forward
- Key code excerpts:

```diff
diff -- vllm/models/inkling/nvidia/mlp.py
@@ -58,6 +58,6 @@ def forward(self, x: torch.Tensor) -> torch.Tensor:
-            x = x * self.global_scale
+            x.mul_(self.global_scale)
diff -- vllm/models/inkling/nvidia/moe.py
@@ -528,7 +528,7 @@ def forward(self, x: torch.Tensor) -> torch.Tensor | None:
-        return out + sink_out
+        return out.add_(sink_out)
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/inkling/nvidia/mlp.py` modified +1/-1; `vllm/models/inkling/nvidia/moe.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/models/inkling/nvidia/mlp.py`, `vllm/models/inkling/nvidia/moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #49258 - [Model] Support llm-compressor Inkling NVFP4 weights

- Link: https://github.com/vllm-project/vllm/pull/49258
- Status/date: merged / 2026-07-24
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/inkling/test_moe_weight_layout.py`, `vllm/models/inkling/nvidia/model.py`, `vllm/models/inkling/nvidia/moe.py`; associated commits `d65acd83d87b`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +83/-10, 133 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/inkling/test_moe_weight_layout.py` modified +55/-0 (55 lines); hunks: -112,6 +112,30 @@ def test_inkling_mapper_maps_modelopt_exclusions() -> None:; -132,6 +156,37 @@ def test_moe_loads_calibrated_input_scale(projection: str,...; symbols: test_inkling_mapper_maps_modelopt_exclusions, test_inkling_mapper_maps_compressed_tensors_expert_params, test_moe_loads_calibrated_input_scale, test_moe_loads_compressed_tensors_global_scale, touching `test_inkling_mapper_maps_modelopt_exclusions, test_inkling_mapper_maps_compressed_tensors_expert_params, test_moe_loads_calibrated_input_scale`; `vllm/models/inkling/nvidia/moe.py` modified +18/-9 (27 lines); hunks: -10,11 +10,10; -575,11 +574,21 @@ def load_expert_weight(self, name: str, weight: torch.Tens...; symbols: load_expert_weight, touching `load_expert_weight`; `vllm/models/inkling/nvidia/model.py` modified +10/-1 (11 lines); hunks: -378,11 +378,20 @@ class _TmlForCausalLMBase(nn.Module, SupportsPP, SupportsL...; symbols: _TmlForCausalLMBase, touching `_TmlForCausalLMBase`.
- Code diff details:
  - `tests/models/inkling/test_moe_weight_layout.py` modified +55/-0 (55 lines); hunks: -112,6 +112,30 @@ def test_inkling_mapper_maps_modelopt_exclusions() -> None:; -132,6 +156,37 @@ def test_moe_loads_calibrated_input_scale(projection: str,...; symbols: test_inkling_mapper_maps_modelopt_exclusions, test_inkling_mapper_maps_compressed_tensors_expert_params, test_moe_loads_calibrated_input_scale, test_moe_loads_compressed_tensors_global_scale
  - `vllm/models/inkling/nvidia/moe.py` modified +18/-9 (27 lines); hunks: -10,11 +10,10; -575,11 +574,21 @@ def load_expert_weight(self, name: str, weight: torch.Tens...; symbols: load_expert_weight
  - `vllm/models/inkling/nvidia/model.py` modified +10/-1 (11 lines); hunks: -378,11 +378,20 @@ class _TmlForCausalLMBase(nn.Module, SupportsPP, SupportsL...; symbols: _TmlForCausalLMBase
- Key code excerpts:

```diff
diff -- tests/models/inkling/test_moe_weight_layout.py
@@ -112,6 +112,30 @@ def test_inkling_mapper_maps_modelopt_exclusions() -> None:
+@pytest.mark.parametrize("projection", ["w13", "w2"])
+@pytest.mark.parametrize("nested", [False, True])
+@pytest.mark.parametrize(
+    "suffix",
+    [
+        "input_global_scale",
diff -- vllm/models/inkling/nvidia/moe.py
@@ -10,11 +10,10 @@
-NVFP4 routed experts reuse vLLM's ModelOpt NVFP4 fused-MoE method; excluded
-(bf16) layers fall back to the unquantized method. The checkpoint's fused
-stacked tensors (interleaved gate/up rows, ``.scale`` / ``.scale2`` /
-``.input_amax`` aux tensors) are translated to the standard per-expert loads
-in :meth:`InklingMoE.load_expert_weight`.
+NVFP4 routed experts reuse vLLM's fused-MoE methods; excluded (bf16) layers
diff -- vllm/models/inkling/nvidia/model.py
@@ -378,11 +378,20 @@ class _TmlForCausalLMBase(nn.Module, SupportsPP, SupportsLoRA):
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/inkling/test_moe_weight_layout.py` modified +55/-0
  - runtime: `vllm/models/inkling/nvidia/moe.py` modified +18/-9; `vllm/models/inkling/nvidia/model.py` modified +10/-1
- Risk and verification: The diff ships test coverage in `tests/models/inkling/test_moe_weight_layout.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #48841 - [ROCm] [Model] Enable TML inkling

- Link: https://github.com/vllm-project/vllm/pull/48841
- Status/date: merged / 2026-07-27
- Trace source: `git log --name-only -- <model-files>` found it through `benchmarks/kernels/benchmark_inkling_qkvr_prep.py`, `tests/models/inkling/rocm/conftest.py`, `tests/models/inkling/rocm/test_model_alignment.py`, `tests/models/inkling/rocm/test_mxfp4_load.py`, `tests/models/inkling/rocm/test_rel_attention.py` and 30 files; associated commits `0906123953a5`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 33 files, +9213/-9, 9286 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/inkling/amd/ops/gluon/rel_mha_decode_gfx950.py` added +1037/-0 (1037 lines); hunks: -0,0 +1,1037; symbols: AttentionConfig, __init__, AttentionProgram, create, touching `AttentionConfig, __init__, AttentionProgram`; `vllm/models/inkling/amd/ops/qkvr_prep.py` added +918/-0 (918 lines); hunks: -0,0 +1,918; symbols: _rel_proj_low_latency_kernel, _rel_proj_throughput_kernel, use_rel_proj_throughput, qkvr_rel_proj, touching `_rel_proj_low_latency_kernel, _rel_proj_throughput_kernel, use_rel_proj_throughput`; `vllm/models/inkling/amd/ops/gluon/rel_mha_extend_gfx950.py` added +760/-0 (760 lines); hunks: -0,0 +1,760; symbols: _select_extend_tile, ExtendConfig, __init__, ExtendProgram, touching `_select_extend_tile, ExtendConfig, __init__`; `vllm/models/inkling/amd/moe.py` added +693/-0 (693 lines); hunks: -0,0 +1,693; symbols: _linear_with_fp32_out, _inkling_gate_select_kernel, inkling_gate_select, InklingGate, touching `_linear_with_fp32_out, _inkling_gate_select_kernel, inkling_gate_select`.
- Code diff details:
  - `vllm/models/inkling/amd/ops/gluon/rel_mha_decode_gfx950.py` added +1037/-0 (1037 lines); hunks: -0,0 +1,1037; symbols: AttentionConfig, __init__, AttentionProgram, create
  - `vllm/models/inkling/amd/ops/qkvr_prep.py` added +918/-0 (918 lines); hunks: -0,0 +1,918; symbols: _rel_proj_low_latency_kernel, _rel_proj_throughput_kernel, use_rel_proj_throughput, qkvr_rel_proj
  - `vllm/models/inkling/amd/ops/gluon/rel_mha_extend_gfx950.py` added +760/-0 (760 lines); hunks: -0,0 +1,760; symbols: _select_extend_tile, ExtendConfig, __init__, ExtendProgram
  - `vllm/models/inkling/amd/moe.py` added +693/-0 (693 lines); hunks: -0,0 +1,693; symbols: _linear_with_fp32_out, _inkling_gate_select_kernel, inkling_gate_select, InklingGate
  - `vllm/models/inkling/amd/model.py` added +669/-0 (669 lines); hunks: -0,0 +1,669; symbols: _layer_id, _sconv_add_norm, InklingDecoderLayer, __init__
- Key code excerpts:

```diff
diff -- vllm/models/inkling/amd/ops/gluon/rel_mha_decode_gfx950.py
@@ -0,0 +1,1037 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Copyright (c) 2026 LightSeek Foundation
+#
+# Permission is hereby granted, free of charge, to any person obtaining a copy
+# of this software and associated documentation files (the "Software"), to deal
diff -- vllm/models/inkling/amd/ops/qkvr_prep.py
@@ -0,0 +1,918 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import torch
+from vllm.triton_utils import tl, triton
+from vllm.utils.torch_utils import aux_stream
+LOW_BLOCK_M = 32
diff -- vllm/models/inkling/amd/ops/gluon/rel_mha_extend_gfx950.py
@@ -0,0 +1,760 @@
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/inkling/amd/ops/gluon/rel_mha_decode_gfx950.py` added +1037/-0; `vllm/models/inkling/amd/ops/qkvr_prep.py` added +918/-0; `vllm/models/inkling/amd/ops/gluon/rel_mha_extend_gfx950.py` added +760/-0; `vllm/models/inkling/amd/moe.py` added +693/-0; `vllm/models/inkling/amd/model.py` added +669/-0; `vllm/models/inkling/amd/ops/fa4_rel_attention.py` added +438/-0
  - tests: `tests/models/inkling/rocm/test_rel_attention.py` added +425/-0
- Risk and verification: The diff ships test coverage in `tests/models/inkling/rocm/conftest.py`, `tests/models/inkling/rocm/test_model_alignment.py`, `tests/models/inkling/rocm/test_mxfp4_load.py`, `tests/models/inkling/rocm/test_rel_attention.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #48876 - [Model] Add Inkling compressed-tensors dynamic FP8 support

- Link: https://github.com/vllm-project/vllm/pull/48876
- Status/date: merged / 2026-07-29
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/inkling/test_moe_weight_layout.py`, `vllm/models/inkling/nvidia/moe.py`; associated commits `17a74b745b45`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +29/-0, 43 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/inkling/test_moe_weight_layout.py` modified +26/-0 (26 lines); hunks: -187,6 +187,32 @@ def test_moe_loads_compressed_tensors_global_scale(; symbols: test_moe_loads_compressed_tensors_global_scale, test_moe_loads_channelwise_scale_for_tp, test_sink_down_projection_is_packed_during_load, touching `test_moe_loads_compressed_tensors_global_scale, test_moe_loads_channelwise_scale_for_tp, test_sink_down_projection_is_packed_during_load`; `vllm/models/inkling/nvidia/moe.py` modified +3/-0 (3 lines); hunks: -589,6 +589,9 @@ def load_expert_weight(self, name: str, weight: torch.Tensor...; symbols: load_expert_weight, touching `load_expert_weight`.
- Code diff details:
  - `tests/models/inkling/test_moe_weight_layout.py` modified +26/-0 (26 lines); hunks: -187,6 +187,32 @@ def test_moe_loads_compressed_tensors_global_scale(; symbols: test_moe_loads_compressed_tensors_global_scale, test_moe_loads_channelwise_scale_for_tp, test_sink_down_projection_is_packed_during_load
  - `vllm/models/inkling/nvidia/moe.py` modified +3/-0 (3 lines); hunks: -589,6 +589,9 @@ def load_expert_weight(self, name: str, weight: torch.Tensor...; symbols: load_expert_weight
- Key code excerpts:

```diff
diff -- tests/models/inkling/test_moe_weight_layout.py
@@ -187,6 +187,32 @@ def test_moe_loads_compressed_tensors_global_scale(
+@pytest.mark.parametrize(("projection", "checkpoint_rows"), [("w13", 8), ("w2", 4)])
+def test_moe_loads_channelwise_scale_for_tp(
+    projection: str, checkpoint_rows: int
+) -> None:
+    param = torch.nn.Parameter(torch.empty(2, 4, 1))
+    experts = SimpleNamespace(
diff -- vllm/models/inkling/nvidia/moe.py
@@ -589,6 +589,9 @@ def load_expert_weight(self, name: str, weight: torch.Tensor) -> list[str]:
+        elif key == "w2_weight_scale" and weight.shape[-1] == 1:
+            # Per-output-channel scales are replicated across TP ranks.
+            param.data[lids] = weight[gids].to(device=param.device, dtype=param.dtype)
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/inkling/test_moe_weight_layout.py` modified +26/-0
  - runtime: `vllm/models/inkling/nvidia/moe.py` modified +3/-0
- Risk and verification: The diff ships test coverage in `tests/models/inkling/test_moe_weight_layout.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #50403 - [Frontend] Preserve bare Inkling text in Python and Rust parsers

- Link: https://github.com/vllm-project/vllm/pull/50403
- Status/date: merged / 2026-07-30
- Trace source: `git log --name-only -- <model-files>` found it through `tests/parser/engine/test_inkling.py`, `vllm/parser/inkling.py`; associated commits `629a938a9284`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 4 files, +144/-18, 313 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/parser/engine/test_inkling.py` modified +29/-0 (29 lines); hunks: -178,6 +178,13 @@ def test_non_object_args_rejected(self):; -426,6 +433,28 @@ def test_generation_prompt_header_hides_tool_name(self, par...; symbols: test_non_object_args_rejected, TestNonStreaming, test_bare_text_after_model_opener, test_plain_text, touching `test_non_object_args_rejected, TestNonStreaming, test_bare_text_after_model_opener`; `vllm/parser/inkling.py` modified +1/-1 (2 lines); hunks: -268,7 +268,7 @@ def inkling_config() -> ParserEngineConfig:; symbols: inkling_config, touching `inkling_config`.
- Code diff details:
  - `tests/parser/engine/test_inkling.py` modified +29/-0 (29 lines); hunks: -178,6 +178,13 @@ def test_non_object_args_rejected(self):; -426,6 +433,28 @@ def test_generation_prompt_header_hides_tool_name(self, par...; symbols: test_non_object_args_rejected, TestNonStreaming, test_bare_text_after_model_opener, test_plain_text
  - `vllm/parser/inkling.py` modified +1/-1 (2 lines); hunks: -268,7 +268,7 @@ def inkling_config() -> ParserEngineConfig:; symbols: inkling_config
- Key code excerpts:

```diff
diff -- tests/parser/engine/test_inkling.py
@@ -178,6 +178,13 @@ def test_non_object_args_rejected(self):
+    @pytest.mark.parametrize("suffix", ["", END_MESSAGE, END_SAMPLING])
+    def test_bare_text_after_model_opener(self, parser, mock_request, suffix):
+        reasoning, content, tools = parser.parse(f"hello world{suffix}", mock_request)
+        assert reasoning is None
+        assert content == "hello world"
+        assert tools is None
diff -- vllm/parser/inkling.py
@@ -268,7 +268,7 @@ def inkling_config() -> ParserEngineConfig:
-            (),
+            (EventType.TEXT_CHUNK,),
```

- Extracted files (not manually reviewed):
  - tests: `tests/parser/engine/test_inkling.py` modified +29/-0
  - runtime: `vllm/parser/inkling.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `tests/parser/engine/test_inkling.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #50697 - [Kernel][Inkling] Fuse shared-expert partial addition into the Lamport collective

- Link: https://github.com/vllm-project/vllm/pull/50697
- Status/date: merged / 2026-08-04
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/inkling/nvidia/model.py`, `vllm/models/inkling/nvidia/moe.py`, `vllm/models/inkling/nvidia/ops/lamport.py`; associated commits `edbc4969a76b`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +48/-6, 180 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/inkling/nvidia/ops/lamport.py` modified +25/-0 (25 lines); hunks: -88,15 +88,18 @@ def _wait_pairs(ptr, offsets, mask):; -120,6 +123,13 @@ def _publish_input_kernel(; symbols: _wait_pairs, _publish_input_kernel, rs_sconv_ag_add_norm, touching `_wait_pairs, _publish_input_kernel, rs_sconv_ag_add_norm`; `vllm/models/inkling/nvidia/model.py` modified +18/-5 (23 lines); hunks: -5,7 +5,7; -57,14 +57,16; symbols: _layer_id, _sconv_add_norm, forward, touching `_layer_id, _sconv_add_norm, forward`; `vllm/models/inkling/nvidia/moe.py` modified +5/-1 (6 lines); hunks: -501,7 +501,7 @@ def _select_routed(; -527,6 +527,10 @@ def forward(self, x: torch.Tensor) -> torch.Tensor | None:; symbols: _select_routed, forward, forward_partials, touching `_select_routed, forward, forward_partials`.
- Code diff details:
  - `vllm/models/inkling/nvidia/ops/lamport.py` modified +25/-0 (25 lines); hunks: -88,15 +88,18 @@ def _wait_pairs(ptr, offsets, mask):; -120,6 +123,13 @@ def _publish_input_kernel(; symbols: _wait_pairs, _publish_input_kernel, rs_sconv_ag_add_norm
  - `vllm/models/inkling/nvidia/model.py` modified +18/-5 (23 lines); hunks: -5,7 +5,7; -57,14 +57,16; symbols: _layer_id, _sconv_add_norm, forward
  - `vllm/models/inkling/nvidia/moe.py` modified +5/-1 (6 lines); hunks: -501,7 +501,7 @@ def _select_routed(; -527,6 +527,10 @@ def forward(self, x: torch.Tensor) -> torch.Tensor | None:; symbols: _select_routed, forward, forward_partials
- Key code excerpts:

```diff
diff -- vllm/models/inkling/nvidia/ops/lamport.py
@@ -88,15 +88,18 @@ def _wait_pairs(ptr, offsets, mask):
+    shared_ptr,
+    stride_shared_t,
+    HAS_SHARED: tl.constexpr,
@@ -120,6 +123,13 @@ def _publish_input_kernel(
+        if HAS_SHARED:
+            shared = tl.load(
diff -- vllm/models/inkling/nvidia/model.py
@@ -5,7 +5,7 @@
-from typing import Any
+from typing import Any, TypeAlias
@@ -57,14 +57,16 @@
+InklingDelta: TypeAlias = torch.Tensor | tuple[torch.Tensor, torch.Tensor]
-    delta: torch.Tensor,
+    delta: InklingDelta,
diff -- vllm/models/inkling/nvidia/moe.py
@@ -501,7 +501,7 @@ def _select_routed(
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/inkling/nvidia/ops/lamport.py` modified +25/-0; `vllm/models/inkling/nvidia/model.py` modified +18/-5; `vllm/models/inkling/nvidia/moe.py` modified +5/-1
- Risk and verification: Runtime changes concentrate in `vllm/models/inkling/nvidia/model.py`, `vllm/models/inkling/nvidia/moe.py`, `vllm/models/inkling/nvidia/ops/lamport.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #50806 - [ROCm] Restore Inkling MTP backend parity

- Link: https://github.com/vllm-project/vllm/pull/50806
- Status/date: merged / 2026-08-05
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/inkling/amd/mtp.py`; associated commits `33c50587d267`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +42/-13, 93 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/inkling/amd/mtp.py` modified +42/-13 (55 lines); hunks: -2,9 +2,13; -54,6 +58,16 @@ def _mtp_depth_from_name(name: str) -> int | None:; symbols: _mtp_depth_from_name, _select_mtp_depth_count, InklingMTPDepthLayer, __init__, touching `_mtp_depth_from_name, _select_mtp_depth_count, InklingMTPDepthLayer`.
- Code diff details:
  - `vllm/models/inkling/amd/mtp.py` modified +42/-13 (55 lines); hunks: -2,9 +2,13; -54,6 +58,16 @@ def _mtp_depth_from_name(name: str) -> int | None:; symbols: _mtp_depth_from_name, _select_mtp_depth_count, InklingMTPDepthLayer, __init__
- Key code excerpts:

```diff
diff -- vllm/models/inkling/amd/mtp.py
@@ -2,9 +2,13 @@
-Implements the first MTP depth from the reference ``mtp_model.py`` shipped with
-the checkpoint. It owns ``hidden_norm`` / ``embed_norm`` RMSNorms, a ``2H -> H``
-input projection, and a full Inkling transformer block with a dense bf16 MLP.
+Mirrors the reference ``mtp_model.py`` shipped with the checkpoint: each MTP
+depth ``i`` owns ``hidden_norm`` / ``embed_norm`` RMSNorms, an ``input_proj``
+(``2H -> H``) and a full Inkling transformer block (dense bf16 MLP, with the
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/inkling/amd/mtp.py` modified +42/-13
- Risk and verification: Runtime changes concentrate in `vllm/models/inkling/amd/mtp.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #51391 - [Bugfix][Parser] Prevent Inkling block-end leakage with tools

- Link: https://github.com/vllm-project/vllm/pull/51391
- Status/date: merged / 2026-08-08
- Trace source: `git log --name-only -- <model-files>` found it through `tests/parser/engine/test_inkling.py`; associated commits `75231eff2f38`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +352/-30, 431 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/parser/engine/test_inkling.py` modified +212/-0 (212 lines); hunks: -19,6 +19,10; -140,6 +144,51 @@ def _collect_reasoning(results) -> str:; symbols: _collect_reasoning, _function_tool, _delegating, _stream_delegating, touching `_collect_reasoning, _function_tool, _delegating`; `vllm/parser/engine/streaming_parser_engine.py` modified +55/-30 (85 lines); hunks: -156,6 +156,13 @@ def __init__(; -186,6 +193,7 @@ def reset(self, initial_state: ParserState | None = None) ->...; symbols: __init__, reset, feed, _on_terminal, touching `__init__, reset, feed`.
- Code diff details:
  - `tests/parser/engine/test_inkling.py` modified +212/-0 (212 lines); hunks: -19,6 +19,10; -140,6 +144,51 @@ def _collect_reasoning(results) -> str:; symbols: _collect_reasoning, _function_tool, _delegating, _stream_delegating
  - `vllm/parser/engine/streaming_parser_engine.py` modified +55/-30 (85 lines); hunks: -156,6 +156,13 @@ def __init__(; -186,6 +193,7 @@ def reset(self, initial_state: ParserState | None = None) ->...; symbols: __init__, reset, feed, _on_terminal
- Key code excerpts:

```diff
diff -- tests/parser/engine/test_inkling.py
@@ -19,6 +19,10 @@
+from vllm.entrypoints.openai.chat_completion.protocol import (
+    ChatCompletionToolsParam,
+    FunctionDefinition,
+)
@@ -140,6 +144,51 @@ def _collect_reasoning(results) -> str:
+def _function_tool(name: str = "get_weather") -> ChatCompletionToolsParam:
diff -- vllm/parser/engine/streaming_parser_engine.py
@@ -156,6 +156,13 @@ def __init__(
+        # TOOL_CALL_END may close an inner call rather than its lexical wrapper,
+        # as in MiniMax, so identify exits from state transitions instead.
+        self._tool_exit_terminals: frozenset[str] = frozenset(
+            terminal
+            for (state, terminal), tr in config.transitions.items()
+            if state in self._TOOL_STATES and tr.next_state not in self._TOOL_STATES
```

- Extracted files (not manually reviewed):
  - tests: `tests/parser/engine/test_inkling.py` modified +212/-0
  - runtime: `vllm/parser/engine/streaming_parser_engine.py` modified +55/-30
- Risk and verification: The diff ships test coverage in `tests/parser/engine/test_inkling.py`, `tests/parser/engine/test_parser_engine.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #49876 - [Bugfix][Parser] Confirm reasoning end when an Inkling content block opens

- Link: https://github.com/vllm-project/vllm/pull/49876
- Status/date: merged / 2026-08-08
- Trace source: `git log --name-only -- <model-files>` found it through `tests/parser/engine/test_inkling.py`, `vllm/parser/inkling.py`; associated commits `1c1077c6cc43`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +57/-11, 137 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/parser/engine/test_inkling.py` modified +46/-5 (51 lines); hunks: -166,9 +166,13 @@ def _delegating(mock_tokenizer, tools=None):; -186,7 +190,14 @@ def _stream_delegating(parser, request, text, chunk_size, p...; symbols: _delegating, _stream_delegating, TestArgConverter, test_plain_text_non_streaming, touching `_delegating, _stream_delegating, TestArgConverter`; `vllm/parser/inkling.py` modified +11/-6 (17 lines); hunks: -189,9 +189,14 @@ def inkling_config() -> ParserEngineConfig:; -204,11 +209,11 @@ def inkling_config() -> ParserEngineConfig:; symbols: inkling_config, touching `inkling_config`.
- Code diff details:
  - `tests/parser/engine/test_inkling.py` modified +46/-5 (51 lines); hunks: -166,9 +166,13 @@ def _delegating(mock_tokenizer, tools=None):; -186,7 +190,14 @@ def _stream_delegating(parser, request, text, chunk_size, p...; symbols: _delegating, _stream_delegating, TestArgConverter, test_plain_text_non_streaming
  - `vllm/parser/inkling.py` modified +11/-6 (17 lines); hunks: -189,9 +189,14 @@ def inkling_config() -> ParserEngineConfig:; -204,11 +209,11 @@ def inkling_config() -> ParserEngineConfig:; symbols: inkling_config
- Key code excerpts:

```diff
diff -- tests/parser/engine/test_inkling.py
@@ -166,9 +166,13 @@ def _delegating(mock_tokenizer, tools=None):
-    tokens per delta; return ``(content, reasoning, ordered tool names)``."""
+    tokens per delta; return ``(content, reasoning, ordered tool names,
+    ordered tool arguments)``. Arguments arrive in fragments, so they are
+    concatenated per tool index."""
-    content, reasoning, tools = "", "", {}
+    content, reasoning = "", ""
diff -- vllm/parser/inkling.py
@@ -189,9 +189,14 @@ def inkling_config() -> ParserEngineConfig:
+        # Opening a block that renders as visible content proves no reasoning
+        # is still open. Confirming that here is what lets DelegatingParser
+        # hand the following blocks to the tool pass: the reasoning pass only
+        # leaves its reasoning phase on an explicit reasoning-end event, and a
+        # response with no thinking block never emits one otherwise.
-            (),
```

- Extracted files (not manually reviewed):
  - tests: `tests/parser/engine/test_inkling.py` modified +46/-5
  - runtime: `vllm/parser/inkling.py` modified +11/-6
- Risk and verification: The diff ships test coverage in `tests/parser/engine/test_inkling.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #50528 - [Bugfix][Parser] Emit REASONING_END for Inkling tool calls that follow no thinking block

- Link: https://github.com/vllm-project/vllm/pull/50528
- Status/date: merged / 2026-08-10
- Trace source: `git log --name-only -- <model-files>` found it through `tests/parser/engine/test_inkling.py`, `vllm/parser/inkling.py`; associated commits `0820125ae99f`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 4 files, +160/-12, 219 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/parser/engine/test_inkling.py` modified +84/-1 (85 lines); hunks: -23,8 +23,10; -786,3 +788,84 @@ def test_visible_text_then_tool_streaming(; symbols: test_visible_text_then_tool_streaming, test_tool_start_from_message_header_streaming, test_function_name_header_before_tool_start_streaming, test_content_state_tool_start_streaming, touching `test_visible_text_then_tool_streaming, test_tool_start_from_message_header_streaming, test_function_name_header_before_tool_start_streaming`; `vllm/parser/inkling.py` modified +4/-2 (6 lines); hunks: -202,9 +202,11 @@ def inkling_config() -> ParserEngineConfig:; -231,7 +233,7 @@ def inkling_config() -> ParserEngineConfig:; symbols: inkling_config, touching `inkling_config`.
- Code diff details:
  - `tests/parser/engine/test_inkling.py` modified +84/-1 (85 lines); hunks: -23,8 +23,10; -786,3 +788,84 @@ def test_visible_text_then_tool_streaming(; symbols: test_visible_text_then_tool_streaming, test_tool_start_from_message_header_streaming, test_function_name_header_before_tool_start_streaming, test_content_state_tool_start_streaming
  - `vllm/parser/inkling.py` modified +4/-2 (6 lines); hunks: -202,9 +202,11 @@ def inkling_config() -> ParserEngineConfig:; -231,7 +233,7 @@ def inkling_config() -> ParserEngineConfig:; symbols: inkling_config
- Key code excerpts:

```diff
diff -- tests/parser/engine/test_inkling.py
@@ -23,8 +23,10 @@
+from vllm.parser.engine.events import EventType
-from vllm.parser.inkling import InklingParser, _inkling_arg_converter
+from vllm.parser.engine.streaming_parser_engine import StreamingParserEngine
+from vllm.parser.inkling import InklingParser, _inkling_arg_converter, inkling_config
@@ -786,3 +788,84 @@ def test_visible_text_then_tool_streaming(
+    @pytest.mark.parametrize("chunk_size", [1, 3, 7, 64])
diff -- vllm/parser/inkling.py
@@ -202,9 +202,11 @@ def inkling_config() -> ParserEngineConfig:
+        # A tool block confirms the same boundary, and is the one opener that
+        # can start a turn with no visible block ahead of it.
-            (EventType.TOOL_CALL_START,),
+            (EventType.REASONING_END, EventType.TOOL_CALL_START),
@@ -231,7 +233,7 @@ def inkling_config() -> ParserEngineConfig:
-            (EventType.TOOL_CALL_START,),
```

- Extracted files (not manually reviewed):
  - tests: `tests/parser/engine/test_inkling.py` modified +84/-1
  - runtime: `vllm/parser/inkling.py` modified +4/-2
- Risk and verification: The diff ships test coverage in `tests/parser/engine/test_engine.py`, `tests/parser/engine/test_inkling.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #49315 - [2/N][Feat][Perf] Add new warmup infrastructure for JITs. Add predicate filtering for JIT warmup, and migrate Inkling FA4

- Link: https://github.com/vllm-project/vllm/pull/49315
- Status/date: merged / 2026-08-11
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/inkling/test_fa4_rel_attention.py`, `tests/models/inkling/test_fa4_warmup.py`, `vllm/models/inkling/nvidia/attention.py`, `vllm/models/inkling/nvidia/ops/__init__.py`, `vllm/models/inkling/nvidia/ops/fa4_rel_attention.py`; associated commits `6c95a641e95c`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 12 files, +743/-390, 1498 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/inkling/nvidia/ops/fa4_rel_attention.py` modified +333/-67 (400 lines); hunks: -3,12 +3,23; -98,73 +109,328 @@ def inkling_fa4_num_splits(; symbols: bucket_max_seqlen_q, inkling_fa4_num_splits, inkling_fa4_rel_attention, _num_warps_bucket, touching `bucket_max_seqlen_q, inkling_fa4_num_splits, inkling_fa4_rel_attention`; `tests/models/inkling/test_fa4_warmup.py` modified +84/-29 (113 lines); hunks: -1,17 +1,47; -28,8 +58,13 @@ def test_bucket_max_seqlen_q():; symbols: _vllm_config_from_reference_config, test_bucket_max_seqlen_q, test_warmup_enumerates_every_runtime_compile_class, touching `_vllm_config_from_reference_config, test_bucket_max_seqlen_q, test_warmup_enumerates_every_runtime_compile_class`; `vllm/models/inkling/nvidia/attention.py` modified +2/-22 (24 lines); hunks: -35,11 +35,10; -175,25 +174,6 @@ def __init__(; symbols: __init__, get_attn_backend, _attention, touching `__init__, get_attn_backend, _attention`; `tests/models/inkling/test_fa4_rel_attention.py` modified +4/-4 (8 lines); hunks: -2,8 +2,8; -23,9 +23,9; symbols: _run_case, touching `_run_case`.
- Code diff details:
  - `vllm/models/inkling/nvidia/ops/fa4_rel_attention.py` modified +333/-67 (400 lines); hunks: -3,12 +3,23; -98,73 +109,328 @@ def inkling_fa4_num_splits(; symbols: bucket_max_seqlen_q, inkling_fa4_num_splits, inkling_fa4_rel_attention, _num_warps_bucket
  - `tests/models/inkling/test_fa4_warmup.py` modified +84/-29 (113 lines); hunks: -1,17 +1,47; -28,8 +58,13 @@ def test_bucket_max_seqlen_q():; symbols: _vllm_config_from_reference_config, test_bucket_max_seqlen_q, test_warmup_enumerates_every_runtime_compile_class
  - `vllm/models/inkling/nvidia/attention.py` modified +2/-22 (24 lines); hunks: -35,11 +35,10; -175,25 +174,6 @@ def __init__(; symbols: __init__, get_attn_backend, _attention
  - `tests/models/inkling/test_fa4_rel_attention.py` modified +4/-4 (8 lines); hunks: -2,8 +2,8; -23,9 +23,9; symbols: _run_case
  - `vllm/models/inkling/nvidia/ops/__init__.py` modified +4/-2 (6 lines); hunks: -15,11 +15,13
- Key code excerpts:

```diff
diff -- vllm/models/inkling/nvidia/ops/fa4_rel_attention.py
@@ -3,12 +3,23 @@
+from dataclasses import dataclass
-from typing import Any
+from typing import TYPE_CHECKING, Any
+from vllm.distributed import get_tensor_model_parallel_world_size
+from vllm.model_executor.warmup.jit_warmup import (
+    VllmJitKernel,
diff -- tests/models/inkling/test_fa4_warmup.py
@@ -1,17 +1,47 @@
+from types import SimpleNamespace
+import vllm.models.inkling.nvidia.ops.fa4_rel_attention as fa4_rel_attention
+from vllm.models.inkling.configs import InklingMMConfig, InklingModelConfig
+    InklingFA4RelAttentionKernel,
+    _num_warps_bucket,
-from vllm.models.inkling.nvidia.ops.fa4_warmup import (
diff -- vllm/models/inkling/nvidia/attention.py
@@ -35,11 +35,10 @@
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/inkling/nvidia/ops/fa4_rel_attention.py` modified +333/-67; `vllm/models/inkling/nvidia/attention.py` modified +2/-22; `vllm/models/inkling/nvidia/ops/__init__.py` modified +4/-2
  - tests: `tests/models/inkling/test_fa4_warmup.py` modified +84/-29; `tests/models/inkling/test_fa4_rel_attention.py` modified +4/-4
- Risk and verification: The diff ships test coverage in `tests/model_executor/test_jit_warmup.py`, `tests/models/inkling/test_fa4_rel_attention.py`, `tests/models/inkling/test_fa4_warmup.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #51850 - [Bugfix] Support HF-config compat for Inkling

- Link: https://github.com/vllm-project/vllm/pull/51850
- Status/date: merged / 2026-08-11
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/inkling/configs.py`; associated commits `ded6c452c365`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +19/-1, 43 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/inkling/configs.py` modified +19/-1 (20 lines); hunks: -22,6 +22,7 @@ def __init__(; -41,7 +42,8 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/models/inkling/configs.py` modified +19/-1 (20 lines); hunks: -22,6 +22,7 @@ def __init__(; -41,7 +42,8 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/models/inkling/configs.py
@@ -22,6 +22,7 @@ def __init__(
+        moe_intermediate_size: int | None = None,
@@ -41,7 +42,8 @@ def __init__(
-        sconv_kernel_size: int = 4,
+        sconv_kernel_size: int | None = None,
+        conv_kernel_size: int | None = None,
@@ -77,8 +79,24 @@ def __init__(
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/inkling/configs.py` modified +19/-1
- Risk and verification: Runtime changes concentrate in `vllm/models/inkling/configs.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #58792 - [Bugfix][Frontend] Fix Inkling tool name leaking into content after reasoning

- Link: https://github.com/vllm-project/vllm/pull/58792
- Status/date: merged / 2026-09-26
- Trace source: `git log --name-only -- <model-files>` found it through `tests/parser/engine/test_inkling.py`, `vllm/parser/inkling.py`; associated commits `fdcce47e9bd9`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +38/-3, 81 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/parser/engine/test_inkling.py` modified +34/-0 (34 lines); hunks: -23,8 +23,13; -166,6 +171,11 @@ def _delegating(mock_tokenizer, tools=None):; symbols: _delegating, _TwoPassInklingParser, _stream_delegating, test_reasoning_then_tool_streaming, touching `_delegating, _TwoPassInklingParser, _stream_delegating`; `vllm/parser/inkling.py` modified +4/-3 (7 lines); hunks: -167,19 +167,20 @@ def inkling_config() -> ParserEngineConfig:; symbols: inkling_config, touching `inkling_config`.
- Code diff details:
  - `tests/parser/engine/test_inkling.py` modified +34/-0 (34 lines); hunks: -23,8 +23,13; -166,6 +171,11 @@ def _delegating(mock_tokenizer, tools=None):; symbols: _delegating, _TwoPassInklingParser, _stream_delegating, test_reasoning_then_tool_streaming
  - `vllm/parser/inkling.py` modified +4/-3 (7 lines); hunks: -167,19 +167,20 @@ def inkling_config() -> ParserEngineConfig:; symbols: inkling_config
- Key code excerpts:

```diff
diff -- tests/parser/engine/test_inkling.py
@@ -23,8 +23,13 @@
+from vllm.parser.abstract_parser import DelegatingParser
+from vllm.parser.engine.registered_adapters import (
+    InklingParserReasoningAdapter,
+    InklingParserToolAdapter,
+)
@@ -166,6 +171,11 @@ def _delegating(mock_tokenizer, tools=None):
diff -- vllm/parser/inkling.py
@@ -167,19 +167,20 @@ def inkling_config() -> ParserEngineConfig:
-    # Block-end terminals behave identically regardless of label. A
-    # closed tool block returns to CONTENT (Inkling has no section wrapper;
+    # A closed tool block returns to CONTENT (Inkling has no section wrapper;
+    # In a reasoning block, THINK_END defers REASONING_END to the next block
+    # opener so the function-name header is not emitted as content.
-            (EventType.REASONING_END,),
```

- Extracted files (not manually reviewed):
  - tests: `tests/parser/engine/test_inkling.py` modified +34/-0
  - runtime: `vllm/parser/inkling.py` modified +4/-3
- Risk and verification: The diff ships test coverage in `tests/parser/engine/test_inkling.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
