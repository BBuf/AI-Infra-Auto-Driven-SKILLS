# vLLM GLM-5 Series (5/5.1/5.2/5.3-Flash) Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `benchmarks/attention_benchmarks/configs/mla_sparse_masked_mha_vs_mqa_glm5.yaml` | [#53785](https://github.com/vllm-project/vllm/pull/53785) |
| `tests/evals/gsm8k/configs/GLM-5.2-NVFP4-HiSparse.yaml` | no direct PR-number commit |
| `tests/evals/gsm8k/configs/GLM-5.2-NVFP4-TP1-PCP4-DCP4-EP.yaml` | no direct PR-number commit |
| `tests/evals/gsm8k/configs/GLM-5.2-NVFP4-TP1-PCP4-EP.yaml` | [#51015](https://github.com/vllm-project/vllm/pull/51015) |
| `tests/evals/gsm8k/configs/GLM-5.2-NVFP4-TP2-PCP2-EP.yaml` | no direct PR-number commit |
| `tests/models/glm5next/__init__.py` | [#55736](https://github.com/vllm-project/vllm/pull/55736) |
| `tests/models/glm5next/test_kda_recurrent.py` | [#55736](https://github.com/vllm-project/vllm/pull/55736), [#56960](https://github.com/vllm-project/vllm/pull/56960), [#57979](https://github.com/vllm-project/vllm/pull/57979) |
| `tests/models/glm5next/test_sequence_parallel.py` | [#57387](https://github.com/vllm-project/vllm/pull/57387), [#58061](https://github.com/vllm-project/vllm/pull/58061) |
| `tests/models/glm5next/test_sparse_indexer_topk_dispatch.py` | [#57327](https://github.com/vllm-project/vllm/pull/57327), [#57546](https://github.com/vllm-project/vllm/pull/57546), [#58167](https://github.com/vllm-project/vllm/pull/58167) |
| `tests/models/multimodal/processing/test_glm5next.py` | [#55389](https://github.com/vllm-project/vllm/pull/55389), [#55647](https://github.com/vllm-project/vllm/pull/55647), [#57387](https://github.com/vllm-project/vllm/pull/57387), [#59565](https://github.com/vllm-project/vllm/pull/59565) |
| `tests/v1/attention/test_rocm_glm5next_sparse.py` | [#53906](https://github.com/vllm-project/vllm/pull/53906), [#55239](https://github.com/vllm-project/vllm/pull/55239), [#58008](https://github.com/vllm-project/vllm/pull/58008) |
| `vllm/models/deepseek_v32/nvidia/glm52_low_latency_gemm.py` | [#49791](https://github.com/vllm-project/vllm/pull/49791) |
| `vllm/models/glm5next/__init__.py` | [#53906](https://github.com/vllm-project/vllm/pull/53906), [#56176](https://github.com/vllm-project/vllm/pull/56176) |
| `vllm/models/glm5next/amd/__init__.py` | [#53906](https://github.com/vllm-project/vllm/pull/53906) |
| `vllm/models/glm5next/amd/ops/__init__.py` | [#53906](https://github.com/vllm-project/vllm/pull/53906) |
| `vllm/models/glm5next/amd/ops/kpool_compress.py` | [#53906](https://github.com/vllm-project/vllm/pull/53906), [#55214](https://github.com/vllm-project/vllm/pull/55214), [#58454](https://github.com/vllm-project/vllm/pull/58454) |
| `vllm/models/glm5next/amd/ops/third_party/__init__.py` | [#53906](https://github.com/vllm-project/vllm/pull/53906) |
| `vllm/models/glm5next/amd/ops/third_party/kda/__init__.py` | [#53906](https://github.com/vllm-project/vllm/pull/53906) |
| `vllm/models/glm5next/amd/ops/third_party/kda/fused_recurrent.py` | [#53906](https://github.com/vllm-project/vllm/pull/53906), [#57979](https://github.com/vllm-project/vllm/pull/57979) |
| `vllm/models/glm5next/amd/ops/third_party/kda/kernels.py` | [#53906](https://github.com/vllm-project/vllm/pull/53906), [#57979](https://github.com/vllm-project/vllm/pull/57979) |
| `vllm/models/glm5next/amd/sparse_indexer.py` | [#55358](https://github.com/vllm-project/vllm/pull/55358), [#57425](https://github.com/vllm-project/vllm/pull/57425), [#57701](https://github.com/vllm-project/vllm/pull/57701), [#58167](https://github.com/vllm-project/vllm/pull/58167) |
| `vllm/models/glm5next/common/__init__.py` | [#56176](https://github.com/vllm-project/vllm/pull/56176) |
| `vllm/models/glm5next/common/attention.py` | [#55222](https://github.com/vllm-project/vllm/pull/55222), [#55358](https://github.com/vllm-project/vllm/pull/55358), [#56176](https://github.com/vllm-project/vllm/pull/56176), [#57387](https://github.com/vllm-project/vllm/pull/57387), [#57701](https://github.com/vllm-project/vllm/pull/57701), [#58454](https://github.com/vllm-project/vllm/pull/58454) |
| `vllm/models/glm5next/common/kda.py` | [#56176](https://github.com/vllm-project/vllm/pull/56176), [#56960](https://github.com/vllm-project/vllm/pull/56960), [#57387](https://github.com/vllm-project/vllm/pull/57387) |
| `vllm/models/glm5next/common/model.py` | [#55389](https://github.com/vllm-project/vllm/pull/55389), [#56176](https://github.com/vllm-project/vllm/pull/56176), [#57387](https://github.com/vllm-project/vllm/pull/57387), [#58061](https://github.com/vllm-project/vllm/pull/58061) |
| `vllm/models/glm5next/common/mtp.py` | [#56176](https://github.com/vllm-project/vllm/pull/56176), [#57387](https://github.com/vllm-project/vllm/pull/57387) |
| `vllm/models/glm5next/common/multimodal.py` | [#55389](https://github.com/vllm-project/vllm/pull/55389), [#55647](https://github.com/vllm-project/vllm/pull/55647), [#56176](https://github.com/vllm-project/vllm/pull/56176), [#57387](https://github.com/vllm-project/vllm/pull/57387), [#59126](https://github.com/vllm-project/vllm/pull/59126), [#59565](https://github.com/vllm-project/vllm/pull/59565) |
| `vllm/models/glm5next/common/sparse_indexer.py` | [#55358](https://github.com/vllm-project/vllm/pull/55358) |
| `vllm/models/glm5next/nvidia/__init__.py` | [#55214](https://github.com/vllm-project/vllm/pull/55214) |
| `vllm/models/glm5next/nvidia/ops/__init__.py` | [#55214](https://github.com/vllm-project/vllm/pull/55214) |
| `vllm/models/glm5next/nvidia/ops/fused_eh_norm.py` | [#53906](https://github.com/vllm-project/vllm/pull/53906) |
| `vllm/models/glm5next/nvidia/ops/kpool_compress.py` | [#53906](https://github.com/vllm-project/vllm/pull/53906), [#55214](https://github.com/vllm-project/vllm/pull/55214), [#57477](https://github.com/vllm-project/vllm/pull/57477), [#58454](https://github.com/vllm-project/vllm/pull/58454) |
| `vllm/models/glm5next/nvidia/ops/third_party/__init__.py` | [#53906](https://github.com/vllm-project/vllm/pull/53906) |
| `vllm/models/glm5next/nvidia/ops/third_party/kda/__init__.py` | [#53906](https://github.com/vllm-project/vllm/pull/53906) |
| `vllm/models/glm5next/nvidia/ops/third_party/kda/fused_recurrent.py` | [#53906](https://github.com/vllm-project/vllm/pull/53906), [#55736](https://github.com/vllm-project/vllm/pull/55736) |
| `vllm/models/glm5next/nvidia/ops/third_party/kda/kernels.py` | [#53906](https://github.com/vllm-project/vllm/pull/53906), [#55736](https://github.com/vllm-project/vllm/pull/55736) |
| `vllm/models/glm5next/nvidia/sparse_indexer.py` | [#55270](https://github.com/vllm-project/vllm/pull/55270), [#55358](https://github.com/vllm-project/vllm/pull/55358), [#57327](https://github.com/vllm-project/vllm/pull/57327), [#57546](https://github.com/vllm-project/vllm/pull/57546), [#57701](https://github.com/vllm-project/vllm/pull/57701) |
| `vllm/models/glm5next/sparse_indexer.py` | [#55358](https://github.com/vllm-project/vllm/pull/55358) |

## PR Coverage Summary

- Git-traced PRs: 27
- Extra PRs preserved from existing docs: 0
- Total PRs in this document: 27
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-08-03 | [#49791](https://github.com/vllm-project/vllm/pull/49791) | merged | [Kernel] Extend CuTe DSL skinny GEMM to GLM-5.2 | `vllm/models/deepseek_v32/nvidia/glm52_low_latency_gemm.py` |
| 2026-08-04 | [#51015](https://github.com/vllm-project/vllm/pull/51015) | merged | [CI] Stabilize GLM-5.2 PCP evaluation | `tests/evals/gsm8k/configs/GLM-5.2-NVFP4-TP1-PCP4-EP.yaml` |
| 2026-08-27 | [#53785](https://github.com/vllm-project/vllm/pull/53785) | merged | [Attention] Enable dense and masked MHA for GLM-5 | `benchmarks/attention_benchmarks/configs/mla_sparse_masked_mha_vs_mqa_glm5.yaml` |
| 2026-09-03 | [#53906](https://github.com/vllm-project/vllm/pull/53906) | merged | [Model] add GLM-5.3-Flash support | `vllm/models/glm5next/amd/ops/third_party/kda/kernels.py`, `vllm/models/glm5next/nvidia/ops/third_party/kda/kernels.py`, `vllm/models/glm5next/nvidia/ops/kpool_compress.py` |
| 2026-09-04 | [#55214](https://github.com/vllm-project/vllm/pull/55214) | merged | [Bugfix][Docs] Package glm5next nvidia subtree and fix its docstrings | `vllm/models/glm5next/amd/ops/kpool_compress.py`, `vllm/models/glm5next/nvidia/ops/kpool_compress.py`, `vllm/models/glm5next/nvidia/__init__.py` |
| 2026-09-10 | [#55736](https://github.com/vllm-project/vllm/pull/55736) | merged | [Perf][GLM-5.3-Flash] Decode hot-path cleanups: strided KDA recurrent inputs, NoPE MQA query without concat, no duplicate router GEMM | `tests/models/glm5next/test_kda_recurrent.py`, `vllm/models/glm5next/nvidia/ops/third_party/kda/fused_recurrent.py`, `vllm/models/glm5next/nvidia/ops/third_party/kda/kernels.py` |
| 2026-09-11 | [#55239](https://github.com/vllm-project/vllm/pull/55239) | merged | [ROCm][Bugfix] Route GLM-5.3-Flash MTP through ragged sparse MLA | `tests/v1/attention/test_rocm_glm5next_sparse.py`, `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py` |
| 2026-09-16 | [#56176](https://github.com/vllm-project/vllm/pull/56176) | merged | [ROCm] [Bugfix] Enable Load and Inference of GLM-5.3-Flash Quark MXFP4 Checkpoint | `vllm/models/glm5next/common/model.py`, `vllm/models/glm5next/__init__.py`, `vllm/models/glm5next/common/__init__.py` |
| 2026-09-16 | [#55358](https://github.com/vllm-project/vllm/pull/55358) | merged | [Refactor][GLM-5.3-Flash] Move sparse_attn_indexer_kpool into the model folder and split AMD/NVIDIA | `vllm/models/glm5next/amd/sparse_indexer.py`, `vllm/models/glm5next/nvidia/sparse_indexer.py`, `vllm/models/glm5next/common/sparse_indexer.py` |
| 2026-09-18 | [#57327](https://github.com/vllm-project/vllm/pull/57327) | merged | [Perf][GLM5.3-Flash] Use cooperative top-k for small GLM decode batches | `tests/models/glm5next/test_sparse_indexer_topk_dispatch.py`, `vllm/models/glm5next/nvidia/sparse_indexer.py` |
| 2026-09-18 | [#57425](https://github.com/vllm-project/vllm/pull/57425) | merged | [Bugfix][ROCm] Alias SparseAttnIndexerKpool.forward_cuda to forward_native (GLM-5.3-Flash boot crash) | `vllm/models/glm5next/amd/sparse_indexer.py` |
| 2026-09-19 | [#57701](https://github.com/vllm-project/vllm/pull/57701) | merged | [GLM5.3 Perf] Size the GLM-5 sparse indexer decode workspace, 3072 MiB GPU memory saved | `vllm/models/glm5next/amd/sparse_indexer.py`, `vllm/models/glm5next/nvidia/sparse_indexer.py`, `vllm/models/glm5next/common/attention.py` |
| 2026-09-20 | [#57477](https://github.com/vllm-project/vllm/pull/57477) | merged | [Bugfix][GLM-5.3-Flash] Address kpool tail blocks by the padded indexer stride in the NVIDIA prefill seed kernel | `vllm/models/glm5next/nvidia/ops/kpool_compress.py` |
| 2026-09-20 | [#57546](https://github.com/vllm-project/vllm/pull/57546) | merged | [GLM-5.3-Flash] Route kpool indexer top-k through the shared SparseIndexerTopk dispatcher | `vllm/models/glm5next/nvidia/sparse_indexer.py`, `tests/models/glm5next/test_sparse_indexer_topk_dispatch.py` |
| 2026-09-22 | [#58061](https://github.com/vllm-project/vllm/pull/58061) | merged | [Bugfix][GLM-5.3-Flash] Run the dense MLP layers on the sequence-parallel shard | `tests/models/glm5next/test_sequence_parallel.py`, `vllm/models/glm5next/common/model.py` |
| 2026-09-25 | [#55270](https://github.com/vllm-project/vllm/pull/55270) | merged | [Bugfix] GLM-5.3-Flash: launch the kpool paged MQA logits in the varlen mode its schedule was built with | `vllm/models/glm5next/nvidia/sparse_indexer.py` |
| 2026-09-25 | [#58454](https://github.com/vllm-project/vllm/pull/58454) | merged | [Bugfix][GLM-5.3-Flash] kpool corruption with speculative decoding | `vllm/models/glm5next/amd/ops/kpool_compress.py`, `vllm/models/glm5next/nvidia/ops/kpool_compress.py`, `vllm/models/glm5next/common/attention.py` |
| 2026-09-29 | [#55647](https://github.com/vllm-project/vllm/pull/55647) | merged | [Bugfix][GLM-5.3-Flash] Take video placeholder timestamps from the pixel path's frame sampler | `tests/models/multimodal/processing/test_glm5next.py`, `vllm/models/glm5next/common/multimodal.py` |
| 2026-09-29 | [#56960](https://github.com/vllm-project/vllm/pull/56960) | merged | [Feat][Model] Enable KDA prefill checkpoints for GLM-5.3-Flash | `tests/models/glm5next/test_kda_recurrent.py`, `vllm/models/glm5next/common/kda.py` |
| 2026-09-29 | [#55222](https://github.com/vllm-project/vllm/pull/55222) | merged | [Bugfix] GLM-5.3-Flash: fp8 plan dtype on SM90 sparse MLA, and right-size the indexer prefill workspace | `vllm/models/glm5next/common/attention.py` |
| 2026-09-30 | [#55389](https://github.com/vllm-project/vllm/pull/55389) | merged | [Model] Extend device-side mm normalization to GLM4V/GLM5Next | `tests/models/multimodal/processing/test_glm5next.py`, `vllm/models/glm5next/common/multimodal.py`, `vllm/models/glm5next/common/model.py` |
| 2026-09-30 | [#57979](https://github.com/vllm-project/vllm/pull/57979) | merged | [ROCm][Perf][GLM-5.3-Flash] Stride-aware decode KDA | `vllm/models/glm5next/amd/ops/third_party/kda/fused_recurrent.py`, `vllm/models/glm5next/amd/ops/third_party/kda/kernels.py`, `tests/models/glm5next/test_kda_recurrent.py` |
| 2026-09-30 | [#58008](https://github.com/vllm-project/vllm/pull/58008) | merged | [ROCm][Perf][GLM-5.3-Flash] "Fit kpool top-k indices to AITER" with a single Triton kernel | `tests/v1/attention/test_rocm_glm5next_sparse.py`, `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py` |
| 2026-10-02 | [#57387](https://github.com/vllm-project/vllm/pull/57387) | merged | [Model] Use upstream GLM-5.3 and Qwen4-Exp configs and processor | `tests/transformers_utils/processors/test_glm5next.py`, `vllm/models/glm5next/common/multimodal.py`, `vllm/models/glm5next/common/model.py` |
| 2026-10-02 | [#59565](https://github.com/vllm-project/vllm/pull/59565) | merged | [Bugfix][GLM-5.3] Size the image encoder cache from the exact token ceiling | `tests/models/multimodal/processing/test_glm5next.py`, `vllm/models/glm5next/common/multimodal.py` |
| 2026-10-02 | [#58167](https://github.com/vllm-project/vllm/pull/58167) | merged | [ROCm][Perf][GLM-5.3-Flash] Add AITER topk backend for decodes | `tests/models/glm5next/test_sparse_indexer_topk_dispatch.py`, `vllm/models/glm5next/amd/sparse_indexer.py` |
| 2026-10-02 | [#59126](https://github.com/vllm-project/vllm/pull/59126) | merged | [Bugfix][Multimodal] Fix GLM-5.3-Flash vision tower crashes on image input | `vllm/models/glm5next/common/multimodal.py` |

## Per-PR Diff Audit Cards

### PR #49791 - [Kernel] Extend CuTe DSL skinny GEMM to GLM-5.2

- Link: https://github.com/vllm-project/vllm/pull/49791
- Status/date: merged / 2026-08-03
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/deepseek_v32/nvidia/glm52_low_latency_gemm.py`; associated commits `c8109375733e`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 7 files, +619/-51, 965 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/deepseek_v32/nvidia/glm52_low_latency_gemm.py` added +199/-0 (199 lines); hunks: -0,0 +1,199; symbols: GLM52ProjectionSpec, build_plan, _is_sm103, _is_supported_row_major, touching `GLM52ProjectionSpec, build_plan, _is_sm103`.
- Code diff details:
  - `vllm/models/deepseek_v32/nvidia/glm52_low_latency_gemm.py` added +199/-0 (199 lines); hunks: -0,0 +1,199; symbols: GLM52ProjectionSpec, build_plan, _is_sm103, _is_supported_row_major
- Key code excerpts:

```diff
diff -- vllm/models/deepseek_v32/nvidia/glm52_low_latency_gemm.py
@@ -0,0 +1,199 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""GLM-5.2 decode GEMM selection for unquantized BF16 on SM103."""
+from __future__ import annotations
+from dataclasses import dataclass
+from typing import Literal
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/deepseek_v32/nvidia/glm52_low_latency_gemm.py` added +199/-0
- Risk and verification: The diff ships test coverage in `tests/kernels/test_bf16_skinny_gemm.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #51015 - [CI] Stabilize GLM-5.2 PCP evaluation

- Link: https://github.com/vllm-project/vllm/pull/51015
- Status/date: merged / 2026-08-04
- Trace source: `git log --name-only -- <model-files>` found it through `tests/evals/gsm8k/configs/GLM-5.2-NVFP4-TP1-PCP4-EP.yaml`; associated commits `a5149b2feeb7`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +1/-0, 7 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/evals/gsm8k/configs/GLM-5.2-NVFP4-TP1-PCP4-EP.yaml` modified +1/-0 (1 lines); hunks: -13,5 +13,6 @@ server_args: >-.
- Code diff details:
  - `tests/evals/gsm8k/configs/GLM-5.2-NVFP4-TP1-PCP4-EP.yaml` modified +1/-0 (1 lines); hunks: -13,5 +13,6 @@ server_args: >-
- Key code excerpts:

```diff
diff -- tests/evals/gsm8k/configs/GLM-5.2-NVFP4-TP1-PCP4-EP.yaml
@@ -13,5 +13,6 @@ server_args: >-
+  PYTORCH_CUDA_ALLOC_CONF: "expandable_segments:True"
```

- Extracted files (not manually reviewed):
  - tests: `tests/evals/gsm8k/configs/GLM-5.2-NVFP4-TP1-PCP4-EP.yaml` modified +1/-0
- Risk and verification: The diff ships test coverage in `tests/evals/gsm8k/configs/GLM-5.2-NVFP4-TP1-PCP4-EP.yaml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #53785 - [Attention] Enable dense and masked MHA for GLM-5

- Link: https://github.com/vllm-project/vllm/pull/53785
- Status/date: merged / 2026-08-27
- Trace source: `git log --name-only -- <model-files>` found it through `benchmarks/attention_benchmarks/configs/mla_sparse_masked_mha_vs_mqa_glm5.yaml`; associated commits `de9250ac9e9b`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 10 files, +527/-74, 846 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `benchmarks/attention_benchmarks/configs/mla_sparse_masked_mha_vs_mqa_glm5.yaml` added +59/-0 (59 lines); hunks: -0,0 +1,59.
- Code diff details:
  - `benchmarks/attention_benchmarks/configs/mla_sparse_masked_mha_vs_mqa_glm5.yaml` added +59/-0 (59 lines); hunks: -0,0 +1,59
- Key code excerpts:

```diff
diff -- benchmarks/attention_benchmarks/configs/mla_sparse_masked_mha_vs_mqa_glm5.yaml
@@ -0,0 +1,59 @@
+mode: mha_vs_mqa
+model:
+  name: "glm-5.x"
+  num_layers: 78
+  num_q_heads: 64
+  num_kv_heads: 1
```

- Extracted files (not manually reviewed):
  - runtime: `benchmarks/attention_benchmarks/configs/mla_sparse_masked_mha_vs_mqa_glm5.yaml` added +59/-0
- Risk and verification: The diff ships test coverage in `tests/model_executor/layers/test_mla_short_prefill_indexer.py`, `tests/v1/attention/test_mla_backends.py`, `tests/v1/attention/test_sparse_mla_backends.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #53906 - [Model] add GLM-5.3-Flash support

- Link: https://github.com/vllm-project/vllm/pull/53906
- Status/date: merged / 2026-09-03
- Trace source: `git log --name-only -- <model-files>` found it through `tests/v1/attention/test_rocm_glm5next_sparse.py`, `vllm/models/glm5next/__init__.py`, `vllm/models/glm5next/amd/__init__.py`, `vllm/models/glm5next/amd/ops/__init__.py`, `vllm/models/glm5next/amd/ops/kpool_compress.py` and 15 files; associated commits `98ed0856f31f`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 94 files, +17926/-265, 19713 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/glm5next/amd/ops/third_party/kda/kernels.py` added +1362/-0 (1362 lines); hunks: -0,0 +1,1362; symbols: fused_recurrent_kda_fwd, fused_recurrent_kda, chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter, chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra, touching `fused_recurrent_kda_fwd, fused_recurrent_kda, chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter`; `vllm/models/glm5next/nvidia/ops/third_party/kda/kernels.py` added +1362/-0 (1362 lines); hunks: -0,0 +1,1362; symbols: fused_recurrent_kda_fwd, fused_recurrent_kda, chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter, chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra, touching `fused_recurrent_kda_fwd, fused_recurrent_kda, chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter`; `vllm/models/glm5next/nvidia/ops/kpool_compress.py` added +891/-0 (891 lines); hunks: -0,0 +1,891; symbols: _hadamard128_stage, _hadamard128, _fwht_stage, _fwht_quant_kernel, touching `_hadamard128_stage, _hadamard128, _fwht_stage`; `vllm/models/glm5next/amd/ops/kpool_compress.py` added +890/-0 (890 lines); hunks: -0,0 +1,890; symbols: _cache_k_offset, _hadamard128_stage, _hadamard128, compute_pooled_write_locs, touching `_cache_k_offset, _hadamard128_stage, _hadamard128`.
- Code diff details:
  - `vllm/models/glm5next/amd/ops/third_party/kda/kernels.py` added +1362/-0 (1362 lines); hunks: -0,0 +1,1362; symbols: fused_recurrent_kda_fwd, fused_recurrent_kda, chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter, chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra
  - `vllm/models/glm5next/nvidia/ops/third_party/kda/kernels.py` added +1362/-0 (1362 lines); hunks: -0,0 +1,1362; symbols: fused_recurrent_kda_fwd, fused_recurrent_kda, chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter, chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra
  - `vllm/models/glm5next/nvidia/ops/kpool_compress.py` added +891/-0 (891 lines); hunks: -0,0 +1,891; symbols: _hadamard128_stage, _hadamard128, _fwht_stage, _fwht_quant_kernel
  - `vllm/models/glm5next/amd/ops/kpool_compress.py` added +890/-0 (890 lines); hunks: -0,0 +1,890; symbols: _cache_k_offset, _hadamard128_stage, _hadamard128, compute_pooled_write_locs
  - `vllm/models/glm5next/amd/ops/third_party/kda/fused_recurrent.py` added +656/-0 (656 lines); hunks: -0,0 +1,656; symbols: fused_recurrent_gated_delta_rule_fwd_kernel, fused_recurrent_gated_delta_rule_fwd, fused_recurrent_gated_delta_rule_packed_decode_kernel, fused_recurrent_gated_delta_rule_packed_decode
- Key code excerpts:

```diff
diff -- vllm/models/glm5next/amd/ops/third_party/kda/kernels.py
@@ -0,0 +1,1362 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
+# mypy: ignore-errors
+#
+# This file contains code copied from the flash-linear-attention project.
diff -- vllm/models/glm5next/nvidia/ops/third_party/kda/kernels.py
@@ -0,0 +1,1362 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
+# mypy: ignore-errors
+#
+# This file contains code copied from the flash-linear-attention project.
diff -- vllm/models/glm5next/nvidia/ops/kpool_compress.py
@@ -0,0 +1,891 @@
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/glm5next/amd/ops/third_party/kda/kernels.py` added +1362/-0; `vllm/models/glm5next/nvidia/ops/third_party/kda/kernels.py` added +1362/-0; `vllm/models/glm5next/nvidia/ops/kpool_compress.py` added +891/-0; `vllm/models/glm5next/amd/ops/kpool_compress.py` added +890/-0; `vllm/models/glm5next/amd/ops/third_party/kda/fused_recurrent.py` added +656/-0; `vllm/models/glm5next/nvidia/ops/third_party/kda/fused_recurrent.py` added +656/-0
- Risk and verification: The diff ships test coverage in `tests/kernels/attention/test_flashinfer_mla_decode.py`, `tests/kernels/attention/test_rocm_aiter_mla_sparse_metadata_sync.py`, `tests/kernels/mamba/test_gdn_prefill_flashinfer.py`, `tests/kernels/test_kpool_decode_update_batched.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #55214 - [Bugfix][Docs] Package glm5next nvidia subtree and fix its docstrings

- Link: https://github.com/vllm-project/vllm/pull/55214
- Status/date: merged / 2026-09-04
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/glm5next/amd/ops/kpool_compress.py`, `vllm/models/glm5next/nvidia/__init__.py`, `vllm/models/glm5next/nvidia/ops/__init__.py`, `vllm/models/glm5next/nvidia/ops/kpool_compress.py`; associated commits `78300cdabffb`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 4 files, +12/-4, 34 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/glm5next/amd/ops/kpool_compress.py` modified +4/-2 (6 lines); hunks: -631,9 +631,11 @@ def kpool_decode_update_and_maybe_write_cache_batched(; symbols: kpool_decode_update_and_maybe_write_cache_batched, touching `kpool_decode_update_and_maybe_write_cache_batched`; `vllm/models/glm5next/nvidia/ops/kpool_compress.py` modified +4/-2 (6 lines); hunks: -637,9 +637,11 @@ def kpool_decode_update_and_maybe_write_cache_batched(; symbols: kpool_decode_update_and_maybe_write_cache_batched, touching `kpool_decode_update_and_maybe_write_cache_batched`; `vllm/models/glm5next/nvidia/__init__.py` added +2/-0 (2 lines); hunks: -0,0 +1,2; `vllm/models/glm5next/nvidia/ops/__init__.py` added +2/-0 (2 lines); hunks: -0,0 +1,2.
- Code diff details:
  - `vllm/models/glm5next/amd/ops/kpool_compress.py` modified +4/-2 (6 lines); hunks: -631,9 +631,11 @@ def kpool_decode_update_and_maybe_write_cache_batched(; symbols: kpool_decode_update_and_maybe_write_cache_batched
  - `vllm/models/glm5next/nvidia/ops/kpool_compress.py` modified +4/-2 (6 lines); hunks: -637,9 +637,11 @@ def kpool_decode_update_and_maybe_write_cache_batched(; symbols: kpool_decode_update_and_maybe_write_cache_batched
  - `vllm/models/glm5next/nvidia/__init__.py` added +2/-0 (2 lines); hunks: -0,0 +1,2
  - `vllm/models/glm5next/nvidia/ops/__init__.py` added +2/-0 (2 lines); hunks: -0,0 +1,2
- Key code excerpts:

```diff
diff -- vllm/models/glm5next/amd/ops/kpool_compress.py
@@ -631,9 +631,11 @@ def kpool_decode_update_and_maybe_write_cache_batched(
-        key / slot_score: ``[num_requests, next_n, head_dim]`` bf16.
+        key: ``[num_requests, next_n, head_dim]`` bf16.
+        slot_score: ``[num_requests, next_n, head_dim]`` bf16.
-        slot_mapping / positions: ``[num_requests, next_n]`` int32.
+        slot_mapping: ``[num_requests, next_n]`` int32.
+        positions: ``[num_requests, next_n]`` int32.
diff -- vllm/models/glm5next/nvidia/ops/kpool_compress.py
@@ -637,9 +637,11 @@ def kpool_decode_update_and_maybe_write_cache_batched(
-        key / slot_score: ``[num_requests, next_n, head_dim]`` bf16.
+        key: ``[num_requests, next_n, head_dim]`` bf16.
+        slot_score: ``[num_requests, next_n, head_dim]`` bf16.
-        slot_mapping / positions: ``[num_requests, next_n]`` int32.
+        slot_mapping: ``[num_requests, next_n]`` int32.
+        positions: ``[num_requests, next_n]`` int32.
diff -- vllm/models/glm5next/nvidia/__init__.py
@@ -0,0 +1,2 @@
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/glm5next/amd/ops/kpool_compress.py` modified +4/-2; `vllm/models/glm5next/nvidia/ops/kpool_compress.py` modified +4/-2; `vllm/models/glm5next/nvidia/__init__.py` added +2/-0; `vllm/models/glm5next/nvidia/ops/__init__.py` added +2/-0
- Risk and verification: Runtime changes concentrate in `vllm/models/glm5next/amd/ops/kpool_compress.py`, `vllm/models/glm5next/nvidia/__init__.py`, `vllm/models/glm5next/nvidia/ops/__init__.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #55736 - [Perf][GLM-5.3-Flash] Decode hot-path cleanups: strided KDA recurrent inputs, NoPE MQA query without concat, no duplicate router GEMM

- Link: https://github.com/vllm-project/vllm/pull/55736
- Status/date: merged / 2026-09-10
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/glm5next/__init__.py`, `tests/models/glm5next/test_kda_recurrent.py`, `vllm/models/glm5next/nvidia/ops/third_party/kda/fused_recurrent.py`, `vllm/models/glm5next/nvidia/ops/third_party/kda/kernels.py`; associated commits `1768273c13d8`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 8 files, +268/-27, 399 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/glm5next/test_kda_recurrent.py` added +186/-0 (186 lines); hunks: -0,0 +1,186; symbols: naive_recurrent_kda, make_inputs, run_kernel, test_fused_recurrent_kda_matches_reference, touching `naive_recurrent_kda, make_inputs, run_kernel`; `vllm/models/glm5next/nvidia/ops/third_party/kda/fused_recurrent.py` modified +35/-9 (44 lines); hunks: -15,6 +15,23; -52,6 +69,11 @@ def fused_recurrent_gated_delta_rule_fwd_kernel(; symbols: token_stride, fused_recurrent_gated_delta_rule_fwd_kernel, fused_recurrent_gated_delta_rule_fwd, touching `token_stride, fused_recurrent_gated_delta_rule_fwd_kernel, fused_recurrent_gated_delta_rule_fwd`; `vllm/models/glm5next/nvidia/ops/third_party/kda/kernels.py` modified +16/-6 (22 lines); hunks: -24,7 +24,10; -70,7 +73,7 @@ def fused_recurrent_kda_fwd(; symbols: fused_recurrent_kda_fwd, fused_recurrent_kda, touching `fused_recurrent_kda_fwd, fused_recurrent_kda`; `tests/models/glm5next/__init__.py` added +2/-0 (2 lines); hunks: -0,0 +1,2.
- Code diff details:
  - `tests/models/glm5next/test_kda_recurrent.py` added +186/-0 (186 lines); hunks: -0,0 +1,186; symbols: naive_recurrent_kda, make_inputs, run_kernel, test_fused_recurrent_kda_matches_reference
  - `vllm/models/glm5next/nvidia/ops/third_party/kda/fused_recurrent.py` modified +35/-9 (44 lines); hunks: -15,6 +15,23; -52,6 +69,11 @@ def fused_recurrent_gated_delta_rule_fwd_kernel(; symbols: token_stride, fused_recurrent_gated_delta_rule_fwd_kernel, fused_recurrent_gated_delta_rule_fwd
  - `vllm/models/glm5next/nvidia/ops/third_party/kda/kernels.py` modified +16/-6 (22 lines); hunks: -24,7 +24,10; -70,7 +73,7 @@ def fused_recurrent_kda_fwd(; symbols: fused_recurrent_kda_fwd, fused_recurrent_kda
  - `tests/models/glm5next/__init__.py` added +2/-0 (2 lines); hunks: -0,0 +1,2
- Key code excerpts:

```diff
diff -- tests/models/glm5next/test_kda_recurrent.py
@@ -0,0 +1,186 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""GLM-5.3-Flash KDA recurrent (decode) kernel.
+The decode path hands the kernel column slices of the merged ``q|k|v`` conv
+output and of the fused ``qkvbfg_a`` projection (beta), so q/k/v/beta are
+token-strided rather than contiguous. The kernel must read them in place,
diff -- vllm/models/glm5next/nvidia/ops/third_party/kda/fused_recurrent.py
@@ -15,6 +15,23 @@
+def token_stride(x: torch.Tensor) -> int:
+    """Token stride (elements) of a ``[B, T, H, D]`` or ``[B, T, H]`` tensor.
+    The recurrent kernel walks tokens with this stride and addresses heads
+    densely inside a token, so each token's ``[H, D]`` (or ``[H]``) block must
+    be contiguous, tokens must not overlap, and with ``B > 1`` sequence ``n``
+    must start at token ``n * T`` (dense batch). Column slices of a wider
diff -- vllm/models/glm5next/nvidia/ops/third_party/kda/kernels.py
@@ -24,7 +24,10 @@
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/glm5next/test_kda_recurrent.py` added +186/-0; `tests/models/glm5next/__init__.py` added +2/-0
  - runtime: `vllm/models/glm5next/nvidia/ops/third_party/kda/fused_recurrent.py` modified +35/-9; `vllm/models/glm5next/nvidia/ops/third_party/kda/kernels.py` modified +16/-6
- Risk and verification: The diff ships test coverage in `tests/models/glm5next/__init__.py`, `tests/models/glm5next/test_kda_recurrent.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #55239 - [ROCm][Bugfix] Route GLM-5.3-Flash MTP through ragged sparse MLA

- Link: https://github.com/vllm-project/vllm/pull/55239
- Status/date: merged / 2026-09-11
- Trace source: `git log --name-only -- <model-files>` found it through `tests/v1/attention/test_rocm_glm5next_sparse.py`; associated commits `828f4f19b4d8`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +11/-5, 40 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/v1/attention/test_rocm_glm5next_sparse.py` modified +4/-1 (5 lines); hunks: -76,7 +76,9 @@ def test_fit_kpool_indices_rejects_narrow_input():; -88,6 +90,7 @@ def test_rocm_sparse_triton_route(; symbols: test_fit_kpool_indices_rejects_narrow_input, test_rocm_sparse_triton_route, touching `test_fit_kpool_indices_rejects_narrow_input, test_rocm_sparse_triton_route`; `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py` modified +7/-4 (11 lines); hunks: -53,13 +53,16 @@ def _use_rocm_sparse_triton(; symbols: _use_rocm_sparse_triton, touching `_use_rocm_sparse_triton`.
- Code diff details:
  - `tests/v1/attention/test_rocm_glm5next_sparse.py` modified +4/-1 (5 lines); hunks: -76,7 +76,9 @@ def test_fit_kpool_indices_rejects_narrow_input():; -88,6 +90,7 @@ def test_rocm_sparse_triton_route(; symbols: test_fit_kpool_indices_rejects_narrow_input, test_rocm_sparse_triton_route
  - `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py` modified +7/-4 (11 lines); hunks: -53,13 +53,16 @@ def _use_rocm_sparse_triton(; symbols: _use_rocm_sparse_triton
- Key code excerpts:

```diff
diff -- tests/v1/attention/test_rocm_glm5next_sparse.py
@@ -76,7 +76,9 @@ def test_fit_kpool_indices_rejects_narrow_input():
-        ("auto", 512, 0, 2, 4, 2, False),
+        ("auto", 512, 0, 2, 4, 2, True),
+        ("auto", 512, 0, 2, 12, 6, True),
+        ("auto", 512, 0, 0, 0, 0, False),
@@ -88,6 +90,7 @@ def test_rocm_sparse_triton_route(
+    """Validate Triton routing for prefill, decode, and MTP verification."""
diff -- vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py
@@ -53,13 +53,16 @@ def _use_rocm_sparse_triton(
-    """Select the rope-free BF16 path not supported by AITER sparse MLA."""
-    plain_decode = num_decode_tokens == num_decodes
+    """Select the rope-free BF16 path not supported by AITER sparse MLA.
+    The ragged Triton kernel indexes metadata per query token, so multi-token
+    speculative verification rows have the same capability requirements as
+    plain decode rows.
```

- Extracted files (not manually reviewed):
  - tests: `tests/v1/attention/test_rocm_glm5next_sparse.py` modified +4/-1
  - runtime: `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py` modified +7/-4
- Risk and verification: The diff ships test coverage in `tests/v1/attention/test_rocm_glm5next_sparse.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #56176 - [ROCm] [Bugfix] Enable Load and Inference of GLM-5.3-Flash Quark MXFP4 Checkpoint

- Link: https://github.com/vllm-project/vllm/pull/56176
- Status/date: merged / 2026-09-16
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/glm5next/__init__.py`, `vllm/models/glm5next/common/__init__.py`, `vllm/models/glm5next/common/attention.py`, `vllm/models/glm5next/common/kda.py`, `vllm/models/glm5next/common/model.py` and 7 files; associated commits `c8d1cf077a78`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 8 files, +164/-4, 208 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/glm5next/common/model.py` renamed +14/-1 (15 lines); hunks: -1036,6 +1036,17 @@ class Glm5NextForConditionalGeneration(; -1259,7 +1270,9 @@ def _try_load_fp8_attn_proj(; symbols: Glm5NextForConditionalGeneration, _try_load_fp8_attn_proj, touching `Glm5NextForConditionalGeneration, _try_load_fp8_attn_proj`; `vllm/models/glm5next/__init__.py` modified +2/-2 (4 lines); hunks: -6,8 +6,8; `vllm/models/glm5next/common/__init__.py` added +2/-0 (2 lines); hunks: -0,0 +1,2; `vllm/models/glm5next/common/mtp.py` renamed +1/-1 (2 lines); hunks: -22,6 +22,7; -33,7 +34,6; symbols: Glm5NextMultiTokenPredictorLayer, touching `Glm5NextMultiTokenPredictorLayer`.
- Code diff details:
  - `vllm/models/glm5next/common/model.py` renamed +14/-1 (15 lines); hunks: -1036,6 +1036,17 @@ class Glm5NextForConditionalGeneration(; -1259,7 +1270,9 @@ def _try_load_fp8_attn_proj(; symbols: Glm5NextForConditionalGeneration, _try_load_fp8_attn_proj
  - `vllm/models/glm5next/__init__.py` modified +2/-2 (4 lines); hunks: -6,8 +6,8
  - `vllm/models/glm5next/common/__init__.py` added +2/-0 (2 lines); hunks: -0,0 +1,2
  - `vllm/models/glm5next/common/mtp.py` renamed +1/-1 (2 lines); hunks: -22,6 +22,7; -33,7 +34,6; symbols: Glm5NextMultiTokenPredictorLayer
  - `vllm/models/glm5next/common/attention.py` renamed +0/-0 (0 lines)
- Key code excerpts:

```diff
diff -- vllm/models/glm5next/common/model.py
@@ -1036,6 +1036,17 @@ class Glm5NextForConditionalGeneration(
+    # GLM-5.3-Flash stores the dense-MLP gate/up as separate tensors (like
+    # ``Glm4vMoeForConditionalGeneration``, ``glm4_moe`` and ``deepseek_v2``),
+    # so the fused ``gate_up_proj`` must expand to its real shard names for
+    # per-layer quant-scheme resolution. The identity ``gate_up_proj`` entry
+    # inherited from ``Glm4vForConditionalGeneration`` (pre-fused gate_up_proj)
+    # would otherwise route the module to ``global_quant_config`` and mismatch
diff -- vllm/models/glm5next/__init__.py
@@ -6,8 +6,8 @@
-from .nvidia.model import Glm5NextForCausalLM, Glm5NextForConditionalGeneration
-from .nvidia.mtp import Glm5NextMTP
+from .common.model import Glm5NextForCausalLM, Glm5NextForConditionalGeneration
+from .common.mtp import Glm5NextMTP
diff -- vllm/models/glm5next/common/__init__.py
@@ -0,0 +1,2 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/glm5next/common/model.py` renamed +14/-1; `vllm/models/glm5next/__init__.py` modified +2/-2; `vllm/models/glm5next/common/__init__.py` added +2/-0; `vllm/models/glm5next/common/mtp.py` renamed +1/-1; `vllm/models/glm5next/common/attention.py` renamed +0/-0; `vllm/models/glm5next/common/kda.py` renamed +0/-0
- Risk and verification: The diff ships test coverage in `tests/quantization/test_quark.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #55358 - [Refactor][GLM-5.3-Flash] Move sparse_attn_indexer_kpool into the model folder and split AMD/NVIDIA

- Link: https://github.com/vllm-project/vllm/pull/55358
- Status/date: merged / 2026-09-16
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/glm5next/amd/sparse_indexer.py`, `vllm/models/glm5next/common/attention.py`, `vllm/models/glm5next/common/sparse_indexer.py`, `vllm/models/glm5next/nvidia/sparse_indexer.py`, `vllm/models/glm5next/sparse_indexer.py`; associated commits `4fe9e6f6e564`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 6 files, +1008/-398, 1532 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/glm5next/amd/sparse_indexer.py` added +729/-0 (729 lines); hunks: -0,0 +1,729; symbols: _kpool_compress_insert, sparse_attn_indexer_kpool, SparseAttnIndexerKpool, __init__, touching `_kpool_compress_insert, sparse_attn_indexer_kpool, SparseAttnIndexerKpool`; `vllm/models/glm5next/nvidia/sparse_indexer.py` renamed +76/-391 (467 lines); hunks: -2,30 +2,29; -36,16 +35,9; symbols: _kpool_compress_insert, _build_decode_scatter_indices, _scatter_decode_tokens_by_request, _decode_topk_seq_lens, touching `_kpool_compress_insert, _build_decode_scatter_indices, _scatter_decode_tokens_by_request`; `vllm/models/glm5next/common/sparse_indexer.py` added +170/-0 (170 lines); hunks: -0,0 +1,170; symbols: _build_decode_scatter_indices, _scatter_decode_tokens_by_request, _decode_topk_seq_lens, _fill_causal_indices, touching `_build_decode_scatter_indices, _scatter_decode_tokens_by_request, _decode_topk_seq_lens`; `vllm/models/glm5next/sparse_indexer.py` added +22/-0 (22 lines); hunks: -0,0 +1,22.
- Code diff details:
  - `vllm/models/glm5next/amd/sparse_indexer.py` added +729/-0 (729 lines); hunks: -0,0 +1,729; symbols: _kpool_compress_insert, sparse_attn_indexer_kpool, SparseAttnIndexerKpool, __init__
  - `vllm/models/glm5next/nvidia/sparse_indexer.py` renamed +76/-391 (467 lines); hunks: -2,30 +2,29; -36,16 +35,9; symbols: _kpool_compress_insert, _build_decode_scatter_indices, _scatter_decode_tokens_by_request, _decode_topk_seq_lens
  - `vllm/models/glm5next/common/sparse_indexer.py` added +170/-0 (170 lines); hunks: -0,0 +1,170; symbols: _build_decode_scatter_indices, _scatter_decode_tokens_by_request, _decode_topk_seq_lens, _fill_causal_indices
  - `vllm/models/glm5next/sparse_indexer.py` added +22/-0 (22 lines); hunks: -0,0 +1,22
  - `vllm/models/glm5next/common/attention.py` modified +1/-1 (2 lines); hunks: -23,14 +23,14
- Key code excerpts:

```diff
diff -- vllm/models/glm5next/amd/sparse_indexer.py
@@ -0,0 +1,729 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Custom Sparse Attention Indexer layers."""
+import torch
+import vllm.envs as envs
+from vllm import _custom_ops  # noqa: F401  # registers the torch.ops._C kernels
diff -- vllm/models/glm5next/nvidia/sparse_indexer.py
@@ -2,30 +2,29 @@
-from typing import TYPE_CHECKING
-from vllm._aiter_ops import rocm_aiter_ops
+from vllm.models.glm5next.common.sparse_indexer import (
+    RADIX_TOPK_WORKSPACE_SIZE,
+    _build_decode_scatter_indices,
+    _decode_topk_seq_lens,
diff -- vllm/models/glm5next/common/sparse_indexer.py
@@ -0,0 +1,170 @@
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/glm5next/amd/sparse_indexer.py` added +729/-0; `vllm/models/glm5next/nvidia/sparse_indexer.py` renamed +76/-391; `vllm/models/glm5next/common/sparse_indexer.py` added +170/-0; `vllm/models/glm5next/sparse_indexer.py` added +22/-0; `vllm/models/glm5next/common/attention.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `tests/v1/attention/test_sparse_indexer_decode_seq_lens.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #57327 - [Perf][GLM5.3-Flash] Use cooperative top-k for small GLM decode batches

- Link: https://github.com/vllm-project/vllm/pull/57327
- Status/date: merged / 2026-09-18
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/glm5next/test_sparse_indexer_topk_dispatch.py`, `vllm/models/glm5next/nvidia/sparse_indexer.py`; associated commits `2bbdfcfcfea7`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +63/-1, 79 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/glm5next/test_sparse_indexer_topk_dispatch.py` added +41/-0 (41 lines); hunks: -0,0 +1,41; symbols: test_cooperative_topk_is_selected_for_supported_batch, test_cooperative_topk_falls_back_on_unsupported_batch, touching `test_cooperative_topk_is_selected_for_supported_batch, test_cooperative_topk_falls_back_on_unsupported_batch`; `vllm/models/glm5next/nvidia/sparse_indexer.py` modified +22/-1 (23 lines); hunks: -38,6 +38,22; -573,7 +589,12 @@ def sparse_attn_indexer_kpool(; symbols: _use_cooperative_topk, sparse_attn_indexer_kpool, touching `_use_cooperative_topk, sparse_attn_indexer_kpool`.
- Code diff details:
  - `tests/models/glm5next/test_sparse_indexer_topk_dispatch.py` added +41/-0 (41 lines); hunks: -0,0 +1,41; symbols: test_cooperative_topk_is_selected_for_supported_batch, test_cooperative_topk_falls_back_on_unsupported_batch
  - `vllm/models/glm5next/nvidia/sparse_indexer.py` modified +22/-1 (23 lines); hunks: -38,6 +38,22; -573,7 +589,12 @@ def sparse_attn_indexer_kpool(; symbols: _use_cooperative_topk, sparse_attn_indexer_kpool
- Key code excerpts:

```diff
diff -- tests/models/glm5next/test_sparse_indexer_topk_dispatch.py
@@ -0,0 +1,41 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Focused unit tests for the GLM decode top-k dispatch."""
+import pytest
+import torch
+from vllm.models.glm5next.nvidia.sparse_indexer import _use_cooperative_topk
diff -- vllm/models/glm5next/nvidia/sparse_indexer.py
@@ -38,6 +38,22 @@
+def _use_cooperative_topk(
+    logits: torch.Tensor,
+    select_k: int,
+    num_rows: int,
+) -> bool:
+    return (
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/glm5next/test_sparse_indexer_topk_dispatch.py` added +41/-0
  - runtime: `vllm/models/glm5next/nvidia/sparse_indexer.py` modified +22/-1
- Risk and verification: The diff ships test coverage in `tests/models/glm5next/test_sparse_indexer_topk_dispatch.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #57425 - [Bugfix][ROCm] Alias SparseAttnIndexerKpool.forward_cuda to forward_native (GLM-5.3-Flash boot crash)

- Link: https://github.com/vllm-project/vllm/pull/57425
- Status/date: merged / 2026-09-18
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/glm5next/amd/sparse_indexer.py`; associated commits `d12c2768530a`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +24/-1, 36 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/glm5next/amd/sparse_indexer.py` modified +24/-1 (25 lines); hunks: -665,7 +665,7 @@ def __init__(; -727,3 +727,26 @@ def forward_native(; symbols: __init__, forward_native, forward_hip, touching `__init__, forward_native, forward_hip`.
- Code diff details:
  - `vllm/models/glm5next/amd/sparse_indexer.py` modified +24/-1 (25 lines); hunks: -665,7 +665,7 @@ def __init__(; -727,3 +727,26 @@ def forward_native(; symbols: __init__, forward_native, forward_hip
- Key code excerpts:

```diff
diff -- vllm/models/glm5next/amd/sparse_indexer.py
@@ -665,7 +665,7 @@ def __init__(
-    def forward_native(
+    def forward_hip(
@@ -727,3 +727,26 @@ def forward_native(
+    def forward_native(
+        self,
+        hidden_states: torch.Tensor,
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/glm5next/amd/sparse_indexer.py` modified +24/-1
- Risk and verification: Runtime changes concentrate in `vllm/models/glm5next/amd/sparse_indexer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #57701 - [GLM5.3 Perf] Size the GLM-5 sparse indexer decode workspace, 3072 MiB GPU memory saved

- Link: https://github.com/vllm-project/vllm/pull/57701
- Status/date: merged / 2026-09-19
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/glm5next/amd/sparse_indexer.py`, `vllm/models/glm5next/common/attention.py`, `vllm/models/glm5next/nvidia/sparse_indexer.py`; associated commits `36fa72d2d0d2`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +17/-17, 153 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/glm5next/amd/sparse_indexer.py` modified +8/-8 (16 lines); hunks: -100,7 +100,7 @@ def sparse_attn_indexer_kpool(; -136,7 +136,7 @@ def sparse_attn_indexer_kpool(; symbols: sparse_attn_indexer_kpool, __init__, touching `sparse_attn_indexer_kpool, __init__`; `vllm/models/glm5next/nvidia/sparse_indexer.py` modified +7/-7 (14 lines); hunks: -117,7 +117,7 @@ def sparse_attn_indexer_kpool(; -153,7 +153,7 @@ def sparse_attn_indexer_kpool(; symbols: sparse_attn_indexer_kpool, __init__, touching `sparse_attn_indexer_kpool, __init__`; `vllm/models/glm5next/common/attention.py` modified +2/-2 (4 lines); hunks: -296,7 +296,7 @@ def __init__(; -307,7 +307,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/models/glm5next/amd/sparse_indexer.py` modified +8/-8 (16 lines); hunks: -100,7 +100,7 @@ def sparse_attn_indexer_kpool(; -136,7 +136,7 @@ def sparse_attn_indexer_kpool(; symbols: sparse_attn_indexer_kpool, __init__
  - `vllm/models/glm5next/nvidia/sparse_indexer.py` modified +7/-7 (14 lines); hunks: -117,7 +117,7 @@ def sparse_attn_indexer_kpool(; -153,7 +153,7 @@ def sparse_attn_indexer_kpool(; symbols: sparse_attn_indexer_kpool, __init__
  - `vllm/models/glm5next/common/attention.py` modified +2/-2 (4 lines); hunks: -296,7 +296,7 @@ def __init__(; -307,7 +307,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/models/glm5next/amd/sparse_indexer.py
@@ -100,7 +100,7 @@ def sparse_attn_indexer_kpool(
-    max_model_len: int,
+    max_pool_len: int,
@@ -136,7 +136,7 @@ def sparse_attn_indexer_kpool(
-        # whose shape is [B * next_n, max_model_len]. This profiling branch
+        # whose shape is [B * next_n, max_pool_len]. This profiling branch
@@ -152,7 +152,7 @@ def sparse_attn_indexer_kpool(
diff -- vllm/models/glm5next/nvidia/sparse_indexer.py
@@ -117,7 +117,7 @@ def sparse_attn_indexer_kpool(
-    max_model_len: int,
+    max_pool_len: int,
@@ -153,7 +153,7 @@ def sparse_attn_indexer_kpool(
-        # whose shape is [B * next_n, max_model_len]. This profiling branch
+        # whose shape is [B * next_n, max_pool_len]. This profiling branch
@@ -169,7 +169,7 @@ def sparse_attn_indexer_kpool(
diff -- vllm/models/glm5next/common/attention.py
@@ -296,7 +296,7 @@ def __init__(
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/glm5next/amd/sparse_indexer.py` modified +8/-8; `vllm/models/glm5next/nvidia/sparse_indexer.py` modified +7/-7; `vllm/models/glm5next/common/attention.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `vllm/models/glm5next/amd/sparse_indexer.py`, `vllm/models/glm5next/common/attention.py`, `vllm/models/glm5next/nvidia/sparse_indexer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #57477 - [Bugfix][GLM-5.3-Flash] Address kpool tail blocks by the padded indexer stride in the NVIDIA prefill seed kernel

- Link: https://github.com/vllm-project/vllm/pull/57477
- Status/date: merged / 2026-09-20
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/glm5next/nvidia/ops/kpool_compress.py`; associated commits `db1bfdd4fb0d`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +23/-5, 73 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/glm5next/nvidia/ops/kpool_compress.py` modified +13/-2 (15 lines); hunks: -383,6 +383,8 @@ def _kpool_tail_seed_kernel(; -393,6 +395,11 @@ def _kpool_tail_seed_kernel(; symbols: _kpool_tail_seed_kernel, kpool_seed_tail_cache, touching `_kpool_tail_seed_kernel, kpool_seed_tail_cache`.
- Code diff details:
  - `vllm/models/glm5next/nvidia/ops/kpool_compress.py` modified +13/-2 (15 lines); hunks: -383,6 +383,8 @@ def _kpool_tail_seed_kernel(; -393,6 +395,11 @@ def _kpool_tail_seed_kernel(; symbols: _kpool_tail_seed_kernel, kpool_seed_tail_cache
- Key code excerpts:

```diff
diff -- vllm/models/glm5next/nvidia/ops/kpool_compress.py
@@ -383,6 +383,8 @@ def _kpool_tail_seed_kernel(
+    TAIL_BLOCK_ELEMS: tl.constexpr,
+    KPOOL_HEAD: tl.constexpr,
@@ -393,6 +395,11 @@ def _kpool_tail_seed_kernel(
+    The tail cache aliases the indexer cache with the indexer's (padded) block
+    stride, so blocks are addressed through ``TAIL_BLOCK_ELEMS`` /
+    ``KPOOL_HEAD`` (``tail.stride(0)`` / ``tail.stride(1)``), never as a dense
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/glm5next/nvidia/ops/kpool_compress.py` modified +13/-2
- Risk and verification: The diff ships test coverage in `tests/kernels/test_kpool_decode_update_batched.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #57546 - [GLM-5.3-Flash] Route kpool indexer top-k through the shared SparseIndexerTopk dispatcher

- Link: https://github.com/vllm-project/vllm/pull/57546
- Status/date: merged / 2026-09-20
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/glm5next/test_sparse_indexer_topk_dispatch.py`, `vllm/models/glm5next/nvidia/sparse_indexer.py`; associated commits `bf01fc4a313c`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 4 files, +94/-83, 277 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/glm5next/nvidia/sparse_indexer.py` modified +20/-44 (64 lines); hunks: -10,6 +10,7; -39,21 +40,6; symbols: _use_cooperative_topk, sparse_attn_indexer_kpool, __init__, forward_cuda, touching `_use_cooperative_topk, sparse_attn_indexer_kpool, __init__`; `tests/models/glm5next/test_sparse_indexer_topk_dispatch.py` modified +28/-26 (54 lines); hunks: -1,41 +1,43; symbols: test_cooperative_topk_is_selected_for_supported_batch, _require_deep_gemm, test_cooperative_topk_falls_back_on_unsupported_batch, test_kpool_indexer_dispatches_through_shared_topk_backend, touching `test_cooperative_topk_is_selected_for_supported_batch, _require_deep_gemm, test_cooperative_topk_falls_back_on_unsupported_batch`.
- Code diff details:
  - `vllm/models/glm5next/nvidia/sparse_indexer.py` modified +20/-44 (64 lines); hunks: -10,6 +10,7; -39,21 +40,6; symbols: _use_cooperative_topk, sparse_attn_indexer_kpool, __init__, forward_cuda
  - `tests/models/glm5next/test_sparse_indexer_topk_dispatch.py` modified +28/-26 (54 lines); hunks: -1,41 +1,43; symbols: test_cooperative_topk_is_selected_for_supported_batch, _require_deep_gemm, test_cooperative_topk_falls_back_on_unsupported_batch, test_kpool_indexer_dispatches_through_shared_topk_backend
- Key code excerpts:

```diff
diff -- vllm/models/glm5next/nvidia/sparse_indexer.py
@@ -10,6 +10,7 @@
+from vllm.model_executor.layers.indexer_topk import get_indexer_topk
@@ -39,21 +40,6 @@
-def _use_cooperative_topk(
-    logits: torch.Tensor,
-    select_k: int,
-    num_rows: int,
diff -- tests/models/glm5next/test_sparse_indexer_topk_dispatch.py
@@ -1,41 +1,43 @@
-"""Focused unit tests for the GLM decode top-k dispatch."""
+"""Tests for the GLM kpool indexer's top-k backend wiring."""
-from vllm.models.glm5next.nvidia.sparse_indexer import _use_cooperative_topk
+from vllm.config import VllmConfig, set_current_vllm_config
-@pytest.mark.parametrize("num_rows", [1, 64], ids=["1row", "64rows"])
-@torch.inference_mode()
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/glm5next/nvidia/sparse_indexer.py` modified +20/-44
  - tests: `tests/models/glm5next/test_sparse_indexer_topk_dispatch.py` modified +28/-26
- Risk and verification: The diff ships test coverage in `tests/kernels/test_top_k_per_row.py`, `tests/models/glm5next/test_sparse_indexer_topk_dispatch.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #58061 - [Bugfix][GLM-5.3-Flash] Run the dense MLP layers on the sequence-parallel shard

- Link: https://github.com/vllm-project/vllm/pull/58061
- Status/date: merged / 2026-09-22
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/glm5next/test_sequence_parallel.py`, `vllm/models/glm5next/common/model.py`; associated commits `c9b34fdb2d0c`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +115/-0, 123 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/glm5next/test_sequence_parallel.py` added +114/-0 (114 lines); hunks: -0,0 +1,114; symbols: _Attention, __init__, _fake_tensor_parallel_world, _forbidden_all_reduce, touching `_Attention, __init__, _fake_tensor_parallel_world`; `vllm/models/glm5next/common/model.py` modified +1/-0 (1 lines); hunks: -355,6 +355,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `tests/models/glm5next/test_sequence_parallel.py` added +114/-0 (114 lines); hunks: -0,0 +1,114; symbols: _Attention, __init__, _fake_tensor_parallel_world, _forbidden_all_reduce
  - `vllm/models/glm5next/common/model.py` modified +1/-0 (1 lines); hunks: -355,6 +355,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- tests/models/glm5next/test_sequence_parallel.py
@@ -0,0 +1,114 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""GLM-5.3-Flash sequence-parallel MoE layout.
+With DP > 1, TP > 1 and expert parallelism, ``Glm5NextModel`` shards the token
+dimension across the TP group once at the model entry and every layer runs its
+MLP on that shard. A module that still does tensor-parallel collectives there
diff -- vllm/models/glm5next/common/model.py
@@ -355,6 +355,7 @@ def __init__(
+                is_sequence_parallel=self.is_sequence_parallel,
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/glm5next/test_sequence_parallel.py` added +114/-0
  - runtime: `vllm/models/glm5next/common/model.py` modified +1/-0
- Risk and verification: The diff ships test coverage in `tests/models/glm5next/test_sequence_parallel.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #55270 - [Bugfix] GLM-5.3-Flash: launch the kpool paged MQA logits in the varlen mode its schedule was built with

- Link: https://github.com/vllm-project/vllm/pull/55270
- Status/date: merged / 2026-09-25
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/glm5next/nvidia/sparse_indexer.py`; associated commits `39724758e7b7`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +1/-0, 8 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/glm5next/nvidia/sparse_indexer.py` modified +1/-0 (1 lines); hunks: -558,6 +558,7 @@ def sparse_attn_indexer_kpool(; symbols: sparse_attn_indexer_kpool, touching `sparse_attn_indexer_kpool`.
- Code diff details:
  - `vllm/models/glm5next/nvidia/sparse_indexer.py` modified +1/-0 (1 lines); hunks: -558,6 +558,7 @@ def sparse_attn_indexer_kpool(; symbols: sparse_attn_indexer_kpool
- Key code excerpts:

```diff
diff -- vllm/models/glm5next/nvidia/sparse_indexer.py
@@ -558,6 +558,7 @@ def sparse_attn_indexer_kpool(
+            indices=decode_metadata.indices,
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/glm5next/nvidia/sparse_indexer.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `vllm/models/glm5next/nvidia/sparse_indexer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #58454 - [Bugfix][GLM-5.3-Flash] kpool corruption with speculative decoding

- Link: https://github.com/vllm-project/vllm/pull/58454
- Status/date: merged / 2026-09-25
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/glm5next/amd/ops/kpool_compress.py`, `vllm/models/glm5next/common/attention.py`, `vllm/models/glm5next/nvidia/ops/kpool_compress.py`; associated commits `2617fe938355`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 5 files, +208/-49, 534 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/glm5next/amd/ops/kpool_compress.py` modified +18/-16 (34 lines); hunks: -327,37 +327,38 @@ def _kpool_tail_seed_kernel(; -385,6 +386,7 @@ def kpool_seed_tail_cache(; symbols: _kpool_tail_seed_kernel, kpool_seed_tail_cache, _kpool_decode_update_batched_kernel, touching `_kpool_tail_seed_kernel, kpool_seed_tail_cache, _kpool_decode_update_batched_kernel`; `vllm/models/glm5next/nvidia/ops/kpool_compress.py` modified +17/-15 (32 lines); hunks: -387,14 +387,15 @@ def _kpool_tail_seed_kernel(; -405,18 +406,18 @@ def _kpool_tail_seed_kernel(; symbols: _kpool_tail_seed_kernel, kpool_seed_tail_cache, _kpool_decode_update_batched_kernel, touching `_kpool_tail_seed_kernel, kpool_seed_tail_cache, _kpool_decode_update_batched_kernel`; `vllm/models/glm5next/common/attention.py` modified +15/-3 (18 lines); hunks: -34,6 +34,7; -160,7 +161,7 @@ class Glm5NextTailCache(DeepseekV32IndexerCache):; symbols: Glm5NextTailCache, __init__, get_kv_cache_spec, get_attn_backend, touching `Glm5NextTailCache, __init__, get_kv_cache_spec`.
- Code diff details:
  - `vllm/models/glm5next/amd/ops/kpool_compress.py` modified +18/-16 (34 lines); hunks: -327,37 +327,38 @@ def _kpool_tail_seed_kernel(; -385,6 +386,7 @@ def kpool_seed_tail_cache(; symbols: _kpool_tail_seed_kernel, kpool_seed_tail_cache, _kpool_decode_update_batched_kernel
  - `vllm/models/glm5next/nvidia/ops/kpool_compress.py` modified +17/-15 (32 lines); hunks: -387,14 +387,15 @@ def _kpool_tail_seed_kernel(; -405,18 +406,18 @@ def _kpool_tail_seed_kernel(; symbols: _kpool_tail_seed_kernel, kpool_seed_tail_cache, _kpool_decode_update_batched_kernel
  - `vllm/models/glm5next/common/attention.py` modified +15/-3 (18 lines); hunks: -34,6 +34,7; -160,7 +161,7 @@ class Glm5NextTailCache(DeepseekV32IndexerCache):; symbols: Glm5NextTailCache, __init__, get_kv_cache_spec, get_attn_backend
- Key code excerpts:

```diff
diff -- vllm/models/glm5next/amd/ops/kpool_compress.py
@@ -327,37 +327,38 @@ def _kpool_tail_seed_kernel(
+    RING: tl.constexpr,
-    """Copy token ``i``'s raw K + gate into its request's tail block.
+    """Copy token ``i``'s raw K + gate into its request's tail ring.
-    slot < 0). ``tslot = block * KPOOL + pos % KPOOL``; the destination is
-    ``tail[block, {0:K, 1:score}, pos % KPOOL, :]``.
+    slot < 0). ``tslot = block * RING + pos % RING``; the destination is
diff -- vllm/models/glm5next/nvidia/ops/kpool_compress.py
@@ -387,14 +387,15 @@ def _kpool_tail_seed_kernel(
+    RING: tl.constexpr,
-    """Copy token ``i``'s raw K + gate into its request's tail block.
+    """Copy token ``i``'s raw K + gate into its request's tail ring.
-    slot < 0). ``tslot = block * KPOOL + pos % KPOOL``; the destination is
-    ``tail[block, {0:K, 1:score}, pos % KPOOL, :]``.
+    slot < 0). ``tslot = block * RING + pos % RING``; the destination is
diff -- vllm/models/glm5next/common/attention.py
@@ -34,6 +34,7 @@
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/glm5next/amd/ops/kpool_compress.py` modified +18/-16; `vllm/models/glm5next/nvidia/ops/kpool_compress.py` modified +17/-15; `vllm/models/glm5next/common/attention.py` modified +15/-3
- Risk and verification: The diff ships test coverage in `tests/kernels/test_kpool_decode_update_batched.py`, `tests/v1/attention/test_kpool_tail_slot_mapping.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #55647 - [Bugfix][GLM-5.3-Flash] Take video placeholder timestamps from the pixel path's frame sampler

- Link: https://github.com/vllm-project/vllm/pull/55647
- Status/date: merged / 2026-09-29
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/multimodal/processing/test_glm5next.py`, `vllm/models/glm5next/common/multimodal.py`; associated commits `65ed0a2642d2`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +229/-0, 251 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/multimodal/processing/test_glm5next.py` added +170/-0 (170 lines); hunks: -0,0 +1,170; symbols: processor, _pixel_path_grid, test_video_placeholders_match_encoder_rows, test_video_placeholders_match_encoder_rows_when_presampled, touching `processor, _pixel_path_grid, test_video_placeholders_match_encoder_rows`; `vllm/models/glm5next/common/multimodal.py` modified +59/-0 (59 lines); hunks: -4,6 +4,7; -42,6 +43,7; symbols: _get_video_max_pixels, _get_video_second_idx_glm46v, _get_vision_info, touching `_get_video_max_pixels, _get_video_second_idx_glm46v, _get_vision_info`.
- Code diff details:
  - `tests/models/multimodal/processing/test_glm5next.py` added +170/-0 (170 lines); hunks: -0,0 +1,170; symbols: processor, _pixel_path_grid, test_video_placeholders_match_encoder_rows, test_video_placeholders_match_encoder_rows_when_presampled
  - `vllm/models/glm5next/common/multimodal.py` modified +59/-0 (59 lines); hunks: -4,6 +4,7; -42,6 +43,7; symbols: _get_video_max_pixels, _get_video_second_idx_glm46v, _get_vision_info
- Key code excerpts:

```diff
diff -- tests/models/multimodal/processing/test_glm5next.py
@@ -0,0 +1,170 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Video placeholder accounting for GLM-5.3-Flash.
+``Glm4vProcessingInfo._construct_video_placeholder`` emits one frame of
+placeholders per timestamp, so the timestamps returned by
+``_get_video_second_idx_glm46v`` decide how many placeholders the prompt gets
diff -- vllm/models/glm5next/common/multimodal.py
@@ -4,6 +4,7 @@
+from typing import Any
@@ -42,6 +43,7 @@
+from vllm.transformers_utils.processors.glm5next import glm_sample_frame_indices
@@ -656,6 +658,63 @@ def _get_video_max_pixels(self) -> int:
+    def _get_video_second_idx_glm46v(
+        self, metadata: dict[str, Any], total_frames: int
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/multimodal/processing/test_glm5next.py` added +170/-0
  - runtime: `vllm/models/glm5next/common/multimodal.py` modified +59/-0
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/processing/test_glm5next.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #56960 - [Feat][Model] Enable KDA prefill checkpoints for GLM-5.3-Flash

- Link: https://github.com/vllm-project/vllm/pull/56960
- Status/date: merged / 2026-09-29
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/glm5next/test_kda_recurrent.py`, `vllm/models/glm5next/common/kda.py`; associated commits `3e2a7e74a556`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 6 files, +307/-11, 478 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/glm5next/test_kda_recurrent.py` modified +167/-0 (167 lines); hunks: -193,3 +193,170 @@ def test_fused_recurrent_kda_rejects_unaddressable_layouts():; symbols: test_fused_recurrent_kda_rejects_unaddressable_layouts, test_prefill_checkpoint_resumes_suffix, run, touching `test_fused_recurrent_kda_rejects_unaddressable_layouts, test_prefill_checkpoint_resumes_suffix, run`; `vllm/models/glm5next/common/kda.py` modified +59/-3 (62 lines); hunks: -2,6 +2,8; -14,7 +16,14; symbols: Glm5NextLinearAttention, get_kv_cache_spec, get_state_dtype, _a_log_weight_loader, touching `Glm5NextLinearAttention, get_kv_cache_spec, get_state_dtype`.
- Code diff details:
  - `tests/models/glm5next/test_kda_recurrent.py` modified +167/-0 (167 lines); hunks: -193,3 +193,170 @@ def test_fused_recurrent_kda_rejects_unaddressable_layouts():; symbols: test_fused_recurrent_kda_rejects_unaddressable_layouts, test_prefill_checkpoint_resumes_suffix, run
  - `vllm/models/glm5next/common/kda.py` modified +59/-3 (62 lines); hunks: -2,6 +2,8; -14,7 +16,14; symbols: Glm5NextLinearAttention, get_kv_cache_spec, get_state_dtype, _a_log_weight_loader
- Key code excerpts:

```diff
diff -- tests/models/glm5next/test_kda_recurrent.py
@@ -193,3 +193,170 @@ def test_fused_recurrent_kda_rejects_unaddressable_layouts():
+@pytest.mark.parametrize("offset", [16, 144, 256])
+@pytest.mark.parametrize("dim_first", [False, True])
+@pytest.mark.parametrize("num_spec", [0, 3])
+@torch.inference_mode()
+def test_prefill_checkpoint_resumes_suffix(monkeypatch, dim_first, num_spec, offset):
+    """Restoring both cached states must reproduce an uninterrupted prefill."""
diff -- vllm/models/glm5next/common/kda.py
@@ -2,6 +2,8 @@
+from dataclasses import replace
@@ -14,7 +16,14 @@
+from vllm.model_executor.layers.mamba.checkpoint import (
+    MambaPrefillCheckpointMetadata,
+)
+from vllm.model_executor.layers.mamba.kda_checkpoint import (
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/glm5next/test_kda_recurrent.py` modified +167/-0
  - runtime: `vllm/models/glm5next/common/kda.py` modified +59/-3
- Risk and verification: The diff ships test coverage in `tests/models/glm5next/test_kda_recurrent.py`, `tests/v1/attention/test_gdn_metadata_builder.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #55222 - [Bugfix] GLM-5.3-Flash: fp8 plan dtype on SM90 sparse MLA, and right-size the indexer prefill workspace

- Link: https://github.com/vllm-project/vllm/pull/55222
- Status/date: merged / 2026-09-29
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/glm5next/common/attention.py`; associated commits `c37f86e5721f`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 5 files, +84/-19, 206 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/glm5next/common/attention.py` modified +3/-1 (4 lines); hunks: -312,7 +312,9 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/models/glm5next/common/attention.py` modified +3/-1 (4 lines); hunks: -312,7 +312,9 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/models/glm5next/common/attention.py
@@ -312,7 +312,9 @@ def __init__(
-        self.max_total_seq_len = get_max_prefill_buffer_size(vllm_config)
+        self.max_total_seq_len = (
+            get_max_prefill_buffer_size(vllm_config) // self.index_kpool
+        )
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/glm5next/common/attention.py` modified +3/-1
- Risk and verification: The diff ships test coverage in `tests/v1/attention/test_flashinfer_mla_sparse_sm90.py`, `tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #55389 - [Model] Extend device-side mm normalization to GLM4V/GLM5Next

- Link: https://github.com/vllm-project/vllm/pull/55389
- Status/date: merged / 2026-09-30
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/multimodal/processing/test_glm5next.py`, `vllm/models/glm5next/common/model.py`, `vllm/models/glm5next/common/multimodal.py`; associated commits `0a30bc3f9ac3`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 7 files, +114/-10, 279 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/multimodal/processing/test_glm5next.py` modified +37/-0 (37 lines); hunks: -13,8 +13,12; -168,3 +172,36 @@ def test_video_shorter_than_one_sampling_interval_is_reject...; symbols: test_video_shorter_than_one_sampling_interval_is_rejected, test_mm_device_do_normalize, touching `test_video_shorter_than_one_sampling_interval_is_rejected, test_mm_device_do_normalize`; `vllm/models/glm5next/common/multimodal.py` modified +4/-1 (5 lines); hunks: -19,6 +19,7; -353,6 +354,7 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`; `vllm/models/glm5next/common/model.py` modified +2/-0 (2 lines); hunks: -22,6 +22,7; -1103,6 +1104,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, touching `__init__`.
- Code diff details:
  - `tests/models/multimodal/processing/test_glm5next.py` modified +37/-0 (37 lines); hunks: -13,8 +13,12; -168,3 +172,36 @@ def test_video_shorter_than_one_sampling_interval_is_reject...; symbols: test_video_shorter_than_one_sampling_interval_is_rejected, test_mm_device_do_normalize
  - `vllm/models/glm5next/common/multimodal.py` modified +4/-1 (5 lines); hunks: -19,6 +19,7; -353,6 +354,7 @@ def __init__(; symbols: __init__, forward
  - `vllm/models/glm5next/common/model.py` modified +2/-0 (2 lines); hunks: -22,6 +22,7; -1103,6 +1104,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__
- Key code excerpts:

```diff
diff -- tests/models/multimodal/processing/test_glm5next.py
@@ -13,8 +13,12 @@
+import torch
+from PIL import Image
+from vllm.model_executor.layers.fusion.mm_input_norm import build_mm_input_norm
+from vllm.platforms import current_platform
@@ -168,3 +172,36 @@ def test_video_shorter_than_one_sampling_interval_is_rejected(processor):
+@pytest.mark.usefixtures("default_vllm_config")
diff -- vllm/models/glm5next/common/multimodal.py
@@ -19,6 +19,7 @@
+from vllm.model_executor.layers.fusion.mm_input_norm import IdentityInputNorm
@@ -353,6 +354,7 @@ def __init__(
+        input_norm: nn.Module | None = None,
@@ -386,6 +388,7 @@ def __init__(
+        self.input_norm = input_norm if input_norm is not None else IdentityInputNorm()
@@ -566,7 +569,7 @@ def forward(
diff -- vllm/models/glm5next/common/model.py
@@ -22,6 +22,7 @@
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/multimodal/processing/test_glm5next.py` modified +37/-0
  - runtime: `vllm/models/glm5next/common/multimodal.py` modified +4/-1; `vllm/models/glm5next/common/model.py` modified +2/-0
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/generation_ppl_test/test_glm.py`, `tests/models/multimodal/processing/test_glm5next.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #57979 - [ROCm][Perf][GLM-5.3-Flash] Stride-aware decode KDA

- Link: https://github.com/vllm-project/vllm/pull/57979
- Status/date: merged / 2026-09-30
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/glm5next/test_kda_recurrent.py`, `vllm/models/glm5next/amd/ops/third_party/kda/fused_recurrent.py`, `vllm/models/glm5next/amd/ops/third_party/kda/kernels.py`; associated commits `3627a6a12489`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +54/-19, 159 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/glm5next/amd/ops/third_party/kda/fused_recurrent.py` modified +35/-9 (44 lines); hunks: -15,6 +15,23; -52,6 +69,11 @@ def fused_recurrent_gated_delta_rule_fwd_kernel(; symbols: token_stride, fused_recurrent_gated_delta_rule_fwd_kernel, fused_recurrent_gated_delta_rule_fwd, touching `token_stride, fused_recurrent_gated_delta_rule_fwd_kernel, fused_recurrent_gated_delta_rule_fwd`; `vllm/models/glm5next/amd/ops/third_party/kda/kernels.py` modified +16/-6 (22 lines); hunks: -24,7 +24,10; -70,7 +73,7 @@ def fused_recurrent_kda_fwd(; symbols: fused_recurrent_kda_fwd, fused_recurrent_kda, touching `fused_recurrent_kda_fwd, fused_recurrent_kda`; `tests/models/glm5next/test_kda_recurrent.py` modified +3/-4 (7 lines); hunks: -5,8 +5,7; -166,8 +165,8 @@ def test_fused_recurrent_kda_strided_inputs_bit_identical_to...; symbols: test_fused_recurrent_kda_strided_inputs_bit_identical_to_contiguous, test_fused_recurrent_kda_rejects_unaddressable_layouts, touching `test_fused_recurrent_kda_strided_inputs_bit_identical_to_contiguous, test_fused_recurrent_kda_rejects_unaddressable_layouts`.
- Code diff details:
  - `vllm/models/glm5next/amd/ops/third_party/kda/fused_recurrent.py` modified +35/-9 (44 lines); hunks: -15,6 +15,23; -52,6 +69,11 @@ def fused_recurrent_gated_delta_rule_fwd_kernel(; symbols: token_stride, fused_recurrent_gated_delta_rule_fwd_kernel, fused_recurrent_gated_delta_rule_fwd
  - `vllm/models/glm5next/amd/ops/third_party/kda/kernels.py` modified +16/-6 (22 lines); hunks: -24,7 +24,10; -70,7 +73,7 @@ def fused_recurrent_kda_fwd(; symbols: fused_recurrent_kda_fwd, fused_recurrent_kda
  - `tests/models/glm5next/test_kda_recurrent.py` modified +3/-4 (7 lines); hunks: -5,8 +5,7; -166,8 +165,8 @@ def test_fused_recurrent_kda_strided_inputs_bit_identical_to...; symbols: test_fused_recurrent_kda_strided_inputs_bit_identical_to_contiguous, test_fused_recurrent_kda_rejects_unaddressable_layouts
- Key code excerpts:

```diff
diff -- vllm/models/glm5next/amd/ops/third_party/kda/fused_recurrent.py
@@ -15,6 +15,23 @@
+def token_stride(x: torch.Tensor) -> int:
+    """Token stride (elements) of a ``[B, T, H, D]`` or ``[B, T, H]`` tensor.
+    The recurrent kernel walks tokens with this stride and addresses heads
+    densely inside a token, so each token's ``[H, D]`` (or ``[H]``) block must
+    be contiguous, tokens must not overlap, and with ``B > 1`` sequence ``n``
+    must start at token ``n * T`` (dense batch). Column slices of a wider
diff -- vllm/models/glm5next/amd/ops/third_party/kda/kernels.py
@@ -24,7 +24,10 @@
-from .fused_recurrent import fused_recurrent_gated_delta_rule_fwd_kernel
+from .fused_recurrent import (
+    fused_recurrent_gated_delta_rule_fwd_kernel,
+    token_stride,
+)
@@ -70,7 +73,7 @@ def fused_recurrent_kda_fwd(
diff -- tests/models/glm5next/test_kda_recurrent.py
@@ -5,8 +5,7 @@
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/glm5next/amd/ops/third_party/kda/fused_recurrent.py` modified +35/-9; `vllm/models/glm5next/amd/ops/third_party/kda/kernels.py` modified +16/-6
  - tests: `tests/models/glm5next/test_kda_recurrent.py` modified +3/-4
- Risk and verification: The diff ships test coverage in `tests/models/glm5next/test_kda_recurrent.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #58008 - [ROCm][Perf][GLM-5.3-Flash] "Fit kpool top-k indices to AITER" with a single Triton kernel

- Link: https://github.com/vllm-project/vllm/pull/58008
- Status/date: merged / 2026-09-30
- Trace source: `git log --name-only -- <model-files>` found it through `tests/v1/attention/test_rocm_glm5next_sparse.py`; associated commits `2df122e65e87`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +109/-12, 165 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/v1/attention/test_rocm_glm5next_sparse.py` modified +44/-0 (44 lines); hunks: -8,6 +8,7; -26,6 +27,26 @@ def _store_sparse_kv_row_offset_kernel(slot_ptr, output_ptr,...; symbols: _store_sparse_kv_row_offset_kernel, _fit_kpool_indices_reference, test_fit_kpool_indices_preserves_tail_and_best_history, touching `_store_sparse_kv_row_offset_kernel, _fit_kpool_indices_reference, test_fit_kpool_indices_preserves_tail_and_best_history`; `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py` modified +65/-12 (77 lines); hunks: -70,6 +70,50 @@ def _use_rocm_sparse_triton(; -79,20 +123,29 @@ def fit_kpool_indices_to_aiter(; symbols: _use_rocm_sparse_triton, _fit_kpool_indices_kernel, fit_kpool_indices_to_aiter, touching `_use_rocm_sparse_triton, _fit_kpool_indices_kernel, fit_kpool_indices_to_aiter`.
- Code diff details:
  - `tests/v1/attention/test_rocm_glm5next_sparse.py` modified +44/-0 (44 lines); hunks: -8,6 +8,7; -26,6 +27,26 @@ def _store_sparse_kv_row_offset_kernel(slot_ptr, output_ptr,...; symbols: _store_sparse_kv_row_offset_kernel, _fit_kpool_indices_reference, test_fit_kpool_indices_preserves_tail_and_best_history
  - `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py` modified +65/-12 (77 lines); hunks: -70,6 +70,50 @@ def _use_rocm_sparse_triton(; -79,20 +123,29 @@ def fit_kpool_indices_to_aiter(; symbols: _use_rocm_sparse_triton, _fit_kpool_indices_kernel, fit_kpool_indices_to_aiter
- Key code excerpts:

```diff
diff -- tests/v1/attention/test_rocm_glm5next_sparse.py
@@ -8,6 +8,7 @@
+from vllm.utils.torch_utils import set_random_seed
@@ -26,6 +27,26 @@ def _store_sparse_kv_row_offset_kernel(slot_ptr, output_ptr, stride: tl.constexp
+def _fit_kpool_indices_reference(
+    token_indices: torch.Tensor, topk_tokens: int
+) -> torch.Tensor:
+    history = token_indices[:, :topk_tokens]
diff -- vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py
@@ -70,6 +70,50 @@ def _use_rocm_sparse_triton(
+@triton.jit
+def _fit_kpool_indices_kernel(
+    token_indices_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS + TAIL_WIDTH]
+    out_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS]
+    ti_stride0,
+    ti_stride1,
```

- Extracted files (not manually reviewed):
  - tests: `tests/v1/attention/test_rocm_glm5next_sparse.py` modified +44/-0
  - runtime: `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py` modified +65/-12
- Risk and verification: The diff ships test coverage in `tests/v1/attention/test_rocm_glm5next_sparse.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #57387 - [Model] Use upstream GLM-5.3 and Qwen4-Exp configs and processor

- Link: https://github.com/vllm-project/vllm/pull/57387
- Status/date: merged / 2026-10-02
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/glm5next/test_sequence_parallel.py`, `tests/models/multimodal/processing/test_glm5next.py`, `vllm/models/glm5next/common/attention.py`, `vllm/models/glm5next/common/kda.py`, `vllm/models/glm5next/common/model.py` and 7 files; associated commits `e3cae8d2ac6b`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 36 files, +365/-2384, 3571 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/transformers_utils/processors/test_glm5next.py` removed +0/-389 (389 lines); hunks: -1,389 +0,0; symbols: resize, test_smart_resize_reference_values, test_smart_resize_stays_snapped_and_positive, test_smart_resize_rejects_degenerate_inputs, touching `resize, test_smart_resize_reference_values, test_smart_resize_stays_snapped_and_positive`; `vllm/models/glm5next/common/multimodal.py` modified +68/-52 (120 lines); hunks: -3,13 +3,14; -44,7 +45,6; symbols: load_weights, Glm5NextProcessingInfo, _glm5_hf_processor, get_hf_processor, touching `load_weights, Glm5NextProcessingInfo, _glm5_hf_processor`; `vllm/models/glm5next/common/model.py` modified +74/-26 (100 lines); hunks: -6,6 +6,7; -83,7 +84,6; symbols: _is_moe, _is_kda_layer, _is_linear_attn, _validate_supported_config, touching `_is_moe, _is_kda_layer, _is_linear_attn`; `tests/models/multimodal/processing/test_glm5next.py` modified +16/-33 (49 lines); hunks: -15,16 +15,15; -48,27 +47,16 @@ def _pixel_path_grid(; symbols: _pixel_path_grid, test_video_placeholders_match_encoder_rows, touching `_pixel_path_grid, test_video_placeholders_match_encoder_rows`.
- Code diff details:
  - `tests/transformers_utils/processors/test_glm5next.py` removed +0/-389 (389 lines); hunks: -1,389 +0,0; symbols: resize, test_smart_resize_reference_values, test_smart_resize_stays_snapped_and_positive, test_smart_resize_rejects_degenerate_inputs
  - `vllm/models/glm5next/common/multimodal.py` modified +68/-52 (120 lines); hunks: -3,13 +3,14; -44,7 +45,6; symbols: load_weights, Glm5NextProcessingInfo, _glm5_hf_processor, get_hf_processor
  - `vllm/models/glm5next/common/model.py` modified +74/-26 (100 lines); hunks: -6,6 +6,7; -83,7 +84,6; symbols: _is_moe, _is_kda_layer, _is_linear_attn, _validate_supported_config
  - `tests/models/multimodal/processing/test_glm5next.py` modified +16/-33 (49 lines); hunks: -15,16 +15,15; -48,27 +47,16 @@ def _pixel_path_grid(; symbols: _pixel_path_grid, test_video_placeholders_match_encoder_rows
  - `vllm/models/qwen4_exp/amd/model.py` modified +10/-37 (47 lines); hunks: -7,6 +7,7; -47,7 +48,6; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- tests/transformers_utils/processors/test_glm5next.py
@@ -1,389 +0,0 @@
-# SPDX-License-Identifier: Apache-2.0
-# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-"""Unit tests for the vLLM-native GLM-5.3-Flash multimodal processor."""
-import math
-import pytest
-import torch
diff -- vllm/models/glm5next/common/multimodal.py
@@ -3,13 +3,14 @@
-from functools import cached_property, partial
+from functools import partial
+from transformers.video_utils import VideoMetadata
@@ -44,7 +45,6 @@
-from vllm.transformers_utils.processors.glm5next import glm_sample_frame_indices
@@ -616,37 +616,48 @@ def load_weights(self, weights) -> set[str]:
diff -- vllm/models/glm5next/common/model.py
@@ -6,6 +6,7 @@
```

- Extracted files (not manually reviewed):
  - tests: `tests/transformers_utils/processors/test_glm5next.py` removed +0/-389; `tests/models/multimodal/processing/test_glm5next.py` modified +16/-33
  - runtime: `vllm/models/glm5next/common/multimodal.py` modified +68/-52; `vllm/models/glm5next/common/model.py` modified +74/-26; `vllm/models/qwen4_exp/amd/model.py` modified +10/-37; `vllm/models/qwen4_exp/nvidia/model.py` modified +10/-37; `vllm/models/glm5next/common/attention.py` modified +18/-14; `vllm/models/glm5next/common/mtp.py` modified +4/-4
- Risk and verification: The diff ships test coverage in `tests/models/glm5next/test_sequence_parallel.py`, `tests/models/multimodal/processing/test_glm5next.py`, `tests/models/qwen4_exp/test_config.py`, `tests/models/qwen4_exp/test_ple.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #59565 - [Bugfix][GLM-5.3] Size the image encoder cache from the exact token ceiling

- Link: https://github.com/vllm-project/vllm/pull/59565
- Status/date: merged / 2026-10-02
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/multimodal/processing/test_glm5next.py`, `vllm/models/glm5next/common/multimodal.py`; associated commits `5688a4dd4a1e`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +75/-0, 93 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/multimodal/processing/test_glm5next.py` modified +51/-0 (51 lines); hunks: -188,3 +188,54 @@ def test_mm_device_do_normalize():; symbols: test_mm_device_do_normalize, _image_info, test_image_encoder_cache_covers_full_token_budget, test_image_encoder_cache_follows_max_pixels_override, touching `test_mm_device_do_normalize, _image_info, test_image_encoder_cache_covers_full_token_budget`; `vllm/models/glm5next/common/multimodal.py` modified +24/-0 (24 lines); hunks: -2,6 +2,7; -672,6 +673,29 @@ def _get_video_max_pixels(self) -> int:; symbols: _get_video_max_pixels, get_image_size_with_most_features, _get_video_second_idx_glm46v, touching `_get_video_max_pixels, get_image_size_with_most_features, _get_video_second_idx_glm46v`.
- Code diff details:
  - `tests/models/multimodal/processing/test_glm5next.py` modified +51/-0 (51 lines); hunks: -188,3 +188,54 @@ def test_mm_device_do_normalize():; symbols: test_mm_device_do_normalize, _image_info, test_image_encoder_cache_covers_full_token_budget, test_image_encoder_cache_follows_max_pixels_override
  - `vllm/models/glm5next/common/multimodal.py` modified +24/-0 (24 lines); hunks: -2,6 +2,7; -672,6 +673,29 @@ def _get_video_max_pixels(self) -> int:; symbols: _get_video_max_pixels, get_image_size_with_most_features, _get_video_second_idx_glm46v
- Key code excerpts:

```diff
diff -- tests/models/multimodal/processing/test_glm5next.py
@@ -188,3 +188,54 @@ def test_mm_device_do_normalize():
+def _image_info(**kwargs):
+    ctx = build_model_context(
+        "zai-org/GLM-5.3-Flash",
+        limit_mm_per_prompt={"image": 1},
+        **kwargs,
+    )
diff -- vllm/models/glm5next/common/multimodal.py
@@ -2,6 +2,7 @@
+import math
@@ -672,6 +673,29 @@ def _get_video_max_pixels(self) -> int:
+    def get_image_size_with_most_features(self) -> ImageSize:
+        # The inherited square probe strands budget whenever the token
+        # ceiling is not a perfect square: with max_image_tokens=8000 the
+        # square refits to 2492x2492 (89x89 = 7921 tokens) while a
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/multimodal/processing/test_glm5next.py` modified +51/-0
  - runtime: `vllm/models/glm5next/common/multimodal.py` modified +24/-0
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/processing/test_glm5next.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #58167 - [ROCm][Perf][GLM-5.3-Flash] Add AITER topk backend for decodes

- Link: https://github.com/vllm-project/vllm/pull/58167
- Status/date: merged / 2026-10-02
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/glm5next/test_sparse_indexer_topk_dispatch.py`, `vllm/models/glm5next/amd/sparse_indexer.py`; associated commits `98b29aa99aee`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 8 files, +705/-240, 1241 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/models/glm5next/test_sparse_indexer_topk_dispatch.py` modified +72/-14 (86 lines); hunks: -8,10 +8,6; -20,17 +16,10 @@ def _require_deep_gemm() -> None:; symbols: _require_deep_gemm, test_kpool_indexer_dispatches_through_shared_topk_backend, _build, touching `_require_deep_gemm, test_kpool_indexer_dispatches_through_shared_topk_backend, _build`; `vllm/models/glm5next/amd/sparse_indexer.py` modified +76/-6 (82 lines); hunks: -8,10 +8,11; -24,6 +25,7; symbols: _kpool_compress_insert, _kpool_decode_topk_backend, sparse_attn_indexer_kpool, touching `_kpool_compress_insert, _kpool_decode_topk_backend, sparse_attn_indexer_kpool`.
- Code diff details:
  - `tests/models/glm5next/test_sparse_indexer_topk_dispatch.py` modified +72/-14 (86 lines); hunks: -8,10 +8,6; -20,17 +16,10 @@ def _require_deep_gemm() -> None:; symbols: _require_deep_gemm, test_kpool_indexer_dispatches_through_shared_topk_backend, _build
  - `vllm/models/glm5next/amd/sparse_indexer.py` modified +76/-6 (82 lines); hunks: -8,10 +8,11; -24,6 +25,7; symbols: _kpool_compress_insert, _kpool_decode_topk_backend, sparse_attn_indexer_kpool
- Key code excerpts:

```diff
diff -- tests/models/glm5next/test_sparse_indexer_topk_dispatch.py
@@ -8,10 +8,6 @@
-pytestmark = pytest.mark.skipif(
-    not current_platform.is_cuda(), reason="CUDA-only dispatch"
-)
@@ -20,17 +16,10 @@ def _require_deep_gemm() -> None:
-@pytest.mark.parametrize("backend", ["auto", "persistent", "cooperative", "torch"])
-def test_kpool_indexer_dispatches_through_shared_topk_backend(backend: str) -> None:
diff -- vllm/models/glm5next/amd/sparse_indexer.py
@@ -8,10 +8,11 @@
-from vllm.config import get_current_vllm_config_or_none
+from vllm.config import CUDAGraphMode, get_current_vllm_config_or_none
+from vllm.model_executor.layers.indexer_topk import get_indexer_topk
@@ -24,6 +25,7 @@
+from vllm.utils.math_utils import cdiv
@@ -37,6 +39,9 @@
```

- Extracted files (not manually reviewed):
  - tests: `tests/models/glm5next/test_sparse_indexer_topk_dispatch.py` modified +72/-14
  - runtime: `vllm/models/glm5next/amd/sparse_indexer.py` modified +76/-6
- Risk and verification: The diff ships test coverage in `tests/kernels/test_top_k_per_row.py`, `tests/models/glm5next/test_sparse_indexer_topk_dispatch.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #59126 - [Bugfix][Multimodal] Fix GLM-5.3-Flash vision tower crashes on image input

- Link: https://github.com/vllm-project/vllm/pull/59126
- Status/date: merged / 2026-10-02
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/models/glm5next/common/multimodal.py`; associated commits `097989299222`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +4/-3, 35 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `vllm/models/glm5next/common/multimodal.py` modified +4/-3 (7 lines); hunks: -46,6 +46,7; -395,7 +396,7 @@ def __init__(; symbols: __init__, rot_pos_emb, forward, touching `__init__, rot_pos_emb, forward`.
- Code diff details:
  - `vllm/models/glm5next/common/multimodal.py` modified +4/-3 (7 lines); hunks: -46,6 +46,7; -395,7 +396,7 @@ def __init__(; symbols: __init__, rot_pos_emb, forward
- Key code excerpts:

```diff
diff -- vllm/models/glm5next/common/multimodal.py
@@ -46,6 +46,7 @@
+from vllm.utils.torch_utils import async_tensor_h2d
@@ -395,7 +396,7 @@ def __init__(
-            max_position=8192,
+            max_position=text_config.max_position_embeddings,
@@ -479,7 +480,7 @@ def rot_pos_emb(
-        pos_ids = pos_ids.to(cos.device, non_blocking=True)
```

- Extracted files (not manually reviewed):
  - runtime: `vllm/models/glm5next/common/multimodal.py` modified +4/-3
- Risk and verification: Runtime changes concentrate in `vllm/models/glm5next/common/multimodal.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
