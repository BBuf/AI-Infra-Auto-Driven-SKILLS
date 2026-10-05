# TensorRT-LLM Qwen3.5 Model PR Optimization History

## 2026-08-23 Source Head Refresh

Rechecked TensorRT-LLM upstream main at `NVIDIA/TensorRT-LLM@da38c1d2e0dffd073b7dfb6d69e15ee7b45d84a9`.
The seven-commit range after the previous recorded head
`1b4ffc0291d75a21ad20118e8f44de6e3831f786` was read in full. It contains no
new Qwen3.5-specific implementation commit; the latest PR #16677 is confined
to VisualGen/Wan Attention2D plus TP, so the four previously promoted runtime
PRs remain the current model evidence.

Result: the Qwen3.5 MoE and dense VLM paths landed, followed by fused attention preprocessing and fused AllReduce + Gemma RMSNorm. Test-only unwaives remain outside the runtime evidence set.

| Merged | PR | Runtime signal |
| --- | --- | --- |
| 2026-07-04 | [#14599](https://github.com/NVIDIA/TensorRT-LLM/pull/14599) | Qwen3.5 MoE VLM + MTP |
| 2026-07-07 | [#15249](https://github.com/NVIDIA/TensorRT-LLM/pull/15249) | Qwen3.5 dense VLM |
| 2026-07-21 | [#16469](https://github.com/NVIDIA/TensorRT-LLM/pull/16469) | fused QK norm + RoPE + gate |
| 2026-07-24 | [#15194](https://github.com/NVIDIA/TensorRT-LLM/pull/15194) | fused AllReduce + Gemma RMSNorm |

## 2026-06-27 PR Backfill Audit

The per-PR diff audit cards on this page were generated from TensorRT-LLM
upstream `HEAD@4164b932c6c8a14d1be85d0fd62e44b7d0171980`. The root
TensorRT-LLM history index now tracks the 2026-06-27 runtime refresh at
`aaffa2f9fef3025e0f698d978385a73460344e0b`. This page provides model
implementation coverage, a PR timeline, and per-PR diff audit cards.

Filter used in this pass: merged PRs whose titles or files matched `Qwen3.5`, `Qwen3_5`, `qwen3_5`, `AutoDeploy`, `NVFP4`, `FP8`, `DFlash`, `reasoning_parser`, `EPLB`, `MoE backend`, or `model_registry`. Pure reshuffling and unrelated infrastructure PRs were excluded.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `docs/source/deployment-guide/deployment-guide-for-qwen3.8-qwen3.5-on-trtllm.md` | no direct PR-number commit |
| `examples/configs/curated/qwen3.5.yaml` | [#15111](https://github.com/NVIDIA/TensorRT-LLM/pull/15111) |
| `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` | [#12242](https://github.com/NVIDIA/TensorRT-LLM/pull/12242), [#13090](https://github.com/NVIDIA/TensorRT-LLM/pull/13090), [#13716](https://github.com/NVIDIA/TensorRT-LLM/pull/13716), [#14164](https://github.com/NVIDIA/TensorRT-LLM/pull/14164), [#14465](https://github.com/NVIDIA/TensorRT-LLM/pull/14465), [#14599](https://github.com/NVIDIA/TensorRT-LLM/pull/14599), [#15067](https://github.com/NVIDIA/TensorRT-LLM/pull/15067), [#15249](https://github.com/NVIDIA/TensorRT-LLM/pull/15249), [#16642](https://github.com/NVIDIA/TensorRT-LLM/pull/16642), [#16936](https://github.com/NVIDIA/TensorRT-LLM/pull/16936), [#17433](https://github.com/NVIDIA/TensorRT-LLM/pull/17433), [#19519](https://github.com/NVIDIA/TensorRT-LLM/pull/19519) |
| `tensorrt_llm/_torch/models/modeling_qwen3_5.py` | [#12242](https://github.com/NVIDIA/TensorRT-LLM/pull/12242), [#12646](https://github.com/NVIDIA/TensorRT-LLM/pull/12646), [#14164](https://github.com/NVIDIA/TensorRT-LLM/pull/14164), [#14465](https://github.com/NVIDIA/TensorRT-LLM/pull/14465), [#14599](https://github.com/NVIDIA/TensorRT-LLM/pull/14599), [#15067](https://github.com/NVIDIA/TensorRT-LLM/pull/15067), [#15249](https://github.com/NVIDIA/TensorRT-LLM/pull/15249), [#16353](https://github.com/NVIDIA/TensorRT-LLM/pull/16353), [#16642](https://github.com/NVIDIA/TensorRT-LLM/pull/16642), [#17293](https://github.com/NVIDIA/TensorRT-LLM/pull/17293), [#17700](https://github.com/NVIDIA/TensorRT-LLM/pull/17700) |
| `tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp1_gentp1_qwen3_5_4b_fp8_tllm.yaml` | no direct PR-number commit |
| `tests/scripts/perf-sanity/aggregated/qwen3_5_397b_fp4_blackwell.yaml` | [#15650](https://github.com/NVIDIA/TensorRT-LLM/pull/15650) |
| `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl.py` | [#15249](https://github.com/NVIDIA/TensorRT-LLM/pull/15249), [#16065](https://github.com/NVIDIA/TensorRT-LLM/pull/16065), [#16264](https://github.com/NVIDIA/TensorRT-LLM/pull/16264) |
| `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py` | [#14164](https://github.com/NVIDIA/TensorRT-LLM/pull/14164), [#14465](https://github.com/NVIDIA/TensorRT-LLM/pull/14465), [#14599](https://github.com/NVIDIA/TensorRT-LLM/pull/14599), [#16065](https://github.com/NVIDIA/TensorRT-LLM/pull/16065), [#17293](https://github.com/NVIDIA/TensorRT-LLM/pull/17293), [#17700](https://github.com/NVIDIA/TensorRT-LLM/pull/17700) |
| `tests/unittest/_torch/modeling/test_qwen3_5_partial_loading.py` | [#16642](https://github.com/NVIDIA/TensorRT-LLM/pull/16642) |
| `tests/unittest/_torch/models/checkpoints/hf/test_qwen3_5_weight_mapper.py` | [#17433](https://github.com/NVIDIA/TensorRT-LLM/pull/17433), [#19519](https://github.com/NVIDIA/TensorRT-LLM/pull/19519) |

## PR Coverage Summary

- Git-traced PRs: 20
- Extra PRs preserved from existing docs: 13
- Total PRs in this document: 33
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-02-26 | [#11728](https://github.com/NVIDIA/TensorRT-LLM/pull/11728) | merged | Added Qwen3.5 Cookbook | AutoDeploy cookbook notebook |
| 2026-03-20 | [#12242](https://github.com/NVIDIA/TensorRT-LLM/pull/12242) | merged | [None][feat] Initial Qwen3.5 text model support for PyT backend (BF16/FP8) | `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py`, `tensorrt_llm/_torch/models/modeling_qwen3_5.py` |
| 2026-03-24 | [#12302](https://github.com/NVIDIA/TensorRT-LLM/pull/12302) | merged | Add Qwen 3.5 supporting (NVFP4) | model wrapper, weight mapper, tests |
| 2026-03-25 | [#12114](https://github.com/NVIDIA/TensorRT-LLM/pull/12114) | merged | Qwen 3.5 fix 3d position ID handling | AutoDeploy Qwen3.5 MoE, mRoPE cache, registry configs |
| 2026-04-30 | [#13090](https://github.com/NVIDIA/TensorRT-LLM/pull/13090) | merged | Qwen3.5 dense weight loading | weight mapper, dense tests |
| 2026-05-04 | [#13716](https://github.com/NVIDIA/TensorRT-LLM/pull/13716) | merged | Fix Qwen3.5 NVFP4 weight loading by preserving weight_scales | HF mapper |
| 2026-05-12 | [#13782](https://github.com/NVIDIA/TensorRT-LLM/pull/13782) | merged | Qwen3.5 DFlash | speculative/DFlash runtime |
| 2026-05-16 | [#13996](https://github.com/NVIDIA/TensorRT-LLM/pull/13996) | merged | Perf optimizations for DFlash | DFlash model engine and speculative code |
| 2026-05-21 | [#12646](https://github.com/NVIDIA/TensorRT-LLM/pull/12646) | merged | [TRTLLM-11547][feat] Add Qwen3.5 MTP support. | `tensorrt_llm/_torch/models/modeling_qwen3_5.py` |
| 2026-05-21 | [#14164](https://github.com/NVIDIA/TensorRT-LLM/pull/14164) | merged | [TRTLLM-12500][feat] Add support for Qwen3.5 VL MoE - REVERTED by #14599 | `tensorrt_llm/_torch/models/modeling_qwen3_5.py`, `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py`, `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py` |
| 2026-05-23 | [#14465](https://github.com/NVIDIA/TensorRT-LLM/pull/14465) | merged | [None][feat] Revert Add support for Qwen3.5 VL MoE (#14164) | `tensorrt_llm/_torch/models/modeling_qwen3_5.py`, `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py`, `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py` |
| 2026-05-29 | [#14659](https://github.com/NVIDIA/TensorRT-LLM/pull/14659) | merged | Add a reasoning parser for qwen3_5 | `reasoning_parser.py` |
| 2026-06-02 | [#14667](https://github.com/NVIDIA/TensorRT-LLM/pull/14667) | merged | AutoDeploy: Qwen3.5 400B NVFP4 accuracy regression fix | shared expert sharding, SwiGLU fusion |
| 2026-06-05 | [#15001](https://github.com/NVIDIA/TensorRT-LLM/pull/15001) | merged | Uncomment Qwen3.5 from model registry | `models.yaml` |
| 2026-06-09 | [#15081](https://github.com/NVIDIA/TensorRT-LLM/pull/15081) | merged | Select CUTLASS MoE backend on non-Blackwell SMs | accuracy test backend selection |
| 2026-06-09 | [#15111](https://github.com/NVIDIA/TensorRT-LLM/pull/15111) | merged | [TRTLLM-11548][doc] Add Qwen3.5 deployment guide doc | `examples/configs/curated/qwen3.5.yaml` |
| 2026-06-11 | [#15067](https://github.com/NVIDIA/TensorRT-LLM/pull/15067) | merged | Generalize FP8 checkpoint loading for Qwen3.5 | weight mapper, modeling |
| 2026-06-13 | [#15185](https://github.com/NVIDIA/TensorRT-LLM/pull/15185) | merged | Qwen3.5 whitelist sharding and lm_head sharding | AutoDeploy sharding IR/tests |
| 2026-06-26 | [#15543](https://github.com/NVIDIA/TensorRT-LLM/pull/15543) | merged | Add EPLB support for Qwen3.5 | MoE load balancer and B200/GB200 tests |
| 2026-06-26 | [#15650](https://github.com/NVIDIA/TensorRT-LLM/pull/15650) | merged | [None][test] Add Qwen3.5-397B-A17B-NVFP4 B200 aggregated perf-sanity tests | `tests/scripts/perf-sanity/aggregated/qwen3_5_397b_fp4_blackwell.yaml` |
| 2026-07-04 | [#14599](https://github.com/NVIDIA/TensorRT-LLM/pull/14599) | Qwen3.5 MoE VLM + MTP |
| 2026-07-07 | [#15249](https://github.com/NVIDIA/TensorRT-LLM/pull/15249) | Qwen3.5 dense VLM |
| 2026-07-08 | [#16065](https://github.com/NVIDIA/TensorRT-LLM/pull/16065) | merged | [https://nvbugs/6422332][fix] Keep SSM cache in weights dtype when ma… | `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py`, `tensorrt_llm/_torch/pyexecutor/config_utils.py`, `tensorrt_llm/_torch/pyexecutor/model_loader.py` |
| 2026-07-14 | [#16264](https://github.com/NVIDIA/TensorRT-LLM/pull/16264) | merged | [None][fix] Align dense Qwen3.5-VL SSM cache dtype test with #16065 semantics | `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl.py` |
| 2026-07-14 | [#16353](https://github.com/NVIDIA/TensorRT-LLM/pull/16353) | merged | [TRTLLM-14054][perf] Qwen3.5-VL: pass the inner LM's normalized model_config to the weight mapper | `tensorrt_llm/_torch/models/modeling_qwen3_5.py` |
| 2026-07-21 | [#16469](https://github.com/NVIDIA/TensorRT-LLM/pull/16469) | fused QK norm + RoPE + gate |
| 2026-07-24 | [#15194](https://github.com/NVIDIA/TensorRT-LLM/pull/15194) | fused AllReduce + Gemma RMSNorm |
| 2026-07-29 | [#16936](https://github.com/NVIDIA/TensorRT-LLM/pull/16936) | merged | [None][fix] Fix Qwen3.5 weight-load memory growth and MTP CUTLASS fallback | `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` |
| 2026-07-31 | [#16642](https://github.com/NVIDIA/TensorRT-LLM/pull/16642) | merged | [TRTLLM-14497][feat] Add BF16/FP8 refit for qwen3.5_397b | `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py`, `tensorrt_llm/_torch/models/modeling_qwen3_5.py`, `tests/unittest/_torch/modeling/test_qwen3_5_partial_loading.py` |
| 2026-08-10 | [#17293](https://github.com/NVIDIA/TensorRT-LLM/pull/17293) | merged | [https://nvbugs/6434512][fix] Select Marlin for Qwen3.5 MoE on Hopper | `tensorrt_llm/_torch/models/modeling_qwen3_5.py`, `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py` |
| 2026-08-11 | [#17433](https://github.com/NVIDIA/TensorRT-LLM/pull/17433) | merged | [None][fix] Qwen3.5 weight mapper for FP8 per-channel checkpoints | `tests/unittest/_torch/models/checkpoints/hf/test_qwen3_5_weight_mapper.py`, `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` |
| 2026-08-21 | [#17700](https://github.com/NVIDIA/TensorRT-LLM/pull/17700) | merged | [None][perf] Qwen3.5/3.8 wave-2: MoE, attention-DP, GDN replay, weight loading | `tensorrt_llm/_torch/models/modeling_qwen3_5.py`, `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py` |
| 2026-09-23 | [#19519](https://github.com/NVIDIA/TensorRT-LLM/pull/19519) | merged | [https://nvbugs/6771102][fix] Support Qwen3.5 global FP8 checkpoints | `tests/unittest/_torch/models/checkpoints/hf/test_qwen3_5_weight_mapper.py`, `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` |

## Per-PR Diff Audit Cards

### PR #11728 - Added Qwen3.5 Cookbook

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/11728
- Status/date: merged / 2026-02-26
- Trace source: `git log --name-only -- <model-files>` plus GitHub Pull Request files API.
- Diff scope read: 1 file, +385/-0, 402 cached patch lines.
- Motivation: document how to deploy Qwen3.5-397B and its NVFP4 checkpoint with AutoDeploy.
- Key implementation: adds a notebook with `trtllm-serve`, AutoDeploy registry config, B200 sizing, and sample OpenAI calls.
- Code diff details: See the diff scope line above and the excerpt below for the audited file-level changes.
- Key code excerpts:

```diff
+trtllm-serve "nvidia/Qwen3.5-397B-A17B-NVFP4" \
+MODEL_ID = "Qwen/Qwen3.5-397B-A17B"
```

- Reviewed files: `examples/auto_deploy/cookbooks/qwen_3.5_trtllm_cookbook.ipynb`
- Risk and verification: use this as deployment evidence, not as proof that the PyTorch backend path is identical to SGLang.

### PR #12242 - [None][feat] Initial Qwen3.5 text model support for PyT backend (BF16/FP8)

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/12242
- Status/date: merged / 2026-03-20
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py`, `tensorrt_llm/_torch/models/modeling_qwen3_5.py`; associated commits `95db1d895e87`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 22 files, +854/-92, 1349 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` added +269/-0 (269 lines); hunks: -0,0 +1,269; symbols: Qwen3_5MoeHfWeightMapper, _normalize_weight_names, handle_special_instance_module, _pack_projection_tensor, touching `Qwen3_5MoeHfWeightMapper, _normalize_weight_names, handle_special_instance_module`; `tensorrt_llm/_torch/models/modeling_qwen3_5.py` added +27/-0 (27 lines); hunks: -0,0 +1,27; symbols: Qwen3_5MoeForCausalLM, exists, that, touching `Qwen3_5MoeForCausalLM, exists, that`.
- Code diff details:
  - `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` added +269/-0 (269 lines); hunks: -0,0 +1,269; symbols: Qwen3_5MoeHfWeightMapper, _normalize_weight_names, handle_special_instance_module, _pack_projection_tensor
  - `tensorrt_llm/_torch/models/modeling_qwen3_5.py` added +27/-0 (27 lines); hunks: -0,0 +1,27; symbols: Qwen3_5MoeForCausalLM, exists, that
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py
@@ -0,0 +1,269 @@
+import math
+import re
+from collections import defaultdict
+import torch
+from torch import nn
+from tensorrt_llm._torch.models.checkpoints.hf.qwen3_next_weight_mapper import (
diff -- tensorrt_llm/_torch/models/modeling_qwen3_5.py
@@ -0,0 +1,27 @@
+from .modeling_qwen3_next import Qwen3NextForCausalLM
+from .modeling_utils import register_auto_model
+@register_auto_model("Qwen3_5MoeForCausalLM")
+class Qwen3_5MoeForCausalLM(Qwen3NextForCausalLM):
+    """Thin wrapper that registers the Qwen3.5 MoE text architecture.
+    Qwen3.5 text reuses the same model internals as Qwen3Next
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` added +269/-0; `tensorrt_llm/_torch/models/modeling_qwen3_5.py` added +27/-0
- Risk and verification: The diff ships test coverage in `tests/integration/defs/accuracy/references/gsm8k.yaml`, `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/integration/test_lists/qa/llm_function_core.txt`, `tests/integration/test_lists/qa/llm_function_core_sanity.txt`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #12302 - Add Qwen 3.5 supporting (NVFP4)

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/12302
- Status/date: merged / 2026-03-24
- Trace source: `git log --name-only -- <model-files>` plus GitHub Pull Request files API.
- Diff scope read: 9 files, +225/-31, 436 cached patch lines.
- Motivation: support Qwen3.5 dense/MoE and the official NVFP4 checkpoint in the PyTorch backend.
- Key implementation: registers dense and MoE Qwen3.5 model wrappers, extends the HF mapper, and adds 397B NVFP4 accuracy tests.
- Code diff details: See the diff scope line above and the excerpt below for the audited file-level changes.
- Key code excerpts:

```diff
+@register_auto_model("Qwen3_5ForCausalLM")
+class Qwen3_5ForCausalLM(Qwen3NextForCausalLM):
```

- Reviewed files: `modeling_qwen3_5.py`, `qwen3_5_weight_mapper.py`, `config_utils.py`, accuracy refs/tests
- Risk and verification: separate dense and MoE wrapper behavior in comparisons.

### PR #12114 - Qwen 3.5 fix 3D position ID handling

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/12114
- Status/date: merged / 2026-03-25
- Trace source: `git log --name-only -- <model-files>` plus GitHub Pull Request files API.
- Diff scope read: 15 files, +3448/-275, 7,822 cached patch lines.
- Motivation: Qwen3.5 VLM/mRoPE needed 3D positions, chunked multimodal positions, video grid normalization, and mRoPE delta cache.
- Key implementation: extends AutoDeploy Qwen3.5 MoE modeling, mRoPE cache transforms, registry configs, and unit tests.
- Code diff details: See the diff scope line above and the excerpt below for the audited file-level changes.
- Key code excerpts:

```diff
+@TransformRegistry.register("initialize_mrope_delta_cache")
+mm_token_positions: torch.Tensor
```

- Reviewed files: `modeling_qwen3_5_moe.py`, `mrope_delta_cache.py`, registry YAMLs, `test_qwen3_5_moe.py`, serving utils tests
- Risk and verification: multimodal correctness depends on position construction and cache resources, not only decode kernels.

### PR #13090 - Qwen3.5 dense weight loading

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/13090
- Status/date: merged / 2026-04-30
- Trace source: `git log --name-only -- <model-files>` plus GitHub Pull Request files API.
- Diff scope read: 5 files, +85/-1, 225 cached patch lines.
- Motivation: dense Qwen3.5 4B/FP8 loading needed direct coverage.
- Key implementation: updates the Qwen3.5 HF mapper and adds dense accuracy refs/tests.
- Code diff details: See the diff scope line above and the excerpt below for the audited file-level changes.
- Key code excerpts:

```diff
+class TestQwen3_5_4B(LlmapiAccuracyTestHarness):
+MODEL_NAME = "Qwen/Qwen3.5-4B"
```

- Reviewed files: HF mapper, accuracy refs, `test_llm_api_pytorch.py`, test lists
- Risk and verification: dense Qwen3.5 has different loading risks from 397B MoE.

### PR #13716 - Preserve Qwen3.5 NVFP4 weight_scales

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/13716
- Status/date: merged / 2026-05-04
- Trace source: `git log --name-only -- <model-files>` plus GitHub Pull Request files API.
- Diff scope read: 1 file, +9/-3, 45 cached patch lines.
- Motivation: FP8 scale remapping broke NVFP4 weight scale loading.
- Key implementation: detects NVFP4 prefixes and preserves `weight_scales`.
- Code diff details: See the diff scope line above and the excerpt below for the audited file-level changes.
- Key code excerpts:

```diff
+        nvfp4_prefixes = {
+            key[: -len(".weight_scale_2")] for key in weights if key.endswith(".weight_scale_2")
+        }
+                if prefix not in nvfp4_prefixes:
```

- Reviewed files: `qwen3_5_weight_mapper.py`
- Risk and verification: scale key remapping is a first check for NVFP4 loading or accuracy issues.

### PR #13782 - Qwen3.5 DFlash

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/13782
- Status/date: merged / 2026-05-12
- Trace source: `git log --name-only -- <model-files>` plus GitHub Pull Request files API.
- Diff scope read: 5 files, +144/-55, 413 cached patch lines.
- Motivation: enable Qwen3.5 hybrid linear-attention models on DFlash/speculative paths.
- Key implementation: wires GDN/Mamba cache and model engine paths into DFlash runtime.
- Code diff details: See the diff scope line above and the excerpt below for the audited file-level changes.
- Key code excerpts:

```diff
+from tensorrt_llm._torch.speculative import dflash
+mamba_cache_manager
```

- Reviewed files: `gdn_mixer.py`, `pyexecutor/_util.py`, `mamba_cache_manager.py`, `model_engine.py`, `speculative/dflash.py`
- Risk and verification: keep DFlash separate from plain decoding and SGLang MTP comparisons.

### PR #13996 - Perf optimizations for DFlash

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/13996
- Status/date: merged / 2026-05-16
- Trace source: `git log --name-only -- <model-files>` plus GitHub Pull Request files API.
- Diff scope read: 5 files, +455/-285, 1,606 cached patch lines.
- Motivation: reduce DFlash overhead after the initial Qwen3.5 support.
- Key implementation: changes speculative modeling, GDN mixer, model engine, DFlash runtime, and `llm_args.py`.
- Code diff details: See the diff scope line above and the excerpt below for the audited file-level changes.
- Key code excerpts:

```diff
+    def _build_fused_kv_buffers(self) -> None:
+        """Stack per-layer KV projection + k_norm weights for a single fused GEMM.
+        return self.max_draft_len + 1
```

- Reviewed files: `modeling_speculative.py`, `gdn_mixer.py`, `model_engine.py`, `speculative/dflash.py`, `llm_args.py`
- Risk and verification: if TensorRT-LLM leads through DFlash, attribute the gap to speculative runtime rather than one kernel.

### PR #12646 - [TRTLLM-11547][feat] Add Qwen3.5 MTP support.

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/12646
- Status/date: merged / 2026-05-21
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_qwen3_5.py`; associated commits `5d19712ae7c1`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 10 files, +303/-42, 564 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_qwen3_5.py` modified +64/-5 (69 lines); hunks: -6,25 +6,84; symbols: _translate_mtp_pattern, _normalize_qwen35_exclude_modules, touching `_translate_mtp_pattern, _normalize_qwen35_exclude_modules`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_qwen3_5.py` modified +64/-5 (69 lines); hunks: -6,25 +6,84; symbols: _translate_mtp_pattern, _normalize_qwen35_exclude_modules
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3_5.py
@@ -6,25 +6,84 @@
+_MTP_TOP_TO_TRTLLM = {
+    "fc": "fc",
+    "norm": "shared_head.norm",
+    "pre_fc_norm_embedding": "pre_fc_norm_embedding",
+    "pre_fc_norm_hidden": "pre_fc_norm_hidden",
+}
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_5.py` modified +64/-5
- Risk and verification: The diff ships test coverage in `tests/integration/defs/accuracy/references/gsm8k.yaml`, `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/integration/test_lists/qa/llm_function_core.txt`, `tests/integration/test_lists/test-db/l0_b200.yml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #14164 - [TRTLLM-12500][feat] Add support for Qwen3.5 VL MoE - REVERTED by #14599

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/14164
- Status/date: merged / 2026-05-21
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py`, `tensorrt_llm/_torch/models/modeling_qwen3_5.py`, `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py`; associated commits `751be5d9b516`, `96a4a0937e37`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 14 files, +1037/-173, 1420 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_qwen3_5.py` modified +312/-2 (314 lines); hunks: -1,7 +1,29; -51,6 +73,248 @@ def _translate_mtp_pattern(name, n_hidden_layers):; symbols: _translate_mtp_pattern, Qwen35ConfigCompat, and, normalize, touching `_translate_mtp_pattern, Qwen35ConfigCompat, and`; `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` modified +1/-0 (1 lines); hunks: -13,6 +13,7; symbols: Qwen3_5MoeHfWeightMapper, touching `Qwen3_5MoeHfWeightMapper`; `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py` added +450/-0 (450 lines); hunks: -0,0 +1,450; symbols: _write_qwen35_moe_vl_config, test_qwen35_moe_vl_config_preserves_vlm_architecture, test_qwen35_moe_vl_resolves_mamba_ssm_cache_dtype, test_qwen35_moe_vl_resolves_model_and_mapper, touching `_write_qwen35_moe_vl_config, test_qwen35_moe_vl_config_preserves_vlm_architecture, test_qwen35_moe_vl_resolves_mamba_ssm_cache_dtype`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_qwen3_5.py` modified +312/-2 (314 lines); hunks: -1,7 +1,29; -51,6 +73,248 @@ def _translate_mtp_pattern(name, n_hidden_layers):; symbols: _translate_mtp_pattern, Qwen35ConfigCompat, and, normalize
  - `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` modified +1/-0 (1 lines); hunks: -13,6 +13,7; symbols: Qwen3_5MoeHfWeightMapper
  - `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py` added +450/-0 (450 lines); hunks: -0,0 +1,450; symbols: _write_qwen35_moe_vl_config, test_qwen35_moe_vl_config_preserves_vlm_architecture, test_qwen35_moe_vl_resolves_mamba_ssm_cache_dtype, test_qwen35_moe_vl_resolves_model_and_mapper
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3_5.py
@@ -1,7 +1,29 @@
+from types import SimpleNamespace
+from typing import Dict, List
+import torch
+from transformers import PretrainedConfig
+from ...inputs import (
+    ContentFormat,
diff -- tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py
@@ -13,6 +13,7 @@
+@register_mapper("HF", "Qwen3_5MoeForConditionalGeneration")
diff -- tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py
@@ -0,0 +1,450 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+import json
+import os
+from copy import deepcopy
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_5.py` modified +312/-2; `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` modified +1/-0
  - tests: `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py` added +450/-0
- Risk and verification: The diff ships test coverage in `tests/integration/defs/accuracy/references/mmmu.yaml`, `tests/integration/defs/accuracy/test_llm_api_pytorch_multimodal.py`, `tests/integration/test_lists/qa/llm_function_core.txt`, `tests/integration/test_lists/test-db/l0_l40s.yml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #14465 - [None][feat] Revert Add support for Qwen3.5 VL MoE (#14164)

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/14465
- Status/date: merged / 2026-05-23
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py`, `tensorrt_llm/_torch/models/modeling_qwen3_5.py`, `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py`; associated commits `751be5d9b516`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 14 files, +173/-1037, 1420 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_qwen3_5.py` modified +2/-312 (314 lines); hunks: -1,29 +1,7; -73,248 +51,6 @@ def _translate_mtp_pattern(name, n_hidden_layers):; symbols: _translate_mtp_pattern, Qwen35ConfigCompat, and, normalize, touching `_translate_mtp_pattern, Qwen35ConfigCompat, and`; `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` modified +0/-1 (1 lines); hunks: -13,7 +13,6; symbols: Qwen3_5MoeHfWeightMapper, touching `Qwen3_5MoeHfWeightMapper`; `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py` removed +0/-450 (450 lines); hunks: -1,450 +0,0; symbols: _write_qwen35_moe_vl_config, test_qwen35_moe_vl_config_preserves_vlm_architecture, test_qwen35_moe_vl_resolves_mamba_ssm_cache_dtype, test_qwen35_moe_vl_resolves_model_and_mapper, touching `_write_qwen35_moe_vl_config, test_qwen35_moe_vl_config_preserves_vlm_architecture, test_qwen35_moe_vl_resolves_mamba_ssm_cache_dtype`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_qwen3_5.py` modified +2/-312 (314 lines); hunks: -1,29 +1,7; -73,248 +51,6 @@ def _translate_mtp_pattern(name, n_hidden_layers):; symbols: _translate_mtp_pattern, Qwen35ConfigCompat, and, normalize
  - `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` modified +0/-1 (1 lines); hunks: -13,7 +13,6; symbols: Qwen3_5MoeHfWeightMapper
  - `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py` removed +0/-450 (450 lines); hunks: -1,450 +0,0; symbols: _write_qwen35_moe_vl_config, test_qwen35_moe_vl_config_preserves_vlm_architecture, test_qwen35_moe_vl_resolves_mamba_ssm_cache_dtype, test_qwen35_moe_vl_resolves_model_and_mapper
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3_5.py
@@ -1,29 +1,7 @@
-from types import SimpleNamespace
-from typing import Dict, List
-import torch
-from transformers import PretrainedConfig
-from ...inputs import (
-    ContentFormat,
diff -- tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py
@@ -13,7 +13,6 @@
-@register_mapper("HF", "Qwen3_5MoeForConditionalGeneration")
diff -- tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py
@@ -1,450 +0,0 @@
-# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-# SPDX-License-Identifier: Apache-2.0
-import json
-import os
-from copy import deepcopy
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_5.py` modified +2/-312; `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` modified +0/-1
  - tests: `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py` removed +0/-450
- Risk and verification: The diff ships test coverage in `tests/integration/defs/accuracy/references/mmmu.yaml`, `tests/integration/defs/accuracy/test_llm_api_pytorch_multimodal.py`, `tests/integration/test_lists/qa/llm_function_core.txt`, `tests/integration/test_lists/test-db/l0_l40s.yml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #14659 - Add a reasoning parser for qwen3_5

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/14659
- Status/date: merged / 2026-05-29
- Trace source: `git log --name-only -- <model-files>` plus GitHub Pull Request files API.
- Diff scope read: 1 file, +9/-0, 30 cached patch lines.
- Motivation: Qwen3.5 forced-thinking output begins inside the reasoning block.
- Key implementation: registers `qwen3_5` with `reasoning_at_start=True`.
- Code diff details: See the diff scope line above and the excerpt below for the audited file-level changes.
- Key code excerpts:

```diff
+@register_reasoning_parser("qwen3_5", reasoning_at_start=True)
```

- Reviewed files: `llmapi/reasoning_parser.py`
- Risk and verification: output parsing can change benchmark scores independently of runtime speed.

### PR #14667 - AutoDeploy Qwen3.5 400B NVFP4 accuracy regression fix

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/14667
- Status/date: merged / 2026-06-02
- Trace source: `git log --name-only -- <model-files>` plus GitHub Pull Request files API.
- Diff scope read: 5 files, +72/-35, 464 cached patch lines.
- Motivation: fix a Qwen3.5 400B NVFP4 AutoDeploy accuracy regression.
- Key implementation: replicates the shared expert instead of TP-sharding it and expands SwiGLU fusion/sharding hints.
- Code diff details: See the diff scope line above and the excerpt below for the audited file-level changes.
- Key code excerpts:

```diff
+# The shared expert is replicated
+apply_sharding_hints:
```

- Reviewed files: `qwen3.5_moe_400b.yaml`, `modeling_qwen3_5_moe.py`, `swiglu.py`, `fuse_swiglu.py`, waives
- Risk and verification: inspect shared expert sharding and SwiGLU fusion before blaming MoE GEMMs.

### PR #15001 - Uncomment Qwen3.5 from model registry

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/15001
- Status/date: merged / 2026-06-05
- Trace source: `git log --name-only -- <model-files>` plus GitHub Pull Request files API.
- Diff scope read: 1 file, +9/-12, 50 cached patch lines.
- Motivation: make Qwen3.5 AutoDeploy entries discoverable by default.
- Key implementation: enables Qwen3.5 35B and 397B entries in `models.yaml`.
- Code diff details: See the diff scope line above and the excerpt below for the audited file-level changes.
- Key code excerpts:

```diff
+- name: Qwen/Qwen3.5-397B-A17B
+  config_id: qwen3_5_moe_400b
```

- Reviewed files: `examples/auto_deploy/model_registry/models.yaml`
- Risk and verification: registry entries are official deployment lanes for fair comparison.

### PR #15081 - Select CUTLASS MoE backend on non-Blackwell SMs

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/15081
- Status/date: merged / 2026-06-09
- Trace source: `git log --name-only -- <model-files>` plus GitHub Pull Request files API.
- Diff scope read: 2 files, +8/-2, 52 cached patch lines.
- Motivation: DeepGEMM should be used on Blackwell, while non-Blackwell tests need CUTLASS.
- Key implementation: picks the MoE backend by SM version in Qwen3.5 FP8 tests.
- Code diff details: See the diff scope line above and the excerpt below for the audited file-level changes.
- Key code excerpts:

```diff
+moe_backend = "DEEPGEMM" if get_sm_version() in (100, 103) else "CUTLASS"
```

- Reviewed files: `test_llm_api_pytorch.py`, `waives.txt`
- Risk and verification: never mix H100 and B200 MoE backend results without recording backend choice.

### PR #15111 - [TRTLLM-11548][doc] Add Qwen3.5 deployment guide doc

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/15111
- Status/date: merged / 2026-06-09
- Trace source: `git log --name-only -- <model-files>` found it through `examples/configs/curated/qwen3.5.yaml`; associated commits `09ebc592e535`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 7 files, +97/-29, 273 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `examples/configs/curated/qwen3.5.yaml` added +15/-0 (15 lines); hunks: -0,0 +1,15.
- Code diff details:
  - `examples/configs/curated/qwen3.5.yaml` added +15/-0 (15 lines); hunks: -0,0 +1,15
- Key code excerpts:

```diff
diff -- examples/configs/curated/qwen3.5.yaml
@@ -0,0 +1,15 @@
+max_batch_size: 512
+max_num_tokens: 2048
+tensor_parallel_size: 4
+moe_expert_parallel_size: 4
+trust_remote_code: true
+enable_attention_dp: true
```

- Extracted files (not manually reviewed):
  - docs: `examples/configs/curated/qwen3.5.yaml` added +15/-0
- Risk and verification: This is mostly docs/examples in `docs/source/_static/config_db.json`, `docs/source/deployment-guide/deployment-guide-for-qwen3.5-on-trtllm.md`, `docs/source/deployment-guide/index.rst`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #15067 - Generalize FP8 checkpoint loading for Qwen3.5

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/15067
- Status/date: merged / 2026-06-11
- Trace source: `git log --name-only -- <model-files>` plus GitHub Pull Request files API.
- Diff scope read: 2 files, +68/-48, 220 cached patch lines.
- Motivation: make FP8 checkpoint loading handle Qwen3.5 naming and exclude-module variants.
- Key implementation: refactors mapper/modeling normalization around FP8/NVFP4.
- Code diff details: See the diff scope line above and the excerpt below for the audited file-level changes.
- Key code excerpts:

```diff
+    # gdn_mixer uses Linear module for weight management of depthwise conv1d
+    # but conv1d is not a proper linear module and should be excluded from quant
+    normalized.add("*linear_attn.conv1d")
```

- Reviewed files: `qwen3_5_weight_mapper.py`, `modeling_qwen3_5.py`
- Risk and verification: check mapper normalization before kernel-level debugging for FP8 loading issues.

### PR #15185 - Qwen3.5 whitelist sharding and lm_head sharding

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/15185
- Status/date: merged / 2026-06-13
- Trace source: `git log --name-only -- <model-files>` plus GitHub Pull Request files API.
- Diff scope read: 5 files, +193/-118, 735 cached patch lines.
- Motivation: AutoDeploy needed whitelist sharding and `lm_head` sharding for Qwen3.5.
- Key implementation: updates registry configs, model sharding hints, SwiGLU fusion, and sharding IR tests.
- Code diff details: See the diff scope line above and the excerpt below for the audited file-level changes.
- Key code excerpts:

```diff
+lm_head:
+apply_sharding_hints
```

- Reviewed files: registry YAML, `modeling_qwen3_5_moe.py`, `fuse_swiglu.py`, `sharding_ir.py`, tests
- Risk and verification: inspect `lm_head` and shared-expert sharding separately from expert GEMMs.

### PR #15543 - Add EPLB support for Qwen3.5

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/15543
- Status/date: merged / 2026-06-26
- Trace source: `git log --name-only -- <model-files>` plus GitHub Pull Request files API.
- Diff scope read: 3 files, +73/-0, 130 cached patch lines.
- Motivation: add EPLB coverage for Qwen3.5 MoE on B200/GB200 test lanes.
- Key implementation: extends the MoE load balancer and test DB entries.
- Code diff details: See the diff scope line above and the excerpt below for the audited file-level changes.
- Key code excerpts:

```diff
+    'Qwen2MoeForCausalLM',
+    'Qwen3MoeForCausalLM',
+    'Qwen3_5MoeForCausalLM',
```

- Reviewed files: `moe_load_balancer.py`, `test_llm_api_pytorch.py`, B200/GB200 test DB YAMLs
- Risk and verification: record whether load balancing is enabled when comparing SGLang EP/EPLB behavior.


### PR #15650 - [None][test] Add Qwen3.5-397B-A17B-NVFP4 B200 aggregated perf-sanity tests

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/15650
- Status/date: merged / 2026-06-26
- Trace source: `git log --name-only -- <model-files>` found it through `tests/scripts/perf-sanity/aggregated/qwen3_5_397b_fp4_blackwell.yaml`; associated commits `d419595e40d3`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +440/-5, 460 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/scripts/perf-sanity/aggregated/qwen3_5_397b_fp4_blackwell.yaml` added +425/-0 (425 lines); hunks: -0,0 +1,425.
- Code diff details:
  - `tests/scripts/perf-sanity/aggregated/qwen3_5_397b_fp4_blackwell.yaml` added +425/-0 (425 lines); hunks: -0,0 +1,425
- Key code excerpts:

```diff
diff -- tests/scripts/perf-sanity/aggregated/qwen3_5_397b_fp4_blackwell.yaml
@@ -0,0 +1,425 @@
+metadata:
+  model_name: qwen3.5_397b_a17b_fp4
+  supported_gpus:
+  - B200
+hardware:
+  gpus_per_node: 8
```

- Extracted files (not manually reviewed):
  - tests: `tests/scripts/perf-sanity/aggregated/qwen3_5_397b_fp4_blackwell.yaml` added +425/-0
- Risk and verification: The diff ships test coverage in `tests/integration/test_lists/test-db/l0_b200_multi_gpus_perf_sanity.yml`, `tests/scripts/perf-sanity/aggregated/qwen3_5_397b_fp4_blackwell.yaml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #14599 - Add support for Qwen3.5 VL MoE with MTP fixes

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/14599
- Status/date: merged / 2026-07-04
- Trace source: `git log --name-only -- <model-files>` plus the final upstream commit and PR body.
- Diff scope read: full 1,734-line diff, 16 files, +1140/-256.
- Motivation: TensorRT-LLM had a reusable Qwen3Next text runtime and Qwen3-VL vision tower, but lacked the composite Qwen3.5-35B-A3B multimodal architecture, native config normalization, weight mapping, and speculative-decoding token plumbing.
- Key implementation: registers `Qwen3_5MoeForConditionalGeneration`, preserves the HF text/vision subconfigs while normalizing runtime aliases, composes `Qwen3VisionModel` with the MoE decoder, maps language-model weights, and recovers `orig_input_ids` for MTP/Eagle after the VLM wrapper builds embeddings.
- Code diff details: the VLM class owns multimodal placeholder metadata and device paths while reusing the Qwen3Next LM; tests cover config routing, weight loading, modality parity, MTP, and MMMU accuracy.
- Key code excerpts:

```diff
+@register_auto_model("Qwen3_5MoeForConditionalGeneration")
+class Qwen3_5MoeVLModel(Qwen3VLModelBase):
+    """VLM wrapper composing Qwen3 vision encoder with Qwen3.5 MoE text decoder."""
+    kwargs["vision_model_class"] = Qwen3VisionModel
```

- Reviewed files: runtime: `modeling_qwen3_5.py`, `qwen3_5_weight_mapper.py`, `modeling_speculative.py`, model loader/config utilities; tests/docs: `test_modeling_qwen3_5_vl_moe.py`, MMMU references, supported-model matrix.
- Risk and verification: benchmark rows must distinguish text-only Qwen3.5 from MoE VLM; verify mRoPE, image/video placeholders, FP8 exclude-module normalization, and MTP prompt-token recovery.

### PR #15249 - Add support for Qwen3.5 VL Dense

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/15249
- Status/date: merged / 2026-07-07
- Trace source: `git log --name-only -- <model-files>` plus the final upstream commit and PR body.
- Diff scope read: full 776-line diff, 11 files, +594/-26.
- Motivation: the MoE VLM wrapper did not cover the dense Qwen3.5-27B checkpoint, whose text decoder uses a dense `GatedMLP` and a different architecture registration.
- Key implementation: generalizes the shared Qwen3.5 VLM base, registers `Qwen3_5ForConditionalGeneration`, selects the dense decoder while keeping Qwen3 vision/mRoPE handling, and extends the same HF mapper to the dense architecture.
- Code diff details: production config normalization preserves dense `intermediate_size` and empty deepstack indexes; the parity suite covers image, multi-image, video, chunked-prefill position slicing, and model construction.
- Key code excerpts:

```diff
+@register_auto_model("Qwen3_5ForConditionalGeneration")
+class Qwen3_5VLModel(_Qwen3_5VLModel):
+    """VLM wrapper composing Qwen3 vision encoder with dense Qwen3.5 text decoder."""
```

- Reviewed files: runtime: `modeling_qwen3_5.py`, `qwen3_5_weight_mapper.py`, model/config registries; tests/docs: `test_modeling_qwen3_5_vl.py`, MMMU references, supported-model matrix.
- Risk and verification: dense and MoE checkpoints need separate accuracy/performance rows; validate composite config aliases, mRoPE chunk slicing, SSM-cache dtype, and multimodal forward parity.

### PR #16065 - [https://nvbugs/6422332][fix] Keep SSM cache in weights dtype when ma…

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/16065
- Status/date: merged / 2026-07-08
- Trace source: `git log --name-only -- <model-files>` found it through `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl.py`, `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py`; associated commits `7d7c364ae997`, `d163e74407cd`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 4 files, +36/-18, 110 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py` modified +11/-2 (13 lines); hunks: -125,15 +125,24 @@ def test_qwen35_moe_vl_resolves_mamba_ssm_cache_dtype(; symbols: test_qwen35_moe_vl_resolves_mamba_ssm_cache_dtype, test_qwen35_moe_vl_resolves_model_and_mapper, touching `test_qwen35_moe_vl_resolves_mamba_ssm_cache_dtype, test_qwen35_moe_vl_resolves_model_and_mapper`; `tensorrt_llm/_torch/pyexecutor/config_utils.py` modified +20/-13 (33 lines); hunks: -52,25 +52,32 @@ def resolve_hf_torch_dtype(config):; -307,7 +314,7 @@ def extract_mamba_kv_cache_params(; symbols: resolve_hf_torch_dtype, resolve_mamba_ssm_cache_dtype, resolve_ssm_cache_dtype, extract_mamba_kv_cache_params, touching `resolve_hf_torch_dtype, resolve_mamba_ssm_cache_dtype, resolve_ssm_cache_dtype`; `tensorrt_llm/_torch/pyexecutor/model_loader.py` modified +2/-2 (4 lines); hunks: -35,7 +35,7; -52,7 +52,7 @@ def validate_and_set_mamba_ssm_cache_dtype(; symbols: validate_and_set_mamba_ssm_cache_dtype, touching `validate_and_set_mamba_ssm_cache_dtype`.
- Code diff details:
  - `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py` modified +11/-2 (13 lines); hunks: -125,15 +125,24 @@ def test_qwen35_moe_vl_resolves_mamba_ssm_cache_dtype(; symbols: test_qwen35_moe_vl_resolves_mamba_ssm_cache_dtype, test_qwen35_moe_vl_resolves_model_and_mapper
  - `tensorrt_llm/_torch/pyexecutor/config_utils.py` modified +20/-13 (33 lines); hunks: -52,25 +52,32 @@ def resolve_hf_torch_dtype(config):; -307,7 +314,7 @@ def extract_mamba_kv_cache_params(; symbols: resolve_hf_torch_dtype, resolve_mamba_ssm_cache_dtype, resolve_ssm_cache_dtype, extract_mamba_kv_cache_params
  - `tensorrt_llm/_torch/pyexecutor/model_loader.py` modified +2/-2 (4 lines); hunks: -35,7 +35,7; -52,7 +52,7 @@ def validate_and_set_mamba_ssm_cache_dtype(; symbols: validate_and_set_mamba_ssm_cache_dtype
- Key code excerpts:

```diff
diff -- tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py
@@ -125,15 +125,24 @@ def test_qwen35_moe_vl_resolves_mamba_ssm_cache_dtype(
+    # "auto" keeps the SSM cache in the model weights dtype for performance:
+    # the checkpoint's mamba_ssm_dtype=float32 expresses SSM compute intent,
+    # and honoring it for cache allocation disables the FlashInfer bf16-state
+    # GDN decode kernel and doubles state memory traffic.
-    assert model_config.quant_config.mamba_ssm_cache_dtype is torch.float32
+    assert model_config.quant_config.mamba_ssm_cache_dtype is torch.bfloat16
diff -- tensorrt_llm/_torch/pyexecutor/config_utils.py
@@ -52,25 +52,32 @@ def resolve_hf_torch_dtype(config):
-def resolve_mamba_ssm_cache_dtype(config):
-    """Return the dtype to use for hybrid Mamba/SSM cache allocations.
-    Qwen3.5-style configs may store this field on the top-level config or the
-    nested text_config, and may call it either mamba_ssm_cache_dtype or
-    mamba_ssm_dtype. This helper centralizes that lookup so cache creation does
-    not fail later with a missing dtype. An "auto" value in any field is
diff -- tensorrt_llm/_torch/pyexecutor/model_loader.py
@@ -35,7 +35,7 @@
```

- Extracted files (not manually reviewed):
  - tests: `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py` modified +11/-2
  - runtime: `tensorrt_llm/_torch/pyexecutor/config_utils.py` modified +20/-13; `tensorrt_llm/_torch/pyexecutor/model_loader.py` modified +2/-2
- Risk and verification: The diff ships test coverage in `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #16264 - [None][fix] Align dense Qwen3.5-VL SSM cache dtype test with #16065 semantics

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/16264
- Status/date: merged / 2026-07-14
- Trace source: `git log --name-only -- <model-files>` found it through `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl.py`; associated commits `7d7c364ae997`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +11/-2, 27 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl.py` modified +11/-2 (13 lines); hunks: -143,15 +143,24 @@ def test_qwen35_dense_vl_resolves_mamba_ssm_cache_dtype(; symbols: test_qwen35_dense_vl_resolves_mamba_ssm_cache_dtype, test_qwen35_dense_vl_resolves_model_and_mapper, touching `test_qwen35_dense_vl_resolves_mamba_ssm_cache_dtype, test_qwen35_dense_vl_resolves_model_and_mapper`.
- Code diff details:
  - `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl.py` modified +11/-2 (13 lines); hunks: -143,15 +143,24 @@ def test_qwen35_dense_vl_resolves_mamba_ssm_cache_dtype(; symbols: test_qwen35_dense_vl_resolves_mamba_ssm_cache_dtype, test_qwen35_dense_vl_resolves_model_and_mapper
- Key code excerpts:

```diff
diff -- tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl.py
@@ -143,15 +143,24 @@ def test_qwen35_dense_vl_resolves_mamba_ssm_cache_dtype(
+    # "auto" keeps the SSM cache in the model weights dtype for performance:
+    # the checkpoint's mamba_ssm_dtype=float32 expresses SSM compute intent,
+    # and honoring it for cache allocation disables the FlashInfer bf16-state
+    # GDN decode kernel and doubles state memory traffic.
-    assert model_config.quant_config.mamba_ssm_cache_dtype is torch.float32
+    assert model_config.quant_config.mamba_ssm_cache_dtype is torch.bfloat16
```

- Extracted files (not manually reviewed):
  - tests: `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl.py` modified +11/-2
- Risk and verification: The diff ships test coverage in `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #16353 - [TRTLLM-14054][perf] Qwen3.5-VL: pass the inner LM's normalized model_config to the weight mapper

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/16353
- Status/date: merged / 2026-07-14
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_qwen3_5.py`; associated commits `924978f7a0d0`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +9/-1, 17 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_qwen3_5.py` modified +9/-1 (10 lines); hunks: -695,7 +695,15 @@ def load_weights(self, weights: Dict[str, torch.Tensor], we...; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_qwen3_5.py` modified +9/-1 (10 lines); hunks: -695,7 +695,15 @@ def load_weights(self, weights: Dict[str, torch.Tensor], we...; symbols: load_weights
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3_5.py
@@ -695,7 +695,15 @@ def load_weights(self, weights: Dict[str, torch.Tensor], weight_mapper: BaseWeig
-        weight_mapper.init_model_and_config(self.llm, self.model_config)
+        # Hand the mapper the inner LM's model_config, not the VLM wrapper's:
+        # only the inner config went through the Qwen3.5 quant-dict
+        # normalization applied in the inner LM's __init__ (HF->TRT-LLM key
+        # translation + synthesis of the fused in_proj_qkvz FP8 entry). With
+        # the wrapper's un-normalized copy the mapper misses that FP8 entry,
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_5.py` modified +9/-1
- Risk and verification: Runtime changes concentrate in `tensorrt_llm/_torch/models/modeling_qwen3_5.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #16469 - Fuse Qwen3.5/3.6 attention preprocessing

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/16469
- Status/date: merged / 2026-07-21
- Trace source: `git log --name-only -- <model-files>` plus the final upstream commit and PR body.
- Diff scope read: full 913-line diff, 6 files, +775/-19.
- Motivation: full-attention layers separately deinterleaved Q/gate, normalized Q and K, applied RoPE, copied V, and later launched sigmoid/multiply for the output gate.
- Key implementation: adds a Triton fast path that reads the interleaved projection once, emits packed QKV plus gate while applying Gemma RMSNorm and plain/interleaved mRoPE, and performs output gating with an in-place fused sigmoid-multiply; unsupported layouts fall back to the generic path.
- Code diff details: `QKNormRoPEAttention.preprocess_qkv` gates the fusion by dtype/layout/RoPE contract, keeping weight loading, LoRA, compilation, HIP, and unsupported scaling decoupled.
- Key code excerpts:

```diff
+qkv, gate = fused_qkv_gemma_rmsnorm_rope_gate(
+    qkv, self.q_norm.weight, self.k_norm.weight,
+    self.rotary_emb.rotary_cos_sin, positions.contiguous(), ...)
+return qkv, None, None, gate
```

- Reviewed files: runtime: `attention.py`, `qk_norm_attention.py`, `modeling_qwen3_next.py`, `fused_qk_norm_rope_gate.py`; tests: `test_fused_qk_norm_rope_gate.py`, B200 test database.
- Risk and verification: validate BF16/FP16, partial/full rotary dimensions, mRoPE sectioning, zero-token and non-contiguous cases, and compare against both the Python reference and production THOP path.

### PR #15194 - Fuse Gemma RMSNorm into AllReduce for Qwen3-Next/Qwen3.5

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/15194
- Status/date: merged / 2026-07-24
- Trace source: `git log --name-only -- <model-files>` plus the final upstream commit and PR body.
- Diff scope read: full 418-line diff, 3 files, +278/-29.
- Motivation: tensor-parallel Qwen3.5 decoder layers paid separate collective and Gemma-RMSNorm work, while the existing fused collective expected a plain norm weight rather than Gemma's `(1 + weight)` convention.
- Key implementation: enables eager fusion for non-attention-DP TP, defers the attention/MoE collective to `RESIDUAL_RMS_NORM`, precomputes the Gemma-adjusted norm weight once after loading, and keeps MTP post-MoE fusion disabled where no next-layer norm is available.
- Code diff details: a related FlashInfer GDN decode guard clones only misaligned scalar slices so the CuTe-DSL 32-byte alignment contract is satisfied without penalizing aligned shapes.
- Key code excerpts:

```diff
+norm._fused_norm_weight = (w.float() + 1.0).to(w.dtype)
+fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
+norm_weight=_fused_norm_weight(self.post_attention_layernorm),
```

- Reviewed files: runtime: `modeling_qwen3_next.py`, `fused_sigmoid_gating_recurrent.py`; tests: `test_qwen3_next_eager_fusion.py`.
- Risk and verification: confirm exactly one collective owner, attention-DP stays unfused, cached derived weights refresh after loading, Gemma numerics use `(1 + weight)`, and misaligned Qwen3.6 head slices take the guarded copy.

### PR #16936 - [None][fix] Fix Qwen3.5 weight-load memory growth and MTP CUTLASS fallback

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/16936
- Status/date: merged / 2026-07-29
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py`; associated commits `2341c704a6e8`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 4 files, +89/-9, 185 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` modified +6/-1 (7 lines); hunks: -5,6 +5,7; -556,6 +557,7 @@ def _remap_dense_mlp_weights(self, weights: dict) -> dict:; symbols: _remap_dense_mlp_weights, preprocess_weights, touching `_remap_dense_mlp_weights, preprocess_weights`.
- Code diff details:
  - `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` modified +6/-1 (7 lines); hunks: -5,6 +5,7; -556,6 +557,7 @@ def _remap_dense_mlp_weights(self, weights: dict) -> dict:; symbols: _remap_dense_mlp_weights, preprocess_weights
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py
@@ -5,6 +5,7 @@
+from tensorrt_llm._torch.models.checkpoints.base_weight_loader import ConsumableWeightsDict
@@ -556,6 +557,7 @@ def _remap_dense_mlp_weights(self, weights: dict) -> dict:
+        is_consumable = isinstance(weights, ConsumableWeightsDict)
@@ -584,4 +586,7 @@ def preprocess_weights(self, weights: dict) -> dict:
-        return super().preprocess_weights(packed_weights)
+        processed_weights = super().preprocess_weights(packed_weights)
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` modified +6/-1
- Risk and verification: Runtime changes concentrate in `tensorrt_llm/_torch/models/checkpoints/base_weight_loader.py`, `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py`, `tensorrt_llm/_torch/models/modeling_qwen3_next.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #16642 - [TRTLLM-14497][feat] Add BF16/FP8 refit for qwen3.5_397b

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/16642
- Status/date: merged / 2026-07-31
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py`, `tensorrt_llm/_torch/models/modeling_qwen3_5.py`, `tests/unittest/_torch/modeling/test_qwen3_5_partial_loading.py`; associated commits `1c797cf7ab69`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 17 files, +1062/-159, 1560 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` modified +118/-4 (122 lines); hunks: -1,3 +1,6; -62,6 +65,109 @@ class Qwen3_5MoeHfWeightMapper(Qwen3NextHfWeightMapper):; symbols: Qwen3_5MoeHfWeightMapper, __init__, begin_update_weights, finalize_update_weights, touching `Qwen3_5MoeHfWeightMapper, __init__, begin_update_weights`; `tensorrt_llm/_torch/models/modeling_qwen3_5.py` modified +20/-6 (26 lines); hunks: -695,26 +695,40 @@ def multimodal_data_device_paths(self) -> List[str]:; symbols: multimodal_data_device_paths, load_weights, touching `multimodal_data_device_paths, load_weights`; `tests/unittest/_torch/modeling/test_qwen3_5_partial_loading.py` added +114/-0 (114 lines); hunks: -0,0 +1,114; symbols: test_qwen35_vl_propagates_partial_loading_to_vision_encoder, _VisualStub, __init__, test_qwen3_vision_loader_propagates_partial_loading, touching `test_qwen35_vl_propagates_partial_loading_to_vision_encoder, _VisualStub, __init__`.
- Code diff details:
  - `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` modified +118/-4 (122 lines); hunks: -1,3 +1,6; -62,6 +65,109 @@ class Qwen3_5MoeHfWeightMapper(Qwen3NextHfWeightMapper):; symbols: Qwen3_5MoeHfWeightMapper, __init__, begin_update_weights, finalize_update_weights
  - `tensorrt_llm/_torch/models/modeling_qwen3_5.py` modified +20/-6 (26 lines); hunks: -695,26 +695,40 @@ def multimodal_data_device_paths(self) -> List[str]:; symbols: multimodal_data_device_paths, load_weights
  - `tests/unittest/_torch/modeling/test_qwen3_5_partial_loading.py` added +114/-0 (114 lines); hunks: -0,0 +1,114; symbols: test_qwen35_vl_propagates_partial_loading_to_vision_encoder, _VisualStub, __init__, test_qwen3_vision_loader_propagates_partial_loading
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py
@@ -1,3 +1,6 @@
+# SPDX-License-Identifier: Apache-2.0
+# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
@@ -62,6 +65,109 @@ class Qwen3_5MoeHfWeightMapper(Qwen3NextHfWeightMapper):
+    def __init__(self) -> None:
+        super().__init__()
+        self._partial_split_weights: dict[str, object] = {}
diff -- tensorrt_llm/_torch/models/modeling_qwen3_5.py
@@ -695,26 +695,40 @@ def multimodal_data_device_paths(self) -> List[str]:
-    def load_weights(self, weights: Dict[str, torch.Tensor], weight_mapper: BaseWeightMapper):
+    def load_weights(
+        self,
+        weights: Dict[str, torch.Tensor],
+        weight_mapper: BaseWeightMapper,
+        allow_partial_loading: bool = False,
diff -- tests/unittest/_torch/modeling/test_qwen3_5_partial_loading.py
@@ -0,0 +1,114 @@
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` modified +118/-4; `tensorrt_llm/_torch/models/modeling_qwen3_5.py` modified +20/-6
  - tests: `tests/unittest/_torch/modeling/test_qwen3_5_partial_loading.py` added +114/-0
- Risk and verification: The diff ships test coverage in `tests/integration/test_lists/test-db/l0_a10.yml`, `tests/integration/test_lists/test-db/l0_dgx_b200.yml`, `tests/unittest/_torch/modeling/test_qwen3_5_partial_loading.py`, `tests/unittest/_torch/ray_orchestrator/multi_gpu/test_llm_update_weights_multi_gpu.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17293 - [https://nvbugs/6434512][fix] Select Marlin for Qwen3.5 MoE on Hopper

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/17293
- Status/date: merged / 2026-08-10
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_qwen3_5.py`, `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py`; associated commits `ec044a2e1ac3`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +74/-2, 143 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_qwen3_5.py` modified +35/-1 (36 lines); hunks: -15,11 +15,14; -32,6 +35,7; symbols: _get_qwen35_moe_model_defaults, _translate_mtp_pattern, Qwen3_5MoeForCausalLM, that, touching `_get_qwen35_moe_model_defaults, _translate_mtp_pattern, Qwen3_5MoeForCausalLM`; `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py` modified +39/-1 (40 lines); hunks: -7,6 +7,7; -15,7 +16,7; symbols: _write_qwen35_moe_vl_config, test_qwen35_moe_vl_resolves_model_and_mapper, test_qwen35_moe_model_defaults, test_qwen35_moe_vl_placeholder_metadata_registered, touching `_write_qwen35_moe_vl_config, test_qwen35_moe_vl_resolves_model_and_mapper, test_qwen35_moe_model_defaults`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_qwen3_5.py` modified +35/-1 (36 lines); hunks: -15,11 +15,14; -32,6 +35,7; symbols: _get_qwen35_moe_model_defaults, _translate_mtp_pattern, Qwen3_5MoeForCausalLM, that
  - `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py` modified +39/-1 (40 lines); hunks: -7,6 +7,7; -15,7 +16,7; symbols: _write_qwen35_moe_vl_config, test_qwen35_moe_vl_resolves_model_and_mapper, test_qwen35_moe_model_defaults, test_qwen35_moe_vl_placeholder_metadata_registered
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3_5.py
@@ -15,11 +15,14 @@
-from typing import Dict, List, Literal
+from typing import TYPE_CHECKING, Dict, List, Literal
+if TYPE_CHECKING:
+    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
@@ -32,6 +35,7 @@
+from ..utils import is_nvfp4_marlin_supported_sm
diff -- tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py
@@ -7,6 +7,7 @@
+import pytest
@@ -15,7 +16,7 @@
-from tensorrt_llm._torch.models import Qwen3_5MoeVLModel
+from tensorrt_llm._torch.models import Qwen3_5MoeForCausalLM, Qwen3_5MoeVLModel
@@ -27,6 +28,10 @@
+from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_5.py` modified +35/-1
  - tests: `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py` modified +39/-1
- Risk and verification: The diff ships test coverage in `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17433 - [None][fix] Qwen3.5 weight mapper for FP8 per-channel checkpoints

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/17433
- Status/date: merged / 2026-08-11
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py`, `tests/unittest/_torch/models/checkpoints/hf/test_qwen3_5_weight_mapper.py`; associated commits `28e03befe6c2`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +155/-23, 216 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/unittest/_torch/models/checkpoints/hf/test_qwen3_5_weight_mapper.py` added +141/-0 (141 lines); hunks: -0,0 +1,141; symbols: _make_mapper, _fp8, _scale, _bf16, touching `_make_mapper, _fp8, _scale`; `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` modified +14/-23 (37 lines); hunks: -178,24 +178,19 @@ def _normalize_weight_names(self, weights: dict) -> dict:; -204,12 +199,6 @@ def _normalize_scale_names(self, weights: dict, quant_algo)...; symbols: _normalize_weight_names, _normalize_scale_names, _normalize_fp8_block_scale_names, preprocess_weights, touching `_normalize_weight_names, _normalize_scale_names, _normalize_fp8_block_scale_names`.
- Code diff details:
  - `tests/unittest/_torch/models/checkpoints/hf/test_qwen3_5_weight_mapper.py` added +141/-0 (141 lines); hunks: -0,0 +1,141; symbols: _make_mapper, _fp8, _scale, _bf16
  - `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` modified +14/-23 (37 lines); hunks: -178,24 +178,19 @@ def _normalize_weight_names(self, weights: dict) -> dict:; -204,12 +199,6 @@ def _normalize_scale_names(self, weights: dict, quant_algo)...; symbols: _normalize_weight_names, _normalize_scale_names, _normalize_fp8_block_scale_names, preprocess_weights
- Key code excerpts:

```diff
diff -- tests/unittest/_torch/models/checkpoints/hf/test_qwen3_5_weight_mapper.py
@@ -0,0 +1,141 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
diff -- tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py
@@ -178,24 +178,19 @@ def _normalize_weight_names(self, weights: dict) -> dict:
-    def _normalize_scale_names(self, weights: dict, quant_algo) -> tuple[dict, bool]:
-        # Canonicalize FP8 weight_scale layout so the Linear loader sees one
-        # shape per quant algo:
-        #   - FP8_BLOCK_SCALES: modelopt fp8_pb_wo stores weight_scale shaped
-        #     [blocks_out, 1, blocks_in, 1]; squeeze to [blocks_out, blocks_in]
-        #     and rename to weight_scale_inv. Returns is_modelopt_pb_wo=True
```

- Extracted files (not manually reviewed):
  - tests: `tests/unittest/_torch/models/checkpoints/hf/test_qwen3_5_weight_mapper.py` added +141/-0
  - runtime: `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` modified +14/-23
- Risk and verification: The diff ships test coverage in `tests/unittest/_torch/models/checkpoints/hf/test_qwen3_5_weight_mapper.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17700 - [None][perf] Qwen3.5/3.8 wave-2: MoE, attention-DP, GDN replay, weight loading

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/17700
- Status/date: merged / 2026-08-21
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_qwen3_5.py`, `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py`; associated commits `2f17320eb253`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 15 files, +875/-71, 1257 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_qwen3_5.py` modified +14/-1 (15 lines); hunks: -36,6 +36,7; -80,6 +81,18 @@ def _get_qwen35_moe_model_defaults(llm_args: "TorchLlmArgs")...; symbols: _get_qwen35_moe_model_defaults, _filter_language_model_weights, _translate_mtp_pattern, load_weights, touching `_get_qwen35_moe_model_defaults, _filter_language_model_weights, _translate_mtp_pattern`; `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py` modified +22/-1 (23 lines); hunks: -18,9 +18,13; -198,6 +202,23 @@ def test_qwen35_moe_model_defaults(; symbols: test_qwen35_moe_model_defaults, test_qwen35_vl_filter_preserves_consumable_weights, test_qwen35_moe_vl_placeholder_metadata_registered, touching `test_qwen35_moe_model_defaults, test_qwen35_vl_filter_preserves_consumable_weights, test_qwen35_moe_vl_placeholder_metadata_registered`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_qwen3_5.py` modified +14/-1 (15 lines); hunks: -36,6 +36,7; -80,6 +81,18 @@ def _get_qwen35_moe_model_defaults(llm_args: "TorchLlmArgs")...; symbols: _get_qwen35_moe_model_defaults, _filter_language_model_weights, _translate_mtp_pattern, load_weights
  - `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py` modified +22/-1 (23 lines); hunks: -18,9 +18,13; -198,6 +202,23 @@ def test_qwen35_moe_model_defaults(; symbols: test_qwen35_moe_model_defaults, test_qwen35_vl_filter_preserves_consumable_weights, test_qwen35_moe_vl_placeholder_metadata_registered
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_qwen3_5.py
@@ -36,6 +36,7 @@
+from .checkpoints.base_weight_loader import ConsumableWeightsDict
@@ -80,6 +81,18 @@ def _get_qwen35_moe_model_defaults(llm_args: "TorchLlmArgs") -> dict:
+def _filter_language_model_weights(weights: Dict[str, torch.Tensor]):
+    """Drop vision weights without disabling incremental weight consumption.
+    Ownership: a ConsumableWeightsDict input is emptied, since the returned
+    mapping aliases its tensors. The caller must use only the return value.
diff -- tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py
@@ -18,9 +18,13 @@
+from tensorrt_llm._torch.models.checkpoints.base_weight_loader import ConsumableWeightsDict
-from tensorrt_llm._torch.models.modeling_qwen3_5 import _normalize_qwen35_moe_vl_config
+from tensorrt_llm._torch.models.modeling_qwen3_5 import (
+    _filter_language_model_weights,
+    _normalize_qwen35_moe_vl_config,
+)
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_qwen3_5.py` modified +14/-1
  - tests: `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py` modified +22/-1
- Risk and verification: The diff ships test coverage in `tests/unittest/_torch/executor/test_py_executor.py`, `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py`, `tests/unittest/_torch/models/checkpoints/test_consumable_weights_dict.py`, `tests/unittest/_torch/modules/fused_moe/test_deepgemm_fused_expand_quant.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #19519 - [https://nvbugs/6771102][fix] Support Qwen3.5 global FP8 checkpoints

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/19519
- Status/date: merged / 2026-09-23
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py`, `tests/unittest/_torch/models/checkpoints/hf/test_qwen3_5_weight_mapper.py`; associated commits `ef9a3340a314`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +115/-19, 182 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/unittest/_torch/models/checkpoints/hf/test_qwen3_5_weight_mapper.py` modified +92/-2 (94 lines); hunks: -19,6 +19,7; -45,10 +46,12; symbols: _make_mapper, test_fp8_rowwise_full, test_modelopt_fp8_per_tensor_linear_attention, test_modelopt_fp8_excluded_linear_attention_falls_back_to_bf16, touching `_make_mapper, test_fp8_rowwise_full, test_modelopt_fp8_per_tensor_linear_attention`; `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` modified +23/-17 (40 lines); hunks: -47,7 +47,7 @@ class Qwen3_5MoeHfWeightMapper(Qwen3NextHfWeightMapper):; -339,13 +339,13 @@ def _dequantize_linear_attn_fp8_qkvz(self, weights: dict)...; symbols: Qwen3_5MoeHfWeightMapper, _dequantize_linear_attn_fp8_qkvz, _requantize_linear_attn_fp8_qkvz, preprocess_weights, touching `Qwen3_5MoeHfWeightMapper, _dequantize_linear_attn_fp8_qkvz, _requantize_linear_attn_fp8_qkvz`.
- Code diff details:
  - `tests/unittest/_torch/models/checkpoints/hf/test_qwen3_5_weight_mapper.py` modified +92/-2 (94 lines); hunks: -19,6 +19,7; -45,10 +46,12; symbols: _make_mapper, test_fp8_rowwise_full, test_modelopt_fp8_per_tensor_linear_attention, test_modelopt_fp8_excluded_linear_attention_falls_back_to_bf16
  - `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` modified +23/-17 (40 lines); hunks: -47,7 +47,7 @@ class Qwen3_5MoeHfWeightMapper(Qwen3NextHfWeightMapper):; -339,13 +339,13 @@ def _dequantize_linear_attn_fp8_qkvz(self, weights: dict)...; symbols: Qwen3_5MoeHfWeightMapper, _dequantize_linear_attn_fp8_qkvz, _requantize_linear_attn_fp8_qkvz, preprocess_weights
- Key code excerpts:

```diff
diff -- tests/unittest/_torch/models/checkpoints/hf/test_qwen3_5_weight_mapper.py
@@ -19,6 +19,7 @@
+from tensorrt_llm.models.modeling_utils import QuantConfig
@@ -45,10 +46,12 @@
-def _make_mapper(quant_algo: QuantAlgo) -> Qwen3_5MoeHfWeightMapper:
+def _make_mapper(
+    quant_algo: QuantAlgo, exclude_modules: list[str] | None = None
+) -> Qwen3_5MoeHfWeightMapper:
diff -- tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py
@@ -47,7 +47,7 @@ class Qwen3_5MoeHfWeightMapper(Qwen3NextHfWeightMapper):
-       checkpoints with per-tensor FP8 in_proj, the projections are
+       checkpoints and global FP8 checkpoints with per-tensor FP8 in_proj, the projections are
@@ -339,13 +339,13 @@ def _dequantize_linear_attn_fp8_qkvz(self, weights: dict) -> dict:
-        When _normalize_qwen35_quant_config_dict synthesized an FP8 entry for
-        the fused ``in_proj_qkvz`` module (i.e. the checkpoint quantizes every
-        split projection per-tensor FP8), keep the weights FP8 instead of
```

- Extracted files (not manually reviewed):
  - tests: `tests/unittest/_torch/models/checkpoints/hf/test_qwen3_5_weight_mapper.py` modified +92/-2
  - runtime: `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` modified +23/-17
- Risk and verification: The diff ships test coverage in `tests/unittest/_torch/models/checkpoints/hf/test_qwen3_5_weight_mapper.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
