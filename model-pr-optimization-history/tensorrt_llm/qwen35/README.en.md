# TensorRT-LLM Qwen3.5 Model PR Optimization History

## 2026-06-26 PR Backfill Audit

Checked against TensorRT-LLM upstream `HEAD@4164b932c6c8a14d1be85d0fd62e44b7d0171980`. This page provides model implementation coverage, a PR timeline, and per-PR diff audit cards.

Filter used in this pass: merged PRs whose titles or files matched `Qwen3.5`, `Qwen3_5`, `qwen3_5`, `AutoDeploy`, `NVFP4`, `FP8`, `DFlash`, `reasoning_parser`, `EPLB`, `MoE backend`, or `model_registry`. Pure reshuffling and unrelated infrastructure PRs were excluded.

## Model Implementation File Coverage

| File | Related PRs |
| --- | --- |
| `tensorrt_llm/_torch/models/modeling_qwen3_5.py` | [#12302](https://github.com/NVIDIA/TensorRT-LLM/pull/12302), [#15067](https://github.com/NVIDIA/TensorRT-LLM/pull/15067) |
| `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` | [#12302](https://github.com/NVIDIA/TensorRT-LLM/pull/12302), [#13090](https://github.com/NVIDIA/TensorRT-LLM/pull/13090), [#13716](https://github.com/NVIDIA/TensorRT-LLM/pull/13716), [#15067](https://github.com/NVIDIA/TensorRT-LLM/pull/15067) |
| `tensorrt_llm/_torch/auto_deploy/models/custom/modeling_qwen3_5_moe.py` | [#12114](https://github.com/NVIDIA/TensorRT-LLM/pull/12114), [#14667](https://github.com/NVIDIA/TensorRT-LLM/pull/14667), [#15185](https://github.com/NVIDIA/TensorRT-LLM/pull/15185) |
| `examples/auto_deploy/model_registry/configs/qwen3.5_moe_*.yaml` | [#12114](https://github.com/NVIDIA/TensorRT-LLM/pull/12114), [#14667](https://github.com/NVIDIA/TensorRT-LLM/pull/14667), [#15185](https://github.com/NVIDIA/TensorRT-LLM/pull/15185) |
| `examples/auto_deploy/model_registry/models.yaml` | [#12114](https://github.com/NVIDIA/TensorRT-LLM/pull/12114), [#15001](https://github.com/NVIDIA/TensorRT-LLM/pull/15001) |
| `tensorrt_llm/_torch/auto_deploy/transform/library/mrope_delta_cache.py` | [#12114](https://github.com/NVIDIA/TensorRT-LLM/pull/12114) |
| `tensorrt_llm/_torch/models/modeling_speculative.py` / `speculative/dflash.py` | [#13782](https://github.com/NVIDIA/TensorRT-LLM/pull/13782), [#13996](https://github.com/NVIDIA/TensorRT-LLM/pull/13996) |
| `tensorrt_llm/llmapi/reasoning_parser.py` | [#14659](https://github.com/NVIDIA/TensorRT-LLM/pull/14659) |
| `tensorrt_llm/_torch/modules/fused_moe/moe_load_balancer.py` | [#15543](https://github.com/NVIDIA/TensorRT-LLM/pull/15543) |
| `tests/integration/defs/accuracy/test_llm_api_pytorch.py` | [#12302](https://github.com/NVIDIA/TensorRT-LLM/pull/12302), [#13090](https://github.com/NVIDIA/TensorRT-LLM/pull/13090), [#15081](https://github.com/NVIDIA/TensorRT-LLM/pull/15081), [#15543](https://github.com/NVIDIA/TensorRT-LLM/pull/15543) |

## PR Coverage Overview

- Reviewed PRs: 14
- Diff source: `gh pr diff` / GitHub PR patches cached under `/tmp/model_pr_diffs/tensorrt_llm/pr*.diff`
- Reviewed patch lines: 12,514
- Main TensorRT-LLM Qwen3.5 themes: AutoDeploy cookbook/registry, mRoPE/3D positions, NVFP4/FP8 weight mapping, dense/MoE wrappers, DFlash speculative decoding, reasoning parser, CUTLASS/DeepGEMM backend selection, and EPLB.

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-02-26 | [#11728](https://github.com/NVIDIA/TensorRT-LLM/pull/11728) | merged | Added Qwen3.5 Cookbook | AutoDeploy cookbook notebook |
| 2026-03-24 | [#12302](https://github.com/NVIDIA/TensorRT-LLM/pull/12302) | merged | Add Qwen 3.5 supporting (NVFP4) | model wrapper, weight mapper, tests |
| 2026-03-25 | [#12114](https://github.com/NVIDIA/TensorRT-LLM/pull/12114) | merged | Qwen 3.5 fix 3d position ID handling | AutoDeploy Qwen3.5 MoE, mRoPE cache, registry configs |
| 2026-04-30 | [#13090](https://github.com/NVIDIA/TensorRT-LLM/pull/13090) | merged | Qwen3.5 dense weight loading | weight mapper, dense tests |
| 2026-05-04 | [#13716](https://github.com/NVIDIA/TensorRT-LLM/pull/13716) | merged | Fix Qwen3.5 NVFP4 weight loading by preserving weight_scales | HF mapper |
| 2026-05-12 | [#13782](https://github.com/NVIDIA/TensorRT-LLM/pull/13782) | merged | Qwen3.5 DFlash | speculative/DFlash runtime |
| 2026-05-16 | [#13996](https://github.com/NVIDIA/TensorRT-LLM/pull/13996) | merged | Perf optimizations for DFlash | DFlash model engine and speculative code |
| 2026-05-29 | [#14659](https://github.com/NVIDIA/TensorRT-LLM/pull/14659) | merged | Add a reasoning parser for qwen3_5 | `reasoning_parser.py` |
| 2026-06-02 | [#14667](https://github.com/NVIDIA/TensorRT-LLM/pull/14667) | merged | AutoDeploy: Qwen3.5 400B NVFP4 accuracy regression fix | shared expert sharding, SwiGLU fusion |
| 2026-06-05 | [#15001](https://github.com/NVIDIA/TensorRT-LLM/pull/15001) | merged | Uncomment Qwen3.5 from model registry | `models.yaml` |
| 2026-06-09 | [#15081](https://github.com/NVIDIA/TensorRT-LLM/pull/15081) | merged | Select CUTLASS MoE backend on non-Blackwell SMs | accuracy test backend selection |
| 2026-06-11 | [#15067](https://github.com/NVIDIA/TensorRT-LLM/pull/15067) | merged | Generalize FP8 checkpoint loading for Qwen3.5 | weight mapper, modeling |
| 2026-06-13 | [#15185](https://github.com/NVIDIA/TensorRT-LLM/pull/15185) | merged | Qwen3.5 whitelist sharding and lm_head sharding | AutoDeploy sharding IR/tests |
| 2026-06-26 | [#15543](https://github.com/NVIDIA/TensorRT-LLM/pull/15543) | merged | Add EPLB support for Qwen3.5 | MoE load balancer and B200/GB200 tests |

## Per-PR Diff Audit Cards

### PR #11728 - Added Qwen3.5 Cookbook

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/11728
- State/time: merged / 2026-02-26
- Diff coverage: 1 file, +385/-0, 402 cached patch lines.
- Motivation: document how to deploy Qwen3.5-397B and its NVFP4 checkpoint with AutoDeploy.
- Key implementation: adds a notebook with `trtllm-serve`, AutoDeploy registry config, B200 sizing, and sample OpenAI calls.
- Code excerpt:

```diff
+trtllm-serve "nvidia/Qwen3.5-397B-A17B-NVFP4" \
+MODEL_ID = "Qwen/Qwen3.5-397B-A17B"
```

- Reviewed files: `examples/auto_deploy/cookbooks/qwen_3.5_trtllm_cookbook.ipynb`
- Validation/risk: use this as deployment evidence, not as proof that the PyTorch backend path is identical to SGLang.

### PR #12302 - Add Qwen 3.5 supporting (NVFP4)

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/12302
- State/time: merged / 2026-03-24
- Diff coverage: 9 files, +225/-31, 436 cached patch lines.
- Motivation: support Qwen3.5 dense/MoE and the official NVFP4 checkpoint in the PyTorch backend.
- Key implementation: registers dense and MoE Qwen3.5 model wrappers, extends the HF mapper, and adds 397B NVFP4 accuracy tests.
- Code excerpt:

```diff
+@register_auto_model("Qwen3_5ForCausalLM")
+class Qwen3_5ForCausalLM(Qwen3NextForCausalLM):
```

- Reviewed files: `modeling_qwen3_5.py`, `qwen3_5_weight_mapper.py`, `config_utils.py`, accuracy refs/tests
- Validation/risk: separate dense and MoE wrapper behavior in comparisons.

### PR #12114 - Qwen 3.5 fix 3D position ID handling

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/12114
- State/time: merged / 2026-03-25
- Diff coverage: 15 files, +3448/-275, 7,822 cached patch lines.
- Motivation: Qwen3.5 VLM/mRoPE needed 3D positions, chunked multimodal positions, video grid normalization, and mRoPE delta cache.
- Key implementation: extends AutoDeploy Qwen3.5 MoE modeling, mRoPE cache transforms, registry configs, and unit tests.
- Code excerpt:

```diff
+@TransformRegistry.register("initialize_mrope_delta_cache")
+mm_token_positions: torch.Tensor
```

- Reviewed files: `modeling_qwen3_5_moe.py`, `mrope_delta_cache.py`, registry YAMLs, `test_qwen3_5_moe.py`, serving utils tests
- Validation/risk: multimodal correctness depends on position construction and cache resources, not only decode kernels.

### PR #13090 - Qwen3.5 dense weight loading

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/13090
- State/time: merged / 2026-04-30
- Diff coverage: 5 files, +85/-1, 225 cached patch lines.
- Motivation: dense Qwen3.5 4B/FP8 loading needed direct coverage.
- Key implementation: updates the Qwen3.5 HF mapper and adds dense accuracy refs/tests.
- Code excerpt:

```diff
+class TestQwen3_5_4B(LlmapiAccuracyTestHarness):
+MODEL_NAME = "Qwen/Qwen3.5-4B"
```

- Reviewed files: HF mapper, accuracy refs, `test_llm_api_pytorch.py`, test lists
- Validation/risk: dense Qwen3.5 has different loading risks from 397B MoE.

### PR #13716 - Preserve Qwen3.5 NVFP4 weight_scales

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/13716
- State/time: merged / 2026-05-04
- Diff coverage: 1 file, +9/-3, 45 cached patch lines.
- Motivation: FP8 scale remapping broke NVFP4 weight scale loading.
- Key implementation: detects NVFP4 prefixes and preserves `weight_scales`.
- Code excerpt:

```diff
+        nvfp4_prefixes = {
+            key[: -len(".weight_scale_2")] for key in weights if key.endswith(".weight_scale_2")
+        }
+                if prefix not in nvfp4_prefixes:
```

- Reviewed files: `qwen3_5_weight_mapper.py`
- Validation/risk: scale key remapping is a first check for NVFP4 loading or accuracy issues.

### PR #13782 - Qwen3.5 DFlash

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/13782
- State/time: merged / 2026-05-12
- Diff coverage: 5 files, +144/-55, 413 cached patch lines.
- Motivation: enable Qwen3.5 hybrid linear-attention models on DFlash/speculative paths.
- Key implementation: wires GDN/Mamba cache and model engine paths into DFlash runtime.
- Code excerpt:

```diff
+from tensorrt_llm._torch.speculative import dflash
+mamba_cache_manager
```

- Reviewed files: `gdn_mixer.py`, `pyexecutor/_util.py`, `mamba_cache_manager.py`, `model_engine.py`, `speculative/dflash.py`
- Validation/risk: keep DFlash separate from plain decoding and SGLang MTP comparisons.

### PR #13996 - Perf optimizations for DFlash

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/13996
- State/time: merged / 2026-05-16
- Diff coverage: 5 files, +455/-285, 1,606 cached patch lines.
- Motivation: reduce DFlash overhead after the initial Qwen3.5 support.
- Key implementation: changes speculative modeling, GDN mixer, model engine, DFlash runtime, and `llm_args.py`.
- Code excerpt:

```diff
+    def _build_fused_kv_buffers(self) -> None:
+        """Stack per-layer KV projection + k_norm weights for a single fused GEMM.
+        return self.max_draft_len + 1
```

- Reviewed files: `modeling_speculative.py`, `gdn_mixer.py`, `model_engine.py`, `speculative/dflash.py`, `llm_args.py`
- Validation/risk: if TensorRT-LLM leads through DFlash, attribute the gap to speculative runtime rather than one kernel.

### PR #14659 - Add a reasoning parser for qwen3_5

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/14659
- State/time: merged / 2026-05-29
- Diff coverage: 1 file, +9/-0, 30 cached patch lines.
- Motivation: Qwen3.5 forced-thinking output begins inside the reasoning block.
- Key implementation: registers `qwen3_5` with `reasoning_at_start=True`.
- Code excerpt:

```diff
+@register_reasoning_parser("qwen3_5", reasoning_at_start=True)
```

- Reviewed files: `llmapi/reasoning_parser.py`
- Validation/risk: output parsing can change benchmark scores independently of runtime speed.

### PR #14667 - AutoDeploy Qwen3.5 400B NVFP4 accuracy regression fix

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/14667
- State/time: merged / 2026-06-02
- Diff coverage: 5 files, +72/-35, 464 cached patch lines.
- Motivation: fix a Qwen3.5 400B NVFP4 AutoDeploy accuracy regression.
- Key implementation: replicates the shared expert instead of TP-sharding it and expands SwiGLU fusion/sharding hints.
- Code excerpt:

```diff
+# The shared expert is replicated
+apply_sharding_hints:
```

- Reviewed files: `qwen3.5_moe_400b.yaml`, `modeling_qwen3_5_moe.py`, `swiglu.py`, `fuse_swiglu.py`, waives
- Validation/risk: inspect shared expert sharding and SwiGLU fusion before blaming MoE GEMMs.

### PR #15001 - Uncomment Qwen3.5 from model registry

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/15001
- State/time: merged / 2026-06-05
- Diff coverage: 1 file, +9/-12, 50 cached patch lines.
- Motivation: make Qwen3.5 AutoDeploy entries discoverable by default.
- Key implementation: enables Qwen3.5 35B and 397B entries in `models.yaml`.
- Code excerpt:

```diff
+- name: Qwen/Qwen3.5-397B-A17B
+  config_id: qwen3_5_moe_400b
```

- Reviewed files: `examples/auto_deploy/model_registry/models.yaml`
- Validation/risk: registry entries are official deployment lanes for fair comparison.

### PR #15081 - Select CUTLASS MoE backend on non-Blackwell SMs

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/15081
- State/time: merged / 2026-06-09
- Diff coverage: 2 files, +8/-2, 52 cached patch lines.
- Motivation: DeepGEMM should be used on Blackwell, while non-Blackwell tests need CUTLASS.
- Key implementation: picks the MoE backend by SM version in Qwen3.5 FP8 tests.
- Code excerpt:

```diff
+moe_backend = "DEEPGEMM" if get_sm_version() in (100, 103) else "CUTLASS"
```

- Reviewed files: `test_llm_api_pytorch.py`, `waives.txt`
- Validation/risk: never mix H100 and B200 MoE backend results without recording backend choice.

### PR #15067 - Generalize FP8 checkpoint loading for Qwen3.5

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/15067
- State/time: merged / 2026-06-11
- Diff coverage: 2 files, +68/-48, 220 cached patch lines.
- Motivation: make FP8 checkpoint loading handle Qwen3.5 naming and exclude-module variants.
- Key implementation: refactors mapper/modeling normalization around FP8/NVFP4.
- Code excerpt:

```diff
+    # gdn_mixer uses Linear module for weight management of depthwise conv1d
+    # but conv1d is not a proper linear module and should be excluded from quant
+    normalized.add("*linear_attn.conv1d")
```

- Reviewed files: `qwen3_5_weight_mapper.py`, `modeling_qwen3_5.py`
- Validation/risk: check mapper normalization before kernel-level debugging for FP8 loading issues.

### PR #15185 - Qwen3.5 whitelist sharding and lm_head sharding

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/15185
- State/time: merged / 2026-06-13
- Diff coverage: 5 files, +193/-118, 735 cached patch lines.
- Motivation: AutoDeploy needed whitelist sharding and `lm_head` sharding for Qwen3.5.
- Key implementation: updates registry configs, model sharding hints, SwiGLU fusion, and sharding IR tests.
- Code excerpt:

```diff
+lm_head:
+apply_sharding_hints
```

- Reviewed files: registry YAML, `modeling_qwen3_5_moe.py`, `fuse_swiglu.py`, `sharding_ir.py`, tests
- Validation/risk: inspect `lm_head` and shared-expert sharding separately from expert GEMMs.

### PR #15543 - Add EPLB support for Qwen3.5

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/15543
- State/time: merged / 2026-06-26
- Diff coverage: 3 files, +73/-0, 130 cached patch lines.
- Motivation: add EPLB coverage for Qwen3.5 MoE on B200/GB200 test lanes.
- Key implementation: extends the MoE load balancer and test DB entries.
- Code excerpt:

```diff
+    'Qwen2MoeForCausalLM',
+    'Qwen3MoeForCausalLM',
+    'Qwen3_5MoeForCausalLM',
```

- Reviewed files: `moe_load_balancer.py`, `test_llm_api_pytorch.py`, B200/GB200 test DB YAMLs
- Validation/risk: record whether load balancing is enabled when comparing SGLang EP/EPLB behavior.
