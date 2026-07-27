# TensorRT-LLM Qwen3.5 Model PR Optimization History

## 2026-07-28 Source Head Refresh

Rechecked TensorRT-LLM upstream main at `NVIDIA/TensorRT-LLM@9fe5853263750ade5b7dc24fb31a1215ec822d45`.
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

## PR Coverage Summary

- Reviewed PRs: 14
- File trace command: `git log --name-only -- <model-files>`
- Diff source: `gh pr diff` / GitHub Pull Request files API patches cached under `/tmp/model_pr_diffs/tensorrt_llm/pr*.diff`
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
