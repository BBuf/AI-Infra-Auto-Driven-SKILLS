# vllm Qwen3.5 Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `examples/pooling/score/colqwen3_5_rerank_online.py` | [#36887](https://github.com/vllm-project/vllm/pull/36887), [#46108](https://github.com/vllm-project/vllm/pull/46108) |
| `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-DEP2.yaml` | [#38083](https://github.com/vllm-project/vllm/pull/38083), [#45002](https://github.com/vllm-project/vllm/pull/45002) |
| `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-FP8-DEP2.yaml` | [#38083](https://github.com/vllm-project/vllm/pull/38083) |
| `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-MXFP4-AITER-TP2.yaml` | [#46520](https://github.com/vllm-project/vllm/pull/46520) |
| `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-MXFP4-EMU-TP2.yaml` | no direct PR-number commit |
| `tests/evals/gsm8k/configs/Qwen3.5-397B-A17B-NVFP4-DEP2-MTP.yaml` | [#44700](https://github.com/vllm-project/vllm/pull/44700) |
| `tests/evals/gsm8k/configs/Qwen3.5-397B-A17B-NVFP4-DEP2.yaml` | [#38083](https://github.com/vllm-project/vllm/pull/38083), [#38632](https://github.com/vllm-project/vllm/pull/38632) |
| `tests/evals/gsm8k/configs/models-qwen35-blackwell.txt` | [#38083](https://github.com/vllm-project/vllm/pull/38083), [#44700](https://github.com/vllm-project/vllm/pull/44700) |
| `tests/evals/gsm8k/configs/models-qwen35-mi355.txt` | [#38155](https://github.com/vllm-project/vllm/pull/38155), [#38664](https://github.com/vllm-project/vllm/pull/38664) |
| `tests/evals/mrcr/configs/Qwen3.5-4B.yaml` | no direct PR-number commit |
| `tests/lora/test_qwen35_densemodel_lora.py` | [#37816](https://github.com/vllm-project/vllm/pull/37816) |
| `tests/model_executor/test_qwen3_5_quantization.py` | no direct PR-number commit |
| `tests/models/multimodal/pooling/test_colqwen3_5.py` | [#36887](https://github.com/vllm-project/vllm/pull/36887), [#46108](https://github.com/vllm-project/vllm/pull/46108) |
| `vllm/model_executor/models/colqwen3_5.py` | [#36887](https://github.com/vllm-project/vllm/pull/36887), [#46108](https://github.com/vllm-project/vllm/pull/46108) |
| `vllm/model_executor/models/qwen3_5.py` | [#34110](https://github.com/vllm-project/vllm/pull/34110), [#34198](https://github.com/vllm-project/vllm/pull/34198), [#34200](https://github.com/vllm-project/vllm/pull/34200), [#34313](https://github.com/vllm-project/vllm/pull/34313), [#34489](https://github.com/vllm-project/vllm/pull/34489), [#34492](https://github.com/vllm-project/vllm/pull/34492), [#34512](https://github.com/vllm-project/vllm/pull/34512), [#34683](https://github.com/vllm-project/vllm/pull/34683), [#34697](https://github.com/vllm-project/vllm/pull/34697), [#34719](https://github.com/vllm-project/vllm/pull/34719), [#34723](https://github.com/vllm-project/vllm/pull/34723), [#35617](https://github.com/vllm-project/vllm/pull/35617), ... (23 total) |
| `vllm/model_executor/models/qwen3_5_mtp.py` | [#34110](https://github.com/vllm-project/vllm/pull/34110), [#34512](https://github.com/vllm-project/vllm/pull/34512), [#35581](https://github.com/vllm-project/vllm/pull/35581), [#37114](https://github.com/vllm-project/vllm/pull/37114), [#38832](https://github.com/vllm-project/vllm/pull/38832), [#42716](https://github.com/vllm-project/vllm/pull/42716), [#45002](https://github.com/vllm-project/vllm/pull/45002) |
| `vllm/transformers_utils/configs/qwen3_5.py` | [#34512](https://github.com/vllm-project/vllm/pull/34512), [#34554](https://github.com/vllm-project/vllm/pull/34554), [#34604](https://github.com/vllm-project/vllm/pull/34604), [#34610](https://github.com/vllm-project/vllm/pull/34610) |
| `vllm/transformers_utils/configs/qwen3_5_moe.py` | [#34512](https://github.com/vllm-project/vllm/pull/34512), [#34554](https://github.com/vllm-project/vllm/pull/34554), [#34604](https://github.com/vllm-project/vllm/pull/34604), [#34610](https://github.com/vllm-project/vllm/pull/34610) |

## PR Coverage Summary

- Git-traced PRs: 2
- Extra PRs preserved from existing docs: 11
- Total PRs in this document: 8
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2026-01-07 | [#31104](https://github.com/vllm-project/vllm/pull/31104) | merged | [BugFix] LoRA: Support loading base_layer of experts | `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/llama4.py` |
| 2026-05-13 | [#42151](https://github.com/vllm-project/vllm/pull/42151) | merged | [MM][Perf][CG] Support ViT full CUDA graph for Qwen3.5 | `vllm/model_executor/models/qwen3_5.py` |
| 2026-05-17 | [#42716](https://github.com/vllm-project/vllm/pull/42716) | merged | Fix Weight loading for Qwen3.5-MTP and Qwen3-VL using runai_streamer | `vllm/model_executor/models/qwen3_5_mtp.py` |
| 2026-05-18 | [#41436](https://github.com/vllm-project/vllm/pull/41436) | merged | [ROCm][Quantization][3/N] Refactor quark_moe w4a4 w/ oracle | `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`, `vllm/model_executor/layers/fused_moe/experts/rocm_aiter_moe.py` |
| 2026-05-22 | [#41126](https://github.com/vllm-project/vllm/pull/41126) | merged | [Attention] Mamba attention module refactor | `vllm/model_executor/models/olmo_hybrid.py`, `vllm/model_executor/layers/mamba/gdn/olmo_gdn_linear_attn.py`, `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py` |
| 2026-05-26 | [#42124](https://github.com/vllm-project/vllm/pull/42124) | merged | Add LM head quantization support for ModelOpt | `tests/model_executor/test_qwen3_5_quantization.py`, `tests/model_executor/test_nemotron_h_quantization.py`, `vllm/model_executor/layers/quantization/modelopt.py` |
| 2026-06-10 | [#39419](https://github.com/vllm-project/vllm/pull/39419) | merged | [SpecDecode] Reduce TP communication for large-vocab draft models speculative decoding | `vllm/model_executor/models/interfaces.py`, `vllm/model_executor/models/llama4_eagle.py`, `vllm/model_executor/models/qwen3.py` |
| 2026-06-11 | [#45161](https://github.com/vllm-project/vllm/pull/45161) | merged | Deprecate Transformers v4 support | `vllm/model_executor/models/transformers/base.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/model_loader/weight_utils.py` |

## Per-PR Diff Audit Cards

### PR #31104 - [BugFix] LoRA: Support loading base_layer of experts

- Link: https://github.com/vllm-project/vllm/pull/31104
- Status/date: merged / 2026-01-07
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 35 files, +46/-3, 319 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BugFix] LoRA: Support loading base_layer of experts"; model line: Qwen3.5; category: bug fix; main diff: `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/llama4.py`; technical summary: Covers "[BugFix] LoRA: Support loading base_layer of experts"; the main implementation surface is `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/llama4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/fused_moe/layer.py` modified +10/-3 (13 lines); hunks: -2007,6 +2007,7 @@ def combine_output(states: torch.Tensor) -> torch.Tensor:; -2025,13 +2026,19 @@ def make_expert_params_mapping(; symbols: combine_output, make_expert_params_mapping, touching `combine_output, make_expert_params_mapping`; `vllm/model_executor/models/deepseek_v2.py` modified +2/-0 (2 lines); hunks: -1486,6 +1486,7 @@ def get_expert_mapping(self) -> list[tuple[str, str, int,...; -1519,6 +1520,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: get_expert_mapping, load_weights, touching `get_expert_mapping, load_weights`; `vllm/model_executor/models/llama4.py` modified +2/-0 (2 lines); hunks: -539,6 +539,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; -548,6 +549,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights, touching `load_weights`; `vllm/model_executor/models/afmoe.py` modified +1/-0 (1 lines); hunks: -475,6 +475,7 @@ def get_expert_mapping(self) -> list[tuple[str, str, int, st...; symbols: get_expert_mapping, touching `get_expert_mapping`.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/layer.py` modified +10/-3 (13 lines); hunks: -2007,6 +2007,7 @@ def combine_output(states: torch.Tensor) -> torch.Tensor:; -2025,13 +2026,19 @@ def make_expert_params_mapping(; symbols: combine_output, make_expert_params_mapping
  - `vllm/model_executor/models/deepseek_v2.py` modified +2/-0 (2 lines); hunks: -1486,6 +1486,7 @@ def get_expert_mapping(self) -> list[tuple[str, str, int,...; -1519,6 +1520,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: get_expert_mapping, load_weights
  - `vllm/model_executor/models/llama4.py` modified +2/-0 (2 lines); hunks: -539,6 +539,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; -548,6 +549,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights
  - `vllm/model_executor/models/afmoe.py` modified +1/-0 (1 lines); hunks: -475,6 +475,7 @@ def get_expert_mapping(self) -> list[tuple[str, str, int, st...; symbols: get_expert_mapping
  - `vllm/model_executor/models/bailing_moe.py` modified +1/-0 (1 lines); hunks: -476,6 +476,7 @@ def forward(; symbols: forward, get_expert_mapping
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/fused_moe/layer.py
@@ -2007,6 +2007,7 @@ def combine_output(states: torch.Tensor) -> torch.Tensor:
+        model: torch.nn.Module,
@@ -2025,13 +2026,19 @@ def make_expert_params_mapping(
+        base_layer = (
+            "base_layer."
+            if any(".base_layer." in name for name, _ in model.named_parameters())
+            else ""
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -1486,6 +1486,7 @@ def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
+            self,
@@ -1519,6 +1520,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
+            self,
diff -- vllm/model_executor/models/llama4.py
@@ -539,6 +539,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
+            self,
@@ -548,6 +549,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
+            self,
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/layer.py` modified +10/-3; `vllm/model_executor/models/deepseek_v2.py` modified +2/-0; `vllm/model_executor/models/llama4.py` modified +2/-0; `vllm/model_executor/models/afmoe.py` modified +1/-0; `vllm/model_executor/models/bailing_moe.py` modified +1/-0; `vllm/model_executor/models/deepseek_eagle.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/afmoe.py`, `vllm/model_executor/models/bailing_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #42151 - [MM][Perf][CG] Support ViT full CUDA graph for Qwen3.5

- Link: https://github.com/vllm-project/vllm/pull/42151
- Status/date: merged / 2026-05-13
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_5.py`; associated commits `92def124bcb7`
- Diff scope read: GitHub Pull Request files API returned 4 files, +112/-5, 187 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MM][Perf][CG] Support ViT full CUDA graph for Qwen3.5"; model line: Qwen3.5; category: performance/backend optimization; main diff: `vllm/model_executor/models/qwen3_5.py`; technical summary: Covers "[MM][Perf][CG] Support ViT full CUDA graph for Qwen3.5"; the main implementation surface is `vllm/model_executor/models/qwen3_5.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/qwen3_5.py` modified +2/-0 (2 lines); hunks: -565,6 +565,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; -778,6 +779,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_5.py` modified +2/-0 (2 lines); hunks: -565,6 +565,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; -778,6 +779,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_5.py
@@ -565,6 +565,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
+        self.model_config = vllm_config.model_config
@@ -778,6 +779,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
+        self.model_config = vllm_config.model_config
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_5.py` modified +2/-0
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/generation/test_vit_cudagraph.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #42716 - Fix Weight loading for Qwen3.5-MTP and Qwen3-VL using runai_streamer

- Link: https://github.com/vllm-project/vllm/pull/42716
- Status/date: merged / 2026-05-17
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/qwen3_5_mtp.py`; associated commits `a94189295b8b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +4/-4, 22 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix Weight loading for Qwen3.5-MTP and Qwen3-VL using runai_streamer"; model line: Qwen3.5; category: bug fix; main diff: `vllm/model_executor/models/qwen3_5_mtp.py`; technical summary: Covers "Fix Weight loading for Qwen3.5-MTP and Qwen3-VL using runai_streamer"; the main implementation surface is `vllm/model_executor/models/qwen3_5_mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/qwen3_5_mtp.py` modified +2/-2 (4 lines); hunks: -175,8 +175,8 @@ def load_fused_expert_weights(; symbols: load_fused_expert_weights, touching `load_fused_expert_weights`.
- Code diff details:
  - `vllm/model_executor/models/qwen3_5_mtp.py` modified +2/-2 (4 lines); hunks: -175,8 +175,8 @@ def load_fused_expert_weights(; symbols: load_fused_expert_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen3_5_mtp.py
@@ -175,8 +175,8 @@ def load_fused_expert_weights(
-                shard_id,
-                expert_id,
+                shard_id=shard_id,
+                expert_id=expert_id,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen3_5_mtp.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/qwen3_5_mtp.py`, `vllm/model_executor/models/qwen3_vl_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #41436 - [ROCm][Quantization][3/N] Refactor quark_moe w4a4 w/ oracle

- Link: https://github.com/vllm-project/vllm/pull/41436
- Status/date: merged / 2026-05-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +224/-158, 564 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm][Quantization][3/N] Refactor quark_moe w4a4 w/ oracle"; model line: Qwen3.5; category: performance/backend optimization; main diff: `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`, `vllm/model_executor/layers/fused_moe/experts/rocm_aiter_moe.py`; technical summary: Covers "[ROCm][Quantization][3/N] Refactor quark_moe w4a4 w/ oracle"; the main implementation surface is `vllm/model_executor/layers/quantization/quark/quark_moe.py`, `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`, `vllm/model_executor/layers/fused_moe/experts/rocm_aiter_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +26/-151 (177 lines); hunks: -55,6 +55,7; -1040,6 +1041,11 @@ def __init__(; symbols: __init__, maybe_roundup_sizes, create_weights, process_weights_after_loading, touching `__init__, maybe_roundup_sizes, create_weights`; `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` modified +76/-2 (78 lines); hunks: -31,6 +31,7; -74,6 +75,7 @@ class Mxfp4MoeBackend(Enum):; symbols: Mxfp4MoeBackend, backend_to_kernel_cls, map_mxfp4_backend, _get_priority_backends_for_gpt_oss, touching `Mxfp4MoeBackend, backend_to_kernel_cls, map_mxfp4_backend`; `vllm/model_executor/layers/fused_moe/experts/rocm_aiter_moe.py` modified +17/-0 (17 lines); hunks: -26,6 +26,7; -377,6 +378,21 @@ def expects_unquantized_inputs(self) -> bool:; symbols: expects_unquantized_inputs, activation_format, is_supported_config, _supports_current_device, touching `expects_unquantized_inputs, activation_format, is_supported_config`; `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-MXFP4-AITER-TP2.yaml` added +12/-0 (12 lines); hunks: -0,0 +1,12.
- Code diff details:
  - `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +26/-151 (177 lines); hunks: -55,6 +55,7; -1040,6 +1041,11 @@ def __init__(; symbols: __init__, maybe_roundup_sizes, create_weights, process_weights_after_loading
  - `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` modified +76/-2 (78 lines); hunks: -31,6 +31,7; -74,6 +75,7 @@ class Mxfp4MoeBackend(Enum):; symbols: Mxfp4MoeBackend, backend_to_kernel_cls, map_mxfp4_backend, _get_priority_backends_for_gpt_oss
  - `vllm/model_executor/layers/fused_moe/experts/rocm_aiter_moe.py` modified +17/-0 (17 lines); hunks: -26,6 +26,7; -377,6 +378,21 @@ def expects_unquantized_inputs(self) -> bool:; symbols: expects_unquantized_inputs, activation_format, is_supported_config, _supports_current_device
  - `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-MXFP4-AITER-TP2.yaml` added +12/-0 (12 lines); hunks: -0,0 +1,12
  - `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-MXFP4-EMU-TP2.yaml` renamed +3/-1 (4 lines); hunks: -1,8 +1,10
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/quantization/quark/quark_moe.py
@@ -55,6 +55,7 @@
+    kMxfp4Dynamic,
@@ -1040,6 +1041,11 @@ def __init__(
+        elif self.ocp_mx_scheme == "w_mxfp4_a_mxfp4":
+            # W4A4: MXFP4 weights + MXFP4 activations
+            self.mxfp4_backend, self.experts_cls = select_mxfp4_moe_backend(
+                moe, activation_key=kMxfp4Dynamic
diff -- vllm/model_executor/layers/fused_moe/oracle/mxfp4.py
@@ -31,6 +31,7 @@
+    kMxfp4Dynamic,
@@ -74,6 +75,7 @@ class Mxfp4MoeBackend(Enum):
+    AITER_MXFP4_MXFP4 = "AITER_MXFP4_MXFP4"  # W4A4: CK kernel
@@ -89,6 +91,7 @@ class Mxfp4MoeBackend(Enum):
+    Mxfp4MoeBackend.AITER_MXFP4_MXFP4,
@@ -193,6 +196,13 @@ def backend_to_kernel_cls(
diff -- vllm/model_executor/layers/fused_moe/experts/rocm_aiter_moe.py
@@ -26,6 +26,7 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/quark/quark_moe.py` modified +26/-151; `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` modified +76/-2; `vllm/model_executor/layers/fused_moe/experts/rocm_aiter_moe.py` modified +17/-0
  - tests: `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-MXFP4-AITER-TP2.yaml` added +12/-0; `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-MXFP4-EMU-TP2.yaml` renamed +3/-1; `tests/evals/gsm8k/configs/models-mi3xx.txt` modified +2/-1; `tests/evals/gsm8k/configs/models-qwen35-mi355.txt` modified +2/-1; `tests/quantization/test_gfx950_moe.py` modified +86/-2
- Risk and verification: The diff ships test coverage in `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-MXFP4-AITER-TP2.yaml`, `tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-MXFP4-EMU-TP2.yaml`, `tests/evals/gsm8k/configs/models-mi3xx.txt`, `tests/evals/gsm8k/configs/models-qwen35-mi355.txt`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #41126 - [Attention] Mamba attention module refactor

- Link: https://github.com/vllm-project/vllm/pull/41126
- Status/date: merged / 2026-05-22
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +765/-774, 1913 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Attention] Mamba attention module refactor"; model line: Qwen3.5; category: model implementation change; main diff: `vllm/model_executor/models/olmo_hybrid.py`, `vllm/model_executor/layers/mamba/gdn/olmo_gdn_linear_attn.py`, `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py`; technical summary: Covers "[Attention] Mamba attention module refactor"; the main implementation surface is `vllm/model_executor/models/olmo_hybrid.py`, `vllm/model_executor/layers/mamba/gdn/olmo_gdn_linear_attn.py`, `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/olmo_hybrid.py` modified +6/-645 (651 lines); hunks: -26,73 +26,47; -107,502 +81,6; symbols: _make_fused_conv1d_weight_loader, weight_loader, OlmoHybridGatedDeltaNet, mamba_type, touching `_make_fused_conv1d_weight_loader, weight_loader, OlmoHybridGatedDeltaNet`; `vllm/model_executor/layers/mamba/gdn/olmo_gdn_linear_attn.py` added +634/-0 (634 lines); hunks: -0,0 +1,634; symbols: OlmoHybridGatedDeltaNetAttention, get_state_shape, __init__, rearrange_mixed_qkv, touching `OlmoHybridGatedDeltaNetAttention, get_state_shape, __init__`; `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py` renamed +26/-45 (71 lines); hunks: -5,39 +5,37; -83,11 +81,8 @@ def kda_attention_fake(; symbols: kda_attention_fake, KimiDeltaAttention, mamba_type, KimiGatedDeltaNetAttention, touching `kda_attention_fake, KimiDeltaAttention, mamba_type`; `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` renamed +19/-52 (71 lines); hunks: -5,7 +5,6; -15,8 +14,6; symbols: forward_native, GatedDeltaNetAttention, mamba_type, get_state_dtype, touching `forward_native, GatedDeltaNetAttention, mamba_type`.
- Code diff details:
  - `vllm/model_executor/models/olmo_hybrid.py` modified +6/-645 (651 lines); hunks: -26,73 +26,47; -107,502 +81,6; symbols: _make_fused_conv1d_weight_loader, weight_loader, OlmoHybridGatedDeltaNet, mamba_type
  - `vllm/model_executor/layers/mamba/gdn/olmo_gdn_linear_attn.py` added +634/-0 (634 lines); hunks: -0,0 +1,634; symbols: OlmoHybridGatedDeltaNetAttention, get_state_shape, __init__, rearrange_mixed_qkv
  - `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py` renamed +26/-45 (71 lines); hunks: -5,39 +5,37; -83,11 +81,8 @@ def kda_attention_fake(; symbols: kda_attention_fake, KimiDeltaAttention, mamba_type, KimiGatedDeltaNetAttention
  - `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` renamed +19/-52 (71 lines); hunks: -5,7 +5,6; -15,8 +14,6; symbols: forward_native, GatedDeltaNetAttention, mamba_type, get_state_dtype
  - `vllm/model_executor/layers/mamba/gdn/base.py` added +58/-0 (58 lines); hunks: -0,0 +1,58; symbols: GatedDeltaNetAttention, for, __init__, mamba_type
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/olmo_hybrid.py
@@ -26,73 +26,47 @@
-from einops import rearrange
-from transformers.activations import ACT2FN
-    CacheConfig,
-    ModelConfig,
-    SpeculativeConfig,
-    get_current_vllm_config,
diff -- vllm/model_executor/layers/mamba/gdn/olmo_gdn_linear_attn.py
@@ -0,0 +1,634 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import torch
+from einops import rearrange
+from torch import nn
+from vllm.config import (
diff -- vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py
@@ -5,39 +5,37 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/olmo_hybrid.py` modified +6/-645; `vllm/model_executor/layers/mamba/gdn/olmo_gdn_linear_attn.py` added +634/-0; `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py` renamed +26/-45; `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` renamed +19/-52; `vllm/model_executor/layers/mamba/gdn/base.py` added +58/-0; `vllm/model_executor/models/kimi_linear.py` modified +13/-27
- Risk and verification: Runtime changes concentrate in `vllm/config/compilation.py`, `vllm/model_executor/layers/mamba/gdn/__init__.py`, `vllm/model_executor/layers/mamba/gdn/base.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #42124 - Add LM head quantization support for ModelOpt

- Link: https://github.com/vllm-project/vllm/pull/42124
- Status/date: merged / 2026-05-26
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +220/-5, 315 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add LM head quantization support for ModelOpt"; model line: Qwen3.5; category: performance/backend optimization; main diff: `tests/model_executor/test_qwen3_5_quantization.py`, `tests/model_executor/test_nemotron_h_quantization.py`, `vllm/model_executor/layers/quantization/modelopt.py`; technical summary: Covers "Add LM head quantization support for ModelOpt"; the main implementation surface is `tests/model_executor/test_qwen3_5_quantization.py`, `tests/model_executor/test_nemotron_h_quantization.py`, `vllm/model_executor/layers/quantization/modelopt.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/model_executor/test_qwen3_5_quantization.py` added +78/-0 (78 lines); hunks: -0,0 +1,78; symbols: test_qwen3_5_lm_head_receives_quant_config, test_qwen3_5_mtp_lm_head_receives_quant_config, touching `test_qwen3_5_lm_head_receives_quant_config, test_qwen3_5_mtp_lm_head_receives_quant_config`; `tests/model_executor/test_nemotron_h_quantization.py` added +34/-0 (34 lines); hunks: -0,0 +1,34; symbols: test_nemotron_h_lm_head_receives_quant_config, touching `test_nemotron_h_lm_head_receives_quant_config`; `vllm/model_executor/layers/quantization/modelopt.py` modified +5/-4 (9 lines); hunks: -85,6 +85,7; -187,7 +188,7 @@ def get_quant_method(; symbols: get_quant_method, touching `get_quant_method`; `vllm/model_executor/layers/vocab_parallel_embedding.py` modified +7/-0 (7 lines); hunks: -290,6 +290,7 @@ def __init__(; -438,6 +439,12 @@ def weight_loader(self, param: Parameter, loaded_weight: to...; symbols: __init__, weight_loader, touching `__init__, weight_loader`.
- Code diff details:
  - `tests/model_executor/test_qwen3_5_quantization.py` added +78/-0 (78 lines); hunks: -0,0 +1,78; symbols: test_qwen3_5_lm_head_receives_quant_config, test_qwen3_5_mtp_lm_head_receives_quant_config
  - `tests/model_executor/test_nemotron_h_quantization.py` added +34/-0 (34 lines); hunks: -0,0 +1,34; symbols: test_nemotron_h_lm_head_receives_quant_config
  - `vllm/model_executor/layers/quantization/modelopt.py` modified +5/-4 (9 lines); hunks: -85,6 +85,7; -187,7 +188,7 @@ def get_quant_method(; symbols: get_quant_method
  - `vllm/model_executor/layers/vocab_parallel_embedding.py` modified +7/-0 (7 lines); hunks: -290,6 +290,7 @@ def __init__(; -438,6 +439,12 @@ def weight_loader(self, param: Parameter, loaded_weight: to...; symbols: __init__, weight_loader
  - `vllm/model_executor/models/nemotron_h.py` modified +1/-0 (1 lines); hunks: -875,6 +875,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__
- Key code excerpts:

```diff
diff -- tests/model_executor/test_qwen3_5_quantization.py
@@ -0,0 +1,78 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from unittest.mock import Mock, patch
+def test_qwen3_5_lm_head_receives_quant_config():
+    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLMBase
+    mock_quant_config = Mock()
diff -- tests/model_executor/test_nemotron_h_quantization.py
@@ -0,0 +1,34 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from unittest.mock import Mock, patch
+def test_nemotron_h_lm_head_receives_quant_config():
+    from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM
+    mock_quant_config = Mock()
diff -- vllm/model_executor/layers/quantization/modelopt.py
@@ -85,6 +85,7 @@
```

- Reviewed files:
  - tests: `tests/model_executor/test_qwen3_5_quantization.py` added +78/-0; `tests/model_executor/test_nemotron_h_quantization.py` added +34/-0; `tests/quantization/test_modelopt.py` modified +93/-1
  - runtime: `vllm/model_executor/layers/quantization/modelopt.py` modified +5/-4; `vllm/model_executor/layers/vocab_parallel_embedding.py` modified +7/-0; `vllm/model_executor/models/nemotron_h.py` modified +1/-0; `vllm/model_executor/models/qwen3_5.py` modified +1/-0; `vllm/model_executor/models/qwen3_5_mtp.py` modified +1/-0
- Risk and verification: The diff ships test coverage in `tests/model_executor/test_nemotron_h_quantization.py`, `tests/model_executor/test_qwen3_5_quantization.py`, `tests/quantization/test_modelopt.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #39419 - [SpecDecode] Reduce TP communication for large-vocab draft models speculative decoding

- Link: https://github.com/vllm-project/vllm/pull/39419
- Status/date: merged / 2026-06-10
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +53/-39, 169 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[SpecDecode] Reduce TP communication for large-vocab draft models speculative decoding"; model line: Qwen3.5; category: model implementation change; main diff: `vllm/model_executor/models/interfaces.py`, `vllm/model_executor/models/llama4_eagle.py`, `vllm/model_executor/models/qwen3.py`; technical summary: Covers "[SpecDecode] Reduce TP communication for large-vocab draft models speculative decoding"; the main implementation surface is `vllm/model_executor/models/interfaces.py`, `vllm/model_executor/models/llama4_eagle.py`, `vllm/model_executor/models/qwen3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/interfaces.py` modified +35/-0 (35 lines); hunks: -1282,6 +1282,41 @@ def supports_any_eagle(; symbols: supports_any_eagle, LocalArgmaxMixin, get_top_tokens, EagleModelMixin, touching `supports_any_eagle, LocalArgmaxMixin, get_top_tokens`; `vllm/model_executor/models/llama4_eagle.py` modified +0/-17 (17 lines); hunks: -208,23 +208,6 @@ def forward(; symbols: forward, get_top_tokens, load_weights, transform, touching `forward, get_top_tokens, load_weights`; `vllm/model_executor/models/qwen3.py` modified +8/-2 (10 lines); hunks: -48,7 +48,13; -259,7 +265,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, Qwen3ForCausalLM, touching `__init__, Qwen3ForCausalLM`; `vllm/model_executor/models/deepseek_eagle3.py` modified +2/-1 (3 lines); hunks: -31,6 +31,7; -309,7 +310,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights, Eagle3DeepseekV2ForCausalLM, __init__, touching `load_weights, Eagle3DeepseekV2ForCausalLM, __init__`.
- Code diff details:
  - `vllm/model_executor/models/interfaces.py` modified +35/-0 (35 lines); hunks: -1282,6 +1282,41 @@ def supports_any_eagle(; symbols: supports_any_eagle, LocalArgmaxMixin, get_top_tokens, EagleModelMixin
  - `vllm/model_executor/models/llama4_eagle.py` modified +0/-17 (17 lines); hunks: -208,23 +208,6 @@ def forward(; symbols: forward, get_top_tokens, load_weights, transform
  - `vllm/model_executor/models/qwen3.py` modified +8/-2 (10 lines); hunks: -48,7 +48,13; -259,7 +265,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, Qwen3ForCausalLM
  - `vllm/model_executor/models/deepseek_eagle3.py` modified +2/-1 (3 lines); hunks: -31,6 +31,7; -309,7 +310,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights, Eagle3DeepseekV2ForCausalLM, __init__
  - `vllm/model_executor/models/llama.py` modified +2/-1 (3 lines); hunks: -62,6 +62,7; -487,7 +488,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights, LlamaForCausalLM
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/interfaces.py
@@ -1282,6 +1282,41 @@ def supports_any_eagle(
+class LocalArgmaxMixin:
+    """Mixin for draft model heads in speculative decoding.
+    Provides a D2T-aware ``get_top_tokens`` that preserves the
+    local-argmax communication reduction even when the draft vocabulary
+    is smaller than the target vocabulary.
+    When ``draft_id_to_target_id`` is present (shape ``(draft_vocab_size,)``,
diff -- vllm/model_executor/models/llama4_eagle.py
@@ -208,23 +208,6 @@ def forward(
-    def get_top_tokens(
-        self,
-        hidden_states: torch.Tensor,
-    ) -> torch.Tensor:
-        """Vocab-parallel argmax without all-gathering full logits.
-        Falls back to full logits when draft_id_to_target_id remapping is
diff -- vllm/model_executor/models/qwen3.py
@@ -48,7 +48,13 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/interfaces.py` modified +35/-0; `vllm/model_executor/models/llama4_eagle.py` modified +0/-17; `vllm/model_executor/models/qwen3.py` modified +8/-2; `vllm/model_executor/models/deepseek_eagle3.py` modified +2/-1; `vllm/model_executor/models/llama.py` modified +2/-1; `vllm/model_executor/models/qwen3_5_mtp.py` modified +2/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_eagle3.py`, `vllm/model_executor/models/interfaces.py`, `vllm/model_executor/models/llama.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #45161 - Deprecate Transformers v4 support

- Link: https://github.com/vllm-project/vllm/pull/45161
- Status/date: merged / 2026-06-11
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 19 files, +62/-268, 612 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Deprecate Transformers v4 support"; model line: Qwen3.5; category: model support/runtime entry; main diff: `vllm/model_executor/models/transformers/base.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/model_loader/weight_utils.py`; technical summary: Covers "Deprecate Transformers v4 support"; the main implementation surface is `vllm/model_executor/models/transformers/base.py`, `vllm/model_executor/models/qwen3_omni_moe_thinker.py`, `vllm/model_executor/model_loader/weight_utils.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/transformers/base.py` modified +16/-42 (58 lines); hunks: -27,6 +27,10; -212,16 +216,9 @@ def _patch_config(self):; symbols: _patch_config, _get_decoder_cls, _create_hf_to_vllm_mapper, _get_tie_word_embeddings, touching `_patch_config, _get_decoder_cls, _create_hf_to_vllm_mapper`; `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +0/-36 (36 lines); hunks: -30,9 +30,7; -1261,40 +1259,6 @@ def pad_to_hop_length(x: np.ndarray, hop_length: int) ->...; symbols: pad_to_hop_length, touching `pad_to_hop_length`; `vllm/model_executor/model_loader/weight_utils.py` modified +1/-18 (19 lines); hunks: -77,30 +77,13; symbols: enable_hf_transfer, enable_xet_high_performance, DisabledTqdm, touching `enable_hf_transfer, enable_xet_high_performance, DisabledTqdm`; `vllm/transformers_utils/configs/qwen3_5.py` modified +5/-12 (17 lines); hunks: -94,18 +94,11 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/transformers/base.py` modified +16/-42 (58 lines); hunks: -27,6 +27,10; -212,16 +216,9 @@ def _patch_config(self):; symbols: _patch_config, _get_decoder_cls, _create_hf_to_vllm_mapper, _get_tie_word_embeddings
  - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +0/-36 (36 lines); hunks: -30,9 +30,7; -1261,40 +1259,6 @@ def pad_to_hop_length(x: np.ndarray, hop_length: int) ->...; symbols: pad_to_hop_length
  - `vllm/model_executor/model_loader/weight_utils.py` modified +1/-18 (19 lines); hunks: -77,30 +77,13; symbols: enable_hf_transfer, enable_xet_high_performance, DisabledTqdm
  - `vllm/transformers_utils/configs/qwen3_5.py` modified +5/-12 (17 lines); hunks: -94,18 +94,11 @@ def __init__(; symbols: __init__
  - `vllm/transformers_utils/configs/qwen3_5_moe.py` modified +5/-12 (17 lines); hunks: -100,18 +100,11 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/transformers/base.py
@@ -27,6 +27,10 @@
+from transformers.conversion_mapping import (
+    WeightRenaming,
+    get_model_conversion_mapping,
+)
@@ -212,16 +216,9 @@ def _patch_config(self):
-        - Propagates this dtype to any sub-configs because Transformers model
diff -- vllm/model_executor/models/qwen3_omni_moe_thinker.py
@@ -30,9 +30,7 @@
-from packaging.version import Version
-from transformers import __version__ as TRANSFORMERS_VERSION
@@ -1261,40 +1259,6 @@ def pad_to_hop_length(x: np.ndarray, hop_length: int) -> np.ndarray:
-            if Version(TRANSFORMERS_VERSION) < Version("4.58.0"):
-                # Extract audio_sample_rate before restructuring
-                audio_sample_rate = mm_kwargs.pop("audio_sample_rate", None)
diff -- vllm/model_executor/model_loader/weight_utils.py
@@ -77,30 +77,13 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/transformers/base.py` modified +16/-42; `vllm/model_executor/models/qwen3_omni_moe_thinker.py` modified +0/-36; `vllm/model_executor/model_loader/weight_utils.py` modified +1/-18; `vllm/transformers_utils/configs/qwen3_5.py` modified +5/-12; `vllm/transformers_utils/configs/qwen3_5_moe.py` modified +5/-12; `vllm/model_executor/models/ultravox.py` modified +0/-15
- Risk and verification: Runtime changes concentrate in `vllm/config/vllm.py`, `vllm/model_executor/model_loader/weight_utils.py`, `vllm/model_executor/models/gemma3n_mm.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
