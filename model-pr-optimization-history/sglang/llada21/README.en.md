# sglang LLaDA 2.1 Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `docs_new/cookbook/autoregressive/InclusionAI/LLaDA-2.1.mdx` | no direct PR-number commit |
| `docs_new/src/snippets/autoregressive/llada-21-deployment.jsx` | no direct PR-number commit |
| `python/sglang/srt/models/llada2.py` | [#18485](https://github.com/sgl-project/sglang/pull/18485) |
| `test/registered/ascend/basic_function/dllm/test_npu_llada2_mini.py` | no direct PR-number commit |
| `test/registered/dllm/test_llada2_mini.py` | no direct PR-number commit |
| `test/registered/dllm/test_llada2_mini_amd.py` | no direct PR-number commit |

## PR Coverage Summary

- Git-traced PRs: 1
- Extra PRs preserved from existing docs: 35
- Total PRs in this document: 35
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2025-11-26 | [#12588](https://github.com/sgl-project/sglang/pull/12588) | merged | [Feature] Initial block diffusion language model support | `python/sglang/srt/models/llada2.py`, `python/sglang/srt/layers/logits_processor.py`, `python/sglang/srt/model_executor/forward_batch_info.py` |
| 2025-12-07 | [#14337](https://github.com/sgl-project/sglang/pull/14337) | merged | remove unecessary dual stream token threshold from the rest of models (qwen moe, kimi linear, etc.) | `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/kimi_linear.py`, `python/sglang/srt/models/llada2.py` |
| 2025-12-12 | [#13730](https://github.com/sgl-project/sglang/pull/13730) | merged | [bugfix] fix TBO crashes when attn_tp_size > 1 | `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/model_executor/forward_batch_info.py`, `python/sglang/srt/models/bailing_moe.py` |
| 2025-12-28 | [#15835](https://github.com/sgl-project/sglang/pull/15835) | merged | [Feature] JIT Fused QK norm + qk norm clean up | `python/sglang/srt/models/utils.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen3.py` |
| 2026-01-06 | [#16420](https://github.com/sgl-project/sglang/pull/16420) | merged | ci: migrate DLLM tests to test/registered/dllm/ | `test/registered/dllm/test_llada2_mini.py`, `test/registered/dllm/test_llada2_mini_amd.py`, `test/srt/run_suite.py` |
| 2026-01-08 | [#16675](https://github.com/sgl-project/sglang/pull/16675) | merged | [AMD] Fix CI - unit-test-backend-1-gpu-amd-mi35x and unit-test-backend-2-gpu-amd, stage-b-test-small-1-gpu-amd | `test/registered/models/test_compressed_tensors_models.py`, `test/registered/models/test_cross_encoder_models.py`, `test/registered/models/test_embedding_models.py` |
| 2026-01-11 | [#16835](https://github.com/sgl-project/sglang/pull/16835) | merged | Update est_time for stage-b-test-small-1-gpu tests | `test/registered/models/test_encoder_embedding_models.py`, `test/registered/models/test_vlm_models.py`, `test/registered/attention/test_torch_native_attention_backend.py` |
| 2026-01-15 | [#16949](https://github.com/sgl-project/sglang/pull/16949) | merged | [AMD CI] migrate and re-enable CI tests to new CI registry | `test/registered/models/test_generation_models.py`, `test/registered/layers/mamba/test_causal_conv1d.py`, `test/registered/layers/mamba/test_mamba_ssm.py` |
| 2026-01-15 | [#16826](https://github.com/sgl-project/sglang/pull/16826) | merged | [CI] Reorganize stage-b 1-GPU tests for 5090 compatibility | `test/registered/models/test_encoder_embedding_models.py`, `test/registered/models/test_embedding_models.py`, `test/registered/models/test_reward_models.py` |
| 2026-01-24 | [#17570](https://github.com/sgl-project/sglang/pull/17570) | merged | Use attn tp group in embedding for more models | `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/bailing_moe_nextn.py`, `python/sglang/srt/models/falcon_h1.py` |
| 2026-02-09 | [#18423](https://github.com/sgl-project/sglang/pull/18423) | merged | [AMD] Update aiter to v0.1.10.post2 | `python/sglang/srt/layers/attention/aiter_backend.py`, `test/registered/models/test_vlm_models.py`, `scripts/ci/amd/amd_ci_warmup_aiter.py` |
| 2026-02-10 | [#17484](https://github.com/sgl-project/sglang/pull/17484) | merged | [DLLM] Basic dLLM scheduling strategy and implementation | `python/sglang/srt/model_executor/forward_batch_info.py`, `python/sglang/srt/dllm/mixin/scheduler.py`, `python/sglang/srt/managers/schedule_batch.py` |
| 2026-02-15 | [#18860](https://github.com/sgl-project/sglang/pull/18860) | merged | update pre-commit config | `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/models/pixtral.py`, `python/sglang/srt/layers/quantization/modelslim/modelslim_moe.py` |
| 2026-02-21 | [#18844](https://github.com/sgl-project/sglang/pull/18844) | merged | [Feature] rewrite rope kernel; remove flashinfer dependencies | `python/sglang/srt/layers/rotary_embedding.py`, `python/sglang/srt/models/llada2.py`, `python/sglang/srt/models/gpt_oss.py` |
| 2026-03-05 | [#18724](https://github.com/sgl-project/sglang/pull/18724) | merged | [DLLM] Add initial radix cache support | `python/sglang/srt/dllm/mixin/scheduler.py`, `python/sglang/srt/managers/scheduler_output_processor_mixin.py`, `python/sglang/srt/dllm/mixin/req.py` |
| 2026-03-09 | [#18485](https://github.com/sgl-project/sglang/pull/18485) | merged | [NPU] [DLLM]DLLM LLaDA2.x graph mode support with NPU speedup modifications | `python/sglang/srt/models/llada2.py` |
| 2026-03-18 | [#17784](https://github.com/sgl-project/sglang/pull/17784) | merged | Upgrade transformers==5.3.0 | `python/sglang/srt/models/gemma3_causal.py`, `python/sglang/srt/layers/rotary_embedding/factory.py`, `python/sglang/srt/configs/model_config.py` |
| 2026-03-23 | [#21187](https://github.com/sgl-project/sglang/pull/21187) | merged | ci: unify PR test suite naming | `test/registered/layers/mamba/test_causal_conv1d.py`, `test/registered/layers/mamba/test_mamba_ssm.py`, `test/registered/layers/mamba/test_mamba_ssm_ssd.py` |
| 2026-03-26 | [#21135](https://github.com/sgl-project/sglang/pull/21135) | merged | fix: use get_rope_config() to support models without rope_parameters | `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4.py`, `python/sglang/srt/models/grok.py` |
| 2026-04-01 | [#20751](https://github.com/sgl-project/sglang/pull/20751) | merged | [NPU]Add a full test pipeline on NPU, resolve issues in the NPU test architecture | `.github/workflows/full-test-npu.yml`, `.github/workflows/nightly-test-npu.yml`, `.github/workflows/pr-test-npu.yml` |
| 2026-04-02 | [#21667](https://github.com/sgl-project/sglang/pull/21667) | merged | Unify GSM8K eval path to Chat API for regression CI readiness | `test/manual/models/test_unsloth_models.py`, `test/manual/models/test_falcon_h1_models.py`, `test/registered/models/test_qwen_models.py` |
| 2026-04-10 | [#22305](https://github.com/sgl-project/sglang/pull/22305) | merged | [CI] Update est_time for 64 tests based on actual elapsed times | `test/registered/models/test_nvidia_nemotron_3_nano.py`, `test/registered/models/test_transformers_backend_eval.py`, `test/registered/models/test_transformers_models.py` |
| 2026-04-11 | [#22565](https://github.com/sgl-project/sglang/pull/22565) | merged | chore: update CI test est_time values | `test/registered/layers/mamba/test_causal_conv1d.py`, `test/registered/layers/mamba/test_mamba2_mixer.py`, `test/registered/layers/mamba/test_mamba_ssm.py` |
| 2026-04-20 | [#23001](https://github.com/sgl-project/sglang/pull/23001) | merged | Add new Mintlify documentation site (docs_new/) | `docs_new/docs/advanced_features/tool_parser.mdx`, `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx`, `docs_new/docs/advanced_features/separate_reasoning.mdx` |
| 2026-04-21 | [#23337](https://github.com/sgl-project/sglang/pull/23337) | merged | [Docs] Sync docs_new with legacy docs and update migration redirects | `docs_new/docs/supported-models/multimodal_language_models.mdx`, `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx` |
| 2026-04-26 | [#23732](https://github.com/sgl-project/sglang/pull/23732) | merged | Apply should_use_dp_reduce_scatterv guard to remaining MoE models (follow-up to #23731) | `python/sglang/srt/models/llada2.py`, `python/sglang/srt/models/hunyuan_v3.py`, `python/sglang/srt/models/bailing_moe_linear.py` |
| 2026-04-27 | [#23785](https://github.com/sgl-project/sglang/pull/23785) | merged | chore: update CI test est_time values | `test/registered/layers/mamba/test_causal_conv1d.py`, `test/registered/layers/mamba/test_mamba2_mixer.py`, `test/registered/layers/mamba/test_mamba_ssm.py` |
| 2026-04-27 | [#23748](https://github.com/sgl-project/sglang/pull/23748) | merged | refactor(moe): centralize post-experts all-reduce skip predicate | `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/models/sarvam_moe.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2026-05-02 | [#23835](https://github.com/sgl-project/sglang/pull/23835) | merged | [NPU] Add GitHub test summary and deduplicate test code. Part 1 | `test/registered/ascend/basic_function/parallel_strategy/expert_parallelism/test_npu_deepep_auto_deepseek_v3_2_w8a8.py`, `test/registered/ascend/basic_function/optimization_debug/test_npu_piecewise_graph_prefill.py`, `test/registered/ascend/basic_function/speculative_inference/test_npu_eagle3.py` |
| 2026-05-14 | [#25197](https://github.com/sgl-project/sglang/pull/25197) | merged | ci: decouple stage and runner for cuda registry | `test/registered/layers/test_fla_layernorm_guard.py`, `test/registered/models/test_dummy_grok_models.py`, `test/registered/models/test_ministral3_models.py` |
| 2026-05-16 | [#25420](https://github.com/sgl-project/sglang/pull/25420) | merged | [CI] Rename basic CI `stage-a/b/c` -> `base-a/b/c` for symmetry with extra CI | `.github/workflows/pr-test-multimodal-gen.yml`, `test/registered/bench_fn/test_bench_serving_reasoning_stream.py`, `test/registered/function_call/test_kimik2_detector.py` |
| 2026-06-10 | [#23906](https://github.com/sgl-project/sglang/pull/23906) | merged | [Refactor] Cuda Graph Runner/Backend Refactor | `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` |
| 2026-06-18 | [#28567](https://github.com/sgl-project/sglang/pull/28567) | merged | Add get_parallel(): a structured accessor for parallel-topology state | `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py` |
| 2026-06-19 | [#28697](https://github.com/sgl-project/sglang/pull/28697) | merged | [docs] Add B300 cookbook deployment options | `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx`, `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx` |
| 2026-06-25 | [#29042](https://github.com/sgl-project/sglang/pull/29042) | merged | [NPU] Fix the DeepSeek-V2-Coder model accuracy issue | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/llada2.py`, `python/sglang/srt/hardware_backend/npu/moe/topk.py` |

## Per-PR Diff Audit Cards

### PR #12588 - [Feature] Initial block diffusion language model support

- Link: https://github.com/sgl-project/sglang/pull/12588
- Status/date: merged / 2025-11-26
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 13 files, +1286/-6, 1544 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feature] Initial block diffusion language model support"; model line: LLaDA 2.1; category: model support/runtime entry; main diff: `python/sglang/srt/models/llada2.py`, `python/sglang/srt/layers/logits_processor.py`, `python/sglang/srt/model_executor/forward_batch_info.py`; technical summary: Covers "[Feature] Initial block diffusion language model support"; the main implementation surface is `python/sglang/srt/models/llada2.py`, `python/sglang/srt/layers/logits_processor.py`, `python/sglang/srt/model_executor/forward_batch_info.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/llada2.py` added +941/-0 (941 lines); hunks: -0,0 +1,941; symbols: LLaDA2MoeMLP, __init__, forward, LLaDA2MoeGate, touching `LLaDA2MoeMLP, __init__, forward`; `python/sglang/srt/layers/logits_processor.py` modified +18/-1 (19 lines); hunks: -99,6 +99,9 @@ class LogitsProcessorOutput:; -229,7 +232,11 @@ def compute_dp_attention_metadata(self):; symbols: LogitsProcessorOutput, LogitsMetadata, compute_dp_attention_metadata, LogitsProcessor, touching `LogitsProcessorOutput, LogitsMetadata, compute_dp_attention_metadata`; `python/sglang/srt/model_executor/forward_batch_info.py` modified +11/-2 (13 lines); hunks: -441,8 +441,17 @@ def init_new(; symbols: init_new, touching `init_new`; `python/sglang/srt/layers/attention/flashinfer_backend.py` modified +8/-1 (9 lines); hunks: -126,6 +126,8 @@ def __init__(; -766,11 +768,16 @@ def forward_extend(; symbols: __init__, forward_extend, touching `__init__, forward_extend`.
- Code diff details:
  - `python/sglang/srt/models/llada2.py` added +941/-0 (941 lines); hunks: -0,0 +1,941; symbols: LLaDA2MoeMLP, __init__, forward, LLaDA2MoeGate
  - `python/sglang/srt/layers/logits_processor.py` modified +18/-1 (19 lines); hunks: -99,6 +99,9 @@ class LogitsProcessorOutput:; -229,7 +232,11 @@ def compute_dp_attention_metadata(self):; symbols: LogitsProcessorOutput, LogitsMetadata, compute_dp_attention_metadata, LogitsProcessor
  - `python/sglang/srt/model_executor/forward_batch_info.py` modified +11/-2 (13 lines); hunks: -441,8 +441,17 @@ def init_new(; symbols: init_new
  - `python/sglang/srt/layers/attention/flashinfer_backend.py` modified +8/-1 (9 lines); hunks: -126,6 +126,8 @@ def __init__(; -766,11 +768,16 @@ def forward_extend(; symbols: __init__, forward_extend
  - `python/sglang/srt/dllm/algorithm/low_confidence.py` added +59/-0 (59 lines); hunks: -0,0 +1,59; symbols: LowConfidence, run
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/llada2.py
@@ -0,0 +1,941 @@
+# coding=utf-8
+# Copyright 2023 Antgroup and The HuggingFace Inc. team. All rights reserved.
+#
+# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
+# and OPT implementations in this library. It has been modified from its
+# original forms to accommodate minor architectural differences compared
diff -- python/sglang/srt/layers/logits_processor.py
@@ -99,6 +99,9 @@ class LogitsProcessorOutput:
+    ## Part 4: Diffusion LLM only.
+    full_logits: Optional[torch.Tensor] = None
@@ -229,7 +232,11 @@ def compute_dp_attention_metadata(self):
-        self, config, skip_all_gather: bool = False, logit_scale: Optional[float] = None
+        self,
+        config,
diff -- python/sglang/srt/model_executor/forward_batch_info.py
@@ -441,8 +441,17 @@ def init_new(
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/llada2.py` added +941/-0; `python/sglang/srt/layers/logits_processor.py` modified +18/-1; `python/sglang/srt/model_executor/forward_batch_info.py` modified +11/-2; `python/sglang/srt/layers/attention/flashinfer_backend.py` modified +8/-1; `python/sglang/srt/dllm/algorithm/low_confidence.py` added +59/-0; `python/sglang/srt/server_args.py` modified +45/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/dllm/algorithm/__init__.py`, `python/sglang/srt/dllm/algorithm/base.py`, `python/sglang/srt/dllm/algorithm/low_confidence.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #14337 - remove unecessary dual stream token threshold from the rest of models (qwen moe, kimi linear, etc.)

- Link: https://github.com/sgl-project/sglang/pull/14337
- Status/date: merged / 2025-12-07
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +0/-8, 50 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "remove unecessary dual stream token threshold from the rest of models (qwen moe, kimi linear, etc.)"; model line: LLaDA 2.1; category: model implementation change; main diff: `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/kimi_linear.py`, `python/sglang/srt/models/llada2.py`; technical summary: Covers "remove unecessary dual stream token threshold from the rest of models (qwen moe, kimi linear, etc.)"; the main implementation surface is `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/kimi_linear.py`, `python/sglang/srt/models/llada2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/bailing_moe.py` modified +0/-2 (2 lines); hunks: -349,11 +349,9 @@ def forward_normal(; symbols: forward_normal, touching `forward_normal`; `python/sglang/srt/models/kimi_linear.py` modified +0/-2 (2 lines); hunks: -125,13 +125,11 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Te...; symbols: forward, touching `forward`; `python/sglang/srt/models/llada2.py` modified +0/-2 (2 lines); hunks: -349,11 +349,9 @@ def forward_normal(; symbols: forward_normal, touching `forward_normal`; `python/sglang/srt/models/qwen2_moe.py` modified +0/-2 (2 lines); hunks: -275,11 +275,9 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/bailing_moe.py` modified +0/-2 (2 lines); hunks: -349,11 +349,9 @@ def forward_normal(; symbols: forward_normal
  - `python/sglang/srt/models/kimi_linear.py` modified +0/-2 (2 lines); hunks: -125,13 +125,11 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Te...; symbols: forward
  - `python/sglang/srt/models/llada2.py` modified +0/-2 (2 lines); hunks: -349,11 +349,9 @@ def forward_normal(; symbols: forward_normal
  - `python/sglang/srt/models/qwen2_moe.py` modified +0/-2 (2 lines); hunks: -275,11 +275,9 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/bailing_moe.py
@@ -349,11 +349,9 @@ def forward_normal(
-        DUAL_STREAM_TOKEN_THRESHOLD = 1024
-            and hidden_states.shape[0] <= DUAL_STREAM_TOKEN_THRESHOLD
diff -- python/sglang/srt/models/kimi_linear.py
@@ -125,13 +125,11 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
-        DUAL_STREAM_TOKEN_THRESHOLD = 1024
-            and hidden_states.shape[0] <= DUAL_STREAM_TOKEN_THRESHOLD
diff -- python/sglang/srt/models/llada2.py
@@ -349,11 +349,9 @@ def forward_normal(
-        DUAL_STREAM_TOKEN_THRESHOLD = 1024
-            and hidden_states.shape[0] <= DUAL_STREAM_TOKEN_THRESHOLD
diff -- python/sglang/srt/models/qwen2_moe.py
@@ -275,11 +275,9 @@ def forward(
-        DUAL_STREAM_TOKEN_THRESHOLD = 1024
-            and hidden_states.shape[0] <= DUAL_STREAM_TOKEN_THRESHOLD
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/bailing_moe.py` modified +0/-2; `python/sglang/srt/models/kimi_linear.py` modified +0/-2; `python/sglang/srt/models/llada2.py` modified +0/-2; `python/sglang/srt/models/qwen2_moe.py` modified +0/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/kimi_linear.py`, `python/sglang/srt/models/llada2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #13730 - [bugfix] fix TBO crashes when attn_tp_size > 1

- Link: https://github.com/sgl-project/sglang/pull/13730
- Status/date: merged / 2025-12-12
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 20 files, +285/-16, 617 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[bugfix] fix TBO crashes when attn_tp_size > 1"; model line: LLaDA 2.1; category: bug fix; main diff: `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/model_executor/forward_batch_info.py`, `python/sglang/srt/models/bailing_moe.py`; technical summary: Covers "[bugfix] fix TBO crashes when attn_tp_size > 1"; the main implementation surface is `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/model_executor/forward_batch_info.py`, `python/sglang/srt/models/bailing_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/communicator.py` modified +14/-1 (15 lines); hunks: -217,14 +217,16 @@ class _LayerModeComputationContext:; -273,6 +275,15 @@ def _compute_mlp_mode(cls, context: _LayerModeComputationCo...; symbols: _LayerModeComputationContext, previous_layer, _compute_mlp_mode, _should_gather_for_tbo, touching `_LayerModeComputationContext, previous_layer, _compute_mlp_mode`; `python/sglang/srt/model_executor/forward_batch_info.py` modified +9/-0 (9 lines); hunks: -376,6 +376,7 @@ class ForwardBatch:; -852,6 +853,14 @@ def prepare_mlp_sync_batch(self, model_runner: ModelRunner):; symbols: ForwardBatch, prepare_mlp_sync_batch, _pad_inputs_to_size, touching `ForwardBatch, prepare_mlp_sync_batch, _pad_inputs_to_size`; `python/sglang/srt/models/bailing_moe.py` modified +4/-0 (4 lines); hunks: -582,12 +582,16 @@ def __init__(; symbols: __init__, touching `__init__`; `python/sglang/srt/models/falcon_h1.py` modified +3/-1 (4 lines); hunks: -198,15 +198,17 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/layers/communicator.py` modified +14/-1 (15 lines); hunks: -217,14 +217,16 @@ class _LayerModeComputationContext:; -273,6 +275,15 @@ def _compute_mlp_mode(cls, context: _LayerModeComputationCo...; symbols: _LayerModeComputationContext, previous_layer, _compute_mlp_mode, _should_gather_for_tbo
  - `python/sglang/srt/model_executor/forward_batch_info.py` modified +9/-0 (9 lines); hunks: -376,6 +376,7 @@ class ForwardBatch:; -852,6 +853,14 @@ def prepare_mlp_sync_batch(self, model_runner: ModelRunner):; symbols: ForwardBatch, prepare_mlp_sync_batch, _pad_inputs_to_size
  - `python/sglang/srt/models/bailing_moe.py` modified +4/-0 (4 lines); hunks: -582,12 +582,16 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/falcon_h1.py` modified +3/-1 (4 lines); hunks: -198,15 +198,17 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/longcat_flash.py` modified +4/-0 (4 lines); hunks: -380,6 +380,8 @@ def __init__(; -398,6 +400,8 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/communicator.py
@@ -217,14 +217,16 @@ class _LayerModeComputationContext:
+    is_next_layer_sparse: Optional[bool]
+            num_layers=self.num_layers,
-            num_layers=self.num_layers,
+            is_next_layer_sparse=self.is_layer_sparse,
@@ -273,6 +275,15 @@ def _compute_mlp_mode(cls, context: _LayerModeComputationContext):
+    @classmethod
diff -- python/sglang/srt/model_executor/forward_batch_info.py
@@ -376,6 +376,7 @@ class ForwardBatch:
+    tbo_padded_len: Optional[int] = None
@@ -852,6 +853,14 @@ def prepare_mlp_sync_batch(self, model_runner: ModelRunner):
+        # TODO: The following is added to make sure sub-batch input_ids are padded
+        # to the multiple of attn_tp_size. It can likely be removed after this
+        # function is refactored and merged into the Scheduler.
+        if self.tbo_children:
diff -- python/sglang/srt/models/bailing_moe.py
@@ -582,12 +582,16 @@ def __init__(
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/communicator.py` modified +14/-1; `python/sglang/srt/model_executor/forward_batch_info.py` modified +9/-0; `python/sglang/srt/models/bailing_moe.py` modified +4/-0; `python/sglang/srt/models/falcon_h1.py` modified +3/-1; `python/sglang/srt/models/longcat_flash.py` modified +4/-0; `python/sglang/srt/models/qwen3_next.py` modified +4/-0
- Risk and verification: The diff ships test coverage in `test/srt/ep/test_deepep_small.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #15835 - [Feature] JIT Fused QK norm + qk norm clean up

- Link: https://github.com/sgl-project/sglang/pull/15835
- Status/date: merged / 2025-12-28
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 15 files, +827/-127, 1151 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feature] JIT Fused QK norm + qk norm clean up"; model line: LLaDA 2.1; category: performance/backend optimization; main diff: `python/sglang/srt/models/utils.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen3.py`; technical summary: Covers "[Feature] JIT Fused QK norm + qk norm clean up"; the main implementation surface is `python/sglang/srt/models/utils.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/utils.py` modified +80/-5 (85 lines); hunks: -11,24 +11,27; -113,6 +116,8 @@ def create_fused_set_kv_buffer_arg(; symbols: create_fused_set_kv_buffer_arg, rot_pos_ids, apply_qk_norm, touching `create_fused_set_kv_buffer_arg, rot_pos_ids, apply_qk_norm`; `python/sglang/srt/models/qwen3_moe.py` modified +9/-27 (36 lines); hunks: -57,12 +57,12; -498,31 +498,6 @@ def __init__(; symbols: __init__, _apply_qk_norm, op_prepare, apply_qk_norm_rope, touching `__init__, _apply_qk_norm, op_prepare`; `python/sglang/srt/models/qwen3.py` modified +9/-24 (33 lines); hunks: -21,14 +21,14; -138,32 +138,17 @@ def __init__(; symbols: __init__, _apply_qk_norm, forward_prepare_native, touching `__init__, _apply_qk_norm, forward_prepare_native`; `python/sglang/srt/models/bailing_moe.py` modified +9/-23 (32 lines); hunks: -75,6 +75,7; -507,28 +508,6 @@ def __init__(; symbols: __init__, _apply_qk_norm, forward, touching `__init__, _apply_qk_norm, forward`.
- Code diff details:
  - `python/sglang/srt/models/utils.py` modified +80/-5 (85 lines); hunks: -11,24 +11,27; -113,6 +116,8 @@ def create_fused_set_kv_buffer_arg(; symbols: create_fused_set_kv_buffer_arg, rot_pos_ids, apply_qk_norm
  - `python/sglang/srt/models/qwen3_moe.py` modified +9/-27 (36 lines); hunks: -57,12 +57,12; -498,31 +498,6 @@ def __init__(; symbols: __init__, _apply_qk_norm, op_prepare, apply_qk_norm_rope
  - `python/sglang/srt/models/qwen3.py` modified +9/-24 (33 lines); hunks: -21,14 +21,14; -138,32 +138,17 @@ def __init__(; symbols: __init__, _apply_qk_norm, forward_prepare_native
  - `python/sglang/srt/models/bailing_moe.py` modified +9/-23 (32 lines); hunks: -75,6 +75,7; -507,28 +508,6 @@ def __init__(; symbols: __init__, _apply_qk_norm, forward
  - `python/sglang/srt/models/glm4_moe.py` modified +9/-23 (32 lines); hunks: -75,6 +75,7; -250,28 +251,6 @@ def __init__(; symbols: __init__, _apply_qk_norm, op_prepare, forward_prepare
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/utils.py
@@ -11,24 +11,27 @@
+from __future__ import annotations
-from typing import Any, Optional
+from typing import TYPE_CHECKING, Any, Optional, Tuple
+from sglang.jit_kernel.norm import can_use_fused_inplace_qknorm, fused_inplace_qknorm
+from sglang.jit_kernel.utils import register_jit_op
+from sglang.srt.environ import envs
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -57,12 +57,12 @@
-from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
+    apply_qk_norm,
@@ -498,31 +498,6 @@ def __init__(
-    def _apply_qk_norm(
-        self, q: torch.Tensor, k: torch.Tensor
-    ) -> Tuple[torch.Tensor, torch.Tensor]:
diff -- python/sglang/srt/models/qwen3.py
@@ -21,14 +21,14 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/utils.py` modified +80/-5; `python/sglang/srt/models/qwen3_moe.py` modified +9/-27; `python/sglang/srt/models/qwen3.py` modified +9/-24; `python/sglang/srt/models/bailing_moe.py` modified +9/-23; `python/sglang/srt/models/glm4_moe.py` modified +9/-23; `python/sglang/srt/models/llada2.py` modified +9/-23
- Risk and verification: The diff ships test coverage in `python/sglang/jit_kernel/tests/test_qknorm.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #16420 - ci: migrate DLLM tests to test/registered/dllm/

- Link: https://github.com/sgl-project/sglang/pull/16420
- Status/date: merged / 2026-01-06
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +8/-2, 32 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "ci: migrate DLLM tests to test/registered/dllm/"; model line: LLaDA 2.1; category: docs/tests/CI; main diff: `test/registered/dllm/test_llada2_mini.py`, `test/registered/dllm/test_llada2_mini_amd.py`, `test/srt/run_suite.py`; technical summary: Covers "ci: migrate DLLM tests to test/registered/dllm/"; the main implementation surface is `test/registered/dllm/test_llada2_mini.py`, `test/registered/dllm/test_llada2_mini_amd.py`, `test/srt/run_suite.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/dllm/test_llada2_mini.py` renamed +4/-0 (4 lines); hunks: -1,3 +1,7; `test/registered/dllm/test_llada2_mini_amd.py` renamed +4/-0 (4 lines); hunks: -1,3 +1,7; `test/srt/run_suite.py` modified +0/-2 (2 lines); hunks: -28,7 +28,6; -156,7 +155,6.
- Code diff details:
  - `test/registered/dllm/test_llada2_mini.py` renamed +4/-0 (4 lines); hunks: -1,3 +1,7
  - `test/registered/dllm/test_llada2_mini_amd.py` renamed +4/-0 (4 lines); hunks: -1,3 +1,7
  - `test/srt/run_suite.py` modified +0/-2 (2 lines); hunks: -28,7 +28,6; -156,7 +155,6
- Key code excerpts:

```diff
diff -- test/registered/dllm/test_llada2_mini.py
@@ -1,3 +1,7 @@
+from sglang.test.ci.ci_register import register_cuda_ci
+register_cuda_ci(est_time=520, suite="stage-b-test-small-1-gpu")
diff -- test/registered/dllm/test_llada2_mini_amd.py
@@ -1,3 +1,7 @@
+from sglang.test.ci.ci_register import register_amd_ci
+register_amd_ci(est_time=520, suite="stage-b-test-small-1-gpu")
diff -- test/srt/run_suite.py
@@ -28,7 +28,6 @@
-        TestFile("dllm/test_llada2_mini.py", 520),
@@ -156,7 +155,6 @@
-        TestFile("dllm/test_llada2_mini_amd.py", 520),
```

- Reviewed files:
  - tests: `test/registered/dllm/test_llada2_mini.py` renamed +4/-0; `test/registered/dllm/test_llada2_mini_amd.py` renamed +4/-0; `test/srt/run_suite.py` modified +0/-2
- Risk and verification: The diff ships test coverage in `test/registered/dllm/test_llada2_mini.py`, `test/registered/dllm/test_llada2_mini_amd.py`, `test/srt/run_suite.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #16675 - [AMD] Fix CI - unit-test-backend-1-gpu-amd-mi35x and unit-test-backend-2-gpu-amd, stage-b-test-small-1-gpu-amd

- Link: https://github.com/sgl-project/sglang/pull/16675
- Status/date: merged / 2026-01-08
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 60 files, +106/-143, 699 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Fix CI - unit-test-backend-1-gpu-amd-mi35x and unit-test-backend-2-gpu-amd, stage-b-test-small-1-gpu-amd"; model line: LLaDA 2.1; category: bug fix; main diff: `test/registered/models/test_compressed_tensors_models.py`, `test/registered/models/test_cross_encoder_models.py`, `test/registered/models/test_embedding_models.py`; technical summary: Covers "[AMD] Fix CI - unit-test-backend-1-gpu-amd-mi35x and unit-test-backend-2-gpu-amd, stage-b-test-small-1-gpu-amd"; the main implementation surface is `test/registered/models/test_compressed_tensors_models.py`, `test/registered/models/test_cross_encoder_models.py`, `test/registered/models/test_embedding_models.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/models/test_compressed_tensors_models.py` modified +1/-1 (2 lines); hunks: -2,7 +2,7; `test/registered/models/test_cross_encoder_models.py` modified +1/-1 (2 lines); hunks: -2,7 +2,7; `test/registered/models/test_embedding_models.py` modified +1/-1 (2 lines); hunks: -4,7 +4,7; `test/registered/models/test_qwen_models.py` modified +1/-1 (2 lines); hunks: -2,7 +2,7.
- Code diff details:
  - `test/registered/models/test_compressed_tensors_models.py` modified +1/-1 (2 lines); hunks: -2,7 +2,7
  - `test/registered/models/test_cross_encoder_models.py` modified +1/-1 (2 lines); hunks: -2,7 +2,7
  - `test/registered/models/test_embedding_models.py` modified +1/-1 (2 lines); hunks: -4,7 +4,7
  - `test/registered/models/test_qwen_models.py` modified +1/-1 (2 lines); hunks: -2,7 +2,7
  - `test/registered/models/test_reward_models.py` modified +1/-1 (2 lines); hunks: -2,7 +2,7
- Key code excerpts:

```diff
diff -- test/registered/models/test_compressed_tensors_models.py
@@ -2,7 +2,7 @@
-register_amd_ci(est_time=42, suite="stage-b-test-small-1-gpu")
+register_amd_ci(est_time=42, suite="stage-b-test-small-1-gpu-amd")
diff -- test/registered/models/test_cross_encoder_models.py
@@ -2,7 +2,7 @@
-register_amd_ci(est_time=150, suite="stage-b-test-small-1-gpu")
+register_amd_ci(est_time=150, suite="stage-b-test-small-1-gpu-amd")
diff -- test/registered/models/test_embedding_models.py
@@ -4,7 +4,7 @@
-    suite="stage-b-test-small-1-gpu",
+    suite="stage-b-test-small-1-gpu-amd",
diff -- test/registered/models/test_qwen_models.py
@@ -2,7 +2,7 @@
-register_amd_ci(est_time=130, suite="stage-b-test-small-1-gpu")
+register_amd_ci(est_time=130, suite="stage-b-test-small-1-gpu-amd")
diff -- test/registered/models/test_reward_models.py
@@ -2,7 +2,7 @@
```

- Reviewed files:
  - tests: `test/registered/models/test_compressed_tensors_models.py` modified +1/-1; `test/registered/models/test_cross_encoder_models.py` modified +1/-1; `test/registered/models/test_embedding_models.py` modified +1/-1; `test/registered/models/test_qwen_models.py` modified +1/-1; `test/registered/models/test_reward_models.py` modified +1/-1; `test/registered/models/test_transformers_models.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `test/registered/attention/test_create_kvindices.py`, `test/registered/attention/test_radix_attention.py`, `test/registered/attention/test_swa_unittest.py`, `test/registered/attention/test_torch_native_attention_backend.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #16835 - Update est_time for stage-b-test-small-1-gpu tests

- Link: https://github.com/sgl-project/sglang/pull/16835
- Status/date: merged / 2026-01-11
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 24 files, +24/-24, 211 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Update est_time for stage-b-test-small-1-gpu tests"; model line: LLaDA 2.1; category: docs/tests/CI; main diff: `test/registered/models/test_encoder_embedding_models.py`, `test/registered/models/test_vlm_models.py`, `test/registered/attention/test_torch_native_attention_backend.py`; technical summary: Covers "Update est_time for stage-b-test-small-1-gpu tests"; the main implementation surface is `test/registered/models/test_encoder_embedding_models.py`, `test/registered/models/test_vlm_models.py`, `test/registered/attention/test_torch_native_attention_backend.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/models/test_encoder_embedding_models.py` modified +1/-1 (2 lines); hunks: -1,7 +1,7; `test/registered/models/test_vlm_models.py` modified +1/-1 (2 lines); hunks: -1,7 +1,7; `test/registered/attention/test_torch_native_attention_backend.py` modified +1/-1 (2 lines); hunks: -18,7 +18,7; `test/registered/backends/test_torch_compile.py` modified +1/-1 (2 lines); hunks: -16,7 +16,7.
- Code diff details:
  - `test/registered/models/test_encoder_embedding_models.py` modified +1/-1 (2 lines); hunks: -1,7 +1,7
  - `test/registered/models/test_vlm_models.py` modified +1/-1 (2 lines); hunks: -1,7 +1,7
  - `test/registered/attention/test_torch_native_attention_backend.py` modified +1/-1 (2 lines); hunks: -18,7 +18,7
  - `test/registered/backends/test_torch_compile.py` modified +1/-1 (2 lines); hunks: -16,7 +16,7
  - `test/registered/core/test_deterministic.py` modified +1/-1 (2 lines); hunks: -15,7 +15,7; symbols: TestFlashinferDeterministic
- Key code excerpts:

```diff
diff -- test/registered/models/test_encoder_embedding_models.py
@@ -1,7 +1,7 @@
-register_cuda_ci(est_time=221, suite="stage-b-test-small-1-gpu")
+register_cuda_ci(est_time=270, suite="stage-b-test-small-1-gpu")
diff -- test/registered/models/test_vlm_models.py
@@ -1,7 +1,7 @@
-register_cuda_ci(est_time=270, suite="stage-b-test-small-1-gpu")
+register_cuda_ci(est_time=228, suite="stage-b-test-small-1-gpu")
diff -- test/registered/attention/test_torch_native_attention_backend.py
@@ -18,7 +18,7 @@
-register_cuda_ci(est_time=150, suite="stage-b-test-small-1-gpu")
+register_cuda_ci(est_time=169, suite="stage-b-test-small-1-gpu")
diff -- test/registered/backends/test_torch_compile.py
@@ -16,7 +16,7 @@
-register_cuda_ci(est_time=190, suite="stage-b-test-small-1-gpu")
+register_cuda_ci(est_time=144, suite="stage-b-test-small-1-gpu")
diff -- test/registered/core/test_deterministic.py
@@ -15,7 +15,7 @@
```

- Reviewed files:
  - tests: `test/registered/models/test_encoder_embedding_models.py` modified +1/-1; `test/registered/models/test_vlm_models.py` modified +1/-1; `test/registered/attention/test_torch_native_attention_backend.py` modified +1/-1; `test/registered/backends/test_torch_compile.py` modified +1/-1; `test/registered/core/test_deterministic.py` modified +1/-1; `test/registered/core/test_gpt_oss_1gpu.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `test/registered/attention/test_torch_native_attention_backend.py`, `test/registered/backends/test_torch_compile.py`, `test/registered/core/test_deterministic.py`, `test/registered/core/test_gpt_oss_1gpu.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #16949 - [AMD CI] migrate and re-enable CI tests to new CI registry

- Link: https://github.com/sgl-project/sglang/pull/16949
- Status/date: merged / 2026-01-15
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 26 files, +86/-40, 420 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD CI] migrate and re-enable CI tests to new CI registry"; model line: LLaDA 2.1; category: docs/tests/CI; main diff: `test/registered/models/test_generation_models.py`, `test/registered/layers/mamba/test_causal_conv1d.py`, `test/registered/layers/mamba/test_mamba_ssm.py`; technical summary: Covers "[AMD CI] migrate and re-enable CI tests to new CI registry"; the main implementation surface is `test/registered/models/test_generation_models.py`, `test/registered/layers/mamba/test_causal_conv1d.py`, `test/registered/layers/mamba/test_mamba_ssm.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/models/test_generation_models.py` modified +7/-3 (10 lines); hunks: -1,7 +1,8; -28,10 +29,11; symbols: ModelCase, assert_close_logits_and_output_strs, touching `ModelCase, assert_close_logits_and_output_strs`; `test/registered/layers/mamba/test_causal_conv1d.py` modified +2/-1 (3 lines); hunks: -1,6 +1,7; `test/registered/layers/mamba/test_mamba_ssm.py` modified +2/-1 (3 lines); hunks: -1,6 +1,7; `test/registered/layers/mamba/test_mamba_ssm_ssd.py` modified +2/-1 (3 lines); hunks: -1,6 +1,7.
- Code diff details:
  - `test/registered/models/test_generation_models.py` modified +7/-3 (10 lines); hunks: -1,7 +1,8; -28,10 +29,11; symbols: ModelCase, assert_close_logits_and_output_strs
  - `test/registered/layers/mamba/test_causal_conv1d.py` modified +2/-1 (3 lines); hunks: -1,6 +1,7
  - `test/registered/layers/mamba/test_mamba_ssm.py` modified +2/-1 (3 lines); hunks: -1,6 +1,7
  - `test/registered/layers/mamba/test_mamba_ssm_ssd.py` modified +2/-1 (3 lines); hunks: -1,6 +1,7
  - `.github/workflows/pr-test-amd.yml` modified +16/-6 (22 lines); hunks: -634,10 +634,18 @@ jobs:; -646,6 +654,8 @@ jobs:
- Key code excerpts:

```diff
diff -- test/registered/models/test_generation_models.py
@@ -1,7 +1,8 @@
-from sglang.test.ci.ci_register import register_cuda_ci
+from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
+register_amd_ci(est_time=106, suite="stage-b-test-small-1-gpu-amd")
@@ -28,10 +29,11 @@
-from typing import List
+from typing import List, Optional
diff -- test/registered/layers/mamba/test_causal_conv1d.py
@@ -1,6 +1,7 @@
-from sglang.test.ci.ci_register import register_cuda_ci
+from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
+register_amd_ci(est_time=25, suite="stage-b-test-small-1-gpu-amd")
diff -- test/registered/layers/mamba/test_mamba_ssm.py
@@ -1,6 +1,7 @@
-from sglang.test.ci.ci_register import register_cuda_ci
+from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
+register_amd_ci(est_time=20, suite="stage-b-test-small-1-gpu-amd")
```

- Reviewed files:
  - tests: `test/registered/models/test_generation_models.py` modified +7/-3; `test/registered/layers/mamba/test_causal_conv1d.py` modified +2/-1; `test/registered/layers/mamba/test_mamba_ssm.py` modified +2/-1; `test/registered/layers/mamba/test_mamba_ssm_ssd.py` modified +2/-1; `test/registered/core/test_hidden_states.py` modified +9/-1; `test/registered/quant/test_torchao.py` modified +7/-2
  - ci: `.github/workflows/pr-test-amd.yml` modified +16/-6
- Risk and verification: The diff ships test coverage in `test/registered/amd/test_deepseek_r1_mxfp4_8gpu.py`, `test/registered/attention/test_mamba_unittest.py`, `test/registered/attention/test_radix_cache_unit.py`, `test/registered/core/test_hidden_states.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #16826 - [CI] Reorganize stage-b 1-GPU tests for 5090 compatibility

- Link: https://github.com/sgl-project/sglang/pull/16826
- Status/date: merged / 2026-01-15
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 136 files, +236/-363, 1885 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CI] Reorganize stage-b 1-GPU tests for 5090 compatibility"; model line: LLaDA 2.1; category: docs/tests/CI; main diff: `test/registered/models/test_encoder_embedding_models.py`, `test/registered/models/test_embedding_models.py`, `test/registered/models/test_reward_models.py`; technical summary: Covers "[CI] Reorganize stage-b 1-GPU tests for 5090 compatibility"; the main implementation surface is `test/registered/models/test_encoder_embedding_models.py`, `test/registered/models/test_embedding_models.py`, `test/registered/models/test_reward_models.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/models/test_encoder_embedding_models.py` modified +11/-11 (22 lines); hunks: -1,8 +1,16; -20,16 +28,8; `test/registered/models/test_embedding_models.py` modified +9/-11 (20 lines); hunks: -1,14 +1,3; -31,6 +20,7; `test/registered/models/test_reward_models.py` modified +9/-9 (18 lines); hunks: -1,9 +1,13; -19,13 +23,9; `test/registered/models/test_cross_encoder_models.py` modified +7/-7 (14 lines); hunks: -1,19 +1,19.
- Code diff details:
  - `test/registered/models/test_encoder_embedding_models.py` modified +11/-11 (22 lines); hunks: -1,8 +1,16; -20,16 +28,8
  - `test/registered/models/test_embedding_models.py` modified +9/-11 (20 lines); hunks: -1,14 +1,3; -31,6 +20,7
  - `test/registered/models/test_reward_models.py` modified +9/-9 (18 lines); hunks: -1,9 +1,13; -19,13 +23,9
  - `test/registered/models/test_cross_encoder_models.py` modified +7/-7 (14 lines); hunks: -1,19 +1,19
  - `test/registered/models/test_nvidia_nemotron_nano_v2_vl.py` modified +7/-6 (13 lines); hunks: -1,16 +1,17
- Key code excerpts:

```diff
diff -- test/registered/models/test_encoder_embedding_models.py
@@ -1,8 +1,16 @@
+import multiprocessing as mp
+import random
+import time
+import unittest
+import torch
+from transformers import AutoConfig, AutoTokenizer
diff -- test/registered/models/test_embedding_models.py
@@ -1,14 +1,3 @@
-from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
-# Embedding model tests
-register_cuda_ci(est_time=73, suite="stage-b-test-small-1-gpu")
-register_cuda_ci(est_time=58, suite="stage-b-test-small-1-gpu-5090")
-register_amd_ci(
-    est_time=73,
diff -- test/registered/models/test_reward_models.py
@@ -1,9 +1,13 @@
```

- Reviewed files:
  - tests: `test/registered/models/test_encoder_embedding_models.py` modified +11/-11; `test/registered/models/test_embedding_models.py` modified +9/-11; `test/registered/models/test_reward_models.py` modified +9/-9; `test/registered/models/test_cross_encoder_models.py` modified +7/-7; `test/registered/models/test_nvidia_nemotron_nano_v2_vl.py` modified +7/-6; `test/registered/models/test_vlm_models.py` modified +7/-6
- Risk and verification: The diff ships test coverage in `test/registered/attention/test_create_kvindices.py`, `test/registered/attention/test_mamba_unittest.py`, `test/registered/attention/test_radix_attention.py`, `test/registered/attention/test_radix_cache_unit.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17570 - Use attn tp group in embedding for more models

- Link: https://github.com/sgl-project/sglang/pull/17570
- Status/date: merged / 2026-01-24
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 19 files, +19/-19, 171 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Use attn tp group in embedding for more models"; model line: LLaDA 2.1; category: model implementation change; main diff: `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/bailing_moe_nextn.py`, `python/sglang/srt/models/falcon_h1.py`; technical summary: Covers "Use attn tp group in embedding for more models"; the main implementation surface is `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/bailing_moe_nextn.py`, `python/sglang/srt/models/falcon_h1.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/bailing_moe.py` modified +1/-1 (2 lines); hunks: -717,7 +717,7 @@ def __init__(; symbols: __init__, touching `__init__`; `python/sglang/srt/models/bailing_moe_nextn.py` modified +1/-1 (2 lines); hunks: -62,7 +62,7 @@ def __init__(; symbols: __init__, touching `__init__`; `python/sglang/srt/models/falcon_h1.py` modified +1/-1 (2 lines); hunks: -394,7 +394,7 @@ def __init__(; symbols: __init__, get_layer, touching `__init__, get_layer`; `python/sglang/srt/models/glm4.py` modified +1/-1 (2 lines); hunks: -307,7 +307,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/bailing_moe.py` modified +1/-1 (2 lines); hunks: -717,7 +717,7 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/bailing_moe_nextn.py` modified +1/-1 (2 lines); hunks: -62,7 +62,7 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/falcon_h1.py` modified +1/-1 (2 lines); hunks: -394,7 +394,7 @@ def __init__(; symbols: __init__, get_layer
  - `python/sglang/srt/models/glm4.py` modified +1/-1 (2 lines); hunks: -307,7 +307,7 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/glm4_moe.py` modified +1/-1 (2 lines); hunks: -895,7 +895,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/bailing_moe.py
@@ -717,7 +717,7 @@ def __init__(
-                enable_tp=not is_dp_attention_enabled(),
+                use_attn_tp_group=is_dp_attention_enabled(),
diff -- python/sglang/srt/models/bailing_moe_nextn.py
@@ -62,7 +62,7 @@ def __init__(
-            enable_tp=not is_dp_attention_enabled(),
+            use_attn_tp_group=is_dp_attention_enabled(),
diff -- python/sglang/srt/models/falcon_h1.py
@@ -394,7 +394,7 @@ def __init__(
-            enable_tp=not is_dp_attention_enabled(),
+            use_attn_tp_group=is_dp_attention_enabled(),
diff -- python/sglang/srt/models/glm4.py
@@ -307,7 +307,7 @@ def __init__(
-                enable_tp=not is_dp_attention_enabled(),
+                use_attn_tp_group=is_dp_attention_enabled(),
diff -- python/sglang/srt/models/glm4_moe.py
@@ -895,7 +895,7 @@ def __init__(
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/bailing_moe.py` modified +1/-1; `python/sglang/srt/models/bailing_moe_nextn.py` modified +1/-1; `python/sglang/srt/models/falcon_h1.py` modified +1/-1; `python/sglang/srt/models/glm4.py` modified +1/-1; `python/sglang/srt/models/glm4_moe.py` modified +1/-1; `python/sglang/srt/models/glm4_moe_lite.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/bailing_moe_nextn.py`, `python/sglang/srt/models/falcon_h1.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18423 - [AMD] Update aiter to v0.1.10.post2

- Link: https://github.com/sgl-project/sglang/pull/18423
- Status/date: merged / 2026-02-09
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 15 files, +79/-41, 391 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Update aiter to v0.1.10.post2"; model line: LLaDA 2.1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/aiter_backend.py`, `test/registered/models/test_vlm_models.py`, `scripts/ci/amd/amd_ci_warmup_aiter.py`; technical summary: Covers "[AMD] Update aiter to v0.1.10.post2"; the main implementation surface is `python/sglang/srt/layers/attention/aiter_backend.py`, `test/registered/models/test_vlm_models.py`, `scripts/ci/amd/amd_ci_warmup_aiter.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/aiter_backend.py` modified +19/-8 (27 lines); hunks: -268,6 +268,7 @@ def make_mla_meta_data(; -287,6 +288,7 @@ def make_mla_meta_data(; symbols: make_mla_meta_data, init_forward_metadata, touching `make_mla_meta_data, init_forward_metadata`; `test/registered/models/test_vlm_models.py` modified +1/-1 (2 lines); hunks: -17,7 +17,7; `scripts/ci/amd/amd_ci_warmup_aiter.py` modified +44/-17 (61 lines); hunks: -32,10 +32,12 @@ def warmup_aiter_kernels():; -44,37 +46,62 @@ def warmup_aiter_kernels():; symbols: warmup_aiter_kernels, touching `warmup_aiter_kernels`; `.github/workflows/pr-test-amd.yml` modified +3/-3 (6 lines); hunks: -251,7 +251,7 @@ jobs:; -273,7 +273,7 @@ jobs:.
- Code diff details:
  - `python/sglang/srt/layers/attention/aiter_backend.py` modified +19/-8 (27 lines); hunks: -268,6 +268,7 @@ def make_mla_meta_data(; -287,6 +288,7 @@ def make_mla_meta_data(; symbols: make_mla_meta_data, init_forward_metadata
  - `test/registered/models/test_vlm_models.py` modified +1/-1 (2 lines); hunks: -17,7 +17,7
  - `scripts/ci/amd/amd_ci_warmup_aiter.py` modified +44/-17 (61 lines); hunks: -32,10 +32,12 @@ def warmup_aiter_kernels():; -44,37 +46,62 @@ def warmup_aiter_kernels():; symbols: warmup_aiter_kernels
  - `.github/workflows/pr-test-amd.yml` modified +3/-3 (6 lines); hunks: -251,7 +251,7 @@ jobs:; -273,7 +273,7 @@ jobs:
  - `docker/rocm.Dockerfile` modified +2/-2 (4 lines); hunks: -21,7 +21,7 @@ ENV BUILD_TRITON="0"; -31,7 +31,7 @@ ENV BUILD_TRITON="0"
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/aiter_backend.py
@@ -268,6 +268,7 @@ def make_mla_meta_data(
+        kv_last_page_len,
@@ -287,6 +288,7 @@ def make_mla_meta_data(
+            kv_last_page_len,
@@ -367,6 +369,7 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):
+                        kv_last_page_len,
@@ -423,6 +426,7 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):
diff -- test/registered/models/test_vlm_models.py
@@ -17,7 +17,7 @@
-register_amd_ci(est_time=420, suite="stage-b-test-small-1-gpu-amd")
+register_amd_ci(est_time=850, suite="stage-b-test-small-1-gpu-amd")
diff -- scripts/ci/amd/amd_ci_warmup_aiter.py
@@ -32,10 +32,12 @@ def warmup_aiter_kernels():
-    # Warmup RMSNorm kernel (module_rmsnorm) - most commonly used
-    # SGLang uses rmsnorm2d_fwd and rmsnorm2d_fwd_with_add from aiter
+    # Warmup module_rmsnorm_quant (small module, ~2MB)
+    # Triggered by rmsnorm2d_fwd when hidden_size <= 8192
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/aiter_backend.py` modified +19/-8
  - tests: `test/registered/models/test_vlm_models.py` modified +1/-1; `test/registered/attention/test_triton_attention_backend.py` modified +1/-1; `test/registered/dllm/test_llada2_mini_amd.py` modified +1/-1
  - other: `scripts/ci/amd/amd_ci_warmup_aiter.py` modified +44/-17; `docker/rocm.Dockerfile` modified +2/-2; `scripts/ci/amd/amd_ci_install_dependency.sh` modified +1/-1
  - ci: `.github/workflows/pr-test-amd.yml` modified +3/-3
- Risk and verification: The diff ships test coverage in `test/registered/attention/test_triton_attention_backend.py`, `test/registered/dllm/test_llada2_mini_amd.py`, `test/registered/eval/test_eval_accuracy_large.py`, `test/registered/mla/test_mla_fp8.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17484 - [DLLM] Basic dLLM scheduling strategy and implementation

- Link: https://github.com/sgl-project/sglang/pull/17484
- Status/date: merged / 2026-02-10
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +461/-210, 911 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DLLM] Basic dLLM scheduling strategy and implementation"; model line: LLaDA 2.1; category: model implementation change; main diff: `python/sglang/srt/model_executor/forward_batch_info.py`, `python/sglang/srt/dllm/mixin/scheduler.py`, `python/sglang/srt/managers/schedule_batch.py`; technical summary: Covers "[DLLM] Basic dLLM scheduling strategy and implementation"; the main implementation surface is `python/sglang/srt/model_executor/forward_batch_info.py`, `python/sglang/srt/dllm/mixin/scheduler.py`, `python/sglang/srt/managers/schedule_batch.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/model_executor/forward_batch_info.py` modified +1/-1 (2 lines); hunks: -95,7 +95,7 @@ class ForwardMode(IntEnum):; symbols: ForwardMode, is_prefill, touching `ForwardMode, is_prefill`; `python/sglang/srt/dllm/mixin/scheduler.py` added +313/-0 (313 lines); hunks: -0,0 +1,313; symbols: SchedulerDllmMixin, init_diffusion_llm, get_new_batch_dllm, _fetch_waiting_reqs, touching `SchedulerDllmMixin, init_diffusion_llm, get_new_batch_dllm`; `python/sglang/srt/managers/schedule_batch.py` modified +4/-77 (81 lines); hunks: -58,6 +58,7; -507,7 +508,7 @@ class RequestStage(str, enum.Enum):; symbols: RequestStage, Req, __init__, touching `RequestStage, Req, __init__`; `python/sglang/srt/managers/scheduler.py` modified +19/-55 (74 lines); hunks: -58,7 +58,7; -144,7 +144,6; symbols: Scheduler, __init__, init_model_config, init_ipc_channels, touching `Scheduler, __init__, init_model_config`.
- Code diff details:
  - `python/sglang/srt/model_executor/forward_batch_info.py` modified +1/-1 (2 lines); hunks: -95,7 +95,7 @@ class ForwardMode(IntEnum):; symbols: ForwardMode, is_prefill
  - `python/sglang/srt/dllm/mixin/scheduler.py` added +313/-0 (313 lines); hunks: -0,0 +1,313; symbols: SchedulerDllmMixin, init_diffusion_llm, get_new_batch_dllm, _fetch_waiting_reqs
  - `python/sglang/srt/managers/schedule_batch.py` modified +4/-77 (81 lines); hunks: -58,6 +58,7; -507,7 +508,7 @@ class RequestStage(str, enum.Enum):; symbols: RequestStage, Req, __init__
  - `python/sglang/srt/managers/scheduler.py` modified +19/-55 (74 lines); hunks: -58,7 +58,7; -144,7 +144,6; symbols: Scheduler, __init__, init_model_config, init_ipc_channels
  - `test/registered/dllm/test_dllm_batching.py` removed +0/-71 (71 lines); hunks: -1,71 +0,0; symbols: TestBatching, setUpClass, tearDownClass, test_gsm8k
- Key code excerpts:

```diff
diff -- python/sglang/srt/model_executor/forward_batch_info.py
@@ -95,7 +95,7 @@ class ForwardMode(IntEnum):
-    # Used in diffusion LLM inference
+    # Used in dLLM
diff -- python/sglang/srt/dllm/mixin/scheduler.py
@@ -0,0 +1,313 @@
+from __future__ import annotations
+import logging
+import time
+from typing import TYPE_CHECKING, List, Optional, Set, Union
+from sglang.srt.dllm.config import DllmConfig
+from sglang.srt.dllm.mixin.req import DllmReqPhase
diff -- python/sglang/srt/managers/schedule_batch.py
@@ -58,6 +58,7 @@
+from sglang.srt.dllm.mixin.req import ReqDllmMixin
@@ -507,7 +508,7 @@ class RequestStage(str, enum.Enum):
-class Req:
+class Req(ReqDllmMixin):
```

- Reviewed files:
  - runtime: `python/sglang/srt/model_executor/forward_batch_info.py` modified +1/-1; `python/sglang/srt/dllm/mixin/scheduler.py` added +313/-0; `python/sglang/srt/managers/schedule_batch.py` modified +4/-77; `python/sglang/srt/managers/scheduler.py` modified +19/-55; `python/sglang/srt/dllm/mixin/req.py` added +67/-0; `python/sglang/srt/managers/schedule_policy.py` modified +29/-3
  - tests: `test/registered/dllm/test_dllm_batching.py` removed +0/-71
- Risk and verification: The diff ships test coverage in `test/registered/dllm/test_dllm_batching.py`, `test/registered/dllm/test_llada2_mini.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18860 - update pre-commit config

- Link: https://github.com/sgl-project/sglang/pull/18860
- Status/date: merged / 2026-02-15
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 135 files, +239/-198, 1632 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "update pre-commit config"; model line: LLaDA 2.1; category: model implementation change; main diff: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/models/pixtral.py`, `python/sglang/srt/layers/quantization/modelslim/modelslim_moe.py`; technical summary: Covers "update pre-commit config"; the main implementation surface is `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/models/pixtral.py`, `python/sglang/srt/layers/quantization/modelslim/modelslim_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +6/-6 (12 lines); hunks: -670,9 +670,9 @@ def forward_decode(; -744,9 +744,9 @@ def forward_extend(; symbols: forward_decode, forward_extend, touching `forward_decode, forward_extend`; `python/sglang/srt/models/pixtral.py` modified +6/-2 (8 lines); hunks: -23,11 +23,15; `python/sglang/srt/layers/quantization/modelslim/modelslim_moe.py` modified +2/-4 (6 lines); hunks: -63,11 +63,9 @@ def get_moe_method(; symbols: get_moe_method, touching `get_moe_method`; `test/registered/layers/test_fla_layernorm_guard.py` modified +4/-1 (5 lines); hunks: -10,7 +10,10.
- Code diff details:
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +6/-6 (12 lines); hunks: -670,9 +670,9 @@ def forward_decode(; -744,9 +744,9 @@ def forward_extend(; symbols: forward_decode, forward_extend
  - `python/sglang/srt/models/pixtral.py` modified +6/-2 (8 lines); hunks: -23,11 +23,15
  - `python/sglang/srt/layers/quantization/modelslim/modelslim_moe.py` modified +2/-4 (6 lines); hunks: -63,11 +63,9 @@ def get_moe_method(; symbols: get_moe_method
  - `test/registered/layers/test_fla_layernorm_guard.py` modified +4/-1 (5 lines); hunks: -10,7 +10,10
  - `python/sglang/srt/layers/attention/dual_chunk_flashattention_backend.py` modified +2/-2 (4 lines); hunks: -1,6 +1,6
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py
@@ -670,9 +670,9 @@ def forward_decode(
-        (q_proj_states, k_proj_states, v_proj_states) = mixed_qkv
-        (q_conv_weights, k_conv_weights, v_conv_weights) = layer.conv_weights
-        (q_conv_bias, k_conv_bias, v_conv_bias) = layer.bias
+        q_proj_states, k_proj_states, v_proj_states = mixed_qkv
+        q_conv_weights, k_conv_weights, v_conv_weights = layer.conv_weights
+        q_conv_bias, k_conv_bias, v_conv_bias = layer.bias
diff -- python/sglang/srt/models/pixtral.py
@@ -23,11 +23,15 @@
-from transformers.models.pixtral.modeling_pixtral import PixtralRotaryEmbedding
+from transformers.models.pixtral.modeling_pixtral import (
+    PixtralRotaryEmbedding,
+)
-from transformers.models.pixtral.modeling_pixtral import position_ids_in_meshgrid
+from transformers.models.pixtral.modeling_pixtral import (
diff -- python/sglang/srt/layers/quantization/modelslim/modelslim_moe.py
@@ -63,11 +63,9 @@ def get_moe_method(
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +6/-6; `python/sglang/srt/models/pixtral.py` modified +6/-2; `python/sglang/srt/layers/quantization/modelslim/modelslim_moe.py` modified +2/-4; `python/sglang/srt/layers/attention/dual_chunk_flashattention_backend.py` modified +2/-2; `python/sglang/srt/models/qwen3_next.py` modified +2/-2; `python/sglang/srt/multimodal/processors/ernie45_vl.py` modified +3/-1
  - tests: `test/registered/layers/test_fla_layernorm_guard.py` modified +4/-1
- Risk and verification: The diff ships test coverage in `sgl-kernel/tests/test_per_token_group_quant_8bit.py`, `test/manual/test_vlm_accuracy.py`, `test/registered/attention/test_triton_sliding_window.py`, `test/registered/layers/test_fla_layernorm_guard.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18844 - [Feature] rewrite rope kernel; remove flashinfer dependencies

- Link: https://github.com/sgl-project/sglang/pull/18844
- Status/date: merged / 2026-02-21
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +1147/-1099, 2459 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feature] rewrite rope kernel; remove flashinfer dependencies"; model line: LLaDA 2.1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/rotary_embedding.py`, `python/sglang/srt/models/llada2.py`, `python/sglang/srt/models/gpt_oss.py`; technical summary: Covers "[Feature] rewrite rope kernel; remove flashinfer dependencies"; the main implementation surface is `python/sglang/srt/layers/rotary_embedding.py`, `python/sglang/srt/models/llada2.py`, `python/sglang/srt/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/rotary_embedding.py` modified +14/-16 (30 lines); hunks: -5,7 +5,7; -37,13 +37,11; symbols: forward_cuda, touching `forward_cuda`; `python/sglang/srt/models/llada2.py` modified +6/-2 (8 lines); hunks: -513,6 +513,10 @@ def forward(; -523,7 +527,7 @@ def forward(; symbols: forward, touching `forward`; `python/sglang/srt/models/gpt_oss.py` modified +1/-6 (7 lines); hunks: -75,17 +75,12; symbols: GptOssConfig, touching `GptOssConfig`; `python/sglang/srt/models/utils.py` modified +2/-4 (6 lines); hunks: -119,20 +119,18 @@ def create_fused_set_kv_buffer_arg(; symbols: create_fused_set_kv_buffer_arg, touching `create_fused_set_kv_buffer_arg`.
- Code diff details:
  - `python/sglang/srt/layers/rotary_embedding.py` modified +14/-16 (30 lines); hunks: -5,7 +5,7; -37,13 +37,11; symbols: forward_cuda
  - `python/sglang/srt/models/llada2.py` modified +6/-2 (8 lines); hunks: -513,6 +513,10 @@ def forward(; -523,7 +527,7 @@ def forward(; symbols: forward
  - `python/sglang/srt/models/gpt_oss.py` modified +1/-6 (7 lines); hunks: -75,17 +75,12; symbols: GptOssConfig
  - `python/sglang/srt/models/utils.py` modified +2/-4 (6 lines); hunks: -119,20 +119,18 @@ def create_fused_set_kv_buffer_arg(; symbols: create_fused_set_kv_buffer_arg
  - `python/sglang/jit_kernel/csrc/elementwise/rope.cuh` modified +424/-616 (1040 lines); hunks: -1,655 +1,463
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/rotary_embedding.py
@@ -5,7 +5,7 @@
-from typing import Any, Dict, List, Optional, Tuple, Union
+from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
@@ -37,13 +37,11 @@
+if TYPE_CHECKING:
+    from sglang.jit_kernel.rope import FusedSetKVBufferArg  # For type check-only
-    from sglang.jit_kernel.rope import (
diff -- python/sglang/srt/models/llada2.py
@@ -513,6 +513,10 @@ def forward(
+        can_fuse_set_kv = (
+            self.head_dim == self.rotary_emb.rotary_dim
+            and enable_fused_set_kv_buffer(forward_batch)
+        )
@@ -523,7 +527,7 @@ def forward(
-                if enable_fused_set_kv_buffer(forward_batch)
diff -- python/sglang/srt/models/gpt_oss.py
@@ -75,17 +75,12 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/rotary_embedding.py` modified +14/-16; `python/sglang/srt/models/llada2.py` modified +6/-2; `python/sglang/srt/models/gpt_oss.py` modified +1/-6; `python/sglang/srt/models/utils.py` modified +2/-4; `python/sglang/jit_kernel/csrc/elementwise/rope.cuh` modified +424/-616; `python/sglang/jit_kernel/benchmark/bench_rope.py` added +350/-0
  - tests: `python/sglang/jit_kernel/tests/test_rope.py` modified +212/-269
- Risk and verification: The diff ships test coverage in `python/sglang/jit_kernel/tests/test_rope.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18724 - [DLLM] Add initial radix cache support

- Link: https://github.com/sgl-project/sglang/pull/18724
- Status/date: merged / 2026-03-05
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +84/-57, 201 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DLLM] Add initial radix cache support"; model line: LLaDA 2.1; category: model support/runtime entry; main diff: `python/sglang/srt/dllm/mixin/scheduler.py`, `python/sglang/srt/managers/scheduler_output_processor_mixin.py`, `python/sglang/srt/dllm/mixin/req.py`; technical summary: Covers "[DLLM] Add initial radix cache support"; the main implementation surface is `python/sglang/srt/dllm/mixin/scheduler.py`, `python/sglang/srt/managers/scheduler_output_processor_mixin.py`, `python/sglang/srt/dllm/mixin/req.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/dllm/mixin/scheduler.py` modified +42/-1 (43 lines); hunks: -7,13 +7,14; -59,6 +60,46 @@ def get_new_batch_dllm(self: Scheduler) -> Optional[ScheduleB...; symbols: SchedulerDllmMixin, get_new_batch_dllm, process_batch_result_dllm, _fetch_waiting_reqs, touching `SchedulerDllmMixin, get_new_batch_dllm, process_batch_result_dllm`; `python/sglang/srt/managers/scheduler_output_processor_mixin.py` modified +0/-40 (40 lines); hunks: -355,46 +355,6 @@ def process_batch_result_idle(; symbols: process_batch_result_idle, process_batch_result_dllm, process_batch_result_decode, touching `process_batch_result_idle, process_batch_result_dllm, process_batch_result_decode`; `python/sglang/srt/dllm/mixin/req.py` modified +18/-11 (29 lines); hunks: -19,7 +19,6 @@ class DllmReqPhase(str, enum.Enum):; -55,13 +54,21 @@ def determine_dllm_phase(self: Req):; symbols: DllmReqPhase, ReqDllmMixin, init_diffusion_llm, determine_dllm_phase, touching `DllmReqPhase, ReqDllmMixin, init_diffusion_llm`; `python/sglang/srt/server_args.py` modified +20/-4 (24 lines); hunks: -2824,11 +2824,27 @@ def _handle_dllm_inference(self):; symbols: _handle_dllm_inference, touching `_handle_dllm_inference`.
- Code diff details:
  - `python/sglang/srt/dllm/mixin/scheduler.py` modified +42/-1 (43 lines); hunks: -7,13 +7,14; -59,6 +60,46 @@ def get_new_batch_dllm(self: Scheduler) -> Optional[ScheduleB...; symbols: SchedulerDllmMixin, get_new_batch_dllm, process_batch_result_dllm, _fetch_waiting_reqs
  - `python/sglang/srt/managers/scheduler_output_processor_mixin.py` modified +0/-40 (40 lines); hunks: -355,46 +355,6 @@ def process_batch_result_idle(; symbols: process_batch_result_idle, process_batch_result_dllm, process_batch_result_decode
  - `python/sglang/srt/dllm/mixin/req.py` modified +18/-11 (29 lines); hunks: -19,7 +19,6 @@ class DllmReqPhase(str, enum.Enum):; -55,13 +54,21 @@ def determine_dllm_phase(self: Req):; symbols: DllmReqPhase, ReqDllmMixin, init_diffusion_llm, determine_dllm_phase
  - `python/sglang/srt/server_args.py` modified +20/-4 (24 lines); hunks: -2824,11 +2824,27 @@ def _handle_dllm_inference(self):; symbols: _handle_dllm_inference
  - `python/sglang/srt/managers/schedule_batch.py` modified +3/-0 (3 lines); hunks: -892,6 +892,9 @@ def init_next_round_input(self, tree_cache: Optional[BasePre...; symbols: init_next_round_input
- Key code excerpts:

```diff
diff -- python/sglang/srt/dllm/mixin/scheduler.py
@@ -7,13 +7,14 @@
+from sglang.srt.mem_cache.common import release_kv_cache
-    from sglang.srt.managers.scheduler import Scheduler
+    from sglang.srt.managers.scheduler import GenerationBatchResult, Scheduler
@@ -59,6 +60,46 @@ def get_new_batch_dllm(self: Scheduler) -> Optional[ScheduleBatch]:
+    def process_batch_result_dllm(
+        self: Scheduler,
diff -- python/sglang/srt/managers/scheduler_output_processor_mixin.py
@@ -355,46 +355,6 @@ def process_batch_result_idle(
-    def process_batch_result_dllm(
-        self: Scheduler,
-        batch: ScheduleBatch,
-        result: GenerationBatchResult,
-    ):
-        if result.copy_done is not None:
diff -- python/sglang/srt/dllm/mixin/req.py
@@ -19,7 +19,6 @@ class DllmReqPhase(str, enum.Enum):
```

- Reviewed files:
  - runtime: `python/sglang/srt/dllm/mixin/scheduler.py` modified +42/-1; `python/sglang/srt/managers/scheduler_output_processor_mixin.py` modified +0/-40; `python/sglang/srt/dllm/mixin/req.py` modified +18/-11; `python/sglang/srt/server_args.py` modified +20/-4; `python/sglang/srt/managers/schedule_batch.py` modified +3/-0
  - tests: `test/registered/dllm/test_llada2_mini.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `test/registered/dllm/test_llada2_mini.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18485 - [NPU] [DLLM]DLLM LLaDA2.x graph mode support with NPU speedup modifications

- Link: https://github.com/sgl-project/sglang/pull/18485
- Status/date: merged / 2026-03-09
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/llada2.py`; associated commits `11b76d24dc11`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +250/-9, 400 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NPU] [DLLM]DLLM LLaDA2.x graph mode support with NPU speedup modifications"; model line: LLaDA 2.1; category: model support/runtime entry; main diff: `python/sglang/srt/models/llada2.py`; technical summary: Covers "[NPU] [DLLM]DLLM LLaDA2.x graph mode support with NPU speedup modifications"; the main implementation surface is `python/sglang/srt/models/llada2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/llada2.py` modified +13/-1 (14 lines); hunks: -77,11 +77,18; -190,6 +197,11 @@ def __init__(; symbols: LLaDA2MoeMLP, __init__, touching `LLaDA2MoeMLP, __init__`.
- Code diff details:
  - `python/sglang/srt/models/llada2.py` modified +13/-1 (14 lines); hunks: -77,11 +77,18; -190,6 +197,11 @@ def __init__(; symbols: LLaDA2MoeMLP, __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/llada2.py
@@ -77,11 +77,18 @@
-from sglang.srt.utils import add_prefix, is_cuda, is_non_idle_and_non_empty, make_layers
+from sglang.srt.utils import (
+    add_prefix,
+    is_cuda,
+    is_non_idle_and_non_empty,
+    is_npu,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/llada2.py` modified +13/-1
- Risk and verification: The diff ships test coverage in `test/srt/ascend/test_llada2_mini_ascend.py`, `test/srt/run_suite.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17784 - Upgrade transformers==5.3.0

- Link: https://github.com/sgl-project/sglang/pull/17784
- Status/date: merged / 2026-03-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 95 files, +1136/-343, 2752 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Upgrade transformers==5.3.0"; model line: LLaDA 2.1; category: model implementation change; main diff: `python/sglang/srt/models/gemma3_causal.py`, `python/sglang/srt/layers/rotary_embedding/factory.py`, `python/sglang/srt/configs/model_config.py`; technical summary: Covers "Upgrade transformers==5.3.0"; the main implementation surface is `python/sglang/srt/models/gemma3_causal.py`, `python/sglang/srt/layers/rotary_embedding/factory.py`, `python/sglang/srt/configs/model_config.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/gemma3_causal.py` modified +87/-14 (101 lines); hunks: -166,18 +166,36 @@ def __init__(; -325,9 +343,10 @@ class Gemma3RotaryEmbedding(nn.Module):; symbols: __init__, Gemma3RotaryEmbedding, _dynamic_frequency_update, touching `__init__, Gemma3RotaryEmbedding, _dynamic_frequency_update`; `python/sglang/srt/layers/rotary_embedding/factory.py` modified +63/-13 (76 lines); hunks: -2,6 +2,7; -26,6 +27,29; symbols: _get_rope_param, get_rope, touching `_get_rope_param, get_rope`; `python/sglang/srt/configs/model_config.py` modified +38/-18 (56 lines); hunks: -51,10 +51,20 @@ class ModelImpl(str, Enum):; -63,7 +73,7 @@ def is_deepseek_nsa(config: PretrainedConfig) -> bool:; symbols: ModelImpl, is_deepseek_nsa, _derive_model_shapes, touching `ModelImpl, is_deepseek_nsa, _derive_model_shapes`; `python/sglang/srt/models/qwen3_moe.py` modified +14/-7 (21 lines); hunks: -115,12 +115,19 @@ def compute_yarn_parameters(; -130,7 +137,7 @@ def compute_yarn_parameters(; symbols: compute_yarn_parameters, forward_prepare_native, apply_qk_norm_rope, __init__, touching `compute_yarn_parameters, forward_prepare_native, apply_qk_norm_rope`.
- Code diff details:
  - `python/sglang/srt/models/gemma3_causal.py` modified +87/-14 (101 lines); hunks: -166,18 +166,36 @@ def __init__(; -325,9 +343,10 @@ class Gemma3RotaryEmbedding(nn.Module):; symbols: __init__, Gemma3RotaryEmbedding, _dynamic_frequency_update
  - `python/sglang/srt/layers/rotary_embedding/factory.py` modified +63/-13 (76 lines); hunks: -2,6 +2,7; -26,6 +27,29; symbols: _get_rope_param, get_rope
  - `python/sglang/srt/configs/model_config.py` modified +38/-18 (56 lines); hunks: -51,10 +51,20 @@ class ModelImpl(str, Enum):; -63,7 +73,7 @@ def is_deepseek_nsa(config: PretrainedConfig) -> bool:; symbols: ModelImpl, is_deepseek_nsa, _derive_model_shapes
  - `python/sglang/srt/models/qwen3_moe.py` modified +14/-7 (21 lines); hunks: -115,12 +115,19 @@ def compute_yarn_parameters(; -130,7 +137,7 @@ def compute_yarn_parameters(; symbols: compute_yarn_parameters, forward_prepare_native, apply_qk_norm_rope, __init__
  - `python/sglang/srt/models/midashenglm.py` modified +6/-14 (20 lines); hunks: -476,20 +476,12 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/gemma3_causal.py
@@ -166,18 +166,36 @@ def __init__(
+        # In transformers v5, rope_parameters is nested per layer type:
+        #   {"sliding_attention": {"rope_theta": 10000}, "full_attention": {"rope_theta": 1000000}}
+        # In v4 it was flat: {"rope_type": "default", "rope_theta": ...}
+        rope_params = config.rope_parameters
+        is_nested = isinstance(rope_params, dict) and "full_attention" in rope_params
-            self.rope_theta = config.rope_local_base_freq
diff -- python/sglang/srt/layers/rotary_embedding/factory.py
@@ -2,6 +2,7 @@
+import logging
@@ -26,6 +27,29 @@
+logger = logging.getLogger(__name__)
+def _get_rope_param(rope_scaling, key, default, scaling_type):
+    """Get a parameter from rope_scaling dict, warn if missing.
+    In transformers v5, config.rope_scaling is an alias for rope_parameters
diff -- python/sglang/srt/configs/model_config.py
@@ -51,10 +51,20 @@ class ModelImpl(str, Enum):
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/gemma3_causal.py` modified +87/-14; `python/sglang/srt/layers/rotary_embedding/factory.py` modified +63/-13; `python/sglang/srt/configs/model_config.py` modified +38/-18; `python/sglang/srt/models/qwen3_moe.py` modified +14/-7; `python/sglang/srt/models/midashenglm.py` modified +6/-14; `python/sglang/srt/models/glm4.py` modified +3/-14
- Risk and verification: The diff ships test coverage in `python/sglang/test/runners.py`, `test/registered/core/test_score_api.py`, `test/registered/quant/test_awq.py`, `test/registered/rl/test_multi_instance_release_memory_occupation.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #21187 - ci: unify PR test suite naming

- Link: https://github.com/sgl-project/sglang/pull/21187
- Status/date: merged / 2026-03-23
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 283 files, +554/-554, 3558 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "ci: unify PR test suite naming"; model line: LLaDA 2.1; category: docs/tests/CI; main diff: `test/registered/layers/mamba/test_causal_conv1d.py`, `test/registered/layers/mamba/test_mamba_ssm.py`, `test/registered/layers/mamba/test_mamba_ssm_ssd.py`; technical summary: Covers "ci: unify PR test suite naming"; the main implementation surface is `test/registered/layers/mamba/test_causal_conv1d.py`, `test/registered/layers/mamba/test_mamba_ssm.py`, `test/registered/layers/mamba/test_mamba_ssm_ssd.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/layers/mamba/test_causal_conv1d.py` modified +2/-2 (4 lines); hunks: -1,7 +1,7; `test/registered/layers/mamba/test_mamba_ssm.py` modified +2/-2 (4 lines); hunks: -1,7 +1,7; `test/registered/layers/mamba/test_mamba_ssm_ssd.py` modified +2/-2 (4 lines); hunks: -1,7 +1,7; `test/registered/models/test_compressed_tensors_models.py` modified +2/-2 (4 lines); hunks: -13,8 +13,8; symbols: TestCompressedTensorsLlama3FP8, touching `TestCompressedTensorsLlama3FP8`.
- Code diff details:
  - `test/registered/layers/mamba/test_causal_conv1d.py` modified +2/-2 (4 lines); hunks: -1,7 +1,7
  - `test/registered/layers/mamba/test_mamba_ssm.py` modified +2/-2 (4 lines); hunks: -1,7 +1,7
  - `test/registered/layers/mamba/test_mamba_ssm_ssd.py` modified +2/-2 (4 lines); hunks: -1,7 +1,7
  - `test/registered/models/test_compressed_tensors_models.py` modified +2/-2 (4 lines); hunks: -13,8 +13,8; symbols: TestCompressedTensorsLlama3FP8
  - `test/registered/models/test_cross_encoder_models.py` modified +2/-2 (4 lines); hunks: -11,8 +11,8
- Key code excerpts:

```diff
diff -- test/registered/layers/mamba/test_causal_conv1d.py
@@ -1,7 +1,7 @@
-register_cuda_ci(est_time=25, suite="stage-b-test-small-1-gpu")
-register_amd_ci(est_time=25, suite="stage-b-test-small-1-gpu-amd")
+register_cuda_ci(est_time=25, suite="stage-b-test-1-gpu-small")
+register_amd_ci(est_time=25, suite="stage-b-test-1-gpu-small-amd")
diff -- test/registered/layers/mamba/test_mamba_ssm.py
@@ -1,7 +1,7 @@
-register_cuda_ci(est_time=7, suite="stage-b-test-small-1-gpu")
-register_amd_ci(est_time=20, suite="stage-b-test-small-1-gpu-amd")
+register_cuda_ci(est_time=7, suite="stage-b-test-1-gpu-small")
+register_amd_ci(est_time=20, suite="stage-b-test-1-gpu-small-amd")
diff -- test/registered/layers/mamba/test_mamba_ssm_ssd.py
@@ -1,7 +1,7 @@
-register_cuda_ci(est_time=15, suite="stage-b-test-small-1-gpu")
-register_amd_ci(est_time=34, suite="stage-b-test-small-1-gpu-amd")
+register_cuda_ci(est_time=15, suite="stage-b-test-1-gpu-small")
+register_amd_ci(est_time=34, suite="stage-b-test-1-gpu-small-amd")
```

- Reviewed files:
  - tests: `test/registered/layers/mamba/test_causal_conv1d.py` modified +2/-2; `test/registered/layers/mamba/test_mamba_ssm.py` modified +2/-2; `test/registered/layers/mamba/test_mamba_ssm_ssd.py` modified +2/-2; `test/registered/models/test_compressed_tensors_models.py` modified +2/-2; `test/registered/models/test_cross_encoder_models.py` modified +2/-2; `test/registered/models/test_generation_models.py` modified +2/-2
- Risk and verification: The diff ships test coverage in `python/sglang/jit_kernel/tests/test_moe_lora_align_block_size.py`, `test/README.md`, `test/registered/amd/disaggregation/test_disaggregation_basic.py`, `test/registered/attention/test_chunk_gated_delta_rule.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #21135 - fix: use get_rope_config() to support models without rope_parameters

- Link: https://github.com/sgl-project/sglang/pull/21135
- Status/date: merged / 2026-03-26
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 18 files, +44/-42, 342 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix: use get_rope_config() to support models without rope_parameters"; model line: LLaDA 2.1; category: bug fix; main diff: `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4.py`, `python/sglang/srt/models/grok.py`; technical summary: Covers "fix: use get_rope_config() to support models without rope_parameters"; the main implementation surface is `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4.py`, `python/sglang/srt/models/grok.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/glm4_moe.py` modified +5/-5 (10 lines); hunks: -94,6 +94,7; -684,11 +685,10 @@ def __init__(; symbols: __init__, touching `__init__`; `python/sglang/srt/models/glm4.py` modified +5/-3 (8 lines); hunks: -52,6 +52,7; -217,9 +218,10 @@ def __init__(; symbols: __init__, touching `__init__`; `python/sglang/srt/models/grok.py` modified +2/-5 (7 lines); hunks: -61,6 +61,7; -477,11 +478,7 @@ def __init__(; symbols: __init__, touching `__init__`; `python/sglang/srt/models/llada2.py` modified +4/-2 (6 lines); hunks: -84,6 +84,7; -486,12 +487,13 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/glm4_moe.py` modified +5/-5 (10 lines); hunks: -94,6 +94,7; -684,11 +685,10 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/glm4.py` modified +5/-3 (8 lines); hunks: -52,6 +52,7; -217,9 +218,10 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/grok.py` modified +2/-5 (7 lines); hunks: -61,6 +61,7; -477,11 +478,7 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/llada2.py` modified +4/-2 (6 lines); hunks: -84,6 +84,7; -486,12 +487,13 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/deepseek.py` modified +2/-2 (4 lines); hunks: -49,6 +49,7; -310,8 +311,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/glm4_moe.py
@@ -94,6 +94,7 @@
+from sglang.srt.utils.hf_transformers_utils import get_rope_config
@@ -684,11 +685,10 @@ def __init__(
-        rope_theta = config.rope_parameters["rope_theta"]
-        rope_scaling = config.rope_parameters
-        partial_rotary_factor = getattr(
-            getattr(config, "rope_parameters", None), "partial_rotary_factor", None
diff -- python/sglang/srt/models/glm4.py
@@ -52,6 +52,7 @@
+from sglang.srt.utils.hf_transformers_utils import get_rope_config
@@ -217,9 +218,10 @@ def __init__(
-        rope_theta = config.rope_parameters["rope_theta"]
-        rope_scaling = config.rope_parameters
-        partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 0.5)
+        rope_theta, rope_scaling = get_rope_config(config)
diff -- python/sglang/srt/models/grok.py
@@ -61,6 +61,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/glm4_moe.py` modified +5/-5; `python/sglang/srt/models/glm4.py` modified +5/-3; `python/sglang/srt/models/grok.py` modified +2/-5; `python/sglang/srt/models/llada2.py` modified +4/-2; `python/sglang/srt/models/deepseek.py` modified +2/-2; `python/sglang/srt/models/ernie4.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/baichuan.py`, `python/sglang/srt/models/deepseek.py`, `python/sglang/srt/models/ernie4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #20751 - [NPU]Add a full test pipeline on NPU, resolve issues in the NPU test architecture

- Link: https://github.com/sgl-project/sglang/pull/20751
- Status/date: merged / 2026-04-01
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 43 files, +673/-106, 1465 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NPU]Add a full test pipeline on NPU, resolve issues in the NPU test architecture"; model line: LLaDA 2.1; category: docs/tests/CI; main diff: `.github/workflows/full-test-npu.yml`, `.github/workflows/nightly-test-npu.yml`, `.github/workflows/pr-test-npu.yml`; technical summary: Covers "[NPU]Add a full test pipeline on NPU, resolve issues in the NPU test architecture"; the main implementation surface is `.github/workflows/full-test-npu.yml`, `.github/workflows/nightly-test-npu.yml`, `.github/workflows/pr-test-npu.yml`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `.github/workflows/full-test-npu.yml` added +355/-0 (355 lines); hunks: -0,0 +1,355; `.github/workflows/nightly-test-npu.yml` modified +124/-36 (160 lines); hunks: -2,7 +2,7 @@ name: Nightly Test (NPU); -21,40 +21,95 @@ on:; `.github/workflows/pr-test-npu.yml` modified +70/-40 (110 lines); hunks: -76,7 +76,7 @@ jobs:; -111,21 +111,8 @@ jobs:; `python/sglang/test/ascend/test_ascend_utils.py` modified +9/-9 (18 lines); hunks: -117,9 +117,18; -133,15 +142,6.
- Code diff details:
  - `.github/workflows/full-test-npu.yml` added +355/-0 (355 lines); hunks: -0,0 +1,355
  - `.github/workflows/nightly-test-npu.yml` modified +124/-36 (160 lines); hunks: -2,7 +2,7 @@ name: Nightly Test (NPU); -21,40 +21,95 @@ on:
  - `.github/workflows/pr-test-npu.yml` modified +70/-40 (110 lines); hunks: -76,7 +76,7 @@ jobs:; -111,21 +111,8 @@ jobs:
  - `python/sglang/test/ascend/test_ascend_utils.py` modified +9/-9 (18 lines); hunks: -117,9 +117,18; -133,15 +142,6
  - `test/registered/ascend/basic_function/quant/test_npu_autoround_moe.py` renamed +8/-1 (9 lines); hunks: -4,6 +4,10; -12,10 +16,13
- Key code excerpts:

```diff
diff -- .github/workflows/full-test-npu.yml
@@ -0,0 +1,355 @@
+name: Full Test (NPU)
+on:
+#  pull_request:
+#    branches:
+#      - main
+#    paths:
diff -- .github/workflows/nightly-test-npu.yml
@@ -2,7 +2,7 @@ name: Nightly Test (NPU)
-    - cron: '0 17 * * *'  # Execute at 1:00 a.m. Beijing Time every day
+    - cron: '0 18 * * *'  # Execute at 2:00 a.m. Beijing Time every day
@@ -21,40 +21,95 @@ on:
+      image_a3:
+        description: 'The a3 running docker image of the test task.'
+        required: false
diff -- .github/workflows/pr-test-npu.yml
@@ -76,7 +76,7 @@ jobs:
```

- Reviewed files:
  - ci: `.github/workflows/full-test-npu.yml` added +355/-0; `.github/workflows/nightly-test-npu.yml` modified +124/-36; `.github/workflows/pr-test-npu.yml` modified +70/-40
  - tests: `python/sglang/test/ascend/test_ascend_utils.py` modified +9/-9; `test/registered/ascend/basic_function/quant/test_npu_autoround_moe.py` renamed +8/-1; `test/registered/ascend/basic_function/quant/test_npu_gptq_moe.py` renamed +8/-1; `test/registered/ascend/basic_function/quant/test_npu_autoround_dense.py` renamed +6/-1; `test/run_suite.py` modified +5/-0
- Risk and verification: The diff ships test coverage in `python/sglang/test/ascend/test_ascend_utils.py`, `test/registered/ascend/basic_function/HiCache/test_npu_hicache_mha.py`, `test/registered/ascend/basic_function/HiCache/test_npu_hicache_mla.py`, `test/registered/ascend/basic_function/backends/test_npu_sampling_backend.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #21667 - Unify GSM8K eval path to Chat API for regression CI readiness

- Link: https://github.com/sgl-project/sglang/pull/21667
- Status/date: merged / 2026-04-02
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 79 files, +1349/-1359, 5014 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Unify GSM8K eval path to Chat API for regression CI readiness"; model line: LLaDA 2.1; category: bug fix; main diff: `test/manual/models/test_unsloth_models.py`, `test/manual/models/test_falcon_h1_models.py`, `test/registered/models/test_qwen_models.py`; technical summary: Covers "Unify GSM8K eval path to Chat API for regression CI readiness"; the main implementation surface is `test/manual/models/test_unsloth_models.py`, `test/manual/models/test_falcon_h1_models.py`, `test/registered/models/test_qwen_models.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/manual/models/test_unsloth_models.py` modified +49/-49 (98 lines); hunks: -2,7 +2,7; -29,17 +29,17 @@ def tearDownClass(cls):; symbols: tearDownClass, test_gsm8k, TestUnslothPhi4Bnb4bit, touching `tearDownClass, test_gsm8k, TestUnslothPhi4Bnb4bit`; `test/manual/models/test_falcon_h1_models.py` modified +33/-33 (66 lines); hunks: -1,7 +1,7; -31,17 +31,17 @@ def tearDownClass(cls):; symbols: tearDownClass, test_gsm8k, TestFalconH1TP4, touching `tearDownClass, test_gsm8k, TestFalconH1TP4`; `test/registered/models/test_qwen_models.py` modified +17/-17 (34 lines); hunks: -5,7 +5,7; -35,17 +35,17 @@ def tearDownClass(cls):; symbols: tearDownClass, test_gsm8k, TestQwen2FP8, touching `tearDownClass, test_gsm8k, TestQwen2FP8`; `test/manual/models/test_kimi_k2_models.py` modified +11/-11 (22 lines); hunks: -4,7 +4,7; -48,22 +48,22 @@ def test_a_gsm8k(; symbols: test_a_gsm8k, touching `test_a_gsm8k`.
- Code diff details:
  - `test/manual/models/test_unsloth_models.py` modified +49/-49 (98 lines); hunks: -2,7 +2,7; -29,17 +29,17 @@ def tearDownClass(cls):; symbols: tearDownClass, test_gsm8k, TestUnslothPhi4Bnb4bit
  - `test/manual/models/test_falcon_h1_models.py` modified +33/-33 (66 lines); hunks: -1,7 +1,7; -31,17 +31,17 @@ def tearDownClass(cls):; symbols: tearDownClass, test_gsm8k, TestFalconH1TP4
  - `test/registered/models/test_qwen_models.py` modified +17/-17 (34 lines); hunks: -5,7 +5,7; -35,17 +35,17 @@ def tearDownClass(cls):; symbols: tearDownClass, test_gsm8k, TestQwen2FP8
  - `test/manual/models/test_kimi_k2_models.py` modified +11/-11 (22 lines); hunks: -4,7 +4,7; -48,22 +48,22 @@ def test_a_gsm8k(; symbols: test_a_gsm8k
  - `test/manual/models/test_mistral_large3_basic.py` modified +11/-10 (21 lines); hunks: -3,7 +3,7; -53,22 +53,23 @@ def test_a_gsm8k(; symbols: test_a_gsm8k, test_bs_1_speed
- Key code excerpts:

```diff
diff -- test/manual/models/test_unsloth_models.py
@@ -2,7 +2,7 @@
-from sglang.test.few_shot_gsm8k import run_eval
+from sglang.test.run_eval import run_eval
@@ -29,17 +29,17 @@ def tearDownClass(cls):
-            num_shots=5,
-            data_path=None,
-            num_questions=200,
diff -- test/manual/models/test_falcon_h1_models.py
@@ -1,7 +1,7 @@
-from sglang.test.few_shot_gsm8k import run_eval
+from sglang.test.run_eval import run_eval
@@ -31,17 +31,17 @@ def tearDownClass(cls):
-            num_shots=5,
-            data_path=None,
-            num_questions=200,
diff -- test/registered/models/test_qwen_models.py
@@ -5,7 +5,7 @@
```

- Reviewed files:
  - tests: `test/manual/models/test_unsloth_models.py` modified +49/-49; `test/manual/models/test_falcon_h1_models.py` modified +33/-33; `test/registered/models/test_qwen_models.py` modified +17/-17; `test/manual/models/test_kimi_k2_models.py` modified +11/-11; `test/manual/models/test_mistral_large3_basic.py` modified +11/-10; `test/registered/models/test_transformers_models.py` modified +9/-12
- Risk and verification: The diff ships test coverage in `python/sglang/test/accuracy_test_runner.py`, `python/sglang/test/few_shot_gsm8k.py`, `python/sglang/test/few_shot_gsm8k_engine.py`, `python/sglang/test/kits/eval_accuracy_kit.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22305 - [CI] Update est_time for 64 tests based on actual elapsed times

- Link: https://github.com/sgl-project/sglang/pull/22305
- Status/date: merged / 2026-04-10
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 61 files, +61/-61, 546 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CI] Update est_time for 64 tests based on actual elapsed times"; model line: LLaDA 2.1; category: docs/tests/CI; main diff: `test/registered/models/test_nvidia_nemotron_3_nano.py`, `test/registered/models/test_transformers_backend_eval.py`, `test/registered/models/test_transformers_models.py`; technical summary: Covers "[CI] Update est_time for 64 tests based on actual elapsed times"; the main implementation surface is `test/registered/models/test_nvidia_nemotron_3_nano.py`, `test/registered/models/test_transformers_backend_eval.py`, `test/registered/models/test_transformers_models.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/models/test_nvidia_nemotron_3_nano.py` modified +1/-1 (2 lines); hunks: -4,7 +4,7; `test/registered/models/test_transformers_backend_eval.py` modified +1/-1 (2 lines); hunks: -7,7 +7,7; symbols: TestTransformersBackendEval, touching `TestTransformersBackendEval`; `test/registered/models/test_transformers_models.py` modified +1/-1 (2 lines); hunks: -21,7 +21,7; `test/registered/openai_server/function_call/test_anthropic_tool_use.py` modified +1/-1 (2 lines); hunks: -25,7 +25,7.
- Code diff details:
  - `test/registered/models/test_nvidia_nemotron_3_nano.py` modified +1/-1 (2 lines); hunks: -4,7 +4,7
  - `test/registered/models/test_transformers_backend_eval.py` modified +1/-1 (2 lines); hunks: -7,7 +7,7; symbols: TestTransformersBackendEval
  - `test/registered/models/test_transformers_models.py` modified +1/-1 (2 lines); hunks: -21,7 +21,7
  - `test/registered/openai_server/function_call/test_anthropic_tool_use.py` modified +1/-1 (2 lines); hunks: -25,7 +25,7
  - `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py` modified +1/-1 (2 lines); hunks: -11,7 +11,7
- Key code excerpts:

```diff
diff -- test/registered/models/test_nvidia_nemotron_3_nano.py
@@ -4,7 +4,7 @@
-register_cuda_ci(est_time=660, suite="stage-b-test-2-gpu-large")
+register_cuda_ci(est_time=540, suite="stage-b-test-2-gpu-large")
diff -- test/registered/models/test_transformers_backend_eval.py
@@ -7,7 +7,7 @@
-register_cuda_ci(est_time=180, suite="stage-b-test-1-gpu-small")
+register_cuda_ci(est_time=40, suite="stage-b-test-1-gpu-small")
diff -- test/registered/models/test_transformers_models.py
@@ -21,7 +21,7 @@
-register_cuda_ci(est_time=450, suite="stage-b-test-1-gpu-small")
+register_cuda_ci(est_time=260, suite="stage-b-test-1-gpu-small")
diff -- test/registered/openai_server/function_call/test_anthropic_tool_use.py
@@ -25,7 +25,7 @@
-register_cuda_ci(est_time=120, suite="stage-b-test-1-gpu-large")
+register_cuda_ci(est_time=50, suite="stage-b-test-1-gpu-large")
diff -- test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py
@@ -11,7 +11,7 @@
```

- Reviewed files:
  - tests: `test/registered/models/test_nvidia_nemotron_3_nano.py` modified +1/-1; `test/registered/models/test_transformers_backend_eval.py` modified +1/-1; `test/registered/models/test_transformers_models.py` modified +1/-1; `test/registered/openai_server/function_call/test_anthropic_tool_use.py` modified +1/-1; `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py` modified +1/-1; `test/registered/4-gpu-models/test_qwen35_hicache.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`, `test/registered/4-gpu-models/test_qwen35_hicache.py`, `test/registered/4-gpu-models/test_qwen3_30b.py`, `test/registered/4-gpu-models/test_qwen3_next_models.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22565 - chore: update CI test est_time values

- Link: https://github.com/sgl-project/sglang/pull/22565
- Status/date: merged / 2026-04-11
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 250 files, +251/-251, 2240 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "chore: update CI test est_time values"; model line: LLaDA 2.1; category: docs/tests/CI; main diff: `test/registered/layers/mamba/test_causal_conv1d.py`, `test/registered/layers/mamba/test_mamba2_mixer.py`, `test/registered/layers/mamba/test_mamba_ssm.py`; technical summary: Covers "chore: update CI test est_time values"; the main implementation surface is `test/registered/layers/mamba/test_causal_conv1d.py`, `test/registered/layers/mamba/test_mamba2_mixer.py`, `test/registered/layers/mamba/test_mamba_ssm.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/layers/mamba/test_causal_conv1d.py` modified +1/-1 (2 lines); hunks: -1,6 +1,6; `test/registered/layers/mamba/test_mamba2_mixer.py` modified +1/-1 (2 lines); hunks: -15,7 +15,7; `test/registered/layers/mamba/test_mamba_ssm.py` modified +1/-1 (2 lines); hunks: -1,6 +1,6; `test/registered/layers/mamba/test_mamba_ssm_ssd.py` modified +1/-1 (2 lines); hunks: -1,6 +1,6.
- Code diff details:
  - `test/registered/layers/mamba/test_causal_conv1d.py` modified +1/-1 (2 lines); hunks: -1,6 +1,6
  - `test/registered/layers/mamba/test_mamba2_mixer.py` modified +1/-1 (2 lines); hunks: -15,7 +15,7
  - `test/registered/layers/mamba/test_mamba_ssm.py` modified +1/-1 (2 lines); hunks: -1,6 +1,6
  - `test/registered/layers/mamba/test_mamba_ssm_ssd.py` modified +1/-1 (2 lines); hunks: -1,6 +1,6
  - `test/registered/models/test_compressed_tensors_models.py` modified +1/-1 (2 lines); hunks: -13,7 +13,7
- Key code excerpts:

```diff
diff -- test/registered/layers/mamba/test_causal_conv1d.py
@@ -1,6 +1,6 @@
-register_cuda_ci(est_time=25, suite="stage-b-test-1-gpu-small")
+register_cuda_ci(est_time=13, suite="stage-b-test-1-gpu-small")
diff -- test/registered/layers/mamba/test_mamba2_mixer.py
@@ -15,7 +15,7 @@
-register_cuda_ci(est_time=50, suite="stage-b-test-2-gpu-large")
+register_cuda_ci(est_time=28, suite="stage-b-test-2-gpu-large")
diff -- test/registered/layers/mamba/test_mamba_ssm.py
@@ -1,6 +1,6 @@
-register_cuda_ci(est_time=7, suite="stage-b-test-1-gpu-small")
+register_cuda_ci(est_time=9, suite="stage-b-test-1-gpu-small")
diff -- test/registered/layers/mamba/test_mamba_ssm_ssd.py
@@ -1,6 +1,6 @@
-register_cuda_ci(est_time=15, suite="stage-b-test-1-gpu-small")
+register_cuda_ci(est_time=9, suite="stage-b-test-1-gpu-small")
diff -- test/registered/models/test_compressed_tensors_models.py
@@ -13,7 +13,7 @@
```

- Reviewed files:
  - tests: `test/registered/layers/mamba/test_causal_conv1d.py` modified +1/-1; `test/registered/layers/mamba/test_mamba2_mixer.py` modified +1/-1; `test/registered/layers/mamba/test_mamba_ssm.py` modified +1/-1; `test/registered/layers/mamba/test_mamba_ssm_ssd.py` modified +1/-1; `test/registered/models/test_compressed_tensors_models.py` modified +1/-1; `test/registered/models/test_generation_models.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `test/registered/4-gpu-models/test_gpt_oss_4gpu.py`, `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`, `test/registered/4-gpu-models/test_qwen35_hicache.py`, `test/registered/4-gpu-models/test_qwen35_models.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #23001 - Add new Mintlify documentation site (docs_new/)

- Link: https://github.com/sgl-project/sglang/pull/23001
- Status/date: merged / 2026-04-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 330 files, +80364/-0, 68714 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add new Mintlify documentation site (docs_new/)"; model line: LLaDA 2.1; category: docs/tests/CI; main diff: `docs_new/docs/advanced_features/tool_parser.mdx`, `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx`, `docs_new/docs/advanced_features/separate_reasoning.mdx`; technical summary: Covers "Add new Mintlify documentation site (docs_new/)"; the main implementation surface is `docs_new/docs/advanced_features/tool_parser.mdx`, `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx`, `docs_new/docs/advanced_features/separate_reasoning.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/docs/advanced_features/tool_parser.mdx` added +740/-0 (740 lines); hunks: -0,0 +1,740; symbols: get_messages, get_current_weather, convert_dict_to_tool, in, touching `get_messages, get_current_weather, convert_dict_to_tool`; `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx` added +663/-0 (663 lines); hunks: -0,0 +1,663; symbols: CapitalInfo, get_messages, touching `CapitalInfo, get_messages`; `docs_new/docs/advanced_features/separate_reasoning.mdx` added +317/-0 (317 lines); hunks: -0,0 +1,317; `docs_new/docs/hardware-platforms/ascend-npus/Support-Features-on-Ascend-NPU.mdx` added +3327/-0 (3327 lines).
- Code diff details:
  - `docs_new/docs/advanced_features/tool_parser.mdx` added +740/-0 (740 lines); hunks: -0,0 +1,740; symbols: get_messages, get_current_weather, convert_dict_to_tool, in
  - `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx` added +663/-0 (663 lines); hunks: -0,0 +1,663; symbols: CapitalInfo, get_messages
  - `docs_new/docs/advanced_features/separate_reasoning.mdx` added +317/-0 (317 lines); hunks: -0,0 +1,317
  - `docs_new/docs/hardware-platforms/ascend-npus/Support-Features-on-Ascend-NPU.mdx` added +3327/-0 (3327 lines)
  - `docs_new/docs/hardware-platforms/ascend-npus/Best-Practice-on-Ascend-NPU.mdx` added +2911/-0 (2911 lines)
- Key code excerpts:

```diff
diff -- docs_new/docs/advanced_features/tool_parser.mdx
@@ -0,0 +1,740 @@
+---
+title: "Tool Parser"
+metatags:
+    description: "SGLang function calling: tool parsers for DeepSeek, Llama, Qwen, Mistral, GLM, Kimi K2. OpenAI-compatible tool use API."
+---
+This guide demonstrates how to use SGLang’s [Function calling](https://platform.openai.com/docs/guides/function-calling) functionality.
diff -- docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx
@@ -0,0 +1,663 @@
+---
+title: "Structured Outputs For Reasoning Models"
+metatags:
+    description: "SGLang structured outputs for reasoning models: free-form thinking with constrained final output for DeepSeek R1, QwQ models."
+---
+When working with reasoning models that use special tokens like `&lt;think&gt;...&lt;/think&gt;` to denote reasoning sections, you might want to allow free-form text within these
diff -- docs_new/docs/advanced_features/separate_reasoning.mdx
@@ -0,0 +1,317 @@
```

- Reviewed files:
  - docs: `docs_new/docs/advanced_features/tool_parser.mdx` added +740/-0; `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx` added +663/-0; `docs_new/docs/advanced_features/separate_reasoning.mdx` added +317/-0; `docs_new/docs/hardware-platforms/ascend-npus/Support-Features-on-Ascend-NPU.mdx` added +3327/-0; `docs_new/docs/hardware-platforms/ascend-npus/Best-Practice-on-Ascend-NPU.mdx` added +2911/-0; `docs_new/docs/advanced_features/server_arguments.mdx` added +2871/-0
- Risk and verification: This is mostly docs/examples in `docs_new/.github/workflows/sync-lmsys-sglang-blogs.yml`, `docs_new/.gitignore`, `docs_new/.mintignore`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23337 - [Docs] Sync docs_new with legacy docs and update migration redirects

- Link: https://github.com/sgl-project/sglang/pull/23337
- Status/date: merged / 2026-04-21
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 179 files, +16004/-8152, 23604 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Docs] Sync docs_new with legacy docs and update migration redirects"; model line: LLaDA 2.1; category: docs/tests/CI; main diff: `docs_new/docs/supported-models/multimodal_language_models.mdx`, `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx`; technical summary: Covers "[Docs] Sync docs_new with legacy docs and update migration redirects"; the main implementation surface is `docs_new/docs/supported-models/multimodal_language_models.mdx`, `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx`, `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/docs/supported-models/multimodal_language_models.mdx` renamed +73/-14 (87 lines); hunks: -1,15 +1,18; -19,11 +22,9 @@ Below the supported models are summarized in a table.; `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx` modified +23/-23 (46 lines); hunks: -3,17 +3,17 @@ title: "Structured Outputs For Reasoning Models"; -252,9 +252,9 @@ If a you choose to call a function ONLY reply in the followi...; `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx` renamed +1809/-463 (2272 lines); `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_support_features.mdx` renamed +671/-1418 (2089 lines).
- Code diff details:
  - `docs_new/docs/supported-models/multimodal_language_models.mdx` renamed +73/-14 (87 lines); hunks: -1,15 +1,18; -19,11 +22,9 @@ Below the supported models are summarized in a table.
  - `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx` modified +23/-23 (46 lines); hunks: -3,17 +3,17 @@ title: "Structured Outputs For Reasoning Models"; -252,9 +252,9 @@ If a you choose to call a function ONLY reply in the followi...
  - `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx` renamed +1809/-463 (2272 lines)
  - `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_support_features.mdx` renamed +671/-1418 (2089 lines)
  - `docs_new/docs/advanced_features/server_arguments.mdx` modified +446/-486 (932 lines)
- Key code excerpts:

```diff
diff -- docs_new/docs/supported-models/multimodal_language_models.mdx
@@ -1,15 +1,18 @@
+---
+title: "Multimodal Language Models"
+metatags:
+  description: "Documentation for Multimodal Language Models"
+---
-<CodeGroup>
diff -- docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx
@@ -3,17 +3,17 @@ title: "Structured Outputs For Reasoning Models"
-When working with reasoning models that use special tokens like `&lt;think&gt;...&lt;/think&gt;` to denote reasoning sections, you might want to allow free-form text within these
+When working with reasoning models that use special tokens like `<think>...</think>` to denote reasoning sections, you might want to allow free-form text within these sections whi
-To enable this feature, use the `--reasoning-parser` flag which decide the think_end_token, such as `&lt;/think&gt;`, when launching the server. You can also specify the reasoning
+To enable this feature, use the `--reasoning-parser` flag which decide the think_end_token, such as `</think>`, when launching the server. You can also specify the reasoning parse
-- [DeepSeek R1 series](https://huggingface.co/collections/deepseek-ai/deepseek-r1-678e1e131c0169c0bc89728d): The reasoning content is wrapped with `&lt;think&gt;` and `&lt;/think&
-- [QwQ](https://huggingface.co/Qwen/QwQ-32B): The reasoning content is wrapped with `&lt;think&gt;` and `&lt;/think&gt;` tags.
diff -- docs_new/docs/hardware-platforms/tpu.mdx
@@ -2,65 +2,67 @@
```

- Reviewed files:
  - docs: `docs_new/docs/supported-models/multimodal_language_models.mdx` renamed +73/-14; `docs_new/docs/advanced_features/structured_outputs_for_reasoning_models.mdx` modified +23/-23; `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx` renamed +1809/-463; `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_support_features.mdx` renamed +671/-1418; `docs_new/docs/advanced_features/server_arguments.mdx` modified +446/-486; `docs_new/docs/hardware-platforms/tpu.mdx` modified +425/-468
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-Math-V2.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR-2.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR.mdx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23732 - Apply should_use_dp_reduce_scatterv guard to remaining MoE models (follow-up to #23731)

- Link: https://github.com/sgl-project/sglang/pull/23732
- Status/date: merged / 2026-04-26
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 13 files, +59/-12, 290 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Apply should_use_dp_reduce_scatterv guard to remaining MoE models (follow-up to #23731)"; model line: LLaDA 2.1; category: model implementation change; main diff: `python/sglang/srt/models/llada2.py`, `python/sglang/srt/models/hunyuan_v3.py`, `python/sglang/srt/models/bailing_moe_linear.py`; technical summary: Covers "Apply should_use_dp_reduce_scatterv guard to remaining MoE models (follow-up to #23731)"; the main implementation surface is `python/sglang/srt/models/llada2.py`, `python/sglang/srt/models/hunyuan_v3.py`, `python/sglang/srt/models/bailing_moe_linear.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/llada2.py` modified +10/-2 (12 lines); hunks: -55,7 +55,11; -379,7 +383,11 @@ def forward_normal(; symbols: forward_normal, touching `forward_normal`; `python/sglang/srt/models/hunyuan_v3.py` modified +7/-4 (11 lines); hunks: -34,6 +34,7; -191,10 +192,11 @@ def _forward_single_stream(self, hidden_states: torch.Tens...; symbols: _forward_single_stream, _forward_dual_stream, touching `_forward_single_stream, _forward_dual_stream`; `python/sglang/srt/models/bailing_moe_linear.py` modified +7/-1 (8 lines); hunks: -34,6 +34,7; -347,7 +348,12 @@ def forward(; symbols: forward, touching `forward`; `python/sglang/srt/models/exaone_moe.py` modified +6/-2 (8 lines); hunks: -47,7 +47,7; -300,7 +300,11 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/llada2.py` modified +10/-2 (12 lines); hunks: -55,7 +55,11; -379,7 +383,11 @@ def forward_normal(; symbols: forward_normal
  - `python/sglang/srt/models/hunyuan_v3.py` modified +7/-4 (11 lines); hunks: -34,6 +34,7; -191,10 +192,11 @@ def _forward_single_stream(self, hidden_states: torch.Tens...; symbols: _forward_single_stream, _forward_dual_stream
  - `python/sglang/srt/models/bailing_moe_linear.py` modified +7/-1 (8 lines); hunks: -34,6 +34,7; -347,7 +348,12 @@ def forward(; symbols: forward
  - `python/sglang/srt/models/exaone_moe.py` modified +6/-2 (8 lines); hunks: -47,7 +47,7; -300,7 +300,11 @@ def forward(; symbols: forward
  - `python/sglang/srt/models/llama4.py` modified +6/-1 (7 lines); hunks: -39,6 +39,7; -145,7 +146,11 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/llada2.py
@@ -55,7 +55,11 @@
-from sglang.srt.layers.moe import get_deepep_mode, get_moe_a2a_backend
+from sglang.srt.layers.moe import (
+    get_deepep_mode,
+    get_moe_a2a_backend,
+    should_use_dp_reduce_scatterv,
+)
diff -- python/sglang/srt/models/hunyuan_v3.py
@@ -34,6 +34,7 @@
+from sglang.srt.layers.moe import should_use_dp_reduce_scatterv
@@ -191,10 +192,11 @@ def _forward_single_stream(self, hidden_states: torch.Tensor) -> torch.Tensor:
-        if self.ep_size > 1:
+        skip_post_reduce = should_use_dp_reduce_scatterv()
+        if self.ep_size > 1 and not skip_post_reduce:
-        if self.tp_size > 1:
diff -- python/sglang/srt/models/bailing_moe_linear.py
@@ -34,6 +34,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/llada2.py` modified +10/-2; `python/sglang/srt/models/hunyuan_v3.py` modified +7/-4; `python/sglang/srt/models/bailing_moe_linear.py` modified +7/-1; `python/sglang/srt/models/exaone_moe.py` modified +6/-2; `python/sglang/srt/models/llama4.py` modified +6/-1; `python/sglang/srt/models/sarvam_moe.py` modified +6/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/bailing_moe_linear.py`, `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23785 - chore: update CI test est_time values

- Link: https://github.com/sgl-project/sglang/pull/23785
- Status/date: merged / 2026-04-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 268 files, +269/-269, 2404 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "chore: update CI test est_time values"; model line: LLaDA 2.1; category: docs/tests/CI; main diff: `test/registered/layers/mamba/test_causal_conv1d.py`, `test/registered/layers/mamba/test_mamba2_mixer.py`, `test/registered/layers/mamba/test_mamba_ssm.py`; technical summary: Covers "chore: update CI test est_time values"; the main implementation surface is `test/registered/layers/mamba/test_causal_conv1d.py`, `test/registered/layers/mamba/test_mamba2_mixer.py`, `test/registered/layers/mamba/test_mamba_ssm.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/layers/mamba/test_causal_conv1d.py` modified +1/-1 (2 lines); hunks: -1,6 +1,6; `test/registered/layers/mamba/test_mamba2_mixer.py` modified +1/-1 (2 lines); hunks: -15,7 +15,7; `test/registered/layers/mamba/test_mamba_ssm.py` modified +1/-1 (2 lines); hunks: -1,6 +1,6; `test/registered/layers/mamba/test_mamba_ssm_ssd.py` modified +1/-1 (2 lines); hunks: -1,6 +1,6.
- Code diff details:
  - `test/registered/layers/mamba/test_causal_conv1d.py` modified +1/-1 (2 lines); hunks: -1,6 +1,6
  - `test/registered/layers/mamba/test_mamba2_mixer.py` modified +1/-1 (2 lines); hunks: -15,7 +15,7
  - `test/registered/layers/mamba/test_mamba_ssm.py` modified +1/-1 (2 lines); hunks: -1,6 +1,6
  - `test/registered/layers/mamba/test_mamba_ssm_ssd.py` modified +1/-1 (2 lines); hunks: -1,6 +1,6
  - `test/registered/models/test_compressed_tensors_models.py` modified +1/-1 (2 lines); hunks: -13,7 +13,7
- Key code excerpts:

```diff
diff -- test/registered/layers/mamba/test_causal_conv1d.py
@@ -1,6 +1,6 @@
-register_cuda_ci(est_time=13, suite="stage-b-test-1-gpu-small")
+register_cuda_ci(est_time=11, suite="stage-b-test-1-gpu-small")
diff -- test/registered/layers/mamba/test_mamba2_mixer.py
@@ -15,7 +15,7 @@
-register_cuda_ci(est_time=28, suite="stage-b-test-2-gpu-large")
+register_cuda_ci(est_time=32, suite="stage-b-test-2-gpu-large")
diff -- test/registered/layers/mamba/test_mamba_ssm.py
@@ -1,6 +1,6 @@
-register_cuda_ci(est_time=9, suite="stage-b-test-1-gpu-small")
+register_cuda_ci(est_time=10, suite="stage-b-test-1-gpu-small")
diff -- test/registered/layers/mamba/test_mamba_ssm_ssd.py
@@ -1,6 +1,6 @@
-register_cuda_ci(est_time=9, suite="stage-b-test-1-gpu-small")
+register_cuda_ci(est_time=10, suite="stage-b-test-1-gpu-small")
diff -- test/registered/models/test_compressed_tensors_models.py
@@ -13,7 +13,7 @@
```

- Reviewed files:
  - tests: `test/registered/layers/mamba/test_causal_conv1d.py` modified +1/-1; `test/registered/layers/mamba/test_mamba2_mixer.py` modified +1/-1; `test/registered/layers/mamba/test_mamba_ssm.py` modified +1/-1; `test/registered/layers/mamba/test_mamba_ssm_ssd.py` modified +1/-1; `test/registered/models/test_compressed_tensors_models.py` modified +1/-1; `test/registered/models/test_generation_models.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `test/registered/4-gpu-models/test_gpt_oss_4gpu.py`, `test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py`, `test/registered/4-gpu-models/test_qwen35_fp4_triton.py`, `test/registered/4-gpu-models/test_qwen3_30b.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #23748 - refactor(moe): centralize post-experts all-reduce skip predicate

- Link: https://github.com/sgl-project/sglang/pull/23748
- Status/date: merged / 2026-04-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 17 files, +134/-132, 532 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "refactor(moe): centralize post-experts all-reduce skip predicate"; model line: LLaDA 2.1; category: model implementation change; main diff: `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/models/sarvam_moe.py`, `python/sglang/srt/models/deepseek_v2.py`; technical summary: Covers "refactor(moe): centralize post-experts all-reduce skip predicate"; the main implementation surface is `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/models/sarvam_moe.py`, `python/sglang/srt/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/moe/utils.py` modified +33/-0 (33 lines); hunks: -346,6 +346,39 @@ def should_use_dp_reduce_scatterv():; symbols: should_use_dp_reduce_scatterv, should_skip_post_experts_all_reduce, speculative_moe_backend_context, touching `should_use_dp_reduce_scatterv, should_skip_post_experts_all_reduce, speculative_moe_backend_context`; `python/sglang/srt/models/sarvam_moe.py` modified +9/-16 (25 lines); hunks: -39,10 +39,7; -373,12 +370,10 @@ def forward_normal_dual_stream(; symbols: forward_normal_dual_stream, forward_normal, touching `forward_normal_dual_stream, forward_normal`; `python/sglang/srt/models/deepseek_v2.py` modified +9/-13 (22 lines); hunks: -85,7 +85,7; -651,12 +651,10 @@ def forward_normal_dual_stream(; symbols: forward_normal_dual_stream, _post_combine_hook, touching `forward_normal_dual_stream, _post_combine_hook`; `python/sglang/srt/models/glm4_moe.py` modified +9/-13 (22 lines); hunks: -61,7 +61,7; -594,12 +594,10 @@ def forward_normal_dual_stream(; symbols: forward_normal_dual_stream, forward_normal, touching `forward_normal_dual_stream, forward_normal`.
- Code diff details:
  - `python/sglang/srt/layers/moe/utils.py` modified +33/-0 (33 lines); hunks: -346,6 +346,39 @@ def should_use_dp_reduce_scatterv():; symbols: should_use_dp_reduce_scatterv, should_skip_post_experts_all_reduce, speculative_moe_backend_context
  - `python/sglang/srt/models/sarvam_moe.py` modified +9/-16 (25 lines); hunks: -39,10 +39,7; -373,12 +370,10 @@ def forward_normal_dual_stream(; symbols: forward_normal_dual_stream, forward_normal
  - `python/sglang/srt/models/deepseek_v2.py` modified +9/-13 (22 lines); hunks: -85,7 +85,7; -651,12 +651,10 @@ def forward_normal_dual_stream(; symbols: forward_normal_dual_stream, _post_combine_hook
  - `python/sglang/srt/models/glm4_moe.py` modified +9/-13 (22 lines); hunks: -61,7 +61,7; -594,12 +594,10 @@ def forward_normal_dual_stream(; symbols: forward_normal_dual_stream, forward_normal
  - `python/sglang/srt/models/qwen3_moe.py` modified +9/-13 (22 lines); hunks: -50,8 +50,7; -332,20 +331,17 @@ def forward_normal(; symbols: forward_normal
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/utils.py
@@ -346,6 +346,39 @@ def should_use_dp_reduce_scatterv():
+def should_skip_post_experts_all_reduce(
+    *,
+    is_tp_path: bool,
+    use_reduce_scatter: bool = False,
+    should_allreduce_fusion: bool = False,
+) -> bool:
diff -- python/sglang/srt/models/sarvam_moe.py
@@ -39,10 +39,7 @@
-from sglang.srt.layers.moe import (
-    should_use_dp_reduce_scatterv,
-    should_use_flashinfer_cutlass_moe_fp4_allgather,
-)
+from sglang.srt.layers.moe import should_skip_post_experts_all_reduce
@@ -373,12 +370,10 @@ def forward_normal_dual_stream(
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -85,7 +85,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/utils.py` modified +33/-0; `python/sglang/srt/models/sarvam_moe.py` modified +9/-16; `python/sglang/srt/models/deepseek_v2.py` modified +9/-13; `python/sglang/srt/models/glm4_moe.py` modified +9/-13; `python/sglang/srt/models/qwen3_moe.py` modified +9/-13; `python/sglang/srt/models/hunyuan_v3.py` modified +13/-7
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/__init__.py`, `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/models/bailing_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23835 - [NPU] Add GitHub test summary and deduplicate test code. Part 1

- Link: https://github.com/sgl-project/sglang/pull/23835
- Status/date: merged / 2026-05-02
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +333/-332, 851 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NPU] Add GitHub test summary and deduplicate test code. Part 1"; model line: LLaDA 2.1; category: bug fix; main diff: `test/registered/ascend/basic_function/parallel_strategy/expert_parallelism/test_npu_deepep_auto_deepseek_v3_2_w8a8.py`, `test/registered/ascend/basic_function/optimization_debug/test_npu_piecewise_graph_prefill.py`, `test/registered/ascend/basic_function/speculative_inference/test_npu_eagle3.py`; technical summary: Covers "[NPU] Add GitHub test summary and deduplicate test code. Part 1"; the main implementation surface is `test/registered/ascend/basic_function/parallel_strategy/expert_parallelism/test_npu_deepep_auto_deepseek_v3_2_w8a8.py`, `test/registered/ascend/basic_function/optimization_debug/test_npu_piecewise_graph_prefill.py`, `test/registered/ascend/basic_function/speculative_inference/test_npu_eagle3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/ascend/basic_function/parallel_strategy/expert_parallelism/test_npu_deepep_auto_deepseek_v3_2_w8a8.py` modified +43/-88 (131 lines); hunks: -1,107 +1,62; symbols: TestDeepEpDeepseekV32, setUpClass, tearDownClass, test_mmlu, touching `TestDeepEpDeepseekV32, setUpClass, tearDownClass`; `test/registered/ascend/basic_function/optimization_debug/test_npu_piecewise_graph_prefill.py` modified +54/-70 (124 lines); hunks: -1,92 +1,76; symbols: TestPiecewiseGraphPrefillCorrectness, setUpClass, tearDownClass, test_gsm8k, touching `TestPiecewiseGraphPrefillCorrectness, setUpClass, tearDownClass`; `test/registered/ascend/basic_function/speculative_inference/test_npu_eagle3.py` modified +40/-78 (118 lines); hunks: -1,99 +1,61; symbols: TestNpuEagle3, setUpClass, test_gsm8k, touching `TestNpuEagle3, setUpClass, test_gsm8k`; `python/sglang/test/ascend/gsm8k_ascend_mixin.py` modified +70/-35 (105 lines); hunks: -1,19 +1,22; -23,48 +26,80 @@ class GSM8KAscendMixin(ABC):; symbols: GSM8KAscendMixin, setUpClass, tearDownClass, test_gsm8k, touching `GSM8KAscendMixin, setUpClass, tearDownClass`.
- Code diff details:
  - `test/registered/ascend/basic_function/parallel_strategy/expert_parallelism/test_npu_deepep_auto_deepseek_v3_2_w8a8.py` modified +43/-88 (131 lines); hunks: -1,107 +1,62; symbols: TestDeepEpDeepseekV32, setUpClass, tearDownClass, test_mmlu
  - `test/registered/ascend/basic_function/optimization_debug/test_npu_piecewise_graph_prefill.py` modified +54/-70 (124 lines); hunks: -1,92 +1,76; symbols: TestPiecewiseGraphPrefillCorrectness, setUpClass, tearDownClass, test_gsm8k
  - `test/registered/ascend/basic_function/speculative_inference/test_npu_eagle3.py` modified +40/-78 (118 lines); hunks: -1,99 +1,61; symbols: TestNpuEagle3, setUpClass, test_gsm8k
  - `python/sglang/test/ascend/gsm8k_ascend_mixin.py` modified +70/-35 (105 lines); hunks: -1,19 +1,22; -23,48 +26,80 @@ class GSM8KAscendMixin(ABC):; symbols: GSM8KAscendMixin, setUpClass, tearDownClass, test_gsm8k
  - `test/registered/ascend/basic_function/dllm/test_npu_llada2_mini.py` modified +23/-59 (82 lines); hunks: -1,77 +1,41; symbols: TestLLaDA2Mini, setUpClass, tearDownClass, test_gsm8k
- Key code excerpts:

```diff
diff -- test/registered/ascend/basic_function/parallel_strategy/expert_parallelism/test_npu_deepep_auto_deepseek_v3_2_w8a8.py
@@ -1,107 +1,62 @@
-from types import SimpleNamespace
-from sglang.srt.utils import kill_process_tree
+from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
+from sglang.test.ascend.test_mmlu import TestMMLU
-from sglang.test.few_shot_gsm8k import run_eval as run_gsm8k
-from sglang.test.run_eval import run_eval
diff -- test/registered/ascend/basic_function/optimization_debug/test_npu_piecewise_graph_prefill.py
@@ -1,92 +1,76 @@
+import subprocess
-from urllib.parse import urlparse
-from sglang.srt.utils import kill_process_tree
+from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
+from sglang.test.ascend.test_ascend_utils import (
+    QWEN2_5_7B_INSTRUCT_WEIGHTS_PATH,
diff -- test/registered/ascend/basic_function/speculative_inference/test_npu_eagle3.py
@@ -1,99 +1,61 @@
```

- Reviewed files:
  - tests: `test/registered/ascend/basic_function/parallel_strategy/expert_parallelism/test_npu_deepep_auto_deepseek_v3_2_w8a8.py` modified +43/-88; `test/registered/ascend/basic_function/optimization_debug/test_npu_piecewise_graph_prefill.py` modified +54/-70; `test/registered/ascend/basic_function/speculative_inference/test_npu_eagle3.py` modified +40/-78; `python/sglang/test/ascend/gsm8k_ascend_mixin.py` modified +70/-35; `test/registered/ascend/basic_function/dllm/test_npu_llada2_mini.py` modified +23/-59; `python/sglang/test/ascend/test_ascend_utils.py` modified +48/-0
- Risk and verification: The diff ships test coverage in `python/sglang/test/ascend/gsm8k_ascend_mixin.py`, `python/sglang/test/ascend/test_ascend_utils.py`, `python/sglang/test/ascend/test_mmlu.py`, `python/sglang/test/ascend/vlm_utils.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #25197 - ci: decouple stage and runner for cuda registry

- Link: https://github.com/sgl-project/sglang/pull/25197
- Status/date: merged / 2026-05-14
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 261 files, +388/-293, 2625 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "ci: decouple stage and runner for cuda registry"; model line: LLaDA 2.1; category: performance/backend optimization; main diff: `test/registered/layers/test_fla_layernorm_guard.py`, `test/registered/models/test_dummy_grok_models.py`, `test/registered/models/test_ministral3_models.py`; technical summary: Covers "ci: decouple stage and runner for cuda registry"; the main implementation surface is `test/registered/layers/test_fla_layernorm_guard.py`, `test/registered/models/test_dummy_grok_models.py`, `test/registered/models/test_ministral3_models.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/layers/test_fla_layernorm_guard.py` modified +2/-1 (3 lines); hunks: -19,7 +19,8; `test/registered/models/test_dummy_grok_models.py` modified +2/-1 (3 lines); hunks: -5,7 +5,8; `test/registered/models/test_ministral3_models.py` modified +2/-1 (3 lines); hunks: -8,7 +8,8; `test/registered/models/test_ministral4_models.py` modified +2/-1 (3 lines); hunks: -8,7 +8,8.
- Code diff details:
  - `test/registered/layers/test_fla_layernorm_guard.py` modified +2/-1 (3 lines); hunks: -19,7 +19,8
  - `test/registered/models/test_dummy_grok_models.py` modified +2/-1 (3 lines); hunks: -5,7 +5,8
  - `test/registered/models/test_ministral3_models.py` modified +2/-1 (3 lines); hunks: -8,7 +8,8
  - `test/registered/models/test_ministral4_models.py` modified +2/-1 (3 lines); hunks: -8,7 +8,8
  - `test/registered/models/test_nvidia_nemotron_3_nano.py` modified +2/-1 (3 lines); hunks: -6,7 +6,8
- Key code excerpts:

```diff
diff -- test/registered/layers/test_fla_layernorm_guard.py
@@ -19,7 +19,8 @@
-    suite="stage-b-test-2-gpu-large",
+    stage="stage-b",
+    runner_config="2-gpu-large",
diff -- test/registered/models/test_dummy_grok_models.py
@@ -5,7 +5,8 @@
-    suite="stage-b-test-2-gpu-large",
+    stage="stage-b",
+    runner_config="2-gpu-large",
diff -- test/registered/models/test_ministral3_models.py
@@ -8,7 +8,8 @@
-    suite="stage-b-test-1-gpu-small",
+    stage="stage-b",
+    runner_config="1-gpu-small",
diff -- test/registered/models/test_ministral4_models.py
@@ -8,7 +8,8 @@
-    suite="stage-b-test-2-gpu-large",
```

- Reviewed files:
  - tests: `test/registered/layers/test_fla_layernorm_guard.py` modified +2/-1; `test/registered/models/test_dummy_grok_models.py` modified +2/-1; `test/registered/models/test_ministral3_models.py` modified +2/-1; `test/registered/models/test_ministral4_models.py` modified +2/-1; `test/registered/models/test_nvidia_nemotron_3_nano.py` modified +2/-1; `test/registered/layers/mamba/test_causal_conv1d.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `python/sglang/test/ci/ci_register.py`, `test/registered/4-gpu-models/test_deepseek_v3_cutedsl_4gpu.py`, `test/registered/4-gpu-models/test_gpt_oss_4gpu.py`, `test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #25420 - [CI] Rename basic CI `stage-a/b/c` -> `base-a/b/c` for symmetry with extra CI

- Link: https://github.com/sgl-project/sglang/pull/25420
- Status/date: merged / 2026-05-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 473 files, +746/-747, 5614 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CI] Rename basic CI `stage-a/b/c` -> `base-a/b/c` for symmetry with extra CI"; model line: LLaDA 2.1; category: docs/tests/CI; main diff: `.github/workflows/pr-test-multimodal-gen.yml`, `test/registered/bench_fn/test_bench_serving_reasoning_stream.py`, `test/registered/function_call/test_kimik2_detector.py`; technical summary: Covers "[CI] Rename basic CI `stage-a/b/c` -> `base-a/b/c` for symmetry with extra CI"; the main implementation surface is `.github/workflows/pr-test-multimodal-gen.yml`, `test/registered/bench_fn/test_bench_serving_reasoning_stream.py`, `test/registered/function_call/test_kimik2_detector.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `.github/workflows/pr-test-multimodal-gen.yml` modified +7/-7 (14 lines); hunks: -31,7 +31,7 @@ on:; -42,7 +42,7 @@ env:; `test/registered/bench_fn/test_bench_serving_reasoning_stream.py` modified +1/-1 (2 lines); hunks: -24,7 +24,7; symbols: _free_port, touching `_free_port`; `test/registered/function_call/test_kimik2_detector.py` modified +1/-1 (2 lines); hunks: -11,7 +11,7; symbols: _make_tool, touching `_make_tool`; `test/registered/layers/mamba/test_causal_conv1d.py` modified +1/-1 (2 lines); hunks: -2,7 +2,7.
- Code diff details:
  - `.github/workflows/pr-test-multimodal-gen.yml` modified +7/-7 (14 lines); hunks: -31,7 +31,7 @@ on:; -42,7 +42,7 @@ env:
  - `test/registered/bench_fn/test_bench_serving_reasoning_stream.py` modified +1/-1 (2 lines); hunks: -24,7 +24,7; symbols: _free_port
  - `test/registered/function_call/test_kimik2_detector.py` modified +1/-1 (2 lines); hunks: -11,7 +11,7; symbols: _make_tool
  - `test/registered/layers/mamba/test_causal_conv1d.py` modified +1/-1 (2 lines); hunks: -2,7 +2,7
  - `test/registered/layers/mamba/test_mamba2_mixer.py` modified +1/-1 (2 lines); hunks: -18,7 +18,7
- Key code excerpts:

```diff
diff -- .github/workflows/pr-test-multimodal-gen.yml
@@ -31,7 +31,7 @@ on:
-      skip_stage_health_check:
+      skip_pr_test_health_check:
@@ -42,7 +42,7 @@ env:
-  SKIP_STAGE_HEALTH_CHECK: ${{ inputs.skip_stage_health_check == 'true' }}
+  SKIP_PR_TEST_HEALTH_CHECK: ${{ inputs.skip_pr_test_health_check == 'true' }}
@@ -90,7 +90,7 @@ jobs:
diff -- test/registered/bench_fn/test_bench_serving_reasoning_stream.py
@@ -24,7 +24,7 @@
-register_cpu_ci(est_time=10, suite="stage-a-test-cpu")
+register_cpu_ci(est_time=10, suite="base-a-test-cpu")
diff -- test/registered/function_call/test_kimik2_detector.py
@@ -11,7 +11,7 @@
-register_cpu_ci(5, "stage-a-test-cpu")
+register_cpu_ci(5, "base-a-test-cpu")
diff -- test/registered/layers/mamba/test_causal_conv1d.py
@@ -2,7 +2,7 @@
```

- Reviewed files:
  - runtime: `.github/workflows/pr-test-multimodal-gen.yml` modified +7/-7
  - tests: `test/registered/bench_fn/test_bench_serving_reasoning_stream.py` modified +1/-1; `test/registered/function_call/test_kimik2_detector.py` modified +1/-1; `test/registered/layers/mamba/test_causal_conv1d.py` modified +1/-1; `test/registered/layers/mamba/test_mamba2_mixer.py` modified +1/-1; `test/registered/layers/mamba/test_mamba_ssm.py` modified +1/-1; `test/registered/layers/mamba/test_mamba_ssm_ssd.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `python/sglang/jit_kernel/tests/deepseek_v4/test_c128_v2.py`, `python/sglang/jit_kernel/tests/deepseek_v4/test_c4_v2.py`, `python/sglang/jit_kernel/tests/test_activation.py`, `python/sglang/jit_kernel/tests/test_add_constant.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #23906 - [Refactor] Cuda Graph Runner/Backend Refactor

- Link: https://github.com/sgl-project/sglang/pull/23906
- Status/date: merged / 2026-06-10
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 160 files, +5197/-3068, 12233 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Refactor] Cuda Graph Runner/Backend Refactor"; model line: LLaDA 2.1; category: performance/backend optimization; main diff: `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py`; technical summary: Covers "[Refactor] Cuda Graph Runner/Backend Refactor"; the main implementation surface is `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` removed +0/-860 (860 lines); hunks: -1,860 +0,0; symbols: freeze_gc, _to_torch, patch_model, get_global_graph_memory_pool, touching `freeze_gc, _to_torch, patch_model`; `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py` added +846/-0 (846 lines); hunks: -0,0 +1,846; symbols: PrefillCudaGraphRunner, __init__, _is_mamba_track_enabled, _cache_loc_dtype, touching `PrefillCudaGraphRunner, __init__, _is_mamba_track_enabled`; `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` renamed +294/-463 (757 lines); hunks: -1,4 +1,4; -11,33 +11,36; symbols: _make_graph_key, build_replay_fb_view, _allocate_decode_buffers, get_is_capture_mode, touching `_make_graph_key, build_replay_fb_view, _allocate_decode_buffers`; `python/sglang/srt/model_executor/breakable_cuda_graph_runner.py` removed +0/-541 (541 lines); hunks: -1,541 +0,0; symbols: BreakableCudaGraphRunner, __init__, _has_inactive_dp_rank, _init_buffers, touching `BreakableCudaGraphRunner, __init__, _has_inactive_dp_rank`.
- Code diff details:
  - `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` removed +0/-860 (860 lines); hunks: -1,860 +0,0; symbols: freeze_gc, _to_torch, patch_model, get_global_graph_memory_pool
  - `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py` added +846/-0 (846 lines); hunks: -0,0 +1,846; symbols: PrefillCudaGraphRunner, __init__, _is_mamba_track_enabled, _cache_loc_dtype
  - `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` renamed +294/-463 (757 lines); hunks: -1,4 +1,4; -11,33 +11,36; symbols: _make_graph_key, build_replay_fb_view, _allocate_decode_buffers, get_is_capture_mode
  - `python/sglang/srt/model_executor/breakable_cuda_graph_runner.py` removed +0/-541 (541 lines); hunks: -1,541 +0,0; symbols: BreakableCudaGraphRunner, __init__, _has_inactive_dp_rank, _init_buffers
  - `python/sglang/srt/model_executor/runner_utils/buffers.py` added +442/-0 (442 lines); hunks: -0,0 +1,442; symbols: _grouped_foreach_copy_, foreach_copy, DecodeInputBuffers, create
- Key code excerpts:

```diff
diff -- python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py
@@ -1,860 +0,0 @@
-# Copyright 2023-2024 SGLang Team
-# Licensed under the Apache License, Version 2.0 (the "License");
-# you may not use this file except in compliance with the License.
-# You may obtain a copy of the License at
-#
-#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py
@@ -0,0 +1,846 @@
+# Copyright 2023-2026 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py
@@ -1,4 +1,4 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` removed +0/-860; `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py` added +846/-0; `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` renamed +294/-463; `python/sglang/srt/model_executor/breakable_cuda_graph_runner.py` removed +0/-541; `python/sglang/srt/model_executor/runner_utils/buffers.py` added +442/-0; `python/sglang/srt/model_executor/runner_backend/tc_piecewise_cuda_graph_backend.py` added +225/-0
- Risk and verification: The diff ships test coverage in `python/sglang/test/doc_patch.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dense_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dsa_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dsv4_attention.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #28567 - Add get_parallel(): a structured accessor for parallel-topology state

- Link: https://github.com/sgl-project/sglang/pull/28567
- Status/date: merged / 2026-06-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 184 files, +1865/-1727, 8932 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add get_parallel(): a structured accessor for parallel-topology state"; model line: LLaDA 2.1; category: model support/runtime entry; main diff: `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py`; technical summary: Covers "Add get_parallel(): a structured accessor for parallel-topology state"; the main implementation surface is `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/apertus.py` modified +686/-687 (1373 lines); hunks: -1,687 +1,686; symbols: ApertusMLP, __init__, forward, ApertusAttention, touching `ApertusMLP, __init__, forward`; `python/sglang/srt/models/solar.py` modified +28/-27 (55 lines); hunks: -1,37 +1,14; -54,6 +31,30; symbols: __init__, forward, load_kv_cache_scales, touching `__init__, forward, load_kv_cache_scales`; `python/sglang/srt/models/gpt_oss.py` modified +17/-24 (41 lines); hunks: -28,21 +28,13; -76,6 +68,7; symbols: _resolve_moe_input_pad_multiple, __init__, touching `_resolve_moe_input_pad_multiple, __init__`; `python/sglang/srt/models/deepseek_v2.py` modified +14/-23 (37 lines); hunks: -47,9 +47,7; -72,12 +70,6; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/apertus.py` modified +686/-687 (1373 lines); hunks: -1,687 +1,686; symbols: ApertusMLP, __init__, forward, ApertusAttention
  - `python/sglang/srt/models/solar.py` modified +28/-27 (55 lines); hunks: -1,37 +1,14; -54,6 +31,30; symbols: __init__, forward, load_kv_cache_scales
  - `python/sglang/srt/models/gpt_oss.py` modified +17/-24 (41 lines); hunks: -28,21 +28,13; -76,6 +68,7; symbols: _resolve_moe_input_pad_multiple, __init__
  - `python/sglang/srt/models/deepseek_v2.py` modified +14/-23 (37 lines); hunks: -47,9 +47,7; -72,12 +70,6; symbols: __init__
  - `python/sglang/srt/layers/communicator.py` modified +13/-19 (32 lines); hunks: -23,8 +23,6; -44,12 +42,7; symbols: apply_aiter_all_reduce_fusion, init_context, should_fuse_mlp_allreduce_with_next_layer, is_same_group_size
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/apertus.py
@@ -1,687 +1,686 @@
-# SPDX-License-Identifier: Apache-2.0
-# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-# Copyright 2025 The SwissAI Initiative
-# Copyright 2023-2024 SGLang Team
-# Licensed under the Apache License, Version 2.0 (the "License");
-# you may not use this file except in compliance with the License.
diff -- python/sglang/srt/models/solar.py
@@ -1,37 +1,14 @@
-# Adapted from
-# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
-# Copyright 2023 The vLLM team.
-# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
-#
-# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
diff -- python/sglang/srt/models/gpt_oss.py
@@ -28,21 +28,13 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/apertus.py` modified +686/-687; `python/sglang/srt/models/solar.py` modified +28/-27; `python/sglang/srt/models/gpt_oss.py` modified +17/-24; `python/sglang/srt/models/deepseek_v2.py` modified +14/-23; `python/sglang/srt/layers/communicator.py` modified +13/-19; `python/sglang/srt/models/qwen3_moe.py` modified +12/-18
- Risk and verification: The diff ships test coverage in `python/sglang/test/kits/attention_unittest/attention_methods/dense_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dsa_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dsv4_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dual_chunk_attention.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #28697 - [docs] Add B300 cookbook deployment options

- Link: https://github.com/sgl-project/sglang/pull/28697
- Status/date: merged / 2026-06-19
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 27 files, +503/-69, 1291 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[docs] Add B300 cookbook deployment options"; model line: LLaDA 2.1; category: performance/backend optimization; main diff: `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx`, `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx`; technical summary: Covers "[docs] Add B300 cookbook deployment options"; the main implementation surface is `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx`, `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx` added +167/-0 (167 lines); hunks: -0,0 +1,167; `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx` modified +68/-2 (70 lines); hunks: -9,6 +9,11 @@ const lookupData = {; -182,6 +187,66 @@ const lookupData = {; `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx` modified +40/-16 (56 lines); hunks: -4,6 +4,7 @@ export const GLM5Deployment = () => {; -13,6 +14,7 @@ export const GLM5Deployment = () => {; `docs_new/src/snippets/autoregressive/deepseek-v32-deployment.jsx` modified +29/-10 (39 lines); hunks: -3,7 +3,7 @@ export const DeepSeekV32Deployment = () => {; -12,6 +12,7 @@ export const DeepSeekV32Deployment = () => {.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx` added +167/-0 (167 lines); hunks: -0,0 +1,167
  - `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx` modified +68/-2 (70 lines); hunks: -9,6 +9,11 @@ const lookupData = {; -182,6 +187,66 @@ const lookupData = {
  - `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx` modified +40/-16 (56 lines); hunks: -4,6 +4,7 @@ export const GLM5Deployment = () => {; -13,6 +14,7 @@ export const GLM5Deployment = () => {
  - `docs_new/src/snippets/autoregressive/deepseek-v32-deployment.jsx` modified +29/-10 (39 lines); hunks: -3,7 +3,7 @@ export const DeepSeekV32Deployment = () => {; -12,6 +12,7 @@ export const DeepSeekV32Deployment = () => {
  - `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx` modified +23/-15 (38 lines); hunks: -8,19 +8,19 @@ export const Qwen35Deployment = () => {; -149,7 +149,7 @@ export const Qwen35Deployment = () => {
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx
@@ -0,0 +1,167 @@
+export const InternS1Deployment = () => {
+  const options = {
+    hardware: {
+      name: 'hardware',
+      title: 'Hardware Platform',
+      items: [
diff -- docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx
@@ -9,6 +9,11 @@ const lookupData = {
+      {
+        "id": "b300",
+        "label": "B300",
+        "default": false
+      },
@@ -182,6 +187,66 @@ const lookupData = {
diff -- docs_new/src/snippets/autoregressive/glm-5-deployment.jsx
@@ -4,6 +4,7 @@ export const GLM5Deployment = () => {
```

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx` added +167/-0; `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx` modified +68/-2; `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx` modified +40/-16; `docs_new/src/snippets/autoregressive/deepseek-v32-deployment.jsx` modified +29/-10; `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx` modified +23/-15; `docs_new/cookbook/autoregressive/InternLM/Intern-S1.mdx` modified +16/-13
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/InternLM/Intern-S1.mdx`, `docs_new/src/snippets/autoregressive/deepseek-math-v2-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #29042 - [NPU] Fix the DeepSeek-V2-Coder model accuracy issue

- Link: https://github.com/sgl-project/sglang/pull/29042
- Status/date: merged / 2026-06-25
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +4/-1, 26 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NPU] Fix the DeepSeek-V2-Coder model accuracy issue"; model line: LLaDA 2.1; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/llada2.py`, `python/sglang/srt/hardware_backend/npu/moe/topk.py`; technical summary: Covers "[NPU] Fix the DeepSeek-V2-Coder model accuracy issue"; the main implementation surface is `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/llada2.py`, `python/sglang/srt/hardware_backend/npu/moe/topk.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +1/-0 (1 lines); hunks: -647,6 +647,7 @@ def __init__(; symbols: __init__, touching `__init__`; `python/sglang/srt/models/llada2.py` modified +1/-0 (1 lines); hunks: -262,6 +262,7 @@ def __init__(; symbols: __init__, touching `__init__`; `python/sglang/srt/hardware_backend/npu/moe/topk.py` modified +2/-1 (3 lines); hunks: -80,7 +80,8 @@ def fused_topk_npu(; symbols: fused_topk_npu, touching `fused_topk_npu`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +1/-0 (1 lines); hunks: -647,6 +647,7 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/llada2.py` modified +1/-0 (1 lines); hunks: -262,6 +262,7 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/hardware_backend/npu/moe/topk.py` modified +2/-1 (3 lines); hunks: -80,7 +80,8 @@ def fused_topk_npu(; symbols: fused_topk_npu
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -647,6 +647,7 @@ def __init__(
+                scoring_func=config.scoring_func,
diff -- python/sglang/srt/models/llada2.py
@@ -262,6 +262,7 @@ def __init__(
+            scoring_func=self.score_function,
diff -- python/sglang/srt/hardware_backend/npu/moe/topk.py
@@ -80,7 +80,8 @@ def fused_topk_npu(
-            norm_type=1,  # 1 for sigmoid, 0 for softmax
+            # 1 for sigmoid, 0 for softmax
+            norm_type=(0 if topk_config.scoring_func == "softmax" else 1),
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +1/-0; `python/sglang/srt/models/llada2.py` modified +1/-0; `python/sglang/srt/hardware_backend/npu/moe/topk.py` modified +2/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/hardware_backend/npu/moe/topk.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/llada2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
