# vllm DeepSeek V3.1 Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `examples/ray_serving/elastic_ep/serve_deepseek_v2.sh` | no direct PR-number commit |
| `examples/tool_chat_template_deepseekv31.jinja` | [#23454](https://github.com/vllm-project/vllm/pull/23454) |
| `tests/tool_parsers/test_deepseekv31_tool_parser.py` | no direct PR-number commit |
| `vllm/model_executor/models/deepseek_mtp.py` | no direct PR-number commit |
| `vllm/model_executor/models/deepseek_v2.py` | no direct PR-number commit |
| `vllm/tool_parsers/deepseekv31_tool_parser.py` | no direct PR-number commit |

## PR Coverage Summary

- Git-traced PRs: 1
- Extra PRs preserved from existing docs: 52
- Total PRs in this document: 53
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2025-08-23 | [#23454](https://github.com/vllm-project/vllm/pull/23454) | merged | Support DeepSeek-V3.1 tool call | `examples/tool_chat_template_deepseekv31.jinja` |
| 2025-08-27 | [#23666](https://github.com/vllm-project/vllm/pull/23666) | merged | [Feature] Add Hopper DeepGEMM E8M0 for DeepSeekV3.1 scale_fmt | `vllm/model_executor/layers/quantization/fp8.py`, `vllm/model_executor/layers/fused_moe/fused_moe.py`, `vllm/model_executor/layers/fused_moe/triton_deep_gemm_moe.py` |
| 2025-10-15 | [#25589](https://github.com/vllm-project/vllm/pull/25589) | merged | [Model] Add DeepSeek-V3.1 reasoning parser (split from PR #24972) | `tests/reasoning/test_deepseekv3_reasoning_parser.py`, `vllm/reasoning/deepseek_v3_reasoning_parser.py`, `vllm/reasoning/identity_reasoning_parser.py` |
| 2026-01-13 | [#29867](https://github.com/vllm-project/vllm/pull/29867) | merged | [Quantization] fix: overflow with static per-tensor scaling | `vllm/model_executor/layers/quantization/utils/quant_utils.py`, `vllm/v1/attention/backends/mla/common.py` |
| 2026-01-15 | [#32361](https://github.com/vllm-project/vllm/pull/32361) | merged | [BugFix] Fix DeepSeek-V3.1 + DeepGEMM incompatible scale shapes | `vllm/model_executor/layers/quantization/utils/quant_utils.py` |
| 2026-01-16 | [#32175](https://github.com/vllm-project/vllm/pull/32175) | merged | [Bugfix] [DeepSeek-V3.2] fix sparse_attn_indexer padding | `vllm/model_executor/models/deepseek_v2.py` |
| 2026-01-20 | [#32652](https://github.com/vllm-project/vllm/pull/32652) | merged | [Bugfix] Fix the fp8_mqa_logits dim mismatch | `vllm/model_executor/models/deepseek_v2.py`, `vllm/utils/deep_gemm.py` |
| 2026-01-21 | [#29287](https://github.com/vllm-project/vllm/pull/29287) | merged | [ROCm][Deepseekv3.2] Refactor Sparse Indexer as CustomOp | `vllm/model_executor/layers/sparse_attn_indexer.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` |
| 2026-01-26 | [#33063](https://github.com/vllm-project/vllm/pull/33063) | merged | [Chore] Update type annotation of `input_ids` in model forward | `vllm/model_executor/models/modernbert.py`, `vllm/model_executor/models/gemma3n.py`, `vllm/model_executor/models/gpt2.py` |
| 2026-01-26 | [#33018](https://github.com/vllm-project/vllm/pull/33018) | merged | [ROCm][Bugfix] Fix ptpc scale load issue for fused shared expert path in deepseek mtp | `vllm/model_executor/models/deepseek_mtp.py` |
| 2026-01-27 | [#32064](https://github.com/vllm-project/vllm/pull/32064) | merged | [5/N][Attention] Finish eliminating `vllm/attention` folder | `vllm/model_executor/layers/attention/mla_attention.py`, `vllm/model_executor/layers/attention/attention.py`, `vllm/model_executor/layers/attention/__init__.py` |
| 2026-01-28 | [#33191](https://github.com/vllm-project/vllm/pull/33191) | merged | Add flake8-implicit-str-concat rules to Ruff | `tests/tool_parsers/test_deepseekv31_tool_parser.py`, `vllm/model_executor/layers/quantization/kernels/scaled_mm/aiter.py`, `vllm/entrypoints/openai/translations/speech_to_text.py` |
| 2026-01-31 | [#33174](https://github.com/vllm-project/vllm/pull/33174) | merged | Add support for Mistral Large 3 inference with Flashinfer MoE | `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200,dtype=fp8_w8a8.json`, `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200.json`, `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_GB200,dtype=fp8_w8a8.json` |
| 2026-02-05 | [#33858](https://github.com/vllm-project/vllm/pull/33858) | merged | [Bugfix] Kimi-K2 grouped_topk usage for Flashinfer monolithic kernels. | `vllm/model_executor/models/deepseek_v2.py` |
| 2026-02-05 | [#33876](https://github.com/vllm-project/vllm/pull/33876) | merged | [Bugfix] Fix Kimi-K2.5 NVFP4 checkpoints weight loading | `vllm/model_executor/models/kimi_k25.py`, `vllm/model_executor/models/deepseek_v2.py` |
| 2026-02-09 | [#34124](https://github.com/vllm-project/vllm/pull/34124) | merged | [Model] GLM adaptation | `vllm/model_executor/models/deepseek_v2.py`, `tests/models/registry.py`, `tests/models/test_initialization.py` |
| 2026-02-11 | [#34353](https://github.com/vllm-project/vllm/pull/34353) | merged | [Bugfix] fix default is_neox_style to be True for deepseekv3.2 | `vllm/model_executor/models/deepseek_v2.py` |
| 2026-02-17 | [#34514](https://github.com/vllm-project/vllm/pull/34514) | merged | [CI][BugFix] ShellCheck cleanup to remove baseline and preserve runtime behavior | `tests/v1/ec_connector/integration/run_epd_correctness_test.sh`, `tools/pre_commit/shellcheck.baseline`, `benchmarks/auto_tune/auto_tune.sh` |
| 2026-02-18 | [#34758](https://github.com/vllm-project/vllm/pull/34758) | merged | [Model Bash] DeepSeek R1 BF16 Min Latency QKV A GEMM (0.5% E2E Speedup) | `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/layers/mla.py`, `csrc/dsv3_fused_a_gemm.cu` |
| 2026-02-19 | [#34876](https://github.com/vllm-project/vllm/pull/34876) | merged | [Bug] Fix DeepSeek V3 weight loading caused by incorrect prefix | `vllm/model_executor/models/deepseek_v2.py` |
| 2026-02-23 | [#34302](https://github.com/vllm-project/vllm/pull/34302) | merged | [ModelBash][DSV3] Add TRTLLM DSV3 Router GEMM kernel (6% B1 Speedup) | `vllm/model_executor/models/deepseek_v2.py`, `csrc/moe/dsv3_router_gemm_bf16_out.cu`, `csrc/moe/dsv3_router_gemm_float_out.cu` |
| 2026-02-26 | [#33724](https://github.com/vllm-project/vllm/pull/33724) | merged | [WideEP] Remove pplx all2all backend | `vllm/model_executor/layers/fused_moe/pplx_prepare_finalize.py`, `vllm/model_executor/layers/fused_moe/all2all_utils.py`, `vllm/model_executor/layers/fused_moe/config.py` |
| 2026-02-27 | [#35121](https://github.com/vllm-project/vllm/pull/35121) | merged | [Performance] Cublas Bf16 Gate with Fp32 Output | `vllm/model_executor/layers/fused_moe/router/gate_linear.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/nemotron_h.py` |
| 2026-02-28 | [#35548](https://github.com/vllm-project/vllm/pull/35548) | merged | [MTP] Validate that MTP weights are actually loaded | `vllm/model_executor/models/deepseek_mtp.py` |
| 2026-03-02 | [#35751](https://github.com/vllm-project/vllm/pull/35751) | merged | [MoE][Perf] Wrap DSV3 QKVAProj GEMM in custom op for torch.compile | `vllm/model_executor/models/deepseek_v2.py` |
| 2026-03-07 | [#36247](https://github.com/vllm-project/vllm/pull/36247) | merged | [Bugfix] Fix compressed-tensors quantization failure for DeepSeek-R1 on MI300x | `vllm/model_executor/models/deepseek_v2.py` |
| 2026-03-11 | [#36361](https://github.com/vllm-project/vllm/pull/36361) | merged | Kimi k2.5 MLA based eagle3 | `vllm/model_executor/models/deepseek_eagle3.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/kimi_k25.py` |
| 2026-03-13 | [#36931](https://github.com/vllm-project/vllm/pull/36931) | merged | [Feat][Bugfix] Enable additional dimension for Flashinfer MLA and fix routing dtype | `vllm/model_executor/models/deepseek_v2.py`, `vllm/v1/attention/backends/mla/flashinfer_mla.py` |
| 2026-03-24 | [#37487](https://github.com/vllm-project/vllm/pull/37487) | merged | [V0 Deprecation] Refactor kv cache from list to element | `vllm/model_executor/layers/attention/mla_attention.py`, `vllm/model_executor/layers/attention/attention.py`, `vllm/model_executor/models/extract_hidden_states.py` |
| 2026-03-26 | [#38029](https://github.com/vllm-project/vllm/pull/38029) | merged | [Tool Parser][1/3] Pass tools to ToolParser constructor | `vllm/tool_parsers/abstract_tool_parser.py`, `vllm/tool_parsers/qwen3coder_tool_parser.py`, `vllm/tool_parsers/step3p5_tool_parser.py` |
| 2026-04-02 | [#38684](https://github.com/vllm-project/vllm/pull/38684) | merged | [Perf] DSV3.2 Indexer Fused Weights Projection | `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py` |
| 2026-04-04 | [#38870](https://github.com/vllm-project/vllm/pull/38870) | merged | [Bugfix] Fix DSV32 weight loading | `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py` |
| 2026-04-08 | [#37421](https://github.com/vllm-project/vllm/pull/37421) | merged | [Perf][Kernel] Persistent TopK scheduler: unified CUDAGraph-safe kernel with dynamic per-row dispatch - DeepSeek-V3.2 DSA decode | `vllm/model_executor/layers/sparse_attn_indexer.py`, `vllm/model_executor/models/deepseek_v2.py`, `csrc/persistent_topk.cuh` |
| 2026-04-15 | [#38928](https://github.com/vllm-project/vllm/pull/38928) | merged | [Bugfix][Perf] Indexer upcast WK to BF16 for fusion | `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py` |
| 2026-04-20 | [#35949](https://github.com/vllm-project/vllm/pull/35949) | merged | [MoE Refactor] Move the shared/fused expert output sum into MoERunnerBase | `vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/exaone_moe.py` |
| 2026-04-21 | [#35782](https://github.com/vllm-project/vllm/pull/35782) | merged | [MoE Refactor] Remove SharedFusedMoE class | `vllm/model_executor/layers/fused_moe/shared_fused_moe.py`, `vllm/model_executor/models/afmoe.py`, `vllm/model_executor/models/llama4.py` |
| 2026-04-23 | [#40671](https://github.com/vllm-project/vllm/pull/40671) | merged | [MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping | `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/llama4.py`, `vllm/model_executor/models/glm4_moe_lite.py` |
| 2026-04-24 | [#39999](https://github.com/vllm-project/vllm/pull/39999) | merged | [ROCm] Cast score correction bias tensor during model construction for DeepSeek/Kimi-K2 | `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py`, `vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py` |
| 2026-04-27 | [#39141](https://github.com/vllm-project/vllm/pull/39141) | merged | [Perf] Update TRTLLM supported MoE routing methods | `vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py`, `vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py`, `vllm/model_executor/layers/fused_moe/config.py` |
| 2026-04-29 | [#37735](https://github.com/vllm-project/vllm/pull/37735) | merged | [Feature]: IndexCache support for DSA models | `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/layers/mla.py`, `docs/features/index_cache.md` |
| 2026-05-01 | [#41217](https://github.com/vllm-project/vllm/pull/41217) | merged | [ROCm][Deepseek] dsv3.2 further optimization | `vllm/model_executor/models/deepseek_v2.py`, `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py`, `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` |
| 2026-05-02 | [#41405](https://github.com/vllm-project/vllm/pull/41405) | merged | [ROCm][Bugfix] Fix init-time bias dtype cast when gate.out_dtype is None | `vllm/model_executor/models/deepseek_v2.py` |
| 2026-05-06 | [#40759](https://github.com/vllm-project/vllm/pull/40759) | merged | [Examples] Resettle Disaggregated examples. | `examples/disaggregated/lmcache/disagg_prefill_lmcache_v1/configs/lmcache-decoder-config.yaml`, `examples/disaggregated/lmcache/disagg_prefill_lmcache_v1/configs/lmcache-prefiller-config.yaml`, `docs/features/disagg_prefill.md` |
| 2026-05-07 | [#41835](https://github.com/vllm-project/vllm/pull/41835) | merged | [ROCm][DeepSeek] Enable V3.2 TP4 AITER MLA | `vllm/model_executor/models/deepseek_v2.py`, `vllm/v1/attention/backends/mla/rocm_aiter_mla.py` |
| 2026-05-09 | [#41706](https://github.com/vllm-project/vllm/pull/41706) | merged | [Model] use AutoWeightsLoader for DeepSeekV2 | `vllm/model_executor/models/deepseek_v2.py` |
| 2026-05-28 | [#43781](https://github.com/vllm-project/vllm/pull/43781) | merged | [Bugfix][ROCm] Fix Accuracy Drop in Sparse Indexer on gfx950 | `vllm/model_executor/models/deepseek_v2.py`, `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` |
| 2026-05-29 | [#42982](https://github.com/vllm-project/vllm/pull/42982) | merged | [ROCm][Perf] DSv3.2 MI355X TP4 decode-step orchestration cleanup (3 micro-opts) | `vllm/model_executor/models/deepseek_v2.py`, `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py` |
| 2026-06-01 | [#42944](https://github.com/vllm-project/vllm/pull/42944) | merged | fix: glm5.1 pp model loading | `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py` |
| 2026-06-07 | [#44420](https://github.com/vllm-project/vllm/pull/44420) | merged | [feature] add index share feature for DSA MTP | `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`, `vllm/v1/spec_decode/llm_base_proposer.py` |
| 2026-06-12 | [#45003](https://github.com/vllm-project/vllm/pull/45003) | merged | [Frontend] Support strict mode for tool calling | `vllm/tool_parsers/qwen3xml_tool_parser.py`, `vllm/tool_parsers/structural_tag_registry.py`, `tests/tool_parsers/test_structural_tag_registry.py` |
| 2026-06-19 | [#45895](https://github.com/vllm-project/vllm/pull/45895) | merged | [bugfix]Indexer init skip and MTP TopK share for iteration | `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`, `vllm/model_executor/layers/attention/mla_attention.py` |
| 2026-06-20 | [#46199](https://github.com/vllm-project/vllm/pull/46199) | merged | [Bugfix] Move extract_layer_index back inside is_v32 guard | `vllm/model_executor/models/deepseek_v2.py` |
| 2026-06-25 | [#46651](https://github.com/vllm-project/vllm/pull/46651) | merged | [Perf] Remove redundant clone for GLM, Deepseek etc | `vllm/model_executor/models/AXK1.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/glm4_moe_lite.py` |

## Per-PR Diff Audit Cards

### PR #23454 - Support DeepSeek-V3.1 tool call

- Link: https://github.com/vllm-project/vllm/pull/23454
- Status/date: merged / 2025-08-23
- Trace source: `git log --name-only -- <model-files>` found it through `examples/tool_chat_template_deepseekv31.jinja`; associated commits `b8f17f5d980e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +468/-0, 491 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support DeepSeek-V3.1 tool call"; model line: DeepSeek V3.1; category: model support/runtime entry; main diff: `examples/tool_chat_template_deepseekv31.jinja`; technical summary: Covers "Support DeepSeek-V3.1 tool call"; the main implementation surface is `examples/tool_chat_template_deepseekv31.jinja`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `examples/tool_chat_template_deepseekv31.jinja` added +91/-0 (91 lines); hunks: -0,0 +1,91.
- Code diff details:
  - `examples/tool_chat_template_deepseekv31.jinja` added +91/-0 (91 lines); hunks: -0,0 +1,91
- Key code excerpts:

```diff
diff -- examples/tool_chat_template_deepseekv31.jinja
@@ -0,0 +1,91 @@
+{% if not add_generation_prompt is defined %}
+  {% set add_generation_prompt = false %}
+{% endif %}
+{% if not thinking is defined %}
+  {% set thinking = false %}
+{% endif %}
```

- Reviewed files:
  - docs: `examples/tool_chat_template_deepseekv31.jinja` added +91/-0
- Risk and verification: Runtime changes concentrate in `vllm/entrypoints/openai/tool_parsers/__init__.py`, `vllm/entrypoints/openai/tool_parsers/deepseekv31_tool_parser.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23666 - [Feature] Add Hopper DeepGEMM E8M0 for DeepSeekV3.1 scale_fmt

- Link: https://github.com/vllm-project/vllm/pull/23666
- Status/date: merged / 2025-08-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +68/-53, 322 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feature] Add Hopper DeepGEMM E8M0 for DeepSeekV3.1 scale_fmt"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `vllm/model_executor/layers/quantization/fp8.py`, `vllm/model_executor/layers/fused_moe/fused_moe.py`, `vllm/model_executor/layers/fused_moe/triton_deep_gemm_moe.py`; technical summary: Covers "[Feature] Add Hopper DeepGEMM E8M0 for DeepSeekV3.1 scale_fmt"; the main implementation surface is `vllm/model_executor/layers/quantization/fp8.py`, `vllm/model_executor/layers/fused_moe/fused_moe.py`, `vllm/model_executor/layers/fused_moe/triton_deep_gemm_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/fp8.py` modified +4/-5 (9 lines); hunks: -48,8 +48,7; -427,7 +426,7 @@ def process_weights_after_loading(self, layer: Module) -> None:; symbols: process_weights_after_loading, touching `process_weights_after_loading`; `vllm/model_executor/layers/fused_moe/fused_moe.py` modified +3/-4 (7 lines); hunks: -40,7 +40,7; -1431,9 +1431,8 @@ def fused_experts(hidden_states: torch.Tensor,; symbols: fused_experts, touching `fused_experts`; `vllm/model_executor/layers/fused_moe/triton_deep_gemm_moe.py` modified +3/-3 (6 lines); hunks: -10,7 +10,7; -107,7 +107,7 @@ def workspace_shapes(; symbols: TritonOrDeepGemmExperts, workspace_shapes, apply, touching `TritonOrDeepGemmExperts, workspace_shapes, apply`; `vllm/model_executor/layers/fused_moe/batched_deep_gemm_moe.py` modified +2/-2 (4 lines); hunks: -12,7 +12,7; -174,7 +174,7 @@ def silu_mul_fp8_quant_deep_gemm(; symbols: silu_mul_fp8_quant_deep_gemm, touching `silu_mul_fp8_quant_deep_gemm`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/fp8.py` modified +4/-5 (9 lines); hunks: -48,8 +48,7; -427,7 +426,7 @@ def process_weights_after_loading(self, layer: Module) -> None:; symbols: process_weights_after_loading
  - `vllm/model_executor/layers/fused_moe/fused_moe.py` modified +3/-4 (7 lines); hunks: -40,7 +40,7; -1431,9 +1431,8 @@ def fused_experts(hidden_states: torch.Tensor,; symbols: fused_experts
  - `vllm/model_executor/layers/fused_moe/triton_deep_gemm_moe.py` modified +3/-3 (6 lines); hunks: -10,7 +10,7; -107,7 +107,7 @@ def workspace_shapes(; symbols: TritonOrDeepGemmExperts, workspace_shapes, apply
  - `vllm/model_executor/layers/fused_moe/batched_deep_gemm_moe.py` modified +2/-2 (4 lines); hunks: -12,7 +12,7; -174,7 +174,7 @@ def silu_mul_fp8_quant_deep_gemm(; symbols: silu_mul_fp8_quant_deep_gemm
  - `vllm/model_executor/layers/quantization/utils/fp8_utils.py` modified +2/-2 (4 lines); hunks: -20,7 +20,7; -385,7 +385,7 @@ def per_token_group_quant_fp8(; symbols: per_token_group_quant_fp8
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/quantization/fp8.py
@@ -48,8 +48,7 @@
-from vllm.utils.deep_gemm import (is_blackwell_deep_gemm_e8m0_used,
-                                  is_deep_gemm_supported)
+from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used, is_deep_gemm_supported
@@ -427,7 +426,7 @@ def process_weights_after_loading(self, layer: Module) -> None:
-        if is_blackwell_deep_gemm_e8m0_used():
+        if is_deep_gemm_e8m0_used():
diff -- vllm/model_executor/layers/fused_moe/fused_moe.py
@@ -40,7 +40,7 @@
-from vllm.utils.deep_gemm import is_blackwell_deep_gemm_e8m0_used
+from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used
@@ -1431,9 +1431,8 @@ def fused_experts(hidden_states: torch.Tensor,
-    if (allow_deep_gemm and use_fp8_w8a8
-            and (is_blackwell_deep_gemm_e8m0_used()
-                 or _valid_deep_gemm(hidden_states, w1, w2))):
diff -- vllm/model_executor/layers/fused_moe/triton_deep_gemm_moe.py
@@ -10,7 +10,7 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/fp8.py` modified +4/-5; `vllm/model_executor/layers/fused_moe/fused_moe.py` modified +3/-4; `vllm/model_executor/layers/fused_moe/triton_deep_gemm_moe.py` modified +3/-3; `vllm/model_executor/layers/fused_moe/batched_deep_gemm_moe.py` modified +2/-2; `vllm/model_executor/layers/quantization/utils/fp8_utils.py` modified +2/-2; `vllm/utils/deep_gemm.py` modified +24/-29
- Risk and verification: The diff ships test coverage in `tests/kernels/moe/test_block_fp8.py`, `tests/kernels/moe/test_deepep_deepgemm_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #25589 - [Model] Add DeepSeek-V3.1 reasoning parser (split from PR #24972)

- Link: https://github.com/vllm-project/vllm/pull/25589
- Status/date: merged / 2025-10-15
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +215/-3, 269 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Add DeepSeek-V3.1 reasoning parser (split from PR #24972)"; model line: DeepSeek V3.1; category: docs/tests/CI; main diff: `tests/reasoning/test_deepseekv3_reasoning_parser.py`, `vllm/reasoning/deepseek_v3_reasoning_parser.py`, `vllm/reasoning/identity_reasoning_parser.py`; technical summary: Covers "[Model] Add DeepSeek-V3.1 reasoning parser (split from PR #24972)"; the main implementation surface is `tests/reasoning/test_deepseekv3_reasoning_parser.py`, `vllm/reasoning/deepseek_v3_reasoning_parser.py`, `vllm/reasoning/identity_reasoning_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/reasoning/test_deepseekv3_reasoning_parser.py` added +76/-0 (76 lines); hunks: -0,0 +1,76; symbols: tokenizer, test_parser_selection, test_identity_reasoning_parser_basic, touching `tokenizer, test_parser_selection, test_identity_reasoning_parser_basic`; `vllm/reasoning/deepseek_v3_reasoning_parser.py` added +66/-0 (66 lines); hunks: -0,0 +1,66; symbols: DeepSeekV3ReasoningParser, __init__, is_reasoning_end, extract_content_ids, touching `DeepSeekV3ReasoningParser, __init__, is_reasoning_end`; `vllm/reasoning/identity_reasoning_parser.py` added +58/-0 (58 lines); hunks: -0,0 +1,58; symbols: IdentityReasoningParser, __init__, is_reasoning_end, extract_content_ids, touching `IdentityReasoningParser, __init__, is_reasoning_end`; `vllm/entrypoints/openai/serving_chat.py` modified +8/-2 (10 lines); hunks: -573,7 +573,10 @@ async def chat_completion_stream_generator(; -1342,7 +1345,10 @@ async def chat_completion_full_generator(; symbols: chat_completion_stream_generator, chat_completion_full_generator, touching `chat_completion_stream_generator, chat_completion_full_generator`.
- Code diff details:
  - `tests/reasoning/test_deepseekv3_reasoning_parser.py` added +76/-0 (76 lines); hunks: -0,0 +1,76; symbols: tokenizer, test_parser_selection, test_identity_reasoning_parser_basic
  - `vllm/reasoning/deepseek_v3_reasoning_parser.py` added +66/-0 (66 lines); hunks: -0,0 +1,66; symbols: DeepSeekV3ReasoningParser, __init__, is_reasoning_end, extract_content_ids
  - `vllm/reasoning/identity_reasoning_parser.py` added +58/-0 (58 lines); hunks: -0,0 +1,58; symbols: IdentityReasoningParser, __init__, is_reasoning_end, extract_content_ids
  - `vllm/entrypoints/openai/serving_chat.py` modified +8/-2 (10 lines); hunks: -573,7 +573,10 @@ async def chat_completion_stream_generator(; -1342,7 +1345,10 @@ async def chat_completion_full_generator(; symbols: chat_completion_stream_generator, chat_completion_full_generator
  - `docs/features/reasoning_outputs.md` modified +3/-1 (4 lines); hunks: -11,6 +11,7 @@ vLLM currently supports the following reasoning models:; -20,8 +21,9 @@ vLLM currently supports the following reasoning models:
- Key code excerpts:

```diff
diff -- tests/reasoning/test_deepseekv3_reasoning_parser.py
@@ -0,0 +1,76 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import pytest
+from transformers import AutoTokenizer
+from vllm.entrypoints.openai.protocol import ChatCompletionRequest, DeltaMessage
+from vllm.reasoning import (
diff -- vllm/reasoning/deepseek_v3_reasoning_parser.py
@@ -0,0 +1,66 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from collections.abc import Sequence
+from transformers import PreTrainedTokenizerBase
+from vllm.entrypoints.openai.protocol import ChatCompletionRequest, DeltaMessage
+from vllm.logger import init_logger
diff -- vllm/reasoning/identity_reasoning_parser.py
@@ -0,0 +1,58 @@
```

- Reviewed files:
  - tests: `tests/reasoning/test_deepseekv3_reasoning_parser.py` added +76/-0
  - runtime: `vllm/reasoning/deepseek_v3_reasoning_parser.py` added +66/-0; `vllm/reasoning/identity_reasoning_parser.py` added +58/-0; `vllm/entrypoints/openai/serving_chat.py` modified +8/-2; `vllm/reasoning/__init__.py` modified +4/-0
  - docs: `docs/features/reasoning_outputs.md` modified +3/-1
- Risk and verification: The diff ships test coverage in `tests/reasoning/test_deepseekv3_reasoning_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #29867 - [Quantization] fix: overflow with static per-tensor scaling

- Link: https://github.com/vllm-project/vllm/pull/29867
- Status/date: merged / 2026-01-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +71/-56, 182 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Quantization] fix: overflow with static per-tensor scaling"; model line: DeepSeek V3.1; category: bug fix; main diff: `vllm/model_executor/layers/quantization/utils/quant_utils.py`, `vllm/v1/attention/backends/mla/common.py`; technical summary: Covers "[Quantization] fix: overflow with static per-tensor scaling"; the main implementation surface is `vllm/model_executor/layers/quantization/utils/quant_utils.py`, `vllm/v1/attention/backends/mla/common.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/utils/quant_utils.py` modified +61/-2 (63 lines); hunks: -5,7 +5,7; -15,6 +15,9; symbols: scaled_dequantize, get_attribute_fallback, get_and_maybe_dequant_weights, pack_quantized_values_into_int32, touching `scaled_dequantize, get_attribute_fallback, get_and_maybe_dequant_weights`; `vllm/v1/attention/backends/mla/common.py` modified +10/-54 (64 lines); hunks: -207,8 +207,9; -1184,35 +1185,13 @@ def __init__(; symbols: __init__, process_weights_after_loading, get_layer_weight, get_and_maybe_dequant_weights, touching `__init__, process_weights_after_loading, get_layer_weight`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/utils/quant_utils.py` modified +61/-2 (63 lines); hunks: -5,7 +5,7; -15,6 +15,9; symbols: scaled_dequantize, get_attribute_fallback, get_and_maybe_dequant_weights, pack_quantized_values_into_int32
  - `vllm/v1/attention/backends/mla/common.py` modified +10/-54 (64 lines); hunks: -207,8 +207,9; -1184,35 +1185,13 @@ def __init__(; symbols: __init__, process_weights_after_loading, get_layer_weight, get_and_maybe_dequant_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/quantization/utils/quant_utils.py
@@ -5,7 +5,7 @@
-from typing import ClassVar, NamedTuple
+from typing import TYPE_CHECKING, ClassVar, NamedTuple
@@ -15,6 +15,9 @@
+if TYPE_CHECKING:
+    from vllm.model_executor.layers.linear import LinearBase
@@ -239,7 +242,7 @@ def scaled_dequantize(
diff -- vllm/v1/attention/backends/mla/common.py
@@ -207,8 +207,9 @@
-    LinearBase,
-    UnquantizedLinearMethod,
+)
+from vllm.model_executor.layers.quantization.utils.quant_utils import (
+    get_and_maybe_dequant_weights,
@@ -1184,35 +1185,13 @@ def __init__(
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/utils/quant_utils.py` modified +61/-2; `vllm/v1/attention/backends/mla/common.py` modified +10/-54
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/quantization/utils/quant_utils.py`, `vllm/v1/attention/backends/mla/common.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #32361 - [BugFix] Fix DeepSeek-V3.1 + DeepGEMM incompatible scale shapes

- Link: https://github.com/vllm-project/vllm/pull/32361
- Status/date: merged / 2026-01-15
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-0, 10 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BugFix] Fix DeepSeek-V3.1 + DeepGEMM incompatible scale shapes"; model line: DeepSeek V3.1; category: bug fix; main diff: `vllm/model_executor/layers/quantization/utils/quant_utils.py`; technical summary: Covers "[BugFix] Fix DeepSeek-V3.1 + DeepGEMM incompatible scale shapes"; the main implementation surface is `vllm/model_executor/layers/quantization/utils/quant_utils.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/utils/quant_utils.py` modified +3/-0 (3 lines); hunks: -299,6 +299,9 @@ def get_and_maybe_dequant_weights(; symbols: get_and_maybe_dequant_weights, touching `get_and_maybe_dequant_weights`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/utils/quant_utils.py` modified +3/-0 (3 lines); hunks: -299,6 +299,9 @@ def get_and_maybe_dequant_weights(; symbols: get_and_maybe_dequant_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/quantization/utils/quant_utils.py
@@ -299,6 +299,9 @@ def get_and_maybe_dequant_weights(
+        # DeepGEMM transforms the scales using `transform_sf_into_required_layout` into
+        # a layout that is not compatible with `scaled_dequantize`.
+        and not layer.quant_method.use_deep_gemm
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/utils/quant_utils.py` modified +3/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/quantization/utils/quant_utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #32175 - [Bugfix] [DeepSeek-V3.2] fix sparse_attn_indexer padding

- Link: https://github.com/vllm-project/vllm/pull/32175
- Status/date: merged / 2026-01-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +9/-2, 38 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] [DeepSeek-V3.2] fix sparse_attn_indexer padding"; model line: DeepSeek V3.1; category: bug fix; main diff: `vllm/model_executor/models/deepseek_v2.py`; technical summary: Covers "[Bugfix] [DeepSeek-V3.2] fix sparse_attn_indexer padding"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #32652 - [Bugfix] Fix the fp8_mqa_logits dim mismatch

- Link: https://github.com/vllm-project/vllm/pull/32652
- Status/date: merged / 2026-01-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +3/-3, 20 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix the fp8_mqa_logits dim mismatch"; model line: DeepSeek V3.1; category: bug fix; main diff: `vllm/model_executor/models/deepseek_v2.py`, `vllm/utils/deep_gemm.py`; technical summary: Covers "[Bugfix] Fix the fp8_mqa_logits dim mismatch"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`, `vllm/utils/deep_gemm.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +1/-1 (2 lines); hunks: -686,7 +686,7 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer, touching `sparse_attn_indexer`; `vllm/utils/deep_gemm.py` modified +2/-2 (4 lines); hunks: -249,8 +249,8 @@ def fp8_mqa_logits(; symbols: fp8_mqa_logits, touching `fp8_mqa_logits`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +1/-1 (2 lines); hunks: -686,7 +686,7 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer
  - `vllm/utils/deep_gemm.py` modified +2/-2 (4 lines); hunks: -249,8 +249,8 @@ def fp8_mqa_logits(; symbols: fp8_mqa_logits
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -686,7 +686,7 @@ def sparse_attn_indexer(
-                (k_fp8, k_scale.view(torch.float32)),
+                (k_fp8, k_scale.view(torch.float32).flatten()),
diff -- vllm/utils/deep_gemm.py
@@ -249,8 +249,8 @@ def fp8_mqa_logits(
-            dtype `torch.float8_e4m3fn` and `k_scales` has shape [N] (or
-            [N, 1]) with dtype `torch.float32`.
+            dtype `torch.float8_e4m3fn` and `k_scales` has shape [N])
+            with dtype `torch.float32`.
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +1/-1; `vllm/utils/deep_gemm.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v2.py`, `vllm/utils/deep_gemm.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #29287 - [ROCm][Deepseekv3.2] Refactor Sparse Indexer as CustomOp

- Link: https://github.com/vllm-project/vllm/pull/29287
- Status/date: merged / 2026-01-21
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +982/-323, 1521 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm][Deepseekv3.2] Refactor Sparse Indexer as CustomOp"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `vllm/model_executor/layers/sparse_attn_indexer.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`; technical summary: Covers "[ROCm][Deepseekv3.2] Refactor Sparse Indexer as CustomOp"; the main implementation surface is `vllm/model_executor/layers/sparse_attn_indexer.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/sparse_attn_indexer.py` added +318/-0 (318 lines); hunks: -0,0 +1,318; symbols: sparse_attn_indexer, sparse_attn_indexer_fake, SparseAttnIndexer, __init__, touching `sparse_attn_indexer, sparse_attn_indexer_fake, SparseAttnIndexer`; `vllm/model_executor/models/deepseek_v2.py` modified +14/-233 (247 lines); hunks: -43,7 +43,6; -63,6 +62,7; symbols: get_attn_backend, sparse_attn_indexer, sparse_attn_indexer_fake, Indexer, touching `get_attn_backend, sparse_attn_indexer, sparse_attn_indexer_fake`; `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` modified +518/-80 (598 lines); hunks: -1,100 +1,220; -183,10 +303,38 @@ def rocm_fp8_paged_mqa_logits(; symbols: fp8_mqa_logits_torch, _indexer_k_quant_and_cache_kernel, rocm_fp8_mqa_logits, indexer_k_quant_and_cache_triton, touching `fp8_mqa_logits_torch, _indexer_k_quant_and_cache_kernel, rocm_fp8_mqa_logits`; `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py` modified +110/-10 (120 lines); hunks: -15,6 +15,7; -33,6 +34,48; symbols: fetch_id_to_ragged_kernel, fetch_id_to_ragged_triton, ROCMAiterMLASparseBackend, ROCMAiterMLASparseMetadata, touching `fetch_id_to_ragged_kernel, fetch_id_to_ragged_triton, ROCMAiterMLASparseBackend`.
- Code diff details:
  - `vllm/model_executor/layers/sparse_attn_indexer.py` added +318/-0 (318 lines); hunks: -0,0 +1,318; symbols: sparse_attn_indexer, sparse_attn_indexer_fake, SparseAttnIndexer, __init__
  - `vllm/model_executor/models/deepseek_v2.py` modified +14/-233 (247 lines); hunks: -43,7 +43,6; -63,6 +62,7; symbols: get_attn_backend, sparse_attn_indexer, sparse_attn_indexer_fake, Indexer
  - `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` modified +518/-80 (598 lines); hunks: -1,100 +1,220; -183,10 +303,38 @@ def rocm_fp8_paged_mqa_logits(; symbols: fp8_mqa_logits_torch, _indexer_k_quant_and_cache_kernel, rocm_fp8_mqa_logits, indexer_k_quant_and_cache_triton
  - `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py` modified +110/-10 (120 lines); hunks: -15,6 +15,7; -33,6 +34,48; symbols: fetch_id_to_ragged_kernel, fetch_id_to_ragged_triton, ROCMAiterMLASparseBackend, ROCMAiterMLASparseMetadata
  - `vllm/_aiter_ops.py` modified +12/-0 (12 lines); hunks: -9,6 +9,10; -1091,6 +1095,14 @@ def register_ops_once() -> None:; symbols: register_ops_once
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/sparse_attn_indexer.py
@@ -0,0 +1,318 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Custom Sparse Attention Indexer layers."""
+import torch
+from vllm._aiter_ops import rocm_aiter_ops
+from vllm.forward_context import get_forward_context
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -43,7 +43,6 @@
-from vllm.forward_context import get_forward_context
@@ -63,6 +62,7 @@
+from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
@@ -74,16 +74,11 @@
-from vllm.utils.deep_gemm import fp8_mqa_logits, fp8_paged_mqa_logits
-from vllm.utils.torch_utils import direct_register_custom_op
diff -- vllm/v1/attention/ops/rocm_aiter_mla_sparse.py
@@ -1,100 +1,220 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/sparse_attn_indexer.py` added +318/-0; `vllm/model_executor/models/deepseek_v2.py` modified +14/-233; `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` modified +518/-80; `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py` modified +110/-10; `vllm/_aiter_ops.py` modified +12/-0; `vllm/v1/attention/backends/mla/indexer.py` modified +6/-0
- Risk and verification: Runtime changes concentrate in `vllm/_aiter_ops.py`, `vllm/config/compilation.py`, `vllm/model_executor/layers/sparse_attn_indexer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #33063 - [Chore] Update type annotation of `input_ids` in model forward

- Link: https://github.com/vllm-project/vllm/pull/33063
- Status/date: merged / 2026-01-26
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 164 files, +243/-241, 2158 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Chore] Update type annotation of `input_ids` in model forward"; model line: DeepSeek V3.1; category: model implementation change; main diff: `vllm/model_executor/models/modernbert.py`, `vllm/model_executor/models/gemma3n.py`, `vllm/model_executor/models/gpt2.py`; technical summary: Covers "[Chore] Update type annotation of `input_ids` in model forward"; the main implementation surface is `vllm/model_executor/models/modernbert.py`, `vllm/model_executor/models/gemma3n.py`, `vllm/model_executor/models/gpt2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/modernbert.py` modified +4/-5 (9 lines); hunks: -54,12 +54,11 @@ def forward(; symbols: forward, ModernBertAttention, touching `forward, ModernBertAttention`; `vllm/model_executor/models/gemma3n.py` modified +4/-4 (8 lines); hunks: -704,7 +704,7 @@ def altup_embed(self, hidden_states_0: torch.Tensor) -> torc...; -887,7 +887,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; symbols: altup_embed, forward, embed_input_ids, fast_prefill_forward, touching `altup_embed, forward, embed_input_ids`; `vllm/model_executor/models/gpt2.py` modified +3/-3 (6 lines); hunks: -218,7 +218,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; -298,7 +298,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; symbols: embed_input_ids, forward, load_weights, touching `embed_input_ids, forward, load_weights`; `vllm/model_executor/models/internlm2.py` modified +3/-3 (6 lines); hunks: -284,7 +284,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; -350,7 +350,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; symbols: embed_input_ids, forward, __init__, touching `embed_input_ids, forward, __init__`.
- Code diff details:
  - `vllm/model_executor/models/modernbert.py` modified +4/-5 (9 lines); hunks: -54,12 +54,11 @@ def forward(; symbols: forward, ModernBertAttention
  - `vllm/model_executor/models/gemma3n.py` modified +4/-4 (8 lines); hunks: -704,7 +704,7 @@ def altup_embed(self, hidden_states_0: torch.Tensor) -> torc...; -887,7 +887,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; symbols: altup_embed, forward, embed_input_ids, fast_prefill_forward
  - `vllm/model_executor/models/gpt2.py` modified +3/-3 (6 lines); hunks: -218,7 +218,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; -298,7 +298,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; symbols: embed_input_ids, forward, load_weights
  - `vllm/model_executor/models/internlm2.py` modified +3/-3 (6 lines); hunks: -284,7 +284,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; -350,7 +350,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; symbols: embed_input_ids, forward, __init__
  - `vllm/model_executor/models/opt.py` modified +3/-3 (6 lines); hunks: -267,7 +267,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; -316,7 +316,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; symbols: embed_input_ids, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/modernbert.py
@@ -54,12 +54,11 @@ def forward(
-        if inputs_embeds is not None:
-            return self.norm(inputs_embeds)
-        else:
+        if inputs_embeds is None:
-            embeddings = self.norm(inputs_embeds)
-            return embeddings
diff -- vllm/model_executor/models/gemma3n.py
@@ -704,7 +704,7 @@ def altup_embed(self, hidden_states_0: torch.Tensor) -> torch.Tensor:
-        input_ids: torch.Tensor,
+        input_ids: torch.Tensor | None,
@@ -887,7 +887,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
-        input_ids: torch.Tensor,
+        input_ids: torch.Tensor | None,
@@ -964,7 +964,7 @@ def fast_prefill_forward(
diff -- vllm/model_executor/models/gpt2.py
@@ -218,7 +218,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/modernbert.py` modified +4/-5; `vllm/model_executor/models/gemma3n.py` modified +4/-4; `vllm/model_executor/models/gpt2.py` modified +3/-3; `vllm/model_executor/models/internlm2.py` modified +3/-3; `vllm/model_executor/models/opt.py` modified +3/-3; `vllm/model_executor/models/afmoe.py` modified +2/-2
- Risk and verification: The diff ships test coverage in `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_gemma_embedding.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #33018 - [ROCm][Bugfix] Fix ptpc scale load issue for fused shared expert path in deepseek mtp

- Link: https://github.com/vllm-project/vllm/pull/33018
- Status/date: merged / 2026-01-26
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +11/-8, 34 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm][Bugfix] Fix ptpc scale load issue for fused shared expert path in deepseek mtp"; model line: DeepSeek V3.1; category: bug fix; main diff: `vllm/model_executor/models/deepseek_mtp.py`; technical summary: Covers "[ROCm][Bugfix] Fix ptpc scale load issue for fused shared expert path in deepseek mtp"; the main implementation surface is `vllm/model_executor/models/deepseek_mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_mtp.py` modified +11/-8 (19 lines); hunks: -316,7 +316,11 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; -329,14 +333,13 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_mtp.py` modified +11/-8 (19 lines); hunks: -316,7 +316,11 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; -329,14 +333,13 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: load_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_mtp.py
@@ -316,7 +316,11 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
-                    split_dim = 1 if "down_proj.weight" in name else 0
+                    split_dim = (
+                        1
+                        if ("down_proj.weight" in name and loaded_weight.ndim > 1)
+                        else 0
+                    )
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_mtp.py` modified +11/-8
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_mtp.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #32064 - [5/N][Attention] Finish eliminating `vllm/attention` folder

- Link: https://github.com/vllm-project/vllm/pull/32064
- Status/date: merged / 2026-01-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 151 files, +585/-527, 2850 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[5/N][Attention] Finish eliminating `vllm/attention` folder"; model line: DeepSeek V3.1; category: model implementation change; main diff: `vllm/model_executor/layers/attention/mla_attention.py`, `vllm/model_executor/layers/attention/attention.py`, `vllm/model_executor/layers/attention/__init__.py`; technical summary: Covers "[5/N][Attention] Finish eliminating `vllm/attention` folder"; the main implementation surface is `vllm/model_executor/layers/attention/mla_attention.py`, `vllm/model_executor/layers/attention/attention.py`, `vllm/model_executor/layers/attention/__init__.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/attention/mla_attention.py` modified +354/-17 (371 lines); hunks: -191,24 +191,38; -217,11 +231,16; symbols: MLAAttention, takes, does, __init__, touching `MLAAttention, takes, does`; `vllm/model_executor/layers/attention/attention.py` renamed +42/-315 (357 lines); hunks: -1,23 +1,22; -33,20 +32,54; symbols: validate_kv_sharing_target, should_load_quant_weights, get_kv_cache_spec, MLAAttention, touching `validate_kv_sharing_target, should_load_quant_weights, get_kv_cache_spec`; `vllm/model_executor/layers/attention/__init__.py` modified +26/-0 (26 lines); hunks: -0,0 +1,26; `vllm/model_executor/models/whisper.py` modified +5/-3 (8 lines); hunks: -17,16 +17,18.
- Code diff details:
  - `vllm/model_executor/layers/attention/mla_attention.py` modified +354/-17 (371 lines); hunks: -191,24 +191,38; -217,11 +231,16; symbols: MLAAttention, takes, does, __init__
  - `vllm/model_executor/layers/attention/attention.py` renamed +42/-315 (357 lines); hunks: -1,23 +1,22; -33,20 +32,54; symbols: validate_kv_sharing_target, should_load_quant_weights, get_kv_cache_spec, MLAAttention
  - `vllm/model_executor/layers/attention/__init__.py` modified +26/-0 (26 lines); hunks: -0,0 +1,26
  - `vllm/model_executor/models/whisper.py` modified +5/-3 (8 lines); hunks: -17,16 +17,18
  - `vllm/model_executor/models/openpangu.py` modified +3/-2 (5 lines); hunks: -29,7 +29,6; -41,7 +40,8
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/attention/mla_attention.py
@@ -191,24 +191,38 @@
-from typing import ClassVar, Generic, TypeVar
+from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar, cast
+if TYPE_CHECKING:
+    from flashinfer import BatchPrefillWithRaggedKVCacheWrapper
+import torch.nn as nn
+import vllm.envs as envs
diff -- vllm/model_executor/layers/attention/attention.py
@@ -1,23 +1,22 @@
-"""Attention layer."""
-from typing import cast
+from typing import TYPE_CHECKING
-from vllm.attention.utils.kv_sharing_utils import validate_kv_sharing_target
-from vllm.attention.utils.kv_transfer_utils import maybe_transfer_kv_layer
+from vllm.model_executor.layers.attention.kv_transfer_utils import (
diff -- vllm/model_executor/layers/attention/__init__.py
@@ -0,0 +1,26 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/attention/mla_attention.py` modified +354/-17; `vllm/model_executor/layers/attention/attention.py` renamed +42/-315; `vllm/model_executor/layers/attention/__init__.py` modified +26/-0; `vllm/model_executor/models/whisper.py` modified +5/-3; `vllm/model_executor/models/openpangu.py` modified +3/-2; `vllm/model_executor/models/apertus.py` modified +2/-2
- Risk and verification: The diff ships test coverage in `tests/compile/test_fusion_attn.py`, `tests/compile/test_qk_norm_rope_fusion.py`, `tests/kernels/attention/test_attention.py`, `tests/kernels/attention/test_mha_attn.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #33191 - Add flake8-implicit-str-concat rules to Ruff

- Link: https://github.com/vllm-project/vllm/pull/33191
- Status/date: merged / 2026-01-28
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 16 files, +34/-33, 201 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add flake8-implicit-str-concat rules to Ruff"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `tests/tool_parsers/test_deepseekv31_tool_parser.py`, `vllm/model_executor/layers/quantization/kernels/scaled_mm/aiter.py`, `vllm/entrypoints/openai/translations/speech_to_text.py`; technical summary: Covers "Add flake8-implicit-str-concat rules to Ruff"; the main implementation surface is `tests/tool_parsers/test_deepseekv31_tool_parser.py`, `vllm/model_executor/layers/quantization/kernels/scaled_mm/aiter.py`, `vllm/entrypoints/openai/translations/speech_to_text.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/tool_parsers/test_deepseekv31_tool_parser.py` modified +8/-8 (16 lines); hunks: -24,9 +24,9 @@ def parser(deepseekv31_tokenizer):; -39,11 +39,11 @@ def test_extract_tool_calls_with_tool(parser):; symbols: parser, test_extract_tool_calls_with_tool, test_extract_tool_calls_with_multiple_tools, touching `parser, test_extract_tool_calls_with_tool, test_extract_tool_calls_with_multiple_tools`; `vllm/model_executor/layers/quantization/kernels/scaled_mm/aiter.py` modified +4/-4 (8 lines); hunks: -32,8 +32,8 @@ def is_supported(; -97,9 +97,9 @@ def apply_weights(; symbols: is_supported, apply_weights, touching `is_supported, apply_weights`; `vllm/entrypoints/openai/translations/speech_to_text.py` modified +2/-2 (4 lines); hunks: -406,8 +406,8 @@ async def _create_speech_to_text(; symbols: _create_speech_to_text, touching `_create_speech_to_text`; `tests/entrypoints/openai/tool_parsers/test_llama4_pythonic_tool_parser.py` modified +1/-1 (2 lines); hunks: -193,7 +193,7 @@ def test_no_tool_call(streaming: bool, default_tokenizer: To...; symbols: test_no_tool_call, touching `test_no_tool_call`.
- Code diff details:
  - `tests/tool_parsers/test_deepseekv31_tool_parser.py` modified +8/-8 (16 lines); hunks: -24,9 +24,9 @@ def parser(deepseekv31_tokenizer):; -39,11 +39,11 @@ def test_extract_tool_calls_with_tool(parser):; symbols: parser, test_extract_tool_calls_with_tool, test_extract_tool_calls_with_multiple_tools
  - `vllm/model_executor/layers/quantization/kernels/scaled_mm/aiter.py` modified +4/-4 (8 lines); hunks: -32,8 +32,8 @@ def is_supported(; -97,9 +97,9 @@ def apply_weights(; symbols: is_supported, apply_weights
  - `vllm/entrypoints/openai/translations/speech_to_text.py` modified +2/-2 (4 lines); hunks: -406,8 +406,8 @@ async def _create_speech_to_text(; symbols: _create_speech_to_text
  - `tests/entrypoints/openai/tool_parsers/test_llama4_pythonic_tool_parser.py` modified +1/-1 (2 lines); hunks: -193,7 +193,7 @@ def test_no_tool_call(streaming: bool, default_tokenizer: To...; symbols: test_no_tool_call
  - `vllm/reasoning/olmo3_reasoning_parser.py` modified +1/-1 (2 lines); hunks: -234,7 +234,7 @@ def __init__(self, tokenizer: "TokenizerLike", *args, **kwar...; symbols: __init__
- Key code excerpts:

```diff
diff -- tests/tool_parsers/test_deepseekv31_tool_parser.py
@@ -24,9 +24,9 @@ def parser(deepseekv31_tokenizer):
-        + "<｜tool▁calls▁begin｜>"
-        + '<｜tool▁call▁begin｜>foo<｜tool▁sep｜>{"x":1}<｜tool▁call▁end｜>'
-        + "<｜tool▁calls▁end｜>"
+        "<｜tool▁calls▁begin｜>"
+        '<｜tool▁call▁begin｜>foo<｜tool▁sep｜>{"x":1}<｜tool▁call▁end｜>'
+        "<｜tool▁calls▁end｜>"
diff -- vllm/model_executor/layers/quantization/kernels/scaled_mm/aiter.py
@@ -32,8 +32,8 @@ def is_supported(
-                + "and `VLLM_ROCM_USE_AITER_LINEAR=1`. "
-                + "`VLLM_ROCM_USE_AITER_LINEAR` default is True.",
+                "and `VLLM_ROCM_USE_AITER_LINEAR=1`. "
+                "`VLLM_ROCM_USE_AITER_LINEAR` default is True.",
@@ -97,9 +97,9 @@ def apply_weights(
-            + " and per-token-per-channel GEMM through AITER"
diff -- vllm/entrypoints/openai/translations/speech_to_text.py
@@ -406,8 +406,8 @@ async def _create_speech_to_text(
```

- Reviewed files:
  - tests: `tests/tool_parsers/test_deepseekv31_tool_parser.py` modified +8/-8; `tests/entrypoints/openai/tool_parsers/test_llama4_pythonic_tool_parser.py` modified +1/-1
  - runtime: `vllm/model_executor/layers/quantization/kernels/scaled_mm/aiter.py` modified +4/-4; `vllm/entrypoints/openai/translations/speech_to_text.py` modified +2/-2; `vllm/reasoning/olmo3_reasoning_parser.py` modified +1/-1; `vllm/compilation/wrapper.py` modified +2/-2
  - docs: `examples/others/lmcache/disagg_prefill_lmcache_v1/disagg_proxy_server.py` modified +4/-4
  - other: `csrc/quantization/machete/generate.py` modified +3/-3
- Risk and verification: The diff ships test coverage in `tests/entrypoints/openai/tool_parsers/test_llama4_pythonic_tool_parser.py`, `tests/tool_parsers/test_deepseekv31_tool_parser.py`, `tests/utils.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #33174 - Add support for Mistral Large 3 inference with Flashinfer MoE

- Link: https://github.com/vllm-project/vllm/pull/33174
- Status/date: merged / 2026-01-31
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 16 files, +1104/-31, 1278 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add support for Mistral Large 3 inference with Flashinfer MoE"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200,dtype=fp8_w8a8.json`, `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200.json`, `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_GB200,dtype=fp8_w8a8.json`; technical summary: Covers "Add support for Mistral Large 3 inference with Flashinfer MoE"; the main implementation surface is `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200,dtype=fp8_w8a8.json`, `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200.json`, `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_GB200,dtype=fp8_w8a8.json`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200,dtype=fp8_w8a8.json` added +147/-0 (147 lines); hunks: -0,0 +1,147; `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200.json` added +147/-0 (147 lines); hunks: -0,0 +1,147; `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_GB200,dtype=fp8_w8a8.json` added +147/-0 (147 lines); hunks: -0,0 +1,147; `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json` added +147/-0 (147 lines); hunks: -0,0 +1,147.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200,dtype=fp8_w8a8.json` added +147/-0 (147 lines); hunks: -0,0 +1,147
  - `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200.json` added +147/-0 (147 lines); hunks: -0,0 +1,147
  - `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_GB200,dtype=fp8_w8a8.json` added +147/-0 (147 lines); hunks: -0,0 +1,147
  - `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json` added +147/-0 (147 lines); hunks: -0,0 +1,147
  - `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_H200.json` added +147/-0 (147 lines); hunks: -0,0 +1,147
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200,dtype=fp8_w8a8.json
@@ -0,0 +1,147 @@
+{
+    "triton_version": "3.4.0",
+    "1": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 32,
+        "BLOCK_SIZE_K": 256,
diff -- vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200.json
@@ -0,0 +1,147 @@
+{
+    "triton_version": "3.4.0",
+    "1": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 32,
+        "BLOCK_SIZE_K": 64,
diff -- vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_GB200,dtype=fp8_w8a8.json
@@ -0,0 +1,147 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200,dtype=fp8_w8a8.json` added +147/-0; `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200.json` added +147/-0; `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_GB200,dtype=fp8_w8a8.json` added +147/-0; `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json` added +147/-0; `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_H200.json` added +147/-0; `vllm/model_executor/layers/fused_moe/configs/E=16,N=4096,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json` added +147/-0
- Risk and verification: The diff ships test coverage in `tests/kernels/moe/test_flashinfer.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #33858 - [Bugfix] Kimi-K2 grouped_topk usage for Flashinfer monolithic kernels.

- Link: https://github.com/vllm-project/vllm/pull/33858
- Status/date: merged / 2026-02-05
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-11, 28 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Kimi-K2 grouped_topk usage for Flashinfer monolithic kernels."; model line: DeepSeek V3.1; category: bug fix; main diff: `vllm/model_executor/models/deepseek_v2.py`; technical summary: Covers "[Bugfix] Kimi-K2 grouped_topk usage for Flashinfer monolithic kernels."; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +3/-11 (14 lines); hunks: -295,14 +295,6 @@ def __init__(; -313,9 +305,9 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +3/-11 (14 lines); hunks: -295,14 +295,6 @@ def __init__(; -313,9 +305,9 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -295,14 +295,6 @@ def __init__(
-        n_group = getattr(config, "n_group", 1)
-        topk_group = getattr(config, "topk_group", 1)
-        use_grouped_topk = True
-        if (n_group, topk_group) == (1, 1):
-            n_group = None
-            topk_group = None
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +3/-11
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #33876 - [Bugfix] Fix Kimi-K2.5 NVFP4 checkpoints weight loading

- Link: https://github.com/vllm-project/vllm/pull/33876
- Status/date: merged / 2026-02-05
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +15/-5, 53 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix Kimi-K2.5 NVFP4 checkpoints weight loading"; model line: DeepSeek V3.1; category: bug fix; main diff: `vllm/model_executor/models/kimi_k25.py`, `vllm/model_executor/models/deepseek_v2.py`; technical summary: Covers "[Bugfix] Fix Kimi-K2.5 NVFP4 checkpoints weight loading"; the main implementation surface is `vllm/model_executor/models/kimi_k25.py`, `vllm/model_executor/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/kimi_k25.py` modified +14/-4 (18 lines); hunks: -24,7 +24,11; -302,7 +306,9 @@ def split_video_chunks(self, video):; symbols: split_video_chunks, KimiK25ForConditionalGeneration, compute_logits, touching `split_video_chunks, KimiK25ForConditionalGeneration, compute_logits`; `vllm/model_executor/models/deepseek_v2.py` modified +1/-1 (2 lines); hunks: -1493,7 +1493,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `vllm/model_executor/models/kimi_k25.py` modified +14/-4 (18 lines); hunks: -24,7 +24,11; -302,7 +306,9 @@ def split_video_chunks(self, video):; symbols: split_video_chunks, KimiK25ForConditionalGeneration, compute_logits
  - `vllm/model_executor/models/deepseek_v2.py` modified +1/-1 (2 lines); hunks: -1493,7 +1493,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: load_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/kimi_k25.py
@@ -24,7 +24,11 @@
-from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP
+from vllm.model_executor.models.interfaces import (
+    SupportsMultiModal,
+    SupportsPP,
+    SupportsQuant,
+)
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -1493,7 +1493,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
-            if not is_fusion_moe_shared_experts_layer:
+            if name is not None and not is_fusion_moe_shared_experts_layer:
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/kimi_k25.py` modified +14/-4; `vllm/model_executor/models/deepseek_v2.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/kimi_k25.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #34124 - [Model] GLM adaptation

- Link: https://github.com/vllm-project/vllm/pull/34124
- Status/date: merged / 2026-02-09
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +13/-3, 72 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] GLM adaptation"; model line: DeepSeek V3.1; category: docs/tests/CI; main diff: `vllm/model_executor/models/deepseek_v2.py`, `tests/models/registry.py`, `tests/models/test_initialization.py`; technical summary: Covers "[Model] GLM adaptation"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`, `tests/models/registry.py`, `tests/models/test_initialization.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +5/-1 (6 lines); hunks: -836,7 +836,7 @@ def __init__(; -1499,6 +1499,10 @@ class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM):; symbols: __init__, DeepseekV3ForCausalLM, GlmMoeDsaForCausalLM, get_spec_layer_idx_from_weight_name, touching `__init__, DeepseekV3ForCausalLM, GlmMoeDsaForCausalLM`; `tests/models/registry.py` modified +3/-0 (3 lines); hunks: -275,6 +275,9 @@ def check_available_online(; symbols: check_available_online, touching `check_available_online`; `tests/models/test_initialization.py` modified +1/-1 (2 lines); hunks: -97,7 +97,7 @@ def _initialize_kv_caches_v1(self, vllm_config):; symbols: _initialize_kv_caches_v1, touching `_initialize_kv_caches_v1`; `vllm/model_executor/models/registry.py` modified +1/-0 (1 lines); hunks: -114,6 +114,7.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +5/-1 (6 lines); hunks: -836,7 +836,7 @@ def __init__(; -1499,6 +1499,10 @@ class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM):; symbols: __init__, DeepseekV3ForCausalLM, GlmMoeDsaForCausalLM, get_spec_layer_idx_from_weight_name
  - `tests/models/registry.py` modified +3/-0 (3 lines); hunks: -275,6 +275,9 @@ def check_available_online(; symbols: check_available_online
  - `tests/models/test_initialization.py` modified +1/-1 (2 lines); hunks: -97,7 +97,7 @@ def _initialize_kv_caches_v1(self, vllm_config):; symbols: _initialize_kv_caches_v1
  - `vllm/model_executor/models/registry.py` modified +1/-0 (1 lines); hunks: -114,6 +114,7
  - `vllm/config/speculative.py` modified +1/-1 (2 lines); hunks: -181,7 +181,7 @@ def compute_hash(self) -> str:; symbols: compute_hash, hf_config_override
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -836,7 +836,7 @@ def __init__(
-                is_neox_style=True,
+                is_neox_style=not getattr(config, "indexer_rope_interleave", True),
@@ -1499,6 +1499,10 @@ class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM):
+class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM):
+    pass
diff -- tests/models/registry.py
@@ -275,6 +275,9 @@ def check_available_online(
+    "GlmMoeDsaForCausalLM": _HfExamplesInfo(
+        "zai-org/GLM-5", min_transformers_version="5.0.1", is_available_online=False
+    ),
diff -- tests/models/test_initialization.py
@@ -97,7 +97,7 @@ def _initialize_kv_caches_v1(self, vllm_config):
-    if model_arch == "DeepseekV32ForCausalLM":
+    if model_arch in ["DeepseekV32ForCausalLM", "GlmMoeDsaForCausalLM"]:
diff -- vllm/model_executor/models/registry.py
@@ -114,6 +114,7 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +5/-1; `vllm/model_executor/models/registry.py` modified +1/-0; `vllm/config/speculative.py` modified +1/-1; `vllm/transformers_utils/model_arch_config_convertor.py` modified +1/-0
  - tests: `tests/models/registry.py` modified +3/-0; `tests/models/test_initialization.py` modified +1/-1
  - other: `benchmarks/kernels/benchmark_moe.py` modified +1/-0
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`, `tests/models/test_initialization.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #34353 - [Bugfix] fix default is_neox_style to be True for deepseekv3.2

- Link: https://github.com/vllm-project/vllm/pull/34353
- Status/date: merged / 2026-02-11
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] fix default is_neox_style to be True for deepseekv3.2"; model line: DeepSeek V3.1; category: bug fix; main diff: `vllm/model_executor/models/deepseek_v2.py`; technical summary: Covers "[Bugfix] fix default is_neox_style to be True for deepseekv3.2"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +1/-1 (2 lines); hunks: -836,7 +836,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +1/-1 (2 lines); hunks: -836,7 +836,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -836,7 +836,7 @@ def __init__(
-                is_neox_style=not getattr(config, "indexer_rope_interleave", True),
+                is_neox_style=not getattr(config, "indexer_rope_interleave", False),
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #34514 - [CI][BugFix] ShellCheck cleanup to remove baseline and preserve runtime behavior

- Link: https://github.com/vllm-project/vllm/pull/34514
- Status/date: merged / 2026-02-17
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 55 files, +338/-464, 2137 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CI][BugFix] ShellCheck cleanup to remove baseline and preserve runtime behavior"; model line: DeepSeek V3.1; category: bug fix; main diff: `tests/v1/ec_connector/integration/run_epd_correctness_test.sh`, `tools/pre_commit/shellcheck.baseline`, `benchmarks/auto_tune/auto_tune.sh`; technical summary: Covers "[CI][BugFix] ShellCheck cleanup to remove baseline and preserve runtime behavior"; the main implementation surface is `tests/v1/ec_connector/integration/run_epd_correctness_test.sh`, `tools/pre_commit/shellcheck.baseline`, `benchmarks/auto_tune/auto_tune.sh`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/v1/ec_connector/integration/run_epd_correctness_test.sh` modified +54/-54 (108 lines); hunks: -24,7 +24,7 @@ MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"; -51,7 +51,7 @@ LOG_PATH="${LOG_PATH:-/tmp}"; `tools/pre_commit/shellcheck.baseline` removed +0/-89 (89 lines); hunks: -1,89 +0,0; `benchmarks/auto_tune/auto_tune.sh` modified +22/-22 (44 lines); hunks: -46,10 +46,10 @@ echo "VLLM_LOGGING_LEVEL=$VLLM_LOGGING_LEVEL"; -114,7 +114,7 @@ start_server() {; `tools/pre_commit/shellcheck.sh` modified +3/-36 (39 lines); hunks: -2,7 +2,6; -20,38 +19,6 @@ if ! [ -x "$(command -v shellcheck)" ]; then.
- Code diff details:
  - `tests/v1/ec_connector/integration/run_epd_correctness_test.sh` modified +54/-54 (108 lines); hunks: -24,7 +24,7 @@ MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"; -51,7 +51,7 @@ LOG_PATH="${LOG_PATH:-/tmp}"
  - `tools/pre_commit/shellcheck.baseline` removed +0/-89 (89 lines); hunks: -1,89 +0,0
  - `benchmarks/auto_tune/auto_tune.sh` modified +22/-22 (44 lines); hunks: -46,10 +46,10 @@ echo "VLLM_LOGGING_LEVEL=$VLLM_LOGGING_LEVEL"; -114,7 +114,7 @@ start_server() {
  - `tools/pre_commit/shellcheck.sh` modified +3/-36 (39 lines); hunks: -2,7 +2,6; -20,38 +19,6 @@ if ! [ -x "$(command -v shellcheck)" ]; then
  - `.buildkite/scripts/hardware_ci/run-npu-test.sh` modified +15/-20 (35 lines); hunks: -41,16 +41,16 @@ get_config() {; -62,14 +62,14 @@ agent_idx=$(echo "${BUILDKITE_AGENT_NAME}" | awk -F'-' '{pri...
- Key code excerpts:

```diff
diff -- tests/v1/ec_connector/integration/run_epd_correctness_test.sh
@@ -24,7 +24,7 @@ MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
-if [ $USE_MM_PROMPTS = "1" ]; then
+if [ "$USE_MM_PROMPTS" = "1" ]; then
@@ -51,7 +51,7 @@ LOG_PATH="${LOG_PATH:-/tmp}"
-mkdir -p $LOG_PATH
+mkdir -p "$LOG_PATH"
@@ -87,20 +87,20 @@ run_baseline() {
diff -- tools/pre_commit/shellcheck.baseline
@@ -1,89 +0,0 @@
-benchmarks/auto_tune/auto_tune.sh:SC2034
-benchmarks/auto_tune/auto_tune.sh:SC2086
-benchmarks/auto_tune/batch_auto_tune.sh:SC2086
-benchmarks/run_structured_output_benchmark.sh:SC2028
-benchmarks/run_structured_output_benchmark.sh:SC2034
-benchmarks/run_structured_output_benchmark.sh:SC2086
diff -- benchmarks/auto_tune/auto_tune.sh
@@ -46,10 +46,10 @@ echo "VLLM_LOGGING_LEVEL=$VLLM_LOGGING_LEVEL"
```

- Reviewed files:
  - tests: `tests/v1/ec_connector/integration/run_epd_correctness_test.sh` modified +54/-54
  - other: `tools/pre_commit/shellcheck.baseline` removed +0/-89; `benchmarks/auto_tune/auto_tune.sh` modified +22/-22; `tools/pre_commit/shellcheck.sh` modified +3/-36; `.buildkite/scripts/hardware_ci/run-npu-test.sh` modified +15/-20; `benchmarks/run_structured_output_benchmark.sh` modified +16/-14
  - docs: `examples/online_serving/disaggregated_encoder/disagg_1e1p1d_example.sh` modified +15/-15; `examples/online_serving/disaggregated_encoder/disagg_1e1pd_example.sh` modified +13/-13
- Risk and verification: The diff ships test coverage in `.buildkite/scripts/scheduled_integration_test/deepseek_v2_lite_ep_eplb.sh`, `.buildkite/scripts/scheduled_integration_test/qwen30b_a3b_fp8_block_ep_eplb.sh`, `.buildkite/scripts/scheduled_integration_test/qwen3_next_mtp_async_eplb.sh`, `tests/standalone_tests/python_only_compile.sh`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #34758 - [Model Bash] DeepSeek R1 BF16 Min Latency QKV A GEMM (0.5% E2E Speedup)

- Link: https://github.com/vllm-project/vllm/pull/34758
- Status/date: merged / 2026-02-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +855/-3, 917 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model Bash] DeepSeek R1 BF16 Min Latency QKV A GEMM (0.5% E2E Speedup)"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/layers/mla.py`, `csrc/dsv3_fused_a_gemm.cu`; technical summary: Covers "[Model Bash] DeepSeek R1 BF16 Min Latency QKV A GEMM (0.5% E2E Speedup)"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/layers/mla.py`, `csrc/dsv3_fused_a_gemm.cu`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +60/-3 (63 lines); hunks: -32,6 +32,7; -711,6 +712,64 @@ def forward(; symbols: forward, DeepSeekV2FusedQkvAProj, __init__, DeepseekV2MLAAttention, touching `forward, DeepSeekV2FusedQkvAProj, __init__`; `vllm/model_executor/layers/mla.py` modified +1/-0 (1 lines); hunks: -129,6 +129,7 @@ def forward(; symbols: forward, touching `forward`; `csrc/dsv3_fused_a_gemm.cu` added +747/-0 (747 lines); hunks: -0,0 +1,747; symbols: Type, touching `Type`; `CMakeLists.txt` modified +19/-0 (19 lines); hunks: -771,6 +771,25 @@ if(VLLM_GPU_LANG STREQUAL "CUDA").
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +60/-3 (63 lines); hunks: -32,6 +32,7; -711,6 +712,64 @@ def forward(; symbols: forward, DeepSeekV2FusedQkvAProj, __init__, DeepseekV2MLAAttention
  - `vllm/model_executor/layers/mla.py` modified +1/-0 (1 lines); hunks: -129,6 +129,7 @@ def forward(; symbols: forward
  - `csrc/dsv3_fused_a_gemm.cu` added +747/-0 (747 lines); hunks: -0,0 +1,747; symbols: Type
  - `CMakeLists.txt` modified +19/-0 (19 lines); hunks: -771,6 +771,25 @@ if(VLLM_GPU_LANG STREQUAL "CUDA")
  - `vllm/_custom_ops.py` modified +18/-0 (18 lines); hunks: -2770,6 +2770,24 @@ def sm100_cutlass_mla_get_workspace_size(; symbols: sm100_cutlass_mla_get_workspace_size, dsv3_fused_a_gemm
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -32,6 +32,7 @@
+import vllm._custom_ops as ops
@@ -711,6 +712,64 @@ def forward(
+class DeepSeekV2FusedQkvAProj(MergedColumnParallelLinear):
+    def __init__(
+        self,
+        input_size: int,
diff -- vllm/model_executor/layers/mla.py
@@ -129,6 +129,7 @@ def forward(
diff -- csrc/dsv3_fused_a_gemm.cu
@@ -0,0 +1,747 @@
+/*
+ * Adapted from
+ * https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/gemm/dsv3_fused_a_gemm.cu
+ * which was adapted from
+ * https://github.com/NVIDIA/TensorRT-LLM/blob/619709fc33bd5dc268f19d6a741fe7ed51c0f8f5/cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3FusedAGemm.cu
+ *
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +60/-3; `vllm/model_executor/layers/mla.py` modified +1/-0; `vllm/_custom_ops.py` modified +18/-0
  - other: `csrc/dsv3_fused_a_gemm.cu` added +747/-0; `CMakeLists.txt` modified +19/-0; `csrc/ops.h` modified +5/-0; `csrc/torch_bindings.cpp` modified +5/-0
- Risk and verification: Runtime changes concentrate in `vllm/_custom_ops.py`, `vllm/model_executor/layers/mla.py`, `vllm/model_executor/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #34876 - [Bug] Fix DeepSeek V3 weight loading caused by incorrect prefix

- Link: https://github.com/vllm-project/vllm/pull/34876
- Status/date: merged / 2026-02-19
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-2, 18 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bug] Fix DeepSeek V3 weight loading caused by incorrect prefix"; model line: DeepSeek V3.1; category: bug fix; main diff: `vllm/model_executor/models/deepseek_v2.py`; technical summary: Covers "[Bug] Fix DeepSeek V3 weight loading caused by incorrect prefix"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +2/-2 (4 lines); hunks: -716,7 +716,7 @@ class DeepSeekV2FusedQkvAProj(MergedColumnParallelLinear):; -726,7 +726,7 @@ def __init__(; symbols: DeepSeekV2FusedQkvAProj, __init__, touching `DeepSeekV2FusedQkvAProj, __init__`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +2/-2 (4 lines); hunks: -716,7 +716,7 @@ class DeepSeekV2FusedQkvAProj(MergedColumnParallelLinear):; -726,7 +726,7 @@ def __init__(; symbols: DeepSeekV2FusedQkvAProj, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -716,7 +716,7 @@ class DeepSeekV2FusedQkvAProj(MergedColumnParallelLinear):
-        output_size: int,
+        output_size: list[int],
@@ -726,7 +726,7 @@ def __init__(
-            prefix=f"{prefix}.kv_a_proj_with_mqa",
+            prefix=prefix,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #34302 - [ModelBash][DSV3] Add TRTLLM DSV3 Router GEMM kernel (6% B1 Speedup)

- Link: https://github.com/vllm-project/vllm/pull/34302
- Status/date: merged / 2026-02-23
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +915/-3, 971 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ModelBash][DSV3] Add TRTLLM DSV3 Router GEMM kernel (6% B1 Speedup)"; model line: DeepSeek V3.1; category: model support/runtime entry; main diff: `vllm/model_executor/models/deepseek_v2.py`, `csrc/moe/dsv3_router_gemm_bf16_out.cu`, `csrc/moe/dsv3_router_gemm_float_out.cu`; technical summary: Covers "[ModelBash][DSV3] Add TRTLLM DSV3 Router GEMM kernel (6% B1 Speedup)"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`, `csrc/moe/dsv3_router_gemm_bf16_out.cu`, `csrc/moe/dsv3_router_gemm_float_out.cu`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +75/-2 (77 lines); hunks: -221,6 +221,73 @@ def forward(self, x):; -249,10 +316,9 @@ def __init__(; symbols: forward, DeepSeekV2Gate, __init__, set_out_dtype, touching `forward, DeepSeekV2Gate, __init__`; `csrc/moe/dsv3_router_gemm_bf16_out.cu` added +291/-0 (291 lines); hunks: -0,0 +1,291; `csrc/moe/dsv3_router_gemm_float_out.cu` added +291/-0 (291 lines); hunks: -0,0 +1,291; `csrc/moe/dsv3_router_gemm_entry.cu` added +163/-0 (163 lines); hunks: -0,0 +1,163.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +75/-2 (77 lines); hunks: -221,6 +221,73 @@ def forward(self, x):; -249,10 +316,9 @@ def __init__(; symbols: forward, DeepSeekV2Gate, __init__, set_out_dtype
  - `csrc/moe/dsv3_router_gemm_bf16_out.cu` added +291/-0 (291 lines); hunks: -0,0 +1,291
  - `csrc/moe/dsv3_router_gemm_float_out.cu` added +291/-0 (291 lines); hunks: -0,0 +1,291
  - `csrc/moe/dsv3_router_gemm_entry.cu` added +163/-0 (163 lines); hunks: -0,0 +1,163
  - `csrc/moe/dsv3_router_gemm_utils.h` added +43/-0 (43 lines); hunks: -0,0 +1,43
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -221,6 +221,73 @@ def forward(self, x):
+class DeepSeekV2Gate(ReplicatedLinear):
+    def __init__(
+        self,
+        hidden_size: int,
+        n_experts: int,
+        quant_config: QuantizationConfig | None = None,
diff -- csrc/moe/dsv3_router_gemm_bf16_out.cu
@@ -0,0 +1,291 @@
+/*
+ * Adapted from SGLang's sgl-kernel implementation, which was adapted from
+ * https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3RouterGemm.cu
+ * https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/thop/dsv3RouterGemmOp.cpp
+ *
+ * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
diff -- csrc/moe/dsv3_router_gemm_float_out.cu
@@ -0,0 +1,291 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +75/-2; `vllm/_custom_ops.py` modified +15/-0
  - other: `csrc/moe/dsv3_router_gemm_bf16_out.cu` added +291/-0; `csrc/moe/dsv3_router_gemm_float_out.cu` added +291/-0; `csrc/moe/dsv3_router_gemm_entry.cu` added +163/-0; `csrc/moe/dsv3_router_gemm_utils.h` added +43/-0; `CMakeLists.txt` modified +21/-0; `csrc/moe/moe_ops.h` modified +12/-1
- Risk and verification: Runtime changes concentrate in `vllm/_custom_ops.py`, `vllm/model_executor/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #33724 - [WideEP] Remove pplx all2all backend

- Link: https://github.com/vllm-project/vllm/pull/33724
- Status/date: merged / 2026-02-26
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 39 files, +107/-2069, 2692 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[WideEP] Remove pplx all2all backend"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `vllm/model_executor/layers/fused_moe/pplx_prepare_finalize.py`, `vllm/model_executor/layers/fused_moe/all2all_utils.py`, `vllm/model_executor/layers/fused_moe/config.py`; technical summary: Covers "[WideEP] Remove pplx all2all backend"; the main implementation surface is `vllm/model_executor/layers/fused_moe/pplx_prepare_finalize.py`, `vllm/model_executor/layers/fused_moe/all2all_utils.py`, `vllm/model_executor/layers/fused_moe/config.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/fused_moe/pplx_prepare_finalize.py` removed +0/-373 (373 lines); hunks: -1,373 +0,0; symbols: pplx_hidden_dim_scale_bytes, PplxPrepareAndFinalize, __init__, activation_format, touching `pplx_hidden_dim_scale_bytes, PplxPrepareAndFinalize, __init__`; `vllm/model_executor/layers/fused_moe/all2all_utils.py` modified +4/-49 (53 lines); hunks: -1,6 +1,7; -24,16 +25,11; symbols: maybe_make_prepare_finalize, touching `maybe_make_prepare_finalize`; `vllm/model_executor/layers/fused_moe/config.py` modified +1/-9 (10 lines); hunks: -939,10 +939,6 @@ def is_sequence_parallel(self) -> bool:; -962,7 +958,7 @@ def use_fi_all2allv_kernels(self):; symbols: is_sequence_parallel, use_all2all_kernels, use_pplx_kernels, use_deepep_ht_kernels, touching `is_sequence_parallel, use_all2all_kernels, use_pplx_kernels`; `vllm/model_executor/layers/fused_moe/topk_weight_and_reduce.py` modified +5/-4 (9 lines); hunks: -14,10 +14,11 @@ class TopKWeightAndReduceDelegate(mk.TopKWeightAndReduce):; symbols: TopKWeightAndReduceDelegate, touching `TopKWeightAndReduceDelegate`.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/pplx_prepare_finalize.py` removed +0/-373 (373 lines); hunks: -1,373 +0,0; symbols: pplx_hidden_dim_scale_bytes, PplxPrepareAndFinalize, __init__, activation_format
  - `vllm/model_executor/layers/fused_moe/all2all_utils.py` modified +4/-49 (53 lines); hunks: -1,6 +1,7; -24,16 +25,11; symbols: maybe_make_prepare_finalize
  - `vllm/model_executor/layers/fused_moe/config.py` modified +1/-9 (10 lines); hunks: -939,10 +939,6 @@ def is_sequence_parallel(self) -> bool:; -962,7 +958,7 @@ def use_fi_all2allv_kernels(self):; symbols: is_sequence_parallel, use_all2all_kernels, use_pplx_kernels, use_deepep_ht_kernels
  - `vllm/model_executor/layers/fused_moe/topk_weight_and_reduce.py` modified +5/-4 (9 lines); hunks: -14,10 +14,11 @@ class TopKWeightAndReduceDelegate(mk.TopKWeightAndReduce):; symbols: TopKWeightAndReduceDelegate
  - `vllm/model_executor/layers/fused_moe/fused_batched_moe.py` modified +3/-3 (6 lines); hunks: -493,7 +493,7 @@ class BatchedPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):; -648,7 +648,7 @@ def finalize(; symbols: BatchedPrepareAndFinalize, that, __init__, finalize
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/fused_moe/pplx_prepare_finalize.py
@@ -1,373 +0,0 @@
-# SPDX-License-Identifier: Apache-2.0
-# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-from collections.abc import Callable
-import pplx_kernels as pplx
-import torch
-import vllm.model_executor.layers.fused_moe.modular_kernel as mk
diff -- vllm/model_executor/layers/fused_moe/all2all_utils.py
@@ -1,6 +1,7 @@
+from typing import Any
@@ -24,16 +25,11 @@
-from vllm.utils.import_utils import has_deep_ep, has_mori, has_pplx
+from vllm.utils.import_utils import has_deep_ep, has_mori
-    if has_pplx():
-        from .pplx_prepare_finalize import (
diff -- vllm/model_executor/layers/fused_moe/config.py
@@ -939,10 +939,6 @@ def is_sequence_parallel(self) -> bool:
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/pplx_prepare_finalize.py` removed +0/-373; `vllm/model_executor/layers/fused_moe/all2all_utils.py` modified +4/-49; `vllm/model_executor/layers/fused_moe/config.py` modified +1/-9; `vllm/model_executor/layers/fused_moe/topk_weight_and_reduce.py` modified +5/-4; `vllm/model_executor/layers/fused_moe/fused_batched_moe.py` modified +3/-3; `vllm/model_executor/layers/fused_moe/modular_kernel.py` modified +3/-3
- Risk and verification: The diff ships test coverage in `tests/kernels/moe/modular_kernel_tools/common.py`, `tests/kernels/moe/modular_kernel_tools/mk_objects.py`, `tests/kernels/moe/modular_kernel_tools/profile_modular_kernel.py`, `tests/kernels/moe/test_modular_kernel_combinations.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #35121 - [Performance] Cublas Bf16 Gate with Fp32 Output

- Link: https://github.com/vllm-project/vllm/pull/35121
- Status/date: merged / 2026-02-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +206/-80, 390 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Performance] Cublas Bf16 Gate with Fp32 Output"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `vllm/model_executor/layers/fused_moe/router/gate_linear.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/nemotron_h.py`; technical summary: Covers "[Performance] Cublas Bf16 Gate with Fp32 Output"; the main implementation surface is `vllm/model_executor/layers/fused_moe/router/gate_linear.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/nemotron_h.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/fused_moe/router/gate_linear.py` added +117/-0 (117 lines); hunks: -0,0 +1,117; symbols: GateLinear, __init__, set_out_dtype, forward, touching `GateLinear, __init__, set_out_dtype`; `vllm/model_executor/models/deepseek_v2.py` modified +2/-70 (72 lines); hunks: -47,7 +47,7; -221,73 +221,6 @@ def forward(self, x):; symbols: forward, DeepSeekV2Gate, __init__, set_out_dtype, touching `forward, DeepSeekV2Gate, __init__`; `vllm/model_executor/models/nemotron_h.py` modified +6/-9 (15 lines); hunks: -34,7 +34,7; -148,13 +148,11 @@ def __init__(; symbols: __init__, forward, _get_max_n_routed_experts, get_expert_mapping, touching `__init__, forward, _get_max_n_routed_experts`; `vllm/model_executor/layers/fused_moe/__init__.py` modified +2/-0 (2 lines); hunks: -28,6 +28,7; -64,6 +65,7 @@ def get_config() -> dict[str, Any] | None:; symbols: get_config, touching `get_config`.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/router/gate_linear.py` added +117/-0 (117 lines); hunks: -0,0 +1,117; symbols: GateLinear, __init__, set_out_dtype, forward
  - `vllm/model_executor/models/deepseek_v2.py` modified +2/-70 (72 lines); hunks: -47,7 +47,7; -221,73 +221,6 @@ def forward(self, x):; symbols: forward, DeepSeekV2Gate, __init__, set_out_dtype
  - `vllm/model_executor/models/nemotron_h.py` modified +6/-9 (15 lines); hunks: -34,7 +34,7; -148,13 +148,11 @@ def __init__(; symbols: __init__, forward, _get_max_n_routed_experts, get_expert_mapping
  - `vllm/model_executor/layers/fused_moe/__init__.py` modified +2/-0 (2 lines); hunks: -28,6 +28,7; -64,6 +65,7 @@ def get_config() -> dict[str, Any] | None:; symbols: get_config
  - `csrc/moe/router_gemm.cu` added +52/-0 (52 lines); hunks: -0,0 +1,52
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/fused_moe/router/gate_linear.py
@@ -0,0 +1,117 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import torch
+from torch.nn.parameter import Parameter
+from vllm.model_executor.custom_op import PluggableLayer
+from vllm.model_executor.layers.linear import ReplicatedLinear
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -47,7 +47,7 @@
-from vllm.model_executor.layers.fused_moe import SharedFusedMoE
+from vllm.model_executor.layers.fused_moe import GateLinear, SharedFusedMoE
@@ -221,73 +221,6 @@ def forward(self, x):
-class DeepSeekV2Gate(ReplicatedLinear):
-    def __init__(
-        self,
diff -- vllm/model_executor/models/nemotron_h.py
@@ -34,7 +34,7 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/router/gate_linear.py` added +117/-0; `vllm/model_executor/models/deepseek_v2.py` modified +2/-70; `vllm/model_executor/models/nemotron_h.py` modified +6/-9; `vllm/model_executor/layers/fused_moe/__init__.py` modified +2/-0; `vllm/_custom_ops.py` modified +17/-0
  - other: `csrc/moe/router_gemm.cu` added +52/-0; `csrc/moe/moe_ops.h` modified +4/-0; `csrc/moe/torch_bindings.cpp` modified +4/-0
- Risk and verification: Runtime changes concentrate in `vllm/_custom_ops.py`, `vllm/model_executor/layers/fused_moe/__init__.py`, `vllm/model_executor/layers/fused_moe/router/gate_linear.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #35548 - [MTP] Validate that MTP weights are actually loaded

- Link: https://github.com/vllm-project/vllm/pull/35548
- Status/date: merged / 2026-02-28
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +20/-0, 27 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MTP] Validate that MTP weights are actually loaded"; model line: DeepSeek V3.1; category: model implementation change; main diff: `vllm/model_executor/models/deepseek_mtp.py`; technical summary: Covers "[MTP] Validate that MTP weights are actually loaded"; the main implementation surface is `vllm/model_executor/models/deepseek_mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_mtp.py` modified +20/-0 (20 lines); hunks: -415,6 +415,26 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: load_weights, _rewrite_spec_layer_name, touching `load_weights, _rewrite_spec_layer_name`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_mtp.py` modified +20/-0 (20 lines); hunks: -415,6 +415,26 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: load_weights, _rewrite_spec_layer_name
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_mtp.py
@@ -415,6 +415,26 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
+        # Validate that weights were loaded for each expected MTP layer.
+        loaded_layers: set[int] = set()
+        for param_name in loaded_params:
+            spec_layer = get_spec_layer_idx_from_weight_name(self.config, param_name)
+            if spec_layer is not None:
+                loaded_layers.add(spec_layer)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_mtp.py` modified +20/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_mtp.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #35751 - [MoE][Perf] Wrap DSV3 QKVAProj GEMM in custom op for torch.compile

- Link: https://github.com/vllm-project/vllm/pull/35751
- Status/date: merged / 2026-03-02
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +41/-13, 75 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MoE][Perf] Wrap DSV3 QKVAProj GEMM in custom op for torch.compile"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `vllm/model_executor/models/deepseek_v2.py`; technical summary: Covers "[MoE][Perf] Wrap DSV3 QKVAProj GEMM in custom op for torch.compile"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +41/-13 (54 lines); hunks: -75,6 +75,7; -717,6 +718,44 @@ def forward(; symbols: forward, _min_latency_fused_qkv_a_proj_impl, _min_latency_fused_qkv_a_proj_fake, DeepSeekV2FusedQkvAProj, touching `forward, _min_latency_fused_qkv_a_proj_impl, _min_latency_fused_qkv_a_proj_fake`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +41/-13 (54 lines); hunks: -75,6 +75,7; -717,6 +718,44 @@ def forward(; symbols: forward, _min_latency_fused_qkv_a_proj_impl, _min_latency_fused_qkv_a_proj_fake, DeepSeekV2FusedQkvAProj
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -75,6 +75,7 @@
+from vllm.utils.torch_utils import direct_register_custom_op
@@ -717,6 +718,44 @@ def forward(
+def _min_latency_fused_qkv_a_proj_impl(
+    input_: torch.Tensor,
+    weight: torch.Tensor,
+) -> torch.Tensor:
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +41/-13
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #36247 - [Bugfix] Fix compressed-tensors quantization failure for DeepSeek-R1 on MI300x

- Link: https://github.com/vllm-project/vllm/pull/36247
- Status/date: merged / 2026-03-07
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-2, 18 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix compressed-tensors quantization failure for DeepSeek-R1 on MI300x"; model line: DeepSeek V3.1; category: bug fix; main diff: `vllm/model_executor/models/deepseek_v2.py`; technical summary: Covers "[Bugfix] Fix compressed-tensors quantization failure for DeepSeek-R1 on MI300x"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +2/-2 (4 lines); hunks: -756,7 +756,7 @@ def _min_latency_fused_qkv_a_proj_fake(; -848,7 +848,7 @@ def __init__(; symbols: _min_latency_fused_qkv_a_proj_fake, DeepSeekV2FusedQkvAProj, DeepSeekV2FusedQkvAProjLinear, __init__, touching `_min_latency_fused_qkv_a_proj_fake, DeepSeekV2FusedQkvAProj, DeepSeekV2FusedQkvAProjLinear`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +2/-2 (4 lines); hunks: -756,7 +756,7 @@ def _min_latency_fused_qkv_a_proj_fake(; -848,7 +848,7 @@ def __init__(; symbols: _min_latency_fused_qkv_a_proj_fake, DeepSeekV2FusedQkvAProj, DeepSeekV2FusedQkvAProjLinear, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -756,7 +756,7 @@ def _min_latency_fused_qkv_a_proj_fake(
-class DeepSeekV2FusedQkvAProj(MergedColumnParallelLinear):
+class DeepSeekV2FusedQkvAProjLinear(MergedColumnParallelLinear):
@@ -848,7 +848,7 @@ def __init__(
-            self.fused_qkv_a_proj = DeepSeekV2FusedQkvAProj(
+            self.fused_qkv_a_proj = DeepSeekV2FusedQkvAProjLinear(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #36361 - Kimi k2.5 MLA based eagle3

- Link: https://github.com/vllm-project/vllm/pull/36361
- Status/date: merged / 2026-03-11
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +499/-8, 649 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Kimi k2.5 MLA based eagle3"; model line: DeepSeek V3.1; category: model implementation change; main diff: `vllm/model_executor/models/deepseek_eagle3.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/kimi_k25.py`; technical summary: Covers "Kimi k2.5 MLA based eagle3"; the main implementation surface is `vllm/model_executor/models/deepseek_eagle3.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/kimi_k25.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_eagle3.py` added +419/-0 (419 lines); hunks: -0,0 +1,419; symbols: DeepseekV2Eagle3DecoderLayer, __init__, _norm_before_residual, _norm_after_residual, touching `DeepseekV2Eagle3DecoderLayer, __init__, _norm_before_residual`; `vllm/model_executor/models/deepseek_v2.py` modified +39/-6 (45 lines); hunks: -82,7 +82,13; -828,6 +834,7 @@ def __init__(; symbols: __init__, embed_input_ids, touching `__init__, embed_input_ids`; `vllm/model_executor/models/kimi_k25.py` modified +14/-1 (15 lines); hunks: -28,6 +28,8; -311,7 +313,12 @@ def split_video_chunks(self, video):; symbols: split_video_chunks, KimiK25ForConditionalGeneration, compute_logits, set_aux_hidden_state_layers, touching `split_video_chunks, KimiK25ForConditionalGeneration, compute_logits`; `tests/models/registry.py` modified +12/-0 (12 lines); hunks: -1137,6 +1137,18 @@ def check_available_online(; symbols: check_available_online, touching `check_available_online`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_eagle3.py` added +419/-0 (419 lines); hunks: -0,0 +1,419; symbols: DeepseekV2Eagle3DecoderLayer, __init__, _norm_before_residual, _norm_after_residual
  - `vllm/model_executor/models/deepseek_v2.py` modified +39/-6 (45 lines); hunks: -82,7 +82,13; -828,6 +834,7 @@ def __init__(; symbols: __init__, embed_input_ids
  - `vllm/model_executor/models/kimi_k25.py` modified +14/-1 (15 lines); hunks: -28,6 +28,8; -311,7 +313,12 @@ def split_video_chunks(self, video):; symbols: split_video_chunks, KimiK25ForConditionalGeneration, compute_logits, set_aux_hidden_state_layers
  - `tests/models/registry.py` modified +12/-0 (12 lines); hunks: -1137,6 +1137,18 @@ def check_available_online(; symbols: check_available_online
  - `vllm/model_executor/models/registry.py` modified +2/-0 (2 lines); hunks: -551,6 +551,8
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_eagle3.py
@@ -0,0 +1,419 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Eagle3 speculative decoding model for DeepseekV2/V3 with MLP (no MoE)."""
+import copy
+from collections.abc import Iterable
+import torch
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -82,7 +82,13 @@
-from .interfaces import MixtureOfExperts, SupportsEagle, SupportsLoRA, SupportsPP
+from .interfaces import (
+    MixtureOfExperts,
+    SupportsEagle,
+    SupportsEagle3,
+    SupportsLoRA,
diff -- vllm/model_executor/models/kimi_k25.py
@@ -28,6 +28,8 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_eagle3.py` added +419/-0; `vllm/model_executor/models/deepseek_v2.py` modified +39/-6; `vllm/model_executor/models/kimi_k25.py` modified +14/-1; `vllm/model_executor/models/registry.py` modified +2/-0; `vllm/v1/spec_decode/eagle.py` modified +8/-1; `vllm/config/speculative.py` modified +4/-0
  - tests: `tests/models/registry.py` modified +12/-0
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #36931 - [Feat][Bugfix] Enable additional dimension for Flashinfer MLA and fix routing dtype

- Link: https://github.com/vllm-project/vllm/pull/36931
- Status/date: merged / 2026-03-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +18/-5, 59 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feat][Bugfix] Enable additional dimension for Flashinfer MLA and fix routing dtype"; model line: DeepSeek V3.1; category: bug fix; main diff: `vllm/model_executor/models/deepseek_v2.py`, `vllm/v1/attention/backends/mla/flashinfer_mla.py`; technical summary: Covers "[Feat][Bugfix] Enable additional dimension for Flashinfer MLA and fix routing dtype"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`, `vllm/v1/attention/backends/mla/flashinfer_mla.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +15/-2 (17 lines); hunks: -47,7 +47,11; -333,8 +337,12 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`; `vllm/v1/attention/backends/mla/flashinfer_mla.py` modified +3/-3 (6 lines); hunks: -75,16 +75,16 @@ def supports_combination(; symbols: supports_combination, touching `supports_combination`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +15/-2 (17 lines); hunks: -47,7 +47,11; -333,8 +337,12 @@ def __init__(; symbols: __init__, forward
  - `vllm/v1/attention/backends/mla/flashinfer_mla.py` modified +3/-3 (6 lines); hunks: -75,16 +75,16 @@ def supports_combination(; symbols: supports_combination
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -47,7 +47,11 @@
-from vllm.model_executor.layers.fused_moe import GateLinear, SharedFusedMoE
+from vllm.model_executor.layers.fused_moe import (
+    GateLinear,
+    RoutingMethodType,
+    SharedFusedMoE,
+)
diff -- vllm/v1/attention/backends/mla/flashinfer_mla.py
@@ -75,16 +75,16 @@ def supports_combination(
-        # FlashInfer MLA kernel requires qk_nope_head_dim == 128
+        # FlashInfer MLA kernel requires qk_nope_head_dim in [64, 128]
-            if qk_nope_head_dim != 128:
+            if qk_nope_head_dim not in [64, 128]:
-                    f"FlashInfer MLA kernel requires qk_nope_head_dim == 128, "
+                    f"FlashInfer MLA kernel requires qk_nope_head_dim in [64, 128], "
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +15/-2; `vllm/v1/attention/backends/mla/flashinfer_mla.py` modified +3/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v2.py`, `vllm/v1/attention/backends/mla/flashinfer_mla.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #37487 - [V0 Deprecation] Refactor kv cache from list to element

- Link: https://github.com/vllm-project/vllm/pull/37487
- Status/date: merged / 2026-03-24
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 27 files, +70/-85, 478 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[V0 Deprecation] Refactor kv cache from list to element"; model line: DeepSeek V3.1; category: model implementation change; main diff: `vllm/model_executor/layers/attention/mla_attention.py`, `vllm/model_executor/layers/attention/attention.py`, `vllm/model_executor/models/extract_hidden_states.py`; technical summary: Covers "[V0 Deprecation] Refactor kv cache from list to element"; the main implementation surface is `vllm/model_executor/layers/attention/mla_attention.py`, `vllm/model_executor/layers/attention/attention.py`, `vllm/model_executor/models/extract_hidden_states.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/attention/mla_attention.py` modified +3/-8 (11 lines); hunks: -416,12 +416,7 @@ def __init__(; -480,7 +475,7 @@ def forward(; symbols: __init__, forward, unified_mla_kv_cache_update, touching `__init__, forward, unified_mla_kv_cache_update`; `vllm/model_executor/layers/attention/attention.py` modified +2/-5 (7 lines); hunks: -350,10 +350,7 @@ def __init__(; -600,7 +597,7 @@ def get_attention_context(; symbols: __init__, get_attention_context, touching `__init__, get_attention_context`; `vllm/model_executor/models/extract_hidden_states.py` modified +2/-5 (7 lines); hunks: -51,7 +51,7 @@ def unified_kv_cache_update(; -288,10 +288,7 @@ def __init__(; symbols: unified_kv_cache_update, __init__, touching `unified_kv_cache_update, __init__`; `vllm/model_executor/layers/sparse_attn_indexer.py` modified +2/-2 (4 lines); hunks: -365,7 +365,7 @@ def forward_cuda(; -389,7 +389,7 @@ def forward_hip(; symbols: forward_cuda, forward_hip, touching `forward_cuda, forward_hip`.
- Code diff details:
  - `vllm/model_executor/layers/attention/mla_attention.py` modified +3/-8 (11 lines); hunks: -416,12 +416,7 @@ def __init__(; -480,7 +475,7 @@ def forward(; symbols: __init__, forward, unified_mla_kv_cache_update
  - `vllm/model_executor/layers/attention/attention.py` modified +2/-5 (7 lines); hunks: -350,10 +350,7 @@ def __init__(; -600,7 +597,7 @@ def get_attention_context(; symbols: __init__, get_attention_context
  - `vllm/model_executor/models/extract_hidden_states.py` modified +2/-5 (7 lines); hunks: -51,7 +51,7 @@ def unified_kv_cache_update(; -288,10 +288,7 @@ def __init__(; symbols: unified_kv_cache_update, __init__
  - `vllm/model_executor/layers/sparse_attn_indexer.py` modified +2/-2 (4 lines); hunks: -365,7 +365,7 @@ def forward_cuda(; -389,7 +389,7 @@ def forward_hip(; symbols: forward_cuda, forward_hip
  - `vllm/model_executor/models/qwen3_next.py` modified +2/-2 (4 lines); hunks: -858,7 +858,7 @@ def _forward_core(; -1046,7 +1046,7 @@ def _forward_core_decode_non_spec(; symbols: _forward_core, _forward_core_decode_non_spec
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/attention/mla_attention.py
@@ -416,12 +416,7 @@ def __init__(
-        self.kv_cache = [
-            torch.tensor([])
-            for _ in range(
-                get_current_vllm_config().parallel_config.pipeline_parallel_size
-            )
-        ]
diff -- vllm/model_executor/layers/attention/attention.py
@@ -350,10 +350,7 @@ def __init__(
-        self.kv_cache = [
-            torch.tensor([])
-            for _ in range(vllm_config.parallel_config.pipeline_parallel_size)
-        ]
+        self.kv_cache = torch.tensor([])
@@ -600,7 +597,7 @@ def get_attention_context(
diff -- vllm/model_executor/models/extract_hidden_states.py
@@ -51,7 +51,7 @@ def unified_kv_cache_update(
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/attention/mla_attention.py` modified +3/-8; `vllm/model_executor/layers/attention/attention.py` modified +2/-5; `vllm/model_executor/models/extract_hidden_states.py` modified +2/-5; `vllm/model_executor/layers/sparse_attn_indexer.py` modified +2/-2; `vllm/model_executor/models/qwen3_next.py` modified +2/-2; `vllm/model_executor/layers/attention/static_sink_attention.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `tests/compile/passes/test_fusion_attn.py`, `tests/compile/passes/test_rope_kvcache_fusion.py`, `tests/v1/e2e/general/test_mamba_prefix_cache.py`, `tests/v1/worker/test_gpu_model_runner.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #38029 - [Tool Parser][1/3] Pass tools to ToolParser constructor

- Link: https://github.com/vllm-project/vllm/pull/38029
- Status/date: merged / 2026-03-26
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 38 files, +147/-92, 858 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Tool Parser][1/3] Pass tools to ToolParser constructor"; model line: DeepSeek V3.1; category: model implementation change; main diff: `vllm/tool_parsers/abstract_tool_parser.py`, `vllm/tool_parsers/qwen3coder_tool_parser.py`, `vllm/tool_parsers/step3p5_tool_parser.py`; technical summary: Covers "[Tool Parser][1/3] Pass tools to ToolParser constructor"; the main implementation surface is `vllm/tool_parsers/abstract_tool_parser.py`, `vllm/tool_parsers/qwen3coder_tool_parser.py`, `vllm/tool_parsers/step3p5_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/tool_parsers/abstract_tool_parser.py` modified +14/-2 (16 lines); hunks: -5,13 +5,18; -30,6 +35,8; symbols: ToolParser, __init__, vocab, touching `ToolParser, __init__, vocab`; `vllm/tool_parsers/qwen3coder_tool_parser.py` modified +5/-7 (12 lines); hunks: -10,7 +10,6; -23,15 +22,16; symbols: Qwen3CoderToolParser, __init__, _reset_streaming_state, _get_arguments_config, touching `Qwen3CoderToolParser, __init__, _reset_streaming_state`; `vllm/tool_parsers/step3p5_tool_parser.py` modified +5/-6 (11 lines); hunks: -11,7 +11,6; -23,7 +22,7; symbols: __init__, setup_parser, set_tools, _reset_xml_parser_after_tool_call, touching `__init__, setup_parser, set_tools`; `vllm/tool_parsers/qwen3xml_tool_parser.py` modified +5/-5 (10 lines); hunks: -11,7 +11,6; -24,6 +23,7; symbols: __init__, setup_parser, set_tools, _reset_xml_parser_after_tool_call, touching `__init__, setup_parser, set_tools`.
- Code diff details:
  - `vllm/tool_parsers/abstract_tool_parser.py` modified +14/-2 (16 lines); hunks: -5,13 +5,18; -30,6 +35,8; symbols: ToolParser, __init__, vocab
  - `vllm/tool_parsers/qwen3coder_tool_parser.py` modified +5/-7 (12 lines); hunks: -10,7 +10,6; -23,15 +22,16; symbols: Qwen3CoderToolParser, __init__, _reset_streaming_state, _get_arguments_config
  - `vllm/tool_parsers/step3p5_tool_parser.py` modified +5/-6 (11 lines); hunks: -11,7 +11,6; -23,7 +22,7; symbols: __init__, setup_parser, set_tools, _reset_xml_parser_after_tool_call
  - `vllm/tool_parsers/qwen3xml_tool_parser.py` modified +5/-5 (10 lines); hunks: -11,7 +11,6; -24,6 +23,7; symbols: __init__, setup_parser, set_tools, _reset_xml_parser_after_tool_call
  - `vllm/tool_parsers/llama4_pythonic_tool_parser.py` modified +7/-2 (9 lines); hunks: -17,6 +17,7; -47,8 +48,12 @@ class Llama4PythonicToolParser(ToolParser):; symbols: Llama4PythonicToolParser, __init__
- Key code excerpts:

```diff
diff -- vllm/tool_parsers/abstract_tool_parser.py
@@ -5,13 +5,18 @@
+from typing import TypeAlias
+from openai.types.responses.tool import Tool as ResponsesTool
-from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
+from vllm.entrypoints.openai.chat_completion.protocol import (
+    ChatCompletionRequest,
+    ChatCompletionToolsParam,
diff -- vllm/tool_parsers/qwen3coder_tool_parser.py
@@ -10,7 +10,6 @@
-    ChatCompletionToolsParam,
@@ -23,15 +22,16 @@
+    Tool,
-    def __init__(self, tokenizer: TokenizerLike):
-        super().__init__(tokenizer)
+    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
diff -- vllm/tool_parsers/step3p5_tool_parser.py
@@ -11,7 +11,6 @@
```

- Reviewed files:
  - runtime: `vllm/tool_parsers/abstract_tool_parser.py` modified +14/-2; `vllm/tool_parsers/qwen3coder_tool_parser.py` modified +5/-7; `vllm/tool_parsers/step3p5_tool_parser.py` modified +5/-6; `vllm/tool_parsers/qwen3xml_tool_parser.py` modified +5/-5; `vllm/tool_parsers/llama4_pythonic_tool_parser.py` modified +7/-2; `vllm/tool_parsers/llama_tool_parser.py` modified +7/-2
- Risk and verification: Runtime changes concentrate in `vllm/entrypoints/openai/chat_completion/serving.py`, `vllm/entrypoints/openai/engine/serving.py`, `vllm/entrypoints/openai/parser/responses_parser.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #38684 - [Perf] DSV3.2 Indexer Fused Weights Projection

- Link: https://github.com/vllm-project/vllm/pull/38684
- Status/date: merged / 2026-04-02
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +25/-14, 79 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Perf] DSV3.2 Indexer Fused Weights Projection"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`; technical summary: Covers "[Perf] DSV3.2 Indexer Fused Weights Projection"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +68/-27, 158 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix DSV32 weight loading"; model line: DeepSeek V3.1; category: bug fix; main diff: `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`; technical summary: Covers "[Bugfix] Fix DSV32 weight loading"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +2039/-483, 2698 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Perf][Kernel] Persistent TopK scheduler: unified CUDAGraph-safe kernel with dynamic per-row dispatch - DeepSeek-V3.2 DSA decode"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `vllm/model_executor/layers/sparse_attn_indexer.py`, `vllm/model_executor/models/deepseek_v2.py`, `csrc/persistent_topk.cuh`; technical summary: Covers "[Perf][Kernel] Persistent TopK scheduler: unified CUDAGraph-safe kernel with dynamic per-row dispatch - DeepSeek-V3.2 DSA decode"; the main implementation surface is `vllm/model_executor/layers/sparse_attn_indexer.py`, `vllm/model_executor/models/deepseek_v2.py`, `csrc/persistent_topk.cuh`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/sparse_attn_indexer.py` modified +24/-24 (48 lines); hunks: -25,6 +25,8; -51,6 +53,7 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer, touching `sparse_attn_indexer`; `vllm/model_executor/models/deepseek_v2.py` modified +6/-2 (8 lines); hunks: -67,7 +67,9; -1203,7 +1205,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, touching `__init__`; `csrc/persistent_topk.cuh` added +1321/-0 (1321 lines); hunks: -0,0 +1,1321; `tests/kernels/test_top_k_per_row.py` modified +540/-78 (618 lines); hunks: -122,6 +122,39 @@ def compare_top_k_results(; -278,111 +311,540 @@ def test_top_k_per_row_decode_large_vocab_size(clean_log...; symbols: compare_top_k_results, validate_topk_against_reference, test_top_k_per_row_decode_large_vocab_size, test_deepseek_hybrid_topk, touching `compare_top_k_results, validate_topk_against_reference, test_top_k_per_row_decode_large_vocab_size`.
- Code diff details:
  - `vllm/model_executor/layers/sparse_attn_indexer.py` modified +24/-24 (48 lines); hunks: -25,6 +25,8; -51,6 +53,7 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer
  - `vllm/model_executor/models/deepseek_v2.py` modified +6/-2 (8 lines); hunks: -67,7 +67,9; -1203,7 +1205,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__
  - `csrc/persistent_topk.cuh` added +1321/-0 (1321 lines); hunks: -0,0 +1,1321
  - `tests/kernels/test_top_k_per_row.py` modified +540/-78 (618 lines); hunks: -122,6 +122,39 @@ def compare_top_k_results(; -278,111 +311,540 @@ def test_top_k_per_row_decode_large_vocab_size(clean_log...; symbols: compare_top_k_results, validate_topk_against_reference, test_top_k_per_row_decode_large_vocab_size, test_deepseek_hybrid_topk
  - `csrc/topk.cu` modified +139/-358 (497 lines); hunks: -1,373 +1,154
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/sparse_attn_indexer.py
@@ -25,6 +25,8 @@
+RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024
@@ -51,6 +53,7 @@ def sparse_attn_indexer(
+            ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
@@ -157,15 +160,6 @@ def sparse_attn_indexer(
-            # Compute lengths from row spans
-            # lengths = (chunk.cu_seqlen_ke - chunk.cu_seqlen_ks).to(torch.int32)
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -67,7 +67,9 @@
-from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
+from vllm.model_executor.layers.sparse_attn_indexer import (
+    SparseAttnIndexer,
+)
@@ -1203,7 +1205,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
-                vllm_config, prefix, topk_indices_buffer=topk_indices_buffer
diff -- csrc/persistent_topk.cuh
@@ -0,0 +1,1321 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/sparse_attn_indexer.py` modified +24/-24; `vllm/model_executor/models/deepseek_v2.py` modified +6/-2; `vllm/v1/attention/backends/mla/indexer.py` modified +0/-12
  - other: `csrc/persistent_topk.cuh` added +1321/-0; `csrc/topk.cu` modified +139/-358; `csrc/torch_bindings.cpp` modified +3/-4; `csrc/ops.h` modified +3/-3
  - tests: `tests/kernels/test_top_k_per_row.py` modified +540/-78
- Risk and verification: The diff ships test coverage in `tests/kernels/test_top_k_per_row.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #38928 - [Bugfix][Perf] Indexer upcast WK to BF16 for fusion

- Link: https://github.com/vllm-project/vllm/pull/38928
- Status/date: merged / 2026-04-15
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +84/-64, 239 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix][Perf] Indexer upcast WK to BF16 for fusion"; model line: DeepSeek V3.1; category: bug fix; main diff: `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`; technical summary: Covers "[Bugfix][Perf] Indexer upcast WK to BF16 for fusion"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +70/-53 (123 lines); hunks: -66,6 +66,10; -628,10 +632,6 @@ def __init__(; symbols: __init__, forward, _try_load_fp8_indexer_wk, touching `__init__, forward, _try_load_fp8_indexer_wk`; `vllm/model_executor/models/deepseek_mtp.py` modified +14/-11 (25 lines); hunks: -30,6 +30,7; -190,10 +191,6 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, set_moe_parameters, load_weights, touching `__init__, set_moe_parameters, load_weights`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +70/-53 (123 lines); hunks: -66,6 +66,10; -628,10 +632,6 @@ def __init__(; symbols: __init__, forward, _try_load_fp8_indexer_wk
  - `vllm/model_executor/models/deepseek_mtp.py` modified +14/-11 (25 lines); hunks: -30,6 +30,7; -190,10 +191,6 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, set_moe_parameters, load_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -66,6 +66,10 @@
+from vllm.model_executor.layers.quantization.utils.quant_utils import (
+    GroupShape,
+    scaled_dequantize,
+)
@@ -628,10 +632,6 @@ def __init__(
-        self.is_fp4_ckpt = (
diff -- vllm/model_executor/models/deepseek_mtp.py
@@ -30,6 +30,7 @@
+    _try_load_fp8_indexer_wk,
@@ -190,10 +191,6 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
-        self.is_fp4_ckpt = (
-            self.quant_config is not None
-            and self.quant_config.get_name() == "modelopt_fp4"
-        )
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +70/-53; `vllm/model_executor/models/deepseek_mtp.py` modified +14/-11
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_mtp.py`, `vllm/model_executor/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #35949 - [MoE Refactor] Move the shared/fused expert output sum into MoERunnerBase

- Link: https://github.com/vllm-project/vllm/pull/35949
- Status/date: merged / 2026-04-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 53 files, +325/-702, 2430 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MoE Refactor] Move the shared/fused expert output sum into MoERunnerBase"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/exaone_moe.py`; technical summary: Covers "[MoE Refactor] Move the shared/fused expert output sum into MoERunnerBase"; the main implementation surface is `vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/exaone_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py` modified +175/-86 (261 lines); hunks: -81,15 +81,17 @@ def _resolve_layer_name(layer_name: str | LayerName) -> str:; -113,7 +115,7 @@ def _moe_forward_shared(; symbols: _resolve_layer_name, _moe_forward, _moe_forward_shared, _moe_forward_shared_fake, touching `_resolve_layer_name, _moe_forward, _moe_forward_shared`; `vllm/model_executor/layers/fused_moe/layer.py` modified +28/-32 (60 lines); hunks: -230,11 +230,18 @@ class FusedMoE(PluggableLayer):; -246,7 +253,6 @@ def __init__(; symbols: FusedMoE, __init__, touching `FusedMoE, __init__`; `vllm/model_executor/models/exaone_moe.py` modified +18/-28 (46 lines); hunks: -31,6 +31,7; -116,12 +117,26 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`; `vllm/model_executor/models/kimi_linear.py` modified +20/-26 (46 lines); hunks: -11,11 +11,10; -132,12 +131,25 @@ def __init__(; symbols: __init__, forward, load_weights, touching `__init__, forward, load_weights`.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py` modified +175/-86 (261 lines); hunks: -81,15 +81,17 @@ def _resolve_layer_name(layer_name: str | LayerName) -> str:; -113,7 +115,7 @@ def _moe_forward_shared(; symbols: _resolve_layer_name, _moe_forward, _moe_forward_shared, _moe_forward_shared_fake
  - `vllm/model_executor/layers/fused_moe/layer.py` modified +28/-32 (60 lines); hunks: -230,11 +230,18 @@ class FusedMoE(PluggableLayer):; -246,7 +253,6 @@ def __init__(; symbols: FusedMoE, __init__
  - `vllm/model_executor/models/exaone_moe.py` modified +18/-28 (46 lines); hunks: -31,6 +31,7; -116,12 +117,26 @@ def __init__(; symbols: __init__, forward
  - `vllm/model_executor/models/kimi_linear.py` modified +20/-26 (46 lines); hunks: -11,11 +11,10; -132,12 +131,25 @@ def __init__(; symbols: __init__, forward, load_weights
  - `vllm/model_executor/models/AXK1.py` modified +5/-30 (35 lines); hunks: -100,7 +100,7 @@ def __init__(; -170,7 +170,6 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py
@@ -81,15 +81,17 @@ def _resolve_layer_name(layer_name: str | LayerName) -> str:
-# the runner's 'forward_dispatch' method.
+# the runner's '_forward_dispatch' method.
+# These functions should never be called directly since they do not
+# include all the functionality of the MoE layer.
-    return layer.runner.forward_dispatch(
+    return layer.runner._forward_dispatch(
diff -- vllm/model_executor/layers/fused_moe/layer.py
@@ -230,11 +230,18 @@ class FusedMoE(PluggableLayer):
-        reduce_results: Whether to all_reduce on the output of the layer
+        routed_scaling_factor: A scaling factor that is applied to the topk_weights
+                               by the router or the output of the layer depending
+                               on the value of `apply_routed_scale_to_output`
+        apply_routed_scale_to_output: Determine whether or not `routed_scaling_factor`
+                                      is applied to the topk_weights or to the experts
diff -- vllm/model_executor/models/exaone_moe.py
@@ -31,6 +31,7 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py` modified +175/-86; `vllm/model_executor/layers/fused_moe/layer.py` modified +28/-32; `vllm/model_executor/models/exaone_moe.py` modified +18/-28; `vllm/model_executor/models/kimi_linear.py` modified +20/-26; `vllm/model_executor/models/AXK1.py` modified +5/-30; `vllm/model_executor/models/ernie45_vl_moe.py` modified +5/-30
- Risk and verification: The diff ships test coverage in `tests/compile/passes/test_vllm_fusion_pattern_matcher_pass.py`, `tests/kernels/moe/test_moe_layer.py`, `tests/kernels/moe/test_shared_fused_moe_routed_transform.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #35782 - [MoE Refactor] Remove SharedFusedMoE class

- Link: https://github.com/vllm-project/vllm/pull/35782
- Status/date: merged / 2026-04-21
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 33 files, +112/-141, 926 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MoE Refactor] Remove SharedFusedMoE class"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `vllm/model_executor/layers/fused_moe/shared_fused_moe.py`, `vllm/model_executor/models/afmoe.py`, `vllm/model_executor/models/llama4.py`; technical summary: Covers "[MoE Refactor] Remove SharedFusedMoE class"; the main implementation surface is `vllm/model_executor/layers/fused_moe/shared_fused_moe.py`, `vllm/model_executor/models/afmoe.py`, `vllm/model_executor/models/llama4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/fused_moe/shared_fused_moe.py` removed +0/-25 (25 lines); hunks: -1,25 +0,0; symbols: SharedFusedMoE, forward, touching `SharedFusedMoE, forward`; `vllm/model_executor/models/afmoe.py` modified +5/-5 (10 lines); hunks: -18,7 +18,7; -124,8 +124,8 @@ def __init__(; symbols: __init__, make_empty_intermediate_tensors, get_expert_mapping, touching `__init__, make_empty_intermediate_tensors, get_expert_mapping`; `vllm/model_executor/models/llama4.py` modified +5/-5 (10 lines); hunks: -36,7 +36,7; -127,7 +127,7 @@ def __init__(self, vllm_config: VllmConfig, prefix: str = ""):; symbols: __init__, load_moe_expert_weights, load_weights, touching `__init__, load_moe_expert_weights, load_weights`; `vllm/model_executor/models/AXK1.py` modified +4/-4 (8 lines); hunks: -42,7 +42,7; -163,7 +163,7 @@ def __init__(; symbols: __init__, compute_logits, get_expert_mapping, load_weights, touching `__init__, compute_logits, get_expert_mapping`.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/shared_fused_moe.py` removed +0/-25 (25 lines); hunks: -1,25 +0,0; symbols: SharedFusedMoE, forward
  - `vllm/model_executor/models/afmoe.py` modified +5/-5 (10 lines); hunks: -18,7 +18,7; -124,8 +124,8 @@ def __init__(; symbols: __init__, make_empty_intermediate_tensors, get_expert_mapping
  - `vllm/model_executor/models/llama4.py` modified +5/-5 (10 lines); hunks: -36,7 +36,7; -127,7 +127,7 @@ def __init__(self, vllm_config: VllmConfig, prefix: str = ""):; symbols: __init__, load_moe_expert_weights, load_weights
  - `vllm/model_executor/models/AXK1.py` modified +4/-4 (8 lines); hunks: -42,7 +42,7; -163,7 +163,7 @@ def __init__(; symbols: __init__, compute_logits, get_expert_mapping, load_weights
  - `vllm/model_executor/models/deepseek_v2.py` modified +4/-4 (8 lines); hunks: -48,9 +48,9; -311,7 +311,7 @@ def __init__(; symbols: __init__, compute_logits, get_expert_mapping, load_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/fused_moe/shared_fused_moe.py
@@ -1,25 +0,0 @@
-# SPDX-License-Identifier: Apache-2.0
-# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-import torch
-from vllm.model_executor.layers.fused_moe.layer import FusedMoE
-# TODO(bnell): Remove this entirely
-class SharedFusedMoE(FusedMoE):
diff -- vllm/model_executor/models/afmoe.py
@@ -18,7 +18,7 @@
-from vllm.model_executor.layers.fused_moe.shared_fused_moe import SharedFusedMoE
+from vllm.model_executor.layers.fused_moe import FusedMoE
@@ -124,8 +124,8 @@ def __init__(
-        # Routed experts using SharedFusedMoE
-        self.experts = SharedFusedMoE(
+        # Routed experts using FusedMoE
diff -- vllm/model_executor/models/llama4.py
@@ -36,7 +36,7 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/shared_fused_moe.py` removed +0/-25; `vllm/model_executor/models/afmoe.py` modified +5/-5; `vllm/model_executor/models/llama4.py` modified +5/-5; `vllm/model_executor/models/AXK1.py` modified +4/-4; `vllm/model_executor/models/deepseek_v2.py` modified +4/-4; `vllm/model_executor/models/ernie45_moe.py` modified +4/-4
- Risk and verification: The diff ships test coverage in `tests/kernels/moe/test_moe_layer.py`, `tests/kernels/moe/test_shared_fused_moe_routed_transform.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #40671 - [MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping

- Link: https://github.com/vllm-project/vllm/pull/40671
- Status/date: merged / 2026-04-23
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 53 files, +254/-98, 1073 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/llama4.py`, `vllm/model_executor/models/glm4_moe_lite.py`; technical summary: Covers "[MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping"; the main implementation surface is `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/llama4.py`, `vllm/model_executor/models/glm4_moe_lite.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/fused_moe/layer.py` modified +19/-0 (19 lines); hunks: -1618,6 +1618,25 @@ def extra_repr(self) -> str:; symbols: extra_repr, fused_moe_make_expert_params_mapping, touching `extra_repr, fused_moe_make_expert_params_mapping`; `vllm/model_executor/models/llama4.py` modified +7/-4 (11 lines); hunks: -36,7 +36,10; -414,7 +417,7 @@ def load_moe_expert_weights(; symbols: load_moe_expert_weights, load_weights, touching `load_moe_expert_weights, load_weights`; `vllm/model_executor/models/glm4_moe_lite.py` modified +6/-4 (10 lines); hunks: -41,7 +41,9; -308,7 +310,7 @@ def make_empty_intermediate_tensors(; symbols: make_empty_intermediate_tensors, get_expert_mapping, load_weights, compute_logits, touching `make_empty_intermediate_tensors, get_expert_mapping, load_weights`; `vllm/model_executor/models/AXK1.py` modified +6/-3 (9 lines); hunks: -42,7 +42,10; -916,7 +919,7 @@ def compute_logits(; symbols: compute_logits, get_expert_mapping, load_weights, touching `compute_logits, get_expert_mapping, load_weights`.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/layer.py` modified +19/-0 (19 lines); hunks: -1618,6 +1618,25 @@ def extra_repr(self) -> str:; symbols: extra_repr, fused_moe_make_expert_params_mapping
  - `vllm/model_executor/models/llama4.py` modified +7/-4 (11 lines); hunks: -36,7 +36,10; -414,7 +417,7 @@ def load_moe_expert_weights(; symbols: load_moe_expert_weights, load_weights
  - `vllm/model_executor/models/glm4_moe_lite.py` modified +6/-4 (10 lines); hunks: -41,7 +41,9; -308,7 +310,7 @@ def make_empty_intermediate_tensors(; symbols: make_empty_intermediate_tensors, get_expert_mapping, load_weights, compute_logits
  - `vllm/model_executor/models/AXK1.py` modified +6/-3 (9 lines); hunks: -42,7 +42,10; -916,7 +919,7 @@ def compute_logits(; symbols: compute_logits, get_expert_mapping, load_weights
  - `vllm/model_executor/models/afmoe.py` modified +5/-2 (7 lines); hunks: -18,7 +18,10; -479,7 +482,7 @@ def make_empty_intermediate_tensors(; symbols: make_empty_intermediate_tensors, get_expert_mapping
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/fused_moe/layer.py
@@ -1618,6 +1618,25 @@ def extra_repr(self) -> str:
+# This is a temporary forwarding method which will be removed/modified layer.
+def fused_moe_make_expert_params_mapping(
+    model: torch.nn.Module,
+    ckpt_gate_proj_name: str,
+    ckpt_down_proj_name: str,
+    ckpt_up_proj_name: str,
diff -- vllm/model_executor/models/llama4.py
@@ -36,7 +36,10 @@
-from vllm.model_executor.layers.fused_moe import FusedMoE
+from vllm.model_executor.layers.fused_moe import (
+    FusedMoE,
+    fused_moe_make_expert_params_mapping,
+)
@@ -414,7 +417,7 @@ def load_moe_expert_weights(
diff -- vllm/model_executor/models/glm4_moe_lite.py
@@ -41,7 +41,9 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/layer.py` modified +19/-0; `vllm/model_executor/models/llama4.py` modified +7/-4; `vllm/model_executor/models/glm4_moe_lite.py` modified +6/-4; `vllm/model_executor/models/AXK1.py` modified +6/-3; `vllm/model_executor/models/afmoe.py` modified +5/-2; `vllm/model_executor/models/bailing_moe.py` modified +5/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/__init__.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/AXK1.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #39999 - [ROCm] Cast score correction bias tensor during model construction for DeepSeek/Kimi-K2

- Link: https://github.com/vllm-project/vllm/pull/39999
- Status/date: merged / 2026-04-24
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +19/-2, 49 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm] Cast score correction bias tensor during model construction for DeepSeek/Kimi-K2"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py`, `vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py`; technical summary: Covers "[ROCm] Cast score correction bias tensor during model construction for DeepSeek/Kimi-K2"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py`, `vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +15/-0 (15 lines); hunks: -348,6 +348,21 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`; `vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py` modified +1/-1 (2 lines); hunks: -152,7 +152,7 @@ def rocm_aiter_grouped_topk(; symbols: rocm_aiter_grouped_topk, touching `rocm_aiter_grouped_topk`; `vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py` modified +1/-1 (2 lines); hunks: -136,7 +136,7 @@ def fused_topk_bias(; symbols: fused_topk_bias, touching `fused_topk_bias`; `vllm/_aiter_ops.py` modified +2/-0 (2 lines); hunks: -1782,6 +1782,8 @@ def biased_grouped_topk(; symbols: biased_grouped_topk, touching `biased_grouped_topk`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +15/-0 (15 lines); hunks: -348,6 +348,21 @@ def __init__(; symbols: __init__, forward
  - `vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py` modified +1/-1 (2 lines); hunks: -152,7 +152,7 @@ def rocm_aiter_grouped_topk(; symbols: rocm_aiter_grouped_topk
  - `vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py` modified +1/-1 (2 lines); hunks: -136,7 +136,7 @@ def fused_topk_bias(; symbols: fused_topk_bias
  - `vllm/_aiter_ops.py` modified +2/-0 (2 lines); hunks: -1782,6 +1782,8 @@ def biased_grouped_topk(; symbols: biased_grouped_topk
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -348,6 +348,21 @@ def __init__(
+        # Pre-cast the bias to match the gate output dtype so the
+        # conversion is not repeated on every forward pass.  All
+        # downstream references (FusedMoE, router) share the same
+        # nn.Parameter object, so mutating .data propagates everywhere.
+        # Weight loading uses copy_(), which handles the dtype conversion.
+        # Only needed on ROCm where the aiter biased_grouped_topk kernel
diff -- vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py
@@ -152,7 +152,7 @@ def rocm_aiter_grouped_topk(
-            e_score_correction_bias.to(gating_output.dtype),
+            e_score_correction_bias,
diff -- vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py
@@ -136,7 +136,7 @@ def fused_topk_bias(
-                e_score_correction_bias.to(gating_output.dtype),
+                e_score_correction_bias,
diff -- vllm/_aiter_ops.py
@@ -1782,6 +1782,8 @@ def biased_grouped_topk(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +15/-0; `vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py` modified +1/-1; `vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py` modified +1/-1; `vllm/_aiter_ops.py` modified +2/-0
- Risk and verification: Runtime changes concentrate in `vllm/_aiter_ops.py`, `vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py`, `vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #39141 - [Perf] Update TRTLLM supported MoE routing methods

- Link: https://github.com/vllm-project/vllm/pull/39141
- Status/date: merged / 2026-04-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +26/-92, 258 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Perf] Update TRTLLM supported MoE routing methods"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py`, `vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py`, `vllm/model_executor/layers/fused_moe/config.py`; technical summary: Covers "[Perf] Update TRTLLM supported MoE routing methods"; the main implementation surface is `vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py`, `vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py`, `vllm/model_executor/layers/fused_moe/config.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py` modified +9/-43 (52 lines); hunks: -175,13 +175,6 @@ def apply(; -196,11 +189,7 @@ def apply(; symbols: apply, TrtLlmFp8ExpertsMonolithic, _supports_router_logits_dtype, touching `apply, TrtLlmFp8ExpertsMonolithic, _supports_router_logits_dtype`; `vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py` modified +3/-29 (32 lines); hunks: -198,13 +198,6 @@ def apply(; -233,7 +226,7 @@ def apply(; symbols: apply, _supports_routing_method, _supports_router_logits_dtype, touching `apply, _supports_routing_method, _supports_router_logits_dtype`; `vllm/model_executor/layers/fused_moe/config.py` modified +14/-7 (21 lines); hunks: -113,14 +113,17 @@ class RoutingMethodType(IntEnum):; -141,12 +144,16 @@ def get_routing_method_type(; symbols: RoutingMethodType, get_routing_method_type, touching `RoutingMethodType, get_routing_method_type`; `vllm/model_executor/models/deepseek_v2.py` modified +0/-12 (12 lines); hunks: -50,7 +50,6; -338,17 +337,6 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py` modified +9/-43 (52 lines); hunks: -175,13 +175,6 @@ def apply(; -196,11 +189,7 @@ def apply(; symbols: apply, TrtLlmFp8ExpertsMonolithic, _supports_router_logits_dtype
  - `vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py` modified +3/-29 (32 lines); hunks: -198,13 +198,6 @@ def apply(; -233,7 +226,7 @@ def apply(; symbols: apply, _supports_routing_method, _supports_router_logits_dtype
  - `vllm/model_executor/layers/fused_moe/config.py` modified +14/-7 (21 lines); hunks: -113,14 +113,17 @@ class RoutingMethodType(IntEnum):; -141,12 +144,16 @@ def get_routing_method_type(; symbols: RoutingMethodType, get_routing_method_type
  - `vllm/model_executor/models/deepseek_v2.py` modified +0/-12 (12 lines); hunks: -50,7 +50,6; -338,17 +337,6 @@ def __init__(; symbols: __init__
  - `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py` modified +0/-1 (1 lines); hunks: -275,7 +275,6 @@ def _return_or_raise(; symbols: _return_or_raise
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py
@@ -175,13 +175,6 @@ def apply(
-        # trtllm_fp8_block_scale_routed_moe does not support autotuning
-        # so skip this kernel during dummy run for autotuning.
-        import vllm.utils.flashinfer as fi_utils
-        if fi_utils._is_fi_autotuning:
-            return
@@ -196,11 +189,7 @@ def apply(
diff -- vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py
@@ -198,13 +198,6 @@ def apply(
-        # trtllm_fp4_block_scale_routed_moe does not support autotuning
-        # so skip this kernel during dummy run for autotuning.
-        import vllm.utils.flashinfer as fi_utils
-        if fi_utils._is_fi_autotuning:
-            return
@@ -233,7 +226,7 @@ def apply(
diff -- vllm/model_executor/layers/fused_moe/config.py
@@ -113,14 +113,17 @@ class RoutingMethodType(IntEnum):
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py` modified +9/-43; `vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py` modified +3/-29; `vllm/model_executor/layers/fused_moe/config.py` modified +14/-7; `vllm/model_executor/models/deepseek_v2.py` modified +0/-12; `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py` modified +0/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/config.py`, `vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py`, `vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #37735 - [Feature]: IndexCache support for DSA models

- Link: https://github.com/vllm-project/vllm/pull/37735
- Status/date: merged / 2026-04-29
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +83/-5, 138 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feature]: IndexCache support for DSA models"; model line: DeepSeek V3.1; category: docs/tests/CI; main diff: `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/layers/mla.py`, `docs/features/index_cache.md`; technical summary: Covers "[Feature]: IndexCache support for DSA models"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/layers/mla.py`, `docs/features/index_cache.md`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +21/-1 (22 lines); hunks: -82,7 +82,10; -963,6 +966,7 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`; `vllm/model_executor/layers/mla.py` modified +8/-4 (12 lines); hunks: -64,6 +64,7 @@ def __init__(; -87,6 +88,11 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`; `docs/features/index_cache.md` added +54/-0 (54 lines); hunks: -0,0 +1,54.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +21/-1 (22 lines); hunks: -82,7 +82,10; -963,6 +966,7 @@ def __init__(; symbols: __init__, forward
  - `vllm/model_executor/layers/mla.py` modified +8/-4 (12 lines); hunks: -64,6 +64,7 @@ def __init__(; -87,6 +88,11 @@ def __init__(; symbols: __init__, forward
  - `docs/features/index_cache.md` added +54/-0 (54 lines); hunks: -0,0 +1,54
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -82,7 +82,10 @@
-from vllm.model_executor.models.utils import sequence_parallel_chunk
+from vllm.model_executor.models.utils import (
+    extract_layer_index,
+    sequence_parallel_chunk,
+)
@@ -963,6 +966,7 @@ def __init__(
diff -- vllm/model_executor/layers/mla.py
@@ -64,6 +64,7 @@ def __init__(
+        skip_topk: bool = False,
@@ -87,6 +88,11 @@ def __init__(
+        # Whether to skip top-k token selection computation in this layer.
+        # When True, the indexer will not be called, and the layer will reuse
+        # the topk_tokens buffer written by a previous layer in the same pass.
+        # Refer: https://arxiv.org/abs/2603.12201 for more details.
diff -- docs/features/index_cache.md
@@ -0,0 +1,54 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +21/-1; `vllm/model_executor/layers/mla.py` modified +8/-4
  - docs: `docs/features/index_cache.md` added +54/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/mla.py`, `vllm/model_executor/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #41217 - [ROCm][Deepseek] dsv3.2 further optimization

- Link: https://github.com/vllm-project/vllm/pull/41217
- Status/date: merged / 2026-05-01
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +293/-73, 605 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm][Deepseek] dsv3.2 further optimization"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `vllm/model_executor/models/deepseek_v2.py`, `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py`, `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`; technical summary: Covers "[ROCm][Deepseek] dsv3.2 further optimization"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`, `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py`, `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +38/-23 (61 lines); hunks: -674,30 +674,45 @@ def forward(; symbols: forward, touching `forward`; `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py` modified +227/-29 (256 lines); hunks: -7,13 +7,15; -25,9 +27,6; symbols: _convert_req_index_to_global_index_kernel, triton_convert_req_index_to_global_index, generate_sparse_seqlen_kernel, generate_sparse_seqlen_triton, touching `_convert_req_index_to_global_index_kernel, triton_convert_req_index_to_global_index, generate_sparse_seqlen_kernel`; `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` modified +22/-19 (41 lines); hunks: -13,9 +13,6; -97,7 +94,8 @@ def indexer_k_quant_and_cache_triton(; symbols: _indexer_k_quant_and_cache_kernel, indexer_k_quant_and_cache_triton, cp_gather_indexer_k_quant_cache_triton, rocm_fp8_paged_mqa_logits, touching `_indexer_k_quant_and_cache_kernel, indexer_k_quant_and_cache_triton, cp_gather_indexer_k_quant_cache_triton`; `vllm/v1/attention/backends/mla/rocm_aiter_mla.py` modified +4/-0 (4 lines); hunks: -396,6 +396,7 @@ class AiterMLAHelper:; -419,6 +420,9 @@ def get_actual_mla_num_heads(num_heads: int) -> int:; symbols: AiterMLAHelper, check_num_heads_validity, get_actual_mla_num_heads, get_mla_padded_q, touching `AiterMLAHelper, check_num_heads_validity, get_actual_mla_num_heads`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +38/-23 (61 lines); hunks: -674,30 +674,45 @@ def forward(; symbols: forward
  - `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py` modified +227/-29 (256 lines); hunks: -7,13 +7,15; -25,9 +27,6; symbols: _convert_req_index_to_global_index_kernel, triton_convert_req_index_to_global_index, generate_sparse_seqlen_kernel, generate_sparse_seqlen_triton
  - `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` modified +22/-19 (41 lines); hunks: -13,9 +13,6; -97,7 +94,8 @@ def indexer_k_quant_and_cache_triton(; symbols: _indexer_k_quant_and_cache_kernel, indexer_k_quant_and_cache_triton, cp_gather_indexer_k_quant_cache_triton, rocm_fp8_paged_mqa_logits
  - `vllm/v1/attention/backends/mla/rocm_aiter_mla.py` modified +4/-0 (4 lines); hunks: -396,6 +396,7 @@ class AiterMLAHelper:; -419,6 +420,9 @@ def get_actual_mla_num_heads(num_heads: int) -> int:; symbols: AiterMLAHelper, check_num_heads_validity, get_actual_mla_num_heads, get_mla_padded_q
  - `docs/design/attention_backends.md` modified +1/-1 (2 lines); hunks: -216,7 +216,7 @@ configuration.
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -674,30 +674,45 @@ def forward(
-        q_pe, q_nope = torch.split(
-            q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
-        )
-        # Fused wk + weights_proj: one GEMM, then split
-        kw, _ = self.wk_weights_proj(hidden_states)
-        k = kw[:, : self.head_dim]
diff -- vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py
@@ -7,13 +7,15 @@
+from vllm import _custom_ops as ops
+from vllm.platforms import current_platform
@@ -25,9 +27,6 @@
-from vllm.v1.attention.backends.mla.flashmla_sparse import (
-    triton_convert_req_index_to_global_index,
-)
diff -- vllm/v1/attention/ops/rocm_aiter_mla_sparse.py
@@ -13,9 +13,6 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +38/-23; `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py` modified +227/-29; `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` modified +22/-19; `vllm/v1/attention/backends/mla/rocm_aiter_mla.py` modified +4/-0; `vllm/v1/attention/backends/mla/indexer.py` modified +1/-1
  - docs: `docs/design/attention_backends.md` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v2.py`, `vllm/v1/attention/backends/mla/indexer.py`, `vllm/v1/attention/backends/mla/rocm_aiter_mla.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #41405 - [ROCm][Bugfix] Fix init-time bias dtype cast when gate.out_dtype is None

- Link: https://github.com/vllm-project/vllm/pull/41405
- Status/date: merged / 2026-05-02
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-1, 11 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm][Bugfix] Fix init-time bias dtype cast when gate.out_dtype is None"; model line: DeepSeek V3.1; category: bug fix; main diff: `vllm/model_executor/models/deepseek_v2.py`; technical summary: Covers "[ROCm][Bugfix] Fix init-time bias dtype cast when gate.out_dtype is None"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +2/-1 (3 lines); hunks: -351,8 +351,9 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +2/-1 (3 lines); hunks: -351,8 +351,9 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -351,8 +351,9 @@ def __init__(
+            gate_out_dtype = self.gate.out_dtype or self.gate.weight.dtype
-                self.gate.e_score_correction_bias.data.to(self.gate.out_dtype)
+                self.gate.e_score_correction_bias.data.to(gate_out_dtype)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +2/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #40759 - [Examples] Resettle Disaggregated examples.

- Link: https://github.com/vllm-project/vllm/pull/40759
- Status/date: merged / 2026-05-06
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 54 files, +29/-31, 215 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Examples] Resettle Disaggregated examples."; model line: DeepSeek V3.1; category: docs/tests/CI; main diff: `examples/disaggregated/lmcache/disagg_prefill_lmcache_v1/configs/lmcache-decoder-config.yaml`, `examples/disaggregated/lmcache/disagg_prefill_lmcache_v1/configs/lmcache-prefiller-config.yaml`, `docs/features/disagg_prefill.md`; technical summary: Covers "[Examples] Resettle Disaggregated examples."; the main implementation surface is `examples/disaggregated/lmcache/disagg_prefill_lmcache_v1/configs/lmcache-decoder-config.yaml`, `examples/disaggregated/lmcache/disagg_prefill_lmcache_v1/configs/lmcache-prefiller-config.yaml`, `docs/features/disagg_prefill.md`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `examples/disaggregated/lmcache/disagg_prefill_lmcache_v1/configs/lmcache-decoder-config.yaml` renamed +0/-0 (0 lines); `examples/disaggregated/lmcache/disagg_prefill_lmcache_v1/configs/lmcache-prefiller-config.yaml` renamed +0/-0 (0 lines); `docs/features/disagg_prefill.md` modified +6/-6 (12 lines); hunks: -17,15 +17,15 @@ Two main reasons:; -44,7 +44,7 @@ For NixlConnector, you may also specify one or multiple NIXL_B...; `docs/features/mooncake_connector_usage.md` modified +3/-3 (6 lines); hunks: -31,7 +31,7 @@ vllm serve Qwen/Qwen2.5-7B-Instruct --port 8020 --kv-transfer-...; -65,5 +65,5 @@ Now you can send requests to the proxy server through port 8000..
- Code diff details:
  - `examples/disaggregated/lmcache/disagg_prefill_lmcache_v1/configs/lmcache-decoder-config.yaml` renamed +0/-0 (0 lines)
  - `examples/disaggregated/lmcache/disagg_prefill_lmcache_v1/configs/lmcache-prefiller-config.yaml` renamed +0/-0 (0 lines)
  - `docs/features/disagg_prefill.md` modified +6/-6 (12 lines); hunks: -17,15 +17,15 @@ Two main reasons:; -44,7 +44,7 @@ For NixlConnector, you may also specify one or multiple NIXL_B...
  - `docs/features/mooncake_connector_usage.md` modified +3/-3 (6 lines); hunks: -31,7 +31,7 @@ vllm serve Qwen/Qwen2.5-7B-Instruct --port 8020 --kv-transfer-...; -65,5 +65,5 @@ Now you can send requests to the proxy server through port 8000.
  - `.github/mergify.yml` modified +1/-3 (4 lines); hunks: -477,9 +477,7 @@ pull_request_rules:
- Key code excerpts:

```diff
diff -- docs/features/disagg_prefill.md
@@ -17,15 +17,15 @@ Two main reasons:
-Please refer to [examples/online_serving/disaggregated_prefill.sh](../../examples/online_serving/disaggregated_prefill.sh) for the example usage of disaggregated prefilling.
+Please refer to [examples/disaggregated/disaggregated_prefill.sh](../../examples/disaggregated/disaggregated_prefill.sh) for the example usage of disaggregated prefilling.
-- **ExampleConnector**: refer to [examples/offline_inference/disaggregated-prefill-v1/run.sh](../../examples/offline_inference/disaggregated-prefill-v1/run.sh) for the example usa
-- **LMCacheConnectorV1**: refer to [examples/others/lmcache/disagg_prefill_lmcache_v1/disagg_example_nixl.sh](../../examples/others/lmcache/disagg_prefill_lmcache_v1/disagg_exampl
+- **ExampleConnector**: refer to [examples/disaggregated/example_connector/run.sh](../../examples/disaggregated/example_connector/run.sh) for the example usage of ExampleConnector
+- **LMCacheConnectorV1**: refer to [examples/disaggregated/lmcache/disagg_prefill_lmcache_v1/disagg_example_nixl.sh](../../examples/disaggregated/lmcache/disagg_prefill_lmcache_v1
diff -- docs/features/mooncake_connector_usage.md
@@ -31,7 +31,7 @@ vllm serve Qwen/Qwen2.5-7B-Instruct --port 8020 --kv-transfer-config '{"kv_conne
-python examples/online_serving/disaggregated_serving/mooncake_connector/mooncake_connector_proxy.py --prefill http://192.168.0.2:8010 --decode http://192.168.0.3:8020
+python examples/disaggregated/disaggregated_serving/mooncake_connector/mooncake_connector_proxy.py --prefill http://192.168.0.2:8010 --decode http://192.168.0.3:8020
@@ -65,5 +65,5 @@ Now you can send requests to the proxy server through port 8000.
-- [run_mooncake_connector.sh](../../examples/online_serving/disaggregated_serving/mooncake_connector/run_mooncake_connector.sh)
-- [mooncake_connector_proxy.py](../../examples/online_serving/disaggregated_serving/mooncake_connector/mooncake_connector_proxy.py)
+- [run_mooncake_connector.sh](../../examples/disaggregated/mooncake_connector/run_mooncake_connector.sh)
diff -- .github/mergify.yml
@@ -477,9 +477,7 @@ pull_request_rules:
```

- Reviewed files:
  - docs: `examples/disaggregated/lmcache/disagg_prefill_lmcache_v1/configs/lmcache-decoder-config.yaml` renamed +0/-0; `examples/disaggregated/lmcache/disagg_prefill_lmcache_v1/configs/lmcache-prefiller-config.yaml` renamed +0/-0; `docs/features/disagg_prefill.md` modified +6/-6; `docs/features/mooncake_connector_usage.md` modified +3/-3; `docs/design/p2p_nccl_connector.md` modified +2/-2; `docs/features/disagg_encoder.md` modified +2/-2
  - ci: `.github/mergify.yml` modified +1/-3
- Risk and verification: The diff ships test coverage in `tests/v1/ec_connector/integration/README.md`, `tests/v1/ec_connector/integration/run_epd_correctness_test.sh`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #41835 - [ROCm][DeepSeek] Enable V3.2 TP4 AITER MLA

- Link: https://github.com/vllm-project/vllm/pull/41835
- Status/date: merged / 2026-05-07
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +12/-10, 50 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm][DeepSeek] Enable V3.2 TP4 AITER MLA"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `vllm/model_executor/models/deepseek_v2.py`, `vllm/v1/attention/backends/mla/rocm_aiter_mla.py`; technical summary: Covers "[ROCm][DeepSeek] Enable V3.2 TP4 AITER MLA"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`, `vllm/v1/attention/backends/mla/rocm_aiter_mla.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +11/-9 (20 lines); hunks: -299,6 +299,15 @@ def __init__(; -338,22 +347,15 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`; `vllm/v1/attention/backends/mla/rocm_aiter_mla.py` modified +1/-1 (2 lines); hunks: -396,7 +396,7 @@ class AiterMLAHelper:; symbols: AiterMLAHelper, check_num_heads_validity, touching `AiterMLAHelper, check_num_heads_validity`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +11/-9 (20 lines); hunks: -299,6 +299,15 @@ def __init__(; -338,22 +347,15 @@ def __init__(; symbols: __init__, forward
  - `vllm/v1/attention/backends/mla/rocm_aiter_mla.py` modified +1/-1 (2 lines); hunks: -396,7 +396,7 @@ class AiterMLAHelper:; symbols: AiterMLAHelper, check_num_heads_validity
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -299,6 +299,15 @@ def __init__(
+        if (
+            self.is_rocm_aiter_moe_enabled
+            and self.gate.e_score_correction_bias is not None
+        ):
+            # AITER biased_grouped_topk requires the correction bias dtype to
+            # match the router logits. Keep DeepSeek's correction bias in fp32
diff -- vllm/v1/attention/backends/mla/rocm_aiter_mla.py
@@ -396,7 +396,7 @@ class AiterMLAHelper:
-    _AITER_UNSUPPORTED_HEADS = [32]
+    _AITER_UNSUPPORTED_HEADS: ClassVar[tuple[int, ...]] = ()
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +11/-9; `vllm/v1/attention/backends/mla/rocm_aiter_mla.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v2.py`, `vllm/v1/attention/backends/mla/rocm_aiter_mla.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #41706 - [Model] use AutoWeightsLoader for DeepSeekV2

- Link: https://github.com/vllm-project/vllm/pull/41706
- Status/date: merged / 2026-05-09
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +186/-169, 389 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] use AutoWeightsLoader for DeepSeekV2"; model line: DeepSeek V3.1; category: model implementation change; main diff: `vllm/model_executor/models/deepseek_v2.py`; technical summary: Covers "[Model] use AutoWeightsLoader for DeepSeekV2"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +186/-169 (355 lines); hunks: -83,6 +83,7; -1254,6 +1255,16 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: st...; symbols: __init__, embed_input_ids, forward, DeepseekV2MixtureOfExperts, touching `__init__, embed_input_ids, forward`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +186/-169 (355 lines); hunks: -83,6 +83,7; -1254,6 +1255,16 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: st...; symbols: __init__, embed_input_ids, forward, DeepseekV2MixtureOfExperts
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -83,6 +83,7 @@
+    AutoWeightsLoader,
@@ -1254,6 +1255,16 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
+        # Needed by load_weights
+        qk_nope_head_dim = getattr(config, "qk_nope_head_dim", 0)
+        qk_rope_head_dim = getattr(config, "qk_rope_head_dim", 0)
+        self.use_mha = config.model_type == "deepseek" or all(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +186/-169
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #43781 - [Bugfix][ROCm] Fix Accuracy Drop in Sparse Indexer on gfx950

- Link: https://github.com/vllm-project/vllm/pull/43781
- Status/date: merged / 2026-05-28
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +14/-4, 82 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix][ROCm] Fix Accuracy Drop in Sparse Indexer on gfx950"; model line: DeepSeek V3.1; category: bug fix; main diff: `vllm/model_executor/models/deepseek_v2.py`, `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`; technical summary: Covers "[Bugfix][ROCm] Fix Accuracy Drop in Sparse Indexer on gfx950"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`, `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +9/-1 (10 lines); hunks: -612,6 +612,7 @@ def __init__(; -673,15 +674,21 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`; `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` modified +5/-3 (8 lines); hunks: -107,6 +107,7 @@ def indexer_k_quant_and_cache_triton(; -118,7 +119,7 @@ def indexer_k_quant_and_cache_triton(; symbols: indexer_k_quant_and_cache_triton, cp_gather_indexer_k_quant_cache_triton, rocm_fp8_paged_mqa_logits, touching `indexer_k_quant_and_cache_triton, cp_gather_indexer_k_quant_cache_triton, rocm_fp8_paged_mqa_logits`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +9/-1 (10 lines); hunks: -612,6 +612,7 @@ def __init__(; -673,15 +674,21 @@ def __init__(; symbols: __init__, forward
  - `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` modified +5/-3 (8 lines); hunks: -107,6 +107,7 @@ def indexer_k_quant_and_cache_triton(; -118,7 +119,7 @@ def indexer_k_quant_and_cache_triton(; symbols: indexer_k_quant_and_cache_triton, cp_gather_indexer_k_quant_cache_triton, rocm_fp8_paged_mqa_logits
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -612,6 +612,7 @@ def __init__(
+        is_inplace_rope: bool = False,
@@ -673,15 +674,21 @@ def __init__(
+        self.is_inplace_rope = is_inplace_rope
-        if current_platform.is_rocm():
+        if current_platform.is_rocm() and self.is_inplace_rope:
+            # This fast path relies on rotary_emb mutating q and k inplace.
diff -- vllm/v1/attention/ops/rocm_aiter_mla_sparse.py
@@ -107,6 +107,7 @@ def indexer_k_quant_and_cache_triton(
+    layout = "NORMAL" if block_size == 1 else "SHUFFLE"
@@ -118,7 +119,7 @@ def indexer_k_quant_and_cache_triton(
-        "SHUFFLE",
+        layout,
@@ -229,6 +230,7 @@ def cp_gather_indexer_k_quant_cache_triton(
+    layout = "NORMAL" if block_size == 1 else "SHUFFLE"
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +9/-1; `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` modified +5/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v2.py`, `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #42982 - [ROCm][Perf] DSv3.2 MI355X TP4 decode-step orchestration cleanup (3 micro-opts)

- Link: https://github.com/vllm-project/vllm/pull/42982
- Status/date: merged / 2026-05-29
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +59/-29, 125 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm][Perf] DSv3.2 MI355X TP4 decode-step orchestration cleanup (3 micro-opts)"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `vllm/model_executor/models/deepseek_v2.py`, `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py`; technical summary: Covers "[ROCm][Perf] DSv3.2 MI355X TP4 decode-step orchestration cleanup (3 micro-opts)"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`, `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +2/-4 (6 lines); hunks: -304,10 +304,8 @@ def __init__(; symbols: __init__, touching `__init__`; `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py` modified +57/-25 (82 lines); hunks: -375,7 +375,7 @@ def __init__(; -458,6 +458,10 @@ def __init__(; symbols: __init__, build, touching `__init__, build`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +2/-4 (6 lines); hunks: -304,10 +304,8 @@ def __init__(; symbols: __init__
  - `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py` modified +57/-25 (82 lines); hunks: -375,7 +375,7 @@ def __init__(; -458,6 +458,10 @@ def __init__(; symbols: __init__, build
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -304,10 +304,8 @@ def __init__(
-            # AITER biased_grouped_topk requires the correction bias dtype to
-            # match the router logits. Keep DeepSeek's correction bias in fp32
-            # by requesting fp32 router logits for this routing path.
-            self.gate.set_out_dtype(torch.float32)
+            # Accumulates in fp32; avoids bf16->fp32 cast.
+            self.gate.set_out_dtype(self.gate.weight.dtype)
diff -- vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py
@@ -375,7 +375,7 @@ def __init__(
-        self.req_id_per_token_buffer = torch.empty(
+        self.req_id_per_token_buffer = torch.zeros(
@@ -458,6 +458,10 @@ def __init__(
+        self._prev_req_extent: int = 0
+        self._prev_indices_extent: int = 0
+        self._prev_metadata_key: tuple | None = None
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +2/-4; `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py` modified +57/-25
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v2.py`, `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #42944 - fix: glm5.1 pp model loading

- Link: https://github.com/vllm-project/vllm/pull/42944
- Status/date: merged / 2026-06-01
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +25/-5, 93 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix: glm5.1 pp model loading"; model line: DeepSeek V3.1; category: bug fix; main diff: `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`; technical summary: Covers "fix: glm5.1 pp model loading"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +17/-3 (20 lines); hunks: -105,6 +105,7; -742,7 +743,9 @@ def forward(; symbols: forward, _try_load_fp8_indexer_wk, touching `forward, _try_load_fp8_indexer_wk`; `vllm/model_executor/models/deepseek_mtp.py` modified +8/-2 (10 lines); hunks: -35,7 +35,7; -267,6 +267,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +17/-3 (20 lines); hunks: -105,6 +105,7; -742,7 +743,9 @@ def forward(; symbols: forward, _try_load_fp8_indexer_wk
  - `vllm/model_executor/models/deepseek_mtp.py` modified +8/-2 (10 lines); hunks: -35,7 +35,7; -267,6 +267,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -105,6 +105,7 @@
+    get_pp_missing_layer_names,
@@ -742,7 +743,9 @@ def forward(
-def _try_load_fp8_indexer_wk(name, tensor, buf, params_dict, loaded_params):
+def _try_load_fp8_indexer_wk(
+    name, tensor, buf, params_dict, loaded_params, pp_missing_layer_names
+):
diff -- vllm/model_executor/models/deepseek_mtp.py
@@ -35,7 +35,7 @@
-from .utils import maybe_prefix
+from .utils import get_pp_missing_layer_names, maybe_prefix
@@ -267,6 +267,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
+        pp_missing_layer_names = get_pp_missing_layer_names(self)
@@ -282,7 +283,12 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
-                name, loaded_weight, _pending_wk_fp8, params_dict, loaded_params
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +17/-3; `vllm/model_executor/models/deepseek_mtp.py` modified +8/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_mtp.py`, `vllm/model_executor/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #44420 - [feature] add index share feature for DSA MTP

- Link: https://github.com/vllm-project/vllm/pull/44420
- Status/date: merged / 2026-06-07
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +114/-25, 230 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[feature] add index share feature for DSA MTP"; model line: DeepSeek V3.1; category: model support/runtime entry; main diff: `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`, `vllm/v1/spec_decode/llm_base_proposer.py`; technical summary: Covers "[feature] add index share feature for DSA MTP"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`, `vllm/v1/spec_decode/llm_base_proposer.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +16/-15 (31 lines); hunks: -1018,19 +1018,20 @@ def __init__(; -1252,8 +1253,8 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, touching `__init__`; `vllm/model_executor/models/deepseek_mtp.py` modified +24/-2 (26 lines); hunks: -115,7 +115,9 @@ def forward(; -147,6 +149,22 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: forward, __init__, set_skip_topk, embed_input_ids, touching `forward, __init__, set_skip_topk`; `vllm/v1/spec_decode/llm_base_proposer.py` modified +32/-3 (35 lines); hunks: -70,6 +70,7 @@ def __init__(; -490,6 +491,11 @@ def propose(; symbols: __init__, propose, _maybe_share_lm_head, touching `__init__, propose, _maybe_share_lm_head`; `vllm/transformers_utils/model_arch_config_convertor.py` modified +33/-1 (34 lines); hunks: -50,7 +50,7 @@ def get_head_size(self) -> int:; -71,6 +71,38 @@ def get_head_size(self) -> int:; symbols: get_head_size, _get_qk_rope_head_dim, get_total_num_kv_heads, touching `get_head_size, _get_qk_rope_head_dim, get_total_num_kv_heads`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +16/-15 (31 lines); hunks: -1018,19 +1018,20 @@ def __init__(; -1252,8 +1253,8 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__
  - `vllm/model_executor/models/deepseek_mtp.py` modified +24/-2 (26 lines); hunks: -115,7 +115,9 @@ def forward(; -147,6 +149,22 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: forward, __init__, set_skip_topk, embed_input_ids
  - `vllm/v1/spec_decode/llm_base_proposer.py` modified +32/-3 (35 lines); hunks: -70,6 +70,7 @@ def __init__(; -490,6 +491,11 @@ def propose(; symbols: __init__, propose, _maybe_share_lm_head
  - `vllm/transformers_utils/model_arch_config_convertor.py` modified +33/-1 (34 lines); hunks: -50,7 +50,7 @@ def get_head_size(self) -> int:; -71,6 +71,38 @@ def get_head_size(self) -> int:; symbols: get_head_size, _get_qk_rope_head_dim, get_total_num_kv_heads
  - `vllm/v1/worker/gpu/spec_decode/eagle/utils.py` modified +9/-4 (13 lines); hunks: -76,10 +76,15 @@ def load_eagle_model(target_model: nn.Module, vllm_config: V...; symbols: load_eagle_model
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -1018,19 +1018,20 @@ def __init__(
-            # Enable IndexCache for DeepSeek models to reduce redundant top-k
-            # token selection computations in sparse attention.
-            use_index_cache = getattr(config, "use_index_cache", False)
-            if use_index_cache:
-                # IndexCache config
-                # Refer: https://arxiv.org/abs/2603.12201 for more details.
diff -- vllm/model_executor/models/deepseek_mtp.py
@@ -115,7 +115,9 @@ def forward(
-            positions=positions, hidden_states=hidden_states, residual=None
+            positions=positions,
+            hidden_states=hidden_states,
+            residual=None,
@@ -147,6 +149,22 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
+    def set_skip_topk(self, skip: bool):
diff -- vllm/v1/spec_decode/llm_base_proposer.py
@@ -70,6 +70,7 @@ def __init__(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +16/-15; `vllm/model_executor/models/deepseek_mtp.py` modified +24/-2; `vllm/v1/spec_decode/llm_base_proposer.py` modified +32/-3; `vllm/transformers_utils/model_arch_config_convertor.py` modified +33/-1; `vllm/v1/worker/gpu/spec_decode/eagle/utils.py` modified +9/-4
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_mtp.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/transformers_utils/model_arch_config_convertor.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #45003 - [Frontend] Support strict mode for tool calling

- Link: https://github.com/vllm-project/vllm/pull/45003
- Status/date: merged / 2026-06-12
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 29 files, +672/-1936, 3162 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Frontend] Support strict mode for tool calling"; model line: DeepSeek V3.1; category: docs/tests/CI; main diff: `vllm/tool_parsers/qwen3xml_tool_parser.py`, `vllm/tool_parsers/structural_tag_registry.py`, `tests/tool_parsers/test_structural_tag_registry.py`; technical summary: Covers "[Frontend] Support strict mode for tool calling"; the main implementation surface is `vllm/tool_parsers/qwen3xml_tool_parser.py`, `vllm/tool_parsers/structural_tag_registry.py`, `tests/tool_parsers/test_structural_tag_registry.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/tool_parsers/qwen3xml_tool_parser.py` removed +0/-1300 (1300 lines); hunks: -1,1300 +0,0; symbols: StreamingXMLToolCallParser, __init__, reset_streaming_state, parse_single_streaming_chunks, touching `StreamingXMLToolCallParser, __init__, reset_streaming_state`; `vllm/tool_parsers/structural_tag_registry.py` modified +174/-240 (414 lines); hunks: -1,14 +1,15; -24,23 +25,51; symbols: register_model_structural_tag, register_vllm_structural_tag, decorator, get_model_structural_tag, touching `register_model_structural_tag, register_vllm_structural_tag, decorator`; `tests/tool_parsers/test_structural_tag_registry.py` added +314/-0 (314 lines); hunks: -0,0 +1,314; symbols: sample_tools, test_supported_structural_tag_models_include_vllm_builtins, test_get_model_structural_tag_supports_all_xgrammar_builtins, test_get_model_structural_tag_supports_vllm_hermes, touching `sample_tools, test_supported_structural_tag_models_include_vllm_builtins, test_get_model_structural_tag_supports_all_xgrammar_builtins`; `tests/tool_parsers/test_qwen3coder_tool_parser.py` modified +13/-190 (203 lines); hunks: -3,6 +3,7; -19,15 +20,12; symbols: qwen3_tool_parser, qwen3_xml_tool_parser, qwen3_tool_parser_parametrized, assert_tool_calls, touching `qwen3_tool_parser, qwen3_xml_tool_parser, qwen3_tool_parser_parametrized`.
- Code diff details:
  - `vllm/tool_parsers/qwen3xml_tool_parser.py` removed +0/-1300 (1300 lines); hunks: -1,1300 +0,0; symbols: StreamingXMLToolCallParser, __init__, reset_streaming_state, parse_single_streaming_chunks
  - `vllm/tool_parsers/structural_tag_registry.py` modified +174/-240 (414 lines); hunks: -1,14 +1,15; -24,23 +25,51; symbols: register_model_structural_tag, register_vllm_structural_tag, decorator, get_model_structural_tag
  - `tests/tool_parsers/test_structural_tag_registry.py` added +314/-0 (314 lines); hunks: -0,0 +1,314; symbols: sample_tools, test_supported_structural_tag_models_include_vllm_builtins, test_get_model_structural_tag_supports_all_xgrammar_builtins, test_get_model_structural_tag_supports_vllm_hermes
  - `tests/tool_parsers/test_qwen3coder_tool_parser.py` modified +13/-190 (203 lines); hunks: -3,6 +3,7; -19,15 +20,12; symbols: qwen3_tool_parser, qwen3_xml_tool_parser, qwen3_tool_parser_parametrized, assert_tool_calls
  - `tests/tool_parsers/test_qwen3xml_tool_parser.py` removed +0/-72 (72 lines); hunks: -1,72 +0,0; symbols: TestQwen3xmlToolParser, test_config
- Key code excerpts:

```diff
diff -- vllm/tool_parsers/qwen3xml_tool_parser.py
@@ -1,1300 +0,0 @@
-# SPDX-License-Identifier: Apache-2.0
-# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-import json
-from collections.abc import Sequence
-from typing import Any
-from xml.parsers.expat import ParserCreate
diff -- vllm/tool_parsers/structural_tag_registry.py
@@ -1,14 +1,15 @@
-# Model-specific structural tag builders adapted from XGrammar's
-# builtin structural tag implementations:
-# https://github.com/mlc-ai/xgrammar/blob/main/python/xgrammar/builtin_structural_tag.py
-from xgrammar import StructuralTag
+from xgrammar import StructuralTag, normalize_tool_choice
+from xgrammar import get_model_structural_tag as get_xgrammar_model_structural_tag
diff -- tests/tool_parsers/test_structural_tag_registry.py
@@ -0,0 +1,314 @@
```

- Reviewed files:
  - runtime: `vllm/tool_parsers/qwen3xml_tool_parser.py` removed +0/-1300; `vllm/tool_parsers/structural_tag_registry.py` modified +174/-240; `vllm/tool_parsers/abstract_tool_parser.py` modified +36/-28; `vllm/entrypoints/serve/render/serving.py` modified +24/-28; `vllm/tool_parsers/deepseekv4_tool_parser.py` modified +1/-15
  - tests: `tests/tool_parsers/test_structural_tag_registry.py` added +314/-0; `tests/tool_parsers/test_qwen3coder_tool_parser.py` modified +13/-190; `tests/tool_parsers/test_qwen3xml_tool_parser.py` removed +0/-72
- Risk and verification: The diff ships test coverage in `requirements/test/rocm.txt`, `tests/entrypoints/openai/chat_completion/test_completion_with_function_calling.py`, `tests/entrypoints/openai/responses/conftest.py`, `tests/parser/test_parse.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #45895 - [bugfix]Indexer init skip and MTP TopK share for iteration

- Link: https://github.com/vllm-project/vllm/pull/45895
- Status/date: merged / 2026-06-19
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +69/-30, 198 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[bugfix]Indexer init skip and MTP TopK share for iteration"; model line: DeepSeek V3.1; category: bug fix; main diff: `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`, `vllm/model_executor/layers/attention/mla_attention.py`; technical summary: Covers "[bugfix]Indexer init skip and MTP TopK share for iteration"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`, `vllm/model_executor/layers/attention/mla_attention.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +22/-17 (39 lines); hunks: -998,8 +998,29 @@ def __init__(; -1017,22 +1038,6 @@ def __init__(; symbols: __init__, touching `__init__`; `vllm/model_executor/models/deepseek_mtp.py` modified +6/-2 (8 lines); hunks: -119,8 +119,12 @@ def forward(; symbols: forward, DeepSeekMultiTokenPredictor, touching `forward, DeepSeekMultiTokenPredictor`; `vllm/model_executor/layers/attention/mla_attention.py` modified +6/-0 (6 lines); hunks: -349,6 +349,7 @@ def __init__(; -437,6 +438,11 @@ def __init__(; symbols: __init__, touching `__init__`; `vllm/model_executor/layers/mla.py` modified +1/-0 (1 lines); hunks: -112,6 +112,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +22/-17 (39 lines); hunks: -998,8 +998,29 @@ def __init__(; -1017,22 +1038,6 @@ def __init__(; symbols: __init__
  - `vllm/model_executor/models/deepseek_mtp.py` modified +6/-2 (8 lines); hunks: -119,8 +119,12 @@ def forward(; symbols: forward, DeepSeekMultiTokenPredictor
  - `vllm/model_executor/layers/attention/mla_attention.py` modified +6/-0 (6 lines); hunks: -349,6 +349,7 @@ def __init__(; -437,6 +438,11 @@ def __init__(; symbols: __init__
  - `vllm/model_executor/layers/mla.py` modified +1/-0 (1 lines); hunks: -112,6 +112,7 @@ def __init__(; symbols: __init__
  - `vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py` modified +7/-3 (10 lines); hunks: -271,7 +271,7 @@ def __init__(; -301,8 +301,12 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -998,8 +998,29 @@ def __init__(
+        # IndexCache config
+        # Refer: https://arxiv.org/abs/2603.12201 for more details.
-        if self.is_v32:
+        _index_topk_freq = getattr(config, "index_topk_freq", 1)
+        _index_topk_pattern = getattr(config, "index_topk_pattern", None)
+        _index_skip_topk_offset = getattr(config, "index_skip_topk_offset", 2)
diff -- vllm/model_executor/models/deepseek_mtp.py
@@ -119,8 +119,12 @@ def forward(
-        hidden_states = residual + hidden_states
-        return hidden_states
+        hidden_states = residual + hidden_states  # pre-final-norm (logits hidden)
+        # Recycle the post-final-norm hidden into the next draft step.
+        # compute_logits applies shared_head (== final norm) to the pre-norm
+        # element, so logits and the recycle each get exactly one final-norm.
diff -- vllm/model_executor/layers/attention/mla_attention.py
@@ -349,6 +349,7 @@ def __init__(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +22/-17; `vllm/model_executor/models/deepseek_mtp.py` modified +6/-2; `vllm/model_executor/layers/attention/mla_attention.py` modified +6/-0; `vllm/model_executor/layers/mla.py` modified +1/-0; `vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py` modified +7/-3; `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py` modified +7/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/attention/mla_attention.py`, `vllm/model_executor/layers/mla.py`, `vllm/model_executor/models/deepseek_mtp.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #46199 - [Bugfix] Move extract_layer_index back inside is_v32 guard

- Link: https://github.com/vllm-project/vllm/pull/46199
- Status/date: merged / 2026-06-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +23/-17, 48 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Move extract_layer_index back inside is_v32 guard"; model line: DeepSeek V3.1; category: bug fix; main diff: `vllm/model_executor/models/deepseek_v2.py`; technical summary: Covers "[Bugfix] Move extract_layer_index back inside is_v32 guard"; the main implementation surface is `vllm/model_executor/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_v2.py` modified +23/-17 (40 lines); hunks: -1001,24 +1001,30 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +23/-17 (40 lines); hunks: -1001,24 +1001,30 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -1001,24 +1001,30 @@ def __init__(
-        _index_topk_freq = getattr(config, "index_topk_freq", 1)
-        _index_topk_pattern = getattr(config, "index_topk_pattern", None)
-        _index_skip_topk_offset = getattr(config, "index_skip_topk_offset", 2)
-        layer_id = extract_layer_index(prefix)
-        if _index_topk_pattern is None:
-            _skip_topk = (
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +23/-17
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #46651 - [Perf] Remove redundant clone for GLM, Deepseek etc

- Link: https://github.com/vllm-project/vllm/pull/46651
- Status/date: merged / 2026-06-25
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +4/-4, 36 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Perf] Remove redundant clone for GLM, Deepseek etc"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `vllm/model_executor/models/AXK1.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/glm4_moe_lite.py`; technical summary: Covers "[Perf] Remove redundant clone for GLM, Deepseek etc"; the main implementation surface is `vllm/model_executor/models/AXK1.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/glm4_moe_lite.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/AXK1.py` modified +1/-1 (2 lines); hunks: -649,7 +649,7 @@ def forward(; symbols: forward, touching `forward`; `vllm/model_executor/models/deepseek_v2.py` modified +1/-1 (2 lines); hunks: -1186,7 +1186,7 @@ def forward(; symbols: forward, touching `forward`; `vllm/model_executor/models/glm4_moe_lite.py` modified +1/-1 (2 lines); hunks: -184,7 +184,7 @@ def forward(; symbols: forward, touching `forward`; `vllm/model_executor/models/openpangu.py` modified +1/-1 (2 lines); hunks: -935,7 +935,7 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `vllm/model_executor/models/AXK1.py` modified +1/-1 (2 lines); hunks: -649,7 +649,7 @@ def forward(; symbols: forward
  - `vllm/model_executor/models/deepseek_v2.py` modified +1/-1 (2 lines); hunks: -1186,7 +1186,7 @@ def forward(; symbols: forward
  - `vllm/model_executor/models/glm4_moe_lite.py` modified +1/-1 (2 lines); hunks: -184,7 +184,7 @@ def forward(; symbols: forward
  - `vllm/model_executor/models/openpangu.py` modified +1/-1 (2 lines); hunks: -935,7 +935,7 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/AXK1.py
@@ -649,7 +649,7 @@ def forward(
-            residual = hidden_states.clone()
+            residual = hidden_states
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -1186,7 +1186,7 @@ def forward(
-            residual = hidden_states.clone()
+            residual = hidden_states
diff -- vllm/model_executor/models/glm4_moe_lite.py
@@ -184,7 +184,7 @@ def forward(
-            residual = hidden_states.clone()
+            residual = hidden_states
diff -- vllm/model_executor/models/openpangu.py
@@ -935,7 +935,7 @@ def forward(
-            residual = hidden_states.clone()
+            residual = hidden_states
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/AXK1.py` modified +1/-1; `vllm/model_executor/models/deepseek_v2.py` modified +1/-1; `vllm/model_executor/models/glm4_moe_lite.py` modified +1/-1; `vllm/model_executor/models/openpangu.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/AXK1.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/glm4_moe_lite.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
