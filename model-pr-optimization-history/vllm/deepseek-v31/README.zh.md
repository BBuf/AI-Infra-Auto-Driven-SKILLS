# vllm DeepSeek V3.1 模型 PR 优化历史

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `examples/ray_serving/elastic_ep/serve_deepseek_v2.sh` | 无直接 PR 号提交 |
| `examples/tool_chat_template_deepseekv31.jinja` | [#23454](https://github.com/vllm-project/vllm/pull/23454) |
| `tests/tool_parsers/test_deepseekv31_tool_parser.py` | 无直接 PR 号提交 |
| `vllm/model_executor/models/deepseek_mtp.py` | 无直接 PR 号提交 |
| `vllm/model_executor/models/deepseek_v2.py` | 无直接 PR 号提交 |
| `vllm/tool_parsers/deepseekv31_tool_parser.py` | 无直接 PR 号提交 |

## PR 覆盖总览

- git 追溯 PR 数: 1
- 原文档显式引用补充 PR 数: 52
- 当前文档总 PR 数: 53
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
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

## 逐 PR diff 审计卡

### PR #23454 - Support DeepSeek-V3.1 tool call

- 链接: https://github.com/vllm-project/vllm/pull/23454
- 状态/时间: merged / 2025-08-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `examples/tool_chat_template_deepseekv31.jinja`；关联提交 `b8f17f5d980e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+468/-0，可读 patch 491 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support DeepSeek-V3.1 tool call」；模型线: DeepSeek V3.1；类别: 模型支持/运行时入口；主要 diff: `examples/tool_chat_template_deepseekv31.jinja`；技术摘要: 覆盖「Support DeepSeek-V3.1 tool call」；主要实现面是 `examples/tool_chat_template_deepseekv31.jinja`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `examples/tool_chat_template_deepseekv31.jinja` added +91/-0 (91 lines); hunks: -0,0 +1,91。
- 代码 diff 细节:
  - `examples/tool_chat_template_deepseekv31.jinja` added +91/-0 (91 lines); hunks: -0,0 +1,91
- 关键代码摘录:

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

- 已读文件:
  - docs: `examples/tool_chat_template_deepseekv31.jinja` added +91/-0
- 验证与风险: runtime 路径改动集中在 `vllm/entrypoints/openai/tool_parsers/__init__.py`, `vllm/entrypoints/openai/tool_parsers/deepseekv31_tool_parser.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23666 - [Feature] Add Hopper DeepGEMM E8M0 for DeepSeekV3.1 scale_fmt

- 链接: https://github.com/vllm-project/vllm/pull/23666
- 状态/时间: merged / 2025-08-27
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+68/-53，可读 patch 322 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Feature] Add Hopper DeepGEMM E8M0 for DeepSeekV3.1 scale_fmt」；模型线: DeepSeek V3.1；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/quantization/fp8.py`, `vllm/model_executor/layers/fused_moe/fused_moe.py`, `vllm/model_executor/layers/fused_moe/triton_deep_gemm_moe.py`；技术摘要: 覆盖「[Feature] Add Hopper DeepGEMM E8M0 for DeepSeekV3.1 scale_fmt」；主要实现面是 `vllm/model_executor/layers/quantization/fp8.py`, `vllm/model_executor/layers/fused_moe/fused_moe.py`, `vllm/model_executor/layers/fused_moe/triton_deep_gemm_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/quantization/fp8.py` modified +4/-5 (9 lines); hunks: -48,8 +48,7; -427,7 +426,7 @@ def process_weights_after_loading(self, layer: Module) -> None:; symbols: process_weights_after_loading，涉及 `process_weights_after_loading`；`vllm/model_executor/layers/fused_moe/fused_moe.py` modified +3/-4 (7 lines); hunks: -40,7 +40,7; -1431,9 +1431,8 @@ def fused_experts(hidden_states: torch.Tensor,; symbols: fused_experts，涉及 `fused_experts`；`vllm/model_executor/layers/fused_moe/triton_deep_gemm_moe.py` modified +3/-3 (6 lines); hunks: -10,7 +10,7; -107,7 +107,7 @@ def workspace_shapes(; symbols: TritonOrDeepGemmExperts, workspace_shapes, apply，涉及 `TritonOrDeepGemmExperts, workspace_shapes, apply`；`vllm/model_executor/layers/fused_moe/batched_deep_gemm_moe.py` modified +2/-2 (4 lines); hunks: -12,7 +12,7; -174,7 +174,7 @@ def silu_mul_fp8_quant_deep_gemm(; symbols: silu_mul_fp8_quant_deep_gemm，涉及 `silu_mul_fp8_quant_deep_gemm`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/fp8.py` modified +4/-5 (9 lines); hunks: -48,8 +48,7; -427,7 +426,7 @@ def process_weights_after_loading(self, layer: Module) -> None:; symbols: process_weights_after_loading
  - `vllm/model_executor/layers/fused_moe/fused_moe.py` modified +3/-4 (7 lines); hunks: -40,7 +40,7; -1431,9 +1431,8 @@ def fused_experts(hidden_states: torch.Tensor,; symbols: fused_experts
  - `vllm/model_executor/layers/fused_moe/triton_deep_gemm_moe.py` modified +3/-3 (6 lines); hunks: -10,7 +10,7; -107,7 +107,7 @@ def workspace_shapes(; symbols: TritonOrDeepGemmExperts, workspace_shapes, apply
  - `vllm/model_executor/layers/fused_moe/batched_deep_gemm_moe.py` modified +2/-2 (4 lines); hunks: -12,7 +12,7; -174,7 +174,7 @@ def silu_mul_fp8_quant_deep_gemm(; symbols: silu_mul_fp8_quant_deep_gemm
  - `vllm/model_executor/layers/quantization/utils/fp8_utils.py` modified +2/-2 (4 lines); hunks: -20,7 +20,7; -385,7 +385,7 @@ def per_token_group_quant_fp8(; symbols: per_token_group_quant_fp8
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/fp8.py` modified +4/-5; `vllm/model_executor/layers/fused_moe/fused_moe.py` modified +3/-4; `vllm/model_executor/layers/fused_moe/triton_deep_gemm_moe.py` modified +3/-3; `vllm/model_executor/layers/fused_moe/batched_deep_gemm_moe.py` modified +2/-2; `vllm/model_executor/layers/quantization/utils/fp8_utils.py` modified +2/-2; `vllm/utils/deep_gemm.py` modified +24/-29
- 验证与风险: diff 自带测试面 `tests/kernels/moe/test_block_fp8.py`, `tests/kernels/moe/test_deepep_deepgemm_moe.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #25589 - [Model] Add DeepSeek-V3.1 reasoning parser (split from PR #24972)

- 链接: https://github.com/vllm-project/vllm/pull/25589
- 状态/时间: merged / 2025-10-15
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+215/-3，可读 patch 269 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Add DeepSeek-V3.1 reasoning parser (split from PR #24972)」；模型线: DeepSeek V3.1；类别: 文档/测试/CI；主要 diff: `tests/reasoning/test_deepseekv3_reasoning_parser.py`, `vllm/reasoning/deepseek_v3_reasoning_parser.py`, `vllm/reasoning/identity_reasoning_parser.py`；技术摘要: 覆盖「[Model] Add DeepSeek-V3.1 reasoning parser (split from PR #24972)」；主要实现面是 `tests/reasoning/test_deepseekv3_reasoning_parser.py`, `vllm/reasoning/deepseek_v3_reasoning_parser.py`, `vllm/reasoning/identity_reasoning_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/reasoning/test_deepseekv3_reasoning_parser.py` added +76/-0 (76 lines); hunks: -0,0 +1,76; symbols: tokenizer, test_parser_selection, test_identity_reasoning_parser_basic，涉及 `tokenizer, test_parser_selection, test_identity_reasoning_parser_basic`；`vllm/reasoning/deepseek_v3_reasoning_parser.py` added +66/-0 (66 lines); hunks: -0,0 +1,66; symbols: DeepSeekV3ReasoningParser, __init__, is_reasoning_end, extract_content_ids，涉及 `DeepSeekV3ReasoningParser, __init__, is_reasoning_end`；`vllm/reasoning/identity_reasoning_parser.py` added +58/-0 (58 lines); hunks: -0,0 +1,58; symbols: IdentityReasoningParser, __init__, is_reasoning_end, extract_content_ids，涉及 `IdentityReasoningParser, __init__, is_reasoning_end`；`vllm/entrypoints/openai/serving_chat.py` modified +8/-2 (10 lines); hunks: -573,7 +573,10 @@ async def chat_completion_stream_generator(; -1342,7 +1345,10 @@ async def chat_completion_full_generator(; symbols: chat_completion_stream_generator, chat_completion_full_generator，涉及 `chat_completion_stream_generator, chat_completion_full_generator`。
- 代码 diff 细节:
  - `tests/reasoning/test_deepseekv3_reasoning_parser.py` added +76/-0 (76 lines); hunks: -0,0 +1,76; symbols: tokenizer, test_parser_selection, test_identity_reasoning_parser_basic
  - `vllm/reasoning/deepseek_v3_reasoning_parser.py` added +66/-0 (66 lines); hunks: -0,0 +1,66; symbols: DeepSeekV3ReasoningParser, __init__, is_reasoning_end, extract_content_ids
  - `vllm/reasoning/identity_reasoning_parser.py` added +58/-0 (58 lines); hunks: -0,0 +1,58; symbols: IdentityReasoningParser, __init__, is_reasoning_end, extract_content_ids
  - `vllm/entrypoints/openai/serving_chat.py` modified +8/-2 (10 lines); hunks: -573,7 +573,10 @@ async def chat_completion_stream_generator(; -1342,7 +1345,10 @@ async def chat_completion_full_generator(; symbols: chat_completion_stream_generator, chat_completion_full_generator
  - `docs/features/reasoning_outputs.md` modified +3/-1 (4 lines); hunks: -11,6 +11,7 @@ vLLM currently supports the following reasoning models:; -20,8 +21,9 @@ vLLM currently supports the following reasoning models:
- 关键代码摘录:

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

- 已读文件:
  - tests: `tests/reasoning/test_deepseekv3_reasoning_parser.py` added +76/-0
  - runtime: `vllm/reasoning/deepseek_v3_reasoning_parser.py` added +66/-0; `vllm/reasoning/identity_reasoning_parser.py` added +58/-0; `vllm/entrypoints/openai/serving_chat.py` modified +8/-2; `vllm/reasoning/__init__.py` modified +4/-0
  - docs: `docs/features/reasoning_outputs.md` modified +3/-1
- 验证与风险: diff 自带测试面 `tests/reasoning/test_deepseekv3_reasoning_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #29867 - [Quantization] fix: overflow with static per-tensor scaling

- 链接: https://github.com/vllm-project/vllm/pull/29867
- 状态/时间: merged / 2026-01-13
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+71/-56，可读 patch 182 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Quantization] fix: overflow with static per-tensor scaling」；模型线: DeepSeek V3.1；类别: 缺陷修复；主要 diff: `vllm/model_executor/layers/quantization/utils/quant_utils.py`, `vllm/v1/attention/backends/mla/common.py`；技术摘要: 覆盖「[Quantization] fix: overflow with static per-tensor scaling」；主要实现面是 `vllm/model_executor/layers/quantization/utils/quant_utils.py`, `vllm/v1/attention/backends/mla/common.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/quantization/utils/quant_utils.py` modified +61/-2 (63 lines); hunks: -5,7 +5,7; -15,6 +15,9; symbols: scaled_dequantize, get_attribute_fallback, get_and_maybe_dequant_weights, pack_quantized_values_into_int32，涉及 `scaled_dequantize, get_attribute_fallback, get_and_maybe_dequant_weights`；`vllm/v1/attention/backends/mla/common.py` modified +10/-54 (64 lines); hunks: -207,8 +207,9; -1184,35 +1185,13 @@ def __init__(; symbols: __init__, process_weights_after_loading, get_layer_weight, get_and_maybe_dequant_weights，涉及 `__init__, process_weights_after_loading, get_layer_weight`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/utils/quant_utils.py` modified +61/-2 (63 lines); hunks: -5,7 +5,7; -15,6 +15,9; symbols: scaled_dequantize, get_attribute_fallback, get_and_maybe_dequant_weights, pack_quantized_values_into_int32
  - `vllm/v1/attention/backends/mla/common.py` modified +10/-54 (64 lines); hunks: -207,8 +207,9; -1184,35 +1185,13 @@ def __init__(; symbols: __init__, process_weights_after_loading, get_layer_weight, get_and_maybe_dequant_weights
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/utils/quant_utils.py` modified +61/-2; `vllm/v1/attention/backends/mla/common.py` modified +10/-54
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/quantization/utils/quant_utils.py`, `vllm/v1/attention/backends/mla/common.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #32361 - [BugFix] Fix DeepSeek-V3.1 + DeepGEMM incompatible scale shapes

- 链接: https://github.com/vllm-project/vllm/pull/32361
- 状态/时间: merged / 2026-01-15
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-0，可读 patch 10 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[BugFix] Fix DeepSeek-V3.1 + DeepGEMM incompatible scale shapes」；模型线: DeepSeek V3.1；类别: 缺陷修复；主要 diff: `vllm/model_executor/layers/quantization/utils/quant_utils.py`；技术摘要: 覆盖「[BugFix] Fix DeepSeek-V3.1 + DeepGEMM incompatible scale shapes」；主要实现面是 `vllm/model_executor/layers/quantization/utils/quant_utils.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/quantization/utils/quant_utils.py` modified +3/-0 (3 lines); hunks: -299,6 +299,9 @@ def get_and_maybe_dequant_weights(; symbols: get_and_maybe_dequant_weights，涉及 `get_and_maybe_dequant_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/utils/quant_utils.py` modified +3/-0 (3 lines); hunks: -299,6 +299,9 @@ def get_and_maybe_dequant_weights(; symbols: get_and_maybe_dequant_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/utils/quant_utils.py
@@ -299,6 +299,9 @@ def get_and_maybe_dequant_weights(
+        # DeepGEMM transforms the scales using `transform_sf_into_required_layout` into
+        # a layout that is not compatible with `scaled_dequantize`.
+        and not layer.quant_method.use_deep_gemm
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/utils/quant_utils.py` modified +3/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/quantization/utils/quant_utils.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #32175 - [Bugfix] [DeepSeek-V3.2] fix sparse_attn_indexer padding

- 链接: https://github.com/vllm-project/vllm/pull/32175
- 状态/时间: merged / 2026-01-16
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+9/-2，可读 patch 38 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] [DeepSeek-V3.2] fix sparse_attn_indexer padding」；模型线: DeepSeek V3.1；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/deepseek_v2.py`；技术摘要: 覆盖「[Bugfix] [DeepSeek-V3.2] fix sparse_attn_indexer padding」；主要实现面是 `vllm/model_executor/models/deepseek_v2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +9/-2 (11 lines); hunks: -717,13 +717,20 @@ def sparse_attn_indexer(; -739,14 +746,14 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer，涉及 `sparse_attn_indexer`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +9/-2 (11 lines); hunks: -717,13 +717,20 @@ def sparse_attn_indexer(; -739,14 +746,14 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +9/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #32652 - [Bugfix] Fix the fp8_mqa_logits dim mismatch

- 链接: https://github.com/vllm-project/vllm/pull/32652
- 状态/时间: merged / 2026-01-20
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+3/-3，可读 patch 20 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix the fp8_mqa_logits dim mismatch」；模型线: DeepSeek V3.1；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/deepseek_v2.py`, `vllm/utils/deep_gemm.py`；技术摘要: 覆盖「[Bugfix] Fix the fp8_mqa_logits dim mismatch」；主要实现面是 `vllm/model_executor/models/deepseek_v2.py`, `vllm/utils/deep_gemm.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +1/-1 (2 lines); hunks: -686,7 +686,7 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer，涉及 `sparse_attn_indexer`；`vllm/utils/deep_gemm.py` modified +2/-2 (4 lines); hunks: -249,8 +249,8 @@ def fp8_mqa_logits(; symbols: fp8_mqa_logits，涉及 `fp8_mqa_logits`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +1/-1 (2 lines); hunks: -686,7 +686,7 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer
  - `vllm/utils/deep_gemm.py` modified +2/-2 (4 lines); hunks: -249,8 +249,8 @@ def fp8_mqa_logits(; symbols: fp8_mqa_logits
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +1/-1; `vllm/utils/deep_gemm.py` modified +2/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_v2.py`, `vllm/utils/deep_gemm.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #29287 - [ROCm][Deepseekv3.2] Refactor Sparse Indexer as CustomOp

- 链接: https://github.com/vllm-project/vllm/pull/29287
- 状态/时间: merged / 2026-01-21
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+982/-323，可读 patch 1521 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[ROCm][Deepseekv3.2] Refactor Sparse Indexer as CustomOp」；模型线: DeepSeek V3.1；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/sparse_attn_indexer.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`；技术摘要: 覆盖「[ROCm][Deepseekv3.2] Refactor Sparse Indexer as CustomOp」；主要实现面是 `vllm/model_executor/layers/sparse_attn_indexer.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/sparse_attn_indexer.py` added +318/-0 (318 lines); hunks: -0,0 +1,318; symbols: sparse_attn_indexer, sparse_attn_indexer_fake, SparseAttnIndexer, __init__，涉及 `sparse_attn_indexer, sparse_attn_indexer_fake, SparseAttnIndexer`；`vllm/model_executor/models/deepseek_v2.py` modified +14/-233 (247 lines); hunks: -43,7 +43,6; -63,6 +62,7; symbols: get_attn_backend, sparse_attn_indexer, sparse_attn_indexer_fake, Indexer，涉及 `get_attn_backend, sparse_attn_indexer, sparse_attn_indexer_fake`；`vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` modified +518/-80 (598 lines); hunks: -1,100 +1,220; -183,10 +303,38 @@ def rocm_fp8_paged_mqa_logits(; symbols: fp8_mqa_logits_torch, _indexer_k_quant_and_cache_kernel, rocm_fp8_mqa_logits, indexer_k_quant_and_cache_triton，涉及 `fp8_mqa_logits_torch, _indexer_k_quant_and_cache_kernel, rocm_fp8_mqa_logits`；`vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py` modified +110/-10 (120 lines); hunks: -15,6 +15,7; -33,6 +34,48; symbols: fetch_id_to_ragged_kernel, fetch_id_to_ragged_triton, ROCMAiterMLASparseBackend, ROCMAiterMLASparseMetadata，涉及 `fetch_id_to_ragged_kernel, fetch_id_to_ragged_triton, ROCMAiterMLASparseBackend`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/sparse_attn_indexer.py` added +318/-0 (318 lines); hunks: -0,0 +1,318; symbols: sparse_attn_indexer, sparse_attn_indexer_fake, SparseAttnIndexer, __init__
  - `vllm/model_executor/models/deepseek_v2.py` modified +14/-233 (247 lines); hunks: -43,7 +43,6; -63,6 +62,7; symbols: get_attn_backend, sparse_attn_indexer, sparse_attn_indexer_fake, Indexer
  - `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` modified +518/-80 (598 lines); hunks: -1,100 +1,220; -183,10 +303,38 @@ def rocm_fp8_paged_mqa_logits(; symbols: fp8_mqa_logits_torch, _indexer_k_quant_and_cache_kernel, rocm_fp8_mqa_logits, indexer_k_quant_and_cache_triton
  - `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py` modified +110/-10 (120 lines); hunks: -15,6 +15,7; -33,6 +34,48; symbols: fetch_id_to_ragged_kernel, fetch_id_to_ragged_triton, ROCMAiterMLASparseBackend, ROCMAiterMLASparseMetadata
  - `vllm/_aiter_ops.py` modified +12/-0 (12 lines); hunks: -9,6 +9,10; -1091,6 +1095,14 @@ def register_ops_once() -> None:; symbols: register_ops_once
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/layers/sparse_attn_indexer.py` added +318/-0; `vllm/model_executor/models/deepseek_v2.py` modified +14/-233; `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` modified +518/-80; `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py` modified +110/-10; `vllm/_aiter_ops.py` modified +12/-0; `vllm/v1/attention/backends/mla/indexer.py` modified +6/-0
- 验证与风险: runtime 路径改动集中在 `vllm/_aiter_ops.py`, `vllm/config/compilation.py`, `vllm/model_executor/layers/sparse_attn_indexer.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #33063 - [Chore] Update type annotation of `input_ids` in model forward

- 链接: https://github.com/vllm-project/vllm/pull/33063
- 状态/时间: merged / 2026-01-26
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 164 个文件，+243/-241，可读 patch 2158 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Chore] Update type annotation of `input_ids` in model forward」；模型线: DeepSeek V3.1；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/modernbert.py`, `vllm/model_executor/models/gemma3n.py`, `vllm/model_executor/models/gpt2.py`；技术摘要: 覆盖「[Chore] Update type annotation of `input_ids` in model forward」；主要实现面是 `vllm/model_executor/models/modernbert.py`, `vllm/model_executor/models/gemma3n.py`, `vllm/model_executor/models/gpt2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/modernbert.py` modified +4/-5 (9 lines); hunks: -54,12 +54,11 @@ def forward(; symbols: forward, ModernBertAttention，涉及 `forward, ModernBertAttention`；`vllm/model_executor/models/gemma3n.py` modified +4/-4 (8 lines); hunks: -704,7 +704,7 @@ def altup_embed(self, hidden_states_0: torch.Tensor) -> torc...; -887,7 +887,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; symbols: altup_embed, forward, embed_input_ids, fast_prefill_forward，涉及 `altup_embed, forward, embed_input_ids`；`vllm/model_executor/models/gpt2.py` modified +3/-3 (6 lines); hunks: -218,7 +218,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; -298,7 +298,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; symbols: embed_input_ids, forward, load_weights，涉及 `embed_input_ids, forward, load_weights`；`vllm/model_executor/models/internlm2.py` modified +3/-3 (6 lines); hunks: -284,7 +284,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; -350,7 +350,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; symbols: embed_input_ids, forward, __init__，涉及 `embed_input_ids, forward, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/modernbert.py` modified +4/-5 (9 lines); hunks: -54,12 +54,11 @@ def forward(; symbols: forward, ModernBertAttention
  - `vllm/model_executor/models/gemma3n.py` modified +4/-4 (8 lines); hunks: -704,7 +704,7 @@ def altup_embed(self, hidden_states_0: torch.Tensor) -> torc...; -887,7 +887,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; symbols: altup_embed, forward, embed_input_ids, fast_prefill_forward
  - `vllm/model_executor/models/gpt2.py` modified +3/-3 (6 lines); hunks: -218,7 +218,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; -298,7 +298,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; symbols: embed_input_ids, forward, load_weights
  - `vllm/model_executor/models/internlm2.py` modified +3/-3 (6 lines); hunks: -284,7 +284,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; -350,7 +350,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; symbols: embed_input_ids, forward, __init__
  - `vllm/model_executor/models/opt.py` modified +3/-3 (6 lines); hunks: -267,7 +267,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; -316,7 +316,7 @@ def embed_input_ids(self, input_ids: torch.Tensor) -> torch....; symbols: embed_input_ids, forward
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/modernbert.py` modified +4/-5; `vllm/model_executor/models/gemma3n.py` modified +4/-4; `vllm/model_executor/models/gpt2.py` modified +3/-3; `vllm/model_executor/models/internlm2.py` modified +3/-3; `vllm/model_executor/models/opt.py` modified +3/-3; `vllm/model_executor/models/afmoe.py` modified +2/-2
- 验证与风险: diff 自带测试面 `tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_gemma_embedding.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #33018 - [ROCm][Bugfix] Fix ptpc scale load issue for fused shared expert path in deepseek mtp

- 链接: https://github.com/vllm-project/vllm/pull/33018
- 状态/时间: merged / 2026-01-26
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+11/-8，可读 patch 34 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[ROCm][Bugfix] Fix ptpc scale load issue for fused shared expert path in deepseek mtp」；模型线: DeepSeek V3.1；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/deepseek_mtp.py`；技术摘要: 覆盖「[ROCm][Bugfix] Fix ptpc scale load issue for fused shared expert path in deepseek mtp」；主要实现面是 `vllm/model_executor/models/deepseek_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_mtp.py` modified +11/-8 (19 lines); hunks: -316,7 +316,11 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; -329,14 +333,13 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_mtp.py` modified +11/-8 (19 lines); hunks: -316,7 +316,11 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; -329,14 +333,13 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: load_weights
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_mtp.py` modified +11/-8
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_mtp.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #32064 - [5/N][Attention] Finish eliminating `vllm/attention` folder

- 链接: https://github.com/vllm-project/vllm/pull/32064
- 状态/时间: merged / 2026-01-27
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 151 个文件，+585/-527，可读 patch 2850 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[5/N][Attention] Finish eliminating `vllm/attention` folder」；模型线: DeepSeek V3.1；类别: 模型实现调整；主要 diff: `vllm/model_executor/layers/attention/mla_attention.py`, `vllm/model_executor/layers/attention/attention.py`, `vllm/model_executor/layers/attention/__init__.py`；技术摘要: 覆盖「[5/N][Attention] Finish eliminating `vllm/attention` folder」；主要实现面是 `vllm/model_executor/layers/attention/mla_attention.py`, `vllm/model_executor/layers/attention/attention.py`, `vllm/model_executor/layers/attention/__init__.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/attention/mla_attention.py` modified +354/-17 (371 lines); hunks: -191,24 +191,38; -217,11 +231,16; symbols: MLAAttention, takes, does, __init__，涉及 `MLAAttention, takes, does`；`vllm/model_executor/layers/attention/attention.py` renamed +42/-315 (357 lines); hunks: -1,23 +1,22; -33,20 +32,54; symbols: validate_kv_sharing_target, should_load_quant_weights, get_kv_cache_spec, MLAAttention，涉及 `validate_kv_sharing_target, should_load_quant_weights, get_kv_cache_spec`；`vllm/model_executor/layers/attention/__init__.py` modified +26/-0 (26 lines); hunks: -0,0 +1,26；`vllm/model_executor/models/whisper.py` modified +5/-3 (8 lines); hunks: -17,16 +17,18。
- 代码 diff 细节:
  - `vllm/model_executor/layers/attention/mla_attention.py` modified +354/-17 (371 lines); hunks: -191,24 +191,38; -217,11 +231,16; symbols: MLAAttention, takes, does, __init__
  - `vllm/model_executor/layers/attention/attention.py` renamed +42/-315 (357 lines); hunks: -1,23 +1,22; -33,20 +32,54; symbols: validate_kv_sharing_target, should_load_quant_weights, get_kv_cache_spec, MLAAttention
  - `vllm/model_executor/layers/attention/__init__.py` modified +26/-0 (26 lines); hunks: -0,0 +1,26
  - `vllm/model_executor/models/whisper.py` modified +5/-3 (8 lines); hunks: -17,16 +17,18
  - `vllm/model_executor/models/openpangu.py` modified +3/-2 (5 lines); hunks: -29,7 +29,6; -41,7 +40,8
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/layers/attention/mla_attention.py` modified +354/-17; `vllm/model_executor/layers/attention/attention.py` renamed +42/-315; `vllm/model_executor/layers/attention/__init__.py` modified +26/-0; `vllm/model_executor/models/whisper.py` modified +5/-3; `vllm/model_executor/models/openpangu.py` modified +3/-2; `vllm/model_executor/models/apertus.py` modified +2/-2
- 验证与风险: diff 自带测试面 `tests/compile/test_fusion_attn.py`, `tests/compile/test_qk_norm_rope_fusion.py`, `tests/kernels/attention/test_attention.py`, `tests/kernels/attention/test_mha_attn.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #33191 - Add flake8-implicit-str-concat rules to Ruff

- 链接: https://github.com/vllm-project/vllm/pull/33191
- 状态/时间: merged / 2026-01-28
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 16 个文件，+34/-33，可读 patch 201 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add flake8-implicit-str-concat rules to Ruff」；模型线: DeepSeek V3.1；类别: 性能/后端优化；主要 diff: `tests/tool_parsers/test_deepseekv31_tool_parser.py`, `vllm/model_executor/layers/quantization/kernels/scaled_mm/aiter.py`, `vllm/entrypoints/openai/translations/speech_to_text.py`；技术摘要: 覆盖「Add flake8-implicit-str-concat rules to Ruff」；主要实现面是 `tests/tool_parsers/test_deepseekv31_tool_parser.py`, `vllm/model_executor/layers/quantization/kernels/scaled_mm/aiter.py`, `vllm/entrypoints/openai/translations/speech_to_text.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/tool_parsers/test_deepseekv31_tool_parser.py` modified +8/-8 (16 lines); hunks: -24,9 +24,9 @@ def parser(deepseekv31_tokenizer):; -39,11 +39,11 @@ def test_extract_tool_calls_with_tool(parser):; symbols: parser, test_extract_tool_calls_with_tool, test_extract_tool_calls_with_multiple_tools，涉及 `parser, test_extract_tool_calls_with_tool, test_extract_tool_calls_with_multiple_tools`；`vllm/model_executor/layers/quantization/kernels/scaled_mm/aiter.py` modified +4/-4 (8 lines); hunks: -32,8 +32,8 @@ def is_supported(; -97,9 +97,9 @@ def apply_weights(; symbols: is_supported, apply_weights，涉及 `is_supported, apply_weights`；`vllm/entrypoints/openai/translations/speech_to_text.py` modified +2/-2 (4 lines); hunks: -406,8 +406,8 @@ async def _create_speech_to_text(; symbols: _create_speech_to_text，涉及 `_create_speech_to_text`；`tests/entrypoints/openai/tool_parsers/test_llama4_pythonic_tool_parser.py` modified +1/-1 (2 lines); hunks: -193,7 +193,7 @@ def test_no_tool_call(streaming: bool, default_tokenizer: To...; symbols: test_no_tool_call，涉及 `test_no_tool_call`。
- 代码 diff 细节:
  - `tests/tool_parsers/test_deepseekv31_tool_parser.py` modified +8/-8 (16 lines); hunks: -24,9 +24,9 @@ def parser(deepseekv31_tokenizer):; -39,11 +39,11 @@ def test_extract_tool_calls_with_tool(parser):; symbols: parser, test_extract_tool_calls_with_tool, test_extract_tool_calls_with_multiple_tools
  - `vllm/model_executor/layers/quantization/kernels/scaled_mm/aiter.py` modified +4/-4 (8 lines); hunks: -32,8 +32,8 @@ def is_supported(; -97,9 +97,9 @@ def apply_weights(; symbols: is_supported, apply_weights
  - `vllm/entrypoints/openai/translations/speech_to_text.py` modified +2/-2 (4 lines); hunks: -406,8 +406,8 @@ async def _create_speech_to_text(; symbols: _create_speech_to_text
  - `tests/entrypoints/openai/tool_parsers/test_llama4_pythonic_tool_parser.py` modified +1/-1 (2 lines); hunks: -193,7 +193,7 @@ def test_no_tool_call(streaming: bool, default_tokenizer: To...; symbols: test_no_tool_call
  - `vllm/reasoning/olmo3_reasoning_parser.py` modified +1/-1 (2 lines); hunks: -234,7 +234,7 @@ def __init__(self, tokenizer: "TokenizerLike", *args, **kwar...; symbols: __init__
- 关键代码摘录:

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

- 已读文件:
  - tests: `tests/tool_parsers/test_deepseekv31_tool_parser.py` modified +8/-8; `tests/entrypoints/openai/tool_parsers/test_llama4_pythonic_tool_parser.py` modified +1/-1
  - runtime: `vllm/model_executor/layers/quantization/kernels/scaled_mm/aiter.py` modified +4/-4; `vllm/entrypoints/openai/translations/speech_to_text.py` modified +2/-2; `vllm/reasoning/olmo3_reasoning_parser.py` modified +1/-1; `vllm/compilation/wrapper.py` modified +2/-2
  - docs: `examples/others/lmcache/disagg_prefill_lmcache_v1/disagg_proxy_server.py` modified +4/-4
  - other: `csrc/quantization/machete/generate.py` modified +3/-3
- 验证与风险: diff 自带测试面 `tests/entrypoints/openai/tool_parsers/test_llama4_pythonic_tool_parser.py`, `tests/tool_parsers/test_deepseekv31_tool_parser.py`, `tests/utils.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #33174 - Add support for Mistral Large 3 inference with Flashinfer MoE

- 链接: https://github.com/vllm-project/vllm/pull/33174
- 状态/时间: merged / 2026-01-31
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 16 个文件，+1104/-31，可读 patch 1278 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add support for Mistral Large 3 inference with Flashinfer MoE」；模型线: DeepSeek V3.1；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200,dtype=fp8_w8a8.json`, `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200.json`, `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_GB200,dtype=fp8_w8a8.json`；技术摘要: 覆盖「Add support for Mistral Large 3 inference with Flashinfer MoE」；主要实现面是 `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200,dtype=fp8_w8a8.json`, `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200.json`, `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_GB200,dtype=fp8_w8a8.json`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200,dtype=fp8_w8a8.json` added +147/-0 (147 lines); hunks: -0,0 +1,147；`vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200.json` added +147/-0 (147 lines); hunks: -0,0 +1,147；`vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_GB200,dtype=fp8_w8a8.json` added +147/-0 (147 lines); hunks: -0,0 +1,147；`vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json` added +147/-0 (147 lines); hunks: -0,0 +1,147。
- 代码 diff 细节:
  - `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200,dtype=fp8_w8a8.json` added +147/-0 (147 lines); hunks: -0,0 +1,147
  - `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200.json` added +147/-0 (147 lines); hunks: -0,0 +1,147
  - `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_GB200,dtype=fp8_w8a8.json` added +147/-0 (147 lines); hunks: -0,0 +1,147
  - `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json` added +147/-0 (147 lines); hunks: -0,0 +1,147
  - `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_H200.json` added +147/-0 (147 lines); hunks: -0,0 +1,147
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200,dtype=fp8_w8a8.json` added +147/-0; `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200.json` added +147/-0; `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_GB200,dtype=fp8_w8a8.json` added +147/-0; `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json` added +147/-0; `vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_H200.json` added +147/-0; `vllm/model_executor/layers/fused_moe/configs/E=16,N=4096,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json` added +147/-0
- 验证与风险: diff 自带测试面 `tests/kernels/moe/test_flashinfer.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #33858 - [Bugfix] Kimi-K2 grouped_topk usage for Flashinfer monolithic kernels.

- 链接: https://github.com/vllm-project/vllm/pull/33858
- 状态/时间: merged / 2026-02-05
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-11，可读 patch 28 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Kimi-K2 grouped_topk usage for Flashinfer monolithic kernels.」；模型线: DeepSeek V3.1；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/deepseek_v2.py`；技术摘要: 覆盖「[Bugfix] Kimi-K2 grouped_topk usage for Flashinfer monolithic kernels.」；主要实现面是 `vllm/model_executor/models/deepseek_v2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +3/-11 (14 lines); hunks: -295,14 +295,6 @@ def __init__(; -313,9 +305,9 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +3/-11 (14 lines); hunks: -295,14 +295,6 @@ def __init__(; -313,9 +305,9 @@ def __init__(; symbols: __init__
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +3/-11
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #33876 - [Bugfix] Fix Kimi-K2.5 NVFP4 checkpoints weight loading

- 链接: https://github.com/vllm-project/vllm/pull/33876
- 状态/时间: merged / 2026-02-05
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+15/-5，可读 patch 53 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix Kimi-K2.5 NVFP4 checkpoints weight loading」；模型线: DeepSeek V3.1；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/kimi_k25.py`, `vllm/model_executor/models/deepseek_v2.py`；技术摘要: 覆盖「[Bugfix] Fix Kimi-K2.5 NVFP4 checkpoints weight loading」；主要实现面是 `vllm/model_executor/models/kimi_k25.py`, `vllm/model_executor/models/deepseek_v2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/kimi_k25.py` modified +14/-4 (18 lines); hunks: -24,7 +24,11; -302,7 +306,9 @@ def split_video_chunks(self, video):; symbols: split_video_chunks, KimiK25ForConditionalGeneration, compute_logits，涉及 `split_video_chunks, KimiK25ForConditionalGeneration, compute_logits`；`vllm/model_executor/models/deepseek_v2.py` modified +1/-1 (2 lines); hunks: -1493,7 +1493,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/models/kimi_k25.py` modified +14/-4 (18 lines); hunks: -24,7 +24,11; -302,7 +306,9 @@ def split_video_chunks(self, video):; symbols: split_video_chunks, KimiK25ForConditionalGeneration, compute_logits
  - `vllm/model_executor/models/deepseek_v2.py` modified +1/-1 (2 lines); hunks: -1493,7 +1493,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: load_weights
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/kimi_k25.py` modified +14/-4; `vllm/model_executor/models/deepseek_v2.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/kimi_k25.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #34124 - [Model] GLM adaptation

- 链接: https://github.com/vllm-project/vllm/pull/34124
- 状态/时间: merged / 2026-02-09
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+13/-3，可读 patch 72 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] GLM adaptation」；模型线: DeepSeek V3.1；类别: 文档/测试/CI；主要 diff: `vllm/model_executor/models/deepseek_v2.py`, `tests/models/registry.py`, `tests/models/test_initialization.py`；技术摘要: 覆盖「[Model] GLM adaptation」；主要实现面是 `vllm/model_executor/models/deepseek_v2.py`, `tests/models/registry.py`, `tests/models/test_initialization.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +5/-1 (6 lines); hunks: -836,7 +836,7 @@ def __init__(; -1499,6 +1499,10 @@ class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM):; symbols: __init__, DeepseekV3ForCausalLM, GlmMoeDsaForCausalLM, get_spec_layer_idx_from_weight_name，涉及 `__init__, DeepseekV3ForCausalLM, GlmMoeDsaForCausalLM`；`tests/models/registry.py` modified +3/-0 (3 lines); hunks: -275,6 +275,9 @@ def check_available_online(; symbols: check_available_online，涉及 `check_available_online`；`tests/models/test_initialization.py` modified +1/-1 (2 lines); hunks: -97,7 +97,7 @@ def _initialize_kv_caches_v1(self, vllm_config):; symbols: _initialize_kv_caches_v1，涉及 `_initialize_kv_caches_v1`；`vllm/model_executor/models/registry.py` modified +1/-0 (1 lines); hunks: -114,6 +114,7。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +5/-1 (6 lines); hunks: -836,7 +836,7 @@ def __init__(; -1499,6 +1499,10 @@ class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM):; symbols: __init__, DeepseekV3ForCausalLM, GlmMoeDsaForCausalLM, get_spec_layer_idx_from_weight_name
  - `tests/models/registry.py` modified +3/-0 (3 lines); hunks: -275,6 +275,9 @@ def check_available_online(; symbols: check_available_online
  - `tests/models/test_initialization.py` modified +1/-1 (2 lines); hunks: -97,7 +97,7 @@ def _initialize_kv_caches_v1(self, vllm_config):; symbols: _initialize_kv_caches_v1
  - `vllm/model_executor/models/registry.py` modified +1/-0 (1 lines); hunks: -114,6 +114,7
  - `vllm/config/speculative.py` modified +1/-1 (2 lines); hunks: -181,7 +181,7 @@ def compute_hash(self) -> str:; symbols: compute_hash, hf_config_override
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +5/-1; `vllm/model_executor/models/registry.py` modified +1/-0; `vllm/config/speculative.py` modified +1/-1; `vllm/transformers_utils/model_arch_config_convertor.py` modified +1/-0
  - tests: `tests/models/registry.py` modified +3/-0; `tests/models/test_initialization.py` modified +1/-1
  - other: `benchmarks/kernels/benchmark_moe.py` modified +1/-0
- 验证与风险: diff 自带测试面 `tests/models/registry.py`, `tests/models/test_initialization.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #34353 - [Bugfix] fix default is_neox_style to be True for deepseekv3.2

- 链接: https://github.com/vllm-project/vllm/pull/34353
- 状态/时间: merged / 2026-02-11
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] fix default is_neox_style to be True for deepseekv3.2」；模型线: DeepSeek V3.1；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/deepseek_v2.py`；技术摘要: 覆盖「[Bugfix] fix default is_neox_style to be True for deepseekv3.2」；主要实现面是 `vllm/model_executor/models/deepseek_v2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +1/-1 (2 lines); hunks: -836,7 +836,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +1/-1 (2 lines); hunks: -836,7 +836,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -836,7 +836,7 @@ def __init__(
-                is_neox_style=not getattr(config, "indexer_rope_interleave", True),
+                is_neox_style=not getattr(config, "indexer_rope_interleave", False),
```

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #34514 - [CI][BugFix] ShellCheck cleanup to remove baseline and preserve runtime behavior

- 链接: https://github.com/vllm-project/vllm/pull/34514
- 状态/时间: merged / 2026-02-17
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 55 个文件，+338/-464，可读 patch 2137 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CI][BugFix] ShellCheck cleanup to remove baseline and preserve runtime behavior」；模型线: DeepSeek V3.1；类别: 缺陷修复；主要 diff: `tests/v1/ec_connector/integration/run_epd_correctness_test.sh`, `tools/pre_commit/shellcheck.baseline`, `benchmarks/auto_tune/auto_tune.sh`；技术摘要: 覆盖「[CI][BugFix] ShellCheck cleanup to remove baseline and preserve runtime behavior」；主要实现面是 `tests/v1/ec_connector/integration/run_epd_correctness_test.sh`, `tools/pre_commit/shellcheck.baseline`, `benchmarks/auto_tune/auto_tune.sh`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/v1/ec_connector/integration/run_epd_correctness_test.sh` modified +54/-54 (108 lines); hunks: -24,7 +24,7 @@ MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"; -51,7 +51,7 @@ LOG_PATH="${LOG_PATH:-/tmp}"；`tools/pre_commit/shellcheck.baseline` removed +0/-89 (89 lines); hunks: -1,89 +0,0；`benchmarks/auto_tune/auto_tune.sh` modified +22/-22 (44 lines); hunks: -46,10 +46,10 @@ echo "VLLM_LOGGING_LEVEL=$VLLM_LOGGING_LEVEL"; -114,7 +114,7 @@ start_server() {；`tools/pre_commit/shellcheck.sh` modified +3/-36 (39 lines); hunks: -2,7 +2,6; -20,38 +19,6 @@ if ! [ -x "$(command -v shellcheck)" ]; then。
- 代码 diff 细节:
  - `tests/v1/ec_connector/integration/run_epd_correctness_test.sh` modified +54/-54 (108 lines); hunks: -24,7 +24,7 @@ MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"; -51,7 +51,7 @@ LOG_PATH="${LOG_PATH:-/tmp}"
  - `tools/pre_commit/shellcheck.baseline` removed +0/-89 (89 lines); hunks: -1,89 +0,0
  - `benchmarks/auto_tune/auto_tune.sh` modified +22/-22 (44 lines); hunks: -46,10 +46,10 @@ echo "VLLM_LOGGING_LEVEL=$VLLM_LOGGING_LEVEL"; -114,7 +114,7 @@ start_server() {
  - `tools/pre_commit/shellcheck.sh` modified +3/-36 (39 lines); hunks: -2,7 +2,6; -20,38 +19,6 @@ if ! [ -x "$(command -v shellcheck)" ]; then
  - `.buildkite/scripts/hardware_ci/run-npu-test.sh` modified +15/-20 (35 lines); hunks: -41,16 +41,16 @@ get_config() {; -62,14 +62,14 @@ agent_idx=$(echo "${BUILDKITE_AGENT_NAME}" | awk -F'-' '{pri...
- 关键代码摘录:

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

- 已读文件:
  - tests: `tests/v1/ec_connector/integration/run_epd_correctness_test.sh` modified +54/-54
  - other: `tools/pre_commit/shellcheck.baseline` removed +0/-89; `benchmarks/auto_tune/auto_tune.sh` modified +22/-22; `tools/pre_commit/shellcheck.sh` modified +3/-36; `.buildkite/scripts/hardware_ci/run-npu-test.sh` modified +15/-20; `benchmarks/run_structured_output_benchmark.sh` modified +16/-14
  - docs: `examples/online_serving/disaggregated_encoder/disagg_1e1p1d_example.sh` modified +15/-15; `examples/online_serving/disaggregated_encoder/disagg_1e1pd_example.sh` modified +13/-13
- 验证与风险: diff 自带测试面 `.buildkite/scripts/scheduled_integration_test/deepseek_v2_lite_ep_eplb.sh`, `.buildkite/scripts/scheduled_integration_test/qwen30b_a3b_fp8_block_ep_eplb.sh`, `.buildkite/scripts/scheduled_integration_test/qwen3_next_mtp_async_eplb.sh`, `tests/standalone_tests/python_only_compile.sh`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #34758 - [Model Bash] DeepSeek R1 BF16 Min Latency QKV A GEMM (0.5% E2E Speedup)

- 链接: https://github.com/vllm-project/vllm/pull/34758
- 状态/时间: merged / 2026-02-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+855/-3，可读 patch 917 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model Bash] DeepSeek R1 BF16 Min Latency QKV A GEMM (0.5% E2E Speedup)」；模型线: DeepSeek V3.1；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/layers/mla.py`, `csrc/dsv3_fused_a_gemm.cu`；技术摘要: 覆盖「[Model Bash] DeepSeek R1 BF16 Min Latency QKV A GEMM (0.5% E2E Speedup)」；主要实现面是 `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/layers/mla.py`, `csrc/dsv3_fused_a_gemm.cu`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +60/-3 (63 lines); hunks: -32,6 +32,7; -711,6 +712,64 @@ def forward(; symbols: forward, DeepSeekV2FusedQkvAProj, __init__, DeepseekV2MLAAttention，涉及 `forward, DeepSeekV2FusedQkvAProj, __init__`；`vllm/model_executor/layers/mla.py` modified +1/-0 (1 lines); hunks: -129,6 +129,7 @@ def forward(; symbols: forward，涉及 `forward`；`csrc/dsv3_fused_a_gemm.cu` added +747/-0 (747 lines); hunks: -0,0 +1,747; symbols: Type，涉及 `Type`；`CMakeLists.txt` modified +19/-0 (19 lines); hunks: -771,6 +771,25 @@ if(VLLM_GPU_LANG STREQUAL "CUDA")。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +60/-3 (63 lines); hunks: -32,6 +32,7; -711,6 +712,64 @@ def forward(; symbols: forward, DeepSeekV2FusedQkvAProj, __init__, DeepseekV2MLAAttention
  - `vllm/model_executor/layers/mla.py` modified +1/-0 (1 lines); hunks: -129,6 +129,7 @@ def forward(; symbols: forward
  - `csrc/dsv3_fused_a_gemm.cu` added +747/-0 (747 lines); hunks: -0,0 +1,747; symbols: Type
  - `CMakeLists.txt` modified +19/-0 (19 lines); hunks: -771,6 +771,25 @@ if(VLLM_GPU_LANG STREQUAL "CUDA")
  - `vllm/_custom_ops.py` modified +18/-0 (18 lines); hunks: -2770,6 +2770,24 @@ def sm100_cutlass_mla_get_workspace_size(; symbols: sm100_cutlass_mla_get_workspace_size, dsv3_fused_a_gemm
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +60/-3; `vllm/model_executor/layers/mla.py` modified +1/-0; `vllm/_custom_ops.py` modified +18/-0
  - other: `csrc/dsv3_fused_a_gemm.cu` added +747/-0; `CMakeLists.txt` modified +19/-0; `csrc/ops.h` modified +5/-0; `csrc/torch_bindings.cpp` modified +5/-0
- 验证与风险: runtime 路径改动集中在 `vllm/_custom_ops.py`, `vllm/model_executor/layers/mla.py`, `vllm/model_executor/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #34876 - [Bug] Fix DeepSeek V3 weight loading caused by incorrect prefix

- 链接: https://github.com/vllm-project/vllm/pull/34876
- 状态/时间: merged / 2026-02-19
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-2，可读 patch 18 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bug] Fix DeepSeek V3 weight loading caused by incorrect prefix」；模型线: DeepSeek V3.1；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/deepseek_v2.py`；技术摘要: 覆盖「[Bug] Fix DeepSeek V3 weight loading caused by incorrect prefix」；主要实现面是 `vllm/model_executor/models/deepseek_v2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +2/-2 (4 lines); hunks: -716,7 +716,7 @@ class DeepSeekV2FusedQkvAProj(MergedColumnParallelLinear):; -726,7 +726,7 @@ def __init__(; symbols: DeepSeekV2FusedQkvAProj, __init__，涉及 `DeepSeekV2FusedQkvAProj, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +2/-2 (4 lines); hunks: -716,7 +716,7 @@ class DeepSeekV2FusedQkvAProj(MergedColumnParallelLinear):; -726,7 +726,7 @@ def __init__(; symbols: DeepSeekV2FusedQkvAProj, __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -716,7 +716,7 @@ class DeepSeekV2FusedQkvAProj(MergedColumnParallelLinear):
-        output_size: int,
+        output_size: list[int],
@@ -726,7 +726,7 @@ def __init__(
-            prefix=f"{prefix}.kv_a_proj_with_mqa",
+            prefix=prefix,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +2/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #34302 - [ModelBash][DSV3] Add TRTLLM DSV3 Router GEMM kernel (6% B1 Speedup)

- 链接: https://github.com/vllm-project/vllm/pull/34302
- 状态/时间: merged / 2026-02-23
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+915/-3，可读 patch 971 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[ModelBash][DSV3] Add TRTLLM DSV3 Router GEMM kernel (6% B1 Speedup)」；模型线: DeepSeek V3.1；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/deepseek_v2.py`, `csrc/moe/dsv3_router_gemm_bf16_out.cu`, `csrc/moe/dsv3_router_gemm_float_out.cu`；技术摘要: 覆盖「[ModelBash][DSV3] Add TRTLLM DSV3 Router GEMM kernel (6% B1 Speedup)」；主要实现面是 `vllm/model_executor/models/deepseek_v2.py`, `csrc/moe/dsv3_router_gemm_bf16_out.cu`, `csrc/moe/dsv3_router_gemm_float_out.cu`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +75/-2 (77 lines); hunks: -221,6 +221,73 @@ def forward(self, x):; -249,10 +316,9 @@ def __init__(; symbols: forward, DeepSeekV2Gate, __init__, set_out_dtype，涉及 `forward, DeepSeekV2Gate, __init__`；`csrc/moe/dsv3_router_gemm_bf16_out.cu` added +291/-0 (291 lines); hunks: -0,0 +1,291；`csrc/moe/dsv3_router_gemm_float_out.cu` added +291/-0 (291 lines); hunks: -0,0 +1,291；`csrc/moe/dsv3_router_gemm_entry.cu` added +163/-0 (163 lines); hunks: -0,0 +1,163。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +75/-2 (77 lines); hunks: -221,6 +221,73 @@ def forward(self, x):; -249,10 +316,9 @@ def __init__(; symbols: forward, DeepSeekV2Gate, __init__, set_out_dtype
  - `csrc/moe/dsv3_router_gemm_bf16_out.cu` added +291/-0 (291 lines); hunks: -0,0 +1,291
  - `csrc/moe/dsv3_router_gemm_float_out.cu` added +291/-0 (291 lines); hunks: -0,0 +1,291
  - `csrc/moe/dsv3_router_gemm_entry.cu` added +163/-0 (163 lines); hunks: -0,0 +1,163
  - `csrc/moe/dsv3_router_gemm_utils.h` added +43/-0 (43 lines); hunks: -0,0 +1,43
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +75/-2; `vllm/_custom_ops.py` modified +15/-0
  - other: `csrc/moe/dsv3_router_gemm_bf16_out.cu` added +291/-0; `csrc/moe/dsv3_router_gemm_float_out.cu` added +291/-0; `csrc/moe/dsv3_router_gemm_entry.cu` added +163/-0; `csrc/moe/dsv3_router_gemm_utils.h` added +43/-0; `CMakeLists.txt` modified +21/-0; `csrc/moe/moe_ops.h` modified +12/-1
- 验证与风险: runtime 路径改动集中在 `vllm/_custom_ops.py`, `vllm/model_executor/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #33724 - [WideEP] Remove pplx all2all backend

- 链接: https://github.com/vllm-project/vllm/pull/33724
- 状态/时间: merged / 2026-02-26
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 39 个文件，+107/-2069，可读 patch 2692 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[WideEP] Remove pplx all2all backend」；模型线: DeepSeek V3.1；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/fused_moe/pplx_prepare_finalize.py`, `vllm/model_executor/layers/fused_moe/all2all_utils.py`, `vllm/model_executor/layers/fused_moe/config.py`；技术摘要: 覆盖「[WideEP] Remove pplx all2all backend」；主要实现面是 `vllm/model_executor/layers/fused_moe/pplx_prepare_finalize.py`, `vllm/model_executor/layers/fused_moe/all2all_utils.py`, `vllm/model_executor/layers/fused_moe/config.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/fused_moe/pplx_prepare_finalize.py` removed +0/-373 (373 lines); hunks: -1,373 +0,0; symbols: pplx_hidden_dim_scale_bytes, PplxPrepareAndFinalize, __init__, activation_format，涉及 `pplx_hidden_dim_scale_bytes, PplxPrepareAndFinalize, __init__`；`vllm/model_executor/layers/fused_moe/all2all_utils.py` modified +4/-49 (53 lines); hunks: -1,6 +1,7; -24,16 +25,11; symbols: maybe_make_prepare_finalize，涉及 `maybe_make_prepare_finalize`；`vllm/model_executor/layers/fused_moe/config.py` modified +1/-9 (10 lines); hunks: -939,10 +939,6 @@ def is_sequence_parallel(self) -> bool:; -962,7 +958,7 @@ def use_fi_all2allv_kernels(self):; symbols: is_sequence_parallel, use_all2all_kernels, use_pplx_kernels, use_deepep_ht_kernels，涉及 `is_sequence_parallel, use_all2all_kernels, use_pplx_kernels`；`vllm/model_executor/layers/fused_moe/topk_weight_and_reduce.py` modified +5/-4 (9 lines); hunks: -14,10 +14,11 @@ class TopKWeightAndReduceDelegate(mk.TopKWeightAndReduce):; symbols: TopKWeightAndReduceDelegate，涉及 `TopKWeightAndReduceDelegate`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/fused_moe/pplx_prepare_finalize.py` removed +0/-373 (373 lines); hunks: -1,373 +0,0; symbols: pplx_hidden_dim_scale_bytes, PplxPrepareAndFinalize, __init__, activation_format
  - `vllm/model_executor/layers/fused_moe/all2all_utils.py` modified +4/-49 (53 lines); hunks: -1,6 +1,7; -24,16 +25,11; symbols: maybe_make_prepare_finalize
  - `vllm/model_executor/layers/fused_moe/config.py` modified +1/-9 (10 lines); hunks: -939,10 +939,6 @@ def is_sequence_parallel(self) -> bool:; -962,7 +958,7 @@ def use_fi_all2allv_kernels(self):; symbols: is_sequence_parallel, use_all2all_kernels, use_pplx_kernels, use_deepep_ht_kernels
  - `vllm/model_executor/layers/fused_moe/topk_weight_and_reduce.py` modified +5/-4 (9 lines); hunks: -14,10 +14,11 @@ class TopKWeightAndReduceDelegate(mk.TopKWeightAndReduce):; symbols: TopKWeightAndReduceDelegate
  - `vllm/model_executor/layers/fused_moe/fused_batched_moe.py` modified +3/-3 (6 lines); hunks: -493,7 +493,7 @@ class BatchedPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):; -648,7 +648,7 @@ def finalize(; symbols: BatchedPrepareAndFinalize, that, __init__, finalize
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/layers/fused_moe/pplx_prepare_finalize.py` removed +0/-373; `vllm/model_executor/layers/fused_moe/all2all_utils.py` modified +4/-49; `vllm/model_executor/layers/fused_moe/config.py` modified +1/-9; `vllm/model_executor/layers/fused_moe/topk_weight_and_reduce.py` modified +5/-4; `vllm/model_executor/layers/fused_moe/fused_batched_moe.py` modified +3/-3; `vllm/model_executor/layers/fused_moe/modular_kernel.py` modified +3/-3
- 验证与风险: diff 自带测试面 `tests/kernels/moe/modular_kernel_tools/common.py`, `tests/kernels/moe/modular_kernel_tools/mk_objects.py`, `tests/kernels/moe/modular_kernel_tools/profile_modular_kernel.py`, `tests/kernels/moe/test_modular_kernel_combinations.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #35121 - [Performance] Cublas Bf16 Gate with Fp32 Output

- 链接: https://github.com/vllm-project/vllm/pull/35121
- 状态/时间: merged / 2026-02-27
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+206/-80，可读 patch 390 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Performance] Cublas Bf16 Gate with Fp32 Output」；模型线: DeepSeek V3.1；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/fused_moe/router/gate_linear.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/nemotron_h.py`；技术摘要: 覆盖「[Performance] Cublas Bf16 Gate with Fp32 Output」；主要实现面是 `vllm/model_executor/layers/fused_moe/router/gate_linear.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/nemotron_h.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/fused_moe/router/gate_linear.py` added +117/-0 (117 lines); hunks: -0,0 +1,117; symbols: GateLinear, __init__, set_out_dtype, forward，涉及 `GateLinear, __init__, set_out_dtype`；`vllm/model_executor/models/deepseek_v2.py` modified +2/-70 (72 lines); hunks: -47,7 +47,7; -221,73 +221,6 @@ def forward(self, x):; symbols: forward, DeepSeekV2Gate, __init__, set_out_dtype，涉及 `forward, DeepSeekV2Gate, __init__`；`vllm/model_executor/models/nemotron_h.py` modified +6/-9 (15 lines); hunks: -34,7 +34,7; -148,13 +148,11 @@ def __init__(; symbols: __init__, forward, _get_max_n_routed_experts, get_expert_mapping，涉及 `__init__, forward, _get_max_n_routed_experts`；`vllm/model_executor/layers/fused_moe/__init__.py` modified +2/-0 (2 lines); hunks: -28,6 +28,7; -64,6 +65,7 @@ def get_config() -> dict[str, Any] | None:; symbols: get_config，涉及 `get_config`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/fused_moe/router/gate_linear.py` added +117/-0 (117 lines); hunks: -0,0 +1,117; symbols: GateLinear, __init__, set_out_dtype, forward
  - `vllm/model_executor/models/deepseek_v2.py` modified +2/-70 (72 lines); hunks: -47,7 +47,7; -221,73 +221,6 @@ def forward(self, x):; symbols: forward, DeepSeekV2Gate, __init__, set_out_dtype
  - `vllm/model_executor/models/nemotron_h.py` modified +6/-9 (15 lines); hunks: -34,7 +34,7; -148,13 +148,11 @@ def __init__(; symbols: __init__, forward, _get_max_n_routed_experts, get_expert_mapping
  - `vllm/model_executor/layers/fused_moe/__init__.py` modified +2/-0 (2 lines); hunks: -28,6 +28,7; -64,6 +65,7 @@ def get_config() -> dict[str, Any] | None:; symbols: get_config
  - `csrc/moe/router_gemm.cu` added +52/-0 (52 lines); hunks: -0,0 +1,52
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/layers/fused_moe/router/gate_linear.py` added +117/-0; `vllm/model_executor/models/deepseek_v2.py` modified +2/-70; `vllm/model_executor/models/nemotron_h.py` modified +6/-9; `vllm/model_executor/layers/fused_moe/__init__.py` modified +2/-0; `vllm/_custom_ops.py` modified +17/-0
  - other: `csrc/moe/router_gemm.cu` added +52/-0; `csrc/moe/moe_ops.h` modified +4/-0; `csrc/moe/torch_bindings.cpp` modified +4/-0
- 验证与风险: runtime 路径改动集中在 `vllm/_custom_ops.py`, `vllm/model_executor/layers/fused_moe/__init__.py`, `vllm/model_executor/layers/fused_moe/router/gate_linear.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #35548 - [MTP] Validate that MTP weights are actually loaded

- 链接: https://github.com/vllm-project/vllm/pull/35548
- 状态/时间: merged / 2026-02-28
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+20/-0，可读 patch 27 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MTP] Validate that MTP weights are actually loaded」；模型线: DeepSeek V3.1；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/deepseek_mtp.py`；技术摘要: 覆盖「[MTP] Validate that MTP weights are actually loaded」；主要实现面是 `vllm/model_executor/models/deepseek_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_mtp.py` modified +20/-0 (20 lines); hunks: -415,6 +415,26 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: load_weights, _rewrite_spec_layer_name，涉及 `load_weights, _rewrite_spec_layer_name`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_mtp.py` modified +20/-0 (20 lines); hunks: -415,6 +415,26 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: load_weights, _rewrite_spec_layer_name
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_mtp.py` modified +20/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_mtp.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #35751 - [MoE][Perf] Wrap DSV3 QKVAProj GEMM in custom op for torch.compile

- 链接: https://github.com/vllm-project/vllm/pull/35751
- 状态/时间: merged / 2026-03-02
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+41/-13，可读 patch 75 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MoE][Perf] Wrap DSV3 QKVAProj GEMM in custom op for torch.compile」；模型线: DeepSeek V3.1；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/deepseek_v2.py`；技术摘要: 覆盖「[MoE][Perf] Wrap DSV3 QKVAProj GEMM in custom op for torch.compile」；主要实现面是 `vllm/model_executor/models/deepseek_v2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +41/-13 (54 lines); hunks: -75,6 +75,7; -717,6 +718,44 @@ def forward(; symbols: forward, _min_latency_fused_qkv_a_proj_impl, _min_latency_fused_qkv_a_proj_fake, DeepSeekV2FusedQkvAProj，涉及 `forward, _min_latency_fused_qkv_a_proj_impl, _min_latency_fused_qkv_a_proj_fake`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +41/-13 (54 lines); hunks: -75,6 +75,7; -717,6 +718,44 @@ def forward(; symbols: forward, _min_latency_fused_qkv_a_proj_impl, _min_latency_fused_qkv_a_proj_fake, DeepSeekV2FusedQkvAProj
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +41/-13
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #36247 - [Bugfix] Fix compressed-tensors quantization failure for DeepSeek-R1 on MI300x

- 链接: https://github.com/vllm-project/vllm/pull/36247
- 状态/时间: merged / 2026-03-07
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-2，可读 patch 18 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix compressed-tensors quantization failure for DeepSeek-R1 on MI300x」；模型线: DeepSeek V3.1；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/deepseek_v2.py`；技术摘要: 覆盖「[Bugfix] Fix compressed-tensors quantization failure for DeepSeek-R1 on MI300x」；主要实现面是 `vllm/model_executor/models/deepseek_v2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +2/-2 (4 lines); hunks: -756,7 +756,7 @@ def _min_latency_fused_qkv_a_proj_fake(; -848,7 +848,7 @@ def __init__(; symbols: _min_latency_fused_qkv_a_proj_fake, DeepSeekV2FusedQkvAProj, DeepSeekV2FusedQkvAProjLinear, __init__，涉及 `_min_latency_fused_qkv_a_proj_fake, DeepSeekV2FusedQkvAProj, DeepSeekV2FusedQkvAProjLinear`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +2/-2 (4 lines); hunks: -756,7 +756,7 @@ def _min_latency_fused_qkv_a_proj_fake(; -848,7 +848,7 @@ def __init__(; symbols: _min_latency_fused_qkv_a_proj_fake, DeepSeekV2FusedQkvAProj, DeepSeekV2FusedQkvAProjLinear, __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -756,7 +756,7 @@ def _min_latency_fused_qkv_a_proj_fake(
-class DeepSeekV2FusedQkvAProj(MergedColumnParallelLinear):
+class DeepSeekV2FusedQkvAProjLinear(MergedColumnParallelLinear):
@@ -848,7 +848,7 @@ def __init__(
-            self.fused_qkv_a_proj = DeepSeekV2FusedQkvAProj(
+            self.fused_qkv_a_proj = DeepSeekV2FusedQkvAProjLinear(
```

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +2/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #36361 - Kimi k2.5 MLA based eagle3

- 链接: https://github.com/vllm-project/vllm/pull/36361
- 状态/时间: merged / 2026-03-11
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+499/-8，可读 patch 649 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Kimi k2.5 MLA based eagle3」；模型线: DeepSeek V3.1；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/deepseek_eagle3.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/kimi_k25.py`；技术摘要: 覆盖「Kimi k2.5 MLA based eagle3」；主要实现面是 `vllm/model_executor/models/deepseek_eagle3.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/kimi_k25.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_eagle3.py` added +419/-0 (419 lines); hunks: -0,0 +1,419; symbols: DeepseekV2Eagle3DecoderLayer, __init__, _norm_before_residual, _norm_after_residual，涉及 `DeepseekV2Eagle3DecoderLayer, __init__, _norm_before_residual`；`vllm/model_executor/models/deepseek_v2.py` modified +39/-6 (45 lines); hunks: -82,7 +82,13; -828,6 +834,7 @@ def __init__(; symbols: __init__, embed_input_ids，涉及 `__init__, embed_input_ids`；`vllm/model_executor/models/kimi_k25.py` modified +14/-1 (15 lines); hunks: -28,6 +28,8; -311,7 +313,12 @@ def split_video_chunks(self, video):; symbols: split_video_chunks, KimiK25ForConditionalGeneration, compute_logits, set_aux_hidden_state_layers，涉及 `split_video_chunks, KimiK25ForConditionalGeneration, compute_logits`；`tests/models/registry.py` modified +12/-0 (12 lines); hunks: -1137,6 +1137,18 @@ def check_available_online(; symbols: check_available_online，涉及 `check_available_online`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_eagle3.py` added +419/-0 (419 lines); hunks: -0,0 +1,419; symbols: DeepseekV2Eagle3DecoderLayer, __init__, _norm_before_residual, _norm_after_residual
  - `vllm/model_executor/models/deepseek_v2.py` modified +39/-6 (45 lines); hunks: -82,7 +82,13; -828,6 +834,7 @@ def __init__(; symbols: __init__, embed_input_ids
  - `vllm/model_executor/models/kimi_k25.py` modified +14/-1 (15 lines); hunks: -28,6 +28,8; -311,7 +313,12 @@ def split_video_chunks(self, video):; symbols: split_video_chunks, KimiK25ForConditionalGeneration, compute_logits, set_aux_hidden_state_layers
  - `tests/models/registry.py` modified +12/-0 (12 lines); hunks: -1137,6 +1137,18 @@ def check_available_online(; symbols: check_available_online
  - `vllm/model_executor/models/registry.py` modified +2/-0 (2 lines); hunks: -551,6 +551,8
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_eagle3.py` added +419/-0; `vllm/model_executor/models/deepseek_v2.py` modified +39/-6; `vllm/model_executor/models/kimi_k25.py` modified +14/-1; `vllm/model_executor/models/registry.py` modified +2/-0; `vllm/v1/spec_decode/eagle.py` modified +8/-1; `vllm/config/speculative.py` modified +4/-0
  - tests: `tests/models/registry.py` modified +12/-0
- 验证与风险: diff 自带测试面 `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #36931 - [Feat][Bugfix] Enable additional dimension for Flashinfer MLA and fix routing dtype

- 链接: https://github.com/vllm-project/vllm/pull/36931
- 状态/时间: merged / 2026-03-13
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+18/-5，可读 patch 59 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Feat][Bugfix] Enable additional dimension for Flashinfer MLA and fix routing dtype」；模型线: DeepSeek V3.1；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/deepseek_v2.py`, `vllm/v1/attention/backends/mla/flashinfer_mla.py`；技术摘要: 覆盖「[Feat][Bugfix] Enable additional dimension for Flashinfer MLA and fix routing dtype」；主要实现面是 `vllm/model_executor/models/deepseek_v2.py`, `vllm/v1/attention/backends/mla/flashinfer_mla.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +15/-2 (17 lines); hunks: -47,7 +47,11; -333,8 +337,12 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`；`vllm/v1/attention/backends/mla/flashinfer_mla.py` modified +3/-3 (6 lines); hunks: -75,16 +75,16 @@ def supports_combination(; symbols: supports_combination，涉及 `supports_combination`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +15/-2 (17 lines); hunks: -47,7 +47,11; -333,8 +337,12 @@ def __init__(; symbols: __init__, forward
  - `vllm/v1/attention/backends/mla/flashinfer_mla.py` modified +3/-3 (6 lines); hunks: -75,16 +75,16 @@ def supports_combination(; symbols: supports_combination
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +15/-2; `vllm/v1/attention/backends/mla/flashinfer_mla.py` modified +3/-3
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_v2.py`, `vllm/v1/attention/backends/mla/flashinfer_mla.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #37487 - [V0 Deprecation] Refactor kv cache from list to element

- 链接: https://github.com/vllm-project/vllm/pull/37487
- 状态/时间: merged / 2026-03-24
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 27 个文件，+70/-85，可读 patch 478 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[V0 Deprecation] Refactor kv cache from list to element」；模型线: DeepSeek V3.1；类别: 模型实现调整；主要 diff: `vllm/model_executor/layers/attention/mla_attention.py`, `vllm/model_executor/layers/attention/attention.py`, `vllm/model_executor/models/extract_hidden_states.py`；技术摘要: 覆盖「[V0 Deprecation] Refactor kv cache from list to element」；主要实现面是 `vllm/model_executor/layers/attention/mla_attention.py`, `vllm/model_executor/layers/attention/attention.py`, `vllm/model_executor/models/extract_hidden_states.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/attention/mla_attention.py` modified +3/-8 (11 lines); hunks: -416,12 +416,7 @@ def __init__(; -480,7 +475,7 @@ def forward(; symbols: __init__, forward, unified_mla_kv_cache_update，涉及 `__init__, forward, unified_mla_kv_cache_update`；`vllm/model_executor/layers/attention/attention.py` modified +2/-5 (7 lines); hunks: -350,10 +350,7 @@ def __init__(; -600,7 +597,7 @@ def get_attention_context(; symbols: __init__, get_attention_context，涉及 `__init__, get_attention_context`；`vllm/model_executor/models/extract_hidden_states.py` modified +2/-5 (7 lines); hunks: -51,7 +51,7 @@ def unified_kv_cache_update(; -288,10 +288,7 @@ def __init__(; symbols: unified_kv_cache_update, __init__，涉及 `unified_kv_cache_update, __init__`；`vllm/model_executor/layers/sparse_attn_indexer.py` modified +2/-2 (4 lines); hunks: -365,7 +365,7 @@ def forward_cuda(; -389,7 +389,7 @@ def forward_hip(; symbols: forward_cuda, forward_hip，涉及 `forward_cuda, forward_hip`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/attention/mla_attention.py` modified +3/-8 (11 lines); hunks: -416,12 +416,7 @@ def __init__(; -480,7 +475,7 @@ def forward(; symbols: __init__, forward, unified_mla_kv_cache_update
  - `vllm/model_executor/layers/attention/attention.py` modified +2/-5 (7 lines); hunks: -350,10 +350,7 @@ def __init__(; -600,7 +597,7 @@ def get_attention_context(; symbols: __init__, get_attention_context
  - `vllm/model_executor/models/extract_hidden_states.py` modified +2/-5 (7 lines); hunks: -51,7 +51,7 @@ def unified_kv_cache_update(; -288,10 +288,7 @@ def __init__(; symbols: unified_kv_cache_update, __init__
  - `vllm/model_executor/layers/sparse_attn_indexer.py` modified +2/-2 (4 lines); hunks: -365,7 +365,7 @@ def forward_cuda(; -389,7 +389,7 @@ def forward_hip(; symbols: forward_cuda, forward_hip
  - `vllm/model_executor/models/qwen3_next.py` modified +2/-2 (4 lines); hunks: -858,7 +858,7 @@ def _forward_core(; -1046,7 +1046,7 @@ def _forward_core_decode_non_spec(; symbols: _forward_core, _forward_core_decode_non_spec
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/layers/attention/mla_attention.py` modified +3/-8; `vllm/model_executor/layers/attention/attention.py` modified +2/-5; `vllm/model_executor/models/extract_hidden_states.py` modified +2/-5; `vllm/model_executor/layers/sparse_attn_indexer.py` modified +2/-2; `vllm/model_executor/models/qwen3_next.py` modified +2/-2; `vllm/model_executor/layers/attention/static_sink_attention.py` modified +1/-1
- 验证与风险: diff 自带测试面 `tests/compile/passes/test_fusion_attn.py`, `tests/compile/passes/test_rope_kvcache_fusion.py`, `tests/v1/e2e/general/test_mamba_prefix_cache.py`, `tests/v1/worker/test_gpu_model_runner.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #38029 - [Tool Parser][1/3] Pass tools to ToolParser constructor

- 链接: https://github.com/vllm-project/vllm/pull/38029
- 状态/时间: merged / 2026-03-26
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 38 个文件，+147/-92，可读 patch 858 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Tool Parser][1/3] Pass tools to ToolParser constructor」；模型线: DeepSeek V3.1；类别: 模型实现调整；主要 diff: `vllm/tool_parsers/abstract_tool_parser.py`, `vllm/tool_parsers/qwen3coder_tool_parser.py`, `vllm/tool_parsers/step3p5_tool_parser.py`；技术摘要: 覆盖「[Tool Parser][1/3] Pass tools to ToolParser constructor」；主要实现面是 `vllm/tool_parsers/abstract_tool_parser.py`, `vllm/tool_parsers/qwen3coder_tool_parser.py`, `vllm/tool_parsers/step3p5_tool_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/tool_parsers/abstract_tool_parser.py` modified +14/-2 (16 lines); hunks: -5,13 +5,18; -30,6 +35,8; symbols: ToolParser, __init__, vocab，涉及 `ToolParser, __init__, vocab`；`vllm/tool_parsers/qwen3coder_tool_parser.py` modified +5/-7 (12 lines); hunks: -10,7 +10,6; -23,15 +22,16; symbols: Qwen3CoderToolParser, __init__, _reset_streaming_state, _get_arguments_config，涉及 `Qwen3CoderToolParser, __init__, _reset_streaming_state`；`vllm/tool_parsers/step3p5_tool_parser.py` modified +5/-6 (11 lines); hunks: -11,7 +11,6; -23,7 +22,7; symbols: __init__, setup_parser, set_tools, _reset_xml_parser_after_tool_call，涉及 `__init__, setup_parser, set_tools`；`vllm/tool_parsers/qwen3xml_tool_parser.py` modified +5/-5 (10 lines); hunks: -11,7 +11,6; -24,6 +23,7; symbols: __init__, setup_parser, set_tools, _reset_xml_parser_after_tool_call，涉及 `__init__, setup_parser, set_tools`。
- 代码 diff 细节:
  - `vllm/tool_parsers/abstract_tool_parser.py` modified +14/-2 (16 lines); hunks: -5,13 +5,18; -30,6 +35,8; symbols: ToolParser, __init__, vocab
  - `vllm/tool_parsers/qwen3coder_tool_parser.py` modified +5/-7 (12 lines); hunks: -10,7 +10,6; -23,15 +22,16; symbols: Qwen3CoderToolParser, __init__, _reset_streaming_state, _get_arguments_config
  - `vllm/tool_parsers/step3p5_tool_parser.py` modified +5/-6 (11 lines); hunks: -11,7 +11,6; -23,7 +22,7; symbols: __init__, setup_parser, set_tools, _reset_xml_parser_after_tool_call
  - `vllm/tool_parsers/qwen3xml_tool_parser.py` modified +5/-5 (10 lines); hunks: -11,7 +11,6; -24,6 +23,7; symbols: __init__, setup_parser, set_tools, _reset_xml_parser_after_tool_call
  - `vllm/tool_parsers/llama4_pythonic_tool_parser.py` modified +7/-2 (9 lines); hunks: -17,6 +17,7; -47,8 +48,12 @@ class Llama4PythonicToolParser(ToolParser):; symbols: Llama4PythonicToolParser, __init__
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/tool_parsers/abstract_tool_parser.py` modified +14/-2; `vllm/tool_parsers/qwen3coder_tool_parser.py` modified +5/-7; `vllm/tool_parsers/step3p5_tool_parser.py` modified +5/-6; `vllm/tool_parsers/qwen3xml_tool_parser.py` modified +5/-5; `vllm/tool_parsers/llama4_pythonic_tool_parser.py` modified +7/-2; `vllm/tool_parsers/llama_tool_parser.py` modified +7/-2
- 验证与风险: runtime 路径改动集中在 `vllm/entrypoints/openai/chat_completion/serving.py`, `vllm/entrypoints/openai/engine/serving.py`, `vllm/entrypoints/openai/parser/responses_parser.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #38684 - [Perf] DSV3.2 Indexer Fused Weights Projection

- 链接: https://github.com/vllm-project/vllm/pull/38684
- 状态/时间: merged / 2026-04-02
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+25/-14，可读 patch 79 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Perf] DSV3.2 Indexer Fused Weights Projection」；模型线: DeepSeek V3.1；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`；技术摘要: 覆盖「[Perf] DSV3.2 Indexer Fused Weights Projection」；主要实现面是 `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +22/-14 (36 lines); hunks: -639,21 +639,19 @@ def __init__(; -694,7 +692,11 @@ def forward(; symbols: __init__, forward, load_weights，涉及 `__init__, forward, load_weights`；`vllm/model_executor/models/deepseek_mtp.py` modified +3/-0 (3 lines); hunks: -241,6 +241,9 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +22/-14 (36 lines); hunks: -639,21 +639,19 @@ def __init__(; -694,7 +692,11 @@ def forward(; symbols: __init__, forward, load_weights
  - `vllm/model_executor/models/deepseek_mtp.py` modified +3/-0 (3 lines); hunks: -241,6 +241,9 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +22/-14; `vllm/model_executor/models/deepseek_mtp.py` modified +3/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_mtp.py`, `vllm/model_executor/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #38870 - [Bugfix] Fix DSV32 weight loading

- 链接: https://github.com/vllm-project/vllm/pull/38870
- 状态/时间: merged / 2026-04-04
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+68/-27，可读 patch 158 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix DSV32 weight loading」；模型线: DeepSeek V3.1；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`；技术摘要: 覆盖「[Bugfix] Fix DSV32 weight loading」；主要实现面是 `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +55/-24 (79 lines); hunks: -625,6 +625,11 @@ def __init__(; -639,18 +644,36 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`；`vllm/model_executor/models/deepseek_mtp.py` modified +13/-3 (16 lines); hunks: -184,11 +184,16 @@ class DeepSeekMTP(nn.Module, DeepseekV2MixtureOfExperts):; -241,11 +246,16 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: DeepSeekMTP, __init__, set_moe_parameters, load_weights，涉及 `DeepSeekMTP, __init__, set_moe_parameters`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +55/-24 (79 lines); hunks: -625,6 +625,11 @@ def __init__(; -639,18 +644,36 @@ def __init__(; symbols: __init__, forward
  - `vllm/model_executor/models/deepseek_mtp.py` modified +13/-3 (16 lines); hunks: -184,11 +184,16 @@ class DeepSeekMTP(nn.Module, DeepseekV2MixtureOfExperts):; -241,11 +246,16 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: DeepSeekMTP, __init__, set_moe_parameters, load_weights
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +55/-24; `vllm/model_executor/models/deepseek_mtp.py` modified +13/-3
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_mtp.py`, `vllm/model_executor/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #37421 - [Perf][Kernel] Persistent TopK scheduler: unified CUDAGraph-safe kernel with dynamic per-row dispatch - DeepSeek-V3.2 DSA decode

- 链接: https://github.com/vllm-project/vllm/pull/37421
- 状态/时间: merged / 2026-04-08
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+2039/-483，可读 patch 2698 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Perf][Kernel] Persistent TopK scheduler: unified CUDAGraph-safe kernel with dynamic per-row dispatch - DeepSeek-V3.2 DSA decode」；模型线: DeepSeek V3.1；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/sparse_attn_indexer.py`, `vllm/model_executor/models/deepseek_v2.py`, `csrc/persistent_topk.cuh`；技术摘要: 覆盖「[Perf][Kernel] Persistent TopK scheduler: unified CUDAGraph-safe kernel with dynamic per-row dispatch - DeepSeek-V3.2 DSA decode」；主要实现面是 `vllm/model_executor/layers/sparse_attn_indexer.py`, `vllm/model_executor/models/deepseek_v2.py`, `csrc/persistent_topk.cuh`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/sparse_attn_indexer.py` modified +24/-24 (48 lines); hunks: -25,6 +25,8; -51,6 +53,7 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer，涉及 `sparse_attn_indexer`；`vllm/model_executor/models/deepseek_v2.py` modified +6/-2 (8 lines); hunks: -67,7 +67,9; -1203,7 +1205,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__，涉及 `__init__`；`csrc/persistent_topk.cuh` added +1321/-0 (1321 lines); hunks: -0,0 +1,1321；`tests/kernels/test_top_k_per_row.py` modified +540/-78 (618 lines); hunks: -122,6 +122,39 @@ def compare_top_k_results(; -278,111 +311,540 @@ def test_top_k_per_row_decode_large_vocab_size(clean_log...; symbols: compare_top_k_results, validate_topk_against_reference, test_top_k_per_row_decode_large_vocab_size, test_deepseek_hybrid_topk，涉及 `compare_top_k_results, validate_topk_against_reference, test_top_k_per_row_decode_large_vocab_size`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/sparse_attn_indexer.py` modified +24/-24 (48 lines); hunks: -25,6 +25,8; -51,6 +53,7 @@ def sparse_attn_indexer(; symbols: sparse_attn_indexer
  - `vllm/model_executor/models/deepseek_v2.py` modified +6/-2 (8 lines); hunks: -67,7 +67,9; -1203,7 +1205,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__
  - `csrc/persistent_topk.cuh` added +1321/-0 (1321 lines); hunks: -0,0 +1,1321
  - `tests/kernels/test_top_k_per_row.py` modified +540/-78 (618 lines); hunks: -122,6 +122,39 @@ def compare_top_k_results(; -278,111 +311,540 @@ def test_top_k_per_row_decode_large_vocab_size(clean_log...; symbols: compare_top_k_results, validate_topk_against_reference, test_top_k_per_row_decode_large_vocab_size, test_deepseek_hybrid_topk
  - `csrc/topk.cu` modified +139/-358 (497 lines); hunks: -1,373 +1,154
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/layers/sparse_attn_indexer.py` modified +24/-24; `vllm/model_executor/models/deepseek_v2.py` modified +6/-2; `vllm/v1/attention/backends/mla/indexer.py` modified +0/-12
  - other: `csrc/persistent_topk.cuh` added +1321/-0; `csrc/topk.cu` modified +139/-358; `csrc/torch_bindings.cpp` modified +3/-4; `csrc/ops.h` modified +3/-3
  - tests: `tests/kernels/test_top_k_per_row.py` modified +540/-78
- 验证与风险: diff 自带测试面 `tests/kernels/test_top_k_per_row.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #38928 - [Bugfix][Perf] Indexer upcast WK to BF16 for fusion

- 链接: https://github.com/vllm-project/vllm/pull/38928
- 状态/时间: merged / 2026-04-15
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+84/-64，可读 patch 239 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix][Perf] Indexer upcast WK to BF16 for fusion」；模型线: DeepSeek V3.1；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`；技术摘要: 覆盖「[Bugfix][Perf] Indexer upcast WK to BF16 for fusion」；主要实现面是 `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +70/-53 (123 lines); hunks: -66,6 +66,10; -628,10 +632,6 @@ def __init__(; symbols: __init__, forward, _try_load_fp8_indexer_wk，涉及 `__init__, forward, _try_load_fp8_indexer_wk`；`vllm/model_executor/models/deepseek_mtp.py` modified +14/-11 (25 lines); hunks: -30,6 +30,7; -190,10 +191,6 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, set_moe_parameters, load_weights，涉及 `__init__, set_moe_parameters, load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +70/-53 (123 lines); hunks: -66,6 +66,10; -628,10 +632,6 @@ def __init__(; symbols: __init__, forward, _try_load_fp8_indexer_wk
  - `vllm/model_executor/models/deepseek_mtp.py` modified +14/-11 (25 lines); hunks: -30,6 +30,7; -190,10 +191,6 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, set_moe_parameters, load_weights
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +70/-53; `vllm/model_executor/models/deepseek_mtp.py` modified +14/-11
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_mtp.py`, `vllm/model_executor/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #35949 - [MoE Refactor] Move the shared/fused expert output sum into MoERunnerBase

- 链接: https://github.com/vllm-project/vllm/pull/35949
- 状态/时间: merged / 2026-04-20
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 53 个文件，+325/-702，可读 patch 2430 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MoE Refactor] Move the shared/fused expert output sum into MoERunnerBase」；模型线: DeepSeek V3.1；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/exaone_moe.py`；技术摘要: 覆盖「[MoE Refactor] Move the shared/fused expert output sum into MoERunnerBase」；主要实现面是 `vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/exaone_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py` modified +175/-86 (261 lines); hunks: -81,15 +81,17 @@ def _resolve_layer_name(layer_name: str | LayerName) -> str:; -113,7 +115,7 @@ def _moe_forward_shared(; symbols: _resolve_layer_name, _moe_forward, _moe_forward_shared, _moe_forward_shared_fake，涉及 `_resolve_layer_name, _moe_forward, _moe_forward_shared`；`vllm/model_executor/layers/fused_moe/layer.py` modified +28/-32 (60 lines); hunks: -230,11 +230,18 @@ class FusedMoE(PluggableLayer):; -246,7 +253,6 @@ def __init__(; symbols: FusedMoE, __init__，涉及 `FusedMoE, __init__`；`vllm/model_executor/models/exaone_moe.py` modified +18/-28 (46 lines); hunks: -31,6 +31,7; -116,12 +117,26 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`；`vllm/model_executor/models/kimi_linear.py` modified +20/-26 (46 lines); hunks: -11,11 +11,10; -132,12 +131,25 @@ def __init__(; symbols: __init__, forward, load_weights，涉及 `__init__, forward, load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py` modified +175/-86 (261 lines); hunks: -81,15 +81,17 @@ def _resolve_layer_name(layer_name: str | LayerName) -> str:; -113,7 +115,7 @@ def _moe_forward_shared(; symbols: _resolve_layer_name, _moe_forward, _moe_forward_shared, _moe_forward_shared_fake
  - `vllm/model_executor/layers/fused_moe/layer.py` modified +28/-32 (60 lines); hunks: -230,11 +230,18 @@ class FusedMoE(PluggableLayer):; -246,7 +253,6 @@ def __init__(; symbols: FusedMoE, __init__
  - `vllm/model_executor/models/exaone_moe.py` modified +18/-28 (46 lines); hunks: -31,6 +31,7; -116,12 +117,26 @@ def __init__(; symbols: __init__, forward
  - `vllm/model_executor/models/kimi_linear.py` modified +20/-26 (46 lines); hunks: -11,11 +11,10; -132,12 +131,25 @@ def __init__(; symbols: __init__, forward, load_weights
  - `vllm/model_executor/models/AXK1.py` modified +5/-30 (35 lines); hunks: -100,7 +100,7 @@ def __init__(; -170,7 +170,6 @@ def __init__(; symbols: __init__, forward
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py` modified +175/-86; `vllm/model_executor/layers/fused_moe/layer.py` modified +28/-32; `vllm/model_executor/models/exaone_moe.py` modified +18/-28; `vllm/model_executor/models/kimi_linear.py` modified +20/-26; `vllm/model_executor/models/AXK1.py` modified +5/-30; `vllm/model_executor/models/ernie45_vl_moe.py` modified +5/-30
- 验证与风险: diff 自带测试面 `tests/compile/passes/test_vllm_fusion_pattern_matcher_pass.py`, `tests/kernels/moe/test_moe_layer.py`, `tests/kernels/moe/test_shared_fused_moe_routed_transform.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #35782 - [MoE Refactor] Remove SharedFusedMoE class

- 链接: https://github.com/vllm-project/vllm/pull/35782
- 状态/时间: merged / 2026-04-21
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 33 个文件，+112/-141，可读 patch 926 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MoE Refactor] Remove SharedFusedMoE class」；模型线: DeepSeek V3.1；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/fused_moe/shared_fused_moe.py`, `vllm/model_executor/models/afmoe.py`, `vllm/model_executor/models/llama4.py`；技术摘要: 覆盖「[MoE Refactor] Remove SharedFusedMoE class」；主要实现面是 `vllm/model_executor/layers/fused_moe/shared_fused_moe.py`, `vllm/model_executor/models/afmoe.py`, `vllm/model_executor/models/llama4.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/fused_moe/shared_fused_moe.py` removed +0/-25 (25 lines); hunks: -1,25 +0,0; symbols: SharedFusedMoE, forward，涉及 `SharedFusedMoE, forward`；`vllm/model_executor/models/afmoe.py` modified +5/-5 (10 lines); hunks: -18,7 +18,7; -124,8 +124,8 @@ def __init__(; symbols: __init__, make_empty_intermediate_tensors, get_expert_mapping，涉及 `__init__, make_empty_intermediate_tensors, get_expert_mapping`；`vllm/model_executor/models/llama4.py` modified +5/-5 (10 lines); hunks: -36,7 +36,7; -127,7 +127,7 @@ def __init__(self, vllm_config: VllmConfig, prefix: str = ""):; symbols: __init__, load_moe_expert_weights, load_weights，涉及 `__init__, load_moe_expert_weights, load_weights`；`vllm/model_executor/models/AXK1.py` modified +4/-4 (8 lines); hunks: -42,7 +42,7; -163,7 +163,7 @@ def __init__(; symbols: __init__, compute_logits, get_expert_mapping, load_weights，涉及 `__init__, compute_logits, get_expert_mapping`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/fused_moe/shared_fused_moe.py` removed +0/-25 (25 lines); hunks: -1,25 +0,0; symbols: SharedFusedMoE, forward
  - `vllm/model_executor/models/afmoe.py` modified +5/-5 (10 lines); hunks: -18,7 +18,7; -124,8 +124,8 @@ def __init__(; symbols: __init__, make_empty_intermediate_tensors, get_expert_mapping
  - `vllm/model_executor/models/llama4.py` modified +5/-5 (10 lines); hunks: -36,7 +36,7; -127,7 +127,7 @@ def __init__(self, vllm_config: VllmConfig, prefix: str = ""):; symbols: __init__, load_moe_expert_weights, load_weights
  - `vllm/model_executor/models/AXK1.py` modified +4/-4 (8 lines); hunks: -42,7 +42,7; -163,7 +163,7 @@ def __init__(; symbols: __init__, compute_logits, get_expert_mapping, load_weights
  - `vllm/model_executor/models/deepseek_v2.py` modified +4/-4 (8 lines); hunks: -48,9 +48,9; -311,7 +311,7 @@ def __init__(; symbols: __init__, compute_logits, get_expert_mapping, load_weights
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/layers/fused_moe/shared_fused_moe.py` removed +0/-25; `vllm/model_executor/models/afmoe.py` modified +5/-5; `vllm/model_executor/models/llama4.py` modified +5/-5; `vllm/model_executor/models/AXK1.py` modified +4/-4; `vllm/model_executor/models/deepseek_v2.py` modified +4/-4; `vllm/model_executor/models/ernie45_moe.py` modified +4/-4
- 验证与风险: diff 自带测试面 `tests/kernels/moe/test_moe_layer.py`, `tests/kernels/moe/test_shared_fused_moe_routed_transform.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #40671 - [MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping

- 链接: https://github.com/vllm-project/vllm/pull/40671
- 状态/时间: merged / 2026-04-23
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 53 个文件，+254/-98，可读 patch 1073 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping」；模型线: DeepSeek V3.1；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/llama4.py`, `vllm/model_executor/models/glm4_moe_lite.py`；技术摘要: 覆盖「[MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping」；主要实现面是 `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/llama4.py`, `vllm/model_executor/models/glm4_moe_lite.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/fused_moe/layer.py` modified +19/-0 (19 lines); hunks: -1618,6 +1618,25 @@ def extra_repr(self) -> str:; symbols: extra_repr, fused_moe_make_expert_params_mapping，涉及 `extra_repr, fused_moe_make_expert_params_mapping`；`vllm/model_executor/models/llama4.py` modified +7/-4 (11 lines); hunks: -36,7 +36,10; -414,7 +417,7 @@ def load_moe_expert_weights(; symbols: load_moe_expert_weights, load_weights，涉及 `load_moe_expert_weights, load_weights`；`vllm/model_executor/models/glm4_moe_lite.py` modified +6/-4 (10 lines); hunks: -41,7 +41,9; -308,7 +310,7 @@ def make_empty_intermediate_tensors(; symbols: make_empty_intermediate_tensors, get_expert_mapping, load_weights, compute_logits，涉及 `make_empty_intermediate_tensors, get_expert_mapping, load_weights`；`vllm/model_executor/models/AXK1.py` modified +6/-3 (9 lines); hunks: -42,7 +42,10; -916,7 +919,7 @@ def compute_logits(; symbols: compute_logits, get_expert_mapping, load_weights，涉及 `compute_logits, get_expert_mapping, load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/fused_moe/layer.py` modified +19/-0 (19 lines); hunks: -1618,6 +1618,25 @@ def extra_repr(self) -> str:; symbols: extra_repr, fused_moe_make_expert_params_mapping
  - `vllm/model_executor/models/llama4.py` modified +7/-4 (11 lines); hunks: -36,7 +36,10; -414,7 +417,7 @@ def load_moe_expert_weights(; symbols: load_moe_expert_weights, load_weights
  - `vllm/model_executor/models/glm4_moe_lite.py` modified +6/-4 (10 lines); hunks: -41,7 +41,9; -308,7 +310,7 @@ def make_empty_intermediate_tensors(; symbols: make_empty_intermediate_tensors, get_expert_mapping, load_weights, compute_logits
  - `vllm/model_executor/models/AXK1.py` modified +6/-3 (9 lines); hunks: -42,7 +42,10; -916,7 +919,7 @@ def compute_logits(; symbols: compute_logits, get_expert_mapping, load_weights
  - `vllm/model_executor/models/afmoe.py` modified +5/-2 (7 lines); hunks: -18,7 +18,10; -479,7 +482,7 @@ def make_empty_intermediate_tensors(; symbols: make_empty_intermediate_tensors, get_expert_mapping
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/layers/fused_moe/layer.py` modified +19/-0; `vllm/model_executor/models/llama4.py` modified +7/-4; `vllm/model_executor/models/glm4_moe_lite.py` modified +6/-4; `vllm/model_executor/models/AXK1.py` modified +6/-3; `vllm/model_executor/models/afmoe.py` modified +5/-2; `vllm/model_executor/models/bailing_moe.py` modified +5/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/fused_moe/__init__.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/AXK1.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #39999 - [ROCm] Cast score correction bias tensor during model construction for DeepSeek/Kimi-K2

- 链接: https://github.com/vllm-project/vllm/pull/39999
- 状态/时间: merged / 2026-04-24
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+19/-2，可读 patch 49 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[ROCm] Cast score correction bias tensor during model construction for DeepSeek/Kimi-K2」；模型线: DeepSeek V3.1；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py`, `vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py`；技术摘要: 覆盖「[ROCm] Cast score correction bias tensor during model construction for DeepSeek/Kimi-K2」；主要实现面是 `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py`, `vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +15/-0 (15 lines); hunks: -348,6 +348,21 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`；`vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py` modified +1/-1 (2 lines); hunks: -152,7 +152,7 @@ def rocm_aiter_grouped_topk(; symbols: rocm_aiter_grouped_topk，涉及 `rocm_aiter_grouped_topk`；`vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py` modified +1/-1 (2 lines); hunks: -136,7 +136,7 @@ def fused_topk_bias(; symbols: fused_topk_bias，涉及 `fused_topk_bias`；`vllm/_aiter_ops.py` modified +2/-0 (2 lines); hunks: -1782,6 +1782,8 @@ def biased_grouped_topk(; symbols: biased_grouped_topk，涉及 `biased_grouped_topk`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +15/-0 (15 lines); hunks: -348,6 +348,21 @@ def __init__(; symbols: __init__, forward
  - `vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py` modified +1/-1 (2 lines); hunks: -152,7 +152,7 @@ def rocm_aiter_grouped_topk(; symbols: rocm_aiter_grouped_topk
  - `vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py` modified +1/-1 (2 lines); hunks: -136,7 +136,7 @@ def fused_topk_bias(; symbols: fused_topk_bias
  - `vllm/_aiter_ops.py` modified +2/-0 (2 lines); hunks: -1782,6 +1782,8 @@ def biased_grouped_topk(; symbols: biased_grouped_topk
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +15/-0; `vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py` modified +1/-1; `vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py` modified +1/-1; `vllm/_aiter_ops.py` modified +2/-0
- 验证与风险: runtime 路径改动集中在 `vllm/_aiter_ops.py`, `vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py`, `vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #39141 - [Perf] Update TRTLLM supported MoE routing methods

- 链接: https://github.com/vllm-project/vllm/pull/39141
- 状态/时间: merged / 2026-04-27
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+26/-92，可读 patch 258 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Perf] Update TRTLLM supported MoE routing methods」；模型线: DeepSeek V3.1；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py`, `vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py`, `vllm/model_executor/layers/fused_moe/config.py`；技术摘要: 覆盖「[Perf] Update TRTLLM supported MoE routing methods」；主要实现面是 `vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py`, `vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py`, `vllm/model_executor/layers/fused_moe/config.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py` modified +9/-43 (52 lines); hunks: -175,13 +175,6 @@ def apply(; -196,11 +189,7 @@ def apply(; symbols: apply, TrtLlmFp8ExpertsMonolithic, _supports_router_logits_dtype，涉及 `apply, TrtLlmFp8ExpertsMonolithic, _supports_router_logits_dtype`；`vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py` modified +3/-29 (32 lines); hunks: -198,13 +198,6 @@ def apply(; -233,7 +226,7 @@ def apply(; symbols: apply, _supports_routing_method, _supports_router_logits_dtype，涉及 `apply, _supports_routing_method, _supports_router_logits_dtype`；`vllm/model_executor/layers/fused_moe/config.py` modified +14/-7 (21 lines); hunks: -113,14 +113,17 @@ class RoutingMethodType(IntEnum):; -141,12 +144,16 @@ def get_routing_method_type(; symbols: RoutingMethodType, get_routing_method_type，涉及 `RoutingMethodType, get_routing_method_type`；`vllm/model_executor/models/deepseek_v2.py` modified +0/-12 (12 lines); hunks: -50,7 +50,6; -338,17 +337,6 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py` modified +9/-43 (52 lines); hunks: -175,13 +175,6 @@ def apply(; -196,11 +189,7 @@ def apply(; symbols: apply, TrtLlmFp8ExpertsMonolithic, _supports_router_logits_dtype
  - `vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py` modified +3/-29 (32 lines); hunks: -198,13 +198,6 @@ def apply(; -233,7 +226,7 @@ def apply(; symbols: apply, _supports_routing_method, _supports_router_logits_dtype
  - `vllm/model_executor/layers/fused_moe/config.py` modified +14/-7 (21 lines); hunks: -113,14 +113,17 @@ class RoutingMethodType(IntEnum):; -141,12 +144,16 @@ def get_routing_method_type(; symbols: RoutingMethodType, get_routing_method_type
  - `vllm/model_executor/models/deepseek_v2.py` modified +0/-12 (12 lines); hunks: -50,7 +50,6; -338,17 +337,6 @@ def __init__(; symbols: __init__
  - `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py` modified +0/-1 (1 lines); hunks: -275,7 +275,6 @@ def _return_or_raise(; symbols: _return_or_raise
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py` modified +9/-43; `vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py` modified +3/-29; `vllm/model_executor/layers/fused_moe/config.py` modified +14/-7; `vllm/model_executor/models/deepseek_v2.py` modified +0/-12; `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py` modified +0/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/fused_moe/config.py`, `vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py`, `vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #37735 - [Feature]: IndexCache support for DSA models

- 链接: https://github.com/vllm-project/vllm/pull/37735
- 状态/时间: merged / 2026-04-29
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+83/-5，可读 patch 138 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Feature]: IndexCache support for DSA models」；模型线: DeepSeek V3.1；类别: 文档/测试/CI；主要 diff: `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/layers/mla.py`, `docs/features/index_cache.md`；技术摘要: 覆盖「[Feature]: IndexCache support for DSA models」；主要实现面是 `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/layers/mla.py`, `docs/features/index_cache.md`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +21/-1 (22 lines); hunks: -82,7 +82,10; -963,6 +966,7 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`；`vllm/model_executor/layers/mla.py` modified +8/-4 (12 lines); hunks: -64,6 +64,7 @@ def __init__(; -87,6 +88,11 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`；`docs/features/index_cache.md` added +54/-0 (54 lines); hunks: -0,0 +1,54。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +21/-1 (22 lines); hunks: -82,7 +82,10; -963,6 +966,7 @@ def __init__(; symbols: __init__, forward
  - `vllm/model_executor/layers/mla.py` modified +8/-4 (12 lines); hunks: -64,6 +64,7 @@ def __init__(; -87,6 +88,11 @@ def __init__(; symbols: __init__, forward
  - `docs/features/index_cache.md` added +54/-0 (54 lines); hunks: -0,0 +1,54
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +21/-1; `vllm/model_executor/layers/mla.py` modified +8/-4
  - docs: `docs/features/index_cache.md` added +54/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/mla.py`, `vllm/model_executor/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #41217 - [ROCm][Deepseek] dsv3.2 further optimization

- 链接: https://github.com/vllm-project/vllm/pull/41217
- 状态/时间: merged / 2026-05-01
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+293/-73，可读 patch 605 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[ROCm][Deepseek] dsv3.2 further optimization」；模型线: DeepSeek V3.1；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/deepseek_v2.py`, `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py`, `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`；技术摘要: 覆盖「[ROCm][Deepseek] dsv3.2 further optimization」；主要实现面是 `vllm/model_executor/models/deepseek_v2.py`, `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py`, `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +38/-23 (61 lines); hunks: -674,30 +674,45 @@ def forward(; symbols: forward，涉及 `forward`；`vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py` modified +227/-29 (256 lines); hunks: -7,13 +7,15; -25,9 +27,6; symbols: _convert_req_index_to_global_index_kernel, triton_convert_req_index_to_global_index, generate_sparse_seqlen_kernel, generate_sparse_seqlen_triton，涉及 `_convert_req_index_to_global_index_kernel, triton_convert_req_index_to_global_index, generate_sparse_seqlen_kernel`；`vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` modified +22/-19 (41 lines); hunks: -13,9 +13,6; -97,7 +94,8 @@ def indexer_k_quant_and_cache_triton(; symbols: _indexer_k_quant_and_cache_kernel, indexer_k_quant_and_cache_triton, cp_gather_indexer_k_quant_cache_triton, rocm_fp8_paged_mqa_logits，涉及 `_indexer_k_quant_and_cache_kernel, indexer_k_quant_and_cache_triton, cp_gather_indexer_k_quant_cache_triton`；`vllm/v1/attention/backends/mla/rocm_aiter_mla.py` modified +4/-0 (4 lines); hunks: -396,6 +396,7 @@ class AiterMLAHelper:; -419,6 +420,9 @@ def get_actual_mla_num_heads(num_heads: int) -> int:; symbols: AiterMLAHelper, check_num_heads_validity, get_actual_mla_num_heads, get_mla_padded_q，涉及 `AiterMLAHelper, check_num_heads_validity, get_actual_mla_num_heads`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +38/-23 (61 lines); hunks: -674,30 +674,45 @@ def forward(; symbols: forward
  - `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py` modified +227/-29 (256 lines); hunks: -7,13 +7,15; -25,9 +27,6; symbols: _convert_req_index_to_global_index_kernel, triton_convert_req_index_to_global_index, generate_sparse_seqlen_kernel, generate_sparse_seqlen_triton
  - `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` modified +22/-19 (41 lines); hunks: -13,9 +13,6; -97,7 +94,8 @@ def indexer_k_quant_and_cache_triton(; symbols: _indexer_k_quant_and_cache_kernel, indexer_k_quant_and_cache_triton, cp_gather_indexer_k_quant_cache_triton, rocm_fp8_paged_mqa_logits
  - `vllm/v1/attention/backends/mla/rocm_aiter_mla.py` modified +4/-0 (4 lines); hunks: -396,6 +396,7 @@ class AiterMLAHelper:; -419,6 +420,9 @@ def get_actual_mla_num_heads(num_heads: int) -> int:; symbols: AiterMLAHelper, check_num_heads_validity, get_actual_mla_num_heads, get_mla_padded_q
  - `docs/design/attention_backends.md` modified +1/-1 (2 lines); hunks: -216,7 +216,7 @@ configuration.
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +38/-23; `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py` modified +227/-29; `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` modified +22/-19; `vllm/v1/attention/backends/mla/rocm_aiter_mla.py` modified +4/-0; `vllm/v1/attention/backends/mla/indexer.py` modified +1/-1
  - docs: `docs/design/attention_backends.md` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_v2.py`, `vllm/v1/attention/backends/mla/indexer.py`, `vllm/v1/attention/backends/mla/rocm_aiter_mla.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #41405 - [ROCm][Bugfix] Fix init-time bias dtype cast when gate.out_dtype is None

- 链接: https://github.com/vllm-project/vllm/pull/41405
- 状态/时间: merged / 2026-05-02
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-1，可读 patch 11 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[ROCm][Bugfix] Fix init-time bias dtype cast when gate.out_dtype is None」；模型线: DeepSeek V3.1；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/deepseek_v2.py`；技术摘要: 覆盖「[ROCm][Bugfix] Fix init-time bias dtype cast when gate.out_dtype is None」；主要实现面是 `vllm/model_executor/models/deepseek_v2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +2/-1 (3 lines); hunks: -351,8 +351,9 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +2/-1 (3 lines); hunks: -351,8 +351,9 @@ def __init__(; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/deepseek_v2.py
@@ -351,8 +351,9 @@ def __init__(
+            gate_out_dtype = self.gate.out_dtype or self.gate.weight.dtype
-                self.gate.e_score_correction_bias.data.to(self.gate.out_dtype)
+                self.gate.e_score_correction_bias.data.to(gate_out_dtype)
```

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +2/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #40759 - [Examples] Resettle Disaggregated examples.

- 链接: https://github.com/vllm-project/vllm/pull/40759
- 状态/时间: merged / 2026-05-06
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 54 个文件，+29/-31，可读 patch 215 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Examples] Resettle Disaggregated examples.」；模型线: DeepSeek V3.1；类别: 文档/测试/CI；主要 diff: `examples/disaggregated/lmcache/disagg_prefill_lmcache_v1/configs/lmcache-decoder-config.yaml`, `examples/disaggregated/lmcache/disagg_prefill_lmcache_v1/configs/lmcache-prefiller-config.yaml`, `docs/features/disagg_prefill.md`；技术摘要: 覆盖「[Examples] Resettle Disaggregated examples.」；主要实现面是 `examples/disaggregated/lmcache/disagg_prefill_lmcache_v1/configs/lmcache-decoder-config.yaml`, `examples/disaggregated/lmcache/disagg_prefill_lmcache_v1/configs/lmcache-prefiller-config.yaml`, `docs/features/disagg_prefill.md`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `examples/disaggregated/lmcache/disagg_prefill_lmcache_v1/configs/lmcache-decoder-config.yaml` renamed +0/-0 (0 lines)；`examples/disaggregated/lmcache/disagg_prefill_lmcache_v1/configs/lmcache-prefiller-config.yaml` renamed +0/-0 (0 lines)；`docs/features/disagg_prefill.md` modified +6/-6 (12 lines); hunks: -17,15 +17,15 @@ Two main reasons:; -44,7 +44,7 @@ For NixlConnector, you may also specify one or multiple NIXL_B...；`docs/features/mooncake_connector_usage.md` modified +3/-3 (6 lines); hunks: -31,7 +31,7 @@ vllm serve Qwen/Qwen2.5-7B-Instruct --port 8020 --kv-transfer-...; -65,5 +65,5 @@ Now you can send requests to the proxy server through port 8000.。
- 代码 diff 细节:
  - `examples/disaggregated/lmcache/disagg_prefill_lmcache_v1/configs/lmcache-decoder-config.yaml` renamed +0/-0 (0 lines)
  - `examples/disaggregated/lmcache/disagg_prefill_lmcache_v1/configs/lmcache-prefiller-config.yaml` renamed +0/-0 (0 lines)
  - `docs/features/disagg_prefill.md` modified +6/-6 (12 lines); hunks: -17,15 +17,15 @@ Two main reasons:; -44,7 +44,7 @@ For NixlConnector, you may also specify one or multiple NIXL_B...
  - `docs/features/mooncake_connector_usage.md` modified +3/-3 (6 lines); hunks: -31,7 +31,7 @@ vllm serve Qwen/Qwen2.5-7B-Instruct --port 8020 --kv-transfer-...; -65,5 +65,5 @@ Now you can send requests to the proxy server through port 8000.
  - `.github/mergify.yml` modified +1/-3 (4 lines); hunks: -477,9 +477,7 @@ pull_request_rules:
- 关键代码摘录:

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

- 已读文件:
  - docs: `examples/disaggregated/lmcache/disagg_prefill_lmcache_v1/configs/lmcache-decoder-config.yaml` renamed +0/-0; `examples/disaggregated/lmcache/disagg_prefill_lmcache_v1/configs/lmcache-prefiller-config.yaml` renamed +0/-0; `docs/features/disagg_prefill.md` modified +6/-6; `docs/features/mooncake_connector_usage.md` modified +3/-3; `docs/design/p2p_nccl_connector.md` modified +2/-2; `docs/features/disagg_encoder.md` modified +2/-2
  - ci: `.github/mergify.yml` modified +1/-3
- 验证与风险: diff 自带测试面 `tests/v1/ec_connector/integration/README.md`, `tests/v1/ec_connector/integration/run_epd_correctness_test.sh`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #41835 - [ROCm][DeepSeek] Enable V3.2 TP4 AITER MLA

- 链接: https://github.com/vllm-project/vllm/pull/41835
- 状态/时间: merged / 2026-05-07
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+12/-10，可读 patch 50 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[ROCm][DeepSeek] Enable V3.2 TP4 AITER MLA」；模型线: DeepSeek V3.1；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/deepseek_v2.py`, `vllm/v1/attention/backends/mla/rocm_aiter_mla.py`；技术摘要: 覆盖「[ROCm][DeepSeek] Enable V3.2 TP4 AITER MLA」；主要实现面是 `vllm/model_executor/models/deepseek_v2.py`, `vllm/v1/attention/backends/mla/rocm_aiter_mla.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +11/-9 (20 lines); hunks: -299,6 +299,15 @@ def __init__(; -338,22 +347,15 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`；`vllm/v1/attention/backends/mla/rocm_aiter_mla.py` modified +1/-1 (2 lines); hunks: -396,7 +396,7 @@ class AiterMLAHelper:; symbols: AiterMLAHelper, check_num_heads_validity，涉及 `AiterMLAHelper, check_num_heads_validity`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +11/-9 (20 lines); hunks: -299,6 +299,15 @@ def __init__(; -338,22 +347,15 @@ def __init__(; symbols: __init__, forward
  - `vllm/v1/attention/backends/mla/rocm_aiter_mla.py` modified +1/-1 (2 lines); hunks: -396,7 +396,7 @@ class AiterMLAHelper:; symbols: AiterMLAHelper, check_num_heads_validity
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +11/-9; `vllm/v1/attention/backends/mla/rocm_aiter_mla.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_v2.py`, `vllm/v1/attention/backends/mla/rocm_aiter_mla.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #41706 - [Model] use AutoWeightsLoader for DeepSeekV2

- 链接: https://github.com/vllm-project/vllm/pull/41706
- 状态/时间: merged / 2026-05-09
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+186/-169，可读 patch 389 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] use AutoWeightsLoader for DeepSeekV2」；模型线: DeepSeek V3.1；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/deepseek_v2.py`；技术摘要: 覆盖「[Model] use AutoWeightsLoader for DeepSeekV2」；主要实现面是 `vllm/model_executor/models/deepseek_v2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +186/-169 (355 lines); hunks: -83,6 +83,7; -1254,6 +1255,16 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: st...; symbols: __init__, embed_input_ids, forward, DeepseekV2MixtureOfExperts，涉及 `__init__, embed_input_ids, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +186/-169 (355 lines); hunks: -83,6 +83,7; -1254,6 +1255,16 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: st...; symbols: __init__, embed_input_ids, forward, DeepseekV2MixtureOfExperts
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +186/-169
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #43781 - [Bugfix][ROCm] Fix Accuracy Drop in Sparse Indexer on gfx950

- 链接: https://github.com/vllm-project/vllm/pull/43781
- 状态/时间: merged / 2026-05-28
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+14/-4，可读 patch 82 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix][ROCm] Fix Accuracy Drop in Sparse Indexer on gfx950」；模型线: DeepSeek V3.1；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/deepseek_v2.py`, `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`；技术摘要: 覆盖「[Bugfix][ROCm] Fix Accuracy Drop in Sparse Indexer on gfx950」；主要实现面是 `vllm/model_executor/models/deepseek_v2.py`, `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +9/-1 (10 lines); hunks: -612,6 +612,7 @@ def __init__(; -673,15 +674,21 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`；`vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` modified +5/-3 (8 lines); hunks: -107,6 +107,7 @@ def indexer_k_quant_and_cache_triton(; -118,7 +119,7 @@ def indexer_k_quant_and_cache_triton(; symbols: indexer_k_quant_and_cache_triton, cp_gather_indexer_k_quant_cache_triton, rocm_fp8_paged_mqa_logits，涉及 `indexer_k_quant_and_cache_triton, cp_gather_indexer_k_quant_cache_triton, rocm_fp8_paged_mqa_logits`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +9/-1 (10 lines); hunks: -612,6 +612,7 @@ def __init__(; -673,15 +674,21 @@ def __init__(; symbols: __init__, forward
  - `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` modified +5/-3 (8 lines); hunks: -107,6 +107,7 @@ def indexer_k_quant_and_cache_triton(; -118,7 +119,7 @@ def indexer_k_quant_and_cache_triton(; symbols: indexer_k_quant_and_cache_triton, cp_gather_indexer_k_quant_cache_triton, rocm_fp8_paged_mqa_logits
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +9/-1; `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` modified +5/-3
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_v2.py`, `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #42982 - [ROCm][Perf] DSv3.2 MI355X TP4 decode-step orchestration cleanup (3 micro-opts)

- 链接: https://github.com/vllm-project/vllm/pull/42982
- 状态/时间: merged / 2026-05-29
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+59/-29，可读 patch 125 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[ROCm][Perf] DSv3.2 MI355X TP4 decode-step orchestration cleanup (3 micro-opts)」；模型线: DeepSeek V3.1；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/deepseek_v2.py`, `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py`；技术摘要: 覆盖「[ROCm][Perf] DSv3.2 MI355X TP4 decode-step orchestration cleanup (3 micro-opts)」；主要实现面是 `vllm/model_executor/models/deepseek_v2.py`, `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +2/-4 (6 lines); hunks: -304,10 +304,8 @@ def __init__(; symbols: __init__，涉及 `__init__`；`vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py` modified +57/-25 (82 lines); hunks: -375,7 +375,7 @@ def __init__(; -458,6 +458,10 @@ def __init__(; symbols: __init__, build，涉及 `__init__, build`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +2/-4 (6 lines); hunks: -304,10 +304,8 @@ def __init__(; symbols: __init__
  - `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py` modified +57/-25 (82 lines); hunks: -375,7 +375,7 @@ def __init__(; -458,6 +458,10 @@ def __init__(; symbols: __init__, build
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +2/-4; `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py` modified +57/-25
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_v2.py`, `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #42944 - fix: glm5.1 pp model loading

- 链接: https://github.com/vllm-project/vllm/pull/42944
- 状态/时间: merged / 2026-06-01
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+25/-5，可读 patch 93 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「fix: glm5.1 pp model loading」；模型线: DeepSeek V3.1；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`；技术摘要: 覆盖「fix: glm5.1 pp model loading」；主要实现面是 `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +17/-3 (20 lines); hunks: -105,6 +105,7; -742,7 +743,9 @@ def forward(; symbols: forward, _try_load_fp8_indexer_wk，涉及 `forward, _try_load_fp8_indexer_wk`；`vllm/model_executor/models/deepseek_mtp.py` modified +8/-2 (10 lines); hunks: -35,7 +35,7; -267,6 +267,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +17/-3 (20 lines); hunks: -105,6 +105,7; -742,7 +743,9 @@ def forward(; symbols: forward, _try_load_fp8_indexer_wk
  - `vllm/model_executor/models/deepseek_mtp.py` modified +8/-2 (10 lines); hunks: -35,7 +35,7; -267,6 +267,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +17/-3; `vllm/model_executor/models/deepseek_mtp.py` modified +8/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_mtp.py`, `vllm/model_executor/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #44420 - [feature] add index share feature for DSA MTP

- 链接: https://github.com/vllm-project/vllm/pull/44420
- 状态/时间: merged / 2026-06-07
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+114/-25，可读 patch 230 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[feature] add index share feature for DSA MTP」；模型线: DeepSeek V3.1；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`, `vllm/v1/spec_decode/llm_base_proposer.py`；技术摘要: 覆盖「[feature] add index share feature for DSA MTP」；主要实现面是 `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`, `vllm/v1/spec_decode/llm_base_proposer.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +16/-15 (31 lines); hunks: -1018,19 +1018,20 @@ def __init__(; -1252,8 +1253,8 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__，涉及 `__init__`；`vllm/model_executor/models/deepseek_mtp.py` modified +24/-2 (26 lines); hunks: -115,7 +115,9 @@ def forward(; -147,6 +149,22 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: forward, __init__, set_skip_topk, embed_input_ids，涉及 `forward, __init__, set_skip_topk`；`vllm/v1/spec_decode/llm_base_proposer.py` modified +32/-3 (35 lines); hunks: -70,6 +70,7 @@ def __init__(; -490,6 +491,11 @@ def propose(; symbols: __init__, propose, _maybe_share_lm_head，涉及 `__init__, propose, _maybe_share_lm_head`；`vllm/transformers_utils/model_arch_config_convertor.py` modified +33/-1 (34 lines); hunks: -50,7 +50,7 @@ def get_head_size(self) -> int:; -71,6 +71,38 @@ def get_head_size(self) -> int:; symbols: get_head_size, _get_qk_rope_head_dim, get_total_num_kv_heads，涉及 `get_head_size, _get_qk_rope_head_dim, get_total_num_kv_heads`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +16/-15 (31 lines); hunks: -1018,19 +1018,20 @@ def __init__(; -1252,8 +1253,8 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__
  - `vllm/model_executor/models/deepseek_mtp.py` modified +24/-2 (26 lines); hunks: -115,7 +115,9 @@ def forward(; -147,6 +149,22 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: forward, __init__, set_skip_topk, embed_input_ids
  - `vllm/v1/spec_decode/llm_base_proposer.py` modified +32/-3 (35 lines); hunks: -70,6 +70,7 @@ def __init__(; -490,6 +491,11 @@ def propose(; symbols: __init__, propose, _maybe_share_lm_head
  - `vllm/transformers_utils/model_arch_config_convertor.py` modified +33/-1 (34 lines); hunks: -50,7 +50,7 @@ def get_head_size(self) -> int:; -71,6 +71,38 @@ def get_head_size(self) -> int:; symbols: get_head_size, _get_qk_rope_head_dim, get_total_num_kv_heads
  - `vllm/v1/worker/gpu/spec_decode/eagle/utils.py` modified +9/-4 (13 lines); hunks: -76,10 +76,15 @@ def load_eagle_model(target_model: nn.Module, vllm_config: V...; symbols: load_eagle_model
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +16/-15; `vllm/model_executor/models/deepseek_mtp.py` modified +24/-2; `vllm/v1/spec_decode/llm_base_proposer.py` modified +32/-3; `vllm/transformers_utils/model_arch_config_convertor.py` modified +33/-1; `vllm/v1/worker/gpu/spec_decode/eagle/utils.py` modified +9/-4
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_mtp.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/transformers_utils/model_arch_config_convertor.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #45003 - [Frontend] Support strict mode for tool calling

- 链接: https://github.com/vllm-project/vllm/pull/45003
- 状态/时间: merged / 2026-06-12
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 29 个文件，+672/-1936，可读 patch 3162 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Frontend] Support strict mode for tool calling」；模型线: DeepSeek V3.1；类别: 文档/测试/CI；主要 diff: `vllm/tool_parsers/qwen3xml_tool_parser.py`, `vllm/tool_parsers/structural_tag_registry.py`, `tests/tool_parsers/test_structural_tag_registry.py`；技术摘要: 覆盖「[Frontend] Support strict mode for tool calling」；主要实现面是 `vllm/tool_parsers/qwen3xml_tool_parser.py`, `vllm/tool_parsers/structural_tag_registry.py`, `tests/tool_parsers/test_structural_tag_registry.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/tool_parsers/qwen3xml_tool_parser.py` removed +0/-1300 (1300 lines); hunks: -1,1300 +0,0; symbols: StreamingXMLToolCallParser, __init__, reset_streaming_state, parse_single_streaming_chunks，涉及 `StreamingXMLToolCallParser, __init__, reset_streaming_state`；`vllm/tool_parsers/structural_tag_registry.py` modified +174/-240 (414 lines); hunks: -1,14 +1,15; -24,23 +25,51; symbols: register_model_structural_tag, register_vllm_structural_tag, decorator, get_model_structural_tag，涉及 `register_model_structural_tag, register_vllm_structural_tag, decorator`；`tests/tool_parsers/test_structural_tag_registry.py` added +314/-0 (314 lines); hunks: -0,0 +1,314; symbols: sample_tools, test_supported_structural_tag_models_include_vllm_builtins, test_get_model_structural_tag_supports_all_xgrammar_builtins, test_get_model_structural_tag_supports_vllm_hermes，涉及 `sample_tools, test_supported_structural_tag_models_include_vllm_builtins, test_get_model_structural_tag_supports_all_xgrammar_builtins`；`tests/tool_parsers/test_qwen3coder_tool_parser.py` modified +13/-190 (203 lines); hunks: -3,6 +3,7; -19,15 +20,12; symbols: qwen3_tool_parser, qwen3_xml_tool_parser, qwen3_tool_parser_parametrized, assert_tool_calls，涉及 `qwen3_tool_parser, qwen3_xml_tool_parser, qwen3_tool_parser_parametrized`。
- 代码 diff 细节:
  - `vllm/tool_parsers/qwen3xml_tool_parser.py` removed +0/-1300 (1300 lines); hunks: -1,1300 +0,0; symbols: StreamingXMLToolCallParser, __init__, reset_streaming_state, parse_single_streaming_chunks
  - `vllm/tool_parsers/structural_tag_registry.py` modified +174/-240 (414 lines); hunks: -1,14 +1,15; -24,23 +25,51; symbols: register_model_structural_tag, register_vllm_structural_tag, decorator, get_model_structural_tag
  - `tests/tool_parsers/test_structural_tag_registry.py` added +314/-0 (314 lines); hunks: -0,0 +1,314; symbols: sample_tools, test_supported_structural_tag_models_include_vllm_builtins, test_get_model_structural_tag_supports_all_xgrammar_builtins, test_get_model_structural_tag_supports_vllm_hermes
  - `tests/tool_parsers/test_qwen3coder_tool_parser.py` modified +13/-190 (203 lines); hunks: -3,6 +3,7; -19,15 +20,12; symbols: qwen3_tool_parser, qwen3_xml_tool_parser, qwen3_tool_parser_parametrized, assert_tool_calls
  - `tests/tool_parsers/test_qwen3xml_tool_parser.py` removed +0/-72 (72 lines); hunks: -1,72 +0,0; symbols: TestQwen3xmlToolParser, test_config
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/tool_parsers/qwen3xml_tool_parser.py` removed +0/-1300; `vllm/tool_parsers/structural_tag_registry.py` modified +174/-240; `vllm/tool_parsers/abstract_tool_parser.py` modified +36/-28; `vllm/entrypoints/serve/render/serving.py` modified +24/-28; `vllm/tool_parsers/deepseekv4_tool_parser.py` modified +1/-15
  - tests: `tests/tool_parsers/test_structural_tag_registry.py` added +314/-0; `tests/tool_parsers/test_qwen3coder_tool_parser.py` modified +13/-190; `tests/tool_parsers/test_qwen3xml_tool_parser.py` removed +0/-72
- 验证与风险: diff 自带测试面 `requirements/test/rocm.txt`, `tests/entrypoints/openai/chat_completion/test_completion_with_function_calling.py`, `tests/entrypoints/openai/responses/conftest.py`, `tests/parser/test_parse.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #45895 - [bugfix]Indexer init skip and MTP TopK share for iteration

- 链接: https://github.com/vllm-project/vllm/pull/45895
- 状态/时间: merged / 2026-06-19
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+69/-30，可读 patch 198 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[bugfix]Indexer init skip and MTP TopK share for iteration」；模型线: DeepSeek V3.1；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`, `vllm/model_executor/layers/attention/mla_attention.py`；技术摘要: 覆盖「[bugfix]Indexer init skip and MTP TopK share for iteration」；主要实现面是 `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/deepseek_mtp.py`, `vllm/model_executor/layers/attention/mla_attention.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +22/-17 (39 lines); hunks: -998,8 +998,29 @@ def __init__(; -1017,22 +1038,6 @@ def __init__(; symbols: __init__，涉及 `__init__`；`vllm/model_executor/models/deepseek_mtp.py` modified +6/-2 (8 lines); hunks: -119,8 +119,12 @@ def forward(; symbols: forward, DeepSeekMultiTokenPredictor，涉及 `forward, DeepSeekMultiTokenPredictor`；`vllm/model_executor/layers/attention/mla_attention.py` modified +6/-0 (6 lines); hunks: -349,6 +349,7 @@ def __init__(; -437,6 +438,11 @@ def __init__(; symbols: __init__，涉及 `__init__`；`vllm/model_executor/layers/mla.py` modified +1/-0 (1 lines); hunks: -112,6 +112,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +22/-17 (39 lines); hunks: -998,8 +998,29 @@ def __init__(; -1017,22 +1038,6 @@ def __init__(; symbols: __init__
  - `vllm/model_executor/models/deepseek_mtp.py` modified +6/-2 (8 lines); hunks: -119,8 +119,12 @@ def forward(; symbols: forward, DeepSeekMultiTokenPredictor
  - `vllm/model_executor/layers/attention/mla_attention.py` modified +6/-0 (6 lines); hunks: -349,6 +349,7 @@ def __init__(; -437,6 +438,11 @@ def __init__(; symbols: __init__
  - `vllm/model_executor/layers/mla.py` modified +1/-0 (1 lines); hunks: -112,6 +112,7 @@ def __init__(; symbols: __init__
  - `vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py` modified +7/-3 (10 lines); hunks: -271,7 +271,7 @@ def __init__(; -301,8 +301,12 @@ def __init__(; symbols: __init__
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +22/-17; `vllm/model_executor/models/deepseek_mtp.py` modified +6/-2; `vllm/model_executor/layers/attention/mla_attention.py` modified +6/-0; `vllm/model_executor/layers/mla.py` modified +1/-0; `vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py` modified +7/-3; `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py` modified +7/-3
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/attention/mla_attention.py`, `vllm/model_executor/layers/mla.py`, `vllm/model_executor/models/deepseek_mtp.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #46199 - [Bugfix] Move extract_layer_index back inside is_v32 guard

- 链接: https://github.com/vllm-project/vllm/pull/46199
- 状态/时间: merged / 2026-06-20
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+23/-17，可读 patch 48 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Move extract_layer_index back inside is_v32 guard」；模型线: DeepSeek V3.1；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/deepseek_v2.py`；技术摘要: 覆盖「[Bugfix] Move extract_layer_index back inside is_v32 guard」；主要实现面是 `vllm/model_executor/models/deepseek_v2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/deepseek_v2.py` modified +23/-17 (40 lines); hunks: -1001,24 +1001,30 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/deepseek_v2.py` modified +23/-17 (40 lines); hunks: -1001,24 +1001,30 @@ def __init__(; symbols: __init__
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/deepseek_v2.py` modified +23/-17
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #46651 - [Perf] Remove redundant clone for GLM, Deepseek etc

- 链接: https://github.com/vllm-project/vllm/pull/46651
- 状态/时间: merged / 2026-06-25
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+4/-4，可读 patch 36 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Perf] Remove redundant clone for GLM, Deepseek etc」；模型线: DeepSeek V3.1；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/AXK1.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/glm4_moe_lite.py`；技术摘要: 覆盖「[Perf] Remove redundant clone for GLM, Deepseek etc」；主要实现面是 `vllm/model_executor/models/AXK1.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/glm4_moe_lite.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/AXK1.py` modified +1/-1 (2 lines); hunks: -649,7 +649,7 @@ def forward(; symbols: forward，涉及 `forward`；`vllm/model_executor/models/deepseek_v2.py` modified +1/-1 (2 lines); hunks: -1186,7 +1186,7 @@ def forward(; symbols: forward，涉及 `forward`；`vllm/model_executor/models/glm4_moe_lite.py` modified +1/-1 (2 lines); hunks: -184,7 +184,7 @@ def forward(; symbols: forward，涉及 `forward`；`vllm/model_executor/models/openpangu.py` modified +1/-1 (2 lines); hunks: -935,7 +935,7 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/AXK1.py` modified +1/-1 (2 lines); hunks: -649,7 +649,7 @@ def forward(; symbols: forward
  - `vllm/model_executor/models/deepseek_v2.py` modified +1/-1 (2 lines); hunks: -1186,7 +1186,7 @@ def forward(; symbols: forward
  - `vllm/model_executor/models/glm4_moe_lite.py` modified +1/-1 (2 lines); hunks: -184,7 +184,7 @@ def forward(; symbols: forward
  - `vllm/model_executor/models/openpangu.py` modified +1/-1 (2 lines); hunks: -935,7 +935,7 @@ def forward(; symbols: forward
- 关键代码摘录:

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

- 已读文件:
  - runtime: `vllm/model_executor/models/AXK1.py` modified +1/-1; `vllm/model_executor/models/deepseek_v2.py` modified +1/-1; `vllm/model_executor/models/glm4_moe_lite.py` modified +1/-1; `vllm/model_executor/models/openpangu.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/AXK1.py`, `vllm/model_executor/models/deepseek_v2.py`, `vllm/model_executor/models/glm4_moe_lite.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
