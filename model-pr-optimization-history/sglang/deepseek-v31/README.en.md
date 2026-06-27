# sglang DeepSeek V3.1 Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `examples/chat_template/tool_chat_template_deepseekv31.jinja` | [#9446](https://github.com/sgl-project/sglang/pull/9446), [#9895](https://github.com/sgl-project/sglang/pull/9895), [#14837](https://github.com/sgl-project/sglang/pull/14837) |
| `python/sglang/srt/function_call/deepseekv31_detector.py` | [#9446](https://github.com/sgl-project/sglang/pull/9446), [#11589](https://github.com/sgl-project/sglang/pull/11589), [#13394](https://github.com/sgl-project/sglang/pull/13394) |
| `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` | no direct PR-number commit |
| `python/sglang/srt/models/deepseek_common/__init__.py` | no direct PR-number commit |
| `python/sglang/srt/models/deepseek_common/amd/__init__.py` | no direct PR-number commit |
| `python/sglang/srt/models/deepseek_common/attention_backend_handler.py` | no direct PR-number commit |
| `python/sglang/srt/models/deepseek_common/attention_forward_methods/__init__.py` | no direct PR-number commit |
| `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_methods.py` | no direct PR-number commit |
| `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` | no direct PR-number commit |
| `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` | no direct PR-number commit |
| `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_cpu.py` | no direct PR-number commit |
| `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_rocm.py` | no direct PR-number commit |
| `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` | no direct PR-number commit |
| `python/sglang/srt/models/deepseek_common/utils.py` | no direct PR-number commit |
| `python/sglang/srt/models/deepseek_v2.py` | [#13954](https://github.com/sgl-project/sglang/pull/13954) |
| `test/manual/nightly/test_deepseek_v31_perf.py` | no direct PR-number commit |
| `test/manual/test_deepseek_v31.py` | no direct PR-number commit |
| `test/registered/amd/accuracy/mi30x/test_deepseek_v31_eval_amd.py` | no direct PR-number commit |
| `test/registered/amd/perf/mi30x/test_deepseek_v31_perf.py` | no direct PR-number commit |

## PR Coverage Summary

- Git-traced PRs: 6
- Extra PRs preserved from existing docs: 100
- Total PRs in this document: 106
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2025-08-21 | [#9464](https://github.com/sgl-project/sglang/pull/9464) | merged | Add deepseek v3.1 thinking parser support and update docs | `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/reasoning_parser.py` |
| 2025-08-23 | [#9544](https://github.com/sgl-project/sglang/pull/9544) | merged | [doc] deepseekv31 support | `benchmark/deepseek_v3/README.md`, `docs/basic_usage/deepseek.md` |
| 2025-08-27 | [#9446](https://github.com/sgl-project/sglang/pull/9446) | merged | Support DeepSeek-V3.1 tool call | `python/sglang/srt/function_call/deepseekv31_detector.py`, `examples/chat_template/tool_chat_template_deepseekv31.jinja` |
| 2025-09-03 | [#9895](https://github.com/sgl-project/sglang/pull/9895) | merged | Update tool_chat_template_deepseekv31.jinja | `examples/chat_template/tool_chat_template_deepseekv31.jinja` |
| 2025-09-27 | [#10550](https://github.com/sgl-project/sglang/pull/10550) | merged | Use jsonschema to constrain required or specific tool choice | `test/srt/function_call/test_json_schema_constraint.py`, `test/srt/openai_server/function_call/test_tool_choice.py`, `test/srt/test_function_call_parser.py` |
| 2025-09-29 | [#10875](https://github.com/sgl-project/sglang/pull/10875) | merged | feat(reasoning): improve enable thinking from request | `python/sglang/srt/entrypoints/openai/serving_chat.py` |
| 2025-10-03 | [#11189](https://github.com/sgl-project/sglang/pull/11189) | merged | Add --thinking-mode to run_eval | `python/sglang/test/run_eval.py`, `python/sglang/test/simple_eval_common.py` |
| 2025-10-07 | [#11223](https://github.com/sgl-project/sglang/pull/11223) | merged | Update tool parser and related documentation | `python/sglang/srt/function_call/function_call_parser.py`, `python/sglang/srt/server_args.py` |
| 2025-10-30 | [#12123](https://github.com/sgl-project/sglang/pull/12123) | merged | Fix DeepSeek chat templates to handle tool call arguments type checking (#11700) | `test/srt/test_deepseek_chat_templates.py`, `examples/chat_template/tool_chat_template_deepseekv3.jinja`, `examples/chat_template/tool_chat_template_deepseekv31.jinja` |
| 2025-11-13 | [#13190](https://github.com/sgl-project/sglang/pull/13190) | merged | Remove enable_dp_attention in deepseek nightly tests | `test/srt/nightly/test_deepseek_v32_perf.py`, `test/srt/nightly/test_deepseek_v31_perf.py` |
| 2025-11-14 | [#11589](https://github.com/sgl-project/sglang/pull/11589) | merged | [Tool Call] Steamline function arguments when tool_choice="auto" for deepseekv31_detector | `python/sglang/srt/function_call/deepseekv31_detector.py` |
| 2025-11-26 | [#13954](https://github.com/sgl-project/sglang/pull/13954) | merged | Fix Deepseek v3.1 loading issue | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-12-10 | [#14837](https://github.com/sgl-project/sglang/pull/14837) | merged | [Auto Sync] Update tool_chat_template_deepseekv31.jinja (20251210) | `examples/chat_template/tool_chat_template_deepseekv31.jinja` |
| 2025-12-31 | [#13394](https://github.com/sgl-project/sglang/pull/13394) | merged | Fix DeepSeekV31's structural tag trigger | `python/sglang/srt/function_call/deepseekv31_detector.py` |
| 2026-01-07 | [#16660](https://github.com/sgl-project/sglang/pull/16660) | merged | [CI] Enable dpsk v31 test on nightly H200 | `test/registered/8-gpu-models/test_deepseek_v31.py` |
| 2026-01-16 | [#17178](https://github.com/sgl-project/sglang/pull/17178) | merged | Remove deepseek-r1 from THINKING_MODE_CHOICES in run_eval.py | `python/sglang/test/run_eval.py` |
| 2026-01-16 | [#17133](https://github.com/sgl-project/sglang/pull/17133) | merged | [DeepSeek V3.1/V3.2] Optimize fused moe configs for H20 & H20-3E based on swapab | `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json` |
| 2026-01-19 | [#17320](https://github.com/sgl-project/sglang/pull/17320) | closed | fix: Fixed the issue where "finish_reason":"stop" appeared when calling the tool and the tool was in the content. | `python/sglang/srt/function_call/deepseekv32_detector.py`, `examples/chat_template/tool_chat_template_deepseekv32.jinja` |
| 2026-01-22 | [#17141](https://github.com/sgl-project/sglang/pull/17141) | closed | fix: Fixed the issue where "finish_reason":"stop" appeared when calling the tool and the tool was in the content. | `python/sglang/srt/function_call/deepseekv32_detector.py`, `examples/chat_template/tool_chat_template_deepseekv32.jinja` |
| 2026-01-24 | [#17558](https://github.com/sgl-project/sglang/pull/17558) | closed | fix: Fixed the issue where "finish_reason":"stop" appeared when calling the tool and the tool was in the content. | `python/sglang/srt/function_call/deepseekv32_detector.py`, `examples/chat_template/tool_chat_template_deepseekv32.jinja` |
| 2026-01-26 | [#17761](https://github.com/sgl-project/sglang/pull/17761) | open | fix: missing Assistant token after tool output in DeepSeek v3.1/v3.2 chat templates | `test/manual/test_deepseek_chat_templates.py`, `examples/chat_template/tool_chat_template_deepseekv31.jinja`, `examples/chat_template/tool_chat_template_deepseekv32.jinja` |
| 2026-02-04 | [#18236](https://github.com/sgl-project/sglang/pull/18236) | open | Fix function call arguments missing in streaming mode for DeepSeek V3.1 | `python/sglang/srt/function_call/deepseekv31_detector.py` |
| 2026-03-31 | [#21739](https://github.com/sgl-project/sglang/pull/21739) | open | [NPU] Update DeepSeek-V3.1 and DeepSeek-V3.2 model deployment instructions in documentation | `docs/platforms/ascend/ascend_npu_best_practice.md` |
| 2026-04-09 | [#22433](https://github.com/sgl-project/sglang/pull/22433) | open | [Test] Add unit tests for DeepSeekV31Detector | `test/registered/unit/function_call/test_deepseekv31_detector.py` |
| 2026-04-11 | [#21593](https://github.com/sgl-project/sglang/pull/21593) | merged | Fix tool call constrained decoding and parsing for models with native formats | `test/registered/unit/function_call/test_function_call_parser.py`, `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/function_call/function_call_parser.py` |
| 2026-04-16 | [#22981](https://github.com/sgl-project/sglang/pull/22981) | open | [Test] Add unit tests for 7 missing function call detectors | `test/registered/unit/function_call/test_function_call_parser.py`, `test/registered/openai_server/function_call/test_tool_choice.py`, `test/registered/unit/function_call/test_kimik2_detector.py` |
| 2026-04-17 | [#22128](https://github.com/sgl-project/sglang/pull/22128) | merged | Allow piecewise CUDA graph with speculative decoding | `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`, `python/sglang/srt/model_executor/model_runner.py`, `test/registered/piecewise_cuda_graph/test_pcg_with_speculative_decoding.py` |
| 2026-04-20 | [#21599](https://github.com/sgl-project/sglang/pull/21599) | merged | [SPEC][1/N] feat: add adaptive speculative_num_steps for EAGLE topk=1 | `python/sglang/srt/model_executor/cuda_graph_runner.py`, `benchmark/bench_adaptive_speculative.py`, `test/registered/unit/spec/test_adaptive_spec_params.py` |
| 2026-04-21 | [#23315](https://github.com/sgl-project/sglang/pull/23315) | merged | Opt-in strip of thinking tokens from radix cache | `test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py`, `python/sglang/srt/managers/schedule_batch.py`, `python/sglang/srt/server_args.py` |
| 2026-04-21 | [#22950](https://github.com/sgl-project/sglang/pull/22950) | closed | [fix] Parser-gated two-phase cache stripping for reasoning radix caches (fixes #22373) | `python/sglang/srt/parser/reasoning_parser.py`, `python/sglang/srt/configs/model_config.py`, `test/registered/unit/mem_cache/test_radix_cache_thinking.py` |
| 2026-04-24 | [#22774](https://github.com/sgl-project/sglang/pull/22774) | merged | [MUSA][16/N] Add MUSA backend support for layers and DeepSeek models (V2/V3/R1) | `python/sglang/srt/layers/layernorm.py`, `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py` |
| 2026-04-26 | [#23732](https://github.com/sgl-project/sglang/pull/23732) | merged | Apply should_use_dp_reduce_scatterv guard to remaining MoE models (follow-up to #23731) | `python/sglang/srt/models/llada2.py`, `python/sglang/srt/models/hunyuan_v3.py`, `python/sglang/srt/models/bailing_moe_linear.py` |
| 2026-04-27 | [#23748](https://github.com/sgl-project/sglang/pull/23748) | merged | refactor(moe): centralize post-experts all-reduce skip predicate | `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/models/sarvam_moe.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2026-04-28 | [#23268](https://github.com/sgl-project/sglang/pull/23268) | merged | 【NPU】【bugfix】accuracy fix when enable both nsa cp and prefixcache | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` |
| 2026-04-28 | [#23285](https://github.com/sgl-project/sglang/pull/23285) | merged | [Flashinfer] Integrate flashinfer router gemm for sm103 | `python/sglang/srt/models/deepseek_v2.py` |
| 2026-04-29 | [#21062](https://github.com/sgl-project/sglang/pull/21062) | merged | Use spec v2 by default | `test/registered/4-gpu-models/test_qwen35_models.py`, `test/registered/ep/test_deepep_large.py`, `test/registered/spec/eagle/test_adaptive_speculative.py` |
| 2026-04-30 | [#21126](https://github.com/sgl-project/sglang/pull/21126) | merged | [4/N] Quantization Refactor: AWQ schemes and Kernel call and weight init split | `python/sglang/srt/layers/quantization/awq.py`, `python/sglang/srt/layers/quantization/awq/awq.py`, `python/sglang/srt/layers/quantization/awq/schemes/awq_moe.py` |
| 2026-04-30 | [#23557](https://github.com/sgl-project/sglang/pull/23557) | merged | [Intel GPU] Integrate flash_mla_decode in Intel XPU attention backend | `python/sglang/srt/layers/attention/xpu_backend.py`, `python/sglang/srt/models/deepseek_common/attention_backend_handler.py`, `python/sglang/srt/model_executor/model_runner.py` |
| 2026-05-02 | [#23850](https://github.com/sgl-project/sglang/pull/23850) | merged | Support RunAI loading for quantized checkpoints | `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`, `test/registered/unit/model_loader/test_runai_model_streamer_loader.py` |
| 2026-05-02 | [#21247](https://github.com/sgl-project/sglang/pull/21247) | merged | [Dependency] Upgrade to Torch 2.11.0 | `.github/workflows/pr-test-multimodal-gen.yml`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/entrypoints/engine.py` |
| 2026-05-05 | [#24392](https://github.com/sgl-project/sglang/pull/24392) | merged | add indexer-topk capture (V3.2 NSA + infra) | `python/sglang/srt/layers/attention/indexer_topk_capturer.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-05-05 | [#24450](https://github.com/sgl-project/sglang/pull/24450) | merged | move topk capturers to srt/state_capturer/ | `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` |
| 2026-05-07 | [#24005](https://github.com/sgl-project/sglang/pull/24005) | merged | [AMD] Enable dual-stream MoE on ROCm | `python/sglang/srt/layers/moe/token_dispatcher/moriep.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/environ.py` |
| 2026-05-08 | [#23882](https://github.com/sgl-project/sglang/pull/23882) | merged | Deepseek V4 | `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/entrypoints/openai/encoding_dsv4.py` |
| 2026-05-08 | [#23336](https://github.com/sgl-project/sglang/pull/23336) | merged | [SPEC V2][2/N] feat: adaptive spec support spec v2 | `python/sglang/srt/speculative/eagle_worker_v2.py`, `test/registered/spec/eagle/test_adaptive_speculative.py`, `python/sglang/srt/speculative/eagle_worker.py` |
| 2026-05-11 | [#24799](https://github.com/sgl-project/sglang/pull/24799) | merged | [AMD] Fix DeepSeek import cascade by supporting both pre- and post-#2958 aiter `fused_qk_rmsnorm` APIs | `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` |
| 2026-05-12 | [#24949](https://github.com/sgl-project/sglang/pull/24949) | merged | Deepseek-v4-Pro share expert tp1 | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/environ.py` |
| 2026-05-13 | [#25120](https://github.com/sgl-project/sglang/pull/25120) | merged | [env] Make max KV chunk capacity configurable via `SGLANG_MAX_KV_CHUNK_CAPACITY` | `python/sglang/srt/model_executor/forward_batch_deepseek_mha_mixin.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`, `docs_new/docs/references/environment_variables.mdx` |
| 2026-05-13 | [#24148](https://github.com/sgl-project/sglang/pull/24148) | merged | [AMD] Add _skip_rope_for_aiter_fused_mla method and check to avoid double rotating with gfx950 and Aiter backend | `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` |
| 2026-05-13 | [#24125](https://github.com/sgl-project/sglang/pull/24125) | merged | [AMD] Skip redundant CatArrayBatchedCopy in GLM-5 NSA TileLang decode | `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2026-05-13 | [#24897](https://github.com/sgl-project/sglang/pull/24897) | merged | Port fused SiLU+clamp+FP8 quant from DSV4 dev branch | `python/sglang/srt/models/deepseek_v2.py` |
| 2026-05-13 | [#25001](https://github.com/sgl-project/sglang/pull/25001) | merged | [LoRA] MLA attention LoRA: q_b_proj / kv_b_proj support | `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/lora/triton_ops/kv_b_lora_absorbed.py` |
| 2026-05-13 | [#25182](https://github.com/sgl-project/sglang/pull/25182) | merged | chore: add vLLM SPDX copyright headers to ported files | `python/sglang/srt/models/baichuan.py`, `python/sglang/srt/models/commandr.py`, `python/sglang/srt/models/dbrx.py` |
| 2026-05-14 | [#24925](https://github.com/sgl-project/sglang/pull/24925) | merged | [attn backend] Integrate tokenspeed_mla prefill/decode kernels (fp8 kv cache, blackwell) | `python/sglang/srt/layers/attention/tokenspeed_mla_backend.py`, `python/sglang/srt/layers/attention/trtllm_mla_backend.py`, `python/sglang/srt/layers/attention/attention_registry.py` |
| 2026-05-14 | [#19290](https://github.com/sgl-project/sglang/pull/19290) | merged | feat: [2/2][DeepEP] Add waterfill load balancing for shared expert dispatch | `python/sglang/srt/layers/moe/deepep_waterfill.py`, `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/model_executor/model_runner.py` |
| 2026-05-15 | [#25279](https://github.com/sgl-project/sglang/pull/25279) | merged | DeepseekV2MoE: defer shared experts when routed kernel is non-mutating | `python/sglang/srt/models/deepseek_v2.py` |
| 2026-05-15 | [#25333](https://github.com/sgl-project/sglang/pull/25333) | merged | perf(mla): hybrid Triton fused cat+FP8-quantize for MLA chunked-prefill K/V | `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`, `python/sglang/jit_kernel/mla_kv_pack_quantize_fp8.py`, `python/sglang/jit_kernel/benchmark/bench_mla_kv_pack_quantize_fp8.py` |
| 2026-05-15 | [#25379](https://github.com/sgl-project/sglang/pull/25379) | merged | feat(moe): reuse prev-layer output as symm_output for FP4 routed MoE | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py`, `python/sglang/srt/layers/moe/moe_runner/base.py` |
| 2026-05-16 | [#25406](https://github.com/sgl-project/sglang/pull/25406) | merged | [MoE] Decouple Mega MoE from DeepEP backend | `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/layers/moe/mega_moe.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` |
| 2026-05-18 | [#25390](https://github.com/sgl-project/sglang/pull/25390) | merged | [AMD] Enable shared-experts fusion with new KIMI-K2.5-MXFP4 model. | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/quantization/quark/quark.py` |
| 2026-05-18 | [#24933](https://github.com/sgl-project/sglang/pull/24933) | merged | Amd/deepseek v4 rebase main 0509 | `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/dsv4/compress_hip.py` |
| 2026-05-18 | [#25454](https://github.com/sgl-project/sglang/pull/25454) | merged | fix(eagle3): drop +1 offset on aux layer ids when first id != 1 | `python/sglang/srt/models/deepseek_v2.py` |
| 2026-05-19 | [#24640](https://github.com/sgl-project/sglang/pull/24640) | merged | Support spec v2 for FlashMLA speculative decoding | `python/sglang/srt/layers/attention/flashmla_backend.py`, `python/sglang/srt/models/deepseek_v2.py`, `test/registered/mla/test_flashmla.py` |
| 2026-05-20 | [#25821](https://github.com/sgl-project/sglang/pull/25821) | merged | [Refactor] Rename NSA → DSA: user-facing aliases, file/class/import rename | `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/dsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2026-05-20 | [#25460](https://github.com/sgl-project/sglang/pull/25460) | merged | [perf] prepare_prefill_qkv hook + fp8 quantize jit kernel | `python/sglang/srt/layers/attention/tokenspeed_mla_backend.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`, `python/sglang/srt/layers/attention/trtllm_mla_backend.py` |
| 2026-05-21 | [#25884](https://github.com/sgl-project/sglang/pull/25884) | merged | [Refactor] major JIT kernel clean up for dsv4 | `python/sglang/srt/layers/attention/dsv4/compressor.py`, `python/sglang/srt/layers/attention/dsv4/metadata.py`, `python/sglang/srt/layers/moe/moe_runner/deep_gemm.py` |
| 2026-05-21 | [#25974](https://github.com/sgl-project/sglang/pull/25974) | merged | [Fix]: Restrict Kimi-K2.5 shared-experts fusion to Quark MXFP4 checkpoints | `python/sglang/srt/models/deepseek_v2.py` |
| 2026-05-21 | [#25983](https://github.com/sgl-project/sglang/pull/25983) | merged | feat(model_runner): remove pool/backend refs from ForwardBatch via ForwardContext | `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`, `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` |
| 2026-05-22 | [#25189](https://github.com/sgl-project/sglang/pull/25189) | merged | [perf] DeepSeekV3: drop redundant FP32 upcasts in trtllm MoE paths | `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2026-05-22 | [#23351](https://github.com/sgl-project/sglang/pull/23351) | merged | Support piecewise CUDA graph with NSA | `python/sglang/srt/layers/attention/dsa/dsa_indexer.py`, `python/sglang/srt/layers/layernorm.py`, `python/sglang/srt/layers/attention/dsa_backend.py` |
| 2026-05-23 | [#25843](https://github.com/sgl-project/sglang/pull/25843) | merged | Route concat MLA to JIT and remove unused downcast | `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`, `python/sglang/srt/models/sarvam_moe.py`, `python/sglang/srt/layers/attention/utils.py` |
| 2026-05-23 | [#23292](https://github.com/sgl-project/sglang/pull/23292) | merged | [CP] 1/N: Support MLA Prefill Context Parallel | `python/sglang/srt/layers/attention/flashattention_backend.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/utils/cp_utils.py` |
| 2026-05-23 | [#25898](https://github.com/sgl-project/sglang/pull/25898) | merged | [AMD] Dsv4/pr1 fix run time issue | `python/sglang/srt/layers/fused_qk_norm_rope_store.py`, `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/layers/attention/dsv4/compress_hip.py` |
| 2026-05-26 | [#26208](https://github.com/sgl-project/sglang/pull/26208) | merged | [AMD] Dsv4/pr2 compressor opt | `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_fused.py`, `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_dsv4.py`, `python/sglang/srt/layers/attention/dsv4/fused_compress_triton.py` |
| 2026-05-27 | [#23269](https://github.com/sgl-project/sglang/pull/23269) | merged | Support batch size > 1 when enable CP | `python/sglang/srt/layers/utils/cp_utils.py`, `python/sglang/srt/layers/attention/dsa/dsa_indexer.py`, `python/sglang/srt/model_executor/forward_batch_info.py` |
| 2026-05-28 | [#24737](https://github.com/sgl-project/sglang/pull/24737) | merged | Support Flashinfer Cute-DSL MLA attention | `python/sglang/srt/layers/attention/trtllm_mla_backend.py`, `python/sglang/srt/layers/attention/attention_registry.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` |
| 2026-05-29 | [#25755](https://github.com/sgl-project/sglang/pull/25755) | merged | [Fix][NPU] Preserve existing packed_modules_mapping when merging model-level fused module mappings | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/quantization/base_config.py`, `python/sglang/srt/layers/quantization/modelslim/modelslim.py` |
| 2026-05-29 | [#25463](https://github.com/sgl-project/sglang/pull/25463) | merged | [ROCm] Eliminate redundant contiguous copy in MLA attention on ROCm MXFP4 | `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` |
| 2026-05-29 | [#26673](https://github.com/sgl-project/sglang/pull/26673) | merged | [refactor] remove unused op_mlp | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_lite.py` |
| 2026-05-29 | [#26626](https://github.com/sgl-project/sglang/pull/26626) | merged | [perf] Fuse NVFP4 gate_up_gemm + swiglu + output FP4 quant | `python/sglang/srt/layers/quantization/nvfp4_gemm_swiglu_nvfp4_quant.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2026-06-02 | [#26970](https://github.com/sgl-project/sglang/pull/26970) | merged | [perf] Replicate embed_tokens to drop the post-embed all-reduce | `python/sglang/srt/layers/vocab_parallel_embedding.py`, `python/sglang/srt/models/deepseek_nextn.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2026-06-05 | [#27329](https://github.com/sgl-project/sglang/pull/27329) | merged | [LoRA] Experimental fast LoRA path with `experimental_sgl_trtllm` MoE backend for FP8 and NVFP4 models | `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/layers/moe/moe_runner/triton_utils/moe_align_block_size.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` |
| 2026-06-05 | [#27150](https://github.com/sgl-project/sglang/pull/27150) | merged | Support Waterfill with dynamic EPLB | `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/models/deepseek_v2.py`, `test/registered/unit/eplb/test_deepep_waterfill_eplb.py` |
| 2026-06-06 | [#27114](https://github.com/sgl-project/sglang/pull/27114) | merged | [Bugfix] Restore overridden HF config fields and support index_skip_topk_offset for DSA topk sharing | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/models/deepseek_nextn.py` |
| 2026-06-08 | [#27289](https://github.com/sgl-project/sglang/pull/27289) | merged | [ROCm] dsv4: remove the redundant fp8 scale transpose-copy on decode | `python/sglang/srt/layers/quantization/fp8_utils.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`, `python/sglang/srt/layers/communicator.py` |
| 2026-06-10 | [#23906](https://github.com/sgl-project/sglang/pull/23906) | merged | [Refactor] Cuda Graph Runner/Backend Refactor | `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` |
| 2026-06-10 | [#27510](https://github.com/sgl-project/sglang/pull/27510) | merged | [deepseek] Enable DP attention + TBO + shared experts fusion | `python/sglang/srt/models/deepseek_v2.py`, `test/registered/ep/test_tbo_shared_experts_fusion.py`, `python/sglang/srt/batch_overlap/two_batch_overlap.py` |
| 2026-06-12 | [#27956](https://github.com/sgl-project/sglang/pull/27956) | merged | Use the correct wrapper for `fp4_quantize` | `python/sglang/srt/models/deepseek_v2.py` |
| 2026-06-13 | [#27720](https://github.com/sgl-project/sglang/pull/27720) | merged | [DeepSeek V3] Defer moe finalize and fused it with main stream add | `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` |
| 2026-06-13 | [#28129](https://github.com/sgl-project/sglang/pull/28129) | merged | [Spec] Remove deprecated EAGLE v1 DRAFT_EXTEND forward mode | `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/model_executor/forward_batch_info.py`, `python/sglang/srt/layers/attention/triton_backend.py` |
| 2026-06-15 | [#28118](https://github.com/sgl-project/sglang/pull/28118) | merged | 【bugfix】The NPU's forward_dsa_prepare_npu also needs special handling for is_nextn | `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` |
| 2026-06-16 | [#24515](https://github.com/sgl-project/sglang/pull/24515) | merged | LPLB: linear-programming load balancer for MoE expert parallelism | `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/layers/moe/hash_topk.py` |
| 2026-06-17 | [#28436](https://github.com/sgl-project/sglang/pull/28436) | merged | [NPU] Use use_dsa to dispatch Ascend DSA attention | `python/sglang/srt/models/deepseek_common/attention_backend_handler.py` |
| 2026-06-17 | [#27798](https://github.com/sgl-project/sglang/pull/27798) | merged | [AMD] Add transpose_scale arg for o_proj to fix GLM accuracy issue | `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` |
| 2026-06-17 | [#28343](https://github.com/sgl-project/sglang/pull/28343) | merged | [Kimi K2.5] Fix eagle3 aux capture for tp>1 when AR fusion is enabled | `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2026-06-18 | [#28567](https://github.com/sgl-project/sglang/pull/28567) | merged | Add get_parallel(): a structured accessor for parallel-topology state | `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py` |
| 2026-06-18 | [#25144](https://github.com/sgl-project/sglang/pull/25144) | merged | [NPU] Add Ascend NPU support for DeepSeek-V4 | `python/sglang/srt/layers/deepseek_v4_rope.py`, `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/layers/mhc.py` |
| 2026-06-18 | [#28559](https://github.com/sgl-project/sglang/pull/28559) | merged | fix: speculative draft worker clobbering target attention backend | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/layers/attention/base_attn_backend.py` |
| 2026-06-19 | [#28532](https://github.com/sgl-project/sglang/pull/28532) | merged | Fix IndexCache PP topk handoff | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/model_executor/model_runner.py` |
| 2026-06-21 | [#28785](https://github.com/sgl-project/sglang/pull/28785) | merged | Pass DSA topk through PP warmup proxy buffers | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/model_executor/runner/base_runner.py`, `python/sglang/srt/model_executor/runner/eager_runner.py` |
| 2026-06-23 | [#28938](https://github.com/sgl-project/sglang/pull/28938) | merged | [AMD] Improve performance of dsv4 in high concurrency | `python/sglang/srt/layers/deepseek_v4_rope.py`, `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2026-06-24 | [#27833](https://github.com/sgl-project/sglang/pull/27833) | merged | [AMD] Enable BCG on ROCm + route aiter prefill via MHA during PCG/BCG capture for Kimi-2.5 | `python/sglang/srt/models/deepseek_common/attention_backend_handler.py`, `test/registered/amd/test_kimi_k25_mxfp4_bcg_mi35x.py` |
| 2026-06-24 | [#27053](https://github.com/sgl-project/sglang/pull/27053) | merged | [BCG][GLM5] perf: BCG support and prefill enhancements | `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/layers/attention/dsa/dsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2026-06-25 | [#29042](https://github.com/sgl-project/sglang/pull/29042) | merged | [NPU] Fix the DeepSeek-V2-Coder model accuracy issue | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/llada2.py`, `python/sglang/srt/hardware_backend/npu/moe/topk.py` |
| 2026-06-25 | [#14194](https://github.com/sgl-project/sglang/pull/14194) | merged | [feature] implement dcp for deepseek_v2 | `python/sglang/srt/layers/utils/dcp_utils.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/layers/attention/flashinfer_mla_backend.py` |
| 2026-06-26 | [#29142](https://github.com/sgl-project/sglang/pull/29142) | merged | [DeepSeek V3] Run routed experts on main stream in dual-stream MoE | `python/sglang/srt/models/deepseek_v2.py` |

## Per-PR Diff Audit Cards

### PR #9464 - Add deepseek v3.1 thinking parser support and update docs

- Link: https://github.com/sgl-project/sglang/pull/9464
- Status/date: merged / 2025-08-21
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +136/-78, 245 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add deepseek v3.1 thinking parser support and update docs"; model line: DeepSeek V3.1; category: docs/tests/CI; main diff: `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/reasoning_parser.py`; technical summary: Covers "Add deepseek v3.1 thinking parser support and update docs"; the main implementation surface is `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/reasoning_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +9/-6 (15 lines); hunks: -859,12 +859,15 @@ def _get_enable_thinking_from_request(self, request: ChatC...; symbols: _get_enable_thinking_from_request, _process_tool_call_stream, touching `_get_enable_thinking_from_request, _process_tool_call_stream`; `python/sglang/srt/reasoning_parser.py` modified +4/-3 (7 lines); hunks: -513,12 +513,13 @@ class ReasoningParser:; symbols: ReasoningParser, __init__, touching `ReasoningParser, __init__`.
- Code diff details:
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +9/-6 (15 lines); hunks: -859,12 +859,15 @@ def _get_enable_thinking_from_request(self, request: ChatC...; symbols: _get_enable_thinking_from_request, _process_tool_call_stream
  - `python/sglang/srt/reasoning_parser.py` modified +4/-3 (7 lines); hunks: -513,12 +513,13 @@ class ReasoningParser:; symbols: ReasoningParser, __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/entrypoints/openai/serving_chat.py
@@ -859,12 +859,15 @@ def _get_enable_thinking_from_request(self, request: ChatCompletionRequest) -> b
-        if (
-            hasattr(request, "chat_template_kwargs")
-            and request.chat_template_kwargs
-            and request.chat_template_kwargs.get("enable_thinking") is not None
-        ):
-            return request.chat_template_kwargs.get("enable_thinking")
diff -- python/sglang/srt/reasoning_parser.py
@@ -513,12 +513,13 @@ class ReasoningParser:
-        "qwen3": Qwen3Detector,
-        "qwen3-thinking": Qwen3Detector,
+        "deepseek-v3": Qwen3Detector,
+        "gpt-oss": GptOssDetector,
+        "qwen3": Qwen3Detector,
+        "qwen3-thinking": Qwen3Detector,
```

- Reviewed files:
  - runtime: `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +9/-6; `python/sglang/srt/reasoning_parser.py` modified +4/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/reasoning_parser.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9544 - [doc] deepseekv31 support

- Link: https://github.com/sgl-project/sglang/pull/9544
- Status/date: merged / 2025-08-23
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +82/-4, 112 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[doc] deepseekv31 support"; model line: DeepSeek V3.1; category: docs/tests/CI; main diff: `benchmark/deepseek_v3/README.md`, `docs/basic_usage/deepseek.md`; technical summary: Covers "[doc] deepseekv31 support"; the main implementation surface is `benchmark/deepseek_v3/README.md`, `docs/basic_usage/deepseek.md`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `benchmark/deepseek_v3/README.md` modified +80/-2 (82 lines); hunks: -1,4 +1,4; -50,7 +50,9 @@ Add performance optimization options as nee; `docs/basic_usage/deepseek.md` modified +2/-2 (4 lines); hunks: -5,9 +5,9 @@ SGLang provides many optimizations specifically designed for the....
- Code diff details:
  - `benchmark/deepseek_v3/README.md` modified +80/-2 (82 lines); hunks: -1,4 +1,4; -50,7 +50,9 @@ Add performance optimization options as nee
  - `docs/basic_usage/deepseek.md` modified +2/-2 (4 lines); hunks: -5,9 +5,9 @@ SGLang provides many optimizations specifically designed for the...
- Key code excerpts:

```diff
diff -- benchmark/deepseek_v3/README.md
@@ -1,4 +1,4 @@
-# DeepSeek V3 Support
+# DeepSeek V3.1/V3/R1 Support
@@ -50,7 +50,9 @@ Add [performance optimization options](#performance-optimization-options) as nee
-### Example: Sending requests with OpenAI API
+### Usage: Chat with DeepSeek
+#### DeepSeek V3/R1
diff -- docs/basic_usage/deepseek.md
@@ -5,9 +5,9 @@ SGLang provides many optimizations specifically designed for the DeepSeek models
-## Launch DeepSeek V3 with SGLang
+## Launch DeepSeek V3.1/V3/R1 with SGLang
-To run DeepSeek V3/R1 models, the requirements are as follows:
+To run DeepSeek V3.1/V3/R1 models, the recommended settings are as follows:
```

- Reviewed files:
  - other: `benchmark/deepseek_v3/README.md` modified +80/-2
  - docs: `docs/basic_usage/deepseek.md` modified +2/-2
- Risk and verification: This is mostly docs/examples in `docs/basic_usage/deepseek.md`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #9446 - Support DeepSeek-V3.1 tool call

- Link: https://github.com/sgl-project/sglang/pull/9446
- Status/date: merged / 2025-08-27
- Trace source: `git log --name-only -- <model-files>` found it through `examples/chat_template/tool_chat_template_deepseekv31.jinja`, `python/sglang/srt/function_call/deepseekv31_detector.py`; associated commits `b9683be6538e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +315/-0, 331 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support DeepSeek-V3.1 tool call"; model line: DeepSeek V3.1; category: model support/runtime entry; main diff: `python/sglang/srt/function_call/deepseekv31_detector.py`, `examples/chat_template/tool_chat_template_deepseekv31.jinja`; technical summary: Covers "Support DeepSeek-V3.1 tool call"; the main implementation surface is `python/sglang/srt/function_call/deepseekv31_detector.py`, `examples/chat_template/tool_chat_template_deepseekv31.jinja`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/function_call/deepseekv31_detector.py` added +222/-0 (222 lines); hunks: -0,0 +1,222; symbols: DeepSeekV31Detector, __init__, has_tool_call, detect_and_parse, touching `DeepSeekV31Detector, __init__, has_tool_call`; `examples/chat_template/tool_chat_template_deepseekv31.jinja` added +91/-0 (91 lines); hunks: -0,0 +1,91.
- Code diff details:
  - `python/sglang/srt/function_call/deepseekv31_detector.py` added +222/-0 (222 lines); hunks: -0,0 +1,222; symbols: DeepSeekV31Detector, __init__, has_tool_call, detect_and_parse
  - `examples/chat_template/tool_chat_template_deepseekv31.jinja` added +91/-0 (91 lines); hunks: -0,0 +1,91
- Key code excerpts:

```diff
diff -- python/sglang/srt/function_call/deepseekv31_detector.py
@@ -0,0 +1,222 @@
+import json
+import logging
+import re
+from typing import List
+from sglang.srt.entrypoints.openai.protocol import Tool
+from sglang.srt.function_call.base_format_detector import BaseFormatDetector
diff -- examples/chat_template/tool_chat_template_deepseekv31.jinja
@@ -0,0 +1,91 @@
+{% if not add_generation_prompt is defined %}
+  {% set add_generation_prompt = false %}
+{% endif %}
+{% if not thinking is defined %}
+  {% set thinking = false %}
+{% endif %}
```

- Reviewed files:
  - runtime: `python/sglang/srt/function_call/deepseekv31_detector.py` added +222/-0
  - docs: `examples/chat_template/tool_chat_template_deepseekv31.jinja` added +91/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/function_call/deepseekv31_detector.py`, `python/sglang/srt/function_call/function_call_parser.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9895 - Update tool_chat_template_deepseekv31.jinja

- Link: https://github.com/sgl-project/sglang/pull/9895
- Status/date: merged / 2025-09-03
- Trace source: `git log --name-only -- <model-files>` found it through `examples/chat_template/tool_chat_template_deepseekv31.jinja`; associated commits `cc9a31c66226`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-3, 17 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Update tool_chat_template_deepseekv31.jinja"; model line: DeepSeek V3.1; category: model implementation change; main diff: `examples/chat_template/tool_chat_template_deepseekv31.jinja`; technical summary: Covers "Update tool_chat_template_deepseekv31.jinja"; the main implementation surface is `examples/chat_template/tool_chat_template_deepseekv31.jinja`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `examples/chat_template/tool_chat_template_deepseekv31.jinja` modified +3/-3 (6 lines); hunks: -43,13 +43,13.
- Code diff details:
  - `examples/chat_template/tool_chat_template_deepseekv31.jinja` modified +3/-3 (6 lines); hunks: -43,13 +43,13
- Key code excerpts:

```diff
diff -- examples/chat_template/tool_chat_template_deepseekv31.jinja
@@ -43,13 +43,13 @@
-          {{'<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>'+ tool['function']['name'] + '<｜tool▁sep｜>' + tool['function']['arguments'] + '<｜tool▁call▁end｜>'}}
+          {{'<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>'+ tool['function']['name'] + '<｜tool▁sep｜>' + tool['function']['arguments']|tojson + '<｜tool▁call▁end｜>'}}
-          {{message['content'] + '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['function']['name'] + '<｜tool▁sep｜>' + tool['function']['arguments'] + '<｜tool▁call▁end｜>'}}
+          {{message['content'] + '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['function']['name'] + '<｜tool▁sep｜>' + tool['function']['arguments']|tojson + '<｜tool▁call▁end｜>'
-        {{'<｜tool▁call▁begin｜>'+ tool['function']['name'] + '<｜tool▁sep｜>' + tool['function']['arguments'] + '<｜tool▁call▁end｜>'}}
+        {{'<｜tool▁call▁begin｜>'+ tool['function']['name'] + '<｜tool▁sep｜>' + tool['function']['arguments']|tojson + '<｜tool▁call▁end｜>'}}
```

- Reviewed files:
  - docs: `examples/chat_template/tool_chat_template_deepseekv31.jinja` modified +3/-3
- Risk and verification: This is mostly docs/examples in `examples/chat_template/tool_chat_template_deepseekv31.jinja`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #10550 - Use jsonschema to constrain required or specific tool choice

- Link: https://github.com/sgl-project/sglang/pull/10550
- Status/date: merged / 2025-09-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 12 files, +1558/-50, 1876 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Use jsonschema to constrain required or specific tool choice"; model line: DeepSeek V3.1; category: docs/tests/CI; main diff: `test/srt/function_call/test_json_schema_constraint.py`, `test/srt/openai_server/function_call/test_tool_choice.py`, `test/srt/test_function_call_parser.py`; technical summary: Covers "Use jsonschema to constrain required or specific tool choice"; the main implementation surface is `test/srt/function_call/test_json_schema_constraint.py`, `test/srt/openai_server/function_call/test_tool_choice.py`, `test/srt/test_function_call_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/srt/function_call/test_json_schema_constraint.py` added +618/-0 (618 lines); hunks: -0,0 +1,618; symbols: TestJsonSchemaConstraint, setUp, test_required_tool_choice_schema, test_specific_tool_choice_schema, touching `TestJsonSchemaConstraint, setUp, test_required_tool_choice_schema`; `test/srt/openai_server/function_call/test_tool_choice.py` modified +319/-14 (333 lines); hunks: -343,6 +343,142 @@ def test_tool_choice_specific_function_streaming(self):; -408,6 +544,10 @@ def test_multi_tool_scenario_required(self):; symbols: test_tool_choice_specific_function_streaming, test_required_streaming_arguments_chunks_json, test_complex_parameters_required_non_streaming, test_multi_tool_scenario_auto, touching `test_tool_choice_specific_function_streaming, test_required_streaming_arguments_chunks_json, test_complex_parameters_required_non_streaming`; `test/srt/test_function_call_parser.py` modified +319/-0 (319 lines); hunks: -5,8 +5,10; -2190,5 +2192,322 @@ def test_partial_tool_call(self):; symbols: test_partial_tool_call, TestJsonArrayParser, setUp, test_json_detector_ebnf, touching `test_partial_tool_call, TestJsonArrayParser, setUp`; `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +115/-22 (137 lines); hunks: -9,6 +9,7; -25,6 +26,8; symbols: _validate_request, _process_messages, _build_sampling_params, _build_chat_response, touching `_validate_request, _process_messages, _build_sampling_params`.
- Code diff details:
  - `test/srt/function_call/test_json_schema_constraint.py` added +618/-0 (618 lines); hunks: -0,0 +1,618; symbols: TestJsonSchemaConstraint, setUp, test_required_tool_choice_schema, test_specific_tool_choice_schema
  - `test/srt/openai_server/function_call/test_tool_choice.py` modified +319/-14 (333 lines); hunks: -343,6 +343,142 @@ def test_tool_choice_specific_function_streaming(self):; -408,6 +544,10 @@ def test_multi_tool_scenario_required(self):; symbols: test_tool_choice_specific_function_streaming, test_required_streaming_arguments_chunks_json, test_complex_parameters_required_non_streaming, test_multi_tool_scenario_auto
  - `test/srt/test_function_call_parser.py` modified +319/-0 (319 lines); hunks: -5,8 +5,10; -2190,5 +2192,322 @@ def test_partial_tool_call(self):; symbols: test_partial_tool_call, TestJsonArrayParser, setUp, test_json_detector_ebnf
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +115/-22 (137 lines); hunks: -9,6 +9,7; -25,6 +26,8; symbols: _validate_request, _process_messages, _build_sampling_params, _build_chat_response
  - `python/sglang/srt/function_call/utils.py` modified +96/-5 (101 lines); hunks: -1,10 +1,13; -37,10 +40,12 @@ def _partial_json_loads(input_str: str, flags: Allow) -> Tup...; symbols: _find_common_prefix, _partial_json_loads, _is_complete_json, _get_tool_schema_defs
- Key code excerpts:

```diff
diff -- test/srt/function_call/test_json_schema_constraint.py
@@ -0,0 +1,618 @@
+"""
+Tests for JSON schema constraint functionality used by JsonArrayParser
+"""
+import json
+import unittest
+import jsonschema
diff -- test/srt/openai_server/function_call/test_tool_choice.py
@@ -343,6 +343,142 @@ def test_tool_choice_specific_function_streaming(self):
+    def test_required_streaming_arguments_chunks_json(self):
+        """In streaming required mode, complete tool call arguments should be valid JSON when all chunks are combined"""
+        tools = self.get_test_tools()
+        messages = self.get_test_messages()
+        response = self.client.chat.completions.create(
+            model=self.model_name,
diff -- test/srt/test_function_call_parser.py
@@ -5,8 +5,10 @@
```

- Reviewed files:
  - tests: `test/srt/function_call/test_json_schema_constraint.py` added +618/-0; `test/srt/openai_server/function_call/test_tool_choice.py` modified +319/-14; `test/srt/test_function_call_parser.py` modified +319/-0; `test/srt/openai_server/function_call/test_openai_function_calling.py` modified +4/-4
  - runtime: `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +115/-22; `python/sglang/srt/function_call/utils.py` modified +96/-5; `python/sglang/srt/function_call/json_array_parser.py` added +63/-0; `python/sglang/srt/entrypoints/openai/protocol.py` modified +12/-2
- Risk and verification: The diff ships test coverage in `test/srt/function_call/test_json_schema_constraint.py`, `test/srt/openai_server/basic/test_serving_chat.py`, `test/srt/openai_server/function_call/test_openai_function_calling.py`, `test/srt/openai_server/function_call/test_tool_choice.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #10875 - feat(reasoning): improve enable thinking from request

- Link: https://github.com/sgl-project/sglang/pull/10875
- Status/date: merged / 2025-09-29
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +8/-10, 54 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "feat(reasoning): improve enable thinking from request"; model line: DeepSeek V3.1; category: model support/runtime entry; main diff: `python/sglang/srt/entrypoints/openai/serving_chat.py`; technical summary: Covers "feat(reasoning): improve enable thinking from request"; the main implementation surface is `python/sglang/srt/entrypoints/openai/serving_chat.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +8/-10 (18 lines); hunks: -64,6 +64,7 @@ def __init__(; -563,10 +564,7 @@ async def _generate_chat_stream(; symbols: __init__, _request_id_prefix, _generate_chat_stream, _build_chat_response, touching `__init__, _request_id_prefix, _generate_chat_stream`.
- Code diff details:
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +8/-10 (18 lines); hunks: -64,6 +64,7 @@ def __init__(; -563,10 +564,7 @@ async def _generate_chat_stream(; symbols: __init__, _request_id_prefix, _generate_chat_stream, _build_chat_response
- Key code excerpts:

```diff
diff -- python/sglang/srt/entrypoints/openai/serving_chat.py
@@ -64,6 +64,7 @@ def __init__(
+        self.reasoning_parser = self.tokenizer_manager.server_args.reasoning_parser
@@ -563,10 +564,7 @@ async def _generate_chat_stream(
-                if (
-                    self.tokenizer_manager.server_args.reasoning_parser
-                    and request.separate_reasoning
-                ):
```

- Reviewed files:
  - runtime: `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +8/-10
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/entrypoints/openai/serving_chat.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #11189 - Add --thinking-mode to run_eval

- Link: https://github.com/sgl-project/sglang/pull/11189
- Status/date: merged / 2025-10-03
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +29/-1, 81 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add --thinking-mode to run_eval"; model line: DeepSeek V3.1; category: docs/tests/CI; main diff: `python/sglang/test/run_eval.py`, `python/sglang/test/simple_eval_common.py`; technical summary: Covers "Add --thinking-mode to run_eval"; the main implementation surface is `python/sglang/test/run_eval.py`, `python/sglang/test/simple_eval_common.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/test/run_eval.py` modified +25/-0 (25 lines); hunks: -16,13 +16,29; -136,6 +152,8 @@ def run_eval(args):; symbols: get_thinking_kwargs, run_eval_once, run_eval, touching `get_thinking_kwargs, run_eval_once, run_eval`; `python/sglang/test/simple_eval_common.py` modified +4/-1 (5 lines); hunks: -93,6 +93,7 @@ def __init__(; -104,9 +105,10 @@ def __init__(; symbols: __init__, _handle_image, __call__, touching `__init__, _handle_image, __call__`.
- Code diff details:
  - `python/sglang/test/run_eval.py` modified +25/-0 (25 lines); hunks: -16,13 +16,29; -136,6 +152,8 @@ def run_eval(args):; symbols: get_thinking_kwargs, run_eval_once, run_eval
  - `python/sglang/test/simple_eval_common.py` modified +4/-1 (5 lines); hunks: -93,6 +93,7 @@ def __init__(; -104,9 +105,10 @@ def __init__(; symbols: __init__, _handle_image, __call__
- Key code excerpts:

```diff
diff -- python/sglang/test/run_eval.py
@@ -16,13 +16,29 @@
+def get_thinking_kwargs(args):
+    if args.thinking_mode in THINKING_MODE_CHOICES:
+        thinking_param = (
+            "thinking" if args.thinking_mode == "deepseek-v3" else "enable_thinking"
+        )
+        return {
diff -- python/sglang/test/simple_eval_common.py
@@ -93,6 +93,7 @@ def __init__(
+        extra_body: Optional[Dict[str, Any]] = None,
@@ -104,9 +105,10 @@ def __init__(
+        self.extra_body = extra_body
-            f"ChatCompletionSampler initialized with {self.system_message=} {self.temperature=} {self.max_tokens=} {self.reasoning_effort=}"
+            f"ChatCompletionSampler initialized with {self.system_message=} {self.temperature=} {self.max_tokens=} {self.reasoning_effort=} {self.extra_body=}"
@@ -144,6 +146,7 @@ def __call__(self, message_list: MessageList) -> str:
```

- Reviewed files:
  - tests: `python/sglang/test/run_eval.py` modified +25/-0; `python/sglang/test/simple_eval_common.py` modified +4/-1
- Risk and verification: The diff ships test coverage in `python/sglang/test/run_eval.py`, `python/sglang/test/simple_eval_common.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #11223 - Update tool parser and related documentation

- Link: https://github.com/sgl-project/sglang/pull/11223
- Status/date: merged / 2025-10-07
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +24/-12, 65 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Update tool parser and related documentation"; model line: DeepSeek V3.1; category: docs/tests/CI; main diff: `python/sglang/srt/function_call/function_call_parser.py`, `python/sglang/srt/server_args.py`; technical summary: Covers "Update tool parser and related documentation"; the main implementation surface is `python/sglang/srt/function_call/function_call_parser.py`, `python/sglang/srt/server_args.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/function_call/function_call_parser.py` modified +8/-6 (14 lines); hunks: -35,17 +35,19 @@ class FunctionCallParser:; symbols: FunctionCallParser, __init__, touching `FunctionCallParser, __init__`; `python/sglang/srt/server_args.py` modified +7/-1 (8 lines); hunks: -527,7 +527,13 @@ def __post_init__(self):; symbols: __post_init__, _handle_deprecated_args, _handle_missing_default_values, touching `__post_init__, _handle_deprecated_args, _handle_missing_default_values`.
- Code diff details:
  - `python/sglang/srt/function_call/function_call_parser.py` modified +8/-6 (14 lines); hunks: -35,17 +35,19 @@ class FunctionCallParser:; symbols: FunctionCallParser, __init__
  - `python/sglang/srt/server_args.py` modified +7/-1 (8 lines); hunks: -527,7 +527,13 @@ def __post_init__(self):; symbols: __post_init__, _handle_deprecated_args, _handle_missing_default_values
- Key code excerpts:

```diff
diff -- python/sglang/srt/function_call/function_call_parser.py
@@ -35,17 +35,19 @@ class FunctionCallParser:
-        "llama3": Llama32Detector,
-        "qwen25": Qwen25Detector,
-        "mistral": MistralDetector,
-        "pythonic": PythonicDetector,
+        "glm": Glm4MoeDetector,
+        "glm45": Glm4MoeDetector,
diff -- python/sglang/srt/server_args.py
@@ -527,7 +527,13 @@ def __post_init__(self):
-        pass
+        # handle deprecated tool call parsers
+        deprecated_tool_call_parsers = {"qwen25": "qwen", "glm45": "glm"}
+        if self.tool_call_parser in deprecated_tool_call_parsers:
+            logger.warning(
+                f"The tool_call_parser '{self.tool_call_parser}' is deprecated. Please use '{deprecated_tool_call_parsers[self.tool_call_parser]}' instead."
```

- Reviewed files:
  - runtime: `python/sglang/srt/function_call/function_call_parser.py` modified +8/-6; `python/sglang/srt/server_args.py` modified +7/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/function_call/function_call_parser.py`, `python/sglang/srt/server_args.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #12123 - Fix DeepSeek chat templates to handle tool call arguments type checking (#11700)

- Link: https://github.com/sgl-project/sglang/pull/12123
- Status/date: merged / 2025-10-30
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +331/-9, 380 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix DeepSeek chat templates to handle tool call arguments type checking (#11700)"; model line: DeepSeek V3.1; category: bug fix; main diff: `test/srt/test_deepseek_chat_templates.py`, `examples/chat_template/tool_chat_template_deepseekv3.jinja`, `examples/chat_template/tool_chat_template_deepseekv31.jinja`; technical summary: Covers "Fix DeepSeek chat templates to handle tool call arguments type checking (#11700)"; the main implementation surface is `test/srt/test_deepseek_chat_templates.py`, `examples/chat_template/tool_chat_template_deepseekv3.jinja`, `examples/chat_template/tool_chat_template_deepseekv31.jinja`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/srt/test_deepseek_chat_templates.py` added +319/-0 (319 lines); hunks: -0,0 +1,319; symbols: TestDeepSeekChatTemplateToolCalls, setUpClass, _render_template, test_tool_arguments_as_dict, touching `TestDeepSeekChatTemplateToolCalls, setUpClass, _render_template`; `examples/chat_template/tool_chat_template_deepseekv3.jinja` modified +4/-3 (7 lines); hunks: -47,15 +47,16; `examples/chat_template/tool_chat_template_deepseekv31.jinja` modified +4/-3 (7 lines); hunks: -41,15 +41,16; `examples/chat_template/tool_chat_template_deepseekv32.jinja` modified +4/-3 (7 lines); hunks: -42,15 +42,16.
- Code diff details:
  - `test/srt/test_deepseek_chat_templates.py` added +319/-0 (319 lines); hunks: -0,0 +1,319; symbols: TestDeepSeekChatTemplateToolCalls, setUpClass, _render_template, test_tool_arguments_as_dict
  - `examples/chat_template/tool_chat_template_deepseekv3.jinja` modified +4/-3 (7 lines); hunks: -47,15 +47,16
  - `examples/chat_template/tool_chat_template_deepseekv31.jinja` modified +4/-3 (7 lines); hunks: -41,15 +41,16
  - `examples/chat_template/tool_chat_template_deepseekv32.jinja` modified +4/-3 (7 lines); hunks: -42,15 +42,16
- Key code excerpts:

```diff
diff -- test/srt/test_deepseek_chat_templates.py
@@ -0,0 +1,319 @@
+"""
+Unit tests for DeepSeek chat template tool call handling.
+Tests verify that the DeepSeek chat templates (v3, v3.1, v3.2) correctly handle
+both dict and string types for tool['function']['arguments'] without double-escaping,
+addressing issue #11700.
+"""
diff -- examples/chat_template/tool_chat_template_deepseekv3.jinja
@@ -47,15 +47,16 @@
+            {%- set formatted_args = tool['function']['arguments'] if tool['function']['arguments'] is string else tool['function']['arguments']|tojson %}
-                    {{- '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\n' + ''''json' + '\n' + tool['function']['argument
+                    {{- '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\n' + ''''json' + '\n' + formatted_args + '\n' + '`
-                    {{- message['content'] + '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\n' + ''''json' + '\n' + tool[
+                    {{- message['content'] + '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\n' + ''''json' + '\n' + forma
-                {{- '\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\n' + ''''json' + '\n' + tool['function']['arguments']|tojson + '\n'
diff -- examples/chat_template/tool_chat_template_deepseekv31.jinja
@@ -41,15 +41,16 @@
```

- Reviewed files:
  - tests: `test/srt/test_deepseek_chat_templates.py` added +319/-0
  - docs: `examples/chat_template/tool_chat_template_deepseekv3.jinja` modified +4/-3; `examples/chat_template/tool_chat_template_deepseekv31.jinja` modified +4/-3; `examples/chat_template/tool_chat_template_deepseekv32.jinja` modified +4/-3
- Risk and verification: The diff ships test coverage in `test/srt/test_deepseek_chat_templates.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #13190 - Remove enable_dp_attention in deepseek nightly tests

- Link: https://github.com/sgl-project/sglang/pull/13190
- Status/date: merged / 2025-11-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +0/-5, 40 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Remove enable_dp_attention in deepseek nightly tests"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `test/srt/nightly/test_deepseek_v32_perf.py`, `test/srt/nightly/test_deepseek_v31_perf.py`; technical summary: Covers "Remove enable_dp_attention in deepseek nightly tests"; the main implementation surface is `test/srt/nightly/test_deepseek_v32_perf.py`, `test/srt/nightly/test_deepseek_v31_perf.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/srt/nightly/test_deepseek_v32_perf.py` modified +0/-3 (3 lines); hunks: -27,7 +27,6 @@ def setUpClass(cls):; -38,7 +37,6 @@ def setUpClass(cls):; symbols: setUpClass, touching `setUpClass`; `test/srt/nightly/test_deepseek_v31_perf.py` modified +0/-2 (2 lines); hunks: -27,7 +27,6 @@ def setUpClass(cls):; -38,7 +37,6 @@ def setUpClass(cls):; symbols: setUpClass, touching `setUpClass`.
- Code diff details:
  - `test/srt/nightly/test_deepseek_v32_perf.py` modified +0/-3 (3 lines); hunks: -27,7 +27,6 @@ def setUpClass(cls):; -38,7 +37,6 @@ def setUpClass(cls):; symbols: setUpClass
  - `test/srt/nightly/test_deepseek_v31_perf.py` modified +0/-2 (2 lines); hunks: -27,7 +27,6 @@ def setUpClass(cls):; -38,7 +37,6 @@ def setUpClass(cls):; symbols: setUpClass
- Key code excerpts:

```diff
diff -- test/srt/nightly/test_deepseek_v32_perf.py
@@ -27,7 +27,6 @@ def setUpClass(cls):
-                    "--enable-dp-attention",
@@ -38,7 +37,6 @@ def setUpClass(cls):
-                    "--enable-dp-attention",
@@ -59,7 +57,6 @@ def setUpClass(cls):
-                    "--enable-dp-attention",
diff -- test/srt/nightly/test_deepseek_v31_perf.py
@@ -27,7 +27,6 @@ def setUpClass(cls):
-                    "--enable-dp-attention",
@@ -38,7 +37,6 @@ def setUpClass(cls):
-                    "--enable-dp-attention",
```

- Reviewed files:
  - tests: `test/srt/nightly/test_deepseek_v32_perf.py` modified +0/-3; `test/srt/nightly/test_deepseek_v31_perf.py` modified +0/-2
- Risk and verification: The diff ships test coverage in `test/srt/nightly/test_deepseek_v31_perf.py`, `test/srt/nightly/test_deepseek_v32_perf.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #11589 - [Tool Call] Steamline function arguments when tool_choice="auto" for deepseekv31_detector

- Link: https://github.com/sgl-project/sglang/pull/11589
- Status/date: merged / 2025-11-14
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/function_call/deepseekv31_detector.py`; associated commits `fc5da1e80b78`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +4/-9, 34 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Tool Call] Steamline function arguments when tool_choice="auto" for deepseekv31_detector"; model line: DeepSeek V3.1; category: model implementation change; main diff: `python/sglang/srt/function_call/deepseekv31_detector.py`; technical summary: Covers "[Tool Call] Steamline function arguments when tool_choice="auto" for deepseekv31_detector"; the main implementation surface is `python/sglang/srt/function_call/deepseekv31_detector.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/function_call/deepseekv31_detector.py` modified +4/-9 (13 lines); hunks: -115,13 +115,14 @@ def parse_streaming_increment(; -180,15 +181,9 @@ def parse_streaming_increment(; symbols: parse_streaming_increment, touching `parse_streaming_increment`.
- Code diff details:
  - `python/sglang/srt/function_call/deepseekv31_detector.py` modified +4/-9 (13 lines); hunks: -115,13 +115,14 @@ def parse_streaming_increment(; -180,15 +181,9 @@ def parse_streaming_increment(; symbols: parse_streaming_increment
- Key code excerpts:

```diff
diff -- python/sglang/srt/function_call/deepseekv31_detector.py
@@ -115,13 +115,14 @@ def parse_streaming_increment(
-                pattern=r"<｜tool▁call▁begin｜>(.*)<｜tool▁sep｜>(.*)<｜tool▁call▁end｜>",
+                pattern=r"<｜tool▁call▁begin｜>(.*)<｜tool▁sep｜>(.*?)(<｜tool▁call▁end｜>|$)",
+                is_tool_end = partial_match.group(3)
@@ -180,15 +181,9 @@ def parse_streaming_increment(
-                        tool_call_end_pattern = (
-                            r"<｜tool▁call▁begin｜>.*?<｜tool▁call▁end｜>"
```

- Reviewed files:
  - runtime: `python/sglang/srt/function_call/deepseekv31_detector.py` modified +4/-9
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/function_call/deepseekv31_detector.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #13954 - Fix Deepseek v3.1 loading issue

- Link: https://github.com/sgl-project/sglang/pull/13954
- Status/date: merged / 2025-11-26
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `13e5beeab499`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix Deepseek v3.1 loading issue"; model line: DeepSeek V3.1; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v2.py`; technical summary: Covers "Fix Deepseek v3.1 loading issue"; the main implementation surface is `python/sglang/srt/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +1/-1 (2 lines); hunks: -3568,7 +3568,7 @@ def post_load_weights(self, is_nextn=False, weight_names=N...; symbols: post_load_weights, touching `post_load_weights`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +1/-1 (2 lines); hunks: -3568,7 +3568,7 @@ def post_load_weights(self, is_nextn=False, weight_names=N...; symbols: post_load_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -3568,7 +3568,7 @@ def post_load_weights(self, is_nextn=False, weight_names=None):
-                        and self_attn.kv_b_proj.executed_weight_requant_ue8m0
+                        and weight_scale.format_ue8m0
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #14837 - [Auto Sync] Update tool_chat_template_deepseekv31.jinja (20251210)

- Link: https://github.com/sgl-project/sglang/pull/14837
- Status/date: merged / 2025-12-10
- Trace source: `git log --name-only -- <model-files>` found it through `examples/chat_template/tool_chat_template_deepseekv31.jinja`; associated commits `ef1ab2302ab2`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-1, 13 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Auto Sync] Update tool_chat_template_deepseekv31.jinja (20251210)"; model line: DeepSeek V3.1; category: model implementation change; main diff: `examples/chat_template/tool_chat_template_deepseekv31.jinja`; technical summary: Covers "[Auto Sync] Update tool_chat_template_deepseekv31.jinja (20251210)"; the main implementation surface is `examples/chat_template/tool_chat_template_deepseekv31.jinja`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `examples/chat_template/tool_chat_template_deepseekv31.jinja` modified +5/-1 (6 lines); hunks: -19,7 +19,11.
- Code diff details:
  - `examples/chat_template/tool_chat_template_deepseekv31.jinja` modified +5/-1 (6 lines); hunks: -19,7 +19,11
- Key code excerpts:

```diff
diff -- examples/chat_template/tool_chat_template_deepseekv31.jinja
@@ -19,7 +19,11 @@
-    {% set tool_ns.text = tool_ns.text + '\n### ' + tool.function.name + '\nDescription: ' + tool.function.description + '\n\nParameters: ' + (tool.function.parameters | tojson) +
+    {% if tool.function.description is not none %}
+      {% set tool_ns.text = tool_ns.text + '\n### ' + tool.function.name + '\nDescription: ' + tool.function.description + '\n\nParameters: ' + (tool.function.parameters | tojson)
+    {% else %}
+      {% set tool_ns.text = tool_ns.text + '\n### ' + tool.function.name + '\n\nParameters: ' + (tool.function.parameters | tojson) + '\n' %}
+    {% endif %}
```

- Reviewed files:
  - docs: `examples/chat_template/tool_chat_template_deepseekv31.jinja` modified +5/-1
- Risk and verification: This is mostly docs/examples in `examples/chat_template/tool_chat_template_deepseekv31.jinja`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #13394 - Fix DeepSeekV31's structural tag trigger

- Link: https://github.com/sgl-project/sglang/pull/13394
- Status/date: merged / 2025-12-31
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/function_call/deepseekv31_detector.py`; associated commits `2667c857a78f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 7 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix DeepSeekV31's structural tag trigger"; model line: DeepSeek V3.1; category: bug fix; main diff: `python/sglang/srt/function_call/deepseekv31_detector.py`; technical summary: Covers "Fix DeepSeekV31's structural tag trigger"; the main implementation surface is `python/sglang/srt/function_call/deepseekv31_detector.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/function_call/deepseekv31_detector.py` modified +1/-1 (2 lines); hunks: -202,5 +202,5 @@ def structure_info(self) -> _GetInfoFunc:; symbols: structure_info, touching `structure_info`.
- Code diff details:
  - `python/sglang/srt/function_call/deepseekv31_detector.py` modified +1/-1 (2 lines); hunks: -202,5 +202,5 @@ def structure_info(self) -> _GetInfoFunc:; symbols: structure_info
- Key code excerpts:

```diff
diff -- python/sglang/srt/function_call/deepseekv31_detector.py
@@ -202,5 +202,5 @@ def structure_info(self) -> _GetInfoFunc:
-            trigger="<｜tool▁call▁begin｜>" + name + "<｜tool▁sep｜>",
+            trigger="<｜tool▁call▁begin｜>",
```

- Reviewed files:
  - runtime: `python/sglang/srt/function_call/deepseekv31_detector.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/function_call/deepseekv31_detector.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #16660 - [CI] Enable dpsk v31 test on nightly H200

- Link: https://github.com/sgl-project/sglang/pull/16660
- Status/date: merged / 2026-01-07
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-2, 17 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CI] Enable dpsk v31 test on nightly H200"; model line: DeepSeek V3.1; category: docs/tests/CI; main diff: `test/registered/8-gpu-models/test_deepseek_v31.py`; technical summary: Covers "[CI] Enable dpsk v31 test on nightly H200"; the main implementation surface is `test/registered/8-gpu-models/test_deepseek_v31.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/8-gpu-models/test_deepseek_v31.py` modified +1/-2 (3 lines); hunks: -4,15 +4,14; symbols: TestDeepseekV31Unified, for, touching `TestDeepseekV31Unified, for`.
- Code diff details:
  - `test/registered/8-gpu-models/test_deepseek_v31.py` modified +1/-2 (3 lines); hunks: -4,15 +4,14; symbols: TestDeepseekV31Unified, for
- Key code excerpts:

```diff
diff -- test/registered/8-gpu-models/test_deepseek_v31.py
@@ -4,15 +4,14 @@
-from sglang.test.test_utils import ModelLaunchSettings, is_blackwell_system
+from sglang.test.test_utils import ModelLaunchSettings
-@unittest.skipIf(not is_blackwell_system(), "Requires B200")
```

- Reviewed files:
  - tests: `test/registered/8-gpu-models/test_deepseek_v31.py` modified +1/-2
- Risk and verification: The diff ships test coverage in `test/registered/8-gpu-models/test_deepseek_v31.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17178 - Remove deepseek-r1 from THINKING_MODE_CHOICES in run_eval.py

- Link: https://github.com/sgl-project/sglang/pull/17178
- Status/date: merged / 2026-01-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-2, 26 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Remove deepseek-r1 from THINKING_MODE_CHOICES in run_eval.py"; model line: DeepSeek V3.1; category: docs/tests/CI; main diff: `python/sglang/test/run_eval.py`; technical summary: Covers "Remove deepseek-r1 from THINKING_MODE_CHOICES in run_eval.py"; the main implementation surface is `python/sglang/test/run_eval.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/test/run_eval.py` modified +3/-2 (5 lines); hunks: -22,6 +22,7 @@ def get_thinking_kwargs(args):; -203,7 +204,7 @@ def run_eval(args):; symbols: get_thinking_kwargs, run_eval, touching `get_thinking_kwargs, run_eval`.
- Code diff details:
  - `python/sglang/test/run_eval.py` modified +3/-2 (5 lines); hunks: -22,6 +22,7 @@ def get_thinking_kwargs(args):; -203,7 +204,7 @@ def run_eval(args):; symbols: get_thinking_kwargs, run_eval
- Key code excerpts:

```diff
diff -- python/sglang/test/run_eval.py
@@ -22,6 +22,7 @@ def get_thinking_kwargs(args):
+            # Qwen3
@@ -203,7 +204,7 @@ def run_eval(args):
-THINKING_MODE_CHOICES = ["deepseek-r1", "deepseek-v3", "qwen3"]
+THINKING_MODE_CHOICES = ["deepseek-v3", "qwen3"]
@@ -241,7 +242,7 @@ def run_eval(args):
-        help="Enable thinking mode in Deepseek R1, V3.1/3.2, or Qwen3",
```

- Reviewed files:
  - tests: `python/sglang/test/run_eval.py` modified +3/-2
- Risk and verification: The diff ships test coverage in `python/sglang/test/run_eval.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17133 - [DeepSeek V3.1/V3.2] Optimize fused moe configs for H20 & H20-3E based on swapab

- Link: https://github.com/sgl-project/sglang/pull/17133
- Status/date: merged / 2026-01-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +959/-217, 1311 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DeepSeek V3.1/V3.2] Optimize fused moe configs for H20 & H20-3E based on swapab"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json`; technical summary: Covers "[DeepSeek V3.1/V3.2] Optimize fused moe configs for H20 & H20-3E based on swapab"; the main implementation surface is `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128]_down.json` added +164/-0 (164 lines); hunks: -0,0 +1,164; `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128]_down.json` added +164/-0 (164 lines); hunks: -0,0 +1,164; `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: -0,0 +1,146; `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: -0,0 +1,146.
- Code diff details:
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128]_down.json` added +164/-0 (164 lines); hunks: -0,0 +1,164
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128]_down.json` added +164/-0 (164 lines); hunks: -0,0 +1,164
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: -0,0 +1,146
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: -0,0 +1,146
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py` modified +2/-2 (4 lines); hunks: -744,8 +744,8 @@ def invoke_fused_moe_kernel(; symbols: invoke_fused_moe_kernel
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128]_down.json
@@ -0,0 +1,164 @@
+{
+    "1": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 64,
+        "GROUP_SIZE_M": 64,
diff -- python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128]_down.json
@@ -0,0 +1,164 @@
+{
+    "1": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 64,
+        "GROUP_SIZE_M": 32,
diff -- python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json
@@ -0,0 +1,146 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128]_down.json` added +164/-0; `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128]_down.json` added +164/-0; `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0; `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0; `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py` modified +2/-2
  - other: `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py` modified +337/-215
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128].json`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17320 - fix: Fixed the issue where "finish_reason":"stop" appeared when calling the tool and the tool was in the content.

- Link: https://github.com/sgl-project/sglang/pull/17320
- Status/date: closed / 2026-01-19
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +14/-16, 56 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix: Fixed the issue where "finish_reason":"stop" appeared when calling the tool and the tool was in the content."; model line: DeepSeek V3.1; category: bug fix; main diff: `python/sglang/srt/function_call/deepseekv32_detector.py`, `examples/chat_template/tool_chat_template_deepseekv32.jinja`; technical summary: Covers "fix: Fixed the issue where "finish_reason":"stop" appeared when calling the tool and the tool was in the content."; the main implementation surface is `python/sglang/srt/function_call/deepseekv32_detector.py`, `examples/chat_template/tool_chat_template_deepseekv32.jinja`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/function_call/deepseekv32_detector.py` modified +8/-4 (12 lines); hunks: -195,12 +195,16 @@ def detect_and_parse(self, text: str, tools: list[Tool]) -...; symbols: detect_and_parse, touching `detect_and_parse`; `examples/chat_template/tool_chat_template_deepseekv32.jinja` modified +6/-12 (18 lines); hunks: -22,7 +22,7; -41,20 +41,14.
- Code diff details:
  - `python/sglang/srt/function_call/deepseekv32_detector.py` modified +8/-4 (12 lines); hunks: -195,12 +195,16 @@ def detect_and_parse(self, text: str, tools: list[Tool]) -...; symbols: detect_and_parse
  - `examples/chat_template/tool_chat_template_deepseekv32.jinja` modified +6/-12 (18 lines); hunks: -22,7 +22,7; -41,20 +41,14
- Key code excerpts:

```diff
diff -- python/sglang/srt/function_call/deepseekv32_detector.py
@@ -195,12 +195,16 @@ def detect_and_parse(self, text: str, tools: list[Tool]) -> StreamingParseResult
-            for func_name, invoke_content, _ in invoke_matches:
+            for i, (func_name, invoke_content, _) in enumerate(invoke_matches):
-                # construct match_result for parse_base_json
-                match_result = {"name": func_name, "parameters": func_args}
-                calls.extend(self.parse_base_json(match_result, tools))
+                calls.append(
diff -- examples/chat_template/tool_chat_template_deepseekv32.jinja
@@ -22,7 +22,7 @@
-  {% set tool_ns.text = tool_ns.text + "\nIMPORTANT: ALWAYS adhere to this exact format for tool use:\n<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>tool_call_name<｜tool▁sep｜>tool_call_a
+  {% set tool_ns.text = tool_ns.text + "\nIMPORTANT: ALWAYS adhere to this exact format for tool use:\n<｜DSML｜function_calls>\n<｜DSML｜invoke name=\"tool_call_name\">\n<｜DSML｜param
@@ -41,20 +41,14 @@
+    {{'<｜DSML｜function_calls>'}}
-      {%- if not ns.is_first %}
-        {%- if message['content'] is none %}
```

- Reviewed files:
  - runtime: `python/sglang/srt/function_call/deepseekv32_detector.py` modified +8/-4
  - docs: `examples/chat_template/tool_chat_template_deepseekv32.jinja` modified +6/-12
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/function_call/deepseekv32_detector.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17141 - fix: Fixed the issue where "finish_reason":"stop" appeared when calling the tool and the tool was in the content.

- Link: https://github.com/sgl-project/sglang/pull/17141
- Status/date: closed / 2026-01-22
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +14/-16, 56 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix: Fixed the issue where "finish_reason":"stop" appeared when calling the tool and the tool was in the content."; model line: DeepSeek V3.1; category: bug fix; main diff: `python/sglang/srt/function_call/deepseekv32_detector.py`, `examples/chat_template/tool_chat_template_deepseekv32.jinja`; technical summary: Covers "fix: Fixed the issue where "finish_reason":"stop" appeared when calling the tool and the tool was in the content."; the main implementation surface is `python/sglang/srt/function_call/deepseekv32_detector.py`, `examples/chat_template/tool_chat_template_deepseekv32.jinja`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/function_call/deepseekv32_detector.py` modified +8/-4 (12 lines); hunks: -195,12 +195,16 @@ def detect_and_parse(self, text: str, tools: list[Tool]) -...; symbols: detect_and_parse, touching `detect_and_parse`; `examples/chat_template/tool_chat_template_deepseekv32.jinja` modified +6/-12 (18 lines); hunks: -22,7 +22,7; -41,20 +41,14.
- Code diff details:
  - `python/sglang/srt/function_call/deepseekv32_detector.py` modified +8/-4 (12 lines); hunks: -195,12 +195,16 @@ def detect_and_parse(self, text: str, tools: list[Tool]) -...; symbols: detect_and_parse
  - `examples/chat_template/tool_chat_template_deepseekv32.jinja` modified +6/-12 (18 lines); hunks: -22,7 +22,7; -41,20 +41,14
- Key code excerpts:

```diff
diff -- python/sglang/srt/function_call/deepseekv32_detector.py
@@ -195,12 +195,16 @@ def detect_and_parse(self, text: str, tools: list[Tool]) -> StreamingParseResult
-            for func_name, invoke_content, _ in invoke_matches:
+            for i, (func_name, invoke_content, _) in enumerate(invoke_matches):
-                # construct match_result for parse_base_json
-                match_result = {"name": func_name, "parameters": func_args}
-                calls.extend(self.parse_base_json(match_result, tools))
+                calls.append(
diff -- examples/chat_template/tool_chat_template_deepseekv32.jinja
@@ -22,7 +22,7 @@
-  {% set tool_ns.text = tool_ns.text + "\nIMPORTANT: ALWAYS adhere to this exact format for tool use:\n<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>tool_call_name<｜tool▁sep｜>tool_call_a
+  {% set tool_ns.text = tool_ns.text + "\nIMPORTANT: ALWAYS adhere to this exact format for tool use:\n<｜DSML｜function_calls>\n<｜DSML｜invoke name=\"tool_call_name\">\n<｜DSML｜param
@@ -41,20 +41,14 @@
+    {{'<｜DSML｜function_calls>'}}
-      {%- if not ns.is_first %}
-        {%- if message['content'] is none %}
```

- Reviewed files:
  - runtime: `python/sglang/srt/function_call/deepseekv32_detector.py` modified +8/-4
  - docs: `examples/chat_template/tool_chat_template_deepseekv32.jinja` modified +6/-12
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/function_call/deepseekv32_detector.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17558 - fix: Fixed the issue where "finish_reason":"stop" appeared when calling the tool and the tool was in the content.

- Link: https://github.com/sgl-project/sglang/pull/17558
- Status/date: closed / 2026-01-24
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +14/-16, 56 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix: Fixed the issue where "finish_reason":"stop" appeared when calling the tool and the tool was in the content."; model line: DeepSeek V3.1; category: bug fix; main diff: `python/sglang/srt/function_call/deepseekv32_detector.py`, `examples/chat_template/tool_chat_template_deepseekv32.jinja`; technical summary: Covers "fix: Fixed the issue where "finish_reason":"stop" appeared when calling the tool and the tool was in the content."; the main implementation surface is `python/sglang/srt/function_call/deepseekv32_detector.py`, `examples/chat_template/tool_chat_template_deepseekv32.jinja`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/function_call/deepseekv32_detector.py` modified +8/-4 (12 lines); hunks: -195,12 +195,16 @@ def detect_and_parse(self, text: str, tools: list[Tool]) -...; symbols: detect_and_parse, touching `detect_and_parse`; `examples/chat_template/tool_chat_template_deepseekv32.jinja` modified +6/-12 (18 lines); hunks: -22,7 +22,7; -41,20 +41,14.
- Code diff details:
  - `python/sglang/srt/function_call/deepseekv32_detector.py` modified +8/-4 (12 lines); hunks: -195,12 +195,16 @@ def detect_and_parse(self, text: str, tools: list[Tool]) -...; symbols: detect_and_parse
  - `examples/chat_template/tool_chat_template_deepseekv32.jinja` modified +6/-12 (18 lines); hunks: -22,7 +22,7; -41,20 +41,14
- Key code excerpts:

```diff
diff -- python/sglang/srt/function_call/deepseekv32_detector.py
@@ -195,12 +195,16 @@ def detect_and_parse(self, text: str, tools: list[Tool]) -> StreamingParseResult
-            for func_name, invoke_content, _ in invoke_matches:
+            for i, (func_name, invoke_content, _) in enumerate(invoke_matches):
-                # construct match_result for parse_base_json
-                match_result = {"name": func_name, "parameters": func_args}
-                calls.extend(self.parse_base_json(match_result, tools))
+                calls.append(
diff -- examples/chat_template/tool_chat_template_deepseekv32.jinja
@@ -22,7 +22,7 @@
-  {% set tool_ns.text = tool_ns.text + "\nIMPORTANT: ALWAYS adhere to this exact format for tool use:\n<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>tool_call_name<｜tool▁sep｜>tool_call_a
+  {% set tool_ns.text = tool_ns.text + "\nIMPORTANT: ALWAYS adhere to this exact format for tool use:\n<｜DSML｜function_calls>\n<｜DSML｜invoke name=\"tool_call_name\">\n<｜DSML｜param
@@ -41,20 +41,14 @@
+    {{'<｜DSML｜function_calls>'}}
-      {%- if not ns.is_first %}
-        {%- if message['content'] is none %}
```

- Reviewed files:
  - runtime: `python/sglang/srt/function_call/deepseekv32_detector.py` modified +8/-4
  - docs: `examples/chat_template/tool_chat_template_deepseekv32.jinja` modified +6/-12
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/function_call/deepseekv32_detector.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17761 - fix: missing Assistant token after tool output in DeepSeek v3.1/v3.2 chat templates

- Link: https://github.com/sgl-project/sglang/pull/17761
- Status/date: open / 2026-01-26
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +79/-2, 102 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix: missing Assistant token after tool output in DeepSeek v3.1/v3.2 chat templates"; model line: DeepSeek V3.1; category: bug fix; main diff: `test/manual/test_deepseek_chat_templates.py`, `examples/chat_template/tool_chat_template_deepseekv31.jinja`, `examples/chat_template/tool_chat_template_deepseekv32.jinja`; technical summary: Covers "fix: missing Assistant token after tool output in DeepSeek v3.1/v3.2 chat templates"; the main implementation surface is `test/manual/test_deepseek_chat_templates.py`, `examples/chat_template/tool_chat_template_deepseekv31.jinja`, `examples/chat_template/tool_chat_template_deepseekv32.jinja`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/manual/test_deepseek_chat_templates.py` modified +77/-0 (77 lines); hunks: -313,6 +313,83 @@ def test_tool_call_with_content(self):; symbols: test_tool_call_with_content, test_assistant_marker_after_tool_output, touching `test_tool_call_with_content, test_assistant_marker_after_tool_output`; `examples/chat_template/tool_chat_template_deepseekv31.jinja` modified +1/-1 (2 lines); hunks: -60,7 +60,7; `examples/chat_template/tool_chat_template_deepseekv32.jinja` modified +1/-1 (2 lines); hunks: -57,7 +57,7.
- Code diff details:
  - `test/manual/test_deepseek_chat_templates.py` modified +77/-0 (77 lines); hunks: -313,6 +313,83 @@ def test_tool_call_with_content(self):; symbols: test_tool_call_with_content, test_assistant_marker_after_tool_output
  - `examples/chat_template/tool_chat_template_deepseekv31.jinja` modified +1/-1 (2 lines); hunks: -60,7 +60,7
  - `examples/chat_template/tool_chat_template_deepseekv32.jinja` modified +1/-1 (2 lines); hunks: -57,7 +57,7
- Key code excerpts:

```diff
diff -- test/manual/test_deepseek_chat_templates.py
@@ -313,6 +313,83 @@ def test_tool_call_with_content(self):
+    def test_assistant_marker_after_tool_output(self):
+        """Test that Assistant marker is present after tool output in multi-turn conversation."""
+        # This tests that when an assistant responds after receiving tool output,
+        # the <｜Assistant｜> marker is correctly added
+        for version in ["v3.1", "v3.2"]:
+            with self.subTest(version=version):
diff -- examples/chat_template/tool_chat_template_deepseekv31.jinja
@@ -60,7 +60,7 @@
-    {%- if ns.is_last_user %}
+    {%- if ns.is_last_user or ns.is_tool %}
diff -- examples/chat_template/tool_chat_template_deepseekv32.jinja
@@ -57,7 +57,7 @@
-    {%- if ns.is_last_user %}
+    {%- if ns.is_last_user or ns.is_tool %}
```

- Reviewed files:
  - tests: `test/manual/test_deepseek_chat_templates.py` modified +77/-0
  - docs: `examples/chat_template/tool_chat_template_deepseekv31.jinja` modified +1/-1; `examples/chat_template/tool_chat_template_deepseekv32.jinja` modified +1/-1
- Risk and verification: The diff ships test coverage in `test/manual/test_deepseek_chat_templates.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18236 - Fix function call arguments missing in streaming mode for DeepSeek V3.1

- Link: https://github.com/sgl-project/sglang/pull/18236
- Status/date: open / 2026-02-04
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +21/-3, 57 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix function call arguments missing in streaming mode for DeepSeek V3.1"; model line: DeepSeek V3.1; category: bug fix; main diff: `python/sglang/srt/function_call/deepseekv31_detector.py`; technical summary: Covers "Fix function call arguments missing in streaming mode for DeepSeek V3.1"; the main implementation surface is `python/sglang/srt/function_call/deepseekv31_detector.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/function_call/deepseekv31_detector.py` modified +21/-3 (24 lines); hunks: -52,6 +52,7 @@ def __init__(self):; -111,6 +112,18 @@ def parse_streaming_increment(; symbols: __init__, has_tool_call, parse_streaming_increment, touching `__init__, has_tool_call, parse_streaming_increment`.
- Code diff details:
  - `python/sglang/srt/function_call/deepseekv31_detector.py` modified +21/-3 (24 lines); hunks: -52,6 +52,7 @@ def __init__(self):; -111,6 +112,18 @@ def parse_streaming_increment(; symbols: __init__, has_tool_call, parse_streaming_increment
- Key code excerpts:

```diff
diff -- python/sglang/srt/function_call/deepseekv31_detector.py
@@ -52,6 +52,7 @@ def __init__(self):
+        self._normal_text_sent = False
@@ -111,6 +112,18 @@ def parse_streaming_increment(
+        # Extract normal text before tool call on first detection
+        normal_text_to_return = ""
+        if not self._normal_text_sent:
+            # Find the first tool call marker
```

- Reviewed files:
  - runtime: `python/sglang/srt/function_call/deepseekv31_detector.py` modified +21/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/function_call/deepseekv31_detector.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21739 - [NPU] Update DeepSeek-V3.1 and DeepSeek-V3.2 model deployment instructions in documentation

- Link: https://github.com/sgl-project/sglang/pull/21739
- Status/date: open / 2026-03-31
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +163/-19, 270 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NPU] Update DeepSeek-V3.1 and DeepSeek-V3.2 model deployment instructions in documentation"; model line: DeepSeek V3.1; category: docs/tests/CI; main diff: `docs/platforms/ascend/ascend_npu_best_practice.md`; technical summary: Covers "[NPU] Update DeepSeek-V3.1 and DeepSeek-V3.2 model deployment instructions in documentation"; the main implementation surface is `docs/platforms/ascend/ascend_npu_best_practice.md`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs/platforms/ascend/ascend_npu_best_practice.md` modified +163/-19 (182 lines); hunks: -20,6 +20,7 @@ you encounter issues or have any questions, please [open an is...; -177,7 +178,148 @@ We tested it based on the `RANDOM` dataset..
- Code diff details:
  - `docs/platforms/ascend/ascend_npu_best_practice.md` modified +163/-19 (182 lines); hunks: -20,6 +20,7 @@ you encounter issues or have any questions, please [open an is...; -177,7 +178,148 @@ We tested it based on the `RANDOM` dataset.
- Key code excerpts:

```diff
diff -- docs/platforms/ascend/ascend_npu_best_practice.md
@@ -20,6 +20,7 @@ you encounter issues or have any questions, please [open an issue](https://githu
+| Deepseek-R1 | Atlas 800I A3 | 24    | PD Separation | 2K+2K     | 50ms | W8A8 INT8    | [Optimal Configuration](#deepseek-r1-2k-2k-50ms-on-a3-24-cards-separation-mode) |
@@ -177,7 +178,148 @@ We tested it based on the `RANDOM` dataset.
+### DeepSeek-R1 2K-2K 50ms on A3 24 Cards Separation Mode
+Model: Deepseek R1
+Hardware: Atlas 800I A3 24Card
+DeployMode: PD Separation
```

- Reviewed files:
  - docs: `docs/platforms/ascend/ascend_npu_best_practice.md` modified +163/-19
- Risk and verification: This is mostly docs/examples in `docs/platforms/ascend/ascend_npu_best_practice.md`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #22433 - [Test] Add unit tests for DeepSeekV31Detector

- Link: https://github.com/sgl-project/sglang/pull/22433
- Status/date: open / 2026-04-09
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +314/-0, 315 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Test] Add unit tests for DeepSeekV31Detector"; model line: DeepSeek V3.1; category: docs/tests/CI; main diff: `test/registered/unit/function_call/test_deepseekv31_detector.py`; technical summary: Covers "[Test] Add unit tests for DeepSeekV31Detector"; the main implementation surface is `test/registered/unit/function_call/test_deepseekv31_detector.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/unit/function_call/test_deepseekv31_detector.py` added +314/-0 (314 lines); hunks: -0,0 +1,314; symbols: _wrap_single, _make_tools, TestDeepSeekV31DetectorHasToolCall, setUp, touching `_wrap_single, _make_tools, TestDeepSeekV31DetectorHasToolCall`.
- Code diff details:
  - `test/registered/unit/function_call/test_deepseekv31_detector.py` added +314/-0 (314 lines); hunks: -0,0 +1,314; symbols: _wrap_single, _make_tools, TestDeepSeekV31DetectorHasToolCall, setUp
- Key code excerpts:

```diff
diff -- test/registered/unit/function_call/test_deepseekv31_detector.py
@@ -0,0 +1,314 @@
+"""Unit tests for DeepSeekV31Detector — no server, no model loading.
+Covers the DeepSeek V3.1 function-call format:
+    <｜tool▁calls▁begin｜>
+      <｜tool▁call▁begin｜>{name}<｜tool▁sep｜>{json_args}<｜tool▁call▁end｜>
+      ...
+    <｜tool▁calls▁end｜>
```

- Reviewed files:
  - tests: `test/registered/unit/function_call/test_deepseekv31_detector.py` added +314/-0
- Risk and verification: The diff ships test coverage in `test/registered/unit/function_call/test_deepseekv31_detector.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #21593 - Fix tool call constrained decoding and parsing for models with native formats

- Link: https://github.com/sgl-project/sglang/pull/21593
- Status/date: merged / 2026-04-11
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +306/-61, 516 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix tool call constrained decoding and parsing for models with native formats"; model line: DeepSeek V3.1; category: bug fix; main diff: `test/registered/unit/function_call/test_function_call_parser.py`, `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/function_call/function_call_parser.py`; technical summary: Covers "Fix tool call constrained decoding and parsing for models with native formats"; the main implementation surface is `test/registered/unit/function_call/test_function_call_parser.py`, `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/function_call/function_call_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/unit/function_call/test_function_call_parser.py` modified +113/-0 (113 lines); hunks: -3859,6 +3859,119 @@ def test_streaming_function_call_marker_json_split_at_qu...; symbols: test_streaming_function_call_marker_json_split_at_quotes, TestGetStructureConstraint, _make_tools, _make_parser, touching `test_streaming_function_call_marker_json_split_at_quotes, TestGetStructureConstraint, _make_tools`; `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +67/-43 (110 lines); hunks: -361,9 +361,11 @@ def _process_messages(; -1136,22 +1138,56 @@ def _process_tool_calls(; symbols: _process_messages, _process_tool_calls, _process_tool_call_stream, touching `_process_messages, _process_tool_calls, _process_tool_call_stream`; `python/sglang/srt/function_call/function_call_parser.py` modified +35/-11 (46 lines); hunks: -3,6 +3,7; -32,7 +33,10; symbols: parse_stream_chunk, get_structure_tag, get_structure_constraint, touching `parse_stream_chunk, get_structure_tag, get_structure_constraint`; `test/registered/openai_server/function_call/test_tool_choice.py` modified +8/-2 (10 lines); hunks: -348,8 +348,12 @@ def test_tool_choice_specific_function_streaming(self):; -406,13 +410,15 @@ def test_required_streaming_arguments_chunks_json(self):; symbols: test_tool_choice_specific_function_streaming, test_required_streaming_arguments_chunks_json, test_complex_parameters_required_non_streaming, touching `test_tool_choice_specific_function_streaming, test_required_streaming_arguments_chunks_json, test_complex_parameters_required_non_streaming`.
- Code diff details:
  - `test/registered/unit/function_call/test_function_call_parser.py` modified +113/-0 (113 lines); hunks: -3859,6 +3859,119 @@ def test_streaming_function_call_marker_json_split_at_qu...; symbols: test_streaming_function_call_marker_json_split_at_quotes, TestGetStructureConstraint, _make_tools, _make_parser
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +67/-43 (110 lines); hunks: -361,9 +361,11 @@ def _process_messages(; -1136,22 +1138,56 @@ def _process_tool_calls(; symbols: _process_messages, _process_tool_calls, _process_tool_call_stream
  - `python/sglang/srt/function_call/function_call_parser.py` modified +35/-11 (46 lines); hunks: -3,6 +3,7; -32,7 +33,10; symbols: parse_stream_chunk, get_structure_tag, get_structure_constraint
  - `test/registered/openai_server/function_call/test_tool_choice.py` modified +8/-2 (10 lines); hunks: -348,8 +348,12 @@ def test_tool_choice_specific_function_streaming(self):; -406,13 +410,15 @@ def test_required_streaming_arguments_chunks_json(self):; symbols: test_tool_choice_specific_function_streaming, test_required_streaming_arguments_chunks_json, test_complex_parameters_required_non_streaming
  - `python/sglang/srt/function_call/deepseekv3_detector.py` modified +5/-3 (8 lines); hunks: -203,7 +203,9 @@ def parse_streaming_increment(; symbols: parse_streaming_increment, structure_info
- Key code excerpts:

```diff
diff -- test/registered/unit/function_call/test_function_call_parser.py
@@ -3859,6 +3859,119 @@ def test_streaming_function_call_marker_json_split_at_quotes(self):
+class TestGetStructureConstraint(unittest.TestCase):
+    """Tests for FunctionCallParser.get_structure_constraint() logic.
+    Verifies that detectors supporting structural_tag use it for required/named
+    tool_choice, and that the generic json_schema fallback is used otherwise.
+    """
+    def _make_tools(self, strict=False):
diff -- python/sglang/srt/entrypoints/openai/serving_chat.py
@@ -361,9 +361,11 @@ def _process_messages(
-            # Handle JSON schema constraint directly for required or named tool choice
-            if request.tool_choice == "required" or isinstance(
-                request.tool_choice, ToolChoice
+            # Fallback: use generic JSON schema for required/named tool choice
+            # only when no parser-specific constraint was set
+            if tool_call_constraint is None and (
diff -- python/sglang/srt/function_call/function_call_parser.py
@@ -3,6 +3,7 @@
```

- Reviewed files:
  - tests: `test/registered/unit/function_call/test_function_call_parser.py` modified +113/-0; `test/registered/openai_server/function_call/test_tool_choice.py` modified +8/-2; `test/registered/openai_server/basic/test_serving_chat.py` modified +72/-0
  - runtime: `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +67/-43; `python/sglang/srt/function_call/function_call_parser.py` modified +35/-11; `python/sglang/srt/function_call/deepseekv3_detector.py` modified +5/-3; `python/sglang/srt/function_call/base_format_detector.py` modified +1/-1; `python/sglang/srt/entrypoints/openai/protocol.py` modified +1/-0
- Risk and verification: The diff ships test coverage in `test/registered/openai_server/basic/test_serving_chat.py`, `test/registered/openai_server/function_call/test_tool_choice.py`, `test/registered/unit/function_call/test_function_call_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22981 - [Test] Add unit tests for 7 missing function call detectors

- Link: https://github.com/sgl-project/sglang/pull/22981
- Status/date: open / 2026-04-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +1017/-1, 1063 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Test] Add unit tests for 7 missing function call detectors"; model line: DeepSeek V3.1; category: docs/tests/CI; main diff: `test/registered/unit/function_call/test_function_call_parser.py`, `test/registered/openai_server/function_call/test_tool_choice.py`, `test/registered/unit/function_call/test_kimik2_detector.py`; technical summary: Covers "[Test] Add unit tests for 7 missing function call detectors"; the main implementation surface is `test/registered/unit/function_call/test_function_call_parser.py`, `test/registered/openai_server/function_call/test_tool_choice.py`, `test/registered/unit/function_call/test_kimik2_detector.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/unit/function_call/test_function_call_parser.py` modified +960/-1 (961 lines); hunks: -2,9 +2,11; -15,16 +17,22; symbols: TestPythonicDetector, setUp, test_has_tool_call, test_detect_and_parse_single, touching `TestPythonicDetector, setUp, test_has_tool_call`; `test/registered/openai_server/function_call/test_tool_choice.py` modified +57/-0 (57 lines); hunks: -894,5 +894,62 @@ def setUpClass(cls):; symbols: setUpClass, TestToolChoiceWithConstrainedDecoding, test_tool_choice_required_strict_finish_reason, touching `setUpClass, TestToolChoiceWithConstrainedDecoding, test_tool_choice_required_strict_finish_reason`; `test/registered/unit/function_call/test_kimik2_detector.py` renamed +0/-0 (0 lines).
- Code diff details:
  - `test/registered/unit/function_call/test_function_call_parser.py` modified +960/-1 (961 lines); hunks: -2,9 +2,11; -15,16 +17,22; symbols: TestPythonicDetector, setUp, test_has_tool_call, test_detect_and_parse_single
  - `test/registered/openai_server/function_call/test_tool_choice.py` modified +57/-0 (57 lines); hunks: -894,5 +894,62 @@ def setUpClass(cls):; symbols: setUpClass, TestToolChoiceWithConstrainedDecoding, test_tool_choice_required_strict_finish_reason
  - `test/registered/unit/function_call/test_kimik2_detector.py` renamed +0/-0 (0 lines)
- Key code excerpts:

```diff
diff -- test/registered/unit/function_call/test_function_call_parser.py
@@ -2,9 +2,11 @@
+from sglang.srt.environ import envs
+from sglang.srt.function_call.deepseekv31_detector import DeepSeekV31Detector
@@ -15,16 +17,22 @@
+from sglang.srt.function_call.gpt_oss_detector import GptOssDetector
+from sglang.srt.function_call.internlm_detector import InternlmDetector
+from sglang.srt.function_call.mimo_detector import MiMoDetector
diff -- test/registered/openai_server/function_call/test_tool_choice.py
@@ -894,5 +894,62 @@ def setUpClass(cls):
+class TestToolChoiceWithConstrainedDecoding(TestToolChoiceLlama32):
+    """Test tool_choice with grammar backend (structural_tag + constrained decoding).
+    Verifies that tool_choice="required" with strict=True produces valid
+    tool calls when the grammar backend is enabled.
+    """
+    @classmethod
```

- Reviewed files:
  - tests: `test/registered/unit/function_call/test_function_call_parser.py` modified +960/-1; `test/registered/openai_server/function_call/test_tool_choice.py` modified +57/-0; `test/registered/unit/function_call/test_kimik2_detector.py` renamed +0/-0
- Risk and verification: The diff ships test coverage in `test/registered/openai_server/function_call/test_tool_choice.py`, `test/registered/unit/function_call/test_function_call_parser.py`, `test/registered/unit/function_call/test_kimik2_detector.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22128 - Allow piecewise CUDA graph with speculative decoding

- Link: https://github.com/sgl-project/sglang/pull/22128
- Status/date: merged / 2026-04-17
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +272/-18, 344 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Allow piecewise CUDA graph with speculative decoding"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`, `python/sglang/srt/model_executor/model_runner.py`, `test/registered/piecewise_cuda_graph/test_pcg_with_speculative_decoding.py`; technical summary: Covers "Allow piecewise CUDA graph with speculative decoding"; the main implementation surface is `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`, `python/sglang/srt/model_executor/model_runner.py`, `test/registered/piecewise_cuda_graph/test_pcg_with_speculative_decoding.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` modified +10/-0 (10 lines); hunks: -417,6 +417,16 @@ def can_run(self, forward_batch: ForwardBatch):; symbols: can_run, touching `can_run`; `python/sglang/srt/model_executor/model_runner.py` modified +4/-0 (4 lines); hunks: -2554,6 +2554,10 @@ def init_piecewise_cuda_graphs(self):; symbols: init_piecewise_cuda_graphs, touching `init_piecewise_cuda_graphs`; `test/registered/piecewise_cuda_graph/test_pcg_with_speculative_decoding.py` added +243/-0 (243 lines); hunks: -0,0 +1,243; symbols: TestPCGWithMTP, setUpClass, tearDownClass, test_gsm8k, touching `TestPCGWithMTP, setUpClass, tearDownClass`; `python/sglang/srt/server_args.py` modified +15/-18 (33 lines); hunks: -1113,56 +1113,53 @@ def _handle_piecewise_cuda_graph(self):; symbols: _handle_piecewise_cuda_graph, touching `_handle_piecewise_cuda_graph`.
- Code diff details:
  - `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` modified +10/-0 (10 lines); hunks: -417,6 +417,16 @@ def can_run(self, forward_batch: ForwardBatch):; symbols: can_run
  - `python/sglang/srt/model_executor/model_runner.py` modified +4/-0 (4 lines); hunks: -2554,6 +2554,10 @@ def init_piecewise_cuda_graphs(self):; symbols: init_piecewise_cuda_graphs
  - `test/registered/piecewise_cuda_graph/test_pcg_with_speculative_decoding.py` added +243/-0 (243 lines); hunks: -0,0 +1,243; symbols: TestPCGWithMTP, setUpClass, tearDownClass, test_gsm8k
  - `python/sglang/srt/server_args.py` modified +15/-18 (33 lines); hunks: -1113,56 +1113,53 @@ def _handle_piecewise_cuda_graph(self):; symbols: _handle_piecewise_cuda_graph
- Key code excerpts:

```diff
diff -- python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py
@@ -417,6 +417,16 @@ def can_run(self, forward_batch: ForwardBatch):
+        # PCG graphs are captured with ForwardMode.EXTEND and spec_info=None.
+        # TARGET_VERIFY has different spec_info and capture_hidden_mode,
+        # so it must not use PCG-captured graphs.
+        if forward_batch.forward_mode.is_target_verify():
+            return False
+        # PCG graphs are captured with the runner's capture_hidden_mode.
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -2554,6 +2554,10 @@ def init_piecewise_cuda_graphs(self):
+        # Draft models use decode CUDA graphs, not PCG
+        if self.is_draft_worker:
+            return
diff -- test/registered/piecewise_cuda_graph/test_pcg_with_speculative_decoding.py
@@ -0,0 +1,243 @@
+"""Test piecewise CUDA graph coexisting with speculative decoding.
+PCG handles prefill/extend path while speculative decoding (MTP/EAGLE3/STANDALONE/NGRAM)
+uses decode CUDA graphs. This test verifies they don't interfere with each other.
```

- Reviewed files:
  - runtime: `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` modified +10/-0; `python/sglang/srt/model_executor/model_runner.py` modified +4/-0; `python/sglang/srt/server_args.py` modified +15/-18
  - tests: `test/registered/piecewise_cuda_graph/test_pcg_with_speculative_decoding.py` added +243/-0
- Risk and verification: The diff ships test coverage in `test/registered/piecewise_cuda_graph/test_pcg_with_speculative_decoding.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #21599 - [SPEC][1/N] feat: add adaptive speculative_num_steps for EAGLE topk=1

- Link: https://github.com/sgl-project/sglang/pull/21599
- Status/date: merged / 2026-04-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 13 files, +1296/-33, 1579 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[SPEC][1/N] feat: add adaptive speculative_num_steps for EAGLE topk=1"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `python/sglang/srt/model_executor/cuda_graph_runner.py`, `benchmark/bench_adaptive_speculative.py`, `test/registered/unit/spec/test_adaptive_spec_params.py`; technical summary: Covers "[SPEC][1/N] feat: add adaptive speculative_num_steps for EAGLE topk=1"; the main implementation surface is `python/sglang/srt/model_executor/cuda_graph_runner.py`, `benchmark/bench_adaptive_speculative.py`, `test/registered/unit/spec/test_adaptive_spec_params.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +26/-12 (38 lines); hunks: -512,7 +512,14 @@ def set_global_graph_memory_pool(val):; -551,6 +558,17 @@ def __init__(self, model_runner: ModelRunner):; symbols: set_global_graph_memory_pool, CudaGraphRunner, __init__, touching `set_global_graph_memory_pool, CudaGraphRunner, __init__`; `benchmark/bench_adaptive_speculative.py` added +263/-0 (263 lines); hunks: -0,0 +1,263; symbols: build_phase_plan, send_request, run_phase, summarize_phases, touching `build_phase_plan, send_request, run_phase`; `test/registered/unit/spec/test_adaptive_spec_params.py` added +195/-0 (195 lines); hunks: -0,0 +1,195; symbols: TestAdaptiveSpeculativeParams, test_initial_steps_snap_to_nearest_candidate_preferring_larger_step, test_update_respects_warmup_and_interval, test_empty_batches_do_not_consume_warmup_or_shift_steps, touching `TestAdaptiveSpeculativeParams, test_initial_steps_snap_to_nearest_candidate_preferring_larger_step, test_update_respects_warmup_and_interval`; `test/registered/spec/eagle/test_adaptive_speculative.py` added +170/-0 (170 lines); hunks: -0,0 +1,170; symbols: TestAdaptiveSpeculativeServer, setUpClass, tearDownClass, _get_internal_state, touching `TestAdaptiveSpeculativeServer, setUpClass, tearDownClass`.
- Code diff details:
  - `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +26/-12 (38 lines); hunks: -512,7 +512,14 @@ def set_global_graph_memory_pool(val):; -551,6 +558,17 @@ def __init__(self, model_runner: ModelRunner):; symbols: set_global_graph_memory_pool, CudaGraphRunner, __init__
  - `benchmark/bench_adaptive_speculative.py` added +263/-0 (263 lines); hunks: -0,0 +1,263; symbols: build_phase_plan, send_request, run_phase, summarize_phases
  - `test/registered/unit/spec/test_adaptive_spec_params.py` added +195/-0 (195 lines); hunks: -0,0 +1,195; symbols: TestAdaptiveSpeculativeParams, test_initial_steps_snap_to_nearest_candidate_preferring_larger_step, test_update_respects_warmup_and_interval, test_empty_batches_do_not_consume_warmup_or_shift_steps
  - `test/registered/spec/eagle/test_adaptive_speculative.py` added +170/-0 (170 lines); hunks: -0,0 +1,170; symbols: TestAdaptiveSpeculativeServer, setUpClass, tearDownClass, _get_internal_state
  - `python/sglang/srt/speculative/eagle_worker.py` modified +162/-4 (166 lines); hunks: -1,5 +1,6; -24,6 +25,7; symbols: __init__, init_cuda_graphs, apply_runtime_state, build_adaptive_runtime_state
- Key code excerpts:

```diff
diff -- python/sglang/srt/model_executor/cuda_graph_runner.py
@@ -512,7 +512,14 @@ def set_global_graph_memory_pool(val):
-    def __init__(self, model_runner: ModelRunner):
+    def __init__(
+        self,
+        model_runner: ModelRunner,
+        *,
+        attn_backend=None,
diff -- benchmark/bench_adaptive_speculative.py
@@ -0,0 +1,263 @@
+"""Benchmark adaptive speculative decoding against static baselines.
+Run the same workload against one adaptive server and one or more static
+servers, then compare throughput, latency, and acceptance length.
+Workloads:
+- low: steady-state low-acceptance generation
+- high: steady-state high-acceptance generation
diff -- test/registered/unit/spec/test_adaptive_spec_params.py
@@ -0,0 +1,195 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +26/-12; `python/sglang/srt/speculative/eagle_worker.py` modified +162/-4; `python/sglang/srt/speculative/adaptive_spec_params.py` added +133/-0; `python/sglang/srt/speculative/adaptive_runtime_state.py` added +121/-0
  - other: `benchmark/bench_adaptive_speculative.py` added +263/-0
  - tests: `test/registered/unit/spec/test_adaptive_spec_params.py` added +195/-0; `test/registered/spec/eagle/test_adaptive_speculative.py` added +170/-0
  - docs: `docs/advanced_features/adaptive_speculative_decoding.md` added +156/-0
- Risk and verification: The diff ships test coverage in `test/registered/spec/eagle/test_adaptive_speculative.py`, `test/registered/unit/spec/test_adaptive_spec_params.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #23315 - Opt-in strip of thinking tokens from radix cache

- Link: https://github.com/sgl-project/sglang/pull/23315
- Status/date: merged / 2026-04-21
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +72/-4, 131 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Opt-in strip of thinking tokens from radix cache"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py`, `python/sglang/srt/managers/schedule_batch.py`, `python/sglang/srt/server_args.py`; technical summary: Covers "Opt-in strip of thinking tokens from radix cache"; the main implementation surface is `test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py`, `python/sglang/srt/managers/schedule_batch.py`, `python/sglang/srt/server_args.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py` modified +52/-1 (53 lines); hunks: -30,7 +30,11; -485,6 +489,53 @@ def test_cache_finished_req_insert(self):; symbols: test_cache_finished_req_insert, test_cache_finished_req_strips_thinking, test_cache_finished_req_no_insert, touching `test_cache_finished_req_insert, test_cache_finished_req_strips_thinking, test_cache_finished_req_no_insert`; `python/sglang/srt/managers/schedule_batch.py` modified +9/-2 (11 lines); hunks: -903,13 +903,20 @@ def output_ids_through_stop(self) -> List[int]:; -921,7 +928,7 @@ def pop_overallocated_kv_cache(self) -> Tuple[int, int]:; symbols: output_ids_through_stop, _cache_commit_len, pop_committed_kv_cache, pop_overallocated_kv_cache, touching `output_ids_through_stop, _cache_commit_len, pop_committed_kv_cache`; `python/sglang/srt/server_args.py` modified +8/-0 (8 lines); hunks: -436,6 +436,7 @@ class ServerArgs:; -4879,6 +4880,13 @@ def add_cli_args(parser: argparse.ArgumentParser):; symbols: ServerArgs, add_cli_args, touching `ServerArgs, add_cli_args`; `python/sglang/srt/mem_cache/common.py` modified +3/-1 (4 lines); hunks: -489,7 +489,9 @@ def release_kv_cache(req: Req, tree_cache: BasePrefixCache,...; symbols: release_kv_cache, touching `release_kv_cache`.
- Code diff details:
  - `test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py` modified +52/-1 (53 lines); hunks: -30,7 +30,11; -485,6 +489,53 @@ def test_cache_finished_req_insert(self):; symbols: test_cache_finished_req_insert, test_cache_finished_req_strips_thinking, test_cache_finished_req_no_insert
  - `python/sglang/srt/managers/schedule_batch.py` modified +9/-2 (11 lines); hunks: -903,13 +903,20 @@ def output_ids_through_stop(self) -> List[int]:; -921,7 +928,7 @@ def pop_overallocated_kv_cache(self) -> Tuple[int, int]:; symbols: output_ids_through_stop, _cache_commit_len, pop_committed_kv_cache, pop_overallocated_kv_cache
  - `python/sglang/srt/server_args.py` modified +8/-0 (8 lines); hunks: -436,6 +436,7 @@ class ServerArgs:; -4879,6 +4880,13 @@ def add_cli_args(parser: argparse.ArgumentParser):; symbols: ServerArgs, add_cli_args
  - `python/sglang/srt/mem_cache/common.py` modified +3/-1 (4 lines); hunks: -489,7 +489,9 @@ def release_kv_cache(req: Req, tree_cache: BasePrefixCache,...; symbols: release_kv_cache
- Key code excerpts:

```diff
diff -- test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py
@@ -30,7 +30,11 @@
-from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
+from sglang.srt.server_args import (
+    ServerArgs,
+    get_global_server_args,
+    set_global_server_args_for_scheduler,
+)
diff -- python/sglang/srt/managers/schedule_batch.py
@@ -903,13 +903,20 @@ def output_ids_through_stop(self) -> List[int]:
+    def _cache_commit_len(self) -> int:
+        # Report only the prompt prefix so thinking + answer fall into the
+        # overallocated range and are reclaimed by release_kv_cache. #22373.
+        if get_global_server_args().strip_thinking_cache and self.reasoning_tokens > 0:
+            return min(self.kv_committed_len, len(self.origin_input_ids))
+        return self.kv_committed_len
diff -- python/sglang/srt/server_args.py
@@ -436,6 +436,7 @@ class ServerArgs:
```

- Reviewed files:
  - tests: `test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py` modified +52/-1
  - runtime: `python/sglang/srt/managers/schedule_batch.py` modified +9/-2; `python/sglang/srt/server_args.py` modified +8/-0; `python/sglang/srt/mem_cache/common.py` modified +3/-1
- Risk and verification: The diff ships test coverage in `test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22950 - [fix] Parser-gated two-phase cache stripping for reasoning radix caches (fixes #22373)

- Link: https://github.com/sgl-project/sglang/pull/22950
- Status/date: closed / 2026-04-21
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 11 files, +597/-64, 850 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[fix] Parser-gated two-phase cache stripping for reasoning radix caches (fixes #22373)"; model line: DeepSeek V3.1; category: bug fix; main diff: `python/sglang/srt/parser/reasoning_parser.py`, `python/sglang/srt/configs/model_config.py`, `test/registered/unit/mem_cache/test_radix_cache_thinking.py`; technical summary: Covers "[fix] Parser-gated two-phase cache stripping for reasoning radix caches (fixes #22373)"; the main implementation surface is `python/sglang/srt/parser/reasoning_parser.py`, `python/sglang/srt/configs/model_config.py`, `test/registered/unit/mem_cache/test_radix_cache_thinking.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/parser/reasoning_parser.py` modified +8/-0 (8 lines); hunks: -19,6 +19,10 @@ def __init__(; -395,6 +399,10 @@ class MiniMaxAppendThinkDetector(BaseReasoningFormatDetector):; symbols: __init__, BaseReasoningFormatDetector, providing, MiniMaxAppendThinkDetector, touching `__init__, BaseReasoningFormatDetector, providing`; `python/sglang/srt/configs/model_config.py` modified +1/-0 (1 lines); hunks: -242,6 +242,7 @@ def __init__(; symbols: __init__, touching `__init__`; `test/registered/unit/mem_cache/test_radix_cache_thinking.py` added +238/-0 (238 lines); hunks: -0,0 +1,238; symbols: _MockReqToTokenPool, __init__, write, _MockAllocator, touching `_MockReqToTokenPool, __init__, write`; `test/registered/unit/mem_cache/test_radix_cache_thinking_gated.py` added +220/-0 (220 lines); hunks: -0,0 +1,220; symbols: _MockReqToTokenPool, __init__, write, _MockAllocator, touching `_MockReqToTokenPool, __init__, write`.
- Code diff details:
  - `python/sglang/srt/parser/reasoning_parser.py` modified +8/-0 (8 lines); hunks: -19,6 +19,10 @@ def __init__(; -395,6 +399,10 @@ class MiniMaxAppendThinkDetector(BaseReasoningFormatDetector):; symbols: __init__, BaseReasoningFormatDetector, providing, MiniMaxAppendThinkDetector
  - `python/sglang/srt/configs/model_config.py` modified +1/-0 (1 lines); hunks: -242,6 +242,7 @@ def __init__(; symbols: __init__
  - `test/registered/unit/mem_cache/test_radix_cache_thinking.py` added +238/-0 (238 lines); hunks: -0,0 +1,238; symbols: _MockReqToTokenPool, __init__, write, _MockAllocator
  - `test/registered/unit/mem_cache/test_radix_cache_thinking_gated.py` added +220/-0 (220 lines); hunks: -0,0 +1,220; symbols: _MockReqToTokenPool, __init__, write, _MockAllocator
  - `python/sglang/srt/mem_cache/mamba_radix_cache.py` modified +62/-50 (112 lines); hunks: -28,7 +28,6; -45,6 +44,7; symbols: cache_finished_req, _skip_cache_unfinished_req
- Key code excerpts:

```diff
diff -- python/sglang/srt/parser/reasoning_parser.py
@@ -19,6 +19,10 @@ def __init__(
+    # Most reasoning parsers separate hidden thinking from visible assistant
+    # content, so those tokens should not be cached across turns.
+    strip_thinking_from_cache: bool = True
@@ -395,6 +399,10 @@ class MiniMaxAppendThinkDetector(BaseReasoningFormatDetector):
+    # MiniMax appends thinking into visible assistant content, so future turns
+    # may include it verbatim and the full output should stay cacheable.
diff -- python/sglang/srt/configs/model_config.py
@@ -242,6 +242,7 @@ def __init__(
+        self.strip_thinking_from_cache: bool = True
diff -- test/registered/unit/mem_cache/test_radix_cache_thinking.py
@@ -0,0 +1,238 @@
+import unittest
+import torch
+from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
+from sglang.srt.mem_cache.cache_init_params import CacheInitParams
+from sglang.srt.mem_cache.common import maybe_strip_thinking_tokens
```

- Reviewed files:
  - runtime: `python/sglang/srt/parser/reasoning_parser.py` modified +8/-0; `python/sglang/srt/configs/model_config.py` modified +1/-0; `python/sglang/srt/mem_cache/mamba_radix_cache.py` modified +62/-50; `python/sglang/srt/mem_cache/radix_cache_cpp.py` modified +27/-14; `python/sglang/srt/mem_cache/common.py` modified +22/-0; `python/sglang/srt/mem_cache/radix_cache.py` modified +7/-0
  - tests: `test/registered/unit/mem_cache/test_radix_cache_thinking.py` added +238/-0; `test/registered/unit/mem_cache/test_radix_cache_thinking_gated.py` added +220/-0
- Risk and verification: The diff ships test coverage in `test/registered/unit/mem_cache/test_radix_cache_thinking.py`, `test/registered/unit/mem_cache/test_radix_cache_thinking_gated.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22774 - [MUSA][16/N] Add MUSA backend support for layers and DeepSeek models (V2/V3/R1)

- Link: https://github.com/sgl-project/sglang/pull/22774
- Status/date: merged / 2026-04-24
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 27 files, +184/-44, 795 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MUSA][16/N] Add MUSA backend support for layers and DeepSeek models (V2/V3/R1)"; model line: DeepSeek V3.1; category: model support/runtime entry; main diff: `python/sglang/srt/layers/layernorm.py`, `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py`; technical summary: Covers "[MUSA][16/N] Add MUSA backend support for layers and DeepSeek models (V2/V3/R1)"; the main implementation surface is `python/sglang/srt/layers/layernorm.py`, `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/layernorm.py` modified +26/-1 (27 lines); hunks: -34,21 +34,23; -284,6 +286,29 @@ def forward_hip(; symbols: forward_hip, forward_musa, forward_native, touching `forward_hip, forward_musa, forward_native`; `python/sglang/srt/layers/moe/topk.py` modified +12/-10 (22 lines); hunks: -62,6 +62,7; -80,8 +81,9; symbols: fused_topk_deepseek, biased_grouped_topk_gpu, select_experts, touching `fused_topk_deepseek, biased_grouped_topk_gpu, select_experts`; `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py` modified +14/-3 (17 lines); hunks: -1,6 +1,6; -14,10 +14,12; symbols: execute, deep_gemm_execution_hook, _deep_gemm_execution_hook, touching `execute, deep_gemm_execution_hook, _deep_gemm_execution_hook`; `python/sglang/srt/models/deepseek_v2.py` modified +13/-4 (17 lines); hunks: -141,6 +141,7; -182,6 +183,8; symbols: forward_normal_dual_stream, _post_combine_hook, __init__, determine_num_fused_shared_experts, touching `forward_normal_dual_stream, _post_combine_hook, __init__`.
- Code diff details:
  - `python/sglang/srt/layers/layernorm.py` modified +26/-1 (27 lines); hunks: -34,21 +34,23; -284,6 +286,29 @@ def forward_hip(; symbols: forward_hip, forward_musa, forward_native
  - `python/sglang/srt/layers/moe/topk.py` modified +12/-10 (22 lines); hunks: -62,6 +62,7; -80,8 +81,9; symbols: fused_topk_deepseek, biased_grouped_topk_gpu, select_experts
  - `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py` modified +14/-3 (17 lines); hunks: -1,6 +1,6; -14,10 +14,12; symbols: execute, deep_gemm_execution_hook, _deep_gemm_execution_hook
  - `python/sglang/srt/models/deepseek_v2.py` modified +13/-4 (17 lines); hunks: -141,6 +141,7; -182,6 +183,8; symbols: forward_normal_dual_stream, _post_combine_hook, __init__, determine_num_fused_shared_experts
  - `python/sglang/srt/layers/moe/ep_moe/kernels.py` modified +11/-3 (14 lines); hunks: -3,12 +3,14; -665,6 +667,8 @@ def _fwd_kernel_ep_scatter_2(; symbols: _fwd_kernel_ep_scatter_2, ep_scatter
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/layernorm.py
@@ -34,21 +34,23 @@
+    is_musa,
+_is_musa = is_musa()
-if _is_cuda or _is_xpu:
+if _is_cuda or _is_xpu or _is_musa:
@@ -284,6 +286,29 @@ def forward_hip(
+    def forward_musa(
diff -- python/sglang/srt/layers/moe/topk.py
@@ -62,6 +62,7 @@
+    is_musa,
@@ -80,8 +81,9 @@
+_is_musa = is_musa()
-if _is_cuda:
+if _is_cuda or _is_musa:
@@ -124,7 +126,7 @@ def fused_topk_deepseek(
diff -- python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py
@@ -1,6 +1,6 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/layernorm.py` modified +26/-1; `python/sglang/srt/layers/moe/topk.py` modified +12/-10; `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py` modified +14/-3; `python/sglang/srt/models/deepseek_v2.py` modified +13/-4; `python/sglang/srt/layers/moe/ep_moe/kernels.py` modified +11/-3; `python/sglang/srt/layers/activation.py` modified +13/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/environ.py`, `python/sglang/srt/layers/activation.py`, `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23732 - Apply should_use_dp_reduce_scatterv guard to remaining MoE models (follow-up to #23731)

- Link: https://github.com/sgl-project/sglang/pull/23732
- Status/date: merged / 2026-04-26
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 13 files, +59/-12, 290 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Apply should_use_dp_reduce_scatterv guard to remaining MoE models (follow-up to #23731)"; model line: DeepSeek V3.1; category: model implementation change; main diff: `python/sglang/srt/models/llada2.py`, `python/sglang/srt/models/hunyuan_v3.py`, `python/sglang/srt/models/bailing_moe_linear.py`; technical summary: Covers "Apply should_use_dp_reduce_scatterv guard to remaining MoE models (follow-up to #23731)"; the main implementation surface is `python/sglang/srt/models/llada2.py`, `python/sglang/srt/models/hunyuan_v3.py`, `python/sglang/srt/models/bailing_moe_linear.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #23748 - refactor(moe): centralize post-experts all-reduce skip predicate

- Link: https://github.com/sgl-project/sglang/pull/23748
- Status/date: merged / 2026-04-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 17 files, +134/-132, 532 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "refactor(moe): centralize post-experts all-reduce skip predicate"; model line: DeepSeek V3.1; category: model implementation change; main diff: `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/models/sarvam_moe.py`, `python/sglang/srt/models/deepseek_v2.py`; technical summary: Covers "refactor(moe): centralize post-experts all-reduce skip predicate"; the main implementation surface is `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/models/sarvam_moe.py`, `python/sglang/srt/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #23268 - 【NPU】【bugfix】accuracy fix when enable both nsa cp and prefixcache

- Link: https://github.com/sgl-project/sglang/pull/23268
- Status/date: merged / 2026-04-28
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +21/-5, 40 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "【NPU】【bugfix】accuracy fix when enable both nsa cp and prefixcache"; model line: DeepSeek V3.1; category: bug fix; main diff: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`; technical summary: Covers "【NPU】【bugfix】accuracy fix when enable both nsa cp and prefixcache"; the main implementation surface is `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +18/-4 (22 lines); hunks: -1463,10 +1463,24 @@ def forward_npu(; symbols: forward_npu, touching `forward_npu`; `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` modified +3/-1 (4 lines); hunks: -359,7 +359,9 @@ def forward_dsa_prepare_npu(; symbols: forward_dsa_prepare_npu, touching `forward_dsa_prepare_npu`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +18/-4 (22 lines); hunks: -1463,10 +1463,24 @@ def forward_npu(; symbols: forward_npu
  - `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` modified +3/-1 (4 lines); hunks: -359,7 +359,9 @@ def forward_dsa_prepare_npu(; symbols: forward_dsa_prepare_npu
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -1463,10 +1463,24 @@ def forward_npu(
-                forward_batch.attn_backend.forward_metadata.actual_seq_lengths_kv = (
-                    forward_batch.attn_cp_metadata.kv_len_prev_tensor,
-                    forward_batch.attn_cp_metadata.kv_len_next_tensor,
-                )
+                if sum(forward_batch.extend_prefix_lens_cpu) > 0:
+                    total_kv_len_prev_tensor = (
diff -- python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py
@@ -359,7 +359,9 @@ def forward_dsa_prepare_npu(
-            if fused_qkv_a_proj_out.shape[0] < 65535:
+            if fused_qkv_a_proj_out.shape[0] < 65535 and not nsa_use_prefill_cp(
+                forward_batch
+            ):
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +18/-4; `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` modified +3/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23285 - [Flashinfer] Integrate flashinfer router gemm for sm103

- Link: https://github.com/sgl-project/sglang/pull/23285
- Status/date: merged / 2026-04-28
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Flashinfer] Integrate flashinfer router gemm for sm103"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v2.py`; technical summary: Covers "[Flashinfer] Integrate flashinfer router gemm for sm103"; the main implementation surface is `python/sglang/srt/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +1/-1 (2 lines); hunks: -334,7 +334,7 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +1/-1 (2 lines); hunks: -334,7 +334,7 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -334,7 +334,7 @@ def forward(
-                if _device_sm == 100 and self.weight.shape[0] == 256:
+                if _device_sm in [100, 103] and self.weight.shape[0] == 256:
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21062 - Use spec v2 by default

- Link: https://github.com/sgl-project/sglang/pull/21062
- Status/date: merged / 2026-04-29
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 27 files, +411/-205, 944 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Use spec v2 by default"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `test/registered/4-gpu-models/test_qwen35_models.py`, `test/registered/ep/test_deepep_large.py`, `test/registered/spec/eagle/test_adaptive_speculative.py`; technical summary: Covers "Use spec v2 by default"; the main implementation surface is `test/registered/4-gpu-models/test_qwen35_models.py`, `test/registered/ep/test_deepep_large.py`, `test/registered/spec/eagle/test_adaptive_speculative.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/4-gpu-models/test_qwen35_models.py` added +242/-0 (242 lines); hunks: -0,0 +1,242; symbols: TestQwen35FP4, test_gsm8k, TestQwen35FP4MTP, setUpClass, touching `TestQwen35FP4, test_gsm8k, TestQwen35FP4MTP`; `test/registered/ep/test_deepep_large.py` modified +44/-42 (86 lines); hunks: -3,6 +3,7; -86,48 +87,49 @@ class TestDeepseekMTP(CustomTestCase):; symbols: TestDeepseekMTP, setUpClass, tearDownClass, touching `TestDeepseekMTP, setUpClass, tearDownClass`; `test/registered/spec/eagle/test_adaptive_speculative.py` modified +28/-26 (54 lines); hunks: -6,6 +6,7; -58,32 +59,33 @@ def setUpClass(cls):; symbols: setUpClass, touching `setUpClass`; `test/registered/8-gpu-models/test_dsa_models_mtp.py` modified +24/-29 (53 lines); hunks: -3,7 +3,6; -48,13 +47,12 @@ def setUpClass(cls):; symbols: setUpClass, tearDownClass, touching `setUpClass, tearDownClass`.
- Code diff details:
  - `test/registered/4-gpu-models/test_qwen35_models.py` added +242/-0 (242 lines); hunks: -0,0 +1,242; symbols: TestQwen35FP4, test_gsm8k, TestQwen35FP4MTP, setUpClass
  - `test/registered/ep/test_deepep_large.py` modified +44/-42 (86 lines); hunks: -3,6 +3,7; -86,48 +87,49 @@ class TestDeepseekMTP(CustomTestCase):; symbols: TestDeepseekMTP, setUpClass, tearDownClass
  - `test/registered/spec/eagle/test_adaptive_speculative.py` modified +28/-26 (54 lines); hunks: -6,6 +6,7; -58,32 +59,33 @@ def setUpClass(cls):; symbols: setUpClass
  - `test/registered/8-gpu-models/test_dsa_models_mtp.py` modified +24/-29 (53 lines); hunks: -3,7 +3,6; -48,13 +47,12 @@ def setUpClass(cls):; symbols: setUpClass, tearDownClass
  - `python/sglang/srt/server_args.py` modified +17/-24 (41 lines); hunks: -1962,11 +1962,6 @@ def _handle_model_specific_adjustments(self):; -1983,11 +1978,6 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments, _handle_speculative_decoding
- Key code excerpts:

```diff
diff -- test/registered/4-gpu-models/test_qwen35_models.py
@@ -0,0 +1,242 @@
+import unittest
+from types import SimpleNamespace
+import requests
+from sglang.srt.utils import kill_process_tree
+from sglang.test.accuracy_test_runner import AccuracyTestParams
+from sglang.test.ci.ci_register import register_cuda_ci
diff -- test/registered/ep/test_deepep_large.py
@@ -3,6 +3,7 @@
+from sglang.srt.environ import envs
@@ -86,48 +87,49 @@ class TestDeepseekMTP(CustomTestCase):
-        cls.process = popen_launch_server(
-            cls.model,
-            cls.base_url,
-            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
diff -- test/registered/spec/eagle/test_adaptive_speculative.py
@@ -6,6 +6,7 @@
```

- Reviewed files:
  - tests: `test/registered/4-gpu-models/test_qwen35_models.py` added +242/-0; `test/registered/ep/test_deepep_large.py` modified +44/-42; `test/registered/spec/eagle/test_adaptive_speculative.py` modified +28/-26; `test/registered/8-gpu-models/test_dsa_models_mtp.py` modified +24/-29; `test/registered/cp/test_deepseek_v32_cp_single_node.py` modified +12/-15; `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py` modified +12/-15
  - runtime: `python/sglang/srt/server_args.py` modified +17/-24
- Risk and verification: The diff ships test coverage in `test/manual/ascend/test_ascend_deepseek_mtp.py`, `test/manual/test_deepseek_v31.py`, `test/manual/test_glm_46_fp8.py`, `test/manual/test_qwen3_235b.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #21126 - [4/N] Quantization Refactor: AWQ schemes and Kernel call and weight init split

- Link: https://github.com/sgl-project/sglang/pull/21126
- Status/date: merged / 2026-04-30
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 20 files, +1419/-1031, 2590 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[4/N] Quantization Refactor: AWQ schemes and Kernel call and weight init split"; model line: DeepSeek V3.1; category: model implementation change; main diff: `python/sglang/srt/layers/quantization/awq.py`, `python/sglang/srt/layers/quantization/awq/awq.py`, `python/sglang/srt/layers/quantization/awq/schemes/awq_moe.py`; technical summary: Covers "[4/N] Quantization Refactor: AWQ schemes and Kernel call and weight init split"; the main implementation surface is `python/sglang/srt/layers/quantization/awq.py`, `python/sglang/srt/layers/quantization/awq/awq.py`, `python/sglang/srt/layers/quantization/awq/schemes/awq_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/quantization/awq.py` removed +0/-966 (966 lines); hunks: -1,966 +0,0; symbols: is_layer_skipped_awq, AWQConfig, for, __init__, touching `is_layer_skipped_awq, AWQConfig, for`; `python/sglang/srt/layers/quantization/awq/awq.py` added +484/-0 (484 lines); hunks: -0,0 +1,484; symbols: is_layer_skipped_awq, AWQConfig, for, __init__, touching `is_layer_skipped_awq, AWQConfig, for`; `python/sglang/srt/layers/quantization/awq/schemes/awq_moe.py` added +156/-0 (156 lines); hunks: -0,0 +1,156; symbols: AWQMoEScheme, __init__, _init_kernel, create_weights, touching `AWQMoEScheme, __init__, _init_kernel`; `python/sglang/srt/layers/quantization/awq/schemes/awq_linear.py` added +110/-0 (110 lines); hunks: -0,0 +1,110; symbols: AWQLinearScheme, __init__, _init_kernel, create_weights, touching `AWQLinearScheme, __init__, _init_kernel`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/awq.py` removed +0/-966 (966 lines); hunks: -1,966 +0,0; symbols: is_layer_skipped_awq, AWQConfig, for, __init__
  - `python/sglang/srt/layers/quantization/awq/awq.py` added +484/-0 (484 lines); hunks: -0,0 +1,484; symbols: is_layer_skipped_awq, AWQConfig, for, __init__
  - `python/sglang/srt/layers/quantization/awq/schemes/awq_moe.py` added +156/-0 (156 lines); hunks: -0,0 +1,156; symbols: AWQMoEScheme, __init__, _init_kernel, create_weights
  - `python/sglang/srt/layers/quantization/awq/schemes/awq_linear.py` added +110/-0 (110 lines); hunks: -0,0 +1,110; symbols: AWQLinearScheme, __init__, _init_kernel, create_weights
  - `python/sglang/srt/layers/quantization/awq/schemes/awq_marlin.py` added +105/-0 (105 lines); hunks: -0,0 +1,105; symbols: AWQMarlinLinearScheme, __init__, create_weights, process_weights_after_loading
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/awq.py
@@ -1,966 +0,0 @@
-# SPDX-License-Identifier: Apache-2.0
-from __future__ import annotations
-import logging
-import warnings
-from typing import TYPE_CHECKING, Any, Dict, List, Optional
-import torch
diff -- python/sglang/srt/layers/quantization/awq/awq.py
@@ -0,0 +1,484 @@
+# SPDX-License-Identifier: Apache-2.0
+from __future__ import annotations
+import logging
+import warnings
+from typing import TYPE_CHECKING, Any, Dict, List, Optional
+import torch
diff -- python/sglang/srt/layers/quantization/awq/schemes/awq_moe.py
@@ -0,0 +1,156 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/awq.py` removed +0/-966; `python/sglang/srt/layers/quantization/awq/awq.py` added +484/-0; `python/sglang/srt/layers/quantization/awq/schemes/awq_moe.py` added +156/-0; `python/sglang/srt/layers/quantization/awq/schemes/awq_linear.py` added +110/-0; `python/sglang/srt/layers/quantization/awq/schemes/awq_marlin.py` added +105/-0; `python/sglang/srt/layers/quantization/awq/schemes/awq_cpu.py` renamed +35/-51
- Risk and verification: The diff ships test coverage in `test/registered/quant/test_awq_dequant.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #23557 - [Intel GPU] Integrate flash_mla_decode in Intel XPU attention backend

- Link: https://github.com/sgl-project/sglang/pull/23557
- Status/date: merged / 2026-04-30
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +68/-66, 248 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Intel GPU] Integrate flash_mla_decode in Intel XPU attention backend"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/xpu_backend.py`, `python/sglang/srt/models/deepseek_common/attention_backend_handler.py`, `python/sglang/srt/model_executor/model_runner.py`; technical summary: Covers "[Intel GPU] Integrate flash_mla_decode in Intel XPU attention backend"; the main implementation surface is `python/sglang/srt/layers/attention/xpu_backend.py`, `python/sglang/srt/models/deepseek_common/attention_backend_handler.py`, `python/sglang/srt/model_executor/model_runner.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/xpu_backend.py` modified +31/-62 (93 lines); hunks: -14,12 +14,13; -30,7 +31,7 @@ class XPUAttentionBackend(AttentionBackend):; symbols: XPUAttentionBackend, __init__, init_forward_metadata, forward_decode, touching `XPUAttentionBackend, __init__, init_forward_metadata`; `python/sglang/srt/models/deepseek_common/attention_backend_handler.py` modified +5/-0 (5 lines); hunks: -172,6 +172,10 @@ def handle_attention_triton(attn, forward_batch):; -182,3 +186,4 @@ def handle_attention_triton(attn, forward_batch):; symbols: handle_attention_triton, handle_attention_intel_xpu, touching `handle_attention_triton, handle_attention_intel_xpu`; `python/sglang/srt/model_executor/model_runner.py` modified +1/-0 (1 lines); hunks: -233,6 +233,7; `python/sglang/srt/models/deepseek_common/utils.py` modified +1/-0 (1 lines); hunks: -62,6 +62,7.
- Code diff details:
  - `python/sglang/srt/layers/attention/xpu_backend.py` modified +31/-62 (93 lines); hunks: -14,12 +14,13; -30,7 +31,7 @@ class XPUAttentionBackend(AttentionBackend):; symbols: XPUAttentionBackend, __init__, init_forward_metadata, forward_decode
  - `python/sglang/srt/models/deepseek_common/attention_backend_handler.py` modified +5/-0 (5 lines); hunks: -172,6 +172,10 @@ def handle_attention_triton(attn, forward_batch):; -182,3 +186,4 @@ def handle_attention_triton(attn, forward_batch):; symbols: handle_attention_triton, handle_attention_intel_xpu
  - `python/sglang/srt/model_executor/model_runner.py` modified +1/-0 (1 lines); hunks: -233,6 +233,7
  - `python/sglang/srt/models/deepseek_common/utils.py` modified +1/-0 (1 lines); hunks: -62,6 +62,7
  - `python/sglang/srt/server_args.py` modified +16/-3 (19 lines); hunks: -2656,10 +2656,23 @@ def _handle_attention_backend_compatibility(self):; symbols: _handle_attention_backend_compatibility
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/xpu_backend.py
@@ -14,12 +14,13 @@
+from sglang.srt.utils import get_device_core_count
-from sgl_kernel import merge_state_v2
+from sgl_kernel import flash_mla_decode, flash_mla_get_workspace_size, merge_state_v2
@@ -30,7 +31,7 @@ class XPUAttentionBackend(AttentionBackend):
-    - MLA support
+    - MLA Prefill support
diff -- python/sglang/srt/models/deepseek_common/attention_backend_handler.py
@@ -172,6 +172,10 @@ def handle_attention_triton(attn, forward_batch):
+def handle_attention_intel_xpu(attn, forward_batch):
+    return _handle_attention_backend(attn, forward_batch, "intel_xpu")
@@ -182,3 +186,4 @@ def handle_attention_triton(attn, forward_batch):
+AttentionBackendRegistry.register("intel_xpu", handle_attention_intel_xpu)
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -233,6 +233,7 @@
+    "intel_xpu",
diff -- python/sglang/srt/models/deepseek_common/utils.py
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/xpu_backend.py` modified +31/-62; `python/sglang/srt/models/deepseek_common/attention_backend_handler.py` modified +5/-0; `python/sglang/srt/model_executor/model_runner.py` modified +1/-0; `python/sglang/srt/models/deepseek_common/utils.py` modified +1/-0; `python/sglang/srt/server_args.py` modified +16/-3
  - tests: `test/srt/xpu/test_intel_xpu_backend.py` modified +14/-1
- Risk and verification: The diff ships test coverage in `test/srt/xpu/test_intel_xpu_backend.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #23850 - Support RunAI loading for quantized checkpoints

- Link: https://github.com/sgl-project/sglang/pull/23850
- Status/date: merged / 2026-05-02
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +198/-37, 308 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support RunAI loading for quantized checkpoints"; model line: DeepSeek V3.1; category: docs/tests/CI; main diff: `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`, `test/registered/unit/model_loader/test_runai_model_streamer_loader.py`; technical summary: Covers "Support RunAI loading for quantized checkpoints"; the main implementation surface is `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`, `test/registered/unit/model_loader/test_runai_model_streamer_loader.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/kimi_k25.py` modified +39/-32 (71 lines); hunks: -743,42 +743,49 @@ def forward(; symbols: forward, load_weights, stream_language_weights, get_model_config_for_expert_location, touching `forward, load_weights, stream_language_weights`; `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +13/-2 (15 lines); hunks: -44,7 +44,10; -67,6 +70,12; symbols: _clone_if_runai_streamed_tensor, NextNEnabledConfig, do_load_weights, touching `_clone_if_runai_streamed_tensor, NextNEnabledConfig, do_load_weights`; `test/registered/unit/model_loader/test_runai_model_streamer_loader.py` added +128/-0 (128 lines); hunks: -0,0 +1,128; symbols: _FakeModel, eval, TestRunaiModelStreamerLoader, test_passes_quant_config_to_model_init, touching `_FakeModel, eval, TestRunaiModelStreamerLoader`; `python/sglang/srt/model_loader/loader.py` modified +13/-2 (15 lines); hunks: -3138,11 +3138,13 @@ def load_model(; -3160,7 +3162,16 @@ def get_model_loader(; symbols: load_model, get_model_loader, touching `load_model, get_model_loader`.
- Code diff details:
  - `python/sglang/srt/models/kimi_k25.py` modified +39/-32 (71 lines); hunks: -743,42 +743,49 @@ def forward(; symbols: forward, load_weights, stream_language_weights, get_model_config_for_expert_location
  - `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +13/-2 (15 lines); hunks: -44,7 +44,10; -67,6 +70,12; symbols: _clone_if_runai_streamed_tensor, NextNEnabledConfig, do_load_weights
  - `test/registered/unit/model_loader/test_runai_model_streamer_loader.py` added +128/-0 (128 lines); hunks: -0,0 +1,128; symbols: _FakeModel, eval, TestRunaiModelStreamerLoader, test_passes_quant_config_to_model_init
  - `python/sglang/srt/model_loader/loader.py` modified +13/-2 (15 lines); hunks: -3138,11 +3138,13 @@ def load_model(; -3160,7 +3162,16 @@ def get_model_loader(; symbols: load_model, get_model_loader
  - `python/sglang/srt/model_loader/weight_utils.py` modified +5/-1 (6 lines); hunks: -69,6 +69,8; -1317,7 +1319,9 @@ def runai_safetensors_weights_iterator(; symbols: runai_safetensors_weights_iterator, set_runai_streamer_env
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/kimi_k25.py
@@ -743,42 +743,49 @@ def forward(
-        """Load weights for the model, separating vision and language weights"""
+        """Stream weights, loading vision weights inline and yielding language weights.
+        The streaming pattern (vs accumulating into lists) is required because RunAI's
+        iterator reuses backing buffers — collecting tensors before consuming them
+        would clobber prior tensors.
+        """
diff -- python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py
@@ -44,7 +44,10 @@
-from sglang.srt.model_loader.weight_utils import default_weight_loader
+from sglang.srt.model_loader.weight_utils import (
+    RUNAI_STREAMER_TENSOR_ATTR,
+    default_weight_loader,
+)
@@ -67,6 +70,12 @@
diff -- test/registered/unit/model_loader/test_runai_model_streamer_loader.py
@@ -0,0 +1,128 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/kimi_k25.py` modified +39/-32; `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +13/-2; `python/sglang/srt/model_loader/loader.py` modified +13/-2; `python/sglang/srt/model_loader/weight_utils.py` modified +5/-1
  - tests: `test/registered/unit/model_loader/test_runai_model_streamer_loader.py` added +128/-0
- Risk and verification: The diff ships test coverage in `test/registered/unit/model_loader/test_runai_model_streamer_loader.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #21247 - [Dependency] Upgrade to Torch 2.11.0

- Link: https://github.com/sgl-project/sglang/pull/21247
- Status/date: merged / 2026-05-02
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 21 files, +658/-211, 1336 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Dependency] Upgrade to Torch 2.11.0"; model line: DeepSeek V3.1; category: docs/tests/CI; main diff: `.github/workflows/pr-test-multimodal-gen.yml`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/entrypoints/engine.py`; technical summary: Covers "[Dependency] Upgrade to Torch 2.11.0"; the main implementation surface is `.github/workflows/pr-test-multimodal-gen.yml`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/entrypoints/engine.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `.github/workflows/pr-test-multimodal-gen.yml` modified +24/-4 (28 lines); hunks: -129,6 +129,7 @@ jobs:; -146,6 +147,15 @@ jobs:; `python/sglang/srt/models/deepseek_v2.py` modified +6/-0 (6 lines); hunks: -600,6 +600,12 @@ def forward(; symbols: forward, touching `forward`; `python/sglang/srt/entrypoints/engine.py` modified +1/-1 (2 lines); hunks: -1170,7 +1170,7 @@ def _set_envs_and_config(server_args: ServerArgs):; symbols: _set_envs_and_config, touching `_set_envs_and_config`; `.github/workflows/pr-test-npu.yml` modified +56/-26 (82 lines); hunks: -327,15 +327,25 @@ jobs:; -376,15 +386,25 @@ jobs:.
- Code diff details:
  - `.github/workflows/pr-test-multimodal-gen.yml` modified +24/-4 (28 lines); hunks: -129,6 +129,7 @@ jobs:; -146,6 +147,15 @@ jobs:
  - `python/sglang/srt/models/deepseek_v2.py` modified +6/-0 (6 lines); hunks: -600,6 +600,12 @@ def forward(; symbols: forward
  - `python/sglang/srt/entrypoints/engine.py` modified +1/-1 (2 lines); hunks: -1170,7 +1170,7 @@ def _set_envs_and_config(server_args: ServerArgs):; symbols: _set_envs_and_config
  - `.github/workflows/pr-test-npu.yml` modified +56/-26 (82 lines); hunks: -327,15 +327,25 @@ jobs:; -376,15 +386,25 @@ jobs:
  - `.github/workflows/pr-test-jit-kernel.yml` modified +63/-4 (67 lines); hunks: -6,6 +6,9 @@ on:; -56,10 +59,24 @@ jobs:
- Key code excerpts:

```diff
diff -- .github/workflows/pr-test-multimodal-gen.yml
@@ -129,6 +129,7 @@ jobs:
+          SGLANG_DIFFUSION_ARTIFACT_DIR: ${{ github.workspace }}/diffusion-failures
@@ -146,6 +147,15 @@ jobs:
+      - name: Upload diffusion failure artifacts
+        if: always()
+        uses: actions/upload-artifact@v4
+        with:
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -600,6 +600,12 @@ def forward(
+                and not (
+                    get_global_server_args().enable_torch_compile
+                    and hidden_states.shape[0]
+                    <= get_global_server_args().torch_compile_max_bs
+                    * (get_global_server_args().speculative_num_draft_tokens or 1)
+                )
diff -- python/sglang/srt/entrypoints/engine.py
@@ -1170,7 +1170,7 @@ def _set_envs_and_config(server_args: ServerArgs):
```

- Reviewed files:
  - runtime: `.github/workflows/pr-test-multimodal-gen.yml` modified +24/-4; `python/sglang/srt/models/deepseek_v2.py` modified +6/-0; `python/sglang/srt/entrypoints/engine.py` modified +1/-1; `python/pyproject.toml` modified +4/-21
  - ci: `.github/workflows/pr-test-npu.yml` modified +56/-26; `.github/workflows/pr-test-jit-kernel.yml` modified +63/-4
  - other: `scripts/ci/cuda/ci_install_dependency.sh` modified +5/-45; `scripts/ci/cuda/cache_nvidia_wheels.sh` removed +0/-44
- Risk and verification: The diff ships test coverage in `test/manual/test_w4a8_deepseek_v3.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #24392 - add indexer-topk capture (V3.2 NSA + infra)

- Link: https://github.com/sgl-project/sglang/pull/24392
- Status/date: merged / 2026-05-05
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 19 files, +428/-18, 785 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "add indexer-topk capture (V3.2 NSA + infra)"; model line: DeepSeek V3.1; category: model support/runtime entry; main diff: `python/sglang/srt/layers/attention/indexer_topk_capturer.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; technical summary: Covers "add indexer-topk capture (V3.2 NSA + infra)"; the main implementation surface is `python/sglang/srt/layers/attention/indexer_topk_capturer.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/indexer_topk_capturer.py` added +103/-0 (103 lines); hunks: -0,0 +1,103; symbols: IndexerTopkCapturer, __init__, get_global_indexer_capturer, set_global_indexer_capturer, touching `IndexerTopkCapturer, __init__, get_global_indexer_capturer`; `python/sglang/srt/model_executor/model_runner.py` modified +49/-1 (50 lines); hunks: -55,7 +55,12; -105,6 +110,11; symbols: ModelRunnerOutput, ModelRunner, initialize, init_routed_experts_capturer, touching `ModelRunnerOutput, ModelRunner, initialize`; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +27/-15 (42 lines); hunks: -12,6 +12,9; -1121,15 +1124,18 @@ def forward_cuda(; symbols: forward_cuda, forward_npu, touching `forward_cuda, forward_npu`; `python/sglang/srt/configs/model_config.py` modified +13/-0 (13 lines); hunks: -129,6 +129,19 @@ def get_nsa_index_n_heads(config: PretrainedConfig) -> int:; symbols: get_nsa_index_n_heads, get_num_indexer_layers, ModelConfig, __init__, touching `get_nsa_index_n_heads, get_num_indexer_layers, ModelConfig`.
- Code diff details:
  - `python/sglang/srt/layers/attention/indexer_topk_capturer.py` added +103/-0 (103 lines); hunks: -0,0 +1,103; symbols: IndexerTopkCapturer, __init__, get_global_indexer_capturer, set_global_indexer_capturer
  - `python/sglang/srt/model_executor/model_runner.py` modified +49/-1 (50 lines); hunks: -55,7 +55,12; -105,6 +110,11; symbols: ModelRunnerOutput, ModelRunner, initialize, init_routed_experts_capturer
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +27/-15 (42 lines); hunks: -12,6 +12,9; -1121,15 +1124,18 @@ def forward_cuda(; symbols: forward_cuda, forward_npu
  - `python/sglang/srt/configs/model_config.py` modified +13/-0 (13 lines); hunks: -129,6 +129,19 @@ def get_nsa_index_n_heads(config: PretrainedConfig) -> int:; symbols: get_nsa_index_n_heads, get_num_indexer_layers, ModelConfig, __init__
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +11/-2 (13 lines); hunks: -6,6 +6,9; -205,7 +208,11 @@ def forward_absorb_prepare(; symbols: forward_absorb_prepare
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/indexer_topk_capturer.py
@@ -0,0 +1,103 @@
+import logging
+from typing import Optional
+import numpy as np
+import pybase64
+import torch
+from sglang.srt.layers.dp_attention import get_attention_tp_size
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -55,7 +55,12 @@
-from sglang.srt.configs.model_config import AttentionArch, ModelConfig, ModelImpl
+from sglang.srt.configs.model_config import (
+    AttentionArch,
+    ModelConfig,
+    ModelImpl,
+    get_num_indexer_layers,
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -12,6 +12,9 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/indexer_topk_capturer.py` added +103/-0; `python/sglang/srt/model_executor/model_runner.py` modified +49/-1; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +27/-15; `python/sglang/srt/configs/model_config.py` modified +13/-0; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +11/-2; `python/sglang/srt/managers/tokenizer_manager.py` modified +7/-0
- Risk and verification: The diff ships test coverage in `test/registered/8-gpu-models/test_return_indexer_topk.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #24450 - move topk capturers to srt/state_capturer/

- Link: https://github.com/sgl-project/sglang/pull/24450
- Status/date: merged / 2026-05-05
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 13 files, +28/-28, 178 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "move topk capturers to srt/state_capturer/"; model line: DeepSeek V3.1; category: model implementation change; main diff: `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`; technical summary: Covers "move topk capturers to srt/state_capturer/"; the main implementation surface is `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/model_executor/model_runner.py` modified +11/-11 (22 lines); hunks: -110,11 +110,6; -126,15 +121,9; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +3/-3 (6 lines); hunks: -12,13 +12,13; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +3/-3 (6 lines); hunks: -6,9 +6,6; -29,6 +26,9; `python/sglang/srt/layers/moe/topk.py` modified +1/-1 (2 lines); hunks: -52,9 +52,9.
- Code diff details:
  - `python/sglang/srt/model_executor/model_runner.py` modified +11/-11 (22 lines); hunks: -110,11 +110,6; -126,15 +121,9
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +3/-3 (6 lines); hunks: -12,13 +12,13
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +3/-3 (6 lines); hunks: -6,9 +6,6; -29,6 +26,9
  - `python/sglang/srt/layers/moe/topk.py` modified +1/-1 (2 lines); hunks: -52,9 +52,9
  - `python/sglang/srt/managers/scheduler_output_processor_mixin.py` modified +4/-4 (8 lines); hunks: -7,11 +7,7; -25,6 +21,10
- Key code excerpts:

```diff
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -110,11 +110,6 @@
-from sglang.srt.layers.attention.indexer_topk_capturer import (
-    create_indexer_capturer,
-    get_global_indexer_capturer,
-    set_global_indexer_capturer,
-)
@@ -126,15 +121,9 @@
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -12,13 +12,13 @@
-from sglang.srt.layers.attention.indexer_topk_capturer import (
-    maybe_capture_indexer_topk,
-)
+from sglang.srt.state_capturer.indexer_topk import (
+    maybe_capture_indexer_topk,
+)
diff -- python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py
@@ -6,9 +6,6 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/model_executor/model_runner.py` modified +11/-11; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +3/-3; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +3/-3; `python/sglang/srt/layers/moe/topk.py` modified +1/-1; `python/sglang/srt/managers/scheduler_output_processor_mixin.py` modified +4/-4; `python/sglang/srt/hardware_backend/npu/moe/topk.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `test/registered/8-gpu-models/test_return_indexer_topk.py`, `test/registered/rl/test_return_routed_experts.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #24005 - [AMD] Enable dual-stream MoE on ROCm

- Link: https://github.com/sgl-project/sglang/pull/24005
- Status/date: merged / 2026-05-07
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +22/-1, 51 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Enable dual-stream MoE on ROCm"; model line: DeepSeek V3.1; category: model support/runtime entry; main diff: `python/sglang/srt/layers/moe/token_dispatcher/moriep.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/environ.py`; technical summary: Covers "[AMD] Enable dual-stream MoE on ROCm"; the main implementation surface is `python/sglang/srt/layers/moe/token_dispatcher/moriep.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/environ.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/moe/token_dispatcher/moriep.py` modified +12/-0 (12 lines); hunks: -916,6 +916,18 @@ def __init__(; symbols: __init__, touching `__init__`; `python/sglang/srt/models/deepseek_v2.py` modified +6/-1 (7 lines); hunks: -1915,7 +1915,12 @@ def __init__(; symbols: __init__, touching `__init__`; `python/sglang/srt/environ.py` modified +3/-0 (3 lines); hunks: -326,6 +326,9 @@ class Envs:; symbols: Envs, touching `Envs`; `docs/references/environment_variables.md` modified +1/-0 (1 lines); hunks: -116,6 +116,7 @@ SGLang supports various environment variables that can be us....
- Code diff details:
  - `python/sglang/srt/layers/moe/token_dispatcher/moriep.py` modified +12/-0 (12 lines); hunks: -916,6 +916,18 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/deepseek_v2.py` modified +6/-1 (7 lines); hunks: -1915,7 +1915,12 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/environ.py` modified +3/-0 (3 lines); hunks: -326,6 +326,9 @@ class Envs:; symbols: Envs
  - `docs/references/environment_variables.md` modified +1/-0 (1 lines); hunks: -116,6 +116,7 @@ SGLang supports various environment variables that can be us...
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/token_dispatcher/moriep.py
@@ -916,6 +916,18 @@ def __init__(
+        async_mode = self.deepep_mode.enable_low_latency()
+        if get_bool_env_var("SGLANG_ROCM_USE_MULTI_STREAM") and not async_mode:
+            logger.warning_once(
+                "SGLANG_ROCM_USE_MULTI_STREAM=1 is set but Mori AsyncLL is "
+                "not enabled (--deepep-mode=%s). The alt-stream overlap only "
+                "frees up CUs when dispatch/combine runs on the AsyncLL "
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -1915,7 +1915,12 @@ def __init__(
-            if _is_cuda or _is_musa or envs.SGLANG_NPU_USE_MULTI_STREAM.get()
+            if (
+                _is_cuda
+                or _is_musa
+                or envs.SGLANG_NPU_USE_MULTI_STREAM.get()
+                or envs.SGLANG_ROCM_USE_MULTI_STREAM.get()
diff -- python/sglang/srt/environ.py
@@ -326,6 +326,9 @@ class Envs:
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/token_dispatcher/moriep.py` modified +12/-0; `python/sglang/srt/models/deepseek_v2.py` modified +6/-1; `python/sglang/srt/environ.py` modified +3/-0
  - docs: `docs/references/environment_variables.md` modified +1/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/environ.py`, `python/sglang/srt/layers/moe/token_dispatcher/moriep.py`, `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23882 - Deepseek V4

- Link: https://github.com/sgl-project/sglang/pull/23882
- Status/date: merged / 2026-05-08
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 154 files, +24534/-712, 27836 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Deepseek V4"; model line: DeepSeek V3.1; category: model implementation change; main diff: `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/entrypoints/openai/encoding_dsv4.py`; technical summary: Covers "Deepseek V4"; the main implementation surface is `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/layers/attention/deepseek_v4_backend.py`, `python/sglang/srt/entrypoints/openai/encoding_dsv4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v4.py` added +1528/-0 (1528 lines); hunks: -0,0 +1,1528; symbols: _rms_normalize_kernel, rms_normalize_triton, MQALayer, __init__, touching `_rms_normalize_kernel, rms_normalize_triton, MQALayer`; `python/sglang/srt/layers/attention/deepseek_v4_backend.py` added +1255/-0 (1255 lines); hunks: -0,0 +1,1255; symbols: _pad_last_dim, _create_flashmla_metadata, _create_dummy_paged_compress_data, DSV4AttnMetadata, touching `_pad_last_dim, _create_flashmla_metadata, _create_dummy_paged_compress_data`; `python/sglang/srt/entrypoints/openai/encoding_dsv4.py` added +850/-0 (850 lines); hunks: -0,0 +1,850; symbols: to_json, tools_from_openai_format, tool_calls_from_openai_format, tool_calls_to_openai_format, touching `to_json, tools_from_openai_format, tool_calls_from_openai_format`; `python/sglang/srt/layers/mhc.py` added +643/-0 (643 lines); hunks: -0,0 +1,643; symbols: hc_split_sinkhorn_kernel, hc_split_sinkhorn_kernel_, hc_split_sinkhorn, mhc_pre_big_fuse_tilelang, touching `hc_split_sinkhorn_kernel, hc_split_sinkhorn_kernel_, hc_split_sinkhorn`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v4.py` added +1528/-0 (1528 lines); hunks: -0,0 +1,1528; symbols: _rms_normalize_kernel, rms_normalize_triton, MQALayer, __init__
  - `python/sglang/srt/layers/attention/deepseek_v4_backend.py` added +1255/-0 (1255 lines); hunks: -0,0 +1,1255; symbols: _pad_last_dim, _create_flashmla_metadata, _create_dummy_paged_compress_data, DSV4AttnMetadata
  - `python/sglang/srt/entrypoints/openai/encoding_dsv4.py` added +850/-0 (850 lines); hunks: -0,0 +1,850; symbols: to_json, tools_from_openai_format, tool_calls_from_openai_format, tool_calls_to_openai_format
  - `python/sglang/srt/layers/mhc.py` added +643/-0 (643 lines); hunks: -0,0 +1,643; symbols: hc_split_sinkhorn_kernel, hc_split_sinkhorn_kernel_, hc_split_sinkhorn, mhc_pre_big_fuse_tilelang
  - `python/sglang/srt/layers/attention/dsv4/indexer.py` added +562/-0 (562 lines); hunks: -0,0 +1,562; symbols: fp8_paged_mqa_logits_torch, topk_transform_512_pytorch_vectorized, _fused_scale_kernel, fused_scale
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -0,0 +1,1528 @@
+from __future__ import annotations
+import concurrent.futures
+import logging
+from typing import TYPE_CHECKING, Iterable, List, Literal, Optional, Set, Tuple
+import torch
+import torch.nn as nn
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend.py
@@ -0,0 +1,1255 @@
+from __future__ import annotations
+import enum
+import functools
+import logging
+from dataclasses import dataclass, field
+from typing import (
diff -- python/sglang/srt/entrypoints/openai/encoding_dsv4.py
@@ -0,0 +1,850 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v4.py` added +1528/-0; `python/sglang/srt/layers/attention/deepseek_v4_backend.py` added +1255/-0; `python/sglang/srt/entrypoints/openai/encoding_dsv4.py` added +850/-0; `python/sglang/srt/layers/mhc.py` added +643/-0; `python/sglang/srt/layers/attention/dsv4/indexer.py` added +562/-0; `python/sglang/srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py` added +461/-0
- Risk and verification: The diff ships test coverage in `python/sglang/test/kits/server_sanity_kit.py`, `python/sglang/test/test_utils.py`, `test/manual/dsv4/__init__.py`, `test/manual/dsv4/_common.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #23336 - [SPEC V2][2/N] feat: adaptive spec support spec v2

- Link: https://github.com/sgl-project/sglang/pull/23336
- Status/date: merged / 2026-05-08
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 12 files, +303/-73, 635 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[SPEC V2][2/N] feat: adaptive spec support spec v2"; model line: DeepSeek V3.1; category: docs/tests/CI; main diff: `python/sglang/srt/speculative/eagle_worker_v2.py`, `test/registered/spec/eagle/test_adaptive_speculative.py`, `python/sglang/srt/speculative/eagle_worker.py`; technical summary: Covers "[SPEC V2][2/N] feat: adaptive spec support spec v2"; the main implementation surface is `python/sglang/srt/speculative/eagle_worker_v2.py`, `test/registered/spec/eagle/test_adaptive_speculative.py`, `python/sglang/srt/speculative/eagle_worker.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/speculative/eagle_worker_v2.py` modified +187/-8 (195 lines); hunks: -30,8 +30,13; -65,6 +70,7; symbols: init_cuda_graphs, draft, __init__, touching `init_cuda_graphs, draft, __init__`; `test/registered/spec/eagle/test_adaptive_speculative.py` modified +26/-28 (54 lines); hunks: -6,7 +6,6; -59,33 +58,32 @@ def setUpClass(cls):; symbols: setUpClass, touching `setUpClass`; `python/sglang/srt/speculative/eagle_worker.py` modified +24/-18 (42 lines); hunks: -73,6 +73,7; -278,42 +279,47 @@ def init_cuda_graphs(self):; symbols: init_cuda_graphs, apply_runtime_state, build_adaptive_runtime_state, _override_worker_state, touching `init_cuda_graphs, apply_runtime_state, build_adaptive_runtime_state`; `python/sglang/srt/speculative/adaptive_spec_params.py` modified +25/-14 (39 lines); hunks: -9,6 +9,8; -32,11 +34,6 @@ def adaptive_unsupported_reason(server_args: ServerArgs) -> s...; symbols: adaptive_unsupported_reason, __init__, update, _recompute_params, touching `adaptive_unsupported_reason, __init__, update`.
- Code diff details:
  - `python/sglang/srt/speculative/eagle_worker_v2.py` modified +187/-8 (195 lines); hunks: -30,8 +30,13; -65,6 +70,7; symbols: init_cuda_graphs, draft, __init__
  - `test/registered/spec/eagle/test_adaptive_speculative.py` modified +26/-28 (54 lines); hunks: -6,7 +6,6; -59,33 +58,32 @@ def setUpClass(cls):; symbols: setUpClass
  - `python/sglang/srt/speculative/eagle_worker.py` modified +24/-18 (42 lines); hunks: -73,6 +73,7; -278,42 +279,47 @@ def init_cuda_graphs(self):; symbols: init_cuda_graphs, apply_runtime_state, build_adaptive_runtime_state, _override_worker_state
  - `python/sglang/srt/speculative/adaptive_spec_params.py` modified +25/-14 (39 lines); hunks: -9,6 +9,8; -32,11 +34,6 @@ def adaptive_unsupported_reason(server_args: ServerArgs) -> s...; symbols: adaptive_unsupported_reason, __init__, update, _recompute_params
  - `python/sglang/srt/managers/scheduler_output_processor_mixin.py` modified +9/-1 (10 lines); hunks: -385,8 +385,16 @@ def _resolve_spec_overlap_token_ids(; symbols: _resolve_spec_overlap_token_ids
- Key code excerpts:

```diff
diff -- python/sglang/srt/speculative/eagle_worker_v2.py
@@ -30,8 +30,13 @@
+from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
+from sglang.srt.speculative.adaptive_runtime_state import (
+    AdaptiveController,
+    SpecRuntimeState,
+)
@@ -65,6 +70,7 @@
diff -- test/registered/spec/eagle/test_adaptive_speculative.py
@@ -6,7 +6,6 @@
-from sglang.srt.environ import envs
@@ -59,33 +58,32 @@ def setUpClass(cls):
-            with envs.SGLANG_ENABLE_SPEC_V2.override(False):
-                cls.process = popen_launch_server(
-                    cls.model,
-                    cls.base_url,
diff -- python/sglang/srt/speculative/eagle_worker.py
@@ -73,6 +73,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/speculative/eagle_worker_v2.py` modified +187/-8; `python/sglang/srt/speculative/eagle_worker.py` modified +24/-18; `python/sglang/srt/speculative/adaptive_spec_params.py` modified +25/-14; `python/sglang/srt/managers/scheduler_output_processor_mixin.py` modified +9/-1; `python/sglang/srt/speculative/base_spec_worker.py` modified +8/-0; `python/sglang/srt/speculative/eagle_info_v2.py` modified +6/-1
  - tests: `test/registered/spec/eagle/test_adaptive_speculative.py` modified +26/-28; `test/registered/unit/spec/test_adaptive_spec_params.py` modified +4/-3
- Risk and verification: The diff ships test coverage in `test/registered/spec/eagle/test_adaptive_speculative.py`, `test/registered/unit/spec/test_adaptive_spec_params.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #24799 - [AMD] Fix DeepSeek import cascade by supporting both pre- and post-#2958 aiter `fused_qk_rmsnorm` APIs

- Link: https://github.com/sgl-project/sglang/pull/24799
- Status/date: merged / 2026-05-11
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +33/-3, 43 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Fix DeepSeek import cascade by supporting both pre- and post-#2958 aiter `fused_qk_rmsnorm` APIs"; model line: DeepSeek V3.1; category: bug fix; main diff: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`; technical summary: Covers "[AMD] Fix DeepSeek import cascade by supporting both pre- and post-#2958 aiter `fused_qk_rmsnorm` APIs"; the main implementation surface is `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +33/-3 (36 lines); hunks: -64,9 +64,39 @@ def bmm_fp8(A, B, A_scale, B_scale, dtype, out=None):; symbols: bmm_fp8, fused_qk_rmsnorm_bf16, touching `bmm_fp8, fused_qk_rmsnorm_bf16`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +33/-3 (36 lines); hunks: -64,9 +64,39 @@ def bmm_fp8(A, B, A_scale, B_scale, dtype, out=None):; symbols: bmm_fp8, fused_qk_rmsnorm_bf16
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py
@@ -64,9 +64,39 @@ def bmm_fp8(A, B, A_scale, B_scale, dtype, out=None):
-    from aiter.ops.fused_qk_norm_rope_cache_quant import (
-        fused_qk_rmsnorm as fused_qk_rmsnorm_bf16,
-    )
+    # aiter ROCm/aiter#2958 renamed the public `fused_qk_rmsnorm` in
+    # `aiter.ops.fused_qk_norm_rope_cache_quant` to a private `_fused_qk_rmsnorm`
+    # and introduced a unified entry point in `aiter.ops.fused_qk_rmsnorm_group_quant`
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +33/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #24949 - Deepseek-v4-Pro share expert tp1

- Link: https://github.com/sgl-project/sglang/pull/24949
- Status/date: merged / 2026-05-12
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +31/-17, 112 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Deepseek-v4-Pro share expert tp1"; model line: DeepSeek V3.1; category: model implementation change; main diff: `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/environ.py`; technical summary: Covers "Deepseek-v4-Pro share expert tp1"; the main implementation surface is `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/environ.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +26/-14 (40 lines); hunks: -534,6 +534,7 @@ def __init__(; -543,7 +544,19 @@ def __init__(; symbols: __init__, forward_normal_dual_stream, touching `__init__, forward_normal_dual_stream`; `python/sglang/srt/model_executor/model_runner.py` modified +4/-2 (6 lines); hunks: -1155,8 +1155,10 @@ def check_quantized_moe_compatibility(self):; symbols: check_quantized_moe_compatibility, touching `check_quantized_moe_compatibility`; `python/sglang/srt/environ.py` modified +1/-1 (2 lines); hunks: -611,7 +611,7 @@ class Envs:; symbols: Envs, touching `Envs`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +26/-14 (40 lines); hunks: -534,6 +534,7 @@ def __init__(; -543,7 +544,19 @@ def __init__(; symbols: __init__, forward_normal_dual_stream
  - `python/sglang/srt/model_executor/model_runner.py` modified +4/-2 (6 lines); hunks: -1155,8 +1155,10 @@ def check_quantized_moe_compatibility(self):; symbols: check_quantized_moe_compatibility
  - `python/sglang/srt/environ.py` modified +1/-1 (2 lines); hunks: -611,7 +611,7 @@ class Envs:; symbols: Envs
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -534,6 +534,7 @@ def __init__(
+        self._shared_expert_tp1 = False
@@ -543,7 +544,19 @@ def __init__(
-            # disable tp for shared experts when enable deepep moe, or with fp4 allgather
+            # Disable TP for shared experts for A2A/FP4 allgather paths, or when
+            # explicitly requested for DSV4 checkpoints whose shared scales are
+            # not divisible by the global TP size.
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -1155,8 +1155,10 @@ def check_quantized_moe_compatibility(self):
-                moe_intermediate_size // moe_tp_size
-            ) % weight_block_size_n != 0 and not _use_aiter:
+                not envs.SGLANG_SHARED_EXPERT_TP1.get()
+                and (moe_intermediate_size // moe_tp_size) % weight_block_size_n != 0
+                and not _use_aiter
+            ):
diff -- python/sglang/srt/environ.py
@@ -611,7 +611,7 @@ class Envs:
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +26/-14; `python/sglang/srt/model_executor/model_runner.py` modified +4/-2; `python/sglang/srt/environ.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/environ.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25120 - [env] Make max KV chunk capacity configurable via `SGLANG_MAX_KV_CHUNK_CAPACITY`

- Link: https://github.com/sgl-project/sglang/pull/25120
- Status/date: merged / 2026-05-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +10/-4, 56 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[env] Make max KV chunk capacity configurable via `SGLANG_MAX_KV_CHUNK_CAPACITY`"; model line: DeepSeek V3.1; category: docs/tests/CI; main diff: `python/sglang/srt/model_executor/forward_batch_deepseek_mha_mixin.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`, `docs_new/docs/references/environment_variables.mdx`; technical summary: Covers "[env] Make max KV chunk capacity configurable via `SGLANG_MAX_KV_CHUNK_CAPACITY`"; the main implementation surface is `python/sglang/srt/model_executor/forward_batch_deepseek_mha_mixin.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`, `docs_new/docs/references/environment_variables.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/model_executor/forward_batch_deepseek_mha_mixin.py` modified +2/-3 (5 lines); hunks: -7,6 +7,7; -44,9 +45,7 @@ class ForwardBatchDeepSeekMHAMixin:; symbols: ForwardBatchDeepSeekMHAMixin, get_max_chunk_capacity, set_prefix_chunk_idx, touching `ForwardBatchDeepSeekMHAMixin, get_max_chunk_capacity, set_prefix_chunk_idx`; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` modified +1/-1 (2 lines); hunks: -53,7 +53,7; `docs_new/docs/references/environment_variables.mdx` modified +5/-0 (5 lines); hunks: -137,6 +137,11 @@ SGLang supports various environment variables that can be u...; `docs/references/environment_variables.md` modified +1/-0 (1 lines); hunks: -32,6 +32,7 @@ SGLang supports various environment variables that can be used....
- Code diff details:
  - `python/sglang/srt/model_executor/forward_batch_deepseek_mha_mixin.py` modified +2/-3 (5 lines); hunks: -7,6 +7,7; -44,9 +45,7 @@ class ForwardBatchDeepSeekMHAMixin:; symbols: ForwardBatchDeepSeekMHAMixin, get_max_chunk_capacity, set_prefix_chunk_idx
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` modified +1/-1 (2 lines); hunks: -53,7 +53,7
  - `docs_new/docs/references/environment_variables.mdx` modified +5/-0 (5 lines); hunks: -137,6 +137,11 @@ SGLang supports various environment variables that can be u...
  - `docs/references/environment_variables.md` modified +1/-0 (1 lines); hunks: -32,6 +32,7 @@ SGLang supports various environment variables that can be used...
  - `python/sglang/srt/environ.py` modified +1/-0 (1 lines); hunks: -406,6 +406,7 @@ class Envs:; symbols: Envs
- Key code excerpts:

```diff
diff -- python/sglang/srt/model_executor/forward_batch_deepseek_mha_mixin.py
@@ -7,6 +7,7 @@
+from sglang.srt.environ import envs
@@ -44,9 +45,7 @@ class ForwardBatchDeepSeekMHAMixin:
-        # Maximum number of tokens in each chunk
-        # TODO: Should be changed to a better value, maybe passed through server args
-        return 128 * 1024
+        return envs.SGLANG_MAX_KV_CHUNK_CAPACITY.get()
diff -- python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py
@@ -53,7 +53,7 @@
-#   The maximum number of tokens in each kv chunk, 128 * 1024 by default (can be get with forward_batch.get_max_chunk_capacity())
+#   The maximum number of tokens in each kv chunk, 128 * 1024 by default (can be changed with SGLANG_MAX_KV_CHUNK_CAPACITY, or get with forward_batch.get_max_chunk_capacity())
diff -- docs_new/docs/references/environment_variables.mdx
@@ -137,6 +137,11 @@ SGLang supports various environment variables that can be used to configure its
+    <tr>
+      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>`SGLANG_MAX_KV_CHUNK_CAPACITY`</td>
+      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>Maximum number of tokens in each KV chunk for DeepSeek MHA chunked prefix cache</td>
+      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>`131072`</td>
```

- Reviewed files:
  - runtime: `python/sglang/srt/model_executor/forward_batch_deepseek_mha_mixin.py` modified +2/-3; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` modified +1/-1; `python/sglang/srt/environ.py` modified +1/-0
  - docs: `docs_new/docs/references/environment_variables.mdx` modified +5/-0; `docs/references/environment_variables.md` modified +1/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/environ.py`, `python/sglang/srt/model_executor/forward_batch_deepseek_mha_mixin.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #24148 - [AMD] Add _skip_rope_for_aiter_fused_mla method and check to avoid double rotating with gfx950 and Aiter backend

- Link: https://github.com/sgl-project/sglang/pull/24148
- Status/date: merged / 2026-05-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +14/-0, 29 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Add _skip_rope_for_aiter_fused_mla method and check to avoid double rotating with gfx950 and Aiter backend"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`; technical summary: Covers "[AMD] Add _skip_rope_for_aiter_fused_mla method and check to avoid double rotating with gfx950 and Aiter backend"; the main implementation surface is `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +14/-0 (14 lines); hunks: -322,10 +322,12 @@ def forward_absorb_prepare(; -634,3 +636,15 @@ def _skip_rope_for_nsa_tilelang_fused(self: DeepseekV2Atten...; symbols: forward_absorb_prepare, _skip_rope_for_nsa_tilelang_fused, _skip_rope_for_aiter_fused_mla, touching `forward_absorb_prepare, _skip_rope_for_nsa_tilelang_fused, _skip_rope_for_aiter_fused_mla`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +14/-0 (14 lines); hunks: -322,10 +322,12 @@ def forward_absorb_prepare(; -634,3 +636,15 @@ def _skip_rope_for_nsa_tilelang_fused(self: DeepseekV2Atten...; symbols: forward_absorb_prepare, _skip_rope_for_nsa_tilelang_fused, _skip_rope_for_aiter_fused_mla
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py
@@ -322,10 +322,12 @@ def forward_absorb_prepare(
+        skip_rope_for_aiter_fused_mla = self._skip_rope_for_aiter_fused_mla()
+            and (not skip_rope_for_aiter_fused_mla)
@@ -634,3 +636,15 @@ def _skip_rope_for_nsa_tilelang_fused(self: DeepseekV2AttentionMLA) -> bool:
+    def _skip_rope_for_aiter_fused_mla(self: DeepseekV2AttentionMLA) -> bool:
+        """
+        Skip rope in prepare and let the fused kernel in forward_absorb_core handle it,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +14/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #24125 - [AMD] Skip redundant CatArrayBatchedCopy in GLM-5 NSA TileLang decode

- Link: https://github.com/sgl-project/sglang/pull/24125
- Status/date: merged / 2026-05-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +60/-20, 110 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Skip redundant CatArrayBatchedCopy in GLM-5 NSA TileLang decode"; model line: DeepSeek V3.1; category: model implementation change; main diff: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; technical summary: Covers "[AMD] Skip redundant CatArrayBatchedCopy in GLM-5 NSA TileLang decode"; the main implementation surface is `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/layers/attention/nsa_backend.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +48/-18 (66 lines); hunks: -415,25 +415,55 @@ def forward_absorb_core(; symbols: forward_absorb_core, touching `forward_absorb_core`; `python/sglang/srt/layers/attention/nsa_backend.py` modified +12/-2 (14 lines); hunks: -1594,7 +1594,13 @@ def forward_decode(; -1643,7 +1649,11 @@ def forward_decode(; symbols: forward_decode, touching `forward_decode`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +48/-18 (66 lines); hunks: -415,25 +415,55 @@ def forward_absorb_core(; symbols: forward_absorb_core
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +12/-2 (14 lines); hunks: -1594,7 +1594,13 @@ def forward_decode(; -1643,7 +1649,11 @@ def forward_decode(; symbols: forward_decode
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py
@@ -415,25 +415,55 @@ def forward_absorb_core(
-                q_nope_fused = q_cat[..., : self.kv_lora_rank]
-                q_pe_fused = q_cat[..., self.kv_lora_rank :]
-                if llama_4_scaling is not None:
-                    q_nope_fused *= llama_4_scaling
-                attn_output = self.attn_mqa(
-                    q_nope_fused,
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -1594,7 +1594,13 @@ def forward_decode(
+            # Caller passed split q_nope / q_rope; we'll need to concat below if
+            # the chosen impl wants q_all.
+            q_all = None
+            # Caller passed already-concatenated q (q_all = q). Reuse it directly
+            # via a zero-copy view; the impl-specific blocks below will skip the
+            # otherwise redundant concat_mla_absorb_q_general call.
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +48/-18; `python/sglang/srt/layers/attention/nsa_backend.py` modified +12/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #24897 - Port fused SiLU+clamp+FP8 quant from DSV4 dev branch

- Link: https://github.com/sgl-project/sglang/pull/24897
- Status/date: merged / 2026-05-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +51/-6, 79 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Port fused SiLU+clamp+FP8 quant from DSV4 dev branch"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v2.py`; technical summary: Covers "Port fused SiLU+clamp+FP8 quant from DSV4 dev branch"; the main implementation surface is `python/sglang/srt/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +51/-6 (57 lines); hunks: -27,6 +27,10; -107,6 +111,9; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +51/-6 (57 lines); hunks: -27,6 +27,10; -107,6 +111,9; symbols: forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -27,6 +27,10 @@
+from sglang.jit_kernel.deepseek_v4 import (
+    silu_and_mul_clamp,
+    silu_and_mul_contig_post_quant,
+)
@@ -107,6 +111,9 @@
+from sglang.srt.layers.quantization.fp8_kernel import (
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +51/-6
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25001 - [LoRA] MLA attention LoRA: q_b_proj / kv_b_proj support

- Link: https://github.com/sgl-project/sglang/pull/25001
- Status/date: merged / 2026-05-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +1013/-0, 1081 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[LoRA] MLA attention LoRA: q_b_proj / kv_b_proj support"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/lora/triton_ops/kv_b_lora_absorbed.py`; technical summary: Covers "[LoRA] MLA attention LoRA: q_b_proj / kv_b_proj support"; the main implementation surface is `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/lora/triton_ops/kv_b_lora_absorbed.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +15/-0 (15 lines); hunks: -13,6 +13,15; -350,6 +359,8 @@ def forward_absorb_prepare(; symbols: forward_absorb_prepare, forward_absorb_core, touching `forward_absorb_prepare, forward_absorb_core`; `python/sglang/srt/models/deepseek_v2.py` modified +4/-0 (4 lines); hunks: -1687,11 +1687,15 @@ def prepare_qkv_latent(; symbols: prepare_qkv_latent, touching `prepare_qkv_latent`; `python/sglang/srt/lora/triton_ops/kv_b_lora_absorbed.py` added +849/-0 (849 lines); hunks: -0,0 +1,849; symbols: _num_segments, _max_segment_len, _segment_grid_size, _step_a_q_kernel, touching `_num_segments, _max_segment_len, _segment_grid_size`; `python/sglang/srt/lora/deepseek_mla_correction.py` added +117/-0 (117 lines); hunks: -0,0 +1,117; symbols: is_kv_b_lora_active, _get_state, apply_q_correction, apply_v_correction, touching `is_kv_b_lora_active, _get_state, apply_q_correction`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +15/-0 (15 lines); hunks: -13,6 +13,15; -350,6 +359,8 @@ def forward_absorb_prepare(; symbols: forward_absorb_prepare, forward_absorb_core
  - `python/sglang/srt/models/deepseek_v2.py` modified +4/-0 (4 lines); hunks: -1687,11 +1687,15 @@ def prepare_qkv_latent(; symbols: prepare_qkv_latent
  - `python/sglang/srt/lora/triton_ops/kv_b_lora_absorbed.py` added +849/-0 (849 lines); hunks: -0,0 +1,849; symbols: _num_segments, _max_segment_len, _segment_grid_size, _step_a_q_kernel
  - `python/sglang/srt/lora/deepseek_mla_correction.py` added +117/-0 (117 lines); hunks: -0,0 +1,117; symbols: is_kv_b_lora_active, _get_state, apply_q_correction, apply_v_correction
  - `python/sglang/srt/lora/utils.py` modified +14/-0 (14 lines); hunks: -134,6 +134,18 @@ def get_hidden_dim(; -274,6 +286,8 @@ def get_target_module_name(full_module_name: str, target_mod...; symbols: get_hidden_dim, get_target_module_name
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py
@@ -13,6 +13,15 @@
+from sglang.srt.lora.deepseek_mla_correction import (
+    apply_q_correction as apply_kv_b_lora_q_correction,
+)
+from sglang.srt.lora.deepseek_mla_correction import (
+    apply_v_correction as apply_kv_b_lora_v_correction,
+)
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -1687,11 +1687,15 @@ def prepare_qkv_latent(
+        # When the module is wrapped with LoRA, the fused GEMM fast-path would
+        # bypass the adapter because it reads weight.T directly.
+        lora_active = getattr(self.fused_qkv_a_proj_with_mqa, "set_lora", False)
+            and not lora_active
diff -- python/sglang/srt/lora/triton_ops/kv_b_lora_absorbed.py
@@ -0,0 +1,849 @@
+"""Triton kernels for absorbed-MLA ``kv_b_proj`` LoRA correction.
+The absorbed-MLA path bypasses ``kv_b_proj.forward()`` and folds the K/V
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +15/-0; `python/sglang/srt/models/deepseek_v2.py` modified +4/-0; `python/sglang/srt/lora/triton_ops/kv_b_lora_absorbed.py` added +849/-0; `python/sglang/srt/lora/deepseek_mla_correction.py` added +117/-0; `python/sglang/srt/lora/utils.py` modified +14/-0; `python/sglang/srt/lora/triton_ops/__init__.py` modified +10/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/lora/deepseek_mla_correction.py`, `python/sglang/srt/lora/triton_ops/__init__.py`, `python/sglang/srt/lora/triton_ops/kv_b_lora_absorbed.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25182 - chore: add vLLM SPDX copyright headers to ported files

- Link: https://github.com/sgl-project/sglang/pull/25182
- Status/date: merged / 2026-05-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 136 files, +255/-0, 872 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "chore: add vLLM SPDX copyright headers to ported files"; model line: DeepSeek V3.1; category: model support/runtime entry; main diff: `python/sglang/srt/models/baichuan.py`, `python/sglang/srt/models/commandr.py`, `python/sglang/srt/models/dbrx.py`; technical summary: Covers "chore: add vLLM SPDX copyright headers to ported files"; the main implementation surface is `python/sglang/srt/models/baichuan.py`, `python/sglang/srt/models/commandr.py`, `python/sglang/srt/models/dbrx.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/baichuan.py` modified +4/-0 (4 lines); hunks: -1,3 +1,7; `python/sglang/srt/models/commandr.py` modified +4/-0 (4 lines); hunks: -1,3 +1,7; `python/sglang/srt/models/dbrx.py` modified +3/-0 (3 lines); hunks: -1,3 +1,6; `python/sglang/srt/models/gemma.py` modified +3/-0 (3 lines); hunks: -1,3 +1,6.
- Code diff details:
  - `python/sglang/srt/models/baichuan.py` modified +4/-0 (4 lines); hunks: -1,3 +1,7
  - `python/sglang/srt/models/commandr.py` modified +4/-0 (4 lines); hunks: -1,3 +1,7
  - `python/sglang/srt/models/dbrx.py` modified +3/-0 (3 lines); hunks: -1,3 +1,6
  - `python/sglang/srt/models/gemma.py` modified +3/-0 (3 lines); hunks: -1,3 +1,6
  - `python/sglang/srt/models/gemma2.py` modified +3/-0 (3 lines); hunks: -1,3 +1,6
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/baichuan.py
@@ -1,3 +1,7 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/baichuan.py
diff -- python/sglang/srt/models/commandr.py
@@ -1,3 +1,7 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/commandr.py
diff -- python/sglang/srt/models/dbrx.py
@@ -1,3 +1,6 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
diff -- python/sglang/srt/models/gemma.py
@@ -1,3 +1,6 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/baichuan.py` modified +4/-0; `python/sglang/srt/models/commandr.py` modified +4/-0; `python/sglang/srt/models/dbrx.py` modified +3/-0; `python/sglang/srt/models/gemma.py` modified +3/-0; `python/sglang/srt/models/gemma2.py` modified +3/-0; `python/sglang/srt/models/gpt_bigcode.py` modified +3/-0
- Risk and verification: The diff ships test coverage in `python/sglang/test/test_custom_ops.py`, `python/sglang/test/test_marlin_utils.py`, `sgl-kernel/tests/test_causal_conv1d.py`, `test/registered/layers/mamba/test_causal_conv1d.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #24925 - [attn backend] Integrate tokenspeed_mla prefill/decode kernels (fp8 kv cache, blackwell)

- Link: https://github.com/sgl-project/sglang/pull/24925
- Status/date: merged / 2026-05-14
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 11 files, +462/-92, 726 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[attn backend] Integrate tokenspeed_mla prefill/decode kernels (fp8 kv cache, blackwell)"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/tokenspeed_mla_backend.py`, `python/sglang/srt/layers/attention/trtllm_mla_backend.py`, `python/sglang/srt/layers/attention/attention_registry.py`; technical summary: Covers "[attn backend] Integrate tokenspeed_mla prefill/decode kernels (fp8 kv cache, blackwell)"; the main implementation surface is `python/sglang/srt/layers/attention/tokenspeed_mla_backend.py`, `python/sglang/srt/layers/attention/trtllm_mla_backend.py`, `python/sglang/srt/layers/attention/attention_registry.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/tokenspeed_mla_backend.py` added +247/-0 (247 lines); hunks: -0,0 +1,247; symbols: _get_tokenspeed_workspace, TokenspeedMLABackend, __init__, _ensure_workspace, touching `_get_tokenspeed_workspace, TokenspeedMLABackend, __init__`; `python/sglang/srt/layers/attention/trtllm_mla_backend.py` modified +132/-91 (223 lines); hunks: -755,6 +755,109 @@ def unpad_draft_extend_output(; -838,46 +941,13 @@ def forward_decode(; symbols: unpad_draft_extend_output, _compute_decode_bmm1_scale, _run_decode_kernel, _run_prefill_kernel, touching `unpad_draft_extend_output, _compute_decode_bmm1_scale, _run_decode_kernel`; `python/sglang/srt/layers/attention/attention_registry.py` modified +11/-0 (11 lines); hunks: -62,6 +62,17 @@ def create_trtllm_mla_backend(runner):; symbols: create_trtllm_mla_backend, create_tokenspeed_mla_backend, create_aiter_backend, touching `create_trtllm_mla_backend, create_tokenspeed_mla_backend, create_aiter_backend`; `python/sglang/srt/models/deepseek_common/attention_backend_handler.py` modified +7/-0 (7 lines); hunks: -134,6 +134,12 @@ def handle_attention_trtllm_mla(attn, forward_batch):; -183,6 +189,7 @@ def handle_attention_intel_xpu(attn, forward_batch):; symbols: handle_attention_trtllm_mla, handle_attention_tokenspeed_mla, handle_attention_aiter, handle_attention_intel_xpu, touching `handle_attention_trtllm_mla, handle_attention_tokenspeed_mla, handle_attention_aiter`.
- Code diff details:
  - `python/sglang/srt/layers/attention/tokenspeed_mla_backend.py` added +247/-0 (247 lines); hunks: -0,0 +1,247; symbols: _get_tokenspeed_workspace, TokenspeedMLABackend, __init__, _ensure_workspace
  - `python/sglang/srt/layers/attention/trtllm_mla_backend.py` modified +132/-91 (223 lines); hunks: -755,6 +755,109 @@ def unpad_draft_extend_output(; -838,46 +941,13 @@ def forward_decode(; symbols: unpad_draft_extend_output, _compute_decode_bmm1_scale, _run_decode_kernel, _run_prefill_kernel
  - `python/sglang/srt/layers/attention/attention_registry.py` modified +11/-0 (11 lines); hunks: -62,6 +62,17 @@ def create_trtllm_mla_backend(runner):; symbols: create_trtllm_mla_backend, create_tokenspeed_mla_backend, create_aiter_backend
  - `python/sglang/srt/models/deepseek_common/attention_backend_handler.py` modified +7/-0 (7 lines); hunks: -134,6 +134,12 @@ def handle_attention_trtllm_mla(attn, forward_batch):; -183,6 +189,7 @@ def handle_attention_intel_xpu(attn, forward_batch):; symbols: handle_attention_trtllm_mla, handle_attention_tokenspeed_mla, handle_attention_aiter, handle_attention_intel_xpu
  - `python/sglang/srt/model_executor/model_runner.py` modified +2/-0 (2 lines); hunks: -244,6 +244,7; -256,6 +257,7
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/tokenspeed_mla_backend.py
@@ -0,0 +1,247 @@
+# Copyright (c) 2026 LightSeek Foundation
+#
+# Permission is hereby granted, free of charge, to any person obtaining a copy
+# of this software and associated documentation files (the "Software"), to deal
+# in the Software without restriction, including without limitation the rights
+# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
diff -- python/sglang/srt/layers/attention/trtllm_mla_backend.py
@@ -755,6 +755,109 @@ def unpad_draft_extend_output(
+    def _compute_decode_bmm1_scale(self, layer: RadixAttention) -> float:
+        """BMM1 scale ``q_scale * k_scale * softmax_scale``. k_scale only
+        applies when the KV cache stores FP8."""
+        q_scale = 1.0
+        if self.data_type == torch.float8_e4m3fn:
+            k_scale = (
diff -- python/sglang/srt/layers/attention/attention_registry.py
@@ -62,6 +62,17 @@ def create_trtllm_mla_backend(runner):
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/tokenspeed_mla_backend.py` added +247/-0; `python/sglang/srt/layers/attention/trtllm_mla_backend.py` modified +132/-91; `python/sglang/srt/layers/attention/attention_registry.py` modified +11/-0; `python/sglang/srt/models/deepseek_common/attention_backend_handler.py` modified +7/-0; `python/sglang/srt/model_executor/model_runner.py` modified +2/-0; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/pyproject.toml`, `python/sglang/srt/layers/attention/attention_registry.py`, `python/sglang/srt/layers/attention/tokenspeed_mla_backend.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19290 - feat: [2/2][DeepEP] Add waterfill load balancing for shared expert dispatch

- Link: https://github.com/sgl-project/sglang/pull/19290
- Status/date: merged / 2026-05-14
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +761/-27, 937 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "feat: [2/2][DeepEP] Add waterfill load balancing for shared expert dispatch"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/deepep_waterfill.py`, `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/model_executor/model_runner.py`; technical summary: Covers "feat: [2/2][DeepEP] Add waterfill load balancing for shared expert dispatch"; the main implementation surface is `python/sglang/srt/layers/moe/deepep_waterfill.py`, `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/model_executor/model_runner.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/moe/deepep_waterfill.py` added +584/-0 (584 lines); hunks: -0,0 +1,584; symbols: WaterfillDispatchPlan, _empty_expanded, _count_routed_per_rank_kernel, _waterfill_expand_kernel, touching `WaterfillDispatchPlan, _empty_expanded, _count_routed_per_rank_kernel`; `python/sglang/srt/layers/moe/topk.py` modified +48/-4 (52 lines); hunks: -284,6 +284,25 @@ def __init__(; -303,6 +322,18 @@ def __init__(; symbols: __init__, _apply_deepep_waterfill, forward_native, touching `__init__, _apply_deepep_waterfill, forward_native`; `python/sglang/srt/model_executor/model_runner.py` modified +45/-0 (45 lines); hunks: -122,6 +122,7; -650,6 +651,7 @@ def initialize(self, pre_model_load_memory: float):; symbols: initialize, load_model, _prepare_moe_topk, update_expert_location, touching `initialize, load_model, _prepare_moe_topk`; `python/sglang/srt/models/deepseek_v2.py` modified +3/-22 (25 lines); hunks: -953,16 +953,6 @@ def forward_deepep(; -2352,19 +2342,10 @@ def determine_num_fused_shared_experts(; symbols: forward_deepep, determine_num_fused_shared_experts, touching `forward_deepep, determine_num_fused_shared_experts`.
- Code diff details:
  - `python/sglang/srt/layers/moe/deepep_waterfill.py` added +584/-0 (584 lines); hunks: -0,0 +1,584; symbols: WaterfillDispatchPlan, _empty_expanded, _count_routed_per_rank_kernel, _waterfill_expand_kernel
  - `python/sglang/srt/layers/moe/topk.py` modified +48/-4 (52 lines); hunks: -284,6 +284,25 @@ def __init__(; -303,6 +322,18 @@ def __init__(; symbols: __init__, _apply_deepep_waterfill, forward_native
  - `python/sglang/srt/model_executor/model_runner.py` modified +45/-0 (45 lines); hunks: -122,6 +122,7; -650,6 +651,7 @@ def initialize(self, pre_model_load_memory: float):; symbols: initialize, load_model, _prepare_moe_topk, update_expert_location
  - `python/sglang/srt/models/deepseek_v2.py` modified +3/-22 (25 lines); hunks: -953,16 +953,6 @@ def forward_deepep(; -2352,19 +2342,10 @@ def determine_num_fused_shared_experts(; symbols: forward_deepep, determine_num_fused_shared_experts
  - `test/registered/unit/server_args/test_server_args.py` modified +41/-0 (41 lines); hunks: -515,6 +515,47 @@ def test_external_corpus_max_tokens_must_be_positive(self):; symbols: test_external_corpus_max_tokens_must_be_positive, TestDeepEPWaterfillArgs, test_waterfill_enforces_shared_experts_fusion, test_waterfill_overrides_moe_a2a_backend_to_deepep
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/deepep_waterfill.py
@@ -0,0 +1,584 @@
+# Copyright 2023-2026 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/layers/moe/topk.py
@@ -284,6 +284,25 @@ def __init__(
+        if num_fused_shared_experts > 0:
+            from sglang.srt.server_args import get_global_server_args
+            try:
+                self.enable_deepep_waterfill = (
+                    get_global_server_args().enable_deepep_waterfill
+                )
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -122,6 +122,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/deepep_waterfill.py` added +584/-0; `python/sglang/srt/layers/moe/topk.py` modified +48/-4; `python/sglang/srt/model_executor/model_runner.py` modified +45/-0; `python/sglang/srt/models/deepseek_v2.py` modified +3/-22; `python/sglang/srt/server_args.py` modified +36/-1; `python/sglang/srt/environ.py` modified +3/-0
  - tests: `test/registered/unit/server_args/test_server_args.py` modified +41/-0
  - docs: `docs/advanced_features/server_arguments.md` modified +1/-0
- Risk and verification: The diff ships test coverage in `test/registered/unit/server_args/test_server_args.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #25279 - DeepseekV2MoE: defer shared experts when routed kernel is non-mutating

- Link: https://github.com/sgl-project/sglang/pull/25279
- Status/date: merged / 2026-05-15
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +11/-3, 29 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "DeepseekV2MoE: defer shared experts when routed kernel is non-mutating"; model line: DeepSeek V3.1; category: model implementation change; main diff: `python/sglang/srt/models/deepseek_v2.py`; technical summary: Covers "DeepseekV2MoE: defer shared experts when routed kernel is non-mutating"; the main implementation surface is `python/sglang/srt/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +11/-3 (14 lines); hunks: -826,10 +826,9 @@ def forward_normal(; -894,6 +893,15 @@ def _post_combine_hook(; symbols: forward_normal, _post_combine_hook, touching `forward_normal, _post_combine_hook`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +11/-3 (14 lines); hunks: -826,10 +826,9 @@ def forward_normal(; -894,6 +893,15 @@ def _post_combine_hook(; symbols: forward_normal, _post_combine_hook
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -826,10 +826,9 @@ def forward_normal(
+        defer_shared = not self.experts.moe_runner_config.inplace
-            if (
-                not self._fuse_shared_experts_inside_sbo
-            ):  # TODO: check if it supports mtp
+            if not defer_shared and not self._fuse_shared_experts_inside_sbo:
@@ -894,6 +893,15 @@ def _post_combine_hook(
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +11/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25333 - perf(mla): hybrid Triton fused cat+FP8-quantize for MLA chunked-prefill K/V

- Link: https://github.com/sgl-project/sglang/pull/25333
- Status/date: merged / 2026-05-15
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +641/-12, 684 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "perf(mla): hybrid Triton fused cat+FP8-quantize for MLA chunked-prefill K/V"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`, `python/sglang/jit_kernel/mla_kv_pack_quantize_fp8.py`, `python/sglang/jit_kernel/benchmark/bench_mla_kv_pack_quantize_fp8.py`; technical summary: Covers "perf(mla): hybrid Triton fused cat+FP8-quantize for MLA chunked-prefill K/V"; the main implementation surface is `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`, `python/sglang/jit_kernel/mla_kv_pack_quantize_fp8.py`, `python/sglang/jit_kernel/benchmark/bench_mla_kv_pack_quantize_fp8.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` modified +28/-12 (40 lines); hunks: -36,6 +36,14; -369,14 +377,19 @@ def _chunked_prefix_attn_mha(; symbols: _resolve_attn_backend, _chunked_prefix_attn_mha, touching `_resolve_attn_backend, _chunked_prefix_attn_mha`; `python/sglang/jit_kernel/mla_kv_pack_quantize_fp8.py` added +295/-0 (295 lines); hunks: -0,0 +1,295; symbols: _v0_kernel, _v1_flat_kernel, _pick_kernel, mla_kv_pack_quantize_fp8, touching `_v0_kernel, _v1_flat_kernel, _pick_kernel`; `python/sglang/jit_kernel/benchmark/bench_mla_kv_pack_quantize_fp8.py` added +214/-0 (214 lines); hunks: -0,0 +1,214; symbols: _triton_mla_kv_pack_quantize_fp8_kernel, _triton_pack, benchmark, fn, touching `_triton_mla_kv_pack_quantize_fp8_kernel, _triton_pack, benchmark`; `python/sglang/jit_kernel/tests/test_mla_kv_pack_quantize_fp8.py` added +104/-0 (104 lines); hunks: -0,0 +1,104; symbols: _ref, test_correctness, test_strided_inputs, test_kpe_2d_accepted, touching `_ref, test_correctness, test_strided_inputs`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` modified +28/-12 (40 lines); hunks: -36,6 +36,14; -369,14 +377,19 @@ def _chunked_prefix_attn_mha(; symbols: _resolve_attn_backend, _chunked_prefix_attn_mha
  - `python/sglang/jit_kernel/mla_kv_pack_quantize_fp8.py` added +295/-0 (295 lines); hunks: -0,0 +1,295; symbols: _v0_kernel, _v1_flat_kernel, _pick_kernel, mla_kv_pack_quantize_fp8
  - `python/sglang/jit_kernel/benchmark/bench_mla_kv_pack_quantize_fp8.py` added +214/-0 (214 lines); hunks: -0,0 +1,214; symbols: _triton_mla_kv_pack_quantize_fp8_kernel, _triton_pack, benchmark, fn
  - `python/sglang/jit_kernel/tests/test_mla_kv_pack_quantize_fp8.py` added +104/-0 (104 lines); hunks: -0,0 +1,104; symbols: _ref, test_correctness, test_strided_inputs, test_kpe_2d_accepted
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py
@@ -36,6 +36,14 @@
+def _resolve_attn_backend(forward_batch: ForwardBatch):
+    backend = forward_batch.attn_backend
+    if isinstance(backend, TboAttnBackend):
+        backend = backend.primary
+    return backend
@@ -369,14 +377,19 @@ def _chunked_prefix_attn_mha(
diff -- python/sglang/jit_kernel/mla_kv_pack_quantize_fp8.py
@@ -0,0 +1,295 @@
+"""Fused ``cat(k_nope, broadcast(k_pe)) + FP8 quantize`` for K and ``FP8 quantize`` for V.
+Dispatches between two Triton kernels per batch size; see ``_pick_kernel``.
+"""
+from __future__ import annotations
+from typing import Optional, Tuple
+import torch
diff -- python/sglang/jit_kernel/benchmark/bench_mla_kv_pack_quantize_fp8.py
@@ -0,0 +1,214 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` modified +28/-12; `python/sglang/jit_kernel/mla_kv_pack_quantize_fp8.py` added +295/-0; `python/sglang/jit_kernel/benchmark/bench_mla_kv_pack_quantize_fp8.py` added +214/-0
  - tests: `python/sglang/jit_kernel/tests/test_mla_kv_pack_quantize_fp8.py` added +104/-0
- Risk and verification: The diff ships test coverage in `python/sglang/jit_kernel/tests/test_mla_kv_pack_quantize_fp8.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #25379 - feat(moe): reuse prev-layer output as symm_output for FP4 routed MoE

- Link: https://github.com/sgl-project/sglang/pull/25379
- Status/date: merged / 2026-05-15
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +85/-18, 190 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "feat(moe): reuse prev-layer output as symm_output for FP4 routed MoE"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py`, `python/sglang/srt/layers/moe/moe_runner/base.py`; technical summary: Covers "feat(moe): reuse prev-layer output as symm_output for FP4 routed MoE"; the main implementation surface is `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py`, `python/sglang/srt/layers/moe/moe_runner/base.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +31/-10 (41 lines); hunks: -826,10 +826,9 @@ def forward_normal(; -894,6 +893,15 @@ def _post_combine_hook(; symbols: forward_normal, _post_combine_hook, forward, touching `forward_normal, _post_combine_hook, forward`; `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py` modified +25/-7 (32 lines); hunks: -10,6 +10,8; -21,6 +23,7; symbols: fused_experts_none_to_flashinfer_trtllm_fp4, touching `fused_experts_none_to_flashinfer_trtllm_fp4`; `python/sglang/srt/layers/moe/moe_runner/base.py` modified +17/-1 (18 lines); hunks: -1,8 +1,10; -26,6 +28,20; symbols: moe_output_buffer_ctx, MoeRunnerConfig, touching `moe_output_buffer_ctx, MoeRunnerConfig`; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +9/-0 (9 lines); hunks: -308,6 +308,15 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +31/-10 (41 lines); hunks: -826,10 +826,9 @@ def forward_normal(; -894,6 +893,15 @@ def _post_combine_hook(; symbols: forward_normal, _post_combine_hook, forward
  - `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py` modified +25/-7 (32 lines); hunks: -10,6 +10,8; -21,6 +23,7; symbols: fused_experts_none_to_flashinfer_trtllm_fp4
  - `python/sglang/srt/layers/moe/moe_runner/base.py` modified +17/-1 (18 lines); hunks: -1,8 +1,10; -26,6 +28,20; symbols: moe_output_buffer_ctx, MoeRunnerConfig
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +9/-0 (9 lines); hunks: -308,6 +308,15 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/layers/communicator.py` modified +3/-0 (3 lines); hunks: -305,6 +305,9 @@ def fetch_hidden_states(self):; symbols: fetch_hidden_states, clear_attn_inputs, maybe_input_scattered
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -826,10 +826,9 @@ def forward_normal(
+        defer_shared = not self.experts.moe_runner_config.inplace
-            if (
-                not self._fuse_shared_experts_inside_sbo
-            ):  # TODO: check if it supports mtp
+            if not defer_shared and not self._fuse_shared_experts_inside_sbo:
@@ -894,6 +893,15 @@ def _post_combine_hook(
diff -- python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py
@@ -10,6 +10,8 @@
+    is_symmetric_memory_enabled,
+    is_tensor_in_symmetric_mempool,
@@ -21,6 +23,7 @@
+    _moe_output_buf,
@@ -877,14 +880,29 @@ def fused_experts_none_to_flashinfer_trtllm_fp4(
-    with use_symmetric_memory(get_tp_group(), disabled=not is_allocation_symmetric()):
diff -- python/sglang/srt/layers/moe/moe_runner/base.py
@@ -1,8 +1,10 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +31/-10; `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py` modified +25/-7; `python/sglang/srt/layers/moe/moe_runner/base.py` modified +17/-1; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +9/-0; `python/sglang/srt/layers/communicator.py` modified +3/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/layers/moe/moe_runner/base.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25406 - [MoE] Decouple Mega MoE from DeepEP backend

- Link: https://github.com/sgl-project/sglang/pull/25406
- Status/date: merged / 2026-05-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 18 files, +41/-29, 258 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MoE] Decouple Mega MoE from DeepEP backend"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/layers/moe/mega_moe.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`; technical summary: Covers "[MoE] Decouple Mega MoE from DeepEP backend"; the main implementation surface is `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/layers/moe/mega_moe.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/moe/utils.py` modified +4/-0 (4 lines); hunks: -29,6 +29,7 @@ class MoeA2ABackend(Enum):; -61,6 +62,9 @@ def is_ascend_fuseep(self):; symbols: MoeA2ABackend, is_ascend_fuseep, is_mori, is_megamoe, touching `MoeA2ABackend, is_ascend_fuseep, is_mori`; `python/sglang/srt/layers/moe/mega_moe.py` modified +2/-1 (3 lines); hunks: -25,6 +25,7; -94,7 +95,7 @@ def _get_mega_moe_symm_buffer(; symbols: _get_mega_moe_symm_buffer, should_use_mega_moe, touching `_get_mega_moe_symm_buffer, should_use_mega_moe`; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +1/-1 (2 lines); hunks: -82,7 +82,7; symbols: create_moe_dispatcher, touching `create_moe_dispatcher`; `python/sglang/srt/layers/quantization/fp8.py` modified +1/-1 (2 lines); hunks: -1193,7 +1193,7 @@ def process_weights_after_loading_block_quant(self, layer:...; symbols: process_weights_after_loading_block_quant, touching `process_weights_after_loading_block_quant`.
- Code diff details:
  - `python/sglang/srt/layers/moe/utils.py` modified +4/-0 (4 lines); hunks: -29,6 +29,7 @@ class MoeA2ABackend(Enum):; -61,6 +62,9 @@ def is_ascend_fuseep(self):; symbols: MoeA2ABackend, is_ascend_fuseep, is_mori, is_megamoe
  - `python/sglang/srt/layers/moe/mega_moe.py` modified +2/-1 (3 lines); hunks: -25,6 +25,7; -94,7 +95,7 @@ def _get_mega_moe_symm_buffer(; symbols: _get_mega_moe_symm_buffer, should_use_mega_moe
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +1/-1 (2 lines); hunks: -82,7 +82,7; symbols: create_moe_dispatcher
  - `python/sglang/srt/layers/quantization/fp8.py` modified +1/-1 (2 lines); hunks: -1193,7 +1193,7 @@ def process_weights_after_loading_block_quant(self, layer:...; symbols: process_weights_after_loading_block_quant
  - `python/sglang/srt/layers/moe/moe_runner/deep_gemm.py` modified +0/-1 (1 lines); hunks: -131,7 +131,6 @@ def __init__(self, config: MoeRunnerConfig):; symbols: __init__, run
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/utils.py
@@ -29,6 +29,7 @@ class MoeA2ABackend(Enum):
+    MEGAMOE = "megamoe"
@@ -61,6 +62,9 @@ def is_ascend_fuseep(self):
+    def is_megamoe(self):
+        return self == MoeA2ABackend.MEGAMOE
diff -- python/sglang/srt/layers/moe/mega_moe.py
@@ -25,6 +25,7 @@
+from sglang.srt.layers.moe.utils import get_moe_a2a_backend
@@ -94,7 +95,7 @@ def _get_mega_moe_symm_buffer(
-    if not envs.SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE.get():
+    if not get_moe_a2a_backend().is_megamoe():
diff -- python/sglang/srt/layers/moe/fused_moe_triton/layer.py
@@ -82,7 +82,7 @@
-    if a2a_backend.is_none():
+    if a2a_backend.is_none() or a2a_backend.is_megamoe():
diff -- python/sglang/srt/layers/quantization/fp8.py
@@ -1193,7 +1193,7 @@ def process_weights_after_loading_block_quant(self, layer: Module) -> None:
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/utils.py` modified +4/-0; `python/sglang/srt/layers/moe/mega_moe.py` modified +2/-1; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +1/-1; `python/sglang/srt/layers/quantization/fp8.py` modified +1/-1; `python/sglang/srt/layers/moe/moe_runner/deep_gemm.py` modified +0/-1; `python/sglang/srt/models/deepseek_v2.py` modified +1/-0
  - tests: `test/registered/dsv4/test_deepseek_v4_flash_fp4_megamoe_b200.py` modified +2/-10
- Risk and verification: The diff ships test coverage in `test/manual/dsv4/test_b200_flash.py`, `test/manual/dsv4/test_b200_pro.py`, `test/manual/dsv4/test_b300_flash.py`, `test/manual/dsv4/test_b300_pro.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #25390 - [AMD] Enable shared-experts fusion with new KIMI-K2.5-MXFP4 model.

- Link: https://github.com/sgl-project/sglang/pull/25390
- Status/date: merged / 2026-05-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +18/-2, 41 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Enable shared-experts fusion with new KIMI-K2.5-MXFP4 model."; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/quantization/quark/quark.py`; technical summary: Covers "[AMD] Enable shared-experts fusion with new KIMI-K2.5-MXFP4 model."; the main implementation surface is `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/quantization/quark/quark.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +11/-1 (12 lines); hunks: -2355,6 +2355,12 @@ def __init__(; -2422,7 +2428,11 @@ def determine_num_fused_shared_experts(; symbols: __init__, determine_num_fused_shared_experts, touching `__init__, determine_num_fused_shared_experts`; `python/sglang/srt/layers/quantization/quark/quark.py` modified +7/-1 (8 lines); hunks: -71,7 +71,13 @@ def get_name(self) -> str:; symbols: get_name, apply_weight_name_mapper, get_quant_method, touching `get_name, apply_weight_name_mapper, get_quant_method`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +11/-1 (12 lines); hunks: -2355,6 +2355,12 @@ def __init__(; -2422,7 +2428,11 @@ def determine_num_fused_shared_experts(; symbols: __init__, determine_num_fused_shared_experts
  - `python/sglang/srt/layers/quantization/quark/quark.py` modified +7/-1 (8 lines); hunks: -71,7 +71,13 @@ def get_name(self) -> str:; symbols: get_name, apply_weight_name_mapper, get_quant_method
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -2355,6 +2355,12 @@ def __init__(
+        # Quant configs like Quark may rely on the model to provide fused-module
+        # mappings so exclusion checks can unfuse derived names back to the
+        # checkpoint's source layer names.
+        if quant_config is not None and hasattr(quant_config, "packed_modules_mapping"):
+            quant_config.packed_modules_mapping = self.packed_modules_mapping
@@ -2422,7 +2428,11 @@ def determine_num_fused_shared_experts(
diff -- python/sglang/srt/layers/quantization/quark/quark.py
@@ -71,7 +71,13 @@ def get_name(self) -> str:
-        self.exclude_layers = hf_to_sglang_mapper.apply_list(self.exclude_layers)
+        mapped = hf_to_sglang_mapper.apply_list(self.exclude_layers)
+        expanded = []
+        for name in mapped:
+            expanded.append(name)
+            if name.startswith("language_model."):
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +11/-1; `python/sglang/srt/layers/quantization/quark/quark.py` modified +7/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/quantization/quark/quark.py`, `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #24933 - Amd/deepseek v4 rebase main 0509

- Link: https://github.com/sgl-project/sglang/pull/24933
- Status/date: merged / 2026-05-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 17 files, +3678/-70, 4186 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Amd/deepseek v4 rebase main 0509"; model line: DeepSeek V3.1; category: model implementation change; main diff: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/dsv4/compress_hip.py`; technical summary: Covers "Amd/deepseek v4 rebase main 0509"; the main implementation surface is `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`, `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/dsv4/compress_hip.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` added +1265/-0 (1265 lines); hunks: -0,0 +1,1265; symbols: _pad_last_dim, _create_flashmla_metadata, _create_dummy_paged_compress_data, DSV4AttnMetadata, touching `_pad_last_dim, _create_flashmla_metadata, _create_dummy_paged_compress_data`; `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` modified +1214/-2 (1216 lines); hunks: -1,5 +1,6; -10,6 +11,28; symbols: _legalize_result_idx_safe, fast_log2_ceil, tilelang_sparse_fwd, fp8_paged_mqa_logits_kernel, touching `_legalize_result_idx_safe, fast_log2_ceil, tilelang_sparse_fwd`; `python/sglang/srt/layers/attention/dsv4/compress_hip.py` added +455/-0 (455 lines); hunks: -0,0 +1,455; symbols: _rms_normalize_kernel, rms_normalize_triton, DeepseekRefRMSNorm, __init__, touching `_rms_normalize_kernel, rms_normalize_triton, DeepseekRefRMSNorm`; `python/sglang/srt/layers/attention/hip_flash_mla.py` added +197/-0 (197 lines); hunks: -0,0 +1,197; symbols: flash_mla_with_kvcache_entrypoint, flash_mla_with_kvcache_torch, _assert_close, touching `flash_mla_with_kvcache_entrypoint, flash_mla_with_kvcache_torch, _assert_close`.
- Code diff details:
  - `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` added +1265/-0 (1265 lines); hunks: -0,0 +1,1265; symbols: _pad_last_dim, _create_flashmla_metadata, _create_dummy_paged_compress_data, DSV4AttnMetadata
  - `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` modified +1214/-2 (1216 lines); hunks: -1,5 +1,6; -10,6 +11,28; symbols: _legalize_result_idx_safe, fast_log2_ceil, tilelang_sparse_fwd, fp8_paged_mqa_logits_kernel
  - `python/sglang/srt/layers/attention/dsv4/compress_hip.py` added +455/-0 (455 lines); hunks: -0,0 +1,455; symbols: _rms_normalize_kernel, rms_normalize_triton, DeepseekRefRMSNorm, __init__
  - `python/sglang/srt/layers/attention/hip_flash_mla.py` added +197/-0 (197 lines); hunks: -0,0 +1,197; symbols: flash_mla_with_kvcache_entrypoint, flash_mla_with_kvcache_torch, _assert_close
  - `python/sglang/srt/layers/deepseek_v4_rope.py` modified +168/-0 (168 lines); hunks: -177,3 +177,171 @@ def apply_rotary_emb_triton(; symbols: apply_rotary_emb_triton, _fused_norm_rope_kernel, fused_norm_rope_inplace_triton
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py
@@ -0,0 +1,1265 @@
+from __future__ import annotations
+import enum
+import functools
+import logging
+from dataclasses import dataclass, field
+from typing import (
diff -- python/sglang/srt/layers/attention/nsa/tilelang_kernel.py
@@ -1,5 +1,6 @@
+import functools
-from typing import Optional, Tuple
+from typing import Any, Optional, Tuple
@@ -10,6 +11,28 @@
+# Workaround a tilelang bug: BaseKernelAdapter._legalize_result_idx mutates the
+# `out_idx` list in place when normalising negative indices to positive ones.
diff -- python/sglang/srt/layers/attention/dsv4/compress_hip.py
@@ -0,0 +1,455 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` added +1265/-0; `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` modified +1214/-2; `python/sglang/srt/layers/attention/dsv4/compress_hip.py` added +455/-0; `python/sglang/srt/layers/attention/hip_flash_mla.py` added +197/-0; `python/sglang/srt/layers/deepseek_v4_rope.py` modified +168/-0; `python/sglang/srt/layers/quantization/fp8.py` modified +143/-16
- Risk and verification: Runtime changes concentrate in `python/sglang/jit_kernel/deepseek_v4.py`, `python/sglang/srt/environ.py`, `python/sglang/srt/layers/attention/attention_registry.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25454 - fix(eagle3): drop +1 offset on aux layer ids when first id != 1

- Link: https://github.com/sgl-project/sglang/pull/25454
- Status/date: merged / 2026-05-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +6/-3, 16 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix(eagle3): drop +1 offset on aux layer ids when first id != 1"; model line: DeepSeek V3.1; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v2.py`; technical summary: Covers "fix(eagle3): drop +1 offset on aux layer ids when first id != 1"; the main implementation surface is `python/sglang/srt/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +6/-3 (9 lines); hunks: -2549,9 +2549,12 @@ def set_eagle3_layers_to_capture(self, layer_ids: Optiona...; symbols: set_eagle3_layers_to_capture, set_dflash_layers_to_capture, touching `set_eagle3_layers_to_capture, set_dflash_layers_to_capture`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +6/-3 (9 lines); hunks: -2549,9 +2549,12 @@ def set_eagle3_layers_to_capture(self, layer_ids: Optiona...; symbols: set_eagle3_layers_to_capture, set_dflash_layers_to_capture
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -2549,9 +2549,12 @@ def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
-            # we plus 1 here because in sglang, for the ith layer, it takes the output
-            # of the (i-1)th layer as aux hidden state
-            self.model.layers_to_capture = [val + 1 for val in layer_ids]
+            # TODO (Qiaolin-Yu): check if other draft models need similar layer id
+            # adjustment
+            if layer_ids and layer_ids[0] == 1:
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +6/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #24640 - Support spec v2 for FlashMLA speculative decoding

- Link: https://github.com/sgl-project/sglang/pull/24640
- Status/date: merged / 2026-05-19
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +11/-12, 51 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support spec v2 for FlashMLA speculative decoding"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/flashmla_backend.py`, `python/sglang/srt/models/deepseek_v2.py`, `test/registered/mla/test_flashmla.py`; technical summary: Covers "Support spec v2 for FlashMLA speculative decoding"; the main implementation surface is `python/sglang/srt/layers/attention/flashmla_backend.py`, `python/sglang/srt/models/deepseek_v2.py`, `test/registered/mla/test_flashmla.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/flashmla_backend.py` modified +4/-3 (7 lines); hunks: -477,9 +477,10 @@ def forward_extend(; symbols: forward_extend, touching `forward_extend`; `python/sglang/srt/models/deepseek_v2.py` modified +1/-1 (2 lines); hunks: -1553,7 +1553,7 @@ def dispatch_attn_forward_method(; symbols: dispatch_attn_forward_method, touching `dispatch_attn_forward_method`; `test/registered/mla/test_flashmla.py` modified +6/-8 (14 lines); hunks: -9,7 +9,6; -54,13 +53,12 @@ def setUpClass(cls):; symbols: setUpClass, tearDownClass, touching `setUpClass, tearDownClass`.
- Code diff details:
  - `python/sglang/srt/layers/attention/flashmla_backend.py` modified +4/-3 (7 lines); hunks: -477,9 +477,10 @@ def forward_extend(; symbols: forward_extend
  - `python/sglang/srt/models/deepseek_v2.py` modified +1/-1 (2 lines); hunks: -1553,7 +1553,7 @@ def dispatch_attn_forward_method(; symbols: dispatch_attn_forward_method
  - `test/registered/mla/test_flashmla.py` modified +6/-8 (14 lines); hunks: -9,7 +9,6; -54,13 +53,12 @@ def setUpClass(cls):; symbols: setUpClass, tearDownClass
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/flashmla_backend.py
@@ -477,9 +477,10 @@ def forward_extend(
-        if (
-            forward_batch.forward_mode == ForwardMode.EXTEND
-            or forward_batch.forward_mode == ForwardMode.DRAFT_EXTEND
+        if forward_batch.forward_mode in (
+            ForwardMode.EXTEND,
+            ForwardMode.DRAFT_EXTEND,
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -1553,7 +1553,7 @@ def dispatch_attn_forward_method(
-            or forward_batch.forward_mode.is_draft_extend()
+            or forward_batch.forward_mode.is_draft_extend(include_v2=True)
diff -- test/registered/mla/test_flashmla.py
@@ -9,7 +9,6 @@
-from sglang.srt.environ import envs
@@ -54,13 +53,12 @@ def setUpClass(cls):
-        with envs.SGLANG_ENABLE_SPEC_V2.override(False):
-            cls.process = popen_launch_server(
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/flashmla_backend.py` modified +4/-3; `python/sglang/srt/models/deepseek_v2.py` modified +1/-1
  - tests: `test/registered/mla/test_flashmla.py` modified +6/-8
- Risk and verification: The diff ships test coverage in `test/registered/mla/test_flashmla.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #25821 - [Refactor] Rename NSA → DSA: user-facing aliases, file/class/import rename

- Link: https://github.com/sgl-project/sglang/pull/25821
- Status/date: merged / 2026-05-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 162 files, +11303/-10745, 15980 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Refactor] Rename NSA → DSA: user-facing aliases, file/class/import rename"; model line: DeepSeek V3.1; category: docs/tests/CI; main diff: `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/dsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; technical summary: Covers "[Refactor] Rename NSA → DSA: user-facing aliases, file/class/import rename"; the main implementation surface is `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/dsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/nsa_backend.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` modified +8/-2587 (2595 lines); `python/sglang/srt/layers/attention/dsa/tilelang_kernel.py` added +2589/-0 (2589 lines); `python/sglang/srt/layers/attention/nsa_backend.py` modified +21/-2518 (2539 lines); `python/sglang/srt/layers/attention/dsa_backend.py` added +2528/-0 (2528 lines).
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` modified +8/-2587 (2595 lines)
  - `python/sglang/srt/layers/attention/dsa/tilelang_kernel.py` added +2589/-0 (2589 lines)
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +21/-2518 (2539 lines)
  - `python/sglang/srt/layers/attention/dsa_backend.py` added +2528/-0 (2528 lines)
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +8/-1744 (1752 lines); hunks: -1,1746 +1,10; symbols: BaseIndexerMetadata, get_seqlens_int32, get_page_table_64, get_page_table_1
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -1,1746 +1,10 @@
-from __future__ import annotations
+# [Deprecated] Re-export shim for backward compatibility. Use dsa.dsa_indexer instead.
+import warnings
-import contextlib
-import logging
-from abc import ABC, abstractmethod
diff -- python/sglang/srt/layers/attention/dsa/dsa_indexer.py
@@ -0,0 +1,1746 @@
+from __future__ import annotations
+import contextlib
+import logging
+from abc import ABC, abstractmethod
+from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
+import torch
diff -- python/sglang/srt/layers/attention/nsa/index_buf_accessor.py
@@ -1,814 +1,10 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` modified +8/-2587; `python/sglang/srt/layers/attention/dsa/tilelang_kernel.py` added +2589/-0; `python/sglang/srt/layers/attention/nsa_backend.py` modified +21/-2518; `python/sglang/srt/layers/attention/dsa_backend.py` added +2528/-0; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +8/-1744; `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` added +1746/-0
- Risk and verification: The diff ships test coverage in `python/sglang/jit_kernel/tests/test_fused_metadata_copy.py`, `python/sglang/jit_kernel/tests/test_fused_store_index_cache.py`, `python/sglang/jit_kernel/tests/test_set_mla_kv_buffer.py`, `python/sglang/test/nightly_utils.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #25460 - [perf] prepare_prefill_qkv hook + fp8 quantize jit kernel

- Link: https://github.com/sgl-project/sglang/pull/25460
- Status/date: merged / 2026-05-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +305/-16, 383 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[perf] prepare_prefill_qkv hook + fp8 quantize jit kernel"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/tokenspeed_mla_backend.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`, `python/sglang/srt/layers/attention/trtllm_mla_backend.py`; technical summary: Covers "[perf] prepare_prefill_qkv hook + fp8 quantize jit kernel"; the main implementation surface is `python/sglang/srt/layers/attention/tokenspeed_mla_backend.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`, `python/sglang/srt/layers/attention/trtllm_mla_backend.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/tokenspeed_mla_backend.py` modified +128/-14 (142 lines); hunks: -33,20 +33,26; -77,6 +83,9 @@ def _get_tokenspeed_workspace(; symbols: _get_tokenspeed_workspace, TokenspeedMLABackend, __init__, _fused_rope_fp8_quantize, touching `_get_tokenspeed_workspace, TokenspeedMLABackend, __init__`; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` modified +18/-0 (18 lines); hunks: -220,6 +220,24 @@ def forward_normal_prepare(; symbols: forward_normal_prepare, touching `forward_normal_prepare`; `python/sglang/srt/layers/attention/trtllm_mla_backend.py` modified +2/-2 (4 lines); hunks: -1144,7 +1144,7 @@ def forward_extend(; -1187,7 +1187,7 @@ def forward_extend(; symbols: forward_extend, touching `forward_extend`; `python/sglang/jit_kernel/fp8_quantize.py` added +157/-0 (157 lines); hunks: -0,0 +1,157; symbols: _fp8_quantize_kernel, _flatten_to_2d, fp8_quantize, touching `_fp8_quantize_kernel, _flatten_to_2d, fp8_quantize`.
- Code diff details:
  - `python/sglang/srt/layers/attention/tokenspeed_mla_backend.py` modified +128/-14 (142 lines); hunks: -33,20 +33,26; -77,6 +83,9 @@ def _get_tokenspeed_workspace(; symbols: _get_tokenspeed_workspace, TokenspeedMLABackend, __init__, _fused_rope_fp8_quantize
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` modified +18/-0 (18 lines); hunks: -220,6 +220,24 @@ def forward_normal_prepare(; symbols: forward_normal_prepare
  - `python/sglang/srt/layers/attention/trtllm_mla_backend.py` modified +2/-2 (4 lines); hunks: -1144,7 +1144,7 @@ def forward_extend(; -1187,7 +1187,7 @@ def forward_extend(; symbols: forward_extend
  - `python/sglang/jit_kernel/fp8_quantize.py` added +157/-0 (157 lines); hunks: -0,0 +1,157; symbols: _fp8_quantize_kernel, _flatten_to_2d, fp8_quantize
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/tokenspeed_mla_backend.py
@@ -33,20 +33,26 @@
+from sglang.jit_kernel.fp8_quantize import fp8_quantize
+from sglang.jit_kernel.mla_kv_pack_quantize_fp8 import mla_kv_pack_quantize_fp8
-    _quantize_fp8_qkv,
-from sglang.srt.utils import is_tokenspeed_mla_available
+from sglang.srt.utils import is_flashinfer_available, is_tokenspeed_mla_available
+if is_flashinfer_available():
diff -- python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py
@@ -220,6 +220,24 @@ def forward_normal_prepare(
+        # Backend prefill hook: the backend owns the BF16->FP8 transition
+        # (fused RoPE + quantize for Q/K, direct FP8 KV-cache write) and
+        # returns FP8 tensors ready for its kernel. Backends without the
+        # hook fall through to the BF16 path below.
+        backend = _resolve_attn_backend(forward_batch)
+        if hasattr(backend, "prepare_prefill_qkv"):
diff -- python/sglang/srt/layers/attention/trtllm_mla_backend.py
@@ -1144,7 +1144,7 @@ def forward_extend(
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/tokenspeed_mla_backend.py` modified +128/-14; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` modified +18/-0; `python/sglang/srt/layers/attention/trtllm_mla_backend.py` modified +2/-2; `python/sglang/jit_kernel/fp8_quantize.py` added +157/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/jit_kernel/fp8_quantize.py`, `python/sglang/srt/layers/attention/tokenspeed_mla_backend.py`, `python/sglang/srt/layers/attention/trtllm_mla_backend.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25884 - [Refactor] major JIT kernel clean up for dsv4

- Link: https://github.com/sgl-project/sglang/pull/25884
- Status/date: merged / 2026-05-21
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 23 files, +1093/-1399, 2663 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Refactor] major JIT kernel clean up for dsv4"; model line: DeepSeek V3.1; category: model implementation change; main diff: `python/sglang/srt/layers/attention/dsv4/compressor.py`, `python/sglang/srt/layers/attention/dsv4/metadata.py`, `python/sglang/srt/layers/moe/moe_runner/deep_gemm.py`; technical summary: Covers "[Refactor] major JIT kernel clean up for dsv4"; the main implementation surface is `python/sglang/srt/layers/attention/dsv4/compressor.py`, `python/sglang/srt/layers/attention/dsv4/metadata.py`, `python/sglang/srt/layers/moe/moe_runner/deep_gemm.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/dsv4/compressor.py` modified +2/-3 (5 lines); hunks: -5,13 +5,12; `python/sglang/srt/layers/attention/dsv4/metadata.py` modified +2/-2 (4 lines); hunks: -109,7 +109,7 @@ def __post_init__(self):; -124,7 +124,7 @@ def __post_init__(self):; symbols: __post_init__, touching `__post_init__`; `python/sglang/srt/layers/moe/moe_runner/deep_gemm.py` modified +2/-2 (4 lines); hunks: -6,7 +6,7; -167,7 +167,7 @@ def _run_contiguous_gemm(; symbols: _run_contiguous_gemm, touching `_run_contiguous_gemm`; `python/sglang/srt/models/deepseek_v2.py` modified +2/-2 (4 lines); hunks: -29,7 +29,7; -420,7 +420,7 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/layers/attention/dsv4/compressor.py` modified +2/-3 (5 lines); hunks: -5,13 +5,12
  - `python/sglang/srt/layers/attention/dsv4/metadata.py` modified +2/-2 (4 lines); hunks: -109,7 +109,7 @@ def __post_init__(self):; -124,7 +124,7 @@ def __post_init__(self):; symbols: __post_init__
  - `python/sglang/srt/layers/moe/moe_runner/deep_gemm.py` modified +2/-2 (4 lines); hunks: -6,7 +6,7; -167,7 +167,7 @@ def _run_contiguous_gemm(; symbols: _run_contiguous_gemm
  - `python/sglang/srt/models/deepseek_v2.py` modified +2/-2 (4 lines); hunks: -29,7 +29,7; -420,7 +420,7 @@ def forward(; symbols: forward
  - `python/sglang/srt/layers/attention/dsv4/indexer.py` modified +1/-1 (2 lines); hunks: -8,7 +8,7
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/dsv4/compressor.py
@@ -5,13 +5,12 @@
-from sglang.jit_kernel.deepseek_v4 import (
+from sglang.jit_kernel.dsv4 import linear_bf16_fp32, triton_create_paged_compress_data
+from sglang.jit_kernel.dsv4.compress_old import (
-    linear_bf16_fp32,
-    triton_create_paged_compress_data,
diff -- python/sglang/srt/layers/attention/dsv4/metadata.py
@@ -109,7 +109,7 @@ def __post_init__(self):
-                from sglang.jit_kernel.deepseek_v4 import get_paged_mqa_logits_metadata
+                from sglang.jit_kernel.dsv4 import get_paged_mqa_logits_metadata
@@ -124,7 +124,7 @@ def __post_init__(self):
-        from sglang.jit_kernel.deepseek_v4 import plan_topk_v2
+        from sglang.jit_kernel.dsv4 import plan_topk_v2
diff -- python/sglang/srt/layers/moe/moe_runner/deep_gemm.py
@@ -6,7 +6,7 @@
-from sglang.jit_kernel.deepseek_v4 import silu_and_mul_masked_post_quant
+from sglang.jit_kernel.dsv4 import silu_and_mul_masked_post_quant
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/dsv4/compressor.py` modified +2/-3; `python/sglang/srt/layers/attention/dsv4/metadata.py` modified +2/-2; `python/sglang/srt/layers/moe/moe_runner/deep_gemm.py` modified +2/-2; `python/sglang/srt/models/deepseek_v2.py` modified +2/-2; `python/sglang/srt/layers/attention/dsv4/indexer.py` modified +1/-1; `python/sglang/srt/layers/moe/hash_topk.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/jit_kernel/csrc/deepseek_v4/topk_1024.cuh`, `python/sglang/jit_kernel/csrc/deepseek_v4/topk_v1.cuh`, `python/sglang/jit_kernel/deepseek_v4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25974 - [Fix]: Restrict Kimi-K2.5 shared-experts fusion to Quark MXFP4 checkpoints

- Link: https://github.com/sgl-project/sglang/pull/25974
- Status/date: merged / 2026-05-21
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +11/-1, 21 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Fix]: Restrict Kimi-K2.5 shared-experts fusion to Quark MXFP4 checkpoints"; model line: DeepSeek V3.1; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v2.py`; technical summary: Covers "[Fix]: Restrict Kimi-K2.5 shared-experts fusion to Quark MXFP4 checkpoints"; the main implementation surface is `python/sglang/srt/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +11/-1 (12 lines); hunks: -2450,9 +2450,19 @@ def determine_num_fused_shared_experts(; symbols: determine_num_fused_shared_experts, touching `determine_num_fused_shared_experts`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +11/-1 (12 lines); hunks: -2450,9 +2450,19 @@ def determine_num_fused_shared_experts(; symbols: determine_num_fused_shared_experts
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -2450,9 +2450,19 @@ def determine_num_fused_shared_experts(
-            #   384 -> Kimi-K2.5 (text_config wraps DeepseekV3ForCausalLM)
+            #   384 -> Kimi-K2.5, only when the checkpoint is Quark MXFP4
+            #          (amd/Kimi-K2.5-MXFP4); the standard
+            #          moonshotai/Kimi-K2.5 (compressed-tensors) checkpoint
+            #          stores the shared expert loose and is NOT pre-fused,
+            #          so the fused path silently mis-loads it.
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +11/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25983 - feat(model_runner): remove pool/backend refs from ForwardBatch via ForwardContext

- Link: https://github.com/sgl-project/sglang/pull/25983
- Status/date: merged / 2026-05-21
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 77 files, +1227/-905, 5236 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "feat(model_runner): remove pool/backend refs from ForwardBatch via ForwardContext"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`, `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`; technical summary: Covers "feat(model_runner): remove pool/backend refs from ForwardBatch via ForwardContext"; the main implementation surface is `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`, `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/model_executor/model_runner.py` modified +107/-84 (191 lines); hunks: -146,6 +146,11; -2638,9 +2643,6 @@ def get_spec_info():; symbols: get_spec_info, run_once, maybe_init_ngram_embedding, forward_decode, touching `get_spec_info, run_once, maybe_init_ngram_embedding`; `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +70/-67 (137 lines); hunks: -65,6 +65,7; -1016,9 +1017,6 @@ def capture_one_batch_size(; symbols: capture_one_batch_size, run_once, touching `capture_one_batch_size, run_once`; `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` modified +60/-58 (118 lines); hunks: -58,6 +58,7; -387,9 +388,6 @@ def warmup_compile(self, num_tokens: int):; symbols: warmup_compile, _cache_loc_dtype, capture_one_batch_size, touching `warmup_compile, _cache_loc_dtype, capture_one_batch_size`; `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` modified +43/-44 (87 lines); hunks: -80,6 +80,11; -449,9 +454,9 @@ def _get_topk_paged(; symbols: _get_topk_paged, _get_topk_ragged, _get_topk_ragged_with_cp, touching `_get_topk_paged, _get_topk_ragged, _get_topk_ragged_with_cp`.
- Code diff details:
  - `python/sglang/srt/model_executor/model_runner.py` modified +107/-84 (191 lines); hunks: -146,6 +146,11; -2638,9 +2643,6 @@ def get_spec_info():; symbols: get_spec_info, run_once, maybe_init_ngram_embedding, forward_decode
  - `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +70/-67 (137 lines); hunks: -65,6 +65,7; -1016,9 +1017,6 @@ def capture_one_batch_size(; symbols: capture_one_batch_size, run_once
  - `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` modified +60/-58 (118 lines); hunks: -58,6 +58,7; -387,9 +388,6 @@ def warmup_compile(self, num_tokens: int):; symbols: warmup_compile, _cache_loc_dtype, capture_one_batch_size
  - `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` modified +43/-44 (87 lines); hunks: -80,6 +80,11; -449,9 +454,9 @@ def _get_topk_paged(; symbols: _get_topk_paged, _get_topk_ragged, _get_topk_ragged_with_cp
  - `python/sglang/srt/model_executor/forward_context.py` added +84/-0 (84 lines); hunks: -0,0 +1,84; symbols: ForwardContext, set_forward_context, has_forward_context, get_forward_context
- Key code excerpts:

```diff
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -146,6 +146,11 @@
+from sglang.srt.model_executor.forward_context import (
+    ForwardContext,
+    forward_context,
+    has_forward_context,
+)
@@ -2638,9 +2643,6 @@ def get_spec_info():
diff -- python/sglang/srt/model_executor/cuda_graph_runner.py
@@ -65,6 +65,7 @@
+from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
@@ -1016,9 +1017,6 @@ def capture_one_batch_size(
-            req_to_token_pool=self.model_runner.req_to_token_pool,
-            token_to_kv_pool=self.model_runner.token_to_kv_pool,
-            attn_backend=attn_backend,
@@ -1040,85 +1038,90 @@ def capture_one_batch_size(
diff -- python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py
@@ -58,6 +58,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/model_executor/model_runner.py` modified +107/-84; `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +70/-67; `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` modified +60/-58; `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` modified +43/-44; `python/sglang/srt/model_executor/forward_context.py` added +84/-0; `python/sglang/srt/model_executor/cpu_graph_runner.py` modified +39/-38
- Risk and verification: The diff ships test coverage in `test/manual/attention/test_flashattn_backend.py`, `test/manual/attention/test_flashattn_mla_backend.py`, `test/manual/attention/test_prefix_chunk_info.py`, `test/manual/attention/test_trtllm_mla_backend.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #25189 - [perf] DeepSeekV3: drop redundant FP32 upcasts in trtllm MoE paths

- Link: https://github.com/sgl-project/sglang/pull/25189
- Status/date: merged / 2026-05-22
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +3/-14, 45 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[perf] DeepSeekV3: drop redundant FP32 upcasts in trtllm MoE paths"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py`, `python/sglang/srt/models/deepseek_v2.py`; technical summary: Covers "[perf] DeepSeekV3: drop redundant FP32 upcasts in trtllm MoE paths"; the main implementation surface is `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py`, `python/sglang/srt/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py` modified +2/-14 (16 lines); hunks: -698,11 +698,7 @@ def fused_experts_none_to_flashinfer_trtllm_fp8(; -758,11 +754,7 @@ def fused_experts_none_to_flashinfer_trtllm_fp8(; symbols: fused_experts_none_to_flashinfer_trtllm_fp8, fused_experts_none_to_flashinfer_trtllm_fp4, touching `fused_experts_none_to_flashinfer_trtllm_fp8, fused_experts_none_to_flashinfer_trtllm_fp4`; `python/sglang/srt/models/deepseek_v2.py` modified +1/-0 (1 lines); hunks: -403,6 +403,7 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py` modified +2/-14 (16 lines); hunks: -698,11 +698,7 @@ def fused_experts_none_to_flashinfer_trtllm_fp8(; -758,11 +754,7 @@ def fused_experts_none_to_flashinfer_trtllm_fp8(; symbols: fused_experts_none_to_flashinfer_trtllm_fp8, fused_experts_none_to_flashinfer_trtllm_fp4
  - `python/sglang/srt/models/deepseek_v2.py` modified +1/-0 (1 lines); hunks: -403,6 +403,7 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py
@@ -698,11 +698,7 @@ def fused_experts_none_to_flashinfer_trtllm_fp8(
-                routing_logits=(
-                    router_logits.to(torch.float32)
-                    if routing_method_type == RoutingMethodType.DeepSeekV3
-                    else router_logits
-                ),
+                routing_logits=router_logits,
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -403,6 +403,7 @@ def forward(
+                    # TODO: will check the dtype to be bf16
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py` modified +2/-14; `python/sglang/srt/models/deepseek_v2.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py`, `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23351 - Support piecewise CUDA graph with NSA

- Link: https://github.com/sgl-project/sglang/pull/23351
- Status/date: merged / 2026-05-22
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 12 files, +317/-58, 682 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support piecewise CUDA graph with NSA"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/dsa/dsa_indexer.py`, `python/sglang/srt/layers/layernorm.py`, `python/sglang/srt/layers/attention/dsa_backend.py`; technical summary: Covers "Support piecewise CUDA graph with NSA"; the main implementation surface is `python/sglang/srt/layers/attention/dsa/dsa_indexer.py`, `python/sglang/srt/layers/layernorm.py`, `python/sglang/srt/layers/attention/dsa_backend.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` modified +167/-35 (202 lines); hunks: -12,6 +12,10; -95,6 +99,80; symbols: k_cache_and_topk_result, _logits_head_gate_pcg_fake_impl, logits_head_gate_pcg, BaseIndexerMetadata, touching `k_cache_and_topk_result, _logits_head_gate_pcg_fake_impl, logits_head_gate_pcg`; `python/sglang/srt/layers/layernorm.py` modified +20/-1 (21 lines); hunks: -53,7 +53,26; symbols: _layernorm_fake_impl, layernorm, touching `_layernorm_fake_impl, layernorm`; `python/sglang/srt/layers/attention/dsa_backend.py` modified +14/-3 (17 lines); hunks: -1,12 +1,15; -2180,8 +2183,8 @@ def _forward_trtllm(; symbols: _forward_trtllm, _pad_topk_indices, set_dsa_prefill_impl, touching `_forward_trtllm, _pad_topk_indices, set_dsa_prefill_impl`; `python/sglang/srt/layers/radix_attention.py` modified +14/-0 (14 lines); hunks: -160,6 +160,12 @@ def unified_attention_with_output(; -178,6 +184,14 @@ def unified_attention_with_output(; symbols: unified_attention_with_output, touching `unified_attention_with_output`.
- Code diff details:
  - `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` modified +167/-35 (202 lines); hunks: -12,6 +12,10; -95,6 +99,80; symbols: k_cache_and_topk_result, _logits_head_gate_pcg_fake_impl, logits_head_gate_pcg, BaseIndexerMetadata
  - `python/sglang/srt/layers/layernorm.py` modified +20/-1 (21 lines); hunks: -53,7 +53,26; symbols: _layernorm_fake_impl, layernorm
  - `python/sglang/srt/layers/attention/dsa_backend.py` modified +14/-3 (17 lines); hunks: -1,12 +1,15; -2180,8 +2183,8 @@ def _forward_trtllm(; symbols: _forward_trtllm, _pad_topk_indices, set_dsa_prefill_impl
  - `python/sglang/srt/layers/radix_attention.py` modified +14/-0 (14 lines); hunks: -160,6 +160,12 @@ def unified_attention_with_output(; -178,6 +184,14 @@ def unified_attention_with_output(; symbols: unified_attention_with_output
  - `python/sglang/srt/model_executor/model_runner.py` modified +6/-0 (6 lines); hunks: -2851,6 +2851,7 @@ def init_piecewise_cuda_graphs(self, force_for_draft_worke...; -2903,6 +2904,11 @@ def init_piecewise_cuda_graphs(self, force_for_draft_work...; symbols: init_piecewise_cuda_graphs
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/dsa/dsa_indexer.py
@@ -12,6 +12,10 @@
+from sglang.srt.compilation.piecewise_context_manager import (
+    get_forward_context,
+    is_in_piecewise_cuda_graph,
+)
@@ -95,6 +99,80 @@
+if _is_cuda:
diff -- python/sglang/srt/layers/layernorm.py
@@ -53,7 +53,26 @@
-            from flashinfer.norm import layernorm
+            import flashinfer.norm
+            from sglang.srt.utils.custom_op import register_custom_op
+            def _layernorm_fake_impl(
+                input: torch.Tensor,
+                gamma: torch.Tensor,
diff -- python/sglang/srt/layers/attention/dsa_backend.py
@@ -1,12 +1,15 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` modified +167/-35; `python/sglang/srt/layers/layernorm.py` modified +20/-1; `python/sglang/srt/layers/attention/dsa_backend.py` modified +14/-3; `python/sglang/srt/layers/radix_attention.py` modified +14/-0; `python/sglang/srt/model_executor/model_runner.py` modified +6/-0; `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` modified +4/-0
- Risk and verification: The diff ships test coverage in `test/registered/piecewise_cuda_graph/test_pcg_glm5_fp4.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #25843 - Route concat MLA to JIT and remove unused downcast

- Link: https://github.com/sgl-project/sglang/pull/25843
- Status/date: merged / 2026-05-23
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +8/-298, 331 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Route concat MLA to JIT and remove unused downcast"; model line: DeepSeek V3.1; category: model implementation change; main diff: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`, `python/sglang/srt/models/sarvam_moe.py`, `python/sglang/srt/layers/attention/utils.py`; technical summary: Covers "Route concat MLA to JIT and remove unused downcast"; the main implementation surface is `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`, `python/sglang/srt/models/sarvam_moe.py`, `python/sglang/srt/layers/attention/utils.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` modified +5/-1 (6 lines); hunks: -28,7 +28,11; `python/sglang/srt/models/sarvam_moe.py` modified +2/-1 (3 lines); hunks: -75,8 +75,9; `python/sglang/srt/layers/attention/utils.py` modified +1/-1 (2 lines); hunks: -10,7 +10,7; `python/sglang/jit_kernel/csrc/elementwise/cast.cuh` removed +0/-137 (137 lines); hunks: -1,137 +0,0.
- Code diff details:
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` modified +5/-1 (6 lines); hunks: -28,7 +28,11
  - `python/sglang/srt/models/sarvam_moe.py` modified +2/-1 (3 lines); hunks: -75,8 +75,9
  - `python/sglang/srt/layers/attention/utils.py` modified +1/-1 (2 lines); hunks: -10,7 +10,7
  - `python/sglang/jit_kernel/csrc/elementwise/cast.cuh` removed +0/-137 (137 lines); hunks: -1,137 +0,0
  - `python/sglang/jit_kernel/benchmark/bench_cast.py` removed +0/-106 (106 lines); hunks: -1,106 +0,0; symbols: benchmark, _report_bandwidth, fmt, report_bandwidth
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py
@@ -28,7 +28,11 @@
-    from sgl_kernel import concat_mla_k, merge_state_v2
+    from sgl_kernel import merge_state_v2
+    from sglang.jit_kernel.concat_mla import concat_mla_k
+elif _is_musa:
+    from sgl_kernel import concat_mla_k
diff -- python/sglang/srt/models/sarvam_moe.py
@@ -75,8 +75,9 @@
-        from sgl_kernel import bmm_fp8, concat_mla_k, merge_state_v2
+        from sgl_kernel import bmm_fp8, merge_state_v2
+        from sglang.jit_kernel.concat_mla import concat_mla_k
diff -- python/sglang/srt/layers/attention/utils.py
@@ -10,7 +10,7 @@
-    from sgl_kernel import concat_mla_absorb_q
+    from sglang.jit_kernel.concat_mla import concat_mla_absorb_q
diff -- python/sglang/jit_kernel/csrc/elementwise/cast.cuh
@@ -1,137 +0,0 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` modified +5/-1; `python/sglang/srt/models/sarvam_moe.py` modified +2/-1; `python/sglang/srt/layers/attention/utils.py` modified +1/-1; `python/sglang/jit_kernel/csrc/elementwise/cast.cuh` removed +0/-137; `python/sglang/jit_kernel/benchmark/bench_cast.py` removed +0/-106; `python/sglang/jit_kernel/cast.py` removed +0/-52
- Risk and verification: Runtime changes concentrate in `python/sglang/jit_kernel/benchmark/bench_cast.py`, `python/sglang/jit_kernel/cast.py`, `python/sglang/jit_kernel/csrc/elementwise/cast.cuh`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23292 - [CP] 1/N: Support MLA Prefill Context Parallel

- Link: https://github.com/sgl-project/sglang/pull/23292
- Status/date: merged / 2026-05-23
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 21 files, +900/-161, 1566 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CP] 1/N: Support MLA Prefill Context Parallel"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/flashattention_backend.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/utils/cp_utils.py`; technical summary: Covers "[CP] 1/N: Support MLA Prefill Context Parallel"; the main implementation surface is `python/sglang/srt/layers/attention/flashattention_backend.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/utils/cp_utils.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/flashattention_backend.py` modified +128/-56 (184 lines); hunks: -508,6 +508,25 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; -627,36 +646,43 @@ def forward_extend(; symbols: init_forward_metadata, forward_extend, _fa_cp_attn, _mla_cp_attn, touching `init_forward_metadata, forward_extend, _fa_cp_attn`; `python/sglang/srt/models/deepseek_v2.py` modified +73/-14 (87 lines); hunks: -123,9 +123,12; -339,6 +342,8 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`; `python/sglang/srt/layers/utils/cp_utils.py` modified +36/-19 (55 lines); hunks: -51,19 +51,41 @@ def is_prefill_cp_in_seq_split():; -395,6 +417,7 @@ def prepare_context_parallel_metadata(; symbols: is_prefill_cp_in_seq_split, is_mla_prefill_cp_enabled, mla_use_prefill_cp, can_cp_split, touching `is_prefill_cp_in_seq_split, is_mla_prefill_cp_enabled, mla_use_prefill_cp`; `python/sglang/srt/models/deepseek_nextn.py` modified +31/-8 (39 lines); hunks: -43,9 +43,12; -136,6 +139,14 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `python/sglang/srt/layers/attention/flashattention_backend.py` modified +128/-56 (184 lines); hunks: -508,6 +508,25 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; -627,36 +646,43 @@ def forward_extend(; symbols: init_forward_metadata, forward_extend, _fa_cp_attn, _mla_cp_attn
  - `python/sglang/srt/models/deepseek_v2.py` modified +73/-14 (87 lines); hunks: -123,9 +123,12; -339,6 +342,8 @@ def __init__(; symbols: __init__, forward
  - `python/sglang/srt/layers/utils/cp_utils.py` modified +36/-19 (55 lines); hunks: -51,19 +51,41 @@ def is_prefill_cp_in_seq_split():; -395,6 +417,7 @@ def prepare_context_parallel_metadata(; symbols: is_prefill_cp_in_seq_split, is_mla_prefill_cp_enabled, mla_use_prefill_cp, can_cp_split
  - `python/sglang/srt/models/deepseek_nextn.py` modified +31/-8 (39 lines); hunks: -43,9 +43,12; -136,6 +139,14 @@ def __init__(; symbols: __init__, forward
  - `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +14/-3 (17 lines); hunks: -56,6 +56,7; -567,7 +568,15 @@ def __init__(; symbols: __init__, capture_one_batch_size, replay_prepare
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/flashattention_backend.py
@@ -508,6 +508,25 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):
+            # MLA/MHA CP: prepare_mlp_sync_batch pads extend tokens up to
+            # lcm(attn_tp_size, attn_cp_size), so cache_seqlens_cp can exceed
+            # seq_lens_cpu.max(). Widen page_table by the pad delta to keep
+            # FA3's causal reads in-bounds; widened columns index KV slot 0
+            # (req_to_token is zero-init) and outputs for padding queries are
+            # discarded downstream.
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -123,9 +123,12 @@
+    can_cp_split,
+    is_prefill_context_parallel_enabled,
+    mla_use_prefill_cp,
@@ -339,6 +342,8 @@ def __init__(
+        dsa_enable_prefill_cp: bool = False,
+        mla_enable_prefill_cp: bool = False,
diff -- python/sglang/srt/layers/utils/cp_utils.py
@@ -51,19 +51,41 @@ def is_prefill_cp_in_seq_split():
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/flashattention_backend.py` modified +128/-56; `python/sglang/srt/models/deepseek_v2.py` modified +73/-14; `python/sglang/srt/layers/utils/cp_utils.py` modified +36/-19; `python/sglang/srt/models/deepseek_nextn.py` modified +31/-8; `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +14/-3; `python/sglang/srt/layers/communicator.py` modified +10/-4
- Risk and verification: The diff ships test coverage in `test/registered/cp/test_deepseek_v3_cp_single_node.py`, `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py`, `test/registered/cp/test_qwen3_30b.py`, `test/registered/kernels/test_cp_prefix_len_fa3_parity.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #25898 - [AMD] Dsv4/pr1 fix run time issue

- Link: https://github.com/sgl-project/sglang/pull/25898
- Status/date: merged / 2026-05-23
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 32 files, +2523/-129, 3203 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Dsv4/pr1 fix run time issue"; model line: DeepSeek V3.1; category: bug fix; main diff: `python/sglang/srt/layers/fused_qk_norm_rope_store.py`, `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/layers/attention/dsv4/compress_hip.py`; technical summary: Covers "[AMD] Dsv4/pr1 fix run time issue"; the main implementation surface is `python/sglang/srt/layers/fused_qk_norm_rope_store.py`, `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/layers/attention/dsv4/compress_hip.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/fused_qk_norm_rope_store.py` added +380/-0 (380 lines); hunks: -0,0 +1,380; symbols: _batched_rmsnorm, _gptj_rotate, _batched_rope, _fused_qk_norm_rope_store_kernel, touching `_batched_rmsnorm, _gptj_rotate, _batched_rope`; `python/sglang/srt/models/deepseek_v4.py` modified +153/-28 (181 lines); hunks: -96,6 +96,8; -105,6 +107,29; symbols: _fused_rmsnorm_fp8_quant, __init__, _forward_prepare_multi_stream, touching `_fused_rmsnorm_fp8_quant, __init__, _forward_prepare_multi_stream`; `python/sglang/srt/layers/attention/dsv4/compress_hip.py` modified +91/-20 (111 lines); hunks: -16,6 +16,11; -91,15 +96,24 @@ class CompressorHip(_CompressorBase):; symbols: CompressorHip, __init__, use_fused_compress, use_hip_fused_compress, touching `CompressorHip, __init__, use_fused_compress`; `python/sglang/srt/models/deepseek_v2.py` modified +39/-1 (40 lines); hunks: -255,6 +255,11 @@ def __init__(; -316,8 +321,41 @@ def forward(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `python/sglang/srt/layers/fused_qk_norm_rope_store.py` added +380/-0 (380 lines); hunks: -0,0 +1,380; symbols: _batched_rmsnorm, _gptj_rotate, _batched_rope, _fused_qk_norm_rope_store_kernel
  - `python/sglang/srt/models/deepseek_v4.py` modified +153/-28 (181 lines); hunks: -96,6 +96,8; -105,6 +107,29; symbols: _fused_rmsnorm_fp8_quant, __init__, _forward_prepare_multi_stream
  - `python/sglang/srt/layers/attention/dsv4/compress_hip.py` modified +91/-20 (111 lines); hunks: -16,6 +16,11; -91,15 +96,24 @@ class CompressorHip(_CompressorBase):; symbols: CompressorHip, __init__, use_fused_compress, use_hip_fused_compress
  - `python/sglang/srt/models/deepseek_v2.py` modified +39/-1 (40 lines); hunks: -255,6 +255,11 @@ def __init__(; -316,8 +321,41 @@ def forward(; symbols: __init__, forward
  - `python/sglang/srt/layers/layernorm.py` modified +6/-0 (6 lines); hunks: -303,6 +303,12 @@ def forward_aiter(; symbols: forward_aiter
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/fused_qk_norm_rope_store.py
@@ -0,0 +1,380 @@
+"""Fused Q per-head RMSNorm + KV RMSNorm + RoPE + FP8 nope quant + paged SWA store.
+Single Triton kernel replacing the 2-kernel path:
+  1. fused_reduce_qk_norm_rope_swa_write (norm + RoPE)
+  2. store_cache -> fused_store_cache (FP8 quant + paged scatter)
+Grid: (cdiv(M, BLOCK_SIZE_M), num_local_heads + 1).
+  pid_h < num_local_heads: Q head programs (split-K reduce + norm + RoPE)
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -96,6 +96,8 @@
+    get_bool_env_var,
+    is_gfx95_supported,
@@ -105,6 +107,29 @@
+_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
+_is_gfx95_supported = is_gfx95_supported()
+if _use_aiter:
diff -- python/sglang/srt/layers/attention/dsv4/compress_hip.py
@@ -16,6 +16,11 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/fused_qk_norm_rope_store.py` added +380/-0; `python/sglang/srt/models/deepseek_v4.py` modified +153/-28; `python/sglang/srt/layers/attention/dsv4/compress_hip.py` modified +91/-20; `python/sglang/srt/models/deepseek_v2.py` modified +39/-1; `python/sglang/srt/layers/layernorm.py` modified +6/-0; `python/sglang/jit_kernel/triton_store_cache.py` added +237/-0
  - other: `sgl-kernel/csrc/elementwise/dsv4_norm_rope.cu` added +700/-0; `sgl-kernel/csrc/elementwise/deepseek_v4_topk.cu` added +372/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/jit_kernel/csrc/deepseek_v4/c128_v2.cuh`, `python/sglang/jit_kernel/csrc/deepseek_v4/c4_v2.cuh`, `python/sglang/jit_kernel/csrc/deepseek_v4/c_plan.cuh`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #26208 - [AMD] Dsv4/pr2 compressor opt

- Link: https://github.com/sgl-project/sglang/pull/26208
- Status/date: merged / 2026-05-26
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 31 files, +8829/-149, 6378 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Dsv4/pr2 compressor opt"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_fused.py`, `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_dsv4.py`, `python/sglang/srt/layers/attention/dsv4/fused_compress_triton.py`; technical summary: Covers "[AMD] Dsv4/pr2 compressor opt"; the main implementation surface is `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_fused.py`, `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_dsv4.py`, `python/sglang/srt/layers/attention/dsv4/fused_compress_triton.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_fused.py` added +3089/-0 (3089 lines); `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_dsv4.py` added +1355/-0 (1355 lines); hunks: -0,0 +1,1355; symbols: _gather_dequant_dsv4_kernel, _gather_dequant_dsv4_kernel_fixed_128, gather_dequant_fp8_dsv4, _gather_dequant_dsv4_1d_fused_kernel, touching `_gather_dequant_dsv4_kernel, _gather_dequant_dsv4_kernel_fixed_128, gather_dequant_fp8_dsv4`; `python/sglang/srt/layers/attention/dsv4/fused_compress_triton.py` added +954/-0 (954 lines); hunks: -0,0 +1,954; symbols: _fused_ape_pool_norm_rope_kernel, fused_ape_pool_norm_rope, _c4_decode_kernel, _c4_prefill_compress_kernel, touching `_fused_ape_pool_norm_rope_kernel, fused_ape_pool_norm_rope, _c4_decode_kernel`; `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_common.py` added +585/-0 (585 lines); hunks: -0,0 +1,585; symbols: _bucket_total_tokens, _get_workload_size_category, _unified_sparse_decode_kernel, run_unified_attention, touching `_bucket_total_tokens, _get_workload_size_category, _unified_sparse_decode_kernel`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_fused.py` added +3089/-0 (3089 lines)
  - `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_dsv4.py` added +1355/-0 (1355 lines); hunks: -0,0 +1,1355; symbols: _gather_dequant_dsv4_kernel, _gather_dequant_dsv4_kernel_fixed_128, gather_dequant_fp8_dsv4, _gather_dequant_dsv4_1d_fused_kernel
  - `python/sglang/srt/layers/attention/dsv4/fused_compress_triton.py` added +954/-0 (954 lines); hunks: -0,0 +1,954; symbols: _fused_ape_pool_norm_rope_kernel, fused_ape_pool_norm_rope, _c4_decode_kernel, _c4_prefill_compress_kernel
  - `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_common.py` added +585/-0 (585 lines); hunks: -0,0 +1,585; symbols: _bucket_total_tokens, _get_workload_size_category, _unified_sparse_decode_kernel, run_unified_attention
  - `python/sglang/srt/layers/attention/dsv4/compressor_v2.py` modified +516/-25 (541 lines); hunks: -10,6 +10,7; -24,12 +25,380; symbols: _c128_compress_decode_kernel, _c128_compress_prefill_write_kernel, _c128_compress_prefill_compress_kernel, _compress_forward_c128_triton
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_dsv4.py
@@ -0,0 +1,1355 @@
+"""
+Triton MLA Decode Kernels for DSV4 (d_qk=512).
+This module contains DSV4-specific gather+dequant kernels and the main
+sparse attention decode entry point for DSV4.
+"""
+import os
diff -- python/sglang/srt/layers/attention/dsv4/fused_compress_triton.py
@@ -0,0 +1,954 @@
+"""HIP fused compressor kernels using the NV/main metadata contract.
+The public wrappers mirror ``compress_forward``:
+ decode: indices, seq_lens, extra_data
+ prefill: indices, compress_plan, write_plan, extra_data
+Prefill plans are the upstream 16-byte ``PrefillPlan`` structs stored as
+``uint8[:, 16]``. The wrappers reinterpret them as ``int32[:, 4]`` before
diff -- python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_common.py
@@ -0,0 +1,585 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_fused.py` added +3089/-0; `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_dsv4.py` added +1355/-0; `python/sglang/srt/layers/attention/dsv4/fused_compress_triton.py` added +954/-0; `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_common.py` added +585/-0; `python/sglang/srt/layers/attention/dsv4/compressor_v2.py` modified +516/-25; `python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_splitk.py` added +534/-0
- Risk and verification: The diff ships test coverage in `sgl-kernel/tests/test_dsv4_norm_rope.py`, `test/manual/dsv4/test_fused_compress_attn_hip.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #23269 - Support batch size > 1 when enable CP

- Link: https://github.com/sgl-project/sglang/pull/23269
- Status/date: merged / 2026-05-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 13 files, +268/-305, 797 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support batch size > 1 when enable CP"; model line: DeepSeek V3.1; category: model support/runtime entry; main diff: `python/sglang/srt/layers/utils/cp_utils.py`, `python/sglang/srt/layers/attention/dsa/dsa_indexer.py`, `python/sglang/srt/model_executor/forward_batch_info.py`; technical summary: Covers "Support batch size > 1 when enable CP"; the main implementation surface is `python/sglang/srt/layers/utils/cp_utils.py`, `python/sglang/srt/layers/attention/dsa/dsa_indexer.py`, `python/sglang/srt/model_executor/forward_batch_info.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/utils/cp_utils.py` modified +236/-153 (389 lines); hunks: -20,24 +20,41; -67,25 +84,45 @@ def mla_use_prefill_cp(forward_batch, mla_enable_prefill_cp=...; symbols: ContextParallelMetadata, is_prefill_context_parallel_enabled, mla_use_prefill_cp, can_cp_split, touching `ContextParallelMetadata, is_prefill_context_parallel_enabled, mla_use_prefill_cp`; `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` modified +8/-4 (12 lines); hunks: -1455,10 +1455,14 @@ def forward_cuda(; symbols: forward_cuda, touching `forward_cuda`; `python/sglang/srt/model_executor/forward_batch_info.py` modified +3/-2 (5 lines); hunks: -889,10 +889,11 @@ def prepare_mlp_sync_batch(self, model_runner: ModelRunner):; symbols: prepare_mlp_sync_batch, touching `prepare_mlp_sync_batch`; `python/sglang/srt/layers/attention/dsa/utils.py` modified +3/-1 (4 lines); hunks: -133,7 +133,9 @@ def cal_padded_tokens(forward_batch: "ForwardBatch"):; symbols: cal_padded_tokens, touching `cal_padded_tokens`.
- Code diff details:
  - `python/sglang/srt/layers/utils/cp_utils.py` modified +236/-153 (389 lines); hunks: -20,24 +20,41; -67,25 +84,45 @@ def mla_use_prefill_cp(forward_batch, mla_enable_prefill_cp=...; symbols: ContextParallelMetadata, is_prefill_context_parallel_enabled, mla_use_prefill_cp, can_cp_split
  - `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` modified +8/-4 (12 lines); hunks: -1455,10 +1455,14 @@ def forward_cuda(; symbols: forward_cuda
  - `python/sglang/srt/model_executor/forward_batch_info.py` modified +3/-2 (5 lines); hunks: -889,10 +889,11 @@ def prepare_mlp_sync_batch(self, model_runner: ModelRunner):; symbols: prepare_mlp_sync_batch
  - `python/sglang/srt/layers/attention/dsa/utils.py` modified +3/-1 (4 lines); hunks: -133,7 +133,9 @@ def cal_padded_tokens(forward_batch: "ForwardBatch"):; symbols: cal_padded_tokens
  - `python/sglang/srt/models/deepseek_nextn.py` modified +2/-2 (4 lines); hunks: -311,7 +311,7 @@ def forward(; -320,7 +320,7 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/utils/cp_utils.py
@@ -20,24 +20,41 @@
+    # Layout lists have length bs * cp_segment_num (= bs * 2 * cp_size).
-    max_rank_len: List[int] = None
-    per_rank_actual_token: List[int] = None
-    reverse_split_len: List[int] = None
+    reverse_split_len: List[int] = None
+    # Per-rank-aggregate lists have length cp_size.
diff -- python/sglang/srt/layers/attention/dsa/dsa_indexer.py
@@ -1455,10 +1455,14 @@ def forward_cuda(
-                    kv_len_prev = forward_batch.attn_cp_metadata.kv_len_prev
-                    kv_len_next = forward_batch.attn_cp_metadata.kv_len_next
-                    actual_seq_q_prev = forward_batch.attn_cp_metadata.actual_seq_q_prev
-                    actual_seq_q_next = forward_batch.attn_cp_metadata.actual_seq_q_next
+                    kv_len_prev = forward_batch.attn_cp_metadata.kv_len_prev_list[0]
+                    kv_len_next = forward_batch.attn_cp_metadata.kv_len_next_list[0]
diff -- python/sglang/srt/model_executor/forward_batch_info.py
@@ -889,10 +889,11 @@ def prepare_mlp_sync_batch(self, model_runner: ModelRunner):
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/utils/cp_utils.py` modified +236/-153; `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` modified +8/-4; `python/sglang/srt/model_executor/forward_batch_info.py` modified +3/-2; `python/sglang/srt/layers/attention/dsa/utils.py` modified +3/-1; `python/sglang/srt/models/deepseek_nextn.py` modified +2/-2; `python/sglang/srt/models/deepseek_v2.py` modified +2/-2
- Risk and verification: The diff ships test coverage in `test/registered/cp/test_qwen3_30b.py`, `test/registered/kernels/test_cp_prefix_len_fa3_parity.py`, `test/registered/kernels/test_mla_cp_fa3_parity.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #24737 - Support Flashinfer Cute-DSL MLA attention

- Link: https://github.com/sgl-project/sglang/pull/24737
- Status/date: merged / 2026-05-28
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +101/-13, 267 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support Flashinfer Cute-DSL MLA attention"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/trtllm_mla_backend.py`, `python/sglang/srt/layers/attention/attention_registry.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`; technical summary: Covers "Support Flashinfer Cute-DSL MLA attention"; the main implementation surface is `python/sglang/srt/layers/attention/trtllm_mla_backend.py`, `python/sglang/srt/layers/attention/attention_registry.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/trtllm_mla_backend.py` modified +35/-9 (44 lines); hunks: -229,6 +229,11 @@ def _quantize_fp8_qkv(q, k, v, layer):; -263,6 +268,7 @@ def __init__(; symbols: _quantize_fp8_qkv, __init__, _run_decode_kernel, touching `_quantize_fp8_qkv, __init__, _run_decode_kernel`; `python/sglang/srt/layers/attention/attention_registry.py` modified +9/-0 (9 lines); hunks: -74,6 +74,15 @@ def create_tokenspeed_mla_backend(runner):; symbols: create_tokenspeed_mla_backend, create_cutedsl_mla_backend, create_aiter_backend, touching `create_tokenspeed_mla_backend, create_cutedsl_mla_backend, create_aiter_backend`; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +2/-1 (3 lines); hunks: -697,7 +697,8 @@ def _fuse_rope_for_trtllm_mla(; symbols: _fuse_rope_for_trtllm_mla, touching `_fuse_rope_for_trtllm_mla`; `python/sglang/srt/model_executor/model_runner.py` modified +2/-0 (2 lines); hunks: -247,6 +247,7; -261,6 +262,7.
- Code diff details:
  - `python/sglang/srt/layers/attention/trtllm_mla_backend.py` modified +35/-9 (44 lines); hunks: -229,6 +229,11 @@ def _quantize_fp8_qkv(q, k, v, layer):; -263,6 +268,7 @@ def __init__(; symbols: _quantize_fp8_qkv, __init__, _run_decode_kernel
  - `python/sglang/srt/layers/attention/attention_registry.py` modified +9/-0 (9 lines); hunks: -74,6 +74,15 @@ def create_tokenspeed_mla_backend(runner):; symbols: create_tokenspeed_mla_backend, create_cutedsl_mla_backend, create_aiter_backend
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +2/-1 (3 lines); hunks: -697,7 +697,8 @@ def _fuse_rope_for_trtllm_mla(; symbols: _fuse_rope_for_trtllm_mla
  - `python/sglang/srt/model_executor/model_runner.py` modified +2/-0 (2 lines); hunks: -247,6 +247,7; -261,6 +262,7
  - `python/sglang/srt/models/deepseek_common/utils.py` modified +1/-0 (1 lines); hunks: -62,6 +62,7
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/trtllm_mla_backend.py
@@ -229,6 +229,11 @@ def _quantize_fp8_qkv(q, k, v, layer):
+# cute-dsl needs its own workspace: it overwrites the buffer with split-KV
+# partials, which corrupts the trtllm-gen multiCtasKv counters that rely on the
+# zero-init buffer (they share it under attention-backend=cutedsl_mla, where
+# draft-extend falls back to trtllm-gen) and deadlocks the reduction.
+global_cute_dsl_workspace_buffer = None
@@ -263,6 +268,7 @@ def __init__(
diff -- python/sglang/srt/layers/attention/attention_registry.py
@@ -74,6 +74,15 @@ def create_tokenspeed_mla_backend(runner):
+@register_attention_backend("cutedsl_mla")
+def create_cutedsl_mla_backend(runner):
+    if not runner.use_mla_backend:
+        raise ValueError("cutedsl_mla backend can only be used with MLA models.")
+    from sglang.srt.layers.attention.trtllm_mla_backend import TRTLLMMLABackend
+    return TRTLLMMLABackend(runner, backend="cute-dsl")
diff -- python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py
@@ -697,7 +697,8 @@ def _fuse_rope_for_trtllm_mla(
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/trtllm_mla_backend.py` modified +35/-9; `python/sglang/srt/layers/attention/attention_registry.py` modified +9/-0; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +2/-1; `python/sglang/srt/model_executor/model_runner.py` modified +2/-0; `python/sglang/srt/models/deepseek_common/utils.py` modified +1/-0; `python/sglang/srt/server_args.py` modified +30/-0
  - docs: `docs_new/docs/advanced_features/attention_backend.mdx` modified +11/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/attention_registry.py`, `python/sglang/srt/layers/attention/trtllm_mla_backend.py`, `python/sglang/srt/model_executor/model_runner.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25755 - [Fix][NPU] Preserve existing packed_modules_mapping when merging model-level fused module mappings

- Link: https://github.com/sgl-project/sglang/pull/25755
- Status/date: merged / 2026-05-29
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +8/-2, 31 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Fix][NPU] Preserve existing packed_modules_mapping when merging model-level fused module mappings"; model line: DeepSeek V3.1; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/quantization/base_config.py`, `python/sglang/srt/layers/quantization/modelslim/modelslim.py`; technical summary: Covers "[Fix][NPU] Preserve existing packed_modules_mapping when merging model-level fused module mappings"; the main implementation surface is `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/quantization/base_config.py`, `python/sglang/srt/layers/quantization/modelslim/modelslim.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +2/-2 (4 lines); hunks: -2462,8 +2462,8 @@ def __init__(; symbols: __init__, touching `__init__`; `python/sglang/srt/layers/quantization/base_config.py` modified +3/-0 (3 lines); hunks: -131,6 +131,9 @@ def __init__(self):; symbols: __init__, update_packed_modules_mapping, get_name, touching `__init__, update_packed_modules_mapping, get_name`; `python/sglang/srt/layers/quantization/modelslim/modelslim.py` modified +3/-0 (3 lines); hunks: -108,6 +108,9 @@ def __init__(self, quant_config: Dict[str, Any] = {}):; symbols: __init__, update_packed_modules_mapping, get_linear_method, touching `__init__, update_packed_modules_mapping, get_linear_method`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +2/-2 (4 lines); hunks: -2462,8 +2462,8 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/layers/quantization/base_config.py` modified +3/-0 (3 lines); hunks: -131,6 +131,9 @@ def __init__(self):; symbols: __init__, update_packed_modules_mapping, get_name
  - `python/sglang/srt/layers/quantization/modelslim/modelslim.py` modified +3/-0 (3 lines); hunks: -108,6 +108,9 @@ def __init__(self, quant_config: Dict[str, Any] = {}):; symbols: __init__, update_packed_modules_mapping, get_linear_method
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -2462,8 +2462,8 @@ def __init__(
-        if quant_config is not None and hasattr(quant_config, "packed_modules_mapping"):
-            quant_config.packed_modules_mapping = self.packed_modules_mapping
+        if quant_config is not None:
+            quant_config.update_packed_modules_mapping(self.packed_modules_mapping)
diff -- python/sglang/srt/layers/quantization/base_config.py
@@ -131,6 +131,9 @@ def __init__(self):
+    def update_packed_modules_mapping(self, mapping: Dict[str, List[str]]) -> None:
+        self.packed_modules_mapping = mapping
diff -- python/sglang/srt/layers/quantization/modelslim/modelslim.py
@@ -108,6 +108,9 @@ def __init__(self, quant_config: Dict[str, Any] = {}):
+    def update_packed_modules_mapping(self, mapping: Dict[str, List[str]]) -> None:
+        self.packed_modules_mapping.update(mapping)
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +2/-2; `python/sglang/srt/layers/quantization/base_config.py` modified +3/-0; `python/sglang/srt/layers/quantization/modelslim/modelslim.py` modified +3/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/quantization/base_config.py`, `python/sglang/srt/layers/quantization/modelslim/modelslim.py`, `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25463 - [ROCm] Eliminate redundant contiguous copy in MLA attention on ROCm MXFP4

- Link: https://github.com/sgl-project/sglang/pull/25463
- Status/date: merged / 2026-05-29
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +21/-5, 50 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm] Eliminate redundant contiguous copy in MLA attention on ROCm MXFP4"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`; technical summary: Covers "[ROCm] Eliminate redundant contiguous copy in MLA attention on ROCm MXFP4"; the main implementation surface is `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +21/-5 (26 lines); hunks: -575,13 +575,18 @@ def forward_absorb_core(; -590,6 +595,7 @@ def forward_absorb_core(; symbols: forward_absorb_core, touching `forward_absorb_core`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +21/-5 (26 lines); hunks: -575,13 +575,18 @@ def forward_absorb_core(; -590,6 +595,7 @@ def forward_absorb_core(; symbols: forward_absorb_core
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py
@@ -575,13 +575,18 @@ def forward_absorb_core(
-                attn_bmm_output = torch.empty(
-                    x.shape[0],
-                    x.shape[1],
-                    self.w_vc.shape[2],
+                B_heads, M_batch = x.shape[0], x.shape[1]
+                N_vdim = self.w_vc.shape[2]
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +21/-5
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #26673 - [refactor] remove unused op_mlp

- Link: https://github.com/sgl-project/sglang/pull/26673
- Status/date: merged / 2026-05-29
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +0/-53, 95 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[refactor] remove unused op_mlp"; model line: DeepSeek V3.1; category: model implementation change; main diff: `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_lite.py`; technical summary: Covers "[refactor] remove unused op_mlp"; the main implementation surface is `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_lite.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +0/-13 (13 lines); hunks: -2114,19 +2114,6 @@ def op_comm_prepare_mlp(self, state):; symbols: op_comm_prepare_mlp, op_mlp, op_comm_postprocess_layer, touching `op_comm_prepare_mlp, op_mlp, op_comm_postprocess_layer`; `python/sglang/srt/models/glm4_moe.py` modified +0/-13 (13 lines); hunks: -1017,19 +1017,6 @@ def op_comm_prepare_mlp(self, state):; symbols: op_comm_prepare_mlp, op_mlp, op_comm_postprocess_layer, touching `op_comm_prepare_mlp, op_mlp, op_comm_postprocess_layer`; `python/sglang/srt/models/glm4_moe_lite.py` modified +0/-13 (13 lines); hunks: -737,19 +737,6 @@ def op_comm_prepare_mlp(self, state):; symbols: op_comm_prepare_mlp, op_mlp, op_comm_postprocess_layer, touching `op_comm_prepare_mlp, op_mlp, op_comm_postprocess_layer`; `python/sglang/srt/models/minimax_m2.py` modified +0/-6 (6 lines); hunks: -1069,12 +1069,6 @@ def op_comm_prepare_mlp(self, state):; symbols: op_comm_prepare_mlp, op_mlp, op_comm_postprocess_layer, touching `op_comm_prepare_mlp, op_mlp, op_comm_postprocess_layer`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +0/-13 (13 lines); hunks: -2114,19 +2114,6 @@ def op_comm_prepare_mlp(self, state):; symbols: op_comm_prepare_mlp, op_mlp, op_comm_postprocess_layer
  - `python/sglang/srt/models/glm4_moe.py` modified +0/-13 (13 lines); hunks: -1017,19 +1017,6 @@ def op_comm_prepare_mlp(self, state):; symbols: op_comm_prepare_mlp, op_mlp, op_comm_postprocess_layer
  - `python/sglang/srt/models/glm4_moe_lite.py` modified +0/-13 (13 lines); hunks: -737,19 +737,6 @@ def op_comm_prepare_mlp(self, state):; symbols: op_comm_prepare_mlp, op_mlp, op_comm_postprocess_layer
  - `python/sglang/srt/models/minimax_m2.py` modified +0/-6 (6 lines); hunks: -1069,12 +1069,6 @@ def op_comm_prepare_mlp(self, state):; symbols: op_comm_prepare_mlp, op_mlp, op_comm_postprocess_layer
  - `python/sglang/srt/models/mimo_v2.py` modified +0/-4 (4 lines); hunks: -808,10 +808,6 @@ def op_comm_prepare_mlp(self, state):; symbols: op_comm_prepare_mlp, op_mlp, op_comm_postprocess_layer
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -2114,19 +2114,6 @@ def op_comm_prepare_mlp(self, state):
-    def op_mlp(self, state):
-        hidden_states = state.pop("hidden_states_mlp_input")
-        if not (
-            enable_moe_dense_fully_dp()
-            and (not self.is_layer_sparse)
-            and hidden_states.shape[0] == 0
diff -- python/sglang/srt/models/glm4_moe.py
@@ -1017,19 +1017,6 @@ def op_comm_prepare_mlp(self, state):
-    def op_mlp(self, state):
-        hidden_states = state.pop("hidden_states_mlp_input")
-        if not (
-            enable_moe_dense_fully_dp()
-            and (not self.is_layer_sparse)
-            and hidden_states.shape[0] == 0
diff -- python/sglang/srt/models/glm4_moe_lite.py
@@ -737,19 +737,6 @@ def op_comm_prepare_mlp(self, state):
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +0/-13; `python/sglang/srt/models/glm4_moe.py` modified +0/-13; `python/sglang/srt/models/glm4_moe_lite.py` modified +0/-13; `python/sglang/srt/models/minimax_m2.py` modified +0/-6; `python/sglang/srt/models/mimo_v2.py` modified +0/-4; `python/sglang/srt/models/qwen3_moe.py` modified +0/-4
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_lite.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #26626 - [perf] Fuse NVFP4 gate_up_gemm + swiglu + output FP4 quant

- Link: https://github.com/sgl-project/sglang/pull/26626
- Status/date: merged / 2026-05-29
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +3137/-7, 178 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[perf] Fuse NVFP4 gate_up_gemm + swiglu + output FP4 quant"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/quantization/nvfp4_gemm_swiglu_nvfp4_quant.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/models/deepseek_v2.py`; technical summary: Covers "[perf] Fuse NVFP4 gate_up_gemm + swiglu + output FP4 quant"; the main implementation surface is `python/sglang/srt/layers/quantization/nvfp4_gemm_swiglu_nvfp4_quant.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/quantization/nvfp4_gemm_swiglu_nvfp4_quant.py` added +3015/-0 (3015 lines); `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +65/-6 (71 lines); hunks: -1503,6 +1503,15 @@ def process_weights_after_loading(self, layer: torch.nn.M...; -1518,23 +1527,73 @@ def process_weights_after_loading(self, layer: torch.nn....; symbols: process_weights_after_loading, apply, touching `process_weights_after_loading, apply`; `python/sglang/srt/models/deepseek_v2.py` modified +55/-0 (55 lines); hunks: -275,6 +275,35 @@ def forward(; -673,6 +702,32 @@ def __init__(; symbols: forward, __init__, touching `forward, __init__`; `.codespellrc` modified +1/-1 (2 lines); hunks: -1,3 +1,3.
- Code diff details:
  - `python/sglang/srt/layers/quantization/nvfp4_gemm_swiglu_nvfp4_quant.py` added +3015/-0 (3015 lines)
  - `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +65/-6 (71 lines); hunks: -1503,6 +1503,15 @@ def process_weights_after_loading(self, layer: torch.nn.M...; -1518,23 +1527,73 @@ def process_weights_after_loading(self, layer: torch.nn....; symbols: process_weights_after_loading, apply
  - `python/sglang/srt/models/deepseek_v2.py` modified +55/-0 (55 lines); hunks: -275,6 +275,35 @@ def forward(; -673,6 +702,32 @@ def __init__(; symbols: forward, __init__
  - `.codespellrc` modified +1/-1 (2 lines); hunks: -1,3 +1,3
  - `python/sglang/srt/environ.py` modified +1/-0 (1 lines); hunks: -638,6 +638,7 @@ class Envs:; symbols: Envs
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/modelopt_quant.py
@@ -1503,6 +1503,15 @@ def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
+        # Snapshot the raw (pre-swizzle) scale BEFORE alias_or_bind_derived_param
+        # overwrites layer.weight_scale.data in-place via .copy_() on the broadcast
+        # path. Without this, the swiglu side-channel below would read the swizzled
+        # bytes when it later re-reads layer.weight_scale.
+        raw_scale_snapshot = (
+            (scales.squeeze(0) if scale_ndim == 2 else scales).detach().clone()
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -275,6 +275,35 @@ def forward(
+        if (
+            getattr(self, "_enable_nvfp4_gemm_swiglu_fusion", False)
+            and self.swiglu_limit is None
+            and not isinstance(x, tuple)
+        ):
+            from flashinfer import fp4_quantize
diff -- .codespellrc
@@ -1,3 +1,3 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/nvfp4_gemm_swiglu_nvfp4_quant.py` added +3015/-0; `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +65/-6; `python/sglang/srt/models/deepseek_v2.py` modified +55/-0; `python/sglang/srt/environ.py` modified +1/-0
  - other: `.codespellrc` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/environ.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/layers/quantization/nvfp4_gemm_swiglu_nvfp4_quant.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #26970 - [perf] Replicate embed_tokens to drop the post-embed all-reduce

- Link: https://github.com/sgl-project/sglang/pull/26970
- Status/date: merged / 2026-06-02
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +37/-4, 126 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[perf] Replicate embed_tokens to drop the post-embed all-reduce"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/vocab_parallel_embedding.py`, `python/sglang/srt/models/deepseek_nextn.py`, `python/sglang/srt/models/deepseek_v2.py`; technical summary: Covers "[perf] Replicate embed_tokens to drop the post-embed all-reduce"; the main implementation surface is `python/sglang/srt/layers/vocab_parallel_embedding.py`, `python/sglang/srt/models/deepseek_nextn.py`, `python/sglang/srt/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/vocab_parallel_embedding.py` modified +24/-0 (24 lines); hunks: -19,13 +19,15; -160,6 +162,28 @@ def get_masked_input_and_mask(; symbols: get_masked_input_and_mask, get_embedding_tp_kwargs, VocabParallelEmbedding, touching `get_masked_input_and_mask, get_embedding_tp_kwargs, VocabParallelEmbedding`; `python/sglang/srt/models/deepseek_nextn.py` modified +2/-2 (4 lines); hunks: -36,7 +36,6; -55,6 +54,7; symbols: __init__, touching `__init__`; `python/sglang/srt/models/deepseek_v2.py` modified +2/-2 (4 lines); hunks: -78,7 +78,6; -134,6 +133,7; symbols: __init__, touching `__init__`; `python/sglang/srt/models/kimi_k25_eagle3.py` modified +2/-0 (2 lines); hunks: -33,6 +33,7; -199,6 +200,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/layers/vocab_parallel_embedding.py` modified +24/-0 (24 lines); hunks: -19,13 +19,15; -160,6 +162,28 @@ def get_masked_input_and_mask(; symbols: get_masked_input_and_mask, get_embedding_tp_kwargs, VocabParallelEmbedding
  - `python/sglang/srt/models/deepseek_nextn.py` modified +2/-2 (4 lines); hunks: -36,7 +36,6; -55,6 +54,7; symbols: __init__
  - `python/sglang/srt/models/deepseek_v2.py` modified +2/-2 (4 lines); hunks: -78,7 +78,6; -134,6 +133,7; symbols: __init__
  - `python/sglang/srt/models/kimi_k25_eagle3.py` modified +2/-0 (2 lines); hunks: -33,6 +33,7; -199,6 +200,7 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/environ.py` modified +7/-0 (7 lines); hunks: -735,6 +735,13 @@ class Envs:; symbols: Envs
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/vocab_parallel_embedding.py
@@ -19,13 +19,15 @@
+from sglang.srt.environ import envs
+    is_dp_attention_enabled,
@@ -160,6 +162,28 @@ def get_masked_input_and_mask(
+def get_embedding_tp_kwargs() -> dict:
+    """Vocab-parallel layout kwargs for the *input embedding* of models that
+    support embedding replication (the DeepSeek-V2 target family: DeepSeek
diff -- python/sglang/srt/models/deepseek_nextn.py
@@ -36,7 +36,6 @@
-    is_dp_attention_enabled,
@@ -55,6 +54,7 @@
+    get_embedding_tp_kwargs,
@@ -99,8 +99,8 @@ def __init__(
-            use_attn_tp_group=is_dp_attention_enabled(),
+            **get_embedding_tp_kwargs(),
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -78,7 +78,6 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/vocab_parallel_embedding.py` modified +24/-0; `python/sglang/srt/models/deepseek_nextn.py` modified +2/-2; `python/sglang/srt/models/deepseek_v2.py` modified +2/-2; `python/sglang/srt/models/kimi_k25_eagle3.py` modified +2/-0; `python/sglang/srt/environ.py` modified +7/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/environ.py`, `python/sglang/srt/layers/vocab_parallel_embedding.py`, `python/sglang/srt/models/deepseek_nextn.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27329 - [LoRA] Experimental fast LoRA path with `experimental_sgl_trtllm` MoE backend for FP8 and NVFP4 models

- Link: https://github.com/sgl-project/sglang/pull/27329
- Status/date: merged / 2026-06-05
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 52 files, +16548/-24, 13041 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[LoRA] Experimental fast LoRA path with `experimental_sgl_trtllm` MoE backend for FP8 and NVFP4 models"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/layers/moe/moe_runner/triton_utils/moe_align_block_size.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`; technical summary: Covers "[LoRA] Experimental fast LoRA path with `experimental_sgl_trtllm` MoE backend for FP8 and NVFP4 models"; the main implementation surface is `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/layers/moe/moe_runner/triton_utils/moe_align_block_size.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/moe/topk.py` modified +131/-0 (131 lines); hunks: -113,6 +113,8 @@ def routing(; -220,6 +222,13 @@ class TopKOutputChecker:; symbols: routing, TopKOutputChecker, format_is_standard, format, touching `routing, TopKOutputChecker, format_is_standard`; `python/sglang/srt/layers/moe/moe_runner/triton_utils/moe_align_block_size.py` modified +36/-10 (46 lines); hunks: -5,8 +5,11; -74,14 +77,37 @@ def moe_align_block_size(; symbols: moe_align_block_size, touching `moe_align_block_size`; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +37/-2 (39 lines); hunks: -5,6 +5,7; -45,6 +46,8; symbols: forward_absorb_prepare, forward_absorb_core, touching `forward_absorb_prepare, forward_absorb_core`; `python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_kernels.py` modified +33/-5 (38 lines); hunks: -379,7 +379,9 @@ def fused_moe_kernel(; -440,11 +442,9 @@ def fused_moe_kernel(; symbols: fused_moe_kernel, invoke_fused_moe_kernel, touching `fused_moe_kernel, invoke_fused_moe_kernel`.
- Code diff details:
  - `python/sglang/srt/layers/moe/topk.py` modified +131/-0 (131 lines); hunks: -113,6 +113,8 @@ def routing(; -220,6 +222,13 @@ class TopKOutputChecker:; symbols: routing, TopKOutputChecker, format_is_standard, format
  - `python/sglang/srt/layers/moe/moe_runner/triton_utils/moe_align_block_size.py` modified +36/-10 (46 lines); hunks: -5,8 +5,11; -74,14 +77,37 @@ def moe_align_block_size(; symbols: moe_align_block_size
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +37/-2 (39 lines); hunks: -5,6 +5,7; -45,6 +46,8; symbols: forward_absorb_prepare, forward_absorb_core
  - `python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_kernels.py` modified +33/-5 (38 lines); hunks: -379,7 +379,9 @@ def fused_moe_kernel(; -440,11 +442,9 @@ def fused_moe_kernel(; symbols: fused_moe_kernel, invoke_fused_moe_kernel
  - `python/sglang/srt/models/qwen2_moe.py` modified +22/-0 (22 lines); hunks: -112,6 +112,8; -468,11 +470,31 @@ def forward_normal_dual_stream(; symbols: forward_normal_dual_stream, forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/topk.py
@@ -113,6 +113,8 @@ def routing(
+_SGLANG_EXPERIMENTAL_LORA_OPTI = envs.SGLANG_EXPERIMENTAL_LORA_OPTI.get()
@@ -220,6 +222,13 @@ class TopKOutputChecker:
+        # ===== TO BE REFACTORED ====
+        # The experimental fused topk+pack carrier only exists under the master switch.
+        if _SGLANG_EXPERIMENTAL_LORA_OPTI:
+            return isinstance(
diff -- python/sglang/srt/layers/moe/moe_runner/triton_utils/moe_align_block_size.py
@@ -5,8 +5,11 @@
+from sglang.srt.environ import envs
+_SGLANG_EXPERIMENTAL_LORA_OPTI = envs.SGLANG_EXPERIMENTAL_LORA_OPTI.get()
@@ -74,14 +77,37 @@ def moe_align_block_size(
-    sgl_moe_align_block_size(
-        topk_ids,
-        num_experts + 1,
diff -- python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py
@@ -5,6 +5,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/topk.py` modified +131/-0; `python/sglang/srt/layers/moe/moe_runner/triton_utils/moe_align_block_size.py` modified +36/-10; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +37/-2; `python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_kernels.py` modified +33/-5; `python/sglang/srt/models/qwen2_moe.py` modified +22/-0; `python/sglang/srt/layers/moe/utils.py` modified +10/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/jit_kernel/csrc/trtllm_lora_temp/kimi_k2_moe_fused_gate.cuh`, `python/sglang/jit_kernel/csrc/trtllm_lora_temp/moe_lora_merged_align_kernel.cu`, `python/sglang/jit_kernel/csrc/trtllm_lora_temp/topk_softmax_pack.cuh`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27150 - Support Waterfill with dynamic EPLB

- Link: https://github.com/sgl-project/sglang/pull/27150
- Status/date: merged / 2026-06-05
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +159/-5, 220 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support Waterfill with dynamic EPLB"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/models/deepseek_v2.py`, `test/registered/unit/eplb/test_deepep_waterfill_eplb.py`; technical summary: Covers "Support Waterfill with dynamic EPLB"; the main implementation surface is `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/models/deepseek_v2.py`, `test/registered/unit/eplb/test_deepep_waterfill_eplb.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/moe/topk.py` modified +14/-4 (18 lines); hunks: -1347,7 +1347,7 @@ def _post_process_topk_ids(; -1357,6 +1357,7 @@ def _post_process_topk_ids(; symbols: _post_process_topk_ids, select_experts, touching `_post_process_topk_ids, select_experts`; `python/sglang/srt/models/deepseek_v2.py` modified +7/-1 (8 lines); hunks: -796,8 +796,14 @@ def __init__(; symbols: __init__, get_moe_weights, touching `__init__, get_moe_weights`; `test/registered/unit/eplb/test_deepep_waterfill_eplb.py` added +138/-0 (138 lines); hunks: -0,0 +1,138; symbols: _FakeExpertParam, __init__, TestDeepEPWaterfillEPLB, test_deepseek_moe_get_moe_weights_excludes_fused_shared_slot, touching `_FakeExpertParam, __init__, TestDeepEPWaterfillEPLB`.
- Code diff details:
  - `python/sglang/srt/layers/moe/topk.py` modified +14/-4 (18 lines); hunks: -1347,7 +1347,7 @@ def _post_process_topk_ids(; -1357,6 +1357,7 @@ def _post_process_topk_ids(; symbols: _post_process_topk_ids, select_experts
  - `python/sglang/srt/models/deepseek_v2.py` modified +7/-1 (8 lines); hunks: -796,8 +796,14 @@ def __init__(; symbols: __init__, get_moe_weights
  - `test/registered/unit/eplb/test_deepep_waterfill_eplb.py` added +138/-0 (138 lines); hunks: -0,0 +1,138; symbols: _FakeExpertParam, __init__, TestDeepEPWaterfillEPLB, test_deepseek_moe_get_moe_weights_excludes_fused_shared_slot
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/topk.py
@@ -1347,7 +1347,7 @@ def _post_process_topk_ids(
-) -> torch.Tensor:
+) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
@@ -1357,6 +1357,7 @@ def _post_process_topk_ids(
+    recorder_topk_ids = None
@@ -1369,11 +1370,18 @@ def _post_process_topk_ids(
+            # ExpertDistributionRecorder tracks EPLB physical routed experts.
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -796,8 +796,14 @@ def __init__(
+        # EPLB only rebalances physical routed experts. Fused shared expert
+        # slots live after each rank's routed slots and must stay stable.
+        num_local_experts_for_eplb = (
+            self.experts.num_local_experts - self.num_fused_shared_experts
+        )
-            x.data
diff -- test/registered/unit/eplb/test_deepep_waterfill_eplb.py
@@ -0,0 +1,138 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/topk.py` modified +14/-4; `python/sglang/srt/models/deepseek_v2.py` modified +7/-1
  - tests: `test/registered/unit/eplb/test_deepep_waterfill_eplb.py` added +138/-0
- Risk and verification: The diff ships test coverage in `test/registered/unit/eplb/test_deepep_waterfill_eplb.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #27114 - [Bugfix] Restore overridden HF config fields and support index_skip_topk_offset for DSA topk sharing

- Link: https://github.com/sgl-project/sglang/pull/27114
- Status/date: merged / 2026-06-06
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +100/-5, 186 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Restore overridden HF config fields and support index_skip_topk_offset for DSA topk sharing"; model line: DeepSeek V3.1; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/models/deepseek_nextn.py`; technical summary: Covers "[Bugfix] Restore overridden HF config fields and support index_skip_topk_offset for DSA topk sharing"; the main implementation surface is `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/models/deepseek_nextn.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +25/-3 (28 lines); hunks: -1463,6 +1463,7 @@ def __init__(; -1550,12 +1551,33 @@ def __init__(; symbols: __init__, touching `__init__`; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +15/-2 (17 lines); hunks: -242,7 +242,15 @@ def forward_absorb_prepare(; -261,7 +269,12 @@ def forward_absorb_prepare(; symbols: forward_absorb_prepare, touching `forward_absorb_prepare`; `python/sglang/srt/models/deepseek_nextn.py` modified +7/-0 (7 lines); hunks: -230,7 +230,14 @@ def forward(; symbols: forward, touching `forward`; `python/sglang/srt/model_executor/forward_batch_info.py` modified +4/-0 (4 lines); hunks: -385,6 +385,10 @@ class ForwardBatch(ForwardBatchDeepSeekMHAMixin):; symbols: ForwardBatch, touching `ForwardBatch`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +25/-3 (28 lines); hunks: -1463,6 +1463,7 @@ def __init__(; -1550,12 +1551,33 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +15/-2 (17 lines); hunks: -242,7 +242,15 @@ def forward_absorb_prepare(; -261,7 +269,12 @@ def forward_absorb_prepare(; symbols: forward_absorb_prepare
  - `python/sglang/srt/models/deepseek_nextn.py` modified +7/-0 (7 lines); hunks: -230,7 +230,14 @@ def forward(; symbols: forward
  - `python/sglang/srt/model_executor/forward_batch_info.py` modified +4/-0 (4 lines); hunks: -385,6 +385,10 @@ class ForwardBatch(ForwardBatchDeepSeekMHAMixin):; symbols: ForwardBatch
  - `python/sglang/srt/utils/hf_transformers/config.py` modified +20/-0 (20 lines); hunks: -67,6 +67,26 @@ def parse(; symbols: parse
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -1463,6 +1463,7 @@ def __init__(
+        self.is_nextn = is_nextn
@@ -1550,12 +1551,33 @@ def __init__(
-                self.skip_topk = False
-                self.next_skip_topk = False
+                self.skip_topk = True
+                self.next_skip_topk = True
diff -- python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py
@@ -242,7 +242,15 @@ def forward_absorb_prepare(
-                if not self.skip_topk or prev_topk_indices is None:
+                # skip_topk (shared) layers carry no indexer weights in the
+                # checkpoint, so they must reuse the carried topk and never run
+                # the indexer. Do NOT widen this to `or prev_topk_indices is
+                # None` (the upstream gate): that recomputes with an
+                # uninitialized indexer whenever cross-layer propagation is
diff -- python/sglang/srt/models/deepseek_nextn.py
@@ -230,7 +230,14 @@ def forward(
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +25/-3; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +15/-2; `python/sglang/srt/models/deepseek_nextn.py` modified +7/-0; `python/sglang/srt/model_executor/forward_batch_info.py` modified +4/-0; `python/sglang/srt/utils/hf_transformers/config.py` modified +20/-0; `python/sglang/srt/server_args.py` modified +14/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/batch_overlap/two_batch_overlap.py`, `python/sglang/srt/model_executor/forward_batch_info.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27289 - [ROCm] dsv4: remove the redundant fp8 scale transpose-copy on decode

- Link: https://github.com/sgl-project/sglang/pull/27289
- Status/date: merged / 2026-06-08
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +20/-3, 142 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm] dsv4: remove the redundant fp8 scale transpose-copy on decode"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/quantization/fp8_utils.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`, `python/sglang/srt/layers/communicator.py`; technical summary: Covers "[ROCm] dsv4: remove the redundant fp8 scale transpose-copy on decode"; the main implementation surface is `python/sglang/srt/layers/quantization/fp8_utils.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`, `python/sglang/srt/layers/communicator.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/quantization/fp8_utils.py` modified +4/-2 (6 lines); hunks: -786,8 +786,10 @@ def aiter_w8a8_block_fp8_linear(; symbols: aiter_w8a8_block_fp8_linear, touching `aiter_w8a8_block_fp8_linear`; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` modified +4/-0 (4 lines); hunks: -19,6 +19,7; -152,6 +153,7 @@ def forward_normal_prepare(; symbols: forward_normal_prepare, touching `forward_normal_prepare`; `python/sglang/srt/layers/communicator.py` modified +3/-0 (3 lines); hunks: -65,6 +65,7; -572,6 +573,7 @@ def prepare_attn(; symbols: prepare_attn, touching `prepare_attn`; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +3/-0 (3 lines); hunks: -38,6 +38,7; -197,6 +198,7 @@ def forward_absorb_prepare(; symbols: forward_absorb_prepare, touching `forward_absorb_prepare`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/fp8_utils.py` modified +4/-2 (6 lines); hunks: -786,8 +786,10 @@ def aiter_w8a8_block_fp8_linear(; symbols: aiter_w8a8_block_fp8_linear
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` modified +4/-0 (4 lines); hunks: -19,6 +19,7; -152,6 +153,7 @@ def forward_normal_prepare(; symbols: forward_normal_prepare
  - `python/sglang/srt/layers/communicator.py` modified +3/-0 (3 lines); hunks: -65,6 +65,7; -572,6 +573,7 @@ def prepare_attn(; symbols: prepare_attn
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +3/-0 (3 lines); hunks: -38,6 +38,7; -197,6 +198,7 @@ def forward_absorb_prepare(; symbols: forward_absorb_prepare
  - `python/sglang/srt/models/deepseek_v2.py` modified +2/-1 (3 lines); hunks: -162,6 +162,7; -376,7 +377,7 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/fp8_utils.py
@@ -786,8 +786,10 @@ def aiter_w8a8_block_fp8_linear(
-        if _use_aiter_bpreshuffle_gfx95 and not use_triton:
-            x_scale = x_scale.transpose(-1, -2).contiguous().view(*x_scale.shape)
+        # On ROCm >= 7.2, scale is in bpreshuffle's transposed layout.
+        # Triton needs a row-major view, so adjust strides only. No copy.
+        if use_triton and _use_aiter_bpreshuffle_gfx95:
+            x_scale = torch.as_strided(x_scale, x_scale.shape, (1, x_scale.shape[0]))
diff -- python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py
@@ -19,6 +19,7 @@
+    _use_aiter_bpreshuffle_gfx95,
@@ -152,6 +153,7 @@ def forward_normal_prepare(
+                        transpose_scale=_use_aiter_bpreshuffle_gfx95,
@@ -193,6 +195,7 @@ def forward_normal_prepare(
+                    transpose_scale=_use_aiter_bpreshuffle_gfx95,
@@ -222,6 +225,7 @@ def forward_normal_prepare(
diff -- python/sglang/srt/layers/communicator.py
@@ -65,6 +65,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/fp8_utils.py` modified +4/-2; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` modified +4/-0; `python/sglang/srt/layers/communicator.py` modified +3/-0; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +3/-0; `python/sglang/srt/models/deepseek_v2.py` modified +2/-1; `python/sglang/srt/models/deepseek_common/utils.py` modified +2/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/layers/quantization/fp8_utils.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23906 - [Refactor] Cuda Graph Runner/Backend Refactor

- Link: https://github.com/sgl-project/sglang/pull/23906
- Status/date: merged / 2026-06-10
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 160 files, +5197/-3068, 12233 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Refactor] Cuda Graph Runner/Backend Refactor"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py`; technical summary: Covers "[Refactor] Cuda Graph Runner/Backend Refactor"; the main implementation surface is `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #27510 - [deepseek] Enable DP attention + TBO + shared experts fusion

- Link: https://github.com/sgl-project/sglang/pull/27510
- Status/date: merged / 2026-06-10
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +79/-8, 109 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[deepseek] Enable DP attention + TBO + shared experts fusion"; model line: DeepSeek V3.1; category: docs/tests/CI; main diff: `python/sglang/srt/models/deepseek_v2.py`, `test/registered/ep/test_tbo_shared_experts_fusion.py`, `python/sglang/srt/batch_overlap/two_batch_overlap.py`; technical summary: Covers "[deepseek] Enable DP attention + TBO + shared experts fusion"; the main implementation surface is `python/sglang/srt/models/deepseek_v2.py`, `test/registered/ep/test_tbo_shared_experts_fusion.py`, `python/sglang/srt/batch_overlap/two_batch_overlap.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +1/-3 (4 lines); hunks: -1333,9 +1333,7 @@ def _forward_shared_experts(; symbols: _forward_shared_experts, op_gate, touching `_forward_shared_experts, op_gate`; `test/registered/ep/test_tbo_shared_experts_fusion.py` added +73/-0 (73 lines); hunks: -0,0 +1,73; symbols: TestTBOWithSharedExpertsFusion, setUpClass, tearDownClass, test_gsm8k, touching `TestTBOWithSharedExpertsFusion, setUpClass, tearDownClass`; `python/sglang/srt/batch_overlap/two_batch_overlap.py` modified +5/-0 (5 lines); hunks: -672,6 +672,11 @@ def filter_batch(; symbols: filter_batch, touching `filter_batch`; `python/sglang/srt/server_args.py` modified +0/-5 (5 lines); hunks: -7492,11 +7492,6 @@ def check_server_args(self):; symbols: check_server_args, touching `check_server_args`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +1/-3 (4 lines); hunks: -1333,9 +1333,7 @@ def _forward_shared_experts(; symbols: _forward_shared_experts, op_gate
  - `test/registered/ep/test_tbo_shared_experts_fusion.py` added +73/-0 (73 lines); hunks: -0,0 +1,73; symbols: TestTBOWithSharedExpertsFusion, setUpClass, tearDownClass, test_gsm8k
  - `python/sglang/srt/batch_overlap/two_batch_overlap.py` modified +5/-0 (5 lines); hunks: -672,6 +672,11 @@ def filter_batch(; symbols: filter_batch
  - `python/sglang/srt/server_args.py` modified +0/-5 (5 lines); hunks: -7492,11 +7492,6 @@ def check_server_args(self):; symbols: check_server_args
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -1333,9 +1333,7 @@ def _forward_shared_experts(
-        if is_non_idle_and_non_empty(
-            state.forward_batch.forward_mode, state.hidden_states_mlp_input
-        ):
+        if state.hidden_states_mlp_input.shape[0] > 0:
diff -- test/registered/ep/test_tbo_shared_experts_fusion.py
@@ -0,0 +1,73 @@
+import os
+import unittest
+from types import SimpleNamespace
+from sglang.srt.utils import kill_process_tree
+from sglang.test.ci.ci_register import register_cuda_ci
+from sglang.test.run_eval import run_eval
diff -- python/sglang/srt/batch_overlap/two_batch_overlap.py
@@ -672,6 +672,11 @@ def filter_batch(
+            elif key == "rids" and len(old_value) != num_seqs:
+                output_dict[key] = old_value[
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +1/-3; `python/sglang/srt/batch_overlap/two_batch_overlap.py` modified +5/-0; `python/sglang/srt/server_args.py` modified +0/-5
  - tests: `test/registered/ep/test_tbo_shared_experts_fusion.py` added +73/-0
- Risk and verification: The diff ships test coverage in `test/registered/ep/test_tbo_shared_experts_fusion.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #27956 - Use the correct wrapper for `fp4_quantize`

- Link: https://github.com/sgl-project/sglang/pull/27956
- Status/date: merged / 2026-06-12
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-2, 10 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Use the correct wrapper for `fp4_quantize`"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v2.py`; technical summary: Covers "Use the correct wrapper for `fp4_quantize`"; the main implementation surface is `python/sglang/srt/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +1/-2 (3 lines); hunks: -286,8 +286,7 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +1/-2 (3 lines); hunks: -286,8 +286,7 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -286,8 +286,7 @@ def forward(
-            from flashinfer import fp4_quantize
+            from sglang.srt.layers.quantization.fp4_utils import fp4_quantize
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +1/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27720 - [DeepSeek V3] Defer moe finalize and fused it with main stream add

- Link: https://github.com/sgl-project/sglang/pull/27720
- Status/date: merged / 2026-06-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +743/-36, 865 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DeepSeek V3] Defer moe finalize and fused it with main stream add"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`; technical summary: Covers "[DeepSeek V3] Defer moe finalize and fused it with main stream add"; the main implementation surface is `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py` modified +102/-29 (131 lines); hunks: -1,7 +1,9; -42,6 +44,46; symbols: FlashInferTrtllmDeferredFinalizeOutput, flashinfer_trtllm_deferred_finalize_context, finalize_flashinfer_trtllm_deferred_output, round_up_to_multiple, touching `FlashInferTrtllmDeferredFinalizeOutput, flashinfer_trtllm_deferred_finalize_context, finalize_flashinfer_trtllm_deferred_output`; `python/sglang/srt/models/deepseek_v2.py` modified +28/-7 (35 lines); hunks: -905,7 +905,18 @@ def forward_normal_dual_stream(; -916,12 +927,22 @@ def forward_normal_dual_stream(; symbols: forward_normal_dual_stream, touching `forward_normal_dual_stream`; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +29/-0 (29 lines); hunks: -23,6 +23,7; -287,6 +288,17 @@ def __init__(; symbols: __init__, forward_impl, forward_deferred_finalize, run_moe_core, touching `__init__, forward_impl, forward_deferred_finalize`; `python/sglang/jit_kernel/csrc/moe/moe_finalize_fuse_shared.cu` added +418/-0 (418 lines); hunks: -0,0 +1,418.
- Code diff details:
  - `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py` modified +102/-29 (131 lines); hunks: -1,7 +1,9; -42,6 +44,46; symbols: FlashInferTrtllmDeferredFinalizeOutput, flashinfer_trtllm_deferred_finalize_context, finalize_flashinfer_trtllm_deferred_output, round_up_to_multiple
  - `python/sglang/srt/models/deepseek_v2.py` modified +28/-7 (35 lines); hunks: -905,7 +905,18 @@ def forward_normal_dual_stream(; -916,12 +927,22 @@ def forward_normal_dual_stream(; symbols: forward_normal_dual_stream
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +29/-0 (29 lines); hunks: -23,6 +23,7; -287,6 +288,17 @@ def __init__(; symbols: __init__, forward_impl, forward_deferred_finalize, run_moe_core
  - `python/sglang/jit_kernel/csrc/moe/moe_finalize_fuse_shared.cu` added +418/-0 (418 lines); hunks: -0,0 +1,418
  - `python/sglang/jit_kernel/csrc/moe/tvm_ffi_utils.h` added +105/-0 (105 lines); hunks: -0,0 +1,105
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py
@@ -1,7 +1,9 @@
+import contextvars
+from contextlib import contextmanager
-from typing import TYPE_CHECKING, cast
+from typing import TYPE_CHECKING, Generator, cast
@@ -42,6 +44,46 @@
+_deferred_finalize_enabled: contextvars.ContextVar[bool] = contextvars.ContextVar(
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -905,7 +905,18 @@ def forward_normal_dual_stream(
-            final_hidden_states = self.experts(hidden_states, topk_output)
+            deferred_finalize = (
+                shared_output is not None
+                and not self._shared_expert_tp1
+                and topk_output.format == TopKOutputFormat.BYPASSED
+                and self.experts.supports_deferred_finalize
diff -- python/sglang/srt/layers/moe/fused_moe_triton/layer.py
@@ -23,6 +23,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py` modified +102/-29; `python/sglang/srt/models/deepseek_v2.py` modified +28/-7; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +29/-0; `python/sglang/jit_kernel/csrc/moe/moe_finalize_fuse_shared.cu` added +418/-0; `python/sglang/jit_kernel/csrc/moe/tvm_ffi_utils.h` added +105/-0; `python/sglang/jit_kernel/moe_finalize_fuse_shared.py` added +60/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/jit_kernel/csrc/moe/moe_finalize_fuse_shared.cu`, `python/sglang/jit_kernel/csrc/moe/tvm_ffi_utils.h`, `python/sglang/jit_kernel/moe_finalize_fuse_shared.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #28129 - [Spec] Remove deprecated EAGLE v1 DRAFT_EXTEND forward mode

- Link: https://github.com/sgl-project/sglang/pull/28129
- Status/date: merged / 2026-06-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 56 files, +219/-2555, 3937 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Spec] Remove deprecated EAGLE v1 DRAFT_EXTEND forward mode"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/model_executor/forward_batch_info.py`, `python/sglang/srt/layers/attention/triton_backend.py`; technical summary: Covers "[Spec] Remove deprecated EAGLE v1 DRAFT_EXTEND forward mode"; the main implementation surface is `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/model_executor/forward_batch_info.py`, `python/sglang/srt/layers/attention/triton_backend.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/aiter_backend.py` modified +2/-157 (159 lines); hunks: -1117,88 +1117,6 @@ def init_forward_metadata(self, forward_batch: ForwardBat...; -1941,72 +1859,6 @@ def _apply_cuda_graph_metadata(; symbols: init_forward_metadata, _apply_cuda_graph_metadata, forward_extend, touching `init_forward_metadata, _apply_cuda_graph_metadata, forward_extend`; `python/sglang/srt/model_executor/forward_batch_info.py` modified +19/-60 (79 lines); hunks: -94,8 +94,6 @@ class ForwardMode(IntEnum):; -115,7 +113,6 @@ def is_extend(self, include_draft_extend_v2: bool = False):; symbols: ForwardMode, is_extend, is_decode_or_idle, is_target_verify, touching `ForwardMode, is_extend, is_decode_or_idle`; `python/sglang/srt/layers/attention/triton_backend.py` modified +20/-52 (72 lines); hunks: -454,31 +454,25 @@ def _update_draft_extend_buffers(; -693,32 +687,6 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; symbols: _update_draft_extend_buffers, init_forward_metadata, _build_cuda_graph_forward_metadata, _apply_cuda_graph_metadata, touching `_update_draft_extend_buffers, init_forward_metadata, _build_cuda_graph_forward_metadata`; `python/sglang/srt/layers/attention/flashattention_backend.py` modified +7/-40 (47 lines); hunks: -352,7 +352,7 @@ def init_forward_metadata_out_graph(; -645,9 +645,10 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; symbols: init_forward_metadata_out_graph, init_forward_metadata, _fa_cp_attn, _bind_metadata_buffers, touching `init_forward_metadata_out_graph, init_forward_metadata, _fa_cp_attn`.
- Code diff details:
  - `python/sglang/srt/layers/attention/aiter_backend.py` modified +2/-157 (159 lines); hunks: -1117,88 +1117,6 @@ def init_forward_metadata(self, forward_batch: ForwardBat...; -1941,72 +1859,6 @@ def _apply_cuda_graph_metadata(; symbols: init_forward_metadata, _apply_cuda_graph_metadata, forward_extend
  - `python/sglang/srt/model_executor/forward_batch_info.py` modified +19/-60 (79 lines); hunks: -94,8 +94,6 @@ class ForwardMode(IntEnum):; -115,7 +113,6 @@ def is_extend(self, include_draft_extend_v2: bool = False):; symbols: ForwardMode, is_extend, is_decode_or_idle, is_target_verify
  - `python/sglang/srt/layers/attention/triton_backend.py` modified +20/-52 (72 lines); hunks: -454,31 +454,25 @@ def _update_draft_extend_buffers(; -693,32 +687,6 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; symbols: _update_draft_extend_buffers, init_forward_metadata, _build_cuda_graph_forward_metadata, _apply_cuda_graph_metadata
  - `python/sglang/srt/layers/attention/flashattention_backend.py` modified +7/-40 (47 lines); hunks: -352,7 +352,7 @@ def init_forward_metadata_out_graph(; -645,9 +645,10 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; symbols: init_forward_metadata_out_graph, init_forward_metadata, _fa_cp_attn, _bind_metadata_buffers
  - `python/sglang/srt/layers/attention/dsa_backend.py` modified +11/-33 (44 lines); hunks: -406,9 +406,7 @@ def _build_paged_mqa_schedule_2d_ctx_lens(; -488,7 +486,7 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; symbols: _build_paged_mqa_schedule_2d_ctx_lens, init_forward_metadata
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/aiter_backend.py
@@ -1117,88 +1117,6 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):
-        elif forward_batch.forward_mode.is_draft_extend():
-            # EAGLE V1: DRAFT_EXTEND mode - uses spec_info.num_accept_tokens
-            if self.use_mla:
-                kv_indices, kv_indptr, qo_indptr, custom_mask = (
-                    spec_info.generate_attn_arg_prefill(
-                        forward_batch.req_pool_indices,
diff -- python/sglang/srt/model_executor/forward_batch_info.py
@@ -94,8 +94,6 @@ class ForwardMode(IntEnum):
-    DRAFT_EXTEND = auto()
@@ -115,7 +113,6 @@ def is_extend(self, include_draft_extend_v2: bool = False):
-            or self == ForwardMode.DRAFT_EXTEND
@@ -148,19 +145,13 @@ def is_decode_or_idle(self):
-    def is_draft_extend(self, include_v2: bool = False):
-        return self == ForwardMode.DRAFT_EXTEND or (
diff -- python/sglang/srt/layers/attention/triton_backend.py
@@ -454,31 +454,25 @@ def _update_draft_extend_buffers(
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/aiter_backend.py` modified +2/-157; `python/sglang/srt/model_executor/forward_batch_info.py` modified +19/-60; `python/sglang/srt/layers/attention/triton_backend.py` modified +20/-52; `python/sglang/srt/layers/attention/flashattention_backend.py` modified +7/-40; `python/sglang/srt/layers/attention/dsa_backend.py` modified +11/-33; `python/sglang/srt/layers/attention/flashinfer_backend.py` modified +2/-24
- Risk and verification: The diff ships test coverage in `python/sglang/test/kits/attention_unittest/attention_methods/dense_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dsa_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dsv4_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/gdn_attention.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #28118 - 【bugfix】The NPU's forward_dsa_prepare_npu also needs special handling for is_nextn

- Link: https://github.com/sgl-project/sglang/pull/28118
- Status/date: merged / 2026-06-15
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-3, 20 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "【bugfix】The NPU's forward_dsa_prepare_npu also needs special handling for is_nextn"; model line: DeepSeek V3.1; category: bug fix; main diff: `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`; technical summary: Covers "【bugfix】The NPU's forward_dsa_prepare_npu also needs special handling for is_nextn"; the main implementation surface is `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` modified +3/-3 (6 lines); hunks: -403,9 +403,7 @@ def forward_dsa_prepare_npu(; -415,6 +413,8 @@ def forward_dsa_prepare_npu(; symbols: forward_dsa_prepare_npu, touching `forward_dsa_prepare_npu`.
- Code diff details:
  - `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` modified +3/-3 (6 lines); hunks: -403,9 +403,7 @@ def forward_dsa_prepare_npu(; -415,6 +413,8 @@ def forward_dsa_prepare_npu(; symbols: forward_dsa_prepare_npu
- Key code excerpts:

```diff
diff -- python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py
@@ -403,9 +403,7 @@ def forward_dsa_prepare_npu(
-    if m.skip_topk:
-        topk_indices = prev_topk_indices
-    else:
+    if not m.skip_topk or (m.is_nextn and prev_topk_indices is None):
@@ -415,6 +413,8 @@ def forward_dsa_prepare_npu(
+    else:
```

- Reviewed files:
  - runtime: `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` modified +3/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #24515 - LPLB: linear-programming load balancer for MoE expert parallelism

- Link: https://github.com/sgl-project/sglang/pull/24515
- Status/date: merged / 2026-06-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 19 files, +2324/-14, 2482 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "LPLB: linear-programming load balancer for MoE expert parallelism"; model line: DeepSeek V3.1; category: model implementation change; main diff: `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/layers/moe/hash_topk.py`; technical summary: Covers "LPLB: linear-programming load balancer for MoE expert parallelism"; the main implementation surface is `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/layers/moe/hash_topk.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/moe/topk.py` modified +47/-6 (53 lines); hunks: -552,7 +552,30 @@ def forward_npu(; -1502,11 +1525,29 @@ def _post_process_topk_ids(; symbols: forward_npu, empty_topk_output, _post_process_topk_ids, touching `forward_npu, empty_topk_output, _post_process_topk_ids`; `python/sglang/srt/model_executor/model_runner.py` modified +42/-0 (42 lines); hunks: -106,6 +106,12; -691,6 +697,9 @@ def initialize(self):; symbols: initialize, _prepare_moe_topk, _init_lplb_solvers, update_expert_location, touching `initialize, _prepare_moe_topk, _init_lplb_solvers`; `python/sglang/srt/layers/moe/hash_topk.py` modified +30/-3 (33 lines); hunks: -34,9 +34,10 @@ def __init__(; -80,8 +81,18 @@ def _init_default_tid2eid(self) -> None:; symbols: __init__, _init_default_tid2eid, empty_topk_output, forward, touching `__init__, _init_default_tid2eid, empty_topk_output`; `python/sglang/srt/models/deepseek_v2.py` modified +20/-3 (23 lines); hunks: -631,6 +631,7 @@ def __init__(; -996,7 +997,9 @@ def forward_normal(; symbols: __init__, forward_normal, forward_deepep, op_select_experts, touching `__init__, forward_normal, forward_deepep`.
- Code diff details:
  - `python/sglang/srt/layers/moe/topk.py` modified +47/-6 (53 lines); hunks: -552,7 +552,30 @@ def forward_npu(; -1502,11 +1525,29 @@ def _post_process_topk_ids(; symbols: forward_npu, empty_topk_output, _post_process_topk_ids
  - `python/sglang/srt/model_executor/model_runner.py` modified +42/-0 (42 lines); hunks: -106,6 +106,12; -691,6 +697,9 @@ def initialize(self):; symbols: initialize, _prepare_moe_topk, _init_lplb_solvers, update_expert_location
  - `python/sglang/srt/layers/moe/hash_topk.py` modified +30/-3 (33 lines); hunks: -34,9 +34,10 @@ def __init__(; -80,8 +81,18 @@ def _init_default_tid2eid(self) -> None:; symbols: __init__, _init_default_tid2eid, empty_topk_output, forward
  - `python/sglang/srt/models/deepseek_v2.py` modified +20/-3 (23 lines); hunks: -631,6 +631,7 @@ def __init__(; -996,7 +997,9 @@ def forward_normal(; symbols: __init__, forward_normal, forward_deepep, op_select_experts
  - `test/registered/eplb/test_lplb_distributed.py` added +446/-0 (446 lines); hunks: -0,0 +1,446; symbols: _make_metadata, test_dispatch_probability_matches_torch_reference, test_solve_ipm_matches_torch_reference, test_lplb_distributed_two_rank
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/topk.py
@@ -552,7 +552,30 @@ def forward_npu(
-    def empty_topk_output(self, device: torch.device) -> TopKOutput:
+    def empty_topk_output(
+        self, device: torch.device, *, layer_id: Optional[int] = None
+    ) -> TopKOutput:
+        """Return an empty topk output for a rank with zero tokens this forward.
+        When ``layer_id`` is provided and the active dispatch algorithm is LP,
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -106,6 +106,12 @@
+from sglang.srt.eplb.lplb_solver import (
+    LPLBSolver,
+    assert_lplb_supported_model,
+    clear_global_lplb_solvers,
+    set_global_lplb_solver,
+)
diff -- python/sglang/srt/layers/moe/hash_topk.py
@@ -34,9 +34,10 @@ def __init__(
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/topk.py` modified +47/-6; `python/sglang/srt/model_executor/model_runner.py` modified +42/-0; `python/sglang/srt/layers/moe/hash_topk.py` modified +30/-3; `python/sglang/srt/models/deepseek_v2.py` modified +20/-3; `python/sglang/jit_kernel/lplb/cuda_solver.py` added +324/-0; `python/sglang/srt/eplb/lplb_solver.py` added +280/-0
  - tests: `test/registered/eplb/test_lplb_distributed.py` added +446/-0
- Risk and verification: The diff ships test coverage in `test/registered/eplb/test_lplb_distributed.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #28436 - [NPU] Use use_dsa to dispatch Ascend DSA attention

- Link: https://github.com/sgl-project/sglang/pull/28436
- Status/date: merged / 2026-06-17
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-2, 15 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NPU] Use use_dsa to dispatch Ascend DSA attention"; model line: DeepSeek V3.1; category: model implementation change; main diff: `python/sglang/srt/models/deepseek_common/attention_backend_handler.py`; technical summary: Covers "[NPU] Use use_dsa to dispatch Ascend DSA attention"; the main implementation surface is `python/sglang/srt/models/deepseek_common/attention_backend_handler.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_common/attention_backend_handler.py` modified +2/-2 (4 lines); hunks: -45,12 +45,12 @@ def handle_attention_ascend(attn, forward_batch):; symbols: handle_attention_ascend, touching `handle_attention_ascend`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_common/attention_backend_handler.py` modified +2/-2 (4 lines); hunks: -45,12 +45,12 @@ def handle_attention_ascend(attn, forward_batch):; symbols: handle_attention_ascend
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_common/attention_backend_handler.py
@@ -45,12 +45,12 @@ def handle_attention_ascend(attn, forward_batch):
-        if hasattr(attn, "indexer"):
+        if hasattr(attn, "use_dsa") and attn.use_dsa:
-        if hasattr(attn, "indexer"):
+        if hasattr(attn, "use_dsa") and attn.use_dsa:
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_common/attention_backend_handler.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_common/attention_backend_handler.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27798 - [AMD] Add transpose_scale arg for o_proj to fix GLM accuracy issue

- Link: https://github.com/sgl-project/sglang/pull/27798
- Status/date: merged / 2026-06-17
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +8/-2, 24 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Add transpose_scale arg for o_proj to fix GLM accuracy issue"; model line: DeepSeek V3.1; category: bug fix; main diff: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`; technical summary: Covers "[AMD] Add transpose_scale arg for o_proj to fix GLM accuracy issue"; the main implementation surface is `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +8/-2 (10 lines); hunks: -664,7 +664,10 @@ def forward_absorb_core(; -674,7 +677,10 @@ def forward_absorb_core(; symbols: forward_absorb_core, touching `forward_absorb_core`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +8/-2 (10 lines); hunks: -664,7 +664,10 @@ def forward_absorb_core(; -674,7 +677,10 @@ def forward_absorb_core(; symbols: forward_absorb_core
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py
@@ -664,7 +664,10 @@ def forward_absorb_core(
-                        _bmm_buf, group_size=128, dtype_quant=torch.float8_e4m3fn
+                        _bmm_buf,
+                        group_size=128,
+                        dtype_quant=torch.float8_e4m3fn,
+                        transpose_scale=_use_aiter_bpreshuffle_gfx95,
@@ -674,7 +677,10 @@ def forward_absorb_core(
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +8/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #28343 - [Kimi K2.5] Fix eagle3 aux capture for tp>1 when AR fusion is enabled

- Link: https://github.com/sgl-project/sglang/pull/28343
- Status/date: merged / 2026-06-17
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +43/-16, 112 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Kimi K2.5] Fix eagle3 aux capture for tp>1 when AR fusion is enabled"; model line: DeepSeek V3.1; category: bug fix; main diff: `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/models/deepseek_v2.py`; technical summary: Covers "[Kimi K2.5] Fix eagle3 aux capture for tp>1 when AR fusion is enabled"; the main implementation surface is `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/communicator.py` modified +31/-2 (33 lines); hunks: -498,11 +498,13 @@ def prepare_attn_and_capture_last_layer_outputs(; -511,12 +513,39 @@ def prepare_attn_and_capture_last_layer_outputs(; symbols: prepare_attn_and_capture_last_layer_outputs, _post_attn_residual_is_read_only, prepare_attn, touching `prepare_attn_and_capture_last_layer_outputs, _post_attn_residual_is_read_only, prepare_attn`; `python/sglang/srt/models/deepseek_v2.py` modified +12/-14 (26 lines); hunks: -75,7 +75,6; -2113,13 +2112,17 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/layers/communicator.py` modified +31/-2 (33 lines); hunks: -498,11 +498,13 @@ def prepare_attn_and_capture_last_layer_outputs(; -511,12 +513,39 @@ def prepare_attn_and_capture_last_layer_outputs(; symbols: prepare_attn_and_capture_last_layer_outputs, _post_attn_residual_is_read_only, prepare_attn
  - `python/sglang/srt/models/deepseek_v2.py` modified +12/-14 (26 lines); hunks: -75,7 +75,6; -2113,13 +2112,17 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/communicator.py
@@ -498,11 +498,13 @@ def prepare_attn_and_capture_last_layer_outputs(
+        quant_format: str = "",
+            quant_format=quant_format,
@@ -511,12 +513,39 @@ def prepare_attn_and_capture_last_layer_outputs(
-            if gathered_last_layer_output is residual:
-                # Clone to avoid modifying the original residual by Custom RMSNorm inplace operation
+            if (
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -75,7 +75,6 @@
-    get_attention_tp_group,
@@ -2113,13 +2112,17 @@ def forward(
+        captured_last_layer_outputs: Optional[List[torch.Tensor]] = None,
-        hidden_states, residual = self.layer_communicator.prepare_attn(
-            hidden_states,
-            residual,
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/communicator.py` modified +31/-2; `python/sglang/srt/models/deepseek_v2.py` modified +12/-14
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #28567 - Add get_parallel(): a structured accessor for parallel-topology state

- Link: https://github.com/sgl-project/sglang/pull/28567
- Status/date: merged / 2026-06-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 184 files, +1865/-1727, 8932 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add get_parallel(): a structured accessor for parallel-topology state"; model line: DeepSeek V3.1; category: model support/runtime entry; main diff: `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py`; technical summary: Covers "Add get_parallel(): a structured accessor for parallel-topology state"; the main implementation surface is `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #25144 - [NPU] Add Ascend NPU support for DeepSeek-V4

- Link: https://github.com/sgl-project/sglang/pull/25144
- Status/date: merged / 2026-06-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 28 files, +4145/-144, 4984 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NPU] Add Ascend NPU support for DeepSeek-V4"; model line: DeepSeek V3.1; category: model support/runtime entry; main diff: `python/sglang/srt/layers/deepseek_v4_rope.py`, `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/layers/mhc.py`; technical summary: Covers "[NPU] Add Ascend NPU support for DeepSeek-V4"; the main implementation surface is `python/sglang/srt/layers/deepseek_v4_rope.py`, `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/layers/mhc.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/deepseek_v4_rope.py` modified +148/-2 (150 lines); hunks: -1,3 +1,4; -6,26 +7,51; symbols: _yarn_get_mscale, precompute_freqs_cis, find_correction_dim, fused_norm_rope_inplace_triton, touching `_yarn_get_mscale, precompute_freqs_cis, find_correction_dim`; `python/sglang/srt/models/deepseek_v4.py` modified +103/-24 (127 lines); hunks: -29,6 +29,7; -47,10 +48,15; symbols: __init__, _forward_prepare, touching `__init__, _forward_prepare`; `python/sglang/srt/layers/mhc.py` modified +104/-9 (113 lines); hunks: -3,8 +3,6; -14,15 +12,55; symbols: _TilelangMissing, __getattr__, _jit, _wrap, touching `_TilelangMissing, __getattr__, _jit`; `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` modified +68/-6 (74 lines); hunks: -389,7 +389,17 @@ def _init_pools(self: ModelRunner):; -410,7 +420,8 @@ def _init_pools(self: ModelRunner):; symbols: _init_pools, touching `_init_pools`.
- Code diff details:
  - `python/sglang/srt/layers/deepseek_v4_rope.py` modified +148/-2 (150 lines); hunks: -1,3 +1,4; -6,26 +7,51; symbols: _yarn_get_mscale, precompute_freqs_cis, find_correction_dim, fused_norm_rope_inplace_triton
  - `python/sglang/srt/models/deepseek_v4.py` modified +103/-24 (127 lines); hunks: -29,6 +29,7; -47,10 +48,15; symbols: __init__, _forward_prepare
  - `python/sglang/srt/layers/mhc.py` modified +104/-9 (113 lines); hunks: -3,8 +3,6; -14,15 +12,55; symbols: _TilelangMissing, __getattr__, _jit, _wrap
  - `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` modified +68/-6 (74 lines); hunks: -389,7 +389,17 @@ def _init_pools(self: ModelRunner):; -410,7 +420,8 @@ def _init_pools(self: ModelRunner):; symbols: _init_pools
  - `python/sglang/srt/model_executor/forward_batch_info.py` modified +63/-0 (63 lines); hunks: -217,6 +217,65 @@ def compute_local_num_token_non_padded(; -286,6 +345,9 @@ class ForwardBatch(ForwardBatchDeepSeekMHAMixin):; symbols: compute_local_num_token_non_padded, DSV4OutCacheLoc, DSV4StateLens, NgramEmbeddingInfo
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/deepseek_v4_rope.py
@@ -1,3 +1,4 @@
+import logging
@@ -6,26 +7,51 @@
+logger = logging.getLogger(__name__)
+# tilelang isn't shipped on every platform (e.g. Ascend NPU images) and the
+# only tilelang artifacts in this file are pass_configs that downstream
+# tilelang.jit decorators would consume — the kernels actually defined here
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -29,6 +29,7 @@
+    get_tensor_model_parallel_world_size,
@@ -47,10 +48,15 @@
+from sglang.srt.layers.deepseek_v4_rope import (
+    v4_rope_inplace_npu,
+)
+    attn_tp_all_reduce,
diff -- python/sglang/srt/layers/mhc.py
@@ -3,8 +3,6 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/deepseek_v4_rope.py` modified +148/-2; `python/sglang/srt/models/deepseek_v4.py` modified +103/-24; `python/sglang/srt/layers/mhc.py` modified +104/-9; `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` modified +68/-6; `python/sglang/srt/model_executor/forward_batch_info.py` modified +63/-0; `python/sglang/srt/layers/attention/dsv4/compressor.py` modified +31/-4
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/arg_groups/deepseek_v4_hook.py`, `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #28559 - fix: speculative draft worker clobbering target attention backend

- Link: https://github.com/sgl-project/sglang/pull/28559
- Status/date: merged / 2026-06-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +30/-10, 90 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix: speculative draft worker clobbering target attention backend"; model line: DeepSeek V3.1; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/layers/attention/base_attn_backend.py`; technical summary: Covers "fix: speculative draft worker clobbering target attention backend"; the main implementation surface is `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/layers/attention/base_attn_backend.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +15/-6 (21 lines); hunks: -140,6 +140,7; -1737,20 +1738,28 @@ def __init__(; symbols: __init__, dispatch_attn_forward_method, touching `__init__, dispatch_attn_forward_method`; `python/sglang/srt/model_executor/model_runner.py` modified +11/-4 (15 lines); hunks: -2415,13 +2415,24 @@ def init_attention_backend(self):; -2463,10 +2474,6 @@ def _get_attention_backend(self, init_new_workspace: bool...; symbols: init_attention_backend, _get_attention_backend, _get_attention_backend_from_str, touching `init_attention_backend, _get_attention_backend, _get_attention_backend_from_str`; `python/sglang/srt/layers/attention/base_attn_backend.py` modified +4/-0 (4 lines); hunks: -38,6 +38,10 @@ class AttentionBackend(ABC):; symbols: AttentionBackend, init_forward_metadata, touching `AttentionBackend, init_forward_metadata`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +15/-6 (21 lines); hunks: -140,6 +140,7; -1737,20 +1738,28 @@ def __init__(; symbols: __init__, dispatch_attn_forward_method
  - `python/sglang/srt/model_executor/model_runner.py` modified +11/-4 (15 lines); hunks: -2415,13 +2415,24 @@ def init_attention_backend(self):; -2463,10 +2474,6 @@ def _get_attention_backend(self, init_new_workspace: bool...; symbols: init_attention_backend, _get_attention_backend, _get_attention_backend_from_str
  - `python/sglang/srt/layers/attention/base_attn_backend.py` modified +4/-0 (4 lines); hunks: -38,6 +38,10 @@ class AttentionBackend(ABC):; symbols: AttentionBackend, init_forward_metadata
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -140,6 +140,7 @@
+from sglang.srt.model_executor.forward_context import get_attn_backend
@@ -1737,20 +1738,28 @@ def __init__(
-        # Determine attention backend used by current forward batch
+        # Determine attention backend name for current forward batch: prefer the
+        # name stamped per-runner on the backend object, else resolve from server args.
+        backend = get_attn_backend()
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -2415,13 +2415,24 @@ def init_attention_backend(self):
+        # Record resolved per-mode backends on the backend for model dispatch.
+        self.attn_backend.prefill_attention_backend_str = (
+            self.prefill_attention_backend_str
+        )
+        self.attn_backend.decode_attention_backend_str = (
+            self.decode_attention_backend_str
diff -- python/sglang/srt/layers/attention/base_attn_backend.py
@@ -38,6 +38,10 @@ class AttentionBackend(ABC):
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +15/-6; `python/sglang/srt/model_executor/model_runner.py` modified +11/-4; `python/sglang/srt/layers/attention/base_attn_backend.py` modified +4/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/base_attn_backend.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #28532 - Fix IndexCache PP topk handoff

- Link: https://github.com/sgl-project/sglang/pull/28532
- Status/date: merged / 2026-06-19
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +99/-43, 268 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix IndexCache PP topk handoff"; model line: DeepSeek V3.1; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/model_executor/model_runner.py`; technical summary: Covers "Fix IndexCache PP topk handoff"; the main implementation surface is `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/model_executor/model_runner.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +43/-42 (85 lines); hunks: -40,6 +40,7; -1603,40 +1604,8 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`; `python/sglang/srt/configs/model_config.py` modified +23/-0 (23 lines); hunks: -132,6 +132,29 @@ def get_dsa_index_topk(config: PretrainedConfig) -> int:; symbols: get_dsa_index_topk, dsa_layer_skips_topk, get_dsa_index_n_heads, touching `get_dsa_index_topk, dsa_layer_skips_topk, get_dsa_index_n_heads`; `python/sglang/srt/model_executor/model_runner.py` modified +14/-0 (14 lines); hunks: -61,7 +61,9; -832,6 +834,17 @@ def initialize(self):; symbols: initialize, get_pp_proxy_topk_size, alloc_memory_pool, _dummy_run, touching `initialize, get_pp_proxy_topk_size, alloc_memory_pool`; `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` modified +6/-0 (6 lines); hunks: -180,6 +180,7 @@ def _allocate_decode_buffers(; -218,6 +219,10 @@ def _allocate_decode_buffers(; symbols: _allocate_decode_buffers, __init__, touching `_allocate_decode_buffers, __init__`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +43/-42 (85 lines); hunks: -40,6 +40,7; -1603,40 +1604,8 @@ def __init__(; symbols: __init__, forward
  - `python/sglang/srt/configs/model_config.py` modified +23/-0 (23 lines); hunks: -132,6 +132,29 @@ def get_dsa_index_topk(config: PretrainedConfig) -> int:; symbols: get_dsa_index_topk, dsa_layer_skips_topk, get_dsa_index_n_heads
  - `python/sglang/srt/model_executor/model_runner.py` modified +14/-0 (14 lines); hunks: -61,7 +61,9; -832,6 +834,17 @@ def initialize(self):; symbols: initialize, get_pp_proxy_topk_size, alloc_memory_pool, _dummy_run
  - `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` modified +6/-0 (6 lines); hunks: -180,6 +180,7 @@ def _allocate_decode_buffers(; -218,6 +219,10 @@ def _allocate_decode_buffers(; symbols: _allocate_decode_buffers, __init__
  - `python/sglang/srt/model_executor/runner_utils/buffers.py` modified +5/-0 (5 lines); hunks: -107,6 +107,7 @@ def create(; -149,6 +150,10 @@ def create(; symbols: create
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -40,6 +40,7 @@
+    dsa_layer_skips_topk,
@@ -1603,40 +1604,8 @@ def __init__(
-                self.index_topk_freq = getattr(config, "index_topk_freq", 1)
-                self.index_topk_pattern = getattr(config, "index_topk_pattern", None)
-                self.index_skip_topk_offset = getattr(
-                    config, "index_skip_topk_offset", None
diff -- python/sglang/srt/configs/model_config.py
@@ -132,6 +132,29 @@ def get_dsa_index_topk(config: PretrainedConfig) -> int:
+def dsa_layer_skips_topk(config: PretrainedConfig, layer_id: int) -> bool:
+    """Return whether a DSA layer reuses the previous layer's top-k indices."""
+    assert is_deepseek_dsa(config)
+    pattern = getattr(config, "index_topk_pattern", None)
+    if pattern is not None:
+        return layer_id < len(pattern) and pattern[layer_id] == "S"
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -61,7 +61,9 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +43/-42; `python/sglang/srt/configs/model_config.py` modified +23/-0; `python/sglang/srt/model_executor/model_runner.py` modified +14/-0; `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` modified +6/-0; `python/sglang/srt/model_executor/runner_utils/buffers.py` modified +5/-0; `python/sglang/srt/managers/scheduler_pp_mixin.py` modified +7/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/managers/scheduler_pp_mixin.py`, `python/sglang/srt/model_executor/model_runner.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #28785 - Pass DSA topk through PP warmup proxy buffers

- Link: https://github.com/sgl-project/sglang/pull/28785
- Status/date: merged / 2026-06-21
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +21/-0, 77 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Pass DSA topk through PP warmup proxy buffers"; model line: DeepSeek V3.1; category: model implementation change; main diff: `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/model_executor/runner/base_runner.py`, `python/sglang/srt/model_executor/runner/eager_runner.py`; technical summary: Covers "Pass DSA topk through PP warmup proxy buffers"; the main implementation surface is `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/model_executor/runner/base_runner.py`, `python/sglang/srt/model_executor/runner/eager_runner.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +10/-0 (10 lines); hunks: -2387,6 +2387,13 @@ def __init__(; -2396,6 +2403,7 @@ def forward(; symbols: __init__, get_input_embeddings, _dsa_forward_uses_topk, forward, touching `__init__, get_input_embeddings, _dsa_forward_uses_topk`; `python/sglang/srt/model_executor/runner/base_runner.py` modified +6/-0 (6 lines); hunks: -78,6 +78,7 @@ def _allocate_decode_buffers(; -115,6 +116,10 @@ def _allocate_decode_buffers(; symbols: _allocate_decode_buffers, _alloc_dummy_decode_buffers, _dummy_run, touching `_allocate_decode_buffers, _alloc_dummy_decode_buffers, _dummy_run`; `python/sglang/srt/model_executor/runner/eager_runner.py` modified +5/-0 (5 lines); hunks: -190,6 +190,11 @@ def _slot(name):; symbols: _slot, touching `_slot`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +10/-0 (10 lines); hunks: -2387,6 +2387,13 @@ def __init__(; -2396,6 +2403,7 @@ def forward(; symbols: __init__, get_input_embeddings, _dsa_forward_uses_topk, forward
  - `python/sglang/srt/model_executor/runner/base_runner.py` modified +6/-0 (6 lines); hunks: -78,6 +78,7 @@ def _allocate_decode_buffers(; -115,6 +116,10 @@ def _allocate_decode_buffers(; symbols: _allocate_decode_buffers, _alloc_dummy_decode_buffers, _dummy_run
  - `python/sglang/srt/model_executor/runner/eager_runner.py` modified +5/-0 (5 lines); hunks: -190,6 +190,11 @@ def _slot(name):; symbols: _slot
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -2387,6 +2387,13 @@ def __init__(
+    def _dsa_forward_uses_topk(self) -> bool:
+        if not self.use_dsa:
+            return False
+        backend = get_attn_backend()
+        backend = getattr(backend, "primary", backend)
+        return not getattr(backend, "use_mha", False)
diff -- python/sglang/srt/model_executor/runner/base_runner.py
@@ -78,6 +78,7 @@ def _allocate_decode_buffers(
+    pp_proxy_topk_size: Optional[int] = None,
@@ -115,6 +116,10 @@ def _allocate_decode_buffers(
+            if pp_proxy_topk_size is not None:
+                pp_proxy_tensors["topk_indices"] = torch.zeros(
+                    (max_num_token, pp_proxy_topk_size), dtype=torch.int32
+                )
diff -- python/sglang/srt/model_executor/runner/eager_runner.py
@@ -190,6 +190,11 @@ def _slot(name):
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +10/-0; `python/sglang/srt/model_executor/runner/base_runner.py` modified +6/-0; `python/sglang/srt/model_executor/runner/eager_runner.py` modified +5/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/model_executor/runner/base_runner.py`, `python/sglang/srt/model_executor/runner/eager_runner.py`, `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #28938 - [AMD] Improve performance of dsv4 in high concurrency

- Link: https://github.com/sgl-project/sglang/pull/28938
- Status/date: merged / 2026-06-23
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +111/-44, 347 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Improve performance of dsv4 in high concurrency"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/deepseek_v4_rope.py`, `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v2.py`; technical summary: Covers "[AMD] Improve performance of dsv4 in high concurrency"; the main implementation surface is `python/sglang/srt/layers/deepseek_v4_rope.py`, `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/deepseek_v4_rope.py` modified +43/-32 (75 lines); hunks: -161,7 +161,7 @@ def apply_rotary_emb_triton_kernel_batched(; -210,66 +210,67 @@ def apply_rotary_emb_triton_kernel_batched(; symbols: apply_rotary_emb_triton_kernel_batched, apply_rotary_emb_contig_kernel, apply_rotary_emb_flat_kernel, apply_rotary_emb_triton, touching `apply_rotary_emb_triton_kernel_batched, apply_rotary_emb_contig_kernel, apply_rotary_emb_flat_kernel`; `python/sglang/srt/models/deepseek_v4.py` modified +29/-0 (29 lines); hunks: -157,6 +157,10 @@ def _is_fused_mhc_post_pre_enabled() -> bool:; -1580,6 +1584,22 @@ def forward(; symbols: _is_fused_mhc_post_pre_enabled, forward, touching `_is_fused_mhc_post_pre_enabled, forward`; `python/sglang/srt/models/deepseek_v2.py` modified +14/-2 (16 lines); hunks: -830,6 +830,7 @@ def forward(; -870,6 +871,7 @@ def forward(; symbols: forward, forward_normal, touching `forward, forward_normal`; `python/sglang/srt/layers/dp_attention.py` modified +5/-7 (12 lines); hunks: -580,13 +580,11 @@ def _dp_gather_via_all_gatherv(; symbols: _dp_gather_via_all_gatherv, _dp_gather, touching `_dp_gather_via_all_gatherv, _dp_gather`.
- Code diff details:
  - `python/sglang/srt/layers/deepseek_v4_rope.py` modified +43/-32 (75 lines); hunks: -161,7 +161,7 @@ def apply_rotary_emb_triton_kernel_batched(; -210,66 +210,67 @@ def apply_rotary_emb_triton_kernel_batched(; symbols: apply_rotary_emb_triton_kernel_batched, apply_rotary_emb_contig_kernel, apply_rotary_emb_flat_kernel, apply_rotary_emb_triton
  - `python/sglang/srt/models/deepseek_v4.py` modified +29/-0 (29 lines); hunks: -157,6 +157,10 @@ def _is_fused_mhc_post_pre_enabled() -> bool:; -1580,6 +1584,22 @@ def forward(; symbols: _is_fused_mhc_post_pre_enabled, forward
  - `python/sglang/srt/models/deepseek_v2.py` modified +14/-2 (16 lines); hunks: -830,6 +830,7 @@ def forward(; -870,6 +871,7 @@ def forward(; symbols: forward, forward_normal
  - `python/sglang/srt/layers/dp_attention.py` modified +5/-7 (12 lines); hunks: -580,13 +580,11 @@ def _dp_gather_via_all_gatherv(; symbols: _dp_gather_via_all_gatherv, _dp_gather
  - `python/sglang/srt/distributed/parallel_state.py` modified +20/-3 (23 lines); hunks: -1013,10 +1013,14 @@ def all_gatherv(; -1027,7 +1031,9 @@ def all_gatherv(; symbols: all_gatherv, _all_gather_allocate_output
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/deepseek_v4_rope.py
@@ -161,7 +161,7 @@ def apply_rotary_emb_triton_kernel_batched(
-    # Batched variant: BLOCK_M tokens per program (mirrors ATOM's inverse_rope_gptj
+    # Batched variant: BLOCK_M tokens per program
@@ -210,66 +210,67 @@ def apply_rotary_emb_triton_kernel_batched(
-def apply_rotary_emb_contig_kernel(
+def apply_rotary_emb_flat_kernel(
-    rope_dim,
diff -- python/sglang/srt/models/deepseek_v4.py
@@ -157,6 +157,10 @@ def _is_fused_mhc_post_pre_enabled() -> bool:
+# PoC: compute the (replicated TP1) shared expert on LOCAL hidden before the dp
+# gather instead of on the gathered global buffer. Requires
+# SGLANG_SHARED_EXPERT_TP1=1 (replicated shared expert). Default OFF.
+_SHARED_EXPERT_LOCAL = get_bool_env_var("SGLANG_DP_SHARED_EXPERT_LOCAL")
@@ -1580,6 +1584,22 @@ def forward(
+        # PoC (SGLANG_DP_SHARED_EXPERT_LOCAL): compute the replicated shared expert
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -830,6 +830,7 @@ def forward(
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/deepseek_v4_rope.py` modified +43/-32; `python/sglang/srt/models/deepseek_v4.py` modified +29/-0; `python/sglang/srt/models/deepseek_v2.py` modified +14/-2; `python/sglang/srt/layers/dp_attention.py` modified +5/-7; `python/sglang/srt/distributed/parallel_state.py` modified +20/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/distributed/parallel_state.py`, `python/sglang/srt/layers/deepseek_v4_rope.py`, `python/sglang/srt/layers/dp_attention.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27833 - [AMD] Enable BCG on ROCm + route aiter prefill via MHA during PCG/BCG capture for Kimi-2.5

- Link: https://github.com/sgl-project/sglang/pull/27833
- Status/date: merged / 2026-06-24
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +187/-0, 202 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Enable BCG on ROCm + route aiter prefill via MHA during PCG/BCG capture for Kimi-2.5"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_common/attention_backend_handler.py`, `test/registered/amd/test_kimi_k25_mxfp4_bcg_mi35x.py`; technical summary: Covers "[AMD] Enable BCG on ROCm + route aiter prefill via MHA during PCG/BCG capture for Kimi-2.5"; the main implementation surface is `python/sglang/srt/models/deepseek_common/attention_backend_handler.py`, `test/registered/amd/test_kimi_k25_mxfp4_bcg_mi35x.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_common/attention_backend_handler.py` modified +8/-0 (8 lines); hunks: -1,6 +1,9; -150,6 +153,11 @@ def handle_attention_tokenspeed_mla(attn, forward_batch):; symbols: handle_attention_tokenspeed_mla, handle_attention_aiter, touching `handle_attention_tokenspeed_mla, handle_attention_aiter`; `test/registered/amd/test_kimi_k25_mxfp4_bcg_mi35x.py` added +179/-0 (179 lines); hunks: -0,0 +1,179; symbols: CaptureConfig, get_capture_configs, TestKimiK25MXFP4BcgMI35x, setUpClass, touching `CaptureConfig, get_capture_configs, TestKimiK25MXFP4BcgMI35x`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_common/attention_backend_handler.py` modified +8/-0 (8 lines); hunks: -1,6 +1,9; -150,6 +153,11 @@ def handle_attention_tokenspeed_mla(attn, forward_batch):; symbols: handle_attention_tokenspeed_mla, handle_attention_aiter
  - `test/registered/amd/test_kimi_k25_mxfp4_bcg_mi35x.py` added +179/-0 (179 lines); hunks: -0,0 +1,179; symbols: CaptureConfig, get_capture_configs, TestKimiK25MXFP4BcgMI35x, setUpClass
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_common/attention_backend_handler.py
@@ -1,6 +1,9 @@
+from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
+    is_in_breakable_cuda_graph,
+)
@@ -150,6 +153,11 @@ def handle_attention_tokenspeed_mla(attn, forward_batch):
+    # During PCG/BCG capture on ROCm, aiter fp8 MLA prefill has no capture
+    # kernels; route through the MHA path (radix_attention swaps attn_mqa for
diff -- test/registered/amd/test_kimi_k25_mxfp4_bcg_mi35x.py
@@ -0,0 +1,179 @@
+"""Kimi-K2.5-MXFP4 aiter breakable CUDA-graph (BCG) capture accuracy test
+(MI35x, PR-CI)
+Exercises the AMD breakable (BCG) CUDA-graph prefill capture path on a
+deepseek-family (Kimi-K2.5) aiter model so the code added in this PR actually
+runs in PR CI:
+  * runner_backend/breakable_cuda_graph_backend.py
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_common/attention_backend_handler.py` modified +8/-0
  - tests: `test/registered/amd/test_kimi_k25_mxfp4_bcg_mi35x.py` added +179/-0
- Risk and verification: The diff ships test coverage in `test/registered/amd/test_kimi_k25_mxfp4_bcg_mi35x.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #27053 - [BCG][GLM5] perf: BCG support and prefill enhancements

- Link: https://github.com/sgl-project/sglang/pull/27053
- Status/date: merged / 2026-06-24
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +694/-224, 1292 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BCG][GLM5] perf: BCG support and prefill enhancements"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/layers/attention/dsa/dsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`; technical summary: Covers "[BCG][GLM5] perf: BCG support and prefill enhancements"; the main implementation surface is `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/layers/attention/dsa/dsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +287/-104 (391 lines); hunks: -1,18 +1,24; -28,6 +34,12; symbols: MlaBmmFusionPlan, init_mla_forward, _can_fuse_bmm_into_attention, _split_q_nope_pe, touching `MlaBmmFusionPlan, init_mla_forward, _can_fuse_bmm_into_attention`; `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` modified +216/-101 (317 lines); hunks: -12,16 +12,24; -39,6 +47,7; symbols: _is_in_piecewise_or_breakable_cuda_graph, _uses_dsa_attention_backend, k_cache_and_topk_result, _logits_head_gate_pcg_fake_impl, touching `_is_in_piecewise_or_breakable_cuda_graph, _uses_dsa_attention_backend, k_cache_and_topk_result`; `python/sglang/srt/models/deepseek_v2.py` modified +89/-16 (105 lines); hunks: -94,7 +94,7; -135,6 +135,13; symbols: DeepseekV2MLP, __init__, get_moe_weights, _can_dual_stream_graph, touching `DeepseekV2MLP, __init__, get_moe_weights`; `python/sglang/srt/layers/attention/dsa/utils.py` modified +20/-1 (21 lines); hunks: -9,9 +9,15; -80,6 +86,19 @@ def is_dsa_prefill_cp_round_robin_split():; symbols: is_dsa_prefill_cp_round_robin_split, is_graph_dsa_split_op_surface, can_dsa_prefill_cp_round_robin_split, touching `is_dsa_prefill_cp_round_robin_split, is_graph_dsa_split_op_surface, can_dsa_prefill_cp_round_robin_split`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +287/-104 (391 lines); hunks: -1,18 +1,24; -28,6 +34,12; symbols: MlaBmmFusionPlan, init_mla_forward, _can_fuse_bmm_into_attention, _split_q_nope_pe
  - `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` modified +216/-101 (317 lines); hunks: -12,16 +12,24; -39,6 +47,7; symbols: _is_in_piecewise_or_breakable_cuda_graph, _uses_dsa_attention_backend, k_cache_and_topk_result, _logits_head_gate_pcg_fake_impl
  - `python/sglang/srt/models/deepseek_v2.py` modified +89/-16 (105 lines); hunks: -94,7 +94,7; -135,6 +135,13; symbols: DeepseekV2MLP, __init__, get_moe_weights, _can_dual_stream_graph
  - `python/sglang/srt/layers/attention/dsa/utils.py` modified +20/-1 (21 lines); hunks: -9,9 +9,15; -80,6 +86,19 @@ def is_dsa_prefill_cp_round_robin_split():; symbols: is_dsa_prefill_cp_round_robin_split, is_graph_dsa_split_op_surface, can_dsa_prefill_cp_round_robin_split
  - `python/sglang/srt/layers/attention/dsa_backend.py` modified +6/-2 (8 lines); hunks: -2432,14 +2432,18 @@ def set_dsa_prefill_impl(self, forward_batch: Optional[F...; symbols: set_dsa_prefill_impl
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py
@@ -1,18 +1,24 @@
+from dataclasses import dataclass
+from sglang.srt.compilation.compilation_config import register_split_op
-from sglang.srt.layers.attention.dsa.utils import dsa_use_prefill_cp
+from sglang.srt.layers.attention.dsa.utils import (
+    dsa_use_prefill_cp,
+    is_graph_dsa_split_op_surface,
diff -- python/sglang/srt/layers/attention/dsa/dsa_indexer.py
@@ -12,16 +12,24 @@
+from sglang.srt.compilation.compilation_config import register_split_op
+    is_graph_dsa_split_op_surface,
+from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
+    eager_on_graph,
+)
+from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -94,7 +94,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +287/-104; `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` modified +216/-101; `python/sglang/srt/models/deepseek_v2.py` modified +89/-16; `python/sglang/srt/layers/attention/dsa/utils.py` modified +20/-1; `python/sglang/srt/layers/attention/dsa_backend.py` modified +6/-2; `python/sglang/srt/environ.py` modified +1/-0
  - tests: `test/registered/cuda_graph/piecewise/test_pcg_glm5_fp8_tp8.py` added +75/-0
- Risk and verification: The diff ships test coverage in `test/registered/cuda_graph/piecewise/test_pcg_glm5_fp8_tp8.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #29042 - [NPU] Fix the DeepSeek-V2-Coder model accuracy issue

- Link: https://github.com/sgl-project/sglang/pull/29042
- Status/date: merged / 2026-06-25
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +4/-1, 26 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NPU] Fix the DeepSeek-V2-Coder model accuracy issue"; model line: DeepSeek V3.1; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/llada2.py`, `python/sglang/srt/hardware_backend/npu/moe/topk.py`; technical summary: Covers "[NPU] Fix the DeepSeek-V2-Coder model accuracy issue"; the main implementation surface is `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/llada2.py`, `python/sglang/srt/hardware_backend/npu/moe/topk.py`. File-level evidence, code excerpts, and validation risks are preserved below.
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

### PR #14194 - [feature] implement dcp for deepseek_v2

- Link: https://github.com/sgl-project/sglang/pull/14194
- Status/date: merged / 2026-06-25
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 20 files, +1770/-30, 2258 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[feature] implement dcp for deepseek_v2"; model line: DeepSeek V3.1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/utils/dcp_utils.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/layers/attention/flashinfer_mla_backend.py`; technical summary: Covers "[feature] implement dcp for deepseek_v2"; the main implementation surface is `python/sglang/srt/layers/utils/dcp_utils.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/layers/attention/flashinfer_mla_backend.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/utils/dcp_utils.py` added +724/-0 (724 lines); hunks: -0,0 +1,724; symbols: dcp_enabled, get_attention_dcp_group, get_attention_dcp_world_size, get_attention_dcp_rank, touching `dcp_enabled, get_attention_dcp_group, get_attention_dcp_world_size`; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +62/-0 (62 lines); hunks: -1,5 +1,6; -20,6 +21,14; symbols: forward_absorb_prepare, forward_absorb_core, touching `forward_absorb_prepare, forward_absorb_core`; `python/sglang/srt/layers/attention/flashinfer_mla_backend.py` modified +44/-3 (47 lines); hunks: -23,6 +23,13; -410,6 +417,7 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; symbols: init_forward_metadata, _apply_cuda_graph_metadata, forward_extend, forward_decode, touching `init_forward_metadata, _apply_cuda_graph_metadata, forward_extend`; `python/sglang/srt/models/deepseek_v2.py` modified +45/-0 (45 lines); hunks: -122,6 +122,11; -1723,6 +1728,18 @@ def __init__(; symbols: __init__, set_dflash_layers_to_capture, prepare_context_parallel_metadata_for_dcp, DeepseekV3ForCausalLM, touching `__init__, set_dflash_layers_to_capture, prepare_context_parallel_metadata_for_dcp`.
- Code diff details:
  - `python/sglang/srt/layers/utils/dcp_utils.py` added +724/-0 (724 lines); hunks: -0,0 +1,724; symbols: dcp_enabled, get_attention_dcp_group, get_attention_dcp_world_size, get_attention_dcp_rank
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +62/-0 (62 lines); hunks: -1,5 +1,6; -20,6 +21,14; symbols: forward_absorb_prepare, forward_absorb_core
  - `python/sglang/srt/layers/attention/flashinfer_mla_backend.py` modified +44/-3 (47 lines); hunks: -23,6 +23,13; -410,6 +417,7 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; symbols: init_forward_metadata, _apply_cuda_graph_metadata, forward_extend, forward_decode
  - `python/sglang/srt/models/deepseek_v2.py` modified +45/-0 (45 lines); hunks: -122,6 +122,11; -1723,6 +1728,18 @@ def __init__(; symbols: __init__, set_dflash_layers_to_capture, prepare_context_parallel_metadata_for_dcp, DeepseekV3ForCausalLM
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` modified +31/-5 (36 lines); hunks: -9,6 +9,12; -271,11 +277,24 @@ def forward_normal_prepare(; symbols: forward_normal_prepare, _chunked_prefix_attn_mha, _get_mla_kv_buffer
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/utils/dcp_utils.py
@@ -0,0 +1,724 @@
+from dataclasses import dataclass
+from typing import Optional
+import torch
+import triton
+import triton.language as tl
+from sglang.srt.distributed.device_communicators.pynccl_allocator import (
diff -- python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py
@@ -1,5 +1,6 @@
+import logging
@@ -20,6 +21,14 @@
+from sglang.srt.layers.utils.dcp_utils import (
+    all_gather_kv_cache_for_mla_extend,
+    all_gather_q_for_mla_decode,
+    cp_lse_ag_out_rs,
diff -- python/sglang/srt/layers/attention/flashinfer_mla_backend.py
@@ -23,6 +23,13 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/utils/dcp_utils.py` added +724/-0; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +62/-0; `python/sglang/srt/layers/attention/flashinfer_mla_backend.py` modified +44/-3; `python/sglang/srt/models/deepseek_v2.py` modified +45/-0; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` modified +31/-5; `python/sglang/srt/model_executor/runner/eager_runner.py` modified +26/-1
- Risk and verification: The diff ships test coverage in `test/registered/dcp/test_dsv31_dcp8_gsm8k.py`, `test/registered/dcp/test_reduce_scatter_along_dim.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #29142 - [DeepSeek V3] Run routed experts on main stream in dual-stream MoE

- Link: https://github.com/sgl-project/sglang/pull/29142
- Status/date: merged / 2026-06-26
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +48/-46, 108 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DeepSeek V3] Run routed experts on main stream in dual-stream MoE"; model line: DeepSeek V3.1; category: model implementation change; main diff: `python/sglang/srt/models/deepseek_v2.py`; technical summary: Covers "[DeepSeek V3] Run routed experts on main stream in dual-stream MoE"; the main implementation surface is `python/sglang/srt/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +48/-46 (94 lines); hunks: -936,59 +936,61 @@ def forward_normal_dual_stream(; symbols: forward_normal_dual_stream, touching `forward_normal_dual_stream`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +48/-46 (94 lines); hunks: -936,59 +936,61 @@ def forward_normal_dual_stream(; symbols: forward_normal_dual_stream
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -936,59 +936,61 @@ def forward_normal_dual_stream(
-        shared_output = self._forward_shared_experts(
-            hidden_states, gemm_output_zero_allocator
-        )
+        # router_logits: (num_tokens, n_experts)
+        router_logits = self.gate(hidden_states, gemm_output_zero_allocator)
+        if use_flashinfer_trtllm_bypass:
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +48/-46
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
